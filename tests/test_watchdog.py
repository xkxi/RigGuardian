"""Watchdog rules on fake telemetry with a fake clock (no processes involved)."""

import json

import pytest

from conftest import outbox, queue

T_SLOW = 87  # FakeSource's slowdown temperature


@pytest.fixture
def rig(make_daemon, clock):
    d = make_daemon(clock=clock)
    jid = queue(d)
    d.db.update_job(jid, state="running", gpus="1", launched_at=clock())
    return d, jid


def feed(d, clock, gpu, temps, **extra):
    for t in temps:
        d.source.set(gpu, temp_c=t, **extra)
        clock.advance(2)
        d.collector.collect_once()
        d.watchdog.tick()


def control(d, jid):
    return json.loads((d.runner.run_dir(d.db.get_job(jid)) / "control.json").read_text())


def test_single_spike_does_nothing(rig, clock):
    d, jid = rig
    feed(d, clock, 1, [60, 60, 85, 60, 60])
    assert d.db.get_job(jid)["state"] == "running"
    assert not d.db.gpu_state(1)["cooling"]


def test_warn_tier_alerts_once_per_hour(rig, clock):
    d, jid = rig
    feed(d, clock, 1, [78] * 6)
    warns = [t for t in outbox(d) if "at 78 °C" in t]
    assert len(warns) == 1 and "slowdown at 87" in warns[0]
    assert d.db.get_job(jid)["state"] == "running"


def test_pause_tier_soft_pauses_job_and_cools_gpu(rig, clock):
    d, jid = rig
    feed(d, clock, 1, [84, 84, 84])
    job = d.db.get_job(jid)
    assert job["state"] == "pausing" and job["pause_reason"] == "thermal"
    assert control(d, jid)["action"] == "pause"
    assert d.db.gpu_state(1)["cooling"] == 1
    assert 1 not in d.runner.free_gpus()
    assert "Will resume when below 72 °C" in job["pause_detail"]


def test_sustained_thermal_throttle_pauses_below_the_line(rig, clock):
    d, jid = rig
    feed(d, clock, 1, [70] * 32, throttle=0x40)  # 64 s of HW thermal slowdown at 70 °C
    assert d.db.get_job(jid)["state"] == "pausing"


def test_emergency_stops_and_quarantines(rig, clock):
    d, jid = rig
    feed(d, clock, 1, [87, 88, 88])
    job = d.db.get_job(jid)
    assert job["pending_action"] == "stop" and job["kill_reason"] == "emergency"
    assert job["kill_deadline"] == pytest.approx(clock() + 60)
    assert control(d, jid)["action"] == "stop"
    st = d.db.gpu_state(1)
    assert st["quarantined"] == 1
    assert any(t.startswith("🚨") and "/clear 1" in t for t in outbox(d))


def test_gpu_lost_is_an_emergency(rig, clock):
    d, jid = rig
    d.source.set(1, lost=True)
    clock.advance(2)
    d.collector.collect_once()
    d.watchdog.tick()
    assert d.db.gpu_state(1)["quarantined"] == 1
    assert d.db.get_job(jid)["pending_action"] == "stop"


def test_resume_after_hold_up_to_three_times_a_day(rig, clock):
    d, jid = rig
    for i in range(4):
        feed(d, clock, 1, [84, 84, 84])
        d.db.update_job(jid, state="paused", pending_action=None)  # the SDK acknowledged
        feed(d, clock, 1, [70] * 160)  # 320 s below T_slow - 15
        job = d.db.get_job(jid)
        if i < 3:
            assert job["state"] == "running" and control(d, jid)["action"] == "resume", i
            assert job["auto_resumes_today"] == i + 1
            d.db.update_job(jid, pending_action=None)
        else:
            assert job["state"] == "paused" and job["waiting_for_user"] == 1
    assert any("stays paused until you /resume" in t for t in outbox(d))


def test_resume_needs_the_full_hold(rig, clock):
    d, jid = rig
    feed(d, clock, 1, [84, 84, 84])
    d.db.update_job(jid, state="paused", pending_action=None)
    feed(d, clock, 1, [70] * 100 + [74] + [70] * 60)  # one reading above 72 resets the timer
    assert d.db.get_job(jid)["state"] == "paused"
    feed(d, clock, 1, [70] * 100)
    assert d.db.get_job(jid)["state"] == "running"


def test_stall_signals_then_kills(rig, clock):
    d, jid = rig
    rd = d.runner.run_dir(d.db.get_job(jid))
    (rd / "heartbeat").write_text(json.dumps({"ts": clock(), "step": 40, "sdk": 1}))
    clock.advance(800)
    d.watchdog.tick()
    assert d.db.get_job(jid)["stall_flagged_at"] is None
    clock.advance(200)
    d.watchdog.tick()
    assert d.db.get_job(jid)["stall_flagged_at"] == clock()
    assert any("no progress for" in t for t in outbox(d))
    clock.advance(301)
    d.watchdog.tick()
    assert d.db.get_job(jid)["kill_reason"] == "stall"


def test_stall_recovers_when_heartbeat_moves(rig, clock):
    d, jid = rig
    rd = d.runner.run_dir(d.db.get_job(jid))
    (rd / "heartbeat").write_text(json.dumps({"ts": clock(), "step": 40, "sdk": 1}))
    clock.advance(1000)
    d.watchdog.tick()
    (rd / "heartbeat").write_text(json.dumps({"ts": clock() + 1, "step": 41, "sdk": 1}))
    clock.advance(5)
    d.watchdog.tick()
    job = d.db.get_job(jid)
    assert job["stall_flagged_at"] is None and job["kill_reason"] is None


def test_ecc_rise_quarantines(rig, clock):
    d, jid = rig
    feed(d, clock, 1, [50], ecc_uncorrected=0)
    feed(d, clock, 1, [50], ecc_uncorrected=2)
    assert d.db.gpu_state(1)["quarantined"] == 1
    assert d.db.get_job(jid)["kill_reason"] == "ecc"


def test_idle_gpu_warns_but_does_not_act(rig, clock):
    d, jid = rig
    d.db.update_job(jid, launched_at=clock() - 1000)
    feed(d, clock, 1, [50] * 305, util=0)
    assert any("near 0% busy" in t for t in outbox(d))
    assert d.db.get_job(jid)["state"] == "running"


def test_stray_process_reported_never_killed(rig, clock):
    d, jid = rig
    feed(d, clock, 0, [50], procs=[(999999, 3 * 2**30)])
    assert any("process 999999" in t and "Left alone" in t for t in outbox(d))


def test_agent_card_heat_only_warns(rig, clock):
    d, jid = rig
    feed(d, clock, 2, [86, 86, 86])
    assert d.db.get_job(jid)["state"] == "running"
    assert any("language-model card" in t for t in outbox(d))


def test_disk_low_holds_queue(rig, clock, monkeypatch):
    d, jid = rig
    import guardian.watchdog as wd

    monkeypatch.setattr(wd, "disk_free_bytes", lambda p: 1 * 2**30)
    d.cfg.checkpoints.min_free_disk_gb = 20
    d.watchdog.check_disk()
    assert d.db.kv_get("disk_hold") == "1"
    monkeypatch.setattr(wd, "disk_free_bytes", lambda p: 100 * 2**30)
    d.watchdog.check_disk()
    assert d.db.kv_get("disk_hold") is None


"""Watchdog findings on fake telemetry + fake clock (no processes). Each test asserts correct behaviour."""
import threading
import time

import pytest

from conftest import events, feed, outbox, queue, running_job


# ---- B1: one unreadable template file disables thermal pause AND emergency stop ----------
@pytest.mark.parametrize("bad", ["gpus: zero\nscript: x.py\n", "name: [unclosed\n", "scirpt: train.py\n"])
def test_b1_bad_template_file_must_not_disable_emergency_stop(make_daemon, clock, bad):
    d = make_daemon(clock=clock)
    jid = running_job(d, clock)
    (d.cfg.templates_dir / "wip.yaml").write_text(bad)  # user is editing a new, unrelated template
    feed(d, clock, 1, [88, 88, 88, 88, 88])  # at/above the 87 °C slowdown temperature
    job = d.db.get_job(jid)
    assert d.db.gpu_state(1)["quarantined"] == 1
    assert job["pending_action"] == "stop" and job["kill_deadline"], (
        f"job on an 88 °C card was not stopped: state={job['state']} pending={job['pending_action']} "
        f"kill_deadline={job['kill_deadline']} outbox={outbox(d)}")
    assert any(t.startswith("🚨") for t in outbox(d)), "no emergency alert was sent"


def test_b1_bad_template_file_must_not_disable_thermal_pause(make_daemon, clock):
    d = make_daemon(clock=clock)
    jid = running_job(d, clock)
    (d.cfg.templates_dir / "wip.yaml").write_text("gpus: zero\nscript: x.py\n")
    feed(d, clock, 1, [84] * 10)  # pause tier
    job = d.db.get_job(jid)
    assert job["state"] == "pausing", f"job at 84 °C (pause tier) still {job['state']}"


# ---- B2: telemetry that stopped updating is trusted forever -------------------------------
def test_b2_stale_telemetry_must_raise_an_alarm(make_daemon, clock):
    d = make_daemon(clock=clock)
    jid = running_job(d, clock)
    feed(d, clock, 1, [60, 60, 60])
    # the collector thread now fails on every read (NVML hang / unexpected exception): no new samples
    d.source.set(1, temp_c=95)  # the card is really at 95 °C, but nobody reads it any more
    for _ in range(900):  # 30 minutes of watchdog ticks on the same last sample
        clock.advance(2)
        d.watchdog.tick()
    age = clock() - d.store.latest(1)["ts"]
    job = d.db.get_job(jid)
    alarms = [t for t in outbox(d) if "telemetry" in t.lower() or "stale" in t.lower() or t.startswith("🚨")]
    assert alarms or job["state"] != "running", (
        f"newest sample is {age:.0f} s old, job still '{job['state']}', outbox={outbox(d)}: "
        f"the watchdog keeps judging a {d.store.latest(1)['temp_c']} °C reading from 30 minutes ago")


# ---- B3: a card whose temperature cannot be read is treated as cool ----------------------
def test_b3_unreadable_temperature_must_not_be_silent(make_daemon, clock):
    d = make_daemon(clock=clock)
    jid = running_job(d, clock)
    feed(d, clock, 1, [None] * 300)  # 10 minutes with no temperature reading at all
    msgs = [t for t in outbox(d) if "busy" not in t]  # ignore the unrelated idle-GPU warning
    assert msgs, (f"10 min without a temperature on a busy card: no temperature-related message at all, job "
                  f"{d.db.get_job(jid)['state']}, tier={d.watchdog.smoothed_tier(1)} (0 = normal), outbox={outbox(d)}")


def test_b3_one_missing_reading_must_not_cancel_an_emergency(make_daemon, clock):
    d = make_daemon(clock=clock)
    jid = running_job(d, clock)
    feed(d, clock, 1, [95, None, 95, None, 95, None, 95, None, 95, None])  # 20 s at 95 °C, every other read fails
    assert d.db.gpu_state(1)["quarantined"] == 1, (
        f"card at 95 °C (slowdown 87) with intermittent read errors: quarantined=0, job "
        f"{d.db.get_job(jid)['state']}, outbox={outbox(d)}")


# ---- B5: thermal pause acknowledged after the card already cooled: never resumed ----------
def test_b5_thermal_pause_acked_after_cooldown_must_still_resume(make_daemon, clock):
    d = make_daemon(clock=clock)
    jid = running_job(d, clock)
    feed(d, clock, 1, [84, 84, 84])
    assert d.db.get_job(jid)["state"] == "pausing"
    feed(d, clock, 1, [70] * 160)  # card cools for 320 s while the job is still inside a long step / eval
    assert d.db.gpu_state(1)["cooling"] == 0
    d.db.update_job(jid, state="paused", pending_action=None)  # now the SDK acknowledges the pause
    feed(d, clock, 1, [60] * 1800)  # one more hour, cool
    job = d.db.get_job(jid)
    assert job["state"] == "running", (
        f"after 1 h at 60 °C the job is still '{job['state']}' (pause_reason={job['pause_reason']}, "
        f"waiting_for_user={job['waiting_for_user']}); it was promised: {job['pause_detail']}")


# ---- B9: ECC counter that resets (reboot / driver reload) keeps the old, higher baseline --
def test_b9_ecc_counter_reset_must_rebaseline(make_daemon, clock):
    d = make_daemon(clock=clock)
    running_job(d, clock)
    feed(d, clock, 1, [50], ecc_uncorrected=7)  # baseline 7 (volatile counter before the reboot)
    feed(d, clock, 1, [50], ecc_uncorrected=0)  # reboot: NVML's volatile counter starts again at 0
    feed(d, clock, 1, [50], ecc_uncorrected=3)  # three NEW uncorrectable errors
    assert d.db.gpu_state(1)["quarantined"] == 1, (
        f"3 new uncorrectable ECC errors ignored; baseline still {d.db.kv_get('ecc_baseline:1')}")


# ---- B10: a job on a card that the new config no longer lists as 'training' is unprotected
def test_b10_job_on_card_removed_from_training_keeps_protection(make_daemon, clock, tmp_path):
    d = make_daemon(clock=clock)
    jid = running_job(d, clock, gpus="1")
    # the agent is restarted with GPU 1 taken out of training (exclude / agent card) while the job runs
    from guardian.daemon import Daemon
    from guardian.runner import LocalLauncher
    from conftest import make_cfg
    cfg = make_cfg(tmp_path, gpus={"training": [0], "agent": 2})
    d2 = Daemon(cfg, launcher=LocalLauncher(), clock=clock)
    d2.collector.collect_once()
    feed(d2, clock, 1, [95] * 10)
    job = d2.db.get_job(jid)
    assert job["pending_action"] == "stop", (
        f"job {jid} runs on GPU 1 at 95 °C (slowdown 87) but is left '{job['state']}', "
        f"pending={job['pending_action']}; outbox={outbox(d2)}")


# ---- B4: the watchdog is blocked while the runner hashes a checkpoint / asks the LLM -----
def test_b4_watchdog_not_blocked_by_checkpoint_hashing(make_daemon, clock, monkeypatch):
    import guardian.checkpoints as ck
    d = make_daemon(clock=clock)
    jid = running_job(d, clock, gpus="0")
    rd = d.runner.run_dir(d.db.get_job(jid))
    (rd / "exit_code").write_text("1\n")
    store = ck.CheckpointStore(rd / "checkpoints")
    store.ensure()
    tmp = store.tmp_path(10)
    tmp.write_bytes(b"x" * 1000)
    store.commit(tmp, {"step": 10, "config_hash": d.db.get_job(jid)["config_hash"]})
    real = ck.sha256_file
    monkeypatch.setattr(ck, "sha256_file", lambda p, chunk=0: (time.sleep(3.0), real(p))[1])  # a 3 s hash ≈ 3-6 GB file
    t = threading.Thread(target=d.runner.tick)
    t.start()
    time.sleep(0.3)
    t0 = time.monotonic()
    d.watchdog.tick()
    blocked = time.monotonic() - t0
    t.join()
    assert blocked < 1.0, f"watchdog tick waited {blocked:.1f} s for the runner lock while a checkpoint was hashed"


# ---- B1 (second trigger): bookkeeping that fails (disk full, locked database) comes BEFORE the stop
def test_b1_database_error_must_not_prevent_the_emergency_stop(make_daemon, clock, monkeypatch):
    import sqlite3
    d = make_daemon(clock=clock)
    jid = running_job(d, clock)
    real_pid = __import__("os").getpid()
    rd = d.runner.run_dir(d.db.get_job(jid))
    sent = []
    monkeypatch.setattr(d.runner, "signal_ranks", lambda job, sig: sent.append(("ranks", sig)) or 0)
    monkeypatch.setattr(d.runner, "signal_group", lambda job, sig: sent.append(("group", sig)) or True)

    def full(*a, **k):
        raise sqlite3.OperationalError("database or disk is full")

    monkeypatch.setattr(d.db, "set_gpu_state", full)   # every write fails; reads still work
    monkeypatch.setattr(d.db, "event", full)
    monkeypatch.setattr(d.db, "update_job", full)
    feed(d, clock, 1, [95] * 10)
    assert sent, "card at 95 °C for 20 s: no signal of any kind was sent to the job because a DB write failed first"

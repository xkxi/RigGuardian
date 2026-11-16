"""Rig shapes and failure handling that are easy to get wrong: 0, 1 and 16 GPUs, cards that
stop reporting, a busy runner during a heat emergency, stop signals from outside, values and
template files that must be refused, and what the stall rule may and may not call a hang.
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

from conftest import feed, make_cfg, outbox, queue, running_job, run_until, state, write_toy_template
from guardian.checks import run_checks
from guardian.daemon import Daemon
from guardian.runner import ActionError, LocalLauncher
from guardian.templates import TemplateError, coerce_value, load_templates

REPO = Path(__file__).resolve().parent.parent
USER = 42
R4090 = {"name": "NVIDIA GeForce RTX 4090", "mem_gb": 24, "slowdown_c": 90}


def rig(tmp_path, specs, gpus=None, clock=time.time, **over):
    write_toy_template(tmp_path)
    cfg = make_cfg(tmp_path, gpus=gpus or {"training": "auto"}, telemetry={"fake_gpus": specs}, **over)
    d = Daemon(cfg, launcher=LocalLauncher(), clock=clock)
    d.collector.collect_once()
    return d


# ---- any number of GPUs --------------------------------------------------------------------
def test_no_gpus_at_all_starts_and_explains_itself(tmp_path):
    d = rig(tmp_path, [])
    d.startup()
    assert d.cfg.gpus.training == []
    assert "Training GPUs: none" in "".join(outbox(d))
    assert d.agent.handle_text(USER, "/status")[0].text.startswith("🖥 Status")
    d.watchdog.tick(), d.runner.tick()  # nothing to guard, nothing to crash on


def test_one_gpu_with_a_desktop_on_it_still_trains(tmp_path):
    d = rig(tmp_path, [R4090])
    d.source.set(0, mem_used=800 * 2**20)  # a display server and a browser, not a training run
    d.collector.collect_once()
    assert d.runner.free_gpus() == [0] and d.runner.busy_reason(0) is None
    d.source.set(0, mem_used=10 * 2**30)  # somebody else's training run
    d.collector.collect_once()
    assert d.runner.free_gpus() == []
    why = d.runner.busy_reason(0)
    assert "in use by other programs" in why
    assert "not available" in d.agent.handle_text(USER, "/status")[0].text  # the screen says why it waits


def test_sixteen_gpus_are_all_tracked_and_an_eight_gpu_job_gets_eight_distinct_cards(tmp_path, clock):
    d = rig(tmp_path, [R4090] * 16, clock=clock)
    write_toy_template(tmp_path, gpus=8)
    assert d.cfg.gpus.training == list(range(16)) and len(d.store.gpus()) == 16
    assert d.store.label(0) == "RTX 4090 #0" and d.store.label(15) == "RTX 4090 #15"  # the number is the index
    started = []
    d.runner._launch = lambda job, gpus: (started.append(list(gpus)),
                                          d.db.update_job(job["id"], state="running", gpus=",".join(map(str, gpus))))
    a, b, c = queue(d), queue(d), queue(d)
    d.runner.tick()
    assert [len(set(g)) for g in started] == [8, 8] and not set(started[0]) & set(started[1])
    assert state(d, c) == "queued"  # no cards left: it waits
    for g in range(16):  # every card is guarded, not just the first few
        feed(d, clock, g, [95, 95, 95])
    assert all(d.db.gpu_state(g)["quarantined"] for g in range(16))


def test_a_job_keeps_its_card_when_the_cards_are_renumbered(tmp_path):
    d = rig(tmp_path, [R4090, R4090, R4090])
    d.startup()
    jid = queue(d)
    d.db.update_job(jid, state="running", gpus="2", gpu_uuids=d.inventory[2]["uuid"], pid=os.getpid(),
                    pgid=None, proc_start=None)
    moved = [dict(d.inventory[2], gpu=0), dict(d.inventory[0], gpu=1)]  # a card was pulled; indexes shifted
    lines = d.runner.remap_gpus(moved)
    assert d.db.get_job(jid)["gpus"] == "0" and "GPU 0 now" in lines[0]


# ---- cards that stop reporting ---------------------------------------------------------------
def test_a_card_that_stops_reporting_is_alerted_then_its_job_is_paused_and_resumed(make_daemon, clock):
    d = make_daemon(clock=clock, telemetry={"stale_alert_s": 20, "stale_pause_s": 60},
                    thermal={"resume_hold_s": 10})
    jid = running_job(d, clock)
    feed(d, clock, 1, [60, 60, 60])
    feed(d, clock, 1, [None] * 15)  # 30 s without a temperature
    assert any("no temperature reading" in t and "Heat protection cannot see" in t for t in outbox(d))
    assert state(d, jid) == "running"
    feed(d, clock, 1, [None] * 20)  # past stale_pause_s
    assert state(d, jid) == "pausing" and d.db.gpu_state(1)["cooling"] == 1
    assert 1 not in d.runner.free_gpus()
    d.db.update_job(jid, state="paused", pending_action=None)  # the job acknowledges the pause
    feed(d, clock, 1, [None] * 30)
    assert state(d, jid) == "paused"  # still blind: stays paused
    feed(d, clock, 1, [60] * 10)  # readings are back and cool for the hold time
    assert state(d, jid) == "running" and d.db.gpu_state(1)["cooling"] == 0


def test_a_card_with_no_sensor_is_named_once_and_never_paused(tmp_path, clock):
    write_toy_template(tmp_path)
    cfg = make_cfg(tmp_path, gpus={"training": "auto"}, telemetry={"fake_gpus": [R4090]})
    d = Daemon(cfg, launcher=LocalLauncher(), clock=clock)
    d.source.set(0, temp_c=None, t_slow=None)  # from the first reading on: this card has no sensor
    d.startup()
    assert sum("No temperature reading" in t for t in outbox(d)) == 1
    jid = running_job(d, clock, gpus="0")
    feed(d, clock, 0, [None] * 400)  # 13 minutes
    assert state(d, jid) == "running"
    assert any("no temperature" in m.lower() for _, m in run_checks(d.cfg, d.source.read()))  # doctor says it too


# ---- a heat emergency does not wait for anything ---------------------------------------------
def test_emergency_signal_goes_out_while_the_runner_is_busy(make_daemon, clock, monkeypatch):
    d = make_daemon(clock=clock)
    jid = running_job(d, clock)
    sent = []
    monkeypatch.setattr(d.runner, "signal_stop", lambda job: sent.append(job["id"]) or True)
    for _ in range(3):
        d.source.set(1, temp_c=96)
        clock.advance(2)
        d.collector.collect_once()
    release = threading.Event()
    holder = threading.Thread(target=lambda: (d.runner.lock.acquire(), release.wait(10), d.runner.lock.release()))
    holder.start()
    time.sleep(0.1)
    try:
        t0 = time.monotonic()
        d.watchdog.tick()  # the runner "is hashing a checkpoint": its lock is taken
        assert time.monotonic() - t0 < 2 and sent == [jid]
        assert 1 in d.runner.hot_gpus and 1 not in d.runner.free_gpus()  # nothing new starts there meanwhile
    finally:
        release.set()
        holder.join()
    d.watchdog.tick()  # the lock is free again: the records and the alert follow
    assert d.db.gpu_state(1)["quarantined"] == 1 and d.db.get_job(jid)["pending_action"] == "stop"
    assert any(t.startswith("🚨") for t in outbox(d))


# ---- stop signals from outside the agent -----------------------------------------------------
def test_a_job_stopped_by_someone_else_is_queued_again_but_not_for_ever(make_daemon):
    d = make_daemon()
    jid = queue(d)
    d.db.kv_set("hold", "1")  # nothing is launched in this test
    rd = d.runner.run_dir(d.db.get_job(jid))
    for n in range(1, 5):
        d.db.update_job(jid, state="running", gpus="0", pid=2_000_000_000, pgid=None, proc_start=1,
                        launched_at=time.time() - 5, pending_action=None)
        (rd / "exit_code").write_text("86\n")  # it checkpointed and left on a SIGTERM nobody here sent
        d.runner.tick()
        if n <= 3:
            assert state(d, jid) == "queued", n
            assert "signal the agent did not send" in outbox(d)[-1] and f"/cancel {jid}" in outbox(d)[-1]
        else:
            assert state(d, jid) == "stopped" and d.db.get_job(jid)["waiting_for_user"] == 1
    d.runner.request(jid, "resume")  # the user's own resume starts the count again
    assert d.db.get_job(jid)["interrupts"] == 0


def test_a_stop_the_user_asked_for_stays_a_stop(make_daemon):
    d = make_daemon()
    jid = queue(d)
    rd = d.runner.run_dir(d.db.get_job(jid))
    d.db.update_job(jid, state="pausing", gpus="0", pid=2_000_000_000, pgid=None, proc_start=1,
                    launched_at=time.time() - 5, pending_action="stop", kill_reason="stall")
    (rd / "exit_code").write_text("137\n")  # it hung, the stop deadline killed it
    d.runner.tick()
    assert state(d, jid) == "stopped" and "was killed" in outbox(d)[-1]


# ---- the stall rule ------------------------------------------------------------------------------
def write_hb(d, jid, clock, **fields):
    rd = d.runner.run_dir(d.db.get_job(jid))
    (rd / "heartbeat").write_text(json.dumps({"sdk": 2, "step": 10, "launch_steps": 10, **fields}))


def test_alive_but_between_epochs_is_not_a_stall_and_a_silent_process_is(make_daemon, clock):
    d = make_daemon(clock=clock, stall={"no_progress_s": 100, "quiet_phase_s": 1000, "kill_after_s": 50})
    jid = running_job(d, clock, gpus="0")
    t0 = clock()
    # evaluating: alive (ts fresh), no step for 500 s
    clock.advance(500)
    write_hb(d, jid, clock, ts=clock(), progress_ts=t0, phase="eval")
    d.watchdog.tick()
    assert d.db.get_job(jid)["stall_flagged_at"] is None
    # the same silence in the middle of an epoch is a stall
    write_hb(d, jid, clock, ts=clock(), progress_ts=t0, phase="train")
    d.watchdog.tick()
    assert d.db.get_job(jid)["stall_flagged_at"] is not None and "no progress" in outbox(d)[-1]
    # progress again: the flag is lifted
    clock.advance(10)
    write_hb(d, jid, clock, ts=clock(), progress_ts=clock(), phase="train")
    d.watchdog.tick()
    assert d.db.get_job(jid)["stall_flagged_at"] is None
    # between epochs, but the process itself has gone silent (its heartbeat thread stopped)
    silent_since = clock()
    clock.advance(200)
    write_hb(d, jid, clock, ts=silent_since, progress_ts=silent_since, phase="eval")
    d.watchdog.tick()
    assert d.db.get_job(jid)["stall_flagged_at"] is not None and "has not answered" in outbox(d)[-1]


# ---- values and files that must be refused -----------------------------------------------------
@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "1e999", True])
def test_overrides_that_are_not_real_numbers_are_refused(make_daemon, raw):
    d = make_daemon()
    with pytest.raises((ActionError, TemplateError, ValueError)):
        d.runner.prepare_run("toy-mlp", {"lr": raw})


def test_a_broken_template_file_is_skipped_and_reported_not_fatal(make_daemon, tmp_path):
    d = make_daemon()
    (tmp_path / "templates" / "wip.yaml").write_text("name: [unclosed\n")
    (tmp_path / "templates" / "evil.yaml").write_text(yaml.safe_dump({
        "name": "evil", "script": "x.py", "env": {"GUARDIAN_RUN_DIR": "/tmp/elsewhere"}}))
    problems = []
    tpls = load_templates(d.cfg.templates_dir, problems)
    assert "toy-mlp" in tpls and "evil" not in tpls and {n for n, _ in problems} == {"wip.yaml", "evil.yaml"}
    jid = queue(d)  # the good template still queues and the scheduler still runs
    d.runner.tick()
    assert state(d, jid) in ("preflight", "running")
    assert any("wip.yaml is being ignored" in t for t in outbox(d))
    job = d.db.get_job(jid)
    if job["pgid"]:
        os.killpg(job["pgid"], 9)


def test_bool_is_not_an_int_and_strings_are_bounded():
    from guardian.templates import Template

    t = Template.model_validate({"name": "t", "script": "x.py", "overridable": {
        "epochs": {"type": "int", "min": 1, "max": 10}, "tag": {"type": "str"}}})
    assert coerce_value(t, "epochs", "3") == 3 and coerce_value(t, "tag", "run-a") == "run-a"
    for key, raw in (("epochs", True), ("tag", "x" * 5000), ("tag", "a\x00b")):
        with pytest.raises((TemplateError, ValueError)):
            coerce_value(t, key, raw)


# ---- the command line --------------------------------------------------------------------------
def cli(home, *args, cwd=None):
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    env.pop("RIG_GUARDIAN_HOME", None)
    return subprocess.run([sys.executable, "-m", "guardian", "--home", str(home), *args], env=env, cwd=cwd,
                          capture_output=True, text=True, timeout=120)


def test_help_after_a_job_command_does_not_run_it_and_a_stray_config_is_not_a_home(tmp_path):
    home = tmp_path / "home"
    assert cli(home, "-y", "init", "--fake-gpus", "2").returncode == 0
    r = cli(home, "hold", "--help")
    assert r.returncode == 0 and "rig-guardian init" in r.stdout  # the help text, nothing else
    from guardian.db import Database

    assert Database(home / "state.db").kv_get("hold") is None  # --help did not hold the queue
    # a config.yaml in the current folder (some other project's) is not picked up as the rig folder
    other = tmp_path / "project"
    other.mkdir()
    (other / "config.yaml").write_text("model: resnet\n")
    env = {**os.environ, "PYTHONPATH": str(REPO), "HOME": str(tmp_path / "nohome")}
    env.pop("RIG_GUARDIAN_HOME", None)
    r = subprocess.run([sys.executable, "-m", "guardian", "status"], env=env, cwd=other, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode != 0 and "init" in (r.stdout + r.stderr)
    assert not (other / "state.db").exists() and not (other / "runs").exists()


def test_status_says_when_the_agent_is_not_running(tmp_path):
    home = tmp_path / "home"
    assert cli(home, "-y", "init", "--fake-gpus", "2").returncode == 0
    r = cli(home, "-y", "run", "toy-mlp")
    assert r.returncode == 0 and "Queued j_0001" in r.stdout and "agent is not running" in r.stderr

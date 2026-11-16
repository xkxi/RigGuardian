"""Runner / lifecycle findings with real processes (toy training script, local launcher)."""
import json
import os
import signal
import sys
import threading
import time

from pathlib import Path

import pytest
import yaml

from conftest import events, outbox, queue, run_until, state
from guardian.checkpoints import CheckpointStore
from guardian.daemon import Daemon
from guardian.runner import LocalLauncher, Runner
from conftest import JOB_PYTHON, make_cfg, write_toy_template


def hb(d, jid):
    try:
        return json.loads((d.runner.run_dir(d.db.get_job(jid)) / "heartbeat").read_text())
    except (OSError, ValueError):
        return {}


def write_script_template(root, name, body, **tpl):
    script = root / f"{name}.py"
    script.write_text(body)
    (root / "templates").mkdir(exist_ok=True)
    data = {"name": name, "script": str(script), "python": JOB_PYTHON, "defaults": {"epochs": 1}, **tpl}
    (root / "templates" / f"{name}.yaml").write_text(yaml.safe_dump(data))


# ---- B6: /stop on a job that hangs: no stop timeout, then it is killed "for stalling" and RESTARTED
def test_b6_stop_of_hung_sdk_job_must_end_stopped_not_relaunched(make_daemon):
    d = make_daemon(stall={"no_progress_s": 4.0, "kill_after_s": 2.0}, runner={"stop_timeout_s": 3.0})
    jid = queue(d, {"epochs": 2, "hang_at_step": 12})
    run_until(d, lambda: state(d, jid) == "running" and time.time() - hb(d, jid).get("progress_ts", 1e12) > 1.5, timeout=60)
    d.runner.request(jid, "stop")
    deadline_set = d.db.get_job(jid)["kill_deadline"]
    run_until(d, lambda: state(d, jid) != "pausing", timeout=60, watchdog=True)
    job = d.db.get_job(jid)
    try:
        assert deadline_set is not None and job["state"] == "stopped" and job["launches"] == 1, (
            f"the user asked to STOP {jid}. kill_deadline after /stop = {deadline_set} (runner.stop_timeout_s is "
            f"not applied to SDK jobs); the job is now '{job['state']}' with launches={job['launches']}; "
            f"events={[(e['kind'], e['detail']) for e in events(d) if e['kind'] != 'queued'][-7:]} "
            f"last messages={outbox(d)[-2:]}")
    finally:
        j = d.db.get_job(jid)
        if j["pgid"]:
            try:
                os.killpg(j["pgid"], signal.SIGKILL)
            except OSError:
                pass


# ---- B7: a clean shutdown (SIGTERM to the job, as on reboot) leaves the job 'stopped' for ever
def test_b7_job_terminated_by_shutdown_must_auto_resume_after_restart(make_daemon):
    d = make_daemon()
    jid = queue(d, {"epochs": 3, "step_sleep": 0.03})
    rd = d.runner.run_dir(d.db.get_job(jid))
    run_until(d, lambda: (CheckpointStore(rd / "checkpoints").latest_name() or "") >= "step_000010.pt")
    pgid = d.db.get_job(jid)["pgid"]
    os.killpg(pgid, signal.SIGTERM)  # what the service manager sends to everything at shutdown/reboot
    end = time.time() + 20
    while time.time() < end and not (rd / "exit_code").exists():
        time.sleep(0.1)
    d.runner.launcher.reap()
    assert (rd / "exit_code").read_text().strip() == "86"  # the SDK checkpointed and left cleanly
    d2 = Runner(d.cfg, d.db, d.store, d.notifier, d.reports)  # the agent after the reboot
    lines = d2.reconcile()
    job = d.db.get_job(jid)
    assert job["state"] == "queued", (
        f"after a clean reboot the job is '{job['state']}' (waiting_for_user={job['waiting_for_user']}) and "
        f"will never run again on its own. reconcile said: {lines}; last message: {outbox(d)[-1]!r}")


# ---- B8: no step for no_progress_s (data loading, evaluation, torch.compile) = killed as a stall
SLOW_START = '''
import time
from guardian_sdk import Guardian
g = Guardian()
time.sleep(8)            # tokenising a dataset / loading weights / torch.compile / a long evaluation
for i in range(5):
    with g.step(batch_size=1):
        time.sleep(0.05)
    g.maybe_checkpoint()
g.finish()
'''


def test_b8_alive_job_without_steps_must_not_be_killed(make_daemon, tmp_path):
    write_script_template(tmp_path, "slowstart", SLOW_START)
    d = make_daemon(stall={"no_progress_s": 3.0, "kill_after_s": 2.0})
    jid = queue(d, template="slowstart")
    run_until(d, lambda: state(d, jid) in ("completed", "failed", "stalled", "stopped"), timeout=90, watchdog=True)
    job = d.db.get_job(jid)
    kills = [e["detail"] for e in events(d, "killed")]
    assert not kills and job["launches"] == 1, (
        f"a healthy job that spent 8 s outside g.step() was killed: kills={kills}, launches={job['launches']}, "
        f"final state={job['state']}, messages={[t for t in outbox(d) if 'progress' in t or 'tall' in t]}")


# ---- B13: secrets and PYTHONPATH in the job environment ----------------------------------
DUMP_ENV = '''
import json, os
json.dump(dict(os.environ), open(os.path.join(os.environ["GUARDIAN_RUN_DIR"], "env.json"), "w"))
try:
    import guardian_sdk
    open(os.path.join(os.environ["GUARDIAN_RUN_DIR"], "sdk_ok"), "w").write("1")
except ImportError as e:
    open(os.path.join(os.environ["GUARDIAN_RUN_DIR"], "sdk_missing"), "w").write(repr(e))
'''


def test_b13_telegram_token_must_not_reach_training_jobs(make_daemon, tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "123456:SECRET-BOT-TOKEN")  # what load_env() does in the daemon
    write_script_template(tmp_path, "dump", DUMP_ENV, sdk=False)
    d = make_daemon()
    jid = queue(d, template="dump")
    run_until(d, lambda: state(d, jid) in ("completed", "failed"))
    env = json.loads((d.runner.run_dir(d.db.get_job(jid)) / "env.json").read_text())
    assert "TELEGRAM_TOKEN" not in env, f"the job sees TELEGRAM_TOKEN={env['TELEGRAM_TOKEN']}"


def test_b13_template_pythonpath_must_not_hide_the_sdk(make_daemon, tmp_path):
    write_script_template(tmp_path, "dump", DUMP_ENV, sdk=False, env={"PYTHONPATH": str(tmp_path / "mylibs")})
    d = make_daemon()
    jid = queue(d, template="dump")
    run_until(d, lambda: state(d, jid) in ("completed", "failed"))
    rd = d.runner.run_dir(d.db.get_job(jid))
    assert (rd / "sdk_ok").exists(), "template env PYTHONPATH replaced the SDK path: " + (rd / "sdk_missing").read_text()


# ---- B14: two agents on one home directory ------------------------------------------------
def test_b14_second_daemon_on_same_home_must_refuse(make_daemon, tmp_path):
    d1 = make_daemon()
    d1.startup()
    d2 = Daemon(make_cfg(tmp_path), launcher=LocalLauncher())
    with pytest.raises((SystemExit, RuntimeError, OSError)):
        d2.startup()


def test_b14_two_daemons_launch_the_same_job_twice(make_daemon, tmp_path):
    d1 = make_daemon()
    d2 = Daemon(make_cfg(tmp_path), launcher=LocalLauncher())
    d2.collector.collect_once()
    jid = queue(d1, {"epochs": 1})
    entered, release = threading.Event(), threading.Event()
    real = d1.runner.launcher.launch
    # agent 1 is inside its launcher (systemd-run takes ~100 ms in production) when agent 2 ticks
    d1.runner.launcher.launch = lambda *a: (entered.set(), release.wait(10), real(*a))[2]
    t = threading.Thread(target=d1.runner.tick)
    t.start()
    assert entered.wait(10)
    d2.runner.tick()
    release.set()
    t.join()
    del d1.runner.launcher.launch  # back to the real method
    launched = [e["detail"] for e in events(d1, "launched") if e["job_id"] == jid]
    time.sleep(6)
    d1.runner.tick()
    d2.runner.tick()
    assert len(launched) == 1, (f"{jid} was launched {len(launched)} times, two training processes on one run "
                                f"folder: {launched}; state now {state(d1, jid)}")


# ---- B11: reconcile after a power cut ignores what the user asked for ----------------------
@pytest.mark.parametrize("st,pending,reason", [("pausing", "stop", "user"), ("paused", None, "user"),
                                               ("paused", None, "thermal_hold")])
def test_b11_power_cut_must_not_restart_jobs_the_user_stopped_or_paused(make_daemon, st, pending, reason):
    d = make_daemon()
    jid = queue(d)
    d.db.update_job(jid, state=st, gpus="0", pid=2_000_000_000, pgid=2_000_000_000, proc_start=1,
                    pending_action=pending, pause_reason=reason, launched_at=time.time() - 100, launches=1)
    lines = Runner(d.cfg, d.db, d.store, d.notifier, d.reports).reconcile()
    job = d.db.get_job(jid)
    assert job["state"] != "queued", f"job was {st}/{pending}/{reason} before the power cut; reconcile: {lines}"


def test_b17_done_marker_without_exit_code_is_a_completed_job(make_daemon):
    d = make_daemon()
    jid = queue(d)
    rd = d.runner.run_dir(d.db.get_job(jid))
    d.db.update_job(jid, state="running", gpus="0", pid=2_000_000_000, pgid=2_000_000_000, proc_start=1,
                    launched_at=time.time() - 100, launches=1)
    (rd / "done").write_text("2026-11-15T10:00:00\n")  # g.finish() ran; power failed before sh wrote exit_code
    lines = Runner(d.cfg, d.db, d.store, d.notifier, d.reports).reconcile()
    assert state(d, jid) == "completed", f"reconcile: {lines}; report: {outbox(d)[-1][:80]!r}"


def test_b17_run_seconds_must_not_include_agent_downtime(make_daemon, clock):
    d = make_daemon(clock=clock)
    jid = queue(d)
    rd = d.runner.run_dir(d.db.get_job(jid))
    d.db.update_job(jid, state="running", gpus="0", pid=2_000_000_000, pgid=2_000_000_000, proc_start=1,
                    launched_at=clock(), launches=1)
    (rd / "heartbeat").write_text(json.dumps({"ts": clock() + 600, "step": 100}))
    (rd / "exit_code").write_text("0\n")
    (rd / "done").write_text("x\n")       # the job ran 10 minutes and finished ...
    clock.advance(3 * 86400)              # ... the agent comes back three days later
    Runner(d.cfg, d.db, d.store, d.notifier, d.reports, clock=clock).reconcile()
    secs = d.db.get_job(jid)["run_seconds"]
    assert secs < 3600, f"run_seconds={secs:.0f} ({secs / 86400:.1f} days) for a 10-minute run; this feeds every estimate"


# ---- B15: a recreated state.db reuses runs/j_0001 and silently resumes the OLD run ----------
def test_b15_new_database_must_not_resume_someone_elses_checkpoints(make_daemon, tmp_path):
    d = make_daemon()
    old = queue(d, {"epochs": 1})
    run_until(d, lambda: state(d, old) == "completed")
    for p in tmp_path.glob("state.db*"):
        p.unlink()  # "reset the queue"; runs/ is kept
    d2 = Daemon(make_cfg(tmp_path), launcher=LocalLauncher())
    d2.collector.collect_once()
    new = queue(d2, {"epochs": 1})
    run_until(d2, lambda: state(d2, new) == "completed")
    log = (d2.runner.run_dir(d2.db.get_job(new)) / "stdout.log").read_text()
    second = log.split("=== rig-guardian launch")[-1]
    assert "resumed from" not in second, (
        f"new job {new} (same id as the old run) did not train: " + [ln for ln in second.splitlines() if "resumed" in ln][0])


# ---- B18: a job without the SDK is shown as 'preflight' for its whole life -----------------
def test_b18_non_sdk_job_reaches_running(make_daemon, tmp_path):
    write_script_template(tmp_path, "plain", "import time\ntime.sleep(4)\n", sdk=False)
    d = make_daemon()
    jid = queue(d, template="plain")
    seen = set()
    run_until(d, lambda: (seen.add(state(d, jid)), state(d, jid) == "completed")[1])
    assert "running" in seen, f"states seen during a 4 s run: {sorted(seen)}"


# ---- B27: local launcher: the wrapper shell dies, training lives on, the agent calls the job failed
def test_b27_orphaned_training_process_is_not_left_running(make_daemon):
    d = make_daemon()
    jid = queue(d, {"epochs": 300, "step_sleep": 0.05})
    run_until(d, lambda: state(d, jid) == "running")
    job = d.db.get_job(jid)
    py = int((d.runner.run_dir(job) / "rank0.pid").read_text())
    os.kill(job["pid"], signal.SIGKILL)  # jobs.pid is the `sh` wrapper, not the trainer (OOM killer, manual kill -9)
    run_until(d, lambda: state(d, jid) not in ("running", "preflight"))
    time.sleep(1.0)
    try:  # killed but not yet reaped by init still has a /proc entry (state Z): that is not running
        alive = Path(f"/proc/{py}/stat").read_text().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except OSError:
        alive = False
    try:
        assert not alive, (f"{jid} is reported '{state(d, jid)}' and GPU {job['gpus']} is "
                           f"{'free' if int(job['gpus']) in d.runner.free_gpus() else 'busy'} for the scheduler, "
                           f"but its training process {py} is still running")
    finally:
        try:
            os.kill(py, signal.SIGKILL)
        except OSError:
            pass

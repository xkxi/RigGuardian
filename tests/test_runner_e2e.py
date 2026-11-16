"""End to end: real training processes (the toy script + SDK) under the runner."""

import json
import os
import signal
import time

import pytest

from conftest import outbox, queue, run_until, state
from guardian.checkpoints import CheckpointStore
from guardian.reports import read_metrics
from guardian.runner import Runner


def losses(d, job_id):
    return {r["step"]: r["loss"] for r in read_metrics(d.runner.run_dir(d.db.get_job(job_id))) if "loss" in r}


def test_normal_run_completes_with_report(make_daemon):
    d = make_daemon()
    jid = queue(d, {"epochs": 2})
    run_until(d, lambda: state(d, jid) == "completed")
    job = d.db.get_job(jid)
    rd = d.runner.run_dir(job)
    assert (rd / "done").exists() and job["exit_code"] == 0
    mans = CheckpointStore(rd / "checkpoints").manifests()
    assert any(m.get("final") for m in mans) and any(m.get("is_best") for m in mans)
    assert len(mans) <= 3 + 1 + 1
    texts = outbox(d)
    assert any(t.startswith(f"▶ {jid}") for t in texts)  # queue update
    report = next(t for t in texts if t.startswith(f"✅ {jid}"))
    assert "val_loss" in report and "final + best saved" in report


def test_kill9_then_resume_matches_uninterrupted(make_daemon):
    d = make_daemon()
    ref = queue(d, {"epochs": 2, "step_sleep": 0.0})
    run_until(d, lambda: state(d, ref) == "completed")

    jid = queue(d, {"epochs": 2, "step_sleep": 0.04})
    rd = d.runner.run_dir(d.db.get_job(jid))
    run_until(d, lambda: (CheckpointStore(rd / "checkpoints").latest_name() or "") >= "step_000015.pt")
    pid = int((rd / "rank0.pid").read_text())
    os.kill(pid, signal.SIGKILL)  # mid-epoch
    run_until(d, lambda: state(d, jid) == "failed")
    assert "Last 30 log lines" in outbox(d)[-1]

    d.runner.request(jid, "resume")
    run_until(d, lambda: state(d, jid) == "completed")
    a, b = losses(d, ref), losses(d, jid)
    assert set(a) == set(b)
    assert max(abs(a[s] - b[s]) for s in a) < 1e-6  # same loss curve as the uninterrupted run
    assert "resumed from step_" in (rd / "stdout.log").read_text()


def test_oom_retries_with_half_micro_batch_same_effective_batch(make_daemon):
    d = make_daemon()
    jid = queue(d, {"epochs": 1, "micro_batch": 64, "oom_at_step": 8})
    run_until(d, lambda: state(d, jid) == "completed")
    job = d.db.get_job(jid)
    assert (job["micro_batch"], job["grad_accum"], job["oom_retries"]) == (32, 4, 1)
    assert any("OOM at step" in t and "micro-batch 32 × 4" in t for t in outbox(d))
    cfg = json.loads((d.runner.run_dir(job) / "config.json").read_text())
    assert cfg["micro_batch"] * cfg["grad_accum"] == 64 * 2


def test_oom_gives_up_after_max_retries(make_daemon):
    d = make_daemon()
    jid = queue(d, {"epochs": 1, "micro_batch": 64, "simulate_oom_above": 8})
    run_until(d, lambda: state(d, jid) == "oom")
    job = d.db.get_job(jid)
    assert job["oom_retries"] == 2 and job["micro_batch"] == 16 and job["waiting_for_user"] == 1
    assert "Gave up after 2 OOM retries" in outbox(d)[-1]


def test_pause_resume_stop_resume(make_daemon):
    d = make_daemon()
    jid = queue(d, {"epochs": 3, "step_sleep": 0.03})
    run_until(d, lambda: state(d, jid) == "running")
    d.runner.request(jid, "pause")
    run_until(d, lambda: state(d, jid) == "paused")
    rd = d.runner.run_dir(d.db.get_job(jid))
    hb = json.loads((rd / "heartbeat").read_text())
    assert hb["paused"] is True
    step = hb["step"]
    assert CheckpointStore(rd / "checkpoints").manifest_for_step(step)["reason"] == "pause"
    time.sleep(1.0)
    assert json.loads((rd / "heartbeat").read_text())["step"] == step  # idle, process alive

    d.runner.request(jid, "resume")
    run_until(d, lambda: json.loads((rd / "heartbeat").read_text())["step"] > step + 2)
    d.runner.request(jid, "stop")
    run_until(d, lambda: state(d, jid) == "stopped")
    assert d.db.get_job(jid)["exit_code"] == 86

    d.runner.request(jid, "resume")
    assert state(d, jid) == "queued"
    run_until(d, lambda: state(d, jid) == "completed")


def test_daemon_restart_reattaches_running_job(make_daemon):
    d = make_daemon()
    jid = queue(d, {"epochs": 2, "step_sleep": 0.03})
    run_until(d, lambda: state(d, jid) == "running")
    pid = d.db.get_job(jid)["pid"]

    # a new daemon process: fresh runner, same database and run folders
    d2 = Runner(d.cfg, d.db, d.store, d.notifier, d.reports)
    lines = d2.reconcile()
    assert lines == [f"Reattached {jid}."]
    assert d.db.get_job(jid)["pid"] == pid
    d.runner = d2
    run_until(d, lambda: state(d, jid) == "completed")


def test_power_cut_resumes_from_last_verified_checkpoint(make_daemon):
    d = make_daemon()
    jid = queue(d, {"epochs": 2, "step_sleep": 0.03})
    rd = d.runner.run_dir(d.db.get_job(jid))
    run_until(d, lambda: (CheckpointStore(rd / "checkpoints").latest_name() or "") >= "step_000010.pt")
    os.killpg(d.db.get_job(jid)["pgid"], signal.SIGKILL)  # everything dies at once, no exit code
    time.sleep(0.3)
    d.runner.launcher.reap()
    (rd / "checkpoints" / "step_999999.pt.tmp").write_bytes(b"half")  # interrupted save

    d2 = Runner(d.cfg, d.db, d.store, d.notifier, d.reports)
    lines = d2.reconcile()
    assert lines and lines[0].startswith(f"Resuming {jid} from step ")
    assert not (rd / "checkpoints" / "step_999999.pt.tmp").exists()
    assert state(d, jid) == "queued"
    d.runner = d2
    d2.settle_until = 0
    run_until(d, lambda: state(d, jid) == "completed")


def test_stall_is_killed_resumed_once_then_waits(make_daemon):
    d = make_daemon(stall={"no_progress_s": 1.0, "kill_after_s": 1.0})
    jid = queue(d, {"epochs": 2, "hang_at_step": 12})
    run_until(d, lambda: state(d, jid) == "stalled", timeout=90, watchdog=True)
    job = d.db.get_job(jid)
    assert job["stalls"] == 1 and job["waiting_for_user"] == 1
    texts = outbox(d)
    assert any("no progress for" in t for t in texts)
    assert any("Resuming once from step" in t for t in texts)


def test_crash_reports_log_tail(make_daemon):
    d = make_daemon()
    jid = queue(d, {"epochs": 1, "crash_at_step": 3})
    run_until(d, lambda: state(d, jid) == "failed")
    report = outbox(d)[-1]
    assert report.startswith(f"❌ {jid}") and "simulated crash" in report
    assert d.db.one("SELECT urgent FROM outbox ORDER BY id DESC LIMIT 1")["urgent"] == 1


def test_scheduler_never_uses_agent_gpu_and_picks_coolest(make_daemon):
    d = make_daemon()
    d.source.set(0, temp_c=60)
    d.source.set(1, temp_c=45)
    d.collector.collect_once()
    assert d.runner.free_gpus() == [1, 0]
    d.source.set(1, mem_used=4 * 2**30)  # another program holds a quarter of GPU 1's memory
    d.collector.collect_once()
    assert d.runner.free_gpus() == [0]
    d.db.set_gpu_state(0, quarantined=1, quarantine_reason="test")
    assert d.runner.free_gpus() == []


def test_static_memory_check_rejects_huge_model(make_daemon):
    d = make_daemon()
    t = d.runner.templates()["toy-mlp"]
    t.params_millions = 2000  # 2 B params × 16 bytes = 32 GB > 90% of 16 GB
    from guardian.templates import TemplateError

    with pytest.raises(TemplateError, match="Rejected"):
        d.runner.static_check(t)

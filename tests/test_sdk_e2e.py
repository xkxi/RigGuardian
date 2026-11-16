"""Training-side end-to-end tests from the independent review, under the real runner: two
processes with DDP on CPU (gloo), stalls, leftover temp files, checkpoint requests that
cannot be met."""
import json
import os
import signal
import sys
import time
from pathlib import Path

import yaml

from conftest import JOB_PYTHON, outbox, queue, run_until, state, write_toy_template
from guardian.checkpoints import CheckpointStore
from guardian.reports import read_metrics

HERE = Path(__file__).resolve().parent / "sdk_harness"


def tpl(root, gpus=2):
    (root / "templates").mkdir(exist_ok=True)
    t = {"name": "ddp", "script": str(HERE / "train_ddp.py"), "python": JOB_PYTHON, "gpus": gpus,
         "best_metric": "val_loss", "mode": "min",
         "defaults": {"lr": 0.01, "epochs": 2, "micro_batch": 32, "grad_accum": 2, "seed": 0, "n_samples": 1024},
         "overridable": {k: {"type": "int", "min": 0, "max": 100000} for k in
                         ("epochs", "oom_rank", "oom_at_step", "rank_metric", "rng_per_rank", "n_samples", "micro_batch")}
                        | {"step_sleep": {"type": "float", "min": 0, "max": 5}, "slow_save": {"type": "float", "min": 0, "max": 60}}}
    (root / "templates" / "ddp.yaml").write_text(yaml.safe_dump(t))

def rows(rd, rank):
    return [json.loads(l) for l in (rd / f"ddp_r{rank}.log").read_text().splitlines()]

def events(d, jid):
    return [(r["kind"], r["detail"]) for r in d.db.all("SELECT * FROM events WHERE job_id=?", (jid,))]

def guardian_lines(rd):
    return "\n".join(l for l in (rd / "stdout.log").read_text().splitlines() if "guardian" in l or "===" in l)


def test_shards_disjoint_and_resume_consistent(make_daemon, tmp_path):
    d = make_daemon(); tpl(tmp_path)
    jid = queue(d, {"n_samples": 1001, "step_sleep": 0.05}, template="ddp")
    rd = d.runner.run_dir(d.db.get_job(jid))
    run_until(d, lambda: (CheckpointStore(rd / "checkpoints").latest_name() or "") >= "step_000005.pt", timeout=120)
    time.sleep(0.4)
    os.killpg(d.db.get_job(jid)["pgid"], signal.SIGKILL)
    run_until(d, lambda: state(d, jid) == "failed")
    d.runner.request(jid, "resume")
    run_until(d, lambda: state(d, jid) == "completed", timeout=120)
    per = {}
    for rank in (0, 1):
        cur = {}
        for r in rows(rd, rank):
            if r.get("launch"):
                cur = {k: v for k, v in cur.items() if k[1] < r["step"]}   # redo after the checkpoint
            elif "idx" in r:
                cur.setdefault((r["e"], r["s"]), []).extend(r["idx"])
        per[rank] = cur
    for e in (0, 1):
        a = [i for (ep, s), v in per[0].items() if ep == e for i in v]
        b = [i for (ep, s), v in per[1].items() if ep == e for i in v]
        print(f"epoch {e}: rank0 {len(a)} rank1 {len(b)} overlap {len(set(a) & set(b))} union {len(set(a) | set(b))} dup0 {len(a) - len(set(a))}")
        assert not set(a) & set(b) and len(set(a) | set(b)) == 1000 and len(a) == len(set(a))


def test_rng_state_after_resume_is_rank0s_on_every_rank(make_daemon, tmp_path):
    d = make_daemon(); tpl(tmp_path)
    jid = queue(d, {"rng_per_rank": 1, "step_sleep": 0.05}, template="ddp")
    rd = d.runner.run_dir(d.db.get_job(jid))
    run_until(d, lambda: (CheckpointStore(rd / "checkpoints").latest_name() or "") >= "step_000005.pt", timeout=120)
    os.killpg(d.db.get_job(jid)["pgid"], signal.SIGKILL)
    run_until(d, lambda: state(d, jid) == "failed")
    d.runner.request(jid, "resume")
    run_until(d, lambda: state(d, jid) == "completed", timeout=120)
    l0 = [r for r in rows(rd, 0) if r.get("launch")]; l1 = [r for r in rows(rd, 1) if r.get("launch")]
    print("first launch  rng fingerprints r0/r1:", l0[0]["rng"], l1[0]["rng"])
    print("after resume  rng fingerprints r0/r1:", l0[1]["rng"], l1[1]["rng"])
    assert l0[0]["rng"] != l1[0]["rng"]
    assert l0[1]["rng"] != l1[1]["rng"], "every rank resumed with rank 0's RNG state"


def test_oom_on_one_rank(make_daemon, tmp_path):
    d = make_daemon(); tpl(tmp_path)
    jid = queue(d, {"micro_batch": 64, "oom_at_step": 2, "oom_rank": 1, "epochs": 1, "step_sleep": 0.05}, template="ddp")
    rd = d.runner.run_dir(d.db.get_job(jid))
    run_until(d, lambda: (rd / "exit_intent").exists(), timeout=120)
    t0 = time.time()
    run_until(d, lambda: d.db.get_job(jid)["launches"] >= 2 or state(d, jid) in ("failed", "oom", "completed"), timeout=180)
    t1 = time.time()
    run_until(d, lambda: state(d, jid) in ("completed", "failed", "oom"), timeout=180)
    job = d.db.get_job(jid)
    print("rank 1 wrote exit_intent; the runner saw the job exit", round(t1 - t0, 1), "s later; final", job["state"],
          job["micro_batch"], job["grad_accum"], job["oom_retries"])
    print(guardian_lines(rd)); print(events(d, jid))
    assert job["state"] == "completed" and job["micro_batch"] == 32


def test_metric_without_force_differs_per_rank(make_daemon, tmp_path):
    d = make_daemon(); tpl(tmp_path)
    jid = queue(d, {"rank_metric": 1, "epochs": 3}, template="ddp")
    try:
        run_until(d, lambda: state(d, jid) in ("completed", "failed"), timeout=45)
    finally:
        job = d.db.get_job(jid); rd = d.runner.run_dir(job)
        print("state after 45 s:", job["state"], "| heartbeat step", json.loads((rd / "heartbeat").read_text())["step"],
              "age", round(time.time() - json.loads((rd / "heartbeat").read_text())["ts"], 1), "s")
        for rank in (0, 1):
            print(f"rank {rank} tail:", [r for r in rows(rd, rank) if "idx" not in r][-3:])
        print((rd / "stdout.log").read_text()[-800:])
        if job["pgid"]:
            try: os.killpg(job["pgid"], signal.SIGKILL)
            except OSError: pass
    assert state(d, jid) == "completed"


def test_periodic_save_due_while_previous_save_still_writing(make_daemon, tmp_path):
    """every_steps=5 at 0.05 s/step = a save every ~0.5 s, each save takes 2 s."""
    d = make_daemon(); tpl(tmp_path)
    jid = queue(d, {"epochs": 3, "step_sleep": 0.05, "slow_save": 2.0}, template="ddp")
    try:
        run_until(d, lambda: state(d, jid) in ("completed", "failed"), timeout=60)
    finally:
        job = d.db.get_job(jid); rd = d.runner.run_dir(job)
        hb = json.loads((rd / "heartbeat").read_text())
        print("state after 60 s:", job["state"], "| heartbeat step", hb["step"], "age", round(time.time() - hb["ts"], 1), "s")
        for rank in (0, 1):
            r = [x for x in rows(rd, rank) if "idx" in x]
            print(f"rank {rank}: last micro-batch logged at epoch {r[-1]['e']} step {r[-1]['s']} ({len(r)} micro-batches)")
        print((rd / "stdout.log").read_text()[-600:])
        if job["pgid"]:
            try: os.killpg(job["pgid"], signal.SIGKILL)
            except OSError: pass
    assert state(d, jid) == "completed"


def test_control_same_slow_save_single_process_completes(make_daemon, tmp_path):
    d = make_daemon(); tpl(tmp_path, gpus=1)
    jid = queue(d, {"epochs": 3, "step_sleep": 0.05, "slow_save": 2.0}, template="ddp")
    run_until(d, lambda: state(d, jid) in ("completed", "failed"), timeout=120)
    assert state(d, jid) == "completed"


def tpl_h(root):
    (root / "templates").mkdir(exist_ok=True)
    t = {"name": "h", "script": str(HERE / "train_h.py"), "python": JOB_PYTHON, "gpus": 1,
         "best_metric": "val_loss", "mode": "min",
         "defaults": {"lr": 0.003, "epochs": 3, "micro_batch": 32, "grad_accum": 2, "seed": 0, "n": 512},
         "overridable": {k: {"type": "int", "min": 0, "max": 100000} for k in ("epochs", "n", "micro_batch")}
                        | {k: {"type": "float", "min": 0, "max": 600} for k in ("step_sleep", "val_sleep", "slow_save")}}
    (root / "templates" / "h.yaml").write_text(yaml.safe_dump(t))

def test_long_validation_is_treated_as_a_stall(make_daemon, tmp_path):
    """Scaled down 900x: stall after 1 s without a heartbeat (default 900 s), kill 1 s later (default 300 s);
    validation takes 6 s (i.e. anything over ~20 min with the defaults)."""
    d = make_daemon(stall={"no_progress_s": 1.0, "kill_after_s": 1.0}); tpl_h(tmp_path)
    jid = queue(d, {"epochs": 2, "val_sleep": 6.0}, template="h")
    run_until(d, lambda: state(d, jid) in ("completed", "stalled", "failed"), timeout=120, watchdog=True)
    job = d.db.get_job(jid)
    print("final state:", job["state"], "stalls:", job["stalls"])
    print([e for e in events(d, jid) if e[0] in ("stall", "killed", "launched", "exited")])
    print([t[:110] for t in outbox(d) if "progress" in t or "Stalled" in t])
    assert job["state"] == "completed"


def test_tmp_file_of_a_save_interrupted_by_kill9_is_never_removed(make_daemon, tmp_path):
    d = make_daemon(checkpoints={"every_steps": 100000}); tpl_h(tmp_path)
    jid = queue(d, {"epochs": 4, "step_sleep": 0.03, "slow_save": 3.0}, template="h")
    rd = d.runner.run_dir(d.db.get_job(jid)); ck = rd / "checkpoints"
    run_until(d, lambda: (rd / "heartbeat").exists() and 9 <= json.loads((rd / "heartbeat").read_text())["step"] <= 12)   # mid-epoch 1
    d.runner.request(jid, "checkpoint")                        # /ckpt: a synchronous save, 3 s long here
    run_until(d, lambda: any(p.name.endswith(".pt.tmp") and "000008" not in p.name for p in ck.iterdir()), timeout=30)
    os.kill(int((rd / "rank0.pid").read_text()), signal.SIGKILL)   # e.g. the kill deadline of an emergency stop, or the OOM killer
    run_until(d, lambda: state(d, jid) == "failed")
    left = sorted(p.name for p in ck.iterdir())
    d.runner.request(jid, "resume")
    run_until(d, lambda: state(d, jid) == "completed", timeout=120)
    after = sorted(p.name for p in ck.iterdir())
    print("after the kill:", left); print("after the job completed:", after)
    assert not [n for n in after if n.endswith(".tmp")]


def test_ckpt_request_is_confirmed_although_nothing_was_saved(make_daemon, tmp_path):
    d = make_daemon(checkpoints={"min_free_disk_gb": 10**7, "every_steps": 5}); tpl_h(tmp_path)
    jid = queue(d, {"epochs": 30, "step_sleep": 0.03}, template="h")
    rd = d.runner.run_dir(d.db.get_job(jid)); ck = rd / "checkpoints"
    run_until(d, lambda: state(d, jid) == "running" and json.loads((rd / "heartbeat").read_text())["step"] >= 3)
    d.runner.request(jid, "checkpoint")
    run_until(d, lambda: d.db.get_job(jid)["pending_action"] is None)
    msgs = [t for t in outbox(d) if "checkpoint" in t.lower() or "disk" in t.lower()]
    files = sorted(p.name for p in ck.iterdir())
    d.runner.request(jid, "stop"); run_until(d, lambda: state(d, jid) == "stopped")
    print("messages:", msgs); print("checkpoint files at that time:", files)
    print("\n".join(l for l in (rd / "stdout.log").read_text().splitlines() if "guardian" in l)[:600])
    assert files or not any("checkpoint saved" in m for m in msgs)


def losses(d, jid):
    return {r["step"]: r["loss"] for r in read_metrics(d.runner.run_dir(d.db.get_job(jid))) if "loss" in r}


def test_ddp_completes_and_kill_resume_matches(make_daemon, tmp_path):
    d = make_daemon()
    write_toy_template(tmp_path, gpus=2)
    ref = queue(d, {"epochs": 2})
    run_until(d, lambda: state(d, ref) in ("completed", "failed"), timeout=120)
    rd0 = d.runner.run_dir(d.db.get_job(ref))
    print((rd0 / "stdout.log").read_text()[-1500:])
    assert state(d, ref) == "completed"
    jid = queue(d, {"epochs": 2, "step_sleep": 0.05})
    rd = d.runner.run_dir(d.db.get_job(jid))
    run_until(d, lambda: (CheckpointStore(rd / "checkpoints").latest_name() or "") >= "step_000010.pt", timeout=120)
    os.killpg(d.db.get_job(jid)["pgid"], signal.SIGKILL)
    run_until(d, lambda: state(d, jid) == "failed")
    d.runner.request(jid, "resume")
    run_until(d, lambda: state(d, jid) == "completed", timeout=120)
    a, b = losses(d, ref), losses(d, jid)
    print("steps", len(a), len(b), "max diff", max(abs(a[s] - b[s]) for s in a if s in b))
    assert set(a) == set(b) and max(abs(a[s] - b[s]) for s in a) < 1e-6


def test_ddp_pause_resume_stop(make_daemon, tmp_path):
    d = make_daemon()
    write_toy_template(tmp_path, gpus=2)
    jid = queue(d, {"epochs": 3, "step_sleep": 0.05})
    run_until(d, lambda: state(d, jid) == "running", timeout=120)
    d.runner.request(jid, "pause")
    run_until(d, lambda: state(d, jid) == "paused")
    rd = d.runner.run_dir(d.db.get_job(jid))
    step = json.loads((rd / "heartbeat").read_text())["step"]
    d.runner.request(jid, "resume")
    run_until(d, lambda: json.loads((rd / "heartbeat").read_text())["step"] > step + 2)
    t0 = time.time()
    d.runner.request(jid, "stop")
    run_until(d, lambda: state(d, jid) == "stopped", timeout=120)
    print("stop took", round(time.time() - t0, 1), "s; exit_code", d.db.get_job(jid)["exit_code"])
    print("\n".join(l for l in (rd / "stdout.log").read_text().splitlines() if "guardian" in l or "===" in l))
    d.runner.request(jid, "resume")
    run_until(d, lambda: state(d, jid) == "completed", timeout=120)

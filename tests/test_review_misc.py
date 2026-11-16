"""Smaller findings: power-cap helper retries, job-environment, CLI vs. daemon, database."""
import json
import os
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from conftest import events, feed, outbox, queue, running_job
from guardian.db import Database

REPO = Path(__file__).resolve().parent.parent


# ---- B23: a failing power-cap helper is re-run through sudo on every 2-second tick ---------
def test_b23_failing_powercap_helper_is_not_retried_every_tick(make_daemon, clock, tmp_path):
    helper = tmp_path / "helper.sh"
    helper.write_text(f"#!/bin/sh\necho call >> {tmp_path}/calls\nexit 3\n")
    helper.chmod(0o755)
    d = make_daemon(clock=clock, thermal={"power_cap": {"enabled": True, "watts": 50, "helper": str(helper)}})
    orig = subprocess.run
    import guardian.watchdog as wd
    wd.subprocess.run = lambda cmd, **kw: orig(cmd[2:] if cmd[:2] == ["sudo", "-n"] else cmd, **kw)  # no sudo needed here
    try:
        running_job(d, clock)
        feed(d, clock, 1, [79] * 60)  # two minutes in the warn tier
    finally:
        wd.subprocess.run = orig
    calls = len((tmp_path / "calls").read_text().splitlines())
    assert calls <= 3, f"helper (sudo) was executed {calls} times in 120 s, each inside the watchdog tick and its lock"


def test_b23_power_cap_survives_agent_restart(make_daemon, clock, tmp_path):
    helper = tmp_path / "helper.sh"
    helper.write_text(f"#!/bin/sh\necho \"$@\" >> {tmp_path}/calls\n")
    helper.chmod(0o755)
    over = dict(clock=clock, thermal={"power_cap": {"enabled": True, "watts": 150, "helper": str(helper)}})
    import guardian.watchdog as wd
    orig = subprocess.run
    wd.subprocess.run = lambda cmd, **kw: orig(cmd[2:] if cmd[:2] == ["sudo", "-n"] else cmd, **kw)
    try:
        d = make_daemon(**over)
        feed(d, clock, 1, [79] * 5)          # warn tier: limit lowered to 150 W
        d2 = make_daemon(**over)              # the agent restarts (upgrade, crash, reboot of the service)
        feed(d2, clock, 1, [50] * 400)        # card cool for 13 minutes
    finally:
        wd.subprocess.run = orig
    calls = (tmp_path / "calls").read_text().split("\n")
    assert "1 reset" in calls, f"helper calls: {calls}: the card stays capped at 150 W for good"


# ---- B24: concurrent enqueue from CLI processes and file-name races ------------------------
def test_b24_job_ids_are_unique_under_concurrent_inserts(tmp_path):
    db = Database(tmp_path / "s.db")
    ids, errs = [], []

    def worker():
        mine = Database(tmp_path / "s.db")
        for _ in range(50):
            try:
                ids.append(mine.insert_job(template="t", state="queued"))
            except Exception as e:  # noqa: BLE001
                errs.append(repr(e))

    ts = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs and len(set(ids)) == 400, (len(set(ids)), errs[:2])


def test_b24_atomic_write_from_two_writers(tmp_path):
    """CLI and daemon may both write control.json; both use the same '<name>.tmp'."""
    from guardian.fsutil import atomic_write_json
    errs = []

    def worker(i):
        for n in range(300):
            try:
                atomic_write_json(tmp_path / "control.json", {"action": "pause", "request_id": n, "w": i}, durable=False)
            except Exception as e:  # noqa: BLE001
                errs.append(repr(e))

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs, f"{len(errs)} of 600 writes failed, e.g. {errs[0]}"


# ---- B25: tables that only grow -----------------------------------------------------------
def test_b25_events_and_audit_are_pruned(make_daemon, clock):
    d = make_daemon(clock=clock)
    for i in range(500):
        d.db.execute("INSERT INTO events VALUES (?,?,?,?,?)", (clock() - 400 * 86400, None, None, "old", "x"))
        d.db.audit(1, "/status", "get_status", {}, None, "ok")
    d.db.execute("UPDATE audit SET ts=?", (clock() - 400 * 86400,))
    clock.advance(86400)
    d.maintenance_tick()
    n_e = d.db.one("SELECT COUNT(*) AS n FROM events WHERE kind='old'")["n"]
    n_a = d.db.one("SELECT COUNT(*) AS n FROM audit")["n"]
    assert n_e == 0 and n_a == 0, f"after the nightly prune: {n_e} events and {n_a} audit rows older than a year remain"


def test_b25_schema_has_a_version(tmp_path):
    db = Database(tmp_path / "s.db")
    v = db.one("PRAGMA user_version")["user_version"]
    assert v > 0 or db.kv_get("schema_version"), "no schema version: a later release cannot tell which migration to run"


# ---- B26: `rig-guardian status` shows the dead agent's last readings as if they were live ----
def test_b26_status_says_when_readings_are_stale(tmp_path):
    from guardian.config import load_config
    from guardian.daemon import Daemon
    from guardian.runner import LocalLauncher
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    cli = [sys.executable, "-m", "guardian", "--home", str(tmp_path / "home")]
    subprocess.run([*cli, "-y", "init", "--fake-gpus", "2"], env=env, capture_output=True, check=True)
    d = Daemon(load_config(tmp_path / "home" / "config.yaml"), launcher=LocalLauncher())
    d.source.set(0, temp_c=91)
    d.startup()  # writes telemetry_latest and gpu_inventory, as the running agent does
    samples = json.loads(d.db.kv_get("telemetry_latest"))
    for s in samples:
        s["ts"] = time.time() - 3 * 86400  # ... and then the agent died three days ago
    d.db.kv_set("telemetry_latest", json.dumps(samples))
    out = subprocess.run([*cli, "status"], env=env, capture_output=True, text=True).stdout
    assert "91" in out
    assert any(w in out.lower() for w in ("stale", "not running", "ago", "old")), (
        "3-day-old readings shown as current, with no hint that the agent is not running:\n" + out)

"""SQLite state database: the single source of truth for jobs, telemetry, events and audit.

WAL mode lets the bot and the CLI read while the watchdog writes. Each thread gets its own
connection; a busy timeout covers the daemon and CLI writing at the same moment.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ACTIVE_STATES = ("preflight", "running", "pausing", "paused")  # hold a GPU and a process
FINAL_STATES = ("completed", "failed", "rejected", "cancelled")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, template TEXT, config_json TEXT, config_hash TEXT,
  state TEXT, priority INTEGER DEFAULT 0, gpus TEXT,
  pid INTEGER, pgid INTEGER, proc_start INTEGER,
  micro_batch INTEGER, grad_accum INTEGER, oom_retries INTEGER DEFAULT 0,
  auto_resumes_today INTEGER DEFAULT 0, parent_job TEXT,
  created_at TEXT, started_at TEXT, ended_at TEXT, exit_code INTEGER,
  -- additions beyond the guide's minimum
  seq INTEGER, n_gpus INTEGER DEFAULT 1, run_dir TEXT, launcher TEXT, unit TEXT,
  launches INTEGER DEFAULT 0, launched_at REAL, run_seconds REAL DEFAULT 0,
  request_id INTEGER DEFAULT 0, pending_action TEXT, pending_since REAL,
  pause_reason TEXT, pause_detail TEXT, auto_resume_day TEXT,
  stalls INTEGER DEFAULT 0, stall_flagged_at REAL,
  kill_deadline REAL, kill_reason TEXT, waiting_for_user INTEGER DEFAULT 0,
  note TEXT, created_by INTEGER
);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state);
CREATE TABLE IF NOT EXISTS telemetry (gpu INTEGER, ts REAL, temp_c REAL, temp_max_c REAL,
  mem_used INTEGER, util REAL, power_w REAL, throttle INTEGER);
CREATE INDEX IF NOT EXISTS telemetry_gpu_ts ON telemetry(gpu, ts);
CREATE TABLE IF NOT EXISTS checkpoints (job_id TEXT, step INTEGER, path TEXT, sha256 TEXT,
  bytes INTEGER, reason TEXT, metric REAL, is_best INTEGER, created_at TEXT);
CREATE UNIQUE INDEX IF NOT EXISTS checkpoints_job_path ON checkpoints(job_id, path);
CREATE TABLE IF NOT EXISTS events (ts REAL, job_id TEXT, gpu INTEGER, kind TEXT, detail TEXT);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
CREATE TABLE IF NOT EXISTS audit (ts REAL, telegram_user INTEGER, text TEXT, action TEXT,
  args_json TEXT, confirmed INTEGER, result TEXT);
CREATE TABLE IF NOT EXISTS outbox (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, kind TEXT,
  text TEXT, attachment TEXT, urgent INTEGER DEFAULT 0, hold_until REAL, sent_at REAL,
  attempts INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS outbox_unsent ON outbox(sent_at);
CREATE TABLE IF NOT EXISTS gpu_state (gpu INTEGER PRIMARY KEY, quarantined INTEGER DEFAULT 0,
  quarantine_reason TEXT, cooling INTEGER DEFAULT 0, cooling_since REAL);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
"""


SCHEMA_VERSION = 2  # stored in PRAGMA user_version; raise it with every change to the tables

# columns added after the first release: {table: {column: declaration}}; see Database._migrate
ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "outbox": {"delivered": "TEXT", "error": "TEXT"},
    "jobs": {"sdk": "INTEGER", "best_metric": "TEXT", "mode": "TEXT", "gpu_uuids": "TEXT",
             "interrupts": "INTEGER DEFAULT 0"},
}


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: str | None) -> float | None:
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._local = threading.local()
        if self.path != ":memory:" and not os.path.exists(self.path):
            # the database holds messages, the audit trail and log excerpts: owner-only
            os.close(os.open(self.path, os.O_CREAT | os.O_WRONLY, 0o600))
        self.conn().executescript(SCHEMA)  # executescript manages its own transaction
        self._migrate()

    def _migrate(self) -> None:
        """Add columns that newer versions introduced to a database made by an older one."""
        c = self.conn()
        for table, cols in ADDED_COLUMNS.items():
            have = {r["name"] for r in c.execute(f"PRAGMA table_info({table})").fetchall()}
            for name, decl in cols.items():
                if name not in have:
                    try:
                        c.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
                    except sqlite3.OperationalError as e:  # another process added it first
                        if "duplicate column" not in str(e).lower():
                            raise
        have = int(c.execute("PRAGMA user_version").fetchone()[0])
        if have > SCHEMA_VERSION:
            raise RuntimeError(f"{self.path} was written by a newer Rig Guardian (schema {have}, this version "
                               f"knows {SCHEMA_VERSION}); upgrade the agent instead of opening it with an old one")
        if have != SCHEMA_VERSION:
            c.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=10, isolation_level=None)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA busy_timeout=10000")
            self._local.conn = c
        return c

    class _Tx:
        def __init__(self, c: sqlite3.Connection):
            self.c = c

        def __enter__(self):
            self.c.execute("BEGIN IMMEDIATE")
            return self.c

        def __exit__(self, et, ev, tb):
            self.c.execute("ROLLBACK" if et else "COMMIT")

    def tx(self) -> "_Tx":
        return self._Tx(self.conn())

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        return self.conn().execute(sql, tuple(params))

    def all(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        return [dict(r) for r in self.conn().execute(sql, tuple(params)).fetchall()]

    def one(self, sql: str, params: Iterable[Any] = ()) -> dict | None:
        r = self.conn().execute(sql, tuple(params)).fetchone()
        return dict(r) if r else None

    # ---- jobs --------------------------------------------------------------------------
    def insert_job(self, _min_seq: int = 0, **fields: Any) -> str:
        """Add a job with the next free ID (above ``_min_seq`` too, if given)."""
        with self.tx() as c:
            row = c.execute("SELECT COALESCE(MAX(seq), 0) + 1 AS n FROM jobs").fetchone()
            seq = max(int(row["n"]), int(_min_seq) + 1)
            job_id = f"j_{seq:04d}"
            fields = {**fields, "id": job_id, "seq": seq, "created_at": now_iso()}
            cols = ",".join(fields)
            marks = ",".join("?" for _ in fields)
            c.execute(f"INSERT INTO jobs ({cols}) VALUES ({marks})", tuple(fields.values()))
        return job_id

    def get_job(self, job_id: str) -> dict | None:
        return self.one("SELECT * FROM jobs WHERE id=?", (job_id,))

    def update_job(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        sets = ",".join(f"{k}=?" for k in fields)
        self.execute(f"UPDATE jobs SET {sets} WHERE id=?", (*fields.values(), job_id))

    def jobs_in(self, states: Iterable[str]) -> list[dict]:
        states = tuple(states)
        marks = ",".join("?" for _ in states)
        return self.all(f"SELECT * FROM jobs WHERE state IN ({marks}) ORDER BY seq", states)

    def queued_jobs(self) -> list[dict]:
        return self.all("SELECT * FROM jobs WHERE state='queued' ORDER BY priority DESC, seq ASC")

    def recent_jobs(self, limit: int = 10) -> list[dict]:
        return self.all("SELECT * FROM jobs ORDER BY seq DESC LIMIT ?", (limit,))

    # ---- events, audit, telemetry ----------------------------------------------------
    def event(self, kind: str, detail: str = "", job_id: str | None = None, gpu: int | None = None) -> None:
        self.execute("INSERT INTO events VALUES (?,?,?,?,?)", (time.time(), job_id, gpu, kind, detail))

    def audit(self, user: int | None, text: str, action: str | None, args: Any,
              confirmed: int | None, result: str) -> int:
        cur = self.execute(
            "INSERT INTO audit VALUES (?,?,?,?,?,?,?)",
            (time.time(), user, text, action, json.dumps(args, default=str) if args is not None else None,
             confirmed, result),
        )
        return int(cur.lastrowid)

    def update_audit(self, rowid: int, confirmed: int, result: str) -> None:
        self.execute("UPDATE audit SET confirmed=?, result=? WHERE rowid=?", (confirmed, result, rowid))

    def add_telemetry(self, rows: list[tuple]) -> None:
        with self.tx() as c:
            c.executemany("INSERT INTO telemetry VALUES (?,?,?,?,?,?,?,?)", rows)

    def prune_telemetry(self, older_than_ts: float) -> int:
        return self.execute("DELETE FROM telemetry WHERE ts < ?", (older_than_ts,)).rowcount

    def prune_log(self, older_than_ts: float) -> int:
        """Drop old rows from the event and audit logs (they only ever grow otherwise)."""
        n = self.execute("DELETE FROM events WHERE ts < ?", (older_than_ts,)).rowcount
        return n + self.execute("DELETE FROM audit WHERE ts < ?", (older_than_ts,)).rowcount

    def record_checkpoint(self, job_id: str, man: dict) -> None:
        self.execute(
            "INSERT OR REPLACE INTO checkpoints VALUES (?,?,?,?,?,?,?,?,?)",
            (job_id, man.get("step"), man.get("file"), man.get("sha256"), man.get("bytes"),
             man.get("reason"), man.get("metric"), int(bool(man.get("is_best"))), man.get("saved_at")),
        )

    # ---- GPU state -------------------------------------------------------------------
    def gpu_state(self, gpu: int) -> dict:
        r = self.one("SELECT * FROM gpu_state WHERE gpu=?", (gpu,))
        return r or {"gpu": gpu, "quarantined": 0, "quarantine_reason": None, "cooling": 0, "cooling_since": None}

    def set_gpu_state(self, gpu: int, **fields: Any) -> None:
        cur = self.gpu_state(gpu)
        cur.update(fields)
        self.execute(
            "INSERT OR REPLACE INTO gpu_state VALUES (?,?,?,?,?)",
            (gpu, cur["quarantined"], cur["quarantine_reason"], cur["cooling"], cur["cooling_since"]),
        )

    # ---- key/value -------------------------------------------------------------------
    def kv_get(self, key: str, default: str | None = None) -> str | None:
        r = self.one("SELECT value FROM kv WHERE key=?", (key,))
        return r["value"] if r else default

    def kv_set(self, key: str, value: str | None) -> None:
        if value is None:
            self.execute("DELETE FROM kv WHERE key=?", (key,))
        else:
            self.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (key, value))

import json
import os
import sys
import time
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
# The Python that runs the training scripts in these tests: it needs PyTorch. By default the
# one running the tests; set RG_TEST_JOB_PYTHON to test the agent under another Python version.
JOB_PYTHON = os.environ.get("RG_TEST_JOB_PYTHON") or sys.executable

from guardian.config import Config  # noqa: E402
from guardian.daemon import Daemon  # noqa: E402
from guardian.examples import toy_template  # noqa: E402
from guardian.runner import LocalLauncher  # noqa: E402


class FakeClock:
    def __init__(self, t: float = 1_800_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


V100 = {"name": "Tesla V100-PCIE-16GB", "mem_gb": 16, "slowdown_c": 87}
P4000 = {"name": "Quadro P4000", "mem_gb": 8, "slowdown_c": 87}


def make_cfg(root: Path, **over) -> Config:
    data = {
        "gpus": {"training": [0, 1], "agent": 2, "expect_names": {"training": "V100", "agent": "P4000"}},
        "telemetry": {"source": "fake", "interval_s": 2, "fake_gpus": [V100, V100, P4000]},
        "runner": {"launcher": "local", "settle_s": 0, "poll_s": 0.2, "heartbeat_s": 0.2},
        "checkpoints": {"every_steps": 5, "every_minutes": 20, "min_free_disk_gb": 0},
        "telegram": {"allowed_user_ids": [42], "group_window_s": 0, "quiet_hours": None, "digest_time": None},
        "brain": {"enabled": False},
    }
    for k, v in over.items():
        if k == "gpus":
            data["gpus"] = v  # replaced whole: roles depend on each other
        else:
            data.setdefault(k, {}).update(v)
    return Config.model_validate(data).with_root(root)


def write_toy_template(root: Path, **changes) -> None:
    (root / "templates").mkdir(exist_ok=True)
    toy = {**toy_template(JOB_PYTHON), **changes}
    (root / "templates" / f"{toy['name']}.yaml").write_text(yaml.safe_dump(toy))


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def make_daemon(tmp_path):
    """A full daemon (fake GPUs, local launcher, real clock) in a temporary rig folder."""

    def _make(clock=time.time, **over) -> Daemon:
        write_toy_template(tmp_path)
        cfg = make_cfg(tmp_path, **over)
        d = Daemon(cfg, launcher=LocalLauncher(), clock=clock)
        d.collector.collect_once()
        return d

    return _make


def outbox(d) -> list[str]:
    return [r["text"] for r in d.db.all("SELECT text FROM outbox ORDER BY id")]


def run_until(d, pred, timeout: float = 60.0, watchdog: bool = False):
    end = time.time() + timeout
    while time.time() < end:
        d.collector.collect_once()
        d.runner.tick()
        if watchdog:
            d.watchdog.tick()
        v = pred()
        if v:
            return v
        time.sleep(0.1)
    raise AssertionError("condition not reached in time; job: "
                         + json.dumps(d.db.recent_jobs(1), default=str)[:1500])


def queue(d, overrides=None, template="toy-mlp") -> str:
    prep = d.runner.prepare_run(template, overrides or {})
    return d.runner.enqueue(prep)


def state(d, job_id) -> str:
    return d.db.get_job(job_id)["state"]


def feed(d, clock, gpu, temps, **extra):
    """One telemetry sample and one watchdog tick per temperature, 2 s apart."""
    for t in temps:
        d.source.set(gpu, temp_c=t, **extra)
        clock.advance(2)
        d.collector.collect_once()
        d.watchdog.tick()


def running_job(d, clock, gpus="1") -> str:
    """A job row that looks like it is running on ``gpus`` (no process is started)."""
    jid = queue(d)
    d.db.update_job(jid, state="running", gpus=gpus, launched_at=clock())
    return jid


def events(d, kind=None) -> list[dict]:
    rows = d.db.all("SELECT * FROM events ORDER BY ts")
    return [r for r in rows if kind is None or r["kind"] == kind]

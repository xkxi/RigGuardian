import time

from conftest import FakeClock
from guardian.config import TelegramCfg
from guardian.db import Database
from guardian.notifier import Notifier, split_message


def make(tmp_path, **cfg):
    clock = FakeClock(time.mktime((2026, 10, 6, 12, 0, 0, 0, 0, -1)))  # local noon
    n = Notifier(Database(tmp_path / "s.db"), TelegramCfg(**cfg), clock=clock)
    return n, clock


def test_warnings_grouped_within_window(tmp_path):
    n, clock = make(tmp_path, group_window_s=60)
    n.warn("a", key="a")
    n.warn("b", key="b")
    n.flush()
    assert n.ready() == []
    clock.advance(61)
    n.flush()
    [m] = n.ready()
    assert m["text"] == "⚠️ Warnings\n• a\n• b"


def test_same_warning_at_most_hourly(tmp_path):
    n, clock = make(tmp_path, group_window_s=0)
    assert n.warn("hot", key="gpu1")
    assert not n.warn("hot", key="gpu1")
    clock.advance(3601)
    assert n.warn("hot", key="gpu1")
    assert len(n.ready()) == 2


def test_quiet_hours_hold_reports_but_not_alerts(tmp_path):
    n, clock = make(tmp_path, quiet_hours=["11:00", "13:00"])
    n.report("run finished")
    n.alert("GPU paused")
    ready = n.ready()
    assert [m["text"] for m in ready] == ["GPU paused"]
    n.mark_sent(ready[0]["ids"])
    clock.advance(3601)  # 13:00:01
    [digest] = n.ready()
    assert digest["text"].startswith("🌙") and "run finished" in digest["text"]


def test_unsent_messages_wait_in_outbox(tmp_path):
    n, _ = make(tmp_path)
    n.send("hello")
    [m] = n.ready()
    n.mark_failed(m["ids"])  # Telegram unreachable
    assert n.ready()[0]["text"] == "hello"


def test_split_long_messages():
    parts = split_message("line\n" * 2000)
    assert all(len(p) <= 4096 for p in parts) and "".join(parts) == "line\n" * 2000

"""Outbox delivery (bot._send_loop) and notifier findings."""
import asyncio
import time
import types

from guardian.bot import TelegramBot
from guardian.config import TelegramCfg
from guardian.db import Database
from guardian.notifier import Notifier, next_time, split_message
from conftest import FakeClock


class FakeBot:
    def __init__(self, fail_chats=(), fail_if=None):
        self.sent, self.fail_chats, self.fail_if = [], set(fail_chats), fail_if

    async def send_message(self, chat_id, text, reply_markup=None):
        if chat_id in self.fail_chats or (self.fail_if and self.fail_if(text)):
            raise RuntimeError("Forbidden: bot can't initiate conversation with a user")
        self.sent.append((chat_id, text))


def run_send_loop(tmp_path, users, bot, messages, seconds=7.0):
    n = Notifier(Database(tmp_path / "s.db"), TelegramCfg(allowed_user_ids=users))
    for m in messages:
        n.alert(m)
    tb = TelegramBot("token", agent=None, notifier=n, allowed_user_ids=users)
    tb.app = types.SimpleNamespace(bot=bot)

    async def go():
        task = asyncio.create_task(tb._send_loop())
        await asyncio.sleep(seconds)
        task.cancel()

    asyncio.run(go())
    return n


# ---- B21: one undeliverable recipient or message blocks the outbox and floods the others ----
def test_b21_one_unreachable_user_must_not_spam_the_other_and_block_the_queue(tmp_path):
    bot = FakeBot(fail_chats={222})  # the second allow-listed user never pressed Start (or blocked the bot)
    run_send_loop(tmp_path, [111, 222], bot, ["🚨 GPU 1 at 88 °C. Stopping j_0001.", "⏹ j_0001 stopped."])
    to_first = [t for c, t in bot.sent if c == 111]
    assert to_first == ["🚨 GPU 1 at 88 °C. Stopping j_0001.", "⏹ j_0001 stopped."], (
        f"user 111 received {len(to_first)} messages in 7 s: {[t[:12] for t in to_first]}")


def test_b21_one_rejected_message_must_not_block_later_alerts(tmp_path):
    bot = FakeBot(fail_if=lambda text: "poison" in text)  # e.g. Telegram answers 400 Bad Request for this text
    run_send_loop(tmp_path, [111], bot, ["poison report", "🚨 GPU 1 at 88 °C. Stopping j_0001."])
    assert any("🚨" in t for _, t in bot.sent), "the emergency alert queued behind a rejected message was never sent"


def test_b22_split_message_respects_telegrams_utf16_limit():
    text = "🧠 " + "x" * 4094  # 4096 Python characters, 4097 UTF-16 units: Telegram rejects it as too long
    parts = split_message(text)
    assert all(len(p.encode("utf-16-le")) // 2 <= 4096 for p in parts), [len(p.encode("utf-16-le")) // 2 for p in parts]


def test_b22_quiet_hours_end_on_a_dst_day(monkeypatch):
    import os
    monkeypatch.setenv("TZ", "Europe/Berlin")
    time.tzset()
    try:
        now = time.mktime((2026, 10, 24, 23, 30, 0, 0, 0, -1))  # the night the clocks go back
        end = next_time(now, "07:00")
        got = time.strftime("%Y-%m-%d %H:%M", time.localtime(end))
        assert got == "2026-10-25 07:00", f"messages held until {got}"
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()


def test_b22_unsent_outbox_rows_are_bounded(tmp_path):
    """Telegram configured but unreachable for months: rows are only ever deleted once sent."""
    clock = FakeClock()
    db = Database(tmp_path / "s.db")
    n = Notifier(db, TelegramCfg(), clock=clock)
    for _ in range(1000):
        n.send("x")
        clock.advance(86400 / 10)
    db.execute("DELETE FROM outbox WHERE sent_at IS NOT NULL AND sent_at < ?", (clock() - 30 * 86400,))  # the nightly prune
    left = db.one("SELECT COUNT(*) AS n FROM outbox")["n"]
    assert left < 1000, f"{left} unsent rows after 100 days; all of them are delivered in a burst when Telegram returns"


def test_b21_empty_allowlist_must_not_discard_alerts(tmp_path):
    """Token in .env, allowed_user_ids still [] (the README's own setup order): alerts are marked sent to nobody."""
    bot = FakeBot()
    n = run_send_loop(tmp_path, [], bot, ["🚨 GPU 1 at 88 °C. Stopping j_0001."], seconds=3)
    row = n.db.one("SELECT sent_at FROM outbox")
    assert bot.sent or row["sent_at"] is None, "the emergency alert was marked as sent although nobody received it"


def test_b28_env_file_common_forms(tmp_path, monkeypatch):
    import os
    from guardian.config import load_env
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.delenv("OTHER", raising=False)
    (tmp_path / ".env").write_text('export TELEGRAM_TOKEN="123:abc"\nOTHER=value # my comment\n')
    (tmp_path / ".env").chmod(0o600)
    load_env(tmp_path / ".env")
    try:
        assert os.environ.get("TELEGRAM_TOKEN") == "123:abc" and os.environ.get("OTHER") == "value", (
            {k: v for k, v in os.environ.items() if "TELEGRAM" in k or k == "OTHER"})
    finally:
        for k in ("export TELEGRAM_TOKEN", "OTHER"):
            os.environ.pop(k, None)

"""The Telegram layer, driven through a real python-telegram-bot Application with an offline
request object (no network): who gets an answer, and what happens to the outbox when Telegram
refuses a chat or a message, or the network is down.
"""

import asyncio
import json
import time

import pytest
from telegram import Update
from telegram.error import TimedOut
from telegram.ext import Application, CallbackQueryHandler, MessageHandler, filters
from telegram.request import BaseRequest

from conftest import outbox
from guardian.bot import TelegramBot
from guardian.notifier import redact, split_message, tg_len

A, B, STRANGER = 42, 43, 999
TOKEN = "123456:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"


class OfflineRequest(BaseRequest):
    """Answers the Bot API locally and records every call."""

    def __init__(self, fail_chats=(), down_chats=()):
        self.calls, self.fail_chats, self.down_chats = [], set(fail_chats), set(down_chats)

    @property
    def read_timeout(self):
        return 1.0

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    async def do_request(self, url, method, request_data=None, *a, **kw):
        endpoint = url.rsplit("/", 1)[-1]
        params = dict(request_data.parameters) if request_data else {}
        if int(params.get("chat_id", 0) or 0) in self.down_chats:
            raise TimedOut("simulated: no network")  # nothing reached Telegram: not recorded
        self.calls.append((endpoint, params))
        if endpoint == "getMe":
            res = {"id": 1, "is_bot": True, "first_name": "rg", "username": "rg_bot"}
        elif endpoint in ("sendMessage", "sendDocument", "sendPhoto", "editMessageReplyMarkup"):
            chat = int(params.get("chat_id", 0))
            if chat in self.fail_chats:
                return 403, json.dumps({"ok": False, "error_code": 403, "description":
                                        "Forbidden: bot can't initiate conversation with a user"}).encode()
            if endpoint == "sendMessage" and not str(params.get("text", "")).strip():
                return 400, json.dumps({"ok": False, "error_code": 400,
                                        "description": "Bad Request: message text is empty"}).encode()
            res = {"message_id": len(self.calls), "date": 0, "chat": {"id": chat, "type": "private"}, "text": "x"}
        else:
            res = True
        return 200, json.dumps({"ok": True, "result": res}).encode()

    def sent(self, chat=None):
        return [p for e, p in self.calls if e == "sendMessage" and (chat is None or int(p["chat_id"]) == chat)]

    def texts(self, chat=None):
        return [p["text"] for p in self.sent(chat)]


async def make_bot(d, req, allowed=(A,)):
    """The same wiring as TelegramBot.start(), minus polling."""
    d.cfg.telegram.allowed_user_ids = list(allowed)
    bot = TelegramBot(TOKEN, d.agent, d.notifier, list(allowed))
    bot.app = Application.builder().token(TOKEN).request(req).get_updates_request(req).build()
    bot.app.add_handler(MessageHandler(filters.TEXT | filters.COMMAND, bot._on_message))
    bot.app.add_handler(CallbackQueryHandler(bot._on_callback))
    await bot.app.initialize()
    return bot


def msg(uid, text, kind="message", chat_type="private", chat_id=None, age_s=0, **extra):
    chat = {"id": chat_id if chat_id is not None else uid, "type": chat_type}
    m = {"message_id": 1, "date": int(time.time() - age_s), "chat": chat, "text": text, **extra}
    if uid is not None:
        m["from"] = {"id": uid, "is_bot": False, "first_name": "u"}
    return {"update_id": 1, kind: m}


def cb(uid, data, chat_id=None):
    return {"update_id": 2, "callback_query": {
        "id": "1", "from": {"id": uid, "is_bot": False, "first_name": "u"}, "chat_instance": "x", "data": data,
        "message": {"message_id": 5, "date": 0, "chat": {"id": chat_id or uid, "type": "private"}, "text": "q"}}}


async def feed(bot, data):
    await bot.app.process_update(Update.de_json(data, bot.app.bot))


def button(offer):
    rm = offer["reply_markup"]
    rm = json.loads(rm) if isinstance(rm, str) else rm
    return rm["inline_keyboard"][0][0]["callback_data"]


async def send_rounds(bot, monkeypatch, n):
    """Run the real send loop for ``n`` rounds."""
    rounds = 0

    async def fake_sleep(_):
        nonlocal rounds
        rounds += 1
        if rounds >= n:
            raise asyncio.CancelledError

    monkeypatch.setattr("guardian.bot.asyncio.sleep", fake_sleep)
    with pytest.raises(asyncio.CancelledError):
        await bot._send_loop()


@pytest.fixture
def d(make_daemon):
    d = make_daemon()
    d.db.execute("DELETE FROM outbox")
    return d


# ---- who gets an answer ------------------------------------------------------------------
def test_only_allowlisted_private_chats_get_an_answer_and_a_confirmation_belongs_to_its_owner(d):
    async def go():
        req = OfflineRequest()
        bot = await make_bot(d, req, allowed=(A, B))
        await feed(bot, msg(STRANGER, "/status"))
        await feed(bot, msg(A, "/status", chat_type="group", chat_id=-100))
        await feed(bot, msg(A, "/status", chat_type="supergroup", chat_id=-100))
        await feed(bot, msg(None, "/status", kind="channel_post", chat_type="channel", chat_id=-200))
        await feed(bot, msg(None, "/status"))  # no sender at all
        assert req.sent() == []
        await feed(bot, msg(A, "/run toy-mlp"))
        [offer] = req.sent(A)
        code = button(offer)
        n = len(req.calls)
        await feed(bot, cb(STRANGER, code))  # a stranger taps A's button
        assert len(req.calls) == n and d.db.queued_jobs() == []  # not even answerCallbackQuery
        await feed(bot, cb(B, code))  # another allow-listed user taps it
        assert d.db.queued_jobs() == [] and "belongs to someone else" in req.texts(B)[-1]
        await feed(bot, cb(A, code))  # ...which must not have used up A's own confirmation
        assert len(d.db.queued_jobs()) == 1 and "Queued j_0001" in req.texts(A)[-1]
        await feed(bot, cb(A, code))  # one use only
        assert len(d.db.queued_jobs()) == 1 and "already been used" in req.texts(A)[-1]
    asyncio.run(go())


def test_edited_forwarded_and_stale_messages_are_not_acted_on(d):
    async def go():
        req = OfflineRequest()
        bot = await make_bot(d, req)
        await feed(bot, msg(A, "/hold", kind="edited_message"))
        assert d.db.kv_get("hold") is None and req.sent() == []
        fwd = {"forward_origin": {"type": "user", "date": 0,
                                  "sender_user": {"id": STRANGER, "is_bot": False, "first_name": "evil"}}}
        await feed(bot, msg(A, "/hold", **fwd))
        assert d.db.kv_get("hold") is None and "forwarded" in req.texts(A)[-1]
        await feed(bot, msg(A, "/hold", age_s=3 * 3600))  # typed while the agent was down
        assert d.db.kv_get("hold") is None and "offline" in req.texts(A)[-1]
        await feed(bot, msg(A, "/hold"))
        assert d.db.kv_get("hold") == "1"
    asyncio.run(go())


def test_a_handler_error_still_answers_the_user(d, monkeypatch):
    async def go():
        req = OfflineRequest()
        bot = await make_bot(d, req)
        monkeypatch.setattr(d.agent, "handle_text", lambda *a, **k: 1 / 0)
        await feed(bot, msg(A, "/status"))
        assert "Something went wrong" in req.texts(A)[-1]
    asyncio.run(go())


# ---- the outbox can never be blocked -----------------------------------------------------
def test_an_unreachable_chat_does_not_block_or_repeat_alerts_for_the_others(d, monkeypatch):
    async def go():
        req = OfflineRequest(fail_chats={B})  # B never pressed Start (or the ID has a typo)
        bot = await make_bot(d, req, allowed=(A, B))
        d.notifier.alert("first alert")
        d.notifier.alert("🚨 GPU 0 at 95 °C, stopping j_0001")
        await send_rounds(bot, monkeypatch, 5)
        texts = req.texts(A)
        assert texts.count("first alert") == 1 and texts.count("🚨 GPU 0 at 95 °C, stopping j_0001") == 1
        assert sum("cannot be messaged" in t for t in texts) == 1  # A is told once about B
        rows = d.db.all("SELECT sent_at, error FROM outbox ORDER BY id")
        assert all(r["sent_at"] is not None for r in rows) and "Forbidden" in rows[0]["error"]
        assert B in bot.unreachable
    asyncio.run(go())


def test_blank_messages_and_missing_pictures_cannot_block_the_outbox(d, monkeypatch):
    async def go():
        req = OfflineRequest()
        bot = await make_bot(d, req)
        d.notifier.alert("   \n  ")  # dropped when written: Telegram refuses blank text
        d.notifier.report("loss curve", attachment=str(d.cfg.root / "gone.png"))
        d.notifier.alert("🚨 real alert")
        await send_rounds(bot, monkeypatch, 3)
        texts = req.texts(A)
        assert texts == ["loss curve\n(the image is no longer available)", "🚨 real alert"]
        assert d.db.one("SELECT COUNT(*) AS n FROM outbox WHERE sent_at IS NULL")["n"] == 0
    asyncio.run(go())


def test_network_down_keeps_messages_in_order_and_never_sends_one_twice_to_the_same_chat(d, monkeypatch):
    async def go():
        req = OfflineRequest(down_chats={B})  # A's message gets through, then the network drops
        bot = await make_bot(d, req, allowed=(A, B))
        d.notifier.alert("one")
        d.notifier.alert("two")
        await send_rounds(bot, monkeypatch, 3)
        assert req.texts(A) == ["one"]  # "two" waits behind "one"; "one" is not repeated to A
        assert d.db.one("SELECT COUNT(*) AS n FROM outbox WHERE sent_at IS NULL")["n"] == 2
        req.down_chats.clear()  # the network is back
        await send_rounds(bot, monkeypatch, 2)
        assert req.texts(A) == ["one", "two"] and req.texts(B) == ["one", "two"]
        assert d.db.one("SELECT COUNT(*) AS n FROM outbox WHERE sent_at IS NULL")["n"] == 0
    asyncio.run(go())


def test_bot_tokens_never_leave_in_a_message(d, monkeypatch):
    async def go():
        req = OfflineRequest()
        bot = await make_bot(d, req)
        d.notifier.alert(f"❌ j_0001 failed\nLast 30 log lines:\nTELEGRAM_TOKEN={TOKEN}\nend")
        await send_rounds(bot, monkeypatch, 2)
        assert TOKEN not in req.texts(A)[0] and "[bot token removed]" in req.texts(A)[0]
        assert TOKEN not in "".join(outbox(d))  # not stored either
    asyncio.run(go())
    assert redact(f"x {TOKEN} y") == "x [bot token removed] y" and redact("step 123456: loss 0.5") == "step 123456: loss 0.5"


# ---- message splitting ---------------------------------------------------------------------
def test_split_counts_the_way_telegram_does_and_never_returns_blank_parts():
    emoji = "😀" * 4096  # 4,096 characters, but 8,192 UTF-16 units
    parts = split_message(emoji)
    assert len(parts) == 2 and all(tg_len(p) <= 4096 for p in parts) and "".join(parts) == emoji
    assert split_message("") == [] and split_message("  \n ") == []
    long = "\n".join(f"line {i} " + "x" * 90 for i in range(200))
    parts = split_message(long)
    assert all(tg_len(p) <= 4096 and p.strip() for p in parts) and "".join(parts) == long

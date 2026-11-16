"""Telegram bot (python-telegram-bot >= 20), run inside the daemon's event loop.

Long polling only: the rig makes outgoing HTTPS requests and opens no ports. Only private
chats from allow-listed user IDs get answers; everyone else is dropped without a reply
(and logged), so the bot does not even confirm it exists. All decisions live in
``actions.Agent``; this module only moves text, buttons, photos and files.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from guardian.actions import BOT_COMMANDS, Agent, Reply
from guardian.notifier import MAX_ATTEMPTS, Notifier, redact, split_message

log = logging.getLogger(__name__)


class TelegramBot:
    def __init__(self, token: str, agent: Agent, notifier: Notifier, allowed_user_ids: list[int],
                 max_message_age_s: float = 120.0):
        self.token, self.agent, self.notifier = token, agent, notifier
        self.chat_ids = list(dict.fromkeys(allowed_user_ids))  # a private chat's ID is the user's ID
        self.max_message_age_s = max_message_age_s
        self.app = None
        self._sender: asyncio.Task | None = None
        self.unreachable: dict[int, str] = {}  # chat -> why Telegram refuses it (until they write to the bot)
        self._told_nobody = False

    async def start(self) -> None:
        from telegram import BotCommand, Update
        from telegram.ext import Application, CallbackQueryHandler, MessageHandler, filters

        self.app = Application.builder().token(self.token).build()
        self.app.add_handler(MessageHandler(filters.TEXT | filters.COMMAND, self._on_message))
        self.app.add_handler(CallbackQueryHandler(self._on_callback))
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(allowed_updates=[Update.MESSAGE, Update.CALLBACK_QUERY])
        try:
            await self.app.bot.set_my_commands([BotCommand(c, d) for c, d in BOT_COMMANDS])
        except Exception as e:  # cosmetic only
            log.warning("could not set the command menu: %s", type(e).__name__)
        self._sender = asyncio.create_task(self._send_loop())
        log.info("telegram bot polling")

    async def stop(self) -> None:
        if self._sender:
            self._sender.cancel()
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()

    # ---- incoming ------------------------------------------------------------------------
    async def _on_message(self, update, context) -> None:
        # only a message the user typed to the bot: not an edit of an old one, not a channel post
        msg, user, chat = update.message, update.effective_user, update.effective_chat
        if msg is None or chat is None:
            return
        uid = user.id if user else None
        if not self.agent.allowed(uid, chat.type):
            await asyncio.to_thread(self.agent.handle_text, uid, msg.text or "", chat.type)  # logs the refusal
            return
        text = msg.text or ""
        if self.unreachable.pop(chat.id, None) is not None:
            log.info("Telegram user %s can be messaged again", chat.id)  # they pressed Start
        if getattr(msg, "forward_origin", None) is not None:
            await self._send_reply(chat.id, Reply("That is a forwarded message, so I did not act on it. "
                                                  "Type the command yourself if you want it."))
            return
        age = (datetime.now(timezone.utc) - msg.date).total_seconds() if msg.date else 0
        if age > self.max_message_age_s:  # sent while the agent was down; Telegram keeps these for a day
            await self._send_reply(chat.id, Reply(
                f"I was offline when you sent “{text[:60]}” ({int(age // 60)} min ago), so I did not act on it. "
                f"Send it again if you still want it."))
            return
        try:
            replies = await asyncio.to_thread(self.agent.handle_text, uid, text, chat.type)
        except Exception:  # never leave the user without an answer
            log.exception("handling a message failed")
            replies = [Reply("🚫 Something went wrong handling that message (details are in the agent's log). "
                             "Nothing was changed.")]
        for r in replies or []:
            try:
                await self._send_reply(chat.id, r)
            except Exception as e:
                log.warning("could not send a reply: %s", type(e).__name__)

    async def _on_callback(self, update, context) -> None:
        q, user = update.callback_query, update.effective_user
        if not self.agent.allowed(user.id if user else None):
            self.agent._denied(user.id if user else None, f"callback from user {user.id if user else None}")
            return  # no answer at all
        try:
            reply = await asyncio.to_thread(self.agent.handle_callback, user.id, q.data or "")
        except Exception:
            log.exception("handling a button failed")
            reply = Reply("🚫 Something went wrong (details are in the agent's log). Check /jobs before retrying.")
        try:
            await q.answer()
            await q.edit_message_reply_markup(reply_markup=None)  # one tap only
        except Exception:
            pass
        if reply is not None and q.message is not None:
            try:
                await self._send_reply(q.message.chat.id, reply)
            except Exception as e:
                log.warning("could not send a reply: %s", type(e).__name__)

    # ---- outgoing ------------------------------------------------------------------------
    async def _send_reply(self, chat_id: int, r: Reply) -> None:
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        bot = self.app.bot
        text = redact(r.text or "")
        if r.photo:
            try:
                with open(r.photo, "rb") as f:
                    await bot.send_photo(chat_id, f, caption=text[:1000] or None)
                return
            except OSError:  # the picture is gone (the run folder was cleaned): the words still go out
                text = (text + "\n(the image is no longer available)").strip()
        if r.document:
            name, data = r.document
            await bot.send_document(chat_id, data, filename=name, caption=text[:1000] or None)
            return
        markup = None
        if r.buttons:
            markup = InlineKeyboardMarkup([[InlineKeyboardButton(label, callback_data=data)
                                            for label, data in r.buttons]])
        parts = split_message(text)
        if not parts:
            if markup is None:
                return  # nothing to say; Telegram refuses blank text
            parts = ["Confirm?"]
        if len(parts) > 3:  # long output goes as a file
            await bot.send_document(chat_id, text.encode(), filename="message.txt",
                                    caption=parts[0][:200], reply_markup=markup)
            return
        for i, part in enumerate(parts):
            await bot.send_message(chat_id, part, reply_markup=markup if i == len(parts) - 1 else None)

    async def _deliver(self, m: dict) -> bool | None:
        """Send one outbox message to every chat that has not had it. False: the network is
        down, stop this round and retry later. None: nobody can be reached at all, stop this round. A chat or a message Telegram refuses for good
        is recorded and skipped, so it can never hold up the messages behind it."""
        from telegram.error import BadRequest, Forbidden, InvalidToken, NetworkError, RetryAfter, TimedOut

        ids, done = m["ids"], set(m.get("delivered") or [])
        errors: list[str] = []
        if not [c for c in self.chat_ids if c not in self.unreachable]:
            # nobody can receive anything yet (empty allowlist, or nobody pressed Start): keep the
            # message, it goes out as soon as someone can
            if not self._told_nobody:
                self._told_nobody = True
                log.error("Telegram: there is nobody to send messages to. Put your user ID in "
                          "telegram.allowed_user_ids (config.yaml), restart, and press Start in the bot chat. "
                          "Messages are kept for a week.")
            return None
        for chat_id in self.chat_ids:
            if chat_id in done:
                continue
            if chat_id in self.unreachable:
                errors.append(f"{chat_id}: {self.unreachable[chat_id]}")
                continue
            try:
                await self._send_reply(chat_id, Reply(m["text"], photo=m["attachment"]))
                await asyncio.to_thread(self.notifier.mark_delivered, ids, chat_id)
                done.add(chat_id)
            except (Forbidden, InvalidToken) as e:  # blocked the bot, or never pressed Start
                why = f"{type(e).__name__}: {redact(str(e))[:120]}"
                self.unreachable[chat_id] = why
                errors.append(f"{chat_id}: {why}")
                log.error("Telegram user %s cannot be messaged (%s). They must open the bot chat and press "
                          "Start; until the next restart they are skipped.", chat_id, why)
                self.notifier.alert(f"⚠️ Telegram user {chat_id} is on the allowlist but cannot be messaged "
                                    f"({type(e).__name__}). They need to open the bot chat and press Start.")
            except RetryAfter as e:
                await asyncio.sleep(min(float(getattr(e, "retry_after", 5) or 5), 60))
                return False
            except BadRequest as e:  # this message is refused for this chat; retrying cannot help
                errors.append(f"{chat_id}: BadRequest: {redact(str(e))[:120]}")
                log.error("Telegram refused message %s for chat %s: %s", ids, chat_id, redact(str(e))[:200])
            except (TimedOut, NetworkError) as e:  # no internet: leave it in the outbox, retry later
                log.warning("telegram send failed, will retry: %s", type(e).__name__)
                return False
            except Exception as e:  # anything else: count it, give up after MAX_ATTEMPTS
                log.warning("telegram send failed (%s), attempt %d", type(e).__name__, m.get("attempts", 0) + 1)
                if m.get("attempts", 0) + 1 >= MAX_ATTEMPTS:
                    errors.append(f"{chat_id}: gave up after {MAX_ATTEMPTS} attempts ({type(e).__name__})")
                    continue
                await asyncio.to_thread(self.notifier.mark_failed, ids)
                return True  # try this one again next round, but let the others through now
        await asyncio.to_thread(self.notifier.mark_sent, ids, "; ".join(errors) or None)
        return True

    async def _send_loop(self) -> None:
        while True:
            try:
                for m in await asyncio.to_thread(self.notifier.ready):
                    ok = await self._deliver(m)
                    if ok is None:
                        break
                    if not ok:
                        await asyncio.to_thread(self.notifier.mark_failed, m["ids"])
                        break
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("send loop error")
            await asyncio.sleep(2)


async def console_send_loop(notifier: Notifier) -> None:
    """Without a Telegram token, print outgoing messages to the log instead."""
    while True:
        for m in await asyncio.to_thread(notifier.ready):
            log.info("MESSAGE%s:\n%s", f" [{m['attachment']}]" if m["attachment"] else "", m["text"])
            await asyncio.to_thread(notifier.mark_sent, m["ids"])
        await asyncio.sleep(2)

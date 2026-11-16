"""Outgoing messages: an outbox table plus the sending rules.

* Everything goes through the ``outbox`` table first, so a message written while the
  internet is down goes out when it returns. Safety actions never wait on Telegram.
* Warnings are grouped (several within 60 s become one message) and the same warning
  repeats at most once an hour.
* During quiet hours, reports and warnings are held and sent as one digest at the end.
  Urgent messages (pause, emergency, GPU lost, crash) always go out at once.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Callable

log = logging.getLogger(__name__)

TELEGRAM_LIMIT = 4096  # counted in UTF-16 code units, as Telegram counts
MAX_ATTEMPTS = 10  # a message that keeps failing for reasons other than the network is given up on

# a Telegram bot token: digits, a colon, 35 or so URL-safe characters
_TOKEN_RE = re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{30,}")


def redact(text: str) -> str:
    """Remove anything shaped like a bot token before text leaves the machine or is stored."""
    return _TOKEN_RE.sub("[bot token removed]", text)


def parse_hhmm(hhmm: str) -> int:
    """Minutes since midnight for 'HH:MM'; raises ValueError with a clear message otherwise."""
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", str(hhmm))
    if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
        raise ValueError(f"{hhmm!r} is not a time of day like '07:30'")
    return int(m.group(1)) * 60 + int(m.group(2))


UNSENT_KEEP_S = 7 * 86400


def _minutes(hhmm: str) -> int:
    return parse_hhmm(hhmm)


def in_window(now: float, start: str, end: str) -> bool:
    t = time.localtime(now)
    m = t.tm_hour * 60 + t.tm_min
    a, b = _minutes(start), _minutes(end)
    return a <= m < b if a <= b else (m >= a or m < b)


def today_at(now: float, hhmm: str) -> float:
    """Epoch seconds of local ``hhmm`` on the same day as ``now``."""
    t = time.localtime(now)
    return time.mktime((t.tm_year, t.tm_mon, t.tm_mday, *divmod(_minutes(hhmm), 60), 0, 0, 0, -1))


def next_time(now: float, hhmm: str) -> float:
    """Epoch seconds of the next local ``hhmm`` after ``now``."""
    t = time.localtime(now)
    h, m = divmod(_minutes(hhmm), 60)
    for days in (0, 1, 2):  # by calendar day, not by 24 hours: a day with a clock change is 23 or 25 long
        target = time.mktime((t.tm_year, t.tm_mon, t.tm_mday + days, h, m, 0, 0, 0, -1))
        if target > now:
            return target
    return now + 86400


class Notifier:
    def __init__(self, db, cfg, clock: Callable[[], float] = time.time):
        self.db, self.cfg, self.clock = db, cfg, clock
        self._lock = threading.Lock()
        self._last_key: dict[str, float] = {}
        self._warn_buf: list[str] = []
        self._warn_first: float | None = None
        self.queue_updates = cfg.queue_updates

    # ---- writing -----------------------------------------------------------------------
    def quiet_now(self) -> bool:
        q = self.cfg.quiet_hours
        return bool(q) and in_window(self.clock(), q[0], q[1])

    def send(self, text: str, kind: str = "info", urgent: bool = False, attachment: str | None = None) -> None:
        text = redact(str(text or "")).strip()
        if not text:
            log.warning("dropped an empty %s message", kind)  # Telegram refuses blank text
            return
        now = self.clock()
        hold = None
        try:
            if not urgent and self.quiet_now():
                hold = next_time(now, self.cfg.quiet_hours[1])
        except ValueError:  # a malformed quiet_hours must never lose a message
            hold = None
        self.db.execute(
            "INSERT INTO outbox (ts, kind, text, attachment, urgent, hold_until) VALUES (?,?,?,?,?,?)",
            (now, kind, text, attachment, int(urgent), hold),
        )
        # a message nobody could be sent for a week is no longer news: without this, months of
        # them would arrive in one burst the day Telegram becomes reachable again
        self.db.execute("DELETE FROM outbox WHERE sent_at IS NULL AND ts < ?", (now - UNSENT_KEEP_S,))

    def alert(self, text: str, attachment: str | None = None) -> None:
        """Action alerts: pause, stop, quarantine, crash. Always immediate."""
        self.send(text, kind="alert", urgent=True, attachment=attachment)

    def report(self, text: str, urgent: bool = False, attachment: str | None = None) -> None:
        self.send(text, kind="report", urgent=urgent, attachment=attachment)

    def queue_update(self, text: str) -> None:
        if self.queue_updates:
            self.send(text, kind="queue")

    def rate_ok(self, key: str, every_s: float | None = None) -> bool:
        """True (and remembered) if ``key`` was not used in the last ``every_s`` seconds."""
        every_s = self.cfg.warn_repeat_s if every_s is None else every_s
        now = self.clock()
        with self._lock:
            last = self._last_key.get(key)
            if last is not None and now - last < every_s:
                return False
            self._last_key[key] = now
            return True

    def warn(self, text: str, key: str | None = None) -> bool:
        if key is not None and not self.rate_ok(key):
            return False
        with self._lock:
            if not self._warn_buf:
                self._warn_first = self.clock()
            self._warn_buf.append(text)
        if self.cfg.group_window_s <= 0:
            self.flush(force=True)
        return True

    def flush(self, force: bool = False) -> None:
        with self._lock:
            if not self._warn_buf:
                return
            if not force and self.clock() - (self._warn_first or 0) < self.cfg.group_window_s:
                return
            items, self._warn_buf, self._warn_first = self._warn_buf, [], None
        text = items[0] if len(items) == 1 else "⚠️ Warnings\n" + "\n".join(f"• {t}" for t in items)
        self.send(text, kind="warning")

    # ---- reading (for the sender) -----------------------------------------------------
    def ready(self, limit: int = 20) -> list[dict]:
        """Messages due now. Rows held through quiet hours are merged into one digest."""
        now = self.clock()
        rows = self.db.all(
            "SELECT * FROM outbox WHERE sent_at IS NULL AND (hold_until IS NULL OR hold_until <= ?) "
            "ORDER BY id LIMIT ?", (now, limit))
        out, held = [], []
        for r in rows:
            if r["hold_until"] is not None and not r["attachment"]:
                held.append(r)
            else:
                out.append({"ids": [r["id"]], "text": r["text"], "attachment": r["attachment"],
                            "urgent": bool(r["urgent"])})
        if held:
            body = "\n\n".join(r["text"] for r in held)
            out.insert(0, {"ids": [r["id"] for r in held], "text": "🌙 While quiet hours were on:\n\n" + body,
                           "attachment": None, "urgent": False})
        by_id = {r["id"]: r for r in rows}
        for m in out:  # chats that already received it (a retry must not send it to them again)
            first = by_id[m["ids"][0]]
            try:
                m["delivered"] = [int(c) for c in json.loads(first.get("delivered") or "[]")]
            except (ValueError, TypeError):
                m["delivered"] = []
            m["attempts"] = int(first.get("attempts") or 0)
        return out

    def _where(self, ids: list[int]) -> str:
        return f"id IN ({','.join('?' * len(ids))})"

    def mark_sent(self, ids: list[int], error: str | None = None) -> None:
        """Done with these rows: delivered, or given up on (``error`` says why)."""
        self.db.execute(f"UPDATE outbox SET sent_at=?, error=? WHERE {self._where(ids)}",
                        (self.clock(), error, *ids))

    def mark_delivered(self, ids: list[int], chat_id: int) -> None:
        for r in self.db.all(f"SELECT id, delivered FROM outbox WHERE {self._where(ids)}", ids):
            try:
                got = set(json.loads(r["delivered"] or "[]"))
            except (ValueError, TypeError):
                got = set()
            got.add(int(chat_id))
            self.db.execute("UPDATE outbox SET delivered=? WHERE id=?", (json.dumps(sorted(got)), r["id"]))

    def mark_failed(self, ids: list[int]) -> None:
        self.db.execute(f"UPDATE outbox SET attempts=attempts+1 WHERE {self._where(ids)}", ids)


def tg_len(text: str) -> int:
    """Length as Telegram counts it: UTF-16 code units (an emoji outside the BMP counts as 2)."""
    return len(text) + sum(1 for ch in text if ord(ch) > 0xFFFF)


def _cut(text: str, limit: int) -> int:
    """The largest number of characters from the start of ``text`` that fit in ``limit`` units."""
    units = 0
    for i, ch in enumerate(text):
        units += 2 if ord(ch) > 0xFFFF else 1
        if units > limit:
            return i
    return len(text)


def split_message(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    """Split on line boundaries to fit Telegram's 4,096-unit limit. Never returns a blank part."""
    if tg_len(text) <= limit:
        return [text] if text.strip() else []
    parts, cur = [], ""
    for line in text.splitlines(keepends=True):
        while tg_len(line) > limit:
            if cur:
                parts.append(cur)
                cur = ""
            n = max(1, _cut(line, limit))
            parts.append(line[:n])
            line = line[n:]
        if tg_len(cur) + tg_len(line) > limit:
            parts.append(cur)
            cur = ""
        cur += line
    if cur:
        parts.append(cur)
    return [p for p in parts if p.strip()]

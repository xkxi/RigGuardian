"""The fixed action menu and the router that every interface goes through.

Slash commands, free text (through the brain) and the CLI all end up in ``prepare``,
which validates the action and its arguments and returns a summary plus a callable.
Actions that start, stop or change a job need a Confirm tap carrying a one-time code
that expires after 5 minutes. Nothing changes before that tap.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import shlex
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable

from guardian.brain import ACTION_MENU, BrainOffline
from guardian.db import ACTIVE_STATES
from guardian.reports import fmt_dur, fmt_num
from guardian.runner import ActionError
from guardian.templates import TemplateError, parse_kv_args

log = logging.getLogger(__name__)

HELP = """Rig Guardian commands
/status – GPUs, jobs and queue at a glance
/gpus – detailed telemetry and throttle state
/jobs – running, queued and last 5 finished
/job <id> – one job: state, step, metrics, ETA, checkpoints
/plot <id> – loss curve
/logs <id> – last 30 lines of output
/templates – templates and their changeable settings
/run <template> key=value … – queue a run (confirm)
/rerun <id> key=value … – queue a copy of a past run with changes (confirm)
/ckpt <id> – checkpoint now
/pause <id>, /resume <id> – soft pause or resume (confirm)
/stop <id> – checkpoint and stop (confirm)
/cancel <id> – remove a queued job (confirm)
/priority <id> <n> – reorder the queue (confirm)
/compare <id> <id> … – compare 2–5 runs
/hold, /unhold – stop or restart launching new jobs
/clear <gpu> – lift a quarantine (confirm)
/digest – today's digest now
Anything else is read by the AI and mapped to one of these."""

BOT_COMMANDS = [
    ("status", "GPUs, jobs and queue"), ("gpus", "Detailed GPU telemetry"), ("jobs", "Running, queued, recent"),
    ("job", "One job: /job <id>"), ("plot", "Loss curve: /plot <id>"), ("logs", "Last 30 log lines"),
    ("templates", "Templates and settings"), ("run", "Queue: /run <template> k=v"),
    ("rerun", "Rerun: /rerun <id> k=v"), ("ckpt", "Checkpoint now"), ("pause", "Soft pause"),
    ("resume", "Resume"), ("stop", "Checkpoint and stop"), ("cancel", "Remove a queued job"),
    ("hold", "Stop launching new jobs"), ("unhold", "Launch new jobs again"), ("clear", "Lift a quarantine"),
    ("help", "All commands"),
]


@dataclass
class Reply:
    text: str
    buttons: list[tuple[str, str]] | None = None  # (label, callback data)
    photo: str | None = None
    document: tuple[str, bytes] | None = None


@dataclass
class Prepared:
    action: str
    args: dict[str, Any]
    summary: str
    confirm: bool
    run: Callable[[], Reply]
    # what the confirmation was shown for; if this differs when Confirm is tapped, nothing runs
    fingerprint: str = ""


MAX_BRAIN_TEXT = 1000  # characters of a free-text message sent to the model
MAX_MODEL_REPLY = 300  # characters of the model's own words shown to the user


def norm_job_id(v: Any) -> str:
    s = str(v).strip().lower()[:64]
    m = re.fullmatch(r"(?:j_?)?(\d{1,9})", s)
    return f"j_{int(m.group(1)):04d}" if m else s


def model_words(text: Any, fallback: str) -> str:
    """The model's own wording, marked as such and kept short: it can be wrong."""
    s = " ".join(str(text or "").split())[:MAX_MODEL_REPLY]
    return s or fallback


class Agent:
    def __init__(self, cfg, db, runner, reports, notifier, brain=None, clock: Callable[[], float] = time.time):
        self.cfg, self.db, self.runner, self.reports, self.notifier, self.brain, self.clock = (
            cfg, db, runner, reports, notifier, brain, clock)
        self.pending: dict[str, dict] = {}
        self.history: dict[int, deque] = defaultdict(lambda: deque(maxlen=max(1, cfg.brain.history)))
        self._lock = threading.Lock()
        self._denied_at: dict[int | None, float] = {}

    # ---- entry points ------------------------------------------------------------------
    def allowed(self, user_id: int | None, chat_type: str = "private") -> bool:
        return chat_type == "private" and user_id in self.cfg.telegram.allowed_user_ids

    def _denied(self, user_id: int | None, detail: str) -> None:
        """Record a refused sender: the first message, then at most one a minute per sender, so a
        stranger hammering the bot cannot fill the database."""
        now = self.clock()
        with self._lock:
            if len(self._denied_at) > 1000:
                self._denied_at.clear()
            last = self._denied_at.get(user_id)
            if last is not None and now - last < 60:
                return
            self._denied_at[user_id] = now
        self.db.event("telegram_denied", detail)

    def handle_text(self, user_id: int | None, text: str, chat_type: str = "private",
                    trusted: bool = False) -> list[Reply] | None:
        """Replies to a message, or None to ignore it silently (unknown sender or group)."""
        if not trusted and not self.allowed(user_id, chat_type):
            self._denied(user_id, f"user {user_id} in {chat_type} chat: {str(text)[:80]!r}")
            return None
        text = (text or "").strip()
        if not text:
            return []
        if text.startswith("/"):
            return [self._slash(user_id, text)]
        return [self._free_text(user_id, text)]

    def handle_callback(self, user_id: int | None, data: str, trusted: bool = False) -> Reply | None:
        if not trusted and not self.allowed(user_id):
            self._denied(user_id, f"callback from user {user_id}")
            return None
        kind, _, code = (data or "").partition(":")
        with self._lock:
            p = self.pending.get(code)
            if p is None:
                return Reply("That button has already been used or has expired. Nothing changed.")
            if p["user"] != user_id and not trusted:  # the owner's code stays valid for the owner
                return Reply("That confirmation belongs to someone else. Nothing changed.")
            del self.pending[code]
        ttl = self.cfg.telegram.confirm_ttl_s
        if self.clock() > p["expires"]:
            self.db.update_audit(p["audit"], 0, "expired")
            return Reply(f"That confirmation expired ({fmt_dur(ttl)}). Nothing changed; send the command again.")
        if kind != "ok":
            self.db.update_audit(p["audit"], 0, "cancelled")
            return Reply("Cancelled. Nothing changed.")
        old: Prepared = p["prepared"]
        try:  # check again now: minutes may have passed since the summary was shown
            fresh = self.prepare(old.action, old.args, p["user"])
        except (ActionError, TemplateError) as e:
            reply = Reply(f"🚫 That is no longer possible: {e}. Nothing changed.")
        else:
            if fresh.fingerprint != old.fingerprint:
                reply = Reply("🚫 Things changed since that confirmation was shown, so nothing was done. "
                              "Send the command again to see the current details.")
            else:
                reply = self._execute(fresh)
        self.db.update_audit(p["audit"], 1 if not reply.text.startswith("🚫") else 0, reply.text[:500])
        return reply

    def confirm_code(self, prep: Prepared, user_id: int | None, audit_id: int) -> str:
        code = secrets.token_urlsafe(9)
        with self._lock:
            now = self.clock()
            for c in [c for c, p in self.pending.items() if p["expires"] < now]:
                del self.pending[c]
            self.pending[code] = {"user": user_id, "expires": now + self.cfg.telegram.confirm_ttl_s,
                                  "prepared": prep, "audit": audit_id}
        return code

    # ---- slash commands -------------------------------------------------------------
    def parse_slash(self, text: str) -> tuple[str, dict[str, Any]]:
        try:  # shell-style words, so a value may contain spaces: /run t note="first try"
            parts = shlex.split(text)
        except ValueError:  # an unbalanced quote: take the words as typed
            parts = text.split()
        if not parts or len(parts[0]) < 2:
            raise ActionError("Type a command such as /status. /help lists them.")
        cmd = parts[0][1:].split("@")[0].lower()
        a = parts[1:]

        def need(n: int, usage: str):
            if len(a) < n:
                raise ActionError(f"Usage: {usage}")

        if cmd in ("start", "help"):
            return "help", {}
        if cmd in ("status", "gpus", "templates", "hold", "unhold", "digest"):
            return {"status": "get_status", "gpus": "get_gpus"}.get(cmd, cmd), {}
        if cmd == "jobs":
            return "list_jobs", {}
        if cmd in ("job", "plot", "logs", "ckpt", "pause", "resume", "stop", "cancel"):
            need(1, f"/{cmd} <job id>")
            action = {"job": "get_job", "plot": "plot", "logs": "logs", "ckpt": "checkpoint_job",
                      "pause": "pause_job", "resume": "resume_job", "stop": "stop_job", "cancel": "cancel_job"}[cmd]
            return action, {"job_id": a[0]}
        if cmd == "run":
            need(1, "/run <template> key=value …")
            return "queue_run", {"template": a[0], "overrides": parse_kv_args(a[1:])}
        if cmd == "rerun":
            need(1, "/rerun <job id> key=value …")
            return "rerun", {"job_id": a[0], "overrides": parse_kv_args(a[1:])}
        if cmd == "priority":
            need(2, "/priority <job id> <number>")
            return "set_priority", {"job_id": a[0], "priority": a[1]}
        if cmd == "compare":
            return "compare_jobs", {"job_ids": a}
        if cmd == "clear":
            need(1, "/clear <gpu index>")
            return "clear_gpu", {"gpu": a[0]}
        raise ActionError(f"Unknown command /{cmd}. /help lists them.")

    def _slash(self, user_id: int | None, text: str) -> Reply:
        try:
            action, args = self.parse_slash(text)
            prep = self.prepare(action, args, user_id)
        except (ActionError, TemplateError) as e:
            self.db.audit(user_id, text, None, None, None, f"refused: {e}")
            return Reply(f"🚫 {e}")
        return self._offer(user_id, text, prep)

    def _offer(self, user_id: int | None, text: str, prep: Prepared) -> Reply:
        if prep.confirm:
            aid = self.db.audit(user_id, text, prep.action, prep.args, 0, "awaiting confirmation")
            code = self.confirm_code(prep, user_id, aid)
            return Reply(prep.summary, buttons=[("Confirm", f"ok:{code}"), ("Cancel", f"no:{code}")])
        reply = self._execute(prep)
        if prep.action not in ("get_status", "get_gpus", "list_jobs", "get_job", "templates", "help", "logs", "plot"):
            self.db.audit(user_id, text, prep.action, prep.args, None, reply.text[:500])
        return reply

    def _execute(self, prep: Prepared) -> Reply:
        try:
            return prep.run()
        except (ActionError, TemplateError) as e:
            return Reply(f"🚫 {e}")

    # ---- free text ---------------------------------------------------------------------
    def _free_text(self, user_id: int | None, text: str) -> Reply:
        if self.brain is None or not self.cfg.brain.enabled:
            return Reply("The AI is turned off; slash commands work. /help lists them.")
        if len(text) > MAX_BRAIN_TEXT:
            return Reply(f"That message is too long for the AI ({len(text):,} characters; the limit is "
                         f"{MAX_BRAIN_TEXT:,}). Slash commands work: /help lists them.")
        offline = Reply("AI offline; slash commands still work. /help lists them.")
        try:
            hist = list(self.history[user_id])
            ctx = self.brain_context()
            try:
                res = self.brain.map_message(text, ctx, hist)
            except BrainOffline as e:
                self.db.event("brain_offline", str(e)[:200])
                self.history[user_id].clear()  # an over-long history must not keep it offline
                return offline
            try:
                prep = self._prepare_from_brain(res, user_id)
            except (ActionError, TemplateError) as first:
                try:  # one retry with the error sent back to the model
                    res = self.brain.map_message(text, ctx, hist, previous=res, error=str(first))
                    prep = self._prepare_from_brain(res, user_id)
                except BrainOffline:
                    return offline
                except (ActionError, TemplateError) as second:
                    self.db.audit(user_id, text, res.action, res.args, None, f"refused: {second}")
                    example = next(iter(self.runner.templates()), "cifar-resnet")
                    return Reply(f"I couldn't map that ({second}). Try /run {example} lr=3e-4 or /help.")
            self.history[user_id].append(
                f"{text[:200]}  →  {prep.action} {json.dumps(prep.args, default=str)[:200]}")
            return self._offer(user_id, text, prep)
        except Exception:  # whatever the model server sent, the user gets an answer
            log.exception("free-text handling failed")
            return Reply("The AI's answer could not be used; slash commands work. /help lists them.")

    def _prepare_from_brain(self, res, user_id) -> Prepared:
        if res.action not in ACTION_MENU:
            raise ActionError(f"'{str(res.action)[:40]}' is not an action on the menu"
                              + (f" ({res.reply})" if res.action == "invalid" and res.reply else ""))
        allowed = set(ACTION_MENU[res.action][0])
        extra = set(res.args) - allowed
        if extra:
            raise ActionError(f"{res.action} takes only {', '.join(sorted(allowed)) or 'no arguments'}")
        missing = [a for a in allowed if a not in res.args and a != "overrides"]
        if missing:
            raise ActionError(f"{res.action} needs {', '.join(missing)}")
        return self.prepare(res.action, res.args, user_id)

    def brain_context(self) -> str:
        tpls = self.runner.templates()
        lines = ["TEMPLATES:"]
        for t in tpls.values():
            lines.append(f"- {t.name}: changeable {t.describe_limits()}")
        q = len(self.db.queued_jobs())
        lines.append("")
        lines.append(f"STATUS: {q} queued; " + "; ".join(
            f"GPU {g} {self.reports.store.label(g)}: " + ("busy" if g not in self.runner.free_gpus() else "free")
            for g in self.cfg.gpus.training))
        lines.append("")
        lines.append("RECENT JOBS (newest first):")
        for j in self.db.recent_jobs(8):
            t = tpls.get(j["template"])
            cfgd = json.loads(j["config_json"] or "{}")
            changed = {k: v for k, v in cfgd.items() if t is None or t.defaults.get(k) != v}
            sets = " ".join(f"{k}={fmt_num(v)}" for k, v in changed.items()) or "defaults"
            lines.append(f"- {j['id']} {j['template']} {j['state']} {sets}")
        return "\n".join(lines)

    # ---- validation + execution ------------------------------------------------------
    def _job(self, args: dict) -> dict:
        if "job_id" not in args:
            raise ActionError("which job? Give a job ID like j_0042")
        return self.runner.get(norm_job_id(args["job_id"]))

    def _overrides(self, args: dict) -> dict:
        ov = args.get("overrides") or {}
        if not isinstance(ov, dict):
            raise ActionError("overrides must be key=value settings")
        return ov

    def _gpu_line(self, prep: dict) -> str:
        need = prep["n_gpus"]
        free = self.runner.same_vendor_pick(prep["free"], need) or []
        est = f" · est. {fmt_dur(prep['est'])}" if prep.get("est") else ""
        if len(free) >= need:
            gpus = " + ".join(self.reports.store.label(g) for g in free[:need])
            return f"GPU: {gpus} {'is' if need == 1 else 'are'} free{est}"
        ahead = len(self.db.queued_jobs())
        return f"GPU: waits for {'a free card' if need == 1 else f'{need} free cards'} ({ahead} queued ahead){est}"

    def prepare(self, action: str, args: dict[str, Any], user_id: int | None = None) -> Prepared:
        args = dict(args or {})
        r, rep = self.runner, self.reports

        def simple(text_fn: Callable[[], str]) -> Prepared:
            return Prepared(action, args, "", False, lambda: Reply(text_fn()))

        if action == "help":
            return simple(lambda: HELP)
        if action == "get_status":
            return simple(rep.status_text)
        if action == "get_gpus":
            return simple(rep.gpus_text)
        if action == "list_jobs":
            return simple(rep.jobs_text)
        if action == "digest":
            return simple(rep.digest_text)
        if action == "templates":
            def tlist() -> str:
                tpls = r.templates()
                if not tpls:
                    return "No templates yet. Add YAML files to templates/ on the rig."
                return "\n\n".join(
                    f"{t.name} ({t.gpus} GPU{'s' if t.gpus > 1 else ''}){': ' + t.description if t.description else ''}"
                    f"\nChangeable: {t.describe_limits()}" for t in tpls.values())
            return simple(tlist)
        if action == "get_job":
            job = self._job(args)
            return simple(lambda: rep.job_text(r.get(job["id"])))
        if action == "logs":
            job = self._job(args)

            def logs() -> Reply:
                text = rep.logs_text(job, 30)
                if len(text) > 3800:
                    return Reply(f"Last 30 lines of {job['id']} attached.",
                                 document=(f"{job['id']}-tail.txt", text.encode()))
                return Reply(f"Last lines of {job['id']}:\n{text}")
            return Prepared(action, args, "", False, logs)
        if action == "plot":
            job = self._job(args)

            def plot() -> Reply:
                path = rep.plot(job)
                return Reply(f"{job['id']} loss curve", photo=path) if path else Reply(f"{job['id']} has no metrics yet.")
            return Prepared(action, args, "", False, plot)
        if action == "compare_jobs":
            ids = args.get("job_ids") or []
            if not isinstance(ids, list) or not 2 <= len(ids) <= 5:
                raise ActionError("compare needs 2 to 5 job IDs")
            jobs = [r.get(norm_job_id(i)) for i in ids]
            return simple(lambda: rep.compare_text(jobs))
        if action in ("queue_run", "rerun"):
            ov = self._overrides(args)
            if action == "rerun":
                parent = self._job(args)
                prep = r.prepare_run(None, ov, parent_id=parent["id"])
                head = f"Rerun {parent['id']} as a new {prep['template']} job?"
                args["job_id"] = parent["id"]
            else:
                prep = r.prepare_run(args.get("template"), ov)
                head = f"Queue {prep['template']}?"
            changes = ", ".join(prep["diffs"]) or ("same settings" if action == "rerun" else "default settings")
            summary = f"{head}\n  {changes}\n  {self._gpu_line(prep)}"

            def run() -> Reply:
                jid = r.enqueue(prep, priority=0, user=user_id)
                pos = [j["id"] for j in self.db.queued_jobs()].index(jid) + 1
                return Reply(f"⏳ Queued {jid} ({prep['template']}), position {pos} in the queue.")
            fp = json.dumps([prep["template"], prep["config"], prep["n_gpus"]], sort_keys=True, default=str)
            return Prepared(action, args, summary, True, run, fingerprint=fp)
        if action in ("pause_job", "resume_job", "stop_job"):
            job = self._job(args)
            verb = action.split("_")[0]
            st = job["state"]
            ok_states = {"pause": ("running", "preflight"),
                         "resume": ("paused", "pausing", "stopped", "oom", "stalled", "failed"),
                         "stop": ACTIVE_STATES}[verb]
            if st not in ok_states:
                done = {"pause": "paused", "resume": "resumed", "stop": "stopped"}[verb]
                raise ActionError(f"{job['id']} is {st}; it cannot be {done} now")
            where = f" on {rep.gpus_label(job)}" if job.get("gpus") and st in ACTIVE_STATES else ""
            what = {"pause": "Pause (checkpoint, then idle holding GPU memory)",
                    "resume": "Resume" if st in ("paused", "pausing") else "Queue again from the last checkpoint:",
                    "stop": "Stop (checkpoint, then exit)"}[verb]
            summary = f"{what} {job['id']}?\n  {job['template']}, {st}{where}"
            return Prepared(action, args, summary, True,
                            lambda: Reply(r.request(job["id"], verb, reason="user")))
        if action == "checkpoint_job":
            job = self._job(args)
            return Prepared(action, args, "", False, lambda: Reply(r.request(job["id"], "checkpoint", reason="user")))
        if action == "cancel_job":
            job = self._job(args)
            if job["state"] != "queued":
                raise ActionError(f"{job['id']} is {job['state']}; only queued jobs can be cancelled (use /stop)")
            return Prepared(action, args, f"Cancel queued job {job['id']} ({job['template']})?", True,
                            lambda: Reply(r.cancel(job["id"])))
        if action == "set_priority":
            job = self._job(args)
            try:
                pr = int(args.get("priority"))
            except (TypeError, ValueError, OverflowError):
                raise ActionError("priority must be a whole number") from None
            if not -100 <= pr <= 100:
                raise ActionError("priority must be between -100 and 100")
            if job["state"] != "queued":
                raise ActionError(f"{job['id']} is {job['state']}; priority only matters while queued")
            return Prepared(action, args, f"Set priority of {job['id']} to {pr} (higher runs first)?", True,
                            lambda: Reply(r.set_priority(job["id"], pr)))
        if action in ("hold", "unhold"):
            def hold() -> Reply:
                self.db.kv_set("hold", "1" if action == "hold" else None)
                self.db.event(action)
                return Reply("⏸ Holding: running jobs continue, no new jobs start. /unhold to continue."
                             if action == "hold" else "▶ Launching queued jobs again.")
            return Prepared(action, args, "", False, hold)
        if action == "clear_gpu":
            try:
                gpu = int(str(args.get("gpu"))[:6])
            except (TypeError, ValueError):
                raise ActionError("give the GPU index, e.g. /clear 1") from None
            if gpu not in self.cfg.gpus.training:
                raise ActionError(f"GPU {gpu} is not a training GPU")
            st = self.db.gpu_state(gpu)
            if not st["quarantined"] and not st["cooling"]:
                raise ActionError(f"GPU {gpu} is not quarantined")

            def clear() -> Reply:
                s = rep.store.latest(gpu) or {}
                if s.get("ecc_uncorrected") is not None:
                    self.db.kv_set(f"ecc_baseline:{gpu}", str(s["ecc_uncorrected"]))
                self.db.set_gpu_state(gpu, quarantined=0, quarantine_reason=None, cooling=0)
                self.db.event("cleared", gpu=gpu)
                return Reply(f"✅ {rep.store.label(gpu)} (GPU {gpu}) is back in service.")
            label = rep.store.label(gpu)
            return Prepared(action, args, f"Lift the quarantine on {label} (GPU {gpu})?\n  Reason was: "
                            f"{st['quarantine_reason'] or 'cooling down'}", True, clear,
                            fingerprint=str(st["quarantine_reason"]))
        if action == "clarify":  # the model's own words: marked as such, and kept short
            q = model_words(args.get("question"), "Could you say that another way?")
            return simple(lambda: f"AI: {q}")
        if action == "unsupported":
            why = model_words(args.get("reason"), "that is not something I can do")
            return simple(lambda: f"AI: I can't do that ({why}). I can only run the actions in /help.")
        raise ActionError(f"unknown action {action}")

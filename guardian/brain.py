"""Agent brain (optional): a small quantized model served by llama.cpp's llama-server, usually on
a spare GPU (``gpus.agent``) so the training cards stay free.

It does two jobs only:
  1. Turn a plain-English message into exactly one action from a fixed menu, as JSON
     constrained by a schema. It never produces commands; the daemon validates every
     argument against the templates and the job list before anything happens.
  2. Reword computed run numbers into a two-sentence phone notification. Any number in the
     summary that is not in the facts gets the summary thrown away.

Turn it on with ``brain.enabled: true``. If the server does not answer within 30 seconds,
callers get ``BrainOffline`` and fall back to slash commands and plain template messages.
"""

from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger(__name__)

# action -> (arguments, needs a confirmation tap)
ACTION_MENU: dict[str, tuple[tuple[str, ...], bool]] = {
    "get_status": ((), False),
    "get_job": (("job_id",), False),
    "compare_jobs": (("job_ids",), False),
    "queue_run": (("template", "overrides"), True),
    "rerun": (("job_id", "overrides"), True),
    "pause_job": (("job_id",), True),
    "resume_job": (("job_id",), True),
    "stop_job": (("job_id",), True),
    "checkpoint_job": (("job_id",), False),
    "set_priority": (("job_id", "priority"), True),
    "clarify": (("question",), False),
    "unsupported": (("reason",), False),
}

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTION_MENU)},
        "args": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "job_ids": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 5},
                "template": {"type": "string"},
                "overrides": {"type": "object",
                              "additionalProperties": {"type": ["number", "string", "boolean"]}},
                "priority": {"type": "integer"},
                "question": {"type": "string"},
                "reason": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "reply": {"type": "string"},
    },
    "required": ["action", "args", "reply"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are the command parser for Rig Guardian, an agent that runs a GPU training queue.
Map the user's message to exactly ONE action from the menu and answer with JSON only:
{"action": ..., "args": {...}, "reply": "<one short sentence saying what you understood>"}

Rules:
- Use only job IDs from RECENT JOBS and template names from TEMPLATES. Never invent IDs.
- "last run", "that run", "it" refer to the most recent matching job in RECENT JOBS or the conversation.
- Overrides use the exact setting names listed for the template. Convert words to numbers
  ("half the learning rate" of 1e-3 is 5e-4). Only include settings the user asked to change.
- If the request is ambiguous, use "clarify" with a short question.
- Anything outside the menu (shell commands, files, deleting things, changing safety limits,
  editing templates, other topics) is "unsupported" with a short reason.

ACTIONS:
get_status {} - overview of GPUs and queue
get_job {job_id}
compare_jobs {job_ids: 2-5 IDs}
queue_run {template, overrides}
rerun {job_id, overrides} - copy of a past run with changes
pause_job {job_id} / resume_job {job_id} / stop_job {job_id}
checkpoint_job {job_id}
set_priority {job_id, priority: integer}
clarify {question}
unsupported {reason}"""


class BrainOffline(Exception):
    pass


def _scalar(v: Any) -> bool:
    if isinstance(v, bool) or isinstance(v, str):
        return True
    return isinstance(v, (int, float)) and math.isfinite(v)


def args_problem(args: dict) -> str | None:
    """Why the model's arguments do not have the types the menu allows (None if they do)."""
    for key in ("job_id", "template", "question", "reason"):
        if key in args and not isinstance(args[key], str):
            return f"{key} must be text"
    if "job_ids" in args and not (isinstance(args["job_ids"], list)
                                  and all(isinstance(x, str) for x in args["job_ids"])):
        return "job_ids must be a list of job IDs"
    if "priority" in args and (isinstance(args["priority"], bool) or not isinstance(args["priority"], int)):
        return "priority must be a whole number"
    if "overrides" in args:
        ov = args["overrides"]
        if not isinstance(ov, dict) or not all(isinstance(k, str) and _scalar(v) for k, v in ov.items()):
            return "overrides must be setting names with a number, text or true/false value each"
    return None


@dataclass
class BrainResult:
    action: str
    args: dict[str, Any] = field(default_factory=dict)
    reply: str = ""
    raw: str = ""


class Brain:
    def __init__(self, cfg, transport: httpx.BaseTransport | None = None):
        self.cfg = cfg
        # trust_env=False: a request to the local model must never be routed through HTTP(S)_PROXY
        self.client = httpx.Client(base_url=cfg.url.rstrip("/"), timeout=cfg.timeout_s, transport=transport,
                                   trust_env=False)

    def _chat(self, messages: list[dict], schema: dict | None, max_tokens: int, temperature: float) -> str:
        body: dict[str, Any] = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens,
                                "stream": False}
        if schema is not None:  # llama-server turns the schema into a grammar: output must match it
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "action", "schema": schema}}
        try:
            r = self.client.post("/v1/chat/completions", json=body)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError) as e:
            raise BrainOffline(f"{type(e).__name__}: {e}") from e
        if not isinstance(content, str):
            raise BrainOffline("the model server returned no text")
        return content

    def map_message(self, text: str, context: str, history: list[str],
                    previous: BrainResult | None = None, error: str | None = None) -> BrainResult:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + context}]
        if history:
            msgs.append({"role": "system", "content": "EARLIER MESSAGES (oldest first):\n"
                         + "\n".join(f"- {h}" for h in history)})
        msgs.append({"role": "user", "content": text})
        if previous is not None and error:
            msgs.append({"role": "assistant", "content": previous.raw})
            msgs.append({"role": "user", "content": f"That action was rejected: {error}. "
                         "Answer again with a corrected action, or clarify/unsupported."})
        raw = self._chat(msgs, RESPONSE_SCHEMA, self.cfg.max_tokens, 0.0)
        try:
            data = json.loads(raw)
        except ValueError:
            return BrainResult("invalid", {}, "", raw)
        if not isinstance(data, dict) or not isinstance(data.get("action"), str):
            return BrainResult("invalid", {}, "", raw)
        args = data.get("args") if isinstance(data.get("args"), dict) else {}
        problem = args_problem(args)  # the server's grammar should guarantee the shape; never rely on it
        if problem:
            return BrainResult("invalid", {}, problem, raw)
        return BrainResult(data["action"], args, str(data.get("reply", "")), raw)

    def summarize(self, facts: dict) -> str | None:
        """Two sentences for a phone notification; None if offline or if it made numbers up."""
        if not self.cfg.enabled:
            return None
        facts_json = json.dumps(facts, sort_keys=True)
        msgs = [
            {"role": "system", "content": "You write two short sentences for a phone notification about a "
             "finished training run. Use only the facts given. Do not calculate or add numbers that are not "
             "in the facts. No preamble."},
            {"role": "user", "content": facts_json},
        ]
        try:
            out = self._chat(msgs, None, 120, 0.3).strip()
        except BrainOffline:
            return None
        return out if out and numbers_grounded(out, facts_json) else None


_NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?")
_NUMBER_WORDS = re.compile(
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|(thir|four|fif|six|seven|eigh|nine)teen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|half|halved|twice|"
    r"double[ds]?|triple[ds]?|quadruple[ds]?|tenfold)\b", re.I)


def numbers_grounded(text: str, facts_json: str) -> bool:
    """A best-effort check that the model only reworded: every number in ``text`` must appear
    in the facts, and it may not smuggle numbers in as words, as non-ASCII digits, or as job
    IDs that are not in the facts. It cannot catch a wrong claim made without numbers; the
    report always prints the computed figures under the summary for that reason."""
    if _NUMBER_WORDS.search(text):
        return False
    if any(unicodedata.category(ch).startswith("N") and not ("0" <= ch <= "9") for ch in text):
        return False  # "Ⅻ", "⁹⁹", "٣"
    for jid in re.findall(r"\bj_?\d+\b", text, flags=re.I):
        if jid not in facts_json:
            return False
    allowed = {n.replace(",", "").lstrip("+") for n in _NUM.findall(facts_json)}
    allowed |= {str(float(a)) for a in allowed if _is_float(a)}
    for n in _NUM.findall(re.sub(r"\bj_?\d+\b", " ", text, flags=re.I)):
        v = n.replace(",", "").lstrip("+")
        if v in allowed or (_is_float(v) and str(float(v)) in allowed):
            continue
        return False
    return True


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False

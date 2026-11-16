"""Reports and screens. The daemon computes every number here; the language model only
rewords finished numbers into a short summary line, and never sees logs.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from guardian.checkpoints import CheckpointStore
from guardian.db import ACTIVE_STATES, parse_iso
from guardian.fsutil import read_json, tail_lines
from guardian.telemetry import THERMAL_THROTTLE_MASK

META_KEYS = ("ts", "step", "epoch", "samples_seen")


# ---- formatting --------------------------------------------------------------------------
def fmt_dur(s: float | None) -> str:
    if s is None:
        return "?"
    s = int(max(0, s))
    if s >= 3600:
        return f"{s // 3600} h {s % 3600 // 60} m"
    if s >= 60:
        return f"{s // 60} m"
    return f"{s} s"


def fmt_num(v: Any) -> str:
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
            return f"{v:.3g}"
        return f"{v:.4g}"
    return str(v)


def fmt_gb(b: float | None) -> str:
    return "?" if b is None else f"{b / 2**30:.1f}"


def fmt_q(v: Any) -> str:
    """A reading for display: '?' when the card does not report it."""
    return "?" if v is None else fmt_num(v)


def is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


# ---- reading run folders -----------------------------------------------------------------
def read_metrics(run_dir: str | Path) -> list[dict]:
    rows = []
    try:
        with open(Path(run_dir) / "metrics.jsonl") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:  # a half-written last line
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        pass
    return rows


def read_heartbeat(run_dir: str | Path) -> dict | None:
    return read_json(Path(run_dir) / "heartbeat")


def throughput(rows: list[dict], window_s: float = 600) -> float | None:
    """Samples per second over the last ``window_s`` of metrics (after any resume)."""
    pts = [(r["ts"], r["samples_seen"]) for r in rows if is_number(r.get("ts")) and is_number(r.get("samples_seen"))]
    if len(pts) < 2:
        return None
    # only the newest contiguous segment: a resume can roll samples back or leave a time gap
    start = 0
    for i in range(1, len(pts)):
        if pts[i][1] < pts[i - 1][1] or pts[i][0] - pts[i - 1][0] > 300:
            start = i
    seg = [p for p in pts[start:] if p[0] >= pts[-1][0] - window_s]
    if len(seg) < 2 or seg[-1][0] <= seg[0][0]:
        return None
    return (seg[-1][1] - seg[0][1]) / (seg[-1][0] - seg[0][0])


def best_of(rows: list[dict], metric: str, mode: str) -> dict | None:
    cands = [r for r in rows if is_number(r.get(metric))]  # a NaN or inf is never "the best"
    if not cands:
        return None
    return (min if mode == "min" else max)(cands, key=lambda r: r[metric])


def final_metrics(rows: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for r in rows:
        for k, v in r.items():
            if k not in META_KEYS:
                out[k] = v
    return out


class Reports:
    def __init__(self, cfg, db, store, templates: Callable[[], dict], brain=None,
                 clock: Callable[[], float] = time.time):
        self.cfg, self.db, self.store, self.templates, self.brain, self.clock = (
            cfg, db, store, templates, brain, clock)
        self.busy_reason: Callable[[int], str | None] | None = None  # set by the runner

    # ---- helpers ---------------------------------------------------------------------
    def run_dir(self, job: dict) -> Path:
        return Path(job["run_dir"]) if job.get("run_dir") else self.cfg.runs_dir / job["id"]

    def job_config(self, job: dict) -> dict:
        return json.loads(job.get("config_json") or "{}")

    def template_for(self, job: dict):
        return self.templates().get(job["template"])

    def gpu_list(self, job: dict) -> list[int]:
        return [int(x) for x in (job.get("gpus") or "").split(",") if x != ""]

    def gpus_label(self, job: dict) -> str:
        return " + ".join(self.store.label(g) for g in self.gpu_list(job)) or "-"

    def eta_seconds(self, job: dict) -> float | None:
        hb = read_heartbeat(self.run_dir(job)) or {}
        epochs = self.job_config(job).get("epochs")
        size = hb.get("epoch_size")
        if not epochs or not size:
            return None
        rate = throughput(read_metrics(self.run_dir(job)))
        if not rate:
            return None
        done = int(hb.get("epoch", 0)) * size + int(hb.get("samples_epoch", 0))
        return max(0.0, (int(epochs) * size - done) / rate)

    def estimate_duration(self, template: str, config: dict) -> float | None:
        """From the last completed run of the same template, scaled by epochs."""
        prev = self.db.one(
            "SELECT * FROM jobs WHERE template=? AND state='completed' AND run_seconds > 0 "
            "ORDER BY seq DESC LIMIT 1", (template,))
        if not prev:
            return None
        pcfg = self.job_config(prev)
        secs = float(prev["run_seconds"])
        if pcfg.get("epochs") and config.get("epochs"):
            secs *= float(config["epochs"]) / float(pcfg["epochs"])
        return secs

    def job_duration(self, job: dict) -> float:
        secs = float(job.get("run_seconds") or 0)
        if job["state"] in ACTIVE_STATES and job.get("launched_at"):
            secs += self.clock() - float(job["launched_at"])
        return secs

    def checkpoint_store(self, job: dict) -> CheckpointStore:
        return CheckpointStore(self.run_dir(job) / "checkpoints")

    # ---- screens ---------------------------------------------------------------------
    def gpu_line(self, gpu: int, detailed: bool = False) -> str:
        s = self.store.latest(gpu)
        st = self.db.gpu_state(gpu)
        label = self.store.label(gpu)
        if s is None:
            return f"{label} (GPU {gpu}) · no telemetry"
        if s.get("lost"):
            return f"{label} (GPU {gpu}) · LOST from the bus"
        parts = [f"{label} (GPU {gpu})",
                 f"{fmt_q(s.get('temp_c'))} °C" if s.get("temp_c") is not None else "no temperature reading",
                 f"{fmt_gb(s.get('mem_used'))}/{fmt_gb(s.get('mem_total'))} GiB",
                 f"{fmt_q(s.get('util'))}% busy"]
        if detailed:
            parts.append(f"{fmt_q(s.get('power_w'))}/{fmt_q(s.get('power_limit_w'))} W")
            if s.get("t_slow"):
                parts.append(f"slowdown {fmt_num(s.get('t_slow'))} °C")
            if s.get("throttle") is not None:  # only NVIDIA reports why a card slows down
                thr = int(s["throttle"])
                parts.append("thermal throttling" if thr & THERMAL_THROTTLE_MASK
                             else ("power-capped" if thr & 0x4 else "no throttling"))
            if s.get("fan") is not None:
                parts.append(f"fan {s['fan']}%")
            if s.get("ecc_uncorrected"):
                parts.append(f"ECC {s['ecc_uncorrected']}")
        if gpu == self.cfg.gpus.agent:
            parts.append("language model")
        elif gpu not in self.cfg.gpus.training:
            parts.append("monitor only")
        else:
            jobs = [j for j in self.db.jobs_in(ACTIVE_STATES) if gpu in self.gpu_list(j)]
            if jobs:
                j = jobs[0]
                hb = read_heartbeat(self.run_dir(j)) or {}
                parts.append(f"{j['id']} {j['state']}, step {fmt_num(hb.get('step', 0))}")
            elif st["quarantined"] or st["cooling"]:
                pass  # said below
            else:  # "free" only when a job could really start on it
                why = self.busy_reason(gpu) if self.busy_reason else None
                parts.append(f"not available: {why}" if why else "free")
        if st["quarantined"]:
            parts.append(f"QUARANTINED ({st['quarantine_reason']}; /clear {gpu})")
        elif st["cooling"]:
            parts.append("cooling down")
        return " · ".join(parts)

    def reading_age(self) -> float | None:
        """Seconds since the newest GPU reading (None if there is none)."""
        ts = [s["ts"] for s in (self.store.latest(g) for g in self.store.gpus())
              if s and isinstance(s.get("ts"), (int, float))]
        return max(0.0, self.clock() - max(ts)) if ts else None

    def all_gpus(self) -> list[int]:
        """Every detected GPU, plus any configured one that stopped reporting."""
        g = set(self.cfg.gpus.training) | set(self.store.gpus())
        if self.cfg.gpus.agent is not None:
            g.add(self.cfg.gpus.agent)
        return sorted(g)

    def status_text(self) -> str:
        q = len(self.db.queued_jobs())
        hold = []
        if self.db.kv_get("hold") == "1":
            hold.append("on hold (/unhold)")
        if self.db.kv_get("disk_hold") == "1":
            hold.append("paused: disk low")
        try:
            settle = float(self.db.kv_get("settle_until") or 0)
        except ValueError:
            settle = 0
        if q and settle > self.clock():
            hold.append("launching resumes at " + time.strftime("%H:%M", time.localtime(settle))
                        + " (settling after a restart)")
        n, k = len(self.all_gpus()), len(self.cfg.gpus.training)
        head = (f"🖥 Status · {n} GPU{'s' if n != 1 else ''} ({k} for training) · {q} queued"
                + (" · " + " · ".join(hold) if hold else ""))
        lines = [head] + [self.gpu_line(g) for g in self.all_gpus()]
        stale = self.reading_age()
        if stale is not None and stale > 60:
            lines.append(f"⚠️ These readings are {fmt_dur(stale)} old: the agent is not running "
                         f"(`rig-guardian start`).")
        waiting = self.waiting_for_user()
        if waiting:
            lines.append("Waiting for you: " + "; ".join(waiting))
        return "\n".join(lines)

    def gpus_text(self) -> str:
        lines = ["🌡 GPUs"]
        for g in self.all_gpus():
            lines.append(self.gpu_line(g, detailed=True))
            s = self.store.latest(g) or {}
            if s.get("name"):
                extra = [s["name"]] + ([f"compute {s['cc']}"] if s.get("cc") else []) + (
                    [s["pci"]] if s.get("pci") else []) + (["integrated"] if s.get("integrated") else [])
                lines.append("    " + " · ".join(extra))
        return "\n".join(lines)

    def jobs_text(self) -> str:
        active = self.db.jobs_in(ACTIVE_STATES)
        queued = self.db.queued_jobs()
        done = self.db.all("SELECT * FROM jobs WHERE state NOT IN ('queued','preflight','running','pausing','paused')"
                           " ORDER BY seq DESC LIMIT 5")
        lines = ["🏃 Running"] + [self._job_row(j) for j in active] if active else ["🏃 Nothing running"]
        if queued:
            lines += ["", "⏳ Queued"] + [self._job_row(j) for j in queued]
        if done:
            lines += ["", "🏁 Recent"] + [self._job_row(j) for j in done]
        return "\n".join(lines)

    def _job_row(self, j: dict) -> str:
        bits = [j["id"], j["template"], j["state"]]
        if j["state"] in ACTIVE_STATES:
            hb = read_heartbeat(self.run_dir(j)) or {}
            bits.append(f"step {fmt_num(hb.get('step', 0))}")
            bits.append(self.gpus_label(j))
            eta = self.eta_seconds(j)
            if eta is not None:
                bits.append(f"ETA {fmt_dur(eta)}")
        elif j["state"] == "queued" and j.get("priority"):
            bits.append(f"priority {j['priority']}")
        else:
            t = self.template_for(j)
            if t:
                b = best_of(read_metrics(self.run_dir(j)), t.best_metric, t.mode)
                if b:
                    bits.append(f"{t.best_metric} {fmt_num(b[t.best_metric])}")
        return " · ".join(bits)

    def job_text(self, job: dict) -> str:
        t = self.template_for(job)
        cfgd = self.job_config(job)
        rows = read_metrics(self.run_dir(job))
        hb = read_heartbeat(self.run_dir(job)) or {}
        lines = [f"{job['id']} · {job['template']} · {job['state']}"]
        if t:
            changed = {k: v for k, v in cfgd.items() if t.defaults.get(k) != v}
            lines.append("Settings: " + (", ".join(f"{k}={fmt_num(v)}" for k, v in changed.items()) or "defaults"))
        if job.get("micro_batch") and cfgd.get("micro_batch") != job["micro_batch"]:
            lines.append(f"Micro-batch now {job['micro_batch']} × {job['grad_accum']} accumulation (after OOM)")
        if job.get("parent_job"):
            lines.append(f"Rerun of {job['parent_job']}")
        if job.get("gpus"):
            lines.append(f"GPU: {self.gpus_label(job)}")
        lines.append(f"Step {fmt_num(hb.get('step', 0))} · epoch {hb.get('epoch', 0)}"
                     f" · run time {fmt_dur(self.job_duration(job))}")
        fm = final_metrics(rows)
        if fm:
            lines.append("Latest: " + ", ".join(f"{k} {fmt_num(v)}" for k, v in fm.items()))
        if t:
            b = best_of(rows, t.best_metric, t.mode)
            if b:
                lines.append(f"Best {t.best_metric} {fmt_num(b[t.best_metric])} (epoch {b.get('epoch')})")
        if job["state"] in ACTIVE_STATES:
            eta = self.eta_seconds(job)
            if eta is not None:
                lines.append(f"ETA {fmt_dur(eta)}")
        mans = self.checkpoint_store(job).manifests()
        if mans:
            scored = [m for m in mans if m.get("metric") is not None]
            best = (min if (t.mode if t else "min") == "min" else max)(
                scored, key=lambda m: m["metric"]) if scored else None
            lines.append("Checkpoints: " + ", ".join(
                f"{m['step']}" + (" final" if m.get("final") else "") + (" best" if m is best else "")
                for m in mans[:6]))
        if job.get("pause_detail") and job["state"] in ("paused", "pausing", "stopped"):
            lines.append(f"Note: {job['pause_detail']}")
        if job.get("note"):
            lines.append(f"Note: {job['note']}")
        return "\n".join(lines)

    def logs_text(self, job: dict, n: int = 30) -> str:
        lines = tail_lines(self.run_dir(job) / "stdout.log", n)
        return "\n".join(lines) if lines else "(no output yet)"

    def plot(self, job: dict) -> str | None:
        rows = read_metrics(self.run_dir(job))
        if not rows:
            return None
        import matplotlib

        matplotlib.use("Agg")  # CPU only; nothing touches the GPUs
        import matplotlib.pyplot as plt

        rows = [r for r in rows if is_number(r.get("step"))]
        tr = [(r["step"], r["loss"]) for r in rows if is_number(r.get("loss"))]
        vals = {key: [(r["step"], r[key]) for r in rows if is_number(r.get(key))]
                for key in sorted({k for r in rows for k in r if str(k).startswith("val")})}
        if not tr and not any(vals.values()):
            return None  # nothing numeric to draw
        fig, ax = plt.subplots(figsize=(7, 4), dpi=110)
        if tr:
            ax.plot(*zip(*tr), label="loss", linewidth=1, alpha=0.8)
        for key, pts in vals.items():
            if pts:
                ax.plot(*zip(*pts), marker="o", label=key, linewidth=1.5)
        ax.set_xlabel("optimizer step")
        ax.set_title(f"{job['id']} · {job['template']}")
        ax.grid(alpha=0.3)
        ax.legend()
        out = self.run_dir(job) / "loss.png"
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        return str(out)

    def compare_text(self, jobs: list[dict]) -> str:
        cfgs = {j["id"]: self.job_config(j) for j in jobs}
        keys = sorted({k for c in cfgs.values() for k in c})
        differing = [k for k in keys if len({json.dumps(c.get(k)) for c in cfgs.values()}) > 1]
        lines = ["⚖️ Compare " + " vs ".join(j["id"] for j in jobs)]
        if len({j["template"] for j in jobs}) > 1:
            lines.append("Note: different templates; metrics may not be comparable.")
        for j in jobs:
            t = self.template_for(j)
            rows = read_metrics(self.run_dir(j))
            bits = [j["id"], j["state"]]
            if t:
                b = best_of(rows, t.best_metric, t.mode)
                if b:
                    bits.append(f"best {t.best_metric} {fmt_num(b[t.best_metric])}")
            bits.append(fmt_dur(self.job_duration(j)))
            bits += [f"{k}={fmt_num(cfgs[j['id']].get(k))}" for k in differing]
            lines.append(" · ".join(bits))
        return "\n".join(lines)

    # ---- run report --------------------------------------------------------------------
    def job_facts(self, job: dict) -> dict:
        t = self.template_for(job)
        rows = read_metrics(self.run_dir(job))
        facts: dict[str, Any] = {"job": job["id"], "template": job["template"], "state": job["state"],
                                 "duration": fmt_dur(self.job_duration(job))}
        if t:
            b = best_of(rows, t.best_metric, t.mode)
            if b:
                facts["best"] = {t.best_metric: round(b[t.best_metric], 4), "epoch": b.get("epoch")}
        # numbers only: text a job logs as a "metric" must never reach the model's prompt
        fm = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in final_metrics(rows).items()
              if is_number(v) and (k.startswith("val") or k in ("loss", "acc", "accuracy"))}
        if fm:
            facts["final"] = fm
        if job.get("parent_job") and t:
            parent = self.db.get_job(job["parent_job"])
            if parent and parent["template"] == job["template"]:
                pb = best_of(read_metrics(self.run_dir(parent)), t.best_metric, t.mode)
                if pb and "best" in facts:
                    facts["vs_parent"] = {
                        "parent": parent["id"],
                        "delta_" + t.best_metric: round(facts["best"][t.best_metric] - pb[t.best_metric], 4),
                        "changed": [f"{k}={fmt_num(v)}" for k, v in self.job_config(job).items()
                                    if self.job_config(parent).get(k) != v],
                    }
        peak, throttled = self._gpu_history(job)
        if peak is not None:
            facts["gpu_peak_c"] = round(peak)
        facts["thermal_throttle_s"] = throttled
        facts["oom_retries"] = int(job.get("oom_retries") or 0)
        return facts

    def _gpu_history(self, job: dict) -> tuple[float | None, int]:
        gpus = self.gpu_list(job)
        start = parse_iso(job.get("started_at"))
        end = parse_iso(job.get("ended_at")) or self.clock()
        if not gpus or start is None:
            return None, 0
        marks = ",".join("?" * len(gpus))
        r = self.db.one(f"SELECT MAX(temp_max_c) AS peak FROM telemetry WHERE gpu IN ({marks}) AND ts BETWEEN ? AND ?",
                        (*gpus, start, end))
        n = self.db.one(f"SELECT COUNT(*) AS n FROM telemetry WHERE gpu IN ({marks}) AND ts BETWEEN ? AND ? "
                        f"AND (throttle & ?) != 0", (*gpus, start, end, THERMAL_THROTTLE_MASK))
        return (r or {}).get("peak"), int((n or {}).get("n", 0) * self.cfg.telemetry.db_every_s)

    def run_report(self, job: dict) -> tuple[str, bool]:
        """(text, urgent) for a job that completed, failed, or gave up after retries."""
        facts = self.job_facts(job)
        t = self.template_for(job)
        ok = job["state"] == "completed"
        icon = {"completed": "✅", "failed": "❌", "oom": "🧠", "stalled": "⏳"}.get(job["state"], "⏹")
        verb = {"completed": "finished", "failed": f"failed (exit {job.get('exit_code')})",
                "oom": "ran out of memory and gave up", "stalled": "stalled"}.get(job["state"], job["state"])
        lines = [f"{icon} {job['id']} {job['template']} {verb} ({facts['duration']})"]
        try:
            summary = self.brain.summarize(facts) if self.brain is not None else None
        except Exception:  # the optional model must never cost a report
            summary = None
        if summary:
            lines.append("AI summary: " + " ".join(summary.split())[:400])
        if "best" in facts and t:
            best = facts["best"]
            extra = [f"{k} {fmt_num(v)}" for k, v in facts.get("final", {}).items() if k != t.best_metric]
            lines.append(f"{t.best_metric} {fmt_num(best[t.best_metric])} (best, epoch {best['epoch']})"
                         + ("" if not extra else " · " + " · ".join(extra)))
        vp = facts.get("vs_parent")
        if vp and t:
            d = vp["delta_" + t.best_metric]
            lines.append(f"vs. {vp['parent']}: {t.best_metric} {d:+.4g}"
                         + (f" ({', '.join(vp['changed'])})" if vp["changed"] else ""))
        hw = []
        if "gpu_peak_c" in facts:
            hw.append(f"GPU peak {facts['gpu_peak_c']} °C")
        if facts["thermal_throttle_s"]:
            hw.append(f"throttled {fmt_dur(facts['thermal_throttle_s'])}")
        elif all((self.store.latest(g) or {}).get("throttle") is not None for g in self.gpu_list(job)):
            hw.append("no throttling")  # only claimed for cards that report throttling at all
        if facts["oom_retries"]:
            cfgd = self.job_config(job)
            hw.append(f"{facts['oom_retries']} OOM retr{'y' if facts['oom_retries'] == 1 else 'ies'} "
                      f"(micro_batch {cfgd.get('micro_batch')}→{job.get('micro_batch')})")
        lines.append(" · ".join(hw))
        mans = self.checkpoint_store(job).manifests()
        kinds = [k for k, has in (("final", any(m.get("final") for m in mans)),
                                  ("best", any(m.get("is_best") for m in mans))) if has]
        if mans:
            lines.append(f"Checkpoint: {' + '.join(kinds) if kinds else 'step ' + str(mans[0]['step'])} saved")
        if ok:
            lines.append("Reply to change settings and run again.")
        else:
            lines.append("")
            lines.append("Last 30 log lines:")
            lines.append(self.logs_text(job, 30))
        return "\n".join(lines), not ok

    # ---- digest --------------------------------------------------------------------------
    def waiting_for_user(self) -> list[str]:
        out = [f"{j['id']} ({j['state']})" for j in self.db.all(
            "SELECT * FROM jobs WHERE waiting_for_user=1 AND state NOT IN ('completed','cancelled') ORDER BY seq")]
        for g in self.cfg.gpus.training:
            st = self.db.gpu_state(g)
            if st["quarantined"]:
                out.append(f"{self.store.label(g)} quarantined: {st['quarantine_reason']} (/clear {g})")
        return out

    def digest_text(self) -> str:
        since = self.clock() - 86400
        lines = ["📋 Daily digest"]
        finished = [j for j in self.db.all("SELECT * FROM jobs WHERE ended_at IS NOT NULL ORDER BY seq")
                    if (parse_iso(j["ended_at"]) or 0) >= since]
        if finished:
            lines.append("Finished:")
            for j in finished:
                t = self.template_for(j)
                b = best_of(read_metrics(self.run_dir(j)), t.best_metric, t.mode) if t else None
                lines.append(f"• {j['id']} {j['template']} {j['state']}"
                             + (f" · {t.best_metric} {fmt_num(b[t.best_metric])}" if b else ""))
        else:
            lines.append("No runs finished in the last 24 h.")
        gpu_s = 0.0
        for j in self.db.all("SELECT * FROM jobs WHERE run_seconds > 0 OR launched_at IS NOT NULL"):
            gpu_s += self.job_duration(j) * max(1, len(self.gpu_list(j))) if (
                (parse_iso(j.get("ended_at")) or self.clock()) >= since) else 0
        lines.append(f"GPU hours: {gpu_s / 3600:.1f}")
        hot = []
        for g in self.all_gpus():
            r = self.db.one("SELECT MAX(temp_max_c) AS t FROM telemetry WHERE gpu=? AND ts>=?", (g, since))
            if r and r["t"] is not None:
                hot.append(f"{self.store.label(g)} {r['t']:.0f} °C")
        if hot:
            lines.append("Hottest: " + " · ".join(hot))
        free = shutil.disk_usage(self.cfg.runs_dir if self.cfg.runs_dir.exists() else self.cfg.root).free
        lines.append(f"Disk free: {free / 1e9:.0f} GB")
        waiting = self.waiting_for_user()
        lines.append("Waiting for you: " + ("; ".join(waiting) if waiting else "nothing"))
        return "\n".join(lines)

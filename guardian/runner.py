"""Job queue and runner: validated templates in, supervised processes out.

Every run is a row in SQLite that moves through a fixed set of states. Jobs are launched
from approved templates only; the runner never builds a command from free text.

Jobs go to free training GPUs that suit them (memory, card model), coolest first; a job
asking for N GPUs runs under torchrun with N processes. Each job runs in its own systemd user unit (``systemd-run --user --unit=rg-job-<id>``),
so restarting the daemon never kills training. Where systemd is unavailable (a dev
machine), the local launcher starts the job in its own session instead.

All control state lives in the database and the run folder, so the CLI can queue, pause
and stop jobs from a separate process while the daemon does the launching.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

import yaml

from guardian import EXIT_OOM, EXIT_STOPPED
from guardian.checkpoints import CheckpointStore
from guardian.db import ACTIVE_STATES, now_iso
from guardian.fsutil import atomic_write_bytes, atomic_write_json, atomic_write_text, read_json
from guardian.reports import fmt_dur, fmt_num, read_heartbeat
from guardian.templates import TemplateError, config_hash, diff_settings, load_templates, merge_overrides
from guardian.vendors import VENDORS, pin_env, vendor_label

log = logging.getLogger(__name__)

# The wrapper runs the job, passes TERM/INT/HUP (stop) and USR1 (checkpoint now) on to it, and
# records its exit code for the runner. Under systemd the wrapper is the unit's main process:
# with KillMode=mixed only it receives the stop signal, so it has to hand it to the trainer,
# or a `systemctl stop`, a logout or a shutdown would end in SIGKILL with no final checkpoint.
WRAPPER = (
    '"$@" >> "$GUARDIAN_RUN_DIR/stdout.log" 2>&1 & child=$!; '
    'trap \'kill -TERM "$child" 2>/dev/null\' TERM INT HUP; '
    'trap \'kill -USR1 "$child" 2>/dev/null\' USR1; '
    'while :; do wait "$child"; rc=$?; kill -0 "$child" 2>/dev/null || break; done; '
    'echo "$rc" > "$GUARDIAN_RUN_DIR/exit_code.tmp" && '
    'mv "$GUARDIAN_RUN_DIR/exit_code.tmp" "$GUARDIAN_RUN_DIR/exit_code"'
)
MAX_INTERRUPTS = 3  # automatic restarts after a stop signal the agent did not send


def prepare_sdk_dir(root: Path) -> Path:
    """Copy the SDK and the two stdlib-only modules it needs into ``<root>/.sdk``.

    Only this folder goes on a job's PYTHONPATH, so the training environment sees
    ``guardian_sdk`` and nothing else from the agent's own environment.
    """
    d = root / ".sdk"
    (d / "guardian").mkdir(parents=True, exist_ok=True)
    pkg = Path(__file__).resolve().parent
    files = {pkg / "guardian_sdk.py": d / "guardian_sdk.py"}
    files.update({pkg / n: d / "guardian" / n for n in ("__init__.py", "fsutil.py", "checkpoints.py")})
    for src, dst in files.items():
        data = src.read_bytes()
        if not dst.exists() or dst.read_bytes() != data:
            atomic_write_bytes(dst, data)
    return d


class ActionError(Exception):
    """A request that cannot be carried out; the message is shown to the user."""


class LaunchError(Exception):
    pass


# ---- processes ---------------------------------------------------------------------------
def _proc_stat(pid: int) -> list[str] | None:
    try:
        data = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return data[data.rindex(")") + 2:].split()  # fields from #3 (state) on


def proc_start_time(pid: int | None) -> int | None:
    """Process start time (clock ticks since boot): guards against PID reuse after a reboot."""
    if not pid:
        return None
    st = _proc_stat(pid)
    return int(st[19]) if st else None


def pid_alive(pid: int | None, start: int | None = None) -> bool:
    if not pid:
        return False
    try:  # reap it if it is our own exited child (a zombie still answers kill(pid, 0))
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    if Path("/proc/self/stat").exists():
        st = _proc_stat(pid)
        if st is None or st[0] in ("Z", "X"):
            return False
        if start is not None and int(st[19]) != int(start):
            return False
    return True


class LocalLauncher:
    """Starts the job in its own session. The job survives a daemon restart, but a crash of
    the whole service cgroup can take it down; prefer systemd on the rig."""

    name = "local"

    def __init__(self):
        self.procs: dict[int, subprocess.Popen] = {}

    def launch(self, job_id: str, argv: list[str], env: dict[str, str], cwd: str) -> tuple[int, int, str | None]:
        # exactly the environment the runner built: the agent's own variables (the bot token
        # among them) are never handed to training code
        try:
            p = subprocess.Popen(["sh", "-c", WRAPPER, "rg-job", *argv], cwd=cwd, env=dict(env),
                                 start_new_session=True, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as e:
            raise LaunchError(str(e)) from e
        self.procs[p.pid] = p
        return p.pid, p.pid, None

    def reap(self) -> None:
        for pid, p in list(self.procs.items()):
            if p.poll() is not None:
                del self.procs[pid]


class SystemdLauncher:
    """``systemd-run --user --unit=rg-job-<id>-<home> --collect --property=KillMode=mixed``."""

    name = "systemd"

    def __init__(self, stop_timeout_s: float = 180.0, tag: str = ""):
        self.stop_timeout_s, self.tag = stop_timeout_s, tag

    @staticmethod
    def available() -> bool:
        if not shutil.which("systemd-run"):
            return False
        r = subprocess.run(["systemctl", "--user", "show-environment"], capture_output=True)
        return r.returncode == 0

    def unit_name(self, job_id: str) -> str:
        # the tag keeps two rig folders of the same user from sharing unit names
        return f"rg-job-{job_id}" + (f"-{self.tag}" if self.tag else "")

    def launch(self, job_id: str, argv: list[str], env: dict[str, str], cwd: str) -> tuple[int, int, str | None]:
        unit = self.unit_name(job_id)
        subprocess.run(["systemctl", "--user", "reset-failed", f"{unit}.service"], capture_output=True)
        cmd = ["systemd-run", "--user", f"--unit={unit}", "--collect", "--quiet",
               "--property=KillMode=mixed", f"--property=TimeoutStopSec={int(self.stop_timeout_s)}",
               f"--working-directory={cwd}"]
        cmd += [f"--setenv={k}={v}" for k, v in env.items()]
        # systemd expands $VAR and ${VAR} in command words; "$$" is its literal dollar sign, so
        # every word reaches the job exactly as written (the same as with the local launcher)
        cmd += [w.replace("$", "$$") for w in ("sh", "-c", WRAPPER, "rg-job", *argv)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            raise LaunchError(r.stderr.strip() or f"systemd-run exited {r.returncode}")
        pid = 0
        for _ in range(50):
            out = subprocess.run(["systemctl", "--user", "show", "-p", "MainPID", "--value", unit],
                                 capture_output=True, text=True).stdout.strip()
            pid = int(out or 0) if out.isdigit() else 0
            if pid:
                break
            time.sleep(0.1)
        if not pid:
            return 0, 0, unit  # it ended before we could look: the runner reads its exit code
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            pgid = pid
        return pid, pgid, unit

    def reap(self) -> None:
        pass


def pick_launcher(kind: str, stop_timeout_s: float = 180.0, tag: str = ""):
    if kind == "systemd" or (kind == "auto" and SystemdLauncher.available()):
        return SystemdLauncher(stop_timeout_s, tag)
    return LocalLauncher()


def job_survivors(job_id: str, run_dir: str, pgid: int | None) -> list[int]:
    """Processes of a job that outlived its wrapper (data-loader workers, or the trainer itself
    when only the wrapper was killed). A process counts only if it is in the job's process
    group and carries the job's own environment, so a reused PID can never be hit."""
    if not pgid or not Path("/proc/self/stat").exists():
        return []
    want = {f"GUARDIAN_JOB_ID={job_id}".encode(), f"GUARDIAN_RUN_DIR={run_dir}".encode()}
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        st = _proc_stat(int(d.name))
        if st is None or st[0] in ("Z", "X") or int(st[2]) != int(pgid):
            continue
        try:
            env = set((d / "environ").read_bytes().split(b"\0"))
        except OSError:
            continue
        if want <= env:
            out.append(int(d.name))
    return out



# ---- runner ------------------------------------------------------------------------------
class Runner:
    def __init__(self, cfg, db, store, notifier, reports, clock: Callable[[], float] = time.time,
                 launcher=None, lock: threading.RLock | None = None):
        self.cfg, self.db, self.store, self.notifier, self.reports = cfg, db, store, notifier, reports
        self.clock = clock
        self._launcher = launcher
        self.lock = lock or threading.RLock()
        reports.busy_reason = self.busy_reason  # so status screens can say why a card is not free
        self.settle_until = 0.0
        self._ckpt_scan: dict[str, float] = {}
        self._reports_due: list[str] = []
        # cards the watchdog saw in emergency while this lock was busy: nothing starts on them
        self.hot_gpus: set[int] = set()
        self.template_problems: list[tuple[str, str]] = []
        self.announce_template_problems = True  # the CLI prints them instead

    @property
    def launcher(self):
        if self._launcher is None:
            import hashlib

            tag = hashlib.sha1(str(self.cfg.root.resolve()).encode()).hexdigest()[:6]
            self._launcher = pick_launcher(self.cfg.runner.launcher, self.cfg.runner.stop_timeout_s, tag)
        return self._launcher

    def templates(self) -> dict:
        """Valid templates by name. A file that cannot be used is skipped and reported (once an
        hour while it stays broken); it never stops the scheduler or the watchdog."""
        problems: list[tuple[str, str]] = []
        tpls = load_templates(self.cfg.templates_dir, problems)
        self.template_problems = problems
        if self.announce_template_problems:
            for name, why in problems:
                self.notifier.warn(f"Template file {name} is being ignored: {why}", key=f"template:{name}")
        return tpls

    def run_dir(self, job: dict) -> Path:
        return Path(job["run_dir"]) if job.get("run_dir") else self.cfg.runs_dir / job["id"]

    def gpu_list(self, job: dict) -> list[int]:
        return [int(x) for x in (job.get("gpus") or "").split(",") if x != ""]

    def get(self, job_id: str) -> dict:
        job = self.db.get_job(job_id)
        if not job:
            raise ActionError(f"There is no job {job_id}.")
        return job

    # ---- queueing --------------------------------------------------------------------
    def unsuitable(self, t, gpu: int) -> str | None:
        """Why template ``t`` cannot use this card (None if it can). Unknown memory is allowed."""
        s = self.store.latest(gpu) or {}
        name = s.get("name") or ""
        vendor = s.get("vendor") or "nvidia"
        if t.vendor and vendor != t.vendor:
            return f"is an {vendor_label(vendor)} card; the template is for {vendor_label(t.vendor)}"
        if t.gpus > 1 and not VENDORS.get(vendor, {}).get("multi_gpu", True):
            return f"{vendor_label(vendor)} has one GPU per machine; the template asks for {t.gpus}"
        if t.gpu_match and t.gpu_match.lower() not in name.lower():
            return f"not a {t.gpu_match}"
        mem = s.get("mem_total")
        if mem is None:
            return None
        if t.min_gpu_mem_gb and mem < t.min_gpu_mem_gb * 2**30:
            return f"{mem / 2**30:.0f} GiB is under the template's {t.min_gpu_mem_gb:g} GiB minimum"
        if t.params_millions:  # weights + gradients + Adam state ≈ 16 bytes per parameter
            need = t.params_millions * 1e6 * self.cfg.memory.bytes_per_param
            if need > self.cfg.memory.static_limit_frac * mem:
                return (f"{fmt_num(t.params_millions)} M parameters × {self.cfg.memory.bytes_per_param} bytes ≈ "
                        f"{need / 2**30:.1f} GiB is more than {self.cfg.memory.static_limit_frac:.0%} of its "
                        f"{mem / 2**30:.0f} GiB, before activations")
        return None

    def eligible_gpus(self, t) -> list[int]:
        return [g for g in self.cfg.gpus.training if self.unsuitable(t, g) is None]

    def vendor_of(self, gpu: int) -> str:
        return (self.store.latest(gpu) or {}).get("vendor") or "nvidia"

    def same_vendor_pick(self, gpus: list[int], need: int) -> list[int] | None:
        """The first ``need`` cards of one vendor, keeping the given (coolest-first) order.

        One job never spans vendors: a PyTorch build drives one vendor's cards.
        """
        def grouped(key) -> list[int] | None:
            groups: dict = {}
            for g in gpus:
                groups.setdefault(key(g), []).append(g)
            for cards in groups.values():
                if len(cards) >= need:
                    return cards[:need]
            return None

        if need > 1:  # identical cards first: in a mixed group the slowest card sets the pace
            same_model = grouped(lambda g: (self.vendor_of(g), (self.store.latest(g) or {}).get("name"),
                                            (self.store.latest(g) or {}).get("mem_total")))
            if same_model is not None:
                return same_model
        return grouped(self.vendor_of)

    def static_check(self, t) -> None:
        """Refuse a template that could never run on this rig, saying why for each card."""
        elig = self.eligible_gpus(t)
        if self.same_vendor_pick(elig, t.gpus) is not None:
            return
        training = self.cfg.gpus.training
        why = "; ".join(f"GPU {g} ({self.store.label(g)}): {self.unsuitable(t, g)}"
                        for g in training if self.unsuitable(t, g))
        vendors = sorted({self.vendor_of(g) for g in elig})
        if len(elig) >= t.gpus and len(vendors) > 1:
            why = (why + "; " if why else "") + (
                "the cards that qualify are from different vendors (" + ", ".join(vendor_label(v) for v in vendors)
                + "), and one job cannot span vendors")
        raise TemplateError(
            f"Rejected: {t.name} needs {t.gpus} GPU{'s' if t.gpus > 1 else ''}, but only {len(elig)} of this rig's "
            f"{len(training)} training GPUs qualify" + (f". {why}." if why else "."))

    def prepare_run(self, template: str | None, overrides: dict[str, Any], parent_id: str | None = None) -> dict:
        tpls = self.templates()
        base = None
        if parent_id:
            parent = self.get(parent_id)
            template = parent["template"]
            base = json.loads(parent["config_json"] or "{}")
        t = tpls.get(template or "")
        if t is None:
            raise ActionError(f"No template named {template!r}. Templates: {', '.join(tpls) or 'none'}.")
        merged = merge_overrides(t, overrides, base)
        self.static_check(t)
        return {
            "template": t.name,
            "config": merged,
            "parent": parent_id,
            "diffs": diff_settings(base if base is not None else t.defaults, merged),
            "est": self.reports.estimate_duration(t.name, merged),
            "free": self.free_gpus(t),
            "n_gpus": t.gpus,
            "sdk": t.sdk, "best_metric": t.best_metric, "mode": t.mode,
        }

    def enqueue(self, prep: dict, priority: int = 0, user: int | None = None) -> str:
        cfgd = prep["config"]
        mb = cfgd.get("micro_batch")
        taken = 0  # run folders outlive a deleted database: a new job never takes an old folder's ID
        try:
            for d in self.cfg.runs_dir.iterdir():
                m = re.fullmatch(r"j_(\d+)", d.name)
                if m:
                    taken = max(taken, int(m.group(1)))
        except OSError:
            pass
        job_id = self.db.insert_job(
            _min_seq=taken, template=prep["template"], config_json=json.dumps(cfgd, sort_keys=True),
            config_hash=config_hash(cfgd), state="queued", priority=int(priority),
            parent_job=prep.get("parent"), n_gpus=prep.get("n_gpus", 1),
            micro_batch=mb, grad_accum=int(cfgd.get("grad_accum", 1)) if mb is not None else None,
            created_by=user, sdk=int(bool(prep.get("sdk", True))), best_metric=prep.get("best_metric"),
            mode=prep.get("mode"))
        rd = self.cfg.runs_dir / job_id
        rd.mkdir(parents=True, exist_ok=True)
        self.db.update_job(job_id, run_dir=str(rd))
        self._write_config(self.db.get_job(job_id))
        self.db.event("queued", json.dumps(cfgd, sort_keys=True), job_id=job_id)
        return job_id

    def _write_config(self, job: dict) -> dict:
        """The exact settings for this launch (micro-batch may differ after an OOM)."""
        cfgd = json.loads(job["config_json"] or "{}")
        if job.get("micro_batch") is not None:
            cfgd["micro_batch"] = int(job["micro_batch"])
            cfgd["grad_accum"] = int(job["grad_accum"] or 1)
        rd = self.run_dir(job)
        rd.mkdir(parents=True, exist_ok=True)
        atomic_write_text(rd / "config.yaml", yaml.safe_dump(cfgd, sort_keys=True))
        atomic_write_json(rd / "config.json", cfgd)
        return cfgd

    def cancel(self, job_id: str) -> str:
        with self.lock:
            job = self.get(job_id)
            if job["state"] != "queued":
                raise ActionError(f"{job_id} is {job['state']}; only queued jobs can be cancelled (use /stop).")
            self.db.update_job(job_id, state="cancelled", ended_at=now_iso())
            self.db.event("cancelled", job_id=job_id)
            return f"🗑 Cancelled {job_id}."

    def set_priority(self, job_id: str, priority: int) -> str:
        with self.lock:
            job = self.get(job_id)
            if job["state"] != "queued":
                raise ActionError(f"{job_id} is {job['state']}; priority only matters while queued.")
            self.db.update_job(job_id, priority=int(priority))
            return f"Priority of {job_id} set to {priority}."

    # ---- GPUs ------------------------------------------------------------------------
    def busy_reason(self, gpu: int, jobs: list[dict] | None = None) -> str | None:
        """Why a training card cannot take a job right now (None: it is free)."""
        for j in (self.db.jobs_in(ACTIVE_STATES) if jobs is None else jobs):
            if gpu in self.gpu_list(j):
                return f"running {j['id']}"
        if gpu == self.cfg.gpus.agent:
            return "language-model card"
        if gpu in self.hot_gpus:
            return "overheating (emergency stop in progress)"
        st = self.db.gpu_state(gpu)
        if st["quarantined"]:
            return f"quarantined ({st['quarantine_reason']})"
        if st["cooling"]:
            return "cooling down"
        s = self.store.latest(gpu)
        if s is None:
            return "no reading yet"
        if s.get("lost"):
            return "not answering"
        if s.get("blind"):
            return "its readings stopped (driver error)"
        mine = os.getpid()
        foreign = [pid for pid, _ in (s.get("procs") or []) if pid != mine]
        if foreign:  # this card has no job of ours, so any compute process is somebody else's
            return f"in use by another program (process {foreign[0]})"
        used, total = s.get("mem_used"), s.get("mem_total")
        if used is not None and not s.get("unified_memory"):  # a unified-memory GPU always shows use
            r = self.cfg.runner
            limit = max(r.free_mem_mb * 2**20, r.free_mem_frac * total if total else 0)
            if used >= limit:
                return (f"{used / 2**30:.1f} GiB of its memory is in use by other programs "
                        f"(more than the {limit / 2**30:.1f} GiB allowed by runner.free_mem_frac)")
        return None

    def free_gpus(self, t=None) -> list[int]:
        """Free training GPUs (that template ``t`` can use, if given): the cards with the most
        memory first (on a mixed machine the bigger card is nearly always the faster one),
        the coolest first among equals."""
        jobs = self.db.jobs_in(ACTIVE_STATES)
        out = []
        for g in self.cfg.gpus.training:
            if self.busy_reason(g, jobs) is not None:
                continue
            if t is not None and self.unsuitable(t, g):
                continue
            s = self.store.latest(g) or {}
            out.append((-(s.get("mem_total") or 0), s.get("temp_c") if s.get("temp_c") is not None else 999, g))
        return [g for _, _, g in sorted(out)]

    def owns_pid(self, pid: int) -> str | None:
        """Job ID owning this PID (same process group or systemd unit), else None."""
        try:
            pgid = os.getpgid(pid)
        except (ProcessLookupError, PermissionError):
            pgid = None
        try:
            cgroup = Path(f"/proc/{pid}/cgroup").read_text()
        except OSError:
            cgroup = ""
        for j in self.db.jobs_in(ACTIVE_STATES):
            if pid == j["pid"] or (pgid is not None and pgid == j["pgid"]) or (
                    j.get("unit") and f"{j['unit']}." in cgroup):
                return j["id"]
        return None

    # ---- control requests ------------------------------------------------------------
    def _uses_sdk(self, job: dict) -> bool:
        if job.get("sdk") is not None:  # recorded when the job was queued
            return bool(job["sdk"])
        try:  # a job queued by an older version
            t = self.templates().get(job["template"])
        except Exception:
            t = None
        return t.sdk if t else True

    def _write_control(self, job: dict, action: str, reason: str | None = None) -> int:
        rid = int(job.get("request_id") or 0) + 1
        atomic_write_json(self.run_dir(job) / "control.json",
                          {"action": action, "request_id": rid, "reason": reason, "ts": self.clock()})
        return rid

    def _rank_pids(self, job: dict) -> list[int]:
        """Training processes that registered through the SDK, verified to belong to the job."""
        out = []
        for p in self.run_dir(job).glob("rank*.pid"):
            try:
                pid = int(p.read_text().strip())
                if job.get("pgid") and os.getpgid(pid) != job["pgid"]:
                    continue
                out.append(pid)
            except (ValueError, OSError, ProcessLookupError):
                continue
        return out

    def signal_ranks(self, job: dict, sig: int) -> int:
        n = 0
        for pid in self._rank_pids(job):
            try:
                os.kill(pid, sig)
                n += 1
            except OSError:
                pass
        return n

    def signal_group(self, job: dict, sig: int) -> bool:
        if not job.get("pgid") or not pid_alive(job["pid"], job["proc_start"]):
            return False
        try:
            os.killpg(job["pgid"], sig)
            return True
        except OSError:
            return False

    def signal_stop(self, job: dict) -> bool:
        """SIGTERM to the job's training processes (the SDK checkpoints and exits), or to the whole
        job if none registered. Needs no lock and no database write."""
        try:
            if self.signal_ranks(job, signal.SIGTERM):  # also wakes a paused SDK loop
                return True
            return self.signal_group(job, signal.SIGTERM)
        except Exception:
            log.exception("could not signal %s", job.get("id"))
            return False

    def request(self, job_id: str, action: str, reason: str | None = None, detail: str | None = None) -> str:
        with self.lock:
            job = self.get(job_id)
            st = job["state"]
            sdk = self._uses_sdk(job)
            now = self.clock()
            if action == "checkpoint":
                if st not in ("running", "preflight"):
                    raise ActionError(f"{job_id} is {st}; only a running job can checkpoint.")
                if not sdk:
                    raise ActionError(f"{job_id} does not use the SDK, so it cannot checkpoint on request.")
                rid = self._write_control(job, "checkpoint", reason)
                self.db.update_job(job_id, request_id=rid, pending_action="checkpoint", pending_since=now)
                return f"💾 Checkpoint requested for {job_id}; it saves at the next optimizer step."
            if action == "pause":
                if st in ("paused", "pausing"):
                    return f"{job_id} is already {st}."
                if st not in ("running", "preflight"):
                    raise ActionError(f"{job_id} is {st}; only a running job can be paused.")
                if not sdk:
                    self.signal_group(job, signal.SIGTERM)
                    self.db.update_job(job_id, state="pausing", pending_action="stop", pending_since=now,
                                       pause_reason="pause_fallback", pause_detail=detail,
                                       kill_deadline=now + self.cfg.runner.stop_timeout_s)
                    return (f"{job_id} does not use the SDK, so pause falls back to stop; /resume {job_id} "
                            f"restarts it (work since its own last save is lost).")
                rid = self._write_control(job, "pause", reason)
                self.db.update_job(job_id, state="pausing", request_id=rid, pending_action="pause",
                                   pending_since=now, pause_reason=reason or "user", pause_detail=detail)
                self.db.event("pause_requested", reason or "user", job_id=job_id)
                return f"⏸ Pausing {job_id}: it checkpoints at the next optimizer step, then idles."
            if action == "resume":
                if st in ("paused", "pausing"):
                    if job["pending_action"] == "stop":
                        raise ActionError(f"{job_id} is stopping; /resume it once it has stopped.")
                    rid = self._write_control(job, "resume", reason)
                    self.db.update_job(job_id, state="running", request_id=rid, pending_action="resume",
                                       pending_since=now, pause_reason=None, pause_detail=None,
                                       waiting_for_user=0)
                    self.db.event("resume_requested", reason or "user", job_id=job_id)
                    return f"▶ Resuming {job_id}."
                if st in ("stopped", "oom", "stalled", "failed"):
                    upd: dict[str, Any] = dict(state="queued", waiting_for_user=0, ended_at=None, pause_reason=None,
                                               pause_detail=None, kill_reason=None, pending_action=None,
                                               interrupts=0)
                    if st == "oom":
                        upd["oom_retries"] = 0
                    if st == "stalled":
                        upd["stalls"] = 0
                    self.db.update_job(job_id, **upd)
                    self.db.event("requeued", f"from {st}", job_id=job_id)
                    return f"⏳ {job_id} queued again; it resumes from its last verified checkpoint."
                raise ActionError(f"{job_id} is {st}; there is nothing to resume.")
            if action == "stop":
                if st == "queued":
                    raise ActionError(f"{job_id} has not started; use /cancel {job_id}.")
                if st not in ACTIVE_STATES:
                    raise ActionError(f"{job_id} is {st}; it is not running.")
                # every stop has a deadline: a job that hangs instead of checkpointing is killed
                upd = dict(state="pausing", pending_action="stop", pending_since=now,
                           pause_reason=reason or "user", pause_detail=detail,
                           kill_deadline=now + self.cfg.runner.stop_timeout_s)
                if sdk:
                    upd["request_id"] = self._write_control(job, "stop", reason)
                    if st in ("paused", "pausing"):
                        self.signal_ranks(job, signal.SIGUSR1)  # wake an idling loop so it reads the request
                else:
                    self.signal_group(job, signal.SIGTERM)
                self.db.update_job(job_id, **upd)
                self.db.event("stop_requested", reason or "user", job_id=job_id)
                return f"⏹ Stopping {job_id}: it checkpoints, then exits."
            raise ActionError(f"Unknown action {action}.")

    def emergency_stop(self, job_id: str, reason: str, detail: str) -> None:
        """Checkpoint with a hard time limit, then stop. Used for heat emergencies and ECC errors.

        The signal goes out first. Recording the stop comes second and may fail (disk full,
        database locked) without leaving the job running on a failing card."""
        job = self.get(job_id)
        if job["state"] not in ACTIVE_STATES:
            return
        self.signal_stop(job)
        with self.lock:
            deadline = self.clock() + self.cfg.thermal.emergency_ckpt_limit_s
            try:  # the deadline next: whatever fails below, the job is killed when it passes
                self.db.update_job(job_id, kill_reason=reason, waiting_for_user=1, kill_deadline=deadline)
            except Exception:
                log.exception("could not record the emergency stop of %s (the signal was sent)", job_id)
            if job["pending_action"] == "stop" and job["kill_deadline"]:
                return
            try:
                self.request(job_id, "stop", reason=reason, detail=detail)
            except ActionError:
                pass
            except Exception:
                log.exception("stop request for %s failed (the signal was sent)", job_id)
                try:
                    self.db.update_job(job_id, state="pausing", pending_action="stop", pending_since=self.clock(),
                                       pause_reason=reason, pause_detail=detail)
                except Exception:
                    pass
            try:  # request() set the ordinary stop deadline; an emergency has its own, shorter one
                self.db.update_job(job_id, kill_reason=reason, waiting_for_user=1, kill_deadline=deadline)
            except Exception:
                pass

    def kill(self, job_id: str, reason: str) -> None:
        with self.lock:
            job = self.get(job_id)
            self.db.update_job(job_id, kill_reason=reason)
            self.signal_group(job, signal.SIGKILL)
            self.db.event("killed", reason, job_id=job_id)

    # ---- the loop ----------------------------------------------------------------------
    def tick(self) -> None:
        with self.lock:
            if self._launcher is not None:
                self._launcher.reap()
            for job in self.db.jobs_in(ACTIVE_STATES):
                try:
                    self._poll(job)
                except Exception:
                    log.exception("polling %s failed", job["id"])
            self._schedule()
            due, self._reports_due = self._reports_due, []
        for job_id in due:  # outside the lock: a report can take a while and nothing waits for it
            self._send_report(job_id)

    def _poll(self, job: dict) -> None:
        rd = self.run_dir(job)
        hb = read_heartbeat(rd)
        now = self.clock()
        if job["pid"] is None and job["state"] == "preflight" and job["launched_at"] \
                and now - float(job["launched_at"]) < 120:
            return  # claimed and being started right now (the row gets its PID in a moment)
        if not pid_alive(job["pid"], job["proc_start"]):
            self._on_exit(job, self._read_exit_code(rd), hb)
            return
        if job["pending_action"] and hb and int(hb.get("request_id") or 0) >= int(job["request_id"] or 0):
            self._on_ack(job, hb)
            job = self.db.get_job(job["id"])
        if job["state"] == "preflight" and hb and hb.get("probe_done"):
            self.db.update_job(job["id"], state="running")
            self.db.event("running", "probe passed", job_id=job["id"])
        elif job["state"] == "preflight" and not self._uses_sdk(job):
            self.db.update_job(job["id"], state="running")  # no SDK: no memory probe to wait for
            self.db.event("running", "no SDK", job_id=job["id"])
        if job["kill_deadline"] and now >= job["kill_deadline"]:
            self.db.event("killed", f"deadline passed ({job['kill_reason'] or job['pending_action']})", job_id=job["id"])
            self.db.update_job(job["id"], kill_deadline=None)
            self.signal_group(job, signal.SIGKILL)
        if hb and hb.get("disk_low"):
            self.notifier.warn(f"💾 Disk low: {job['id']} has pruned its old checkpoints and is now SKIPPING its "
                               f"regular saves (less than checkpoints.min_free_disk_gb free). Free some space.",
                               key=f"disk_low:{job['id']}")
        if hb and hb.get("writer_error"):
            self.notifier.warn(f"{job['id']}: {hb['writer_error']}", key=f"writer:{job['id']}")
        if now - self._ckpt_scan.get(job["id"], 0) > 30:
            self._record_checkpoints(job)

    def _on_ack(self, job: dict, hb: dict) -> None:
        act, jid, step = job["pending_action"], job["id"], fmt_num(int(hb.get("step") or 0))
        if act == "pause":
            self.db.update_job(jid, state="paused", pending_action=None)
            self.db.event("paused", f"step {step}", job_id=jid)
            if job["pause_reason"] == "thermal" and job["pause_detail"]:
                self.notifier.alert(job["pause_detail"].replace("{step}", step))
            else:
                self.notifier.send(f"⏸ Paused {jid} at step {step} after a checkpoint. /resume {jid} to continue.")
        elif act == "resume":
            self.db.update_job(jid, pending_action=None)
            self.db.event("resumed", f"step {step}", job_id=jid)
        elif act == "checkpoint":
            self.db.update_job(jid, pending_action=None)
            self._record_checkpoints(job)
            if hb.get("last_request_failed") is not None and int(hb["last_request_failed"]) >= int(job["request_id"] or 0):
                why = hb.get("writer_error") or ("the disk is too full" if hb.get("disk_low") else "see its log")
                self.db.event("checkpoint_failed", f"step {step}: {why}", job_id=jid)
                self.notifier.alert(f"💾 {jid}: the checkpoint you asked for at step {step} could NOT be saved "
                                    f"({why}).")
            else:
                self.db.event("checkpoint", f"step {step}", job_id=jid)
                self.notifier.send(f"💾 {jid}: checkpoint saved at step {step}.")

    def _read_exit_code(self, rd: Path) -> int | None:
        try:
            return int((rd / "exit_code").read_text().strip())
        except (OSError, ValueError):
            return None

    def _record_checkpoints(self, job: dict) -> None:
        self._ckpt_scan[job["id"]] = self.clock()
        for m in CheckpointStore(self.run_dir(job) / "checkpoints").manifests():
            self.db.record_checkpoint(job["id"], m)

    def _on_exit(self, job: dict, code: int | None, hb: dict | None, down: bool = False) -> None:
        """``down``: the job ended while the agent was not running (found at startup)."""
        jid, rd, now = job["id"], self.run_dir(job), self.clock()
        intent = read_json(rd / "exit_intent") or {}
        eff = code
        if code not in (0, EXIT_STOPPED, EXIT_OOM) and intent.get("code") in (EXIT_STOPPED, EXIT_OOM):
            eff = intent["code"]  # torchrun reports 1 when a rank exits 86/87
        if code is None and (rd / "done").exists():
            eff = 0  # g.finish() ran; only the wrapper's exit-code file is missing (power cut)
        elif code == 0 and intent.get("code") in (EXIT_STOPPED, EXIT_OOM) and not (rd / "done").exists():
            eff = intent["code"]  # the script swallowed the SDK's exit and returned normally
        step = int((hb or {}).get("step") or 0)
        launched = float(job["launched_at"]) if job["launched_at"] else None
        end = now
        if down and launched is not None:
            # the time it really ran, not the time until the agent came back
            seen = [float(x) for x in ((hb or {}).get("ts"),) if x]
            try:
                seen.append((rd / "exit_code").stat().st_mtime)
            except OSError:
                pass
            seen = [x for x in seen if launched <= x <= now]
            end = max(seen) if seen else launched
        run_s = float(job["run_seconds"] or 0) + (max(0.0, end - launched) if launched is not None else 0)
        # anything of the job still alive (its wrapper died first, or workers outlived the trainer)
        left = job_survivors(jid, str(rd), job["pgid"])
        for pid in left:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        if left:
            self.db.event("killed", f"{len(left)} leftover process(es) after exit", job_id=jid)
        self.db.update_job(jid, pid=None, pgid=None, proc_start=None, exit_code=code, run_seconds=run_s,
                           launched_at=None, kill_deadline=None, stall_flagged_at=None)
        try:  # a save cut short by a kill leaves a checkpoint-sized temp file behind
            CheckpointStore(rd / "checkpoints").cleanup_tmp()
        except OSError:
            pass
        self._record_checkpoints(job)
        self.db.event("exited", f"code {code} (effective {eff})", job_id=jid)
        # shallow check here (file present, right size, right settings): the full hash is done
        # by the job itself when it resumes, and hashing gigabytes would hold up everything else
        latest, _ = CheckpointStore(rd / "checkpoints").find_latest_valid(job["config_hash"], deep=False)
        from_step = f"step {fmt_num(int(latest['step']))}" if latest else "the start (no checkpoint)"
        stop_asked = job["pending_action"] == "stop"

        if eff == 0:
            note = None if ((rd / "done").exists() or not self._uses_sdk(job)) else "exited 0 without g.finish()"
            self.db.update_job(jid, state="completed", ended_at=now_iso(), pending_action=None, note=note,
                               kill_reason=None)
            self._report(jid)
            return
        if stop_asked:  # before anything else: a job the user (or an emergency) stopped stays stopped
            self.db.update_job(jid, state="stopped", pending_action=None,
                               kill_reason=job["kill_reason"] if job["kill_reason"] != "stall" else None)
            reason = job["pause_reason"] or "user"
            how = "" if eff == EXIT_STOPPED else " (it did not exit in time and was killed)"
            if job["kill_reason"] in ("emergency", "ecc", "gpu_lost"):
                self.notifier.alert(f"⏹ {jid} stopped{how}; it can resume from {from_step}. Its GPU is "
                                    f"quarantined; /resume {jid} to run it on another card.")
            elif reason == "pause_fallback":
                self.notifier.send(f"⏹ {jid} stopped (no SDK, so pause = stop). /resume {jid} restarts it.")
            else:
                self.notifier.send(f"⏹ Stopped {jid}{how}; it can resume from {from_step}. /resume {jid} to continue.")
            return
        if job["kill_reason"] == "stall":
            if job["pending_action"] == "pause":  # it hung while being paused: do not restart it
                self.db.update_job(jid, state="stopped", waiting_for_user=1, kill_reason=None, pending_action=None)
                self.notifier.alert(f"⏹ {jid} hung while pausing and was killed. /resume {jid} continues "
                                    f"from {from_step}.")
            elif int(job["stalls"] or 0) < 1:
                self.db.update_job(jid, state="queued", stalls=int(job["stalls"] or 0) + 1, kill_reason=None,
                                   pending_action=None)
                self.notifier.alert(f"⏳ {jid} made no progress for {fmt_dur(self.cfg.stall.no_progress_s)} and was "
                                    f"killed. Resuming once from {from_step}.")
            else:
                self.db.update_job(jid, state="stalled", waiting_for_user=1, kill_reason=None, pending_action=None)
                self._report(jid)
            return
        if eff == EXIT_STOPPED:
            # It checkpointed and left on a stop signal nobody here asked for: a shutdown, a
            # logout, `systemctl stop`, or a manual kill. Treated as an interruption.
            n = int(job.get("interrupts") or 0)
            if job["kill_reason"] in ("emergency", "ecc", "gpu_lost"):
                self.db.update_job(jid, state="stopped", pending_action=None, waiting_for_user=1)
                self.notifier.alert(f"⏹ {jid} stopped; it can resume from {from_step}. /resume {jid} to run it "
                                    f"on another card.")
            elif self.cfg.runner.auto_resume_after_reboot and n < MAX_INTERRUPTS:
                self.db.update_job(jid, state="queued", pending_action=None, interrupts=n + 1,
                                   pause_reason=None, pause_detail=None)
                self.db.event("interrupted", f"stop signal from outside; resumable from {from_step}", job_id=jid)
                if not down:
                    self.notifier.send(f"⏹ {jid} was stopped by a signal the agent did not send (shutdown, logout "
                                       f"or a manual kill). It is queued to continue from {from_step}; "
                                       f"/cancel {jid} if you meant to stop it.", kind="alert")
            else:
                self.db.update_job(jid, state="stopped", pending_action=None, waiting_for_user=1)
                why = (f"for the {n + 1}th time " if n >= MAX_INTERRUPTS else "")
                self.notifier.alert(f"⏹ {jid} was stopped {why}by a signal the agent did not send. "
                                    f"/resume {jid} continues from {from_step}.")
            return
        if eff == EXIT_OOM:
            self._on_oom(self.db.get_job(jid), hb, intent, step)
            return
        note = None
        if code is None:
            note = "its wrapper process died without recording an exit code (killed from outside?)"
        self.db.update_job(jid, state="failed", ended_at=now_iso(), pending_action=None, note=note)
        self._report(jid)

    def _on_oom(self, job: dict, hb: dict | None, intent: dict, step: int) -> None:
        jid = job["id"]
        probe = "probe" in ((intent or {}).get("oom_reason"), (hb or {}).get("oom_reason"))
        mb, ga, retries = job["micro_batch"], int(job["grad_accum"] or 1), int(job["oom_retries"] or 0)
        can = self.cfg.memory.oom_auto_retry and isinstance(mb, int) and mb >= 2 and mb % 2 == 0
        if can and not probe and retries >= self.cfg.memory.max_oom_retries:
            can = False
        if can:
            nmb, nga = mb // 2, ga * 2  # same effective batch
            self.db.update_job(jid, state="queued", micro_batch=nmb, grad_accum=nga, pending_action=None,
                               oom_retries=retries + (0 if probe else 1))
            self.db.event("oom", f"{'probe' if probe else 'step ' + str(step)}: micro_batch {mb}->{nmb}", job_id=jid)
            if probe:
                msg = (f"🧠 {jid}: the 20-step probe left less than {self.cfg.memory.headroom_frac:.0%} memory free "
                       f"at micro-batch {mb}; restarting with {nmb} × {nga} accumulation (same effective batch).")
            else:
                kept = ("No data lost." if (intent or {}).get("oom_saved") else
                        "It continues from its last checkpoint.")
                msg = (f"🧠 OOM at step {fmt_num(step)} on {jid}; resuming with micro-batch {nmb} × {nga} "
                       f"accumulation. {kept}")
            self.notifier.send(msg, kind="alert")
            return
        self.db.update_job(jid, state="oom", waiting_for_user=1, ended_at=now_iso(), pending_action=None,
                           note=f"out of memory at micro-batch {mb} after {retries} retries")
        self.db.event("oom", "gave up", job_id=jid)
        self._report(jid)

    def _report(self, job_id: str) -> None:
        """Queue the end-of-run report; it is written after the runner lock is released."""
        self._reports_due.append(job_id)

    def _send_report(self, job_id: str) -> None:
        job = self.db.get_job(job_id)
        if not job:
            return
        try:
            text, urgent = self.reports.run_report(job)
        except Exception as e:  # a report must never break the runner
            log.exception("report for %s failed", job_id)
            text, urgent = f"{job_id} {job['state']} (report failed: {e!r})", True
        if job["state"] == "oom":
            text += (f"\nGave up after {job['oom_retries']} OOM retries. /resume {job_id} to try again, "
                     f"or /rerun {job_id} micro_batch=<smaller>.")
        if job["state"] == "stalled":
            text += f"\nStalled twice; waiting for you. /resume {job_id} to try again."
        self.notifier.report(text, urgent=urgent)

    # ---- scheduling ----------------------------------------------------------------------
    def _schedule(self) -> None:
        """First in, first out (priority first). A job that cannot start yet reserves the cards it
        could use; later jobs may still take other cards, and may take a reserved card only if
        they are expected to finish before the blocked job could start."""
        if self.clock() < self.settle_until:
            return
        if self.db.kv_get("hold") == "1" or self.db.kv_get("disk_hold") == "1":
            return
        queued = self.db.queued_jobs()
        if not queued:
            return
        tpls = self.templates()
        free = self.free_gpus()
        reserved: set[int] = set()
        blocked_for: float | None = None
        for job in queued:
            if not free:
                break
            t = tpls.get(job["template"])
            if t is None:
                self._launch(job, free[:1])  # fails the job with a clear message
                continue
            need = int(job["n_gpus"] or 1)
            cands = [g for g in free if not self.unsuitable(t, g)]
            open_ = [g for g in cands if g not in reserved]
            pick = self.same_vendor_pick(open_, need)
            if pick is None and reserved and blocked_for is not None:
                backfill = self.same_vendor_pick(open_ + [g for g in cands if g in reserved], need)
                if backfill is not None:
                    est = self.reports.estimate_duration(job["template"], json.loads(job["config_json"] or "{}"))
                    if est is not None and est <= blocked_for:
                        pick = backfill
            if pick is not None:
                self._launch(job, pick)
                free = [g for g in free if g not in pick]
                continue
            elig = set(self.eligible_gpus(t))
            usable = [g for g in elig if not self.needs_user(g)]
            if len(usable) < need or self.same_vendor_pick(usable, need) is None:
                # it cannot start until someone acts (a card is quarantined, gone or unsuitable):
                # say so, and let the jobs behind it use the cards meanwhile
                self._say_blocked(job, t, need, elig, usable)
                continue
            if not reserved:
                blocked_for = self._seconds_until_free(need - len(cands), elig)
            reserved |= elig

    def needs_user(self, gpu: int) -> bool:
        """True if this card will not take a job again until someone steps in."""
        try:
            if self.db.gpu_state(gpu)["quarantined"]:
                return True
        except Exception:
            return True
        s = self.store.latest(gpu)
        return bool(s and s.get("lost"))

    def _say_blocked(self, job: dict, t, need: int, elig: set[int], usable: list[int]) -> None:
        out = sorted(elig - set(usable))
        why = []
        for g in out:
            st = self.db.gpu_state(g)
            why.append(f"GPU {g} is quarantined ({st['quarantine_reason']}); /clear {g} when it is safe"
                       if st["quarantined"] else f"GPU {g} is not answering")
        if not why:
            why.append("no set of matching cards exists on this machine any more")
        self.notifier.warn(
            f"⏸ {job['id']} ({t.name}) needs {need} GPU{'s' if need != 1 else ''} but only {len(usable)} can be "
            f"used: {'; '.join(why)}. It stays queued and the jobs behind it go ahead; "
            f"/cancel {job['id']} to drop it.", key=f"blocked:{job['id']}")

    def _seconds_until_free(self, more: int, gpus: set[int]) -> float | None:
        etas = []
        for j in self.db.jobs_in(ACTIVE_STATES):
            mine = set(self.gpu_list(j)) & gpus
            if not mine:
                continue
            eta = self.reports.eta_seconds(j)
            if eta is None:
                return None
            etas += [eta] * len(mine)
        etas.sort()
        if more <= 0:
            return 0.0
        return etas[more - 1] if len(etas) >= more else None

    def _launch(self, job: dict, gpus: list[int]) -> None:
        jid, now = job["id"], self.clock()
        t = self.templates().get(job["template"])
        if t is None:
            bad = dict(self.template_problems)
            why = next((f"its file {n} is being ignored: {w}" for n, w in bad.items()
                        if Path(n).stem == job["template"]), "it no longer exists")
            self.db.update_job(jid, state="failed", ended_at=now_iso(), note=f"template unusable: {why}"[:500])
            self.notifier.alert(f"❌ {jid}: template {job['template']} cannot be used ({why}). "
                                f"Fix it, then /resume {jid}.")
            return
        try:
            self.static_check(t)
        except TemplateError as e:
            self.db.update_job(jid, state="rejected", ended_at=now_iso(), note=str(e))
            self.notifier.alert(f"🚫 {jid} {e}")
            return
        root = self.cfg.root
        script = t.resolve(root, t.script)
        python = str(t.resolve(root, t.python)) if "/" in t.python else t.python
        if not script.exists():
            self.db.update_job(jid, state="failed", ended_at=now_iso(), note=f"script not found: {script}")
            self.notifier.alert(f"❌ {jid}: script not found: {script}")
            return
        if len(gpus) > 1:
            argv = [python, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(gpus)}",
                    str(script), *t.args]
        else:
            argv = [python, str(script), *t.args]
        rd = self.run_dir(job)
        cfgd = self._write_config(job)
        for name in ("heartbeat", "exit_code", "exit_intent", "done"):
            (rd / name).unlink(missing_ok=True)
        for pattern in ("rank*.pid", "*.tmp", "checkpoints/*.tmp"):  # the last launch's leftovers
            for p in rd.glob(pattern):
                p.unlink(missing_ok=True)
        ck = self.cfg.checkpoints
        atomic_write_json(rd / "guardian.json", {
            "job_id": jid, "config_hash": job["config_hash"], "best_metric": t.best_metric, "mode": t.mode,
            "every_steps": ck.every_steps, "every_minutes": ck.every_minutes, "keep_last": ck.keep_last,
            "keep_best": ck.keep_best, "min_free_disk_gb": ck.min_free_disk_gb, "async": ck.async_,
            "probe_steps": self.cfg.memory.probe_steps, "headroom_frac": self.cfg.memory.headroom_frac,
            "heartbeat_s": self.cfg.runner.heartbeat_s, "launch_request_id": int(job["request_id"] or 0),
            "launch": int(job["launches"] or 0) + 1,
        })
        atomic_write_json(rd / "control.json", {"action": "none", "request_id": int(job["request_id"] or 0)})
        with open(rd / "stdout.log", "a") as f:
            f.write(f"\n=== rig-guardian launch {int(job['launches'] or 0) + 1} of {jid} at {now_iso()} "
                    f"on GPU {','.join(map(str, gpus))} (micro_batch {cfgd.get('micro_batch')}) ===\n")
        cards = [self.store.latest(g) or {"gpu": g} for g in gpus]
        sdk_dir = str(prepare_sdk_dir(self.cfg.root))
        # A minimal environment, the same under both launchers: locale, proxy and library-path
        # basics, anything named in runner.pass_env, the template's env, then the agent's own
        # variables, which a template cannot override. The bot token is never passed.
        basics = ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TMPDIR",
                  "LD_LIBRARY_PATH", "XDG_RUNTIME_DIR", "XDG_CACHE_HOME", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE",
                  "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy")
        env = {k: os.environ[k] for k in (*basics, *self.cfg.runner.pass_env)
               if k in os.environ and k != "TELEGRAM_TOKEN"}
        env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
        env.update(t.env)
        env.update({
            "GUARDIAN_JOB_ID": jid,
            "GUARDIAN_RUN_DIR": str(rd),
            "GUARDIAN_GPU_VENDOR": self.vendor_of(gpus[0]) if gpus else "",
            **pin_env(cards),  # CUDA_VISIBLE_DEVICES, HIP_VISIBLE_DEVICES or ZE_AFFINITY_MASK
            "PYTHONPATH": os.pathsep.join(p for p in (sdk_dir, t.env.get("PYTHONPATH")) if p),
            "PYTHONUNBUFFERED": "1",
        })
        cwd = str(t.resolve(root, t.workdir)) if t.workdir else str(script.parent)
        # claim the job before starting anything: if another process took it first, do nothing
        if self.db.execute("UPDATE jobs SET state='preflight', pid=NULL, pgid=NULL, proc_start=NULL, "
                           "gpus=?, launched_at=? WHERE id=? AND state='queued'",
                           (",".join(map(str, gpus)), now, jid)).rowcount != 1:
            log.warning("%s was no longer queued when its launch started; skipping", jid)
            return
        try:
            pid, pgid, unit = self.launcher.launch(jid, argv, env, cwd)
        except (LaunchError, subprocess.TimeoutExpired, OSError) as e:
            self.db.update_job(jid, state="failed", ended_at=now_iso(), note=f"launch failed: {e}"[:500],
                               launched_at=None, gpus=None)
            self.notifier.alert(f"❌ {jid}: launch failed: {e}")
            return
        uuids = ",".join(str((c.get("uuid") or "")) for c in cards)
        self.db.update_job(jid, state="preflight", gpus=",".join(map(str, gpus)), gpu_uuids=uuids, pid=pid,
                           pgid=pgid or None,
                           proc_start=proc_start_time(pid), launcher=self.launcher.name, unit=unit,
                           started_at=job["started_at"] or now_iso(), launched_at=now,
                           launches=int(job["launches"] or 0) + 1, exit_code=None, kill_reason=None,
                           pending_action=None, waiting_for_user=0, ended_at=None)
        self.db.event("launched", f"pid {pid} on GPU {gpus}", job_id=jid)
        label = " + ".join(self.store.label(g) for g in gpus)
        est = self.reports.estimate_duration(t.name, cfgd)
        resumed = " (resuming)" if int(job["launches"] or 0) > 0 else ""
        eta = f" · est. finish {time.strftime('%H:%M', time.localtime(now + est))}" if est else ""
        mixed = ""
        if len({((c.get("name"), c.get("mem_total"))) for c in cards}) > 1:
            mixed = " · note: these are different card models, so the slower one sets the pace"
        self.notifier.queue_update(f"▶ {jid} {t.name} started on {label}{resumed}{eta}{mixed}")

    # ---- startup reconciliation ------------------------------------------------------
    def reconcile(self) -> list[str]:
        """Match the database to what is actually running, before launching anything."""
        lines = []
        with self.lock:
            for job in self.db.jobs_in(ACTIVE_STATES):
                jid, rd = job["id"], self.run_dir(job)
                if pid_alive(job["pid"], job["proc_start"]):
                    lines.append(f"Reattached {jid}.")
                    self.db.event("reattached", job_id=jid)
                    continue
                store = CheckpointStore(rd / "checkpoints")
                removed = store.cleanup_tmp()
                if removed:
                    self.db.event("cleanup", "removed " + ", ".join(removed), job_id=jid)
                if (rd / "done").exists() or (rd / "exit_code").exists():
                    self._on_exit(job, self._read_exit_code(rd), read_heartbeat(rd), down=True)
                    after = self.db.get_job(jid)["state"]
                    lines.append(f"Resuming {jid} (it was stopped by the shutdown)." if after == "queued"
                                 else f"{jid} ended while the agent was down: {after}.")
                    continue
                latest, failures = store.find_latest_valid(job["config_hash"])
                for name, why in failures:
                    self.db.event("checkpoint_rejected", f"{name}: {why}", job_id=jid)
                last_seen = (read_heartbeat(rd) or {}).get("ts")  # run time up to the power cut
                run_s = float(job["run_seconds"] or 0) + (
                    max(0.0, float(last_seen) - float(job["launched_at"])) if last_seen and job["launched_at"] else 0)
                from_step = f"step {fmt_num(int(latest['step']))}" if latest else "the start"
                gone = dict(pid=None, pgid=None, proc_start=None, launched_at=None, pending_action=None,
                            kill_deadline=None, stall_flagged_at=None, run_seconds=run_s)
                # what the user (or a safety rule) had decided before the interruption still holds
                held = None
                if job["kill_reason"] in ("emergency", "ecc", "gpu_lost"):
                    held = "it was being stopped for a hardware problem"
                elif job["pending_action"] == "stop":
                    held = "it was being stopped"
                elif job["state"] in ("paused", "pausing") and job["pause_reason"] != "thermal":
                    held = "it was paused"
                elif not self.cfg.runner.auto_resume_after_reboot:
                    held = "automatic resume is off"
                if held is None:
                    self.db.update_job(jid, state="queued", pause_reason=None, pause_detail=None, **gone)
                    lines.append(f"Resuming {jid} from {from_step}.")
                else:
                    self.db.update_job(jid, state="stopped", waiting_for_user=1, **gone)
                    lines.append(f"{jid} was interrupted ({held}); it can continue from {from_step} "
                                 f"(/resume {jid}).")
                self.db.event("interrupted", f"resumable from {from_step}", job_id=jid)
            # the pause before launching lets drivers and cooling settle after a reboot or a power
            # cut; a plain start with nothing to recover does not need it
            interrupted = any(ln.startswith("Resuming") or "interrupted" in ln for ln in lines)
            self.settle_until = self.clock() + (self.cfg.runner.settle_s if interrupted else 0)
            self.db.kv_set("settle_until", str(self.settle_until) if interrupted else None)
            due, self._reports_due = self._reports_due, []
        for job_id in due:
            self._send_report(job_id)
        return lines

    def remap_gpus(self, inventory: list[dict]) -> list[str]:
        """Jobs remember their cards by ID. If cards were added, removed or reordered while a
        job kept running, point the job at its cards' new indexes, so the watchdog guards the
        card the job is really on."""
        where = {g.get("uuid"): g["gpu"] for g in inventory if g.get("uuid")}
        lines = []
        with self.lock:
            for job in self.db.jobs_in(ACTIVE_STATES):
                uuids = [u for u in (job.get("gpu_uuids") or "").split(",")]
                old = self.gpu_list(job)
                if len(uuids) != len(old) or not all(uuids):
                    continue  # launched by an older version: nothing to go by
                new = [where.get(u) for u in uuids]
                if new == old:
                    continue
                if any(n is None for n in new):
                    lines.append(f"{job['id']} was running on a card that is no longer detected "
                                 f"(GPU {','.join(map(str, old))} before). Check it: /job {job['id']}.")
                    continue
                self.db.update_job(job["id"], gpus=",".join(map(str, new)))
                self.db.event("gpu_remap", f"{old} -> {new}", job_id=job["id"])
                lines.append(f"{job['id']} is on GPU {','.join(map(str, new))} now "
                             f"(was GPU {','.join(map(str, old))}).")
        return lines

"""Watchdog: a rule engine over telemetry and job heartbeats, run every 2 seconds.

Its rules are plain code with fixed thresholds; it acts without waiting for you or the
language model. Every action is logged and sent to Telegram.

Smoothing: a thermal rule fires only when its condition holds for 3 samples in a row
(6 seconds), so one noisy reading never pauses a run.

Thermal tiers are relative to each card's own slowdown temperature (T_slow):

    Normal     below T_slow - 10 °C             nothing
    Warn       at or above T_slow - 10 °C       one alert per hour per GPU
    Pause      at or above T_slow - 4 °C, or    checkpoint, then soft-pause the job
               thermal throttling for 60 s
    Emergency  at or above T_slow, or GPU lost  checkpoint (60 s limit), stop, quarantine
    Resume     below T_slow - 15 °C for 5 min   resume, up to 3 times per job per day

A reading that is missing does not count as cool: the tiers are judged on the last three
temperatures the card actually reported. A card whose readings stop (driver error, hung
sensor, collector not running) is reported after ``telemetry.stale_alert_s`` and its jobs are
paused after ``telemetry.stale_pause_s``. Cards that never report a temperature (some
integrated and Apple GPUs) are named once at startup; heat rules cannot apply to them.

The emergency stop signals the job before any bookkeeping, and still does so when the
database cannot be written or the runner is busy.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Callable

from guardian.checkpoints import CheckpointStore, disk_free_bytes
from guardian.db import ACTIVE_STATES
from guardian.reports import fmt_dur, fmt_num, read_heartbeat
from guardian.runner import ActionError
from guardian.telemetry import THERMAL_THROTTLE_MASK

log = logging.getLogger(__name__)

NORMAL, WARN, PAUSE, EMERGENCY = range(4)
LOCK_WAIT_S = 0.5
CAP_RETRY_S = (60.0, 300.0, 1800.0)  # waits after the power-limit helper fails


def _proc_name(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return "?"


class Watchdog:
    def __init__(self, cfg, db, store, runner, notifier, clock: Callable[[], float] = time.time):
        self.cfg, self.db, self.store, self.runner, self.notifier, self.clock = (
            cfg, db, store, runner, notifier, clock)
        self.cool_since: dict[int, float | None] = {}
        try:  # power limits lowered before a restart still have to be restored
            self.capped: set[int] = {int(g) for g in json.loads(db.kv_get("power_capped") or "[]")}
        except (ValueError, TypeError):
            self.capped = set()
        self._cap_retry: dict[int, tuple[int, float]] = {}  # gpu -> (failures, earliest next try)
        self._last_disk = 0.0
        self.temp_seen: dict[int, float] = {}  # when each card last reported a temperature
        self.started = clock()

    # ---- classification -------------------------------------------------------------
    def t_slow(self, s: dict, gpu: int | None = None) -> float:
        """The card's own slowdown temperature, unless overridden in the config."""
        th = self.cfg.thermal
        g = s.get("gpu") if gpu is None else gpu
        if g in th.slowdown_override_c:
            return float(th.slowdown_override_c[g])
        return float(s.get("t_slow") or th.fallback_slowdown_c)

    def tier_of(self, s: dict) -> int:
        if s.get("lost"):
            return EMERGENCY
        t = s.get("temp_c")
        if t is None:
            return NORMAL
        th, ts = self.cfg.thermal, self.t_slow(s)
        if t >= ts:
            return EMERGENCY
        if t >= ts - th.pause_margin_c:
            return PAUSE
        if t >= ts - th.warn_margin_c:
            return WARN
        return NORMAL

    def smoothed_tier(self, gpu: int) -> int:
        n = max(1, self.cfg.thermal.smoothing_samples)
        latest = self.store.latest(gpu)
        if latest is None:
            return NORMAL
        if latest.get("lost"):
            return EMERGENCY  # nothing from this card can be trusted any more
        # the last n readings that carry a temperature: a failed read in between neither
        # counts as cool nor restarts the count (they must still be recent)
        recent = [x for x in self.store.window(gpu, max(30.0, 4 * n * self.cfg.telemetry.interval_s), latest["ts"])
                  if x.get("temp_c") is not None and not x.get("lost")][-n:]
        tier = min(self.tier_of(x) for x in recent) if len(recent) >= n else NORMAL
        # sustained thermal throttling counts as the pause tier even below the temperature line
        span = self.cfg.thermal.throttle_pause_s
        now = latest["ts"]
        oldest = self.store.oldest_ts(gpu)
        if oldest is not None and oldest <= now - span:
            win = self.store.window(gpu, span, now)
            if win and all(int(x.get("throttle") or 0) & THERMAL_THROTTLE_MASK for x in win):
                tier = max(tier, PAUSE)
        return tier

    # ---- the loop --------------------------------------------------------------------
    def protected(self) -> list[int]:
        """Cards the rules act on: the training cards, plus any card a job is on right now (a job
        keeps its protection when a restart with a new config takes its card out of training)."""
        gpus = set(self.cfg.gpus.training)
        try:
            for j in self.db.jobs_in(ACTIVE_STATES):
                gpus.update(self.runner.gpu_list(j))
        except Exception:
            log.exception("could not list active jobs")
        return sorted(gpus)

    def tick(self) -> None:
        now = self.clock()
        # The runner holds this lock while it verifies a checkpoint or starts a job, which can
        # take many seconds. Heat does not wait: if the lock is busy, do the one thing that
        # cannot be late (signal jobs on a card in emergency) and leave the rest to the next tick.
        if not self.runner.lock.acquire(timeout=LOCK_WAIT_S):
            self._urgent_only()
            return
        try:
            self._tick(now)
        finally:
            self.runner.lock.release()

    def _urgent_only(self) -> None:
        for gpu in self.protected():
            try:
                if self.smoothed_tier(gpu) != EMERGENCY or self.db.gpu_state(gpu)["quarantined"]:
                    continue
                self.runner.hot_gpus.add(gpu)  # the scheduler must not start anything here meanwhile
                for j in self._jobs_on(gpu):
                    log.error("GPU %d in emergency while the runner is busy: signalling %s now", gpu, j["id"])
                    self.runner.signal_stop(j)
            except Exception:
                log.exception("urgent check failed on GPU %d", gpu)

    def _tick(self, now: float) -> None:
        protected = self.protected()
        for gpu in protected:
            for rule in (self._thermal, self._blind, self._ecc, self._stray):
                try:
                    rule(gpu, now)
                except Exception:
                    log.exception("watchdog rule %s failed on GPU %d", rule.__name__, gpu)
        for gpu in self.store.gpus():
            if gpu not in protected:
                try:
                    self._monitor_only(gpu, now)
                except Exception:
                    log.exception("monitoring GPU %d failed", gpu)
        for rule in (self._thermal_resume, self._stalls, self._idle):
            try:
                rule(now)
            except Exception:
                log.exception("watchdog rule %s failed", rule.__name__)
        if now - self._last_disk >= 60:
            self._last_disk = now
            try:
                self.check_disk()
            except Exception:
                log.exception("disk check failed")

    def _jobs_on(self, gpu: int, states=ACTIVE_STATES) -> list[dict]:
        return [j for j in self.db.jobs_in(states) if gpu in self.runner.gpu_list(j)]

    # ---- thermal ------------------------------------------------------------------------
    def _thermal(self, gpu: int, now: float) -> None:
        s = self.store.latest(gpu)
        if s is None:
            return
        tier = self.smoothed_tier(gpu)
        th, label, st = self.cfg.thermal, self.store.label(gpu), self.db.gpu_state(gpu)
        ts, temp = self.t_slow(s), s.get("temp_c")
        if temp is None and tier >= PAUSE:  # the newest read failed: speak of the last real reading
            temp = next((x["temp_c"] for x in reversed(self.store.window(gpu, 60.0, s["ts"]))
                         if x.get("temp_c") is not None), None)
        if tier != EMERGENCY:
            # an emergency seen while the runner was busy that did not last: the card is usable again
            self.runner.hot_gpus.discard(gpu)

        if tier == EMERGENCY:
            self.cool_since[gpu] = None
            lost = bool(s.get("lost"))
            reason = ("GPU lost from the bus" if lost else
                      f"{fmt_num(temp)} °C, at or above its slowdown temperature {ts:.0f} °C")
            kind = "gpu_lost" if lost else "emergency"
            jobs = self._jobs_on(gpu)
            if st["quarantined"]:
                # already handled, unless a job on the card somehow has no stop on its way
                for j in jobs:
                    if not (j["pending_action"] == "stop" and j["kill_deadline"]):
                        self.runner.emergency_stop(j["id"], kind, f"{label}: {reason}")
                return
            # the jobs first; the records and the message come after and may fail without harm
            for j in jobs:
                try:
                    self.runner.emergency_stop(j["id"], kind, f"{label}: {reason}")
                except Exception:
                    log.exception("emergency stop of %s failed", j["id"])
            self.db.set_gpu_state(gpu, quarantined=1, quarantine_reason=reason, cooling=1)
            self.runner.hot_gpus.discard(gpu)
            self.db.event("emergency", reason, gpu=gpu)
            stopping = (f"Stopping {', '.join(j['id'] for j in jobs)} (checkpoint first, "
                        f"{th.emergency_ckpt_limit_s:.0f} s limit). ") if jobs else ""
            self.notifier.alert(f"🚨 {label}: {reason}. {stopping}No new jobs on this card until you /clear {gpu}.")
            return

        if tier == PAUSE:
            self.cool_since[gpu] = None
            if not st["cooling"]:
                self.db.set_gpu_state(gpu, cooling=1)
                self.db.event("cooling", f"{temp} °C", gpu=gpu)
            throttling = temp is None or temp < ts - th.pause_margin_c
            why = (f"{label} thermal-throttling for {th.throttle_pause_s:.0f} s at {fmt_num(temp)} °C" if throttling
                   else f"{label} at {temp:.0f} °C (slowdown at {ts:.0f} °C)")
            for j in self._jobs_on(gpu, ("running", "preflight")):
                day = time.strftime("%Y-%m-%d", time.localtime(now))
                count = int(j["auto_resumes_today"] or 0) if j["auto_resume_day"] == day else 0
                if count < th.max_auto_resumes_per_day:
                    tail = (f"Will resume when below {ts - th.resume_below_margin_c:.0f} °C for "
                            f"{fmt_dur(th.resume_hold_s)}.")
                else:
                    tail = (f"It already auto-resumed {count} times today, so it stays paused until you "
                            f"/resume {j['id']}; repeated pauses usually mean the case needs better airflow.")
                detail = f"🌡 {why}. Paused {j['id']} at step {{step}} after a checkpoint. {tail}"
                try:
                    msg = self.runner.request(j["id"], "pause", reason="thermal", detail=detail)
                    self.db.event("thermal_pause", why, job_id=j["id"], gpu=gpu)
                    if "falls back to stop" in msg:
                        self.notifier.alert(f"🌡 {why}. {msg}")
                except ActionError as e:
                    log.warning("thermal pause of %s failed: %s", j["id"], e)
            return

        if tier == WARN and temp is not None:
            if self.notifier.warn(f"🌡 {label} at {temp:.0f} °C (warn at {ts - th.warn_margin_c:.0f}, "
                                  f"slowdown at {ts:.0f} °C)", key=f"warn:{gpu}"):
                self.db.event("thermal_warn", f"{temp} °C", gpu=gpu)
            self._power_cap(gpu, True)

        self._cooling_check(gpu, s, ts, now)

    def _cooling_check(self, gpu: int, s: dict, ts: float, now: float) -> None:
        """Clear the card's cooling flag (and restore its power limit) once it has stayed
        below the resume temperature for the hold time."""
        th = self.cfg.thermal
        st = self.db.gpu_state(gpu)
        if not (st["cooling"] or gpu in self.capped) or st["quarantined"]:
            self.cool_since[gpu] = None
            return
        temp, thr = s.get("temp_c"), ts - th.resume_below_margin_c
        if temp is None:
            if not self._never_reports(gpu, now):
                self.cool_since[gpu] = None  # no reading is not a cool reading
                return
            temp = thr - 1  # a card with no sensor at all: only the hold time can apply
        if temp >= thr:
            self.cool_since[gpu] = None
            return
        if self.cool_since.get(gpu) is None:
            self.cool_since[gpu] = now
            return
        if now - self.cool_since[gpu] < th.resume_hold_s:
            return
        self.cool_since[gpu] = None
        self._power_cap(gpu, False)
        if st["cooling"]:
            self.db.set_gpu_state(gpu, cooling=0)
            self.db.event("cooled", f"{s.get('temp_c')} °C", gpu=gpu)

    def _thermal_resume(self, now: float) -> None:
        """Resume jobs that were paused for heat once every card they use has cooled. Checked
        on every tick, so a job that only finished pausing after its card had already cooled
        is still resumed."""
        th = self.cfg.thermal
        day = time.strftime("%Y-%m-%d", time.localtime(now))
        for j in self.db.jobs_in(("paused",)):
            if j["pause_reason"] != "thermal" or j["pending_action"]:
                continue
            gpus = self.runner.gpu_list(j)
            states = [self.db.gpu_state(g) for g in gpus]
            if not gpus or any(st["cooling"] or st["quarantined"] for st in states):
                continue  # a multi-GPU job waits for every one of its cards
            if any(self._blind_for(g, now) > 0 for g in gpus):
                continue  # not before the card can be read again
            label = " + ".join(self.store.label(g) for g in gpus)
            temps = [(self.store.latest(g) or {}).get("temp_c") for g in gpus]
            hottest = max((t for t in temps if t is not None), default=None)
            count = int(j["auto_resumes_today"] or 0) if j["auto_resume_day"] == day else 0
            if count < th.max_auto_resumes_per_day:
                try:
                    self.runner.request(j["id"], "resume", reason="thermal")
                except ActionError as e:
                    log.warning("thermal resume of %s failed: %s", j["id"], e)
                    continue
                self.db.update_job(j["id"], auto_resumes_today=count + 1, auto_resume_day=day)
                back = f"is back to {hottest:.0f} °C" if hottest is not None else "has had its cooling time"
                self.notifier.send(f"▶ {label} {back}; resumed {j['id']} "
                                   f"({count + 1}/{th.max_auto_resumes_per_day} auto-resumes today).")
            else:
                self.db.update_job(j["id"], pause_reason="thermal_hold", waiting_for_user=1,
                                   auto_resumes_today=count, auto_resume_day=day)
                self.notifier.alert(f"🌡 {label} has cooled, but {j['id']} already auto-resumed {count} times "
                                    f"today. It stays paused until you /resume {j['id']}.")

    # ---- cards that stop reporting ---------------------------------------------------------
    def _never_reports(self, gpu: int, now: float) -> bool:
        """True for a card that has never been seen to report a temperature: it has no usable
        sensor (some integrated and Apple GPUs), which was announced at startup."""
        if gpu in self.temp_seen:
            return False
        for x in reversed(self.store.window(gpu, 86400.0, now)):  # the ring buffer, newest first
            if x.get("temp_c") is not None and not x.get("lost"):
                self.temp_seen[gpu] = float(x["ts"])
                return False
        return self.store.latest(gpu) is not None

    def _blind_for(self, gpu: int, now: float) -> float:
        """Seconds since this card last gave a trustworthy temperature (0: it is fine, or it
        never had a sensor)."""
        s = self.store.latest(gpu)
        if s is not None and s.get("temp_c") is not None and not s.get("lost"):
            self.temp_seen[gpu] = max(self.temp_seen.get(gpu, 0.0), float(s["ts"]))
        elif self._never_reports(gpu, now):
            return 0.0
        seen = self.temp_seen.get(gpu)
        if seen is None:  # no sample of any kind yet
            return max(0.0, now - self.started)
        return max(0.0, now - seen)

    def _blind(self, gpu: int, now: float) -> None:
        """No reading is not good news. Alert when a card's temperature has been missing for
        ``stale_alert_s`` (hung driver, failed sensor, telemetry thread stuck); pause its jobs
        after ``stale_pause_s``, because nothing here can protect a card it cannot see."""
        tc = self.cfg.telemetry
        s = self.store.latest(gpu)
        if s is not None and s.get("lost"):
            return  # the emergency rule owns a lost card
        blind = self._blind_for(gpu, now)
        if blind < tc.stale_alert_s:
            return
        label = self.store.label(gpu)
        stale = s is None or now - float(s["ts"]) >= tc.stale_alert_s
        what = ("no readings at all (the telemetry loop or the driver is stuck)" if stale
                else "no temperature reading" + (f" ({s['error']})" if s.get("error") else ""))
        jobs = self._jobs_on(gpu, ("running", "preflight"))
        pausing = tc.stale_pause_s > 0 and blind >= tc.stale_pause_s and jobs
        if not pausing:
            tail = ""
            if jobs and tc.stale_pause_s > 0:
                tail = (f" {', '.join(j['id'] for j in jobs)} will be paused if this lasts "
                        f"{fmt_dur(tc.stale_pause_s)}.")
            if self.notifier.rate_ok(f"blind:{gpu}"):
                self.notifier.alert(f"⚠️ {label} (GPU {gpu}): {what} for {fmt_dur(blind)}. "
                                    f"Heat protection cannot see this card.{tail}")
                self.db.event("blind", what, gpu=gpu)
            return
        if not self.db.gpu_state(gpu)["cooling"]:
            self.db.set_gpu_state(gpu, cooling=1)  # no new jobs here until it can be read again
        ts = self.t_slow(s or {}, gpu)
        for j in jobs:
            detail = (f"⚠️ {label}: {what} for {fmt_dur(blind)}. Paused {j['id']} at step {{step}} after a "
                      f"checkpoint. It resumes on its own once the card reports again and stays below "
                      f"{ts - self.cfg.thermal.resume_below_margin_c:.0f} °C for {fmt_dur(self.cfg.thermal.resume_hold_s)}.")
            try:
                msg = self.runner.request(j["id"], "pause", reason="thermal", detail=detail)
                self.db.event("blind_pause", what, job_id=j["id"], gpu=gpu)
                self.notifier.alert(f"⚠️ {label} (GPU {gpu}): {what} for {fmt_dur(blind)}. Pausing {j['id']}. "
                                    + (msg if "falls back to stop" in msg else ""))
            except ActionError as e:
                log.warning("pause of %s on an unreadable card failed: %s", j["id"], e)

    def _power_cap(self, gpu: int, on: bool) -> None:
        """Optional: lower the card's power limit at the warn tier through a root-owned helper."""
        pc = self.cfg.thermal.power_cap
        if not pc.enabled or on == (gpu in self.capped):
            return
        card = self.store.latest(gpu) or {}
        if (card.get("vendor") or "nvidia") != "nvidia":
            return  # the helper drives nvidia-smi; other vendors are left at their own limits
        now = self.clock()
        fails, retry_at = self._cap_retry.get(gpu, (0, 0.0))
        if now < retry_at:
            return  # the helper failed a moment ago: not through sudo again on every tick
        arg = str(pc.watts) if on else "reset"
        err = None
        try:  # nvidia-smi numbers only the NVIDIA cards: pass the card's index among them
            r = subprocess.run(["sudo", "-n", pc.helper, str(card.get("local", gpu)), arg],
                               capture_output=True, text=True, timeout=20)
            if r.returncode != 0:
                err = (r.stderr or r.stdout or f"exit code {r.returncode}").strip()[:200]
        except (OSError, subprocess.TimeoutExpired) as e:
            err = str(e)
        if err is not None:
            wait = CAP_RETRY_S[min(fails, len(CAP_RETRY_S) - 1)]
            self._cap_retry[gpu] = (fails + 1, now + wait)
            log.warning("power cap helper failed on GPU %d (%s); next try in %.0f s", gpu, err, wait)
            if fails == 0:
                self.notifier.warn(f"⚡ The power-limit helper failed on {self.store.label(gpu)}: {err}. "
                                   f"Check the sudoers entry (see deploy/rg-powercap).", key=f"powercap:{gpu}")
            return
        self._cap_retry.pop(gpu, None)
        (self.capped.add if on else self.capped.discard)(gpu)
        self.db.kv_set("power_capped", json.dumps(sorted(self.capped)) if self.capped else None)
        label = self.store.label(gpu)
        self.db.event("power_cap", arg, gpu=gpu)
        self.notifier.send(f"⚡ {label} power limit {'lowered to ' + arg + ' W' if on else 'restored'}.")

    def _monitor_only(self, gpu: int, now: float) -> None:
        """Cards that do not train (language-model card, excluded cards): warn, never act."""
        s = self.store.latest(gpu)
        if s is None:
            return
        tier = self.smoothed_tier(gpu)
        role = "language-model card" if gpu == self.cfg.gpus.agent else "not used for training"
        label = f"{self.store.label(gpu)} (GPU {gpu}, {role})"
        if s.get("lost"):
            if self.notifier.rate_ok(f"lost:{gpu}"):
                extra = "; free-text replies will be offline" if gpu == self.cfg.gpus.agent else ""
                self.notifier.alert(f"🚨 {label} lost from the bus{extra}.")
        elif tier >= WARN and s.get("temp_c") is not None:
            self.notifier.warn(f"🌡 {label} at {s['temp_c']:.0f} °C (slowdown at {self.t_slow(s, gpu):.0f} °C)",
                               key=f"warn:{gpu}")

    # ---- memory errors and strays ----------------------------------------------------
    def _ecc(self, gpu: int, now: float) -> None:
        s = self.store.latest(gpu)
        v = s.get("ecc_uncorrected") if s else None
        if v is None:
            return
        key = f"ecc_baseline:{gpu}"
        base = self.db.kv_get(key)
        if base is None:
            self.db.kv_set(key, str(v))
            return
        if int(v) < int(base):
            self.db.kv_set(key, str(v))  # the counter restarted (reboot, driver reload): count from here
            return
        if int(v) == int(base):
            return
        self.db.kv_set(key, str(v))
        if self.db.gpu_state(gpu)["quarantined"]:
            return
        label = self.store.label(gpu)
        reason = f"uncorrectable ECC errors rose from {base} to {v}"
        jobs = self._jobs_on(gpu)
        for j in jobs:
            try:
                self.runner.emergency_stop(j["id"], "ecc", f"{label}: {reason}")
            except Exception:
                log.exception("stop of %s after ECC errors failed", j["id"])
        self.db.set_gpu_state(gpu, quarantined=1, quarantine_reason=reason)
        self.db.event("ecc", reason, gpu=gpu)
        self.notifier.alert(f"🧯 {label}: {reason}. " + (f"Stopping {', '.join(j['id'] for j in jobs)}. " if jobs else "")
                            + f"Quarantined until you /clear {gpu}.")

    def _stray(self, gpu: int, now: float) -> None:
        s = self.store.latest(gpu)
        if not s:
            return
        for pid, used in s.get("procs") or []:
            if pid == os.getpid() or self.runner.owns_pid(pid):
                continue
            mib = f"{used / 2**20:.0f} MiB" if used else "memory"
            self.notifier.warn(f"{self.store.label(gpu)}: process {pid} ({_proc_name(pid)}) is using {mib} and was "
                               f"not started by the agent. Left alone.", key=f"stray:{gpu}:{pid}")

    # ---- stalls and idle GPUs ----------------------------------------------------------
    def _stalls(self, now: float) -> None:
        sc = self.cfg.stall
        for j in self.db.jobs_in(("preflight", "running", "pausing")):
            if j["pending_action"] == "stop":
                continue  # being stopped: its stop deadline ends it, not the stall rule
            hb = read_heartbeat(self.runner.run_dir(j))
            if not hb or hb.get("paused"):
                continue  # not an SDK job (yet), or idling on purpose
            alive_ts = float(hb.get("ts") or 0)  # written by the job's own heartbeat thread
            ts = float(hb.get("progress_ts") or alive_ts)  # last training step, beat() or save
            # Between epochs (evaluation) and before the first step of a launch (loading data
            # and weights, compiling) there are legitimately no steps for a long time.
            phase = hb.get("phase") or ("start" if hb.get("launch_steps") == 0 else "train")
            limit = sc.no_progress_s if phase == "train" else max(sc.no_progress_s, sc.quiet_phase_s)
            # ... but a process whose heartbeat thread has gone silent is not doing anything
            frozen = "progress_ts" in hb and now - alive_ts > sc.no_progress_s
            if j["stall_flagged_at"]:
                flagged = float(j["stall_flagged_at"])
                if ts > flagged or (alive_ts > flagged and now - ts <= limit):
                    self.db.update_job(j["id"], stall_flagged_at=None)
                    self.db.event("stall_recovered", job_id=j["id"])
                    self.notifier.send(f"✅ {j['id']} is making progress again.")
                elif now - float(j["stall_flagged_at"]) >= sc.kill_after_s:
                    self.runner.kill(j["id"], "stall")
                continue
            age = now - ts
            if age > limit or frozen:
                self.runner.signal_ranks(j, signal.SIGUSR1)
                self.db.update_job(j["id"], stall_flagged_at=now)
                what = ("the process has not answered" if frozen and age <= limit else
                        {"start": "no first training step", "eval": "no progress between epochs"}.get(
                            phase, "no progress"))
                since = now - alive_ts if frozen and age <= limit else age
                self.db.event("stall", f"{what} for {since:.0f} s", job_id=j["id"])
                self.notifier.alert(
                    f"⏳ {j['id']}: {what} for {fmt_dur(since)} (last step {fmt_num(int(hb.get('step') or 0))}). "
                    f"Asked it to checkpoint; if still stuck in {fmt_dur(sc.kill_after_s)} it is killed and "
                    f"resumed once from its last checkpoint.")

    def _idle(self, now: float) -> None:
        sc = self.cfg.stall
        for j in self.db.jobs_in(("running",)):
            if not j["launched_at"] or now - float(j["launched_at"]) < sc.idle_warn_s:
                continue
            for gpu in self.runner.gpu_list(j):
                oldest = self.store.oldest_ts(gpu)
                if oldest is None or now - oldest < sc.idle_warn_s - 2 * self.cfg.telemetry.interval_s:
                    continue
                win = self.store.window(gpu, sc.idle_warn_s, now)
                if win and all(s.get("util") is not None and s["util"] < sc.idle_util_pct for s in win):
                    self.notifier.warn(
                        f"{j['id']}: {self.store.label(gpu)} near 0% busy for {fmt_dur(sc.idle_warn_s)}; "
                        f"usually a data-loader problem.", key=f"idle:{j['id']}:{gpu}")

    # ---- disk --------------------------------------------------------------------------
    def check_disk(self) -> None:
        runs = self.cfg.runs_dir
        min_free = self.cfg.checkpoints.min_free_disk_gb * 2**30
        free = disk_free_bytes(runs)
        held = self.db.kv_get("disk_hold") == "1"
        if free >= min_free:
            if held and free >= min_free * 1.1:
                self.db.kv_set("disk_hold", None)
                self.db.event("disk_ok", f"{free / 2**30:.0f} GiB free")
                self.notifier.send(f"💾 Disk back to {free / 2**30:.0f} GiB free; the queue continues.")
            return
        removed = 0
        tpls = self.runner.templates()
        for j in self.db.all("SELECT * FROM jobs WHERE run_dir IS NOT NULL"):
            if j["state"] in ACTIVE_STATES:
                continue  # running jobs prune their own folder before each save
            t = tpls.get(j["template"])
            store = CheckpointStore(Path(j["run_dir"]) / "checkpoints")
            removed += len(store.prune(1, self.cfg.checkpoints.keep_best, t.mode if t else "min"))
        free = disk_free_bytes(runs)
        if free < min_free and not held:
            self.db.kv_set("disk_hold", "1")
            self.db.event("disk_low", f"{free / 2**30:.1f} GiB free")
            self.notifier.alert(
                f"💾 Disk low: {free / 2**30:.1f} GiB free (limit {self.cfg.checkpoints.min_free_disk_gb:g} GiB). "
                f"Pruned {removed} old checkpoint files; still low, so no new jobs start until space is freed.")
        elif removed:
            self.notifier.warn(f"💾 Disk was low; pruned {removed} old checkpoint files.", key="disk_pruned")


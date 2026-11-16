"""The guardian daemon: one Python service with a few threads and an async Telegram task.

    telemetry   reads every GPU every 2 s into the ring buffer, 30-s averages into SQLite
    watchdog    applies the safety rules every 2 s (no model involved)
    runner      polls jobs every 2 s and launches queued ones on free, suitable GPUs
    maintenance warning grouping, daily digest, nightly pruning, external heartbeat
    main loop   Telegram long polling and the outbox sender (asyncio)

Every detected GPU is monitored; ``gpus.training: auto`` trains on all of them except the
optional language-model card and any excluded ones. Run with ``rig-guardian start``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from guardian.actions import Agent
from guardian.brain import Brain
from guardian.checks import gpu_identity_problems
from guardian.config import Config, load_env
from guardian.db import Database
from guardian.notifier import Notifier, redact, today_at
from guardian.reports import Reports
from guardian.runner import Runner
from guardian.telemetry import Collector, NoGpuError, TelemetryStore, inventory_table, make_source, short_name
from guardian.watchdog import Watchdog

log = logging.getLogger("guardian")


class LoopThread(threading.Thread):
    def __init__(self, name: str, fn: Callable[[], None], interval_s: float):
        super().__init__(name=name, daemon=True)
        self.fn, self.interval_s = fn, interval_s
        self.stop_event = threading.Event()
        self.failures = 0
        self.last_ok = time.monotonic()

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.fn()
                self.failures = 0
            except Exception:
                self.failures += 1
                log.exception("%s loop failed (%d in a row)", self.name, self.failures)
            self.last_ok = time.monotonic()  # the round came back (a hung call never gets here)
            self.stop_event.wait(self.interval_s)


class Daemon:
    def __init__(self, cfg: Config, source=None, launcher=None, clock: Callable[[], float] = time.time):
        self.cfg, self.clock = cfg, clock
        cfg.runs_dir.mkdir(parents=True, exist_ok=True)
        self.db = Database(cfg.db_path)
        self.source = source or make_source(cfg, clock=clock)
        self.inventory = self.source.inventory()
        cfg.resolve_gpus(self.inventory)
        t = cfg.telemetry
        self.store = TelemetryStore(maxlen=int(t.ring_minutes * 60 / t.interval_s) + 10)
        self.notifier = Notifier(self.db, cfg.telegram, clock=clock)
        self.brain = Brain(cfg.brain) if cfg.brain.enabled else None
        self.reports = Reports(cfg, self.db, self.store, lambda: self.runner.templates(), brain=self.brain,
                               clock=clock)
        self.runner = Runner(cfg, self.db, self.store, self.notifier, self.reports, clock=clock, launcher=launcher)
        self.watchdog = Watchdog(cfg, self.db, self.store, self.runner, self.notifier, clock=clock)
        self.agent = Agent(cfg, self.db, self.runner, self.reports, self.notifier, brain=self.brain, clock=clock)
        self.collector = Collector(self.source, self.store, self.db, t.interval_s, t.db_every_s,
                                   on_lost=self._on_lost, clock=clock)
        self.threads: list[threading.Thread] = []
        self._last_health = 0.0
        self._lock_fd: int | None = None

    # ---- one agent per rig folder ---------------------------------------------------------
    def acquire_lock(self) -> None:
        """Two agents on one folder would both launch, poll and signal the same jobs."""
        if self._lock_fd is not None:
            return
        import fcntl

        path = Path(self.cfg.root) / "agent.lock"
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            try:
                other = os.pread(fd, 32, 0).decode().strip() or "?"
            finally:
                os.close(fd)
            raise SystemExit(f"another Rig Guardian agent (process {other}) is already running on "
                             f"{self.cfg.root}. Stop it first (systemctl --user stop rig-guardian), or use "
                             f"another folder with --home.")
        os.ftruncate(fd, 0)
        os.pwrite(fd, f"{os.getpid()}\n".encode(), 0)
        self._lock_fd = fd

    def release_lock(self) -> None:
        if self._lock_fd is not None:
            try:
                os.close(self._lock_fd)  # closing the file gives the lock up
            finally:
                self._lock_fd = None

    def close(self) -> None:
        self.release_lock()

    def __del__(self):  # a daemon object dropped without stop_threads() (tests) must not keep the lock
        try:
            self.release_lock()
        except Exception:
            pass

    def _on_lost(self, s: dict) -> None:
        self.db.event("gpu_lost", s.get("error", ""), gpu=s["gpu"])  # the watchdog acts on it

    # ---- startup -----------------------------------------------------------------------
    def check_inventory(self) -> list[str]:
        """Compare the detected cards (by UUID) with the last run. State such as a quarantine
        follows the physical card, not its index, if cards were added, removed or reordered."""
        new = {str(g["gpu"]): {"uuid": g.get("uuid"), "name": g.get("name"), "vendor": g.get("vendor"),
                               "local": g.get("local"), "integrated": bool(g.get("integrated")),
                               "trainable": g.get("trainable", True)} for g in self.inventory}
        old = json.loads(self.db.kv_get("gpu_inventory") or "null")
        self.db.kv_set("gpu_inventory", json.dumps(new))
        if old is None or {i: v.get("uuid") for i, v in old.items()} == {i: v["uuid"] for i, v in new.items()}:
            return []
        old_state = {int(i): self.db.gpu_state(int(i)) for i in old}
        where_was = {v["uuid"]: int(i) for i, v in old.items()}
        lines = []
        new_uuids = {v["uuid"] for v in new.values()}
        for i, v in old.items():
            if v["uuid"] not in new_uuids:
                lines.append(f"GPU {i} ({short_name(v['name'])}) is gone.")
        for i, v in new.items():
            if old.get(i, {}).get("uuid") == v["uuid"]:
                continue
            prev = where_was.get(v["uuid"])
            carried = old_state.get(prev) if prev is not None else None
            self.db.set_gpu_state(int(i), quarantined=carried["quarantined"] if carried else 0,
                                  quarantine_reason=carried["quarantine_reason"] if carried else None,
                                  cooling=0)
            self.db.kv_set(f"ecc_baseline:{i}", None)
            lines.append(f"GPU {i} is now {short_name(v['name'])}" + (f" (was GPU {prev})." if prev is not None
                                                                         else " (new card)."))
        for i in old:
            if i not in new:
                self.db.execute("DELETE FROM gpu_state WHERE gpu=?", (int(i),))
        return lines

    def startup(self) -> list[str]:
        self.acquire_lock()
        samples = self.collector.collect_once()
        problems = gpu_identity_problems(self.cfg, samples)
        if problems:
            raise SystemExit("Refusing to start: " + "; ".join(problems))
        changes = self.check_inventory()
        try:  # jobs that kept running follow their cards to the cards' new numbers
            changes += self.runner.remap_gpus(self.inventory)
        except Exception:
            log.exception("could not match running jobs to the detected cards")
        if changes:
            self.notifier.alert("🔧 GPU inventory changed since the last start: " + " ".join(changes))
        if not self.cfg.gpus.training:
            log.warning("no training GPUs: monitoring only, queued jobs will not start")
        by_gpu = {s["gpu"]: s for s in samples}
        blind = [g for g in self.cfg.gpus.training
                 if g in by_gpu and not by_gpu[g].get("lost") and by_gpu[g].get("temp_c") is None]
        if blind and self.db.kv_get("blind_gpus_told") != json.dumps(blind):
            self.db.kv_set("blind_gpus_told", json.dumps(blind))  # say it once, not on every restart
            self.notifier.alert(
                "⚠️ No temperature reading from " + ", ".join(f"{self.store.label(g)} (GPU {g})" for g in blind)
                + ": heat protection (warn, pause, emergency stop) cannot work for "
                + ("this card" if len(blind) == 1 else "these cards")
                + ". Out-of-memory, stall and checkpoint protection still work.")
        if self.cfg.telegram.digest_time and self.db.kv_get("last_digest") is None:
            # first start: the first digest goes out tomorrow, not the moment you install
            self.db.kv_set("last_digest", time.strftime("%Y-%m-%d", time.localtime(self.clock())))
        lines = self.runner.reconcile()
        stamp = time.strftime("%H:%M", time.localtime(self.clock()))
        settle = f" Launching resumes after a {self.cfg.runner.settle_s / 60:g}-minute settle." if any(
            ln.startswith("Resuming") for ln in lines) else ""
        cards = ", ".join(f"{self.store.label(g)} (GPU {g})" for g in self.cfg.gpus.training) or "none"
        self.notifier.alert(f"🟢 Rig Guardian started at {stamp}. Training GPUs: {cards}. "
                            + (" ".join(lines) if lines else "No jobs were running.") + settle)
        self.db.event("startup", " ".join(lines))
        return lines

    def start_threads(self) -> None:
        self.collector.start()
        self.threads = [
            self.collector,
            LoopThread("watchdog", self.watchdog.tick, self.cfg.telemetry.interval_s),
            LoopThread("runner", self.runner.tick, self.cfg.runner.poll_s),
            LoopThread("maintenance", self.maintenance_tick, 5.0),
        ]
        for th in self.threads[1:]:
            th.start()

    def stop_threads(self) -> None:
        for th in self.threads:
            th.stop_event.set()
        for th in self.threads:
            th.join(timeout=5)
        self.collector.flush()
        self.db.kv_set("agent_heartbeat", None)  # a clean stop: the CLI says "not running" at once
        self.release_lock()

    # ---- maintenance -----------------------------------------------------------------
    def maintenance_tick(self) -> None:
        now = self.clock()
        self.db.kv_set("agent_heartbeat", str(time.time()))  # the CLI reads it to tell if the agent is up
        self.notifier.flush()
        self._check_loops()
        today = time.strftime("%Y-%m-%d", time.localtime(now))
        dt = self.cfg.telegram.digest_time
        if dt and self.db.kv_get("last_digest") != today and now >= today_at(now, dt):
            self.db.kv_set("last_digest", today)
            self.notifier.report(self.reports.digest_text())
        if self.db.kv_get("last_prune") != today and time.localtime(now).tm_hour >= 3:
            self.db.kv_set("last_prune", today)
            n = self.db.prune_telemetry(now - self.cfg.telemetry.history_days * 86400)
            self.db.execute("DELETE FROM outbox WHERE sent_at IS NOT NULL AND sent_at < ?", (now - 30 * 86400,))
            m = self.db.prune_log(now - self.cfg.telemetry.events_days * 86400)
            self.db.event("prune", f"{n} telemetry rows, {m} log rows")
        hc = self.cfg.healthcheck
        if hc.url and now - self._last_health >= hc.every_s:
            self._last_health = now
            stuck = self.stuck_loops()
            if stuck:  # "alive" means the safety loops run, not just this thread
                log.error("no healthcheck ping: %s not running", ", ".join(stuck))
                return
            try:
                import httpx

                httpx.get(hc.url, timeout=10, trust_env=True)
            except Exception as e:
                log.info("healthcheck ping failed: %s", type(e).__name__)

    def stuck_loops(self, limit_s: float = 60.0) -> list[str]:
        """Names of worker loops that have not completed a round for ``limit_s`` (hung driver
        call, deadlock) or that have died."""
        out = []
        mono = time.monotonic()
        for th in self.threads:
            last = getattr(th, "last_ok", None)
            if not th.is_alive() or (last is not None and mono - last > max(limit_s, 10 * th.interval_s)):
                out.append(th.name)
        return out

    def _check_loops(self) -> None:
        stuck = [n for n in self.stuck_loops() if n != "maintenance"]
        if stuck and self.notifier.rate_ok("loops_stuck"):
            self.notifier.alert(f"⚠️ Rig Guardian's own {' and '.join(stuck)} loop has not run for over a minute "
                                f"(a hung driver call?). Protection that depends on it is not working. "
                                f"Check the rig; restarting the agent does not stop training: "
                                f"systemctl --user restart rig-guardian")

    # ---- main loop ---------------------------------------------------------------------
    async def run(self) -> None:
        from guardian.bot import TelegramBot, console_send_loop

        self.startup()
        self.start_threads()
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop.set)
        token = os.environ.get("TELEGRAM_TOKEN")
        bot, console = None, None
        if token:
            from telegram.error import InvalidToken

            bot = TelegramBot(token, self.agent, self.notifier, self.cfg.telegram.allowed_user_ids,
                              max_message_age_s=self.cfg.telegram.max_message_age_s)
            while not stop.is_set():
                try:
                    await bot.start()
                    break
                except InvalidToken:
                    # never log the exception itself: its text contains the token
                    log.error("Telegram rejected the bot token in %s/.env (TELEGRAM_TOKEN). Get the token again "
                              "from @BotFather and restart. Monitoring and protection keep running; "
                              "messages go to this log.", self.cfg.root)
                    bot = None
                    console = asyncio.create_task(console_send_loop(self.notifier))
                    break
                except Exception as e:  # no internet at boot: safety keeps running, retry
                    log.warning("telegram start failed (%s); retrying in 60 s", redact(str(e)))
                    try:
                        await asyncio.wait_for(stop.wait(), 60)
                    except asyncio.TimeoutError:
                        pass
        else:
            log.warning("TELEGRAM_TOKEN not set; messages are written to this log")
            console = asyncio.create_task(console_send_loop(self.notifier))
        log.info("rig guardian running (launcher: %s)", self.runner.launcher.name)
        await stop.wait()
        log.info("stopping (training jobs keep running)")
        if bot and bot.app:
            await bot.stop()
        if console:
            console.cancel()
        self.stop_threads()


def start(cfg: Config) -> None:
    """Run the daemon in the foreground until SIGTERM/SIGINT (systemd runs exactly this)."""
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    load_env(Path(cfg.root) / ".env")
    try:
        d = Daemon(cfg)
    except (ValueError, NoGpuError) as e:
        log.error("%s", e)
        sys.exit(2)
    log.info("GPUs:\n%s", inventory_table(d.inventory, cfg))
    try:
        asyncio.run(d.run())
    except SystemExit as e:
        log.error("%s", e)
        sys.exit(2)

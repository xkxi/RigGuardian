"""Rig Guardian training SDK: heartbeats, metrics, checkpoints, resume, OOM recovery and
pause/stop handling for a normal PyTorch loop, in about ten added lines.

    from guardian_sdk import Guardian

    g = Guardian()                       # reads env vars and run_dir/config.json
    cfg = g.config
    model = model.to(g.device)           # cuda (NVIDIA, AMD/ROCm), xpu (Intel), mps (Apple) or cpu
    scaler = g.grad_scaler()             # active only where it is needed (fp16 on CUDA)
    state = g.resume(model=model, optimizer=opt, scheduler=sched, scaler=scaler)
    sampler = g.sampler(train_set, seed=cfg.seed)
    loader = DataLoader(train_set, batch_size=cfg.micro_batch, sampler=sampler)
    for epoch in range(state.epoch, cfg.epochs):
        sampler.set_epoch(epoch)
        for batch in g.batches(loader):
            with g.step(batch_size=cfg.micro_batch) as s:
                with g.autocast():
                    loss = model(batch) / cfg.grad_accum
                scaler.scale(loss).backward()
                if s.optimizer_step:      # every grad_accum micro-batches, and on the epoch's last one
                    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            g.log(loss=loss.item() * cfg.grad_accum)
            if s.optimizer_step:
                g.maybe_checkpoint()      # only right after an optimizer step
        val = evaluate(model)             # a long evaluation: `for b in g.watch(val_loader)`
        g.log(val_loss=val, epoch=epoch)
        g.end_epoch(metric=val)           # marks the epoch complete and saves
    g.finish()

Talks to the daemon only through files in the run folder (``heartbeat``, ``metrics.jsonl``,
``control.json``, ``exit_intent``) and Unix signals (SIGUSR1 = checkpoint now,
SIGTERM = checkpoint and exit). No sockets, so nothing here can block training.

Works without the daemon too: with no GUARDIAN_RUN_DIR set it uses ./guardian_run.

Checkpoints are loaded with ``torch.load(weights_only=False)`` (they hold optimizer and
random-generator state), so only resume from run folders you wrote yourself.
"""

from __future__ import annotations

import atexit
import contextlib
import gc
import hashlib
import json
import math
import os
import random
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from guardian.checkpoints import CheckpointStore, disk_free_bytes
from guardian.fsutil import atomic_write_json, atomic_write_text, read_json

try:
    import torch
except ImportError:  # the SDK is useless without torch, but importing it should not crash
    torch = None  # type: ignore

EXIT_STOPPED = 86
EXIT_OOM = 87

_DEFAULTS = {
    "every_steps": 500,
    "every_minutes": 20.0,
    "keep_last": 3,
    "keep_best": 1,
    "min_free_disk_gb": 2.0,  # used standalone; under the agent, checkpoints.min_free_disk_gb applies
    "async": True,
    "probe_steps": 20,
    "headroom_frac": 0.10,
    "best_metric": "val_loss",
    "mode": "min",
    "heartbeat_s": 10.0,
    "metrics_every": 10,
    "launch_request_id": 0,
    "config_hash": None,
}

# flag slots combined across DDP ranks with an all-reduce(max)
_F_CKPT, _F_PAUSE, _F_STOP, _F_PERIODIC, _F_PROBE, _F_BEST = range(6)


class RunConfig(dict):
    """The exact settings for this run; ``cfg.lr`` and ``cfg["lr"]`` both work."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"run config has no setting {key!r}") from None


@dataclass
class ResumeState:
    epoch: int = 0
    step: int = 0
    samples_seen: int = 0
    best_metric: float | None = None
    resumed: bool = False
    checkpoint: str | None = None


def _is_oom(e: BaseException) -> bool:
    """Out of GPU memory on any backend: CUDA and ROCm raise torch.OutOfMemoryError (older
    builds: torch.cuda.OutOfMemoryError); MPS and XPU raise a RuntimeError that says so."""
    if torch is not None:
        kinds = tuple(k for k in (getattr(torch, "OutOfMemoryError", None),
                                  getattr(torch.cuda, "OutOfMemoryError", None)) if isinstance(k, type))
        if kinds and isinstance(e, kinds):
            return True
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


def detect_backend() -> str:
    """The PyTorch device type this process trains on: 'cuda' (NVIDIA, and AMD cards through
    a ROCm build), 'xpu' (Intel), 'mps' (Apple Silicon) or 'cpu'."""
    if torch is None:
        return "cpu"
    probes = (
        ("cuda", lambda: torch.cuda.is_available()),
        ("xpu", lambda: torch.xpu.is_available()),
        ("mps", lambda: torch.backends.mps.is_available()),
    )
    for name, available in probes:
        try:
            if available():
                return name
        except Exception:  # a backend this build does not have
            continue
    return "cpu"


def _to_cpu(obj: Any) -> Any:
    """Deep copy of a state tree with every tensor moved to CPU memory (the snapshot)."""
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)
    return obj


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


class ResumableSampler:
    """Shuffles reproducibly per epoch and skips the examples already seen this epoch.

    Skipping is just slicing the index list, so no data is loaded to skip it. Under DDP
    each rank takes every ``world``-th index of the same global order.
    """

    def __init__(self, guardian: "Guardian", n: int, seed: int = 0, shuffle: bool = True):
        self.g, self.n, self.seed, self.shuffle = guardian, int(n), int(seed), shuffle
        self._pass: tuple[int, int] | None = None  # (epoch, number of indexes) of the pass being read
        if guardian._sampler is not None:
            guardian._say("a second g.sampler() replaces the first as the training sampler. g.sampler() is "
                          "for the training set only; give validation data a normal DataLoader.")
        guardian._sampler = self
        guardian._tracks_epochs = True
        guardian.epoch_size = self.n

    @property
    def epoch(self) -> int:
        return self.g.epoch

    def set_epoch(self, epoch: int) -> None:
        self.g.start_epoch(epoch)
        self._pass = None

    def _order(self) -> list[int]:
        if not self.shuffle:
            return list(range(self.n))
        # a seed of its own for every (seed, epoch): seed 0 at epoch 1 must not repeat seed 1 at epoch 0
        mixed = int.from_bytes(hashlib.sha256(f"{self.seed}:{self.g.epoch}".encode()).digest()[:8], "big") >> 1
        if torch is not None:
            gen = torch.Generator().manual_seed(mixed)
            return torch.randperm(self.n, generator=gen).tolist()
        order = list(range(self.n))
        random.Random(mixed).shuffle(order)
        return order

    def _indices(self) -> list[int]:
        order = self._order()
        order = order[min(self.g.samples_epoch, self.n):]
        w = self.g.world
        if w > 1:
            order = order[: len(order) - len(order) % w][self.g.rank :: w]
        return order

    def __iter__(self) -> Iterator[int]:
        g = self.g
        if g._pass_done:  # the last pass ran to its end and set_epoch() was not called: next epoch
            g._enter_epoch(g.epoch + 1)
        g._reapply_rng()
        idx = self._indices()
        self._pass = (g.epoch, len(idx))
        g._pass_len, g._pass_samples = len(idx), 0

        def run() -> Iterator[int]:
            yield from idx
            g._pass_done = True  # (a DataLoader reads ahead, so this can be a few batches early)

        return run()

    def __len__(self) -> int:
        # constant while an epoch is being read: progress bars and `i + 1 == len(loader)` rely on it
        if self._pass is not None and self._pass[0] == self.g.epoch:
            return self._pass[1]
        return len(self._indices())


class Guardian:
    def __init__(self, run_dir: str | os.PathLike | None = None, config: dict | None = None):
        self.job_id = os.environ.get("GUARDIAN_JOB_ID", "local")
        self.managed = "GUARDIAN_RUN_DIR" in os.environ
        self.run_dir = Path(run_dir or os.environ.get("GUARDIAN_RUN_DIR") or "guardian_run").expanduser()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config = RunConfig(config if config is not None else read_json(self.run_dir / "config.json", {}))
        s = dict(_DEFAULTS)
        s.update(read_json(self.run_dir / "guardian.json", {}) or {})
        self.settings = s
        self.config_hash: str | None = s.get("config_hash")
        self.mode = s["mode"]

        self.rank = int(os.environ.get("RANK", 0))
        self.world = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.is_main = self.rank == 0
        self.grad_accum = max(1, int(self.config.get("grad_accum", 1) or 1))
        self.micro_batch = self.config.get("micro_batch")

        self.store = CheckpointStore(self.run_dir / "checkpoints")
        if self.is_main:
            self.store.ensure()

        # position in training; samples seen (not batch index) marks the data position
        self.step_count = 0  # optimizer steps
        self.micro = 0  # micro-batch steps
        self.epoch = 0
        self.samples_seen = 0
        self.samples_epoch = 0
        self.epoch_size: int | None = None
        self._boundary = (0, 0, 0, 0)  # counters at the last optimizer-step boundary
        self._epoch_micro = 0  # micro-batches trained in this epoch
        self._pass_len: int | None = None  # indexes the SDK sampler handed out for this pass (this rank)
        self._pass_samples = 0  # of those, how many this rank has trained on
        self._pass_done = False  # the current epoch's data has been read to the end
        self._in_batches = False  # g.batches() is driving the loop
        self._last_batch = False  # ... and the batch in hand is the epoch's last
        self._opt_step = False  # the current (or latest) micro-batch completes an optimizer step
        self._uses_opt_flag = False  # the loop reads g.optimizer_step
        self._epoch_ended = False  # end_epoch() has been called for self.epoch
        self._tracks_epochs = False  # a sampler, g.batches() or g.start_epoch() is in use
        self._rng_pending: Any = None  # generator state to re-apply when data loading restarts after a resume
        self._warned: set[str] = set()
        self._sampler: ResumableSampler | None = None
        self._objs: dict[str, Any] = {}
        self._best: float | None = None

        # requests
        self._handled_rid = int(s.get("launch_request_id") or 0)
        self._control_key: tuple | None = None
        self._pending: tuple[str, int] | None = None
        self._sig_ckpt = False
        self._sig_stop = False
        self._paused = False

        # checkpoint bookkeeping
        self._last_ckpt_time = time.monotonic()
        self._last_ckpt_step = 0
        self._writer: threading.Thread | None = None
        self._writer_error: str | None = None
        self._disk_low = False
        self._request_failed: int | None = None  # ID of a checkpoint request that could not be saved

        # probe (first N steps of each launch)
        self._probe_count = 0
        self._probe_done = False
        self._probe_restart = False
        self._peak_mem = 0
        self._mem_total = 0
        self._oom_reason: str | None = None
        self._oom_saved: bool | None = None

        self._metrics_buf: list[dict] = []
        self._last_hb = 0.0
        self._backend: str | None = None
        self._launch_micro = 0  # micro-batches completed since this process started
        self._progress_ts = time.time()  # last training step, beat() or save
        self._phase = "start"  # start (no step yet) | train | eval (between epochs)
        self._pid = os.getpid()
        self._hb_stop = threading.Event()
        self._hb_thread: threading.Thread | None = None

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGUSR1, self._on_sigusr1)
            signal.signal(signal.SIGTERM, self._on_sigterm)
        if self.managed:
            atomic_write_text(self.run_dir / f"rank{self.rank}.pid", f"{os.getpid()}\n", durable=False)
        self._check_control()
        self._heartbeat(force=True)
        if self.is_main:
            atexit.register(self._at_exit)
            # liveness: the heartbeat keeps coming while the process is alive, whatever the main
            # thread is doing (loading data, evaluating, compiling). Progress is reported apart.
            self._hb_thread = threading.Thread(target=self._hb_loop, name="guardian-heartbeat", daemon=True)
            self._hb_thread.start()

    def _hb_loop(self) -> None:
        every = max(0.05, float(self.settings["heartbeat_s"]))
        while not self._hb_stop.wait(every):
            if os.getpid() != self._pid:
                return  # a forked data-loader worker
            try:
                self._heartbeat(force=True)
            except Exception:  # a full disk must not kill the thread; the next round tries again
                pass

    def _at_exit(self) -> None:
        self._hb_stop.set()
        try:
            self._flush_metrics()  # the last rows before a crash are the interesting ones
        except Exception:
            pass

    # ---- device and precision --------------------------------------------------------
    @property
    def backend(self) -> str:
        """'cuda' (NVIDIA, or AMD through ROCm), 'xpu' (Intel), 'mps' (Apple) or 'cpu'."""
        if self._backend is None:
            self._backend = detect_backend()
        return self._backend

    @property
    def device(self):
        """The device this process should train on. Under multi-GPU (DDP) launches, each
        process gets its own card: the agent has already limited the visible cards to the
        job's, so the local rank is the index."""
        if self.backend in ("cuda", "xpu") and self.world > 1:
            return torch.device(self.backend, self.local_rank)
        return torch.device(self.backend)

    @property
    def amp_dtype(self):
        """Mixed-precision dtype for this card: bfloat16 where the hardware supports it
        (NVIDIA Ampere and newer, Intel, CPU), else float16 (NVIDIA Volta/Turing/Pascal, Apple)."""
        if torch is None:
            return None
        if self.backend == "cuda":
            try:  # native bf16 only: PyTorch otherwise answers "yes" for any card that can hold a bf16 tensor
                native = torch.cuda.is_bf16_supported(including_emulation=False)
            except TypeError:  # older PyTorch: the check was hardware-only already
                native = torch.cuda.is_bf16_supported()
            return torch.bfloat16 if native else torch.float16
        if self.backend == "mps":
            return torch.float16
        return torch.bfloat16

    @property
    def use_grad_scaler(self) -> bool:
        """fp16 on CUDA needs loss scaling; bf16 does not."""
        return torch is not None and self.backend == "cuda" and self.amp_dtype == torch.float16

    def autocast(self, enabled: bool = True):
        """``torch.autocast`` for this device with the right dtype; a no-op on CPU."""
        return torch.autocast(self.backend, dtype=self.amp_dtype, enabled=enabled and self.backend != "cpu")

    def grad_scaler(self):
        """A ``GradScaler`` that is active only where it is needed (fp16 on CUDA)."""
        return torch.amp.GradScaler("cuda" if self.backend == "cuda" else "cpu", enabled=self.use_grad_scaler)

    def _mem_peak_total(self) -> tuple[int, int] | None:
        """(peak bytes in use on the device: ours plus other programs', the device's total bytes),
        where the backend can tell."""
        try:
            if self.backend == "cuda":
                peak = int(torch.cuda.max_memory_reserved())
                try:  # memory other programs hold on this card is not available to us either
                    free, total = torch.cuda.mem_get_info()
                    others = max(0, int(total) - int(free) - int(torch.cuda.memory_reserved()))
                    return peak + others, int(total)
                except Exception:
                    return peak, int(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory)
            if self.backend == "xpu":
                return (int(torch.xpu.max_memory_reserved()),
                        int(torch.xpu.get_device_properties(torch.xpu.current_device()).total_memory))
            if self.backend == "mps":
                return int(torch.mps.driver_allocated_memory()), int(torch.mps.recommended_max_memory())
        except Exception:  # an older build without these calls: no probe, the OOM handler still works
            return None
        return None

    def _empty_cache(self) -> None:
        try:
            if self.backend == "cuda":
                torch.cuda.empty_cache()
            elif self.backend == "xpu":
                torch.xpu.empty_cache()
            elif self.backend == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass

    # ---- signals and control file ----------------------------------------------------
    def _on_sigusr1(self, signum, frame) -> None:
        self._sig_ckpt = True

    def _on_sigterm(self, signum, frame) -> None:
        self._sig_stop = True

    def _check_control(self) -> None:
        """One cheap stat per step; parse control.json only when it changed."""
        p = self.run_dir / "control.json"
        try:
            st = p.stat()
        except OSError:
            return
        key = (st.st_mtime_ns, st.st_size, st.st_ino)
        if key == self._control_key:
            return
        self._control_key = key
        c = read_json(p)
        if not isinstance(c, dict):
            return
        rid = int(c.get("request_id") or 0)
        action = c.get("action")
        if rid > self._handled_rid and action in ("checkpoint", "pause", "resume", "stop"):
            self._pending = (action, rid)

    def _ack(self, rid: int | None) -> None:
        if rid is not None and rid > self._handled_rid:
            self._handled_rid = rid
        self._heartbeat(force=True)

    # ---- resume ----------------------------------------------------------------------
    def resume(self, model=None, optimizer=None, scheduler=None, scaler=None) -> ResumeState:
        """Load the newest verified checkpoint into these objects, or start fresh."""
        self._objs = {"model": model, "optimizer": optimizer, "scheduler": scheduler, "scaler": scaler}
        man, failures = self.store.find_latest_valid(self.config_hash)
        for name, why in failures:
            self._say(f"skipped checkpoint {name}: {why}")
            if self.is_main and ("SHA-256" in why or why.startswith("size ")):
                # damaged: set it aside, so that reaching its step again writes a fresh file
                # instead of trusting the one already there
                self.store.set_aside(name)
        if man is None:
            return ResumeState()
        path = self.store.dir / man["file"]
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if self.config_hash and ckpt.get("config_hash") not in (None, self.config_hash):
            raise RuntimeError(f"{path} was saved for a different config; refusing to load it")
        m = self._unwrap(model)
        if m is not None:
            m.load_state_dict(ckpt["model"])
        for key, obj in (("optimizer", optimizer), ("scheduler", scheduler)):
            if obj is not None and ckpt.get(key) is not None:
                obj.load_state_dict(ckpt[key])
        if scaler is not None and ckpt.get("scaler"):
            try:  # absent or mismatched when a job resumes on a card with different precision support
                scaler.load_state_dict(ckpt["scaler"])
            except (RuntimeError, KeyError, ValueError) as e:
                self._say(f"grad scaler state not restored ({e}); starting a fresh scale")
        rng = ckpt.get("rng") or {}
        if self.rank > 0:  # every rank has its own random state; rank 0's is in the checkpoint itself
            rng = {}
            mine = self.store.rng_path(int(man["step"]), self.rank)
            try:
                saved = torch.load(mine, map_location="cpu", weights_only=False)
                if saved.get("world") == self.world:
                    rng = saved.get("rng") or {}
            except Exception:
                pass  # none saved (older checkpoint, different number of GPUs): keep this launch's seeds
        self._restore_rng(rng)

        c = ckpt["counters"]
        self.step_count = int(c["step"])
        self.epoch = int(c["epoch"])
        self.samples_seen = int(c["samples_seen"])
        self.samples_epoch = int(c["samples_epoch"])
        self._epoch_micro = int(c.get("epoch_micro") or 0)
        self._epoch_ended = False
        if c.get("epoch_done"):
            self.epoch += 1
            self.samples_epoch = 0
            self._epoch_micro = 0
        elif rng.get("torch") is not None:
            # The uninterrupted run was in the middle of this epoch's data. Creating the
            # DataLoader iterator again draws a number from the generator; put the saved
            # state back right after that, so everything that follows matches.
            self._rng_pending = rng["torch"]
        self.micro = self.step_count * self.grad_accum
        self._boundary = (self.step_count, self.epoch, self.samples_seen, self.samples_epoch)
        self._best = ckpt.get("best_metric")
        self._last_ckpt_step = self.step_count
        self._say(
            f"resumed from {man['file']} (step {self.step_count}, epoch {self.epoch}, "
            f"{self.samples_epoch} samples into the epoch)"
        )
        self._heartbeat(force=True)
        return ResumeState(self.epoch, self.step_count, self.samples_seen, self._best, True, man["file"])

    def _reapply_rng(self) -> None:
        st, self._rng_pending = self._rng_pending, None
        if st is not None and torch is not None:
            torch.set_rng_state(st)

    def sampler(self, dataset_or_len, seed: int = 0, shuffle: bool = True) -> ResumableSampler:
        """A sampler for the TRAINING set that can continue in the middle of an epoch."""
        n = dataset_or_len if isinstance(dataset_or_len, int) else len(dataset_or_len)
        return ResumableSampler(self, n, seed=seed, shuffle=shuffle)

    def start_epoch(self, epoch: int) -> None:
        """Tell the SDK which epoch begins (``sampler.set_epoch`` does this for you). Needed for
        ``state.epoch`` to be right after a resume when you use neither ``g.sampler`` nor
        ``g.batches``."""
        self._tracks_epochs = True
        self._enter_epoch(int(epoch))

    def _enter_epoch(self, epoch: int) -> None:
        if self.micro % self.grad_accum and not self._uses_opt_flag:
            self._warn("leftover", f"the epoch ended with {self.micro % self.grad_accum} micro-batch(es) of "
                       f"gradient accumulated but not applied (the batch count is not a multiple of "
                       f"grad_accum). Step the optimizer on `g.optimizer_step` and loop over "
                       f"`g.batches(loader)` so the epoch's last window is applied too.")
        if epoch != self.epoch:
            self.epoch = epoch
            self.samples_epoch = 0
            self._epoch_micro = 0
            self._epoch_ended = False
        self._pass_done = False
        self.micro = self.step_count * self.grad_accum  # realign accumulation with the loop's i
        self._boundary = (self.step_count, self.epoch, self.samples_seen, self.samples_epoch)

    def batches(self, loader):
        """Iterate the training loader for one epoch: ``for batch in g.batches(loader)``.

        It knows which batch is the epoch's last (so ``g.optimizer_step`` can close the final,
        shorter accumulation window), counts epochs by itself, keeps the heartbeat fresh, and,
        if you do not use ``g.sampler``, skips the batches this epoch already trained on after a
        resume (by reading and dropping them).
        """
        self._tracks_epochs = True
        if self._pass_done:  # the previous pass ran to its end: this is the next epoch
            self._enter_epoch(self.epoch + 1)
        it = iter(loader)
        self._reapply_rng()
        self._in_batches = True
        try:
            skip = 0
            if self._sampler is None and self.samples_epoch > 0:
                per = (int(self.micro_batch) if isinstance(self.micro_batch, int) and self.micro_batch > 0
                       else 0) * self.world
                skip = self.samples_epoch // per if per else self._epoch_micro
            cur = None
            try:
                for _ in range(skip):
                    next(it)
                    self.beat()
                cur = next(it)
            except StopIteration:
                self._pass_done = True  # nothing left of this epoch (resumed right at its end)
                self._phase = "eval"
                return
            for nxt in it:
                yield cur
                cur = nxt
            self._last_batch = True
            yield cur
            self._pass_done = True
        finally:
            self._last_batch = False
            self._in_batches = False

    def watch(self, iterable):
        """Iterate anything that is not training (a validation loader, preprocessing) while
        telling the agent the job is alive and working: ``for batch in g.watch(val_loader)``."""
        for item in iterable:
            self.beat()
            yield item
        self.beat()

    @property
    def optimizer_step(self) -> bool:
        """True when the micro-batch in hand (or the one just finished) completes an optimizer
        step: every ``grad_accum`` micro-batches, and on the epoch's last batch, so no
        accumulated gradient is left over at the end of an epoch."""
        self._uses_opt_flag = True
        return self._opt_step

    def _is_last(self, bs: int) -> bool:
        if self._in_batches:
            return self._last_batch
        # with only the SDK sampler: the last batch is the one that uses up the indexes handed out
        return self._sampler is not None and self._pass_len is not None and self._pass_samples + bs >= self._pass_len

    # ---- the step wrapper ------------------------------------------------------------
    @contextlib.contextmanager
    def step(self, batch_size: int | None = None):
        """Wrap one micro-batch (forward, backward and, when due, the optimizer step)."""
        self._check_control()
        bs = int(batch_size if batch_size is not None else (self.micro_batch or 1))
        last = self._is_last(bs)
        self._opt_step = (self.micro + 1) % self.grad_accum == 0 or last
        oom = False
        try:
            yield self
        except BaseException as e:
            if not _is_oom(e):
                raise
            oom = True
            traceback.clear_frames(e.__traceback__)  # release the activations the frames hold
            del e
        if oom:
            self._handle_oom()
        self.micro += 1
        self._launch_micro += 1
        self._epoch_micro += 1
        self._pass_samples += bs
        self.samples_seen += bs * self.world
        self.samples_epoch += bs * self.world
        # the epoch's last, shorter window counts as a step only for loops that apply it
        if self.micro % self.grad_accum == 0 or (last and self._uses_opt_flag):
            self.step_count += 1
            self.micro = self.step_count * self.grad_accum
            self._boundary = (self.step_count, self.epoch, self.samples_seen, self.samples_epoch)
        self._progress_ts = time.time()
        self._phase = "eval" if last else "train"
        self._probe_tick()
        self._heartbeat(force=self._launch_micro == 1)  # the first step ends the start-up allowance

    def _probe_tick(self) -> None:
        if self._probe_done:
            return
        self._probe_count += 1
        mem = self._mem_peak_total() if torch is not None else None
        if mem:
            self._peak_mem = max(self._peak_mem, mem[0])
        if self._probe_count < int(self.settings["probe_steps"]):
            return
        self._probe_done = True
        if mem and mem[1] > 0:
            self._mem_total = mem[1]
            free_frac = 1 - self._peak_mem / self._mem_total
            mb = self.micro_batch
            # only where the agent can act on it: it halves the micro-batch, which must be even
            if free_frac < float(self.settings["headroom_frac"]) and isinstance(mb, int) and mb >= 2 and mb % 2 == 0:
                self._say(
                    f"probe: peak {self._peak_mem / 2**30:.1f} of {self._mem_total / 2**30:.1f} GiB "
                    f"leaves {free_frac:.0%} free; restarting with a smaller micro-batch"
                )
                self._probe_restart = True
                return
        self._heartbeat(force=True)

    def _handle_oom(self) -> None:
        opt, model = self._objs.get("optimizer"), self._unwrap(self._objs.get("model"))
        for obj in (opt, model):
            if obj is not None:
                obj.zero_grad(set_to_none=True)
        gc.collect()
        if torch is not None:
            self._empty_cache()
        # gradients half-accumulated since the last optimizer step are gone; go back to it
        oom_step = self.step_count
        self.step_count, self.epoch, self.samples_seen, self.samples_epoch = self._boundary
        self._oom_reason = "train"
        self._say(f"out of GPU memory at step {oom_step}; "
                  + (f"saving step {self.step_count} and " if self.world == 1 else "") + f"exiting {EXIT_OOM}")
        self._oom_saved = False
        if self.world == 1:  # under DDP another rank may sit in a collective; rely on the last save
            try:
                self._oom_saved = bool(self._save("oom", wait=True, ddp=False))
            except Exception as e:  # still exit 87; the runner resumes from the previous checkpoint
                self._say(f"OOM checkpoint failed: {e!r}")
        self._exit(EXIT_OOM)

    # ---- metrics ---------------------------------------------------------------------
    def log(self, **metrics: Any) -> None:
        if not self.is_main:
            return
        rec: dict[str, Any] = {"ts": round(time.time(), 3), "step": self.step_count, "epoch": self.epoch,
                               "samples_seen": self.samples_seen}
        for k, v in metrics.items():
            if hasattr(v, "item"):
                try:
                    v = v.item()
                except Exception:  # not a single number (a whole tensor or array)
                    v = v.tolist() if hasattr(v, "tolist") else str(v)
            if isinstance(v, float) and not math.isfinite(v):
                v = str(v)  # "nan" / "inf": bare NaN is not valid JSON, and a diverged run should show it
            rec[k] = v
        self._metrics_buf.append(rec)
        if len(self._metrics_buf) >= int(self.settings["metrics_every"]) or any(
            k.startswith("val") for k in metrics
        ):
            self._flush_metrics()

    def _flush_metrics(self) -> None:
        if not self._metrics_buf or not self.is_main:
            return
        rows, self._metrics_buf = self._metrics_buf, []
        lines = []
        for rec in rows:
            try:
                lines.append(json.dumps(rec, default=str, allow_nan=False))
            except (TypeError, ValueError):
                lines.append(json.dumps({k: (v if isinstance(v, (int, str, bool, type(None))) or _finite(v)
                                             else str(v)) for k, v in rec.items()}))
        with open(self.run_dir / "metrics.jsonl", "a") as f:
            f.write("".join(line + "\n" for line in lines))

    # ---- checkpoints -----------------------------------------------------------------
    def _improves(self, metric: float) -> bool:
        if not _finite(metric):
            return False  # a NaN or infinite metric is never a best
        if self._best is None:
            return True
        return metric < self._best if self.mode == "min" else metric > self._best

    def _warn(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            self._say("WARNING: " + msg)

    def _gather_flags(self, metric: float | None, force: bool) -> list[int]:
        if self.world == 1:
            self._check_control()
        if self._pending and self._pending[0] == "resume":  # not paused: nothing to do but ack
            rid = self._pending[1]
            self._pending = None
            self._ack(rid)
        pend = self._pending[0] if self._pending else None
        due = (
            self.step_count - self._last_ckpt_step >= int(self.settings["every_steps"])
            or time.monotonic() - self._last_ckpt_time >= 60 * float(self.settings["every_minutes"])
        )
        # Everything that decides WHETHER to save is decided on rank 0 and shared, because a save
        # is a collective operation: if ranks disagreed, some would wait in it for ever.
        busy = self._writer is not None and self._writer.is_alive()  # one background save at a time
        flags = [0] * 6
        flags[_F_CKPT] = int(self._sig_ckpt or pend == "checkpoint")
        flags[_F_PAUSE] = int(pend == "pause")
        flags[_F_STOP] = int(self._sig_stop or pend == "stop")
        flags[_F_PERIODIC] = int(due and self.is_main and not busy)  # rank 0's clock and writer decide
        flags[_F_PROBE] = int(self._probe_restart)
        flags[_F_BEST] = int(self.is_main and metric is not None and self._improves(float(metric)))
        dist = self._dist()
        if dist is not None:  # every rank acts on the same step
            dev = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else "cpu"
            t = torch.tensor(flags, dtype=torch.int32, device=dev)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            flags = [int(x) for x in t.tolist()]
        return flags

    def end_epoch(self, metric: float | None = None) -> bool:
        """Call once at the end of each epoch, after validation: marks the epoch complete (a
        resume then starts the next one) and saves a checkpoint, flagged best if ``metric`` is."""
        self._epoch_ended = True
        self._phase = "eval"
        return self.maybe_checkpoint(metric=metric, force=True)

    def maybe_checkpoint(self, metric: float | None = None, force: bool = False) -> bool:
        """Save when a timer, signal, request or new best says so; then obey pause or stop.

        Call only right after an optimizer step: half-accumulated gradients are not saved.
        Returns True when a checkpoint was written.
        """
        if hasattr(metric, "item"):
            metric = metric.item()
        flags = self._gather_flags(metric, force)
        pend_rid = self._pending[1] if self._pending else None
        is_best = bool(flags[_F_BEST]) and metric is not None
        if is_best:
            self._best = float(metric)
        if flags[_F_STOP]:
            reason = "stop"
        elif flags[_F_PAUSE]:
            reason = "pause"
        elif flags[_F_PROBE]:
            reason = "probe"
        elif flags[_F_CKPT]:
            reason = "request"
        elif is_best:
            reason = "best"
        elif force:
            reason = "forced"
        elif flags[_F_PERIODIC]:
            reason = "periodic"
        else:
            return False

        if (force and not self._epoch_ended and self.epoch_size
                and self.samples_epoch >= self.epoch_size - self.epoch_size % self.world):
            self._epoch_ended = True  # a forced save once the epoch's data is used up: the epoch is over
        if self.micro % self.grad_accum:
            self._warn("offstep", "a checkpoint was taken with gradients accumulated but not yet applied; "
                       "they are not part of it. Call g.maybe_checkpoint() only when g.optimizer_step is true.")
        if not self._tracks_epochs:
            self._warn("epochs", "the SDK is not being told about epochs (no g.sampler(), g.batches() or "
                       "g.start_epoch()): after a resume, state.epoch will be 0 and the data position is "
                       "not restored. Loop over `g.batches(loader)` to fix both.")
        saved = False
        try:
            saved = self._save(reason, metric=metric, is_best=is_best,
                               wait=reason in ("stop", "pause", "probe", "request"))
        except Exception as e:
            # a failed save (disk full, I/O error) must not turn a pause or a stop into a crash
            self._writer_error = f"checkpoint step {self.step_count} failed: {e!r}"
            self._say(self._writer_error)
        self._sig_ckpt = False
        if reason in ("request",) or (self._pending and self._pending[0] == "checkpoint"):
            self._request_failed = None if saved else pend_rid
            self._pending = None
            self._ack(pend_rid)
        if reason == "probe":
            self._oom_reason = "probe"
            self._exit(EXIT_OOM)
        if reason == "pause":
            self._pending = None
            self._pause_loop(pend_rid)
        if reason == "stop" or self._sig_stop:
            self._pending = None
            self._ack(pend_rid)
            self._exit(EXIT_STOPPED)
        return saved

    def _save(self, reason: str, metric: float | None = None, is_best: bool = False,
              wait: bool = True, final: bool = False, ddp: bool = True) -> bool:
        dist = self._dist() if ddp else None
        if dist is not None:
            dist.barrier()
        saved = False
        if not self.is_main and dist is not None:
            self._save_rank_rng()
        if self.is_main:
            self._flush_metrics()
            if self._writer is not None:
                self._writer.join()  # a pause or stop waits for the running save, then takes a fresh one
            saved = self._save_main(reason, metric, is_best, wait, final)
        if dist is not None:
            dist.barrier()
        return saved

    def _save_rank_rng(self) -> None:
        """Ranks other than 0 keep their own random state next to the checkpoint rank 0 writes."""
        try:
            path = self.store.rng_path(self.step_count, self.rank)
            tmp = path.with_name(path.name + ".tmp")
            torch.save({"rng": self._rng_state(), "world": self.world, "step": self.step_count}, tmp)
            os.replace(tmp, path)
        except Exception as e:  # not worth failing a save for
            self._say(f"could not save this rank's random state: {e!r}")

    def _save_main(self, reason, metric, is_best, wait, final) -> bool:
        step = self.step_count
        meta = {
            "job_id": self.job_id,
            "step": step,
            "epoch": self.epoch,
            "samples_seen": self.samples_seen,
            "reason": "final" if final else reason,
            "metric": float(metric) if _finite(metric) else None,
            "metric_name": self.settings["best_metric"],
            "epoch_done": bool(self._epoch_ended),
            "best_metric": self._best if _finite(self._best) else None,
            "is_best": bool(is_best),
            "final": bool(final),
            "config_hash": self.config_hash,
            "torch": getattr(torch, "__version__", None),
        }
        existing = self.store.manifest_for_step(step)
        same = existing is not None and (existing.get("epoch_done"), existing.get("best_metric")) == (
            meta["epoch_done"], meta["best_metric"])
        if same:  # already saved this exact step with the same state (e.g. stop right after pause)
            upd = {"reason": meta["reason"]}
            if _finite(metric):
                upd.update(metric=meta["metric"], is_best=bool(is_best or existing.get("is_best")))
            if final:
                upd["final"] = True
            self.store.update_manifest(step, **upd)
            self.store.prune(int(self.settings["keep_last"]), int(self.settings["keep_best"]), self.mode)
            self._mark_saved(step)
            return True

        if not self._disk_ok(reason):
            return False
        state = self._state_tree()
        use_async = bool(self.settings["async"]) and not wait
        if use_async:
            snapshot = _to_cpu(state)  # 1. snapshot to CPU memory; training continues after this
            self._writer = threading.Thread(target=self._write, args=(snapshot, meta), daemon=True)
            self._writer.start()
        else:
            self._write(state, meta)
            if self._writer_error:
                raise RuntimeError(self._writer_error)
        self._mark_saved(step)
        return True

    def _mark_saved(self, step: int) -> None:
        self._last_ckpt_step = step
        self._last_ckpt_time = time.monotonic()

    def _write(self, state: dict, meta: dict) -> None:
        """2-7 of the write protocol: torch.save to .tmp, fsync, checksum, rename, LATEST, prune."""
        self._writer_error = None
        tmp = self.store.tmp_path(meta["step"])
        try:
            with open(tmp, "wb") as f:
                torch.save(state, f)
                f.flush()
                os.fsync(f.fileno())
            self.store.commit(tmp, meta)
            self.store.prune(int(self.settings["keep_last"]), int(self.settings["keep_best"]), self.mode)
            self._progress_ts = time.time()
            self._heartbeat(force=True)
        except Exception as e:
            self._writer_error = f"checkpoint step {meta['step']} failed: {e!r}"
            self._say(self._writer_error)
            try:
                tmp.unlink()
            except OSError:
                pass

    def _disk_ok(self, reason: str) -> bool:
        need = int(self.store.newest_bytes() * 1.1)
        min_free = int(float(self.settings["min_free_disk_gb"]) * 2**30)
        free = disk_free_bytes(self.store.dir)
        if free - need >= min_free:
            self._disk_low = False
            return True
        # prune down to the newest + best before giving up
        self.store.prune(1, int(self.settings["keep_best"]), self.mode)
        free = disk_free_bytes(self.store.dir)
        self._disk_low = free - need < min_free
        if self._disk_low:
            self._heartbeat(force=True)
        if free - need >= min_free:
            return True
        if reason in ("stop", "pause", "oom", "final", "probe", "request") and free > need:
            self._say(f"disk low ({free / 2**30:.1f} GiB free); saving anyway because this is a {reason}")
            return True
        self._say(f"disk low ({free / 2**30:.1f} GiB free); skipped {reason} checkpoint")
        return False

    def _state_tree(self) -> dict:
        model = self._unwrap(self._objs.get("model"))
        tree: dict[str, Any] = {
            "model": model.state_dict() if model is not None else None,
            "counters": {
                "step": self.step_count,
                "epoch": self.epoch,
                "samples_seen": self.samples_seen,
                "samples_epoch": self.samples_epoch,
                "epoch_micro": self._epoch_micro,
                "epoch_done": bool(self._epoch_ended),
                "micro_batch": self.micro_batch,
                "grad_accum": self.grad_accum,
            },
            "rng": self._rng_state(),
            "best_metric": self._best,
            "job_id": self.job_id,
            "config_hash": self.config_hash,
        }
        for key in ("optimizer", "scheduler", "scaler"):
            obj = self._objs.get(key)
            tree[key] = obj.state_dict() if obj is not None else None
        return tree

    # ---- RNG -------------------------------------------------------------------------
    @staticmethod
    def _rng_state() -> dict:
        st: dict[str, Any] = {"python": random.getstate()}
        try:
            import numpy as np

            st["numpy"] = np.random.get_state()
        except ImportError:
            pass
        if torch is not None:
            st["torch"] = torch.get_rng_state()
            if torch.cuda.is_available():
                st["cuda"] = torch.cuda.get_rng_state_all()
        return st

    @staticmethod
    def _restore_rng(st: dict) -> None:
        if "python" in st:
            random.setstate(st["python"])
        if "numpy" in st:
            try:
                import numpy as np

                np.random.set_state(st["numpy"])
            except ImportError:
                pass
        if torch is not None:
            if st.get("torch") is not None:
                torch.set_rng_state(st["torch"])
            cuda = st.get("cuda")
            if cuda is not None and torch.cuda.is_available() and len(cuda) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(cuda)

    # ---- pause / stop / finish -------------------------------------------------------
    def _pause_loop(self, rid: int | None) -> None:
        """Soft pause: keep GPU memory, idle until resumed. Stop still works while paused."""
        self._paused = True
        self._say(f"paused at step {self.step_count}")
        self._ack(rid)
        while True:
            time.sleep(1.0)
            self._check_control()
            if self._sig_stop:
                break
            if self._pending:
                action, prid = self._pending
                self._pending = None
                if action == "resume":
                    self._paused = False
                    self._say(f"resumed at step {self.step_count}")
                    self._ack(prid)
                    return
                if action == "stop":
                    self._handled_rid = prid
                    break
                self._ack(prid)  # a checkpoint request while paused: this step is already saved
            self._heartbeat()
        self._save("stop", wait=True, ddp=False)  # same step as the pause: just relabels the manifest
        self._exit(EXIT_STOPPED)

    def beat(self) -> None:
        """Tell the agent this process is alive. Training steps do this by themselves; call it
        inside long loops that are not training steps (a long evaluation pass, preprocessing
        between epochs), so the stall detector does not take the silence for a hang."""
        self._progress_ts = time.time()
        self._heartbeat()

    def finish(self) -> None:
        """Write the final checkpoint and the ``done`` marker. Call once, at the very end."""
        self._epoch_ended = True
        try:
            self._save("final", wait=True, final=True)
        except Exception as e:  # training did finish; say what could not be written
            self._writer_error = f"final checkpoint failed: {e!r}"
            self._say(self._writer_error)
        if self._writer is not None:
            self._writer.join()
        if self.is_main:
            self._flush_metrics()
            atomic_write_text(self.run_dir / "done", time.strftime("%Y-%m-%dT%H:%M:%S\n"))
            self._heartbeat(force=True)

    def _exit(self, code: int) -> None:
        self._hb_stop.set()
        try:
            if self._writer is not None:
                self._writer.join()
            self._flush_metrics()
            atomic_write_json(self.run_dir / "exit_intent", {"code": code, "rank": self.rank,
                                                             "oom_reason": self._oom_reason,
                                                             "oom_saved": self._oom_saved})
            self._heartbeat(force=True)
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            # if something (a stuck worker thread) keeps the interpreter alive, force the exit
            t = threading.Timer(30.0, os._exit, args=(code,))
            t.daemon = True
            t.start()
            sys.exit(code)

    # ---- heartbeat and helpers -------------------------------------------------------
    def _heartbeat(self, force: bool = False) -> None:
        if not self.is_main:
            return
        now = time.time()
        if not force and now - self._last_hb < float(self.settings["heartbeat_s"]):
            return
        self._last_hb = now
        atomic_write_json(
            self.run_dir / "heartbeat",
            {
                "ts": now,  # liveness: written at least every heartbeat_s while the process runs
                "progress_ts": self._progress_ts,  # the last training step, beat() or save
                "phase": "paused" if self._paused else self._phase,
                "sdk": 2,
                "pid": os.getpid(),
                "job_id": self.job_id,
                "step": self.step_count,
                "epoch": self.epoch,
                "samples_seen": self.samples_seen,
                "samples_epoch": self.samples_epoch,
                "epoch_size": self.epoch_size,
                "request_id": self._handled_rid,
                "launch_steps": self._launch_micro,
                "paused": self._paused,
                "probe_done": self._probe_done and not self._probe_restart,
                "peak_mem": self._peak_mem,
                "mem_total": self._mem_total,
                "last_ckpt_step": self._last_ckpt_step,
                "disk_low": self._disk_low,
                "oom_reason": self._oom_reason,
                "micro_batch": self.micro_batch,
                "grad_accum": self.grad_accum,
                "writer_error": self._writer_error,
                "last_request_failed": self._request_failed,
            },
            durable=False,
        )

    @staticmethod
    def _unwrap(model):
        """The model inside DataParallel / DistributedDataParallel / torch.compile wrappers, so a
        checkpoint loads with or without them. A model's own attribute named ``module`` is
        not a wrapper."""
        if torch is None or model is None:
            return model
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model = model.module
        inner = getattr(model, "_orig_mod", None)  # torch.compile
        if isinstance(inner, torch.nn.Module):
            model = inner
        return model

    @staticmethod
    def _dist():
        if torch is None:
            return None
        d = torch.distributed
        if d.is_available() and d.is_initialized() and d.get_world_size() > 1:
            return d
        return None

    def _say(self, msg: str) -> None:
        print(f"[guardian r{self.rank}] {msg}", file=sys.stderr, flush=True)

"""Settings (``config.yaml``) and secrets (``.env``).

Unknown keys are rejected, so a typo in a safety setting fails loudly at startup instead
of silently falling back to a default.
"""

from __future__ import annotations

import logging
import os
import re
import stat
from pathlib import Path
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

log = logging.getLogger(__name__)


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class GpusCfg(_M):
    """Which detected GPUs do what. Indexes are the ones ``rig-guardian gpus`` prints: NVIDIA
    cards first, numbered as nvidia-smi numbers them (PCI bus order), then AMD and Intel
    cards in PCI bus order, then an Apple GPU.

    Every detected GPU is monitored. ``training: auto`` uses all of them for training
    except the ``agent`` card (if any), anything in ``exclude`` (e.g. a display card), and
    graphics built into the CPU.
    """

    training: Union[Literal["auto"], list[int]] = "auto"
    agent: Optional[int] = None  # card that runs the optional language model; never trains
    exclude: list[int] = []  # monitored, never trained on
    expect_names: dict[str, str] = {}  # optional guard, e.g. {"training": "A100", "agent": "T4"}

    @property
    def resolved(self) -> bool:
        return self.training != "auto"


class FakeGpu(_M):
    name: str = "NVIDIA GeForce RTX 4090"
    mem_gb: float = 24
    slowdown_c: Optional[float] = 87
    vendor: Literal["nvidia", "amd", "intel", "apple"] = "nvidia"
    integrated: bool = False


class TelemetryCfg(_M):
    interval_s: float = Field(2.0, ge=0.1, le=60)
    history_days: int = Field(14, ge=1)
    db_every_s: float = Field(30.0, gt=0)
    ring_minutes: float = Field(10.0, gt=0, le=120)
    # A card that stops giving readings (driver error, hung sensor) is reported after
    # stale_alert_s; its jobs are paused after stale_pause_s (0: only report, never pause).
    stale_alert_s: float = Field(30.0, ge=5)
    stale_pause_s: float = Field(300.0, ge=0)
    events_days: int = Field(365, ge=1)  # how long the event and audit logs are kept
    # auto: every GPU found (NVIDIA through NVML, AMD/Intel through /sys, Apple Silicon);
    # nvml / sysfs / apple: only that interface; fake: simulated cards
    source: Literal["auto", "nvml", "sysfs", "apple", "fake"] = "auto"
    fake_gpus: list[FakeGpu] = [FakeGpu(), FakeGpu()]  # only for source: fake


class PowerCapCfg(_M):
    enabled: bool = False
    watts: int = Field(200, ge=1)
    helper: str = "/usr/local/sbin/rg-powercap"


class ThermalCfg(_M):
    warn_margin_c: float = Field(10, ge=0, le=60)  # below the card's own slowdown temperature
    pause_margin_c: float = Field(4, ge=0, le=60)
    resume_below_margin_c: float = Field(15, ge=0, le=80)
    resume_hold_s: float = Field(300, ge=0)
    throttle_pause_s: float = Field(60, gt=0)
    smoothing_samples: int = Field(3, ge=1, le=100)
    max_auto_resumes_per_day: int = Field(3, ge=0)
    emergency_ckpt_limit_s: float = Field(60, gt=0)
    fallback_slowdown_c: float = Field(85, ge=40, le=125)  # only if the card reports no limit of its own
    slowdown_override_c: dict[int, float] = {}  # per-GPU override of the card's own limit
    power_cap: PowerCapCfg = PowerCapCfg()

    @model_validator(mode="after")
    def _check_order(self):
        if not self.pause_margin_c < self.warn_margin_c:
            raise ValueError("thermal.pause_margin_c must be smaller than warn_margin_c "
                             "(the pause level is hotter than the warning level)")
        if not self.resume_below_margin_c > self.pause_margin_c:
            raise ValueError("thermal.resume_below_margin_c must be larger than pause_margin_c "
                             "(a card must cool below the pause level before its job resumes)")
        for gpu, t in self.slowdown_override_c.items():
            if not 40 <= t <= 125:
                raise ValueError(f"thermal.slowdown_override_c[{gpu}] = {t:g} °C is not a plausible limit (40–125)")
        return self


class MemoryCfg(_M):
    headroom_frac: float = Field(0.10, ge=0, lt=1)  # keep 10% VRAM free at preflight
    oom_auto_retry: bool = True
    max_oom_retries: int = Field(2, ge=0)
    probe_steps: int = Field(20, ge=0)
    static_limit_frac: float = Field(0.90, gt=0, le=1)
    bytes_per_param: int = Field(16, ge=1)  # fp32 weights + grads + Adam's two moments


class StallCfg(_M):
    no_progress_s: float = Field(900, gt=0)
    # Outside an epoch's training steps a job gets this long instead (never less than
    # no_progress_s): before the first step of a launch (loading data and weights, compiling)
    # and between epochs (evaluation). A process that stops answering altogether is still
    # caught after no_progress_s.
    quiet_phase_s: float = Field(3600, ge=0)
    kill_after_s: float = Field(300, ge=0)
    idle_util_pct: float = Field(5, ge=0, le=100)
    idle_warn_s: float = Field(600, gt=0)


class CheckpointsCfg(_M):
    every_steps: int = Field(500, ge=1)
    every_minutes: float = Field(20, gt=0)
    keep_last: int = Field(3, ge=1)
    keep_best: int = Field(1, ge=0)
    min_free_disk_gb: float = Field(20, ge=0)
    async_: bool = Field(True, alias="async")


class RunnerCfg(_M):
    launcher: Literal["auto", "systemd", "local"] = "auto"
    poll_s: float = Field(2.0, ge=0.05, le=60)
    # A card with no job of ours counts as in use by something else when another program's
    # compute process is on it (NVIDIA lists these), or when more of its memory is taken than
    # BOTH of these allow. A desktop and a browser fit under the defaults; another training run does not.
    free_mem_mb: int = Field(500, ge=0)
    free_mem_frac: float = Field(0.20, ge=0, le=1)
    # names of the agent's own environment variables to hand to every job (e.g. WANDB_API_KEY,
    # HF_TOKEN). Jobs otherwise get a minimal environment plus the template's `env:`.
    pass_env: list[str] = []

    @model_validator(mode="after")
    def _check_env(self):
        if "TELEGRAM_TOKEN" in self.pass_env:
            raise ValueError("runner.pass_env must not contain TELEGRAM_TOKEN: jobs never get the bot token")
        return self
    settle_s: float = Field(120, ge=0)  # wait after a restart that found interrupted jobs
    auto_resume_after_reboot: bool = True
    stop_timeout_s: float = Field(180, gt=0)  # a requested stop that has not exited by then is killed
    heartbeat_s: float = Field(10.0, gt=0)


class TelegramCfg(_M):
    allowed_user_ids: list[int] = []
    quiet_hours: Optional[list[str]] = None  # ["00:00", "07:00"], local time
    digest_time: Optional[str] = "08:00"
    confirm_ttl_s: float = Field(300, gt=0)
    group_window_s: float = Field(60, ge=0)
    warn_repeat_s: float = Field(3600, ge=0)
    queue_updates: bool = True
    max_message_age_s: float = Field(120, gt=0)  # older messages (sent while offline) are not acted on

    @model_validator(mode="after")
    def _check_times(self):
        from guardian.notifier import parse_hhmm

        if self.quiet_hours is not None:
            if len(self.quiet_hours) != 2:
                raise ValueError("telegram.quiet_hours must be [start, end], e.g. ['00:00', '07:00']")
            for t in self.quiet_hours:
                parse_hhmm(t)
        if self.digest_time is not None:
            parse_hhmm(self.digest_time)
        if len(set(self.allowed_user_ids)) != len(self.allowed_user_ids):
            raise ValueError("telegram.allowed_user_ids lists the same ID twice")
        return self


class BrainCfg(_M):
    url: str = "http://127.0.0.1:8081"
    enabled: bool = False  # optional: a llama.cpp server for plain-English messages
    timeout_s: float = Field(30, gt=0)
    history: int = Field(4, ge=0, le=20)
    max_tokens: int = Field(300, ge=16, le=4096)


class HealthcheckCfg(_M):
    url: Optional[str] = None  # e.g. https://hc-ping.com/<uuid>; optional
    every_s: float = Field(300, ge=10)


class Config(_M):
    gpus: GpusCfg = GpusCfg()
    telemetry: TelemetryCfg = TelemetryCfg()
    thermal: ThermalCfg = ThermalCfg()
    memory: MemoryCfg = MemoryCfg()
    stall: StallCfg = StallCfg()
    checkpoints: CheckpointsCfg = CheckpointsCfg()
    runner: RunnerCfg = RunnerCfg()
    telegram: TelegramCfg = TelegramCfg()
    brain: BrainCfg = BrainCfg()
    healthcheck: HealthcheckCfg = HealthcheckCfg()

    _root: Path = PrivateAttr(default_factory=lambda: Path.cwd())
    _file: Optional[Path] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _check_gpus(self):
        g = self.gpus
        if g.resolved:
            if len(set(g.training)) != len(g.training):
                raise ValueError("gpus.training lists the same GPU twice")
            if g.agent is not None and g.agent in g.training:
                raise ValueError("gpus.agent must not also be a training GPU")
            both = sorted(set(g.training) & set(g.exclude))
            if both:
                raise ValueError(f"GPU {both} is in both gpus.training and gpus.exclude")
        if any(i < 0 for i in list(g.exclude) + ([g.agent] if g.agent is not None else [])
               + (list(g.training) if g.resolved else [])):
            raise ValueError("GPU indexes cannot be negative")
        return self

    def resolve_gpus(self, detected: list) -> "Config":
        """Turn ``training: auto`` into concrete indexes, and check every configured index exists.

        ``detected`` is the inventory (one dict per GPU) or just the indexes. With the
        inventory, ``auto`` leaves out cards that cannot train: graphics built into the CPU
        and cards whose driver is not usable.
        """
        g = self.gpus
        rows = [d if isinstance(d, dict) else {"gpu": d} for d in detected]
        indexes = [int(r["gpu"]) for r in rows]
        configured = ([] if not g.resolved else list(g.training)) + ([g.agent] if g.agent is not None else [])
        missing = sorted({i for i in configured + list(g.exclude) if i not in indexes})
        if missing:
            raise ValueError(f"configured GPU index {missing} not found; detected GPUs: {indexes or 'none'}")
        if not g.resolved:
            g.training = [int(r["gpu"]) for r in rows
                          if r["gpu"] != g.agent and r["gpu"] not in g.exclude
                          and not r.get("integrated") and r.get("trainable", True)]
        return self

    @property
    def root(self) -> Path:
        return self._root

    @property
    def config_path(self) -> Path:
        """The file this config was loaded from (``<home>/config.yaml`` unless --config named another)."""
        return self._file or self._root / "config.yaml"

    @property
    def runs_dir(self) -> Path:
        return self._root / "runs"

    @property
    def db_path(self) -> Path:
        return self._root / "state.db"

    @property
    def templates_dir(self) -> Path:
        return self._root / "templates"

    def with_root(self, root: str | os.PathLike) -> "Config":
        self._root = Path(root).resolve()
        return self


def default_config_path() -> Path:
    """$RIG_GUARDIAN_HOME/config.yaml, else ~/rig-guardian/config.yaml.

    The current folder is never used implicitly: training projects often have their own
    ``config.yaml``, and treating one as the agent's home would be a nasty surprise. Use
    ``--home .`` to run from the current folder.
    """
    home = os.environ.get("RIG_GUARDIAN_HOME")
    if home:
        return Path(home).expanduser() / "config.yaml"
    return Path("~/rig-guardian/config.yaml").expanduser()


def load_config(path: str | os.PathLike) -> Config:
    path = Path(path)
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    if data is not None and not isinstance(data, dict):
        raise ValueError(f"{path} is not a Rig Guardian config (it must be a YAML mapping)")
    cfg = Config.model_validate(data or {})
    cfg.with_root(path.resolve().parent)
    cfg._file = path.resolve()
    return cfg


def load_env(path: str | os.PathLike) -> None:
    """Minimal .env reader: KEY=VALUE lines. Existing environment variables win."""
    path = Path(path)
    if not path.exists():
        return
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        log.warning("%s is readable by others (mode %o); run: chmod 600 %s", path, mode, path)
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):  # the shell form `export KEY=value` is accepted too
            line = line[len("export "):].lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if v[:1] in ("'", '"') and v.find(v[0], 1) > 0:
            v = v[1:v.find(v[0], 1)]  # quoted: everything up to the closing quote
        else:
            v = re.split(r"\s+#", v, maxsplit=1)[0].strip()  # unquoted: a trailing comment is not part of it
        os.environ.setdefault(k.strip(), v)

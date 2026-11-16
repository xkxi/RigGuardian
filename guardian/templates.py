"""Run templates: the only things the agent can launch.

A template fixes the script and environment and lists which settings may change, with
limits. A request becomes ``defaults`` plus validated overrides; anything outside the
limits is refused with the reason. Templates live on the rig and are never edited from
Telegram.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

log = logging.getLogger(__name__)

# settings the agent itself may change (OOM retry) without changing what is being trained
EXECUTION_KEYS = ("micro_batch", "grad_accum")
# environment variables the agent sets for every job; a template may not override them
RESERVED_ENV = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                "ZE_AFFINITY_MASK")
MAX_STR_SETTING = 200


class TemplateError(ValueError):
    """A user-facing reason a request does not fit a template."""


class Overridable(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["float", "int", "str", "bool"]
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Optional[list[Any]] = None
    pattern: Optional[str] = None  # for type str: a regular expression the whole value must match

    @model_validator(mode="after")
    def _check(self):
        if self.type not in ("float", "int") and (self.min is not None or self.max is not None):
            raise ValueError(f"min/max only apply to float and int settings, not {self.type}")
        for bound in (self.min, self.max):
            if bound is not None and not math.isfinite(bound):
                raise ValueError("min/max must be finite numbers")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"min {self.min:g} is greater than max {self.max:g}")
        if self.pattern is not None:
            if self.type != "str":
                raise ValueError("pattern only applies to str settings")
            try:
                re.compile(self.pattern)
            except re.error as e:
                raise ValueError(f"pattern is not a valid regular expression: {e}") from None
        return self


class Template(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    script: str
    python: str = "python3"
    workdir: Optional[str] = None  # default: the script's folder
    args: list[str] = []  # fixed command-line arguments
    env: dict[str, str] = {}
    gpus: int = Field(1, ge=1)  # GPUs per job; more than 1 launches under torchrun (DDP)
    min_gpu_mem_gb: Optional[float] = None  # only cards with at least this much memory
    gpu_match: Optional[str] = None  # only cards whose name contains this, e.g. "A100" or "4090"
    # only cards of this vendor; needed on a machine with cards from several vendors, because a
    # PyTorch build drives one vendor's cards (CUDA, ROCm, XPU or MPS)
    vendor: Optional[Literal["nvidia", "amd", "intel", "apple"]] = None
    sdk: bool = True  # False: launch and watch only; pause and checkpoint fall back to stop
    params_millions: Optional[float] = None  # for the static memory check
    best_metric: str = "val_loss"
    mode: Literal["min", "max"] = "min"
    description: str = ""
    defaults: dict[str, Any] = {}
    overridable: dict[str, Overridable] = {}

    @model_validator(mode="after")
    def _check(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", self.name):
            raise ValueError("name must be 1-64 letters, digits, '.', '_' or '-' and start with a letter or digit")
        bad = sorted(k for k in self.env if k in RESERVED_ENV or k.startswith("GUARDIAN_"))
        if bad:
            raise ValueError(f"env may not set {', '.join(bad)}: the agent sets these for every job")
        for key in self.overridable:  # a default outside its own limits would be launched unchecked
            if key in self.defaults:
                try:
                    coerce_value(self, key, self.defaults[key])
                except TemplateError as e:
                    raise ValueError(f"default {e}") from None
        return self

    def resolve(self, root: Path, value: str) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else (root / p)

    def describe_limits(self) -> str:
        parts = []
        for k, o in self.overridable.items():
            if o.choices is not None:
                lim = "one of " + ", ".join(str(c) for c in o.choices)
            elif o.min is not None or o.max is not None:
                lim = f"{_fmt(o.min)}–{_fmt(o.max)}"
            else:
                lim = o.type
            parts.append(f"{k} ({lim}; default {_fmt(self.defaults.get(k))})")
        return ", ".join(parts) or "nothing"


_ADAPTERS = {
    "float": TypeAdapter(float),
    "int": TypeAdapter(int),
    "str": TypeAdapter(str),
    "bool": TypeAdapter(bool),
}


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def load_templates(directory: Path, problems: list[tuple[str, str]] | None = None) -> dict[str, Template]:
    """Every valid template in ``directory``, by name.

    A file that cannot be read or is not a valid template is skipped, never raised: the
    watchdog and the scheduler call this, and one half-edited file must not stop them. Pass
    a list as ``problems`` to receive ``(file name, reason)`` for each skipped file.
    """
    out: dict[str, Template] = {}
    source: dict[str, str] = {}
    if not directory.is_dir():
        return out
    for p in sorted(directory.glob("*.y*ml")):
        try:
            data = yaml.safe_load(p.read_text())
            if data is None:
                data = {}
            if not isinstance(data, dict):
                raise TemplateError("the file must be a YAML mapping (key: value lines)")
            data.setdefault("name", p.stem)
            t = Template.model_validate(data)
            if t.name in out:
                raise TemplateError(f"its name {t.name!r} is already used by {source[t.name]}")
        except ValidationError as e:
            why = "; ".join(f"{'.'.join(str(x) for x in err['loc']) or 'file'}: {err['msg']}" for err in e.errors())
        except (TemplateError, yaml.YAMLError, OSError, UnicodeDecodeError, RecursionError, TypeError) as e:
            why = " ".join(str(e).split())[:300]
        else:
            out[t.name] = t
            source[t.name] = p.name
            continue
        if problems is not None:
            problems.append((p.name, why))
        else:
            log.warning("template %s skipped: %s", p.name, why)
    return out


def coerce_value(t: Template, key: str, raw: Any) -> Any:
    o = t.overridable[key]
    shown = raw if not isinstance(raw, str) or len(raw) <= 60 else raw[:60] + "…"
    if isinstance(raw, bool) and o.type in ("int", "float"):
        raise TemplateError(f"{key}={raw} is not a valid {o.type}")  # True is not 1 here
    try:
        v = _ADAPTERS[o.type].validate_python(raw)
    except ValidationError:
        raise TemplateError(f"{key}={shown} is not a valid {o.type}") from None
    if isinstance(v, float) and not math.isfinite(v):
        raise TemplateError(f"{key}={shown} is not a finite number")
    if isinstance(v, str):
        if len(v) > MAX_STR_SETTING or any(ord(c) < 32 or ord(c) == 127 for c in v):
            raise TemplateError(f"{key} must be at most {MAX_STR_SETTING} characters with no control characters")
        if o.pattern is not None and not re.fullmatch(o.pattern, v):
            raise TemplateError(f"{key}={shown} does not match the allowed pattern {o.pattern}")
    if o.choices is not None and v not in o.choices:
        raise TemplateError(f"{key}={_fmt(v)} is not allowed; choose one of {', '.join(map(str, o.choices))}")
    if o.min is not None and v < o.min or o.max is not None and v > o.max:
        raise TemplateError(f"{key}={_fmt(v)} is outside the allowed range {_fmt(o.min)}–{_fmt(o.max)}")
    return v


def merge_overrides(t: Template, overrides: dict[str, Any], base: dict[str, Any] | None = None) -> dict[str, Any]:
    """The template's defaults, then a parent run's settings (for a rerun), then the overrides.

    Everything is checked against the template as it is today. A parent's value for a
    setting that is still changeable must still be within the limits; a parent's value for
    a setting that is no longer changeable is replaced by today's fixed value (the caller
    shows the difference before anything is queued).
    """
    overrides = overrides or {}
    for key in overrides:
        if key not in t.overridable:
            raise TemplateError(
                f"{key} is not a setting you can change for {t.name}. "
                f"Changeable: {', '.join(t.overridable) or 'none'}"
            )
    merged = dict(t.defaults)
    for key, old in (base or {}).items():
        if key in overrides or key not in t.overridable:
            continue
        try:
            merged[key] = coerce_value(t, key, old)
        except TemplateError as e:
            raise TemplateError(f"the earlier run used a value the template no longer allows ({e}). "
                                f"Add {key}=<value> to choose a new one.") from None
    for key, raw in overrides.items():
        merged[key] = coerce_value(t, key, raw)
    return merged


def config_hash(cfg: dict[str, Any]) -> str:
    """Hash of what is being trained. Excludes micro_batch/grad_accum, which the OOM retry
    changes while keeping the effective batch, so a resumed run still matches its checkpoints."""
    core = {k: v for k, v in cfg.items() if k not in EXECUTION_KEYS}
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()


def diff_settings(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Human-readable differences, e.g. ['lr 3e-4 (was 1e-3)']."""
    out = []
    for k in sorted(set(old) | set(new)):
        if old.get(k) != new.get(k):
            out.append(f"{k} {_fmt(new.get(k))} (was {_fmt(old.get(k))})")
    return out


def parse_kv_args(tokens: list[str]) -> dict[str, str]:
    """``['lr=3e-4', 'epochs=50']`` -> ``{'lr': '3e-4', 'epochs': '50'}``."""
    out = {}
    for tok in tokens:
        if "=" not in tok:
            raise TemplateError(f"expected key=value, got {tok!r}")
        k, v = tok.split("=", 1)
        if not k:
            raise TemplateError(f"expected key=value, got {tok!r}")
        out[k.strip()] = v.strip()
    return out

"""Install and startup checks (``rig-guardian doctor``).

Works for any set of cards: it lists what was detected and the role of each card, says
which protections a card's interface cannot support (no temperature reading, no limit),
checks that each template's PyTorch build can drive the cards it may be given (CUDA build
with kernels for the card's compute capability on NVIDIA, ROCm on AMD, XPU on Intel, MPS
on Apple), and (optionally) that configured indexes point at the expected card models.
The daemon runs the identity check at startup and refuses to start on a mismatch.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

OK, WARN, FAIL = "ok", "warn", "FAIL"


def gpu_identity_problems(cfg, samples: list[dict]) -> list[str]:
    names = {s["gpu"]: (s.get("name") or "") for s in samples}
    problems = []
    for g in cfg.gpus.training:
        if g not in names:
            problems.append(f"training GPU index {g} does not exist")
    if cfg.gpus.agent is not None and cfg.gpus.agent not in names:
        problems.append(f"agent GPU index {cfg.gpus.agent} does not exist")
    want_t = cfg.gpus.expect_names.get("training")
    want_a = cfg.gpus.expect_names.get("agent")
    for g in cfg.gpus.training:
        if want_t and g in names and want_t not in names[g]:
            problems.append(f"GPU {g} is {names[g]!r}, expected a {want_t} for training")
    if want_a and cfg.gpus.agent in names and want_a not in names[cfg.gpus.agent]:
        problems.append(f"GPU {cfg.gpus.agent} is {names[cfg.gpus.agent]!r}, expected the {want_a} for the agent")
    return problems


def torch_supports(cc: str, archs: list[str]) -> bool:
    """True if a build with these arch entries runs on compute capability ``cc`` (e.g. '7.0').

    sm_XY cubins run on the same major version with an equal or newer minor; compute_XY
    (PTX) entries can be JIT-compiled for any equal or newer capability.
    """
    major, minor = (int(x) for x in cc.split("."))
    for a in archs:
        m = re.fullmatch(r"(sm|compute)_(\d+)(\d)\w*", a)
        if not m:
            continue
        am, an = int(m.group(2)), int(m.group(3))
        if m.group(1) == "sm" and am == major and an <= minor:
            return True
        if m.group(1) == "compute" and (am, an) <= (major, minor):
            return True
    return False


TORCH_PROBE = (
    "import json, torch; v = torch.version; x = getattr(torch, 'xpu', None); "
    "m = getattr(torch.backends, 'mps', None); "
    "print(json.dumps({'ver': torch.__version__, 'cuda': v.cuda, 'hip': getattr(v, 'hip', None), "
    "'archs': torch.cuda.get_arch_list() if v.cuda else [], "
    "'xpu': bool(x and x.is_available()), 'mps': bool(m and m.is_built())}))"
)


def build_supports(vendor: str, build: dict) -> str | None:
    """Why this PyTorch build cannot drive this vendor's cards (None if it can)."""
    if vendor == "nvidia":
        if build.get("hip"):
            return "it is a ROCm (AMD) build"
        return None if build.get("cuda") else "it is not a CUDA build"
    if vendor == "amd":
        return None if build.get("hip") else "AMD cards need a ROCm build of PyTorch"
    if vendor == "intel":
        return None if build.get("xpu") else "Intel cards need a PyTorch build with XPU support"
    if vendor == "apple":
        return None if build.get("mps") else "it was built without MPS (Apple GPU) support"
    return None


def run_checks(cfg, inventory: list[dict] | None = None) -> list[tuple[str, str]]:
    from guardian.telemetry import NoGpuError, inventory_table, make_source
    from guardian.vendors import vendor_label

    out: list[tuple[str, str]] = []
    if inventory is None:
        try:
            src = make_source(cfg)
            inventory = src.inventory()
            samples = src.read()
        except NoGpuError as e:
            return out + [(FAIL, str(e))]
    else:
        samples = inventory
    vendors = sorted({g.get("vendor") or "nvidia" for g in inventory})
    fake = cfg.telemetry.source == "fake"
    if fake:
        out.append((WARN, "telemetry.source is 'fake': simulated GPUs, nothing real is watched"))
    elif "nvidia" in vendors:
        order = os.environ.get("CUDA_DEVICE_ORDER")
        out.append((OK, "CUDA_DEVICE_ORDER=PCI_BUS_ID") if order == "PCI_BUS_ID" else
                   (WARN, "CUDA_DEVICE_ORDER is not PCI_BUS_ID system-wide; jobs launched by the agent get it, but "
                          "add it to /etc/environment so your own tools number the cards the same way"))
        try:
            import pynvml

            drv = pynvml.nvmlSystemGetDriverVersion()
            out.append((OK, f"NVIDIA driver {drv.decode() if isinstance(drv, bytes) else drv}"))
        except Exception as e:
            out.append((WARN, f"could not read the NVIDIA driver version: {e}"))
    if not fake:
        for v in vendors:
            if v in ("amd", "intel"):
                drivers = sorted({g.get("driver") or "?" for g in inventory if g.get("vendor") == v})
                out.append((OK, f"{vendor_label(v)} cards read from /sys (kernel driver: {', '.join(drivers)})"))
    try:
        cfg.resolve_gpus(inventory)
    except ValueError as e:
        return out + [(FAIL, str(e))]
    out.append((OK, f"{len(inventory)} GPU(s) detected, {len(cfg.gpus.training)} for training:\n"
                + inventory_table(inventory, cfg)))
    if not cfg.gpus.training:
        out.append((WARN, "no training GPUs: the agent will monitor but not run jobs"))
    for p in gpu_identity_problems(cfg, samples):
        out.append((FAIL, p))
    by_gpu = {s["gpu"]: s for s in samples}
    for g in inventory:
        i = g["gpu"]
        if g.get("note"):
            out.append((WARN, f"GPU {i} ({g.get('name') or '?'}): {g['note']}"))
        if i not in cfg.gpus.training:
            continue
        if "temp_c" in by_gpu.get(i, {}) and by_gpu[i].get("temp_c") is None and not by_gpu[i].get("lost"):
            out.append((WARN, f"GPU {i} reports no temperature: heat protection (warn, pause, emergency stop) "
                              "cannot work for this card"))
        elif not g.get("t_slow") and i not in cfg.thermal.slowdown_override_c:
            out.append((WARN, f"GPU {i} does not report a slowdown temperature; using "
                              f"{cfg.thermal.fallback_slowdown_c:g} °C (set thermal.slowdown_override_c)"))
    if len({g.get("vendor") or "nvidia" for g in inventory if g["gpu"] in cfg.gpus.training}) > 1:
        out.append((WARN, "training cards from more than one vendor: give each template a `vendor:` so its jobs "
                          "only go to cards its PyTorch build can use (one job never spans vendors)"))
    legacy = [g for g in inventory if (g.get("vendor") or "nvidia") == "nvidia" and g.get("cc")
              and float(g["cc"]) < 7.5]
    if legacy and not fake:
        out.append((WARN, "older cards (" + ", ".join(f"GPU {g['gpu']} compute {g['cc']}" for g in legacy)
                    + "): the R580 driver branch is the last for Maxwell/Pascal/Volta and CUDA 13 dropped them; "
                      "pin the driver and use CUDA 12.x builds of PyTorch"))

    from guardian.templates import load_templates

    problems: list[tuple[str, str]] = []
    tpls = load_templates(cfg.templates_dir, problems)
    for name, why in problems:  # the agent skips such a file; say exactly what is wrong with it
        out.append((FAIL, f"template file {name} cannot be used: {why}"))
    if not tpls:
        out.append((WARN, f"no templates in {cfg.templates_dir}"))
    training = [g for g in inventory if g["gpu"] in cfg.gpus.training]
    builds: dict[str, dict | str] = {}
    for t in tpls.values():
        script = t.resolve(cfg.root, t.script)
        if not script.exists():
            out.append((WARN, f"template {t.name}: script not found: {script}"))
        cards = [g for g in training if not t.vendor or (g.get("vendor") or "nvidia") == t.vendor]
        per_vendor: dict[str, list[dict]] = {}
        for g in cards:
            per_vendor.setdefault(g.get("vendor") or "nvidia", []).append(g)
        if t.vendor and not cards:
            out.append((WARN, f"template {t.name} is for {vendor_label(t.vendor)} cards; this rig trains on none"))
        elif max((len(v) for v in per_vendor.values()), default=0) < t.gpus:
            out.append((WARN, f"template {t.name} needs {t.gpus} GPUs of one vendor; this rig trains on "
                              + (", ".join(f"{len(v)} {vendor_label(k)}" for k, v in per_vendor.items()) or "none")))
        py = str(t.resolve(cfg.root, t.python)) if "/" in t.python else t.python
        if py not in builds:
            try:
                r = subprocess.run([py, "-c", TORCH_PROBE], capture_output=True, text=True, timeout=120)
                builds[py] = json.loads(r.stdout.strip().splitlines()[-1])
            except Exception as e:
                builds[py] = f"could not inspect torch in {py}: {type(e).__name__}: {e}"
        b = builds[py]
        if isinstance(b, str):
            out.append((WARN, f"{t.name}: {b}"))
            continue
        kind = (f"CUDA {b['cuda']}" if b.get("cuda") else f"ROCm {b['hip']}" if b.get("hip")
                else "XPU" if b.get("xpu") else "MPS" if b.get("mps") else "CPU-only")
        msg = f"{t.name}: torch {b['ver']} ({kind})"
        problems = []
        for vendor, vcards in per_vendor.items():
            why = build_supports(vendor, b)
            if why:
                problems.append(f"{why}, so it cannot use GPU {', '.join(str(g['gpu']) for g in vcards)} "
                                f"({vendor_label(vendor)})")
                continue
            if vendor == "nvidia" and b.get("archs"):
                bad = [f"GPU {g['gpu']} (compute {g['cc']})" for g in vcards
                       if g.get("cc") and not torch_supports(g["cc"], b["archs"])]
                if bad:
                    problems.append(f"no kernels for {', '.join(bad)}; the build has {' '.join(b['archs'])}")
        if problems and len(problems) == len(per_vendor):
            out.append((WARN if fake else FAIL, msg + ": " + "; ".join(problems)))
        elif problems:  # usable on some of this rig's cards: say which it will be kept away from only by `vendor:`
            out.append((WARN, msg + ": " + "; ".join(problems) + f". Add `vendor:` to {t.name} to keep it off them"))
        elif per_vendor:
            out.append((OK, msg + " · usable on every card it can be given"))

    env = cfg.root / ".env"
    if not os.environ.get("TELEGRAM_TOKEN"):
        out.append((WARN, f"TELEGRAM_TOKEN not set (put it in {env}); messages go to the log instead"))
    elif not cfg.telegram.allowed_user_ids:
        out.append((WARN, "telegram.allowed_user_ids is empty: nobody can talk to the bot"))
    if env.exists() and env.stat().st_mode & 0o077:
        out.append((FAIL, f"{env} is readable by others: chmod 600 {env}"))
    if cfg.brain.enabled:
        try:
            import httpx

            from urllib.parse import urlsplit

            if urlsplit(cfg.brain.url).hostname not in ("127.0.0.1", "localhost", "::1"):
                out.append((WARN, f"brain.url {cfg.brain.url} is not on this machine: your messages, run names and "
                                  f"metrics are sent there, unencrypted unless it is https"))
            # like the agent itself: never through a proxy from the environment
            r = httpx.get(cfg.brain.url.rstrip("/") + "/health", timeout=3, trust_env=False)
            out.append((OK if r.status_code == 200 else WARN, f"model server {cfg.brain.url}: HTTP {r.status_code}"))
        except Exception as e:
            out.append((WARN, f"model server {cfg.brain.url} not reachable ({type(e).__name__}); "
                              "free text will answer 'AI offline'"))
    return out


def doctor(cfg) -> int:
    from guardian.config import load_env

    load_env(Path(cfg.root) / ".env")
    results = run_checks(cfg)
    for level, msg in results:
        print(f"[{level:>4}] {msg}")
    return 1 if any(level == FAIL for level, _ in results) else 0


def main() -> None:
    from guardian.config import default_config_path, load_config

    sys.exit(doctor(load_config(sys.argv[1] if len(sys.argv) > 1 else default_config_path())))


if __name__ == "__main__":
    main()

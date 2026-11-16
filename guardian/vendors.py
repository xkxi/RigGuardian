"""GPUs beyond NVIDIA, and one numbered list for every card in the machine.

    NvmlSource    (telemetry.py)  NVIDIA cards through the driver's NVML library
    SysfsSource   AMD (amdgpu) and Intel (xe / i915) cards through the Linux kernel's own
                  files under /sys/class/drm: no vendor tools or extra packages needed
    AppleSource   the one GPU of an Apple Silicon Mac (unified memory), through sysctl/ioreg
    CompositeSource  joins any of these into one list: GPU 0..N-1 for the rest of the agent

Every sample carries ``vendor`` ("nvidia", "amd", "intel", "apple") and ``local``: the
card's index among the cards of the same vendor, which is how that vendor's runtime numbers
it. ``pin_env`` turns the cards picked for a job into the environment variables that make
the training process see exactly those cards.

What each vendor's interface can report differs (see ``VENDORS`` and the README table); a
reading a card cannot give is stored as empty, exactly like an unsupported NVML metric.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from guardian.telemetry import GpuLost, NoGpuError

log = logging.getLogger(__name__)

PCI_VENDORS = {"0x10de": "nvidia", "0x1002": "amd", "0x8086": "intel"}

# What the agent can do per vendor. ``backend`` is the PyTorch device type a job uses.
VENDORS: dict[str, dict[str, Any]] = {
    "nvidia": {"label": "NVIDIA", "backend": "cuda", "multi_gpu": True},
    "amd": {"label": "AMD", "backend": "cuda", "multi_gpu": True},  # ROCm builds present as torch.cuda
    "intel": {"label": "Intel", "backend": "xpu", "multi_gpu": True},
    "apple": {"label": "Apple", "backend": "mps", "multi_gpu": False},
}


def vendor_label(vendor: str | None) -> str:
    return VENDORS.get(vendor or "", {}).get("label", (vendor or "unknown").capitalize())


def pin_env(cards: list[dict]) -> dict[str, str]:
    """Environment that pins a training process to exactly these cards (all of one vendor).

    ``cards`` are telemetry samples or inventory rows; ``local`` is each card's index in its
    own vendor's numbering.
    """
    if not cards:
        return {}
    vendor = cards[0].get("vendor") or "nvidia"
    ids = ",".join(str(c.get("local", c["gpu"])) for c in cards)
    if vendor == "nvidia":
        # PCI_BUS_ID makes CUDA number the cards like NVML and nvidia-smi do
        return {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": ids,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    if vendor == "amd":
        # ROCm builds of PyTorch present AMD cards as torch.cuda devices and honour
        # HIP_VISIBLE_DEVICES; CUDA_VISIBLE_DEVICES is left unset so the two cannot disagree.
        return {"HIP_VISIBLE_DEVICES": ids}
    if vendor == "intel":
        return {"ZE_AFFINITY_MASK": ids}  # Level Zero device indexes, used by torch.xpu
    return {}  # apple: one GPU, nothing to select


def mixed_vendors(cards: list[dict]) -> bool:
    return len({c.get("vendor") or "nvidia" for c in cards}) > 1


# ---------------------------------------------------------------------------------------
# small file helpers
# ---------------------------------------------------------------------------------------
def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except (OSError, UnicodeDecodeError):
        return None


def _read_int(path: Path) -> int | None:
    v = _read(path)
    if v is None:
        return None
    try:
        return int(v, 0)
    except ValueError:
        return None


PCI_IDS_PATHS = ("/usr/share/hwdata/pci.ids", "/usr/share/misc/pci.ids", "/usr/share/pci.ids")
_PCI_TABLES: dict[tuple[str, ...], dict[tuple[str, str], str]] = {}


def _pci_table(paths: tuple[str, ...]) -> dict[tuple[str, str], str]:
    """(vendor id, device id) -> name, from the first pci.ids database that exists."""
    if paths not in _PCI_TABLES:
        table: dict[tuple[str, str], str] = {}
        for p in paths:
            try:
                text = Path(p).read_text(errors="replace")
            except OSError:
                continue
            cur = None
            for line in text.splitlines():
                if not line or line.startswith("#"):
                    continue
                if line.startswith("C "):  # device classes follow the vendor list
                    break
                if not line.startswith("\t"):
                    cur = line[:4].lower()
                elif not line.startswith("\t\t") and cur:
                    table[(cur, line[1:5].lower())] = line[5:].strip()
            if table:
                break
        _PCI_TABLES[paths] = table
    return _PCI_TABLES[paths]


def pci_device_name(vendor_id: str, device_id: str, paths: tuple[str, ...] = PCI_IDS_PATHS) -> str | None:
    """Marketing name from the system's pci.ids, e.g. 'Navi 21 [Radeon RX 6800 XT]' -> 'Radeon RX 6800 XT'."""
    key = (vendor_id.lower().removeprefix("0x"), device_id.lower().removeprefix("0x"))
    return _clean_pci_name(_pci_table(tuple(paths)).get(key))


def _clean_pci_name(raw: str | None) -> str | None:
    if not raw:
        return None
    m = re.search(r"\[([^\]]+)\]", raw)  # 'Navi 31 [Radeon RX 7900 XT/7900 XTX]'
    return (m.group(1) if m else raw).strip()


# ---------------------------------------------------------------------------------------
# AMD and Intel on Linux
# ---------------------------------------------------------------------------------------
class SysfsSource:
    """AMD and Intel GPUs through /sys/class/drm/card*/device (and its hwmon folder).

    AMD (amdgpu driver): temperature with its own critical and shutdown limits, VRAM used
    and total, busy percent, power and its cap, fan. Intel (xe / i915): package temperature
    on recent kernels, power from the energy counter, fan; no memory or busy readings.
    Nothing here needs root or any package. ``root`` is replaceable so tests can use a
    recorded tree.
    """

    def __init__(self, root: str | os.PathLike = "/sys", skip_vendors: tuple[str, ...] = (),
                 clock: Callable[[], float] = time.time, pci_ids: tuple[str, ...] = PCI_IDS_PATHS):
        self.root, self.clock = Path(root), clock
        self.cards: list[dict] = []
        drm = self.root / "class" / "drm"
        found = []
        try:
            names = sorted(os.listdir(drm))
        except OSError:
            names = []
        for name in names:
            if not re.fullmatch(r"card\d+", name):
                continue  # connectors such as card0-DP-1
            dev = drm / name / "device"
            vid = (_read(dev / "vendor") or "").lower()
            vendor = PCI_VENDORS.get(vid)
            if vendor is None or vendor in skip_vendors:
                continue
            cls = _read(dev / "class")
            if cls and not cls.lower().startswith("0x03"):
                continue  # not a display/3D controller
            found.append((self._pci_addr(dev) or name, name, dev, vendor, vid))
        counts: dict[str, int] = {}
        for i, (pci, name, dev, vendor, vid) in enumerate(sorted(found)):  # PCI bus order
            did = (_read(dev / "device") or "").lower()
            pretty = _read(dev / "product_name") or pci_device_name(vid, did, pci_ids)
            label = VENDORS[vendor]["label"]
            full = f"{label} {pretty}" if pretty and not pretty.lower().startswith(label.lower()) else (
                pretty or f"{label} GPU [{vid.removeprefix('0x')}:{did.removeprefix('0x')}]")
            uid = _read(dev / "unique_id")
            card = {"dev": dev, "card": name, "vendor": vendor, "energy": None}
            local = counts.get(vendor, 0)
            counts[vendor] = local + 1
            card["static"] = {
                "gpu": i, "vendor": vendor, "local": local, "name": full,
                "uuid": f"{vendor.upper()}-{uid}" if uid else f"PCI-{pci}",
                "pci": pci, "cc": None, "driver": self._driver(dev),
                "integrated": self._integrated(vendor, pci, dev),
                "mem_total": _read_int(dev / "mem_info_vram_total"),
                "t_slow": None, "t_shutdown": None,
            }
            if vendor == "nvidia":
                card["static"]["note"] = "NVIDIA driver (NVML) not available: monitored only"
                card["static"]["trainable"] = False
            temp = self._worst_temp(self._hwmon(dev))
            if temp:
                card["static"]["t_slow"], card["static"]["t_shutdown"] = temp[1], temp[2]
            self.cards.append(card)

    # ---- discovery helpers -------------------------------------------------------------
    @staticmethod
    def _pci_addr(dev: Path) -> str | None:
        for line in (_read(dev / "uevent") or "").splitlines():
            if line.startswith("PCI_SLOT_NAME="):
                return line.split("=", 1)[1].strip()
        real = Path(os.path.realpath(dev)).name
        return real if re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.\d", real) else None

    @staticmethod
    def _driver(dev: Path) -> str | None:
        for line in (_read(dev / "uevent") or "").splitlines():
            if line.startswith("DRIVER="):
                return line.split("=", 1)[1].strip()
        try:
            return Path(os.path.realpath(dev / "driver")).name if (dev / "driver").exists() else None
        except OSError:
            return None

    @staticmethod
    def _integrated(vendor: str, pci: str, dev: Path) -> bool:
        """Best guess at 'built into the CPU' (shares system memory; not a training card)."""
        if vendor == "intel":
            return pci.endswith("00:02.0")  # Intel's integrated graphics always sit here
        if vendor == "amd":
            # the kernel documents in1_input (northbridge voltage) as present on APUs only
            return any(h.joinpath("in1_input").exists() for h in SysfsSource._hwmons(dev))
        return False

    @staticmethod
    def _hwmons(dev: Path) -> list[Path]:
        try:
            return sorted(p for p in (dev / "hwmon").iterdir() if p.name.startswith("hwmon"))
        except OSError:
            return []

    def _hwmon(self, dev: Path) -> Path | None:
        hs = self._hwmons(dev)
        return hs[0] if hs else None

    @staticmethod
    def _worst_temp(hw: Path | None) -> tuple[float, float | None, float | None, str] | None:
        """(°C, critical °C, shutdown °C, label) of the sensor closest to its own limit.

        Cards expose several sensors (edge, junction, memory), each with its own limits, so
        the one with the least room decides. Sensors without a limit only count when none
        has one; then the hottest is reported.
        """
        if hw is None:
            return None
        best, hottest = None, None
        for n in range(1, 22):
            t = _read_int(hw / f"temp{n}_input")
            if t is None:
                continue
            crit = _read_int(hw / f"temp{n}_crit")
            emerg = _read_int(hw / f"temp{n}_emergency")
            label = _read(hw / f"temp{n}_label") or f"temp{n}"
            row = (t / 1000, crit / 1000 if crit else None, emerg / 1000 if emerg else None, label)
            if hottest is None or row[0] > hottest[0]:
                hottest = row
            if row[1] is not None and (best is None or row[1] - row[0] < best[1] - best[0]):
                best = row
        return best or hottest

    # ---- the source interface --------------------------------------------------------
    @property
    def gpus(self) -> list[int]:
        return [c["static"]["gpu"] for c in self.cards]

    def inventory(self) -> list[dict]:
        return [dict(c["static"]) for c in self.cards]

    def read(self) -> list[dict]:
        out = []
        for c in self.cards:
            ts = self.clock()
            try:
                out.append(self._read_card(c, ts))
            except GpuLost as e:
                out.append({**c["static"], "ts": ts, "lost": True, "error": str(e)})
        return out

    def _read_card(self, c: dict, ts: float) -> dict:
        dev = c["dev"]
        if _read(dev / "vendor") is None:
            raise GpuLost(f"{c['card']} is gone from {dev.parent.parent}")
        s: dict[str, Any] = {**c["static"], "ts": ts, "lost": False}
        hw = self._hwmon(dev)
        temp = self._worst_temp(hw)
        s["temp_c"] = temp[0] if temp else None
        if temp:
            s["t_slow"], s["t_shutdown"], s["temp_sensor"] = temp[1], temp[2], temp[3]
        s["mem_used"] = _read_int(dev / "mem_info_vram_used")
        total = _read_int(dev / "mem_info_vram_total")
        s["mem_total"] = total if total is not None else s.get("mem_total")
        s["util"] = _read_int(dev / "gpu_busy_percent")
        s["power_w"], s["power_limit_w"] = self._power(c, hw, ts)
        pwm, pwm_max = (_read_int(hw / "pwm1"), _read_int(hw / "pwm1_max")) if hw else (None, None)
        s["fan"] = round(100 * pwm / (pwm_max or 255)) if pwm is not None else None
        s["throttle"] = None  # NVML-only readings
        s["ecc_uncorrected"] = None
        s["procs"] = []  # the kernel files do not list processes per card
        return s

    def _power(self, c: dict, hw: Path | None, ts: float) -> tuple[float | None, float | None]:
        if hw is None:
            return None, None
        uw = _read_int(hw / "power1_average")
        if uw is None:
            uw = _read_int(hw / "power1_input")
        watts = uw / 1e6 if uw is not None else None
        if watts is None:  # Intel reports an energy counter (microjoules) instead of power
            uj = _read_int(hw / "energy1_input")
            if uj is not None:
                prev = c["energy"]
                c["energy"] = (uj, ts)
                if prev and ts > prev[1] and uj >= prev[0]:
                    watts = (uj - prev[0]) / 1e6 / (ts - prev[1])
        cap = _read_int(hw / "power1_cap")
        if cap is None:
            cap = _read_int(hw / "power1_max")
        return watts, (cap / 1e6 if cap else None)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------------------
# Apple Silicon
# ---------------------------------------------------------------------------------------
def _run_text(cmd: list[str], timeout: float = 3.0) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 else None


class AppleSource:
    """The built-in GPU of an Apple Silicon Mac: one device, sharing the system's memory.

    macOS gives an unprivileged process the chip name, the amount of memory and (through
    ``ioreg``) how busy the GPU is and how much memory it holds. It does not give the
    temperature without root, so heat protection cannot work on a Mac; the agent says so
    at startup.
    """

    def __init__(self, run: Callable[[list[str]], str | None] = _run_text,
                 clock: Callable[[], float] = time.time, force: bool = False):
        if not force and not (sys.platform == "darwin" and platform.machine() == "arm64"):
            raise NoGpuError("not an Apple Silicon Mac")
        self.run, self.clock = run, clock
        chip = (run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "").strip() or "Apple GPU"
        try:
            mem = int((run(["sysctl", "-n", "hw.memsize"]) or "").strip())
        except ValueError:
            mem = None
        uid = (run(["sysctl", "-n", "kern.uuid"]) or "").strip()
        self.static = {
            "gpu": 0, "vendor": "apple", "local": 0, "name": chip if chip.startswith("Apple") else f"Apple {chip}",
            "uuid": f"APPLE-{uid}" if uid else "APPLE-GPU", "pci": None, "cc": None, "driver": "metal",
            "integrated": False, "unified_memory": True, "mem_total": mem, "t_slow": None, "t_shutdown": None,
            "note": "unified memory; macOS does not report GPU temperature without root",
        }

    gpus = [0]

    def inventory(self) -> list[dict]:
        return [dict(self.static)]

    def read(self) -> list[dict]:
        s: dict[str, Any] = {**self.static, "ts": self.clock(), "lost": False, "temp_c": None, "mem_used": None,
                             "util": None, "power_w": None, "power_limit_w": None, "throttle": None,
                             "ecc_uncorrected": None, "fan": None, "procs": []}
        out = self.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"]) or ""
        m = re.search(r'"Device Utilization %"\s*=\s*(\d+)', out)
        if m:
            s["util"] = int(m.group(1))
        m = re.search(r'"In use system memory"\s*=\s*(\d+)', out)
        if m:
            s["mem_used"] = int(m.group(1))
        return [s]

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------------------
# every card, one list
# ---------------------------------------------------------------------------------------
class CompositeSource:
    """Several sources presented as one: GPU 0..N-1 in the order the sources are given.

    NVIDIA cards come first, so on a machine with only NVIDIA cards (and on a mixed one,
    for the NVIDIA cards) the numbers are the ones ``nvidia-smi`` shows.
    """

    def __init__(self, sources: list):
        self.sources = list(sources)
        self._slots: list[tuple[Any, int]] = []  # global index -> (source, that source's own index)
        for src in self.sources:
            for row in src.inventory():
                self._slots.append((src, row["gpu"]))

    @property
    def gpus(self) -> list[int]:
        return list(range(len(self._slots)))

    def _renumber(self, per_source: dict[int, dict[int, dict]]) -> list[dict]:
        out = []
        for g, (src, own) in enumerate(self._slots):
            row = per_source.get(id(src), {}).get(own)
            if row is not None:
                row = dict(row)
                row.setdefault("local", own)
                row["gpu"] = g
                out.append(row)
        return out

    def inventory(self) -> list[dict]:
        return self._renumber({id(s): {r["gpu"]: r for r in s.inventory()} for s in self.sources})

    def read(self) -> list[dict]:
        per: dict[int, dict[int, dict]] = {}
        for src in self.sources:
            try:
                per[id(src)] = {r["gpu"]: r for r in src.read()}
            except Exception as e:  # one vendor's interface failing must not blind the others
                log.exception("reading %s failed", type(src).__name__)
                ts = time.time()
                per[id(src)] = {r["gpu"]: {**r, "ts": ts, "lost": True, "error": f"{type(e).__name__}: {e}"}
                                for r in src.inventory()}
        return self._renumber(per)

    def close(self) -> None:
        for s in self.sources:
            try:
                s.close()
            except Exception:
                pass


def detect_sources(clock: Callable[[], float] = time.time, sysfs_root: str | os.PathLike = "/sys") -> tuple[list, list[str]]:
    """Every GPU source that finds at least one card on this machine, and notes on what was skipped."""
    from guardian.telemetry import NvmlSource

    sources: list = []
    notes: list[str] = []
    nvml_ok = False
    try:
        nv = NvmlSource()
        if nv.gpus:
            sources.append(nv)
            nvml_ok = True
    except NoGpuError as e:
        notes.append(f"NVIDIA: {e}")
    if sys.platform.startswith("linux"):
        # NVIDIA cards are read through NVML when it works; otherwise they are still listed
        sf = SysfsSource(sysfs_root, skip_vendors=("nvidia",) if nvml_ok else (), clock=clock)
        if sf.cards:
            sources.append(sf)
    try:
        sources.append(AppleSource(clock=clock))
    except NoGpuError:
        pass
    return sources, notes

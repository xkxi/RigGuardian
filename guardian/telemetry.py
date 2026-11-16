"""Telemetry collector: reads every GPU in the machine every 2 seconds.

NVIDIA cards are read through NVML in-process (well under a millisecond per card;
launching nvidia-smi every 2 seconds would spawn hundreds of processes an hour). AMD and
Intel cards are read from the Linux kernel's sysfs files and an Apple Silicon GPU through
macOS tools (see ``vendors.py``); ``make_source`` joins whatever it finds into one list.

Every read is wrapped, so a metric a card does not support (fan speed on passively cooled
cards, ECC on consumer cards, temperature on a Mac) is stored as empty instead of crashing
the loop. If a card stops answering, its sample is marked ``lost``.

Run ``python -m guardian.telemetry`` to print live readings.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import deque
from typing import Any, Callable

log = logging.getLogger(__name__)

# NVML clock-event ("throttle") reason bits
THROTTLE_SW_POWER_CAP = 0x4
THROTTLE_HW_SLOWDOWN = 0x8
THROTTLE_SW_THERMAL = 0x20
THROTTLE_HW_THERMAL = 0x40
THROTTLE_HW_POWER_BRAKE = 0x80
THERMAL_THROTTLE_MASK = THROTTLE_HW_SLOWDOWN | THROTTLE_SW_THERMAL | THROTTLE_HW_THERMAL


def short_name(name: str | None) -> str:
    """'Tesla V100-PCIE-16GB' -> 'V100', 'NVIDIA GeForce RTX 4090' -> 'RTX 4090',
    'NVIDIA H100 80GB HBM3' -> 'H100', 'AMD Radeon RX 7900 XTX' -> 'RX 7900 XTX', 'Apple M3 Max' -> 'M3 Max'."""
    if not name or not name.strip():
        return "GPU"
    s = re.sub(r"^(NVIDIA|AMD|Intel|Apple)(\(R\))?\s+", "", name.strip(), flags=re.I)
    s = re.sub(r"^(Tesla|Quadro|GeForce|Radeon)(\(TM\))?\s+", "", s, flags=re.I)
    s = re.sub(r"-(PCIE|SXM\d*|\d+\s*GB)\b.*$", "", s, flags=re.I)  # "-PCIE-16GB", "-SXM4-80GB", "-24GB"
    s = re.sub(r"\s+\d+\s*GB\b.*$", "", s, flags=re.I)
    return s.strip() or name.strip()


class GpuLost(Exception):
    pass


class NoGpuError(RuntimeError):
    """No usable GPU interface: the driver is missing, or the machine has no GPUs."""


# ---------------------------------------------------------------------------------------
# Sources: read() returns one sample per GPU; inventory() describes every GPU once.
# ---------------------------------------------------------------------------------------
class NvmlSource:
    """Reads real NVIDIA GPUs through pynvml (package ``nvidia-ml-py``). Finds them all."""

    def __init__(self, gpus: list[int] | None = None):
        try:
            import pynvml

            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
        except Exception as e:
            raise NoGpuError(f"cannot talk to the NVIDIA driver through NVML ({e}). Is the driver installed? "
                             "Use --fake-gpus to try Rig Guardian without GPUs.") from e
        self.nv = pynvml
        self.gpus = list(range(count)) if gpus is None else list(gpus)
        self.handles: dict[int, Any] = {}
        self.static: dict[int, dict] = {}
        self._hard_error: str | None = None  # last NVML error that was not "unsupported on this card"
        self._last_reinit = 0.0
        # errors that mean "this card or driver cannot report this metric", not "something broke"
        self._soft = {getattr(pynvml, n) for n in ("NVML_ERROR_NOT_SUPPORTED", "NVML_ERROR_NO_PERMISSION",
                                                   "NVML_ERROR_NOT_FOUND", "NVML_ERROR_INVALID_ARGUMENT",
                                                   "NVML_ERROR_FUNCTION_NOT_FOUND", "NVML_ERROR_NO_DATA")
                      if hasattr(pynvml, n)}
        for i in self.gpus:
            try:
                self._open(i)
            except GpuLost:
                log.error("GPU %d could not be opened through NVML", i)

    def _open(self, i: int) -> Any:
        nv = self.nv
        try:
            h = nv.nvmlDeviceGetHandleByIndex(i)
        except nv.NVMLError as e:
            raise GpuLost(str(e)) from e

        def text(fn):
            v = self._safe(fn)
            return v.decode() if isinstance(v, bytes) else v

        # read the card's own limits once; thresholds are set relative to them. Cards that do
        # not report a slowdown temperature usually report a maximum operating temperature.
        t_slow = self._safe(lambda: nv.nvmlDeviceGetTemperatureThreshold(h, nv.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN))
        if not t_slow and hasattr(nv, "NVML_TEMPERATURE_THRESHOLD_GPU_MAX"):
            t_slow = self._safe(lambda: nv.nvmlDeviceGetTemperatureThreshold(h, nv.NVML_TEMPERATURE_THRESHOLD_GPU_MAX))
        cc = self._safe(lambda: nv.nvmlDeviceGetCudaComputeCapability(h))
        mem = self._safe(lambda: nv.nvmlDeviceGetMemoryInfo(h))
        self.static[i] = {
            "gpu": i,
            "vendor": "nvidia",
            "local": i,  # NVML's own index: what CUDA_VISIBLE_DEVICES takes with PCI_BUS_ID order
            "name": text(lambda: nv.nvmlDeviceGetName(h)),
            "uuid": text(lambda: nv.nvmlDeviceGetUUID(h)),
            "pci": text(lambda: nv.nvmlDeviceGetPciInfo(h).busId),
            "cc": f"{cc[0]}.{cc[1]}" if cc else None,
            "mem_total": mem.total if mem else None,
            "t_slow": t_slow or None,
            "t_shutdown": self._safe(lambda: nv.nvmlDeviceGetTemperatureThreshold(
                h, nv.NVML_TEMPERATURE_THRESHOLD_SHUTDOWN)),
        }
        self.handles[i] = h  # only now: a card lost half-way through is opened again from scratch
        return h

    def inventory(self) -> list[dict]:
        return [dict(self.static.get(i, {"gpu": i, "vendor": "nvidia", "local": i, "name": None, "uuid": None}))
                for i in self.gpus]

    def _safe(self, fn: Callable[[], Any]) -> Any:
        nv = self.nv
        try:
            return fn()
        except nv.NVMLError as e:
            code = getattr(e, "value", None)
            if code == nv.NVML_ERROR_GPU_IS_LOST:
                raise GpuLost(str(e)) from e
            if code not in self._soft:
                self._hard_error = str(e)  # e.g. driver/library version mismatch after a driver update
            return None

    def _reinit(self) -> None:
        """After the driver stopped answering: start NVML again (at most once a minute)."""
        now = time.monotonic()
        if now - self._last_reinit < 60:
            return
        self._last_reinit = now
        try:
            try:
                self.nv.nvmlShutdown()
            except Exception:
                pass
            self.nv.nvmlInit()
            self.handles.clear()  # old handles are not valid in the new session
            log.warning("NVML re-initialised after an error")
        except Exception as e:
            log.warning("NVML re-initialisation failed: %s", e)

    def read(self) -> list[dict]:
        out = []
        for i in self.gpus:
            ts = time.time()
            try:
                out.append(self._read_one(i, ts))
            except GpuLost as e:
                self.handles.pop(i, None)  # look the card up again next time: it may come back
                out.append({**self.static.get(i, {"vendor": "nvidia", "local": i, "name": None, "t_slow": None,
                                                  "t_shutdown": None}),
                            "gpu": i, "ts": ts, "lost": True, "error": str(e)})
        if any(s.get("blind") for s in out):
            self._reinit()
        return out

    def _read_one(self, i: int, ts: float) -> dict:
        nv = self.nv
        h = self.handles.get(i) or self._open(i)
        s: dict[str, Any] = {**self.static[i], "gpu": i, "ts": ts, "lost": False}
        self._hard_error = None
        s["temp_c"] = self._safe(lambda: nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU))
        mem = self._safe(lambda: nv.nvmlDeviceGetMemoryInfo(h))
        if s["temp_c"] is None and mem is None and self._hard_error:
            # nothing can be read and the reason is not "unsupported": the agent is blind on this
            # card. It must not be mistaken for a cool, idle one.
            s["blind"], s["error"] = True, self._hard_error
        s["mem_used"] = mem.used if mem else None
        s["mem_total"] = mem.total if mem else s.get("mem_total")
        util = self._safe(lambda: nv.nvmlDeviceGetUtilizationRates(h))
        s["util"] = util.gpu if util else None
        pw = self._safe(lambda: nv.nvmlDeviceGetPowerUsage(h))
        s["power_w"] = pw / 1000 if pw is not None else None
        pl = self._safe(lambda: nv.nvmlDeviceGetEnforcedPowerLimit(h))
        s["power_limit_w"] = pl / 1000 if pl is not None else None
        # newer drivers call it "clocks event reasons", older ones "clocks throttle reasons";
        # the Python package may know a name the installed driver library does not
        s["throttle"] = None
        for fn_name in ("nvmlDeviceGetCurrentClocksEventReasons", "nvmlDeviceGetCurrentClocksThrottleReasons"):
            fn = getattr(nv, fn_name, None)
            if fn is not None:
                s["throttle"] = self._safe(lambda: fn(h))
                if s["throttle"] is not None:
                    break
        s["ecc_uncorrected"] = self._safe(lambda: nv.nvmlDeviceGetTotalEccErrors(
            h, nv.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, nv.NVML_VOLATILE_ECC))  # None on consumer cards
        s["fan"] = self._safe(lambda: nv.nvmlDeviceGetFanSpeed(h))  # None on passively cooled cards
        procs = self._safe(lambda: nv.nvmlDeviceGetComputeRunningProcesses(h)) or []
        s["procs"] = [(p.pid, getattr(p, "usedGpuMemory", None)) for p in procs]
        return s

    def close(self) -> None:
        try:
            self.nv.nvmlShutdown()
        except Exception:
            pass


class FakeSource:
    """Scriptable GPUs for tests and for trying the agent on a machine without GPUs.

    Each spec may name a ``vendor`` ("nvidia", "amd", "intel", "apple"), so a machine with
    cards from several vendors can be simulated too.
    """

    def __init__(self, specs: list, clock: Callable[[], float] = time.time):
        self.clock = clock
        self.state: dict[int, dict] = {}
        per_vendor: dict[str, int] = {}
        for i, sp in enumerate(specs):
            sp = sp if isinstance(sp, dict) else sp.model_dump()
            t_slow = sp.get("slowdown_c")
            vendor = sp.get("vendor") or "nvidia"
            local = per_vendor.get(vendor, 0)
            per_vendor[vendor] = local + 1
            self.state[i] = {
                "gpu": i, "vendor": vendor, "local": local, "name": sp.get("name", "Fake GPU"),
                "uuid": f"GPU-fake-{i:04d}", "integrated": bool(sp.get("integrated", False)),
                "pci": f"00000000:{i + 1:02X}:00.0", "cc": "8.9" if vendor == "nvidia" else None, "t_slow": t_slow,
                "t_shutdown": t_slow + 3 if t_slow else None, "temp_c": 40.0, "mem_used": 12 * 2**20,
                "mem_total": int(float(sp.get("mem_gb", 24)) * 2**30), "util": 0.0, "power_w": 30.0,
                "power_limit_w": 250.0, "throttle": 0, "ecc_uncorrected": 0, "fan": None, "procs": [],
                "lost": False,
            }

    @property
    def gpus(self) -> list[int]:
        return sorted(self.state)

    def inventory(self) -> list[dict]:
        keys = ("gpu", "vendor", "local", "name", "uuid", "integrated", "pci", "cc", "mem_total", "t_slow",
                "t_shutdown")
        return [{k: s[k] for k in keys} for s in self.state.values()]

    def set(self, gpu: int, **fields: Any) -> None:
        self.state[gpu].update(fields)

    def read(self) -> list[dict]:
        now = self.clock()
        return [{**s, "procs": list(s["procs"]), "ts": now} for s in self.state.values()]

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------------------
# Ring buffer + collector
# ---------------------------------------------------------------------------------------
class TelemetryStore:
    """Last ``ring_minutes`` of samples at full resolution, for the watchdog's trend checks."""

    def __init__(self, maxlen: int = 310):
        self._lock = threading.Lock()
        self._buf: dict[int, deque] = {}
        self.maxlen = maxlen

    def add(self, samples: list[dict]) -> None:
        with self._lock:
            for s in samples:
                self._buf.setdefault(s["gpu"], deque(maxlen=self.maxlen)).append(s)

    def latest(self, gpu: int) -> dict | None:
        with self._lock:
            b = self._buf.get(gpu)
            return b[-1] if b else None

    def last_n(self, gpu: int, n: int) -> list[dict]:
        with self._lock:
            b = self._buf.get(gpu)
            return list(b)[-n:] if b else []

    def window(self, gpu: int, seconds: float, now: float | None = None) -> list[dict]:
        now = time.time() if now is None else now
        with self._lock:
            b = self._buf.get(gpu)
            return [s for s in b if s["ts"] >= now - seconds] if b else []

    def oldest_ts(self, gpu: int) -> float | None:
        with self._lock:
            b = self._buf.get(gpu)
            return b[0]["ts"] if b else None

    def gpus(self) -> list[int]:
        with self._lock:
            return sorted(self._buf)

    def label(self, gpu: int) -> str:
        """Short card name; when several cards share it, followed by the GPU index, which is the
        number every command takes: 'RTX 4090 #2' is GPU 2. Same-named cards with different
        memory also show it ('A100 80G #1'). A card with no name is 'GPU 3'."""
        s = self.latest(gpu)
        short = short_name(s.get("name") if s else None)
        if short == "GPU":
            return f"GPU {gpu}"
        same = [g for g in self.gpus() if short_name((self.latest(g) or {}).get("name")) == short]
        if len(same) <= 1 or gpu not in same:
            return short
        mems = {(self.latest(g) or {}).get("mem_total") for g in same}
        mem = (s or {}).get("mem_total")
        size = f" {round(mem / 2**30)}G" if len(mems) > 1 and mem else ""
        return f"{short}{size} #{gpu}"

    @classmethod
    def from_db(cls, db) -> "TelemetryStore":
        """A store holding the daemon's newest reading per GPU (for the CLI, which has no live feed)."""
        st = cls(maxlen=5)
        try:
            samples = json.loads(db.kv_get("telemetry_latest", "[]") or "[]")
        except ValueError:
            samples = []
        st.add([{**s, "procs": []} for s in samples if isinstance(s, dict) and "gpu" in s])
        return st


class Collector(threading.Thread):
    def __init__(self, source, store: TelemetryStore, db, interval_s: float = 2.0, db_every_s: float = 30.0,
                 on_lost: Callable[[dict], None] | None = None, clock: Callable[[], float] = time.time):
        super().__init__(name="telemetry", daemon=True)
        self.source, self.store, self.db = source, store, db
        self.interval_s, self.db_every_s = interval_s, db_every_s
        self.on_lost = on_lost
        self.clock = clock
        self.stop_event = threading.Event()
        self.last_ok = time.monotonic()
        self._acc: dict[int, list[dict]] = {}
        self._last_flush = clock()
        self._lost_reported: set[int] = set()

    def collect_once(self) -> list[dict]:
        samples = self.source.read()
        self.store.add(samples)
        for s in samples:
            if s.get("lost"):
                if s["gpu"] not in self._lost_reported and self.on_lost:
                    self._lost_reported.add(s["gpu"])
                    self.on_lost(s)
                continue
            self._lost_reported.discard(s["gpu"])
            self._acc.setdefault(s["gpu"], []).append(s)
        if self.db is not None:  # newest reading for other processes (the CLI)
            self.db.kv_set("telemetry_latest", json.dumps([{k: v for k, v in s.items() if k != "procs"}
                                                           for s in samples], default=str))
        if self.clock() - self._last_flush >= self.db_every_s:
            self.flush()
        return samples

    def flush(self) -> None:
        """One averaged row per GPU (plus the window's max temperature) every 30 s."""
        rows = []
        for gpu, ss in self._acc.items():
            if not ss:
                continue

            def avg(k):
                v = [s[k] for s in ss if s.get(k) is not None]
                return sum(v) / len(v) if v else None

            temps = [s["temp_c"] for s in ss if s.get("temp_c") is not None]
            throttle = 0
            for s in ss:
                throttle |= int(s.get("throttle") or 0)
            mem = avg("mem_used")
            rows.append((gpu, ss[-1]["ts"], avg("temp_c"), max(temps) if temps else None,
                         int(mem) if mem is not None else None, avg("util"), avg("power_w"), throttle))
        self._acc = {}
        self._last_flush = self.clock()
        if rows and self.db is not None:
            self.db.add_telemetry(rows)

    def run(self) -> None:
        while not self.stop_event.is_set():
            t0 = time.monotonic()
            try:
                self.collect_once()
            except Exception:
                log.exception("telemetry read failed")
            self.last_ok = time.monotonic()  # a hung driver call never gets here; the daemon notices
            self.stop_event.wait(max(0.0, self.interval_s - (time.monotonic() - t0)))


def make_source(cfg, clock: Callable[[], float] = time.time):
    """The GPU source the config asks for. ``auto`` finds every card of every supported vendor."""
    kind = cfg.telemetry.source
    if kind == "fake":
        return FakeSource(cfg.telemetry.fake_gpus, clock=clock)
    if kind == "nvml":  # NVIDIA only, as before
        return NvmlSource()
    from guardian.vendors import AppleSource, CompositeSource, SysfsSource, detect_sources

    if kind == "sysfs":
        src = SysfsSource(clock=clock)
        if not src.cards:
            raise NoGpuError("no AMD or Intel GPU found under /sys/class/drm")
        return src
    if kind == "apple":
        return AppleSource(clock=clock)
    sources, notes = detect_sources(clock=clock)
    if not sources:
        raise NoGpuError("no GPU found. " + " ".join(notes) + " Looked for NVIDIA cards (NVML), AMD and Intel "
                         "cards (/sys/class/drm) and an Apple Silicon GPU. Use --fake-gpus to try Rig Guardian "
                         "without GPUs.")
    return sources[0] if len(sources) == 1 else CompositeSource(sources)


def inventory_table(inventory: list[dict], cfg=None) -> str:
    """One line per detected GPU, with its role when a resolved config is given."""
    lines = []
    for g in inventory:
        role = ""
        if cfg is not None:
            i = g["gpu"]
            role = ("training" if i in cfg.gpus.training else "language model" if i == cfg.gpus.agent
                    else "monitor only")
            role = f"  [{role}]"
        mem = f"{g['mem_total'] / 2**30:.0f} GiB" if g.get("mem_total") else "? GiB"
        if g.get("unified_memory"):
            mem += " shared with the system"
        parts = [g.get("name") or "?", mem]
        if g.get("cc"):
            parts.append(f"compute {g['cc']}")
        parts.append(f"slowdown {g['t_slow']:g} °C" if g.get("t_slow") else "no temperature limit reported")
        if g.get("integrated"):
            parts.append("integrated")
        lines.append(f"  GPU {g['gpu']}: " + " · ".join(parts) + role)
    return "\n".join(lines) if lines else "  (no GPUs found)"


def main() -> None:  # pragma: no cover - manual hardware check
    import argparse

    from guardian.config import default_config_path, load_config

    ap = argparse.ArgumentParser(description="Print live GPU readings")
    ap.add_argument("--config", default=None)
    ap.add_argument("--count", type=int, default=5)
    a = ap.parse_args()
    cfg = load_config(a.config or default_config_path())
    src = make_source(cfg)
    print(inventory_table(src.inventory()))
    for _ in range(a.count):
        for s in src.read():
            if s.get("lost"):
                print(f"GPU {s['gpu']}: LOST ({s.get('error')})")
                continue
            print(
                f"GPU {s['gpu']} {s['name']}: {s['temp_c']} °C (slowdown {s['t_slow']}, shutdown {s['t_shutdown']})"
                f" mem {((s['mem_used'] or 0) / 2**30):.1f}/{((s['mem_total'] or 0) / 2**30):.1f} GiB"
                f" util {s['util']}% power {s['power_w']} W/{s['power_limit_w']} W"
                f" throttle {s['throttle']!r} ecc {s['ecc_uncorrected']!r} fan {s['fan']!r} procs {s['procs']}"
            )
        print("-")
        time.sleep(cfg.telemetry.interval_s)
    src.close()


if __name__ == "__main__":
    main()

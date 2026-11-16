"""NvmlSource against a simulated NVML (the real pynvml module with its C calls replaced)."""
import types

import pynvml
import pytest

from guardian.telemetry import NvmlSource


Mem = types.SimpleNamespace(total=16 * 2**30, used=300 * 2**20)
Util = types.SimpleNamespace(gpu=97)


@pytest.fixture
def nvml(monkeypatch):
    """Two healthy cards; tests then break individual calls."""
    st = types.SimpleNamespace(fail={}, count=2)

    def call(name, value):
        def fn(*a):
            err = st.fail.get(name) or st.fail.get("*")
            if err and not (name in ("nvmlInit", "nvmlDeviceGetCount")):
                raise pynvml.NVMLError(err)
            return value(*a) if callable(value) else value
        monkeypatch.setattr(pynvml, name, fn)

    call("nvmlInit", None)
    call("nvmlDeviceGetCount", lambda: st.count)
    call("nvmlDeviceGetHandleByIndex", lambda i: f"h{i}")
    call("nvmlDeviceGetTemperatureThreshold", 87)
    call("nvmlDeviceGetCudaComputeCapability", (7, 0))
    call("nvmlDeviceGetMemoryInfo", Mem)
    call("nvmlDeviceGetName", "Tesla V100-PCIE-16GB")
    call("nvmlDeviceGetUUID", lambda h: f"GPU-{h}")
    call("nvmlDeviceGetPciInfo", types.SimpleNamespace(busId=b"0000:01:00.0"))
    call("nvmlDeviceGetTemperature", 91)
    call("nvmlDeviceGetUtilizationRates", Util)
    call("nvmlDeviceGetPowerUsage", 250_000)
    call("nvmlDeviceGetEnforcedPowerLimit", 250_000)
    call("nvmlDeviceGetCurrentClocksEventReasons", 0x40)
    call("nvmlDeviceGetCurrentClocksThrottleReasons", 0x40)
    call("nvmlDeviceGetTotalEccErrors", 0)
    call("nvmlDeviceGetFanSpeed", 40)
    call("nvmlDeviceGetComputeRunningProcesses", [])
    return st


def test_healthy_baseline(nvml):
    s = NvmlSource().read()[0]
    assert (s["temp_c"], s["t_slow"], s["throttle"], s["lost"]) == (91, 87, 0x40, False)


def test_b3_card_answering_unknown_error_is_not_a_healthy_free_card(nvml):
    """Xid 79 'fallen off the bus' is commonly reported as NVML_ERROR_UNKNOWN, not GPU_IS_LOST."""
    src = NvmlSource()
    nvml.fail["*"] = pynvml.NVML_ERROR_UNKNOWN
    s = src.read()[0]
    assert s.get("lost") or s.get("error"), (
        f"every NVML call fails with 'Unknown Error', yet the sample is lost={s['lost']} temp={s['temp_c']} "
        f"mem_used={s['mem_used']}: the watchdog sees tier NORMAL and free_gpus() sees an empty card")


def test_b20_gpu_lost_during_startup_must_not_break_all_telemetry(nvml):
    nvml.fail["nvmlDeviceGetTemperatureThreshold"] = pynvml.NVML_ERROR_GPU_IS_LOST  # handle opens, next call fails
    src = NvmlSource()
    nvml.fail.clear()
    samples = src.read()  # KeyError: static[0] was never filled, but handles[0] was
    assert len(samples) == 2


def test_b19_clock_reasons_fall_back_when_the_driver_lacks_the_new_symbol(nvml):
    """pynvml raises FUNCTION_NOT_FOUND when libnvidia-ml.so (an older driver) has no ...ClocksEventReasons."""
    nvml.fail["nvmlDeviceGetCurrentClocksEventReasons"] = pynvml.NVML_ERROR_FUNCTION_NOT_FOUND
    s = NvmlSource().read()[0]
    assert s["throttle"] == 0x40, f"throttle={s['throttle']!r}: the 60-s thermal-throttle pause rule can never fire"

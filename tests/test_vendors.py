"""AMD / Intel (kernel sysfs files), Apple Silicon, and several vendors in one machine.

No such hardware is needed: the sysfs readers run against a small recorded-style tree built
in a temporary folder, and the Apple reader against canned sysctl/ioreg output.
"""

import os

import pytest

from guardian.telemetry import FakeSource, GpuLost, short_name
from guardian.vendors import AppleSource, CompositeSource, SysfsSource, mixed_vendors, pin_env

GIB = 2**30


def make_card(root, n, pci, vendor, device, driver, files=None, hwmon=None, cls="0x030000"):
    """One /sys/class/drm/card<n> entry whose `device` is a symlink to a PCI device folder."""
    dev = root / "devices" / "pci0000:00" / pci
    dev.mkdir(parents=True)
    (dev / "vendor").write_text(vendor + "\n")
    (dev / "device").write_text(device + "\n")
    (dev / "class").write_text(cls + "\n")
    (dev / "uevent").write_text(f"DRIVER={driver}\nPCI_SLOT_NAME={pci}\n")
    for k, v in (files or {}).items():
        (dev / k).write_text(f"{v}\n")
    if hwmon is not None:
        h = dev / "hwmon" / f"hwmon{n + 3}"
        h.mkdir(parents=True)
        for k, v in hwmon.items():
            (h / k).write_text(f"{v}\n")
    card = root / "class" / "drm" / f"card{n}"
    card.mkdir(parents=True)
    os.symlink(dev, card / "device")
    (root / "class" / "drm" / f"card{n}-DP-1").mkdir()  # a connector: must be ignored
    return dev


@pytest.fixture
def pci_ids(tmp_path):
    p = tmp_path / "pci.ids"
    p.write_text("# test database\n"
                 "1002  Advanced Micro Devices, Inc. [AMD/ATI]\n"
                 "\t744c  Navi 31 [Radeon RX 7900 XT/7900 XTX/7900 GRE/7900M]\n"
                 "\t\t1002 0e3b  RX 7900 GRE [ASRock Steel Legend]\n"
                 "\t1681  Rembrandt [Radeon 680M]\n"
                 "8086  Intel Corporation\n"
                 "\t56a0  DG2 [Arc A770]\n"
                 "\t46a6  Alder Lake-P GT2 [Iris Xe Graphics]\n"
                 "C 03  Display controller\n"
                 "\t00  VGA compatible controller\n")
    return (str(p),)


def amd_dgpu(root, n=1, pci="0000:03:00.0", **over):
    hw = {"temp1_input": 61000, "temp1_label": "edge", "temp1_crit": 100000, "temp1_emergency": 105000,
          "temp2_input": 78000, "temp2_label": "junction", "temp2_crit": 110000, "temp2_emergency": 115000,
          "temp3_input": 70000, "temp3_label": "mem", "temp3_crit": 108000,
          "power1_average": 251000000, "power1_cap": 339000000, "pwm1": 128, "pwm1_max": 255, "fan1_input": 1500}
    hw.update(over)
    return make_card(root, n, pci, "0x1002", "0x744c", "amdgpu",
                     files={"mem_info_vram_total": 24 * GIB, "mem_info_vram_used": 3 * GIB, "gpu_busy_percent": 97,
                            "unique_id": f"a1b2c3d4e5f6071{n}"}, hwmon=hw)


def test_amd_card_reads_every_field_and_its_own_limits(tmp_path, pci_ids):
    amd_dgpu(tmp_path)
    src = SysfsSource(tmp_path, pci_ids=pci_ids)
    (inv,) = src.inventory()
    assert inv["vendor"] == "amd" and inv["local"] == 0 and inv["pci"] == "0000:03:00.0"
    assert inv["name"] == "AMD Radeon RX 7900 XT/7900 XTX/7900 GRE/7900M"
    assert inv["uuid"] == "AMD-a1b2c3d4e5f60711" and inv["integrated"] is False
    assert inv["mem_total"] == 24 * GIB and inv["driver"] == "amdgpu"
    (s,) = src.read()
    assert not s["lost"] and s["mem_used"] == 3 * GIB and s["util"] == 97
    assert s["power_w"] == pytest.approx(251.0) and s["power_limit_w"] == pytest.approx(339.0)
    assert s["fan"] == 50 and s["procs"] == [] and s["throttle"] is None and s["ecc_uncorrected"] is None
    # edge has 39 °C of room, junction 32, memory 38: the junction sensor decides, with its own limits
    assert (s["temp_c"], s["t_slow"], s["t_shutdown"], s["temp_sensor"]) == (78.0, 110.0, 115.0, "junction")


def test_the_sensor_with_least_room_decides_even_when_it_is_not_the_hottest(tmp_path, pci_ids):
    amd_dgpu(tmp_path, temp1_input=96000, temp2_input=99000)  # edge: 4 °C left of 100; junction: 11 left of 110
    (s,) = SysfsSource(tmp_path, pci_ids=pci_ids).read()
    assert (s["temp_c"], s["t_slow"], s["temp_sensor"]) == (96.0, 100.0, "edge")


def test_amd_apu_is_marked_integrated(tmp_path, pci_ids):
    make_card(tmp_path, 0, "0000:c5:00.0", "0x1002", "0x1681", "amdgpu",
              files={"mem_info_vram_total": GIB // 2, "mem_info_vram_used": GIB // 8, "gpu_busy_percent": 3},
              hwmon={"temp1_input": 44000, "in0_input": 800, "in1_input": 900, "power1_average": 9000000})
    (inv,) = SysfsSource(tmp_path, pci_ids=pci_ids).inventory()
    assert inv["integrated"] is True and inv["name"] == "AMD Radeon 680M"
    assert inv["uuid"] == "PCI-0000:c5:00.0"  # no unique_id file: the PCI address is the identity
    assert inv["t_slow"] is None  # no critical limit reported: the configured fallback applies


def test_intel_arc_reports_what_the_kernel_gives_and_nothing_invented(tmp_path, pci_ids):
    make_card(tmp_path, 0, "0000:00:02.0", "0x8086", "0x46a6", "i915")  # integrated: no hwmon at all
    make_card(tmp_path, 1, "0000:03:00.0", "0x8086", "0x56a0", "xe",
              hwmon={"temp2_input": 55000, "temp3_input": 51000, "energy1_input": 1_000_000, "power1_max": 190000000,
                     "fan1_input": 900})
    t = [100.0]
    src = SysfsSource(tmp_path, pci_ids=pci_ids, clock=lambda: t[0])
    igpu, arc = src.inventory()
    assert igpu["integrated"] is True and igpu["local"] == 0 and igpu["name"] == "Intel Iris Xe Graphics"
    assert arc["integrated"] is False and arc["local"] == 1 and arc["name"] == "Intel Arc A770"
    assert arc["mem_total"] is None and arc["t_slow"] is None
    s = src.read()[1]
    assert s["temp_c"] == 55.0 and s["mem_used"] is None and s["util"] is None
    assert s["power_w"] is None and s["power_limit_w"] == pytest.approx(190.0)  # needs two energy readings
    hw = next((tmp_path / "devices" / "pci0000:00" / "0000:03:00.0" / "hwmon").iterdir())
    (hw / "energy1_input").write_text("301000000\n")  # 300 J later
    t[0] += 2.0
    assert src.read()[1]["power_w"] == pytest.approx(150.0)
    s0 = src.read()[0]
    assert s0["temp_c"] is None and not s0["lost"]  # no sensors is not the same as a lost card


def test_cards_are_numbered_in_pci_order_and_other_devices_are_ignored(tmp_path, pci_ids):
    amd_dgpu(tmp_path, n=2, pci="0000:0a:00.0")
    amd_dgpu(tmp_path, n=0, pci="0000:03:00.0")
    make_card(tmp_path, 1, "0000:04:00.0", "0x1a03", "0x2000", "ast")  # a server's BMC graphics chip
    make_card(tmp_path, 3, "0000:05:00.0", "0x1002", "0xaaf0", "snd", cls="0x040300")  # not a display device
    src = SysfsSource(tmp_path, pci_ids=pci_ids)
    assert [(g["gpu"], g["local"], g["pci"]) for g in src.inventory()] == [(0, 0, "0000:03:00.0"),
                                                                             (1, 1, "0000:0a:00.0")]


def test_nvidia_card_without_its_driver_is_still_listed_but_skipped_when_nvml_has_it(tmp_path, pci_ids):
    make_card(tmp_path, 0, "0000:01:00.0", "0x10de", "0x2684", "nouveau", hwmon={"temp1_input": 39000})
    (inv,) = SysfsSource(tmp_path, pci_ids=pci_ids).inventory()
    assert inv["vendor"] == "nvidia" and "monitored only" in inv["note"]
    assert SysfsSource(tmp_path, skip_vendors=("nvidia",), pci_ids=pci_ids).inventory() == []


def test_a_card_that_vanishes_is_reported_lost_not_raised(tmp_path, pci_ids):
    dev = amd_dgpu(tmp_path)
    src = SysfsSource(tmp_path, pci_ids=pci_ids)
    assert not src.read()[0]["lost"]
    (dev / "vendor").unlink()
    (s,) = src.read()
    assert s["lost"] and s["gpu"] == 0 and s["name"].startswith("AMD") and "gone" in s["error"]


def test_empty_machine_has_no_cards(tmp_path):
    assert SysfsSource(tmp_path).inventory() == [] and SysfsSource(tmp_path).read() == []


# ---- Apple Silicon ---------------------------------------------------------------------
def fake_mac(ioreg=""):
    def run(cmd, timeout=3.0):
        if cmd[0] == "sysctl":
            return {"machdep.cpu.brand_string": "Apple M3 Max\n", "hw.memsize": f"{64 * GIB}\n",
                    "kern.uuid": "0A1B2C3D-0000-1111-2222-333344445555\n"}.get(cmd[-1])
        return ioreg
    return run


def test_apple_silicon_is_one_unified_memory_gpu_without_temperature():
    ioreg = ('+-o AGXAcceleratorG15X  <class AGXAcceleratorG15X, registered>\n    {\n'
             '      "PerformanceStatistics" = {"In use system memory"=6442450944,"Device Utilization %"=83,'
             '"Alloc system memory"=9000000000}\n    }\n')
    src = AppleSource(run=fake_mac(ioreg), force=True)
    (inv,) = src.inventory()
    assert inv["vendor"] == "apple" and inv["name"] == "Apple M3 Max" and inv["mem_total"] == 64 * GIB
    assert inv["unified_memory"] and inv["uuid"].startswith("APPLE-0A1B")
    (s,) = src.read()
    assert s["util"] == 83 and s["mem_used"] == 6 * GIB and s["temp_c"] is None and not s["lost"]


def test_apple_reader_survives_ioreg_failing():
    (s,) = AppleSource(run=fake_mac(None), force=True).read()
    assert s["util"] is None and s["mem_used"] is None and not s["lost"]


# ---- several vendors in one machine ----------------------------------------------------
def test_composite_numbers_every_card_once_and_keeps_each_vendors_own_index(tmp_path, pci_ids):
    nv = FakeSource([{"name": "NVIDIA GeForce RTX 4090"}, {"name": "NVIDIA GeForce RTX 3060", "mem_gb": 12}])
    amd_dgpu(tmp_path, n=0, pci="0000:03:00.0")
    amd_dgpu(tmp_path, n=1, pci="0000:0a:00.0")
    comp = CompositeSource([nv, SysfsSource(tmp_path, pci_ids=pci_ids)])
    assert comp.gpus == [0, 1, 2, 3]
    inv = comp.inventory()
    assert [(g["gpu"], g["vendor"], g["local"]) for g in inv] == [
        (0, "nvidia", 0), (1, "nvidia", 1), (2, "amd", 0), (3, "amd", 1)]
    samples = comp.read()
    assert [s["gpu"] for s in samples] == [0, 1, 2, 3] and samples[3]["pci"] == "0000:0a:00.0"
    assert len({g["uuid"] for g in inv}) == 4


def test_one_failing_source_marks_only_its_own_cards_lost(tmp_path, pci_ids):
    class Broken(FakeSource):
        def read(self):
            raise RuntimeError("driver crashed")

    amd_dgpu(tmp_path)
    comp = CompositeSource([Broken([{"name": "NVIDIA GeForce RTX 4090"}]), SysfsSource(tmp_path, pci_ids=pci_ids)])
    nv, amd = comp.read()
    assert nv["lost"] and "driver crashed" in nv["error"] and nv["gpu"] == 0
    assert not amd["lost"] and amd["temp_c"] == 78.0


def test_pin_env_uses_each_vendors_own_numbering():
    nv = [{"gpu": 0, "vendor": "nvidia", "local": 0}, {"gpu": 1, "vendor": "nvidia", "local": 1}]
    assert pin_env(nv)["CUDA_VISIBLE_DEVICES"] == "0,1" and pin_env(nv)["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    amd = [{"gpu": 3, "vendor": "amd", "local": 1}]
    env = pin_env(amd)
    assert env == {"HIP_VISIBLE_DEVICES": "1"}  # global GPU 3 is the second AMD card
    assert pin_env([{"gpu": 4, "vendor": "intel", "local": 0}]) == {"ZE_AFFINITY_MASK": "0"}
    assert pin_env([{"gpu": 0, "vendor": "apple", "local": 0}]) == {}
    assert mixed_vendors(nv + amd) and not mixed_vendors(nv)


@pytest.mark.parametrize("name,short", [
    ("AMD Radeon RX 7900 XTX", "RX 7900 XTX"),
    ("Intel Arc A770", "Arc A770"),
    ("Apple M3 Max", "M3 Max"),
    ("Tesla V100-PCIE-16GB", "V100"),
    ("NVIDIA GeForce RTX 4090", "RTX 4090"),
])
def test_short_names_across_vendors(name, short):
    assert short_name(name) == short


def test_gpulost_is_importable_from_telemetry():
    assert issubclass(GpuLost, Exception)


# ---- the agent on a machine with several vendors (simulated cards) ----------------------
class RecordingLauncher:
    """Records what would be launched instead of starting a process."""

    name = "recording"

    def __init__(self):
        self.calls = []

    def launch(self, job_id, argv, env, cwd):
        self.calls.append({"job": job_id, "argv": argv, "env": env})
        return os.getpid(), os.getpgid(0), None

    def reap(self):
        pass


def mixed_daemon(tmp_path, cards, **template_changes):
    from conftest import make_cfg, write_toy_template
    from guardian.daemon import Daemon

    write_toy_template(tmp_path, **template_changes)
    cfg = make_cfg(tmp_path, gpus={"training": "auto"}, telemetry={"fake_gpus": cards})
    d = Daemon(cfg, launcher=RecordingLauncher())
    d.collector.collect_once()
    return d


NV = {"name": "NVIDIA GeForce RTX 4090", "mem_gb": 24, "vendor": "nvidia"}
AMD = {"name": "AMD Radeon RX 7900 XTX", "mem_gb": 24, "vendor": "amd"}
INTEL_IGPU = {"name": "Intel UHD Graphics 770", "mem_gb": 0, "vendor": "intel", "integrated": True}
APPLE = {"name": "Apple M3 Max", "mem_gb": 64, "vendor": "apple", "slowdown_c": None}


def test_auto_roles_train_on_every_discrete_card_and_skip_integrated_graphics(tmp_path):
    d = mixed_daemon(tmp_path, [NV, INTEL_IGPU, AMD, AMD])
    assert d.cfg.gpus.training == [0, 2, 3]
    assert [g["local"] for g in d.inventory] == [0, 0, 0, 1]  # each vendor numbers its own cards from 0


def test_a_two_gpu_job_never_spans_vendors_and_is_pinned_in_that_vendors_numbering(tmp_path):
    from conftest import queue

    d = mixed_daemon(tmp_path, [NV, AMD, AMD], gpus=2)
    d.source.set(0, temp_c=30)  # the NVIDIA card is the coolest, but there is only one of it
    d.collector.collect_once()
    jid = queue(d)
    d.runner._schedule()
    (call,) = d.runner.launcher.calls
    assert d.db.get_job(jid)["gpus"] == "1,2"
    assert call["env"]["HIP_VISIBLE_DEVICES"] == "0,1" and "CUDA_VISIBLE_DEVICES" not in call["env"]
    assert call["env"]["GUARDIAN_GPU_VENDOR"] == "amd"
    assert "--nproc_per_node=2" in call["argv"]


def test_template_vendor_keeps_a_job_on_that_vendors_cards(tmp_path):
    from conftest import queue

    d = mixed_daemon(tmp_path, [AMD, NV, NV], vendor="nvidia")
    d.source.set(0, temp_c=25)  # coolest card is AMD; the template may not use it
    d.collector.collect_once()
    jid = queue(d)
    d.runner._schedule()
    (call,) = d.runner.launcher.calls
    assert d.db.get_job(jid)["gpus"] in ("1", "2")
    local = d.store.latest(int(d.db.get_job(jid)["gpus"]))["local"]
    assert call["env"]["CUDA_VISIBLE_DEVICES"] == str(local) and call["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert "HIP_VISIBLE_DEVICES" not in call["env"]


def test_a_job_that_would_have_to_span_vendors_is_refused_with_the_reason(tmp_path):
    from guardian.templates import TemplateError

    d = mixed_daemon(tmp_path, [NV, AMD], gpus=2)
    with pytest.raises(TemplateError, match="different vendors"):
        d.runner.prepare_run("toy-mlp", {})


def test_apple_gpu_takes_single_gpu_jobs_only_and_needs_no_pinning(tmp_path):
    from conftest import queue
    from guardian.templates import TemplateError

    d = mixed_daemon(tmp_path, [APPLE])
    jid = queue(d)
    d.runner._schedule()
    (call,) = d.runner.launcher.calls
    assert d.db.get_job(jid)["gpus"] == "0"
    assert not any(k.endswith("VISIBLE_DEVICES") or k == "ZE_AFFINITY_MASK" for k in call["env"])
    (tmp_path / "two").mkdir()
    d2 = mixed_daemon(tmp_path / "two", [APPLE], gpus=2)
    with pytest.raises(TemplateError, match="one GPU per machine"):
        d2.runner.prepare_run("toy-mlp", {})


def test_doctor_explains_which_cards_a_pytorch_build_cannot_drive():
    from guardian.checks import build_supports

    cuda = {"ver": "2.5", "cuda": "12.4", "hip": None, "xpu": False, "mps": False}
    rocm = {"ver": "2.5", "cuda": None, "hip": "6.2", "xpu": False, "mps": False}
    assert build_supports("nvidia", cuda) is None and build_supports("amd", rocm) is None
    assert "ROCm" in build_supports("amd", cuda) and "ROCm" in build_supports("nvidia", rocm)
    assert "XPU" in build_supports("intel", cuda) and "MPS" in build_supports("apple", cuda)

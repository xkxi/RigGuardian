"""Any number and mix of NVIDIA cards: detection, roles, fit, scheduling, inventory changes, init."""


import os
import sys

import pytest
import yaml

from conftest import make_cfg, outbox, queue, write_toy_template
from guardian.cli import main as cli_main
from guardian.config import Config, load_config
from guardian.daemon import Daemon
from guardian.runner import LocalLauncher
from guardian.telemetry import FakeSource, TelemetryStore
from guardian.templates import TemplateError

R4090 = {"name": "NVIDIA GeForce RTX 4090", "mem_gb": 24, "slowdown_c": 90}
R3060 = {"name": "NVIDIA GeForce RTX 3060", "mem_gb": 12, "slowdown_c": 93}
A100 = {"name": "NVIDIA A100-SXM4-80GB", "mem_gb": 80, "slowdown_c": 85}


def rig(tmp_path, cards, roles=None, clock=None, **tpl):
    write_toy_template(tmp_path, **tpl)
    cfg = make_cfg(tmp_path, gpus=roles or {"training": "auto"}, telemetry={"fake_gpus": cards})
    d = Daemon(cfg, launcher=LocalLauncher(), **({"clock": clock} if clock else {}))
    d.collector.collect_once()
    return d


def test_auto_uses_every_card_except_agent_and_excluded(tmp_path):
    d = rig(tmp_path, [R4090] * 5, {"training": "auto", "agent": 4, "exclude": [0]})
    assert d.cfg.gpus.training == [1, 2, 3]
    assert "5 GPUs (3 for training)" in d.reports.status_text()
    assert "monitor only" in d.reports.gpu_line(0) and "language model" in d.reports.gpu_line(4)


def test_single_card_rig(tmp_path):
    d = rig(tmp_path, [R3060])
    assert d.cfg.gpus.training == [0] and d.runner.free_gpus() == [0]
    assert d.store.label(0) == "RTX 3060"


def test_missing_configured_index_is_refused(tmp_path):
    cfg = Config.model_validate({"gpus": {"training": [0, 3]}})
    with pytest.raises(ValueError, match=r"\[3\] not found"):
        cfg.resolve_gpus([0, 1])


def test_labels_number_identical_cards():
    st = TelemetryStore()
    st.add(FakeSource([R4090, R3060, R4090]).read())
    assert [st.label(g) for g in (0, 1, 2)] == ["RTX 4090 #0", "RTX 3060", "RTX 4090 #2"]  # the number is the GPU index


def test_jobs_only_land_on_cards_they_fit(tmp_path):
    # 900 M params × 16 bytes ≈ 13.4 GiB: fits the 24 GiB card, not the 12 GiB one
    d = rig(tmp_path, [R3060, R4090], params_millions=900)
    t = d.runner.templates()["toy-mlp"]
    assert d.runner.free_gpus(t) == [1]
    assert "13.4 GiB" in d.runner.unsuitable(t, 0)
    t.gpus = 2
    with pytest.raises(TemplateError, match=r"needs 2 GPUs, but only 1 of this rig's 2 training GPUs qualify.*GPU 0"):
        d.runner.static_check(t)


def test_template_can_pin_a_card_model(tmp_path):
    d = rig(tmp_path, [R4090, A100, R4090], gpu_match="A100")
    t = d.runner.templates()["toy-mlp"]
    assert d.runner.eligible_gpus(t) == [1]


def test_n_gpu_job_gets_n_coolest_cards(tmp_path, monkeypatch):
    d = rig(tmp_path, [R4090] * 4, gpus=3)
    for g, temp in enumerate([60, 40, 50, 45]):
        d.source.set(g, temp_c=temp)
    d.collector.collect_once()
    launched = []
    monkeypatch.setattr(d.runner, "_launch", lambda job, gpus: launched.append((job["id"], gpus)))
    jid = queue(d)
    d.runner.tick()
    assert launched == [(jid, [1, 3, 2])]


def test_blocked_job_reserves_only_cards_it_could_use(tmp_path, monkeypatch):
    d = rig(tmp_path, [A100, R4090, R4090])
    write_toy_template(tmp_path, name="big", gpu_match="A100")
    busy = queue(d)
    d.db.update_job(busy, state="running", gpus="0", pid=os.getpid())  # the A100 is taken, ETA unknown
    launched = []
    monkeypatch.setattr(d.runner, "_launch", lambda job, gpus: launched.append((job["id"], gpus)))
    big = queue(d, template="big")
    small = queue(d)
    d.runner.tick()
    assert launched == [(small, [1])]  # the A100 job waits; the 4090s are not held for it
    assert d.db.get_job(big)["state"] == "queued"


def test_job_wider_than_rig_is_rejected(tmp_path):
    d = rig(tmp_path, [R4090, R4090], gpus=4)
    with pytest.raises(TemplateError, match="needs 4 GPUs"):
        d.runner.prepare_run("toy-mlp", {})


def test_inventory_change_alerts_and_quarantine_follows_the_card(tmp_path):
    d = rig(tmp_path, [R4090, R3060])
    d.startup()
    d.db.set_gpu_state(1, quarantined=1, quarantine_reason="ECC")
    # the 3060 moved to slot 0 and a new A100 sits in slot 1
    src = FakeSource([R3060, A100])
    src.state[0]["uuid"] = "GPU-fake-0001"
    src.state[1]["uuid"] = "GPU-new"
    cfg = make_cfg(tmp_path, gpus={"training": "auto"})
    d.close()  # the first agent has stopped
    d2 = Daemon(cfg, source=src, launcher=LocalLauncher())
    d2.startup()
    alert = next(t for t in outbox(d2) if t.startswith("🔧"))
    assert "GPU 0 is now RTX 3060 (was GPU 1)" in alert and "GPU 1 is now A100 (new card)" in alert
    assert "RTX 4090) is gone" in alert
    assert d2.db.gpu_state(0)["quarantined"] == 1 and d2.db.gpu_state(1)["quarantined"] == 0


def test_init_detects_and_writes_a_working_config(tmp_path, capsys):
    home = tmp_path / "home"
    with pytest.raises(SystemExit) as e:
        cli_main(["--home", str(home), "-y", "init", "--fake-gpus", "3", "--agent-gpu", "2"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "Found 3 GPUs" in out and "Training on: GPU 0, GPU 1" in out
    cfg = load_config(home / "config.yaml")
    assert cfg.gpus.training == "auto" and cfg.gpus.agent == 2 and cfg.brain.enabled
    assert len(cfg.telemetry.fake_gpus) == 3
    assert (home / ".env").stat().st_mode & 0o077 == 0
    toy = yaml.safe_load((home / "templates" / "toy-mlp.yaml").read_text())
    assert toy["python"] == sys.executable
    with pytest.raises(SystemExit) as e:  # job commands work before the agent ever ran
        cli_main(["--home", str(home), "status"])
    assert e.value.code == 0
    assert "3 GPUs (2 for training)" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        cli_main(["--home", str(home), "init", "--fake-gpus", "1"])
    assert e.value.code == 1  # refuses to overwrite


def test_gpus_command_lists_every_card(tmp_path, capsys):
    home = tmp_path / "h"
    with pytest.raises(SystemExit):
        cli_main(["--home", str(home), "-y", "init", "--fake-gpus", "2"])
    capsys.readouterr()
    with pytest.raises(SystemExit) as e:
        cli_main(["--home", str(home), "gpus"])
    out = capsys.readouterr().out
    assert e.value.code == 0 and "GPU 0" in out and "GPU 1" in out and "[training]" in out



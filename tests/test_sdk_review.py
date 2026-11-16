"""Training-side correctness: regression tests from the independent review of the SDK and the
checkpoint store (finding IDs A1-A18 in the test names). No agent needed; the resume tests run
tests/sdk_harness/train_h.py (the README's loop) in a bare run folder, kill it, and compare
the result with an uninterrupted run."""
import errno, itertools, json, math, os, re, sys, tempfile, threading, time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # these tests train a model in this process
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(HERE / "sdk_harness"), str(REPO / "guardian")]
import drive  # noqa: E402
from guardian_sdk import Guardian  # noqa: E402
from guardian.checkpoints import CheckpointStore  # noqa: E402

BASE = {"n": 512, "micro_batch": 32, "grad_accum": 2, "epochs": 3, "seed": 0}


@pytest.fixture(autouse=True)
def _tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("RG_TMP", str(tmp_path))


def new_g(tmp_path, settings=None, config=None, name="run"):
    rd = tmp_path / name
    rd.mkdir(exist_ok=True)
    (rd / "guardian.json").write_text(json.dumps({"min_free_disk_gb": 0, "every_steps": 1, "async": False, **(settings or {})}))
    return Guardian(run_dir=rd, config=config or {"micro_batch": 4, "grad_accum": 1})


def interrupted_vs_reference(cfg, deaths, settings=None):
    ref, _ = drive.run_with_deaths(cfg, [], settings)
    rd, codes = drive.run_with_deaths(cfg, deaths, settings)
    assert codes[-1] == 0
    return ref, rd


def val_rows(rd):
    return [(r["epoch"], r["step"]) for r in drive.metrics(rd) if "val_loss" in r]


# ---- A1 ------------------------------------------------------------------------------------
def test_A1_ddp_ranks_agree_on_whether_to_save():
    """Static check of the rank-local decisions that guard a collective (the e2e hang is in test_ddp2_e2e.py)."""
    src = (REPO / "guardian" / "guardian_sdk.py").read_text()
    body = src[src.index("def maybe_checkpoint"):src.index("def _save(")]
    assert "self._writer.is_alive()" not in body, "rank 0 skips _save (and its barriers) while other ranks enter it"


# ---- A3 ------------------------------------------------------------------------------------
def test_A3_resume_without_sdk_sampler_does_not_repeat_epochs():
    ref, rd = interrupted_vs_reference({**BASE, "use_sampler": False}, [{"DIE_AT": "25"}])
    assert drive.result(rd)["opt_steps"] == drive.result(ref)["opt_steps"]


# ---- A5 ------------------------------------------------------------------------------------
def test_A5_amp_dtype_is_fp16_on_pre_ampere_cards(tmp_path, monkeypatch):
    import torch.cuda as c

    class Props:
        major, minor = 7, 0  # V100 / T4 / RTX 20xx

    monkeypatch.setattr(c, "is_available", lambda: True)
    monkeypatch.setattr(c, "current_device", lambda: 0)
    monkeypatch.setattr(c, "get_device_properties", lambda d: Props)
    monkeypatch.setattr(c, "_check_bf16_tensor_supported", lambda d: True)  # any CUDA card can hold a bf16 tensor
    g = new_g(tmp_path)
    assert g.amp_dtype == torch.float16 and g.use_grad_scaler


# ---- A4 ------------------------------------------------------------------------------------
def test_A4_model_with_attribute_named_module_is_saved_whole(tmp_path):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.module = nn.Linear(4, 4)
            self.head = nn.Linear(4, 1)

    m = Net()
    g = new_g(tmp_path)
    g.resume(model=m)
    with g.step():
        pass
    g.maybe_checkpoint(force=True)
    saved = torch.load(g.store.dir / g.store.latest_name(), weights_only=False)["model"]
    assert set(saved) == set(m.state_dict())


# ---- A6 ------------------------------------------------------------------------------------
def test_A6_leftover_accumulation_resume_matches_uninterrupted():
    ref, rd = interrupted_vs_reference({**BASE, "grad_accum": 3}, [{"DIE_AFTER_VAL": "0"}], {"every_steps": 4})
    assert drive.result(rd)["weights"] == drive.result(ref)["weights"]


# ---- A7 ------------------------------------------------------------------------------------
def test_A7_validation_not_skipped_when_a_save_lands_on_the_last_step_of_the_epoch():
    ref, rd = interrupted_vs_reference(BASE, [{"DIE_IN_VAL": "0"}], {"every_steps": 8})  # 8 optimizer steps per epoch
    assert set(val_rows(rd)) == set(val_rows(ref))


def test_A7_best_metric_after_resume_is_the_one_known_at_that_checkpoint():
    ref, rd = interrupted_vs_reference(BASE, [{"DIE_AFTER_VAL": "1"}], {"every_steps": 8})
    best_seen = min(r["val_loss"] for r in drive.metrics(ref) if "val_loss" in r and r["epoch"] <= 1)
    resumed = [l for l in (rd / "trace.log").read_text().splitlines() if "resumed=True" in l][0]
    assert f"best={best_seen}" in resumed, resumed


# ---- A8 ------------------------------------------------------------------------------------
def test_A8_mid_epoch_resume_is_bit_identical_with_dropout():
    ref, rd = interrupted_vs_reference({**BASE, "dropout": 0.2}, [{"DIE_AT": "13"}])
    assert drive.result(rd)["weights"] == drive.result(ref)["weights"]


# ---- A10 ------------------------------------------------------------------------------------
def test_A10_heartbeat_can_be_written_from_two_threads(tmp_path):
    g = new_g(tmp_path)
    errs = []

    def loop():
        try:
            for _ in range(4000):
                g._heartbeat(force=True)
        except Exception as e:  # noqa: BLE001
            errs.append(repr(e))

    ts = [threading.Thread(target=loop) for _ in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs, errs[:1]


# ---- A11 ------------------------------------------------------------------------------------
def test_A11_newest_valid_checkpoint_wins_over_a_stale_LATEST(tmp_path):
    s = CheckpointStore(tmp_path / "ck")
    s.ensure()
    for step in (100, 200, 300):
        tmp = s.tmp_path(step)
        tmp.write_bytes(b"x" * 64 + str(step).encode())
        s.commit(tmp, {"step": step, "config_hash": "h"})
    (s.dir / "LATEST").write_text("step_000200.pt\n")  # the save of step 300 died after the rename, before LATEST
    man, _ = s.find_latest_valid("h")
    assert man["step"] == 300


# ---- A13 -----------------------------------------------------------------------------------
def test_A13_checkpoint_request_is_not_acknowledged_when_nothing_was_saved(tmp_path):
    g = new_g(tmp_path, {"min_free_disk_gb": 10**9})
    g.resume(model=nn.Linear(2, 1))
    (g.run_dir / "control.json").write_text(json.dumps({"action": "checkpoint", "request_id": 7}))
    with g.step():
        pass
    saved = g.maybe_checkpoint()
    hb = json.loads((g.run_dir / "heartbeat").read_text())
    assert saved or hb["request_id"] < 7 or hb.get("last_request_failed")


# ---- A12 -----------------------------------------------------------------------------------
def test_A12_nan_metric_is_never_the_best(tmp_path):
    nan = float("nan")
    for n, perm in enumerate(itertools.permutations([0.50, nan, 0.30, 0.40])):
        s = CheckpointStore(tmp_path / f"ck{n}")
        s.ensure()
        seq = list(perm) + [None, None, None]
        for step, metric in enumerate(seq, 1):
            tmp = s.tmp_path(step)
            tmp.write_bytes(b"x")
            s.commit(tmp, {"step": step, "config_hash": "h", "metric": metric})
        s.prune(keep_last=3, keep_best=1, mode="min")
        assert seq.index(0.30) + 1 in {int(m["step"]) for m in s.manifests()}, seq
    g = new_g(tmp_path)
    assert not g._improves(nan)


# ---- A16 -----------------------------------------------------------------------------------
def test_A16_corrupt_checkpoint_is_replaced_when_its_step_is_reached_again(tmp_path):
    def one(g, m, opt):
        with g.step():
            m(torch.ones(4, 2)).sum().backward(); opt.step(); opt.zero_grad()
        return g.maybe_checkpoint()

    m = nn.Linear(2, 1); opt = torch.optim.SGD(m.parameters(), lr=0.1)
    g = new_g(tmp_path, {"every_steps": 2}); g.resume(model=m, optimizer=opt)
    for _ in range(4):
        one(g, m, opt)
    p = g.store.path_for(4)
    b = bytearray(p.read_bytes()); b[len(b) // 2] ^= 0xFF; p.write_bytes(bytes(b))
    m2 = nn.Linear(2, 1); opt2 = torch.optim.SGD(m2.parameters(), lr=0.1)
    g2 = new_g(tmp_path, {"every_steps": 2}); assert g2.resume(model=m2, optimizer=opt2).step == 2
    one(g2, m2, opt2); one(g2, m2, opt2)
    man, _ = g2.store.find_latest_valid(None)
    assert man["step"] == 4


# ---- A17 -----------------------------------------------------------------------------------
def test_A17_len_loader_is_constant_within_an_epoch(tmp_path):
    g = new_g(tmp_path, config={"micro_batch": 10})
    ds = TensorDataset(torch.arange(100.0))
    s = g.sampler(ds); loader = DataLoader(ds, batch_size=10, sampler=s); s.set_epoch(0)
    lens = []
    for _ in loader:
        with g.step(batch_size=10):
            pass
        lens.append(len(loader))
    assert len(set(lens)) == 1, lens


# ---- A15 -----------------------------------------------------------------------------------
def test_A15_readme_example_runs_as_written(tmp_path, monkeypatch):
    block = next(b for b in re.findall(r"```python\n(.*?)```", (REPO / "README.md").read_text(), re.S) if "Guardian()" in b)
    rd = tmp_path / "readme"; rd.mkdir()
    monkeypatch.setenv("GUARDIAN_RUN_DIR", str(rd))
    (rd / "config.json").write_text(json.dumps({"seed": 0, "micro_batch": 16, "grad_accum": 2, "epochs": 2}))
    (rd / "guardian.json").write_text(json.dumps({"min_free_disk_gb": 0}))

    class M(nn.Module):
        def __init__(self):
            super().__init__(); self.l = nn.Linear(4, 1)

        def forward(self, batch):
            return self.l(batch[0]).pow(2).mean()

    model = M(); opt = torch.optim.SGD(model.parameters(), lr=0.01)
    ns = dict(torch=torch, DataLoader=DataLoader, model=model, opt=opt, sched=torch.optim.lr_scheduler.StepLR(opt, 10),
              train_set=TensorDataset(torch.randn(96, 4)), evaluate=lambda m: 1.0)
    exec(compile(block, "README example", "exec"), ns)  # everything else must be defined by the example itself


# ---- A18 -----------------------------------------------------------------------------------
def test_A18_forgetting_set_epoch_does_not_silently_empty_later_epochs(tmp_path):
    g = new_g(tmp_path, config={"micro_batch": 10})
    ds = TensorDataset(torch.arange(100.0))
    loader = DataLoader(ds, batch_size=10, sampler=g.sampler(ds))
    counts = []
    for _ in range(2):
        n = 0
        for _b in loader:
            with g.step(batch_size=10):
                pass
            n += 1
        counts.append(n)
    assert counts == [10, 10], counts

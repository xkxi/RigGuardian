"""Parametrised training script for the SDK tests. Follows the README loop exactly.
Config (config.json in GUARDIAN_RUN_DIR): n, micro_batch, grad_accum, epochs, seed, lr,
dropout, drop_last, num_workers, use_sampler, step_sleep, val_sleep.
Env: DIE_AT=<k> SIGKILL self after the k-th micro-batch of THIS launch (after maybe_checkpoint);
     DIE_IN_VAL=<epoch> SIGKILL self during validation of that epoch;
     DIE_AFTER_VAL=<epoch> SIGKILL self after the epoch-end forced checkpoint call."""
import hashlib, json, os, signal, sys, time
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from guardian_sdk import Guardian

g = Guardian()
if os.environ.get("OWN_SIGTERM"):
    def _mine(sig, frm):
        raise SystemExit("user cleanup handler: exiting")
    signal.signal(signal.SIGTERM, _mine)
cfg = g.config
if cfg.get("slow_save"):   # stands in for a multi-GB checkpoint on a slow disk
    _orig_save = torch.save
    def _slow(*a, **k):
        time.sleep(float(cfg.slow_save)); return _orig_save(*a, **k)
    torch.save = _slow
_cp = os.environ.get("CRASH_POINT")
if _cp:   # die (os._exit) at one point of the commit protocol of the CRASH_SAVE_NO-th checkpoint of this launch
    import guardian.checkpoints as C
    from pathlib import Path as _P
    _n = {"c": 0, "armed": False, "unl": 0}
    _commit, _awj, _rep, _awt, _unlink = C.CheckpointStore.commit, C.atomic_write_json, os.replace, C.atomic_write_text, _P.unlink
    def _die(): os._exit(9)
    def commit(self, tmp, manifest):
        _n["c"] += 1
        _n["armed"] = _n["c"] == int(os.environ.get("CRASH_SAVE_NO", 2))
        if _n["armed"] and _cp == "before_manifest": _die()
        return _commit(self, tmp, manifest)
    def awj(path, obj, durable=True):
        _awj(path, obj, durable)
        if _n["armed"] and _cp == "after_manifest" and str(path).endswith(".json") and "step_" in str(path): _die()
    def rep(a, b):
        _rep(a, b)
        if _n["armed"] and _cp == "after_rename" and str(b).endswith(".pt"): _die()
    def awt(path, text, durable=True):
        if _n["armed"] and _cp == "before_latest" and str(path).endswith("LATEST"): _die()
        _awt(path, text, durable)
        if _n["armed"] and _cp == "after_latest" and str(path).endswith("LATEST"): _die()
    def unlink(self, *a, **k):
        _unlink(self, *a, **k)
        if _n["armed"] and _cp == "mid_prune" and self.name.endswith(".pt"): _die()
    C.CheckpointStore.commit, C.atomic_write_json, os.replace, C.atomic_write_text, _P.unlink = commit, awj, rep, awt, unlink
seed = int(cfg.get("seed", 0))
torch.manual_seed(seed)
n, d = int(cfg.get("n", 512)), 8
gen = torch.Generator().manual_seed(99)
X = torch.randn(n, d, generator=gen); y = X.sum(1) + 0.1 * torch.randn(n, generator=gen)
VX = torch.randn(64, d, generator=gen); Vy = VX.sum(1)
train_set = TensorDataset(X, y, torch.arange(n))
p = float(cfg.get("dropout", 0.0))
model = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Dropout(p), nn.Linear(32, 1))
opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 3e-3)))
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=7, gamma=0.9)
scaler = g.grad_scaler()
state = g.resume(model=model, optimizer=opt, scheduler=sched, scaler=scaler)
mb, ga = int(cfg.micro_batch), int(cfg.get("grad_accum", 1))
kw = dict(batch_size=mb, drop_last=bool(cfg.get("drop_last", False)), num_workers=int(cfg.get("num_workers", 0)))
if cfg.get("use_sampler", True):
    sampler = g.sampler(train_set, seed=seed)
    loader = DataLoader(train_set, sampler=sampler, **kw)
else:
    sampler = None
    loader = DataLoader(train_set, shuffle=True, **kw)

def evaluate():
    model.eval()
    with torch.no_grad():
        v = nn.functional.mse_loss(model(VX).squeeze(1), Vy).item()
    model.train()
    return v

die_at = int(os.environ.get("DIE_AT", -1)); launch_micro = 0
oom_at = int(os.environ.get("OOM_AT", -1))
idxlog = open(os.path.join(g.run_dir, "idx.log"), "a")
print(json.dumps({"launch_from_step": state.step}), file=idxlog, flush=True)
trace = open(os.path.join(g.run_dir, "trace.log"), "a")
print(f"LAUNCH resumed={state.resumed} state.epoch={state.epoch} step={state.step} samples_seen={state.samples_seen} best={state.best_metric}", file=trace, flush=True)
for epoch in range(state.epoch, int(cfg.epochs)):
    if sampler is not None:
        sampler.set_epoch(epoch)
    nb = 0
    for xb, yb, ib in g.batches(loader):
        step_before = g.step_count
        with g.step(batch_size=mb) as s:
            if launch_micro == oom_at:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
            loss = nn.functional.mse_loss(model(xb).squeeze(1), yb) / ga
            scaler.scale(loss).backward()
            if s.optimizer_step:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            if cfg.get("step_sleep"):
                time.sleep(float(cfg.step_sleep))
        print(json.dumps({"e": epoch, "s": step_before, "idx": ib.tolist()}), file=idxlog, flush=True)
        g.log(loss=loss.item() * ga)
        if s.optimizer_step:
            g.maybe_checkpoint()
        nb += 1; launch_micro += 1
        if launch_micro == die_at:
            if g._writer is not None: g._writer.join()   # deterministic: a background save already started is on disk
            os.kill(os.getpid(), signal.SIGKILL)
    print(f"EPOCH {epoch} batches_this_launch={nb} g.step={g.step_count}", file=trace, flush=True)
    if os.environ.get("DIE_IN_VAL") == str(epoch):
        time.sleep(0.5)  # validation takes a while; a background save started at the last step finishes
        os.kill(os.getpid(), signal.SIGKILL)
    if cfg.get("val_sleep"):
        time.sleep(float(cfg.val_sleep))
    val = evaluate()
    g.log(val_loss=val, epoch=epoch)
    g.end_epoch(metric=val)
    if os.environ.get("DIE_AFTER_VAL") == str(epoch):
        if g._writer is not None: g._writer.join()
        os.kill(os.getpid(), signal.SIGKILL)
g.finish()
h = hashlib.sha256()
for k, v in model.state_dict().items():
    h.update(k.encode()); h.update(v.numpy().tobytes())
json.dump({"weights": h.hexdigest(), "opt_steps": g.step_count, "sched_last_epoch": sched.last_epoch,
           "val": evaluate()}, open(os.path.join(g.run_dir, "result.json"), "w"))

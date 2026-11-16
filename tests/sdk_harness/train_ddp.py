"""DDP (gloo/CPU) variant of the toy for reviewer repros. Extra settings:
oom_rank/oom_at_step: only that rank raises OOM (first launch); rank_metric: epoch-end metric differs per rank
and is passed WITHOUT force=True; rng_per_rank: seed + rank (common practice)."""
import hashlib, json, os, time
import torch, torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from guardian_sdk import Guardian

g = Guardian(); cfg = g.config
if cfg.get("slow_save"):   # stands in for a multi-GB checkpoint on a slow disk
    _orig_save = torch.save
    def _slow(*a, **k):
        time.sleep(float(cfg.slow_save)); return _orig_save(*a, **k)
    torch.save = _slow
seed = int(cfg.get("seed", 0))
torch.manual_seed(seed + (g.rank if cfg.get("rng_per_rank") else 0))
ddp = g.world > 1
if ddp:
    dist.init_process_group("gloo")
gen = torch.Generator().manual_seed(1234)
n, d = int(cfg.get("n_samples", 1024)), 8
X = torch.randn(n, d, generator=gen); y = X.sum(1)
train_set = TensorDataset(X, y, torch.arange(n))
model = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 1))
torch.manual_seed(seed + (g.rank if cfg.get("rng_per_rank") else 0) + 1000)
if ddp:
    model = DDP(model)
opt = torch.optim.SGD(model.parameters(), lr=float(cfg.lr))
state = g.resume(model=model, optimizer=opt)
def fp():
    return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()[:12]
log = open(os.path.join(g.run_dir, f"ddp_r{g.rank}.log"), "a")
print(json.dumps({"launch": True, "resumed": state.resumed, "step": state.step, "rng": fp()}), file=log, flush=True)
sampler = g.sampler(train_set, seed=seed)
mb, ga = int(cfg.micro_batch), int(cfg.grad_accum)
loader = DataLoader(train_set, batch_size=mb, sampler=sampler)
oom_marker = os.path.join(g.run_dir, 'oom_done')   # OOM once per job, also when no checkpoint existed yet
for epoch in range(state.epoch, int(cfg.epochs)):
    sampler.set_epoch(epoch)
    for i, (xb, yb, ib) in enumerate(loader):
        s0 = g.step_count
        with g.step(batch_size=mb):
            if cfg.get("oom_at_step") is not None and g.rank == int(cfg.get("oom_rank", 0)) and g.step_count == int(cfg.oom_at_step) \
                    and not os.path.exists(oom_marker):
                open(oom_marker, "w").close()
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated, one rank)")
            loss = nn.functional.mse_loss(model(xb).squeeze(1), yb) / ga
            loss.backward()
            if (i + 1) % ga == 0:
                opt.step(); opt.zero_grad(set_to_none=True)
            if cfg.get("step_sleep"):
                time.sleep(float(cfg.step_sleep))
        print(json.dumps({"e": epoch, "s": s0, "idx": ib.tolist()}), file=log, flush=True)
        g.log(loss=loss.item() * ga)
        if (i + 1) % ga == 0:
            g.maybe_checkpoint()
    if cfg.get("rank_metric"):
        val = float(-epoch if g.rank == 0 else epoch)      # improves on rank 0, gets worse on rank 1
        print(json.dumps({"epoch_end": epoch, "metric": val}), file=log, flush=True)
        g.maybe_checkpoint(metric=val)
    else:
        g.log(val_loss=1.0 / (epoch + 1), epoch=epoch)
        g.maybe_checkpoint(metric=1.0 / (epoch + 1), force=True)
    print(json.dumps({"epoch_done": epoch}), file=log, flush=True)
g.finish()
if ddp:
    dist.destroy_process_group()

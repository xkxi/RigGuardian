"""A tiny, deterministic training script that uses the Guardian SDK exactly like the guide.

Runs on CPU in seconds, so it is used by the tests and to try the whole agent on a
machine without GPUs. Extra settings exist only to provoke failures on purpose:

    step_sleep          seconds to sleep per micro-batch (makes runs long enough to poke)
    simulate_oom_above  raise an out-of-memory error whenever micro_batch is larger than this
    oom_at_step         raise an out-of-memory error once at this optimizer step (first launch only)
    hang_at_step        sleep forever at this optimizer step (stall detection)
    val_sleep           seconds to sleep before each validation (a long evaluation pass)
    crash_at_step       raise an ordinary exception at this optimizer step
"""

import os
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset

from guardian_sdk import Guardian

g = Guardian()
cfg = g.config
torch.manual_seed(int(cfg.get("seed", 0)))
dev = g.device  # cuda (NVIDIA, or AMD with a ROCm build), xpu (Intel), mps (Apple) or cpu
ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1  # a template with gpus: N launches N of these under torchrun
if ddp:
    dist.init_process_group("nccl" if g.backend == "cuda" else "gloo")
    if g.backend == "cuda":
        torch.cuda.set_device(dev)

# fixed synthetic regression data, independent of the run seed
gen = torch.Generator().manual_seed(1234)
n, d = int(cfg.get("n_samples", 2048)), 16
X = torch.randn(n, d, generator=gen)
y = (X @ torch.randn(d, 1, generator=gen)).squeeze(1) + 0.1 * torch.randn(n, generator=gen)
train_set = TensorDataset(X[: n * 7 // 8], y[: n * 7 // 8])
val_X, val_y = X[n * 7 // 8 :].to(dev), y[n * 7 // 8 :].to(dev)

hidden = int(cfg.get("hidden", 64))
model = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1)).to(dev)
if ddp:
    model = DDP(model)
opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr))
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.9)
scaler = g.grad_scaler()  # active only where it is needed (fp16 on CUDA)

state = g.resume(model=model, optimizer=opt, scheduler=sched, scaler=scaler)
sampler = g.sampler(train_set, seed=int(cfg.get("seed", 0)))
loader = DataLoader(train_set, batch_size=int(cfg.micro_batch), sampler=sampler)


def evaluate() -> float:
    model.eval()
    with torch.no_grad():
        v = nn.functional.mse_loss(model(val_X).squeeze(1), val_y).item()
    model.train()
    return v


oom_once = cfg.get("oom_at_step")
first_launch = not state.resumed
for epoch in range(state.epoch, int(cfg.epochs)):
    sampler.set_epoch(epoch)
    for xb, yb in g.batches(loader):  # knows the epoch's last batch; keeps the heartbeat fresh
        with g.step(batch_size=int(cfg.micro_batch)) as s:
            if cfg.get("simulate_oom_above") and int(cfg.micro_batch) > int(cfg.simulate_oom_above):
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
            if oom_once is not None and first_launch and g.step_count == int(oom_once):
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated, once)")
            if cfg.get("hang_at_step") is not None and g.step_count == int(cfg.hang_at_step):
                time.sleep(10**6)
            if cfg.get("crash_at_step") is not None and g.step_count == int(cfg.crash_at_step):
                raise ValueError("simulated crash")
            with g.autocast():  # bf16 or fp16, whichever this card supports; off on CPU
                loss = nn.functional.mse_loss(model(xb.to(dev)).squeeze(1), yb.to(dev)) / int(cfg.grad_accum)
            scaler.scale(loss).backward()
            if s.optimizer_step:  # every grad_accum micro-batches, and on the epoch's last batch
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            if cfg.get("step_sleep"):
                time.sleep(float(cfg.step_sleep))
        g.log(loss=loss.item() * int(cfg.grad_accum), lr=sched.get_last_lr()[0])
        if s.optimizer_step:
            g.maybe_checkpoint()
    if cfg.get("val_sleep"):
        time.sleep(float(cfg.val_sleep))  # stands in for a long evaluation pass
    val = evaluate()
    g.log(val_loss=val, epoch=epoch)
    g.end_epoch(metric=val)  # the epoch is complete: a resume starts the next one

g.finish()
print(f"done: val_loss {evaluate():.5f}")
if ddp:
    dist.destroy_process_group()

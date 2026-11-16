# Rig Guardian

A background agent that runs your PyTorch training queue on your own GPU machine while you're away. It finds every GPU in the machine, whatever the make, and tracks each one. It protects the cards from overheating and out-of-memory crashes, checkpoints every run so nothing is lost, and lets you check results and launch new runs from Telegram or the command line.

**Works with:** any number of GPUs, from one to as many as the machine holds, of any mix of makes: NVIDIA, AMD, Intel and Apple Silicon. Production use needs Linux with systemd. On a machine without GPUs it runs on simulated cards, for trying it out and for development.

> **Status: read this before trusting it.** The system is implemented and its 200+ tests pass. Those tests use simulated GPUs (including simulated AMD, Intel and Apple cards built from the files and commands the real ones expose), real training processes on CPU, and a fake model server. **It has not run on real GPU hardware of any make.** Go through the [hardware checklist](#hardware-checklist) before you leave it alone with a run.

## Quick start

```sh
pip install RigGuardian         # Python ≥ 3.11; the agent itself does not need PyTorch
rig-guardian init               # finds your GPUs, writes ~/rig-guardian/config.yaml
rig-guardian doctor             # checks drivers, GPUs, templates and PyTorch builds
rig-guardian start              # runs the agent (Ctrl-C stops the agent, never the jobs)
rig-guardian run toy-mlp -y     # in another terminal: a 30-second test job (needs torch)
rig-guardian status
rig-guardian service install --enable    # Linux: run in the background, start at boot
```

The package is called `RigGuardian` and installs one command, `rig-guardian`. Give it an environment of its own, apart from your training environments: `pipx install RigGuardian`, or `python -m venv ~/.rig-guardian-venv && ~/.rig-guardian-venv/bin/pip install RigGuardian`. To install from a source checkout instead, run `pip install .`, or `sh deploy/install.sh`, which also puts the command on your PATH and runs `init`.

`init` prints what it found and asks for an optional Telegram bot token. Without a token, messages go to the agent's log. With no GPU present, try everything on simulated cards with `rig-guardian init --fake-gpus 4`.

```
$ rig-guardian gpus
  GPU 0: NVIDIA GeForce RTX 4090 · 24 GiB · compute 8.9 · slowdown 90 °C  [training]
  GPU 1: NVIDIA GeForce RTX 4090 · 24 GiB · compute 8.9 · slowdown 90 °C  [training]
  GPU 2: AMD Radeon RX 7900 XTX · 24 GiB · slowdown 100 °C  [training]
  GPU 3: Intel UHD Graphics 770 · integrated  [monitor only]

  GPU 0 RTX 4090 #0  64 °C    9.1/24.0 GiB   98% busy  402/450 W  running j_0007
  ...
```

## Which GPUs, and what works on each

| Make | How it is read | What is tracked | Jobs train through | Heat protection |
|---|---|---|---|---|
| NVIDIA | the driver's NVML library | temperature, memory, utilization, power, throttle reasons, ECC errors, processes | CUDA | yes |
| AMD | Linux kernel files (`amdgpu` driver) | temperature, memory, utilization, power, fan | a ROCm build of PyTorch | yes |
| Intel | Linux kernel files (`xe` / `i915` driver) | temperature and power where the kernel exposes them (Arc cards, recent kernels); memory use often is not reported | PyTorch's `xpu` device | only when a temperature is reported |
| Apple Silicon | `sysctl` and `ioreg` on macOS | utilization, memory | PyTorch's `mps` device | no: macOS gives no GPU temperature without root |

What this means in practice:

- **One numbered list.** `rig-guardian gpus` shows every card as GPU 0…N-1: NVIDIA cards first, numbered as `nvidia-smi` numbers them, then AMD and Intel cards in PCI bus order, then an Apple GPU. Everything else (config, commands, messages) uses these numbers.
- **A reading a card cannot give is left empty,** never guessed. A training card with no temperature reading is named in an alert at startup, because the heat rules cannot protect it; out-of-memory, stall and checkpoint protection still work.
- **A job never spans makes.** A 2-GPU job gets two cards of the same make, the same model when it can. A template can ask for a make with `vendor: amd`.
- **Graphics built into the CPU are monitored, not trained on,** unless you list them under `gpus.training` yourself.
- **An NVIDIA card without a working driver** is listed (it is on the PCI bus) but marked not trainable.
- **Apple Silicon** has one GPU, no systemd and no temperature reading. It is useful for development and single-GPU runs started with `rig-guardian start` under tmux, not for an unattended rig.

## How GPUs are handled

- **Detection.** Every GPU is found at startup and read every 2 seconds.
- **Roles.** `gpus.training: auto` (the default) trains on every card except the optional language-model card (`gpus.agent`), any you list in `gpus.exclude` (such as the card driving your monitor) and integrated graphics. Cards that do not train are still monitored and warned about.
- **Free means free.** A card takes a new job only if no job of the agent's is on it, it is not quarantined or cooling, and nothing else is using it: no foreign compute process (NVIDIA reports these) and less of its memory in use than `runner.free_mem_frac` (20 %) or `runner.free_mem_mb`, whichever is larger. `status` says why a card is not free.
- **Each card's own limits.** Heat thresholds are set relative to each card's own slowdown temperature, so one config works for any mix. If a card reports no limit, 85 °C is assumed; set the real one in `thermal.slowdown_override_c`.
- **Mixed cards.** A job only lands on cards it fits. The template's `params_millions` gives a memory estimate (about 16 bytes per parameter for fp32 + Adam). `min_gpu_mem_gb` sets a hard floor, `gpu_match: "A100"` pins a job to one card model and `vendor:` to one make. A job that could never fit on this machine is refused with the reason for each card.
- **Any job width.** `gpus: N` in a template launches N processes under torchrun on N suitable free cards, the largest first. The queue is first in, first out (with optional priority). A job that has to wait holds back only the cards it could use; other jobs may still take a held card if they're expected to finish first. A job that cannot start until you act (its card is quarantined or gone) says so and lets the jobs behind it go ahead.
- **Cards that change.** Cards are recorded by their ID. If you add, remove or reorder cards, you get an alert on the next start; a quarantine follows the physical card, and a job that kept running is matched to its card's new number. If you pin indexes (`training: [0, 1]`), the agent refuses to start when one is missing. With `expect_names`, it also refuses when an index points at the wrong card model.
- **NVIDIA MIG** slices are not handled: the agent sees the physical card, not its slices.

## Safety design

- **Safety does not depend on AI.** Temperature limits, out-of-memory handling, stall detection and checkpointing are plain rule-based code. The optional language model only turns your messages into menu actions and rewords summaries. If it crashes, the machine is still protected and every command still works.
- **The AI picks from a fixed menu.** The model's output is constrained to one of 12 actions, and the agent checks every argument against your templates and the real job list. There is no path from a message to a shell command. Anything the model itself wrote is shown with an "AI:" label.
- **Confirmations and allowlist.** Anything that starts, stops or changes a job needs a Confirm tap, which carries a one-time code that expires after 5 minutes and is checked again against the current state when tapped. Only Telegram user IDs on the allowlist get a reply, only in a private chat, and forwarded or stale messages are not acted on. Everything is recorded in an audit table.
- **Training survives the agent.** Each job runs in its own systemd unit, so restarting the agent reattaches to running jobs instead of killing them. Only one agent can run on a rig folder at a time.
- **Every interruption is resumable.** Checkpoints are written atomically and verified by SHA-256 before they're trusted, so a pause, stop, crash, shutdown or power cut resumes from the last good checkpoint.
- **No reading is not good news.** If a card's readings stop (driver error, hung sensor), you are told after 30 seconds and its jobs are paused after 5 minutes. If one of the agent's own loops hangs, you are told and the external health ping stops.
- **The emergency stop comes first.** In a heat emergency the job is signalled before anything is written down, so a full disk or a locked database cannot delay it.
- **Jobs do not get the agent's secrets.** A job's environment is built from scratch: basics such as `PATH` and `HOME`, the template's `env:`, and the variables you name in `runner.pass_env`. The bot token is never passed, and it is removed from any text before it is stored or sent.

| Heat level | Trigger (relative to the card's own slowdown temperature) | Action |
|---|---|---|
| Warn | 10 °C below | one alert per hour per GPU |
| Pause | 4 °C below, or thermal throttling for 60 s | checkpoint, then soft-pause (keeps GPU memory) |
| Emergency | at the slowdown temperature, or the GPU drops off the bus | checkpoint (60 s limit), stop, quarantine the card |
| Resume | 15 °C below for 5 minutes | resume automatically, up to 3 times per job per day |

Each level is judged on the card's last 3 temperature readings (6 seconds), so a single noisy reading never pauses a run, and a failed reading in between does not reset the count.

## Adding your training script

A template is a short YAML file in `~/rig-guardian/templates/` that fixes the script and its Python environment and lists which settings may be changed, within limits. `init` puts `toy-mlp.yaml` and `my-project.yaml.example` there. A template file with a mistake in it is skipped and reported; it never stops the agent.

```yaml
name: cifar-resnet
script: ~/projects/cifar/train.py
python: ~/projects/cifar/.venv/bin/python    # the project's own environment, with its own torch
gpus: 1                                      # >1 runs under torchrun (DDP)
params_millions: 11                          # optional, for the memory fit check
# min_gpu_mem_gb: 12   gpu_match: "4090"   vendor: nvidia     # optional, for mixed machines
best_metric: val_loss
mode: min
defaults: {lr: 1.0e-3, epochs: 30, micro_batch: 256, grad_accum: 1, seed: 0}
overridable:
  lr:          {type: float, min: 1.0e-6, max: 1.0e-1}
  epochs:      {type: int, min: 1, max: 300}
  micro_batch: {type: int, choices: [32, 64, 128, 256, 512]}
```

In the script, about ten lines of the SDK add heartbeats, metrics, checkpoints, resume, out-of-memory recovery and pause/stop handling. The agent makes `guardian_sdk` importable in the job's environment automatically; it needs only torch. Build your model on `g.device` (`cuda`, `xpu`, `mps` or `cpu`, whichever the job's card needs).

```python
from guardian_sdk import Guardian

g = Guardian()
cfg = g.config                                    # the exact settings for this run
scaler = g.grad_scaler()                          # active only where it is needed (fp16 on CUDA)
state = g.resume(model=model, optimizer=opt, scheduler=sched, scaler=scaler)
sampler = g.sampler(train_set, seed=cfg.seed)     # resumes mid-epoch at the right example
loader = DataLoader(train_set, batch_size=cfg.micro_batch, sampler=sampler)
for epoch in range(state.epoch, cfg.epochs):
    sampler.set_epoch(epoch)
    for batch in g.batches(loader):               # knows which batch is the epoch's last
        with g.step(batch_size=cfg.micro_batch) as s:     # OOM catch, control checks
            with g.autocast():                    # bf16 where the card has it, else fp16; off on CPU
                loss = model(batch) / cfg.grad_accum
            scaler.scale(loss).backward()
            if s.optimizer_step:                  # every grad_accum batches, and on the epoch's last
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        g.log(loss=loss.item() * cfg.grad_accum)
        if s.optimizer_step:
            g.maybe_checkpoint()                  # only right after an optimizer step
    val = evaluate(model)                         # a long evaluation: for b in g.watch(val_loader)
    g.log(val_loss=val, epoch=epoch)
    g.end_epoch(metric=val)                       # the epoch is complete: save, flag a new best
g.finish()
```

Three things in this loop matter for a correct resume:

- **`s.optimizer_step`** decides when to step the optimizer. It is true every `grad_accum` micro-batches and on the epoch's last batch, so no accumulated gradient is left over when the batch count is not a multiple of `grad_accum`.
- **`g.end_epoch()`** is what marks an epoch as finished. A checkpoint taken on an epoch's last step, before validation ran, resumes into validation instead of skipping it.
- **`g.sampler` and `g.batches`** keep the data position. With your own sampler or an iterable dataset, keep `g.batches(loader)`: it counts epochs and, after a resume, skips the batches the epoch already trained on by reading and dropping them.

A job reports that it is alive from a background thread, so a long evaluation or data-loading phase is not mistaken for a hang. Progress is judged separately: no training step for 15 minutes inside an epoch (`stall.no_progress_s`), or for an hour before the first step and between epochs (`stall.quiet_phase_s`), counts as a stall. If your evaluation takes longer than that, loop over `g.watch(val_loader)` or call `g.beat()` inside it.

The full working example is `guardian/examples/train_toy.py` in the package, which also shows multi-GPU (DDP). Scripts without the SDK can still be launched and watched, but pause and checkpoint fall back to stop-and-restart. To run a script by hand outside the agent, install RigGuardian in that environment; the same import works and checkpoints go to `./guardian_run`.

Exit codes the runner understands:

| Code | Meaning | Response |
|---|---|---|
| `0` | finished | completed, run report |
| `86` | checkpointed and stopped | stopped and resumable if the agent asked for it; if something else sent the stop signal (a shutdown, a logout, a manual `kill`), the job is queued again and continues by itself, at most 3 times |
| `87` | out of memory | retried at half the micro-batch with double gradient accumulation (same effective batch), up to 2 times |
| other | crashed | failed, report with the last 30 log lines |

## Commands

The same commands work from the terminal (`rig-guardian <command>`) and from Telegram (`/<command>`):

| Command | What it does | Asks to confirm |
|---|---|---|
| `status`, `gpus`, `jobs`, `job <id>`, `templates` | overview, telemetry, queue, one job, changeable settings | no |
| `logs <id>`, `plot <id>`, `compare <id> <id>…`, `digest` | last 30 log lines, loss curve, side by side, daily summary | no |
| `run <template> key=value…`, `rerun <id> key=value…` | queue a run, or a copy of a past run with changes | yes |
| `pause <id>`, `resume <id>`, `stop <id>`, `cancel <id>` | soft pause, resume, checkpoint and stop, drop from the queue | yes |
| `ckpt <id>` | checkpoint now | no |
| `priority <id> <n>`, `clear <gpu>` | reorder the queue, lift a quarantine | yes |
| `hold`, `unhold` | stop or restart launching new jobs | no |

Terminal only: `init`, `doctor`, `start`, `service install|uninstall|status` and `ask "…"`, which sends free text to the model. In the terminal, `-y` skips the confirmation. Job IDs can be short: `job 42` means `j_0042`. The terminal commands work while the agent is stopped too (they share its database), and say so, since a queued job only starts once the agent runs.

**Telegram:** message `@BotFather`, send `/newbot`, and put the token in `~/rig-guardian/.env` as `TELEGRAM_TOKEN=…` (`init` asks for it). Then message your bot, open `https://api.telegram.org/bot<token>/getUpdates`, and copy `message.from.id` into `telegram.allowed_user_ids`. Everyone on the allowlist must press Start in the bot chat once, or Telegram will not let the bot write to them. The bot uses long polling, so no ports are opened.

**Plain-English messages (optional):** run a 3–8B instruct model with llama.cpp's `llama-server` on a spare card. For example, `rig-guardian init --agent-gpu 2` keeps GPU 2 out of training, and `rig-guardian service install --llm-model model.gguf` adds a service that serves the model on that card, on this machine only (127.0.0.1). After that, *"that last cifar run looked good, try it with half the learning rate and 50 epochs"* becomes a confirmed `rerun`. Keep the model server on the same machine: run names and metrics are sent to it.

## Settings worth knowing

`init` writes `config.yaml` with the common settings; every other one has a default. Values that make no sense (a negative interval, a pause margin above the warn margin, a time that is not `HH:MM`) are refused when the config is loaded, with the reason.

| Setting | Default | What it does |
|---|---|---|
| `gpus.training` / `agent` / `exclude` | `auto` / none / `[]` | which cards train, which serves the language model, which are only monitored |
| `telemetry.source` | `auto` | `auto` finds every make; `nvml`, `sysfs`, `apple` read one interface only; `fake` simulates |
| `telemetry.stale_alert_s` / `stale_pause_s` | 30 / 300 | a card's readings stop: alert after the first, pause its jobs after the second (0 = never pause) |
| `thermal.warn_margin_c` / `pause_margin_c` | 10 / 4 | °C below the card's slowdown temperature |
| `stall.no_progress_s` / `quiet_phase_s` | 900 / 3600 | no training step for this long inside an epoch / outside one |
| `runner.free_mem_frac` / `free_mem_mb` | 0.20 / 500 | memory another program may hold on a card that still counts as free |
| `runner.pass_env` | `[]` | names of your environment variables to hand to jobs, e.g. `[WANDB_API_KEY, HF_TOKEN]` |
| `runner.stop_timeout_s` | 180 | a job that has not exited this long after a stop is killed |
| `checkpoints.min_free_disk_gb` | 20 | below this, old checkpoints are pruned, then regular saves are skipped and you are told |
| `telegram.max_message_age_s` | 120 | messages older than this (sent while the agent was down) are not acted on |

## Example: 2× Tesla V100 + Quadro P4000

The setup this project was first designed for:

```sh
rig-guardian init --agent-gpu 2      # GPUs 0 and 1 train, the P4000 serves the model
```

Then add `expect_names: {training: V100, agent: P4000}` under `gpus:` so a numbering mix-up can't put training on the P4000. These cards are on NVIDIA's legacy track: pin the R580 driver and use CUDA 12.x builds of PyTorch, since CUDA 13 dropped Volta and Pascal. `doctor` checks that each template's PyTorch build has kernels for each card's compute capability. V100s have no native bf16, so `g.autocast()` uses fp16 and `g.grad_scaler()` is active on them.

On a single-GPU machine there is no spare card for the language model: leave `gpus.agent` unset and use the slash commands.

## Files

```
~/rig-guardian/            (your "home"; override with --home or $RIG_GUARDIAN_HOME)
  config.yaml              settings (written by init)
  .env                     TELEGRAM_TOKEN, chmod 600
  templates/*.yaml         approved run templates
  runs/<job_id>/           config, metrics.jsonl, heartbeat, control.json, stdout.log, checkpoints/
  state.db                 SQLite: jobs, telemetry, checkpoints, events, audit, outbox
  agent.lock               held by the running agent

guardian/                  the agent (repository)
  cli.py                   the rig-guardian command
  telemetry.py  vendors.py  watchdog.py  runner.py  checkpoints.py  actions.py  bot.py  brain.py
  reports.py  notifier.py  checks.py  db.py  config.py  templates.py  daemon.py
  guardian_sdk.py          the training SDK (copied into <home>/.sdk for jobs)
  examples/                toy training script and an example template
guardian_sdk.py            lets a pip-installed training environment import the SDK by the same name
deploy/install.sh          one-step install; deploy/rg-powercap: optional root power-cap helper (NVIDIA)
```

The home folder and the database are created readable by you only. They hold your messages, the audit trail and log excerpts.

## Testing

From a source checkout:

```sh
pip install -e '.[dev]' torch && pytest -q      # 214 tests, about 5 minutes, no GPU needed
```

Using simulated GPUs and real training processes, the tests cover:
- **Checkpoints and resume:** `kill -9` mid-epoch resumes with bit-identical weights, with dropout and with gradient accumulation that does not divide the epoch; death at every point of the save protocol; corrupt and half-written checkpoints fall back to the previous good one.
- **Heat and memory:** every heat level, including that a single spike is ignored and that missing readings neither count as cool nor cancel an emergency; the out-of-memory retry and give-up paths.
- **Stalls and restarts:** stall kill-and-resume; a long evaluation is not a stall; reattaching after an agent restart; a power cut mid-save; a shutdown signal.
- **Scheduling:** mixed-card fit, N-GPU jobs, FIFO reservation, inventory changes, 1 to 16 cards.
- **Every make:** AMD, Intel and Apple cards simulated from the kernel files and commands the real ones expose; mixed machines; jobs pinned to the right cards for each make.
- **Multi-GPU training:** two processes with DDP on CPU: pause, resume, stop and out-of-memory on one rank act on the same step on both.
- **Setup:** `init` and the other commands.
- **Telegram and the model:** allowlist, expiring one-time confirmations, an unreachable user or a refused message cannot hold up alerts, limits refused with the reason, `unsupported` for shell requests, and the model being offline.

## Hardware checklist

To do once on real hardware before running unattended:

1. Run `rig-guardian gpus` and check that every card shows up with sensible temperatures, memory and limits. Run `rig-guardian doctor` and read every WARN.
2. Run a short real job and `kill -9` it mid-epoch; it should resume at the right example.
3. Temporarily set large heat margins (e.g. `warn_margin_c: 45`, `pause_margin_c: 40`) so the levels trigger at normal temperatures. Don't actually overheat a card.
4. Force a real out-of-memory error with a huge micro-batch; the job should retry at half the size.
5. Restart the agent with `systemctl --user restart rig-guardian` during a run; the job should keep going. Then reboot during a run; it should resume after the 2-minute settle.
6. If you use multi-GPU jobs, check that one pauses and resumes on the same step on every card. This is tested with two CPU processes; NCCL on real cards is not.
7. Check that a second Telegram account gets no reply.
8. On AMD or Intel cards: check that a job started by the agent really runs on the card it was given (`rig-guardian gpus` should show that card busy and the others idle), and that the temperature shown matches `sensors`.

## Choices beyond the original design

- **What jobs see.** A job's `PYTHONPATH` gets only a private copy of the SDK (`<home>/.sdk`), never the agent's own environment.
- **Exit codes under torchrun.** torchrun turns a rank's exit code 86/87 into 1, so the SDK also writes an `exit_intent` file, which the runner reads.
- **Signals.** The launch wrapper passes TERM and USR1 on to the training process and always records the exit code, so `systemctl stop`, a logout or a shutdown gives the job its chance to checkpoint. `SIGUSR1` ("checkpoint now") goes only to the training processes that registered through the SDK, because its default action would kill other scripts and data-loader workers.
- **Steps and out-of-memory.** A checkpoint's step is the optimizer step. After an out-of-memory error, the position rolls back to the last optimizer step, because the gradients accumulated since then are lost. The config hash ignores `micro_batch`/`grad_accum`, so a retried run still matches its checkpoints.
- **Emergency stops.** Jobs stopped by a heat emergency or memory (ECC) errors wait for `resume`, and their card stays quarantined until `clear`.
- **What a restart keeps.** A job you had paused or were stopping stays stopped after a power cut. Only jobs that were running, or paused for heat, start again by themselves.
- **Summaries.** A summary written by the model is thrown away if it contains a number or a job ID the agent did not give it. That catches invented figures, not every possible misstatement, so summaries are labelled as the model's.

## Known limits

- **Not yet run on real GPUs.** See the status note at the top. AMD, Intel and Apple support in particular is built from those platforms' documented interfaces and tested only against simulations of them.
- **Clean control needs the SDK.** Without it, pause means stop-and-restart.
- **Heat handling is reactive.** If pauses keep repeating, the fix is airflow.
- **The memory estimate is rough.** The static check ignores activations. The 20-step probe catches most cases, but memory can still spike late, for example on evaluation with long inputs. An out-of-memory error during evaluation is outside `g.step()`, so it is an ordinary crash, not a retry.
- **Exact resume depends on your data pipeline.** Weights after a resume are bit-identical for randomness in the main process (dropout, CPU-side augmentation) when you use `g.sampler`. Random augmentation inside data-loader workers gets different worker seeds after a mid-epoch resume, and a loader with its own shuffling is not replayed in the same order. Under multi-GPU each process keeps its own random state.
- **Background service is Linux-only.** On other systems, run `rig-guardian start` under tmux/screen. On macOS, `torchrun --standalone` can hang on hostname lookup.
- **Shutdown gives jobs limited time.** At shutdown systemd gives a user's services a limited time in total (typically 90 to 120 seconds); a job whose final checkpoint takes longer is killed and resumes from its previous one.
- **Telegram is a dependency.** If it's down, safety actions still happen, but you hear about them later. Messages that could not be sent for a week are dropped.
- **Import name.** The Python package is named `guardian`, which is also the import name of `django-guardian`. Install RigGuardian in its own environment (as above) and the two never meet.

## Releasing to PyPI

```sh
pip install build twine
python -m build                 # dist/rigguardian-<version>.tar.gz and .whl
twine check dist/*
twine upload dist/*             # needs a PyPI account and API token
```

Bump `version` in `pyproject.toml` for every release, because PyPI never accepts the same version twice. Never commit `.env`, `config.yaml`, `state.db` or `runs/` (all are in `.gitignore`). Add a `LICENSE` file before publishing: without one, nobody else is legally allowed to use the code.

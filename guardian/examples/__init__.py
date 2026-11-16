"""Example training script and templates shipped with Rig Guardian."""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
TOY_SCRIPT = HERE / "train_toy.py"


def toy_template(python: str) -> dict:
    """A tiny model that trains in seconds (on CPU too), for testing the whole agent:
    kill it, pause it, force an OOM (simulate_oom_above=16), hang it (hang_at_step=40)."""
    return {
        "name": "toy-mlp",
        "description": "tiny MLP on synthetic data, for testing the agent",
        "script": str(TOY_SCRIPT),
        "python": python,
        "gpus": 1,
        "params_millions": 0.01,
        "best_metric": "val_loss",
        "mode": "min",
        "defaults": {"lr": 3.0e-3, "epochs": 3, "micro_batch": 32, "grad_accum": 2, "seed": 0, "step_sleep": 0.0},
        "overridable": {
            "lr": {"type": "float", "min": 1.0e-6, "max": 1.0e-1},
            "epochs": {"type": "int", "min": 1, "max": 300},
            "micro_batch": {"type": "int", "choices": [8, 16, 32, 64, 128, 256, 512]},
            "seed": {"type": "int", "min": 0, "max": 100000},
            "step_sleep": {"type": "float", "min": 0, "max": 5},
            "val_sleep": {"type": "float", "min": 0, "max": 600},
            "simulate_oom_above": {"type": "int", "min": 1, "max": 4096},
            "oom_at_step": {"type": "int", "min": 0, "max": 100000},
            "hang_at_step": {"type": "int", "min": 0, "max": 100000},
            "crash_at_step": {"type": "int", "min": 0, "max": 100000},
        },
    }

"""Driver helpers: run train_h.py in a bare run folder (no agent), with deaths, until it is done."""
import json, os, shutil, signal, subprocess, sys, tempfile, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
SETTINGS = {"every_steps": 5, "every_minutes": 20, "min_free_disk_gb": 0, "heartbeat_s": 0.2,
            "keep_last": 3, "keep_best": 1, "async": True, "probe_steps": 20}

def mk(cfg, settings=None, root=None):
    rd = Path(tempfile.mkdtemp(prefix="rg_", dir=root or os.environ.get("RG_TMP", "/tmp")))
    (rd / "config.json").write_text(json.dumps(cfg))
    (rd / "guardian.json").write_text(json.dumps({**SETTINGS, **(settings or {})}))
    return rd

def launch(rd, env=None, script="train_h.py", wait=True):
    e = {**os.environ, "GUARDIAN_RUN_DIR": str(rd), "PYTHONPATH": f"{REPO}:{REPO / 'guardian'}",
         "PYTHONUNBUFFERED": "1", **(env or {})}
    f = open(rd / "stdout.log", "a")
    p = subprocess.Popen([PY, str(Path(__file__).parent / script)], env=e, stdout=f, stderr=subprocess.STDOUT)
    if wait:
        return p.wait()
    return p

def result(rd):
    return json.loads((rd / "result.json").read_text())

def metrics(rd):
    rows = []
    for line in (rd / "metrics.jsonl").read_text().splitlines():
        rows.append(json.loads(line))
    return rows

def losses(rd):
    return {r["step"]: r["loss"] for r in metrics(rd) if "loss" in r}  # later rows win

def run_with_deaths(cfg, deaths, settings=None, script="train_h.py"):
    """deaths: list of env dicts, one per launch that should die; then a final clean launch."""
    rd = mk(cfg, settings)
    codes = []
    for env in deaths:
        codes.append(launch(rd, env, script))
    codes.append(launch(rd, None, script))
    return rd, codes

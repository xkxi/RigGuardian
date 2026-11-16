"""Checkpoint manager: atomic commit, verification, resume lookup and retention.

Used by the daemon (verify, prune, clean up after a crash) and by the training SDK
(commit after ``torch.save``). Stdlib only: it never imports torch, so the daemon stays
light and the SDK can import it from any training environment.

On-disk layout of ``runs/<job_id>/checkpoints/``::

    step_000500.pt      the checkpoint (written as .pt.tmp, then renamed)
    step_000500.json    its manifest (size, SHA-256, step, reason, metric, config hash)
    LATEST              name of the newest committed checkpoint

A checkpoint is only trusted when its file exists and its size and SHA-256 match the
manifest, so a crash mid-save can never replace a good checkpoint with a broken one.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from guardian.fsutil import atomic_write_json, atomic_write_text, fsync_dir, read_json

CKPT_RE = re.compile(r"^step_(\d+)\.pt$")
MANIFEST_RE = re.compile(r"^step_(\d+)\.json$")
ORPHAN_MANIFEST_AGE_S = 600


def sha256_file(path: str | os.PathLike, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def disk_free_bytes(path: str | os.PathLike) -> int:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class CheckpointStore:
    def __init__(self, directory: str | os.PathLike):
        self.dir = Path(directory)

    # ---- names -------------------------------------------------------------------
    @staticmethod
    def name_for(step: int) -> str:
        return f"step_{step:06d}.pt"

    def path_for(self, step: int) -> Path:
        return self.dir / self.name_for(step)

    def tmp_path(self, step: int) -> Path:
        return self.dir / (self.name_for(step) + ".tmp")

    def manifest_path(self, step: int) -> Path:
        return self.dir / f"step_{step:06d}.json"

    def rng_path(self, step: int, rank: int) -> Path:
        """Random-generator state of one multi-GPU rank (rank 0's is inside the checkpoint)."""
        return self.dir / f"step_{step:06d}.rng{rank}"

    def set_aside(self, name: str) -> None:
        """Take a damaged checkpoint out of the way (kept as ``<name>.bad``, with its manifest
        removed), so it is neither offered for a resume nor mistaken for a good save."""
        m = CKPT_RE.match(name)
        if not m:
            return
        try:
            os.replace(self.dir / name, self.dir / (name + ".bad"))
        except OSError:
            pass
        try:
            self.manifest_path(int(m.group(1))).unlink()
        except OSError:
            pass

    def ensure(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

    # ---- write protocol (steps 3-6; the caller did 1-2) ----------------------------
    def commit(self, tmp: str | os.PathLike, manifest: dict[str, Any]) -> dict[str, Any]:
        """Flush, checksum, write the manifest, rename into place and point LATEST at it."""
        tmp = Path(tmp)
        step = int(manifest["step"])
        final = self.path_for(step)
        with open(tmp, "rb") as f:  # 3. flush: physically on disk, not just in the OS cache
            os.fsync(f.fileno())
        man = dict(manifest)
        man.update(  # 4. checksum + manifest
            file=final.name,
            bytes=tmp.stat().st_size,
            sha256=sha256_file(tmp),
            saved_at=_utc_now(),
        )
        atomic_write_json(self.manifest_path(step), man)
        os.replace(tmp, final)  # 5. commit: atomic rename
        fsync_dir(self.dir)
        try:
            atomic_write_text(self.dir / "LATEST", final.name + "\n")  # 6. point
        except OSError:
            pass  # the checkpoint is committed; LATEST is only a pointer for people and tools
        return man

    def update_manifest(self, step: int, **fields: Any) -> dict[str, Any] | None:
        man = read_json(self.manifest_path(step))
        if man is None:
            return None
        man.update(fields)
        atomic_write_json(self.manifest_path(step), man)
        return man

    # ---- reading -----------------------------------------------------------------------
    def manifests(self) -> list[dict[str, Any]]:
        """All readable manifests, newest step first."""
        out = []
        if not self.dir.is_dir():
            return out
        for p in self.dir.iterdir():
            if MANIFEST_RE.match(p.name):
                man = read_json(p)
                if isinstance(man, dict) and "step" in man and "file" in man:
                    out.append(man)
        out.sort(key=lambda m: int(m["step"]), reverse=True)
        return out

    def manifest_for_step(self, step: int) -> dict[str, Any] | None:
        man = read_json(self.manifest_path(step))
        if man and (self.dir / man.get("file", "")).exists():
            return man
        return None

    def latest_name(self) -> str | None:
        try:
            return (self.dir / "LATEST").read_text().strip() or None
        except OSError:
            return None

    def verify(self, man: dict[str, Any], config_hash: str | None = None, deep: bool = True) -> str | None:
        """Return None when the checkpoint is trustworthy, otherwise the reason it is not."""
        path = self.dir / man.get("file", "")
        if not path.is_file():
            return "file missing"
        size = path.stat().st_size
        if size != man.get("bytes"):
            return f"size {size} != manifest {man.get('bytes')}"
        if config_hash and man.get("config_hash") and man["config_hash"] != config_hash:
            return "config hash differs from this job"
        if deep and sha256_file(path) != man.get("sha256"):
            return "SHA-256 mismatch"
        return None

    def find_latest_valid(
        self, config_hash: str | None = None, deep: bool = True
    ) -> tuple[dict[str, Any] | None, list[tuple[str, str]]]:
        """Resume lookup: the newest checkpoint that verifies, by step. ``LATEST`` is only a
        pointer for people and tools; a save that died before updating it still counts.

        ``deep=False`` skips hashing the file (presence, size and settings are still checked);
        use it for status messages, never for the checkpoint a job is about to load."""
        failures: list[tuple[str, str]] = []
        for man in self.manifests():  # newest step first
            err = self.verify(man, config_hash, deep=deep)
            if err is None:
                return man, failures
            failures.append((man["file"], err))
        return None, failures

    # ---- housekeeping ----------------------------------------------------------------
    def cleanup_tmp(self) -> list[str]:
        """Delete half-written files left by a crash or power cut."""
        removed = []
        if not self.dir.is_dir():
            return removed
        for p in self.dir.iterdir():
            if p.name.endswith(".tmp"):
                try:
                    p.unlink()
                    removed.append(p.name)
                except OSError:
                    pass
        return removed

    def prune(self, keep_last: int, keep_best: int, mode: str = "min") -> list[str]:
        """Keep the newest ``keep_last``, the best ``keep_best`` and any ``final``. Delete the rest.

        Call only right after a newer checkpoint was committed, so nothing is deleted
        before its replacement exists.
        """
        removed: list[str] = []
        mans = self.manifests()
        now = time.time()
        live, orphans = [], []
        for m in mans:
            (live if (self.dir / m["file"]).exists() else orphans).append(m)
        for m in orphans:  # manifest written but rename never happened (crash mid-save)
            mp = self.manifest_path(int(m["step"]))
            try:
                if now - mp.stat().st_mtime > ORPHAN_MANIFEST_AGE_S:
                    mp.unlink()
                    removed.append(mp.name)
            except OSError:
                pass

        keep: set[int] = {int(m["step"]) for m in live[: max(keep_last, 1)]}
        scored = [m for m in live if isinstance(m.get("metric"), (int, float)) and math.isfinite(m["metric"])]
        scored.sort(key=lambda m: float(m["metric"]), reverse=(mode == "max"))
        keep |= {int(m["step"]) for m in scored[:keep_best]}
        keep |= {int(m["step"]) for m in live if m.get("final")}
        latest = self.latest_name()
        keep |= {int(m["step"]) for m in live if m["file"] == latest}

        for m in live:
            step = int(m["step"])
            if step in keep:
                continue
            for p in (self.dir / m["file"], self.manifest_path(step)):
                try:
                    p.unlink()
                    removed.append(p.name)
                except OSError:
                    pass
        # side files of checkpoints that no longer exist, and all but the newest damaged file
        have = {int(m["step"]) for m in live if int(m["step"]) in keep}
        newest = max(have, default=-1)  # a newer step may be half-way through being saved: leave it
        bad = sorted(p for p in self.dir.glob("step_*.pt.bad"))
        for p in list(self.dir.glob("step_*.rng*")) + bad[:-1]:
            m2 = re.match(r"^step_(\d+)\.", p.name)
            step2 = int(m2.group(1)) if m2 else newest + 1
            if p.name.endswith(".bad") or (step2 < newest and step2 not in have and not p.name.endswith(".tmp")):
                try:
                    p.unlink()
                    removed.append(p.name)
                except OSError:
                    pass
        if removed:
            fsync_dir(self.dir)
        return removed

    def newest_bytes(self) -> int:
        for m in self.manifests():
            if (self.dir / m["file"]).exists():
                return int(m.get("bytes") or 0)
        return 0

    def total_bytes(self) -> int:
        if not self.dir.is_dir():
            return 0
        return sum(p.stat().st_size for p in self.dir.iterdir() if p.is_file())

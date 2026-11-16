import os

from guardian.checkpoints import CheckpointStore


def save(store: CheckpointStore, step: int, payload: bytes = b"", **meta):
    store.ensure()
    tmp = store.tmp_path(step)
    tmp.write_bytes(payload or f"weights at {step}".encode() * 100)
    return store.commit(tmp, {"step": step, "config_hash": "h", **meta})


def test_commit_and_latest(tmp_path):
    s = CheckpointStore(tmp_path)
    save(s, 5)
    m = save(s, 10)
    assert (tmp_path / "LATEST").read_text().strip() == "step_000010.pt"
    assert m["bytes"] > 0 and len(m["sha256"]) == 64
    man, failures = s.find_latest_valid("h")
    assert man["step"] == 10 and failures == []


def test_corrupt_byte_falls_back_to_previous(tmp_path):
    s = CheckpointStore(tmp_path)
    save(s, 5)
    save(s, 10)
    p = tmp_path / "step_000010.pt"
    data = bytearray(p.read_bytes())
    data[3] ^= 0xFF
    p.write_bytes(bytes(data))
    man, failures = s.find_latest_valid("h")
    assert man["step"] == 5
    assert failures and "SHA-256" in failures[0][1]


def test_half_written_tmp_is_ignored_and_cleaned(tmp_path):
    s = CheckpointStore(tmp_path)
    save(s, 5)
    (tmp_path / "step_000010.pt.tmp").write_bytes(b"partial")  # killed mid-save
    man, _ = s.find_latest_valid("h")
    assert man["step"] == 5
    assert s.cleanup_tmp() == ["step_000010.pt.tmp"]


def test_crash_between_manifest_and_rename(tmp_path):
    s = CheckpointStore(tmp_path)
    save(s, 5)
    # manifest for step 10 written, but the rename never happened
    (tmp_path / "step_000010.json").write_text('{"step": 10, "file": "step_000010.pt", "bytes": 3, "sha256": "x"}')
    man, _ = s.find_latest_valid("h")
    assert man["step"] == 5
    os.unlink(tmp_path / "LATEST")  # even when the orphan is the newest manifest
    man, failures = s.find_latest_valid("h")
    assert man["step"] == 5
    assert ("step_000010.pt", "file missing") in failures


def test_config_hash_mismatch_refused(tmp_path):
    s = CheckpointStore(tmp_path)
    save(s, 5)
    man, failures = s.find_latest_valid("other")
    assert man is None and "config hash" in failures[0][1]


def test_missing_latest_uses_newest_manifest(tmp_path):
    s = CheckpointStore(tmp_path)
    save(s, 5)
    save(s, 10)
    os.unlink(tmp_path / "LATEST")
    man, _ = s.find_latest_valid("h")
    assert man["step"] == 10


def test_retention_keeps_last_best_and_final(tmp_path):
    s = CheckpointStore(tmp_path)
    save(s, 1, metric=0.5, final=False)
    save(s, 2, metric=0.1)  # best
    save(s, 3, metric=0.9)
    save(s, 4)
    save(s, 5)
    save(s, 6, final=True)
    s.prune(keep_last=3, keep_best=1, mode="min")
    left = sorted(int(m["step"]) for m in s.manifests())
    assert left == [2, 4, 5, 6]
    assert not (tmp_path / "step_000001.pt").exists()

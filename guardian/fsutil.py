"""Small file helpers shared by the daemon and the training SDK (stdlib only)."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


def fsync_dir(path: str | os.PathLike) -> None:
    """Flush a directory entry so a rename inside it survives power loss."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass  # some filesystems refuse fsync on directories
    finally:
        os.close(fd)


def atomic_write_bytes(path: str | os.PathLike, data: bytes, durable: bool = True) -> None:
    """Write ``data`` so readers see either the old file or the new one, never half of it."""
    path = Path(path)
    # a temp name of its own for every writer: the daemon and the CLI may write the same file
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            if durable:
                os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    if durable:
        fsync_dir(path.parent)


def atomic_write_text(path: str | os.PathLike, text: str, durable: bool = True) -> None:
    atomic_write_bytes(path, text.encode(), durable=durable)


def atomic_write_json(path: str | os.PathLike, obj: Any, durable: bool = True) -> None:
    atomic_write_text(path, json.dumps(obj, indent=1, sort_keys=True, default=str), durable=durable)


def read_json(path: str | os.PathLike, default: Any = None) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def tail_lines(path: str | os.PathLike, n: int = 30, max_bytes: int = 256_000) -> list[str]:
    """Last ``n`` lines of a text file without reading all of it."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read()
    except OSError:
        return []
    lines = data.decode(errors="replace").splitlines()
    return lines[-n:]

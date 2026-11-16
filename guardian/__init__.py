"""Rig Guardian: a background agent that runs, watches and checkpoints training jobs.

This package is imported by training processes too (through the SDK), so keep this
module free of heavy imports. ``fsutil`` and ``checkpoints`` must stay stdlib-only.
"""

__version__ = "0.1.0"

EXIT_FINISHED = 0
EXIT_STOPPED = 86
EXIT_OOM = 87

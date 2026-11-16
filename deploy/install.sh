#!/bin/sh
# Rig Guardian installer: a virtual environment with the agent in it, then `rig-guardian init`.
# Usage: deploy/install.sh [init options, e.g. --agent-gpu 2 --exclude 3]
# Run from the repository as your normal user (not root).
set -eu
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else "Rig Guardian needs Python 3.11 or newer (set PYTHON=/path/to/python3.11)")'
VENV=${RIG_GUARDIAN_VENV:-$HOME/.local/share/rig-guardian/venv}
[ -d "$VENV" ] || "$PY" -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -e .
mkdir -p "$HOME/.local/bin"
ln -sf "$VENV/bin/rig-guardian" "$HOME/.local/bin/rig-guardian"
echo "Installed: $HOME/.local/bin/rig-guardian (make sure ~/.local/bin is on your PATH)"
if ! "$VENV/bin/rig-guardian" init "$@"; then
  echo "The command is installed, but 'rig-guardian init' did not finish (see the message above)." >&2
  echo "Fix that, then run: rig-guardian init" >&2
  exit 1
fi

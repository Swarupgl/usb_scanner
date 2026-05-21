#!/usr/bin/env bash
set -euo pipefail

# Wrapper for systemd/udev: runs the scan using the repo venv if present.
# Usage: usb-sanitize-run.sh /dev/sda1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PY="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

exec "$PY" -u "$REPO_ROOT/hardware/pi/usb_scan_once.py" "$@"


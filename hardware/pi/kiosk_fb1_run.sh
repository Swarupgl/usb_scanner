#!/usr/bin/env bash
set -euo pipefail

DISPLAY_NUM=":0"
XSOCK="/tmp/.X11-unix/X0"
XORG_LOG="/home/pi/.local/share/xorg/Xorg.0.log"
XORG_CONF="/home/pi/Ai/hardware/pi/xorg/xorg-fb1.conf"

# Start Xorg on the SPI TFT framebuffer (/dev/fb1).
# -ac disables access control (safe-ish with -nolisten tcp), which avoids Xauthority hassles.
/usr/bin/Xorg "$DISPLAY_NUM" \
  -config "$XORG_CONF" \
  -nolisten tcp \
  -nocursor \
  -ac \
  vt2 \
  >>"$XORG_LOG" 2>&1 &
XORG_PID=$!

cleanup() {
  if kill -0 "$XORG_PID" 2>/dev/null; then
    kill "$XORG_PID" 2>/dev/null || true
    sleep 0.5
    kill -KILL "$XORG_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Wait for X socket or Xorg death.
for _ in $(seq 1 50); do
  if [[ -S "$XSOCK" ]]; then
    break
  fi
  if ! kill -0 "$XORG_PID" 2>/dev/null; then
    echo "Xorg exited early; check $XORG_LOG" >&2
    exit 1
  fi
  sleep 0.1
done

if [[ ! -S "$XSOCK" ]]; then
  echo "Timed out waiting for X socket; check $XORG_LOG" >&2
  exit 1
fi

# Run the kiosk as the pi user.
exec /bin/su - pi -c 'DISPLAY='"$DISPLAY_NUM"' bash -lc "cd /home/pi/Ai; if [ -x .venv/bin/python ]; then .venv/bin/python -u hardware/pi/kiosk_ui.py; else python3 -u hardware/pi/kiosk_ui.py; fi"'

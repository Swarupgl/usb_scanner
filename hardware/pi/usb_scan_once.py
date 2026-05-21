from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _read_extra_args_file(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    lines: list[str] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        lines.append(ln)
    if not lines:
        return []
    # Allow multi-line; treat as whitespace-separated argv.
    return shlex.split(" ".join(lines))


def _get_mountpoint(device: str) -> Optional[str]:
    # Works for /dev/sda1 etc.
    p = _run(["lsblk", "-no", "MOUNTPOINT", device])
    if p.returncode != 0:
        return None
    lines = [ln.strip() for ln in (p.stdout or "").splitlines() if ln.strip()]
    if not lines:
        return None
    return lines[-1]


def _device_exists(device: str) -> bool:
    try:
        return os.path.exists(device)
    except Exception:
        return False


def _wait_for_device_node(device: str, timeout_s: float = 10.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _device_exists(device):
            return True
        time.sleep(0.2)
    return False


def _mount_readonly(device: str, mount_base: Path) -> Optional[str]:
    """Try to mount a block device read-only with safe flags.

    Returns mountpoint if successful, else None.
    """
    dev_name = os.path.basename(device)
    mp = (mount_base / dev_name).resolve()
    mp.mkdir(parents=True, exist_ok=True)

    # ro + execution hardening. umask helps for vfat/exfat.
    opts = "ro,nosuid,nodev,noexec,umask=077"
    p = _run(["mount", "-o", opts, device, str(mp)])
    if p.returncode == 0:
        return str(mp)

    # Cleanup dir on failure (best-effort)
    try:
        mp.rmdir()
    except Exception:
        pass
    return None


def _unmount(mountpoint: str) -> None:
    _ = _run(["umount", mountpoint])


def _wait_for_mount(device: str, timeout_s: float = 30.0) -> Optional[str]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        mp = _get_mountpoint(device)
        if mp:
            return mp
        time.sleep(0.5)
    return None


def _default_repo_root() -> Path:
    # hardware/pi/usb_scan_once.py -> repo root
    return Path(__file__).resolve().parents[2]


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="USB scan runner (udev/systemd-friendly)")
    p.add_argument(
        "target",
        help="Either a block device (e.g., /dev/sda1) or an already-mounted path (e.g., /media/pi/USB)",
    )
    p.add_argument("--repo", default=str(_default_repo_root()), help="Path to repo root on the Pi")
    p.add_argument(
        "--quarantine",
        default=str(Path("outputs") / "quarantine"),
        help="Quarantine folder (relative to repo unless absolute)",
    )
    p.add_argument("--onnx", default=None, help="Optional MalConv ONNX model path (passed to pi_usb_sanitizer.py)")
    p.add_argument("--yara-rules", default=None, help="Optional YARA rules file path")
    p.add_argument("--entropy-threshold", type=float, default=7.5)
    p.add_argument("--risk-threshold", type=float, default=0.5)
    p.add_argument(
        "--status-json",
        default=None,
        help="Status JSON output path (default: <repo>/outputs/pi/logs/status.json)",
    )
    p.add_argument(
        "--log",
        default=None,
        help="Log file path (default: <repo>/outputs/pi/logs/usb_sanitizer.log)",
    )
    p.add_argument("--mount-timeout", type=float, default=30.0)
    p.add_argument(
        "--try-ro-mount",
        action="store_true",
        help="If target is a block device and not mounted yet, attempt a read-only mount with noexec/nodev/nosuid before falling back to waiting for automount.",
    )
    p.add_argument(
        "--extra-args-file",
        default=None,
        help="Optional file containing extra arguments to append to pi_usb_sanitizer.py (default: <repo>/hardware/pi/usb-sanitizer.conf if present)",
    )

    args = p.parse_args(argv)

    repo = Path(args.repo).resolve()
    status_path = (
        Path(args.status_json).resolve() if args.status_json else (repo / "outputs" / "pi" / "logs" / "status.json")
    )
    log_path = Path(args.log).resolve() if args.log else (repo / "outputs" / "pi" / "logs" / "usb_sanitizer.log")

    quarantine = args.quarantine
    quarantine_path = Path(quarantine)
    if not quarantine_path.is_absolute():
        quarantine_path = (repo / quarantine_path).resolve()

    target = args.target
    mountpoint: Optional[str]
    mounted_by_us = False
    mount_base = Path("/mnt/usb-scan")

    if os.path.isdir(target):
        mountpoint = os.path.abspath(target)
        device = None
    else:
        device = target
        _wait_for_device_node(device, timeout_s=10.0)

        mountpoint = _get_mountpoint(device)
        if not mountpoint and args.try_ro_mount:
            mp = _mount_readonly(device, mount_base=mount_base)
            if mp:
                mountpoint = mp
                mounted_by_us = True

        if not mountpoint:
            mountpoint = _wait_for_mount(device, timeout_s=float(args.mount_timeout))

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    if not mountpoint:
        _atomic_write_json(
            status_path,
            {
                "state": "error",
                "time": now,
                "device": device,
                "mountpoint": None,
                "message": f"Timed out waiting for mount: {device}",
            },
        )
        return 2

    sanitizer = repo / "hardware" / "pi" / "pi_usb_sanitizer.py"
    if not sanitizer.exists():
        _atomic_write_json(
            status_path,
            {
                "state": "error",
                "time": now,
                "device": device,
                "mountpoint": mountpoint,
                "message": f"Missing pi_usb_sanitizer.py at: {sanitizer}",
            },
        )
        return 2

    cmd = [
        sys.executable,
        str(sanitizer),
        "--scan",
        mountpoint,
        "--quarantine",
        str(quarantine_path),
        "--entropy-threshold",
        str(args.entropy_threshold),
        "--risk-threshold",
        str(args.risk_threshold),
    ]
    if args.yara_rules:
        cmd += ["--yara-rules", args.yara_rules]
    if args.onnx:
        cmd += ["--onnx", args.onnx]

    extra_file = (
        Path(args.extra_args_file).expanduser().resolve()
        if args.extra_args_file
        else (repo / "hardware" / "pi" / "usb-sanitizer.conf")
    )
    extra_args = _read_extra_args_file(extra_file)
    if extra_args:
        cmd += extra_args

    status = {
        "state": "scanning",
        "time": now,
        "device": device,
        "mountpoint": mountpoint,
        "mounted_by_runner": mounted_by_us,
        "cmd": cmd,
        "progress": {"scanned": 0, "quarantined": 0, "errors": 0},
        "last_line": None,
    }
    _atomic_write_json(status_path, status)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n=== {now} scan start device={device} mount={mountpoint} ===\n")
        logf.flush()

        proc = subprocess.Popen(cmd, cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None

        for line in proc.stdout:
            line = line.rstrip("\n")
            logf.write(line + "\n")
            logf.flush()

            status["last_line"] = line
            if line.startswith("[error]"):
                status["progress"]["errors"] += 1
            if line.startswith("[quarantined"):
                status["progress"]["quarantined"] += 1
            if line.startswith("[") and "]" in line and "::" in line:
                status["progress"]["scanned"] += 1

            _atomic_write_json(status_path, status)

        rc = proc.wait()

    if mounted_by_us and mountpoint:
        try:
            _unmount(mountpoint)
            # Best-effort remove mount dir
            try:
                Path(mountpoint).rmdir()
            except Exception:
                pass
        except Exception:
            pass

    status["state"] = "done" if rc == 0 else "error"
    status["time_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status["returncode"] = rc
    _atomic_write_json(status_path, status)

    return 0 if rc == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

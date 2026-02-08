from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import psutil
import torch

from model import MalConv
from preprocess import file_to_tensor


def load_model(checkpoint_path: str, device: torch.device) -> Optional[MalConv]:
    if not os.path.exists(checkpoint_path):
        return None

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        max_len = int(ckpt.get("max_len", 1048576))
        window_size = int(ckpt.get("window_size", 512))
        model = MalConv(input_length=max_len, window_size=window_size).to(device)
        model.load_state_dict(ckpt["state_dict"])
    else:
        # Back-compat: raw state_dict
        model = MalConv().to(device)
        model.load_state_dict(ckpt)

    model.eval()
    return model


def scan_path(model: Optional[MalConv], device: torch.device, root_path: str, threshold: float, max_len: int) -> None:
    root_path = os.path.abspath(root_path)
    print(f"Scanning: {root_path}")

    for root, _, files in os.walk(root_path):
        for name in files:
            lower = name.lower()
            if not any(lower.endswith(ext) for ext in scan_path.extensions):
                continue

            path = os.path.join(root, name)
            data = file_to_tensor(path, max_len=max_len)
            if data is None:
                continue

            if model is None:
                print(f"[NO MODEL] {path}")
                continue

            with torch.no_grad():
                pred = model(data.to(device)).item()

            status = "MALICIOUS" if pred >= threshold else "CLEAN"
            print(f"[{status}] {path} (score={pred:.4f})")


# default extensions; overridden by CLI
scan_path.extensions = (".exe",)


def watch_usb(model: Optional[MalConv], device: torch.device, threshold: float, max_len: int, poll_seconds: int) -> None:
    print("Watching for new drives...")
    seen_devices = {d.device for d in psutil.disk_partitions(all=False)}

    while True:
        partitions = psutil.disk_partitions(all=False)
        current_devices = {p.device for p in partitions}
        new_devices = current_devices - seen_devices

        for p in partitions:
            if p.device in new_devices:
                # On Windows, mountpoint is typically like 'E:\\'
                try:
                    scan_path(model, device, p.mountpoint, threshold=threshold, max_len=max_len)
                except PermissionError:
                    print(f"Permission denied scanning {p.mountpoint}")
                except Exception as exc:
                    print(f"Error scanning {p.mountpoint}: {exc}")

        seen_devices = current_devices
        time.sleep(poll_seconds)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="USB watcher + .exe scanner using MalConv")
    parser.add_argument("--checkpoint", type=str, default="malconv_model.pth")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-len", type=int, default=1048576, help="Used if checkpoint doesn't specify")
    parser.add_argument("--poll", type=int, default=3)
    parser.add_argument(
        "--extensions",
        type=str,
        default=".exe",
        help="Comma-separated list of file extensions to scan (e.g. .exe,.dll,.com)",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--watch", action="store_true", help="Watch for new drives and scan")
    mode.add_argument("--scan", type=str, default=None, help="Scan a specific path (folder or drive)")

    args = parser.parse_args(argv)

    exts = tuple(
        e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}"
        for e in args.extensions.split(",")
        if e.strip()
    )
    if not exts:
        exts = (".exe",)
    scan_path.extensions = exts

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device=device)
    if model is None:
        print(f"Warning: checkpoint not found at '{args.checkpoint}'. Scans will run without predictions.")

    max_len = getattr(model, "input_length", args.max_len) if model is not None else args.max_len

    if args.watch:
        watch_usb(model, device, threshold=args.threshold, max_len=max_len, poll_seconds=args.poll)
    else:
        scan_path(model, device, args.scan, threshold=args.threshold, max_len=max_len)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

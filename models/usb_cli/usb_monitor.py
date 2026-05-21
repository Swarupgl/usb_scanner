from __future__ import annotations

import argparse
import hashlib
import math
import os
import shutil
import tempfile
import time
from typing import Optional
import zipfile

import psutil
import torch

from ..malconv.model import MalConv
from ..malconv.preprocess import file_to_tensor


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def shannon_entropy(path: str, max_bytes: int = 256 * 1024) -> Optional[float]:
    """Compute Shannon entropy over the first max_bytes of a file.

    Returns a value in [0, 8] for byte distributions.
    """
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes)
        if not data:
            return 0.0
        counts = [0] * 256
        for b in data:
            counts[b] += 1
        n = len(data)
        ent = 0.0
        for c in counts:
            if c:
                p = c / n
                ent -= p * math.log2(p)
        return ent
    except Exception:
        return None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_relpath(path: str, start: str) -> str:
    try:
        rel = os.path.relpath(path, start)
    except Exception:
        rel = os.path.basename(path)
    rel = rel.replace("..", "_")
    return rel


def apply_action_on_file(path: str, action: str, quarantine_dir: str, root_base: str) -> None:
    if action == "report":
        return

    if action == "delete":
        os.remove(path)
        print(f"[ACTION] Deleted: {path}")
        return

    if action == "quarantine":
        rel = safe_relpath(path, root_base)
        dest = os.path.join(quarantine_dir, rel)
        ensure_dir(os.path.dirname(dest))
        # Avoid overwrite collisions
        if os.path.exists(dest):
            base, ext = os.path.splitext(dest)
            dest = f"{base}_{int(time.time())}{ext}"
        shutil.move(path, dest)
        print(f"[ACTION] Quarantined: {path} -> {dest}")
        return

    raise ValueError(f"Unknown action: {action}")


def score_file(model: Optional[MalConv], device: torch.device, path: str, max_len: int) -> Optional[float]:
    data = file_to_tensor(path, max_len=max_len)
    if data is None:
        return None
    if model is None:
        return None
    with torch.no_grad():
        return float(model(data.to(device)).item())


def iter_zip_members(zip_path: str):
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            yield zf, info


def scan_zip(
    model: Optional[MalConv],
    device: torch.device,
    zip_path: str,
    threshold: float,
    max_len: int,
    extensions: tuple[str, ...],
    *,
    show_metadata: bool,
    show_sha256: bool,
    action: str,
    quarantine_dir: str,
    root_base: str,
    zip_mode: str,
) -> list[tuple[float, str, str]]:
    """Scan a .zip archive.

    Returns list of (score, status, member_display_path) for members that were scanned.

    zip_mode:
      - 'quarantine-archive': if any member is flagged, action applies to the ZIP file itself
      - 'sanitize': create a new ZIP without flagged members (does not modify original)
    """
    results: list[tuple[float, str, str]] = []
    flagged_members: set[str] = set()

    if zip_mode not in ("quarantine-archive", "sanitize"):
        raise ValueError(f"Unknown zip_mode: {zip_mode}")

    print(f"[ZIP] Scanning archive: {zip_path}")

    with tempfile.TemporaryDirectory(prefix="usbscan_zip_") as tmpdir:
        for zf, info in iter_zip_members(zip_path):
            name = info.filename
            lower = name.lower()
            if not any(lower.endswith(ext) for ext in extensions):
                continue

            # Extract to temp and score
            extracted_path = os.path.join(tmpdir, name)
            ensure_dir(os.path.dirname(extracted_path))
            try:
                with zf.open(info, "r") as src, open(extracted_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            except Exception as exc:
                print(f"[ZIP] Error extracting {name}: {exc}")
                continue

            pred = score_file(model, device, extracted_path, max_len=max_len)
            if pred is None:
                continue

            status = "MALICIOUS" if pred >= threshold else "CLEAN"
            display = f"{zip_path}::{name}"
            results.append((pred, status, display))

            if pred >= threshold:
                flagged_members.add(name)

            if show_metadata or show_sha256:
                try:
                    size = os.path.getsize(extracted_path)
                except OSError:
                    size = None
                ent = shannon_entropy(extracted_path) if show_metadata else None
                digest = sha256_file(extracted_path) if show_sha256 else None

                meta_parts = [f"score={pred:.4f}"]
                if size is not None:
                    meta_parts.append(f"size={size}")
                if ent is not None:
                    meta_parts.append(f"entropy={ent:.3f}")
                if digest is not None:
                    meta_parts.append(f"sha256={digest}")
                print(f"[{status}] {display} ({', '.join(meta_parts)})")
            else:
                print(f"[{status}] {display} (score={pred:.4f})")

    if not flagged_members:
        return results

    if zip_mode == "quarantine-archive":
        # Apply action to the archive file itself
        try:
            ensure_dir(quarantine_dir)
            apply_action_on_file(zip_path, action=action, quarantine_dir=quarantine_dir, root_base=root_base)
        except Exception as exc:
            print(f"[ZIP] Could not apply action '{action}' to archive {zip_path}: {exc}")
        return results

    # zip_mode == 'sanitize'
    try:
        sanitized_path = zip_path[:-4] + ".sanitized.zip" if zip_path.lower().endswith(".zip") else zip_path + ".sanitized.zip"
        with zipfile.ZipFile(zip_path, "r") as zin, zipfile.ZipFile(sanitized_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                if info.is_dir():
                    continue
                if info.filename in flagged_members:
                    continue
                with zin.open(info, "r") as src:
                    data = src.read()
                zout.writestr(info, data)
        print(f"[ZIP] Sanitized archive created (flagged members removed): {sanitized_path}")
        if action in ("delete", "quarantine"):
            print(f"[ZIP] Note: original archive left unchanged; zip_mode='sanitize' does not delete/modify original by default.")
    except Exception as exc:
        print(f"[ZIP] Failed to create sanitized archive for {zip_path}: {exc}")

    return results


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


def scan_path(
    model: Optional[MalConv],
    device: torch.device,
    root_path: str,
    threshold: float,
    max_len: int,
    *,
    show_metadata: bool = False,
    show_sha256: bool = False,
    top_n: int = 0,
    scan_archives: bool = False,
    action: str = "report",
    quarantine_dir: str = "quarantine",
    zip_mode: str = "quarantine-archive",
) -> None:
    root_path = os.path.abspath(root_path)
    print(f"Scanning: {root_path}")

    results: list[tuple[float, str, str]] = []  # (score, status, path)

    action = action.lower().strip()
    zip_mode = zip_mode.lower().strip()

    def handle_candidate_file(path: str) -> None:
        lower = os.path.basename(path).lower()

        # Zip archive scanning (optional)
        if scan_archives and lower.endswith(".zip"):
            try:
                zip_results = scan_zip(
                    model,
                    device,
                    path,
                    threshold=threshold,
                    max_len=max_len,
                    extensions=scan_path.extensions,
                    show_metadata=show_metadata,
                    show_sha256=show_sha256,
                    action=action,
                    quarantine_dir=quarantine_dir,
                    root_base=root_path if os.path.isdir(root_path) else os.path.dirname(root_path),
                    zip_mode=zip_mode,
                )
                results.extend(zip_results)
            except zipfile.BadZipFile:
                print(f"[ZIP] Bad zip file: {path}")
            except Exception as exc:
                print(f"[ZIP] Error scanning zip {path}: {exc}")
            return

        if not any(lower.endswith(ext) for ext in scan_path.extensions):
            return

        pred = score_file(model, device, path, max_len=max_len)
        if pred is None:
            return
        status = "MALICIOUS" if pred >= threshold else "CLEAN"
        results.append((pred, status, path))

        if show_metadata or show_sha256:
            try:
                size = os.path.getsize(path)
            except OSError:
                size = None
            ent = shannon_entropy(path) if show_metadata else None
            digest = sha256_file(path) if show_sha256 else None

            meta_parts = [f"score={pred:.4f}"]
            if size is not None:
                meta_parts.append(f"size={size}")
            if ent is not None:
                meta_parts.append(f"entropy={ent:.3f}")
            if digest is not None:
                meta_parts.append(f"sha256={digest}")
            print(f"[{status}] {path} ({', '.join(meta_parts)})")
        else:
            print(f"[{status}] {path} (score={pred:.4f})")

        if pred >= threshold:
            try:
                ensure_dir(quarantine_dir)
                base = root_path if os.path.isdir(root_path) else os.path.dirname(root_path)
                apply_action_on_file(path, action=action, quarantine_dir=quarantine_dir, root_base=base)
            except Exception as exc:
                print(f"[ACTION] Failed to apply action '{action}' on {path}: {exc}")

    # Support scanning a single file path
    if os.path.isfile(root_path):
        handle_candidate_file(root_path)
        if top_n and results:
            print("\nTop suspicious (highest scores):")
            for score, status, path in sorted(results, key=lambda t: t[0], reverse=True)[: int(top_n)]:
                print(f"[{status}] {path} (score={score:.4f})")
        return

    for root, _, files in os.walk(root_path):
        for name in files:
            path = os.path.join(root, name)
            handle_candidate_file(path)

    if top_n and results:
        top_n = max(0, int(top_n))
        if top_n > 0:
            print("\nTop suspicious (highest scores):")
            for score, status, path in sorted(results, key=lambda t: t[0], reverse=True)[:top_n]:
                print(f"[{status}] {path} (score={score:.4f})")


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
    parser.add_argument("--checkpoint", type=str, default=str(os.path.join("outputs", "models", "malconv_model.pth")))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-len", type=int, default=1048576, help="Used if checkpoint doesn't specify")
    parser.add_argument("--poll", type=int, default=3)
    parser.add_argument("--top", type=int, default=0, help="If >0, print a Top-N suspicious summary")
    parser.add_argument("--metadata", action="store_true", help="Show file size + entropy in output")
    parser.add_argument("--sha256", action="store_true", help="Show SHA-256 hash in output")
    parser.add_argument(
        "--action",
        type=str,
        default="report",
        choices=("report", "quarantine", "delete"),
        help="What to do when a file is flagged: report|quarantine|delete",
    )
    parser.add_argument(
        "--quarantine-dir",
        type=str,
        default=str(os.path.join("outputs", "quarantine")),
        help="Where to move files when --action quarantine is used",
    )
    parser.add_argument(
        "--archives",
        action="store_true",
        help="Also scan .zip archives (scans executable-like members by extension)",
    )
    parser.add_argument(
        "--zip-mode",
        type=str,
        default="quarantine-archive",
        choices=("quarantine-archive", "sanitize"),
        help="How to handle flagged members in a zip: quarantine-archive|sanitize",
    )
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
        scan_path(
            model,
            device,
            args.scan,
            threshold=args.threshold,
            max_len=max_len,
            show_metadata=args.metadata,
            show_sha256=args.sha256,
            top_n=args.top,
            scan_archives=args.archives,
            action=args.action,
            quarantine_dir=args.quarantine_dir,
            zip_mode=args.zip_mode,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

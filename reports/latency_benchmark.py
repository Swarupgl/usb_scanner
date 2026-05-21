from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from models.malconv.dataset import BinaryFolderDataset
from models.malconv.model import MalConv
from models.malconv.preprocess import file_to_tensor


@dataclass(frozen=True)
class LatencyStats:
    n: int
    avg_ms: float
    median_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float
    files_per_sec: float


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]

    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _summarize(ms: list[float]) -> LatencyStats:
    ms_sorted = sorted(ms)
    n = len(ms_sorted)
    avg = sum(ms_sorted) / max(n, 1)
    median = _percentile(ms_sorted, 50)
    p90 = _percentile(ms_sorted, 90)
    mn = ms_sorted[0] if ms_sorted else 0.0
    mx = ms_sorted[-1] if ms_sorted else 0.0
    fps = 1000.0 / avg if avg > 0 else 0.0
    return LatencyStats(n=n, avg_ms=avg, median_ms=median, p90_ms=p90, min_ms=mn, max_ms=mx, files_per_sec=fps)


def _human(stats: LatencyStats) -> str:
    return (
        f"n={stats.n} avg={stats.avg_ms:.2f}ms median={stats.median_ms:.2f}ms "
        f"p90={stats.p90_ms:.2f}ms min={stats.min_ms:.2f}ms max={stats.max_ms:.2f}ms "
        f"throughput~{stats.files_per_sec:.2f} files/s"
    )


def _load_ckpt(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return ckpt


def _build_model(ckpt: dict) -> MalConv:
    max_len = int(ckpt.get("max_len", 1048576))
    window_size = int(ckpt.get("window_size", 512))
    model = MalConv(input_length=max_len, window_size=window_size)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model


@torch.no_grad()
def _time_infer_with_io(
    model: MalConv,
    paths: list[str],
    max_len: int,
    device: torch.device,
) -> list[float]:
    """End-to-end timing: read file -> tensorize -> forward pass."""
    times_ms: list[float] = []
    for p in paths:
        t0 = time.perf_counter()
        x = file_to_tensor(p, max_len=max_len)
        if x is None:
            continue
        x = x.to(device)
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    return times_ms


@torch.no_grad()
def _time_infer_model_only(
    model: MalConv,
    paths: list[str],
    max_len: int,
    device: torch.device,
) -> list[float]:
    """Model-only timing: preload tensors, then time only the forward pass."""
    xs: list[torch.Tensor] = []
    for p in paths:
        x = file_to_tensor(p, max_len=max_len)
        if x is None:
            continue
        xs.append(x)

    times_ms: list[float] = []
    for x in xs:
        x = x.to(device)
        t0 = time.perf_counter()
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    return times_ms


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Benchmark MalConv inference latency")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pth")
    p.add_argument("--benign-dir", default=os.path.join("datasets", "local", "benign"))
    p.add_argument("--malicious-dir", default=os.path.join("datasets", "local", "malicious"))
    p.add_argument("--max-files", type=int, default=20, help="Max files total to benchmark")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--warmup", type=int, default=3, help="Warmup runs (end-to-end) before measuring")
    p.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Set torch CPU threads (0 means leave default)",
    )
    args = p.parse_args(argv)

    if args.threads and args.device == "cpu":
        torch.set_num_threads(int(args.threads))

    ckpt = _load_ckpt(args.ckpt)
    max_len = int(ckpt.get("max_len", 1048576))
    model = _build_model(ckpt)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    model = model.to(device)
    model.eval()

    ds = BinaryFolderDataset(
        benign_dir=args.benign_dir,
        malicious_dir=args.malicious_dir,
        max_len=max_len,
        limit_per_class=None,
    )
    paths = [s.path for s in ds.samples]

    rng = random.Random(args.seed)
    rng.shuffle(paths)
    paths = paths[: max(1, min(args.max_files, len(paths)))]

    print(f"Checkpoint: {os.path.basename(args.ckpt)}")
    print(f"Device: {device}")
    print(f"max_len: {max_len} bytes")
    print(f"Benchmark files: {len(paths)}")

    # Warmup (end-to-end)
    if args.warmup > 0:
        _ = _time_infer_with_io(model, paths[: min(len(paths), args.warmup)], max_len=max_len, device=device)

    end_to_end_ms = _time_infer_with_io(model, paths, max_len=max_len, device=device)
    model_only_ms = _time_infer_model_only(model, paths, max_len=max_len, device=device)

    print("\nEnd-to-end (I/O + preprocess + model):")
    print(_human(_summarize(end_to_end_ms)))
    print("Model-only (preprocess excluded):")
    print(_human(_summarize(model_only_ms)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

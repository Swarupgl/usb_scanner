from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional

import torch

from models.malconv.model import MalConv


@dataclass(frozen=True)
class Stats:
    checkpoint: str
    file_size_bytes: int
    max_len: int
    window_size: int
    params: int
    params_trainable: int
    weights_bytes_fp32: int
    weights_bytes_int8: int
    conv1_macs: int
    conv2_macs: int
    gate_mul_ops: int
    fc1_macs: int
    fc2_macs: int
    total_macs: int


@dataclass(frozen=True)
class LayerCompute:
    name: str
    output_shape: tuple[str, ...]
    macs: int
    notes: str


def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    v = float(n)
    for u in units:
        if v < 1024.0 or u == units[-1]:
            return f"{v:.2f} {u}"
        v /= 1024.0
    return f"{v:.2f} GB"


def _count_params(model: torch.nn.Module) -> tuple[int, int, int]:
    total = 0
    trainable = 0
    bytes_fp32 = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        bytes_fp32 += n * 4  # float32
    return total, trainable, bytes_fp32


def _param_breakdown(model: torch.nn.Module) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for name, module in model.named_children():
        n = 0
        for p in module.parameters(recurse=True):
            n += p.numel()
        out.append((name, n))
    return out


def _estimate_conv_macs(max_len: int, window_size: int, in_ch: int = 8, out_ch: int = 128) -> int:
    # Conv1d output length (PyTorch): floor((L + 2P - D*(K-1) - 1)/S + 1)
    # Here P=0, D=1, so floor((L - K)/S + 1)
    if window_size <= 0 or max_len <= 0:
        return 0
    if max_len < window_size:
        out_len = 0
    else:
        out_len = ((max_len - window_size) // window_size) + 1
    macs_per_out = in_ch * window_size
    return int(out_ch * out_len * macs_per_out)


def _conv_out_len(max_len: int, window_size: int) -> int:
    if window_size <= 0 or max_len <= 0:
        return 0
    if max_len < window_size:
        return 0
    return ((max_len - window_size) // window_size) + 1


def _layer_compute(max_len: int, window_size: int) -> list[LayerCompute]:
    """Compute a simple per-layer operation report for MalConv.

    Notes:
      - We report MACs for conv/linear layers (common proxy for compute).
      - Nonlinearities (sigmoid/relu), indexing (embedding), and pooling are reported as notes.
    """

    in_ch = 8
    out_ch = 128
    out_len = _conv_out_len(max_len=max_len, window_size=window_size)

    conv_macs = _estimate_conv_macs(max_len=max_len, window_size=window_size, in_ch=in_ch, out_ch=out_ch)
    gate_mul = int(out_ch * out_len)  # elementwise multiply gated=value*gate
    fc1_macs = 128 * 128
    fc2_macs = 128 * 1

    return [
        LayerCompute(
            name="embed",
            output_shape=("B", str(in_ch), str(max_len)),
            macs=0,
            notes="Embedding lookup (no MACs counted); output is (B,8,L) after transpose",
        ),
        LayerCompute(
            name="conv_1 (value)",
            output_shape=("B", str(out_ch), str(out_len)),
            macs=conv_macs,
            notes=f"MACs = {out_ch} * {out_len} * ({in_ch}*{window_size})",
        ),
        LayerCompute(
            name="conv_2 (gate)",
            output_shape=("B", str(out_ch), str(out_len)),
            macs=conv_macs,
            notes=f"MACs = {out_ch} * {out_len} * ({in_ch}*{window_size}); then sigmoid",
        ),
        LayerCompute(
            name="gating multiply",
            output_shape=("B", str(out_ch), str(out_len)),
            macs=0,
            notes=f"Elementwise multiply ops ~= {gate_mul} per sample (not MACs)",
        ),
        LayerCompute(
            name="adaptive max pool",
            output_shape=("B", str(out_ch)),
            macs=0,
            notes=f"Comparisons ~= {out_ch}*({max(out_len - 1, 0)}) per sample",
        ),
        LayerCompute(
            name="fc_1",
            output_shape=("B", "128"),
            macs=fc1_macs,
            notes="MACs = 128*128; then ReLU",
        ),
        LayerCompute(
            name="fc_2",
            output_shape=("B", "1"),
            macs=fc2_macs,
            notes="MACs = 128*1; then sigmoid",
        ),
    ]


def compute_stats(ckpt_path: str) -> Stats:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

    max_len = int(ckpt.get("max_len", 1048576))
    window_size = int(ckpt.get("window_size", 512))

    model = MalConv(input_length=max_len, window_size=window_size)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    params, params_trainable, weights_bytes_fp32 = _count_params(model)
    weights_bytes_int8 = params * 1  # rough lower bound for INT8 params (1 byte each)

    conv1 = _estimate_conv_macs(max_len=max_len, window_size=window_size)
    conv2 = _estimate_conv_macs(max_len=max_len, window_size=window_size)
    out_len = _conv_out_len(max_len=max_len, window_size=window_size)
    gate_mul_ops = int(128 * out_len)
    fc1 = 128 * 128
    fc2 = 128 * 1
    total = conv1 + conv2 + fc1 + fc2

    file_size = os.path.getsize(ckpt_path)

    return Stats(
        checkpoint=os.path.basename(ckpt_path),
        file_size_bytes=file_size,
        max_len=max_len,
        window_size=window_size,
        params=params,
        params_trainable=params_trainable,
        weights_bytes_fp32=weights_bytes_fp32,
        weights_bytes_int8=weights_bytes_int8,
        conv1_macs=conv1,
        conv2_macs=conv2,
        gate_mul_ops=gate_mul_ops,
        fc1_macs=fc1,
        fc2_macs=fc2,
        total_macs=total,
    )


def _print(stats: Stats) -> None:
    print(f"\n=== {stats.checkpoint} ===")
    print(f"Checkpoint file size: {_human_bytes(stats.file_size_bytes)}")
    print(f"Model config: max_len={stats.max_len} window_size={stats.window_size}")
    print(f"Parameters: {stats.params:,} (trainable {stats.params_trainable:,})")
    print(f"Weights size (FP32, params only): {_human_bytes(stats.weights_bytes_fp32)}")
    print(f"Weights size (INT8, params only, rough): {_human_bytes(stats.weights_bytes_int8)}")
    print("Estimated compute (MACs, forward, conv+fc layers):")
    print(f"  conv_1: {stats.conv1_macs:,}")
    print(f"  conv_2: {stats.conv2_macs:,}")
    print(f"  fc_1:   {stats.fc1_macs:,}")
    print(f"  fc_2:   {stats.fc2_macs:,}")
    print(f"  total:  {stats.total_macs:,}")
    print(f"Other ops (not counted as MACs): gating_mul~{stats.gate_mul_ops:,} elemwise multiplies/sample")


def _print_detail(stats: Stats) -> None:
    layers = _layer_compute(max_len=stats.max_len, window_size=stats.window_size)
    print("\nPer-layer compute breakdown (approx):")
    print(f"Assumptions: stride=kernel={stats.window_size}, embed_dim=8, conv_out=128")
    for l in layers:
        shape = "(" + ", ".join(l.output_shape) + ")"
        macs = f"{l.macs:,}" if l.macs else "-"
        print(f"- {l.name:16s} out={shape:18s} MACs={macs:>14s} | {l.notes}")


def _print_json(stats: Stats) -> None:
    payload = {
        "checkpoint": stats.checkpoint,
        "file_size_bytes": stats.file_size_bytes,
        "max_len": stats.max_len,
        "window_size": stats.window_size,
        "params": stats.params,
        "params_trainable": stats.params_trainable,
        "weights_bytes_fp32": stats.weights_bytes_fp32,
        "weights_bytes_int8_rough": stats.weights_bytes_int8,
        "compute": {
            "conv1_macs": stats.conv1_macs,
            "conv2_macs": stats.conv2_macs,
            "fc1_macs": stats.fc1_macs,
            "fc2_macs": stats.fc2_macs,
            "total_macs": stats.total_macs,
            "gate_mul_ops": stats.gate_mul_ops,
        },
        "layers": [
            {
                "name": l.name,
                "output_shape": list(l.output_shape),
                "macs": l.macs,
                "notes": l.notes,
            }
            for l in _layer_compute(max_len=stats.max_len, window_size=stats.window_size)
        ],
    }
    print(json.dumps(payload, indent=2))


def _print_breakdown(ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    max_len = int(ckpt.get("max_len", 1048576))
    window_size = int(ckpt.get("window_size", 512))
    model = MalConv(input_length=max_len, window_size=window_size)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    parts = _param_breakdown(model)
    print("Parameter breakdown (top-level modules):")
    for name, n in parts:
        print(f"  {name:8s} {n:,}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Print model size/complexity stats for a MalConv checkpoint")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pth")
    p.add_argument("--breakdown", action="store_true", help="Print per-layer parameter breakdown")
    p.add_argument("--detail", action="store_true", help="Print per-layer compute breakdown (shapes + MAC formulas)")
    p.add_argument("--json", action="store_true", help="Print a JSON report (includes per-layer compute notes)")
    args = p.parse_args(argv)

    stats = compute_stats(args.ckpt)
    if args.json:
        _print_json(stats)
        return 0
    _print(stats)
    if args.detail:
        _print_detail(stats)
    if args.breakdown:
        _print_breakdown(args.ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

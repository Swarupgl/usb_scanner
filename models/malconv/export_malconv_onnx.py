from __future__ import annotations

import argparse
import os
from typing import Optional

import torch

from .model import MalConv


def export_onnx(ckpt_path: str, out_path: str, opset: int = 18) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

    max_len = int(ckpt.get("max_len", 1048576))
    window_size = int(ckpt.get("window_size", 512))

    model = MalConv(input_length=max_len, window_size=window_size)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    dummy = torch.zeros((1, max_len), dtype=torch.long)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["x"],
        output_names=["y"],
        opset_version=opset,
        dynamo=False,
        external_data=False,
        do_constant_folding=True,
    )


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Export MalConv checkpoint (.pth) to ONNX")
    p.add_argument("--ckpt", required=True, help="Input checkpoint .pth")
    p.add_argument("--out", required=True, help="Output .onnx path")
    p.add_argument("--opset", type=int, default=18)
    args = p.parse_args(argv)

    export_onnx(args.ckpt, args.out, opset=args.opset)
    print(f"Exported: {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

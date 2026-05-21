from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch

PADDING_TOKEN = 256


def read_bytes(file_path: str, max_len: int = 1048576) -> Optional[bytes]:
    """Read up to max_len bytes from a file.

    Returns:
      - bytes (length <= max_len) on success
      - None on error
    """
    try:
        with open(file_path, "rb") as f:
            return f.read(max_len)
    except Exception as exc:
        print(f"Error processing {file_path}: {exc}")
        return None


def file_to_tensor(
    file_path: str,
    max_len: int = 1048576,
    *,
    noise_bytes: int = 0,
    noise_mode: str = "none",
    noise_prob: float = 0.0,
    noise_seed: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Convert a file into a (1, max_len) LongTensor of byte tokens.

    Tokenization:
      - Real byte values: 0..255
      - Padding token: 256
    """
    raw = read_bytes(file_path, max_len=max_len)
    if raw is None:
        return None

    # Optional training-time robustness augmentation: append bytes (zeros/random)
    # before padding, simulating common evasion / junk-padding patterns.
    if noise_bytes > 0 and noise_prob > 0.0 and len(raw) < max_len and noise_mode != "none":
        rng = random.Random(noise_seed)
        if rng.random() < noise_prob:
            n = min(int(noise_bytes), max_len - len(raw))
            if n > 0:
                if noise_mode == "zeros":
                    extra = b"\x00" * n
                elif noise_mode == "random":
                    extra = rng.randbytes(n)
                else:
                    raise ValueError(f"Unknown noise_mode: {noise_mode}")
                raw = raw + extra

    byte_array = np.frombuffer(raw, dtype=np.uint8)
    tensor = torch.from_numpy(byte_array.astype(np.int64, copy=False))

    if tensor.numel() < max_len:
        pad = torch.full((max_len - tensor.numel(),), PADDING_TOKEN, dtype=torch.long)
        tensor = torch.cat([tensor.long(), pad], dim=0)
    else:
        tensor = tensor[:max_len].long()

    return tensor.unsqueeze(0)


def is_probably_binary(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    lower = path.lower()
    return lower.endswith(".exe") or lower.endswith(".dll")

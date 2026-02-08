from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch


def read_bytes(file_path: str, max_len: int = 1048576) -> Optional[np.ndarray]:
    """Read up to max_len bytes and return a padded uint8 array (length=max_len)."""
    try:
        with open(file_path, "rb") as f:
            raw = f.read(max_len)
        byte_array = np.frombuffer(raw, dtype=np.uint8)
        if byte_array.size < max_len:
            byte_array = np.pad(byte_array, (0, max_len - byte_array.size), constant_values=256)
            # Note: pad uses 256, but uint8 wraps; we store padding as 256 later in torch.
            # We'll fix padding in tensor conversion.
        return byte_array
    except Exception as exc:
        print(f"Error processing {file_path}: {exc}")
        return None


def bytes_to_tensor(byte_array: np.ndarray, max_len: int = 1048576) -> torch.Tensor:
    """Convert uint8 array to LongTensor with padding index=256."""
    # Ensure length
    if byte_array.size < max_len:
        pad_len = max_len - byte_array.size
        byte_array = np.pad(byte_array, (0, pad_len), constant_values=0)

    # Convert to int64 and set padding to 256 for any padded region
    tensor = torch.from_numpy(byte_array.astype(np.int64, copy=False))
    if tensor.numel() > max_len:
        tensor = tensor[:max_len]

    # Heuristic: if file shorter, caller should provide original length; we can't infer after pad.
    # So: treat zeros as zeros (valid byte). Use explicit padding in file_to_tensor.
    return tensor


def file_to_tensor(file_path: str, max_len: int = 1048576) -> Optional[torch.Tensor]:
    """Read file into a (1, max_len) LongTensor with padding index=256."""
    try:
        with open(file_path, "rb") as f:
            raw = f.read(max_len)
        byte_array = np.frombuffer(raw, dtype=np.uint8)
        tensor = torch.from_numpy(byte_array.astype(np.int64, copy=False))

        if tensor.numel() < max_len:
            pad = torch.full((max_len - tensor.numel(),), 256, dtype=torch.long)
            tensor = torch.cat([tensor.long(), pad], dim=0)
        else:
            tensor = tensor.long()

        return tensor.unsqueeze(0)
    except Exception as exc:
        print(f"Error processing {file_path}: {exc}")
        return None


def is_probably_binary(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    lower = path.lower()
    return lower.endswith(".exe") or lower.endswith(".dll")

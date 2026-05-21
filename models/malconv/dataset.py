from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .preprocess import file_to_tensor, is_probably_binary


@dataclass(frozen=True)
class Sample:
    path: str
    label: int


def index_folder(folder: str, label: int, extensions: Sequence[str] = (".exe", ".dll")) -> List[Sample]:
    samples: List[Sample] = []
    folder = os.path.abspath(folder)
    for root, _, files in os.walk(folder):
        for name in files:
            lower = name.lower()
            if any(lower.endswith(ext) for ext in extensions):
                samples.append(Sample(path=os.path.join(root, name), label=label))
    return samples


class BinaryFoldersDataset(Dataset):
    def __init__(
        self,
        benign_dirs: Sequence[str],
        malicious_dirs: Sequence[str],
        max_len: int = 1048576,
        extensions: Sequence[str] = (".exe", ".dll", ".com"),
        limit_per_class: Optional[int] = None,
        *,
        noise_bytes: int = 0,
        noise_mode: str = "none",
        noise_prob: float = 0.0,
        noise_seed: int = 1337,
    ):
        self.max_len = max_len
        self.samples: List[Sample] = []

        self.noise_bytes = int(noise_bytes)
        self.noise_mode = str(noise_mode)
        self.noise_prob = float(noise_prob)
        self.noise_seed = int(noise_seed)

        for d in benign_dirs:
            benign = index_folder(d, label=0, extensions=extensions)
            if limit_per_class:
                benign = benign[:limit_per_class]
            self.samples.extend(benign)

        for d in malicious_dirs:
            malicious = index_folder(d, label=1, extensions=extensions)
            if limit_per_class:
                malicious = malicious[:limit_per_class]
            self.samples.extend(malicious)

        if not self.samples:
            raise ValueError("No samples found. Provide benign/malicious dirs with executable files.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[idx]
        x = file_to_tensor(
            sample.path,
            max_len=self.max_len,
            noise_bytes=self.noise_bytes,
            noise_mode=self.noise_mode,
            noise_prob=self.noise_prob,
            noise_seed=self.noise_seed + idx,
        )
        if x is None:
            # If a file can't be read, return a padded all-256 tensor and label.
            x = torch.full((1, self.max_len), 256, dtype=torch.long)
        y = torch.tensor([sample.label], dtype=torch.float32)
        return x.squeeze(0), y, sample.path


class BinaryFolderDataset(BinaryFoldersDataset):
    def __init__(
        self,
        benign_dir: Optional[str],
        malicious_dir: Optional[str],
        max_len: int = 1048576,
        extensions: Sequence[str] = (".exe", ".dll", ".com"),
        limit_per_class: Optional[int] = None,
        *,
        noise_bytes: int = 0,
        noise_mode: str = "none",
        noise_prob: float = 0.0,
        noise_seed: int = 1337,
    ):
        super().__init__(
            benign_dirs=[benign_dir] if benign_dir else [],
            malicious_dirs=[malicious_dir] if malicious_dir else [],
            max_len=max_len,
            extensions=extensions,
            limit_per_class=limit_per_class,
            noise_bytes=noise_bytes,
            noise_mode=noise_mode,
            noise_prob=noise_prob,
            noise_seed=noise_seed,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[idx]
        x = file_to_tensor(
            sample.path,
            max_len=self.max_len,
            noise_bytes=self.noise_bytes,
            noise_mode=self.noise_mode,
            noise_prob=self.noise_prob,
            noise_seed=self.noise_seed + idx,
        )
        if x is None:
            # If a file can't be read, return a padded all-256 tensor and label.
            x = torch.full((1, self.max_len), 256, dtype=torch.long)
        y = torch.tensor([sample.label], dtype=torch.float32)
        return x.squeeze(0), y, sample.path


def collate_batch(batch: Sequence[Tuple[torch.Tensor, torch.Tensor, str]]):
    xs, ys, paths = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    return x, y, list(paths)

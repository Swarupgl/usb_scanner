from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from models.malconv.dataset import BinaryFolderDataset, collate_batch
from models.malconv.model import MalConv


@dataclass(frozen=True)
class EvalResult:
    name: str
    accuracy: float
    tn: int
    fp: int
    fn: int
    tp: int


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _load_checkpoint(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return ckpt


def _build_model_from_ckpt(ckpt: dict) -> MalConv:
    max_len = int(ckpt.get("max_len", 1048576))
    window_size = int(ckpt.get("window_size", 512))
    model = MalConv(input_length=max_len, window_size=window_size)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model


@torch.no_grad()
def _eval_on_indices(
    model: MalConv,
    dataset: BinaryFolderDataset,
    indices: list[int],
    batch_size: int,
    device: torch.device,
) -> EvalResult:
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )

    model.eval()
    tn = fp = fn = tp = 0
    total = 0
    correct = 0

    for x, y, _paths in loader:
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        pred = (out > 0.5).float()

        correct += (pred == y).sum().item()
        total += y.numel()

        # y, pred are shape (B,1)
        yb = y.view(-1).to(torch.int64)
        pb = pred.view(-1).to(torch.int64)
        for yi, pi in zip(yb.tolist(), pb.tolist()):
            if yi == 0 and pi == 0:
                tn += 1
            elif yi == 0 and pi == 1:
                fp += 1
            elif yi == 1 and pi == 0:
                fn += 1
            else:
                tp += 1

    acc = _safe_div(correct, total)
    return EvalResult(name="", accuracy=acc, tn=tn, fp=fp, fn=fn, tp=tp)


def _print_metrics(res: EvalResult) -> None:
    tn, fp, fn, tp = res.tn, res.fp, res.fn, res.tp
    total = tn + fp + fn + tp
    acc = _safe_div(tn + tp, total)

    prec_pos = _safe_div(tp, tp + fp)
    rec_pos = _safe_div(tp, tp + fn)
    f1_pos = _safe_div(2 * prec_pos * rec_pos, prec_pos + rec_pos)

    prec_neg = _safe_div(tn, tn + fn)
    rec_neg = _safe_div(tn, tn + fp)
    f1_neg = _safe_div(2 * prec_neg * rec_neg, prec_neg + rec_neg)

    print(f"\n=== {res.name} ===")
    print(f"Accuracy: {acc:.3f} ({tn + tp}/{total})")
    print("Confusion matrix [[TN FP],[FN TP]]:")
    print(f"[[{tn} {fp}],[{fn} {tp}]]")
    print(f"FP (benign flagged): {fp}")
    print(f"FN (missed suspicious): {fn}")
    print(f"Malicious precision/recall/F1: {prec_pos:.3f} / {rec_pos:.3f} / {f1_pos:.3f}")
    print(f"Benign    precision/recall/F1: {prec_neg:.3f} / {rec_neg:.3f} / {f1_neg:.3f}")


def _get_split_indices(dataset: BinaryFolderDataset, seed: int, no_group_split: bool) -> tuple[list[int], list[int]]:
    """Returns (test_idx, all_idx).

    We only need test_idx for evaluation, but return all_idx for debugging.
    """

    if no_group_split:
        n = len(dataset)
        g = torch.Generator()
        g.manual_seed(seed)
        idx = torch.randperm(n, generator=g).tolist()
        val_split = int(0.9 * n)
        test_idx = idx[val_split:]
        return test_idx, idx

    # Import from train.py so we match training behavior exactly.
    from models.malconv.train import _group_split_indices  # type: ignore

    _train_idx, _val_idx, test_idx, _num_groups = _group_split_indices(dataset, seed=seed)
    all_idx = _train_idx + _val_idx + test_idx
    return test_idx, all_idx


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate two MalConv checkpoints on the same test split")
    p.add_argument("--ckpt-a", required=True, help="Path to checkpoint A")
    p.add_argument("--ckpt-b", required=True, help="Path to checkpoint B")
    p.add_argument("--benign-dir", default=os.path.join("datasets", "local", "benign"))
    p.add_argument("--malicious-dir", default=os.path.join("datasets", "local", "malicious"))
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--limit-per-class", type=int, default=200)
    p.add_argument(
        "--no-group-split",
        action="store_true",
        help="Use old random per-file split (can leak twins across splits)",
    )
    args = p.parse_args(argv)

    if not os.path.exists(args.ckpt_a):
        raise FileNotFoundError(args.ckpt_a)
    if not os.path.exists(args.ckpt_b):
        raise FileNotFoundError(args.ckpt_b)

    ckpt_a = _load_checkpoint(args.ckpt_a)
    ckpt_b = _load_checkpoint(args.ckpt_b)

    # Use a single dataset instance so sample ordering is identical.
    # We'll mutate dataset.max_len before each evaluation to match each checkpoint.
    dataset = BinaryFolderDataset(
        benign_dir=args.benign_dir,
        malicious_dir=args.malicious_dir,
        max_len=int(ckpt_a.get("max_len", 1048576)),
        limit_per_class=args.limit_per_class,
    )

    test_idx, _all_idx = _get_split_indices(dataset, seed=args.seed, no_group_split=args.no_group_split)
    print(
        f"Samples: total={len(dataset)} test={len(test_idx)} split={'random-file' if args.no_group_split else 'group-by-basename'} seed={args.seed}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_a = _build_model_from_ckpt(ckpt_a).to(device)
    dataset.max_len = int(ckpt_a.get("max_len", 1048576))
    res_a = _eval_on_indices(model_a, dataset, test_idx, batch_size=args.batch_size, device=device)
    res_a = EvalResult(
        name=os.path.basename(args.ckpt_a),
        accuracy=res_a.accuracy,
        tn=res_a.tn,
        fp=res_a.fp,
        fn=res_a.fn,
        tp=res_a.tp,
    )

    model_b = _build_model_from_ckpt(ckpt_b).to(device)
    dataset.max_len = int(ckpt_b.get("max_len", 1048576))
    res_b = _eval_on_indices(model_b, dataset, test_idx, batch_size=args.batch_size, device=device)
    res_b = EvalResult(
        name=os.path.basename(args.ckpt_b),
        accuracy=res_b.accuracy,
        tn=res_b.tn,
        fp=res_b.fp,
        fn=res_b.fn,
        tp=res_b.tp,
    )

    _print_metrics(res_a)
    _print_metrics(res_b)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

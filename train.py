from __future__ import annotations

import argparse
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import BinaryFolderDataset, collate_batch
from model import MalConv


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: MalConv,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    count = 0

    for x, y, _paths in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        count += x.size(0)

    return total_loss / max(count, 1)


@torch.no_grad()
def eval_epoch(model: MalConv, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    for x, y, _paths in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        outputs = model(x)
        preds = (outputs > 0.5).float()
        correct += (preds == y).sum().item()
        total += y.numel()

    return correct / max(total, 1)


def demo_random_training(
    model: MalConv,
    device: torch.device,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    max_len: int,
    lr: float,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    print("Starting DEMO random training (not meaningful for real detection)...")
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for _ in range(steps_per_epoch):
            x = torch.randint(0, 256, (batch_size, max_len), device=device, dtype=torch.long)
            y = torch.randint(0, 2, (batch_size, 1), device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {running / steps_per_epoch:.4f}")

try:
    from sklearn.metrics import confusion_matrix, classification_report

    _SKLEARN_AVAILABLE = True
except Exception:
    confusion_matrix = None  # type: ignore[assignment]
    classification_report = None  # type: ignore[assignment]
    _SKLEARN_AVAILABLE = False

@torch.no_grad()
def final_project_report(model: MalConv, loader: DataLoader, device: torch.device) -> None:
    if not _SKLEARN_AVAILABLE:
        print("\nFinal report skipped: scikit-learn is not installed.")
        print("Install it with: pip install scikit-learn")
        return

    model.eval()
    all_preds = []
    all_labels = []

    for x, y, _paths in loader:
        x = x.to(device)
        outputs = model(x)
        preds = (outputs > 0.5).float().cpu().numpy()
        all_preds.extend(preds.flatten())
        all_labels.extend(y.cpu().numpy().flatten())

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['Benign', 'Malicious'])

    print("\n" + "!"*40)
    print("      FINAL CYBERSECURITY REPORT      ")
    print("!"*40)
    print(f"\nCONFUSION MATRIX:\n{cm}")
    print(f"\nDETAILED STATS:\n{report}")
    
    if getattr(cm, "size", 0) == 4:
        tn, fp, fn, tp = cm.ravel()
        print(f"False Positives (Clean files blocked): {fp}")
        print(f"False Negatives (Viruses missed):     {fn}")
    print("!"*40)

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train MalConv on folders of binaries")
    parser.add_argument("--benign-dir", type=str, default=None, help="Folder of benign .exe/.dll")
    parser.add_argument("--malicious-dir", type=str, default=None, help="Folder of malicious/test samples")
    parser.add_argument("--max-len", type=int, default=1048576, help="Bytes per file (default 1MB)")
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0, help="Keep 0 on Windows initially")
    parser.add_argument("--limit-per-class", type=int, default=200, help="Cap files per class")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", type=str, default="malconv_model.pth")
    parser.add_argument("--demo-random", action="store_true", help="Train on random bytes/labels")
    args = parser.parse_args(argv)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MalConv(input_length=args.max_len, window_size=args.window_size).to(device)

    test_ds = None
    test_loader = None

    if args.demo_random or (not args.benign_dir and not args.malicious_dir):
        demo_random_training(
            model=model,
            device=device,
            epochs=args.epochs,
            steps_per_epoch=10,
            batch_size=args.batch_size,
            max_len=args.max_len,
            lr=args.lr,
        )
    else:
        if not args.benign_dir or not args.malicious_dir:
            print("Error: provide both --benign-dir and --malicious-dir (or use --demo-random).")
            return 2

        dataset = BinaryFolderDataset(
            benign_dir=args.benign_dir,
            malicious_dir=args.malicious_dir,
            max_len=args.max_len,
            limit_per_class=args.limit_per_class,
        )

        # Simple split
        n = len(dataset)
        idx = torch.randperm(n)
        train_split = int(0.8 * n)
        val_split = int(0.9 * n) # Marks the 90% point

        train_idx = idx[:train_split].tolist()
        val_idx = idx[train_split:val_split].tolist()
        test_idx = idx[val_split:].tolist()

        train_ds = torch.utils.data.Subset(dataset, train_idx)
        val_ds = torch.utils.data.Subset(dataset, val_idx)
        test_ds = torch.utils.data.Subset(dataset, test_idx)

        # Standard loaders for Train and Val
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
            collate_fn=collate_batch,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
            collate_fn=collate_batch,
        )
        # Loader for the Final Exam (Test Set)
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
            collate_fn=collate_batch,
        )

        print(f"Dataset Split: Train={len(train_ds)} | Val={len(val_ds)} | Test={len(test_ds)}")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.BCELoss()

        print(f"Training samples: {len(train_ds)} | Val samples: {len(val_ds)}")
        for epoch in range(args.epochs):
            loss = train_epoch(model, train_loader, optimizer, criterion, device)
            acc = eval_epoch(model, val_loader, device)
            print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {loss:.4f} - ValAcc: {acc:.3f}")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "max_len": args.max_len,
            "window_size": args.window_size,
        },
        args.out,
    )
    print(f"Saved: {os.path.abspath(args.out)}")

    model.eval()
    print("\nTraining complete.")

    if test_ds is None or test_loader is None:
        print("Final report skipped: no test split was created (demo-random mode).")
        return 0

    print("Generating Final Report...", flush=True)
    if len(test_ds) > 0:
        final_project_report(model, test_loader, device)
    else:
        print("Final report skipped: test dataset is empty. Check your data split logic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

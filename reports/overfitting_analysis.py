"""
overfitting_analysis.py
=======================
Re-trains MalConv from a RANDOM init (same architecture + hyperparams as the
saved checkpoint) and logs train loss, val loss, train accuracy, val accuracy
at every epoch.  After training it evaluates the SAVED checkpoint on all three
splits (train / val / test) to check whether the final saved model generalises.

Produces:
  outputs/training_history.json          – epoch-level log
  summary/usb_reports/overfitting_curves.png  – 4-panel curve figure

Run from project root:
    .\.venv\Scripts\python -m summary.usb_reports.overfitting_analysis
"""

from __future__ import annotations
import os, sys, json, random, copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, _ROOT)

from models.malconv.model import MalConv
from models.malconv.dataset import BinaryFolderDataset, collate_batch
from models.malconv.train import (
    set_seed, train_epoch, eval_epoch,
    _group_split_indices, final_project_report
)

# ── config ────────────────────────────────────────────────────────────────────
CKPT        = "outputs/models/group_split_model.pth"
BENIGN_DIR  = os.path.join("datasets", "local", "benign")
MAL_DIR     = os.path.join("datasets", "local", "malicious")
SEED        = 1337
EPOCHS      = 30        # enough to see convergence / over-fitting plateau
BATCH_SIZE  = 4
LR          = 1e-3
OUT_JSON    = "outputs/training_history.json"
OUT_PNG     = "summary/usb_reports/overfitting_curves.png"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_loss_acc(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        out  = model(x)
        total_loss += criterion(out, y).item() * x.size(0)
        correct    += ((out > 0.5).float() == y).sum().item()
        total      += y.numel()
    return total_loss / max(total, 1), correct / max(total, 1)


def make_loaders(dataset, train_idx, val_idx, test_idx, batch_size, device):
    kw = dict(collate_fn=collate_batch, num_workers=0,
              pin_memory=(device.type == "cuda"))
    tr = DataLoader(Subset(dataset, train_idx), batch_size=batch_size,
                    shuffle=True,  **kw)
    va = DataLoader(Subset(dataset, val_idx),   batch_size=batch_size,
                    shuffle=False, **kw)
    te = DataLoader(Subset(dataset, test_idx),  batch_size=batch_size,
                    shuffle=False, **kw)
    return tr, va, te


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    SEP = "=" * 68
    print(SEP)
    print("  OVERFITTING ANALYSIS – MalConv Gated CNN")
    print(SEP)

    set_seed(SEED)
    device = torch.device("cpu")  # same as Pi target

    # ── load checkpoint to read arch params ──────────────────────────────────
    if not os.path.exists(CKPT):
        print(f"ERROR: {CKPT} not found"); return 1
    ckpt       = torch.load(CKPT, map_location="cpu")
    max_len    = int(ckpt.get("max_len",     262144))
    win_size   = int(ckpt.get("window_size", 512))
    print(f"Architecture: max_len={max_len} ({max_len//1024} KB), window={win_size}")

    # ── build dataset & splits (SAME seed as training) ───────────────────────
    dataset = BinaryFolderDataset(
        benign_dir=BENIGN_DIR, malicious_dir=MAL_DIR,
        max_len=max_len, limit_per_class=200,
    )
    train_idx, val_idx, test_idx, num_groups = _group_split_indices(
        dataset, seed=SEED
    )
    train_loader, val_loader, test_loader = make_loaders(
        dataset, train_idx, val_idx, test_idx, BATCH_SIZE, device
    )

    print(f"\nDataset : {len(dataset)} samples "
          f"({len(train_idx)} train | {len(val_idx)} val | {len(test_idx)} test)"
          f"  [{num_groups} file-groups, group-split]")
    print(f"Benign  in train : {sum(1 for i in train_idx if dataset.samples[i].label==0)}")
    print(f"Malicious in train: {sum(1 for i in train_idx if dataset.samples[i].label==1)}")
    print(f"Benign  in val   : {sum(1 for i in val_idx   if dataset.samples[i].label==0)}")
    print(f"Malicious in val : {sum(1 for i in val_idx   if dataset.samples[i].label==1)}")
    print(f"Benign  in test  : {sum(1 for i in test_idx  if dataset.samples[i].label==0)}")
    print(f"Malicious in test: {sum(1 for i in test_idx  if dataset.samples[i].label==1)}")

    criterion = nn.BCELoss()

    # =========================================================================
    # PART A – Train from RANDOM INIT and record curves every epoch
    # =========================================================================
    print(f"\n{'─'*68}")
    print(f"  PART A: Training from random init for {EPOCHS} epochs")
    print(f"{'─'*68}")

    fresh_model = MalConv(input_length=max_len, window_size=win_size).to(device)
    optimizer   = torch.optim.Adam(fresh_model.parameters(), lr=LR)

    history = {
        "epoch":      [],
        "train_loss": [], "val_loss":  [], "test_loss": [],
        "train_acc":  [], "val_acc":   [], "test_acc":  [],
    }

    # measure random-init baseline BEFORE any training
    tr_l0, tr_a0 = eval_loss_acc(fresh_model, train_loader, criterion, device)
    va_l0, va_a0 = eval_loss_acc(fresh_model, val_loader,   criterion, device)
    te_l0, te_a0 = eval_loss_acc(fresh_model, test_loader,  criterion, device)
    history["epoch"].append(0)
    history["train_loss"].append(tr_l0); history["val_loss"].append(va_l0);  history["test_loss"].append(te_l0)
    history["train_acc"].append(tr_a0);  history["val_acc"].append(va_a0);   history["test_acc"].append(te_a0)
    print(f"Epoch  0/{EPOCHS} | "
          f"TrainLoss={tr_l0:.4f} ValLoss={va_l0:.4f} TestLoss={te_l0:.4f} | "
          f"TrainAcc={tr_a0:.3f} ValAcc={va_a0:.3f} TestAcc={te_a0:.3f}")

    for epoch in range(1, EPOCHS + 1):
        train_epoch(fresh_model, train_loader, optimizer, criterion, device)
        tr_l, tr_a = eval_loss_acc(fresh_model, train_loader, criterion, device)
        va_l, va_a = eval_loss_acc(fresh_model, val_loader,   criterion, device)
        te_l, te_a = eval_loss_acc(fresh_model, test_loader,  criterion, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_l); history["val_loss"].append(va_l);  history["test_loss"].append(te_l)
        history["train_acc"].append(tr_a);  history["val_acc"].append(va_a);   history["test_acc"].append(te_a)

        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"TrainLoss={tr_l:.4f} ValLoss={va_l:.4f} TestLoss={te_l:.4f} | "
              f"TrainAcc={tr_a:.3f} ValAcc={va_a:.3f} TestAcc={te_a:.3f}")

    # =========================================================================
    # PART B – Evaluate the SAVED checkpoint on all 3 splits
    # =========================================================================
    print(f"\n{'─'*68}")
    print(f"  PART B: Evaluating SAVED checkpoint on all 3 splits")
    print(f"{'─'*68}")

    saved_model = MalConv(input_length=max_len, window_size=win_size).to(device)
    saved_model.load_state_dict(ckpt["state_dict"])
    saved_model.eval()

    sv_tr_l, sv_tr_a = eval_loss_acc(saved_model, train_loader, criterion, device)
    sv_va_l, sv_va_a = eval_loss_acc(saved_model, val_loader,   criterion, device)
    sv_te_l, sv_te_a = eval_loss_acc(saved_model, test_loader,  criterion, device)

    print(f"\n  {'Split':<10} {'Loss':>8} {'Accuracy':>10}")
    print(f"  {'─'*10} {'─'*8} {'─'*10}")
    print(f"  {'Train':<10} {sv_tr_l:>8.4f} {sv_tr_a*100:>9.1f}%")
    print(f"  {'Val':<10} {sv_va_l:>8.4f} {sv_va_a*100:>9.1f}%")
    print(f"  {'Test':<10} {sv_te_l:>8.4f} {sv_te_a*100:>9.1f}%")

    gap_loss = sv_va_l - sv_tr_l
    gap_acc  = sv_tr_a - sv_va_a
    print(f"\n  Generalisation gap (Val - Train loss) : {gap_loss:+.4f}")
    print(f"  Generalisation gap (Train - Val acc)  : {gap_acc*100:+.2f}%")

    if abs(gap_acc) < 0.05 and abs(gap_loss) < 0.15:
        verdict = "LOW OVERFITTING – model generalises well across splits."
    elif gap_acc > 0.2 or gap_loss > 0.4:
        verdict = "HIGH OVERFITTING – train >> val performance."
    else:
        verdict = "MODERATE – small gap, acceptable for this dataset size."
    print(f"\n  Verdict: {verdict}")

    # =========================================================================
    # PART C – Save JSON
    # =========================================================================
    os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
    saved_summary = {
        "saved_checkpoint_eval": {
            "train_loss": sv_tr_l, "train_acc": sv_tr_a,
            "val_loss":   sv_va_l, "val_acc":   sv_va_a,
            "test_loss":  sv_te_l, "test_acc":  sv_te_a,
            "generalisation_gap_loss": gap_loss,
            "generalisation_gap_acc":  gap_acc,
            "verdict": verdict,
        },
        "fresh_training_history": history,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(saved_summary, f, indent=2)
    print(f"\n  Saved JSON → {os.path.abspath(OUT_JSON)}")

    # =========================================================================
    # PART D – Plot curves
    # =========================================================================
    print(f"\n  Generating overfitting curves plot …")
    ep = history["epoch"]

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#0f172a")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    COLORS = {
        "train": "#38bdf8",   # sky blue
        "val":   "#fb923c",   # orange
        "test":  "#4ade80",   # green
        "saved": "#f472b6",   # pink marker
    }

    def styled_ax(ax, title):
        ax.set_facecolor("#1e293b")
        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        ax.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")
        return ax

    # ── Panel 1: Loss curves ──────────────────────────────────────────────────
    ax1 = styled_ax(fig.add_subplot(gs[0, 0]), "Training & Validation Loss (Fresh Re-train)")
    ax1.plot(ep, history["train_loss"], color=COLORS["train"], lw=2, label="Train Loss",  marker="o", ms=4)
    ax1.plot(ep, history["val_loss"],   color=COLORS["val"],   lw=2, label="Val Loss",    marker="s", ms=4)
    ax1.plot(ep, history["test_loss"],  color=COLORS["test"],  lw=2, label="Test Loss",   marker="^", ms=4, linestyle="--")
    # mark saved-checkpoint metrics as horizontal reference lines
    ax1.axhline(sv_tr_l, color=COLORS["train"], lw=1, linestyle=":", alpha=0.7)
    ax1.axhline(sv_va_l, color=COLORS["val"],   lw=1, linestyle=":", alpha=0.7)
    ax1.axhline(sv_te_l, color=COLORS["test"],  lw=1, linestyle=":", alpha=0.7)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("BCE Loss", fontsize=11)
    ax1.legend(fontsize=10, facecolor="#1e293b", labelcolor="white")

    # ── Panel 2: Accuracy curves ──────────────────────────────────────────────
    ax2 = styled_ax(fig.add_subplot(gs[0, 1]), "Training & Validation Accuracy (Fresh Re-train)")
    ax2.plot(ep, [a*100 for a in history["train_acc"]], color=COLORS["train"], lw=2, label="Train Acc",  marker="o", ms=4)
    ax2.plot(ep, [a*100 for a in history["val_acc"]],   color=COLORS["val"],   lw=2, label="Val Acc",    marker="s", ms=4)
    ax2.plot(ep, [a*100 for a in history["test_acc"]],  color=COLORS["test"],  lw=2, label="Test Acc",   marker="^", ms=4, linestyle="--")
    ax2.axhline(sv_tr_a*100, color=COLORS["train"], lw=1, linestyle=":", alpha=0.7)
    ax2.axhline(sv_va_a*100, color=COLORS["val"],   lw=1, linestyle=":", alpha=0.7)
    ax2.axhline(sv_te_a*100, color=COLORS["test"],  lw=1, linestyle=":", alpha=0.7)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=10, facecolor="#1e293b", labelcolor="white")

    # ── Panel 3: Generalisation gap (loss) ───────────────────────────────────
    ax3 = styled_ax(fig.add_subplot(gs[1, 0]), "Generalisation Gap – Loss (Val − Train)")
    gap_loss_curve = [v - t for v, t in zip(history["val_loss"], history["train_loss"])]
    ax3.bar(ep, gap_loss_curve,
            color=["#ef4444" if g > 0.15 else "#4ade80" for g in gap_loss_curve],
            edgecolor="#1e293b", width=0.6)
    ax3.axhline(0,    color="white",    lw=1, linestyle="--")
    ax3.axhline(0.15, color="#ef4444",  lw=1, linestyle=":", label="Overfitting threshold (0.15)")
    ax3.set_xlabel("Epoch", fontsize=11)
    ax3.set_ylabel("Val Loss − Train Loss", fontsize=11)
    ax3.legend(fontsize=10, facecolor="#1e293b", labelcolor="white")
    # annotate final saved-ckpt gap
    ax3.annotate(
        f"Saved ckpt gap\n{gap_loss:+.4f}",
        xy=(max(ep)*0.7, gap_loss),
        color=COLORS["saved"], fontsize=10, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLORS["saved"]),
        xytext=(max(ep)*0.5, gap_loss + 0.05),
    )

    # ── Panel 4: Saved checkpoint split comparison ────────────────────────────
    ax4 = styled_ax(fig.add_subplot(gs[1, 1]), "Saved Checkpoint – All Splits Comparison")
    splits    = ["Train", "Validation", "Test"]
    acc_vals  = [sv_tr_a*100, sv_va_a*100, sv_te_a*100]
    loss_vals = [sv_tr_l,     sv_va_l,     sv_te_l]
    bar_colors = [COLORS["train"], COLORS["val"], COLORS["test"]]

    x_pos = np.arange(len(splits))
    bars = ax4.bar(x_pos, acc_vals, color=bar_colors, edgecolor="#0f172a",
                   width=0.5, label="Accuracy (%)", alpha=0.9)
    for bar, acc, loss in zip(bars, acc_vals, loss_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{acc:.1f}%\n(loss={loss:.4f})",
                 ha="center", va="bottom", color="white", fontsize=11, fontweight="bold")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(splits, fontsize=12)
    ax4.set_ylabel("Accuracy (%)", fontsize=11)
    ax4.set_ylim(0, 115)
    ax4.legend(fontsize=10, facecolor="#1e293b", labelcolor="white")

    # ── Overall title ─────────────────────────────────────────────────────────
    fig.suptitle(
        "MalConv Overfitting Analysis  ·  USB Malware Scanner  ·  EAI Project",
        color="white", fontsize=15, fontweight="bold", y=0.98
    )

    # ── Verdict box ───────────────────────────────────────────────────────────
    verdict_color = "#4ade80" if "LOW" in verdict else ("#ef4444" if "HIGH" in verdict else "#fb923c")
    fig.text(0.5, 0.01,
             f"Verdict: {verdict}",
             ha="center", color=verdict_color, fontsize=12, fontweight="bold",
             bbox=dict(facecolor="#1e293b", edgecolor=verdict_color, pad=6, boxstyle="round"))

    os.makedirs(os.path.dirname(OUT_PNG) or ".", exist_ok=True)
    plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved plot  → {os.path.abspath(OUT_PNG)}")

    print(f"\n{SEP}")
    print("  ANALYSIS COMPLETE")
    print(SEP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

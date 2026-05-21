"""
verify_all_methods.py
=====================
Deep verification script that re-applies every optimization method from
scratch and, for each variant:
  - Shows PER-SAMPLE prediction vs ground truth
  - Shows confusion matrix (TP / TN / FP / FN)
  - Shows final accuracy
  - Flags any mistake in the claim of 100% accuracy

Run from the project root:
    .\.venv\Scripts\python -m summary.usb_reports.verify_all_methods
"""
import os, sys, copy, time
import torch
import torch.nn as nn
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, _ROOT)

from models.malconv.model import MalConv
from models.malconv.dataset import BinaryFolderDataset, collate_batch
from models.malconv.preprocess import file_to_tensor
from summary.usb_reports.compare_checkpoints import _get_split_indices

try:
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    ONNX_OK = True
except ImportError:
    ONNX_OK = False
    print("[WARN] onnxruntime not found – ONNX quantisation will be skipped.")

CKPT = "outputs/models/group_split_model.pth"
BENIGN_DIR  = os.path.join("datasets", "local", "benign")
MALICIOUS_DIR = os.path.join("datasets", "local", "malicious")
SEED = 1337
THRESHOLD = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
SEP = "=" * 68

def banner(title):
    pad = max(0, 68 - len(title) - 4)
    print(f"\n{SEP}")
    print(f"  {title}  " + "─" * (pad // 2))
    print(SEP)

def load_baseline(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    max_len     = int(ckpt.get("max_len",     262144))
    window_size = int(ckpt.get("window_size", 512))
    model = MalConv(input_length=max_len, window_size=window_size)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, max_len, window_size

def get_test_samples(dataset, test_idx):
    """Return list of (path, label) for test split."""
    samples = []
    for i in test_idx:
        s = dataset.samples[i]
        samples.append((s.path, int(s.label)))
    return samples

@torch.no_grad()
def eval_pytorch(model, samples, max_len, label="model"):
    """
    Evaluate a PyTorch model sample by sample.
    Returns accuracy and prints per-sample detail.
    """
    correct = 0
    tp = tn = fp = fn = 0
    print(f"\n  {'File':<42} {'True':>6} {'Pred':>6} {'Score':>8} {'OK?':>5}")
    print(f"  {'─'*42} {'─'*6} {'─'*6} {'─'*8} {'─'*5}")

    for path, true_label in samples:
        x = file_to_tensor(path, max_len=max_len)
        if x is None:
            print(f"  [SKIP – tensor is None] {os.path.basename(path)}")
            continue
        score = model(x).item()
        pred  = int(score > THRESHOLD)
        ok    = (pred == true_label)
        correct += int(ok)

        fname = os.path.basename(path)[:42]
        status = "✓" if ok else "✗ WRONG"
        print(f"  {fname:<42} {true_label:>6} {pred:>6} {score:>8.4f} {status:>5}")

        if true_label == 1 and pred == 1: tp += 1
        elif true_label == 0 and pred == 0: tn += 1
        elif true_label == 0 and pred == 1: fp += 1
        else: fn += 1

    total = len(samples)
    acc = correct / total if total else 0.0
    print(f"\n  Confusion Matrix: TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  Accuracy : {correct}/{total} = {acc*100:.1f}%")
    if acc < 1.0:
        print(f"  *** ACCURACY IS NOT 100% — {fn} miss(es), {fp} false alarm(s) ***")
    return acc, tp, tn, fp, fn

@torch.no_grad()
def eval_onnx(session, samples, max_len, input_name):
    """Evaluate an ONNX session sample by sample."""
    correct = 0
    tp = tn = fp = fn = 0
    print(f"\n  {'File':<42} {'True':>6} {'Pred':>6} {'Score':>8} {'OK?':>5}")
    print(f"  {'─'*42} {'─'*6} {'─'*6} {'─'*8} {'─'*5}")

    for path, true_label in samples:
        x = file_to_tensor(path, max_len=max_len)
        if x is None:
            print(f"  [SKIP – tensor is None] {os.path.basename(path)}")
            continue
        inp = {input_name: x.numpy().astype(np.int64)}
        score = float(session.run(None, inp)[0][0][0])
        pred  = int(score > THRESHOLD)
        ok    = (pred == true_label)
        correct += int(ok)

        fname = os.path.basename(path)[:42]
        status = "✓" if ok else "✗ WRONG"
        print(f"  {fname:<42} {true_label:>6} {pred:>6} {score:>8.4f} {status:>5}")

        if true_label == 1 and pred == 1: tp += 1
        elif true_label == 0 and pred == 0: tn += 1
        elif true_label == 0 and pred == 1: fp += 1
        else: fn += 1

    total = len(samples)
    acc = correct / total if total else 0.0
    print(f"\n  Confusion Matrix: TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  Accuracy : {correct}/{total} = {acc*100:.1f}%")
    if acc < 1.0:
        print(f"  *** ACCURACY IS NOT 100% — {fn} miss(es), {fp} false alarm(s) ***")
    return acc, tp, tn, fp, fn

def apply_pruning(model, amount):
    import torch.nn.utils.prune as prune
    m = copy.deepcopy(model)
    for layer in [m.conv_1, m.conv_2, m.fc_1]:
        prune.l1_unstructured(layer, name="weight", amount=amount)
        prune.remove(layer, "weight")
    return m

def apply_weight_sharing(model, k):
    from sklearn.cluster import KMeans
    m = copy.deepcopy(model)
    for _, module in m.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            w = module.weight.data.cpu().numpy()
            shape = w.shape
            flat  = w.flatten().reshape(-1, 1)
            km    = KMeans(n_clusters=k, random_state=42, max_iter=20, n_init=3)
            km.fit(flat)
            new_w = km.cluster_centers_[km.labels_].reshape(shape)
            module.weight.data = torch.from_numpy(new_w.astype(np.float32))
    return m

# ─────────────────────────────────────────────────────────────────────────────
# Main verification
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(SEP)
    print("  DEEP VERIFICATION – ALL OPTIMIZATION METHODS")
    print("  Re-applying every technique fresh from the baseline checkpoint")
    print(SEP)

    if not os.path.exists(CKPT):
        print(f"ERROR: checkpoint not found: {CKPT}")
        return 1

    # ── Load baseline ────────────────────────────────────────────────────────
    baseline, max_len, window_size = load_baseline(CKPT)
    print(f"\nCheckpoint  : {CKPT}")
    print(f"max_len     : {max_len} bytes ({max_len//1024} KB)")
    print(f"window_size : {window_size}")

    # ── Build dataset & test split ───────────────────────────────────────────
    dataset = BinaryFolderDataset(
        benign_dir    = BENIGN_DIR,
        malicious_dir = MALICIOUS_DIR,
        max_len       = max_len,
        limit_per_class = 200,
    )
    test_idx, _ = _get_split_indices(dataset, seed=SEED, no_group_split=False)
    samples = get_test_samples(dataset, test_idx)

    print(f"\nTotal dataset samples : {len(dataset)}")
    print(f"Test split size       : {len(samples)}")
    print(f"Benign  in test       : {sum(1 for _,l in samples if l==0)}")
    print(f"Malicious in test     : {sum(1 for _,l in samples if l==1)}")

    report = {}   # will store (acc, fp, fn) per method

    # ────────────────────────────────────────────────────────────────────────
    # 1. Baseline FP32
    # ────────────────────────────────────────────────────────────────────────
    banner("1. Baseline FP32 Model")
    acc, tp, tn, fp, fn = eval_pytorch(baseline, samples, max_len)
    report["baseline"] = dict(acc=acc, fp=fp, fn=fn, note="No compression")

    # ────────────────────────────────────────────────────────────────────────
    # 2. PyTorch Dynamic Quantization (INT8 Linear layers only)
    # ────────────────────────────────────────────────────────────────────────
    banner("2. PyTorch Dynamic Quantization (INT8 Linear)")
    q_model = torch.quantization.quantize_dynamic(
        copy.deepcopy(baseline), {nn.Linear}, dtype=torch.qint8
    )
    q_model.eval()
    acc, tp, tn, fp, fn = eval_pytorch(q_model, samples, max_len)
    report["pytorch_int8"] = dict(acc=acc, fp=fp, fn=fn, note="Linear layers INT8")

    # ────────────────────────────────────────────────────────────────────────
    # 3. ONNX Dynamic Quantization (INT8 full graph)
    # ────────────────────────────────────────────────────────────────────────
    banner("3. ONNX Dynamic Quantization (INT8 full graph)")
    if ONNX_OK:
        onnx_fp32_path  = "outputs/models/verify_baseline.onnx"
        onnx_quant_path = "outputs/models/verify_quant.onnx"

        # Always re-export so we're sure it's the correct checkpoint.
        # Must use dynamo=False — the new PyTorch 2.11 exporter does not yet
        # support adaptive_max_pool2d (1-D) via the dynamo path.
        dummy = torch.zeros((1, max_len), dtype=torch.long)
        with torch.no_grad():
            torch.onnx.export(
                baseline, dummy, onnx_fp32_path,
                input_names=["x"], output_names=["y"],
                opset_version=18,
                dynamo=False,
                do_constant_folding=True,
                external_data=False,
            )
        print(f"  Exported ONNX FP32 → {onnx_fp32_path}")

        quantize_dynamic(
            model_input  = onnx_fp32_path,
            model_output = onnx_quant_path,
            weight_type  = QuantType.QInt8
        )
        print(f"  Quantised ONNX INT8 → {onnx_quant_path}")

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        session    = ort.InferenceSession(
            onnx_quant_path, sess_opts, providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        acc, tp, tn, fp, fn = eval_onnx(session, samples, max_len, input_name)
        report["onnx_int8"] = dict(acc=acc, fp=fp, fn=fn, note="Full graph INT8 via ONNX")
    else:
        print("  SKIPPED – onnxruntime not installed.")
        report["onnx_int8"] = dict(acc=None, fp=None, fn=None, note="SKIPPED")

    # ────────────────────────────────────────────────────────────────────────
    # 4. Weight Pruning 50 %
    # ────────────────────────────────────────────────────────────────────────
    banner("4. Weight Pruning – 50% Sparsity (L1 Unstructured)")
    p50 = apply_pruning(baseline, amount=0.50)
    p50.eval()
    # quick sanity: count zeros in conv_1 weight
    nz_ratio = (p50.conv_1.weight == 0).float().mean().item()
    print(f"  conv_1 zero-weight fraction after pruning : {nz_ratio*100:.1f}%")
    acc, tp, tn, fp, fn = eval_pytorch(p50, samples, max_len)
    report["prune_50"] = dict(acc=acc, fp=fp, fn=fn, note="50% L1 sparsity")

    # ────────────────────────────────────────────────────────────────────────
    # 5. Weight Pruning 75 %
    # ────────────────────────────────────────────────────────────────────────
    banner("5. Weight Pruning – 75% Sparsity (L1 Unstructured)")
    p75 = apply_pruning(baseline, amount=0.75)
    p75.eval()
    nz_ratio = (p75.conv_1.weight == 0).float().mean().item()
    print(f"  conv_1 zero-weight fraction after pruning : {nz_ratio*100:.1f}%")
    acc, tp, tn, fp, fn = eval_pytorch(p75, samples, max_len)
    report["prune_75"] = dict(acc=acc, fp=fp, fn=fn, note="75% L1 sparsity")

    # ────────────────────────────────────────────────────────────────────────
    # 6. Weight Sharing k=16
    # ────────────────────────────────────────────────────────────────────────
    banner("6. K-Means Weight Sharing – k=16 centroids (4-bit)")
    s16 = apply_weight_sharing(baseline, k=16)
    s16.eval()
    unique_vals = len(torch.unique(s16.conv_1.weight).tolist())
    print(f"  Unique weight values in conv_1 after sharing : {unique_vals}  (expect ≤16)")
    acc, tp, tn, fp, fn = eval_pytorch(s16, samples, max_len)
    report["share_16"] = dict(acc=acc, fp=fp, fn=fn, note="K-Means k=16")

    # ────────────────────────────────────────────────────────────────────────
    # 7. Weight Sharing k=32
    # ────────────────────────────────────────────────────────────────────────
    banner("7. K-Means Weight Sharing – k=32 centroids (5-bit)")
    s32 = apply_weight_sharing(baseline, k=32)
    s32.eval()
    unique_vals = len(torch.unique(s32.conv_1.weight).tolist())
    print(f"  Unique weight values in conv_1 after sharing : {unique_vals}  (expect ≤32)")
    acc, tp, tn, fp, fn = eval_pytorch(s32, samples, max_len)
    report["share_32"] = dict(acc=acc, fp=fp, fn=fn, note="K-Means k=32")

    # ────────────────────────────────────────────────────────────────────────
    # 8-10. Sequence downscaling 128 KB / 64 KB / 32 KB
    # ────────────────────────────────────────────────────────────────────────
    for prefix_kb, prefix_bytes in [(128, 131072), (64, 65536), (32, 32768)]:
        banner(f"{8 + [128,64,32].index(prefix_kb)}. Sequence Downscale – {prefix_kb} KB prefix")
        # Need dataset at this max_len so padding is correct
        ds_small = BinaryFolderDataset(
            benign_dir=BENIGN_DIR, malicious_dir=MALICIOUS_DIR,
            max_len=prefix_bytes, limit_per_class=200,
        )
        small_samples = get_test_samples(ds_small, test_idx)
        acc, tp, tn, fp, fn = eval_pytorch(baseline, small_samples, prefix_bytes)
        report[f"downscale_{prefix_kb}"] = dict(
            acc=acc, fp=fp, fn=fn, note=f"{prefix_kb} KB prefix"
        )

    # ────────────────────────────────────────────────────────────────────────
    # Final summary table
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n\n{SEP}")
    print("  VERIFICATION SUMMARY")
    print(SEP)
    print(f"  {'Method':<38} {'Accuracy':>9} {'FP':>5} {'FN':>5}  {'Claim OK?':>10}")
    print(f"  {'─'*38} {'─'*9} {'─'*5} {'─'*5}  {'─'*10}")

    all_ok = True
    for key, v in report.items():
        if v["acc"] is None:
            print(f"  {key:<38} {'SKIPPED':>9}")
            continue
        acc_str = f"{v['acc']*100:.1f}%"
        ok      = (v["acc"] == 1.0)
        flag    = "✓ CORRECT" if ok else "✗ WRONG – needs update in report"
        if not ok:
            all_ok = False
        print(f"  {key:<38} {acc_str:>9} {v['fp']:>5} {v['fn']:>5}  {flag:>10}")

    print(f"\n{SEP}")
    if all_ok:
        print("  RESULT: All methods verified – 100% accuracy claim is CORRECT.")
    else:
        print("  RESULT: One or more methods did NOT achieve 100% accuracy.")
        print("          Review the per-sample output above and correct the report.")
    print(SEP + "\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

import os
import sys
import time
import copy
import json
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Add the project root to sys.path so we can import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models.malconv.model import MalConv
from models.malconv.dataset import BinaryFolderDataset, collate_batch
from models.malconv.preprocess import file_to_tensor
from summary.usb_reports.compare_checkpoints import _get_split_indices

# For ONNX dynamic quantization
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

def get_file_size(path):
    if os.path.exists(path):
        return os.path.getsize(path)
    return 0

def estimate_macs(max_len, window_size, sparse_ratio=0.0):
    # conv1_macs = out_ch * out_len * in_ch * window_size
    # Here: out_ch = 128, in_ch = 8, window_size = 512
    # out_len = ((max_len - window_size) // window_size) + 1
    if max_len < window_size:
        out_len = 0
    else:
        out_len = ((max_len - window_size) // window_size) + 1
    
    conv1 = 128 * out_len * 8 * window_size
    conv2 = 128 * out_len * 8 * window_size
    fc1 = 128 * 128
    fc2 = 128 * 1
    
    total = conv1 + conv2 + fc1 + fc2
    # Pruned layers reduce actual multiply operations
    non_zero_macs = int((conv1 + conv2 + fc1) * (1.0 - sparse_ratio) + fc2)
    return total, non_zero_macs

@torch.no_grad()
def evaluate_pytorch_model(model, dataset, test_idx, device, max_len):
    model.eval()
    correct = 0
    total = 0
    
    subset = torch.utils.data.Subset(dataset, test_idx)
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_batch
    )
    
    # Latency measurement
    times_e2e = []
    times_model = []
    
    # Pre-warmup
    for i, (x, y, paths) in enumerate(loader):
        if i >= 2: break
        x = x.to(device)
        _ = model(x)
        
    for x, y, paths in loader:
        path = paths[0]
        # End-to-end timing
        t0 = time.perf_counter()
        xt = file_to_tensor(path, max_len=max_len)
        if xt is None:
            continue
        xt = xt.to(device)
        out = model(xt)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_e2e.append((t1 - t0) * 1000.0)
        
        # Model-only timing
        t2 = time.perf_counter()
        _ = model(xt)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        times_model.append((t3 - t2) * 1000.0)
        
        pred = (out.item() > 0.5)
        label = (y.item() > 0.5)
        if pred == label:
            correct += 1
        total += 1
        
    acc = correct / total if total > 0 else 0.0
    avg_e2e = np.mean(times_e2e) if times_e2e else 0.0
    p90_e2e = np.percentile(times_e2e, 90) if times_e2e else 0.0
    avg_model = np.mean(times_model) if times_model else 0.0
    p90_model = np.percentile(times_model, 90) if times_model else 0.0
    
    return acc, avg_e2e, p90_e2e, avg_model, p90_model

@torch.no_grad()
def evaluate_onnx_model(onnx_path, dataset, test_idx, max_len):
    # Force CPU for ONNX benchmark to match target Raspberry Pi
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    
    correct = 0
    total = 0
    
    subset = torch.utils.data.Subset(dataset, test_idx)
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_batch
    )
    
    times_e2e = []
    times_model = []
    
    # Warmup
    for i, (x, y, paths) in enumerate(loader):
        if i >= 2: break
        inputs = {input_name: x.numpy().astype(np.int64)}
        _ = session.run(None, inputs)
        
    for x, y, paths in loader:
        path = paths[0]
        # End-to-end
        t0 = time.perf_counter()
        xt = file_to_tensor(path, max_len=max_len)
        if xt is None:
            continue
        inputs = {input_name: xt.numpy().astype(np.int64)}
        out = session.run(None, inputs)[0]
        t1 = time.perf_counter()
        times_e2e.append((t1 - t0) * 1000.0)
        
        # Model-only
        t2 = time.perf_counter()
        _ = session.run(None, inputs)[0]
        t3 = time.perf_counter()
        times_model.append((t3 - t2) * 1000.0)
        
        pred = (out[0][0] > 0.5)
        label = (y.item() > 0.5)
        if pred == label:
            correct += 1
        total += 1
        
    acc = correct / total if total > 0 else 0.0
    avg_e2e = np.mean(times_e2e) if times_e2e else 0.0
    p90_e2e = np.percentile(times_e2e, 90) if times_e2e else 0.0
    avg_model = np.mean(times_model) if times_model else 0.0
    p90_model = np.percentile(times_model, 90) if times_model else 0.0
    
    return acc, avg_e2e, p90_e2e, avg_model, p90_model

def apply_pruning(model, amount=0.5):
    import torch.nn.utils.prune as prune
    pruned_model = copy.deepcopy(model)
    # Prune weights in convolutional and linear layers
    prune.l1_unstructured(pruned_model.conv_1, name="weight", amount=amount)
    prune.l1_unstructured(pruned_model.conv_2, name="weight", amount=amount)
    prune.l1_unstructured(pruned_model.fc_1, name="weight", amount=amount)
    
    # Make it permanent
    prune.remove(pruned_model.conv_1, "weight")
    prune.remove(pruned_model.conv_2, "weight")
    prune.remove(pruned_model.fc_1, "weight")
    return pruned_model

def apply_weight_sharing(model, k=16):
    shared_model = copy.deepcopy(model)
    for name, module in shared_model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            weights = module.weight.data.cpu().numpy()
            shape = weights.shape
            weights_flat = weights.flatten().reshape(-1, 1)
            
            # Fast KMeans to avoid timeout (max_iter=15, n_init=2)
            kmeans = KMeans(n_clusters=k, random_state=42, max_iter=15, n_init=2)
            kmeans.fit(weights_flat)
            
            new_weights = kmeans.cluster_centers_[kmeans.labels_]
            new_weights = new_weights.reshape(shape)
            module.weight.data = torch.from_numpy(new_weights).float().to(module.weight.device)
    return shared_model

def get_param_count(model):
    total = sum(p.numel() for p in model.parameters())
    return total

def main():
    print("==================================================")
    print("  MalConv Optimization & Benchmarking Pipeline   ")
    print("==================================================")
    
    device = torch.device("cpu") # Force CPU for local evaluation to represent Pi accurately
    ckpt_path = "outputs/models/group_split_model.pth"
    
    if not os.path.exists(ckpt_path):
        print(f"Error: Baseline checkpoint {ckpt_path} not found.")
        return 1
        
    ckpt = torch.load(ckpt_path, map_location="cpu")
    max_len = int(ckpt.get("max_len", 262144))
    window_size = int(ckpt.get("window_size", 512))
    
    print(f"Loaded Baseline Checkpoint: {ckpt_path}")
    print(f"Config: max_len={max_len} bytes, window_size={window_size}")
    
    # Instantiate Baseline Model
    baseline_model = MalConv(input_length=max_len, window_size=window_size)
    baseline_model.load_state_dict(ckpt["state_dict"])
    baseline_model.to(device)
    baseline_model.eval()
    
    # Base dataset evaluation on CPU
    dataset = BinaryFolderDataset(
        benign_dir=os.path.join("datasets", "local", "benign"),
        malicious_dir=os.path.join("datasets", "local", "malicious"),
        max_len=max_len,
        limit_per_class=200,
    )
    test_idx, _ = _get_split_indices(dataset, seed=1337, no_group_split=False)
    print(f"Evaluation Test Split Size: {len(test_idx)} samples")
    
    results = {}
    
    # ----------------------------------------------------
    # 1. Baseline Model (FP32)
    # ----------------------------------------------------
    print("\n--- 1. Evaluating Baseline Model ---")
    base_acc, base_e2e_avg, base_e2e_p90, base_model_avg, base_model_p90 = evaluate_pytorch_model(
        baseline_model, dataset, test_idx, device, max_len
    )
    base_size = get_file_size(ckpt_path)
    base_params = get_param_count(baseline_model)
    base_macs, base_non_zero_macs = estimate_macs(max_len, window_size)
    
    results["baseline"] = {
        "name": "Original Model (FP32)",
        "file_size_kb": base_size / 1024.0,
        "parameters": base_params,
        "macs": base_macs,
        "non_zero_macs": base_non_zero_macs,
        "accuracy": base_acc,
        "latency_e2e_ms": base_e2e_avg,
        "latency_e2e_p90_ms": base_e2e_p90,
        "latency_model_ms": base_model_avg,
        "latency_model_p90_ms": base_model_p90,
        "sparsity": 0.0,
        "type": "baseline"
    }
    print(f"Accuracy: {base_acc:.3f} | Latency: {base_model_avg:.2f}ms | Size: {base_size/1024:.1f} KB")

    # ----------------------------------------------------
    # 2. PyTorch Dynamic Quantization (INT8)
    # ----------------------------------------------------
    print("\n--- 2. Applying PyTorch Dynamic Quantization ---")
    quantized_model = torch.quantization.quantize_dynamic(
        baseline_model,
        {nn.Linear},
        dtype=torch.qint8
    )
    # Save quantized model to measure file size
    quant_ckpt_path = "outputs/models/quantized_dynamic.pth"
    torch.save({"state_dict": quantized_model.state_dict(), "max_len": max_len, "window_size": window_size}, quant_ckpt_path)
    
    q_acc, q_e2e_avg, q_e2e_p90, q_model_avg, q_model_p90 = evaluate_pytorch_model(
        quantized_model, dataset, test_idx, device, max_len
    )
    q_size = get_file_size(quant_ckpt_path)
    q_params = get_param_count(baseline_model) # parameters remain identical count-wise
    
    results["pytorch_quant"] = {
        "name": "PyTorch Dynamic Quant (INT8 Linear)",
        "file_size_kb": q_size / 1024.0,
        "parameters": q_params,
        "macs": base_macs,
        "non_zero_macs": base_non_zero_macs,
        "accuracy": q_acc,
        "latency_e2e_ms": q_e2e_avg,
        "latency_e2e_p90_ms": q_e2e_p90,
        "latency_model_ms": q_model_avg,
        "latency_model_p90_ms": q_model_p90,
        "sparsity": 0.0,
        "type": "quantization"
    }
    print(f"Accuracy: {q_acc:.3f} | Latency: {q_model_avg:.2f}ms | Size: {q_size/1024:.1f} KB")

    # ----------------------------------------------------
    # 3. ONNX Export and ONNX Dynamic Quantization (INT8)
    # ----------------------------------------------------
    print("\n--- 3. Applying ONNX Dynamic Quantization ---")
    onnx_path = "outputs/models/group_split_model.onnx"
    # Export baseline to ONNX if not exists
    if not os.path.exists(onnx_path):
        dummy = torch.zeros((1, max_len), dtype=torch.long)
        torch.onnx.export(
            baseline_model, dummy, onnx_path,
            input_names=["x"], output_names=["y"],
            opset_version=18, do_constant_folding=True
        )
        
    onnx_quant_path = "outputs/models/group_split_model_quant.onnx"
    quantize_dynamic(
        model_input=onnx_path,
        model_output=onnx_quant_path,
        weight_type=QuantType.QInt8
    )
    
    oq_acc, oq_e2e_avg, oq_e2e_p90, oq_model_avg, oq_model_p90 = evaluate_onnx_model(
        onnx_quant_path, dataset, test_idx, max_len
    )
    oq_size = get_file_size(onnx_quant_path)
    
    results["onnx_quant"] = {
        "name": "ONNX Dynamic Quant (INT8 Full)",
        "file_size_kb": oq_size / 1024.0,
        "parameters": base_params,
        "macs": base_macs,
        "non_zero_macs": base_non_zero_macs,
        "accuracy": oq_acc,
        "latency_e2e_ms": oq_e2e_avg,
        "latency_e2e_p90_ms": oq_e2e_p90,
        "latency_model_ms": oq_model_avg,
        "latency_model_p90_ms": oq_model_p90,
        "sparsity": 0.0,
        "type": "quantization"
    }
    print(f"Accuracy: {oq_acc:.3f} | Latency: {oq_model_avg:.2f}ms | Size: {oq_size/1024:.1f} KB")

    # ----------------------------------------------------
    # 4. Weight Pruning (50% Sparsity)
    # ----------------------------------------------------
    print("\n--- 4. Applying 50% Weight Pruning ---")
    pruned_50 = apply_pruning(baseline_model, amount=0.5)
    prune_50_ckpt = "outputs/models/pruned_50.pth"
    torch.save({"state_dict": pruned_50.state_dict(), "max_len": max_len, "window_size": window_size}, prune_50_ckpt)
    
    p50_acc, p50_e2e_avg, p50_e2e_p90, p50_model_avg, p50_model_p90 = evaluate_pytorch_model(
        pruned_50, dataset, test_idx, device, max_len
    )
    p50_size = get_file_size(prune_50_ckpt)
    _, p50_nz_macs = estimate_macs(max_len, window_size, sparse_ratio=0.49) # Embedding layer not pruned, so sparse ratio overall is slightly lower than 50%
    
    results["pruned_50"] = {
        "name": "Weight Pruning (50% Sparsity)",
        "file_size_kb": p50_size / 1024.0,
        "parameters": base_params,
        "macs": base_macs,
        "non_zero_macs": p50_nz_macs,
        "accuracy": p50_acc,
        "latency_e2e_ms": p50_e2e_avg,
        "latency_e2e_p90_ms": p50_e2e_p90,
        "latency_model_ms": p50_model_avg,
        "latency_model_p90_ms": p50_model_p90,
        "sparsity": 0.50,
        "type": "pruning"
    }
    print(f"Accuracy: {p50_acc:.3f} | Latency: {p50_model_avg:.2f}ms | Size: {p50_size/1024:.1f} KB")

    # ----------------------------------------------------
    # 5. Weight Pruning (75% Sparsity)
    # ----------------------------------------------------
    print("\n--- 5. Applying 75% Weight Pruning ---")
    pruned_75 = apply_pruning(baseline_model, amount=0.75)
    prune_75_ckpt = "outputs/models/pruned_75.pth"
    torch.save({"state_dict": pruned_75.state_dict(), "max_len": max_len, "window_size": window_size}, prune_75_ckpt)
    
    p75_acc, p75_e2e_avg, p75_e2e_p90, p75_model_avg, p75_model_p90 = evaluate_pytorch_model(
        pruned_75, dataset, test_idx, device, max_len
    )
    p75_size = get_file_size(prune_75_ckpt)
    _, p75_nz_macs = estimate_macs(max_len, window_size, sparse_ratio=0.74)
    
    results["pruned_75"] = {
        "name": "Weight Pruning (75% Sparsity)",
        "file_size_kb": p75_size / 1024.0,
        "parameters": base_params,
        "macs": base_macs,
        "non_zero_macs": p75_nz_macs,
        "accuracy": p75_acc,
        "latency_e2e_ms": p75_e2e_avg,
        "latency_e2e_p90_ms": p75_e2e_p90,
        "latency_model_ms": p75_model_avg,
        "latency_model_p90_ms": p75_model_p90,
        "sparsity": 0.75,
        "type": "pruning"
    }
    print(f"Accuracy: {p75_acc:.3f} | Latency: {p75_model_avg:.2f}ms | Size: {p75_size/1024:.1f} KB")

    # ----------------------------------------------------
    # 6. Weight Sharing (K-Means k=16)
    # ----------------------------------------------------
    print("\n--- 6. Applying K-Means Weight Sharing (k=16) ---")
    shared_16 = apply_weight_sharing(baseline_model, k=16)
    share_16_ckpt = "outputs/models/shared_16.pth"
    torch.save({"state_dict": shared_16.state_dict(), "max_len": max_len, "window_size": window_size}, share_16_ckpt)
    
    s16_acc, s16_e2e_avg, s16_e2e_p90, s16_model_avg, s16_model_p90 = evaluate_pytorch_model(
        shared_16, dataset, test_idx, device, max_len
    )
    s16_size = get_file_size(share_16_ckpt)
    # K-means represents weights as 4-bit indices (16 clusters = 4 bits)
    # Savings in memory are realized if stored as codebook + indices
    # Compressed footprint: codebook (16 * 32-bit = 64 bytes) + weight indices (params * 4-bits)
    est_shared_16_size_kb = (16 * 4 + (base_params * 4) / 8.0) / 1024.0
    
    results["shared_16"] = {
        "name": "Weight Sharing (K-Means k=16, 4-bit)",
        "file_size_kb": s16_size / 1024.0,
        "est_compressed_size_kb": est_shared_16_size_kb,
        "parameters": base_params,
        "macs": base_macs,
        "non_zero_macs": base_non_zero_macs,
        "accuracy": s16_acc,
        "latency_e2e_ms": s16_e2e_avg,
        "latency_e2e_p90_ms": s16_e2e_p90,
        "latency_model_ms": s16_model_avg,
        "latency_model_p90_ms": s16_model_p90,
        "sparsity": 0.0,
        "type": "sharing"
    }
    print(f"Accuracy: {s16_acc:.3f} | Latency: {s16_model_avg:.2f}ms | Size: {s16_size/1024:.1f} KB")

    # ----------------------------------------------------
    # 7. Weight Sharing (K-Means k=32)
    # ----------------------------------------------------
    print("\n--- 7. Applying K-Means Weight Sharing (k=32) ---")
    shared_32 = apply_weight_sharing(baseline_model, k=32)
    share_32_ckpt = "outputs/models/shared_32.pth"
    torch.save({"state_dict": shared_32.state_dict(), "max_len": max_len, "window_size": window_size}, share_32_ckpt)
    
    s32_acc, s32_e2e_avg, s32_e2e_p90, s32_model_avg, s32_model_p90 = evaluate_pytorch_model(
        shared_32, dataset, test_idx, device, max_len
    )
    s32_size = get_file_size(share_32_ckpt)
    # 32 clusters = 5 bits per index
    est_shared_32_size_kb = (32 * 4 + (base_params * 5) / 8.0) / 1024.0
    
    results["shared_32"] = {
        "name": "Weight Sharing (K-Means k=32, 5-bit)",
        "file_size_kb": s32_size / 1024.0,
        "est_compressed_size_kb": est_shared_32_size_kb,
        "parameters": base_params,
        "macs": base_macs,
        "non_zero_macs": base_non_zero_macs,
        "accuracy": s32_acc,
        "latency_e2e_ms": s32_e2e_avg,
        "latency_e2e_p90_ms": s32_e2e_p90,
        "latency_model_ms": s32_model_avg,
        "latency_model_p90_ms": s32_model_p90,
        "sparsity": 0.0,
        "type": "sharing"
    }
    print(f"Accuracy: {s32_acc:.3f} | Latency: {s32_model_avg:.2f}ms | Size: {s32_size/1024:.1f} KB")

    # ----------------------------------------------------
    # 8. Sequence Downscaling (128KB max_len)
    # ----------------------------------------------------
    print("\n--- 8. Evaluating Sequence Downscaling to 128KB ---")
    ds_128_len = 131072
    ds_128_dataset = BinaryFolderDataset(
        benign_dir=os.path.join("datasets", "local", "benign"),
        malicious_dir=os.path.join("datasets", "local", "malicious"),
        max_len=ds_128_len,
        limit_per_class=200,
    )
    ds128_acc, ds128_e2e_avg, ds128_e2e_p90, ds128_model_avg, ds128_model_p90 = evaluate_pytorch_model(
        baseline_model, ds_128_dataset, test_idx, device, ds_128_len
    )
    ds128_macs, ds128_nz_macs = estimate_macs(ds_128_len, window_size)
    
    results["downscale_128"] = {
        "name": "Sequence Downscale (128KB)",
        "file_size_kb": base_size / 1024.0, # no change to weights
        "parameters": base_params,
        "macs": ds128_macs,
        "non_zero_macs": ds128_nz_macs,
        "accuracy": ds128_acc,
        "latency_e2e_ms": ds128_e2e_avg,
        "latency_e2e_p90_ms": ds128_e2e_p90,
        "latency_model_ms": ds128_model_avg,
        "latency_model_p90_ms": ds128_model_p90,
        "sparsity": 0.0,
        "type": "downscaling"
    }
    print(f"Accuracy: {ds128_acc:.3f} | Latency: {ds128_model_avg:.2f}ms | MACs: {ds128_macs:,}")

    # ----------------------------------------------------
    # 9. Sequence Downscaling (64KB max_len)
    # ----------------------------------------------------
    print("\n--- 9. Evaluating Sequence Downscaling to 64KB ---")
    ds_64_len = 65536
    ds_64_dataset = BinaryFolderDataset(
        benign_dir=os.path.join("datasets", "local", "benign"),
        malicious_dir=os.path.join("datasets", "local", "malicious"),
        max_len=ds_64_len,
        limit_per_class=200,
    )
    ds64_acc, ds64_e2e_avg, ds64_e2e_p90, ds64_model_avg, ds64_model_p90 = evaluate_pytorch_model(
        baseline_model, ds_64_dataset, test_idx, device, ds_64_len
    )
    ds64_macs, ds64_nz_macs = estimate_macs(ds_64_len, window_size)
    
    results["downscale_64"] = {
        "name": "Sequence Downscale (64KB)",
        "file_size_kb": base_size / 1024.0,
        "parameters": base_params,
        "macs": ds64_macs,
        "non_zero_macs": ds64_nz_macs,
        "accuracy": ds64_acc,
        "latency_e2e_ms": ds64_e2e_avg,
        "latency_e2e_p90_ms": ds64_e2e_p90,
        "latency_model_ms": ds64_model_avg,
        "latency_model_p90_ms": ds64_model_p90,
        "sparsity": 0.0,
        "type": "downscaling"
    }
    print(f"Accuracy: {ds64_acc:.3f} | Latency: {ds64_model_avg:.2f}ms | MACs: {ds64_macs:,}")

    # ----------------------------------------------------
    # 10. Sequence Downscaling (32KB max_len)
    # ----------------------------------------------------
    print("\n--- 10. Evaluating Sequence Downscaling to 32KB ---")
    ds_32_len = 32768
    ds_32_dataset = BinaryFolderDataset(
        benign_dir=os.path.join("datasets", "local", "benign"),
        malicious_dir=os.path.join("datasets", "local", "malicious"),
        max_len=ds_32_len,
        limit_per_class=200,
    )
    ds32_acc, ds32_e2e_avg, ds32_e2e_p90, ds32_model_avg, ds32_model_p90 = evaluate_pytorch_model(
        baseline_model, ds_32_dataset, test_idx, device, ds_32_len
    )
    ds32_macs, ds32_nz_macs = estimate_macs(ds_32_len, window_size)
    
    results["downscale_32"] = {
        "name": "Sequence Downscale (32KB)",
        "file_size_kb": base_size / 1024.0,
        "parameters": base_params,
        "macs": ds32_macs,
        "non_zero_macs": ds32_nz_macs,
        "accuracy": ds32_acc,
        "latency_e2e_ms": ds32_e2e_avg,
        "latency_e2e_p90_ms": ds32_e2e_p90,
        "latency_model_ms": ds32_model_avg,
        "latency_model_p90_ms": ds32_model_p90,
        "sparsity": 0.0,
        "type": "downscaling"
    }
    print(f"Accuracy: {ds32_acc:.3f} | Latency: {ds32_model_avg:.2f}ms | MACs: {ds32_macs:,}")

    # ----------------------------------------------------
    # Save Results to JSON
    # ----------------------------------------------------
    out_json = "outputs/optimization_results.json"
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved optimization results to: {os.path.abspath(out_json)}")
    
    # ----------------------------------------------------
    # Generate Static Trade-off Chart using Matplotlib
    # ----------------------------------------------------
    print("\nGenerating static trade-off comparison chart...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Embedded AI Optimization Trade-off Analysis (MalConv Model)", fontsize=16, fontweight="bold")
    
    names = [r["name"] for r in results.values()]
    accuracies = [r["accuracy"] * 100.0 for r in results.values()]
    latencies = [r["latency_model_ms"] for r in results.values()]
    sizes = [r.get("est_compressed_size_kb", r["file_size_kb"]) for r in results.values()]
    macs = [r["non_zero_macs"] / 1e6 for r in results.values()]
    types = [r["type"] for r in results.values()]
    
    colors_map = {
        "baseline": "#1f77b4",       # Blue
        "quantization": "#2ca02c",   # Green
        "pruning": "#d62728",        # Red
        "sharing": "#9467bd",        # Purple
        "downscaling": "#ff7f0e"     # Orange
    }
    colors = [colors_map[t] for t in types]
    
    # Chart 1: Accuracy vs Latency (Model-only)
    for name, acc, lat, color in zip(names, accuracies, latencies, colors):
        ax1.scatter(lat, acc, color=color, s=150, alpha=0.8, edgecolors="black")
        ax1.annotate(name.split(" (")[0], (lat + 1, acc), fontsize=9)
    ax1.set_xlabel("Inference Latency (Model-only) [ms]", fontsize=11)
    ax1.set_ylabel("Test Split Accuracy [%]", fontsize=11)
    ax1.set_title("Accuracy vs Latency Trade-off", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Chart 2: Footprint Size Comparison
    y_pos = np.arange(len(names))
    bars = ax2.barh(y_pos, sizes, color=colors, edgecolor="black", height=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=9)
    ax2.invert_yaxis()  # top-down
    ax2.set_xlabel("Model Storage Size (or Est. Shared Size) [KB]", fontsize=11)
    ax2.set_title("Memory Footprint Comparison", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    # Add values on bars
    for bar in bars:
        width = bar.get_width()
        ax2.text(width + 20, bar.get_y() + bar.get_height()/2, f"{int(width)} KB", 
                 va="center", ha="left", fontsize=9, fontweight="bold")
                 
    # Chart 3: Computational Cost (MACs)
    bars_mac = ax3.barh(y_pos, macs, color=colors, edgecolor="black", height=0.6)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([n.split(" (")[0] for n in names], fontsize=9)
    ax3.invert_yaxis()
    ax3.set_xlabel("Computation Complexity [Million MACs]", fontsize=11)
    ax3.set_title("Computational Complexity Comparison", fontsize=12, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.5)
    for bar in bars_mac:
        width = bar.get_width()
        ax3.text(width + 5, bar.get_y() + bar.get_height()/2, f"{width:.1f}M", 
                 va="center", ha="left", fontsize=9, fontweight="bold")

    # Chart 4: Multi-criteria Trade-off Index (Efficiency Score)
    # Efficiency Score = (Accuracy / Baseline_Accuracy) / (Latency/Baseline_Latency * Size/Baseline_Size * MACs/Baseline_MACs)^(1/3)
    # Just a visual index showing overall goodness
    eff_scores = []
    for r in results.values():
        size_comp = r.get("est_compressed_size_kb", r["file_size_kb"])
        norm_acc = r["accuracy"] / base_acc
        norm_lat = max(r["latency_model_ms"] / base_model_avg, 0.05)
        norm_size = size_comp / (base_size / 1024.0)
        norm_mac = r["non_zero_macs"] / base_macs
        score = norm_acc / ((norm_lat * norm_size * norm_mac) ** (1.0/3.0))
        eff_scores.append(score)
        
    bars_eff = ax4.bar(y_pos, eff_scores, color=colors, edgecolor="black", width=0.5)
    ax4.set_xticks(y_pos)
    ax4.set_xticklabels([n.split(" (")[0] for n in names], rotation=45, ha="right", fontsize=9)
    ax4.set_ylabel("Embedded Efficiency Index", fontsize=11)
    ax4.set_title("Embedded Efficiency Index (Higher is Better)", fontsize=12, fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.5)
    
    # Draw legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, edgecolor="black", label=t.capitalize()) for t, c in colors_map.items()]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=11, frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    chart_path = "summary/usb_reports/tradeoff_comparison.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved static trade-off chart to: {os.path.abspath(chart_path)}")
    print("==================================================")
    
if __name__ == "__main__":
    main()

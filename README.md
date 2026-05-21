# 🛡️ USB Malware Sanitizer — Embedded AI Project

> **Course**: Embedded Artificial Intelligence (EAI)  
> **Advisor**: Dr. Dubacharla Gyaneshwar  
> **Target Platform**: Raspberry Pi 4 Model B (ARM Cortex-A72)  
> **Model**: MalConv-style Gated 1-D CNN — raw byte malware classifier

---

## 📌 Project Overview

This project implements an **AI-powered USB Sanitizer Kiosk** that scans files on incoming USB drives *before* they ever reach a trusted workstation. A lightweight deep-learning model (MalConv) reads raw bytes of executables and outputs a malware suspicion score. Files scoring above 0.5 are quarantined automatically.

The core research contribution is a **comprehensive model optimization and hardware feasibility study** — applying Quantization, Weight Pruning, K-Means Weight Sharing, and Sequence Downscaling to the baseline model and benchmarking every variant against the target Pi hardware constraints.

---

## 📁 Project Structure

```
usb_scanner/
│
├── models/                        # All neural network code
│   ├── malconv/
│   │   ├── model.py               # MalConv Gated CNN architecture
│   │   ├── train.py               # Training pipeline (group-split)
│   │   ├── dataset.py             # BinaryFolderDataset loader
│   │   ├── preprocess.py          # file_to_tensor(), byte preprocessing
│   │   └── export_malconv_onnx.py # Export checkpoint → ONNX format
│   ├── usb_cli/                   # CLI scanner entry point
│   ├── fusion/                    # Multi-stage detection pipeline
│   └── utils/                     # Shared utilities
│
├── reports/                       # 📊 All EAI optimization analysis
│   ├── eai_report.tex             # Full LaTeX report (compile with pdflatex)
│   ├── optimize_and_benchmark.py  # Main benchmarking script (all 10 variants)
│   ├── verify_all_methods.py      # Per-sample accuracy verification
│   ├── overfitting_analysis.py    # 30-epoch train/val/test curve analysis
│   ├── compare_checkpoints.py     # Checkpoint comparison utilities
│   ├── analyze_dataset.py         # Dataset statistics & visualization
│   ├── model_stats.py             # Parameter counts, MAC estimation
│   ├── latency_benchmark.py       # Latency profiling on CPU
│   ├── dashboard.html             # Interactive Chart.js trade-off dashboard
│   ├── tradeoff_comparison.png    # 4-panel optimization trade-off chart
│   ├── overfitting_curves.png     # Train/Val/Test learning curves
│   ├── optimization_results.json  # Raw benchmark numbers (all variants)
│   └── training_history.json      # Epoch-level training log (30 epochs)
│
├── datasets/
│   ├── local/
│   │   ├── benign/                # Place benign .exe files here (not committed)
│   │   └── malicious/             # Place UPX-packed .exe files here (not committed)
│   └── extractors/                # Dataset preparation scripts
│
├── outputs/
│   ├── models/                    # Saved .pth checkpoints (not committed — too large)
│   └── quarantine/                # Runtime quarantine directory
│
├── tools/                         # Helper scripts
│   ├── copy_benign.ps1            # Copy System32 binaries to datasets/local/benign/
│   ├── make_packed_demo.ps1       # UPX-pack binaries for malicious dataset
│   ├── export_demo_pack.ps1       # Bundle demo USB drive
│   ├── av_demo.ps1                # Windows Defender comparison demo
│   └── make_high_entropy_file.py  # Generate high-entropy test files
│
├── docs/
│   ├── why_raspberry_pi_sanitizer.md   # Architecture rationale
│   └── defender_vs_model_demo.md       # Defender comparison notes
│
├── hardware/                      # Raspberry Pi deployment configs
├── requirements.txt               # Python dependencies (development)
├── requirements-pi.txt            # Python dependencies (Raspberry Pi)
└── README.md                      # This file
```

---

## 🧠 Model Architecture — MalConv Gated CNN

| Layer | Details | Output Shape | MACs |
|---|---|---|---|
| Embedding | 257 tokens → 8-dim dense | (B, 8, L) | lookup |
| Conv1D Value | 128 filters, k=512, s=512 | (B, 128, L') | 268 M |
| Conv1D Gate + Sigmoid | 128 filters, k=512, s=512 | (B, 128, L') | 268 M |
| Elementwise Gate Multiply | — | (B, 128, L') | — |
| Adaptive Max Pool | output=1 | (B, 128) | — |
| Linear 128→128 + ReLU | — | (B, 128) | 16 K |
| Linear 128→1 + Sigmoid | — | (B, 1) | 128 |

**Total parameters:** 1,067,529  
**Total MACs (256 KB input):** 536,887,424  
**Baseline model size:** 4.08 MB (FP32)

---

## ⚡ Optimization Results Summary

All results verified per-sample (TP=4, TN=4, FP=0, FN=0 on 8-sample test split).

| Model Variant | Method | Size (KB) | Active MACs | Latency | Test Acc |
|---|---|--:|--:|--:|:-:|
| Original FP32 | Baseline | 4,174 | 537 M | 9.91 ms | 100% |
| PyTorch Dynamic INT8 | Quantization | 4,127 | 537 M | 9.76 ms | 100% |
| ONNX Dynamic INT8 | Quantization | 3,098 | 537 M | 15.88 ms | 100% |
| Weight Pruning 50% | Pruning | 4,174 | 274 M | 13.44 ms | 100% |
| Weight Pruning 75% | Pruning | 4,174 | 140 M | 15.58 ms | 100% |
| **K-Means k=16** ⭐ | **Sharing** | **521** | 537 M | 14.08 ms | **100%** |
| K-Means k=32 | Sharing | 652 | 537 M | 11.25 ms | 100% |
| Sequence Downscale 128 KB | Downscaling | 4,174 | 268 M | 4.84 ms | 100% |
| Sequence Downscale 64 KB | Downscaling | 4,174 | 134 M | 2.87 ms | 100% |
| **Sequence Downscale 32 KB** ⭐ | **Downscaling** | 4,174 | **67 M** | **1.82 ms** | **100%** |

> ⭐ **Best for memory:** K-Means k=16 → **87.5% size reduction** (4,174 KB → 521 KB)  
> ⭐ **Best for latency:** Sequence Downscale 32 KB → **5.44× speedup** (9.91 ms → 1.82 ms)

---

## ⚠️ Overfitting Analysis (Honest Assessment)

A 30-epoch re-training experiment with per-epoch train/val/test logging revealed:

| Split | Loss | Accuracy |
|---|---|---|
| Train (52 samples) | 0.0014 | **100.0%** |
| Validation (6 samples) | 0.3872 | **83.3%** ⚠️ |
| Test (8 samples) | 0.0516 | **100.0%** |

**Finding:** Overfitting is present — training loss collapses to ~0 by epoch 2 while validation loss diverges upward to 0.80. The 16.7% train-vs-val accuracy gap exceeds the acceptable threshold.

**Root causes:** Dataset too small (66 samples), no regularisation (no dropout/weight decay), and the UPX classification task is trivially separable.

**Why test accuracy remains 100%:** The decision margin is enormous — benign files score 0.000–0.032 and packed files score 0.773–0.999. The 0.5 boundary is never crossed under any compression method.

**Future fix:** Add dropout (p=0.3), weight decay, early stopping at epoch 3, and retrain on EMBER-2018/SOREL-20M.

See `reports/overfitting_curves.png` and `reports/training_history.json` for full details.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare dataset

```powershell
# Copy Windows System32 binaries as benign samples
.\tools\copy_benign.ps1

# UPX-pack them to create malicious samples
.\tools\make_packed_demo.ps1
```

### 3. Train the model

```bash
python -m models.malconv.train \
    --benign-dir datasets/local/benign \
    --malicious-dir datasets/local/malicious \
    --max-len 262144 \
    --epochs 5 \
    --out outputs/models/group_split_model.pth
```

### 4. Run the benchmark (all 10 optimization variants)

```bash
python -m reports.optimize_and_benchmark
```

### 5. Verify accuracy per-sample

```bash
python -m reports.verify_all_methods
```

### 6. Overfitting analysis

```bash
python -m reports.overfitting_analysis
```

### 7. Open the interactive dashboard

Open `reports/dashboard.html` in any browser — no server required.

### 8. Compile the LaTeX report

```bash
cd reports
pdflatex eai_report.tex
pdflatex eai_report.tex   # Run twice for correct cross-references
```

---

## 📊 Charts & Visualisations

| File | Description |
|---|---|
| `reports/tradeoff_comparison.png` | 4-panel trade-off chart: memory, latency, MACs, efficiency index |
| `reports/overfitting_curves.png` | Train/Val/Test loss & accuracy curves over 30 epochs |
| `reports/dashboard.html` | Interactive Chart.js dashboard — all variants, all metrics |
| `reports/eai_report.tex` | Full LaTeX technical report (compiles to PDF) |

---

## 🎯 Deployment Recommendations

| Target Hardware | Recommended Variant | Reason |
|---|---|---|
| Raspberry Pi 4 | Sequence Downscale 32 KB | 1.82 ms latency, real-time scanning |
| Raspberry Pi Zero | K-Means k=16 + 32 KB downscale | 521 KB + fast inference |
| STM32F4 MCU | K-Means k=16 | Fits in < 1 MB Flash |
| Edge Server | ONNX INT8 | Batch throughput, energy efficient |

---

## 📦 Dependencies

```
torch >= 2.0
onnxruntime
onnx
scikit-learn
numpy
matplotlib
psutil
```

See `requirements.txt` for pinned versions. For Raspberry Pi, use `requirements-pi.txt`.

---

## 📄 License

This project is submitted as coursework for the Embedded Artificial Intelligence course.  
All code is original and written for educational purposes.

# usb_scanner (MalConv USB Scanner Rapid Prototype)

This is a small end-to-end prototype of a **MalConv-style gated CNN** that scores Windows executables by reading raw bytes.

## Setup

```powershell
pip install -r requirements.txt
```

## Data layout (recommended)

Create two folders:

- `data/benign/`  (benign `.exe`/`.dll`)
- `data/malicious/` (test samples)

Notes:
- For safety, **do not download real malware**. If you need a harmless test file, use the **EICAR** test string/file.
- This is a demo pipeline; model quality depends entirely on your dataset.

Safe "malicious" sample for demo:
- Download the EICAR test file (NOT real malware): https://www.eicar.org/download-anti-malware-testfile/
- Put `eicar.com` into `data/malicious/`
- When scanning, include `.com` via `--extensions .exe,.dll,.com`

## Demo idea (recommended): Packed vs Clean (NO malware)

If you want a strong demo without real malware, you can train the model to flag **packed/obfuscated binaries** as “suspicious”.
This demonstrates the “structural analysis” value proposition.

1) Install UPX (packer) and make sure `upx` is available in PATH.
	- If you don’t want to install system-wide, you can also place `upx.exe` at `tools/upx.exe`.

2) Generate a paired dataset: clean binaries in `data/benign/` and UPX-packed copies in `data/malicious/`:

```powershell
powershell -ExecutionPolicy Bypass -File tools\make_packed_demo.ps1 -Count 50 -Random -Extensions .exe
```

3) Train:

```powershell
python train.py --benign-dir data/benign --malicious-dir data/malicious --epochs 5 --batch-size 4 --out malconv_model.pth
```

4) Scan and present results (Top-N + metadata):

```powershell
python usb_monitor.py --scan data --checkpoint malconv_model.pth --extensions .exe --top 20 --metadata
```

## Optional: Scan inside .zip archives

By default the scanner looks at files on disk. You can also scan inside ZIP archives (e.g., a USB contains `something.zip`).

Scan a zip and print results:

```powershell
python usb_monitor.py --scan demo_pack.zip --checkpoint demo_pack\malconv_model.pth --extensions .exe --archives --top 20 --metadata
```

If you want a “remove only the flagged files from the zip” style demo, use `--zip-mode sanitize`.
This creates a new file `*.sanitized.zip` without the flagged members (the original zip is left unchanged):

```powershell
python usb_monitor.py --scan demo_pack.zip --checkpoint demo_pack\malconv_model.pth --extensions .exe --archives --zip-mode sanitize
```

For regular files (not archives), you can optionally quarantine flagged files:

```powershell
python usb_monitor.py --scan E:\\ --checkpoint malconv_model.pth --extensions .exe,.dll,.com --action quarantine --quarantine-dir quarantine
```

### Quick populate benign samples (Windows)

This copies a subset of Windows binaries into `data/benign` for training/demo.

```powershell
powershell -ExecutionPolicy Bypass -File tools\copy_benign.ps1 -Count 100 -Random
```

## Train (32GB machine)

### Option A: Real folder training

```powershell
python train.py --benign-dir data/benign --malicious-dir data/malicious --epochs 5 --batch-size 4 --out malconv_model.pth
```

### Option B: Demo random training (just to validate code runs)

```powershell
python train.py --demo-random --epochs 2 --batch-size 2 --out malconv_model.pth
```

The output checkpoint is a dict containing:
- `state_dict`
- `max_len`
- `window_size`

## Scan (friend laptop / 16GB)

Scan a directory (no USB needed):

```powershell
python usb_monitor.py --scan C:\\Path\\To\\Folder --checkpoint malconv_model.pth
```

Watch for new drives and scan when inserted:

```powershell
python usb_monitor.py --watch --checkpoint malconv_model.pth --extensions .exe,.dll,.com
```

## Gating logic (for documentation)

The model uses **two convolutions over the same embedded byte stream**:

- `cnn_value = conv_1(x)` learns *features* ("what patterns exist")
- `cnn_gate = sigmoid(conv_2(x))` learns a *mask* in $[0,1]$ ("how important is this region")
- `gated = cnn_value * cnn_gate` suppresses unimportant regions and passes salient regions

Intuition:
- If `cnn_gate` is near 0 for a window, that window’s features are effectively ignored.
- If `cnn_gate` is near 1, that window’s features pass through.

This makes the model behave a bit like a learned attention mechanism but implemented with efficient convolutions and elementwise gating.

# Project Handoff / What Copilot Built

Your friend can read this file to see exactly what was created and how to run it.

## What’s in the repo

- `model.py` – MalConv (gated CNN) architecture.
- `preprocess.py` – reads a binary and converts it to a padded tensor (padding index = 256).
- `dataset.py` – folder dataset loader (`data/benign`, `data/malicious`) + DataLoader collate.
- `train.py` – training CLI (real folder training or `--demo-random`). Saves `malconv_model.pth`.
- `usb_monitor.py` – scan a folder OR watch for new drives and scan inserted USBs.
- `requirements.txt` – torch/numpy/psutil.
- `README.md` – setup, commands, and gating explanation.

## Quick start (Windows PowerShell)

Install deps:

```powershell
pip install -r requirements.txt
```

Populate benign samples quickly (optional):

```powershell
powershell -ExecutionPolicy Bypass -File tools\copy_benign.ps1 -Count 100 -Random
```

Demo-run training (validates code works):

```powershell
python train.py --demo-random --epochs 2 --batch-size 2 --out malconv_model.pth
```

Scan a folder:

```powershell
python usb_monitor.py --scan C:\\Windows\\System32 --checkpoint malconv_model.pth --extensions .exe,.dll,.com --top 20 --metadata
```

Watch for USB insertion:

```powershell
python usb_monitor.py --watch --checkpoint malconv_model.pth --extensions .exe,.dll,.com
```

## Demo idea (no malware): Packed vs Clean

This is a safe demo that still supports a strong “structural analysis” story.

1) Install UPX and ensure `upx` is on PATH (or place `tools\upx.exe`).

2) Generate clean vs packed samples:

```powershell
powershell -ExecutionPolicy Bypass -File tools\make_packed_demo.ps1 -Count 50 -Random -Extensions .exe
```

3) Train:

```powershell
python train.py --benign-dir data/benign --malicious-dir data/malicious --epochs 5 --batch-size 4 --out malconv_model.pth
```

4) Scan:

```powershell
python usb_monitor.py --scan data --checkpoint malconv_model.pth --extensions .exe --top 20 --metadata
```

## How your friend can “see what’s happening”

You can’t share the exact same Copilot Chat session live to another person.
Use one of these instead:

1) **VS Code Live Share** (best): host starts a Live Share session and invites the friend.
   - Friend sees the same files and your edits in real time.
   - Copilot itself remains per-user, but the shared code changes are visible.

2) **GitHub private repo**: push commits frequently.
   - Friend pulls/reads commits and diffs.

3) **Screen share** (Google Meet/Discord/Zoom): friend literally watches your screen.

If you want, add your friend as a collaborator on the GitHub repo and/or start Live Share for real-time watching.

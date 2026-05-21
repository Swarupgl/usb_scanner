# Defender vs. Our USB Sanitizer: Safe Proof Demo (No Real Malware)

This write-up gives you a **safe, classroom-friendly “proof”** for the claim:

> “Renaming the file extension can bypass *human expectations* and some *real-time workflows*, but our sanitizer/model does not trust the extension — it looks at the file’s bytes (magic + structure), so it catches disguised executables.”

Important: modern AVs (Windows Defender, Kaspersky, etc.) are **not dumb**. If you do a full scan, they often look inside files. Also, AVs focus on **known-malicious content**, not on “policy violations” like “an EXE pretending to be a JPG”. Your project is a **USB safety gate**: it can block suspicious *structure* even if it’s not proven malware.

---

## What your sir asked, translated into a test you can actually prove

Your sir wants evidence for this story:

- “If malware is renamed from `.exe` to `.jpg`, Defender won’t detect it.”

The issue: you can’t guarantee that outcome on every modern AV configuration (and you should not demonstrate with real malware on a real laptop).

So instead, do a **stronger, safer proof**:

- Create a **benign executable** (e.g., Windows `notepad.exe`) and rename it to `.jpg`.
- Windows Defender will typically **not quarantine it** because it is not malware.
- Your sanitizer will **quarantine immediately** because the file’s internal magic is `MZ` (PE executable) while the extension claims `jpg`.

That is an objective “extension trick” proof: your system is **extension-agnostic** and checks **internal structure**.

---

## Demo A (recommended): “Extension spoofing” (safe + reliable)

### Step 1 — Generate the demo files (safe)

Run this in PowerShell from the repo root:

- Command: `powershell -ExecutionPolicy Bypass -File tools/av_demo.ps1`

It creates:

- `outputs/demo/av_demo/benign_notepad.exe`
- `outputs/demo/av_demo/vacation_photo.jpg`  (this is actually the same EXE bytes)

It also prints SHA256 hashes to prove the bytes are identical.

### Step 2 — The “traditional Defender” observation

1. Copy `outputs/demo/av_demo/vacation_photo.jpg` to a USB.
2. Plug the USB into a Windows laptop.
3. Show your sir:
   - The file is **not auto-deleted** just because it is “weird”.
   - If someone double-clicks it, Windows won’t open it as a photo (it’s not a real JPEG).

What you’re proving here:

- AV is primarily a **malware detector**.
- A disguised file that is not malware usually won’t be quarantined.
- But it is still a **delivery trick**: users see `.jpg` and may trust it.

### Step 3 — Run your sanitizer and show it catches the trick

Run the sanitizer against that same folder (works on Windows too):

- Command (no ONNX needed for this proof):
  - `python hardware/pi/pi_usb_sanitizer.py --scan outputs/demo/av_demo --quarantine outputs/quarantine`

Expected behavior:

- The sanitizer prints something like:
  - `action=quarantined (mismatched)`
  - `kind=pe`
  - and quarantines `vacation_photo.jpg`

Why it works (the one sentence to say):

- “Our system ignores the filename and checks **magic/structure**; `vacation_photo.jpg` starts with `MZ`, so it’s executable content disguised as an image.”

---

## Demo B (optional): Show MalConv score is independent of the filename

If you want an “AI-model” angle (not only the Stage-1 mismatch policy), run with your MalConv ONNX:

- Model file already in this repo:
  - `outputs/pi/models/malconv_finalfinal.onnx`

Command:

- `python hardware/pi/pi_usb_sanitizer.py --scan outputs/demo/av_demo --quarantine outputs/quarantine --onnx outputs/pi/models/malconv_finalfinal.onnx`

What to show:

- The “risk / stage3 score” comes from bytes, so renaming the file does not change what the model sees.

Note:

- This model is intended for **PE-like malware detection**. It is not a general “detect anything hidden in any image” model.

---

## Safe alternative to “Frankenstein EXE”: high-entropy random file

If someone suggests “inject random bytes into an EXE to confuse AV”, don’t do that.

Instead, you can safely demonstrate the **same ‘packed/encrypted-like entropy’ idea** with a file that is *pure random bytes* (not an EXE, not malware):

1) Create a high-entropy file:

- `python tools/make_high_entropy_file.py --out outputs/demo/av_demo/chaos.bin --size 2097152`

2) Traditional AV observation:

- Defender typically reports **no threats** because it’s not malware.

3) Your sanitizer policy:

- `python hardware/pi/pi_usb_sanitizer.py --scan outputs/demo/av_demo --quarantine outputs/quarantine`

Expected: it may quarantine/flag it due to **high entropy**, demonstrating your gate can block packed/encrypted-looking content even when there’s no known signature.

---

## About EICAR (only if your sir insists on an AV pop-up)

The EICAR test string is **safe** and widely used to test AV pipelines. It is *not* a real virus.

If you want EICAR files too, run:

- `powershell -ExecutionPolicy Bypass -File tools/av_demo.ps1 -IncludeEicar`

It tries to create:

- `outputs/demo/av_demo/eicar.com`, `eicar.exe`, `eicar.jpg`, `eicar_photo.jpg`, `eicar.txt`, `eicar.zip`

What may happen:

- Defender may quarantine it immediately even when renamed to `.jpg` (that’s normal — modern AVs scan by content in many cases).

So do not claim:

- “Defender never detects EICAR if it’s `.jpg`.”

Instead claim:

- “AV behavior depends on **real-time vs full scan**, settings, performance tradeoffs, and file-type heuristics. Our sanitizer is deterministic: it always enforces structure/extension integrity and then applies ML/signatures.”

---

## What screenshots / evidence to bring

- PowerShell output from `tools/av_demo.ps1` showing identical SHA256 for `.exe` and `.jpg`.
- Sanitizer terminal output showing `quarantined (mismatched)` for the spoofed `.jpg`.
- Optional: Windows Security → Protection History screen (whatever it shows on your laptop).

---

## Safety notes

- Do **not** use real malware for a live demo.
- Do **not** double-click/run the disguised file during the demo.
- Prefer a VM or a spare test laptop if you demonstrate EICAR.

# Raspberry Pi 4B (8GB) — USB Sanitizer + Touchscreen Kiosk (from scratch)

This guide assumes **Raspberry Pi OS Desktop** (recommended if you want the touchscreen UI).

## 0) Hardware connections (simple)
- Power the Pi from its **USB‑C power** port (use an official-ish 5V/3A PSU if possible).
  - Note: on a Raspberry Pi 4B the **USB‑C port is power-only** (not a data/USB gadget link to your PC). A USB‑C↔USB‑C cable is great for power, but you’ll still use **network (SSH/SCP)** or a **USB stick** to move files.
- Connect your **touchscreen**.
  - If your display is **HDMI + USB touch**, it’s usually plug-and-play.
  - If your display is **GPIO-mounted** (SPI/DPI “hat” style), you must enable the right interfaces and add a boot overlay (details below).
- Plug your **test pendrive** into a USB-A port.
- (Optional) Ethernet is easiest for setup; Wi‑Fi also works.

### GPIO-mounted display (SPI/DPI) — how to initialize/configure
GPIO displays are not one universal standard. The exact config depends on:
- Brand/model (often Waveshare / HyperPixel / “3.5\" TFT”)
- Bus type: **SPI** (most common on small TFTs) vs **DPI** (parallel RGB) vs **DSI** (ribbon cable; not “GPIO pins”)
- Touch controller: **USB**, **I2C**, or **SPI**

Because of this, the safest workflow is:

1) Identify your display
- Look at the back of the PCB for a controller/board name (examples: `ILI9341`, `ILI9486`, `ST7735`, `XPT2046`, `ADS7846`, `HyperPixel`, `Waveshare35a`).
- If you can, share the exact product link/model number.

2) Check your Pi OS boot config location
- Newer Raspberry Pi OS (Bookworm) uses:
  - `/boot/firmware/config.txt`
- Older images often use:
  - `/boot/config.txt`

3) Enable required interfaces (SPI/I2C)
Run:
```bash
sudo raspi-config
```
Then:
- `Interface Options` → enable **SPI** (most GPIO TFTs)
- enable **I2C** if your touch controller uses I2C
Reboot:
```bash
sudo reboot
```

4) Apply the vendor overlay (the important step)
Most GPIO displays ship with a ready overlay or install script.
- If the vendor provided an installer (Waveshare/HyperPixel often do), use that first.
- Otherwise you add a line like this to the config file (replace the name/params with your real values):
```text
dtoverlay=YOUR_OVERLAY_NAME,param1=value1,param2=value2
```

Important:
- `config.txt` is a boot configuration file. Do **not** try to run it as a command (e.g. don’t type `/boot/firmware/config.txt` in the terminal).
- Do **not** paste placeholder text like `dtoverlay=<...>` into `config.txt` — those angle-bracket placeholders are only documentation.

To see available overlays on your system:
```bash
ls /boot/overlays 2>/dev/null | head
ls /boot/firmware/overlays 2>/dev/null | head
dtoverlay -l
```

After editing the config file, reboot.

5) Verify the display is active
- Check kernel messages:
```bash
dmesg -T | egrep -i "spi|drm|fb|ili|st77|waveshare|hyperpixel" | tail -n 60
```
- Check what displays/framebuffers exist:
```bash
ls -l /dev/fb*
ls -l /dev/dri/* 2>/dev/null
```

6) Touch verification
- If touch is USB: it usually appears automatically.
- If touch is not aligned (rotated), you’ll likely need to set rotation in the overlay or in desktop settings.

If you tell me your exact display model (or send a photo of the back text), I’ll give you the exact `dtoverlay=...` line and rotation values for it.

#### Common case: 3.5" SPI TFT + XPT2046 (resistive touch)
This is the most common “GPIO 3.5 inch” display style.

**What it is**
- Display: SPI framebuffer (often an ILI9486/ILI9341-class controller)
- Touch: XPT2046 over SPI (kernel overlay is usually `ads7846`, which is XPT2046-compatible)

**1) Enable SPI**
```bash
sudo raspi-config
```
Interface Options → **SPI** → Enable → reboot.

**2) Edit the boot config**
- Bookworm: `sudo nano /boot/firmware/config.txt`
- Older: `sudo nano /boot/config.txt`

**3) Add/adjust these lines (typical Waveshare-style wiring)**
Add near the bottom:
```text
# --- 3.5" SPI TFT + XPT2046 touch ---
dtparam=spi=on

# Display overlay (one of these usually exists; see troubleshooting if not)
dtoverlay=waveshare35a,rotate=270,speed=48000000,fps=60

# Touch overlay (XPT2046-compatible)
dtoverlay=ads7846,penirq=25,swapxy=1,xmin=200,xmax=3900,ymin=200,ymax=3900,pmax=255
```

Notes:
- `rotate=270` is common for these screens; if your UI is sideways/upside-down try `rotate=0/90/180/270`.
- `penirq=25` is the common interrupt pin for many 3.5" hats; some boards use a different GPIO. If touch doesn’t respond, this is the first thing to verify.

**4) Reboot**
```bash
sudo reboot
```

**5) Verify display + touch are detected**
```bash
dmesg -T | egrep -i "waveshare|fb|fbtft|spi|ads7846|xpt2046|touch" | tail -n 80
ls -l /dev/fb*
cat /proc/bus/input/devices | egrep -i "ads|xpt|touch" -n
```

**6) Troubleshooting (quick)**
- If `waveshare35a` overlay is missing:
  - List overlays: `ls /boot/firmware/overlays | egrep -i "wave|35|pitft|fb"`
  - Some images name it differently (e.g. `waveshare35b`, `pitft35-resistive`, etc.). Use whatever exists.
  - For Raspberry Pi OS Bookworm/Trixie, Waveshare’s current guide for the **3.5inch RPi LCD (A)** installs the overlay by downloading a zip and copying `waveshare35a.dtbo` into the overlays folder, then using `dtoverlay=waveshare35a` in `config.txt`.
- If the screen is white/black or stays on HDMI:
  - Some SPI TFT setups require disabling full KMS. In `config.txt` try using FKMS:
    - Replace `dtoverlay=vc4-kms-v3d` with `dtoverlay=vc4-fkms-v3d`, reboot.
- If touch is rotated/wrong direction:
  - Adjust `swapxy=0/1` and try different `rotate=` values.
  - Calibration: install tools and calibrate (X11 easiest):
    - `sudo apt -y install xinput-calibrator`
    - run `xinput_calibrator` (if you’re on X11/Bullseye). For Wayland/Bookworm, calibration depends on the compositor; tell me your OS and I’ll give the exact method.
  - Waveshare’s Bookworm/Trixie touch guidance typically uses an Xorg calibration snippet matching `MatchProduct "ADS7846 Touchscreen"`.

Security note (USB safety):
- A **hardware USB write-blocker / forensic bridge** is the strongest protection against a compromised USB writing back to your system.
- Software can’t provide perfect write-blocking against kernel/driver exploits, but we can still reduce risk by mounting USB storage **read-only** and with `noexec,nodev,nosuid`.

## 1) Install Raspberry Pi OS
1. On your PC, install **Raspberry Pi Imager**.
2. Flash **Raspberry Pi OS (64-bit) with Desktop** to a microSD.
3. In Imager settings (gear icon):
   - Set hostname (e.g. `usb-sanitizer`)
   - Enable SSH
   - Set username/password
   - Configure Wi‑Fi (optional)
4. Boot the Pi.

## 2) First boot essentials
Open a terminal on the Pi and run:
```bash
sudo apt update
sudo apt -y upgrade
sudo apt -y install git python3-venv python3-pip python3-tk util-linux

# If pip needs to compile native wheels (common for yara-python on ARM):
sudo apt -y install build-essential python3-dev libyara-dev
```
Notes:
- `python3-tk` is required for the touchscreen kiosk UI (`pi/kiosk_ui.py`).
- `util-linux` provides `lsblk`, which the udev-triggered runner uses to find mountpoints.
- If `libyara-dev` is not available on your image, you can either install `yara` packages from apt, or temporarily remove `yara-python` from `requirements-pi.txt` (YARA scanning is optional in our pipeline).

## 3) Get your project onto the Pi
Option A (recommended): clone from git
```bash
cd /home/pi
git clone <YOUR_REPO_URL> Ai
cd Ai
```

Option B: copy your existing folder from Windows
- Use **SCP / WinSCP** to copy the entire `Ai/` folder to `/home/pi/Ai`.

## 4) Create venv and install Pi requirements
```bash
cd /home/pi/Ai
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-pi.txt
```
If you also want to use the PyTorch MalConv pipeline, install its deps separately (Torch on Pi can be heavy).

## 5) (Recommended) Test the UI manually first
In one terminal:
```bash
cd /home/pi/Ai
source .venv/bin/activate
python -u hardware/pi/kiosk_ui.py
```
- Fullscreen opens.
- Press `Esc` to exit.

## 6) Test scanning manually (no udev yet)
If your USB auto-mounts under `/media/pi/<NAME>`:
```bash
cd /home/pi/Ai
source .venv/bin/activate
python -u hardware/pi/pi_usb_sanitizer.py --scan /media/pi/<USB_NAME> --quarantine /home/pi/quarantine
```

Important: `pi_usb_sanitizer.py` **moves** quarantined files off the USB into the quarantine folder.

### Fixing false positives on packed installers (recommended)
Packed/signed installers (Chrome/VLC/Zoom) often look malware-like to ML models.

Fast, safe deployment fix: create a **SHA256 allowlist** of installers you trust:

```bash
cd /home/pi/Ai
source .venv/bin/activate

# Example: build allowlist from a folder you control
python -u hardware/pi/make_allowlist.py --folder /home/pi/trusted_installers --out /tmp/allowlist.sha256

# Store the allowlist on the Pi's root filesystem (NOT on the USB), and lock permissions
sudo mkdir -p /etc/usb-sanitizer
sudo mv /tmp/allowlist.sha256 /etc/usb-sanitizer/allowlist.sha256
sudo chown root:root /etc/usb-sanitizer/allowlist.sha256
sudo chmod 0644 /etc/usb-sanitizer/allowlist.sha256

# Then scan using the allowlist
python -u hardware/pi/pi_usb_sanitizer.py --scan /media/pi/<USB_NAME> --quarantine /home/pi/quarantine --allowlist /etc/usb-sanitizer/allowlist.sha256
```

Optional policy knob (use carefully):
- `--signed-entropy-override` lowers packer-driven entropy alerts for PEs that have an Authenticode signature blob present (presence-only; not trust validation).

## 7) Enable “run immediately on USB insert” (udev + systemd)
### 7.1 Copy the udev rule
```bash
sudo cp /home/pi/Ai/hardware/pi/udev/99-usb-sanitize.rules /etc/udev/rules.d/99-usb-sanitize.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 7.2 Copy the systemd unit template
```bash
sudo cp /home/pi/Ai/hardware/pi/systemd/usb-sanitize@.service /etc/systemd/system/usb-sanitize@.service
sudo systemctl daemon-reload
```

### 7.3 Make the runner executable
```bash
sudo chmod +x /home/pi/Ai/hardware/pi/usb-sanitize-run.sh
```

### 7.4 Plug in a USB drive to test
- Insert the USB.
- Check logs:
```bash
sudo journalctl -u "usb-sanitize@*" -f
```
- The scan runner writes files under:
  - `/home/pi/Ai/outputs/pi/logs/status.json`
  - `/home/pi/Ai/outputs/pi/logs/usb_sanitizer.log`

Notes:
- Some USB drives show up as a partition like `/dev/sda1`.
- Some show up as a whole-disk filesystem like `/dev/sda` (no `/dev/sda1`). In that case the systemd unit instance will be `usb-sanitize@sda.service`.
- If you’re unsure of the device name, run:
  - `lsblk -o NAME,SIZE,FSTYPE,LABEL,MOUNTPOINT`

To use an allowlist with udev auto-run, edit the service command in:
- `/etc/systemd/system/usb-sanitize@.service`

and add:
- `--allowlist /home/pi/Ai/allowlist.sha256`

Simpler (recommended): edit the config file used by the udev runner:
- `/home/pi/Ai/hardware/pi/usb-sanitizer.conf`

Example content:
```text
--allowlist /etc/usb-sanitizer/allowlist.sha256
```

If your repo path is not `/home/pi/Ai`, edit:
- `/etc/systemd/system/usb-sanitize@.service` (`WorkingDirectory` + `ExecStart`)

### Optional: read-only mount (software write-blocking approximation)
If you are running headless (no desktop automount) or you want a safer mount, the runner can attempt to mount the partition itself using:

- `ro,nosuid,nodev,noexec`

To enable it, add this line to:
- `/home/pi/Ai/hardware/pi/usb-sanitizer.conf`

```text
--try-ro-mount
```

Notes:
- This is not a substitute for a hardware write blocker, but it prevents execution from the USB mount and reduces accidental writes.

## 8) Do you need anything else?
If you want the UI to auto-start on boot, the simplest approach is **Desktop autostart** (starts when the desktop user logs in):

```bash
mkdir -p ~/.config/autostart
cp /home/pi/Ai/hardware/pi/autostart/usb-sanitizer.desktop ~/.config/autostart/usb-sanitizer.desktop
```

Reboot, and the kiosk should launch fullscreen. Press `Esc` to exit.

### If you are using an SPI TFT framebuffer (/dev/fb1)
On many 3.5" GPIO SPI TFTs the display appears as a **framebuffer** device (often `/dev/fb1`). In that setup, a Tkinter UI needs a local GUI server.

If you are SSH'ing in and see errors like:
- `_tkinter.TclError: couldn't connect to display ":0"`

it usually means **no X server is running**, or Xorg can't start due to VT permissions.

#### Known issue: Xorg can't open a VT (Permission denied)
On some images, console devices like `/dev/tty2` can end up with permissions `root:root 0600`, which breaks rootless Xorg with:
- `xf86OpenConsole: Cannot open virtual console 2 (Permission denied)`

Quick (temporary) test fix:
```bash
sudo chgrp tty /dev/tty2
sudo chmod 0620 /dev/tty2
```

Persistent fix (recommended): install a udev rule that restores standard permissions:
```bash
sudo cp /home/pi/Ai/hardware/pi/udev/99-tty-perms.rules /etc/udev/rules.d/99-tty-perms.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Reboot once to be safe:
```bash
sudo reboot
```

#### Start the kiosk on the SPI TFT without relying on LightDM
If you want a reliable kiosk on the TFT even when LightDM/desktop isn't working, you can run **Xorg directly on `/dev/fb1`** (fbdev) and launch the kiosk.

1) Ensure packages exist:
```bash
sudo apt update
sudo apt -y install xserver-xorg xserver-xorg-video-fbdev xinit
```

2) Make the runner executable:
```bash
sudo chmod +x /home/pi/Ai/hardware/pi/kiosk_fb1_run.sh
```

3) Install + enable the service:
```bash
sudo cp /home/pi/Ai/hardware/pi/systemd/kiosk-fb1.service /etc/systemd/system/kiosk-fb1.service
sudo systemctl daemon-reload
sudo systemctl enable kiosk-fb1.service
sudo systemctl start kiosk-fb1.service
```

Logs:
```bash
sudo journalctl -u kiosk-fb1 -f
tail -n 200 /home/pi/.local/share/xorg/Xorg.0.log
```

Notes:
- This service assumes your repo is at `/home/pi/Ai` and that your TFT is `/dev/fb1`.
- If your repo path differs, edit `/etc/systemd/system/kiosk-fb1.service`.

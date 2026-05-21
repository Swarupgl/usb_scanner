from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import pyzipper  # Essential for AES-256

# --- CONFIGURATION ---
# This helper is intentionally *disabled by default*.
# If you choose to use it, point SOURCE_DIR at a folder of passworded ZIPs you
# are authorized to analyze, and keep DEST_DIR in a controlled location.

REPO_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIR = os.environ.get("USB_SANITIZER_SOURCE_DIR", "")
DEST_DIR = os.environ.get(
    "USB_SANITIZER_DEST_DIR",
    str(REPO_ROOT / "datasets" / "local" / "malicious"),
)

PASSWORD = os.environ.get("USB_SANITIZER_ZIP_PASSWORD", "infected").encode("utf-8")
ALLOWED_EXTENSIONS = (".exe", ".dll")

def get_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def extract_and_filter():
    if not SOURCE_DIR:
        print(
            "SOURCE_DIR is not set. To enable, set env var USB_SANITIZER_SOURCE_DIR to a folder containing ZIPs."
        )
        return

    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

    # Load existing hashes to skip what you already have
    existing_hashes = set()
    for f in os.listdir(DEST_DIR):
        f_path = os.path.join(DEST_DIR, f)
        if os.path.isfile(f_path):
            try:
                existing_hashes.add(get_sha256(f_path))
            except:
                continue

    count = 0
    print("Starting deep extraction of ZIP members...")

    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.endswith(".zip"):
                zip_path = os.path.join(root, file)
                try:
                    # We use AESZipFile specifically for theZoo's AES encryption
                    with pyzipper.AESZipFile(zip_path, 'r', encryption=pyzipper.WZ_AES) as zf:
                        zf.setpassword(PASSWORD)
                        for member in zf.namelist():
                            if member.lower().endswith(ALLOWED_EXTENSIONS):
                                # Extract to the current folder
                                zf.extract(member, path=DEST_DIR)
                                old_path = os.path.join(DEST_DIR, member)
                                
                                # De-duplication check
                                file_hash = get_sha256(old_path)
                                if file_hash in existing_hashes:
                                    os.remove(old_path)
                                else:
                                    existing_hashes.add(file_hash)
                                    new_name = f"{file_hash[:10]}_{os.path.basename(member)}"
                                    new_path = os.path.join(DEST_DIR, new_name)
                                    shutil.move(old_path, new_path)
                                    print(f"Added Unique: {new_name}")
                                    count += 1
                except Exception as e:
                    # If even pyzipper fails, it might be a corrupted download or different format
                    print(f"Skipping {file}: {e}")

    print(f"\n--- SUCCESS ---")
    print(f"Total NEW UNIQUE binaries added: {count}")

if __name__ == "__main__":
    extract_and_filter()
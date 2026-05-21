from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import struct
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Optional

import math


def load_allowlist(path: str) -> set[str]:
    """Load a SHA256 allowlist (one hex digest per line; comments allowed with '#')."""
    allowed: set[str] = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            token = line.split()[0].strip().lower()
            if len(token) == 64 and all(c in "0123456789abcdef" for c in token):
                allowed.add(token)
    return allowed


def pe_has_authenticode_signature(path: str, max_read: int = 4_000_000) -> bool:
    """Best-effort check: True if the PE has a non-empty SECURITY data directory.

    This detects *presence* of an Authenticode signature blob; it does not validate trust.
    """
    try:
        with open(path, "rb") as f:
            data = f.read(max_read)
    except Exception:
        return False

    if len(data) < 0x100 or not data.startswith(b"MZ"):
        return False
    try:
        pe_off = struct.unpack_from("<I", data, 0x3C)[0]
    except Exception:
        return False
    if pe_off + 4 + 20 > len(data) or data[pe_off : pe_off + 4] != b"PE\x00\x00":
        return False

    coff_off = pe_off + 4
    try:
        opt_size = struct.unpack_from("<H", data, coff_off + 16)[0]
    except Exception:
        return False
    opt_off = coff_off + 20
    if opt_off + opt_size > len(data) or opt_size < 2:
        return False

    try:
        magic = struct.unpack_from("<H", data, opt_off)[0]
    except Exception:
        return False

    # Offsets per PE/COFF for NumberOfRvaAndSizes.
    if magic == 0x10B:  # PE32
        num_rva_off = opt_off + 92
        dd_off = opt_off + 96
    elif magic == 0x20B:  # PE32+
        num_rva_off = opt_off + 108
        dd_off = opt_off + 112
    else:
        return False

    if num_rva_off + 4 > len(data):
        return False
    try:
        num_dirs = struct.unpack_from("<I", data, num_rva_off)[0]
    except Exception:
        return False
    if num_dirs < 5:
        return False

    sec_index = 4  # IMAGE_DIRECTORY_ENTRY_SECURITY
    entry_off = dd_off + sec_index * 8
    if entry_off + 8 > len(data):
        return False
    try:
        sec_rva_or_off, sec_size = struct.unpack_from("<II", data, entry_off)
    except Exception:
        return False

    # For SECURITY: RVA field is actually a file offset.
    return int(sec_size) > 0 and int(sec_rva_or_off) > 0


# ---------------------------
# Stage 0 (optional): Signatures (YARA)
# ---------------------------


@dataclass(frozen=True)
class Stage0Result:
    enabled: bool
    matched: bool
    rule_names: list[str]
    reason: str


class YaraEngine:
    def __init__(self, rules_path: str):
        try:
            import yara  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "yara-python is not installed. Install on Raspberry Pi with: pip install yara-python"
            ) from exc

        self._yara = yara
        self.rules_path = rules_path
        self.rules = yara.compile(filepath=rules_path)

    def match(self, file_path: str, max_rules: int = 10) -> Stage0Result:
        try:
            matches = self.rules.match(file_path)
            names = [m.rule for m in matches][:max_rules]
            return Stage0Result(
                enabled=True,
                matched=bool(names),
                rule_names=names,
                reason="matched" if names else "no match",
            )
        except Exception as exc:
            return Stage0Result(enabled=True, matched=False, rule_names=[], reason=f"yara error: {exc}")


# ---------------------------
# Stage 1: Magic-number checks
# ---------------------------

MAGIC = {
    "pe": b"MZ",
    "pdf": b"%PDF-",
    "zip": b"PK\x03\x04",
    "png": b"\x89PNG\r\n\x1a\n",
    "jpg": b"\xff\xd8\xff",
    "gif": b"GIF8",
    "7z": b"7z\xbc\xaf\x27\x1c",
    "rar": b"Rar!\x1a\x07",
    "elf": b"\x7fELF",
}

EXT_EXPECTED_KIND = {
    ".exe": "pe",
    ".dll": "pe",
    ".sys": "pe",
    ".com": "pe",
    ".pdf": "pdf",
    ".zip": "zip",
    ".docx": "zip",
    ".xlsx": "zip",
    ".pptx": "zip",
    ".png": "png",
    ".jpg": "jpg",
    ".jpeg": "jpg",
    ".gif": "gif",
    ".7z": "7z",
    ".rar": "rar",
    ".so": "elf",
}

TEXT_EXTS = {".txt", ".md", ".csv", ".log", ".json", ".xml"}


def read_head(path: str, n: int = 4096) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)


def detect_kind_by_magic(head: bytes) -> Optional[str]:
    for kind, sig in MAGIC.items():
        if head.startswith(sig):
            return kind
    return None


def is_probably_text(head: bytes) -> bool:
    if not head:
        return True
    # Reject if it has NUL bytes early.
    if b"\x00" in head:
        return False
    # ASCII-ish heuristic
    printable = sum(1 for b in head if b in b"\t\n\r" or 32 <= b <= 126)
    return (printable / max(len(head), 1)) > 0.9


@dataclass(frozen=True)
class Stage1Result:
    expected_kind: Optional[str]
    actual_kind: Optional[str]
    is_text_like: bool
    mismatched: bool
    reason: str


def stage1_structural_integrity(path: str) -> Stage1Result:
    ext = os.path.splitext(path)[1].lower()
    expected = EXT_EXPECTED_KIND.get(ext)

    head = read_head(path, n=4096)
    actual = detect_kind_by_magic(head)
    text_like = is_probably_text(head)

    # Rules:
    # - If extension has an expected magic kind, enforce it.
    # - Special strong rule: if it *looks executable* (MZ) but extension is not executable, it's mismatched.
    mismatched = False
    reason = ""

    # Strong rule for text-ish extensions: they should be text-like and not match a known binary magic.
    if ext in TEXT_EXTS:
        if actual is not None:
            mismatched = True
            reason = f"text extension {ext} but magic indicates {actual}"
        elif not text_like:
            mismatched = True
            reason = f"text extension {ext} but content not text-like"
        else:
            mismatched = False
            reason = "text-like content"

    elif expected is not None:
        if actual != expected:
            # If actual is unknown but file is text-like and extension is a text extension, allow.
            mismatched = True
            reason = f"expected {expected}, got {actual or 'unknown'}"
    else:
        # Extension not in our table.
        if actual == "pe" and ext not in {".exe", ".dll", ".sys", ".com"}:
            mismatched = True
            reason = f"executable content (MZ) with non-executable extension {ext or '(none)'}"
        else:
            reason = "no expected kind for extension"

    return Stage1Result(
        expected_kind=expected,
        actual_kind=actual,
        is_text_like=text_like,
        mismatched=mismatched,
        reason=reason,
    )


# ---------------------------
# Stage 2: Heuristics (entropy + PE imports)
# ---------------------------

SUSPICIOUS_IMPORTS = {
    "internetopen",
    "createremotethread",
    "writeprocessmemory",
}


def shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    ent = 0.0
    n = len(data)
    for c in counts:
        if c:
            p = c / n
            ent -= p * math.log2(p)
    return ent


@dataclass(frozen=True)
class PEImportResult:
    ok: bool
    reason: str
    imports: set[str]


def _u16(b: bytes, off: int) -> int:
    return struct.unpack_from("<H", b, off)[0]


def _u32(b: bytes, off: int) -> int:
    return struct.unpack_from("<I", b, off)[0]


def _read_c_string(data: bytes, off: int, limit: int = 512) -> str:
    end = data.find(b"\x00", off, off + limit)
    if end == -1:
        end = min(len(data), off + limit)
    try:
        return data[off:end].decode("ascii", errors="ignore")
    except Exception:
        return ""


def _rva_to_offset(rva: int, sections: list[tuple[int, int, int]]) -> Optional[int]:
    # sections: (virt_addr, virt_size, raw_ptr)
    for va, vsz, raw_ptr in sections:
        if va <= rva < va + max(vsz, 1):
            return raw_ptr + (rva - va)
    return None


def extract_pe_imports(path: str, max_scan: int = 2_000_000) -> PEImportResult:
    """Minimal PE import parser.

    Reads up to max_scan bytes. Extracts imported function names where possible.
    Returns ok=False if the file isn't a parseable PE.
    """
    try:
        with open(path, "rb") as f:
            data = f.read(max_scan)
    except Exception as exc:
        return PEImportResult(ok=False, reason=f"read error: {exc}", imports=set())

    if len(data) < 0x100 or not data.startswith(b"MZ"):
        return PEImportResult(ok=False, reason="missing MZ", imports=set())

    pe_off = _u32(data, 0x3C)
    if pe_off + 4 > len(data) or data[pe_off : pe_off + 4] != b"PE\x00\x00":
        return PEImportResult(ok=False, reason="missing PE signature", imports=set())

    coff_off = pe_off + 4
    if coff_off + 20 > len(data):
        return PEImportResult(ok=False, reason="truncated COFF header", imports=set())

    num_sections = _u16(data, coff_off + 2)
    opt_size = _u16(data, coff_off + 16)
    opt_off = coff_off + 20
    if opt_off + opt_size > len(data):
        return PEImportResult(ok=False, reason="truncated optional header", imports=set())

    magic = _u16(data, opt_off)
    if magic == 0x10B:
        data_dir_off = opt_off + 96
    elif magic == 0x20B:
        data_dir_off = opt_off + 112
    else:
        return PEImportResult(ok=False, reason=f"unknown optional header magic: {hex(magic)}", imports=set())

    if data_dir_off + 8 * 2 > len(data):
        return PEImportResult(ok=False, reason="truncated data directory", imports=set())

    # DataDirectory[1] = Import Table
    import_rva = _u32(data, data_dir_off + 8 * 1)
    import_size = _u32(data, data_dir_off + 8 * 1 + 4)
    if import_rva == 0 or import_size == 0:
        return PEImportResult(ok=True, reason="no import table", imports=set())

    # Section table
    sec_off = opt_off + opt_size
    sec_size = 40
    if sec_off + num_sections * sec_size > len(data):
        return PEImportResult(ok=False, reason="truncated section table", imports=set())

    sections: list[tuple[int, int, int]] = []
    for i in range(num_sections):
        off = sec_off + i * sec_size
        virt_size = _u32(data, off + 8)
        virt_addr = _u32(data, off + 12)
        raw_ptr = _u32(data, off + 20)
        sections.append((virt_addr, virt_size, raw_ptr))

    imp_off = _rva_to_offset(import_rva, sections)
    if imp_off is None or imp_off + 20 > len(data):
        return PEImportResult(ok=False, reason="import rva not mapped", imports=set())

    imports: set[str] = set()

    # IMAGE_IMPORT_DESCRIPTOR is 20 bytes; array terminated by all zeros.
    for j in range(0, min(import_size, 4096), 20):
        d_off = imp_off + j
        if d_off + 20 > len(data):
            break
        orig_first_thunk = _u32(data, d_off + 0)
        name_rva = _u32(data, d_off + 12)
        first_thunk = _u32(data, d_off + 16)

        if orig_first_thunk == 0 and name_rva == 0 and first_thunk == 0:
            break

        thunk_rva = orig_first_thunk or first_thunk
        thunk_off = _rva_to_offset(thunk_rva, sections)
        if thunk_off is None:
            continue

        # Iterate thunks (32-bit or 64-bit)
        is_pe64 = magic == 0x20B
        step = 8 if is_pe64 else 4
        max_thunks = 2048

        for k in range(max_thunks):
            t_off = thunk_off + k * step
            if t_off + step > len(data):
                break
            if is_pe64:
                val = struct.unpack_from("<Q", data, t_off)[0]
                if val == 0:
                    break
                is_ordinal = (val >> 63) & 1
                if is_ordinal:
                    continue
                hint_name_rva = val & 0x7FFF_FFFF_FFFF_FFFF
            else:
                val = _u32(data, t_off)
                if val == 0:
                    break
                is_ordinal = (val >> 31) & 1
                if is_ordinal:
                    continue
                hint_name_rva = val & 0x7FFF_FFFF

            hn_off = _rva_to_offset(int(hint_name_rva), sections)
            if hn_off is None or hn_off + 2 > len(data):
                continue

            func_name = _read_c_string(data, hn_off + 2)
            if func_name:
                imports.add(func_name)

    return PEImportResult(ok=True, reason="ok", imports=imports)


def extract_pe_imports_pefile(path: str) -> PEImportResult:
    """Extract imports using pefile (more robust than the minimal parser).

    This is optional and used when pefile is installed.
    """
    try:
        import pefile  # type: ignore
    except Exception as exc:  # pragma: no cover
        return PEImportResult(ok=False, reason=f"pefile not installed: {exc}", imports=set())

    try:
        pe = pefile.PE(path, fast_load=True)
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])

        imports: set[str] = set()
        if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                for imp in entry.imports:
                    if not imp.name:
                        continue
                    try:
                        imports.add(imp.name.decode("ascii", errors="ignore"))
                    except Exception:
                        continue

        return PEImportResult(ok=True, reason="ok", imports=imports)
    except Exception as exc:
        return PEImportResult(ok=False, reason=f"pefile parse error: {exc}", imports=set())


@dataclass(frozen=True)
class Stage2Result:
    entropy: float
    entropy_high: bool
    pe_imports_ok: bool
    suspicious_imports: set[str]


def stage2_header_heuristics(path: str, kind: Optional[str], entropy_threshold: float) -> Stage2Result:
    # Entropy on a limited prefix is usually enough for "packed" signal.
    with open(path, "rb") as f:
        sample = f.read(256 * 1024)  # 256KB sample for speed
    ent = shannon_entropy(sample)
    ent_high = ent > entropy_threshold

    suspicious: set[str] = set()
    pe_ok = False

    if kind == "pe":
        # Prefer pefile if available; fall back to minimal parser.
        imp = extract_pe_imports_pefile(path)
        if not imp.ok:
            imp = extract_pe_imports(path)

        pe_ok = imp.ok
        for fn in imp.imports:
            if fn.lower() in SUSPICIOUS_IMPORTS:
                suspicious.add(fn)

    return Stage2Result(entropy=ent, entropy_high=ent_high, pe_imports_ok=pe_ok, suspicious_imports=suspicious)


# ---------------------------
# Stage 3: Neural Analysis (ONNX)
# ---------------------------


class MalConvOnnx:
    def __init__(self, onnx_path: str):
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "onnxruntime is not installed. On Raspberry Pi install: pip install onnxruntime"
            ) from exc

        self._ort = ort
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def predict(self, x_tokens):
        # x_tokens must be shape (1, max_len) int64
        out = self.sess.run([self.output_name], {self.input_name: x_tokens})
        y = out[0]
        # y is usually shape (1,1)
        return float(y.reshape(-1)[0])


def file_to_tokens_int64(path: str, max_len: int) -> Optional["object"]:
    try:
        with open(path, "rb") as f:
            raw = f.read(max_len)
    except Exception:
        return None

    import numpy as np

    arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int64, copy=False)
    if arr.size < max_len:
        pad = np.full((max_len - arr.size,), 256, dtype=np.int64)
        arr = np.concatenate([arr, pad], axis=0)
    else:
        arr = arr[:max_len]

    return arr.reshape(1, -1)


# ---------------------------
# Optional cloud intelligence: VirusTotal hash lookup
# ---------------------------


@dataclass(frozen=True)
class VTResult:
    enabled: bool
    ok: bool
    malicious: int
    suspicious: int
    undetected: int
    harmless: int
    timeout: bool
    reason: str


def vt_lookup_sha256(sha256_hex: str, api_key: str, timeout_sec: float = 10.0) -> VTResult:
    try:
        import requests  # type: ignore
    except Exception as exc:  # pragma: no cover
        return VTResult(
            enabled=True,
            ok=False,
            malicious=0,
            suspicious=0,
            undetected=0,
            harmless=0,
            timeout=False,
            reason=f"requests not installed: {exc}",
        )

    url = f"https://www.virustotal.com/api/v3/files/{sha256_hex}"
    headers = {"x-apikey": api_key}

    try:
        resp = requests.get(url, headers=headers, timeout=timeout_sec)
    except requests.Timeout:
        return VTResult(
            enabled=True,
            ok=False,
            malicious=0,
            suspicious=0,
            undetected=0,
            harmless=0,
            timeout=True,
            reason="timeout",
        )
    except Exception as exc:
        return VTResult(
            enabled=True,
            ok=False,
            malicious=0,
            suspicious=0,
            undetected=0,
            harmless=0,
            timeout=False,
            reason=f"request error: {exc}",
        )

    if resp.status_code == 404:
        return VTResult(
            enabled=True,
            ok=True,
            malicious=0,
            suspicious=0,
            undetected=0,
            harmless=0,
            timeout=False,
            reason="not found",
        )

    if resp.status_code != 200:
        return VTResult(
            enabled=True,
            ok=False,
            malicious=0,
            suspicious=0,
            undetected=0,
            harmless=0,
            timeout=False,
            reason=f"http {resp.status_code}",
        )

    try:
        js = resp.json()
        stats = js["data"]["attributes"]["last_analysis_stats"]
        return VTResult(
            enabled=True,
            ok=True,
            malicious=int(stats.get("malicious", 0)),
            suspicious=int(stats.get("suspicious", 0)),
            undetected=int(stats.get("undetected", 0)),
            harmless=int(stats.get("harmless", 0)),
            timeout=False,
            reason="ok",
        )
    except Exception as exc:
        return VTResult(
            enabled=True,
            ok=False,
            malicious=0,
            suspicious=0,
            undetected=0,
            harmless=0,
            timeout=False,
            reason=f"json parse error: {exc}",
        )


# ---------------------------
# Unified decision + quarantine
# ---------------------------


@dataclass(frozen=True)
class ScanResult:
    path: str
    stage0: Optional[Stage0Result]
    stage1: Stage1Result
    stage2: Stage2Result
    stage3_score: Optional[float]
    vt: Optional[VTResult]
    risk_score: float
    action: str
    allowlisted: bool = False
    sha256: Optional[str] = None
    signed_present: Optional[bool] = None


@dataclass(frozen=True)
class QuarantineResult:
    dest: str
    method: str  # "moved" | "copied"


def sha256_file(path: str, max_bytes: Optional[int] = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if max_bytes is None:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        else:
            remaining = max_bytes
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                h.update(chunk)
                remaining -= len(chunk)
    return h.hexdigest()


def quarantine_file(path: str, quarantine_dir: str) -> QuarantineResult:
    os.makedirs(quarantine_dir, exist_ok=True)
    base = os.path.basename(path)
    digest = sha256_file(path)[:12]
    ts = int(time.time())
    dest = os.path.join(quarantine_dir, f"{ts}_{digest}_{base}")
    try:
        shutil.move(path, dest)
        return QuarantineResult(dest=dest, method="moved")
    except OSError as exc:
        # Common case on hardened deployments: USB mounted read-only (EROFS=30).
        # In that case, we still want to preserve the suspect file by copying it
        # off the USB into quarantine, even if we cannot delete/rename the original.
        if getattr(exc, "errno", None) in {30, 13, 1}:  # EROFS, EACCES, EPERM
            shutil.copy2(path, dest)
            return QuarantineResult(dest=dest, method="copied")
        raise


def compute_risk(stage1: Stage1Result, stage2: Stage2Result, stage3_score: Optional[float]) -> float:
    # Conservative aggregation: treat stage1 mismatch as hard 1.0 risk.
    if stage1.mismatched:
        return 1.0

    s2 = 0.0
    if stage2.entropy_high:
        s2 = max(s2, 0.8)
    if stage2.suspicious_imports:
        s2 = max(s2, 0.9)

    s3 = float(stage3_score) if stage3_score is not None else 0.0

    # "Unified" risk score: max across stages.
    return max(s2, s3)


def scan_file(
    path: str,
    quarantine_dir: str,
    entropy_threshold: float,
    risk_threshold: float,
    yara_engine: Optional[YaraEngine],
    onnx: Optional[MalConvOnnx],
    onnx_max_len: int,
    allow_clean_exit: bool,
    vt_api_key: Optional[str],
    vt_min_score: float,
    vt_max_score: float,
    vt_malicious_threshold: int,
    allowlist_sha256: Optional[set[str]] = None,
    signed_entropy_override: bool = False,
    signed_quarantine_threshold: float = 0.9,
) -> ScanResult:
    # Stage -1: allowlist (fast deterministic fix for known-good installers)
    digest: Optional[str] = None
    if allowlist_sha256 is not None:
        try:
            digest = sha256_file(path)
        except Exception:
            digest = None
        if digest and digest.lower() in allowlist_sha256:
            s1 = stage1_structural_integrity(path)
            s2 = Stage2Result(entropy=0.0, entropy_high=False, pe_imports_ok=False, suspicious_imports=set())
            return ScanResult(
                path=path,
                stage0=None,
                stage1=s1,
                stage2=s2,
                stage3_score=None,
                vt=None,
                risk_score=0.0,
                action="clean (allowlisted)",
                allowlisted=True,
                sha256=digest,
                signed_present=None,
            )

    s0: Optional[Stage0Result] = None
    s1 = stage1_structural_integrity(path)

    # Stage 1 hard-fail: mismatch -> quarantine immediately.
    if s1.mismatched:
        q = quarantine_file(path, quarantine_dir)
        s2 = Stage2Result(entropy=0.0, entropy_high=False, pe_imports_ok=False, suspicious_imports=set())
        return ScanResult(
            path=path,
            stage0=None,
            stage1=s1,
            stage2=s2,
            stage3_score=None,
            vt=None,
            risk_score=1.0,
            action=("quarantined" if q.method == "moved" else "quarantined_copy") + " (mismatched)",
        )

    # Stage 0: YARA signatures (fast known-bad match)
    if yara_engine is not None:
        s0 = yara_engine.match(path)
        if s0.matched:
            q = quarantine_file(path, quarantine_dir)
            s2 = Stage2Result(entropy=0.0, entropy_high=False, pe_imports_ok=False, suspicious_imports=set())
            return ScanResult(
                path=path,
                stage0=s0,
                stage1=s1,
                stage2=s2,
                stage3_score=None,
                vt=None,
                risk_score=1.0,
                action=("quarantined" if q.method == "moved" else "quarantined_copy") + " (yara)",
            )

    s2 = stage2_header_heuristics(path, kind=s1.actual_kind, entropy_threshold=entropy_threshold)

    signed_present: Optional[bool] = None
    if signed_entropy_override and s1.actual_kind == "pe":
        signed_present = pe_has_authenticode_signature(path)
        if signed_present:
            # Reduce "packed==malware" false positives for signed installers.
            s2 = Stage2Result(
                entropy=s2.entropy,
                entropy_high=False,
                pe_imports_ok=s2.pe_imports_ok,
                suspicious_imports=s2.suspicious_imports,
            )

    # Clean exit for efficiency:
    # - If the file is text-like and not high-entropy, skip neural analysis.
    # - If it's a PE and has no suspicious imports and entropy is low, you may also skip.
    if allow_clean_exit:
        if s1.is_text_like and not s2.entropy_high and s1.actual_kind != "pe":
            risk = compute_risk(s1, s2, stage3_score=None)
            return ScanResult(
                path=path,
                stage0=s0,
                stage1=s1,
                stage2=s2,
                stage3_score=None,
                vt=None,
                risk_score=risk,
                action="clean",
            )
        if s1.actual_kind == "pe" and (not s2.entropy_high) and (not s2.suspicious_imports):
            risk = compute_risk(s1, s2, stage3_score=None)
            return ScanResult(
                path=path,
                stage0=s0,
                stage1=s1,
                stage2=s2,
                stage3_score=None,
                vt=None,
                risk_score=risk,
                action="clean",
            )

    stage3_score: Optional[float] = None
    if onnx is not None:
        x = file_to_tokens_int64(path, max_len=onnx_max_len)
        if x is not None:
            stage3_score = onnx.predict(x)

    risk = compute_risk(s1, s2, stage3_score)

    # Optional cloud intelligence: only if score is "uncertain".
    vt: Optional[VTResult] = None
    if vt_api_key and (vt_min_score <= risk <= vt_max_score):
        digest_vt = digest or sha256_file(path)
        vt = vt_lookup_sha256(digest_vt, api_key=vt_api_key)
        if vt.ok and vt.malicious >= vt_malicious_threshold:
            q = quarantine_file(path, quarantine_dir)
            return ScanResult(
                path=path,
                stage0=s0,
                stage1=s1,
                stage2=s2,
                stage3_score=stage3_score,
                vt=vt,
                risk_score=1.0,
                action=("quarantined" if q.method == "moved" else "quarantined_copy") + " (virustotal)",
                allowlisted=False,
                sha256=digest_vt,
                signed_present=signed_present,
            )

    # Optional: be more conservative quarantining signed PEs.
    effective_threshold = risk_threshold
    if signed_present and s1.actual_kind == "pe":
        effective_threshold = max(effective_threshold, float(signed_quarantine_threshold))

    if risk >= effective_threshold:
        q = quarantine_file(path, quarantine_dir)
        return ScanResult(
            path=path,
            stage0=s0,
            stage1=s1,
            stage2=s2,
            stage3_score=stage3_score,
            vt=vt,
            risk_score=risk,
            action="quarantined" if q.method == "moved" else "quarantined_copy",
            allowlisted=False,
            sha256=digest,
            signed_present=signed_present,
        )

    return ScanResult(
        path=path,
        stage0=s0,
        stage1=s1,
        stage2=s2,
        stage3_score=stage3_score,
        vt=vt,
        risk_score=risk,
        action="clean",
        allowlisted=False,
        sha256=digest,
        signed_present=signed_present,
    )


def iter_files(root: str) -> Iterable[str]:
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            yield os.path.join(dirpath, name)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Raspberry Pi USB sanitizer: 3-stage detection pipeline")
    p.add_argument("--scan", required=True, help="Path to scan (e.g., /media/pi/USB)")
    p.add_argument("--quarantine", default=str(Path("outputs") / "quarantine"), help="Quarantine directory")
    p.add_argument("--entropy-threshold", type=float, default=7.5)
    p.add_argument("--risk-threshold", type=float, default=0.5)
    p.add_argument("--yara-rules", default=None, help="Path to YARA rules file (.yar/.yara) (optional)")
    p.add_argument("--onnx", default=None, help="Path to MalConv ONNX model (optional)")
    p.add_argument("--onnx-max-len", type=int, default=1048576, help="Bytes to feed to ONNX model")
    p.add_argument("--no-clean-exit", action="store_true", help="Always run Stage 3 when ONNX is set")
    p.add_argument(
        "--vt",
        action="store_true",
        help="Enable VirusTotal hash lookup (only sends SHA-256, only used for uncertain scores)",
    )
    p.add_argument(
        "--vt-api-key",
        default=None,
        help="VirusTotal API key (if omitted, uses VT_API_KEY env var)",
    )
    p.add_argument("--vt-min-score", type=float, default=0.4, help="Min score to trigger VT lookup")
    p.add_argument("--vt-max-score", type=float, default=0.7, help="Max score to trigger VT lookup")
    p.add_argument(
        "--vt-malicious-threshold",
        type=int,
        default=1,
        help="Quarantine if VT malicious count is >= this value",
    )
    p.add_argument(
        "--allowlist",
        default=None,
        help="Optional SHA256 allowlist file (one digest per line). Best immediate fix for packed installer false positives.",
    )
    p.add_argument(
        "--signed-entropy-override",
        action="store_true",
        help="If a PE has an Authenticode signature blob present, do not treat high entropy as suspicious (presence-only; not trust validation).",
    )
    p.add_argument(
        "--signed-quarantine-threshold",
        type=float,
        default=0.9,
        help="If signed-entropy-override is enabled and signature is present, require risk>=this threshold to quarantine.",
    )
    args = p.parse_args(argv)
    allowlist_sha256: Optional[set[str]] = None
    if args.allowlist:
        allowlist_sha256 = load_allowlist(args.allowlist)
        print(f"Allowlist: {os.path.abspath(args.allowlist)} (entries={len(allowlist_sha256)})")
    print(f"Signed entropy override: {'enabled' if args.signed_entropy_override else '(disabled)'}")
    if args.signed_entropy_override:
        print(f"Signed quarantine threshold: {args.signed_quarantine_threshold}")

    yara_engine = None
    if args.yara_rules:
        yara_engine = YaraEngine(args.yara_rules)

    onnx = None
    if args.onnx:
        onnx = MalConvOnnx(args.onnx)

    vt_api_key = None
    if args.vt:
        vt_api_key = args.vt_api_key or os.environ.get("VT_API_KEY")
        if not vt_api_key:
            print("Warning: --vt enabled but no API key found (set VT_API_KEY or pass --vt-api-key).")

    quarantine_dir = os.path.abspath(args.quarantine)
    scan_root = os.path.abspath(args.scan)

    if not os.path.isdir(scan_root):
        print(f"[error] scan path does not exist or is not a directory: {scan_root}")
        return 2

    try:
        os.makedirs(quarantine_dir, exist_ok=True)
    except Exception as exc:
        print(f"[error] could not create quarantine dir {quarantine_dir}: {exc}")
        return 2

    print(f"Scanning: {scan_root}")
    print(f"Quarantine: {quarantine_dir}")
    print(f"Stage2 entropy threshold: {args.entropy_threshold}")
    print(f"Risk threshold: {args.risk_threshold}")
    print(f"Stage0 YARA: {os.path.abspath(args.yara_rules) if args.yara_rules else '(disabled)'}")
    print(f"Stage3 ONNX: {os.path.abspath(args.onnx) if args.onnx else '(disabled)'}")
    print(f"Cloud VT: {'enabled' if (args.vt and vt_api_key) else '(disabled)'}")

    total = 0
    quarantined = 0

    for fp in iter_files(scan_root):
        total += 1
        try:
            res = scan_file(
                fp,
                quarantine_dir=quarantine_dir,
                entropy_threshold=args.entropy_threshold,
                risk_threshold=args.risk_threshold,
                yara_engine=yara_engine,
                onnx=onnx,
                onnx_max_len=args.onnx_max_len,
                allow_clean_exit=(not args.no_clean_exit),
                vt_api_key=vt_api_key,
                vt_min_score=args.vt_min_score,
                vt_max_score=args.vt_max_score,
                vt_malicious_threshold=args.vt_malicious_threshold,
                allowlist_sha256=allowlist_sha256,
                signed_entropy_override=args.signed_entropy_override,
                signed_quarantine_threshold=args.signed_quarantine_threshold,
            )
            if res.action.startswith("quarantined"):
                quarantined += 1
            yara_tag = (
                f"yara={','.join(res.stage0.rule_names)}" if (res.stage0 and res.stage0.matched) else "yara=-"
            )
            vt_tag = (
                f"vt_mal={res.vt.malicious}" if (res.vt and res.vt.ok) else "vt=-"
            )
            allow_tag = "allow=Y" if res.allowlisted else "allow=-"
            sig_tag = "sig=?" if res.signed_present is None else ("sig=Y" if res.signed_present else "sig=-")
            print(
                f"[{res.action}] risk={res.risk_score:.3f} "
                f"kind={res.stage1.actual_kind or 'unknown'} "
                f"ent={res.stage2.entropy:.2f} "
                f"imports={sorted(res.stage2.suspicious_imports) if res.stage2.suspicious_imports else '-'} "
                f"stage3={res.stage3_score if res.stage3_score is not None else '-'} "
                f"{yara_tag} {vt_tag} {allow_tag} {sig_tag} :: {fp}"
            )
        except Exception as exc:
            print(f"[error] {fp}: {exc}")

    print(f"\nDone. scanned={total} quarantined={quarantined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

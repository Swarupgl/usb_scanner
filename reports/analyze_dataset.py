from __future__ import annotations

import argparse
import collections
import hashlib
import math
import os
import statistics
import struct
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class FileRecord:
    path: str
    label: str  # "benign" | "malicious"
    size: int
    sha256: str
    entropy: Optional[float]
    is_pe: bool
    pe_machine: Optional[str]
    pe_bits: Optional[str]  # "PE32" | "PE32+" | None
    section_names: tuple[str, ...]
    has_upx: bool


def iter_files(root: str, extensions: tuple[str, ...]) -> Iterable[str]:
    root = os.path.abspath(root)
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            lower = name.lower()
            if extensions and not any(lower.endswith(ext) for ext in extensions):
                continue
            yield os.path.join(dirpath, name)


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def shannon_entropy_bytes(data: bytes) -> Optional[float]:
    if not data:
        return 0.0
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    n = len(data)
    ent = 0.0
    for c in counts:
        if not c:
            continue
        p = c / n
        ent -= p * math.log2(p)
    return ent


def shannon_entropy_file(path: str, max_bytes: int = 256 * 1024) -> Optional[float]:
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes)
        return shannon_entropy_bytes(data)
    except Exception:
        return None


def _read_at(f, offset: int, size: int) -> bytes:
    f.seek(offset)
    return f.read(size)


def parse_pe_info(path: str) -> tuple[bool, Optional[str], Optional[str], tuple[str, ...], bool]:
    """Minimal PE header parser.

    Returns:
      (is_pe, machine, bits, section_names, has_upx)

    Notes:
      - This does *not* execute any code; it only reads header bytes.
      - It is intentionally minimal to avoid external dependencies.
    """

    try:
        with open(path, "rb") as f:
            mz = _read_at(f, 0, 64)
            if len(mz) < 64 or mz[0:2] != b"MZ":
                return False, None, None, tuple(), False

            e_lfanew = struct.unpack_from("<I", mz, 0x3C)[0]
            if e_lfanew < 0 or e_lfanew > 10 * 1024 * 1024:
                return False, None, None, tuple(), False

            pe_sig = _read_at(f, e_lfanew, 4)
            if pe_sig != b"PE\x00\x00":
                return False, None, None, tuple(), False

            coff = _read_at(f, e_lfanew + 4, 20)
            if len(coff) != 20:
                return False, None, None, tuple(), False

            machine_u16, number_of_sections, _time_date, _ptr_sym, _num_sym, size_of_optional_header, _chars = struct.unpack(
                "<HHIIIHH", coff
            )

            machine = {
                0x014C: "x86",
                0x8664: "x64",
                0x01C0: "ARM",
                0x01C4: "ARMv7",
                0xAA64: "ARM64",
            }.get(machine_u16, f"0x{machine_u16:04X}")

            opt_magic = _read_at(f, e_lfanew + 4 + 20, 2)
            bits = None
            if len(opt_magic) == 2:
                magic_u16 = struct.unpack("<H", opt_magic)[0]
                if magic_u16 == 0x10B:
                    bits = "PE32"
                elif magic_u16 == 0x20B:
                    bits = "PE32+"

            section_table_off = e_lfanew + 4 + 20 + int(size_of_optional_header)
            section_names: list[str] = []
            has_upx = False

            # IMAGE_SECTION_HEADER is 40 bytes
            for i in range(int(number_of_sections)):
                hdr = _read_at(f, section_table_off + i * 40, 40)
                if len(hdr) != 40:
                    break
                raw_name = hdr[0:8].split(b"\x00", 1)[0]
                try:
                    name = raw_name.decode("ascii", errors="replace")
                except Exception:
                    name = "?"
                name = name.strip()
                if name:
                    section_names.append(name)
                    if name.upper().startswith("UPX"):
                        has_upx = True

            return True, machine, bits, tuple(section_names), has_upx
    except Exception:
        return False, None, None, tuple(), False


def base_name_for_leakage(path: str) -> str:
    """Normalize filename for checking clean-vs-packed twins.

    Example:
      - esentutl.exe -> esentutl
      - esentutl_packed.exe -> esentutl
    """
    name = os.path.basename(path)
    lower = name.lower()
    if lower.endswith(".exe"):
        lower = lower[: -len(".exe")]
    if lower.endswith(".dll"):
        lower = lower[: -len(".dll")]
    if lower.endswith(".com"):
        lower = lower[: -len(".com")]
    if lower.endswith("_packed"):
        lower = lower[: -len("_packed")]
    return lower


def summarize_numbers(values: list[float]) -> str:
    if not values:
        return "(no data)"
    if len(values) == 1:
        return f"n=1, value={values[0]:.3f}"

    return (
        f"n={len(values)}, min={min(values):.3f}, p50={statistics.median(values):.3f}, "
        f"p90={statistics.quantiles(values, n=10)[8]:.3f}, max={max(values):.3f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze dataset folders (counts + PE header/entropy stats).")
    ap.add_argument("--benign-dir", default=os.path.join("datasets", "local", "benign"))
    ap.add_argument("--malicious-dir", default=os.path.join("datasets", "local", "malicious"))
    ap.add_argument(
        "--extensions",
        default=".exe",
        help="Comma-separated extensions to include (e.g. .exe,.dll,.com).",
    )
    ap.add_argument(
        "--entropy-bytes",
        type=int,
        default=256 * 1024,
        help="How many bytes to read for entropy (default 256KB).",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="If >0, cap number of files analyzed per class (for speed).",
    )
    args = ap.parse_args()

    exts = tuple(
        e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}"
        for e in str(args.extensions).split(",")
        if e.strip()
    )

    records: list[FileRecord] = []

    def collect(label: str, root: str) -> None:
        paths = list(iter_files(root, exts))
        if args.max_files and args.max_files > 0:
            paths = paths[: int(args.max_files)]
        for p in paths:
            try:
                size = os.path.getsize(p)
            except OSError:
                continue
            digest = ""
            try:
                digest = sha256_file(p)
            except Exception:
                digest = ""

            ent = shannon_entropy_file(p, max_bytes=int(args.entropy_bytes))
            is_pe, machine, bits, sections, has_upx = parse_pe_info(p)

            records.append(
                FileRecord(
                    path=os.path.abspath(p),
                    label=label,
                    size=size,
                    sha256=digest,
                    entropy=ent,
                    is_pe=is_pe,
                    pe_machine=machine,
                    pe_bits=bits,
                    section_names=sections,
                    has_upx=has_upx,
                )
            )

    collect("benign", args.benign_dir)
    collect("malicious", args.malicious_dir)

    by_label: dict[str, list[FileRecord]] = collections.defaultdict(list)
    for r in records:
        by_label[r.label].append(r)

    print("=== DATASET ANALYSIS REPORT ===")
    print(f"Extensions filter: {exts}")
    print(f"Entropy bytes read: {args.entropy_bytes}")
    print("")

    for label in ("benign", "malicious"):
        items = by_label.get(label, [])
        print(f"--- {label.upper()} ---")
        print(f"Files: {len(items)}")

        if not items:
            print("")
            continue

        sizes = [float(r.size) for r in items]
        ents = [float(r.entropy) for r in items if r.entropy is not None]
        pe_count = sum(1 for r in items if r.is_pe)
        upx_count = sum(1 for r in items if r.has_upx)

        print(f"Size bytes: n={len(sizes)}, min={int(min(sizes))}, median={int(statistics.median(sizes))}, max={int(max(sizes))}")
        print(f"Entropy [0..8]: {summarize_numbers(ents)}")
        print(f"PE detected: {pe_count}/{len(items)}")
        if pe_count:
            machines = collections.Counter(r.pe_machine for r in items if r.pe_machine)
            bits = collections.Counter(r.pe_bits for r in items if r.pe_bits)
            print(f"PE machine breakdown: {dict(machines)}")
            print(f"PE bits breakdown: {dict(bits)}")

            # Most common section names
            sec_counter = collections.Counter()
            for r in items:
                sec_counter.update(s.upper() for s in r.section_names)
            common_secs = sec_counter.most_common(10)
            if common_secs:
                print("Top section names (count): " + ", ".join(f"{n}({c})" for n, c in common_secs))

        print(f"UPX indicator sections (UPX*): {upx_count}/{len(items)}")

        # Duplicate hashes (exact duplicates)
        hashes = [r.sha256 for r in items if r.sha256]
        dup_hashes = len(hashes) - len(set(hashes))
        if hashes:
            print(f"Exact duplicates by SHA256: {dup_hashes}")

        print("")

    # Leakage/twin analysis
    benign_bases = {base_name_for_leakage(r.path) for r in by_label.get("benign", [])}
    mal_bases = {base_name_for_leakage(r.path) for r in by_label.get("malicious", [])}
    twins = sorted(benign_bases.intersection(mal_bases))

    print("--- TWIN / LEAKAGE CHECK ---")
    print("This checks whether the same base program name appears in BOTH classes (e.g., clean foo.exe and packed foo_packed.exe).")
    print(f"Base-name overlap count: {len(twins)}")
    if twins:
        print("Examples: " + ", ".join(twins[:15]) + ("" if len(twins) <= 15 else ", ..."))
    print("")

    print("--- INTERPRETATION NOTES ---")
    print("- This script cannot reliably tell 'malware family/type' unless you have labeled metadata.")
    print("- It CAN tell you structural properties (PE header, machine arch, section-name patterns, entropy, UPX indicators).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

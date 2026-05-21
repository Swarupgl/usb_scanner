from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

FEATURE_DIM = 2381


@dataclass(frozen=True)
class LabeledVector:
    x: np.ndarray
    y: int
    source: str
    id: str
    weight: float = 1.0


def _iter_files(folder: str, exts: tuple[str, ...] = (".exe", ".dll", ".com")) -> Iterable[str]:
    folder_path = Path(folder)
    if not folder_path.exists():
        return
    for root, _dirs, files in os.walk(folder):
        for name in files:
            lower = name.lower()
            if any(lower.endswith(e) for e in exts):
                yield str(Path(root) / name)


def _to_float32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _safe_log1p(v: float) -> float:
    try:
        return float(math.log1p(max(v, 0.0)))
    except Exception:
        return 0.0


def _class_balance(y: np.ndarray) -> dict:
    y = np.asarray(y).astype(np.int64, copy=False)
    if y.size == 0:
        return {"n": 0, "benign": 0, "malicious": 0, "malicious_rate": 0.0}
    benign = int(np.sum(y == 0))
    malicious = int(np.sum(y == 1))
    return {
        "n": int(y.size),
        "benign": benign,
        "malicious": malicious,
        "malicious_rate": float(malicious / max(int(y.size), 1)),
    }


class EmberV2Vectorizer:
    """A lightweight EMBER-v2-style (2381-d) vectorizer.

    This is designed to fuse:
      - Local PE binaries (via `pefile`)
      - EMBER-2018 JSONL raw features

    Feature layout (dims sum to 2381):
      - byte histogram: 256
      - byte entropy histogram: 256
      - strings: 104
      - general: 10
      - header: 62 (10 numeric + 52 hashed)
      - section: 255 (5 numeric + 5*50 hashed)
      - imports: 1280 (hashed)
      - exports: 128 (hashed)
      - data directories: 30
    """

    def __init__(self):
        from sklearn.feature_extraction import FeatureHasher

        self._hasher_header = FeatureHasher(52, input_type="string")
        self._hasher_sections = FeatureHasher(50, input_type="string")
        self._hasher_imports = FeatureHasher(1280, input_type="string")
        self._hasher_exports = FeatureHasher(128, input_type="string")
        self._re_printable = re.compile(rb"[\x20-\x7e]{5,}")

    def _hash(self, hasher, tokens: list[str]) -> np.ndarray:
        if not tokens:
            return np.zeros((hasher.n_features,), dtype=np.float32)
        v = hasher.transform([tokens]).toarray()[0]
        return _to_float32(v)

    def vectorize_ember_jsonl(self, obj: dict) -> np.ndarray:
        hist = _to_float32(np.asarray(obj.get("histogram", [0] * 256)))
        if hist.shape[0] != 256:
            hist = np.zeros((256,), dtype=np.float32)
        hist = hist / max(float(hist.sum()), 1.0)

        be = _to_float32(np.asarray(obj.get("byteentropy", [0] * 256)))
        if be.shape[0] != 256:
            be = np.zeros((256,), dtype=np.float32)
        be = be / max(float(be.sum()), 1.0)

        s = obj.get("strings", {}) or {}
        printabledist = _to_float32(np.asarray(s.get("printabledist", [0] * 96)))
        if printabledist.shape[0] != 96:
            printabledist = np.zeros((96,), dtype=np.float32)
        printabledist = printabledist / max(float(printabledist.sum()), 1.0)
        strings_vec = np.hstack(
            [
                float(s.get("numstrings", 0.0)),
                float(s.get("avlength", 0.0)),
                float(s.get("printables", 0.0)),
                float(s.get("entropy", 0.0)),
                float(s.get("paths", 0.0)),
                float(s.get("urls", 0.0)),
                float(s.get("registry", 0.0)),
                float(s.get("MZ", 0.0)),
                printabledist,
            ]
        ).astype(np.float32)
        if strings_vec.shape[0] != 104:
            strings_vec = np.zeros((104,), dtype=np.float32)

        g = obj.get("general", {}) or {}
        general_vec = np.asarray(
            [
                _safe_log1p(float(g.get("size", 0.0))),
                _safe_log1p(float(g.get("vsize", 0.0))),
                float(g.get("has_debug", 0.0)),
                float(g.get("exports", 0.0)),
                float(g.get("imports", 0.0)),
                float(g.get("has_relocations", 0.0)),
                float(g.get("has_resources", 0.0)),
                float(g.get("has_signature", 0.0)),
                float(g.get("has_tls", 0.0)),
                float(g.get("symbols", 0.0)),
            ],
            dtype=np.float32,
        )

        header = obj.get("header", {}) or {}
        coff = (header.get("coff", {}) or {})
        opt = (header.get("optional", {}) or {})
        header_num = np.asarray(
            [
                _safe_log1p(float(coff.get("timestamp", 0.0))),
                float(opt.get("major_image_version", 0.0)),
                float(opt.get("minor_image_version", 0.0)),
                float(opt.get("major_linker_version", 0.0)),
                float(opt.get("minor_linker_version", 0.0)),
                float(opt.get("major_operating_system_version", 0.0)),
                float(opt.get("minor_operating_system_version", 0.0)),
                float(opt.get("major_subsystem_version", 0.0)),
                float(opt.get("minor_subsystem_version", 0.0)),
                _safe_log1p(float(opt.get("sizeof_code", 0.0))),
            ],
            dtype=np.float32,
        )
        header_tokens: list[str] = []
        if coff.get("machine") is not None:
            header_tokens.append(f"machine={coff.get('machine')}")
        for c in (coff.get("characteristics") or []):
            header_tokens.append(f"coff_char={c}")
        if opt.get("subsystem") is not None:
            header_tokens.append(f"subsystem={opt.get('subsystem')}")
        if opt.get("magic") is not None:
            header_tokens.append(f"magic={opt.get('magic')}")
        for dc in (opt.get("dll_characteristics") or []):
            header_tokens.append(f"dll_char={dc}")
        header_hash = self._hash(self._hasher_header, header_tokens)
        header_vec = np.hstack([header_num, header_hash]).astype(np.float32)
        if header_vec.shape[0] != 62:
            header_vec = np.zeros((62,), dtype=np.float32)

        section = obj.get("section", {}) or {}
        sections = section.get("sections", []) or []
        sec_sizes = [float(s2.get("size", 0.0)) for s2 in sections if isinstance(s2, dict)]
        sec_ent = [float(s2.get("entropy", 0.0)) for s2 in sections if isinstance(s2, dict)]
        sec_names = [str(s2.get("name", "")) for s2 in sections if isinstance(s2, dict)]
        sec_num = np.asarray(
            [
                float(len(sections)),
                float(sum(1 for z in sec_sizes if z <= 0.0)),
                float(sum(1 for n in sec_names if not n)),
                float(np.mean(sec_ent) if sec_ent else 0.0),
                _safe_log1p(float(np.mean(sec_sizes) if sec_sizes else 0.0)),
            ],
            dtype=np.float32,
        )
        tok_names = [f"name={n.lower()}" for n in sec_names if n]
        tok_size = [f"size={n.lower()}:{int(s // 1024)}kb" for n, s in zip(sec_names, sec_sizes) if n]
        tok_ent = [f"ent={n.lower()}:{int(e * 2)}" for n, e in zip(sec_names, sec_ent) if n]
        tok_vsize = [f"vsize={str(s2.get('name','')).lower()}:{int(float(s2.get('vsize',0.0))//1024)}kb" for s2 in sections if isinstance(s2, dict) and s2.get('name')]
        tok_props: list[str] = []
        for s2 in sections:
            if not isinstance(s2, dict):
                continue
            nm = str(s2.get("name", "")).lower()
            for pr in (s2.get("props") or []):
                tok_props.append(f"prop={nm}:{pr}")
        sec_hash = np.hstack(
            [
                self._hash(self._hasher_sections, tok_names),
                self._hash(self._hasher_sections, tok_size),
                self._hash(self._hasher_sections, tok_ent),
                self._hash(self._hasher_sections, tok_vsize),
                self._hash(self._hasher_sections, tok_props),
            ]
        ).astype(np.float32)
        sec_vec = np.hstack([sec_num, sec_hash]).astype(np.float32)
        if sec_vec.shape[0] != 255:
            sec_vec = np.zeros((255,), dtype=np.float32)

        imports = obj.get("imports", {}) or {}
        import_tokens: list[str] = []
        if isinstance(imports, dict):
            for dll, funcs in imports.items():
                dll_l = str(dll).lower()
                for fn in (funcs or []):
                    import_tokens.append(f"{dll_l}:{str(fn).lower()}")
        imports_vec = self._hash(self._hasher_imports, import_tokens)

        exports = obj.get("exports", []) or []
        export_tokens = [str(e).lower() for e in exports if e]
        exports_vec = self._hash(self._hasher_exports, export_tokens)

        dds = obj.get("datadirectories", []) or []
        dd_vec = np.zeros((30,), dtype=np.float32)
        for i, dd in enumerate(dds[:15]):
            if not isinstance(dd, dict):
                continue
            dd_vec[2 * i + 0] = _safe_log1p(float(dd.get("size", 0.0)))
            dd_vec[2 * i + 1] = _safe_log1p(float(dd.get("virtual_address", 0.0)))

        x = np.hstack([hist, be, strings_vec, general_vec, header_vec, sec_vec, imports_vec, exports_vec, dd_vec]).astype(
            np.float32
        )
        if x.shape[0] != FEATURE_DIM:
            raise ValueError(f"Vector dim mismatch: got {x.shape[0]} expected {FEATURE_DIM}")
        return x

    def vectorize_pe_file(self, path: str) -> Optional[np.ndarray]:
        try:
            with open(path, "rb") as f:
                bytez = f.read()
        except Exception:
            return None

        # histogram
        arr = np.frombuffer(bytez, dtype=np.uint8)
        hist = np.bincount(arr, minlength=256).astype(np.float32)
        hist = hist / max(float(hist.sum()), 1.0)

        # byteentropy (EMBER-style-ish 16x16)
        be_mat = np.zeros((16, 16), dtype=np.float32)
        win = 2048
        for i in range(0, len(arr), win):
            chunk = arr[i : i + win]
            if chunk.size == 0:
                continue
            nib = (chunk >> 4).astype(np.int32)
            counts = np.bincount(nib, minlength=16).astype(np.float32)
            p = counts / max(float(counts.sum()), 1.0)
            ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
            ent_bin = min(int(ent / 0.5), 15)
            be_mat[ent_bin] += counts
        be = be_mat.flatten()
        be = be / max(float(be.sum()), 1.0)

        # strings
        strings = self._re_printable.findall(bytez)
        numstrings = float(len(strings))
        avlen = float(np.mean([len(s) for s in strings]) if strings else 0.0)
        printables = float(sum(len(s) for s in strings))
        # printable distribution over 0x20..0x7f (96)
        pd = np.zeros((96,), dtype=np.float32)
        for s in strings:
            for b in s:
                if 0x20 <= b <= 0x7F:
                    pd[b - 0x20] += 1.0
        pd = pd / max(float(pd.sum()), 1.0)
        # entropy of printable distribution
        p2 = pd[pd > 0]
        sent = float(-(p2 * np.log2(p2)).sum()) if p2.size else 0.0
        paths = float(bytez.lower().count(b"c:\\") + bytez.lower().count(b"\\windows\\"))
        urls = float(bytez.lower().count(b"http://") + bytez.lower().count(b"https://") + bytez.lower().count(b"www."))
        registry = float(bytez.lower().count(b"hkey_") + bytez.lower().count(b"software\\"))
        mz = float(bytez.count(b"MZ"))
        strings_vec = np.hstack([numstrings, avlen, printables, sent, paths, urls, registry, mz, pd]).astype(np.float32)

        # PE parsing
        try:
            import pefile

            pe = pefile.PE(data=bytez, fast_load=True)
            # Parse only what we need. Full parse can be slow/hang on some binaries.
            dirs = [
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXPORT"],
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DEBUG"],
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_RESOURCE"],
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_TLS"],
            ]
            pe.parse_data_directories(directories=dirs)
        except Exception:
            pe = None

        size = float(len(bytez))
        vsize = float(getattr(getattr(pe, "OPTIONAL_HEADER", None), "SizeOfImage", 0) if pe is not None else 0.0)
        has_debug = 0.0
        if pe is not None and hasattr(pe, "OPTIONAL_HEADER"):
            try:
                # IMAGE_DIRECTORY_ENTRY_DEBUG = 6
                has_debug = 1.0 if getattr(pe.OPTIONAL_HEADER.DATA_DIRECTORY[6], "Size", 0) > 0 else 0.0
            except Exception:
                has_debug = 1.0 if hasattr(pe, "DIRECTORY_ENTRY_DEBUG") else 0.0

        exports_count = 0.0
        if pe is not None and hasattr(pe, "DIRECTORY_ENTRY_EXPORT"):
            exports_count = float(len(getattr(pe.DIRECTORY_ENTRY_EXPORT, "symbols", []) or []))

        imports_count = 0.0
        if pe is not None and hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            for e in pe.DIRECTORY_ENTRY_IMPORT:
                imports_count += float(len(getattr(e, "imports", []) or []))

        has_reloc = 0.0
        has_res = 0.0
        has_tls = 0.0
        if pe is not None and hasattr(pe, "OPTIONAL_HEADER"):
            try:
                # BASE RELOC = 5, RESOURCE = 2, TLS = 9
                has_reloc = 1.0 if getattr(pe.OPTIONAL_HEADER.DATA_DIRECTORY[5], "Size", 0) > 0 else 0.0
                has_res = 1.0 if getattr(pe.OPTIONAL_HEADER.DATA_DIRECTORY[2], "Size", 0) > 0 else 0.0
                has_tls = 1.0 if getattr(pe.OPTIONAL_HEADER.DATA_DIRECTORY[9], "Size", 0) > 0 else 0.0
            except Exception:
                has_reloc = 1.0 if hasattr(pe, "DIRECTORY_ENTRY_BASERELOC") else 0.0
                has_res = 1.0 if hasattr(pe, "DIRECTORY_ENTRY_RESOURCE") else 0.0
                has_tls = 1.0 if hasattr(pe, "DIRECTORY_ENTRY_TLS") else 0.0

        symbols = 0.0
        if pe is not None and hasattr(pe, "FILE_HEADER"):
            try:
                symbols = float(getattr(pe.FILE_HEADER, "NumberOfSymbols", 0) or 0)
            except Exception:
                symbols = 0.0

        # Authenticode: check security directory size
        has_sig = 0.0
        if pe is not None and hasattr(pe, "OPTIONAL_HEADER"):
            try:
                # IMAGE_DIRECTORY_ENTRY_SECURITY = 4
                sec_dir = pe.OPTIONAL_HEADER.DATA_DIRECTORY[4]
                has_sig = 1.0 if getattr(sec_dir, "Size", 0) > 0 else 0.0
            except Exception:
                has_sig = 0.0

        general_vec = np.asarray(
            [
                _safe_log1p(size),
                _safe_log1p(vsize),
                has_debug,
                exports_count,
                imports_count,
                has_reloc,
                has_res,
                has_sig,
                has_tls,
                symbols,
            ],
            dtype=np.float32,
        )

        # Header
        header_num = np.zeros((10,), dtype=np.float32)
        header_tokens: list[str] = []
        if pe is not None:
            try:
                header_num[0] = _safe_log1p(float(pe.FILE_HEADER.TimeDateStamp))
                header_tokens.append(f"machine={pe.FILE_HEADER.Machine}")
                header_tokens.append(f"char={pe.FILE_HEADER.Characteristics}")
            except Exception:
                pass
            try:
                oh = pe.OPTIONAL_HEADER
                header_num[1] = float(getattr(oh, "MajorImageVersion", 0))
                header_num[2] = float(getattr(oh, "MinorImageVersion", 0))
                header_num[3] = float(getattr(oh, "MajorLinkerVersion", 0))
                header_num[4] = float(getattr(oh, "MinorLinkerVersion", 0))
                header_num[5] = float(getattr(oh, "MajorOperatingSystemVersion", 0))
                header_num[6] = float(getattr(oh, "MinorOperatingSystemVersion", 0))
                header_num[7] = float(getattr(oh, "MajorSubsystemVersion", 0))
                header_num[8] = float(getattr(oh, "MinorSubsystemVersion", 0))
                header_num[9] = _safe_log1p(float(getattr(oh, "SizeOfCode", 0)))
                header_tokens.append(f"subsystem={getattr(oh, 'Subsystem', 0)}")
                header_tokens.append(f"magic={getattr(oh, 'Magic', 0)}")
                header_tokens.append(f"dll_char={getattr(oh, 'DllCharacteristics', 0)}")
            except Exception:
                pass
        header_hash = self._hash(self._hasher_header, header_tokens)
        header_vec = np.hstack([header_num, header_hash]).astype(np.float32)

        # Sections
        sec_num = np.zeros((5,), dtype=np.float32)
        tok_names: list[str] = []
        tok_size: list[str] = []
        tok_ent: list[str] = []
        tok_vsize: list[str] = []
        tok_props: list[str] = []
        if pe is not None and hasattr(pe, "sections"):
            secs = pe.sections
            sec_num[0] = float(len(secs))
            sizes = [float(getattr(s, "SizeOfRawData", 0)) for s in secs]
            ents = [float(getattr(s, "get_entropy")()) for s in secs]
            sec_num[1] = float(sum(1 for z in sizes if z <= 0.0))
            names = []
            for s in secs:
                try:
                    nm = s.Name.rstrip(b"\x00").decode("utf-8", errors="ignore").lower()
                except Exception:
                    nm = ""
                names.append(nm)
                if nm:
                    tok_names.append(f"name={nm}")
                    tok_size.append(f"size={nm}:{int(float(getattr(s, 'SizeOfRawData', 0)) // 1024)}kb")
                    tok_ent.append(f"ent={nm}:{int(float(s.get_entropy()) * 2)}")
                    tok_vsize.append(f"vsize={nm}:{int(float(getattr(s, 'Misc_VirtualSize', 0)) // 1024)}kb")
                    tok_props.append(f"chars={nm}:{int(getattr(s, 'Characteristics', 0))}")
            sec_num[2] = float(sum(1 for n in names if not n))
            sec_num[3] = float(np.mean(ents) if ents else 0.0)
            sec_num[4] = _safe_log1p(float(np.mean(sizes) if sizes else 0.0))

        sec_hash = np.hstack(
            [
                self._hash(self._hasher_sections, tok_names),
                self._hash(self._hasher_sections, tok_size),
                self._hash(self._hasher_sections, tok_ent),
                self._hash(self._hasher_sections, tok_vsize),
                self._hash(self._hasher_sections, tok_props),
            ]
        ).astype(np.float32)
        sec_vec = np.hstack([sec_num, sec_hash]).astype(np.float32)

        # Imports/exports
        import_tokens: list[str] = []
        export_tokens: list[str] = []
        if pe is not None and hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            for e in pe.DIRECTORY_ENTRY_IMPORT:
                try:
                    dll = e.dll.decode("utf-8", errors="ignore").lower()
                except Exception:
                    dll = ""
                for imp in (getattr(e, "imports", []) or []):
                    fn = getattr(imp, "name", None)
                    if fn is None:
                        continue
                    try:
                        fn_s = fn.decode("utf-8", errors="ignore").lower()
                    except Exception:
                        fn_s = str(fn).lower()
                    import_tokens.append(f"{dll}:{fn_s}")
        if pe is not None and hasattr(pe, "DIRECTORY_ENTRY_EXPORT"):
            for sym in (getattr(pe.DIRECTORY_ENTRY_EXPORT, "symbols", []) or []):
                nm = getattr(sym, "name", None)
                if nm is None:
                    continue
                try:
                    export_tokens.append(nm.decode("utf-8", errors="ignore").lower())
                except Exception:
                    export_tokens.append(str(nm).lower())

        imports_vec = self._hash(self._hasher_imports, import_tokens)
        exports_vec = self._hash(self._hasher_exports, export_tokens)

        # Data directories (15)
        dd_vec = np.zeros((30,), dtype=np.float32)
        if pe is not None and hasattr(pe, "OPTIONAL_HEADER"):
            try:
                dds = pe.OPTIONAL_HEADER.DATA_DIRECTORY
                for i, dd in enumerate(dds[:15]):
                    dd_vec[2 * i + 0] = _safe_log1p(float(getattr(dd, "Size", 0)))
                    dd_vec[2 * i + 1] = _safe_log1p(float(getattr(dd, "VirtualAddress", 0)))
            except Exception:
                pass

        x = np.hstack([hist, be, strings_vec, general_vec, header_vec, sec_vec, imports_vec, exports_vec, dd_vec]).astype(
            np.float32
        )
        if x.shape[0] != FEATURE_DIM:
            return None
        return x


def load_local_binaries(
    benign_dir: str,
    malicious_dir: str,
    *,
    vectorizer: EmberV2Vectorizer,
    max_per_class: Optional[int],
    seed: int,
) -> tuple[list[LabeledVector], list[LabeledVector]]:
    rng = random.Random(seed)

    benign_paths = list(_iter_files(benign_dir))
    malicious_paths = list(_iter_files(malicious_dir))

    rng.shuffle(benign_paths)
    rng.shuffle(malicious_paths)

    if max_per_class is not None:
        benign_paths = benign_paths[:max_per_class]
        malicious_paths = malicious_paths[:max_per_class]

    benign: list[LabeledVector] = []
    for p in benign_paths:
        x = vectorizer.vectorize_pe_file(p)
        if x is None:
            continue
        benign.append(LabeledVector(x=x, y=0, source="local_benign", id=p,))

    malicious: list[LabeledVector] = []
    for p in malicious_paths:
        x = vectorizer.vectorize_pe_file(p)
        if x is None:
            continue
        malicious.append(LabeledVector(x=x, y=1, source="local_malicious", id=p,))

    return benign, malicious


def _iter_ember_jsonl_files(ember_dir: str) -> list[str]:
    p = Path(ember_dir)
    if p.is_file():
        return [str(p)]
    if not p.exists():
        return []

    files = sorted(str(x) for x in p.glob("*.jsonl"))
    # Prefer train files first for better class balance.
    train_first = sorted([f for f in files if "train" in Path(f).name]) + sorted(
        [f for f in files if "train" not in Path(f).name]
    )
    return train_first


def load_ember_jsonl_subset(
    ember_dir: str,
    *,
    vectorizer: EmberV2Vectorizer,
    max_samples: int,
    seed: int,
    balanced: bool,
) -> list[LabeledVector]:
    """Loads up to `max_samples` examples from EMBER-2018 JSONL feature files.

    Each JSONL row is a dict with a `label` field (0/1 or -1 for unlabeled).
    """

    rng = random.Random(seed)
    files = _iter_ember_jsonl_files(ember_dir)
    if not files:
        print(f"[EMBER] No JSONL files found at: {ember_dir}")
        return []

    out: list[LabeledVector] = []
    out_b: list[LabeledVector] = []
    out_m: list[LabeledVector] = []
    target_b = max_samples // 2
    target_m = max_samples - target_b
    for fp in files:
        if (len(out_b) + len(out_m) if balanced else len(out)) >= max_samples:
            break
        try:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    if (len(out_b) + len(out_m) if balanced else len(out)) >= max_samples:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue

                    label = obj.get("label", None)
                    if label is None:
                        continue
                    try:
                        label_i = int(label)
                    except Exception:
                        continue
                    if label_i not in (0, 1):
                        # skip unlabeled (-1)
                        continue

                    if balanced:
                        if label_i == 0 and len(out_b) >= target_b:
                            continue
                        if label_i == 1 and len(out_m) >= target_m:
                            continue

                    try:
                        x = vectorizer.vectorize_ember_jsonl(obj)
                    except Exception:
                        continue

                    sha = str(obj.get("sha256", ""))
                    lv = LabeledVector(x=x, y=label_i, source="ember2018", id=sha)
                    if balanced:
                        if label_i == 0:
                            out_b.append(lv)
                        else:
                            out_m.append(lv)
                    else:
                        out.append(lv)
        except Exception:
            continue

    if balanced:
        out = out_b + out_m

    rng.shuffle(out)
    if len(out) > max_samples:
        out = out[:max_samples]

    return out


def load_sorel_lmdb_subset(
    sorel_lmdb_dir: str,
    *,
    max_samples: int,
    seed: int,
) -> list[LabeledVector]:
    """Loads up to `max_samples` malicious vectors from a SOREL LMDB.

    Expected input is the *directory* containing `data.mdb` and `lock.mdb`.

    NOTE: Your current workspace only shows `datasets/external/sorel_features/lock.mdb`.
    LMDB requires `data.mdb` as well; if it is missing this returns an empty list and prints a warning.
    """

    import lmdb
    import pickle

    rng = random.Random(seed)
    p = Path(sorel_lmdb_dir)
    if p.is_file():
        p = p.parent

    data_mdb = p / "data.mdb"
    if not data_mdb.exists():
        print(f"[SOREL] Skipping: missing {data_mdb}")
        return []

    out: list[LabeledVector] = []
    env = lmdb.open(str(p), readonly=True, lock=False, readahead=False, max_dbs=1)
    with env.begin(buffers=True) as txn:
        cur = txn.cursor()
        for i, (k, v) in enumerate(cur):
            if len(out) >= max_samples:
                break

            vb = bytes(v)
            x: Optional[np.ndarray] = None

            # Try packed vectors first.
            if len(vb) == FEATURE_DIM * 4:
                x = np.frombuffer(vb, dtype=np.float32)
            elif len(vb) == FEATURE_DIM * 8:
                x = np.frombuffer(vb, dtype=np.float64).astype(np.float32)
            else:
                # Common: pickled numpy array or dict.
                try:
                    obj = pickle.loads(vb)
                    if isinstance(obj, np.ndarray):
                        x = obj.astype(np.float32, copy=False)
                    elif isinstance(obj, (list, tuple)):
                        x = np.asarray(obj, dtype=np.float32)
                    elif isinstance(obj, dict) and "x" in obj:
                        x = np.asarray(obj["x"], dtype=np.float32)
                except Exception:
                    x = None

            if x is None:
                continue

            if x.shape[0] != FEATURE_DIM:
                continue

            key = k.tobytes() if hasattr(k, "tobytes") else bytes(k)
            out.append(LabeledVector(x=x, y=1, source="sorel", id=key.hex()))

            if i > 5_000_000:
                # Safety valve for pathological DBs.
                break

    rng.shuffle(out)
    if len(out) > max_samples:
        out = out[:max_samples]

    return out


def stack_dataset(samples: list[LabeledVector]) -> tuple[np.ndarray, np.ndarray]:
    if not samples:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    X = np.stack([s.x for s in samples], axis=0).astype(np.float32, copy=False)
    y = np.asarray([s.y for s in samples], dtype=np.int64)
    return X, y


def stack_dataset_with_weights(samples: list[LabeledVector]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not samples:
        return (
            np.zeros((0, FEATURE_DIM), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

    X = np.stack([s.x for s in samples], axis=0).astype(np.float32, copy=False)
    y = np.asarray([s.y for s in samples], dtype=np.int64)
    w = np.asarray([float(getattr(s, "weight", 1.0)) for s in samples], dtype=np.float32)
    return X, y, w


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: Optional[np.ndarray],
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    n_estimators: int,
    num_leaves: int,
    learning_rate: float,
    n_jobs: int,
):
    from lightgbm import LGBMClassifier

    clf = LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=-1,
    )

    if w_train is not None and w_train.size == y_train.size:
        clf.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_val, y_val)])
    else:
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    return clf


def choose_threshold(
    *,
    val_probs: np.ndarray,
    y_val: np.ndarray,
    benign_complex_probs: Optional[np.ndarray],
    strategy: str,
) -> float:
    if strategy == "fixed0.5":
        return 0.5

    def _strict_above_max(probs: np.ndarray) -> float:
        if probs.size == 0:
            return 0.5
        thr = float(np.max(probs))
        return min(max(thr + 1e-6, 0.0), 1.0)

    if strategy == "fpr0_val":
        return _strict_above_max(val_probs[y_val == 0])

    if strategy == "fpr0_benign_complex":
        if benign_complex_probs is None:
            return _strict_above_max(val_probs[y_val == 0])
        return _strict_above_max(benign_complex_probs)

    if strategy == "fpr0_both":
        thr_val = _strict_above_max(val_probs[y_val == 0])
        thr_bc = (
            _strict_above_max(benign_complex_probs)
            if benign_complex_probs is not None
            else thr_val
        )
        return max(thr_val, thr_bc)

    raise ValueError(f"Unknown threshold strategy: {strategy}")


def export_lightgbm_to_onnx(clf, out_path: str, *, target_opset: int = 15) -> None:
    from onnxmltools.convert import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    onx = convert_lightgbm(
        clf,
        initial_types=[("input", FloatTensorType([None, FEATURE_DIM]))],
        target_opset=target_opset,
    )
    with open(out_path, "wb") as f:
        f.write(onx.SerializeToString())


def quantize_onnx(in_path: str, out_path: str) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(in_path, out_path, weight_type=QuantType.QInt8)


def benchmark_onnx_latency(onnx_path: str, X: np.ndarray, *, max_samples: int, seed: int) -> dict:
    import onnxruntime as ort

    if X.shape[0] == 0:
        return {"onnx_path": onnx_path, "samples": 0, "ms_per_sample_p50": None, "ms_per_sample_p95": None}

    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=min(max_samples, X.shape[0]), replace=False)
    Xb = X[idx].astype(np.float32, copy=False)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # Warmup
    for i in range(min(10, Xb.shape[0])):
        _ = sess.run(None, {input_name: Xb[i : i + 1]})

    times_ms: list[float] = []
    for i in range(Xb.shape[0]):
        t0 = time.perf_counter()
        _ = sess.run(None, {input_name: Xb[i : i + 1]})
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "onnx_path": onnx_path,
        "samples": int(Xb.shape[0]),
        "ms_per_sample_p50": float(np.percentile(arr, 50)),
        "ms_per_sample_p95": float(np.percentile(arr, 95)),
        "ms_per_sample_mean": float(arr.mean()),
    }


def report_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, title: str) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)

    # For quick scanning, also compute FP/FN.
    tn, fp, fn, tp = cm.ravel().tolist()

    print("\n" + title)
    print("Confusion Matrix [[TN FP],[FN TP]]:\n", cm)
    print(f"Accuracy: {acc:.4f}")
    print(f"FP: {fp} | FN: {fn} | TP: {tp} | TN: {tn}")

    return {
        "title": title,
        "confusion_matrix": cm.tolist(),
        "accuracy": float(acc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision_benign": float(prec[0]),
        "recall_benign": float(rec[0]),
        "f1_benign": float(f1[0]),
        "precision_malicious": float(prec[1]),
        "recall_malicious": float(rec[1]),
        "f1_malicious": float(f1[1]),
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Master Fusion trainer: Local PE + EMBER JSONL + SOREL LMDB -> LightGBM -> ONNX")
    p.add_argument(
        "--benign-complex-dir",
        default=str(Path("datasets") / "local" / "benign_complex"),
        help="Local benign installers/complex binaries (used for FP-focused evaluation)",
    )
    p.add_argument(
        "--malicious-dir",
        default=str(Path("datasets") / "local" / "malicious"),
        help="Local malicious folder (or packed demo samples)",
    )
    p.add_argument(
        "--ember2018-dir",
        default=str(Path("datasets") / "external" / "ember2018"),
        help="Folder containing EMBER-2018 *.jsonl feature files",
    )
    p.add_argument(
        "--ember-max-samples",
        type=int,
        default=100_000,
        help="Max number of EMBER JSONL samples to load (streamed)",
    )
    p.add_argument(
        "--ember-balance",
        choices=["balanced", "none"],
        default="balanced",
        help="How to sample EMBER when streaming. 'balanced' targets ~50/50 benign/malicious; 'none' takes first N rows.",
    )
    p.add_argument(
        "--sorel-lmdb-dir",
        default=str(Path("datasets") / "external" / "sorel_features"),
        help="Directory containing SOREL LMDB (must include data.mdb + lock.mdb). Use --disable-sorel or set to 'none' to skip.",
    )
    p.add_argument(
        "--disable-sorel",
        action="store_true",
        help="Disable SOREL ingestion entirely (useful when you only have lock.mdb or no LMDB at all)",
    )
    p.add_argument(
        "--sorel-max-samples",
        type=int,
        default=100_000,
        help="Max number of SOREL samples to load (malicious-only)",
    )
    p.add_argument("--local-max-per-class", type=int, default=None, help="Cap local benign/malicious files per class")
    p.add_argument(
        "--benign-complex-train-ratio",
        type=float,
        default=0.0,
        help=(
            "Fraction of benign_complex samples to include as BENIGN training data (rest kept for FP audit). "
            "Use this to teach the model that packed installers can be benign."
        ),
    )
    p.add_argument(
        "--benign-complex-train-weight",
        type=float,
        default=5.0,
        help="Sample weight applied to benign_complex training subset when benign_complex-train-ratio > 0",
    )

    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument(
        "--threshold",
        choices=["fixed0.5", "fpr0_val", "fpr0_benign_complex", "fpr0_both"],
        default="fixed0.5",
        help=(
            "Decision threshold strategy. "
            "fixed0.5 uses 0.5. "
            "fpr0_val sets threshold above max benign probability in validation (0 FP on val benign). "
            "fpr0_benign_complex sets threshold above max benign probability on benign_complex audit (0 FP there). "
            "fpr0_both enforces both."
        ),
    )

    p.add_argument("--n-estimators", type=int, default=600)
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--jobs", type=int, default=max(os.cpu_count() or 2, 2))

    p.add_argument("--out-dir", default=str(Path("outputs") / "fusion" / "fusion_out"))
    p.add_argument(
        "--save-npz",
        action="store_true",
        help="Save fused arrays (X_train/y_train/X_val/y_val and benign_complex audit) as an .npz in out-dir",
    )
    p.add_argument("--export-onnx", action="store_true", help="Export LightGBM to ONNX")
    p.add_argument("--quantize", action="store_true", help="Quantize ONNX weights to INT8 (onnxruntime)")
    p.add_argument("--latency-max-samples", type=int, default=2000)

    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vectorizer = EmberV2Vectorizer()
    print(f"Vectorizer: EmberV2Vectorizer (dim={FEATURE_DIM})")

    # 1) Load local binaries (benign_complex used as special FP eval set)
    t0 = time.time()
    local_benign_complex, local_malicious = load_local_binaries(
        args.benign_complex_dir,
        args.malicious_dir,
        vectorizer=vectorizer,
        max_per_class=args.local_max_per_class,
        seed=args.seed,
    )
    print(f"[Local] benign_complex={len(local_benign_complex)} malicious={len(local_malicious)} in {time.time()-t0:.1f}s")

    # Optionally use part of benign_complex as training benign; keep remainder as FP audit holdout.
    benign_complex_train: list[LabeledVector] = []
    benign_complex_audit: list[LabeledVector] = list(local_benign_complex)
    ratio = float(args.benign_complex_train_ratio)
    if ratio > 0.0 and len(local_benign_complex) >= 2:
        ratio = min(max(ratio, 0.0), 0.95)
        n_train = int(round(len(local_benign_complex) * ratio))
        n_train = min(max(n_train, 0), len(local_benign_complex) - 1)
        benign_complex_train = [
            LabeledVector(x=s.x, y=0, source="local_benign_complex_train", id=s.id, weight=float(args.benign_complex_train_weight))
            for s in local_benign_complex[:n_train]
        ]
        benign_complex_audit = list(local_benign_complex[n_train:])
        print(
            f"[BenignComplex] train={len(benign_complex_train)} (w={args.benign_complex_train_weight}) "
            f"audit={len(benign_complex_audit)}"
        )

    # 2) Load EMBER subset
    t0 = time.time()
    ember_samples = load_ember_jsonl_subset(
        args.ember2018_dir,
        vectorizer=vectorizer,
        max_samples=args.ember_max_samples,
        seed=args.seed,
        balanced=(args.ember_balance == "balanced"),
    )
    ember_ben = sum(1 for s in ember_samples if s.y == 0)
    ember_mal = sum(1 for s in ember_samples if s.y == 1)
    print(f"[EMBER2018] loaded={len(ember_samples)} benign={ember_ben} mal={ember_mal} in {time.time()-t0:.1f}s")

    # 3) Load SOREL LMDB malicious subset
    t0 = time.time()
    sorel_samples: list[LabeledVector] = []
    sorel_dir_norm = str(args.sorel_lmdb_dir).strip() if args.sorel_lmdb_dir is not None else ""
    if args.disable_sorel or sorel_dir_norm.lower() in {"", "none", "null"}:
        print("[SOREL] skipped (disabled)")
    else:
        sorel_samples = load_sorel_lmdb_subset(
            args.sorel_lmdb_dir,
            max_samples=args.sorel_max_samples,
            seed=args.seed,
        )
        print(f"[SOREL] loaded={len(sorel_samples)} in {time.time()-t0:.1f}s")

    # Fusion: training set optionally includes some benign_complex; remainder is a targeted FP holdout.
    train_pool = list(local_malicious) + list(ember_samples) + list(sorel_samples) + list(benign_complex_train)

    # Ensure feature dim and finite values
    cleaned: list[LabeledVector] = []
    dropped = 0
    for s in train_pool:
        if s.x.shape[0] != FEATURE_DIM:
            dropped += 1
            continue
        if not np.isfinite(s.x).all():
            dropped += 1
            continue
        cleaned.append(s)

    print(f"[Fusion] train_pool={len(train_pool)} cleaned={len(cleaned)} dropped={dropped}")

    X, y, w = stack_dataset_with_weights(cleaned)
    X_fp, y_fp = stack_dataset(benign_complex_audit)

    if X.shape[0] == 0:
        print("No training samples available after fusion.")
        return 2

    # Split
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X,
        y,
        w,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    if args.save_npz:
        npz_path = out_dir / "fusion_dataset.npz"
        np.savez_compressed(
            npz_path,
            X_train=X_train.astype(np.float32, copy=False),
            y_train=y_train.astype(np.int64, copy=False),
            X_val=X_val.astype(np.float32, copy=False),
            y_val=y_val.astype(np.int64, copy=False),
            X_benign_complex=X_fp.astype(np.float32, copy=False),
            y_benign_complex=y_fp.astype(np.int64, copy=False),
        )
        print(f"Saved fused dataset: {npz_path}")

        # Small sidecar manifest for quick inspection / traceability.
        source_counts_train_pool = collections.Counter(s.source for s in cleaned)
        source_counts_benign_complex = collections.Counter(s.source for s in local_benign_complex)
        manifest = {
            "feature_dim": FEATURE_DIM,
            "paths": {
                "npz": str(npz_path),
                "manifest": str(out_dir / "npz_manifest.json"),
            },
            "splits": {
                "train": {
                    "shape": [int(X_train.shape[0]), int(X_train.shape[1])],
                    "class_balance": _class_balance(y_train),
                },
                "val": {
                    "shape": [int(X_val.shape[0]), int(X_val.shape[1])],
                    "class_balance": _class_balance(y_val),
                },
                "benign_complex_audit": {
                    "shape": [int(X_fp.shape[0]), int(X_fp.shape[1]) if X_fp.ndim == 2 else FEATURE_DIM],
                    "class_balance": _class_balance(y_fp),
                },
            },
            "sources": {
                "train_pool_counts": dict(source_counts_train_pool),
                "benign_complex_counts": dict(source_counts_benign_complex),
            },
            "args": {
                "benign_complex_dir": args.benign_complex_dir,
                "malicious_dir": args.malicious_dir,
                "ember2018_dir": args.ember2018_dir,
                "sorel_lmdb_dir": args.sorel_lmdb_dir,
                "ember_max_samples": int(args.ember_max_samples),
                "sorel_max_samples": int(args.sorel_max_samples),
                "local_max_per_class": args.local_max_per_class,
                "seed": int(args.seed),
                "val_ratio": float(args.val_ratio),
            },
        }
        manifest_path = out_dir / "npz_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote NPZ manifest: {manifest_path}")

    print(f"Train={X_train.shape[0]} Val={X_val.shape[0]} (dim={X_train.shape[1]})")

    # Train LightGBM
    t0 = time.time()
    clf = train_lightgbm(
        X_train,
        y_train,
        w_train,
        X_val,
        y_val,
        seed=args.seed,
        n_estimators=args.n_estimators,
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        n_jobs=args.jobs,
    )
    print(f"[Train] LightGBM done in {time.time()-t0:.1f}s")

    # Predict + choose threshold
    val_probs = clf.predict_proba(X_val)[:, 1]
    benign_complex_probs = clf.predict_proba(X_fp)[:, 1] if X_fp.shape[0] > 0 else None
    thr = choose_threshold(
        val_probs=val_probs,
        y_val=y_val,
        benign_complex_probs=benign_complex_probs,
        strategy=args.threshold,
    )
    val_pred = (val_probs >= thr).astype(np.int64)

    metrics_val = report_metrics(y_val, val_pred, title=f"Validation (threshold={thr:.6f}, strategy={args.threshold})")

    # FP-focused report on benign_complex
    if X_fp.shape[0] > 0 and benign_complex_probs is not None:
        fp_pred = (benign_complex_probs >= thr).astype(np.int64)
        metrics_fp = report_metrics(y_fp, fp_pred, title="Benign_complex FP Audit (all labels=0)")
    else:
        metrics_fp = {"title": "Benign_complex FP Audit", "note": "No files loaded"}
        print("[WARN] No benign_complex samples loaded; FP audit skipped")

    # Save native LightGBM model
    native_path = out_dir / "fusion_lgbm.txt"
    try:
        clf.booster_.save_model(str(native_path))
        print(f"Saved native LightGBM model: {native_path}")
    except Exception:
        pass

    onnx_paths: dict[str, str] = {}
    latency: dict[str, dict] = {}

    if args.export_onnx:
        onnx_path = out_dir / "fusion_lgbm.onnx"
        export_lightgbm_to_onnx(clf, str(onnx_path), target_opset=15)
        onnx_paths["onnx"] = str(onnx_path)
        print(f"Exported ONNX: {onnx_path}")

        if args.quantize:
            q_path = out_dir / "fusion_lgbm.int8.onnx"
            quantize_onnx(str(onnx_path), str(q_path))
            onnx_paths["onnx_int8"] = str(q_path)
            print(f"Quantized ONNX (INT8): {q_path}")

        # Latency benchmark (ONNX only)
        latency["val_onnx"] = benchmark_onnx_latency(str(onnx_path), X_val, max_samples=args.latency_max_samples, seed=args.seed)
        if args.quantize:
            latency["val_onnx_int8"] = benchmark_onnx_latency(str(q_path), X_val, max_samples=args.latency_max_samples, seed=args.seed)

    report = {
        "feature_dim": FEATURE_DIM,
        "extractor": "ember_v2_lite",
        "counts": {
            "local_benign_complex": len(local_benign_complex),
            "local_malicious": len(local_malicious),
            "ember2018": len(ember_samples),
            "sorel": len(sorel_samples),
            "train_cleaned": int(X.shape[0]),
            "val": int(X_val.shape[0]),
        },
        "threshold": {"strategy": args.threshold, "value": float(thr)},
        "metrics": {"validation": metrics_val, "benign_complex": metrics_fp},
        "onnx": onnx_paths,
        "latency": latency,
    }

    report_path = out_dir / "fusion_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote report: {report_path}")

    # Quick "Accuracy vs. Inference Time" summary
    if latency:
        print("\nAccuracy vs. Inference Time (ONNX)")
        for k, v in latency.items():
            print(f"- {k}: acc={metrics_val.get('accuracy'):.4f} | p50={v.get('ms_per_sample_p50')}ms | p95={v.get('ms_per_sample_p95')}ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

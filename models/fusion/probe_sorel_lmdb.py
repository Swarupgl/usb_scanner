from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Optional

import lmdb
import numpy as np


def decode_value(vb: bytes) -> Any:
    # Common encodings we might see in malware feature LMDBs.
    if len(vb) == 2381 * 4:
        return np.frombuffer(vb, dtype=np.float32)
    if len(vb) == 2381 * 8:
        return np.frombuffer(vb, dtype=np.float64)

    for decoder in (pickle.loads, lambda b: json.loads(b.decode("utf-8"))):
        try:
            return decoder(vb)
        except Exception:
            pass

    return vb


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Probe a SOREL LMDB and print a few decoded entries")
    p.add_argument(
        "--lmdb-dir",
        default=str(Path("datasets") / "external" / "sorel_features"),
        help="Directory containing data.mdb and lock.mdb",
    )
    p.add_argument("--max", type=int, default=3, help="Max entries to print")
    args = p.parse_args(argv)

    lmdb_dir = os.path.abspath(args.lmdb_dir)
    if not os.path.exists(lmdb_dir):
        raise SystemExit(f"Missing LMDB directory: {lmdb_dir}")

    env = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=False, max_dbs=1)
    with env.begin(buffers=True) as txn:
        cur = txn.cursor()
        for i, (k, v) in enumerate(cur):
            vb = bytes(v)
            print(f"--- entry {i}")
            print("key_len=", len(k), "val_len=", len(vb))
            obj = decode_value(vb)
            print("decoded_type=", type(obj))
            if isinstance(obj, np.ndarray):
                print("ndarray shape=", obj.shape, "dtype=", obj.dtype, "head=", obj[:5])
            elif isinstance(obj, dict):
                print("dict keys sample=", list(obj.keys())[:15])
            elif isinstance(obj, (list, tuple)):
                print("seq len=", len(obj), "head=", obj[:3])
            else:
                print("raw bytes head=", vb[:32])

            if i >= max(int(args.max) - 1, 0):
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

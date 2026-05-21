from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional


def iter_files(root: str) -> Iterable[str]:
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            yield os.path.join(dirpath, name)


def sha256_file(path: str, max_bytes: Optional[int] = None) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Create a SHA256 allowlist file from a folder")
    p.add_argument("--folder", required=True, help="Folder containing known-good files (e.g. installers)")
    p.add_argument("--out", default="allowlist.sha256", help="Output allowlist text file")
    args = p.parse_args(argv)

    folder = os.path.abspath(args.folder)
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    count = 0
    for fp in iter_files(folder):
        try:
            digest = sha256_file(fp)
        except Exception:
            continue
        rel = os.path.relpath(fp, folder)
        lines.append(f"{digest}  {rel}")
        count += 1

    lines.sort()
    header = [
        "# SHA256 allowlist (one per line)",
        f"# source_folder={folder}",
        f"# entries={count}",
        "",
    ]
    out_path.write_text("\n".join(header + lines) + "\n", encoding="utf-8")

    print(f"Wrote allowlist: {out_path} (entries={count})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

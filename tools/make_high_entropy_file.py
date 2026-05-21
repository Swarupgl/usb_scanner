from __future__ import annotations

import argparse
import os
import secrets
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Create a high-entropy random file for safe AV/sanitizer demos")
    p.add_argument("--out", default=str(Path("outputs") / "demo" / "av_demo" / "chaos.bin"))
    p.add_argument("--size", type=int, default=2 * 1024 * 1024, help="Size in bytes (default: 2 MiB)")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pure random bytes (high entropy). Not an executable, not malware.
    data = secrets.token_bytes(args.size)
    out_path.write_bytes(data)

    print(f"Wrote: {out_path.resolve()} ({args.size} bytes)")
    print("Note: this is random data, not a valid file format.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

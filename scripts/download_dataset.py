#!/usr/bin/env python3
"""Download the official STAG classification pressure-only archive."""
from __future__ import annotations
import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path

URL = "https://stag.csail.mit.edu/datasets/classification_lite.zip"
DEFAULT_OUTPUT = Path("data/raw/classification_lite.zip")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    target = args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not args.force:
        print(f"Already exists: {target}")
        print(f"SHA-256: {sha256(target)}")
        return 0
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(URL, headers={"User-Agent": "touch-research-data-audit/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
            total = response.headers.get("Content-Length")
            shutil.copyfileobj(response, out)
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Downloaded: {target}")
    print(f"Bytes: {target.stat().st_size}")
    if total:
        print(f"HTTP Content-Length: {total}")
    print(f"SHA-256: {sha256(target)}")
    print(f"Next: python scripts/audit_dataset.py --archive {target}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

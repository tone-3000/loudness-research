#!/usr/bin/env python3
"""
Download the .nam files of the corpus into data/models/<anon_id>.nam
(git-ignored).  Needs the private mapping (data/private/mapping_models.csv,
written by anonymize.py) because the public corpus carries no URLs.
Idempotent; skips files that already exist and parse as JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS_DIR, PRIVATE  # noqa: E402


def fetch(row: dict) -> tuple[str, str]:
    dst = MODELS_DIR / f"{row['anon_id']}.nam"
    if dst.exists():
        try:
            json.load(dst.open())
            return row["anon_id"], "cached"
        except Exception:  # noqa: BLE001
            dst.unlink()
    try:
        with urllib.request.urlopen(row["model_url"], timeout=120) as r:
            data = r.read()
        json.loads(data)  # validate
        dst.write_bytes(data)
        return row["anon_id"], f"ok {len(data) // 1024} KB"
    except Exception as e:  # noqa: BLE001
        return row["anon_id"], f"FAIL {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default=str(PRIVATE / "mapping_models.csv"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not Path(args.mapping).exists():
        sys.exit(f"{args.mapping} not found: the model URLs are not part of the public repository")
    rows = list(csv.DictReader(open(args.mapping)))
    fails = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch, r) for r in rows]
        for i, f in enumerate(as_completed(futs), 1):
            mid, status = f.result()
            if status.startswith("FAIL"):
                fails += 1
                print(f"[{i}/{len(rows)}] {mid}: {status}", flush=True)
            elif i % 25 == 0:
                print(f"[{i}/{len(rows)}] ...", flush=True)
    print(f"done; {fails} failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())

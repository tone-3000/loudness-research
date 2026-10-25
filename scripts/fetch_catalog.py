#!/usr/bin/env python3
"""
Fetch the TONE3000 A2 model catalog used to build the experimental corpus.

Pages through an internal TONE3000 catalogue endpoint (not public; the URL
and access code are read from the environment and are *not* in this
repository) and writes a compact JSONL (one model per line) with only the
fields the study needs.  The endpoint returns architecture-v2 (A2 /
SlimmableContainer) models from verified creators, each with its parent tone
(gear type, tags, makes) embedded.

Usage:
    export T3K_CATALOG_API="https://.../models/..."   # paged: ?code=..&page=N&page_size=500
    export T3K_CATALOG_CODE="..."
    python fetch_catalog.py [--out data/private/catalog.jsonl] [--max-pages N]

The output is git-ignored. Without access to the endpoint, the published
corpus (data/corpus.csv) and aggregated results (results/*.csv.gz) are
sufficient to re-run every analysis in this repository.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import os

API = os.environ.get("T3K_CATALOG_API", "")
CODE = os.environ.get("T3K_CATALOG_CODE", "")
PAGE_SIZE = 500

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def fetch_page(page: int, retries: int = 4) -> dict:
    qs = urllib.parse.urlencode({"code": CODE, "page": page, "page_size": PAGE_SIZE})
    url = f"{API}?{qs}"
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"page {page} failed after {retries} attempts: {last_err}")


def compact(row: dict) -> dict:
    tone = row.get("tone") or {}
    training = row.get("training") or {}
    return {
        "id": row["id"],
        "user_id": row.get("user_id"),
        "name": row.get("name"),
        "model_url": row.get("model_url"),
        "input_level": row.get("input_level"),
        "output_level": row.get("output_level"),
        "created_at": row.get("created_at"),
        "training_type": training.get("type"),
        "esrs": training.get("esrs"),
        "tone_id": tone.get("id"),
        "tone_title": tone.get("title"),
        "gear": tone.get("gear"),
        "tags": tone.get("tags") or [],
        "makes": tone.get("makes") or [],
        "description": (tone.get("description") or "")[:400],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "private" / "catalog.jsonl"))
    ap.add_argument("--max-pages", type=int, default=None)
    args = ap.parse_args()
    if not API or not CODE:
        sys.exit("set T3K_CATALOG_API and T3K_CATALOG_CODE (internal TONE3000 endpoint; not public)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    first = fetch_page(1)
    total_pages = first["total_pages"]
    total = first["total"]
    if args.max_pages:
        total_pages = min(total_pages, args.max_pages)
    print(f"total models={total} total_pages={total_pages} page_size={PAGE_SIZE}")

    # Resume: pages are written atomically as whole pages, so count complete
    # pages already on disk and continue from there.
    done_rows = 0
    if out.exists():
        with out.open() as fp:
            done_rows = sum(1 for _ in fp)
    start_page = done_rows // PAGE_SIZE + 1
    # Truncate any partial trailing page.
    if done_rows % PAGE_SIZE:
        keep = (done_rows // PAGE_SIZE) * PAGE_SIZE
        lines = out.read_text().splitlines()[:keep]
        out.write_text("\n".join(lines) + ("\n" if lines else ""))
        done_rows = keep
    print(f"resuming at page {start_page} (rows on disk: {done_rows})")

    n = done_rows
    with out.open("a") as fp:
        for page in range(start_page, total_pages + 1):
            t0 = time.time()
            data = first if page == 1 else fetch_page(page)
            fp.write("".join(json.dumps(compact(row)) + "\n" for row in data["data"]))
            n += len(data["data"])
            fp.flush()
            print(f"page {page}/{total_pages}  rows={n}  ({time.time() - t0:.1f}s)", flush=True)
    print(f"wrote {n} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

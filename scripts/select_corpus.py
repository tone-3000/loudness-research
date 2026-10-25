#!/usr/bin/env python3
"""
Stratified, creator-capped corpus from the TONE3000 catalogue (private input,
see fetch_catalog.py).

Rules
  * all seven TONE3000 gear types; `cab` has only 6 models in the catalogue,
    all taken; `experimental` sampled;
  * 1 model per tone (pack);
  * at most MAX_PER_CREATOR models from any creator (the catalogue is 56 %
    one creator);
  * bass deliberately oversampled (2.5 % of the catalogue) so it can be
    evaluated; the instrument label is for sampling/reporting only;
  * optionally disjoint (by model and tone) from an earlier corpus;
  * seeded.

Output: data/private/corpus_tone3000.csv (with TONE3000 ids / urls), which
anonymize.py turns into the public data/corpus.csv.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PRIVATE, iter_jsonl  # noqa: E402

_BASS_RE = re.compile(r"\bbass\b", re.I)


def instrument(row: dict) -> str:
    if "bass" in (row.get("tags") or []):
        return "bass"
    text = " ".join([row.get("tone_title") or "", " ".join(row.get("makes") or [])])
    return "bass" if _BASS_RE.search(text) else "guitar"


TARGETS = {
    ("amp", "guitar"): 55,
    ("amp", "bass"): 25,
    ("amp-cab", "guitar"): 70,
    ("amp-cab", "bass"): 25,
    ("pedal", "guitar"): 30,
    ("pedal", "bass"): 15,
    ("outboard", "guitar"): 25,
    ("outboard", "bass"): 10,
    ("experimental", "guitar"): 18,
    ("experimental", "bass"): 2,
    ("cab", "guitar"): 6,
}
MAX_PER_CREATOR = 22  # ~8 % of the corpus


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=str(PRIVATE / "catalog.jsonl"))
    ap.add_argument("--exclude", default="", help="earlier corpus csv to stay disjoint from (model and tone)")
    ap.add_argument("--out", default=str(PRIVATE / "corpus_tone3000.csv"))
    ap.add_argument("--seed", type=int, default=20261023)
    args = ap.parse_args()

    exclude_ids = set()
    exclude_tones = set()
    if args.exclude and Path(args.exclude).exists():
        for r in csv.DictReader(open(args.exclude)):
            exclude_ids.add(str(r["model_id"]))
            exclude_tones.add(str(r["tone_id"]))

    rng = random.Random(args.seed)
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in iter_jsonl(args.catalog):
        if not r.get("model_url") or not r["model_url"].endswith(".nam"):
            continue
        if str(r["id"]) in exclude_ids or str(r["tone_id"]) in exclude_tones:
            continue
        key = (r.get("gear"), instrument(r))
        if key in TARGETS:
            buckets[key].append(r)

    per_creator: dict[str, int] = defaultdict(int)
    used_tones: set = set()
    chosen: list[dict] = []
    # Fill the scarce strata first so the creator cap does not starve them.
    order = sorted(TARGETS, key=lambda k: len(buckets[k]))
    for key in order:
        n = TARGETS[key]
        pool = buckets[key]
        rng.shuffle(pool)
        picked = []
        for r in pool:
            if r["tone_id"] in used_tones:
                continue
            if per_creator[r["user_id"]] >= MAX_PER_CREATOR and key[0] != "cab":
                continue
            used_tones.add(r["tone_id"])
            per_creator[r["user_id"]] += 1
            picked.append(r)
            if len(picked) >= n:
                break
        print(f"{key[0]:13s} {key[1]:6s}: pool={len(pool):5d} picked={len(picked)}")
        for r in picked:
            chosen.append(
                {
                    "model_id": r["id"],
                    "gear": key[0],
                    "instrument": key[1],
                    "tone_id": r["tone_id"],
                    "user_id": r["user_id"],
                    "training_type": r.get("training_type"),
                    "input_level": r.get("input_level"),
                    "name": r.get("name"),
                    "tone_title": r.get("tone_title"),
                    "tags": "|".join(r.get("tags") or []),
                    "model_url": r["model_url"],
                }
            )

    out = Path(args.out)
    with out.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(chosen[0].keys()))
        w.writeheader()
        w.writerows(chosen)
    top = sorted(per_creator.items(), key=lambda kv: -kv[1])[:5]
    print(f"wrote {len(chosen)} rows -> {out}; creators={len(per_creator)}; top creator counts={[v for _, v in top]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

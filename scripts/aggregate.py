#!/usr/bin/env python3
"""
Aggregate the per-model experiment output (results/raw/<id>.json, git-ignored,
~17 MB) into the compact tables that are committed and that analyze.py reads:

    results/models.csv          one row per model (gear, instrument, creator label, stamped value ...)
    results/candidates.csv.gz   model_id, stim, measure, value      (every candidate calculation)
    results/bursts.csv.gz       model_id, stim, measure, burst, value (per-burst levels of the sweep stimuli)
    results/gold.csv.gz         model_id, clip, clip_instrument, offset, ir, measure, value

    python scripts/aggregate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS  # noqa: E402

RAW = RESULTS / "raw"
MODEL_FIELDS = ("model_id", "gear", "instrument", "training_type", "creator", "stamped_loudness", "stamped_gain", "nam_version", "recomputed_current", "seconds")


def main() -> int:
    models, cands, bursts, gold = [], [], [], []
    files = sorted(RAW.glob("*.json"))
    for p in files:
        r = json.loads(p.read_text())
        mid = r["model_id"]
        models.append({k: r.get(k) for k in MODEL_FIELDS})
        for key, m in r["candidates"].items():
            for meas, v in m.items():
                if isinstance(v, list):
                    bursts.extend({"model_id": mid, "stim": key, "measure": meas, "burst": i, "value": x} for i, x in enumerate(v))
                else:
                    cands.append({"model_id": mid, "stim": key, "measure": meas, "value": v})
        for key, m in r["gold"].items():
            clip_off, _, irn = key.partition("|")
            clip, off = clip_off.rsplit("@", 1)
            for meas, v in m.items():
                if meas in ("instrument", "ir"):
                    continue
                gold.append({"model_id": mid, "clip": clip, "clip_instrument": m["instrument"], "offset": float(off), "ir": irn or "", "measure": meas, "value": v})
    pd.DataFrame(models).to_csv(RESULTS / "models.csv", index=False)
    pd.DataFrame(cands).to_csv(RESULTS / "candidates.csv.gz", index=False, float_format="%.5f")
    pd.DataFrame(bursts).to_csv(RESULTS / "bursts.csv.gz", index=False, float_format="%.5f")
    pd.DataFrame(gold).to_csv(RESULTS / "gold.csv.gz", index=False, float_format="%.5f")
    print(f"{len(files)} models -> {len(cands)} candidate values, {len(bursts)} burst values, {len(gold)} gold values")
    return 0


if __name__ == "__main__":
    sys.exit(main())

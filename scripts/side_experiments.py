#!/usr/bin/env python3
"""
Cheap side experiments (no psychoacoustic model, ~1 s per model):

  1. DC offset: how much does the -28 dBFS DC offset in the shipped
     loudness_input.wav change the stamped value?  (flat RMS of the response to
     the file as shipped vs. the same file with its mean removed)

  2. Warm-up / settling: is 1 s long enough?  Feed 6 s of stationary
     instrument-shaped noise and report the RMS of each 1 s segment relative
     to the last one.  Models with slow sag / recovery (issue #588) drift.

  3. Linearity in the stamped value's own terms: response RMS vs input level
     for the shipped pink stimulus (-27 .. -3 dBFS RMS), i.e. how far the
     current number is from being a property of the model rather than of the
     stimulus level.

Writes results/side_experiments.csv
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, MODELS_DIR, RESULTS, SR, current_stimulus, load_model, run_model  # noqa: E402
from measures import rms_flat  # noqa: E402
from candidates import shaped_noise  # noqa: E402
from stimuli import set_rms  # noqa: E402


def main() -> int:
    import torch

    torch.set_num_threads(1)
    rows = list(csv.DictReader(open(DATA / "corpus.csv")))
    x0 = current_stimulus()
    x_nodc = x0 - x0.mean()
    noise6 = set_rms(shaped_noise("inst", 6.0), -15.0)
    out = []
    for i, r in enumerate(rows):
        p = MODELS_DIR / f"{r['model_id']}.nam"
        if not p.exists():
            continue
        model, d = load_model(p)
        y0 = run_model(model, x0)
        y1 = run_model(model, x_nodc)
        rec = {
            "model_id": r["model_id"],
            "gear": r["gear"],
            "instrument": r["instrument"],
            "L_current": rms_flat(y0),
            "L_nodc": rms_flat(y1),
            "dc_effect_db": rms_flat(y1) - rms_flat(y0),
            "out_dc_dbfs": 20 * np.log10(abs(float(y0.mean())) + 1e-12),
        }
        y6 = run_model(model, noise6)
        segs = [rms_flat(y6[k * SR : (k + 1) * SR]) for k in range(6)]
        for k, s in enumerate(segs):
            rec[f"seg{k}_rel_last_db"] = s - segs[-1]
        rec["settle_drift_db"] = segs[0] - segs[-1]
        for lvl in (-27, -21, -15, -9, -3):
            y = run_model(model, x_nodc * 10 ** ((lvl + 15) / 20))
            rec[f"pink_out_at_{lvl}"] = rms_flat(y)
        rec["pink_slope_27_15"] = (rec["pink_out_at_-15"] - rec["pink_out_at_-27"]) / 12
        rec["pink_slope_15_3"] = (rec["pink_out_at_-3"] - rec["pink_out_at_-15"]) / 12
        out.append(rec)
        if (i + 1) % 25 == 0:
            print(f"{i+1}/{len(rows)}", flush=True)
    with (RESULTS / "side_experiments.csv").open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    a = np.array([o["dc_effect_db"] for o in out])
    s = np.array([o["settle_drift_db"] for o in out])
    print(f"DC-offset effect on stamped value: mean {a.mean():+.3f} dB, SD {a.std():.3f}, max |Δ| {np.abs(a).max():.2f} dB")
    print(f"first-second vs last-second drift: mean {s.mean():+.3f} dB, SD {s.std():.3f}, max |Δ| {np.abs(s).max():.2f} dB, >0.5 dB in {100*np.mean(np.abs(s)>0.5):.1f}% of models")
    return 0


if __name__ == "__main__":
    sys.exit(main())

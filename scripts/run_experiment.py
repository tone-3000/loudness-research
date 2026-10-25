#!/usr/bin/env python3
"""
Main experiment: quick parametric noise candidates, level-agnostic gold.

Per model -> results/raw/<id>.json
  candidates: "{shape}|{env}@{level}[|{seconds}s]" -> {flat, hp80, K, A}
              sweeps additionally carry per-burst values and the derived
              mean-of-dB / median / energy-mean summaries.
  gold:       "{clip}@{offset}[|{ir}]" -> {flat, K, LUFS [, N5_phon, LLeq_phon]}
              offsets -12..+12 dB around the DI's native level; Zwicker on a
              subset of clips at -12 / 0 / +12.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, MODELS_DIR, RESULTS, SR, load_model, metadata_loudness_stamped, run_model, current_stimulus  # noqa: E402
from measures import apply_ir, ir_inband_gain_db, lufs_integrated, rms_a, rms_flat, rms_hp, rms_k, zwicker_phon  # noqa: E402
from stimuli import BASS_IRS, GOLD_BASS, GOLD_GUITAR, GUITAR_IRS, di, ir, _loudest_window  # noqa: E402
from candidates import SHAPES, STEADY_LEVELS, SWEEPS, burst_slices, stimulus  # noqa: E402

RAW = RESULTS / "raw"
GOLD_OFFSETS_DB = [-12.0, -6.0, 0.0, 6.0, 12.0]
GOLD_EXCERPT_S = 6.0
ZWICKER_CLIPS = {"Brit - Guitar", "Metalcore - Guitar", "Jazz Trot - Guitar", "Downtown - Bass", "Garden - Bass"}
ZWICKER_OFFSETS = {-12.0, 0.0, 12.0}
ZWICKER_IRS = {"celestion", "ampeg"}
ENVELOPES = ["steady", "pluck"]
LENGTH_TEST = {"shape": "ip1k", "seconds": [0.5, 2.0, 4.0], "levels": [-30.0, -24.0, -18.0]}


def measures(y: np.ndarray) -> dict:
    return {"flat": rms_flat(y), "hp80": rms_hp(y, 80.0), "K": rms_k(y), "A": rms_a(y)}


def sweep_measures(y: np.ndarray, env: str) -> dict:
    out = {}
    per = {k: [] for k in ("flat", "hp80", "K", "A")}
    interior = []
    for sl in burst_slices(env):
        seg = y[sl]
        interior.append(seg)
        for k, v in measures(seg).items():
            per[k].append(v)
    cat = np.concatenate(interior)
    for k, vals in per.items():
        vals = np.array(vals)
        out[f"{k}_meandb"] = float(vals.mean())
        out[f"{k}_median"] = float(np.median(vals))
        out[f"{k}_energy"] = measures(cat)[k]
        out[f"{k}_bursts"] = [float(v) for v in vals]
    return out


def gold_measures(y: np.ndarray, zwicker: bool) -> dict:
    d = {"flat": rms_flat(y), "K": rms_k(y), "LUFS": lufs_integrated(y)}
    if zwicker:
        z = zwicker_phon(y)
        d.update({"N5_phon": z["N5_phon"], "LLeq_phon": z["LLeq_phon"]})
    return d


def _ir_cache() -> dict:
    out = {}
    for name in GUITAR_IRS + BASS_IRS:
        taps = ir(name)
        out[name] = (taps, ir_inband_gain_db(taps))
    return out


def process_model(row: dict, zwicker: bool) -> dict:
    import torch

    torch.set_num_threads(1)
    t0 = time.time()
    mid = row["model_id"]
    model, d = load_model(MODELS_DIR / f"{mid}.nam")
    md = d.get("metadata") or {}
    res = {
        "model_id": mid,
        "gear": row["gear"],
        "instrument": row["instrument"],
        "training_type": row.get("training_type"),
        "creator": row.get("creator"),
        "stamped_loudness": metadata_loudness_stamped(d),
        "stamped_gain": md.get("gain"),
        "nam_version": d.get("version"),
        "candidates": {},
        "gold": {},
    }
    # baseline: the shipped file as-is
    y = run_model(model, current_stimulus())
    res["candidates"]["current|shipped@-15"] = measures(y)
    res["recomputed_current"] = res["candidates"]["current|shipped@-15"]["flat"]

    for shape in SHAPES:
        for env in ENVELOPES:
            for lvl in STEADY_LEVELS:
                y = run_model(model, stimulus(shape, env, lvl, 1.0))
                res["candidates"][f"{shape}|{env}@{lvl:g}"] = measures(y)
        for sw in SWEEPS:
            y = run_model(model, stimulus(shape, sw, 0.0))
            res["candidates"][f"{shape}|{sw}@0"] = sweep_measures(y, sw)
    for sec in LENGTH_TEST["seconds"]:
        for lvl in LENGTH_TEST["levels"]:
            y = run_model(model, stimulus(LENGTH_TEST["shape"], "steady", lvl, sec))
            res["candidates"][f"{LENGTH_TEST['shape']}|steady@{lvl:g}|{sec:g}s"] = measures(y)

    irs = _ir_cache()
    for inst, clips, ir_names in (("guitar", GOLD_GUITAR, GUITAR_IRS), ("bass", GOLD_BASS, BASS_IRS)):
        for clip in clips:
            x0 = _loudest_window(di(clip), GOLD_EXCERPT_S)
            for off in GOLD_OFFSETS_DB:
                y = run_model(model, x0 * 10 ** (off / 20))
                zw = zwicker and clip in ZWICKER_CLIPS and off in ZWICKER_OFFSETS
                key = f"{clip}@{off:+g}"
                res["gold"][key] = {"instrument": inst, "ir": None, **gold_measures(y, zw)}
                if row["gear"] == "amp":
                    for irn in ir_names:
                        taps, g = irs[irn]
                        yi = apply_ir(y, taps, g)
                        res["gold"][f"{key}|{irn}"] = {"instrument": inst, "ir": irn, **gold_measures(yi, zw and irn in ZWICKER_IRS)}
    res["seconds"] = time.time() - t0
    return res


def _worker(row: dict, zwicker: bool) -> tuple[str, str]:
    out = RAW / f"{row['model_id']}.json"
    try:
        res = process_model(row, zwicker)
        out.write_text(json.dumps(res))
        return row["model_id"], f"ok {res['seconds']:.0f}s"
    except Exception:  # noqa: BLE001
        err = traceback.format_exc()
        (RAW / f"{row['model_id']}.error.txt").write_text(err)
        return row["model_id"], "FAIL " + err.strip().splitlines()[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(DATA / "corpus.csv"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-zwicker", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ids", default="")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.corpus)))
    rows = [r for r in rows if (MODELS_DIR / f"{r['model_id']}.nam").exists()]
    if args.ids:
        want = set(args.ids.split(","))
        rows = [r for r in rows if r["model_id"] in want]
    if not args.force:
        rows = [r for r in rows if not (RAW / f"{r['model_id']}.json").exists()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} models to process, workers={args.workers}, zwicker={not args.no_zwicker}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(_worker, r, not args.no_zwicker) for r in rows]
        for i, f in enumerate(as_completed(futs), 1):
            mid, status = f.result()
            el = (time.time() - t0) / 60
            print(f"[{i}/{len(rows)}] {mid}: {status}   elapsed {el:.1f} min, eta {el / i * (len(rows) - i):.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

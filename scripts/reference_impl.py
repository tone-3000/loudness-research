#!/usr/bin/env python3
"""
Reference implementation of the recommended calculation: an in-place
improvement of `metadata.loudness` (same key, same units, same median).

    loudness(model) = CALIBRATION_DB + mean_b [ -0.691 + 10*log10( mean( K(y_b)^2 ) ) ]

where
  * the stimulus is a 1.75 s "level-sweep burst train": 7 bursts of 0.25 s of
    instrument-shaped pink noise (pink × HP2 70 Hz × LP2 1 kHz), at
    -39, -35, -31, -27, -23, -19, -15 dBFS RMS, 10 ms raised-cosine edges
    (results/stimulus/loudness_input_proposed.wav);
  * y = model(stimulus); y_b is the response to burst b with 30 ms trimmed
    off each edge;
  * K is the ITU-R BS.1770 K-weighting pre-filter at 48 kHz, applied once
    to the whole response;
  * the per-burst K levels (dB) are averaged in dB (mean over levels), not
    in energy - this is what makes the value level-agnostic;
  * CALIBRATION_DB puts the catalog median back where today's calculation
    has it, so engines that already do `gain = target - loudness` with a
    -18 dB target keep working unchanged.

Usage:
    python scripts/reference_impl.py --write-stimulus
    python scripts/reference_impl.py data/models/m001.nam
    python scripts/reference_impl.py --check        # vs results/raw/*.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, SR, load_model, metadata_loudness_stamped, run_model  # noqa: E402

STIMULUS_PATH = RESULTS / "stimulus" / "loudness_input_proposed.wav"
STIMULUS_SHAPE = "ip1k"
STIMULUS_ENV = "sweep7_-39_-15"
BURST_S = 0.25
EDGE_TRIM_S = 0.03
N_BURSTS = 7
CALIBRATION_DB = 3.0  # median(L_old) - median(L_new_raw) = +3.07 dB on the corpus

# ITU-R BS.1770-4 K-weighting, 48 kHz coefficients.
_K_STAGE1_B = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
_K_STAGE1_A = np.array([1.0, -1.69065929318241, 0.73248077421585])
_K_STAGE2_B = np.array([1.0, -2.0, 1.0])
_K_STAGE2_A = np.array([1.0, -1.99004745483398, 0.99007225036621])


def k_weight(x: np.ndarray) -> np.ndarray:
    return lfilter(_K_STAGE2_B, _K_STAGE2_A, lfilter(_K_STAGE1_B, _K_STAGE1_A, x))


def k_level_db(y: np.ndarray) -> float:
    return -0.691 + 10.0 * np.log10(max(float(np.mean(np.square(y))), 1e-24))


def burst_slices() -> list[slice]:
    n, k = int(BURST_S * SR), int(EDGE_TRIM_S * SR)
    return [slice(i * n + k, (i + 1) * n - k) for i in range(N_BURSTS)]


def loudness_proposed(model, calibrated: bool = True) -> float:
    y = k_weight(run_model(model, load_stimulus()))
    per_burst = [k_level_db(y[sl]) for sl in burst_slices()]
    return float(np.mean(per_burst)) + (CALIBRATION_DB if calibrated else 0.0)


def write_stimulus() -> None:
    from candidates import stimulus

    x = stimulus(STIMULUS_SHAPE, STIMULUS_ENV, 0.0).astype(np.float32)
    STIMULUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    sf.write(STIMULUS_PATH, x, SR, subtype="FLOAT")
    rms = 20 * np.log10(np.sqrt(np.mean(x**2)))
    print(f"wrote {STIMULUS_PATH}  ({len(x)/SR:.2f} s, RMS {rms:.2f} dBFS, peak {20*np.log10(np.abs(x).max()):.2f} dBFS)")


def load_stimulus() -> np.ndarray:
    x, sr = sf.read(STIMULUS_PATH, dtype="float64")
    assert sr == SR, sr
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("nam", nargs="*")
    ap.add_argument("--write-stimulus", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.write_stimulus or not STIMULUS_PATH.exists():
        write_stimulus()
    if args.check:
        import json

        files = sorted((RESULTS / "raw").glob("*.json"))
        for p in files[:: max(1, len(files) // 10)]:
            r = json.loads(p.read_text())
            ref = r["candidates"][f"{STIMULUS_SHAPE}|{STIMULUS_ENV}@0"]["K_meandb"]
            model, _ = load_model(RESULTS.parent / "data" / "models" / f"{r['model_id']}.nam")
            v = loudness_proposed(model, calibrated=False)
            print(f"{r['model_id']}: reference {v:8.3f}  experiment {ref:8.3f}  Δ {v - ref:+.4f}")
        return 0
    for p in args.nam:
        model, d = load_model(p)
        v1 = metadata_loudness_stamped(d)
        v2 = loudness_proposed(model)
        print(f"{Path(p).name}: loudness (current, stamped) {v1:8.3f} dB   loudness (proposed) {v2:8.3f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

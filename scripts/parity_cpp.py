#!/usr/bin/env python3
"""
Parity check: does the number we compute in PyTorch describe what the C++
engines actually output?

Renders the same stimuli through (a) the trainer's PyTorch model, (b) the
upstream NeuralAmpModelerCore `render` tool (generic WaveNet), and (c) the
TONE3000 core `render` tool built with NAM_ENABLE_A2_FAST (the A2 fast path
used by the plugin / pedal / wasm player), and compares output level and
sample-level error.

Usage:
    python parity_cpp.py --generic /path/to/NeuralAmpModelerCore/build/tools/render \
                         --fast    /tmp/namcore-fast/tools/render  [--n 8]

Writes results/parity_cpp.csv.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, MODELS_DIR, RESULTS, SR, load_model, run_model  # noqa: E402
from measures import lufs_integrated, rms_flat  # noqa: E402
from common import current_stimulus  # noqa: E402
from candidates import stimulus  # noqa: E402
from stimuli import gold_clip  # noqa: E402


def cpp_render(tool: str, model: Path, x: np.ndarray, tmp: Path) -> np.ndarray:
    xin = tmp / "in.wav"
    out = tmp / "out.wav"
    sf.write(xin, x.astype(np.float32), SR, subtype="FLOAT")
    r = subprocess.run([tool, str(model), str(xin), str(out)], capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"{tool} failed: {r.stderr[-400:]}")
    y, fs = sf.read(out, dtype="float64")
    if y.ndim > 1:
        y = y[:, 0]
    return y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generic", default=None, help="upstream NeuralAmpModelerCore render tool (optional; cannot load A2 files at the time of the study)")
    ap.add_argument("--fast", required=True)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(DATA / "corpus.csv")))
    # spread across strata
    by = {}
    for r in rows:
        by.setdefault((r["gear"], r["instrument"]), []).append(r)
    picks = [v[0] for v in by.values()][: args.n]

    stims = {
        "current stimulus": current_stimulus(),
        "proposed stimulus": stimulus("ip1k", "sweep7_-39_-15", 0.0),
        "Brit - Guitar": gold_clip("Brit - Guitar"),
        "Garden - Bass": gold_clip("Garden - Bass"),
    }
    out_rows = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for r in picks:
            mp = MODELS_DIR / f"{r['model_id']}.nam"
            model, _ = load_model(mp)
            for sname, x in stims.items():
                y_py = run_model(model, x)
                for tool_name, tool in (("generic", args.generic), ("fast", args.fast)):
                    if not tool:
                        continue
                    try:
                        y_c = cpp_render(tool, mp, x, tmp)
                    except Exception as e:  # noqa: BLE001
                        print(f"{r['model_id']} {sname} {tool_name}: {e}")
                        continue
                    n = min(len(y_py), len(y_c))
                    a, b = y_py[:n], y_c[:n]
                    # skip the first receptive field where the C++ tool's warm-up may differ
                    s = int(getattr(model, "receptive_field", 1))
                    a2, b2 = a[s:], b[s:]
                    err = b2 - a2
                    out_rows.append(
                        {
                            "model_id": r["model_id"],
                            "gear": r["gear"],
                            "stimulus": sname,
                            "tool": tool_name,
                            "rms_py_db": rms_flat(a2),
                            "rms_cpp_db": rms_flat(b2),
                            "rms_delta_db": rms_flat(b2) - rms_flat(a2),
                            "lufs_delta_db": lufs_integrated(b2) - lufs_integrated(a2),
                            "err_rms_db_rel": rms_flat(err) - rms_flat(a2),
                            "max_abs_err": float(np.max(np.abs(err))),
                            "len_py": len(y_py),
                            "len_cpp": len(y_c),
                        }
                    )
                    print(f"{r['model_id']:>7} {r['gear']:8s} {sname:18s} {tool_name:7s} Δrms={out_rows[-1]['rms_delta_db']:+.4f} dB  err={out_rows[-1]['err_rms_db_rel']:.1f} dB rel  maxabs={out_rows[-1]['max_abs_err']:.2e}")
    with (RESULTS / "parity_cpp.csv").open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    d = np.array([r["rms_delta_db"] for r in out_rows])
    e = np.array([r["err_rms_db_rel"] for r in out_rows])
    print(f"\nlevel delta C++ vs PyTorch: mean {d.mean():+.4f} dB, max |Δ| {np.abs(d).max():.4f} dB; sample error {np.median(e):.0f} dB below signal (median)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Shared plumbing for the loudness study: paths, constants, .nam loading, and
running audio through a model.

Everything here is deliberately independent of the trainer's training code;
only `nam.models.init_from_nam` is reused so that the models evaluated are
bit-identical to what the trainer exports.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # repository root
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
MODELS_DIR = DATA / "models"  # git-ignored .nam downloads
PRIVATE = DATA / "private"  # git-ignored: mapping from anonymised ids to TONE3000 ids

# Test material (committed): TONE3000 demo DIs and cabinet IRs, from
# https://github.com/tone-3000/neural-amp-modeler-wasm (ui/public/{inputs,irs}).
DI_DIR = DATA / "audio" / "inputs"
IR_DIR = DATA / "audio" / "irs"

SR = 48_000
# The stimulus shipped with the trainer (nam/models/_resources/loudness_input.wav).
CURRENT_STIMULUS = DATA / "stimulus" / "loudness_input_current.wav"

# Engine conventions (plugin, diy-pedal, wasm are all identical here).
ENGINE_TARGET_DB = -18.0
ENGINE_CLAMP_DB = 12.0

# Playback calibration for the psychoacoustic (ISO 532-1) reference:
# a -18 dBFS RMS signal is reproduced at 85 dB SPL, i.e. 0 dBFS RMS = 103 dB SPL.
# (K-20 convention is -20 dBFS pink = 83 dB SPL/channel; this is equivalent.)
SPL_AT_0DBFS_RMS = 103.0
P_REF = 20e-6


def dbfs_to_pa(x: np.ndarray) -> np.ndarray:
    return x * P_REF * 10 ** (SPL_AT_0DBFS_RMS / 20.0)


def db(x: float) -> float:
    return 20.0 * np.log10(max(float(x), 1e-12))


def rms_db(x: np.ndarray) -> float:
    return 10.0 * np.log10(max(float(np.mean(np.square(x))), 1e-24))


# --------------------------------------------------------------------------
# Audio I/O
# --------------------------------------------------------------------------
def load_wav(path: Path | str, sr: int = SR, mono: bool = True) -> np.ndarray:
    x, fs = sf.read(str(path), dtype="float64", always_2d=True)
    if mono:
        x = x[:, 0]
    else:
        x = x.T
    if fs != sr:
        from math import gcd

        g = gcd(fs, sr)
        x = resample_poly(x, sr // g, fs // g, axis=-1)
    return np.ascontiguousarray(x)


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def _highest_quality_submodel(d: dict) -> dict:
    subs = d["config"]["submodels"]
    return max(subs, key=lambda s: s["max_value"])["model"]


def read_nam_json(path: Path | str) -> dict:
    with open(path) as fp:
        return json.load(fp)


def load_model(path: Path | str) -> tuple[torch.nn.Module, dict]:
    """
    Load a .nam file into a PyTorch module.

    For A2 (`SlimmableContainer`) files the highest-quality submodel is used;
    this is what the trainer syncs the container's metadata.loudness from
    (`PackedWaveNet._sync_container_metadata_to_highest_quality_submodel`),
    and it is what the engines run at size 1.0.

    Returns (model, top_level_json).
    """
    from nam.models import init_from_nam

    d = read_nam_json(path)
    cfg = _highest_quality_submodel(d) if d["architecture"] == "SlimmableContainer" else d
    if "sample_rate" not in cfg and "sample_rate" in d:
        cfg = dict(cfg, sample_rate=d["sample_rate"])
    model = init_from_nam(cfg)
    model.eval()
    return model, d


@torch.no_grad()
def run_model(model: torch.nn.Module, x: np.ndarray, chunk_s: float = 30.0) -> np.ndarray:
    """
    Run a mono float array through the model with the trainer's own forward
    (pad_start=True so the output has the same length as the input, exactly as
    `_metadata_loudness` does).  Long inputs are processed in overlapping
    chunks that carry the receptive field so results are identical to a single
    pass.
    """
    x = np.asarray(x, dtype=np.float32)
    rf = int(getattr(model, "receptive_field", 1))
    n = len(x)
    chunk = int(chunk_s * SR)
    if n <= chunk:
        y = model(torch.from_numpy(x), pad_start=True)
        return y.detach().cpu().numpy().astype(np.float64)
    out = np.empty(n, dtype=np.float64)
    start = 0
    while start < n:
        stop = min(n, start + chunk)
        lo = max(0, start - (rf - 1))
        xi = torch.from_numpy(x[lo:stop])
        yi = model(xi, pad_start=(lo == 0)).detach().cpu().numpy()
        # yi has length stop-lo if padded, else stop-lo-(rf-1)
        out[start:stop] = yi[-(stop - start):]
        start = stop
    return out


def metadata_loudness_stamped(d: dict) -> float | None:
    md = d.get("metadata") or {}
    v = md.get("loudness")
    return float(v) if v is not None else None


# --------------------------------------------------------------------------
# Corpus helpers
# --------------------------------------------------------------------------
def iter_jsonl(path: Path | str) -> Iterable[dict]:
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if line:
                yield json.loads(line)


@lru_cache(maxsize=None)
def current_stimulus() -> np.ndarray:
    return load_wav(CURRENT_STIMULUS)

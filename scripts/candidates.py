"""
Candidate stimuli: short noises, fully parametric and deterministic.

A stimulus is   noise(shape)  ×  envelope  ×  level.

shape (spectral):
    pink          1/f, DC-free
    pink_hp80     pink, 2nd-order Butterworth HP 80 Hz           (issue #588)
    ip500/ip1k/ip2k
                  "instrument pink": pink × HP2(70 Hz) × LP2(fc), fc = 500 Hz / 1 kHz / 2 kHz
                  (the mean guitar+bass DI spectrum is ≈ pink × HP2(70) × LP2(1 kHz))
    ip1k_o4       same as ip1k but 4th-order LP
    inst          non-parametric: noise shaped to the measured mean DI spectrum
                  (reference for how close the parametric shapes get)

envelope:
    steady        constant
    pluck         plucked-note envelope, 2 notes/s, 350 ms decay, ±3 dB velocity
    sweepN_lo_hi  burst train: N bursts of BURST_S seconds, levels log-spaced
                  from lo to hi dBFS RMS; 10 ms raised-cosine edges.  The
                  per-burst levels are absolute (the overall `level` argument
                  is ignored) - this is the "don't assume a player level"
                  stimulus.

Everything is generated from one fixed Gaussian noise seed, so a stimulus
can be regenerated bit-exactly from its description (plus the seed).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import butter, sosfilt

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SR, rms_db  # noqa: E402
from stimuli import _pluck_envelope, _shape_noise, instrument_mean_spectrum, set_rms  # noqa: E402

SEED = 20261023
BURST_S = 0.25
GAP_S = 0.0


@lru_cache(maxsize=None)
def _white(seconds: float) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return rng.standard_normal(int(seconds * SR))


def _pink_from_white(w: np.ndarray) -> np.ndarray:
    X = np.fft.rfft(w)
    f = np.fft.rfftfreq(len(w), 1 / SR)
    f[0] = f[1]
    X = X / np.sqrt(f)
    X[0] = 0.0
    return np.fft.irfft(X, n=len(w))


def _sos(kind: str, fc: float, order: int):
    return butter(order, fc, btype=kind, fs=SR, output="sos")


@lru_cache(maxsize=None)
def shaped_noise(shape: str, seconds: float) -> np.ndarray:
    w = _white(seconds)
    if shape == "inst":
        f, p = instrument_mean_spectrum()
        x = _shape_noise(w, f, p)
    else:
        x = _pink_from_white(w)
        if shape == "pink":
            pass
        elif shape == "pink_hp80":
            x = sosfilt(_sos("highpass", 80.0, 2), x)
        elif shape.startswith("ip"):
            fc = {"ip500": 500.0, "ip1k": 1000.0, "ip2k": 2000.0, "ip1k_o4": 1000.0}[shape]
            order = 4 if shape.endswith("_o4") else 2
            x = sosfilt(_sos("highpass", 70.0, 2), x)
            x = sosfilt(_sos("lowpass", fc, order), x)
        else:
            raise ValueError(shape)
    x = x - x.mean()
    return x / np.sqrt(np.mean(x**2))


def _edges(n: int, edge_s: float = 0.01) -> np.ndarray:
    k = int(edge_s * SR)
    env = np.ones(n)
    r = 0.5 * (1 - np.cos(np.linspace(0, np.pi, k)))
    env[:k] = r
    env[-k:] = r[::-1]
    return env


def parse_sweep(env: str) -> tuple[int, float, float]:
    # "sweep7_-39_-15"
    parts = env.split("_")
    n = int(parts[0][len("sweep") :])
    return n, float(parts[1]), float(parts[2])


def sweep_levels(env: str) -> np.ndarray:
    n, lo, hi = parse_sweep(env)
    return np.linspace(lo, hi, n)


@lru_cache(maxsize=None)
def stimulus(shape: str, env: str, level_db: float, seconds: float = 1.0) -> np.ndarray:
    if env.startswith("sweep"):
        levels = sweep_levels(env)
        n = int(BURST_S * SR)
        base = shaped_noise(shape, BURST_S * len(levels) + 0.5)
        segs = []
        for i, lv in enumerate(levels):
            seg = base[i * n : (i + 1) * n] * _edges(n)
            segs.append(set_rms(seg, lv))
        return np.concatenate(segs)
    x = shaped_noise(shape, seconds).copy()
    if env == "pluck":
        x = x * _pluck_envelope(len(x))
    elif env != "steady":
        raise ValueError(env)
    return set_rms(x, level_db)


def burst_slices(env: str) -> list[slice]:
    """Sample ranges of each burst, excluding 30 ms at each edge."""
    n = int(BURST_S * SR)
    k = int(0.03 * SR)
    levels = sweep_levels(env)
    return [slice(i * n + k, (i + 1) * n - k) for i in range(len(levels))]


SHAPES = ["pink", "pink_hp80", "ip500", "ip1k", "ip2k", "ip1k_o4", "inst"]
STEADY_LEVELS = [-39.0, -36.0, -33.0, -30.0, -27.0, -24.0, -21.0, -18.0, -15.0, -12.0]
SWEEPS = ["sweep7_-39_-15", "sweep7_-33_-21", "sweep7_-45_-9", "sweep5_-36_-12"]

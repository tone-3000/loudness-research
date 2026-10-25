"""
Evaluation ("gold") material and shared signal helpers.

Hard constraint of the study: a candidate stimulus (see candidates.py) is ONE
fixed signal used for EVERY model, regardless of gear type or the instrument
that will later be played through it.  Instrument-specific material appears
only on the evaluation side, where every model is scored with BOTH guitar and
bass.

Gold clips are TONE3000's public demo DIs (data/audio/inputs); three more
DIs are held out of the gold set and used only, together with the gold clips,
to measure the mean guitar+bass DI spectrum that the instrument-shaped noise
candidates are built from.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import welch, firwin2, fftconvolve

from common import DI_DIR, IR_DIR, SR, load_wav, rms_db

# --------------------------------------------------------------------------
# Evaluation (gold) material - instrument split is used ONLY for evaluation
# --------------------------------------------------------------------------
GOLD_GUITAR = [
    "Brit - Guitar", "Cream - Guitar", "Fear - Guitar", "Hotrod - Guitar",
    "Mayer - Guitar", "Metalcore - Guitar", "Smooth - Guitar", "Tomb - Guitar",
    "Jazz Trot - Guitar", "Slide Lead - Guitar",
]
GOLD_BASS = ["Downtown - Bass", "Drivin' - Bass", "Garden - Bass", "Rollin' - Bass", "Smokin' - Bass"]
# Held out from gold; only these feed the music_mix candidate stimuli.
HELDOUT_CLEAN_GUITAR = "Pluck - Guitar"
HELDOUT_RHYTHM_GUITAR = "Power Thrash - Guitar"
HELDOUT_BASS = "Frogger - Bass"

GOLD_EXCERPT_S = 8.0

# Real IRs used on the evaluation side for amp-only ("amp") models.
GUITAR_IRS = ["celestion", "mesa"]
BASS_IRS = ["ampeg"]


def _loudest_window(x: np.ndarray, seconds: float) -> np.ndarray:
    n = int(seconds * SR)
    if len(x) <= n:
        return x
    hop = SR // 2
    cs = np.concatenate([[0.0], np.cumsum(np.square(x))])
    starts = np.arange(0, len(x) - n + 1, hop)
    e = cs[starts + n] - cs[starts]
    s = int(starts[np.argmax(e)])
    return x[s : s + n]


@lru_cache(maxsize=None)
def di(name: str) -> np.ndarray:
    return load_wav(DI_DIR / f"{name}.wav")


@lru_cache(maxsize=None)
def gold_clip(name: str) -> np.ndarray:
    """8 s loudest window of a DI, at the DI's native level (~ -28 dBFS RMS, peaks -8..-12 dBFS)."""
    return _loudest_window(di(name), GOLD_EXCERPT_S)


@lru_cache(maxsize=None)
def ir(name: str) -> np.ndarray:
    return load_wav(IR_DIR / f"{name}.wav")


# --------------------------------------------------------------------------
# Shared helpers for the synthetic candidate stimuli
# --------------------------------------------------------------------------
def set_rms(x: np.ndarray, target_db: float) -> np.ndarray:
    return x * 10 ** ((target_db - rms_db(x)) / 20)


@lru_cache(maxsize=None)
def instrument_mean_spectrum(nfft: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """
    Mean long-term PSD (dB) over all guitar and bass DIs, each clip RMS-equalised
    first so that no single clip dominates.  Returned as (freqs, psd_db).
    """
    names = GOLD_GUITAR + GOLD_BASS + [HELDOUT_CLEAN_GUITAR, HELDOUT_RHYTHM_GUITAR, HELDOUT_BASS]
    acc = None
    for n in names:
        x = set_rms(di(n), -20.0)
        f, p = welch(x, fs=SR, nperseg=nfft)
        acc = p if acc is None else acc + p
    acc /= len(names)
    return f, 10 * np.log10(acc + 1e-20)


def _shape_noise(white: np.ndarray, f: np.ndarray, psd_db: np.ndarray, ntaps: int = 2047) -> np.ndarray:
    mag = 10 ** (psd_db / 20)
    mag = mag / mag.max()
    fir = firwin2(ntaps, f / (SR / 2), mag)
    y = fftconvolve(white, fir, mode="same")
    return y


def _pluck_envelope(n: int, rate_hz: float = 2.0, decay_s: float = 0.35, seed: int = 7) -> np.ndarray:
    """Sequence of plucked-note envelopes: fast attack, exponential decay, slight velocity variation."""
    rng = np.random.default_rng(seed)
    env = np.zeros(n)
    t = np.arange(n) / SR
    period = 1.0 / rate_hz
    onsets = np.arange(0.0, n / SR, period)
    for o in onsets:
        v = 10 ** (rng.uniform(-3, 0) / 20)
        m = t >= o
        env[m] = np.maximum(env[m], v * np.exp(-(t[m] - o) / decay_s))
    # 5 ms attack smoothing
    k = int(0.005 * SR)
    env = np.convolve(env, np.ones(k) / k, mode="same")
    return env

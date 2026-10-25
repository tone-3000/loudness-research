"""
Loudness *measures*: functions that map an output signal y (mono, 48 kHz,
float, dBFS-scaled) to a single number in dB (or dB-equivalent).

Every measure is written so that scaling y by g (linear) shifts the result by
exactly 20*log10(g).  That property is required for a value to be usable as a
normalisation offset by the engines (gain = target - value).

Weightings implemented
    flat        plain mean-square (what the trainer does today)
    hp(f)       Butterworth high-pass before measuring
    band(lo,hi) Butterworth band-pass ("instrument band" / generic-speaker
                proxy) before measuring
    A           IEC 61672 A-weighting
    K           ITU-R BS.1770 K-weighting (pre-filter + RLB)
    LUFS-I      BS.1770-4 integrated loudness with absolute/relative gating
    LUFS-S95    95th percentile of 3 s short-term loudness (EBU Tech 3341)
    ISO532-1    Zwicker time-varying loudness (mosqito), reported as phon
                (N5 percentile per ISO 532-1 §6.4, and the energetic mean)

Also provides real-IR convolution with the TONE3000 engines' unity in-band
gain convention (wasm `irGain.ts` / pedal `IrGain`) for the evaluation side.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import bilinear, butter, lfilter, sosfilt, sosfiltfilt, zpk2tf, fftconvolve

from common import SR, dbfs_to_pa

# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _butter_sos(kind: str, f1: float, f2: float | None, order: int):
    if kind == "hp":
        return butter(order, f1, btype="highpass", fs=SR, output="sos")
    if kind == "lp":
        return butter(order, f1, btype="lowpass", fs=SR, output="sos")
    if kind == "bp":
        return butter(order, [f1, f2], btype="bandpass", fs=SR, output="sos")
    raise ValueError(kind)


def highpass(x: np.ndarray, fc: float, order: int = 2) -> np.ndarray:
    return sosfilt(_butter_sos("hp", fc, None, order), x)


def lowpass(x: np.ndarray, fc: float, order: int = 2) -> np.ndarray:
    return sosfilt(_butter_sos("lp", fc, None, order), x)


def bandpass(x: np.ndarray, lo: float, hi: float, order: int = 2) -> np.ndarray:
    """
    `order` is the order of each edge (HP and LP are cascaded), so order=2
    gives 12 dB/oct skirts on both sides, order=4 gives 24 dB/oct.
    """
    return lowpass(highpass(x, lo, order), hi, order)


@lru_cache(maxsize=None)
def _a_weighting_ba():
    # IEC 61672-1 A-weighting, analog poles/zeros -> bilinear at SR.
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    A1000 = 1.9997
    numer = [(2 * np.pi * f4) ** 2 * (10 ** (A1000 / 20)), 0, 0, 0, 0]
    denom = np.polymul(
        [1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
        [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2],
    )
    denom = np.polymul(np.polymul(denom, [1, 2 * np.pi * f3]), [1, 2 * np.pi * f2])
    return bilinear(numer, denom, SR)


def a_weight(x: np.ndarray) -> np.ndarray:
    b, a = _a_weighting_ba()
    return lfilter(b, a, x)


# BS.1770-4 K-weighting coefficients at 48 kHz (Table 1 and Table 2).
_K_STAGE1_B = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
_K_STAGE1_A = np.array([1.0, -1.69065929318241, 0.73248077421585])
_K_STAGE2_B = np.array([1.0, -2.0, 1.0])
_K_STAGE2_A = np.array([1.0, -1.99004745483398, 0.99007225036621])


def k_weight(x: np.ndarray) -> np.ndarray:
    y = lfilter(_K_STAGE1_B, _K_STAGE1_A, x)
    return lfilter(_K_STAGE2_B, _K_STAGE2_A, y)


# --------------------------------------------------------------------------
# Scalar measures (all in dB, all shift-equivariant under gain)
# --------------------------------------------------------------------------
def ms_db(x: np.ndarray) -> float:
    return 10.0 * np.log10(max(float(np.mean(np.square(x))), 1e-24))


def rms_flat(y: np.ndarray) -> float:
    return ms_db(y)


def rms_hp(y: np.ndarray, fc: float, order: int = 2) -> float:
    return ms_db(highpass(y, fc, order))


def rms_band(y: np.ndarray, lo: float, hi: float, order: int = 2) -> float:
    return ms_db(bandpass(y, lo, hi, order))


def rms_a(y: np.ndarray) -> float:
    return ms_db(a_weight(y))


def rms_k(y: np.ndarray) -> float:
    """Ungated K-weighted level, i.e. BS.1770 'momentary' over the whole signal."""
    return -0.691 + ms_db(k_weight(y))


def _blocks_k(y: np.ndarray, block_s: float = 0.4, hop_s: float = 0.1) -> np.ndarray:
    z = np.square(k_weight(y))
    n = int(block_s * SR)
    h = int(hop_s * SR)
    if len(z) < n:
        return np.array([-0.691 + 10 * np.log10(max(z.mean(), 1e-24))])
    cs = np.concatenate([[0.0], np.cumsum(z)])
    starts = np.arange(0, len(z) - n + 1, h)
    msq = (cs[starts + n] - cs[starts]) / n
    return -0.691 + 10 * np.log10(np.maximum(msq, 1e-24))


def lufs_integrated(y: np.ndarray) -> float:
    """BS.1770-4 integrated loudness (absolute -70 LUFS gate, relative -10 LU gate)."""
    lk = _blocks_k(y)
    sel = lk > -70.0
    if not sel.any():
        return -np.inf
    abs_gated = 10 * np.log10(np.mean(10 ** (lk[sel] / 10)))
    rel = abs_gated - 10.0
    sel2 = lk > rel
    return float(10 * np.log10(np.mean(10 ** (lk[sel2] / 10))))


def lufs_short_term_percentile(y: np.ndarray, pct: float = 95.0) -> float:
    """Percentile of 3 s short-term loudness (EBU Tech 3341), 100 ms hop."""
    lk = _blocks_k(y, block_s=3.0, hop_s=0.1)
    lk = lk[lk > -70.0]
    if len(lk) == 0:
        return -np.inf
    return float(np.percentile(lk, pct))


def zwicker_phon(y: np.ndarray, field: str = "free") -> dict:
    """
    ISO 532-1 time-varying loudness (Zwicker) via mosqito.  The digital signal
    is mapped to pascals with the study's playback calibration (common.py).

    Returns dict with N5 (sone), N5 as phon, energetic-mean loudness level
    (phon), and arithmetic mean sone.  Phon is the dB-like scale (10 phon per
    doubling of sone above 40 phon), so differences in phon are comparable to
    differences in dB for the purpose of gain matching.
    """
    from mosqito.sq_metrics import loudness_zwtv

    p = dbfs_to_pa(np.asarray(y, dtype=np.float64))
    N, _, _, _ = loudness_zwtv(p, SR, field_type=field)
    N = np.asarray(N)
    N = N[N > 0]
    if len(N) == 0:
        return {"N5_sone": 0.0, "N5_phon": -np.inf, "LLeq_phon": -np.inf, "Nmean_sone": 0.0}
    n5 = float(np.percentile(N, 95))
    phon = lambda n: 40.0 + 10.0 * np.log2(n) if n >= 1 else 40.0 * (n + 0.0005) ** 0.35  # noqa: E731
    # Energetic mean of loudness level (ISO 532-1 note on LL(P) / Rennies et al.)
    ll = np.array([phon(n) for n in N])
    lleq = float(10 * np.log10(np.mean(10 ** (ll / 10))))
    return {"N5_sone": n5, "N5_phon": float(phon(n5)), "LLeq_phon": lleq, "Nmean_sone": float(N.mean())}


# --------------------------------------------------------------------------
# Real cabinet IRs (evaluation side)
# --------------------------------------------------------------------------
def ir_inband_gain_db(taps: np.ndarray, sr: int = SR, lo: float = 100.0, hi: float = 5000.0, points: int = 64,
                      max_boost: float = 24.0, max_cut: float = 48.0) -> float:
    """
    Port of neural-amp-modeler-wasm `irInBandGainDb` / diy-pedal `IrGain`:
    gain (dB) that makes the kernel's mean power response over log-spaced
    points in [lo, hi] equal 0 dB.
    """
    f = lo * (hi / lo) ** (np.arange(points) / (points - 1))
    n = np.arange(len(taps))
    acc = 0.0
    for fk in f:
        w = 2 * np.pi * fk / sr
        h = np.sum(taps * np.exp(-1j * w * n))
        acc += abs(h) ** 2
    mean_h2 = acc / points
    if not (mean_h2 > 0) or not np.isfinite(mean_h2):
        return 0.0
    g = -10 * np.log10(mean_h2)
    return float(np.clip(g, -max_cut, max_boost))


def apply_ir(y: np.ndarray, taps: np.ndarray, gain_db: float) -> np.ndarray:
    out = fftconvolve(y, taps, mode="full")[: len(y)]
    return out * 10 ** (gain_db / 20)

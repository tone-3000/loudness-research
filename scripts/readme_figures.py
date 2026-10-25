#!/usr/bin/env python3
"""
Figures used only by the top-level README: old vs new stimulus, operating
point curves of the models, old vs new stamped value, creator bias.
Reads results/ produced by analyze.py / side_experiments.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import welch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIGS, RESULTS, SR, current_stimulus, rms_db  # noqa: E402
from stimuli import instrument_mean_spectrum  # noqa: E402
from candidates import stimulus  # noqa: E402

REC = "ip1k|sweep7_-39_-15@0/K_meandb"


def fig_stimuli():
    old = current_stimulus()
    new = stimulus("ip1k", "sweep7_-39_-15", 0.0)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), gridspec_kw={"width_ratios": [1.4, 1]})
    for ax, x, name in zip(axes[:, 0], (old, new), ("current: 1 s pink noise, −15 dBFS RMS", "proposed: 7 × 0.25 s instrument-pink bursts, −39 … −15 dBFS RMS")):
        t = np.arange(len(x)) / SR
        ax.plot(t, x, lw=0.3, color="C0")
        ax.set_xlim(0, max(len(old), len(new)) / SR)
        ax.set_ylim(-0.8, 0.8)
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("amplitude")
        ax.grid(alpha=0.3)
    axes[1, 0].set_xlabel("time (s)")
    # per-burst levels annotation
    for i, lv in enumerate(np.linspace(-39, -15, 7)):
        axes[1, 0].text(i * 0.25 + 0.125, 0.65, f"{lv:.0f}", ha="center", fontsize=8)
    # spectra
    ax = axes[0, 1]
    f, p_old = welch(old / 10 ** (rms_db(old) / 20), SR, nperseg=4096)
    f, p_new = welch(new / 10 ** (rms_db(new) / 20), SR, nperseg=4096)
    fi, p_inst = instrument_mean_spectrum(4096)
    ref = lambda p: 10 * np.log10(p[(f > 150) & (f < 300)].mean())  # noqa: E731
    ax.semilogx(f[1:], 10 * np.log10(p_old[1:]) - ref(p_old), label="current (pink)", color="C3")
    ax.semilogx(f[1:], 10 * np.log10(p_new[1:]) - ref(p_new), label="proposed (pink × HP70 × LP1k)", color="C0")
    m = (fi > 150) & (fi < 300)
    ax.semilogx(fi[1:], p_inst[1:] - p_inst[m].mean(), label="mean guitar + bass DI", color="k", ls="--", lw=1)
    ax.set_xlim(20, 20000)
    ax.set_ylim(-60, 15)
    ax.set_xlabel("Hz")
    ax.set_ylabel("PSD (dB, re 150–300 Hz)")
    ax.set_title("spectrum", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    # level over time
    ax = axes[1, 1]
    win = int(0.05 * SR)
    for x, name, c in ((old, "current", "C3"), (new, "proposed", "C0")):
        lv = [rms_db(x[i : i + win]) for i in range(0, len(x) - win, win // 2)]
        ax.plot(np.arange(len(lv)) * win / 2 / SR, lv, label=name, color=c)
    ax.axhspan(-30, -20, color="grey", alpha=0.15, label="typical DI level (−29 … −20 dBFS)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("RMS (dBFS, 50 ms)")
    ax.set_title("level over time", fontsize=10)
    ax.set_ylim(-45, -10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "readme_stimuli.png", dpi=130)
    plt.close(fig)


def fig_operating_point():
    s = pd.read_csv(RESULTS / "side_experiments.csv", dtype={"model_id": str})
    levels = [-27, -21, -15, -9, -3]
    cols = [f"pink_out_at_{l}" for l in levels]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for _, r in s.sample(min(90, len(s)), random_state=1).iterrows():
        y = r[cols].values.astype(float)
        ax.plot(levels, y - y[2], color="C0", alpha=0.25, lw=0.8)
    med = np.median(s[cols].values - s[cols].values[:, [2]], axis=0)
    ax.plot(levels, med, color="k", lw=2, label="median model")
    ax.plot(levels, np.array(levels) + 15, color="C3", ls="--", label="linear device")
    ax.axvline(-15, color="C3", lw=1)
    ax.text(-14.7, -11, "current\nstimulus", color="C3", fontsize=8, va="top")
    ax.axvspan(-30, -20, color="grey", alpha=0.15, label="typical DI level")
    ax.set_xlabel("pink noise input level (dBFS RMS)")
    ax.set_ylabel("output level relative to output at −15 (dB)")
    ax.set_title(f"Where the stimulus sits on each model's input→output curve ({len(s)} models)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1]
    ax.hist(s.pink_slope_27_15, bins=30, range=(0, 1.1), color="C0", alpha=0.8)
    ax.set_xlabel("slope of output vs input between −27 and −15 dBFS (dB/dB)")
    ax.set_ylabel("models")
    ax.set_title("1.0 = linear, 0 = fully compressed; median %.2f" % s.pink_slope_27_15.median(), fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "readme_operating_point.png", dpi=130)
    plt.close(fig)


def fig_old_vs_new():
    m = pd.read_csv(RESULTS / "migration.csv", dtype={"model_id": str})
    meta = pd.read_csv(RESULTS / "models.csv", dtype={"model_id": str})[["model_id", "gear", "instrument"]]
    m = m.merge(meta, on="model_id")
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for (g, i), d in m.groupby(["gear", "instrument"]):
        ax.scatter(d.L_old, d.L_new_plus_c, s=18, alpha=0.75, label=f"{g} {i}", marker="o" if i == "guitar" else "^")
    lo, hi = -52, -3
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("current metadata.loudness (dB)")
    ax.set_ylabel("proposed metadata.loudness (dB, incl. +3 calibration)")
    ax.set_title("Same key, same median, different values\n(per-model difference SD 5.6 dB — this is the error being removed)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGS / "readme_old_vs_new.png", dpi=130)
    plt.close(fig)


def fig_creator_bias():
    from analyze import add_derived, gold_summary, load, residuals

    models, cands, gold = load()
    cands = add_derived(cands)
    gs = gold_summary(gold, models)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, cid, title in zip(axes, ("current|shipped@-15/flat", REC), ("current", "proposed")):
        df = residuals(cands, gs, models, "N5_phon", "realistic", cid).merge(models[["model_id", "creator"]], on="model_id")
        d0 = df[df.offset == 0].copy()
        cnt = d0.creator.value_counts()
        keep = cnt[cnt >= 8].index
        d0 = d0[d0.creator.isin(keep)]
        order = d0.groupby("creator").r.mean().sort_values().index
        ax.boxplot([d0[d0.creator == c].r.values for c in order], labels=[f"{c}\n(n={cnt[c]})" for c in order], showfliers=False)
        ax.axhline(0, color="k", lw=1)
        ax.set_title(f"{title}: residual by creator (≥ 8 models), native level\nrange of creator means {d0.groupby('creator').r.mean().agg(lambda s: s.max()-s.min()):.1f} dB", fontsize=10)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("perceived − stamped, dB (mean removed)")
    fig.tight_layout()
    fig.savefig(FIGS / "readme_creator_bias.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    fig_stimuli()
    fig_operating_point()
    fig_old_vs_new()
    fig_creator_bias()
    print("wrote readme_*.png")

#!/usr/bin/env python3
"""
Analysis: level-agnostic scoring of quick-noise candidates.

For candidate L and gold G measured at player offsets o in {-12..+12} dB:
    r_o(m) = G_o(m) - L(m)
    SD_o   = std over models of r_o
    sd_uniform = mean_o SD_o        (uniform prior on the player's level)
    sd_max     = max_o  SD_o
    sd_native  = SD_0
Lower bound for ANY scalar under the same prior: L*(m) = mean_o G_o(m).

Outputs results/*.csv, results/figures/*.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ENGINE_CLAMP_DB, ENGINE_TARGET_DB, FIGS, RESULTS  # noqa: E402
from candidates import STEADY_LEVELS  # noqa: E402

GEAR_ORDER = ["amp", "amp-cab", "pedal", "outboard", "experimental", "cab"]
GEAR_COLORS = dict(zip(GEAR_ORDER, ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]))
BASELINE = "current|shipped@-15/flat"
RECOMMENDED = "ip1k|sweep7_-39_-15@0/K_meandb"  # chosen in docs/03_method_and_results.md; None -> best sweep by consensus
LEVEL_SETS = {
    "avg(-39..-15)": [-39, -36, -33, -30, -27, -24, -21, -18, -15],
    "avg(-33..-21)": [-33, -30, -27, -24, -21],
    "avg(-36..-12)": [-36, -30, -24, -18, -12],
    "avg(-36,-24,-12)": [-36, -24, -12],
}


BURSTS: dict[tuple[str, str, str], list[float]] = {}  # (model_id, stim, measure) -> per-burst dB


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the aggregated tables written by aggregate.py (committed; no raw data or models needed)."""
    models = pd.read_csv(RESULTS / "models.csv", dtype={"model_id": str})
    cands = pd.read_csv(RESULTS / "candidates.csv.gz", dtype={"model_id": str})
    gold = pd.read_csv(RESULTS / "gold.csv.gz", dtype={"model_id": str}, keep_default_na=False)
    gold["offset"] = gold.offset.astype(float)
    gold["value"] = gold.value.astype(float)
    b = pd.read_csv(RESULTS / "bursts.csv.gz", dtype={"model_id": str}).sort_values("burst")
    for (mid, stim, meas), d in b.groupby(["model_id", "stim", "measure"], sort=False):
        BURSTS[(mid, stim, meas)] = d.value.tolist()
    return models, cands, gold


def add_derived(cands: pd.DataFrame) -> pd.DataFrame:
    """Multi-level averages of the steady 1 s stimuli (mean of dB over a level set)."""
    c = cands[cands.stim.str.contains(r"\|steady@") & ~cands.stim.str.contains(r"\|\d")].copy()
    c["shape"] = c.stim.str.split("|").str[0]
    c["level"] = c.stim.str.split("@").str[1].astype(float)
    rows = []
    for (mid, shape, meas), d in c.groupby(["model_id", "shape", "measure"]):
        lv = d.set_index("level").value
        for name, levels in LEVEL_SETS.items():
            sel = lv.reindex([float(x) for x in levels])
            if sel.isna().any():
                continue
            rows.append({"model_id": mid, "stim": f"{shape}|steady@{name}", "measure": meas, "value": float(sel.mean())})
    return pd.concat([cands, pd.DataFrame(rows)], ignore_index=True)


def gold_summary(gold: pd.DataFrame, models: pd.DataFrame) -> pd.DataFrame:
    """pooled (= mean of guitar-mean and bass-mean) per model, measure, condition, offset."""
    g = gold.merge(models[["model_id", "gear"]], on="model_id")
    rows = []
    for (mid, gear, meas, off), d in g.groupby(["model_id", "gear", "measure", "offset"]):
        for cond in ("raw", "realistic"):
            dd = d[d.ir != ""] if (cond == "realistic" and gear == "amp") else d[d.ir == ""]
            if dd.empty or not np.isfinite(dd.value).all():
                continue
            per = dd.groupby("clip_instrument").value.mean()
            if not {"guitar", "bass"} <= set(per.index):
                continue
            rows.append({"model_id": mid, "measure": meas, "condition": cond, "offset": off, "guitar": per["guitar"], "bass": per["bass"], "pooled": 0.5 * (per["guitar"] + per["bass"])})
    return pd.DataFrame(rows)


CLIP_RMS_DBFS = -27.0  # loudest-6 s windows of the demo DIs sit at -25..-29 dBFS RMS


def curve_scores(gs: pd.DataFrame, models: pd.DataFrame, gm: str, cond: str, stim: str, meas: str) -> dict | None:
    """
    Beyond-a-scalar simulation: the metadata is the whole per-burst curve of `stim`
    (input level -> output K level) and the engine knows the player's input
    level exactly.  Gain = target - interp(curve, player level).  Reports the
    SD per offset of G_o - L_o, i.e. what is left once level is handled.
    """
    from candidates import sweep_levels

    env = stim.split("|")[1].split("@")[0]
    xs = sweep_levels(env)
    G = gs[(gs.measure == gm) & (gs.condition == cond)].pivot(index="model_id", columns="offset", values="pooled").dropna()
    rows = {}
    for mid in G.index:
        b = BURSTS.get((mid, stim, meas + "_bursts"))
        if b is None:
            continue
        rows[mid] = {o: float(np.interp(CLIP_RMS_DBFS + o, xs, b)) for o in G.columns}
    if len(rows) < 10:
        return None
    L = pd.DataFrame(rows).T.loc[:, G.columns]
    idx = L.index
    sds = {o: float((G.loc[idx, o] - L[o]).std()) for o in G.columns}
    # residual pooled over offsets, after removing per-offset mean
    return {"stim": stim, "measure": f"{meas}_curve", "n": len(idx), "sd_uniform": float(np.mean(list(sds.values()))), "sd_max": float(max(sds.values())), "sd_native": sds.get(0.0, np.nan), **{f"sd@{o:+g}": v for o, v in sds.items()}}


def score(cands: pd.DataFrame, gs: pd.DataFrame, models: pd.DataFrame, gm: str, cond: str) -> tuple[pd.DataFrame, dict]:
    G = gs[(gs.measure == gm) & (gs.condition == cond)].pivot(index="model_id", columns="offset", values="pooled")
    G = G.dropna()
    wide = cands.pivot_table(index="model_id", columns=["stim", "measure"], values="value")
    idx = wide.index.intersection(G.index)
    G = G.loc[idx]
    wide = wide.loc[idx]
    meta = models.set_index("model_id").loc[idx]
    offsets = list(G.columns)
    # oracle lower bound
    Lstar = G.mean(axis=1)
    oracle = {f"sd@{o:+g}": float((G[o] - Lstar).std()) for o in offsets}
    oracle["sd_uniform"] = float(np.mean([(G[o] - Lstar).std() for o in offsets]))
    out = []
    for col in wide.columns:
        L = wide[col]
        ok = np.isfinite(L)
        if ok.sum() < min(20, len(L)):
            continue
        sds = {}
        pooled_r = []
        for o in offsets:
            r = (G[o] - L)[ok]
            sds[o] = r.std()
            pooled_r.append(r - r.mean())
        pr = pd.concat(pooled_r)
        gear = pd.concat([meta.gear[ok]] * len(offsets))
        inst = pd.concat([meta.instrument[ok]] * len(offsets))
        gm_ = pr.groupby(gear.values).mean()
        im = pr.groupby(inst.values).mean()
        r0 = (G[0.0] - L)[ok] if 0.0 in sds else None
        out.append(
            {
                "stim": col[0],
                "measure": col[1],
                "n": int(ok.sum()),
                "sd_uniform": float(np.mean(list(sds.values()))),
                "sd_max": float(np.max(list(sds.values()))),
                "sd_native": float(sds.get(0.0, np.nan)),
                **{f"sd@{o:+g}": float(v) for o, v in sds.items()},
                "gear_offset_range": float(gm_.max() - gm_.min()),
                **{f"off_{k}": float(v) for k, v in gm_.items()},
                "inst_offset": float(im.get("bass", np.nan) - im.get("guitar", np.nan)),
                "p90_range_native": float(np.percentile(r0, 95) - np.percentile(r0, 5)) if r0 is not None else np.nan,
            }
        )
    df = pd.DataFrame(out).sort_values("sd_uniform").reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    df["id"] = df.stim + "/" + df.measure
    return df, oracle


def parse(stim: str) -> dict:
    shape, rest = stim.split("|", 1)
    env, _, lvl = rest.partition("@")
    secs = None
    if "|" in lvl:
        lvl, secs = lvl.split("|")
    return {"shape": shape, "env": env, "level": lvl, "secs": secs}


# ------------------------------------------------------------------ figures
def fig_level_curves(rk: pd.DataFrame, gold_name: str, measure: str):
    d = rk[rk.measure == measure].copy()
    d = d.join(pd.DataFrame([parse(s) for s in d.stim], index=d.index))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, env in zip(axes, ("steady", "pluck")):
        dd = d[(d.env == env) & d.level.str.match(r"^-?\d+$") & d.secs.isna()]
        for shape, s in dd.groupby("shape"):
            s = s.assign(l=s.level.astype(float)).sort_values("l")
            ax.plot(s.l, s.sd_uniform, marker="o", label=shape)
        base = rk[rk.id == BASELINE]
        if len(base):
            ax.axhline(base.sd_uniform.iloc[0], color="k", ls="--", lw=1, label="current calc")
        sw = d[d.env.str.startswith("sweep")]
        for i, (_, s) in enumerate(sw[sw["shape"] == "ip1k"].iterrows()):
            ax.axhline(s.sd_uniform, color="grey", ls=":", lw=1)
            ax.text(-39, s.sd_uniform, f"ip1k {s.env} ({measure}_meandb)" if measure.endswith("meandb") else f"ip1k {s.env}", fontsize=7, va="bottom")
        ax.set_title(f"envelope = {env}")
        ax.set_xlabel("stimulus RMS level (dBFS)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"mean SD of residual over player levels −12…+12 dB (dB)\ngold = {gold_name}")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / f"level_curves_{measure}_{gold_name}.png", dpi=130)
    plt.close(fig)


def fig_sd_vs_offset(rk: pd.DataFrame, ids: list[str], gold_name: str, oracle: dict):
    offs = sorted(float(c[3:]) for c in rk.columns if c.startswith("sd@"))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for cid in ids:
        r = rk[rk.id == cid]
        if r.empty:
            continue
        r = r.iloc[0]
        ax.plot(offs, [r[f"sd@{o:+g}"] for o in offs], marker="o", label=f"{cid} (mean {r.sd_uniform:.2f})")
    ax.plot(offs, [oracle[f"sd@{o:+g}"] for o in offs], "k--", label=f"lower bound for any scalar (mean {oracle['sd_uniform']:.2f})")
    ax.set_xlabel("player level relative to the demo DIs (dB)   [−12 ≈ −40 dBFS RMS … +12 ≈ −16 dBFS RMS]")
    ax.set_ylabel("SD of residual across models (dB)")
    ax.set_title(f"How the error depends on how hard the player hits the model (gold={gold_name})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / f"sd_vs_offset_{gold_name}.png", dpi=130)
    plt.close(fig)


def residuals(cands, gs, models, gm, cond, cid):
    stim, meas = cid.rsplit("/", 1)
    L = cands[(cands.stim == stim) & (cands.measure == meas)].set_index("model_id").value
    G = gs[(gs.measure == gm) & (gs.condition == cond)].pivot(index="model_id", columns="offset", values="pooled").dropna()
    idx = L.index.intersection(G.index)
    rows = []
    for o in G.columns:
        r = G.loc[idx, o] - L.loc[idx]
        r = r - r.mean()
        for mid, v in r.items():
            rows.append({"model_id": mid, "offset": o, "r": v})
    df = pd.DataFrame(rows).merge(models[["model_id", "gear", "instrument"]], on="model_id")
    return df


def fig_amp_ir_effect(gs: pd.DataFrame, models: pd.DataFrame, gm: str = "N5_phon"):
    """Amp-only models: perceived loudness through a cab IR (unity in-band) minus raw, native level."""
    g = gs[(gs.measure == gm) & (gs.offset == 0.0)].merge(models[["model_id", "gear"]], on="model_id")
    g = g[g.gear == "amp"].pivot(index="model_id", columns="condition", values="pooled").dropna()
    d = g.realistic - g.raw
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(d, bins=30, color="#d62728", alpha=0.75)
    ax.set_xlabel(f"loudness change when an amp-only model goes through a cab IR (unity in-band gain), {'phon' if 'phon' in gm else 'dB'}")
    ax.set_ylabel("models")
    ax.set_title(f"Amp-only models ({len(d)}): IR effect (gold={gm}); mean {d.mean():+.1f}, SD {d.std():.1f}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / f"amp_ir_effect_{gm}.png", dpi=130)
    plt.close(fig)


def fig_instrument_gap(gs: pd.DataFrame, models: pd.DataFrame, gm: str = "N5_phon"):
    """Through the same model: guitar-mean minus bass-mean perceived loudness, native level, realistic."""
    g = gs[(gs.measure == gm) & (gs.offset == 0.0) & (gs.condition == "realistic")].merge(models[["model_id", "gear"]], on="model_id")
    g = g.assign(gap=g.guitar - g.bass)
    fig, ax = plt.subplots(figsize=(7, 4))
    for gear in GEAR_ORDER:
        d = g[g.gear == gear].gap
        if len(d) < 5:
            continue
        ax.hist(d, bins=np.arange(-6, 8.5, 0.5), alpha=0.5, color=GEAR_COLORS[gear], label=f"{gear} (SD {d.std():.1f})")
    ax.set_xlabel("guitar loudness − bass loudness through the same model (dB)")
    ax.set_ylabel("models")
    ax.set_title(f"Irreducible instrument gap per model (gold={gm}); overall SD={g.gap.std():.2f} dB")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / f"instrument_gap_{gm}.png", dpi=130)
    plt.close(fig)


def fig_box_by_gear(res: dict[str, pd.DataFrame], gold_name: str):
    fig, axes = plt.subplots(1, len(res), figsize=(6 * len(res), 4.8), sharey=True)
    for ax, (name, df) in zip(axes, res.items()):
        data, labels, cols = [], [], []
        for g in GEAR_ORDER:
            for inst in ("guitar", "bass"):
                sel = df[(df.gear == g) & (df.instrument == inst)].r
                if len(sel):
                    data.append(sel.values)
                    labels.append(f"{g}\n{inst}")
                    cols.append(GEAR_COLORS[g])
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.45)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{name}\nSD over models×levels = {df.r.std():.2f} dB")
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(f"residual after normalisation, dB (gold={gold_name}; all player levels pooled)")
    fig.tight_layout()
    fig.savefig(FIGS / f"residuals_by_gear_{gold_name}.png", dpi=130)
    plt.close(fig)


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    models, cands, gold = load()
    cands = add_derived(cands)
    print(f"models={len(models)}  candidates={cands.stim.nunique()} stimuli × measures -> {len(cands.groupby(['stim','measure']))} candidates")
    gs = gold_summary(gold, models)
    gs.to_csv(RESULTS / "gold_summary.csv", index=False)
    for gm in ("N5_phon", "LUFS"):
        fig_amp_ir_effect(gs, models, gm)
        fig_instrument_gap(gs, models, gm)

    golds = {
        "N5_realistic": ("N5_phon", "realistic"),
        "LLeq_realistic": ("LLeq_phon", "realistic"),
        "LUFS_realistic": ("LUFS", "realistic"),
        "K_realistic": ("K", "realistic"),
        "LUFS_raw": ("LUFS", "raw"),
        "N5_raw": ("N5_phon", "raw"),
    }
    rankings, oracles = {}, {}
    for name, (gm, cond) in golds.items():
        if gs[(gs.measure == gm) & (gs.condition == cond)].empty:
            continue
        rk, oracle = score(cands, gs, models, gm, cond)
        rk.to_csv(RESULTS / f"ranking_{name}.csv", index=False)
        rankings[name], oracles[name] = rk, oracle
        base = rk[rk.id == BASELINE].iloc[0]
        print(f"\n== {name}: oracle sd_uniform={oracle['sd_uniform']:.2f}; current: sd_uniform={base.sd_uniform:.2f} sd_native={base.sd_native:.2f} sd_max={base.sd_max:.2f} (rank {base['rank']}/{len(rk)})")
        curves = [c for c in (curve_scores(gs, models, gm, cond, f"{sh}|sweep7_-45_-9@0", m) for sh in ("ip1k", "ip2k", "inst") for m in ("K", "flat")) if c]
        if curves:
            cdf = pd.DataFrame(curves)
            cdf.to_csv(RESULTS / f"curve_{name}.csv", index=False)
            print("   beyond-a-scalar simulation (curve metadata + engine knows the input level):")
            print(cdf[["stim", "measure", "sd_uniform", "sd_native", "sd_max"]].to_string(index=False))
        cols = ["rank", "id", "sd_uniform", "sd_native", "sd_max", "gear_offset_range", "inst_offset"]
        print(rk.head(15)[cols].to_string(index=False))
        for meas in ("flat", "K"):
            fig_level_curves(rk, name, meas)

    prim = [p for p in ("N5_realistic", "LUFS_realistic") if p in rankings]
    cons = None
    for p in prim:
        r = rankings[p][["id", "sd_uniform", "sd_max"]].rename(columns={"sd_uniform": f"sdu_{p}", "sd_max": f"sdmax_{p}"})
        cons = r if cons is None else cons.merge(r, on="id")
    cons["sd_uniform_mean"] = cons[[c for c in cons.columns if c.startswith("sdu_")]].mean(axis=1)
    cons["sd_max_mean"] = cons[[c for c in cons.columns if c.startswith("sdmax_")]].mean(axis=1)
    cons = cons.sort_values("sd_uniform_mean").reset_index(drop=True)
    cons.to_csv(RESULTS / "ranking_consensus.csv", index=False)
    print("\n== consensus (mean sd_uniform over N5_realistic & LUFS_realistic):")
    print(cons.head(25).to_string(index=False))
    best = cons.iloc[0].id
    # best "simple" variants for the report: best steady single-level, best sweep, best multi-level avg
    simple = {
        "steady": cons[cons.id.str.contains(r"\|steady@-\d+/")].iloc[0].id,
        "pluck": cons[cons.id.str.contains(r"\|pluck@-\d+/")].iloc[0].id,
        "sweep": cons[cons.id.str.contains(r"\|sweep")].iloc[0].id,
        "avg": cons[cons.id.str.contains(r"steady@avg")].iloc[0].id,
    }
    print("best overall:", best, " best by family:", simple)
    if RECOMMENDED:
        simple["sweep"] = RECOMMENDED
        print("recommended (figures + migration):", RECOMMENDED)
    for p in prim:
        gm, cond = golds[p]
        fig_sd_vs_offset(rankings[p], [BASELINE, simple["steady"], simple["sweep"], simple["avg"]], p, oracles[p])
        fig_box_by_gear({"current: " + BASELINE: residuals(cands, gs, models, gm, cond, BASELINE), "best sweep: " + simple["sweep"]: residuals(cands, gs, models, gm, cond, simple["sweep"])}, p)

    # In-place migration numbers for the best sweep candidate
    stim, meas = simple["sweep"].rsplit("/", 1)
    Lb = cands[(cands.stim == "current|shipped@-15") & (cands.measure == "flat")].set_index("model_id").value
    Ln = cands[(cands.stim == stim) & (cands.measure == meas)].set_index("model_id").value
    idx = Lb.index.intersection(Ln.index)
    c = float(np.median(Lb.loc[idx]) - np.median(Ln.loc[idx]))
    gain_old = np.clip(ENGINE_TARGET_DB - Lb.loc[idx], -ENGINE_CLAMP_DB, ENGINE_CLAMP_DB)
    summary = {
        "candidate": simple["sweep"],
        "median_L_old": float(np.median(Lb.loc[idx])),
        "median_L_new_raw": float(np.median(Ln.loc[idx])),
        "calibration_offset_c_db": c,
        "sd_L_old": float(Lb.loc[idx].std()),
        "sd_L_new": float(Ln.loc[idx].std()),
        "sd_(L_new+c - L_old)": float((Ln.loc[idx] + c - Lb.loc[idx]).std()),
        "clamp_hits_old_pct": float(100 * np.mean(np.abs(ENGINE_TARGET_DB - Lb.loc[idx]) >= ENGINE_CLAMP_DB)),
        "clamp_hits_new_pct_at_c": float(100 * np.mean(np.abs(ENGINE_TARGET_DB - (Ln.loc[idx] + c)) > ENGINE_CLAMP_DB)),
        "clamp_hits_new_pct_c_minimising": None,
        "max_L_new_plus_c": float((Ln.loc[idx] + c).max()),
    }
    best_c, best_hits = c, 1e9
    for cc in np.arange(c - 6, c + 6.01, 0.5):
        h = float(100 * np.mean(np.abs(ENGINE_TARGET_DB - (Ln.loc[idx] + cc)) > ENGINE_CLAMP_DB))
        if h < best_hits:
            best_hits, best_c = h, float(cc)
    summary["clamp_hits_new_pct_c_minimising"] = {"c": best_c, "pct": best_hits}
    pd.DataFrame({"model_id": idx, "L_old": Lb.loc[idx].values, "L_new": Ln.loc[idx].values, "L_new_plus_c": (Ln.loc[idx] + c).values, "gain_old": gain_old.values, "gain_new": (ENGINE_TARGET_DB - Ln.loc[idx] - c).values}).to_csv(RESULTS / "migration.csv", index=False)
    (RESULTS / "migration_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n== migration:", json.dumps(summary, indent=1))

    # Length test table
    lt = cands[cands.stim.str.contains(r"\|[\d.]+s$")]
    if len(lt):
        lt = lt[lt.measure == "K"].copy()
        lt["secs"] = lt.stim.str.extract(r"\|([\d.]+)s$").astype(float)
        lt["level"] = lt.stim.str.extract(r"@(-?\d+)\|").astype(float)
        ref = cands[(cands.measure == "K") & cands.stim.str.match(r"^ip1k\|steady@-(30|24|18)$")].copy()
        ref["secs"] = 1.0
        ref["level"] = ref.stim.str.extract(r"@(-?\d+)$").astype(float)
        both = pd.concat([lt, ref])
        w = both.pivot_table(index=["model_id", "level"], columns="secs", values="value")
        d = (w.sub(w[4.0], axis=0)).drop(columns=4.0)
        tab = d.groupby("level").agg(["mean", "std"]).round(3)
        tab.to_csv(RESULTS / "length_test.csv")
        print("\n== length test (K level minus the 4 s value, dB):\n", tab.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

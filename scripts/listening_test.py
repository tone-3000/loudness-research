#!/usr/bin/env python3
"""
Build a self-contained listening test: real DI clips rendered through a
representative set of models, with a web page that lets you switch models
while the clip keeps playing and compare how well each normalisation scheme
(current / proposed / none) keeps the volume steady.

Audio is rendered once per model x clip x player level and stored
peak-normalised (MP3, 48 kHz); the per-approach gains are applied in the
browser with `audio.volume`, so the page works from file:// with no server.
Models are referred to by their anonymised corpus ids.

    python scripts/listening_test.py            # -> listening_test/index.html + audio/
    open listening_test/index.html
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ENGINE_TARGET_DB, MODELS_DIR, RESULTS, ROOT, SR, load_model, run_model  # noqa: E402
from measures import apply_ir, ir_inband_gain_db, lufs_integrated, rms_k  # noqa: E402
from stimuli import _loudest_window, di, ir  # noqa: E402

OUT = ROOT / "listening_test"
AUDIO = OUT / "audio"
CLIP_S = 6.0
CLIPS = [  # (label, DI name, instrument)
    ("clean guitar (Mayer)", "Mayer - Guitar", "guitar"),
    ("rhythm guitar (Metalcore)", "Metalcore - Guitar", "guitar"),
    ("lead guitar (Slide Lead)", "Slide Lead - Guitar", "guitar"),
    ("bass (Downtown)", "Downtown - Bass", "bass"),
    ("bass (Drivin')", "Drivin' - Bass", "bass"),
]
LEVELS = [0.0, -9.0, 6.0]  # player level relative to the demo DIs (dB)
IR_FOR = {"guitar": "celestion", "bass": "ampeg"}
CALIBRATION_DB = 3.0
# how many models per gear x instrument stratum; picked at evenly spaced
# quantiles of (new - old) so that the selection spans the spread, not only
# the extremes.
PICK = {
    ("amp", "guitar"): 3, ("amp", "bass"): 2,
    ("amp-cab", "guitar"): 3, ("amp-cab", "bass"): 2,
    ("pedal", "guitar"): 2, ("pedal", "bass"): 1,
    ("outboard", "guitar"): 2, ("outboard", "bass"): 1,
    ("experimental", "guitar"): 1, ("cab", "guitar"): 1,
}


def pick_models() -> pd.DataFrame:
    m = pd.read_csv(RESULTS / "migration.csv", dtype={"model_id": str})
    meta = pd.read_csv(RESULTS / "models.csv", dtype={"model_id": str})
    m = m.merge(meta[["model_id", "gear", "instrument", "creator", "training_type"]], on="model_id")
    m["d"] = m.L_new_plus_c - m.L_old
    rows = []
    for (gear, inst), n in PICK.items():
        g = m[(m.gear == gear) & (m.instrument == inst)].sort_values("d")
        if g.empty:
            continue
        qs = np.linspace(0.08, 0.92, n) if n > 1 else [0.5]
        idx = sorted({int(round(q * (len(g) - 1))) for q in qs})
        rows.append(g.iloc[idx])
    sel = pd.concat(rows).reset_index(drop=True)
    return sel


def main() -> int:
    AUDIO.mkdir(parents=True, exist_ok=True)
    sel = pick_models()
    print(f"{len(sel)} models")
    clips = {}
    for label, name, inst in CLIPS:
        x = _loudest_window(di(name), CLIP_S)
        clips[label] = (x, inst)
    irs = {k: ir(v) for k, v in IR_FOR.items()}
    ir_gain = {k: ir_inband_gain_db(v) for k, v in irs.items()}

    models_out, files = [], []
    for i, r in sel.iterrows():
        model, _ = load_model(MODELS_DIR / f"{r.model_id}.nam")
        L_old = float(r.L_old)
        L_new = float(r.L_new) + CALIBRATION_DB
        # raw engine arithmetic, no clamp: the test is about the metadata, not engine limits
        g_cur = ENGINE_TARGET_DB - L_old
        g_new = ENGINE_TARGET_DB - L_new
        entry = {
            "id": r.model_id, "gear": r.gear, "instrument": r.instrument, "training_type": r.training_type,
            "creator": r.creator, "L_old": round(L_old, 2), "L_new": round(L_new, 2),
            "gain": {"current": round(g_cur, 2), "proposed": round(g_new, 2), "none": 0.0},
            "renders": {},
        }
        for ci, (label, (x, inst)) in enumerate(clips.items()):
            for lvl in LEVELS:
                xin = x * 10 ** (lvl / 20)
                y = run_model(model, xin)
                through_ir = r.gear == "amp"
                if through_ir:
                    y = apply_ir(y, irs[inst], ir_gain[inst])
                peak = float(np.abs(y).max())
                trim = -1.0 - 20 * np.log10(max(peak, 1e-9))  # dB applied so the file peaks at -1 dBFS
                fname = f"{r.model_id}_c{ci}_{int(lvl):+d}.mp3"
                sf.write(AUDIO / fname, (y * 10 ** (trim / 20)).astype(np.float32), SR, format="MP3", subtype="MPEG_LAYER_III")
                k = rms_k(y)
                entry["renders"][f"{label}|{lvl:+g}"] = {
                    "file": f"audio/{fname}", "trim": round(trim, 2), "K": round(k, 2),
                    "LUFS": round(float(lufs_integrated(y)), 2), "ir": IR_FOR[inst] if through_ir else "",
                }
                files.append((g_cur - trim, g_new - trim, -trim))
        models_out.append(entry)
        print(f"[{i+1}/{len(sel)}] {r.model_id} {r.gear}/{r.instrument}  L_old {L_old:.1f} L_new {L_new:.1f}  gain cur {g_cur:+.1f} new {g_new:+.1f}")

    # largest (gain - trim) over all files and approaches -> volume reference so every volume <= 1
    gref = float(max(max(f) for f in files)) + 0.5
    data = {
        "clips": [{"label": c[0], "instrument": c[2]} for c in CLIPS],
        "levels": LEVELS,
        "target": ENGINE_TARGET_DB,
        "calibration": CALIBRATION_DB,
        "gref": round(gref, 2),
        "models": models_out,
    }
    (OUT / "data.json").write_text(json.dumps(data, indent=1))
    (OUT / "index.html").write_text(HTML.replace("__DATA__", json.dumps(data)))
    print(f"wrote {OUT/'index.html'}  ({len(files)} renders, gref {gref:.1f} dB)")
    return 0


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NAM loudness metadata — listening test</title>
<style>
  :root { color-scheme: light; --fg:#1b1b1b; --muted:#6b6b6b; --line:#e3e3e3; --cur:#d9534f; --new:#2b7bba; --none:#8a8a8a; --hi:#fff7d6; }
  body { font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; color:var(--fg); margin:0; padding:24px 28px 80px; max-width:1200px; }
  h1 { font-size:20px; margin:0 0 4px; } h2 { font-size:15px; margin:22px 0 8px; }
  p { max-width:900px; margin:6px 0; } .muted { color:var(--muted); }
  .controls { display:flex; flex-wrap:wrap; gap:18px 28px; align-items:flex-end; padding:14px 16px; border:1px solid var(--line); border-radius:8px; margin:14px 0; background:#fafafa; }
  .ctl label { display:block; font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); margin-bottom:4px; }
  .seg { display:inline-flex; border:1px solid #c9c9c9; border-radius:6px; overflow:hidden; }
  .seg button { border:0; background:#fff; padding:6px 12px; cursor:pointer; font:inherit; border-right:1px solid #c9c9c9; }
  .seg button:last-child { border-right:0; } .seg button.on { background:#1b1b1b; color:#fff; }
  .seg button.cur.on { background:var(--cur);} .seg button.new.on { background:var(--new);} .seg button.none.on { background:var(--none);}
  select, input[type=number] { font:inherit; padding:5px 8px; border:1px solid #c9c9c9; border-radius:6px; background:#fff; }
  button.big { font:inherit; padding:7px 14px; border-radius:6px; border:1px solid #c9c9c9; background:#fff; cursor:pointer; } button.big.on { background:#1b1b1b; color:#fff; }
  table { border-collapse:collapse; width:100%; margin-top:8px; }
  th { text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); font-weight:600; padding:6px 8px; border-bottom:1px solid var(--line); }
  td { padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:middle; }
  tr.playing td { background:var(--hi); }
  tr:hover td { background:#f5f5f5; } tr.playing:hover td { background:var(--hi); }
  .play { width:30px; height:30px; border-radius:50%; border:1px solid #c9c9c9; background:#fff; cursor:pointer; font-size:13px; }
  .play.on { background:#1b1b1b; color:#fff; border-color:#1b1b1b; }
  td.model { max-width:300px; } .name { font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:300px; }
  .sub { color:var(--muted); font-size:12px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:300px; }
  .gear { display:inline-block; font-size:11px; padding:1px 6px; border-radius:10px; background:#eee; margin-right:4px; }
  .gear.bass { background:#e6eefb; }
  .bar { position:relative; height:26px; background:#f3f3f3; border-radius:3px; min-width:220px; }
  .bar .tick { position:absolute; top:0; bottom:0; width:1px; background:#ccc; }
  .bar .target { position:absolute; top:-3px; bottom:-3px; width:2px; background:#1b1b1b; }
  .bar .m { position:absolute; top:4px; height:18px; width:4px; border-radius:2px; opacity:.35; }
  .bar .m.cur { background:var(--cur);} .bar .m.new { background:var(--new);} .bar .m.none { background:var(--none);}
  .bar .m.sel { opacity:1; width:8px; margin-left:-2px; top:1px; height:24px; box-shadow:0 0 0 1px #fff; }
  .num { font-variant-numeric:tabular-nums; text-align:right; white-space:nowrap; }
  .legend span { display:inline-block; margin-right:14px; } .legend i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:-1px; }
  .stat { display:inline-block; min-width:120px; margin-right:18px; } .stat b { font-size:18px; }
  kbd { font: 11px ui-monospace, Menlo, monospace; border:1px solid #ccc; border-bottom-width:2px; border-radius:4px; padding:0 5px; background:#fff; }
  footer { margin-top:26px; color:var(--muted); font-size:12px; }
</style>
</head>
<body>
<h1>NAM loudness metadata — listening test</h1>
<p>Real DI clips rendered through a representative set of public TONE3000 models, referred to by anonymised ids (every gear type, guitar and bass; amp-only models go through a cabinet IR at unity in-band gain, as the pedal and web engines do). Pick a normalisation scheme and switch models while the clip plays — exactly what a player does when stepping through presets. The clip position carries over between models, so what you hear is the volume jump between presets, nothing else.</p>
<p class="muted">Gains are the raw engine arithmetic, <code>−18 − loudness</code>, with no clamp, so what you hear reflects the metadata alone. All renders share one fixed trim so the three schemes are directly comparable; turn your monitor up — the files are deliberately quiet so that no gain ever clips.</p>

<div class="controls">
  <div class="ctl"><label>Normalisation</label>
    <div class="seg" id="approach">
      <button class="cur" data-v="current">Current metadata</button>
      <button class="new" data-v="proposed">Proposed metadata</button>
      <button class="none" data-v="none">None</button>
    </div></div>
  <div class="ctl"><label>Clip</label><select id="clip"></select></div>
  <div class="ctl"><label>Player level</label><select id="level"></select></div>
  <div class="ctl"><label>Auto-scan</label>
    <button class="big" id="scan">▶ step through models</button>
    every <input type="number" id="scanS" value="3" min="1" max="15" step="0.5" style="width:60px"> s</div>
  <div class="ctl"><label>Keys</label><span class="muted"><kbd>space</kbd> play/pause &nbsp; <kbd>↑</kbd><kbd>↓</kbd> model &nbsp; <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd> scheme &nbsp; <kbd>s</kbd> scan</span></div>
</div>

<div id="stats"></div>

<table id="tbl">
  <thead><tr>
    <th></th><th>model</th><th>metadata.loudness<br><span class="muted" style="text-transform:none">current → proposed</span></th><th>gain applied<br><span class="muted" style="text-transform:none">current / proposed</span></th>
    <th>what you hear — K-weighted level of the render after gain (dB)<div class="legend" style="margin-top:3px"><span><i style="background:var(--cur)"></i>current</span><span><i style="background:var(--new)"></i>proposed</span><span><i style="background:var(--none)"></i>none</span><span><i style="background:#1b1b1b;width:2px"></i>target (mean of the selected scheme)</span></div></th>
    <th class="num">selected</th>
  </tr></thead>
  <tbody></tbody>
</table>

<footer>
  Generated by <code>scripts/listening_test.py</code>. Metadata values: current = flat RMS of 1 s pink noise at −15 dBFS; proposed = K-weighted mean-of-dB over a 7-burst −39…−15 dBFS instrument-pink sweep, +3 dB calibration. “What you hear” is the measured K-weighted level of each render plus the gain the scheme applies; the spread of those markers is the thing the metadata is supposed to make small. See the README in the repository root.
</footer>

<script>
const DATA = __DATA__;
const S = { approach: 'proposed', clip: 0, level: 0, playing: -1, scan: null, t: 0 };
const BAR_LO = -50, BAR_HI = -5;
const audios = [];
const tbody = document.querySelector('#tbl tbody');

function key() { return DATA.clips[S.clip].label + '|' + (DATA.levels[S.level] >= 0 ? '+' : '') + DATA.levels[S.level]; }
function heard(m, approach) { return m.renders[key()].K + m.gain[approach]; }
function volumeFor(m, approach) {
  const r = m.renders[key()];
  const db = m.gain[approach] - r.trim - DATA.gref;
  return Math.min(1, Math.pow(10, db / 20));
}
function sd(a) { const mu = a.reduce((x, y) => x + y, 0) / a.length; return Math.sqrt(a.reduce((x, y) => x + (y - mu) ** 2, 0) / a.length); }
function pct(a, p) { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; }

function build() {
  const clipSel = document.getElementById('clip');
  DATA.clips.forEach((c, i) => clipSel.add(new Option(c.label, i)));
  const lvlSel = document.getElementById('level');
  DATA.levels.forEach((l, i) => lvlSel.add(new Option((l === 0 ? 'as the demo DI' : (l > 0 ? '+' : '') + l + ' dB (' + (l > 0 ? 'hot' : 'quiet') + ' player)'), i)));
  clipSel.onchange = () => { S.clip = +clipSel.value; reload(); };
  lvlSel.onchange = () => { S.level = +lvlSel.value; reload(); };
  document.querySelectorAll('#approach button').forEach(b => b.onclick = () => setApproach(b.dataset.v));
  document.getElementById('scan').onclick = toggleScan;
  DATA.models.forEach((m, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td><button class="play" data-i="${i}">▶</button></td>
      <td class="model"><div class="name">${m.gear} · ${m.instrument} <span class="muted" style="font-weight:400">(${m.id})</span></div><div class="sub">creator ${m.creator} · ${m.training_type}<span class="irnote"></span></div></td>
      <td class="num">${m.L_old.toFixed(1)} → ${m.L_new.toFixed(1)}</td>
      <td class="num">${fmt(m.gain.current)}<br>${fmt(m.gain.proposed)}</td>
      <td><div class="bar" data-i="${i}"></div></td>
      <td class="num heard"></td>`;
    tbody.appendChild(tr);
    const a = new Audio(); a.preload = 'auto'; a.loop = true; audios.push(a);
    a.ontimeupdate = () => { if (S.playing === i) S.t = a.currentTime; };
  });
  tbody.addEventListener('click', e => { const b = e.target.closest('.play'); if (b) toggle(+b.dataset.i); });
  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.code === 'Space') { e.preventDefault(); S.playing >= 0 ? toggle(S.playing) : play(0); }
    else if (e.key === 'ArrowDown') { e.preventDefault(); play((Math.max(S.playing, 0) + 1) % DATA.models.length); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); play((Math.max(S.playing, 0) - 1 + DATA.models.length) % DATA.models.length); }
    else if (e.key === '1') setApproach('current'); else if (e.key === '2') setApproach('proposed'); else if (e.key === '3') setApproach('none');
    else if (e.key === 's') toggleScan();
  });
  reload();
}
function esc(s) { return String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c])); }
function fmt(g) { return (g >= 0 ? '+' : '') + g.toFixed(1) + ' dB'; }

function reload() {
  DATA.models.forEach((m, i) => {
    const r = m.renders[key()];
    audios[i].src = r.file;
    audios[i].volume = volumeFor(m, S.approach);
    tbody.rows[i].querySelector('.irnote').textContent = r.ir ? ' · through ' + r.ir + ' IR' : '';
  });
  if (S.playing >= 0) { const i = S.playing; audios[i].currentTime = S.t; audios[i].play(); }
  redraw();
}
function setApproach(v) {
  S.approach = v;
  document.querySelectorAll('#approach button').forEach(b => b.classList.toggle('on', b.dataset.v === v));
  DATA.models.forEach((m, i) => audios[i].volume = volumeFor(m, v));
  redraw();
}
function redraw() {
  const vals = DATA.models.map(m => heard(m, S.approach));
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  DATA.models.forEach((m, i) => {
    const bar = tbody.rows[i].querySelector('.bar');
    let h = '';
    for (let d = -45; d <= -10; d += 5) h += `<div class="tick" style="left:${x(d)}%"></div>`;
    h += `<div class="target" style="left:${x(mean)}%"></div>`;
    for (const ap of ['none', 'current', 'proposed']) {
      const cls = ap === 'current' ? 'cur' : ap === 'proposed' ? 'new' : 'none';
      h += `<div class="m ${cls} ${ap === S.approach ? 'sel' : ''}" style="left:${x(heard(m, ap))}%" title="${ap}: ${heard(m, ap).toFixed(1)} dB"></div>`;
    }
    bar.innerHTML = h;
    const dev = heard(m, S.approach) - mean;
    tbody.rows[i].querySelector('.heard').textContent = (dev >= 0 ? '+' : '') + dev.toFixed(1) + ' dB';
    tbody.rows[i].classList.toggle('playing', i === S.playing);
    tbody.rows[i].querySelector('.play').classList.toggle('on', i === S.playing);
    tbody.rows[i].querySelector('.play').textContent = i === S.playing ? '❚❚' : '▶';
  });
  const stats = ['current', 'proposed', 'none'].map(ap => {
    const v = DATA.models.map(m => heard(m, ap));
    const mu = v.reduce((a, b) => a + b, 0) / v.length;
    const c = v.map(y => y - mu);
    return `<span class="stat"><span class="muted">${ap}</span><br><b>${sd(c).toFixed(1)} dB</b> SD · ${(pct(c, .95) - pct(c, .05)).toFixed(1)} dB 5–95 %</span>`;
  }).join('');
  document.getElementById('stats').innerHTML = `<span class="muted">Spread of the level you hear across these ${DATA.models.length} models, this clip and player level —</span> &nbsp; ${stats}`;
}
function x(db) { return 100 * (Math.min(BAR_HI, Math.max(BAR_LO, db)) - BAR_LO) / (BAR_HI - BAR_LO); }

function play(i) {
  if (S.playing >= 0 && S.playing !== i) { S.t = audios[S.playing].currentTime; audios[S.playing].pause(); }
  const a = audios[i];
  try { a.currentTime = S.t % (a.duration || 6); } catch (e) {}
  a.volume = volumeFor(DATA.models[i], S.approach);
  a.play();
  S.playing = i;
  redraw();
}
function toggle(i) {
  if (S.playing === i) { S.t = audios[i].currentTime; audios[i].pause(); S.playing = -1; redraw(); }
  else play(i);
}
function toggleScan() {
  const b = document.getElementById('scan');
  if (S.scan) { clearInterval(S.scan); S.scan = null; b.classList.remove('on'); b.textContent = '▶ step through models'; return; }
  if (S.playing < 0) play(0);
  const every = Math.max(0.5, +document.getElementById('scanS').value || 3) * 1000;
  S.scan = setInterval(() => play((S.playing + 1) % DATA.models.length), every);
  b.classList.add('on'); b.textContent = '■ stop scanning';
}
build();
setApproach('proposed');
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())

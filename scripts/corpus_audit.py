#!/usr/bin/env python3
"""
Compare the corpus to the full catalogue: gear, creator, training type, bass
share. -> results/corpus_audit.md

Needs the private catalogue and mapping (data/private/); creators are
reported under their anonymised labels.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PRIVATE, RESULTS  # noqa: E402

_BASS = re.compile(r"\bbass\b", re.I)


def isbass(r) -> bool:
    tags = [str(t).lower() for t in (r.get("tags") or [])]
    txt = " ".join([str(r.get("tone_title") or ""), " ".join(r.get("makes") or [])])
    return "bass" in tags or bool(_BASS.search(txt))


def md(df: pd.DataFrame) -> str:
    cols = [str(df.index.name or "")] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row.values]) + " |")
    return "\n".join(lines)


def share(s: pd.Series) -> pd.Series:
    return (s.value_counts(normalize=True) * 100).round(1)


def main() -> int:
    cat = pd.DataFrame([json.loads(l) for l in open(PRIVATE / "catalog.jsonl")])
    cat["id"] = cat.id.astype(str)
    cat["bass"] = cat.apply(isbass, axis=1)
    cmap = pd.read_csv(PRIVATE / "mapping_creators.csv").set_index("user_id").creator
    cat["short"] = cat.user_id.map(cmap).fillna("?")
    out = [f"# Corpus audit vs. catalogue ({len(cat)} A2 models, {cat.user_id.nunique()} creators, {cat.tone_id.nunique()} tones)\n"]
    tabs = {"catalogue": cat}
    ids = pd.read_csv(PRIVATE / "mapping_models.csv", dtype={"model_id": str}).model_id
    tabs["corpus"] = cat.set_index("id").loc[ids].reset_index()
    for title, col in (("Gear share (%)", "gear"), ("Training type share (%)", "training_type")):
        t = pd.concat({k: share(v[col]) for k, v in tabs.items()}, axis=1).fillna(0)
        out.append(f"## {title}\n\n" + t.pipe(md) + "\n")
    t = pd.concat({k: share(v["short"]).head(12) for k, v in tabs.items()}, axis=1).fillna(0)
    out.append("## Creator share (%), top creators\n\n" + t.pipe(md) + "\n")
    rows = {k: {"models": len(v), "creators": v.user_id.nunique(), "tones": v.tone_id.nunique(), "top creator %": round(share(v.user_id).iloc[0], 1), "top-3 creators %": round(share(v.user_id).head(3).sum(), 1), "bass %": round(100 * v.bass.mean(), 1), "has input_level %": round(100 * v.input_level.notna().mean(), 1)} for k, v in tabs.items()}
    out.append("## Summary\n\n" + pd.DataFrame(rows).pipe(md) + "\n")
    out.append("Bass share is deliberately high in the corpus (2.5 % of the catalogue) so that bass can be evaluated; it is a sampling stratum only. Creator labels are anonymised (c01 ...).\n")
    (RESULTS / "corpus_audit.md").write_text("\n".join(out))
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

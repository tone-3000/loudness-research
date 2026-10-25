#!/usr/bin/env python3
"""
One-time anonymisation of the corpus.

The study was run on public TONE3000 models.  Everything published in this
repository refers to them only by opaque ids (m001 ...) and to their creators
by opaque labels (c01 ...).  The mapping back to TONE3000 model ids / creator
ids is kept in data/private/ (git-ignored) so the maintainers can re-download
the models and re-run the experiment.

    python scripts/anonymize.py --corpus data/private/corpus_tone3000.csv --catalog data/private/catalog.jsonl

Produces
    data/corpus.csv                         public corpus (anon ids, gear, instrument, training type, creator label)
    data/private/mapping_models.csv         anon id -> model id, url, name ...
    data/private/mapping_creators.csv       creator label -> user id
and renames data/models/<model_id>.nam -> data/models/<anon>.nam and
results/raw/<model_id>.json -> results/raw/<anon>.json (model_id / user_id
fields rewritten) when those exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, MODELS_DIR, PRIVATE, RESULTS  # noqa: E402

SEED = 20261025


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(PRIVATE / "corpus_tone3000.csv"))
    ap.add_argument("--catalog", default=str(PRIVATE / "catalog.jsonl"))
    args = ap.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    corpus = pd.read_csv(args.corpus, dtype={"model_id": str, "tone_id": str})
    creators = sorted({json.loads(l)["user_id"] for l in open(args.catalog)} | set(corpus.user_id))
    order = rng.permutation(len(creators))
    cmap = {creators[i]: f"c{rank + 1:02d}" for rank, i in enumerate(order)}
    pd.DataFrame({"creator": list(cmap.values()), "user_id": list(cmap.keys())}).sort_values("creator").to_csv(PRIVATE / "mapping_creators.csv", index=False)

    ids = list(corpus.model_id)
    order = rng.permutation(len(ids))
    mmap = {ids[i]: f"m{rank + 1:03d}" for rank, i in enumerate(order)}
    priv = corpus.assign(anon_id=corpus.model_id.map(mmap), creator=corpus.user_id.map(cmap))
    priv.sort_values("anon_id").to_csv(PRIVATE / "mapping_models.csv", index=False)

    pub = pd.DataFrame(
        {
            "model_id": priv.anon_id,
            "gear": priv.gear,
            "instrument": priv.instrument,
            "training_type": priv.training_type,
            "input_level_dbu": priv.input_level,
            "creator": priv.creator,
        }
    ).sort_values("model_id")
    pub.to_csv(DATA / "corpus.csv", index=False)
    print(f"corpus.csv: {len(pub)} models, {pub.creator.nunique()} creators")

    n = 0
    for mid, anon in mmap.items():
        p = MODELS_DIR / f"{mid}.nam"
        if p.exists():
            p.rename(MODELS_DIR / f"{anon}.nam")
            n += 1
    print(f"renamed {n} model files")

    raw = RESULTS / "raw"
    n = 0
    for mid, anon in mmap.items():
        p = raw / f"{mid}.json"
        if p.exists():
            r = json.loads(p.read_text())
            r["model_id"] = anon
            r["creator"] = cmap.get(r.pop("user_id", None), None)
            (raw / f"{anon}.json").write_text(json.dumps(r))
            p.unlink()
            n += 1
    print(f"rewrote {n} raw result files")
    return 0


if __name__ == "__main__":
    sys.exit(main())

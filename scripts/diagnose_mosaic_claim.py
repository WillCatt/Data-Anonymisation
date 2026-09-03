"""
Is "1,268 / 1,268 documents are uniquely identifiable" a finding or an artefact?

The suspicion
-------------
Every k threshold in the reported distribution comes out at 100%. A measurement
with no spread in it is usually measuring its own construction, not the data.

Two things look wrong on inspection:

1. `k_anonymity_table` counts a document as re-identifiable when NO other
   document shares its *entire* quasi-identifier set. The median set has ~14
   facts. Two distinct court judgments matching on all 14 exactly is close to
   impossible, so k=1 is very nearly guaranteed before any data is read.

2. 57% of the fingerprint is DATETIME, and 79% of those normalise to a bare
   year ("1999", "2003"). Procedural years identify a *case*, not a person, and
   they are shared widely — "1999" appears in 55 documents.

The control
-----------
Replace every real fact with a meaningless token drawn from a fixed pool,
keeping each document's set SIZE identical. If uniqueness is driven by the data
this should collapse. If it is driven by exact whole-set matching, it will not
move at all.

The replacement measurement
---------------------------
`min_facts_to_identify` already implements the question worth asking: how many
facts must an attacker learn, choosing the rarest first, before the target is
alone? That has a real distribution, survives dropping dates and money, and is
what the mosaic effect actually claims.

    python scripts/diagnose_mosaic_claim.py
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from anonymisation.data import load_tab
from anonymisation.mosaic import min_facts_to_identify, quasi_identifier_signature

OUT = REPO / "results/mosaic_diagnosis.csv"

ATTRIBUTE_SETS = {
    "all_as_reported": ("DEM", "DATETIME", "LOC", "QUANTITY"),
    "person_dem_loc": ("DEM", "LOC"),
    "dem_only": ("DEM",),
}


def signatures(docs, types, normalise=True):
    sig = defaultdict(set)
    for d in docs:
        s = quasi_identifier_signature(d, entity_types=types, normalise=normalise)
        if s:
            sig[d["doc_id"]].update(s)
    return {k: v for k, v in sig.items() if len(v) >= 2}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=int, default=300, help="size of the random-fact pool for the control")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tab = load_tab()
    docs = [d for s in tab for d in tab[s]]
    print(f"TAB: {len(docs):,} annotator rows over {len({d['doc_id'] for d in docs}):,} unique documents\n")

    # ── 1. what is the fingerprint made of? ──────────────────────────────
    sigs = signatures(docs, ATTRIBUTE_SETS["all_as_reported"])
    comp = Counter(t for v in sigs.values() for t, _ in v)
    total = sum(comp.values())
    print("Fingerprint composition (as reported):")
    for t, c in comp.most_common():
        print(f"  {t:9s} {c:6,}  {c / total:5.1%}")
    dt = [v for s in sigs.values() for t, v in s if t == "DATETIME"]
    years = [x for x in dt if re.fullmatch(r"(1[5-9]\d{2}|20\d{2})", x)]
    print(f"  -> {len(years):,}/{len(dt):,} ({len(years) / len(dt):.0%}) of DATETIME values are a bare year")
    print(f"  -> most common: {', '.join(v for v, _ in Counter(dt).most_common(5))}\n")

    # ── 2. the control ───────────────────────────────────────────────────
    real = {k: tuple(sorted(v)) for k, v in sigs.items()}
    rc = Counter(real.values())
    real_k1 = sum(1 for s in real.values() if rc[s] == 1)

    random.seed(args.seed)
    pool = [f"fact{i}" for i in range(args.pool)]
    fake = {k: tuple(sorted(random.sample(pool, min(len(v), args.pool)))) for k, v in sigs.items()}
    fc = Counter(fake.values())
    fake_k1 = sum(1 for s in fake.values() if fc[s] == 1)

    print("Exact whole-signature matching — the reported method:")
    print(f"  real facts                  k=1 for {real_k1:,}/{len(real):,} ({real_k1 / len(real):.1%})")
    print(f"  random facts, same set sizes k=1 for {fake_k1:,}/{len(fake):,} ({fake_k1 / len(fake):.1%})  <-- CONTROL")
    verdict = ("ARTEFACT — meaningless facts score identically, so the metric is "
               "measuring set size, not identifiability")
    if fake_k1 / len(fake) < 0.9 * (real_k1 / len(real)):
        verdict = "the real facts are meaningfully more unique than the control"
    print(f"  verdict: {verdict}\n")

    # ── 3. the measurement that survives ─────────────────────────────────
    rows = []
    print("Facts an attacker must learn (rarest first) before the document is alone:")
    for name, types in ATTRIBUTE_SETS.items():
        s = signatures(docs, types)
        vals = list(s.values())
        n = len(vals)
        mins = [min_facts_to_identify(t, vals[:i] + vals[i + 1:]) for i, t in enumerate(vals)]
        never = sum(1 for m in mins if m is None)
        c = Counter(m for m in mins if m is not None)
        cum, cums = 0, {}
        for k in range(1, 6):
            cum += c.get(k, 0)
            cums[k] = cum / n
        med = sorted(len(v) for v in vals)[n // 2]
        print(f"\n  {name}  (n={n:,}, median {med} facts)")
        print("    " + "  ".join(f"<={k}: {cums[k]:5.1%}" for k in range(1, 6)))
        print(f"    never unique even on the full set: {never} ({never / n:.1%})")
        rows.append({"attribute_set": name, "n_docs": n, "median_facts": med,
                     **{f"unique_within_{k}_facts": cums[k] for k in range(1, 6)},
                     "never_unique": never, "never_unique_pct": never / n})

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\n  Unlike the 100% figure, this has a distribution — some documents are")
    print(f"  never unique — and it holds after dropping dates and money entirely.")
    print(f"\n  Standing caveat: uniqueness here is uniqueness *within this corpus*,")
    print(f"  not within a population. A 'Bulgarian nurse' may be alone among 1,268")
    print(f"  judgments and one of thousands in the world. This bounds how")
    print(f"  distinguishable the documents are; it is not a re-identification rate.")
    print(f"\n→ {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()

"""
Does the pseudonymiser link the right mentions to the same pseudonym?

Why this exists
---------------
Redaction only has to answer "is this a name?". Pseudonymisation has to answer
a second, harder question: "is this the *same* name as one I have already
seen?" — because that is what decides whether two mentions share a token.

The two errors are not symmetric. A missed link splits one person across
`[PERSON_A]` and `[PERSON_B]`: the document still leaks nothing, an LLM just
loses the thread. A false link is worse than it looks. Two people share one
token, so `restore()` writes *one* real name back over both — the wrong name
into a sentence about someone else, silently, in a document the firm believes
it has cleaned. That is an integrity failure, not a leak, and no amount of
detection F1 will catch it.

So it is worth measuring, and TAB lets us: every gold mention carries an
`entity_id` linking it to the other mentions of the same entity. Feeding the
*gold* mentions into the pseudonymiser isolates the linking decision from the
detection decision — no model is loaded here, and the whole script runs in
under a minute on CPU.

What it measures
----------------
1. `--profile`  — the ground truth first. Does `entity_id` encode real
   coreference, or just string equality? (Both, partly: it never splits an
   identical string, but it does link 1,061 clusters that span more than one
   surface form. The consequence is in the notebook: TAB cannot see a
   same-string false merge, so the two-different-Smiths failure mode is
   invisible to this evaluation and has to be argued by construction.)

2. `--fit`      — the calibration. Replay every link the legacy policy makes
   over train+validation, ask gold whether it was right, and report precision
   per (evidence, entity_type). That table is `LINK_CONFIDENCE` in
   `pipeline/pseudonymise.py`. Cells are shrunk toward the tier's own rate
   (k=20) so a rule seen five times cannot claim 1.0.

3. default      — the held-out evaluation on test: pairwise precision, recall
   and F1 of the induced clusters against gold, per policy and per entity
   type — including `calibrated_no_initialism` and `calibrated_all_types`,
   which turn off the acronym rule and the PERSON/ORG scope gate respectively,
   so each is costed on its own — with 95% cluster-bootstrap intervals over
   the 127 unique judgments
   (see `bootstrap_ci.py` for why the resampling unit is not the 555 rows),
   plus a sweep of the confidence threshold and the precision of merges
   where two or more entities matched equally well.

Run from the repo root:
    python scripts/evaluate_coref_links.py --profile
    python scripts/evaluate_coref_links.py --fit
    python scripts/evaluate_coref_links.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from anonymisation.data import load_tab  # noqa: E402
from anonymisation.pipeline.pseudonymise import (  # noqa: E402
    LINK_CONFIDENCE_PRIOR,
    Pseudonymiser,
)

RESULTS = REPO / "results"
SHRINKAGE_K = 20  # pseudo-observations pulling a small cell toward its tier rate


# --------------------------------------------------------------------------
# Pairwise clustering metric
# --------------------------------------------------------------------------
def pair_counts(gold: Dict[str, str], pred: Dict[str, str]) -> Tuple[int, int, int]:
    """
    Compare two partitions of the same mentions by the pairs they agree on.

    Pairwise (rather than B-cubed or MUC) because the pairs *are* the product
    decision: every same-cluster pair is one claim that two mentions may share
    a pseudonym, and a false pair is exactly the merge that corrupts a restore.
    """
    def pairs(assignment: Dict[str, str]) -> set:
        groups: Dict[str, List[str]] = defaultdict(list)
        for mention, cluster in assignment.items():
            groups[cluster].append(mention)
        out = set()
        for members in groups.values():
            out |= set(combinations(sorted(members), 2))
        return out

    gold_pairs, pred_pairs = pairs(gold), pairs(pred)
    return (
        len(gold_pairs & pred_pairs),
        len(pred_pairs - gold_pairs),
        len(gold_pairs - pred_pairs),
    )


def prf(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def mentions_in_order(doc: dict) -> List[dict]:
    """Gold mentions in reading order — the order the pipeline would meet them."""
    return sorted(doc["entity_mentions"], key=lambda m: m["start_offset"])


def run_linker(doc: dict, **kwargs) -> Tuple[Dict[str, str], Dict[str, str], Pseudonymiser]:
    """Feed one document's gold mentions through the pseudonymiser."""
    pseudo = Pseudonymiser(**kwargs)
    gold: Dict[str, str] = {}
    pred: Dict[str, str] = {}
    for mention in mentions_in_order(doc):
        token = pseudo.token_for(mention["entity_type"], mention["span_text"])
        pred[mention["entity_mention_id"]] = token
        gold[mention["entity_mention_id"]] = mention["entity_id"]
    return gold, pred, pseudo


# --------------------------------------------------------------------------
# 1 · Is the ground truth what we think it is?
# --------------------------------------------------------------------------
def profile_gold(dataset, split: str = "test") -> dict:
    same_string_total: Counter = Counter()
    same_string_linked: Counter = Counter()
    cross_form_clusters: Counter = Counter()
    cluster_sizes: Counter = Counter()
    examples: Dict[str, List[List[str]]] = defaultdict(list)

    for doc in dataset[split]:
        by_form: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        by_entity: Dict[str, List[dict]] = defaultdict(list)
        for mention in doc["entity_mentions"]:
            by_form[(mention["entity_type"], mention["span_text"].strip())].append(
                mention["entity_id"]
            )
            by_entity[mention["entity_id"]].append(mention)

        for (entity_type, _form), ids in by_form.items():
            if len(ids) > 1:
                same_string_total[entity_type] += 1
                if len(set(ids)) == 1:
                    same_string_linked[entity_type] += 1

        for mentions in by_entity.values():
            cluster_sizes[len(mentions)] += 1
            forms = {m["span_text"].strip() for m in mentions}
            if len(forms) > 1:
                entity_type = mentions[0]["entity_type"]
                cross_form_clusters[entity_type] += 1
                if len(examples[entity_type]) < 8:
                    examples[entity_type].append(sorted(forms, key=len, reverse=True)[:4])

    return {
        "split": split,
        "unique_documents": len({doc["doc_id"] for doc in dataset[split]}),
        "annotator_rows": len(dataset[split]),
        "same_string_repeats": {
            entity_type: {
                "n": same_string_total[entity_type],
                "given_one_entity_id": same_string_linked[entity_type],
            }
            for entity_type in sorted(same_string_total)
        },
        "cross_form_clusters": dict(cross_form_clusters.most_common()),
        "cluster_size_distribution": dict(sorted(cluster_sizes.items())),
        "cross_form_examples": {k: v for k, v in examples.items()},
    }


# --------------------------------------------------------------------------
# 2 · Fit the calibration
# --------------------------------------------------------------------------
def fit_calibration(dataset, splits: Iterable[str]) -> pd.DataFrame:
    """
    Score every link the legacy policy makes. Legacy, deliberately: it merges on
    any evidence at all, so every tier is observed. Scoring only the links the
    calibrated policy accepts would measure a rule by the cases it was already
    trusted on.
    """
    cells: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0])

    for split in splits:
        for doc in dataset[split]:
            gold, _pred, pseudo = run_linker(doc, link_policy="legacy")
            # Same surface form implies same gold entity (verified in --profile),
            # so a form is enough to identify the entity it was linked to.
            form_to_entity = {
                (m["entity_type"], m["span_text"].strip()): m["entity_id"]
                for m in mentions_in_order(doc)
            }
            for decision, mention in zip(pseudo.decisions, mentions_in_order(doc)):
                if not decision.merged or decision.evidence == "exact":
                    continue
                matched_entity = form_to_entity.get(
                    (decision.entity_type, decision.matched_form.strip())
                )
                cell = cells[(decision.evidence, decision.entity_type)]
                cell[0] += int(matched_entity == mention["entity_id"])
                cell[1] += 1
            # The exact tier is scored too, but it needs no lookup: an identical
            # string is the same entity by construction of the annotation.
            for decision in pseudo.decisions:
                if decision.evidence == "exact":
                    cell = cells[("exact", decision.entity_type)]
                    cell[0] += 1
                    cell[1] += 1

    tier_totals: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for (evidence, _entity_type), (correct, total) in cells.items():
        tier_totals[evidence][0] += correct
        tier_totals[evidence][1] += total

    rows = []
    for (evidence, entity_type), (correct, total) in sorted(cells.items()):
        tier_correct, tier_total = tier_totals[evidence]
        tier_rate = tier_correct / tier_total if tier_total else 0.0
        rows.append({
            "evidence": evidence,
            "entity_type": entity_type,
            "n": total,
            "correct": correct,
            "precision_raw": correct / total if total else 0.0,
            "tier_rate": tier_rate,
            "confidence": (correct + SHRINKAGE_K * tier_rate) / (total + SHRINKAGE_K),
        })
    return pd.DataFrame(rows).sort_values(["evidence", "n"], ascending=[True, False])


def tier_breakdown(dataset, split: str) -> pd.DataFrame:
    """Per-tier precision on a held-out split, same method as the fit."""
    frame = fit_calibration(dataset, [split])
    frame.insert(0, "split", split)
    return frame


def ambiguity_breakdown(dataset, splits: Iterable[str]) -> pd.DataFrame:
    """
    Precision of merges where two or more entities matched equally well,
    against those where exactly one did.

    A tie means the evidence does not distinguish the candidates, and the
    winner is whichever was seen first. Whether that matters is an empirical
    question, so it is measured on every split rather than assumed.
    """
    cells: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0])
    for split in splits:
        for doc in dataset[split]:
            form_to_entity = {
                (m["entity_type"], m["span_text"].strip()): m["entity_id"]
                for m in mentions_in_order(doc)
            }
            _gold, _pred, pseudo = run_linker(
                doc, link_policy="calibrated", reject_ambiguous=False
            )
            for decision, mention in zip(pseudo.decisions, mentions_in_order(doc)):
                if not decision.merged:
                    continue
                matched_entity = form_to_entity.get(
                    (decision.entity_type, decision.matched_form.strip())
                )
                cell = cells[(split, "ambiguous" if decision.ambiguous else "unique")]
                cell[0] += int(matched_entity == mention["entity_id"])
                cell[1] += 1
    return pd.DataFrame([
        {"split": split, "kind": kind, "n": total, "correct": correct,
         "precision": correct / total if total else 0.0}
        for (split, kind), (correct, total) in sorted(cells.items())
    ])


# --------------------------------------------------------------------------
# 3 · Held-out cluster evaluation
# --------------------------------------------------------------------------
def evaluate_policy(dataset, split: str, **kwargs) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Pairwise counts per entity type, plus per-document counts for the bootstrap.

    Documents are keyed by `doc_id`, so the several annotator rows of one
    judgment are summed into a single resampling unit.
    """
    overall: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])
    per_doc: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])

    for doc in dataset[split]:
        gold, pred, _pseudo = run_linker(doc, **kwargs)
        tp, fp, fn = pair_counts(gold, pred)
        counts = per_doc[doc["doc_id"]]
        counts[0] += tp
        counts[1] += fp
        counts[2] += fn

        agg = overall["_ALL"]
        agg[0] += tp
        agg[1] += fp
        agg[2] += fn

        types = {m["entity_type"] for m in doc["entity_mentions"]}
        for entity_type in types:
            ids = {
                m["entity_mention_id"]
                for m in doc["entity_mentions"]
                if m["entity_type"] == entity_type
            }
            tp_t, fp_t, fn_t = pair_counts(
                {k: v for k, v in gold.items() if k in ids},
                {k: v for k, v in pred.items() if k in ids},
            )
            agg_t = overall[entity_type]
            agg_t[0] += tp_t
            agg_t[1] += fp_t
            agg_t[2] += fn_t

    rows = []
    for entity_type, (tp, fp, fn) in overall.items():
        precision, recall, f1 = prf(tp, fp, fn)
        rows.append({
            "entity_type": entity_type, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        })
    frame = pd.DataFrame(rows).sort_values("tp", ascending=False)
    counts = np.array([per_doc[k] for k in sorted(per_doc)], dtype=float)
    return frame, counts


def bootstrap_f1(counts: np.ndarray, iters: int, rng: np.random.Generator) -> Tuple[float, float]:
    """95% cluster-bootstrap interval for pairwise F1, resampling whole judgments."""
    n = len(counts)
    idx = rng.integers(0, n, size=(iters, n))
    resampled = counts[idx].sum(axis=1)
    tp, fp, fn = resampled[:, 0], resampled[:, 1], resampled[:, 2]
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)
    return float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))


def threshold_sweep(dataset, split: str, thresholds: Iterable[float]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        frame, _counts = evaluate_policy(
            dataset, split, link_policy="calibrated", min_link_confidence=threshold
        )
        overall = frame[frame.entity_type == "_ALL"].iloc[0]
        rows.append({
            "min_link_confidence": threshold,
            "tp": overall.tp, "fp": overall.fp, "fn": overall.fn,
            "precision": overall.precision, "recall": overall.recall, "f1": overall.f1,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--profile", action="store_true",
                        help="interrogate the gold annotation and exit")
    parser.add_argument("--fit", action="store_true",
                        help="fit LINK_CONFIDENCE on train+validation and exit")
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    dataset = load_tab()

    if args.profile:
        profile = profile_gold(dataset, args.split)
        (RESULTS / "coref_gold_profile.json").write_text(json.dumps(profile, indent=2))
        linked = profile["same_string_repeats"]
        print(f"{args.split}: {profile['annotator_rows']} annotator rows over "
              f"{profile['unique_documents']} unique judgments")
        print("\nIdentical string repeated in a document — gold gives it one entity_id:")
        for entity_type, counts in linked.items():
            print(f"  {entity_type:9s} {counts['given_one_entity_id']:5d}/{counts['n']:5d}")
        print("\nGold clusters spanning more than one surface form:")
        for entity_type, n in profile["cross_form_clusters"].items():
            print(f"  {entity_type:9s} {n:5d}")
        print(f"\nwrote {RESULTS / 'coref_gold_profile.json'}")
        return

    if args.fit:
        frame = fit_calibration(dataset, ["train", "validation"])
        frame.to_csv(RESULTS / "coref_link_calibration.csv", index=False)
        print(frame.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("\n# paste into LINK_CONFIDENCE in pipeline/pseudonymise.py")
        for evidence, group in frame.groupby("evidence"):
            cells = ", ".join(
                f'("{evidence}", "{row.entity_type}"): {row.confidence:.3f}'
                for row in group.itertuples()
            )
            print(f"    {cells},")
        print("\n# LINK_CONFIDENCE_PRIOR")
        for evidence, group in frame.groupby("evidence"):
            rate = group.tier_rate.iloc[0]
            print(f'    "{evidence}": {rate:.3f},')
        print(f"\nwrote {RESULTS / 'coref_link_calibration.csv'}")
        return

    rng = np.random.default_rng(args.seed)

    tiers = tier_breakdown(dataset, args.split)
    tiers.to_csv(RESULTS / "coref_link_tiers.csv", index=False)
    print(f"── link evidence on {args.split} (held out from the fit) ──")
    print(tiers[["evidence", "entity_type", "n", "correct", "precision_raw"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    ambiguity = ambiguity_breakdown(dataset, ["train", "validation", args.split])
    ambiguity.to_csv(RESULTS / "coref_ambiguity.csv", index=False)
    print("\n── merges where two or more entities matched equally well ──")
    print(ambiguity.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    frames = []
    for label, kwargs in (
        ("legacy", {"link_policy": "legacy"}),
        ("calibrated_no_guard", {"link_policy": "calibrated", "reject_ambiguous": False}),
        ("calibrated_no_initialism", {
            "link_policy": "calibrated",
            "disabled_evidence": ("initialism", "initialism_loose", "initialism_reverse"),
        }),
        ("calibrated_all_types", {"link_policy": "calibrated",
                                  "coref_types": ("PERSON", "ORG", "LOC", "DATETIME",
                                                  "DEM", "MISC", "CODE", "QUANTITY")}),
        ("calibrated", {"link_policy": "calibrated"}),
    ):
        frame, counts = evaluate_policy(dataset, args.split, **kwargs)
        low, high = bootstrap_f1(counts, args.iters, rng)
        frame.insert(0, "policy", label)
        frame["f1_lo"] = np.where(frame.entity_type == "_ALL", low, np.nan)
        frame["f1_hi"] = np.where(frame.entity_type == "_ALL", high, np.nan)
        frames.append(frame)
        overall = frame[frame.entity_type == "_ALL"].iloc[0]
        print(f"\n── {label} ── pairwise P {overall.precision:.4f} "
              f"R {overall.recall:.4f} F1 {overall.f1:.4f}  [{low:.4f}, {high:.4f}]")
        print(frame[frame.entity_type != "_ALL"]
              [["entity_type", "tp", "fp", "fn", "precision", "recall", "f1"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    clusters = pd.concat(frames, ignore_index=True)
    clusters.to_csv(RESULTS / "coref_clusters.csv", index=False)

    sweep = threshold_sweep(
        dataset, args.split,
        [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.9, 1.0],
    )
    sweep.to_csv(RESULTS / "coref_threshold_sweep.csv", index=False)
    print("\n── confidence threshold sweep ──")
    print(sweep.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\nwrote {RESULTS / 'coref_link_tiers.csv'}, {RESULTS / 'coref_clusters.csv'}, "
          f"{RESULTS / 'coref_ambiguity.csv'}, {RESULTS / 'coref_threshold_sweep.csv'}")


if __name__ == "__main__":
    main()

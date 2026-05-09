"""
Command-line entry point.

Usage
-----
    python -m anonymisation.cli redact \\
        --variant {lite|pro} \\
        [--ner spacy|hf|finetuned] \\
        [--k-target 5] [--max-iterations 5] \\
        [--json] [--mosaic-haystack tab] \\
        FILE

The defaults pick spaCy as the NER provider so the CLI runs out of the box
on a Phase-1 install. Pass `--ner finetuned` once you have a Phase-2 trained
model in `phase2_baseline_comparison/checkpoints/roberta-tab/final/`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

from .pipeline import LitePipeline, MosaicScorer, ProPipeline


# -----------------------------------------------------------------------
# NER provider construction (lazy imports — these deps may be Phase-2-only)
# -----------------------------------------------------------------------
def _spacy_predictor(model_name: str = "en_core_web_trf"):
    import spacy
    from .mapping import SPACY_TO_TAB

    nlp = spacy.load(model_name)

    def predict(text: str) -> List[Tuple[int, int, str, str]]:
        doc = nlp(text)
        return [
            (ent.start_char, ent.end_char, SPACY_TO_TAB[ent.label_], ent.text)
            for ent in doc.ents
            if ent.label_ in SPACY_TO_TAB
        ]
    return predict


def _hf_predictor(model_name: str = "dslim/bert-base-NER"):
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
    from .predictors import make_hf_predictor

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    pipe = pipeline("ner", model=model, tokenizer=tok, aggregation_strategy="simple", device=-1)
    return make_hf_predictor(pipe)


def _finetuned_predictor(model_dir: str):
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from .predictors import make_finetuned_predictor
    from .device import best_device

    tok = AutoTokenizer.from_pretrained(model_dir, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    device, _ = best_device()
    return make_finetuned_predictor(model, tok, device=device)


def build_ner_provider(choice: str, model_path: str | None):
    if choice == "spacy":
        return _spacy_predictor(model_path or "en_core_web_trf")
    if choice == "hf":
        return _hf_predictor(model_path or "dslim/bert-base-NER")
    if choice == "finetuned":
        if not model_path:
            raise SystemExit(
                "--ner finetuned requires --ner-model PATH "
                "(e.g. phase2_baseline_comparison/checkpoints/roberta-tab/final)"
            )
        return _finetuned_predictor(model_path)
    raise SystemExit(f"unknown --ner choice: {choice}")


# -----------------------------------------------------------------------
# Mosaic haystack
# -----------------------------------------------------------------------
def build_scorer(choice: str) -> MosaicScorer:
    if choice == "tab":
        from .data import load_tab
        ds = load_tab()
        return MosaicScorer.from_tab(list(ds["test"]))
    if choice == "empty":
        return MosaicScorer.empty()
    raise SystemExit(f"unknown --mosaic-haystack: {choice}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def cmd_redact(args: argparse.Namespace) -> int:
    # Read input
    if args.file == "-":
        text = sys.stdin.read()
    else:
        text = Path(args.file).read_text()

    # Build NER + (optional) scorer
    ner = build_ner_provider(args.ner, args.ner_model)

    if args.variant == "lite":
        pipeline = LitePipeline(ner_provider=ner, run_regex=not args.no_regex)
    else:
        scorer = build_scorer(args.mosaic_haystack)
        pipeline = ProPipeline(
            ner_provider=ner,
            scorer=scorer,
            k_target=args.k_target,
            max_iterations=args.max_iterations,
            run_regex=not args.no_regex,
        )

    result = pipeline(text)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(result.redacted_text)
        if args.variant == "pro":
            print(
                f"\n# mosaic risk: k_initial={result.mosaic_risk_initial} "
                f"→ k_final={result.mosaic_risk_final} "
                f"(target k≥{args.k_target}, iterations={result.iterations_used}, "
                f"converged={result.converged})",
                file=sys.stderr,
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="anonymisation", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_redact = sub.add_parser("redact", help="Redact a document")
    p_redact.add_argument("file", help="Input file path, or '-' for stdin")
    p_redact.add_argument(
        "--variant", choices=["lite", "pro"], default="lite",
        help="Lite = DIRECT-only; Pro = DIRECT + mosaic-aware QUASI generalization.",
    )
    p_redact.add_argument("--ner", choices=["spacy", "hf", "finetuned"], default="spacy")
    p_redact.add_argument("--ner-model", default=None,
                          help="Model name or path. Defaults to en_core_web_trf for spacy.")
    p_redact.add_argument("--no-regex", action="store_true",
                          help="Disable the regex post-pass.")
    p_redact.add_argument("--k-target", type=int, default=5,
                          help="Pro only: target k-anonymity (default 5).")
    p_redact.add_argument("--max-iterations", type=int, default=5,
                          help="Pro only: max generalization iterations (default 5).")
    p_redact.add_argument("--mosaic-haystack", choices=["tab", "empty"], default="tab",
                          help="Pro only: source of the mosaic comparison corpus.")
    p_redact.add_argument("--json", action="store_true",
                          help="Output the full audit log as JSON instead of just the text.")
    p_redact.set_defaults(func=cmd_redact)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

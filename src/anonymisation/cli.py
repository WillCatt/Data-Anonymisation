"""
Command-line entry point.

Usage
-----
    # Redact (Lite or Pro)
    python -m anonymisation.cli redact \\
        --variant {lite|pro} \\
        [--ner spacy|hf|finetuned] \\
        [--k-target 5] [--max-iterations 5] \\
        [--pseudonymise] [--vault-out PATH] \\
        [--json] [--mosaic-haystack tab] \\
        FILE

    # Round-trip an LLM answer back through the pseudonym vault
    python -m anonymisation.cli restore \\
        --vault PATH \\
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

from .pipeline import LitePipeline, MosaicScorer, ProPipeline, restore


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
        pipeline = LitePipeline(
            ner_provider=ner,
            run_regex=not args.no_regex,
            coref_extend=not args.no_coref,
            pseudonymise=args.pseudonymise,
        )
    else:
        scorer = build_scorer(args.mosaic_haystack)
        pipeline = ProPipeline(
            ner_provider=ner,
            scorer=scorer,
            k_target=args.k_target,
            max_iterations=args.max_iterations,
            run_regex=not args.no_regex,
            coref_extend=not args.no_coref,
            pseudonymise=args.pseudonymise,
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

    # Persist the pseudonym vault if asked
    if args.pseudonymise and args.vault_out:
        Path(args.vault_out).write_text(
            json.dumps(result.pseudonym_vault, indent=2, ensure_ascii=False)
        )
        print(
            f"# pseudonym vault written to {args.vault_out} "
            f"({len(result.pseudonym_vault)} entries)",
            file=sys.stderr,
        )
    elif args.pseudonymise and not args.json:
        # No file specified — emit vault on stderr so stdout stays clean
        print("\n# pseudonym vault:", file=sys.stderr)
        for token, original in result.pseudonym_vault.items():
            print(f"#   {token} -> {original!r}", file=sys.stderr)
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    """Take a pseudonymised text + vault and restore the original surface forms."""
    text = sys.stdin.read() if args.file == "-" else Path(args.file).read_text()
    vault = json.loads(Path(args.vault).read_text())
    if not isinstance(vault, dict):
        raise SystemExit(f"vault file {args.vault} did not parse as a JSON object")
    print(restore(text, vault), end="")
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
    p_redact.add_argument("--no-coref", action="store_true",
                          help="Disable the coreference extension pass (Phase 5).")
    p_redact.add_argument("--k-target", type=int, default=5,
                          help="Pro only: target k-anonymity (default 5).")
    p_redact.add_argument("--max-iterations", type=int, default=5,
                          help="Pro only: max generalization iterations (default 5).")
    p_redact.add_argument("--mosaic-haystack", choices=["tab", "empty"], default="tab",
                          help="Pro only: source of the mosaic comparison corpus.")
    p_redact.add_argument("--pseudonymise", "--pseudonymize", action="store_true",
                          help=("Use referential tokens ([PERSON_A], [PERSON_B], …) "
                                "instead of plain [TYPE] tags. Pair with --vault-out "
                                "to save the mapping for round-trip restore."))
    p_redact.add_argument("--vault-out", default=None,
                          help="Path to write the pseudonym vault as JSON. Implies --pseudonymise.")
    p_redact.add_argument("--json", action="store_true",
                          help="Output the full audit log as JSON instead of just the text.")
    p_redact.set_defaults(func=cmd_redact)

    p_restore = sub.add_parser(
        "restore",
        help="Round-trip a pseudonymised text back to original surface forms.",
    )
    p_restore.add_argument("file", help="Path to redacted text, or '-' for stdin")
    p_restore.add_argument("--vault", required=True,
                           help="Path to the pseudonym vault JSON produced by `redact --pseudonymise`.")
    p_restore.set_defaults(func=cmd_restore)

    args = parser.parse_args(argv)
    # If --vault-out is set, --pseudonymise is implied
    if hasattr(args, "vault_out") and args.vault_out and not args.pseudonymise:
        args.pseudonymise = True
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

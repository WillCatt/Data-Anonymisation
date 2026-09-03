"""
Gradio demo — Legal Text Anonymisation.

This is the *strongest available* version of the pipeline, in a single UI:

- NER backend is auto-detected at startup with a three-tier fallback:
    1. Phase 2 RoBERTa fine-tuned on TAB
       (models/roberta-tab/final/)
    2. spaCy en_core_web_trf (transformer)
    3. spaCy en_core_web_sm (small) — last resort

- Phase 5 CorefExtender is enabled by default (catches shorthand mentions).
- Phase 4 Pseudonymisation can be toggled on for referential tokens
  ([PERSON_A], [PERSON_B] …) with the vault displayed in its own tab.
- Phase 3 Pro mode (mosaic-aware QUASI generalisation) is selectable from
  the variant radio.

Run locally via:
    ./demo/run_gradio.sh

That script provisions a separate venv with Gradio + spaCy in matching
versions; the main repo venv stays untouched.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import gradio as gr

# Allow `import anonymisation` whether running from repo root or from this folder
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from anonymisation.mapping import SPACY_TO_TAB                                 # noqa: E402
from anonymisation.pipeline import LitePipeline, ProPipeline, MosaicScorer     # noqa: E402


# ---------------------------------------------------------------------------
# Three-tier NER backend with auto-detection
# ---------------------------------------------------------------------------
_FINETUNED_DIR = ROOT / "models" / "roberta-tab" / "final"

_state = {
    "predictor": None,
    "backend_label": None,
    "scorer": None,
}


def _make_finetuned_predictor() -> Tuple[Callable, str]:
    """Try loading the Phase 2 RoBERTa fine-tune. Raises on failure."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from anonymisation.predictors import make_finetuned_predictor
    from anonymisation.device import best_device

    tok = AutoTokenizer.from_pretrained(str(_FINETUNED_DIR), add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(str(_FINETUNED_DIR))
    device, label = best_device()
    predict = make_finetuned_predictor(model, tok, device=device)
    return predict, f"Phase 2 RoBERTa fine-tuned on TAB · {label}"


def _make_spacy_predictor(model_name: str) -> Tuple[Callable, str]:
    """Wrap a spaCy model as a TAB-flavoured predictor."""
    import spacy
    nlp = spacy.load(model_name)

    def predict(text: str):
        doc = nlp(text)
        return [
            (e.start_char, e.end_char, SPACY_TO_TAB[e.label_], e.text)
            for e in doc.ents if e.label_ in SPACY_TO_TAB
        ]
    return predict, f"spaCy {model_name}"


def _get_predictor() -> Tuple[Callable, str]:
    """Three-tier fallback. First call pays the load cost; subsequent calls cached."""
    if _state["predictor"] is not None:
        return _state["predictor"], _state["backend_label"]

    # Environment override — bypass auto-detection if the user has a preference.
    forced = os.environ.get("PIPELINE_BACKEND", "").strip().lower()

    # 1. Try the fine-tuned model (best quality).
    if forced in ("", "auto", "finetuned"):
        if _FINETUNED_DIR.exists():
            try:
                predict, label = _make_finetuned_predictor()
                _state["predictor"] = predict
                _state["backend_label"] = label
                print(f"[demo] Loaded NER backend: {label}")
                return predict, label
            except Exception as exc:
                print(f"[demo] Could not load fine-tuned RoBERTa ({exc}); falling back.")
        elif forced == "finetuned":
            raise SystemExit(
                f"PIPELINE_BACKEND=finetuned but no model at {_FINETUNED_DIR}. "
                f"Run notebooks/05_finetune_roberta.ipynb first."
            )

    # 2. Try spaCy transformer (medium quality, common install).
    if forced in ("", "auto", "trf", "transformer"):
        try:
            predict, label = _make_spacy_predictor("en_core_web_trf")
            _state["predictor"] = predict
            _state["backend_label"] = label
            print(f"[demo] Loaded NER backend: {label}")
            return predict, label
        except Exception as exc:
            print(f"[demo] spaCy trf not available ({exc}); falling back to sm.")

    # 3. spaCy small (always works after demo/run_gradio.sh bootstraps).
    predict, label = _make_spacy_predictor("en_core_web_sm")
    _state["predictor"] = predict
    _state["backend_label"] = label
    print(f"[demo] Loaded NER backend: {label}")
    return predict, label


def _get_scorer() -> MosaicScorer:
    if _state["scorer"] is None:
        try:
            from anonymisation.data import load_tab
            ds = load_tab()
            _state["scorer"] = MosaicScorer.from_tab(list(ds["test"]))
        except Exception as exc:
            print(f"[demo] Could not load TAB for mosaic scoring ({exc}); using empty haystack.")
            _state["scorer"] = MosaicScorer.empty()
    return _state["scorer"]


# ---------------------------------------------------------------------------
# Inference callback
# ---------------------------------------------------------------------------
def redact(text: str, variant: str, k_target: int, max_iters: int,
           pseudonymise: bool, coref_extend: bool):
    if not text or not text.strip():
        return (
            "",
            _backend_badge(),
            "_paste some text on the left and click Redact_",
            "",
            "",
        )

    predictor, _ = _get_predictor()

    if variant == "Lite (DIRECT only)":
        pipeline = LitePipeline(
            ner_provider=predictor,
            coref_extend=coref_extend,
            pseudonymise=pseudonymise,
        )
    else:
        scorer = _get_scorer()
        pipeline = ProPipeline(
            ner_provider=predictor,
            scorer=scorer,
            k_target=k_target,
            max_iterations=max_iters,
            coref_extend=coref_extend,
            pseudonymise=pseudonymise,
        )

    result = pipeline(text)

    # Status line — mosaic risk (Pro only) + pseudonym vault size
    parts: List[str] = []
    if variant.startswith("Pro"):
        ck = "✓ converged" if result.converged else "✗ fallback to suppression"
        parts.append(
            f"**Mosaic risk:** k_initial = `{result.mosaic_risk_initial}` → "
            f"k_final = `{result.mosaic_risk_final}` · "
            f"iterations: `{result.iterations_used}` · {ck}"
        )
    if pseudonymise:
        parts.append(f"**Vault:** {len(result.pseudonym_vault)} entries (see Vault tab)")
    status = "  ·  ".join(parts) if parts else "_(no status — Lite plain redaction)_"

    # Vault rendering (markdown table)
    if result.pseudonym_vault:
        vault_md = "| Token | Original surface form |\n|---|---|\n" + "\n".join(
            f"| `{tok}` | {orig} |" for tok, orig in result.pseudonym_vault.items()
        )
    else:
        vault_md = "_(no vault — enable Pseudonymise to see referential tokens)_"

    audit_payload = result.to_dict()

    return (
        result.redacted_text,
        _backend_badge(),
        status,
        vault_md,
        json.dumps(audit_payload, indent=2, ensure_ascii=False),
    )


def _backend_badge() -> str:
    _, label = _get_predictor() if _state["backend_label"] is None else (_state["predictor"], _state["backend_label"])
    return f"**NER backend:** `{label}` · **Coref:** Phase 5 post-processor"


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
def _load_examples() -> List[List]:
    """Pre-canned examples loaded from demo/examples/*.txt."""
    examples_dir = Path(__file__).parent / "examples"
    if not examples_dir.exists():
        return []
    rows = []
    for path in sorted(examples_dir.glob("*.txt")):
        rows.append([
            path.read_text().strip(),
            "Pro (mosaic-aware)",   # variant
            5,                        # k_target
            3,                        # max iters
            False,                    # pseudonymise
            True,                     # coref_extend
        ])
    return rows


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
DESCRIPTION = """\
# Legal Text Anonymisation

A redaction pipeline for legal text. Paste a document on the left and pick
a redaction strategy. The full per-decision audit log is on the **Audit log**
tab.

- **Lite** — strips DIRECT identifiers (names, organisations, case numbers, IBANs).
  Fast and predictable. Right choice when the threat is "an LLM provider might log
  our prompts".
- **Pro** — DIRECT redaction *plus* mosaic-aware QUASI generalisation. Iteratively
  broadens dates, locations, demographics and quantities until the document's
  residual fingerprint reaches the target *k*-anonymity. Right choice when the
  threat is "a determined adversary could re-identify the document".
- **Pseudonymise** — assigns referential tokens (`[PERSON_A]`, `[PERSON_B]`, ...)
  instead of plain `[PERSON]`. The mapping is held in a vault and can be reversed
  locally via `restore()` — see the Vault tab.

> Portfolio demo, not a deployed product. The mosaic scorer compares against the
> TAB corpus (1,268 ECHR cases) as a methodological stand-in for a firm's own
> document corpus.
"""


def _safe_textbox(**kwargs):
    """Construct a Textbox, dropping kwargs the installed Gradio rejects."""
    try:
        return gr.Textbox(**kwargs)
    except TypeError as exc:
        bad = str(exc).split("'")[1] if "'" in str(exc) else None
        if bad and bad in kwargs:
            kwargs.pop(bad)
            return _safe_textbox(**kwargs)
        return gr.Textbox(label=kwargs.get("label", ""), lines=kwargs.get("lines", 10))


def build_ui():
    with gr.Blocks(title="Legal Text Anonymisation") as demo:
        gr.Markdown(DESCRIPTION)

        # Load the predictor once so the badge can render at page load.
        _get_predictor()

        backend_md = gr.Markdown(_backend_badge())

        with gr.Row():
            # ── Input column ──
            with gr.Column(scale=1):
                input_text = gr.Textbox(
                    label="Input",
                    placeholder="Paste legal-style text here…",
                    lines=14,
                )
                variant = gr.Radio(
                    choices=["Lite (DIRECT only)", "Pro (mosaic-aware)"],
                    value="Pro (mosaic-aware)",
                    label="Pipeline variant",
                )
                with gr.Row():
                    k_target = gr.Slider(2, 10, value=5, step=1,
                                         label="Pro: k_target")
                    max_iters = gr.Slider(1, 5, value=3, step=1,
                                          label="Pro: max iterations")
                with gr.Row():
                    pseudonymise = gr.Checkbox(
                        value=False,
                        label="Pseudonymise (referential tokens + vault)",
                    )
                    coref_extend = gr.Checkbox(
                        value=True,
                        label="Coref extension (catch shorthand)",
                    )
                run_btn = gr.Button("Redact", variant="primary")

            # ── Output column ──
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("Redacted"):
                        redacted_out = _safe_textbox(
                            label="Redacted text", lines=12, show_copy_button=True,
                        )
                        status_md = gr.Markdown()
                    with gr.Tab("Vault"):
                        vault_md = gr.Markdown(
                            value="_(Pseudonymisation off — enable to see referential tokens here)_"
                        )
                    with gr.Tab("Audit log"):
                        audit_out = gr.Code(
                            label="Full per-decision audit (JSON)",
                            language="json",
                        )

        examples = _load_examples()
        if examples:
            gr.Examples(
                examples=examples,
                inputs=[input_text, variant, k_target, max_iters, pseudonymise, coref_extend],
                label="Pre-canned examples",
                cache_examples=False,  # caching needs fn+outputs and slows cold start
            )

        run_btn.click(
            fn=redact,
            inputs=[input_text, variant, k_target, max_iters, pseudonymise, coref_extend],
            outputs=[redacted_out, backend_md, status_md, vault_md, audit_out],
        )

    return demo


def _launch_kwargs() -> dict:
    return {
        "server_name": os.environ.get("HOST", "127.0.0.1"),
        "server_port": int(os.environ.get("PORT", "7860")),
        "share": False,
    }


if __name__ == "__main__":
    app = build_ui()
    app.launch(**_launch_kwargs())

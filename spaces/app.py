"""
HuggingFace Spaces entry point for the Legal Text Anonymisation demo.

This is a Spaces-tailored copy of the main repo's demo/app.py:
- NER backbone forced to `en_core_web_sm` for fast cold-start (the
  fine-tuned RoBERTa checkpoint isn't bundled — too large for the free tier).
- Working directory layout is flat (no `src/` prefix), so the imports
  resolve via the local `anonymisation/` package.

To run locally for testing:
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    python app.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, List, Tuple

import gradio as gr

from anonymisation.mapping import SPACY_TO_TAB
from anonymisation.pipeline import LitePipeline, ProPipeline, MosaicScorer


_state = {"predictor": None, "backend_label": None, "scorer": None}


def _make_spacy_predictor(model_name: str) -> Tuple[Callable, str]:
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
    if _state["predictor"] is not None:
        return _state["predictor"], _state["backend_label"]
    model_name = os.environ.get("SPACY_MODEL", "en_core_web_sm")
    predict, label = _make_spacy_predictor(model_name)
    _state["predictor"] = predict
    _state["backend_label"] = label
    print(f"[spaces] Loaded NER backend: {label}")
    return predict, label


def _get_scorer() -> MosaicScorer:
    if _state["scorer"] is None:
        try:
            from anonymisation.data import load_tab
            ds = load_tab()
            _state["scorer"] = MosaicScorer.from_tab(list(ds["test"]))
        except Exception as exc:
            print(f"[spaces] Could not load TAB ({exc}); empty haystack.")
            _state["scorer"] = MosaicScorer.empty()
    return _state["scorer"]


def _backend_badge() -> str:
    label = _state["backend_label"] or _get_predictor()[1]
    return f"**NER backend:** `{label}` · **Coref:** Phase 5 post-processor enabled"


def redact(text: str, variant: str, k_target: int, max_iters: int,
           pseudonymise: bool, coref_extend: bool):
    if not text or not text.strip():
        return (
            "", _backend_badge(),
            "_paste some text on the left and click Redact_",
            "_(no vault)_", "",
        )

    predictor, _ = _get_predictor()

    if variant == "Lite (DIRECT only)":
        pipeline = LitePipeline(
            ner_provider=predictor,
            coref_extend=coref_extend,
            pseudonymise=pseudonymise,
        )
    else:
        pipeline = ProPipeline(
            ner_provider=predictor, scorer=_get_scorer(),
            k_target=k_target, max_iterations=max_iters,
            coref_extend=coref_extend, pseudonymise=pseudonymise,
        )

    result = pipeline(text)

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

    if result.pseudonym_vault:
        vault_md = "| Token | Original surface form |\n|---|---|\n" + "\n".join(
            f"| `{tok}` | {orig} |" for tok, orig in result.pseudonym_vault.items()
        )
    else:
        vault_md = "_(Pseudonymisation off — enable to see referential tokens here)_"

    return (
        result.redacted_text,
        _backend_badge(),
        status,
        vault_md,
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
    )


def _load_examples() -> List[List]:
    examples_dir = Path(__file__).parent / "examples"
    if not examples_dir.exists():
        return []
    return [
        [
            path.read_text().strip(),
            "Pro (mosaic-aware)", 5, 3, False, True,
        ]
        for path in sorted(examples_dir.glob("*.txt"))
    ]


DESCRIPTION = """\
# Legal Text Anonymisation

A redaction pipeline for legal text. Paste a document on the left and pick
a redaction strategy. The full per-decision audit log is on the **Audit log** tab.

- **Lite** — strips DIRECT identifiers (names, organisations, case numbers, IBANs).
- **Pro** — DIRECT redaction + mosaic-aware QUASI generalisation. Iteratively
  broadens dates, locations, demographics until the document's residual fingerprint
  reaches the target k-anonymity.
- **Pseudonymise** — assigns referential tokens (`[PERSON_A]`, `[PERSON_B]` …)
  instead of plain `[PERSON]`. The vault tab shows the mapping; restore() can
  reverse it locally so an LLM's answer comes back with real names.

> Portfolio demo. Source: [github.com/wcatt98/data-anonymisation](https://github.com/wcatt98/data-anonymisation).
> The mosaic scorer compares against the TAB corpus (1,268 ECHR cases) as a
> methodological stand-in for a firm's own document corpus. NER backbone here is
> spaCy `en_core_web_sm` for fast Space boot — the full project uses a RoBERTa
> fine-tune on TAB train (Phase 2).
"""


def _safe_textbox(**kwargs):
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
        _get_predictor()
        backend_md = gr.Markdown(_backend_badge())

        with gr.Row():
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
                    k_target = gr.Slider(2, 10, value=5, step=1, label="Pro: k_target")
                    max_iters = gr.Slider(1, 5, value=3, step=1, label="Pro: max iterations")
                with gr.Row():
                    pseudonymise = gr.Checkbox(value=False, label="Pseudonymise")
                    coref_extend = gr.Checkbox(value=True, label="Coref extension")
                run_btn = gr.Button("Redact", variant="primary")

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
                            label="Full per-decision audit (JSON)", language="json"
                        )

        examples = _load_examples()
        if examples:
            gr.Examples(
                examples=examples,
                inputs=[input_text, variant, k_target, max_iters, pseudonymise, coref_extend],
                label="Pre-canned examples",
            )

        run_btn.click(
            fn=redact,
            inputs=[input_text, variant, k_target, max_iters, pseudonymise, coref_extend],
            outputs=[redacted_out, backend_md, status_md, vault_md, audit_out],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    # HF Spaces sets PORT=7860 by default; honour it if present.
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
    )

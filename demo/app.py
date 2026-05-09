"""
Gradio side-by-side demo for the Lite and Pro redaction pipelines.

Local: `python demo/app.py` — opens at http://localhost:7860
Spaces: copy this folder into a new HF Space (Gradio runtime).

Layout:
    ┌──────────────────┐  ┌──────────────────────────────┐
    │  INPUT (textarea)│  │  Tabs:                       │
    │                  │  │   • Lite output              │
    │                  │  │   • Pro  output + k badge    │
    │                  │  │   • Audit log (JSON)         │
    └──────────────────┘  └──────────────────────────────┘

The mosaic scorer is built lazily on first use and cached.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import gradio as gr

# Allow `import anonymisation` whether running from repo root or from this folder
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from anonymisation.mapping import SPACY_TO_TAB                                 # noqa: E402
from anonymisation.pipeline import LitePipeline, ProPipeline, MosaicScorer     # noqa: E402


# ---------------------------------------------------------------------------
# Lazy initialisation — first request pays the cost, then cached
# ---------------------------------------------------------------------------
_state = {"nlp": None, "scorer": None, "lite": None, "pro": None}


def _get_predictor():
    if _state["nlp"] is None:
        import spacy
        _state["nlp"] = spacy.load(os.environ.get("SPACY_MODEL", "en_core_web_sm"))
    nlp = _state["nlp"]

    def predict(text: str):
        doc = nlp(text)
        return [
            (e.start_char, e.end_char, SPACY_TO_TAB[e.label_], e.text)
            for e in doc.ents if e.label_ in SPACY_TO_TAB
        ]
    return predict


def _get_scorer():
    if _state["scorer"] is None:
        try:
            from anonymisation.data import load_tab
            ds = load_tab()
            _state["scorer"] = MosaicScorer.from_tab(list(ds["test"]))
        except Exception as exc:
            # Fall back to an empty scorer so the demo still runs offline
            print(f"⚠️  Could not load TAB for mosaic scoring ({exc}); using empty haystack.")
            _state["scorer"] = MosaicScorer.empty()
    return _state["scorer"]


def _get_pipelines(k_target: int, max_iters: int):
    predictor = _get_predictor()
    scorer = _get_scorer()
    lite = LitePipeline(ner_provider=predictor)
    pro = ProPipeline(
        ner_provider=predictor, scorer=scorer,
        k_target=k_target, max_iterations=max_iters,
    )
    return lite, pro


# ---------------------------------------------------------------------------
# Inference callback
# ---------------------------------------------------------------------------
def redact(text: str, k_target: int, max_iterations: int):
    if not text or not text.strip():
        return "", "", "", "_paste some text on the left_"

    lite, pro = _get_pipelines(k_target=k_target, max_iters=max_iterations)
    lite_result = lite(text)
    pro_result = pro(text)

    pro_summary = (
        f"**Mosaic risk:** k_initial = `{pro_result.mosaic_risk_initial}` "
        f"→ k_final = `{pro_result.mosaic_risk_final}`  ·  "
        f"iterations: `{pro_result.iterations_used}`  ·  "
        f"converged: `{pro_result.converged}`"
    )

    audit_payload = {
        "lite": lite_result.to_dict(),
        "pro":  pro_result.to_dict(),
    }
    return (
        lite_result.redacted_text,
        pro_result.redacted_text,
        pro_summary,
        json.dumps(audit_payload, indent=2, ensure_ascii=False),
    )


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
def _load_examples() -> list[list]:
    """Pre-canned examples loaded from demo/examples/*.txt."""
    examples_dir = Path(__file__).parent / "examples"
    if not examples_dir.exists():
        return [[
            "The applicant, Maria Petrova, is a 47-year-old Bulgarian national living in Plovdiv. "
            "On 12 March 2018 she filed a complaint (Application no. 12345/67) against "
            "the Sofia District Court alleging discrimination on grounds of her Roma ethnicity.",
            5, 5,
        ]]
    rows = []
    for path in sorted(examples_dir.glob("*.txt")):
        rows.append([path.read_text().strip(), 5, 5])
    return rows


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
DESCRIPTION = """\
# Legal Text Anonymisation — Lite vs Pro

Two redaction pipelines for legal text, side by side.

- **Lite** — strips DIRECT identifiers (names, organisations, case numbers). Fast and cheap. Right choice when the threat is "an LLM provider might log our prompts".
- **Pro** — DIRECT redaction *plus* mosaic-aware QUASI generalization. Iteratively broadens dates, locations, demographics, and quantities until the document's residual fingerprint is shared by at least *k* corpus documents (default *k* = 5). Right choice when the threat is "a determined adversary could re-identify the document via combined quasi-identifiers".

Paste a paragraph of legal-style text on the left. The full per-decision audit log is on the **Audit log** tab.

> ⚠️ This is a portfolio demo, not a deployed product. The mosaic scorer compares against the TAB corpus (1,268 ECHR cases) as a methodological stand-in for a firm's own document corpus.
"""


def _safe_textbox(**kwargs):
    """Construct a Textbox, dropping kwargs that the installed Gradio rejects."""
    try:
        return gr.Textbox(**kwargs)
    except TypeError as exc:
        # Strip any unsupported kwarg name out of the signature and retry.
        bad = str(exc).split("'")[1] if "'" in str(exc) else None
        if bad and bad in kwargs:
            kwargs.pop(bad)
            return _safe_textbox(**kwargs)
        # Last-ditch: minimal Textbox
        return gr.Textbox(label=kwargs.get("label", ""), lines=kwargs.get("lines", 10))


def build_ui() -> "gr.Blocks":
    with gr.Blocks(
        title="Legal Text Anonymisation — Lite vs Pro",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                input_text = gr.Textbox(
                    label="Input",
                    placeholder="Paste legal-style text here…",
                    lines=14,
                )
                with gr.Row():
                    k_target = gr.Slider(2, 10, value=5, step=1, label="Pro: k_target")
                    max_iters = gr.Slider(1, 5, value=5, step=1, label="Pro: max iterations")
                run_btn = gr.Button("Redact", variant="primary")

            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("Lite"):
                        lite_out = _safe_textbox(
                            label="DIRECT-only redaction", lines=10, show_copy_button=True,
                        )
                    with gr.Tab("Pro"):
                        pro_out = _safe_textbox(
                            label="DIRECT + mosaic-aware QUASI", lines=10, show_copy_button=True,
                        )
                        pro_summary = gr.Markdown()
                    with gr.Tab("Audit log"):
                        audit_out = gr.Code(label="Full per-decision audit (JSON)", language="json")

        examples = _load_examples()
        if examples:
            gr.Examples(
                examples=examples,
                inputs=[input_text, k_target, max_iters],
                label="Pre-canned examples",
            )

        run_btn.click(
            fn=redact,
            inputs=[input_text, k_target, max_iters],
            outputs=[lite_out, pro_out, pro_summary, audit_out],
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

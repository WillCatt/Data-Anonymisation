# Demo

**Status:** Planned. Not yet implemented.

Once Phase 2 produces a fine-tuned model, this folder will host a live demo of the anonymisation pipeline — paste in a paragraph of legal-style text and see what gets redacted, with the mosaic risk score reported alongside.

## Likely deployment

* **Gradio app on HuggingFace Spaces** — free, embeddable in the portfolio site as an iframe.
* Two side-by-side panes: input text (left), redacted output with hover-to-explain entity spans (right).
* A small "Mosaic risk" indicator at the top: how unique is the QUASI fingerprint of this document.

## Fallback if the demo isn't ready by portfolio deadline

A 90-second screen recording of the notebook running on a sample doc, plus three pre-canned before/after screenshots embedded in the writeup.

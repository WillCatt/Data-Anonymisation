"""
Anonymisation toolkit — supporting code for the Legal Text Anonymisation
portfolio project.

Modules
-------
data         Load the Text Anonymization Benchmark (TAB) from HuggingFace.
mapping      TAB ↔ spaCy entity-type mapping (Phase 1 baseline).
evaluation   Span-level precision / recall / F1 scoring (partial + exact).
demo         Interactive `demo_anonymise()` helper for try-it-yourself cells.
mosaic       Re-identification / mosaic-effect helpers (k-anonymity, etc.).
predictors   (Phase 2) Adapters wrapping HF / Presidio / fine-tuned models
             behind the standard text → spans interface.
iob          (Phase 2) BIO tagging utilities for token-classification training.
device       (Phase 2) PyTorch device detection (CUDA → MPS → CPU).
"""
from .data import load_tab, get_dataset_summary
from .mapping import TAB_TO_SPACY, SPACY_TO_TAB, MAPPING_NOTES
from .evaluation import (
    EvalResult,
    spans_overlap,
    extract_gold_entities,
    predict_entities,
    evaluate_document,
    merge_results,
    results_to_dataframe,
)
from .demo import demo_anonymise
from .mosaic import quasi_identifier_signature, k_anonymity_table

__all__ = [
    "load_tab",
    "get_dataset_summary",
    "TAB_TO_SPACY",
    "SPACY_TO_TAB",
    "MAPPING_NOTES",
    "EvalResult",
    "spans_overlap",
    "extract_gold_entities",
    "predict_entities",
    "evaluate_document",
    "merge_results",
    "results_to_dataframe",
    "demo_anonymise",
    "quasi_identifier_signature",
    "k_anonymity_table",
]

# --- Phase 2 modules are imported lazily to keep Phase-1-only installs working ---
# Use them via:
#     from anonymisation.predictors import make_hf_predictor, ...
#     from anonymisation.iob import BIO_LABELS, offsets_to_bio, bio_to_spans
#     from anonymisation.device import best_device, report_device

"""
Predictor adapters — wrap each Phase-2 model behind a uniform
    text -> [(start, end, tab_type, span_text), ...]
interface, so the same evaluation framework drives all of them.

Each adapter is gated behind its optional dependency import so the rest
of the package still works on a Phase-1-only install.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

# ---------------------------------------------------------------------------
# HuggingFace token-classification pipeline (e.g. dslim/bert-base-NER)
# ---------------------------------------------------------------------------
# bert-base-NER outputs OntoNotes-flavour CoNLL-2003 labels:
#   PER, ORG, LOC, MISC
# We map them onto TAB types. DATETIME, QUANTITY, CODE and DEM have no
# equivalent — they will all fall to 0% recall, exactly the point we want
# the comparison to surface.
DEFAULT_HF_LABEL_TO_TAB: Dict[str, str] = {
    "PER":  "PERSON",
    "ORG":  "ORG",
    "LOC":  "LOC",
    "MISC": "MISC",
}


def make_hf_predictor(
    pipeline,
    label_to_tab: Dict[str, str] | None = None,
) -> Callable[[str], List[Tuple[int, int, str, str]]]:
    """
    Wrap a HuggingFace `transformers.pipeline("ner", aggregation_strategy=...)`
    into the standard predictor interface.

    Expected pipeline output (with `aggregation_strategy="simple"` or "first"):
        [{'entity_group': 'PER', 'word': 'Maria', 'start': 0, 'end': 5, 'score': 0.99}, ...]
    """
    mapping = label_to_tab or DEFAULT_HF_LABEL_TO_TAB

    def predict(text: str) -> List[Tuple[int, int, str, str]]:
        if not text.strip():
            return []
        out = []
        for ent in pipeline(text):
            label = ent.get("entity_group") or ent.get("entity")
            tab_type = mapping.get(label)
            if tab_type is None:
                continue
            start, end = int(ent["start"]), int(ent["end"])
            out.append((start, end, tab_type, text[start:end]))
        return out

    return predict


# ---------------------------------------------------------------------------
# Microsoft Presidio
# ---------------------------------------------------------------------------
# Presidio's default recogniser set covers more PII categories than spaCy
# (phone numbers, IBANs, etc.) but uses its own taxonomy. We map the
# overlap with TAB; out-of-scope categories (PHONE_NUMBER, EMAIL_ADDRESS, …)
# are dropped because TAB has no gold annotations for them — counting them
# as predictions would just inflate FP.
DEFAULT_PRESIDIO_LABEL_TO_TAB: Dict[str, str] = {
    "PERSON":          "PERSON",
    "LOCATION":        "LOC",
    "ORGANIZATION":    "ORG",
    "DATE_TIME":       "DATETIME",
    "NRP":             "DEM",      # nationality / religion / political group
    "CASE_NUMBER":     "CODE",     # custom recogniser added below
}


def build_presidio_analyzer(
    add_case_number_recognizer: bool = True,
    spacy_model: str = "en_core_web_lg",
):
    """
    Build a Presidio AnalyzerEngine with the default recogniser set, plus
    (optionally) a custom regex recogniser for ECHR application numbers
    such as "Application no. 12345/67". This is the "Presidio + bespoke
    recognizer" variant that closes spaCy's CODE blind spot.
    """
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    # Use the requested spaCy model under the hood
    nlp_config = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": spacy_model}],
    }
    provider = NlpEngineProvider(nlp_configuration=nlp_config)
    nlp_engine = provider.create_engine()

    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=["en"],
    )

    if add_case_number_recognizer:
        # ECHR / common-law application numbers: "Application no. 12345/67",
        # "App. No. 1234/00", "Case no. 12/2010", etc.
        case_no_pattern = Pattern(
            name="application_no",
            regex=r"\b(?:Application|App\.?|Case)\s+[Nn]o\.?\s*\d{1,7}/\d{2,4}\b",
            score=0.85,
        )
        recognizer = PatternRecognizer(
            supported_entity="CASE_NUMBER",
            patterns=[case_no_pattern],
            name="ApplicationNumberRecognizer",
            supported_language="en",
        )
        analyzer.registry.add_recognizer(recognizer)

    return analyzer


def make_presidio_predictor(
    analyzer,
    label_to_tab: Dict[str, str] | None = None,
    threshold: float = 0.4,
) -> Callable[[str], List[Tuple[int, int, str, str]]]:
    """
    Wrap a `presidio_analyzer.AnalyzerEngine` as a TAB-flavoured predictor.
    Results below `threshold` confidence are dropped.
    """
    mapping = label_to_tab or DEFAULT_PRESIDIO_LABEL_TO_TAB

    def predict(text: str) -> List[Tuple[int, int, str, str]]:
        if not text.strip():
            return []
        out = []
        results = analyzer.analyze(text=text, language="en")
        for r in results:
            if r.score < threshold:
                continue
            tab_type = mapping.get(r.entity_type)
            if tab_type is None:
                continue
            start, end = r.start, r.end
            out.append((start, end, tab_type, text[start:end]))
        return out

    return predict


# ---------------------------------------------------------------------------
# Fine-tuned token-classification model (Phase 2 contender)
# ---------------------------------------------------------------------------
def make_finetuned_predictor(
    model,
    tokenizer,
    *,
    device: str = "cpu",
    max_length: int = 512,
    stride: int = 64,
) -> Callable[[str], List[Tuple[int, int, str, str]]]:
    """
    Wrap a fine-tuned `AutoModelForTokenClassification` + `AutoTokenizer` pair
    into the standard predictor interface.

    Long documents are sliding-windowed with `stride` tokens of overlap; the
    final span list is deduplicated by (start, end, type).
    """
    import torch
    from .iob import bio_to_spans, ID_TO_LABEL  # local import — avoids hard dep at module load

    model.eval()
    model.to(device)

    def predict(text: str) -> List[Tuple[int, int, str, str]]:
        if not text.strip():
            return []
        encoded = tokenizer(
            text,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            truncation=True,
            max_length=max_length,
            stride=stride,
            padding=False,
        )

        all_spans: List[Tuple[int, int, str, str]] = []

        # Each "chunk" is a (max_length)-token window over the text
        for chunk_idx in range(len(encoded["input_ids"])):
            input_ids = torch.tensor([encoded["input_ids"][chunk_idx]], device=device)
            attn_mask = torch.tensor([encoded["attention_mask"][chunk_idx]], device=device)

            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attn_mask).logits[0]
            label_ids = logits.argmax(dim=-1).tolist()

            offsets = encoded["offset_mapping"][chunk_idx]
            spans = bio_to_spans(label_ids, offsets, text)
            all_spans.extend(spans)

        # Deduplicate — overlap windows can produce the same span twice
        seen = set()
        deduped: List[Tuple[int, int, str, str]] = []
        for s in all_spans:
            key = (s[0], s[1], s[2])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(s)
        return sorted(deduped, key=lambda s: s[0])

    return predict

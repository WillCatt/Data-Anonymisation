"""
Anonymiser Studio — the pipeline as an actual application.

What this is
------------
`demo/app.py` is a Gradio form: pick one variant, get one output. That is fine
for a hosted toy, but it cannot answer the question anyone actually asks, which
is *what is the difference between the three modes on my document*.

This serves a real front-end instead: one document, every mode computed at once,
each detected entity clickable to show its fate under all three treatments plus
the audit rationale that produced it.

It is deliberately dependency-light — FastAPI + uvicorn, both of which are
already in `legal-anon-env`. No Gradio, so it does not need the separate
`demo/venv-gradio` (which has no torch, and therefore has never been able to
load the fine-tuned model at all).

Run it:
    ./demo/run_studio.sh                    # http://127.0.0.1:7861
    ANON_DEVICE=mps ./demo/run_studio.sh    # if nothing else is using the GPU

Inference config
----------------
The predictor is pinned to max_length=384, stride=64 — the window the reported
F1 (0.8559, 95% CI [0.8475, 0.8644]) was measured at. `demo/app.py` never
passed these and silently ran at the 512 default, which is the same mismatch
that caused the two-numbers-for-one-model discrepancy in the evaluation path.
"""
from __future__ import annotations

import re

import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
STATIC = Path(__file__).resolve().parent / "static"

from anonymisation.mapping import SPACY_TO_TAB                                  # noqa: E402
from anonymisation.pipeline import (  # noqa: E402
    LitePipeline, ProPipeline, MosaicScorer, generalize, restore,
)
from anonymisation.pipeline.roles import DEFAULT_ROLE_BY_TYPE                   # noqa: E402
from anonymisation.pipeline.types import RedactionResult, Span                  # noqa: E402

# The window the reported numbers were measured at. Do not change these
# without re-running scripts/bootstrap_ci.py — see the module docstring.
MAX_LENGTH = 384
STRIDE = 64

FINETUNED_DIR = ROOT / "models" / "roberta-tab" / "final"
LEGALBERT_DIR = ROOT / "models" / "legalbert-tab" / "final"

# Headline metrics, so the UI can state what it is running rather than implying it.
MODEL_CARD = {
    "roberta": {"name": "RoBERTa, fine-tuned on TAB", "f1": 0.8559, "ci": [0.8475, 0.8644]},
    "legalbert": {"name": "LegalBERT, fine-tuned on TAB", "f1": 0.8538, "ci": [0.8432, 0.8638]},
    "spacy_trf": {"name": "spaCy en_core_web_trf (off the shelf)", "f1": 0.5656, "ci": [0.5544, 0.5764]},
    "spacy_sm": {"name": "spaCy en_core_web_sm (off the shelf, small)", "f1": None, "ci": None},
}

_state: Dict[str, object] = {
    "predictor": None,
    "backend_key": None,
    "device": None,
    "scorer": None,
    "scorer_status": "loading",
    "load_seconds": None,
}


# --------------------------------------------------------------------------- #
# Backend loading
# --------------------------------------------------------------------------- #
def _load_finetuned(directory: Path, device: str) -> Callable:
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from anonymisation.predictors import make_finetuned_predictor

    kwargs = {"add_prefix_space": True} if "roberta" in directory.parts[-2].lower() else {}
    tok = AutoTokenizer.from_pretrained(str(directory), **kwargs)
    model = AutoModelForTokenClassification.from_pretrained(str(directory))
    return make_finetuned_predictor(
        model, tok, device=device, max_length=MAX_LENGTH, stride=STRIDE,
    )


def _load_spacy(model_name: str) -> Callable:
    import spacy
    nlp = spacy.load(model_name)

    def predict(text: str):
        doc = nlp(text)
        return [
            (e.start_char, e.end_char, SPACY_TO_TAB[e.label_], e.text)
            for e in doc.ents if e.label_ in SPACY_TO_TAB
        ]
    return predict


def load_backend() -> None:
    """Load the strongest available detector. Fails loudly rather than silently."""
    from anonymisation.device import best_device

    requested = os.environ.get("ANON_BACKEND", "auto").strip().lower()
    device = os.environ.get("ANON_DEVICE", "").strip().lower()
    if not device:
        # Default to CPU: the GPU may well be busy with a training run, and a
        # paragraph through roberta-base on CPU is well under a second.
        device = "cpu"
    elif device == "auto":
        device = best_device()[0]

    started = time.time()
    attempts: List[Tuple[str, Callable[[], Callable]]] = []
    if requested in ("auto", "roberta", "finetuned") and FINETUNED_DIR.exists():
        attempts.append(("roberta", lambda: _load_finetuned(FINETUNED_DIR, device)))
    if requested in ("auto", "legalbert") and LEGALBERT_DIR.exists():
        attempts.append(("legalbert", lambda: _load_finetuned(LEGALBERT_DIR, device)))
    if requested in ("auto", "trf", "spacy_trf"):
        attempts.append(("spacy_trf", lambda: _load_spacy("en_core_web_trf")))
    if requested in ("auto", "sm", "spacy_sm"):
        attempts.append(("spacy_sm", lambda: _load_spacy("en_core_web_sm")))

    for key, make in attempts:
        try:
            _state["predictor"] = make()
            _state["backend_key"] = key
            _state["device"] = device
            _state["load_seconds"] = round(time.time() - started, 1)
            print(f"[studio] detector: {MODEL_CARD[key]['name']} on {device} "
                  f"({_state['load_seconds']}s, window {MAX_LENGTH}/{STRIDE})")
            return
        except Exception as exc:                                # noqa: BLE001
            print(f"[studio] could not load {key}: {exc}")

    raise SystemExit(
        "No detector could be loaded. Expected the fine-tune at "
        f"{FINETUNED_DIR} or a spaCy model installed in this environment."
    )


def load_scorer_async() -> None:
    """Build the mosaic haystack in the background so the first request is fast."""
    def _work():
        try:
            from anonymisation.data import load_tab
            ds = load_tab()
            _state["scorer"] = MosaicScorer.from_tab(list(ds["test"]))
            _state["scorer_status"] = "tab"
            print("[studio] mosaic haystack: TAB test split (555 documents)")
        except Exception as exc:                                # noqa: BLE001
            _state["scorer"] = MosaicScorer.empty()
            _state["scorer_status"] = "empty"
            print(f"[studio] mosaic haystack unavailable ({exc}); k-anonymity disabled")
    threading.Thread(target=_work, daemon=True).start()


def _scorer() -> MosaicScorer:
    scorer = _state.get("scorer")
    if scorer is None:                       # still loading — block briefly
        for _ in range(300):
            time.sleep(0.1)
            scorer = _state.get("scorer")
            if scorer is not None:
                break
    return scorer or MosaicScorer.empty()


# --------------------------------------------------------------------------- #
# Running the modes
# --------------------------------------------------------------------------- #
def _memoised(predictor: Callable) -> Callable:
    """
    Every pipeline calls the detector itself. We run four pipelines per request
    over identical text, so cache the one model pass and let the cheap parts
    (regex, coref, dedupe, role classification) re-run per pipeline — they must,
    because the pipelines mutate the Span objects they are given.
    """
    cache: Dict[str, list] = {}

    def predict(text: str):
        if text not in cache:
            cache[text] = predictor(text)
        return list(cache[text])
    return predict


def _segments(text: str, result: RedactionResult) -> List[dict]:
    """
    Rebuild the redacted text as an ordered run of kept and replaced segments,
    so the front-end can highlight what changed and animate between modes.

    Asserted against the pipeline's own output — if these ever disagree the
    rendering is lying about what the pipeline did, and we want to know.
    """
    spans = sorted(result.spans, key=lambda s: s.start)
    out: List[dict] = []
    cursor = 0
    for s in spans:
        if s.start < cursor:                  # defensive; dedupe should prevent it
            continue
        if s.start > cursor:
            out.append({"kind": "keep", "text": text[cursor:s.start]})
        replacement = s.replacement if s.replacement is not None else s.text
        out.append({
            "kind": "span",
            "id": f"{s.start}-{s.end}",
            "before": s.text,
            "text": replacement,
            "type": s.entity_type,
            "role": s.identifier_role,
            "source": s.source,
            "level": s.generalization_level,
            "changed": s.replacement is not None,
        })
        cursor = s.end
    if cursor < len(text):
        out.append({"kind": "keep", "text": text[cursor:]})

    rebuilt = "".join(seg["text"] for seg in out)
    if rebuilt != result.redacted_text:
        # Never silently render something other than what the pipeline produced.
        out = [{"kind": "keep", "text": result.redacted_text}]
        out.append({"kind": "warning", "text": "segment rebuild mismatch"})
    return out


def _audit_rows(result: RedactionResult) -> List[dict]:
    rows = []
    for e in result.audit:
        rows.append({
            "id": f"{e.span.start}-{e.span.end}",
            "text": e.span.text,
            "type": e.span.entity_type,
            "role": e.span.identifier_role,
            "source": e.span.source,
            "action": e.action,
            "rationale": e.rationale,
            "iteration": e.iteration,
            "replacement": e.span.replacement,
        })
    return rows


def _link_rows(result: RedactionResult) -> Dict[str, dict]:
    """
    Pair every pseudonym link decision with the span it resolved.

    The pipelines call `token_for` once per DIRECT span, in span order, so the
    decisions line up with those spans one for one. That ordering is an
    assumption, so it is checked rather than trusted: if the surface forms stop
    matching, return nothing and show nothing, because a link attributed to the
    wrong phrase is worse than no link at all.
    """
    direct = [s for s in result.spans if s.identifier_role == "DIRECT"]
    if len(direct) != len(result.pseudonym_links):
        return {}
    rows: Dict[str, dict] = {}
    for span, decision in zip(direct, result.pseudonym_links):
        if decision.surface_form != span.text.strip():
            return {}
        rows[f"{span.start}-{span.end}"] = {
            "token": decision.token,
            "evidence": decision.evidence,
            "confidence": decision.confidence,
            "matched_form": decision.matched_form,
            "tied": decision.tied_candidates,
            "refusal": decision.refusal,
            "rationale": decision.describe(),
            "merged": decision.merged,
        }
    return rows


def _mode_payload(text: str, result: RedactionResult) -> dict:
    return {
        "text": result.redacted_text,
        "segments": _segments(text, result),
        "audit": _audit_rows(result),
        "vault": result.pseudonym_vault,
        "links": _link_rows(result),
        "mosaic": {
            "initial": result.mosaic_risk_initial,
            "final": result.mosaic_risk_final,
            "iterations": result.iterations_used,
            "converged": result.converged,
        },
    }


class AnalyseRequest(BaseModel):
    text: str
    k_target: int = 5
    max_iterations: int = 3
    coref: bool = True
    # Entity types the reader has chosen to keep in the clear. They are still
    # detected and still counted in the mosaic fingerprint — the risk figure
    # reflects the document that actually leaves the building.
    keep_types: List[str] = []


class RestoreRequest(BaseModel):
    text: str
    vault: Dict[str, str]


_NO_STORE = {"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"}

app = FastAPI(title="Anonymiser Studio")


@app.get("/api/health")
def health() -> dict:
    key = _state.get("backend_key")
    card = MODEL_CARD.get(key, {}) if key else {}
    return {
        "ready": _state.get("predictor") is not None,
        "backend": key,
        "model_name": card.get("name"),
        "f1": card.get("f1"),
        "ci": card.get("ci"),
        "device": _state.get("device"),
        "window": f"{MAX_LENGTH}/{STRIDE}",
        "load_seconds": _state.get("load_seconds"),
        "haystack": _state.get("scorer_status"),
        "is_finetuned": key in ("roberta", "legalbert"),
    }


@app.post("/api/analyse")
def analyse(req: AnalyseRequest) -> JSONResponse:
    text = req.text
    if not text.strip():
        return JSONResponse({"error": "empty document"}, status_code=400)

    started = time.time()
    predict = _memoised(_state["predictor"])           # one model pass, four pipelines
    scorer = _scorer()

    common = {"coref_extend": req.coref, "exempt_types": tuple(req.keep_types)}
    redact = LitePipeline(ner_provider=predict, **common)(text)
    pseudo = LitePipeline(ner_provider=predict, pseudonymise=True, **common)(text)
    anonymise = ProPipeline(
        ner_provider=predict, scorer=scorer,
        k_target=req.k_target, max_iterations=req.max_iterations, **common,
    )(text)
    anonymise_pseudo = ProPipeline(
        ner_provider=predict, scorer=scorer, pseudonymise=True,
        k_target=req.k_target, max_iterations=req.max_iterations, **common,
    )(text)

    modes = {
        "redact": _mode_payload(text, redact),
        "anonymise": _mode_payload(text, anonymise),
        "pseudonymise": _mode_payload(text, pseudo),
        "pseudonymise_pro": _mode_payload(text, anonymise_pseudo),
    }

    # What each entity is, and what happens to it in every mode — this is the
    # comparison the single-variant demo could never show.
    by_id: Dict[str, dict] = {}
    for mode_key, payload in modes.items():
        for row in payload["audit"]:
            entry = by_id.setdefault(row["id"], {
                "id": row["id"], "text": row["text"], "type": row["type"],
                "role": row["role"], "source": row["source"], "fates": {},
            })
            entry["fates"][mode_key] = {
                "action": row["action"],
                "replacement": row["replacement"],
                "rationale": row["rationale"],
                "iteration": row["iteration"],
                "steps": [],
                # Only the pseudonymise modes carry one: it is the record of
                # which entity this mention was decided to be, and on what.
                "link": payload["links"].get(row["id"]),
            }
    entities = sorted(by_id.values(), key=lambda e: int(e["id"].split("-")[0]))

    # The per-iteration path a quasi-identifier took.
    #
    # It cannot be read off the audit rows: AuditEntry holds a *reference* to
    # the Span, and the Pro loop keeps mutating that span after appending each
    # entry, so every iteration's `replacement` field ends up showing the final
    # value. (The rationale string, formatted at the time, is the only place the
    # intermediate value survives.) Recompute it from the same generalize()
    # the pipeline calls, which is exact rather than parsed.
    for entry in entities:
        if entry["role"] != "QUASI":
            continue
        for mode_key in ("anonymise", "pseudonymise_pro"):
            fate = entry["fates"].get(mode_key)
            if not fate:
                continue
            used = modes[mode_key]["mosaic"]["iterations"] or 0
            fate["steps"] = [
                {"iteration": lv, "replacement": generalize(entry["type"], entry["text"], lv)}
                for lv in range(1, used + 1)
            ]

    # How k moved as the loop broadened — the pipeline logs it in the rationale
    # of each generalize entry, one value per iteration.
    k_path: List[dict] = []
    seen_iters = set()
    for row in modes["anonymise"]["audit"]:
        m = re.search(r"Post-step k=(\d+)", row["rationale"] or "")
        if m and row["iteration"] not in seen_iters:
            seen_iters.add(row["iteration"])
            k_path.append({"iteration": row["iteration"], "k": int(m.group(1))})
    k_path.sort(key=lambda r: r["iteration"])

    # Coref is a kept null on TAB (+0.0001 macro recall). Show exactly what it
    # adds on *this* document rather than claiming a gain it does not have.
    coref_added: List[dict] = []
    if req.coref:
        plain = LitePipeline(ner_provider=predict, coref_extend=False)(text)
        baseline_ids = {f"{s.start}-{s.end}" for s in plain.spans}
        coref_added = [
            {"id": f"{s.start}-{s.end}", "text": s.text, "type": s.entity_type}
            for s in redact.spans if f"{s.start}-{s.end}" not in baseline_ids
        ]

    counts: Dict[str, int] = {}
    for s in redact.spans:
        counts[s.entity_type] = counts.get(s.entity_type, 0) + 1

    return JSONResponse({
        "source": text,
        "modes": modes,
        "entities": entities,
        "coref": {"added": coref_added, "enabled": req.coref},
        "k_path": k_path,
        "counts": counts,
        "roles": DEFAULT_ROLE_BY_TYPE,
        "haystack": _state.get("scorer_status"),
        "elapsed_ms": int((time.time() - started) * 1000),
    })


@app.post("/api/restore")
def restore_text(req: RestoreRequest) -> dict:
    """The round trip: an answer that came back from an LLM, re-identified locally."""
    return {"restored": restore(req.text, req.vault)}


@app.get("/api/samples")
def samples() -> List[dict]:
    folder = ROOT / "demo" / "examples"
    fixups = {"echr": "ECHR", "hr": "HR", "ip": "IP", "ma": "M&A", "b": "", "c": ""}
    out = []
    for path in sorted(folder.glob("*.txt")):
        stem = re.sub(r"^sample_\d*_?", "", path.stem)
        words = [fixups.get(w, w) for w in stem.split("_")]
        label = " ".join(w for w in words if w).capitalize()
        label = re.sub(r"\b(echr|hr|ip|m&a)\b", lambda m: fixups.get(m.group(1), m.group(1).upper()),
                       label, flags=re.I)
        text = path.read_text().strip()
        out.append({
            "id": path.stem,
            "label": f"{label} · {len(text.split())} words",
            "text": text,
        })
    return out


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html", headers=_NO_STORE)


class _FreshStatic(StaticFiles):
    """
    Static files, never cached.

    This is a development tool that is edited while it is running. A browser
    holding on to yesterday's app.js looks exactly like a broken feature, and
    costs more than the handful of milliseconds re-fetching a 20 KB file saves.
    """

    def is_not_modified(self, response_headers, request_headers) -> bool:
        return False

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers.update(_NO_STORE)
        return response


app.mount("/static", _FreshStatic(directory=str(STATIC)), name="static")


def main() -> None:
    import uvicorn
    load_backend()
    load_scorer_async()
    port = int(os.environ.get("PORT", "7861"))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"[studio] http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()

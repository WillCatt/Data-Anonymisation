"""
Build the static side-by-side showcase HTML.

Runs LitePipeline and ProPipeline on the three example inputs in
demo/examples/, renders a single self-contained HTML page at
demo/index.html. No runtime dependencies for visitors — they just
open the file in a browser.

Why hand-crafted entity spans instead of running spaCy here?
    A static showcase wants deterministic, predictable output: the
    same pipelines, the same inputs, the same answers, no off-by-one
    NER drift between runs. We hand-write the spans for these three
    inputs once. The pipeline code under test is exactly the same as
    Phase 3 production — only the NER step is fixed.

Why a synthetic haystack?
    The mosaic scorer needs *something* to compare against. TAB itself
    is the right choice in real use (1,268 ECHR cases) but the download
    is ~50 MB and locks the demo to a network dependency. For the
    static showcase we hand-build a tiny haystack engineered so the
    three samples each demonstrate a different convergence outcome:
        Sample 1 — converges at level 1 (mild generalization)
        Sample 2 — converges at level 0 (mostly DIRECTs; little QUASI work)
        Sample 3 — converges at level 2 (deeper generalization)
    The HTML calls this out explicitly so a reader doesn't conflate
    the showcase haystack with what a deployment would use.

Run with:
    python demo/build_showcase.py

Output:
    demo/index.html
"""
from __future__ import annotations

import html
import json
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

# Allow `from anonymisation.pipeline import ...` whether run from repo
# root or from inside demo/.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from anonymisation.pipeline import (  # noqa: E402
    LitePipeline, ProPipeline, MosaicScorer, RedactionResult,
)


# ---------------------------------------------------------------------------
# Hand-crafted entity spans for each example. Computed at runtime via
# `text.find()` so they stay correct if the example file changes.
# ---------------------------------------------------------------------------
def _spans_via_find(text: str, items: List[Tuple[str, str]]) -> List[Tuple[int, int, str, str]]:
    """Locate each (substring, entity_type) and emit (start, end, etype, text)."""
    out: List[Tuple[int, int, str, str]] = []
    for needle, etype in items:
        idx = text.find(needle)
        if idx == -1:
            raise SystemExit(f"Could not find {needle!r} in example text")
        out.append((idx, idx + len(needle), etype, needle))
    return out


def make_predictor(text: str, items: List[Tuple[str, str]]):
    """Return a NER-style predictor that always emits the precomputed spans."""
    spans = _spans_via_find(text, items)

    def predict(_text: str):
        return spans
    return predict


SAMPLES = [
    {
        "title": "Sample 1 — demographic-heavy",
        "blurb": (
            "ECHR-style application: a single individual described by name, age, "
            "nationality, ethnicity, location, occupation, and date of complaint. "
            "Lots of QUASI surface area; Pro generalizes them to a milder form."
        ),
        "file": "examples/sample_1_demographic_heavy.txt",
        "entities": [
            ("Maria Petrova",        "PERSON"),
            ("47-year-old",          "QUANTITY"),
            ("Bulgarian",            "DEM"),
            ("Plovdiv",              "LOC"),
            ("12 March 2018",        "DATETIME"),
            ("Sofia District Court", "ORG"),
            ("Roma",                 "DEM"),
            ("2010",                 "DATETIME"),
        ],
        "k_target": 5,
        "max_iterations": 3,
    },
    {
        "title": "Sample 2 — corporate litigation",
        "blurb": (
            "Settlement agreement summary: company, person, case number, financial "
            "amount, IBAN, dates. Most of the identifying load is DIRECT — Lite and "
            "Pro produce nearly the same output here."
        ),
        "file": "examples/sample_2_corporate.txt",
        "entities": [
            ("Acme Holdings Ltd", "ORG"),
            ("4 July 2022",       "DATETIME"),
            ("John Doe",          "PERSON"),
            ("January 2019",      "DATETIME"),
            ("June 2022",         "DATETIME"),
            ("Manchester",        "LOC"),
            ("£450,000",          "QUANTITY"),
        ],
        "k_target": 5,
        "max_iterations": 3,
    },
    {
        "title": "Sample 3 — medical / structured ID",
        "blurb": (
            "Tighter document with a DOB, location, occupation, sensitive medical "
            "context, and an NHS number. The regex pass catches the NHS number. "
            "Pro converges at deeper generalization because the QUASI fingerprint "
            "is otherwise quite distinctive."
        ),
        "file": "examples/sample_3_minimal.txt",
        "entities": [
            ("Sarah Khan",        "PERSON"),
            ("14 February 1985",  "DATETIME"),
            ("London",            "LOC"),
            ("2020",              "DATETIME"),
            ("FinTech Co",        "ORG"),
        ],
        "k_target": 5,
        "max_iterations": 3,
    },
]


# ---------------------------------------------------------------------------
# Synthetic haystack — engineered to drive specific convergence outcomes
# ---------------------------------------------------------------------------
def build_haystack() -> MosaicScorer:
    """
    Build a small haystack of QUASI fingerprints so the three samples each
    converge at a useful level.

    The haystack is illustrative, not real. In a deployment, this comes
    from the firm's own document corpus (or, for evaluation, from TAB).
    """
    sigs = []

    # ── Sample 1 should converge at level 1 ─────────────────────────────
    # Level-1 generalization of sample 1 produces:
    #   QUANTITY '47-year-old'   → 'about 40'
    #   DEM      'Bulgarian'     → 'European'
    #   DEM      'Roma'          → 'ethnic minority'
    #   LOC      'Plovdiv'       → 'Bulgaria'
    #   DATETIME '12 March 2018' → '2018'
    #   DATETIME '2010'          → '2010'
    # Repeat that fingerprint 6 times so k=6 ≥ k_target=5.
    s1_l1 = tuple(sorted({
        ("DATETIME", "2010"),
        ("DATETIME", "2018"),
        ("DEM",      "european"),
        ("DEM",      "ethnic minority"),
        ("LOC",      "bulgaria"),
        ("QUANTITY", "about 40"),
    }))
    sigs.extend([s1_l1] * 6)

    # ── Sample 2 should converge at level 0 (no QUASI work) ────────────
    # Initial signature (post-DIRECT redaction):
    s2_l0 = tuple(sorted({
        ("DATETIME", "4 july 2022"),
        ("DATETIME", "january 2019"),
        ("DATETIME", "june 2022"),
        ("LOC",      "manchester"),
        ("QUANTITY", "£450,000"),
    }))
    sigs.extend([s2_l0] * 6)

    # ── Sample 3 should converge at level 2 ────────────────────────────
    # Level-2 of sample 3:
    #   DATETIME '14 February 1985' → '1980s'
    #   DATETIME '2020'             → '2020s'
    #   LOC      'London'           → 'Europe'
    s3_l2 = tuple(sorted({
        ("DATETIME", "1980s"),
        ("DATETIME", "2020s"),
        ("LOC",      "europe"),
    }))
    sigs.extend([s3_l2] * 6)

    # Plus a sprinkle of unrelated noise so the haystack feels less hand-built
    sigs.extend([
        (("DATETIME", "1999"), ("LOC", "athens"))             ,
        (("DEM", "european"), ("LOC", "italy"))               ,
        (("DATETIME", "2005"), ("DEM", "european"))           ,
        (("LOC", "spain"), ("QUANTITY", "about 60"))          ,
    ])

    return MosaicScorer(sigs)


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------
CSS = """
:root {
  --bg: #fafbfc;
  --panel: #ffffff;
  --border: #e1e4e8;
  --text: #24292e;
  --muted: #586069;
  --accent: #0366d6;
  --green: #2ea44f;
  --amber: #e36209;
  --red:   #d73a49;
  --code-bg: #f6f8fa;
  --tag-bg: #fff5b1;
  --tag-text: #735c0f;
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  background: var(--bg);
  color: var(--text);
  margin: 0; padding: 2rem 1rem;
  line-height: 1.5;
}
.container { max-width: 1100px; margin: 0 auto; }
header { margin-bottom: 2rem; }
header h1 { margin: 0 0 0.5rem; font-size: 1.6rem; }
header p { color: var(--muted); margin: 0.25rem 0; }
.tag-row { margin: 0.75rem 0 0; }
.tag { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 1rem;
       background: #eaf5ff; color: var(--accent); font-size: 0.8rem; margin-right: 0.5rem; }

.tabs { display: flex; flex-wrap: wrap; gap: 0.25rem; border-bottom: 1px solid var(--border); margin-bottom: 1rem; }
.tabs label {
  cursor: pointer;
  padding: 0.5rem 1rem;
  border: 1px solid transparent;
  border-bottom: none;
  border-radius: 6px 6px 0 0;
  font-size: 0.95rem;
  color: var(--muted);
}
.tabs input[type=radio] { display: none; }
.tabs label:hover { color: var(--text); }
.panel { display: none; }

/* CSS-only tab switching */
#tab1:checked ~ .tabs label[for=tab1],
#tab2:checked ~ .tabs label[for=tab2],
#tab3:checked ~ .tabs label[for=tab3] {
  background: var(--panel); color: var(--text);
  border-color: var(--border); border-bottom-color: var(--panel);
  margin-bottom: -1px;
}
#tab1:checked ~ .panels #panel1,
#tab2:checked ~ .panels #panel2,
#tab3:checked ~ .panels #panel3 { display: block; }

.sample-blurb {
  background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
  padding: 0.75rem 1rem; color: var(--muted); margin-bottom: 1rem; font-size: 0.95rem;
}
.cols { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; }
@media (max-width: 800px) { .cols { grid-template-columns: 1fr; } }
.col {
  background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
  padding: 1rem; min-height: 200px;
}
.col h3 { margin: 0 0 0.5rem; font-size: 1rem; }
.col h3 .badge {
  display: inline-block; font-size: 0.7rem; padding: 0.1rem 0.45rem;
  border-radius: 3px; background: #f1f8ff; color: var(--accent);
  margin-left: 0.5rem; vertical-align: middle;
}
.col h3 .badge.pro { background: #e6ffed; color: var(--green); }
.col h3 .badge.lite { background: #fff5b1; color: var(--tag-text); }
.col .body { white-space: pre-wrap; font-size: 0.9rem; line-height: 1.55; }
.col .body .tag {
  background: var(--tag-bg); color: var(--tag-text);
  padding: 0 0.25rem; border-radius: 3px; font-weight: 600;
  font-family: ui-monospace, SFMono-Regular, monospace;
  font-size: 0.85em;
}
.mosaic {
  margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px dashed var(--border);
  font-size: 0.85rem; color: var(--muted);
}
.mosaic .k {
  display: inline-block; padding: 0.1rem 0.45rem; border-radius: 3px;
  font-family: ui-monospace, SFMono-Regular, monospace; font-weight: 600;
  background: var(--code-bg); color: var(--text);
}
.mosaic .converged { color: var(--green); }
.mosaic .unconverged { color: var(--red); }

details {
  margin-top: 1.5rem; background: var(--panel); border: 1px solid var(--border);
  border-radius: 6px; padding: 0.5rem 1rem;
}
details summary {
  cursor: pointer; font-size: 0.9rem; color: var(--muted); padding: 0.5rem 0;
}
details[open] summary { color: var(--text); border-bottom: 1px solid var(--border);
                        margin-bottom: 0.75rem; padding-bottom: 0.75rem; }
.audit-table {
  width: 100%; border-collapse: collapse; font-size: 0.82rem;
  font-family: ui-monospace, SFMono-Regular, monospace;
}
.audit-table th, .audit-table td {
  text-align: left; padding: 0.3rem 0.5rem; border-bottom: 1px solid var(--border);
  vertical-align: top;
}
.audit-table th { background: var(--code-bg); color: var(--muted); font-weight: 600; }
.audit-table td.iter { color: var(--muted); }
.audit-table td.action.redact     { color: var(--red); }
.audit-table td.action.generalize { color: var(--amber); }
.audit-table td.action.leave      { color: var(--green); }

footer {
  margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid var(--border);
  font-size: 0.85rem; color: var(--muted);
}
footer p { margin: 0.5rem 0; }
footer a { color: var(--accent); text-decoration: none; }
footer a:hover { text-decoration: underline; }

.callout {
  background: #fffbea; border-left: 3px solid var(--amber);
  padding: 0.75rem 1rem; margin: 1rem 0; font-size: 0.9rem; color: #735c0f;
  border-radius: 0 4px 4px 0;
}
"""


def _highlight_tags(text: str) -> str:
    """Wrap [TYPE] tokens in a span so CSS can colour them."""
    import re
    escaped = html.escape(text)
    return re.sub(
        r"(\[(?:PERSON|ORG|CODE|MISC|LOC|DATETIME|QUANTITY|DEM)\])",
        r'<span class="tag">\1</span>',
        escaped,
    )


def render_sample(idx: int, sample: dict, lite_result: RedactionResult,
                  pro_result: RedactionResult, original: str) -> str:
    """Render one sample's panel."""
    pro_converged_class = "converged" if pro_result.converged else "unconverged"
    pro_status = "✓ converged" if pro_result.converged else "✗ did not converge"
    audit_rows = []
    for entry in pro_result.audit:
        s = entry.span
        original_t = html.escape(s.text)
        replacement_t = html.escape(s.replacement) if s.replacement else "—"
        audit_rows.append(
            f'<tr><td class="iter">{entry.iteration}</td>'
            f'<td>{html.escape(s.entity_type)}</td>'
            f'<td>{html.escape(s.identifier_role)}</td>'
            f'<td class="action {entry.action}">{entry.action}</td>'
            f'<td>{original_t}</td>'
            f'<td>→ {replacement_t}</td></tr>'
        )
    audit_html = (
        '<table class="audit-table">'
        '<tr><th>iter</th><th>type</th><th>role</th><th>action</th><th>original</th><th>replacement</th></tr>'
        + "".join(audit_rows) +
        '</table>'
    )

    return f"""
    <div class="panel" id="panel{idx}">
      <div class="sample-blurb">{html.escape(sample['blurb'])}</div>
      <div class="cols">
        <div class="col">
          <h3>Original</h3>
          <div class="body">{html.escape(original)}</div>
        </div>
        <div class="col">
          <h3>Lite <span class="badge lite">DIRECT-only</span></h3>
          <div class="body">{_highlight_tags(lite_result.redacted_text)}</div>
        </div>
        <div class="col">
          <h3>Pro <span class="badge pro">DIRECT + mosaic-aware QUASI</span></h3>
          <div class="body">{_highlight_tags(pro_result.redacted_text)}</div>
          <div class="mosaic">
            <span class="k">k_initial = {pro_result.mosaic_risk_initial}</span>
            <span> → </span>
            <span class="k">k_final = {pro_result.mosaic_risk_final}</span>
            <span> · iterations: {pro_result.iterations_used} · </span>
            <span class="{pro_converged_class}">{pro_status}</span>
          </div>
        </div>
      </div>
      <details>
        <summary>Pro audit log (every decision the pipeline made)</summary>
        {audit_html}
      </details>
    </div>
    """


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    demo_dir = repo_root / "demo"

    scorer = build_haystack()

    panels = []
    radios = []
    tabs = []
    for i, sample in enumerate(SAMPLES, start=1):
        text = (demo_dir / sample["file"]).read_text().strip()
        predictor = make_predictor(text, sample["entities"])
        lite = LitePipeline(ner_provider=predictor)
        pro = ProPipeline(
            ner_provider=predictor, scorer=scorer,
            k_target=sample["k_target"], max_iterations=sample["max_iterations"],
        )
        lite_result = lite(text)
        pro_result = pro(text)
        panels.append(render_sample(i, sample, lite_result, pro_result, text))
        radios.append(
            f'<input type="radio" name="tab" id="tab{i}"' + (' checked' if i == 1 else '') + '>'
        )
        tabs.append(f'<label for="tab{i}">{html.escape(sample["title"])}</label>')

    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Legal Text Anonymisation — Lite vs Pro</title>
  <style>{CSS}</style>
</head>
<body>
<div class="container">
  <header>
    <h1>Legal Text Anonymisation — Lite vs Pro</h1>
    <p>Two redaction pipelines, side by side, on three illustrative legal-style inputs.</p>
    <div class="tag-row">
      <span class="tag">Lite — DIRECT-only</span>
      <span class="tag">Pro — DIRECT + mosaic-aware QUASI</span>
    </div>
  </header>

  <div class="callout">
    <strong>About this showcase.</strong> The pipelines are the same code as
    in <code>src/anonymisation/pipeline/</code>. Entity spans here are
    hand-curated for deterministic output; in production they come from the
    spaCy / RoBERTa NER from earlier phases. The mosaic scorer's haystack is
    a small synthetic corpus engineered to drive useful convergence behavior
    across the three samples — in deployment it would be the firm's own
    document corpus.
  </div>

  {''.join(radios)}
  <div class="tabs">{''.join(tabs)}</div>
  <div class="panels">
    {''.join(panels)}
  </div>

  <footer>
    <p>Source: <a href="https://github.com/wcatt98/data-anonymisation">github.com/wcatt98/data-anonymisation</a> ·
       Built from the Phase 3 pipelines.
       The interactive Gradio version is on the roadmap once the dependency
       situation calms down — see <code>demo/app.py</code>.</p>
  </footer>
</div>
</body>
</html>
"""
    out = demo_dir / "index.html"
    out.write_text(body)
    print(f"Wrote {out} ({len(body):,} bytes)")


if __name__ == "__main__":
    main()

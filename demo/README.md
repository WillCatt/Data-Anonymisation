# Demo — Lite vs Pro

A static, self-contained HTML page showing both pipelines running side-by-side on three illustrative inputs. No backend, no runtime dependencies, no hosting cost — just open `demo/index.html` in a browser.

## View it

```bash
open demo/index.html         # macOS
xdg-open demo/index.html     # Linux
start demo/index.html        # Windows
```

Or push the file to any static host (GitHub Pages, your portfolio site, S3) and `<iframe>` it in:

```html
<iframe src="path/to/index.html" width="100%" height="900" style="border:0"></iframe>
```

## Regenerate after editing examples or pipeline code

```bash
python demo/build_showcase.py
```

Reads the example texts from `demo/examples/`, runs both pipelines, and writes a fresh `demo/index.html`. Takes a couple of seconds; no external deps beyond what's already in the project venv.

## What's in the showcase

Three sample inputs, each chosen to exercise a different convergence path through the Pro pipeline:

| # | Sample | What it shows |
|---|---|---|
| 1 | Demographic-heavy ECHR-style application | Pro converges at level 1 — mild generalization (`47-year-old` → `about 40`, `Bulgarian` → `European`, `Plovdiv` → `Bulgaria`, etc.) |
| 2 | Corporate litigation summary | Pro converges at level 0 — no QUASI work needed because most of the identifying load is already DIRECT |
| 3 | Medical / structured-ID record | Pro converges at level 2 — deeper generalization (`14 February 1985` → `1980s`, `London` → `Europe`) |

Each sample renders three columns: original input, Lite output, Pro output. The Pro column has a mosaic-risk badge (`k_initial → k_final`, iteration count, convergence status) and an expandable audit log table showing every decision the pipeline made.

## Important caveats

These are written into the showcase itself so a viewer can't miss them, but worth re-stating:

- **Entity spans are hand-curated for the showcase**, not produced by an NER model. The pipelines are exactly the same code as `src/anonymisation/pipeline/`; only the NER input is fixed for deterministic output.
- **The mosaic scorer's haystack is synthetic** — 22 hand-built fingerprints engineered to drive the three convergence outcomes above. In a deployment, the haystack is the firm's own document corpus; for evaluation it would be TAB.
- **The generalization rules are intentionally simple** (lookup tables for cities/countries, year/decade extraction for dates, broadening tables for nationality/ethnicity). A real deployment would have richer rules or an LLM-driven fallback.

## Folder layout

```
demo/
├── README.md                          this file
├── index.html                         the showcase (regenerable)
├── build_showcase.py                  generator script
├── examples/
│   ├── sample_1_demographic_heavy.txt
│   ├── sample_2_corporate.txt
│   └── sample_3_minimal.txt
├── app.py                             ⚠️ deferred Gradio version (see below)
└── requirements.txt                   used only by the deferred Gradio app
```

## About `demo/app.py` (Gradio version — deferred)

The interactive Gradio app is included for future use but is currently parked. Gradio's recent dependency tree (HuggingFace Hub, Typer, FastAPI versions) conflicts with spaCy 3.7's pinned deps in non-trivial ways. Rather than fight that for a demo, I went with the static showcase you're looking at.

If you want to revive it later, the route is:

```bash
pip install --force-reinstall \
    "gradio>=4.20.0,<5.0.0" \
    "huggingface-hub>=0.19.0,<0.27.0" \
    "typer>=0.3.0,<0.10.0"
python demo/app.py
```

The app code is already version-tolerant (handles Gradio 4 / 6 API differences gracefully). The hard part is just the dep cascade.

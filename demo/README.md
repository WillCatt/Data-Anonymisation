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

## Interactive Gradio demo — separate venv

The `demo/app.py` Gradio version runs in its own isolated venv so the
dep-tree conflicts (Gradio vs spaCy / typer / huggingface-hub) can't touch
the main repo venv.

**First run:**

```bash
./demo/run_gradio.sh
```

That script will:
1. Create `demo/venv-gradio/` (gitignored)
2. Install pinned versions from `demo/requirements-gradio.txt`
3. Download `en_core_web_sm` into that venv
4. Launch the Gradio app at <http://localhost:7860>

**Subsequent runs** just activate the existing venv and launch — typically a
few seconds.

### NER backend — auto-detected at startup

The app picks the strongest available backend at boot, in this order:

1. **Phase 2 fine-tuned RoBERTa** if `phase2_baseline_comparison/checkpoints/roberta-tab/final/`
   exists. (Best quality. Run `phase2_baseline_comparison/notebooks/03_finetune_roberta.ipynb`
   first to produce it.) Note: also needs `transformers` + `torch` installed
   in the gradio venv — add them to `demo/requirements-gradio.txt` if you go
   this route.
2. **spaCy `en_core_web_trf`** (transformer) — if installed in the gradio venv.
3. **spaCy `en_core_web_sm`** (small, the bootstrap default).

The UI shows which backend is loaded under the description. Override the
auto-detection with `PIPELINE_BACKEND=spacy ./demo/run_gradio.sh`.

### Features exposed in the UI

- **Lite vs Pro** variant selector (radio button)
- **k_target** slider (Pro only — mosaic-anonymity threshold)
- **Pseudonymise** toggle — switch to referential tokens (`[PERSON_A]`, …)
  with a vault tab showing the mapping
- **Coref extension** toggle — Phase 5's shorthand-recovery post-processor
  (defaults on)
- Pre-canned examples that exercise all four sample categories

**Three demo surfaces — which is which:**

| | Static showcase | Local Gradio | HF Spaces |
|---|---|---|---|
| File | `demo/index.html` | `demo/app.py` | `spaces/app.py` |
| Backend | None | Python + Gradio | Python + Gradio |
| Setup | Zero — open in browser | `./demo/run_gradio.sh` | See `spaces/README.md` |
| Inputs | 12 pre-canned examples | Paste any text | Paste any text |
| NER backend | n/a (pre-computed) | Auto-detect (RoBERTa > trf > sm) | spaCy `_sm` (fast cold-start) |
| Use case | Portfolio embed (iframe) | Local demo + screen-recording | Public demo, iframe-able |
| Hosting cost | $0 (static file) | $0 (local) | $0 (HF free tier) |

All three are kept in sync — the static showcase is the embedded artefact, the
local Gradio is for development and screen-recording demos, and the HF Space
is the public live version for portfolio visitors. See `../spaces/README.md`
for deploying to HuggingFace Spaces.

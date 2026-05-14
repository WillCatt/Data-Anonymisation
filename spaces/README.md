---
title: Legal Text Anonymisation
emoji: 🔒
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
short_description: Lite vs Pro redaction pipelines for legal text. Phase 1–5 of the data-anonymisation project.
---

# Legal Text Anonymisation — HuggingFace Spaces deployment

This folder contains everything needed to deploy the Gradio app as a public
HuggingFace Space.

## What's in this Space

A trimmed-down version of the main project's Gradio app:

- **NER backbone:** `en_core_web_sm` (small spaCy model — 12 MB) for fast
  cold-start on the free Space tier. The full project's strongest model
  is the Phase 2 RoBERTa fine-tune (~500 MB), which would push the image
  size past 1 GB and slow boot. Running locally via `./demo/run_gradio.sh`
  uses the fine-tuned model when available.
- **Phase 5 CorefExtender:** enabled (catches shorthand mentions).
- **Phase 3 Lite + Pro pipelines** with mosaic-aware QUASI generalisation.
- **Phase 4 Pseudonymisation:** referential tokens + vault display.

Everything else (the audit log, the round-trip pattern, the example
documents) is unchanged from the main project.

## Files

```
spaces/
├── README.md           ← this file (with HF Spaces YAML frontmatter)
├── app.py              ← entry point HF Spaces runs
├── requirements.txt    ← Python deps installed at build time
├── pre-build.sh        ← downloads en_core_web_sm
├── anonymisation/      ← flat copy of src/anonymisation/
├── examples/           ← flat copy of demo/examples/
└── sync.sh             ← refresh anonymisation/ + examples/ from src/
```

## Deploying — first time

1. Create a Space on HF: <https://huggingface.co/new-space>.
   Pick **Gradio** as the SDK, leave Python at 3.11, give it any name.
2. Clone the Space locally:

   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<space-name>
   cd <space-name>
   ```

3. From this repo, copy the entire `spaces/` contents into your Space:

   ```bash
   cp -r /path/to/data-anonymisation/spaces/* .
   ```

4. Commit and push:

   ```bash
   git add .
   git commit -m "Initial Space deployment"
   git push
   ```

   First build takes ~3–5 minutes (downloads spaCy, builds the image).
   After that the Space is live at:
   `https://huggingface.co/spaces/<your-username>/<space-name>`

5. To embed in your portfolio site:

   ```html
   <iframe
     src="https://huggingface.co/spaces/<you>/<space-name>"
     frameborder="0" width="100%" height="900"
     allow="clipboard-read; clipboard-write"></iframe>
   ```

## Refreshing the Space after code changes in the main repo

The `anonymisation/` and `examples/` folders here are *copies* of the main
project's source — keep them in sync:

```bash
./spaces/sync.sh
```

That copies `src/anonymisation/` → `spaces/anonymisation/` and
`demo/examples/` → `spaces/examples/`. Then commit + push the Space:

```bash
cd /path/to/<space-name>
cp -r /path/to/data-anonymisation/spaces/* .
git add . && git commit -m "Sync from main repo" && git push
```

## Sizing notes

- spaCy `en_core_web_sm` is 12 MB. With Gradio + dependencies the Space
  image is ~600 MB. Builds in ~3 minutes.
- TAB dataset (used by the mosaic scorer) is fetched from HF Datasets on
  first request (~50 MB, cached after).
- Cold-start latency from cold: ~10 seconds.

If you want a stronger model in the Space, the path is:
1. Push the RoBERTa fine-tune checkpoint to a private HF model repo.
2. Edit `app.py` to load it via `AutoModelForTokenClassification.from_pretrained("your-org/roberta-tab")`.
3. Switch the Space hardware to GPU (paid tier).

For a portfolio piece, the small-model deployment is usually enough — the
*pipeline architecture* is the showcased work, not raw model F1.

# Example documents

Documents the Studio (`./demo/run_studio.sh`) offers from its sample picker.
It reads every `*.txt` in this folder, so adding a file is enough.

**`sample_0_real_echr_judgment.txt` is real** — TAB test document 001-85728, a
published European Court of Human Rights judgment, already public. It is here
so the app opens on something the fine-tune was actually trained to read.

**Everything else is fabricated.** The people, organisations, addresses, case
numbers, NHS numbers, email addresses and telephone numbers in these files do
not refer to anyone. They were written to exercise the pipeline, and any
resemblance to a real matter is accidental. Do not treat any of it as a record
of anything.

The three `sample_long_*` files are the demonstration documents: several
hundred words each, and deliberately dense in all eight TAB entity types so
that every part of the pipeline has something to do.

| File | What it is for |
|---|---|
| `sample_long_employment_tribunal.txt` | The broadest one. Names with shorthand (`Dr Amara Okonjo-Bright` … `Okonjo-Bright`), an organisation and its acronym (`Fleet Decarbonisation Programme` … `FDP`), statutes, money, percentages, an email address and a phone number for the regex pass, and demographics that make the claimant identifiable in combination even after the name goes. |
| `sample_long_clinical_negligence.txt` | Medical register. Dates and times that carry the clinical argument, an NHS number, a date of birth, nationality and diagnoses — the case where a reader has a real reason to keep the dates and needs to see what that costs. |
| `sample_long_engagement_survey.txt` | Free-text survey comments and an analyst note. The mosaic argument in its clearest form: comment 5 names nobody and identifies one person anyway, through function, ethnicity, gender and a named network. This is the shape of the problem for any employee-survey product. |

The shorter `sample_1` … `sample_4` and the domain pairs (`*_b_*`, `*_c_*`) are
older, terser fixtures kept for quick checks.

## Local-only samples

Drop any `.txt` in here and the picker finds it on the next request — no
restart. A sample written for one audience, naming real people or a real
organisation, belongs in `.gitignore` rather than in a public repository, even
when its contents are invented. There is at least one such file locally that
is deliberately not tracked.

# Changelog

## v2.0 — 2026-11

### Added
- Bilingual interface (Russian / English) and Russian review analysis.
- CSV and XLSX import with automatic encoding and delimiter detection and manual column mapping; only the text column is required.
- Text sentiment independent of ratings: Positive, Negative, Neutral, Mixed; rating/text contradiction and possible irony flags.
- Aspect-based sentiment at clause level with industry templates; aspect editor (rename, merge, hide) with JSON settings export.
- Main findings with quotes and links to source reviews; "finding is wrong" feedback.
- Period comparison with a negative-share alert; comparison of up to four branches/companies/products.
- Optional semantic clustering with a local Sentence Transformers model.
- Export: summary report, aspect statistics, mentions, analyzed reviews, history.
- Dark theme with an orange accent, new home screen and unified chart style.
- 14 automated tests (core logic and Streamlit AppTest), evaluation scripts and recorded results.

### Changed
- Project restructured into the `reviewlens/` package, `scripts/` and `data/` subfolders; entry point renamed to `app.py`.
- No NLTK corpora downloads and no multiprocessing at startup.

## v1.0 — MVP
- Streamlit app for English reviews from a fixed CSV format: VADER sentiment, rating/text contradictions, TF-IDF + MiniBatchKMeans topics, bigrams, word cloud.

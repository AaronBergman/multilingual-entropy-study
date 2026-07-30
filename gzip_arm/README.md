# Do language rankings change between byte n-grams and gzip?

This repository contains a reproducible, exploratory study of whether languages
that look relatively easy or hard under a byte n-gram language model keep the
same ranking under gzip compression.

The short answer is **not necessarily**. The two measurements share substantial
signal, but gzip is a dictionary compressor rather than a probabilistic language
model, and some languages move markedly in the ranking. See
[`METHODS_RESULTS.md`](METHODS_RESULTS.md) for the complete interpretation after
running the study.

## What is measured

- **Byte n-gram:** held-out cross-entropy in bits per UTF-8 byte. Models of
  orders 1–5 are trained on the first aligned 70% of each language and evaluated
  on the remaining aligned 30%.
- **gzip:** compressed bits per original UTF-8 byte for the held-out portion,
  serialized as one concatenated newline-delimited stream. The fixed size of an
  empty gzip member is subtracted from the primary metric.

These numbers are comparable in units, but **gzip bits/byte is not a true
cross-entropy**. gzip uses DEFLATE (LZ77 plus Huffman coding), adapts within the
same stream it compresses, has a finite window, and includes format overhead.
It is best treated as a finite-sample compression proxy.

## Data

The experiment uses the official
[MicrosoftTranslator/NTREX](https://github.com/MicrosoftTranslator/NTREX)
corpus at commit `468c6b69c7f6a75d31d4743d9daba2af566cc18d`.
At that pinned revision, the `NTREX-128` directory contains 1,997
professionally translated, sentence-aligned news items in 127 translated
references plus the English source, making 128 measured language varieties.
The data is CC BY-SA 4.0. Raw files are downloaded locally and excluded from
this repository.

All text is normalized to Unicode NFC, encoded as UTF-8, and joined with one LF
after every sentence (including the final sentence). Bits per code point and
UTF-8 bytes per code point are retained as diagnostics, but bits per byte is the
primary normalization.

## Reproduce

Python scripts require no arguments; configuration constants are at the top of
each script.

```bash
uv sync --all-groups
uv run python download_data.py
uv run python run_study.py
uv run python make_figures.py
uv run pytest
```

The full analysis takes a few minutes and displays a language-level progress
bar. Machine-readable outputs are written to `results/`, and publication-ready
PNG/SVG figures to `figures/`.

## Repository map

```text
.
├── download_data.py              # pinned, validated NTREX acquisition
├── run_study.py                  # metrics, ranks, correlations, provenance
├── make_figures.py               # deterministic figures
├── src/gzip_arm/metrics.py       # tested gzip and n-gram implementations
├── tests/test_metrics.py
├── results/                      # committed CSV and JSON outputs
├── figures/                      # committed PNG and SVG outputs
└── METHODS_RESULTS.md            # concise methods/results memo
```

## Scope and interpretation

The study describes these translations in the NTREX news domain, not an
intrinsic entropy of languages or the cognitive complexity of their speakers.
Translation style, orthography, UTF-8 encoding, morphology, sentence length,
the n-gram smoothing rule, and gzip's implementation all affect the results.
Rank correlations therefore answer a narrow operational question: how similarly
do these two byte-based metrics order the same translated content?

## License and citation

Code in this repository is MIT licensed. NTREX data is not redistributed here;
it remains under CC BY-SA 4.0. If using the study, cite the NTREX paper:

> Federmann, Christian, Tom Kocmi, and Ying Xin. 2022. “NTREX-128—News Test
> References for MT Evaluation of 128 Languages.” Proceedings of the First
> Workshop on Scaling Up Multilingual Evaluation, 21–24.

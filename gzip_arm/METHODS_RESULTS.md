# Methods and results: byte n-grams versus gzip

## Executive answer

Yes, the cross-language ordering changes, although the broad ordering is highly
similar. Across 128 aligned NTREX language varieties, gzip compressed
bits-per-byte and held-out byte 4-gram cross-entropy have Spearman
`rho = 0.974` and Kendall `tau = 0.897`. Even with that high agreement, the
median language moves 3 rank positions, the 90th percentile moves 10.3
positions, and the largest shift is 65 positions.

The right conclusion is therefore not that gzip and n-grams measure unrelated
things. They capture a great deal of the same byte-level regularity, but they
are not interchangeable at the level of individual languages.

## Methods

The study uses the official
[NTREX-128 corpus](https://github.com/MicrosoftTranslator/NTREX) at commit
`468c6b69c7f6a75d31d4743d9daba2af566cc18d`. It contains 1,997 aligned news
sentences for each of 128 measured varieties, including the English source.
The first 1,397 aligned sentences form the n-gram training partition and the
remaining 600 form the common evaluation partition.

Text is normalized to Unicode NFC, serialized with one LF after every sentence,
and encoded as UTF-8. Byte n-grams of orders 1 through 5 use add-0.1 smoothing
over the full 256-byte alphabet. Their metric is held-out cross-entropy in bits
per byte.

The gzip metric is the size of one gzip-compressed evaluation stream, divided
by its original byte length. The measured 20-byte empty-member size is
subtracted. Compression uses Python `gzip.compress`, level 9, with `mtime=0`.
gzip learns repeated strings while compressing that same evaluation stream; it
does not use the n-gram training partition.

## Results

Rank agreement rises sharply beyond unigrams:

| Byte n-gram order | Spearman rho with gzip | Kendall tau with gzip |
|---:|---:|---:|
| 1 | 0.824 | 0.637 |
| 2 | 0.967 | 0.847 |
| 3 | 0.978 | 0.887 |
| 4 | 0.974 | 0.897 |
| 5 | 0.979 | 0.910 |

For the preregistered order-4 comparison, 88 of 128 varieties move at most five
rank positions and 115 move at most ten. Yoruba is the clear outlier: it ranks
123rd under the byte 4-gram but 58th under gzip, a 65-position shift. Somali
moves 25 positions in the same direction; French moves 17 positions in the
opposite direction.

The result is robust in the narrow sense that both metrics agree strongly
across most languages and across n-gram orders 2 through 5. It is also
substantively non-identical: high correlation does not prevent large
language-specific rank changes.

## Interpretation

gzip is not a probabilistic language model, so its compressed bits per byte are
not literally cross-entropy. DEFLATE combines a finite-window LZ77 dictionary
with Huffman coding, adapts on the evaluation stream, and pays format overhead.
The byte n-gram instead estimates held-out conditional probabilities from a
separate training partition.

Both primary metrics are also affected by UTF-8 representation. UTF-8 bytes per
code point correlate negatively with gzip bits per byte (`rho = -0.617`) and
byte 4-gram bits per byte (`rho = -0.619`). The secondary code-point
normalization remains highly correlated between methods (`rho = 0.976`) but
does not remove all representational effects.

These results characterize these translated news texts under two byte-based
coding schemes. They do not establish an intrinsic entropy or complexity
ranking of languages.

## Reproducibility

Machine-readable estimates are in `results/summary.json` and the accompanying
CSV files. The complete deterministic workflow is:

```bash
uv sync --all-groups
uv run python download_data.py
uv run python run_study.py
uv run python make_figures.py
uv run pytest
```

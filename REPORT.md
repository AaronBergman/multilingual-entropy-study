# Results: n-gram versus LLM cross-language entropy

## Executive answer

Yes. On this 13-language FLORES panel, byte n-grams and the local Qwen3 LLMs
share a meaningful cross-language signal, but they do not preserve the same
ordering.

The preregistered monolingual byte 5-gram and Qwen3 4B comparison has Spearman
`rho = 0.637` and Kendall `tau = 0.462`. The median language moves one rank
position and the largest movement is seven positions. English looks much
easier relative to the other languages under either LLM than under the 5-gram;
Yoruba moves in the opposite direction.

Increasing the LLM from 0.6B to 4B parameters has a different pattern: it
reduces bits per UTF-8 byte for all 13 languages, while mostly preserving their
ordering (`rho = 0.973`, `tau = 0.897`).

This experiment required local LLM inference, not a hosted API. Both quantized
models were scored locally with MLX.

## What was measured

The fixed protocol in [DESIGN.md](DESIGN.md) uses the professionally translated,
sentence-aligned FLORES-200 `dev` and `devtest` splits:

- 13 languages spanning nine scripts;
- 997 training and 1,012 evaluation sentences per language;
- monolingual and balanced multilingual byte n-grams, orders 1 through 5;
- `mlx-community/Qwen3-0.6B-4bit` and
  `mlx-community/Qwen3-4B-4bit`; and
- a primary ratio-of-sums metric in bits per UTF-8 byte.

Each sentence is evaluated independently. LLM losses include every text token
but exclude the prefix token. Aggregate estimates and paired model-size
differences use deterministic delete-one-sentence jackknife intervals.

## Main comparison

| Comparison | Spearman rho | Kendall tau | Median absolute rank change | Maximum |
|---|---:|---:|---:|---:|
| Monolingual byte 5-gram vs Qwen3 0.6B | 0.643 | 0.462 | 2 | 7 |
| Monolingual byte 5-gram vs Qwen3 4B | 0.637 | 0.462 | 1 | 7 |
| Balanced multilingual byte 5-gram vs Qwen3 4B | 0.632 | 0.462 | 1 | 7 |
| Qwen3 0.6B vs Qwen3 4B | 0.973 | 0.897 | 1 | 2 |

The two byte 5-gram training scopes agree almost perfectly with one another
(`rho = 0.989`), and both produce essentially the same conclusion against the
4B LLM. The n-gram/LLM mismatch is therefore not an artifact of choosing a
monolingual rather than a balanced shared n-gram.

The largest changes from the monolingual byte 5-gram to Qwen3 4B are:

| Language | 5-gram rank | Qwen3 4B rank | Movement |
|---|---:|---:|---:|
| English | 12 | 5 | 7 places easier under the LLM |
| Yoruba | 6 | 13 | 7 places harder under the LLM |
| Mandarin Chinese | 13 | 9 | 4 places easier under the LLM |
| Hindi | 2 | 4 | 2 places harder under the LLM |
| Thai | 3 | 1 | 2 places easier under the LLM |
| Swahili | 10 | 12 | 2 places harder under the LLM |

“Easier” and “harder” here mean only a lower or higher rank within this panel
under the named metric.

## Effect of LLM scale

Qwen3 4B has lower bits per byte than Qwen3 0.6B for all 13 languages, and all
13 paired 95% jackknife intervals exclude zero in the improvement direction.
The mean language improvement is 0.195 bits per byte. It ranges from 0.048 for
Mandarin Chinese to 0.702 for Swahili.

| Language | 4B improvement over 0.6B (bits/byte) | 95% interval |
|---|---:|---:|
| Swahili | 0.702 | [0.686, 0.717] |
| Turkish | 0.374 | [0.361, 0.386] |
| Tamil | 0.328 | [0.322, 0.334] |
| Yoruba | 0.279 | [0.266, 0.291] |
| Spanish | 0.151 | [0.143, 0.160] |
| Hindi | 0.139 | [0.132, 0.146] |
| Russian | 0.128 | [0.121, 0.135] |
| Japanese | 0.110 | [0.100, 0.121] |
| Korean | 0.097 | [0.085, 0.109] |
| English | 0.066 | [0.058, 0.073] |
| Modern Standard Arabic | 0.065 | [0.056, 0.074] |
| Thai | 0.053 | [0.048, 0.059] |
| Mandarin Chinese | 0.048 | [0.038, 0.057] |

The cross-language standard deviation falls from 0.845 bits per byte at 0.6B
to 0.721 at 4B, a roughly 15% reduction. The larger model therefore improves
every language and compresses the spread somewhat, but it does not equalize
performance: Yoruba and Swahili remain the two highest-loss languages.

## Sensitivity checks

### N-gram order

Agreement is not monotonic in n-gram order. For monolingual n-grams versus
Qwen3 4B, byte-normalized Spearman correlations for orders 1 through 5 are
`0.758, 0.560, 0.626, 0.610, 0.637`. The broad conclusion—positive but
imperfect agreement—does not depend on selecting order 5, but the exact
coefficient does.

### Normalization and tokenization

Changing the primary order-5 comparison from bits per UTF-8 byte to bits per
Unicode character reduces Spearman agreement with Qwen3 4B from `0.637` to
`0.544`. Neither unit is linguistically neutral: UTF-8 uses different numbers
of bytes for different scripts, while Unicode code points are not grapheme
clusters.

The two Qwen3 sizes share the same tokenizer. Tokenizer fertility has almost no
rank association with bits per byte in this panel (`rho = -0.082` for both
models), but it correlates strongly with bits per character (`rho = 0.720` at
0.6B and `0.676` at 4B). This is a concrete warning that the apparent
cross-language pattern depends on the denominator.

## Interpretation

The results support three narrow claims:

1. N-grams and LLMs capture some common cross-language predictability signal.
2. Moving from n-grams to this LLM family can materially reorder individual
   languages even when the overall rank correlation is positive.
3. Scaling within this LLM family improves every measured language and changes
   ranks much less than changing the model class.

The results do **not** rank the intrinsic complexity of natural languages.
Measured loss also reflects writing system, UTF-8 representation, FLORES
translation style, tokenizer allocation, unknown pretraining exposure,
post-training, model architecture, and 4-bit quantization. Absolute n-gram and
LLM loss levels should not be treated as a fair model contest because their
training data and objectives are radically different.

The sample contains only 13 languages and one LLM family. Conventional
correlation p-values are included in the machine-readable tables, but the
effect sizes and language-level movements are more informative than a binary
significance label.

## Reproducibility and audit

The analysis retains all 131,560 n-gram and 26,312 LLM sentence scores.
`results/summary.json`, `results/aggregate_metrics.csv`,
`results/metric_correlations.csv`, `results/ngram_order_sensitivity.csv`,
`results/rank_changes.csv`, and `results/model_size_deltas.csv` contain the
reported values.

The memory-bounded LLM scorer was checked against direct full-sequence scoring
on five short examples; the maximum absolute whole-sentence difference was
0.086 bits. The longest-sequence memory benchmark measured MLX peaks of
0.40 GiB for Qwen3 0.6B and 2.24 GiB for Qwen3 4B. The full scorer uses
eight-token logit blocks, an 8 GiB allocator limit, a 128 MiB cache limit, and
a 6 GiB measured-peak abort threshold.

The delegated gzip comparison is reported separately in
[`gzip_arm/METHODS_RESULTS.md`](gzip_arm/METHODS_RESULTS.md).

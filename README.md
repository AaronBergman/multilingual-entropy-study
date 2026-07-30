# Do language difficulty patterns change from n-grams to LLMs?

This repository contains an end-to-end, reproducible study of a deceptively
simple question:

> Do the measured cross-entropies of different languages change when we move
> from n-gram language models to modern LLMs—and does using a larger LLM change
> the answer?

The experiment scores professionally translated, sentence-aligned FLORES-200
text in 13 languages. It compares byte n-grams with two local Qwen3 models from
the same family and at the same 4-bit quantization level (0.6B and 4B).

## Why this is not just a perplexity table

Token-level perplexities from different tokenizers are not directly comparable.
A tokenizer can split one writing system into many more tokens than another,
making the language appear harder for partly mechanical reasons. The primary
unit here is therefore **bits per UTF-8 byte**; bits per Unicode character and
tokenizer fertility are reported as secondary diagnostics.

The central outcomes are:

1. cross-language rank agreement between n-grams and LLMs;
2. whether the larger LLM lowers loss for each language;
3. whether scaling changes language rankings or cross-language dispersion; and
4. whether conclusions depend on a language-specific versus balanced
   multilingual n-gram baseline.

See [DESIGN.md](DESIGN.md) for the fixed protocol and [REPORT.md](REPORT.md) for
results and interpretation.

## Findings

- The monolingual byte 5-gram and Qwen3 4B language rankings agree only
  moderately (`Spearman rho = 0.637`); the largest language movements are seven
  rank positions.
- Qwen3 4B improves bits per byte for all 13 languages relative to Qwen3 0.6B,
  with positive paired 95% intervals for every language.
- Scaling preserves rankings much more closely (`rho = 0.973`) than switching
  from n-grams to LLMs.
- Conclusions are directionally robust to n-gram order and training scope, but
  exact correlations change under byte versus character normalization.

## Companion gzip study

The delegated compression arm is in [`gzip_arm/`](gzip_arm/). It repeats the
cross-language ranking question for byte n-grams versus gzip on 128 aligned
NTREX language varieties. Its complete empirical memo is
[`gzip_arm/METHODS_RESULTS.md`](gzip_arm/METHODS_RESULTS.md).

## Reproduce

Requirements:

- Apple silicon macOS for the MLX scoring script;
- about 8 GB of free disk space;
- [`uv`](https://docs.astral.sh/uv/); and
- internet access for the first corpus/model download.

```bash
UV_CACHE_DIR=/tmp/multilingual-entropy-uv uv sync
UV_CACHE_DIR=/tmp/multilingual-entropy-uv uv run python run_all.py
```

Scripts intentionally require no command-line arguments. Study configuration is
at the top of `src/multilingual_entropy/config.py`.

To run individual stages:

```bash
UV_CACHE_DIR=/tmp/multilingual-entropy-uv uv run python scripts/01_prepare_data.py
UV_CACHE_DIR=/tmp/multilingual-entropy-uv uv run python scripts/02_score_ngrams.py
UV_CACHE_DIR=/tmp/multilingual-entropy-uv uv run python scripts/03_benchmark_memory.py
UV_CACHE_DIR=/tmp/multilingual-entropy-uv uv run python scripts/validate_mlx_scoring.py
UV_CACHE_DIR=/tmp/multilingual-entropy-uv uv run python scripts/03_score_llms.py
UV_CACHE_DIR=/tmp/multilingual-entropy-uv uv run python scripts/04_analyze.py
UV_CACHE_DIR=/tmp/multilingual-entropy-uv uv run python scripts/05_verify_artifacts.py
```

The LLM scorer is memory-bounded: it materializes logits for only eight tokens
at a time, caps the MLX allocator at 8 GiB, caps its free-buffer cache at
128 MiB, aborts above a measured 6 GiB MLX peak, and refuses to start until a
worst-case sequence benchmark passes for both models. It checkpoints after each
language. Models are cached under
`/Users/aaronbergman/hf_models/hub`; change `HF_HOME` in `config.py` on another
machine.

## Validate

```bash
UV_CACHE_DIR=/tmp/multilingual-entropy-uv uv run pytest
UV_CACHE_DIR=/tmp/multilingual-entropy-uv uv run ruff check .
```

The generated result artifacts include:

- sentence-level scores for every metric;
- aggregate language tables with deterministic jackknife intervals;
- rank correlations and model-size deltas;
- model and dataset revision/hash provenance;
- PNG and PDF figures; and
- a concise research report.

## Interpretation boundary

This is an exploratory case study, not a claim about intrinsic language
complexity. Measured loss combines properties of language, writing system,
UTF-8 representation, corpus translation, tokenizer, training-data exposure,
model scale, post-training, and 4-bit quantization. The matched content removes
one major confound, but it cannot remove the others.

## Data and licensing

FLORES-200 contains 3,001 professionally translated sentences drawn from 842
web articles; the public `dev` and `devtest` splits contain 997 and 1,012 aligned
sentences per language. The corpus is
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Downloaded
corpus files are not committed.

Study code is MIT-licensed. Qwen3 weights are separately licensed under Apache
2.0 and are not redistributed here.

## Sources

- NLLB Team. [No Language Left Behind: Scaling Human-Centered Machine
  Translation](https://arxiv.org/abs/2207.04672), 2022.
- Goyal et al. [The FLORES-101 Evaluation Benchmark for Low-Resource and
  Multilingual Machine Translation](https://arxiv.org/abs/2106.03193), 2021.
- Yang et al. [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388), 2025.
- [FLORES-200 repository and dataset documentation](https://github.com/facebookresearch/flores/tree/main/flores200).
- [MLX Community Qwen3 model collection](https://huggingface.co/collections/mlx-community/qwen3).
- Federmann, Kocmi, and Xin.
  [NTREX-128: News Test References for MT Evaluation of 128 Languages](https://aclanthology.org/2022.sumeval-1.4/),
  2022.

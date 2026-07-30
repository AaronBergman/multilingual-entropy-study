# AGENTS.md

## Project purpose

This repository is an empirical study of whether cross-language entropy
patterns change between byte n-grams and multilingual LLMs, and whether an LLM
size increase changes those patterns. The `gzip_arm/` companion performs the
delegated byte n-gram versus gzip comparison on NTREX-128 and has its own
reproduction instructions and agent notes.

## Fixed conventions

- Use `uv`; set `UV_CACHE_DIR=/tmp/multilingual-entropy-uv` in this environment
  because the default global uv cache may be sandbox-inaccessible.
- Scripts require no CLI arguments. Change study constants in
  `src/multilingual_entropy/config.py`.
- Do not introduce random seeds or random-state settings. Current uncertainty
  uses deterministic delete-one-sentence jackknife estimates.
- Primary metric is bits per UTF-8 byte. Token-level perplexity must not be used
  for cross-language comparisons.
- Preserve Unicode NFC normalization and validate the 997/1,012 FLORES split
  counts for every language.
- Keep sentence-level result files and provenance manifests; aggregates alone
  are not sufficient evidence.
- Do not commit downloaded FLORES data or model weights.

## Commands

```bash
UV_CACHE_DIR=/tmp/multilingual-entropy-uv uv sync
UV_CACHE_DIR=/tmp/multilingual-entropy-uv uv run python run_all.py
UV_CACHE_DIR=/tmp/multilingual-entropy-uv uv run pytest
UV_CACHE_DIR=/tmp/multilingual-entropy-uv uv run ruff check .
```

## Pipeline

1. `scripts/01_prepare_data.py`: download, validate, NFC-normalize, subset.
2. `scripts/02_score_ngrams.py`: train and score both n-gram scopes.
3. `scripts/03_benchmark_memory.py`: probe the longest sequence under a 6 GiB
   MLX peak ceiling for every model.
4. `scripts/validate_mlx_scoring.py`: compare streaming and direct likelihoods
   on short sequences.
5. `scripts/03_score_llms.py`: stream tiny fixed logit blocks with per-language
   checkpoints and score devtest.
6. `scripts/04_analyze.py`: aggregate, jackknife, correlate, and plot.
7. `scripts/05_verify_artifacts.py`: fail unless row coverage, summaries,
   provenance, memory checks, and figures are complete.

The LLM scorer saves after each completed language and can resume completed
model/language/sentence keys. Do not increase `STREAM_BLOCK_TOKENS`, allocator
limits, or the 6 GiB safety ceiling without a new explicit memory benchmark.

# Agent notes

This subproject is the gzip-compression arm of the multilingual entropy study.

- Run everything through `uv`; no script takes required command-line arguments.
- Configuration constants live near the top of each executable script.
- `data/raw/` is intentionally ignored. `download_data.py` retrieves the pinned
  official NTREX-128 revision.
- Committed outputs belong in `results/` and `figures/`.
- Text is normalized to Unicode NFC and encoded as UTF-8. Do not change this
  silently because it changes both gzip and byte n-gram measurements.
- The main gzip metric is header-corrected compressed bits per original byte on
  one concatenated held-out stream. It is a compression proxy, not a true
  probabilistic cross-entropy.
- Do not introduce random sampling or random seeds. The sentence split and all
  analyses are deterministic.

"""Probe worst-case MLX scoring memory before permitting a full corpus run."""

from __future__ import annotations

import gc
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import pandas as pd
from huggingface_hub import snapshot_download
from mlx_lm import load

from multilingual_entropy.config import (
    CORPUS_PATH,
    HF_CACHE_DIR,
    HF_HOME,
    LLM_MODELS,
    RESULTS_DIR,
)
from multilingual_entropy.mlx_scoring import (
    configure_memory_limits,
    memory_snapshot,
    score_tokens_streaming,
)

STREAM_BLOCK_TOKENS = 8
MEMORY_LIMIT_GIB = 8
CACHE_LIMIT_MIB = 128
ABORT_PEAK_GIB = 6.0
OUTPUT_PATH = RESULTS_DIR / "memory_benchmark.json"


def prefix_token(tokenizer: object) -> int:
    token_id = tokenizer.bos_token_id
    if token_id is None:
        token_id = tokenizer.eos_token_id
    if token_id is None:
        raise ValueError("Tokenizer has neither BOS nor EOS token")
    return int(token_id)


def main() -> None:
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR)
    configure_memory_limits(MEMORY_LIMIT_GIB, CACHE_LIMIT_MIB)
    corpus = pd.read_parquet(CORPUS_PATH)
    texts = corpus.loc[corpus["split"] == "devtest", "text"].tolist()
    records: list[dict[str, object]] = []

    for model_id in LLM_MODELS:
        snapshot_path = Path(
            snapshot_download(repo_id=model_id, cache_dir=HF_CACHE_DIR)
        ).resolve()
        model, tokenizer = load(str(snapshot_path))
        tokenized = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
        longest = max(tokenized, key=len)
        mx.clear_cache()
        mx.reset_peak_memory()
        bits, peak = score_tokens_streaming(
            model,
            longest,
            prefix_token(tokenizer),
            block_tokens=STREAM_BLOCK_TOKENS,
            abort_peak_gib=ABORT_PEAK_GIB,
        )
        final_memory = memory_snapshot()
        records.append(
            {
                "model": model_id,
                "snapshot_revision": snapshot_path.name,
                "probe_tokens": len(longest),
                "probe_bits": bits,
                "peak": peak.to_dict(),
                "final": final_memory.to_dict(),
                "passed": peak.peak_gib <= ABORT_PEAK_GIB,
            }
        )
        del model, tokenizer, tokenized, longest
        gc.collect()
        mx.clear_cache()

    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "stream_block_tokens": STREAM_BLOCK_TOKENS,
        "memory_limit_gib": MEMORY_LIMIT_GIB,
        "cache_limit_mib": CACHE_LIMIT_MIB,
        "abort_peak_gib": ABORT_PEAK_GIB,
        "models": records,
        "all_passed": all(record["passed"] for record in records),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not payload["all_passed"]:
        raise MemoryError(f"Memory benchmark failed; see {OUTPUT_PATH}")
    print(f"Memory benchmark passed; wrote {OUTPUT_PATH}")
    for record in records:
        print(
            f"{record['model']}: {record['peak']['peak_gib']:.2f} GiB MLX peak "
            f"on {record['probe_tokens']} tokens"
        )


if __name__ == "__main__":
    main()

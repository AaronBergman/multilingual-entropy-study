"""Validate streaming scores against direct full-sequence scores."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pandas as pd
from huggingface_hub import snapshot_download
from mlx_lm import load

from multilingual_entropy.config import (
    CORPUS_PATH,
    HF_CACHE_DIR,
    LLM_MODELS,
    RESULTS_DIR,
)
from multilingual_entropy.mlx_scoring import (
    configure_memory_limits,
    extract_logits,
    score_tokens_streaming,
)

MODEL_ID = LLM_MODELS[0]
STREAM_BLOCK_TOKENS = 8
MEMORY_LIMIT_GIB = 8
CACHE_LIMIT_MIB = 128
ABORT_PEAK_GIB = 6.0
ABSOLUTE_TOLERANCE_BITS = 0.1
N_EXAMPLES = 5
OUTPUT_PATH = RESULTS_DIR / "mlx_scoring_validation.json"
LOG2 = math.log(2.0)


def full_sequence_score(model: object, tokens: list[int], prefix_token_id: int) -> float:
    inputs = mx.array([prefix_token_id, *tokens[:-1]], dtype=mx.int32)[None, :]
    targets = mx.array(tokens, dtype=mx.int32)[None, :]
    logits = extract_logits(model(inputs)).astype(mx.float32)
    bits = (mx.sum(nn.losses.cross_entropy(logits, targets, reduction="none")) / LOG2).astype(
        mx.float32
    )
    mx.eval(bits)
    result = float(bits.item())
    del inputs, targets, logits, bits
    mx.clear_cache()
    return result


def main() -> None:
    configure_memory_limits(MEMORY_LIMIT_GIB, CACHE_LIMIT_MIB)
    snapshot_path = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            cache_dir=HF_CACHE_DIR,
            local_files_only=True,
        )
    ).resolve()
    model, tokenizer = load(str(snapshot_path))
    prefix_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    if prefix_token_id is None:
        raise ValueError("Tokenizer has neither BOS nor EOS")

    corpus = pd.read_parquet(CORPUS_PATH)
    candidates = corpus[corpus["split"] == "devtest"].copy()
    candidates["tokens"] = candidates["text"].map(
        lambda text: tokenizer.encode(text, add_special_tokens=False)
    )
    examples = candidates.sort_values("tokens", key=lambda values: values.map(len)).head(
        N_EXAMPLES
    )

    records: list[dict[str, object]] = []
    for row in examples.itertuples(index=False):
        direct = full_sequence_score(model, row.tokens, int(prefix_token_id))
        streaming, peak = score_tokens_streaming(
            model,
            row.tokens,
            int(prefix_token_id),
            block_tokens=STREAM_BLOCK_TOKENS,
            abort_peak_gib=ABORT_PEAK_GIB,
        )
        difference = streaming - direct
        records.append(
            {
                "language": row.language,
                "sentence_id": row.sentence_id,
                "tokens": len(row.tokens),
                "direct_bits": direct,
                "streaming_bits": streaming,
                "difference_bits": difference,
                "absolute_difference_bits": abs(difference),
                "peak_gib": peak.peak_gib,
                "passed": abs(difference) <= ABSOLUTE_TOLERANCE_BITS,
            }
        )

    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": MODEL_ID,
        "snapshot_revision": snapshot_path.name,
        "absolute_tolerance_bits": ABSOLUTE_TOLERANCE_BITS,
        "examples": records,
        "all_passed": all(record["passed"] for record in records),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not payload["all_passed"]:
        raise AssertionError(f"Streaming validation failed; see {OUTPUT_PATH}")
    print(f"Streaming validation passed on {len(records)} examples; wrote {OUTPUT_PATH}")
    print(
        "Maximum absolute discrepancy: "
        f"{max(record['absolute_difference_bits'] for record in records):.6f} bits"
    )


if __name__ == "__main__":
    main()

"""Score FLORES devtest under local MLX causal language models."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import pandas as pd
from huggingface_hub import snapshot_download
from mlx_lm import load
from tqdm import tqdm

from multilingual_entropy.config import (
    CORPUS_PATH,
    HF_CACHE_DIR,
    HF_HOME,
    LLM_MODELS,
    MAX_SENTENCE_TOKENS,
    REPO_ROOT,
    RESULTS_DIR,
)
from multilingual_entropy.mlx_scoring import (
    configure_memory_limits,
    memory_snapshot,
    score_tokens_streaming,
)

OUTPUT_PATH = RESULTS_DIR / "llm_sentence_scores.parquet"
PROVENANCE_PATH = RESULTS_DIR / "llm_provenance.json"
MEMORY_BENCHMARK_PATH = RESULTS_DIR / "memory_benchmark.json"
STREAM_BLOCK_TOKENS = 8
MEMORY_LIMIT_GIB = 8
CACHE_LIMIT_MIB = 128
ABORT_PEAK_GIB = 6.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def snapshot_metadata(model_id: str, snapshot_path: Path) -> dict[str, object]:
    metadata: dict[str, object] = {
        "model_id": model_id,
        "snapshot_revision": snapshot_path.name,
    }
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        path = snapshot_path / filename
        if path.exists():
            metadata[f"{filename}_sha256"] = sha256(path)
    return metadata


def score_model(
    model_id: str,
    test: pd.DataFrame,
    existing: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    snapshot_path = Path(
        snapshot_download(repo_id=model_id, cache_dir=HF_CACHE_DIR)
    ).resolve()
    model, tokenizer = load(str(snapshot_path))
    prefix_token_id = tokenizer.bos_token_id
    if prefix_token_id is None:
        prefix_token_id = tokenizer.eos_token_id
    if prefix_token_id is None:
        raise ValueError(f"{model_id} tokenizer has neither BOS nor EOS token")

    roundtrip_failures = sum(
        tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)) != text
        for text in test["text"]
    )
    done = set()
    if not existing.empty:
        model_existing = existing[existing["model"] == model_id]
        done = set(zip(model_existing["language"], model_existing["sentence_id"], strict=False))

    total_rows_written = 0
    language_memory: list[dict[str, object]] = []
    for language, language_test in test.groupby("language", sort=True):
        completed_sentence_ids = {
            sentence_id for done_language, sentence_id in done if done_language == language
        }
        pending = language_test[
            ~language_test["sentence_id"].isin(completed_sentence_ids)
        ]
        tokenized: list[tuple[object, list[int]]] = []
        for record in pending.itertuples(index=False):
            tokens = tokenizer.encode(record.text, add_special_tokens=False)
            if len(tokens) > MAX_SENTENCE_TOKENS:
                raise ValueError(
                    f"{model_id}: {record.language}/{record.sentence_id} has "
                    f"{len(tokens)} tokens, above {MAX_SENTENCE_TOKENS}"
                )
            tokenized.append((record, tokens))
        tokenized.sort(key=lambda item: len(item[1]))

        rows: list[dict[str, object]] = []
        if tokenized:
            mx.reset_peak_memory()
            for record, tokens in tqdm(
                tokenized,
                desc=f"{model_id} / {language}",
                mininterval=30.0,
                maxinterval=60.0,
                miniters=100,
            ):
                sentence_bits, _ = score_tokens_streaming(
                    model,
                    tokens,
                    int(prefix_token_id),
                    block_tokens=STREAM_BLOCK_TOKENS,
                    abort_peak_gib=ABORT_PEAK_GIB,
                )
                rows.append(
                    {
                        "metric_type": "llm",
                        "metric": model_slug(model_id),
                        "model": model_id,
                        "language": record.language,
                        "language_name": record.language_name,
                        "sentence_id": record.sentence_id,
                        "bits": sentence_bits,
                        "bytes": record.bytes,
                        "characters": record.characters,
                        "tokens": len(tokens),
                        "bits_per_byte": sentence_bits / record.bytes,
                        "bits_per_character": sentence_bits / record.characters,
                        "tokens_per_byte": len(tokens) / record.bytes,
                        "tokens_per_character": len(tokens) / record.characters,
                    }
                )
        if rows:
            new = pd.DataFrame(rows)
            existing = pd.concat([existing, new], ignore_index=True)
            existing = existing.drop_duplicates(
                subset=["model", "language", "sentence_id"], keep="last"
            )
            existing.to_parquet(OUTPUT_PATH, index=False)
            total_rows_written += len(rows)
            current_memory = memory_snapshot()
            language_memory.append(
                {
                    "language": language,
                    "status": "scored",
                    "rows_scored": len(rows),
                    **current_memory.to_dict(),
                }
            )
        else:
            language_memory.append(
                {
                    "language": language,
                    "status": "checkpoint_reused",
                    "rows_scored": 0,
                }
            )
        mx.clear_cache()

    model_total_rows = int((existing["model"] == model_id).sum())
    metadata = snapshot_metadata(model_id, snapshot_path)
    metadata.update(
        {
            "prefix_token_id": int(prefix_token_id),
            "roundtrip_failures": roundtrip_failures,
            "rows_scored_this_invocation": total_rows_written,
            "rows_total_in_output": model_total_rows,
            "stream_block_tokens": STREAM_BLOCK_TOKENS,
            "memory_limit_gib": MEMORY_LIMIT_GIB,
            "cache_limit_mib": CACHE_LIMIT_MIB,
            "abort_peak_gib": ABORT_PEAK_GIB,
            "language_memory": language_memory,
        }
    )
    del model
    del tokenizer
    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    return existing, metadata


def main() -> None:
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR)
    if not MEMORY_BENCHMARK_PATH.exists():
        raise FileNotFoundError(
            "Run scripts/03_benchmark_memory.py and inspect its passing result "
            "before full LLM scoring."
        )
    benchmark = json.loads(MEMORY_BENCHMARK_PATH.read_text(encoding="utf-8"))
    if not benchmark.get("all_passed"):
        raise MemoryError("The required memory benchmark did not pass")
    configure_memory_limits(MEMORY_LIMIT_GIB, CACHE_LIMIT_MIB)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_parquet(CORPUS_PATH)
    test = corpus[corpus["split"] == "devtest"].copy()
    existing = pd.read_parquet(OUTPUT_PATH) if OUTPUT_PATH.exists() else pd.DataFrame()
    provenance: list[dict[str, object]] = []

    for model_id in LLM_MODELS:
        existing, metadata = score_model(model_id, test, existing)
        provenance.append(metadata)

    expected_rows = len(test) * len(LLM_MODELS)
    if len(existing) != expected_rows:
        raise ValueError(f"Expected {expected_rows:,} rows, found {len(existing):,}")
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "models": provenance,
        "evaluation_split": "devtest",
        "context_policy": "each sentence independently prefixed by tokenizer BOS, or EOS if no BOS",
        "special_tokens_scored": False,
        "quantization_note": "Both primary models use MLX 4-bit quantization.",
        "memory_benchmark": str(MEMORY_BENCHMARK_PATH.relative_to(REPO_ROOT)),
        "rows": len(existing),
    }
    PROVENANCE_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(existing):,} sentence scores to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

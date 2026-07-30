"""Train byte n-grams on FLORES dev and score FLORES devtest."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd
from tqdm import tqdm

from multilingual_entropy.config import (
    CORPUS_PATH,
    LANGUAGE_BY_CODE,
    LANGUAGES,
    NGRAM_ALPHA,
    NGRAM_ORDERS,
    RESULTS_DIR,
)
from multilingual_entropy.ngram import ByteNGram

OUTPUT_PATH = RESULTS_DIR / "ngram_sentence_scores.parquet"
PROVENANCE_PATH = RESULTS_DIR / "ngram_provenance.json"


def score_scope(
    train: pd.DataFrame,
    test: pd.DataFrame,
    training_scope: str,
    order: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if training_scope == "monolingual":
        models = {
            language.code: ByteNGram(order=order, alpha=NGRAM_ALPHA).fit(
                train.loc[train["language"] == language.code, "text"]
            )
            for language in LANGUAGES
        }
    elif training_scope == "multilingual_balanced":
        pooled = ByteNGram(order=order, alpha=NGRAM_ALPHA).fit(train["text"])
        models = {language.code: pooled for language in LANGUAGES}
    else:
        raise ValueError(f"Unknown scope: {training_scope}")

    description = f"{training_scope} byte {order}-gram"
    for record in tqdm(
        test.itertuples(index=False),
        total=len(test),
        desc=description,
        leave=False,
    ):
        bits = models[record.language].score_bits(record.text)
        rows.append(
            {
                "metric_type": "ngram",
                "metric": f"ngram_{training_scope}_o{order}_a{NGRAM_ALPHA:g}",
                "training_scope": training_scope,
                "order": order,
                "alpha": NGRAM_ALPHA,
                "language": record.language,
                "language_name": LANGUAGE_BY_CODE[record.language].name,
                "sentence_id": record.sentence_id,
                "bits": bits,
                "bytes": record.bytes,
                "characters": record.characters,
                "bits_per_byte": bits / record.bytes,
                "bits_per_character": bits / record.characters,
            }
        )
    return rows


def main() -> None:
    corpus = pd.read_parquet(CORPUS_PATH)
    train = corpus[corpus["split"] == "dev"].copy()
    test = corpus[corpus["split"] == "devtest"].copy()
    all_rows: list[dict[str, object]] = []
    for order in NGRAM_ORDERS:
        for scope in ("monolingual", "multilingual_balanced"):
            all_rows.extend(score_scope(train, test, scope, order))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(all_rows)
    scores.to_parquet(OUTPUT_PATH, index=False)
    provenance = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "orders": list(NGRAM_ORDERS),
        "alpha": NGRAM_ALPHA,
        "training_split": "dev",
        "evaluation_split": "devtest",
        "unit": "UTF-8 bytes",
        "bos_policy": "order-1 dedicated BOS symbols per sentence; BOS not scored",
        "eos_policy": "no EOS symbol is scored",
        "rows": len(scores),
    }
    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(scores):,} sentence scores to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

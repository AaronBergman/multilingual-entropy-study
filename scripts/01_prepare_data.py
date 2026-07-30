"""Download, validate, normalize, and subset FLORES-200."""

from __future__ import annotations

import hashlib
import json
import tarfile
import unicodedata
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from multilingual_entropy.config import (
    CORPUS_MANIFEST,
    CORPUS_PATH,
    FLORES_ARCHIVE,
    FLORES_EXTRACTED,
    FLORES_URL,
    LANGUAGES,
    PROCESSED_DIR,
    RAW_DIR,
)

EXPECTED_DEV = 997
EXPECTED_DEVTEST = 1_012
UNICODE_NORMALIZATION = "NFC"


class DownloadProgress(tqdm):
    def update_to(
        self, blocks: int = 1, block_size: int = 1, total_size: int | None = None
    ) -> None:
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if FLORES_ARCHIVE.exists():
        return
    with DownloadProgress(
        unit="B", unit_scale=True, miniters=1, desc="Downloading FLORES-200"
    ) as progress:
        urllib.request.urlretrieve(
            FLORES_URL, filename=FLORES_ARCHIVE, reporthook=progress.update_to
        )


def extract() -> None:
    if FLORES_EXTRACTED.exists():
        return
    with tarfile.open(FLORES_ARCHIVE) as archive:
        archive.extractall(RAW_DIR, filter="data")


def locate_split(language_code: str, split: str) -> Path:
    candidates = [
        FLORES_EXTRACTED / split / f"{language_code}.{split}",
        FLORES_EXTRACTED / "flores200_dataset" / split / f"{language_code}.{split}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(FLORES_EXTRACTED.rglob(f"{language_code}.{split}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {language_code}.{split} file, found {len(matches)}"
        )
    return matches[0]


def load_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    expected = {"dev": EXPECTED_DEV, "devtest": EXPECTED_DEVTEST}
    for language in tqdm(LANGUAGES, desc="Reading languages"):
        for split, expected_count in expected.items():
            path = locate_split(language.code, split)
            raw_sentences = path.read_text(encoding="utf-8").splitlines()
            if len(raw_sentences) != expected_count:
                raise ValueError(
                    f"{language.code}/{split}: expected {expected_count}, got {len(raw_sentences)}"
                )
            for sentence_id, raw_sentence in enumerate(raw_sentences):
                sentence = unicodedata.normalize(UNICODE_NORMALIZATION, raw_sentence)
                if not sentence:
                    raise ValueError(f"Empty sentence: {language.code}/{split}/{sentence_id}")
                rows.append(
                    {
                        "split": split,
                        "sentence_id": sentence_id,
                        "language": language.code,
                        "language_name": language.name,
                        "script": language.script,
                        "family": language.family,
                        "text": sentence,
                        "bytes": len(sentence.encode("utf-8")),
                        "characters": len(sentence),
                        "normalization_changed": sentence != raw_sentence,
                    }
                )
    frame = pd.DataFrame(rows)
    for split, expected_count in expected.items():
        split_frame = frame[frame["split"] == split]
        counts = split_frame.groupby("language")["sentence_id"].nunique()
        if not (counts == expected_count).all():
            raise ValueError(f"Alignment failure in {split}: {counts.to_dict()}")
    return frame


def main() -> None:
    download()
    extract()
    frame = load_rows()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(CORPUS_PATH, index=False)
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_url": FLORES_URL,
        "archive_sha256": sha256(FLORES_ARCHIVE),
        "unicode_normalization": UNICODE_NORMALIZATION,
        "languages": [language.__dict__ for language in LANGUAGES],
        "splits": frame.groupby("split").size().to_dict(),
        "rows": len(frame),
        "normalization_changes": int(frame["normalization_changed"].sum()),
        "corpus_sha256": sha256(CORPUS_PATH),
    }
    CORPUS_MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(frame):,} aligned rows to {CORPUS_PATH}")
    print(f"Archive SHA-256: {manifest['archive_sha256']}")


if __name__ == "__main__":
    main()

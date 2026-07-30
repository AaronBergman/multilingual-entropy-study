from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

# Configuration: edit these constants if intentionally changing the data source.
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
DATASET_DIR = RAW_ROOT / "NTREX"
SOURCE_COMMIT = "468c6b69c7f6a75d31d4743d9daba2af566cc18d"
SOURCE_ARCHIVE_URL = (
    "https://github.com/MicrosoftTranslator/NTREX/archive/"
    f"{SOURCE_COMMIT}.zip"
)
# The pinned NTREX-128 directory contains 127 translated references plus the
# aligned English source, for 128 measured language varieties in total.
EXPECTED_LANGUAGE_FILES = 127
EXPECTED_SENTENCES = 1997


class DownloadProgress(tqdm):
    def update_to(
        self, blocks: int = 1, block_size: int = 1, total_size: int | None = None
    ) -> None:
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(dataset_dir: Path) -> dict[str, int]:
    corpus_dir = dataset_dir / "NTREX-128"
    source_file = corpus_dir / "newstest2019-src.eng.txt"
    reference_files = sorted(corpus_dir.glob("newstest2019-ref.*.txt"))
    if not source_file.exists():
        raise RuntimeError(f"Missing source file: {source_file}")
    if len(reference_files) != EXPECTED_LANGUAGE_FILES:
        raise RuntimeError(
            f"Expected {EXPECTED_LANGUAGE_FILES} references, "
            f"found {len(reference_files)}"
        )

    files = [source_file, *reference_files]
    line_counts = {
        path.name: len(path.read_text(encoding="utf-8").splitlines())
        for path in files
    }
    bad = {
        name: count
        for name, count in line_counts.items()
        if count != EXPECTED_SENTENCES
    }
    if bad:
        raise RuntimeError(f"Unexpected aligned line counts: {bad}")
    return {
        "reference_language_files": len(reference_files),
        "sentences_per_language": EXPECTED_SENTENCES,
    }


def main() -> None:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    if DATASET_DIR.exists():
        validation = validate_dataset(DATASET_DIR)
        print(f"Validated existing dataset at {DATASET_DIR}")
        archive_sha256 = None
    else:
        with tempfile.TemporaryDirectory(dir=RAW_ROOT) as temp_name:
            temp_dir = Path(temp_name)
            archive_path = temp_dir / "ntrex.zip"
            with DownloadProgress(
                unit="B", unit_scale=True, miniters=1, desc="Downloading NTREX"
            ) as progress:
                urllib.request.urlretrieve(
                    SOURCE_ARCHIVE_URL,
                    archive_path,
                    reporthook=progress.update_to,
                )
            archive_sha256 = sha256_file(archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(temp_dir)
            extracted = temp_dir / f"NTREX-{SOURCE_COMMIT}"
            validation = validate_dataset(extracted)
            shutil.move(str(extracted), str(DATASET_DIR))
            print(f"Installed dataset at {DATASET_DIR}")

    manifest = {
        "dataset": "NTREX-128",
        "source_repository": "https://github.com/MicrosoftTranslator/NTREX",
        "source_commit": SOURCE_COMMIT,
        "source_archive_url": SOURCE_ARCHIVE_URL,
        "downloaded_archive_sha256": archive_sha256,
        "license": "CC BY-SA 4.0",
        **validation,
    }
    manifest_path = RAW_ROOT / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()

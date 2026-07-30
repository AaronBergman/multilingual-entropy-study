"""Run the complete reproducible pipeline without command-line arguments."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SCRIPTS = (
    "scripts/01_prepare_data.py",
    "scripts/02_score_ngrams.py",
    "scripts/03_benchmark_memory.py",
    "scripts/validate_mlx_scoring.py",
    "scripts/03_score_llms.py",
    "scripts/04_analyze.py",
    "scripts/05_verify_artifacts.py",
)


def main() -> None:
    for script in SCRIPTS:
        print(f"\n=== Running {script} ===", flush=True)
        subprocess.run([sys.executable, str(REPO_ROOT / script)], cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()

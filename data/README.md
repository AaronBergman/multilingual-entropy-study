# Data

Run `uv run python scripts/01_prepare_data.py` to download the original public
FLORES-200 archive, select the preregistered 13-language panel, normalize text
to NFC, validate alignment, and create `data/processed/flores_panel.parquet`.

Downloaded and processed corpus files are intentionally excluded from Git.
The script records source and output SHA-256 hashes in
`data/processed/manifest.json`.

FLORES-200 is licensed separately under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The code in
this repository is MIT-licensed.

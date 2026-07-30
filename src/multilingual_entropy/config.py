"""Study-wide constants fixed before outcome inspection."""

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
FLORES_ARCHIVE = RAW_DIR / "flores200_dataset.tar.gz"
FLORES_EXTRACTED = RAW_DIR / "flores200_dataset"
CORPUS_PATH = PROCESSED_DIR / "flores_panel.parquet"
CORPUS_MANIFEST = PROCESSED_DIR / "manifest.json"

HF_HOME = Path("/Users/aaronbergman/hf_models")
HF_CACHE_DIR = HF_HOME / "hub"

NGRAM_ORDERS = (1, 2, 3, 4, 5)
NGRAM_ALPHA = 0.1
PRIMARY_NGRAM_ORDER = 5

LLM_MODELS = (
    "mlx-community/Qwen3-0.6B-4bit",
    "mlx-community/Qwen3-4B-4bit",
)
MAX_SENTENCE_TOKENS = 1_024


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    script: str
    family: str


# A compact panel chosen before inspecting outcomes. It spans nine scripts and
# typologically diverse language families while keeping the full LLM run tractable.
LANGUAGES = (
    Language("eng_Latn", "English", "Latin", "Germanic"),
    Language("spa_Latn", "Spanish", "Latin", "Romance"),
    Language("rus_Cyrl", "Russian", "Cyrillic", "Slavic"),
    Language("arb_Arab", "Modern Standard Arabic", "Arabic", "Semitic"),
    Language("hin_Deva", "Hindi", "Devanagari", "Indo-Aryan"),
    Language("zho_Hans", "Mandarin Chinese", "Han (simplified)", "Sinitic"),
    Language("jpn_Jpan", "Japanese", "Japanese mixed", "Japonic"),
    Language("kor_Hang", "Korean", "Hangul", "Koreanic"),
    Language("tur_Latn", "Turkish", "Latin", "Turkic"),
    Language("swh_Latn", "Swahili", "Latin", "Bantu"),
    Language("yor_Latn", "Yoruba", "Latin", "Volta-Niger"),
    Language("tam_Taml", "Tamil", "Tamil", "Dravidian"),
    Language("tha_Thai", "Thai", "Thai", "Kra-Dai"),
)

LANGUAGE_BY_CODE = {language.code: language for language in LANGUAGES}

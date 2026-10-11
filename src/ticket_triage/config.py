"""
Configuration management module.
Loads environment variables and exposes paths and default system parameters.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


def _project_path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    path = Path(value).expanduser() if value else default

    if not path.is_absolute():
        path = BASE_DIR / path

    return path.resolve()


DATA_DIR = BASE_DIR / "data"
HISTORICAL_TICKETS_PATH = DATA_DIR / "historical_tickets.json"
DAY1_TICKETS_PATH = DATA_DIR / "day1_tickets.json"
HELDOUT_TICKETS_PATH = DATA_DIR / "heldout_tickets.json"
HELDOUT_GOLD_LABELS_PATH = DATA_DIR / "heldout_gold_labels.json"

VECTOR_STORE_PATH = _project_path_from_env(
    "VECTOR_STORE_PATH",
    BASE_DIR / "storage" / "vector_store",
)

OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"
LOG_FILE_PATH = _project_path_from_env(
    "LOG_FILE_PATH",
    LOGS_DIR / "validation_failures.log",
)

for directory in (OUTPUTS_DIR, LOGS_DIR, VECTOR_STORE_PATH, LOG_FILE_PATH.parent):
    directory.mkdir(parents=True, exist_ok=True)


# Provider credentials. Empty values are ignored.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3.8-flash").strip()

GROQ_API_KEYS = tuple(
    key
    for key in (
        os.getenv("GROQ_API_KEY", "").strip(),
        os.getenv("GROQ_API_KEY_3", "").strip(),
        os.getenv("GROQ_API_KEY_4", "").strip(),
    )
    if key
)
GROQ_MODEL_NAME = os.getenv(
    "GROQ_MODEL_NAME",
    "openai/gpt-oss-20b",
).strip()


# RAG and decision settings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K_SIMILAR = 3

# Starting threshold; it is not statistically calibrated.
CONFIDENCE_ABSTENTION_THRESHOLD = 0.65

# One repair attempt after an invalid response, then safe escalation.
MAX_SCHEMA_RETRY_ATTEMPTS = 1

# Applied to every API call per provider, including repair calls.
PROVIDER_MIN_INTERVAL_SECONDS = 15.0

# If a provider gives no retry delay with its rate-limit error, temporarily
# skip it and route the request to another configured provider.
RATE_LIMIT_COOLDOWN_SECONDS = 60.0
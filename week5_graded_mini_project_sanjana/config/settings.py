"""Application settings for the MP1 prompt comparison project."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE_PATH = BASE_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    DEFAULT_MODEL: str = "gpt-4o-mini"
    JUDGE_MODEL: str = "gpt-4o"
    TEMPERATURE: float = 0.0
    MAX_TOKENS: int = 500

    DATA_DIR: Path = BASE_DIR / "data"
    JOB_SNIPPETS_PATH: Path = DATA_DIR / "job_snippets.jsonl"
    GOLDEN_SET_PATH: Path = DATA_DIR / "golden_set.jsonl"
    RESULTS_JSON_PATH: Path = BASE_DIR / "results.json"

    OPENAI_COST_RATES: dict[str, dict[str, float]] = {
        "gpt-4o-mini": {"in": 0.15 / 1_000_000, "out": 0.60 / 1_000_000},
        "gpt-4o": {"in": 2.50 / 1_000_000, "out": 10.00 / 1_000_000},
    }


settings = Settings()

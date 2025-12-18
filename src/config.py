"""Configuration module for loading environment variables and settings."""

import os
from pathlib import Path
from typing import Optional

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, will use environment variables directly
    pass


class Config:
    """Application configuration loaded from environment variables."""

    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "200"))

    # Pipeline Configuration
    MAX_REVIEWS_PER_SKU: int = int(os.getenv("MAX_REVIEWS_PER_SKU", "50"))
    MAX_TAGS_PER_SKU: int = int(os.getenv("MAX_TAGS_PER_SKU", "6"))
    MAX_CHARS_PER_REVIEW: int = int(os.getenv("MAX_CHARS_PER_REVIEW", "500"))
    MIN_REVIEW_LENGTH: int = int(os.getenv("MIN_REVIEW_LENGTH", "10"))
    MIN_REVIEWS_PER_SKU: int = int(os.getenv("MIN_REVIEWS_PER_SKU", "1"))

    # Data Paths
    INPUT_CSV_PATH: str = os.getenv("INPUT_CSV_PATH", "data/raw/lamoda_reviews.csv")
    OUTPUT_CSV_PATH: str = os.getenv(
        "OUTPUT_CSV_PATH", "data/processed/llm_tags_results.csv"
    )

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate(cls) -> None:
        """Validate that required configuration is present."""
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Please set it in .env file or environment variables."
            )

    @classmethod
    def get_openai_config(cls) -> dict:
        """Get OpenAI client configuration."""
        return {
            "api_key": cls.OPENAI_API_KEY,
            "model": cls.OPENAI_MODEL,
            "temperature": cls.OPENAI_TEMPERATURE,
            "max_tokens": cls.OPENAI_MAX_TOKENS,
        }


# Validate configuration on module import
# Comment this out if you want to handle validation manually
# Config.validate()


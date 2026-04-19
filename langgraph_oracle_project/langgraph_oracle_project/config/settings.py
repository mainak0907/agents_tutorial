"""
config/settings.py
──────────────────
Centralised configuration loaded from .env / environment variables.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Anthropic
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Oracle
    ORACLE_USER: str = os.getenv("ORACLE_USER", "")
    ORACLE_PASSWORD: str = os.getenv("ORACLE_PASSWORD", "")
    ORACLE_DSN: str = os.getenv("ORACLE_DSN", "localhost:1521/XEPDB1")
    ORACLE_WALLET_LOCATION: str | None = os.getenv("ORACLE_WALLET_LOCATION") or None
    ORACLE_WALLET_PASSWORD: str | None = os.getenv("ORACLE_WALLET_PASSWORD") or None

    # App
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # LLM model
    MODEL_NAME: str = "claude-sonnet-4-5"
    MAX_TOKENS: int = 2048

    # Oracle pool
    POOL_MIN: int = 1
    POOL_MAX: int = 5
    POOL_INCREMENT: int = 1


settings = Settings()

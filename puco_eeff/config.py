"""Configuration management for puco-EEFF.

Handles environment variables, API clients, and config file loading.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
AUDIT_DIR = Path(os.getenv("AUDIT_DIR", PROJECT_ROOT / "audit"))
LOGS_DIR = Path(os.getenv("LOGS_DIR", PROJECT_ROOT / "logs"))

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# API Keys
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def get_config() -> dict[str, Any]:
    """Load configuration from config.json."""
    config_path = CONFIG_DIR / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def get_period_paths(year: int, quarter: int) -> dict[str, Path]:
    """Get paths for a specific period (year/quarter).

    Args:
        year: The year (e.g., 2024)
        quarter: The quarter (1-4)

    Returns:
        Dictionary with paths for raw, processed, output, and audit directories.
    """
    period_str = f"{year}_Q{quarter}"

    return {
        "raw_pdf": DATA_DIR / "raw" / "pdf",
        "raw_xbrl": DATA_DIR / "raw" / "xbrl",
        "processed": DATA_DIR / "processed",
        "output": DATA_DIR / "output",
        "audit": AUDIT_DIR / period_str,
    }


def setup_logging(name: str = "puco_eeff") -> logging.Logger:
    """Set up logging for a run.

    Args:
        name: Logger name

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        # File handler
        log_filename = f"{datetime.now().strftime('%Y-%m-%d')}_run.log"
        file_handler = logging.FileHandler(LOGS_DIR / log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def validate_api_keys() -> dict[str, bool]:
    """Check which API keys are configured.

    Returns:
        Dictionary mapping API name to whether it's configured.
    """
    return {
        "mistral": bool(MISTRAL_API_KEY),
        "anthropic": bool(ANTHROPIC_API_KEY),
        "openrouter": bool(OPENROUTER_API_KEY),
        "openai": bool(OPENAI_API_KEY),
    }


def get_mistral_client() -> Any:
    """Get Mistral AI client.

    Returns:
        Configured Mistral client.

    Raises:
        ValueError: If MISTRAL_API_KEY is not set.
    """
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY is not set")

    from mistralai import Mistral

    return Mistral(api_key=MISTRAL_API_KEY)


def get_openrouter_client() -> Any:
    """Get OpenRouter client (uses OpenAI SDK).

    Returns:
        Configured OpenAI client pointing to OpenRouter.

    Raises:
        ValueError: If OPENROUTER_API_KEY is not set.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set")

    from openai import OpenAI

    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )

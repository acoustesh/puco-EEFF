"""Environment validation tests for puco-EEFF."""

import sys

import pytest


def test_python_version() -> None:
    """Verify Python version is 3.12 or higher."""
    assert sys.version_info >= (3, 12), f"Python 3.12+ required, got {sys.version}"


def test_core_imports() -> None:
    """Verify core packages can be imported."""
    import httpx  # noqa: F401
    import lxml  # noqa: F401
    import openpyxl  # noqa: F401
    import pandas as pd  # noqa: F401
    import pdfplumber  # noqa: F401


def test_api_client_imports() -> None:
    """Verify API client packages can be imported."""
    from anthropic import Anthropic  # noqa: F401
    from mistralai import Mistral  # noqa: F401
    from openai import OpenAI  # noqa: F401


def test_playwright_import() -> None:
    """Verify Playwright can be imported."""
    from playwright.sync_api import sync_playwright  # noqa: F401


def test_questionary_import() -> None:
    """Verify questionary can be imported."""
    import questionary  # noqa: F401


def test_project_structure() -> None:
    """Verify project module structure."""
    from puco_eeff import __version__
    from puco_eeff.config import PROJECT_ROOT

    assert __version__ == "0.1.0"
    assert PROJECT_ROOT.exists()


def test_config_loads() -> None:
    """Verify config.json can be loaded."""
    from puco_eeff.config import get_config

    config = get_config()
    assert "sources" in config
    assert "ocr" in config
    assert "period_types" in config  # General config, not sheet-specific


def test_data_directories_exist() -> None:
    """Verify data directories exist."""
    from puco_eeff.config import AUDIT_DIR, DATA_DIR, LOGS_DIR

    assert DATA_DIR.exists()
    assert AUDIT_DIR.exists()
    assert LOGS_DIR.exists()


@pytest.mark.skipif(
    not __import__("os").getenv("MISTRAL_API_KEY"),
    reason="MISTRAL_API_KEY not set",
)
def test_mistral_client_creation() -> None:
    """Verify Mistral client can be created (requires API key)."""
    from puco_eeff.config import get_mistral_client

    client = get_mistral_client()
    assert client is not None


@pytest.mark.skipif(
    not __import__("os").getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)
def test_openrouter_client_creation() -> None:
    """Verify OpenRouter client can be created (requires API key)."""
    from puco_eeff.config import get_openrouter_client

    client = get_openrouter_client()
    assert client is not None


@pytest.mark.skipif(
    not __import__("os").getenv("VOYAGE_API_KEY"),
    reason="VOYAGE_API_KEY not set",
)
def test_voyage_client_creation() -> None:
    """Verify Voyage AI client can be created (requires API key)."""
    import os

    import voyageai

    client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    assert client is not None

"""Pytest configuration for puco_eeff tests.

This module provides:
- CLI options for baseline management (--update-baselines, --cached-only)
- Fixtures for code metrics tests
- Marker registration for similarity tests
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load environment variables from project .env so OPENAI_API_KEY is available in tests
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom CLI options for code metrics tests."""
    parser.addoption(
        "--update-baselines",
        action="store_true",
        default=False,
        help="Update code metrics baselines (line counts, embeddings cache)",
    )
    parser.addoption(
        "--cached-only",
        action="store_true",
        default=False,
        help="Run similarity tests using only cached embeddings (no API calls)",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "similarity: marks tests that use embedding similarity (may require API key)",
    )
    config.addinivalue_line(
        "markers",
        "codestral: marks tests that use Codestral embeddings via OpenRouter",
    )
    config.addinivalue_line(
        "markers",
        "voyage: marks tests that use Voyage AI embeddings",
    )
    config.addinivalue_line(
        "markers",
        "combined: marks tests that use combined weighted embeddings from all providers",
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Reorder tests so combined-marked tests run last.

    Combined embedding tests depend on OpenAI, Codestral, and Voyage embedding
    caches being populated first, so they must run after all other similarity tests.
    """
    combined_tests = []
    other_tests = []

    for item in items:
        if item.get_closest_marker("combined"):
            combined_tests.append(item)
        else:
            other_tests.append(item)

    # Reorder: all other tests first, then combined tests
    items[:] = other_tests + combined_tests


@pytest.fixture
def update_baselines(request: pytest.FixtureRequest) -> bool:
    """Fixture to check if --update-baselines flag was passed."""
    return bool(request.config.getoption("--update-baselines"))


@pytest.fixture
def cached_only(request: pytest.FixtureRequest) -> bool:
    """Fixture to check if --cached-only flag was passed."""
    return bool(request.config.getoption("--cached-only"))

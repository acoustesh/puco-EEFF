"""Pytest configuration for puco_eeff tests.

This module provides:
- CLI options for baseline management (--update-baselines, --cached-only)
- Session-scoped cache sync fixture (validates/cleans caches before tests)
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


@pytest.fixture(scope="session", autouse=True)
def sync_embedding_caches() -> None:
    """Session-scoped fixture: sync, cleanup, and populate embedding caches.

    Order of operations:
    1. Update function_hashes with current functions (hash + text_hash format)
    2. Count missing embeddings
    3. Populate missing embeddings from APIs (if API keys available)
    4. Remove orphans AFTER populate (ensures clean final state)
    5. Rebuild combined embeddings from 6 base providers
    6. Verify counts match
    7. Save (only if changes were made)

    Runs once at session start, before any tests execute.
    """
    from tests.similarity.ast_helpers import extract_all_function_infos
    from tests.similarity.populate_embeddings import (
        _populate_codestral_ast,
        _populate_codestral_text,
        _populate_openai_ast,
        _populate_openai_text,
        _populate_voyage_ast,
        _populate_voyage_text,
        _update_function_hashes,
    )
    from tests.similarity.storage import (
        count_missing_embeddings,
        load_baselines,
        rebuild_combined_embeddings,
        remove_orphan_embeddings,
        save_baselines,
        verify_cache_counts,
    )

    baselines = load_baselines()
    functions = extract_all_function_infos(min_loc=1)

    # 1. Update function_hashes with current functions
    _update_function_hashes(baselines, functions)

    # 2. Count missing embeddings
    missing = count_missing_embeddings(baselines)
    needs_populate = any(v > 0 for v in missing.values())

    # 3. Populate missing embeddings from APIs
    total_new = 0
    if needs_populate:
        total_new += _populate_openai_text(baselines, functions)
        total_new += _populate_openai_ast(baselines, functions)
        total_new += _populate_codestral_text(baselines, functions)
        total_new += _populate_codestral_ast(baselines, functions)
        total_new += _populate_voyage_text(baselines, functions)
        total_new += _populate_voyage_ast(baselines, functions)

    # 4. Remove orphans AFTER populate
    orphan_stats = remove_orphan_embeddings(baselines)
    orphans_removed = sum(orphan_stats.values())

    # 5. Rebuild combined embeddings
    combined_count = rebuild_combined_embeddings(baselines)

    # 6. Verify counts (logged internally if needed)
    _counts = verify_cache_counts(baselines)

    # 7. Save if any changes were made
    if total_new > 0 or orphans_removed > 0 or combined_count > 0:
        save_baselines(baselines)


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
        "openai_text: marks tests that use OpenAI text embeddings",
    )
    config.addinivalue_line(
        "markers",
        "openai_ast: marks tests that use OpenAI AST embeddings",
    )
    config.addinivalue_line(
        "markers",
        "codestral_text: marks tests that use Codestral text embeddings",
    )
    config.addinivalue_line(
        "markers",
        "codestral_ast: marks tests that use Codestral AST embeddings",
    )
    config.addinivalue_line(
        "markers",
        "voyage_text: marks tests that use Voyage text embeddings",
    )
    config.addinivalue_line(
        "markers",
        "voyage_ast: marks tests that use Voyage AST embeddings",
    )
    config.addinivalue_line(
        "markers",
        "combined: marks tests that use combined 6-way embeddings from all providers",
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

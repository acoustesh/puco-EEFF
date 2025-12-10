"""Code Similarity Detection Tests for puco_eeff/.

This module detects near-duplicate functions using embedding providers:
- Text variants: signature + docstring + comments (keyed by text_hash)
- AST variants: ast.unparse() output (keyed by hash)
- Combined: 6-way concatenation of all base embeddings

Each variant detects:
- Pairwise similarity detection (finds copy-paste code)
- Neighbor clustering (finds functions with multiple similar counterparts)
- Refactor index computation (combines complexity + similarity for priority)

Providers:
- OpenAI-Text, OpenAI-AST (text-embedding-3-large)
- Codestral-Text, Codestral-AST (codestral-embed via OpenRouter)
- Voyage-Text, Voyage-AST (voyage-code-3)
- Combined (6-way concatenation)

Baseline Management:
    Run `pytest --update-baselines` to update similarity baselines.
    Review the diff in tests/baselines/similarity_baselines.json and commit via PR.

    For similarity tests without API calls:
    Run `pytest -m similarity --cached-only` to use only cached embeddings.
"""

from __future__ import annotations

import pytest

# Re-export for backward compatibility with test_code_complexity.py
from tests.similarity import (
    CODESTRAL_AST_PROVIDER,
    CODESTRAL_TEXT_PROVIDER,
    COMBINED_PROVIDER,
    OPENAI_AST_PROVIDER,
    OPENAI_TEXT_PROVIDER,
    VOYAGE_AST_PROVIDER,
    VOYAGE_TEXT_PROVIDER,
    _load_all_complexity_maps,
    extract_all_function_infos,
    get_refactor_priority_message_for_complexity,  # noqa: F401
    load_baselines,
    run_provider_similarity_checks,
)


def _run_similarity_test(provider, update_baselines: bool, cached_only: bool) -> None:
    """Common logic for all similarity tests."""
    baselines = load_baselines()
    config = baselines.get("config", {})
    min_loc = config.get("min_loc_for_similarity", 1)
    threshold_pair = config.get(provider.threshold_pair_key, provider.default_threshold_pair)
    threshold_neighbor = config.get(
        provider.threshold_neighbor_key,
        provider.default_threshold_neighbor,
    )

    functions = extract_all_function_infos(min_loc=min_loc)
    run_provider_similarity_checks(
        baselines=baselines,
        functions=functions,
        update_baselines=update_baselines,
        cached_only=cached_only,
        provider=provider,
        threshold_pair=threshold_pair,
        threshold_neighbor=threshold_neighbor,
        load_complexity_maps_fn=_load_all_complexity_maps,
    )


@pytest.mark.similarity
@pytest.mark.openai_text
class TestFunctionSimilarityOpenAIText:
    """Tests using OpenAI text embeddings (signature + docstring + comments)."""

    def test_function_similarity_openai_text(
        self,
        update_baselines: bool,
        cached_only: bool,
    ) -> None:
        """Detect near-duplicate functions using OpenAI text embeddings."""
        _run_similarity_test(OPENAI_TEXT_PROVIDER, update_baselines, cached_only)


@pytest.mark.similarity
@pytest.mark.openai_ast
class TestFunctionSimilarityOpenAIAST:
    """Tests using OpenAI AST embeddings (ast.unparse() output)."""

    def test_function_similarity_openai_ast(
        self,
        update_baselines: bool,
        cached_only: bool,
    ) -> None:
        """Detect near-duplicate functions using OpenAI AST embeddings."""
        _run_similarity_test(OPENAI_AST_PROVIDER, update_baselines, cached_only)


@pytest.mark.similarity
@pytest.mark.codestral_text
class TestFunctionSimilarityCodestralText:
    """Tests using Codestral text embeddings (signature + docstring + comments)."""

    def test_function_similarity_codestral_text(
        self,
        update_baselines: bool,
        cached_only: bool,
    ) -> None:
        """Detect near-duplicate functions using Codestral text embeddings."""
        _run_similarity_test(CODESTRAL_TEXT_PROVIDER, update_baselines, cached_only)


@pytest.mark.similarity
@pytest.mark.codestral_ast
class TestFunctionSimilarityCodestralAST:
    """Tests using Codestral AST embeddings (ast.unparse() output)."""

    def test_function_similarity_codestral_ast(
        self,
        update_baselines: bool,
        cached_only: bool,
    ) -> None:
        """Detect near-duplicate functions using Codestral AST embeddings."""
        _run_similarity_test(CODESTRAL_AST_PROVIDER, update_baselines, cached_only)


@pytest.mark.similarity
@pytest.mark.voyage_text
class TestFunctionSimilarityVoyageText:
    """Tests using Voyage text embeddings (signature + docstring + comments)."""

    def test_function_similarity_voyage_text(
        self,
        update_baselines: bool,
        cached_only: bool,
    ) -> None:
        """Detect near-duplicate functions using Voyage text embeddings."""
        _run_similarity_test(VOYAGE_TEXT_PROVIDER, update_baselines, cached_only)


@pytest.mark.similarity
@pytest.mark.voyage_ast
class TestFunctionSimilarityVoyageAST:
    """Tests using Voyage AST embeddings (ast.unparse() output)."""

    def test_function_similarity_voyage_ast(
        self,
        update_baselines: bool,
        cached_only: bool,
    ) -> None:
        """Detect near-duplicate functions using Voyage AST embeddings."""
        _run_similarity_test(VOYAGE_AST_PROVIDER, update_baselines, cached_only)


@pytest.mark.similarity
@pytest.mark.combined
class TestFunctionSimilarityCombined:
    """Tests using combined 6-way embeddings (all Text + AST variants).

    This test runs LAST because it requires all six base embedding caches
    to be populated first.
    """

    def test_function_similarity_combined(
        self,
        update_baselines: bool,
        cached_only: bool,
    ) -> None:
        """Detect near-duplicate functions using 6-way combined embeddings."""
        _run_similarity_test(COMBINED_PROVIDER, update_baselines, cached_only)

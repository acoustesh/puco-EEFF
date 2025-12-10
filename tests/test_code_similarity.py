"""Code Similarity Detection Tests for puco_eeff/.

This module detects near-duplicate functions using OpenAI embeddings:
- Pairwise similarity detection (finds copy-paste code)
- Neighbor clustering (finds functions with multiple similar counterparts)
- Refactor index computation (combines complexity + similarity for priority)

Baseline Management:
    Run `pytest --update-baselines` to update similarity baselines.
    Review the diff in tests/baselines/similarity_baselines.json and commit via PR.

    For similarity tests without API calls:
    Run `pytest -m similarity --cached-only` to use only cached embeddings.
"""

from __future__ import annotations

import pytest

from tests.similarity import (
    CODESTRAL_PROVIDER,
    COMBINED_PROVIDER,
    OPENAI_PROVIDER,
    VOYAGE_PROVIDER,
    _load_all_complexity_maps,
    _load_complexity_maps,
    extract_all_function_infos,
    extract_function_infos,
    load_baselines,
    run_provider_similarity_checks,
)

# Re-export for backward compatibility with test_code_complexity.py
from tests.similarity import get_refactor_priority_message_for_complexity  # noqa: F401


@pytest.mark.similarity
class TestFunctionSimilarity:
    """Tests for detecting near-duplicate functions using OpenAI embeddings."""

    def test_function_similarity_all_directories(
        self,
        update_baselines: bool,
        cached_only: bool,
    ) -> None:
        """Detect near-duplicate functions across ALL puco_eeff directories.

        Uses OpenAI embeddings with hash-based caching.
        PCA dimensionality reduction is fitted on ALL cached embeddings.
        Fails if:
        - Any pair has similarity >= threshold_pair (default: 0.86)
        - Any function has >=2 neighbors with similarity >= threshold_neighbor (0.80)
        """
        baselines = load_baselines()
        config = baselines.get("config", {})
        min_loc = config.get("min_loc_for_similarity", 1)
        threshold_pair = config.get(
            OPENAI_PROVIDER.threshold_pair_key,
            OPENAI_PROVIDER.default_threshold_pair,
        )
        threshold_neighbor = config.get(
            OPENAI_PROVIDER.threshold_neighbor_key,
            OPENAI_PROVIDER.default_threshold_neighbor,
        )

        functions = extract_all_function_infos(min_loc=min_loc)
        run_provider_similarity_checks(
            baselines=baselines,
            functions=functions,
            update_baselines=update_baselines,
            cached_only=cached_only,
            provider=OPENAI_PROVIDER,
            threshold_pair=threshold_pair,
            threshold_neighbor=threshold_neighbor,
            load_complexity_maps_fn=_load_all_complexity_maps,
        )

    def test_function_similarity(
        self,
        update_baselines: bool,
        cached_only: bool,
    ) -> None:
        """Detect near-duplicate functions in extractor/ only (legacy test).

        Uses OpenAI embeddings with hash-based caching.
        Fails if:
        - Any pair has similarity >= threshold_pair (default: 0.86)
        - Any function has >=2 neighbors with similarity >= threshold_neighbor (0.80)
        """
        baselines = load_baselines()
        config = baselines.get("config", {})
        min_loc = config.get("min_loc_for_similarity", 1)
        threshold_pair = config.get(
            OPENAI_PROVIDER.threshold_pair_key,
            OPENAI_PROVIDER.default_threshold_pair,
        )
        threshold_neighbor = config.get(
            OPENAI_PROVIDER.threshold_neighbor_key,
            OPENAI_PROVIDER.default_threshold_neighbor,
        )

        functions = extract_function_infos(min_loc=min_loc)
        run_provider_similarity_checks(
            baselines=baselines,
            functions=functions,
            update_baselines=update_baselines,
            cached_only=cached_only,
            provider=OPENAI_PROVIDER,
            threshold_pair=threshold_pair,
            threshold_neighbor=threshold_neighbor,
            load_complexity_maps_fn=_load_complexity_maps,
        )


@pytest.mark.similarity
@pytest.mark.codestral
class TestFunctionSimilarityCodestral:
    """Tests for detecting near-duplicate functions using Codestral embeddings via OpenRouter."""

    def test_function_similarity_codestral_all_directories(
        self,
        update_baselines: bool,
        cached_only: bool,
    ) -> None:
        """Detect near-duplicate functions across ALL puco_eeff directories.

        Uses Mistral Codestral-embed embeddings via OpenRouter with hash-based caching.
        Fails if:
        - Any pair has similarity >= threshold_pair (default: 0.97)
        - Any function has >=2 neighbors with similarity >= threshold_neighbor (0.93)
        """
        baselines = load_baselines()
        config = baselines.get("config", {})
        min_loc = config.get("min_loc_for_similarity", 1)
        threshold_pair = config.get(
            CODESTRAL_PROVIDER.threshold_pair_key,
            CODESTRAL_PROVIDER.default_threshold_pair,
        )
        threshold_neighbor = config.get(
            CODESTRAL_PROVIDER.threshold_neighbor_key,
            CODESTRAL_PROVIDER.default_threshold_neighbor,
        )

        functions = extract_all_function_infos(min_loc=min_loc)
        run_provider_similarity_checks(
            baselines=baselines,
            functions=functions,
            update_baselines=update_baselines,
            cached_only=cached_only,
            provider=CODESTRAL_PROVIDER,
            threshold_pair=threshold_pair,
            threshold_neighbor=threshold_neighbor,
            load_complexity_maps_fn=_load_all_complexity_maps,
        )

    def test_function_similarity_codestral(
        self,
        update_baselines: bool,
        cached_only: bool,
    ) -> None:
        """Detect near-duplicate functions in extractor/ only.

        Uses Mistral Codestral-embed embeddings via OpenRouter with hash-based caching.
        Fails if:
        - Any pair has similarity >= threshold_pair (default: 0.97)
        - Any function has >=2 neighbors with similarity >= threshold_neighbor (0.93)
        """
        baselines = load_baselines()
        config = baselines.get("config", {})
        min_loc = config.get("min_loc_for_similarity", 1)
        threshold_pair = config.get(
            CODESTRAL_PROVIDER.threshold_pair_key,
            CODESTRAL_PROVIDER.default_threshold_pair,
        )
        threshold_neighbor = config.get(
            CODESTRAL_PROVIDER.threshold_neighbor_key,
            CODESTRAL_PROVIDER.default_threshold_neighbor,
        )

        functions = extract_function_infos(min_loc=min_loc)
        run_provider_similarity_checks(
            baselines=baselines,
            functions=functions,
            update_baselines=update_baselines,
            cached_only=cached_only,
            provider=CODESTRAL_PROVIDER,
            threshold_pair=threshold_pair,
            threshold_neighbor=threshold_neighbor,
            load_complexity_maps_fn=_load_complexity_maps,
        )


@pytest.mark.similarity
@pytest.mark.voyage
class TestFunctionSimilarityVoyage:
    """Tests for detecting near-duplicate functions using Voyage AI embeddings."""

    def test_function_similarity_voyage_all_directories(
        self,
        update_baselines: bool,
        cached_only: bool,
    ) -> None:
        """Detect near-duplicate functions across ALL puco_eeff directories.

        Uses Voyage voyage-code-3 embeddings with hash-based caching.
        Fails if:
        - Any pair has similarity >= threshold_pair (default: 0.95)
        - Any function has >=2 neighbors with similarity >= threshold_neighbor (0.90)
        """
        baselines = load_baselines()
        config = baselines.get("config", {})
        min_loc = config.get("min_loc_for_similarity", 1)
        threshold_pair = config.get(
            VOYAGE_PROVIDER.threshold_pair_key,
            VOYAGE_PROVIDER.default_threshold_pair,
        )
        threshold_neighbor = config.get(
            VOYAGE_PROVIDER.threshold_neighbor_key,
            VOYAGE_PROVIDER.default_threshold_neighbor,
        )

        functions = extract_all_function_infos(min_loc=min_loc)
        run_provider_similarity_checks(
            baselines=baselines,
            functions=functions,
            update_baselines=update_baselines,
            cached_only=cached_only,
            provider=VOYAGE_PROVIDER,
            threshold_pair=threshold_pair,
            threshold_neighbor=threshold_neighbor,
            load_complexity_maps_fn=_load_all_complexity_maps,
        )

    def test_function_similarity_voyage(
        self,
        update_baselines: bool,
        cached_only: bool,
    ) -> None:
        """Detect near-duplicate functions in extractor/ only.

        Uses Voyage voyage-code-3 embeddings with hash-based caching.
        Fails if:
        - Any pair has similarity >= threshold_pair (default: 0.95)
        - Any function has >=2 neighbors with similarity >= threshold_neighbor (0.90)
        """
        baselines = load_baselines()
        config = baselines.get("config", {})
        min_loc = config.get("min_loc_for_similarity", 1)
        threshold_pair = config.get(
            VOYAGE_PROVIDER.threshold_pair_key,
            VOYAGE_PROVIDER.default_threshold_pair,
        )
        threshold_neighbor = config.get(
            VOYAGE_PROVIDER.threshold_neighbor_key,
            VOYAGE_PROVIDER.default_threshold_neighbor,
        )

        functions = extract_function_infos(min_loc=min_loc)
        run_provider_similarity_checks(
            baselines=baselines,
            functions=functions,
            update_baselines=update_baselines,
            cached_only=cached_only,
            provider=VOYAGE_PROVIDER,
            threshold_pair=threshold_pair,
            threshold_neighbor=threshold_neighbor,
            load_complexity_maps_fn=_load_complexity_maps,
        )


@pytest.mark.similarity
@pytest.mark.combined
class TestFunctionSimilarityCombined:
    """Tests for detecting near-duplicate functions using combined weighted embeddings.

    This test combines OpenAI, Codestral, and Voyage embeddings into a single
    weighted vector. Each provider's embedding is scaled by (n_provider / n_total)
    where n_total is the sum of all provider dimensions.

    This test runs LAST because it requires all three component embedding caches
    to be populated first.
    """

    def test_function_similarity_combined_all_directories(
        self,
        update_baselines: bool,
        cached_only: bool,
    ) -> None:
        """Detect near-duplicate functions across ALL puco_eeff directories.

        Uses weighted combination of OpenAI + Codestral + Voyage embeddings.
        Fails if:
        - Any pair has similarity >= threshold_pair (default: 0.88)
        - Any function has >=2 neighbors with similarity >= threshold_neighbor (0.82)
        """
        baselines = load_baselines()
        config = baselines.get("config", {})
        min_loc = config.get("min_loc_for_similarity", 1)
        threshold_pair = config.get(
            COMBINED_PROVIDER.threshold_pair_key,
            COMBINED_PROVIDER.default_threshold_pair,
        )
        threshold_neighbor = config.get(
            COMBINED_PROVIDER.threshold_neighbor_key,
            COMBINED_PROVIDER.default_threshold_neighbor,
        )

        functions = extract_all_function_infos(min_loc=min_loc)
        run_provider_similarity_checks(
            baselines=baselines,
            functions=functions,
            update_baselines=update_baselines,
            cached_only=cached_only,
            provider=COMBINED_PROVIDER,
            threshold_pair=threshold_pair,
            threshold_neighbor=threshold_neighbor,
            load_complexity_maps_fn=_load_all_complexity_maps,
        )

    def test_function_similarity_combined(
        self,
        update_baselines: bool,
        cached_only: bool,
    ) -> None:
        """Detect near-duplicate functions in extractor/ only.

        Uses weighted combination of OpenAI + Codestral + Voyage embeddings.
        Fails if:
        - Any pair has similarity >= threshold_pair (default: 0.88)
        - Any function has >=2 neighbors with similarity >= threshold_neighbor (0.82)
        """
        baselines = load_baselines()
        config = baselines.get("config", {})
        min_loc = config.get("min_loc_for_similarity", 1)
        threshold_pair = config.get(
            COMBINED_PROVIDER.threshold_pair_key,
            COMBINED_PROVIDER.default_threshold_pair,
        )
        threshold_neighbor = config.get(
            COMBINED_PROVIDER.threshold_neighbor_key,
            COMBINED_PROVIDER.default_threshold_neighbor,
        )

        functions = extract_function_infos(min_loc=min_loc)
        run_provider_similarity_checks(
            baselines=baselines,
            functions=functions,
            update_baselines=update_baselines,
            cached_only=cached_only,
            provider=COMBINED_PROVIDER,
            threshold_pair=threshold_pair,
            threshold_neighbor=threshold_neighbor,
            load_complexity_maps_fn=_load_complexity_maps,
        )

"""Unified similarity check workflow for all embedding providers.

This module provides a single generic function that handles all embedding providers,
eliminating the code duplication that existed in the original 4 separate functions.

Tests operate on cached embeddings only - no API calls are made during tests.
To populate embeddings, run: python -m tests.similarity.populate_embeddings
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from tests.similarity.embeddings import (
    compute_cosine_similarity,
    get_cached_codestral_ast_embedding,
    get_cached_codestral_text_embedding,
    get_cached_combined_embedding,
    get_cached_openai_ast_embedding,
    get_cached_openai_text_embedding,
    get_cached_voyage_ast_embedding,
    get_cached_voyage_text_embedding,
)
from tests.similarity.pca import apply_pca_to_functions
from tests.similarity.refactor_index import get_refactor_priority_message

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.similarity.types import FunctionInfo


@dataclass
class ProviderConfig:
    """Configuration for an embedding provider."""

    name: str
    cache_key: str  # Key in baselines dict for this provider's embeddings
    hash_field: str  # Field on FunctionInfo to use as cache key ("hash" or "text_hash")
    threshold_pair_key: str  # Config key for pair threshold
    threshold_neighbor_key: str  # Config key for neighbor threshold
    pca_variance_key: str  # Config key for PCA variance threshold
    default_threshold_pair: float
    default_threshold_neighbor: float
    default_pca_variance: float
    get_cached_fn: Callable[[dict, str], list[float] | None]


# Provider configurations (6 base + 1 combined)

# Text variants (signature + docstring + comments, keyed by text_hash)
OPENAI_TEXT_PROVIDER = ProviderConfig(
    name="OpenAI-Text",
    cache_key="openai_text_embeddings",
    hash_field="text_hash",
    threshold_pair_key="openai_text_similarity_threshold_pair",
    threshold_neighbor_key="openai_text_similarity_threshold_neighbor",
    pca_variance_key="openai_text_pca_variance_threshold",
    default_threshold_pair=0.86,
    default_threshold_neighbor=0.80,
    default_pca_variance=0.99,
    get_cached_fn=get_cached_openai_text_embedding,
)

CODESTRAL_TEXT_PROVIDER = ProviderConfig(
    name="Codestral-Text",
    cache_key="codestral_text_embeddings",
    hash_field="text_hash",
    threshold_pair_key="codestral_text_similarity_threshold_pair",
    threshold_neighbor_key="codestral_text_similarity_threshold_neighbor",
    pca_variance_key="codestral_text_pca_variance_threshold",
    default_threshold_pair=0.97,
    default_threshold_neighbor=0.93,
    default_pca_variance=0.99,
    get_cached_fn=get_cached_codestral_text_embedding,
)

VOYAGE_TEXT_PROVIDER = ProviderConfig(
    name="Voyage-Text",
    cache_key="voyage_text_embeddings",
    hash_field="text_hash",
    threshold_pair_key="voyage_text_similarity_threshold_pair",
    threshold_neighbor_key="voyage_text_similarity_threshold_neighbor",
    pca_variance_key="voyage_text_pca_variance_threshold",
    default_threshold_pair=0.95,
    default_threshold_neighbor=0.90,
    default_pca_variance=0.99,
    get_cached_fn=get_cached_voyage_text_embedding,
)

# AST variants (ast.unparse() output, keyed by hash)
OPENAI_AST_PROVIDER = ProviderConfig(
    name="OpenAI-AST",
    cache_key="openai_ast_embeddings",
    hash_field="hash",
    threshold_pair_key="openai_ast_similarity_threshold_pair",
    threshold_neighbor_key="openai_ast_similarity_threshold_neighbor",
    pca_variance_key="openai_ast_pca_variance_threshold",
    default_threshold_pair=0.86,
    default_threshold_neighbor=0.80,
    default_pca_variance=0.99,
    get_cached_fn=get_cached_openai_ast_embedding,
)

CODESTRAL_AST_PROVIDER = ProviderConfig(
    name="Codestral-AST",
    cache_key="codestral_ast_embeddings",
    hash_field="hash",
    threshold_pair_key="codestral_ast_similarity_threshold_pair",
    threshold_neighbor_key="codestral_ast_similarity_threshold_neighbor",
    pca_variance_key="codestral_ast_pca_variance_threshold",
    default_threshold_pair=0.97,
    default_threshold_neighbor=0.93,
    default_pca_variance=0.99,
    get_cached_fn=get_cached_codestral_ast_embedding,
)

VOYAGE_AST_PROVIDER = ProviderConfig(
    name="Voyage-AST",
    cache_key="voyage_ast_embeddings",
    hash_field="hash",
    threshold_pair_key="voyage_ast_similarity_threshold_pair",
    threshold_neighbor_key="voyage_ast_similarity_threshold_neighbor",
    pca_variance_key="voyage_ast_pca_variance_threshold",
    default_threshold_pair=0.95,
    default_threshold_neighbor=0.90,
    default_pca_variance=0.99,
    get_cached_fn=get_cached_voyage_ast_embedding,
)

# Combined (6-way concatenation, keyed by text_hash)
COMBINED_PROVIDER = ProviderConfig(
    name="Combined",
    cache_key="combined_embeddings",
    hash_field="text_hash",
    threshold_pair_key="combined_similarity_threshold_pair",
    threshold_neighbor_key="combined_similarity_threshold_neighbor",
    pca_variance_key="combined_pca_variance_threshold",
    default_threshold_pair=0.88,
    default_threshold_neighbor=0.82,
    default_pca_variance=0.99,
    get_cached_fn=get_cached_combined_embedding,
)


def _load_cached_embeddings(
    baselines: dict,
    functions: list[FunctionInfo],
    provider: ProviderConfig,
) -> list[FunctionInfo]:
    """Load cached embeddings and return list of uncached functions."""
    uncached = []
    for func in functions:
        # Use provider-specific hash field (text_hash for Text/Combined, hash for AST)
        cache_key = getattr(func, provider.hash_field)
        cached = provider.get_cached_fn(baselines, cache_key)
        if cached is not None:
            func.embedding = cached
        else:
            uncached.append(func)
    return uncached


def _skip_missing_embeddings(uncached: list[FunctionInfo], provider: ProviderConfig) -> None:
    """Skip test if any functions lack cached embeddings."""
    uncached_names = [f"{f.file}:{f.name}" for f in uncached[:10]]
    more_msg = f"\n  ... and {len(uncached) - 10} more" if len(uncached) > 10 else ""
    pytest.skip(
        f"{len(uncached)} functions lack cached {provider.name} embeddings.\n"
        f"Run: python -m tests.similarity.populate_embeddings --provider "
        f"{provider.name.lower()}\n  " + "\n  ".join(uncached_names) + more_msg,
    )


def _format_pair_violation(func_a: FunctionInfo, func_b: FunctionInfo, similarity: float) -> str:
    """Format a single pair violation message."""
    return (
        f"{func_a.file}:{func_a.start_line} {func_a.name}() vs "
        f"{func_b.file}:{func_b.start_line} {func_b.name}() - "
        f"similarity: {similarity:.1%}"
    )


def _format_neighbor_violation(func_a: FunctionInfo, similar_neighbors: list) -> str:
    """Format a neighbor violation message."""
    neighbor_info = ", ".join(f"{f}:{n}() ({s:.1%})" for f, n, _, s in similar_neighbors[:3])
    return (
        f"{func_a.file}:{func_a.start_line} {func_a.name}() has "
        f"{len(similar_neighbors)} similar functions: {neighbor_info}"
    )


def _check_function_pair(
    func_a: FunctionInfo,
    func_b: FunctionInfo,
    threshold_pair: float,
    threshold_neighbor: float,
) -> tuple[str | None, tuple | None]:
    """Check similarity between two functions, return violations if any."""
    # Caller guarantees embeddings are not None
    similarity = compute_cosine_similarity(func_a.embedding, func_b.embedding)  # type: ignore[arg-type]

    pair_violation = None
    if similarity >= threshold_pair:
        pair_violation = _format_pair_violation(func_a, func_b, similarity)

    neighbor_entry = None
    if similarity >= threshold_neighbor:
        neighbor_entry = (func_b.file, func_b.name, func_b.start_line, similarity)

    return pair_violation, neighbor_entry


def _check_against_others(
    func_a: FunctionInfo,
    func_a_idx: int,
    functions: list[FunctionInfo],
    threshold_pair: float,
    threshold_neighbor: float,
) -> tuple[list[str], list[tuple]]:
    """Check one function against all later functions in the list."""
    pair_violations: list[str] = []
    neighbor_entries: list[tuple] = []

    for j, func_b in enumerate(functions):
        if func_a_idx >= j or func_b.embedding is None:
            continue

        pair_viol, neighbor_entry = _check_function_pair(
            func_a,
            func_b,
            threshold_pair,
            threshold_neighbor,
        )
        if pair_viol:
            pair_violations.append(pair_viol)
        if neighbor_entry:
            neighbor_entries.append(neighbor_entry)

    return pair_violations, neighbor_entries


def _find_violations(
    functions: list[FunctionInfo],
    threshold_pair: float,
    threshold_neighbor: float,
) -> tuple[list[str], list[str]]:
    """Find similarity violations among functions."""
    pair_violations: list[str] = []
    neighbor_violations: list[str] = []

    for i, func_a in enumerate(functions):
        if func_a.embedding is None:
            continue

        func_pair_viols, similar_neighbors = _check_against_others(
            func_a,
            i,
            functions,
            threshold_pair,
            threshold_neighbor,
        )
        pair_violations.extend(func_pair_viols)

        if len(similar_neighbors) >= 2:
            neighbor_violations.append(_format_neighbor_violation(func_a, similar_neighbors))

    return pair_violations, neighbor_violations


def _report_violations(
    pair_violations: list[str],
    neighbor_violations: list[str],
    threshold_pair: float,
    threshold_neighbor: float,
    functions: list[FunctionInfo],
    load_complexity_maps_fn: Callable[[], tuple[dict[str, int], dict[str, int]]],
) -> None:
    """Report violations and fail test if any found."""
    all_violations: list[str] = []
    if pair_violations:
        all_violations.append(
            f"High similarity pairs (>={threshold_pair:.0%}):\n  " + "\n  ".join(pair_violations),
        )
    if neighbor_violations:
        all_violations.append(
            f"Functions with multiple similar neighbors (>={threshold_neighbor:.0%}):\n  "
            + "\n  ".join(neighbor_violations),
        )

    if all_violations:
        error_msg = "\n\n".join(all_violations)
        cc_map, cog_map = load_complexity_maps_fn()
        refactor_msg = get_refactor_priority_message(functions, cc_map, cog_map)
        if refactor_msg:
            error_msg += refactor_msg
        pytest.fail(error_msg)


def run_provider_similarity_checks(
    *,
    baselines: dict,
    functions: list[FunctionInfo],
    update_baselines: bool,
    cached_only: bool,
    provider: ProviderConfig,
    threshold_pair: float,
    threshold_neighbor: float,
    load_complexity_maps_fn: Callable[[], tuple[dict[str, int], dict[str, int]]],
) -> None:
    """Unified workflow for similarity tests across all embedding providers.

    This single function handles OpenAI, Codestral, Voyage, and Combined providers
    with the same flow: load cached → skip if uncached → apply PCA → find violations.

    Tests operate on cached embeddings only - no API calls are made.
    To populate embeddings, run: python -m tests.similarity.populate_embeddings
    """
    if len(functions) < 2:
        pytest.skip("Not enough functions to compare")

    # Load cached embeddings, skip test if any are missing
    uncached = _load_cached_embeddings(baselines, functions, provider)
    if uncached:
        _skip_missing_embeddings(uncached, provider)

    # Apply PCA dimensionality reduction using provider-specific threshold
    config = baselines.get("config", {})
    pca_variance = config.get(provider.pca_variance_key, provider.default_pca_variance)
    embeddings_cache = baselines.get(provider.cache_key, {})
    apply_pca_to_functions(functions, embeddings_cache, variance_threshold=pca_variance)

    # Find and report violations
    pair_violations, neighbor_violations = _find_violations(
        functions,
        threshold_pair,
        threshold_neighbor,
    )
    _report_violations(
        pair_violations,
        neighbor_violations,
        threshold_pair,
        threshold_neighbor,
        functions,
        load_complexity_maps_fn,
    )

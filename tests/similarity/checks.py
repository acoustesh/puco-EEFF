"""Unified similarity check workflow for all embedding providers.

This module provides a single generic function that handles all embedding providers,
eliminating the code duplication that existed in the original 4 separate functions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from tests.similarity.embeddings import (
    compute_combined_embedding,
    compute_cosine_similarity,
    get_cached_codestral_embedding,
    get_cached_combined_embedding,
    get_cached_openai_embedding,
    get_cached_voyage_embedding,
    get_embeddings_batch,
    get_embeddings_batch_codestral,
    get_embeddings_batch_voyage,
)
from tests.similarity.pca import apply_pca_to_functions
from tests.similarity.refactor_index import get_refactor_priority_message
from tests.similarity.storage import save_baselines

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.similarity.types import FunctionInfo


@dataclass
class ProviderConfig:
    """Configuration for an embedding provider."""

    name: str
    cache_key: str  # Key in baselines dict for this provider's embeddings
    threshold_pair_key: str  # Config key for pair threshold
    threshold_neighbor_key: str  # Config key for neighbor threshold
    default_threshold_pair: float
    default_threshold_neighbor: float
    api_key_env_var: str
    api_key_invalid_prefixes: tuple[str, ...]
    get_cached_fn: Callable[[dict, str], list[float] | None]
    get_embeddings_fn: Callable[[list[str]], list[list[float]]] | None  # None for combined


# Provider configurations
OPENAI_PROVIDER = ProviderConfig(
    name="OpenAI",
    cache_key="embeddings",
    threshold_pair_key="similarity_threshold_pair",
    threshold_neighbor_key="similarity_threshold_neighbor",
    default_threshold_pair=0.86,
    default_threshold_neighbor=0.80,
    api_key_env_var="OPENAI_API_KEY",
    api_key_invalid_prefixes=("your_", "sk-xxx"),
    get_cached_fn=get_cached_openai_embedding,
    get_embeddings_fn=get_embeddings_batch,
)

CODESTRAL_PROVIDER = ProviderConfig(
    name="Codestral",
    cache_key="codestral_embeddings",
    threshold_pair_key="codestral_similarity_threshold_pair",
    threshold_neighbor_key="codestral_similarity_threshold_neighbor",
    default_threshold_pair=0.97,
    default_threshold_neighbor=0.93,
    api_key_env_var="OPENROUTER_API_KEY",
    api_key_invalid_prefixes=("your_", "sk-xxx"),
    get_cached_fn=get_cached_codestral_embedding,
    get_embeddings_fn=get_embeddings_batch_codestral,
)

VOYAGE_PROVIDER = ProviderConfig(
    name="Voyage",
    cache_key="voyage_embeddings",
    threshold_pair_key="voyage_similarity_threshold_pair",
    threshold_neighbor_key="voyage_similarity_threshold_neighbor",
    default_threshold_pair=0.95,
    default_threshold_neighbor=0.90,
    api_key_env_var="VOYAGE_API_KEY",
    api_key_invalid_prefixes=("your_", "pa-xxx"),
    get_cached_fn=get_cached_voyage_embedding,
    get_embeddings_fn=get_embeddings_batch_voyage,
)

COMBINED_PROVIDER = ProviderConfig(
    name="Combined",
    cache_key="combined_embeddings",
    threshold_pair_key="combined_similarity_threshold_pair",
    threshold_neighbor_key="combined_similarity_threshold_neighbor",
    default_threshold_pair=0.88,
    default_threshold_neighbor=0.82,
    api_key_env_var="",  # Not used - combined computes from base providers
    api_key_invalid_prefixes=(),
    get_cached_fn=get_cached_combined_embedding,
    get_embeddings_fn=None,  # Set dynamically based on baselines
)


def _load_cached_embeddings(
    baselines: dict,
    functions: list[FunctionInfo],
    provider: ProviderConfig,
) -> list[FunctionInfo]:
    """Load cached embeddings and return list of uncached functions."""
    uncached = []
    for func in functions:
        cached = provider.get_cached_fn(baselines, func.hash)
        if cached is not None:
            func.embedding = cached
        else:
            uncached.append(func)
    return uncached


def _handle_uncached_skip(uncached: list[FunctionInfo], provider: ProviderConfig) -> None:
    """Skip test if in cached-only mode with uncached functions."""
    uncached_names = [f"{f.file}:{f.name}" for f in uncached[:10]]
    more_msg = f"\n  ... and {len(uncached) - 10} more" if len(uncached) > 10 else ""
    pytest.skip(
        f"--cached-only mode: {len(uncached)} functions lack "
        f"cached {provider.name} embeddings:\n  " + "\n  ".join(uncached_names) + more_msg,
    )


def _check_api_key(provider: ProviderConfig) -> str | None:
    """Check if API key is valid, return None if invalid."""
    from tests.similarity.embeddings import _load_api_key_from_env

    api_key = _load_api_key_from_env(provider.api_key_env_var) or os.environ.get(
        provider.api_key_env_var,
    )
    if not api_key or api_key.startswith(provider.api_key_invalid_prefixes) or len(api_key) < 20:
        return None
    return api_key


def _fetch_and_cache_embeddings(
    baselines: dict,
    functions: list[FunctionInfo],
    uncached: list[FunctionInfo],
    provider: ProviderConfig,
) -> None:
    """Fetch embeddings for uncached functions and update cache.

    For Combined provider, computes from base providers instead of API call.
    """
    if provider.name == "Combined":
        _fetch_combined_from_base_providers(baselines, uncached)
        return

    if provider.get_embeddings_fn is None:
        return

    api_key = _check_api_key(provider)
    if not api_key:
        pytest.skip(
            f"{provider.api_key_env_var} not set (or invalid) and some functions lack cached "
            f"{provider.name} embeddings. Set a valid API key or run with --cached-only to skip.",
        )

    texts = [f.text for f in uncached]
    try:
        new_embeddings = provider.get_embeddings_fn(texts)
    except Exception as e:
        error_msg = str(e).lower()
        if "401" in error_msg or "authentication" in error_msg or "api key" in error_msg:
            pytest.skip(
                f"Invalid {provider.api_key_env_var} and some functions lack cached {provider.name} embeddings. "
                f"Set a valid API key or run with --cached-only to skip. Error: {e}",
            )
        pytest.fail(f"Failed to get {provider.name} embeddings: {e}")

    for func, embedding in zip(uncached, new_embeddings, strict=True):
        func.embedding = embedding
        baselines.setdefault(provider.cache_key, {})[func.hash] = embedding

    for func in functions:
        baselines.setdefault("function_hashes", {})[
            f"{func.file}:{func.name}:{func.start_line}"
        ] = func.hash

    save_baselines(baselines)


def _fetch_combined_from_base_providers(
    baselines: dict,
    uncached: list[FunctionInfo],
) -> None:
    """Compute combined embeddings from base providers for uncached functions."""
    missing_base = []
    for func in uncached:
        openai_emb = baselines.get("embeddings", {}).get(func.hash)
        codestral_emb = baselines.get("codestral_embeddings", {}).get(func.hash)
        voyage_emb = baselines.get("voyage_embeddings", {}).get(func.hash)

        if openai_emb and codestral_emb and voyage_emb:
            combined = compute_combined_embedding(openai_emb, codestral_emb, voyage_emb)
            func.embedding = combined
            baselines.setdefault("combined_embeddings", {})[func.hash] = combined
        else:
            missing = []
            if not openai_emb:
                missing.append("OpenAI")
            if not codestral_emb:
                missing.append("Codestral")
            if not voyage_emb:
                missing.append("Voyage")
            missing_base.append((func, missing))

    if missing_base:
        details = [f"{f.file}:{f.name} (missing: {', '.join(m)})" for f, m in missing_base[:10]]
        more_msg = f"\n  ... and {len(missing_base) - 10} more" if len(missing_base) > 10 else ""
        pytest.skip(
            f"{len(missing_base)} functions lack base provider embeddings. "
            f"Run OpenAI, Codestral, and Voyage tests first:\n  " + "\n  ".join(details) + more_msg,
        )

    if baselines.get("combined_embeddings"):
        save_baselines(baselines)


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
    with the same flow: load cached → fetch uncached → apply PCA → find violations.
    """
    if len(functions) < 2:
        pytest.skip("Not enough functions to compare")

    # Same flow for all providers
    uncached = _load_cached_embeddings(baselines, functions, provider)

    if cached_only and uncached:
        _handle_uncached_skip(uncached, provider)

    if uncached:
        _fetch_and_cache_embeddings(baselines, functions, uncached, provider)

    if update_baselines:
        pytest.skip(f"Updated {provider.name} embedding baselines")

    # Apply PCA dimensionality reduction
    config = baselines.get("config", {})
    pca_variance = config.get("pca_variance_threshold", 0.95)
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

"""Refactor index computation for prioritizing refactoring candidates."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import numpy as np

from tests.similarity.constants import (
    DEFAULT_REFACTOR_INDEX_THRESHOLD,
    DEFAULT_REFACTOR_INDEX_TOP_N,
)

if TYPE_CHECKING:
    from tests.similarity.types import FunctionInfo


def compute_similarity_matrix(functions: list[FunctionInfo]) -> np.ndarray:
    """Compute pairwise cosine similarity matrix for all functions."""
    embeddings = []
    for func in functions:
        if func.embedding is not None:
            embeddings.append(func.embedding)
        else:
            embeddings.append([0.0] * 3072)  # text-embedding-3-large dimension

    emb_matrix = np.array(embeddings, dtype=np.float32)

    # Normalize rows
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    emb_normalized = emb_matrix / norms

    # Compute similarity matrix via matrix multiplication
    similarity_matrix = emb_normalized @ emb_normalized.T
    np.fill_diagonal(similarity_matrix, 0.0)

    return similarity_matrix


def compute_max_similarities(similarity_matrix: np.ndarray) -> np.ndarray:
    """Compute max similarity for each function (excluding self)."""
    return np.max(similarity_matrix, axis=1)


def compute_similarity_indices(max_similarities: np.ndarray) -> np.ndarray:
    """Compute similarity index: 25.403 * max_similarity^5."""
    return 25.403 * np.power(max_similarities, 5)


def compute_refactor_indices(
    cc_values: np.ndarray,
    cog_values: np.ndarray,
    similarity_indices: np.ndarray,
) -> np.ndarray:
    """Compute refactor index: 0.25*CC + 0.15*COG + 0.6*similarity_index."""
    return 0.25 * cc_values + 0.15 * cog_values + 0.6 * similarity_indices


def get_refactor_priority_message(
    functions: list[FunctionInfo],
    cc_map: dict[str, int],
    cog_map: dict[str, int],
    threshold: float = DEFAULT_REFACTOR_INDEX_THRESHOLD,
    top_n: int = DEFAULT_REFACTOR_INDEX_TOP_N,
) -> str | None:
    """Generate refactoring priority message for functions above threshold."""
    if len(functions) < 2 or not any(f.embedding is not None for f in functions):
        return None

    similarity_matrix = compute_similarity_matrix(functions)
    max_sims = compute_max_similarities(similarity_matrix)
    similarity_indices = compute_similarity_indices(max_sims)

    cc_values = np.zeros(len(functions), dtype=np.float32)
    cog_values = np.zeros(len(functions), dtype=np.float32)

    for i, func in enumerate(functions):
        key = f"{func.file}:{func.name}"
        cc_values[i] = cc_map.get(key, 0)
        cog_values[i] = cog_map.get(key, 0)

    refactor_indices = compute_refactor_indices(cc_values, cog_values, similarity_indices)

    above_threshold = [
        (
            refactor_indices[i],
            functions[i],
            max_sims[i],
            cc_values[i],
            cog_values[i],
            similarity_indices[i],
        )
        for i in range(len(functions))
        if refactor_indices[i] >= threshold
    ]

    if not above_threshold:
        return None

    above_threshold.sort(key=operator.itemgetter(0), reverse=True)
    top_funcs = above_threshold[:top_n]

    lines = [
        f"\n{'=' * 70}",
        "REFACTORING PRIORITY RECOMMENDATION",
        f"{'=' * 70}",
        f"The following function(s) have Refactor Index >= {threshold:.1f} and should be",
        "prioritized for refactoring:",
        "",
    ]

    for ri, func, max_sim, cc, cog, sim_idx in top_funcs:
        lines.extend((
            f"  {func.file}:{func.start_line} - {func.name}()",
            f"    Refactor Index: {ri:.2f}",
            f"      CC={cc:.0f}, COG={cog:.0f}, MaxSim={max_sim:.2%}, SimIdx={sim_idx:.2f}",
            "",
        ))

    lines.extend((
        "Formula: RefactorIndex = 0.25*CC + 0.15*COG + 0.6*(25.403 * MaxSimilarity^5)",
        f"{'=' * 70}",
    ))

    return "\n".join(lines)


def get_refactor_priority_message_for_complexity() -> str:
    """Get the refactoring priority message for complexity test failures."""
    try:
        from tests.similarity.ast_helpers import extract_all_function_infos
        from tests.similarity.complexity import _load_all_complexity_maps
        from tests.similarity.storage import load_baselines

        baselines = load_baselines()
        config = baselines.get("config", {})
        min_loc = config.get("min_loc_for_similarity", 1)
        threshold = config.get("refactor_index_threshold", DEFAULT_REFACTOR_INDEX_THRESHOLD)
        top_n = config.get("refactor_index_top_n", DEFAULT_REFACTOR_INDEX_TOP_N)

        functions = extract_all_function_infos(min_loc=min_loc)
        if len(functions) < 2:
            return ""

        for func in functions:
            cached = baselines.get("embeddings", {}).get(func.hash)
            if cached is not None:
                func.embedding = cached

        if not any(f.embedding is not None for f in functions):
            return ""

        cc_map, cog_map = _load_all_complexity_maps()
        msg = get_refactor_priority_message(
            functions, cc_map, cog_map, threshold=threshold, top_n=top_n
        )
        return msg or ""
    except Exception:
        return ""

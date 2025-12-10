"""File split proposal generator using PCA and k-means clustering."""

from __future__ import annotations

from pathlib import Path

from tests.similarity.ast_helpers import extract_function_infos_from_file
from tests.similarity.pca import cluster_functions_kmeans_with_pca, fit_pca


def generate_file_split_proposal(file_path: Path, baselines: dict) -> str | None:
    """Generate a proposal to split a file into two based on k-means clustering."""
    functions = extract_function_infos_from_file(file_path, min_loc=1)

    if len(functions) < 4:
        return None

    # Load cached embeddings
    for func in functions:
        cached = baselines.get("embeddings", {}).get(func.hash)
        if cached is not None:
            func.embedding = cached

    funcs_with_embeddings = [f for f in functions if f.embedding is not None]
    if len(funcs_with_embeddings) < 4:
        return f"  (Cannot generate split proposal: only {len(funcs_with_embeddings)} functions have cached embeddings)"

    config = baselines.get("config", {})
    pca_variance = config.get("pca_variance_threshold", 0.95)

    embeddings_cache = baselines.get("embeddings", {})
    pca_model, n_components, _is_gpu = fit_pca(embeddings_cache, variance_threshold=pca_variance)

    if pca_model is None:
        return "  (Cannot generate split proposal: not enough cached embeddings for PCA)"

    clusters, _ = cluster_functions_kmeans_with_pca(
        funcs_with_embeddings,
        pca_model,
        n_components,
        n_clusters=2,
    )

    if len(clusters) < 2 or not clusters[0] or not clusters[1]:
        return None

    base_name = file_path.stem
    pca_variance_pct = int(pca_variance * 100)
    lines = [
        "",
        f"  PROPOSED FILE SPLIT for {file_path.name}:",
        f"  {'─' * 60}",
        f"  (Using PCA with {n_components} components explaining {pca_variance_pct}% variance)",
        "",
        f"  File 1: {base_name}_part1.py ({len(clusters[0])} functions/classes)",
    ]

    lines.extend(
        f"    - {func.name} (lines {func.start_line}-{func.end_line})" for func in clusters[0]
    )
    lines.extend(("", f"  File 2: {base_name}_part2.py ({len(clusters[1])} functions/classes)"))
    lines.extend(
        f"    - {func.name} (lines {func.start_line}-{func.end_line})" for func in clusters[1]
    )
    lines.extend((
        "",
        "  Note: Functions are grouped by semantic similarity using PCA + k-means.",
        f"  {'─' * 60}",
    ))

    return "\n".join(lines)

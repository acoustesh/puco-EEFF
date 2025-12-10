"""PCA dimensionality reduction and clustering helpers."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tests.similarity.types import FunctionInfo


def _get_pca_class() -> tuple:
    """Get the best available PCA implementation (GPU or CPU).

    Returns
    -------
        Tuple of (PCA_class, is_gpu). Falls back to sklearn if cuML unavailable.

    """
    try:
        from cuml.decomposition import PCA

        return PCA, True
    except ImportError:
        from sklearn.decomposition import PCA

        return PCA, False


def fit_pca(
    embeddings_cache: dict[str, list[float]],
    variance_threshold: float,
) -> tuple:
    """Fit PCA on embeddings cache for dimensionality reduction.

    Returns
    -------
        Tuple of (fitted PCA model, number of components to use, is_gpu)
        Returns (None, 0, False) if not enough embeddings (<10)

    """
    if len(embeddings_cache) < 10:
        return None, 0, False

    pca_class, is_gpu = _get_pca_class()
    all_embeddings = np.array(list(embeddings_cache.values()), dtype=np.float32)
    n_samples, n_features = all_embeddings.shape
    max_components = min(n_samples, n_features)

    if is_gpu:
        import cupy as cp

        all_embeddings_gpu = cp.asarray(all_embeddings)
        pca = pca_class(n_components=max_components)
        pca.fit(all_embeddings_gpu)
        cumulative_variance = cp.cumsum(pca.explained_variance_ratio_).get()
    else:
        pca = pca_class(n_components=max_components, random_state=42)
        pca.fit(all_embeddings)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    n_components = int(np.searchsorted(cumulative_variance, variance_threshold).item()) + 1
    n_components = min(n_components, max_components)

    return pca, n_components, is_gpu


def _transform_embeddings(
    functions: list[FunctionInfo],
    pca_model,
    n_components: int,
    is_gpu: bool,
) -> None:
    """Transform embeddings using PCA (GPU or CPU backend)."""
    if not is_gpu:
        warnings.warn(
            "\n" + "=" * 60 + "\n"
            "⚠️  WARNING: Using CPU for PCA transform!\n"
            "    GPU (cuML) is strongly recommended for performance.\n"
            "    Install cuML: conda install -c rapidsai cuml\n" + "=" * 60,
            stacklevel=3,
        )

    # Collect embeddings and indices
    embeddings_to_transform = []
    indices = []
    for i, func in enumerate(functions):
        if func.embedding is not None:
            embeddings_to_transform.append(func.embedding)
            indices.append(i)

    if not embeddings_to_transform:
        return

    # Backend-specific array creation and transform
    if is_gpu:
        import cupy as cp

        emb_array = cp.asarray(embeddings_to_transform, dtype=cp.float32)
        reduced = pca_model.transform(emb_array)[:, :n_components].get()
    else:
        emb_array = np.array(embeddings_to_transform, dtype=np.float32)
        reduced = pca_model.transform(emb_array)[:, :n_components]

    # Assign back
    for idx, emb_idx in enumerate(indices):
        functions[emb_idx].embedding = reduced[idx].tolist()


def transform_embeddings_with_pca(
    functions: list[FunctionInfo],
    pca_model,
    n_components: int,
    is_gpu: bool = True,
) -> None:
    """Transform function embeddings using pre-fitted PCA model in-place."""
    _transform_embeddings(functions, pca_model, n_components, is_gpu)


def apply_pca_to_functions(
    functions: list[FunctionInfo],
    embeddings_cache: dict[str, list[float]],
    variance_threshold: float,
) -> None:
    """Fit PCA on cache and transform function embeddings in-place."""
    pca_model, n_components, is_gpu = fit_pca(embeddings_cache, variance_threshold)

    if pca_model is None:
        return

    original_dim = len(next(iter(embeddings_cache.values())))

    if is_gpu:
        import cupy as cp

        actual_variance = float(cp.sum(pca_model.explained_variance_ratio_[:n_components]).get())
    else:
        actual_variance = float(sum(pca_model.explained_variance_ratio_[:n_components]))

    backend = "GPU" if is_gpu else "CPU"
    warnings.warn(
        f"PCA: {original_dim} → {n_components} dims ({actual_variance:.1%} variance, {backend})",
        stacklevel=2,
    )

    transform_embeddings_with_pca(functions, pca_model, n_components, is_gpu)


def cluster_functions_kmeans_with_pca(
    functions: list[FunctionInfo],
    pca_model,
    n_components: int,
    n_clusters: int = 2,
) -> tuple[list[list[FunctionInfo]], list[str]]:
    """Cluster functions using k-means on PCA-reduced embeddings."""
    from sklearn.cluster import KMeans

    embeddings = []
    valid_functions = []
    for func in functions:
        if func.embedding is not None:
            embeddings.append(func.embedding)
            valid_functions.append(func)

    if len(valid_functions) < n_clusters:
        return [valid_functions], ["all_functions"]

    emb_matrix = np.array(embeddings, dtype=np.float32)
    emb_reduced = pca_model.transform(emb_matrix)[:, :n_components]

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=30, max_iter=3000)
    labels = kmeans.fit_predict(emb_reduced)

    clusters: list[list[FunctionInfo]] = [[] for _ in range(n_clusters)]
    for func, label in zip(valid_functions, labels, strict=True):
        clusters[label].append(func)

    for cluster in clusters:
        cluster.sort(key=lambda f: f.start_line)

    cluster_names = [f"cluster_{i}" for i in range(n_clusters)]
    return clusters, cluster_names

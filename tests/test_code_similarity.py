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

import ast
import base64
import hashlib
import json
import operator
import os
import tempfile
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

# =============================================================================
# Constants
# =============================================================================

# Base directories
PUCO_EEFF_DIR = Path(__file__).parent.parent / "puco_eeff"
EXTRACTOR_DIR = PUCO_EEFF_DIR / "extractor"

# All subdirectories under puco_eeff (for tests that span all)
PUCO_EEFF_SUBDIRS = ["extractor", "scraper", "sheets", "transformer", "writer"]

# Baseline files
BASELINES_FILE = Path(__file__).parent / "baselines" / "similarity_baselines.json"
FUNCTION_HASHES_FILE = Path(__file__).parent / "baselines" / "function_hashes.json"
EMBEDDINGS_FILE = Path(__file__).parent / "baselines" / "embeddings_cache.json.zlib"
CODESTRAL_EMBEDDINGS_FILE = (
    Path(__file__).parent / "baselines" / "codestral_embeddings_cache.json.zlib"
)
VOYAGE_EMBEDDINGS_FILE = (
    Path(__file__).parent / "baselines" / "voyage_embeddings_cache.json.zlib"
)
COMBINED_EMBEDDINGS_FILE = (
    Path(__file__).parent / "baselines" / "combined_embeddings_cache.json.zlib"
)

# OpenRouter config
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/embeddings"
CODESTRAL_EMBED_MODEL = "mistralai/codestral-embed-2505"

# Voyage AI config
VOYAGE_CODE_MODEL = "voyage-code-3"

# Refactor index defaults (can be overridden in config)
DEFAULT_REFACTOR_INDEX_THRESHOLD = 15.0
DEFAULT_REFACTOR_INDEX_TOP_N = 5

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class FunctionInfo:
    """Information about an extracted function."""

    name: str
    file: str
    start_line: int
    end_line: int
    loc: int
    hash: str
    text: str
    embedding: list[float] | None = field(default=None, repr=False)


# =============================================================================
# Embedding Storage Management
# =============================================================================


def _compress_embedding(vec: list[float]) -> str:
    """Compress a single embedding vector to base64-encoded zlib-compressed float32."""
    arr = np.array(vec, dtype=np.float32)
    raw = arr.tobytes()
    comp = zlib.compress(raw, level=6)
    return base64.b64encode(comp).decode("ascii")


def _decompress_embedding(b64_str: str) -> list[float]:
    """Decompress a base64-encoded zlib-compressed float32 embedding."""
    comp = base64.b64decode(b64_str)
    raw = zlib.decompress(comp)
    arr = np.frombuffer(raw, dtype=np.float32)
    return arr.tolist()


def _load_embeddings() -> dict[str, list[float]]:
    """Load and decompress embeddings from the compressed cache file."""
    if not EMBEDDINGS_FILE.exists():
        return {}
    with Path(EMBEDDINGS_FILE).open(encoding="utf-8") as f:
        compressed = json.load(f)
    return {h: _decompress_embedding(b64) for h, b64 in compressed.items()}


def _save_embeddings(embeddings: dict[str, list[float]]) -> None:
    """Save embeddings to compressed cache file atomically."""
    EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

    compressed = {h: _compress_embedding(vec) for h, vec in embeddings.items()}

    fd, temp_path = tempfile.mkstemp(
        suffix=".json.zlib",
        prefix="embeddings_",
        dir=EMBEDDINGS_FILE.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(compressed, f, indent=2, sort_keys=True)
            f.write("\n")
        Path(temp_path).replace(EMBEDDINGS_FILE)
    except Exception:
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        raise


# =============================================================================
# Codestral Embeddings Storage Management (OpenRouter)
# =============================================================================


def _load_codestral_embeddings() -> dict[str, list[float]]:
    """Load and decompress Codestral embeddings from the compressed cache file."""
    if not CODESTRAL_EMBEDDINGS_FILE.exists():
        return {}
    with Path(CODESTRAL_EMBEDDINGS_FILE).open(encoding="utf-8") as f:
        compressed = json.load(f)
    return {h: _decompress_embedding(b64) for h, b64 in compressed.items()}


def _save_codestral_embeddings(embeddings: dict[str, list[float]]) -> None:
    """Save Codestral embeddings to compressed cache file atomically."""
    CODESTRAL_EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

    compressed = {h: _compress_embedding(vec) for h, vec in embeddings.items()}

    fd, temp_path = tempfile.mkstemp(
        suffix=".json.zlib",
        prefix="codestral_embeddings_",
        dir=CODESTRAL_EMBEDDINGS_FILE.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(compressed, f, indent=2, sort_keys=True)
            f.write("\n")
        Path(temp_path).replace(CODESTRAL_EMBEDDINGS_FILE)
    except Exception:
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        raise


# =============================================================================
# Voyage AI Embeddings Storage Management
# =============================================================================


def _load_voyage_embeddings() -> dict[str, list[float]]:
    """Load and decompress Voyage embeddings from the compressed cache file."""
    if not VOYAGE_EMBEDDINGS_FILE.exists():
        return {}
    with Path(VOYAGE_EMBEDDINGS_FILE).open(encoding="utf-8") as f:
        compressed = json.load(f)
    return {h: _decompress_embedding(b64) for h, b64 in compressed.items()}


def _save_voyage_embeddings(embeddings: dict[str, list[float]]) -> None:
    """Save Voyage embeddings to compressed cache file atomically."""
    VOYAGE_EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

    compressed = {h: _compress_embedding(vec) for h, vec in embeddings.items()}

    fd, temp_path = tempfile.mkstemp(
        suffix=".json.zlib",
        prefix="voyage_embeddings_",
        dir=VOYAGE_EMBEDDINGS_FILE.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(compressed, f, indent=2, sort_keys=True)
            f.write("\n")
        Path(temp_path).replace(VOYAGE_EMBEDDINGS_FILE)
    except Exception:
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        raise


# =============================================================================
# Combined Embeddings Storage Management (weighted concatenation of all providers)
# =============================================================================


def _load_combined_embeddings() -> dict[str, list[float]]:
    """Load and decompress Combined embeddings from the compressed cache file."""
    if not COMBINED_EMBEDDINGS_FILE.exists():
        return {}
    with Path(COMBINED_EMBEDDINGS_FILE).open(encoding="utf-8") as f:
        compressed = json.load(f)
    return {h: _decompress_embedding(b64) for h, b64 in compressed.items()}


def _save_combined_embeddings(embeddings: dict[str, list[float]]) -> None:
    """Save Combined embeddings to compressed cache file atomically."""
    COMBINED_EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

    compressed = {h: _compress_embedding(vec) for h, vec in embeddings.items()}

    fd, temp_path = tempfile.mkstemp(
        suffix=".json.zlib",
        prefix="combined_embeddings_",
        dir=COMBINED_EMBEDDINGS_FILE.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(compressed, f, indent=2, sort_keys=True)
            f.write("\n")
        Path(temp_path).replace(COMBINED_EMBEDDINGS_FILE)
    except Exception:
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        raise


def compute_combined_embedding(
    openai_emb: list[float],
    codestral_emb: list[float],
    voyage_emb: list[float],
) -> list[float]:
    """Compute combined embedding as weighted concatenation of all three providers.

    The combined embedding is:
        [codestral * (n_codestral/n_total),
         openai * (n_openai/n_total),
         voyage * (n_voyage/n_total)]

    where n_total = n_codestral + n_openai + n_voyage.

    This creates a single vector that gives each provider's embedding a weight
    proportional to its dimensionality.

    Args
    ----
        openai_emb: OpenAI text-embedding-3-large vector
        codestral_emb: Codestral-embed vector
        voyage_emb: Voyage-code-3 vector

    Returns
    -------
        Combined weighted embedding vector of size n_total
    """
    n_openai = len(openai_emb)
    n_codestral = len(codestral_emb)
    n_voyage = len(voyage_emb)
    n_total = n_openai + n_codestral + n_voyage

    # Weight each embedding by its dimension ratio
    w_openai = n_openai / n_total
    w_codestral = n_codestral / n_total
    w_voyage = n_voyage / n_total

    # Scale each embedding
    scaled_codestral = [x * w_codestral for x in codestral_emb]
    scaled_openai = [x * w_openai for x in openai_emb]
    scaled_voyage = [x * w_voyage for x in voyage_emb]

    # Concatenate: codestral, openai, voyage
    combined = scaled_codestral + scaled_openai + scaled_voyage

    # L2 normalize the combined vector
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = (np.array(combined) / norm).tolist()

    return combined


def get_cached_combined_embedding(baselines: dict, content_hash: str) -> list[float] | None:
    """Return cached Combined embedding vector if hash exists, else None."""
    return baselines.get("combined_embeddings", {}).get(content_hash)


# =============================================================================
# Baseline Management
# =============================================================================


def load_baselines() -> dict:
    """Load baselines from JSON file with fallback to empty defaults."""
    baselines: dict = {
        "function_hashes": {},
        "config": {
            "similarity_threshold_pair": 0.86,
            "similarity_threshold_neighbor": 0.80,
            "min_loc_for_similarity": 1,
            "pca_variance_threshold": 0.99,
            "refactor_index_threshold": 12.0,
            "refactor_index_top_n": 5,
        },
    }

    if BASELINES_FILE.exists():
        with Path(BASELINES_FILE).open(encoding="utf-8") as f:
            baselines.update(json.load(f))

    # Load function_hashes from separate file
    if FUNCTION_HASHES_FILE.exists():
        with Path(FUNCTION_HASHES_FILE).open(encoding="utf-8") as f:
            baselines["function_hashes"] = json.load(f)

    # Load embeddings from separate compressed files
    baselines["embeddings"] = _load_embeddings()
    baselines["codestral_embeddings"] = _load_codestral_embeddings()
    baselines["voyage_embeddings"] = _load_voyage_embeddings()
    baselines["combined_embeddings"] = _load_combined_embeddings()

    return baselines


def save_baselines(baselines: dict) -> None:
    """Save baselines atomically (write to temp, then rename).

    Embeddings and function_hashes are saved separately to their own files.
    """
    BASELINES_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Extract embeddings and function_hashes to save separately
    embeddings = baselines.pop("embeddings", {})
    codestral_embeddings = baselines.pop("codestral_embeddings", {})
    voyage_embeddings = baselines.pop("voyage_embeddings", {})
    combined_embeddings = baselines.pop("combined_embeddings", {})
    function_hashes = baselines.pop("function_hashes", {})

    # Save main baselines (without embeddings and function_hashes)
    fd, temp_path = tempfile.mkstemp(suffix=".json", prefix="baselines_", dir=BASELINES_FILE.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(baselines, f, indent=2, sort_keys=True)
            f.write("\n")
        Path(temp_path).replace(BASELINES_FILE)
    except Exception:
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        raise
    finally:
        # Restore embeddings and function_hashes to dict (in case caller continues to use it)
        baselines["embeddings"] = embeddings
        baselines["codestral_embeddings"] = codestral_embeddings
        baselines["voyage_embeddings"] = voyage_embeddings
        baselines["combined_embeddings"] = combined_embeddings
        baselines["function_hashes"] = function_hashes

    # Save function_hashes to separate file
    if function_hashes:
        fd, temp_path = tempfile.mkstemp(
            suffix=".json",
            prefix="function_hashes_",
            dir=FUNCTION_HASHES_FILE.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(function_hashes, f, indent=2, sort_keys=True)
                f.write("\n")
            Path(temp_path).replace(FUNCTION_HASHES_FILE)
        except Exception:
            if Path(temp_path).exists():
                Path(temp_path).unlink()
            raise

    # Save OpenAI embeddings to compressed file
    if embeddings:
        _save_embeddings(embeddings)

    # Save Codestral embeddings to compressed file
    if codestral_embeddings:
        _save_codestral_embeddings(codestral_embeddings)

    # Save Voyage embeddings to compressed file
    if voyage_embeddings:
        _save_voyage_embeddings(voyage_embeddings)

    # Save Combined embeddings to compressed file
    if combined_embeddings:
        _save_combined_embeddings(combined_embeddings)


# =============================================================================
# AST Helpers - Function Extraction
# =============================================================================


def get_function_text(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    source: str,
) -> str:
    """Extract the full text of a function or class from source code.

    Args:
        node: AST function or class node
        source: Full source code of the file

    Returns
    -------
        Function/class text including signature, docstring, and body

    """
    lines = source.splitlines()
    start = node.lineno - 1  # AST uses 1-based line numbers
    end = node.end_lineno or start + 1
    return "\n".join(lines[start:end])


def normalize_ast_tokens(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
    """Produce deterministic token sequence from function or class AST.

    Strips comments, normalizes whitespace, produces canonical representation
    for hashing purposes.
    """
    # Use ast.dump for deterministic representation
    # This captures structure but not formatting/comments
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def compute_content_hash(normalized_tokens: str) -> str:
    """Compute SHA256 hash of normalized token sequence."""
    return hashlib.sha256(normalized_tokens.encode("utf-8")).hexdigest()


def extract_function_infos(
    min_loc: int = 15,
    directory: Path | None = None,
) -> list[FunctionInfo]:
    """Extract all functions and classes from a directory meeting LOC threshold.

    Args:
        min_loc: Minimum lines of code for inclusion
        directory: Directory to scan (default: EXTRACTOR_DIR)

    Returns
    -------
        List of FunctionInfo objects

    """
    if directory is None:
        directory = EXTRACTOR_DIR

    functions = []

    for py_file in directory.glob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                start = node.lineno
                end = node.end_lineno or start
                loc = end - start + 1

                if loc >= min_loc:
                    text = get_function_text(node, source)
                    normalized = normalize_ast_tokens(node)
                    content_hash = compute_content_hash(normalized)

                    functions.append(
                        FunctionInfo(
                            name=node.name,
                            file=py_file.name,
                            start_line=start,
                            end_line=end,
                            loc=loc,
                            hash=content_hash,
                            text=text,
                        ),
                    )

    return functions


def extract_all_function_infos(min_loc: int = 15) -> list[FunctionInfo]:
    """Extract all functions and classes from ALL puco_eeff directories meeting LOC threshold.

    Args:
        min_loc: Minimum lines of code for inclusion

    Returns
    -------
        List of FunctionInfo objects from all directories

    """
    all_functions = []

    for subdir in PUCO_EEFF_SUBDIRS:
        dir_path = PUCO_EEFF_DIR / subdir
        if dir_path.exists():
            functions = extract_function_infos(min_loc=min_loc, directory=dir_path)
            # Update file field to include subdir for uniqueness
            for func in functions:
                func.file = f"{subdir}/{func.file}"
            all_functions.extend(functions)

    # Also check root-level files in puco_eeff/
    for py_file in PUCO_EEFF_DIR.glob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                start = node.lineno
                end = node.end_lineno or start
                loc = end - start + 1

                if loc >= min_loc:
                    text = get_function_text(node, source)
                    normalized = normalize_ast_tokens(node)
                    content_hash = compute_content_hash(normalized)

                    all_functions.append(
                        FunctionInfo(
                            name=node.name,
                            file=py_file.name,
                            start_line=start,
                            end_line=end,
                            loc=loc,
                            hash=content_hash,
                            text=text,
                        ),
                    )

    return all_functions


def extract_function_infos_from_file(
    file_path: Path,
    min_loc: int = 1,
) -> list[FunctionInfo]:
    """Extract all functions from a single file meeting LOC threshold.

    Args:
        file_path: Path to the Python file
        min_loc: Minimum lines of code for inclusion (default: 1 to include all)

    Returns
    -------
        List of FunctionInfo objects

    """
    functions = []

    source = file_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return functions

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            start = node.lineno
            end = node.end_lineno or start
            loc = end - start + 1

            if loc >= min_loc:
                text = get_function_text(node, source)
                normalized = normalize_ast_tokens(node)
                content_hash = compute_content_hash(normalized)

                functions.append(
                    FunctionInfo(
                        name=node.name,
                        file=file_path.name,
                        start_line=start,
                        end_line=end,
                        loc=loc,
                        hash=content_hash,
                        text=text,
                    ),
                )

    return functions


# =============================================================================
# PCA and Clustering Helpers
# =============================================================================


def _get_pca_class():
    """Get the best available PCA implementation (GPU or CPU).

    Returns (PCA_class, is_gpu). Falls back to sklearn if cuML unavailable.
    CUDA library paths are configured in .env file.
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
):
    """Fit PCA on embeddings cache for dimensionality reduction.

    Uses cuML GPU PCA if available, falls back to sklearn CPU PCA.

    Args:
        embeddings_cache: Dict mapping hash -> embedding vector
        variance_threshold: Cumulative variance to retain (from config pca_variance_threshold)

    Returns
    -------
        Tuple of (fitted PCA model, number of components to use, is_gpu)
        Returns (None, 0, False) if not enough embeddings (<10)

    """
    if len(embeddings_cache) < 10:
        return None, 0, False

    pca_class, is_gpu = _get_pca_class()

    # Build matrix from all cached embeddings
    all_embeddings = np.array(list(embeddings_cache.values()), dtype=np.float32)

    # Fit PCA with all components first to analyze variance
    n_samples, n_features = all_embeddings.shape
    max_components = min(n_samples, n_features)

    if is_gpu:
        import cupy as cp

        all_embeddings_gpu = cp.asarray(all_embeddings)
        pca = pca_class(n_components=max_components)
        pca.fit(all_embeddings_gpu)
        # cuML returns cupy array for explained_variance_ratio_
        cumulative_variance = cp.cumsum(pca.explained_variance_ratio_).get()
    else:
        pca = pca_class(n_components=max_components, random_state=42)
        pca.fit(all_embeddings)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find number of components for desired variance
    n_components = int(np.searchsorted(cumulative_variance, variance_threshold).item()) + 1
    n_components = min(n_components, max_components)

    return pca, n_components, is_gpu


def transform_embeddings_with_pca(
    functions: list[FunctionInfo],
    pca_model,
    n_components: int,
    is_gpu: bool = False,
) -> None:
    """Transform function embeddings using pre-fitted PCA model in-place.

    Args:
        functions: List of FunctionInfo objects with embeddings to transform
        pca_model: Pre-fitted PCA model (cuML or sklearn)
        n_components: Number of components to keep
        is_gpu: Whether PCA model is GPU-based (cuML)
    """
    if is_gpu:
        import cupy as cp

        # Batch transform for GPU efficiency
        embeddings_to_transform = []
        indices = []
        for i, func in enumerate(functions):
            if func.embedding is not None:
                embeddings_to_transform.append(func.embedding)
                indices.append(i)

        if embeddings_to_transform:
            emb_array = cp.asarray(embeddings_to_transform, dtype=cp.float32)
            reduced = pca_model.transform(emb_array)[:, :n_components].get()
            for idx, emb_idx in enumerate(indices):
                functions[emb_idx].embedding = reduced[idx].tolist()
    else:
        for func in functions:
            if func.embedding is not None:
                emb_array = np.array([func.embedding], dtype=np.float32)
                reduced = pca_model.transform(emb_array)[:, :n_components]
                func.embedding = reduced[0].tolist()


def apply_pca_to_functions(
    functions: list[FunctionInfo],
    embeddings_cache: dict[str, list[float]],
    variance_threshold: float,
) -> None:
    """Fit PCA on cache and transform function embeddings in-place.

    Convenience wrapper that calls fit_pca() then transform_embeddings_with_pca().

    Args:
        functions: List of FunctionInfo objects with embeddings to transform
        embeddings_cache: Full cache of embeddings for this provider (to fit PCA on)
        variance_threshold: Cumulative variance to retain (from config pca_variance_threshold)
    """
    import warnings

    pca_model, n_components, is_gpu = fit_pca(embeddings_cache, variance_threshold)
    if pca_model is not None:
        # Get original dimensionality from first embedding in cache
        original_dim = len(next(iter(embeddings_cache.values())))

        # Calculate actual variance retained
        if is_gpu:
            import cupy as cp

            actual_variance = float(
                cp.sum(pca_model.explained_variance_ratio_[:n_components]).get(),
            )
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
    """Cluster functions using k-means on PCA-reduced embeddings.

    Args:
        functions: List of FunctionInfo objects with embeddings
        pca_model: Fitted PCA model on all extractor embeddings
        n_components: Number of PCA components to use
        n_clusters: Number of clusters (default: 2)

    Returns
    -------
        Tuple of (list of function lists per cluster, list of cluster names)

    """
    from sklearn.cluster import KMeans

    # Build embedding matrix
    embeddings = []
    valid_functions = []
    for func in functions:
        if func.embedding is not None:
            embeddings.append(func.embedding)
            valid_functions.append(func)

    if len(valid_functions) < n_clusters:
        return [valid_functions], ["all_functions"]

    emb_matrix = np.array(embeddings, dtype=np.float32)

    # Transform using PCA (reduce dimensionality)
    emb_reduced = pca_model.transform(emb_matrix)[:, :n_components]

    # Since PCA components are orthogonal, no need to normalize
    emb_normalized = emb_reduced

    # Run k-means on reduced embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=30, max_iter=3000)
    labels = kmeans.fit_predict(emb_normalized)

    # Group functions by cluster
    clusters: list[list[FunctionInfo]] = [[] for _ in range(n_clusters)]
    for func, label in zip(valid_functions, labels, strict=True):
        clusters[label].append(func)

    # Sort each cluster by start line for readability
    for cluster in clusters:
        cluster.sort(key=lambda f: f.start_line)

    # Generate cluster names based on dominant theme
    cluster_names = [f"cluster_{i}" for i in range(n_clusters)]

    return clusters, cluster_names


def generate_file_split_proposal(
    file_path: Path,
    baselines: dict,
) -> str | None:
    """Generate a proposal to split a file into two based on k-means clustering.

    Uses PCA fitted on ALL extractor embeddings to reduce dimensionality,
    keeping components that explain 95% of variance, then runs k-means.

    Args:
        file_path: Path to the Python file to analyze
        baselines: Baselines dict with cached embeddings

    Returns
    -------
        Formatted proposal string, or None if not enough functions

    """
    # Extract all functions/classes from the file
    functions = extract_function_infos_from_file(file_path, min_loc=1)

    if len(functions) < 4:
        return None

    # Load cached embeddings for this file's functions
    for func in functions:
        cached = baselines.get("embeddings", {}).get(func.hash)
        if cached is not None:
            func.embedding = cached

    # If too many uncached, we can't cluster effectively
    funcs_with_embeddings = [f for f in functions if f.embedding is not None]
    if len(funcs_with_embeddings) < 4:
        return f"  (Cannot generate split proposal: only {len(funcs_with_embeddings)} functions have cached embeddings)"

    # Get PCA variance threshold from config
    config = baselines.get("config", {})
    pca_variance = config.get("pca_variance_threshold", 0.95)

    # Fit PCA on ALL OpenAI embeddings
    embeddings_cache = baselines.get("embeddings", {})
    pca_model, n_components, _is_gpu = fit_pca(embeddings_cache, variance_threshold=pca_variance)

    if pca_model is None:
        return "  (Cannot generate split proposal: not enough cached embeddings for PCA)"

    # Run k-means clustering on PCA-reduced embeddings
    clusters, _ = cluster_functions_kmeans_with_pca(
        funcs_with_embeddings,
        pca_model,
        n_components,
        n_clusters=2,
    )

    if len(clusters) < 2 or not clusters[0] or not clusters[1]:
        return None

    # Build proposal message
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


# =============================================================================
# Embedding Helpers
# =============================================================================


def get_cached_embedding(baselines: dict, content_hash: str) -> list[float] | None:
    """Return cached embedding vector if hash exists, else None."""
    return baselines.get("embeddings", {}).get(content_hash)


def get_cached_codestral_embedding(baselines: dict, content_hash: str) -> list[float] | None:
    """Return cached Codestral embedding vector if hash exists, else None."""
    return baselines.get("codestral_embeddings", {}).get(content_hash)


def get_embeddings_batch(
    texts: list[str],
    model: str = "text-embedding-3-large",
    max_retries: int = 3,
    timeout: float = 30.0,
) -> list[list[float]]:
    """Get embeddings for a batch of texts using OpenAI API.

    Args:
        texts: List of text strings to embed
        model: OpenAI embedding model name
        max_retries: Number of retry attempts for rate limits
        timeout: Timeout per API call in seconds

    Returns
    -------
        List of embedding vectors

    """
    import openai

    client = openai.OpenAI(timeout=timeout)

    delays = [1.0, 2.0, 4.0]  # Exponential backoff

    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(input=texts, model=model)
            return [item.embedding for item in response.data]
        except openai.RateLimitError:
            if attempt < max_retries - 1:
                time.sleep(delays[attempt])
            else:
                raise
        except openai.APITimeoutError:
            if attempt < max_retries - 1:
                time.sleep(delays[attempt])
            else:
                raise

    return []


def _load_openrouter_api_key() -> str | None:
    """Load OpenRouter API key from .env file or environment."""
    # First check environment
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        return api_key

    # Try loading from .env file
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        with env_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return None


def get_embeddings_batch_codestral(
    texts: list[str],
    model: str = CODESTRAL_EMBED_MODEL,
    max_retries: int = 3,
    timeout: float = 120.0,
) -> list[list[float]]:
    """Get embeddings for a batch of texts using OpenRouter Codestral Embed API.

    Args:
        texts: List of text strings to embed
        model: Codestral model name (default: mistralai/codestral-embed-2505)
        max_retries: Number of retry attempts for rate limits
        timeout: Timeout per API call in seconds

    Returns
    -------
        List of embedding vectors

    """
    import requests

    api_key = _load_openrouter_api_key()
    if not api_key:
        msg = "OPENROUTER_API_KEY not found in environment or .env file"
        raise ValueError(msg)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    delays = [1.0, 2.0, 4.0]  # Exponential backoff

    for attempt in range(max_retries):
        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json={"model": model, "input": texts},
                timeout=timeout,
            )

            if response.status_code == 429:  # Rate limit
                if attempt < max_retries - 1:
                    time.sleep(delays[attempt])
                    continue
                response.raise_for_status()

            response.raise_for_status()
            data = response.json()
            if "data" not in data:
                msg = f"Unexpected API response: {data}"
                raise ValueError(msg)
            return [item["embedding"] for item in data["data"]]

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(delays[attempt])
            else:
                raise

    return []


def get_cached_voyage_embedding(baselines: dict, content_hash: str) -> list[float] | None:
    """Return cached Voyage embedding vector if hash exists, else None."""
    return baselines.get("voyage_embeddings", {}).get(content_hash)


def _load_voyage_api_key() -> str | None:
    """Load Voyage API key from .env file or environment."""
    # First check environment
    api_key = os.environ.get("VOYAGE_API_KEY")
    if api_key:
        return api_key

    # Try loading from .env file
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        with env_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("VOYAGE_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return None


def get_embeddings_batch_voyage(
    texts: list[str],
    model: str = VOYAGE_CODE_MODEL,
    max_retries: int = 3,
    timeout: float = 120.0,
) -> list[list[float]]:
    """Get embeddings for a batch of texts using Voyage AI API.

    Args:
        texts: List of text strings to embed
        model: Voyage model name (default: voyage-code-3)
        max_retries: Number of retry attempts for rate limits
        timeout: Timeout per API call in seconds

    Returns
    -------
        List of embedding vectors

    """
    import voyageai

    api_key = _load_voyage_api_key()
    if not api_key:
        msg = "VOYAGE_API_KEY not found in environment or .env file"
        raise ValueError(msg)

    client = voyageai.Client(api_key=api_key, timeout=timeout)

    delays = [1.0, 2.0, 4.0]  # Exponential backoff

    for attempt in range(max_retries):
        try:
            result = client.embed(texts, model=model, input_type="document")
            return [list(map(float, emb)) for emb in result.embeddings]

        except Exception as e:
            # Check for rate limit error
            error_str = str(e).lower()
            is_rate_limit = "rate" in error_str and "limit" in error_str
            if is_rate_limit and attempt < max_retries - 1:
                time.sleep(delays[attempt])
                continue
            if attempt < max_retries - 1:
                time.sleep(delays[attempt])
            else:
                raise

    return []


def compute_cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors using numpy."""
    a_arr = np.array(a)
    b_arr = np.array(b)

    dot_product = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


# =============================================================================
# Refactor Index Helpers
# =============================================================================


def compute_similarity_matrix(functions: list[FunctionInfo]) -> np.ndarray:
    """Compute pairwise cosine similarity matrix for all functions.

    Args:
        functions: List of FunctionInfo objects with embeddings

    Returns
    -------
        n x n numpy array of cosine similarities

    """
    # Build embedding matrix (n x dim)
    embeddings = []
    for func in functions:
        if func.embedding is not None:
            embeddings.append(func.embedding)
        else:
            # Use zeros for missing embeddings
            embeddings.append([0.0] * 3072)  # text-embedding-3-large dimension

    emb_matrix = np.array(embeddings, dtype=np.float32)

    # Normalize rows
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    emb_normalized = emb_matrix / norms

    # Compute similarity matrix via matrix multiplication
    similarity_matrix = emb_normalized @ emb_normalized.T

    # Set diagonal to 0 (don't compare function to itself)
    np.fill_diagonal(similarity_matrix, 0.0)

    return similarity_matrix


def compute_max_similarities(similarity_matrix: np.ndarray) -> np.ndarray:
    """Compute max similarity for each function (excluding self).

    Args:
        similarity_matrix: n x n similarity matrix

    Returns
    -------
        Array of max similarities for each function

    """
    return np.max(similarity_matrix, axis=1)


def compute_similarity_indices(max_similarities: np.ndarray) -> np.ndarray:
    """Compute similarity index from max similarities.

    Formula: similarity_index = 25.403 * max_similarity^5

    Args:
        max_similarities: Array of max similarity values

    Returns
    -------
        Array of similarity indices

    """
    return 25.403 * np.power(max_similarities, 5)


def compute_refactor_indices(
    cc_values: np.ndarray,
    cog_values: np.ndarray,
    similarity_indices: np.ndarray,
) -> np.ndarray:
    """Compute refactor index for each function.

    Formula: refactor_index = 0.25*CC + 0.15*COG + 0.6*similarity_index

    Args:
        cc_values: Array of cyclomatic complexity values
        cog_values: Array of cognitive complexity values
        similarity_indices: Array of similarity index values

    Returns
    -------
        Array of refactor indices

    """
    return 0.25 * cc_values + 0.15 * cog_values + 0.6 * similarity_indices


def get_refactor_priority_message(
    functions: list[FunctionInfo],
    cc_map: dict[str, int],
    cog_map: dict[str, int],
    threshold: float = DEFAULT_REFACTOR_INDEX_THRESHOLD,
    top_n: int = DEFAULT_REFACTOR_INDEX_TOP_N,
) -> str | None:
    """Generate refactoring priority message for functions above threshold.

    Args:
        functions: List of FunctionInfo objects with embeddings
        cc_map: Dict mapping "file:func" to cyclomatic complexity
        cog_map: Dict mapping "file:func" to cognitive complexity
        threshold: Minimum refactor index to include
        top_n: Maximum number of functions to show

    Returns
    -------
        Formatted message string, or None if no functions meet threshold

    """
    if len(functions) < 2:
        return None

    # Check if any function has an embedding
    if not any(f.embedding is not None for f in functions):
        return None

    # Compute similarity matrix and max similarities
    similarity_matrix = compute_similarity_matrix(functions)
    max_sims = compute_max_similarities(similarity_matrix)
    similarity_indices = compute_similarity_indices(max_sims)

    # Build arrays for CC and COG values
    cc_values = np.zeros(len(functions), dtype=np.float32)
    cog_values = np.zeros(len(functions), dtype=np.float32)

    for i, func in enumerate(functions):
        key = f"{func.file}:{func.name}"
        cc_values[i] = cc_map.get(key, 0)
        cog_values[i] = cog_map.get(key, 0)

    # Compute refactor indices
    refactor_indices = compute_refactor_indices(cc_values, cog_values, similarity_indices)

    # Find functions above threshold
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

    # Sort by refactor index descending
    above_threshold.sort(key=operator.itemgetter(0), reverse=True)

    # Take top N
    top_funcs = above_threshold[:top_n]

    # Format message
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


# =============================================================================
# Complexity Map Loaders (for refactor index integration)
# =============================================================================


def _get_all_function_complexities(file_path: Path) -> list[tuple[str, int, int]]:
    """Get cyclomatic complexity for all functions in a file.

    Returns
    -------
        List of (function_name, line_number, complexity) tuples

    """
    from radon.complexity import cc_visit

    source = file_path.read_text(encoding="utf-8")
    try:
        blocks = cc_visit(source)
    except SyntaxError:
        return []

    return [(block.name, block.lineno, block.complexity) for block in blocks]


def _get_all_cognitive_complexities(file_path: Path) -> list[tuple[str, int, int]]:
    """Get cognitive complexity for all functions in a file using complexipy.

    Returns
    -------
        List of (function_name, line_number, complexity) tuples

    """
    from complexipy import file_complexity

    try:
        result = file_complexity(str(file_path))
    except Exception:
        return []

    return [(func.name, func.line_start, func.complexity) for func in result.functions]


def _load_complexity_maps(directory: Path | None = None) -> tuple[dict[str, int], dict[str, int]]:
    """Load cyclomatic and cognitive complexity maps for functions in a directory.

    Args:
        directory: Directory to scan (default: EXTRACTOR_DIR)

    Returns
    -------
        Tuple of (cc_map, cog_map) where each maps "file:func" to complexity value

    """
    if directory is None:
        directory = EXTRACTOR_DIR

    cc_map: dict[str, int] = {}
    cog_map: dict[str, int] = {}

    for py_file in directory.glob("*.py"):
        # Load cyclomatic complexity
        for func_name, _, cc in _get_all_function_complexities(py_file):
            key = f"{py_file.name}:{func_name}"
            cc_map[key] = cc

        # Load cognitive complexity
        for func_name, _, cog in _get_all_cognitive_complexities(py_file):
            key = f"{py_file.name}:{func_name}"
            cog_map[key] = cog

    return cc_map, cog_map


def _load_all_complexity_maps() -> tuple[dict[str, int], dict[str, int]]:
    """Load cyclomatic and cognitive complexity maps for ALL puco_eeff functions.

    Returns
    -------
        Tuple of (cc_map, cog_map) where each maps "subdir/file:func" to complexity value

    """
    cc_map: dict[str, int] = {}
    cog_map: dict[str, int] = {}

    for subdir in PUCO_EEFF_SUBDIRS:
        dir_path = PUCO_EEFF_DIR / subdir
        if dir_path.exists():
            for py_file in dir_path.glob("*.py"):
                # Load cyclomatic complexity
                for func_name, _, cc in _get_all_function_complexities(py_file):
                    key = f"{subdir}/{py_file.name}:{func_name}"
                    cc_map[key] = cc

                # Load cognitive complexity
                for func_name, _, cog in _get_all_cognitive_complexities(py_file):
                    key = f"{subdir}/{py_file.name}:{func_name}"
                    cog_map[key] = cog

    # Also check root-level files
    for py_file in PUCO_EEFF_DIR.glob("*.py"):
        for func_name, _, cc in _get_all_function_complexities(py_file):
            key = f"{py_file.name}:{func_name}"
            cc_map[key] = cc
        for func_name, _, cog in _get_all_cognitive_complexities(py_file):
            key = f"{py_file.name}:{func_name}"
            cog_map[key] = cog

    return cc_map, cog_map


# =============================================================================
# Public API for complexity tests
# =============================================================================


def get_refactor_priority_message_for_complexity() -> str:
    """Get the refactoring priority message to append to complexity test failures.

    Loads embeddings and complexity data to compute refactor indices.
    Uses ALL puco_eeff directories for comprehensive analysis.
    Returns empty string if no functions meet the threshold.
    """
    try:
        baselines = load_baselines()
        config = baselines.get("config", {})
        min_loc = config.get("min_loc_for_similarity", 1)

        # Get refactor index config values
        threshold = config.get("refactor_index_threshold", DEFAULT_REFACTOR_INDEX_THRESHOLD)
        top_n = config.get("refactor_index_top_n", DEFAULT_REFACTOR_INDEX_TOP_N)

        # Extract functions from ALL puco_eeff directories
        functions = extract_all_function_infos(min_loc=min_loc)
        if len(functions) < 2:
            return ""

        # Load cached embeddings
        for func in functions:
            cached = baselines.get("embeddings", {}).get(func.hash)
            if cached is not None:
                func.embedding = cached

        # Check if any function has an embedding
        if not any(f.embedding is not None for f in functions):
            return ""

        # Load complexity maps for all directories
        cc_map, cog_map = _load_all_complexity_maps()

        # Get refactor priority message
        msg = get_refactor_priority_message(
            functions,
            cc_map,
            cog_map,
            threshold=threshold,
            top_n=top_n,
        )
        return msg or ""
    except Exception:
        # Don't fail the test due to refactor index computation errors
        return ""


# =============================================================================
# Shared test helpers
# =============================================================================


def _run_similarity_checks(
    *,
    baselines: dict,
    functions: list[FunctionInfo],
    update_baselines: bool,
    cached_only: bool,
    threshold_pair: float,
    threshold_neighbor: float,
    load_complexity_maps_fn: Callable[[], tuple[dict[str, int], dict[str, int]]],
) -> None:
    """Common workflow for similarity tests across different scopes."""
    if len(functions) < 2:
        pytest.skip("Not enough functions to compare")

    uncached_functions: list[FunctionInfo] = []
    for func in functions:
        cached = get_cached_embedding(baselines, func.hash)
        if cached is not None:
            func.embedding = cached
        else:
            uncached_functions.append(func)

    if cached_only and uncached_functions:
        uncached_names = [f"{f.file}:{f.name}" for f in uncached_functions]
        pytest.skip(
            f"--cached-only mode: {len(uncached_functions)} functions lack "
            f"cached embeddings:\n  "
            + "\n  ".join(uncached_names[:10])
            + (f"\n  ... and {len(uncached_names) - 10} more" if len(uncached_names) > 10 else ""),
        )

    if uncached_functions:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key or api_key.startswith(("your_", "sk-xxx")) or len(api_key) < 20:
            pytest.skip(
                "OPENAI_API_KEY not set (or invalid) and some functions lack cached embeddings. "
                "Set a valid API key or run with --cached-only to skip.",
            )

        texts = [f.text for f in uncached_functions]
        try:
            new_embeddings = get_embeddings_batch(texts)
        except Exception as e:
            error_msg = str(e).lower()
            if "401" in error_msg or "authentication" in error_msg or "api key" in error_msg:
                pytest.skip(
                    f"Invalid OPENAI_API_KEY and some functions lack cached embeddings. "
                    f"Set a valid API key or run with --cached-only to skip. Error: {e}",
                )
            pytest.fail(f"Failed to get embeddings from OpenAI: {e}")

        for func, embedding in zip(uncached_functions, new_embeddings, strict=True):
            func.embedding = embedding
            baselines.setdefault("embeddings", {})[func.hash] = embedding

        for func in functions:
            baselines.setdefault("function_hashes", {})[
                f"{func.file}:{func.name}:{func.start_line}"
            ] = func.hash

        save_baselines(baselines)

    if update_baselines:
        pytest.skip("Updated embedding baselines")

    # Apply PCA dimensionality reduction before similarity comparison
    config = baselines.get("config", {})
    pca_variance = config.get("pca_variance_threshold", 0.95)
    embeddings_cache = baselines.get("embeddings", {})
    apply_pca_to_functions(functions, embeddings_cache, variance_threshold=pca_variance)

    pair_violations: list[str] = []
    neighbor_violations: list[str] = []

    for i, func_a in enumerate(functions):
        if func_a.embedding is None:
            continue

        similar_neighbors = []

        for j, func_b in enumerate(functions):
            if i >= j or func_b.embedding is None:
                continue

            similarity = compute_cosine_similarity(func_a.embedding, func_b.embedding)

            if similarity >= threshold_pair:
                pair_violations.append(
                    f"{func_a.file}:{func_a.start_line} {func_a.name}() vs "
                    f"{func_b.file}:{func_b.start_line} {func_b.name}() - "
                    f"similarity: {similarity:.1%}",
                )

            if similarity >= threshold_neighbor:
                similar_neighbors.append((func_b.file, func_b.name, func_b.start_line, similarity))

        if len(similar_neighbors) >= 2:
            neighbor_info = ", ".join(
                f"{f}:{n}() ({s:.1%})" for f, n, _, s in similar_neighbors[:3]
            )
            neighbor_violations.append(
                f"{func_a.file}:{func_a.start_line} {func_a.name}() has "
                f"{len(similar_neighbors)} similar functions: {neighbor_info}",
            )

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


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.similarity
class TestFunctionSimilarity:
    """Tests for detecting near-duplicate functions using embeddings."""

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
        threshold_pair = config.get("similarity_threshold_pair", 0.86)
        threshold_neighbor = config.get("similarity_threshold_neighbor", 0.80)

        # Extract functions from ALL puco_eeff directories
        functions = extract_all_function_infos(min_loc=min_loc)
        _run_similarity_checks(
            baselines=baselines,
            functions=functions,
            update_baselines=update_baselines,
            cached_only=cached_only,
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
        threshold_pair = config.get("similarity_threshold_pair", 0.86)
        threshold_neighbor = config.get("similarity_threshold_neighbor", 0.80)

        # Extract functions meeting LOC threshold
        functions = extract_function_infos(min_loc=min_loc)
        _run_similarity_checks(
            baselines=baselines,
            functions=functions,
            update_baselines=update_baselines,
            cached_only=cached_only,
            threshold_pair=threshold_pair,
            threshold_neighbor=threshold_neighbor,
            load_complexity_maps_fn=_load_complexity_maps,
        )


# =============================================================================
# Codestral Similarity Checks (OpenRouter)
# =============================================================================


def _run_codestral_similarity_checks(
    *,
    baselines: dict,
    functions: list[FunctionInfo],
    update_baselines: bool,
    cached_only: bool,
    threshold_pair: float,
    threshold_neighbor: float,
    load_complexity_maps_fn: Callable[[], tuple[dict[str, int], dict[str, int]]],
) -> None:
    """Common workflow for Codestral similarity tests across different scopes."""
    if len(functions) < 2:
        pytest.skip("Not enough functions to compare")

    uncached_functions: list[FunctionInfo] = []
    for func in functions:
        cached = get_cached_codestral_embedding(baselines, func.hash)
        if cached is not None:
            func.embedding = cached
        else:
            uncached_functions.append(func)

    if cached_only and uncached_functions:
        uncached_names = [f"{f.file}:{f.name}" for f in uncached_functions]
        pytest.skip(
            f"--cached-only mode: {len(uncached_functions)} functions lack "
            f"cached Codestral embeddings:\n  "
            + "\n  ".join(uncached_names[:10])
            + (f"\n  ... and {len(uncached_names) - 10} more" if len(uncached_names) > 10 else ""),
        )

    if uncached_functions:
        api_key = _load_openrouter_api_key()
        if not api_key or api_key.startswith(("your_", "sk-xxx")) or len(api_key) < 20:
            pytest.skip(
                "OPENROUTER_API_KEY not set (or invalid) and some functions lack cached "
                "Codestral embeddings. Set a valid API key or run with --cached-only to skip.",
            )

        texts = [f.text for f in uncached_functions]
        try:
            new_embeddings = get_embeddings_batch_codestral(texts)
        except Exception as e:
            error_msg = str(e).lower()
            if "401" in error_msg or "authentication" in error_msg or "api key" in error_msg:
                pytest.skip(
                    f"Invalid OPENROUTER_API_KEY and some functions lack cached Codestral embeddings. "
                    f"Set a valid API key or run with --cached-only to skip. Error: {e}",
                )
            pytest.fail(f"Failed to get Codestral embeddings from OpenRouter: {e}")

        for func, embedding in zip(uncached_functions, new_embeddings, strict=True):
            func.embedding = embedding
            baselines.setdefault("codestral_embeddings", {})[func.hash] = embedding

        for func in functions:
            baselines.setdefault("function_hashes", {})[
                f"{func.file}:{func.name}:{func.start_line}"
            ] = func.hash

        save_baselines(baselines)

    if update_baselines:
        pytest.skip("Updated Codestral embedding baselines")

    # Apply PCA dimensionality reduction before similarity comparison
    config = baselines.get("config", {})
    pca_variance = config.get("pca_variance_threshold", 0.95)
    embeddings_cache = baselines.get("codestral_embeddings", {})
    apply_pca_to_functions(functions, embeddings_cache, variance_threshold=pca_variance)

    pair_violations: list[str] = []
    neighbor_violations: list[str] = []

    for i, func_a in enumerate(functions):
        if func_a.embedding is None:
            continue

        similar_neighbors = []

        for j, func_b in enumerate(functions):
            if i >= j or func_b.embedding is None:
                continue

            similarity = compute_cosine_similarity(func_a.embedding, func_b.embedding)

            if similarity >= threshold_pair:
                pair_violations.append(
                    f"{func_a.file}:{func_a.start_line} {func_a.name}() vs "
                    f"{func_b.file}:{func_b.start_line} {func_b.name}() - "
                    f"similarity: {similarity:.1%}",
                )

            if similarity >= threshold_neighbor:
                similar_neighbors.append((func_b.file, func_b.name, func_b.start_line, similarity))

        if len(similar_neighbors) >= 2:
            neighbor_info = ", ".join(
                f"{f}:{n}() ({s:.1%})" for f, n, _, s in similar_neighbors[:3]
            )
            neighbor_violations.append(
                f"{func_a.file}:{func_a.start_line} {func_a.name}() has "
                f"{len(similar_neighbors)} similar functions: {neighbor_info}",
            )

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
        threshold_pair = config.get("codestral_similarity_threshold_pair", 0.97)
        threshold_neighbor = config.get("codestral_similarity_threshold_neighbor", 0.93)

        # Extract functions from ALL puco_eeff directories
        functions = extract_all_function_infos(min_loc=min_loc)
        _run_codestral_similarity_checks(
            baselines=baselines,
            functions=functions,
            update_baselines=update_baselines,
            cached_only=cached_only,
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
        threshold_pair = config.get("codestral_similarity_threshold_pair", 0.97)
        threshold_neighbor = config.get("codestral_similarity_threshold_neighbor", 0.93)

        # Extract functions meeting LOC threshold
        functions = extract_function_infos(min_loc=min_loc)
        _run_codestral_similarity_checks(
            baselines=baselines,
            functions=functions,
            update_baselines=update_baselines,
            cached_only=cached_only,
            threshold_pair=threshold_pair,
            threshold_neighbor=threshold_neighbor,
            load_complexity_maps_fn=_load_complexity_maps,
        )


def _run_voyage_similarity_checks(
    *,
    baselines: dict,
    functions: list[FunctionInfo],
    update_baselines: bool,
    cached_only: bool,
    threshold_pair: float,
    threshold_neighbor: float,
    load_complexity_maps_fn: Callable[[], tuple[dict[str, int], dict[str, int]]],
) -> None:
    """Common workflow for Voyage similarity tests across different scopes."""
    if len(functions) < 2:
        pytest.skip("Not enough functions to compare")

    uncached_functions: list[FunctionInfo] = []
    for func in functions:
        cached = get_cached_voyage_embedding(baselines, func.hash)
        if cached is not None:
            func.embedding = cached
        else:
            uncached_functions.append(func)

    if cached_only and uncached_functions:
        uncached_names = [f"{f.file}:{f.name}" for f in uncached_functions]
        pytest.skip(
            f"--cached-only mode: {len(uncached_functions)} functions lack "
            f"cached Voyage embeddings:\n  "
            + "\n  ".join(uncached_names[:10])
            + (f"\n  ... and {len(uncached_names) - 10} more" if len(uncached_names) > 10 else ""),
        )

    if uncached_functions:
        api_key = _load_voyage_api_key()
        if not api_key or api_key.startswith(("your_", "pa-xxx")) or len(api_key) < 20:
            pytest.skip(
                "VOYAGE_API_KEY not set (or invalid) and some functions lack cached "
                "Voyage embeddings. Set a valid API key or run with --cached-only to skip.",
            )

        texts = [f.text for f in uncached_functions]
        try:
            new_embeddings = get_embeddings_batch_voyage(texts)
        except Exception as e:
            error_msg = str(e).lower()
            if "401" in error_msg or "authentication" in error_msg or "api key" in error_msg:
                pytest.skip(
                    f"Invalid VOYAGE_API_KEY and some functions lack cached Voyage embeddings. "
                    f"Set a valid API key or run with --cached-only to skip. Error: {e}",
                )
            pytest.fail(f"Failed to get Voyage embeddings: {e}")

        for func, embedding in zip(uncached_functions, new_embeddings, strict=True):
            func.embedding = embedding
            baselines.setdefault("voyage_embeddings", {})[func.hash] = embedding

        for func in functions:
            baselines.setdefault("function_hashes", {})[
                f"{func.file}:{func.name}:{func.start_line}"
            ] = func.hash

        save_baselines(baselines)

    if update_baselines:
        pytest.skip("Updated Voyage embedding baselines")

    # Apply PCA dimensionality reduction before similarity comparison
    config = baselines.get("config", {})
    pca_variance = config.get("pca_variance_threshold", 0.95)
    embeddings_cache = baselines.get("voyage_embeddings", {})
    apply_pca_to_functions(functions, embeddings_cache, variance_threshold=pca_variance)

    pair_violations: list[str] = []
    neighbor_violations: list[str] = []

    for i, func_a in enumerate(functions):
        if func_a.embedding is None:
            continue

        similar_neighbors = []

        for j, func_b in enumerate(functions):
            if i >= j or func_b.embedding is None:
                continue

            similarity = compute_cosine_similarity(func_a.embedding, func_b.embedding)

            if similarity >= threshold_pair:
                pair_violations.append(
                    f"{func_a.file}:{func_a.start_line} {func_a.name}() vs "
                    f"{func_b.file}:{func_b.start_line} {func_b.name}() - "
                    f"similarity: {similarity:.1%}",
                )

            if similarity >= threshold_neighbor:
                similar_neighbors.append((func_b.file, func_b.name, func_b.start_line, similarity))

        if len(similar_neighbors) >= 2:
            neighbor_info = ", ".join(
                f"{f}:{n}() ({s:.1%})" for f, n, _, s in similar_neighbors[:3]
            )
            neighbor_violations.append(
                f"{func_a.file}:{func_a.start_line} {func_a.name}() has "
                f"{len(similar_neighbors)} similar functions: {neighbor_info}",
            )

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
        threshold_pair = config.get("voyage_similarity_threshold_pair", 0.95)
        threshold_neighbor = config.get("voyage_similarity_threshold_neighbor", 0.90)

        # Extract functions from ALL puco_eeff directories
        functions = extract_all_function_infos(min_loc=min_loc)
        _run_voyage_similarity_checks(
            baselines=baselines,
            functions=functions,
            update_baselines=update_baselines,
            cached_only=cached_only,
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
        threshold_pair = config.get("voyage_similarity_threshold_pair", 0.95)
        threshold_neighbor = config.get("voyage_similarity_threshold_neighbor", 0.90)

        # Extract functions meeting LOC threshold
        functions = extract_function_infos(min_loc=min_loc)
        _run_voyage_similarity_checks(
            baselines=baselines,
            functions=functions,
            update_baselines=update_baselines,
            cached_only=cached_only,
            threshold_pair=threshold_pair,
            threshold_neighbor=threshold_neighbor,
            load_complexity_maps_fn=_load_complexity_maps,
        )


def _run_combined_similarity_checks(
    *,
    baselines: dict,
    functions: list[FunctionInfo],
    update_baselines: bool,
    cached_only: bool,
    threshold_pair: float,
    threshold_neighbor: float,
    load_complexity_maps_fn: Callable[[], tuple[dict[str, int], dict[str, int]]],
) -> None:
    """Common workflow for Combined similarity tests across different scopes.

    This test requires all three embedding providers (OpenAI, Codestral, Voyage)
    to have cached embeddings for each function, then computes a weighted
    concatenation of all three.
    """
    if len(functions) < 2:
        pytest.skip("Not enough functions to compare")

    # Check which functions have all three embeddings cached
    uncached_functions: list[FunctionInfo] = []
    for func in functions:
        cached_combined = get_cached_combined_embedding(baselines, func.hash)
        if cached_combined is not None:
            func.embedding = cached_combined
        else:
            # Check if we can compute from component embeddings
            openai_emb = baselines.get("embeddings", {}).get(func.hash)
            codestral_emb = baselines.get("codestral_embeddings", {}).get(func.hash)
            voyage_emb = baselines.get("voyage_embeddings", {}).get(func.hash)

            if openai_emb and codestral_emb and voyage_emb:
                # Compute and cache the combined embedding
                combined = compute_combined_embedding(openai_emb, codestral_emb, voyage_emb)
                func.embedding = combined
                baselines.setdefault("combined_embeddings", {})[func.hash] = combined
            else:
                uncached_functions.append(func)

    if cached_only and uncached_functions:
        # Report which component embeddings are missing
        missing_details = []
        for f in uncached_functions[:10]:
            missing = []
            if not baselines.get("embeddings", {}).get(f.hash):
                missing.append("openai")
            if not baselines.get("codestral_embeddings", {}).get(f.hash):
                missing.append("codestral")
            if not baselines.get("voyage_embeddings", {}).get(f.hash):
                missing.append("voyage")
            missing_details.append(f"{f.file}:{f.name} (missing: {', '.join(missing)})")

        pytest.skip(
            f"--cached-only mode: {len(uncached_functions)} functions lack "
            f"complete embeddings for combined test:\n  "
            + "\n  ".join(missing_details)
            + (
                f"\n  ... and {len(uncached_functions) - 10} more"
                if len(uncached_functions) > 10
                else ""
            ),
        )

    if uncached_functions:
        pytest.skip(
            f"{len(uncached_functions)} functions lack one or more component embeddings "
            f"(OpenAI, Codestral, or Voyage). Run the individual embedding tests first "
            f"to populate all caches, then run this combined test.",
        )

    # Save any newly computed combined embeddings
    if any(baselines.get("combined_embeddings", {})):
        save_baselines(baselines)

    if update_baselines:
        pytest.skip("Updated Combined embedding baselines")

    # Apply PCA dimensionality reduction before similarity comparison
    config = baselines.get("config", {})
    pca_variance = config.get("pca_variance_threshold", 0.95)
    embeddings_cache = baselines.get("combined_embeddings", {})
    apply_pca_to_functions(functions, embeddings_cache, variance_threshold=pca_variance)

    pair_violations: list[str] = []
    neighbor_violations: list[str] = []

    for i, func_a in enumerate(functions):
        if func_a.embedding is None:
            continue

        similar_neighbors = []

        for j, func_b in enumerate(functions):
            if i >= j or func_b.embedding is None:
                continue

            similarity = compute_cosine_similarity(func_a.embedding, func_b.embedding)

            if similarity >= threshold_pair:
                pair_violations.append(
                    f"{func_a.file}:{func_a.start_line} {func_a.name}() vs "
                    f"{func_b.file}:{func_b.start_line} {func_b.name}() - "
                    f"similarity: {similarity:.1%}",
                )

            if similarity >= threshold_neighbor:
                similar_neighbors.append((func_b.file, func_b.name, func_b.start_line, similarity))

        if len(similar_neighbors) >= 2:
            neighbor_info = ", ".join(
                f"{f}:{n}() ({s:.1%})" for f, n, _, s in similar_neighbors[:3]
            )
            neighbor_violations.append(
                f"{func_a.file}:{func_a.start_line} {func_a.name}() has "
                f"{len(similar_neighbors)} similar functions: {neighbor_info}",
            )

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
        threshold_pair = config.get("combined_similarity_threshold_pair", 0.88)
        threshold_neighbor = config.get("combined_similarity_threshold_neighbor", 0.82)

        # Extract functions from ALL puco_eeff directories
        functions = extract_all_function_infos(min_loc=min_loc)
        _run_combined_similarity_checks(
            baselines=baselines,
            functions=functions,
            update_baselines=update_baselines,
            cached_only=cached_only,
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
        threshold_pair = config.get("combined_similarity_threshold_pair", 0.88)
        threshold_neighbor = config.get("combined_similarity_threshold_neighbor", 0.82)

        # Extract functions meeting LOC threshold
        functions = extract_function_infos(min_loc=min_loc)
        _run_combined_similarity_checks(
            baselines=baselines,
            functions=functions,
            update_baselines=update_baselines,
            cached_only=cached_only,
            threshold_pair=threshold_pair,
            threshold_neighbor=threshold_neighbor,
            load_complexity_maps_fn=_load_complexity_maps,
        )

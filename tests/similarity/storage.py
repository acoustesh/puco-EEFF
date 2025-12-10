"""Embedding storage management with compression and atomic writes."""

from __future__ import annotations

import base64
import json
import os
import tempfile
import zlib
from pathlib import Path

import numpy as np

from tests.similarity.constants import (
    BASELINES_FILE,
    CODESTRAL_EMBEDDINGS_FILE,
    COMBINED_EMBEDDINGS_FILE,
    EMBEDDINGS_FILE,
    FUNCTION_HASHES_FILE,
    VOYAGE_EMBEDDINGS_FILE,
)


def _compress_embedding(vec: list[float]) -> str:
    """Compress a single embedding vector to base64-encoded zlib-compressed float32."""
    arr = np.array(vec, dtype=np.float32)
    return base64.b64encode(zlib.compress(arr.tobytes(), level=6)).decode("ascii")


def _decompress_embedding(b64_str: str) -> list[float]:
    """Decompress a base64-encoded zlib-compressed float32 embedding."""
    return np.frombuffer(zlib.decompress(base64.b64decode(b64_str)), dtype=np.float32).tolist()


def _load_compressed_embeddings(file_path: Path) -> dict[str, list[float]]:
    """Load and decompress embeddings from a compressed cache file."""
    if not file_path.exists():
        return {}
    with file_path.open(encoding="utf-8") as f:
        compressed = json.load(f)
    return {h: _decompress_embedding(b64) for h, b64 in compressed.items()}


def _save_compressed_embeddings(embeddings: dict[str, list[float]], file_path: Path) -> None:
    """Save embeddings to compressed cache file atomically."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    compressed = {h: _compress_embedding(vec) for h, vec in embeddings.items()}

    fd, temp_path = tempfile.mkstemp(suffix=".json.zlib", prefix="emb_", dir=file_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(compressed, f, indent=2, sort_keys=True)
            f.write("\n")
        Path(temp_path).replace(file_path)
    except Exception:
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        raise


# Provider cache file mapping
_PROVIDER_FILES = {
    "embeddings": EMBEDDINGS_FILE,
    "codestral_embeddings": CODESTRAL_EMBEDDINGS_FILE,
    "voyage_embeddings": VOYAGE_EMBEDDINGS_FILE,
    "combined_embeddings": COMBINED_EMBEDDINGS_FILE,
}


def validate_embedding_coverage(baselines: dict) -> dict[str, int]:
    """Validate embedding coverage across all providers.

    Returns a dict with coverage statistics. Raises warnings for inconsistencies.
    """
    import warnings

    openai_keys = set(baselines.get("embeddings", {}).keys())
    codestral_keys = set(baselines.get("codestral_embeddings", {}).keys())
    voyage_keys = set(baselines.get("voyage_embeddings", {}).keys())
    combined_keys = set(baselines.get("combined_embeddings", {}).keys())

    all_three = openai_keys & codestral_keys & voyage_keys
    missing_from_combined = all_three - combined_keys

    stats = {
        "openai": len(openai_keys),
        "codestral": len(codestral_keys),
        "voyage": len(voyage_keys),
        "combined": len(combined_keys),
        "all_three_providers": len(all_three),
        "missing_from_combined": len(missing_from_combined),
    }

    if missing_from_combined:
        warnings.warn(
            f"Embedding coverage gap: {len(missing_from_combined)} functions have all 3 "
            f"provider embeddings but are missing from combined cache. "
            f"Run the combined similarity test to regenerate.",
            stacklevel=2,
        )

    if len(openai_keys) != len(codestral_keys) or len(openai_keys) != len(voyage_keys):
        warnings.warn(
            f"Provider embedding counts differ: OpenAI={len(openai_keys)}, "
            f"Codestral={len(codestral_keys)}, Voyage={len(voyage_keys)}. "
            f"Run individual provider tests to sync caches.",
            stacklevel=2,
        )

    return stats


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
        with BASELINES_FILE.open(encoding="utf-8") as f:
            baselines.update(json.load(f))

    if FUNCTION_HASHES_FILE.exists():
        with FUNCTION_HASHES_FILE.open(encoding="utf-8") as f:
            baselines["function_hashes"] = json.load(f)

    # Load all embedding caches using provider mapping
    for cache_key, file_path in _PROVIDER_FILES.items():
        baselines[cache_key] = _load_compressed_embeddings(file_path)

    return baselines


def save_baselines(baselines: dict) -> None:
    """Save baselines atomically. Embeddings and function_hashes are saved separately."""
    BASELINES_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Extract embedding caches to save separately
    embedding_caches = {key: baselines.pop(key, {}) for key in _PROVIDER_FILES}
    function_hashes = baselines.pop("function_hashes", {})

    # Save main baselines
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
        # Restore to dict
        baselines.update(embedding_caches)
        baselines["function_hashes"] = function_hashes

    # Save function_hashes
    if function_hashes:
        fd, temp_path = tempfile.mkstemp(
            suffix=".json",
            prefix="hashes_",
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

    # Save embedding caches using provider mapping
    for cache_key, file_path in _PROVIDER_FILES.items():
        cache = embedding_caches.get(cache_key, {})
        if cache:
            _save_compressed_embeddings(cache, file_path)

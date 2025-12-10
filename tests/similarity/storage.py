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
    CODESTRAL_AST_EMBEDDINGS_FILE,
    CODESTRAL_TEXT_EMBEDDINGS_FILE,
    COMBINED_EMBEDDINGS_FILE,
    FUNCTION_HASHES_FILE,
    OPENAI_AST_EMBEDDINGS_FILE,
    OPENAI_TEXT_EMBEDDINGS_FILE,
    VOYAGE_AST_EMBEDDINGS_FILE,
    VOYAGE_TEXT_EMBEDDINGS_FILE,
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


# Provider cache file mapping (6 base + 1 combined)
_PROVIDER_FILES = {
    # Text variants (keyed by text_hash)
    "openai_text_embeddings": OPENAI_TEXT_EMBEDDINGS_FILE,
    "codestral_text_embeddings": CODESTRAL_TEXT_EMBEDDINGS_FILE,
    "voyage_text_embeddings": VOYAGE_TEXT_EMBEDDINGS_FILE,
    # AST variants (keyed by hash)
    "openai_ast_embeddings": OPENAI_AST_EMBEDDINGS_FILE,
    "codestral_ast_embeddings": CODESTRAL_AST_EMBEDDINGS_FILE,
    "voyage_ast_embeddings": VOYAGE_AST_EMBEDDINGS_FILE,
    # Combined (keyed by text_hash)
    "combined_embeddings": COMBINED_EMBEDDINGS_FILE,
}

# Cache keys grouped by hash type for sync operations
_TEXT_HASH_CACHES = [
    "openai_text_embeddings",
    "codestral_text_embeddings",
    "voyage_text_embeddings",
]
_AST_HASH_CACHES = ["openai_ast_embeddings", "codestral_ast_embeddings", "voyage_ast_embeddings"]


def get_valid_hashes(baselines: dict) -> tuple[set[str], set[str], dict[str, str]]:
    """Extract valid hashes from function_hashes.

    Returns
    -------
        tuple: (valid_text_hashes, valid_ast_hashes, text_to_ast_map)
            - valid_text_hashes: set of text_hash values
            - valid_ast_hashes: set of hash (AST) values
            - text_to_ast_map: mapping from text_hash -> hash for combined lookup
    """
    function_hashes = baselines.get("function_hashes", {})
    valid_text_hashes: set[str] = set()
    valid_ast_hashes: set[str] = set()
    text_to_ast_map: dict[str, str] = {}

    for entry in function_hashes.values():
        if isinstance(entry, dict):
            ast_hash = entry.get("hash")
            text_hash = entry.get("text_hash")
            if ast_hash:
                valid_ast_hashes.add(ast_hash)
            if text_hash:
                valid_text_hashes.add(text_hash)
            if ast_hash and text_hash:
                text_to_ast_map[text_hash] = ast_hash
        else:
            # Legacy format: entry is just the hash string (AST only)
            valid_ast_hashes.add(entry)

    return valid_text_hashes, valid_ast_hashes, text_to_ast_map


def count_missing_embeddings(baselines: dict) -> dict[str, int]:
    """Count missing embeddings per provider.

    Returns dict with missing_<provider> counts.
    """
    valid_text, valid_ast, _ = get_valid_hashes(baselines)

    return {
        "missing_openai_text": len(
            valid_text - set(baselines.get("openai_text_embeddings", {}).keys())
        ),
        "missing_codestral_text": len(
            valid_text - set(baselines.get("codestral_text_embeddings", {}).keys())
        ),
        "missing_voyage_text": len(
            valid_text - set(baselines.get("voyage_text_embeddings", {}).keys())
        ),
        "missing_openai_ast": len(
            valid_ast - set(baselines.get("openai_ast_embeddings", {}).keys())
        ),
        "missing_codestral_ast": len(
            valid_ast - set(baselines.get("codestral_ast_embeddings", {}).keys())
        ),
        "missing_voyage_ast": len(
            valid_ast - set(baselines.get("voyage_ast_embeddings", {}).keys())
        ),
    }


def remove_orphan_embeddings(baselines: dict) -> dict[str, int]:
    """Remove orphan embeddings (hashes not in function_hashes).

    Modifies baselines in place.
    Returns dict with orphans_removed_<provider> counts.
    """
    valid_text, valid_ast, _ = get_valid_hashes(baselines)

    orphan_stats: dict[str, int] = {}
    cache_configs = [
        ("openai_text", "openai_text_embeddings", valid_text),
        ("codestral_text", "codestral_text_embeddings", valid_text),
        ("voyage_text", "voyage_text_embeddings", valid_text),
        ("openai_ast", "openai_ast_embeddings", valid_ast),
        ("codestral_ast", "codestral_ast_embeddings", valid_ast),
        ("voyage_ast", "voyage_ast_embeddings", valid_ast),
        ("combined", "combined_embeddings", valid_text),
    ]

    for name, cache_key, valid_hashes in cache_configs:
        cache = baselines.get(cache_key, {})
        orphans = set(cache.keys()) - valid_hashes
        for h in orphans:
            del cache[h]
        orphan_stats[f"orphans_removed_{name}"] = len(orphans)

    return orphan_stats


def rebuild_combined_embeddings(baselines: dict) -> int:
    """Rebuild combined embeddings from all 6 base providers.

    Combined uses text_hash as key and requires all 6 embeddings.
    Modifies baselines["combined_embeddings"] in place.

    Returns count of combined embeddings built.
    """
    from tests.similarity.embeddings import compute_combined_embedding

    valid_text, _, text_to_ast = get_valid_hashes(baselines)

    openai_text = baselines.get("openai_text_embeddings", {})
    codestral_text = baselines.get("codestral_text_embeddings", {})
    voyage_text = baselines.get("voyage_text_embeddings", {})
    openai_ast = baselines.get("openai_ast_embeddings", {})
    codestral_ast = baselines.get("codestral_ast_embeddings", {})
    voyage_ast = baselines.get("voyage_ast_embeddings", {})

    combined_cache: dict[str, list[float]] = {}
    for text_hash in valid_text:
        ast_hash = text_to_ast.get(text_hash)
        if not ast_hash:
            continue

        ot = openai_text.get(text_hash)
        ct = codestral_text.get(text_hash)
        vt = voyage_text.get(text_hash)
        oa = openai_ast.get(ast_hash)
        ca = codestral_ast.get(ast_hash)
        va = voyage_ast.get(ast_hash)

        if all([ot, ct, vt, oa, ca, va]):
            combined_cache[text_hash] = compute_combined_embedding(ot, oa, ct, ca, vt, va)

    baselines["combined_embeddings"] = combined_cache
    return len(combined_cache)


def verify_cache_counts(baselines: dict) -> dict[str, int]:
    """Verify cache counts for all providers.

    Returns dict with counts per cache. All text caches should match,
    all AST caches should match.
    """
    valid_text, valid_ast, _ = get_valid_hashes(baselines)

    return {
        "valid_text_hashes": len(valid_text),
        "valid_ast_hashes": len(valid_ast),
        "openai_text": len(baselines.get("openai_text_embeddings", {})),
        "codestral_text": len(baselines.get("codestral_text_embeddings", {})),
        "voyage_text": len(baselines.get("voyage_text_embeddings", {})),
        "openai_ast": len(baselines.get("openai_ast_embeddings", {})),
        "codestral_ast": len(baselines.get("codestral_ast_embeddings", {})),
        "voyage_ast": len(baselines.get("voyage_ast_embeddings", {})),
        "combined": len(baselines.get("combined_embeddings", {})),
    }


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

#!/usr/bin/env python3
"""Populate embedding caches for all providers.

This script fetches embeddings from APIs for functions that don't have cached values.
Run this before running tests if you need to populate missing embeddings.

Usage:
    python -m tests.similarity.populate_embeddings [--provider PROVIDER]

Options:
    --provider  One of: openai-text, openai-ast, codestral-text, codestral-ast,
                voyage-text, voyage-ast, combined, all (default: all)

Examples
--------
    # Populate all providers
    python -m tests.similarity.populate_embeddings

    # Populate only OpenAI text variant
    python -m tests.similarity.populate_embeddings --provider openai-text

    # Populate combined (requires all 6 base providers to be populated first)
    python -m tests.similarity.populate_embeddings --provider combined
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.similarity.types import FunctionInfo


def _load_api_key(env_var: str, invalid_prefixes: tuple[str, ...]) -> str | None:
    """Load and validate API key from environment."""
    from tests.similarity.embeddings import _load_api_key_from_env

    api_key = _load_api_key_from_env(env_var) or os.environ.get(env_var)
    if not api_key or api_key.startswith(invalid_prefixes) or len(api_key) < 20:
        return None
    return api_key


def _populate_openai_text(
    baselines: dict,
    functions: list[FunctionInfo],
) -> int:
    """Populate OpenAI text embeddings (signature + docstring + comments)."""
    from tests.similarity.embeddings import (
        get_cached_openai_text_embedding,
        get_embeddings_batch,
    )

    api_key = _load_api_key("OPENAI_API_KEY", ("your_", "sk-xxx"))
    if not api_key:
        print("OPENAI_API_KEY not set or invalid, skipping OpenAI-Text")
        return 0

    uncached = [
        f for f in functions if get_cached_openai_text_embedding(baselines, f.text_hash) is None
    ]
    if not uncached:
        print("OpenAI-Text: all functions already cached")
        return 0

    print(f"OpenAI-Text: fetching embeddings for {len(uncached)} functions...")
    try:
        texts = [f.text_for_embedding for f in uncached]
        embeddings = get_embeddings_batch(texts)
        for func, emb in zip(uncached, embeddings, strict=True):
            baselines.setdefault("openai_text_embeddings", {})[func.text_hash] = emb
        print(f"OpenAI-Text: cached {len(uncached)} new embeddings")
        return len(uncached)
    except Exception as e:
        print(f"OpenAI-Text: failed to fetch embeddings: {e}")
        return 0


def _populate_openai_ast(
    baselines: dict,
    functions: list[FunctionInfo],
) -> int:
    """Populate OpenAI AST embeddings (ast.unparse() output)."""
    from tests.similarity.embeddings import get_cached_openai_ast_embedding, get_embeddings_batch

    api_key = _load_api_key("OPENAI_API_KEY", ("your_", "sk-xxx"))
    if not api_key:
        print("OPENAI_API_KEY not set or invalid, skipping OpenAI-AST")
        return 0

    uncached = [f for f in functions if get_cached_openai_ast_embedding(baselines, f.hash) is None]
    if not uncached:
        print("OpenAI-AST: all functions already cached")
        return 0

    print(f"OpenAI-AST: fetching embeddings for {len(uncached)} functions...")
    try:
        texts = [f.ast_text for f in uncached]
        embeddings = get_embeddings_batch(texts)
        for func, emb in zip(uncached, embeddings, strict=True):
            baselines.setdefault("openai_ast_embeddings", {})[func.hash] = emb
        print(f"OpenAI-AST: cached {len(uncached)} new embeddings")
        return len(uncached)
    except Exception as e:
        print(f"OpenAI-AST: failed to fetch embeddings: {e}")
        return 0


def _populate_codestral_text(
    baselines: dict,
    functions: list[FunctionInfo],
) -> int:
    """Populate Codestral text embeddings (signature + docstring + comments)."""
    from tests.similarity.embeddings import (
        get_cached_codestral_text_embedding,
        get_embeddings_batch_codestral,
    )

    api_key = _load_api_key("OPENROUTER_API_KEY", ("your_", "sk-xxx"))
    if not api_key:
        print("OPENROUTER_API_KEY not set or invalid, skipping Codestral-Text")
        return 0

    uncached = [
        f for f in functions if get_cached_codestral_text_embedding(baselines, f.text_hash) is None
    ]
    if not uncached:
        print("Codestral-Text: all functions already cached")
        return 0

    print(f"Codestral-Text: fetching embeddings for {len(uncached)} functions...")
    try:
        texts = [f.text_for_embedding for f in uncached]
        embeddings = get_embeddings_batch_codestral(texts)
        for func, emb in zip(uncached, embeddings, strict=True):
            baselines.setdefault("codestral_text_embeddings", {})[func.text_hash] = emb
        print(f"Codestral-Text: cached {len(uncached)} new embeddings")
        return len(uncached)
    except Exception as e:
        print(f"Codestral-Text: failed to fetch embeddings: {e}")
        return 0


def _populate_codestral_ast(
    baselines: dict,
    functions: list[FunctionInfo],
) -> int:
    """Populate Codestral AST embeddings (ast.unparse() output)."""
    from tests.similarity.embeddings import (
        get_cached_codestral_ast_embedding,
        get_embeddings_batch_codestral,
    )

    api_key = _load_api_key("OPENROUTER_API_KEY", ("your_", "sk-xxx"))
    if not api_key:
        print("OPENROUTER_API_KEY not set or invalid, skipping Codestral-AST")
        return 0

    uncached = [
        f for f in functions if get_cached_codestral_ast_embedding(baselines, f.hash) is None
    ]
    if not uncached:
        print("Codestral-AST: all functions already cached")
        return 0

    print(f"Codestral-AST: fetching embeddings for {len(uncached)} functions...")
    try:
        texts = [f.ast_text for f in uncached]
        embeddings = get_embeddings_batch_codestral(texts)
        for func, emb in zip(uncached, embeddings, strict=True):
            baselines.setdefault("codestral_ast_embeddings", {})[func.hash] = emb
        print(f"Codestral-AST: cached {len(uncached)} new embeddings")
        return len(uncached)
    except Exception as e:
        print(f"Codestral-AST: failed to fetch embeddings: {e}")
        return 0


def _populate_voyage_text(
    baselines: dict,
    functions: list[FunctionInfo],
) -> int:
    """Populate Voyage text embeddings (signature + docstring + comments)."""
    from tests.similarity.embeddings import (
        get_cached_voyage_text_embedding,
        get_embeddings_batch_voyage,
    )

    api_key = _load_api_key("VOYAGE_API_KEY", ("your_", "pa-xxx"))
    if not api_key:
        print("VOYAGE_API_KEY not set or invalid, skipping Voyage-Text")
        return 0

    uncached = [
        f for f in functions if get_cached_voyage_text_embedding(baselines, f.text_hash) is None
    ]
    if not uncached:
        print("Voyage-Text: all functions already cached")
        return 0

    print(f"Voyage-Text: fetching embeddings for {len(uncached)} functions...")
    try:
        texts = [f.text_for_embedding for f in uncached]
        embeddings = get_embeddings_batch_voyage(texts)
        for func, emb in zip(uncached, embeddings, strict=True):
            baselines.setdefault("voyage_text_embeddings", {})[func.text_hash] = emb
        print(f"Voyage-Text: cached {len(uncached)} new embeddings")
        return len(uncached)
    except Exception as e:
        print(f"Voyage-Text: failed to fetch embeddings: {e}")
        return 0


def _populate_voyage_ast(
    baselines: dict,
    functions: list[FunctionInfo],
) -> int:
    """Populate Voyage AST embeddings (ast.unparse() output)."""
    from tests.similarity.embeddings import (
        get_cached_voyage_ast_embedding,
        get_embeddings_batch_voyage,
    )

    api_key = _load_api_key("VOYAGE_API_KEY", ("your_", "pa-xxx"))
    if not api_key:
        print("VOYAGE_API_KEY not set or invalid, skipping Voyage-AST")
        return 0

    uncached = [f for f in functions if get_cached_voyage_ast_embedding(baselines, f.hash) is None]
    if not uncached:
        print("Voyage-AST: all functions already cached")
        return 0

    print(f"Voyage-AST: fetching embeddings for {len(uncached)} functions...")
    try:
        texts = [f.ast_text for f in uncached]
        embeddings = get_embeddings_batch_voyage(texts)
        for func, emb in zip(uncached, embeddings, strict=True):
            baselines.setdefault("voyage_ast_embeddings", {})[func.hash] = emb
        print(f"Voyage-AST: cached {len(uncached)} new embeddings")
        return len(uncached)
    except Exception as e:
        print(f"Voyage-AST: failed to fetch embeddings: {e}")
        return 0


def _populate_combined(
    baselines: dict,
    functions: list[FunctionInfo],
) -> int:
    """Populate Combined embeddings from all 6 base providers."""
    from tests.similarity.embeddings import (
        compute_combined_embedding,
        get_cached_combined_embedding,
    )

    uncached = [
        f for f in functions if get_cached_combined_embedding(baselines, f.text_hash) is None
    ]
    if not uncached:
        print("Combined: all functions already cached")
        return 0

    computed = 0
    missing_base = []
    for func in uncached:
        # Text variants keyed by text_hash
        openai_text = baselines.get("openai_text_embeddings", {}).get(func.text_hash)
        codestral_text = baselines.get("codestral_text_embeddings", {}).get(func.text_hash)
        voyage_text = baselines.get("voyage_text_embeddings", {}).get(func.text_hash)
        # AST variants keyed by hash
        openai_ast = baselines.get("openai_ast_embeddings", {}).get(func.hash)
        codestral_ast = baselines.get("codestral_ast_embeddings", {}).get(func.hash)
        voyage_ast = baselines.get("voyage_ast_embeddings", {}).get(func.hash)

        if all([openai_text, openai_ast, codestral_text, codestral_ast, voyage_text, voyage_ast]):
            combined = compute_combined_embedding(
                openai_text,
                openai_ast,
                codestral_text,
                codestral_ast,
                voyage_text,
                voyage_ast,
            )
            baselines.setdefault("combined_embeddings", {})[func.text_hash] = combined
            computed += 1
        else:
            missing = []
            if not openai_text:
                missing.append("OpenAI-Text")
            if not openai_ast:
                missing.append("OpenAI-AST")
            if not codestral_text:
                missing.append("Codestral-Text")
            if not codestral_ast:
                missing.append("Codestral-AST")
            if not voyage_text:
                missing.append("Voyage-Text")
            if not voyage_ast:
                missing.append("Voyage-AST")
            missing_base.append((func.name, missing))

    if computed:
        print(f"Combined: computed {computed} new embeddings from base providers")
    if missing_base:
        print(f"Combined: {len(missing_base)} functions missing base provider embeddings")
        for name, missing in missing_base[:5]:
            print(f"  - {name}: missing {', '.join(missing)}")
        if len(missing_base) > 5:
            print(f"  ... and {len(missing_base) - 5} more")

    return computed


def _update_function_hashes(baselines: dict, functions: list[FunctionInfo]) -> None:
    """Update function_hashes with the new format containing both hash and text_hash."""
    function_hashes = baselines.setdefault("function_hashes", {})

    for func in functions:
        key = f"{func.file}:{func.name}:{func.start_line}"
        function_hashes[key] = {"hash": func.hash, "text_hash": func.text_hash}


# Map provider names to functions
_PROVIDER_MAP = {
    "openai-text": _populate_openai_text,
    "openai-ast": _populate_openai_ast,
    "codestral-text": _populate_codestral_text,
    "codestral-ast": _populate_codestral_ast,
    "voyage-text": _populate_voyage_text,
    "voyage-ast": _populate_voyage_ast,
    "combined": _populate_combined,
}

_ALL_PROVIDERS = [
    "openai-text",
    "openai-ast",
    "codestral-text",
    "codestral-ast",
    "voyage-text",
    "voyage-ast",
    "combined",
]


def populate_embeddings(provider: str = "all") -> None:
    """Populate embedding caches for specified provider(s).

    Args:
        provider: One of the 7 providers or 'all'
    """
    from tests.similarity.ast_helpers import extract_all_function_infos
    from tests.similarity.storage import load_baselines, save_baselines

    print("Loading baselines and extracting functions...")  # noqa: T201
    baselines = load_baselines()
    functions = extract_all_function_infos(min_loc=1)
    print(f"Found {len(functions)} functions in codebase")

    # Always update function_hashes with new format (hash + text_hash)
    _update_function_hashes(baselines, functions)

    total_new = 0
    providers = _ALL_PROVIDERS if provider == "all" else [provider]

    for prov in providers:
        if prov in _PROVIDER_MAP:
            total_new += _PROVIDER_MAP[prov](baselines, functions)
        else:
            print(f"Unknown provider: {prov}")
            continue

    if total_new > 0:
        print(f"\nSaving {total_new} new embeddings to cache...")
        save_baselines(baselines)
        print("Done!")
    else:
        print("\nNo new embeddings to save.")
        # Still save to update function_hashes format
        save_baselines(baselines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Populate embedding caches for similarity tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--provider",
        choices=[*_ALL_PROVIDERS, "all"],
        default="all",
        help="Which provider to populate (default: all)",
    )
    args = parser.parse_args()

    try:
        populate_embeddings(args.provider)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)


if __name__ == "__main__":
    main()

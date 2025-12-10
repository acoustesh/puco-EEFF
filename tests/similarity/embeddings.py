"""Embedding API clients for OpenAI, Codestral, and Voyage."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from tests.similarity.constants import (
    CODESTRAL_EMBED_MODEL,
    OPENROUTER_API_URL,
    VOYAGE_CODE_MODEL,
)

# Exponential backoff delays
_RETRY_DELAYS = [1.0, 2.0, 4.0]


def _load_api_key_from_env(env_var: str) -> str | None:
    """Load API key from environment or .env file."""
    api_key = os.environ.get(env_var)
    if api_key:
        return api_key

    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        prefix = f"{env_var}="
        with env_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(prefix):
                    return line.split("=", 1)[1].strip()
    return None


def get_embeddings_batch(
    texts: list[str],
    model: str = "text-embedding-3-large",
    max_retries: int = 3,
    timeout: float = 30.0,
) -> list[list[float]]:
    """Get embeddings using OpenAI API."""
    import openai

    client = openai.OpenAI(timeout=timeout)

    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(input=texts, model=model)
            return [item.embedding for item in response.data]
        except (openai.RateLimitError, openai.APITimeoutError):
            if attempt < max_retries - 1:
                time.sleep(_RETRY_DELAYS[attempt])
            else:
                raise

    return []


def get_embeddings_batch_codestral(
    texts: list[str],
    model: str = CODESTRAL_EMBED_MODEL,
    max_retries: int = 3,
    timeout: float = 120.0,
) -> list[list[float]]:
    """Get embeddings using OpenRouter Codestral Embed API."""
    import requests

    api_key = _load_api_key_from_env("OPENROUTER_API_KEY")
    if not api_key:
        msg = "OPENROUTER_API_KEY not found in environment or .env file"
        raise ValueError(msg)

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(max_retries):
        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json={"model": model, "input": texts},
                timeout=timeout,
            )

            if response.status_code == 429 and attempt < max_retries - 1:
                time.sleep(_RETRY_DELAYS[attempt])
                continue

            response.raise_for_status()
            data = response.json()
            if "data" not in data:
                msg = f"Unexpected API response: {data}"
                raise ValueError(msg)
            return [item["embedding"] for item in data["data"]]

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(_RETRY_DELAYS[attempt])
            else:
                raise

    return []


def get_embeddings_batch_voyage(
    texts: list[str],
    model: str = VOYAGE_CODE_MODEL,
    max_retries: int = 3,
    timeout: float = 120.0,
) -> list[list[float]]:
    """Get embeddings using Voyage AI API."""
    import voyageai

    api_key = _load_api_key_from_env("VOYAGE_API_KEY")
    if not api_key:
        msg = "VOYAGE_API_KEY not found in environment or .env file"
        raise ValueError(msg)

    client = voyageai.Client(api_key=api_key, timeout=timeout)

    for attempt in range(max_retries):
        try:
            result = client.embed(texts, model=model, input_type="document")
            return [list(map(float, emb)) for emb in result.embeddings]

        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = "rate" in error_str and "limit" in error_str
            if (is_rate_limit or attempt < max_retries - 1) and attempt < max_retries - 1:
                time.sleep(_RETRY_DELAYS[attempt])
                continue
            raise

    return []


def compute_cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)

    dot_product = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def compute_combined_embedding(
    openai_emb: list[float],
    codestral_emb: list[float],
    voyage_emb: list[float],
) -> list[float]:
    """Compute combined embedding by concatenating all three providers with L2 normalization."""
    combined = openai_emb + codestral_emb + voyage_emb

    # L2 normalize the combined vector
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = (np.array(combined) / norm).tolist()

    return combined


def get_cached_embedding(baselines: dict, content_hash: str, cache_key: str) -> list[float] | None:
    """Return cached embedding for any provider."""
    return baselines.get(cache_key, {}).get(content_hash)


# Convenience aliases for backward compatibility with ProviderConfig.get_cached_fn
def get_cached_openai_embedding(baselines: dict, content_hash: str) -> list[float] | None:
    """Return cached OpenAI embedding."""
    return get_cached_embedding(baselines, content_hash, "embeddings")


def get_cached_codestral_embedding(baselines: dict, content_hash: str) -> list[float] | None:
    """Return cached Codestral embedding."""
    return get_cached_embedding(baselines, content_hash, "codestral_embeddings")


def get_cached_voyage_embedding(baselines: dict, content_hash: str) -> list[float] | None:
    """Return cached Voyage embedding."""
    return get_cached_embedding(baselines, content_hash, "voyage_embeddings")


def get_cached_combined_embedding(baselines: dict, content_hash: str) -> list[float] | None:
    """Return cached Combined embedding."""
    return get_cached_embedding(baselines, content_hash, "combined_embeddings")

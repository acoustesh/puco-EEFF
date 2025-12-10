"""Constants for similarity detection tests."""

from __future__ import annotations

from pathlib import Path

# Base directories
PUCO_EEFF_DIR = Path(__file__).parent.parent.parent / "puco_eeff"
EXTRACTOR_DIR = PUCO_EEFF_DIR / "extractor"

# All subdirectories under puco_eeff (for tests that span all)
PUCO_EEFF_SUBDIRS = ["extractor", "scraper", "sheets", "transformer", "writer"]

# Baseline files
BASELINES_DIR = Path(__file__).parent.parent / "baselines"
BASELINES_FILE = BASELINES_DIR / "similarity_baselines.json"
FUNCTION_HASHES_FILE = BASELINES_DIR / "function_hashes.json"
EMBEDDINGS_FILE = BASELINES_DIR / "embeddings_cache.json.zlib"
CODESTRAL_EMBEDDINGS_FILE = BASELINES_DIR / "codestral_embeddings_cache.json.zlib"
VOYAGE_EMBEDDINGS_FILE = BASELINES_DIR / "voyage_embeddings_cache.json.zlib"
COMBINED_EMBEDDINGS_FILE = BASELINES_DIR / "combined_embeddings_cache.json.zlib"

# OpenRouter config
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/embeddings"
CODESTRAL_EMBED_MODEL = "mistralai/codestral-embed-2505"

# Voyage AI config
VOYAGE_CODE_MODEL = "voyage-code-3"

# Refactor index defaults (can be overridden in config)
DEFAULT_REFACTOR_INDEX_THRESHOLD = 15.0
DEFAULT_REFACTOR_INDEX_TOP_N = 5

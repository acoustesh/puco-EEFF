"""Code Similarity Detection package for puco_eeff/.

This package detects near-duplicate functions using embedding providers:
- Pairwise similarity detection (finds copy-paste code)
- Neighbor clustering (finds functions with multiple similar counterparts)
- Refactor index computation (combines complexity + similarity for priority)

Modules
-------
    types: FunctionInfo dataclass
    storage: Embedding compression/decompression and cache management
    ast_helpers: Function extraction from Python AST
    pca: PCA dimensionality reduction and clustering
    embeddings: Embedding API clients (OpenAI, Codestral, Voyage)
    refactor_index: Refactor index computation
    complexity: Complexity map loaders (CC, COG)
    checks: Unified similarity check workflow
"""

from tests.similarity.ast_helpers import (
    extract_all_function_infos,
    extract_function_infos,
    extract_function_infos_from_file,
)
from tests.similarity.checks import (
    CODESTRAL_AST_PROVIDER,
    CODESTRAL_TEXT_PROVIDER,
    COMBINED_PROVIDER,
    OPENAI_AST_PROVIDER,
    OPENAI_TEXT_PROVIDER,
    VOYAGE_AST_PROVIDER,
    VOYAGE_TEXT_PROVIDER,
    run_provider_similarity_checks,
)
from tests.similarity.complexity import _load_all_complexity_maps, _load_complexity_maps
from tests.similarity.embeddings import (
    compute_combined_embedding,
    compute_cosine_similarity,
    get_cached_codestral_ast_embedding,
    get_cached_codestral_text_embedding,
    get_cached_combined_embedding,
    get_cached_openai_ast_embedding,
    get_cached_openai_text_embedding,
    get_cached_voyage_ast_embedding,
    get_cached_voyage_text_embedding,
    get_embeddings_batch,
    get_embeddings_batch_codestral,
    get_embeddings_batch_voyage,
)
from tests.similarity.file_split import generate_file_split_proposal
from tests.similarity.pca import (
    apply_pca_to_functions,
    cluster_functions_kmeans_with_pca,
    fit_pca,
    transform_embeddings_with_pca,
)
from tests.similarity.refactor_index import (
    compute_max_similarities,
    compute_refactor_indices,
    compute_similarity_indices,
    compute_similarity_matrix,
    get_refactor_priority_message,
    get_refactor_priority_message_for_complexity,
)
from tests.similarity.storage import (
    count_missing_embeddings,
    get_valid_hashes,
    load_baselines,
    rebuild_combined_embeddings,
    remove_orphan_embeddings,
    save_baselines,
    verify_cache_counts,
)
from tests.similarity.types import FunctionInfo

__all__ = [
    # Types
    "FunctionInfo",
    # Storage
    "load_baselines",
    "save_baselines",
    "get_valid_hashes",
    "count_missing_embeddings",
    "remove_orphan_embeddings",
    "rebuild_combined_embeddings",
    "verify_cache_counts",
    # AST helpers
    "extract_function_infos",
    "extract_all_function_infos",
    "extract_function_infos_from_file",
    # PCA
    "fit_pca",
    "transform_embeddings_with_pca",
    "apply_pca_to_functions",
    "cluster_functions_kmeans_with_pca",
    # Embeddings
    "get_embeddings_batch",
    "get_embeddings_batch_codestral",
    "get_embeddings_batch_voyage",
    "compute_cosine_similarity",
    "compute_combined_embedding",
    "get_cached_openai_text_embedding",
    "get_cached_openai_ast_embedding",
    "get_cached_codestral_text_embedding",
    "get_cached_codestral_ast_embedding",
    "get_cached_voyage_text_embedding",
    "get_cached_voyage_ast_embedding",
    "get_cached_combined_embedding",
    # Refactor index
    "compute_similarity_matrix",
    "compute_max_similarities",
    "compute_similarity_indices",
    "compute_refactor_indices",
    "get_refactor_priority_message",
    "get_refactor_priority_message_for_complexity",
    # Complexity
    "_load_complexity_maps",
    "_load_all_complexity_maps",
    # Checks (7 providers)
    "run_provider_similarity_checks",
    "OPENAI_TEXT_PROVIDER",
    "OPENAI_AST_PROVIDER",
    "CODESTRAL_TEXT_PROVIDER",
    "CODESTRAL_AST_PROVIDER",
    "VOYAGE_TEXT_PROVIDER",
    "VOYAGE_AST_PROVIDER",
    "COMBINED_PROVIDER",
    # File split
    "generate_file_split_proposal",
]

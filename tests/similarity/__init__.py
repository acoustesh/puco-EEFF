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
    CODESTRAL_PROVIDER,
    COMBINED_PROVIDER,
    OPENAI_PROVIDER,
    VOYAGE_PROVIDER,
    run_provider_similarity_checks,
)
from tests.similarity.complexity import _load_all_complexity_maps, _load_complexity_maps
from tests.similarity.embeddings import (
    compute_combined_embedding,
    compute_cosine_similarity,
    get_cached_codestral_embedding,
    get_cached_combined_embedding,
    get_cached_embedding,
    get_cached_voyage_embedding,
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
from tests.similarity.storage import load_baselines, save_baselines
from tests.similarity.types import FunctionInfo

__all__ = [
    # Types
    "FunctionInfo",
    # Storage
    "load_baselines",
    "save_baselines",
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
    "get_cached_embedding",
    "get_cached_codestral_embedding",
    "get_cached_voyage_embedding",
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
    # Checks
    "run_provider_similarity_checks",
    "OPENAI_PROVIDER",
    "CODESTRAL_PROVIDER",
    "VOYAGE_PROVIDER",
    "COMBINED_PROVIDER",
    # File split
    "generate_file_split_proposal",
]

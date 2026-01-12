"""
Retrieval Factory

Factory pattern for creating retrievers based on RETRIEVAL_MODE environment variable.
Supports A/B testing between flat and hierarchical retrieval modes.
"""

import logging
import os
from typing import Optional

from azure.search.documents import SearchClient

from .retrieve import Retriever
from .hierarchical_retriever import HierarchicalAzureRetriever

logger = logging.getLogger(__name__)

def create_retriever(
    search_client: SearchClient,
    embeddings_client,
    search_type: str = "hybrid",
    retrieval_mode: Optional[str] = None
) -> Retriever:
    """
    Create retriever based on RETRIEVAL_MODE environment variable or explicit parameter.
    
    Args:
        search_client: Azure Search client
        embeddings_client: OpenAI embeddings client
        search_type: Type of search ("vector", "hybrid")
        retrieval_mode: Override for retrieval mode ("flat", "hierarchical", "adaptive")
                       If None, reads from RETRIEVAL_MODE env var, defaults to "flat"
    
    Returns:
        Configured retriever instance
    """
    # Determine retrieval mode
    mode = retrieval_mode or os.getenv("RETRIEVAL_MODE", "flat").lower()
    
    logger.info(f"Creating retriever with mode: {mode}, search_type: {search_type}")
    
    if mode == "flat":
        # Standard flat retriever for backward compatibility
        return Retriever(
            search_client=search_client,
            embeddings_client=embeddings_client,
            search_type=search_type
        )
    
    elif mode == "hierarchical":
        # Optimized hierarchical retriever with all performance enhancements
        return HierarchicalAzureRetriever(
            search_client=search_client,
            embeddings_client=embeddings_client,
            search_type=search_type,
            max_parent_ids_per_lookup=int(os.getenv("MAX_PARENT_IDS_PER_LOOKUP", "25")),
            cache_ttl_seconds=int(os.getenv("PARENT_CACHE_TTL_SECONDS", "600")),
            cache_max_size=int(os.getenv("PARENT_CACHE_MAX_SIZE", "5000")),
            parent_lookup_timeout=float(os.getenv("PARENT_LOOKUP_TIMEOUT", "10.0")),
            max_parents_total=int(os.getenv("MAX_PARENTS_TOTAL", "8")),
            final_top_k=int(os.getenv("FINAL_TOP_K")) if os.getenv("FINAL_TOP_K") else None
        )
    
    elif mode == "adaptive":
        # Future: Adaptive mode that switches based on query characteristics
        # For now, default to hierarchical
        logger.warning("Adaptive mode not yet implemented, using hierarchical")
        return create_retriever(
            search_client=search_client,
            embeddings_client=embeddings_client,
            search_type=search_type,
            retrieval_mode="hierarchical"
        )
    
    else:
        logger.warning(f"Unknown retrieval mode '{mode}', falling back to flat")
        return create_retriever(
            search_client=search_client,
            embeddings_client=embeddings_client,
            search_type=search_type,
            retrieval_mode="flat"
        )

def get_supported_modes() -> list[str]:
    """Get list of supported retrieval modes."""
    return ["flat", "hierarchical", "adaptive"]

def is_hierarchical_mode(retrieval_mode: Optional[str] = None) -> bool:
    """Check if current or specified mode uses hierarchical retrieval."""
    mode = retrieval_mode or os.getenv("RETRIEVAL_MODE", "flat").lower()
    return mode in ["hierarchical", "adaptive"]
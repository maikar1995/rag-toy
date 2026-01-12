"""
Hierarchical Azure Retriever

Optimized implementation with:
- Parent lookup batching for performance
- LRU cache for frequently accessed parents
- Timeout and resilience handling with graceful degradation
- Backward compatibility with non-hierarchical chunks
"""

import logging
from typing import List, Dict, Any, Optional, Union, Set
from dataclasses import dataclass
from functools import lru_cache
import time
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from azure.search.documents import SearchClient
from azure.search.documents.models import SearchResults

from .retrieve import Retriever, SearchResult
from .hierarchical_utils import (
    get_direct_parents, 
    build_parent_lookup_filter,
    deduplicate_by_node_id
)

logger = logging.getLogger(__name__)

@dataclass
class CachedParent:
    """Cache entry for parent document with timestamp."""
    document: Dict[str, Any]
    timestamp: float

class HierarchicalAzureRetriever(Retriever):
    """
    Hierarchical retriever with optimizations:
    - Parent lookup batching (max 25 node_ids per query)
    - LRU cache for parent documents (TTL 10 minutes)
    - Graceful degradation on failures
    - Backward compatibility with flat chunks
    """
    
    def __init__(
        self, 
        search_client: SearchClient, 
        embeddings_client, 
        search_type: str = "hybrid",
        max_parent_ids_per_lookup: int = 25,
        cache_ttl_seconds: int = 600,  # 10 minutes
        cache_max_size: int = 5000,
        parent_lookup_timeout: float = 10.0,  # seconds
        max_parents_total: int = 8,
        final_top_k: Optional[int] = None
    ):
        super().__init__(search_client, embeddings_client, search_type)
        self.max_parent_ids_per_lookup = max_parent_ids_per_lookup
        self.cache_ttl_seconds = cache_ttl_seconds
        self.cache_max_size = cache_max_size
        self.parent_lookup_timeout = parent_lookup_timeout
        self.max_parents_total = max_parents_total
        self.final_top_k = final_top_k
        
        # In-memory cache for parent documents
        self._parent_cache: Dict[str, CachedParent] = {}
        
    def _is_cache_valid(self, cached_parent: CachedParent) -> bool:
        """Check if cached parent is still valid based on TTL."""
        return time.time() - cached_parent.timestamp < self.cache_ttl_seconds
    
    def _get_cached_parent(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get parent from cache if valid."""
        if node_id in self._parent_cache:
            cached = self._parent_cache[node_id]
            if self._is_cache_valid(cached):
                return cached.document
            else:
                # Remove expired entry
                del self._parent_cache[node_id]
        return None
    
    def _cache_parent(self, node_id: str, document: Dict[str, Any]) -> None:
        """Cache parent document with LRU eviction."""
        # Simple LRU: if at max size, remove oldest entry
        if len(self._parent_cache) >= self.cache_max_size:
            oldest_key = min(
                self._parent_cache.keys(), 
                key=lambda k: self._parent_cache[k].timestamp
            )
            del self._parent_cache[oldest_key]
        
        self._parent_cache[node_id] = CachedParent(
            document=document,
            timestamp=time.time()
        )
    
    def _lookup_parents_batch(
        self, 
        parent_node_ids: Set[str], 
        doc_filters: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Lookup parent documents in batches with caching and timeout.
        
        Args:
            parent_node_ids: Set of parent node IDs to lookup
            doc_filters: Additional document-level filters
            
        Returns:
            List of parent documents found
        """
        if not parent_node_ids:
            return []
        
        # Check cache first
        uncached_ids = set()
        cached_parents = []
        
        for node_id in parent_node_ids:
            cached = self._get_cached_parent(node_id)
            if cached:
                cached_parents.append(cached)
                logger.debug(f"Cache hit for parent node_id: {node_id}")
            else:
                uncached_ids.add(node_id)
        
        if not uncached_ids:
            logger.debug(f"All {len(parent_node_ids)} parents found in cache")
            return cached_parents
        
        # Lookup uncached parents in batches
        fetched_parents = []
        uncached_list = list(uncached_ids)
        
        try:
            for i in range(0, len(uncached_list), self.max_parent_ids_per_lookup):
                batch_ids = uncached_list[i:i + self.max_parent_ids_per_lookup]
                logger.debug(f"Looking up parent batch {i//self.max_parent_ids_per_lookup + 1}: {len(batch_ids)} ids")
                
                # Build filter for this batch
                filter_query = build_parent_lookup_filter(batch_ids, doc_filters)
                
                # Execute search with timeout
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self.search_client.search,
                        search_text="*",
                        filter=filter_query,
                        select=["id", "content", "doc_id", "page", "source_uri", "content_type", "node_id", "node_type", "parent_path"],
                        top=len(batch_ids) * 2  # Allow some buffer for duplicates
                    )
                    
                    try:
                        search_results = future.result(timeout=self.parent_lookup_timeout)
                        
                        batch_parents = []
                        for result in search_results:
                            doc = {key: result.get(key) for key in result.keys()}
                            batch_parents.append(doc)
                            
                            # Cache the parent
                            if doc.get('node_id'):
                                self._cache_parent(doc['node_id'], doc)
                        
                        fetched_parents.extend(batch_parents)
                        logger.debug(f"Fetched {len(batch_parents)} parents in batch")
                        
                    except FutureTimeoutError:
                        logger.warning(f"Parent lookup batch timed out after {self.parent_lookup_timeout}s")
                        # Continue with next batch rather than failing completely
                        continue
                        
        except Exception as e:
            logger.error(f"Error during parent lookup: {e}")
            # Graceful degradation: continue with cached results only
        
        all_parents = cached_parents + fetched_parents
        logger.info(f"Parent lookup: {len(cached_parents)} cached + {len(fetched_parents)} fetched = {len(all_parents)} total")
        
        return all_parents
    
    def search(
        self, 
        query: Union[str, List[float]], 
        top_k: int = 10, 
        filters: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Hierarchical search with two-stage retrieval:
        1. Find leaf chunks matching the query
        2. Lookup their parent documents with score inheritance
        
        Includes optimizations and graceful degradation.
        """
        try:
            # Stage 1: Get leaf chunks (same as base retriever)
            logger.debug(f"Stage 1: Searching for {top_k} leaf chunks")
            leaf_results = super().search(query, top_k, filters)
            
            if not leaf_results:
                logger.debug("No leaf results found")
                return []
            
            # Extract parent node IDs from leaf results that have hierarchical metadata
            parent_node_ids = set()
            hierarchical_leaves = []
            non_hierarchical_leaves = []
            
            for result in leaf_results:
                if result.parent_path and result.node_id:
                    # This is a hierarchical chunk
                    parents = get_direct_parents(result.parent_path)
                    parent_node_ids.update(parents)
                    hierarchical_leaves.append(result)
                else:
                    # Non-hierarchical chunk - keep as-is for backward compatibility
                    non_hierarchical_leaves.append(result)
                    logger.debug(f"Non-hierarchical chunk: {result.chunk_id}")
            
            logger.debug(f"Found {len(hierarchical_leaves)} hierarchical + {len(non_hierarchical_leaves)} non-hierarchical leaves")
            
            if not parent_node_ids:
                logger.debug("No parent nodes to lookup - returning flat results")
                return leaf_results
            
            # Limit parent lookup to avoid explosion
            if len(parent_node_ids) > self.max_parents_total * 3:  # Some buffer
                sorted_leaves = sorted(hierarchical_leaves, key=lambda x: x.score, reverse=True)
                limited_leaves = sorted_leaves[:self.max_parents_total]
                
                parent_node_ids = set()
                for result in limited_leaves:
                    parents = get_direct_parents(result.parent_path)
                    parent_node_ids.update(parents)
                
                hierarchical_leaves = limited_leaves
                logger.debug(f"Limited parent lookup to top {len(limited_leaves)} hierarchical leaves")
            
            # Stage 2: Lookup parent documents
            logger.debug(f"Stage 2: Looking up {len(parent_node_ids)} unique parent nodes")
            
            # Separate document-level vs search-level filters
            doc_filters = [f for f in (filters or []) if 'node_type' in f or 'doc_id' in f]
            
            try:
                parent_docs = self._lookup_parents_batch(parent_node_ids, doc_filters)
                logger.debug(f"Retrieved {len(parent_docs)} parent documents")
                
                # Convert parents to SearchResult objects
                parent_results = []
                for doc in parent_docs:
                    parent_result = SearchResult(
                        chunk_id=doc.get('id', ''),
                        content=doc.get('content', ''),
                        score=0.0,  # Will inherit max child score
                        doc_id=doc.get('doc_id', ''),
                        page=doc.get('page'),
                        source_uri=doc.get('source_uri', ''),
                        content_type=doc.get('content_type', 'text'),
                        search_type=f"{self.search_type}_hierarchical",
                        node_id=doc.get('node_id'),
                        node_type=doc.get('node_type', 'parent'),
                        parent_path=doc.get('parent_path')
                    )
                    parent_results.append(parent_result)
                
            except Exception as e:
                logger.error(f"Parent lookup failed: {e}, falling back to flat results")
                parent_results = []
            
            # Combine and deduplicate results
            all_results = hierarchical_leaves + parent_results + non_hierarchical_leaves
            
            # Deduplicate by node_id and inherit max child scores to parents  
            final_results = deduplicate_by_node_id(all_results)
            
            # Apply final top_k limit if specified
            if self.final_top_k:
                final_results = final_results[:self.final_top_k]
            
            logger.info(f"Hierarchical search complete: {len(leaf_results)} leaves → {len(final_results)} final results")
            return final_results
            
        except Exception as e:
            logger.error(f"Hierarchical search failed: {e}, falling back to flat search")
            # Graceful degradation: return flat results
            try:
                return super().search(query, top_k, filters)
            except Exception as fallback_error:
                logger.error(f"Fallback flat search also failed: {fallback_error}")
                return []
    
    def clear_cache(self) -> None:
        """Clear the parent document cache."""
        self._parent_cache.clear()
        logger.info("Parent cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        valid_entries = sum(1 for cached in self._parent_cache.values() if self._is_cache_valid(cached))
        return {
            "total_entries": len(self._parent_cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self._parent_cache) - valid_entries,
            "hit_rate": getattr(self, '_cache_hits', 0) / max(getattr(self, '_cache_requests', 1), 1)
        }
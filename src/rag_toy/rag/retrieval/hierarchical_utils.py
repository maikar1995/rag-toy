"""
Hierarchical utilities for parent path serialization and retrieval
"""
from typing import Optional, List, Union, Any


def serialize_parent_path(parent_ids: Any) -> Optional[str]:
    """
    Serialize parent IDs to path string for Azure Search.
    
    Args:
        parent_ids: Parent IDs (list, string, or None)
        
    Returns:
        Serialized path string like "a>b>c" or None
    """
    if not parent_ids:
        return None
    if isinstance(parent_ids, list):
        return ">".join([str(p) for p in parent_ids if p])
    return str(parent_ids)


def extract_direct_parent(parent_path: str) -> Optional[str]:
    """
    Extract direct parent ID from path.
    
    Args:
        parent_path: Path string like "a>b>c"
        
    Returns:
        Direct parent ID (last in path) or None
    """
    if not parent_path:
        return None
    parts = parent_path.split(">")
    return parts[-1] if parts else None


def extract_all_parents(parent_path: str) -> List[str]:
    """
    Extract all parent IDs from path.
    
    Args:
        parent_path: Path string like "a>b>c"
        
    Returns:
        List of all parent IDs
    """
    if not parent_path:
        return []
    return [p for p in parent_path.split(">") if p]


def build_parent_lookup_filter(parent_ids: List[str]) -> str:
    """
    Build OData filter for parent lookup by node_id.
    
    Args:
        parent_ids: List of parent node IDs to lookup
        
    Returns:
        OData filter string for Azure Search
    """
    if not parent_ids:
        return ""
    
    # Escape single quotes in IDs and build OR conditions
    escaped_ids = [id.replace("'", "''") for id in parent_ids]
    conditions = [f"node_id eq '{id}'" for id in escaped_ids]
    
    # Add node_type filter to only get parents
    parent_filter = f"node_type eq 'parent'"
    id_filter = f"({' or '.join(conditions)})"
    
    return f"{parent_filter} and {id_filter}"


def separate_doc_and_page_filters(filters: dict) -> tuple[dict, dict]:
    """
    Separate document-level filters from page-specific filters.
    
    Args:
        filters: Original filter dictionary
        
    Returns:
        Tuple of (doc_filters, page_filters)
    """
    if not filters:
        return {}, {}
    
    doc_level_keys = {"doc_id", "content_type", "source_uri", "node_type"}
    page_level_keys = {"page"}
    
    doc_filters = {k: v for k, v in filters.items() if k in doc_level_keys}
    page_filters = {k: v for k, v in filters.items() if k in page_level_keys}
    
    return doc_filters, page_filters


def deduplicate_by_node_id(results: List[Any], score_key: str = "score") -> List[Any]:
    """
    Deduplicate results by node_id, keeping the one with highest score.
    
    Args:
        results: List of results (SearchResult objects or dicts)
        score_key: Key/attribute name for score
        
    Returns:
        Deduplicated list
    """
    seen = {}
    
    for result in results:
        # Get node_id (try both attribute and dict access)
        if hasattr(result, 'node_id'):
            node_id = result.node_id
            score = getattr(result, score_key, 0.0)
        else:
            node_id = result.get('node_id')
            score = result.get(score_key, 0.0)
        
        if not node_id:
            # Fallback to chunk_id if no node_id
            if hasattr(result, 'chunk_id'):
                node_id = result.chunk_id
            else:
                node_id = result.get('chunk_id', str(id(result)))
        
        # Keep result with highest score
        if node_id not in seen or score > getattr(seen[node_id], score_key, seen[node_id].get(score_key, 0.0)):
            seen[node_id] = result
    
    return list(seen.values())

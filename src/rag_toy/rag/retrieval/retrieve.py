"""
RAG Retrieval Module

Provides vector and hybrid search capabilities for Azure AI Search.
Supports both text queries (with automatic embedding) and pre-computed vectors.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import os
from dotenv import load_dotenv

from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery

from rag_toy.rag.interfaces import Retriever

# Load environment variables
load_dotenv()


@dataclass 
class SearchResult:
    """Structured search result."""
    chunk_id: str
    content: str
    score: float
    doc_id: str
    page: Optional[int]
    source_uri: str
    content_type: str
    search_type: str
    # Hierarchical fields (optional for backward compatibility)
    node_id: Optional[str] = None
    node_type: Optional[str] = None
    parent_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "doc_id": self.doc_id,
            "page": self.page,
            "source_uri": self.source_uri,
            "content_type": self.content_type,
            "search_type": self.search_type
        }
        
        # Add hierarchical fields if present
        if self.node_id is not None:
            result["node_id"] = self.node_id
        if self.node_type is not None:
            result["node_type"] = self.node_type
        if self.parent_path is not None:
            result["parent_path"] = self.parent_path
            
        return result


class Retriever:
    """
    RAG Retriever with vector and hybrid search capabilities.
    
    Supports:
    - Vector-only search
    - Hybrid search (vector + keyword with RRF)
    - Flexible filtering with OData
    - Query embedding with retry logic
    """
    
    def __init__(
        self,
        search_client: Optional[SearchClient] = None,
        openai_client: Optional[AzureOpenAI] = None,
        index_name: Optional[str] = None,
        default_top_k: int = 5,
        default_search_type: str = "hybrid"
    ):
        """
        Initialize the retriever.
        
        Args:
            search_client: Azure Search client (optional, will create if None)
            openai_client: Azure OpenAI client (optional, will create if None)
            index_name: Search index name (optional, will load from env)
            default_top_k: Default number of results to return
            default_search_type: Default search strategy ("vector" or "hybrid")
        """
        self.default_top_k = default_top_k
        self.default_search_type = default_search_type
        
        # Initialize clients
        self.search_client = search_client or self._create_search_client(index_name)
        self.openai_client = openai_client or self._create_openai_client()
        
        # Get deployment/model info
        self.embedding_deployment = os.getenv('EMBEDDING_1_DEPLOYMENT') or os.getenv('EMBEDDING_1_MODEL_NAME')
        if not self.embedding_deployment:
            raise ValueError("Missing EMBEDDING_1_DEPLOYMENT or EMBEDDING_1_MODEL_NAME in environment")
        
        logging.info(f"✅ Retriever initialized: index={self.search_client._index_name}, "
                    f"deployment={self.embedding_deployment}, default_search_type={default_search_type}")
    
    def _create_search_client(self, index_name: Optional[str] = None) -> SearchClient:
        """Create Azure Search client from environment config."""
        endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')
        api_key = os.getenv('AZURE_SEARCH_KEY')  # Using query key for read operations
        index_name = index_name or os.getenv('AZURE_SEARCH_INDEX_NAME', 'rag-toy-index-v1')
        
        if not endpoint or not api_key:
            raise ValueError("Missing AZURE_SEARCH_ENDPOINT or AZURE_SEARCH_KEY in environment")
        
        return SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(api_key)
        )
    
    def _create_openai_client(self) -> AzureOpenAI:
        """Create Azure OpenAI client from environment config."""
        endpoint = os.getenv('EMBEDDING_1_ENDPOINT')
        api_key = os.getenv('EMBEDDING_1_API_KEY')
        api_version = os.getenv('EMBEDDING_1_API_VERSION', '2024-02-01')
        
        if not endpoint or not api_key:
            raise ValueError("Missing EMBEDDING_1_ENDPOINT or EMBEDDING_1_API_KEY in environment")
        
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )
    
    def _generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for query text with retry logic."""
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_deployment,
                input=[query]
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"Failed to generate query embedding: {e}")
            raise
    
    def _build_odata_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        Convert filter dictionary to OData filter string.
        
        Supports:
        - doc_id: str
        - page: int or List[int] 
        - content_type: str
        
        Args:
            filters: Dictionary of filter conditions
            
        Returns:
            OData filter string or None
        """
        if not filters:
            return None
        
        filter_parts = []
        
        for key, value in filters.items():
            if key == "doc_id" and isinstance(value, str):
                filter_parts.append(f"doc_id eq '{value}'")
            
            elif key == "page":
                if isinstance(value, int):
                    filter_parts.append(f"page eq {value}")
                elif isinstance(value, list):
                    page_conditions = [f"page eq {p}" for p in value if isinstance(p, int)]
                    if page_conditions:
                        filter_parts.append(f"({' or '.join(page_conditions)})")
            
            elif key == "content_type" and isinstance(value, str):
                filter_parts.append(f"content_type eq '{value}'")
            
            else:
                logging.warning(f"Unsupported filter: {key}={value}")
        
        return " and ".join(filter_parts) if filter_parts else None
    
    def _parse_search_results(self, results, search_type: str) -> List[SearchResult]:
        """Parse Azure Search results into structured format."""
        parsed_results = []
        
        for result in results:
            try:
                search_result = SearchResult(
                    chunk_id=result.get('id', ''),
                    content=result.get('content', ''),
                    score=result.get('@search.score', 0.0),
                    doc_id=result.get('doc_id', ''),
                    page=result.get('page'),
                    source_uri=result.get('source_uri', ''),
                    content_type=result.get('content_type', 'text'),
                    search_type=search_type,
                    node_id=result.get('node_id'),
                    node_type=result.get('node_type'),
                    parent_path=result.get('parent_path')
                )
                parsed_results.append(search_result)
            except Exception as e:
                logging.error(f"Failed to parse search result: {e}")
                continue
        
        return parsed_results
    
    def retrieve(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        search_type: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        min_score: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Search for relevant chunks.
        
        Args:
            query: Text query string
            top_k: Number of results to return (default: self.default_top_k)
            filters: Filter conditions (dict with doc_id, page, content_type)
            search_type: "vector" or "hybrid" (default: self.default_search_type)
            query_vector: Pre-computed query vector (optional, will compute if None)
            min_score: Minimum score threshold (optional post-filter)
            
        Returns:
            List of SearchResult objects
        """
        top_k = top_k or self.default_top_k
        search_type = search_type or self.default_search_type
        
        logging.debug(f"🔍 Search: query='{query[:50]}...', top_k={top_k}, type={search_type}")
        
        try:
            # Generate query embedding if not provided
            if query_vector is None:
                query_vector = self._generate_query_embedding(query)
                logging.debug(f"Generated embedding with {len(query_vector)} dimensions")
            
            # Build OData filter
            odata_filter = self._build_odata_filter(filters)
            if odata_filter:
                logging.debug(f"Applied filter: {odata_filter}")
            
            # Perform search based on type
            if search_type == "vector":
                results = self._vector_search(query_vector, top_k, odata_filter)
            elif search_type == "hybrid":
                results = self._hybrid_search(query, query_vector, top_k, odata_filter)
            else:
                raise ValueError(f"Unsupported search_type: {search_type}. Use 'vector' or 'hybrid'")
            
            # Parse results
            parsed_results = self._parse_search_results(results, search_type)
            
            # Apply min_score filter if specified
            if min_score is not None:
                parsed_results = [r for r in parsed_results if r.score >= min_score]
            
            logging.info(f"✅ Found {len(parsed_results)} results (search_type={search_type})")
            return parsed_results
            
        except Exception as e:
            logging.error(f"Search failed: {e}")
            raise
    
    def _vector_search(
        self, 
        query_vector: List[float], 
        top_k: int, 
        odata_filter: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Perform vector-only search."""
        
        vector_query = VectorizedQuery(
            vector=query_vector,
            k=top_k,
            fields="contentVector"
        )
        
        search_results = self.search_client.search(
            search_text="",
            vector_queries=[vector_query],
            select=["id", "content", "doc_id", "page", "source_uri", "content_type", "node_id", "node_type", "parent_path"],
            filter=odata_filter,
            top=top_k
        )
        
        return list(search_results)
    
    def _hybrid_search(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: int,
        odata_filter: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search (vector + keyword with RRF)."""
        
        vector_query = VectorizedQuery(
            vector=query_vector,
            k=top_k,
            fields="contentVector"
        )
        
        search_results = self.search_client.search(
            search_text=query_text,
            vector_queries=[vector_query],
            select=["id", "content", "doc_id", "page", "source_uri", "content_type", "node_id", "node_type", "parent_path"],
            filter=odata_filter,
            top=top_k,
            query_type="semantic",  # Enable semantic search for better keyword matching
            semantic_configuration_name="default"
        )
        
        return list(search_results)
    
    def search_dict(
        self,
        query: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Convenience method that returns results as dictionaries.
        
        Args:
            query: Text query
            **kwargs: Same arguments as search()
            
        Returns:
            List of result dictionaries
        """
        results = self.retrieve(query, **kwargs)
        return [result.to_dict() for result in results]


def create_retriever(
    index_name: Optional[str] = None,
    default_top_k: int = 5,
    default_search_type: str = "hybrid"
) -> Retriever:
    """
    Factory function to create a Retriever with default configuration.
    
    Args:
        index_name: Search index name (loads from env if None)
        default_top_k: Default number of results
        default_search_type: Default search strategy
        
    Returns:
        Configured Retriever instance
    """
    return Retriever(
        index_name=index_name,
        default_top_k=default_top_k,
        default_search_type=default_search_type
    )


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create retriever
    retriever = create_retriever()
    
    # Example searches
    query = "What is the Minto Pyramid Principle?"
    
    # Vector search
    vector_results = retriever.retrieve(query, search_type="vector", top_k=3)
    print(f"Vector search found {len(vector_results)} results")
    
    # Hybrid search  
    hybrid_results = retriever.retrieve(query, search_type="hybrid", top_k=3)
    print(f"Hybrid search found {len(hybrid_results)} results")
    
    # Search with filters
    filtered_results = retriever.retrieve(
        query,
        filters={"doc_id": "specific_document", "page": [1, 2, 3]},
        top_k=5
    )
    print(f"Filtered search found {len(filtered_results)} results")
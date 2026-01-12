"""
Backward Compatibility Tests for Hierarchical Retrieval

Tests to ensure:
1. Flat mode works identically to v1 with v2 index
2. Hierarchical mode gracefully handles non-hierarchical chunks
3. Error conditions degrade gracefully
4. SearchResult format is backward compatible
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Import classes under test
from rag_toy.rag.retrieval.factory import create_retriever, is_hierarchical_mode, get_supported_modes
from rag_toy.rag.retrieval.retrieve import Retriever, SearchResult
from rag_toy.rag.retrieval.hierarchical_retriever import HierarchicalAzureRetriever

class TestBackwardCompatibility:
    """Test backward compatibility guarantees."""
    
    @pytest.fixture
    def mock_search_client(self):
        """Create mock Azure Search client."""
        client = Mock()
        
        # Mock search results for flat chunks (no hierarchical fields)
        flat_results = [
            {
                'id': 'chunk1',
                'content': 'This is content 1',
                '@search.score': 0.9,
                'doc_id': 'doc1',
                'page': 1,
                'source_uri': 'http://example.com',
                'content_type': 'text',
                # No hierarchical fields
            },
            {
                'id': 'chunk2', 
                'content': 'This is content 2',
                '@search.score': 0.8,
                'doc_id': 'doc1',
                'page': 2,
                'source_uri': 'http://example.com',
                'content_type': 'text',
                # No hierarchical fields
            }
        ]
        
        client.search.return_value = flat_results
        return client
    
    @pytest.fixture
    def mock_embeddings_client(self):
        """Create mock OpenAI embeddings client."""
        client = Mock()
        client.embeddings.create.return_value.data = [Mock(embedding=[0.1] * 1536)]
        return client

    def test_flat_mode_compatibility(self, mock_search_client, mock_embeddings_client):
        """Test that flat mode produces same results as v1."""
        
        # Test with RETRIEVAL_MODE=flat
        with patch.dict(os.environ, {'RETRIEVAL_MODE': 'flat'}):
            retriever = create_retriever(
                search_client=mock_search_client,
                embeddings_client=mock_embeddings_client,
                search_type="hybrid"
            )
            
            # Should create standard Retriever, not HierarchicalAzureRetriever
            assert isinstance(retriever, Retriever)
            assert not isinstance(retriever, HierarchicalAzureRetriever)
            
            # Search should work normally
            results = retriever.search("test query", top_k=5)
            
            # Verify results have expected flat structure
            assert len(results) == 2
            for result in results:
                assert isinstance(result, SearchResult)
                assert result.chunk_id is not None
                assert result.content is not None
                assert result.score > 0
                assert result.search_type == "hybrid"
                
                # Hierarchical fields should be None for backward compatibility
                assert result.node_id is None
                assert result.node_type is None  
                assert result.parent_path is None

    def test_hierarchical_mode_with_flat_chunks(self, mock_search_client, mock_embeddings_client):
        """Test hierarchical mode gracefully handles non-hierarchical chunks."""
        
        with patch.dict(os.environ, {'RETRIEVAL_MODE': 'hierarchical'}):
            retriever = create_retriever(
                search_client=mock_search_client,
                embeddings_client=mock_embeddings_client,
                search_type="hybrid"
            )
            
            # Should create HierarchicalAzureRetriever
            assert isinstance(retriever, HierarchicalAzureRetriever)
            
            # Search with non-hierarchical chunks should work (graceful degradation)
            results = retriever.search("test query", top_k=5)
            
            # Should return flat results since no hierarchical metadata
            assert len(results) == 2
            for result in results:
                assert isinstance(result, SearchResult)
                assert result.chunk_id is not None
                assert result.content is not None
                assert result.score > 0
                # search_type should indicate hierarchical mode was attempted
                assert "hierarchical" in result.search_type

    def test_hierarchical_mode_with_mixed_chunks(self, mock_embeddings_client):
        """Test hierarchical mode with mix of hierarchical and flat chunks."""
        
        # Mock search client with mixed results
        mock_search_client = Mock()
        
        mixed_results = [
            # Hierarchical chunk
            {
                'id': 'chunk_h1',
                'content': 'Hierarchical content 1',
                '@search.score': 0.9,
                'doc_id': 'doc1',
                'page': 1,
                'source_uri': 'http://example.com',
                'content_type': 'text',
                'node_id': 'h1',
                'node_type': 'chunk',
                'parent_path': 'doc1>section1'
            },
            # Flat chunk (no hierarchical fields)
            {
                'id': 'chunk_f1',
                'content': 'Flat content 1', 
                '@search.score': 0.8,
                'doc_id': 'doc2',
                'page': 1,
                'source_uri': 'http://example2.com',
                'content_type': 'text',
                # No hierarchical fields
            }
        ]
        
        mock_search_client.search.return_value = mixed_results
        
        with patch.dict(os.environ, {'RETRIEVAL_MODE': 'hierarchical'}):
            retriever = create_retriever(
                search_client=mock_search_client,
                embeddings_client=mock_embeddings_client,
                search_type="vector"
            )
            
            results = retriever.search("test query", top_k=5)
            
            # Should handle both types gracefully
            assert len(results) >= 2
            
            # Find the flat chunk result
            flat_result = next((r for r in results if r.chunk_id == 'chunk_f1'), None)
            assert flat_result is not None
            assert flat_result.node_id is None
            assert flat_result.node_type is None
            assert flat_result.parent_path is None
            
            # Find the hierarchical chunk result  
            hier_result = next((r for r in results if r.chunk_id == 'chunk_h1'), None)
            assert hier_result is not None
            assert hier_result.node_id == 'h1'
            assert hier_result.node_type == 'chunk'
            assert hier_result.parent_path == 'doc1>section1'

    def test_parent_lookup_failure_degradation(self, mock_embeddings_client):
        """Test graceful degradation when parent lookup fails."""
        
        # Mock search client that fails on parent lookup
        mock_search_client = Mock()
        
        # First call (leaf search) succeeds
        leaf_results = [
            {
                'id': 'chunk1',
                'content': 'Content with parent',
                '@search.score': 0.9,
                'doc_id': 'doc1',
                'page': 1,
                'source_uri': 'http://example.com',
                'content_type': 'text',
                'node_id': 'leaf1',
                'node_type': 'chunk',
                'parent_path': 'doc1>section1'
            }
        ]
        
        # Second call (parent lookup) fails
        def search_side_effect(*args, **kwargs):
            if kwargs.get('filter') and 'node_type eq' in kwargs.get('filter', ''):
                raise Exception("Parent lookup timeout")
            return leaf_results
        
        mock_search_client.search.side_effect = search_side_effect
        
        with patch.dict(os.environ, {'RETRIEVAL_MODE': 'hierarchical'}):
            retriever = create_retriever(
                search_client=mock_search_client,
                embeddings_client=mock_embeddings_client,
                search_type="hybrid"
            )
            
            # Should not raise exception, should return leaf results
            results = retriever.search("test query", top_k=5)
            
            # Should have leaf results despite parent lookup failure
            assert len(results) >= 1
            assert results[0].chunk_id == 'chunk1'
            assert results[0].node_id == 'leaf1'

    def test_complete_search_failure_fallback(self, mock_search_client, mock_embeddings_client):
        """Test fallback to flat search when hierarchical search completely fails."""
        
        # Mock hierarchical search to fail completely
        with patch.dict(os.environ, {'RETRIEVAL_MODE': 'hierarchical'}):
            retriever = create_retriever(
                search_client=mock_search_client,
                embeddings_client=mock_embeddings_client,
                search_type="hybrid"
            )
            
            # Mock the hierarchical search method to raise exception
            with patch.object(retriever, 'search', side_effect=Exception("Complete failure")):
                # Should not raise, should attempt fallback
                with patch.object(Retriever, 'search', return_value=[]):
                    results = retriever.search("test query", top_k=5)
                    # Should return empty list rather than crashing
                    assert isinstance(results, list)

    def test_search_result_serialization_compatibility(self):
        """Test that SearchResult.to_dict() maintains backward compatibility."""
        
        # Create SearchResult without hierarchical fields (flat mode)
        flat_result = SearchResult(
            chunk_id="chunk1",
            content="content",
            score=0.9,
            doc_id="doc1",
            page=1,
            source_uri="http://example.com",
            content_type="text",
            search_type="hybrid"
            # hierarchical fields default to None
        )
        
        flat_dict = flat_result.to_dict()
        
        # Should have all required fields
        required_fields = ["chunk_id", "content", "score", "doc_id", "page", "source_uri", "content_type", "search_type"]
        for field in required_fields:
            assert field in flat_dict
        
        # Hierarchical fields should not appear when None (backward compatibility)
        assert "node_id" not in flat_dict
        assert "node_type" not in flat_dict
        assert "parent_path" not in flat_dict
        
        # Create SearchResult with hierarchical fields
        hier_result = SearchResult(
            chunk_id="chunk2",
            content="content",
            score=0.9,
            doc_id="doc1", 
            page=1,
            source_uri="http://example.com",
            content_type="text",
            search_type="hybrid_hierarchical",
            node_id="node2",
            node_type="chunk",
            parent_path="doc1>section1"
        )
        
        hier_dict = hier_result.to_dict()
        
        # Should have all fields including hierarchical ones
        assert hier_dict["node_id"] == "node2"
        assert hier_dict["node_type"] == "chunk"
        assert hier_dict["parent_path"] == "doc1>section1"

class TestFactoryConfiguration:
    """Test factory configuration and environment variables."""
    
    def test_supported_modes(self):
        """Test that all expected modes are supported."""
        modes = get_supported_modes()
        assert "flat" in modes
        assert "hierarchical" in modes
        assert "adaptive" in modes
    
    def test_is_hierarchical_mode_detection(self):
        """Test hierarchical mode detection."""
        
        # Test explicit parameter
        assert is_hierarchical_mode("hierarchical") is True
        assert is_hierarchical_mode("adaptive") is True
        assert is_hierarchical_mode("flat") is False
        
        # Test environment variable
        with patch.dict(os.environ, {'RETRIEVAL_MODE': 'hierarchical'}):
            assert is_hierarchical_mode() is True
        
        with patch.dict(os.environ, {'RETRIEVAL_MODE': 'flat'}):
            assert is_hierarchical_mode() is False
        
        # Test default (no env var)
        with patch.dict(os.environ, {}, clear=True):
            if 'RETRIEVAL_MODE' in os.environ:
                del os.environ['RETRIEVAL_MODE']
            assert is_hierarchical_mode() is False  # Default is flat
    
    def test_unknown_mode_fallback(self, mock_search_client, mock_embeddings_client):
        """Test that unknown modes fall back to flat."""
        
        with patch.dict(os.environ, {'RETRIEVAL_MODE': 'unknown_mode'}):
            retriever = create_retriever(
                search_client=mock_search_client,
                embeddings_client=mock_embeddings_client,
                search_type="hybrid"
            )
            
            # Should fall back to flat mode (standard Retriever)
            assert isinstance(retriever, Retriever)
            assert not isinstance(retriever, HierarchicalAzureRetriever)
    
    @patch.dict(os.environ, {
        'RETRIEVAL_MODE': 'hierarchical',
        'MAX_PARENT_IDS_PER_LOOKUP': '50',
        'PARENT_CACHE_TTL_SECONDS': '300',
        'PARENT_CACHE_MAX_SIZE': '10000',
        'PARENT_LOOKUP_TIMEOUT': '5.0',
        'MAX_PARENTS_TOTAL': '5',
        'FINAL_TOP_K': '20'
    })
    def test_hierarchical_configuration_from_env(self, mock_search_client, mock_embeddings_client):
        """Test that hierarchical retriever picks up configuration from environment."""
        
        retriever = create_retriever(
            search_client=mock_search_client,
            embeddings_client=mock_embeddings_client,
            search_type="vector"
        )
        
        assert isinstance(retriever, HierarchicalAzureRetriever)
        
        # Check that configuration was applied
        assert retriever.max_parent_ids_per_lookup == 50
        assert retriever.cache_ttl_seconds == 300
        assert retriever.cache_max_size == 10000
        assert retriever.parent_lookup_timeout == 5.0
        assert retriever.max_parents_total == 5
        assert retriever.final_top_k == 20

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__])
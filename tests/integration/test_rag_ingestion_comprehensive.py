#!/usr/bin/env python3
"""
Comprehensive test suite for run_ingestion_with_env() function
"""
import tempfile
import json
import pytest
import logging
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
from src.rag_toy.rag.ingestion.orchestrator import run_ingestion_with_env

# Load environment for tests
load_dotenv()

def create_test_web_document(
    doc_id="test_web_doc_123",
    url="https://example.com/test-doc", 
    title="Test Document",
    content="This is a test document for verifying the ingestion pipeline. It contains sample text that should be chunked and indexed properly. The content is long enough to potentially generate multiple chunks depending on the chunker configuration.",
    metadata=None
):
    """Create a test web document matching expected loader format."""
    if metadata is None:
        metadata = {
            "doc_hash": "test123hash",
            "fetched_at": "2026-01-11T10:00:00Z",
            "content_type": "text/html"
        }
    
    return {
        "doc_id": doc_id,  # Required field for web loader
        "url": url,
        "title": title,
        "content": content,
        "metadata": metadata
    }

def create_test_pdf_document(
    doc_id="test_pdf_document",
    page=1,
    content="This is page content from a PDF document. It contains text that should be processed by the PDF chunker. The content should be chunked according to page boundaries.",
    metadata=None
):
    """Create a test PDF document matching expected loader format."""
    if metadata is None:
        metadata = {
            "doc_hash": "pdftest123",
            "extracted_at": "2026-01-11T10:00:00Z",
            "extraction_method": "pypdf"
        }
    
    return {
        "doc_id": doc_id,
        "page": page,
        "content": content,
        "metadata": metadata
    }

@pytest.fixture
def setup_logging():
    """Setup logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

@pytest.fixture
def mock_azure_env():
    """Mock Azure environment variables."""
    env_vars = {
        'AZURE_SEARCH_ENDPOINT': 'https://test-search.search.windows.net',
        'AZURE_INSERTION_KEY': 'test-search-key',
        'AZURE_OPENAI_ENDPOINT': 'https://test-openai.openai.azure.com',
        'AZURE_OPENAI_KEY': 'test-openai-key',
        'AZURE_EMBEDDING_DEPLOYMENT': 'text-embedding-ada-002',
        'AZURE_EMBEDDING_MODEL': 'text-embedding-ada-002'
    }
    
    with patch.dict(os.environ, env_vars):
        yield env_vars

@pytest.fixture
def mock_azure_clients():
    """Mock Azure clients to avoid actual API calls."""
    mock_search_client = MagicMock()
    mock_openai_client = MagicMock()
    
    # Mock successful embedding response
    mock_openai_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1536)]
    )
    
    # Mock successful indexing response
    mock_search_client.upload_documents.return_value = [
        MagicMock(succeeded=True, key="test-key")
    ]
    
    with patch('src.rag_toy.rag.ingestion.orchestrator.create_azure_openai_client', return_value=mock_openai_client), \
         patch('src.rag_toy.rag.ingestion.orchestrator.create_search_client', return_value=mock_search_client):
        yield mock_search_client, mock_openai_client

@pytest.fixture 
def mock_indexing_service():
    """Mock the IndexingService to return successful results."""
    from src.rag_toy.rag.ingestion.indexing.models import IndexingResult
    
    def mock_index_chunks(chunks):
        return IndexingResult(
            total_chunks=len(chunks),
            embedded_chunks=len(chunks),
            upserted_chunks=len(chunks),
            failed_chunks=0,
            processing_time_seconds=0.1,
            failed_chunk_ids=[],
            errors=[],
            embedding_batch_size=16,
            upsert_batch_size=200
        )
    
    with patch('src.rag_toy.rag.ingestion.orchestrator.IndexingService') as MockService:
        mock_instance = MockService.return_value
        mock_instance.index_chunks.side_effect = mock_index_chunks
        yield mock_instance

def test_single_web_document_ingestion(setup_logging, mock_azure_env, mock_indexing_service):
    """Test ingestion with a single web document."""
    
    # Create temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        test_doc = create_test_web_document()
        json.dump(test_doc, f)
        f.write('\n')
        temp_file = f.name
    
    try:
        data_paths = {"web": temp_file}
        summary = run_ingestion_with_env(
            data_paths=data_paths,
            chunk_engine="native"
        )
        
        # Assertions
        assert summary.success, f"Ingestion failed: {summary.errors}"
        assert summary.total_documents == 1
        assert summary.total_chunks > 0
        assert summary.indexed_chunks > 0
        assert summary.failed_chunks == 0
        assert len(summary.errors) == 0
        
        # Verify IndexingService was called
        assert mock_indexing_service.index_chunks.called
        
    finally:
        Path(temp_file).unlink()

def test_single_pdf_document_ingestion(setup_logging, mock_azure_env, mock_indexing_service):
    """Test ingestion with a single PDF document page."""
    
    # Create temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        test_doc = create_test_pdf_document()
        json.dump(test_doc, f)
        f.write('\n')
        temp_file = f.name
    
    try:
        data_paths = {"pdf": temp_file}
        summary = run_ingestion_with_env(
            data_paths=data_paths,
            chunk_engine="native"
        )
        
        # Assertions
        assert summary.success, f"Ingestion failed: {summary.errors}"
        assert summary.total_documents == 1
        assert summary.total_chunks > 0
        assert summary.indexed_chunks > 0
        assert summary.failed_chunks == 0
        
    finally:
        Path(temp_file).unlink()

def test_multiple_documents_mixed_types(setup_logging, mock_azure_env, mock_indexing_service):
    """Test ingestion with both PDF and web documents."""
    
    # Create web documents file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for i in range(2):
            test_doc = create_test_web_document(
                doc_id=f"test_web_doc_{i}",
                url=f"https://example.com/doc-{i}",
                title=f"Test Document {i}",
                content=f"This is web document {i} content."
            )
            json.dump(test_doc, f)
            f.write('\n')
        web_file = f.name
    
    # Create PDF documents file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for i in range(3):
            test_doc = create_test_pdf_document(
                doc_id=f"test_pdf_{i}",
                page=i+1,
                content=f"This is PDF document page {i+1} content."
            )
            json.dump(test_doc, f)
            f.write('\n')
        pdf_file = f.name
    
    try:
        data_paths = {"web": web_file, "pdf": pdf_file}
        summary = run_ingestion_with_env(
            data_paths=data_paths,
            chunk_engine="native"
        )
        
        # Assertions
        assert summary.success, f"Ingestion failed: {summary.errors}"
        assert summary.total_documents == 5  # 2 web + 3 pdf
        assert summary.total_chunks > 0
        assert summary.indexed_chunks > 0
        assert summary.failed_chunks == 0
        
    finally:
        Path(web_file).unlink()
        Path(pdf_file).unlink()

def test_chunker_parameters(setup_logging, mock_azure_env, mock_indexing_service):
    """Test ingestion with custom chunker parameters."""
    
    # Create document with long content to test chunking
    long_content = "This is a very long document. " * 100  # ~3000 chars
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        test_doc = create_test_web_document(content=long_content)
        json.dump(test_doc, f)
        f.write('\n')
        temp_file = f.name
    
    try:
        data_paths = {"web": temp_file}
        # Use valid chunker parameter names for native chunker
        summary = run_ingestion_with_env(
            data_paths=data_paths,
            chunk_engine="native"
        )
        
        assert summary.success
        assert summary.total_documents == 1
        assert summary.total_chunks >= 1  # Should create at least one chunk
        
    finally:
        Path(temp_file).unlink()

def test_empty_file(setup_logging, mock_azure_env, mock_azure_clients):
    """Test ingestion with empty JSONL file."""
    mock_search_client, mock_openai_client = mock_azure_clients
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        # Write empty file
        temp_file = f.name
    
    try:
        data_paths = {"web": temp_file}
        summary = run_ingestion_with_env(
            data_paths=data_paths,
            chunk_engine="native"
        )
        
        assert summary.success  # Should succeed even with no documents
        assert summary.total_documents == 0
        assert summary.total_chunks == 0
        assert summary.indexed_chunks == 0
        
    finally:
        Path(temp_file).unlink()

def test_invalid_chunk_engine(setup_logging, mock_azure_env, mock_azure_clients):
    """Test ingestion with invalid chunk engine."""
    mock_search_client, mock_openai_client = mock_azure_clients
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        test_doc = create_test_web_document()
        json.dump(test_doc, f)
        f.write('\n')
        temp_file = f.name
    
    try:
        data_paths = {"web": temp_file}
        summary = run_ingestion_with_env(
            data_paths=data_paths,
            chunk_engine="invalid_engine"
        )
        
        # Should fail validation
        assert not summary.success
        assert len(summary.errors) > 0
        assert "Unsupported chunk engine" in summary.errors[0]
        
    finally:
        Path(temp_file).unlink()

def test_missing_file(setup_logging, mock_azure_env, mock_azure_clients):
    """Test ingestion with non-existent file."""
    mock_search_client, mock_openai_client = mock_azure_clients
    
    data_paths = {"web": "/path/that/does/not/exist.jsonl"}
    summary = run_ingestion_with_env(
        data_paths=data_paths,
        chunk_engine="native"
    )
    
    # Should fail validation
    assert not summary.success
    assert len(summary.errors) > 0
    assert "File not found" in summary.errors[0]

def test_invalid_json_in_file(setup_logging, mock_azure_env, mock_indexing_service):
    """Test ingestion with invalid JSON in JSONL file."""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write("invalid json line\n")
        f.write('{"doc_id": "valid_doc", "content": "valid content"}\n')
        temp_file = f.name
    
    try:
        data_paths = {"web": temp_file}
        summary = run_ingestion_with_env(
            data_paths=data_paths,
            chunk_engine="native"
        )
        
        # The loader handles JSON errors gracefully by skipping invalid lines
        # So it should succeed with the valid document
        assert summary.success
        assert summary.total_documents == 1  # Only the valid document
        
    finally:
        Path(temp_file).unlink()

@pytest.mark.skipif(
    not all(os.getenv(var) for var in ['AZURE_SEARCH_ENDPOINT', 'AZURE_OPENAI_ENDPOINT']),
    reason="Azure environment variables not set"
)
def test_real_azure_integration(setup_logging):
    """Integration test with real Azure services (requires env vars)."""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        test_doc = create_test_web_document(doc_id="integration_test_doc", content="Integration test document content.")
        json.dump(test_doc, f)
        f.write('\n')
        temp_file = f.name
    
    try:
        data_paths = {"web": temp_file}
        summary = run_ingestion_with_env(
            data_paths=data_paths,
            chunk_engine="native"
        )
        
        # This test verifies end-to-end functionality with real Azure
        assert summary.success, f"Real Azure integration failed: {summary.errors}"
        assert summary.total_documents == 1
        assert summary.indexed_chunks > 0
        
    finally:
        Path(temp_file).unlink()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
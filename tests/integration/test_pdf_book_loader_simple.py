#!/usr/bin/env python3
"""
Test suite for PDFBookLoader
"""
import tempfile
import pytest
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.rag_toy.rag.ingestion.loaders.pdf_book_loader import PDFBookLoader

@pytest.fixture
def setup_logging():
    """Setup logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

@pytest.fixture
def create_dummy_pdf():
    """Create a temporary dummy PDF file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        f.write(b'%PDF-1.4\n%dummy PDF content for testing\n')
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    Path(temp_file).unlink(missing_ok=True)

def test_load_single_pdf_file(setup_logging, create_dummy_pdf):
    """Test loading a single PDF file."""
    
    # Create mock document
    mock_doc = MagicMock()
    mock_doc.text = "This is sample PDF content extracted by LlamaIndex."
    mock_doc.metadata = {
        'file_name': Path(create_dummy_pdf).name,
        'file_path': create_dummy_pdf,
        'file_size': 12345
    }
    
    # Mock the reader
    mock_reader = MagicMock()
    mock_reader.load_data.return_value = [mock_doc]
    
    with patch('src.rag_toy.rag.ingestion.loaders.pdf_book_loader.SimpleDirectoryReader', return_value=mock_reader), \
         patch('src.rag_toy.rag.ingestion.loaders.pdf_book_loader.UnstructuredReader'):
        
        loader = PDFBookLoader()
        documents = loader.load(create_dummy_pdf)
    
    # Assertions
    assert len(documents) == 1
    doc = documents[0]
    
    assert doc.content == "This is sample PDF content extracted by LlamaIndex."
    assert doc.content_type == "pdf_book"
    assert doc.source_uri == create_dummy_pdf
    assert doc.page is None
    assert doc.metadata['loader'] == 'pdf_book'
    assert doc.metadata['file_name'] == Path(create_dummy_pdf).name

def test_load_pdf_from_directory(setup_logging):
    """Test loading PDF from a directory."""
    
    mock_doc = MagicMock()
    mock_doc.text = "Directory PDF content"
    mock_doc.metadata = {'file_name': 'test.pdf', 'file_path': '/dir/test.pdf'}
    
    mock_reader = MagicMock()
    mock_reader.load_data.return_value = [mock_doc]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('src.rag_toy.rag.ingestion.loaders.pdf_book_loader.SimpleDirectoryReader', return_value=mock_reader), \
             patch('src.rag_toy.rag.ingestion.loaders.pdf_book_loader.UnstructuredReader'):
            
            loader = PDFBookLoader()
            documents = loader.load(temp_dir)
            
            assert len(documents) == 1
            assert documents[0].content_type == "pdf_book"

def test_load_multiple_documents(setup_logging):
    """Test loading multiple PDF documents."""
    
    mock_doc1 = MagicMock()
    mock_doc1.text = "First PDF document content"
    mock_doc1.metadata = {'file_name': 'doc1.pdf', 'file_path': '/path/doc1.pdf'}
    
    mock_doc2 = MagicMock()
    mock_doc2.text = "Second PDF document content"
    mock_doc2.metadata = {'file_name': 'doc2.pdf', 'file_path': '/path/doc2.pdf'}
    
    mock_reader = MagicMock()
    mock_reader.load_data.return_value = [mock_doc1, mock_doc2]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('src.rag_toy.rag.ingestion.loaders.pdf_book_loader.SimpleDirectoryReader', return_value=mock_reader), \
             patch('src.rag_toy.rag.ingestion.loaders.pdf_book_loader.UnstructuredReader'):
            
            loader = PDFBookLoader()
            documents = loader.load(temp_dir)
            
            assert len(documents) == 2
            assert documents[0].content == "First PDF document content"
            assert documents[1].content == "Second PDF document content"

def test_document_without_text_attribute(setup_logging):
    """Test handling document without text attribute."""
    
    mock_doc = MagicMock()
    # Remove text attribute
    del mock_doc.text
    mock_doc.__str__ = MagicMock(return_value="Fallback content")
    mock_doc.metadata = {'file_name': 'test.pdf'}
    
    mock_reader = MagicMock()
    mock_reader.load_data.return_value = [mock_doc]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('src.rag_toy.rag.ingestion.loaders.pdf_book_loader.SimpleDirectoryReader', return_value=mock_reader), \
             patch('src.rag_toy.rag.ingestion.loaders.pdf_book_loader.UnstructuredReader'):
            
            loader = PDFBookLoader()
            documents = loader.load(temp_dir)
            
            assert len(documents) == 1
            assert "Fallback content" in documents[0].content

def test_document_without_metadata(setup_logging):
    """Test handling document without metadata."""
    
    mock_doc = MagicMock()
    mock_doc.text = "Content without metadata"
    mock_doc.metadata = None
    
    mock_reader = MagicMock()
    mock_reader.load_data.return_value = [mock_doc]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('src.rag_toy.rag.ingestion.loaders.pdf_book_loader.SimpleDirectoryReader', return_value=mock_reader), \
             patch('src.rag_toy.rag.ingestion.loaders.pdf_book_loader.UnstructuredReader'):
            
            loader = PDFBookLoader()
            documents = loader.load(temp_dir)
            
            assert len(documents) == 1
            doc = documents[0]
            assert doc.content == "Content without metadata"
            assert doc.metadata['loader'] == 'pdf_book'
            assert doc.source_uri == temp_dir

def test_invalid_file_path(setup_logging):
    """Test error handling for invalid file paths."""
    loader = PDFBookLoader()
    
    # Test non-existent file
    with pytest.raises(ValueError, match="Invalid PDF path"):
        loader.load("/path/that/does/not/exist.pdf")
    
    # Test non-PDF file
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        temp_file = f.name
    
    try:
        with pytest.raises(ValueError, match="Invalid PDF path"):
            loader.load(temp_file)
    finally:
        Path(temp_file).unlink(missing_ok=True)

def test_empty_directory(setup_logging):
    """Test loading from empty directory."""
    
    mock_reader = MagicMock()
    mock_reader.load_data.return_value = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('src.rag_toy.rag.ingestion.loaders.pdf_book_loader.SimpleDirectoryReader', return_value=mock_reader), \
             patch('src.rag_toy.rag.ingestion.loaders.pdf_book_loader.UnstructuredReader'):
            
            loader = PDFBookLoader()
            documents = loader.load(temp_dir)
            
            assert len(documents) == 0

def test_file_filtering(setup_logging, create_dummy_pdf):
    """Test that file filtering works correctly."""
    
    # Mock multiple documents
    target_filename = Path(create_dummy_pdf).name
    
    mock_target = MagicMock()
    mock_target.text = "Target PDF content"
    mock_target.metadata = {'file_name': target_filename}
    
    mock_other = MagicMock()
    mock_other.text = "Other PDF content"
    mock_other.metadata = {'file_name': 'other.pdf'}
    
    mock_reader = MagicMock()
    mock_reader.load_data.return_value = [mock_target, mock_other]
    
    with patch('src.rag_toy.rag.ingestion.loaders.pdf_book_loader.SimpleDirectoryReader', return_value=mock_reader), \
         patch('src.rag_toy.rag.ingestion.loaders.pdf_book_loader.UnstructuredReader'):
        
        loader = PDFBookLoader()
        documents = loader.load(create_dummy_pdf)
        
        # Should only load the target file
        assert len(documents) == 1
        assert documents[0].content == "Target PDF content"

def test_logging_behavior(setup_logging, caplog):
    """Test that logging works correctly."""
    
    mock_doc = MagicMock()
    mock_doc.text = "Test content"
    mock_doc.metadata = {}
    
    mock_reader = MagicMock()
    mock_reader.load_data.return_value = [mock_doc]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('src.rag_toy.rag.ingestion.loaders.pdf_book_loader.SimpleDirectoryReader', return_value=mock_reader), \
             patch('src.rag_toy.rag.ingestion.loaders.pdf_book_loader.UnstructuredReader'):
            
            loader = PDFBookLoader()
            with caplog.at_level(logging.INFO):
                documents = loader.load(temp_dir)
            
            assert "PDFBookLoader loaded 1 document(s)" in caplog.text

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
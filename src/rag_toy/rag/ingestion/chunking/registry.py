"""Factory for creating chunkers based on engine and document type."""

from typing import Optional, Type
from .base import Chunker, ChunkerConfig
from ...models import Document


def get_chunker(engine: str, doc_type: str, config: Optional[ChunkerConfig] = None) -> Chunker:
    """
    Factory function to get appropriate chunker instance.
    
    Args:
        engine: Chunking engine ('native', 'langchain', 'llamaindex')
        doc_type: Document type ('pdf', 'web', etc.)
        config: Optional chunking configuration
        
    Returns:
        Chunker instance
        
    Raises:
        ValueError: If engine or doc_type combination is not supported
        ImportError: If required dependencies are not available
    """
    if config is None:
        config = ChunkerConfig()
        
    # Validate engine
    supported_engines = ["native", "langchain", "llamaindex"]
    if engine not in supported_engines:
        raise ValueError(f"Unsupported engine: {engine}. Supported engines: {supported_engines}")
        
    # Validate document type  
    supported_doc_types = ["pdf", "web", "pdf_book"]
    if doc_type not in supported_doc_types:
        raise ValueError(f"Unsupported document type: {doc_type}. Supported types: {supported_doc_types}")
    
    try:
        if engine == "native":
            return _get_native_chunker(doc_type, config)
        elif engine == "langchain":
            return _get_langchain_chunker(doc_type, config)
        elif engine == "llamaindex":
            return _get_llamaindex_chunker(doc_type, config)
    except ImportError as e:
        raise ImportError(f"Failed to import {engine} chunker: {e}")


def _get_native_chunker(doc_type: str, config: ChunkerConfig) -> Chunker:
    """Get native chunker implementation."""
    if doc_type == "pdf":
        from .native.pdf_chunker import NativePDFChunker
        return NativePDFChunker(config)
    elif doc_type == "web":
        from .native.web_chunker import NativeWebChunker
        return NativeWebChunker(config)
    else:
        raise ValueError(f"Native engine doesn't support doc_type: {doc_type}")


def _get_langchain_chunker(doc_type: str, config: ChunkerConfig) -> Chunker:
    """Get LangChain chunker implementation."""
    if doc_type == "pdf":
        from .llamaindex.pdf_chunker import LangChainPDFChunker
        return LangChainPDFChunker(config)
    elif doc_type == "web":
        from .langchain.web_chunker import LangChainWebChunker
        return LangChainWebChunker(config)
    else:
        raise ValueError(f"LangChain engine doesn't support doc_type: {doc_type}")


def _get_llamaindex_chunker(doc_type: str, config: ChunkerConfig) -> Chunker:
    """Get LlamaIndex chunker implementation."""
    if doc_type == "pdf_book":
        from .llamaindex.pdf_chunker import LlamaIndexPDFChunker
        return LlamaIndexPDFChunker(
            chunk_sizes=getattr(config, 'chunk_sizes', None),
            chunk_overlap=getattr(config, 'chunk_overlap', 20)
        )
    else:
        raise ValueError(f"LlamaIndex engine doesn't support doc_type: {doc_type}")
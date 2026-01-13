"""Factory for creating chunkers based on engine and document type."""

from typing import Optional, Type
from .base import Chunker, ChunkerConfig
from ...models import Document


def get_chunker(engine: str, doc_type: str, config: Optional[ChunkerConfig] = None, **chunker_kwargs) -> Chunker:
    """
    Factory function to get appropriate chunker instance.
    
    Args:
        engine: Chunking engine ('native', 'langchain', 'llamaindex')
        doc_type: Document type ('pdf', 'web', etc.)
        config: Optional chunking configuration
        **chunker_kwargs: Additional chunker-specific configuration parameters
        
    Returns:
        Chunker instance
        
    Raises:
        ValueError: If engine or doc_type combination is not supported
        ImportError: If required dependencies are not available
    """
    if config is None:
        config = ChunkerConfig()
        
    # Validate engine
    supported_engines = ["native", "langchain", "llamaindex", "llamaparse"]
    if engine not in supported_engines:
        raise ValueError(f"Unsupported engine: {engine}. Supported engines: {supported_engines}")

    # Validate document type  
    supported_doc_types = ["pdf", "web", "pdf_book", "pdf_llamaparse"]
    if doc_type not in supported_doc_types:
        raise ValueError(f"Unsupported document type: {doc_type}. Supported types: {supported_doc_types}")

    try:
        if engine == "native":
            return _get_native_chunker(doc_type, config, **chunker_kwargs)
        elif engine == "langchain":
            return _get_langchain_chunker(doc_type, config, **chunker_kwargs)
        elif engine == "llamaindex":
            return _get_llamaindex_chunker(doc_type, config, **chunker_kwargs)
        elif engine == "llamaparse":
            return _get_llamaparse_chunker(doc_type, config, **chunker_kwargs)
    except ImportError as e:
        raise ImportError(f"Failed to import {engine} chunker: {e}")


def _get_native_chunker(doc_type: str, config: ChunkerConfig, **chunker_kwargs) -> Chunker:
    """Get native chunker implementation."""
    if doc_type == "pdf":
        raise NotImplementedError("Native web chunker is not implemented yet.")
    elif doc_type == "web":
        raise NotImplementedError("Native web chunker is not implemented yet.")
    else:
        raise ValueError(f"Native engine doesn't support doc_type: {doc_type}")


def _get_langchain_chunker(doc_type: str, config: ChunkerConfig, **chunker_kwargs) -> Chunker:
    """Get LangChain chunker implementation."""
    if doc_type == "pdf":
        from .llamaindex.pdf_chunker import LangChainPDFChunker
        return LangChainPDFChunker(config, **chunker_kwargs)
    elif doc_type == "web":
        raise NotImplementedError("LangChain web chunker is not implemented yet.")
    else:
        raise ValueError(f"LangChain engine doesn't support doc_type: {doc_type}")


def _get_llamaindex_chunker(doc_type: str, config: ChunkerConfig, **chunker_kwargs) -> Chunker:
    """Get LlamaIndex chunker implementation."""
    if doc_type == "pdf_book":
        from .llamaindex.pdf_chunker import LlamaIndexPDFChunker
        # Use chunker_kwargs if provided, otherwise fallback to config attributes or defaults
        chunk_sizes = chunker_kwargs.get('chunk_sizes') or getattr(config, 'chunk_sizes', None)
        chunk_overlap = chunker_kwargs.get('chunk_overlap') or getattr(config, 'chunk_overlap', 20)
        
        return LlamaIndexPDFChunker(
            chunk_sizes=chunk_sizes,
            chunk_overlap=chunk_overlap
        )
    if doc_type == "pdf_llamaparse":
        from .llamaindex.llamaparse_pdf_chunker import LlamaParsePDFChunker
        return LlamaParsePDFChunker()
    else:
        raise ValueError(f"LlamaIndex engine doesn't support doc_type: {doc_type}")


def _get_llamaparse_chunker(doc_type: str, config: ChunkerConfig, **chunker_kwargs) -> Chunker:
    """Get LlamaParse chunker implementation."""
    if doc_type == "pdf_llamaparse":
        from .llamaindex.llamaparse_pdf_chunker import LlamaParsePDFChunker
        return LlamaParsePDFChunker()
    else:
        raise ValueError(f"LlamaParse engine doesn't support doc_type: {doc_type}")
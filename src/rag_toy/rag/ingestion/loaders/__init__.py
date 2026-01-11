"""Document loaders for various source types."""

from .base import DocumentLoader
from .pdf_loader import PDFLoader
from .web_loader import WebLoader

# Registry of available loaders
_LOADERS = {
    "pdf": PDFLoader,
    "web": WebLoader
}


def get_loader(source_type: str) -> DocumentLoader:
    """
    Get appropriate loader for the given source type.
    
    Args:
        source_type: Type of source ("pdf", "web")
        
    Returns:
        DocumentLoader instance for the source type
        
    Raises:
        ValueError: If source_type is not supported
    """
    if source_type not in _LOADERS:
        supported = ", ".join(sorted(_LOADERS.keys()))
        raise ValueError(f"Unsupported source type: {source_type}. Supported types: {supported}")
    
    loader_class = _LOADERS[source_type]
    return loader_class()


__all__ = [
    "DocumentLoader",
    "PDFLoader", 
    "WebLoader",
    "get_loader"
]
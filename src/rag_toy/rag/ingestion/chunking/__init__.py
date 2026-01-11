"""Chunking module for RAG system with multiple engine support."""

from .base import Chunker, ChunkerConfig
from .registry import get_chunker
from .heuristics import is_slide_like, needs_recursive_chunking, default_chunk_params

__all__ = [
    "Chunker",
    "ChunkerConfig", 
    "get_chunker",
    "is_slide_like",
    "needs_recursive_chunking",
    "default_chunk_params"
]
"""Base chunker interfaces and configuration."""

from typing import Protocol, List, Optional
from pydantic import BaseModel, Field

from ...models import Document, Chunk


class ChunkerConfig(BaseModel):
    """Configuration for chunking operations."""
    min_chars: int = Field(default=100, ge=1, description="Minimum characters per chunk")
    max_chars: int = Field(default=1200, ge=1, description="Maximum characters per chunk")
    overlap_chars: int = Field(default=100, ge=0, description="Overlap characters between chunks")
    engine: str = Field(default="native", description="Chunking engine to use")
    
    # Engine-specific parameters can be added here
    extra_params: dict = Field(default_factory=dict, description="Engine-specific parameters")
    
    def model_post_init(self, __context):
        """Validate configuration after initialization."""
        if self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be less than max_chars")
        if self.min_chars > self.max_chars:
            raise ValueError("min_chars must be less than or equal to max_chars")


class Chunker(Protocol):
    """Protocol for chunking implementations."""
    engine: str          # "native" | "langchain" | ...
    doc_type: str        # "pdf" | "web"
    version: str         # "v1"
    
    def chunk(self, document: Document, config: Optional[ChunkerConfig] = None) -> List[Chunk]:
        """
        Chunk a document into smaller pieces.
        
        Args:
            document: The document to chunk
            config: Chunking configuration (optional, uses defaults if not provided)
            
        Returns:
            List of chunks with stable chunk_id, doc_id, page, content_type, 
            source_uri, content, and metadata fields
        """
        ...
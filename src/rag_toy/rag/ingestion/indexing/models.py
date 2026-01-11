"""Data models for indexing operations."""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class IndexingResult(BaseModel):
    """
    Structured result from indexing operation.
    
    Provides clear metrics and error information for monitoring
    and debugging indexing operations.
    """
    
    total_chunks: int = Field(..., description="Total number of chunks processed")
    embedded_chunks: int = Field(..., description="Number of chunks that got embeddings")
    upserted_chunks: int = Field(..., description="Number of chunks successfully upserted to search")
    failed_chunks: int = Field(..., description="Number of chunks that failed processing")
    
    processing_time_seconds: float = Field(..., description="Total processing time in seconds")
    
    # Error tracking
    failed_chunk_ids: List[str] = Field(default_factory=list, description="IDs of chunks that failed")
    errors: List[str] = Field(default_factory=list, description="Error messages encountered")
    
    # Processing metadata
    embedding_batch_size: int = Field(..., description="Batch size used for embedding generation")
    upsert_batch_size: int = Field(..., description="Batch size used for upsert operations")
    
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When processing completed")
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_chunks == 0:
            return 0.0
        return (self.upserted_chunks / self.total_chunks) * 100.0
    
    @property
    def is_successful(self) -> bool:
        """True if all chunks were processed successfully."""
        return self.failed_chunks == 0 and self.total_chunks > 0
    
    def summary(self) -> str:
        """Human-readable summary of results."""
        return (
            f"Indexed {self.upserted_chunks}/{self.total_chunks} chunks "
            f"({self.success_rate:.1f}% success) in {self.processing_time_seconds:.2f}s"
        )


class IngestionSummary(BaseModel):
    """Summary of complete ingestion pipeline execution."""
    success: bool
    total_documents: int
    total_chunks: int
    indexed_chunks: int
    failed_chunks: int
    errors: List[str] = []
    chunk_engine: str
    index_name: str
    processing_time: float
    timestamp: datetime
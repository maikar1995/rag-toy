from pydantic import BaseModel, Field
from typing import Optional, List

from enum import Enum
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

class Document(BaseModel):
    id: str  # doc_id for consistency
    source: str
    title: Optional[str] = None
    url: Optional[str] = None
    
    # Required fields for chunking/indexing pipeline
    content: Optional[str] = None
    content_type: Optional[str] = None  # pdf_page, web_doc, etc.
    source_uri: Optional[str] = None
    page: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def doc_id(self) -> str:
        """Alias for id to match Azure Search and chunk naming."""
        return self.id

class Chunk(BaseModel):
    id: str
    document_id: str
    text: str
    chunk_index: int
    embedding: Optional[List[float]] = None
    source: Optional[str] = None
    url: Optional[str] = None
    page: Optional[int] = None
    content_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # Add other flat fields as needed



class AbstractionReason(Enum):
    """Reasons for abstaining from answering."""
    NO_CHUNKS = "no_chunks_provided"
    NO_CITATIONS = "no_valid_citations" 
    OUTSIDE_CONTEXT = "response_outside_context"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    GENERATION_ERROR = "generation_error"


class Citation(BaseModel):
    """Citation reference to a specific chunk."""
    chunk_id: str
    doc_id: str
    page: Optional[int] = None
    relevance: float = 1.0


class AnswerResponse(BaseModel):
    """Complete answer response with metadata."""
    answer: Optional[str] = None
    citations: List[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    abstain_reason: Optional[AbstractionReason] = None
    notes: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "answer": self.answer,
            "citations": [asdict(citation) for citation in self.citations],
            "confidence": self.confidence,
            "notes": self.notes
        }

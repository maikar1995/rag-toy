from pydantic import BaseModel, Field
from typing import Optional, List

from enum import Enum
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

class Document(BaseModel):
    id: str
    source: str
    title: Optional[str] = None
    url: Optional[str] = None
    # Add other document-level fields as needed

class Chunk(BaseModel):
    id: str
    document_id: str
    text: str
    chunk_index: int
    embedding: Optional[List[float]] = None
    source: Optional[str] = None
    url: Optional[str] = None
    # Add other flat fields as needed



class AbstractionReason(Enum):
    """Reasons for abstaining from answering."""
    NO_CHUNKS = "no_chunks_provided"
    NO_CITATIONS = "no_valid_citations" 
    OUTSIDE_CONTEXT = "response_outside_context"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    GENERATION_ERROR = "generation_error"


@dataclass
class Citation:
    """Citation reference to a specific chunk."""
    chunk_id: str
    doc_id: str
    page: Optional[int]
    relevance: float


@dataclass
class AnswerResponse:
    """Complete answer response with metadata."""
    answer: Optional[str]
    citations: List[Citation]
    confidence: float
    notes: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "answer": self.answer,
            "citations": [asdict(citation) for citation in self.citations],
            "confidence": self.confidence,
            "notes": self.notes
        }

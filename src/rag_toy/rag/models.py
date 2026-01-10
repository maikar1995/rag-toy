from pydantic import BaseModel, Field
from typing import Optional, List

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


class Evidence(BaseModel):
    chunk_id: str
    text: str
    score: Optional[float] = None
    # Add other evidence fields as needed

    @classmethod
    def from_search_hit(cls, hit: dict) -> "Evidence":
        """
        Build an Evidence instance from a search result dict (e.g., Azure Cognitive Search hit).
        """
        return cls(
            chunk_id=hit.get("chunk_id") or hit.get("id"),
            text=hit.get("text"),
            score=hit.get("score"),
        )


class AnswerResponse(BaseModel):
    answer: str
    evidences: List[Evidence]
    # Add other response fields as needed

    @classmethod
    def from_llm_json(cls, data: dict) -> "AnswerResponse":
        """
        Build an AnswerResponse from a dict (e.g., LLM output or API response).
        """
        evidences = [Evidence(**ev) if not isinstance(ev, Evidence) else ev for ev in data.get("evidences", [])]
        return cls(
            answer=data.get("answer", ""),
            evidences=evidences,
        )

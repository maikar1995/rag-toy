from typing import Protocol, List
from .models import Chunk, Evidence, AnswerResponse

class Chunker(Protocol):
    def chunk(self, document: str, **kwargs) -> List[Chunk]:
        ...

class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        ...

class AnswerGenerator(Protocol):
    def generate(self, query: str, evidences: List[Evidence]) -> AnswerResponse:
        ...


from typing import List, Optional
from .interfaces import Retriever, AnswerGenerator
from .models import Citation, AnswerResponse
from .generation.citations import validate_citations_hard_fail
import logging

logger = logging.getLogger(__name__)

def abstain(reason: str) -> Optional[AnswerResponse]:
    logger.warning(f"Abstaining: {reason}")
    return None

class RAGService:
    def __init__(self, retriever: Retriever, answer_generator: AnswerGenerator):
        self.retriever = retriever
        self.answer_generator = answer_generator

    async def ask(self, query: str, top_k: int = 5) -> Optional[AnswerResponse]:
        citations: List[Citation] = self.retriever.retrieve(query, top_k=top_k)
        if not citations:
            return abstain("no_evidence")

        draft: AnswerResponse = self.answer_generator.generate(query, citations)
        def retry():
            return self.answer_generator.generate(query, citations)

        validated = validate_citations_hard_fail(
            answer=draft,
            used_citations=citations,
            retry_fn=retry
        )

        if validated is None:
            return abstain("invalid_citations")

        return validated
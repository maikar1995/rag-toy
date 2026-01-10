from typing import List
from .interfaces import Retriever, AnswerGenerator
from .models import Evidence, AnswerResponse

class RAGService:
    def __init__(self, retriever: Retriever, answer_generator: AnswerGenerator):
        self.retriever = retriever
        self.answer_generator = answer_generator

    def ask(self, query: str, top_k: int = 5) -> AnswerResponse:
        """
        Main entrypoint for RAG pipeline: retrieves evidences and generates answer.
        Returns an AnswerResponse (never a dict).
        """
        evidences: List[Evidence] = self.retriever.retrieve(query, top_k=top_k)
        answer_response: AnswerResponse = self.answer_generator.generate(query, evidences)
        return answer_response

class RAGService:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    async def ask(self, query: str, context_id: str | None = None):
        results = self.retriever.search(query=query, top_k=5, filters=None)
        return self.generator.generate(query=query, search_results=results)
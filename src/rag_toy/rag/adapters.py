from typing import List
from .interfaces import Retriever, AnswerGenerator
from .models import Evidence, AnswerResponse

class AzureSearchRetrieverAdapter(Retriever):
    def __init__(self, search_client):
        self.search_client = search_client

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        # Call the real Azure Search client, get dicts, convert to Evidence
        hits = self.search_client.search(query, top_k=top_k)
        return [Evidence.from_search_hit(hit) for hit in hits]


class AzureOpenAIAnswerGeneratorAdapter(AnswerGenerator):
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def generate(self, query: str, evidences: List[Evidence]) -> AnswerResponse:
        # Call the real LLM client, get dict, convert to AnswerResponse
        data = self.llm_client.generate_answer(query, [ev.dict() for ev in evidences])
        return AnswerResponse.from_llm_json(data)
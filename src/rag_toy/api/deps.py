import os
from functools import lru_cache

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

from rag_toy.rag.retrieval.retrieve import Retriever
from rag_toy.rag.generation.answer import AnswerGenerator
from rag_toy.rag.service import RAGService


@lru_cache
def get_search_client(index:str = None) -> SearchClient:
    return SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name=index or os.environ["AZURE_SEARCH_INDEX"],
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"]),
    )


@lru_cache
def get_openai_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.environ["LLM_API_KEY"],
        azure_endpoint=os.environ["LLM_AZURE_ENDPOINT"],
        api_version="2024-02-15-preview",
    )


def get_rag_service(search_type=None, index=None) -> RAGService:
    index = index or os.environ["AZURE_SEARCH_INDEX"]
    search_client = get_search_client(index)
    openai_client = get_openai_client()

    retriever = Retriever(
        search_client=search_client,
        openai_client=openai_client,
        index_name=index,
        default_search_type=search_type,
    )

    generator = AnswerGenerator(
        openai_client=openai_client,
        model_deployment=os.environ["LLM_MODEL_NAME"],
        search_type=search_type,
    )

    return RAGService(retriever=retriever, answer_generator=generator)
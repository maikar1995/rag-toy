import os
from functools import lru_cache

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

from rag_toy.rag.retrieval.retrieve import Retriever
from rag_toy.rag.generation.answer import AnswerGenerator
from rag_toy.rag.service import RAGService


@lru_cache
def get_search_client() -> SearchClient:
    return SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name=os.environ["AZURE_SEARCH_INDEX"],
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"]),
    )


@lru_cache
def get_openai_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.environ["LLM_API_KEY"],
        azure_endpoint=os.environ["LLM_AZURE_ENDPOINT"],
        api_version="2024-02-15-preview",
    )


def get_rag_service() -> RAGService:
    search_client = get_search_client()
    openai_client = get_openai_client()

    retriever = Retriever(
        search_client=search_client,
        openai_client=openai_client,
    )

    generator = AnswerGenerator(
        openai_client=openai_client,
        model_deployment=os.environ["LLM_MODEL_NAME"],
    )

    return RAGService(retriever=retriever, generator=generator)
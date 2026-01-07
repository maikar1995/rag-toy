from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.deps import get_rag_service
from rag.service import RAGService
from rag_toy.rag import rag_query  # Tu servicio RAG con Azure AI Search

router = APIRouter(prefix="/rag", tags=["RAG"])  # Prefix para /api/v1/rag/ask

class AskRequest(BaseModel):
    query: str
    context_id: str = None  # Opcional para namespaces/índices

@router.post("/ask")  # POST /api/v1/rag/ask
async def ask_rag(req: AskRequest, svc: RAGService = Depends(get_rag_service)):
    return await svc.ask(req.query, req.context_id)

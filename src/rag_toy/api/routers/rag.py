from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.rag_toy.api.deps import get_rag_service
from src.rag_toy.rag.service import RAGService

router = APIRouter(prefix="/rag", tags=["RAG"])  # Prefix para /api/v1/rag/ask

class AskRequest(BaseModel):
    query: str
    context_id: str = None  # Opcional para namespaces/índices

@router.post("/ask")  # POST /api/v1/rag/ask
async def ask_rag(req: AskRequest, svc: RAGService = Depends(get_rag_service)):
    return await svc.ask(req.query, req.context_id)

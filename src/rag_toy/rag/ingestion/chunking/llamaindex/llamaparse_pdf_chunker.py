"""Chunker for LlamaParse PDF loader: 1 table = 1 chunk, text/caption atomic, correct content_type."""
from typing import Iterator, List, Optional
import uuid
from ....models import Document, Chunk
from ..base import Chunker, ChunkerConfig

class LlamaParsePDFChunker(Chunker):
    engine = "llamaindex"
    doc_type = "pdf_llamaparse"
    version = "v1"

    def chunk(self, document: Document, config: Optional[ChunkerConfig] = None) -> List[Chunk]:
        # 1 table = 1 chunk, text/caption atomic, no splitting unless content is huge
        chunks = []
        content = document.content or ""
        chunk_id = str(uuid.uuid4())
        chunk = Chunk(
            id=chunk_id,
            document_id=document.id,
            text=content,
            chunk_index=0,
            source=document.source,
            url=document.url,
            page=document.page,
            content_type=document.content_type,
            metadata=document.metadata or {}
        )
        chunks.append(chunk)
        return chunks

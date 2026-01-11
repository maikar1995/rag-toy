"""
LlamaIndexPDFChunker: Chunker for PDFBookLoader using HierarchicalNodeParser
"""
from typing import Iterator, Optional, List
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.schema import Document as LlamaDocument
from ....models import Document, Chunk
from ..base import ChunkerConfig, Chunker
import logging
import uuid

logger = logging.getLogger(__name__)

class LlamaIndexPDFChunker(Chunker):
    """
    Chunker for PDFBookLoader using LlamaIndex HierarchicalNodeParser.
    """
    def __init__(self, chunk_sizes: Optional[List[int]] = None, chunk_overlap: int = 20, **kwargs):
        self.chunk_sizes = chunk_sizes or [2048, 512, 128]
        self.chunk_overlap = chunk_overlap
        self.node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=self.chunk_sizes,
            chunk_overlap=self.chunk_overlap
        )

    def chunk(self, doc: Document) -> Iterator[Chunk]:
        # Convert our domain Document to LlamaIndex Document
        llama_doc = LlamaDocument(
            text=doc.content,
            metadata=doc.metadata or {}
        )
        nodes = self.node_parser.get_nodes_from_documents([llama_doc])
        for node in nodes:
            # Compose our domain Chunk
            chunk_id = str(uuid.uuid4())
            yield Chunk(
                id=chunk_id,
                document_id=getattr(doc, 'doc_id', None) or doc.metadata.get('doc_id') or chunk_id,
                text=node.text,
                content_type="pdf_book_chunk",
                page=None,
                source=doc.source_uri,
                metadata={
                    **(node.metadata or {}),
                    "parent_ids": getattr(node, 'parent_ids', None),
                    "node_id": getattr(node, 'node_id', None),
                    "chunk_method": "llamaindex_hierarchical",
                }
            )

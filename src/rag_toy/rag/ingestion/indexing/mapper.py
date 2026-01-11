"""Chunk to Azure Search document mapping."""

import re
from typing import Dict, Any, List

from ...models import Chunk


class ChunkMapper:
    """
    Handles mapping between Chunk models and Azure Search documents.
    
    This boundary abstraction allows the search schema to evolve
    independently from the chunk model.
    """
    
    @staticmethod
    def clean_document_id(chunk_id: str) -> str:
        """
        Clean chunk ID to be valid Azure Search document ID.
        
        Args:
            chunk_id: Original chunk identifier
            
        Returns:
            Cleaned ID safe for Azure Search
        """
        cleaned = re.sub(r'[^a-zA-Z0-9_\-=]', '_', chunk_id)
        cleaned = cleaned.lstrip('_')
        return cleaned
    
    @staticmethod
    def to_search_document(chunk: Chunk, embedding: List[float]) -> Dict[str, Any]:
        """
        Convert Chunk model to Azure Search document format.
        
        Args:
            chunk: Source chunk to convert
            embedding: Generated embedding vector
            
        Returns:
            Dictionary ready for Azure Search upsert
        """
        clean_chunk_id = ChunkMapper.clean_document_id(chunk.id)
        
        # Map chunk fields to search document
        # Uses flat structure - no nested metadata for Azure Search compatibility
        document = {
            "id": clean_chunk_id,
            "content": chunk.text,
            "contentVector": embedding,
            "doc_id": chunk.document_id,
            "page": getattr(chunk, 'page', None),
            "content_type": getattr(chunk, 'content_type', 'text'),
            "chunk_id": chunk.id,
            "source_uri": chunk.source or '',
        }
        
        # Add metadata fields from chunk.metadata if present
        if hasattr(chunk, 'metadata') and chunk.metadata:
            metadata = chunk.metadata
            document.update({
                "emb_version": metadata.get('emb_version'),
                "doc_hash": metadata.get('doc_hash'),
                "ingested_at": metadata.get('ingested_at'),
                "fetched_at": metadata.get('fetched_at'),
                "chunk_method": metadata.get('chunk_method')
            })
        
        return document
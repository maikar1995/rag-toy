"""Chunk to Azure Search document mapping."""

import re
from datetime import datetime
from typing import Dict, Any, List, Optional

from ...models import Chunk


class ChunkMapper:
    """
    Handles mapping between Chunk models and Azure Search documents.
    
    This boundary abstraction allows the search schema to evolve
    independently from the chunk model.
    """
    
    @staticmethod
    def _format_datetime_for_search(value: Any) -> Optional[str]:
        """
        Format datetime value for Azure Search Edm.DateTimeOffset compatibility.
        
        Azure Search requires ISO 8601 format with timezone (RFC 3339).
        
        Args:
            value: Datetime value (string, datetime object, or None)
            
        Returns:
            Formatted datetime string or None
        """
        if value is None:
            return None
            
        if isinstance(value, str):
            try:
                # Try to parse existing string
                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                return None
        elif isinstance(value, datetime):
            dt = value
        else:
            return None
        
        # Ensure UTC timezone and format as ISO 8601 with 'Z' suffix
        if dt.tzinfo is None:
            # Assume UTC if no timezone
            return dt.isoformat() + 'Z'
        else:
            # Convert to UTC and format
            utc_dt = dt.utctimetuple()
            return datetime(*utc_dt[:6]).isoformat() + 'Z'
    
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
                "ingested_at": ChunkMapper._format_datetime_for_search(metadata.get('ingested_at')),
                "fetched_at": ChunkMapper._format_datetime_for_search(metadata.get('fetched_at')),
                "chunk_method": metadata.get('chunk_method')
            })
        
        return document
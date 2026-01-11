"""Native web content chunking implementation."""

import re
from typing import List, Optional
from datetime import datetime

from ..base import Chunker, ChunkerConfig
from ....models import Document, Chunk


class NativeWebChunker:
    """Native web content chunking implementation.
    
    Strategy:
    - Group paragraphs to reach 800-1200 character chunks
    - No overlap to preserve semantic boundaries
    - Preserve document structure and metadata
    """
    
    def __init__(self, config: ChunkerConfig, engine: str = "native", doc_type: str = "web", version: str = "1.0"):
        self.config = config
        self.engine = engine
        self.doc_type = doc_type
        self.version = version
        self.config = config
        
    def chunk(self, document: Document, config: Optional[ChunkerConfig] = None) -> List[Chunk]:
        """
        Chunk web content by grouping paragraphs.
        
        Args:
            document: Document to chunk
            config: Optional config override
            
        Returns:
            List of chunks with stable IDs and no page numbers
        """
        chunk_config = config or self.config
        
        content = getattr(document, 'content', None)
        if not content:
            return []
            
        # Split content into paragraphs
        paragraphs = self._split_paragraphs(content)
        
        # Group paragraphs into chunks
        chunk_texts = self._group_paragraphs(
            paragraphs, 
            chunk_config.min_chars, 
            chunk_config.max_chars
        )
        
        # Create chunk objects
        chunks = []
        for chunk_index, chunk_text in enumerate(chunk_texts):
            if not chunk_text.strip():
                continue
                
            chunk_id = self._generate_chunk_id(document.id, chunk_index)
            
            chunk = Chunk(
                id=chunk_id,
                document_id=document.id,
                text=chunk_text.strip(),
                chunk_index=chunk_index,
                page=None,  # Web content doesn't have pages
                content_type="web",
                source=getattr(document, 'source', None) or getattr(document, 'url', None),
                url=getattr(document, 'url', None),
                metadata={
                    **getattr(document, 'metadata', {}),
                    "chunk_method": "web_paragraphs",
                    "original_char_count": len(content),
                    "chunk_char_count": len(chunk_text.strip()),
                    "created_at": datetime.utcnow().isoformat(),
                }
            )
            chunks.append(chunk)
            
        return chunks
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs using various heuristics."""
        # Split on double newlines first (standard paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Further split very long paragraphs on single newlines
        result = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # If paragraph is very long, split on single newlines
            if len(para) > 500:
                sub_paras = [p.strip() for p in para.split('\n') if p.strip()]
                result.extend(sub_paras)
            else:
                result.append(para)
        
        return result
    
    def _group_paragraphs(
        self, 
        paragraphs: List[str], 
        min_chars: int, 
        max_chars: int
    ) -> List[str]:
        """
        Group paragraphs into chunks that meet size requirements.
        
        Args:
            paragraphs: List of paragraph strings
            min_chars: Minimum characters per chunk
            max_chars: Maximum characters per chunk
            
        Returns:
            List of chunk texts
        """
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if not para.strip():
                continue
                
            # Check if adding this paragraph would exceed max_chars
            if current_chunk and len(current_chunk) + len(para) + 1 > max_chars:
                # Current chunk is complete, start new one
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n" + para
                else:
                    current_chunk = para
        
        # Add the final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Handle chunks that are too small by combining them
        return self._merge_small_chunks(chunks, min_chars, max_chars)
    
    def _merge_small_chunks(
        self, 
        chunks: List[str], 
        min_chars: int, 
        max_chars: int
    ) -> List[str]:
        """Merge chunks that are too small with adjacent chunks."""
        if not chunks:
            return []
            
        result = []
        i = 0
        
        while i < len(chunks):
            current = chunks[i]
            
            # If current chunk is too small, try to merge with next
            if len(current) < min_chars and i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                merged = current + "\n" + next_chunk
                
                # If merged chunk fits within max_chars, use it
                if len(merged) <= max_chars:
                    result.append(merged)
                    i += 2  # Skip next chunk since we merged it
                else:
                    # Can't merge, keep original chunk
                    result.append(current)
                    i += 1
            else:
                result.append(current)
                i += 1
                
        return result
    
    def _generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """Generate stable, deterministic chunk ID for web content."""
        # Use the existing pattern from the scripts
        return f"web__{doc_id}_chunk_{chunk_index:03d}"
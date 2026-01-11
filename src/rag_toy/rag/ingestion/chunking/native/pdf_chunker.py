"""Native PDF chunking implementation."""

import hashlib
import re
from typing import List, Optional, Tuple
from datetime import datetime

from ..base import Chunker, ChunkerConfig
from ....models import Document, Chunk


class NativePDFChunker(Chunker):
    """Native PDF chunking implementation.
    
    Strategy:
    - Page-based chunking with smart splitting for long pages
    - Never mix content from different pages
    - Preserve page boundaries and metadata
    """
    
    def __init__(self, config: ChunkerConfig, engine: str = "native", doc_type: str = "pdf", version: str = "1.0"):
        self.config = config
        self.engine = engine
        self.doc_type = doc_type
        self.version = version
        self.config = config
    
    def chunk(self, document: Document, config: Optional[ChunkerConfig] = None) -> List[Chunk]:
        """
        Chunk a PDF document page by page.
        
        Args:
            document: Document to chunk (should have page-based content structure)
            config: Optional config override
            
        Returns:
            List of chunks with stable IDs and page preservation
        """
        chunk_config = config or self.config
        
        content = getattr(document, 'content', None)
        if not content:
            return []
            
        # For PDF documents from JSONL, each document represents a single page
        # Use the actual page number from the document metadata
        actual_page_num = getattr(document, 'page', 1)
        
        page_chunks = self._chunk_page(
            document=document,
            page_num=actual_page_num,  # Use actual page number
            page_content=content,
            config=chunk_config
        )
        
        return page_chunks
    
    def _extract_pages(self, document: Document) -> List[str]:
        """
        Extract pages from document content.
        
        For PDF documents that come from JSONL with page structure,
        each document represents a single page.
        """
        # Each document in PDF JSONL represents a single page
        content = getattr(document, 'content', None)
        if content:
            return [content]
        return []
    
    def _chunk_page(
        self, 
        document: Document, 
        page_num: int, 
        page_content: str, 
        config: ChunkerConfig
    ) -> List[Chunk]:
        """Chunk a single page of content."""
        page_length = len(page_content)
        
        # If page fits in one chunk, create single chunk
        if page_length <= config.max_chars:
            chunk_id = self._generate_chunk_id(document.id, page_num, 0)
            return [Chunk(
                id=chunk_id,
                document_id=document.id,
                text=page_content.strip(),
                chunk_index=0,
                page=page_num,
                content_type="pdf",
                source=getattr(document, 'source', None) or getattr(document, 'url', None),
                url=getattr(document, 'url', None),
                metadata={
                    **getattr(document, 'metadata', {}),
                    "chunk_method": "page_single",
                    "chunk_char_count": len(page_content.strip()),
                    "page_char_count": page_length,
                    "created_at": datetime.utcnow().isoformat(),
                }
            )]
        
        # Split long page into multiple chunks
        return self._split_page_into_chunks(document, page_num, page_content, config)
    
    def _split_page_into_chunks(
        self, 
        document: Document, 
        page_num: int, 
        page_content: str, 
        config: ChunkerConfig
    ) -> List[Chunk]:
        """Split a long page into multiple chunks with overlap."""
        chunks = []
        chunk_data = self._smart_split_text(page_content, config.max_chars, config.overlap_chars)
        
        for chunk_index, (chunk_text, start_pos, end_pos) in enumerate(chunk_data):
            chunk_id = self._generate_chunk_id(document.id, page_num, chunk_index)
            
            # Calculate actual overlap
            actual_overlap = 0
            if chunk_index > 0:
                prev_end = chunk_data[chunk_index - 1][2]
                if start_pos < prev_end:
                    actual_overlap = prev_end - start_pos
            
            chunk = Chunk(
                id=chunk_id,
                document_id=document.id,
                text=chunk_text.strip(),
                chunk_index=chunk_index,
                page=page_num,
                content_type="pdf",
                source=getattr(document, 'source', None) or getattr(document, 'url', None),
                url=getattr(document, 'url', None),
                metadata={
                    **getattr(document, 'metadata', {}),
                    "chunk_method": "page_split_chars",
                    "chunk_char_start": start_pos,
                    "chunk_char_end": end_pos,
                    "overlap_chars": actual_overlap,
                    "page_char_count": len(page_content),
                    "chunk_char_count": len(chunk_text.strip()),
                    "created_at": datetime.utcnow().isoformat(),
                }
            )
            chunks.append(chunk)
            
        return chunks
    
    def _smart_split_text(
        self, 
        text: str, 
        max_chars: int, 
        overlap_chars: int
    ) -> List[Tuple[str, int, int]]:
        """
        Smart text splitting with overlap, preferring natural break points.
        
        Returns:
            List of (chunk_text, start_pos, end_pos) tuples
        """
        if len(text) <= max_chars:
            return [(text, 0, len(text))]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Determine end position for this chunk
            end = min(start + max_chars, len(text))
            
            # If we're not at the end of text, try to find a good break point
            if end < len(text):
                end = self._find_best_split_point(text, start, end)
            
            chunk_text = text[start:end]
            chunks.append((chunk_text, start, end))
            
            # Move to next chunk with overlap
            if end >= len(text):
                break
                
            start = max(end - overlap_chars, start + 1)  # Ensure progress
            
        return chunks
    
    def _find_best_split_point(self, text: str, start: int, max_end: int) -> int:
        """Find the best split point preferring paragraph breaks, then line breaks, then spaces."""
        # Look for paragraph breaks (double newline)
        for i in range(max_end - 1, start + len(text) // 4, -1):
            if i + 1 < len(text) and text[i:i+2] == '\n\n':
                return i
        
        # Look for line breaks
        for i in range(max_end - 1, start + len(text) // 4, -1):
            if text[i] == '\n':
                return i + 1
        
        # Look for sentence endings
        for i in range(max_end - 1, start + len(text) // 4, -1):
            if text[i] in '.!?' and i + 1 < len(text) and text[i + 1] == ' ':
                return i + 1
        
        # Look for spaces
        for i in range(max_end - 1, start + len(text) // 4, -1):
            if text[i] == ' ':
                return i + 1
        
        # If no good break point found, split at max position
        return max_end
    
    def _generate_chunk_id(self, doc_id: str, page_num: int, chunk_index: int) -> str:
        """Generate stable, deterministic chunk ID."""
        # Use the existing pattern from the scripts
        return f"{doc_id}_page_{page_num:03d}_chunk_{chunk_index:03d}"
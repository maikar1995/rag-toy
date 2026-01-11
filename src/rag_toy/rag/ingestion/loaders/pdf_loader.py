"""PDF document loader that reads from pdf_pages.jsonl files."""

import json
import logging
from pathlib import Path
from typing import Iterator, Dict, Any

from ...models import Document

logger = logging.getLogger(__name__)

# Required fields for PDF documents
REQUIRED_PDF_FIELDS = ["doc_id", "content", "page"]


class PDFLoader:
    """Loader for PDF documents from pre-processed JSONL files."""
    
    def load(self, file_path: Path) -> Iterator[Document]:
        """
        Load PDF documents from pdf_pages.jsonl file.
        
        Expected JSONL format per line:
        {
            "doc_id": "document_id",
            "content": "page text content",
            "page": 1,
            "source_uri": "path/to/file.pdf",
            "metadata": { ... }
        }
        
        Args:
            file_path: Path to pdf_pages.jsonl file
            
        Yields:
            Document objects with content_type="pdf_page"
        """
        if not file_path.exists():
            raise FileNotFoundError(f"PDF JSONL file not found: {file_path}")
        
        logger.info(f"Loading PDF documents from {file_path}")
        
        with file_path.open('r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    document = self._parse_pdf_document(data, file_path, line_num)
                    if document:
                        yield document
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON at {file_path}:{line_num}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing {file_path}:{line_num}: {e}")
                    continue
    
    def _parse_pdf_document(self, data: Dict[str, Any], file_path: Path, line_num: int) -> Document:
        """
        Parse a single PDF document from JSONL data.
        
        Args:
            data: Parsed JSON data from JSONL line
            file_path: Source file path for error reporting
            line_num: Line number for error reporting
            
        Returns:
            Document object or None if should be skipped
        """
        # Validate required fields
        missing_fields = [field for field in REQUIRED_PDF_FIELDS if field not in data]
        if missing_fields:
            raise ValueError(
                f"Missing required PDF fields {missing_fields} at {file_path}:{line_num}"
            )
        
        # Skip empty content with warning
        content = data.get("content", "").strip()
        if not content:
            logger.warning(
                f"Skipping PDF document with empty content: {data.get('doc_id')} "
                f"at {file_path}:{line_num}"
            )
            return None
        
        # Create Document
        doc_id = data["doc_id"]
        page = data.get("page")
        source_uri = data.get("source_uri", str(file_path))
        metadata = data.get("metadata", {})
        
        return Document(
            id=doc_id,
            source=source_uri,
            content=content,
            content_type="pdf_page",
            source_uri=source_uri,
            page=page,
            metadata=metadata
        )
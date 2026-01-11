"""Web document loader that reads from web_docs.jsonl files."""

import json
import logging
from pathlib import Path
from typing import Iterator, Dict, Any

from ...models import Document

logger = logging.getLogger(__name__)

# Required fields for Web documents
REQUIRED_WEB_FIELDS = ["doc_id", "content"]


class WebLoader:
    """Loader for web documents from pre-processed JSONL files."""
    
    def load(self, file_path: Path) -> Iterator[Document]:
        """
        Load web documents from web_docs.jsonl file.
        
        Expected JSONL format per line:
        {
            "doc_id": "document_id",
            "content": "web page text content",
            "url": "https://example.com/page",
            "title": "Page Title",
            "source_uri": "https://example.com/page",
            "metadata": { ... }
        }
        
        Args:
            file_path: Path to web_docs.jsonl file
            
        Yields:
            Document objects with content_type="web_doc"
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Web JSONL file not found: {file_path}")
        
        logger.info(f"Loading web documents from {file_path}")
        
        with file_path.open('r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    document = self._parse_web_document(data, file_path, line_num)
                    if document:
                        yield document
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON at {file_path}:{line_num}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing {file_path}:{line_num}: {e}")
                    continue
    
    def _parse_web_document(self, data: Dict[str, Any], file_path: Path, line_num: int) -> Document:
        """
        Parse a single web document from JSONL data.
        
        Args:
            data: Parsed JSON data from JSONL line
            file_path: Source file path for error reporting
            line_num: Line number for error reporting
            
        Returns:
            Document object or None if should be skipped
        """
        # Validate required fields
        missing_fields = [field for field in REQUIRED_WEB_FIELDS if field not in data]
        if missing_fields:
            raise ValueError(
                f"Missing required web fields {missing_fields} at {file_path}:{line_num}"
            )
        
        # Skip empty content with warning
        content = data.get("content", "").strip()
        if not content:
            logger.warning(
                f"Skipping web document with empty content: {data.get('doc_id')} "
                f"at {file_path}:{line_num}"
            )
            return None
        
        # Create Document
        doc_id = data["doc_id"]
        url = data.get("url")
        title = data.get("title")
        source_uri = data.get("source_uri", url or str(file_path))
        metadata = data.get("metadata", {})
        
        return Document(
            id=doc_id,
            source=source_uri,
            title=title,
            url=url,
            content=content,
            content_type="web_doc",
            source_uri=source_uri,
            page=None,  # Web documents don't have pages
            metadata=metadata
        )
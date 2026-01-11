"""
PDFBookLoader: Loader for full-book PDF ingestion using LlamaIndex
"""
from typing import List
from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import UnstructuredReader
from .base import DocumentLoader
from ...models import Document
import logging
import os

logger = logging.getLogger(__name__)

class PDFBookLoader(DocumentLoader):
    """
    Loads a full PDF (book) as a single Document using LlamaIndex UnstructuredReader.
    Returns a list of Document objects compatible with the current domain.
    """
    def load(self, file_path: str) -> List[Document]:
        # Accepts either a single PDF file or a directory containing a PDF
        path = Path(file_path)
        if path.is_file() and path.suffix.lower() == ".pdf":
            input_dir = path.parent
            file_name = path.name
        elif path.is_dir():
            input_dir = path
            file_name = None
        else:
            raise ValueError(f"Invalid PDF path: {file_path}")

        file_extractor = {".pdf": UnstructuredReader()}
        reader = SimpleDirectoryReader(input_dir=str(input_dir), file_extractor=file_extractor)
        documents = reader.load_data()

        # Optionally filter for a specific file if needed
        if file_name:
            documents = [doc for doc in documents if getattr(doc, 'metadata', {}).get('file_name', '') == file_name]

        result = []
        for i, doc in enumerate(documents):
            # Compose Document compatible with current domain
            content = doc.text if hasattr(doc, 'text') else str(doc)
            metadata = getattr(doc, 'metadata', {}) or {}
            
            # Generate document ID from file path
            doc_id = metadata.get('file_name', f"pdf_book_{i}")
            if doc_id.endswith('.pdf'):
                doc_id = doc_id[:-4]  # Remove .pdf extension
            
            result.append(Document(
                id=doc_id,
                source=metadata.get('file_path', file_path),
                content=content,
                content_type="pdf_book",
                source_uri=metadata.get('file_path', file_path),
                page=None,  # Not paginated
                metadata={
                    **metadata,
                    "loader": "pdf_book",
                }
            ))
        logger.info(f"PDFBookLoader loaded {len(result)} document(s) from {file_path}")
        return result

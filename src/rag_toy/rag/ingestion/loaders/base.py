"""Base protocol for document loaders."""

from typing import Protocol, Iterator
from pathlib import Path

from ...models import Document


class DocumentLoader(Protocol):
    """Protocol for document loaders that convert JSONL files to Document objects."""
    
    def load(self, file_path: Path) -> Iterator[Document]:
        """
        Load documents from a JSONL file.
        
        Args:
            file_path: Path to the JSONL file to load
            
        Yields:
            Document objects parsed from the JSONL file
            
        Raises:
            FileNotFoundError: If file_path does not exist
            ValueError: If required fields are missing
        """
        ...
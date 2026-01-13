"""PDF document loader using LlamaParse for advanced parsing (tables, charts, captions)."""

import logging
import os
from pathlib import Path
from typing import Iterator

from ...models import Document

logger = logging.getLogger(__name__)

class LlamaParsePDFLoader:
    """Loader for PDFs using LlamaParse, outputs Document objects with content_type and required metadata."""
    
    def load(self, file_path: Path) -> Iterator[Document]:
        """
        Load PDF documents using LlamaParse.
        Args:
            file_path: Path to the PDF file
        Yields:
            Document objects with content_type in {"pdf_text", "pdf_table", "figure_caption"}
        """
        file_path = Path(file_path)  # Ensure file_path is a Path object
        try:
            from llama_parse import LlamaParse
            from llama_index.readers.pdf_table import PDFTableReader 
        except ImportError as e:
            logger.error("llama-index and plugins must be installed: %s", e)
            raise

        # 1. Parse with LlamaParse (text, tables, captions)
        parser = LlamaParse(
            result_type="markdown",
            api_key=os.environ["LLAMA_API_KEY"],
            use_vendor_multimodal_model=True,
            system_prompt_append=(
                "Extrae todo el contenido del PDF. "
                "Cuando haya tablas o gráficos, conviértelos a tablas Markdown "
                "con encabezados claros y unidades si aparecen en el eje."
            ),
        )
        parsed_docs = parser.load_data(str(file_path))

        # 2. (Optional) Extract tables with PDFTableReader
        table_reader = PDFTableReader()
        table_docs = table_reader.load_data(file=str(file_path))

        # 3. Combine and yield as Document objects
        all_docs = []
        all_docs.extend(parsed_docs)
        all_docs.extend(table_docs)

        doc_id = file_path.stem
        source_uri = str(file_path)

        for doc in all_docs:
            # Determine content_type
            if hasattr(doc, 'metadata') and doc.metadata:
                meta = doc.metadata
                if meta.get('type') == 'table':
                    content_type = "pdf_table"
                elif meta.get('type') == 'caption':
                    content_type = "figure_caption"
                else:
                    content_type = "pdf_text"
            else:
                content_type = "pdf_text"

            # Required metadata
            page = getattr(doc, 'page_label', None) or getattr(doc, 'page', None)
            # Only include table_id/figure_id if present
            table_id = getattr(doc, 'table_id', None) if hasattr(doc, 'table_id') else None
            figure_id = getattr(doc, 'figure_id', None) if hasattr(doc, 'figure_id') else None
            metadata = dict(doc.metadata) if hasattr(doc, 'metadata') and doc.metadata else {}
            if table_id:
                metadata['table_id'] = table_id
            if figure_id:
                metadata['figure_id'] = figure_id

            yield Document(
                id=doc_id,
                source=source_uri,
                content=getattr(doc, 'text', None) or getattr(doc, 'content', None),
                content_type=content_type,
                source_uri=source_uri,
                page=page,
                metadata=metadata
            )

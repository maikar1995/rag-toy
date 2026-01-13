import os
from pathlib import Path
from rag_toy.rag.ingestion.loaders import get_loader
from rag_toy.rag.ingestion.chunking import get_chunker
from rag_toy.rag.ingestion.chunking.base import ChunkerConfig

def test_llamaparse_table_chunking(tmp_path):
    # Create a fake LlamaParse output JSONL file with a table
    pdf_path = tmp_path / "foo.pdf"
    pdf_path.write_text("fake-pdf-content")  # Just to have a file

    # Simulate loader output (normally would use LlamaParse, here we mock)
    from rag_toy.rag.models import Document
    doc = Document(
        id="foo",
        source=str(pdf_path),
        content="| Col1 | Col2 |\n|------|------|\n|  1   |  2   |",
        content_type="pdf_table",
        source_uri=str(pdf_path),
        page=1,
        metadata={}
    )

    # Use the LlamaParsePDFChunker
    chunker = get_chunker(engine="llamaparse", doc_type="pdf_llamaparse", config=ChunkerConfig())
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1, "Table should be a single chunk"
    chunk = chunks[0]
    assert chunk.content_type == "pdf_table"
    assert chunk.page == 1

#!/usr/bin/env python3
"""
Example: Run complete ingestion pipeline
"""

import logging
from pathlib import Path

from src.rag_toy.rag.ingestion.orchestrator import run_ingestion_with_env

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Data paths for JSONL files
    data_paths = {
        # "pdf": "data/processed/pdf_pages.jsonl",
        # "web": "data/processed/web_docs.jsonl",
        "pdf_book": "data/input/_OceanofPDF.com_The_Minto_Pyramid_Principle_-_Barbara_Minto.pdf"
    }
    
    # Run ingestion with environment config
    summary = run_ingestion_with_env(
        data_paths=data_paths,
        chunk_engine="llamaindex",  # or "langchain", "llamaindex"
        chunker_kwargs={
            "chunk_sizes": [1024]
        }
    )
    
    # Print results
    print(f"✅ Ingestion completed!")
    print(f"📄 Documents processed: {summary.total_documents}")
    print(f"🧩 Chunks generated: {summary.total_chunks}")
    print(f"📊 Chunks indexed: {summary.indexed_chunks}")
    print(f"🔍 Index used: {summary.index_name}")
    
    if summary.failed_chunks > 0:
        print(f"⚠️  Failed chunks: {summary.failed_chunks}")
        for error in summary.errors[:3]:  # Show first 3 errors
            print(f"   - {error}")
    
    print(f"⏱️  Total time: {summary.processing_time:.2f}s")

if __name__ == "__main__":
    main()
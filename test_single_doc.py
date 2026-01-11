#!/usr/bin/env python3
"""
Test ingestion with a single document
"""

import logging
import json
import tempfile
from pathlib import Path

from src.rag_toy.rag.ingestion.orchestrator import run_ingestion_with_env

def create_test_document():
    """Create a minimal test document in JSONL format."""
    test_doc = {
        "content": "This is a test document for verifying the ingestion pipeline. It contains sample text that should be chunked and indexed properly.",
        "content_type": "test_doc",
        "source_uri": "test://example.com/test-doc",
        "page": 1,
        "metadata": {
            "doc_hash": "test123",
            "fetched_at": "2026-01-11T10:00:00Z",
            "ingested_at": "2026-01-11T10:05:00Z",
            "chunk_method": "test"
        }
    }
    return test_doc

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        test_doc = create_test_document()
        json.dump(test_doc, f)
        f.write('\n')
        temp_file = f.name
    
    try:
        print("🧪 Testing ingestion with single document...")
        print(f"📄 Test document: {test_doc['content'][:50]}...")
        
        # Create data paths with single test document
        data_paths = {
            "web": temp_file  # Use web loader for simple test
        }
        
        # Run ingestion
        summary = run_ingestion_with_env(
            data_paths=data_paths,
            chunk_engine="native"
        )
        
        # Print detailed results
        print("\n" + "="*50)
        print("🎯 SINGLE DOCUMENT TEST RESULTS")
        print("="*50)
        print(f"✅ Success: {summary.success}")
        print(f"📄 Documents: {summary.total_documents}")
        print(f"🧩 Chunks generated: {summary.total_chunks}")
        print(f"📊 Chunks indexed: {summary.indexed_chunks}")
        print(f"❌ Failed chunks: {summary.failed_chunks}")
        print(f"🔍 Index: {summary.index_name}")
        print(f"⏱️  Processing time: {summary.processing_time:.2f}s")
        
        if summary.errors:
            print(f"\n⚠️  Errors ({len(summary.errors)}):")
            for error in summary.errors:
                print(f"   - {error}")
        
        print("="*50)
        
        return 0 if summary.success else 1
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Clean up temp file
        Path(temp_file).unlink()

if __name__ == "__main__":
    exit(main())
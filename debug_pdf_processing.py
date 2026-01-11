"""Debug script para identificar dónde fallan los PDFs."""

import logging
from pathlib import Path
from src.rag_toy.rag.ingestion.loaders import get_loader
from src.rag_toy.rag.ingestion.chunking import get_chunker

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_pdf_processing():
    """Debug paso a paso del procesamiento de PDFs."""
    
    # 1. Cargar documentos PDF
    print("🔍 Step 1: Loading PDF documents...")
    loader = get_loader("pdf")
    pdf_path = "data/processed/pdf_pages.jsonl"  # Ajusta el path
    
    if not Path(pdf_path).exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    documents = []
    doc_count = 0
    try:
        for doc in loader.load(pdf_path):
            documents.append(doc)
            doc_count += 1
            if doc_count <= 5:  # Mostrar primeros 5
                print(f"  📄 Loaded: {doc.id}, page: {getattr(doc, 'page', 'N/A')}")
            if doc_count == 10:  # Solo procesar primeros 10 para debug
                break
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        return
    
    print(f"✅ Loaded {len(documents)} documents")
    
    # 2. Probar chunker
    print("\n🔍 Step 2: Testing chunker...")
    try:
        chunker = get_chunker(engine="native", doc_type="pdf")
        print(f"✅ Chunker created: {type(chunker).__name__}")
    except Exception as e:
        print(f"❌ Error creating chunker: {e}")
        return
    
    successful_docs = 0
    failed_docs = 0
    
    for i, doc in enumerate(documents):
        try:
            print(f"\n📄 Processing doc {i+1}: {doc.id}")
            print(f"   Page: {getattr(doc, 'page', 'N/A')}")
            print(f"   Content length: {len(getattr(doc, 'content', ''))}")
            
            chunks = list(chunker.chunk(doc))
            print(f"   ✅ Generated {len(chunks)} chunks")
            
            if chunks:
                print(f"   📝 First chunk ID: {chunks[0].id}")
                print(f"   📝 First chunk length: {len(chunks[0].text)}")
                print(f"   📝 Chunk page: {getattr(chunks[0], 'page', 'N/A')}")
            
            successful_docs += 1
            
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            logger.error(f"Chunking failed for {doc.id}: {e}", exc_info=True)
            failed_docs += 1
    
    print(f"\n📊 Results:")
    print(f"   ✅ Successful: {successful_docs}")
    print(f"   ❌ Failed: {failed_docs}")

if __name__ == "__main__":
    debug_pdf_processing()
"""
Debug script to query Azure Search index and analyze chunk IDs.
"""
import os
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from collections import Counter
import json

load_dotenv()

def create_search_client(index_name: str) -> SearchClient:
    """Create Azure Search client."""
    return SearchClient(
        endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'),
        index_name=index_name,
        credential=AzureKeyCredential(os.getenv('AZURE_INSERTION_KEY'))
    )

def analyze_index_ids(index_name: str = "rag-toy-native-v1"):
    """Analyze all document IDs in the index."""
    client = create_search_client(index_name)
    
    print(f"🔍 Analyzing index: {index_name}")
    print("=" * 60)
    
    try:
        # Query all documents, only return id and related fields
        results = client.search(
            search_text="*",
            select=["id", "doc_id", "page", "chunk_id"],
            top=1000,  # Adjust if you have more than 1000 chunks
            include_total_count=True
        )
        
        # Collect data
        all_docs = []
        for result in results:
            all_docs.append({
                'id': result.get('id'),
                'doc_id': result.get('doc_id'), 
                'page': result.get('page'),
                'chunk_id': result.get('chunk_id')
            })
        
        total_count = results.get_count() or len(all_docs)
        
        print(f"📊 Total documents in index: {total_count}")
        print(f"📊 Documents retrieved: {len(all_docs)}")
        
        # Analyze IDs
        analyze_id_patterns(all_docs)
        
        # Check for duplicates
        check_duplicates(all_docs)
        
        # Analyze by doc_id and page
        analyze_by_document(all_docs)
        
    except Exception as e:
        print(f"❌ Error querying index: {e}")

def analyze_id_patterns(docs):
    """Analyze ID patterns."""
    print("\n🔍 ID PATTERNS ANALYSIS")
    print("-" * 40)
    
    id_patterns = Counter()
    chunk_id_patterns = Counter()
    
    for doc in docs[:10]:  # Show first 10 as examples
        print(f"ID: {doc['id']}")
        print(f"  chunk_id: {doc['chunk_id']}")
        print(f"  doc_id: {doc['doc_id']}")
        print(f"  page: {doc['page']}")
        print()
    
    # Extract patterns
    for doc in docs:
        if doc['id']:
            # Count doc_id patterns
            if doc['doc_id']:
                id_patterns[doc['doc_id']] += 1
            
            # Count chunk_id patterns
            if doc['chunk_id']:
                # Extract base pattern (before _chunk_)
                chunk_id = doc['chunk_id']
                if '_chunk_' in chunk_id:
                    base_pattern = chunk_id.split('_chunk_')[0]
                    chunk_id_patterns[base_pattern] += 1
    
    print(f"\n📈 Documents by doc_id (top 10):")
    for doc_id, count in id_patterns.most_common(10):
        print(f"  {doc_id}: {count} chunks")
    
    print(f"\n📈 Documents by page pattern (top 10):")
    for pattern, count in chunk_id_patterns.most_common(10):
        print(f"  {pattern}: {count} chunks")

def check_duplicates(docs):
    """Check for duplicate IDs."""
    print("\n🔍 DUPLICATE CHECK")
    print("-" * 40)
    
    # Check Azure Search ID duplicates (shouldn't exist)
    search_ids = [doc['id'] for doc in docs if doc['id']]
    search_id_counts = Counter(search_ids)
    duplicates = {id_: count for id_, count in search_id_counts.items() if count > 1}
    
    if duplicates:
        print(f"❌ Found {len(duplicates)} duplicate Azure Search IDs:")
        for id_, count in list(duplicates.items())[:5]:
            print(f"  {id_}: {count} times")
    else:
        print("✅ No duplicate Azure Search IDs found")
    
    # Check original chunk_id duplicates
    chunk_ids = [doc['chunk_id'] for doc in docs if doc['chunk_id']]
    chunk_id_counts = Counter(chunk_ids)
    chunk_duplicates = {id_: count for id_, count in chunk_id_counts.items() if count > 1}
    
    if chunk_duplicates:
        print(f"❌ Found {len(chunk_duplicates)} duplicate chunk_ids:")
        for id_, count in list(chunk_duplicates.items())[:5]:
            print(f"  {id_}: {count} times")
    else:
        print("✅ No duplicate chunk_ids found")

def analyze_by_document(docs):
    """Analyze chunks by document and page."""
    print("\n🔍 DOCUMENT & PAGE ANALYSIS") 
    print("-" * 40)
    
    # Group by doc_id
    by_doc = {}
    for doc in docs:
        doc_id = doc['doc_id']
        if doc_id not in by_doc:
            by_doc[doc_id] = []
        by_doc[doc_id].append(doc)
    
    print(f"📚 Unique documents: {len(by_doc)}")
    
    # Analyze each document
    for doc_id, doc_chunks in list(by_doc.items())[:5]:  # Show first 5 documents
        print(f"\n📄 Document: {doc_id}")
        print(f"   Chunks: {len(doc_chunks)}")
        
        # Group by page
        pages = {}
        for chunk in doc_chunks:
            page = chunk['page']
            if page not in pages:
                pages[page] = []
            pages[page].append(chunk)
        
        print(f"   Pages: {len(pages)} -> {sorted(pages.keys())}")
        
        # Show chunk distribution by page
        for page, page_chunks in sorted(pages.items()):
            chunk_ids = [c['chunk_id'] for c in page_chunks]
            print(f"     Page {page}: {len(page_chunks)} chunks")
            if len(page_chunks) <= 3:  # Show chunk IDs if few
                for chunk_id in chunk_ids:
                    print(f"       - {chunk_id}")

def export_to_json(docs, filename="index_analysis.json"):
    """Export results to JSON for further analysis."""
    print(f"\n💾 Exporting data to {filename}")
    with open(filename, 'w') as f:
        json.dump(docs, f, indent=2)
    print(f"✅ Exported {len(docs)} documents")

if __name__ == "__main__":
    analyze_index_ids()
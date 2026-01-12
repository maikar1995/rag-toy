"""
Hierarchical Retrieval Demo

Demonstrates the new optimized hierarchical retrieval system with:
- Performance optimizations (batching, caching, timeouts)
- Graceful degradation for robustness
- Backward compatibility with existing flat chunks
- A/B testing support via RETRIEVAL_MODE environment variable

Usage:
    # Flat mode (v1 compatibility)
    RETRIEVAL_MODE=flat python hierarchical_demo.py

    # Hierarchical mode (v2 with optimizations)
    RETRIEVAL_MODE=hierarchical python hierarchical_demo.py
"""

import os
import logging
from dotenv import load_dotenv

from rag_toy.rag.utils.client_setup import create_search_client, create_embeddings_client
from rag_toy.rag.retrieval.factory import create_retriever, is_hierarchical_mode, get_supported_modes

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_retrieval_modes():
    """Compare flat vs hierarchical retrieval modes."""
    
    load_dotenv()
    
    # Create clients
    search_client = create_search_client()
    embeddings_client = create_embeddings_client()
    
    # Test query
    query = "What are the main features of machine learning?"
    
    print("🔍 Hierarchical Retrieval System Demo")
    print("=" * 50)
    print(f"Available modes: {get_supported_modes()}")
    print(f"Current mode: {os.getenv('RETRIEVAL_MODE', 'flat')}")
    print(f"Is hierarchical: {is_hierarchical_mode()}")
    print()
    
    # Create retriever based on environment
    retriever = create_retriever(
        search_client=search_client,
        embeddings_client=embeddings_client,
        search_type="hybrid"
    )
    
    print(f"📊 Retriever type: {type(retriever).__name__}")
    print(f"🔍 Query: '{query}'")
    print()
    
    try:
        # Perform search
        results = retriever.search(query, top_k=5)
        
        print(f"📋 Results: {len(results)} documents found")
        print("-" * 40)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result.score:.3f}")
            print(f"   ID: {result.chunk_id}")
            print(f"   Doc: {result.doc_id}")
            print(f"   Type: {result.search_type}")
            
            # Show hierarchical metadata if present
            if result.node_id or result.parent_path or result.node_type:
                print(f"   🌳 Hierarchical:")
                print(f"      Node ID: {result.node_id}")
                print(f"      Node Type: {result.node_type}")
                print(f"      Parent Path: {result.parent_path}")
            
            # Preview content
            content_preview = result.content[:100] + "..." if len(result.content) > 100 else result.content
            print(f"   Content: {content_preview}")
        
        # Show cache stats if hierarchical retriever
        from rag_toy.rag.retrieval.hierarchical_retriever import HierarchicalAzureRetriever
        if isinstance(retriever, HierarchicalAzureRetriever):
            cache_stats = retriever.get_cache_stats()
            print(f"\n📈 Cache Stats:")
            print(f"   Total entries: {cache_stats['total_entries']}")
            print(f"   Valid entries: {cache_stats['valid_entries']}")
            print(f"   Expired entries: {cache_stats['expired_entries']}")
            print(f"   Hit rate: {cache_stats['hit_rate']:.2%}")
    
    except Exception as e:
        print(f"❌ Error during search: {e}")
        logger.exception("Search failed")

def demo_mode_comparison():
    """Compare results between flat and hierarchical modes."""
    
    load_dotenv()
    
    search_client = create_search_client()
    embeddings_client = create_embeddings_client()
    
    query = "explain neural networks"
    
    print("🆚 Mode Comparison Demo")
    print("=" * 30)
    
    modes = ["flat", "hierarchical"]
    results_by_mode = {}
    
    for mode in modes:
        print(f"\n🔍 Testing {mode.upper()} mode...")
        
        retriever = create_retriever(
            search_client=search_client,
            embeddings_client=embeddings_client,
            search_type="hybrid",
            retrieval_mode=mode
        )
        
        try:
            results = retriever.search(query, top_k=3)
            results_by_mode[mode] = results
            
            print(f"   ✅ Found {len(results)} results")
            for i, result in enumerate(results[:2], 1):  # Show top 2
                hierarchical_info = ""
                if result.node_type:
                    hierarchical_info = f" [{result.node_type}]"
                print(f"   {i}. {result.chunk_id}{hierarchical_info} (score: {result.score:.3f})")
        
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results_by_mode[mode] = []
    
    # Compare results
    print(f"\n📊 Comparison Summary:")
    for mode, results in results_by_mode.items():
        print(f"   {mode.capitalize()}: {len(results)} results")
        if results and hasattr(results[0], 'node_type'):
            hierarchical_count = sum(1 for r in results if r.node_type)
            print(f"      Hierarchical: {hierarchical_count}/{len(results)}")

def demo_error_resilience():
    """Demonstrate error handling and graceful degradation."""
    
    print("🛡️  Error Resilience Demo")
    print("=" * 30)
    
    # Test with invalid configuration
    with_invalid_config = {
        'RETRIEVAL_MODE': 'hierarchical',
        'PARENT_LOOKUP_TIMEOUT': '0.1',  # Very short timeout
        'MAX_PARENT_IDS_PER_LOOKUP': '1'  # Very small batch
    }
    
    print("Testing with stress configuration...")
    
    try:
        # Temporarily set stress config
        original_env = {key: os.environ.get(key) for key in with_invalid_config.keys()}
        os.environ.update(with_invalid_config)
        
        load_dotenv()
        search_client = create_search_client()
        embeddings_client = create_embeddings_client()
        
        retriever = create_retriever(
            search_client=search_client,
            embeddings_client=embeddings_client,
            search_type="hybrid"
        )
        
        # This should work despite stress configuration
        results = retriever.search("test query", top_k=2)
        print(f"✅ Stress test passed: {len(results)} results")
        
    except Exception as e:
        print(f"⚠️  Stress test degraded gracefully: {e}")
    
    finally:
        # Restore original environment
        for key, value in original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]

if __name__ == "__main__":
    print("🚀 Starting Hierarchical Retrieval Demo\n")
    
    try:
        # Main demo
        demo_retrieval_modes()
        
        print("\n" + "="*60 + "\n")
        
        # Mode comparison
        demo_mode_comparison()
        
        print("\n" + "="*60 + "\n")
        
        # Error resilience
        demo_error_resilience()
        
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        logger.exception("Demo error")
    
    print("\n🎯 Demo complete! Set RETRIEVAL_MODE=hierarchical for production use.")
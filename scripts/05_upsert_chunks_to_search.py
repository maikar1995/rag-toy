#!/usr/bin/env python3
"""
Upsert Chunks to Azure AI Search

Processes chunks.jsonl: generates embeddings → upserts to Azure Search
Input: data/processed/chunks.jsonl  
Output: indexed documents + checkpoint file

Features:
- Streaming batch processing with checkpoints
- Exponential backoff retry policies
- Split-on-failure for problematic batches
- Resume from checkpoint support
"""

import json
import logging
import argparse
import os
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError
import requests

# Load environment variables
load_dotenv()

# Configuration constants
EMBED_BATCH_SIZE = 16
UPLOAD_BATCH_SIZE = 200
MAX_RETRIES_EMBEDDING = 6
MAX_RETRIES_SEARCH = 4
BASE_DELAY = 1.0


@dataclass
class CheckpointState:
    """Checkpoint state for resuming processing."""
    last_processed_line: int = 0
    successful_upserts: int = 0
    failed_chunk_ids: List[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if self.failed_chunk_ids is None:
            self.failed_chunk_ids = []


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_azure_configs() -> Tuple[Dict[str, str], Dict[str, str], str]:
    """Load Azure OpenAI and Search configurations."""
    
    # Azure OpenAI config
    openai_config = {
        'endpoint': os.getenv('EMBEDDING_1_ENDPOINT'),
        'api_key': os.getenv('EMBEDDING_1_API_KEY'),
        'api_version': os.getenv('EMBEDDING_1_API_VERSION', '2024-02-01'),
        'model_name': os.getenv('EMBEDDING_1_MODEL_NAME', 'text-embedding-3-small'),
        'deployment': os.getenv('EMBEDDING_1_DEPLOYMENT')
    }
    
    # Azure Search config  
    search_config = {
        'endpoint': os.getenv('AZURE_SEARCH_ENDPOINT'),
        'api_key': os.getenv('AZURE_INSERTION_KEY'),
        'api_version': os.getenv('AZURE_SEARCH_API_VERSION', '2023-11-01')
    }
    
    # Index name (try to get from configs or use default)
    index_name = os.getenv('AZURE_SEARCH_INDEX_NAME', 'rag-toy-index-v1')
    
    # Validate required configs
    for name, config in [("OpenAI", openai_config), ("Search", search_config)]:
        missing = [k for k, v in config.items() if not v and k != 'deployment']
        if missing:
            raise ValueError(f"Missing {name} environment variables: {missing}")
    
    return openai_config, search_config, index_name


def exponential_backoff_with_jitter(attempt: int, base_delay: float = BASE_DELAY, max_delay: float = 60.0) -> float:
    """Calculate delay with exponential backoff and jitter."""
    delay = min(max_delay, base_delay * (2 ** attempt))
    jitter = random.uniform(0, 0.25 * delay)
    return delay + jitter


def retry_with_backoff(func, max_retries: int, operation_name: str, *args, **kwargs):
    """Execute function with exponential backoff retry logic."""
    
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
            
        except Exception as e:
            # Determine if we should retry
            should_retry = False
            
            if hasattr(e, 'status_code'):
                status_code = e.status_code
                should_retry = status_code in [429, 500, 502, 503, 504]
            elif "timeout" in str(e).lower() or "connection" in str(e).lower():
                should_retry = True
            
            if attempt == max_retries or not should_retry:
                logging.error(f"❌ {operation_name} failed after {attempt + 1} attempts: {e}")
                raise e
            
            delay = exponential_backoff_with_jitter(attempt)
            logging.warning(f"⚠️  {operation_name} failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.2f}s: {e}")
            time.sleep(delay)
    
    raise Exception(f"{operation_name} exhausted all retries")


def create_azure_openai_client(config: Dict[str, str]) -> AzureOpenAI:
    """Create Azure OpenAI client."""
    return AzureOpenAI(
        azure_endpoint=config['endpoint'],
        api_key=config['api_key'],
        api_version=config['api_version']
    )


def create_search_client(config: Dict[str, str], index_name: str) -> SearchClient:
    """Create Azure Search client."""
    return SearchClient(
        endpoint=config['endpoint'],
        index_name=index_name,
        credential=AzureKeyCredential(config['api_key'])
    )


def generate_embeddings_batch(client: AzureOpenAI, texts: List[str], model_deployment: str) -> List[List[float]]:
    """Generate embeddings for a batch of texts with retry logic."""
    
    def _generate():
        response = client.embeddings.create(
            model=model_deployment,
            input=texts
        )
        return [embedding.embedding for embedding in response.data]
    
    return retry_with_backoff(
        _generate,
        MAX_RETRIES_EMBEDDING,
        f"Generate embeddings (batch size: {len(texts)})"
    )


def map_chunk_to_search_document(chunk: Dict[str, Any], embedding: List[float]) -> Dict[str, Any]:
    """Map chunk from JSONL to Azure Search document format."""
    metadata = chunk.get('metadata', {})
    
    # Extract content_type from metadata or default to "text"
    content_type = metadata.get('content_type', 'text')
    
    return {
        "id": chunk['chunk_id'],
        "content": chunk['content'],
        "contentVector": embedding,
        "doc_id": chunk['doc_id'],
        "page": chunk.get('page'),
        "content_type": content_type,
        "source_path": metadata.get('source_path', ''),
        "doc_hash": metadata.get('doc_hash', '')
    }


def upsert_documents_batch(search_client: SearchClient, documents: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
    """Upsert batch of documents to Azure Search with retry logic."""
    
    def _upsert():
        result = search_client.merge_or_upload_documents(documents=documents)
        return result
    
    try:
        results = retry_with_backoff(
            _upsert,
            MAX_RETRIES_SEARCH,
            f"Upsert documents (batch size: {len(documents)})"
        )
        
        # Process results
        successful_count = 0
        failed_ids = []
        
        for result in results:
            if result.succeeded:
                successful_count += 1
            else:
                failed_ids.append(result.key)
                logging.error(f"Failed to upsert document {result.key}: {result.error_message}")
        
        return successful_count, failed_ids
        
    except Exception as e:
        # If entire batch fails, return all as failed
        failed_ids = [doc['id'] for doc in documents]
        logging.error(f"Entire batch failed: {e}")
        return 0, failed_ids


def split_on_failure(search_client: SearchClient, documents: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
    """Handle failed batch by splitting it into smaller pieces."""
    
    if len(documents) == 1:
        # Single document failed, nothing to split
        return 0, [documents[0]['id']]
    
    logging.info(f"Splitting batch of {len(documents)} documents into smaller batches")
    
    # Split into two halves
    mid = len(documents) // 2
    batch1 = documents[:mid]
    batch2 = documents[mid:]
    
    total_successful = 0
    total_failed = []
    
    # Process first half
    try:
        success1, failed1 = upsert_documents_batch(search_client, batch1)
        total_successful += success1
        
        # If first half still fails, split it further
        if failed1 and len(batch1) > 1:
            subsuccess, subfailed = split_on_failure(search_client, 
                [doc for doc in batch1 if doc['id'] in failed1])
            total_successful += subsuccess
            total_failed.extend(subfailed)
        else:
            total_failed.extend(failed1)
            
    except Exception as e:
        logging.error(f"First half failed completely: {e}")
        total_failed.extend([doc['id'] for doc in batch1])
    
    # Process second half
    try:
        success2, failed2 = upsert_documents_batch(search_client, batch2)
        total_successful += success2
        
        # If second half still fails, split it further
        if failed2 and len(batch2) > 1:
            subsuccess, subfailed = split_on_failure(search_client,
                [doc for doc in batch2 if doc['id'] in failed2])
            total_successful += subsuccess  
            total_failed.extend(subfailed)
        else:
            total_failed.extend(failed2)
            
    except Exception as e:
        logging.error(f"Second half failed completely: {e}")
        total_failed.extend([doc['id'] for doc in batch2])
    
    return total_successful, total_failed


def load_checkpoint(checkpoint_file: str) -> CheckpointState:
    """Load checkpoint from file."""
    if not os.path.exists(checkpoint_file):
        return CheckpointState()
    
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        checkpoint = CheckpointState(
            last_processed_line=data.get('last_processed_line', 0),
            successful_upserts=data.get('successful_upserts', 0),
            failed_chunk_ids=data.get('failed_chunk_ids', []),
            timestamp=data.get('timestamp', '')
        )
        
        logging.info(f"📄 Loaded checkpoint: processed {checkpoint.last_processed_line} lines, "
                    f"{checkpoint.successful_upserts} successful upserts, "
                    f"{len(checkpoint.failed_chunk_ids)} failed chunks")
        
        return checkpoint
        
    except Exception as e:
        logging.warning(f"Failed to load checkpoint {checkpoint_file}: {e}")
        return CheckpointState()


def save_checkpoint(checkpoint: CheckpointState, checkpoint_file: str):
    """Save checkpoint to file."""
    try:
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        
        checkpoint_data = {
            'last_processed_line': checkpoint.last_processed_line,
            'successful_upserts': checkpoint.successful_upserts,
            'failed_chunk_ids': checkpoint.failed_chunk_ids,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")


def process_chunks_file(
    chunks_file: str,
    openai_client: AzureOpenAI,
    search_client: SearchClient,
    model_deployment: str,
    checkpoint_file: str,
    embed_batch_size: int = EMBED_BATCH_SIZE,
    upload_batch_size: int = UPLOAD_BATCH_SIZE
):
    """Process chunks file with streaming batch processing."""
    
    checkpoint = load_checkpoint(checkpoint_file)
    
    if not os.path.exists(chunks_file):
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
    
    # Count total lines for progress tracking
    with open(chunks_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    logging.info(f"📊 Processing {chunks_file}")
    logging.info(f"Total lines: {total_lines}, Starting from line: {checkpoint.last_processed_line + 1}")
    logging.info(f"Batch sizes: embed={embed_batch_size}, upload={upload_batch_size}")
    
    current_batch = []
    line_number = 0
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_number += 1
            
            # Skip lines already processed
            if line_number <= checkpoint.last_processed_line:
                continue
            
            try:
                chunk = json.loads(line.strip())
                current_batch.append((chunk, line_number))
                
                # Process batch when it reaches upload_batch_size
                if len(current_batch) >= upload_batch_size:
                    success_count, failed_ids = process_batch(
                        current_batch, openai_client, search_client, 
                        model_deployment, embed_batch_size
                    )
                    
                    # Update checkpoint
                    checkpoint.last_processed_line = current_batch[-1][1]
                    checkpoint.successful_upserts += success_count
                    checkpoint.failed_chunk_ids.extend(failed_ids)
                    
                    save_checkpoint(checkpoint, checkpoint_file)
                    
                    # Log progress
                    progress = (checkpoint.last_processed_line / total_lines) * 100
                    logging.info(f"📈 Progress: {progress:.1f}% ({checkpoint.last_processed_line}/{total_lines}), "
                                f"Successful: {checkpoint.successful_upserts}, Failed: {len(checkpoint.failed_chunk_ids)}")
                    
                    current_batch = []
                    
            except json.JSONDecodeError as e:
                logging.error(f"Invalid JSON on line {line_number}: {e}")
                continue
                
        # Process remaining batch
        if current_batch:
            success_count, failed_ids = process_batch(
                current_batch, openai_client, search_client,
                model_deployment, embed_batch_size
            )
            
            checkpoint.last_processed_line = current_batch[-1][1]
            checkpoint.successful_upserts += success_count
            checkpoint.failed_chunk_ids.extend(failed_ids)
            
            save_checkpoint(checkpoint, checkpoint_file)
    
    # Final summary
    logging.info("=" * 60)
    logging.info("🎉 PROCESSING COMPLETE")
    logging.info(f"Total lines processed: {checkpoint.last_processed_line}")
    logging.info(f"Successful upserts: {checkpoint.successful_upserts}")
    logging.info(f"Failed chunks: {len(checkpoint.failed_chunk_ids)}")
    if checkpoint.failed_chunk_ids:
        logging.info(f"Failed chunk IDs: {checkpoint.failed_chunk_ids[:10]}...")
    logging.info("=" * 60)


def process_batch(
    batch_with_lines: List[Tuple[Dict[str, Any], int]],
    openai_client: AzureOpenAI,
    search_client: SearchClient,
    model_deployment: str,
    embed_batch_size: int
) -> Tuple[int, List[str]]:
    """Process a single batch of chunks."""
    
    chunks = [item[0] for item in batch_with_lines]
    
    try:
        # Extract texts for embedding
        texts = [chunk['content'] for chunk in chunks]
        
        # Generate embeddings in sub-batches
        all_embeddings = []
        for i in range(0, len(texts), embed_batch_size):
            sub_batch = texts[i:i + embed_batch_size]
            sub_embeddings = generate_embeddings_batch(openai_client, sub_batch, model_deployment)
            all_embeddings.extend(sub_embeddings)
        
        # Map to search documents
        search_documents = []
        for chunk, embedding in zip(chunks, all_embeddings):
            search_doc = map_chunk_to_search_document(chunk, embedding)
            search_documents.append(search_doc)
        
        # Upsert to Azure Search
        success_count, failed_ids = upsert_documents_batch(search_client, search_documents)
        
        # If some documents failed, try split-on-failure
        if failed_ids and len(search_documents) > 1:
            failed_docs = [doc for doc in search_documents if doc['id'] in failed_ids]
            retry_success, retry_failed = split_on_failure(search_client, failed_docs)
            success_count += retry_success
            failed_ids = retry_failed
        
        return success_count, failed_ids
        
    except Exception as e:
        logging.error(f"Batch processing failed: {e}")
        failed_ids = [chunk['chunk_id'] for chunk in chunks]
        return 0, failed_ids


def main():
    parser = argparse.ArgumentParser(description="Upsert chunks to Azure AI Search")
    parser.add_argument(
        '--chunks-file',
        default='data/processed/chunks.jsonl',
        help='Input chunks file (default: data/processed/chunks.jsonl)'
    )
    parser.add_argument(
        '--checkpoint-file', 
        default='data/processed/upsert_checkpoint.json',
        help='Checkpoint file (default: data/processed/upsert_checkpoint.json)'
    )
    parser.add_argument(
        '--embed-batch-size',
        type=int,
        default=EMBED_BATCH_SIZE,
        help=f'Embedding batch size (default: {EMBED_BATCH_SIZE})'
    )
    parser.add_argument(
        '--upload-batch-size',
        type=int, 
        default=UPLOAD_BATCH_SIZE,
        help=f'Upload batch size (default: {UPLOAD_BATCH_SIZE})'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start from beginning, ignore checkpoint'
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    try:
        # Load configurations
        logging.info("🔧 Loading Azure configurations...")
        openai_config, search_config, index_name = load_azure_configs()
        
        # Create clients
        logging.info("🌐 Creating Azure clients...")
        openai_client = create_azure_openai_client(openai_config)
        search_client = create_search_client(search_config, index_name)
        
        model_deployment = openai_config['deployment'] or openai_config['model_name']
        
        # Clear checkpoint if no-resume
        if args.no_resume and os.path.exists(args.checkpoint_file):
            os.remove(args.checkpoint_file)
            logging.info("🔄 Removed existing checkpoint, starting fresh")
        
        # Process chunks
        process_chunks_file(
            args.chunks_file,
            openai_client,
            search_client, 
            model_deployment,
            args.checkpoint_file,
            args.embed_batch_size,
            args.upload_batch_size
        )
        
        return 0
        
    except Exception as e:
        logging.error(f"❌ Script failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
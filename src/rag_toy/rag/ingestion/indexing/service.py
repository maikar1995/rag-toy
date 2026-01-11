"""IndexingService for embedding generation and Azure Search upsert."""

import logging
import time
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import AzureOpenAI
from azure.search.documents import SearchClient

from ...models import Chunk
from .models import IndexingResult
from .mapper import ChunkMapper
from .retry import retry_with_backoff

logger = logging.getLogger(__name__)

# Configuration constants from proven script
EMBED_BATCH_SIZE = 16
UPSERT_BATCH_SIZE = 200
MAX_RETRIES_EMBEDDING = 6
MAX_RETRIES_SEARCH = 4


class IndexingService:
    """
    Stateless service for embedding generation and indexing to Azure Search.
    
    Uses dependency injection for clients and provides structured results.
    Handles batching, retries, and error recovery automatically.
    """
    
    def __init__(
        self,
        openai_client: AzureOpenAI,
        search_client: SearchClient,
        model_deployment: str,
        embed_batch_size: int = EMBED_BATCH_SIZE,
        upsert_batch_size: int = UPSERT_BATCH_SIZE
    ):
        """
        Initialize IndexingService with injected dependencies.
        
        Args:
            openai_client: Azure OpenAI client for embeddings
            search_client: Azure Search client for indexing
            model_deployment: Model deployment name for embeddings
            embed_batch_size: Batch size for embedding generation
            upsert_batch_size: Batch size for search upsert
        """
        self.openai_client = openai_client
        self.search_client = search_client
        self.model_deployment = model_deployment
        self.embed_batch_size = embed_batch_size
        self.upsert_batch_size = upsert_batch_size
        self.mapper = ChunkMapper()
    
    def index_chunks(self, chunks: List[Chunk]) -> IndexingResult:
        """
        Process chunks: generate embeddings and upsert to Azure Search.
        
        Args:
            chunks: List of chunks to index
            
        Returns:
            Structured result with metrics and error information
        """
        start_time = time.time()
        
        if not chunks:
            return IndexingResult(
                total_chunks=0,
                embedded_chunks=0,
                upserted_chunks=0,
                failed_chunks=0,
                processing_time_seconds=0.0,
                embedding_batch_size=self.embed_batch_size,
                upsert_batch_size=self.upsert_batch_size
            )
        
        logger.info(f"Starting indexing of {len(chunks)} chunks")
        
        # Process in batches
        batches = self._create_batches(chunks, self.upsert_batch_size)
        
        total_embedded = 0
        total_upserted = 0
        failed_chunk_ids = []
        errors = []
        
        for batch in batches:
            try:
                batch_embedded, batch_upserted, batch_failed_ids, batch_errors = self._process_batch(batch)
                total_embedded += batch_embedded
                total_upserted += batch_upserted
                failed_chunk_ids.extend(batch_failed_ids)
                errors.extend(batch_errors)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                failed_chunk_ids.extend([chunk.id for chunk in batch])
                errors.append(f"Batch error: {str(e)}")
        
        processing_time = time.time() - start_time
        
        result = IndexingResult(
            total_chunks=len(chunks),
            embedded_chunks=total_embedded,
            upserted_chunks=total_upserted,
            failed_chunks=len(failed_chunk_ids),
            processing_time_seconds=processing_time,
            failed_chunk_ids=failed_chunk_ids,
            errors=errors,
            embedding_batch_size=self.embed_batch_size,
            upsert_batch_size=self.upsert_batch_size
        )
        
        logger.info(f"Indexing completed: {result.summary()}")
        return result
    
    def _process_batch(self, chunks: List[Chunk]) -> Tuple[int, int, List[str], List[str]]:
        """
        Process a single batch of chunks.
        
        Returns:
            Tuple of (embedded_count, upserted_count, failed_ids, errors)
        """
        try:
            # Generate embeddings in sub-batches
            texts = [chunk.text for chunk in chunks]
            embeddings = self._generate_embeddings_for_texts(texts)
            embedded_count = len(embeddings)
            
            # Map to search documents
            search_docs = []
            for chunk, embedding in zip(chunks, embeddings):
                search_doc = self.mapper.to_search_document(chunk, embedding)
                search_docs.append(search_doc)
            
            # Upsert to search
            upserted_count, failed_ids = self._upsert_documents_with_retry(search_docs)
            
            return embedded_count, upserted_count, failed_ids, []
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            failed_ids = [chunk.id for chunk in chunks]
            return 0, 0, failed_ids, [str(e)]
    
    def _generate_embeddings_for_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts using sub-batching.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        # Process in embedding sub-batches
        for i in range(0, len(texts), self.embed_batch_size):
            sub_batch = texts[i:i + self.embed_batch_size]
            sub_embeddings = self._generate_embeddings_batch(sub_batch)
            all_embeddings.extend(sub_embeddings)
        
        return all_embeddings
    
    def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts with retry logic.
        """
        def _generate():
            response = self.openai_client.embeddings.create(
                model=self.model_deployment,
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
        
        return retry_with_backoff(
            _generate,
            MAX_RETRIES_EMBEDDING,
            f"Generate embeddings (batch size: {len(texts)})"
        )
    
    def _upsert_documents_with_retry(self, documents: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
        """
        Upsert documents to Azure Search with retry and split-on-failure logic.
        
        Args:
            documents: Documents to upsert
            
        Returns:
            Tuple of (successful_count, failed_ids)
        """
        try:
            success_count, failed_ids = self._upsert_documents_batch(documents)
            
            # If some failed and batch is splittable, try split recovery
            if failed_ids and len(documents) > 1:
                failed_docs = [doc for doc in documents if doc['id'] in failed_ids]
                retry_success, retry_failed = self._split_on_failure(failed_docs)
                success_count += retry_success
                failed_ids = retry_failed
            
            return success_count, failed_ids
            
        except Exception as e:
            logger.error(f"Upsert failed: {e}")
            failed_ids = [doc['id'] for doc in documents]
            return 0, failed_ids
    
    def _upsert_documents_batch(self, documents: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
        """
        Upsert a batch of documents to Azure Search.
        """
        def _upsert():
            return self.search_client.merge_or_upload_documents(documents=documents)
        
        try:
            results = retry_with_backoff(
                _upsert,
                MAX_RETRIES_SEARCH,
                f"Upsert documents (batch size: {len(documents)})"
            )
            
            successful_count = 0
            failed_ids = []
            
            for result in results:
                if result.succeeded:
                    successful_count += 1
                else:
                    failed_ids.append(result.key)
                    error_msg = getattr(result, 'error_message', 'Unknown error')
                    logger.warning(f"Failed to upsert document {result.key}: {error_msg}")
            
            return successful_count, failed_ids
            
        except Exception as e:
            failed_ids = [doc['id'] for doc in documents]
            logger.error(f"Entire batch failed: {e}")
            return 0, failed_ids
    
    def _split_on_failure(self, documents: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
        """
        Recursively split failed batches to isolate problematic documents.
        """
        if len(documents) == 1:
            return 0, [documents[0]['id']]
        
        logger.info(f"Splitting batch of {len(documents)} documents")
        mid = len(documents) // 2
        batch1 = documents[:mid]
        batch2 = documents[mid:]
        
        total_successful = 0
        total_failed = []
        
        # Process first half
        try:
            success1, failed1 = self._upsert_documents_batch(batch1)
            total_successful += success1
            
            if failed1 and len(batch1) > 1:
                subsuccess, subfailed = self._split_on_failure(
                    [doc for doc in batch1 if doc['id'] in failed1]
                )
                total_successful += subsuccess
                total_failed.extend(subfailed)
            else:
                total_failed.extend(failed1)
        except Exception as e:
            logger.error(f"First half failed: {e}")
            total_failed.extend([doc['id'] for doc in batch1])
        
        # Process second half
        try:
            success2, failed2 = self._upsert_documents_batch(batch2)
            total_successful += success2
            
            if failed2 and len(batch2) > 1:
                subsuccess, subfailed = self._split_on_failure(
                    [doc for doc in batch2 if doc['id'] in failed2]
                )
                total_successful += subsuccess
                total_failed.extend(subfailed)
            else:
                total_failed.extend(failed2)
        except Exception as e:
            logger.error(f"Second half failed: {e}")
            total_failed.extend([doc['id'] for doc in batch2])
        
        return total_successful, total_failed
    
    def _create_batches(self, chunks: List[Chunk], batch_size: int) -> List[List[Chunk]]:
        """
        Split chunks into batches of specified size.
        """
        batches = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batches.append(batch)
        return batches
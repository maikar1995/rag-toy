"""
Ingestion Orchestrator

Manages the complete ingestion pipeline: Loaders → Chunkers → IndexingService
Implements streaming processing with deterministic chunker selection.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Iterator, Optional, List, Tuple
from datetime import datetime

from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

from ..models import Document, Chunk
from .loaders import get_loader
from .chunking import get_chunker, ChunkerConfig
from .indexing.service import IndexingService
from .indexing.models import IngestionSummary

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


def load_azure_configs() -> Tuple[Dict[str, str], Dict[str, str], str]:
    """Load Azure OpenAI and Search configurations from environment."""
    openai_config = {
        'endpoint': os.getenv('EMBEDDING_1_ENDPOINT'),
        'api_key': os.getenv('EMBEDDING_1_API_KEY'),
        'api_version': os.getenv('EMBEDDING_1_API_VERSION', '2024-02-01'),
        'model_name': os.getenv('EMBEDDING_1_MODEL_NAME', 'text-embedding-3-small'),
        'deployment': os.getenv('EMBEDDING_1_DEPLOYMENT')
    }
    search_config = {
        'endpoint': os.getenv('AZURE_SEARCH_ENDPOINT'),
        'api_key': os.getenv('AZURE_INSERTION_KEY'),
        'api_version': os.getenv('AZURE_SEARCH_API_VERSION', '2023-11-01')
    }
    index_name = os.getenv('AZURE_SEARCH_INDEX_NAME', 'rag-toy-index-v2')
    
    for name, config in [("OpenAI", openai_config), ("Search", search_config)]:
        missing = [k for k, v in config.items() if not v and k != 'deployment']
        if missing:
            raise ValueError(f"Missing {name} environment variables: {missing}")
    
    return openai_config, search_config, index_name


def create_azure_openai_client(config: Dict[str, str]) -> AzureOpenAI:
    """Create Azure OpenAI client from config."""
    return AzureOpenAI(
        azure_endpoint=config['endpoint'],
        api_key=config['api_key'],
        api_version=config['api_version']
    )


def create_search_client(config: Dict[str, str], index_name: str) -> SearchClient:
    """Create Azure Search client from config."""
    return SearchClient(
        endpoint=config['endpoint'],
        index_name=index_name,
        credential=AzureKeyCredential(config['api_key'])
    )


class IngestionOrchestrator:
    """Orchestrates the complete ingestion pipeline."""
    
    def __init__(
        self,
        search_client: SearchClient,
        openai_client: AzureOpenAI,
        embedding_model: str = "text-embedding-ada-002"
    ):
        self.indexing_service = IndexingService(
            search_client=search_client,
            openai_client=openai_client,
            embedding_model=embedding_model
        )
        
    def run_ingestion(
        self,
        data_paths: Dict[str, str],
        chunk_engine: str = "native",
        **chunker_kwargs
    ) -> IngestionSummary:
        """
        Execute complete ingestion pipeline.
        
        Args:
            data_paths: Dict mapping source types to file paths
                        e.g., {"pdf": "data/pdf_pages.jsonl", "web": "data/web_docs.jsonl"}
            chunk_engine: Chunking engine to use ("native", "langchain", "llamaindex")
            **chunker_kwargs: Additional arguments for chunker configuration
            
        Returns:
            IngestionSummary with execution results
        """
        start_time = time.time()
        
        # Derive index name
        index_name = f"rag-toy-{chunk_engine}-v1"
        logger.info(f"🚀 Starting ingestion pipeline: engine='{chunk_engine}', index='{index_name}'")
        logger.info(f"📁 Data paths: {data_paths}")
        logger.info(f"⚙️  Chunker kwargs: {chunker_kwargs}")
        
        # Validate inputs
        validation_errors = self._validate_inputs(data_paths, chunk_engine)
        if validation_errors:
            logger.error(f"❌ Validation failed: {validation_errors}")
            return IngestionSummary(
                success=False,
                total_documents=0,
                total_chunks=0,
                indexed_chunks=0,
                failed_chunks=0,
                errors=validation_errors,
                chunk_engine=chunk_engine,
                index_name=index_name,
                processing_time=time.time() - start_time,
                timestamp=datetime.now()
            )
        
        # Initialize counters
        total_documents = 0
        total_chunks = 0
        indexed_chunks = 0
        failed_chunks = 0
        errors: List[str] = []
        
        try:
            # Stream processing: Documents → Chunks → Indexing
            for source_type, file_path in data_paths.items():
                logger.info(f"📄 Processing {source_type} documents from {file_path}")
                
                # Load documents
                try:
                    loader = get_loader(source_type)
                    documents = loader.load(file_path)
                    
                    # Get chunker (deterministic selection)
                    chunker = get_chunker(
                        engine=chunk_engine,
                        doc_type=source_type,
                        **chunker_kwargs
                    )
                    logger.info(f"🔧 Using chunker: {type(chunker).__name__} for {source_type}")
                    
                    # Stream processing
                    doc_count_in_source = 0
                    for doc in documents:
                        total_documents += 1
                        doc_count_in_source += 1
                        
                        try:
                            logger.debug(f"🔍 Processing document: {doc.id} (source: {source_type})")
                            
                            # Generate chunks
                            chunks = list(chunker.chunk(doc))
                            total_chunks += len(chunks)
                            
                            if chunks:
                                logger.debug(f"✂️  Generated {len(chunks)} chunks for {doc.id}")
                                
                                # Index chunks
                                result = self.indexing_service.index_chunks(
                                    chunks=chunks
                                )
                                
                                indexed_chunks += result.upserted_chunks
                                failed_chunks += result.failed_chunks
                                
                                # Log indexing results
                                if result.failed_chunks > 0:
                                    logger.warning(f"⚠️  Document {doc.id}: {result.upserted_chunks} indexed, {result.failed_chunks} failed")
                                else:
                                    logger.debug(f"✅ Document {doc.id}: {result.upserted_chunks} chunks indexed successfully")
                                
                                if result.errors:
                                    for error in result.errors:
                                        logger.error(f"🔥 Indexing error for {doc.id}: {error}")
                                    errors.extend(result.errors)
                            else:
                                logger.warning(f"⚠️  No chunks generated for document {doc.id} (source: {source_type})")
                        
                        except Exception as e:
                            error_msg = f"Failed to process document {doc.id} (source: {source_type}): {e}"
                            logger.error(f"❌ {error_msg}")
                            logger.debug(f"💥 Document processing stacktrace:", exc_info=True)
                            errors.append(error_msg)
                            failed_chunks += 1
                        
                        # Progress logging for large datasets
                        if doc_count_in_source % 100 == 0:
                            logger.info(f"📈 Progress {source_type}: {doc_count_in_source} documents processed...")
                    
                    logger.info(f"✅ Completed {source_type}: {doc_count_in_source} documents processed")
                        
                except Exception as e:
                    error_msg = f"Failed to load {source_type} from {file_path}: {e}"
                    logger.error(f"❌ {error_msg}")
                    logger.debug(f"💥 Source loading stacktrace:", exc_info=True)
                    errors.append(error_msg)
            
            processing_time = time.time() - start_time
            success = len(errors) == 0 and failed_chunks == 0
            
            summary = IngestionSummary(
                success=success,
                total_documents=total_documents,
                total_chunks=total_chunks,
                indexed_chunks=indexed_chunks,
                failed_chunks=failed_chunks,
                errors=errors,
                chunk_engine=chunk_engine,
                index_name=index_name,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
            # Enhanced final logging
            logger.info("=" * 60)
            logger.info("📊 INGESTION SUMMARY")
            logger.info(f"⏱️  Processing time: {processing_time:.2f}s")
            logger.info(f"📄 Documents processed: {total_documents}")
            logger.info(f"✂️  Total chunks generated: {total_chunks}")
            logger.info(f"💾 Chunks indexed: {indexed_chunks}")
            if failed_chunks > 0:
                logger.error(f"❌ Failed chunks: {failed_chunks}")
            if errors:
                logger.error(f"🔥 Total errors: {len(errors)}")
                logger.error("First 3 errors:")
                for i, error in enumerate(errors[:3], 1):
                    logger.error(f"  {i}. {error}")
                if len(errors) > 3:
                    logger.error(f"  ... and {len(errors) - 3} more errors")
            
            if success:
                logger.info("🎉 Ingestion completed successfully!")
            else:
                logger.warning("⚠️  Ingestion completed with errors")
            logger.info("=" * 60)
            
            return summary
            
        except Exception as e:
            error_msg = f"Critical ingestion failure: {e}"
            logger.error(f"💥 {error_msg}")
            logger.error("💀 Critical failure stacktrace:", exc_info=True)
            
            return IngestionSummary(
                success=False,
                total_documents=total_documents,
                total_chunks=total_chunks,
                indexed_chunks=indexed_chunks,
                failed_chunks=failed_chunks,
                errors=errors + [error_msg],
                chunk_engine=chunk_engine,
                index_name=index_name,
                processing_time=time.time() - start_time,
                timestamp=datetime.now()
            )
    
    def _validate_inputs(self, data_paths: Dict[str, str], chunk_engine: str) -> List[str]:
        """Validate inputs and return list of errors."""
        errors = []
        
        # Validate chunk engine
        supported_engines = {"native", "langchain", "llamaindex"}
        if chunk_engine not in supported_engines:
            errors.append(f"Unsupported chunk engine: {chunk_engine}. Supported: {sorted(supported_engines)}")
        
        # Validate chunker creation for each source type
        if chunk_engine in supported_engines:
            try:
                for source_type in data_paths.keys():
                    try:
                        get_chunker(engine=chunk_engine, doc_type=source_type, config=ChunkerConfig())
                    except Exception as e:
                        errors.append(f"Cannot create {chunk_engine} chunker for {source_type}: {e}")
            except Exception as e:
                errors.append(f"Chunker validation failed: {e}")
        
        # Validate data paths
        if not data_paths:
            errors.append("No data paths provided")
        
        for source_type, file_path in data_paths.items():
            # Check if source type is supported
            supported_sources = {"pdf", "web", "pdf_book"}
            if source_type not in supported_sources:
                errors.append(f"Unsupported source type: {source_type}. Supported: {sorted(supported_sources)}")
            
            # Check if file exists
            if not Path(file_path).exists():
                errors.append(f"File not found: {file_path}")
        
        return errors


def run_ingestion(
    data_paths: Dict[str, str],
    search_client: SearchClient,
    openai_client: AzureOpenAI,
    chunk_engine: str = "native",
    embedding_model: str = "text-embedding-ada-002",
    **chunker_kwargs
) -> IngestionSummary:
    """
    Convenience function to run complete ingestion pipeline.
    
    Args:
        data_paths: Dict mapping source types to file paths
        search_client: Azure Search client
        openai_client: OpenAI client for embeddings
        chunk_engine: Chunking engine ("native", "langchain", "llamaindex")
        embedding_model: OpenAI embedding model name
        **chunker_kwargs: Additional chunker configuration
        
    Returns:
        IngestionSummary with execution results
    """
    orchestrator = IngestionOrchestrator(
        search_client=search_client,
        openai_client=openai_client,
        embedding_model=embedding_model
    )
    
    return orchestrator.run_ingestion(
        data_paths=data_paths,
        chunk_engine=chunk_engine,
        **chunker_kwargs
    )


def run_ingestion_with_env(
    data_paths: Dict[str, str],
    chunk_engine: str = "native",
    **chunker_kwargs
) -> IngestionSummary:
    """
    Convenience function that creates clients from environment variables.
    
    Args:
        data_paths: Dict mapping source types to file paths
        chunk_engine: Chunking engine ("native", "langchain", "llamaindex")
        **chunker_kwargs: Additional chunker configuration
        
    Returns:
        IngestionSummary with execution results
    """
    # Load configurations
    logger.info("🔧 Loading Azure configurations...")
    openai_config, search_config, base_index_name = load_azure_configs()
    
    # Derive index name based on chunk engine
    derived_index_name = f"rag-toy-{chunk_engine}-v1"
    logger.info(f"Using derived index name: {derived_index_name} (base: {base_index_name})")
    
    # Create clients with derived index name
    logger.info("🌐 Creating Azure clients...")
    openai_client = create_azure_openai_client(openai_config)
    search_client = create_search_client(search_config, derived_index_name)
    
    # Get embedding model
    embedding_model = openai_config['deployment'] or openai_config['model_name']
    
    return run_ingestion(
        data_paths=data_paths,
        search_client=search_client,
        openai_client=openai_client,
        chunk_engine=chunk_engine,
        embedding_model=embedding_model,
        **chunker_kwargs
    )


if __name__ == "__main__":

    # Data paths for JSONL files
    data_paths = {
        "pdf": "data/pdf_pages.jsonl",
        "web": "data/web_docs.jsonl"
    }
    
    # Run ingestion with environment config
    summary = run_ingestion_with_env(
        data_paths=data_paths,
        chunk_engine="native"  # or "langchain", "llamaindex"
    )
    
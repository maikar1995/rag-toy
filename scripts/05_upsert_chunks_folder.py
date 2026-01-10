import os
import json
import logging
import argparse
import re
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError

# Contrato estricto de chunk
REQUIRED_FIELDS = [
    "chunk_id",
    "doc_id",
    "page",
    "content_type",
    "source_uri",
    "content",
    "metadata"
]


CHUNKS_DIR = Path("data/chunks/")
CHECKPOINT_PREFIX = "upsert_checkpoint_"

# Configuración embedding/upsert
EMBED_BATCH_SIZE = 16
UPLOAD_BATCH_SIZE = 200
MAX_RETRIES_EMBEDDING = 6
MAX_RETRIES_SEARCH = 4
BASE_DELAY = 1.0

load_dotenv()
@dataclass
class CheckpointState:
    last_processed_line: int = 0
    successful_upserts: int = 0
    failed_chunk_ids: List[str] = None
    timestamp: str = ""
    def __post_init__(self):
        if self.failed_chunk_ids is None:
            self.failed_chunk_ids = []

def exponential_backoff_with_jitter(attempt: int, base_delay: float = BASE_DELAY, max_delay: float = 60.0) -> float:
    delay = min(max_delay, base_delay * (2 ** attempt))
    jitter = random.uniform(0, 0.25 * delay)
    return delay + jitter

def retry_with_backoff(func, max_retries: int, operation_name: str, *args, **kwargs):
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
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

def load_azure_configs() -> Tuple[Dict[str, str], Dict[str, str], str]:
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
    return AzureOpenAI(
        azure_endpoint=config['endpoint'],
        api_key=config['api_key'],
        api_version=config['api_version']
    )

def create_search_client(config: Dict[str, str], index_name: str) -> SearchClient:
    return SearchClient(
        endpoint=config['endpoint'],
        index_name=index_name,
        credential=AzureKeyCredential(config['api_key'])
    )

def generate_embeddings_batch(client: AzureOpenAI, texts: List[str], model_deployment: str) -> List[List[float]]:
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

def clean_document_id(chunk_id: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9_\-=]', '_', chunk_id)
    cleaned = cleaned.lstrip('_')
    return cleaned

def map_chunk_to_search_document(chunk: Dict[str, Any], embedding: List[float]) -> Dict[str, Any]:
    # Usa los campos del chunk directamente, sin metadata anidado
    clean_chunk_id = clean_document_id(chunk['chunk_id'])
    document = {
        "id": clean_chunk_id,
        "content": chunk['content'],
        "contentVector": embedding,
        "doc_id": chunk['doc_id'],
        "page": chunk.get('page'),
        "content_type": chunk.get('content_type', 'text'),
        "chunk_id": chunk.get('chunk_id'),
        "source_uri": chunk.get('source_uri', ''),
        "emb_version": chunk.get('emb_version'),
        "doc_hash": chunk.get('doc_hash'),
        "ingested_at": chunk.get('ingested_at'),
        "fetched_at": chunk.get('fetched_at'),
        "chunk_method": chunk.get('chunk_method')
    }
    return document

def upsert_documents_batch(search_client: SearchClient, documents: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
    def _upsert():
        result = search_client.merge_or_upload_documents(documents=documents)
        return result
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
                logging.error(f"Failed to upsert document {result.key}: {result.error_message}")
        return successful_count, failed_ids
    except Exception as e:
        failed_ids = [doc['id'] for doc in documents]
        logging.error(f"Entire batch failed: {e}")
        return 0, failed_ids

def split_on_failure(search_client: SearchClient, documents: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
    if len(documents) == 1:
        return 0, [documents[0]['id']]
    logging.info(f"Splitting batch of {len(documents)} documents into smaller batches")
    mid = len(documents) // 2
    batch1 = documents[:mid]
    batch2 = documents[mid:]
    total_successful = 0
    total_failed = []
    try:
        success1, failed1 = upsert_documents_batch(search_client, batch1)
        total_successful += success1
        if failed1 and len(batch1) > 1:
            subsuccess, subfailed = split_on_failure(search_client, [doc for doc in batch1 if doc['id'] in failed1])
            total_successful += subsuccess
            total_failed.extend(subfailed)
        else:
            total_failed.extend(failed1)
    except Exception as e:
        logging.error(f"First half failed completely: {e}")
        total_failed.extend([doc['id'] for doc in batch1])
    try:
        success2, failed2 = upsert_documents_batch(search_client, batch2)
        total_successful += success2
        if failed2 and len(batch2) > 1:
            subsuccess, subfailed = split_on_failure(search_client, [doc for doc in batch2 if doc['id'] in failed2])
            total_successful += subsuccess
            total_failed.extend(subfailed)
        else:
            total_failed.extend(failed2)
    except Exception as e:
        logging.error(f"Second half failed completely: {e}")
        total_failed.extend([doc['id'] for doc in batch2])
    return total_successful, total_failed

def load_checkpoint(checkpoint_file: str) -> CheckpointState:
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
    checkpoint = load_checkpoint(checkpoint_file)
    if not os.path.exists(chunks_file):
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
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
            if line_number <= checkpoint.last_processed_line:
                continue
            try:
                chunk = json.loads(line.strip())
                validate_chunk_contract(chunk, chunks_file)
                current_batch.append((chunk, line_number))
                if len(current_batch) >= upload_batch_size:
                    success_count, failed_ids = process_batch(
                        current_batch, openai_client, search_client, 
                        model_deployment, embed_batch_size
                    )
                    checkpoint.last_processed_line = current_batch[-1][1]
                    checkpoint.successful_upserts += success_count
                    checkpoint.failed_chunk_ids.extend(failed_ids)
                    save_checkpoint(checkpoint, checkpoint_file)
                    progress = (checkpoint.last_processed_line / total_lines) * 100
                    logging.info(f"📈 Progress: {progress:.1f}% ({checkpoint.last_processed_line}/{total_lines}), "
                                f"Successful: {checkpoint.successful_upserts}, Failed: {len(checkpoint.failed_chunk_ids)}")
                    current_batch = []
            except json.JSONDecodeError as e:
                logging.error(f"Invalid JSON on line {line_number}: {e}")
                continue
            except Exception as e:
                logging.error(f"Error processing line {line_number}: {e}")
                continue
        if current_batch:
            success_count, failed_ids = process_batch(
                current_batch, openai_client, search_client,
                model_deployment, embed_batch_size
            )
            checkpoint.last_processed_line = current_batch[-1][1]
            checkpoint.successful_upserts += success_count
            checkpoint.failed_chunk_ids.extend(failed_ids)
            save_checkpoint(checkpoint, checkpoint_file)
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
    chunks = [item[0] for item in batch_with_lines]
    try:
        texts = [chunk['content'] for chunk in chunks]
        all_embeddings = []
        for i in range(0, len(texts), embed_batch_size):
            sub_batch = texts[i:i + embed_batch_size]
            sub_embeddings = generate_embeddings_batch(openai_client, sub_batch, model_deployment)
            all_embeddings.extend(sub_embeddings)
        search_documents = []
        for chunk, embedding in zip(chunks, all_embeddings):
            search_doc = map_chunk_to_search_document(chunk, embedding)
            search_documents.append(search_doc)
        success_count, failed_ids = upsert_documents_batch(search_client, search_documents)
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

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def validate_chunk_contract(chunk: Dict[str, Any], filename: str) -> None:
    missing = [f for f in REQUIRED_FIELDS if f not in chunk]
    if missing:
        logging.error(f"Contrato inválido en archivo: {filename}\nPrimer chunk_id inválido: {chunk.get('chunk_id', '<no_id>')}\nCampos faltantes: {missing}")
        raise ValueError(f"Contrato inválido en {filename}: faltan campos {missing}")

def dry_run_stats(chunks_files: List[Path]):
    total_chunks = 0
    total_chars = 0
    error_files = []
    for file in chunks_files:
        try:
            with file.open("r", encoding="utf-8") as f:
                for line in f:
                    chunk = json.loads(line)
                    validate_chunk_contract(chunk, file.name)
                    total_chunks += 1
                    total_chars += len(chunk.get("content", ""))
        except ValueError as e:
            # Extrae info del error para el log final
            error_files.append((file.name, str(e)))
            logging.warning(f"Saltando archivo por error de contrato: {file.name}")
            continue
    logging.info(f"[DRY-RUN] Archivos validados correctamente: {len(chunks_files) - len(error_files)}")
    logging.info(f"[DRY-RUN] Total chunks: {total_chunks}")
    logging.info(f"[DRY-RUN] Total caracteres: {total_chars}")
    logging.info(f"[DRY-RUN] Batches estimados (@16): {total_chunks // 16 + 1}")
    if error_files:
        logging.error("Resumen de archivos con contrato inválido:")
        for fname, err in error_files:
            logging.error(f"  - {fname}: {err}")

def main():
    parser = argparse.ArgumentParser(description="Upsert chunks from all JSONL files in data/chunks/")
    parser.add_argument('--dry-run', action='store_true', help='Solo valida y muestra stats, no upsertea')
    args = parser.parse_args()

    if not CHUNKS_DIR.exists():
        logging.error(f"Directorio no encontrado: {CHUNKS_DIR}")
        return 1
    chunks_files = sorted(CHUNKS_DIR.glob("*.jsonl"))
    if not chunks_files:
        logging.error(f"No se encontraron archivos .jsonl en {CHUNKS_DIR}")
        return 1


    if args.dry_run:
        dry_run_stats(chunks_files)
        return 0

    # Upsert real: procesa todos los archivos secuencialmente
    logging.info("🔧 Loading Azure configurations...")
    openai_config, search_config, index_name = load_azure_configs()
    logging.info("🌐 Creating Azure clients...")
    openai_client = create_azure_openai_client(openai_config)
    search_client = create_search_client(search_config, index_name)
    model_deployment = openai_config['deployment'] or openai_config['model_name']

    import concurrent.futures

    def process_file(file):
        checkpoint_file = CHUNKS_DIR / f"{CHECKPOINT_PREFIX}{file.stem}.json"
        logging.info(f"\n=== Procesando archivo: {file.name} ===")
        try:
            process_chunks_file(
                str(file),
                openai_client,
                search_client,
                model_deployment,
                str(checkpoint_file),
                EMBED_BATCH_SIZE,
                UPLOAD_BATCH_SIZE
            )
        except Exception as e:
            logging.error(f"❌ Error procesando {file.name}: {e}")

    # Paraleliza la ingesta de archivos en batches
    max_workers = min(4, len(chunks_files))  # Puedes ajustar el valor por defecto
    batch_size = max_workers
    batches = [chunks_files[i:i+batch_size] for i in range(0, len(chunks_files), batch_size)]
    for batch in batches:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_file, file) for file in batch]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    logging.info("✅ Proceso de upsert completado para todos los archivos.")
    return 0

if __name__ == "__main__":
    main()

import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any

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

    # TODO: Implementar embed + upsert secuencial por archivo
    logging.info("Upsert real no implementado aún. Usa --dry-run para validar.")
    # for file in chunks_files:
    #     ...
    # TODO: parallelize by file/batch
    return 0

if __name__ == "__main__":
    main()

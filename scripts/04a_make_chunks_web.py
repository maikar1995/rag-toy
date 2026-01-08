import json
import logging
from pathlib import Path

# Configurables
WEB_CHUNK_MIN_CHARS = 800
WEB_CHUNK_MAX_CHARS = 1200
WEB_OVERLAP_CHARS = 0  # No overlap for web
RAW_HTML_DEBUG = False  # Set True to save HTML for debug
RAW_HTML_DIR = Path("data/raw/web")


INPUT_PATH = Path("data/processed/web_docs.jsonl")
OUTPUT_PATH = Path("data/chunks/chunks_web.jsonl")

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def split_paragraphs(text):
    # Split by double newlines, fallback to single newline if needed
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paras

def group_paragraphs(paragraphs, min_chars, max_chars):
    chunks = []
    current = []
    current_len = 0
    for para in paragraphs:
        if len(para) > max_chars:
            # Hard split this paragraph
            for i in range(0, len(para), max_chars):
                part = para[i:i+max_chars]
                if current_len >= min_chars:
                    chunks.append("\n\n".join(current))
                    current = []
                    current_len = 0
                chunks.append(part)
            continue
        if current_len + len(para) + (2 if current else 0) > max_chars:
            if current:
                chunks.append("\n\n".join(current))
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para) + (2 if current_len > 0 else 0)
    if current:
        chunks.append("\n\n".join(current))
    return chunks

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RAW_HTML_DEBUG:
        RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)
    chunk_ids = set()
    chunk_count = 0
    with INPUT_PATH.open("r", encoding="utf-8") as fin, OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            doc = json.loads(line)
            doc_id = doc["doc_id"]
            source_url = doc.get("source_url")
            content = doc.get("content", "")
            original_char_count = len(content)
            # Optionally save HTML
            if RAW_HTML_DEBUG and doc["metadata"].get("raw_html_path"):
                # Already saved
                pass
            elif RAW_HTML_DEBUG and "html" in doc["metadata"]:
                html_path = RAW_HTML_DIR / f"{doc_id}.html"
                with html_path.open("w", encoding="utf-8") as fhtml:
                    fhtml.write(doc["metadata"]["html"])
                doc["metadata"]["raw_html_path"] = str(html_path)
            # Chunking
            paragraphs = split_paragraphs(content)
            chunks = group_paragraphs(paragraphs, WEB_CHUNK_MIN_CHARS, WEB_CHUNK_MAX_CHARS)
            for idx, chunk_text in enumerate(chunks):
                chunk_id = f"web__{doc_id}_chunk_{idx:03d}"
                if chunk_id in chunk_ids:
                    logging.warning(f"Duplicate chunk_id detected: {chunk_id} (source: {source_url})")
                    raise ValueError(f"Duplicate chunk_id: {chunk_id}")
                chunk_ids.add(chunk_id)
                chunk = {
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                    "doc_id": doc_id,
                    "page": None,
                    "content_type": "web",
                    "source_uri": source_url,
                    "content": chunk_text,
                    "metadata": {
                        "chunk_method": "web_paragraphs",
                        "original_char_count": original_char_count,
                        "chunk_char_count": len(chunk_text)
                    }
                }
                fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                chunk_count += 1
    logging.info(f"Total chunks written: {chunk_count}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Chunking Script - Process PDF pages into manageable chunks

Input: data/processed/pdf_pages.jsonl (from 02_bis_parse_pdf.py)
Output: data/chunks/chunks_pdf.jsonl

Chunking strategy:
- If page content <= ~6k chars: 1 chunk = 1 page
- If page content > ~6k chars: split into multiple chunks with 150-200 char overlap
- Smart splitting: prefer paragraph breaks, then line breaks, then spaces, then hard cut
"""

import json
import logging
import argparse
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def smart_split_text(
    text: str, max_chars: int, overlap_chars: int = 150
) -> List[Tuple[str, int, int]]:
    """
    Split text into chunks with smart boundary detection.

    Args:
        text: Text to split
        max_chars: Maximum characters per chunk
        overlap_chars: Characters to overlap between chunks

    Returns:
        List of (chunk_text, start_pos, end_pos) tuples
    """
    if len(text) <= max_chars:
        return [(text, 0, len(text))]

    chunks = []
    start = 0

    while start < len(text):
        # Calculate end position for this chunk
        end = min(start + max_chars, len(text))

        # If this would be the last chunk, take everything remaining
        if end >= len(text):
            chunk_text = text[start:]
            chunks.append((chunk_text, start, len(text)))
            break

        # Find the best split point before 'end'
        split_point = find_best_split_point(text, start, end)

        chunk_text = text[start:split_point]
        chunks.append((chunk_text, start, split_point))

        # Next chunk starts with overlap
        start = max(split_point - overlap_chars, start + 1)

    return chunks


def find_best_split_point(text: str, start: int, max_end: int) -> int:
    """
    Find the best point to split text, preferring natural boundaries.

    Priority order:
    1. Paragraph break (\n\n)
    2. Line break (\n)
    3. Space
    4. Hard cut at max_end
    """
    # Search backwards from max_end for best split point
    search_text = text[start:max_end]

    # 1. Try to find paragraph break
    paragraph_matches = list(re.finditer(r"\n\n", search_text))
    if paragraph_matches:
        last_paragraph = paragraph_matches[-1]
        return start + last_paragraph.end()

    # 2. Try to find line break
    line_matches = list(re.finditer(r"\n", search_text))
    if line_matches:
        last_line = line_matches[-1]
        return start + last_line.end()

    # 3. Try to find space
    space_matches = list(re.finditer(r" ", search_text))
    if space_matches:
        last_space = space_matches[-1]
        return start + last_space.end()

    # 4. Hard cut
    return max_end


def create_chunks_from_page(
    page_record: Dict[str, Any], max_chars: int = 6000, overlap_chars: int = 175
) -> List[Dict[str, Any]]:
    """
    Create chunks from a single page record.

    Args:
        page_record: Page record from pdf_pages.jsonl
        max_chars: Maximum characters per chunk
        overlap_chars: Overlap between chunks

    Returns:
        List of chunk records
    """
    content = page_record.get("content", "")
    doc_id = page_record.get("doc_id", "unknown")
    page = page_record.get("page", 0)
    original_metadata = page_record.get("metadata", {})

    # Split content into chunks
    chunk_data = smart_split_text(content, max_chars, overlap_chars)

    chunks = []
    for chunk_index, (chunk_text, start_pos, end_pos) in enumerate(chunk_data):
        chunk_id = f"{doc_id}_page_{page:03d}_chunk_{chunk_index:03d}"

        # Calculate overlap for this chunk
        actual_overlap = 0
        if chunk_index > 0:
            # Check how much text overlaps with previous chunk
            prev_end = chunk_data[chunk_index - 1][2] if chunk_index > 0 else 0
            if start_pos < prev_end:
                actual_overlap = prev_end - start_pos

        chunk_record = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "page": page,
            "chunk_index": chunk_index,
            "content": chunk_text.strip(),
            "metadata": {
                # Inherit original metadata
                **original_metadata,
                # Add chunk-specific metadata
                "chunk_method": "page_split_chars",
                "chunk_char_start": start_pos,
                "chunk_char_end": end_pos,
                "overlap_chars": actual_overlap,
                "original_page_char_count": len(content),
                "chunk_char_count": len(chunk_text.strip()),
            },
        }
        chunks.append(chunk_record)

    return chunks


def process_pdf_pages(
    input_file: str, output_file: str, max_chars: int = 6000, overlap_chars: int = 175
):
    """
    Process PDF pages file and create chunks.

    Args:
        input_file: Path to pdf_pages.jsonl
        output_file: Path to output chunks.jsonl
        max_chars: Maximum characters per chunk
        overlap_chars: Overlap between chunks
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total_pages = 0
    total_chunks = 0
    doc_stats = {}

    logging.info(f"Processing {input_file} -> {output_file}")
    logging.info(
        f"Chunking config: max_chars={max_chars}, overlap_chars={overlap_chars}"
    )

    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:
        for line_num, line in enumerate(infile, 1):
            try:
                page_record = json.loads(line.strip())

                # Create chunks from this page
                chunks = create_chunks_from_page(page_record, max_chars, overlap_chars)

                # Write chunks to output
                for chunk in chunks:
                    outfile.write(json.dumps(chunk, ensure_ascii=False) + "\n")

                # Update stats
                doc_id = page_record.get("doc_id", "unknown")
                if doc_id not in doc_stats:
                    doc_stats[doc_id] = {"pages": 0, "chunks": 0}

                doc_stats[doc_id]["pages"] += 1
                doc_stats[doc_id]["chunks"] += len(chunks)
                total_pages += 1
                total_chunks += len(chunks)

                if line_num % 100 == 0:
                    logging.info(
                        f"Processed {line_num} pages, created {total_chunks} chunks"
                    )

            except json.JSONDecodeError as e:
                logging.error(f"Invalid JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                logging.error(f"Error processing line {line_num}: {e}")
                continue

    # Log final stats
    logging.info(f"Processing complete:")
    logging.info(f"  Total pages processed: {total_pages}")
    logging.info(f"  Total chunks created: {total_chunks}")
    logging.info(f"  Average chunks per page: {total_chunks/total_pages:.2f}")
    logging.info(f"  Documents processed: {len(doc_stats)}")

    # Show sample document stats
    for doc_id, stats in list(doc_stats.items())[:5]:
        avg_chunks = stats["chunks"] / stats["pages"]
        logging.info(
            f"  {doc_id}: {stats['pages']} pages -> {stats['chunks']} chunks (avg: {avg_chunks:.2f})"
        )


def validate_chunks_file(chunks_file: str):
    """Validate the chunks.jsonl file format and show stats."""
    if not os.path.exists(chunks_file):
        print(f"Chunks file {chunks_file} does not exist")
        return

    print(f"Validating {chunks_file}...")

    total_chunks = 0
    doc_ids = set()
    chunks_per_doc = {}
    chunk_sizes = []
    overlap_sizes = []

    with open(chunks_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                chunk = json.loads(line.strip())

                # Check required fields
                required_fields = [
                    "chunk_id",
                    "doc_id",
                    "page",
                    "chunk_index",
                    "content",
                    "metadata",
                ]
                for field in required_fields:
                    if field not in chunk:
                        print(f"Line {line_num}: Missing field '{field}'")

                # Collect stats
                doc_id = chunk.get("doc_id")
                content = chunk.get("content", "")
                metadata = chunk.get("metadata", {})

                if doc_id:
                    doc_ids.add(doc_id)
                    if doc_id not in chunks_per_doc:
                        chunks_per_doc[doc_id] = 0
                    chunks_per_doc[doc_id] += 1

                chunk_sizes.append(len(content))
                overlap_chars = metadata.get("overlap_chars", 0)
                if overlap_chars > 0:
                    overlap_sizes.append(overlap_chars)

                total_chunks += 1

            except json.JSONDecodeError as e:
                print(f"Line {line_num}: Invalid JSON - {e}")

    # Print stats
    print(f"Validation complete:")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Unique documents: {len(doc_ids)}")
    print(
        f"  Average chunks per doc: {total_chunks / len(doc_ids) if doc_ids else 0:.1f}"
    )

    if chunk_sizes:
        print(f"  Chunk size stats:")
        print(f"    Min: {min(chunk_sizes)} chars")
        print(f"    Max: {max(chunk_sizes)} chars")
        print(f"    Avg: {sum(chunk_sizes) / len(chunk_sizes):.0f} chars")

    if overlap_sizes:
        print(f"  Overlap stats (when > 0):")
        print(f"    Min: {min(overlap_sizes)} chars")
        print(f"    Max: {max(overlap_sizes)} chars")
        print(f"    Avg: {sum(overlap_sizes) / len(overlap_sizes):.0f} chars")
        print(f"    Chunks with overlap: {len(overlap_sizes)}/{total_chunks}")


def main():
    parser = argparse.ArgumentParser(description="Create chunks from PDF pages")
    parser.add_argument(
        "--input",
        default="data/processed/pdf_pages.jsonl",
        help="Input PDF pages file (default: data/processed/pdf_pages.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="data/chunks/chunks_pdf.jsonl",
        help="Output chunks file (default: data/chunks/chunks_pdf.jsonl)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=6000,
        help="Maximum characters per chunk (default: 6000)",
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=175,
        help="Overlap characters between chunks (default: 175)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing chunks file instead of processing",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.validate:
        validate_chunks_file(args.output)
    else:
        try:
            process_pdf_pages(
                args.input, args.output, args.max_chars, args.overlap_chars
            )
            logging.info(f"Chunks saved to: {args.output}")
            logging.info("Run with --validate to check the output")
        except Exception as e:
            logging.error(f"Failed to process chunks: {e}")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())

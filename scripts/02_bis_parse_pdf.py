#!/usr/bin/env python3
"""
PDF Parsing Script - Batch process PDFs using StandalonePDFParser
Extracts text page by page and saves to JSONL format.

Usage:
    python scripts/02_parse_pdf.py --input_dir data/raw/pdfs --output data/processed/pdf_pages.jsonl
    python scripts/02_parse_pdf.py --single_file document.pdf --output data/processed/pdf_pages.jsonl
"""
import argparse
import json
import logging
import os
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm
import hashlib

try:
    import pdfplumber
    from pypdf import PdfReader
except ImportError as e:
    print("Missing dependencies. Install with: pip install pdfplumber pypdf")
    raise e

# Optional OCR dependencies
try:
    import pytesseract

    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    print("Warning: pytesseract not found. OCR functionality disabled.")


class StandalonePDFParser:
    """
    Lightweight PDF parser extracted from RAGFlow's deepdoc module.
    Handles text extraction, basic layout analysis, and optional OCR.
    """

    def __init__(
        self,
        enable_ocr: bool = True,
        ocr_lang: str = "eng",
        min_text_height: int = 8,
        merge_threshold: float = 1.5,
    ):
        """
        Initialize the PDF parser.

        Args:
            enable_ocr: Whether to enable OCR for scanned PDFs
            ocr_lang: Language for OCR (if enabled)
            min_text_height: Minimum text height to consider
            merge_threshold: Threshold for merging text blocks
        """
        self.enable_ocr = enable_ocr and HAS_OCR
        self.ocr_lang = ocr_lang
        self.min_text_height = min_text_height
        self.merge_threshold = merge_threshold

        # Initialize processing variables
        self.reset()

    def reset(self):
        """Reset parser state for new document."""
        self.boxes = []
        self.page_images = []
        self.total_pages = 0
        self.page_height = []

    def parse(
        self,
        pdf_path: Union[str, bytes],
        page_range: Optional[tuple] = None,
        zoom_factor: int = 3,
    ) -> Dict[str, Any]:
        """
        Main parsing method. Extracts text and structure from PDF.

        Args:
            pdf_path: Path to PDF file or binary data
            page_range: Tuple (start, end) for page range, None for all pages
            zoom_factor: Zoom factor for image processing (default: 3)

        Returns:
            Dict containing:
            - 'text': Full extracted text
            - 'pages': List of page texts
            - 'metadata': Document metadata
            - 'boxes': Text boxes with coordinates (if available)
        """
        self.reset()

        try:
            # Load PDF and extract images
            self._load_pdf(pdf_path, page_range, zoom_factor)

            # Extract text using multiple methods
            text_data = self._extract_text_multi_method(pdf_path, page_range)

            # Combine and clean results
            result = self._process_results(text_data)

            return result

        except Exception as e:
            logging.error(f"PDF parsing failed: {str(e)}")
            return {
                "content": "",
                "pages": [],
                "metadata": {},
                "boxes": [],
                "error": str(e),
            }

    def _load_pdf(
        self, pdf_path: Union[str, bytes], page_range: Optional[tuple], zoom_factor: int
    ):
        """Load PDF and extract page images."""
        try:
            # Open PDF with pdfplumber
            if isinstance(pdf_path, str):
                self.pdf = pdfplumber.open(pdf_path)
            else:
                self.pdf = pdfplumber.open(BytesIO(pdf_path))

            self.total_pages = len(self.pdf.pages)

            # Determine page range
            if page_range:
                start_page, end_page = page_range
                start_page = max(0, start_page)
                end_page = min(end_page, self.total_pages)
            else:
                start_page, end_page = 0, self.total_pages

            # Extract page images for OCR if needed
            if self.enable_ocr:
                self.page_images = []
                for i in range(start_page, end_page):
                    try:
                        img = self.pdf.pages[i].to_image(resolution=72 * zoom_factor)
                        self.page_images.append(img.annotated)
                    except Exception as e:
                        logging.warning(f"Could not extract image from page {i}: {e}")
                        self.page_images.append(None)

        except Exception as e:
            logging.error(f"Failed to load PDF: {e}")
            self.page_images = []
            self.total_pages = 0

    def _extract_text_multi_method(
        self, pdf_path: Union[str, bytes], page_range: Optional[tuple]
    ) -> Dict[str, List[str]]:
        """Extract text using multiple methods and combine results."""
        methods_results = {}

        # Method 1: pdfplumber text extraction
        try:
            methods_results["pdfplumber"] = self._extract_with_pdfplumber(page_range)
        except Exception as e:
            logging.warning(f"pdfplumber extraction failed: {e}")
            methods_results["pdfplumber"] = []

        # Method 2: pypdf text extraction
        try:
            methods_results["pypdf"] = self._extract_with_pypdf(pdf_path, page_range)
        except Exception as e:
            logging.warning(f"pypdf extraction failed: {e}")
            methods_results["pypdf"] = []

        # Method 3: OCR extraction (if enabled and needed)
        if self.enable_ocr and self._needs_ocr(methods_results):
            try:
                methods_results["ocr"] = self._extract_with_ocr()
            except Exception as e:
                logging.warning(f"OCR extraction failed: {e}")
                methods_results["ocr"] = []

        return methods_results

    def _extract_with_pdfplumber(self, page_range: Optional[tuple]) -> List[str]:
        """Extract text using pdfplumber with layout preservation."""
        pages_text = []

        if page_range:
            start_page, end_page = page_range
        else:
            start_page, end_page = 0, self.total_pages

        for i in range(start_page, min(end_page, len(self.pdf.pages))):
            try:
                page = self.pdf.pages[i]

                # Extract characters with position info
                chars = page.chars
                if chars:
                    # Group characters into text blocks
                    text_blocks = self._group_chars_to_blocks(chars)
                    page_text = self._merge_text_blocks(text_blocks)
                else:
                    # Fallback to simple text extraction
                    page_text = page.extract_text() or ""

                pages_text.append(self._clean_text(page_text))

            except Exception as e:
                logging.warning(f"pdfplumber failed for page {i}: {e}")
                pages_text.append("")

        return pages_text

    def _extract_with_pypdf(
        self, pdf_path: Union[str, bytes], page_range: Optional[tuple]
    ) -> List[str]:
        """Extract text using pypdf as fallback."""
        pages_text = []

        try:
            if isinstance(pdf_path, str):
                reader = PdfReader(pdf_path)
            else:
                reader = PdfReader(BytesIO(pdf_path))

            if page_range:
                start_page, end_page = page_range
            else:
                start_page, end_page = 0, len(reader.pages)

            for i in range(start_page, min(end_page, len(reader.pages))):
                try:
                    page_text = reader.pages[i].extract_text()
                    pages_text.append(self._clean_text(page_text))
                except Exception as e:
                    logging.warning(f"pypdf failed for page {i}: {e}")
                    pages_text.append("")

        except Exception as e:
            logging.error(f"pypdf reader initialization failed: {e}")

        return pages_text

    def _extract_with_ocr(self) -> List[str]:
        """Extract text using OCR on page images."""
        if not HAS_OCR:
            return []

        pages_text = []
        for i, img in enumerate(self.page_images):
            if img is None:
                pages_text.append("")
                continue

            try:
                # Convert PIL image to format suitable for pytesseract
                if hasattr(img, "convert"):
                    img_for_ocr = img.convert("RGB")
                else:
                    img_for_ocr = img

                # Perform OCR
                text = pytesseract.image_to_string(img_for_ocr, lang=self.ocr_lang)
                pages_text.append(self._clean_text(text))

            except Exception as e:
                logging.warning(f"OCR failed for page {i}: {e}")
                pages_text.append("")

        return pages_text

    def _group_chars_to_blocks(self, chars: List[Dict]) -> List[Dict]:
        """Group characters into text blocks based on position."""
        if not chars:
            return []

        # Sort characters by position
        sorted_chars = sorted(
            chars, key=lambda c: (c["page_number"], c["top"], c["x0"])
        )

        blocks = []
        current_block = None

        for char in sorted_chars:
            if not char.get("text", "").strip():
                continue

            # Start new block or continue current one
            if current_block is None or abs(
                char["top"] - current_block["bottom"]
            ) > self.merge_threshold * char.get("height", self.min_text_height):
                # Save current block
                if current_block and current_block["text"].strip():
                    blocks.append(current_block)

                # Start new block
                current_block = {
                    "text": char["text"],
                    "x0": char["x0"],
                    "x1": char["x1"],
                    "top": char["top"],
                    "bottom": char["bottom"],
                    "page_number": char["page_number"],
                }
            else:
                # Continue current block
                current_block["text"] += char["text"]
                current_block["x1"] = max(current_block["x1"], char["x1"])
                current_block["bottom"] = max(current_block["bottom"], char["bottom"])

        # Add last block
        if current_block and current_block["text"].strip():
            blocks.append(current_block)

        return blocks

    def _merge_text_blocks(self, blocks: List[Dict]) -> str:
        """Merge text blocks into continuous text with proper spacing."""
        if not blocks:
            return ""

        # Sort blocks by reading order
        sorted_blocks = sorted(
            blocks, key=lambda b: (b["page_number"], b["top"], b["x0"])
        )

        merged_text = []
        for i, block in enumerate(sorted_blocks):
            text = block["text"].strip()
            if not text:
                continue

            # Add spacing logic
            if i > 0:
                prev_block = sorted_blocks[i - 1]
                # Add line break if significant vertical gap
                if block["top"] - prev_block["bottom"] > self.min_text_height:
                    merged_text.append("\n")
                elif not merged_text[-1].endswith(" "):
                    merged_text.append(" ")

            merged_text.append(text)

        return "".join(merged_text)

    def _needs_ocr(self, methods_results: Dict[str, List[str]]) -> bool:
        """Determine if OCR is needed based on text extraction results."""
        total_text = ""
        for method_name, pages in methods_results.items():
            total_text += " ".join(pages)

        # If very little text was extracted, likely needs OCR
        return len(total_text.strip()) < 100

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Fix common OCR errors (basic cleanup)
        text = re.sub(r'[^\w\s\.,;:!?\-()[\]{}"\']', " ", text)

        # Normalize line breaks
        text = re.sub(r"\n\s*\n", "\n\n", text)

        return text.strip()

    def _process_results(self, methods_results: Dict[str, List[str]]) -> Dict[str, Any]:
        """Process and combine results from different extraction methods."""

        # Choose best method based on text quantity and quality
        best_pages = self._select_best_extraction(methods_results)

        # Combine all pages
        full_text = "\n\n".join(page for page in best_pages if page.strip())

        # Extract basic metadata
        metadata = self._extract_metadata()

        return {
            "content": full_text,
            "pages": best_pages,
            "metadata": metadata,
            "boxes": getattr(self, "boxes", []),
            "extraction_methods": list(methods_results.keys()),
        }

    def _select_best_extraction(
        self, methods_results: Dict[str, List[str]]
    ) -> List[str]:
        """Select the best extraction method for each page."""
        if not methods_results:
            return []

        # Get maximum pages across methods
        max_pages = max(len(pages) for pages in methods_results.values())
        best_pages = []

        for page_idx in range(max_pages):
            page_texts = {}

            # Collect text from each method for this page
            for method, pages in methods_results.items():
                if page_idx < len(pages):
                    page_texts[method] = pages[page_idx]

            # Choose best text for this page (longest non-empty text)
            best_text = ""
            for method in ["pdfplumber", "pypdf", "ocr"]:  # Priority order
                if method in page_texts and len(page_texts[method].strip()) > len(
                    best_text.strip()
                ):
                    best_text = page_texts[method]

            best_pages.append(best_text)

        return best_pages

    def _extract_metadata(self) -> Dict[str, Any]:
        """Extract basic document metadata."""
        metadata = {}

        try:
            if hasattr(self, "pdf") and self.pdf:
                pdf_info = self.pdf.metadata
                if pdf_info:
                    metadata.update(
                        {
                            "title": pdf_info.get("Title", ""),
                            "author": pdf_info.get("Author", ""),
                            "subject": pdf_info.get("Subject", ""),
                            "creator": pdf_info.get("Creator", ""),
                            "producer": pdf_info.get("Producer", ""),
                            "creation_date": str(pdf_info.get("CreationDate", "")),
                            "modification_date": str(pdf_info.get("ModDate", "")),
                        }
                    )
        except Exception as e:
            logging.warning(f"Could not extract metadata: {e}")

        metadata["total_pages"] = self.total_pages
        return metadata

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, "pdf") and self.pdf:
            try:
                self.pdf.close()
            except:
                pass


def get_pdf_hash(pdf_path: str) -> str:
    """Calculate SHA256 hash of the entire PDF file."""
    try:
        with open(pdf_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        logging.warning(f"Could not calculate hash for {pdf_path}: {e}")
        return ""


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_doc_id_from_filename(filepath: str) -> str:
    """Extract document ID from filename (remove extension)."""
    return Path(filepath).stem


def find_pdf_files(input_path: str) -> List[str]:
    """Find all PDF files in input path (file or directory)."""
    input_path = Path(input_path)

    if input_path.is_file():
        if input_path.suffix.lower() == ".pdf":
            return [str(input_path)]
        else:
            raise ValueError(f"File {input_path} is not a PDF")

    elif input_path.is_dir():
        pdf_files = list(input_path.rglob("*.pdf"))
        return [str(f) for f in pdf_files]

    else:
        raise FileNotFoundError(f"Input path {input_path} does not exist")


def process_single_pdf(
    pdf_path: str, parser: StandalonePDFParser, doc_id: str = None
) -> List[Dict[str, Any]]:
    """
    Process a single PDF and return list of page records.

    Args:
        pdf_path: Path to PDF file
        parser: Initialized StandalonePDFParser instance
        doc_id: Custom document ID (optional, defaults to filename)

    Returns:
        List of dicts with format: {"doc_id": str, "page": int, "text": str}
    """
    if doc_id is None:
        doc_id = get_doc_id_from_filename(pdf_path)

    logging.info(f"Processing: {pdf_path} (doc_id: {doc_id})")

    try:
        # Generate hash once for the entire PDF
        doc_hash = get_pdf_hash(pdf_path)

        # Parse the PDF
        result = parser.parse(pdf_path)

        if "error" in result:
            logging.error(f"Error parsing {pdf_path}: {result['error']}")
            return []

        # Extract pages and metadata
        pages_text = result.get("pages", [])
        pdf_metadata = result.get("metadata", {})
        extraction_methods = result.get("extraction_methods", [])
        total_pages = len(pages_text)

        if total_pages == 0:
            logging.warning(f"No text extracted from {pdf_path}")
            return []

        # Determine extraction method used and if fallback was needed
        primary_method = extraction_methods[0] if extraction_methods else "unknown"
        used_fallback = len(extraction_methods) > 1

        # Create page records
        page_records = []
        for page_num, page_text in enumerate(pages_text, 1):
            # Skip empty pages (optional - you might want to keep them)
            if not page_text.strip():
                logging.debug(f"Skipping empty page {page_num} in {doc_id}")
                continue

            page_record = {
                "doc_id": doc_id,
                "page": page_num,  # Keep for backward compatibility
                "content": page_text.strip(),  # Renamed from "text"
                "metadata": {
                    "source_path": str(pdf_path),
                    "doc_hash": doc_hash,
                    "page_index": page_num,  # 1-indexed
                    "char_count": len(page_text.strip()),
                    "extraction_tool": primary_method,
                    "used_fallback": used_fallback,
                    "pdf_meta": pdf_metadata,  # Raw PDF metadata
                },
            }

            page_records.append(page_record)

        logging.info(
            f"Extracted {len(page_records)} non-empty pages from {doc_id} (total pages: {total_pages})"
        )
        return page_records

    except Exception as e:
        logging.error(f"Failed to process {pdf_path}: {str(e)}")
        return []


def save_to_jsonl(
    records: List[Dict[str, Any]], output_file: str, append: bool = False
):
    """
    Save records to JSONL format.

    Args:
        records: List of dictionaries to save
        output_file: Output file path
        append: Whether to append to existing file
    """
    mode = "a" if append else "w"

    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, mode, encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    logging.info(f"Saved {len(records)} records to {output_file}")


def load_existing_doc_ids(output_file: str) -> set:
    """Load existing document IDs from output file to avoid reprocessing."""
    existing_ids = set()

    if not os.path.exists(output_file):
        return existing_ids

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    existing_ids.add(record.get("doc_id"))
                except json.JSONDecodeError:
                    continue

        logging.info(f"Found {len(existing_ids)} existing documents in {output_file}")

    except Exception as e:
        logging.warning(f"Could not read existing file {output_file}: {e}")

    return existing_ids


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(
        description="Parse PDFs and extract text page by page"
    )

    # Input options - default to data/raw if no specific input given
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw",
        help="Directory containing PDF files (default: data/raw)",
    )
    parser.add_argument(
        "--single_file",
        type=str,
        help="Single PDF file to process (overrides input_dir)",
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/pdf_pages.jsonl",
        help="Output JSONL file (default: data/processed/pdf_pages.jsonl)",
    )

    # Processing options
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip documents that already exist in output file",
    )
    parser.add_argument(
        "--enable_ocr",
        action="store_true",
        default=True,
        help="Enable OCR for scanned documents (default: True)",
    )
    parser.add_argument(
        "--ocr_lang", type=str, default="eng", help="OCR language (default: eng)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Number of files to process before saving (default: 10)",
    )

    # Logging options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Initialize PDF parser
    logging.info("Initializing PDF parser...")
    pdf_parser = StandalonePDFParser(
        enable_ocr=args.enable_ocr,
        ocr_lang=args.ocr_lang,
        min_text_height=8,
        merge_threshold=1.5,
    )

    # Find PDF files - prioritize single_file over input_dir
    if args.single_file:
        input_path = args.single_file
        logging.info(f"Processing single file: {input_path}")
    else:
        input_path = args.input_dir
        logging.info(f"Processing directory: {input_path}")

        # Create data/raw if it doesn't exist
        if input_path == "data/raw" and not os.path.exists(input_path):
            os.makedirs(input_path, exist_ok=True)
            logging.info(f"Created directory: {input_path}")
            logging.info("Please add PDF files to data/raw/ and run again.")
            return

    pdf_files = find_pdf_files(input_path)

    logging.info(f"Found {len(pdf_files)} PDF files to process")

    if not pdf_files:
        logging.warning("No PDF files found!")
        return

    # Load existing document IDs if skipping
    existing_doc_ids = set()
    if args.skip_existing:
        existing_doc_ids = load_existing_doc_ids(args.output)

    # Process PDFs
    all_records = []
    processed_count = 0
    skipped_count = 0

    # Create progress bar
    pbar = tqdm(pdf_files, desc="Processing PDFs", unit="file")

    for pdf_file in pbar:
        doc_id = get_doc_id_from_filename(pdf_file)

        # Skip if already exists
        if doc_id in existing_doc_ids:
            skipped_count += 1
            pbar.set_postfix(processed=processed_count, skipped=skipped_count)
            logging.debug(f"Skipping existing document: {doc_id}")
            continue

        # Process PDF
        page_records = process_single_pdf(pdf_file, pdf_parser, doc_id)

        if page_records:
            all_records.extend(page_records)
            processed_count += 1

            # Save in batches
            if len(all_records) >= args.batch_size:
                append_mode = os.path.exists(args.output)
                save_to_jsonl(all_records, args.output, append=append_mode)
                all_records = []  # Clear batch

        pbar.set_postfix(processed=processed_count, skipped=skipped_count)

    # Save remaining records
    if all_records:
        append_mode = os.path.exists(args.output)
        save_to_jsonl(all_records, args.output, append=append_mode)

    # Final summary
    logging.info("=" * 50)
    logging.info("PROCESSING COMPLETE")
    logging.info(f"Total PDFs found: {len(pdf_files)}")
    logging.info(f"Successfully processed: {processed_count}")
    logging.info(f"Skipped (existing): {skipped_count}")
    logging.info(f"Failed: {len(pdf_files) - processed_count - skipped_count}")
    logging.info(f"Output saved to: {args.output}")
    logging.info("=" * 50)


def validate_output_file(output_file: str):
    """Validate the output JSONL file format."""
    if not os.path.exists(output_file):
        print(f"Output file {output_file} does not exist")
        return

    print(f"Validating {output_file}...")

    total_records = 0
    doc_ids = set()
    pages_per_doc = {}

    with open(output_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())

                # Check required fields
                required_fields = ["doc_id", "page", "text"]
                for field in required_fields:
                    if field not in record:
                        print(f"Line {line_num}: Missing field '{field}'")

                # Track stats
                doc_id = record.get("doc_id")
                page = record.get("page")

                if doc_id:
                    doc_ids.add(doc_id)
                    if doc_id not in pages_per_doc:
                        pages_per_doc[doc_id] = []
                    pages_per_doc[doc_id].append(page)

                total_records += 1

            except json.JSONDecodeError as e:
                print(f"Line {line_num}: Invalid JSON - {e}")

    print(f"Validation complete:")
    print(f"  Total records: {total_records}")
    print(f"  Unique documents: {len(doc_ids)}")
    print(
        f"  Average pages per doc: {total_records / len(doc_ids) if doc_ids else 0:.1f}"
    )

    # Show sample of documents with page counts
    print(f"\nSample documents:")
    for doc_id in list(doc_ids)[:5]:
        pages = sorted(pages_per_doc[doc_id])
        print(
            f"  {doc_id}: {len(pages)} pages (pages: {pages[:10]}{'...' if len(pages) > 10 else ''})"
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--validate":
        # Special validation mode
        output_file = (
            sys.argv[2] if len(sys.argv) > 2 else "data/processed/pdf_pages.jsonl"
        )
        validate_output_file(output_file)
    else:
        main()

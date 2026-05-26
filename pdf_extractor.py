import fitz  # PyMuPDF
from pathlib import Path
from config import MAX_PDF_SIZE_MB, MAX_TEXT_LENGTH


def extract_text_from_pdf(pdf_path: str) -> dict:
    """
    Extract text from a PDF file using PyMuPDF (fitz).

    Args:
        pdf_path: path to PDF file

    Returns:
        {
          "text": str — full extracted text,
          "page_count": int,
          "word_count": int,
          "truncated": bool — True if text was truncated to MAX_TEXT_LENGTH,
          "extraction_method": "pymupdf"
        }

    Raises:
        ValueError if file is not a PDF or exceeds MAX_PDF_SIZE_MB.
        RuntimeError if text extraction fails.

    PyMuPDF (imported as fitz) extracts text from PDF pages
    in reading order. For scanned PDFs (image-only), extraction returns
    empty string — OCR is outside scope for this project.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise ValueError(f'File not found: {pdf_path}')

    if path.suffix.lower() != '.pdf':
        raise ValueError(f'File must be a PDF, got: {path.suffix}')

    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_PDF_SIZE_MB:
        raise ValueError(f'File size {file_size_mb:.1f}MB exceeds maximum {MAX_PDF_SIZE_MB}MB')

    try:
        doc = fitz.open(pdf_path)
        full_text = ''
        for page in doc:
            full_text += page.get_text()
        doc.close()
    except Exception as e:
        raise RuntimeError(f'Text extraction failed: {str(e)}')

    truncated = len(full_text) > MAX_TEXT_LENGTH
    text = full_text[:MAX_TEXT_LENGTH] if truncated else full_text
    word_count = len(full_text.split())

    return {
        'text': text,
        'page_count': doc.page_count if hasattr(doc, 'page_count') else len(fitz.open(pdf_path)),
        'word_count': word_count,
        'truncated': truncated,
        'extraction_method': 'pymupdf',
    }


def extract_text_from_bytes(pdf_bytes: bytes) -> dict:
    """
    Extract text from PDF bytes (for FastAPI file upload).
    Same return format as extract_text_from_pdf.
    """
    size_mb = len(pdf_bytes) / (1024 * 1024)
    if size_mb > MAX_PDF_SIZE_MB:
        raise ValueError(f'File size {size_mb:.1f}MB exceeds maximum {MAX_PDF_SIZE_MB}MB')

    try:
        doc = fitz.open(stream=pdf_bytes, filetype='pdf')
        full_text = ''
        page_count = doc.page_count
        for page in doc:
            full_text += page.get_text()
        doc.close()
    except Exception as e:
        raise RuntimeError(f'Text extraction failed: {str(e)}')

    truncated = len(full_text) > MAX_TEXT_LENGTH
    text = full_text[:MAX_TEXT_LENGTH] if truncated else full_text
    word_count = len(full_text.split())

    return {
        'text': text,
        'page_count': page_count,
        'word_count': word_count,
        'truncated': truncated,
        'extraction_method': 'pymupdf',
    }

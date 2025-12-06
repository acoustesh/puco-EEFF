"""Extractor module for parsing PDFs and XBRL, with OCR capabilities."""

from puco_eeff.extractor.ocr_fallback import ocr_with_fallback
from puco_eeff.extractor.ocr_mistral import ocr_with_mistral
from puco_eeff.extractor.pdf_parser import extract_tables_from_pdf, extract_text_from_pdf
from puco_eeff.extractor.xbrl_parser import parse_xbrl_file

__all__ = [
    "extract_tables_from_pdf",
    "extract_text_from_pdf",
    "ocr_with_fallback",
    "ocr_with_mistral",
    "parse_xbrl_file",
]

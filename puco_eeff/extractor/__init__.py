"""Extractor module for parsing PDFs and XBRL, with OCR capabilities."""

from puco_eeff.extractor.cost_extractor import (
    CostBreakdown,
    ExtractionResult,
    LineItem,
    Sheet1Data,
    ValidationResult,
    extract_detailed_costs,
    extract_nota_21,
    extract_nota_22,
    extract_sheet1,
    extract_sheet1_from_analisis_razonado,
    extract_sheet1_from_xbrl,
    format_quarter_label,
    print_extraction_report,
    print_sheet1_report,
    save_extraction_result,
    save_sheet1_data,
)
from puco_eeff.extractor.ocr_fallback import ocr_with_fallback
from puco_eeff.extractor.ocr_mistral import ocr_with_mistral
from puco_eeff.extractor.pdf_parser import extract_tables_from_pdf, extract_text_from_pdf
from puco_eeff.extractor.xbrl_parser import get_facts_by_name, parse_xbrl_file

__all__ = [
    "CostBreakdown",
    "ExtractionResult",
    "LineItem",
    "Sheet1Data",
    "ValidationResult",
    "extract_detailed_costs",
    "extract_nota_21",
    "extract_nota_22",
    "extract_sheet1",
    "extract_sheet1_from_analisis_razonado",
    "extract_sheet1_from_xbrl",
    "extract_tables_from_pdf",
    "extract_text_from_pdf",
    "format_quarter_label",
    "get_facts_by_name",
    "ocr_with_fallback",
    "ocr_with_mistral",
    "parse_xbrl_file",
    "print_extraction_report",
    "print_sheet1_report",
    "save_extraction_result",
    "save_sheet1_data",
]

"""Extractor module for parsing PDFs and XBRL, with OCR capabilities.

Key exports:
    SectionBreakdown: Generic container for PDF section data
    extract_pdf_section: Generic config-driven PDF section extraction
    extract_sheet1: Main entry point for Sheet1 extraction
    run_sheet1_validations: Unified validation entry point
    ValidationReport: Aggregated validation results
    SumValidationResult: Result of line-item sum validation
    CrossValidationResult: Result of cross-validation formula check
"""

from puco_eeff.extractor.cost_extractor import (
    CrossValidationResult,
    ExtractionResult,
    LineItem,
    SectionBreakdown,
    Sheet1Data,
    SumValidationResult,
    ValidationReport,
    ValidationResult,
    extract_detailed_costs,
    extract_ingresos_from_pdf,
    extract_pdf_section,
    extract_sheet1,
    extract_sheet1_from_analisis_razonado,
    extract_sheet1_from_xbrl,
    extract_xbrl_totals,
    find_section_page,
    find_text_page,
    format_quarter_label,
    format_validation_report,
    print_extraction_report,
    print_sheet1_report,
    run_sheet1_validations,
    save_extraction_result,
    save_sheet1_data,
    validate_extraction,
)
from puco_eeff.extractor.ocr_fallback import ocr_with_fallback
from puco_eeff.extractor.ocr_mistral import ocr_with_mistral
from puco_eeff.extractor.pdf_parser import extract_tables_from_pdf, extract_text_from_pdf
from puco_eeff.extractor.xbrl_parser import get_facts_by_name, parse_xbrl_file

__all__ = [
    # Dataclasses
    "SectionBreakdown",
    "ExtractionResult",
    "LineItem",
    "Sheet1Data",
    "ValidationResult",
    "ValidationReport",
    "SumValidationResult",
    "CrossValidationResult",
    # Generic extraction (preferred)
    "extract_pdf_section",
    "find_text_page",
    "find_section_page",
    # High-level Sheet1 extraction
    "extract_sheet1",
    "extract_sheet1_from_analisis_razonado",
    "extract_sheet1_from_xbrl",
    "extract_detailed_costs",
    "extract_ingresos_from_pdf",
    "extract_xbrl_totals",
    # Validation
    "run_sheet1_validations",
    "validate_extraction",  # Deprecated, use run_sheet1_validations()
    "format_validation_report",
    # PDF/XBRL utilities
    "extract_tables_from_pdf",
    "extract_text_from_pdf",
    "get_facts_by_name",
    "parse_xbrl_file",
    # OCR
    "ocr_with_fallback",
    "ocr_with_mistral",
    # Output
    "format_quarter_label",
    "print_extraction_report",
    "print_sheet1_report",
    "save_extraction_result",
    "save_sheet1_data",
]

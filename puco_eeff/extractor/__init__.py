"""Extractor module for parsing PDFs and XBRL, with OCR capabilities.

Key exports:
    SectionBreakdown: Generic container for PDF section data
    extract_pdf_section: Generic config-driven PDF section extraction
    extract_sheet1: Unified entry point for Sheet1 extraction (PDF/XBRL/both)
    run_sheet1_validations: Unified validation entry point
    ValidationReport: Aggregated validation results
    SumValidationResult: Result of line-item sum validation
    CrossValidationResult: Result of cross-validation formula check
"""

from puco_eeff.extractor.extraction import (
    LineItem,
    SectionBreakdown,
    extract_ingresos_from_pdf,
    extract_pdf_section,
    extract_xbrl_totals,
    find_section_page,
    find_text_page,
    format_quarter_label,
)
from puco_eeff.extractor.extraction_pipeline import (
    extract_detailed_costs,
    extract_sheet1,
    print_extraction_report,
    save_extraction_result,
)
from puco_eeff.extractor.ocr_fallback import ocr_with_fallback
from puco_eeff.extractor.ocr_mistral import ocr_with_mistral
from puco_eeff.extractor.pdf_parser import extract_tables_from_pdf, extract_text_from_pdf
from puco_eeff.extractor.validation import (
    ComparisonResult,
    CrossValidationResult,
    ExtractionResult,
    ReferenceValidationResult,
    SumValidationResult,
    ValidationReport,
    ValidationResult,
    format_validation_report,
    run_sheet1_validations,
)
from puco_eeff.extractor.xbrl_parser import get_facts_by_name, parse_xbrl_file
from puco_eeff.sheets.sheet1 import Sheet1Data, print_sheet1_report, save_sheet1_data

__all__ = [
    "ComparisonResult",
    "CrossValidationResult",
    "ExtractionResult",
    "LineItem",
    "ReferenceValidationResult",
    # Dataclasses
    "SectionBreakdown",
    "Sheet1Data",
    "SumValidationResult",
    "ValidationReport",
    "ValidationResult",
    "extract_detailed_costs",
    "extract_ingresos_from_pdf",
    # Generic extraction (preferred)
    "extract_pdf_section",
    # High-level Sheet1 extraction (unified API)
    "extract_sheet1",
    # PDF/XBRL utilities
    "extract_tables_from_pdf",
    "extract_text_from_pdf",
    "extract_xbrl_totals",
    "find_section_page",
    "find_text_page",
    # Output
    "format_quarter_label",
    "format_validation_report",
    "get_facts_by_name",
    # OCR
    "ocr_with_fallback",
    "ocr_with_mistral",
    "parse_xbrl_file",
    "print_extraction_report",
    "print_sheet1_report",
    # Validation
    "run_sheet1_validations",
    "save_extraction_result",
    "save_sheet1_data",
]

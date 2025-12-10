"""Validation package for puco_eeff extraction pipeline.

This package provides validation types, formatting, formula evaluation,
and validation runners for the extraction pipeline.
"""

from puco_eeff.extractor.validation.format import (
    format_cross_validation_status,
    format_sum_validation_status,
    format_validation_report,
    log_validation_report,
)
from puco_eeff.extractor.validation.runner import (
    compare_with_tolerance,
    run_sheet1_validations,
)
from puco_eeff.extractor.validation.types import (
    ComparisonResult,
    CrossValidationResult,
    ExtractionResult,
    PDFXBRLComparisonResult,
    ReferenceValidationResult,
    SumValidationResult,
    ValidationReport,
    ValidationResult,
)

__all__ = [
    # Types
    "ComparisonResult",
    "CrossValidationResult",
    "ExtractionResult",
    "PDFXBRLComparisonResult",
    "ReferenceValidationResult",
    "SumValidationResult",
    "ValidationReport",
    "ValidationResult",
    # Formatting
    "format_cross_validation_status",
    "format_sum_validation_status",
    "format_validation_report",
    "log_validation_report",
    # Runners
    "compare_with_tolerance",
    "run_sheet1_validations",
]

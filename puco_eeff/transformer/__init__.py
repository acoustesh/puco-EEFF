"""Transformer module for data normalization, formatting, and source tracking.

This package transforms raw extracted data into structured output formats
suitable for Excel workbooks and audit reports.

Submodules
----------
formatter
    Data validation and structure mapping functions.
    Maps extracted data to row structures, validates against reference values.
normalizer
    Pandas-based data normalization.
    Converts raw values to consistent units (thousands USD), applies sign conventions.
source_tracker
    Provenance tracking for audit trails.
    Records which source (PDF page, XBRL fact) each value came from.

Key Classes
-----------
SourceTracker
    Tracks source provenance for each extracted field.
ValidationResult
    Result of comparing extracted value against expected (from validation_core).
ComparisonResult
    Comparison outcome with match status and difference.

Key Functions
-------------
normalize_financial_data
    Apply sign conventions and unit normalization to extracted data.
validate_balance_sheet
    Verify sum validations for cost sections.
format_validation_report
    Generate human-readable validation summary.

Notes
-----
ValidationResult and ComparisonResult are re-exported from extractor.validation_core
for convenience, since they are commonly used with transformer functions.
"""

# Re-export validation types from validation_core for convenience
from puco_eeff.extractor.validation_core import (
    ComparisonResult,
    ReferenceValidationResult,
    ValidationResult,
)

# Formatter functions for structure mapping and validation
from puco_eeff.transformer.formatter import (
    format_validation_report,
    get_field_labels,
    get_standard_structure,
    map_to_structure,
    validate_against_reference,
    validate_balance_sheet,
    validate_section_total,
)

# Normalizer for data cleaning and unit conversion
from puco_eeff.transformer.normalizer import normalize_financial_data

# Source tracking for audit trails
from puco_eeff.transformer.source_tracker import SourceTracker, create_source_mapping

# Public API - all symbols available at package level
__all__ = [
    # Validation types (from validation_core)
    "ComparisonResult",
    "ReferenceValidationResult",
    "ValidationResult",
    # Source tracking
    "SourceTracker",
    "create_source_mapping",
    # Formatter functions
    "format_validation_report",
    "get_field_labels",
    "get_standard_structure",
    "map_to_structure",
    # Normalizer
    "normalize_financial_data",
    # Validation functions
    "validate_against_reference",
    "validate_balance_sheet",
    "validate_section_total",
]

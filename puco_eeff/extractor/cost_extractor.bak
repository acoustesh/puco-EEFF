"""Extract detailed cost breakdowns from PDF and validate against XBRL.

This module provides generic PDF/XBRL extraction primitives with config-driven
rules. Sheet1-specific config is in config/sheet1/.

Key Classes:
    SectionBreakdown: Generic container for extracted PDF section data.
        Uses section_id/section_title.
    ExtractionResult: Complete extraction result with sections dict.
    LineItem: Single line item with concepto and period values.
    SumValidationResult: Result of line-item sum validation.
    CrossValidationResult: Result of cross-validation formula check.
    ValidationReport: Aggregated validation results for unified reporting.

Key Functions:
    extract_pdf_section(): Generic config-driven PDF section extraction.
    find_text_page(): Generic PDF page finder - searches for required text strings.
    find_section_page(): Config-driven wrapper using find_text_page.
    extract_sheet1(): Main entry point for Sheet1 extraction.
    run_sheet1_validations(): Unified validation entry point (sum, PDF↔XBRL, cross).
    format_validation_report(): Format ValidationReport for display.

Config Accessors (from sheet1 module):
    get_section_config(): Canonical accessor for section config with validation.
    get_section_fallback(): Get fallback section for page lookup (config-driven).
    get_ingresos_pdf_fallback_config(): Get PDF extraction settings for ingresos.

Validation System:
    The extraction pipeline includes multi-level validation:
    1. PDF ↔ XBRL totals comparison (always runs)
    2. Sum validations: Line items should sum to totals (config-driven)
    3. Cross-validations: Accounting identities (config-driven formulas)
    4. Reference validation: Compare against known-good values (opt-in)

    Tolerance for all comparisons is configured via sum_tolerance in
    config/sheet1/xbrl_mappings.json. Cross-validations can specify
    per-rule tolerances.

Architecture:
- General config (config/): File patterns, period types, sources, XBRL specs
- Sheet-specific config (config/sheet1/): Field definitions, extraction rules,
  XBRL mappings, reference data
- Sheet1Data class and config loaders are in puco_eeff.sheets.sheet1

Configuration Files:
- config/config.json: File patterns, period types, sources, OCR settings
- config/extraction_specs.json: General PDF extraction settings (number format)
- config/xbrl_specs.json: XBRL namespaces, scaling factor, period filters
- config/sheet1/fields.json: Sheet1 field definitions, row mapping (27 rows)
- config/sheet1/extraction.json: Sheet1 sections with fallback_section, min_value_threshold
- config/sheet1/xbrl_mappings.json: Sheet1 XBRL fact mappings, validation rules
- config/sheet1/reference_data.json: Known-good values for validation

Deprecated Functions (use alternatives):
- validate_extraction() → use run_sheet1_validations()
- _section_breakdowns_to_sheet1data() → use sections_to_sheet1data()
- _map_nota21_item_to_sheet1() → use _map_nota_item_to_sheet1() or sections_to_sheet1data()
- _map_nota22_item_to_sheet1() → use _map_nota_item_to_sheet1() or sections_to_sheet1data()
"""

from __future__ import annotations

import ast
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber

from puco_eeff.config import (
    extract_pdf_page_to_temp,  # noqa: F401 - exported for user review workflow
    find_file_with_alternatives,
    format_filename,
    format_period_display,
    get_config,
    get_period_paths,
    get_xbrl_scaling_factor,
    setup_logging,
)
from puco_eeff.extractor.xbrl_parser import get_facts_by_name, parse_xbrl_file
from puco_eeff.sheets.sheet1 import (
    Sheet1Data,
    get_ingresos_pdf_fallback_config,
    get_section_fallback,
    get_sheet1_cross_validations,
    get_sheet1_detail_fields,
    get_sheet1_extraction_sections,
    get_sheet1_pdf_xbrl_validations,
    get_sheet1_result_key_mapping,
    get_sheet1_section_expected_items,
    get_sheet1_section_field_mappings,
    get_sheet1_section_spec,
    get_sheet1_section_total_mapping,
    get_sheet1_sum_tolerance,
    get_sheet1_total_validations,
    get_sheet1_xbrl_fact_mapping,
    match_concepto_to_field,
    # Re-exported for backward compatibility - canonical location is puco_eeff.sheets.sheet1
    print_sheet1_report,
    save_sheet1_data,
    sections_to_sheet1data,
)

logger = setup_logging(__name__)

# Public API exports
__all__ = [
    # Dataclasses
    "LineItem",
    "SectionBreakdown",
    "ValidationResult",
    "ExtractionResult",
    "SumValidationResult",
    "CrossValidationResult",
    "ValidationReport",
    # Generic extraction functions (preferred)
    "extract_pdf_section",
    "find_text_page",
    "find_section_page",
    "extract_table_from_page",
    "parse_chilean_number",
    # XBRL extraction
    "extract_xbrl_totals",
    # High-level Sheet1 extraction
    "extract_sheet1",
    "extract_sheet1_from_xbrl",
    "extract_sheet1_from_analisis_razonado",
    "extract_detailed_costs",
    "extract_ingresos_from_pdf",
    # Validation
    "run_sheet1_validations",
    "validate_extraction",  # Deprecated, use run_sheet1_validations()
    "format_validation_report",
    "log_validation_report",
    # Output functions
    "save_extraction_result",
    "print_extraction_report",
    # Sheet1-specific (re-exported from puco_eeff.sheets.sheet1)
    "save_sheet1_data",
    "print_sheet1_report",
    # Config helpers
    "get_section_expected_labels",
    "get_all_field_labels",
    # Period formatting
    "quarter_to_roman",
    "format_period_label",
    "format_quarter_label",
]


# =============================================================================
# Sheet1 Section Extraction (Config-Driven)
# =============================================================================


def get_section_expected_labels(section_name: str, sheet_name: str = "sheet1") -> list[str]:
    """Get expected PDF labels for a section from config.

    Args:
        section_name: Section key from config/<sheet>/extraction.json
        sheet_name: Sheet name for config lookup (default: "sheet1")

    Returns:
        List of expected label strings for line items.

    Raises:
        ValueError: If section not found in config.
    """
    # Currently only sheet1 is implemented
    if sheet_name != "sheet1":
        raise ValueError(f"Sheet '{sheet_name}' not yet implemented. Only 'sheet1' is supported.")
    return get_sheet1_section_expected_items(section_name)


def get_all_field_labels(sheet_name: str = "sheet1") -> dict[str, str]:
    """Get all field labels from all sections in config.

    Args:
        sheet_name: Sheet name for config lookup (default: "sheet1")

    Returns:
        Dictionary mapping field names to their primary PDF labels.

    Raises:
        ValueError: If sections not found in config.
    """
    # Currently only sheet1 is implemented
    if sheet_name != "sheet1":
        raise ValueError(f"Sheet '{sheet_name}' not yet implemented. Only 'sheet1' is supported.")

    field_labels = {}
    for section_name in get_sheet1_extraction_sections():
        field_mappings = get_sheet1_section_field_mappings(section_name)
        for field_name, field_spec in field_mappings.items():
            labels = field_spec.get("pdf_labels", [])
            if labels:
                field_labels[field_name] = labels[0]
    return field_labels


def _get_pdf_labels_for_field(section_spec: dict[str, Any], field_name: str) -> list[str]:
    """Get PDF label variations for a field from section spec.

    Args:
        section_spec: Section specification dict
        field_name: Field name (e.g., "cv_gastos_personal")

    Returns:
        List of possible PDF label strings
    """
    field_mappings = section_spec.get("field_mappings", {})
    field_spec = field_mappings.get(field_name, {})
    return field_spec.get("pdf_labels", [])


def _get_table_identifiers(section_spec: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Get unique and exclude items for table identification.

    Args:
        section_spec: Section specification dict

    Returns:
        Tuple of (unique_items, exclude_items)
    """
    identifiers = section_spec.get("table_identifiers", {})
    return (
        identifiers.get("unique_items", []),
        identifiers.get("exclude_items", []),
    )


def _get_extraction_labels(
    config: dict | None = None,
) -> tuple[list[str], list[str], dict[str, str]]:
    """Get extraction labels from config/sheet1/extraction.json.

    This function loads labels from the Sheet1 extraction config for all
    configured sections. Returns labels for nota_21, nota_22, and all fields.

    Args:
        config: Configuration dict (ignored, kept for backward compat)

    Returns:
        Tuple of (costo_venta_items, gasto_admin_items, field_labels)
        where costo_venta_items are from nota_21 and gasto_admin_items from nota_22

    Raises:
        ValueError: If config loading fails
    """
    # Get all extraction sections from config
    sections = get_sheet1_extraction_sections()
    if len(sections) < 2:
        raise ValueError(f"Expected at least 2 extraction sections in config, got {len(sections)}")

    # Build items lists for the first two cost sections
    section1_items = []
    section2_items = []
    field_labels = {}

    for i, section_name in enumerate(sections):
        section_spec = get_sheet1_section_spec(section_name)
        field_mappings = section_spec.get("field_mappings", {})

        for field_name, field_spec in field_mappings.items():
            labels = field_spec.get("pdf_labels", [])
            if labels:
                # Build field_labels for all sections
                field_labels[field_name] = labels[0]

                # Build items lists (excluding totals) for first two cost sections
                if not field_name.startswith("total_"):
                    if i == 0:
                        section1_items.append(labels[0])
                    elif i == 1:
                        section2_items.append(labels[0])

    return section1_items, section2_items, field_labels


@dataclass
class LineItem:
    """A single line item from the cost breakdown."""

    concepto: str
    ytd_actual: int | None = None
    ytd_anterior: int | None = None
    quarter_actual: int | None = None
    quarter_anterior: int | None = None


@dataclass
class SectionBreakdown:
    """Breakdown from a PDF section (e.g., nota_21, nota_22, ingresos).

    Generic container for extracted table data from any section defined
    in config/sheet1/extraction.json.
    """

    section_id: str  # e.g., "nota_21", "nota_22", "ingresos"
    section_title: str  # e.g., "Costo de Venta", "Gastos de Administración"
    items: list[LineItem] = field(default_factory=list)
    total_ytd_actual: int | None = None
    total_ytd_anterior: int | None = None
    total_quarter_actual: int | None = None
    total_quarter_anterior: int | None = None
    page_number: int | None = None

    def sum_items_ytd_actual(self) -> int:
        """Sum all YTD actual values (excluding total row)."""
        return sum(item.ytd_actual or 0 for item in self.items if "total" not in item.concepto.lower())

    def is_valid(self) -> bool:
        """Check if the sum of items equals the total."""
        if self.total_ytd_actual is None:
            return False
        calculated_sum = self.sum_items_ytd_actual()
        return calculated_sum == self.total_ytd_actual


@dataclass
class ValidationResult:
    """Result of cross-validation between PDF and XBRL."""

    field_name: str
    pdf_value: int | None
    xbrl_value: int | None
    match: bool
    source: str  # "pdf_only", "xbrl_only", "both"
    difference: int | None = None

    @property
    def status(self) -> str:
        """Return validation status string."""
        if self.source == "pdf_only":
            return "⚠ PDF only (no XBRL)"
        elif self.source == "xbrl_only":
            return "⚠ XBRL only (PDF extraction failed)"
        elif self.match:
            return "✓ Match"
        else:
            return f"✗ Mismatch (diff: {self.difference:,})"


@dataclass
class ExtractionResult:
    """Complete extraction result with optional validation.

    Uses sections dict for generic section access.
    """

    year: int
    quarter: int
    sections: dict[str, SectionBreakdown] = field(default_factory=dict)
    xbrl_available: bool = False
    xbrl_totals: dict[str, int | None] = field(default_factory=dict)
    validations: list[ValidationResult] = field(default_factory=list)
    validation_report: ValidationReport | None = None  # Full validation report
    source: str = "cmf"  # "cmf" or "pucobre.cl"
    pdf_path: Path | None = None
    xbrl_path: Path | None = None

    def get_section(self, section_id: str) -> SectionBreakdown | None:
        """Get a section by its ID."""
        return self.sections.get(section_id)

    def is_valid(self) -> bool:
        """Check if all validations passed."""
        if not self.validations:
            # No validations performed (PDF-only extraction)
            return len(self.sections) > 0 and all(s is not None for s in self.sections.values())
        return all(v.match for v in self.validations)


# =============================================================================
# Validation Result Dataclasses (Phase 1, 2, 4)
# =============================================================================


@dataclass
class SumValidationResult:
    """Result of line-item sum validation.

    Checks that sum of individual line items equals the total field.
    """

    description: str  # e.g., "Nota 21 - Costo de Venta"
    total_field: str  # e.g., "total_costo_venta"
    expected_total: int | None  # From PDF total row
    calculated_sum: int  # Sum of line items
    match: bool
    difference: int
    tolerance: int

    @property
    def status(self) -> str:
        """Return validation status string."""
        if self.expected_total is None:
            return "⚠ No total value to compare"
        elif self.match:
            return f"✓ Sum matches total ({self.calculated_sum:,})"
        else:
            return f"✗ Sum mismatch: items={self.calculated_sum:,}, total={self.expected_total:,} (diff: {self.difference})"


@dataclass
class CrossValidationResult:
    """Result of cross-validation formula check.

    Evaluates formulas like: gross_profit == ingresos - abs(cost_of_sales)
    """

    description: str  # e.g., "Gross Profit = Revenue - Cost of Sales"
    formula: str  # e.g., "gross_profit == ingresos_ordinarios - abs(total_costo_venta)"
    expected_value: int | None  # Left side of formula
    calculated_value: int | None  # Right side of formula
    match: bool
    difference: int | None
    tolerance: int
    missing_facts: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """Return validation status string."""
        if self.missing_facts:
            return f"⚠ Skipped - missing: {', '.join(self.missing_facts)}"
        elif self.match:
            return f"✓ {self.description}: {self.expected_value:,}"
        else:
            return f"✗ {self.description}: expected={self.expected_value}, calculated={self.calculated_value} (diff: {self.difference})"


@dataclass
class ValidationReport:
    """Aggregated validation results for unified reporting."""

    sum_validations: list[SumValidationResult] = field(default_factory=list)
    cross_validations: list[CrossValidationResult] = field(default_factory=list)
    pdf_xbrl_validations: list[ValidationResult] = field(default_factory=list)
    reference_issues: list[str] | None = None  # None = not run, [] = passed, [...] = issues

    def has_failures(self) -> bool:
        """Check if any validation failed."""
        sum_ok = all(v.match for v in self.sum_validations)
        cross_ok = all(v.match for v in self.cross_validations)
        pdf_xbrl_ok = all(v.match for v in self.pdf_xbrl_validations)
        ref_ok = self.reference_issues is None or len(self.reference_issues) == 0
        return not (sum_ok and cross_ok and pdf_xbrl_ok and ref_ok)

    def has_sum_failures(self) -> bool:
        """Check if any sum validation failed."""
        return any(not v.match for v in self.sum_validations)

    def has_reference_failures(self) -> bool:
        """Check if reference validation failed."""
        return self.reference_issues is not None and len(self.reference_issues) > 0


def format_validation_report(report: ValidationReport, verbose: bool = True) -> str:
    """Format validation report for display.

    Args:
        report: ValidationReport to format
        verbose: If True, include more detail

    Returns:
        Formatted string for display
    """
    lines = []
    separator = "═" * 60

    lines.append(separator)
    lines.append("                    VALIDATION REPORT")
    lines.append(separator)
    lines.append("")

    # Sum Validations
    if report.sum_validations:
        lines.append("Sum Validations:")
        for v in report.sum_validations:
            lines.append(f"  {v.status}")
    else:
        lines.append("Sum Validations: (none configured)")
    lines.append("")

    # PDF ↔ XBRL Validations
    if report.pdf_xbrl_validations:
        lines.append("PDF ↔ XBRL Validations:")
        for v in report.pdf_xbrl_validations:
            if v.source == "both":
                symbol = "✓" if v.match else "✗"
                if v.match:
                    lines.append(f"  {symbol} {v.field_name}: {v.pdf_value:,} (PDF) = {v.xbrl_value:,} (XBRL)")
                else:
                    lines.append(
                        f"  {symbol} {v.field_name}: {v.pdf_value:,} (PDF) ≠ {v.xbrl_value:,} (XBRL) [diff: {v.difference}]"
                    )
            elif v.source == "xbrl_only":
                lines.append(f"  ⚠ {v.field_name}: {v.xbrl_value:,} (XBRL only, used as source)")
            else:
                lines.append(f"  ⚠ {v.field_name}: {v.pdf_value:,} (PDF only, no XBRL)")
    else:
        lines.append("PDF ↔ XBRL Validations: (no XBRL available)")
    lines.append("")

    # Cross-Validations
    if report.cross_validations:
        lines.append("Cross-Validations:")
        for v in report.cross_validations:
            lines.append(f"  {v.status}")
    else:
        lines.append("Cross-Validations: (none configured)")
    lines.append("")

    # Reference Validation
    if report.reference_issues is None:
        lines.append("Reference Validation: (not run - use --validate-reference to enable)")
    elif len(report.reference_issues) == 0:
        lines.append("Reference Validation: ✓ All values match reference data")
    else:
        lines.append("Reference Validation: ✗ MISMATCHES FOUND")
        for issue in report.reference_issues:
            lines.append(f"  • {issue}")

    lines.append("")
    lines.append(separator)

    return "\n".join(lines)


def log_validation_report(report: ValidationReport) -> None:
    """Log validation results with appropriate log levels.

    Args:
        report: ValidationReport to log
    """
    # Sum validations
    for v in report.sum_validations:
        if v.match:
            logger.info(f"✓ Sum: {v.description}")
        else:
            logger.warning(f"✗ Sum: {v.description} - mismatch (diff: {v.difference})")

    # PDF ↔ XBRL validations (already logged during _validate_sheet1_with_xbrl)

    # Cross-validations (already logged during _run_cross_validations)

    # Reference validation - prominent warning
    if report.reference_issues is not None and len(report.reference_issues) > 0:
        logger.warning("=" * 60)
        logger.warning("⚠️  REFERENCE DATA MISMATCH - Values differ from known-good data!")
        logger.warning("=" * 60)
        for issue in report.reference_issues:
            logger.warning(f"  • {issue}")
        logger.warning("Review extraction or update reference_data.json if values are correct.")
        logger.warning("=" * 60)


def parse_chilean_number(value: str | None) -> int | None:
    """Parse a Chilean-formatted number.

    Chilean format uses:
    - Period as thousands separator: 30.294 = 30,294
    - Parentheses for negatives: (30.294) = -30,294
    - Comma for decimals (rare in financial statements)

    Args:
        value: String value to parse

    Returns:
        Integer value or None if parsing fails
    """
    if not value:
        return None

    # Clean the string
    value = str(value).strip()

    # Check for negative (parentheses)
    is_negative = "(" in value and ")" in value

    # Remove non-numeric characters except period and minus
    value = re.sub(r"[^\d.\-]", "", value)

    if not value or value in (".", "-"):
        return None

    try:
        # Remove period (thousands separator) and convert
        value = value.replace(".", "")
        result = int(value)
        return -abs(result) if is_negative else result
    except ValueError:
        return None


def find_text_page(
    pdf_path: Path,
    required_texts: list[str],
    optional_texts: list[str] | None = None,
    min_required: int | None = None,
    min_optional: int = 0,
) -> int | None:
    """Find the first page containing required text strings.

    Generic PDF page finder that searches for text patterns. This is the
    primitive function used by find_section_page for config-driven lookups.

    Args:
        pdf_path: Path to the PDF file
        required_texts: List of text strings that MUST appear (at least one).
            Matching is case-insensitive.
        optional_texts: Additional texts to look for (for scoring/validation).
            If provided with min_optional > 0, page must have at least that many.
        min_required: Minimum number of required_texts that must match.
            Defaults to 1 (any single match). Set higher for stricter matching.
        min_optional: Minimum number of optional_texts that must match.
            Defaults to 0 (no optional requirement).

    Returns:
        0-indexed page number or None if not found

    Example:
        >>> # Find page with "Nota 21" header
        >>> find_text_page(pdf, ["21. costo", "nota 21"])

        >>> # Find page with header AND at least 3 detail items
        >>> find_text_page(pdf, ["estado de resultados"],
        ...                optional_texts=["ingresos", "gastos", "utilidad"],
        ...                min_optional=2)
    """
    if min_required is None:
        min_required = 1

    # Normalize required texts for matching (case-insensitive + accent-stripped)
    required_normalized = []
    for text in required_texts:
        required_normalized.append(text.lower())
        normalized = _normalize_for_matching(text)
        if normalized != text.lower():
            required_normalized.append(normalized)

    # Normalize optional texts if provided
    optional_lower = []
    if optional_texts:
        for text in optional_texts:
            optional_lower.append(text.lower())
            normalized = _normalize_for_matching(text)
            if normalized != text.lower():
                optional_lower.append(normalized)

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").lower()

            # Count required text matches
            required_count = sum(1 for req in required_normalized if req in text)
            if required_count < min_required:
                continue

            # Count optional text matches if specified
            if optional_lower and min_optional > 0:
                optional_count = sum(1 for opt in optional_lower if opt in text)
                if optional_count < min_optional:
                    continue

            logger.debug(f"Found text match on page {page_idx + 1}")
            return page_idx

    return None


def find_section_page(
    pdf_path: Path,
    section_name: str,
    year: int | None = None,
    quarter: int | None = None,
) -> int | None:
    """Find the page number where a config-defined section exists.

    Config-driven wrapper around find_text_page. Uses config/sheet1/extraction.json
    to get search patterns and unique identifiers for the section.

    Args:
        pdf_path: Path to the PDF file
        section_name: Section name from config/sheet1/extraction.json (e.g., "nota_21", "ingresos")
        year: Optional year for period-specific specs (reserved for future use)
        quarter: Optional quarter for period-specific specs (reserved for future use)

    Returns:
        0-indexed page number or None if not found

    See Also:
        find_text_page: Generic text search (use directly for non-config searches)
    """
    section_spec = get_sheet1_section_spec(section_name)

    # Get unique items for this section (used to identify correct page)
    unique_items, _ = _get_table_identifiers(section_spec)

    if not unique_items:
        raise ValueError(
            f"No unique_items defined in config/sheet1/extraction.json for section '{section_name}'. "
            "Please add table_identifiers.unique_items to the section spec."
        )

    # Get search patterns from spec (check both top-level and pdf_fallback)
    search_patterns = section_spec.get("search_patterns", [])
    if not search_patterns:
        # Fallback to pdf_fallback.search_patterns (used by ingresos section)
        pdf_fallback = section_spec.get("pdf_fallback", {})
        search_patterns = pdf_fallback.get("search_patterns", [])
    if not search_patterns:
        raise ValueError(
            f"No search_patterns defined in config/sheet1/extraction.json for section '{section_name}'. "
            "Please add search_patterns to the section spec or pdf_fallback."
        )

    # Get validation requirements from spec
    validation = section_spec.get("validation", {})
    min_detail_items = validation.get("min_detail_items", 3)
    has_totales = validation.get("has_totales_row", True)

    # Build optional texts from unique items + totales requirement
    optional_texts = list(unique_items)
    if has_totales:
        optional_texts.append("totales")

    # Use find_text_page with search patterns as required, unique items as optional
    page_idx = find_text_page(
        pdf_path,
        required_texts=search_patterns,
        optional_texts=optional_texts,
        min_required=1,
        min_optional=min_detail_items,
    )

    if page_idx is not None:
        logger.info(f"Found section '{section_name}' with details on page {page_idx + 1}")

    return page_idx


def extract_table_from_page(
    pdf_path: Path,
    page_index: int,
    expected_items: list[str],
    nota_number: int = 0,
    section_name: str = "",
    year: int | None = None,
    quarter: int | None = None,
) -> list[dict[str, Any]]:
    """Extract cost table data from a specific page.

    Uses config/sheet1/extraction.json to get table identifiers for scoring.

    Args:
        pdf_path: Path to the PDF file
        page_index: 0-indexed page number
        expected_items: List of expected line item names
        nota_number: Nota number (21 or 22) to help select correct table (deprecated, use section_name)
        section_name: Section name from config/sheet1/extraction.json (preferred over nota_number)
        year: Optional year for period-specific specs
        quarter: Optional quarter for period-specific specs

    Returns:
        List of dictionaries with concepto and value columns
    """
    # Get table identifiers from spec
    if not section_name and nota_number:
        section_name = f"nota_{nota_number}"
    if section_name:
        section_spec = get_sheet1_section_spec(section_name)
        unique_items, exclude_items = _get_table_identifiers(section_spec)
    else:
        unique_items, exclude_items = [], []

    with pdfplumber.open(pdf_path) as pdf:
        if page_index >= len(pdf.pages):
            return []

        page = pdf.pages[page_index]

        # Try to extract tables
        tables = page.extract_tables()

        if not tables:
            logger.warning(f"No tables found on page {page_index + 1}")
            return []

        # Find the table for the specific Nota using spec-driven scoring
        best_table = None
        best_score = 0

        for table in tables:
            if not table:
                continue

            table_text = str(table).lower()

            # Score based on matches with expected items
            match_count = sum(1 for item in expected_items if item.lower() in table_text)

            # Boost score for unique identifiers from spec
            for unique_item in unique_items:
                # Check both original and normalized versions
                if unique_item.lower() in table_text:
                    match_count += 5
                normalized = _normalize_for_matching(unique_item)
                if normalized in table_text and normalized != unique_item.lower():
                    match_count += 5

            # Penalize for exclude items from spec
            for exclude_item in exclude_items:
                if exclude_item.lower() in table_text:
                    match_count -= 5
                normalized = _normalize_for_matching(exclude_item)
                if normalized in table_text and normalized != exclude_item.lower():
                    match_count -= 5

            if match_count > best_score:
                best_score = match_count
                best_table = table

        if best_table is None or best_score < 3:
            logger.warning(f"Could not find expected cost table on page {page_index + 1}")
            return []

        # Parse the table rows
        return _parse_cost_table(best_table, expected_items)


def _normalize_for_matching(text: str) -> str:
    """Normalize text for fuzzy matching.

    Removes accents, punctuation, and extra whitespace for comparison.

    Args:
        text: Text to normalize

    Returns:
        Normalized lowercase text
    """
    # Normalize unicode to decomposed form (separate base char from accent)
    text = unicodedata.normalize("NFD", text)
    # Remove combining characters (accents)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Lowercase
    text = text.lower()
    # Remove periods and extra punctuation (but keep spaces)
    text = re.sub(r"[.,;:()\[\]]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _match_item(concept: str, expected_items: list[str]) -> str | None:
    """Match a concept text against expected items.

    Uses normalized fuzzy matching to handle accent and punctuation differences.
    Tries more specific (longer) items first to avoid false matches.

    Args:
        concept: Concept text from PDF
        expected_items: List of expected item names

    Returns:
        Matched expected item or None
    """
    norm_concept = _normalize_for_matching(concept)

    # Sort expected items by length (descending) to match more specific items first
    # This prevents "Servicios de terceros" from matching "Servicios mineros de terceros"
    sorted_items = sorted(expected_items, key=lambda x: len(x), reverse=True)

    for expected in sorted_items:
        norm_expected = _normalize_for_matching(expected)

        # Check if normalized expected is in normalized concept
        if norm_expected in norm_concept:
            return expected

        # Also check for partial matches (handle variations like "amort." vs "amort")
        # Split into words and check if most words match
        expected_words = norm_expected.split()
        concept_words = norm_concept.split()

        # Count how many expected words appear in the concept
        matching_words = sum(1 for word in expected_words if any(word in cw for cw in concept_words))

        # If at least 80% of expected words match, consider it a match
        if expected_words and matching_words >= len(expected_words) * 0.8:
            return expected

    return None


def _parse_cost_table(table: list[list[str | None]], expected_items: list[str]) -> list[dict[str, Any]]:
    """Parse a cost breakdown table.

    Handles tables where multiple items are concatenated with newlines into single cells.
    Example: One cell contains "Gastos en personal\\nMateriales y repuestos\\n..."
    and adjacent cells contain corresponding values "19.721\\n23.219\\n..."

    Args:
        table: Raw table data from pdfplumber
        expected_items: List of expected line item names

    Returns:
        List of parsed row dictionaries
    """
    parsed_rows = []

    for row in table:
        if not row or not any(row):
            continue

        # First cell contains concept name(s)
        concept_cell = str(row[0] or "").strip()

        # Check if this is a multi-line cell (items concatenated with newlines)
        if "\n" in concept_cell:
            # Split the concept names and corresponding values
            concepts = concept_cell.split("\n")

            # Get value columns (each also has newline-separated values)
            value_columns: list[list[str]] = []
            for cell in row[1:]:
                if cell:
                    values = str(cell).strip().split("\n")
                    value_columns.append(values)
                else:
                    value_columns.append([])

            # Match concepts with their values
            for idx, concept in enumerate(concepts):
                concept = concept.strip()
                if not concept:
                    continue

                # Check if any expected item is in this concept (using fuzzy matching)
                matched_item = _match_item(concept, expected_items)

                # Also check for "Totales" or "Total"
                if not matched_item and "total" in concept.lower():
                    matched_item = "Totales"

                if matched_item:
                    # Get corresponding values from each column
                    values = []
                    for col_values in value_columns:
                        if idx < len(col_values):
                            parsed = parse_chilean_number(col_values[idx])
                            if parsed is not None:
                                values.append(parsed)

                    parsed_rows.append({
                        "concepto": matched_item,
                        "values": values,
                    })

        else:
            # Single-line row (normal format)
            row_text = concept_cell

            # Check if any expected item is in this row (using fuzzy matching)
            matched_item = _match_item(row_text, expected_items)

            # Also check for "Totales" or "Total"
            if not matched_item and "total" in row_text.lower():
                matched_item = "Totales"

            if matched_item:
                # Extract numeric values from remaining columns
                values = []
                for cell in row[1:]:
                    if cell:
                        # Handle multi-value cells (take first value)
                        cell_str = str(cell).split("\n")[0].strip()
                        parsed = parse_chilean_number(cell_str)
                        if parsed is not None:
                            values.append(parsed)

                parsed_rows.append({
                    "concepto": matched_item,
                    "values": values,
                })

    return parsed_rows


def extract_pdf_section(
    pdf_path: Path,
    section_name: str,
    sheet_name: str = "sheet1",
    year: int | None = None,
    quarter: int | None = None,
) -> SectionBreakdown | None:
    """Extract a section from PDF using config-driven rules.

    Generic extraction function that works with any section defined in
    config/<sheet>/extraction.json.

    Args:
        pdf_path: Path to Estados Financieros PDF
        section_name: Section key from config (e.g., "nota_21", "nota_22")
        sheet_name: Sheet name for config lookup (default: "sheet1")
        year: Optional year for period-specific specs
        quarter: Optional quarter for period-specific specs

    Returns:
        SectionBreakdown object or None if extraction fails
    """
    # Get section spec from config
    section_spec = get_sheet1_section_spec(section_name)
    section_title = section_spec.get("title", section_name)

    # Get expected items for this section
    expected_items = get_section_expected_labels(section_name, sheet_name)

    # Find the page
    page_idx = find_section_page(pdf_path, section_name, year, quarter)

    if page_idx is None:
        # Try fallback: some sections share pages (e.g., nota_22 often on same page as nota_21)
        fallback = get_section_fallback(section_name)
        if fallback:
            page_idx = find_section_page(pdf_path, fallback, year, quarter)

    if page_idx is None:
        logger.error(f"Could not find section '{section_name}' in PDF")
        return None

    # Extract table from page
    rows = extract_table_from_page(
        pdf_path, page_idx, expected_items, section_name=section_name, year=year, quarter=quarter
    )

    if not rows:
        # Try next page (table might span pages)
        rows = extract_table_from_page(
            pdf_path, page_idx + 1, expected_items, section_name=section_name, year=year, quarter=quarter
        )

    if not rows:
        logger.error(f"Could not extract table for section '{section_name}'")
        return None

    # Build SectionBreakdown
    breakdown = SectionBreakdown(
        section_id=section_name,
        section_title=section_title,
        page_number=page_idx + 1,
    )

    for row in rows:
        concepto = row["concepto"]
        values = row.get("values", [])

        if concepto.lower() in ("totales", "total"):
            breakdown.total_ytd_actual = values[0] if len(values) > 0 else None
            breakdown.total_ytd_anterior = values[1] if len(values) > 1 else None
            breakdown.total_quarter_actual = values[2] if len(values) > 2 else None
            breakdown.total_quarter_anterior = values[3] if len(values) > 3 else None
        else:
            item = LineItem(
                concepto=concepto,
                ytd_actual=values[0] if len(values) > 0 else None,
                ytd_anterior=values[1] if len(values) > 1 else None,
                quarter_actual=values[2] if len(values) > 2 else None,
                quarter_anterior=values[3] if len(values) > 3 else None,
            )
            breakdown.items.append(item)

    logger.info(f"Extracted {len(breakdown.items)} items from section '{section_name}'")
    return breakdown


def extract_ingresos_from_pdf(pdf_path: Path) -> int | None:
    """Extract Ingresos de actividades ordinarias from Estado de Resultados page.

    This is a PDF fallback for when XBRL is not available. Uses the generic
    find_section_page for page location, then extracts the Ingresos value directly
    from the table (Estado de Resultados has a different structure than Nota tables).

    Args:
        pdf_path: Path to Estados Financieros PDF

    Returns:
        Ingresos value in thousands USD, or None if extraction fails
    """
    # Get ingresos section spec from config
    ingresos_spec = get_sheet1_section_spec("ingresos")
    field_mappings = ingresos_spec.get("field_mappings", {})
    ingresos_mapping = field_mappings.get("ingresos_ordinarios", {})
    match_keywords = ingresos_mapping.get("match_keywords")
    if not match_keywords:
        raise KeyError(
            "ingresos.field_mappings.ingresos_ordinarios.match_keywords missing from config. "
            "Add it to config/sheet1/extraction.json."
        )

    # Get minimum value threshold from config (no default - must be in config)
    pdf_config = get_ingresos_pdf_fallback_config()
    min_threshold = pdf_config["min_value_threshold"]

    # Find the Estado de Resultados page using generic function
    page_idx = find_section_page(pdf_path, "ingresos")
    if page_idx is None:
        logger.warning("Could not find Estado de Resultados page for Ingresos extraction")
        return None

    # Estado de Resultados table has a special structure:
    # Multi-line cells where "Ganancia" header has no value, so indices don't align.
    # Extract the value by finding the Ingresos line and getting the FIRST positive value
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_idx]
        tables = page.extract_tables()

        for table in tables:
            for row in table:
                if not row or len(row) < 2:
                    continue

                first_col = str(row[0] or "").lower()
                # Check if this cell contains Ingresos label
                if not all(kw.lower() in first_col for kw in match_keywords):
                    continue

                # Found cell with Ingresos - parse the multi-line structure
                labels = first_col.split("\n")
                values_col = str(row[1] or "") if len(row) > 1 else ""
                values = values_col.split("\n")

                # Find the Ingresos label line index
                ingresos_idx = None
                for i, label in enumerate(labels):
                    if all(kw.lower() in label.lower() for kw in match_keywords):
                        ingresos_idx = i
                        break

                if ingresos_idx is not None:
                    # Count value-bearing labels before Ingresos to find the value index
                    # Labels with nota numbers (digits at end) have values
                    value_idx = 0
                    for i in range(ingresos_idx):
                        label = labels[i].strip()
                        # Headers like "Ganancia" have no nota number and no value
                        if label and any(c.isdigit() for c in label.split()[-1] if label.split()):
                            value_idx += 1

                    if value_idx < len(values):
                        value = parse_chilean_number(values[value_idx].strip())
                        if value is not None and value > min_threshold:
                            logger.info(f"Extracted Ingresos from PDF page {page_idx + 1}: {value:,}")
                            return value

                # Fallback: Ingresos is typically the first positive large value
                for val_str in values:
                    value = parse_chilean_number(val_str.strip())
                    if value is not None and value > min_threshold:
                        logger.info(f"Extracted Ingresos from PDF page {page_idx + 1}: {value:,}")
                        return value

    logger.warning("Could not extract Ingresos from Estado de Resultados page")
    return None


def extract_xbrl_totals(xbrl_path: Path) -> dict[str, int | None]:
    """Extract relevant totals from XBRL file using config-driven fact names.

    Uses fact_mappings from config/sheet1/xbrl_mappings.json to look up the correct
    XBRL fact names for each field.

    Args:
        xbrl_path: Path to XBRL file

    Returns:
        Dictionary with cost_of_sales, admin_expense, and ingresos totals
    """
    try:
        data = parse_xbrl_file(xbrl_path)
    except Exception as e:
        logger.error(f"Failed to parse XBRL: {e}")
        return {"cost_of_sales": None, "admin_expense": None, "ingresos": None}

    # Get field->result mapping from config
    field_to_result = get_sheet1_result_key_mapping()

    # Initialize result dict with all keys set to None
    result: dict[str, int | None] = {key: None for key in set(field_to_result.values())}

    scaling_factor = get_xbrl_scaling_factor()

    for field_name, result_key in field_to_result.items():
        fact_mapping = get_sheet1_xbrl_fact_mapping(field_name)
        if not fact_mapping:
            logger.debug(f"No XBRL mapping found for field: {field_name}")
            continue

        # Try primary fact name first
        primary_fact = fact_mapping.get("primary")
        fallback_facts = fact_mapping.get("fallbacks", [])

        facts = []
        if primary_fact:
            facts = get_facts_by_name(data, primary_fact)

        # Try fallbacks if primary not found
        if not facts:
            for fallback in fallback_facts:
                facts = get_facts_by_name(data, fallback)
                if facts:
                    logger.debug(f"Using fallback XBRL fact '{fallback}' for {field_name}")
                    break

        # Extract value from first matching fact
        for fact in facts:
            if fact.get("value"):
                try:
                    raw_value = int(float(fact["value"]))
                    # Apply scaling factor for fields that need it
                    if fact_mapping.get("apply_scaling", False):
                        result[result_key] = raw_value // scaling_factor
                        logger.debug(f"Scaled {field_name}: {raw_value} -> {result[result_key]}")
                    else:
                        result[result_key] = raw_value
                    break
                except (ValueError, TypeError):
                    continue

    return result


# =============================================================================
# Unified Validation API
# =============================================================================


def run_sheet1_validations(
    data: Sheet1Data,
    xbrl_totals: dict[str, int | None] | None = None,
    *,
    run_sum_validations: bool = True,
    run_pdf_xbrl_validations: bool = True,
    run_cross_validations: bool = True,
    use_xbrl_fallback: bool = True,
) -> ValidationReport:
    """Unified validation entry point for Sheet1 data.

    Runs config-driven validations:
    1. Sum validations: Line items should sum to totals
    2. PDF ↔ XBRL comparison: Compare extracted values against XBRL
    3. Cross-validations: Accounting identity checks (e.g., gross_profit formula)

    All config is loaded from config/sheet1/xbrl_mappings.json:
    - sum_tolerance: Global tolerance for comparisons
    - total_validations: Sum validation rules
    - cross_validations: Formula-based validation rules
    - pdf_xbrl_validations: PDF↔XBRL comparison definitions

    Args:
        data: Sheet1Data with extracted values
        xbrl_totals: XBRL totals dict (None if unavailable)
        run_sum_validations: Enable sum validation checks
        run_pdf_xbrl_validations: Enable PDF↔XBRL comparison
        run_cross_validations: Enable cross-validation formulas
        use_xbrl_fallback: If True, set missing PDF values from XBRL

    Returns:
        ValidationReport with all validation results
    """
    sum_results: list[SumValidationResult] = []
    pdf_xbrl_results: list[ValidationResult] = []
    cross_results: list[CrossValidationResult] = []

    # Phase 1: Sum validations
    if run_sum_validations:
        sum_results = _run_sum_validations(data)

    # Phase 2: PDF ↔ XBRL validations
    if run_pdf_xbrl_validations:
        pdf_xbrl_results = _run_pdf_xbrl_validations(data, xbrl_totals, use_xbrl_fallback)

    # Phase 3: Cross-validations
    if run_cross_validations:
        cross_results = _run_cross_validations(data, xbrl_totals)

    return ValidationReport(
        sum_validations=sum_results,
        cross_validations=cross_results,
        pdf_xbrl_validations=pdf_xbrl_results,
        reference_issues=None,  # Reference validation done separately via validate_sheet1_against_reference
    )


def _run_pdf_xbrl_validations(
    data: Sheet1Data,
    xbrl_totals: dict[str, int | None] | None,
    use_fallback: bool = True,
) -> list[ValidationResult]:
    """Config-driven PDF ↔ XBRL comparison.

    Uses pdf_xbrl_validations from config/sheet1/xbrl_mappings.json.

    Args:
        data: Sheet1Data with extracted PDF values
        xbrl_totals: XBRL totals dict (or None if unavailable)
        use_fallback: If True, set missing PDF values from XBRL

    Returns:
        List of ValidationResult objects
    """
    results = []
    tolerance = get_sheet1_sum_tolerance()
    pdf_xbrl_config = get_sheet1_pdf_xbrl_validations()

    for validation in pdf_xbrl_config:
        field_name = validation["field_name"]
        xbrl_key = validation["xbrl_key"]
        display_name = validation["display_name"]
        xbrl_value = xbrl_totals.get(xbrl_key) if xbrl_totals else None
        pdf_value = data.get_value(field_name)

        if xbrl_value is not None:
            if pdf_value is not None:
                # Both available - compare using helper
                match, diff = _compare_with_tolerance(pdf_value, xbrl_value, tolerance)

                results.append(
                    ValidationResult(
                        field_name=display_name,
                        pdf_value=pdf_value,
                        xbrl_value=xbrl_value,
                        match=match,
                        source="both",
                        difference=diff if not match else None,
                    )
                )

                if match:
                    logger.info(f"✓ {display_name} matches XBRL: {pdf_value:,}")
                else:
                    logger.warning(
                        f"✗ {display_name} mismatch - PDF: {pdf_value:,}, XBRL: {xbrl_value:,} (diff: {diff})"
                    )
            else:
                # PDF extraction failed - use XBRL value if fallback enabled
                if use_fallback:
                    logger.info(f"Using XBRL value for {display_name}: {xbrl_value:,}")
                    data.set_value(field_name, xbrl_value)

                results.append(
                    ValidationResult(
                        field_name=display_name,
                        pdf_value=None,
                        xbrl_value=xbrl_value,
                        match=True,  # Used XBRL as source
                        source="xbrl_only",
                    )
                )
        elif pdf_value is not None:
            # PDF only (no XBRL)
            results.append(
                ValidationResult(
                    field_name=display_name,
                    pdf_value=pdf_value,
                    xbrl_value=None,
                    match=True,  # Can't validate without XBRL
                    source="pdf_only",
                )
            )

    return results


def _section_breakdowns_to_sheet1data(
    nota_21: SectionBreakdown | None,
    nota_22: SectionBreakdown | None,
    year: int = 0,
    quarter: int = 0,
) -> Sheet1Data:
    """Convert SectionBreakdown objects to Sheet1Data for validation.

    Creates a minimal Sheet1Data with only the total fields populated.
    Uses config-driven mapping from section_total_mapping.

    .. deprecated::
        Use :func:`sections_to_sheet1data` from puco_eeff.sheets.sheet1 instead
        for full config-driven conversion with detail fields.

    Args:
        nota_21: Nota 21 SectionBreakdown (or None)
        nota_22: Nota 22 SectionBreakdown (or None)
        year: Year for period label
        quarter: Quarter for period label

    Returns:
        Sheet1Data with totals populated
    """
    import warnings

    warnings.warn(
        "_section_breakdowns_to_sheet1data() is deprecated. "
        "Use sections_to_sheet1data() from puco_eeff.sheets.sheet1 instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Always use format_quarter_label for consistency
    quarter_label = format_quarter_label(year, quarter)
    data = Sheet1Data(quarter=quarter_label, year=year, quarter_num=quarter)

    # Use config-driven mapping to set totals
    section_total_mapping = get_sheet1_section_total_mapping()

    if nota_21 and "nota_21" in section_total_mapping:
        field_name = section_total_mapping["nota_21"]
        data.set_value(field_name, nota_21.total_ytd_actual)

    if nota_22 and "nota_22" in section_total_mapping:
        field_name = section_total_mapping["nota_22"]
        data.set_value(field_name, nota_22.total_ytd_actual)

    return data


def validate_extraction(
    pdf_nota_21: SectionBreakdown | None,
    pdf_nota_22: SectionBreakdown | None,
    xbrl_totals: dict[str, int | None] | None,
) -> list[ValidationResult]:
    """Cross-validate PDF extraction against XBRL totals.

    .. deprecated::
        Use :func:`run_sheet1_validations` instead for full validation support.
        This function is kept for backward compatibility with extract_detailed_costs().

    Args:
        pdf_nota_21: Extracted Nota 21 from PDF
        pdf_nota_22: Extracted Nota 22 from PDF
        xbrl_totals: Totals extracted from XBRL (or None if no XBRL)

    Returns:
        List of validation results (PDF↔XBRL comparison only)
    """
    import warnings

    warnings.warn(
        "validate_extraction() is deprecated. Use run_sheet1_validations() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Convert SectionBreakdowns to Sheet1Data
    data = _section_breakdowns_to_sheet1data(pdf_nota_21, pdf_nota_22)

    # Run unified validation with only PDF↔XBRL enabled (legacy behavior)
    report = run_sheet1_validations(
        data,
        xbrl_totals,
        run_sum_validations=False,
        run_pdf_xbrl_validations=True,
        run_cross_validations=False,
        use_xbrl_fallback=False,  # Legacy: don't modify data
    )

    return report.pdf_xbrl_validations


def extract_detailed_costs(
    year: int,
    quarter: int,
    validate: bool = True,
) -> ExtractionResult:
    """Extract detailed cost breakdowns for a period.

    This function:
    1. Locates the Estados Financieros PDF using config patterns
    2. Extracts Nota 21 (Costo de Venta) and Nota 22 (Gastos Admin)
    3. If XBRL is available, extracts totals for validation
    4. Cross-validates PDF totals against XBRL

    Works with both CMF Chile (with XBRL) and Pucobre.cl fallback (PDF only).
    File patterns are read from config/config.json.

    Args:
        year: Year of the financial statement
        quarter: Quarter (1-4)
        validate: If True, validate against XBRL when available

    Returns:
        ExtractionResult with breakdowns and validation status
    """
    paths = get_period_paths(year, quarter)
    raw_dir = paths["raw_pdf"]

    # Find the PDF file using config patterns
    pdf_path = find_file_with_alternatives(raw_dir, "estados_financieros_pdf", year, quarter)
    if not pdf_path:
        pdf_path = raw_dir / format_filename("estados_financieros_pdf", year, quarter)

    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return ExtractionResult(
            year=year,
            quarter=quarter,
            xbrl_available=False,
        )

    # Determine source (check for pucobre_combined file)
    combined_path = find_file_with_alternatives(raw_dir, "pucobre_combined", year, quarter)
    source = "pucobre.cl" if (combined_path and combined_path.exists()) else "cmf"

    result = ExtractionResult(
        year=year,
        quarter=quarter,
        source=source,
        pdf_path=pdf_path,
    )

    # Extract from PDF
    nota_21 = extract_pdf_section(pdf_path, "nota_21")
    nota_22 = extract_pdf_section(pdf_path, "nota_22")
    if nota_21 is not None:
        result.sections["nota_21"] = nota_21
    if nota_22 is not None:
        result.sections["nota_22"] = nota_22

    # Check for XBRL using config patterns
    xbrl_dir = paths["raw_xbrl"]
    xbrl_path = find_file_with_alternatives(xbrl_dir, "estados_financieros_xbrl", year, quarter)
    if not xbrl_path:
        xbrl_path = xbrl_dir / format_filename("estados_financieros_xbrl", year, quarter)

    if xbrl_path and xbrl_path.exists():
        result.xbrl_available = True
        result.xbrl_path = xbrl_path

        if validate:
            xbrl_totals = extract_xbrl_totals(xbrl_path)

            # Copy XBRL totals using config-driven key mapping (no hard-coded keys)
            result_key_mapping = get_sheet1_result_key_mapping()
            for _field_name, xbrl_key in result_key_mapping.items():
                if xbrl_key in xbrl_totals:
                    result.xbrl_totals[xbrl_key] = xbrl_totals.get(xbrl_key)

            # Build Sheet1Data from sections using config-driven mapping
            sheet1_data = sections_to_sheet1data(result.sections, year, quarter)
            sheet1_data.xbrl_available = True
            sheet1_data.source = source

            # Run unified validation (sum, PDF↔XBRL, cross-validations)
            report = run_sheet1_validations(
                sheet1_data,
                xbrl_totals,
                run_sum_validations=True,
                run_pdf_xbrl_validations=True,
                run_cross_validations=True,
                use_xbrl_fallback=True,  # Update totals from XBRL if PDF missing
            )

            # Store validation results (PDF↔XBRL results for backward compat)
            result.validations = report.pdf_xbrl_validations
            result.validation_report = report
    else:
        logger.info(f"No XBRL available for {year} Q{quarter} - using PDF only")
        result.xbrl_available = False

        if validate:
            # Build Sheet1Data from sections for PDF-only validation
            sheet1_data = sections_to_sheet1data(result.sections, year, quarter)
            sheet1_data.xbrl_available = False
            sheet1_data.source = source

            # Run sum validations only (no XBRL for comparison or cross-validation)
            report = run_sheet1_validations(
                sheet1_data,
                None,
                run_sum_validations=True,
                run_pdf_xbrl_validations=True,  # Will produce pdf_only results
                run_cross_validations=False,  # Skip cross-validation without XBRL
                use_xbrl_fallback=False,
            )

            result.validations = report.pdf_xbrl_validations
            result.validation_report = report

    return result


def save_extraction_result(result: ExtractionResult, output_dir: Path | None = None) -> Path:
    """Save extraction result to JSON file.

    Args:
        result: ExtractionResult to save
        output_dir: Output directory (defaults to processed dir)

    Returns:
        Path to saved file
    """
    if output_dir is None:
        paths = get_period_paths(result.year, result.quarter)
        output_dir = paths["processed"]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "detailed_costs.json"

    # Convert to serializable dict
    nota_21 = result.sections.get("nota_21")
    nota_22 = result.sections.get("nota_22")
    data = {
        "period": f"{result.year}_Q{result.quarter}",
        "source": result.source,
        "pdf_path": str(result.pdf_path) if result.pdf_path else None,
        "xbrl_path": str(result.xbrl_path) if result.xbrl_path else None,
        "xbrl_available": result.xbrl_available,
        "nota_21": _breakdown_to_dict(nota_21) if nota_21 else None,
        "nota_22": _breakdown_to_dict(nota_22) if nota_22 else None,
        "validations": [
            {
                "field": v.field_name,
                "pdf_value": v.pdf_value,
                "xbrl_value": v.xbrl_value,
                "match": v.match,
                "source": v.source,
                "status": v.status,
            }
            for v in result.validations
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved extraction result to: {output_path}")
    return output_path


def _breakdown_to_dict(breakdown: SectionBreakdown) -> dict[str, Any]:
    """Convert SectionBreakdown to dictionary."""
    return {
        "section_id": breakdown.section_id,
        "section_title": breakdown.section_title,
        "page_number": breakdown.page_number,
        "items": [
            {
                "concepto": item.concepto,
                "ytd_actual": item.ytd_actual,
                "ytd_anterior": item.ytd_anterior,
                "quarter_actual": item.quarter_actual,
                "quarter_anterior": item.quarter_anterior,
            }
            for item in breakdown.items
        ],
        "total_ytd_actual": breakdown.total_ytd_actual,
        "total_ytd_anterior": breakdown.total_ytd_anterior,
        "total_quarter_actual": breakdown.total_quarter_actual,
        "total_quarter_anterior": breakdown.total_quarter_anterior,
    }


def print_extraction_report(result: ExtractionResult) -> None:
    """Print a formatted extraction report.

    Args:
        result: ExtractionResult to report
    """
    print(f"\n{'=' * 60}")
    print(f"Cost Extraction Report: {result.year} Q{result.quarter}")
    print(f"{'=' * 60}")
    print(f"Source: {result.source}")
    print(f"XBRL Available: {'Yes' if result.xbrl_available else 'No'}")

    nota_21 = result.sections.get("nota_21")
    if nota_21:
        print(f"\n--- Nota 21: {nota_21.section_title} ---")
        print(f"Page: {nota_21.page_number}")
        print(f"Items extracted: {len(nota_21.items)}")
        for item in nota_21.items:
            val = f"{item.ytd_actual:,}" if item.ytd_actual else "N/A"
            print(f"  {item.concepto}: {val}")
        total = f"{nota_21.total_ytd_actual:,}" if nota_21.total_ytd_actual else "N/A"
        print(f"  TOTAL: {total}")
    else:
        print("\n--- Nota 21: EXTRACTION FAILED ---")

    nota_22 = result.sections.get("nota_22")
    if nota_22:
        print(f"\n--- Nota 22: {nota_22.section_title} ---")
        print(f"Page: {nota_22.page_number}")
        print(f"Items extracted: {len(nota_22.items)}")
        for item in nota_22.items:
            val = f"{item.ytd_actual:,}" if item.ytd_actual else "N/A"
            print(f"  {item.concepto}: {val}")
        total = f"{nota_22.total_ytd_actual:,}" if nota_22.total_ytd_actual else "N/A"
        print(f"  TOTAL: {total}")
    else:
        print("\n--- Nota 22: EXTRACTION FAILED ---")

    if result.validations:
        print("\n--- Validation Results ---")
        for v in result.validations:
            print(f"  {v.field_name}:")
            if v.pdf_value is not None:
                print(f"    PDF:  {v.pdf_value:,}")
            if v.xbrl_value is not None:
                print(f"    XBRL: {v.xbrl_value:,}")
            print(f"    {v.status}")

    print(f"\n{'=' * 60}\n")


# =============================================================================
# Sheet1 Period Formatting (uses config from puco_eeff.sheets.sheet1)
# =============================================================================


def quarter_to_roman(quarter: int) -> str:
    """Convert quarter number to Roman numeral format.

    Uses config/config.json period_types for the mapping.
    Financial statements use Roman numerals for quarters.

    Args:
        quarter: Quarter number (1-4)

    Returns:
        Roman numeral string (I, II, III, IV)

    Raises:
        ValueError: If roman_numerals not found in config.
    """
    config = get_config()
    period_types = config.get("period_types", {})
    quarterly_config = period_types.get("quarterly", {})
    roman_map = quarterly_config.get("roman_numerals")

    if roman_map is None:
        raise ValueError(
            "roman_numerals not found in config/config.json period_types.quarterly. "
            "Config is the single source of truth - no hardcoded fallback."
        )

    result = roman_map.get(str(quarter))
    if result is None:
        raise ValueError(f"Quarter {quarter} not found in roman_numerals mapping. Available: {list(roman_map.keys())}")

    return result


def format_period_label(
    year: int,
    period: int,
    period_type: str = "quarterly",
) -> str:
    """Format period label as used in Sheet1 headers.

    Supports quarterly, monthly, and yearly formats from config.

    Args:
        year: Year (e.g., 2024)
        period: Period number (1-4 for quarterly, 1-12 for monthly, 1 for yearly)
        period_type: One of "quarterly", "monthly", "yearly"

    Returns:
        Formatted string like "IIQ2024", "06-2024", or "FY2024"
    """
    return format_period_display(year, period, period_type)


def format_quarter_label(year: int, quarter: int) -> str:
    """Format quarter label as used in Sheet1 headers (backward compatible).

    Args:
        year: Year (e.g., 2024)
        quarter: Quarter number (1-4)

    Returns:
        Formatted string like "IIQ2024"
    """
    return format_period_label(year, quarter, "quarterly")


def extract_sheet1_from_analisis_razonado(
    year: int,
    quarter: int,
    validate_with_xbrl: bool = True,
    return_report: bool = False,
) -> Sheet1Data | None | tuple[Sheet1Data | None, ValidationReport | None]:
    """Extract Sheet1 data from Estados Financieros PDF (Nota 21 & 22).

    This extracts the detailed cost breakdown from Nota 21 (Costo de Venta)
    and Nota 22 (Gastos de Administración y Ventas) which contain:
    - Costo de Venta breakdown (11 items) + Total
    - Gasto Admin breakdown (6 items) + Totales

    Ingresos de actividades ordinarias is extracted from XBRL when available.

    File patterns are read from config/config.json file_patterns section.

    Note: Despite the function name (kept for backward compatibility),
    the detailed cost breakdown is in Estados Financieros PDF, not Análisis Razonado.

    Args:
        year: Year of the financial statement
        quarter: Quarter (1-4)
        validate_with_xbrl: If True, validate PDF data against XBRL totals
        return_report: If True, return tuple of (data, ValidationReport)

    Returns:
        Sheet1Data object or None if extraction fails.
        If return_report=True, returns tuple of (Sheet1Data, ValidationReport).
    """
    paths = get_period_paths(year, quarter)
    raw_dir = paths["raw_pdf"]

    # Find Estados Financieros PDF using config patterns
    ef_path = find_file_with_alternatives(raw_dir, "estados_financieros_pdf", year, quarter)
    if not ef_path:
        # Fallback to default filename
        ef_path = raw_dir / format_filename("estados_financieros_pdf", year, quarter)

    if not ef_path.exists():
        logger.warning(f"Estados Financieros PDF not found: {ef_path}")
        return (None, None) if return_report else None

    # Determine source - check if we have pucobre combined file
    combined_path = find_file_with_alternatives(raw_dir, "pucobre_combined", year, quarter)
    source = "pucobre.cl" if (combined_path and combined_path.exists()) else "cmf"

    # Check XBRL availability
    xbrl_path = find_file_with_alternatives(paths["raw_xbrl"], "estados_financieros_xbrl", year, quarter)
    if not xbrl_path:
        xbrl_path = paths["raw_xbrl"] / format_filename("estados_financieros_xbrl", year, quarter)
    xbrl_available = xbrl_path.exists() if xbrl_path else False

    # Extract data from PDF using Nota 21 and Nota 22
    data = Sheet1Data(
        quarter=format_quarter_label(year, quarter),
        year=year,
        quarter_num=quarter,
        source=source,
        xbrl_available=xbrl_available,
    )

    # Extract Nota 21 (Costo de Venta) and Nota 22 (Gastos Admin)
    nota_21 = extract_pdf_section(ef_path, "nota_21")
    nota_22 = extract_pdf_section(ef_path, "nota_22")

    if nota_21 is None and nota_22 is None:
        logger.error(f"Could not extract Nota 21 or 22 from {ef_path}")
        return (None, None) if return_report else None

    # Map Nota 21 items to Sheet1Data
    if nota_21:
        data.total_costo_venta = nota_21.total_ytd_actual
        for item in nota_21.items:
            _map_nota21_item_to_sheet1(item, data)

    # Map Nota 22 items to Sheet1Data
    if nota_22:
        data.total_gasto_admin = nota_22.total_ytd_actual
        for item in nota_22.items:
            _map_nota22_item_to_sheet1(item, data)

    # Validation report (may be None if no XBRL)
    report: ValidationReport | None = None

    # Extract Ingresos: prefer XBRL, fallback to PDF
    if xbrl_available and validate_with_xbrl:
        report = _validate_sheet1_with_xbrl(data, xbrl_path)
    else:
        # No XBRL available - extract Ingresos from PDF
        logger.info("No XBRL available, extracting Ingresos from Estado de Resultados")
        ingresos_value = extract_ingresos_from_pdf(ef_path)
        if ingresos_value is not None:
            data.ingresos_ordinarios = ingresos_value
            logger.info(f"Set Ingresos from PDF: {ingresos_value:,}")
        else:
            logger.warning("Could not extract Ingresos from PDF")

        # Still run sum validations even without XBRL
        sum_results = _run_sum_validations(data)
        report = ValidationReport(sum_validations=sum_results)

    return (data, report) if return_report else data


def _map_nota_item_to_sheet1(item: LineItem, data: Sheet1Data, section_name: str) -> None:
    """Map a Nota line item to Sheet1Data fields using config-driven matching.

    Uses match_keywords and exclude_keywords from config/sheet1/extraction.json
    to determine which field a line item belongs to.

    Args:
        item: LineItem from Nota 21 or 22
        data: Sheet1Data to update
        section_name: Section name ("nota_21" or "nota_22")
    """
    field_name = match_concepto_to_field(item.concepto, section_name)

    if field_name:
        data.set_value(field_name, item.ytd_actual)
        logger.debug(f"Mapped '{item.concepto}' -> {field_name} = {item.ytd_actual}")
    else:
        logger.warning(f"Could not map item from {section_name}: '{item.concepto}'")


def _map_nota21_item_to_sheet1(item: LineItem, data: Sheet1Data) -> None:
    """Map a Nota 21 line item to Sheet1Data fields.

    .. deprecated::
        Use :func:`_map_nota_item_to_sheet1` with section_name="nota_21" directly,
        or use :func:`sections_to_sheet1data` from puco_eeff.sheets.sheet1 for
        complete config-driven conversion.

    Args:
        item: LineItem from Nota 21
        data: Sheet1Data to update
    """
    import warnings

    warnings.warn(
        "_map_nota21_item_to_sheet1() is deprecated. "
        "Use _map_nota_item_to_sheet1(item, data, 'nota_21') or sections_to_sheet1data() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    _map_nota_item_to_sheet1(item, data, "nota_21")


def _map_nota22_item_to_sheet1(item: LineItem, data: Sheet1Data) -> None:
    """Map a Nota 22 line item to Sheet1Data fields.

    .. deprecated::
        Use :func:`_map_nota_item_to_sheet1` with section_name="nota_22" directly,
        or use :func:`sections_to_sheet1data` from puco_eeff.sheets.sheet1 for
        complete config-driven conversion.

    Args:
        item: LineItem from Nota 22
        data: Sheet1Data to update
    """
    import warnings

    warnings.warn(
        "_map_nota22_item_to_sheet1() is deprecated. "
        "Use _map_nota_item_to_sheet1(item, data, 'nota_22') or sections_to_sheet1data() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    _map_nota_item_to_sheet1(item, data, "nota_22")


# =============================================================================
# Validation Functions (Phase 1, 2, 3)
# =============================================================================


def _compare_with_tolerance(a: int | None, b: int | None, tolerance: int) -> tuple[bool, int]:
    """Compare two values with tolerance, using absolute values for sign-agnostic comparison.

    Args:
        a: First value (can be None)
        b: Second value (can be None)
        tolerance: Maximum allowed difference

    Returns:
        Tuple of (match, difference). If either value is None, returns (True, 0).
    """
    if a is None or b is None:
        return True, 0
    diff = abs(abs(a) - abs(b))
    return diff <= tolerance, diff


def _run_sum_validations(data: Sheet1Data) -> list[SumValidationResult]:
    """Run config-driven sum validations on Sheet1Data.

    Uses validation_rules.total_validations from config/sheet1/xbrl_mappings.json.
    Supports per-rule tolerance override via rule's "tolerance" field.
    Non-interactive: returns structured results, logs warnings, never blocks.

    Args:
        data: Sheet1Data with extracted values

    Returns:
        List of SumValidationResult objects
    """
    results = []
    global_tolerance = get_sheet1_sum_tolerance()
    total_validations = get_sheet1_total_validations()

    for rule in total_validations:
        description = rule.get("description", "Unknown validation")
        total_field = rule.get("total_field", "")
        sum_fields = rule.get("sum_fields", [])
        # Per-rule tolerance override (falls back to global)
        rule_tolerance = rule.get("tolerance", global_tolerance)

        # Get the expected total from PDF
        expected_total = data.get_value(total_field)

        # Calculate sum of line items
        calculated_sum = 0
        for field in sum_fields:
            value = data.get_value(field)
            if value is not None:
                calculated_sum += value

        # Compare with tolerance
        match, difference = _compare_with_tolerance(expected_total, calculated_sum, rule_tolerance)
        if expected_total is None:
            difference = 0
            match = True  # Can't validate without expected total

        result = SumValidationResult(
            description=description,
            total_field=total_field,
            expected_total=expected_total,
            calculated_sum=calculated_sum,
            match=match,
            difference=difference,
            tolerance=rule_tolerance,
        )
        results.append(result)

        # Log result
        if expected_total is None:
            logger.info(f"⚠ {description}: no total value to compare (sum={calculated_sum:,})")
        elif match:
            logger.info(f"✓ {description}: sum={calculated_sum:,} matches total={expected_total:,}")
        else:
            logger.warning(f"✗ {description}: sum={calculated_sum:,} != total={expected_total:,} (diff: {difference})")

    return results


def _run_cross_validations(
    data: Sheet1Data,
    xbrl_totals: dict[str, int | None] | None,
) -> list[CrossValidationResult]:
    """Run config-driven cross-validations.

    Source priority: PDF fields from Sheet1Data first, then XBRL if available.
    Uses per-rule tolerance from config, falls back to global sum_tolerance.
    Safe evaluation: no eval(), explicit formula handling.

    Args:
        data: Sheet1Data with extracted values
        xbrl_totals: XBRL totals dict (or None if unavailable)

    Returns:
        List of CrossValidationResult objects
    """
    results = []
    global_tolerance = get_sheet1_sum_tolerance()
    cross_validations = get_sheet1_cross_validations()

    for rule in cross_validations:
        description = rule.get("description", "Unknown validation")
        formula = rule.get("formula", "")
        rule_tolerance = rule.get("tolerance", global_tolerance)

        # Resolve values needed for this formula
        values, missing = _resolve_cross_validation_values(data, xbrl_totals, formula)

        if missing:
            result = CrossValidationResult(
                description=description,
                formula=formula,
                expected_value=None,
                calculated_value=None,
                match=True,  # Can't fail if we can't evaluate
                difference=None,
                tolerance=rule_tolerance,
                missing_facts=missing,
            )
            results.append(result)
            logger.info(f"⚠ {description}: skipped - missing {', '.join(missing)}")
            continue

        # Evaluate formula safely
        expected, calculated, match, diff = _evaluate_cross_validation(formula, values, rule_tolerance)

        result = CrossValidationResult(
            description=description,
            formula=formula,
            expected_value=expected,
            calculated_value=calculated,
            match=match,
            difference=diff,
            tolerance=rule_tolerance,
        )
        results.append(result)

        # Log result
        if match:
            logger.info(f"✓ {description}: {expected:,}")
        else:
            logger.warning(f"✗ {description}: expected={expected}, calculated={calculated} (diff: {diff})")

    return results


def _resolve_cross_validation_values(
    data: Sheet1Data,
    xbrl_totals: dict[str, int | None] | None,
    formula: str,
) -> tuple[dict[str, int], list[str]]:
    """Resolve values needed for a cross-validation formula.

    Priority: Sheet1Data fields first, then XBRL totals.

    Args:
        data: Sheet1Data with extracted values
        xbrl_totals: XBRL totals dict (or None)
        formula: Formula string to parse for variable names

    Returns:
        Tuple of (values dict, list of missing field names)
    """
    # Get mapping from config and invert it (result_key -> field_name)
    field_to_result = get_sheet1_result_key_mapping()
    xbrl_key_map = {v: k for k, v in field_to_result.items()}

    # Extract variable names from formula (simple pattern matching)
    # Matches: gross_profit, ingresos_ordinarios, total_costo_venta, etc.
    var_pattern = re.compile(r"\b([a-z_]+)\b")
    var_names = set(var_pattern.findall(formula))

    # Remove keywords that aren't variables
    keywords = {"abs", "and", "or", "not", "if", "else", "true", "false"}
    var_names = var_names - keywords

    values = {}
    missing = []

    for var in var_names:
        # Try Sheet1Data first
        value = data.get_value(var)

        if value is None and xbrl_totals:
            # Try XBRL totals
            # Check if var is a known XBRL key or map it
            for xbrl_key, mapped_name in xbrl_key_map.items():
                if var == mapped_name or var == xbrl_key:
                    value = xbrl_totals.get(xbrl_key)
                    break

            # Also try direct lookup
            if value is None:
                value = xbrl_totals.get(var)

        if value is not None:
            values[var] = value
        else:
            missing.append(var)

    return values, missing


def _evaluate_cross_validation(
    formula: str,
    values: dict[str, int],
    tolerance: int,
) -> tuple[int | None, int | None, bool, int | None]:
    """Safely evaluate a cross-validation formula.

    Supports formulas like:
    - "gross_profit == ingresos_ordinarios - abs(total_costo_venta)"

    Args:
        formula: Formula string
        values: Dictionary of variable values
        tolerance: Tolerance for comparison

    Returns:
        Tuple of (expected_value, calculated_value, match, difference)
    """
    # Parse the formula - expect format: "lhs == rhs"
    if "==" not in formula:
        logger.warning(f"Unsupported formula format (no '=='): {formula}")
        return None, None, True, None

    parts = formula.split("==")
    if len(parts) != 2:
        logger.warning(f"Unsupported formula format (multiple '=='): {formula}")
        return None, None, True, None

    lhs_expr = parts[0].strip()
    rhs_expr = parts[1].strip()

    try:
        expected = _safe_eval_expression(lhs_expr, values)
        calculated = _safe_eval_expression(rhs_expr, values)

        if expected is None or calculated is None:
            return expected, calculated, True, None

        diff = abs(expected - calculated)
        match = diff <= tolerance

        return expected, calculated, match, diff

    except Exception as e:
        logger.warning(f"Error evaluating formula '{formula}': {e}")
        return None, None, True, None


def _safe_eval_expression(expr: str, values: dict[str, int]) -> int | None:
    """Safely evaluate a simple arithmetic expression using AST parsing.

    Supports: variable names, integers, +, -, *, abs(), parentheses.
    Uses ast.parse() with a whitelist of allowed node types for security.

    Args:
        expr: Expression string (e.g., "a + b", "abs(x)", "a - abs(b)")
        values: Dictionary of variable values

    Returns:
        Evaluated integer value, or None if evaluation fails or expression is unsupported.
    """
    expr = expr.strip()
    if not expr:
        return None

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        logger.debug(f"Syntax error parsing expression '{expr}': {e}")
        return None

    def _eval_node(node: ast.AST) -> int | None:
        """Recursively evaluate an AST node."""
        # Integer literal
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value

        # Negative number (UnaryOp with USub)
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                operand = _eval_node(node.operand)
                return -operand if operand is not None else None
            elif isinstance(node.op, ast.UAdd):
                return _eval_node(node.operand)
            else:
                logger.debug(f"Unsupported unary operator: {type(node.op).__name__}")
                return None

        # Variable name
        if isinstance(node, ast.Name):
            var_name = node.id
            if var_name in values:
                return values[var_name]
            else:
                logger.debug(f"Unknown variable: {var_name}")
                return None

        # Binary operation (+, -, *)
        if isinstance(node, ast.BinOp):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.Mult):
                return left * right
            else:
                logger.debug(f"Unsupported binary operator: {type(node.op).__name__}")
                return None

        # Function call (only abs() allowed)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "abs":
                if len(node.args) == 1 and not node.keywords:
                    arg_val = _eval_node(node.args[0])
                    return abs(arg_val) if arg_val is not None else None
            logger.debug(f"Unsupported function call: {ast.dump(node)}")
            return None

        logger.debug(f"Unsupported AST node type: {type(node).__name__}")
        return None

    try:
        return _eval_node(tree.body)
    except Exception as e:
        logger.debug(f"Error evaluating expression '{expr}': {e}")
        return None


def _validate_sheet1_with_xbrl(data: Sheet1Data, xbrl_path: Path) -> ValidationReport:
    """Validate and supplement Sheet1 data with XBRL totals.

    Internal wrapper that extracts XBRL totals then calls the unified
    validation function.

    Args:
        data: Sheet1Data populated from PDF
        xbrl_path: Path to XBRL file

    Returns:
        ValidationReport with all validation results
    """
    xbrl_totals = extract_xbrl_totals(xbrl_path)
    return run_sheet1_validations(data, xbrl_totals)


# save_sheet1_data and print_sheet1_report are re-exported from puco_eeff.sheets.sheet1
# for backward compatibility. They are imported at the top of this file.


def extract_sheet1_from_xbrl(year: int, quarter: int) -> Sheet1Data | None:
    """Extract Sheet1 totals directly from XBRL file.

    This extracts only the totals available in XBRL using fact_mappings
    from config/sheet1/xbrl_mappings.json:
    - Ingresos (Revenue)
    - Total Costo de Venta (CostOfSales)
    - Total Gasto Admin (AdministrativeExpense)

    Note: XBRL does not contain the detailed line items (cv_*, ga_*),
    only totals. Use PDF extraction for full breakdown.

    Args:
        year: Year of the financial statement
        quarter: Quarter (1-4)

    Returns:
        Sheet1Data with totals only, or None if extraction fails
    """
    paths = get_period_paths(year, quarter)
    raw_xbrl = paths["raw_xbrl"]

    # Find XBRL file using config patterns
    xbrl_path = find_file_with_alternatives(raw_xbrl, "estados_financieros_xbrl", year, quarter)

    if not xbrl_path or not xbrl_path.exists():
        # Fallback to direct filename
        xbrl_path = raw_xbrl / format_filename("estados_financieros_xbrl", year, quarter)
        # Also try with .xml extension if .xbrl not found
        if not xbrl_path.exists():
            alt_name = format_filename("estados_financieros_xbrl", year, quarter)
            xbrl_path = raw_xbrl / alt_name.replace(".xbrl", ".xml")

    if not xbrl_path.exists():
        logger.warning(f"XBRL file not found for {year} Q{quarter}")
        return None

    try:
        xbrl_data = parse_xbrl_file(xbrl_path)
    except Exception as e:
        logger.error(f"Failed to parse XBRL: {e}")
        return None

    data = Sheet1Data(
        quarter=format_quarter_label(year, quarter),
        year=year,
        quarter_num=quarter,
        source="cmf",
        xbrl_available=True,
    )

    scaling_factor = get_xbrl_scaling_factor()

    # Fields to extract from XBRL
    xbrl_fields = ["ingresos_ordinarios", "total_costo_venta", "total_gasto_admin"]

    for field_name in xbrl_fields:
        fact_mapping = get_sheet1_xbrl_fact_mapping(field_name)
        if not fact_mapping:
            continue

        # Try primary fact name first
        primary_fact = fact_mapping.get("primary")
        fallback_facts = fact_mapping.get("fallbacks", [])

        facts = []
        if primary_fact:
            facts = get_facts_by_name(xbrl_data, primary_fact)

        # Try fallbacks if primary not found
        if not facts:
            for fallback in fallback_facts:
                facts = get_facts_by_name(xbrl_data, fallback)
                if facts:
                    logger.debug(f"Using fallback XBRL fact '{fallback}' for {field_name}")
                    break

        # Extract value from first matching fact
        for fact in facts:
            if fact.get("value"):
                try:
                    raw_value = int(float(fact["value"]))
                    # Apply scaling factor for fields that need it
                    if fact_mapping.get("apply_scaling", False):
                        value = raw_value // scaling_factor
                        logger.info(f"XBRL {field_name}: {value:,} (scaled from {raw_value:,})")
                    else:
                        value = raw_value
                        logger.info(f"XBRL {field_name}: {value:,}")
                    data.set_value(field_name, value)
                    break
                except (ValueError, TypeError):
                    continue

    # Check if we got at least one value
    if all(data.get_value(f) is None for f in ["ingresos_ordinarios", "total_costo_venta", "total_gasto_admin"]):
        logger.warning(f"No Sheet1 data found in XBRL for {year} Q{quarter}")
        return None

    return data


def extract_sheet1(
    year: int,
    quarter: int,
    prefer_pdf: bool = True,
    validate: bool = True,
    return_report: bool = False,
) -> Sheet1Data | None | tuple[Sheet1Data | None, ValidationReport | None]:
    """Extract Sheet1 data from available sources (PDF and/or XBRL).

    This is the main entry point for Sheet1 extraction. It:
    1. First tries PDF extraction from Análisis Razonado (full breakdown)
    2. Validates/supplements with XBRL if available
    3. Falls back to XBRL-only if PDF extraction fails

    Args:
        year: Year of the financial statement
        quarter: Quarter (1-4)
        prefer_pdf: If True, prefer PDF for detailed breakdown
        validate: If True, validate PDF data against XBRL
        return_report: If True, return tuple of (data, ValidationReport)

    Returns:
        Sheet1Data object or None if extraction fails from all sources.
        If return_report=True, returns tuple of (Sheet1Data, ValidationReport).
    """
    data = None
    report: ValidationReport | None = None

    if prefer_pdf:
        # Try PDF first (has detailed breakdown)
        result = extract_sheet1_from_analisis_razonado(year, quarter, validate_with_xbrl=validate, return_report=True)
        data, report = result if isinstance(result, tuple) else (result, None)

        if data is None:
            # Fall back to XBRL (totals only)
            logger.info(f"PDF extraction failed, trying XBRL for {year} Q{quarter}")
            data = extract_sheet1_from_xbrl(year, quarter)
            report = None  # No validation report from XBRL-only extraction
    else:
        # Try XBRL first
        data = extract_sheet1_from_xbrl(year, quarter)

        if data is None or validate:
            # Try PDF for detailed breakdown or validation
            result = extract_sheet1_from_analisis_razonado(year, quarter, validate_with_xbrl=False, return_report=True)
            pdf_data, pdf_report = result if isinstance(result, tuple) else (result, None)

            if pdf_data is not None:
                if data is None:
                    data = pdf_data
                    report = pdf_report
                else:
                    # Merge: use PDF detailed items, XBRL totals
                    _merge_pdf_into_xbrl_data(pdf_data, data)
                    report = pdf_report

    if data is None:
        logger.error(f"Failed to extract Sheet1 from any source for {year} Q{quarter}")

    return (data, report) if return_report else data


def _merge_pdf_into_xbrl_data(pdf_data: Sheet1Data, xbrl_data: Sheet1Data) -> None:
    """Merge PDF detailed items into XBRL data.

    PDF has detailed line items (cv_*, ga_*), XBRL has validated totals.
    This combines the best of both sources.

    Detail fields are determined from config/sheet1/fields.json:
    - Fields from nota_21 and nota_22 sections
    - Excludes total fields (is_total: true)

    Args:
        pdf_data: Sheet1Data from PDF (detailed breakdown)
        xbrl_data: Sheet1Data from XBRL (totals) - modified in place
    """
    # Get detail fields from config (non-total fields from nota_21 and nota_22)
    detail_fields = get_sheet1_detail_fields(sections=["nota_21", "nota_22"])

    for field_name in detail_fields:
        pdf_value = getattr(pdf_data, field_name, None)
        if pdf_value is not None:
            setattr(xbrl_data, field_name, pdf_value)


# print_sheet1_report is re-exported from puco_eeff.sheets.sheet1 (imported at top of file)

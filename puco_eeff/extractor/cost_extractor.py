"""Extract detailed cost breakdowns from PDF and validate against XBRL.

This module provides generic PDF/XBRL extraction primitives with config-driven
rules. Sheet1-specific config is in config/sheet1/.

Key Classes:
    SectionBreakdown: Generic container for extracted PDF section data.
        Uses section_id/section_title.
    ExtractionResult: Complete extraction result with sections dict.
    LineItem: Single line item with concepto and period values.

Key Functions:
    extract_pdf_section(): Generic config-driven PDF section extraction.
    find_text_page(): Generic PDF page finder - searches for required text strings.
    find_section_page(): Config-driven wrapper using find_text_page.
    extract_sheet1(): Main entry point for Sheet1 extraction.

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
- config/sheet1/extraction.json: Sheet1 sections (nota_21, nota_22, ingresos)
- config/sheet1/xbrl_mappings.json: Sheet1 XBRL fact mappings
- config/sheet1/reference_data.json: Known-good values for validation
"""

from __future__ import annotations

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
    get_sheet1_extraction_sections,
    get_sheet1_section_expected_items,
    get_sheet1_section_field_mappings,
    get_sheet1_section_spec,
    get_sheet1_sum_tolerance,
    get_sheet1_xbrl_fact_mapping,
    match_concepto_to_field,
    # Re-exported for backward compatibility - canonical location is puco_eeff.sheets.sheet1
    print_sheet1_report,
    save_sheet1_data,
)

logger = setup_logging(__name__)

# Public API exports
__all__ = [
    # Dataclasses
    "LineItem",
    "SectionBreakdown",
    "ValidationResult",
    "ExtractionResult",
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
    "validate_extraction",
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
        # Try fallback: some sections share pages
        fallback_sections = {
            "nota_22": "nota_21",  # Nota 22 often on same page as Nota 21
        }
        if section_name in fallback_sections:
            fallback = fallback_sections[section_name]
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
    match_keywords = ingresos_mapping.get("match_keywords", ["ingresos", "actividades ordinarias"])

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
                        if value is not None and value > 1000:
                            logger.info(f"Extracted Ingresos from PDF page {page_idx + 1}: {value:,}")
                            return value

                # Fallback: Ingresos is typically the first positive large value
                for val_str in values:
                    value = parse_chilean_number(val_str.strip())
                    if value is not None and value > 1000:
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

    result: dict[str, int | None] = {
        "cost_of_sales": None,
        "admin_expense": None,
        "ingresos": None,
    }

    scaling_factor = get_xbrl_scaling_factor()

    # Field name -> result key mapping
    field_to_result = {
        "total_costo_venta": "cost_of_sales",
        "total_gasto_admin": "admin_expense",
        "ingresos_ordinarios": "ingresos",
    }

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


def validate_extraction(
    pdf_nota_21: SectionBreakdown | None,
    pdf_nota_22: SectionBreakdown | None,
    xbrl_totals: dict[str, int | None] | None,
) -> list[ValidationResult]:
    """Cross-validate PDF extraction against XBRL totals.

    Args:
        pdf_nota_21: Extracted Nota 21 from PDF
        pdf_nota_22: Extracted Nota 22 from PDF
        xbrl_totals: Totals extracted from XBRL (or None if no XBRL)

    Returns:
        List of validation results
    """
    validations = []

    # Validate Cost of Sales (Nota 21)
    pdf_cost = pdf_nota_21.total_ytd_actual if pdf_nota_21 else None
    xbrl_cost = xbrl_totals.get("cost_of_sales") if xbrl_totals else None

    if pdf_cost is not None and xbrl_cost is not None:
        # Both sources available - compare
        # Use absolute values since signs may differ
        match = abs(pdf_cost) == abs(xbrl_cost)
        validations.append(
            ValidationResult(
                field_name="Costo de Venta (Nota 21)",
                pdf_value=pdf_cost,
                xbrl_value=xbrl_cost,
                match=match,
                source="both",
                difference=abs(pdf_cost) - abs(xbrl_cost) if not match else None,
            )
        )
    elif pdf_cost is not None:
        # PDF only
        validations.append(
            ValidationResult(
                field_name="Costo de Venta (Nota 21)",
                pdf_value=pdf_cost,
                xbrl_value=None,
                match=True,  # Can't validate, but extraction succeeded
                source="pdf_only",
            )
        )
    elif xbrl_cost is not None:
        # XBRL only
        validations.append(
            ValidationResult(
                field_name="Costo de Venta (Nota 21)",
                pdf_value=None,
                xbrl_value=xbrl_cost,
                match=False,
                source="xbrl_only",
            )
        )

    # Validate Administrative Expense (Nota 22)
    pdf_admin = pdf_nota_22.total_ytd_actual if pdf_nota_22 else None
    xbrl_admin = xbrl_totals.get("admin_expense") if xbrl_totals else None

    if pdf_admin is not None and xbrl_admin is not None:
        match = abs(pdf_admin) == abs(xbrl_admin)
        validations.append(
            ValidationResult(
                field_name="Gastos Administración (Nota 22)",
                pdf_value=pdf_admin,
                xbrl_value=xbrl_admin,
                match=match,
                source="both",
                difference=abs(pdf_admin) - abs(xbrl_admin) if not match else None,
            )
        )
    elif pdf_admin is not None:
        validations.append(
            ValidationResult(
                field_name="Gastos Administración (Nota 22)",
                pdf_value=pdf_admin,
                xbrl_value=None,
                match=True,
                source="pdf_only",
            )
        )
    elif xbrl_admin is not None:
        validations.append(
            ValidationResult(
                field_name="Gastos Administración (Nota 22)",
                pdf_value=None,
                xbrl_value=xbrl_admin,
                match=False,
                source="xbrl_only",
            )
        )

    return validations


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
            result.xbrl_totals["cost_of_sales"] = xbrl_totals.get("cost_of_sales")
            result.xbrl_totals["admin_expense"] = xbrl_totals.get("admin_expense")

            result.validations = validate_extraction(
                result.sections.get("nota_21"),
                result.sections.get("nota_22"),
                xbrl_totals,
            )
    else:
        logger.info(f"No XBRL available for {year} Q{quarter} - using PDF only")
        result.xbrl_available = False

        if validate:
            # Still perform PDF-only validation
            result.validations = validate_extraction(
                result.sections.get("nota_21"),
                result.sections.get("nota_22"),
                None,
            )

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
) -> Sheet1Data | None:
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

    Returns:
        Sheet1Data object or None if extraction fails
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
        return None

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
        return None

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

    # Extract Ingresos: prefer XBRL, fallback to PDF
    if xbrl_available and validate_with_xbrl:
        _validate_sheet1_with_xbrl(data, xbrl_path)
    else:
        # No XBRL available - extract Ingresos from PDF
        logger.info("No XBRL available, extracting Ingresos from Estado de Resultados")
        ingresos_value = extract_ingresos_from_pdf(ef_path)
        if ingresos_value is not None:
            data.ingresos_ordinarios = ingresos_value
            logger.info(f"Set Ingresos from PDF: {ingresos_value:,}")
        else:
            logger.warning("Could not extract Ingresos from PDF")

    return data


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

    Args:
        item: LineItem from Nota 21
        data: Sheet1Data to update
    """
    _map_nota_item_to_sheet1(item, data, "nota_21")


def _map_nota22_item_to_sheet1(item: LineItem, data: Sheet1Data) -> None:
    """Map a Nota 22 line item to Sheet1Data fields.

    Args:
        item: LineItem from Nota 22
        data: Sheet1Data to update
    """
    _map_nota_item_to_sheet1(item, data, "nota_22")


def _validate_sheet1_with_xbrl(data: Sheet1Data, xbrl_path: Path) -> None:
    """Validate and supplement Sheet1 data with XBRL totals.

    This cross-validates PDF extraction against XBRL to:
    1. Log validation results (match/mismatch)
    2. Use XBRL values if PDF extraction failed for totals

    Uses sum_tolerance from config/sheet1/xbrl_mappings.json for tolerance settings.

    Args:
        data: Sheet1Data populated from PDF
        xbrl_path: Path to XBRL file
    """
    xbrl_totals = extract_xbrl_totals(xbrl_path)
    tolerance = get_sheet1_sum_tolerance()

    # Define validations: (field_name, xbrl_key, display_name)
    validations = [
        ("total_costo_venta", "cost_of_sales", "Total Costo de Venta"),
        ("total_gasto_admin", "admin_expense", "Total Gasto Admin"),
        ("ingresos_ordinarios", "ingresos", "Ingresos Ordinarios"),
    ]

    for field_name, xbrl_key, display_name in validations:
        xbrl_value = xbrl_totals.get(xbrl_key)
        pdf_value = data.get_value(field_name)

        if xbrl_value is not None:
            if pdf_value is not None:
                # Both available - compare (using absolute values due to sign differences)
                diff = abs(abs(pdf_value) - abs(xbrl_value))
                if diff <= tolerance:
                    logger.info(f"✓ {display_name} matches XBRL: {pdf_value:,}")
                else:
                    logger.warning(
                        f"✗ {display_name} mismatch - PDF: {pdf_value:,}, XBRL: {xbrl_value:,} (diff: {diff})"
                    )
            else:
                # PDF extraction failed - use XBRL value
                logger.info(f"Using XBRL value for {display_name}: {xbrl_value:,}")
                data.set_value(field_name, xbrl_value)


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
) -> Sheet1Data | None:
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

    Returns:
        Sheet1Data object or None if extraction fails from all sources
    """
    data = None

    if prefer_pdf:
        # Try PDF first (has detailed breakdown)
        data = extract_sheet1_from_analisis_razonado(year, quarter, validate_with_xbrl=validate)

        if data is None:
            # Fall back to XBRL (totals only)
            logger.info(f"PDF extraction failed, trying XBRL for {year} Q{quarter}")
            data = extract_sheet1_from_xbrl(year, quarter)
    else:
        # Try XBRL first
        data = extract_sheet1_from_xbrl(year, quarter)

        if data is None or validate:
            # Try PDF for detailed breakdown or validation
            pdf_data = extract_sheet1_from_analisis_razonado(year, quarter, validate_with_xbrl=False)
            if pdf_data is not None:
                if data is None:
                    data = pdf_data
                else:
                    # Merge: use PDF detailed items, XBRL totals
                    _merge_pdf_into_xbrl_data(pdf_data, data)

    if data is None:
        logger.error(f"Failed to extract Sheet1 from any source for {year} Q{quarter}")

    return data


def _merge_pdf_into_xbrl_data(pdf_data: Sheet1Data, xbrl_data: Sheet1Data) -> None:
    """Merge PDF detailed items into XBRL data.

    PDF has detailed line items (cv_*, ga_*), XBRL has validated totals.
    This combines the best of both sources.

    Args:
        pdf_data: Sheet1Data from PDF (detailed breakdown)
        xbrl_data: Sheet1Data from XBRL (totals) - modified in place
    """
    # Copy detailed line items from PDF to XBRL data
    detail_fields = [
        "cv_gastos_personal",
        "cv_materiales",
        "cv_energia",
        "cv_servicios_terceros",
        "cv_depreciacion_amort",
        "cv_deprec_leasing",
        "cv_deprec_arrend",
        "cv_serv_mineros",
        "cv_fletes",
        "cv_gastos_diferidos",
        "cv_convenios",
        "ga_gastos_personal",
        "ga_materiales",
        "ga_servicios_terceros",
        "ga_gratificacion",
        "ga_comercializacion",
        "ga_otros",
    ]

    for field_name in detail_fields:
        pdf_value = getattr(pdf_data, field_name, None)
        if pdf_value is not None:
            setattr(xbrl_data, field_name, pdf_value)


# print_sheet1_report is re-exported from puco_eeff.sheets.sheet1 (imported at top of file)

"""Extract detailed cost breakdowns from PDF and validate against XBRL.

This module extracts cost data from Estados Financieros PDF using
configurable extraction specs from config/extraction_specs.json.

The module is designed to be general and reusable:
- All field mappings come from config files
- Sheet1Data structure is generated from config at runtime
- File patterns are configurable
- XBRL validation is optional and configured separately

Configuration Files:
- config/config.json: File patterns, period types, sources
- config/extraction_specs.json: PDF extraction rules, field mappings
- config/xbrl_specs.json: XBRL fact names, validation rules, scaling
- config/reference_data.json: Known-good values for validation
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
    get_period_specs,
    get_sheet1_row_mapping,
    get_sheet1_value_fields,
    get_sum_tolerance,
    get_xbrl_fact_mapping,
    get_xbrl_scaling_factor,
    match_concepto_to_field,
    setup_logging,
)
from puco_eeff.extractor.xbrl_parser import get_facts_by_name, parse_xbrl_file

logger = setup_logging(__name__)


# =============================================================================
# Extraction Specs Loading
# =============================================================================


def _get_section_spec(section_name: str, year: int | None = None, quarter: int | None = None) -> dict[str, Any]:
    """Get extraction specification for a section from extraction_specs.json.

    Merges default specs with period-specific deviations.

    Args:
        section_name: Section name ("nota_21", "nota_22", or "ingresos")
        year: Optional year for period-specific specs
        quarter: Optional quarter for period-specific specs

    Returns:
        Section specification dictionary
    """
    specs = get_period_specs(year, quarter) if year and quarter else get_period_specs(2024, 2)
    sections = specs.get("sections", {})
    return sections.get(section_name, {})


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


def _get_extraction_labels(config: dict | None = None) -> tuple[list[str], list[str], dict[str, str]]:
    """Get extraction labels from extraction_specs.json.

    This function loads labels from the new extraction_specs.json,
    falling back to config.json for backward compatibility.

    Args:
        config: Configuration dict (ignored, kept for backward compat)

    Returns:
        Tuple of (costo_venta_items, gasto_admin_items, field_labels)
    """
    try:
        # Load from extraction_specs.json
        nota_21_spec = _get_section_spec("nota_21")
        nota_22_spec = _get_section_spec("nota_22")

        # Extract unique PDF labels for each section
        costo_venta_items = []
        for field_name, field_spec in nota_21_spec.get("field_mappings", {}).items():
            if not field_name.startswith("total_"):
                labels = field_spec.get("pdf_labels", [])
                if labels:
                    costo_venta_items.append(labels[0])  # Use primary label

        gasto_admin_items = []
        for field_name, field_spec in nota_22_spec.get("field_mappings", {}).items():
            if not field_name.startswith("total_"):
                labels = field_spec.get("pdf_labels", [])
                if labels:
                    gasto_admin_items.append(labels[0])

        # Build field_labels mapping
        field_labels = {}
        ingresos_spec = _get_section_spec("ingresos")
        for field_name, field_spec in ingresos_spec.get("field_mappings", {}).items():
            labels = field_spec.get("pdf_labels", [])
            if labels:
                field_labels[field_name] = labels[0]

        for field_name, field_spec in nota_21_spec.get("field_mappings", {}).items():
            labels = field_spec.get("pdf_labels", [])
            if labels:
                field_labels[field_name] = labels[0]

        for field_name, field_spec in nota_22_spec.get("field_mappings", {}).items():
            labels = field_spec.get("pdf_labels", [])
            if labels:
                field_labels[field_name] = labels[0]

        return costo_venta_items, gasto_admin_items, field_labels

    except Exception as e:
        logger.warning(f"Failed to load from extraction_specs.json, using fallback: {e}")
        # Fallback to hardcoded defaults
        return _get_extraction_labels_fallback()


def load_sheet1_config(config: dict | None = None) -> dict[str, Any]:
    """Load sheet1 configuration from config.json.

    Args:
        config: Configuration dict, or None to load from file

    Returns:
        Sheet1 configuration dictionary
    """
    if config is None:
        config = get_config()
    return config.get("sheets", {}).get("sheet1", {})


# =============================================================================
# Emergency Fallback Labels (used only when config files are unavailable)
# =============================================================================


def _get_extraction_labels_fallback() -> tuple[list[str], list[str], dict[str, str]]:
    """Emergency fallback extraction labels when config files are unavailable.

    This function should rarely be called. If it is called frequently,
    it indicates a problem with config file loading.
    """
    logger.warning("FALLBACK: Using emergency hardcoded labels - config files may be unavailable")

    costo_venta_items = [
        "Gastos en personal",
        "Materiales y repuestos",
        "Energía eléctrica",
        "Servicios de terceros",
        "Depreciación y amort del periodo",
        "Depreciación Activos en leasing",
        "Depreciación Arrendamientos",
        "Servicios mineros de terceros",
        "Fletes y otros gastos operacionales",
        "Gastos Diferidos, ajustes existencias y otros",
        "Obligaciones por convenios colectivos",
    ]

    gasto_admin_items = [
        "Gastos en personal",
        "Materiales y repuestos",
        "Servicios de terceros",
        "Provision gratificacion legal y otros",
        "Gastos comercializacion",
        "Otros gastos",
    ]

    field_labels = {
        "ingresos_ordinarios": "Ingresos de actividades ordinarias",
        "cv_gastos_personal": "Gastos en personal",
        "cv_materiales": "Materiales y repuestos",
        "cv_energia": "Energía eléctrica",
        "cv_servicios_terceros": "Servicios de terceros",
        "cv_depreciacion_amort": "Depreciación y amort del periodo",
        "cv_deprec_leasing": "Depreciación Activos en leasing",
        "cv_deprec_arrend": "Depreciación Arrendamientos",
        "cv_serv_mineros": "Servicios mineros de terceros",
        "cv_fletes": "Fletes y otros gastos operacionales",
        "cv_gastos_diferidos": "Gastos Diferidos, ajustes existencias y otros",
        "cv_convenios": "Obligaciones por convenios colectivos",
        "total_costo_venta": "Total Costo de Venta",
        "ga_gastos_personal": "Gastos en personal",
        "ga_materiales": "Materiales y repuestos",
        "ga_servicios_terceros": "Servicios de terceros",
        "ga_gratificacion": "Provision gratificacion legal y otros",
        "ga_comercializacion": "Gastos comercializacion",
        "ga_otros": "Otros gastos",
        "total_gasto_admin": "Totales",
    }

    return costo_venta_items, gasto_admin_items, field_labels


@dataclass
class LineItem:
    """A single line item from the cost breakdown."""

    concepto: str
    ytd_actual: int | None = None
    ytd_anterior: int | None = None
    quarter_actual: int | None = None
    quarter_anterior: int | None = None


@dataclass
class CostBreakdown:
    """Cost breakdown from a Nota section."""

    nota_number: int
    nota_title: str
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
    """Complete extraction result with optional validation."""

    year: int
    quarter: int
    nota_21: CostBreakdown | None = None
    nota_22: CostBreakdown | None = None
    xbrl_available: bool = False
    xbrl_cost_of_sales: int | None = None
    xbrl_admin_expense: int | None = None
    validations: list[ValidationResult] = field(default_factory=list)
    source: str = "cmf"  # "cmf" or "pucobre.cl"
    pdf_path: Path | None = None
    xbrl_path: Path | None = None

    def is_valid(self) -> bool:
        """Check if all validations passed."""
        if not self.validations:
            # No validations performed (PDF-only extraction)
            return self.nota_21 is not None and self.nota_22 is not None
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


def find_nota_page(
    pdf_path: Path,
    nota_number: int,
    year: int | None = None,
    quarter: int | None = None,
) -> int | None:
    """Find the page number where a Nota section with detailed breakdown exists.

    Uses extraction_specs.json to get search patterns and unique identifiers.

    Args:
        pdf_path: Path to the PDF file
        nota_number: Nota number to find (21 or 22)
        year: Optional year for period-specific specs
        quarter: Optional quarter for period-specific specs

    Returns:
        0-indexed page number or None if not found
    """
    # Get section spec for table identifiers
    section_name = f"nota_{nota_number}"
    section_spec = _get_section_spec(section_name, year, quarter)

    # Get unique items for this section (used to identify correct page)
    unique_items, _ = _get_table_identifiers(section_spec)

    # Build detail items list from unique identifiers
    # Add lowercase variations and accent-stripped versions
    detail_items = []
    for item in unique_items:
        detail_items.append(item.lower())
        # Also add without accents
        normalized = _normalize_for_matching(item)
        if normalized != item.lower():
            detail_items.append(normalized)

    # Fallback to default patterns if no unique items in spec
    if not detail_items:
        if nota_number == 21:
            detail_items = [
                "gastos en personal",
                "materiales",
                "energía",
                "energia",
                "servicios de terceros",
                "depreciación",
                "depreciacion",
            ]
        else:
            detail_items = [
                "gastos en personal",
                "materiales",
                "servicios de terceros",
                "gratificación",
                "gratificacion",
                "comercialización",
                "comercializacion",
            ]

    # Get search patterns from spec
    search_patterns = section_spec.get("search_patterns", [])
    if not search_patterns:
        # Fallback patterns
        if nota_number == 21:
            search_patterns = [f"{nota_number}. costo", f"{nota_number} costo"]
        else:
            search_patterns = [f"{nota_number}. gastos", f"{nota_number} gastos"]

    # Get validation requirements from spec
    validation = section_spec.get("validation", {})
    min_detail_items = validation.get("min_detail_items", 3)
    has_totales = validation.get("has_totales_row", True)

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").lower()

            # Look for section header pattern
            has_header = any(pattern.lower() in text for pattern in search_patterns)
            if not has_header:
                continue

            # Check if this page has detailed breakdown
            detail_count = sum(1 for item in detail_items if item in text)

            # Check for totales if required
            totales_present = not has_totales or "totales" in text

            if detail_count >= min_detail_items and totales_present:
                logger.info(f"Found Nota {nota_number} with details on page {page_idx + 1}")
                return page_idx

    return None


def extract_table_from_page(
    pdf_path: Path,
    page_index: int,
    expected_items: list[str],
    nota_number: int = 0,
    year: int | None = None,
    quarter: int | None = None,
) -> list[dict[str, Any]]:
    """Extract cost table data from a specific page.

    Uses extraction_specs.json to get table identifiers for scoring.

    Args:
        pdf_path: Path to the PDF file
        page_index: 0-indexed page number
        expected_items: List of expected line item names
        nota_number: Nota number (21 or 22) to help select correct table
        year: Optional year for period-specific specs
        quarter: Optional quarter for period-specific specs

    Returns:
        List of dictionaries with concepto and value columns
    """
    # Get table identifiers from spec
    section_name = f"nota_{nota_number}" if nota_number else ""
    if section_name:
        section_spec = _get_section_spec(section_name, year, quarter)
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


def extract_nota_21(pdf_path: Path, config: dict | None = None) -> CostBreakdown | None:
    """Extract Nota 21 - Costo de Venta from PDF.

    Args:
        pdf_path: Path to Estados Financieros PDF
        config: Configuration dict, or None to load from file

    Returns:
        CostBreakdown object or None if extraction fails
    """
    costo_venta_items, _, _ = _get_extraction_labels(config)

    page_idx = find_nota_page(pdf_path, 21)
    if page_idx is None:
        logger.error("Could not find Nota 21 in PDF")
        return None

    rows = extract_table_from_page(pdf_path, page_idx, costo_venta_items, nota_number=21)

    if not rows:
        # Try next page (table might span pages)
        rows = extract_table_from_page(pdf_path, page_idx + 1, costo_venta_items, nota_number=21)

    if not rows:
        logger.error("Could not extract Nota 21 table")
        return None

    breakdown = CostBreakdown(
        nota_number=21,
        nota_title="Costo de Venta",
        page_number=page_idx + 1,
    )

    for row in rows:
        concepto = row["concepto"]
        values = row.get("values", [])

        if concepto.lower() == "totales":
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

    logger.info(f"Extracted {len(breakdown.items)} items from Nota 21")
    return breakdown


def extract_nota_22(pdf_path: Path, config: dict | None = None) -> CostBreakdown | None:
    """Extract Nota 22 - Gastos de Administración y Ventas from PDF.

    Args:
        pdf_path: Path to Estados Financieros PDF
        config: Configuration dict, or None to load from file

    Returns:
        CostBreakdown object or None if extraction fails
    """
    _, gasto_admin_items, _ = _get_extraction_labels(config)

    page_idx = find_nota_page(pdf_path, 22)
    if page_idx is None:
        # Nota 22 is often on the same page as Nota 21
        page_idx = find_nota_page(pdf_path, 21)
        if page_idx is None:
            logger.error("Could not find Nota 22 in PDF")
            return None

    rows = extract_table_from_page(pdf_path, page_idx, gasto_admin_items, nota_number=22)

    if not rows:
        # Try next page
        rows = extract_table_from_page(pdf_path, page_idx + 1, gasto_admin_items, nota_number=22)

    if not rows:
        logger.error("Could not extract Nota 22 table")
        return None

    breakdown = CostBreakdown(
        nota_number=22,
        nota_title="Gastos de Administración y Ventas",
        page_number=page_idx + 1,
    )

    for row in rows:
        concepto = row["concepto"]
        values = row.get("values", [])

        if concepto.lower() == "totales":
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

    logger.info(f"Extracted {len(breakdown.items)} items from Nota 22")
    return breakdown


def extract_xbrl_totals(xbrl_path: Path) -> dict[str, int | None]:
    """Extract relevant totals from XBRL file using config-driven fact names.

    Uses fact_mappings from config/xbrl_specs.json to look up the correct
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
        fact_mapping = get_xbrl_fact_mapping(field_name)
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
    pdf_nota_21: CostBreakdown | None,
    pdf_nota_22: CostBreakdown | None,
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
    result.nota_21 = extract_nota_21(pdf_path)
    result.nota_22 = extract_nota_22(pdf_path)

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
            result.xbrl_cost_of_sales = xbrl_totals.get("cost_of_sales")
            result.xbrl_admin_expense = xbrl_totals.get("admin_expense")

            result.validations = validate_extraction(
                result.nota_21,
                result.nota_22,
                xbrl_totals,
            )
    else:
        logger.info(f"No XBRL available for {year} Q{quarter} - using PDF only")
        result.xbrl_available = False

        if validate:
            # Still perform PDF-only validation
            result.validations = validate_extraction(
                result.nota_21,
                result.nota_22,
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
    data = {
        "period": f"{result.year}_Q{result.quarter}",
        "source": result.source,
        "pdf_path": str(result.pdf_path) if result.pdf_path else None,
        "xbrl_path": str(result.xbrl_path) if result.xbrl_path else None,
        "xbrl_available": result.xbrl_available,
        "nota_21": _breakdown_to_dict(result.nota_21) if result.nota_21 else None,
        "nota_22": _breakdown_to_dict(result.nota_22) if result.nota_22 else None,
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


def _breakdown_to_dict(breakdown: CostBreakdown) -> dict[str, Any]:
    """Convert CostBreakdown to dictionary."""
    return {
        "nota_number": breakdown.nota_number,
        "nota_title": breakdown.nota_title,
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

    if result.nota_21:
        print(f"\n--- Nota 21: {result.nota_21.nota_title} ---")
        print(f"Page: {result.nota_21.page_number}")
        print(f"Items extracted: {len(result.nota_21.items)}")
        for item in result.nota_21.items:
            val = f"{item.ytd_actual:,}" if item.ytd_actual else "N/A"
            print(f"  {item.concepto}: {val}")
        total = f"{result.nota_21.total_ytd_actual:,}" if result.nota_21.total_ytd_actual else "N/A"
        print(f"  TOTAL: {total}")
    else:
        print("\n--- Nota 21: EXTRACTION FAILED ---")

    if result.nota_22:
        print(f"\n--- Nota 22: {result.nota_22.nota_title} ---")
        print(f"Page: {result.nota_22.page_number}")
        print(f"Items extracted: {len(result.nota_22.items)}")
        for item in result.nota_22.items:
            val = f"{item.ytd_actual:,}" if item.ytd_actual else "N/A"
            print(f"  {item.concepto}: {val}")
        total = f"{result.nota_22.total_ytd_actual:,}" if result.nota_22.total_ytd_actual else "N/A"
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
# Sheet1 Data Structure (27 rows as per config.json)
# =============================================================================


@dataclass
class Sheet1Data:
    """Data structure for Sheet1 - Ingresos y Costos.

    This follows the 27-row structure defined in config/extraction_specs.json.
    Field definitions are loaded from the sheet1_fields section.

    The dataclass fields are kept for IDE autocomplete, but the config
    is the source of truth for field definitions and row mappings.
    """

    quarter: str  # e.g., "IIQ2024", "06-2024", "FY2024"
    year: int
    quarter_num: int  # Period number (1-4 for quarterly, 1-12 for monthly, 1 for yearly)
    period_type: str = "quarterly"  # "quarterly", "monthly", "yearly"
    source: str = "cmf"  # "cmf" or "pucobre.cl"
    xbrl_available: bool = False

    # Row 1: Ingresos
    ingresos_ordinarios: int | None = None

    # Rows 4-14: Costo de Venta breakdown
    cv_gastos_personal: int | None = None
    cv_materiales: int | None = None
    cv_energia: int | None = None
    cv_servicios_terceros: int | None = None
    cv_depreciacion_amort: int | None = None
    cv_deprec_leasing: int | None = None
    cv_deprec_arrend: int | None = None
    cv_serv_mineros: int | None = None
    cv_fletes: int | None = None
    cv_gastos_diferidos: int | None = None
    cv_convenios: int | None = None

    # Row 15: Total Costo de Venta
    total_costo_venta: int | None = None

    # Rows 20-25: Gasto Admin breakdown
    ga_gastos_personal: int | None = None
    ga_materiales: int | None = None
    ga_servicios_terceros: int | None = None
    ga_gratificacion: int | None = None
    ga_comercializacion: int | None = None
    ga_otros: int | None = None

    # Row 27: Totales (specifically Gasto Admin total, NOT Costo de Venta)
    total_gasto_admin: int | None = None

    def get_value(self, field_name: str) -> int | None:
        """Get a field value by name (config-driven access).

        Args:
            field_name: Field name from config (e.g., "cv_gastos_personal")

        Returns:
            Field value or None if not found.
        """
        return getattr(self, field_name, None)

    def set_value(self, field_name: str, value: int | None) -> None:
        """Set a field value by name (config-driven access).

        Args:
            field_name: Field name from config
            value: Value to set
        """
        if hasattr(self, field_name):
            setattr(self, field_name, value)
        else:
            logger.warning(f"Unknown field name: {field_name}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary matching config row_mapping."""
        return {
            "quarter": self.quarter,
            "year": self.year,
            "quarter_num": self.quarter_num,
            "period_type": self.period_type,
            "source": self.source,
            "xbrl_available": self.xbrl_available,
            "ingresos_ordinarios": self.ingresos_ordinarios,
            "cv_gastos_personal": self.cv_gastos_personal,
            "cv_materiales": self.cv_materiales,
            "cv_energia": self.cv_energia,
            "cv_servicios_terceros": self.cv_servicios_terceros,
            "cv_depreciacion_amort": self.cv_depreciacion_amort,
            "cv_deprec_leasing": self.cv_deprec_leasing,
            "cv_deprec_arrend": self.cv_deprec_arrend,
            "cv_serv_mineros": self.cv_serv_mineros,
            "cv_fletes": self.cv_fletes,
            "cv_gastos_diferidos": self.cv_gastos_diferidos,
            "cv_convenios": self.cv_convenios,
            "total_costo_venta": self.total_costo_venta,
            "ga_gastos_personal": self.ga_gastos_personal,
            "ga_materiales": self.ga_materiales,
            "ga_servicios_terceros": self.ga_servicios_terceros,
            "ga_gratificacion": self.ga_gratificacion,
            "ga_comercializacion": self.ga_comercializacion,
            "ga_otros": self.ga_otros,
            "total_gasto_admin": self.total_gasto_admin,
        }

    def to_row_list(self) -> list[tuple[int, str, int | None]]:
        """Convert to list of (row_number, label, value) tuples.

        Uses config/config.json row_mapping and config/extraction_specs.json
        sheet1_fields for row definitions. Raises error if config unavailable.

        Returns:
            List of (row_number, label, value) tuples for all 27 rows.

        Raises:
            ValueError: If required config is not available.
        """
        value_fields = get_sheet1_value_fields()
        if not value_fields:
            raise ValueError(
                "sheet1_fields.value_fields not found in config/extraction_specs.json. "
                "Config is the single source of truth - no hardcoded fallback."
            )

        row_mapping = get_sheet1_row_mapping()
        if not row_mapping:
            raise ValueError(
                "row_mapping not found in config/config.json sheets.sheet1. "
                "Config is the single source of truth - no hardcoded fallback."
            )

        return self._to_row_list_from_config(value_fields, row_mapping)

    def _to_row_list_from_config(
        self,
        value_fields: dict[str, Any],
        row_mapping: dict[str, dict[str, Any]],
    ) -> list[tuple[int, str, int | None]]:
        """Generate row list from config field definitions.

        Args:
            value_fields: Field definitions from extraction_specs.json
            row_mapping: Row mapping from config.json

        Returns:
            List of (row_number, label, value) tuples.
        """
        result = []

        for row_num in range(1, 28):
            row_key = str(row_num)
            row_def = row_mapping.get(row_key, {})
            field_name = row_def.get("field")
            label = row_def.get("label", "")

            if field_name and field_name not in ("costo_venta_header", "gasto_admin_header"):
                # Data field - get value
                value = self.get_value(field_name)
                result.append((row_num, label, value))
            else:
                # Header or blank row - no value
                result.append((row_num, label, None))

        return result


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


def find_cuadro_resumen_page(pdf_path: Path) -> int | None:
    """Find the page containing 'Cuadro Resumen de Costos' in Análisis Razonado.

    Args:
        pdf_path: Path to Análisis Razonado PDF

    Returns:
        0-indexed page number or None if not found
    """
    search_patterns = [
        "CUADRO RESUMEN DE COSTOS",
        "RESUMEN DE COSTOS",
        "Cuadro Resumen de Costos",
    ]

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").upper()

            for pattern in search_patterns:
                if pattern.upper() in text:
                    logger.info(f"Found 'Cuadro Resumen de Costos' on page {page_idx + 1}")
                    return page_idx

    return None


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
    nota_21 = extract_nota_21(ef_path)
    nota_22 = extract_nota_22(ef_path)

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

    # Validate/supplement with XBRL if available
    if xbrl_available and validate_with_xbrl:
        _validate_sheet1_with_xbrl(data, xbrl_path)

    return data


def _map_nota_item_to_sheet1(item: LineItem, data: Sheet1Data, section_name: str) -> None:
    """Map a Nota line item to Sheet1Data fields using config-driven matching.

    Uses match_keywords and exclude_keywords from extraction_specs.json
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

    Uses validation_rules from config/xbrl_specs.json for tolerance settings.

    Args:
        data: Sheet1Data populated from PDF
        xbrl_path: Path to XBRL file
    """
    xbrl_totals = extract_xbrl_totals(xbrl_path)
    tolerance = get_sum_tolerance()

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


def _find_cost_summary_table(tables: list[list[list[str | None]]]) -> list[list[str | None]] | None:
    """Find the cost summary table among extracted tables.

    The cost summary table should contain:
    - "Ingresos" or "actividades ordinarias"
    - "Costo de Venta"
    - "Gasto" and "Admin" or "Ventas"

    Args:
        tables: List of tables extracted from page

    Returns:
        The cost summary table or None
    """
    for table in tables:
        if not table:
            continue

        table_text = str(table).lower()

        has_ingresos = "ingresos" in table_text or "actividades ordinarias" in table_text
        has_costo = "costo" in table_text and "venta" in table_text
        has_gasto = "gasto" in table_text and ("admin" in table_text or "ventas" in table_text)

        if has_ingresos and has_costo and has_gasto:
            return table

    return None


def _parse_cost_summary_table(table: list[list[str | None]], data: Sheet1Data) -> None:
    """Parse cost summary table and populate Sheet1Data.

    The table has sections:
    1. Ingresos row
    2. Costo de Venta section (header + 11 items + total)
    3. Gasto Admin section (header + 6 items + Totales)

    Important: "Totales" at the end is specifically Gasto Admin total (row 27),
    not to be confused with "Total Costo de Venta".

    Args:
        table: Raw table data
        data: Sheet1Data object to populate
    """
    current_section = None  # None, "costo_venta", "gasto_admin"

    for row in table:
        if not row or not any(row):
            continue

        row_text = str(row[0] or "").strip().lower()

        # Detect section headers
        if "costo" in row_text and "venta" in row_text and "total" not in row_text:
            current_section = "costo_venta"
            continue
        elif "gasto" in row_text and ("admin" in row_text or "ventas" in row_text):
            current_section = "gasto_admin"
            continue

        # Extract first numeric value from row
        value = None
        for cell in row[1:]:
            if cell:
                parsed = parse_chilean_number(str(cell))
                if parsed is not None:
                    value = parsed
                    break

        # Map row to data field based on content and current section
        _map_row_to_field(row_text, value, current_section, data)


def _map_row_to_field(
    row_text: str,
    value: int | None,
    section: str | None,
    data: Sheet1Data,
) -> None:
    """Map a table row to the appropriate Sheet1Data field.

    Uses section context to disambiguate items that appear in both sections
    (e.g., "Gastos en personal" appears in both Costo de Venta and Gasto Admin).

    Args:
        row_text: Lowercase text from the row's first cell
        value: Parsed numeric value
        section: Current section ("costo_venta", "gasto_admin", or None)
        data: Sheet1Data object to update
    """
    # Ingresos (no section context needed)
    if "ingresos" in row_text and "ordinarias" in row_text:
        data.ingresos_ordinarios = value
        return

    # Total Costo de Venta (explicit match to avoid confusion with "Totales")
    if "total" in row_text and "costo" in row_text and "venta" in row_text:
        data.total_costo_venta = value
        return

    # Totales at end = Gasto Admin total (row 27)
    # This is the ONLY "Totales" without "Costo" or "Venta" qualification
    if row_text == "totales" or (row_text.startswith("totales") and "costo" not in row_text):
        if section == "gasto_admin":
            data.total_gasto_admin = value
        return

    # Section-specific items
    if section == "costo_venta":
        if "gastos en personal" in row_text or row_text == "gastos en personal":
            data.cv_gastos_personal = value
        elif "materiales" in row_text and "repuestos" in row_text:
            data.cv_materiales = value
        elif "energía" in row_text or "energia" in row_text:
            data.cv_energia = value
        elif "servicios de terceros" in row_text:
            data.cv_servicios_terceros = value
        elif "depreciación" in row_text or "depreciacion" in row_text:
            if "leasing" in row_text:
                data.cv_deprec_leasing = value
            elif "arrendamiento" in row_text:
                data.cv_deprec_arrend = value
            else:
                data.cv_depreciacion_amort = value
        elif "servicios mineros" in row_text:
            data.cv_serv_mineros = value
        elif "fletes" in row_text:
            data.cv_fletes = value
        elif "gastos diferidos" in row_text or "ajustes existencias" in row_text:
            data.cv_gastos_diferidos = value
        elif "convenios" in row_text or "obligaciones" in row_text:
            data.cv_convenios = value

    elif section == "gasto_admin":
        if "gastos en personal" in row_text or row_text == "gastos en personal":
            data.ga_gastos_personal = value
        elif "materiales" in row_text and "repuestos" in row_text:
            data.ga_materiales = value
        elif "servicios de terceros" in row_text:
            data.ga_servicios_terceros = value
        elif "gratificacion" in row_text or "gratificación" in row_text:
            data.ga_gratificacion = value
        elif "comercializacion" in row_text or "comercialización" in row_text:
            data.ga_comercializacion = value
        elif "otros gastos" in row_text:
            data.ga_otros = value


def save_sheet1_data(data: Sheet1Data, output_dir: Path | None = None) -> Path:
    """Save Sheet1 data to JSON file.

    Args:
        data: Sheet1Data to save
        output_dir: Output directory (defaults to processed dir)

    Returns:
        Path to saved file
    """
    if output_dir is None:
        paths = get_period_paths(data.year, data.quarter_num)
        output_dir = paths["processed"]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"sheet1_{data.quarter}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info(f"Saved Sheet1 data to: {output_path}")
    return output_path


def extract_sheet1_from_xbrl(year: int, quarter: int) -> Sheet1Data | None:
    """Extract Sheet1 totals directly from XBRL file.

    This extracts only the totals available in XBRL using fact_mappings
    from config/xbrl_specs.json:
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
        fact_mapping = get_xbrl_fact_mapping(field_name)
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


def print_sheet1_report(data: Sheet1Data) -> None:
    """Print a formatted Sheet1 report.

    Args:
        data: Sheet1Data to report
    """
    print(f"\n{'=' * 60}")
    print(f"Sheet1 Report: {data.quarter}")
    print(f"{'=' * 60}")
    print(f"Source: {data.source}")
    print(f"XBRL Available: {'Yes' if data.xbrl_available else 'No'}")

    print(f"\n{'Row':<4} {'Label':<45} {'Value':>12}")
    print("-" * 65)

    for row_num, label, value in data.to_row_list():
        if value is not None:
            val_str = f"{value:,}"
        elif label:
            val_str = ""
        else:
            val_str = ""
        print(f"{row_num:<4} {label:<45} {val_str:>12}")

    print(f"\n{'=' * 60}\n")

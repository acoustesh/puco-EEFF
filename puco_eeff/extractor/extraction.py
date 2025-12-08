"""PDF and XBRL extraction logic for cost data.

This module provides high-level extraction functions for parsing PDFs and XBRL files.
It contains config-driven extraction logic, page finding, and section extraction.

Key Classes:
    LineItem: Single line item with concepto and period values.
    SectionBreakdown: Generic container for PDF section data.

Key Functions:
    extract_pdf_section(): Generic config-driven PDF section extraction.
    find_text_page(): Generic PDF page finder - searches for required text strings.
    find_section_page(): Config-driven wrapper using find_text_page.
    extract_table_from_page(): Extract cost table data from a specific page.
    extract_xbrl_totals(): Extract relevant totals from XBRL file.
    extract_ingresos_from_pdf(): Extract Ingresos from Estado de Resultados.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber

from puco_eeff.config import (
    format_period_display,
    get_config,
    get_xbrl_scaling_factor,
    setup_logging,
)
from puco_eeff.extractor.table_parser import (
    extract_value_from_row,
    normalize_for_matching,
    parse_cost_table,
    score_table_match,
)
from puco_eeff.extractor.xbrl_parser import get_facts_by_name, parse_xbrl_file
from puco_eeff.sheets.sheet1 import (
    get_ingresos_pdf_fallback_config,
    get_section_fallback,
    get_sheet1_extraction_sections,
    get_sheet1_result_key_mapping,
    get_sheet1_section_expected_items,
    get_sheet1_section_field_mappings,
    get_sheet1_section_spec,
    get_sheet1_xbrl_fact_mapping,
)

logger = setup_logging(__name__)

# Public API exports
__all__ = [
    "LineItem",
    "SectionBreakdown",
    "get_section_expected_labels",
    "get_all_field_labels",
    "get_extraction_labels",
    "get_table_identifiers",
    "extract_pdf_section",
    "find_text_page",
    "find_section_page",
    "extract_table_from_page",
    "extract_xbrl_totals",
    "extract_ingresos_from_pdf",
    "quarter_to_roman",
    "format_period_label",
    "format_quarter_label",
]


# =============================================================================
# Dataclasses
# =============================================================================


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
    """Breakdown from a PDF section (e.g., nota_21, nota_22, ingresos)."""

    section_id: str
    section_title: str
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
        return self.sum_items_ytd_actual() == self.total_ytd_actual


# =============================================================================
# Config Helpers
# =============================================================================


def get_section_expected_labels(section_name: str, sheet_name: str = "sheet1") -> list[str]:
    """Get expected PDF labels for a section from config."""
    if sheet_name != "sheet1":
        raise ValueError(f"Sheet '{sheet_name}' not yet implemented.")
    return get_sheet1_section_expected_items(section_name)


def get_all_field_labels(sheet_name: str = "sheet1") -> dict[str, str]:
    """Get all field labels from all sections in config."""
    if sheet_name != "sheet1":
        raise ValueError(f"Sheet '{sheet_name}' not yet implemented.")

    field_labels = {}
    for section_name in get_sheet1_extraction_sections():
        field_mappings = get_sheet1_section_field_mappings(section_name)
        for field_name, field_spec in field_mappings.items():
            labels = field_spec.get("pdf_labels", [])
            if labels:
                field_labels[field_name] = labels[0]
    return field_labels


def get_extraction_labels(
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
    section1_items: list[str] = []
    section2_items: list[str] = []
    field_labels: dict[str, str] = {}

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


def get_table_identifiers(section_spec: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Get unique and exclude items for table identification."""
    identifiers = section_spec.get("table_identifiers", {})
    return (identifiers.get("unique_items", []), identifiers.get("exclude_items", []))


# =============================================================================
# PDF Page Finding
# =============================================================================


def find_text_page(
    pdf_path: Path,
    required_texts: list[str],
    optional_texts: list[str] | None = None,
    min_required: int | None = None,
    min_optional: int = 0,
) -> int | None:
    """Find the first page containing required text strings."""
    if min_required is None:
        min_required = 1

    required_normalized = []
    for text in required_texts:
        required_normalized.append(text.lower())
        normalized = normalize_for_matching(text)
        if normalized != text.lower():
            required_normalized.append(normalized)

    optional_lower = []
    if optional_texts:
        for text in optional_texts:
            optional_lower.append(text.lower())
            normalized = normalize_for_matching(text)
            if normalized != text.lower():
                optional_lower.append(normalized)

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").lower()

            required_count = sum(1 for req in required_normalized if req in text)
            if required_count < min_required:
                continue

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
    """Find the page number where a config-defined section exists."""
    section_spec = get_sheet1_section_spec(section_name)
    unique_items, _ = get_table_identifiers(section_spec)

    if not unique_items:
        raise ValueError(f"No unique_items defined for section '{section_name}'.")

    search_patterns = section_spec.get("search_patterns", [])
    if not search_patterns:
        pdf_fallback = section_spec.get("pdf_fallback", {})
        search_patterns = pdf_fallback.get("search_patterns", [])
    if not search_patterns:
        raise ValueError(f"No search_patterns defined for section '{section_name}'.")

    validation = section_spec.get("validation", {})
    min_detail_items = validation.get("min_detail_items", 3)
    has_totales = validation.get("has_totales_row", True)

    optional_texts = list(unique_items)
    if has_totales:
        optional_texts.append("totales")

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


# =============================================================================
# Table Extraction
# =============================================================================


def extract_table_from_page(
    pdf_path: Path,
    page_index: int,
    expected_items: list[str],
    nota_number: int = 0,
    section_name: str = "",
    year: int | None = None,
    quarter: int | None = None,
) -> list[dict[str, Any]]:
    """Extract cost table data from a specific page."""
    if not section_name and nota_number:
        section_name = f"nota_{nota_number}"
    if section_name:
        section_spec = get_sheet1_section_spec(section_name)
        unique_items, exclude_items = get_table_identifiers(section_spec)
    else:
        unique_items, exclude_items = [], []

    with pdfplumber.open(pdf_path) as pdf:
        if page_index >= len(pdf.pages):
            return []

        page = pdf.pages[page_index]
        tables = page.extract_tables()

        if not tables:
            logger.warning(f"No tables found on page {page_index + 1}")
            return []

        best_table = None
        best_score = 0

        for table in tables:
            if not table:
                continue
            score = score_table_match(table, expected_items, unique_items, exclude_items)
            if score > best_score:
                best_score = score
                best_table = table

        if best_table is None or best_score < 3:
            logger.warning(f"Could not find expected cost table on page {page_index + 1}")
            return []

        return parse_cost_table(best_table, expected_items)


# =============================================================================
# PDF Section Extraction
# =============================================================================


def extract_pdf_section(
    pdf_path: Path,
    section_name: str,
    sheet_name: str = "sheet1",
    year: int | None = None,
    quarter: int | None = None,
) -> SectionBreakdown | None:
    """Extract a section from PDF using config-driven rules."""
    section_spec = get_sheet1_section_spec(section_name)
    section_title = section_spec.get("title", section_name)
    expected_items = get_section_expected_labels(section_name, sheet_name)

    page_idx = find_section_page(pdf_path, section_name, year, quarter)

    if page_idx is None:
        fallback = get_section_fallback(section_name)
        if fallback:
            page_idx = find_section_page(pdf_path, fallback, year, quarter)

    if page_idx is None:
        logger.error(f"Could not find section '{section_name}' in PDF")
        return None

    rows = extract_table_from_page(
        pdf_path, page_idx, expected_items, section_name=section_name, year=year, quarter=quarter
    )

    if not rows:
        rows = extract_table_from_page(
            pdf_path, page_idx + 1, expected_items, section_name=section_name, year=year, quarter=quarter
        )

    if not rows:
        logger.error(f"Could not extract table for section '{section_name}'")
        return None

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
    """Extract Ingresos de actividades ordinarias from Estado de Resultados page."""
    ingresos_spec = get_sheet1_section_spec("ingresos")
    field_mappings = ingresos_spec.get("field_mappings", {})
    ingresos_mapping = field_mappings.get("ingresos_ordinarios", {})
    match_keywords = ingresos_mapping.get("match_keywords")
    if not match_keywords:
        raise KeyError("ingresos.field_mappings.ingresos_ordinarios.match_keywords missing from config.")

    pdf_config = get_ingresos_pdf_fallback_config()
    min_threshold = pdf_config["min_value_threshold"]

    page_idx = find_section_page(pdf_path, "ingresos")
    if page_idx is None:
        logger.warning("Could not find Estado de Resultados page for Ingresos extraction")
        return None

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_idx]
        tables = page.extract_tables()

        for table in tables:
            for row in table:
                if not row or len(row) < 2:
                    continue
                value = extract_value_from_row(row, match_keywords, min_threshold)
                if value is not None:
                    logger.info(f"Extracted Ingresos from PDF page {page_idx + 1}: {value:,}")
                    return value

    logger.warning("Could not extract Ingresos from Estado de Resultados page")
    return None


# =============================================================================
# XBRL Extraction
# =============================================================================


def _find_xbrl_facts(data: dict, fact_mapping: dict, field_name: str) -> list[dict]:
    """Find XBRL facts using primary and fallback names."""
    primary_fact = fact_mapping.get("primary")
    facts = get_facts_by_name(data, primary_fact) if primary_fact else []

    if not facts:
        for fallback in fact_mapping.get("fallbacks", []):
            facts = get_facts_by_name(data, fallback)
            if facts:
                logger.debug(f"Using fallback XBRL fact '{fallback}' for {field_name}")
                break
    return facts


def _extract_fact_value(facts: list[dict], fact_mapping: dict, scaling_factor: int) -> int | None:
    """Extract and optionally scale the value from XBRL facts."""
    for fact in facts:
        if not fact.get("value"):
            continue
        try:
            raw_value = int(float(fact["value"]))
            if fact_mapping.get("apply_scaling", False):
                return raw_value // scaling_factor
            return raw_value
        except (ValueError, TypeError):
            continue
    return None


def extract_xbrl_totals(xbrl_path: Path) -> dict[str, int | None]:
    """Extract relevant totals from XBRL file using config-driven fact names."""
    try:
        data = parse_xbrl_file(xbrl_path)
    except Exception as e:
        logger.error(f"Failed to parse XBRL: {e}")
        return {"cost_of_sales": None, "admin_expense": None, "ingresos": None}

    field_to_result = get_sheet1_result_key_mapping()
    result: dict[str, int | None] = {key: None for key in set(field_to_result.values())}
    scaling_factor = get_xbrl_scaling_factor()

    for field_name, result_key in field_to_result.items():
        fact_mapping = get_sheet1_xbrl_fact_mapping(field_name)
        if not fact_mapping:
            logger.debug(f"No XBRL mapping found for field: {field_name}")
            continue

        facts = _find_xbrl_facts(data, fact_mapping, field_name)
        value = _extract_fact_value(facts, fact_mapping, scaling_factor)
        if value is not None:
            result[result_key] = value
            logger.debug(f"Extracted {field_name}: {value}")

    return result


# =============================================================================
# Period Formatting
# =============================================================================


def quarter_to_roman(quarter: int) -> str:
    """Convert quarter number to Roman numeral format."""
    config = get_config()
    period_types = config.get("period_types", {})
    quarterly_config = period_types.get("quarterly", {})
    roman_map = quarterly_config.get("roman_numerals")

    if roman_map is None:
        raise ValueError("roman_numerals not found in config/config.json period_types.quarterly.")

    result = roman_map.get(str(quarter))
    if result is None:
        raise ValueError(f"Quarter {quarter} not found in roman_numerals mapping.")

    return result


def format_period_label(year: int, period: int, period_type: str = "quarterly") -> str:
    """Format period label as used in Sheet1 headers."""
    return format_period_display(year, period, period_type)


def format_quarter_label(year: int, quarter: int) -> str:
    """Format quarter label as used in Sheet1 headers (backward compatible)."""
    return format_period_label(year, quarter, "quarterly")


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# Re-export table_parser functions for backward compatibility
from puco_eeff.extractor.table_parser import (  # noqa: E402, F401
    parse_chilean_number,
    normalize_for_matching as _normalize_for_matching,
    match_item as _match_item,
    score_table_match as _score_table_match,
    parse_multiline_row as _parse_multiline_row,
    parse_single_row as _parse_single_row,
    parse_cost_table as _parse_cost_table,
    find_label_index as _find_label_index,
    count_value_offset as _count_value_offset,
    extract_value_from_row as _extract_value_from_row,
)

# Private aliases for backward compatibility
_get_extraction_labels = get_extraction_labels
_get_table_identifiers = get_table_identifiers

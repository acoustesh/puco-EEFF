"""PDF and XBRL extraction logic for cost data.

This module uses pdfplumber to locate and parse PDF tables and lxml-powered
helpers to read XBRL facts. Extraction is config-driven via ``config/sheet1``
mappings and returns structured dataclasses used downstream by Sheet1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pdfplumber

from puco_eeff.config import (
    get_xbrl_scaling_factor,
    quarter_to_roman,
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
    get_sheet1_section_table_identifiers,
    get_sheet1_xbrl_fact_mapping,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = setup_logging(__name__)

# Public API exports
__all__ = [
    "LineItem",
    "SectionBreakdown",
    "extract_ingresos_from_pdf",
    "extract_pdf_section",
    "extract_table_from_page",
    "extract_xbrl_totals",
    "find_section_page",
    "find_text_page",
    "format_quarter_label",
    "get_all_field_labels",
    "get_extraction_labels",
    "quarter_to_roman",
]


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class LineItem:
    """Line item with periodized values.

    Attributes
    ----------
    concepto : str
        Label as it appears in the PDF table.
    ytd_actual, ytd_anterior, quarter_actual, quarter_anterior : int | None
        Values parsed for current/prior year-to-date and quarter figures.
    """

    concepto: str
    ytd_actual: int | None = None
    ytd_anterior: int | None = None
    quarter_actual: int | None = None
    quarter_anterior: int | None = None


@dataclass
class SectionBreakdown:
    """Parsed PDF section payload (e.g., nota_21, nota_22, ingresos).

    Attributes
    ----------
    section_id : str
        Internal section key from configuration.
    section_title : str
        Human-readable section heading.
    items : list[LineItem]
        Detail rows excluding totals.
    total_ytd_actual, total_ytd_anterior, total_quarter_actual, total_quarter_anterior : int | None
        Parsed totals captured from the PDF table.
    page_number : int | None
        One-based page index for traceability.
    """

    section_id: str
    section_title: str
    items: list[LineItem] = field(default_factory=list)
    total_ytd_actual: int | None = None
    total_ytd_anterior: int | None = None
    total_quarter_actual: int | None = None
    total_quarter_anterior: int | None = None
    page_number: int | None = None

    def sum_items_ytd_actual(self) -> int:
        """Return the sum of YTD actual values excluding total rows."""
        return sum(item.ytd_actual or 0 for item in self.items if "total" not in item.concepto.lower())

    def is_valid(self) -> bool:
        """Return whether detail rows reconcile to the extracted total."""
        if self.total_ytd_actual is None:
            return False
        return self.sum_items_ytd_actual() == self.total_ytd_actual


# =============================================================================
# Config Helpers
# =============================================================================


def get_all_field_labels(
    sheet_name: str = "sheet1",
    config: dict | None = None,
) -> dict[str, str]:
    """Aggregate field-to-label mappings from extraction configs.

    When a custom config is provided (e.g., in tests), reads from the config's
    ``sheets.<sheet_name>.extraction_labels.field_labels`` path. Otherwise,
    builds labels dynamically from sheet1 section field mappings.

    Parameters
    ----------
    sheet_name : str, optional
        Sheet identifier; only ``"sheet1"`` is supported.
    config : dict | None, optional
        Optional configuration dictionary. When provided (e.g., in tests),
        reads field_labels from config. When ``None``, builds from extraction
        section configs.

    Returns
    -------
    dict[str, str]
        Mapping of field identifiers to their primary PDF labels used when
        matching parsed table headers.

    Raises
    ------
    ValueError
        If ``sheet_name`` is not implemented.
    """
    if sheet_name != "sheet1":
        msg = f"Sheet '{sheet_name}' not yet implemented."
        raise ValueError(msg)

    # When config is explicitly passed (e.g., tests), use it directly
    if config is not None:
        return (
            config.get("sheets", {})
            .get(sheet_name, {})
            .get("extraction_labels", {})
            .get("field_labels", {})
        )

    # Build from sheet1 section configs (the authoritative source)
    return {
        field_id: defn["pdf_labels"][0]
        for section_id in get_sheet1_extraction_sections()
        for field_id, defn in get_sheet1_section_field_mappings(section_id).items()
        if defn.get("pdf_labels")
    }


def _extract_labels_for_section(
    section_name: str,
    section_index: int,
    section1_items: list[str],
    section2_items: list[str],
    field_labels: dict[str, str],
) -> None:
    """Populate intermediate label collections for configured sections."""
    section_spec = get_sheet1_section_spec(section_name)
    field_mappings = section_spec.get("field_mappings", {})

    for field_name, field_spec in field_mappings.items():
        labels = field_spec.get("pdf_labels", [])
        if not labels:
            continue
        field_labels[field_name] = labels[0]
        # Exclude totals from items lists
        if field_name.startswith("total_"):
            continue
        if section_index == 0:
            section1_items.append(labels[0])
        elif section_index == 1:
            section2_items.append(labels[0])


def get_extraction_labels(
    config: dict | None = None,
) -> tuple[list[str], list[str], dict[str, str]]:
    """Load configured PDF labels for Sheet1 sections.

    Parameters
    ----------
    config : dict | None, optional
        Deprecated parameter retained for backward compatibility; ignored.

    Returns
    -------
    tuple[list[str], list[str], dict[str, str]]
        ``(costo_venta_items, gasto_admin_items, field_labels)`` extracted from
        sheet1 section definitions.

    Raises
    ------
    ValueError
        If fewer than two extraction sections are defined.
    """
    sections = get_sheet1_extraction_sections()
    if len(sections) < 2:
        msg = f"Expected at least 2 extraction sections in config, got {len(sections)}"
        raise ValueError(msg)

    section1_items: list[str] = []
    section2_items: list[str] = []
    field_labels: dict[str, str] = {}

    for i, section_name in enumerate(sections):
        _extract_labels_for_section(
            section_name,
            i,
            section1_items,
            section2_items,
            field_labels,
        )

    return section1_items, section2_items, field_labels


# =============================================================================
# PDF Page Finding
# =============================================================================


def _normalize_text_list(texts: list[str]) -> list[str]:
    """Return lowercase and normalized variants of provided strings."""
    result = []
    for text in texts:
        result.append(text.lower())
        normalized = normalize_for_matching(text)
        if normalized != text.lower():
            result.append(normalized)
    return result


def _page_matches_criteria(
    page_text: str,
    required_normalized: list[str],
    optional_normalized: list[str],
    min_required: int,
    min_optional: int,
) -> bool:
    """Return whether a page satisfies required/optional token thresholds."""
    required_count = sum(1 for req in required_normalized if req in page_text)
    if required_count < min_required:
        return False

    if optional_normalized and min_optional > 0:
        optional_count = sum(1 for opt in optional_normalized if opt in page_text)
        if optional_count < min_optional:
            return False

    return True


def find_text_page(
    pdf_path: Path,
    required_texts: list[str],
    optional_texts: list[str] | None = None,
    min_required: int | None = None,
    min_optional: int = 0,
) -> int | None:
    """Locate the first PDF page meeting text presence criteria.

    Parameters
    ----------
    pdf_path : Path
        PDF to scan.
    required_texts : list[str]
        Terms that must appear at least ``min_required`` times on a page.
    optional_texts : list[str] | None, optional
        Additional terms that count toward ``min_optional`` matches.
    min_required : int | None, optional
        Minimum number of required text matches (defaults to 1).
    min_optional : int, optional
        Minimum optional matches when ``optional_texts`` is provided.

    Returns
    -------
    int | None
        Zero-based page index if found; otherwise ``None``.
    """
    if min_required is None:
        min_required = 1

    required_normalized = _normalize_text_list(required_texts)
    optional_normalized = _normalize_text_list(optional_texts) if optional_texts else []

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_text = (page.extract_text() or "").lower()
            if _page_matches_criteria(
                page_text,
                required_normalized,
                optional_normalized,
                min_required,
                min_optional,
            ):
                logger.debug(f"Found text match on page {page_idx + 1}")
                return page_idx

    return None


def find_section_page(
    pdf_path: Path,
    section_name: str,
    year: int | None = None,
    quarter: int | None = None,
) -> int | None:
    """Find the PDF page index for a configured section.

    Parameters
    ----------
    pdf_path : Path
        PDF to scan.
    section_name : str
        Section identifier as defined in sheet1 extraction config.
    year, quarter : int | None
        Optional metadata used for logging context.

    Returns
    -------
    int | None
        Zero-based page index if located; otherwise ``None``.

    Raises
    ------
    ValueError
        If required search patterns are missing from configuration.
    """
    section_spec = get_sheet1_section_spec(section_name)
    unique_items, _ = get_sheet1_section_table_identifiers(section_name)

    if not unique_items:
        msg = f"No unique_items defined for section '{section_name}'."
        raise ValueError(msg)

    search_patterns = section_spec.get("search_patterns", [])
    if not search_patterns:
        pdf_fallback = section_spec.get("pdf_fallback", {})
        search_patterns = pdf_fallback.get("search_patterns", [])
    if not search_patterns:
        msg = f"No search_patterns defined for section '{section_name}'."
        raise ValueError(msg)

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


def _find_best_matching_table(
    tables: list[list[list[str | None]]],
    expected_items: list[str],
    unique_items: list[str],
    exclude_items: list[str],
    min_score: int = 3,
) -> list[list[str | None]] | None:
    """Select the table whose content best matches expectations.

    Parameters
    ----------
    tables : list[list[list[str | None]]]
        Tables extracted from a PDF page.
    expected_items : list[str]
        Concept labels expected to appear.
    unique_items : list[str]
        Labels that should appear uniquely and boost match score.
    exclude_items : list[str]
        Labels that disqualify a table when present.
    min_score : int, optional
        Minimum acceptable match score computed by :func:`score_table_match`.

    Returns
    -------
    list[list[str | None]] | None
        Best matching table when a score threshold is met; otherwise ``None``.
    """
    best_table = None
    best_score = 0

    for table in tables:
        if not table:
            continue
        score = score_table_match(table, expected_items, unique_items, exclude_items)
        if score > best_score:
            best_score = score
            best_table = table

    return best_table if best_score >= min_score else None


def extract_table_from_page(
    pdf_path: Path,
    page_index: int,
    expected_items: list[str],
    nota_number: int = 0,
    section_name: str = "",
    year: int | None = None,
    quarter: int | None = None,
) -> list[dict[str, Any]]:
    """Extract a structured table from a PDF page.

    Parameters
    ----------
    pdf_path : Path
        Source PDF.
    page_index : int
        Zero-based page index to parse.
    expected_items : list[str]
        Expected concept labels used to score tables.
    nota_number : int, optional
        Legacy note number used when ``section_name`` is absent.
    section_name : str, optional
        Section identifier to resolve unique/exclude items.
    year, quarter : int | None
        Optional metadata for logging context.

    Returns
    -------
    list[dict[str, Any]]
        Parsed rows with ``concepto`` and ``values`` keys; empty when no table matches.
    """
    # Resolve section name and get identifiers
    effective = section_name or (f"nota_{nota_number}" if nota_number else "")
    unique_items, exclude_items = get_sheet1_section_table_identifiers(effective) if effective else ([], [])

    with pdfplumber.open(pdf_path) as pdf:
        if page_index >= len(pdf.pages):
            return []

        tables = pdf.pages[page_index].extract_tables()
        if not tables:
            logger.warning(f"No tables found on page {page_index + 1}")
            return []

        best_table = _find_best_matching_table(tables, expected_items, unique_items, exclude_items)
        if best_table is None:
            logger.warning(f"Could not find expected cost table on page {page_index + 1}")
            return []

        return parse_cost_table(best_table, expected_items)


# =============================================================================
# PDF Section Extraction
# =============================================================================


def _extract_table_with_next_page_fallback(
    pdf_path: Path,
    page_idx: int,
    expected_items: list[str],
    section_name: str,
    year: int | None,
    quarter: int | None,
) -> list[dict[str, Any]]:
    """Extract from the target page, then try the next if nothing is found."""
    rows = extract_table_from_page(
        pdf_path,
        page_idx,
        expected_items,
        section_name=section_name,
        year=year,
        quarter=quarter,
    )
    if rows:
        return rows
    return extract_table_from_page(
        pdf_path,
        page_idx + 1,
        expected_items,
        section_name=section_name,
        year=year,
        quarter=quarter,
    )


def _safe_get_value(values: list, index: int) -> int | None:
    """Return a value by index or ``None`` when missing."""
    return values[index] if len(values) > index else None


def _set_breakdown_totals(breakdown: SectionBreakdown, values: list) -> None:
    """Populate total fields on a breakdown from an extracted values list."""
    breakdown.total_ytd_actual = _safe_get_value(values, 0)
    breakdown.total_ytd_anterior = _safe_get_value(values, 1)
    breakdown.total_quarter_actual = _safe_get_value(values, 2)
    breakdown.total_quarter_anterior = _safe_get_value(values, 3)


def _create_line_item(concepto: str, values: list) -> LineItem:
    """Build a :class:`LineItem` from a concept label and values list."""
    return LineItem(
        concepto=concepto,
        ytd_actual=_safe_get_value(values, 0),
        ytd_anterior=_safe_get_value(values, 1),
        quarter_actual=_safe_get_value(values, 2),
        quarter_anterior=_safe_get_value(values, 3),
    )


def _populate_breakdown_from_rows(breakdown: SectionBreakdown, rows: list[dict[str, Any]]) -> None:
    """Fill a :class:`SectionBreakdown` with parsed table rows."""
    for row in rows:
        concepto = row["concepto"]
        values = row.get("values", [])

        if concepto.lower() in {"totales", "total"}:
            _set_breakdown_totals(breakdown, values)
        else:
            breakdown.items.append(_create_line_item(concepto, values))


def extract_pdf_section(
    pdf_path: Path,
    section_name: str,
    sheet_name: str = "sheet1",
    year: int | None = None,
    quarter: int | None = None,
) -> SectionBreakdown | None:
    """Extract a section from a PDF using config-driven rules.

    Parameters
    ----------
    pdf_path : Path
        PDF containing the desired section.
    section_name : str
        Section identifier (e.g., ``"nota_21"``).
    sheet_name : str, optional
        Sheet identifier (only ``"sheet1"`` supported).
    year, quarter : int | None
        Optional metadata for logging.

    Returns
    -------
    SectionBreakdown | None
        Structured section data with item rows and totals; ``None`` when the
        section cannot be found or parsed.

    Raises
    ------
    ValueError
        If an unsupported sheet is requested.
    """
    if sheet_name != "sheet1":
        msg = f"Unsupported sheet: {sheet_name}"
        raise ValueError(msg)

    section_spec = get_sheet1_section_spec(section_name)
    section_title = section_spec.get("title", section_name)
    expected_items = get_sheet1_section_expected_items(section_name)

    # Find section page, trying fallback section if primary not found
    page_idx = find_section_page(pdf_path, section_name, year, quarter)
    if page_idx is None:
        fallback_section = get_section_fallback(section_name)
        if fallback_section:
            page_idx = find_section_page(pdf_path, fallback_section, year, quarter)
    if page_idx is None:
        logger.error("Could not find section '%s' in PDF", section_name)
        return None

    rows = _extract_table_with_next_page_fallback(
        pdf_path,
        page_idx,
        expected_items,
        section_name,
        year,
        quarter,
    )
    if not rows:
        logger.error("Could not extract table for section '%s'", section_name)
        return None

    breakdown = SectionBreakdown(
        section_id=section_name,
        section_title=section_title,
        page_number=page_idx + 1,
    )
    _populate_breakdown_from_rows(breakdown, rows)

    logger.info(f"Extracted {len(breakdown.items)} items from section '{section_name}'")
    return breakdown


def _get_ingresos_config() -> tuple[list[str], int]:
    """Return ingresos keyword list and minimum value threshold from config."""
    ingresos_spec = get_sheet1_section_spec("ingresos")
    field_mappings = ingresos_spec.get("field_mappings", {})
    ingresos_mapping = field_mappings.get("ingresos_ordinarios", {})
    match_keywords = ingresos_mapping.get("match_keywords")
    if not match_keywords:
        msg = "ingresos.field_mappings.ingresos_ordinarios.match_keywords missing from config."
        raise KeyError(msg)
    pdf_config = get_ingresos_pdf_fallback_config()
    return match_keywords, pdf_config["min_value_threshold"]


def _search_tables_for_ingresos(
    tables: list,
    match_keywords: list[str],
    min_threshold: int,
) -> int | None:
    """Search table rows for an ingresos value that meets matching rules."""
    for table in tables:
        for row in table:
            if not row or len(row) < 2:
                continue
            value = extract_value_from_row(row, match_keywords, min_threshold)
            if value is not None:
                return value
    return None


def extract_ingresos_from_pdf(pdf_path: Path) -> int | None:
    """Extract ingresos from the Estado de Resultados section.

    Parameters
    ----------
    pdf_path : Path
        PDF containing the Estado de Resultados.

    Returns
    -------
    int | None
        Parsed ingresos value when found; otherwise ``None``.
    """
    match_keywords, min_threshold = _get_ingresos_config()

    page_idx = find_section_page(pdf_path, "ingresos")
    if page_idx is None:
        logger.warning("Could not find Estado de Resultados page for Ingresos extraction")
        return None

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_idx]
        value = _search_tables_for_ingresos(page.extract_tables(), match_keywords, min_threshold)

    if value is not None:
        logger.info(f"Extracted Ingresos from PDF page {page_idx + 1}: {value:,}")
        return value

    logger.warning("Could not extract Ingresos from Estado de Resultados page")
    return None


# =============================================================================
# XBRL Extraction
# =============================================================================


def _find_xbrl_facts(data: dict, fact_mapping: dict, field_name: str) -> list[dict]:
    """Locate XBRL facts using primary and fallback fact names."""
    primary_fact = fact_mapping.get("primary")
    facts = get_facts_by_name(data, primary_fact) if primary_fact else []

    if not facts:
        for fallback in fact_mapping.get("fallbacks", []):
            facts = get_facts_by_name(data, fallback)
            if facts:
                logger.debug("Using fallback XBRL fact '%s' for %s", fallback, field_name)
                break
    return facts


def _extract_fact_value(facts: list[dict], fact_mapping: dict, scaling_factor: int) -> int | None:
    """Return the first valid XBRL fact value, applying scaling if configured."""
    for fact in facts:
        if not fact.get("value"):
            continue
        try:
            raw_value = int(float(fact["value"]))
            if fact_mapping.get("apply_scaling"):
                return raw_value // scaling_factor
            return raw_value
        except (ValueError, TypeError):
            continue
    return None


def extract_xbrl_totals(xbrl_path: Path) -> dict[str, int | None]:
    """Extract configured totals from an XBRL file.

    Parameters
    ----------
    xbrl_path : Path
        Path to the XBRL instance document.

    Returns
    -------
    dict[str, int | None]
        Values keyed by result mapping (e.g., ``cost_of_sales``); ``None`` when
        facts are absent or parsing fails.
    """
    try:
        data = parse_xbrl_file(xbrl_path)
    except Exception as e:
        logger.exception("Failed to parse XBRL: %s", e)
        return {"cost_of_sales": None, "admin_expense": None, "ingresos": None}

    field_to_result = get_sheet1_result_key_mapping()
    result: dict[str, int | None] = dict.fromkeys(set(field_to_result.values()))
    scaling_factor = get_xbrl_scaling_factor()

    for field_name, result_key in field_to_result.items():
        fact_mapping = get_sheet1_xbrl_fact_mapping(field_name)
        if not fact_mapping:
            logger.debug("No XBRL mapping found for field: %s", field_name)
            continue

        facts = _find_xbrl_facts(data, fact_mapping, field_name)
        value = _extract_fact_value(facts, fact_mapping, scaling_factor)
        if value is not None:
            result[result_key] = value
            logger.debug("Extracted %s: %s", field_name, value)

    return result


# Re-export format_quarter_label from config for backward compatibility
from puco_eeff.config import format_quarter_label as format_quarter_label  # noqa: E402

# =============================================================================
# Backward Compatibility Aliases
# =============================================================================
# Re-export table_parser functions for backward compatibility
from puco_eeff.extractor.table_parser import (  # noqa: E402, F401
    parse_chilean_number,
)

# Private aliases for backward compatibility
_get_extraction_labels = get_extraction_labels

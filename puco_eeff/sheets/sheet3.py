"""Sheet3 - Estado de Resultados (Income Statement) extraction module.

This module handles extraction of income statement data from XBRL (primary) or
Estados Financieros PDF (fallback).

Key Classes:
    Sheet3Data: Dataclass containing all extracted income statement values.

Key Config Accessors:
    get_sheet3_fields(): Load field definitions from config.
    get_sheet3_extraction_config(): Load extraction rules from config.
    get_sheet3_xbrl_mappings(): Load XBRL mappings from config.
    get_sheet3_reference_data(): Load reference data for validation.

Key Functions:
    extract_sheet3(): Main extraction entry point (XBRL primary, PDF fallback).
    save_sheet3_to_json(): Save Sheet3Data to JSON file.
    validate_sheet3_against_reference(): Compare against known-good values.

Configuration files:
- config/sheet3/fields.json: Field definitions, row mapping (15 rows)
- config/sheet3/extraction.json: Extraction rules (XBRL/PDF config)
- config/sheet3/xbrl_mappings.json: XBRL fact mappings
- config/sheet3/reference_data.json: Known-good values for validation
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from puco_eeff.config import (
    CONFIG_DIR,
    format_quarter_label,
    get_period_paths,
    quarter_to_roman,
)
from puco_eeff.extractor.xbrl_parser import get_facts_by_name, parse_xbrl_file
from puco_eeff.utils.parsing import (
    get_current_value,
    normalize_pdf_line,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Sheet3 Config Loading
# =============================================================================

_sheet3_fields_cache: dict | None = None
_sheet3_extraction_cache: dict | None = None
_sheet3_xbrl_cache: dict | None = None
_sheet3_reference_cache: dict | None = None


def get_sheet3_fields() -> dict:
    """Load Sheet3 field definitions from config.

    Returns
    -------
    dict
        Field definitions including value_fields, row_mapping, and layout.
    """
    global _sheet3_fields_cache
    if _sheet3_fields_cache is None:
        fields_path = CONFIG_DIR / "sheet3" / "fields.json"
        with fields_path.open("r", encoding="utf-8") as f:
            _sheet3_fields_cache = json.load(f)
    return _sheet3_fields_cache


def get_sheet3_extraction_config() -> dict:
    """Load Sheet3 extraction configuration.

    Returns
    -------
    dict
        Extraction rules including source_priority, xbrl_config, pdf_config.
    """
    global _sheet3_extraction_cache
    if _sheet3_extraction_cache is None:
        extraction_path = CONFIG_DIR / "sheet3" / "extraction.json"
        with extraction_path.open("r", encoding="utf-8") as f:
            _sheet3_extraction_cache = json.load(f)
    return _sheet3_extraction_cache


def get_sheet3_xbrl_mappings() -> dict:
    """Load Sheet3 XBRL mappings.

    Returns
    -------
    dict
        XBRL fact mappings and validation rules.
    """
    global _sheet3_xbrl_cache
    if _sheet3_xbrl_cache is None:
        xbrl_path = CONFIG_DIR / "sheet3" / "xbrl_mappings.json"
        with xbrl_path.open("r", encoding="utf-8") as f:
            _sheet3_xbrl_cache = json.load(f)
    return _sheet3_xbrl_cache


def get_sheet3_reference_data() -> dict:
    """Load Sheet3 reference data for validation.

    Returns
    -------
    dict
        Reference values keyed by period (e.g., '2024_Q2').
    """
    global _sheet3_reference_cache
    if _sheet3_reference_cache is None:
        ref_path = CONFIG_DIR / "sheet3" / "reference_data.json"
        with ref_path.open("r", encoding="utf-8") as f:
            _sheet3_reference_cache = json.load(f)
    return _sheet3_reference_cache


def get_sheet3_field_keywords() -> dict[str, list[str]]:
    """Get field keywords for PDF extraction.

    Returns
    -------
    dict[str, list[str]]
        Field name to keywords mapping.
    """
    config = get_sheet3_extraction_config()
    return config.get("field_keywords", {})


def get_reference_values_sheet3(year: int, quarter: int) -> dict[str, Any] | None:
    """Get reference values for a specific period.

    Parameters
    ----------
    year
        Year (e.g., 2024).
    quarter
        Quarter number (1-4).

    Returns
    -------
    dict[str, Any] | None
        Reference values for the period when present.
    """
    ref_data = get_sheet3_reference_data()
    period_key = f"{year}_Q{quarter}"

    period_data = ref_data.get(period_key, {})
    if period_data.get("verified") and period_data.get("values"):
        return cast("dict[str, Any]", period_data["values"])

    return None


# =============================================================================
# Sheet3 Data Class
# =============================================================================


@dataclass
class Sheet3Data:
    """Income Statement / Estado de Resultados data container.

    All monetary values are in MUS$ (thousands of USD) except where noted.
    Costs and expenses are stored as negative values.

    Attributes
    ----------
    quarter : str
        Quarter label (e.g., 'IIQ2024').
    year : int
        Fiscal year.
    quarter_num : int
        Quarter number (1-4).
    period_type : str
        'accumulated' - values are YTD accumulated.
    source : str
        Data source ('xbrl', 'pdf', or 'mixed').
    xbrl_available : bool
        Whether XBRL data was available.

    Income Statement Fields (MUS$):
    ingresos_ordinarios : int | None
        Revenue from ordinary activities.
    costo_ventas : int | None
        Cost of sales (negative).
    ganancia_bruta : int | None
        Gross profit = revenue + cost_ventas.
    otros_ingresos : int | None
        Other income.
    otros_egresos_funcion : int | None
        Other expenses by function (negative, often 0).
    ingresos_financieros : int | None
        Financial income.
    gastos_admin_ventas : int | None
        Admin & selling expenses (negative).
    costos_financieros : int | None
        Financial costs (negative).
    diferencias_cambio : int | None
        Exchange differences (can be +/-).
    ganancia_antes_impuestos : int | None
        Profit before taxes.
    gasto_impuestos : int | None
        Income tax expense (negative).
    ganancia_periodo : int | None
        Net profit for the period.
    resultado_accionistas : int | None
        Result attributable to shareholders.

    Share Data:
    acciones_emitidas : int | None
        Number of shares issued.
    acciones_dividendo : int | None
        Number of shares entitled to dividends.

    EPS (US$):
    ganancia_por_accion : float | None
        Basic earnings per share in US$.
    """

    # Metadata
    quarter: str = ""
    year: int = 0
    quarter_num: int = 0
    period_type: str = "accumulated"
    source: str = ""
    xbrl_available: bool = False

    # Income Statement Fields (MUS$)
    ingresos_ordinarios: int | None = None
    costo_ventas: int | None = None
    ganancia_bruta: int | None = None
    otros_ingresos: int | None = None
    otros_egresos_funcion: int | None = None
    ingresos_financieros: int | None = None
    gastos_admin_ventas: int | None = None
    costos_financieros: int | None = None
    diferencias_cambio: int | None = None
    ganancia_antes_impuestos: int | None = None
    gasto_impuestos: int | None = None
    ganancia_periodo: int | None = None
    resultado_accionistas: int | None = None

    # Share Data
    acciones_emitidas: int | None = None
    acciones_dividendo: int | None = None

    # EPS (US$, full precision)
    ganancia_por_accion: float | None = None

    # Tracking
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "quarter": self.quarter,
            "year": self.year,
            "quarter_num": self.quarter_num,
            "period_type": self.period_type,
            "source": self.source,
            "xbrl_available": self.xbrl_available,
            "ingresos_ordinarios": self.ingresos_ordinarios,
            "costo_ventas": self.costo_ventas,
            "ganancia_bruta": self.ganancia_bruta,
            "otros_ingresos": self.otros_ingresos,
            "otros_egresos_funcion": self.otros_egresos_funcion,
            "ingresos_financieros": self.ingresos_financieros,
            "gastos_admin_ventas": self.gastos_admin_ventas,
            "costos_financieros": self.costos_financieros,
            "diferencias_cambio": self.diferencias_cambio,
            "ganancia_antes_impuestos": self.ganancia_antes_impuestos,
            "gasto_impuestos": self.gasto_impuestos,
            "ganancia_periodo": self.ganancia_periodo,
            "resultado_accionistas": self.resultado_accionistas,
            "acciones_emitidas": self.acciones_emitidas,
            "acciones_dividendo": self.acciones_dividendo,
            "ganancia_por_accion": self.ganancia_por_accion,
            "issues": self.issues,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Sheet3Data:
        """Create instance from dictionary."""
        return cls(
            quarter=data.get("quarter", ""),
            year=data.get("year", 0),
            quarter_num=data.get("quarter_num", 0),
            period_type=data.get("period_type", "accumulated"),
            source=data.get("source", ""),
            xbrl_available=data.get("xbrl_available", False),
            ingresos_ordinarios=data.get("ingresos_ordinarios"),
            costo_ventas=data.get("costo_ventas"),
            ganancia_bruta=data.get("ganancia_bruta"),
            otros_ingresos=data.get("otros_ingresos"),
            otros_egresos_funcion=data.get("otros_egresos_funcion"),
            ingresos_financieros=data.get("ingresos_financieros"),
            gastos_admin_ventas=data.get("gastos_admin_ventas"),
            costos_financieros=data.get("costos_financieros"),
            diferencias_cambio=data.get("diferencias_cambio"),
            ganancia_antes_impuestos=data.get("ganancia_antes_impuestos"),
            gasto_impuestos=data.get("gasto_impuestos"),
            ganancia_periodo=data.get("ganancia_periodo"),
            resultado_accionistas=data.get("resultado_accionistas"),
            acciones_emitidas=data.get("acciones_emitidas"),
            acciones_dividendo=data.get("acciones_dividendo"),
            ganancia_por_accion=data.get("ganancia_por_accion"),
            issues=data.get("issues", []),
        )


# =============================================================================
# XBRL Extraction
# =============================================================================


def _find_current_context(contexts: dict[str, Any], year: int, quarter: int) -> str | None:
    """Find the appropriate context for current period extraction.

    Strategy:
    1. Look for context ID containing 'AcumuladoActual'
    2. Fallback: match by date range (start=Jan 1, end=quarter end)

    Parameters
    ----------
    contexts
        Context dictionary from parsed XBRL.
    year
        Target year.
    quarter
        Target quarter (1-4).

    Returns
    -------
    str | None
        Context ID or None if not found.
    """
    # Build expected end date
    quarter_end_dates = {
        1: f"{year}-03-31",
        2: f"{year}-06-30",
        3: f"{year}-09-30",
        4: f"{year}-12-31",
    }
    expected_end = quarter_end_dates.get(quarter)
    expected_start = f"{year}-01-01"

    # Strategy 1: Look for AcumuladoActual in context ID (not AcumuladoAnoActual)
    for ctx_id, ctx_info in contexts.items():
        if "AcumuladoActual" in ctx_id and "AcumuladoAnoActual" not in ctx_id:
            logger.debug("Found context by ID pattern: %s", ctx_id)
            return ctx_id

    # Strategy 2: Match by date range
    for ctx_id, ctx_info in contexts.items():
        if ctx_info.get("period_type") == "duration":
            start = ctx_info.get("start_date", "")
            end = ctx_info.get("end_date", "")
            if start == expected_start and end == expected_end:
                logger.debug("Found context by date range: %s (%s to %s)", ctx_id, start, end)
                return ctx_id

    # Strategy 3: Latest duration context
    latest_ctx = None
    latest_end = ""
    for ctx_id, ctx_info in contexts.items():
        if ctx_info.get("period_type") == "duration":
            end = ctx_info.get("end_date", "")
            if end > latest_end:
                latest_end = end
                latest_ctx = ctx_id

    if latest_ctx:
        logger.warning("Using latest context as fallback: %s (end: %s)", latest_ctx, latest_end)
        return latest_ctx

    return None


def _find_instant_context(contexts: dict[str, Any], year: int) -> str | None:
    """Find an instant/yearly context for share counts.

    Share counts in XBRL use a different context pattern - often 'AcumuladoAnoActual'
    instead of 'AcumuladoActual', as they represent point-in-time values.

    Parameters
    ----------
    contexts
        XBRL contexts dictionary.
    year
        Target year.

    Returns
    -------
    str | None
        Context ID for instant/yearly facts or None if not found.
    """
    # Strategy 1: Look for AcumuladoAnoActual pattern
    for ctx_id in contexts:
        if "AcumuladoAnoActual" in ctx_id:
            logger.debug("Found instant context by ID pattern: %s", ctx_id)
            return ctx_id

    # Strategy 2: Look for context with year start-to-end range (yearly context)
    expected_start = f"{year}-01-01"
    expected_end = f"{year}-12-31"

    for ctx_id, ctx_info in contexts.items():
        if ctx_info.get("period_type") == "duration":
            start = ctx_info.get("start_date", "")
            end = ctx_info.get("end_date", "")
            if start == expected_start and end == expected_end:
                logger.debug("Found instant context by yearly date range: %s", ctx_id)
                return ctx_id

    # Strategy 3: Look for any instant context ending in the current year
    for ctx_id, ctx_info in contexts.items():
        if ctx_info.get("period_type") == "instant":
            instant_date = ctx_info.get("instant_date", "")
            if instant_date.startswith(str(year)):
                logger.debug("Found instant context by date: %s (%s)", ctx_id, instant_date)
                return ctx_id

    return None


def _get_xbrl_value(
    data: dict[str, Any],
    fact_name: str,
    context_id: str,
    scale_factor: int = 1000,
    apply_scaling: bool = True,
) -> int | float | None:
    """Extract a single XBRL fact value for a given context.

    Parameters
    ----------
    data
        Parsed XBRL data.
    fact_name
        XBRL fact name to search for.
    context_id
        Context ID to filter by.
    scale_factor
        Divisor for scaling (default 1000 for MUS$).
    apply_scaling
        Whether to apply scaling.

    Returns
    -------
    int | float | None
        Extracted value or None if not found.
    """
    matching = get_facts_by_name(data, fact_name, exact=True)

    for fact in matching:
        if fact.get("context_ref") == context_id:
            try:
                raw_value = float(fact.get("value", 0))
                if apply_scaling and scale_factor > 1:
                    return int(raw_value / scale_factor)
                return raw_value
            except (ValueError, TypeError):
                logger.warning("Could not parse value for %s: %s", fact_name, fact.get("value"))
                return None

    return None


def _extract_from_xbrl(xbrl_path: Path, data: Sheet3Data) -> tuple[bool, list[str]]:
    """Extract Sheet3 data from XBRL file.

    Parameters
    ----------
    xbrl_path
        Path to XBRL file.
    data
        Sheet3Data instance to populate.

    Returns
    -------
    tuple[bool, list[str]]
        (success, issues).
    """
    issues: list[str] = []
    xbrl_mappings = get_sheet3_xbrl_mappings()
    extraction_config = get_sheet3_extraction_config()

    scale_factor = extraction_config.get("xbrl_config", {}).get("scale_factor", 1000)

    try:
        parsed = parse_xbrl_file(xbrl_path)
        contexts = parsed.get("contexts", {})
    except Exception as e:
        issues.append(f"XBRL parse error: {e}")
        return False, issues

    # Find current period context (for duration/income statement facts)
    context_id = _find_current_context(contexts, data.year, data.quarter_num)
    if not context_id:
        issues.append("Could not find appropriate XBRL context")
        return False, issues

    logger.info("Using XBRL context: %s", context_id)

    # Find instant context for share counts (they use different context)
    instant_context_id = _find_instant_context(contexts, data.year)
    if instant_context_id:
        logger.info("Using instant context for share fields: %s", instant_context_id)
    else:
        logger.warning("No instant context found for share fields, will use duration context")
        instant_context_id = context_id

    # Extract each field
    fact_mappings = xbrl_mappings.get("fact_mappings", {})
    fields_config = get_sheet3_fields().get("value_fields", {})

    extracted_count = 0

    for field_name, mapping in fact_mappings.items():
        primary_fact = mapping.get("primary")
        fallbacks = mapping.get("fallbacks", [])
        apply_scaling = mapping.get("apply_scaling", True)
        ensure_negative = mapping.get("ensure_negative", False)
        context_type = mapping.get("context_type", "duration")
        default_value = mapping.get("default_value")
        no_scaling = fields_config.get(field_name, {}).get("no_scaling", False)

        # Override scaling for specific fields
        if no_scaling:
            apply_scaling = False

        # Use instant context for share fields (context_type == "instant")
        field_context = instant_context_id if context_type == "instant" else context_id

        # Try primary fact first, then fallbacks
        value = None
        used_fact_name = None
        for fact_name in [primary_fact, *fallbacks]:
            value = _get_xbrl_value(
                parsed,
                fact_name,
                field_context,
                scale_factor=scale_factor,
                apply_scaling=apply_scaling,
            )
            if value is not None:
                used_fact_name = fact_name
                logger.debug(
                    "Found %s via %s: %s (context: %s)", field_name, fact_name, value, field_context
                )
                break

        if value is not None:
            # Ensure negative for cost fields
            if ensure_negative and value > 0:
                value = -value

            # Special handling: OtherExpenseByFunction is always an expense (negate if positive)
            # but OtherGainsLosses preserves its natural sign
            if (
                field_name == "otros_egresos_funcion"
                and used_fact_name in ("OtherExpenseByFunction", "OtherExpense")
                and value > 0
            ):
                value = -value
                logger.debug("Negated %s (expense fallback): %s", field_name, value)

            setattr(data, field_name, value)
            extracted_count += 1
        elif default_value is not None:
            # Use default value for fields not found in XBRL
            logger.debug(
                "Using default value %s for %s (not found in XBRL)", default_value, field_name
            )
            setattr(data, field_name, default_value)
            extracted_count += 1
        else:
            logger.debug("No XBRL value found for %s", field_name)

    success = extracted_count >= 10  # At least 10 of 16 fields
    if not success:
        issues.append(f"Only extracted {extracted_count} fields from XBRL")

    return success, issues


# =============================================================================
# PDF Extraction
# =============================================================================


def _find_pdf_path(year: int, quarter: int) -> Path | None:
    """Find the Estados Financieros PDF for the given period.

    Parameters
    ----------
    year
        Year.
    quarter
        Quarter number.

    Returns
    -------
    Path | None
        Path to PDF or None if not found.
    """
    paths = get_period_paths(year, quarter)
    raw_pdf = paths.get("raw_pdf")

    if not raw_pdf or not raw_pdf.exists():
        return None

    # Look for estados_financieros PDF
    quarter_roman = quarter_to_roman(quarter)
    patterns = [
        # New naming convention: estados_financieros_2024_Q1.pdf
        f"estados_financieros_{year}_Q{quarter}.pdf",
        # Alternative with roman numeral: estados_financieros_IQ2024.pdf
        f"estados_financieros_{quarter_roman}Q{year}.pdf",
        f"estados_financieros_{quarter_roman}T{year}.pdf",
        f"EEFF_{year}_Q{quarter}.pdf",
        f"EEFF_{quarter_roman}Q{year}.pdf",
        f"EstadosFinancieros_{year}_Q{quarter}.pdf",
    ]

    for pattern in patterns:
        pdf_path = raw_pdf / pattern
        if pdf_path.exists():
            return pdf_path

    # Try glob patterns with year and quarter
    glob_patterns = [
        f"*estados*{year}*Q{quarter}*.pdf",
        f"*estados*{year}*{quarter}*.pdf",
        f"*eeff*{year}*Q{quarter}*.pdf",
    ]
    for glob_pattern in glob_patterns:
        for pdf_file in raw_pdf.glob(glob_pattern):
            return pdf_file

    return None


def _extract_from_pdf(pdf_path: Path, data: Sheet3Data) -> tuple[bool, list[str]]:
    """Extract Sheet3 data from PDF using keyword matching.

    Parameters
    ----------
    pdf_path
        Path to Estados Financieros PDF.
    data
        Sheet3Data instance to populate.

    Returns
    -------
    tuple[bool, list[str]]
        (success, issues).
    """
    import pdfplumber

    issues: list[str] = []
    field_keywords = get_sheet3_field_keywords()

    extracted_count = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            all_text = ""
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                all_text += page_text + "\n"

            # Find Estado de Resultados section
            # Strategy: Find ALL sections matching the header, then pick the one with actual data
            lines = all_text.split("\n")
            all_sections: list[list[str]] = []
            current_section: list[str] = []
            in_section = False

            for line in lines:
                line_lower = line.lower()

                # Start of section - look for various header formats
                # Handle: "Estado de Resultados", "Estados de Resultados",
                # "Estados Consolidados Intermedios de Resultados", etc.
                is_results_header = (
                    "estado de resultados" in line_lower
                    or "estados de resultados" in line_lower
                    or "estado consolidado de resultados" in line_lower
                    or ("estados" in line_lower and "de resultados" in line_lower)
                )
                if is_results_header:
                    # Save previous section if it exists
                    if current_section:
                        all_sections.append(current_section)
                    current_section = []
                    in_section = True
                    continue

                # End of section (look for next major section)
                if in_section and (
                    "estado de flujo" in line_lower
                    or "estado de cambios" in line_lower
                    or "notas a los estados" in line_lower
                    or "las notas adjuntas" in line_lower
                ):
                    if current_section:
                        all_sections.append(current_section)
                    current_section = []
                    in_section = False
                    continue

                if in_section:
                    current_section.append(line)

            # Don't forget last section if file ended
            if current_section:
                all_sections.append(current_section)

            # Pick the section with actual data (has "ganancia bruta" or sufficient lines)
            section_lines = []
            for section in all_sections:
                section_text = " ".join(section).lower()
                # Check if this section has actual income statement content
                if "ganancia bruta" in section_text or "ingresos de actividades" in section_text:
                    section_lines = section
                    break
                # Fallback: pick longest section with at least 10 lines
                if len(section) > len(section_lines) and len(section) >= 10:
                    section_lines = section

            if not section_lines:
                issues.append("Could not find Estado de Resultados section in PDF")
                return False, issues

            logger.debug("Found %d lines in Estado de Resultados section", len(section_lines))

            # Match fields using keywords
            for field_name, keywords in field_keywords.items():
                for line in section_lines:
                    line_lower = line.lower()
                    line_normalized = normalize_pdf_line(line)

                    # Check if any keyword matches
                    if any(kw.lower() in line_lower for kw in keywords):
                        value = get_current_value(line_normalized)

                        if value is not None:
                            # Handle sign convention
                            extraction_config = get_sheet3_extraction_config()
                            negative_fields = extraction_config.get("sign_conventions", {}).get(
                                "ensure_negative_fields",
                                [],
                            )

                            if field_name in negative_fields and value > 0:
                                value = -value

                            # Set the value
                            if field_name == "ganancia_por_accion":
                                setattr(data, field_name, float(value))
                            elif field_name in ("acciones_emitidas", "acciones_dividendo"):
                                setattr(data, field_name, int(value))
                            else:
                                setattr(data, field_name, int(value))

                            extracted_count += 1
                            logger.debug(
                                "PDF extracted %s: %s from '%s'",
                                field_name,
                                value,
                                line.strip()[:60],
                            )
                            break

    except Exception as e:
        issues.append(f"PDF extraction error: {e}")
        return False, issues

    success = extracted_count >= 8  # At least 8 of 15 fields for PDF (less reliable)
    if not success:
        issues.append(f"Only extracted {extracted_count} fields from PDF")

    return success, issues


# =============================================================================
# Main Extraction Function
# =============================================================================


def extract_sheet3(
    year: int,
    quarter: int,
    xbrl_path: Path | None = None,
    pdf_path: Path | None = None,
) -> Sheet3Data:
    """Extract Sheet3 (Income Statement) data for a period.

    Strategy:
    1. Try XBRL extraction first (primary source from IIQ2024+)
    2. Fall back to PDF extraction if XBRL fails or unavailable
    3. Validate against reference data when available

    Parameters
    ----------
    year
        Fiscal year (e.g., 2024).
    quarter
        Quarter number (1-4).
    xbrl_path
        Optional explicit path to XBRL file.
    pdf_path
        Optional explicit path to PDF file.

    Returns
    -------
    Sheet3Data
        Extracted income statement data.
    """
    quarter_label = format_quarter_label(year, quarter)

    data = Sheet3Data(
        quarter=quarter_label,
        year=year,
        quarter_num=quarter,
        period_type="accumulated",
    )

    logger.info("Extracting Sheet3 for %s", quarter_label)

    # Get paths if not provided
    if xbrl_path is None or pdf_path is None:
        paths = get_period_paths(year, quarter)

        if xbrl_path is None:
            raw_xbrl = paths.get("raw_xbrl")
            if raw_xbrl and raw_xbrl.exists():
                # Find XBRL file matching year/quarter (try .xbrl first, then .xml)
                # Files are named like: estados_financieros_2024_Q2.xbrl
                patterns = [
                    f"*{year}_Q{quarter}*.xbrl",
                    f"*{year}_Q{quarter}*.xml",
                    f"*{year}*Q{quarter}*.xbrl",
                    f"*{year}*Q{quarter}*.xml",
                ]
                for pattern in patterns:
                    for xbrl_file in raw_xbrl.glob(pattern):
                        if "zip" not in xbrl_file.name:  # Skip zip files
                            xbrl_path = xbrl_file
                            break
                    if xbrl_path:
                        break

        if pdf_path is None:
            pdf_path = _find_pdf_path(year, quarter)

    # Try XBRL first
    xbrl_success = False
    if xbrl_path and xbrl_path.exists():
        logger.info("Attempting XBRL extraction from %s", xbrl_path)
        xbrl_success, xbrl_issues = _extract_from_xbrl(xbrl_path, data)
        data.issues.extend(xbrl_issues)

        if xbrl_success:
            data.source = "xbrl"
            data.xbrl_available = True
            logger.info("XBRL extraction successful")
        else:
            logger.warning("XBRL extraction failed: %s", xbrl_issues)
    else:
        logger.info("No XBRL file available for %s", quarter_label)

    # Try PDF if XBRL failed
    if not xbrl_success:
        if pdf_path and pdf_path.exists():
            logger.info("Attempting PDF extraction from %s", pdf_path)
            pdf_success, pdf_issues = _extract_from_pdf(pdf_path, data)
            data.issues.extend(pdf_issues)

            if pdf_success:
                data.source = "pdf"
                logger.info("PDF extraction successful")
            else:
                logger.error("PDF extraction failed: %s", pdf_issues)
                data.issues.append("Both XBRL and PDF extraction failed")
        else:
            logger.warning("No PDF file available for %s", quarter_label)
            data.issues.append("No XBRL or PDF files available")

    return data


# =============================================================================
# Output and Validation
# =============================================================================


def save_sheet3_to_json(data: Sheet3Data, output_path: Path) -> Path:
    """Save Sheet3Data to JSON file.

    Parameters
    ----------
    data
        Sheet3Data to save.
    output_path
        Destination path.

    Returns
    -------
    Path
        Written file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info("Saved Sheet3 data to %s", output_path)
    return output_path


def load_sheet3_from_json(json_path: Path) -> Sheet3Data:
    """Load Sheet3Data from JSON file.

    Parameters
    ----------
    json_path
        Path to JSON file.

    Returns
    -------
    Sheet3Data
        Loaded data.
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Sheet3Data.from_dict(data)


def validate_sheet3_against_reference(
    data: Sheet3Data,
    tolerance: int = 1,
) -> tuple[bool, list[str]]:
    """Validate Sheet3 data against reference values.

    Parameters
    ----------
    data
        Sheet3Data to validate.
    tolerance
        Allowed difference for monetary values.

    Returns
    -------
    tuple[bool, list[str]]
        (all_valid, list of discrepancy messages).
    """
    ref_values = get_reference_values_sheet3(data.year, data.quarter_num)
    if ref_values is None:
        return True, ["No reference data available for validation"]

    discrepancies: list[str] = []

    # Fields to validate
    fields_to_check = [
        "ingresos_ordinarios",
        "costo_ventas",
        "ganancia_bruta",
        "otros_ingresos",
        "ingresos_financieros",
        "gastos_admin_ventas",
        "costos_financieros",
        "diferencias_cambio",
        "ganancia_antes_impuestos",
        "gasto_impuestos",
        "ganancia_periodo",
        "resultado_accionistas",
        "acciones_emitidas",
        "acciones_dividendo",
        "ganancia_por_accion",
    ]

    for field_name in fields_to_check:
        extracted = getattr(data, field_name)
        reference = ref_values.get(field_name)

        if extracted is None or reference is None:
            continue

        # Special handling for EPS (float)
        if field_name == "ganancia_por_accion":
            if abs(extracted - reference) > 0.001:
                discrepancies.append(f"{field_name}: {extracted} vs ref {reference}")
        # Special handling for share counts (large integers)
        elif field_name in ("acciones_emitidas", "acciones_dividendo"):
            if extracted != reference:
                discrepancies.append(f"{field_name}: {extracted} vs ref {reference}")
        # Monetary values
        elif abs(extracted - reference) > tolerance:
            discrepancies.append(f"{field_name}: {extracted} vs ref {reference}")

    all_valid = len(discrepancies) == 0
    return all_valid, discrepancies


def validate_sheet3_subtotals(
    data: Sheet3Data,
    tolerance: int = 1,
) -> tuple[bool, list[str]]:
    """Validate that Sheet3 subtotals are internally consistent.

    Validates:
    1. Ganancia bruta = Ingresos ordinarios + Costo de ventas
    2. Ganancia antes de impuestos = Ganancia bruta + Otros ingresos + Otras ganancias
       + Ingresos financieros + Gastos admin + Costos financieros + Diferencias de cambio

    Parameters
    ----------
    data
        Sheet3Data to validate.
    tolerance
        Allowed difference for rounding errors (default 1 MUS$).

    Returns
    -------
    tuple[bool, list[str]]
        (all_valid, list of discrepancy messages).
    """
    issues: list[str] = []

    # Check 1: Ganancia bruta = Ingresos ordinarios + Costo de ventas
    if (
        data.ingresos_ordinarios is not None
        and data.costo_ventas is not None
        and data.ganancia_bruta is not None
    ):
        expected_gross = data.ingresos_ordinarios + data.costo_ventas
        diff = abs(expected_gross - data.ganancia_bruta)
        if diff > tolerance:
            issues.append(
                f"Ganancia bruta mismatch: {data.ganancia_bruta} != "
                f"{data.ingresos_ordinarios} + ({data.costo_ventas}) = {expected_gross}",
            )

    # Check 2: Ganancia antes de impuestos = sum of operating items
    # Components: ganancia_bruta + otros_ingresos + otros_egresos_funcion
    #           + ingresos_financieros + gastos_admin_ventas
    #           + costos_financieros + diferencias_cambio
    components = [
        ("ganancia_bruta", data.ganancia_bruta),
        ("otros_ingresos", data.otros_ingresos),
        ("otros_egresos_funcion", data.otros_egresos_funcion),
        ("ingresos_financieros", data.ingresos_financieros),
        ("gastos_admin_ventas", data.gastos_admin_ventas),
        ("costos_financieros", data.costos_financieros),
        ("diferencias_cambio", data.diferencias_cambio),
    ]

    # Check if all components are available
    all_available = all(v is not None for _, v in components)
    if all_available and data.ganancia_antes_impuestos is not None:
        expected_pretax = sum(v for _, v in components)  # type: ignore[misc]
        diff = abs(expected_pretax - data.ganancia_antes_impuestos)
        if diff > tolerance:
            breakdown = " + ".join(f"{name}={v}" for name, v in components)
            issues.append(
                f"Ganancia antes de impuestos mismatch: {data.ganancia_antes_impuestos} != "
                f"{expected_pretax} (sum of: {breakdown})",
            )

    all_valid = len(issues) == 0
    return all_valid, issues

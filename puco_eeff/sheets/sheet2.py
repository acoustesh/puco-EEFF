"""Sheet2 - Cuadro Resumen KPIs extraction module.

This module handles extraction of revenue breakdown by product, EBITDA, and
operational KPIs from Análisis Razonado PDF (or Estados Financieros fallback).

Key Classes:
    Sheet2Data: Dataclass containing all 14 extracted values (15-row structure).

Key Config Accessors:
    get_sheet2_fields(): Load field definitions from config.
    get_sheet2_extraction_config(): Load extraction rules from config.
    get_sheet2_xbrl_mappings(): Load XBRL mappings from config.
    get_sheet2_reference_data(): Load reference data from config.
    get_sheet2_section_spec(): Get extraction spec for a specific section.

Key Functions:
    extract_sheet2(): Main extraction entry point (PDF + optional XBRL validation).
    match_concepto_to_field_sheet2(): Match PDF label to field using keywords.
    parse_spanish_number(): Parse Spanish locale numbers (comma decimal separator).
    save_sheet2_to_json(): Save Sheet2Data to JSON file.
    validate_sheet2_against_reference(): Compare against known-good values.

Configuration files:
- config/sheet2/fields.json: Field definitions, row mapping (15 rows)
- config/sheet2/extraction.json: PDF extraction rules (sections, patterns)
- config/sheet2/xbrl_mappings.json: XBRL fact mappings, validation rules
- config/sheet2/reference_data.json: Known-good values for validation
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from puco_eeff.config import (
    CONFIG_DIR,
    format_quarter_label,
    get_period_paths,
    quarter_to_roman,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Sheet2 Config Loading
# =============================================================================

SHEET2_CONFIG_DIR = CONFIG_DIR / "sheet2"


def _load_sheet2_config(filename: str) -> dict[str, Any]:
    """Load a sheet2 config file.

    Parameters
    ----------
    filename
        Config filename (e.g., ``"fields.json"``).

    Returns
    -------
    dict[str, Any]
        Parsed JSON configuration.

    Raises
    ------
    FileNotFoundError
        If the config file cannot be located.
    """
    config_path = SHEET2_CONFIG_DIR / filename
    if not config_path.exists():
        msg = f"Sheet2 config not found: {config_path}"
        raise FileNotFoundError(msg)

    with config_path.open(encoding="utf-8") as f:
        return cast("dict[str, Any]", json.load(f))


def get_sheet2_fields() -> dict[str, Any]:
    """Load Sheet2 field definitions from fields.json."""
    return _load_sheet2_config("fields.json")


def get_sheet2_extraction_config() -> dict[str, Any]:
    """Load Sheet2 extraction config from extraction.json."""
    return _load_sheet2_config("extraction.json")


def get_sheet2_xbrl_mappings() -> dict[str, Any]:
    """Load Sheet2 XBRL mappings from xbrl_mappings.json."""
    return _load_sheet2_config("xbrl_mappings.json")


def get_sheet2_reference_data() -> dict[str, Any]:
    """Load Sheet2 reference data from reference_data.json."""
    return _load_sheet2_config("reference_data.json")


def get_sheet2_value_fields() -> dict[str, dict[str, Any]]:
    """Get Sheet2 value field definitions.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping of field names to their configuration entries.
    """
    fields = get_sheet2_fields()
    return cast("dict[str, dict[str, Any]]", fields.get("value_fields", {}))


def get_sheet2_metadata_fields() -> list[str]:
    """Get Sheet2 metadata field names.

    Returns
    -------
    list[str]
        Metadata field names to exclude from value comparisons.
    """
    fields = get_sheet2_fields()
    return cast("list[str]", fields.get("metadata_fields", []))


def get_sheet2_row_mapping() -> dict[str, dict[str, Any]]:
    """Get Sheet2 row mapping (row number -> field definition).

    Returns
    -------
    dict[str, dict[str, Any]]
        Row definitions keyed by row number as a string.
    """
    fields = get_sheet2_fields()
    return cast("dict[str, dict[str, Any]]", fields.get("row_mapping", {}))


def get_sheet2_section_spec(section_name: str) -> dict[str, Any]:
    """Get extraction spec for a specific section.

    Parameters
    ----------
    section_name
        Section key (e.g., ``"resumen_ingresos"``, ``"indicadores_operacionales"``).

    Returns
    -------
    dict[str, Any]
        Section specification dictionary.

    Raises
    ------
    ValueError
        If the section is not defined in ``sheet2/extraction.json``.
    """
    extraction_config = get_sheet2_extraction_config()
    sections = extraction_config.get("sections", {})
    section = sections.get(section_name)
    if section is None:
        msg = f"Section '{section_name}' not found in sheet2/extraction.json"
        raise ValueError(msg)
    return cast("dict[str, Any]", section)


def get_sheet2_section_field_mappings(section_name: str) -> dict[str, dict[str, Any]]:
    """Get field mappings for a specific section.

    Parameters
    ----------
    section_name
        Section name (``"resumen_ingresos"``, ``"indicadores_operacionales"``).

    Returns
    -------
    dict[str, dict[str, Any]]
        Field mapping entries including keywords and labels.
    """
    section = get_sheet2_section_spec(section_name)
    return cast("dict[str, Any]", section.get("field_mappings", {}))


def get_sheet2_extraction_sections() -> list[str]:
    """Get list of all extraction section keys for Sheet2.

    Returns
    -------
    list[str]
        Section keys such as ``"resumen_ingresos"``, ``"indicadores_operacionales"``.
    """
    extraction_config = get_sheet2_extraction_config()
    sections = extraction_config.get("sections", {})
    return list(sections.keys())


def get_sheet2_section_search_patterns(section_name: str) -> list[str]:
    """Get search patterns for finding a section in PDF.

    Parameters
    ----------
    section_name
        Section key from ``extraction.json``.

    Returns
    -------
    list[str]
        Search pattern strings.

    Raises
    ------
    ValueError
        If ``search_patterns`` are missing for the section.
    """
    section = get_sheet2_section_spec(section_name)
    patterns = section.get("search_patterns")
    if patterns is None:
        msg = f"search_patterns not found for section '{section_name}' in sheet2/extraction.json"
        raise ValueError(msg)
    return cast("list[str]", patterns)


def get_sheet2_sum_tolerance() -> int:
    """Get the numeric tolerance value for sum validation comparisons.

    Returns
    -------
    int
        Tolerance value (default: 1).
    """
    mappings = get_sheet2_xbrl_mappings()
    validation_rules = mappings.get("validation_rules", {})
    return cast("int", validation_rules.get("sum_tolerance", 1))


def get_sheet2_total_validations() -> list[dict[str, Any]]:
    """Get the list of total validation rule specifications.

    Returns
    -------
    list[dict[str, Any]]
        Validation rule dictionaries with ``sum_fields``, ``total_field``, and
        ``description``.
    """
    mappings = get_sheet2_xbrl_mappings()
    validation_rules = mappings.get("validation_rules", {})
    return cast("list[dict[str, Any]]", validation_rules.get("total_validations", []))


def get_sheet2_reference_values(year: int, quarter: int) -> dict[str, int | float | None] | None:
    """Get reference values for a specific period.

    Parameters
    ----------
    year
        Statement year.
    quarter
        Quarter number (1-4).

    Returns
    -------
    dict[str, int | float | None] | None
        Reference values for the period when present.
    """
    ref_data = get_sheet2_reference_data()
    period_key = f"{year}_Q{quarter}"

    period_data = ref_data.get(period_key, {})
    if period_data.get("verified") and period_data.get("values"):
        return cast("dict[str, int | float | None]", period_data["values"])

    return None


# =============================================================================
# Number Parsing (Spanish Locale)
# =============================================================================


def parse_spanish_number(value_str: str) -> int | float | None:
    """Parse a number string using Spanish locale conventions.

    Spanish locale uses:
    - Comma (,) as decimal separator
    - Period (.) as thousands separator

    Examples
    --------
    - "19,5" -> 19.5
    - "1.342" -> 1342
    - "64.057" -> 64057
    - "3,97" -> 3.97

    Parameters
    ----------
    value_str
        String representation of number in Spanish locale.

    Returns
    -------
    int | float | None
        Parsed numeric value, or None if parsing fails.
    """
    if not value_str or value_str.strip() == "":
        return None

    # Clean the string
    cleaned = value_str.strip()

    # Remove any currency symbols or extra characters
    cleaned = re.sub(r"[^\d.,\-]", "", cleaned)

    if not cleaned:
        return None

    try:
        # Check if it has a comma (decimal separator in Spanish)
        if "," in cleaned:
            # Replace thousands separator (.) with nothing
            # Then replace decimal separator (,) with period
            cleaned = cleaned.replace(".", "").replace(",", ".")
            return float(cleaned)

        if "." in cleaned:
            # Could be thousands separator or decimal
            # If there are multiple dots, they're thousands separators
            dot_count = cleaned.count(".")
            if dot_count > 1:
                # Multiple dots = thousands separators
                cleaned = cleaned.replace(".", "")
                return int(cleaned)

            # Single dot - check position for disambiguation
            # If exactly 3 digits after dot, it's likely thousands separator
            parts = cleaned.split(".")
            if len(parts) == 2 and len(parts[1]) == 3:
                # Thousands separator (e.g., "64.057" = 64057)
                cleaned = cleaned.replace(".", "")
                return int(cleaned)

            # Decimal separator (e.g., "3.97" = 3.97)
            return float(cleaned)

        # No separators - just a plain integer
        return int(cleaned)
    except ValueError:
        logger.warning("Could not parse number: %s", value_str)
        return None


# =============================================================================
# Field Matching
# =============================================================================


def match_concepto_to_field_sheet2(concepto: str, section_name: str) -> str | None:
    """Match a concepto string to a Sheet2 field using config-driven keywords.

    Uses ``match_keywords`` (at least one must match) and ``exclude_keywords``
    (none may match) from ``extraction.json`` to determine the field.

    Parameters
    ----------
    concepto
        Concept string from the PDF; compared in lowercase.
    section_name
        Section name (``"resumen_ingresos"``, ``"indicadores_operacionales"``).

    Returns
    -------
    str | None
        Field name when matched, otherwise ``None``.
    """
    concepto_lower = concepto.lower()
    field_mappings = get_sheet2_section_field_mappings(section_name)

    for field_name, mapping in field_mappings.items():
        match_keywords = mapping.get("match_keywords", [])
        exclude_keywords = mapping.get("exclude_keywords", [])

        # Check exclude keywords first - if any match, skip this field
        if exclude_keywords and any(kw.lower() in concepto_lower for kw in exclude_keywords):
            continue

        # Check match keywords - at least one must match
        if match_keywords and any(kw.lower() in concepto_lower for kw in match_keywords):
            return field_name

    return None


# =============================================================================
# Sheet2 Data Class
# =============================================================================


@dataclass
class Sheet2Data:
    """Data structure for Sheet2 - Cuadro Resumen KPIs.

    This follows the 15-row structure defined in config/sheet2/fields.json.
    Contains revenue breakdown by product and operational KPIs.
    """

    quarter: str  # e.g., "IIQ2024"
    year: int
    quarter_num: int  # Period number (1-4)
    period_type: str = "quarterly"
    source: str = "analisis_razonado"  # "analisis_razonado", "estados_financieros", or "xbrl"
    xbrl_available: bool = False

    # Rows 1-5: Resumen de Ingresos (MUS$)
    cobre_concentrados: int | None = None
    cobre_catodos: int | None = None
    oro_subproducto: int | None = None
    plata_subproducto: int | None = None
    total_ingresos: int | None = None

    # Rows 7-15: Indicadores Operacionales (mixed units)
    ebitda: int | None = None  # MUS$
    libras_vendidas: float | None = None  # MM lbs
    cobre_fino: float | None = None  # MM lbs
    precio_efectivo: float | None = None  # US$/lb
    cash_cost: float | None = None  # US$/lb
    costo_unitario_total: float | None = None  # US$/lb
    non_cash_cost: float | None = None  # US$/lb
    toneladas_procesadas: float | None = None  # miles ton
    oro_onzas: float | None = None  # miles oz

    def get_value(self, field_name: str) -> int | float | None:
        """Get a field value by name."""
        return getattr(self, field_name, None)

    def set_value(self, field_name: str, value: float | None) -> None:
        """Set a field value by name."""
        if hasattr(self, field_name):
            setattr(self, field_name, value)
        else:
            logger.warning("Unknown field name: %s", field_name)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary matching config row_mapping."""
        return {
            "quarter": self.quarter,
            "year": self.year,
            "quarter_num": self.quarter_num,
            "period_type": self.period_type,
            "source": self.source,
            "xbrl_available": self.xbrl_available,
            "cobre_concentrados": self.cobre_concentrados,
            "cobre_catodos": self.cobre_catodos,
            "oro_subproducto": self.oro_subproducto,
            "plata_subproducto": self.plata_subproducto,
            "total_ingresos": self.total_ingresos,
            "ebitda": self.ebitda,
            "libras_vendidas": self.libras_vendidas,
            "cobre_fino": self.cobre_fino,
            "precio_efectivo": self.precio_efectivo,
            "cash_cost": self.cash_cost,
            "costo_unitario_total": self.costo_unitario_total,
            "non_cash_cost": self.non_cash_cost,
            "toneladas_procesadas": self.toneladas_procesadas,
            "oro_onzas": self.oro_onzas,
        }

    def to_row_list(self) -> list[tuple[int, str, int | float | None]]:
        """Convert to list of (row_number, label, value) tuples.

        Uses config/sheet2/fields.json for row definitions.

        Returns
        -------
            List of (row_number, label, value) tuples for all 15 rows.
        """
        row_mapping = get_sheet2_row_mapping()

        result = []
        for row_num in range(1, 16):
            row_key = str(row_num)
            row_def = row_mapping.get(row_key, {})
            field_name = row_def.get("field")
            label = row_def.get("label", "")

            if field_name:
                value = self.get_value(field_name)
                result.append((row_num, label, value))
            else:
                result.append((row_num, label, None))

        return result


# =============================================================================
# Sheet2 I/O Functions
# =============================================================================


def save_sheet2_to_json(data: Sheet2Data, output_dir: Path | None = None) -> Path:
    """Save Sheet2Data to JSON file.

    Parameters
    ----------
    data
        Sheet2Data instance to save.
    output_dir
        Optional output directory. If None, uses default processed directory.

    Returns
    -------
    Path
        Path to the saved JSON file.
    """
    if output_dir is None:
        paths = get_period_paths(data.year, data.quarter_num)
        output_dir = paths["processed"]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"sheet2_{quarter_to_roman(data.quarter_num)}Q{data.year}.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info("Saved Sheet2 data to: %s", output_path)
    return output_path


def print_sheet2_report(data: Sheet2Data) -> None:
    """Print a formatted Sheet2 report.

    Parameters
    ----------
    data
        Sheet2Data instance to display.
    """
    print(f"\n{'=' * 60}")
    print(f"Sheet2 - Cuadro Resumen KPIs: {data.quarter}")
    print(f"Source: {data.source} | XBRL: {'Yes' if data.xbrl_available else 'No'}")
    print(f"{'=' * 60}")

    print("\nResumen de Ingresos (MUS$):")
    print(
        f"  Cobre en concentrados:  {data.cobre_concentrados:>12,}"
        if data.cobre_concentrados
        else "  Cobre en concentrados:  -"
    )
    print(
        f"  Cobre en cátodos:       {data.cobre_catodos:>12,}"
        if data.cobre_catodos
        else "  Cobre en cátodos:       -"
    )
    print(
        f"  Oro subproducto:        {data.oro_subproducto:>12,}"
        if data.oro_subproducto
        else "  Oro subproducto:        -"
    )
    print(
        f"  Plata subproducto:      {data.plata_subproducto:>12,}"
        if data.plata_subproducto
        else "  Plata subproducto:      -"
    )
    print(f"  {'─' * 40}")
    print(
        f"  TOTAL:                  {data.total_ingresos:>12,}"
        if data.total_ingresos
        else "  TOTAL:                  -"
    )

    print("\nIndicadores Operacionales:")
    print(
        f"  EBITDA (MUS$):          {data.ebitda:>12,}"
        if data.ebitda
        else "  EBITDA (MUS$):          -"
    )
    print(
        f"  Libras vendidas (MM):   {data.libras_vendidas:>12.1f}"
        if data.libras_vendidas
        else "  Libras vendidas (MM):   -"
    )
    print(
        f"  Cobre fino (MM lbs):    {data.cobre_fino:>12.1f}"
        if data.cobre_fino
        else "  Cobre fino (MM lbs):    -"
    )
    print(
        f"  Precio efectivo ($/lb): {data.precio_efectivo:>12.2f}"
        if data.precio_efectivo
        else "  Precio efectivo ($/lb): -"
    )
    print(
        f"  Cash cost ($/lb):       {data.cash_cost:>12.2f}"
        if data.cash_cost
        else "  Cash cost ($/lb):       -"
    )
    print(
        f"  Costo unitario ($/lb):  {data.costo_unitario_total:>12.2f}"
        if data.costo_unitario_total
        else "  Costo unitario ($/lb):  -"
    )
    print(
        f"  Non-cash cost ($/lb):   {data.non_cash_cost:>12.2f}"
        if data.non_cash_cost
        else "  Non-cash cost ($/lb):   -"
    )
    print(
        f"  Toneladas (miles):      {data.toneladas_procesadas:>12,.0f}"
        if data.toneladas_procesadas
        else "  Toneladas (miles):      -"
    )
    print(
        f"  Oro (miles oz):         {data.oro_onzas:>12.1f}"
        if data.oro_onzas
        else "  Oro (miles oz):         -"
    )

    print(f"{'=' * 60}\n")


# =============================================================================
# Validation Functions
# =============================================================================


def validate_sheet2_against_reference(data: Sheet2Data) -> list[str] | None:
    """Validate Sheet2 data against reference values.

    Parameters
    ----------
    data
        Sheet2Data to validate against reference baselines.

    Returns
    -------
    list[str] | None
        Issue strings when mismatches exceed tolerance, or ``None`` when no
        verified reference data is available.
    """
    ref_values = get_sheet2_reference_values(data.year, data.quarter_num)
    if ref_values is None:
        return None  # No verified reference data for this period

    issues = []
    value_fields = get_sheet2_value_fields()
    tolerance = get_sheet2_sum_tolerance()

    for field_name, field_def in value_fields.items():
        ref_value = ref_values.get(field_name)
        actual_value = data.get_value(field_name)

        if ref_value is not None and actual_value is not None:
            # Use appropriate tolerance based on field type
            field_type = field_def.get("type", "int")
            if field_type == "float":
                # For floats, use relative tolerance (1%)
                if abs(ref_value) > 0:
                    rel_diff = abs(ref_value - actual_value) / abs(ref_value)
                    if rel_diff > 0.01:  # 1% tolerance
                        issues.append(
                            f"{field_name}: expected {ref_value}, got {actual_value} (diff: {rel_diff:.1%})"
                        )
            else:
                # For integers, use absolute tolerance
                diff = abs(ref_value - actual_value)
                if diff > tolerance:
                    issues.append(
                        f"{field_name}: expected {ref_value:,}, got {actual_value:,} (diff: {diff})"
                    )
        elif ref_value is not None and actual_value is None:
            issues.append(f"{field_name}: expected {ref_value}, got None")

    return issues


def validate_sheet2_sums(data: Sheet2Data) -> list[str]:
    """Validate that sum of product revenues equals total.

    Parameters
    ----------
    data
        Sheet2Data to validate.

    Returns
    -------
    list[str]
        List of validation issues (empty if all pass).
    """
    issues = []
    tolerance = get_sheet2_sum_tolerance()

    # Check sum of revenue products
    products = [
        data.cobre_concentrados,
        data.cobre_catodos,
        data.oro_subproducto,
        data.plata_subproducto,
    ]

    if all(p is not None for p in products) and data.total_ingresos is not None:
        calculated_sum = sum(p for p in products if p is not None)
        diff = abs(calculated_sum - data.total_ingresos)
        if diff > tolerance:
            issues.append(
                f"Sum validation failed: {calculated_sum:,} != {data.total_ingresos:,} (diff: {diff})",
            )

    return issues


# =============================================================================
# Extraction Entry Point
# =============================================================================


def extract_sheet2(
    year: int,
    quarter: int,
    validate_with_xbrl: bool = True,
) -> tuple[Sheet2Data | None, list[str]]:
    """Extract Sheet2 data for a given period.

    This is the main entry point for Sheet2 extraction. It attempts to:
    1. Find and parse Análisis Razonado PDF (primary source)
    2. Fallback to Estados Financieros PDF if needed
    3. Optionally cross-validate totals with XBRL

    Parameters
    ----------
    year
        Fiscal year to extract.
    quarter
        Quarter number (1-4).
    validate_with_xbrl
        Whether to validate totals against XBRL when available.

    Returns
    -------
    tuple[Sheet2Data | None, list[str]]
        Extracted data and list of validation issues.
        Returns (None, [error_message]) on extraction failure.

    Note
    ----
    This is a stub implementation. Full PDF extraction logic should be
    implemented when the extraction patterns are finalized based on
    actual Análisis Razonado PDF structure.
    """
    logger.info("Extracting Sheet2 for %s Q%s", year, quarter)

    # Create data instance with metadata
    quarter_label = format_quarter_label(year, quarter)
    data = Sheet2Data(
        quarter=quarter_label,
        year=year,
        quarter_num=quarter,
        source="reference_data",  # Will be updated when PDF extraction is implemented
    )

    # For now, populate from reference data if available
    ref_values = get_sheet2_reference_values(year, quarter)
    if ref_values:
        for field_name, value in ref_values.items():
            data.set_value(field_name, value)
        logger.info("Populated Sheet2 from reference data for %s Q%s", year, quarter)
    else:
        logger.warning("No reference data available for %s Q%s", year, quarter)
        return None, [f"No reference data available for {year} Q{quarter}"]

    # Run sum validations
    validation_issues = validate_sheet2_sums(data)

    # Cross-validate with XBRL if requested
    if validate_with_xbrl:
        xbrl_issues = _validate_with_xbrl(data, year, quarter)
        if xbrl_issues:
            validation_issues.extend(xbrl_issues)

    return data, validation_issues


def _validate_with_xbrl(data: Sheet2Data, year: int, quarter: int) -> list[str]:
    """Cross-validate total_ingresos against XBRL Revenue.

    Parameters
    ----------
    data
        Sheet2Data instance to validate (updates xbrl_available flag).
    year
        Fiscal year.
    quarter
        Quarter number (1-4).

    Returns
    -------
    list[str]
        List of validation issues (empty if XBRL matches or unavailable).
    """
    from pathlib import Path

    from puco_eeff.extractor.xbrl_parser import parse_xbrl_file

    issues: list[str] = []
    xbrl_dir = Path("data/raw/xbrl")

    # Check for XBRL file
    xbrl_path = xbrl_dir / f"estados_financieros_{year}_Q{quarter}.xbrl"
    if not xbrl_path.exists():
        xbrl_path = xbrl_dir / f"estados_financieros_{year}_Q{quarter}.xml"

    if not xbrl_path.exists():
        logger.debug("No XBRL file found for %s Q%s", year, quarter)
        data.xbrl_available = False
        return issues

    data.xbrl_available = True

    try:
        xbrl_data = parse_xbrl_file(xbrl_path)
        contexts = xbrl_data.get("contexts", {})

        # Find YTD Revenue for current period (start of year to end of quarter)
        ytd_revenue = None
        for fact in xbrl_data.get("facts", []):
            if fact.get("name") == "Revenue":
                ctx = contexts.get(fact["context_ref"], {})
                # Look for YTD context (Jan 1 to quarter end)
                if ctx.get("start_date", "").startswith(f"{year}-01-01"):
                    try:
                        val = int(float(fact["value"])) // 1000  # Convert to MUS$
                        if ytd_revenue is None or val > ytd_revenue:
                            ytd_revenue = val
                    except (ValueError, TypeError):
                        pass

        if ytd_revenue is not None and data.total_ingresos is not None:
            # Compare with tolerance (1% or 1000 MUS$, whichever is larger)
            tolerance_abs = max(1000, int(data.total_ingresos * 0.01))
            diff = abs(ytd_revenue - data.total_ingresos)
            if diff > tolerance_abs:
                issues.append(
                    f"XBRL Revenue ({ytd_revenue:,}) differs from total_ingresos "
                    f"({data.total_ingresos:,}) by {diff:,} MUS$",
                )
            else:
                logger.debug(
                    "XBRL validation passed: Revenue=%s, total_ingresos=%s (diff=%s)",
                    ytd_revenue,
                    data.total_ingresos,
                    diff,
                )

    except Exception as e:
        logger.warning("XBRL validation failed: %s", e)
        issues.append(f"XBRL validation error: {e}")

    return issues


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "Sheet2Data",
    "extract_sheet2",
    "format_quarter_label",
    "get_sheet2_extraction_config",
    "get_sheet2_extraction_sections",
    "get_sheet2_fields",
    "get_sheet2_metadata_fields",
    "get_sheet2_reference_data",
    "get_sheet2_reference_values",
    "get_sheet2_row_mapping",
    "get_sheet2_section_field_mappings",
    "get_sheet2_section_search_patterns",
    "get_sheet2_section_spec",
    "get_sheet2_sum_tolerance",
    "get_sheet2_total_validations",
    "get_sheet2_value_fields",
    "get_sheet2_xbrl_mappings",
    "match_concepto_to_field_sheet2",
    "parse_spanish_number",
    "print_sheet2_report",
    "quarter_to_roman",
    "save_sheet2_to_json",
    "validate_sheet2_against_reference",
    "validate_sheet2_sums",
]

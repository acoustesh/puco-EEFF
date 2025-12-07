"""Sheet1 - Ingresos y Costos extraction module.

This module handles extraction of revenue and cost breakdown data from
Estados Financieros PDF (Nota 21 & 22) with optional XBRL validation.

Key Classes:
    Sheet1Data: Dataclass containing all 20 extracted values (27-row structure).

Key Functions:
    get_sheet1_fields(): Load field definitions from config.
    get_sheet1_row_mapping(): Load row mapping from config.
    get_sheet1_sum_tolerance(): Get tolerance for validation comparisons.
    get_sheet1_total_validations(): Get sum validation rules from config.
    get_sheet1_cross_validations(): Get cross-validation rules from config.
    validate_sheet1_against_reference(): Compare against known-good values.

Configuration files:
- config/sheet1/fields.json: Field definitions, row mapping (27 rows)
- config/sheet1/extraction.json: PDF extraction rules (sections, patterns)
- config/sheet1/xbrl_mappings.json: XBRL fact mappings, validation rules
- config/sheet1/reference_data.json: Known-good values for validation
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from puco_eeff.config import (
    CONFIG_DIR,
    format_period_display,
    get_config,
    get_period_paths,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Sheet1 Config Loading
# =============================================================================

SHEET1_CONFIG_DIR = CONFIG_DIR / "sheet1"


def _load_sheet1_config(filename: str) -> dict[str, Any]:
    """Load a sheet1 config file.

    Args:
        filename: Config filename (e.g., "fields.json")

    Returns:
        Parsed JSON config dict.

    Raises:
        FileNotFoundError: If config file not found.
    """
    config_path = SHEET1_CONFIG_DIR / filename
    if not config_path.exists():
        raise FileNotFoundError(f"Sheet1 config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def get_sheet1_fields() -> dict[str, Any]:
    """Load Sheet1 field definitions from fields.json."""
    return _load_sheet1_config("fields.json")


def get_sheet1_extraction_config() -> dict[str, Any]:
    """Load Sheet1 extraction config from extraction.json."""
    return _load_sheet1_config("extraction.json")


def get_sheet1_xbrl_mappings() -> dict[str, Any]:
    """Load Sheet1 XBRL mappings from xbrl_mappings.json."""
    return _load_sheet1_config("xbrl_mappings.json")


def get_sheet1_reference_data() -> dict[str, Any]:
    """Load Sheet1 reference data from reference_data.json."""
    return _load_sheet1_config("reference_data.json")


def get_sheet1_value_fields() -> dict[str, dict[str, Any]]:
    """Get Sheet1 value field definitions.

    Returns:
        Dictionary mapping field names to their definitions.
    """
    fields = get_sheet1_fields()
    return fields.get("value_fields", {})


def get_sheet1_metadata_fields() -> list[str]:
    """Get Sheet1 metadata field names.

    Returns:
        List of metadata field names.
    """
    fields = get_sheet1_fields()
    return fields.get("metadata_fields", [])


def get_sheet1_detail_fields(sections: list[str] | None = None) -> list[str]:
    """Get Sheet1 detail field names (non-total fields from specified sections).

    Detail fields are line items that have individual values extracted from PDF.
    Total fields (is_total: true) are excluded since they are computed or validated
    separately.

    Args:
        sections: List of section names to filter by (e.g., ["nota_21", "nota_22"]).
                  If None, returns detail fields from all PDF sections.

    Returns:
        List of detail field names (e.g., ["cv_gastos_personal", "cv_materiales", ...]).
    """
    value_fields = get_sheet1_value_fields()

    # Default to PDF sections (nota_21, nota_22) if not specified
    if sections is None:
        sections = ["nota_21", "nota_22"]

    detail_fields = []
    for field_name, field_def in value_fields.items():
        # Include field if it's in one of the specified sections and not a total
        field_section = field_def.get("section")
        is_total = field_def.get("is_total", False)
        if field_section in sections and not is_total:
            detail_fields.append(field_name)

    return detail_fields


def get_sheet1_row_mapping() -> dict[str, dict[str, Any]]:
    """Get Sheet1 row mapping (row number -> field definition).

    Returns:
        Dictionary mapping row numbers (as strings) to row definitions.
    """
    fields = get_sheet1_fields()
    return fields.get("row_mapping", {})


def get_sheet1_section_spec(section_name: str) -> dict[str, Any]:
    """Get extraction spec for a specific section.

    Args:
        section_name: Section key (e.g., "nota_21", "nota_22", "ingresos")

    Returns:
        Section specification dictionary.

    Raises:
        ValueError: If section not found.
    """
    extraction_config = get_sheet1_extraction_config()
    sections = extraction_config.get("sections", {})
    section = sections.get(section_name)
    if section is None:
        raise ValueError(f"Section '{section_name}' not found in sheet1/extraction.json")
    return section


def get_sheet1_section_field_mappings(section_name: str) -> dict[str, dict[str, Any]]:
    """Get field mappings for a specific section.

    Args:
        section_name: Section name ("nota_21", "nota_22", "ingresos")

    Returns:
        Dictionary of field mappings with match_keywords, exclude_keywords, etc.
    """
    section = get_sheet1_section_spec(section_name)
    return section.get("field_mappings", {})


def get_sheet1_extraction_sections() -> list[str]:
    """Get list of all extraction section keys for Sheet1.

    Returns:
        List of section keys (e.g., ["nota_21", "nota_22", "ingresos"])
    """
    extraction_config = get_sheet1_extraction_config()
    sections = extraction_config.get("sections", {})
    return list(sections.keys())


def get_sheet1_section_search_patterns(section_name: str) -> list[str]:
    """Get search patterns for finding a section in PDF.

    Args:
        section_name: Section key from extraction.json

    Returns:
        List of search pattern strings.

    Raises:
        ValueError: If search_patterns not found for section.
    """
    section = get_sheet1_section_spec(section_name)
    patterns = section.get("search_patterns")
    if patterns is None:
        raise ValueError(f"search_patterns not found for section '{section_name}' in sheet1/extraction.json")
    return patterns


def get_sheet1_section_table_identifiers(section_name: str) -> tuple[list[str], list[str]]:
    """Get unique and exclude items for identifying a section's table.

    Args:
        section_name: Section key from extraction.json

    Returns:
        Tuple of (unique_items, exclude_items) lists.
    """
    section = get_sheet1_section_spec(section_name)
    identifiers = section.get("table_identifiers", {})
    return (
        identifiers.get("unique_items", []),
        identifiers.get("exclude_items", []),
    )


def get_sheet1_section_expected_items(section_name: str) -> list[str]:
    """Get list of expected PDF labels for a section's line items.

    Extracts pdf_labels from field_mappings for the section.

    Args:
        section_name: Section key from extraction.json

    Returns:
        List of expected item label strings.
    """
    field_mappings = get_sheet1_section_field_mappings(section_name)
    items = []
    for mapping in field_mappings.values():
        pdf_labels = mapping.get("pdf_labels", [])
        items.extend(pdf_labels)
    return items


def get_sheet1_xbrl_fact_mapping(field_name: str) -> dict[str, Any] | None:
    """Get XBRL fact mapping for a specific field.

    Args:
        field_name: Field name (e.g., "ingresos_ordinarios")

    Returns:
        Dictionary with primary, fallbacks, and other XBRL info, or None.
    """
    xbrl_mappings = get_sheet1_xbrl_mappings()
    return xbrl_mappings.get("fact_mappings", {}).get(field_name)


def get_sheet1_validation_rules() -> dict[str, Any]:
    """Get validation rules from xbrl_mappings.json.

    Returns:
        Dictionary with sum_tolerance, total_validations, cross_validations.
    """
    xbrl_mappings = get_sheet1_xbrl_mappings()
    return xbrl_mappings.get("validation_rules", {})


def get_sheet1_sum_tolerance() -> int:
    """Get sum tolerance for validation (allows for rounding differences).

    Returns:
        Tolerance value (default 1).
    """
    rules = get_sheet1_validation_rules()
    return rules.get("sum_tolerance", 1)


def get_sheet1_total_validations() -> list[dict[str, Any]]:
    """Get total validation rules from xbrl_mappings.json.

    Returns:
        List of total validation rules, each containing:
        - total_field: Field name for the total
        - sum_fields: List of field names to sum
        - xbrl_fact: XBRL fact name for cross-validation
        - description: Human-readable description
    """
    rules = get_sheet1_validation_rules()
    return rules.get("total_validations", [])


def get_sheet1_cross_validations() -> list[dict[str, Any]]:
    """Get cross-validation rules from xbrl_mappings.json.

    Returns:
        List of cross-validation rules, each containing:
        - description: Human-readable description
        - formula: Formula string (e.g., "gross_profit == ingresos - cost")
        - tolerance: Optional per-rule tolerance (defaults to sum_tolerance)
    """
    rules = get_sheet1_validation_rules()
    return rules.get("cross_validations", [])


def get_sheet1_result_key_mapping() -> dict[str, str]:
    """Get result key mapping from xbrl_mappings.json.

    Maps Sheet1 field names to XBRL result keys used in extract_xbrl_totals().

    Returns:
        Dictionary mapping field names to result keys.
        Example: {"total_costo_venta": "cost_of_sales", ...}
    """
    xbrl_mappings = get_sheet1_xbrl_mappings()
    return xbrl_mappings.get("result_key_mapping", {})


def get_sheet1_pdf_xbrl_validations() -> list[dict[str, str]]:
    """Get PDF ↔ XBRL validation definitions from xbrl_mappings.json.

    Returns:
        List of validation definitions, each containing:
        - field_name: Sheet1 field name
        - xbrl_key: Key in XBRL result dict
        - display_name: Human-readable name for logging/reporting
    """
    xbrl_mappings = get_sheet1_xbrl_mappings()
    return xbrl_mappings.get("pdf_xbrl_validations", [])


def get_sheet1_reference_values(year: int, quarter: int) -> dict[str, int] | None:
    """Get reference values for a specific period.

    Args:
        year: Year of the financial statement
        quarter: Quarter (1-4)

    Returns:
        Dictionary of reference values, or None if not available.
    """
    ref_data = get_sheet1_reference_data()
    period_key = f"{year}_Q{quarter}"

    period_data = ref_data.get(period_key, {})
    if period_data.get("verified") and period_data.get("values"):
        return period_data["values"]

    return None


def match_concepto_to_field(concepto: str, section_name: str) -> str | None:
    """Match a concepto string to a Sheet1 field using config-driven keywords.

    Uses match_keywords (at least one must match) and exclude_keywords (none can match)
    from extraction.json to determine which field a concepto belongs to.

    Args:
        concepto: The concepto string from the PDF (will be lowercased)
        section_name: Section name ("nota_21", "nota_22", "ingresos")

    Returns:
        Field name if matched, None otherwise.
    """
    concepto_lower = concepto.lower()
    field_mappings = get_sheet1_section_field_mappings(section_name)

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
# Sheet1 Data Class
# =============================================================================


@dataclass
class Sheet1Data:
    """Data structure for Sheet1 - Ingresos y Costos.

    This follows the 27-row structure defined in config/sheet1/fields.json.
    Field definitions are loaded from the value_fields section.
    """

    quarter: str  # e.g., "IIQ2024"
    year: int
    quarter_num: int  # Period number (1-4)
    period_type: str = "quarterly"
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

    # Row 27: Totales (specifically Gasto Admin total)
    total_gasto_admin: int | None = None

    def get_value(self, field_name: str) -> int | None:
        """Get a field value by name."""
        return getattr(self, field_name, None)

    def set_value(self, field_name: str, value: int | None) -> None:
        """Set a field value by name."""
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

        Uses config/sheet1/fields.json for row definitions.

        Returns:
            List of (row_number, label, value) tuples for all 27 rows.
        """
        row_mapping = get_sheet1_row_mapping()

        result = []
        for row_num in range(1, 28):
            row_key = str(row_num)
            row_def = row_mapping.get(row_key, {})
            field_name = row_def.get("field")
            label = row_def.get("label", "")

            if field_name and field_name not in ("costo_venta_header", "gasto_admin_header"):
                value = self.get_value(field_name)
                result.append((row_num, label, value))
            else:
                result.append((row_num, label, None))

        return result


# =============================================================================
# Period Formatting (Sheet1-specific)
# =============================================================================


def quarter_to_roman(quarter: int) -> str:
    """Convert quarter number to Roman numeral format.

    Args:
        quarter: Quarter number (1-4)

    Returns:
        Roman numeral string (I, II, III, IV)
    """
    config = get_config()
    period_types = config.get("period_types", {})
    quarterly_config = period_types.get("quarterly", {})
    roman_map = quarterly_config.get("roman_numerals", {"1": "I", "2": "II", "3": "III", "4": "IV"})
    return roman_map.get(str(quarter), str(quarter))


def format_quarter_label(year: int, quarter: int) -> str:
    """Format quarter label as used in Sheet1 headers.

    Args:
        year: Year (e.g., 2024)
        quarter: Quarter number (1-4)

    Returns:
        Formatted string like "IIQ2024"
    """
    return format_period_display(year, quarter, "quarterly")


# =============================================================================
# Sheet1 I/O Functions
# =============================================================================


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


# =============================================================================
# Validation Functions
# =============================================================================


def validate_sheet1_against_reference(data: Sheet1Data) -> list[str] | None:
    """Validate Sheet1 data against reference values.

    Args:
        data: Sheet1Data to validate

    Returns:
        List of validation issue strings (empty if all match).
        Returns None if no verified reference data exists for the period.
    """
    ref_values = get_sheet1_reference_values(data.year, data.quarter_num)
    if ref_values is None:
        return None  # No verified reference data for this period

    issues = []
    value_fields = get_sheet1_value_fields()
    tolerance = get_sheet1_sum_tolerance()

    for field_name in value_fields:
        ref_value = ref_values.get(field_name)
        actual_value = data.get_value(field_name)

        if ref_value is not None and actual_value is not None:
            diff = abs(ref_value - actual_value)
            if diff > tolerance:
                issues.append(f"{field_name}: expected {ref_value:,}, got {actual_value:,} (diff: {diff})")
        elif ref_value is not None and actual_value is None:
            issues.append(f"{field_name}: expected {ref_value:,}, got None")

    return issues

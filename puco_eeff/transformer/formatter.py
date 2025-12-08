"""Sheet formatting and validation functions.

This module provides functions to:
- Get standard sheet structure from config
- Map extracted data to standard structure
- Validate balance sheets and cost totals
- Validate against reference data from config
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from puco_eeff.config import get_config, setup_logging

logger = setup_logging(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""

    field: str
    expected: int | None
    actual: int | None
    match: bool
    difference: int | None = None

    @property
    def status(self) -> str:
        """Return human-readable status."""
        if self.expected is None:
            return "⚠ No reference value"
        if self.actual is None:
            return "⚠ No actual value"
        if self.match:
            return "✓ Match"
        return f"✗ Mismatch (diff: {self.difference:,})"


def get_standard_structure(sheet_name: str = "sheet1", config: dict | None = None) -> list[dict]:
    """Get the standard row structure for a sheet.

    Args:
        sheet_name: Name of the sheet (e.g., "sheet1")
        config: Configuration dict, or None to load from file

    Returns:
        List of row definitions with field, label, section
    """
    if config is None:
        config = get_config()

    sheet_config = config.get("sheets", {}).get(sheet_name, {})
    row_mapping = sheet_config.get("row_mapping", {})

    # Convert row_mapping to ordered list
    structure = []
    for row_num in sorted(row_mapping.keys(), key=int):
        row_def = row_mapping[row_num]
        structure.append(
            {
                "row": int(row_num),
                "field": row_def.get("field"),
                "label": row_def.get("label", ""),
                "section": row_def.get("section"),
            }
        )

    return structure


def get_field_labels(sheet_name: str = "sheet1", config: dict | None = None) -> dict[str, str]:
    """Get field to label mapping for a sheet.

    Args:
        sheet_name: Name of the sheet
        config: Configuration dict, or None to load from file

    Returns:
        Dictionary mapping field names to display labels
    """
    if config is None:
        config = get_config()

    sheet_config = config.get("sheets", {}).get(sheet_name, {})
    extraction_labels = sheet_config.get("extraction_labels", {})

    return extraction_labels.get("field_labels", {})


def map_to_structure(
    data: dict[str, Any],
    sheet_name: str = "sheet1",
    config: dict | None = None,
) -> list[dict]:
    """Map extracted data to standard sheet structure.

    Args:
        data: Dictionary of field_name -> value
        sheet_name: Name of the sheet
        config: Configuration dict, or None to load from file

    Returns:
        List of row dictionaries with concepto and valor
    """
    structure = get_standard_structure(sheet_name, config)

    rows = []
    for row_def in structure:
        field = row_def.get("field")
        label = row_def.get("label", "")

        row = {
            "row": row_def["row"],
            "concepto": label,
            "valor": data.get(field) if field else None,
            "field": field,
            "section": row_def.get("section"),
        }
        rows.append(row)

    return rows


def _validate_section_total(
    data: dict[str, Any],
    item_fields: list[str],
    total_field: str,
) -> ValidationResult:
    """Validate that section items sum to total.

    Generic validation function for any section with line items that
    should sum to a total field.

    Args:
        data: Dictionary of field_name -> value
        item_fields: List of field names for line items
        total_field: Name of the total field to validate against

    Returns:
        ValidationResult with comparison
    """
    calculated_sum = sum(data.get(f, 0) or 0 for f in item_fields)
    reported_total = data.get(total_field)

    match = calculated_sum == reported_total if reported_total is not None else False
    difference = (reported_total - calculated_sum) if reported_total is not None else None

    return ValidationResult(
        field=total_field,
        expected=reported_total,
        actual=calculated_sum,
        match=match,
        difference=difference,
    )


# Field definitions for each section
_COSTO_VENTA_FIELDS = [
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
]

_GASTO_ADMIN_FIELDS = [
    "ga_gastos_personal",
    "ga_materiales",
    "ga_servicios_terceros",
    "ga_gratificacion",
    "ga_comercializacion",
    "ga_otros",
]


def validate_costo_venta_total(
    data: dict[str, Any],
    config: dict | None = None,
) -> ValidationResult:
    """Validate that Costo de Venta items sum to total.

    Args:
        data: Dictionary of field_name -> value
        config: Configuration dict (unused, kept for API compatibility)

    Returns:
        ValidationResult with comparison
    """
    return _validate_section_total(data, _COSTO_VENTA_FIELDS, "total_costo_venta")


def validate_gasto_admin_total(
    data: dict[str, Any],
    config: dict | None = None,
) -> ValidationResult:
    """Validate that Gasto Admin items sum to total.

    Args:
        data: Dictionary of field_name -> value
        config: Configuration dict (unused, kept for API compatibility)

    Returns:
        ValidationResult with comparison
    """
    return _validate_section_total(data, _GASTO_ADMIN_FIELDS, "total_gasto_admin")


def validate_balance_sheet(
    data: dict[str, Any],
    config: dict | None = None,
) -> list[ValidationResult]:
    """Validate all balance sheet totals.

    Args:
        data: Dictionary of field_name -> value
        config: Configuration dict, or None to load from file

    Returns:
        List of ValidationResults
    """
    results = []

    # Validate Costo de Venta total
    results.append(validate_costo_venta_total(data, config))

    # Validate Gasto Admin total
    results.append(validate_gasto_admin_total(data, config))

    return results


def validate_against_reference(
    extracted_data: dict[str, Any],
    period: str,
    config: dict | None = None,
) -> list[ValidationResult]:
    """Validate extracted data against reference values from config.

    Compares extracted values against known-good reference data stored
    in config["sheets"]["sheet1"]["data"][period].

    Args:
        extracted_data: Dictionary of field_name -> extracted value
        period: Period label (e.g., "IIQ2024")
        config: Configuration dict, or None to load from file

    Returns:
        List of ValidationResults comparing each field
    """
    if config is None:
        config = get_config()

    sheet1_config = config.get("sheets", {}).get("sheet1", {})
    reference_data = sheet1_config.get("data", {}).get(period, {})

    if not reference_data:
        logger.warning(f"No reference data found for period: {period}")
        return []

    results = []

    # Fields to validate (excluding metadata fields)
    metadata_fields = {"source", "xbrl_available"}

    for field, reference_value in reference_data.items():
        if field in metadata_fields:
            continue

        if not isinstance(reference_value, (int, float)):
            continue

        extracted_value = extracted_data.get(field)

        if extracted_value is None:
            results.append(
                ValidationResult(
                    field=field,
                    expected=int(reference_value),
                    actual=None,
                    match=False,
                    difference=None,
                )
            )
            continue

        expected = int(reference_value)
        actual = int(extracted_value)
        match = expected == actual
        difference = actual - expected if not match else 0

        results.append(
            ValidationResult(
                field=field,
                expected=expected,
                actual=actual,
                match=match,
                difference=difference,
            )
        )

    return results


def format_validation_report(results: list[ValidationResult]) -> str:
    """Format validation results as a human-readable report.

    Args:
        results: List of ValidationResults

    Returns:
        Formatted string report
    """
    if not results:
        return "No validations performed."

    lines = ["Validation Report", "=" * 50]

    passed = sum(1 for r in results if r.match)
    failed = len(results) - passed

    lines.append(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    lines.append("-" * 50)

    for result in results:
        lines.append(f"{result.field}: {result.status}")
        if not result.match and result.expected is not None and result.actual is not None:
            lines.append(f"  Expected: {result.expected:,}")
            lines.append(f"  Actual:   {result.actual:,}")

    return "\n".join(lines)

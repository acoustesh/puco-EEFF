"""Sheet formatting and validation functions.

This module provides functions to:
- Get standard sheet structure from config
- Map extracted data to standard structure
- Validate balance sheets and cost totals
- Validate against reference data from config
"""

from __future__ import annotations

from typing import Any, Protocol

from puco_eeff.config import get_config, setup_logging
from puco_eeff.extractor.validation import ComparisonResult

logger = setup_logging(__name__)

# Import unified validation result class - use ReferenceValidationResult alias
# for backward compatibility with existing code that uses this name
ReferenceValidationResult = ComparisonResult
ValidationResult = ComparisonResult


class ValidatorFunc(Protocol):
    """Protocol for validator functions with optional config parameter."""

    def __call__(
        self,
        data: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> ReferenceValidationResult:
        """Call validator with data and optional config."""
        ...


def get_standard_structure(
    sheet_name: str = "sheet1", config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Get the standard row structure for a sheet.

    Parameters
    ----------
    sheet_name
        Sheet identifier (e.g., ``"sheet1"``).
    config
        Optional configuration dictionary. When ``None``, configuration is
        loaded from disk.

    Returns
    -------
    list[dict[str, Any]]
        Ordered row definitions containing ``row``, ``field``, ``label``, and
        ``section`` keys.
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
            },
        )

    return structure


# Re-export get_all_field_labels as get_field_labels for backward compatibility.
# The unified implementation lives in extraction module and handles both
# dynamic config loading and explicit config passing for tests.


def map_to_structure(
    data: dict[str, Any],
    sheet_name: str = "sheet1",
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Map extracted data to standard sheet structure.

    Parameters
    ----------
    data
        Mapping of field name to extracted numeric value.
    sheet_name
        Sheet identifier used to load row metadata.
    config
        Optional configuration dictionary. When ``None``, configuration is
        loaded from disk.

    Returns
    -------
    list[dict[str, Any]]
        Row dictionaries containing ``concepto`` and ``valor`` aligned with the
        configured row mapping.
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

# Tuple versions for validator functions (immutable)
_COSTO_VENTA_FIELDS_TUPLE = tuple(_COSTO_VENTA_FIELDS)
_GASTO_ADMIN_FIELDS_TUPLE = tuple(_GASTO_ADMIN_FIELDS)


def _sum_fields(data: dict[str, Any], fields: tuple[str, ...]) -> int:
    """Sum numeric values from dict for given field names, treating None as 0."""
    return sum(data.get(f, 0) or 0 for f in fields)


# Section configuration for validation
_SECTION_CONFIG: dict[str, tuple[str, tuple[str, ...]]] = {
    "costo_venta": ("total_costo_venta", _COSTO_VENTA_FIELDS_TUPLE),
    "gasto_admin": ("total_gasto_admin", _GASTO_ADMIN_FIELDS_TUPLE),
}


def validate_section_total(
    data: dict[str, Any],
    section: str,
    config: dict[str, Any] | None = None,
) -> ValidationResult:
    """Validate that a section's total equals the sum of its line items.

    Parameters
    ----------
    data
        Mapping of field names to numeric values.
    section
        Section identifier: ``"costo_venta"`` or ``"gasto_admin"``.
    config
        Optional configuration (currently unused; kept for interface
        compatibility).

    Returns
    -------
    ValidationResult
        Comparison between reported total and calculated subtotal.

    Raises
    ------
    KeyError
        If the provided ``section`` is not recognized.
    """
    total_field, component_fields = _SECTION_CONFIG[section]
    calculated = _sum_fields(data, component_fields)
    reported = data.get(total_field)
    is_match = calculated == reported if reported is not None else False
    return ValidationResult(
        field_name=total_field,
        value_a=reported,
        value_b=calculated,
        match=is_match,
        comparison_type="reference",
        difference=(reported - calculated) if reported is not None else None,
    )


def validate_balance_sheet(
    data: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> list[ValidationResult]:
    """Validate all balance sheet totals.

    Parameters
    ----------
    data
        Mapping of field names to numeric values.
    config
        Optional configuration dictionary. When ``None``, configuration is
        loaded from disk.

    Returns
    -------
    list[ValidationResult]
        Validation results for each section total.
    """
    return [
        validate_section_total(data, "costo_venta", config),
        validate_section_total(data, "gasto_admin", config),
    ]


def validate_against_reference(
    extracted_data: dict[str, Any],
    period: str,
    config: dict[str, Any] | None = None,
) -> list[ValidationResult]:
    """Validate extracted data against reference values from config.

    Compares extracted values against known-good reference data stored
    in config["sheets"]["sheet1"]["data"][period].

    Parameters
    ----------
    extracted_data
        Mapping of field names to extracted numeric values.
    period
        Period label (e.g., ``"IIQ2024"``) used to select reference data.
    config
        Optional configuration dictionary. When ``None``, configuration is
        loaded from disk.

    Returns
    -------
    list[ValidationResult]
        Reference comparison results per field.
    """
    if config is None:
        config = get_config()

    sheet1_config = config.get("sheets", {}).get("sheet1", {})
    reference_data = sheet1_config.get("data", {}).get(period, {})

    if not reference_data:
        logger.warning("No reference data found for period: %s", period)
        return []

    results = []

    # Fields to validate (excluding metadata fields)
    metadata_fields = {"source", "xbrl_available"}

    for field, reference_value in reference_data.items():
        if field in metadata_fields:
            continue

        if not isinstance(reference_value, int | float):
            continue

        extracted_value = extracted_data.get(field)

        if extracted_value is None:
            results.append(
                ValidationResult(
                    field_name=field,
                    value_a=int(reference_value),  # expected
                    value_b=None,  # actual
                    match=False,
                    comparison_type="reference",
                    difference=None,
                ),
            )
            continue

        expected = int(reference_value)
        actual = int(extracted_value)
        match = expected == actual
        difference = actual - expected if not match else 0

        results.append(
            ValidationResult(
                field_name=field,
                value_a=expected,  # expected
                value_b=actual,  # actual
                match=match,
                comparison_type="reference",
                difference=difference,
            ),
        )

    return results


def format_validation_report(results: list[ValidationResult]) -> str:
    """Format validation results as a human-readable report.

    Parameters
    ----------
    results
        List of validation outcomes to summarize.

    Returns
    -------
    str
        Text report showing pass/fail counts and per-field status.
    """
    if not results:
        return "No validations performed."

    lines = ["Validation Report", "=" * 50]

    passed = sum(1 for r in results if r.match)
    failed = len(results) - passed

    lines.extend((f"Total: {len(results)} | Passed: {passed} | Failed: {failed}", "-" * 50))

    for result in results:
        lines.append(f"{result.field}: {result.status}")
        if not result.match and result.value_a is not None and result.value_b is not None:
            lines.extend((f"  Expected: {result.value_a:,}", f"  Actual:   {result.value_b:,}"))

    return "\n".join(lines)

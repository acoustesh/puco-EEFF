"""Validation runners for Sheet1 data.

This module provides the validation execution logic that runs sum validations,
PDF-XBRL comparisons, and cross-validations against extracted data.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from puco_eeff.extractor.validation.formula import (
    evaluate_cross_validation,
    resolve_cross_validation_values,
)
from puco_eeff.extractor.validation.types import (
    CrossValidationResult,
    SumValidationResult,
    ValidationReport,
    ValidationResult,
)
from puco_eeff.sheets.sheet1 import (
    get_sheet1_cross_validations,
    get_sheet1_pdf_xbrl_validations,
    get_sheet1_result_key_mapping,
    get_sheet1_sum_tolerance,
    get_sheet1_total_validations,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from puco_eeff.sheets.sheet1 import Sheet1Data

logger = logging.getLogger(__name__)

__all__ = [
    "compare_with_tolerance",
    "run_sheet1_validations",
]


def compare_with_tolerance(a: int | None, b: int | None, tolerance: int) -> tuple[bool, int]:
    """Compare two values with tolerance, using absolute values.

    Parameters
    ----------
    a
        First value (may be None).
    b
        Second value (may be None).
    tolerance
        Maximum allowed difference.

    Returns
    -------
    tuple
        (match, difference) where match is True if diff <= tolerance or either value is None.
    """
    # Treat missing values as a soft pass; downstream callers decide how to log.
    if a is None or b is None:
        return True, 0
    diff = abs(abs(a) - abs(b))
    return diff <= tolerance, diff


def _run_sum_validations_impl(data: Sheet1Data) -> list[SumValidationResult]:
    """Run config-driven sum validations on Sheet1Data.

    Parameters
    ----------
    data
        Sheet1Data instance containing extracted values.

    Returns
    -------
    list
        List of SumValidationResult for each configured validation rule.
    """
    results: list[SumValidationResult] = []
    global_tolerance = get_sheet1_sum_tolerance()
    total_validations = get_sheet1_total_validations()

    for rule in total_validations:
        description = rule.get("description", "Unknown validation")
        total_field = rule.get("total_field", "")
        sum_fields = rule.get("sum_fields", [])
        rule_tolerance = rule.get("tolerance", global_tolerance)

        expected_total = data.get_value(total_field)
        calculated_sum = sum(data.get_value(fld) or 0 for fld in sum_fields)

        match, difference = compare_with_tolerance(expected_total, calculated_sum, rule_tolerance)
        if expected_total is None:
            difference = 0
            match = True

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
            logger.info(
                "⚠ %s: no total value to compare (sum=%s)", description, f"{calculated_sum:,}"
            )
        elif match:
            logger.info(
                "✓ %s: sum=%s matches total=%s",
                description,
                f"{calculated_sum:,}",
                f"{expected_total:,}",
            )
        else:
            logger.warning(
                "✗ %s: sum=%s != total=%s (diff: %s)",
                description,
                f"{calculated_sum:,}",
                f"{expected_total:,}",
                difference,
            )

    return results


def _resolve_single_pdf_xbrl_validation(
    display_name: str,
    pdf_value: int | None,
    xbrl_value: int | None,
    tolerance: int,
    data: Sheet1Data,
    field_name: str,
    use_fallback: bool,
) -> ValidationResult | None:
    """Resolve a single PDF-XBRL validation based on available values."""
    if pdf_value is None and xbrl_value is None:
        return None

    # Both sources present - compare with tolerance
    if pdf_value is not None and xbrl_value is not None:
        match, diff = compare_with_tolerance(pdf_value, xbrl_value, tolerance)
        if match:
            logger.info("✓ %s matches XBRL: %s", display_name, f"{pdf_value:,}")
        else:
            logger.warning(
                "✗ %s mismatch - PDF: %s, XBRL: %s (diff: %s)",
                display_name,
                f"{pdf_value:,}",
                f"{xbrl_value:,}",
                diff,
            )
        return ValidationResult(
            field_name=display_name,
            value_a=pdf_value,
            value_b=xbrl_value,
            match=match,
            comparison_type="pdf_xbrl",
            source="both",
            difference=diff if not match else None,
        )

    # Single source - apply XBRL fallback if needed
    is_xbrl = xbrl_value is not None
    if is_xbrl and use_fallback:
        logger.info("Using XBRL value for %s: %s", display_name, f"{xbrl_value:,}")
        data.set_value(field_name, xbrl_value)

    return ValidationResult(
        field_name=display_name,
        value_a=None if is_xbrl else pdf_value,
        value_b=xbrl_value if is_xbrl else None,
        match=True,
        comparison_type="pdf_xbrl",
        source="xbrl_only" if is_xbrl else "pdf_only",
    )


def _run_pdf_xbrl_validations_impl(
    sheet1_data: Sheet1Data,
    xbrl_total_values: Mapping[str, int | None] | None,
    enable_fallback: bool = True,
) -> list[ValidationResult]:
    """Config-driven PDF ↔ XBRL comparison.

    Iterates through configured validations and compares PDF-extracted
    values against XBRL facts for each field.

    Parameters
    ----------
    sheet1_data
        Sheet1Data instance with PDF-extracted values.
    xbrl_total_values
        Mapping of XBRL fact keys to values.
    enable_fallback
        If True, use XBRL values when PDF extraction fails.

    Returns
    -------
    list
        List of ValidationResult for each configured validation.
    """
    comparison_results: list[ValidationResult] = []
    sum_tolerance = get_sheet1_sum_tolerance()

    for validation_spec in get_sheet1_pdf_xbrl_validations():
        field_id = validation_spec["field_name"]
        xbrl_fact_key = validation_spec["xbrl_key"]
        label = validation_spec["display_name"]
        fact_value = xbrl_total_values.get(xbrl_fact_key) if xbrl_total_values else None
        extracted_value = sheet1_data.get_value(field_id)

        validated = _resolve_single_pdf_xbrl_validation(
            label,
            extracted_value,
            fact_value,
            sum_tolerance,
            sheet1_data,
            field_id,
            enable_fallback,
        )
        if validated:
            comparison_results.append(validated)

    return comparison_results


def _run_cross_validations_impl(
    data: Sheet1Data,
    xbrl_totals: Mapping[str, int | None] | None,
) -> list[CrossValidationResult]:
    """Run config-driven cross-validations.

    Parameters
    ----------
    data
        Sheet1Data instance with extracted values.
    xbrl_totals
        Optional mapping of XBRL fact keys to values.

    Returns
    -------
    list
        List of CrossValidationResult for each configured validation.
    """
    results: list[CrossValidationResult] = []
    global_tolerance = get_sheet1_sum_tolerance()
    cross_validations = get_sheet1_cross_validations()
    field_to_result = get_sheet1_result_key_mapping()

    for rule in cross_validations:
        description = rule.get("description", "Unknown validation")
        formula = rule.get("formula", "")
        rule_tolerance = rule.get("tolerance", global_tolerance)

        values, missing = resolve_cross_validation_values(
            data, xbrl_totals, formula, field_to_result
        )

        if missing:
            result = CrossValidationResult(
                description=description,
                formula=formula,
                expected_value=None,
                calculated_value=None,
                match=True,
                difference=None,
                tolerance=rule_tolerance,
                missing_facts=missing,
            )
            results.append(result)
            logger.info("⚠ %s: skipped - missing %s", description, ", ".join(missing))
            continue

        expected, calculated, match, diff = evaluate_cross_validation(
            formula, values, rule_tolerance
        )

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

        if match:
            logger.info("✓ %s: %s", description, f"{expected:,}" if expected else "n/a")
        else:
            logger.warning(
                "✗ %s: expected=%s, calculated=%s (diff: %s)",
                description,
                expected,
                calculated,
                diff,
            )

    return results


def run_sheet1_validations(
    data: Sheet1Data,
    xbrl_totals: Mapping[str, int | None] | None = None,
    *,
    run_sum_validations: bool = True,
    run_pdf_xbrl_validations: bool = True,
    run_cross_validations: bool = True,
    use_xbrl_fallback: bool = True,
) -> ValidationReport:
    """Unified validation entry point for Sheet1 data.

    Parameters
    ----------
    data
        Sheet1Data instance with extracted values.
    xbrl_totals
        Optional mapping of XBRL fact keys to values.
    run_sum_validations
        Whether to run sum validations.
    run_pdf_xbrl_validations
        Whether to run PDF-XBRL comparisons.
    run_cross_validations
        Whether to run cross-validations.
    use_xbrl_fallback
        Whether to use XBRL values when PDF extraction fails.

    Returns
    -------
    ValidationReport
        Aggregated validation results.
    """
    sum_results: list[SumValidationResult] = []
    pdf_xbrl_results: list[ValidationResult] = []
    cross_results: list[CrossValidationResult] = []

    if run_sum_validations:
        sum_results = _run_sum_validations_impl(data)

    if run_pdf_xbrl_validations:
        pdf_xbrl_results = _run_pdf_xbrl_validations_impl(data, xbrl_totals, use_xbrl_fallback)

    if run_cross_validations:
        cross_results = _run_cross_validations_impl(data, xbrl_totals)

    return ValidationReport(
        sum_validations=sum_results,
        cross_validations=cross_results,
        pdf_xbrl_validations=pdf_xbrl_results,
        reference_issues=None,
    )

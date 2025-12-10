"""Validation report formatting utilities.

This module provides functions to format validation results for display
and logging. All functions are pure formatters with no side effects
beyond logging.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from puco_eeff.extractor.validation.types import (
    CrossValidationResult,
    SumValidationResult,
    ValidationReport,
    ValidationResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

__all__ = [
    "format_cross_validation_status",
    "format_sum_validation_status",
    "format_validation_report",
    "log_validation_report",
]


def format_sum_validation_status(result: SumValidationResult) -> str:
    """Format sum validation status comparing calculated to expected total.

    Parameters
    ----------
    result
        Sum validation outcome with calculated and expected totals.

    Returns
    -------
    str
        Status string with an icon and mismatch context when relevant.
    """
    if result.expected_total is None:
        return "⚠ No total value to compare"
    if result.match:
        return f"✓ Sum matches total ({result.calculated_sum:,})"
    return (
        f"✗ Sum mismatch: items={result.calculated_sum:,}, "
        f"total={result.expected_total:,} (diff: {result.difference})"
    )


def format_cross_validation_status(r: CrossValidationResult) -> str:
    """Format cross-validation status with formula check results."""
    if r.missing_facts:
        return f"⚠ Skipped - missing: {', '.join(r.missing_facts)}"
    if r.match:
        return f"✓ {r.description}: {r.expected_value:,}"
    return (
        f"✗ {r.description}: expected={r.expected_value}, "
        f"calculated={r.calculated_value} (diff: {r.difference})"
    )


def _format_section(
    header: str,
    items: list,
    formatter: Callable,
    empty_line: str,
) -> list[str]:
    """Format validation sections generically."""
    if not items:
        return [empty_line]
    lines = [header]
    lines.extend(formatter(item) for item in items)
    return lines


def _format_pdf_xbrl_validation_item(v: ValidationResult) -> str:
    """Format a single PDF-XBRL validation item."""
    if v.source == "both":
        symbol = "✓" if v.match else "✗"
        diff_display = f"{v.difference:,}" if v.difference is not None else "n/a"
        if v.match:
            return f"  {symbol} {v.field_name}: {v.value_a:,} (PDF) = {v.value_b:,} (XBRL)"
        return (
            f"  {symbol} {v.field_name}: {v.value_a:,} (PDF) ≠ "
            f"{v.value_b:,} (XBRL) [diff: {diff_display}]"
        )
    if v.source == "xbrl_only":
        return f"  ⚠ {v.field_name}: {v.value_b:,} (XBRL only, used as source)"
    return f"  ⚠ {v.field_name}: {v.value_a:,} (PDF only, no XBRL)"


def _format_reference_validations(report: ValidationReport) -> list[str]:
    """Format reference validations section."""
    lines: list[str] = []
    if report.reference_issues is None:
        lines.append("Reference Validation: (not run - use --validate-reference to enable)")
    elif len(report.reference_issues) == 0:
        lines.append("Reference Validation: ✓ All values match reference data")
    else:
        lines.append("Reference Validation: ✗ MISMATCHES FOUND")
        lines.extend(f"  • {issue}" for issue in report.reference_issues)
    return lines


def format_validation_report(report: ValidationReport, verbose: bool = True) -> str:
    """Format validation report for display.

    Parameters
    ----------
    report
        Aggregated validation results.
    verbose
        If True, include all details. Currently unused but kept for API compatibility.

    Returns
    -------
    str
        Formatted multi-line report string.
    """
    separator = "═" * 60
    lines = [separator, "                    VALIDATION REPORT", separator, ""]

    sections = [
        (
            "Sum Validations:",
            report.sum_validations,
            lambda v: f"  {format_sum_validation_status(v)}",
            "Sum Validations: (none configured)",
        ),
        (
            "PDF ↔ XBRL Validations:",
            report.pdf_xbrl_validations,
            _format_pdf_xbrl_validation_item,
            "PDF ↔ XBRL Validations: (no XBRL available)",
        ),
        (
            "Cross-Validations:",
            report.cross_validations,
            lambda v: f"  {format_cross_validation_status(v)}",
            "Cross-Validations: (none configured)",
        ),
    ]

    for header, items, formatter, empty_line in sections:
        lines.extend(_format_section(header, items, formatter, empty_line))
        lines.append("")

    lines.extend(_format_reference_validations(report))
    lines.extend(["", separator])

    return "\n".join(lines)


def log_validation_report(report: ValidationReport) -> None:
    """Log validation results with appropriate log levels.

    Parameters
    ----------
    report
        Aggregated validation results to log.
    """
    for v in report.sum_validations:
        if v.match:
            logger.info("✓ Sum: %s", v.description)
        else:
            logger.warning("✗ Sum: %s - mismatch (diff: %s)", v.description, v.difference)

    if report.reference_issues is not None and len(report.reference_issues) > 0:
        logger.warning("=" * 60)
        logger.warning("⚠️  REFERENCE DATA MISMATCH")
        logger.warning("=" * 60)
        for issue in report.reference_issues:
            logger.warning("  • %s", issue)
        logger.warning("=" * 60)

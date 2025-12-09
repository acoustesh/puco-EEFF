"""Validation dataclasses, formatting, and helper functions.

This module provides the core validation types and utilities used throughout
the extraction pipeline.

Key Classes:
    ValidationResult: Result of cross-validation between PDF and XBRL.
    ExtractionResult: Complete extraction result with optional validation.
    SumValidationResult: Result of line-item sum validation.
    CrossValidationResult: Result of cross-validation formula check.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from puco_eeff.config import setup_logging
from puco_eeff.sheets.sheet1 import (
    Sheet1Data,
    get_sheet1_cross_validations,
    get_sheet1_pdf_xbrl_validations,
    get_sheet1_result_key_mapping,
    get_sheet1_sum_tolerance,
    get_sheet1_total_validations,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from puco_eeff.extractor.extraction import SectionBreakdown

logger = setup_logging(__name__)

__all__ = [
    "CrossValidationResult",
    "ExtractionResult",
    "SumValidationResult",
    "ValidationReport",
    # Dataclasses
    "ValidationResult",
    "_compare_with_tolerance",
    "_run_cross_validations",
    "_run_pdf_xbrl_validations",
    # Internal helpers (needed by extraction_pipeline)
    "_run_sum_validations",
    # Formatting
    "format_sum_validation_status",
    "format_validation_report",
    "log_validation_report",
    # Validation runners
    "run_sheet1_validations",
]


# =============================================================================
# Validation Result Dataclasses
# =============================================================================


@dataclass
class PDFXBRLComparisonResult:
    """Result of cross-validation between PDF and XBRL data sources.

    This class tracks the comparison of a single field between two sources,
    recording which sources were available and whether values matched.

    Attributes:
        field_name: Name of the validated field (e.g., "total_costo_venta")
        pdf_value: Value extracted from PDF (None if unavailable)
        xbrl_value: Value from XBRL file (None if unavailable)
        match: True if both sources agree (or one missing)
        source: Origin indicator ("pdf_only", "xbrl_only", or "both")
        difference: Absolute difference when mismatch (None if match/missing)

    """

    field_name: str
    pdf_value: int | None
    xbrl_value: int | None
    match: bool
    source: str
    difference: int | None = None

    @property
    def status(self) -> str:
        """Format PDF/XBRL validation status with source-specific warnings."""
        # Use dictionary-based dispatch for source-specific messages
        messages = {
            "pdf_only": "⚠ PDF only (no XBRL)",
            "xbrl_only": "⚠ XBRL only (PDF extraction failed)",
        }
        if msg := messages.get(self.source):
            return msg
        # Standard match/mismatch formatting
        return (
            "✓ Match"
            if self.match
            else f"✗ Mismatch (diff: {self.difference:,})"
            if self.difference
            else "✗ Mismatch (diff: n/a)"
        )


# Backward-compatible alias
ValidationResult = PDFXBRLComparisonResult


@dataclass
class ExtractionResult:
    """Complete extraction result with optional validation."""

    year: int
    quarter: int
    sections: dict[str, SectionBreakdown] = field(default_factory=dict)
    xbrl_available: bool = False
    xbrl_totals: dict[str, int | None] = field(default_factory=dict)
    validations: list[ValidationResult] = field(default_factory=list)
    validation_report: ValidationReport | None = None
    source: str = "cmf"
    pdf_path: Path | None = None
    xbrl_path: Path | None = None

    def get_section(self, section_id: str) -> SectionBreakdown | None:
        """Get a section by its ID."""
        return self.sections.get(section_id)

    def is_valid(self) -> bool:
        """Check if all validations passed."""
        if not self.validations:
            return len(self.sections) > 0 and all(s is not None for s in self.sections.values())
        return all(v.match for v in self.validations)


@dataclass
class SumValidationResult:
    """Result of line-item sum validation."""

    description: str
    total_field: str
    expected_total: int | None
    calculated_sum: int
    match: bool
    difference: int
    tolerance: int


def format_sum_validation_status(result: SumValidationResult) -> str:
    """Format sum validation status comparing calculated to expected total.

    Args:
        result: The sum validation result to format

    Returns:
        Formatted status string with appropriate icon

    """
    if result.expected_total is None:
        return "⚠ No total value to compare"
    if result.match:
        return f"✓ Sum matches total ({result.calculated_sum:,})"
    return (
        f"✗ Sum mismatch: items={result.calculated_sum:,}, "
        f"total={result.expected_total:,} (diff: {result.difference})"
    )


@dataclass
class CrossValidationResult:
    """Result of cross-validation formula check."""

    description: str
    formula: str
    expected_value: int | None
    calculated_value: int | None
    match: bool
    difference: int | None
    tolerance: int
    missing_facts: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """Format cross-validation status with formula check results."""
        if self.missing_facts:
            missing_list = ", ".join(self.missing_facts)
            return f"⚠ Skipped - missing: {missing_list}"
        desc, exp, calc = self.description, self.expected_value, self.calculated_value
        return (
            f"✓ {desc}: {exp:,}"
            if self.match
            else f"✗ {desc}: expected={exp}, calculated={calc} (diff: {self.difference})"
        )


@dataclass
class ValidationReport:
    """Aggregated validation results for unified reporting."""

    sum_validations: list[SumValidationResult] = field(default_factory=list)
    cross_validations: list[CrossValidationResult] = field(default_factory=list)
    pdf_xbrl_validations: list[ValidationResult] = field(default_factory=list)
    reference_issues: list[str] | None = None

    def _has_validation_failures(self, validations: list) -> bool:
        """Check if any validation in the given list failed."""
        return any(not v.match for v in validations)

    def has_failures(self) -> bool:
        """Check if any validation failed."""
        return (
            self._has_validation_failures(self.sum_validations)
            or self._has_validation_failures(self.cross_validations)
            or self._has_validation_failures(self.pdf_xbrl_validations)
            or self.has_reference_failures()
        )

    def has_sum_failures(self) -> bool:
        """Check if any sum validation failed (convenience wrapper)."""
        # Delegates to has_failures logic to avoid duplication
        return self._has_validation_failures(self.sum_validations)

    def has_reference_failures(self) -> bool:
        """Check if reference validation failed."""
        return self.reference_issues is not None and len(self.reference_issues) > 0


# =============================================================================
# Validation Report Formatting
# =============================================================================


def _format_section(
    header: str,
    items: list,
    formatter: callable,
    empty_line: str,
) -> list[str]:
    """Generic formatter for validation sections."""
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
            return f"  {symbol} {v.field_name}: {v.pdf_value:,} (PDF) = {v.xbrl_value:,} (XBRL)"
        return f"  {symbol} {v.field_name}: {v.pdf_value:,} (PDF) ≠ {v.xbrl_value:,} (XBRL) [diff: {diff_display}]"
    if v.source == "xbrl_only":
        return f"  ⚠ {v.field_name}: {v.xbrl_value:,} (XBRL only, used as source)"
    return f"  ⚠ {v.field_name}: {v.pdf_value:,} (PDF only, no XBRL)"


def _format_reference_validations(report: ValidationReport) -> list[str]:
    """Format reference validations section."""
    lines = []
    if report.reference_issues is None:
        lines.append("Reference Validation: (not run - use --validate-reference to enable)")
    elif len(report.reference_issues) == 0:
        lines.append("Reference Validation: ✓ All values match reference data")
    else:
        lines.append("Reference Validation: ✗ MISMATCHES FOUND")
        lines.extend(f"  • {issue}" for issue in report.reference_issues)
    return lines


def format_validation_report(report: ValidationReport, verbose: bool = True) -> str:
    """Format validation report for display."""
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
            lambda v: f"  {v.status}",
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
    """Log validation results with appropriate log levels."""
    for v in report.sum_validations:
        if v.match:
            logger.info(f"✓ Sum: {v.description}")
        else:
            logger.warning(f"✗ Sum: {v.description} - mismatch (diff: {v.difference})")

    if report.reference_issues is not None and len(report.reference_issues) > 0:
        logger.warning("=" * 60)
        logger.warning("⚠️  REFERENCE DATA MISMATCH")
        logger.warning("=" * 60)
        for issue in report.reference_issues:
            logger.warning("  • %s", issue)
        logger.warning("=" * 60)


# =============================================================================
# Validation Helper Functions
# =============================================================================


def _compare_with_tolerance(a: int | None, b: int | None, tolerance: int) -> tuple[bool, int]:
    """Compare two values with tolerance, using absolute values."""
    if a is None or b is None:
        return True, 0
    diff = abs(abs(a) - abs(b))
    return diff <= tolerance, diff


def _run_sum_validations(data: Sheet1Data) -> list[SumValidationResult]:
    """Run config-driven sum validations on Sheet1Data."""
    results = []
    global_tolerance = get_sheet1_sum_tolerance()
    total_validations = get_sheet1_total_validations()

    for rule in total_validations:
        description = rule.get("description", "Unknown validation")
        total_field = rule.get("total_field", "")
        sum_fields = rule.get("sum_fields", [])
        rule_tolerance = rule.get("tolerance", global_tolerance)

        expected_total = data.get_value(total_field)
        calculated_sum = 0
        for fld in sum_fields:
            value = data.get_value(fld)
            if value is not None:
                calculated_sum += value

        match, difference = _compare_with_tolerance(expected_total, calculated_sum, rule_tolerance)
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

        if expected_total is None:
            logger.info(f"⚠ {description}: no total value to compare (sum={calculated_sum:,})")
        elif match:
            logger.info(f"✓ {description}: sum={calculated_sum:,} matches total={expected_total:,}")
        else:
            logger.warning(f"✗ {description}: sum={calculated_sum:,} != total={expected_total:,} (diff: {difference})")

    return results


def _create_both_sources_result(
    display_name: str,
    pdf_value: int,
    xbrl_value: int,
    tolerance: int,
) -> ValidationResult:
    """Create validation result when both PDF and XBRL values exist."""
    match, diff = _compare_with_tolerance(pdf_value, xbrl_value, tolerance)
    if match:
        logger.info(f"✓ {display_name} matches XBRL: {pdf_value:,}")
    else:
        logger.warning(
            f"✗ {display_name} mismatch - PDF: {pdf_value:,}, XBRL: {xbrl_value:,} (diff: {diff})",
        )
    return ValidationResult(
        field_name=display_name,
        pdf_value=pdf_value,
        xbrl_value=xbrl_value,
        match=match,
        source="both",
        difference=diff if not match else None,
    )


def _create_single_source_result(
    display_name: str,
    pdf_value: int | None,
    xbrl_value: int | None,
    data: Sheet1Data | None = None,
    field_name: str | None = None,
    use_fallback: bool = False,
) -> ValidationResult:
    """Create ValidationResult when only one source (PDF or XBRL) has a value."""
    is_xbrl_only = xbrl_value is not None and pdf_value is None
    if is_xbrl_only and use_fallback and data is not None and field_name is not None:
        logger.info(f"Using XBRL value for {display_name}: {xbrl_value:,}")
        data.set_value(field_name, xbrl_value)
    return ValidationResult(
        field_name=display_name,
        pdf_value=pdf_value if not is_xbrl_only else None,
        xbrl_value=xbrl_value if is_xbrl_only else None,
        match=True,
        source="xbrl_only" if is_xbrl_only else "pdf_only",
    )


def _run_pdf_xbrl_validations(
    sheet1_data: Sheet1Data,
    xbrl_total_values: Mapping[str, int | None] | None,
    enable_fallback: bool = True,
) -> list[ValidationResult]:
    """Config-driven PDF ↔ XBRL comparison.

    Iterates through configured validations and compares PDF-extracted
    values against XBRL facts for each field.
    """
    comparison_results = []
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
    if xbrl_value is not None and pdf_value is not None:
        return _create_both_sources_result(display_name, pdf_value, xbrl_value, tolerance)
    if xbrl_value is not None or pdf_value is not None:
        return _create_single_source_result(
            display_name,
            pdf_value,
            xbrl_value,
            data,
            field_name,
            use_fallback,
        )
    return None


def _run_cross_validations(
    data: Sheet1Data,
    xbrl_totals: Mapping[str, int | None] | None,
) -> list[CrossValidationResult]:
    """Run config-driven cross-validations."""
    results = []
    global_tolerance = get_sheet1_sum_tolerance()
    cross_validations = get_sheet1_cross_validations()

    for rule in cross_validations:
        description = rule.get("description", "Unknown validation")
        formula = rule.get("formula", "")
        rule_tolerance = rule.get("tolerance", global_tolerance)

        values, missing = _resolve_cross_validation_values(data, xbrl_totals, formula)

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
            logger.info(f"⚠ {description}: skipped - missing {', '.join(missing)}")
            continue

        expected, calculated, match, diff = _evaluate_cross_validation(formula, values, rule_tolerance)

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
            logger.info(f"✓ {description}: {expected:,}")
        else:
            logger.warning("✗ %s: expected=%s, calculated=%s (diff: %s)", description, expected, calculated, diff)

    return results


def _extract_formula_variables(formula: str) -> set[str]:
    """Extract variable names from a formula, excluding keywords."""
    var_pattern = re.compile(r"\b([a-z_]+)\b")
    var_names = set(var_pattern.findall(formula))
    keywords = {"abs", "and", "or", "not", "if", "else", "true", "false"}
    return var_names - keywords


def _lookup_xbrl_value(
    var: str,
    xbrl_totals: Mapping[str, int | None],
    xbrl_key_map: dict[str, str],
) -> int | None:
    """Look up a variable value from XBRL totals using key mapping."""
    for xbrl_key, mapped_name in xbrl_key_map.items():
        if var in {mapped_name, xbrl_key}:
            return xbrl_totals.get(xbrl_key)
    return xbrl_totals.get(var)


def _resolve_single_variable(
    var: str,
    data: Sheet1Data,
    xbrl_totals: Mapping[str, int | None] | None,
    xbrl_key_map: dict[str, str],
) -> int | None:
    """Resolve a single variable from data or XBRL totals."""
    value = data.get_value(var)
    if value is None and xbrl_totals:
        value = _lookup_xbrl_value(var, xbrl_totals, xbrl_key_map)
    return value


def _resolve_cross_validation_values(
    data: Sheet1Data,
    xbrl_totals: Mapping[str, int | None] | None,
    formula: str,
) -> tuple[dict[str, int], list[str]]:
    """Resolve values needed for a cross-validation formula."""
    field_to_result = get_sheet1_result_key_mapping()
    xbrl_key_map = {v: k for k, v in field_to_result.items()}
    var_names = _extract_formula_variables(formula)

    values = {}
    missing = []
    for var in var_names:
        value = _resolve_single_variable(var, data, xbrl_totals, xbrl_key_map)
        if value is not None:
            values[var] = value
        else:
            missing.append(var)

    return values, missing


def _evaluate_cross_validation(
    formula: str,
    values: dict[str, int],
    tolerance: int,
) -> tuple[int | None, int | None, bool, int | None]:
    """Safely evaluate a cross-validation formula."""
    if "==" not in formula:
        logger.warning("Unsupported formula format (no '=='): %s", formula)
        return None, None, True, None

    parts = formula.split("==")
    if len(parts) != 2:
        logger.warning("Unsupported formula format (multiple '=='): %s", formula)
        return None, None, True, None

    lhs_expr = parts[0].strip()
    rhs_expr = parts[1].strip()

    try:
        expected = _safe_eval_expression(lhs_expr, values)
        calculated = _safe_eval_expression(rhs_expr, values)

        if expected is None or calculated is None:
            return expected, calculated, True, None

        diff = abs(expected - calculated)
        match = diff <= tolerance
        return expected, calculated, match, diff
    except Exception as e:
        logger.warning("Error evaluating formula '%s': %s", formula, e)
        return None, None, True, None


def _eval_constant(node: ast.Constant, values: dict[str, int]) -> int | None:
    """Evaluate a constant node."""
    return node.value if isinstance(node.value, int) else None


def _eval_arithmetic_op(node: ast.UnaryOp | ast.BinOp, values: dict[str, int]) -> int | None:
    """Evaluate unary or binary arithmetic nodes using dispatch tables.

    Unary: +x, -x
    Binary: x+y, x-y, x*y
    """
    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast_node(node.operand, values)
        if operand is None:
            return None
        unary_ops = {ast.USub: lambda x: -x, ast.UAdd: lambda x: x}
        op_func = unary_ops.get(type(node.op))
        return op_func(operand) if op_func else None

    # BinOp case
    left = _eval_ast_node(node.left, values)
    right = _eval_ast_node(node.right, values)
    if left is None or right is None:
        return None
    binary_ops = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
    }
    op_func = binary_ops.get(type(node.op))
    return op_func(left, right) if op_func else None


def _eval_name(node: ast.Name, values: dict[str, int]) -> int | None:
    """Evaluate a variable name node."""
    return values.get(node.id)


def _eval_call(node: ast.Call, values: dict[str, int]) -> int | None:
    """Evaluate a function call node (only abs() supported)."""
    if not (isinstance(node.func, ast.Name) and node.func.id == "abs"):
        return None
    if len(node.args) != 1 or node.keywords:
        return None
    arg_val = _eval_ast_node(node.args[0], values)
    return abs(arg_val) if arg_val is not None else None


# Dispatch table for AST node evaluation
_AST_EVALUATORS: dict[type, callable] = {
    ast.Constant: _eval_constant,
    ast.UnaryOp: _eval_arithmetic_op,
    ast.BinOp: _eval_arithmetic_op,
    ast.Name: _eval_name,
    ast.Call: _eval_call,
}


def _eval_ast_node(node: ast.AST, values: dict[str, int]) -> int | None:
    """Recursively evaluate an AST node to an integer value.

    Supports: integer constants, unary +/-, variable names, binary +/-/*,
    and the abs() function call.
    """
    evaluator = _AST_EVALUATORS.get(type(node))
    return evaluator(node, values) if evaluator else None


def _safe_eval_expression(expr: str, values: dict[str, int]) -> int | None:
    """Safely evaluate a simple arithmetic expression using AST parsing."""
    expr = expr.strip()
    if not expr:
        return None

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        logger.debug("Syntax error parsing expression '%s': %s", expr, e)
        return None

    try:
        return _eval_ast_node(tree.body, values)
    except Exception:
        return None


# =============================================================================
# Unified Validation API
# =============================================================================


def run_sheet1_validations(
    data: Sheet1Data,
    xbrl_totals: Mapping[str, int | None] | None = None,
    *,
    run_sum_validations: bool = True,
    run_pdf_xbrl_validations: bool = True,
    run_cross_validations: bool = True,
    use_xbrl_fallback: bool = True,
) -> ValidationReport:
    """Unified validation entry point for Sheet1 data."""
    sum_results: list[SumValidationResult] = []
    pdf_xbrl_results: list[ValidationResult] = []
    cross_results: list[CrossValidationResult] = []

    if run_sum_validations:
        sum_results = _run_sum_validations(data)

    if run_pdf_xbrl_validations:
        pdf_xbrl_results = _run_pdf_xbrl_validations(data, xbrl_totals, use_xbrl_fallback)

    if run_cross_validations:
        cross_results = _run_cross_validations(data, xbrl_totals)

    return ValidationReport(
        sum_validations=sum_results,
        cross_validations=cross_results,
        pdf_xbrl_validations=pdf_xbrl_results,
        reference_issues=None,
    )

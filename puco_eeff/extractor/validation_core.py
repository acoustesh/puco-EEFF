"""Validation dataclasses, formatting, and helper functions.

This module provides the core validation types and utilities used throughout
the extraction pipeline.

Key Classes:
    ValidationResult: Result of cross-validation between PDF and XBRL.
    ExtractionResult: Complete extraction result with optional validation.
    SumValidationResult: Result of line-item sum validation.
    CrossValidationResult: Result of cross-validation formula check.
    ValidationReport: Aggregated validation results for unified reporting.

Key Functions:
    format_validation_report(): Format ValidationReport for display.
    log_validation_report(): Log validation results with appropriate levels.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from puco_eeff.config import setup_logging
from puco_eeff.extractor.extraction import SectionBreakdown
from puco_eeff.sheets.sheet1 import (
    Sheet1Data,
    get_sheet1_cross_validations,
    get_sheet1_pdf_xbrl_validations,
    get_sheet1_result_key_mapping,
    get_sheet1_section_total_mapping,
    get_sheet1_sum_tolerance,
    get_sheet1_total_validations,
)

logger = setup_logging(__name__)

# Public API exports
__all__ = [
    # Dataclasses
    "ValidationResult",
    "ExtractionResult",
    "SumValidationResult",
    "CrossValidationResult",
    "ValidationReport",
    # Formatting
    "format_validation_report",
    "log_validation_report",
    # Validation runners
    "run_sheet1_validations",
    "validate_extraction",
    # Internal helpers (needed by extraction_pipeline)
    "_run_sum_validations",
    "_run_pdf_xbrl_validations",
    "_run_cross_validations",
    "_compare_with_tolerance",
]


# =============================================================================
# Validation Result Dataclasses
# =============================================================================


@dataclass
class ValidationResult:
    """Result of cross-validation between PDF and XBRL."""

    field_name: str
    pdf_value: int | None
    xbrl_value: int | None
    match: bool
    source: str
    difference: int | None = None

    @property
    def status(self) -> str:
        """Return validation status string."""
        if self.source == "pdf_only":
            return "⚠ PDF only (no XBRL)"
        elif self.source == "xbrl_only":
            return "⚠ XBRL only (PDF extraction failed)"
        elif self.match:
            return "✓ Match"
        else:
            return f"✗ Mismatch (diff: {self.difference:,})"


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

    @property
    def status(self) -> str:
        """Return validation status string."""
        if self.expected_total is None:
            return "⚠ No total value to compare"
        elif self.match:
            return f"✓ Sum matches total ({self.calculated_sum:,})"
        else:
            return f"✗ Sum mismatch: items={self.calculated_sum:,}, total={self.expected_total:,} (diff: {self.difference})"


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
        """Return validation status string."""
        if self.missing_facts:
            return f"⚠ Skipped - missing: {', '.join(self.missing_facts)}"
        elif self.match:
            return f"✓ {self.description}: {self.expected_value:,}"
        else:
            return f"✗ {self.description}: expected={self.expected_value}, calculated={self.calculated_value} (diff: {self.difference})"


@dataclass
class ValidationReport:
    """Aggregated validation results for unified reporting."""

    sum_validations: list[SumValidationResult] = field(default_factory=list)
    cross_validations: list[CrossValidationResult] = field(default_factory=list)
    pdf_xbrl_validations: list[ValidationResult] = field(default_factory=list)
    reference_issues: list[str] | None = None

    def has_failures(self) -> bool:
        """Check if any validation failed."""
        sum_ok = all(v.match for v in self.sum_validations)
        cross_ok = all(v.match for v in self.cross_validations)
        pdf_xbrl_ok = all(v.match for v in self.pdf_xbrl_validations)
        ref_ok = self.reference_issues is None or len(self.reference_issues) == 0
        return not (sum_ok and cross_ok and pdf_xbrl_ok and ref_ok)

    def has_sum_failures(self) -> bool:
        """Check if any sum validation failed."""
        return any(not v.match for v in self.sum_validations)

    def has_reference_failures(self) -> bool:
        """Check if reference validation failed."""
        return self.reference_issues is not None and len(self.reference_issues) > 0


# =============================================================================
# Validation Report Formatting
# =============================================================================


def format_validation_report(report: ValidationReport, verbose: bool = True) -> str:
    """Format validation report for display."""
    lines = []
    separator = "═" * 60

    lines.append(separator)
    lines.append("                    VALIDATION REPORT")
    lines.append(separator)
    lines.append("")

    if report.sum_validations:
        lines.append("Sum Validations:")
        for v in report.sum_validations:
            lines.append(f"  {v.status}")
    else:
        lines.append("Sum Validations: (none configured)")
    lines.append("")

    if report.pdf_xbrl_validations:
        lines.append("PDF ↔ XBRL Validations:")
        for v in report.pdf_xbrl_validations:
            if v.source == "both":
                symbol = "✓" if v.match else "✗"
                if v.match:
                    lines.append(f"  {symbol} {v.field_name}: {v.pdf_value:,} (PDF) = {v.xbrl_value:,} (XBRL)")
                else:
                    lines.append(
                        f"  {symbol} {v.field_name}: {v.pdf_value:,} (PDF) ≠ {v.xbrl_value:,} (XBRL) [diff: {v.difference}]"
                    )
            elif v.source == "xbrl_only":
                lines.append(f"  ⚠ {v.field_name}: {v.xbrl_value:,} (XBRL only, used as source)")
            else:
                lines.append(f"  ⚠ {v.field_name}: {v.pdf_value:,} (PDF only, no XBRL)")
    else:
        lines.append("PDF ↔ XBRL Validations: (no XBRL available)")
    lines.append("")

    if report.cross_validations:
        lines.append("Cross-Validations:")
        for v in report.cross_validations:
            lines.append(f"  {v.status}")
    else:
        lines.append("Cross-Validations: (none configured)")
    lines.append("")

    if report.reference_issues is None:
        lines.append("Reference Validation: (not run - use --validate-reference to enable)")
    elif len(report.reference_issues) == 0:
        lines.append("Reference Validation: ✓ All values match reference data")
    else:
        lines.append("Reference Validation: ✗ MISMATCHES FOUND")
        for issue in report.reference_issues:
            lines.append(f"  • {issue}")

    lines.append("")
    lines.append(separator)
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
            logger.warning(f"  • {issue}")
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


def _run_pdf_xbrl_validations(
    data: Sheet1Data,
    xbrl_totals: dict[str, int | None] | None,
    use_fallback: bool = True,
) -> list[ValidationResult]:
    """Config-driven PDF ↔ XBRL comparison."""
    results = []
    tolerance = get_sheet1_sum_tolerance()
    pdf_xbrl_config = get_sheet1_pdf_xbrl_validations()

    for validation in pdf_xbrl_config:
        field_name = validation["field_name"]
        xbrl_key = validation["xbrl_key"]
        display_name = validation["display_name"]
        xbrl_value = xbrl_totals.get(xbrl_key) if xbrl_totals else None
        pdf_value = data.get_value(field_name)

        if xbrl_value is not None:
            if pdf_value is not None:
                match, diff = _compare_with_tolerance(pdf_value, xbrl_value, tolerance)
                results.append(
                    ValidationResult(
                        field_name=display_name,
                        pdf_value=pdf_value,
                        xbrl_value=xbrl_value,
                        match=match,
                        source="both",
                        difference=diff if not match else None,
                    )
                )
                if match:
                    logger.info(f"✓ {display_name} matches XBRL: {pdf_value:,}")
                else:
                    logger.warning(
                        f"✗ {display_name} mismatch - PDF: {pdf_value:,}, XBRL: {xbrl_value:,} (diff: {diff})"
                    )
            else:
                if use_fallback:
                    logger.info(f"Using XBRL value for {display_name}: {xbrl_value:,}")
                    data.set_value(field_name, xbrl_value)
                results.append(
                    ValidationResult(
                        field_name=display_name,
                        pdf_value=None,
                        xbrl_value=xbrl_value,
                        match=True,
                        source="xbrl_only",
                    )
                )
        elif pdf_value is not None:
            results.append(
                ValidationResult(
                    field_name=display_name,
                    pdf_value=pdf_value,
                    xbrl_value=None,
                    match=True,
                    source="pdf_only",
                )
            )

    return results


def _run_cross_validations(
    data: Sheet1Data,
    xbrl_totals: dict[str, int | None] | None,
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
            logger.warning(f"✗ {description}: expected={expected}, calculated={calculated} (diff: {diff})")

    return results


def _resolve_cross_validation_values(
    data: Sheet1Data,
    xbrl_totals: dict[str, int | None] | None,
    formula: str,
) -> tuple[dict[str, int], list[str]]:
    """Resolve values needed for a cross-validation formula."""
    field_to_result = get_sheet1_result_key_mapping()
    xbrl_key_map = {v: k for k, v in field_to_result.items()}

    var_pattern = re.compile(r"\b([a-z_]+)\b")
    var_names = set(var_pattern.findall(formula))
    keywords = {"abs", "and", "or", "not", "if", "else", "true", "false"}
    var_names = var_names - keywords

    values = {}
    missing = []

    for var in var_names:
        value = data.get_value(var)

        if value is None and xbrl_totals:
            for xbrl_key, mapped_name in xbrl_key_map.items():
                if var == mapped_name or var == xbrl_key:
                    value = xbrl_totals.get(xbrl_key)
                    break
            if value is None:
                value = xbrl_totals.get(var)

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
        logger.warning(f"Unsupported formula format (no '=='): {formula}")
        return None, None, True, None

    parts = formula.split("==")
    if len(parts) != 2:
        logger.warning(f"Unsupported formula format (multiple '=='): {formula}")
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
        logger.warning(f"Error evaluating formula '{formula}': {e}")
        return None, None, True, None


def _safe_eval_expression(expr: str, values: dict[str, int]) -> int | None:
    """Safely evaluate a simple arithmetic expression using AST parsing."""
    expr = expr.strip()
    if not expr:
        return None

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        logger.debug(f"Syntax error parsing expression '{expr}': {e}")
        return None

    def _eval_node(node: ast.AST) -> int | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value

        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                operand = _eval_node(node.operand)
                return -operand if operand is not None else None
            elif isinstance(node.op, ast.UAdd):
                return _eval_node(node.operand)
            else:
                return None

        if isinstance(node, ast.Name):
            var_name = node.id
            if var_name in values:
                return values[var_name]
            else:
                return None

        if isinstance(node, ast.BinOp):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.Mult):
                return left * right
            else:
                return None

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "abs":
                if len(node.args) == 1 and not node.keywords:
                    arg_val = _eval_node(node.args[0])
                    return abs(arg_val) if arg_val is not None else None
            return None

        return None

    try:
        return _eval_node(tree.body)
    except Exception:
        return None


# =============================================================================
# Unified Validation API
# =============================================================================


def run_sheet1_validations(
    data: Sheet1Data,
    xbrl_totals: dict[str, int | None] | None = None,
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


def validate_extraction(
    pdf_nota_21: SectionBreakdown | None,
    pdf_nota_22: SectionBreakdown | None,
    xbrl_totals: dict[str, int | None] | None,
) -> list[ValidationResult]:
    """Cross-validate PDF extraction against XBRL totals (deprecated)."""
    import warnings

    warnings.warn(
        "validate_extraction() is deprecated. Use run_sheet1_validations() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    data = _section_breakdowns_to_sheet1data(pdf_nota_21, pdf_nota_22)
    report = run_sheet1_validations(
        data,
        xbrl_totals,
        run_sum_validations=False,
        run_pdf_xbrl_validations=True,
        run_cross_validations=False,
        use_xbrl_fallback=False,
    )
    return report.pdf_xbrl_validations


def _section_breakdowns_to_sheet1data(
    nota_21: SectionBreakdown | None,
    nota_22: SectionBreakdown | None,
    year: int = 0,
    quarter: int = 0,
) -> Sheet1Data:
    """Convert SectionBreakdown objects to Sheet1Data (deprecated)."""
    import warnings

    from puco_eeff.extractor.extraction import format_quarter_label

    warnings.warn(
        "_section_breakdowns_to_sheet1data() is deprecated. Use sections_to_sheet1data() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    quarter_label = format_quarter_label(year, quarter)
    data = Sheet1Data(quarter=quarter_label, year=year, quarter_num=quarter)

    section_total_mapping = get_sheet1_section_total_mapping()

    if nota_21 and "nota_21" in section_total_mapping:
        field_name = section_total_mapping["nota_21"]
        data.set_value(field_name, nota_21.total_ytd_actual)

    if nota_22 and "nota_22" in section_total_mapping:
        field_name = section_total_mapping["nota_22"]
        data.set_value(field_name, nota_22.total_ytd_actual)

    return data

"""Formula evaluation utilities using AST parsing.

This module provides safe evaluation of arithmetic expressions used in
cross-validation formulas. All functions are pure and have no side effects.
"""

from __future__ import annotations

# =============================================================================
# Standard Library Imports
# =============================================================================
import ast
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from puco_eeff.sheets.sheet1 import Sheet1Data

logger = logging.getLogger(__name__)

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    "evaluate_cross_validation",
    "extract_formula_variables",
    "resolve_cross_validation_values",
    "safe_eval_expression",
]


# =============================================================================
# Formula Variable Extraction
# =============================================================================
def extract_formula_variables(formula: str) -> set[str]:
    """Extract variable names from a formula, excluding keywords."""
    # Match lowercase identifiers (variable names in formulas)
    var_pattern = re.compile(r"\b([a-z_]+)\b")
    var_names = set(var_pattern.findall(formula))
    # Filter out Python/formula keywords that aren't actual variables
    keywords = {"abs", "and", "or", "not", "if", "else", "true", "false"}
    return var_names - keywords


# =============================================================================
# Value Resolution Helpers
# =============================================================================
def _lookup_xbrl_value(
    var: str,
    xbrl_totals: Mapping[str, int | None],
    xbrl_key_map: dict[str, str],
) -> int | None:
    """Look up a variable value from XBRL totals using key mapping."""
    # Try mapped names first (field_name -> xbrl_key translation)
    for xbrl_key, mapped_name in xbrl_key_map.items():
        if var in {mapped_name, xbrl_key}:
            return xbrl_totals.get(xbrl_key)
    # Fall back to direct key lookup
    return xbrl_totals.get(var)


def _resolve_single_variable(
    var: str,
    data: Sheet1Data,
    xbrl_totals: Mapping[str, int | None] | None,
    xbrl_key_map: dict[str, str],
) -> int | None:
    """Resolve a single variable from data or XBRL totals."""
    # Try extracted PDF data first, then fall back to XBRL source
    value = data.get_value(var)
    if value is None and xbrl_totals:
        value = _lookup_xbrl_value(var, xbrl_totals, xbrl_key_map)
    return value


def resolve_cross_validation_values(
    data: Sheet1Data,
    xbrl_totals: Mapping[str, int | None] | None,
    formula: str,
    field_to_result_mapping: dict[str, str],
) -> tuple[dict[str, int], list[str]]:
    """Resolve values needed for a cross-validation formula.

    Parameters
    ----------
    data
        Sheet1Data instance with extracted values.
    xbrl_totals
        Optional mapping of XBRL fact keys to values.
    formula
        Formula string containing variable references.
    field_to_result_mapping
        Mapping from field names to result keys.

    Returns
    -------
    tuple
        A tuple of (resolved_values, missing_variables).
    """
    # Build reverse mapping: xbrl_key -> field_name for lookups
    xbrl_key_map = {v: k for k, v in field_to_result_mapping.items()}
    # Parse formula to find all variable references
    var_names = extract_formula_variables(formula)

    # Resolve each variable, tracking which ones couldn't be found
    values: dict[str, int] = {}
    missing: list[str] = []
    for var in var_names:
        value = _resolve_single_variable(var, data, xbrl_totals, xbrl_key_map)
        if value is not None:
            values[var] = value
        else:
            missing.append(var)

    return values, missing


# =============================================================================
# AST Evaluation
# =============================================================================


def _eval_arithmetic_op(node: ast.UnaryOp | ast.BinOp, values: dict[str, int]) -> int | None:
    """Evaluate unary or binary arithmetic nodes using dispatch tables."""
    # Handle unary operators: -x, +x
    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast_node(node.operand, values)
        if operand is None:
            return None
        # Dispatch table for unary operations
        unary_ops: dict[type, Callable[[int], int]] = {
            ast.USub: lambda x: -x,
            ast.UAdd: lambda x: x,
        }
        op_func = unary_ops.get(type(node.op))
        return op_func(operand) if op_func else None

    # Handle binary operators: a + b, a - b, a * b
    left = _eval_ast_node(node.left, values)
    right = _eval_ast_node(node.right, values)
    if left is None or right is None:
        return None
    # Dispatch table for binary operations (division excluded for safety)
    binary_ops: dict[type, Callable[[int, int], int]] = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
    }
    op_func = binary_ops.get(type(node.op))
    return op_func(left, right) if op_func else None


def _eval_call(node: ast.Call, values: dict[str, int]) -> int | None:
    """Evaluate a function call node (only abs() supported)."""
    # Security: only allow abs() function - reject all other calls
    if not (isinstance(node.func, ast.Name) and node.func.id == "abs"):
        return None
    # abs() must have exactly one positional argument, no keywords
    if len(node.args) != 1 or node.keywords:
        return None
    arg_val = _eval_ast_node(node.args[0], values)
    return abs(arg_val) if arg_val is not None else None


def _eval_constant(node: ast.Constant, _values: dict[str, int]) -> int | None:
    """Evaluate a constant node."""
    return node.value if isinstance(node.value, int) else None


def _eval_name(node: ast.Name, values: dict[str, int]) -> int | None:
    """Evaluate a name node (variable lookup)."""
    return values.get(node.id)


# Dispatch table for AST node evaluation
_AST_EVALUATORS: dict[type, Callable] = {
    ast.Constant: _eval_constant,
    ast.Name: _eval_name,
    ast.UnaryOp: _eval_arithmetic_op,
    ast.BinOp: _eval_arithmetic_op,
    ast.Call: _eval_call,
}


def _eval_ast_node(node: ast.AST, values: dict[str, int]) -> int | None:
    """Recursively evaluate an AST node to an integer value.

    Supports: integer constants, unary +/-, variable names, binary +/-/*,
    and the abs() function call.
    """
    evaluator = _AST_EVALUATORS.get(type(node))
    return evaluator(node, values) if evaluator else None


# =============================================================================
# Safe Expression Evaluation Entry Point
# =============================================================================
def safe_eval_expression(expr: str, values: dict[str, int]) -> int | None:
    """Safely evaluate a simple arithmetic expression using AST parsing.

    Parameters
    ----------
    expr
        Arithmetic expression string (e.g., "a + b * 2").
    values
        Mapping of variable names to integer values.

    Returns
    -------
    int | None
        The evaluated result, or None if evaluation fails.
    """
    expr = expr.strip()
    if not expr:
        return None

    # Parse expression into AST - safer than eval() as it doesn't execute
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
# Cross-Validation Formula Evaluation
# =============================================================================
def evaluate_cross_validation(
    formula: str,
    values: dict[str, int],
    tolerance: int,
) -> tuple[int | None, int | None, bool, int | None]:
    """Evaluate a cross-validation formula comparing LHS == RHS.

    Parameters
    ----------
    formula
        Formula string in format "lhs == rhs".
    values
        Mapping of variable names to integer values.
    tolerance
        Maximum allowed difference for match.

    Returns
    -------
    tuple
        (expected, calculated, match, difference) where expected is LHS value,
        calculated is RHS value, match indicates if difference <= tolerance.
    """
    # Formula must be an equality comparison: lhs == rhs
    if "==" not in formula:
        logger.warning("Unsupported formula format (no '=='): %s", formula)
        return None, None, True, None

    # Split and validate formula structure
    parts = formula.split("==")
    if len(parts) != 2:
        logger.warning("Unsupported formula format (multiple '=='): %s", formula)
        return None, None, True, None

    lhs_expr = parts[0].strip()
    rhs_expr = parts[1].strip()

    try:
        expected = safe_eval_expression(lhs_expr, values)
        calculated = safe_eval_expression(rhs_expr, values)

        if expected is None or calculated is None:
            return expected, calculated, True, None

        diff = abs(expected - calculated)
        match = diff <= tolerance
        return expected, calculated, match, diff
    except Exception as e:
        logger.warning("Error evaluating formula '%s': %s", formula, e)
        return None, None, True, None

"""Table parsing primitives for PDF extraction.

This module provides low-level functions for parsing table data from PDFs,
including number parsing, text normalization, and row/table parsing.

Key Functions:
    parse_chilean_number(): Parse Chilean-formatted numbers.
    normalize_for_matching(): Normalize text for fuzzy matching.
    match_item(): Match concept text against expected items.
    score_table_match(): Score how well a table matches expected content.
    parse_cost_table(): Parse a cost breakdown table.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from puco_eeff.config import setup_logging

logger = setup_logging(__name__)

# Public API exports
__all__ = [
    "count_value_offset",
    "extract_value_from_row",
    "find_label_index",
    "match_item",
    "normalize_for_matching",
    "parse_chilean_number",
    "parse_cost_table",
    "parse_multiline_row",
    "parse_single_row",
    "score_table_match",
]


# =============================================================================
# Number Parsing
# =============================================================================


def parse_chilean_number(value: str | None) -> int | None:
    """Parse a Chilean-formatted number.

    Chilean format uses:
    - Period as thousands separator: 30.294 = 30,294
    - Parentheses for negatives: (30.294) = -30,294
    """
    if not value:
        return None

    value = str(value).strip()
    is_negative = "(" in value and ")" in value
    value = re.sub(r"[^\d.\-]", "", value)

    if not value or value in {".", "-"}:
        return None

    try:
        value = value.replace(".", "")
        result = int(value)
        return -abs(result) if is_negative else result
    except ValueError:
        return None


# =============================================================================
# Text Normalization and Matching
# =============================================================================


def normalize_for_matching(text: str) -> str:
    """Normalize text for fuzzy matching."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[.,;:()\[\]]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def match_item(concept: str, expected_items: list[str]) -> str | None:
    """Match a concept text against expected items."""
    norm_concept = normalize_for_matching(concept)
    sorted_items = sorted(expected_items, key=len, reverse=True)

    for expected in sorted_items:
        norm_expected = normalize_for_matching(expected)

        if norm_expected in norm_concept:
            return expected

        expected_words = norm_expected.split()
        concept_words = norm_concept.split()
        matching_words = sum(1 for word in expected_words if any(word in cw for cw in concept_words))

        if expected_words and matching_words >= len(expected_words) * 0.8:
            return expected

    return None


# =============================================================================
# Table Matching and Scoring
# =============================================================================


def score_table_match(
    table: list[list[str | None]],
    expected_items: list[str],
    unique_items: list[str],
    exclude_items: list[str],
) -> int:
    """Score how well a table matches the expected content.

    Args:
        table: The extracted table data
        expected_items: Items that should be in the table
        unique_items: Items that strongly indicate the correct table (+5 each)
        exclude_items: Items that indicate the wrong table (-5 each)

    Returns:
        Match score (higher is better)

    """
    table_text = str(table).lower()
    score = sum(1 for item in expected_items if item.lower() in table_text)

    for unique_item in unique_items:
        if unique_item.lower() in table_text:
            score += 5
        normalized = normalize_for_matching(unique_item)
        if normalized in table_text and normalized != unique_item.lower():
            score += 5

    for exclude_item in exclude_items:
        if exclude_item.lower() in table_text:
            score -= 5
        normalized = normalize_for_matching(exclude_item)
        if normalized in table_text and normalized != exclude_item.lower():
            score -= 5

    return score


# =============================================================================
# Row Parsing
# =============================================================================


def _match_concept_or_total(concept: str, expected_items: list[str]) -> str | None:
    """Match concept text against expected items or recognize as 'Totales'."""
    matched = match_item(concept, expected_items)
    if matched:
        return matched
    return "Totales" if "total" in concept.lower() else None


def _extract_values_at_index(value_columns: list[list[str]], idx: int) -> list[int]:
    """Extract parsed numeric values at a specific index from all columns."""
    values = []
    for col_values in value_columns:
        if idx < len(col_values):
            parsed = parse_chilean_number(col_values[idx])
            if parsed is not None:
                values.append(parsed)
    return values


def parse_multiline_row(
    concepts: list[str],
    value_columns: list[list[str]],
    expected_items: list[str],
) -> list[dict[str, Any]]:
    """Parse a row with multiple concepts in a single cell (newline-separated)."""
    parsed_rows = []
    for idx, concept in enumerate(concepts):
        concept = concept.strip()
        if not concept:
            continue

        matched_item = _match_concept_or_total(concept, expected_items)
        if matched_item:
            values = _extract_values_at_index(value_columns, idx)
            parsed_rows.append({"concepto": matched_item, "values": values})

    return parsed_rows


def _extract_values_from_row_cells(row: list[str | None]) -> list[int]:
    """Extract parsed numeric values from row cells (skipping first cell)."""
    values = []
    for cell in row[1:]:
        if cell:
            cell_str = str(cell).split("\n")[0].strip()
            parsed = parse_chilean_number(cell_str)
            if parsed is not None:
                values.append(parsed)
    return values


def parse_single_row(
    row_text: str,
    row: list[str | None],
    expected_items: list[str],
) -> dict[str, Any] | None:
    """Parse a single-concept row."""
    matched_item = _match_concept_or_total(row_text, expected_items)
    if not matched_item:
        return None

    values = _extract_values_from_row_cells(row)

    return {"concepto": matched_item, "values": values}


def _build_value_columns(row: list[str | None]) -> list[list[str]]:
    """Build value columns from row cells (skipping first cell)."""
    value_columns: list[list[str]] = []
    for cell in row[1:]:
        if cell:
            values = str(cell).strip().split("\n")
            value_columns.append(values)
        else:
            value_columns.append([])
    return value_columns


def _parse_multiline_cell(
    concept_cell: str,
    row: list[str | None],
    expected_items: list[str],
) -> list[dict[str, Any]]:
    """Parse a multi-line concept cell."""
    concepts = concept_cell.split("\n")
    value_columns = _build_value_columns(row)
    return parse_multiline_row(concepts, value_columns, expected_items)


def parse_cost_table(table: list[list[str | None]], expected_items: list[str]) -> list[dict[str, Any]]:
    """Parse a cost breakdown table."""
    parsed_rows = []

    for row in table:
        if not row or not any(row):
            continue

        concept_cell = str(row[0] or "").strip()
        if not concept_cell:
            continue

        if "\n" in concept_cell:
            parsed_rows.extend(_parse_multiline_cell(concept_cell, row, expected_items))
        else:
            result = parse_single_row(concept_cell, row, expected_items)
            if result:
                parsed_rows.append(result)

    return parsed_rows


# =============================================================================
# Value Extraction Helpers
# =============================================================================


def find_label_index(labels: list[str], match_keywords: list[str]) -> int | None:
    """Find the index of a label that matches all keywords."""
    for i, label in enumerate(labels):
        if all(kw.lower() in label.lower() for kw in match_keywords):
            return i
    return None


def count_value_offset(labels: list[str], target_idx: int) -> int:
    """Count how many prior labels have trailing digits (value offset)."""
    offset = 0
    for i in range(target_idx):
        label = labels[i].strip()
        if label:
            parts = label.split()
            if parts and any(c.isdigit() for c in parts[-1]):
                offset += 1
    return offset


def _try_aligned_extraction(
    labels: list[str],
    values: list[str],
    match_keywords: list[str],
    min_threshold: int,
) -> int | None:
    """Try to extract value using aligned label position."""
    ingresos_idx = find_label_index(labels, match_keywords)
    if ingresos_idx is None:
        return None

    value_idx = count_value_offset(labels, ingresos_idx)
    if value_idx >= len(values):
        return None

    value = parse_chilean_number(values[value_idx].strip())
    return value if value is not None and value > min_threshold else None


def _try_fallback_extraction(values: list[str], min_threshold: int) -> int | None:
    """Try to extract any value above threshold as fallback."""
    for val_str in values:
        value = parse_chilean_number(val_str.strip())
        if value is not None and value > min_threshold:
            return value
    return None


def extract_value_from_row(row: list[Any], match_keywords: list[str], min_threshold: int) -> int | None:
    """Try to extract ingresos value from a table row."""
    first_col = str(row[0] or "").lower()
    if not all(kw.lower() in first_col for kw in match_keywords):
        return None

    labels = first_col.split("\n")
    values_col = str(row[1] or "") if len(row) > 1 else ""
    values = values_col.split("\n")

    aligned_value = _try_aligned_extraction(labels, values, match_keywords, min_threshold)
    if aligned_value is not None:
        return aligned_value

    return _try_fallback_extraction(values, min_threshold)

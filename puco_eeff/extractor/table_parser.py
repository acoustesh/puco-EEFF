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


def parse_multiline_row(
    concepts: list[str],
    value_columns: list[list[str]],
    expected_items: list[str],
) -> list[dict[str, Any]]:
    """Parse a row with multiple concepts in a single cell (newline-separated).

    Args:
        concepts: List of concept names from split cell
        value_columns: List of value lists, one per column
        expected_items: List of expected item names for matching

    Returns:
        List of parsed row dictionaries with concepto and values

    """
    parsed_rows = []
    for idx, concept in enumerate(concepts):
        concept = concept.strip()
        if not concept:
            continue

        matched_item = match_item(concept, expected_items)
        if not matched_item and "total" in concept.lower():
            matched_item = "Totales"

        if matched_item:
            values = []
            for col_values in value_columns:
                if idx < len(col_values):
                    parsed = parse_chilean_number(col_values[idx])
                    if parsed is not None:
                        values.append(parsed)
            parsed_rows.append({"concepto": matched_item, "values": values})

    return parsed_rows


def parse_single_row(
    row_text: str,
    row: list[str | None],
    expected_items: list[str],
) -> dict[str, Any] | None:
    """Parse a single-concept row.

    Args:
        row_text: The concept text from the first cell
        row: The full row including value cells
        expected_items: List of expected item names for matching

    Returns:
        Parsed row dictionary with concepto and values, or None if not matched

    """
    matched_item = match_item(row_text, expected_items)
    if not matched_item and "total" in row_text.lower():
        matched_item = "Totales"

    if not matched_item:
        return None

    values = []
    for cell in row[1:]:
        if cell:
            cell_str = str(cell).split("\n")[0].strip()
            parsed = parse_chilean_number(cell_str)
            if parsed is not None:
                values.append(parsed)

    return {"concepto": matched_item, "values": values}


def parse_cost_table(table: list[list[str | None]], expected_items: list[str]) -> list[dict[str, Any]]:
    """Parse a cost breakdown table."""
    parsed_rows = []

    for row in table:
        if not row or not any(row):
            continue

        concept_cell = str(row[0] or "").strip()

        if "\n" in concept_cell:
            # Multi-line cell: split and process each concept
            concepts = concept_cell.split("\n")
            value_columns: list[list[str]] = []
            for cell in row[1:]:
                if cell:
                    values = str(cell).strip().split("\n")
                    value_columns.append(values)
                else:
                    value_columns.append([])

            parsed_rows.extend(parse_multiline_row(concepts, value_columns, expected_items))
        else:
            # Single concept row
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


def extract_value_from_row(row: list[Any], match_keywords: list[str], min_threshold: int) -> int | None:
    """Try to extract ingresos value from a table row."""
    first_col = str(row[0] or "").lower()
    if not all(kw.lower() in first_col for kw in match_keywords):
        return None

    labels = first_col.split("\n")
    values_col = str(row[1] or "") if len(row) > 1 else ""
    values = values_col.split("\n")

    # Try aligned extraction based on label position
    ingresos_idx = find_label_index(labels, match_keywords)
    if ingresos_idx is not None:
        value_idx = count_value_offset(labels, ingresos_idx)
        if value_idx < len(values):
            value = parse_chilean_number(values[value_idx].strip())
            if value is not None and value > min_threshold:
                return value

    # Fallback: try any value above threshold
    for val_str in values:
        value = parse_chilean_number(val_str.strip())
        if value is not None and value > min_threshold:
            return value

    return None

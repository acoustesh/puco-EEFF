"""Low-level PDF table parsing helpers.

This module normalizes concept labels, parses localized numbers, and scores PDF
tables to locate cost breakdowns for Sheet1 extraction.
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
    """Parse a Chilean-formatted numeric string.

    Notes
    -----
    Thousands separators are periods (``30.294`` → ``30294``). Parentheses
    denote negative values (``(30.294)`` → ``-30294``).

    Parameters
    ----------
    value : str | None
        Raw string value to parse.

    Returns
    -------
    int | None
        Parsed integer or ``None`` when parsing fails.
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
    """Strip accents, punctuation, and extra spaces for fuzzy matching."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[.,;:()\[\]]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def match_item(concept: str, expected_items: list[str]) -> str | None:
    """Return the best-matching expected label for a concept string."""
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
    """Score how well a table aligns with expected content.

    Parameters
    ----------
    table : list[list[str | None]]
        Extracted table grid.
    expected_items : list[str]
        Concepts that should appear somewhere in the table.
    unique_items : list[str]
        High-signal items worth +5 points each.
    exclude_items : list[str]
        Disqualifying items worth -5 points each.

    Returns
    -------
    int
        Match score (higher indicates closer alignment).
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
    """Match a concept against expected items or return ``"Totales"`` for totals."""
    matched = match_item(concept, expected_items)
    if matched:
        return matched
    return "Totales" if "total" in concept.lower() else None


def _extract_values_at_index(value_columns: list[list[str]], idx: int) -> list[int]:
    """Extract parsed numeric values across value columns at a given index."""
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
    """Parse a row whose first cell contains multiple newline-separated concepts."""
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
    """Parse numeric values from a row, skipping the concept cell."""
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
    """Parse a row that contains a single concept label."""
    matched_item = _match_concept_or_total(row_text, expected_items)
    if not matched_item:
        return None

    values = _extract_values_from_row_cells(row)

    return {"concepto": matched_item, "values": values}


def _build_value_columns(row: list[str | None]) -> list[list[str]]:
    """Split value cells into columnar lists for multi-line parsing."""
    value_columns: list[list[str]] = []
    for cell in row[1:]:
        if cell:
            values = str(cell).strip().split("\n")
            value_columns.append(values)
        else:
            value_columns.append([])
    return value_columns


def parse_cost_table(table: list[list[str | None]], expected_items: list[str]) -> list[dict[str, Any]]:
    """Convert a cost table into a list of concept/value dictionaries."""
    parsed_rows = []

    for row in table:
        if not row or not any(row):
            continue

        concept_cell = str(row[0] or "").strip()
        if not concept_cell:
            continue

        if "\n" in concept_cell:
            # Multi-line concept cell: split and parse each concept
            concepts = concept_cell.split("\n")
            value_columns = _build_value_columns(row)
            parsed_rows.extend(parse_multiline_row(concepts, value_columns, expected_items))
        else:
            result = parse_single_row(concept_cell, row, expected_items)
            if result:
                parsed_rows.append(result)

    return parsed_rows


# =============================================================================
# Value Extraction Helpers
# =============================================================================


def find_label_index(labels: list[str], match_keywords: list[str]) -> int | None:
    """Return the index of the first label containing all keywords."""
    for i, label in enumerate(labels):
        if all(kw.lower() in label.lower() for kw in match_keywords):
            return i
    return None


def count_value_offset(labels: list[str], target_idx: int) -> int:
    """Count how many prior labels likely consume value slots (digit-ending labels)."""
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
    """Extract value aligned to a matched ingresos label if above threshold."""
    ingresos_idx = find_label_index(labels, match_keywords)
    if ingresos_idx is None:
        return None

    value_idx = count_value_offset(labels, ingresos_idx)
    if value_idx >= len(values):
        return None

    value = parse_chilean_number(values[value_idx].strip())
    return value if value is not None and value > min_threshold else None


def _try_fallback_extraction(values: list[str], min_threshold: int) -> int | None:
    """Fallback: return the first value above the threshold if parsing succeeds."""
    for val_str in values:
        value = parse_chilean_number(val_str.strip())
        if value is not None and value > min_threshold:
            return value
    return None


def extract_value_from_row(row: list[Any], match_keywords: list[str], min_threshold: int) -> int | None:
    """Extract ingresos value from a table row when keywords match the first column.

    Parameters
    ----------
    row : list[Any]
        Table row cells.
    match_keywords : list[str]
        Keywords that must all appear in the first column.
    min_threshold : int
        Minimum acceptable value to avoid spurious matches.

    Returns
    -------
    int | None
        Parsed value if found; otherwise ``None``.
    """
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

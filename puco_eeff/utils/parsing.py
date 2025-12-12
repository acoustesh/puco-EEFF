"""Shared parsing utilities for Spanish-locale numbers and PDF text.

This module provides common functions used across Sheet2 and Sheet3 extraction.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def parse_spanish_number(value_str: str) -> int | float | None:
    """Parse a number string using Spanish locale conventions.

    Spanish locale uses:
    - Comma (,) as decimal separator
    - Period (.) as thousands separator

    Examples
    --------
    - "19,5" -> 19.5
    - "1.342" -> 1342
    - "64.057" -> 64057
    - "3,97" -> 3.97
    - "-62.982" -> -62982

    Parameters
    ----------
    value_str
        String representation of number in Spanish locale.

    Returns
    -------
    int | float | None
        Parsed numeric value, or None if parsing fails.
    """
    if not value_str or value_str.strip() == "":
        return None

    # Clean the string
    cleaned = value_str.strip()

    # Check for negative sign and preserve it
    is_negative = cleaned.startswith("-")
    if is_negative:
        cleaned = cleaned[1:]

    # Remove any currency symbols or extra characters (but not digits, dots, commas)
    cleaned = re.sub(r"[^\d.,]", "", cleaned)

    if not cleaned:
        return None

    try:
        result: int | float
        # Check if it has a comma (decimal separator in Spanish)
        if "," in cleaned:
            # Replace thousands separator (.) with nothing
            # Then replace decimal separator (,) with period
            cleaned = cleaned.replace(".", "").replace(",", ".")
            result = float(cleaned)
        elif "." in cleaned:
            # Could be thousands separator or decimal
            # If there are multiple dots, they're thousands separators
            dot_count = cleaned.count(".")
            if dot_count > 1:
                # Multiple dots = thousands separators
                cleaned = cleaned.replace(".", "")
                result = int(cleaned)
            else:
                # Single dot - check position for disambiguation
                # If exactly 3 digits after dot, it's likely thousands separator
                parts = cleaned.split(".")
                if len(parts) == 2 and len(parts[1]) == 3:
                    # Thousands separator (e.g., "64.057" = 64057)
                    cleaned = cleaned.replace(".", "")
                    result = int(cleaned)
                else:
                    # Decimal separator (e.g., "3.97" = 3.97)
                    result = float(cleaned)
        else:
            # No separators - just a plain integer
            result = int(cleaned)

        return -result if is_negative else result
    except ValueError:
        logger.warning("Could not parse number: %s", value_str)
        return None


def normalize_pdf_line(line: str) -> str:
    """Fix OCR artifacts in PDF line - merge split numbers and handle parentheses.

    Examples
    --------
    - '6 5.483' -> '65.483' (split integer)
    - '3 8,4' -> '38,4' (split decimal)
    - '2 ,59' -> '2,59' (space before comma)
    - '( 62.982)' -> '-62.982' (parentheses for negative)

    Parameters
    ----------
    line
        Raw line from PDF text.

    Returns
    -------
    str
        Normalized line with merged numbers and converted negatives.
    """
    result = line

    # Convert parentheses notation for negative numbers: ( 62.982) -> -62.982
    # Handle both with and without internal spaces
    result = re.sub(r"\(\s*([\d.,]+)\s*\)", r"-\1", result)

    # Fix space before comma/dot: '2 ,59' -> '2,59'
    result = re.sub(r"(\d)\s+([,.])(\d)", r"\1\2\3", result)
    # Fix single digit space digits at word boundary: '6 5.483' -> '65.483'
    # Also handle '1 7.785' -> '17.785'
    result = re.sub(r"(?<=\s)(\d)\s+(\d)", r"\1\2", result)

    return result


def get_numbers_from_line(line: str) -> list[str]:
    """Extract number tokens from a line, including negative numbers.

    Parameters
    ----------
    line
        PDF line (should be normalized first).

    Returns
    -------
    list[str]
        Number strings found in the line.
    """
    # Match numbers with optional leading minus sign
    pattern = r"-?[\d.,]+"
    matches = re.findall(pattern, line)
    # Filter out items that are just punctuation
    return [m for m in matches if re.search(r"\d", m)]


def get_current_value(line: str, column_index: int = 0) -> int | float | None:
    """Extract current period value from a PDF line.

    PDF lines have format: Label [Note#] Value1 Value2 [Value3 Value4]
    where Value1 is current period accumulated.

    For 2-column PDFs: [current_accumulated, prior_accumulated]
    For 4-column PDFs: [current_accumulated, prior_accum, current_quarter, prior_quarter]

    Parameters
    ----------
    line
        PDF line with data.
    column_index
        Which value column to extract (0=first/current accumulated, default).

    Returns
    -------
    int | float | None
        Current period value.
    """
    line = normalize_pdf_line(line)
    nums = get_numbers_from_line(line)

    if not nums:
        return None

    # Filter out note references (small integers at the beginning, typically 1-99)
    # Note references are like "18", "19", "21", "22" etc.
    # Real values are larger or have decimals
    value_nums = []
    for num in nums:
        parsed = parse_spanish_number(num)
        if parsed is None:
            continue
        # Skip small positive integers that are likely note references
        # (typically appear at start of line and are < 100)
        if len(value_nums) == 0 and isinstance(parsed, int) and 0 < parsed < 100:
            continue
        value_nums.append(parsed)

    if not value_nums:
        return None

    # Return the requested column (default: first = current accumulated)
    if column_index < len(value_nums):
        return value_nums[column_index]

    # Fallback: return first value
    return value_nums[0]

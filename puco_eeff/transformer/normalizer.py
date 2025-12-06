"""Data normalization for financial statements."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

import pandas as pd

from puco_eeff.config import setup_logging

logger = setup_logging(__name__)


def normalize_financial_data(
    data: dict[str, Any] | pd.DataFrame,
    date_columns: list[str] | None = None,
    numeric_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Normalize financial data for consistency.

    Args:
        data: Raw data as dictionary or DataFrame
        date_columns: Columns to parse as dates
        numeric_columns: Columns to parse as numbers

    Returns:
        Normalized DataFrame
    """
    logger.info("Normalizing financial data")

    # Convert to DataFrame if needed
    df = pd.DataFrame(data) if isinstance(data, dict) else data.copy()

    # Normalize column names
    df.columns = [_normalize_column_name(col) for col in df.columns]

    # Process date columns
    if date_columns:
        for col in date_columns:
            norm_col = _normalize_column_name(col)
            if norm_col in df.columns:
                df[norm_col] = pd.to_datetime(df[norm_col], errors="coerce")

    # Process numeric columns
    if numeric_columns:
        for col in numeric_columns:
            norm_col = _normalize_column_name(col)
            if norm_col in df.columns:
                df[norm_col] = df[norm_col].apply(_parse_number)

    logger.info(f"Normalized data: {len(df)} rows, {len(df.columns)} columns")
    return df


def _normalize_column_name(name: str) -> str:
    """Normalize a column name to snake_case.

    Args:
        name: Original column name

    Returns:
        Normalized column name
    """
    # Convert to string and strip
    name = str(name).strip()

    # Replace spaces and special chars with underscore
    name = re.sub(r"[\s\-\.]+", "_", name)

    # Remove non-alphanumeric (except underscore)
    name = re.sub(r"[^\w]", "", name)

    # Convert to lowercase
    name = name.lower()

    # Remove multiple underscores
    name = re.sub(r"_+", "_", name)

    # Remove leading/trailing underscores
    name = name.strip("_")

    return name


def _parse_number(value: Any) -> float | None:
    """Parse a value as a number, handling various formats.

    Handles:
    - Thousands separators (. or ,)
    - Decimal separators (. or ,)
    - Parentheses for negative numbers
    - Currency symbols

    Args:
        value: Value to parse

    Returns:
        Parsed float or None if unparseable
    """
    if value is None or pd.isna(value):
        return None

    # Convert to string
    text = str(value).strip()

    if not text or text == "-":
        return None

    # Check for negative (parentheses notation)
    is_negative = False
    if text.startswith("(") and text.endswith(")"):
        is_negative = True
        text = text[1:-1]

    # Remove currency symbols and whitespace
    text = re.sub(r"[$€£¥CLP\s]", "", text)

    # Handle different number formats
    # Chilean format: 1.234.567,89 (dot for thousands, comma for decimal)
    # US format: 1,234,567.89 (comma for thousands, dot for decimal)

    # Count separators to determine format
    dots = text.count(".")
    commas = text.count(",")

    try:
        if dots > 1 or (dots == 1 and commas == 1 and text.index(".") < text.index(",")):
            # Chilean format: remove thousand separators, replace comma with dot
            text = text.replace(".", "").replace(",", ".")
        elif commas > 1 or (commas == 1 and dots == 1 and text.index(",") < text.index(".")):
            # US format: just remove thousand separators
            text = text.replace(",", "")
        elif commas == 1 and dots == 0:
            # Single comma - could be Chilean decimal
            text = text.replace(",", ".")
        # Single dot is already correct

        result = float(Decimal(text))
        return -result if is_negative else result

    except (InvalidOperation, ValueError):
        logger.debug(f"Could not parse number: {value}")
        return None


def clean_text(text: str) -> str:
    """Clean extracted text for processing.

    Args:
        text: Raw text

    Returns:
        Cleaned text
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def extract_tables_from_ocr_markdown(ocr_content: str) -> list[pd.DataFrame]:
    """Extract tables from OCR output in markdown format.

    Args:
        ocr_content: OCR-extracted content with markdown tables

    Returns:
        List of DataFrames extracted from markdown tables
    """
    tables: list[pd.DataFrame] = []

    # Split by lines
    lines = ocr_content.split("\n")

    current_table: list[list[str]] = []
    in_table = False

    for line in lines:
        line = line.strip()

        # Check if line looks like a table row
        if line.startswith("|") and line.endswith("|"):
            # Skip separator lines (|---|---|)
            if re.match(r"^\|[\s\-:]+\|$", line.replace("|", "||")):
                continue

            # Parse table row
            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            current_table.append(cells)
            in_table = True
        else:
            # End of table
            if in_table and current_table:
                if len(current_table) > 1:
                    # First row is header
                    df = pd.DataFrame(current_table[1:], columns=current_table[0])
                    tables.append(df)
                current_table = []
            in_table = False

    # Handle table at end of content
    if current_table and len(current_table) > 1:
        df = pd.DataFrame(current_table[1:], columns=current_table[0])
        tables.append(df)

    logger.info(f"Extracted {len(tables)} tables from OCR markdown")
    return tables

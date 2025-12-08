"""Workbook combiner - combines multiple quarters into a single Excel file.

Supports two modes:
1. Full generation: Build workbook from all available sheet1_YYYY_QX.json files
2. Incremental append: Add new quarter column to existing workbook

Output format for Sheet1:
| Row Label | 2024_QI | 2024_QII | 2024_QIII | 2024_QIV | 2025_QI | ...
|-----------|---------|----------|-----------|----------|---------|
| Ingresos  | 50000   | 179165   | ...       | ...      | ...     |
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from puco_eeff.config import DATA_DIR, get_config, setup_logging
from puco_eeff.writer.sheet_writer import (
    format_period,
    list_available_sheets,
    load_sheet_json,
    parse_period,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = setup_logging(__name__)


def combine_sheet1_quarters(
    year: int,
    output_dir: Path | None = None,
    input_dir: Path | None = None,
    append_to_existing: bool = True,
) -> Path:
    """Combine multiple quarters of Sheet1 data into a single Excel workbook.

    Horizontally stacks quarters as columns:
    | Row Label | 2024_QI | 2024_QII | 2024_QIII | ...

    Supports incremental append - if workbook exists and append_to_existing=True,
    adds new quarter columns without regenerating existing ones.

    Args:
        year: Year to combine (e.g., 2024)
        output_dir: Directory for output Excel file (defaults to DATA_DIR/output)
        input_dir: Directory with sheet JSON files (defaults to DATA_DIR/processed)
        append_to_existing: If True, append new quarters to existing workbook

    Returns:
        Path to the created/updated Excel workbook

    """
    save_dir = output_dir if output_dir is not None else DATA_DIR / "output"
    save_dir.mkdir(parents=True, exist_ok=True)

    load_dir = input_dir if input_dir is not None else DATA_DIR / "processed"

    # Get config for row labels
    config = get_config()
    row_mapping = config["sheets"]["sheet1"]["row_mapping"]

    # Find available quarters for this year
    available = list_available_sheets(sheet_name="sheet1", year=year, input_dir=load_dir)
    if not available:
        msg = f"No sheet1 data found for year {year}"
        raise ValueError(msg)

    logger.info(f"Found {len(available)} quarters for {year}: {[a['period'] for a in available]}")

    # Output file path
    filename = f"EEFF_{year}.xlsx"
    filepath = save_dir / filename

    # Load existing workbook data if appending
    existing_data: dict[str, dict[str, Any]] = {}
    existing_quarters: set[str] = set()

    if append_to_existing and filepath.exists():
        existing_data, existing_quarters = _load_existing_sheet1(filepath)
        logger.info(f"Loaded existing workbook with quarters: {sorted(existing_quarters)}")

    # Build combined DataFrame
    # Row structure from config
    row_labels = []
    row_fields = []
    for row_num in sorted(row_mapping.keys(), key=int):
        row_info = row_mapping[row_num]
        label = row_info.get("label", "")
        field = row_info.get("field")
        row_labels.append(label)
        row_fields.append(field)

    # Start with row labels column
    combined_df = pd.DataFrame({"Row": row_labels})

    # Add existing quarter columns first (preserve order)
    for period in sorted(existing_quarters):
        if period in existing_data:
            combined_df[period] = pd.Series(
                [existing_data[period].get(field) for field in row_fields],
                dtype="Int64",
            )

    # Add new quarter columns from JSON files
    for sheet_info in available:
        period = sheet_info["period"]
        if period in existing_quarters:
            logger.debug(f"Skipping {period} - already in workbook")
            continue

        # Load the JSON file
        json_data = load_sheet_json(sheet_info["filepath"])
        content = json_data.get("content", {})

        # Map field names to values
        values = []
        for field in row_fields:
            if field is None:
                values.append(None)
            else:
                values.append(content.get(field))

        combined_df[period] = pd.Series(values, dtype="Int64")
        logger.info(f"Added quarter: {period}")

    # Ensure columns are in chronological order (Row first, then sorted periods)
    period_cols = [c for c in combined_df.columns if c != "Row"]
    period_cols_sorted = sorted(period_cols, key=_period_sort_key)
    combined_df = combined_df[["Row", *period_cols_sorted]]

    # Write to Excel
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        combined_df.to_excel(writer, sheet_name="Ingresos y Costos", index=False)
        logger.debug(f"Wrote Sheet1 with {len(period_cols_sorted)} quarter columns")

    logger.info(f"Workbook saved: {filepath}")
    return filepath


def _period_sort_key(period: str) -> tuple[int, int]:
    """Sort key for period strings (e.g., '2024_QII' -> (2024, 2))."""
    try:
        year, quarter = parse_period(period)
        return (year, quarter)
    except ValueError:
        return (9999, 9)  # Put invalid periods at the end


def _load_existing_sheet1(filepath: Path) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Load existing Sheet1 data from workbook.

    Args:
        filepath: Path to existing Excel workbook

    Returns:
        Tuple of (data dict mapping period -> field values, set of existing periods)

    """
    try:
        df = pd.read_excel(filepath, sheet_name="Ingresos y Costos")
    except Exception as e:
        logger.warning(f"Could not load existing Sheet1: {e}")
        return {}, set()

    # Get config for field mapping
    config = get_config()
    row_mapping = config["sheets"]["sheet1"]["row_mapping"]

    # Build field list in row order
    row_fields = []
    for row_num in sorted(row_mapping.keys(), key=int):
        row_info = row_mapping[row_num]
        row_fields.append(row_info.get("field"))

    # Extract data for each period column
    existing_data: dict[str, dict[str, Any]] = {}
    existing_quarters: set[str] = set()

    for col in df.columns:
        if col == "Row":
            continue

        # This should be a period column like "2024_QII"
        try:
            parse_period(col)  # Validate it's a valid period
            existing_quarters.add(col)

            # Map values to field names
            existing_data[col] = {}
            for i, field in enumerate(row_fields):
                if field is not None and i < len(df):
                    value = df.iloc[i][col] if col in df.columns else None
                    existing_data[col][field] = value

        except ValueError:
            # Not a valid period column, skip
            continue

    return existing_data, existing_quarters


def append_quarter_to_workbook(
    year: int,
    quarter: int,
    output_dir: Path | None = None,
    input_dir: Path | None = None,
) -> Path:
    """Append a single quarter to an existing workbook (or create new).

    Convenience function that calls combine_sheet1_quarters with append mode.

    Args:
        year: Year (e.g., 2024)
        quarter: Quarter to append (1-4)
        output_dir: Directory for output Excel file
        input_dir: Directory with sheet JSON files

    Returns:
        Path to the updated workbook

    """
    period = format_period(year, quarter)
    logger.info(f"Appending {period} to workbook")

    return combine_sheet1_quarters(
        year=year,
        output_dir=output_dir,
        input_dir=input_dir,
        append_to_existing=True,
    )


def _format_sheet_name(name: str) -> str:
    """Format a sheet name for display.

    Args:
        name: Raw sheet name (e.g., "balance_general")

    Returns:
        Formatted name (e.g., "Balance General")

    """
    # Replace underscores with spaces and title case
    formatted = name.replace("_", " ").title()

    # Handle some common abbreviations
    replacements = {
        "Eeff": "EEFF",
        "Ifrs": "IFRS",
    }

    for old, new in replacements.items():
        formatted = formatted.replace(old, new)

    return formatted


def create_workbook_from_dataframes(
    sheets: dict[str, pd.DataFrame],
    output_path: Path,
) -> Path:
    """Create an Excel workbook directly from DataFrames.

    Args:
        sheets: Dictionary mapping sheet names to DataFrames
        output_path: Path for the output Excel file

    Returns:
        Path to the created workbook

    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating workbook with {len(sheets)} sheets: {output_path}")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            # Limit sheet name to 31 characters
            display_name = sheet_name[:31] if len(sheet_name) > 31 else sheet_name

            df.to_excel(writer, sheet_name=display_name, index=False)
            logger.debug(f"Added sheet: {display_name} ({len(df)} rows)")

    logger.info(f"Workbook created: {output_path}")
    return output_path


def list_workbook_quarters(
    year: int,
    output_dir: Path | None = None,
) -> list[str]:
    """List quarters already present in a workbook.

    Args:
        year: Year to check
        output_dir: Directory with workbooks (defaults to DATA_DIR/output)

    Returns:
        List of period strings (e.g., ["2024_QI", "2024_QII"])

    """
    search_dir = output_dir if output_dir is not None else DATA_DIR / "output"
    filepath = search_dir / f"EEFF_{year}.xlsx"

    if not filepath.exists():
        return []

    try:
        _, existing_quarters = _load_existing_sheet1(filepath)
        return sorted(existing_quarters, key=_period_sort_key)
    except Exception as e:
        logger.warning(f"Could not read workbook quarters: {e}")
        return []

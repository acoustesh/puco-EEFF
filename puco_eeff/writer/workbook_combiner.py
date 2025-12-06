"""Workbook combiner - combines multiple sheets into a single Excel file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from puco_eeff.config import DATA_DIR, setup_logging
from puco_eeff.writer.sheet_writer import list_available_sheets, load_sheet_data

logger = setup_logging(__name__)


def combine_sheets_to_workbook(
    period: str,
    sheet_order: list[str] | None = None,
    output_dir: Path | None = None,
    input_dir: Path | None = None,
) -> Path:
    """Combine all sheet data into a single Excel workbook.

    Args:
        period: Period identifier (e.g., "2024_Q3")
        sheet_order: Optional list specifying sheet order. If None, uses alphabetical.
        output_dir: Directory for output Excel file (defaults to DATA_DIR/output)
        input_dir: Directory with sheet JSON files (defaults to DATA_DIR/processed)

    Returns:
        Path to the created Excel workbook
    """
    save_dir = output_dir if output_dir is not None else DATA_DIR / "output"
    save_dir.mkdir(parents=True, exist_ok=True)

    load_dir = input_dir if input_dir is not None else DATA_DIR / "processed"

    # Get available sheets
    available = list_available_sheets(period, load_dir)
    if not available:
        raise ValueError(f"No sheet data found for period: {period}")

    logger.info(f"Found {len(available)} sheets for period {period}: {available}")

    # Determine sheet order
    if sheet_order:
        # Use specified order, but only include available sheets
        sheets_to_include = [s for s in sheet_order if s in available]
        # Add any remaining sheets not in order
        for s in available:
            if s not in sheets_to_include:
                sheets_to_include.append(s)
    else:
        sheets_to_include = available

    # Create Excel workbook
    year, quarter = _parse_period(period)
    filename = f"EEFF_{year}_Q{quarter}.xlsx"
    filepath = save_dir / filename

    logger.info(f"Creating workbook: {filepath}")

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        for sheet_name in sheets_to_include:
            try:
                df = load_sheet_data(sheet_name, period, load_dir)

                # Create a nice sheet name (title case, spaces)
                display_name = _format_sheet_name(sheet_name)

                # Excel sheet names are limited to 31 characters
                if len(display_name) > 31:
                    display_name = display_name[:31]

                df.to_excel(writer, sheet_name=display_name, index=False)
                logger.debug(f"Added sheet: {display_name} ({len(df)} rows)")

            except Exception as e:
                logger.error(f"Failed to add sheet '{sheet_name}': {e}")

    logger.info(f"Workbook created: {filepath}")
    return filepath


def _parse_period(period: str) -> tuple[int, int]:
    """Parse period string into year and quarter.

    Args:
        period: Period string (e.g., "2024_Q3")

    Returns:
        Tuple of (year, quarter)
    """
    parts = period.split("_")
    if len(parts) != 2:
        raise ValueError(f"Invalid period format: {period}")

    year = int(parts[0])
    quarter = int(parts[1].replace("Q", "").replace("q", ""))

    return year, quarter


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

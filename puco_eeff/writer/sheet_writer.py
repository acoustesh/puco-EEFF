"""Sheet writer for individual sheet data output.

Naming convention for per-quarter JSON files:
- sheet1_2024_QI.json
- sheet1_2024_QII.json
- sheet1_2024_QIII.json
- sheet1_2024_QIV.json

This preserves alphabetical ordering while using Roman numerals for consistency.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from puco_eeff.config import DATA_DIR, setup_logging

logger = setup_logging(__name__)

# Quarter to Roman numeral mapping
QUARTER_TO_ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV"}
ROMAN_TO_QUARTER = {"I": 1, "II": 2, "III": 3, "IV": 4}


def quarter_to_roman(quarter: int) -> str:
    """Convert quarter number to Roman numeral.

    Args:
        quarter: Quarter number (1-4)

    Returns:
        Roman numeral string (I, II, III, IV)
    """
    if quarter not in QUARTER_TO_ROMAN:
        raise ValueError(f"Invalid quarter: {quarter}. Must be 1-4.")
    return QUARTER_TO_ROMAN[quarter]


def roman_to_quarter(roman: str) -> int:
    """Convert Roman numeral to quarter number.

    Args:
        roman: Roman numeral string (I, II, III, IV)

    Returns:
        Quarter number (1-4)
    """
    roman = roman.upper()
    if roman not in ROMAN_TO_QUARTER:
        raise ValueError(f"Invalid Roman numeral: {roman}. Must be I, II, III, or IV.")
    return ROMAN_TO_QUARTER[roman]


def format_period(year: int, quarter: int) -> str:
    """Format year and quarter into period string.

    Args:
        year: Year (e.g., 2024)
        quarter: Quarter number (1-4)

    Returns:
        Period string (e.g., "2024_QII")
    """
    return f"{year}_Q{quarter_to_roman(quarter)}"


def parse_period(period: str) -> tuple[int, int]:
    """Parse period string into year and quarter.

    Args:
        period: Period string (e.g., "2024_QII" or "2024_Q2")

    Returns:
        Tuple of (year, quarter)
    """
    match = re.match(r"(\d{4})_Q(I{1,3}|IV|\d)", period)
    if not match:
        raise ValueError(f"Invalid period format: {period}")

    year = int(match.group(1))
    quarter_str = match.group(2)

    # Handle both Roman and numeric
    quarter = int(quarter_str) if quarter_str.isdigit() else roman_to_quarter(quarter_str)

    return year, quarter


def save_sheet_data(
    sheet_name: str,
    data: pd.DataFrame | dict[str, Any],
    year: int,
    quarter: int,
    output_dir: Path | None = None,
) -> Path:
    """Save sheet data as JSON for partial re-runs.

    Uses naming convention: sheet1_2024_QII.json for alphabetical ordering
    with Roman numeral quarters.

    Args:
        sheet_name: Name of the sheet (e.g., "sheet1", "balance_general")
        data: DataFrame or dictionary with sheet data
        year: Year (e.g., 2024)
        quarter: Quarter number (1-4)
        output_dir: Directory to save to (defaults to DATA_DIR/processed)

    Returns:
        Path to saved JSON file
    """
    save_dir = output_dir if output_dir is not None else DATA_DIR / "processed"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Format: sheet1_2024_QII.json (alphabetical + Roman)
    safe_name = sheet_name.lower().replace(" ", "_")
    period = format_period(year, quarter)
    filename = f"{safe_name}_{period}.json"
    filepath = save_dir / filename

    # Convert DataFrame to dict if needed
    if isinstance(data, pd.DataFrame):
        data_dict: dict[str, Any] = {
            "columns": list(data.columns),
            "data": data.to_dict(orient="records"),
            "shape": list(data.shape),
        }
    else:
        data_dict = data

    # Add metadata
    output = {
        "sheet_name": sheet_name,
        "year": year,
        "quarter": quarter,
        "period": period,
        "content": data_dict,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Saved sheet data: {filepath}")
    return filepath


def load_sheet_data(
    sheet_name: str,
    year: int,
    quarter: int,
    input_dir: Path | None = None,
) -> pd.DataFrame:
    """Load sheet data from JSON.

    Args:
        sheet_name: Name of the sheet
        year: Year (e.g., 2024)
        quarter: Quarter number (1-4)
        input_dir: Directory to load from (defaults to DATA_DIR/processed)

    Returns:
        DataFrame with sheet data
    """
    load_dir = input_dir if input_dir is not None else DATA_DIR / "processed"

    safe_name = sheet_name.lower().replace(" ", "_")
    period = format_period(year, quarter)
    filename = f"{safe_name}_{period}.json"
    filepath = load_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Sheet data not found: {filepath}")

    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    content = data["content"]
    df = pd.DataFrame(content["data"], columns=content["columns"])

    logger.info(f"Loaded sheet data: {filepath}")
    return df


def load_sheet_json(filepath: Path) -> dict[str, Any]:
    """Load raw JSON data from a sheet file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Dictionary with sheet data including metadata
    """
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def write_sheet_to_csv(
    sheet_name: str,
    data: pd.DataFrame,
    year: int,
    quarter: int,
    output_dir: Path | None = None,
) -> Path:
    """Write sheet data to CSV file.

    Args:
        sheet_name: Name of the sheet
        data: DataFrame with sheet data
        year: Year (e.g., 2024)
        quarter: Quarter number (1-4)
        output_dir: Directory to save to (defaults to DATA_DIR/processed)

    Returns:
        Path to saved CSV file
    """
    save_dir = output_dir if output_dir is not None else DATA_DIR / "processed"
    save_dir.mkdir(parents=True, exist_ok=True)

    safe_name = sheet_name.lower().replace(" ", "_")
    period = format_period(year, quarter)
    filename = f"{safe_name}_{period}.csv"
    filepath = save_dir / filename

    data.to_csv(filepath, index=False, encoding="utf-8")

    logger.info(f"Saved sheet CSV: {filepath}")
    return filepath


def list_available_sheets(
    sheet_name: str | None = None,
    year: int | None = None,
    input_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """List available sheet data files.

    Args:
        sheet_name: Optional sheet name filter (e.g., "sheet1")
        year: Optional year filter (e.g., 2024)
        input_dir: Directory to search (defaults to DATA_DIR/processed)

    Returns:
        List of dicts with sheet_name, year, quarter, period, and filepath
    """
    search_dir = input_dir if input_dir is not None else DATA_DIR / "processed"

    if not search_dir.exists():
        return []

    results = []

    # Pattern: sheet1_2024_QII.json
    for filepath in search_dir.glob("*.json"):
        # Parse filename: sheetname_YYYY_QX.json
        match = re.match(r"(.+)_(\d{4})_Q(I{1,3}|IV)\.json$", filepath.name)
        if not match:
            continue

        file_sheet_name = match.group(1)
        file_year = int(match.group(2))
        file_quarter = roman_to_quarter(match.group(3))
        file_period = format_period(file_year, file_quarter)

        # Apply filters
        if sheet_name is not None and file_sheet_name != sheet_name.lower().replace(" ", "_"):
            continue
        if year is not None and file_year != year:
            continue

        results.append(
            {
                "sheet_name": file_sheet_name,
                "year": file_year,
                "quarter": file_quarter,
                "period": file_period,
                "filepath": filepath,
            }
        )

    # Sort by sheet name, then year, then quarter
    results.sort(key=lambda x: (x["sheet_name"], x["year"], x["quarter"]))
    return results

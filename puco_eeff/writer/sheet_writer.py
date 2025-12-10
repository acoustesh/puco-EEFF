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
import operator
import re
from typing import TYPE_CHECKING, Any

import pandas as pd

from puco_eeff.config import DATA_DIR, format_period, setup_logging

if TYPE_CHECKING:
    from pathlib import Path

logger = setup_logging(__name__)

ROMAN_TO_QUARTER = {"I": 1, "II": 2, "III": 3, "IV": 4}


def roman_to_quarter(roman: str) -> int:
    """Parse a Roman numeral quarter string to an integer.

    Parameters
    ----------
    roman
        Roman numeral representation (``"I"``–``"IV"``).

    Returns
    -------
    int
        Quarter number (1–4).

    Raises
    ------
    ValueError
        If the Roman numeral is not one of ``I``, ``II``, ``III``, or ``IV``.
    """
    q = ROMAN_TO_QUARTER.get(roman.upper())
    if q is None:
        msg = f"Invalid Roman numeral: {roman}. Must be I, II, III, or IV."
        raise ValueError(msg)
    return q


# Use centralized format_period from config (defaults to key style)
# format_period(year, period, period_type, style) - use style="key" for file names


def parse_period(period: str) -> tuple[int, int]:
    """Parse period string into year and quarter.

    Parameters
    ----------
    period
        Period key such as ``"2024_QII"`` or ``"2024_Q2"``.

    Returns
    -------
    tuple[int, int]
        Parsed ``(year, quarter)`` pair.

    Raises
    ------
    ValueError
        If the period string does not match the expected pattern.
    """
    match = re.match(r"(\d{4})_Q(I{1,3}|IV|\d)", period)
    if not match:
        msg = f"Invalid period format: {period}"
        raise ValueError(msg)

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

    Parameters
    ----------
    sheet_name
        Name of the sheet (e.g., ``"sheet1"`` or ``"balance_general"``).
    data
        Sheet payload as a DataFrame or mapping.
    year
        Year of the statement.
    quarter
        Quarter number in ``1–4``.
    output_dir
        Custom output directory; defaults to ``DATA_DIR/processed``.

    Returns
    -------
    Path
        Location of the written JSON file.
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

    with filepath.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Saved sheet data: %s", filepath)
    return filepath


def load_sheet_data(
    sheet_name: str,
    year: int,
    quarter: int,
    input_dir: Path | None = None,
) -> pd.DataFrame:
    """Load sheet data from JSON.

    Parameters
    ----------
    sheet_name
        Target sheet name.
    year
        Year of the statement.
    quarter
        Quarter number in ``1–4``.
    input_dir
        Optional directory to load from; defaults to ``DATA_DIR/processed``.

    Returns
    -------
    pd.DataFrame
        Sheet data reconstructed from the JSON file.
    """
    load_dir = input_dir if input_dir is not None else DATA_DIR / "processed"

    safe_name = sheet_name.lower().replace(" ", "_")
    period = format_period(year, quarter)
    filename = f"{safe_name}_{period}.json"
    filepath = load_dir / filename

    if not filepath.exists():
        msg = f"Sheet data not found: {filepath}"
        raise FileNotFoundError(msg)

    with filepath.open(encoding="utf-8") as f:
        data = json.load(f)

    content = data["content"]
    df = pd.DataFrame(content["data"], columns=content["columns"])

    logger.info("Loaded sheet data: %s", filepath)
    return df


def load_sheet_json(filepath: Path) -> dict[str, Any]:
    """Load raw JSON data from a sheet file.

    Parameters
    ----------
    filepath
        Path to the JSON file.

    Returns
    -------
    dict[str, Any]
        Raw JSON object including metadata and content.
    """
    with filepath.open(encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def write_sheet_to_csv(
    sheet_name: str,
    data: pd.DataFrame,
    year: int,
    quarter: int,
    output_dir: Path | None = None,
) -> Path:
    """Write sheet data to CSV file.

    Parameters
    ----------
    sheet_name
        Name of the sheet.
    data
        DataFrame containing the sheet content.
    year
        Statement year.
    quarter
        Quarter number in ``1–4``.
    output_dir
        Optional output directory; defaults to ``DATA_DIR/processed``.

    Returns
    -------
    Path
        Location of the written CSV file.
    """
    save_dir = output_dir if output_dir is not None else DATA_DIR / "processed"
    save_dir.mkdir(parents=True, exist_ok=True)

    safe_name = sheet_name.lower().replace(" ", "_")
    period = format_period(year, quarter)
    filename = f"{safe_name}_{period}.csv"
    filepath = save_dir / filename

    data.to_csv(filepath, index=False, encoding="utf-8")

    logger.info("Saved sheet CSV: %s", filepath)
    return filepath


def list_available_sheets(
    sheet_name: str | None = None,
    year: int | None = None,
    input_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """List available sheet data files.

    Parameters
    ----------
    sheet_name
        Optional sheet name filter (e.g., ``"sheet1"``).
    year
        Optional year filter.
    input_dir
        Directory to search; defaults to ``DATA_DIR/processed``.

    Returns
    -------
    list[dict[str, Any]]
        Metadata for each discovered sheet JSON file.
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
            },
        )

    # Sort by sheet name, then year, then quarter
    results.sort(key=operator.itemgetter("sheet_name", "year", "quarter"))
    return results

"""Sheet writer for individual sheet data output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from puco_eeff.config import DATA_DIR, setup_logging

logger = setup_logging(__name__)


def save_sheet_data(
    sheet_name: str,
    data: pd.DataFrame | dict[str, Any],
    period: str,
    output_dir: Path | None = None,
) -> Path:
    """Save sheet data as JSON for partial re-runs.

    Args:
        sheet_name: Name of the sheet (e.g., "sheet1", "balance_general")
        data: DataFrame or dictionary with sheet data
        period: Period identifier (e.g., "2024_Q3")
        output_dir: Directory to save to (defaults to DATA_DIR/processed)

    Returns:
        Path to saved JSON file
    """
    save_dir = output_dir if output_dir is not None else DATA_DIR / "processed"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Normalize sheet name for filename
    safe_name = sheet_name.lower().replace(" ", "_")
    filename = f"{period}_{safe_name}.json"
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
        "period": period,
        "content": data_dict,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Saved sheet data: {filepath}")
    return filepath


def load_sheet_data(
    sheet_name: str,
    period: str,
    input_dir: Path | None = None,
) -> pd.DataFrame:
    """Load sheet data from JSON.

    Args:
        sheet_name: Name of the sheet
        period: Period identifier
        input_dir: Directory to load from (defaults to DATA_DIR/processed)

    Returns:
        DataFrame with sheet data
    """
    load_dir = input_dir if input_dir is not None else DATA_DIR / "processed"

    safe_name = sheet_name.lower().replace(" ", "_")
    filename = f"{period}_{safe_name}.json"
    filepath = load_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Sheet data not found: {filepath}")

    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    content = data["content"]
    df = pd.DataFrame(content["data"], columns=content["columns"])

    logger.info(f"Loaded sheet data: {filepath}")
    return df


def write_sheet_to_csv(
    sheet_name: str,
    data: pd.DataFrame,
    period: str,
    output_dir: Path | None = None,
) -> Path:
    """Write sheet data to CSV file.

    Args:
        sheet_name: Name of the sheet
        data: DataFrame with sheet data
        period: Period identifier
        output_dir: Directory to save to (defaults to DATA_DIR/processed)

    Returns:
        Path to saved CSV file
    """
    save_dir = output_dir if output_dir is not None else DATA_DIR / "processed"
    save_dir.mkdir(parents=True, exist_ok=True)

    safe_name = sheet_name.lower().replace(" ", "_")
    filename = f"{period}_{safe_name}.csv"
    filepath = save_dir / filename

    data.to_csv(filepath, index=False, encoding="utf-8")

    logger.info(f"Saved sheet CSV: {filepath}")
    return filepath


def list_available_sheets(period: str, input_dir: Path | None = None) -> list[str]:
    """List available sheet data files for a period.

    Args:
        period: Period identifier
        input_dir: Directory to search (defaults to DATA_DIR/processed)

    Returns:
        List of sheet names with available data
    """
    search_dir = input_dir if input_dir is not None else DATA_DIR / "processed"

    if not search_dir.exists():
        return []

    sheets = []
    pattern = f"{period}_*.json"

    for filepath in search_dir.glob(pattern):
        # Extract sheet name from filename
        name = filepath.stem.replace(f"{period}_", "")
        sheets.append(name)

    return sorted(sheets)

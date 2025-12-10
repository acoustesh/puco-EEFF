"""Writer module for Excel and CSV output.

Per-quarter JSON naming convention: sheet1_2024_QII.json
Workbook output: EEFF_2024.xlsx with quarters as columns
"""

from puco_eeff.config import quarter_to_roman
from puco_eeff.writer.sheet_writer import (
    format_period,
    list_available_sheets,
    load_sheet_data,
    parse_period,
    roman_to_quarter,
    save_sheet_data,
    write_sheet_to_csv,
)
from puco_eeff.writer.workbook_combiner import (
    append_quarter_to_workbook,
    combine_sheet1_quarters,
    create_workbook_from_dataframes,
    list_workbook_quarters,
)

__all__ = [
    "append_quarter_to_workbook",
    # Workbook combiner
    "combine_sheet1_quarters",
    "create_workbook_from_dataframes",
    # Period utilities
    "format_period",
    "list_available_sheets",
    "list_workbook_quarters",
    "load_sheet_data",
    "parse_period",
    "quarter_to_roman",
    "roman_to_quarter",
    # Sheet writer
    "save_sheet_data",
    "write_sheet_to_csv",
]

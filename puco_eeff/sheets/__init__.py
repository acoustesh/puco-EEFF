"""Sheet extraction modules for puco-EEFF.

Each sheet has its own module with:
- Data class definition
- Extraction logic
- Validation functions

Configuration is loaded from config/sheetN/ directories.

Modules:
- sheet1: Ingresos y Costos (Nota 21, 22 from PDF + XBRL)
"""

from puco_eeff.sheets.sheet1 import (
    Sheet1Data,
    format_quarter_label,
    get_sheet1_extraction_config,
    get_sheet1_extraction_sections,
    get_sheet1_fields,
    get_sheet1_reference_data,
    get_sheet1_reference_values,
    get_sheet1_row_mapping,
    get_sheet1_section_expected_items,
    get_sheet1_section_field_mappings,
    get_sheet1_section_search_patterns,
    get_sheet1_section_spec,
    get_sheet1_section_table_identifiers,
    get_sheet1_sum_tolerance,
    get_sheet1_validation_rules,
    get_sheet1_value_fields,
    get_sheet1_xbrl_fact_mapping,
    get_sheet1_xbrl_mappings,
    match_concepto_to_field,
    print_sheet1_report,
    quarter_to_roman,
    save_sheet1_data,
    validate_sheet1_against_reference,
)

__all__ = [
    "Sheet1Data",
    "format_quarter_label",
    "get_sheet1_extraction_config",
    "get_sheet1_extraction_sections",
    "get_sheet1_fields",
    "get_sheet1_reference_data",
    "get_sheet1_reference_values",
    "get_sheet1_row_mapping",
    "get_sheet1_section_expected_items",
    "get_sheet1_section_field_mappings",
    "get_sheet1_section_search_patterns",
    "get_sheet1_section_spec",
    "get_sheet1_section_table_identifiers",
    "get_sheet1_sum_tolerance",
    "get_sheet1_validation_rules",
    "get_sheet1_value_fields",
    "get_sheet1_xbrl_fact_mapping",
    "get_sheet1_xbrl_mappings",
    "match_concepto_to_field",
    "print_sheet1_report",
    "quarter_to_roman",
    "save_sheet1_data",
    "validate_sheet1_against_reference",
]

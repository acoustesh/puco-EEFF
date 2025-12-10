"""Sheet extraction modules for puco-EEFF.

This package contains sheet-specific data classes, config loaders, and validation
logic. Each sheet corresponds to a tab in the final Excel workbook.

Architecture
------------
Each sheet module (e.g., sheet1.py) provides:
- Data class: Typed container for extracted values (e.g., Sheet1Data)
- Config loaders: Functions to load JSON config from config/sheetN/
- Validation: Reference value comparison and sum checks
- Serialization: to_dict(), to_row_list() for output formatting

Configuration Loading
---------------------
Config files live in config/sheet1/, config/sheet2/, etc.:
- fields.json: Field definitions and row mappings
- extraction.json: Section specs, search patterns, field mappings
- xbrl_mappings.json: XBRL fact names to field mappings
- reference_data.json: Verified reference values for validation

Modules
-------
sheet1
    Ingresos y Costos (Nota 21, Nota 22 from PDF + XBRL)
    Contains cost breakdown for Costo de Venta and Gasto Administrativo.

Notes
-----
To add a new sheet, create config/sheetN/ directory with required JSON files,
then create puco_eeff/sheets/sheetN.py following sheet1.py as a template.
"""

# Re-export all public symbols from sheet1 for convenience
from puco_eeff.sheets.sheet1 import (
    Sheet1Data,
    format_quarter_label,
    get_sheet1_cross_validations,
    get_sheet1_detail_fields,
    get_sheet1_extraction_config,
    get_sheet1_extraction_sections,
    get_sheet1_fields,
    get_sheet1_pdf_xbrl_validations,
    get_sheet1_reference_data,
    get_sheet1_reference_values,
    get_sheet1_result_key_mapping,
    get_sheet1_row_mapping,
    get_sheet1_section_expected_items,
    get_sheet1_section_field_mappings,
    get_sheet1_section_search_patterns,
    get_sheet1_section_spec,
    get_sheet1_section_table_identifiers,
    get_sheet1_section_total_mapping,
    get_sheet1_sum_tolerance,
    get_sheet1_total_validations,
    get_sheet1_validation_rules,
    get_sheet1_value_fields,
    get_sheet1_xbrl_fact_mapping,
    get_sheet1_xbrl_mappings,
    match_concepto_to_field,
    print_sheet1_report,
    quarter_to_roman,
    save_sheet1_data,
    sections_to_sheet1data,
    validate_sheet1_against_reference,
)

# Public API - all symbols re-exported from sheet1
__all__ = [
    # Data class
    "Sheet1Data",
    # Quarter formatting
    "format_quarter_label",
    "quarter_to_roman",
    # Validation config loaders
    "get_sheet1_cross_validations",
    "get_sheet1_pdf_xbrl_validations",
    "get_sheet1_sum_tolerance",
    "get_sheet1_total_validations",
    "get_sheet1_validation_rules",
    # Extraction config loaders
    "get_sheet1_detail_fields",
    "get_sheet1_extraction_config",
    "get_sheet1_extraction_sections",
    "get_sheet1_fields",
    "get_sheet1_row_mapping",
    "get_sheet1_value_fields",
    # Section config loaders
    "get_sheet1_section_expected_items",
    "get_sheet1_section_field_mappings",
    "get_sheet1_section_search_patterns",
    "get_sheet1_section_spec",
    "get_sheet1_section_table_identifiers",
    "get_sheet1_section_total_mapping",
    # XBRL config loaders
    "get_sheet1_xbrl_fact_mapping",
    "get_sheet1_xbrl_mappings",
    # Reference data loaders
    "get_sheet1_reference_data",
    "get_sheet1_reference_values",
    "get_sheet1_result_key_mapping",
    # Extraction helpers
    "match_concepto_to_field",
    "sections_to_sheet1data",
    # Output functions
    "print_sheet1_report",
    "save_sheet1_data",
    "validate_sheet1_against_reference",
]

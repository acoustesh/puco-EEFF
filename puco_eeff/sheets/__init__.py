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

sheet2
    Cuadro Resumen KPIs (from Análisis Razonado PDF)
    Contains revenue breakdown by product, EBITDA, and operational metrics.

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

# Re-export all public symbols from sheet2
from puco_eeff.sheets.sheet2 import (
    Sheet2Data,
    extract_sheet2,
    get_sheet2_extraction_config,
    get_sheet2_extraction_sections,
    get_sheet2_fields,
    get_sheet2_metadata_fields,
    get_sheet2_reference_data,
    get_sheet2_reference_values,
    get_sheet2_row_mapping,
    get_sheet2_section_field_mappings,
    get_sheet2_section_search_patterns,
    get_sheet2_section_spec,
    get_sheet2_sum_tolerance,
    get_sheet2_total_validations,
    get_sheet2_value_fields,
    get_sheet2_xbrl_mappings,
    match_concepto_to_field_sheet2,
    parse_spanish_number,
    print_sheet2_report,
    save_sheet2_to_json,
    validate_sheet2_against_reference,
    validate_sheet2_sums,
)

# Public API - all symbols re-exported from sheet1 and sheet2
__all__ = [
    # Sheet1 Data class
    "Sheet1Data",
    # Sheet2 Data class
    "Sheet2Data",
    # Quarter formatting
    "format_quarter_label",
    "quarter_to_roman",
    # Sheet1 Validation config loaders
    "get_sheet1_cross_validations",
    "get_sheet1_pdf_xbrl_validations",
    "get_sheet1_sum_tolerance",
    "get_sheet1_total_validations",
    "get_sheet1_validation_rules",
    # Sheet1 Extraction config loaders
    "get_sheet1_detail_fields",
    "get_sheet1_extraction_config",
    "get_sheet1_extraction_sections",
    "get_sheet1_fields",
    "get_sheet1_row_mapping",
    "get_sheet1_value_fields",
    # Sheet1 Section config loaders
    "get_sheet1_section_expected_items",
    "get_sheet1_section_field_mappings",
    "get_sheet1_section_search_patterns",
    "get_sheet1_section_spec",
    "get_sheet1_section_table_identifiers",
    "get_sheet1_section_total_mapping",
    # Sheet1 XBRL config loaders
    "get_sheet1_xbrl_fact_mapping",
    "get_sheet1_xbrl_mappings",
    # Sheet1 Reference data loaders
    "get_sheet1_reference_data",
    "get_sheet1_reference_values",
    "get_sheet1_result_key_mapping",
    # Sheet1 Extraction helpers
    "match_concepto_to_field",
    "sections_to_sheet1data",
    # Sheet1 Output functions
    "print_sheet1_report",
    "save_sheet1_data",
    "validate_sheet1_against_reference",
    # Sheet2 Config loaders
    "get_sheet2_extraction_config",
    "get_sheet2_extraction_sections",
    "get_sheet2_fields",
    "get_sheet2_metadata_fields",
    "get_sheet2_reference_data",
    "get_sheet2_reference_values",
    "get_sheet2_row_mapping",
    "get_sheet2_section_field_mappings",
    "get_sheet2_section_search_patterns",
    "get_sheet2_section_spec",
    "get_sheet2_sum_tolerance",
    "get_sheet2_total_validations",
    "get_sheet2_value_fields",
    "get_sheet2_xbrl_mappings",
    # Sheet2 Extraction
    "extract_sheet2",
    "match_concepto_to_field_sheet2",
    "parse_spanish_number",
    # Sheet2 Output functions
    "print_sheet2_report",
    "save_sheet2_to_json",
    "validate_sheet2_against_reference",
    "validate_sheet2_sums",
]

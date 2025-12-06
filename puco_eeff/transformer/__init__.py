"""Transformer module for data normalization, formatting, and source tracking."""

from puco_eeff.transformer.formatter import (
    ValidationResult,
    format_validation_report,
    get_field_labels,
    get_standard_structure,
    map_to_structure,
    validate_against_reference,
    validate_balance_sheet,
    validate_costo_venta_total,
    validate_gasto_admin_total,
)
from puco_eeff.transformer.normalizer import normalize_financial_data
from puco_eeff.transformer.source_tracker import SourceTracker, create_source_mapping

__all__ = [
    "SourceTracker",
    "ValidationResult",
    "create_source_mapping",
    "format_validation_report",
    "get_field_labels",
    "get_standard_structure",
    "map_to_structure",
    "normalize_financial_data",
    "validate_against_reference",
    "validate_balance_sheet",
    "validate_costo_venta_total",
    "validate_gasto_admin_total",
]

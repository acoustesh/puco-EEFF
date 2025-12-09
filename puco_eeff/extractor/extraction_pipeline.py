"""High-level extraction orchestration for Sheet1 data.

This module provides the main entry points for extracting Sheet1 data from
PDF and XBRL files.

Key Functions:
    extract_sheet1(): Unified entry point for Sheet1 extraction (PDF/XBRL/both).
    extract_detailed_costs(): Extract detailed cost breakdowns.
    save_extraction_result(): Save extraction result to JSON file.
    print_extraction_report(): Print extraction report to console.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from puco_eeff.config import (
    find_file_with_alternatives,
    format_filename,
    get_period_paths,
    setup_logging,
)
from puco_eeff.extractor.extraction import (
    LineItem,
    SectionBreakdown,
    extract_ingresos_from_pdf,
    extract_pdf_section,
    extract_xbrl_totals,
    format_quarter_label,
    quarter_to_roman,
)
from puco_eeff.extractor.validation_core import (
    ExtractionResult,
    ValidationReport,
    _run_sum_validations,
    run_sheet1_validations,
)
from puco_eeff.sheets.sheet1 import (
    Sheet1Data,
    get_sheet1_result_key_mapping,
    match_concepto_to_field,
    print_sheet1_report,
    save_sheet1_data,
    sections_to_sheet1data,
)

logger = setup_logging(__name__)

# Public API exports
__all__ = [
    "extract_detailed_costs",
    "extract_sheet1",
    "print_extraction_report",
    "print_sheet1_report",
    "save_extraction_result",
    # Re-exports
    "save_sheet1_data",
]

# =============================================================================
# Source Resolution Helpers
# =============================================================================


def _determine_source(raw_dir: Path, year: int, quarter: int) -> str:
    """Determine the data source based on available files."""
    combined_path = find_file_with_alternatives(raw_dir, "pucobre_combined", year, quarter)
    return "pucobre.cl" if (combined_path and combined_path.exists()) else "cmf"


def _resolve_pdf_path(raw_dir: Path, year: int, quarter: int) -> Path | None:
    """Resolve the Estados Financieros PDF path."""
    ef_path = find_file_with_alternatives(raw_dir, "estados_financieros_pdf", year, quarter)
    if not ef_path:
        ef_path = raw_dir / format_filename("estados_financieros_pdf", year, quarter)
    return ef_path if ef_path.exists() else None


def _resolve_xbrl_path(raw_xbrl_dir: Path, year: int, quarter: int) -> tuple[Path | None, bool]:
    """Resolve the XBRL path and availability."""
    xbrl_path = find_file_with_alternatives(raw_xbrl_dir, "estados_financieros_xbrl", year, quarter)
    if not xbrl_path:
        xbrl_path = raw_xbrl_dir / format_filename("estados_financieros_xbrl", year, quarter)
    xbrl_available = xbrl_path.exists() if xbrl_path else False
    return xbrl_path, xbrl_available


# =============================================================================
# Sheet1Data Factory
# =============================================================================


def _create_sheet1_data(year: int, quarter: int, source: str, xbrl_available: bool) -> Sheet1Data:
    """Create a new Sheet1Data instance with common initialization."""
    return Sheet1Data(
        quarter=format_quarter_label(year, quarter),
        year=year,
        quarter_num=quarter,
        source=source,
        xbrl_available=xbrl_available,
    )


# =============================================================================
# PDF Extraction Helpers
# =============================================================================


def _extract_nota_sections(pdf_path: Path, result: ExtractionResult) -> None:
    """Extract Nota 21 and 22 sections into result."""
    for nota_name in ("nota_21", "nota_22"):
        section = extract_pdf_section(pdf_path, nota_name)
        if section is not None:
            result.sections[nota_name] = section


def _map_nota_item_to_sheet1(item: LineItem, data: Sheet1Data, section_name: str) -> None:
    """Map a Nota line item to Sheet1Data fields using config-driven matching."""
    field_name = match_concepto_to_field(item.concepto, section_name)

    if field_name:
        data.set_value(field_name, item.ytd_actual)
        logger.debug("Mapped '%s' -> %s = %s", item.concepto, field_name, item.ytd_actual)
    else:
        logger.warning("Could not map item from %s: '%s'", section_name, item.concepto)


def _populate_sheet1_from_notas(data: Sheet1Data, ef_path: Path) -> bool:
    """Extract Nota 21 and 22 from PDF and populate Sheet1Data.

    Returns True if at least one nota was successfully extracted.
    """
    nota_sections = [
        ("nota_21", "total_costo_venta"),
        ("nota_22", "total_gasto_admin"),
    ]

    any_extracted = False
    for section_name, total_field in nota_sections:
        nota = extract_pdf_section(ef_path, section_name)
        if nota:
            data.set_value(total_field, nota.total_ytd_actual)
            for item in nota.items:
                _map_nota_item_to_sheet1(item, data, section_name)
            any_extracted = True

    return any_extracted


def _extract_ingresos_fallback(data: Sheet1Data, ef_path: Path) -> ValidationReport:
    """Extract Ingresos from PDF and return sum-only validation report."""
    logger.info("No XBRL available, extracting Ingresos from Estado de Resultados")
    ingresos_value = extract_ingresos_from_pdf(ef_path)
    if ingresos_value is not None:
        data.ingresos_ordinarios = ingresos_value
        logger.info("Set Ingresos from PDF: %s", f"{ingresos_value:,}")
    else:
        logger.warning("Could not extract Ingresos from PDF")

    return ValidationReport(sum_validations=_run_sum_validations(data))


# =============================================================================
# XBRL Extraction Helpers
# =============================================================================


# XBRL field to Sheet1Data field mapping
_XBRL_TO_SHEET1_FIELDS = {
    "ingresos_de_actividades_ordinarias": "ingresos_ordinarios",
    "costo_de_ventas": "total_costo_venta",
    "gastos_de_administracion": "total_gasto_admin",
}


def _transfer_xbrl_data(
    xbrl_totals: dict[str, int | None],
    target: ExtractionResult | Sheet1Data,
) -> None:
    """Transfer XBRL data to target (ExtractionResult or Sheet1Data).

    For ExtractionResult: Populates xbrl_totals dict using result_key_mapping.
    For Sheet1Data: Populates field values using _XBRL_TO_SHEET1_FIELDS mapping.

    Args:
        xbrl_totals: Source XBRL totals dictionary.
        target: Destination ExtractionResult or Sheet1Data object.

    """
    if isinstance(target, ExtractionResult):
        # ExtractionResult: copy XBRL totals using key mapping
        result_key_mapping = get_sheet1_result_key_mapping()
        for xbrl_key in result_key_mapping.values():
            if xbrl_key in xbrl_totals:
                target.xbrl_totals[xbrl_key] = xbrl_totals.get(xbrl_key)
    else:
        # Sheet1Data: set field values using field mapping
        for xbrl_field, sheet1_field in _XBRL_TO_SHEET1_FIELDS.items():
            if (value := xbrl_totals.get(xbrl_field)) is not None:
                target.set_value(sheet1_field, value)


# =============================================================================
# Validation Helpers
# =============================================================================


def _validate_extraction(
    result: ExtractionResult,
    source: str,
    xbrl_totals: dict[str, int | None] | None = None,
) -> None:
    """Validate extraction result, optionally against XBRL totals.

    Consolidates validation logic for both XBRL-backed and PDF-only modes.
    When xbrl_totals is provided, runs full validation including cross-validations.
    When xbrl_totals is None, runs only sum and PDF-only validations.
    """
    has_xbrl = xbrl_totals is not None

    if has_xbrl:
        _transfer_xbrl_data(xbrl_totals, result)

    sheet1_data = sections_to_sheet1data(result.sections, result.year, result.quarter)
    sheet1_data.xbrl_available = has_xbrl
    sheet1_data.source = source

    report = run_sheet1_validations(
        sheet1_data,
        xbrl_totals,
        run_sum_validations=True,
        run_pdf_xbrl_validations=True,
        run_cross_validations=has_xbrl,
        use_xbrl_fallback=has_xbrl,
    )
    result.validations = report.pdf_xbrl_validations
    result.validation_report = report


# =============================================================================
# High-Level Extraction Functions
# =============================================================================


def _extract_from_pdf(
    year: int,
    quarter: int,
    validate: bool,
) -> tuple[Sheet1Data | None, ValidationReport | None]:
    """Extract Sheet1 data from PDF Nota 21/22 sections, optionally validating against XBRL.

    This is the core PDF extraction workflow - it handles path resolution, table parsing,
    nota extraction, and optional XBRL cross-validation. The implementation intentionally
    uses a linear flow with early returns for clarity in error handling.

    Pipeline stages:
        1. Resolve PDF path → fail fast if missing
        2. Check for XBRL availability → sets cross-validation mode
        3. Parse nota sections from PDF tables
        4. Run validation against XBRL (if available and requested)
        5. Fall back to ingresos extraction if no XBRL
    """
    period_paths = get_period_paths(year, quarter)
    raw_pdf_dir = period_paths["raw_pdf"]
    raw_xbrl_dir = period_paths["raw_xbrl"]

    # Stage 1: PDF resolution
    estados_financieros_pdf = _resolve_pdf_path(raw_pdf_dir, year, quarter)
    if estados_financieros_pdf is None:
        logger.warning("PDF not found in %s", raw_pdf_dir)
        return None, None

    # Stage 2: XBRL availability check
    xbrl_file_path, xbrl_exists = _resolve_xbrl_path(raw_xbrl_dir, year, quarter)
    data_source = _determine_source(raw_pdf_dir, year, quarter)
    sheet1_data = _create_sheet1_data(year, quarter, data_source, xbrl_exists)

    # Stage 3: Nota parsing
    extraction_succeeded = _populate_sheet1_from_notas(sheet1_data, estados_financieros_pdf)
    if not extraction_succeeded:
        logger.error("Nota extraction failed from %s", estados_financieros_pdf)
        return None, None

    # Stage 4/5: Validation or fallback
    should_cross_validate = xbrl_exists and validate and xbrl_file_path is not None
    if should_cross_validate:
        xbrl_data = extract_xbrl_totals(xbrl_file_path)
        return sheet1_data, run_sheet1_validations(sheet1_data, xbrl_data)

    return sheet1_data, _extract_ingresos_fallback(sheet1_data, estados_financieros_pdf)


def _load_xbrl_sheet1_data(year: int, quarter: int) -> tuple[Sheet1Data | None, dict | None]:
    """Load XBRL file and return Sheet1Data with raw totals. Returns (None, None) on failure."""
    xbrl_dir = get_period_paths(year, quarter)["raw_xbrl"]
    xbrl_file = find_file_with_alternatives(xbrl_dir, "estados_financieros_xbrl", year, quarter)
    if xbrl_file is None:
        xbrl_file = xbrl_dir / format_filename("estados_financieros_xbrl", year, quarter)
    if not xbrl_file or not xbrl_file.exists():
        logger.warning("XBRL file not found for %dQ%d", year, quarter)
        return None, None

    totals_from_xbrl = extract_xbrl_totals(xbrl_file)
    if not totals_from_xbrl:
        logger.error("Could not extract totals from XBRL: %s", xbrl_file)
        return None, None

    xbrl_sheet1 = _create_sheet1_data(year, quarter, "xbrl", xbrl_available=True)
    _transfer_xbrl_data(totals_from_xbrl, xbrl_sheet1)
    return xbrl_sheet1, totals_from_xbrl


def extract_detailed_costs(year: int, quarter: int, validate: bool = True) -> ExtractionResult:
    """Extract detailed cost breakdowns for a period."""
    paths = get_period_paths(year, quarter)
    pdf_path = _resolve_pdf_path(paths["raw_pdf"], year, quarter)
    if not pdf_path:
        logger.error("PDF not found in: %s", paths["raw_pdf"])
        return ExtractionResult(year=year, quarter=quarter, xbrl_available=False)

    source = _determine_source(paths["raw_pdf"], year, quarter)
    result = ExtractionResult(year=year, quarter=quarter, source=source, pdf_path=pdf_path)
    _extract_nota_sections(pdf_path, result)

    xbrl_path, xbrl_available = _resolve_xbrl_path(paths["raw_xbrl"], year, quarter)
    result.xbrl_available = xbrl_available
    result.xbrl_path = xbrl_path if xbrl_available else None

    if not validate:
        return result

    if xbrl_available and xbrl_path:
        xbrl_totals = extract_xbrl_totals(xbrl_path)
        _validate_extraction(result, source, xbrl_totals)
    else:
        logger.info("No XBRL available for %s Q%s - using PDF only", year, quarter)
        _validate_extraction(result, source)

    return result


def extract_sheet1(
    year: int,
    quarter: int,
    prefer_source: str = "pdf",
    merge_sources: bool = True,
    return_report: bool = False,
    validate_with_xbrl: bool = True,
) -> Sheet1Data | None | tuple[Sheet1Data | None, ValidationReport | None]:
    """Unified API entry point for Sheet1 extraction with configurable source strategy."""

    def _xbrl_extraction():
        """Load XBRL totals and create Sheet1Data with sum validations."""
        xbrl_sheet1, _ = _load_xbrl_sheet1_data(year, quarter)
        return (
            (None, None)
            if xbrl_sheet1 is None
            else (xbrl_sheet1, ValidationReport(sum_validations=_run_sum_validations(xbrl_sheet1)))
        )

    def _result(d, r):
        """Format output based on return_report flag."""
        return (d, r) if return_report else d

    # Source strategy dispatch using match/case
    match prefer_source:
        case "xbrl":
            data, report = _xbrl_extraction()
            if data is None:
                logger.info("XBRL extraction failed, trying PDF fallback")
                return _result(*_extract_from_pdf(year, quarter, validate=False))
            if merge_sources:
                pdf_result, _ = _extract_from_pdf(year, quarter, validate=False)
                if pdf_result:
                    data = _merge_pdf_into_xbrl_data(data, pdf_result)
                    report = ValidationReport(sum_validations=_run_sum_validations(data))
        case _:  # Default to PDF-first
            data, report = _extract_from_pdf(year, quarter, validate_with_xbrl)
            if data is None:
                logger.info("PDF extraction failed, trying XBRL fallback")
                data, report = _xbrl_extraction()

    return _result(data, report)


def _merge_pdf_into_xbrl_data(xbrl_data: Sheet1Data, pdf_data: Sheet1Data) -> Sheet1Data:
    """Merge detailed PDF data into XBRL data."""
    for field_name in pdf_data.__dataclass_fields__:
        if field_name in {"quarter", "year", "quarter_num", "source", "xbrl_available"}:
            continue

        pdf_value = getattr(pdf_data, field_name, None)
        xbrl_value = getattr(xbrl_data, field_name, None)

        if pdf_value is not None and xbrl_value is None:
            setattr(xbrl_data, field_name, pdf_value)
            logger.debug("Merged PDF field %s = %s into XBRL data", field_name, pdf_value)

    xbrl_data.source = f"xbrl+{pdf_data.source}"
    return xbrl_data


def _breakdown_to_dict(breakdown: SectionBreakdown) -> dict:
    """Convert SectionBreakdown to dictionary."""
    return {
        "section_id": breakdown.section_id,
        "section_title": breakdown.section_title,
        "page_number": breakdown.page_number,
        "items": [
            {
                "concepto": item.concepto,
                "ytd_actual": item.ytd_actual,
                "ytd_anterior": item.ytd_anterior,
                "quarter_actual": item.quarter_actual,
                "quarter_anterior": item.quarter_anterior,
            }
            for item in breakdown.items
        ],
        "total_ytd_actual": breakdown.total_ytd_actual,
        "total_ytd_anterior": breakdown.total_ytd_anterior,
        "total_quarter_actual": breakdown.total_quarter_actual,
        "total_quarter_anterior": breakdown.total_quarter_anterior,
    }


def _save_legacy_extraction_result(result: ExtractionResult, output_dir: Path | None) -> Path:
    """Save ExtractionResult in legacy format (detailed_costs.json)."""
    if output_dir is None:
        paths = get_period_paths(result.year, result.quarter)
        output_dir = paths["processed"]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "detailed_costs.json"

    nota_21 = result.sections.get("nota_21")
    nota_22 = result.sections.get("nota_22")
    data = {
        "period": f"{result.year}_Q{result.quarter}",
        "source": result.source,
        "pdf_path": str(result.pdf_path) if result.pdf_path else None,
        "xbrl_path": str(result.xbrl_path) if result.xbrl_path else None,
        "xbrl_available": result.xbrl_available,
        "nota_21": _breakdown_to_dict(nota_21) if nota_21 else None,
        "nota_22": _breakdown_to_dict(nota_22) if nota_22 else None,
        "validations": [
            {
                "field": v.field_name,
                "pdf_value": v.value_a,  # PDF extracted data
                "xbrl_value": v.value_b,  # XBRL source data
                "match": v.match,
                "source": v.source,
                "status": v.status,
            }
            for v in result.validations
        ],
    }

    with Path(output_path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Saved extraction result to: %s", output_path)
    return output_path


def _build_validation_summary(report: ValidationReport) -> dict:
    """Build validation summary dictionary from report."""
    return {
        "sum_validations": len(report.sum_validations),
        "pdf_xbrl_validations": len(report.pdf_xbrl_validations),
        "cross_validations": len(report.cross_validations),
        "sum_passed": sum(1 for v in report.sum_validations if v.match),
        "pdf_xbrl_passed": sum(1 for v in report.pdf_xbrl_validations if v.match),
        "cross_passed": sum(1 for v in report.cross_validations if v.match),
    }


def save_extraction_result(
    data_or_result: Sheet1Data | ExtractionResult,
    year_or_output_dir: int | Path | None = None,
    quarter: int | None = None,
    report: ValidationReport | None = None,
) -> Path:
    """Save extraction result to JSON file.

    Supports two signatures for backward compatibility:
    1. save_extraction_result(Sheet1Data, year, quarter, report) - new style
    2. save_extraction_result(ExtractionResult, output_dir) - legacy style
    """
    # Legacy signature: save_extraction_result(ExtractionResult, output_dir)
    if isinstance(data_or_result, ExtractionResult):
        output_dir = year_or_output_dir if isinstance(year_or_output_dir, Path) else None
        return _save_legacy_extraction_result(data_or_result, output_dir)

    # New signature: save_extraction_result(Sheet1Data, year, quarter, report)
    data = data_or_result
    year = year_or_output_dir if isinstance(year_or_output_dir, int) else 0
    if quarter is None:
        msg = "save_extraction_result() missing required argument: 'quarter'"
        raise TypeError(msg)

    paths = get_period_paths(year, quarter)
    output_dir = paths["processed"]
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"sheet1_{quarter_to_roman(quarter)}Q{year}.json"

    result_dict = asdict(data)
    if report:
        result_dict["_validation_summary"] = _build_validation_summary(report)

    with Path(output_path).open("w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Saved extraction result to %s", output_path)
    return output_path


def print_extraction_report(
    data: Sheet1Data,
    report: ValidationReport | None = None,
    detailed: bool = False,
) -> None:
    """Print extraction report to console."""
    if report:
        pass

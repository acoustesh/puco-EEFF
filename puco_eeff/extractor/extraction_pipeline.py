"""High-level extraction orchestration for Sheet1 data.

This module provides the main entry points for extracting Sheet1 data from
PDF and XBRL files.

Key Functions:
    extract_sheet1(): Main entry point for Sheet1 extraction.
    extract_sheet1_from_xbrl(): Extract Sheet1 totals directly from XBRL.
    extract_sheet1_from_analisis_razonado(): Extract Sheet1 from PDF.
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
    "extract_sheet1_from_analisis_razonado",
    "extract_sheet1_from_xbrl",
    "print_extraction_report",
    "print_sheet1_report",
    "save_extraction_result",
    # Re-exports
    "save_sheet1_data",
]


# =============================================================================
# High-Level Extraction Functions
# =============================================================================


def _determine_source(raw_dir: Path, year: int, quarter: int) -> str:
    """Determine the data source based on available files."""
    combined_path = find_file_with_alternatives(raw_dir, "pucobre_combined", year, quarter)
    return "pucobre.cl" if (combined_path and combined_path.exists()) else "cmf"


def _extract_nota_sections(pdf_path: Path, result: ExtractionResult) -> None:
    """Extract Nota 21 and 22 sections into result."""
    for nota_name in ("nota_21", "nota_22"):
        section = extract_pdf_section(pdf_path, nota_name)
        if section is not None:
            result.sections[nota_name] = section


def _populate_xbrl_totals(result: ExtractionResult, xbrl_totals: dict[str, int | None]) -> None:
    """Populate result.xbrl_totals from XBRL data."""
    result_key_mapping = get_sheet1_result_key_mapping()
    for xbrl_key in result_key_mapping.values():
        if xbrl_key in xbrl_totals:
            result.xbrl_totals[xbrl_key] = xbrl_totals.get(xbrl_key)


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
        _populate_xbrl_totals(result, xbrl_totals)

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


def extract_detailed_costs(year: int, quarter: int, validate: bool = True) -> ExtractionResult:
    """Extract detailed cost breakdowns for a period."""
    paths = get_period_paths(year, quarter)
    raw_dir = paths["raw_pdf"]

    pdf_path = _resolve_pdf_path(raw_dir, year, quarter)
    if not pdf_path:
        logger.error("PDF not found in: %s", raw_dir)
        return ExtractionResult(year=year, quarter=quarter, xbrl_available=False)

    source = _determine_source(raw_dir, year, quarter)
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


def _extract_ingresos_fallback(data: Sheet1Data, ef_path: Path) -> ValidationReport:
    """Extract Ingresos from PDF and return sum-only validation report."""
    logger.info("No XBRL available, extracting Ingresos from Estado de Resultados")
    ingresos_value = extract_ingresos_from_pdf(ef_path)
    if ingresos_value is not None:
        data.ingresos_ordinarios = ingresos_value
        logger.info(f"Set Ingresos from PDF: {ingresos_value:,}")
    else:
        logger.warning("Could not extract Ingresos from PDF")

    sum_results = _run_sum_validations(data)
    return ValidationReport(sum_validations=sum_results)


def _wrap_result(
    data: Sheet1Data | None,
    report: ValidationReport | None,
    return_report: bool,
) -> Sheet1Data | None | tuple[Sheet1Data | None, ValidationReport | None]:
    """Wrap extraction result based on return_report flag."""
    return (data, report) if return_report else data


def _extract_and_populate_notas(data: Sheet1Data, ef_path: Path) -> bool:
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
            # Inline: populate Sheet1Data from the Nota section
            data.set_value(total_field, nota.total_ytd_actual)
            for item in nota.items:
                _map_nota_item_to_sheet1(item, data, section_name)
            any_extracted = True

    return any_extracted


def _create_sheet1_data(year: int, quarter: int, source: str, xbrl_available: bool) -> Sheet1Data:
    """Create a new Sheet1Data instance with common initialization."""
    return Sheet1Data(
        quarter=format_quarter_label(year, quarter),
        year=year,
        quarter_num=quarter,
        source=source,
        xbrl_available=xbrl_available,
    )


def extract_sheet1_from_analisis_razonado(
    year: int,
    quarter: int,
    validate_with_xbrl: bool = True,
    return_report: bool = False,
) -> Sheet1Data | None | tuple[Sheet1Data | None, ValidationReport | None]:
    """Extract Sheet1 data from Estados Financieros PDF.

    This function extracts cost breakdown data from Nota 21 and Nota 22
    sections of the Estados Financieros PDF, optionally validating against
    XBRL data if available.
    """
    period_paths = get_period_paths(year, quarter)
    raw_pdf_dir = period_paths["raw_pdf"]

    ef_pdf_path = _resolve_pdf_path(raw_pdf_dir, year, quarter)
    if ef_pdf_path is None:
        logger.warning("Estados Financieros PDF not found in %s", raw_pdf_dir)
        return _wrap_result(None, None, return_report)

    extraction_source = _determine_source(raw_pdf_dir, year, quarter)
    xbrl_file_path, has_xbrl = _resolve_xbrl_path(period_paths["raw_xbrl"], year, quarter)

    sheet1_data = _create_sheet1_data(year, quarter, extraction_source, has_xbrl)

    notas_extracted = _extract_and_populate_notas(sheet1_data, ef_pdf_path)
    if not notas_extracted:
        logger.error("Could not extract Nota 21 or 22 from %s", ef_pdf_path)
        return _wrap_result(None, None, return_report)

    can_validate_with_xbrl = has_xbrl and validate_with_xbrl and xbrl_file_path is not None
    if can_validate_with_xbrl:
        xbrl_fact_totals = extract_xbrl_totals(xbrl_file_path)  # type: ignore[arg-type]
        validation_report = run_sheet1_validations(sheet1_data, xbrl_fact_totals)
    else:
        validation_report = _extract_ingresos_fallback(sheet1_data, ef_pdf_path)

    return _wrap_result(sheet1_data, validation_report, return_report)


def _map_nota_item_to_sheet1(item: LineItem, data: Sheet1Data, section_name: str) -> None:
    """Map a Nota line item to Sheet1Data fields using config-driven matching."""
    field_name = match_concepto_to_field(item.concepto, section_name)

    if field_name:
        data.set_value(field_name, item.ytd_actual)
        logger.debug(f"Mapped '{item.concepto}' -> {field_name} = {item.ytd_actual}")
    else:
        logger.warning(f"Could not map item from {section_name}: '{item.concepto}'")


# XBRL field to Sheet1Data field mapping
_XBRL_TO_SHEET1_FIELDS = {
    "ingresos_de_actividades_ordinarias": "ingresos_ordinarios",
    "costo_de_ventas": "total_costo_venta",
    "gastos_de_administracion": "total_gasto_admin",
}


def extract_sheet1_from_xbrl(
    year: int,
    quarter: int,
    return_report: bool = False,
) -> Sheet1Data | None | tuple[Sheet1Data | None, ValidationReport | None]:
    """Extract Sheet1 data directly from XBRL file only (no PDF extraction).

    Bypasses PDF processing, extracting high-level totals from structured XBRL.
    Use when PDF is unavailable or only summary totals are needed.
    """
    paths = get_period_paths(year, quarter)
    xbrl_path = find_file_with_alternatives(paths["raw_xbrl"], "estados_financieros_xbrl", year, quarter)
    if not xbrl_path:
        xbrl_path = paths["raw_xbrl"] / format_filename("estados_financieros_xbrl", year, quarter)

    if not xbrl_path.exists():
        logger.warning("XBRL file not found for %dQ%d", year, quarter)
        return (None, None) if return_report else None

    xbrl_totals = extract_xbrl_totals(xbrl_path)
    if not xbrl_totals:
        logger.error("Could not extract totals from XBRL: %s", xbrl_path)
        return (None, None) if return_report else None

    data = _create_sheet1_data(year, quarter, "xbrl", xbrl_available=True)
    for xbrl_field, sheet1_field in _XBRL_TO_SHEET1_FIELDS.items():
        if (value := xbrl_totals.get(xbrl_field)) is not None:
            data.set_value(sheet1_field, value)
            logger.debug("XBRL: %s = %s", sheet1_field, value)

    if not return_report:
        return data
    return data, ValidationReport(sum_validations=_run_sum_validations(data))


def _unpack_extraction_result(
    result: Sheet1Data | None | tuple[Sheet1Data | None, ValidationReport | None],
) -> tuple[Sheet1Data | None, ValidationReport | None]:
    """Unpack extraction result into (data, report) tuple."""
    if isinstance(result, tuple):
        return result
    return result, None


def _orchestrate_extraction(
    year: int,
    quarter: int,
    prefer_source: str,
    merge_sources: bool,
) -> tuple[Sheet1Data | None, ValidationReport | None]:
    """Internal orchestration of multi-source extraction with fallback and optional merge."""
    extractors = {
        "pdf": lambda: extract_sheet1_from_analisis_razonado(year, quarter, validate_with_xbrl=True, return_report=True),
        "xbrl": lambda: extract_sheet1_from_xbrl(year, quarter, return_report=True),
    }
    primary, secondary = (prefer_source, "xbrl" if prefer_source == "pdf" else "pdf")
    data, report = _unpack_extraction_result(extractors[primary]())

    if data is None:
        logger.info("%s extraction failed, trying %s extraction", primary.upper(), secondary.upper())
        fallback = (
            lambda: extract_sheet1_from_xbrl(year, quarter, return_report=True)
            if secondary == "xbrl"
            else extract_sheet1_from_analisis_razonado(year, quarter, validate_with_xbrl=False, return_report=True)
        )
        return _unpack_extraction_result(fallback())

    if primary == "xbrl" and merge_sources:
        pdf_data, _ = _unpack_extraction_result(
            extract_sheet1_from_analisis_razonado(
                year,
                quarter,
                validate_with_xbrl=False,
                return_report=True,
            ),
        )
        if pdf_data is not None:
            data = _merge_pdf_into_xbrl_data(data, pdf_data)
            report = ValidationReport(sum_validations=_run_sum_validations(data))

    return data, report


def extract_sheet1(
    year: int,
    quarter: int,
    prefer_source: str = "pdf",
    merge_sources: bool = True,
    return_report: bool = False,
) -> Sheet1Data | None | tuple[Sheet1Data | None, ValidationReport | None]:
    """High-level API: extract Sheet1 data from available sources with optional merge and fallback."""
    result = _orchestrate_extraction(year, quarter, prefer_source, merge_sources)
    return result if return_report else result[0]


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
                "pdf_value": v.pdf_value,
                "xbrl_value": v.xbrl_value,
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

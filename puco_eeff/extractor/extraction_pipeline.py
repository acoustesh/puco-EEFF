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
    "extract_sheet1",
    "extract_sheet1_from_xbrl",
    "extract_sheet1_from_analisis_razonado",
    "extract_detailed_costs",
    "save_extraction_result",
    "print_extraction_report",
    # Re-exports
    "save_sheet1_data",
    "print_sheet1_report",
]


# =============================================================================
# High-Level Extraction Functions
# =============================================================================


def extract_detailed_costs(year: int, quarter: int, validate: bool = True) -> ExtractionResult:
    """Extract detailed cost breakdowns for a period."""
    paths = get_period_paths(year, quarter)
    raw_dir = paths["raw_pdf"]

    pdf_path = find_file_with_alternatives(raw_dir, "estados_financieros_pdf", year, quarter)
    if not pdf_path:
        pdf_path = raw_dir / format_filename("estados_financieros_pdf", year, quarter)

    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return ExtractionResult(year=year, quarter=quarter, xbrl_available=False)

    combined_path = find_file_with_alternatives(raw_dir, "pucobre_combined", year, quarter)
    source = "pucobre.cl" if (combined_path and combined_path.exists()) else "cmf"

    result = ExtractionResult(year=year, quarter=quarter, source=source, pdf_path=pdf_path)

    nota_21 = extract_pdf_section(pdf_path, "nota_21")
    nota_22 = extract_pdf_section(pdf_path, "nota_22")
    if nota_21 is not None:
        result.sections["nota_21"] = nota_21
    if nota_22 is not None:
        result.sections["nota_22"] = nota_22

    xbrl_dir = paths["raw_xbrl"]
    xbrl_path = find_file_with_alternatives(xbrl_dir, "estados_financieros_xbrl", year, quarter)
    if not xbrl_path:
        xbrl_path = xbrl_dir / format_filename("estados_financieros_xbrl", year, quarter)

    if xbrl_path and xbrl_path.exists():
        result.xbrl_available = True
        result.xbrl_path = xbrl_path

        if validate:
            xbrl_totals = extract_xbrl_totals(xbrl_path)
            result_key_mapping = get_sheet1_result_key_mapping()
            for _field_name, xbrl_key in result_key_mapping.items():
                if xbrl_key in xbrl_totals:
                    result.xbrl_totals[xbrl_key] = xbrl_totals.get(xbrl_key)

            sheet1_data = sections_to_sheet1data(result.sections, year, quarter)
            sheet1_data.xbrl_available = True
            sheet1_data.source = source

            report = run_sheet1_validations(sheet1_data, xbrl_totals, use_xbrl_fallback=True)
            result.validations = report.pdf_xbrl_validations
            result.validation_report = report
    else:
        logger.info(f"No XBRL available for {year} Q{quarter} - using PDF only")
        result.xbrl_available = False

        if validate:
            sheet1_data = sections_to_sheet1data(result.sections, year, quarter)
            sheet1_data.xbrl_available = False
            sheet1_data.source = source

            report = run_sheet1_validations(
                sheet1_data, None, run_sum_validations=True, run_pdf_xbrl_validations=True, run_cross_validations=False
            )
            result.validations = report.pdf_xbrl_validations
            result.validation_report = report

    return result


def extract_sheet1_from_analisis_razonado(
    year: int,
    quarter: int,
    validate_with_xbrl: bool = True,
    return_report: bool = False,
) -> Sheet1Data | None | tuple[Sheet1Data | None, ValidationReport | None]:
    """Extract Sheet1 data from Estados Financieros PDF."""
    paths = get_period_paths(year, quarter)
    raw_dir = paths["raw_pdf"]

    ef_path = find_file_with_alternatives(raw_dir, "estados_financieros_pdf", year, quarter)
    if not ef_path:
        ef_path = raw_dir / format_filename("estados_financieros_pdf", year, quarter)

    if not ef_path.exists():
        logger.warning(f"Estados Financieros PDF not found: {ef_path}")
        return (None, None) if return_report else None

    combined_path = find_file_with_alternatives(raw_dir, "pucobre_combined", year, quarter)
    source = "pucobre.cl" if (combined_path and combined_path.exists()) else "cmf"

    xbrl_path = find_file_with_alternatives(paths["raw_xbrl"], "estados_financieros_xbrl", year, quarter)
    if not xbrl_path:
        xbrl_path = paths["raw_xbrl"] / format_filename("estados_financieros_xbrl", year, quarter)
    xbrl_available = xbrl_path.exists() if xbrl_path else False

    data = Sheet1Data(
        quarter=format_quarter_label(year, quarter),
        year=year,
        quarter_num=quarter,
        source=source,
        xbrl_available=xbrl_available,
    )

    nota_21 = extract_pdf_section(ef_path, "nota_21")
    nota_22 = extract_pdf_section(ef_path, "nota_22")

    if nota_21 is None and nota_22 is None:
        logger.error(f"Could not extract Nota 21 or 22 from {ef_path}")
        return (None, None) if return_report else None

    if nota_21:
        data.total_costo_venta = nota_21.total_ytd_actual
        for item in nota_21.items:
            _map_nota_item_to_sheet1(item, data, "nota_21")

    if nota_22:
        data.total_gasto_admin = nota_22.total_ytd_actual
        for item in nota_22.items:
            _map_nota_item_to_sheet1(item, data, "nota_22")

    report: ValidationReport | None = None

    if xbrl_available and validate_with_xbrl:
        report = _validate_sheet1_with_xbrl(data, xbrl_path)
    else:
        logger.info("No XBRL available, extracting Ingresos from Estado de Resultados")
        ingresos_value = extract_ingresos_from_pdf(ef_path)
        if ingresos_value is not None:
            data.ingresos_ordinarios = ingresos_value
            logger.info(f"Set Ingresos from PDF: {ingresos_value:,}")
        else:
            logger.warning("Could not extract Ingresos from PDF")

        sum_results = _run_sum_validations(data)
        report = ValidationReport(sum_validations=sum_results)

    return (data, report) if return_report else data


def _map_nota_item_to_sheet1(item: LineItem, data: Sheet1Data, section_name: str) -> None:
    """Map a Nota line item to Sheet1Data fields using config-driven matching."""
    field_name = match_concepto_to_field(item.concepto, section_name)

    if field_name:
        data.set_value(field_name, item.ytd_actual)
        logger.debug(f"Mapped '{item.concepto}' -> {field_name} = {item.ytd_actual}")
    else:
        logger.warning(f"Could not map item from {section_name}: '{item.concepto}'")


def _validate_sheet1_with_xbrl(data: Sheet1Data, xbrl_path: Path) -> ValidationReport:
    """Validate and supplement Sheet1 data with XBRL totals."""
    xbrl_totals = extract_xbrl_totals(xbrl_path)
    return run_sheet1_validations(data, xbrl_totals)


def extract_sheet1_from_xbrl(
    year: int,
    quarter: int,
    return_report: bool = False,
) -> Sheet1Data | None | tuple[Sheet1Data | None, ValidationReport | None]:
    """Extract Sheet1 data from XBRL file."""
    paths = get_period_paths(year, quarter)
    xbrl_path = find_file_with_alternatives(paths["raw_xbrl"], "estados_financieros_xbrl", year, quarter)
    if not xbrl_path:
        xbrl_path = paths["raw_xbrl"] / format_filename("estados_financieros_xbrl", year, quarter)

    if not xbrl_path.exists():
        logger.warning(f"XBRL file not found: {xbrl_path}")
        return (None, None) if return_report else None

    xbrl_totals = extract_xbrl_totals(xbrl_path)
    if not xbrl_totals:
        logger.error(f"Could not extract totals from XBRL: {xbrl_path}")
        return (None, None) if return_report else None

    data = Sheet1Data(
        quarter=format_quarter_label(year, quarter),
        year=year,
        quarter_num=quarter,
        source="xbrl",
        xbrl_available=True,
    )

    field_mapping = {
        "ingresos_ordinarios": "ingresos_de_actividades_ordinarias",
        "total_costo_venta": "costo_de_ventas",
        "total_gasto_admin": "gastos_de_administracion",
    }

    for sheet1_field, xbrl_field in field_mapping.items():
        value = xbrl_totals.get(xbrl_field)
        if value is not None:
            data.set_value(sheet1_field, value)
            logger.debug(f"XBRL: {sheet1_field} = {value}")

    report: ValidationReport | None = None
    if return_report:
        sum_results = _run_sum_validations(data)
        report = ValidationReport(sum_validations=sum_results)

    return (data, report) if return_report else data


def extract_sheet1(
    year: int,
    quarter: int,
    prefer_source: str = "pdf",
    merge_sources: bool = True,
    return_report: bool = False,
) -> Sheet1Data | None | tuple[Sheet1Data | None, ValidationReport | None]:
    """
    Extract Sheet1 data from available sources.

    Args:
        year: Year
        quarter: Quarter number (1-4)
        prefer_source: "pdf" or "xbrl" - which source to prefer
        merge_sources: If True, merge PDF data into XBRL data
        return_report: If True, return (data, report) tuple

    Returns:
        Sheet1Data or None, optionally with ValidationReport
    """
    pdf_data: Sheet1Data | None = None
    xbrl_data: Sheet1Data | None = None
    report: ValidationReport | None = None

    if prefer_source == "pdf":
        result = extract_sheet1_from_analisis_razonado(year, quarter, validate_with_xbrl=True, return_report=True)
        if isinstance(result, tuple):
            pdf_data, report = result
        else:
            pdf_data = result

        if pdf_data is not None:
            return (pdf_data, report) if return_report else pdf_data

        logger.info("PDF extraction failed, trying XBRL-only extraction")
        result = extract_sheet1_from_xbrl(year, quarter, return_report=True)
        if isinstance(result, tuple):
            xbrl_data, report = result
        else:
            xbrl_data = result
        return (xbrl_data, report) if return_report else xbrl_data

    else:
        result = extract_sheet1_from_xbrl(year, quarter, return_report=True)
        if isinstance(result, tuple):
            xbrl_data, report = result
        else:
            xbrl_data = result

        if xbrl_data is None:
            logger.info("XBRL extraction failed, trying PDF extraction")
            result = extract_sheet1_from_analisis_razonado(year, quarter, validate_with_xbrl=False, return_report=True)
            if isinstance(result, tuple):
                pdf_data, report = result
            else:
                pdf_data = result
            return (pdf_data, report) if return_report else pdf_data

        if merge_sources:
            result = extract_sheet1_from_analisis_razonado(year, quarter, validate_with_xbrl=False, return_report=True)
            if isinstance(result, tuple):
                pdf_data, _ = result
            else:
                pdf_data = result

            if pdf_data is not None:
                xbrl_data = _merge_pdf_into_xbrl_data(xbrl_data, pdf_data)
                # Skip XBRL validation since we don't have the path
                sum_results = _run_sum_validations(xbrl_data)
                report = ValidationReport(sum_validations=sum_results)

        return (xbrl_data, report) if return_report else xbrl_data


def _merge_pdf_into_xbrl_data(xbrl_data: Sheet1Data, pdf_data: Sheet1Data) -> Sheet1Data:
    """Merge detailed PDF data into XBRL data."""
    for field_name in pdf_data.__dataclass_fields__:
        if field_name in ("quarter", "year", "quarter_num", "source", "xbrl_available"):
            continue

        pdf_value = getattr(pdf_data, field_name, None)
        xbrl_value = getattr(xbrl_data, field_name, None)

        if pdf_value is not None and xbrl_value is None:
            setattr(xbrl_data, field_name, pdf_value)
            logger.debug(f"Merged PDF field {field_name} = {pdf_value} into XBRL data")

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
        result = data_or_result
        output_dir = year_or_output_dir if isinstance(year_or_output_dir, Path) else None

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

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved extraction result to: {output_path}")
        return output_path

    # New signature: save_extraction_result(Sheet1Data, year, quarter, report)
    data = data_or_result
    year = year_or_output_dir if isinstance(year_or_output_dir, int) else 0
    if quarter is None:
        raise TypeError("save_extraction_result() missing required argument: 'quarter'")

    paths = get_period_paths(year, quarter)
    output_dir = paths["processed"]
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"sheet1_{quarter_to_roman(quarter)}Q{year}.json"

    result_dict = asdict(data)
    if report:
        result_dict["_validation_summary"] = {
            "sum_validations": len(report.sum_validations),
            "pdf_xbrl_validations": len(report.pdf_xbrl_validations),
            "cross_validations": len(report.cross_validations),
            "sum_passed": sum(1 for v in report.sum_validations if v.match),
            "pdf_xbrl_passed": sum(1 for v in report.pdf_xbrl_validations if v.match),
            "cross_passed": sum(1 for v in report.cross_validations if v.match),
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Saved extraction result to {output_path}")
    return output_path


def print_extraction_report(
    data: Sheet1Data,
    report: ValidationReport | None = None,
    detailed: bool = False,
) -> None:
    """Print extraction report to console."""
    from puco_eeff.extractor.validation_core import format_validation_report

    print(f"\n{'=' * 60}")
    print(f"Sheet1 Extraction Report: {data.quarter}")
    print(f"{'=' * 60}")
    print(f"Source: {data.source}")
    print(f"XBRL Available: {data.xbrl_available}")
    print()

    print("Key Totals:")
    print(
        f"  Ingresos Ordinarios: {data.ingresos_ordinarios:,.0f}"
        if data.ingresos_ordinarios
        else "  Ingresos Ordinarios: N/A"
    )
    print(
        f"  Total Costo Venta: {data.total_costo_venta:,.0f}" if data.total_costo_venta else "  Total Costo Venta: N/A"
    )
    print(
        f"  Total Gasto Admin: {data.total_gasto_admin:,.0f}" if data.total_gasto_admin else "  Total Gasto Admin: N/A"
    )

    if report:
        print()
        print(format_validation_report(report))

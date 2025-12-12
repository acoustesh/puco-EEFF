#!/usr/bin/env python3
"""Sheet3 orchestrator - download files if needed, then extract and save.

This module orchestrates the complete Sheet3 (Estado de Resultados) extraction workflow:
1. Download files from CMF/Pucobre if needed
2. Extract data from XBRL (primary) or PDF (fallback)
3. Run validation against reference data
4. Save extracted data to JSON
5. Print formatted report

Usage (from project root):
    cd /path/to/puco-EEFF
    python -m puco_eeff.main_sheet3 -y 2024 -q 2
    python -m puco_eeff.main_sheet3 -y 2024 -q 2 --skip-download
    python -m puco_eeff.main_sheet3 -y 2024 -q 2 --no-save --quiet

    # With validation options:
    python -m puco_eeff.main_sheet3 -y 2024 -q 2 --validate-reference
    python -m puco_eeff.main_sheet3 -y 2024 -q 2 --fail-on-reference-mismatch

CLI Flags:
    --year, -y          Year to process (required)
    --quarter, -q       Quarter(s) to process (default: all 4)
    --skip-download, -s Use existing files, skip download
    --no-save           Don't save JSON output
    --quiet             Suppress report output
    --no-headless       Show browser window during download
    --validate-reference Enable reference data validation (opt-in)
    --fail-on-reference-mismatch Exit with error code if reference validation fails
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path when running directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from puco_eeff.config import (  # noqa: E402
    DATA_DIR,
    find_file_with_alternatives,
    get_period_paths,
    setup_logging,
)
from puco_eeff.sheets.sheet3 import (  # noqa: E402
    Sheet3Data,
    extract_sheet3,
    save_sheet3_to_json,
    validate_sheet3_against_reference,
)

logger = setup_logging(__name__)


# =============================================================================
# File Existence Check
# =============================================================================


def files_exist_for_period(year: int, quarter: int, require_xbrl: bool = False) -> bool:
    """Determine whether prerequisite files are already on disk.

    Parameters
    ----------
    year : int
        Fiscal year to inspect.
    quarter : int
        Quarter number (1–4).
    require_xbrl : bool, optional
        When ``True`` also require the XBRL file alongside the PDF.

    Returns
    -------
    bool
        ``True`` when all required files are present.
    """
    paths = get_period_paths(year, quarter)

    # For Sheet3, we need either XBRL or PDF
    xbrl_available = False
    pdf_available = False

    # Check XBRL
    raw_xbrl = paths.get("raw_xbrl")
    if raw_xbrl and raw_xbrl.exists():
        for xml_file in raw_xbrl.glob("*.xml"):
            xbrl_available = True
            break

    # Check PDF
    pdf_path = find_file_with_alternatives(
        paths.get("raw_pdf"),
        "estados_financieros_pdf",
        year,
        quarter,
    )
    if pdf_path is not None:
        pdf_available = True

    if require_xbrl:
        return xbrl_available

    return xbrl_available or pdf_available


# =============================================================================
# Download Orchestration
# =============================================================================


def ensure_files_downloaded(
    year: int,
    quarter: int,
    headless: bool = True,
    force: bool = False,
    skip_download: bool = False,
) -> bool:
    """Guarantee presence of raw XBRL/PDF assets for a period.

    Parameters
    ----------
    year : int
        Fiscal year to download.
    quarter : int
        Quarter number (1–4).
    headless : bool, optional
        Whether to run Playwright downloads without a visible browser.
    force : bool, optional
        Download even if files already exist.
    skip_download : bool, optional
        Only check for presence; do not fetch missing files.

    Returns
    -------
    bool
        ``True`` when required files exist or were downloaded successfully.
    """
    files_present = files_exist_for_period(year, quarter)

    # Skip download mode: just check existence
    if skip_download:
        if not files_present:
            logger.error("Files not found for %s Q%s and download skipped", year, quarter)
        return files_present

    # Check if files already exist
    if not force and files_present:
        logger.info("Files already exist for %s Q%s, skipping download", year, quarter)
        return True

    # Import here to avoid loading playwright when not needed
    from puco_eeff.scraper.cmf_downloader import download_all_documents

    logger.info("Downloading files for %s Q%s...", year, quarter)
    results = download_all_documents(year, quarter, headless=headless)

    # Check if at least one source was successful
    any_success = any(r.success for r in results)

    if not any_success:
        logger.error("Failed to download any files for %s Q%s", year, quarter)
        for r in results:
            if not r.success and r.error:
                logger.error(f"  {r.document_type}: {r.error}")
        return False

    # Log results
    for r in results:
        status = "✓" if r.success else "✗"
        size = f" ({r.file_size:,} bytes)" if r.file_size else ""
        source = f" [{r.source}]" if r.source != "cmf" else ""
        logger.info(f"  {status} {r.document_type}{size}{source}")

    return True


# =============================================================================
# Report Printing
# =============================================================================


def print_sheet3_report(data: Sheet3Data) -> None:
    """Print a formatted report of Sheet3 data.

    Parameters
    ----------
    data
        Sheet3Data to display.
    """
    print("\n" + "=" * 70)
    print(f"  SHEET3 - Estado de Resultados: {data.quarter}")
    print("=" * 70)

    print(f"\n  Source: {data.source.upper()}")
    print(f"  Period Type: {data.period_type}")
    if data.xbrl_available:
        print("  XBRL: Available ✓")
    else:
        print("  XBRL: Not Available (PDF fallback)")

    print("\n  INCOME STATEMENT (MUS$)")
    print("  " + "-" * 50)

    fields = [
        ("Ingresos de actividades ordinarias", data.ingresos_ordinarios),
        ("Costo de ventas", data.costo_ventas),
        ("Ganancia bruta", data.ganancia_bruta),
        ("Otros ingresos, por función", data.otros_ingresos),
        ("Otras ganancias (pérdidas)", data.otros_egresos_funcion),
        ("Ingresos financieros", data.ingresos_financieros),
        ("Gastos de administración y ventas", data.gastos_admin_ventas),
        ("Costos financieros", data.costos_financieros),
        ("Diferencias de cambio", data.diferencias_cambio),
        ("Ganancia antes de impuestos", data.ganancia_antes_impuestos),
        ("Gasto por impuestos", data.gasto_impuestos),
        ("Ganancia del período", data.ganancia_periodo),
        ("Resultado para accionistas", data.resultado_accionistas),
    ]

    for label, value in fields:
        if value is not None:
            print(f"  {label:<40} {value:>12,}")
        else:
            print(f"  {label:<40} {'N/A':>12}")

    print("\n  SHARE DATA")
    print("  " + "-" * 50)

    if data.acciones_emitidas is not None:
        print(f"  {'Acciones emitidas':<40} {data.acciones_emitidas:>12,}")
    if data.acciones_dividendo is not None:
        print(f"  {'Acciones con derecho a dividendo':<40} {data.acciones_dividendo:>12,}")

    print("\n  EARNINGS PER SHARE (US$)")
    print("  " + "-" * 50)
    if data.ganancia_por_accion is not None:
        print(f"  {'Ganancia por acción básica':<40} {data.ganancia_por_accion:>12.5f}")

    if data.issues:
        print("\n  ISSUES")
        print("  " + "-" * 50)
        for issue in data.issues:
            print(f"  ⚠ {issue}")

    print("\n" + "=" * 70 + "\n")


# =============================================================================
# Main Processing
# =============================================================================


def process_sheet3(
    year: int,
    quarter: int,
    skip_download: bool = False,
    save: bool = True,
    verbose: bool = True,
    headless: bool = True,
    validate_reference: bool = False,
    fail_on_reference_mismatch: bool = False,
) -> tuple[Sheet3Data | None, list[str]]:
    """Run the end-to-end Sheet3 workflow for a period.

    Parameters
    ----------
    year : int
        Fiscal year to process.
    quarter : int
        Quarter number (1–4).
    skip_download : bool, optional
        Skip network fetches; fail if inputs are missing.
    save : bool, optional
        Persist extracted data to JSON when ``True``.
    verbose : bool, optional
        Print a human-readable report when ``True``.
    headless : bool, optional
        Run Playwright without UI when downloading.
    validate_reference : bool, optional
        Compare results against curated reference values when available.
    fail_on_reference_mismatch : bool, optional
        Abort on reference mismatches (implies ``validate_reference``).

    Returns
    -------
    tuple[Sheet3Data | None, list[str]]
        Extracted data and validation issues (if any).
    """
    logger.info("Processing Sheet3 for %s Q%s", year, quarter)
    validation_issues: list[str] = []

    # Step 1: Ensure files are available
    if not ensure_files_downloaded(year, quarter, headless=headless, skip_download=skip_download):
        logger.error("Cannot proceed without required files")
        return None, ["No files available"]

    # Step 2: Extract Sheet3 data
    data = extract_sheet3(year, quarter)

    if data.source == "":
        logger.error("Extraction failed for %s Q%s", year, quarter)
        return None, data.issues

    # Step 3: Reference validation (opt-in)
    if validate_reference or fail_on_reference_mismatch:
        is_valid, ref_issues = validate_sheet3_against_reference(data)
        validation_issues.extend(ref_issues)

        if not is_valid:
            logger.warning("=" * 60)
            logger.warning("⚠️  REFERENCE DATA MISMATCH - Values differ from known-good data!")
            logger.warning("=" * 60)
            for issue in ref_issues:
                logger.warning("  • %s", issue)
            logger.warning("=" * 60)

            if fail_on_reference_mismatch:
                logger.error("Reference mismatch and --fail-on-reference-mismatch is set")
                return None, validation_issues
        else:
            logger.info("✓ Reference validation passed for %s", data.quarter)

    # Step 4: Save to JSON
    if save:
        output_path = DATA_DIR / "processed" / f"sheet3_{data.quarter}.json"
        save_sheet3_to_json(data, output_path)
        logger.info("Saved to: %s", output_path)

    # Step 5: Print report
    if verbose:
        print_sheet3_report(data)

    return data, validation_issues


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    """Parse CLI flags and process requested quarters.

    Returns
    -------
    int
        ``0`` when at least one quarter succeeded; ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(
        description="Process Sheet3: download, extract, and save income statement data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m puco_eeff.main_sheet3 --year 2024                    # All quarters
  python -m puco_eeff.main_sheet3 --year 2024 --quarter 2        # Single quarter
  python -m puco_eeff.main_sheet3 --year 2024 -q 2 3             # Multiple quarters
  python -m puco_eeff.main_sheet3 --year 2024 --skip-download
  python -m puco_eeff.main_sheet3 --year 2024 --validate-reference  # With reference check
        """,
    )
    parser.add_argument("--year", "-y", type=int, required=True, help="Year (e.g., 2024)")
    parser.add_argument(
        "--quarter",
        "-q",
        type=int,
        nargs="*",
        choices=[1, 2, 3, 4],
        default=[1, 2, 3, 4],
        help="Quarter(s) to process (default: all 1-4)",
    )
    parser.add_argument(
        "--skip-download",
        "-s",
        action="store_true",
        help="Skip download, use existing files",
    )
    parser.add_argument("--no-save", action="store_true", help="Don't save to JSON")
    parser.add_argument("--quiet", action="store_true", help="Don't print report")
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Show browser window during download",
    )

    # Validation flags
    parser.add_argument(
        "--validate-reference",
        action="store_true",
        help="Run reference data validation (compare to known-good values)",
    )
    parser.add_argument(
        "--fail-on-reference-mismatch",
        action="store_true",
        help="Exit with error if reference validation fails (implies --validate-reference)",
    )

    args = parser.parse_args()

    # --fail-on-reference-mismatch implies --validate-reference
    validate_reference = args.validate_reference or args.fail_on_reference_mismatch

    # Process each quarter
    success_count = 0
    for quarter in args.quarter:
        data, _issues = process_sheet3(
            year=args.year,
            quarter=quarter,
            skip_download=args.skip_download,
            save=not args.no_save,
            verbose=not args.quiet,
            headless=not args.no_headless,
            validate_reference=validate_reference,
            fail_on_reference_mismatch=args.fail_on_reference_mismatch,
        )
        if data is not None:
            success_count += 1

    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

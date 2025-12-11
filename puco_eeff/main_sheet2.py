#!/usr/bin/env python3
"""Sheet2 orchestrator - download files if needed, then extract and save.

This module orchestrates the complete Sheet2 extraction workflow:
1. Download files from CMF/Pucobre if needed (Análisis Razonado + Estados Financieros)
2. Extract data from PDF (Cuadro Resumen KPIs)
3. Run validation (sum checks, reference checks)
4. Save extracted data to JSON
5. Print formatted report

Usage (from project root):
    cd /path/to/puco-EEFF
    python -m puco_eeff.main_sheet2 -y 2024 -q 2
    python -m puco_eeff.main_sheet2 -y 2024 -q 2 --skip-download
    python -m puco_eeff.main_sheet2 -y 2024 -q 2 --no-save --quiet

    # With validation options:
    python -m puco_eeff.main_sheet2 -y 2024 -q 2 --validate-reference
    python -m puco_eeff.main_sheet2 -y 2024 -q 2 --fail-on-sum-mismatch
    python -m puco_eeff.main_sheet2 -y 2024 -q 2 --fail-on-reference-mismatch

CLI Flags:
    --year, -y          Year to process (required)
    --quarter, -q       Quarter(s) to process (default: all 4)
    --skip-download, -s Use existing files, skip download
    --no-save           Don't save JSON output
    --quiet             Suppress report output
    --no-headless       Show browser window during download
    --validate-reference Enable reference data validation (opt-in)
    --fail-on-sum-mismatch Exit with error code if sum validations fail
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
    find_file_with_alternatives,
    get_period_paths,
    setup_logging,
)
from puco_eeff.sheets.sheet2 import (  # noqa: E402
    Sheet2Data,
    extract_sheet2,
    print_sheet2_report,
    save_sheet2_to_json,
    validate_sheet2_against_reference,
)

logger = setup_logging(__name__)


# =============================================================================
# File Existence Check
# =============================================================================


def files_exist_for_period(year: int, quarter: int) -> bool:
    """Determine whether prerequisite files are already on disk.

    For Sheet2, we need either:
    - Análisis Razonado PDF (preferred), OR
    - Estados Financieros PDF (fallback - may contain Análisis Razonado at end)

    Parameters
    ----------
    year : int
        Fiscal year to inspect.
    quarter : int
        Quarter number (1-4).

    Returns
    -------
    bool
        ``True`` when at least one required file is present.
    """
    paths = get_period_paths(year, quarter)

    # Check for Análisis Razonado PDF (preferred)
    ar_path = find_file_with_alternatives(paths["raw_pdf"], "analisis_razonado", year, quarter)
    if ar_path is not None:
        logger.debug("Found Análisis Razonado for %s Q%s", year, quarter)
        return True

    # Check fallback: Estados Financieros PDF
    ef_path = find_file_with_alternatives(
        paths["raw_pdf"], "estados_financieros_pdf", year, quarter
    )
    if ef_path is not None:
        logger.debug("Found Estados Financieros (fallback) for %s Q%s", year, quarter)
        return True

    logger.debug("No PDF found for %s Q%s", year, quarter)
    return False


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
    """Guarantee presence of raw PDF assets for a period.

    Parameters
    ----------
    year : int
        Fiscal year to download.
    quarter : int
        Quarter number (1-4).
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

    # Check if at least one relevant PDF was successful
    pdf_success = any(
        r.success and r.document_type in ("analisis_razonado", "estados_financieros_pdf")
        for r in results
    )

    if not pdf_success:
        logger.error("Failed to download PDF for %s Q%s", year, quarter)
        for r in results:
            if not r.success and r.error:
                logger.error("  %s: %s", r.document_type, r.error)
        return False

    # Log results
    for r in results:
        status = "✓" if r.success else "✗"
        size = f" ({r.file_size:,} bytes)" if r.file_size else ""
        source = f" [{r.source}]" if r.source != "cmf" else ""
        logger.info("  %s %s%s%s", status, r.document_type, size, source)

    return True


# =============================================================================
# Main Processing
# =============================================================================


def _run_reference_validation(
    data: Sheet2Data,
    fail_on_mismatch: bool,
) -> tuple[bool, list[str] | None]:
    """Validate extracted values against reference data.

    Parameters
    ----------
    data : Sheet2Data
        Extracted sheet payload to compare.
    fail_on_mismatch : bool
        When ``True`` aborts on any mismatch.

    Returns
    -------
    tuple[bool, list[str] | None]
        ``(should_continue, issues_or_none)`` where ``should_continue`` reflects
        the strictness flag and whether mismatches were found.
    """
    ref_issues = validate_sheet2_against_reference(data)

    if ref_issues is None:
        logger.info("Reference validation skipped: no verified data for %s", data.quarter)
        return True, ref_issues

    if len(ref_issues) == 0:
        logger.info("✓ Reference validation passed: all values match %s", data.quarter)
        return True, ref_issues

    # Log prominent warning for reference mismatches
    logger.warning("=" * 60)
    logger.warning("⚠️  REFERENCE DATA MISMATCH - Values differ from known-good data!")
    logger.warning("=" * 60)
    for issue in ref_issues:
        logger.warning("  • %s", issue)
    logger.warning("Review extraction or update reference_data.json if values are correct.")
    logger.warning("=" * 60)

    if fail_on_mismatch:
        logger.error("Reference mismatch and --fail-on-reference-mismatch is set")
        return False, ref_issues

    return True, ref_issues


def process_sheet2(
    year: int,
    quarter: int,
    skip_download: bool = False,
    save: bool = True,
    verbose: bool = True,
    headless: bool = True,
    validate_reference: bool = False,
    fail_on_sum_mismatch: bool = False,
    fail_on_reference_mismatch: bool = False,
) -> tuple[Sheet2Data | None, list[str]]:
    """Run the end-to-end Sheet2 workflow for a period.

    Parameters
    ----------
    year : int
        Fiscal year to process.
    quarter : int
        Quarter number (1-4).
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
    fail_on_sum_mismatch : bool, optional
        Abort when sum validations fail.
    fail_on_reference_mismatch : bool, optional
        Abort on reference mismatches (implies ``validate_reference``).

    Returns
    -------
    tuple[Sheet2Data | None, list[str]]
        Extracted data and validation issues when successful; ``(None, [error])``
        on failure or aborted validations.
    """
    logger.info("Processing Sheet2 for %s Q%s", year, quarter)

    # Step 1: Ensure files are available
    if not ensure_files_downloaded(year, quarter, headless=headless, skip_download=skip_download):
        logger.error("Cannot proceed without required files")
        return None, ["Required files not found"]

    # Step 2: Extract Sheet2 data (includes sum validations)
    data, validation_issues = extract_sheet2(year, quarter)
    if data is None:
        logger.error("Extraction failed for %s Q%s", year, quarter)
        return None, validation_issues

    # Step 3: Check sum validation failures (if strict mode)
    if fail_on_sum_mismatch and validation_issues:
        logger.error("Sum validation failed and --fail-on-sum-mismatch is set")
        return None, validation_issues

    # Step 4: Reference validation (opt-in)
    if validate_reference or fail_on_reference_mismatch:
        should_continue, ref_issues = _run_reference_validation(data, fail_on_reference_mismatch)
        if ref_issues:
            validation_issues.extend(ref_issues)
        if not should_continue:
            return None, validation_issues

    # Step 5: Save to JSON
    if save:
        output_path = save_sheet2_to_json(data)
        logger.info("Saved to: %s", output_path)

    # Step 6: Print report
    if verbose:
        print_sheet2_report(data)

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
        description="Process Sheet2: download, extract, and save KPI data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m puco_eeff.main_sheet2 --year 2024                    # All quarters
  python -m puco_eeff.main_sheet2 --year 2024 --quarter 2        # Single quarter
  python -m puco_eeff.main_sheet2 --year 2024 -q 2 3             # Multiple quarters
  python -m puco_eeff.main_sheet2 --year 2024 --skip-download
  python -m puco_eeff.main_sheet2 --year 2024 --validate-reference  # With reference check
  python -m puco_eeff.main_sheet2 --year 2024 --fail-on-sum-mismatch  # Strict mode for CI
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
        "--skip-download", "-s", action="store_true", help="Skip download, use existing files"
    )
    parser.add_argument("--no-save", action="store_true", help="Don't save to JSON")
    parser.add_argument("--quiet", action="store_true", help="Don't print report")
    parser.add_argument(
        "--no-headless", action="store_true", help="Show browser window during download"
    )

    # Validation flags
    parser.add_argument(
        "--validate-reference",
        action="store_true",
        help="Run reference data validation (compare to known-good values)",
    )
    parser.add_argument(
        "--fail-on-sum-mismatch",
        action="store_true",
        help="Exit with error if sum validation fails (for CI)",
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
        data, _issues = process_sheet2(
            year=args.year,
            quarter=quarter,
            skip_download=args.skip_download,
            save=not args.no_save,
            verbose=not args.quiet,
            headless=not args.no_headless,
            validate_reference=validate_reference,
            fail_on_sum_mismatch=args.fail_on_sum_mismatch,
            fail_on_reference_mismatch=args.fail_on_reference_mismatch,
        )
        if data is not None:
            success_count += 1

    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

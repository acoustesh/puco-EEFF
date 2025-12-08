#!/usr/bin/env python3
"""Sheet1 orchestrator - download files if needed, then extract and save.

This module orchestrates the complete Sheet1 extraction workflow:
1. Download files from CMF/Pucobre if needed
2. Extract data from PDF (Nota 21/22) and XBRL
3. Run validation (sum checks, cross-validations, reference checks)
4. Save extracted data to JSON
5. Print formatted report

Usage:
    python -m puco_eeff.main_sheet1 --year 2024 --quarter 2
    python -m puco_eeff.main_sheet1 --year 2024 --quarter 2 --skip-download
    python -m puco_eeff.main_sheet1 --year 2024 --quarter 2 --no-save --quiet

    # With validation options:
    python -m puco_eeff.main_sheet1 -y 2024 -q 2 --validate-reference
    python -m puco_eeff.main_sheet1 -y 2024 -q 2 --fail-on-sum-mismatch
    python -m puco_eeff.main_sheet1 -y 2024 -q 2 --fail-on-reference-mismatch

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
from puco_eeff.extractor import (  # noqa: E402
    ValidationReport,
    extract_sheet1,
)
from puco_eeff.sheets.sheet1 import (  # noqa: E402
    Sheet1Data,
    print_sheet1_report,
    save_sheet1_data,
    validate_sheet1_against_reference,
)

logger = setup_logging(__name__)


# =============================================================================
# File Existence Check
# =============================================================================


def files_exist_for_period(year: int, quarter: int, require_xbrl: bool = False) -> bool:
    """Check if required files exist for a period.

    Args:
        year: Year of the financial statement
        quarter: Quarter (1-4)
        require_xbrl: If True, also require XBRL file to exist

    Returns:
        True if all required files exist

    """
    paths = get_period_paths(year, quarter)

    # Check required: Estados Financieros PDF
    pdf_path = find_file_with_alternatives(paths["raw_pdf"], "estados_financieros_pdf", year, quarter)
    if pdf_path is None:
        logger.debug(f"PDF not found for {year} Q{quarter}")
        return False

    # Check optional: XBRL (only if required)
    if require_xbrl:
        xbrl_path = find_file_with_alternatives(paths["raw_xbrl"], "estados_financieros_xbrl", year, quarter)
        if xbrl_path is None:
            logger.debug(f"XBRL not found for {year} Q{quarter}")
            return False

    return True


# =============================================================================
# Download Orchestration
# =============================================================================


def ensure_files_downloaded(
    year: int,
    quarter: int,
    headless: bool = True,
    force: bool = False,
) -> bool:
    """Ensure files are downloaded for a period.

    Checks if files exist, downloads if missing.

    Args:
        year: Year of the financial statement
        quarter: Quarter (1-4)
        headless: Run browser in headless mode
        force: If True, download even if files exist

    Returns:
        True if files are available (existed or downloaded successfully)

    """
    # Check if files already exist
    if not force and files_exist_for_period(year, quarter):
        logger.info(f"Files already exist for {year} Q{quarter}, skipping download")
        return True

    # Import here to avoid loading playwright when not needed
    from puco_eeff.scraper.cmf_downloader import download_all_documents

    logger.info(f"Downloading files for {year} Q{quarter}...")
    results = download_all_documents(year, quarter, headless=headless)

    # Check if at least PDF was successful
    pdf_success = any(r.success and r.document_type == "estados_financieros_pdf" for r in results)

    if not pdf_success:
        logger.error(f"Failed to download Estados Financieros PDF for {year} Q{quarter}")
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
# Main Processing
# =============================================================================


def process_sheet1(
    year: int,
    quarter: int,
    skip_download: bool = False,
    save: bool = True,
    verbose: bool = True,
    headless: bool = True,
    validate_reference: bool = False,
    fail_on_sum_mismatch: bool = False,
    fail_on_reference_mismatch: bool = False,
) -> tuple[Sheet1Data | None, ValidationReport | None]:
    """Process Sheet1: download if needed, extract, save, and report.

    Main entry point for Sheet1 processing. Orchestrates:
    1. Download files if they don't exist (unless skip_download=True)
    2. Extract Sheet1 data from PDF/XBRL (includes sum and cross-validations)
    3. Run reference validation (if validate_reference=True)
    4. Save to JSON (if save=True)
    5. Print report (if verbose=True)

    Args:
        year: Year of the financial statement
        quarter: Quarter (1-4)
        skip_download: If True, skip download step (fail if files missing)
        save: If True, save extracted data to JSON
        verbose: If True, print extraction report
        headless: Run browser in headless mode for downloads
        validate_reference: If True, validate against reference data
        fail_on_sum_mismatch: If True, return None on sum validation failure
        fail_on_reference_mismatch: If True, return None on reference mismatch

    Returns:
        Tuple of (Sheet1Data, ValidationReport) if successful, (None, None) otherwise

    """
    logger.info(f"Processing Sheet1 for {year} Q{quarter}")

    # Step 1: Ensure files are available
    if not skip_download:
        if not ensure_files_downloaded(year, quarter, headless=headless):
            logger.error("Cannot proceed without required files")
            return None, None
    elif not files_exist_for_period(year, quarter):
        logger.error(f"Files not found for {year} Q{quarter} and download skipped")
        return None, None

    # Step 2: Extract Sheet1 data (includes sum and cross-validations)
    data, report = extract_sheet1(year, quarter, return_report=True)
    if data is None:
        logger.error(f"Extraction failed for {year} Q{quarter}")
        return None, report

    # Step 3: Check sum validation failures (if strict mode)
    if fail_on_sum_mismatch and report and report.has_sum_failures():
        logger.error("Sum validation failed and --fail-on-sum-mismatch is set")
        return None, report

    # Step 4: Reference validation (opt-in)
    if validate_reference or fail_on_reference_mismatch:
        ref_issues = validate_sheet1_against_reference(data)
        if report:
            report.reference_issues = ref_issues if ref_issues is not None else []

        if ref_issues is None:
            logger.info(f"Reference validation skipped: no verified data for {data.quarter}")
        elif len(ref_issues) == 0:
            logger.info(f"✓ Reference validation passed: all values match {data.quarter}")
        else:
            # Prominent warning for reference mismatches
            logger.warning("=" * 60)
            logger.warning("⚠️  REFERENCE DATA MISMATCH - Values differ from known-good data!")
            logger.warning("=" * 60)
            for issue in ref_issues:
                logger.warning(f"  • {issue}")
            logger.warning("Review extraction or update reference_data.json if values are correct.")
            logger.warning("=" * 60)

            if fail_on_reference_mismatch:
                logger.error("Reference mismatch and --fail-on-reference-mismatch is set")
                return None, report

    # Step 5: Save to JSON
    if save:
        output_path = save_sheet1_data(data)
        logger.info(f"Saved to: {output_path}")

    # Step 6: Print report
    if verbose:
        print_sheet1_report(data)
        if report:
            pass

    return data, report


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Process Sheet1: download, extract, and save financial data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m puco_eeff.main_sheet1 --year 2024                    # All quarters
  python -m puco_eeff.main_sheet1 --year 2024 --quarter 2        # Single quarter
  python -m puco_eeff.main_sheet1 --year 2024 -q 2 3             # Multiple quarters
  python -m puco_eeff.main_sheet1 --year 2024 --skip-download
  python -m puco_eeff.main_sheet1 --year 2024 --validate-reference  # With reference check
  python -m puco_eeff.main_sheet1 --year 2024 --fail-on-sum-mismatch  # Strict mode for CI
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
    parser.add_argument("--skip-download", "-s", action="store_true", help="Skip download, use existing files")
    parser.add_argument("--no-save", action="store_true", help="Don't save to JSON")
    parser.add_argument("--quiet", action="store_true", help="Don't print report")
    parser.add_argument("--no-headless", action="store_true", help="Show browser window during download")

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
        data, _report = process_sheet1(
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

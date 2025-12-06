#!/usr/bin/env python3
"""Sheet1 orchestrator - download files if needed, then extract and save.

Usage:
    python -m puco_eeff.main_sheet1 --year 2024 --quarter 2
    python -m puco_eeff.main_sheet1 --year 2024 --quarter 2 --skip-download
    python -m puco_eeff.main_sheet1 --year 2024 --quarter 2 --no-save --quiet

    # Or run directly:
    python puco_eeff/main_sheet1.py --year 2024 --quarter 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path when running directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from puco_eeff.config import (
    find_file_with_alternatives,
    get_period_paths,
    setup_logging,
)
from puco_eeff.extractor.cost_extractor import extract_sheet1
from puco_eeff.sheets.sheet1 import (
    Sheet1Data,
    print_sheet1_report,
    save_sheet1_data,
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
) -> Sheet1Data | None:
    """Process Sheet1: download if needed, extract, save, and report.

    Main entry point for Sheet1 processing. Orchestrates:
    1. Download files if they don't exist (unless skip_download=True)
    2. Extract Sheet1 data from PDF/XBRL
    3. Save to JSON (if save=True)
    4. Print report (if verbose=True)

    Args:
        year: Year of the financial statement
        quarter: Quarter (1-4)
        skip_download: If True, skip download step (fail if files missing)
        save: If True, save extracted data to JSON
        verbose: If True, print extraction report
        headless: Run browser in headless mode for downloads

    Returns:
        Sheet1Data if extraction successful, None otherwise
    """
    logger.info(f"Processing Sheet1 for {year} Q{quarter}")

    # Step 1: Ensure files are available
    if not skip_download:
        if not ensure_files_downloaded(year, quarter, headless=headless):
            logger.error("Cannot proceed without required files")
            return None
    else:
        if not files_exist_for_period(year, quarter):
            logger.error(f"Files not found for {year} Q{quarter} and download skipped")
            return None

    # Step 2: Extract Sheet1 data
    data = extract_sheet1(year, quarter)
    if data is None:
        logger.error(f"Extraction failed for {year} Q{quarter}")
        return None

    # Step 3: Save to JSON
    if save:
        output_path = save_sheet1_data(data)
        logger.info(f"Saved to: {output_path}")

    # Step 4: Print report
    if verbose:
        print_sheet1_report(data)

    return data


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

    args = parser.parse_args()

    # Process each quarter
    success_count = 0
    for quarter in args.quarter:
        result = process_sheet1(
            year=args.year,
            quarter=quarter,
            skip_download=args.skip_download,
            save=not args.no_save,
            verbose=not args.quiet,
            headless=not args.no_headless,
        )
        if result is not None:
            success_count += 1

    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

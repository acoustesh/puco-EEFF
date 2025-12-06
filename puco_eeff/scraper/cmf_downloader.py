"""Unified downloader for CMF Chile financial documents.

Downloads all three financial document types from the CMF Chile portal:
- Análisis Razonado (PDF) - Management Discussion & Analysis
- Estados Financieros (PDF) - Financial Statements PDF
- Estados Financieros (XBRL) - Financial Statements in XBRL/XML format

All documents are available on the same CMF Chile page with filter selection.
"""

from __future__ import annotations

import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

from playwright.sync_api import Download, Page
from playwright.sync_api import TimeoutError as PlaywrightTimeout

from puco_eeff.config import get_config, get_period_paths, setup_logging
from puco_eeff.scraper.browser import browser_session

logger = setup_logging(__name__)

# Document types available for download
DOCUMENT_TYPES = ("analisis_razonado", "estados_financieros_pdf", "estados_financieros_xbrl")


@dataclass
class DownloadResult:
    """Result of a download operation."""

    document_type: str
    success: bool
    file_path: Path | None
    file_size: int | None
    error: str | None = None


def download_all_documents(
    year: int,
    quarter: int,
    headless: bool = True,
    tipo: str = "C",
    tipo_norma: str = "IFRS",
) -> list[DownloadResult]:
    """Download all three financial documents for a period.

    Navigates to CMF Chile, applies filters, and downloads:
    1. Análisis Razonado (PDF)
    2. Estados Financieros (PDF)
    3. Estados Financieros (XBRL ZIP)

    Args:
        year: Year of the financial statement (e.g., 2024)
        quarter: Quarter (1-4)
        headless: Run browser in headless mode
        tipo: Type of balance ("C" for Consolidado or "I" for Individual)
        tipo_norma: Accounting standard ("IFRS" for Estándar IFRS or "NCH" for Norma Chilena)

    Returns:
        List of DownloadResult for each document type
    """
    config = get_config()
    paths = get_period_paths(year, quarter)
    cmf_config = config["sources"]["cmf_chile"]

    results: list[DownloadResult] = []

    # Create output directories - use raw_pdf for all downloads initially
    raw_dir = paths["raw_pdf"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Also ensure XBRL dir exists for extraction
    paths["raw_xbrl"].mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading all documents for {year} Q{quarter}")
    logger.debug(f"Filters: tipo={tipo}, tipo_norma={tipo_norma}")

    with browser_session(headless=headless) as (browser, context, page):
        # Navigate and apply filters
        if not _navigate_and_filter(page, cmf_config, year, quarter, tipo, tipo_norma):
            logger.error("Failed to navigate and apply filters")
            # Return all failures
            for doc_type in DOCUMENT_TYPES:
                results.append(DownloadResult(
                    document_type=doc_type,
                    success=False,
                    file_path=None,
                    file_size=None,
                    error="Failed to navigate and apply filters",
                ))
            return results

        # Download each document type
        for doc_type, doc_config in cmf_config["downloads"].items():
            result = _download_single_document(
                page=page,
                doc_type=doc_type,
                doc_config=doc_config,
                output_dir=raw_dir,
                year=year,
                quarter=quarter,
            )
            results.append(result)

    # Post-process: Extract XBRL if downloaded
    for result in results:
        if result.document_type == "estados_financieros_xbrl" and result.success:
            _extract_xbrl_zip(result.file_path, raw_dir, year, quarter)

    return results


def download_single_document(
    year: int,
    quarter: int,
    document_type: str,
    headless: bool = True,
    tipo: str = "C",
    tipo_norma: str = "IFRS",
) -> DownloadResult:
    """Download a single document type.

    Args:
        year: Year of the financial statement
        quarter: Quarter (1-4)
        document_type: Which document to download
        headless: Run browser in headless mode
        tipo: Type of balance
        tipo_norma: Accounting standard

    Returns:
        DownloadResult with status and file path
    """
    config = get_config()
    paths = get_period_paths(year, quarter)
    cmf_config = config["sources"]["cmf_chile"]

    if document_type not in cmf_config["downloads"]:
        return DownloadResult(
            document_type=document_type,
            success=False,
            file_path=None,
            file_size=None,
            error=f"Unknown document type: {document_type}",
        )

    raw_dir = paths["raw_pdf"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {document_type} for {year} Q{quarter}")

    with browser_session(headless=headless) as (browser, context, page):
        if not _navigate_and_filter(page, cmf_config, year, quarter, tipo, tipo_norma):
            return DownloadResult(
                document_type=document_type,
                success=False,
                file_path=None,
                file_size=None,
                error="Failed to navigate and apply filters",
            )

        doc_config = cmf_config["downloads"][document_type]
        result = _download_single_document(
            page=page,
            doc_type=document_type,
            doc_config=doc_config,
            output_dir=raw_dir,
            year=year,
            quarter=quarter,
        )

        # Extract XBRL if applicable
        if result.document_type == "estados_financieros_xbrl" and result.success:
            _extract_xbrl_zip(result.file_path, raw_dir, year, quarter)

    return result


def _navigate_and_filter(
    page: Page,
    cmf_config: dict,
    year: int,
    quarter: int,
    tipo: str,
    tipo_norma: str,
) -> bool:
    """Navigate to CMF Chile and apply period/type filters.

    Args:
        page: Playwright page instance
        cmf_config: CMF configuration from config.json
        year: Target year
        quarter: Target quarter (1-4)
        tipo: Balance type
        tipo_norma: Accounting standard

    Returns:
        True if navigation and filtering succeeded
    """
    base_url = cmf_config["base_url"]
    selectors = cmf_config["filters"]["selectors"]
    quarter_to_month = cmf_config["filters"]["quarter_to_month"]

    month = quarter_to_month[str(quarter)]

    try:
        logger.debug(f"Navigating to: {base_url}")
        page.goto(base_url, wait_until="networkidle", timeout=60000)

        # Apply filters
        logger.debug(f"Selecting month: {month}")
        page.select_option(selectors["month"], month)

        logger.debug(f"Selecting year: {year}")
        page.select_option(selectors["year"], str(year))

        logger.debug(f"Selecting tipo: {tipo}")
        page.select_option(selectors["tipo"], tipo)

        logger.debug(f"Selecting tipo_norma: {tipo_norma}")
        page.select_option(selectors["tipo_norma"], tipo_norma)

        # Click submit
        logger.debug("Clicking Consultar button")
        page.click(selectors["submit"])
        page.wait_for_load_state("networkidle")

        # Small delay to ensure results are loaded
        time.sleep(2)

        logger.info(f"Successfully filtered for {year} Q{quarter}")
        return True

    except PlaywrightTimeout as e:
        logger.error(f"Timeout during navigation: {e}")
        return False
    except Exception as e:
        logger.error(f"Error during navigation: {e}")
        return False


def _download_single_document(
    page: Page,
    doc_type: str,
    doc_config: dict,
    output_dir: Path,
    year: int,
    quarter: int,
) -> DownloadResult:
    """Download a single document from the filtered results page.

    Args:
        page: Playwright page instance (already filtered)
        doc_type: Document type key
        doc_config: Document configuration with link_text and filename_pattern
        output_dir: Directory to save the file
        year: Year for filename
        quarter: Quarter for filename

    Returns:
        DownloadResult with status
    """
    link_text = doc_config["link_text"]
    filename = doc_config["filename_pattern"].format(year=year, quarter=quarter)
    output_path = output_dir / filename

    logger.debug(f"Looking for link: {link_text}")

    try:
        # Find the download link
        link = page.get_by_text(link_text, exact=True)

        if link.count() == 0:
            logger.warning(f"Link not found: {link_text}")
            return DownloadResult(
                document_type=doc_type,
                success=False,
                file_path=None,
                file_size=None,
                error=f"Download link not found: {link_text}",
            )

        # Trigger download
        with page.expect_download(timeout=60000) as download_info:
            link.click()

        download: Download = download_info.value
        download.save_as(str(output_path))

        if output_path.exists():
            file_size = output_path.stat().st_size
            logger.info(f"Downloaded: {filename} ({file_size:,} bytes)")
            return DownloadResult(
                document_type=doc_type,
                success=True,
                file_path=output_path,
                file_size=file_size,
            )
        else:
            logger.error(f"File not saved: {output_path}")
            return DownloadResult(
                document_type=doc_type,
                success=False,
                file_path=None,
                file_size=None,
                error="File download completed but not saved",
            )

    except PlaywrightTimeout:
        logger.error(f"Timeout downloading: {link_text}")
        return DownloadResult(
            document_type=doc_type,
            success=False,
            file_path=None,
            file_size=None,
            error="Download timeout",
        )
    except Exception as e:
        logger.error(f"Error downloading {link_text}: {e}")
        return DownloadResult(
            document_type=doc_type,
            success=False,
            file_path=None,
            file_size=None,
            error=str(e),
        )


def _extract_xbrl_zip(zip_path: Path | None, output_dir: Path, year: int, quarter: int) -> Path | None:
    """Extract XBRL instance document from downloaded ZIP file.

    The ZIP typically contains multiple files:
    - .xbrl file: Main instance document with actual financial data
    - .xml files: Usually label linkbases, definitions, etc. (not the main data)
    - .xsd file: Schema definition

    This function prioritizes .xbrl files as they contain the actual financial facts.

    Args:
        zip_path: Path to the ZIP file
        output_dir: Directory to extract to
        year: Year for naming
        quarter: Quarter for naming

    Returns:
        Path to the extracted XBRL file, or None if extraction failed
    """
    if zip_path is None or not zip_path.exists():
        logger.warning(f"ZIP file not found: {zip_path}")
        return None

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # List all files in archive
            all_files = zf.namelist()
            logger.debug(f"Files in ZIP: {all_files}")

            # Prioritize .xbrl files (main instance document with actual data)
            # .xml files are often just label linkbases without financial facts
            xbrl_files = [f for f in all_files if f.endswith(".xbrl")]
            xml_files = [f for f in all_files if f.endswith(".xml")]

            if xbrl_files:
                # Use .xbrl file - this is the main instance document
                main_file = xbrl_files[0]
                output_filename = f"estados_financieros_{year}_Q{quarter}.xbrl"
                logger.info(f"Found .xbrl instance document: {main_file}")
            elif xml_files:
                # Fallback to .xml if no .xbrl found
                main_file = xml_files[0]
                output_filename = f"estados_financieros_{year}_Q{quarter}.xml"
                logger.warning(f"No .xbrl found, falling back to .xml: {main_file}")
            else:
                logger.error("No XML/XBRL files found in ZIP archive")
                return None

            output_path = output_dir / output_filename
            logger.debug(f"Extracting: {main_file} -> {output_path}")

            with zf.open(main_file) as source:
                xbrl_content = source.read()
                output_path.write_bytes(xbrl_content)

        logger.info(f"Extracted XBRL to: {output_path} ({output_path.stat().st_size:,} bytes)")
        return output_path

    except zipfile.BadZipFile:
        logger.error(f"Invalid ZIP file: {zip_path}")
        return None
    except Exception as e:
        logger.error(f"Error extracting ZIP: {e}")
        return None


def list_available_periods(headless: bool = True) -> list[dict]:
    """List all available periods from the CMF Chile page.

    Useful for discovering what periods are available for download.

    Args:
        headless: Run browser in headless mode

    Returns:
        List of available periods with year, month, and quarter
    """
    config = get_config()
    cmf_config = config["sources"]["cmf_chile"]
    base_url = cmf_config["base_url"]
    month_to_quarter = cmf_config["filters"]["month_to_quarter"]

    periods: list[dict] = []

    with browser_session(headless=headless) as (browser, context, page):
        page.goto(base_url, wait_until="networkidle", timeout=60000)

        # Get available years
        year_select = page.query_selector("select[name='aa']")
        if year_select:
            year_options = year_select.query_selector_all("option")
            years = [
                opt.get_attribute("value")
                for opt in year_options
                if opt.get_attribute("value")
            ]
            logger.debug(f"Available years: {years}")

            # Get available months
            month_select = page.query_selector("select[name='mm']")
            if month_select:
                month_options = month_select.query_selector_all("option")
                months = [
                    opt.get_attribute("value")
                    for opt in month_options
                    if opt.get_attribute("value")
                ]
                logger.debug(f"Available months: {months}")

                for year in years:
                    for month in months:
                        if year and month:
                            quarter = month_to_quarter.get(month)
                            if quarter:
                                periods.append({
                                    "year": int(year),
                                    "month": month,
                                    "quarter": quarter,
                                })

    return periods

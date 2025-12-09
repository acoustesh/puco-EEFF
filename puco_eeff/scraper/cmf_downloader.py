"""Unified downloader for CMF Chile financial documents.

Downloads all three financial document types from the CMF Chile portal:
- Análisis Razonado (PDF) - Management Discussion & Analysis
- Estados Financieros (PDF) - Financial Statements PDF
- Estados Financieros (XBRL) - Financial Statements in XBRL/XML format

All documents are available on the same CMF Chile page with filter selection.

When CMF Chile doesn't have the data (e.g., Q1 before Q2 is published),
the downloader falls back to Pucobre.cl which has the PDF available.
"""

from __future__ import annotations

import time
import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

from playwright.sync_api import Download, Page
from playwright.sync_api import TimeoutError as PlaywrightTimeout

from puco_eeff.config import get_config, get_period_paths, setup_logging
from puco_eeff.scraper.browser import PeriodExtractor, browser_session, list_periods_from_page

if TYPE_CHECKING:
    from pathlib import Path

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
    source: str = "cmf"  # "cmf" or "pucobre.cl"


def download_all_documents(
    year: int,
    quarter: int,
    headless: bool = True,
    tipo: str = "C",
    tipo_norma: str = "IFRS",
    fallback_to_pucobre: bool = True,
) -> list[DownloadResult]:
    """Download all three financial documents for a period.

    Navigates to CMF Chile, applies filters, and downloads:
    1. Análisis Razonado (PDF)
    2. Estados Financieros (PDF)
    3. Estados Financieros (XBRL ZIP)

    If CMF Chile doesn't have the data and fallback_to_pucobre is True,
    attempts to download Estados Financieros PDF from Pucobre.cl.

    Args:
        year: Year of the financial statement (e.g., 2024)
        quarter: Quarter (1-4)
        headless: Run browser in headless mode
        tipo: Type of balance ("C" for Consolidado or "I" for Individual)
        tipo_norma: Accounting standard ("IFRS" for Estándar IFRS or "NCH" for Norma Chilena)
        fallback_to_pucobre: If True, try Pucobre.cl when CMF fails

    Returns:
        List of DownloadResult for each document type

    """
    config = get_config()
    paths = get_period_paths(year, quarter)
    cmf_config = config["sources"]["cmf_chile"]

    results: list[DownloadResult] = []

    # Create output directories - PDFs and XBRL in separate directories
    pdf_dir = paths["raw_pdf"]
    xbrl_dir = paths["raw_xbrl"]
    pdf_dir.mkdir(parents=True, exist_ok=True)
    xbrl_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading all documents for %s Q%s", year, quarter)
    logger.debug("Filters: tipo=%s, tipo_norma=%s", tipo, tipo_norma)

    cmf_success = False

    with browser_session(headless=headless) as (_browser, _context, page):
        # Navigate and apply filters
        if not _navigate_and_filter(page, cmf_config, year, quarter, tipo, tipo_norma):
            logger.warning("Failed to navigate and apply filters on CMF Chile")
        else:
            # Download each document type
            for doc_type, doc_config in cmf_config["downloads"].items():
                # XBRL goes to xbrl directory, PDFs go to pdf directory
                output_dir = xbrl_dir if doc_type == "estados_financieros_xbrl" else pdf_dir
                result = _download_single_document(
                    page=page,
                    doc_type=doc_type,
                    doc_config=doc_config,
                    output_dir=output_dir,
                    year=year,
                    quarter=quarter,
                )
                results.append(result)

            # Check if we got the main PDF
            pdf_result = next(
                (r for r in results if r.document_type == "estados_financieros_pdf"), None,
            )
            cmf_success = pdf_result is not None and pdf_result.success

    # Fallback to Pucobre.cl if CMF Chile failed and fallback is enabled
    if not cmf_success and fallback_to_pucobre:
        logger.info("CMF Chile download failed, trying Pucobre.cl fallback...")
        results = _download_with_pucobre_fallback(year, quarter, results, pdf_dir, headless)

    # If we still don't have results, return failures
    if not results:
        for doc_type in DOCUMENT_TYPES:
            results.append(
                DownloadResult(
                    document_type=doc_type,
                    success=False,
                    file_path=None,
                    file_size=None,
                    error="Failed to download from both CMF Chile and Pucobre.cl",
                ),
            )

    # Post-process: Extract XBRL if downloaded
    for result in results:
        if result.document_type == "estados_financieros_xbrl" and result.success:
            _extract_xbrl_zip(result.file_path, xbrl_dir, year, quarter)

    return results


def _download_with_pucobre_fallback(
    year: int,
    quarter: int,
    existing_results: list[DownloadResult],
    pdf_dir: Path,
    headless: bool,
) -> list[DownloadResult]:
    """Try to download from Pucobre.cl as a fallback.

    Args:
        year: Target year
        quarter: Target quarter
        existing_results: Results from CMF Chile attempt
        pdf_dir: Output directory for PDFs
        headless: Run browser in headless mode

    Returns:
        Updated results list

    """
    from puco_eeff.scraper.pucobre_downloader import download_from_pucobre

    pucobre_result = download_from_pucobre(year, quarter, headless, split_pdf=True)

    # Update or add the PDF result
    pdf_idx = next(
        (i for i, r in enumerate(existing_results) if r.document_type == "estados_financieros_pdf"),
        None,
    )
    ar_idx = next(
        (i for i, r in enumerate(existing_results) if r.document_type == "analisis_razonado"), None,
    )

    if pucobre_result.success:
        # Estados Financieros PDF
        eeff_result = DownloadResult(
            document_type="estados_financieros_pdf",
            success=True,
            file_path=pucobre_result.file_path,
            file_size=pucobre_result.file_size,
            error=None,
            source="pucobre.cl",
        )
        if pdf_idx is not None:
            existing_results[pdf_idx] = eeff_result
        else:
            existing_results.append(eeff_result)

        logger.info(
            f"Successfully downloaded Estados Financieros from Pucobre.cl: {pucobre_result.file_path}",
        )

        # Análisis Razonado (if split was successful)
        if pucobre_result.analisis_razonado_path is not None:
            ar_result = DownloadResult(
                document_type="analisis_razonado",
                success=True,
                file_path=pucobre_result.analisis_razonado_path,
                file_size=pucobre_result.analisis_razonado_size,
                error=None,
                source="pucobre.cl",
            )
            if ar_idx is not None:
                existing_results[ar_idx] = ar_result
            else:
                existing_results.append(ar_result)

            logger.info(
                f"Successfully extracted Análisis Razonado from Pucobre.cl: {pucobre_result.analisis_razonado_path}",
            )
        # Análisis Razonado not extracted (split failed or not available)
        elif not any(
            r.document_type == "analisis_razonado" and r.success for r in existing_results
        ):
            ar_result = DownloadResult(
                document_type="analisis_razonado",
                success=False,
                file_path=None,
                file_size=None,
                error="Could not extract Análisis Razonado from combined PDF",
                source="pucobre.cl",
            )
            if ar_idx is not None:
                existing_results[ar_idx] = ar_result
            else:
                existing_results.append(ar_result)

        # XBRL is never available from Pucobre.cl
        if not any(r.document_type == "estados_financieros_xbrl" for r in existing_results):
            existing_results.append(
                DownloadResult(
                    document_type="estados_financieros_xbrl",
                    success=False,
                    file_path=None,
                    file_size=None,
                    error="XBRL not available on Pucobre.cl",
                    source="pucobre.cl",
                ),
            )
    else:
        logger.error(f"Pucobre.cl fallback also failed: {pucobre_result.error}")
        if pdf_idx is not None:
            existing_results[pdf_idx] = DownloadResult(
                document_type="estados_financieros_pdf",
                success=False,
                file_path=None,
                file_size=None,
                error=f"CMF and Pucobre both failed: {pucobre_result.error}",
            )

    return existing_results


def download_single_document(
    year: int,
    quarter: int,
    document_type: str,
    headless: bool = True,
    tipo: str = "C",
    tipo_norma: str = "IFRS",
    fallback_to_pucobre: bool = True,
) -> DownloadResult:
    """Download a single document type.

    Args:
        year: Year of the financial statement
        quarter: Quarter (1-4)
        document_type: Which document to download
        headless: Run browser in headless mode
        tipo: Type of balance
        tipo_norma: Accounting standard
        fallback_to_pucobre: If True, try Pucobre.cl when CMF fails (PDF only)

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

    # Use appropriate directory based on document type
    pdf_dir = paths["raw_pdf"]
    xbrl_dir = paths["raw_xbrl"]
    pdf_dir.mkdir(parents=True, exist_ok=True)
    xbrl_dir.mkdir(parents=True, exist_ok=True)

    output_dir = xbrl_dir if document_type == "estados_financieros_xbrl" else pdf_dir

    logger.info("Downloading %s for %s Q%s", document_type, year, quarter)

    result = None

    with browser_session(headless=headless) as (_browser, _context, page):
        if not _navigate_and_filter(page, cmf_config, year, quarter, tipo, tipo_norma):
            logger.warning("Failed to navigate and apply CMF Chile filters")
        else:
            doc_config = cmf_config["downloads"][document_type]
            result = _download_single_document(
                page=page,
                doc_type=document_type,
                doc_config=doc_config,
                output_dir=output_dir,
                year=year,
                quarter=quarter,
            )

            # Extract XBRL if applicable
            if result.document_type == "estados_financieros_xbrl" and result.success:
                _extract_xbrl_zip(result.file_path, xbrl_dir, year, quarter)

    # Fallback to Pucobre for PDF if CMF failed
    if (result is None or not result.success) and fallback_to_pucobre:
        if document_type == "estados_financieros_pdf":
            logger.info("Trying Pucobre.cl fallback for PDF...")
            from puco_eeff.scraper.pucobre_downloader import download_from_pucobre

            pucobre_result = download_from_pucobre(year, quarter, headless)
            if pucobre_result.success:
                return DownloadResult(
                    document_type=document_type,
                    success=True,
                    file_path=pucobre_result.file_path,
                    file_size=pucobre_result.file_size,
                    error=None,
                )
            return DownloadResult(
                document_type=document_type,
                success=False,
                file_path=None,
                file_size=None,
                error=f"CMF and Pucobre both failed: {pucobre_result.error}",
            )
        # XBRL and Análisis Razonado not available on Pucobre
        return DownloadResult(
            document_type=document_type,
            success=False,
            file_path=None,
            file_size=None,
            error="Not available on CMF Chile; document type not available on Pucobre.cl fallback",
        )

    if result is None:
        return DownloadResult(
            document_type=document_type,
            success=False,
            file_path=None,
            file_size=None,
            error="Failed to download from CMF Chile",
        )

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
        logger.debug("Navigating to: %s", base_url)
        page.goto(base_url, wait_until="networkidle", timeout=60000)

        # Apply filters
        logger.debug("Selecting month: %s", month)
        page.select_option(selectors["month"], month)

        logger.debug("Selecting year: %s", year)
        page.select_option(selectors["year"], str(year))

        logger.debug("Selecting tipo: %s", tipo)
        page.select_option(selectors["tipo"], tipo)

        logger.debug("Selecting tipo_norma: %s", tipo_norma)
        page.select_option(selectors["tipo_norma"], tipo_norma)

        # Click submit
        logger.debug("Clicking Consultar button")
        page.click(selectors["submit"])
        page.wait_for_load_state("networkidle")

        # Small delay to ensure results are loaded
        time.sleep(2)

        logger.info("Successfully filtered for %s Q%s", year, quarter)
        return True

    except PlaywrightTimeout as e:
        logger.exception("Timeout during navigation: %s", e)
        return False
    except Exception as e:
        logger.exception("Error during navigation: %s", e)
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

    logger.debug("Looking for link: %s", link_text)

    try:
        # Find the download link
        link = page.get_by_text(link_text, exact=True)

        if link.count() == 0:
            logger.warning("Link not found: %s", link_text)
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
        logger.error("File not saved: %s", output_path)
        return DownloadResult(
            document_type=doc_type,
            success=False,
            file_path=None,
            file_size=None,
            error="File download completed but not saved",
        )

    except PlaywrightTimeout:
        logger.exception("Timeout downloading: %s", link_text)
        return DownloadResult(
            document_type=doc_type,
            success=False,
            file_path=None,
            file_size=None,
            error="Download timeout",
        )
    except Exception as e:
        logger.exception("Error downloading %s: %s", link_text, e)
        return DownloadResult(
            document_type=doc_type,
            success=False,
            file_path=None,
            file_size=None,
            error=str(e),
        )


def _extract_xbrl_zip(
    zip_path: Path | None, xbrl_dir: Path, year: int, quarter: int,
) -> Path | None:
    """Extract XBRL instance document from downloaded ZIP file.

    The ZIP typically contains multiple files:
    - .xbrl file: Main instance document with actual financial data
    - .xml files: Usually label linkbases, definitions, etc. (not the main data)
    - .xsd file: Schema definition

    This function prioritizes .xbrl files as they contain the actual financial facts.

    Args:
        zip_path: Path to the ZIP file
        xbrl_dir: Directory to extract XBRL files to (data/raw/xbrl/)
        year: Year for naming
        quarter: Quarter for naming

    Returns:
        Path to the extracted XBRL file, or None if extraction failed

    """
    if zip_path is None or not zip_path.exists():
        logger.warning("ZIP file not found: %s", zip_path)
        return None

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # List all files in archive
            all_files = zf.namelist()
            logger.debug("Files in ZIP: %s", all_files)

            # Prioritize .xbrl files (main instance document with actual data)
            # .xml files are often just label linkbases without financial facts
            xbrl_files = [f for f in all_files if f.endswith(".xbrl")]
            xml_files = [f for f in all_files if f.endswith(".xml")]

            if xbrl_files:
                # Use .xbrl file - this is the main instance document
                main_file = xbrl_files[0]
                output_filename = f"estados_financieros_{year}_Q{quarter}.xbrl"
                logger.info("Found .xbrl instance document: %s", main_file)
            elif xml_files:
                # Fallback to .xml if no .xbrl found
                main_file = xml_files[0]
                output_filename = f"estados_financieros_{year}_Q{quarter}.xml"
                logger.warning("No .xbrl found, falling back to .xml: %s", main_file)
            else:
                logger.error("No XML/XBRL files found in ZIP archive")
                return None

            # Save to xbrl directory (separate from PDFs)
            xbrl_dir.mkdir(parents=True, exist_ok=True)
            output_path = xbrl_dir / output_filename
            logger.debug("Extracting: %s -> %s", main_file, output_path)

            with zf.open(main_file) as source:
                xbrl_content = source.read()
                output_path.write_bytes(xbrl_content)

        logger.info(f"Extracted XBRL to: {output_path} ({output_path.stat().st_size:,} bytes)")
        return output_path

    except zipfile.BadZipFile:
        logger.exception("Invalid ZIP file: %s", zip_path)
        return None
    except Exception as e:
        logger.exception("Error extracting ZIP: %s", e)
        return None


def _extract_cmf_periods_from_page(page: Page) -> list[dict]:
    """Extract available periods from CMF page selectors."""
    config = get_config()
    month_to_quarter = config["sources"]["cmf_chile"]["filters"]["month_to_quarter"]
    periods: list[dict] = []

    year_select = page.query_selector("select[name='aa']")
    month_select = page.query_selector("select[name='mm']")

    if not year_select or not month_select:
        return periods

    years = [
        opt.get_attribute("value")
        for opt in year_select.query_selector_all("option")
        if opt.get_attribute("value")
    ]
    months = [
        opt.get_attribute("value")
        for opt in month_select.query_selector_all("option")
        if opt.get_attribute("value")
    ]

    for year in years:
        for month in months:
            quarter = month_to_quarter.get(month) if year and month else None
            if quarter:
                periods.append({"year": int(year), "month": month, "quarter": quarter})

    return periods


def list_available_periods(headless: bool = True) -> list[dict]:
    """List all available periods from the CMF Chile page.

    Discovers what periods are available for download by extracting
    year/month combinations from the CMF filter dropdown selectors.

    Args:
        headless: Run browser in headless mode

    Returns:
        List of available periods with year, month, and quarter keys

    """
    cmf_config = get_config()["sources"]["cmf_chile"]
    extractor = PeriodExtractor(
        url=cmf_config["base_url"],
        page_extractor=_extract_cmf_periods_from_page,
        source_name="cmf_chile",
    )
    return list_periods_from_page(extractor, headless=headless)

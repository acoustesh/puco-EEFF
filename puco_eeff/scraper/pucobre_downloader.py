"""Fallback downloader for Pucobre.cl financial documents.

Downloads financial statements from Pucobre's website when CMF Chile
doesn't have the data available (e.g., Q1 reports before Q2 is published).

Note: Pucobre.cl has a combined PDF containing both Estados Financieros
AND Análisis Razonado. This module downloads the combined file and splits
it into two separate PDFs matching the CMF file naming convention.
No XBRL available from this source.
"""

from __future__ import annotations

import contextlib
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from playwright.sync_api import Locator, Page, Response
from playwright.sync_api import TimeoutError as PlaywrightTimeout

from puco_eeff.config import get_config, get_period_paths, setup_logging
from puco_eeff.scraper.browser import PeriodExtractor, browser_session, list_periods_from_page

if TYPE_CHECKING:
    from pathlib import Path

logger = setup_logging(__name__)


def _get_pucobre_config(config: dict | None = None) -> tuple[str, dict[int, str]]:
    """Get Pucobre URL and quarter-to-date mapping from config.

    Args:
        config: Configuration dict, or None to load from file

    Returns:
        Tuple of (base_url, quarter_to_date_mapping)

    """
    if config is None:
        config = get_config()

    pucobre_config = config.get("sources", {}).get("pucobre", {})
    base_url = pucobre_config.get(
        "base_url",
        "https://www.pucobre.cl/OpenDocs/asp/pagDefault.asp?"
        "boton=Doc51&argInstanciaId=51&argCarpetaId=32&"
        "argTreeNodosAbiertos=(32)&argTreeNodoActual=32&argTreeNodoSel=32",
    )

    # Config stores keys as strings, convert to int keys
    quarter_to_date_raw = pucobre_config.get(
        "quarter_to_date",
        {
            "1": "31-03",
            "2": "30-06",
            "3": "30-09",
            "4": "31-12",
        },
    )
    quarter_to_date = {int(k): v for k, v in quarter_to_date_raw.items()}

    return base_url, quarter_to_date


@dataclass
class PucobreDownloadResult:
    """Result of a Pucobre.cl download operation.

    This dataclass handles the combined PDF download from Pucobre's website,
    which bundles Estados Financieros and Análisis Razonado into a single file
    that gets split into separate documents after download.
    """

    success: bool
    file_path: Path | None  # Estados Financieros PDF after splitting
    file_size: int | None
    error: str | None = None
    source: str = "pucobre.cl"
    # Secondary files extracted from the combined PDF bundle
    analisis_razonado_path: Path | None = None
    analisis_razonado_size: int | None = None
    combined_pdf_path: Path | None = None  # Retained original for debugging


def _find_analisis_razonado_page(pdf_path: Path) -> int | None:
    """Find the page number where Análisis Razonado starts.

    The Pucobre.cl combined PDF contains Estados Financieros followed by
    Análisis Razonado. The Análisis Razonado section typically starts with
    a title page containing "ANALISIS RAZONADO" and resets page numbering to 1.

    Args:
        pdf_path: Path to the combined PDF file

    Returns:
        0-based page index where Análisis Razonado starts, or None if not found

    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF not installed, cannot split PDF. Install with: pip install PyMuPDF")
        return None

    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        text_raw = page.get_text()
        text = str(text_raw or "")
        text_upper = text.upper()

        # Look for the Análisis Razonado title page
        # It typically contains "ANALISIS RAZONADO" as a prominent heading
        if "ANALISIS RAZONADO" in text_upper:
            # Verify this looks like a title page (short text, near start)
            text_lines = [line.strip() for line in text.split("\n") if line.strip()]
            # Title pages typically have few lines and contain the company name
            if len(text_lines) < 20 or "SOCIEDAD PUNTA DEL COBRE" in text_upper:
                doc.close()
                logger.info(f"Found Análisis Razonado at page {page_num + 1}")
                return page_num

    doc.close()
    return None


def _split_combined_pdf(
    combined_pdf_path: Path,
    estados_financieros_path: Path,
    analisis_razonado_path: Path,
    split_page: int,
) -> tuple[bool, str | None]:
    """Split combined PDF into Estados Financieros and Análisis Razonado.

    Args:
        combined_pdf_path: Path to the original combined PDF
        estados_financieros_path: Output path for Estados Financieros (pages 0 to split_page-1)
        analisis_razonado_path: Output path for Análisis Razonado (pages split_page to end)
        split_page: 0-based page index where Análisis Razonado starts

    Returns:
        Tuple of (success, error_message)

    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return False, "PyMuPDF not installed"

    try:
        doc = fitz.open(combined_pdf_path)
        total_pages = len(doc)

        # Create Estados Financieros PDF (pages before Análisis Razonado)
        eeff_doc = fitz.open()
        eeff_doc.insert_pdf(doc, from_page=0, to_page=split_page - 1)
        eeff_doc.save(estados_financieros_path)
        eeff_doc.close()

        # Create Análisis Razonado PDF (pages from split_page to end)
        ar_doc = fitz.open()
        ar_doc.insert_pdf(doc, from_page=split_page, to_page=total_pages - 1)
        ar_doc.save(analisis_razonado_path)
        ar_doc.close()

        doc.close()

        logger.info(
            f"Split PDF: Estados Financieros ({split_page} pages), Análisis Razonado ({total_pages - split_page} pages)",
        )
        return True, None

    except Exception as e:
        return False, str(e)


def download_from_pucobre(
    year: int,
    quarter: int,
    headless: bool = True,
    split_pdf: bool = True,
    config: dict | None = None,
) -> PucobreDownloadResult:
    """Download financial documents from Pucobre.cl.

    Pucobre.cl provides a combined PDF containing both Estados Financieros
    AND Análisis Razonado. This function downloads the combined file and
    optionally splits it into separate PDFs matching the CMF naming convention.

    Args:
        year: Year of the financial statement (e.g., 2024)
        quarter: Quarter (1-4)
        headless: Run browser in headless mode
        split_pdf: If True, split combined PDF into separate files
        config: Configuration dict, or None to load from file

    Returns:
        PucobreDownloadResult with status and file paths

    """
    pucobre_url, quarter_to_date = _get_pucobre_config(config)

    paths = get_period_paths(year, quarter)
    raw_dir = paths["raw_pdf"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    # File paths
    combined_filename = f"pucobre_combined_{year}_Q{quarter}.pdf"
    combined_path = raw_dir / combined_filename

    eeff_filename = f"estados_financieros_{year}_Q{quarter}.pdf"
    eeff_path = raw_dir / eeff_filename

    ar_filename = f"analisis_razonado_{year}_Q{quarter}.pdf"
    ar_path = raw_dir / ar_filename

    # Expected link text pattern: "Estados Financieros DD-MM-YYYY"
    date_str = f"{quarter_to_date[quarter]}-{year}"
    expected_link = f"Estados Financieros {date_str}"

    logger.info("Attempting to download from Pucobre.cl: %s", expected_link)

    with browser_session(headless=headless) as (_browser, context, page):
        # Track PDF URLs from responses
        pdf_url: str | None = None
        pdf_content: bytes | None = None

        def capture_pdf_response(response: Response) -> None:
            """Capture PDF response content from browser network traffic."""
            nonlocal pdf_url, pdf_content
            content_type = response.headers.get("content-type", "")
            if "pdf" in content_type.lower():
                pdf_url = response.url
                with contextlib.suppress(Exception):
                    pdf_content = response.body()

        context.on("response", capture_pdf_response)

        try:
            # Navigate to Pucobre Estados Financieros page
            logger.debug("Navigating to: %s", pucobre_url)
            page.goto(pucobre_url, wait_until="networkidle", timeout=60000)

            # Wait for the table to load
            time.sleep(2)

            # Find the link for the requested period
            link = page.get_by_role("link", name=expected_link)

            if link.count() == 0:
                logger.warning("Link not found: %s", expected_link)
                # Try alternative search - look for partial match
                link = _find_period_link(page, year, quarter, quarter_to_date)
                if link is None:
                    return PucobreDownloadResult(
                        success=False,
                        file_path=None,
                        file_size=None,
                        error=f"Period not found on Pucobre.cl: {year} Q{quarter}",
                    )

            # Click the link - it opens in a new tab with PDF viewer
            with context.expect_page() as new_page_info:
                link.click()

            new_page = new_page_info.value
            # Wait for the PDF to load
            time.sleep(5)

            # Download the PDF content
            downloaded = False
            if pdf_content:
                combined_path.write_bytes(pdf_content)
                downloaded = True
            elif pdf_url:
                logger.debug("PDF URL captured: %s", pdf_url)
                response = page.request.get(pdf_url)
                if response.ok:
                    combined_path.write_bytes(response.body())
                    downloaded = True
                else:
                    new_page.close()
                    return PucobreDownloadResult(
                        success=False,
                        file_path=None,
                        file_size=None,
                        error=f"Failed to download PDF: HTTP {response.status}",
                    )

            new_page.close()

            if not downloaded:
                return PucobreDownloadResult(
                    success=False,
                    file_path=None,
                    file_size=None,
                    error="Failed to capture PDF URL from viewer page",
                )

            combined_size = combined_path.stat().st_size
            logger.info(
                f"Downloaded combined PDF from Pucobre: {combined_filename} ({combined_size:,} bytes)",
            )

            # Split the PDF if requested
            if split_pdf:
                split_page = _find_analisis_razonado_page(combined_path)

                if split_page is not None:
                    success, error = _split_combined_pdf(
                        combined_path,
                        eeff_path,
                        ar_path,
                        split_page,
                    )

                    if success:
                        eeff_size = eeff_path.stat().st_size
                        ar_size = ar_path.stat().st_size

                        logger.info(
                            f"Split into: {eeff_filename} ({eeff_size:,} bytes), {ar_filename} ({ar_size:,} bytes)",
                        )

                        return PucobreDownloadResult(
                            success=True,
                            file_path=eeff_path,
                            file_size=eeff_size,
                            analisis_razonado_path=ar_path,
                            analisis_razonado_size=ar_size,
                            combined_pdf_path=combined_path,
                        )
                    logger.warning("Failed to split PDF: %s", error)
                        # Fall through to return combined as EEFF
                else:
                    logger.warning("Could not find Análisis Razonado section in PDF")

            # If not splitting or split failed, use combined as Estados Financieros
            # Copy combined to EEFF path
            import shutil

            shutil.copy(combined_path, eeff_path)
            eeff_size = eeff_path.stat().st_size

            return PucobreDownloadResult(
                success=True,
                file_path=eeff_path,
                file_size=eeff_size,
                combined_pdf_path=combined_path,
            )

        except PlaywrightTimeout as e:
            logger.exception("Timeout downloading from Pucobre: %s", e)
            return PucobreDownloadResult(
                success=False,
                file_path=None,
                file_size=None,
                error=f"Timeout: {e}",
            )
        except Exception as e:
            logger.exception("Error downloading from Pucobre: %s", e)
            return PucobreDownloadResult(
                success=False,
                file_path=None,
                file_size=None,
                error=str(e),
            )


def _find_period_link(
    page: Page,
    year: int,
    quarter: int,
    quarter_to_date: dict[int, str],
) -> Locator | None:
    """Find the download link for a specific period using various patterns.

    Args:
        page: Playwright page instance
        year: Target year
        quarter: Target quarter (1-4)
        quarter_to_date: Mapping from quarter number to date string

    Returns:
        Locator for the link, or None if not found

    """
    date_str = f"{quarter_to_date[quarter]}-{year}"

    # Try different patterns
    patterns = [
        f"Estados Financieros {date_str}",
        f"Estados Financieros {date_str.replace('-', '/')}",
        re.compile(rf"Estados\s+Financieros.*{date_str}", re.IGNORECASE),
    ]

    for pattern in patterns:
        if isinstance(pattern, str):
            link = page.get_by_role("link", name=pattern)
        else:
            link = page.locator(f"a:has-text('{quarter_to_date[quarter]}-{year}')")

        if link.count() > 0:
            logger.debug("Found link with pattern: %s", pattern)
            return link

    return None


def _extract_pucobre_periods_from_page(page: Page) -> list[dict]:
    """Extract available periods from Pucobre page links."""
    time.sleep(2)  # Allow page to settle

    periods: list[dict] = []
    links = page.locator("a:has-text('Estados Financieros')")
    count = links.count()

    for i in range(count):
        link_text = links.nth(i).inner_text()
        match = re.search(r"(\d{2})-(\d{2})-(\d{4})", link_text)
        if not match:
            continue

        _day, month, year = match.groups()
        month_to_quarter = {"03": 1, "06": 2, "09": 3, "12": 4}
        quarter = month_to_quarter.get(month)
        if quarter:
            periods.append(
                {
                    "year": int(year),
                    "quarter": quarter,
                    "link_text": link_text,
                    "source": "pucobre.cl",
                },
            )

    return periods


def list_pucobre_periods(headless: bool = True, config: dict | None = None) -> list[dict]:
    """List all available periods from Pucobre.cl.

    Wraps the generic period extraction with Pucobre-specific configuration
    and logging of the found periods.

    Args:
        headless: Run browser in headless mode
        config: Configuration dict, or None to load from file

    Returns:
        List of available periods with year, quarter, link_text, and source

    """
    pucobre_url, _ = _get_pucobre_config(config)
    extractor = PeriodExtractor(
        url=pucobre_url,
        page_extractor=_extract_pucobre_periods_from_page,
        source_name="pucobre.cl",
    )
    found_periods = list_periods_from_page(extractor, headless=headless)
    logger.info("Found %d periods on Pucobre.cl", len(found_periods))
    return found_periods


def check_pucobre_availability(
    year: int,
    quarter: int,
    headless: bool = True,
    config: dict | None = None,
) -> bool:
    """Check if a specific period is available on Pucobre.cl.

    Args:
        year: Target year
        quarter: Target quarter (1-4)
        headless: Run browser in headless mode
        config: Configuration dict, or None to load from file

    Returns:
        True if the period is available

    """
    pucobre_url, quarter_to_date = _get_pucobre_config(config)
    date_str = f"{quarter_to_date[quarter]}-{year}"
    expected_link = f"Estados Financieros {date_str}"

    with browser_session(headless=headless) as (_browser, _context, page):
        page.goto(pucobre_url, wait_until="networkidle", timeout=60000)
        time.sleep(2)

        link = page.get_by_role("link", name=expected_link)
        available = link.count() > 0

        logger.debug("Pucobre availability for %s Q%s: %s", year, quarter, available)
        return available

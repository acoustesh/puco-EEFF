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
from pathlib import Path

from playwright.sync_api import Page
from playwright.sync_api import TimeoutError as PlaywrightTimeout

from puco_eeff.config import get_period_paths, setup_logging
from puco_eeff.scraper.browser import browser_session

logger = setup_logging(__name__)

# Pucobre Estados Financieros page
PUCOBRE_EEFF_URL = (
    "https://www.pucobre.cl/OpenDocs/asp/pagDefault.asp?"
    "boton=Doc51&argInstanciaId=51&argCarpetaId=32&"
    "argTreeNodosAbiertos=(32)&argTreeNodoActual=32&argTreeNodoSel=32"
)

# Quarter to end-of-period date mapping
QUARTER_TO_DATE = {
    1: "31-03",  # Q1 ends March 31
    2: "30-06",  # Q2 ends June 30
    3: "30-09",  # Q3 ends September 30
    4: "31-12",  # Q4 ends December 31
}


@dataclass
class PucobreDownloadResult:
    """Result of a Pucobre download operation."""

    success: bool
    file_path: Path | None  # Estados Financieros PDF
    file_size: int | None
    error: str | None = None
    source: str = "pucobre.cl"
    # Additional files extracted from combined PDF
    analisis_razonado_path: Path | None = None
    analisis_razonado_size: int | None = None
    combined_pdf_path: Path | None = None  # Original combined file (for reference)


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
        text = page.get_text().upper()

        # Look for the Análisis Razonado title page
        # It typically contains "ANALISIS RAZONADO" as a prominent heading
        if "ANALISIS RAZONADO" in text:
            # Verify this looks like a title page (short text, near start)
            text_lines = [line.strip() for line in page.get_text().split("\n") if line.strip()]
            # Title pages typically have few lines and contain the company name
            if len(text_lines) < 20 or "SOCIEDAD PUNTA DEL COBRE" in text.upper():
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
            f"Split PDF: Estados Financieros ({split_page} pages), Análisis Razonado ({total_pages - split_page} pages)"
        )
        return True, None

    except Exception as e:
        return False, str(e)


def download_from_pucobre(
    year: int,
    quarter: int,
    headless: bool = True,
    split_pdf: bool = True,
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

    Returns:
        PucobreDownloadResult with status and file paths
    """
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
    date_str = f"{QUARTER_TO_DATE[quarter]}-{year}"
    expected_link = f"Estados Financieros {date_str}"

    logger.info(f"Attempting to download from Pucobre.cl: {expected_link}")

    with browser_session(headless=headless) as (browser, context, page):
        # Track PDF URLs from responses
        pdf_url: str | None = None
        pdf_content: bytes | None = None

        def capture_pdf_response(response):
            nonlocal pdf_url, pdf_content
            content_type = response.headers.get("content-type", "")
            if "pdf" in content_type.lower():
                pdf_url = response.url
                with contextlib.suppress(Exception):
                    pdf_content = response.body()

        context.on("response", capture_pdf_response)

        try:
            # Navigate to Pucobre Estados Financieros page
            logger.debug(f"Navigating to: {PUCOBRE_EEFF_URL}")
            page.goto(PUCOBRE_EEFF_URL, wait_until="networkidle", timeout=60000)

            # Wait for the table to load
            time.sleep(2)

            # Find the link for the requested period
            link = page.get_by_role("link", name=expected_link)

            if link.count() == 0:
                logger.warning(f"Link not found: {expected_link}")
                # Try alternative search - look for partial match
                link = _find_period_link(page, year, quarter)
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
                logger.debug(f"PDF URL captured: {pdf_url}")
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
            logger.info(f"Downloaded combined PDF from Pucobre: {combined_filename} ({combined_size:,} bytes)")

            # Split the PDF if requested
            if split_pdf:
                split_page = _find_analisis_razonado_page(combined_path)

                if split_page is not None:
                    success, error = _split_combined_pdf(combined_path, eeff_path, ar_path, split_page)

                    if success:
                        eeff_size = eeff_path.stat().st_size
                        ar_size = ar_path.stat().st_size

                        logger.info(
                            f"Split into: {eeff_filename} ({eeff_size:,} bytes), {ar_filename} ({ar_size:,} bytes)"
                        )

                        return PucobreDownloadResult(
                            success=True,
                            file_path=eeff_path,
                            file_size=eeff_size,
                            analisis_razonado_path=ar_path,
                            analisis_razonado_size=ar_size,
                            combined_pdf_path=combined_path,
                        )
                    else:
                        logger.warning(f"Failed to split PDF: {error}")
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
            logger.error(f"Timeout downloading from Pucobre: {e}")
            return PucobreDownloadResult(
                success=False,
                file_path=None,
                file_size=None,
                error=f"Timeout: {e}",
            )
        except Exception as e:
            logger.error(f"Error downloading from Pucobre: {e}")
            return PucobreDownloadResult(
                success=False,
                file_path=None,
                file_size=None,
                error=str(e),
            )


def _find_period_link(page: Page, year: int, quarter: int):
    """Find the download link for a specific period using various patterns.

    Args:
        page: Playwright page instance
        year: Target year
        quarter: Target quarter (1-4)

    Returns:
        Locator for the link, or None if not found
    """
    date_str = f"{QUARTER_TO_DATE[quarter]}-{year}"

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
            link = page.locator(f"a:has-text('{QUARTER_TO_DATE[quarter]}-{year}')")

        if link.count() > 0:
            logger.debug(f"Found link with pattern: {pattern}")
            return link

    return None


def list_pucobre_periods(headless: bool = True) -> list[dict]:
    """List all available periods from Pucobre.cl.

    Args:
        headless: Run browser in headless mode

    Returns:
        List of available periods with year, quarter, and link text
    """
    periods: list[dict] = []

    with browser_session(headless=headless) as (browser, context, page):
        page.goto(PUCOBRE_EEFF_URL, wait_until="networkidle", timeout=60000)
        time.sleep(2)

        # Find all "Estados Financieros" links
        links = page.locator("a:has-text('Estados Financieros')")
        count = links.count()

        for i in range(count):
            link_text = links.nth(i).inner_text()

            # Parse date from link text: "Estados Financieros DD-MM-YYYY"
            match = re.search(r"(\d{2})-(\d{2})-(\d{4})", link_text)
            if match:
                day, month, year = match.groups()

                # Map end-month to quarter
                month_to_quarter = {"03": 1, "06": 2, "09": 3, "12": 4}
                quarter = month_to_quarter.get(month)

                if quarter:
                    periods.append({
                        "year": int(year),
                        "quarter": quarter,
                        "link_text": link_text,
                        "source": "pucobre.cl",
                    })

    logger.info(f"Found {len(periods)} periods on Pucobre.cl")
    return periods


def check_pucobre_availability(year: int, quarter: int, headless: bool = True) -> bool:
    """Check if a specific period is available on Pucobre.cl.

    Args:
        year: Target year
        quarter: Target quarter (1-4)
        headless: Run browser in headless mode

    Returns:
        True if the period is available
    """
    date_str = f"{QUARTER_TO_DATE[quarter]}-{year}"
    expected_link = f"Estados Financieros {date_str}"

    with browser_session(headless=headless) as (browser, context, page):
        page.goto(PUCOBRE_EEFF_URL, wait_until="networkidle", timeout=60000)
        time.sleep(2)

        link = page.get_by_role("link", name=expected_link)
        available = link.count() > 0

        logger.debug(f"Pucobre availability for {year} Q{quarter}: {available}")
        return available

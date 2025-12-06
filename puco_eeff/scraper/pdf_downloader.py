"""PDF downloader for Pucobre financial statements."""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import Page

from puco_eeff.config import get_config, get_period_paths, setup_logging
from puco_eeff.scraper.browser import browser_session

# wait_for_download will be used when navigation is implemented
# from puco_eeff.scraper.browser import wait_for_download

logger = setup_logging(__name__)


def download_pdf_from_pucobre(
    year: int,
    quarter: int,
    headless: bool = True,
) -> Path:
    """Download EEFF PDF from Pucobre website.

    Navigates to the Pucobre document portal and downloads the
    Estados Financieros PDF for the specified period.

    Args:
        year: Year of the financial statement (e.g., 2024)
        quarter: Quarter (1-4)
        headless: Run browser in headless mode

    Returns:
        Path to the downloaded PDF file
    """
    config = get_config()
    paths = get_period_paths(year, quarter)

    base_url = config["sources"]["pdf"]["base_url"]
    output_dir = paths["raw_pdf"]
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"EEFF_{year}_Q{quarter}.pdf"
    output_path = output_dir / filename

    logger.info(f"Downloading PDF for {year} Q{quarter} from Pucobre")
    logger.debug(f"URL: {base_url}")

    with browser_session(headless=headless) as (browser, page):
        # Navigate to the document portal
        page.goto(base_url, wait_until="networkidle")
        logger.debug("Loaded document portal")

        # TODO: Implement navigation to find the correct document
        # This will need to be customized based on the actual page structure
        # The navigation steps should:
        # 1. Find the document list/tree
        # 2. Locate the EEFF document for the specified period
        # 3. Click to download

        # Placeholder: The actual implementation will be developed
        # during the instruction execution phase
        _navigate_and_download(page, year, quarter, output_path)

    logger.info(f"Downloaded: {output_path}")
    return output_path


def _navigate_and_download(page: Page, year: int, quarter: int, output_path: Path) -> None:
    """Navigate the Pucobre portal and download the PDF.

    This is a placeholder that will be implemented during development.

    Args:
        page: Playwright page instance
        year: Target year
        quarter: Target quarter
        output_path: Path to save the downloaded file
    """
    # TODO: Implement actual navigation logic
    # This will be developed based on exploring the actual website structure
    # During instruction execution:
    # 1. Inspect the page structure
    # 2. Identify clickable elements for navigation
    # 3. Find the download link/button
    # 4. Use wait_for_download to capture the file

    logger.warning("PDF download navigation not yet implemented")
    logger.info("Please implement _navigate_and_download based on actual site structure")

"""XBRL downloader from CMF Chile."""

from __future__ import annotations

import zipfile
from pathlib import Path

from playwright.sync_api import Page

from puco_eeff.config import get_config, get_period_paths, setup_logging
from puco_eeff.scraper.browser import browser_session

logger = setup_logging(__name__)


def download_xbrl_from_cmf(
    year: int,
    quarter: int,
    headless: bool = True,
) -> Path:
    """Download XBRL/XML from CMF Chile website.

    Navigates to the CMF Chile portal, selects the appropriate filters
    (Consolidado, IFRS), and downloads the XBRL ZIP file for the specified period.

    Args:
        year: Year of the financial statement (e.g., 2024)
        quarter: Quarter (1-4)
        headless: Run browser in headless mode

    Returns:
        Path to the extracted XBRL/XML file
    """
    config = get_config()
    paths = get_period_paths(year, quarter)

    base_url = config["sources"]["xbrl"]["base_url"]
    filters = config["sources"]["xbrl"]["filters"]

    output_dir = paths["raw_xbrl"]
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_filename = f"EEFF_{year}_Q{quarter}_xbrl.zip"
    zip_path = output_dir / zip_filename

    logger.info(f"Downloading XBRL for {year} Q{quarter} from CMF Chile")
    logger.debug(f"URL: {base_url}")
    logger.debug(f"Filters: {filters}")

    with browser_session(headless=headless) as (browser, page):
        # Navigate to the CMF portal
        page.goto(base_url, wait_until="networkidle")
        logger.debug("Loaded CMF portal")

        # Apply filters and download
        _apply_filters_and_download(page, year, quarter, filters, zip_path)

    # Extract the ZIP file
    xml_path = _extract_xbrl_zip(zip_path, output_dir, year, quarter)

    logger.info(f"Downloaded and extracted: {xml_path}")
    return xml_path


def _apply_filters_and_download(
    page: Page,
    year: int,
    quarter: int,
    filters: dict[str, str],
    zip_path: Path,
) -> None:
    """Apply filters on CMF portal and download the XBRL ZIP.

    This is a placeholder that will be implemented during development.

    Args:
        page: Playwright page instance
        year: Target year
        quarter: Target quarter
        filters: Filter settings (tipo_balance, tipo_norma)
        zip_path: Path to save the downloaded ZIP file
    """
    # TODO: Implement actual filter selection and download logic
    # The CMF portal requires:
    # 1. Select "Tipo Balance" = "Consolidado"
    # 2. Select "Tipo Norma" = "Estandar IFRS"
    # 3. Select the date/period
    # 4. Click download to get the ZIP file

    logger.warning("XBRL download navigation not yet implemented")
    logger.info("Please implement _apply_filters_and_download based on actual site structure")
    logger.debug(f"Target filters: tipo_balance={filters['tipo_balance']}, "
                 f"tipo_norma={filters['tipo_norma']}")


def _extract_xbrl_zip(zip_path: Path, output_dir: Path, year: int, quarter: int) -> Path:
    """Extract XBRL XML from downloaded ZIP file.

    Args:
        zip_path: Path to the ZIP file
        output_dir: Directory to extract to
        year: Year for naming
        quarter: Quarter for naming

    Returns:
        Path to the extracted XML file
    """
    if not zip_path.exists():
        logger.warning(f"ZIP file not found: {zip_path}")
        # Return expected path for placeholder
        return output_dir / f"EEFF_{year}_Q{quarter}.xml"

    xml_filename = f"EEFF_{year}_Q{quarter}.xml"
    xml_path = output_dir / xml_filename

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find the main XBRL/XML file in the archive
        xml_files = [f for f in zf.namelist() if f.endswith((".xml", ".xbrl"))]

        if not xml_files:
            logger.error("No XML/XBRL files found in ZIP archive")
            raise ValueError("No XML/XBRL files found in ZIP archive")

        # Extract the first XML file (usually the main one)
        main_xml = xml_files[0]
        logger.debug(f"Extracting: {main_xml}")

        with zf.open(main_xml) as source, open(xml_path, "wb") as target:
            target.write(source.read())

    logger.info(f"Extracted XBRL to: {xml_path}")
    return xml_path

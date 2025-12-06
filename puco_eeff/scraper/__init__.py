"""Scraper module for downloading PDFs and XBRL files."""

from puco_eeff.scraper.browser import create_browser, create_browser_context
from puco_eeff.scraper.downloader import download_file
from puco_eeff.scraper.pdf_downloader import download_pdf_from_pucobre
from puco_eeff.scraper.xbrl_downloader import download_xbrl_from_cmf

__all__ = [
    "create_browser",
    "create_browser_context",
    "download_file",
    "download_pdf_from_pucobre",
    "download_xbrl_from_cmf",
]

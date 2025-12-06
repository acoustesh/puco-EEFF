"""Scraper module for downloading financial documents from CMF Chile."""

from puco_eeff.scraper.browser import create_browser, create_browser_context
from puco_eeff.scraper.cmf_downloader import (
    DOCUMENT_TYPES,
    DownloadResult,
    download_all_documents,
    download_single_document,
    list_available_periods,
)
from puco_eeff.scraper.downloader import download_file

# Legacy imports for backward compatibility
from puco_eeff.scraper.pdf_downloader import download_pdf_from_pucobre
from puco_eeff.scraper.xbrl_downloader import download_xbrl_from_cmf

__all__ = [
    # Browser utilities
    "create_browser",
    "create_browser_context",
    # New unified CMF downloader (recommended)
    "DOCUMENT_TYPES",
    "DownloadResult",
    "download_all_documents",
    "download_single_document",
    "list_available_periods",
    # Generic file download
    "download_file",
    # Legacy (deprecated - use download_all_documents instead)
    "download_pdf_from_pucobre",
    "download_xbrl_from_cmf",
]

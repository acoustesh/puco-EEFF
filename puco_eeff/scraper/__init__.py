"""Scraper module for downloading financial documents from CMF Chile and Pucobre.cl.

Primary downloaders:
- download_all_documents: Download all documents from CMF Chile (with Pucobre.cl fallback)
- download_single_document: Download a specific document type
- download_from_pucobre: Download directly from Pucobre.cl (fallback source)

Files are saved to separate directories:
- data/raw/pdf/ - PDF documents (Análisis Razonado, Estados Financieros PDF)
- data/raw/xbrl/ - XBRL/XML documents (Estados Financieros XBRL)
"""

from puco_eeff.scraper.browser import create_browser, create_browser_context
from puco_eeff.scraper.cmf_downloader import (
    DOCUMENT_TYPES,
    DownloadResult,
    download_all_documents,
    download_single_document,
    list_available_periods,
)
from puco_eeff.scraper.downloader import download_file
from puco_eeff.scraper.pucobre_downloader import (
    PucobreDownloadResult,
    check_pucobre_availability,
    download_from_pucobre,
    list_pucobre_periods,
)

__all__ = [
    # Unified CMF downloader (primary)
    "DOCUMENT_TYPES",
    "DownloadResult",
    # Pucobre.cl fallback downloader
    "PucobreDownloadResult",
    "check_pucobre_availability",
    # Browser utilities
    "create_browser",
    "create_browser_context",
    "download_all_documents",
    # Generic file download
    "download_file",
    "download_from_pucobre",
    "download_single_document",
    "list_available_periods",
    "list_pucobre_periods",
]

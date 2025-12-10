"""Browser automation utilities using Playwright.

This module provides reusable browser session management and download utilities
for web scraping. It wraps Playwright's sync API with context managers for
clean resource handling.

Main components:
- browser_session: Context manager for complete browser lifecycle
- wait_for_download: Handle file downloads with proper waiting
- list_periods_from_page: Extract period data using configurable extractors

Notes
-----
Uses Chromium headless mode by default. Chrome args disable GPU and sandbox
for compatibility with containerized/server environments (Ubuntu headless).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from playwright.sync_api import Playwright


@dataclass
class PeriodExtractor:
    """Configuration for extracting periods from a web page.

    Attributes
    ----------
    url : str
        Target URL to scrape for period data.
    page_extractor : Callable[[Page], list[dict]]
        Function that extracts period dicts from a loaded Page.
    source_name : str
        Identifier for the data source (e.g., "cmf", "pucobre.cl").
    """

    url: str
    page_extractor: Callable[[Page], list[dict]]
    source_name: str = ""


def create_browser(playwright: Playwright, headless: bool = True) -> Browser:
    """Create a Chromium browser instance.

    Parameters
    ----------
    playwright : Playwright
        Playwright instance from sync_playwright context.
    headless : bool, optional
        Run browser in headless mode. Default True for server use.

    Returns
    -------
    Browser
        Configured Chromium browser instance.

    Notes
    -----
    Chrome args (--disable-gpu, --no-sandbox, --disable-dev-shm-usage)
    are required for headless server compatibility on Ubuntu.
    """
    # Chrome args ensure compatibility with containerized environments
    return playwright.chromium.launch(
        headless=headless,
        args=[
            "--disable-gpu",  # No GPU in headless environments
            "--disable-dev-shm-usage",  # Prevents /dev/shm overflow in Docker
            "--no-sandbox",  # Required for root/containerized execution
        ],
    )


def create_browser_context(browser: Browser) -> BrowserContext:
    """Create a browser context with appropriate settings.

    Parameters
    ----------
    browser : Browser
        Browser instance to create context on.

    Returns
    -------
    BrowserContext
        Context configured with realistic viewport and user agent.

    Notes
    -----
    User agent mimics Chrome on Windows to avoid bot detection.
    Viewport is set to 1920x1080 for proper rendering of full pages.
    """
    # Realistic user agent prevents bot detection on some sites
    return browser.new_context(
        viewport={"width": 1920, "height": 1080},
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        accept_downloads=True,  # Enable file downloads in this context
    )


@contextmanager
def browser_session(
    headless: bool = True,
) -> Generator[tuple[Browser, BrowserContext, Page], None, None]:
    """Context manager for a complete browser session.

    Parameters
    ----------
    headless : bool, optional
        Run browser in headless mode. Default True.

    Yields
    ------
    tuple[Browser, BrowserContext, Page]
        Tuple containing browser, context, and initial page for interaction.

    Notes
    -----
    Ensures proper cleanup of all browser resources even on exceptions.
    Playwright is started and stopped within this context.
    """
    # Use sync_playwright as outer context for proper resource management
    with sync_playwright() as playwright:
        browser = create_browser(playwright, headless=headless)
        context = create_browser_context(browser)
        page = context.new_page()

        try:
            yield browser, context, page
        finally:
            # Cleanup in reverse order: context closes pages, browser closes contexts
            context.close()
            browser.close()


def wait_for_download(page: Page, trigger_action: Any, download_path: str) -> str:
    """Wait for a download to complete and save to specified path.

    Parameters
    ----------
    page : Page
        Page instance where download will be triggered.
    trigger_action : Callable
        Function that triggers the download (e.g., clicking a link).
    download_path : str
        Absolute path where the downloaded file should be saved.

    Returns
    -------
    str
        Path to the saved downloaded file.

    Notes
    -----
    Uses Playwright's expect_download to properly wait for download events.
    The trigger_action is called inside the expect_download context.
    """
    # expect_download waits for download event triggered by action
    with page.expect_download() as download_info:
        trigger_action()

    download = download_info.value
    download.save_as(download_path)

    return download_path


def list_periods_from_page(
    extractor: PeriodExtractor,
    headless: bool = True,
) -> list[dict]:
    """List periods from a page using a configured extractor.

    Parameters
    ----------
    extractor : PeriodExtractor
        Configuration specifying URL and extraction logic.
    headless : bool, optional
        Run browser in headless mode. Default True.

    Returns
    -------
    list[dict]
        List of period dicts extracted from the page.

    Notes
    -----
    Waits for network idle before extraction to ensure page is fully loaded.
    Timeout is 60 seconds for slow-loading pages (CMF can be slow).
    """
    with browser_session(headless=headless) as (_browser, _context, page):
        # networkidle ensures JavaScript-rendered content is available
        page.goto(extractor.url, wait_until="networkidle", timeout=60000)
        return extractor.page_extractor(page)

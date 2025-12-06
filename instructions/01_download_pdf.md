# 01 - Download PDF from Pucobre

## Objective

Download the Estados Financieros (EEFF) PDF document from Pucobre's public document portal for a specified period.

## Prerequisites

- Browser installed (`./setup_browser.sh` executed)
- Virtual environment active (`poetry shell` or `poetry run`)

## Source

- **URL**: https://www.pucobre.cl/OpenDocs/asp/pagDefault.asp?boton=Doc51&argInstanciaId=51&argCarpetaId=32&argTreeNodosAbiertos=(32)&argTreeNodoActual=32&argTreeNodoSel=32
- **Type**: Document portal with tree navigation

## Steps

### 1. Interactive Period Selection

Run the period selector to choose year and quarter:

```python
import questionary

def select_period() -> tuple[int, int]:
    """Prompt user to select period."""
    year = questionary.text(
        "Enter year (e.g., 2024):",
        validate=lambda x: x.isdigit() and 2020 <= int(x) <= 2030
    ).ask()
    
    quarter = questionary.select(
        "Select quarter:",
        choices=["Q1", "Q2", "Q3", "Q4"]
    ).ask()
    
    return int(year), int(quarter[1])

year, quarter = select_period()
print(f"Selected period: {year} Q{quarter}")
```

### 2. Navigate to Document Portal

```python
from puco_eeff.scraper.browser import browser_session
from puco_eeff.config import get_config

config = get_config()
url = config["sources"]["pdf"]["base_url"]

with browser_session(headless=False) as (browser, page):  # headless=False for exploration
    page.goto(url, wait_until="networkidle")
    
    # Explore the page structure
    # TODO: Identify the document tree/list
    # TODO: Find links to EEFF documents
    # TODO: Match documents by period (year, quarter)
    
    input("Press Enter after exploring the page structure...")
```

### 3. Identify Document Links

**Tasks to complete:**

1. Inspect the document tree structure
2. Identify how documents are organized (by year? by type?)
3. Find the naming pattern for EEFF documents
4. Determine the click sequence to reach the PDF

**Expected patterns to look for:**
- Links containing "EEFF" or "Estados Financieros"
- Date/period indicators in document names
- Download buttons or direct PDF links

### 4. Implement Download Logic

Once the page structure is understood, update `puco_eeff/scraper/pdf_downloader.py`:

```python
# In _navigate_and_download function:

# Example navigation (adapt to actual structure):
# 1. Find the document list container
# 2. Locate the correct year folder
# 3. Find the EEFF document for the quarter
# 4. Click to download

# Use wait_for_download for file capture:
from puco_eeff.scraper.browser import wait_for_download

def download_trigger():
    page.click("selector_for_download_button")

wait_for_download(page, download_trigger, str(output_path))
```

### 5. Verify Download

```python
from puco_eeff.config import get_period_paths

paths = get_period_paths(year, quarter)
pdf_path = paths["raw_pdf"] / f"EEFF_{year}_Q{quarter}.pdf"

if pdf_path.exists():
    print(f"✓ PDF downloaded: {pdf_path}")
    print(f"  Size: {pdf_path.stat().st_size:,} bytes")
else:
    print("✗ Download failed")
```

## Output

- Downloaded PDF saved to: `data/raw/pdf/EEFF_YYYY_QN.pdf`

## Next Steps

After completing this instruction:
1. Document the actual selectors and navigation steps in the code
2. Test with headless=True for production use
3. Proceed to `02_download_xbrl.md`

## Troubleshooting

- **Page doesn't load**: Check network connectivity, try increasing wait time
- **Document not found**: Verify the period exists in the portal
- **Download fails**: Check if popup blocker is interfering, ensure accept_downloads=True in context

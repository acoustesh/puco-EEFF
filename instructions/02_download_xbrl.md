# 02 - Download XBRL from CMF Chile

## Objective

Download the XBRL/XML financial data from CMF Chile's portal for the same period as the PDF, using filters for Consolidado and Estándar IFRS.

## Prerequisites

- Period selected (same as 01_download_pdf.md)
- Browser installed

## Source

- **URL**: https://www.cmfchile.cl/institucional/mercados/entidad.php?mercado=V&rut=96561560&grupo=&tipoentidad=RVEMI&row=&vig=VI&control=svs&pestania=3
- **Entity**: Pucobre (RUT 96561560)
- **Filters Required**:
  - Tipo Balance: Consolidado
  - Tipo Norma: Estándar IFRS

## Steps

### 1. Navigate to CMF Portal

```python
from puco_eeff.scraper.browser import browser_session
from puco_eeff.config import get_config

config = get_config()
url = config["sources"]["xbrl"]["base_url"]
filters = config["sources"]["xbrl"]["filters"]

with browser_session(headless=False) as (browser, page):  # headless=False for exploration
    page.goto(url, wait_until="networkidle")
    
    print("Page loaded. Exploring filter controls...")
    input("Press Enter after exploring the page...")
```

### 2. Identify Filter Controls

**Tasks to complete:**

1. Locate "Tipo Balance" dropdown/select
2. Locate "Tipo Norma" dropdown/select
3. Find date/period selector
4. Identify the search/filter button
5. Find the download link/button for XBRL ZIP

**Expected element patterns:**
- `<select>` elements for filters
- Date picker or period dropdown
- "Descargar" or "Download" button
- Links to ZIP files containing XBRL

### 3. Apply Filters

```python
# Example filter application (adapt to actual selectors):

# Select Tipo Balance = Consolidado
page.select_option("select#tipoBalance", "Consolidado")
# or
page.click("text=Consolidado")

# Select Tipo Norma = Estándar IFRS
page.select_option("select#tipoNorma", "Estandar IFRS")

# Select period
# This might be a date picker or dropdown - identify the format
page.fill("input#fecha", "31-03-2024")  # End of Q1 2024

# Click search/apply button
page.click("button#buscar")
page.wait_for_load_state("networkidle")
```

### 4. Download XBRL ZIP

```python
from pathlib import Path
from puco_eeff.config import get_period_paths

paths = get_period_paths(year, quarter)
zip_path = paths["raw_xbrl"] / f"EEFF_{year}_Q{quarter}_xbrl.zip"

# Trigger download
with page.expect_download() as download_info:
    page.click("selector_for_xbrl_download")

download = download_info.value
download.save_as(str(zip_path))

print(f"Downloaded: {zip_path}")
```

### 5. Extract XBRL from ZIP

```python
import zipfile

output_dir = paths["raw_xbrl"]
xml_path = output_dir / f"EEFF_{year}_Q{quarter}.xml"

with zipfile.ZipFile(zip_path, 'r') as zf:
    # List contents
    print("ZIP contents:")
    for name in zf.namelist():
        print(f"  - {name}")
    
    # Find and extract main XBRL file
    xbrl_files = [f for f in zf.namelist() if f.endswith(('.xml', '.xbrl'))]
    
    if xbrl_files:
        main_file = xbrl_files[0]
        with zf.open(main_file) as src, open(xml_path, 'wb') as dst:
            dst.write(src.read())
        print(f"Extracted: {xml_path}")
```

### 6. Verify XBRL Content

```python
from puco_eeff.extractor.xbrl_parser import parse_xbrl_file

# Quick validation
try:
    data = parse_xbrl_file(xml_path)
    print(f"✓ XBRL parsed successfully")
    print(f"  Facts: {len(data['facts'])}")
    print(f"  Contexts: {len(data['contexts'])}")
except Exception as e:
    print(f"✗ XBRL parsing failed: {e}")
```

## Output

- ZIP file: `data/raw/xbrl/EEFF_YYYY_QN_xbrl.zip`
- Extracted XML: `data/raw/xbrl/EEFF_YYYY_QN.xml`

## Notes on CMF Portal

- The CMF portal may require waiting for AJAX requests
- Some filters may trigger page reloads
- The download might be a direct link or trigger a file generation

## Quarter End Dates Reference

| Quarter | End Date |
|---------|----------|
| Q1 | March 31 |
| Q2 | June 30 |
| Q3 | September 30 |
| Q4 | December 31 |

## Next Steps

After completing this instruction:
1. Document actual selectors in `puco_eeff/scraper/xbrl_downloader.py`
2. Both PDF and XBRL should now be in `data/raw/`
3. Proceed to `03_locate_sheet1.md`

## Troubleshooting

- **Filters not working**: Check if page uses JavaScript frameworks, may need to wait for elements
- **No download button**: The period might not be available yet
- **ZIP extraction fails**: File might be corrupted, re-download

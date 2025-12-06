# 01 - Download All Documents from CMF Chile

## Objective

Download all three financial document types from CMF Chile for a specified period:
1. **Análisis Razonado** (PDF) - Management Discussion & Analysis
2. **Estados Financieros** (PDF) - Financial Statements 
3. **Estados Financieros** (XBRL) - Financial Statements in structured XML format

**Single source**: All documents come from the same CMF Chile portal page.

## Prerequisites

- Browser installed (`playwright install chromium`)
- Virtual environment active (`poetry shell` or `poetry run`)
- API keys configured in `.env` (for later OCR steps)

## Source

- **URL**: https://www.cmfchile.cl/institucional/mercados/entidad.php?mercado=V&rut=96561560&grupo=&tipoentidad=RVEMI&row=&vig=VI&control=svs&pestania=3
- **Entity**: Sociedad Punta del Cobre S.A. (RUT: 96561560)
- **Type**: CMF Chile financial information portal

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
        choices=["Q1 (March)", "Q2 (June)", "Q3 (September)", "Q4 (December)"]
    ).ask()

    return int(year), int(quarter[1])

year, quarter = select_period()
print(f"Selected period: {year} Q{quarter}")
```

### 2. Download All Documents (Recommended)

Use the unified downloader to get all three files at once:

```python
from puco_eeff.scraper import download_all_documents

# Download all documents for the selected period
results = download_all_documents(
    year=year,
    quarter=quarter,
    headless=True,  # Set to False for debugging
    tipo="Consolidado",  # or "Individual"
    tipo_norma="Estándar IFRS",  # or "Norma Chilena"
)

# Check results
print("\n=== Download Results ===")
for r in results:
    status = "✓" if r.success else "✗"
    size = f"{r.file_size:,} bytes" if r.file_size else "N/A"
    path = r.file_path.name if r.file_path else "N/A"
    print(f"  {status} {r.document_type}: {path} ({size})")
    if r.error:
        print(f"      Error: {r.error}")
```

### 3. Alternative: Download Individual Documents

If you only need a specific document:

```python
from puco_eeff.scraper import download_single_document

# Download only the Análisis Razonado
result = download_single_document(
    year=2024,
    quarter=3,
    document_type="analisis_razonado",  # Options: analisis_razonado, estados_financieros_pdf, estados_financieros_xbrl
    headless=True,
)

if result.success:
    print(f"Downloaded: {result.file_path}")
else:
    print(f"Error: {result.error}")
```

### 4. Verify Downloads

```python
from puco_eeff.config import get_period_paths

paths = get_period_paths(year, quarter)
pdf_dir = paths["raw_pdf"]

print(f"\n=== Downloaded Files in {pdf_dir} ===")
for f in pdf_dir.iterdir():
    print(f"  {f.name}: {f.stat().st_size:,} bytes")
```

### 5. List Available Periods (Optional)

Discover what periods are available for download:

```python
from puco_eeff.scraper import list_available_periods

periods = list_available_periods(headless=True)
print("Available periods:")
for p in periods[:10]:  # Show first 10
    print(f"  {p['year']} Q{p['quarter']} (month: {p['month']})")
```

## Filter Options

The CMF portal supports these filters:

| Filter | Options | Description |
|--------|---------|-------------|
| `tipo` | `Consolidado`, `Individual` | Balance type |
| `tipo_norma` | `Estándar IFRS`, `Norma Chilena` | Accounting standard |

## Output Files

Downloads are saved to `data/raw/pdf/`:

| Document | Filename | Description |
|----------|----------|-------------|
| Análisis Razonado | `analisis_razonado_YYYY_QN.pdf` | MD&A narrative |
| Estados Financieros PDF | `estados_financieros_YYYY_QN.pdf` | Full financial statements |
| Estados Financieros XBRL | `estados_financieros_YYYY_QN_xbrl.zip` | XBRL package |
| Extracted XML | `estados_financieros_YYYY_QN.xml` | Extracted from ZIP |

## How It Works

The downloader performs these steps:

1. **Navigate** to CMF Chile portal
2. **Apply filters**:
   - Select month (03/06/09/12 for Q1/Q2/Q3/Q4)
   - Select year
   - Select tipo (Consolidado/Individual)
   - Select tipo_norma (IFRS/Norma Chilena)
3. **Click "Consultar"** to load results
4. **Download** each document by clicking its link:
   - "Análisis Razonado" → PDF
   - "Estados financieros (PDF)" → PDF
   - "Estados financieros (XBRL)" → ZIP
5. **Extract** the XBRL ZIP to get the XML file

## Next Steps

After completing this instruction:
1. Verify all files downloaded successfully
2. Proceed to `03_parse_xbrl.md` to extract structured data from XBRL
3. Or proceed to `04_ocr_pdf.md` if you need to extract data from PDFs

## Troubleshooting

### Page doesn't load
- Check network connectivity
- Try increasing timeout: `page.goto(url, timeout=90000)`
- Run with `headless=False` to see what's happening

### Document not found
- Verify the period exists: use `list_available_periods()`
- Check if filters match (Consolidado vs Individual)
- Some historical periods may not be available

### Download timeout
- Increase timeout: `expect_download(timeout=120000)`
- Check if CMF Chile is experiencing high traffic
- Try downloading individual documents instead of all at once

### XBRL extraction fails
- Check if the ZIP file is valid
- Ensure sufficient disk space
- Some periods may have different ZIP structure

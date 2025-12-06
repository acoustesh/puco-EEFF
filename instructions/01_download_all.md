# 01 - Download All Documents from CMF Chile (with Pucobre.cl Fallback)

## Objective

Download all three financial document types from CMF Chile for a specified period:
1. **Análisis Razonado** (PDF) - Management Discussion & Analysis
2. **Estados Financieros** (PDF) - Financial Statements with detailed notes
3. **Estados Financieros** (XBRL) - Financial Statements in structured XML format

**Primary source**: CMF Chile portal
**Fallback source**: Pucobre.cl (when CMF Chile doesn't have the data)

## What Each Document Contains

| Document | Format | Key Data | Use For |
|----------|--------|----------|---------|
| Análisis Razonado | PDF | Narrative, cost analysis, KPIs | Context, EBITDA calculation |
| Estados Financieros | PDF | Full statements + **Nota 21 & 22** (detailed cost breakdown) | Line-item extraction |
| Estados Financieros | XBRL | Structured data, aggregate totals | Validation, primary totals |

**Important**: The **detailed cost breakdown** (11 items for Costo de Venta, 6 for Gastos Admin) is ONLY in the PDF (Nota 21-22, page ~71), not in XBRL.

## Data Sources

### Primary: CMF Chile
- **URL**: https://www.cmfchile.cl/institucional/mercados/entidad.php?mercado=V&rut=96561560&grupo=&tipoentidad=RVEMI&row=&vig=VI&control=svs&pestania=3
- **Entity**: Sociedad Punta del Cobre S.A. (RUT: 96561560)
- **Documents available**: All three as separate files (PDF, XBRL, Análisis Razonado)
- **Limitation**: Q1 data may not be available until Q2 is published

### Fallback: Pucobre.cl
- **URL**: https://www.pucobre.cl/OpenDocs/asp/pagDefault.asp?boton=Doc51&argInstanciaId=51&argCarpetaId=32
- **Entity**: Pucobre company website
- **Documents available**: Combined PDF containing both Estados Financieros AND Análisis Razonado
- **Advantage**: Q1 data available before CMF publishes Q2
- **Note**: The combined PDF is automatically split into separate files

## Pucobre.cl Combined PDF Structure

The Pucobre.cl website provides a single combined PDF that contains:

1. **Estados Financieros** (pages 1-95 typically)
   - Full financial statements
   - Notes including Nota 21 & 22 with detailed cost breakdown
   
2. **Análisis Razonado** (pages 96+ typically)
   - Starts with title page: "ANALISIS RAZONADO A los Estados Financieros al DD de MMM de YYYY"
   - Page numbering restarts at 1
   - Contains ~11 pages of management discussion

**The downloader automatically detects and splits these sections** into separate files matching the CMF naming convention.

## Prerequisites

- Browser installed (\`playwright install chromium\`)
- Virtual environment active (\`poetry shell\` or \`poetry run\`)
- PyMuPDF for PDF splitting (\`pip install PyMuPDF\`)
- API keys configured in \`.env\` (for later OCR steps)

## Steps

### 1. Interactive Period Selection

Run the period selector to choose year and quarter:

\`\`\`python
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
\`\`\`

### 2. Download All Documents (Recommended)

Use the unified downloader to get all three files at once.
**The downloader automatically falls back to Pucobre.cl if CMF Chile doesn't have the data.**

\`\`\`python
from puco_eeff.scraper import download_all_documents

# Download all documents for the selected period
# Automatically tries Pucobre.cl if CMF Chile fails
results = download_all_documents(
    year=year,
    quarter=quarter,
    headless=True,  # Set to False for debugging
    tipo="C",  # "C" for Consolidado, "I" for Individual
    tipo_norma="IFRS",  # "IFRS" for Estándar IFRS, "NCH" for Norma Chilena
    fallback_to_pucobre=True,  # Enable fallback (default)
)

# Check results
print("\n=== Download Results ===")
for r in results:
    status = "✓" if r.success else "✗"
    size = f"{r.file_size:,} bytes" if r.file_size else "N/A"
    path = r.file_path.name if r.file_path else "N/A"
    print(f"  {status} {r.document_type}: {path} ({size}) [source: {r.source}]")
    if r.error:
        print(f"      Error: {r.error}")
\`\`\`

### 3. Download Only from Pucobre.cl (Direct)

If you know CMF Chile won't have the data, use Pucobre.cl directly:

\`\`\`python
from puco_eeff.scraper import download_from_pucobre

# Download directly from Pucobre.cl
# The combined PDF is automatically split into separate files
result = download_from_pucobre(
    year=2024,
    quarter=1,  # Q1 often not on CMF until Q2 is published
    headless=True,
    split_pdf=True,  # Split combined PDF into separate files (default)
)

if result.success:
    print(f"Estados Financieros: {result.file_path} ({result.file_size:,} bytes)")
    if result.analisis_razonado_path:
        print(f"Análisis Razonado: {result.analisis_razonado_path} ({result.analisis_razonado_size:,} bytes)")
    if result.combined_pdf_path:
        print(f"Combined (original): {result.combined_pdf_path}")
else:
    print(f"Error: {result.error}")
\`\`\`

### 4. Check Available Periods on Pucobre.cl

\`\`\`python
from puco_eeff.scraper import list_pucobre_periods, check_pucobre_availability

# List all available periods on Pucobre.cl
periods = list_pucobre_periods(headless=True)
print("Available on Pucobre.cl:")
for p in periods[:10]:
    print(f"  {p['year']} Q{p['quarter']}: {p['link_text']}")

# Check specific period
available = check_pucobre_availability(2024, 1, headless=True)
print(f"\n2024 Q1 available on Pucobre.cl: {available}")
\`\`\`

### 5. Alternative: Download Individual Documents

If you only need a specific document:

\`\`\`python
from puco_eeff.scraper import download_single_document

# Download only the Estados Financieros PDF
# Will try CMF first, then Pucobre.cl fallback
result = download_single_document(
    year=2024,
    quarter=1,
    document_type="estados_financieros_pdf",
    headless=True,
    fallback_to_pucobre=True,  # Enable fallback
)

if result.success:
    print(f"Downloaded: {result.file_path}")
else:
    print(f"Error: {result.error}")
\`\`\`

### 6. Verify Downloads

\`\`\`python
from puco_eeff.config import get_period_paths

paths = get_period_paths(year, quarter)
pdf_dir = paths["raw_pdf"]

print(f"\n=== Downloaded Files in {pdf_dir} ===")
for f in pdf_dir.iterdir():
    print(f"  {f.name}: {f.stat().st_size:,} bytes")
\`\`\`

## Filter Options (CMF Chile)

| Filter | Options | Description |
|--------|---------|-------------|
| \`tipo\` | \`C\` (Consolidado), \`I\` (Individual) | Balance type |
| \`tipo_norma\` | \`IFRS\` (Estándar IFRS), \`NCH\` (Norma Chilena) | Accounting standard |

## Quarter to Date Mapping

| Quarter | CMF Month | Pucobre Date | Report Period |
|---------|-----------|--------------|---------------|
| Q1 | \`03\` (March) | \`31-03-YYYY\` | Jan 1 - Mar 31 |
| Q2 | \`06\` (June) | \`30-06-YYYY\` | Jan 1 - Jun 30 |
| Q3 | \`09\` (September) | \`30-09-YYYY\` | Jan 1 - Sep 30 |
| Q4 | \`12\` (December) | \`31-12-YYYY\` | Jan 1 - Dec 31 |

## Output Files

Downloads are saved to \`data/raw/pdf/\`:

| Document | Filename | Source |
|----------|----------|--------|
| Análisis Razonado | \`analisis_razonado_YYYY_QN.pdf\` | CMF or Pucobre (split) |
| Estados Financieros PDF | \`estados_financieros_YYYY_QN.pdf\` | CMF or Pucobre (split) |
| Estados Financieros XBRL | \`estados_financieros_YYYY_QN_xbrl.zip\` | CMF only |
| Extracted XBRL | \`estados_financieros_YYYY_QN.xbrl\` | From ZIP |
| Combined (Pucobre) | \`pucobre_combined_YYYY_QN.pdf\` | Pucobre only (original) |

## How It Works

### CMF Chile Download Flow:
1. Navigate to CMF Chile portal
2. Apply filters (month, year, tipo, tipo_norma)
3. Click "Consultar" to load results
4. Download each document by clicking its link
5. Extract XBRL ZIP to get .xbrl file

### Fallback to Pucobre.cl (when CMF fails):
1. Navigate to Pucobre Estados Financieros page
2. Find link matching "Estados Financieros DD-MM-YYYY"
3. Click to open PDF viewer in new tab
4. Download combined PDF
5. **Split the combined PDF**:
   - Detect "ANALISIS RAZONADO" title page (typically page 96)
   - Extract pages 1 to split_page-1 → \`estados_financieros_YYYY_QN.pdf\`
   - Extract pages split_page to end → \`analisis_razonado_YYYY_QN.pdf\`
   - Keep original as \`pucobre_combined_YYYY_QN.pdf\`

## When to Use Pucobre.cl Fallback

The downloader **automatically** uses Pucobre.cl when:
- CMF Chile doesn't have data for the requested period
- Q1 data is requested before Q2 is published on CMF

**What Pucobre.cl provides**:
- ✓ Estados Financieros PDF (split from combined)
- ✓ Análisis Razonado PDF (split from combined)
- ✗ XBRL data (not available)

## Next Steps

After completing this instruction:
1. Verify all files downloaded successfully
2. Proceed to \`02_parse_xbrl.md\` to extract aggregate totals from XBRL (if available)
3. Proceed to \`03_extract_detailed_costs.md\` to extract line-item breakdown from PDF (Nota 21-22)
4. See \`data_mapping.md\` for field mappings and number format rules

## Data Extraction Flow

\`\`\`
CMF Chile Portal                      Pucobre.cl (Fallback)
      │                                      │
      ├── analisis_razonado.pdf              │
      │         ▲                            │
      │         │                   ┌────────┴────────┐
      │         └───────────────────┤ combined.pdf    │
      │                             │ (auto-split)    │
      ├── estados_financieros.pdf ◄─┤                 │
      │         │                   │ Pages 1-95:     │
      │         │                   │  EEFF           │
      │         │                   │ Pages 96+:      │
      │         │                   │  Análisis Raz.  │
      │         │                   └─────────────────┘
      │         │
      │         └──> Nota 21 & 22 (DETAILED BREAKDOWN)
      │              Page 71: 11 Costo de Venta items
      │                       6 Gastos Admin items
      │
      └── estados_financieros_xbrl.zip
                  │
                  └── .xbrl (AGGREGATE TOTALS)
                      Revenue, CostOfSales, GrossProfit,
                      AdminExpense, ProfitLoss
\`\`\`

## Troubleshooting

### CMF Chile - Page doesn't load
- Check network connectivity
- Try increasing timeout: \`page.goto(url, timeout=90000)\`
- Run with \`headless=False\` to see what's happening

### CMF Chile - Document not found
- Period may not exist on CMF (especially Q1 before Q2 is published)
- Use \`fallback_to_pucobre=True\` to automatically try Pucobre.cl
- Check filters (Consolidado vs Individual)

### Pucobre.cl - Period not found
- Check \`list_pucobre_periods()\` for available periods
- Verify the date format matches: "Estados Financieros DD-MM-YYYY"
- Historical data goes back to 2011

### PDF splitting fails
- Ensure PyMuPDF is installed: \`pip install PyMuPDF\`
- The combined PDF will still be available as \`estados_financieros_YYYY_QN.pdf\`
- Check logs for "Could not find Análisis Razonado section"

### XBRL extraction fails
- Check if the ZIP file is valid
- Ensure sufficient disk space
- XBRL not available from Pucobre.cl fallback

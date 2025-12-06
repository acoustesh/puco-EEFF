# 04 - Extract Data for Sheet 1

## Objective

Extract the Estado de Situación Financiera (Balance General) data from XBRL and PDF sources, using OCR as fallback when needed.

## Prerequisites

- Source locations documented (from 03_locate_sheet1.md)
- XBRL and PDF files available

## Steps

### 1. Extract from XBRL (Primary Source)

```python
from pathlib import Path
from puco_eeff.extractor.xbrl_parser import parse_xbrl_file, extract_by_xpath
from puco_eeff.config import get_config, get_period_paths

year, quarter = 2024, 3  # Adjust as needed
paths = get_period_paths(year, quarter)
config = get_config()

xml_path = paths["raw_xbrl"] / f"EEFF_{year}_Q{quarter}.xml"

# Parse full XBRL
xbrl_data = parse_xbrl_file(xml_path)

# Extract specific values using XPath
sheet1_config = config["sheets"]["sheet1"]
extracted_data = {}

for xpath in sheet1_config.get("xml_paths", []):
    values = extract_by_xpath(xml_path, xpath)
    field_name = xpath.split(":")[-1] if ":" in xpath else xpath
    extracted_data[field_name] = values
    print(f"{field_name}: {values}")
```

### 2. Extract from PDF Tables (If XML incomplete)

```python
from puco_eeff.extractor.pdf_parser import extract_tables_from_pdf, find_section_in_pdf

pdf_path = paths["raw_pdf"] / f"EEFF_{year}_Q{quarter}.pdf"

# Find the balance sheet section
sections = find_section_in_pdf(pdf_path, "Estado de Situación Financiera")
if sections:
    print(f"Found on pages: {[s['page'] for s in sections]}")

    # Extract tables from those pages
    pages_to_extract = [s['page'] for s in sections]
    tables = extract_tables_from_pdf(pdf_path, pages=pages_to_extract)

    for page, page_tables in tables.items():
        print(f"\nPage {page}: {len(page_tables)} tables found")
        for i, table in enumerate(page_tables):
            print(f"  Table {i+1}: {len(table)} rows")
```

### 3. Use OCR for Image-Based Tables

```python
from puco_eeff.extractor.ocr_fallback import ocr_with_fallback

# If tables couldn't be extracted directly, use OCR
if not tables or all(len(t) == 0 for t in tables.values()):
    print("Using OCR for extraction...")

    # Create audit directory for this period
    audit_dir = paths["audit"]
    audit_dir.mkdir(parents=True, exist_ok=True)

    # OCR the PDF pages containing balance sheet
    ocr_result = ocr_with_fallback(
        pdf_path=pdf_path,
        prompt="""Extract the Estado de Situación Financiera (Balance General) table.
Return the data as a markdown table with columns:
- Concepto (Concept)
- Período Actual (Current Period)
- Período Anterior (Previous Period)

Include all line items for Activos, Pasivos, and Patrimonio.""",
        save_all_responses=True,
        audit_dir=audit_dir
    )

    if ocr_result["success"]:
        print("OCR successful!")
        print(ocr_result["content"][:500])
    else:
        print(f"OCR failed: {ocr_result.get('error')}")
```

### 4. Parse OCR Results

```python
from puco_eeff.transformer.normalizer import extract_tables_from_ocr_markdown

if ocr_result.get("success"):
    # Extract tables from OCR markdown output
    ocr_tables = extract_tables_from_ocr_markdown(ocr_result["content"])

    print(f"Extracted {len(ocr_tables)} tables from OCR")
    for i, df in enumerate(ocr_tables):
        print(f"\nTable {i+1}:")
        print(df.head())
```

### 5. Combine Data from All Sources

```python
import pandas as pd
from puco_eeff.transformer.source_tracker import SourceTracker

# Initialize source tracker
tracker = SourceTracker(period=f"{year}_Q{quarter}")

# Combine data, preferring XML over PDF over OCR
sheet1_data = {}

# Process XML data
for field, values in extracted_data.items():
    if values:
        sheet1_data[field] = values[0]  # Take first value
        tracker.add_source(
            field_name=field,
            source_type="xml",
            file_path=str(xml_path),
            location=f"//ifrs-full:{field}",
            extraction_method="xbrl_parser",
            confidence=1.0,
            raw_value=str(values[0])
        )

# Add PDF/OCR data for missing fields
# ... (implement based on actual gaps)

print(f"\nExtracted {len(sheet1_data)} fields for Sheet 1")
```

### 6. Structure as DataFrame

```python
# Create a structured DataFrame
balance_sheet = pd.DataFrame([
    {"concepto": "Activos Corrientes", "valor": sheet1_data.get("CurrentAssets")},
    {"concepto": "Activos No Corrientes", "valor": sheet1_data.get("NoncurrentAssets")},
    {"concepto": "Total Activos", "valor": sheet1_data.get("Assets")},
    {"concepto": "Pasivos Corrientes", "valor": sheet1_data.get("CurrentLiabilities")},
    {"concepto": "Pasivos No Corrientes", "valor": sheet1_data.get("NoncurrentLiabilities")},
    {"concepto": "Total Pasivos", "valor": sheet1_data.get("Liabilities")},
    {"concepto": "Patrimonio Total", "valor": sheet1_data.get("Equity")},
])

print(balance_sheet)
```

### 7. Save Intermediate Results

```python
from puco_eeff.writer.sheet_writer import save_sheet_data

# Save for partial re-run capability
output_path = save_sheet_data(
    sheet_name="sheet1",
    data=balance_sheet,
    period=f"{year}_Q{quarter}"
)

print(f"Sheet 1 data saved to: {output_path}")

# Save source tracking
tracker.save()
```

## Output

- Extracted data: `data/processed/YYYY_QN_sheet1.json`
- Source mapping: `audit/YYYY_QN/source_mapping.json`
- OCR responses (if used): `audit/YYYY_QN/ocr_*.json`

## Validation Checklist

- [ ] Total Activos = Activos Corrientes + Activos No Corrientes
- [ ] Total Pasivos = Pasivos Corrientes + Pasivos No Corrientes
- [ ] Total Activos = Total Pasivos + Patrimonio
- [ ] Values match between XML and PDF (if both available)

## Next Steps

After completing this instruction:
1. Verify data accuracy against source documents
2. Note any discrepancies in audit log
3. Proceed to `05_format_sheet1.md`

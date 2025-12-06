# 03 - Locate Data Sources for Sheet 1

## Objective

Identify and document where to find the data for Sheet 1 (Estado de Situación Financiera / Balance General) in both XML and PDF sources.

## Prerequisites

- PDF downloaded: `data/raw/pdf/EEFF_YYYY_QN.pdf`
- XBRL downloaded: `data/raw/xbrl/EEFF_YYYY_QN.xml`

## Context

Sheet 1 typically contains the Balance General / Estado de Situación Financiera, which includes:
- Activos (Assets)
- Pasivos (Liabilities)
- Patrimonio (Equity)

## Steps

### 1. Explore XBRL Structure

```python
from pathlib import Path
from puco_eeff.extractor.xbrl_parser import parse_xbrl_file
from puco_eeff.config import get_period_paths

year, quarter = 2024, 3  # Adjust as needed
paths = get_period_paths(year, quarter)
xml_path = paths["raw_xbrl"] / f"EEFF_{year}_Q{quarter}.xml"

# Parse XBRL
data = parse_xbrl_file(xml_path)

# List unique fact names
fact_names = set(f["name"] for f in data["facts"])
print(f"Found {len(fact_names)} unique facts")

# Look for balance sheet related facts
balance_keywords = ["Asset", "Liability", "Equity", "Activo", "Pasivo", "Patrimonio"]
balance_facts = [n for n in fact_names if any(k.lower() in n.lower() for k in balance_keywords)]
print(f"\nBalance sheet related facts ({len(balance_facts)}):")
for name in sorted(balance_facts)[:20]:
    print(f"  - {name}")
```

### 2. Explore PDF Structure

```python
from puco_eeff.extractor.pdf_parser import extract_text_from_pdf, get_pdf_info

pdf_path = paths["raw_pdf"] / f"EEFF_{year}_Q{quarter}.pdf"

# Get PDF info
info = get_pdf_info(pdf_path)
print(f"PDF has {info['num_pages']} pages")

# Extract text and search for Balance General
text_by_page = extract_text_from_pdf(pdf_path)

for page_num, text in text_by_page.items():
    if "estado de situación" in text.lower() or "balance general" in text.lower():
        print(f"\n--- Found on page {page_num} ---")
        # Show context
        idx = text.lower().find("estado de situación")
        if idx == -1:
            idx = text.lower().find("balance general")
        print(text[max(0, idx-50):idx+200])
```

### 3. Map XML Facts to Sheet Fields

Create a mapping of which XBRL facts correspond to which sheet fields:

```python
# Example mapping structure (to be filled based on exploration)
sheet1_xml_mapping = {
    "activos_corrientes": {
        "xpath": "//ifrs-full:CurrentAssets",
        "description": "Total current assets"
    },
    "activos_no_corrientes": {
        "xpath": "//ifrs-full:NoncurrentAssets", 
        "description": "Total non-current assets"
    },
    "total_activos": {
        "xpath": "//ifrs-full:Assets",
        "description": "Total assets"
    },
    # Add more mappings...
}
```

### 4. Map PDF Sections

Identify PDF sections that contain balance sheet data:

```python
# Example PDF section mapping
sheet1_pdf_mapping = {
    "activos_corrientes": {
        "section": "Estado de Situación Financiera",
        "page_hint": "first occurrence",
        "table_position": "first table after header"
    },
    # Fallback for OCR
    "ocr_sections": [
        {"page_range": [3, 5], "description": "Balance sheet pages"}
    ]
}
```

### 5. Document Source Priority

For Sheet 1, establish which source to use for each field:

```python
# Source priority mapping
sheet1_sources = {
    # Field: [primary_source, fallback_source]
    "total_activos": ["xml", "pdf_table"],
    "total_pasivos": ["xml", "pdf_table"],
    "patrimonio_total": ["xml", "pdf_table"],
    # Fields that might need OCR
    "notas_explicativas": ["pdf_ocr", None],
}
```

### 6. Update config.json

Add the discovered mappings to `config/config.json`:

```json
{
  "sheets": {
    "sheet1": {
      "name": "Estado de Situación Financiera",
      "description": "Balance General / Statement of Financial Position",
      "xml_paths": [
        "//ifrs-full:CurrentAssets",
        "//ifrs-full:NoncurrentAssets",
        "//ifrs-full:Assets",
        "//ifrs-full:CurrentLiabilities",
        "//ifrs-full:NoncurrentLiabilities",
        "//ifrs-full:Liabilities",
        "//ifrs-full:Equity"
      ],
      "pdf_sections": [
        "Estado de Situación Financiera",
        "Balance General"
      ]
    }
  }
}
```

### 7. Create Source Tracker Entry

```python
from puco_eeff.transformer.source_tracker import SourceTracker

tracker = SourceTracker(period=f"{year}_Q{quarter}")

# Document the mapping decisions
tracker.add_source(
    field_name="total_activos",
    source_type="xml",
    file_path=str(xml_path),
    location="//ifrs-full:Assets",
    extraction_method="xbrl_parser",
    confidence=1.0
)

# Save for audit
tracker.save()
```

## Output

- Updated `config/config.json` with Sheet 1 mappings
- Source mapping saved to `audit/YYYY_QN/source_mapping.json`

## Fields to Map for Sheet 1

| Field | Spanish Term | IFRS Concept |
|-------|--------------|--------------|
| Current Assets | Activos Corrientes | CurrentAssets |
| Non-current Assets | Activos No Corrientes | NoncurrentAssets |
| Total Assets | Total Activos | Assets |
| Current Liabilities | Pasivos Corrientes | CurrentLiabilities |
| Non-current Liabilities | Pasivos No Corrientes | NoncurrentLiabilities |
| Total Liabilities | Total Pasivos | Liabilities |
| Equity | Patrimonio | Equity |

## Next Steps

After completing this instruction:
1. Verify XPath expressions work with actual data
2. Note any fields that require PDF/OCR extraction
3. Proceed to `04_extract_sheet1.md`

# 02 - Parse XBRL and Explore Data Structure

## Objective

Parse the downloaded XBRL file and explore its structure to understand available data for extraction. This step identifies which financial data can be extracted directly from the structured XBRL/XML.

## Prerequisites

- Documents downloaded via `01_download_all.md`
- Files present:
  - `data/raw/pdf/estados_financieros_YYYY_QN_xbrl.zip`
  - `data/raw/pdf/estados_financieros_YYYY_QN.xml` (extracted)

## Context

XBRL (eXtensible Business Reporting Language) provides structured financial data that can be directly parsed without OCR. This is the **preferred data source** when available.

The XBRL file contains:
- **Contexts**: Define the reporting period and entity
- **Facts**: Individual financial data points with values
- **Units**: Currencies and measurement units

## Steps

### 1. Load the Downloaded XBRL

```python
from pathlib import Path
from puco_eeff.config import get_period_paths, DATA_DIR

year, quarter = 2024, 3  # Adjust as needed
paths = get_period_paths(year, quarter)

# XBRL is extracted to the raw_pdf directory (same as other downloads)
xml_path = paths["raw_pdf"] / f"estados_financieros_{year}_Q{quarter}.xml"

if xml_path.exists():
    print(f"✓ XBRL file found: {xml_path}")
    print(f"  Size: {xml_path.stat().st_size:,} bytes")
else:
    print("✗ XBRL file not found. Run 01_download_all.md first.")
```

### 2. Parse XBRL Structure

```python
from puco_eeff.extractor.xbrl_parser import parse_xbrl_file

# Parse the XBRL file
data = parse_xbrl_file(xml_path)

print(f"\n=== XBRL Summary ===")
print(f"Contexts: {len(data['contexts'])}")
print(f"Facts: {len(data['facts'])}")
print(f"Units: {len(data['units'])}")

# Show sample contexts
print("\n=== Sample Contexts ===")
for ctx_id, ctx_info in list(data['contexts'].items())[:5]:
    print(f"  {ctx_id}: {ctx_info}")
```

### 3. Explore Available Facts

```python
# Get unique fact names
fact_names = set(f["name"] for f in data["facts"])
print(f"\nFound {len(fact_names)} unique fact types")

# Group facts by namespace/category
fact_categories = {}
for name in fact_names:
    # Extract category from fact name
    parts = name.split("_") if "_" in name else [name]
    category = parts[0] if len(parts) > 1 else "Other"
    fact_categories.setdefault(category, []).append(name)

print("\n=== Fact Categories ===")
for cat, names in sorted(fact_categories.items()):
    print(f"  {cat}: {len(names)} facts")
```

### 4. Find Balance Sheet Data (Estado de Situación Financiera)

```python
# Keywords for Balance Sheet items
balance_keywords = [
    "Asset", "Liability", "Equity", 
    "Activo", "Pasivo", "Patrimonio",
    "CurrentAsset", "NoncurrentAsset",
    "CurrentLiabilit", "NoncurrentLiabilit"
]

balance_facts = [
    n for n in fact_names 
    if any(k.lower() in n.lower() for k in balance_keywords)
]

print(f"\n=== Balance Sheet Facts ({len(balance_facts)}) ===")
for name in sorted(balance_facts)[:30]:  # Show first 30
    # Get sample value
    sample = next((f for f in data["facts"] if f["name"] == name), None)
    value = sample["value"] if sample else "N/A"
    print(f"  {name}: {value}")
```

### 5. Find Income Statement Data (Estado de Resultados)

```python
# Keywords for Income Statement items
income_keywords = [
    "Revenue", "Profit", "Loss", "Expense", "Income",
    "Ingreso", "Gasto", "Utilidad", "Perdida",
    "OperatingIncome", "NetIncome", "GrossProfit"
]

income_facts = [
    n for n in fact_names 
    if any(k.lower() in n.lower() for k in income_keywords)
]

print(f"\n=== Income Statement Facts ({len(income_facts)}) ===")
for name in sorted(income_facts)[:30]:
    sample = next((f for f in data["facts"] if f["name"] == name), None)
    value = sample["value"] if sample else "N/A"
    print(f"  {name}: {value}")
```

### 6. Find Cash Flow Data (Estado de Flujos de Efectivo)

```python
# Keywords for Cash Flow items
cashflow_keywords = [
    "CashFlow", "Cash", "Operating", "Investing", "Financing",
    "Efectivo", "Operacion", "Inversion", "Financiamiento"
]

cashflow_facts = [
    n for n in fact_names 
    if any(k.lower() in n.lower() for k in cashflow_keywords)
]

print(f"\n=== Cash Flow Facts ({len(cashflow_facts)}) ===")
for name in sorted(cashflow_facts)[:30]:
    sample = next((f for f in data["facts"] if f["name"] == name), None)
    value = sample["value"] if sample else "N/A"
    print(f"  {name}: {value}")
```

### 7. Extract Specific Values

```python
from puco_eeff.extractor.xbrl_parser import get_facts_by_name

# Example: Get Total Assets
assets = get_facts_by_name(data, "Assets")
print("\n=== Total Assets ===")
for fact in assets:
    print(f"  Period: {fact.get('context', 'N/A')}")
    print(f"  Value: {fact['value']}")
    print(f"  Unit: {fact.get('unit', 'N/A')}")
    print()

# Example: Get Equity
equity = get_facts_by_name(data, "Equity")
print("=== Equity ===")
for fact in equity:
    print(f"  Period: {fact.get('context', 'N/A')}")
    print(f"  Value: {fact['value']}")
```

### 8. Document Available Facts for Sheets

Based on exploration, document which facts map to which sheets:

```python
# Create mapping document
sheet_mappings = {
    "sheet1_situacion_financiera": {
        "description": "Estado de Situación Financiera (Balance)",
        "facts": [
            "Assets",
            "CurrentAssets", 
            "NoncurrentAssets",
            "Liabilities",
            "CurrentLiabilities",
            "NoncurrentLiabilities",
            "Equity",
            # Add discovered facts...
        ]
    },
    "sheet2_resultados": {
        "description": "Estado de Resultados",
        "facts": [
            "Revenue",
            "CostOfSales",
            "GrossProfit",
            "OperatingIncome",
            "ProfitLoss",
            # Add discovered facts...
        ]
    },
    "sheet3_flujos_efectivo": {
        "description": "Estado de Flujos de Efectivo",
        "facts": [
            "CashFlowsFromOperatingActivities",
            "CashFlowsFromInvestingActivities",
            "CashFlowsFromFinancingActivities",
            # Add discovered facts...
        ]
    }
}

# Save mapping for reference
import json
mapping_path = paths["audit"]
mapping_path.mkdir(parents=True, exist_ok=True)
with open(mapping_path / "xbrl_fact_mapping.json", "w") as f:
    json.dump(sheet_mappings, f, indent=2, ensure_ascii=False)

print(f"\nSaved fact mapping to: {mapping_path / 'xbrl_fact_mapping.json'}")
```

### 9. Identify Gaps (What Needs PDF/OCR)

```python
# Check if key financial data is available in XBRL
required_facts = [
    ("Total Assets", "Assets"),
    ("Total Liabilities", "Liabilities"),
    ("Equity", "Equity"),
    ("Revenue", "Revenue"),
    ("Net Profit", "ProfitLoss"),
]

print("\n=== Data Availability Check ===")
gaps = []
for label, fact_name in required_facts:
    found = any(fact_name.lower() in n.lower() for n in fact_names)
    status = "✓ XBRL" if found else "✗ Need PDF/OCR"
    print(f"  {label}: {status}")
    if not found:
        gaps.append(label)

if gaps:
    print(f"\n⚠️  Fields requiring PDF/OCR extraction: {gaps}")
else:
    print("\n✓ All key fields available in XBRL")
```

## Output

- Understanding of XBRL structure and available facts
- Fact mapping saved to `audit/YYYY_QN/xbrl_fact_mapping.json`
- List of fields that need PDF/OCR extraction

## XBRL Namespaces Reference

Common IFRS XBRL namespaces you'll encounter:

| Prefix | Description |
|--------|-------------|
| `ifrs-full` | IFRS full taxonomy |
| `cl-ci` | Chile-specific extensions |
| `iso4217` | Currency codes |

## Next Steps

After completing this instruction:
1. Review the discovered XBRL facts
2. Map facts to desired Excel sheet structure
3. For any gaps, proceed to PDF extraction
4. Continue to `03_extract_sheet1.md`

## Troubleshooting

### XML parsing errors
- Check if the file is valid XML: `xmllint --noout file.xml`
- Ensure proper encoding (UTF-8)

### Facts not found
- Try different namespace prefixes
- Search for partial matches
- Check if data is in a different period context

### Empty values
- Some facts may only have values for specific contexts
- Filter by context period to get the right value

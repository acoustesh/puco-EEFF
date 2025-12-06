# 03 - Extract and Format Sheet1 (Detailed Costs)

> **📌 This is the main extraction instruction.** It covers extracting Nota 21/22 from PDF, validating against XBRL, formatting, and saving. The previous `04_format_sheet1.md` has been merged into this document.

## Unified Workflow (Recommended)

For most use cases, use the unified orchestrator that handles everything:

```python
from puco_eeff.main_sheet1 import process_sheet1

# Complete workflow: download (if needed) → extract → save → report
data = process_sheet1(year=2024, quarter=2)

# Options:
data = process_sheet1(year=2024, quarter=2, skip_download=True)   # Use existing files
data = process_sheet1(year=2024, quarter=2, save=False)           # Don't save JSON
data = process_sheet1(year=2024, quarter=2, verbose=False)        # No report output
```

**CLI equivalent:**
```bash
python -m puco_eeff.main_sheet1 --year 2024 --quarter 2
python -m puco_eeff.main_sheet1 --year 2024                    # All quarters
python -m puco_eeff.main_sheet1 --year 2024 -q 2 3             # Multiple quarters
python -m puco_eeff.main_sheet1 --year 2024 --skip-download    # Use existing files
python -m puco_eeff.main_sheet1 --year 2024 --no-save --quiet  # Extract only
```

---

## What `process_sheet1()` Does

The orchestrator in `puco_eeff/main_sheet1.py` performs 4 steps:

| Step | Function | Description |
|------|----------|-------------|
| 1. Download | `ensure_files_downloaded()` → `download_all_documents()` | Downloads PDF/XBRL if missing |
| 2. Extract | `extract_sheet1()` | Extracts from PDF + validates with XBRL |
| 3. Save | `save_sheet1_data()` | Saves to `data/processed/sheet1_{quarter}.json` |
| 4. Report | `print_sheet1_report()` | Prints formatted 27-row report |

---

## Objective

Extract **Nota 21 (Costo de Venta)** and **Nota 22 (Gastos de Administración)** from Estados Financieros PDF into Sheet1 with a 27-row structure:

- **Row 1**: Ingresos de actividades ordinarias (from XBRL or PDF fallback)
- **Rows 3-15**: Costo de Venta section (header + 11 line items + total)
- **Rows 19-27**: Gasto Admin y Ventas section (header + 6 line items + **Totales**)

**Source Documents**:
- **Estados Financieros PDF**: Contains Nota 21 and Nota 22 with detailed cost breakdowns
- **XBRL file**: Contains Revenue (Ingresos) and totals for validation

## Quick Start (Python API)

```python
from puco_eeff.extractor.cost_extractor import extract_sheet1
from puco_eeff.sheets.sheet1 import save_sheet1_data, print_sheet1_report

# Extract data (automatically uses PDF + XBRL validation)
data = extract_sheet1(2024, 2)

# Access values
print(f"Quarter: {data.quarter}")                    # "IIQ2024"
print(f"Ingresos: {data.ingresos_ordinarios:,}")     # 179,165
print(f"Total Costo de Venta: {data.total_costo_venta:,}")  # -126,202
print(f"Totales (Gasto Admin): {data.total_gasto_admin:,}") # -11,632

# Dynamic field access
value = data.get_value("cv_gastos_personal")  # -19,721
data.set_value("cv_energia", -9589)

# Save to JSON
output_path = save_sheet1_data(data)
print(f"Saved to: {output_path}")

# Print formatted report
print_sheet1_report(data)
```

### Validation Output

When running extraction, you'll see:

```
✓ Total Costo de Venta matches XBRL: -126,202
✓ Total Gasto Admin matches XBRL: -11,632
Using XBRL value for Ingresos Ordinarios: 179,165
```

If there's a mismatch:
```
✗ Total Costo de Venta mismatch - PDF: -126,202, XBRL: -126,300 (diff: 98)
```

---

## Architecture Overview

The extraction is **fully config-driven** with no hardcoded values:

```
┌────────────────────────────────────────────────────────────────────┐
│                    CONFIGURATION FILES                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  config/config.json              - File patterns, period types      │
│  config/xbrl_specs.json          - XBRL namespaces, scaling factor  │
│                                                                     │
│  config/sheet1/                  - Sheet1-specific config:          │
│    ├── fields.json               - Field definitions, row mapping   │
│    ├── extraction.json           - PDF sections (nota_21, nota_22)  │
│    ├── xbrl_mappings.json        - XBRL fact mappings, validation   │
│    └── reference_data.json       - Known-good values per period     │
│                                                                     │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│  puco_eeff/sheets/sheet1.py      - Sheet1Data class, config loaders │
│  puco_eeff/extractor/cost_extractor.py - PDF/XBRL extraction logic  │
│  puco_eeff/main_sheet1.py        - Orchestrator (process_sheet1)    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Data Structure (27 Rows)

| Row | Field | Label | Section |
|-----|-------|-------|---------|
| 1 | ingresos_ordinarios | Ingresos de actividades ordinarias M USD | - |
| 2 | - | (blank) | - |
| 3 | - | Costo de Venta | header |
| 4 | cv_gastos_personal | Gastos en personal | costo_venta |
| 5 | cv_materiales | Materiales y repuestos | costo_venta |
| 6 | cv_energia | Energía eléctrica | costo_venta |
| 7 | cv_servicios_terceros | Servicios de terceros | costo_venta |
| 8 | cv_depreciacion_amort | Depreciación y amort del periodo | costo_venta |
| 9 | cv_deprec_leasing | Depreciación Activos en leasing -Nota 20 | costo_venta |
| 10 | cv_deprec_arrend | Depreciación Arrendamientos -Nota 20 | costo_venta |
| 11 | cv_serv_mineros | Servicios mineros de terceros | costo_venta |
| 12 | cv_fletes | Fletes y otros gastos operacionales | costo_venta |
| 13 | cv_gastos_diferidos | Gastos Diferidos, ajustes existencias y otros | costo_venta |
| 14 | cv_convenios | Obligaciones por convenios colectivos | costo_venta |
| 15 | total_costo_venta | **Total Costo de Venta** | costo_venta_total |
| 16-18 | - | (blank) | - |
| 19 | - | Gasto Adm, y Ventas | header |
| 20 | ga_gastos_personal | Gastos en personal | gasto_admin |
| 21 | ga_materiales | Materiales y repuestos | gasto_admin |
| 22 | ga_servicios_terceros | Servicios de terceros | gasto_admin |
| 23 | ga_gratificacion | Provision gratificacion legal y otros | gasto_admin |
| 24 | ga_comercializacion | Gastos comercializacion | gasto_admin |
| 25 | ga_otros | Otros gastos | gasto_admin |
| 26 | - | (blank) | - |
| 27 | total_gasto_admin | **Totales** | gasto_admin_total |

**IMPORTANT DISAMBIGUATION**:
- "**Total Costo de Venta**" (row 15) is the sum of Costo de Venta items
- "**Totales**" (row 27) is the sum of Gasto Admin items - this is the ONLY "Totales" that goes in row 27

## Sample Data (IIQ2024)

```
Row  Label                                          IIQ2024
───────────────────────────────────────────────────────────
1    Ingresos de actividades ordinarias M USD       179,165
2
3    Costo de Venta
4    Gastos en personal                             -19,721
5    Materiales y repuestos                         -23,219
6    Energía eléctrica                               -9,589
7    Servicios de terceros                          -25,063
8    Depreciación y amort del periodo               -21,694
9    Depreciación Activos en leasing -Nota 20          -881
10   Depreciación Arrendamientos -Nota 20            -1,577
11   Servicios mineros de terceros                  -10,804
12   Fletes y otros gastos operacionales             -5,405
13   Gastos Diferidos, ajustes existencias y otros   -1,587
14   Obligaciones por convenios colectivos           -6,662
15   Total Costo de Venta                          -126,202
16
17
18
19   Gasto Adm, y Ventas
20   Gastos en personal                              -3,818
21   Materiales y repuestos                            -129
22   Servicios de terceros                           -4,239
23   Provision gratificacion legal y otros             -639
24   Gastos comercializacion                         -2,156
25   Otros gastos                                      -651
26
27   Totales                                        -11,632
```

---

## Configuration Files

### config/sheet1/fields.json

Field definitions and row mapping (27 rows):

```json
{
  "name": "Ingresos y Costos",
  "value_fields": {
    "ingresos_ordinarios": {"type": "int", "section": "ingresos", "row": 1},
    "cv_gastos_personal": {"type": "int", "section": "nota_21", "row": 4},
    "cv_materiales": {"type": "int", "section": "nota_21", "row": 5},
    ...
    "total_costo_venta": {"type": "int", "section": "nota_21", "row": 15, "is_total": true},
    "ga_gastos_personal": {"type": "int", "section": "nota_22", "row": 20},
    ...
    "total_gasto_admin": {"type": "int", "section": "nota_22", "row": 27, "is_total": true}
  },
  "row_mapping": {
    "1": {"field": "ingresos_ordinarios", "label": "Ingresos de actividades ordinarias M USD"},
    "4": {"field": "cv_gastos_personal", "label": "Gastos en personal", "section": "costo_venta"},
    ...
  }
}
```

### config/sheet1/extraction.json

PDF extraction rules with keyword-based field matching:

```json
{
  "sections": {
    "nota_21": {
      "title": "Costo de Venta",
      "search_patterns": ["21. costo", "21 costo", "nota 21"],
      "table_identifiers": {
        "unique_items": ["energía eléctrica", "servicios mineros", "fletes"],
        "exclude_items": ["gratificación", "comercialización"]
      },
      "field_mappings": {
        "cv_gastos_personal": {
          "pdf_labels": ["Gastos en personal"],
          "match_keywords": ["gastos en personal"]
        },
        "cv_deprec_leasing": {
          "pdf_labels": ["Depreciación Activos en leasing"],
          "match_keywords": ["leasing"]
        },
        "cv_depreciacion_amort": {
          "pdf_labels": ["Depreciación y amort del periodo"],
          "match_keywords": ["amort"],
          "exclude_keywords": ["leasing", "arrendamiento"]
        }
      }
    },
    "nota_22": {
      "title": "Gastos de Administración y Ventas",
      "search_patterns": ["22. gastos", "22 gastos", "nota 22"],
      "table_identifiers": {
        "unique_items": ["gratificación", "comercialización"],
        "exclude_items": ["energía", "servicios mineros"]
      },
      "field_mappings": {
        "ga_gastos_personal": {"match_keywords": ["gastos en personal"]},
        "ga_gratificacion": {"match_keywords": ["gratificación", "gratificacion"]},
        "ga_comercializacion": {"match_keywords": ["comercialización", "comercializacion"]}
      }
    }
  }
}
```

**Keyword matching rules:**
- `match_keywords`: At least one must match (case-insensitive)
- `exclude_keywords`: If any match, skip this field
- Fields processed by section to avoid cross-section confusion

### config/sheet1/xbrl_mappings.json

XBRL fact mappings and validation rules:

```json
{
  "fact_mappings": {
    "ingresos_ordinarios": {
      "primary": "RevenueFromContractsWithCustomers",
      "fallbacks": ["Revenue", "IngresosPorActividadesOrdinarias"],
      "context_type": "duration",
      "apply_scaling": true
    },
    "total_costo_venta": {
      "primary": "CostOfSales",
      "fallbacks": ["CostoDeVentas"],
      "apply_scaling": true
    },
    "total_gasto_admin": {
      "primary": "AdministrativeExpense",
      "fallbacks": ["GastosDeAdministracion"],
      "apply_scaling": true
    }
  },
  "validation_rules": {
    "sum_tolerance": 1,
    "total_validations": [
      {"total_field": "total_costo_venta", "sum_fields": ["cv_gastos_personal", "cv_materiales", ...]},
      {"total_field": "total_gasto_admin", "sum_fields": ["ga_gastos_personal", "ga_materiales", ...]}
    ]
  }
}
```

### config/sheet1/reference_data.json

Known-good values for validation:

```json
{
  "2024_Q2": {
    "verified": true,
    "verified_date": "2024-12-06",
    "values": {
      "ingresos_ordinarios": 179165,
      "cv_gastos_personal": -19721,
      "cv_materiales": -23219,
      ...
      "total_costo_venta": -126202,
      "total_gasto_admin": -11632
    }
  },
  "2024_Q1": {
    "verified": true,
    "source": "Estados Financieros PDF - pucobre.cl (no XBRL)",
    "values": {...}
  }
}
```

---

## Extraction from Different Sources

### CMF Chile (Q2-Q4)
- **Estados Financieros PDF**: Contains Nota 21 and Nota 22 with detailed breakdowns (~page 70-75)
- **XBRL file**: Contains Revenue and totals for validation

### Pucobre.cl Fallback (Q1)
- **Document**: `estados_financieros_YYYY_Q1.pdf` (split from combined PDF)
- **XBRL**: NOT available - PDF-only extraction
- Ingresos extracted from Estado de Resultados page in PDF

---

## Output Files

| File | Description |
|------|-------------|
| `data/processed/sheet1_IQ2024.json` | Q1 2024 data |
| `data/processed/sheet1_IIQ2024.json` | Q2 2024 data |
| `data/processed/sheet1_IIIQ2024.json` | Q3 2024 data |
| `data/processed/sheet1_IVQ2024.json` | Q4 2024 data |

### Example JSON Output

```json
{
  "quarter": "IIQ2024",
  "year": 2024,
  "quarter_num": 2,
  "period_type": "quarterly",
  "source": "cmf",
  "xbrl_available": true,
  "ingresos_ordinarios": 179165,
  "cv_gastos_personal": -19721,
  "cv_materiales": -23219,
  "cv_energia": -9589,
  "cv_servicios_terceros": -25063,
  "cv_depreciacion_amort": -21694,
  "cv_deprec_leasing": -881,
  "cv_deprec_arrend": -1577,
  "cv_serv_mineros": -10804,
  "cv_fletes": -5405,
  "cv_gastos_diferidos": -1587,
  "cv_convenios": -6662,
  "total_costo_venta": -126202,
  "ga_gastos_personal": -3818,
  "ga_materiales": -129,
  "ga_servicios_terceros": -4239,
  "ga_gratificacion": -639,
  "ga_comercializacion": -2156,
  "ga_otros": -651,
  "total_gasto_admin": -11632
}
```

---

## Validation Checklist

### Automatic Validation (in `extract_sheet1()`)
- ✓ Sum of cv_* items (rows 4-14) ≈ total_costo_venta (row 15)
- ✓ Sum of ga_* items (rows 20-25) ≈ total_gasto_admin (row 27)
- ✓ PDF totals ≈ XBRL totals (when XBRL available)
- ✓ Reference validation against known-good values

### Manual Validation
```python
from puco_eeff.sheets.sheet1 import validate_sheet1_against_reference

issues = validate_sheet1_against_reference(data)
if issues:
    print("Validation issues:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("✓ All values match reference")
```

---

## Batch Extraction Across Quarters

```python
from puco_eeff.main_sheet1 import process_sheet1

# Extract all quarters for a year
for quarter in [1, 2, 3, 4]:
    data = process_sheet1(year=2024, quarter=quarter, skip_download=True)
    if data:
        print(f"✓ {data.quarter}: Ingresos={data.ingresos_ordinarios:,}")
```

**CLI:**
```bash
python -m puco_eeff.main_sheet1 --year 2024  # Processes all 4 quarters
```

---

## Disambiguation Rules

### Items that appear in BOTH sections:

| Item | Costo de Venta (rows 4-14) | Gasto Admin (rows 20-25) |
|------|---------------------------|-------------------------|
| Gastos en personal | cv_gastos_personal (~-20K) | ga_gastos_personal (~-4K) |
| Materiales y repuestos | cv_materiales (~-23K) | ga_materiales (~-100) |
| Servicios de terceros | cv_servicios_terceros (~-25K) | ga_servicios_terceros (~-4K) |

**Rule**: Fields are matched by section context - Costo de Venta values are typically 5-10x larger.

### "Totales" vs "Total Costo de Venta":

- **Row 15**: "Total Costo de Venta" - explicit label, sum of rows 4-14
- **Row 27**: "Totales" - implicit Gasto Admin total, sum of rows 20-25

---

## Number Format Notes

Chilean financial documents use:
- **Period as thousands separator**: `30.294` = 30,294
- **Parentheses for negatives**: `(30.294)` = -30,294
- **Currency**: MUS$ = Miles de dólares estadounidenses (thousands of USD)

---

## Troubleshooting

### Issue: Field not being matched

Check the `match_keywords` in `config/sheet1/extraction.json`. Debug by inspecting raw PDF items:

```python
from puco_eeff.extractor.cost_extractor import extract_pdf_section
from puco_eeff.config import get_period_paths

paths = get_period_paths(2024, 2)
pdf_path = paths["raw_pdf"] / "estados_financieros_2024_Q2.pdf"

section = extract_pdf_section(pdf_path, "nota_21")
for item in section.items:
    print(f'"{item.concepto}" = {item.ytd_actual}')
```

Then update keywords in `config/sheet1/extraction.json` to match the actual PDF text.

### Issue: XBRL values seem wrong (off by 1000x)

Check `apply_scaling` in `config/sheet1/xbrl_mappings.json`. All XBRL values from financial statements are in full USD and should have `apply_scaling: true` to convert to MUS$ (thousands).

### Issue: Depreciation fields swapped

The three depreciation fields in Nota 21 have similar labels. Use **unique** keywords in `config/sheet1/extraction.json`:

```json
"cv_depreciacion_amort": {
  "match_keywords": ["amort"],
  "exclude_keywords": ["leasing", "arrendamiento"]
},
"cv_deprec_leasing": {
  "match_keywords": ["leasing"]
},
"cv_deprec_arrend": {
  "match_keywords": ["arrendamiento"]
}
```

### Issue: File not found

File patterns are in `config/config.json`. Check:

```python
from puco_eeff.config import format_filename, find_file_with_alternatives, get_period_paths

paths = get_period_paths(2024, 2)
pdf_path = find_file_with_alternatives(paths["raw_pdf"], "estados_financieros_pdf", 2024, 2)
print(f"Found: {pdf_path}")
```

---

## Adding Support for New Periods

### Step 1: Run extraction
```python
from puco_eeff.main_sheet1 import process_sheet1
data = process_sheet1(year=2024, quarter=4)
```

### Step 2: If successful, add reference values

In `config/sheet1/reference_data.json`:
```json
{
  "2024_Q4": {
    "verified": true,
    "verified_date": "2024-12-06",
    "values": {
      "ingresos_ordinarios": 295000,
      "total_costo_venta": -210000,
      ...
    }
  }
}
```

### Step 3: If field matching fails

Update `match_keywords` in `config/sheet1/extraction.json` to add variations.

### Step 4: Run tests to validate

```bash
cd puco-EEFF
poetry run pytest tests/test_config_integrity.py -v
poetry run pytest tests/test_cost_extractor.py -v
```

---

## Learning from Failures (Feedback Loop)

When extraction fails but is successfully recovered:

1. **Document what worked** in `config/sheet1/extraction.json` under `period_overrides`
2. **Add reference values** to `config/sheet1/reference_data.json`
3. **Consider updating defaults** if pattern works for 2+ periods
4. **Run tests** to validate config integrity

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXTRACTION FEEDBACK LOOP                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Attempt extraction with current specs                        │
│           │                                                      │
│           ▼                                                      │
│  2. FAILURE? ──No──► Success ──► Add to reference_data.json     │
│           │                              │                       │
│          Yes                             ▼                       │
│           │                      Mark period verified            │
│           ▼                                                      │
│  3. Try recovery strategies:                                     │
│     - Adjust search patterns                                     │
│     - Try OCR fallback                                           │
│     - Manual page review                                         │
│           │                                                      │
│           ▼                                                      │
│  4. SUCCESS? ──No──► Document issue, request help               │
│           │                                                      │
│          Yes                                                     │
│           │                                                      │
│           ▼                                                      │
│  5. UPDATE config/sheet1/extraction.json:                        │
│     - Add period-specific deviations                             │
│     - Note what worked in recovery_notes                         │
│           │                                                      │
│           ▼                                                      │
│  6. UPDATE config/sheet1/reference_data.json:                    │
│     - Add verified values                                        │
│           │                                                      │
│           ▼                                                      │
│  7. RUN TESTS to validate config integrity                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Running Tests

```bash
cd puco-EEFF
poetry run pytest tests/ -v                        # All tests
poetry run pytest tests/test_cost_extractor.py -v  # Extractor tests
poetry run pytest tests/test_config_integrity.py -v # Config tests
```

---

## Next Steps

1. See `data_mapping.md` for detailed XBRL and PDF field mappings
2. Proceed to `05_locate_sheet2.md` for the next sheet
3. Or proceed to `NN_combine_workbook.md` if all sheets ready

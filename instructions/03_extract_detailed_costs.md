# 03 - Extract Detailed Cost Breakdown (Sheet1)

## Objective

Extract **Nota 21 (Costo de Venta)** and **Nota 22 (Gastos de Administración)** from Estados Financieros PDF into Sheet1 with a 27-row structure:

- **Row 1**: Ingresos de actividades ordinarias (from XBRL)
- **Rows 3-15**: Costo de Venta section (header + 11 line items + total)
- **Rows 19-27**: Gasto Admin y Ventas section (header + 6 line items + **Totales**)

## Architecture Overview

The extraction is **fully config-driven** with no hardcoded values:

```
┌────────────────────────────────────────────────────────────────────┐
│                    CONFIGURATION FILES                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  config/config.json           - File patterns, period types         │
│  config/extraction_specs.json - PDF extraction rules, field maps    │
│  config/xbrl_specs.json       - XBRL fact names, scaling, validation│
│  config/reference_data.json   - Known-good values for validation    │
│                                                                     │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│           puco_eeff/extractor/cost_extractor.py                     │
│                                                                     │
│  • Loads all config from JSON files at runtime                      │
│  • No hardcoded field names, XBRL facts, or file patterns          │
│  • match_concepto_to_field() uses config keywords                   │
│  • XBRL extraction uses fact_mappings with apply_scaling            │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

**Source Documents**:
- **Estados Financieros PDF**: Contains Nota 21 and Nota 22 with detailed cost breakdowns
- **XBRL file**: Contains Revenue (Ingresos) and totals for validation

**IMPORTANT DISAMBIGUATION**:
- "**Total Costo de Venta**" (row 15) is the sum of Costo de Venta items
- "**Totales**" (row 27) is the sum of Gasto Admin items - this is the ONLY "Totales" that goes in row 27

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

## Quick Start (Using the Module)

```python
from puco_eeff.extractor.cost_extractor import extract_sheet1

# Extract data (automatically uses PDF + XBRL validation)
data = extract_sheet1(2024, 2)

# Access values
print(f"Quarter: {data.quarter}")           # "IIQ2024"
print(f"Ingresos: {data.ingresos_ordinarios:,}")  # 179,165
print(f"Total Costo de Venta: {data.total_costo_venta:,}")  # -126,202
print(f"Totales (Gasto Admin): {data.total_gasto_admin:,}")  # -11,632

# Dynamic field access (config-driven)
value = data.get_value("cv_gastos_personal")  # -19,721
data.set_value("cv_energia", -9589)

# Format quarter label
from puco_eeff.config import format_period_key, format_period_display
key = format_period_key(2024, 2, "quarterly")       # "2024_Q2"
label = format_period_display(2024, 2, "quarterly") # "IIQ2024"
```

### Validation Output

When running extraction, you'll see:

```
✓ Total Costo de Venta matches XBRL: -126,202
✓ Total Gasto Admin matches XBRL: -11,632
Using XBRL value for Ingresos Ordinarios: 179,165
✓ All 20 values match reference
```

If there's a mismatch:
```
✗ Total Costo de Venta mismatch - PDF: -126,202, XBRL: -126,300 (diff: 98)
```

## Disambiguation Rules

### Items that appear in BOTH sections:

| Item | Costo de Venta (rows 4-14) | Gasto Admin (rows 20-25) |
|------|---------------------------|-------------------------|
| Gastos en personal | cv_gastos_personal (~-20K) | ga_gastos_personal (~-4K) |
| Materiales y repuestos | cv_materiales (~-23K) | ga_materiales (~-100) |
| Servicios de terceros | cv_servicios_terceros (~-25K) | ga_servicios_terceros (~-4K) |

**Rule**: Use section context to distinguish - Costo de Venta values are typically 5-10x larger.

### "Totales" vs "Total Costo de Venta":

- **Row 15**: "Total Costo de Venta" - explicit label, sum of rows 4-14
- **Row 27**: "Totales" - implicit Gasto Admin total, sum of rows 20-25

**Rule**: "Totales" without "Costo" or "Venta" qualifier = Gasto Admin total (row 27)

## Extraction from Different Sources

### CMF Chile (Q2-Q4)
- **Estados Financieros PDF**: Contains Nota 21 and Nota 22 with detailed breakdowns (~page 70-75)
- **XBRL file**: Contains Revenue and totals for validation

### Pucobre.cl Fallback (Q1)
- **Document**: `estados_financieros_YYYY_Q1.pdf` (split from combined PDF)
- **XBRL**: NOT available - PDF-only extraction

## Batch Extraction Across Quarters

```python
from puco_eeff.extractor.cost_extractor import extract_sheet1

# Extract all available quarters
quarters_data = {}
for quarter in [1, 2, 3, 4]:
    for year in [2024]:
        data = extract_sheet1(year, quarter)
        if data:
            quarters_data[data.quarter] = data
            save_sheet1_data(data)
            print(f"✓ {data.quarter}: Ingresos={data.ingresos_ordinarios:,}")

# Output all quarters in Excel format
# (One column per quarter, 27 rows)
```

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

## Number Format Notes

Chilean financial documents use:
- **Period as thousands separator**: `30.294` = 30,294
- **Parentheses for negatives**: `(30.294)` = -30,294
- **Currency**: MUS$ = Miles de dólares estadounidenses (thousands of USD)

## Validation Checklist

### Internal Validation
- [ ] Sum of cv_* items (rows 4-14) = total_costo_venta (row 15)
- [ ] Sum of ga_* items (rows 20-25) = total_gasto_admin (row 27)
- [ ] Row 27 "Totales" is NOT "Total Costo de Venta"

### Cross-Source Validation (when XBRL available)
- [ ] total_costo_venta ≈ XBRL CostOfSales
- [ ] total_gasto_admin ≈ XBRL AdministrativeExpense
- [ ] ingresos_ordinarios ≈ XBRL RevenueFromContractsWithCustomers

## Configuration Files (4-File Architecture)

All configuration is externalized into JSON files in `config/`:

### config/xbrl_specs.json (NEW - XBRL-specific)

XBRL fact names, scaling, and validation rules:

```json
{
  "scaling_factor": 1000,
  "fact_mappings": {
    "ingresos_ordinarios": {
      "primary": "RevenueFromContractsWithCustomers",
      "fallbacks": ["Revenue", "IngresosPorActividadesOrdinarias"],
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
    },
    "gross_profit": {
      "primary": "GrossProfit",
      "fallbacks": ["UtilidadBruta"],
      "apply_scaling": true
    },
    "profit_loss": {
      "primary": "ProfitLoss",
      "fallbacks": ["GananciaPerdida"],
      "apply_scaling": true
    }
  },
  "validation_rules": {
    "sum_tolerance": 1
  }
}
```

**Key features:**
- `apply_scaling: true` = divide by `scaling_factor` (1000) to convert USD → MUS$
- `fallbacks`: Alternative XBRL fact names tried if primary not found
- `sum_tolerance`: Allow ±1 difference due to rounding

### config/extraction_specs.json (PDF extraction)

PDF extraction rules with **keyword-based field matching**:

```json
{
  "sheet1_fields": {
    "value_fields": {
      "cv_gastos_personal": {"section": "nota_21", "row": 4},
      "cv_deprec_leasing": {"section": "nota_21", "row": 9},
      "cv_deprec_arrend": {"section": "nota_21", "row": 10}
    }
  },
  "default": {
    "sections": {
      "nota_21": {
        "field_mappings": {
          "cv_gastos_personal": {
            "match_keywords": ["gastos en personal"],
            "sheet1_field": "cv_gastos_personal"
          },
          "cv_deprec_leasing": {
            "match_keywords": ["leasing"],
            "sheet1_field": "cv_deprec_leasing"
          },
          "cv_deprec_arrend": {
            "match_keywords": ["arrendamiento"],
            "sheet1_field": "cv_deprec_arrend"
          },
          "cv_depreciacion_amort": {
            "match_keywords": ["amort"],
            "exclude_keywords": ["leasing", "arrendamiento"],
            "sheet1_field": "cv_depreciacion_amort"
          }
        }
      }
    }
  }
}
```

**Keyword matching rules:**
- `match_keywords`: At least one must match (case-insensitive)
- `exclude_keywords`: If any match, skip this field
- Fields processed in order - more specific patterns should come first

### config/config.json (Shared config)

File patterns and period types:

```json
{
  "period_types": {
    "quarterly": {
      "key_format": "{year}_Q{period}",
      "display_format": "{roman}Q{year}",
      "roman_numerals": {"1": "I", "2": "II", "3": "III", "4": "IV"}
    },
    "monthly": {
      "key_format": "{year}_M{period:02d}",
      "display_format": "M{period:02d}/{year}"
    },
    "yearly": {
      "key_format": "{year}_FY",
      "display_format": "FY{year}"
    }
  },
  "file_patterns": {
    "estados_financieros_pdf": {
      "pattern": "estados_financieros_{year}_Q{quarter}.pdf"
    },
    "estados_financieros_xbrl": {
      "pattern": "estados_financieros_{year}_Q{quarter}.xbrl",
      "alt_patterns": ["estados_financieros_{year}_Q{quarter}.xml"]
    }
  }
}
```

### config/reference_data.json (Validation)
Known-good values for validation:

```json
{
  "2024_Q2": {
    "verified": true,
    "values": {
      "ingresos_ordinarios": 179165,
      "total_costo_venta": -126202,
      "total_gasto_admin": -11632
    }
  }
}
```

The `cost_extractor.py` module loads extraction specs dynamically from these files.

## Learning from Failures (Feedback Loop)

When extraction fails on a new period but is successfully recovered (via OCR, manual review, or adjusted patterns), **always update the config files** to capture what worked:

### Step 1: Document What Failed and What Worked

After successful recovery, note:
- Which search patterns failed/succeeded
- Which page numbers contained the data
- Any label variations encountered
- OCR vs pdfplumber success

### Step 2: Update extraction_specs.json

**Add period-specific deviations:**
```json
{
  "2024_Q3": {
    "_comment": "Q3 2024 - Required adjusted search pattern",
    "verified": true,
    "verified_date": "2024-12-06",
    "deviations": {
      "sections": {
        "nota_21": {
          "search_patterns": ["21.- costo", "nota 21"],
          "_deviation_reason": "Q3 uses '21.-' format instead of '21.'"
        }
      }
    },
    "page_numbers": {
      "nota_21": 74,
      "nota_22": 74
    },
    "recovery_notes": "OCR fallback required for table extraction"
  }
}
```

**Consider updating defaults if pattern is common:**
```json
{
  "default": {
    "sections": {
      "nota_21": {
        "search_patterns": [
          "21. costo",
          "21.- costo",  // Added after Q3 experience
          "nota 21"
        ]
      }
    }
  }
}
```

### Step 3: Update reference_data.json

After validating extracted values:
```json
{
  "2024_Q3": {
    "verified": true,
    "verified_date": "2024-12-06",
    "values": {
      "ingresos_ordinarios": 231472,
      "total_costo_venta": -170862,
      "total_gasto_admin": -17363
    },
    "internal_checks": {
      "nota_21_sum_matches_total": true,
      "nota_22_sum_matches_total": true
    }
  }
}
```

### Step 4: Evaluate Default Updates

After 2-3 successful extractions with similar deviations, consider:

| Deviation Type | When to Update Default |
|----------------|------------------------|
| New search pattern | If same pattern works for 2+ periods |
| New pdf_label variant | Always add to defaults (accumulative) |
| Different page range | Only if consistently different |
| New table identifier | If distinguishes tables better |

**Example: Promoting a deviation to default:**
```python
# If Q2, Q3, Q4 all needed "energía" without accent
# Update default's unique_items:
"unique_items": ["energía eléctrica", "energia electrica", "servicios mineros"]
```

### Step 5: Run Tests

After updating configs, verify integrity:
```bash
poetry run pytest tests/test_config_integrity.py -v
```

This ensures:
- JSON syntax is valid
- Required fields are present
- Field names match across files
- Reference values are internally consistent

### Feedback Loop Summary

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
│  5. UPDATE extraction_specs.json:                                │
│     - Add period-specific deviations                             │
│     - Note what worked in recovery_notes                         │
│     - Consider updating defaults                                 │
│           │                                                      │
│           ▼                                                      │
│  6. UPDATE reference_data.json:                                  │
│     - Add verified values                                        │
│     - Mark verified: true                                        │
│           │                                                      │
│           ▼                                                      │
│  7. RUN TESTS to validate config integrity                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Troubleshooting

### Issue: Field not being matched

Check the `match_keywords` in extraction_specs.json. Debug by inspecting raw PDF items:

```python
from puco_eeff.extractor.cost_extractor import extract_nota_21

breakdown = extract_nota_21(pdf_path)
for item in breakdown.items:
    print(f'"{item.concepto}" = {item.ytd_actual}')
```

Then update keywords in `extraction_specs.json` to match the actual PDF text.

### Issue: XBRL values seem wrong (off by 1000x)

Check `apply_scaling` in `xbrl_specs.json`. All XBRL values from financial statements are in full USD and should have `apply_scaling: true` to convert to MUS$ (thousands).

**Example fix:**
```json
"total_costo_venta": {
  "primary": "CostOfSales",
  "apply_scaling": true  // This was likely false
}
```

### Issue: Depreciation fields swapped

The three depreciation fields in Nota 21 have similar labels. Use **unique** keywords:

```json
"cv_depreciacion_amort": {
  "match_keywords": ["amort"],
  "exclude_keywords": ["leasing", "arrendamiento"]
},
"cv_deprec_leasing": {
  "match_keywords": ["leasing"]  // Unique identifier
},
"cv_deprec_arrend": {
  "match_keywords": ["arrendamiento"]  // Unique identifier
}
```

### Issue: File not found

File patterns are in `config.json`. Check:

```python
from puco_eeff.config import format_filename
print(format_filename("estados_financieros_pdf", 2024, 2))
```

### Issue: Wrong "Totales" value
- Check section context - must be in Gasto Admin section (Nota 22)
- "Total Costo de Venta" should be in row 15, not row 27

### Issue: Duplicate field values
- "Gastos en personal" appears in both sections - use section context
- Costo de Venta values are typically 5x larger than Gasto Admin

## Adding Support for New Periods

### Step 1: Add period entry to extraction_specs.json

```json
{
  "2024_Q4": {
    "verified": false,
    "deviations": {}
  }
}
```

### Step 2: Run extraction and check output

```python
from puco_eeff.extractor.cost_extractor import extract_sheet1
data = extract_sheet1(2024, 4)
```

### Step 3: If successful, add reference values

In `config/reference_data.json`:

```json
{
  "2024_Q4": {
    "verified": true,
    "values": {
      "ingresos_ordinarios": 295000,
      "total_costo_venta": -210000,
      "total_gasto_admin": -15000,
      "cv_gastos_personal": -28000,
      ...
    }
  }
}
```

### Step 4: If field matching fails

Update `match_keywords` in extraction_specs.json to add variations:

```json
"cv_energia": {
  "match_keywords": ["energía", "energia", "eléctrica", "electrica"],
  "sheet1_field": "cv_energia"
}
```

### Step 5: Run tests to validate

```bash
cd puco-EEFF
poetry run pytest tests/test_cost_extractor.py tests/test_config_integrity.py -v
```

## Running Tests

```bash
cd puco-EEFF
poetry run pytest tests/ -v                        # All tests (124)
poetry run pytest tests/test_cost_extractor.py -v  # Extractor tests (50)
poetry run pytest tests/test_config_integrity.py -v # Config tests (43)
poetry run pytest tests/test_environment.py -v     # Environment tests (31)
```

## Next Steps

1. Continue to `04_format_sheet1.md` for Excel formatting
2. See `data_mapping.md` for detailed XBRL and PDF field mappings

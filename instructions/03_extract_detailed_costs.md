# 03 - Extract Detailed Cost Breakdown (Sheet1)

## Objective

Extract the **"Cuadro Resumen de Costos"** from Análisis Razonado into Sheet1 with a 27-row structure:

- **Row 1**: Ingresos de actividades ordinarias
- **Rows 3-15**: Costo de Venta section (header + 11 line items + total)
- **Rows 19-27**: Gasto Admin y Ventas section (header + 6 line items + **Totales**)

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
from puco_eeff.extractor import (
    Sheet1Data,
    extract_sheet1_from_analisis_razonado,
    print_sheet1_report,
    save_sheet1_data,
    format_quarter_label,
)

# Extract from Análisis Razonado
data = extract_sheet1_from_analisis_razonado(year=2024, quarter=2)

# Print formatted report
print_sheet1_report(data)

# Save to JSON
save_sheet1_data(data)

# Access specific fields
print(f"Quarter: {data.quarter}")  # "IIQ2024"
print(f"Ingresos: {data.ingresos_ordinarios:,}")  # 179,165
print(f"Total Costo de Venta: {data.total_costo_venta:,}")  # -126,202
print(f"Totales (Gasto Admin): {data.total_gasto_admin:,}")  # -11,632
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
- **Document**: `analisis_razonado_YYYY_QN.pdf`
- **Section**: "Cuadro Resumen de Costos" 
- **XBRL Validation**: Available for cross-checking totals

### Pucobre.cl Fallback (Q1)
- **Document**: `analisis_razonado_YYYY_Q1.pdf` (split from combined PDF)
- **XBRL**: NOT available - PDF-only extraction

## Batch Extraction Across Quarters

```python
from puco_eeff.extractor import extract_sheet1_from_analisis_razonado, save_sheet1_data

# Extract all available quarters
quarters_data = {}
for quarter in [1, 2, 3, 4]:
    for year in [2024]:
        data = extract_sheet1_from_analisis_razonado(year, quarter)
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

## Configuration

Sheet1 structure is defined in `config/config.json`:

```json
{
  "sheets": {
    "sheet1": {
      "name": "Ingresos y Costos",
      "description": "Revenue and detailed cost breakdown from Análisis Razonado",
      "layout": {
        "total_rows": 27,
        "sections": [...]
      },
      "row_mapping": {
        "1": {"field": "ingresos_ordinarios", ...},
        "15": {"field": "total_costo_venta", ...},
        "27": {"field": "total_gasto_admin", "label": "Totales", ...}
      },
      "extraction_rules": {
        "disambiguation": {
          "totales_gasto_admin": {
            "context": "Must appear AFTER 'Gasto Adm, y Ventas' header",
            "not_to_confuse_with": "Total Costo de Venta"
          }
        }
      }
    }
  }
}
```

## Troubleshooting

### Issue: Wrong "Totales" value
- Check section context - must be in Gasto Admin section
- "Total Costo de Venta" should be in row 15, not row 27

### Issue: Duplicate field values
- "Gastos en personal" appears twice - use section context
- Costo de Venta values are typically 5x larger than Gasto Admin

### Issue: Missing Análisis Razonado
- Q1 may require Pucobre.cl fallback
- Check that PDF splitting worked correctly

## Next Steps

1. Continue to `04_extract_main_statements.md` for Income Statement and Balance Sheet
2. Format data for Excel output in `05_format_sheets.md`

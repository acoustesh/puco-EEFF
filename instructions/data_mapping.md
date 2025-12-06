# Data Mapping - Pucobre Financial Statements

## Overview

This document maps the data structure between what was requested (Q1 2024 example) and what was found in actual CMF documents (Q3 2024 verified).

## Key Findings

### Data Source Summary

| Data Type | Source | Location |
|-----------|--------|----------|
| Ingresos de actividades ordinarias | XBRL | `Revenue` fact |
| Costo de Venta (total) | XBRL | `CostOfSales` fact |
| Costo de Venta (desglose) | PDF | Nota 21, Page 71 |
| Gastos Admin y Ventas (total) | XBRL | `AdministrativeExpense` fact |
| Gastos Admin y Ventas (desglose) | PDF | Nota 22, Page 71 |
| Ganancia bruta | XBRL | `GrossProfit` fact |
| Ganancia del periodo | XBRL | `ProfitLoss` fact |

### Q1 2024 User Example vs Q3 2024 Actual Structure

#### Costo de Venta - Line Items Comparison

| User Example (Q1 2024) | Actual PDF (Q3 2024) | Status |
|------------------------|----------------------|--------|
| Gastos en personal | Gastos en personal | ✓ Match |
| Materiales y repuestos | Materiales y repuestos | ✓ Match |
| Energía eléctrica | Energía eléctrica | ✓ Match |
| Servicios de terceros | Servicios de terceros | ✓ Match |
| Depreciación y amort. | Depreciación y amort. del periodo | ✓ Match |
| Depreciación Activos en leasing | Depreciación Activos en leasing (Nota 20) | ✓ Match |
| Depreciación Arrendamientos | Depreciación Arrendamientos (Nota 20) | ✓ Match |
| Servicios mineros de terceros | Servicios mineros de terceros | ✓ Match |
| Fletes y otros | Fletes y otros gastos operacionales | ✓ Match |
| Gastos Diferidos | Gastos Diferidos, ajustes existencias y otros | ✓ Match |
| Obligaciones por convenios colectivos | Obligaciones por convenios colectivos | ✓ Match |

**Note**: Q3 2024 has the same 11 line items as Q1 2024 example.

#### Gastos Admin y Ventas - Line Items Comparison

| User Example (Q1 2024) | Actual PDF (Q3 2024) | Status |
|------------------------|----------------------|--------|
| Gastos en personal | Gastos en personal | ✓ Match |
| Materiales y repuestos | Materiales y repuestos | ✓ Match |
| Servicios de terceros | Servicios de terceros | ✓ Match |
| Provisión gratificación legal | Provisión gratificación legal y otros | ✓ Match |
| Gastos comercialización | Gastos comercialización | ✓ Match |
| Otros gastos | Otros gastos | ✓ Match |

**Note**: Q3 2024 has the same 6 line items as Q1 2024 example.

## Sample Values Comparison

### Q1 2024 (User Example) - Values in MUSD

```
Ingresos de actividades ordinarias:  80,767

COSTO DE VENTA:
  Gastos en personal              -9,857
  Materiales y repuestos         -10,986
  Energía eléctrica               -4,727
  Servicios de terceros          -11,723
  Depreciación y amort.          -10,834
  Depreciación Activos en leasing   -485
  Depreciación Arrendamientos       -711
  Servicios mineros de terceros   -3,675
  Fletes y otros                  -2,489
  Gastos Diferidos                  -875
  Obligaciones convenios colect.  -6,620
  TOTAL COSTO VENTA:             -62,982

GASTOS ADM Y VENTAS:
  Gastos en personal              -1,831
  Materiales y repuestos             -56
  Servicios de terceros           -1,250
  Provisión gratificación legal     -966
  Gastos comercialización           -805
  Otros gastos                      -229
  TOTAL GASTOS ADM:               -5,137
```

### Q3 2024 (Actual Extracted) - Values in MUSD (YTD 9 months)

```
Ingresos de actividades ordinarias: 231,472

COSTO DE VENTA (Nota 21):
  Gastos en personal             -30,294
  Materiales y repuestos         -37,269
  Energía eléctrica              -14,710
  Servicios de terceros          -39,213
  Depreciación y amort.          -33,178
  Depreciación Activos en leasing -1,354
  Depreciación Arrendamientos     -2,168
  Servicios mineros de terceros  -19,577
  Fletes y otros gastos          -7,405
  Gastos Diferidos               +20,968  (positive adjustment)
  Obligaciones convenios colect.  -6,662
  TOTAL COSTO VENTA:            -170,862

GASTOS ADM Y VENTAS (Nota 22):
  Gastos en personal              -5,727
  Materiales y repuestos            -226
  Servicios de terceros           -4,474
  Provisión gratificación legal   -2,766
  Gastos comercialización         -3,506
  Otros gastos                      -664
  TOTAL GASTOS ADM:              -17,363
```

## Number Formatting

### Chilean Financial Documents

| Format | Meaning | Example |
|--------|---------|---------|
| `1.234` | 1,234 (thousands separator) | 30.294 = 30,294 |
| `1,5` | 1.5 (decimal separator) | 1,5 = 1.5 |
| `(30.294)` | Negative value | -30,294 |
| `MUS$` | Miles de USD | Thousands of dollars |
| `MUSD` | Miles de USD | Same as MUS$ |

### Parsing Rules

1. **Parentheses = Negative**: `(30.294)` → `-30294`
2. **Period = Thousands**: `30.294` → `30294`
3. **Remove currency symbols**: `MUS$`, `US$`, `$`
4. **Scale**: Values are already in thousands (MUSD/MUS$)

## Column Mapping

### PDF Table Columns (Nota 21 & 22)

| Column Index | Header | Description |
|--------------|--------|-------------|
| 0 | Concepto | Line item description |
| 1 | 01-01-YYYY 30-09-YYYY | YTD Current Year |
| 2 | 01-01-YYYY 30-09-YYYY | YTD Prior Year |
| 3 | 01-07-YYYY 30-09-YYYY | Q3 Current Year |
| 4 | 01-07-YYYY 30-09-YYYY | Q3 Prior Year |

**For extraction**: Use Column 1 (YTD current year) as the primary value.

## XBRL Fact Name Mapping

| Spanish Name | XBRL Fact | Context |
|--------------|-----------|---------|
| Ingresos de actividades ordinarias | `Revenue` | Duration (YTD) |
| Costo de ventas | `CostOfSales` | Duration (YTD) |
| Ganancia bruta | `GrossProfit` | Duration (YTD) |
| Gastos de administración y ventas | `AdministrativeExpense` | Duration (YTD) |
| Ganancia (pérdida) | `ProfitLoss` | Duration (YTD) |
| Total activos | `Assets` | Instant (end of period) |
| Total pasivos | `Liabilities` | Instant (end of period) |
| Patrimonio total | `Equity` | Instant (end of period) |

## Validation Rules

### Cross-Validation Between Sources

1. **XBRL vs PDF Totals**:
   - `CostOfSales` (XBRL) = Sum of Nota 21 line items
   - `AdministrativeExpense` (XBRL) = Sum of Nota 22 line items

2. **Income Statement Check**:
   - `GrossProfit` = `Revenue` - `CostOfSales`
   - Should match within rounding tolerance (±1 MUSD)

3. **Line Item Sum Check**:
   - Sum all Nota 21 items = "Totales" row
   - Sum all Nota 22 items = "Totales" row

## Period-Specific Notes

### Q1 Reports (Trimestre 1)
- Downloaded month: March (03)
- YTD period: 01-01 to 31-03
- 3 months of data

### Q2 Reports (Trimestre 2)
- Downloaded month: June (06)
- YTD period: 01-01 to 30-06
- 6 months of data

### Q3 Reports (Trimestre 3)
- Downloaded month: September (09)
- YTD period: 01-01 to 30-09
- 9 months of data

### Q4 Reports / Annual
- Downloaded month: December (12)
- YTD period: 01-01 to 31-12
- 12 months of data (full year)

## File Naming Convention

| Document Type | File Pattern |
|---------------|--------------|
| Análisis Razonado | `analisis_razonado_YYYY_QN.pdf` |
| Estados Financieros | `estados_financieros_YYYY_QN.pdf` |
| XBRL Data | `estados_financieros_YYYY_QN.xml` |

## Known Variations by Period

Some line items may vary between periods:

1. **"Obligaciones por convenios colectivos"**: May be 0 or absent in some quarters (only present when collective bargaining occurs)
2. **"Gastos Diferidos, ajustes existencias y otros"**: Can be positive (adjustment) or negative
3. **Line item names**: May have slight variations (e.g., "y otros" suffix)

## Extraction Priority

1. **XBRL first**: Use for all aggregate totals (most reliable, structured data)
2. **PDF tables**: Use pdfplumber for detailed line items
3. **OCR fallback**: Only when PDF extraction fails (image-based tables)

Always validate OCR results against XBRL totals.

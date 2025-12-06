# 08 - Format Sheet 2 for Excel

## Objective

Format Estado de Resultados data for the final Excel workbook.

## Steps

### 1. Load and Normalize

```python
from puco_eeff.writer.sheet_writer import load_sheet_data
from puco_eeff.transformer.normalizer import normalize_financial_data

year, quarter = 2024, 3
period = f"{year}_Q{quarter}"

df = load_sheet_data("sheet2", period)
df_normalized = normalize_financial_data(df, numeric_columns=["valor"])
```

### 2. Apply Standard Structure

```python
standard_structure = [
    {"concepto": "INGRESOS DE ACTIVIDADES ORDINARIAS", "nivel": 0, "tipo": "total"},
    {"concepto": "Costo de ventas", "nivel": 1, "tipo": "item"},
    {"concepto": "GANANCIA BRUTA", "nivel": 0, "tipo": "subtotal"},
    {"concepto": "Gastos de administración", "nivel": 1, "tipo": "item"},
    {"concepto": "Gastos de distribución", "nivel": 1, "tipo": "item"},
    {"concepto": "Otros ingresos", "nivel": 1, "tipo": "item"},
    {"concepto": "Otros gastos", "nivel": 1, "tipo": "item"},
    {"concepto": "RESULTADO OPERACIONAL", "nivel": 0, "tipo": "subtotal"},
    {"concepto": "Ingresos financieros", "nivel": 1, "tipo": "item"},
    {"concepto": "Costos financieros", "nivel": 1, "tipo": "item"},
    {"concepto": "Diferencia de cambio", "nivel": 1, "tipo": "item"},
    {"concepto": "RESULTADO ANTES DE IMPUESTO", "nivel": 0, "tipo": "subtotal"},
    {"concepto": "Gasto por impuesto a las ganancias", "nivel": 1, "tipo": "item"},
    {"concepto": "GANANCIA (PÉRDIDA) DEL PERÍODO", "nivel": 0, "tipo": "grand_total"},
]
```

### 3. Validate

```python
def validate_income_statement(df):
    """Validate income statement calculations."""
    # Gross Profit = Revenue - COGS
    # Operating Income = Gross Profit - Operating Expenses
    # Net Income = Operating Income +/- Financial items - Tax
    pass

validate_income_statement(df_normalized)
```

### 4. Save

```python
from puco_eeff.writer.sheet_writer import save_sheet_data, write_sheet_to_csv

save_sheet_data("estado_resultados", df_normalized, period)
write_sheet_to_csv("estado_resultados", df_normalized, period)
```

## Next Steps

Continue with additional sheets (Sheet 3: Estado de Flujos de Efectivo, Sheet 4: Notas) following the same locate → extract → format pattern, or proceed to `NN_combine_workbook.md` if all sheets are ready.

import re

text = """Cobre en concentrados MUS$ 142.391 123.356
Cobre en Cátodos MUS$ 15.022 12.464
Oro subproducto MUS$ 20.223 14.934
Plata subproducto MUS$ 1.529 761
Total Ingresos operacionales MUS$ 179.165 151.515
Libras de cobre vendidas Millones Libras 3 8,4 3 8,6
Total Toneladas Procesadas 2.683 2.817
Cobre Fino Obtenido 38,0 38,3
Subproducto Oro 8,2 8,2
Cash Cost descontado US$ / libra 2 ,59 2 ,72
Precio ventas efectivo pucobre US$/libra Cu 4 ,30 3 ,87
Costo unitario total US$ / libra 3 ,57 3 ,43
Ebitda del periodo MUS$ 6 5.483 4 3.742"""


def parse_num(s):
    s = s.replace(" ", "")
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
        return float(s)
    if "." in s and len(s.split(".")[-1]) == 3:
        return int(s.replace(".", ""))
    return int(s) if s.isdigit() else float(s)


for line in text.split("\n"):
    nums = re.findall(r"\d[\d\s.,]*", line)
    if len(nums) >= 2:
        current = nums[-2].strip()
        try:
            curr_val = parse_num(current)
            print(f"{line[:50]:<50} -> {curr_val}")
        except Exception as e:
            print(f"{line[:50]:<50} -> FAIL: {e}")

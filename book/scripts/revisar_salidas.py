"""Revisión de la construcción: busca errores de ejecución en las salidas de las celdas de cada página.
Uso: python scripts/revisar_salidas.py [_build/site/content] [-v]"""
import json
import sys
from pathlib import Path

carpeta = Path(next((a for a in sys.argv[1:] if not a.startswith("-")), "_build/site/content"))
verboso = "-v" in sys.argv
errores = 0


def recorrer(nodo, pagina):
    global errores
    if isinstance(nodo, dict):
        if nodo.get("type") == "output":
            datos = nodo.get("jupyter_data") or {}
            if datos.get("output_type") == "error":
                errores += 1
                print(f"ERROR en {pagina}: {datos.get('ename')}: {datos.get('evalue')}")
            elif verboso:
                texto = datos.get("text") or (datos.get("data") or {}).get("text/plain") or ""
                print(f"[{pagina}]", (texto if isinstance(texto, str) else "".join(texto))[:400])
        for hijo in nodo.get("children", []):
            recorrer(hijo, pagina)


for ruta in sorted(carpeta.glob("*.json")):
    pagina = json.loads(ruta.read_text(encoding="utf-8"))
    recorrer(pagina.get("mdast", {}), ruta.stem)
print(f"revisar_salidas: {errores} celdas con error")
sys.exit(1 if errores else 0)

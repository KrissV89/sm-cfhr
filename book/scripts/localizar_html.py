"""Traduce al español las etiquetas fijas que mystmd escribe en los pies («Figure 1:», «Table 2:», «Program 1:»).

mystmd fija esas etiquetas en inglés; las referencias cruzadas ya usan las plantillas de myst.yml («Figura %s»).
Se reemplazan solo las cadenas con espacio no separable que usa mystmd en los pies, en HTML y JSON.
Uso: python scripts/localizar_html.py _build/html
"""
import sys
from pathlib import Path

PARES = [("Figure", "Figura"), ("Table", "Tabla"), ("Program", "Programa"), ("Algorithm", "Algoritmo"),
         ("Proposition", "Proposición"), ("Proof", "Demostración"), ("Equation", "Ecuación")]
NBSP = [" ", "\\u00a0", "&nbsp;", "&#160;"]


def main(carpeta: str) -> None:
    cambiados = 0
    for ruta in Path(carpeta).rglob("*"):
        if ruta.suffix not in {".html", ".json"} or not ruta.is_file():
            continue
        texto = ruta.read_text(encoding="utf-8")
        nuevo = texto
        for en, es in PARES:
            for sep in NBSP:
                nuevo = nuevo.replace(en + sep, es + sep)
        if nuevo != texto:
            ruta.write_text(nuevo, encoding="utf-8")
            cambiados += 1
    print(f"localizar_html: {cambiados} archivos con etiquetas traducidas")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "_build/html")

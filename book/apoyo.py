"""Funciones comunes de las celdas del libro: rutas, lectura de resultados y formato de tablas.

Las celdas solo leen archivos ya calculados en ``resultados/``, ``data/`` y ``artifacts/`` (o, en el capítulo 2,
los datos de ``base_generacion_clima/`` y ``datos_pronostico/``). Ningún cálculo costoso se repite aquí: cada
tabla indica el cuaderno o el script que generó su archivo de origen.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from IPython.display import Markdown, display

RAIZ = Path(__file__).resolve().parent.parent      # raíz del repositorio (book/ está dentro)
RES = RAIZ / "resultados"
if str(RAIZ) not in sys.path:                      # para importar src.original_model, src.features, ...
    sys.path.insert(0, str(RAIZ))

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 200)


def leer(ruta: str, **kw) -> pd.DataFrame:
    """CSV relativo a la raíz del repositorio."""
    return pd.read_csv(RAIZ / ruta, **kw)


def leer_json(ruta: str) -> dict:
    return json.loads((RAIZ / ruta).read_text(encoding="utf-8"))


def existe(ruta: str) -> bool:
    return (RAIZ / ruta).exists()


def tabla(df: pd.DataFrame, decimales: int | dict = 4, indice: str | None = None) -> pd.DataFrame:
    """Redondea las columnas numéricas y, si se indica, usa una columna como índice para mostrarla."""
    df = df.copy()
    if isinstance(decimales, dict):
        for col, d in decimales.items():
            if col in df:
                df[col] = df[col].round(d)
    else:
        num = df.select_dtypes("number").columns
        df[num] = df[num].round(decimales)
    if indice is not None:
        df = df.set_index(indice)
        df.index.name = indice
    return df


def nota(texto: str) -> None:
    """Muestra un párrafo en Markdown (se usa para declarar el origen de cada tabla)."""
    display(Markdown(texto))

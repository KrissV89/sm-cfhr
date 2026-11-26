---
title: Reproducibilidad y bibliografía
short_title: 10. Reproducibilidad
kernelspec:
  name: python3
  display_name: Python 3
---

# Reproducibilidad y bibliografía

## Entorno

Todos los cálculos se ejecutaron con Python 3.14.10 (fijado en `.python-version` y `render.yaml`), scikit-learn 1.9.1, XGBoost 3.4.1, NumPy 2.5.4, pandas 3.1.1 y SciPy 1.18.1, en un procesador Intel Core i5-12450H de 12 hilos con 16 GB de memoria y sin GPU. El estudio de optimizadores usó además Optuna 5.1.0, DEAP 1.4, SHAP 0.54.1 y LIME 0.2.0.1. Las dependencias exactas del dashboard están en `requirements.txt`; las de las pruebas y los scripts de `experiments/`, en `requirements-dev.txt`. El libro se construyó con Jupyter Book 2.1.7 (mystmd).

```bash
python3.14 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt     # app, pruebas y SHAP
.venv/bin/python -m pytest -q                      # 30 pruebas
.venv/bin/python scripts/ejecutar_notebook.py notebooks/actuales/<cuaderno>.ipynb
cd book && ./construir.sh                          # este libro (HTML en book/_build/html)
```

Los cuadernos, las figuras y el libro necesitan además los paquetes de `requirements-notebooks.txt` (Optuna, DEAP, LIME, Matplotlib, nbclient y Jupyter Book, con las versiones usadas). Esa lista se derivó del entorno en uso y no se verificó con una instalación desde cero.

La tabla siguiente reúne lo que cada cuaderno registró en su `entorno.json`.

```{code-cell} python
import hashlib
import json
import apoyo
import pandas as pd

filas = []
for f in sorted(apoyo.RES.glob("*/entorno.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    filas.append({"carpeta": f.parent.name, "fecha": d.get("fecha", ""), "python": d.get("python", ""),
                  "scikit-learn": d.get("scikit-learn", ""), "semilla": d.get("semilla", ""), "procesos": d.get("procesos", ""),
                  "minutos": d.get("minutos", d.get("minutos_principal", ""))})
pd.DataFrame(filas).set_index("carpeta")
```

## Semillas y guardas

- **Semilla 2026** en todos los modelos y optimizadores con componente aleatorio; los estudios de semillas usan 2026 a 2035 (cuadernos 21 y 22).
- **Prueba usada una vez.** El cuaderno 07 escribió la selección de desarrollo (`resultados/pronostico_d1/seleccion_desarrollo.json`) antes de ejecutar la prueba e impide repetirla. El cuaderno 19 guarda `inicio.json` y `predicciones.pkl` para que la evaluación de W1 no se repita. Estas guardas no deben borrarse.
- **Protocolos con huella.** Los cuadernos 21 y 23 registraron su protocolo con su huella SHA-256 antes de calcular y comprueban la huella al empezar; las desviaciones se anotan aparte (`DESVIACIONES.md`). La celda siguiente vuelve a calcular las huellas.

```{code-cell} python
filas = []
for f in sorted(apoyo.RES.glob("*/PROTOCOLO.sha256")):
    registrada = f.read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256((f.parent / "PROTOCOLO.md").read_bytes()).hexdigest()
    filas.append({"carpeta": f.parent.name, "huella registrada": registrada[:16], "huella actual": actual[:16],
                  "coincide": registrada == actual})
pd.DataFrame(filas).set_index("carpeta")
```

## Mapa de cuadernos

| Cuaderno | Contenido | Resultados | Capítulo |
|---|---|---|---|
| 01 | Descarga y consolidación de XM, Open-Meteo y GEM | `base_generacion_clima/` | 2 |
| 02 | EDA y calidad de datos | `paper/figuras/eda_*` | 2 |
| 03 | Benchmark de la Tarea 1 | `resultados/benchmark/` | 3 |
| 04 | Adaptaciones propias C1–C5 de la Tarea 1 | `resultados/modelos_propuestos/` | 3 |
| 05 | Exploración de variables de planta | `resultados/exploracion_planta/` | 2 |
| 06 | Descarga del pronóstico archivado | `datos_pronostico/` | 2 |
| 07 | Benchmark $D{+}1$, selección y prueba única | `resultados/pronostico_d1/` | 3, 6 |
| 08 | Análisis complementario | — | 3 |
| 09 | Revisión bibliográfica | — | 4 |
| 10 | Ablación de K1 | `resultados/ablacion_k1/` | 5 |
| 11 | Plantas no vistas | `resultados/pronostico_d1_plantas_no_vistas/` | 7 |
| 12 | Ampliación de la cohorte | `base_generacion_clima/` | 2, 7 |
| 13 | Significancia e intervalos (reanálisis) | `resultados/significancia_intervalos/` | 6 |
| 14 | Plantas nuevas | `resultados/pronostico_d1_plantas_nuevas/` | 7 |
| 15 | Catálogo de conjuntos tabulares | — | 7 |
| 16 | Validación conjunta con 32 plantas | `resultados/pronostico_d1_conjunto_loo/` | 7 |
| 17 | Ronda de candidatos (selección de W1) | `resultados/modelos_adicionales/` | 3, 5 |
| 18 | Limpieza de plantas nuevas | `resultados/limpieza_plantas_nuevas/` | 2 |
| 19 | Evaluación exploratoria de W1 | `resultados/evaluacion_ronda1/` | 6 |
| 20 | Lasso, Ridge bayesiana y árbol | `resultados/referencias_curso/` | 3 |
| 21 | Cuatro optimizadores, semillas, SHAP y LIME | `resultados/optimizadores/` | 3 |
| 22 | Sensibilidad, semillas, ruido, costo e intervalos | `resultados/robustez/` | 7 |
| 23 | Tres conjuntos de datos adicionales | `resultados/datasets_adicionales/` | 7 |

Las pruebas de la guía (DM-HLN, Giacomini–White, bayesiana y MCS) las calcula `scripts/pruebas_guia.py` en `resultados/pruebas_guia/`; las figuras del manuscrito, `scripts/generar_figuras_paper.py`; y el manuscrito se compila con `scripts/compilar_paper.sh` (Tectonic).

## Lista de verificación

| Elemento | Estado |
|---|---|
| Cuadernos 01–23 ejecutados y guardados con salidas | Hecho |
| Protocolos con huella en 21 y 23; desviaciones registradas | Hecho |
| Guardas de la prueba única (07, 19) | Hecho; no borrar |
| Pruebas unitarias (30) e integración continua | La batería pasó en local el 16-11-2026; no hay constancia de una ejecución en GitHub Actions |
| Artefactos serializados con prueba de ida y vuelta | Hecho ([capítulo 8](08_dashboard.md)) |
| Libro reconstruible desde `resultados/` | Hecho: las celdas solo leen archivos |
| Reproducción desde un clon limpio | **No verificada**: varios CSV y PKL intermedios están excluidos por `.gitignore`, y los paquetes de los cuadernos se fijaron a posteriori (`requirements-notebooks.txt`) sin probarlos en un entorno limpio |
| Licencia del código | PolyForm Noncommercial 1.0.0 (`LICENSE`) |
| DOI y firma | Pendiente de depósito en Zenodo |

## Declaración de autoría

Autor: Enmanuel David Diaz Molinares, Departamento de Matemáticas y Ciencia de Datos, Programa de Ciencia de Datos, Universidad del Norte, Barranquilla, Colombia.

Los datos proceden de XM, Open-Meteo y Global Energy Monitor; todas las cifras del manuscrito, del libro y del tablero provienen directamente de los cuadernos y de los archivos generados en `resultados/`. Las decisiones de diseño, las ejecuciones y sus límites quedaron registrados en los cuadernos, en los protocolos con huella SHA-256 y en el historial de Git.

Los autores declaran que revisaron íntegramente el código, los experimentos y el análisis estadístico presentados.

| Autor | Institución | Fecha |
|---|---|---|
| Enmanuel David Diaz Molinares | Universidad del Norte | 2026-11-26 |

## Bibliografía

Cada capítulo lista al pie las referencias que cita. La celda siguiente reúne todas las entradas de `book/references.bib`, incluidas las que solo se citan en el manuscrito.

```{code-cell} python
import re
texto = (apoyo.RAIZ / "book" / "references.bib").read_text(encoding="utf-8")
entradas = []
for bloque in re.split(r"\n(?=@)", texto):
    m = re.match(r"@(\w+)\{([^,]+),", bloque.strip())
    if not m:
        continue
    # Campo entre llaves equilibradas hasta dos niveles de anidamiento, como está escrito references.bib.
    campo = lambda c: (re.search(r"\b" + c + r"\s*=\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", bloque) or [None, ""])[1]
    acentos = {"\\'e": "é", "\\'a": "á", "\\'o": "ó", "\\'u": "ú", "\\'i": "í", "\\'\\i": "í", '\\"u': "ü", '\\"o': "ö",
               "\\~n": "ñ", "\\c{c}": "ç", "\\c c": "ç"}
    def limpio(t):
        t = re.sub(r"\s+", " ", t)
        for a, b in acentos.items():
            t = t.replace("{" + a + "}", b).replace(a, b)
        return re.sub(r"[{}\\]", "", t).strip()
    entradas.append({"clave": m.group(2), "autores": limpio(campo("author"))[:80], "año": limpio(campo("year")),
                     "título": limpio(campo("title"))})
pd.DataFrame(entradas).sort_values(["autores", "año"]).set_index("clave")
```

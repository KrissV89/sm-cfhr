# Auditoría del Dashboard Predictivo, Jupyter Book y Trazabilidad Entregable 2

**Fecha:** 2026-11-23

---

## 1. Auditoría del Dashboard (`app.py`, `pages/`, `render.yaml`)

- **Arquitectura verificada (`DOCUMENTADO`):**
  - Aplicación multipágina en Dash (`app.py`) con tres rutas: `/` (`pages/contexto.py`), `/eda` (`pages/eda.py`) y `/modelos` (`pages/modelos.py`).
  - Configuración de despliegue en `render.yaml`: `gunicorn app:server --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT`, `healthCheckPath: /`, `DASH_DEBUG: "0"`.
  - **Buena práctica confirmada:** `data.precargar()` carga únicamente extractos livianos precalculados en `data/` y artefactos serializados en `artifacts/`; **no entrena ni optimiza en tiempo de ejecución**.
- **Brechas y estado de pruebas:**
  1. **Versión de modelos mostrada:** actualmente `app.py` y `pages/modelos.py` presentan `K1` (`ModeloAnomalias`) y `W1` (`RegresionLocal`) del ciclo previo. La integración de `SM-CFHR` (`AdaptiveStructuredHuberRegressor`) y sus métricas OOF definitivas en el dashboard queda **`PENDIENTE`** hasta ejecutar `--ejecutar-oof-definitivo` y generar sus artefactos serializados.
  2. **Prueba de arranque en entorno actual:** el paquete `dash` no está instalado en el entorno `solarpv-eda` (`{'dash': False, 'plotly': True}`), por lo que el levantamiento en vivo del servidor desde `solarpv-eda` se registra como **`NO PROBADO EN ESTE ENTORNO`** (requiere el entorno de `requirements.txt` usado en Render).

---

## 2. Auditoría del Jupyter Book (`book/`)

- **Estructura verificada:**
  - `book/_config.yml`, `book/_toc.yml`, `book/myst.yml`, `book/construir.sh`, `book/references.bib` y 10 capítulos (`01_correcciones.md` a `10_reproducibilidad.md`).
  - Herramienta CLI verificada en el sistema: `C:\Users\Enma\miniconda3\Scripts\jupyter-book.exe` **v2.1.6** (Jupyter Book 2 / MyST).
- **Estado y hoja de ruta:**
  - El libro existente compila la narrativa técnica del Entregable 2 (`K1`/`W1`) y mantiene una tabla verificadora ejecutable de evidencias en `book/01_correcciones.md`.
  - No se reescribe `book/` antes de la ejecución OOF definitiva para no mezclar resultados preliminares con definitivos; una vez autorizado y ejecutado `--ejecutar-oof-definitivo`, se incorporará el capítulo definitivo de `SM-CFHR` leyendo `resultados/evaluacion_ashr_oof/` y `figuras/ashr_oof/`.

---

## 3. Trazabilidad de Correcciones del Entregable 2

Auditado frente a `book/01_correcciones.md` y `MODEL_CARD.md`:
- Las brechas reportadas en `book/01_correcciones.md` para `K1/W1` (R1 no acreditado, R2 parcial porque `K1` delegaba en `sklearn.Ridge`, R3 no cumplido en `K1/W1`, falta de compatibilidad completa `BaseEstimator` en modelos propios) quedan **resueltas arquitectónicamente** en `SM-CFHR`:
  - **R2:** núcleo propio 100% NumPy/SciPy + herencia `(BaseEstimator, RegressorMixin)` verificada.
  - **R1:** protocolo WoS documentado + afirmación de novedad acotada al dominio FV multi-planta.
  - **R3:** infraestructura estadística completa (`DM-HLN-HAC`, `GW`, `Clark-West`, `MCS`, Holm) lista para evaluación confirmatoria en los 28 escenarios solares.
- Gobernanza documental presente y verificada en disco: `MODEL_CARD.md`, `data/DATA_STATEMENT.md`, `CITATION.cff`, `LICENSE` (MIT), `.github/workflows/tests.yml`.

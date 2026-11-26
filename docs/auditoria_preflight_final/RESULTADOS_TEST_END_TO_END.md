# Resultados del Experimento End-to-End sobre Desarrollo y Estado B Aislado

**Script:** `scripts/evaluacion/preflight_ashr_desarrollo.py`  
**Directorio exclusivo de preflight:** `resultados/preflight_ashr/`

---

## 1. Cobertura End-to-End Real en 5 Regímenes de Desarrollo (`e2e_desarrollo_resumen.csv`)

| Escenario | Variante | Trials | Obs. eval | MAE macro (kWh) | Tiempo (s) | Mem. pico (MB) |
|---|---|---:|---:|---:|---:|---:|
| `D1_espaciotemporal_cinco_variables` | `Fixed` | 0 | 365 | 43,729.13 | 0.088 | 5.179 |
| `D1_espaciotemporal_cinco_variables` | `HPO_B8` | 8 | 365 | 43,364.03 | 1.430 | — |
| `D1_espaciotemporal_cinco_variables` | **`HPO_B25` (real)** | **25** | **365** | **43,275.81** | **3.652** | — |
| `D1_intraplanta_sin_meteorologia` | `Fixed` | 0 | 184 | 80,762.87 | 0.036 | 0.412 |
| `D1_intraplanta_sin_meteorologia` | `HPO_B8` | 8 | 184 | 105,182.73 | 0.669 | — |
| `RecGEFSv12_intraplanta_14d` | `Fixed` | 0 | 28 | 27,767.18 | 0.330 | 0.214 |
| `RecGEFSv12_intraplanta_14d` | `HPO_B8` | 8 | 28 | 24,045.77 | 1.286 | — |
| `RecGEFSv12_espaciotemporal_28d` | `Fixed` | 0 | 28 | 95,509.88 | 0.196 | 0.219 |
| `RecGEFSv12_espaciotemporal_28d` | `HPO_B8` | 8 | 28 | 154,967.64 | 0.598 | — |
| `DirectoSinClima_espaciotemporal_28d` | `Fixed` | 0 | 2 | 1,610,719.66 | 0.062 | 0.234 |
| `DirectoSinClima_espaciotemporal_28d` | `HPO_B8` | 8 | 2 | 2,363,481.63 | 0.681 | — |

*(Nota: en `DirectoSinClima_espaciotemporal_28d` el objetivo `y_H` es energía acumulada de 28 días, de ahí la escala ~1.6M kWh; los subconjuntos `HPO_B8` con 1–2 unidades son únicamente pruebas de integración E2E, no comparaciones confirmatorias.)*

## 2. Prueba Aislada del Estado B del Notebook (`estado_b_fixture_resumen.json`)

- Fixture sintético etiquetado `SINTETICO_TEST` en `resultados/preflight_ashr/test_estado_b_notebook/`:
  - Rechazo estricto con `allow_synthetic_test=False`: `INVALID`.
  - Carga completa en Estado B (`allow_synthetic_test=True`): **28 escenarios**, **28 filas `tabla_mcs`**, **10 archivos de tablas (`CSV` + `LaTeX`)**, **5 figuras (`PNG` + `PDF`)**.

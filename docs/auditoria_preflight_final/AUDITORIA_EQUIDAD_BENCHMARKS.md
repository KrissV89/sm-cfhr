# Auditoría de Equidad Experimental frente a los 22 Benchmarks Oficiales

**Fecha:** 2026-11-23

---

## 1. Auditoría de Ajuste de los 22 Benchmarks Oficiales

Inspección directa de `scripts/evaluacion/produccion_intraplanta.py`, `produccion_multihorizonte.py`, `produccion_multihorizonte_clima.py` y `produccion_multihorizonte_directo.py`:

| Familia / Modelo | Cantidad | Búsqueda / Grid en los 28 escenarios | Evaluaciones $\times$ 3 splits | Clasificación frente a `SM-CFHR-Fixed` ($B=0$) | Clasificación frente a `SM-CFHR-HPO` ($B=25$ TPE) |
|---|---|---|---|---|---|
| `XGBoost`, `Random Forest`, `Ridge`, `SVR`, `HistGradientBoosting`, `k-NN` | 6 librerías | `GRIDS`: **4 configuraciones** por modelo | $4 \times 3 = 12$ fits | `COMPARABLE` (ventaja leve para bibliotecas: 4 configs vs 0) | `PARCIALMENTE COMPARABLE` (ventaja de presupuesto para ASHR-HPO: 25 TPE vs 4 grid) |
| `Lasso`, `BayesianRidge`, `DecisionTree` | 3 librerías | Configuración fija por defecto en scripts de 28 escenarios | 1 fit (sin HPO interna en 28 escenarios) | `COMPARABLE` | `NO COMPARABLE` en igualdad estricta de HPO (25 TPE vs 0) |
| `ABGPR`, `ELAC`, `HARC`, `IWLAR`, `KLAC`, `MARB`, `PDEC`, `RARC`, `RLAC`, `RQRF`, `SRMOS`, `WCFC` | 12 custom | `GRIDS`: **4 configuraciones** por modelo | $4 \times 3 = 12$ fits | `COMPARABLE` | `PARCIALMENTE COMPARABLE` (25 TPE vs 4 grid) |
| `Media móvil de 30 días` | 1 baseline | Sin parámetros aprendidos | 0 fits | `COMPARABLE` (línea base causal) | `COMPARABLE` (línea base causal) |

---

## 2. Resolución Metodológica de la Asimetría

1. **Comparación principal científicamente defendible:** **`SM-CFHR-Fixed` ($B=0$ evaluaciones de HPO) vs. los 22 benchmarks oficiales**. En esta comparación `SM-CFHR` **no** tiene ventaja de presupuesto (usa parámetros congelados `delta=1.35, alpha_lin=0.01, alpha_meteo=0.10`, frente a rejillas de 4 puntos en 18 de los 22 benchmarks).
2. **Comparación complementaria (`SM-CFHR-HPO`, $B=25$ TPE):** Se ejecuta y reporta por separado para cuantificar la ganancia real de la optimización bayesiana en CV anidada, **declarando explícitamente** la asimetría frente a las rejillas de 4 puntos y los 3 modelos sin grid.
3. **Protocolo si la rúbrica exigiera igualdad estricta de presupuesto para HPO:** Reoptimizar las 9 librerías con `Optuna TPESampler(n_trials=25, seed=2026)` sobre los mismos `3` splits internos sin modificar los CSVs históricos (guardando en directorio separado). Coste estimado: $9\times$ el tiempo de `SM-CFHR-HPO` (~90 h secuenciales para los 28 escenarios, por lo que no se ejecuta en el preflight).

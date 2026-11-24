# Auditoría de Benchmarks Oficiales y Equidad de Presupuestos

**Fecha:** 2026-11-23  
**Alcance:** 28 escenarios solares oficiales (17 plantas fotovoltaicas de Colombia).

---

## 1. Inventario Exacto de los 22 Modelos Oficiales de Referencia

Verificado programáticamente mediante `verify_benchmark_inventory()` en `src/ashr_oof_pipeline/config.py` sobre los 28 archivos oficiales de `resultados/benchmark_*`:

- **9 modelos de librería:**
  1. `BayesianRidge`
  2. `DecisionTree`
  3. `HistGradientBoosting`
  4. `Lasso`
  5. `Random Forest`
  6. `Ridge`
  7. `SVR`
  8. `XGBoost`
  9. `k-NN`
- **12 modelos propios oficiales previos:**
  1. `ABGPR` (Proceso gaussiano sobre residuos)
  2. `ELAC` (Corrección residual sobre nivel exponencial)
  3. `HARC` (Efectos penalizados de grupo y sitio)
  4. `IWLAR` (Regresión local ponderada de residuos)
  5. `KLAC` (Corrección residual sobre nivel local de Kalman)
  6. `MARB` (Boosting residual monótono en anomalía de radiación)
  7. `PDEC` (Corrección por desviación del nivel previo)
  8. `RARC` (Corrección residual robusta Huber)
  9. `RLAC` (Corrección Ridge del residuo sobre nivel móvil)
  10. `RQRF` (Bosque cuantílico de residuos)
  11. `SRMOS` (MOS con ventana temporal de entrenamiento)
  12. `WCFC` (Mezcla temporal de correcciones Ridge–Huber)
- **1 baseline causal sin aprendizaje:**
  1. `Media móvil de 30 días`
- **Exclusión explícita:** `CF-BRH` permanece **excluido** de los rankings oficiales (22 referencias + `SM-CFHR` = **23 modelos**).

---

## 2. Auditoría de Ajuste y Presupuestos de los Benchmarks Históricos

Inspeccionados `scripts/evaluacion/produccion_intraplanta.py`, `produccion_multihorizonte.py`, `produccion_multihorizonte_clima.py`, `produccion_multihorizonte_directo.py` y `resultados/optimizadores/PROTOCOLO.md`:

1. **Particiones y causalidad común:** Todos los benchmarks en los 28 escenarios comparten exactamente los mismos bloques externos (`D1`, `D2`, `D3`), las mismas columnas causales disponibles al origen (`kwh_D`, `media7_kwh`, `media30_kwh`, `cap_MW_D`, variables cíclicas de calendario y pronósticos `D+1` o `GEFSv12` cuando `weather_on=True`) y las mismas métricas (`MAE_macro` por planta en `kWh/día`, `RMSE`, `MBE`).
2. **Asimetría de presupuesto detectada:**
   - En el estudio específico `resultados/optimizadores/PROTOCOLO.md` (cuaderno 21, solo `D+1` sin `SM-CFHR`), se igualó el presupuesto a **30 evaluaciones** por optimizador (`Grid`, `Random`, `Optuna TPE`, `DEAP Genético`).
   - En cambio, en los **28 escenarios unificados** (`benchmark_produccion_definitivo`, `benchmark_multihorizonte*`), varios benchmarks de librería y custom utilizaron configuraciones fijas o rejillas internas pequeñas (p. ej. 3–9 puntos por planta/bloque), sin búsqueda bayesiana continua de 25–30 evaluaciones.
3. **Política de Equidad y Doble Reporte (para evitar sesgo de presupuesto):**
   - Si se comparara únicamente `SM-CFHR-HPO` (con 25 evaluaciones TPE) contra benchmarks históricos de rejilla fija, existiría una ventaja de presupuesto de búsqueda a favor de `SM-CFHR-HPO`.
   - **Solución adoptada en el protocolo definitivo (`experiments/protocolo.md`):**
     1. **Reportar siempre `SM-CFHR-Fixed` (presupuesto = 0 evaluaciones de HPO)** con hiperparámetros congelados (`delta=1.35, alpha_lin=0.01, alpha_meteo=0.10`), como comparación conservadora de mínimo costo computacional.
     2. **Reportar en paralelo `SM-CFHR-HPO` (presupuesto = 25 evaluaciones TPE dentro de CV causal anidada)** para demostrar el desempeño con optimización continua sin fuga y cuantificar la ganancia real atribuible a la HPO frente a la configuración fija.
     3. En los análisis estadísticos confirmatorios (`DM-HLN-HAC`, `GW`, `MCS`), distinguir claramente ambos reportes para que ningún revisor pueda objetar asimetría de presupuesto.

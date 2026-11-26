# Protocolo de Optimización de Hiperparámetros y Validación Cruzada Anidada (`SM-CFHR`)

**Módulo:** `src/ashr_oof_pipeline/optimizacion_hiperparametros.py`  
**Fecha:** 2026-11-23

---

## 1. Arquitectura Fija vs. Hiperparámetros Optimizables

La arquitectura estructural de `SM-CFHR` permanece **fija** (no se reactiva el selector multimodo):
- `selection = "fixed"`
- `n_knots = 0`, `max_interactions = 0`
- `anchor = "none"`, `lead_damp = 1.0`
- `structured = True`, `use_mono = True`
- `max_iter = 400`, `tol = 1e-7`

La optimización bayesiana actúa **únicamente** sobre los tres hiperparámetros continuos del núcleo Huber estructurado:

| Hiperparámetro | Escala | Dominio | Valor congelado (`B=0`) | Justificación |
|---|---|---|---|---|
| `delta` | Uniforme (`float`) | `[0.5, 3.0]` | `1.35` | Controla la transición cuadrática–lineal en unidades de MAD residual; `1.35` corresponde al 95% de eficiencia gaussiana, `[0.5, 3.0]` cubre desde alta resistencia a outliers hasta cuasi-L2. |
| `alpha_lin` | Log-uniforme (`float`) | `[1e-4, 1.0]` | `0.01` | Regularización $L_2$ para variables base (calendario, `cf30`, diferencias temporales); evita sobreajuste manteniendo señal de persistencia. |
| `alpha_meteo` | Log-uniforme (`float`) | `[1e-3, 10.0]` | `0.10` | Regularización $L_2$ diferenciada para variables NWP y anomalías meteorológicas cuando `weather_on=True`. Cuando `weather_on=False`, se fija igual a `alpha_lin` para no desperdiciar evaluaciones. |

---

## 2. Optimizador y Presupuesto

- **Algoritmo:** `optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(8, max(2, n_trials // 3)), multivariate=True)`.
- **Presupuesto por pliegue externo:** $B = 25$ evaluaciones (configurable hasta 30; comparado siempre contra `SM-CFHR-Fixed` con $B = 0$).
- **Función objetivo interna:** `MAE_macro_kWh` promediado sobre las particiones causales internas.
- **Trazabilidad:** Cada evaluación registra `outer_fold`, `trial_number`, `delta`, `alpha_lin`, `alpha_meteo`, `inner_mae_macro_kwh`, `n_inner_splits`, `seed`, `wall_time_s`, `peak_memory_kb` y `best_so_far_kwh`.

---

## 3. Esquema de Validación Cruzada Causal Anidada

Dentro de cada conjunto de entrenamiento externo `train_outer` (`test_outer` nunca es tocado):
1. **D+1 (`mode_eval="d1"`):** Ventana temporal creciente con embargo causal (`embargo_days=30` cuando el historial lo permite, mínimo 1 día), garantizando $\max(\text{date}_{\text{train}}) < \min(\text{date}_{\text{val}})$.
2. **Directo `7–28d` (`mode_eval="direct"`):** Regla causal estricta $\text{target\_end\_date}_{\text{train}} \le \min(\text{origin}_{\text{val}})$, impidiendo que la ventana objetivo de entrenamiento se solape con el origen de validación.
3. **Recursivo `7–28d` (`mode_eval="recursive"`):** Regla causal estricta $\text{origin}_{\text{train}} + H \le \min(\text{origin}_{\text{val}})$. En `v1.0.3` (`HPO_PROTOCOL_VERSION = "v1.0.3-exact-recursive-state"`), cada trial interno ejecuta exactamente `rollout_recursive_from_origin` con `PlantRecursiveState` / `PlantRecursiveWeatherState`, ventanas deslizantes reales de 7 y 30 días, y realimentación exclusivamente con predicciones propias. Los estudios SQLite generados con `v1.0.2` son rechazados por incompatibilidad de protocolo.
4. **Espaciotemporal (`task="espaciotemporal"`):** Además del corte temporal causal, cuando existen $\ge 2$ grupos espaciales (`grupo`), cada corte interno retiene un grupo espacial completo en validación y lo excluye del entrenamiento.
5. **Preprocesamiento sin fuga:** La imputación por mediana (`nanmedian`) y la estandarización robusta (`MAD`) se calculan exclusivamente sobre `X_train_inner` en cada pliegue interno, y posteriormente sobre todo `train_outer` en el reajuste final.

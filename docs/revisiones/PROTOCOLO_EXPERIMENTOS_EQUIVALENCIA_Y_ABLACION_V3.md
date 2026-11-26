# Protocolo Experimental de Equivalencia de Representación y Ablaciones Controladas (`v3.0.0`)

## 1. Propósito y preservación de predicciones congeladas
El protocolo `v3.0.0` responde experimentalmente si la ventaja observada de SM-CFHR en la evaluación OOF congelada (`v1.0.3`) proviene de su formulación algorítmica o de la normalización causal por capacidad (`24.000 * C_{i,t}`). **Ninguna predicción OOF de `resultados/evaluacion_ashr_oof/` fue modificada ni regenerada** (integridad verificada en `57/57` hashes SHA-256).

## 2. Experimento 1 — Equivalencia de Representación (`experimento_equivalencia_v3`)
Implementado en `src/sm_cfhr/representation_benchmark.py` y ejecutado mediante `scripts/evaluacion/ejecutar_equivalencia_representacion_v3.py`.

- **Familias evaluadas (7):** `SM-CFHR`, `Ridge`, `BayesianRidge`, `HuberRegressor`, `ElasticNet`, `RandomForest`, `XGBoost` (+ control causal `MM30`).
- **Representaciones comparadas (2 por familia = 14 variantes + `MM30`):**
  1. `kWh_bruto`: predictores de nivel histórico y objetivo en `kWh` brutos, con proyección física final a `[0, 24.000 * H_dir * C_{i,t}]`.
  2. `CF`: exactamente la misma representación causal adimensional en factor de capacidad (`media30_kwh / (24.000 * C_{i,t})`, `cf7 - cf30`, `cf1 - cf7`, objetivo `y / (24.000 * H_dir * C_{i,t})`), imputación por mediana y estandarización en entrenamiento, y reconstrucción física acotada idéntica.
- **Condición A (configuraciones fijas, 10 escenarios):**
  - `D1_espaciotemporal_cinco_variables`, `D1_intraplanta_cinco_variables`, `D1_espaciotemporal_sin_meteorologia`, `D1_intraplanta_sin_meteorologia`, `RecGEFSv12_espaciotemporal_7d`, `RecGEFSv12_espaciotemporal_28d`, `RecGEFSv12_intraplanta_28d`, `RecSinClima_espaciotemporal_28d`, `DirectoSinClima_espaciotemporal_28d`, `DirectoSinClima_intraplanta_28d`.
  - Inferencia pareada `SM-CFHR__CF` vs cada rival `__CF` mediante serie macro-alineada `z_t^macro`, DM-HLN-HAC (`h_serie=1`) y corrección secuencial de Holm-Bonferroni (`inferencia_pareada_smcfhr_vs_cf_baselines.csv`).
- **Condición B (validación causal interna de 3 pliegues con presupuesto equiparado, 6 escenarios):**
  - Cada familia evalúa una rejilla acotada de 4 configuraciones sobre 3 particiones causales internas (embargo de 30 días en `D+1`, `t_end <= t_val` en multihorizonte, exclusión espacial en ET).
  - Escenarios: `D1_espaciotemporal_cinco_variables`, `D1_intraplanta_cinco_variables`, `D1_espaciotemporal_sin_meteorologia`, `D1_intraplanta_sin_meteorologia`, `RecGEFSv12_espaciotemporal_28d`, `DirectoSinClima_intraplanta_28d`.

## 3. Experimento 2 — Ablaciones Controladas, Auditorías y Sesgo de Selección (`ablaciones_controladas_v3`)
Implementado en `scripts/evaluacion/ejecutar_ablaciones_y_sesgo_v3.py`.

1. **Ablación factorial en 6 escenarios:** compara `SM_CFHR_completo` frente a `sin_CF_energia_bruta`, `sin_Huber_L2` (`delta=1e6`), `sin_CF_y_sin_Huber_L2`, `sin_monotonia`, `escala_STD_en_target`, `sin_L2_anisotropica_001`, `sin_L2_anisotropica_010`, `sin_anomalias_meteo` y `sin_meteorologia` (cuando aplica clima). Reporta `MAE_CF_macro`, `delta_CF_vs_completo`, IC 95% bootstrap estacionario y `p_dm_CF`.
2. **Auditoría de activación de monotonía:** registra por cada pliegue de los 6 escenarios los pesos no restringidos `w_cf30_sin` y `w_rad_sin` y la fracción exacta de pliegues donde `w_con <= 1e-9` y `w_sin < -1e-9` (`auditoria_monotonia_activacion.csv`).
3. **Auditoría de Huber operacional vs estrés sintético y apalancamiento:** evalúa `SM_CFHR_completo` vs `sin_Huber_L2` en `D1_espaciotemporal_cinco_variables` (`Operacional_completo`, `Episodios_bajo_rendimiento_sy<0.3`, `Cola_superior_5pct_error`) y en corte cronológico limpio/contaminado (`Estres_0pct_limpio`, `Contaminacion_target_5pct`, `Contaminacion_target_10pct`, `Apalancamiento_covariable_rad_5pct`).
4. **Sensibilidad al sesgo de selección `horas_xm <= 6`:** reincorpora en evaluación `D+1` (ET e IP) las jornadas observadas con `horas_xm <= 6` sin imputación de objetivo (`sensibilidad_sesgo_seleccion_horas_xm.csv`).

## 4. Entorno y Manifiestos
- Entorno Python: `solarpv-eda` (Python 3.11.15, NumPy 2.3.5, SciPy 1.16.3, pandas 2.3.3, scikit-learn 1.9.0).
- Manifiestos SHA-256 generados: `resultados/experimento_equivalencia_v3/MANIFIESTO_EQUIVALENCIA_V3.csv` y `resultados/ablaciones_controladas_v3/MANIFIESTO_ABLACIONES_V3.csv`.

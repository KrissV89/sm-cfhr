# Auditoría de Alineación, Unidades y Métricas (28 Escenarios)

1. **Claves exactas de alineación (`SCENARIOS_28` en `src/ashr_oof_pipeline/config.py`):**
   - `D+1` (4 escenarios): `("code", "date")`, unidad `kWh/dia`, columnas `prediccion_kwh` vs `y_kwh`.
   - `Recursivo sin clima` y `Recursivo + GEFSv12` (16 escenarios): `("code", "origen", "lead_dias")`, unidad `kWh/dia`, columnas `prediccion_kwh` vs `y_kwh`.
   - `Directo sin clima` (8 escenarios): `("code", "origen")`, unidad `kWh_total` (energía acumulada en `H` días), columnas `prediccion_total_kwh` vs `y_total_kwh`.
2. **Conversión física y ausencia de desajuste CF vs kWh:**
   - El regresor opera internamente en factor de capacidad acotado $c_i = \operatorname{clip}(y_i / (B_i H_{\text{dir}}), 0, 1)$ con $B_i = \text{cap\_MW\_D} \cdot 24{,}000$.
   - Al predecir, reconstruye $\hat{y}_i = \operatorname{clip}(\hat{c}_i, 0, 1) \cdot B_i H_{\text{dir}}$ en `kWh` (diarios o acumulados), y tanto `optimize_hyperparams_tpe` como la evaluación oficial optimizan y reportan exactamente **`MAE_macro_kWh`** ($\frac{1}{P}\sum_p \mathrm{MAE}_p$ en `kWh`).
3. **Validación estricta (`validate_predictions_df` + `align_predictions`):**
   - Rechaza columnas faltantes, NaNs, valores no finitos, predicciones negativas (`< -1e-9`) y claves duplicadas.
   - `align_predictions` realiza *inner merge* sobre `join_keys` exactas, garantizando que `SM-CFHR-Fixed`, `SM-CFHR-HPO` y los 22 benchmarks se evalúen exactamente sobre las mismas filas externas.

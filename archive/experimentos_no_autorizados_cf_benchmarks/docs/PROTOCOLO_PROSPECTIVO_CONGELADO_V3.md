# Protocolo Prospectivo Congelado (`v3.0.0`)

## 1. Justificación
Tanto la evaluación OOF congelada (`v1.0.3`), la reconciliación inferencial (`v2.0.0`) como los experimentos de equivalencia y ablación (`v3.0.0`) reutilizan el panel histórico enero de 2024 – octubre de 2026 (`17` centrales colombianas, `G=7` zonas espaciales). Para obtener evidencia confirmatoria libre de reutilización adaptativa, cualquier evaluación futura sobre datos posteriores a octubre de 2026 o sobre flotas externas debe seguir este protocolo preespecificado e inmutable.

## 2. Reglas de Congelación Pre-Evaluación
1. **Modelos y configuraciones congeladas:**
   - Primario propuesto: `SM-CFHR-Fixed` (`delta = 1.35`, `alpha_lin = 0.01`, `alpha_meteo = 0.10`, `B = 0`).
   - Pares obligatorios en representación causal en factor de capacidad (`CF`): `BayesianRidge-CF`, `ElasticNet-CF`, `HuberRegressor-CF`, `Ridge-CF`, `RandomForest-CF`, `XGBoost-CF` y persistencia `MM30`.
   - Prohibido ajustar hiperparámetros o umbrales tras observar la partición prospectiva.
2. **Estimandos primarios preespecificados:**
   - Métrica primaria adimensional: `MAE_CF_macro` (promedio macro por planta usando exclusivamente capacidad vigente al origen `24.000 * H_dir * C_{i,t}`).
   - Métrica secundaria física: `MAE_macro_kWh` y `MAE_macro_kWh_por_dia`.
   - Criterio de inferencia: serie macro-alineada `z_t^macro` con `h_serie = 1` en bloques disjuntos, prueba DM-HLN-HAC bilateral ajustada por Holm-Bonferroni (`p_Holm < 0.05`) y tamaño mínimo de bloque `T_eff,macro >= 30`.
3. **Separación estricta de regímenes espaciales:**
   - **ET con historial local:** central o grupo espacial excluido del entrenamiento paramétrico, pero con historial previo causal propio (`>= 30` días) para construir `cf30`, `cf7`, `cf1` y anomalías NWP.
   - **Cold-start / zero-shot puro:** central completamente nueva sin historial propio (`cf30` sustituido por climatología regional o NWP puro), reportado siempre por separado.
4. **Tratamiento de sesgo operativo e incertidumbre:**
   - Reportar métricas por separado con el filtro `horas_xm > 6` y sobre el total de jornadas con telemetría válida observada (`sin imputar objetivos`).
   - Para intervalos predictivos bajo no estacionariedad, emplear predicción conforme adaptativa o ponderada temporalmente en lugar de cuantiles estáticos intercambiables.
5. **Alcance y autorización:**
   - Se mantiene `alcance_solar_exclusivo: sin datasets externos`; toda extensión fuera del dominio fotovoltaico requiere pre-registro independiente.

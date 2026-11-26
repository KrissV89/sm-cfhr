# 02. Matriz de Trazabilidad: Literatura Científica vs. Proyecto SM-CFHR

**Proyecto:** SM-CFHR (`sm-cfhr v1.0.3`)  
**Criterio de clasificación:**
- **Cumplida:** existe implementación verificable y correcta en el código y artefactos.
- **Parcial:** está implementada con limitaciones metodológicas o de alcance.
- **Incumplida:** debería aplicarse según la literatura y no se aplica o contiene errores de implementación.
- **Discutible:** existen alternativas metodológicas defendibles con compromisos distintos.
- **No aplicable:** no corresponde al diseño diario/agregado o alcance del estudio.
- **No verificable:** falta evidencia suficiente o no se ha ejecutado/inspeccionado el insumo completo.

---

## 1. Matriz general de trazabilidad

| ID | Recomendación | Artículo y página | Implementación actual (archivo:líneas) | Estado | Riesgo | Acción |
|---|---|---|---|---|---|---|
| **M01** | Separación estricta desarrollo vs. evaluación confirmatoria y sin *peeking* | Tashman (2000), p. 438; Linca et al. (2026), p. 9 | `experiments/protocolo.md:10-16`; `experiments/config_ashr_congelada.json:1-80` | **Cumplida** | Bajo | Mantener congelado |
| **M02** | Validación causal temporal/espaciotemporal (`tsCV` / `PtBsR_grW`) sin entrenar con futuro | Oliveira et al. (2021), pp. 21–22, 24; Hewamalage et al. (2023), pp. 807–812 | `src/ashr_oof_pipeline/optimizacion_hiperparametros.py`; `resultados/evaluacion_ashr_oof/particiones/nested_cv_audit.csv` | **Cumplida** | Bajo | Mantener |
| **M03** | Comparar contra benchmarks simples (persistencia / MM30) y múltiples familias | Hewamalage et al. (2023), pp. 795–800; Linca et al. (2026), p. 3 | `src/ashr_oof_pipeline/resultados_loader.py:27-34, 152-160` (22 benchmarks + MM30) | **Cumplida** | Bajo | Mantener; añadir climatología-persistencia convexa |
| **M04** | Presupuesto de HPO equivalente entre modelo propuesto y competidores | Linca et al. (2026), p. 9, Tabla 1 (p. 3) | `experiments/protocolo.md:27-31` (`SM-CFHR-Fixed` $B=0$, `SM-CFHR-HPO` $B=25$) | **Parcial** | Medio | Para comparación justa con bibliotecas usar `Fixed` ($B=0$) o documentar asimetría de `HPO` |
| **M05** | Consistencia entre función de pérdida de entrenamiento y métrica de evaluación | Hewamalage et al. (2023), pp. 790, 813; Yang et al. (2020), pp. 22–23 | `config_ashr_congelada.json` (Huber en CF) vs. `evaluacion_estadistica.py:33-34` (`MAE_macro` + `RMSE`) | **Discutible** | Medio | Reportar ambas (`MAE` y `RMSE`) sin declarar ventaja solo por asimetría L1/L2 |
| **M06** | Evitar métricas dependientes de escala (`kWh`) al promediar entre series heterogéneas | Tashman (2000), p. 445; Hewamalage et al. (2023), p. 802; Yang et al. (2020), p. 25 | `evaluacion_estadistica.py:33` (`MAE_macro` en `kWh/día` y `kWh_total` en 17 plantas) | **Incumplida** | **Alto** | Añadir `nMAE` / `nRMSE` en `CF` (0–1) como métrica primaria libre de escala |
| **M07** | Reportar `RMSE skill score` contra combinación convexa óptima climatología–persistencia | Yang et al. (2020), pp. 26, 30–31 (Ecs. 4, 17–18), p. 33 | `resultados_loader.py:182` (solo `Skill_vs_MM30_pct` en MAE) | **Incumplida** | Medio | Calcular post-OOF sin reentrenar sobre predicciones guardadas |
| **M08** | Verificación orientada a distribuciones (Murphy–Winkler: sesgo-varianza, ECDF, densidad conjunta) | Yang et al. (2020), pp. 27–30 (Ecs. 12–14, Tabla 3) | `evaluacion_estadistica.py:35` (solo MBE escalar) y figuras de dispersión/error básicas | **Parcial** | Medio | Añadir descomposición MSE Murphy–Winkler y ECDF post-OOF |
| **M09** | Coherencia entre paso temporal de la serie de diferencias $d_t$ y horizonte $h$ en DM-HLN | Harvey et al. (1997); Diebold & Mariano (1995) | `resultados_loader.py:205-206, 236-241` + `inferencia_estadistica_v2.py:119` (`horizon_h=H` en bloques disjuntos de $H$ días) | **Incumplida** | **Crítico (P0)** | Corregir a `horizon_h=1` cuando `origen` está espaciado cada $H$ días |
| **M10** | Umbral de longitud efectiva $T_{\text{eff}}$ en unidades coherentes con la serie | Diagnóstico de tamaño muestral en `evaluacion_estadistica.py` | `evaluacion_estadistica.py:156` (`t_eff < 2.0 * horizon_h` mezcla bloques de $H$ días con $2H$ días) | **Incumplida** | **Crítico (P0)** | Corregir unidades del umbral (`T_eff` en bloques vs. $H$) |
| **M11** | No seleccionar al mejor competidor *ex post* sobre la misma muestra de prueba DM | Tashman (2000), pp. 446–447 (Fildes 1989) | `resultados_loader.py:156, 230` (`best_lib_n` elegido *ex post* por escenario) | **Incumplida** | **Crítico (P0)** | Usar benchmark pre-especificado o contraste múltiple contra todas las bibliotecas + SPA/MCS |
| **M12** | Implementación unilateral de $T_{\max}$ en MCS y número suficiente de réplicas $B$ | Hansen et al. (2011) [ref. código]; `experiments/protocolo.md:41` | `evaluacion_estadistica.py:118-119` (compara `max` vs. `abs` bilateral) y `resultados_loader.py:215` (`B=200`, `mean_block=7`) | **Incumplida** | **Crítico (P0)** | Usar `mcs_stationary_v2` (unilateral) con `B>=2000` y `mean_block` escalado |
| **M13** | Implementación consistente de SPA (usar error estándar bootstrap/HAC) | Hansen (2005) [ref. código] | `src/sm_cfhr/inference.py:117-122` (calcula `se` en L117 pero divide por `std(d)` i.i.d. en L118 y L122) | **Incumplida** | **Alto (P1)** | Corregir `spa_test` para usar varianza HAC/bootstrap |
| **M14** | Corrección por multiplicidad coherente por familia (Holm) | Holm (1979); Hewamalage et al. (2023), p. 824; `protocolo.md:42` | `resultados_loader.py:274` (Holm global 56 filas) vs. `pipeline_stages.py:56-76` (Holm por familia) | **Parcial** | **Alto (P1)** | Alinear `evaluate_r3_superiority` con familias pre-especificadas sin diluir con filas inválidas |
| **M15** | Panel balanceado / ponderación coherente al construir $d_t$ entre plantas | Tashman (2000), p. 445; Hewamalage et al. (2023), p. 802 | `resultados_loader.py:206` (`plants/t` varía de 1 a 17 en `Rec*`/`Directo*`) | **Incumplida** | **Alto (P1)** | Ponderar por capacidad/planta o reportar inferencia sobre panel común / nivel planta |
| **M16** | No tratar escenarios que comparten plantas/fechas como experimentos independientes | Oliveira et al. (2021), pp. 2–4, 19 | `resultados_loader.py:274` + `veredicto_r3.csv` (28 escenarios con mismas 17 plantas y fechas solapadas) | **Parcial** | **Alto (P1)** | Tratar los 28 escenarios como estructura dependiente por régimen/horizonte |
| **M17** | Transparencia ante resultados no significativos sin relajar criterios *post-hoc* | Linca et al. (2026), p. 23; Hewamalage et al. (2023), p. 826 | `veredicto_r3.csv` y `evaluacion_cientifica_ashr_oof.ipynb` | **Cumplida** | Bajo | Mantener rigor conservador tras corregir bugs P0/P1 |
| **M18** | Cobertura de al menos un ciclo anual completo en evaluación | Linca et al. (2026), pp. 9, 23–24 | `D1_*` cubre `2025-11-01..2026-07-31` (262 días, 9 meses; faltan ago–oct) | **Parcial** | Medio | Documentar limitación estacional de `D+1` |
| **M19** | Intervalos predictivos condicionados a estación y horizonte | Linca et al. (2026), p. 23 | `src/sm_cfhr/pipeline_stages.py:217-224` (conformal solo en D+1, sin desagregación estacional/horizonte largo) | **Parcial** | Medio | Extender diagnóstico conformal por horizonte y estación |
| **M20** | Exclusión de horas nocturnas ($\theta_z < 85^\circ$ / elevación $<5^\circ$) | Yang et al. (2020), p. 32; Linca et al. (2026), p. 7 | Agregado diario en `kWh/día` (`SCENARIOS_28`) | **No aplicable** | N/A | N/A en resolución diaria |

---

## 2. Fichas detalladas de trazabilidad por hallazgo crítico

### Ficha 1 (P0) — Factor HLN aplicado con `horizon_h = H` sobre serie ya espaciada cada $H$ días
1. **Referencia y página:** Harvey, Leybourne & Newbold (1997), *IJF*, 13(2), 281–291 (`InvML/.../Harvey_Leybourne_Newbold_1997_Testing_Equality_Prediction_MSE.pdf`); Diebold & Mariano (1995), *JBES*, 13(3), 253–263.
2. **Paráfrasis fiel:** La corrección HLN ajusta la varianza de muestra finita del estadístico DM cuando la serie de diferencias de pérdida $d_t$ proviene de pronósticos multi-paso **emitidos en cada paso de la serie** (estructura MA($h-1$) por solapamiento de $h-1$ periodos entre $t$ y $t+1$).
3. **Archivo y función:** `src/ashr_oof_pipeline/resultados_loader.py` (`load_oof_bundle`, líneas 205–206 y 236–241); `scripts/evaluacion/inferencia_estadistica_v2.py` (`dm_hln_hac`, línea 119).
4. **Evidencia concreta:** En `resultados/evaluacion_ashr_oof/predicciones/RecSinClima_espaciotemporal_28d.csv`, `origen` avanza en saltos exactos de `28` días (`diffs = {28: 30}`, $T=31$ orígenes), por lo que las ventanas de pronóstico $[t+1, t+28]$ y $[t+29, t+56]$ **no se solapan**. Sin embargo, `dm_hln_hac` recibe `horizon_h=28`, reduciendo el estadístico por un factor `hln = 0.1117` (casi $9\times$ menor).
5. **Consecuencia científica:** Colapsa artificialmente la potencia en $H=21$ y $H=28$, ocultando tanto victorias reales (`RecGEFSv12_espaciotemporal_28d`: $p=0.5968 \to 0.0001$) como derrotas reales (`DirectoSinClima_intraplanta_28d`: $p=0.7492 \to 0.0302$).
6. **Corrección propuesta:** Pasar `horizon_h=1` cuando la distancia entre orígenes consecutivos en `time_col` sea $\ge H$ días, o bien conservar orígenes diarios ($d_t$ diario solapado) si se desea usar $h = H$ con HAC de ancho de banda $\ge H-1$.

### Ficha 2 (P0) — Umbral `T_eff < 2 * H` en unidades incompatibles
1. **Referencia y página:** Diagnóstico interno de diseño efectivo (`src/sm_cfhr/validation.py:41-48`; `src/ashr_oof_pipeline/evaluacion_estadistica.py:156`).
2. **Paráfrasis fiel:** La regla pretende exigir al menos $2H$ observaciones efectivas antes de confiar en la aproximación asintótica de DM-HLN-HAC.
3. **Archivo y función:** `src/ashr_oof_pipeline/evaluacion_estadistica.py` (`run_paired_inference`, línea 156); `resultados/evaluacion_ashr_oof/inferencia/diseno_efectivo.csv`.
4. **Evidencia concreta:** Para `H=28`, existen $T=31$ orígenes no solapados ($T_{\text{eff}} \le 31$), pero la línea 156 evalúa `t_eff < 2.0 * 28 = 56`. Es algebraicamente imposible que cualquier serie de $31$ bloques alcance $56$. Esto invalida `13/28` escenarios por construcción.
5. **Consecuencia científica:** Descarta artificialmente toda la evidencia inferencial en $H=21$ y $H=28$, reduciendo el denominador de R3 de 28 a 15 escenarios.
6. **Corrección propuesta:** Expresar $T_{\text{eff}}$ y el umbral mínimo en la **misma unidad** (número de bloques efectivos $T_{\text{eff,bloques}} \ge 30$ para aproximación $t$/normal, o $T_{\text{eff,días}} \ge 2H$ si se mide en días).

### Ficha 3 (P0) — Selección retrospectiva (*ex post*) de `Mejor_Bib`
1. **Referencia y página:** Tashman (2000), pp. 446–447 (PDF pp. 10–11, citando a Fildes, 1989); Hewamalage et al. (2023), p. 824 (PDF p. 37).
2. **Paráfrasis fiel:** Identificar al mejor competidor *ex post* en la muestra de prueba y evaluarlo luego con una prueba de dos muestras como si hubiera sido elegido *ex ante* introduce sesgo de selección múltiple.
3. **Archivo y función:** `src/ashr_oof_pipeline/resultados_loader.py` (`load_oof_bundle`, líneas 156 y 230).
4. **Evidencia concreta:** `best_lib_n` se selecciona como el mínimo empírico de 9 modelos de biblioteca en cada escenario (cambiando entre `Lasso`, `BayesianRidge`, `Ridge`, `SVR` y `k-NN` según escenario en `tabla_inferencia.csv`) y se pasa a DM-HLN sin ajustar por la selección sobre 9 candidatos.
5. **Consecuencia científica:** Penaliza de forma asimétrica a SM-CFHR frente a un "oráculo retrospectivo" específico de cada escenario.
6. **Corrección propuesta:** Mantener un benchmark pre-especificado fijo (p. ej. combinación climatología–persistencia y/o `Ridge`/`Lasso` fijado *a priori*), o aplicar pruebas conjuntas diseñadas para múltiples modelos (SPA y MCS corregidos).

### Ficha 4 (P0) — Implementación bilateral en `model_confidence_set`
1. **Referencia y página:** `scripts/evaluacion/inferencia_estadistica_v2.py:251-252` (cita Ec. 12 unilateral de Hansen et al., 2011; sin PDF local leído).
2. **Paráfrasis fiel:** El estadístico $T_{\max}$ del MCS evalúa la desviación máxima unilateral positiva (pérdida relativa superior al promedio) bajo la hipótesis nula de igual capacidad predictiva.
3. **Archivo y función:** `src/ashr_oof_pipeline/evaluacion_estadistica.py` (`model_confidence_set`, líneas 118–119) vs. `src/ashr_oof_pipeline/resultados_loader.py` (línea 214).
4. **Evidencia concreta:** Mientras `inferencia_estadistica_v2.py:252` usa `np.max(centered / se_i)`, el pipeline oficial en `evaluacion_estadistica.py:118` usa `t_boot = np.abs(centered_boot) / se` frente a `stat_obs = np.max(t_obs)`, y `resultados_loader.py:215` corre solo `B = 200` réplicas (en vez de `B = 500` de `protocolo.md:41`). Resultado en `tabla_mcs.csv`: 15–22 modelos sobreviven simultáneamente en casi todos los escenarios multihorizonte.
5. **Consecuencia científica:** Infla los p-valores de eliminación del MCS y vuelve al procedimiento incapaz de discriminar modelos inferiores.
6. **Corrección propuesta:** Unificar `evaluacion_estadistica.py` con `mcs_stationary_v2` (unilateral, piso `1/(B+1)`, `B >= 2000`, bloque estacionario coherente con la frecuencia de `time_col`).

### Ficha 5 (P1) — `spa_test` ignora dependencia serial y `MAE_macro` en `kWh` mezcla escalas
1. **Referencia y página:** Tashman (2000), p. 445; Hewamalage et al. (2023), p. 802; Yang et al. (2020), p. 25, pp. 30–31; Linca et al. (2026), p. 7.
2. **Archivo y función:** `src/sm_cfhr/inference.py` (`spa_test`, líneas 117–122); `src/ashr_oof_pipeline/evaluacion_estadistica.py` (`compute_scenario_metrics`, líneas 31–44).
3. **Evidencia concreta:** `spa_test` calcula `se` bootstrap en la línea 117 pero nunca la usa en las líneas 118 y 122 (divide por `np.std(d, ddof=1)` i.i.d.). Además, `compute_scenario_metrics` promedia errores absolutos en `kWh/día` (`kWh_total` en directos) entre 17 plantas de distinta capacidad instalada, y `resultados_loader.py:206` promedia sobre `1..17` plantas disponibles por origen sin balancear.
4. **Consecuencia científica:** Plantas grandes dominan la métrica y la inferencia, y los orígenes iniciales con pocas plantas introducen heteroscedasticidad en $d_t$.
5. **Corrección propuesta:** Usar `se` bootstrap/HAC en `spa_test`, reportar `nMAE`/`nRMSE` en factor de capacidad (`CF`) + `RMSE skill score` (Yang et al., 2020) y verificar sensibilidad en panel balanceado.

# 05. Plan Priorizado de Cambios (P0 – P3)

**Proyecto:** SM-CFHR (`sm-cfhr v1.0.3`)  
**Nota de fase:** Este documento planifica las correcciones posteriores tras revisión con el profesor. **Ninguno de estos cambios ha sido implementado aún.**

---

## Leyenda de clasificación operativa
Cada cambio indica explícitamente:
- **Reentrenamiento:** `No` (usa predicciones OOF ya guardadas) vs. `Sí` (requiere correr nuevos modelos).
- **Cálculo estadístico nuevo:** `Sí` / `No`.
- **Impacto en protocolo confirmatorio:** `Altera protocolo confirmatorio` vs. `Complementario / Post-hoc`.

---

## P0 — Errores metodológicos críticos (comprometen la validez de conclusiones actuales)

| ID | Cambio propuesto | Qué corrige | Reentrenar | Nuevos cálculos estadísticos | Naturaleza confirmatoria vs. post-hoc |
|---|---|---|---|---|---|
| **P0.1** | Corregir el parámetro `horizon_h` pasado a `dm_hln_hac` según el paso real de `time_col` (`horizon_h=1` cuando los orígenes están espaciados cada $H$ días) | `resultados_loader.py:236-241` + `inferencia_estadistica_v2.py:119`; elimina deflación artificial de hasta $9\times$ en $H=21, 28$ | **No** | **Sí** | Corrige bug de implementación del protocolo confirmatorio ya pre-especificado |
| **P0.2** | Corregir el umbral `t_eff_insufficient` a unidades coherentes con los bloques temporales evaluados (`T_eff_bloques >= 30` en lugar de `T_eff_bloques < 2*H_dias`) | `evaluacion_estadistica.py:156`; elimina descarte algebraicamente imposible de 13/28 escenarios | **No** | **Sí** | Corrige bug de unidades en el protocolo confirmatorio (previa validación docente) |
| **P0.3** | Reemplazar `model_confidence_set` bilateral de `evaluacion_estadistica.py:118` por la versión unilateral correcta (`mcs_stationary_v2`), con `B >= 2000` y `mean_block` ajustado al paso de `time_col` | `evaluacion_estadistica.py:90-133`; restaura poder discriminativo del MCS | **No** | **Sí** | Corrige bug de implementación del protocolo confirmatorio |
| **P0.4** | Eliminar la selección retrospectiva (*ex post*) de `Mejor_Bib` en inferencia DM-HLN; sustituir por benchmark fijo pre-especificado (`CP_opt` / `Ridge` o `Lasso`) + pruebas múltiples conjuntas (MCS/SPA) | `resultados_loader.py:156, 230` | **No** | **Sí** | **Altera protocolo confirmatorio** (requiere aprobación del profesor) |

---

## P1 — Correcciones importantes (fortalecen inferencia, comparabilidad y reproducibilidad)

| ID | Cambio propuesto | Qué fortalece | Reentrenar | Nuevos cálculos estadísticos | Naturaleza confirmatoria vs. post-hoc |
|---|---|---|---|---|---|
| **P1.1** | Corregir `spa_test` (`src/sm_cfhr/inference.py:117-122`) para usar la varianza dependiente HAC/bootstrap (`se`) en lugar de `std(d)` i.i.d. | Validez asintótica de SPA bajo autocorrelación | **No** | **Sí** | Corrige bug de implementación del protocolo confirmatorio |
| **P1.2** | Reportar métricas primarias normalizadas por capacidad (`nMAE_CF`, `nRMSE_CF`) y el `RMSE skill score` contra combinación convexa climatología–persistencia (Yang et al., 2020) | Elimina sesgo de escala en `kWh` entre 17 plantas; estandariza comparación solar | **No** | **Sí** | **Altera métricas primarias** si sustituye a `MAE_macro`, o complementaria si se añade en paralelo |
| **P1.3** | Alinear multiplicidad Holm estrictamente por familia pre-especificada (`suite × referencia`) en `evaluate_r3_superiority` y verificar sensibilidad de $d_t$ en panel balanceado de plantas | Evita penalización cruzada de 56 filas y heteroscedasticidad por orígenes con 1 planta | **No** | **Sí** | Alinea código con `protocolo.md:42` |
| **P1.4** | Corregir `mean_block` en `stationary_bootstrap_ci` (`evaluacion_estadistica.py:151`) para escalar en bloques del paso temporal real y no en días sobre series de bloques | Evita bloques de 28 bloques (= 784 días) sobre series de longitud $T=31$ | **No** | **Sí** | Corrige bug de implementación |
| **P1.5** | Reformular conclusiones y veredicto R3 separando explícitamente **superioridad condicionada (`ET + Clima`)** de **no superioridad global (`IP`, `DirectoSinClima`)** | Rigor y honestidad científica (Linca et al., 2026) | **No** | **No** (tras P0–P1) | Interpretativo / editorial |

---

## P2 — Análisis complementarios (aportan profundidad sin invalidar lo existente)

| ID | Cambio propuesto | Qué aporta | Reentrenar | Nuevos cálculos estadísticos | Naturaleza confirmatoria vs. post-hoc |
|---|---|---|---|---|---|
| **P2.1** | Verificación orientada a distribuciones Murphy–Winkler (descomposición sesgo-varianza de MSE, densidad conjunta, ECDFs, *ridgeline plots*; Yang et al., 2020) | Diagnóstico fino de calibración y resolución en `ET` vs. `IP` | **No** | **Sí** | **Exploratorio / Post-hoc** |
| **P2.2** | Análisis por planta y agrupación geográfica en escala `CF` (`nMAE_CF`, `s_RMSE`), más comparación de asimetría pérdida Huber vs. L2 | Explica dónde y por qué gana `SM-CFHR` y cuantifica efecto de función de pérdida | **No** (análisis `CF`); **Sí** (si se reentrena competencia con Huber) | **Sí** | **Exploratorio / Post-hoc** |
| **P2.3** | Extender diagnóstico de intervalos conformes por horizonte (`7d..28d`) y estación del año (Linca et al., 2026, p. 23) | Explica y cuantifica subcobertura en horizontes largos | **No** (sobre predicciones OOF existentes) | **Sí** | **Exploratorio / Post-hoc** |

---

## P3 — Extensiones futuras (requieren nuevos datos, entrenamientos o recursos importantes)

| ID | Cambio propuesto | Qué requiere | Reentrenar | Nuevos cálculos estadísticos | Naturaleza confirmatoria vs. post-hoc |
|---|---|---|---|---|---|
| **P3.1** | Holdout temporal prospectivo de $\ge 12$ meses posteriores (`>= 2026-11..2027-10`) con pronósticos GEFSv12/Open-Meteo archivados pre-emisión y `SM-CFHR-Fixed` congelado | Nuevos datos XM + archivo operacional meteorológico | **No** (evaluación de modelo congelado) o actualización causal | **Sí** | Nuevo estudio confirmatorio prospectivo |
| **P3.2** | Re-evaluación multihorizonte con orígenes diarios solapados ($N \approx 800$) para $H=7,14,21,28$ o comparación justa con HPO equivalente ($B=25$) en todas las bibliotecas | Re-ejecutar pipeline OOF completo | **Sí** | **Sí** | Nuevo protocolo experimental (requiere recursos de cómputo) |
| **P3.3** | Validación espacial en plantas externas de otras regiones climáticas | Datos de nuevas plantas PV | **Sí** | **Sí** | Extensión de generalización externa |

---

## Orden recomendado de ejecución (una vez aprobado por el profesor)
1. **Fase inmediata (sin reentrenar, bajo costo):** Ejecutar **P0.1 + P0.2 + P0.3 + P1.1 + P1.3 + P1.4** como una actualización limpia de post-procesamiento estadístico sobre las predicciones OOF ya persistidas, etiquetando claramente qué cambia por corrección de bugs vs. decisión de protocolo (**P0.4**, **P1.2**).
2. **Fase diagnóstica (sin reentrenar):** Añadir **P2.1**, **P2.2 (en CF)** y **P2.3** como análisis exploratorios *post-hoc* en el notebook y manuscrito.
3. **Fase prospectiva (cuando haya $\ge 12$ meses nuevos):** Ejecutar **P3.1** con `SM-CFHR-Fixed` congelado.

# 07. Verificación Independiente y Correcciones Estadísticas — Informe para el revisor técnico

**Proyecto:** SM-CFHR (`sm-cfhr v1.0.3`)  
**Alcance:** Verificación empírica e independiente de los hallazgos de la auditoría bibliográfica sobre los resultados OOF definitivos (`28` escenarios, `14.625` unidades por variante), sin reentrenar modelos ni modificar los archivos OOF originales (`resultados/evaluacion_ashr_oof/`).

---

## 1. Resumen ejecutivo: errores confirmados vs. sospechas matizadas

### A. Errores objetivos de código confirmados y corregidos en módulo paralelo (`src/sm_cfhr/inference_v2.py`)
1. **P0.1 — Desajuste de unidades en DM-HLN-HAC:** Verificado en los 28 archivos CSV de predicciones que todos los escenarios multihorizonte (`7d`, `14d`, `21d`, `28d`) evalúan orígenes espaciados en **bloques disjuntos exactos** de $H$ días (`median_step_days == horizon_days`, sin solapamiento entre orígenes consecutivos). Sin embargo, el código original pasaba `horizon_h = H` a la fórmula de Harvey, Leybourne & Newbold (1997, pp. 282–284), encogiendo el estadístico DM por un factor de `0.112` (casi **9×**) en `28d` y `0.500` (**2×**) en `21d`.
2. **P0.2 — Desajuste de unidades en `T_eff >= 2H`:** Verificado en `evaluacion_estadistica.py:156` que $T_{\text{eff}}$ (medido en *bloques*, máximo `31` en `28d`) se comparaba contra `2 * horizon_days` (medido en *días*, `56`), descartando `13/28` escenarios por imposibilidad algebraica.
3. **P0.3 — MCS bilateral en `evaluacion_estadistica.py`:** Confirmado que el pipeline oficial comparaba $\max(t)$ (unilateral) contra $\max(|t^*|)$ (bilateral) con `B = 200` réplicas, inflando los p-valores de supervivencia frente a la formulación de Hansen, Lunde & Nason (2011) y `arch.bootstrap.MCS`.
4. **P1.1 — Varianza i.i.d. en `spa_test`:** Confirmado que `src/sm_cfhr/inference.py:117-122` calculaba el `se` bootstrap dependiente pero luego dividía por `std(d)` i.i.d. en lugar de la varianza de largo plazo dependiente $\hat{\omega}_k$ (Hansen, 2005).
5. **P1.4 — Longitud de bloque bootstrap en días:** Confirmado que `stationary_bootstrap_ci` usaba `mean_block = max(7, H)`, generando bloques de `28` observaciones sobre series de `31` orígenes.

### B. Sospechas matizadas o elecciones metodológicas discutibles (no modificadas silenciosamente)
- **`T_eff >= 30` tampoco es mágico:** Como muestra el análisis de sensibilidad (`B_analisis_sensibilidad/sensibilidad_umbral_teff.csv`), `T_eff >= 2*h_overlap` y `T_eff >= 20` habilitan los 28 escenarios (`4` victorias Holm), `T_eff >= 25` excluye `5` escenarios (aún `4` victorias Holm), y `T_eff >= 30` excluye `8` escenarios (`3` victorias Holm, excluyendo `RecGEFSv12_ET_14d` con `T_eff=25.48`).
- **Selección retrospectiva de `Mejor_Bib` (`ex post`):** Elegir al mejor de 9 competidores dentro de la misma muestra OOF y correr DM pareado contra él penaliza al modelo propio sin ajuste por búsqueda (Tashman, 2000, pp. 446–447), pero cambiar la referencia confirmatoria requiere aprobación de revisión externa, por lo que se mantiene en el Bloque A original y se evalúan referencias fijas *ex ante* en el Bloque C exploratorio.

---

## 2. Referencias científicas y páginas verificables aplicadas
- **Harvey, Leybourne & Newbold (1997)**, *IJF*, 13(2), pp. 282–284 (fórmula HLN y distribución $t_{T-1}$ bajo estructura MA($h-1$)).
- **Diebold & Mariano (1995)**, *JBES*, 13(3), pp. 253–263.
- **Tashman (2000)**, *IJF*, 16(4), pp. 445–447 (evitar medidas dependientes de escala entre series heterogéneas; distinción *ex ante* vs. *ex post*).
- **Holm (1979)**, *Scand. J. Stat.*, 6(2), pp. 65–70 (ajuste secuencial por familia).
- **Oliveira, Torgo & Santos Costa (2021)**, *Mathematics*, 9, 691, pp. 19–24 (validación causal espaciotemporal `PtBsR_grW` y no independencia entre escenarios compartidos).
- **Hewamalage, Ackermann & Bergmeir (2023)**, *DMKD*, 37, pp. 802, 824–826 (métricas dependientes de escala, multiplicidad e interpretación de potencia limitada).
- **Yang et al. (2020)**, *Solar Energy*, 210, pp. 25–31 (`nRMSE` vs. `RMSE skill score` frente a climatología–persistencia).
- **Linca et al. (2026)**, *Applied Sciences*, 16, 11052, pp. 7–9, 19, 23 (normalización a `CF`, *block bootstrap* y reporte transparente de victorias descriptivas no significativas tras Holm).
- **Hansen (2005)** y **Hansen, Lunde & Nason (2011):** Verificados empíricamente mediante contraste directo contra `arch.bootstrap.SPA` y `arch.bootstrap.MCS` (`arch 8.0.0`).

---

## 3. Cambios implementados y pruebas automatizadas
- **Código nuevo y compatible:** [`src/sm_cfhr/inference_v2.py`](file:///<local-path>), corrección compatible en [`src/ashr_oof_pipeline/evaluacion_estadistica.py`](file:///<local-path>) y [`src/sm_cfhr/inference.py`](file:///<local-path>).
- **Pruebas automatizadas:** [`tests/test_auditoria_inferencia_v2.py`](file:///<local-path>) (`7/7 PASSED`), validando DM-HLN (1-paso, solapado, no solapado, AR(1), degenerados), MCS unilateral vs. `arch.MCS`, SPA studentizado vs. `arch.SPA` y bootstrap en unidades de observación.
- **Recálculo independiente:** [`scripts/evaluacion/ejecutar_auditoria_inferencia.py`](file:///<local-path>) generó [`resultados/auditoria_inferencia_corregida/`](file:///<local-path>) y el notebook [`auditoria_inferencia_estadistica.ipynb`](file:///<local-path>).

---

## 4. Resultados anteriores vs. corregidos (impacto simétrico real)

| Categoría tras corregir unidades DM-HLN, `T_eff` y Holm por familia | Conteo | Escenarios concretos |
|---|---:|---|
| **Victorias significativas mantenidas (Holm)** | **2** | `D1_espaciotemporal_cinco_variables` ($p_{\text{Holm}}=4.56\times 10^{-23}$), `RecGEFSv12_espaciotemporal_7d` ($p_{\text{Holm}}=9.45\times 10^{-9}$) |
| **Nuevas victorias significativas tras Holm** | **2** | `RecGEFSv12_espaciotemporal_14d` ($p_{\text{Holm}}=0.00070$), `RecGEFSv12_espaciotemporal_28d` ($p_{\text{Holm}}=0.00037$, `dm_hln`: `-0.535 -> -4.707`) |
| **Nuevas victorias nominales ($p<0.05$ sin Holm)** | **1** | `RecGEFSv12_espaciotemporal_21d` ($p=0.0158$, $p_{\text{Holm}}=0.0789$) |
| **Nuevas derrotas nominales ($p<0.05$ sin Holm)** | **3** | `DirectoSinClima_intraplanta_21d` ($p=0.0326$), `DirectoSinClima_intraplanta_28d` ($p=0.0302$), `RecGEFSv12_intraplanta_14d` ($p=0.0352$) |
| **Sin evidencia suficiente tras corrección** | **20** | Incluyen todos los `RecSinClima`, `DirectoSinClima_ET`, `D1` sin clima / IP, y `RecGEFSv12_IP_7d/21d/28d` |

- **MCS corregido (unilateral, $B=2000$):** Concordancia de `28/28` con `arch.MCS` en inclusión de `ASHR` (`27/28` en 23 modelos). En el MCS solo de bibliotecas (`ASHR` + 9 bibliotecas), `ASHR` es **sobreviviente único (`1/10`) en los 5 escenarios `ET + Clima`** (`D1`, `7d`, `14d`, `21d`, `28d`).
- **Comparación exploratoria *ex ante* (Bloque C, sin `Mejor_Bib` retrospectivo):** Frente a referencias fijadas *a priori*, `SM-CFHR-Fixed` vence significativamente tras Holm a `Ridge` en `11/28`, a `BayesianRidge` en `8/28` y a `Lasso` en `6/28`, con `0` derrotas significativas tras Holm.

---

## 5. Implicaciones para SM-CFHR y decisiones para el revisor técnico
La evidencia corregida **refuerza la conclusión condicionada** (superioridad estadística sólida en **Espaciotemporal + Clima**), mientras **confirma que no existe superioridad global** en `Intraplanta` ni en `DirectoSinClima`.

**Decisiones que requieren aprobación de revisión externa:**
1. **Aprobar la actualización oficial de las tablas de `resultados/evaluacion_ashr_oof/`** con las correcciones objetivas de implementación (Bloque A: `h_overlap=1` en bloques disjuntos, `MCS` unilateral `B=2000`, `SPA` dependiente y `mean_block` en observaciones).
2. **Definir el criterio oficial de $T_{\text{eff}}$** (p. ej. `T_eff >= 20` o `>= 25` bloques vs. `>= 30` bloques).
3. **Aprobar la sustitución o complemento de `Mejor_Bib` *ex post*** por benchmarks fijos *ex ante* + MCS/SPA corregidos para eliminar el sesgo de selección retrospectiva.
4. **Aprobar la incorporación de métricas libres de escala (`CF` / `s_RMSE`)** como métricas primarias junto a `MAE_macro` en `kWh/día`.

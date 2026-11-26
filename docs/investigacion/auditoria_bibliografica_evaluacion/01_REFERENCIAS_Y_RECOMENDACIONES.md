# 01. Referencias Bibliográficas y Extracción de Recomendaciones

**Proyecto:** SM-CFHR (`sm-cfhr v1.0.3`)  
**Fase:** Auditoría metodológica integral basada en literatura científica (solo lectura e investigación; sin modificaciones a código ni resultados)  
**Regla de trazabilidad:** Todas las páginas citadas corresponden a los PDFs leídos directamente en esta sesión. Las referencias citadas secundariamente o presentes en el código sin PDF leído localmente se declaran de forma explícita y **sin** asignación de números de página.

---

## 1. Inventario de artículos localizados y verificados

### A. Artículos principales recién descargados en `Downloads` (leídos completos)

| ID | Título exacto | Autores y año | Revista / Vol. / Págs. | DOI / Identificador | Ubicación local | Págs. PDF |
|---|---|---|---|---|---|---|
| **R1** | *Forecast evaluation for data scientists: common pitfalls and best practices* | Hewamalage, H., Ackermann, K., & Bergmeir, C. (2023) | *Data Mining and Knowledge Discovery*, 37, 788–832 | `https://doi.org/10.1007/s10618-022-00894-5` | `<local-path>` | 45 |
| **R2** | *Verification of deterministic solar forecasts* | Yang, D., Alessandrini, S., Antonanzas, J., et al. (2020) | *Solar Energy*, 210, 20–37 | `https://doi.org/10.1016/j.solener.2020.04.019` | `<local-path>` | 18 |
| **R3** | *Evaluation Procedures for Forecasting with Spatiotemporal Data* | Oliveira, M., Torgo, L., & Santos Costa, V. (2021) | *Mathematics*, 9, 691 | `https://doi.org/10.3390/math9060691` | `<local-path>` | 27 |
| **R4** | *Out-of-sample tests of forecasting accuracy: an analysis and review* | Tashman, L. J. (2000) | *International Journal of Forecasting*, 16(4), 437–450 | `PII: S0169-2070(00)00065-0` | `<local-path>` | 14 |
| **R5** | *Solar Forecasting: Small-Window Performance and Cross-Site Validation* | Linca, N.-R., Ruseti, S., Dascalu, M., & Rebedea, T. (2026) | *Applied Sciences*, 16, 11052 | `https://doi.org/10.3390/app162111052` | `<local-path>` | 28 |

### B. Artículos complementarios verificados en el repositorio hermano (`InvML`)

| ID | Título exacto | Autores y año | Revista / Vol. / Págs. | Ubicación local | Págs. PDF |
|---|---|---|---|---|---|
| **R6** | *Comparing Predictive Accuracy* | Diebold, F. X., & Mariano, R. S. (1995) | *Journal of Business & Economic Statistics*, 13(3), 253–263 | `<workspace>\fuentes\bibliografia\articulos_pdf\Diebold_Mariano_1995_Comparing_Predictive_Accuracy.pdf` | 11 |
| **R7** | *Testing the equality of prediction mean squared errors* | Harvey, D., Leybourne, S., & Newbold, P. (1997) | *International Journal of Forecasting*, 13(2), 281–291 | `<workspace>\fuentes\bibliografia\articulos_pdf\Harvey_Leybourne_Newbold_1997_Testing_Equality_Prediction_MSE.pdf` | 11 |
| **R8** | *Evaluating time series forecasting models: an empirical study on performance estimation methods* | Cerqueira, V., Torgo, L., & Mozetič, I. (2020) | *Machine Learning*, 109(11), 1997–2028 | `<workspace>\fuentes\bibliografia\articulos_pdf\Cerqueira_et_al_2020_Evaluating_Time_Series_Forecasting_Models.pdf` | 32 |
| **R9** | *A Simple Sequentially Rejective Multiple Test Procedure* | Holm, S. (1979) | *Scandinavian Journal of Statistics*, 6(2), 65–70 | `<workspace>\fuentes\bibliografia\articulos_pdf\Holm_1979_Simple_Sequentially_Rejective_Multiple_Test.pdf` | 7 |

### C. Referencias metodológicas usadas en el código sin PDF local leído en esta sesión
No se encontraron archivos PDF locales correspondientes a:
- Hansen, P. R. (2005) — *Superior Predictive Ability (SPA)*.
- Hansen, P. R., Lunde, A., & Nason, J. M. (2011) — *The Model Confidence Set (MCS)*.
- Giacomini, R., & White, H. (2006) — *Tests of Conditional Predictive Ability (CPA)* (citado secundariamente en Hewamalage et al., 2023, p. 821 / PDF p. 34).
- Clark, T. E., & West, K. D. (2007) — *Approximately normal tests for equal predictive accuracy in nested models*.
- Newey, W. K., & West, K. D. (1987 / 1994) — Estimador HAC.
- Politis, D. N., & Romano, J. P. (1994) — *Stationary bootstrap*.
- Nadeau, C., & Bengio, Y. (2003) — Prueba *t* corregida para CV repetida.
- Driscoll, J. C., & Kraay, A. C. (1998) — Errores estándar robustos a dependencia espacial y temporal en paneles.

Cualquier referencia a estos métodos se realiza exclusivamente sobre su implementación verificada en `src/ashr_oof_pipeline/evaluacion_estadistica.py`, `src/sm_cfhr/inference.py` y `scripts/evaluacion/inferencia_estadistica_v2.py`, o sobre menciones secundarias verificadas en **R1–R5**, **sin inventar números de página**.

---

## 2. Extracción exhaustiva por artículo (Recomendación explícita vs. Inferencia para SM-CFHR)

### R1. Hewamalage, Ackermann & Bergmeir (2023) — *Forecast evaluation for data scientists*

#### Diseño experimental
- **Explícito:**
  - *Rolling-origin / tsCV* frente a *fixed-origin*: en series suficientemente largas, `tsCV` (ventana creciente o deslizante) es el procedimiento preferido cuando existen no estacionariedades o modelos con estado continuo (pp. 807–812, PDF pp. 20–25; Fig. 11 p. 812). La CV aleatoria es válida en configuraciones autorregresivas puras estacionarias solo si los residuos OOS no muestran autocorrelación (verificada con la prueba de Ljung-Box, pp. 809–810, PDF pp. 22–23).
  - **Prevención de fuga (*data leakage*):** prohibido aplicar normalización, suavizado o descomposiciones (como EMD) sobre la serie completa antes de particionar; también hay fuga si una serie contiene información futura de otra al entrenar modelos globales con series desalineadas o al usar variables resumen del día completo para un pronóstico emitido antes de medianoche (pp. 804–806, PDF pp. 17–19).
  - **Benchmarks obligatorios y paridad de entrenamiento:** comparar siempre contra referencias simples (naïve, seasonal naïve, DHR-ARIMA en series estacionales, pp. 795–800, PDF pp. 8–13). Evitar la asimetría de entrenar el modelo propio con pérdida L1 y los competidores con L2 cuando se evalúa con MAE (p. 790, PDF p. 3). No comparar pronósticos directos multi-paso contra iterativos/recursivos atribuyendo la ganancia a la arquitectura cuando proviene de la estrategia multi-horizonte (citando a Zeng et al., 2022, pp. 800–801, PDF pp. 13–14).
- **Inferencia para SM-CFHR:** SM-CFHR entrena con pérdida Huber en factor de capacidad (`CF`), mientras `Ridge`, `BayesianRidge`, `Lasso`, `Random Forest` y `XGBoost` minimizan L2, y la evaluación principal usa `MAE_macro` (mediana); esto introduce asimetría pérdida-métrica. Además, `Directo` y `Recursivo` deben analizarse separados (como ya hace SM-CFHR en sus 4 suites).

#### Métricas
- **Explícito:**
  - Las métricas dependientes de escala (`MAE`, `RMSE`, `MSE`) en conjuntos multiserie quedan dominadas por las series de mayor magnitud si las escalas difieren (p. 802, PDF p. 15; p. 812, PDF p. 25).
  - `MAPE` falla cerca de cero (p. 802, PDF p. 15) y `R^2` es engañoso en procesos tipo caminata aleatoria (p. 802, PDF p. 15).
  - Consistencia función de pérdida–métrica: MSE/RMSE optimizan la media condicional; MAE/MASE optimizan la mediana (p. 813, PDF p. 26). Evaluar con múltiples métricas sirve como *sanity check* (p. 813). Sugieren como trabajo futuro métricas combinadas tipo Huber entre MAE y RMSE (p. 828, PDF p. 41).
  - Checklist de selección (Tabla 9, pp. 822–823; Fig. 12, p. 824).
- **Inferencia para SM-CFHR:** `MAE_macro` en `kWh/día` (y `kWh_total` en directos) está dominado por las plantas colombianas de mayor capacidad instalada (MW). Normalizar por capacidad (`CF`) o reportar métricas relativas/escaladas evita ese sesgo de escala.

#### Inferencia estadística
- **Explícito:**
  - Pruebas pareadas para series temporales: Diebold-Mariano, Giacomini-White (CPA) y Wilcoxon (p. 821, PDF p. 34).
  - Cuando se comparan **más de 2 modelos**, es obligatorio corregir por múltiples pruebas (Bonferroni, Holm, Hochberg, Nemenyi/MCB; pp. 821, 824–825, PDF pp. 34, 37–38).
  - Advertencia sobre tamaño muestral y composición del conjunto de modelos: añadir muchos modelos malos infla la distancia crítica (CD) y oculta diferencias reales entre buenos modelos; asimismo, con muestras pequeñas falta potencia para distinguir competidores cercanos (p. 826, PDF p. 39).
- **Inferencia para SM-CFHR:** Incluir 22 benchmarks en MCS/rankings aumenta la multiplicidad; la falta de potencia con $T \in [31, 63]$ bloques en horizontes largos es esperada si no se controlan las pruebas y la unidad temporal.

#### Evaluación adicional
- **Explícito:** Los gráficos de pronóstico vs. observados pueden inducir conclusiones espurias por la ilusión de distancia horizontal vs. vertical; deben usarse solo como verificación visual de sanidad y no como prueba de superioridad (pp. 802–804, PDF pp. 15–17).

---

### R2. Yang, Alessandrini, Antonanzas et al. (2020) — *Verification of deterministic solar forecasts*

#### Diseño experimental
- **Explícito:**
  - Excluir siempre horas nocturnas mediante filtro de ángulo cenital ($\theta_z < 85^\circ$), manteniendo el espaciado temporal original al calcular autocorrelaciones $\rho_h$ (p. 32, PDF p. 13).
  - Control de calidad (*data trimming*) previo, **nunca** eliminando pares por tener error grande sin investigar su causa (p. 32, PDF p. 13).
  - Verificar la convención de estampado temporal (*floor*, *ceiling*, *round*) al alinear observaciones y NWP para evitar desalineaciones artificiales (p. 32, PDF p. 13).
  - Considerar el efecto de suavizado geográfico (*geographical smoothing*): la generación agregada de una planta grande o un clúster de plantas tiene menor variabilidad y es más predecible que una ubicación puntual (p. 32, PDF p. 13).
  - Publicar código y datos para reproducibilidad (p. 33, PDF p. 14).
- **Inferencia para SM-CFHR:** SM-CFHR trabaja con energía diaria agregada (`kWh/día`), por lo que el filtro nocturno intra-horario no aplica directamente al agregado diario, pero el suavizado geográfico y la correlación espacial entre las 17 plantas en Colombia sí aplican al analizar escenarios espaciotemporales frente a intraplanta.

#### Métricas
- **Explícito:**
  - **Falsa comparabilidad de `nRMSE` / `nMAE`:** normalizar por la media, rango, máximo o capacidad instalada no elimina la dependencia respecto a la variabilidad y predictibilidad del sitio/periodo; solo permite comparar sistemas distintos si comparten ubicación y variabilidad (pp. 24–25, PDF pp. 5–6, nota al pie 10).
  - **Recomendación central — *RMSE skill score* frente a climatología + persistencia:** se recomienda reportar universalmente $s = 1 - \text{RMSE}(f, x) / \text{RMSE}(r, x)$ (Ec. 4, p. 26) usando como referencia $r$ la **combinación convexa óptima de climatología interna y persistencia a $h$ pasos** sobre la variable desestacionalizada / índice de cielo claro (pp. 30–31, PDF pp. 11–12; Ecs. 17–18, p. 31), cuyo MSE es $\text{MSE}_{cp} = (1 - \rho_h^2)\text{Var}(x) \le \min(\text{MSE}_c, \text{MSE}_p)$.
  - **Por qué RMSE y no MAE para el *skill score*:** errores grandes tienen mayor impacto en la integración a red y RMSE es consistente con la media condicional y con la factorización de MSE (pp. 22–23, 29–30, PDF pp. 3–4, 10–11).
  - **MBE:** tener MBE cercano a cero es un prerrequisito básico de cualquier pronóstico calibrado, no un mérito diferenciador ni apto para *skill score* (pp. 30, 33, PDF pp. 11, 14).
  - Si se reportan errores normalizados, tabular explícitamente los denominadores de normalización (p. 33, PDF p. 14).
- **Inferencia para SM-CFHR:** SM-CFHR reporta `Skill_vs_MM30_pct` basado en `MAE`, no `RMSE skill score` contra la combinación convexa climatología-persistencia. Además, aunque `CF` normaliza por capacidad instalada entre las 17 plantas, aún refleja diferencias climáticas entre regiones de Colombia.

#### Inferencia estadística
- **Explícito:** No desarrolla pruebas de hipótesis formales (DM/MCS/SPA), sino verificación cuantitativa y distributiva. Menciona la prueba de Kolmogorov-Smirnov sobre ECDFs marginales como ejemplo en su caso de estudio (p. 29, PDF p. 10).

#### Evaluación adicional
- **Explícito:** Adopción obligatoria y complementaria del marco **orientado a distribuciones de Murphy-Winkler** (`p(f, x) = p(x|f)p(f) = p(f|x)p(x)`), descomposición sesgo-varianza (Ec. 12, p. 27) y factorizaciones calibración-refinamiento y verosimilitud-tasa base (Ecs. 13–14, p. 27; Tabla 3, p. 28), acompañadas de gráficos de densidad conjunta, ECDFs y *ridgeline plots* (pp. 28–29, PDF pp. 9–10).

---

### R3. Oliveira, Torgo & Santos Costa (2021) — *Evaluation Procedures for Forecasting with Spatiotemporal Data*

#### Diseño experimental
- **Explícito:**
  - Las dependencias espaciales y temporales rompen el supuesto de independencia de la CV estándar (`CVtRsR`), generando **subestimaciones severas y optimistas del error** tanto en datos sintéticos STARMA/NLSTAR como en 17 series ambientales reales (pp. 2–3, 17–22, 24). El *holdout* 80/20 único también exhibe una proporción elevada de estimaciones optimistas en datos reales (pp. 17, 21–22, 24).
  - Distinguir claramente entre **interpolación espaciotemporal**, **pronóstico futuro en sitios conocidos** y **pronóstico en ubicaciones nuevas no vistas** (p. 4, p. 24).
  - En datos reales espaciotemporales, la **evaluación prequential por bloques espaciotemporales (`PtBsR_grW`)** destaca al aproximar el *gold standard* con alta precisión sin caer en estimaciones optimistas y respetando estrictamente el orden temporal (no entrena con futuro para predecir pasado) (pp. 21–22, 24). La CV con bloqueo espacial y *buffer* (`CVtAsR_S`) también mitiga el optimismo usando todo el conjunto de datos (pp. 22, 24).
  - La dimensión temporal resulta especialmente crítica al evaluar pronósticos espaciotemporales (p. 22).
- **Inferencia para SM-CFHR:** Valida directamente la distinción entre el régimen **Espaciotemporal (`ET`)** (generalización a plantas/sitios no vistos) y el régimen **Intraplanta (`IP`)** (pronóstico temporal dentro de cada planta), así como el uso de partición causal prequential/rolling-origin. Advierte que cualquier evaluación sin bloqueo temporal/espacial (o con mezcla de plantas correlacionadas sin control) subestima la incertidumbre real.

#### Métricas
- **Explícito:** Emplea `NMAE` (error absoluto medio normalizado) en sus experimentos comparativos y evalúa tanto error absoluto respecto al *gold standard* como sesgo direccional (optimismo vs. pesimismo) (pp. 17–21).

#### Inferencia estadística
- **Explícito:** Aplica la prueba no paramétrica de Friedman-Nemenyi separadamente por cada algoritmo de aprendizaje para no violar la independencia entre datasets (p. 19).

#### Evaluación adicional
- **Explícito:** Recomienda en estudios futuros definir *gold standards* tanto temporales como sobre ubicaciones nuevas no observadas (p. 24).

---

### R4. Tashman (2000) — *Out-of-sample tests of forecasting accuracy: an analysis and review*

#### Diseño experimental
- **Explícito:**
  - *Fixed-origin* sufre tres defectos: un único error por horizonte, alta vulnerabilidad a eventos atípicos del origen y mezcla de horizontes en promedios globales (p. 439, PDF p. 3). Debe usarse **rolling-origin** con múltiples orígenes y, de ser posible, múltiples periodos de prueba para diversidad cíclica (pp. 439–441, PDF pp. 3–5).
  - Regla de partición: para garantizar al menos $M$ pronósticos en el horizonte máximo $H$, el periodo de prueba debe medir $N = H + M - 1$ (p. 440, PDF p. 4).
  - **Recalibración frente a actualización:** recalibrar los parámetros en cada avance del origen es preferible a solo actualizar entradas; en modelos causales de regresión, no recalibrar degenera la evaluación (p. 440, PDF p. 4).
  - **Ventana deslizante (*rolling window*):** prunar la observación más antigua mantiene constante el tamaño de ajuste cuando se compara la evolución del rendimiento entre periodos tempranos y tardíos (p. 442, PDF p. 6).
  - Prohibido *peeking* sobre el conjunto de prueba durante la selección del método (p. 438, PDF p. 2).
  - **Distinción *ex post* vs. *ex ante* (citando a Fildes, 1989):** una competencia muestra qué método ganó *ex post* en esa muestra, pero no demuestra por sí sola cómo seleccionar al ganador *ex ante* (pp. 446–447, PDF pp. 10–11).
- **Inferencia para SM-CFHR:** El diseño de orígenes espaciados cada $H$ días en SM-CFHR evita el defecto de *fixed-origin*, pero reduce $M$ a solo $31$ orígenes para $H=28$. Además, elegir `Mejor_Bib` *ex post* en cada escenario viola la distinción *ex post* vs. *ex ante*.

#### Métricas
- **Explícito:**
  - **En estructuras agrupadas (*pooled*) de múltiples series, evitar medidas dependientes de escala como RMSE o MAD/MAE**, porque un subconjunto de series con valores numéricos grandes domina el promedio (p. 445, PDF p. 9).
  - Usar medidas relativas (`RAE`, `MdRAE`, `GMRAE`) o errores porcentuales robustos (`MdAPE`, `sMAPE`) (p. 445, PDF p. 9).
  - **No mezclar horizontes:** calcular y reportar por cada horizonte individual (`lead time`), o promediar manteniendo igual ponderación por horizonte (`\sum_n \text{MAPE}` vs. doble suma) (pp. 445–446, PDF pp. 9–10).
- **Inferencia para SM-CFHR:** Refuerza la crítica contra `MAE_macro` en `kWh/día` entre 17 plantas de distinta potencia instalada.

#### Inferencia estadística y evaluación adicional
- **Explícito:** Recalca que los intervalos de predicción basados en errores *in-sample* son demasiado estrechos (citando a Chatfield, 1993, p. 438, PDF p. 2), y que los errores en un mismo horizonte forman una distribución empírica que debe analizarse por valores atípicos y estabilidad a través de orígenes (pp. 439, 446, PDF pp. 3, 10).

---

### R5. Linca, Ruseti, Dascalu & Rebedea (2026) — *Solar Forecasting: Small-Window Performance and Cross-Site Validation*

#### Diseño experimental
- **Explícito:**
  - **Entradas estrictamente operacionales:** entrenar y evaluar únicamente con pronósticos meteorológicos archivados emitidos **antes** del origen (Open-Meteo *previous-runs archive* emitido 24 h antes); el reanálisis ERA5 solo debe reportarse como cota retrospectiva no operacional (p. 8, Tabla 3 p. 3, Tabla 10 p. 20).
  - **Mismo presupuesto de HPO para todas las familias** sobre una ventana de validación estrictamente anterior a todo mes de prueba (p. 9, Tabla 1 p. 3).
  - Exclusión de horas nocturnas (elevación solar $< 5^\circ$, p. 7) y preservación explícita de huecos (*gaps*) sin imputación que contamine *lags* (p. 7).
  - Replicación cruzada en un segundo sitio independiente (`Site B`) y evaluación multi-horizonte *fixed-origin* (1–14 días) sin observaciones post-origen (pp. 6, 18–20).
- **Inferencia para SM-CFHR:** En `SM-CFHR-HPO`, solo SM-CFHR recibe $B=25$ trials de Optuna por pliegue externo, mientras no está garantizado que los 22 benchmarks tengan presupuesto equivalente de HPO anidada por pliegue. Asimismo, cualquier evaluación prospectiva nueva debe usar archivo de pronósticos emitidos *antes* del origen, jamás reanálisis ERA5 como predictor operacional.

#### Métricas
- **Explícito:** Normalización a **factor de capacidad** `CF = clip(P_ac / P_rated, 0, 1)` (Ec. 1, p. 7) para comparabilidad entre sistemas; reporte uniforme de 5 métricas (`MAE`, `RMSE`, `MAPE`, `nRMSE`, `R^2`) con `nRMSE` (% de potencia nominal) como métrica primaria y `Skill` frente a *smart persistence* (pp. 5, 7; Tabla 3 p. 3).

#### Inferencia estadística
- **Explícito:**
  - En evaluación multi-horizonte con orígenes diarios consecutivos solapados (`167` orígenes), calculan intervalos de confianza del 95% mediante **moving-block bootstrap con bloques de 7 orígenes consecutivos** (p. 19, Tabla 10 p. 20).
  - En comparaciones por pares entre 7 modelos y 3 referencias sobre 12 meses *walk-forward*, aplican la **prueba exacta de rangos con signo de Wilcoxon ajustada con Holm** (Tabla 3 p. 3, Tabla A3 p. 26).
  - **Transparencia ante resultados no significativos:** aunque LightGBM y XGBoost superan a *smart persistence* en 11/12 meses, obtienen $p_{\text{Holm}} = 0.082$ y $0.102$ (no significativos a $\alpha=0.05$). Los autores declaran explícitamente que **no está estadísticamente establecida** la ventaja sobre *smart persistence*, que los meses *walk-forward* comparten datos de entrenamiento y no son independientes (por lo que los p-valores son indicativos), y que confirmar esa diferencia requiere un registro temporal más largo (p. 23).
- **Inferencia para SM-CFHR:** Precedente directo y reciente (2026) en energía fotovoltaica sobre cómo interpretar honestamente victorias descriptivas frecuentes (`11/12` allá; `18/28` en SM-CFHR) que no alcanzan significancia tras Holm por tamaño muestral limitado y dependencia entre pliegues.

#### Evaluación adicional
- **Explícito:** Ablación de entradas, sensibilidad a hiperparámetros ($\lambda$ de decaimiento), análisis estacional mes a mes, coste computacional de reentrenamiento en CPU, análisis de degradación de la ganancia meteorológica con el horizonte (Tabla 10 p. 20) y recomendación de **intervalos predictivos condicionados simultáneamente a estación y horizonte**, no solo al horizonte (p. 23).

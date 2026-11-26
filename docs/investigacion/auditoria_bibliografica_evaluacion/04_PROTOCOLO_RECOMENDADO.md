# 04. Protocolo de Evaluación Recomendado para SM-CFHR

**Proyecto:** SM-CFHR (`sm-cfhr v1.0.3`)  
**Propósito:** Formular un protocolo de evaluación metodológicamente sólido, coherente entre sí y respaldado por la literatura verificada, evitando combinar procedimientos incompatibles y documentando las discrepancias entre autores.

---

## 1. Preguntas científicas e hipótesis
El protocolo distingue explícitamente dos preguntas (separando superioridad global de superioridad condicionada, Tashman, 2000, pp. 446–447; Linca et al., 2026, p. 23):
1. **H1 (Régimen Espaciotemporal con Clima — `ET + Clima`):** Al generalizar a plantas no vistas usando pronósticos meteorológicos operacionales (`D+1` y `RecGEFSv12` 7–28d), `SM-CFHR-Fixed` reduce el error de pronóstico respecto a la referencia climatología–persistencia y a los modelos de biblioteca evaluados con el mismo presupuesto ($B=0$ o HPO anidada equivalente).
2. **H2 (Regímenes sin Clima e Intraplanta — `RecSinClima`, `DirectoSinClima`, `IP`):** Evaluar si las restricciones monótonas y de regularización estructurada aportan mejora o equivalencia predictiva frente a modelos lineales regularizados (`Ridge`, `BayesianRidge`, `Lasso`) y no lineales (`k-NN`, `SVR`, árboles) sin ocultar las derrotas (p. ej. `DirectoSinClima_intraplanta`).

---

## 2. Unidad experimental y estructura de dependencia
- **Unidad experimental primaria:** El par `(origen_temporal, planta)` evaluado en escala adimensional de **Factor de Capacidad (`CF`)** (`[0, 1]`), agrupado por fecha/origen para inferencia temporal sobre el promedio de cartera (o por planta en análisis desagregado), evitando tratar las $17$ plantas o los $28$ escenarios como muestras independientes (Oliveira et al., 2021, pp. 2–4, 19; Hewamalage et al., 2023, p. 802).
- **Panel balanceado vs. desbalanceado:** Reportar la serie temporal de diferencias de pérdida $d_t$ sobre el **subconjunto balanceado de plantas comunes** (o ponderado explícitamente por $1/P$ con registro de $P_t$) para evitar heteroscedasticidad artificial entre orígenes con $1$ planta y orígenes con $17$ plantas.

---

## 3. Diseño de particiones
- **Evaluación causal OOF:** Mantener el esquema **prequential por bloques espaciotemporales con ventana creciente (`PtBsR_grW`)** y separación temporal estricta (`embargo/gap >= H`), respaldado por **Oliveira et al. (2021, pp. 21–22, 24)** como el procedimiento OOS que mejor aproxima el *gold standard* en datos espaciotemporales reales sin sesgo optimista ni fuga futura, y por **Hewamalage et al. (2023, pp. 807–812, Fig. 11)** para series con no estacionariedades y estado continuo.
- **Discrepancia documentada en literatura:**
  - *Bergmeir et al. (2018, citado en Hewamalage et al., 2023, pp. 809–810)* defienden CV aleatoria en modelos AR puros estacionarios sin autocorrelación residual; sin embargo, *Cerqueira et al. (2020)* y *Oliveira et al. (2021, pp. 3, 21–22)* muestran que en series reales no estacionarias y espaciotemporales la CV estándar subestima severamente el error. **Decisión:** mantener validación causal/prequential (`PtBsR_grW`).
  - *Tashman (2000, p. 442)* sugiere *rolling window* (ventana fija deslizante) para nivelar el tamaño de ajuste al comparar periodos tempranos vs. tardíos, mientras *Hewamalage et al. (2023, p. 808)* señalan que la *expanding window* aprovecha mejor series cortas. **Decisión:** mantener ventana creciente en el OOF actual (dado el tamaño limitado de 2.4 años) y documentar esta elección.

---

## 4. Métricas principales y secundarias
- **Métricas primarias (libres de escala entre plantas):**
  1. **`nMAE_CF` y `nRMSE_CF`** (error sobre Factor de Capacidad `CF = energia_dia / (24 * P_nominal)`, análogo a Linca et al., 2026, Ec. 1, p. 7), evitando el sesgo de escala de `kWh` advertido por **Tashman (2000, p. 445)** y **Hewamalage et al. (2023, p. 802)**.
  2. **`RMSE skill score` (`s_RMSE`)** frente a la **combinación convexa óptima de climatología interna y persistencia a $h$ pasos** (`MSE_cp = (1 - rho_h^2) Var(x)`), siguiendo exactamente a **Yang et al. (2020, pp. 26, 30–31, Ecs. 4 y 17–18)**.
- **Métricas secundarias y de control:**
  - `MAE_macro` en `kWh/día` (solo como métrica operativa de impacto en cartera, nunca como única métrica de comparación entre plantas).
  - `MBE` en `CF` y `kWh/día` como prerrequisito de calibración sin sesgo (Yang et al., 2020, pp. 30, 33).
  - Factorizaciones **Murphy–Winkler** (descomposición sesgo-varianza de MSE, calibración tipo-1/tipo-2 y resolución, Yang et al., 2020, pp. 27–29, Ecs. 12–14, Tabla 3).
- **Discrepancia documentada (MAE vs. RMSE y consistencia L1/L2):**
  - *Yang et al. (2020, pp. 22–23, 29–30)* prefieren `RMSE` porque es consistente con la media condicional y penaliza errores grandes críticos para la red; *Hewamalage et al. (2023, pp. 790, 813, 828)* enfatizan que la pérdida de entrenamiento debe ser coherente con la métrica (`L1 -> MAE`, `L2 -> RMSE`) y proponen métricas híbridas tipo Huber a futuro. **Decisión:** reportar simultáneamente `nMAE_CF` y `nRMSE_CF` (con `s_RMSE` estandarizado) para no favorecer artificialmente un solo tipo de pérdida.

---

## 5. Comparaciones entre modelos
- **Presupuesto equitativo:** La comparación confirmatoria principal frente a los 9 modelos de biblioteca debe hacerse con **`SM-CFHR-Fixed` ($B=0$)** para evitar ventajas asimétricas de HPO (Linca et al., 2026, p. 9), manteniendo `SM-CFHR-HPO` ($B=25$) en una comparación separada explícitamente etiquetada.
- **Eliminación del sesgo *ex post* de `Mejor_Bib`:** No seleccionar al mejor competidor retrospectivamente por escenario para luego aplicar DM-HLN pareado (Tashman, 2000, pp. 446–447). En su lugar:
  1. Fijar **referencias pre-especificadas** (*ex ante*): combinación climatología–persistencia (`CP_opt`), `MM30` y un modelo lineal regularizado fuerte fijado *a priori* (`Ridge` o `Lasso`).
  2. Evaluar la competencia simultánea contra las **9 bibliotecas** mediante **MCS** y **SPA** corregidos, o mediante pruebas pareadas DM-HLN contra cada una con ajuste de Holm.
- **Separación estricta de estrategias multi-paso:** Mantener `Recursivo` y `Directo` en familias separadas (Hewamalage et al., 2023, pp. 800–801).

---

## 6. Procedimientos de inferencia
1. **Diebold–Mariano con HLN y varianza HAC (`dm_hln_hac`):**
   - **Regla de paso temporal:** Usar `horizon_h = 1` cuando la serie $d_t$ esté indexada por orígenes no solapados ($\Delta \text{origen} \ge H$), reservando `horizon_h = H` únicamente si $d_t$ es diaria y se solapa $H-1$ pasos (Harvey et al., 1997; Diebold & Mariano, 1995).
   - **Criterio de tamaño efectivo:** Evaluar $T_{\text{eff}}$ en la **misma unidad** que $d_t$ (exigir $T_{\text{eff,bloques}} \ge 30$ para inferencia asintótica; si $T < 30$, usar bootstrap por bloques o marcar como muestra pequeña indicativa, no inválida por error de unidades).
2. **Model Confidence Set (`MCS`):**
   - Implementar exclusivamente la variante **unilateral** $T_{\max}$ (`mcs_stationary_v2`, piso $1/(B+1)$, $B \ge 2000$), con longitud de bloque estacionario calibrada a la unidad de `time_col` (p. ej. `mean_block = 7` días para `D+1`, y `mean_block = max(2, round(28/H))` bloques para orígenes espaciados cada $H$ días).
3. **Superior Predictive Ability (`SPA`):**
   - Corregir `spa_test` para dividir por el error estándar dependiente HAC/bootstrap (`se`), no por `std(d)` i.i.d.
4. **Intervalos de confianza:**
   - Bootstrap por bloques estacionarios o *moving-block bootstrap* (Linca et al., 2026, p. 19) con longitud de bloque acorde al paso de `time_col`.

---

## 7. Tratamiento de múltiples pruebas
- Aplicar **Holm-Bonferroni** (Holm, 1979) **dentro de cada familia pre-especificada** (`suite × referencia`, `m = 4` u `8` según suite, Hewamalage et al., 2023, p. 824), sin mezclar `Mejor_Bib` con `MM30` ni inflar $m$ con filas no evaluadas.
- Mantener explícitamente la separación entre escenarios que comparten datos: no agrupar los 28 escenarios como si fueran 28 muestras independientes (Oliveira et al., 2021, p. 19).

---

## 8. Evaluación por planta, horizonte y régimen
- **Por horizonte:** Reportar siempre separado `D+1`, `7d`, `14d`, `21d`, `28d` sin mezclar horizontes en un único promedio (Tashman, 2000, pp. 445–446).
- **Por régimen:** Reportar por separado `ET` vs. `IP`, con clima vs. sin clima, y recursivo vs. directo.
- **Por planta:** Analizar heterogeneidad entre las 17 plantas usando `nMAE_CF` y `nRMSE_CF`, verificando sensibilidad geográfica y estacional (Linca et al., 2026, pp. 18–19, 23).

---

## 9. Visualizaciones necesarias
1. **Verificación orientada a distribuciones (Yang et al., 2020, pp. 28–29):** gráficos de densidad conjunta $(f, x)$, comparación de ECDFs marginales y *ridgeline plots* de distribuciones condicionales $p(x|f)$ y $p(f|x)$ para diagnosticar sesgos condicionales.
2. **Resumen comparativo multi-modelo (Hewamalage et al., 2023, pp. 825–826):** diagramas de rango medio / supervivencia en MCS por familia (`ET+Clima`, `SinClima`, `IP`).
3. **Degradación por horizonte y estacionalidad (Linca et al., 2026, pp. 19–20):** curvas de `s_RMSE` y `nMAE_CF` por horizonte `1..28d` con IC por *block bootstrap*, y desglose mensual/estacional (evitando usar gráficos de serie temporal pronóstico vs. real como prueba de superioridad, Hewamalage et al., 2023, pp. 802–804).

---

## 10. Criterios para formular conclusiones
1. **Distinción obligatoria entre victoria descriptiva y significancia estadística:** Reportar ambas por separado (Linca et al., 2026, p. 23).
2. **No declarar superioridad global si es condicionada:** Si `SM-CFHR` domina en `ET + Clima` pero empata o pierde en `IP` y `DirectoSinClima_intraplanta`, la conclusión científica debe acotarse exactamente a ese régimen.
3. **Transparencia ante potencia limitada:** Cuando $T \in [31, 63]$ bloques no alcanza significancia tras Holm, reportar tamaño del efecto, IC por bloques y limitación de longitud muestral sin suavizar umbrales *post-hoc*.

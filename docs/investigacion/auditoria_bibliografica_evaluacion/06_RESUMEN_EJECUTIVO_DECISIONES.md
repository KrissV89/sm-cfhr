# 06. Resumen Ejecutivo para Consultar al revisor técnico

**Proyecto:** SM-CFHR (`sm-cfhr v1.0.3`)  
**Objetivo de esta nota:** Explicar de forma directa qué encontramos al contrastar nuestra evaluación actual contra la literatura científica reciente de pronóstico solar y validación espaciotemporal, y qué decisiones metodológicas conviene acordar en revisión técnica antes de recalcular o reportar resultados finales.

---

## 1. ¿Qué estamos evaluando?
Evaluamos el modelo propio **SM-CFHR** (*Structured Monotone Capacity-Factor Huber Regressor*) en **28 escenarios fotovoltaicos** de 17 plantas en Colombia, combinando:
- Dos tareas: generalización a plantas no vistas (**Espaciotemporal, `ET`**) y pronóstico dentro de cada planta (**Intraplanta, `IP`**).
- Cuatro familias de horizontes y entradas: **`D+1`** (con y sin clima), **Recursivo sin clima** (7–28 días), **Directo sin clima** (7–28 días) y **Recursivo con pronóstico GEFSv12** (7–28 días).
- Dos presupuestos: **`SM-CFHR-Fixed`** ($B=0$ HPO) y **`SM-CFHR-HPO`** ($B=25$ trials), frente a **22 benchmarks** (9 de biblioteca, 12 *custom* y persistencia MM30).

---

## 2. ¿Cuál es nuestro principal problema estadístico?
En los resultados actuales, `SM-CFHR-Fixed` gana descriptivamente en **18/28 escenarios** frente al mejor modelo de biblioteca (y domina claramente en `ET + Clima`), pero solo **2/15 comparaciones evaluables** salen significativas tras Holm. Al auditar el código frente a la literatura, descubrimos que este resultado mezcla **dos fenómenos distintos**:

1. **Un problema real de potencia y alcance por régimen:**
   - En horizontes de 21 y 28 días solo tenemos $T = 31\text{–}41$ orígenes no solapados; como señalan **Linca et al. (2026, p. 23)** y **Hewamalage et al. (2023, p. 826)**, con muestras cortas es normal no alcanzar significancia tras corrección de Holm aunque haya victorias descriptivas frecuentes.
   - Además, `SM-CFHR-Fixed` **no es superior en todos los regímenes**: empata o pierde en **Intraplanta (`IP`)** y falla de forma clara en **`DirectoSinClima_intraplanta`**.

2. **Cuatro errores técnicos (P0) en los scripts de inferencia que distorsionan las pruebas en ambas direcciones:**
   - **Bug 1 (HLN mal escalado):** Nuestros datos de `7d/14d/21d/28d` ya están espaciados cada $H$ días (bloques que **no** se solapan entre sí), pero el código pasa `horizon_h = H` a la corrección HLN (`inferencia_estadistica_v2.py:119`). En $H=28$ ($T=31$), esto multiplica el estadístico DM por `0.112`, encogiéndolo casi **9 veces**. Al corregirlo, aparecen tanto victorias ocultas en `RecGEFSv12_espaciotemporal` ($p=0.597 \to 0.0001$ en 28d) como **derrotas ocultas** en `DirectoSinClima_intraplanta` ($p=0.749 \to 0.030$ en 28d).
   - **Bug 2 (`T_eff >= 2H` en unidades incompatibles):** Se compara $T_{\text{eff}}$ medido en *bloques* ($\le 31$ en 28d) contra $2H$ medido en *días* ($56$). Es algebraicamente imposible cumplirlo en $H=21$ y $H=28$, eliminando `13/28` escenarios.
   - **Bug 3 (`Mejor_Bib` elegido *ex post*):** Escoger al mejor de 9 competidores dentro de la misma muestra OOF y aplicar DM pareado contra él introduce sesgo retrospectivo (**Tashman, 2000, pp. 446–447**).
   - **Bug 4 (`MCS` bilateral y `SPA` sin varianza dependiente):** El MCS oficial compara un estadístico unilateral contra un bootstrap bilateral (`evaluacion_estadistica.py:118`) con solo `B=200`, dejando sobrevivir 15–22 modelos de 23; y `spa_test` (`inference.py:118-122`) ignora el error estándar bootstrap calculado y usa desviación i.i.d.

---

## 3. ¿Qué hacemos actualmente vs. qué recomienda la literatura?

| Aspecto | Qué hacemos actualmente | Qué recomienda la literatura verificada |
|---|---|---|
| **Métrica primaria** | `MAE_macro` en `kWh/día` (y `kWh_total` en directos) | **Evitar métricas dependientes de escala** al promediar series heterogéneas (**Tashman, 2000, p. 445**; **Hewamalage et al., 2023, p. 802**). Usar **Factor de Capacidad (`CF`)** (**Linca et al., 2026, p. 7**) y **`RMSE skill score` contra combinación convexa climatología–persistencia** (**Yang et al., 2020, pp. 30–31**). |
| **Partición OOF** | Causal espaciotemporal creciente (`PtBsR_grW`) | **Correcto:** respaldado por **Oliveira et al. (2021, pp. 21–22, 24)** y **Hewamalage et al. (2023, pp. 807–812)** frente a CV aleatoria optimista. |
| **Diagnóstico de error** | Tablas escalares (`MAE`, `RMSE`, `MBE`) | Complementar con **verificación orientada a distribuciones Murphy–Winkler** (**Yang et al., 2020, pp. 27–29**). |
| **Comparación múltiple** | Holm global sobre 56 filas + `Mejor_Bib` *ex post* | Holm **por familia** pre-especificada (**Holm, 1979**; **Hewamalage et al., 2023, p. 824**), referencia fija *ex ante*, y **MCS/SPA corregidos** sin tratar los 28 escenarios como independientes (**Oliveira et al., 2021, p. 19**). |
| **Evaluación prospectiva nueva** | Datos terminan en `2026-07-31` (`D+1`) / `2026-10-18` (`Rec`) | Un holdout de solo 1–4 meses es insuficiente; se requiere **$\ge 12$ meses nuevos** con pronósticos operacionales archivados pre-emisión, nunca reanálisis ERA5 (**Linca et al., 2026, p. 8**). |

---

## 4. Decisiones concretas para consultar en revisión técnica

1. **Corrección de bugs de inferencia sin reentrenar (P0.1–P0.3, P1.1, P1.4):** ¿Autoriza recalcular las tablas estadísticas de `resultados/evaluacion_ashr_oof/` corrigiendo las unidades de HLN (`horizon_h=1` en bloques disjuntos), el umbral `T_eff`, el MCS unilateral (`B=2000`) y el `se` dependiente en SPA, manteniendo intactas las predicciones OOF y aceptando que aparecerán significativas tanto victorias (`ET+Clima`) como derrotas (`DirectoSinClima_IP`)?
2. **Criterio de superioridad (global vs. condicionada):** Dado que SM-CFHR no gana en `Intraplanta` ni en `DirectoSinClima`, ¿validamos formular la contribución del proyecto como **superioridad condicionada al régimen `Espaciotemporal + Clima`** (generalización a plantas nuevas con pronóstico NWP), en lugar de exigir victorias en los 28 escenarios?
3. **Referencia de comparación justa (`Mejor_Bib` *ex post* vs. benchmark fijo / MCS+SPA):** ¿Reemplazamos la prueba DM contra `Mejor_Bib` elegido *ex post* por comparación contra un benchmark fijo *ex ante* (climatología–persistencia y/o `Ridge`/`Lasso`) complementada con MCS y SPA corregidos?
4. **Métricas libres de escala (`CF` y `s_RMSE`):** ¿Adoptamos `nMAE_CF`, `nRMSE_CF` y el `RMSE skill score` de Yang et al. (2020) como métricas primarias en el informe/manuscrito, dejando `MAE_macro` en `kWh/día` como métrica operativa secundaria? (Todo calculable sin reentrenar).
5. **Alcance y holdout temporal futuro:** Confirmar formalmente el alcance exclusivo en las 17 plantas solares (sin datasets no solares) y definir si el holdout prospectivo ($\ge 12$ meses nuevos con pronósticos archivados) queda como trabajo futuro o extensión post-entrega.

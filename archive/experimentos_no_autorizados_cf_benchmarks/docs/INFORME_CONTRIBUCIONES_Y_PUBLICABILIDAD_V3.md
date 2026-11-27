# Informe de Contribuciones y Publicabilidad (`v3.0.0`)

Este documento responde de forma directa y cuantitativa a las **12 preguntas de evaluación científica** basándose exclusivamente en los artefactos verificados de `v1.0.3`, `v2.0.0` y `v3.0.0`.

## 1. Respuesta a la pregunta central: ¿SM-CFHR mejora frente a convencionales en representación comparable, o la ventaja proviene de la normalización por capacidad?
La ventaja observada inicialmente en transferencia espaciotemporal **proviene principalmente de la normalización causal por factor de capacidad (`24.000 * C_{i,t}`), no de superioridad algorítmica de SM-CFHR frente a lineales regularizados en CF**.
- Al pasar de `kWh_bruto` a `CF` en los 10 escenarios de la Condición A, todas las familias mejoran su `MAE_CF_macro` medio: `XGBoost` (`-23,24%`), `RandomForest` (`-19,06%`), `ElasticNet` (`-17,66%`), `SM-CFHR` (`-14,91%`), `BayesianRidge` (`-14,59%`), `Ridge` (`-13,96%`) y `HuberRegressor` (`-10,13%`).
- Bajo representación `CF` equiparada, `SM-CFHR__CF` empata o queda entre `+0,13%` y `+1,58%` por detrás del mejor competidor en `ET` (alcanzando rango 1 solo en `RecGEFSv12_ET_7d` en Condición A, `-0,56%`, y en `D1_ET_5var` en Condición B, `-0,09%`), y pierde en todos los escenarios `IP` (`+3,25%` a `+16,24%` en Condición A; `+3,23%` a `+10,71%` en Condición B).

## 2. ¿Cuál es entonces la contribución científica principal verificada del trabajo?
La contribución principal demostrada es **la normalización causal por capacidad vigente al origen integrada con un marco de evaluación multihorizonte sin fuga temporal/espacial y una reconciliación inferencial macro-alineada**, acompañada de la formalización convexa y su verificación en Lean 4 como garantía de consistencia matemática (no como ventaja predictiva única sobre `BayesianRidge-CF` o `ElasticNet-CF`).

## 3. ¿Cómo se comportó SM-CFHR frente a cada par en CF bajo inferencia pareada Holm (Condición A, 10 escenarios)?
- Frente a `HuberRegressor__CF`: `4` victorias Holm, `0` derrotas (`9/10` descriptivas).
- Frente a `Ridge__CF`: `4` victorias Holm, `0` derrotas (`9/10` descriptivas).
- Frente a `BayesianRidge__CF`: `2` victorias Holm vs `3` derrotas Holm (`6/10` descriptivas).
- Frente a `ElasticNet__CF`: `3` victorias Holm vs `3` derrotas Holm (`3/10` descriptivas).
- Frente a `RandomForest__CF`: `3` victorias Holm vs `2` derrotas Holm (`6/10` descriptivas).
- Frente a `XGBoost__CF`: `2` victorias Holm vs `1` derrota Holm (`7/10` descriptivas).

## 4. ¿Qué componentes aportan mejora verificada en las ablaciones controladas de 6 escenarios?
Según `veredictos_componentes_v3.csv`:
- **Mejora observada consistente:** normalización causal FC (`+25,86%` medio al retirarla, rango `[+1,25%, +62,65%]`), variables NWP (`+18,52%`, `[+2,05%, +33,64%]`) y anomalías meteorológicas causales (`+5,61%`, `[+1,66%, +10,63%]`).
- **Resultado dependiente del escenario:** monotonía (`+2,26%` medio, `[0,00%, +7,77%]`), regularización anisotrópica vs isotrópica `0,01` (`+3,79%`, `[-0,20%, +13,42%]`) y vs `0,10` (`-0,99%`, `[-5,52%, +0,99%]`), pérdida de Huber vs `L2` (`+0,31%`, `[+0,05%, +0,68%]`), y escalado MAD vs STD (`+0,19%`, `[-0,02%, +1,02%]`).

## 5. ¿Con qué frecuencia se activan realmente las restricciones de monotonía?
Según `auditoria_monotonia_activacion.csv`:
- `D1_espaciotemporal_cinco_variables` (`20` pliegues): `0,00%` `cf30`, `0,00%` `rad`.
- `D1_espaciotemporal_sin_meteorologia` (`20` pliegues): `0,00%` `cf30`.
- `RecGEFSv12_espaciotemporal_7d` (`746` pliegues): `2,41%` `cf30`, `0,00%` `rad`.
- `RecGEFSv12_espaciotemporal_28d` (`180` pliegues): `1,67%` `cf30`, `0,00%` `rad`.
- `D1_intraplanta_cinco_variables` (`49` pliegues): `26,53%` `cf30`, `2,04%` `rad` (donde evita degradación de `+7,77%`).
- `DirectoSinClima_intraplanta_28d` (`350` pliegues): `56,00%` `cf30` (`+5,58%` al retirarla).

## 6. ¿La pérdida de Huber protege en operación real y ante apalancamiento en covariables?
Según `auditoria_huber_operacional_y_estres.csv`:
- En operación real (`4.312` obs.), Huber y `L2` son casi idénticos (`0,03477` vs `0,03481` CF); en `sy < 0,3` (`11` obs.) y cola `5%` (`216` obs.), `L2` es marginalmente menor (`0,17477` vs `0,17517`).
- Bajo contaminación sintética del objetivo al `10%`, Huber mantiene `0,03432` CF frente a `0,03760` en `L2`.
- Bajo apalancamiento sintético del `5%` en `rad_pron`, **ambos se degradan por igual** (`0,03664` y `0,03671` vs `0,03430` limpio), confirmando que Huber no protege ante apalancamiento en `X`.

## 7. ¿Cuál es el impacto del sesgo de selección por el filtro `horas_xm <= 6`?
Según `sensibilidad_sesgo_seleccion_horas_xm.csv`, al reincorporar los días observados con `horas_xm <= 6` sin imputación:
- `D1_espaciotemporal_cinco_variables`: `MAE_CF_macro` pasa de `0,035350` a `0,035643` (`+0,83%`).
- `D1_intraplanta_cinco_variables`: `MAE_CF_macro` pasa de `0,038922` a `0,039031` (`+0,28%`).

## 8. ¿Cómo cambia la evaluación al pasar de `MAE_macro (kWh)` a `MAE_CF_macro`?
En las `28` configuraciones congeladas (`v2.0.0`), el rango medio de `SM-CFHR-Fixed` pasa de `2,93/23` (`kWh`) a `4,00/23` (`CF`), y las victorias frente a `Mejor_Bib` de `18/28` a `16/28`, mostrando que `kWh` pondera fuertemente centrales grandes (`GYPO`, `370–388 MW`).

## 9. ¿Qué se concluye sobre `SM-CFHR-Fixed (B=0)` vs `SM-CFHR-HPO (B=25)`?
`SM-CFHR-Fixed` vence a `HPO` en `19/28` escenarios (`14/14` en ET; `4` victorias Holm vs `1` derrota) con un coste computacional `59x` menor (`499,75` s vs `29.550,3` s). La hipótesis H4 a favor de `HPO` queda refutada.

## 10. ¿Qué sucede con la incertidumbre conforme temporal?
Con `n_cal = 1.720`, los intervalos conformes presentan subcobertura bajo deriva temporal: `83,24%` para nominal `90%` (semiancho `0,0662` CF) y `69,25%` para nominal `80%` (semiancho `0,0480` CF).

## 11. ¿El régimen ET demuestra generalización *zero-shot* o transferibilidad de grupos con historial local?
Demuestra **transferibilidad entre grupos espaciales con historial causal propio de la planta** (`cf30`, `cf7`, `cf1` disponibles al origen), **no** *zero-shot* sin historial ni generalización fuera de las `17` centrales colombianas (`G=7` zonas).

## 12. Diagnóstico honesto de publicabilidad
Con el reencuadre `v3.0.0`, el manuscrito es **publicable en una revista técnica de energía solar o pronóstico energético aplicado** (p. ej., *Solar Energy*, *Renewable Energy*, *Applied Energy* o *International Journal of Forecasting* en modalidad metodológica/empírica aplicada) siempre que se presente como contribución principal **el marco de normalización causal en factor de capacidad y evaluación multihorizonte anti-fuga**, reportando con total transparencia el empate/paridad con `BayesianRidge-CF` y `ElasticNet-CF`, las derrotas en `IP`, la reutilización adaptativa del panel 2024–2026 y la vigencia de `alcance_solar_exclusivo: sin datasets externos`. Un envío que siguiera afirmando superioridad algorítmica intrínseca de SM-CFHR sería rechazable ante revisión por pares.

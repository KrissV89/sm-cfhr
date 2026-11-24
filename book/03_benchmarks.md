---
title: Modelos benchmark
short_title: 3. Benchmarks
kernelspec:
  name: python3
  display_name: Python 3
---

# Modelos benchmark

## Modelos de referencia

Las referencias sin aprendizaje de la Tarea 2 son la persistencia ($\hat y_{p,T}=y_{p,D}$), las medias móviles de 7 y 30 días, la climatología mensual del FC de la planta en el entrenamiento y la mediana global del entrenamiento. Las referencias con aprendizaje se citan sin derivarlas:

- **Ridge** y **Lasso** minimizan $\sum_i (y_i-\beta_0-x_i^\top\beta)^2+\lambda R(\beta)$ con $R(\beta)=\lVert\beta\rVert_2^2$ y $\lVert\beta\rVert_1$ {cite:p}`hoerl1970ridge,tibshirani1996lasso`; la **regresión Ridge bayesiana** estima además las precisiones del ruido y de los coeficientes por máxima evidencia {cite:p}`mackay1992bayesian`.
- **$k$ vecinos más cercanos** promedia la respuesta de los $k$ puntos de entrenamiento más próximos, con pesos uniformes o inversos a la distancia {cite:p}`cover1967nearest`.
- **SVR** con kernel gaussiano ajusta $f(x)=w^\top\phi(x)+b$ minimizando $\tfrac12\lVert w\rVert^2+C\sum_i\max(0,|y_i-f(x_i)|-\varepsilon)$ {cite:p}`drucker1997svr`.
- El **árbol de decisión** particiona el espacio por reducción de varianza {cite:p}`breiman1984cart`; **Random Forest** promedia árboles ajustados sobre muestras bootstrap con un subconjunto aleatorio de variables en cada división {cite:p}`breiman2001rf`.
- **HistGradientBoosting** y **XGBoost** construyen un ensamble aditivo de árboles por descenso de gradiente sobre una pérdida regularizada {cite:p}`friedman2001gbm,chen2016xgboost`.

Se usan las implementaciones de scikit-learn 1.9.1 {cite:p}`pedregosa2011sklearn` y XGBoost 3.4.1. Lasso, la Ridge bayesiana y el árbol de decisión no formaban parte del benchmark $D{+}1$ original; se añadieron después con el mismo protocolo (cuaderno 20) para completar los siete modelos base de la guía del Entregable 2.

Todos los modelos de librería reciben el conjunto de predictores $B_2$: el FC de $D$, las medias móviles de 7 y 30 días ([ecuación %s](#eq-nivel)) y las cinco variables pronosticadas para $T$. Los conjuntos alternativos son $B_0$ (solo historial), $B_1$ (historial e irradiación pronosticada), $B_{1r}$ ($B_1$ con el historial retrasado un día) y $B_3$ (solo irradiación pronosticada); todos se evalúan sobre las mismas filas.

**Métodos recientes del estado del arte.** La guía pide al menos dos métodos recientes para la misma tarea. Los modelos de árboles con gradiente (XGBoost, HistGradientBoosting) son los más usados en la literatura reciente de pronóstico fotovoltaico con aprendizaje clásico ([capítulo 4](04_literatura.md)), pero no son métodos recientes. Los métodos recientes del área que aparecen en la búsqueda bibliográfica son en su mayoría redes neuronales, que el proyecto excluyó por diseño (aprendizaje clásico). Las once variantes de posprocesamiento de la ronda de candidatos ([capítulo 5](05_modelo_original.md)), como el MOS con ventana deslizante, el filtro de Kalman o la combinación de pronósticos, provienen de la literatura de posprocesamiento meteorológico y son las comparaciones más cercanas a K1, pero se implementaron en el proyecto y no reproducen un método publicado concreto. Este requisito de la guía no se cumple.

## Hiperparámetros, presupuesto y costo

Las rejillas de la [Tabla %s](#tbl-rejillas) se fijaron antes de ejecutar cada cuaderno y se usan en todos los bloques. La búsqueda interna es exhaustiva en rejilla (`GridSearchCV`) y minimiza el MAE por fila con tres particiones temporales de ventana creciente y un hueco de 30 días entre entrenamiento y validación ([capítulo 6](06_validacion.md)). Los modelos de árboles usan 300 árboles y la semilla 2026.

:::{table} Rejillas de la Tarea 2 (cuadernos 07, 17 y 20). La última columna es el número de combinaciones evaluadas en cada búsqueda.
:label: tbl-rejillas

| Modelo | Rejilla | Comb. |
|---|---|---:|
| Ridge | $\alpha\in\{10^{-3},\dots,10^{3}\}$ (7 valores) | 7 |
| Lasso | $\alpha\in\{10^{-5},\dots,10^{-1}\}$ (7 valores) | 7 |
| Ridge bayesiana | $\alpha_1,\lambda_1\in\{10^{-6},10^{-4}\}$ | 4 |
| $k$ vecinos | $k\in\{5,15,30,60\}$; pesos uniformes o por distancia | 8 |
| SVR (RBF) | $C\in\{0.1,1,10\}$; $\gamma\in\{\text{scale},0.1\}$; $\varepsilon\in\{0.01,0.03\}$ | 12 |
| Árbol de decisión | profundidad $\in\{3,5,8,\text{sin límite}\}$; hoja mínima $\in\{10,30,100\}$ | 12 |
| Random Forest | hoja mínima $\in\{3,10,30\}$; fracción de variables $\in\{0.5,1\}$ | 6 |
| HistGradientBoosting | tasa $\in\{0.03,0.1\}$; hojas $\in\{7,15,31\}$; L2 $\in\{0,1\}$ | 12 |
| XGBoost | profundidad $\in\{3,5\}$; tasa $\in\{0.03,0.1\}$; peso mínimo $\in\{1,10\}$; submuestreo 0.8 | 8 |
| K1 | $\alpha\in\{10^{-3},\dots,10^{3}\}$ (7 valores) | 7 |
| W1 | $h\in\{0.5,1,2\}$; $(\rho_g,\rho_o)\in\{(0.5,0.2),(0.3,0.1)\}$; $\gamma=10^{-3}$ fijo | 6 |
:::

El presupuesto no es igual entre modelos: va de 4 a 12 combinaciones. Esa diferencia es una de las razones por las que la comparación no cumple el requisito de equidad de la guía ([capítulo 6](06_validacion.md)); el estudio de optimizadores del cuaderno 21 iguala el presupuesto en 30 evaluaciones por modelo ([sección %s](#sec-optimizadores)). La [Tabla %s](#tbl-costo) muestra los hiperparámetros que eligió la búsqueda en el bloque de prueba y el tiempo de reloj de cada búsqueda.

::::{table} Hiperparámetros elegidos en el bloque de prueba (entrenamiento hasta el 31 de julio de 2026) y tiempo de búsqueda, conjunto $B_2$. Fuente: `resultados/pruebas_guia/costo_computacional.csv`, que reúne los registros de los cuadernos 07 y 20.
:label: tbl-costo
```{code-cell} python
# Hiperparámetros y costo de la búsqueda en rejilla. Los tiempos son de reloj con varias tareas en paralelo
# en la misma CPU (i5-12450H, 12 hilos); sirven para comparar órdenes de magnitud, no como medida exacta.
import apoyo
import pandas as pd

costo = apoyo.leer("resultados/pruebas_guia/costo_computacional.csv")
costo = costo[costo.origen.str.contains("B2")]               # modelos del benchmark D+1 (no la ronda de candidatos)
cols = {"modelo": "modelo", "combinaciones_rejilla": "comb. rejilla", "ajustes_por_bloque": "ajustes por bloque",
        "seg_desarrollo_total": "s búsqueda D1–D3", "seg_FINAL": "s búsqueda prueba",
        "hiperparametros_FINAL": "hiperparámetros elegidos (prueba)"}
apoyo.tabla(costo[list(cols)].rename(columns=cols), decimales=1, indice="modelo")
```
::::

## Tarea 1: estimación retrospectiva en plantas no vistas

En la Tarea 1 la evaluación deja fuera un grupo espacial completo a la vez (siete particiones externas) y elige los hiperparámetros con validación cruzada agrupada dentro de cada entrenamiento. La mediana del entrenamiento tiene un MAE de 0.055 y Ridge con la irradiación como único predictor lo reduce a 0.0429, un 22 % ([Figura %s](#fig-tarea1), [Tabla %s](#tbl-tarea1)). Añadir variables a la irradiación no reduce el error; con las cinco variables, Ridge empeora a 0.048 y su error se concentra en una planta (Sunnorte, MAE 0.094). Las diferencias entre los modelos no lineales (0.043 a 0.045) son menores que la desviación estándar del MAE entre grupos (0.004 a 0.006). Ninguno de los cinco ensambles ni de las cinco adaptaciones propias (C1–C5) mejora a Ridge con irradiación: el mejor, el árbol de modelos lineales C3, lo iguala a cuatro decimales.

:::{figure} figuras/res_tarea1.png
:label: fig-tarea1
:width: 100%
Tarea 1: MAE macro (unidades de FC) en grupos espaciales excluidos por modelo y conjunto de predictores (librería, ensambles, adaptaciones propias C1–C5 y mediana del entrenamiento). La línea marca a Ridge con irradiación.
:::

::::{table} Tarea 1: mejor modelo de cada conjunto de predictores y MAE de la mediana del entrenamiento, en grupos espaciales excluidos. Fuente: `resultados/benchmark/resumen_macro.csv` (cuaderno 03) y `resultados/modelos_propuestos/resumen_macro.csv` (cuaderno 04).
:label: tbl-tarea1
```{code-cell} python
# Para cada conjunto de predictores: mejor modelo de librería o ensamble, mejor adaptación propia (C1–C5) y mediana.
lib = apoyo.leer("resultados/benchmark/resumen_macro.csv")
prop = apoyo.leer("resultados/modelos_propuestos/resumen_macro.csv")
prop = prop[prop.modelo.str.match(r"C\d")]
filas = []
for conj, g in lib.groupby("conjunto"):
    g_ref = g[g.modelo != "Mediana"].nsmallest(1, "MAE_macro").iloc[0]
    g_prop = prop[prop.conjunto == conj].nsmallest(1, "MAE_macro").iloc[0]
    filas.append({"conjunto": conj,
                  "mejor librería o ensamble": g_ref.modelo, "MAE": g_ref.MAE_macro,
                  "d.e. entre grupos": g_ref.MAE_desv_entre_grupos,
                  "mejor propia": g_prop.modelo, "MAE propia": g_prop.MAE_macro,
                  "mediana": g[g.modelo == "Mediana"].MAE_macro.iloc[0]})
apoyo.tabla(pd.DataFrame(filas).sort_values("MAE"), decimales=4, indice="conjunto")
```
::::

El sesgo promedio entre plantas es pequeño (0.003 con Ridge e irradiación), pero el valor absoluto medio del sesgo por planta es 0.026, más de la mitad del MAE, y el $R^2$ por planta tiene mediana de 0.31. El error de la Tarea 1 está dominado por el nivel de cada planta, que la meteorología no explica ([capítulo 2](02_datos.md)). Esa lectura motivó la Tarea 2.

## Tarea 2: selección en desarrollo

En los tres bloques de desarrollo (D1: noviembre de 2025 a enero de 2026; D2: febrero a abril; D3: mayo a julio), el historial solo ($B_0$) da MAE de 0.045 a 0.046; añadir la irradiación pronosticada ($B_1$) lo baja a 0.037–0.038, y las cinco variables ($B_2$) a 0.036–0.038 en los modelos de librería ([Tabla %s](#tbl-desarrollo-conjuntos)). La irradiación pronosticada sola ($B_3$) queda en 0.042. Retrasar el historial un día ($B_{1r}$) cuesta entre 0.0001 y 0.0007. Con $B_2$, la mejor referencia de desarrollo es Random Forest (0.0363), y K1 obtiene 0.0347. La comparación decisiva quedó fijada como K1 frente a Random Forest en `resultados/pronostico_d1/seleccion_desarrollo.json` antes de ejecutar la prueba.

::::{table} MAE macro de desarrollo (predicciones de D1–D3 agrupadas) por modelo y conjunto de predictores. Fuente: `resultados/pronostico_d1/desarrollo_resumen_macro.csv` (cuaderno 07) y `resultados/referencias_curso/resumen_macro.csv` (cuaderno 20).
:label: tbl-desarrollo-conjuntos
```{code-cell} python
# Tabla modelo × conjunto de predictores en desarrollo. Las referencias del cuaderno 20 se añaden con el mismo protocolo.
d07 = apoyo.leer("resultados/pronostico_d1/desarrollo_resumen_macro.csv")
d20 = apoyo.leer("resultados/referencias_curso/resumen_macro.csv")
d20 = d20[d20.periodo == "desarrollo"]
dev = pd.concat([d07[d07.conjunto != "Sin aprendizaje"], d20], ignore_index=True)
piv = dev.pivot_table(index="modelo", columns="conjunto", values="MAE_macro")[["B0", "B1", "B1r", "B2", "B3"]]
apoyo.tabla(piv.sort_values("B2").reset_index(), decimales=4, indice="modelo")
```
::::

La ronda posterior evaluó once candidatos que modifican un componente de K1, con 107 combinaciones de hiperparámetros, solo en desarrollo ([Figura %s](#fig-candidatos)). W1 obtuvo el menor MAE (0.0341), seguido de las pendientes aleatorias H1 (0.0344) y de la corrección con el error del día anterior R1 (0.0345); seis de los once candidatos mejoraron a K1 (0.0347), ninguno por más de 0.0006, y todos mejoraron a Random Forest y a XGBoost. La diferencia entre los candidatos es menor que la diferencia entre bloques (de 0.030 en D1 a 0.038 en D3 para W1).

:::{figure} figuras/res_candidatos.png
:label: fig-candidatos
:width: 100%
MAE macro de desarrollo (unidades de FC) de los once candidatos de la ronda en la que se seleccionó W1, frente a K1 y a las mejores referencias de librería (a), y MAE por bloque D1–D3 (b).
:::

## Tarea 2: resultados de las mejores combinaciones

La [Tabla %s](#tbl-principal) es la tabla resumen que pide la guía: MAE del bucle externo (media ± desviación estándar de los bloques D1–D3, y MAE de las predicciones agrupadas), MAE de la prueba con intervalo bootstrap, métricas secundarias y valor $p$ del MCS. Cada modelo aparece con la combinación de la búsqueda en rejilla original, fijada antes de la prueba. La [sección %s](#sec-optimizadores) compara después cuatro optimizadores con presupuesto igualado; es un estudio *post hoc* y no sustituye esta tabla. La columna de desarrollo «media ± d.e.» promedia los tres MAE macro de D1–D3; «agrupado» calcula el MAE macro con las predicciones de los tres bloques juntas.

::::{table} Tarea 2, conjunto $B_2$: MAE del bucle externo (D1–D3), MAE de la prueba (agosto a octubre de 2026) con IC 95 % por bootstrap de bloques de 7 días, RMSE, sesgo (MBE = predicción − observación), $R^2$ y MASE de la prueba, y $p$ del MCS con W1 en el universo. W1 recibe además la identidad de planta y de grupo. Fuente: `resultados/pruebas_guia/tabla_desempeno.csv` y `desarrollo_candidatos.csv` (script `scripts/pruebas_guia.py`).
:label: tbl-principal
```{code-cell} python
# Tabla resumen: desarrollo (media ± d.e. de los tres bloques y MAE de las predicciones agrupadas) y prueba.
des = apoyo.leer("resultados/pruebas_guia/desarrollo_candidatos.csv")
des["clave"] = des.modelo
des["MAE D1–D3 (media ± d.e.)"] = des[["D1", "D2", "D3"]].apply(lambda r: f"{r.mean():.4f} ± {r.std(ddof=1):.4f}", axis=1)
des = des.rename(columns={"MAE_macro": "MAE des. agrupado"})[["clave", "MAE des. agrupado", "MAE D1–D3 (media ± d.e.)"]]

pru = apoyo.leer("resultados/pruebas_guia/tabla_desempeno.csv")
pru = pru[pru.principal & pru.tipo.isin(["propuesto", "librería", "sin aprendizaje"])]
pru["IC 95 %"] = pru.apply(lambda r: f"[{r.MAE_ic95_inf:.4f}, {r.MAE_ic95_sup:.4f}]", axis=1)
pru = pru.rename(columns={"MAE_macro": "MAE prueba", "RMSE_macro": "RMSE", "MBE_macro": "MBE", "R2_macro": "R²",
                          "MASE_macro": "MASE", "p_mcs_con_w1": "p MCS"})
principal = pru.merge(des, on="clave", how="left")[
    ["modelo", "MAE D1–D3 (media ± d.e.)", "MAE des. agrupado", "MAE prueba", "IC 95 %", "RMSE", "MBE", "R²", "MASE", "p MCS"]]
apoyo.tabla(principal.sort_values("MAE prueba"), decimales={"MAE des. agrupado": 4, "MAE prueba": 4, "RMSE": 4,
            "MBE": 4, "R²": 3, "MASE": 3, "p MCS": 3}, indice="modelo")
```
::::

En la prueba, K1 obtiene 0.0392 frente a 0.0406 de Random Forest (3.3 % menos), con mejor RMSE y menor sesgo absoluto, y W1 obtiene 0.0388. Todos los modelos con aprendizaje mejoran a las referencias sin aprendizaje. SVR (0.0403) supera a Random Forest en la prueba, a diferencia de lo ocurrido en desarrollo. Los intervalos del MAE se solapan entre todos los modelos con aprendizaje ([Figura %s](#fig-comparacion)); si esas diferencias son significativas se analiza en el [capítulo 6](06_validacion.md). La desviación estándar entre bloques (0.004 en casi todos los modelos) es mayor que la diferencia entre K1 y Random Forest, lo que anticipa que una sola ventana de prueba tiene poca potencia para separarlos.

:::{figure} figuras/res_comparacion_modelos.png
:label: fig-comparacion
:width: 100%
MAE macro (unidades de FC) en la prueba (agosto a octubre de 2026; marcador lleno, IC 95 % por bootstrap de bloques de 7 días) y en desarrollo (media de D1–D3; marcador hueco), conjunto $B_2$. Se distinguen los modelos propuestos, las demás adaptaciones propias (C1–C4), los modelos de librería y las referencias sin aprendizaje. La línea punteada es Random Forest en la prueba.
:::

Los artefactos del dashboard (`artifacts/`) se reentrenaron con estos hiperparámetros y reproducen las mismas predicciones de prueba (prueba de ida y vuelta en `artifacts/indice.json`). Sus métricas del bucle externo están en `data/leaderboard.csv`, que lee la tabla de líderes del dashboard ([capítulo 8](08_dashboard.md)).

(sec-optimizadores)=
## Mejores combinaciones por optimizador (cuaderno 21)

La guía pide, para cada uno de los siete modelos base, la mejor combinación entre cuatro métodos de optimización según el MAE medio del bucle externo, una tabla maestra con el motivo de la selección y la verificación de que la mejor combinación global está entre los finalistas del MCS. El protocolo del cuaderno 21 (`resultados/optimizadores/PROTOCOLO.md`) se registró con su huella SHA-256 el 16 de noviembre de 2026 a las 00:39 (UTC−5), antes de ejecutar ninguna búsqueda. Fija:

- **Modelos:** Ridge, Lasso, $k$ vecinos, SVR, árbol de decisión, Random Forest y XGBoost, más K1 y W1 con el mismo presupuesto.
- **Optimizadores:** rejilla, búsqueda aleatoria, bayesiana con TPE de Optuna {cite:p}`bergstra2011tpe,akiba2019optuna` y algoritmo genético con DEAP {cite:p}`fortin2012deap` (población de 6, elitismo de 1, torneo de 2, cruce SBX y mutación polinómica), con 30 evaluaciones de la función objetivo por modelo, optimizador y bloque externo (32 en la rejilla de XGBoost). Los cuatro comparten dominio y escala.
- **Validación anidada:** bucle externo D1, D2, D3 y prueba; bucle interno con las tres particiones temporales con hueco de 30 días. La mejor combinación por modelo es la de menor MAE de desarrollo medio (media de los tres MAE macro de D1–D3); la prueba no interviene en la selección.
- **Carácter:** *post hoc*, porque la ventana de prueba ya se había usado en los cuadernos 07 y 19. Las desviaciones se anotan en `resultados/optimizadores/DESVIACIONES.md`: una interrupción antes de calcular nada y la importación de K1 y W1 desde `src/original_model.py` para poder serializarlos, con la misma lógica numérica.

La ejecución terminó el 16 de noviembre de 2026: 144 búsquedas principales en 15.7 minutos y 180 del estudio de semillas en 20.7 minutos, con cuatro procesos. K1 y W1 reprodujeron las predicciones guardadas de 07 y 19 (diferencias de 0 y $1.7\times10^{-16}$) y los archivos de esos cuadernos no cambiaron. La lectura completa está en la última celda del cuaderno.

::::{table} Mejor combinación de cada modelo (criterio: menor MAE de desarrollo medio). MAE de desarrollo como media ± desviación estándar de D1–D3; MAE de la prueba (descriptivo: la prueba ya se conocía); rango del MAE entre los cuatro optimizadores; tiempo total de búsqueda en los cuatro bloques. Fuente: `resultados/optimizadores/tabla_maestra.csv` y `mejores_combinaciones.csv`.
:label: tbl-optimizadores
```{code-cell} python
# Mejor combinación por modelo y rango entre optimizadores (cuaderno 21). Solo lee archivos guardados.
maestra = apoyo.leer("resultados/optimizadores/tabla_maestra.csv")
mejores = apoyo.leer("resultados/optimizadores/mejores_combinaciones.csv")
rango = maestra.groupby("modelo").agg(rango_des=("MAE_dev_media", lambda s: s.max() - s.min()),
                                      rango_prueba=("MAE_prueba", lambda s: s.max() - s.min()))
t = mejores.merge(rango, left_on="modelo", right_index=True)
t["MAE des. (media ± d.e.)"] = [f"{m:.4f} ± {d:.4f}" for m, d in zip(t.MAE_dev_media, t.MAE_dev_de)]
t = t.rename(columns={"MAE_prueba": "MAE prueba", "rango_des": "rango des.", "rango_prueba": "rango prueba",
                      "segundos_busqueda_total": "búsqueda (s)", "hp_FINAL": "hiperparámetros (bloque de prueba)"})
cols = ["modelo", "tipo", "optimizador", "MAE des. (media ± d.e.)", "MAE prueba", "rango des.", "rango prueba", "búsqueda (s)",
        "hiperparámetros (bloque de prueba)"]
apoyo.tabla(t[cols].sort_values("MAE prueba"), decimales={"MAE prueba": 4, "rango des.": 5, "rango prueba": 5, "búsqueda (s)": 0},
            indice="modelo")
```
::::

**Error.** El optimizador casi no cambia el error externo. Dentro de cada modelo, la diferencia entre el mejor y el peor optimizador va de 0.00001 (Ridge, K1 y W1) a 0.0009 (árbol de decisión) en desarrollo, mientras la desviación estándar entre bloques es de 0.003 a 0.005. Ningún optimizador gana en todos los modelos: la rejilla es la mejor en cuatro, la aleatoria en tres y la genética en dos. La mejor combinación global de librería es Random Forest con búsqueda aleatoria (desarrollo 0.0354, prueba 0.0405). K1 y W1 quedan por debajo de las 28 combinaciones de librería en desarrollo y en la prueba, con cifras de prueba casi iguales a las de las rejillas originales ([Tabla %s](#tbl-principal)). Esa tabla usa el MAE de las predicciones de D1–D3 agrupadas; con esa misma agregación el estudio da 0.0347 para K1, 0.0342 para W1 y 0.0362 para Random Forest.

**MCS.** El MCS se calculó con las predicciones de las rejillas originales ([capítulo 6](06_validacion.md)): Random Forest permanece en el conjunto al 90 % ($p_{\mathrm{MCS}}=0.152$). No se recalculó con las predicciones del cuaderno 21: que la mejor combinación de ese estudio también esté entre los finalistas no está verificado.

:::{figure} figuras/res_optimizadores.png
:label: fig-optimizadores
:width: 100%
Comparación de optimizadores. (a) MAE macro de desarrollo por modelo y optimizador (media y desviación estándar de D1–D3). (b) Exceso relativo del mejor MAE interno acumulado sobre el mejor valor encontrado por cualquier optimizador, en función del número de evaluaciones, promediado sobre 9 modelos y 4 bloques externos (escala logarítmica; en la rejilla la curva depende del orden de evaluación).
:::

**Convergencia y costo.** Sobre la función objetivo interna, el optimizador bayesiano es el que más se acerca al mejor valor encontrado ([Figura %s](#fig-optimizadores)b): tras 30 evaluaciones su exceso medio es 0.04 %, queda a menos de 0.5 % en 35 de 36 casos modelo × bloque y encuentra el mejor valor en 20. La aleatoria y la genética bajan antes del 1.5 % pero se estancan en 0.3 y 0.6 %; la rejilla avanza despacio por su orden lexicográfico y termina en 0.3 %. En XGBoost, el modelo con cinco hiperparámetros, la ventaja del bayesiano es más clara. Estas diferencias de décimas de punto en el objetivo interno no se trasladan al error externo. La búsqueda genética fue la más barata en tiempo total (671 s) y la rejilla la más cara (1,215 s), sobre todo por SVR. Los tiempos se midieron con otros trabajos en la máquina.

```{code-cell} python
# Convergencia por optimizador: exceso relativo medio (%) a 5, 10, 20 y 30 evaluaciones sobre 36 casos modelo × bloque.
conv = apoyo.leer("resultados/optimizadores/convergencia_por_optimizador.csv")
apoyo.tabla(conv, decimales=3, indice="optimizador")
```

**Hiperparámetros y semillas.** W1 eligió $h$ entre 7.1 y 7.9 con $(\rho_g,\rho_o)=(0.3,0.1)$ en D1, D3 y la prueba, y entre 2.8 y 3.6 con $(0.5,0.2)$ en D2: el óptimo quedó cerca del borde superior del dominio ($h\le8$). Con diez semillas, la desviación estándar del MAE de desarrollo de las combinaciones estocásticas va de 0.00001 a 0.0003, y ninguna semilla de ningún modelo de librería alcanza a K1 o W1. En XGBoost, Lasso y el árbol, la semilla 2026 fue la mejor de las diez en desarrollo, de modo que el MAE de la combinación elegida es optimista frente a la media de semillas.

```{code-cell} python
# Estudio de semillas (2026-2035) de las combinaciones estocásticas.
sem = apoyo.leer("resultados/optimizadores/semillas_resumen.csv")
cols = ["modelo", "optimizador", "variacion", "MAE_dev_media", "MAE_dev_de_semillas", "MAE_dev_min", "MAE_dev_max",
        "MAE_prueba_media", "MAE_prueba_de_semillas"]
apoyo.tabla(sem[cols], decimales=5, indice="modelo")
```

(sec-interpretabilidad)=
### Interpretabilidad: SHAP y LIME

Se explicaron 400 filas de la prueba con SHAP {cite:p}`lundberg2017shap` para Random Forest (mejor combinación de librería, explicador exacto para árboles), K1 y W1 (explicador por permutaciones), y tres filas (menor, mediano y mayor error de Random Forest) con LIME {cite:p}`ribeiro2016lime`. Los tres modelos se apoyan en el nivel reciente de la planta y en la radiación pronosticada ([Figura %s](#fig-interpretabilidad)). En K1, lineal en sus entradas, la atribución por permutaciones coincide con la analítica (diferencia máxima $2\times10^{-8}$). La baja importancia SHAP de la identidad en W1 (2.3 %) no significa que no influya: entra como peso de la regresión local, no como término aditivo, y el muestreo independiente de SHAP puede combinar una planta con un grupo que no es el suyo.

:::{figure} figuras/res_interpretabilidad.png
:label: fig-interpretabilidad
:width: 100%
SHAP en 400 filas de la prueba. (a) Fracción de la media de $|\mathrm{SHAP}|$ por grupo de variables para Random Forest, K1 y W1. (b) Contribución SHAP de la radiación pronosticada (Random Forest) o de su anomalía (K1 y W1) frente a su valor.
:::

::::{table} Concordancia entre SHAP y LIME en tres filas de la prueba. «Relevantes»: variables con $|\mathrm{SHAP}|$ de al menos el 5 % del total de la fila; «mismo signo»: fracción de esas variables con igual signo en ambos métodos; Spearman entre $|\mathrm{SHAP}|$ y $|\text{peso LIME}|$; $R^2$ del modelo local de LIME. Fuente: `resultados/optimizadores/concordancia_shap_lime.csv`.
:label: tbl-shap-lime
```{code-cell} python
conc = apoyo.leer("resultados/optimizadores/concordancia_shap_lime.csv")
apoyo.tabla(conc, decimales=3, indice="explicado")
```
::::

SHAP y LIME coinciden en el signo de 33 de 34 variables relevantes, con más concordancia en K1 y W1 que en Random Forest. El $R^2$ de los modelos locales de LIME es bajo (0.27–0.63) y tres filas no estiman la concordancia general. La fila de mayor error es un día con FC observado de 0.0003, que los tres modelos pronostican entre 0.19 y 0.23: una indisponibilidad o un registro incompleto que ninguna variable disponible anticipa.

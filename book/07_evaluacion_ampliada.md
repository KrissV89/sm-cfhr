---
title: Evaluación ampliada y robustez
short_title: 7. Evaluación ampliada
kernelspec:
  name: python3
  display_name: Python 3
---

# Evaluación ampliada y robustez

Este capítulo reúne las evaluaciones que van más allá de la prueba principal: transferencia a plantas no vistas y nuevas, tres conjuntos de datos públicos adicionales, sensibilidad a hiperparámetros, semillas, ruido, tamaño de muestra, costo e intervalos de predicción. Salvo que se indique, los errores son MAE macro por planta (o por unidad) y los estudios usan solo los bloques de desarrollo D1–D3. Todos son posteriores a la selección: describen el comportamiento de los modelos ya elegidos.

```{code-cell} python
import apoyo
import pandas as pd
from IPython.display import display
```

## Plantas no vistas y plantas nuevas

**Plantas originales excluidas por grupo (cuaderno 11).** Entrenando sin el grupo espacial de la planta evaluada, en la ventana de prueba K1 obtiene 0.0396 frente a 0.0423 de Random Forest y 0.0434 de Ridge, con mejora en 14 de 17 plantas y 6 de 7 grupos ($p=0.004$ por planta y $0.031$ por grupo sin ajustar; $0.031$ y $0.109$ con Holm). El error de K1 apenas cambia respecto de la prueba con plantas conocidas (0.0392), mientras Random Forest pasa de 0.0406 a 0.0423: K1 depende poco de haber visto la planta porque su nivel sale de la serie propia.

::::{table} Plantas originales con su grupo excluido del entrenamiento, ventana de prueba. Fuente: `resultados/pronostico_d1_plantas_no_vistas/resumen_macro.csv`.
:label: tbl-no-vistas
```{code-cell} python
nv = apoyo.leer("resultados/pronostico_d1_plantas_no_vistas/resumen_macro.csv")
apoyo.tabla(nv[["modelo", "conjunto", "MAE_macro", "RMSE_macro", "MBE_macro", "MAE_desv_entre_grupos"]], decimales=4, indice="modelo")
```
::::

**Plantas nuevas (cuaderno 14).** Las 15 plantas de la ampliación no intervinieron en ninguna decisión. Entrenados solo con las 17 originales hasta julio, K1 obtiene en ellas 0.0356 y Random Forest 0.0468, una reducción del 24 % con mejora en 13 de 15 plantas. Esa cifra no es representativa: cuatro plantas de FC atípico, probablemente autogeneradores que inyectan excedentes, explican el 92 % de la reducción porque Random Forest no se adapta a su nivel. Sin ellas, la reducción es del 3.4 % ([Figura %s](#fig-nuevas)). Cinco plantas nuevas comparten coordenadas o celda meteorológica con plantas originales, de modo que las 15 plantas equivalen a menos unidades independientes.

:::{figure} figuras/res_plantas_nuevas.png
:label: fig-nuevas
:width: 100%
K1 frente a Random Forest en las 15 plantas nuevas, entrenados solo con las 17 originales: (a) MAE por planta (* = FC atípico); (b) MAE macro con y sin las cuatro plantas atípicas y, como referencia, en las 17 originales.
:::

**Validación conjunta (cuaderno 16, exploratoria).** Dejando fuera un grupo a la vez con las 32 plantas y 9 grupos, K1 reduce el MAE de Random Forest en un 16.8 % en fechas nuevas (0.0378 frente a 0.0455) y mejora en 28 de 32 plantas y 7 de 9 grupos. Esta cifra también incluye a las plantas atípicas; sin los autogeneradores la reducción es del 5.8 %.

## Conjuntos de datos adicionales

La guía pide evaluar el modelo propio en al menos tres conjuntos adicionales de regresión, de naturaleza distinta y con el mismo protocolo. Ninguno de los diez conjuntos del catálogo de contingencia (cuaderno 15) sirve para K1 y W1, porque son de corte transversal y no tienen una serie por unidad. El cuaderno 23 eligió tres series de UCI con criterios fijados y huella SHA-256 antes de descargar ningún dato: PM2.5 diario en 12 estaciones de Pekín con meteorología (501), CO diario de una estación italiana con sensores y meteorología (*Air Quality*, 360) y tráfico de la autopista I-94 agregado por día con clima (492). El conjunto previsto de cinco ciudades chinas (394) se sustituyó por el suplente de Pekín porque su archivo comprimido no se pudo abrir; el motivo se registró antes de ver desempeño. Las covariables son las observadas el día objetivo, no pronósticos: la tarea es más fácil que la principal y favorece a los modelos que las usan directamente. Todos los modelos recibieron la misma búsqueda, las mismas particiones de origen móvil y diez semillas en los modelos con árboles.

::::{table} MAE macro por unidad en los tres conjuntos adicionales (media de los bloques de evaluación, en sus unidades) y en el problema principal (prueba). Fuente: `resultados/datasets_adicionales/mae_por_conjunto.csv`.
:label: tbl-datasets
```{code-cell} python
mc = apoyo.leer("resultados/datasets_adicionales/mae_por_conjunto.csv")
apoyo.tabla(mc.sort_values("FV Colombia (principal)"), decimales={"Pekín multisitio (501)": 2, "Air Quality Italia (360)": 3,
            "Tráfico I-94 (492)": 1, "FV Colombia (principal)": 4}, indice="modelo")
```
::::

El resultado es desfavorable para los modelos propuestos. W1 es el primero en el problema principal y queda sexto, sexto y octavo en los conjuntos nuevos. Frente al mejor modelo de librería de cada conjunto, su MAE es un 27 % mayor en Pekín (DM-HLN con Holm $p<10^{-4}$; peor en las 12 estaciones), un 13 % mayor en *Air Quality* ($p=0.71$, con solo 94 días de prueba) y un 36 % mayor en el tráfico ($p<10^{-4}$). K1 queda por detrás de W1 en los tres. La prueba de Friedman con la corrección de Iman–Davenport {cite:p}`friedman1937ranks,iman1980friedman` sobre los cuatro conjuntos rechaza la igualdad de rangos ($F=2.82$; gl 9 y 27; $p=0.018$), pero ninguna comparación por pares de W1 es significativa con Holm {cite:p}`demsar2006comparisons`. El criterio de consistencia de la guía (mejor en al menos la mitad de los conjuntos y no significativamente peor en ninguno) **no se cumple** para K1 ni para W1.

```{code-cell} python
# Criterio de consistencia de la guía y prueba de Friedman (cuaderno 23).
display(apoyo.tabla(apoyo.leer("resultados/datasets_adicionales/criterio4_veredicto.csv"), indice="modelo"))
display(apoyo.tabla(apoyo.leer("resultados/datasets_adicionales/friedman.csv"), decimales=4, indice="universo"))
```

:::{figure} figuras/dat_mae_relativo.png
:label: fig-dat-relativo
:width: 90%
MAE de cada modelo relativo al mejor de cada conjunto adicional (cuaderno 23).
:::

Una lectura exploratoria, añadida después de ver los resultados, sugiere la causa. En los tres conjuntos la desviación del día objetivo respecto de la media de 30 días está correlacionada con la del día anterior (0.54 en Pekín, 0.44 en *Air Quality* y 0.38 en el tráfico, frente a 0.16 en las plantas FV). K1 y W1 parten de la media de 30 días y no usan el valor del día anterior, que los modelos de librería sí reciben. El tráfico tiene además un ciclo semanal que ningún modelo recibió como variable. Ambos mecanismos son hipótesis que el diseño no pone a prueba.

## Sensibilidad a hiperparámetros

Se evaluaron 114 configuraciones de W1 ($h\in\{0.25,0.5,1,2,4,\infty\}$ por 19 pares $(\rho_g,\rho_o)$) y 17 de K1 (cuaderno 22). Las 57 configuraciones de W1 con $h\ge2$ mejoran a K1 y a Random Forest, con MAE entre 0.0339 y 0.0346; con $h\le1$ ninguna mejora a K1, y con $h\le0.5$ el error sube a 0.0377–0.0416, peor que Random Forest ([Figura %s](#fig-sensibilidad)). Con $h=\infty$ y $\rho_g=\rho_o=1$ el error coincide con el de K1, como predice la [Proposición %s](#prop-w1k1). La región favorable de W1 es la de ventanas anchas: la ponderación por identidad aporta más que la localidad en el espacio de anomalías. El estudio de optimizadores llegó a lo mismo por otro camino: con $h\in[0.25,8]$, W1 eligió $h$ de 7 a 8 en tres de cuatro bloques ([capítulo 3](03_benchmarks.md)). K1 es insensible a $\alpha\le100$ y solo pierde frente a Random Forest con $\alpha\ge10^4$.

:::{figure} figuras/res_sensibilidad.png
:label: fig-sensibilidad
:width: 100%
Sensibilidad en desarrollo. (a) Diferencia de MAE de W1 respecto de K1 ($\times10^{-3}$ de FC) según $h$ y $(\rho_g,\rho_o)$; × marca las configuraciones que no mejoran a Random Forest. (b) MAE de K1 y Ridge al variar $\alpha$.
:::

```{code-cell} python
sens = apoyo.leer("resultados/robustez/sensibilidad_resumen.csv")
apoyo.tabla(sens, decimales=4, indice="modelo")
```

## Semillas, ruido y atípicos

**Semillas.** W1, K1, Ridge y SVR son deterministas. Con diez semillas (2026 a 2035), Random Forest da $0.03629\pm0.00002$ y XGBoost $0.03687\pm0.00007$ en desarrollo, y en ninguna de las 60 combinaciones de semilla y bloque mejoran a K1 o a W1. La variación entre bloques (de 0.0297 a 0.0379 en W1) es dos órdenes de magnitud mayor que la variación por semilla. El cuaderno 21 repitió el estudio para las mejores combinaciones de su búsqueda, con la misma conclusión.

```{code-cell} python
sem = apoyo.leer("resultados/robustez/estabilidad_semillas_resumen.csv")
apoyo.tabla(sem[sem.bloque == "desarrollo"], decimales=5, indice="modelo")
```

**Ruido y atípicos.** Con ruido gaussiano del 20 % de la desviación estándar en las variables pronosticadas, el MAE de desarrollo aumenta entre 0.9 y 1.6 % en todos los modelos y el orden no cambia. Con un 10 % de atípicos de $\pm5\sigma$ en la respuesta de entrenamiento, K1 aumenta un 0.3 %, W1 y Random Forest un 2.5 % y XGBoost un 4.4 %; en ese nivel W1 pierde su ventaja sobre K1, que conserva con un 5 %. W1 y K1 quedan por debajo de todas las referencias en todos los niveles evaluados.

:::{figure} figuras/rob_ruido_atipicos.png
:label: fig-ruido
:width: 100%
Aumento relativo del MAE de desarrollo con ruido en el pronóstico (en entrenamiento y evaluación, o solo en evaluación) y con atípicos en la respuesta de entrenamiento (cuaderno 22).
:::

## Tamaño de muestra y costo

K1 es casi insensible al tamaño del historial: con el 10 % de las filas (unas 900 a 1,250) su MAE es 0.0348, frente a 0.0347 con todas. W1 necesita más datos: con el 10 % da 0.0379, como Ridge, y supera a K1 a partir del 50 %. Con $n=8{,}000$ y un hilo, K1 se ajusta en unos 4 ms y W1 predice 1,000 filas en 0.24 s. En el barrido complementario, con $n$ hasta 128,000, las pendientes empíricas en escala log-log (0.98 para el ajuste de K1 y 1.09 para la predicción de W1) coinciden con el orden lineal en $n$ del [capítulo 5](05_modelo_original.md); en el barrido principal, con $n$ menor, los costos fijos dominan y las pendientes son más bajas; en $k$ la pendiente medida queda por debajo de la cuadrática asintótica porque con $k\le81$ dominan los términos $O(nk)$.

:::{figure} figuras/rob_curva_aprendizaje.png
:label: fig-aprendizaje
:width: 90%
MAE de desarrollo según la fracción del entrenamiento usada (cuaderno 22).
:::

```{code-cell} python
# Pendientes log-log del tiempo frente a n y k: barrido complementario (n grande) y barrido principal.
display(apoyo.tabla(apoyo.leer("resultados/robustez/tiempos_complemento_pendientes.csv"), decimales=3, indice="medida"))
tp = apoyo.leer("resultados/robustez/tiempos_pendientes.csv")
apoyo.tabla(tp, decimales=3, indice="modelo")
```

## Intervalos de predicción

Con intervalos conformes {cite:p}`vovk2005algorithmic` calibrados con los residuos absolutos de desarrollo, la cobertura en la prueba queda por debajo de la nominal en todos los modelos: entre 74.1 y 76.0 % para el 80 % y entre 85.5 y 87.1 % para el 90 % ([Figura %s](#fig-conformal)). La causa probable es que el error de la prueba es mayor que el de desarrollo (W1 pasa de 0.0341 a 0.0388): la intercambiabilidad entre calibración y prueba no se cumple bajo este cambio de periodo. Recalibrar con una ventana móvil de los 90 días anteriores a cada origen restablece la cobertura (W1: 80.5 y 90.3 %; K1: 80.2 y 89.9 %) con intervalos más anchos. La cobertura por planta es desigual: en una planta baja al 58.7 % con el método global. El CRPS {cite:p}`gneiting2007scoring` ordena a los modelos como el MAE (W1 0.0283, K1 0.0285, Random Forest 0.0294).

:::{figure} figuras/res_conformal.png
:label: fig-conformal
:width: 100%
Intervalos conformes en la prueba (1,561 filas): cobertura al 80 % (a) y al 90 % (b), con intervalo de Wilson al 95 %, y ancho medio al 90 % (c), por modelo y método de calibración. Las líneas discontinuas marcan el nivel nominal.
:::

::::{table} Cobertura de los intervalos conformes en la prueba por modelo, método y nivel. Fuente: `resultados/robustez/conformal_cobertura.csv`.
:label: tbl-conformal
```{code-cell} python
cc = apoyo.leer("resultados/robustez/conformal_cobertura.csv")
cols = ["modelo", "metodo", "nivel_nominal", "cobertura", "wilson95_inf", "wilson95_sup", "ancho_medio", "cobertura_planta_min", "CRPS"]
apoyo.tabla(cc[cols], decimales=3, indice="modelo")
```
::::

## Impacto en energía

El FC es energía dividida por capacidad neta AC por 24 h, de modo que el error de una predicción en MWh es $(\hat y-y)\,C_{p,t}\,24$, con la capacidad de cada planta y día (la identidad se verificó en toda la base con error máximo $10^{-16}$). La conversión se hace fila a fila y se agrega después; no se multiplica el MAE macro por una capacidad media. Es un análisis derivado y posterior a los resultados (`scripts/impacto_energetico.py`, salidas en `resultados/impacto_energetico/`). **Una reducción del error no es generación adicional**, no se anualiza (hay 92 días) y no se estima valor económico.

::::{table} Prueba (17 plantas, 92 días): distintos estimandos del mismo error. Fuente: `resultados/impacto_energetico/prueba_agregados.csv`.
:label: tbl-energia
```{code-cell} python
ag = apoyo.leer("resultados/impacto_energetico/prueba_agregados.csv").set_index("modelo")
ag = ag.loc[["K1 Modelo de anomalías", "W1 Regresión local", "Random Forest", "SVR", "XGBoost", "Media móvil 30 días"]]
cols = {"MAE_fc_macro": "MAE macro (FC)", "MAE_fc_ponderado_capacidad": "MAE pond. por capacidad (FC)",
        "suma_errores_abs_flota_mwh_dia": "suma de |error| (MWh/día)", "error_neto_abs_flota_mwh_dia": "error neto cohorte (MWh/día)",
        "MBE_flota_mwh_dia": "sesgo cohorte (MWh/día)", "WAPE_flota_pct": "error total / energía (%)"}
apoyo.tabla(ag[list(cols)].rename(columns=cols), decimales={"MAE macro (FC)": 4, "MAE pond. por capacidad (FC)": 4, "suma de |error| (MWh/día)": 0,
            "error neto cohorte (MWh/día)": 0, "sesgo cohorte (MWh/día)": 0, "error total / energía (%)": 1})
```
::::

La energía media producida por las 17 plantas es de 7,296 MWh/día. K1 reduce la suma diaria de errores absolutos de 1,151 a 1,085 MWh/día frente a Random Forest (−5.7 %); la reducción relativa depende del estimando (3.3 % en el MAE macro, 5.9 % ponderando por capacidad). El efecto más visible es el sesgo de la cohorte: Random Forest sobreestima 173 MWh/día y K1 subestima 40.

::::{table} K1 frente a Random Forest por planta en la prueba (negativo favorece a K1). Fuente: `resultados/impacto_energetico/comparaciones_por_planta.csv`.
:label: tbl-energia-planta
```{code-cell} python
cp = apoyo.leer("resultados/impacto_energetico/comparaciones_por_planta.csv")
k = cp[cp.comparacion.str.startswith("K1 − Random Forest (pre")]
t = k[["planta", "grupo", "cap_MW", "energia_media_mwh", "MAE_fc_a", "MAE_fc_b", "dif_MAE_fc", "dif_relativa_pct", "MAE_mwh_a", "MAE_mwh_b", "dif_MAE_mwh_dia"]]
t = t.rename(columns={"cap_MW": "cap. (MW)", "energia_media_mwh": "energía (MWh/día)", "MAE_fc_a": "MAE K1", "MAE_fc_b": "MAE RF", "dif_MAE_fc": "Δ MAE",
                      "dif_relativa_pct": "Δ (%)", "MAE_mwh_a": "K1 (MWh/día)", "MAE_mwh_b": "RF (MWh/día)", "dif_MAE_mwh_dia": "Δ (MWh/día)"})
apoyo.tabla(t, decimales={"cap. (MW)": 1, "energía (MWh/día)": 0, "MAE K1": 4, "MAE RF": 4, "Δ MAE": 4, "Δ (%)": 1, "K1 (MWh/día)": 1, "RF (MWh/día)": 1, "Δ (MWh/día)": 2},
            indice="planta")
```
::::

K1 mejora en 14 de 17 plantas con mediana de −0.95 MWh/día; Guayepo (380 MW) y Portón del Sol (102 MW) aportan el 67 % de la reducción; La Loma (150 MW) y otras dos plantas empeoran.

::::{table} Comparaciones en energía (suma diaria sobre las plantas, MWh/día; negativo favorece al primer modelo) con IC 95 % por bootstrap de bloques de 7 días. El intervalo es descriptivo: el estimando se eligió después de ver los resultados. Fuente: `resultados/impacto_energetico/comparaciones_resumen.csv`.
:label: tbl-energia-comp
```{code-cell} python
cr = apoyo.leer("resultados/impacto_energetico/comparaciones_resumen.csv")
cr["Δ suma (MWh/día) [IC 95 %]"] = [f"{a:.0f} [{b:.0f}, {c:.0f}]" for a, b, c in zip(cr.dif_suma_flota_mwh_dia, cr.ic95_suma_flota_inf, cr.ic95_suma_flota_sup)]
cr["plantas"] = cr.plantas_que_mejoran.astype(str) + "/" + cr.plantas.astype(str)
t = cr[["comparacion", "dif_MAE_fc_macro", "dif_relativa_macro_pct", "Δ suma (MWh/día) [IC 95 %]", "dif_suma_flota_relativa_pct", "plantas", "planta_mayor_aporte"]]
t = t.rename(columns={"dif_MAE_fc_macro": "Δ MAE macro", "dif_relativa_macro_pct": "Δ macro (%)", "dif_suma_flota_relativa_pct": "Δ energía (%)",
                      "plantas": "plantas que mejoran", "planta_mayor_aporte": "mayor aporte"})
apoyo.tabla(t, decimales={"Δ MAE macro": 4, "Δ macro (%)": 1, "Δ energía (%)": 1}, indice="comparacion")
```
::::

**Plantas nuevas.** La reducción del 24 % del MAE macro viene de cuatro autogeneradores de 3.3 a 5 MW, el 2.9 % de la capacidad, que pesan lo mismo que el resto en el promedio macro (explican el 92 % de la mejora). En energía la reducción es del 8 % y, sin esas plantas, del 6 % con un intervalo que incluye el cero. En la validación conjunta, el 16.8 % baja a 5.8 % sin los autogeneradores.

**Alcance.** El análisis cubre energía diaria. El despacho y la liquidación de desvíos son horarios: un error diario pequeño puede ocultar errores horarios que se compensan, y esa evaluación no se hizo.

## Horizonte D+1 y disponibilidad de los datos

El FC de $D$ y el pronóstico para $T$ (corridas `previous_day1` de ECMWF, GFS e ICON) se suponen disponibles al cierre de $D$. No se verificó cuándo publica XM la generación del día ni la hora de cada corrida, y las decisiones del mercado para el día siguiente se toman durante $D$. El resultado es un pronóstico $D{+}1$ **retrospectivo** con pronósticos archivados, no un pronosticador operacional verificado. La Tarea 1, con clima observado del mismo día, es una estimación retrospectiva y no se compara con el pronóstico. Con el historial retrasado un día (sensibilidad $B_{1r}$ frente a $B_1$), el MAE de K1 pasa de 0.03961 a 0.03966 y el de Random Forest de 0.04213 a 0.04295.

## Análisis de errores

Los errores grandes se concentran en los días de FC bajo, que incluyen fallas, restricciones y registros incompletos que ningún modelo meteorológico puede pronosticar. Los tres modelos sobreestiman esos días y subestiman los de FC alto, como corresponde a predictores de la media condicional con información incompleta ([Figura %s](#fig-segmentos)). La fila de mayor error de la muestra explicada con SHAP es un día con FC observado de 0.0003, que los tres modelos pronostican entre 0.19 y 0.23 ([sección %s](#sec-interpretabilidad)). Separar esos días con información de disponibilidad de XM mejoraría la evaluación, pero no se hizo.

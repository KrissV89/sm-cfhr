---
title: Protocolo y validación estadística
short_title: 6. Validación
kernelspec:
  name: python3
  display_name: Python 3
---

# Protocolo y validación estadística

Este capítulo describe cómo se separaron desarrollo y prueba, qué tan equitativa es la comparación, qué pruebas de hipótesis se calcularon y qué concluyen. Todos los errores son MAE macro por planta en unidades de FC: se promedia el error dentro de cada planta y luego entre plantas ([ecuación %s](#eq-maemacro)). Un MAE de 0.01 equivale a 0.24 horas de capacidad plena por día; en una planta de 100 MW, a 24 MWh diarios.

```{code-cell} python
import apoyo
import pandas as pd
```

## Particiones y selección

La partición respeta el orden temporal ([Figura %s](#fig-protocolo)). El desarrollo usa tres bloques consecutivos de origen móvil: D1 (noviembre de 2025 a enero de 2026), D2 (febrero a abril de 2026) y D3 (mayo a julio de 2026); cada entrenamiento contiene solo fechas objetivo anteriores al inicio del bloque. La prueba es agosto a octubre de 2026 (92 días, 1,561 filas, 17 plantas), con entrenamiento hasta el 31 de julio. Dentro de cada entrenamiento, los hiperparámetros se eligen con tres particiones temporales de ventana creciente y un hueco de 30 días entre entrenamiento y validación, para que las medias móviles de la validación no se solapen con el entrenamiento. Durante la prueba el historial se actualiza con observaciones anteriores a cada origen; los coeficientes no se reajustan.

:::{figure} figuras/met_protocolo_temporal.png
:label: fig-protocolo
:width: 100%
Particiones temporales de la tarea $D{+}1$. Cada bloque de desarrollo se evalúa con un modelo entrenado solo con fechas anteriores a su inicio; la prueba se usó una vez, con entrenamiento hasta el 31 de julio de 2026.
:::

La selección se hizo solo con el desarrollo. El mejor candidato propio y la mejor referencia se escribieron en `resultados/pronostico_d1/seleccion_desarrollo.json` antes de ejecutar la prueba, y el cuaderno 07 impide repetirla. La comparación decisiva quedó fijada como **K1 frente a Random Forest**, ambos con el conjunto $B_2$. W1 se seleccionó después en una ronda de candidatos evaluada solo en desarrollo (cuaderno 17) y se evaluó una vez en la misma ventana (cuaderno 19, con las guardas `inicio.json` y `predicciones.pkl`). Como la ventana ya se había leído en 07, la evaluación de W1 es **exploratoria**.

**Criterio de mejora fijado antes de la prueba.** Un candidato mejora a la referencia si tiene menor MAE macro, no empeora el RMSE ni el sesgo absoluto, mejora en la mayoría de plantas y de grupos, y la diferencia resiste la corrección por comparaciones múltiples.

**Resultado de la regla.** La comparación decisiva cumplió las cinco primeras condiciones y **no la última**: los valores $p$ de Wilcoxon de K1 frente a Random Forest son 0.017 por planta y 0.016 por grupo, y con la corrección de Holm sobre las 48 pruebas del cuaderno 07 pasan a 0.70 y 0.69 (`resultados/pronostico_d1/criterio_mejora.json`). Por tanto, la regla de mejora prefijada no se cumplió completa.

## Equidad de la comparación

| Aspecto | Situación | Consecuencia |
|---|---|---|
| Filas y particiones | Todos los modelos usan las mismas filas, bloques y particiones internas | Comparación pareada válida |
| Información | Las referencias reciben $B_2$ (historial y pronóstico en bruto); K1 recibe la media de 30 días y las anomalías, que se derivan de la misma información; W1 recibe además la identidad de planta y de grupo | K1 frente a referencias compara formas de usar la misma información; W1 frente a referencias compara sistemas con información distinta |
| Presupuesto de búsqueda | En la selección original, de 4 a 12 combinaciones según el modelo ([Tabla %s](#tbl-rejillas)) | Desigual; el cuaderno 21 lo iguala en 30 evaluaciones y el orden no cambia ([sección %s](#sec-optimizadores)) |
| Semillas | Una semilla (2026) en la selección; diez en los estudios de robustez (22) y de optimizadores (21) | La variación por semilla es al menos un orden de magnitud menor que la diferencia entre K1 y Random Forest |
| Uso de la prueba | K1 frente a Random Forest fijada antes; W1 y las pruebas de la guía, después | Solo la comparación de K1 con Random Forest es confirmatoria en sentido estricto |

## Resultados de la prueba

La [Tabla %s](#tbl-principal) del [capítulo 3](03_benchmarks.md) resume el desempeño. En la prueba, K1 obtiene 0.0392 frente a 0.0406 de Random Forest (3.3 % menos), con mejor RMSE y menor sesgo absoluto, y mejora en 14 de 17 plantas y en los 7 grupos. W1 obtiene 0.0388. La ventaja no es uniforme ([Figura %s](#fig-porplanta), [Figura %s](#fig-segmentos)): K1 empeora frente a Random Forest en septiembre (0.0405 frente a 0.0395) y lo mejora en agosto y octubre; por terciles del FC observado, la mejora se concentra en los días de FC bajo, y en los de FC alto Random Forest tiene menor error.

:::{figure} figuras/res_mejora_por_planta.png
:label: fig-porplanta
:width: 100%
Diferencia de MAE en la prueba de K1 y W1 frente a Random Forest (a) por planta y (b) por grupo espacial. Los valores negativos favorecen al modelo propuesto.
:::

:::{figure} figuras/res_errores_segmento.png
:label: fig-segmentos
:width: 100%
MAE macro (a, b) y sesgo MBE (c, d) en la prueba por mes y por tercil del FC observado dentro de cada planta, para K1, W1 y Random Forest.
:::

:::{figure} figuras/res_serie_ejemplo.png
:label: fig-serie
:width: 100%
FC diario observado y pronosticado por K1, W1 y Random Forest durante la prueba en tres plantas elegidas por regla: (a) la de mayor mejora de K1 frente a Random Forest, (b) la mediana y (c) la de mayor deterioro.
:::

## Pruebas de hipótesis

Las pruebas que prescribe la guía para pronóstico se calcularon el 16 de noviembre de 2026 sobre las predicciones ya guardadas (`scripts/pruebas_guia.py`, diseño en `resultados/pruebas_guia/PROTOCOLO.md`). Su diseño se escribió antes de calcularlas y se reportan todas, pero son un análisis *post hoc*. Todas usan la serie diaria del diferencial de pérdida entre un modelo propuesto $A$ y una referencia $B$,

$$
d_t=\frac{1}{|\mathcal{P}_t|}\sum_{p\in\mathcal{P}_t}\Bigl(\bigl|y_{p,t}-\hat y^{A}_{p,t}\bigr|-\bigl|y_{p,t}-\hat y^{B}_{p,t}\bigr|\Bigr),\qquad t=1,\dots,92,
$$ (eq-diferencial)

donde $\mathcal{P}_t$ son las plantas con dato el día $t$; un valor negativo favorece a $A$.

- **Diebold–Mariano con corrección HLN** {cite:p}`diebold1995comparing,harvey1997testing`: $H_0:\mathbb{E}[d_t]=0$, con $\mathrm{DM}=\bar d/\sqrt{\hat\gamma_0/T}$ corregido por $\sqrt{(T+1-2h+h(h-1)/T)/T}$ con $h=1$ y referencia $t_{T-1}$. Sensibilidad con la varianza de largo plazo de Newey–West con 4 rezagos (DM-HAC) {cite:p}`newey1987simple`.
- **Giacomini–White** {cite:p}`giacomini2006tests`: capacidad predictiva condicional, $H_0:\mathbb{E}[d_t\mid\mathcal{F}_{t-1}]=0$, con instrumentos $(1,d_{t-1})$ y estadístico de Wald con referencia $\chi^2_2$.
- **Prueba bayesiana de rangos con signo** {cite:p}`benavoli2014bayesian,benavoli2017time`: sobre las diferencias de MAE por planta (17) y por grupo (7), probabilidades posteriores de que $A$ sea mejor, de equivalencia práctica (ROPE = 1 % del MAE de la referencia) y de que $B$ sea mejor.
- **Conjunto de confianza de modelos (MCS)** {cite:p}`hansen2011mcs`: eliminación secuencial con el estadístico $T_{\max}$, bootstrap por bloques circulares de 7 días y 10,000 réplicas, al 90 %. Universo: los 16 modelos con aprendizaje y la media móvil de 30 días.

**Multiplicidad y efecto.** Holm {cite:p}`holm1979simple` se aplica por prueba dentro de cada familia: K1 frente a sus diez referencias ($m=10$) y W1 frente a las mismas y a K1 ($m=11$). El tamaño del efecto es la diferencia de MAE con IC 95 % por bootstrap de bloques de 7 días y la $d$ de Cohen de $d_t$. Para cada $d_t$ se reportan la autocorrelación, Ljung–Box a 7 rezagos {cite:p}`ljung1978measure` y Shapiro–Wilk {cite:p}`shapiro1965analysis`. La prueba de Wilcoxon sobre $d_t$ comparte la dependencia temporal y no cuenta como prueba independiente. Las pruebas se implementaron con NumPy y SciPy y se validaron con datos sintéticos (tamaño bajo $H_0$, potencia y equivalencia de DM-HLN con la prueba $t$ cuando $h=1$; `resultados/pruebas_guia/validacion_sintetica.csv`).

::::{table} Pruebas en la ventana de prueba ($T=92$ días, pérdida absoluta) para cada comparación de K1 y W1. Mejora relativa del MAE, IC 95 % de la diferencia, $d$ de Cohen, plantas que mejoran, valores $p$ con Holm de DM-HLN, Giacomini–White y Wilcoxon, y probabilidad bayesiana por planta de que el modelo propuesto sea mejor. Fuente: `resultados/pruebas_guia/pruebas_por_comparacion.csv`.
:label: tbl-pruebas
```{code-cell} python
p = apoyo.leer("resultados/pruebas_guia/pruebas_por_comparacion.csv")
p["IC 95 % de ΔMAE"] = [f"[{a:.4f}, {b:.4f}]" for a, b in zip(p.ic95_inf, p.ic95_sup)]
p["plantas"] = p.plantas_mejora.astype(str) + "/" + p.plantas.astype(str)
p["comparación"] = p.familia + " frente a " + p.referencia + p.decisiva_prefijada.map({True: " (decisiva)", False: ""})
cols = {"comparación": "comparación", "mejora_pct": "mejora (%)", "IC 95 % de ΔMAE": "IC 95 % de ΔMAE", "cohen_d": "d",
        "plantas": "plantas que mejoran", "dm_hln_p_holm": "p DM-HLN", "gw_p_holm": "p GW", "wilcoxon_p_holm": "p Wilcoxon",
        "bayes_planta_p_propuesto": "P(mejor) por planta"}
apoyo.tabla(p[list(cols)].rename(columns=cols), decimales={"mejora (%)": 1, "d": 2, "p DM-HLN": 4, "p GW": 4, "p Wilcoxon": 4,
            "P(mejor) por planta": 3}, indice="comparación")
```
::::

**K1 frente a Random Forest (decisiva).** La serie $d_t$ no muestra autocorrelación significativa (Ljung–Box $p=0.19$) ni desviación de la normalidad (Shapiro–Wilk $p=0.19$). DM-HLN da $p=0.034$ sin ajustar y $0.076$ con Holm; Giacomini–White, $0.116$ y $0.233$; y el IC 95 % de la diferencia de MAE, $[-0.0028, +0.0002]$, incluye el cero. La prueba bayesiana asigna probabilidad 0.994 a que K1 sea mejor por planta, y la $d$ de Cohen es $-0.22$. Con la regla de lectura fijada, la evidencia es mixta: la dirección es consistente entre plantas y grupos, pero las pruebas sobre la serie temporal no rechazan la igualdad tras la corrección.

**K1 frente a las demás referencias.** Frente a Ridge, Lasso y la Ridge bayesiana, DM-HLN, Giacomini–White y Wilcoxon rechazan con Holm (reducción de 5.7 a 6.1 %, $|d|\ge0.41$); frente al árbol de decisión también. Frente a SVR, HistGradientBoosting y XGBoost ninguna prueba temporal rechaza con Holm.

**W1.** Frente a Random Forest, XGBoost, HistGradientBoosting y $k$ vecinos, W1 rechaza con Holm en DM-HLN, Giacomini–White y Wilcoxon ($p_{\mathrm{Holm}}\le0.046$), con $|d|\ge0.3$. Frente a SVR, Giacomini–White no rechaza ($p_{\mathrm{Holm}}=0.125$). Frente a K1 la diferencia es del 1.0 % y las pruebas temporales rechazan, pero la prueba bayesiana por planta sitúa la diferencia en la ROPE con probabilidad 0.56: es detectable y pequeña. Tres razones impiden presentar estos rechazos como confirmación: W1 se seleccionó después de conocer los resultados de K1 en esa ventana (aunque solo con datos de desarrollo), recibe la identidad de planta y de grupo que las referencias no tienen, y las pruebas son posteriores a los resultados.

:::{figure} figuras/res_diferencial_diario.png
:label: fig-diferencial
:width: 100%
Diferencial diario de pérdida absoluta $d_t$ ([ecuación %s](#eq-diferencial)) frente a Random Forest de (a) K1 y (b) W1 durante la prueba, con su media acumulada.
:::

::::{table} Supuestos de las series $d_t$: autocorrelación de orden 1, Ljung–Box a 7 rezagos, Shapiro–Wilk y días favorables al modelo propuesto. Fuente: `resultados/pruebas_guia/supuestos.csv`.
:label: tbl-supuestos
```{code-cell} python
s = apoyo.leer("resultados/pruebas_guia/supuestos.csv")
s["comparación"] = s.familia + " frente a " + s.referencia
cols = ["comparación", "acf_1", "banda_acf_95", "ljung_box_p", "shapiro_p", "asimetria", "dias_favorables_propuesto"]
apoyo.tabla(s[cols], decimales=3, indice="comparación")
```
::::

## Conjunto de confianza de modelos

El MCS al 90 % no separa los modelos propuestos de las mejores referencias ([Figura %s](#fig-mcs), [Tabla %s](#tbl-mcs)). Con W1 en el universo conserva a W1, K1, SVR, Random Forest, XGBoost, HistGradientBoosting, $k$ vecinos, el árbol y los candidatos C2 y C3, y elimina a Ridge, Lasso, la Ridge bayesiana, C1, C4 y la media móvil. Sin W1 el conjunto es análogo con K1 como modelo de menor pérdida, y con pérdida cuadrática los resultados son similares. Con 92 días, el MCS distingue los modelos lineales de los no lineales, pero no los no lineales entre sí ni de K1 y W1. La mejor combinación de librería según el desarrollo, Random Forest, permanece en el conjunto ($p_{\mathrm{MCS}}=0.152$).

:::{figure} figuras/res_mcs.png
:label: fig-mcs
:width: 100%
Valor $p$ del MCS por modelo en la prueba, con W1 en el universo, para la pérdida absoluta (principal) y la cuadrática. La línea marca $\alpha=0.10$; los modelos por encima permanecen en el conjunto.
:::

::::{table} MCS al 90 %, pérdida absoluta, con W1 en el universo: orden de eliminación, valor $p$ del MCS y pérdida media. Fuente: `resultados/pruebas_guia/mcs.csv`.
:label: tbl-mcs
```{code-cell} python
m = apoyo.leer("resultados/pruebas_guia/mcs.csv")
m = m[(m.universo == "con W1") & (m.perdida == "absoluta")]
apoyo.tabla(m[["modelo", "orden_eliminacion", "p_mcs", "perdida_media", "en_mcs_90"]], decimales=4, indice="modelo")
```
::::

## Lectura

Ninguna prueba sobre una sola ventana de tres meses estima la variación entre periodos, y septiembre ya invierte el signo de K1 frente a Random Forest. Con esa limitación:

- La regla de mejora prefijada de 07 no se cumple completa (no resiste Holm). La comparación prefijada, K1 frente a Random Forest, favorece a K1 en dirección (14 de 17 plantas, 7 de 7 grupos, probabilidad bayesiana 0.994) pero no es significativa tras la corrección en las pruebas sobre la serie temporal.
- K1 es significativamente mejor que las referencias lineales y el árbol, no que los ensambles de árboles ni SVR.
- W1 reúne tres pruebas con $p_{\mathrm{Holm}}<0.05$ frente a varios ensambles, pero de forma *post hoc* y con información adicional; frente a K1 la diferencia es pequeña.
- El MCS conserva a Random Forest, SVR y XGBoost junto a K1 y W1.

La condición R3 de la guía (superioridad significativa con tres pruebas, una global y efecto relevante) **no se cumple** ([capítulo 9](09_conclusiones.md)).

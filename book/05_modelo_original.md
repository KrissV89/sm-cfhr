---
title: Modelo original
short_title: 5. Modelo original
kernelspec:
  name: python3
  display_name: Python 3
---

# Modelo original: K1 y W1

Este capítulo formula los dos modelos propuestos para el pronóstico $D{+}1$, demuestra sus propiedades básicas, muestra el código y las pruebas, y resume la ablación que localiza de dónde viene la mejora. La formulación coincide con la del manuscrito (`paper/manuscrito.tex`, secciones K1 y W1) y con los docstrings de `src/original_model.py`.

## Motivación

El análisis exploratorio ([capítulo 2](02_datos.md)) mostró dos hechos. Entre plantas, el nivel del FC no sigue a la irradiación modelada: depende de características que la base no observa, como la relación entre potencia DC y capacidad AC, el montaje y el registro de capacidad. Dentro de cada planta, el FC diario sí sigue a la irradiación. La Tarea 1 lo confirmó: sin historial de la planta ningún modelo baja de un MAE macro de 0.043 ([capítulo 3](03_benchmarks.md)). Un modelo común a todas las plantas que reciba el pronóstico en bruto tiene que reconstruir el nivel de cada planta. K1 separa los dos problemas: toma el nivel de la propia serie y aprende solo la desviación diaria respecto de ese nivel. W1 relaja dos supuestos de K1 (pendiente común y respuesta lineal global) con una regresión local.

## K1: anomalías sobre el nivel reciente

**Nivel.** $L_{p,D}=\bar y^{(30)}_{p,D}$ es la media del FC de la planta $p$ en los 30 días que terminan en $D$, con al menos 15 observaciones.

**Anomalías.** Para cada variable pronosticada $j=1,\dots,k$ ($k=5$: irradiación, temperatura, nubosidad, humedad y viento),

$$
a_{j,p,T}=x^{\mathrm{pr}}_{j,p,T}-\frac{1}{|\mathcal{O}'_{30}(T)|}\sum_{s\in\mathcal{O}'_{30}(T)}x^{\mathrm{pr}}_{j,p,s},
$$ (eq-anomalia)

donde $\mathcal{O}'_{30}(T)$ son los días con pronóstico entre los 30 que terminan en $T$ (al menos 15). La ventana incluye el pronóstico de $T$, que ya está emitido al cierre de $D$; nunca incluye el FC de $T$. Las anomalías se estandarizan con la media $m_j$ y la desviación $s_j$ del entrenamiento: $z_j=(a_j-m_j)/s_j$.

**Predicción y objetivo.**

$$
\hat y_{p,T}=L_{p,D}+\beta_0+\beta^\top z_{p,T},\qquad
(\hat\beta_0,\hat\beta)=\arg\min_{\beta_0,\beta}\ \sum_{i=1}^{n}\bigl(r_i-\beta_0-\beta^\top z_i\bigr)^2+\alpha\lVert\beta\rVert_2^2,
$$ (eq-k1)

con $r_i=y_i-L_i$ el residuo de nivel de la fila de entrenamiento $i$.

:::{prf:proposition} Convexidad y solución cerrada de K1
:label: prop-k1

Si $\alpha>0$, el objetivo de la [ecuación %s](#eq-k1) es estrictamente convexo y su único minimizador es

$$
\hat\beta_0=\bar r,\qquad \hat\beta=\bigl(Z^\top Z+\alpha I_k\bigr)^{-1}Z^\top(r-\bar r\,\mathbf{1}),
$$

donde $Z\in\mathbb{R}^{n\times k}$ tiene las anomalías estandarizadas (columnas centradas en el entrenamiento).
:::

:::{prf:proof}
Sea $J(\beta_0,\beta)=\lVert r-\beta_0\mathbf{1}-Z\beta\rVert^2+\alpha\lVert\beta\rVert^2$. Su hessiana es

$$
\nabla^2J=2\begin{pmatrix} n & \mathbf{1}^\top Z\\ Z^\top\mathbf{1} & Z^\top Z+\alpha I_k\end{pmatrix}
=2\begin{pmatrix} n & 0\\ 0 & Z^\top Z+\alpha I_k\end{pmatrix},
$$

porque las columnas de $Z$ están centradas ($Z^\top\mathbf{1}=0$). Ambos bloques son definidos positivos ($n>0$; $Z^\top Z\succeq0$ y $\alpha>0$), de modo que $J$ es estrictamente convexa y tiene un único mínimo. Igualando el gradiente a cero: $\partial J/\partial\beta_0=-2\,\mathbf{1}^\top(r-\beta_0\mathbf{1}-Z\beta)=0$ da $\beta_0=\bar r$ (usando $\mathbf{1}^\top Z=0$), y $\partial J/\partial\beta=-2Z^\top(r-\beta_0\mathbf{1}-Z\beta)+2\alpha\beta=0$ da $(Z^\top Z+\alpha I)\beta=Z^\top(r-\bar r\mathbf{1})$.
:::

:::{prf:algorithm} K1: ajuste y predicción
:label: alg-k1

**Entrada:** series $y_{p,t}$ y $x^{\mathrm{pr}}_{p,t}$ por planta; penalización $\alpha$; fecha de corte del entrenamiento.
**Salida:** $\hat y_{p,T}$ para cada fila de consulta.

1. Para cada planta $p$ y fecha objetivo $T$: $L_{p,D}\gets$ media de $y_{p,t}$ en los 30 días que terminan en $D=T-1$; $a_{p,T}\gets$ [ecuación %s](#eq-anomalia).
2. $(m,s)\gets$ media y desviación de $a$ en el entrenamiento; $z\gets(a-m)/s$.
3. $r_i\gets y_i-L_i$ en el entrenamiento.
4. $\hat\beta_0\gets\bar r$; $\hat\beta\gets(Z^\top Z+\alpha I)^{-1}Z^\top(r-\bar r\mathbf{1})$.
5. Para cada consulta $(p,T)$: $\hat y_{p,T}\gets L_{p,D}+\hat\beta_0+\hat\beta^\top z_{p,T}$. El nivel usa observaciones hasta $D$; $\hat\beta$ no se reajusta durante la prueba.
:::

**Interpretación y supuestos.** La [ecuación %s](#eq-k1) equivale a un modelo de panel con efecto de planta que varía lentamente, $y_{p,T}=\mu_p(T)+\beta^\top(x^{\mathrm{pr}}_{p,T}-\xi_p(T))+\varepsilon_{p,T}$, con $\mu_p$ y $\xi_p$ sustituidos por medias móviles de 30 días. Restar la media propia cumple el papel de la transformación intragrupos de los modelos de efectos fijos sin estimar un parámetro por planta. Supuestos: (i) el nivel cambia poco en 30 días; (ii) la respuesta a una anomalía es lineal y común a todas las plantas; (iii) el coeficiente del nivel es 1; (iv) la relación entre anomalías y residuo es estable entre entrenamiento y pronóstico. La pérdida cuadrática estima la media condicional.

**Complejidad.** Seis parámetros entrenables y diez constantes de estandarización. Ajuste $O(nk^2+k^3)$ en tiempo y $O(nk)$ en memoria; predicción $O(k)$ por fila. Medido en el cuaderno 21: unos 3 ms de ajuste con 10,547 filas y 1.6 kB serializado.

**Qué es propio.** El estimador es `Ridge` de scikit-learn. Lo propio es la parametrización: respuesta $r=y-L$, anomalías del pronóstico respecto de su media móvil y coeficiente fijo del nivel. Por eso la condición R2 de la guía (implementación propia) no se acredita para K1 ([capítulo 9](09_conclusiones.md)).

## W1: regresión local ponderada por planta y grupo

Para una consulta $q$ de la planta $p_q$ con anomalías estandarizadas $z_q$, el peso de la fila de entrenamiento $i$ es

$$
w_i(q)=\exp\!\Bigl(-\frac{\lVert z_i-z_q\rVert_2^2}{h^2}\Bigr)c_i(q)+10^{-8},\qquad
c_i(q)=\begin{cases}1,& p_i=p_q\\ \rho_g,& p_i\ne p_q,\ g(p_i)=g(p_q)\\ \rho_o,& g(p_i)\ne g(p_q)\end{cases}
$$ (eq-w1pesos)

y los coeficientes locales resuelven

$$
\hat\theta(q)=\arg\min_{\theta_0,\theta}\ \sum_{i=1}^{n}w_i(q)\bigl(r_i-\theta_0-\theta^\top z_i\bigr)^2+\gamma\,S(q)\,\lVert\theta\rVert_2^2,\qquad S(q)=\sum_i w_i(q).
$$ (eq-w1)

Con $\tilde Z=[\mathbf{1}\ Z]$, $W(q)=\mathrm{diag}(w(q))$ y $\Lambda=\mathrm{diag}(0,1,\dots,1)$, la solución es $\hat\theta(q)=(\tilde Z^\top W(q)\tilde Z+\gamma S(q)\Lambda)^{-1}\tilde Z^\top W(q)r$ y la predicción $\hat y_q=L_{p_q,D}+\hat\theta_0(q)+\hat\theta(q)^\top z_q$. La penalización proporcional a $S(q)$ hace que su efecto no dependa del número efectivo de vecinos; $\gamma=10^{-3}$ es fijo.

:::{prf:proposition} W1 contiene a K1
:label: prop-w1k1

Si $\rho_g=\rho_o=1$ y $h\to\infty$, la predicción de W1 converge a la de K1 con $\alpha=\gamma n$.
:::

:::{prf:proof}
Con $\rho_g=\rho_o=1$, $c_i(q)=1$ para todo $i$. Si $h\to\infty$, $\exp(-\lVert z_i-z_q\rVert^2/h^2)\to1$, de modo que $w_i(q)\to w=1+10^{-8}$ para todo $i$ y $S(q)\to nw$. El objetivo de la [ecuación %s](#eq-w1) queda $w\bigl[\sum_i(r_i-\theta_0-\theta^\top z_i)^2+\gamma n\lVert\theta\rVert^2\bigr]$, que tiene el mismo minimizador que la [ecuación %s](#eq-k1) con $\alpha=\gamma n$ y no depende de $q$. Las dos predicciones coinciden porque ambas suman el mismo nivel $L_{p_q,D}$. La prueba `test_w1_tiende_a_k1_con_h_infinito_y_rho_uno` lo comprueba numéricamente.
:::

Con $h\to\infty$ y $\rho_g,\rho_o\to0$, W1 es una regresión separada por planta (prueba `test_w1_con_rho_cero_es_regresion_por_planta`). Los pesos $\rho$ interpolan entre ambos extremos, como un agrupamiento parcial. Para cada $q$ el problema es estrictamente convexo: su hessiana es $2(\tilde Z^\top W(q)\tilde Z+\gamma S(q)\Lambda)$ y, para $v=(v_0,v_\theta)\neq0$, $v^\top(\cdot)v$ vale al menos $\gamma S(q)\lVert v_\theta\rVert^2>0$ si $v_\theta\neq0$, y $v_0^2S(q)>0$ si $v_\theta=0$, porque todos los pesos son positivos.

:::{prf:algorithm} W1: predicción para una consulta
:label: alg-w1

**Entrada:** entrenamiento estandarizado $\{(z_i,r_i,p_i,g_i)\}$; consulta $(z_q,p_q,g_q,L_q)$; $h$, $\rho_g$, $\rho_o$, $\gamma$.
**Salida:** $\hat y_q$.

1. Para cada $i$: $c_i\gets1$ si $p_i=p_q$, $\rho_g$ si $g_i=g_q$, $\rho_o$ en otro caso; $w_i\gets\exp(-\lVert z_i-z_q\rVert^2/h^2)c_i+10^{-8}$.
2. $G\gets\sum_iw_i\tilde z_i\tilde z_i^\top+\gamma(\sum_iw_i)\Lambda$; $b\gets\sum_iw_i\tilde z_ir_i$, con $\tilde z_i=(1,z_i)$.
3. $\hat\theta\gets G^{-1}b$ (sistema de $(k+1)\times(k+1)$).
4. $\hat y_q\gets L_q+\hat\theta_0+\hat\theta_{1:k}^\top z_q$.
:::

**Supuestos y complejidad.** W1 conserva los supuestos (i), (iii) y (iv) de K1 y sustituye el (ii) por linealidad local. El ajuste solo estandariza y guarda el entrenamiento, con memoria $O(nk^2)$ porque precalcula los productos $\tilde z_i\tilde z_i^\top$. Cada predicción cuesta $O(nk^2+k^3)$: es un método perezoso que paga su costo en la inferencia (unos 290 µs por fila en el cuaderno 21, el más lento de los nueve modelos). Hiperparámetros: $h$ y el par $(\rho_g,\rho_o)$; la selección original usó $h\in\{0.5,1,2\}$ y dos pares, y el estudio de optimizadores amplió $h$ a $[0.25,8]$ ([capítulo 3](03_benchmarks.md)).

**Qué es propio.** La regresión localmente ponderada es una técnica establecida {cite:p}`cleveland1979robust,atkeson1997locally`. Lo propio es aplicarla al residuo de nivel en el espacio de anomalías, el peso por identidad de planta y grupo, y la penalización escalada por la suma de pesos. W1 usa la identidad de planta y grupo, que las referencias no reciben: su comparación con ellas es entre sistemas completos.

## Código

Las dos clases están en `src/original_model.py` y son compatibles con scikit-learn (`fit`, `predict`, `get_params`, `clone`, `Pipeline` y `GridSearchCV`). Son una adaptación de las del cuaderno 17 con la misma lógica numérica y validación de entradas añadida. La celda siguiente muestra la predicción de W1, que es el cálculo propio.

```{code-cell} python
# Código de la predicción de W1 tal como está en el repositorio (se lee del módulo, no se copia a mano).
import inspect
import apoyo
from src.original_model import ModeloAnomalias, RegresionLocal

print(inspect.getsource(RegresionLocal.predict))
```

La celda siguiente comprueba la [Proposición %s](#prop-k1) con datos sintéticos: la solución cerrada coincide con la de la clase, y W1 con $h$ muy grande y $\rho_g=\rho_o=1$ coincide con K1 con $\alpha=\gamma n$.

```{code-cell} python
# Verificación numérica de la solución cerrada de K1 y del límite W1 -> K1 (datos sintéticos, semilla fija).
import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
n, k = 400, 5
anom = rng.normal(size=(n, k))
nivel = rng.uniform(0.15, 0.30, size=n)
y = nivel + 0.02 + anom @ np.array([0.03, -0.01, 0.0, 0.005, 0.0]) + rng.normal(0, 0.01, size=n)
X = pd.DataFrame(np.column_stack([nivel, anom]), columns=["media30", "anom_rad", "anom_temp", "anom_nube", "anom_humedad", "anom_viento"])

alpha = 10.0
k1 = ModeloAnomalias(alpha=alpha).fit(X, y)
Z = (anom - anom.mean(0)) / anom.std(0)
r = y - nivel
beta = np.linalg.solve(Z.T @ Z + alpha * np.eye(k), Z.T @ (r - r.mean()))
print("K1, diferencia máxima con la solución cerrada:", float(np.abs(beta - k1.coef_).max()), "| intercepto:", abs(r.mean() - k1.intercept_))

gamma = 1e-3
Xw = X.assign(code=rng.choice(["a", "b", "c"], size=n), grupo="g")
w1 = RegresionLocal(h=1e6, rho=(1.0, 1.0), ridge=gamma).fit(Xw, y)
k1_eq = ModeloAnomalias(alpha=gamma * n).fit(X, y)
print("W1 (h→∞, ρ=1) frente a K1 (α=γn), diferencia máxima de predicción:", float(np.abs(w1.predict(Xw) - k1_eq.predict(X)).max()))
```

## Pruebas unitarias

`tests/test_original_model.py` contiene 19 casos, dentro de una batería de 30 (`tests/`) que se ejecuta en integración continua (`.github/workflows/tests.yml`). El 16 de noviembre de 2026 la batería completa pasó en 2.8 s con el entorno de `.venv`. Las pruebas del modelo verifican, entre otras cosas, la recuperación de pendientes conocidas, la solución cerrada de K1 frente a la ecuación del manuscrito, que el objetivo es mínimo en esa solución, el límite W1 → K1, el caso por planta con $\rho\to0$, la igualdad con una implementación independiente de mínimos cuadrados ponderados, la invariancia al orden de las filas, el rechazo de NaN e infinitos y la integración con `Pipeline` y `GridSearchCV`.

```{code-cell} python
# Lista de pruebas del modelo (nombres de las funciones de test).
import re
texto = (apoyo.RAIZ / "tests" / "test_original_model.py").read_text(encoding="utf-8")
pd.DataFrame({"prueba": re.findall(r"^def (test_\w+)", texto, flags=re.M)})
```

La reproducción de las predicciones guardadas está en `experiments/verificar_modelos_originales.py`, y el cuaderno 21 la repite: K1 con el $\alpha$ de 07 y W1 con la configuración de 19 reproducen las predicciones de la prueba con diferencias de 0 y $1.7\times10^{-16}$.

## Ablación

La ablación, calculada solo con los bloques de desarrollo (cuaderno de ablación, `resultados/ablacion_k1/`), quita o reemplaza un componente de K1 a la vez ([Figura %s](#fig-ablacion)).

:::{figure} figuras/res_ablacion.png
:label: fig-ablacion
:width: 100%
Ablación de K1 en desarrollo: (a) MAE macro de cada variante (línea: Random Forest con $B_2$); (b) diferencia de MAE de cada variante frente a V3 (K1 completo) con IC 95 % bootstrap.
:::

::::{table} Ablación de K1 en desarrollo (predicciones de D1–D3 agrupadas, 262 días): MAE macro de cada variante, diferencia con K1 completo (V3) con IC 95 %, y valores $p$ con Holm de DM-HLN, Giacomini–White y Wilcoxon. Fuente: `resultados/ablacion_k1/resumen_macro.csv` y `resultados/pruebas_guia/pruebas_ablacion.csv`.
:label: tbl-ablacion
```{code-cell} python
abl = apoyo.leer("resultados/ablacion_k1/resumen_macro.csv")[["modelo", "MAE_macro", "RMSE_macro", "MBE_macro"]]
pr = apoyo.leer("resultados/pruebas_guia/pruebas_ablacion.csv")
pr = pr[["referencia", "diferencia_mae", "ic95_inf", "ic95_sup", "dm_hln_p_holm", "gw_p_holm", "wilcoxon_p_holm", "bayes_planta_p_rope"]]
t = abl.merge(pr, left_on="modelo", right_on="referencia", how="left").drop(columns="referencia")
t = t.rename(columns={"MAE_macro": "MAE", "RMSE_macro": "RMSE", "MBE_macro": "MBE", "diferencia_mae": "V3 − variante",
                      "ic95_inf": "IC inf", "ic95_sup": "IC sup", "dm_hln_p_holm": "p DM-HLN (Holm)", "gw_p_holm": "p GW (Holm)",
                      "wilcoxon_p_holm": "p Wilcoxon (Holm)", "bayes_planta_p_rope": "P(ROPE)"})
apoyo.tabla(t.sort_values("MAE"), decimales=4, indice="modelo")
```
::::

La media móvil de 30 días sola (V0) tiene MAE 0.0486 y añadirle un intercepto (V1) no la cambia. La anomalía de radiación sola (V2) la baja a 0.0349, un 28 % menos; las otras cuatro anomalías (V3, K1 completo, 0.0347) añaden 0.0002, una diferencia que ninguna prueba distingue. Liberar el coeficiente del nivel (V4) da el mismo error: fijarlo en 1 no pierde nada. Usar el pronóstico en bruto (V5, +14 %) o anomalías de 7 días (V6, +9 %) empeora con $p_{\mathrm{Holm}}<10^{-4}$. El aporte de K1 está en restar al pronóstico su propia media móvil de 30 días y aplicar esa anomalía al nivel reciente. SHAP en la prueba ([sección %s](#sec-interpretabilidad)) lleva a la misma lectura: en K1 y W1 el nivel pesa un 44–45 % y la anomalía de radiación un 31 %.

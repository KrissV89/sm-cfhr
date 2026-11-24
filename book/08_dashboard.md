---
title: Dashboard y despliegue
short_title: 8. Dashboard
kernelspec:
  name: python3
  display_name: Python 3
---

# Dashboard y despliegue

:::{important} Estado del despliegue
La aplicación funciona en local y está desplegada en Render (`render.yaml`, plan gratuito): <https://pronostico-fc-solar.onrender.com>. Con el plan gratuito el servicio se suspende tras un rato sin tráfico y el primer acceso tarda cerca de un minuto.
:::

## Arquitectura

El dashboard es una aplicación Dash multipágina (`app.py` y `pages/`) que expone `server` para `gunicorn app:server`. Al iniciar carga una sola vez los extractos de `data/` y los nueve modelos serializados de `artifacts/`; nunca entrena ni optimiza. La lógica está separada de las páginas:

| Capa | Archivos | Función |
|---|---|---|
| Datos | `src/data.py`, `src/config.py` | Carga y caché de `data/*.csv(.gz)` y de los artefactos |
| Variables | `src/features.py`, `src/series.py` | Construcción de las variables $D{+}1$ sin fuga y pruebas de series |
| Modelos | `src/original_model.py`, `src/pipelines.py` | Clases K1 y W1, pipelines de las referencias |
| Predicción y explicación | `src/prediccion.py`, `src/explicacion.py` | Predicción con intervalo conforme y SHAP exacto por enumeración de coaliciones |
| Gráficos | `src/plots.py` | Figuras Plotly |
| Páginas | `pages/contexto.py`, `pages/eda.py`, `pages/modelos.py` | Contexto, análisis exploratorio y modelos |
| Construcción | `experiments/serializar_artefactos.py`, `experiments/construir_datos_app.py` | Reentrenan los nueve pipelines con el entrenamiento hasta el 31 de julio de 2026 y preparan `data/` |

**Contexto.** Pregunta, objetivos, datos, metodología, limitaciones y autoría. Declara que la ventana de prueba ya se usó en análisis previos.

**EDA.** Distribución del FC, explorador bivariado, correlación, faltantes y atípicos, serie de tiempo y pruebas de estacionariedad.

**Modelos.** Tabla de líderes, predicción en vivo para una planta y fecha con intervalo conforme, diagnóstico del modelo seleccionado en la prueba (observado frente a pronosticado, residuos, autocorrelación), SHAP global, tabla del MCS, pruebas de hipótesis con el veredicto sobre R3 («no se cumple») y descripción de K1 y W1.

:::{figure} figuras/dash_contexto.png
:label: fig-dash-contexto
:width: 90%
Página de contexto del dashboard (captura local).
:::

:::{figure} figuras/dash_modelos.png
:label: fig-dash-modelos
:width: 90%
Página de modelos: tabla de líderes, predicción en vivo y diagnóstico (captura local).
:::

:::{figure} figuras/dash_eda.png
:label: fig-dash-eda
:width: 90%
Página de análisis exploratorio (captura local).
:::

## Decisiones de diseño

- **Sin cálculo en el servidor.** La app solo carga resultados y modelos ya serializados. Así arranca en el plan gratuito de Render y no puede alterar resultados.
- **Predicción en vivo con los mismos pipelines.** Cada artefacto es el pipeline completo (escalado incluido) reentrenado con fechas objetivo hasta el 31 de julio de 2026. La entrada se valida contra el esquema y los rangos físicos de `artifacts/*_metadata.json`.
- **Intervalos conformes.** El intervalo usa el cuantil conforme de los residuos absolutos de D1–D3. Como esa calibración pierde cobertura en la prueba ([capítulo 7](07_evaluacion_ampliada.md)), la app lo declara junto al intervalo.
- **SHAP exacto.** Con 6 a 8 variables, la app calcula los valores de Shapley por enumeración de coaliciones; se validó contra `shap.explainers.Exact` con diferencias máximas de $2.5\times10^{-16}$ (Random Forest) y $4.5\times10^{-17}$ (K1) en 100 filas (`data/shap_validacion.json`).
- **Resultados negativos visibles.** La página de modelos muestra el MCS y el veredicto de R3 sin filtrar.
- **Accesibilidad.** Cada gráfico tiene texto alternativo (`aria-label`) y la maqueta se adapta a pantallas estrechas (captura `dash_modelos_movil.png`).

## Artefactos y coherencia con la selección

Los nueve artefactos usan los hiperparámetros de la **selección original**, elegidos con la búsqueda en rejilla de los cuadernos 07, 19 y 20 en el bloque de prueba, la misma que alimenta la [Tabla %s](#tbl-principal) y el manuscrito. La celda siguiente comprueba, leyendo solo archivos guardados, que: (i) cada artefacto pasó la prueba de ida y vuelta y reproduce la predicción guardada; (ii) el MAE de prueba de la tabla de líderes de la app es el de la tabla del manuscrito; y (iii) sus hiperparámetros difieren de los que eligió el estudio *post hoc* de optimizadores (cuaderno 21).

::::{table} Coherencia de los artefactos del dashboard. MAE de prueba de la app (`data/leaderboard.csv`) frente al del manuscrito (`resultados/pruebas_guia/tabla_desempeno.csv`), y hiperparámetros frente a la mejor combinación del cuaderno 21 (`resultados/optimizadores/mejores_combinaciones.csv`).
:label: tbl-coherencia
```{code-cell} python
import json
import apoyo
import pandas as pd

lb = apoyo.leer("data/leaderboard.csv")
indice = apoyo.leer_json("artifacts/indice.json")["modelos"]
ida = {m["modelo"]: (m["ida_y_vuelta"], m["dif_guardada"]) for m in indice}
td = apoyo.leer("resultados/pruebas_guia/tabla_desempeno.csv")
td = td[td.principal].set_index("clave").MAE_macro
mej = apoyo.leer("resultados/optimizadores/mejores_combinaciones.csv").set_index("modelo")
nombre21 = {"K1 Modelo de anomalías": "K1", "W1 Regresión local": "W1"}
filas = []
for _, r in lb.iterrows():
    f21 = mej.loc[nombre21.get(r.modelo, r.modelo)]
    filas.append({"modelo": r.modelo, "ida y vuelta": ida[r.modelo][0], "dif. con predicción guardada": ida[r.modelo][1],
                  "MAE prueba app": r.MAE_prueba, "MAE prueba manuscrito": td[r.modelo],
                  "hiperparámetros app": r.hiperparametros, "optimizador 21": f21.optimizador,
                  "mismos hp que 21": json.loads(r.hiperparametros) == json.loads(f21.hp_FINAL),
                  "MAE prueba 21": f21.MAE_prueba})
coh = pd.DataFrame(filas)
coh["app − manuscrito"] = coh["MAE prueba app"] - coh["MAE prueba manuscrito"]
assert coh["ida y vuelta"].all() and coh["app − manuscrito"].abs().max() < 1e-12
apoyo.tabla(coh, decimales={"MAE prueba app": 5, "MAE prueba manuscrito": 5, "MAE prueba 21": 5}, indice="modelo")
```
::::

Los nueve artefactos reproducen las predicciones guardadas y el MAE de la tabla del manuscrito con diferencia nula; ninguno usa los hiperparámetros del cuaderno 21. Es la decisión coherente con el protocolo: la selección original se fijó antes de la prueba, mientras el cuaderno 21 es un estudio posterior. Las diferencias de MAE de prueba entre ambas configuraciones son menores de 0.0005 en todos los modelos.

:::{note} Qué selección muestra la app al reconstruir sus datos
La app muestra la selección original (rejilla de los cuadernos 07, 19 y 20, fijada antes de abrir la prueba), igual que el manuscrito. El estudio de optimizadores del cuaderno 21 es posterior y no se copia a la app: `experiments/construir_datos_app.py` ya no copia `mejores_combinaciones.csv` a `data/` (y borra la copia si existe), y `experiments/serializar_artefactos.py` ignora a propósito `resultados/optimizadores/hiperparametros_mejores.json`. Así la tabla de líderes y los artefactos usan los mismos hiperparámetros. Al cerrar este libro, `data/mejores_combinaciones.csv` no existe.
:::

## Pruebas de la aplicación

`tests/test_app.py` (4 casos) comprueba que la metadata de los artefactos está completa y pasa la ida y vuelta, que las páginas y sus *callbacks* responden, que la predicción en vivo acepta entradas válidas y rechaza las inválidas, y que las anomalías que deriva la app coinciden con las de la tabla de trabajo; forma parte de la batería de 30 pruebas que pasó completa el 16 de noviembre de 2026. El agente que construyó la app informó además respuestas HTTP 200 con `gunicorn` y la equivalencia numérica de las predicciones de la app con las guardadas (`experiments/salidas/verificacion_modelos_originales.json`: diferencia máxima 0 en K1 y W1 sobre 1,561 filas).

## Despliegue en Render

1. Crear un servicio web desde el repositorio, que Render detecta con `render.yaml` (plan gratuito, Python 3.14.10).
2. Comando de construcción: `pip install -r requirements.txt` (versiones exactas, las mismas que serializaron `artifacts/`).
3. Comando de inicio: `gunicorn app:server --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT`.
4. Comprobar `/`, `/eda` y `/modelos`, y anotar la URL en la introducción de este libro, en `README.md` y en `CITATION.cff`.

`data/` y `artifacts/` deben estar versionados para que el servicio arranque; `.gitignore` tiene excepciones para ellos. Un servicio gratuito se suspende tras un periodo sin tráfico y tarda en reanudarse.

## Ficha del modelo y declaración de datos

`MODEL_CARD.md` describe K1 y W1: uso previsto, usos no recomendados, datos, métricas, limitaciones y consideraciones éticas. `data/DATA_STATEMENT.md` describe los extractos de `data/`, su procedencia (XM, Open-Meteo y Global Energy Monitor) y sus condiciones de uso. Ambos tienen autores y licencia **POR COMPLETAR**.

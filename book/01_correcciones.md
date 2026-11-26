---
title: Trazabilidad de correcciones
short_title: 1. Correcciones
kernelspec:
  name: python3
  display_name: Python 3
---

# Trazabilidad de correcciones

## Autoauditoría frente a la guía del proyecto

El equipo contrastó el proyecto con la guía del curso (`docs/entregables/AUDITORIA_GUIAS_2_Y_3.md`). La [Tabla %s](#tbl-autoauditoria) reproduce cada hallazgo, la acción posterior y su estado al cerrar el libro. «Pendiente» significa que el requisito sigue sin cumplirse; el libro no lo presenta como resuelto.

:::{table} Autoauditoría: hallazgo en el momento de la auditoría, acción posterior y evidencia.
:label: tbl-autoauditoria

| Requisito de la guía | Hallazgo de la auditoría | Acción realizada | Estado | Ubicación | Evidencia |
|---|---|---|---|---|---|
| Siete modelos de regresión × cuatro optimizadores (28 combinaciones) | Solo GridSearchCV; Lasso no evaluado; árbol de decisión solo como componente | Lasso, Ridge bayesiana y árbol de decisión añadidos como referencias con el mismo protocolo (cuaderno 20). Estudio de cuatro optimizadores (rejilla, aleatoria, bayesiana TPE y genético DEAP) con protocolo registrado antes de ejecutar (cuaderno 21) | Hecho (estudio *post hoc*: la prueba ya se conocía) | [Capítulo 3](03_benchmarks.md) | `resultados/referencias_curso/`; `resultados/optimizadores/tabla_maestra.csv` |
| Validación anidada y control de fuga | Implementado con limitaciones documentadas | Auditoría de fuga explícita ([Tabla %s](#tbl-fuga)) y comprobaciones ejecutables en el capítulo 2 | Hecho | [Capítulo 2](02_datos.md) | `src/features.py`; `tests/test_features_series_explicacion.py` |
| Comparación de optimizadores, convergencia y eficiencia | Sin comparación de los cuatro optimizadores | Cuaderno 21: curvas de convergencia, exceso relativo a 5, 10, 20 y 30 evaluaciones, tiempo, memoria y tamaño serializado | Hecho | [Capítulo 3](03_benchmarks.md) | `resultados/optimizadores/convergencia_resumen.csv`; `resultados/optimizadores/eficiencia.csv` |
| Interpretabilidad LIME y SHAP | No acreditada | SHAP exacto por enumeración de coaliciones en el dashboard, validado contra `shap.explainers.Exact`; SHAP y LIME de la mejor combinación en el cuaderno 21 | Hecho (LIME en tres filas: ilustrativo) | [Capítulo 3](03_benchmarks.md), [capítulo 8](08_dashboard.md) | `src/explicacion.py`; `data/shap_validacion.json`; `resultados/optimizadores/concordancia_shap_lime.csv` |
| Selección por modelo y artefactos serializados | Sin paquete `artifacts/` | Nueve pipelines serializados con `joblib`, `metadata.json` y prueba de ida y vuelta | Hecho, con los hiperparámetros de la selección original (GridSearchCV de 07, 19 y 20); no se sustituyeron por los del estudio *post hoc* 21 | [Capítulo 8](08_dashboard.md) | `artifacts/indice.json` |
| R1, novedad | Novedad no demostrada | Revisión de literatura ampliada (consultas Q19 a Q24); la conclusión no cambia | No acreditado | [Capítulo 4](04_literatura.md) | `notebooks/legacy/actuales/09_revision_bibliografica_custom.ipynb` |
| R2, implementación propia | W1 programado con NumPy y SciPy; K1 delega en `Ridge` | Clases en `src/original_model.py` con docstrings y pruebas unitarias en integración continua | Parcial (W1 sí, K1 no) | [Capítulo 5](05_modelo_original.md) | `tests/test_original_model.py`; `.github/workflows/tests.yml` |
| Presupuesto equivalente, mismas variables, diez semillas | Rejillas de distinto tamaño; una semilla; W1 recibe identidad espacial | Estudio de diez semillas (cuaderno 22); presupuesto igualado de 30 evaluaciones en el cuaderno 21. W1 sigue recibiendo la identidad de planta y grupo | Parcial | [Capítulo 6](06_validacion.md), [capítulo 7](07_evaluacion_ampliada.md) | `resultados/robustez/estabilidad_semillas_resumen.csv` |
| Tres conjuntos de datos adicionales | Ninguno evaluado | Tres conjuntos de UCI con criterios fijados y huella SHA-256 antes de descargar (cuaderno 23) | Hecho; resultado desfavorable | [Capítulo 7](07_evaluacion_ampliada.md) | `resultados/datasets_adicionales/PROTOCOLO.md` |
| R3, tres pruebas, una global, efectos | Sin DM-HLN, GW ni MCS | DM-HLN, DM-HAC, Giacomini–White, Wilcoxon, bayesiana con ROPE y MCS, con Holm y efectos (análisis *post hoc*) | Calculado; R3 no se cumple | [Capítulo 6](06_validacion.md) | `resultados/pruebas_guia/` |
| Ablación, sensibilidad, robustez | Parcial | Sensibilidad a hiperparámetros, semillas, ruido, atípicos, tamaño de muestra, costo e intervalos conformes (cuaderno 22) | Hecho | [Capítulo 7](07_evaluacion_ampliada.md) | `resultados/robustez/` |
| Dashboard en Dash y despliegue en Render | Solo un tablero exploratorio | Aplicación multipágina (Contexto, EDA, Modelos) con predicción en vivo; `render.yaml` | Aplicación: hecha. Despliegue: hecho en Render (URL en el [capítulo 8](08_dashboard.md)) | [Capítulo 8](08_dashboard.md) | `app.py`; `pages/`; `render.yaml` |
| Jupyter Book | Sin configuración | Este libro | Hecho | — | `book/` |
| CI, ficha del modelo, declaración de datos, licencia, CITATION, DOI y firma | Parcial | CI, `MODEL_CARD.md`, `data/DATA_STATEMENT.md`, `CITATION.cff` y `LICENSE` creados | Licencia (MIT) y autores: hechos. DOI y firma: pendiente de depósito | [Capítulo 10](10_reproducibilidad.md) | `CITATION.cff`; `LICENSE` |
:::

La celda siguiente comprueba que los archivos citados como evidencia existen en el repositorio en el momento de construir el libro. Una fila con `False` indica una evidencia que falta.

```{code-cell} python
# Comprueba la existencia de cada archivo o carpeta citado como evidencia en las tablas de este capítulo.
import apoyo
import pandas as pd

evidencias = [
    "docs/entregables/AUDITORIA_GUIAS_2_Y_3.md", "resultados/referencias_curso", "resultados/optimizadores/PROTOCOLO.md",
    "src/features.py", "tests/test_features_series_explicacion.py", "src/explicacion.py", "data/shap_validacion.json",
    "artifacts/indice.json", "notebooks/legacy/actuales/09_revision_bibliografica_custom.ipynb", "src/original_model.py",
    "tests/test_original_model.py", ".github/workflows/tests.yml", "resultados/robustez/estabilidad_semillas_resumen.csv",
    "resultados/datasets_adicionales/PROTOCOLO.md", "resultados/pruebas_guia", "resultados/robustez", "app.py", "pages",
    "render.yaml", "MODEL_CARD.md", "data/DATA_STATEMENT.md", "CITATION.cff", "LICENSE",
    "resultados/pronostico_d1/seleccion_desarrollo.json",
    # Archivos que producirá el cuaderno 21 (optimizadores); False mientras no termine
    "resultados/optimizadores/mejores_combinaciones.csv", "resultados/optimizadores/tabla_maestra.csv",
]
pd.DataFrame({"evidencia": evidencias, "existe": [apoyo.existe(e) for e in evidencias]}).set_index("evidencia")
```

## Auditoría de fuga de información

La guía pide revisar cuatro fuentes de fuga antes de congelar los resultados. La [Tabla %s](#tbl-fuga) resume cómo se controla cada una en la Tarea 2; las comprobaciones ejecutables están en la [sección de fuga del capítulo 2](#sec-fuga-datos).

:::{table} Auditoría de fuga de información en la tarea $D{+}1$.
:label: tbl-fuga

| Fuente de fuga | Riesgo concreto en este proyecto | Control | Evidencia |
|---|---|---|---|
| Preprocesamiento ajustado sobre todo el conjunto | Estandarización de predictores y de anomalías con media y desviación de todas las fechas | Todo lo que tiene parámetros (escalado, coeficientes, hiperparámetros) se ajusta dentro del entrenamiento de cada partición: `Pipeline` en las referencias y `StandardScaler` dentro de `fit` en K1 y W1. Las medias móviles no tienen parámetros aprendidos y solo usan datos anteriores al origen | `src/original_model.py`; cuadernos 07 y 17 |
| Variables que codifican la respuesta | Energía, capacidad, FC del día objetivo o promedios del FC por planta calculados con todo el periodo | Lista explícita de columnas excluidas; el historial usa el FC hasta $D$; prueba de perturbación de la etiqueta ([capítulo 2](#sec-fuga-datos)) | `tests/test_features_series_explicacion.py::test_no_hay_fuga_de_la_etiqueta` |
| Información del día objetivo en los predictores meteorológicos | La Historical Forecast API empalma las primeras horas de cada corrida y casi coincide con el clima realizado (correlación 0.976 con la irradiación del reanálisis) | Se usa la Previous Runs API: valor pronosticado 24 h antes de la hora de validez (correlación 0.615 en la comprobación previa) | `datos_pronostico/procedencia.json`; [capítulo 2](02_datos.md) |
| Duplicados entre particiones | Dos pares de plantas comparten celda meteorológica; cinco plantas nuevas comparten coordenadas o celda con originales | En la Tarea 2 la partición es temporal: una fecha objetivo está en un solo bloque. En las evaluaciones espaciales se deja fuera el grupo completo (plantas a menos de 100 km) y se reporta la sensibilidad sin las plantas que comparten celda | `resultados/pronostico_d1_plantas_nuevas/sensibilidades.csv` |
| Mezcla temporal entre entrenamiento y prueba | Medias móviles de 30 días de la validación que se solapan con el entrenamiento | Bloques de origen móvil; validación interna con ventana creciente y hueco de 30 días; la prueba reservada se usó una vez y el cuaderno impide repetirla | `resultados/pronostico_d1/seleccion_desarrollo.json` |
| Mezcla espacial | Plantas cercanas con FC correlacionado (mediana 0.76 dentro del grupo) | Grupos espaciales por enlace simple a 100 km como unidad de exclusión en la Tarea 1 y en la evaluación de plantas no vistas | Cuadernos 03 y 11 |
| Reutilización de la prueba para seleccionar | Elegir modelos o pruebas mirando la ventana de agosto a octubre | Selección solo con desarrollo. Excepciones declaradas: W1 se seleccionó con desarrollo después de conocer el resultado de K1 en la prueba, y las pruebas de la guía se calcularon *post hoc* | `resultados/pruebas_guia/PROTOCOLO.md` |
:::

## Cambios de resultados y del orden de los modelos

Ninguna corrección de fuga cambió cifras ya reportadas: los controles de la [Tabla %s](#tbl-fuga) estaban en el código desde los cuadernos 03 y 07. Lo que cambió entre versiones del análisis, y se reporta en los capítulos correspondientes, es lo siguiente:

- La mejor referencia de desarrollo es Random Forest; en la prueba, SVR (0.0403) queda por delante de Random Forest (0.0406). La comparación decisiva se mantuvo fijada contra Random Forest, como se escribió antes de la prueba ([capítulo 6](06_validacion.md)).
- Lasso, Ridge bayesiana y árbol de decisión entraron en las familias de comparaciones, que quedaron en 10 contrastes para K1 y 11 para W1 (`resultados/pruebas_guia/entorno.json`); más contrastes elevan los valores $p$ ajustados por Holm.
- Las pruebas de la guía (DM-HLN, Giacomini–White, bayesiana, MCS) sustituyen como evidencia principal a las pruebas de Wilcoxon por planta del protocolo original; el veredicto sobre K1 frente a Random Forest pasó de «significativo sin ajustar» (Wilcoxon por planta $p=0.017$) a «no significativo con Holm» en las pruebas temporales.
- Los conjuntos de datos adicionales dieron un resultado desfavorable para K1 y W1, que se reporta sin filtrar ([capítulo 7](07_evaluacion_ampliada.md)).

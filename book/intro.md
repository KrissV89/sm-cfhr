---
title: Introducción
short_title: 0. Introducción
---

# Pronóstico del factor de capacidad fotovoltaico del día siguiente en Colombia

:::{important} Dashboard desplegado
URL pública del dashboard en Render: <https://pronostico-fc-solar.onrender.com>. La aplicación (`app.py`, `pages/`) y su configuración de despliegue (`render.yaml`) están en el repositorio; el [capítulo 8](08_dashboard.md) describe cómo se despliega.
:::

## Resumen ejecutivo

El proyecto pronostica el factor de capacidad (FC) diario de 17 plantas fotovoltaicas colombianas para el día siguiente ($D{+}1$), con la generación que reporta XM, la capacidad registrada y pronósticos meteorológicos archivados de Open-Meteo. Propone dos modelos: K1, que suma al nivel reciente de cada planta (media del FC en los 30 días previos) una corrección lineal con las anomalías del pronóstico respecto de su propia media móvil, y W1, una regresión localmente ponderada del mismo residuo que da más peso a las filas de la misma planta y del mismo grupo espacial.

Resultados principales, todos en MAE macro por planta (unidades de FC) sobre la ventana de prueba de agosto a octubre de 2026 (fuente: `resultados/pruebas_guia/tabla_desempeno.csv`, [capítulo 6](06_validacion.md)):

- K1 obtiene 0.0392 y Random Forest, la mejor referencia de librería en desarrollo, 0.0406: una reducción del 3.3 %. W1 obtiene 0.0388 (4.3 % menos que Random Forest). La media móvil de 30 días queda en 0.0499 y la persistencia en 0.0628.
- En la comparación fijada antes de la prueba, K1 frente a Random Forest, las pruebas sobre la serie diaria no rechazan la igualdad tras la corrección de Holm (DM-HLN $p_{\mathrm{Holm}}=0.076$; Giacomini–White $0.233$). W1 frente a Random Forest sí rechaza en DM-HLN, Giacomini–White y Wilcoxon con Holm, pero esas pruebas son *post hoc*, W1 recibe la identidad de planta y grupo que las referencias no reciben, y el conjunto de confianza de modelos (MCS) al 90 % conserva diez modelos, entre ellos Random Forest, SVR y XGBoost.
- En tres conjuntos de datos adicionales de UCI (calidad del aire en Pekín, *Air Quality* en Italia y tráfico en la I-94), K1 y W1 no son mejores que el mejor modelo de librería en ninguno y son significativamente peores en dos ([capítulo 7](07_evaluacion_ampliada.md)).
- Veredicto frente a la guía del curso ([capítulo 9](09_conclusiones.md)): R1 (novedad) no se acredita; R2 (implementación propia) es defendible para W1, no para K1, que delega el ajuste en `Ridge` de scikit-learn; R3 (superioridad significativa) no se cumple.

## Problema

Sea $p$ una planta y $t$ un día. La variable objetivo es el factor de capacidad diario,

$$
y_{p,t}=\mathrm{FC}_{p,t}=\frac{E_{p,t}}{C_{p,t}\times 24\ \mathrm{h}},
$$ (eq-fc)

con $E_{p,t}$ la energía generada en el día (kWh) y $C_{p,t}$ la capacidad efectiva neta registrada ese día (kW). Es una variable continua, adimensional y acotada ($0<y_{p,t}\le 1.05$ en la base). El proyecto aborda dos tareas de regresión:

1. **Estimación retrospectiva en plantas no vistas (Tarea 1).** Estimar el FC de una planta que no aporta ninguna fila al entrenamiento, con la meteorología del mismo día. Fue el planteamiento inicial.
2. **Pronóstico para el día siguiente (Tarea 2, tarea principal).** Para la fecha objetivo $T=D+1$, predecir $y_{p,T}$ con la información disponible al cierre de $D$:

$$
\mathcal{I}_{p,D}=\{y_{p,s}: s\le D\}\ \cup\ \{x^{\mathrm{pr}}_{p,s}: s\le D+1\},
$$ (eq-info)

donde $x^{\mathrm{pr}}_{p,s}\in\mathbb{R}^5$ es el pronóstico meteorológico para el día $s$ emitido el día anterior (irradiación, temperatura, nubosidad, humedad relativa y viento). Ni el FC ni el clima observado de $T$ pueden entrar en el modelo.

La métrica primaria de las dos tareas es el MAE promediado primero dentro de cada planta y luego entre plantas,

$$
\mathrm{MAE}_{\mathrm{macro}}=\frac{1}{P}\sum_{p=1}^{P}\frac{1}{n_p}\sum_{t\in\mathcal{T}_p}\bigl|y_{p,t}-\hat y_{p,t}\bigr|,
$$ (eq-maemacro)

para que las plantas con más días no dominen el promedio. Un MAE de 0.01 equivale a 0.24 horas equivalentes de capacidad plena por día (ejemplo: en una planta hipotética de 100 MW, 24 MWh diarios). Medido en la prueba, el error total de K1 es el 14.9 % de la energía entregada por las 17 plantas (15.8 % con Random Forest).

## Motivación

La potencia nominal de una planta no dice cuánta energía entregará mañana: la irradiancia, la temperatura, el seguimiento solar, las pérdidas, la disponibilidad y las restricciones de red separan la producción observada de la que sugiere la ficha técnica {cite:p}`mueller2016yield,trigo2021models`. En Colombia hay trabajo de pronóstico de corto plazo con series de generación simuladas a diez minutos {cite:p}`pico2026colombia`; aquí la etiqueta es la generación observada que publica el operador del mercado, la resolución es diaria y la evaluación separa plantas y periodos.

La Tarea 1 mostró la limitación que motiva los modelos propios. Con la meteorología del mismo día, ningún modelo entre seis de librería, cinco ensambles y cinco adaptaciones propias bajó de un MAE de 0.043 en plantas no vistas, y una regresión Ridge con la irradiación como único predictor quedó igual o mejor que todos ([capítulo 3](03_benchmarks.md)). El análisis exploratorio explica por qué: dentro de cada planta el FC diario sigue a la irradiación (Spearman de 0.58 a 0.76), pero entre plantas el FC mediano no sigue a la irradiación mediana (Spearman $-0.08$, 17 plantas) porque depende de la relación DC/AC, del montaje y del registro de capacidad, que la base no observa ([capítulo 2](02_datos.md)). Un modelo común que recibe el pronóstico en bruto debe reconstruir el nivel de cada planta; K1 lo toma de la serie propia y aprende solo la desviación diaria.

## Contribuciones

1. Una base planta-día reproducible que enlaza generación observada de XM, capacidad registrada, meteorología modelada y pronósticos archivados para 17 plantas en siete grupos espaciales (32 plantas y nueve grupos en la ampliación), con sus controles de calidad documentados ([capítulo 2](02_datos.md)).
2. Evidencia de que, con clima del mismo día, ningún modelo evaluado reduce el error en plantas no vistas por debajo de una regresión Ridge con la irradiación como único predictor ([capítulo 3](03_benchmarks.md)).
3. Dos modelos para el pronóstico $D{+}1$, K1 y W1, con función objetivo, solución, supuestos, complejidad, demostración de convexidad y de la relación W1 → K1, código comentado y pruebas unitarias ([capítulo 5](05_modelo_original.md)).
4. Un protocolo temporal con selección en desarrollo y una prueba reservada usada una vez, una ablación que localiza el origen de la mejora, y las pruebas DM-HLN, Giacomini–White, bayesiana y MCS con valores $p$ crudos y ajustados ([capítulo 6](06_validacion.md)).
5. Una evaluación ampliada con plantas no vistas, plantas nuevas, tres conjuntos de datos públicos adicionales, sensibilidad, semillas, ruido e intervalos conformes, que incluye resultados desfavorables ([capítulo 7](07_evaluacion_ampliada.md)).
6. Un dashboard en Dash con predicción en vivo a partir de modelos serializados ([capítulo 8](08_dashboard.md)) y una delimitación explícita de los requisitos de la guía que el proyecto no cumple ([capítulo 9](09_conclusiones.md)).

## Objetivos

- **General.** Determinar si una corrección basada en el nivel reciente de generación reduce el error de los modelos de referencia al pronosticar el FC del día siguiente, y con qué certeza.
- **Específicos.** (i) Construir la base planta-día y la tabla $D{+}1$ sin fuga de información; (ii) ajustar y comparar los modelos de referencia con validación temporal anidada; (iii) formular, implementar y probar K1 y W1; (iv) contrastar la diferencia con las pruebas que prescribe la guía y corregir por comparaciones múltiples; (v) evaluar la robustez y la generalización a otras plantas y a otros conjuntos de datos; (vi) llevar los modelos a un dashboard que solo carga artefactos ya entrenados.

## Mapa del libro

| Capítulo | Contenido |
|---|---|
| [1. Trazabilidad de correcciones](01_correcciones.md) | Autoauditoría frente a la guía y auditoría de fuga |
| [2. Datos: ETL y EDA](02_datos.md) | Fuentes, cohorte, variable objetivo, calidad, relaciones, tabla $D{+}1$, auditoría de fuga y declaración de los datos |
| [3. Modelos benchmark](03_benchmarks.md) | Modelos de referencia, rejillas, presupuesto, resultados de las dos tareas y mejores combinaciones por optimizador |
| [4. Revisión de literatura y novedad](04_literatura.md) | Matriz de búsqueda en Web of Science, comparación con trabajos afines y declaración de novedad |
| [5. Modelo original](05_modelo_original.md) | K1 y W1: motivación, formulación, pseudocódigo, propiedades, complejidad, código, pruebas y ablación |
| [6. Protocolo y validación estadística](06_validacion.md) | Protocolo, equidad, supuestos, DM-HLN, Giacomini–White, prueba bayesiana, MCS y efectos |
| [7. Evaluación ampliada y robustez](07_evaluacion_ampliada.md) | Plantas no vistas y nuevas, conjuntos adicionales, sensibilidad, semillas, ruido, escalabilidad, incertidumbre y errores |
| [8. Dashboard y despliegue](08_dashboard.md) | Arquitectura, capturas, decisiones de diseño, ficha del modelo y guía de despliegue |
| [9. Conclusiones y limitaciones](09_conclusiones.md) | Hallazgos, veredicto R1–R3, limitaciones, trabajo futuro y declaración de uso de IA |
| [10. Reproducibilidad y bibliografía](10_reproducibilidad.md) | Entorno, semillas, tabla maestra, lista de verificación y referencias |

Las tablas de los capítulos 2 a 10 se generan con celdas de Python que leen los archivos de `resultados/`, `data/` y `artifacts/` del repositorio. Los cálculos costosos (búsquedas de hiperparámetros, reentrenamientos, bootstrap) no se repiten al construir el libro: cada tabla indica el cuaderno o el script que generó su archivo de origen.

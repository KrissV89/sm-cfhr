---
title: Conclusiones y limitaciones
short_title: 9. Conclusiones
---

# Conclusiones y limitaciones

La pregunta del proyecto era si una corrección basada en el nivel reciente de generación reduce el error de los modelos de referencia al pronosticar el FC del día siguiente, y con qué certeza. Todos los errores citados son MAE macro por planta en unidades de FC.

## Hallazgos

1. **Sin historial, la meteorología no basta.** Con clima del mismo día y grupos de plantas no vistos (Tarea 1), el mejor resultado lo da Ridge con la irradiación (0.0429); más variables, modelos no lineales, ensambles y adaptaciones propias no lo mejoran ([capítulo 3](03_benchmarks.md)). El error restante es el nivel propio de cada planta.
2. **Con historial, la corrección por anomalías reduce el error, poco.** En la prueba de agosto a octubre de 2026, K1 baja el MAE de Random Forest de 0.0406 a 0.0392 (3.3 %) y W1 a 0.0388 (4.3 %). K1 mejora en 14 de 17 plantas y en los 7 grupos. La ablación atribuye casi toda la mejora a restar al pronóstico de radiación su media móvil de 30 días y aplicar esa anomalía al nivel reciente ([capítulo 5](05_modelo_original.md)); SHAP muestra que todos los modelos, incluidos los de librería, se apoyan en el nivel y la radiación.
3. **La certeza es limitada.** La regla de mejora prefijada se cumple en cinco de seis condiciones: la diferencia no resiste Holm (Wilcoxon por planta 0.017 → 0.70). La comparación de K1 con Random Forest no es significativa tras la corrección de Holm (DM-HLN $p=0.076$, Giacomini–White $p=0.233$; IC 95 % de la diferencia $[-0.0028,+0.0002]$), y el MCS al 90 % conserva a Random Forest, SVR y XGBoost junto a K1 y W1 ([capítulo 6](06_validacion.md)).
4. **El orden es estable dentro del problema.** Se mantiene con plantas excluidas por grupo, con plantas nuevas (aunque la mayor parte de esa mejora viene de cuatro plantas atípicas), con diez semillas, con ruido en el pronóstico y con cuatro optimizadores de hiperparámetros, que cambian el MAE en menos de 0.001 ([capítulos 3](03_benchmarks.md) y [7](07_evaluacion_ampliada.md)).
5. **No se generaliza fuera del problema.** En tres conjuntos públicos de otra naturaleza, K1 y W1 no superan al mejor modelo de librería en ninguno y son significativamente peores en dos ([capítulo 7](07_evaluacion_ampliada.md)). El mecanismo depende de un nivel estable y de una respuesta lineal a anomalías exógenas.
6. **La incertidumbre debe recalibrarse.** Los intervalos conformes calibrados en desarrollo pierden cobertura en la prueba; una ventana móvil de 90 días la recupera con intervalos más anchos.

**Magnitud práctica.** Una reducción del MAE es menos desvío entre la energía pronosticada y la entregada, no generación adicional. Convertida a MWh con la capacidad de cada planta y día ([capítulo 7](07_evaluacion_ampliada.md)), la suma diaria de errores absolutos de las 17 plantas baja de 1,151 a 1,085 MWh/día (−5.7 %) con K1, sobre 7,296 MWh/día producidos. La reducción se concentra en Guayepo y Portón del Sol (67 %), la mediana por planta es de −0.95 MWh/día y tres plantas empeoran. No se anualiza ni se estima valor económico: hay 92 días y no existe un modelo de costos.

## Veredicto frente a la guía del curso

| Condición | Veredicto | Evidencia |
|---|---|---|
| R1: novedad en la aplicación | **No acreditada** | La revisión no encontró el uso de anomalías respecto de la media móvil del propio pronóstico, pero K1 es posprocesamiento estadístico y W1 una regresión localmente ponderada, técnicas establecidas; una búsqueda sin resultado no demuestra novedad y varias fuentes se leyeron solo a nivel de resumen ([capítulo 4](04_literatura.md)) |
| R2: implementación propia | **Defendible para W1; no para K1** | W1 está programado con NumPy y SciPy como clase compatible con scikit-learn, con formulación, solución, demostraciones y 19 pruebas unitarias; K1 delega el ajuste en `Ridge` de scikit-learn ([capítulo 5](05_modelo_original.md)). La suficiencia queda para la defensa del equipo |
| R3: superioridad significativa | **No se cumple** | La comparación prefijada no es significativa con Holm; el MCS conserva a las mejores referencias; los rechazos de W1 son *post hoc* y con información de identidad adicional; en los conjuntos adicionales no es mejor en ninguno ([capítulos 6](06_validacion.md) y [7](07_evaluacion_ampliada.md)) |

## Limitaciones

1. Una sola ventana de prueba de tres meses; septiembre ya invierte el signo de K1 frente a Random Forest. Ninguna prueba sobre ella estima la variación entre periodos.
2. La ventana de prueba se usó primero en el cuaderno 07; W1 (19), las pruebas de la guía y los estudios 21–23 son posteriores y exploratorios.
3. Las plantas comparten clima dentro de cada grupo y entre grupos: 17 plantas equivalen a menos unidades independientes.
4. Los pronósticos son los archivados de la Previous Runs API. No se verificó que estuvieran disponibles a la misma hora en operación, ni que el FC del día $D$ esté publicado al cierre de $D$.
5. La etiqueta mezcla recurso, disponibilidad y registro: el FC es base AC neta, y los días de FC bajo, que concentran el error, incluyen fallas y restricciones que ningún modelo meteorológico pronostica.
6. W1 usa la identidad de planta y de grupo, que las referencias no reciben.
7. Las rejillas originales eran pequeñas; con dominios ampliados el orden no cambió, pero el ancho de banda de W1 quedó cerca del borde de su dominio.
8. Los intervalos calibrados en desarrollo no alcanzan la cobertura nominal en la prueba.
9. La revisión bibliográfica es parcial: se hizo sobre exportaciones de Web of Science, sin búsqueda sistemática en otras bases ([capítulo 4](04_literatura.md)).

## Trabajo futuro

- Evaluar K1 y W1 en un periodo nuevo (desde noviembre de 2026) con el protocolo congelado: es la única forma de convertir la evidencia en confirmatoria.
- Dar a las referencias la misma información de identidad que W1 (codificación de planta y grupo, o modelos por planta) para aislar el aporte del algoritmo.
- Incorporar la potencia DC, el tipo de montaje y la disponibilidad reportada por XM, y separar los días de indisponibilidad.
- Verificar los horarios reales de publicación de XM y de los pronósticos antes de cualquier uso operativo: el pronóstico D+1 aquí es retrospectivo, con pronósticos archivados, y las decisiones del mercado para el día siguiente se toman durante $D$.
- Evaluar el error horario: el despacho y los desvíos son horarios y un error diario pequeño puede ocultar errores horarios que se compensan.
- Añadir el valor del día anterior a K1 y W1 en series con persistencia de corto plazo, como sugiere la lectura exploratoria de los conjuntos adicionales.

## Uso de herramientas de inteligencia artificial

La declaración de autoría y de uso de herramientas, con su detalle y la firma del equipo, está en el [último capítulo](10_reproducibilidad.md).

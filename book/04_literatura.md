---
title: Revisión de literatura y novedad
short_title: 4. Literatura
kernelspec:
  name: python3
  display_name: Python 3
---

# Revisión de literatura y declaración de novedad

La revisión se hizo sobre exportaciones de Web of Science (WoS) cribadas por consulta, en el cuaderno `notebooks/legacy/actuales/09_revision_bibliografica_custom.ipynb`. Las consultas están en `docs/investigacion/QUERIES_BUSQUEDA.md` (Q1 a Q11) y `docs/investigacion/QUERIES_MODELOS_CUSTOM.md` (Q12 a Q24); el catálogo de candidatos con sus fuentes, en `docs/investigacion/CANDIDATOS_ADICIONALES.md`; y la lectura del trabajo colombiano más cercano, en `docs/investigacion/REVISION_PICO_2026.md`.

**Alcance y límites.** Es una búsqueda dirigida, no una revisión sistemática. La clasificación de los registros se hace con título, resumen y palabras clave; solo unos pocos artículos se leyeron en texto completo. Los exports son los registros que WoS devolvió para cada consulta, con su tope de exportación. Una búsqueda sin resultado no demuestra que una técnica no exista.

## Matriz de búsqueda

La [Tabla %s](#tbl-matriz) resume las consultas: qué pregunta responde cada bloque, cuántos registros devolvió y qué se concluyó. El texto exacto de cada consulta y su archivo de exportación están en los dos documentos de consultas.

:::{table} Matriz de búsqueda en Web of Science Core Collection. Registros según el registro de búsqueda y la lectura del cuaderno 09.
:label: tbl-matriz

| Bloque | Consultas | Pregunta | Registros | Resultado |
|---|---|---|---:|---|
| Contexto fotovoltaico | Q1 (modelos FV con variables meteorológicas), Q2 (Colombia, nivel de planta), Q3 (Latinoamérica) | Qué modelos y entradas se usan para estimar o pronosticar generación FV | 1000, 39, 894 | SVR, Random Forest y boosting son las técnicas más mencionadas; en Q2 la única coincidencia directa de predicción ML de potencia FV en varias plantas colombianas es {cite:t}`pico2026colombia` (`fuentes/bibliografia/wos/HALLAZGOS_WOS.md`) |
| Validación y calidad | Q4 (validación temporal, espacial y entre plantas), Q5 (calidad de datos FV), Q6 (calidad meteorológica), Q8 (métricas), Q9 (normalización entre plantas), Q10 (preprocesamiento), Q11 (exploratoria) | Cómo validar sin fuga, qué controles de calidad aplicar y cómo comparar plantas | 158, 50, 1000, 50, 116, 312, 372 | Q4 respalda particiones que respeten el tiempo y ajustar las transformaciones solo con el entrenamiento; Q9 distingue normalizar el error entre plantas de transformar los predictores (`HALLAZGOS_WOS.md`) |
| A. ¿Ya se usó en FV? | Q12 (respuesta fraccionaria), Q13 (monotonía, árboles de modelos, GAM), Q14 (cambio de covariables), Q15 (factor de capacidad y física) | Si los candidatos C1–C5 de la Tarea 1 ya se aplicaron a generación FV | 3, 28, 4, 40 | C1 y C2 no encontrados en FV; C3 y C4 ya usados; C5 con variantes de transferencia |
| B. Origen en otras áreas | Q16 (crédito), Q17 (hidrología y suelos) | De dónde provienen C1–C5 | 53, 1000 | Origen en riesgo crediticio (LGD, puntuación) e hidrología |
| K1 | Q18 (anomalías del pronóstico, persistencia, posprocesamiento) | Si la construcción de K1 está descrita en la literatura | sin ejecutar | Pendiente; K1 se clasificó con 11 fuentes leídas (sección 6 del cuaderno 09) |
| C. Candidatos de la ronda de K1 | Q19 (nivel adaptativo, Kalman), Q20 (MOS, EMOS, análogos), Q21 (modelos mixtos multiplanta), Q22 (combinación de pronósticos), Q23 (regresión cuantílica), Q24 (Kalman en NWP e hidrología) | Respaldo de N1, N2, R1, M1, AE1, H1, F1, Q1 y Q2 | 22, 157, 1, 4, 27, 111 | Respaldo parcial (solo resumen) para casi todos; Q21 no aportó registros pertinentes |
:::

La celda siguiente cuenta los registros de cada exportación disponible en el repositorio (cada registro de WoS termina en una línea `ER`), para que los números de la tabla se puedan comprobar.

::::{table} Registros por archivo de exportación de WoS presentes en `fuentes/bibliografia/wos/`.
:label: tbl-wos
```{code-cell} python
# Cuenta registros por exportación de Web of Science (formato de texto etiquetado: un registro termina en "ER").
import re
import apoyo
import pandas as pd

carpeta = apoyo.RAIZ / "fuentes" / "bibliografia" / "wos"
filas = []
for ruta in sorted(carpeta.glob("WoS_Q*.txt"), key=lambda p: int(re.search(r"Q(\d+)", p.name).group(1))):
    texto = ruta.read_text(encoding="utf-8-sig", errors="replace")
    filas.append({"consulta": re.search(r"(Q\d+[a-z]?)", ruta.name).group(1),
                  "archivo": ruta.name,
                  "registros": sum(1 for linea in texto.splitlines() if linea.strip() == "ER")})
pd.DataFrame(filas).set_index("consulta") if filas else apoyo.nota("No hay exportaciones de WoS en el repositorio.")
```
::::

Las diferencias entre la [Tabla %s](#tbl-matriz) y la [Tabla %s](#tbl-wos), si las hay, se deben a que el registro de búsqueda anota lo que WoS mostró y la celda cuenta lo exportado (Q11, por ejemplo, mostró 300 y exportó 372). Tras unificar por DOI o título, Q1 a Q17 suman 4 882 registros únicos (cuaderno 09, sección 1), y Q19 a Q24 suman 307, de los cuales 90 mencionan redes neuronales y no se usaron como respaldo.

## Técnicas en el corpus fotovoltaico

En los registros únicos con tema fotovoltaico de Q1 a Q16 (2 837), las técnicas más mencionadas son máquinas de vectores de soporte (228 registros), Random Forest (180) y boosting de gradiente (176); siguen stacking (58), $k$ vecinos (47) y regresiones regularizadas (44) (cuaderno 09, sección 2). Los modelos de referencia del [capítulo 3](03_benchmarks.md) son, por tanto, los que la literatura usa para esta familia de problemas. Un conteo en título y resumen indica que la técnica aparece, no que se use para predecir generación.

## Trabajos cercanos y métodos afines

La [Tabla %s](#tbl-afines) compara el problema de este proyecto con los trabajos más cercanos citados en el manuscrito y con las familias de métodos de las que provienen K1 y W1.

:::{table} Comparación con trabajos cercanos y métodos afines. «Leído» indica el nivel de lectura que respalda la fila.
:label: tbl-afines

| Trabajo o familia | Tarea y datos | Técnica | Relación con este proyecto | Leído |
|---|---|---|---|---|
| {cite:t}`pico2026colombia` | Siete plantas colombianas, potencia simulada con SAM a 10 min, día siguiente | Regresión lineal, red feedforward, LSTM | Mismo país y horizonte; etiqueta simulada, resolución intradía y redes neuronales. No es referencia numérica | Texto completo (`REVISION_PICO_2026.md`) |
| {cite:t}`trigo2019hourly,trigo2021models` | Producción horaria en Chile y España | Regresión lineal múltiple exportable; SVM y redes | Estimación entre plantas (análoga a la Tarea 1) con otra frecuencia | Manuscrito |
| {cite:t}`lehmann2023benchmark` | Siete plantas, intradía | Ocho soluciones comerciales, MAE, RMSE y sesgo | Métricas y comparación de pronósticos | Manuscrito |
| {cite:t}`mueller2016yield`, {cite:t}`purohit2018performance` | Rendimiento observado frente a estimado | Modelos de rendimiento | Justifican usar generación observada como etiqueta | Manuscrito |
| Model output statistics {cite:p}`glahn1972mos` y posprocesamiento solar {cite:p}`yang2021postprocessing` | Corrección estadística de la salida de un modelo numérico con observaciones locales | Regresión de la observación sobre el pronóstico | K1 pertenece a esta familia: regresión lineal sobre el pronóstico, entrenada con pares pasados | Texto parcial (Mayer et al.; cuaderno 09, sección 6) |
| Persistencia combinada con pronóstico exógeno {cite:p}`bacher2009online` | Pronóstico solar de corto plazo | Modelos adaptativos con persistencia y pronóstico meteorológico | K1 parte del nivel reciente (persistencia suavizada) | Manuscrito |
| Regresión localmente ponderada {cite:p}`cleveland1979robust,atkeson1997locally` | General | Ajuste lineal local con pesos por distancia | W1 es esta técnica aplicada al residuo de nivel, con un factor de peso por planta y grupo | Texto completo (Atkeson et al.) |
| Modelos de efectos mixtos {cite:p}`bates2015lme4` | General | Agrupamiento parcial por unidad | El factor $\rho$ de W1 cumple un papel análogo; H1 lo implementa de forma explícita | Texto completo |
| Bai et al. (2026), en el cuaderno 09 | Pronóstico FV con variables de desviación | Desviaciones respecto del promedio simultáneo entre sitios | Única fuente con «anomalías» del pronóstico; respecto de otros sitios, no de la media reciente de la propia serie | Solo resumen |
:::

## Clasificación de K1 y W1

**K1.** El cuaderno 09 (sección 6) separa la construcción de K1 en cinco componentes y busca cada uno en once fuentes, cuatro de ellas leídas en texto completo o en parte:

1. Nivel reciente como punto de partida (persistencia, media móvil): técnica establecida.
2. Regresión de la observación sobre el pronóstico con pares pasados (MOS, EMOS): técnica establecida.
3. Corrección con el historial reciente medido o con errores recientes: técnica establecida.
4. Anomalía del pronóstico respecto de su propia media móvil como predictora: no encontrada en la búsqueda.
5. Objetivo normalizado por capacidad a escala diaria: técnica establecida.

La clasificación del cuaderno es «variante de técnica establecida». Una lectura posible es que K1 equivale a un MOS lineal formulado en anomalías sobre una línea base de persistencia; el cuaderno la presenta como interpretación propia, no respaldada por una fuente leída. La consulta Q18, diseñada para buscar precisamente esa construcción, no se ejecutó.

**W1.** W1 no tuvo consulta propia (cuaderno 09, sección 13: «candidatos sin consulta propia: B1, W1 y G1»). Su fuente es {cite:t}`atkeson1997locally`, leída en texto completo; la adaptación (residuo de nivel, espacio de anomalías, factor de peso por planta y grupo, penalización escalada por la suma de pesos) es del proyecto. No se buscó si la regresión local con pesos por identidad de sitio ya se ha usado en pronóstico fotovoltaico, de modo que tampoco hay evidencia de búsqueda a favor de su novedad.

## Declaración de novedad

:::{important} Declaración
K1 es posprocesamiento estadístico del pronóstico meteorológico sobre el nivel reciente de cada planta, en la familia de MOS y de la corrección con historial reciente. W1 es una regresión localmente ponderada con un factor de peso por planta y grupo. Ambas son adaptaciones de técnicas establecidas. La búsqueda no encontró el uso de anomalías respecto de la media móvil del propio pronóstico como entrada de la corrección, pero la búsqueda no demuestra novedad: cubre exportaciones de WoS leídas por título y resumen, varias fuentes del área solo se conocen por su resumen o devolvieron acceso denegado, la consulta específica de K1 (Q18) no se ejecutó y W1 no tuvo consulta propia. El proyecto no afirma que K1 o W1 sean métodos nuevos ni que no hayan sido usados para este problema; la condición R1 de la guía no se acredita ([capítulo 9](09_conclusiones.md)).
:::

Lo que sí documenta la revisión es de dónde viene cada componente y qué cambia respecto de la fuente, que es la información que la guía pide para la tabla comparativa ([Tabla %s](#tbl-afines)) y que sostiene la defensa del diseño en el [capítulo 5](05_modelo_original.md).

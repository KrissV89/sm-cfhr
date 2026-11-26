---
title: "Datos: ETL y EDA"
short_title: 2. Datos
kernelspec:
  name: python3
  display_name: Python 3
---

# Datos: ETL, análisis exploratorio y declaración de los datos

El estudio es observacional y su unidad de análisis es el día-planta. Las cifras de este capítulo provienen de los cuadernos 01 (descarga y consolidación), 02 (EDA y calidad), 06 (descarga del pronóstico) y 12 (ampliación de la cohorte), y del manuscrito (`paper/manuscrito.tex`, sección de datos). Las figuras las genera `scripts/generar_figuras_paper.py`; las celdas de este capítulo recalculan los controles principales desde los archivos versionados de `base_generacion_clima/` y `datos_pronostico/`.

## Fuentes y flujo ETL

La base enlaza tres fuentes públicas:

1. **XM**, operador del mercado eléctrico colombiano: energía horaria (métrica de generación real) y capacidad efectiva neta diaria por recurso {cite:p}`xm2026api`.
2. **Open-Meteo**: meteorología modelada por coordenadas en la zona horaria de Colombia. Se conservan por separado el producto `best_match` y el reanálisis ERA5 {cite:p}`openmeteoHistorical`, y, para la Tarea 2, los pronósticos archivados de la Previous Runs API {cite:p}`openmeteoPreviousRuns`.
3. **Global Solar Power Tracker** de Global Energy Monitor para las coordenadas de las plantas {cite:p}`gem2026solar`, verificadas con prensa o con el desarrollador.

La ventana va del 1 de enero de 2024 al 31 de octubre de 2026 y la instantánea se descargó el 15 de noviembre de 2026 (`base_generacion_clima/procedencia.json`). La fecha de corte es una constante del cuaderno de descarga, de modo que volver a ejecutarlo reproduce la misma base.

El flujo es: (i) **extracción** de las series de XM y de Open-Meteo con peticiones paginadas (cuadernos 01 y 06); (ii) **transformación**: suma diaria de la energía horaria, unión con la capacidad del mismo día, cálculo del FC con la [ecuación %s](#eq-fc), resúmenes meteorológicos diarios solo cuando el producto trae las 24 horas, y promedio de los tres modelos de pronóstico; (iii) **carga** en tablas planta-día (`base_generacion_clima/base_planta_dia.csv`, `datos_pronostico/pronostico_diario.csv`) con sus banderas de calidad, y construcción de la tabla de trabajo $D{+}1$ con `src/features.py`.

## Construcción de la cohorte

Una fila recurso-día entra en la base si cumple, en este orden: capacidad efectiva neta registrada ese día; energía y capacidad positivas; $\mathrm{FC}\le 1.05$ (un FC mayor que 1 es imposible con este denominador y se admite una tolerancia de redondeo); recurso enlazado a coordenadas; resumen meteorológico completo; al menos 60 días desde el primer día con capacidad registrada, cuando esa fecha cae dentro de la ventana; y al menos 100 días útiles en la planta. Ninguna regla usa el valor del FC ni la irradiación, salvo la cota física.

:::{figure} figuras/eda_flujo_cohorte.png
:label: fig-cohorte
:width: 100%
Filas recurso-día que cumplen todas las condiciones de inclusión hasta cada etapa. Datos de XM y Open-Meteo con corte al 31 de octubre de 2026.
:::

Dos condiciones explican la reducción de 30 964 a 12 756 filas ([Figura %s](#fig-cohorte)). En 11 279 filas XM reporta generación y no registra capacidad, por lo que el FC no se puede calcular. Otras 6 034 corresponden a los 33 recursos sin coordenadas verificadas. La exclusión del periodo inicial retira 895 filas y las condiciones restantes no retiran ninguna. La cohorte es el conjunto de plantas que se pudo ubicar; no es una muestra aleatoria ni un censo. Las 17 plantas se reparten en siete grupos de tamaño desigual ([Figura %s](#fig-mapa)): uno reúne seis plantas y el 30.9 % de la base, y otro tiene una sola planta y el 2.0 %.

:::{figure} figuras/eda_mapa_plantas.png
:label: fig-mapa
:width: 80%
Ubicación de las 17 plantas (área del marcador proporcional a la capacidad neta AC, en MW) y grupos espaciales (plantas a menos de 100 km, por enlace simple). Las ciudades son solo de referencia.
:::

En 15 de las 17 plantas XM reporta generación antes de registrar capacidad: 2 466 días, entre 36 y 356 por planta. La base no permite saber si corresponden a pruebas previas a la operación comercial o a un rezago del registro; en consecuencia no se observan los primeros meses de producción de la mayoría de las plantas, y la proporción de días retenidos va de 59.8 % a 100 % ([Figura %s](#fig-cobertura)).

:::{figure} figuras/eda_cobertura_temporal.png
:label: fig-cobertura
:width: 100%
Días de cada planta en la base, días excluidos por el periodo inicial de 60 días o por generación sin capacidad registrada, huecos de XM y cambios de capacidad registrada; a la derecha, días en la base.
:::

## Controles de calidad

La celda siguiente recalcula los controles de la [Tabla %s](#tbl-eda) a partir de la base versionada. Coinciden con los del cuaderno 02 y con el manuscrito.

::::{table} Controles del análisis exploratorio sobre la base de días-planta (recalculados al construir el libro).
:label: tbl-eda
```{code-cell} python
# Controles de calidad de la base planta-día (cuaderno 02). Lee los CSV versionados de base_generacion_clima/.
import numpy as np
import pandas as pd
import apoyo

base = apoyo.leer("base_generacion_clima/base_planta_dia.csv", parse_dates=["date"])
banderas = apoyo.leer("base_generacion_clima/banderas_calidad.csv")
proc = apoyo.leer_json("base_generacion_clima/procedencia.json")

meteo = ["rad_bm", "temp_bm", "nube_bm", "humedad_bm", "viento_bm"]
rend = base["fc"] * 24 / base["gti10"]                 # rendimiento específico: horas equivalentes por kWh/m²
cambios = (base.sort_values("date").groupby("code")["cap_MW"]
           .apply(lambda s: int((s.diff().fillna(0) != 0).sum())))   # escalones de capacidad registrada
q1, q3 = base["fc"].quantile([0.25, 0.75])

controles = {
    "Días-planta; plantas; grupos espaciales": f"{len(base):,}; {base.code.nunique()}; {base.grupo.nunique()}",
    "Filas de generación de XM en la ventana (antes de filtrar)": f"{proc['filas_xm_generacion']:,}",
    "Llaves planta-fecha duplicadas": int(base.duplicated(["code", "date"]).sum()),
    "Valores faltantes en etiqueta y predictores": int(base[["fc"] + meteo].isna().sum().sum()),
    "FC: media; mediana; rango intercuartílico": f"{base.fc.mean():.3f}; {base.fc.median():.3f}; {q1:.3f}–{q3:.3f}",
    "FC máximo": f"{base.fc.max():.3f}",
    "Días-planta con 12 a 14 horas con dato en XM": f"{base.horas_xm.between(12, 14).sum():,} ({base.horas_xm.between(12, 14).mean():.1%})",
    "Días-planta con 6 horas o menos con dato": f"{(base.horas_xm <= 6).sum()} ({(base.horas_xm <= 6).mean():.2%})",
    "Días-planta con rendimiento específico < 0.3": int((rend < 0.3).sum()),
    "Mediana del rendimiento específico (h por kWh/m²)": f"{rend.median():.2f}",
    "Días-planta marcados como anómalos": f"{int(banderas.dia_anomalo.sum())} ({banderas.dia_anomalo.mean():.2%})",
    "Cambios de capacidad registrada (plantas con cambios)": f"{cambios.sum()} ({(cambios > 0).sum()})",
}
pd.DataFrame({"resultado": controles}).rename_axis("control")
```
::::

## Variable objetivo

El FC de la [ecuación %s](#eq-fc) tiene cuatro propiedades que delimitan lo que mide. Primero, el denominador es capacidad en corriente alterna registrada ante el operador y no potencia pico de los módulos: dos plantas con el mismo recurso y distinta relación DC/AC tienen FC distintos. Segundo, se usa la capacidad del mismo día porque el registro cambia por escalones: siete veces en seis plantas, siempre al alza, con cambios de hasta 90 a 100 MW (Fundación) y 83 a 100 MW (Tepuy). Tercero, XM no reporta ceros: una hora sin generación queda vacía y un día sin generación no aparece. Cuarto, el FC incorpora el recurso solar junto con diseño, disponibilidad, mantenimiento y restricciones de red; se trata como rendimiento realizado y no como propiedad del emplazamiento.

:::{figure} figuras/eda_distribucion_fc.png
:label: fig-fc
:width: 100%
(a) Distribución del FC diario (adimensional) en los 12 756 días-planta, con mediana y rango intercuartílico. (b) FC diario por planta, ordenado por mediana (caja: P25–P75; bigotes: P5–P95), con el número de días.
:::

Todos los valores son positivos y menores que 0.44, con media 0.248 y mediana 0.250 ([Figura %s](#fig-fc), [Tabla %s](#tbl-eda)). La distribución es unimodal con una cola inferior asociada al número de horas reportadas: el FC mediano es 0.253 en los días con 12 a 14 horas con dato y menor que 0.04 en los 31 días-planta con 6 horas o menos ([Figura %s](#fig-horas)). El FC mediano por planta va de 0.193 a 0.307, con 253 a 1 034 días por planta: el nivel difiere entre plantas más de lo que varía el error de los mejores modelos (MAE cercano a 0.04).

:::{figure} figuras/eda_horas_y_anomalos.png
:label: fig-horas
:width: 100%
(a) Días-planta según las horas con generación reportada (escala logarítmica). (b) FC frente a horas reportadas; las cruces son los 70 días marcados como anómalos y la línea es la mediana por hora.
:::

## Valores atípicos y banderas de calidad

La plausibilidad de cada día se revisa con el rendimiento específico, $\mathrm{FC}_{p,t}\times 24/G_{p,t}$, donde $G_{p,t}$ es la irradiación diaria en un plano fijo inclinado 10° (kWh/m²/día): las horas equivalentes a capacidad neta por cada kWh/m² recibido. Su mediana es 1.08 y difiere entre plantas, de 0.86 a 1.28 ([Figura %s](#fig-rendimiento)); valores mayores que 1 son compatibles con un denominador en capacidad AC y con plantas con seguidores. Se marcan como anómalos los días-planta con 6 horas o menos con dato o con rendimiento específico menor que 0.3: son 70 (0.55 %). En ellos el clima no explica el FC. Los dos umbrales son decisiones de este análisis y la bandera depende de la etiqueta; por eso los días marcados se conservan, se usan en un análisis de sensibilidad declarado y no se eliminan del conjunto de evaluación {cite:p}`jensen2024solarqc`. Hay además 446 días-planta con rendimiento específico mayor que 1.5, concentrados en Alma II (17.8 % de sus días) y en las dos fases de Bosques Solares de los Llanos (8.3 %).

:::{figure} figuras/eda_rendimiento_especifico.png
:label: fig-rendimiento
:width: 100%
(a) FC diario frente a la irradiación en plano fijo a 10° (kWh/m²/día), con las rectas de rendimiento específico 0.3 y 1.5 y los 68 días por debajo de 0.3. (b) Rendimiento específico por planta.
:::

## Predictores meteorológicos

Los resúmenes meteorológicos están completos y no hay irradiaciones diarias iguales a cero. La irradiación horizontal de `best_match` va de 0.85 a 7.46 kWh/m²/día, con mediana 5.65; la temperatura media, de 19.2 a 36.0 °C; y el viento medio tiene mediana de 1.9 m/s ([Figura %s](#fig-clima)). Son salidas de modelos por celda y no mediciones en la planta. La celda de `best_match` queda entre 2.0 y 5.7 km de cada planta; dos pares de plantas comparten celda y tienen predictores idénticos todos los días, de modo que la cohorte tiene 15 sitios meteorológicos distintos.

:::{figure} figuras/eda_clima.png
:label: fig-clima
:width: 100%
Distribución de los resúmenes meteorológicos diarios de `best_match` en los días-planta de la base (irradiación en kWh/m²/día, temperatura en °C, nubosidad y humedad en %, viento en m/s) y de la irradiación en plano fijo a 10° (gris), con su mediana.
:::

Los dos productos de irradiación difieren: `best_match` supera a ERA5 en 0.18 kWh/m²/día en promedio, con desviación estándar de la diferencia de 0.68 y correlación de 0.80 ([Figura %s](#fig-radiacion)). Esa diferencia es una medida de la incertidumbre del predictor principal.

:::{figure} figuras/eda_radiacion_productos.png
:label: fig-radiacion
:width: 100%
(a) Irradiación horizontal diaria de `best_match` frente a ERA5 (kWh/m²/día). (b) Diferencia `best_match` − ERA5 (media +0.18, desviación estándar 0.68 kWh/m²/día).
:::

## Relaciones entre variables

La irradiación es el predictor más asociado con el FC (correlación de Pearson 0.63 con `best_match` y 0.56 con ERA5), seguida de la nubosidad ([Figura %s](#fig-correlaciones)). La asociación cambia según el nivel en que se mida, y este es el hallazgo que orienta el modelado. Dentro de cada planta, el FC diario sigue a la irradiación diaria (Spearman entre 0.58 y 0.76). Entre plantas, el FC mediano no sigue a la irradiación mediana modelada (Spearman $-0.08$, 17 plantas). En las cuatro plantas con potencia DC publicada, la relación DC/AC va de 1.17 a 1.40 y el FC mediano crece con ella (0.193, 0.244, 0.263 y 0.266). Un modelo con solo variables meteorológicas puede reproducir la variación diaria dentro de una planta y carece de información para predecir el nivel de una planta nueva; esta lectura motiva la Tarea 2 y el diseño de K1 ([capítulo 5](05_modelo_original.md)).

:::{figure} figuras/eda_correlaciones.png
:label: fig-correlaciones
:width: 100%
(a) Correlación de Spearman entre el FC y los predictores meteorológicos. (b) Ajuste lineal FC–irradiación (kWh/m²/día) dentro de cada planta (P10–P90) frente al ajuste entre medianas de planta.
:::

## Estructura temporal y dependencia

El FC tiene un ciclo anual moderado y cambios de nivel entre meses que difieren entre grupos ([Figura %s](#fig-estacionalidad)). La autocorrelación del FC a un día va de 0.14 a 0.63 entre plantas. El FC del mismo día está correlacionado entre plantas: la mediana es 0.76 en los 20 pares del mismo grupo y 0.29 en los 116 pares de grupos distintos, con un máximo de 0.66 ([Figura %s](#fig-dependencia)). Esto tiene dos consecuencias para el diseño: las particiones deben mantener juntas las plantas cercanas y respetar el orden temporal, y las 17 plantas no pueden tratarse como 17 unidades independientes en las pruebas estadísticas ([capítulo 6](06_validacion.md)).

:::{figure} figuras/eda_estacionalidad.png
:label: fig-estacionalidad
:width: 100%
(a) FC medio mensual: mediana y rango intercuartílico entre plantas (marcador hueco: meses con menos de 17 plantas). (b) FC medio mensual por planta, agrupado por grupo espacial (meses con al menos 20 días).
:::

:::{figure} figuras/eda_dependencia.png
:label: fig-dependencia
:width: 100%
(a) Autocorrelación del FC diario por planta y su mediana, por rezago en días. (b) Correlación del FC del mismo día entre pares de plantas frente a la distancia (km), según pertenezcan o no al mismo grupo; la línea punteada marca 100 km.
:::

## Datos para el pronóstico del día siguiente

Para la Tarea 2 se descargaron los pronósticos archivados de la Previous Runs API: el valor que cada corrida pronosticó 24 horas antes de la hora de validez, para los modelos ECMWF IFS 0.25°, GFS e ICON, promediados. Un día-planta se descarta si alguno de los tres modelos no trae las 24 horas de alguna variable; quedan 16 286 de 17 595 días-planta (92.6 %, `datos_pronostico/procedencia.json`). Se descartó la Historical Forecast API porque empalma las primeras horas de cada corrida y casi coincide con el clima realizado (correlación 0.976 con la irradiación del reanálisis, frente a 0.615 del pronóstico a un día en la comprobación previa), lo que habría introducido información del día objetivo.

La tabla de trabajo conserva las filas con FC observado, historial suficiente y pronóstico completo: 12 108 de 12 756 (94.9 %), con fechas objetivo del 21 de marzo de 2024 al 31 de octubre de 2026. Cada fila es una planta y una fecha objetivo $T$. Las variables de historial son el FC de $D$ y las medias móviles

$$
\bar y^{(w)}_{p,D}=\frac{1}{|\mathcal{O}_w(D)|}\sum_{t\in\mathcal{O}_w(D)}y_{p,t},\qquad w\in\{7,30\},\qquad L_{p,D}=\bar y^{(30)}_{p,D},
$$ (eq-nivel)

con $\mathcal{O}_w(D)$ los días con FC observado entre los $w$ que terminan en $D$ (al menos 4 en la ventana de 7 días y 15 en la de 30). Las cinco variables de pronóstico son la irradiación (kWh/m²/día), la temperatura (°C), la nubosidad y la humedad relativa (%) y el viento (m/s) pronosticados para $T$. En la tabla, la irradiación pronosticada tiene una correlación de 0.76 con la del reanálisis del mismo día y de 0.57 con el FC; el nivel $L_{p,D}$, de 0.43 con el FC; y la anomalía de irradiación, de 0.60 con el residuo de nivel $y_{p,T}-L_{p,D}$ ([Figura %s](#fig-pronostico)). El supuesto de que el FC de $D$ está disponible al cierre de $D$ debe verificarse con los horarios de publicación de XM antes de un uso operativo.

:::{figure} figuras/eda_pronostico.png
:label: fig-pronostico
:width: 100%
Datos de la tarea $D{+}1$: (a) irradiación pronosticada el día anterior frente al reanálisis del mismo día (kWh/m²/día); (b) FC frente al nivel de los 30 días previos; (c) FC frente a la irradiación pronosticada; (d) residuo de nivel frente a la anomalía de la irradiación pronosticada. Densidad de días-planta por hexágono.
:::

## Cohorte ampliada

Para evaluar plantas que no intervinieron en el desarrollo se amplió la cohorte con el mismo procedimiento (cuaderno 12). De los 36 recursos solares de XM fuera de la cohorte, 16 cumplen las reglas de inclusión y 15 se ubicaron con fuente: en total 32 plantas en 9 grupos ([Figura %s](#fig-ampliada)). Cuatro plantas nuevas tienen un FC atípico, sin meseta diurna y con mayor inyección en fin de semana en dos de ellas, lo que es coherente con autogeneradores que inyectan excedentes; concentran el 95 % de los días anómalos de las plantas nuevas (cuaderno 18). Cinco plantas nuevas comparten coordenadas o celda meteorológica con plantas originales. Las banderas se conservan y ninguna fila se elimina según el error de los modelos.

:::{figure} figuras/eda_cohorte_ampliada.png
:label: fig-ampliada
:width: 100%
Mediana y rango intercuartílico del FC diario de las 32 plantas de la cohorte ampliada (grupo entre paréntesis); los marcadores huecos son las cuatro plantas de FC atípico.
:::

## Preprocesamiento

La etiqueta y los predictores no tienen valores faltantes y no se imputa ningún valor: un día sin resumen meteorológico o sin pronóstico completo sale de la tabla. No se transforma la etiqueta. Los modelos sensibles a la escala reciben los predictores estandarizados, con media y desviación estimadas solo con el entrenamiento de cada partición. No hay variables categóricas entre los predictores de las referencias; W1 recibe además la identidad de planta y de grupo como factores de peso, no como predictores lineales ([capítulo 5](05_modelo_original.md)).

(sec-fuga-datos)=
## Auditoría de fuga: comprobaciones ejecutables

La [Tabla %s](#tbl-fuga) del capítulo 1 enumera los controles. Las celdas siguientes ejecutan tres de ellos sobre los datos reales: (i) reconstruir la tabla $D{+}1$ desde la base y el pronóstico versionados reproduce la tabla que usan los modelos y el dashboard; (ii) sumar una constante al FC de los días objetivo no cambia ninguna variable de sus filas; (iii) la media móvil de 30 días coincide con un recálculo manual. La comprobación con series permutadas (Ridge no supera a un predictor constante por planta cuando la serie de FC se permuta en el tiempo, y añadir el FC verdadero de $T$ lleva el error a casi cero) está en el cuaderno 07 y no se repite aquí.

```{code-cell} python
# (i) Reconstrucción de la tabla D+1 con el mismo código que usan los cuadernos 07, 17 y 20 (src/features.py).
from src.features import construir_tabla, HIST, PRON, ANOM

pron = apoyo.leer("datos_pronostico/pronostico_diario.csv", parse_dates=["date"])
tabla = construir_tabla(base, pron)
guardada = apoyo.leer("data/tabla_d1.csv.gz", parse_dates=["date"])          # extracto que lee el dashboard
cmp = tabla.merge(guardada, on=["code", "date"], suffixes=("", "_g"), how="outer", indicator=True)
dif = max(float(np.abs(cmp[c] - cmp[c + "_g"]).max()) for c in ["fc"] + HIST + PRON + ANOM)
print(f"Filas reconstruidas: {len(tabla):,}; filas guardadas: {len(guardada):,}; "
      f"sin pareja: {(cmp['_merge'] != 'both').sum()}")
print(f"Diferencia absoluta máxima en las 14 columnas del modelo: {dif:.1e} (redondeo de los CSV)")
```

```{code-cell} python
# (ii) Perturbación de la etiqueta: en cada repetición se elige al azar un día objetivo por planta (semilla 2026),
#      se suma 1 a su FC y se reconstruye la tabla. Si alguna variable de esa fila dependiera del FC del día
#      objetivo, cambiaría. Se perturba un solo día por planta para que el cambio no entre en la historia de
#      otro día perturbado de la misma planta (eso sí debe cambiar las filas posteriores).
rng = np.random.default_rng(2026)
cols = HIST + ["fc_D_r", "media7_r", "media30_r"] + PRON + ANOM
cambio_variables, cambio_etiqueta, filas = 0.0, [], 0
for _ in range(10):
    elegidos = tabla.groupby("code").sample(1, random_state=int(rng.integers(1_000_000)))[["code", "date"]]
    base_mod = base.merge(elegidos.assign(perturbar=True), on=["code", "date"], how="left")
    base_mod.loc[base_mod.perturbar.fillna(False).astype(bool), "fc"] += 1.0
    tabla_mod = construir_tabla(base_mod.drop(columns="perturbar"), pron)
    a = tabla.merge(elegidos, on=["code", "date"]).set_index(["code", "date"]).sort_index()
    b = tabla_mod.merge(elegidos, on=["code", "date"]).set_index(["code", "date"]).sort_index()
    cambio_variables = max(cambio_variables, float(np.abs(a[cols] - b[cols]).to_numpy().max()))
    cambio_etiqueta.extend((b["fc"] - a["fc"]).tolist())
    filas += len(a)
print(f"Filas perturbadas comprobadas: {filas} (10 repeticiones × 17 plantas)")
print("Cambio máximo en las 16 variables de la fila (debe ser 0):", cambio_variables)
print(f"Cambio de la etiqueta (debe ser 1): entre {min(cambio_etiqueta):.6f} y {max(cambio_etiqueta):.6f}")
```

```{code-cell} python
# (iii) Recálculo manual de la media móvil de 30 días para 200 filas al azar: media del FC observado en los
#       30 días calendario que terminan en D = T - 1, con al menos 15 observaciones.
muestra = tabla.sample(200, random_state=2026)
serie = base.set_index(["code", "date"])["fc"]
def media30_manual(code, T):
    s = serie.loc[code]
    v = s[(s.index >= T - pd.Timedelta(days=30)) & (s.index <= T - pd.Timedelta(days=1))]
    return v.mean() if len(v) >= 15 else np.nan
manual = np.array([media30_manual(c, T) for c, T in zip(muestra.code, muestra.date)])
print("Diferencia máxima con media30 de la tabla:", float(np.nanmax(np.abs(manual - muestra.media30.to_numpy()))))
```

Las tres comprobaciones dan diferencias nulas o de redondeo: la tabla de trabajo no contiene información del día objetivo en sus predictores, y la que usan los modelos se reproduce desde los datos versionados.

## Declaración de los datos

La declaración completa está en `data/DATA_STATEMENT.md` y se reproduce a continuación. Describe los extractos que lee el dashboard, que salen de la misma base de este capítulo.

:::{include} ../data/DATA_STATEMENT.md
:start-line: 1
:::

**Sesgos adicionales de la cohorte.** La cohorte depende de que XM registre la capacidad y de que la planta se pueda ubicar: excluye 33 recursos sin coordenadas verificadas y los primeros meses de operación de 15 plantas. Las plantas se concentran en el Caribe y los Llanos; los resultados no se pueden extrapolar a otras regiones ni a plantas pequeñas o autogeneradoras sin evaluarlas (las cuatro plantas atípicas de la ampliación muestran que su comportamiento es distinto).

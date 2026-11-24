"""Sección 1: Contexto (problema, cifra de contexto, pregunta, datos, metodología, limitaciones y autoría)."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html

from src import config, data
from src.data import DatosNoDisponibles
from src.features import DICCIONARIO
from src.plots import COLORES

dash.register_page(__name__, path="/", name="Contexto", title="Contexto · Pronóstico D+1 del FC solar", order=0)


def _kpi(valor, etiqueta, fuente=None):
    return dbc.Col(html.Div([html.Div(valor, className="valor"), html.Div(etiqueta, className="etiqueta"),
                             html.Div(fuente, className="fuente") if fuente else None], className="kpi"), xs=6, md=3, className="mb-2")


def _mapa(p):
    f = go.Figure(go.Scattergeo(lat=p.lat, lon=p.lon, mode="markers", text=p.planta,
                                marker=dict(size=(p.cap_MW_ultima.clip(lower=10) ** 0.5) * 1.6, color=COLORES["acento"], opacity=0.85,
                                            line=dict(color="#0B1220", width=1)),
                                customdata=p[["code", "grupo", "cap_MW_ultima"]],
                                hovertemplate="%{text} (%{customdata[0]})<br>grupo %{customdata[1]} · %{customdata[2]:.1f} MW<extra></extra>"))
    f.update_geos(projection_type="mercator", lataxis_range=[-2, 13.5], lonaxis_range=[-80, -66], showcountries=True, countrycolor="rgba(148,163,184,0.45)",
                  showland=True, landcolor="#162033", showocean=True, oceancolor="#0B1220", showframe=False, bgcolor="rgba(0,0,0,0)")
    f.update_layout(height=380, margin=dict(l=0, r=0, t=40, b=0), title=dict(text="Ubicación de las 17 plantas (tamaño ∝ capacidad)"))
    return f


PASOS = [
    ("ETL", "XM (generación y capacidad), Open-Meteo (clima y pronóstico archivado a 24 h), GEM (coordenadas)", False),
    ("EDA y calidad", "Cobertura, faltantes, banderas de días anómalos, dependencia temporal y espacial", False),
    ("Tabla D+1 sin fuga", "Historial hasta D, pronóstico emitido antes del cierre de D, auditoría de fuga", False),
    ("Modelos", "7 de librería (k-NN, Ridge, Lasso, árbol, RF, XGBoost, SVR) + K1 y W1 propios", True),
    ("Optimización", "GridSearchCV con 3 particiones temporales internas y hueco de 30 días", False),
    ("Validación anidada", "Bucle externo walk-forward: bloques D1, D2 y D3 (nov 2025 – jul 2026)", False),
    ("Prueba reservada", "Ago–oct 2026, entrenamiento hasta el 31 jul 2026 (1 561 filas)", False),
    ("Comparación estadística", "DM-HLN, Giacomini-White, Wilcoxon con Holm, prueba bayesiana y MCS", False),
    ("Modelo original", "K1 (anomalías sobre el nivel) y W1 (regresión local por planta y grupo); ablación", True),
    ("Evaluación ampliada", "Plantas no vistas, 15 plantas nuevas, estabilidad mensual, intervalos conformes", False),
]


def layout(**_):
    try:
        c = data.contexto()
        p = data.plantas()
    except DatosNoDisponibles as error:
        return dbc.Alert(str(error), color="danger", className="mt-3")

    pipeline = []
    for i, (t, d, propio) in enumerate(PASOS):
        pipeline.append(html.Div([html.Strong(f"{i + 1}. {t}"), d], className="paso propio" if propio else "paso"))
        if i < len(PASOS) - 1:
            pipeline.append(html.Span("→", className="flecha", **{"aria-hidden": "true"}))

    dicc = [html.Tr([html.Td(html.Code(k)), html.Td(v[0]), html.Td(v[1]),
                     html.Td("" if v[2] is None else f"[{v[2][0]:g}, {v[2][1]:g}]")]) for k, v in DICCIONARIO.items()]

    return html.Div([
        html.Div([
            html.H1("¿Cuánto generará mañana cada planta solar?"),
            html.P(["Pronóstico del ", html.Strong("factor de capacidad (FC) diario del día siguiente (D+1)"),
                    " de 17 plantas fotovoltaicas colombianas a partir del historial de generación reportado a XM y del pronóstico "
                    "meteorológico emitido 24 horas antes. El FC es la energía entregada en el día dividida por la que se habría "
                    "producido operando las 24 horas a la capacidad neta registrada."], className="lead mb-2"),
            html.P("Operar un sistema eléctrico con generación solar exige anticipar cuánta energía entregará cada planta: la potencia "
                   "nominal no basta, porque la irradiancia, la temperatura, el seguimiento solar, las pérdidas, la disponibilidad y las "
                   "restricciones de red separan la producción real de la ficha técnica (Müller et al., 2016; Trigo-González et al., 2021).",
                   className="mb-0"),
        ], className="hero"),
        dbc.Row([
            _kpi(f"{c['capacidad_total_MW']:,.0f} MW".replace(",", " "), "capacidad efectiva neta de las 17 plantas", f"Fuente: {c['fuente_capacidad']}"),
            _kpi(f"{c['energia_total_GWh']:,.0f} GWh".replace(",", " "), f"generados entre {c['fecha_min']} y {c['fecha_max']}", f"Fuente: {c['fuente_energia']}"),
            _kpi(f"{c['filas_tabla_d1']:,}".replace(",", " "), "días-planta en la tabla de pronóstico D+1", "12 756 días-planta en la base"),
            _kpi(f"{c['fc_medio']:.3f}", "FC diario medio de la cohorte", "AC neta; 0.25 ≈ 6 h equivalentes a plena carga"),
        ], className="g-2 mb-3"),

        html.H2("Pregunta de investigación y objetivos", className="seccion"),
        dbc.Card(dbc.CardBody([
            html.P(html.Strong("¿Reduce una corrección basada en el nivel reciente de generación de cada planta el error de los modelos de "
                               "referencia al pronosticar el FC del día siguiente, y con qué certeza estadística?")),
            html.Ul([
                html.Li("Construir una base planta-día reproducible y una tabla D+1 sin fuga de información."),
                html.Li("Comparar siete modelos de librería con búsqueda de hiperparámetros dentro de una validación temporal anidada."),
                html.Li("Formular, implementar y probar dos modelos propios: K1 (anomalías del pronóstico sobre el nivel reciente) y W1 "
                        "(regresión local ponderada por planta y grupo)."),
                html.Li("Verificar la diferencia con pruebas para series de tiempo (DM-HLN, Giacomini-White, MCS) y estudios de robustez."),
                html.Li("Publicar los modelos en un tablero que permita explorar los datos y predecir con incertidumbre y explicación."),
            ], className="mb-0"),
        ])),

        html.H2("Datos", className="seccion"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Dl([
                    html.Dt("Fuentes"), html.Dd("XM (métricas Gene y CapEfecNeta), Open-Meteo Previous Runs API (ECMWF IFS 0.25°, GFS e ICON, "
                                                "valor pronosticado 24 h antes, promediado) y reanálisis/best match, Global Energy Monitor (coordenadas)."),
                    html.Dt("Periodo"), html.Dd(f"{c['fecha_min']} a {c['fecha_max']}; tabla D+1 desde el 2024-03-21."),
                    html.Dt("Observaciones y variables"), html.Dd(f"{c['filas_base']:,} días-planta en la base; {c['filas_tabla_d1']:,} filas D+1 "
                                                                  f"({c['filas_entrenamiento']:,} de entrenamiento y {c['filas_prueba']:,} de prueba); "
                                                                  f"{c['plantas']} plantas en {c['grupos']} grupos espaciales; 8 predictores (B2) y 5 anomalías derivadas."),
                    html.Dt("Variable objetivo y tarea"), html.Dd("FC diario de la planta en T = D+1 (continuo en [0, 1.05]); regresión."),
                    html.Dt("Licencia"), html.Dd(["Open-Meteo: CC BY 4.0 según su sitio. XM y Global Energy Monitor: ",
                                                  html.Strong("POR COMPLETAR (verificar condiciones de uso y redistribución)."),
                                                  " Ver data/DATA_STATEMENT.md."]),
                ], className="mb-0"),
            ])), lg=6),
            dbc.Col(dbc.Card(dbc.CardBody(html.Div(dcc.Graph(figure=_mapa(p), config={"displayModeBar": False}), role="img",
                                                    **{"aria-label": "Mapa de Colombia con la ubicación de las 17 plantas"}))), lg=6),
        ]),
        dbc.Card([dbc.CardHeader("Diccionario de variables"), dbc.CardBody(dbc.Table(
            [html.Thead(html.Tr([html.Th("Variable"), html.Th("Descripción"), html.Th("Unidad"), html.Th("Rango válido")]))] + [html.Tbody(dicc)],
            bordered=False, hover=True, size="sm", className="mb-0"))]),

        html.H2("Metodología", className="seccion"),
        html.Div(pipeline, className="pipeline mb-2", role="img",
                 **{"aria-label": "Diagrama del flujo: ETL, EDA, tabla D+1, modelos, optimización, validación anidada, prueba reservada, "
                                  "comparación estadística, modelo original y evaluación ampliada"}),
        html.Small("Los pasos en cian corresponden al modelo original. Detalle en la página Modelos y en paper/manuscrito.tex.", className="text-muted"),

        html.H2("Limitaciones, ética y sesgo", className="seccion"),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardHeader("Limitaciones"), dbc.CardBody(html.Ul([
                html.Li("La prueba de ago–oct 2026 ya se usó en análisis previos: las cifras de prueba no son una confirmación independiente."),
                html.Li("La mejora de K1 y W1 es una reducción del error de pronóstico, no un aumento de la generación; no se evaluó su valor económico ni su efecto en el despacho horario, y el pronóstico D+1 es retrospectivo (pronósticos archivados)."),
                html.Li("Tres meses de prueba y 17 plantas correlacionadas (7 grupos): poca potencia para detectar diferencias pequeñas."),
                html.Li("El FC mezcla recurso solar, disponibilidad, mantenimiento y restricciones de red; no distingue la causa de un día bajo."),
                html.Li("Se supone que el FC del día D está disponible al cierre de D; debe verificarse con los horarios de publicación de XM."),
                html.Li("Las variables meteorológicas son celdas de un modelo numérico, no mediciones en sitio."),
            ], className="mb-0"))]), md=6),
            dbc.Col(dbc.Card([dbc.CardHeader("Ética y sesgo"), dbc.CardBody(html.Ul([
                html.Li("Datos públicos y agregados por planta: no hay datos personales."),
                html.Li("La cohorte se concentra en el Caribe y los Llanos; el desempeño puede no trasladarse a otras regiones o tecnologías."),
                html.Li("El tablero es una herramienta académica: no debe usarse para ofertas en el mercado ni para despacho sin validación operativa."),
                html.Li("Se usaron asistentes de IA (Claude Code y Codex) para programación y redacción; el equipo responde por el contenido."),
            ], className="mb-0"))]), md=6),
        ]),

        html.H2("Autoría y versión", className="seccion"),
        dbc.Card(dbc.CardBody([
            html.P([html.Strong("Autor: "), "Enmanuel David Diaz Molinares (Departamento de Matemáticas y Ciencia de Datos, Programa de Ciencia de Datos, Universidad del Norte, Barranquilla, Colombia)."]),
            html.P([html.Strong("Versión: "), f"{config.VERSION_APP} · ", html.Strong("actualización: "), config.FECHA_VERSION, " · ",
                    html.Strong("semilla global: "), str(config.SEMILLA)]),
            html.P([html.Strong("Repositorio: "), html.A("github.com/KrissV89/sm-cfhr", href="https://github.com/KrissV89/sm-cfhr", target="_blank"),
                    " · Tablero: https://pronostico-fc-solar.onrender.com"]),
            html.P([html.Strong("Referencias citadas: "), "Müller, B. et al. (2016) y Trigo-González, M. et al. (2021), ver paper/referencias.bib; "
                    "Open-Meteo Previous Runs API; XM, portal de datos del mercado."], className="mb-0"),
        ])),
    ])

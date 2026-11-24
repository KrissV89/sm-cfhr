"""Punto de entrada del dashboard (Dash multipágina). Expone ``server`` para ``gunicorn app:server``.

Páginas (carpeta ``pages/``): Contexto (``/``), EDA (``/eda``) y Modelos (``/modelos``). La app solo carga los
extractos de ``data/`` y los artefactos de ``artifacts/`` una vez al iniciar; nunca entrena ni optimiza.
Ejecución local: ``python app.py`` (modo desarrollo) o ``gunicorn app:server -b 127.0.0.1:8050``.
"""

from __future__ import annotations

import os

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html

from src import config, data

PROBLEMAS_INICIO = data.precargar()   # carga datos y modelos una sola vez por proceso

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP,
                          "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap"],
    title="Pronóstico D+1 del FC solar",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"},
               {"name": "color-scheme", "content": "dark"},
               {"name": "description", "content": "Pronóstico del factor de capacidad del día siguiente en 17 plantas solares colombianas"}],
)
app.index_string = app.index_string.replace("<html>", '<html lang="es" data-bs-theme="dark">')
server = app.server

# Ruta de lectura: el orden en que conviene recorrer el tablero (la misma que exige la guía: Contexto, EDA, Modelos).
RUTA = [
    ("/", "Contexto", "Problema, datos y método"),
    ("/eda", "EDA", "Qué muestran los datos"),
    ("/modelos", "Modelos", "Referencias, K1 y conclusiones"),
]

lateral = html.Nav([
    dcc.Link([html.Span("☀", className="marca-icono", **{"aria-hidden": "true"}),
              html.Span(["Pronóstico D+1", html.Small("Factor de capacidad solar")], className="marca-texto")], href="/", className="marca"),
    html.Ul(id="ruta", className="ruta"),
    html.Div([
        dbc.Button("Ficha del modelo", id="abrir-ficha", color="light", outline=True, size="sm"),
        html.Small("Universidad del Norte · Machine Learning"),
    ], className="lateral-pie"),
], className="lateral", **{"aria-label": "Navegación principal"})

ficha = dbc.Modal([
    dbc.ModalHeader(dbc.ModalTitle("Ficha del modelo (model card)")),
    dbc.ModalBody(dcc.Markdown(data.model_card(), mathjax=False, className="ficha-md")),
    dbc.ModalFooter(dbc.Button("Cerrar", id="cerrar-ficha", color="primary")),
], id="modal-ficha", size="xl", scrollable=True, is_open=False)

aviso_inicio = (dbc.Alert([html.Strong("Faltan datos o artefactos. "), html.Ul([html.Li(p) for p in PROBLEMAS_INICIO])],
                          color="danger", className="mt-3") if PROBLEMAS_INICIO else None)

app.layout = html.Div([
    dcc.Location(id="url"),
    lateral,
    ficha,
    html.Main([aviso_inicio, dash.page_container, html.Div(id="paginador", className="paginador")], className="principal"),
    html.Footer(html.Small(f"Versión {config.VERSION_APP} · actualizado el {config.FECHA_VERSION} · Datos: XM, Open-Meteo y Global Energy Monitor · "
                           "Proyecto del curso de Machine Learning"), className="pie"),
])


@app.callback(Output("ruta", "children"), Output("paginador", "children"), Input("url", "pathname"))
def actualizar_ruta(ruta):
    """Marca la sección actual y las ya recorridas, y ofrece los botones Anterior / Siguiente."""
    ruta = (ruta or "/").rstrip("/") or "/"
    actual = next((i for i, (href, _, _) in enumerate(RUTA) if href == ruta), 0)
    pasos = []
    for i, (href, nombre, detalle) in enumerate(RUTA):
        estado = "actual" if i == actual else ""
        pasos.append(html.Li(dcc.Link([
            html.Span(str(i + 1), className="punto", **{"aria-hidden": "true"}),
            html.Span([nombre, html.Small(detalle)], className="paso-nombre")], href=href, className=estado)))
    anterior = (dcc.Link(dbc.Button(f"← {RUTA[actual - 1][1]}", color="secondary"), href=RUTA[actual - 1][0]) if actual > 0 else html.Span())
    siguiente = (dcc.Link(dbc.Button(f"Siguiente: {RUTA[actual + 1][1]} →", color="primary"), href=RUTA[actual + 1][0]) if actual < len(RUTA) - 1
                 else dcc.Link(dbc.Button("Volver al inicio ↑", color="secondary"), href="/"))
    return pasos, [anterior, siguiente]


@app.callback(Output("modal-ficha", "is_open"), Input("abrir-ficha", "n_clicks"), Input("cerrar-ficha", "n_clicks"),
              State("modal-ficha", "is_open"), prevent_initial_call=True)
def alternar_ficha(_a, _c, abierto):
    return not abierto


if __name__ == "__main__":
    app.run(debug=os.environ.get("DASH_DEBUG", "1") == "1", host="127.0.0.1", port=int(os.environ.get("PORT", 8050)))

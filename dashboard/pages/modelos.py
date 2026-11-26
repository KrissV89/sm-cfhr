"""Sección 3: Modelos (tabla de líderes, predicción en vivo con intervalo y SHAP local, diagnóstico, SHAP global,
comparación estadística, modelo original, robustez y veredicto sobre R3)."""

from __future__ import annotations

import json

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import ALL, Input, Output, State, callback, dash_table, dcc, html

from src import config, data, plots, prediccion
from src.data import DatosNoDisponibles
from src.series import acf, banda_95

dash.register_page(__name__, path="/modelos", name="Modelos", title="Modelos · Pronóstico D+1 del FC solar", order=2)

GRAF = {"displaylogo": False, "responsive": True, "displayModeBar": "hover", "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
ESTILO_TABLA = dict(
    style_table={"overflowX": "auto"},
    style_cell={"fontFamily": "Inter, system-ui, sans-serif", "fontSize": 13, "padding": "6px 8px", "textAlign": "left",
                "whiteSpace": "normal", "height": "auto", "minWidth": "80px", "maxWidth": "320px"},
    style_header={"fontWeight": 600},
    style_data_conditional=[{"if": {"filter_query": '{tipo} = "propio"'}, "backgroundColor": "rgba(34, 211, 238, .12)"},
                            {"if": {"state": "active"}, "backgroundColor": "rgba(245, 165, 36, .16)", "border": "1px solid #F5A524"}],
)

PSEUDO_K1 = """Algoritmo K1 · ajuste y predicción
Entrada: series y[p,t] (FC) y x[p,t] (pronóstico) por planta; penalización α; corte del entrenamiento
Salida: ŷ[p,T] para cada fila de consulta
1  para cada planta p y fecha objetivo T = D+1:
2      L[p,D] ← media de y[p,·] en los 30 días que terminan en D            (mínimo 15 con dato)
3      a[p,T] ← x[p,T] − media de x[p,·] en los 30 días que terminan en T   (mínimo 15)
4  (m, s) ← media y desviación de a en el entrenamiento;  z ← (a − m) / s
5  r_i ← y_i − L_i en el entrenamiento
6  β0 ← media(r);  β ← (ZᵀZ + αI)⁻¹ Zᵀ (r − β0·1)                         (solución cerrada, convexa)
7  para cada consulta (p, T):  ŷ ← L[p,D] + β0 + βᵀ z[p,T]
Parada: no iterativo (solución exacta). Costo: ajuste O(nk² + k³), predicción O(k)."""

PSEUDO_W1 = """Algoritmo W1 · predicción de una fila de consulta q
Entrada: entrenamiento {(z_i, r_i, p_i, g_i)} estandarizado; consulta (z_q, p_q, g_q, L_q); h, ρ_g, ρ_o, γ
Salida: ŷ_q
1  para i = 1..n:
2      c_i ← 1 si p_i = p_q;  ρ_g si g_i = g_q;  ρ_o en otro caso
3      w_i ← exp(−‖z_i − z_q‖² / h²) · c_i + 10⁻⁸
4  G ← Σ w_i z̃_i z̃_iᵀ + γ (Σ w_i) Λ ;  b ← Σ w_i z̃_i r_i          (z̃_i = (1, z_i), Λ = diag(0,1,…,1))
5  θ ← G⁻¹ b                                                     (sistema (k+1)×(k+1))
6  ŷ_q ← L_q + θ0 + θ_{1:k}ᵀ z_q
Parada: no iterativo. Costo por consulta O(nk² + k³); ajuste O(nk²) (método perezoso).
Casos límite: h→∞, ρ=1 ⇒ K1 con α = γn;  h→∞, ρ→0 ⇒ regresión separada por planta."""


ALTOS = {"diag-real-pred": 380, "diag-hist": 380, "diag-tiempo": 380, "diag-acf": 320, "shap-barras": 400, "shap-puntos": 400, "curva": 380}


def _grafico(id_, alt, **kw):
    return html.Div(dcc.Loading(dcc.Graph(id=id_, config=GRAF, responsive=True, style={"width": "100%", "height": f"{ALTOS.get(id_, 380)}px"}, **kw),
                                type="dot", parent_style={"width": "100%", "minWidth": 0}), role="img", **{"aria-label": alt})


def _opciones_modelo():
    return [{"label": f"{n} ({'propio' if k in config.MODELOS_PROPIOS else 'librería'})", "value": k} for k, n in config.MODELOS.items()]


def _tabla_lideres():
    lb, origen = data.leaderboard()
    if "MAE_externo_media" in lb.columns:
        t = lb.copy()
        t["MAE bucle externo (media ± d.e.)"] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(t.MAE_externo_media, t.MAE_externo_desv)]
        cols = ["modelo", "tipo", "optimizador", "MAE bucle externo (media ± d.e.)", "MAE_D1", "MAE_D2", "MAE_D3", "MAE_prueba", "RMSE_prueba",
                "us_por_fila", "MB", "hiperparametros"]
        t = t[cols].rename(columns={"us_por_fila": "µs por fila", "MAE_prueba": "MAE prueba", "RMSE_prueba": "RMSE prueba", "MB": "tamaño (MB)"})
        for c in ["MAE_D1", "MAE_D2", "MAE_D3", "MAE prueba", "RMSE prueba"]:
            t[c] = t[c].round(4)
        t["µs por fila"] = t["µs por fila"].round(1)
        t["tamaño (MB)"] = t["tamaño (MB)"].round(2)
    else:
        t = lb.copy()
        if "tipo" not in t.columns:
            t["tipo"] = ""
    columnas = [{"name": c, "id": c, "type": "numeric" if pd.api.types.is_numeric_dtype(t[c]) else "text"} for c in t.columns]
    return t, columnas, origen



def _kpi(valor, etiqueta, nota, clase="", principal=False):
    return dbc.Col(html.Div([html.Div(etiqueta, className="etiqueta"), html.Div(valor, className="valor"), html.Div(nota, className="fuente")],
                            className=f"kpi {clase}" + (" principal-kpi" if principal else "")), xs=12, sm=6, xl=3)


def _cabecera():
    """Parte alta de la página (lectura en Z): pregunta, cuatro cifras clave y el veredicto, antes de los detalles."""
    imp = data.impacto().set_index("modelo")
    k1, rf = imp.loc["K1 Modelo de anomalías"], imp.loc["Random Forest"]
    red = (k1.MAE_fc_macro / rf.MAE_fc_macro - 1) * 100
    red_e = (k1.suma_errores_abs_flota_mwh_dia / rf.suma_errores_abs_flota_mwh_dia - 1) * 100
    return html.Div([
        html.Div([
            html.H1("¿Corregir con el nivel reciente de cada planta reduce el error del pronóstico?"),
            html.P(["Random Forest fue la mejor referencia en desarrollo. K1 corrige la media de 30 días de cada planta con la anomalía del "
                    "pronóstico meteorológico del día siguiente. Todas las cifras son de la prueba reservada (ago–oct 2026, 92 días, 17 plantas) "
                    "y el pronóstico es retrospectivo: usa pronósticos archivados."], className="lead mb-0"),
        ], className="hero"),
        dbc.Row([
            _kpi(f"{rf.MAE_fc_macro:.4f}", "Random Forest · mejor referencia", "MAE macro por planta (unidades de FC)", "ambar"),
            _kpi(f"{k1.MAE_fc_macro:.4f}", "K1 · modelo de anomalías", f"{red:+.1f} % frente a Random Forest", "cian", principal=True),
            _kpi(f"{red_e:+.1f} %", "Error de energía de la flota",
                 f"{rf.suma_errores_abs_flota_mwh_dia:,.0f} → {k1.suma_errores_abs_flota_mwh_dia:,.0f} MWh/día sobre {k1.energia_flota_mwh_dia:,.0f} producidos".replace(",", "\u00a0"), "verde"),
            _kpi("p = 0.076", "K1 frente a Random Forest, DM-HLN con Holm", "Diferencia no significativa; la regla prefijada no resiste Holm (R3)", ""),
        ], className="g-3 mb-2"),
        html.Small("La reducción de error no es generación adicional ni ahorro económico: no se evaluó valor económico. "
                   "MAE macro = promedio por planta; la suma de errores se calcula día a día con la capacidad de cada planta.", className="text-muted"),
    ])

def layout(**_):
    try:
        t, columnas, origen = _tabla_lideres()
        tabla_d1 = data.tabla_d1()
        p = data.plantas()
        campos = prediccion.esquema_formulario()
        lb = data.leaderboard_artefactos()
    except DatosNoDisponibles as error:
        return dbc.Alert(str(error), color="danger", className="mt-3")
    mejor = lb.sort_values("MAE_externo_media").iloc[0]
    mejor_lib = lb[lb.tipo == "librería"].sort_values("MAE_externo_media").iloc[0]
    fmin, fmax = tabla_d1.date.min().date(), tabla_d1.date.max().date()

    formulario = []
    for c in campos:
        formulario.append(dbc.Col([
            dbc.Label(f"{c['nombre']} ({c['unidad']})", html_for={"type": "campo-pred", "nombre": c["nombre"]}, className="small fw-semibold mb-0"),
            dbc.Input(id={"type": "campo-pred", "nombre": c["nombre"]}, type="number", min=c["min"], max=c["max"], step="any", size="sm",
                      debounce=True, required=True),
            dbc.FormText(f"{c['descripcion']}. Válido: [{c['min']:g}, {c['max']:g}].", className="small"),
        ], md=6, className="mb-2"))

    return html.Div([
        _cabecera(),
        dbc.Card(dbc.CardBody(dbc.Row([
            dbc.Col([dbc.Label("Modelo seleccionado (controla predicción, diagnóstico y SHAP global)", html_for="mod-modelo", className="fw-semibold small"),
                     dcc.Dropdown(id="mod-modelo", options=_opciones_modelo(), value="k1", clearable=False)], md=7),
            dbc.Col(html.Div(id="mod-resumen-modelo", className="small"), md=5),
        ], className="g-2 align-items-center")), className="filtros"),

        html.H2("Tabla de líderes: mejores combinaciones", className="seccion"),
        dash_table.DataTable(id="mod-lideres", data=t.to_dict("records"), columns=columnas, sort_action="native", filter_action="native",
                             page_size=12, **ESTILO_TABLA),
        html.P([f"Origen: {origen}. Bucle externo = bloques de desarrollo D1–D3 (walk-forward, cada uno con GridSearchCV interno de 3 particiones temporales). "
                f"Menor MAE medio del bucle externo: {mejor.modelo} ({mejor.MAE_externo_media:.4f} ± {mejor.MAE_externo_desv:.4f}); mejor de librería: "
                f"{mejor_lib.modelo} ({mejor_lib.MAE_externo_media:.4f} ± {mejor_lib.MAE_externo_desv:.4f}). Las desviaciones entre bloques (≈0.004) superan las "
                "diferencias entre los mejores modelos: el orden no es concluyente por sí solo; ver la comparación estadística. «MAE prueba» es la ventana "
                "ago–oct 2026, ya usada antes (post hoc). Filas en naranja: modelos propios. Escriba en la fila de filtros para filtrar (p. ej. «propio»)."],
               className="insight"),

        html.H2("Predicción en vivo", className="seccion"),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardHeader("Entrada (generada desde el esquema de artifacts/*_metadata.json)"), dbc.CardBody([
                dbc.Row([
                    dbc.Col([dbc.Label("Planta", html_for="pred-planta", className="small fw-semibold mb-0"),
                             dcc.Dropdown(id="pred-planta", options=[{"label": f"{r.code} · {r.planta.title()}", "value": r.code} for r in p.itertuples()],
                                          value="GYPO", clearable=False)], md=6),
                    dbc.Col([dbc.Label("Fecha objetivo T", html_for="pred-fecha", className="small fw-semibold mb-0"),
                             dcc.DatePickerSingle(id="pred-fecha", min_date_allowed=fmin, max_date_allowed=fmax, date=fmax, display_format="YYYY-MM-DD",
                                                  first_day_of_week=1)], md=6),
                ], className="mb-2"),
                html.Small("Al elegir planta y fecha se carga el historial guardado (hasta D = T−1) y el pronóstico archivado para T; puede editarlos.",
                           className="text-muted d-block mb-2"),
                html.Div(id="pred-info-fila", className="small mb-2"),
                dbc.Row(formulario),
                dbc.Row([
                    dbc.Col([dbc.Label("Nivel del intervalo", className="small fw-semibold mb-0"),
                             dbc.RadioItems(id="pred-nivel", options=[{"label": f"{int(n * 100)} %", "value": n} for n in (0.8, 0.9, 0.95)], value=0.9, inline=True),
                             dbc.Label("Calibración", className="small fw-semibold mb-0 mt-1"),
                             dbc.RadioItems(id="pred-metodo", options=[{"label": "global", "value": "global"}, {"label": "por planta (Mondrian)", "value": "planta"}],
                                            value="global", inline=True)], md=7),
                    dbc.Col(dbc.Button("Predecir", id="pred-boton", color="primary", className="w-100 mt-4"), md=5),
                ]),
            ])]), lg=5),
            dbc.Col(dbc.Card([dbc.CardHeader("Resultado"), dbc.CardBody(dcc.Loading(html.Div(id="pred-salida", children=html.P(
                "Elija planta y fecha y pulse «Predecir».", className="text-muted")), type="dot"))]), lg=7),
        ]),

        html.H2("Diagnóstico del modelo seleccionado (ventana de prueba ago–oct 2026)", className="seccion"),
        dbc.Card(dbc.CardBody([
            dbc.Row(dbc.Col([dbc.Label("Plantas", html_for="diag-plantas", className="small fw-semibold mb-0"),
                             dcc.Dropdown(id="diag-plantas", options=[{"label": f"{r.code} · {r.planta.title()}", "value": r.code} for r in p.itertuples()],
                                          multi=True, placeholder="Todas")], md=6)),
            html.P(id="diag-metricas", className="insight"),
            dbc.Row([dbc.Col(_grafico("diag-real-pred", "Dispersión del FC real frente al predicho"), lg=6),
                     dbc.Col(_grafico("diag-hist", "Histograma de los residuos"), lg=6)]),
            _grafico("diag-tiempo", "Residuos a lo largo del tiempo"),
            _grafico("diag-acf", "Autocorrelación de la media diaria de los residuos"),
            html.P(id="diag-txt", className="insight"),
        ])),

        html.H2("Interpretabilidad global (SHAP)", className="seccion"),
        dbc.Card(dbc.CardBody([
            dbc.Row([dbc.Col(_grafico("shap-barras", "Importancia media absoluta SHAP por variable"), lg=5),
                     dbc.Col(_grafico("shap-puntos", "Resumen SHAP por fila y variable"), lg=7)]),
            html.P(id="shap-txt", className="insight"),
        ])),

        html.H2("Comparación estadística: Model Confidence Set (MCS)", className="seccion"),
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col(dbc.RadioItems(id="mcs-universo", options=[{"label": u, "value": u} for u in data.mcs().universo.unique()],
                                       value="con W1", inline=True), md=7),
                dbc.Col(dbc.RadioItems(id="mcs-perdida", options=[{"label": f"pérdida {x}", "value": x} for x in data.mcs().perdida.unique()],
                                       value="absoluta", inline=True), md=5),
            ]),
            dash_table.DataTable(id="mcs-tabla", sort_action="native", filter_action="native", page_size=16,
                                 style_data_conditional=[{"if": {"filter_query": '{en_mcs_90} = "sí"'}, "fontWeight": 600},
                                                         {"if": {"filter_query": '{modelo} contains "K1" || {modelo} contains "W1"'}, "backgroundColor": "rgba(34, 211, 238, .12)"}],
                                 **{k: v for k, v in ESTILO_TABLA.items() if k != "style_data_conditional"}),
            html.P(id="mcs-txt", className="insight"),
        ])),

        html.H2("Modelo original", className="seccion"),
        dbc.Tabs([
            dbc.Tab(dbc.Card(dbc.CardBody([
                html.P(["K1 predice el FC de mañana como el ", html.Strong("nivel reciente de la planta"), " (media de los 30 días que terminan en D) más una "
                        "corrección lineal con las ", html.Strong("anomalías del pronóstico"), " respecto de su propia media móvil de 30 días. El nivel absorbe la "
                        "heterogeneidad entre plantas (DC/AC, montaje, registro de capacidad) sin estimar un parámetro por planta."]),
                html.Pre(PSEUDO_K1, className="pseudo"),
                html.H3("Diferencias respecto de la literatura", className="h6"),
                html.Ul([
                    html.Li("El estimador es la regresión Ridge de scikit-learn (técnica establecida); lo propio es la parametrización: respuesta r = y − L, "
                            "anomalías del pronóstico respecto de su media móvil y coeficiente del nivel fijo en 1."),
                    html.Li("Equivale a la transformación intragrupos de un modelo de efectos fijos con efecto de planta que varía lentamente; las referencias "
                            "de librería reciben el historial y el pronóstico en bruto (B2)."),
                    html.Li("La ablación (cuaderno 10) muestra que casi toda la mejora viene del nivel corregido con la anomalía de radiación; la auditoría del "
                            "proyecto (docs/entregables/AUDITORIA_GUIAS_2_Y_3.md) no acredita novedad de mecanismo (R1)."),
                ]),
            ])), label="K1 · anomalías sobre el nivel", tab_id="tab-k1"),
            dbc.Tab(dbc.Card(dbc.CardBody([
                html.P(["W1 generaliza K1: para cada fila de consulta ajusta una regresión lineal propia en la que pesan más las filas de entrenamiento ",
                        html.Strong("parecidas en el espacio de anomalías"), " y las de la ", html.Strong("misma planta o grupo espacial"),
                        " (agrupamiento parcial). La penalización escala con la suma de pesos."]),
                html.Pre(PSEUDO_W1, className="pseudo"),
                html.H3("Diferencias respecto de la literatura", className="h6"),
                html.Ul([
                    html.Li("La regresión localmente ponderada es una técnica establecida (Cleveland, 1979; Atkeson, Moore y Schaal, 1997)."),
                    html.Li("Lo propio: su aplicación al residuo de nivel en el espacio de anomalías, el factor de peso por identidad de planta y de grupo "
                            "(ρ_g, ρ_o) y la penalización γ·S(q) proporcional a la suma de pesos. Implementación propia con NumPy/SciPy (src/original_model.py)."),
                    html.Li("W1 usa la identidad de planta y grupo, que las referencias no reciben: la comparación es entre sistemas completos y fue exploratoria."),
                ]),
            ])), label="W1 · regresión local", tab_id="tab-w1"),
        ], active_tab="tab-k1", className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardHeader("Curva de aprendizaje (exploratoria, bloque D3)"), dbc.CardBody([
                _grafico("curva", "Curva de aprendizaje: MAE en D3 según el número de filas de entrenamiento"),
                html.P(id="curva-txt", className="insight")])]), lg=6),
            dbc.Col(dbc.Card([dbc.CardHeader("Pruebas unitarias del modelo original (tests/, pytest en CI)"), dbc.CardBody(dbc.Table([
                html.Thead(html.Tr([html.Th("Requisito 5.4"), html.Th("Prueba")])),
                html.Tbody([html.Tr([html.Td(a), html.Td(b)]) for a, b in [
                    ("(i) aprende con solución conocida", "K1 recupera β verdaderas; W1 captura una relación cuadrática que K1 no recoge"),
                    ("(ii) reproducibilidad", "mismas predicciones al reajustar; W1 invariante al orden de las filas"),
                    ("(iii) entradas inválidas", "NaN, infinitos, columnas faltantes, h ≤ 0, planta faltante → ValueError; sin ajustar → NotFittedError"),
                    ("(iv) clone", "sklearn.base.clone conserva hiperparámetros; integración con Pipeline, GridSearchCV y cross_val_score"),
                    ("(v) propiedades teóricas", "solución cerrada de K1 = ecuación del paper; mínimo estricto; W1 → K1 con h→∞, ρ=1, α=γn; "
                                                 "pesos por planta y grupo; ρ→0 = regresión por planta; W1 = mínimos cuadrados ponderados independiente"),
                    ("Reproducción", "experiments/verificar_modelos_originales.py: K1 y W1 reproducen las predicciones guardadas (diferencia 0)"),
                ]])], size="sm", bordered=False, className="mb-0"))]), lg=6),
        ]),
        dbc.Card([dbc.CardHeader("Pruebas de hipótesis con p ajustados (Holm) y tamaño del efecto · ventana de prueba"), dbc.CardBody([
            dbc.RadioItems(id="pruebas-familia", options=[{"label": "Familia K1", "value": "K1"}, {"label": "Familia W1", "value": "W1"}], value="K1", inline=True),
            dash_table.DataTable(id="pruebas-tabla", sort_action="native", filter_action="native", page_size=11,
                                 style_data_conditional=[{"if": {"filter_query": "{decisiva} = sí"}, "backgroundColor": "rgba(34, 211, 238, .12)", "fontWeight": 600}],
                                 **{k: v for k, v in ESTILO_TABLA.items() if k != "style_data_conditional"}),
            html.P(id="pruebas-txt", className="insight"),
        ])]),
        dbc.Card([dbc.CardHeader("Resumen de robustez"), dbc.CardBody(dash_table.DataTable(
            data=data.robustez().to_dict("records"), columns=[{"name": c, "id": c} for c in data.robustez().columns], **ESTILO_TABLA))]),
        html.Div(id="veredicto-r3"),
        dbc.Button("Abrir la ficha del modelo (model card)", id="abrir-ficha-modelos", color="secondary", outline=True, className="mt-2"),
    ])


# ----------------------------------------------------------------------------------------------------------- callbacks
@callback(Output("mod-resumen-modelo", "children"), Input("mod-modelo", "value"))
def resumen_modelo(clave):
    try:
        m = data.metadata(clave)
    except DatosNoDisponibles as error:
        return dbc.Alert(str(error), color="danger")
    return [html.Div([html.Strong("Hiperparámetros: "), html.Code(json.dumps(m["hiperparametros"], ensure_ascii=False))]),
            html.Div([html.Strong("Origen: "), m["origen_hiperparametros"]]),
            html.Div([html.Strong("Ida y vuelta: "), "idéntica" if m["verificacion"]["ida_y_vuelta_identica"] else "FALLA",
                      " · ", html.Strong("entrenado hasta "), m["entrenamiento"]["fechas_objetivo"][1]])]


@callback(Output("modal-ficha", "is_open", allow_duplicate=True), Input("abrir-ficha-modelos", "n_clicks"), prevent_initial_call=True)
def abrir_ficha(n):
    return True if n else dash.no_update   # no abrir al montar la página


@callback(Output({"type": "campo-pred", "nombre": ALL}, "value"), Output("pred-info-fila", "children"),
          Input("pred-planta", "value"), Input("pred-fecha", "date"), State({"type": "campo-pred", "nombre": ALL}, "id"))
def cargar_fila(planta, fecha, ids):
    try:
        fila = prediccion.fila_guardada(planta, fecha)
    except (ValueError, DatosNoDisponibles) as error:
        return [dash.no_update] * len(ids), dbc.Alert(str(error), color="warning", className="py-2 mb-0")
    valores = [round(float(fila[i["nombre"]]), 6) for i in ids]
    periodo = "prueba (fuera del entrenamiento)" if fila["periodo"] == "prueba" else "entrenamiento (el modelo ya vio esta fila)"
    return valores, html.Span([html.Strong(f"Fila cargada: {planta}, T = {pd.Timestamp(fecha).date()}"), f" · periodo: {periodo} · FC observado en T: {fila['fc']:.3f}"])


@callback(Output("pred-salida", "children"), Input("pred-boton", "n_clicks"),
          State("mod-modelo", "value"), State("pred-planta", "value"), State("pred-fecha", "date"),
          State({"type": "campo-pred", "nombre": ALL}, "value"), State({"type": "campo-pred", "nombre": ALL}, "id"), State("pred-nivel", "value"), State("pred-metodo", "value"),
          prevent_initial_call=True)
def predecir(_n, clave, planta, fecha, valores, ids, nivel, metodo="global"):
    try:
        campos = prediccion.esquema_formulario()
        entrada = {i["nombre"]: v for i, v in zip(ids, valores)}
        errores, avisos = prediccion.validar(entrada, campos)
        if errores:
            return dbc.Alert([html.Strong("Entrada inválida: no se calcula la predicción."), html.Ul([html.Li(e) for e in errores])], color="danger")
        base = prediccion.fila_guardada(planta, fecha)
        X = prediccion.entradas(clave, base, entrada)
        y = prediccion.predecir(clave, X)
        iv = prediccion.intervalo(clave, y, nivel, planta, metodo)
        sh = prediccion.shap_local(clave, X)
    except (ValueError, KeyError, DatosNoDisponibles) as error:
        return dbc.Alert(f"No se pudo predecir: {error}", color="danger")
    p = data.plantas().set_index("code").loc[planta]
    historia = data.panel()
    historia = historia[(historia.code == planta) & (historia.date < pd.Timestamp(fecha)) & (historia.date >= pd.Timestamp(fecha) - pd.Timedelta(days=30))]
    editado = any(abs(float(entrada[k]) - float(base[k])) > 1e-9 for k in entrada)
    real = None if editado else float(base["fc"])
    nombre = config.MODELOS[clave]
    derivadas = None
    if clave in ("k1", "w1"):
        derivadas = html.Small("Entradas derivadas para " + nombre.split()[0] + ": " + ", ".join(f"{c} = {X[c].iloc[0]:.3f}" for c in X.columns if c.startswith(("anom", "media30"))),
                               className="text-muted d-block")
    return html.Div([
        dbc.Alert([html.Strong("Aviso de extrapolación: "), html.Ul([html.Li(a) for a in avisos], className="mb-0")], color="warning", className="py-2") if avisos else None,
        dbc.Row([
            dbc.Col([html.Div("FC predicho", className="small text-muted"), html.Div(f"{y:.3f}", className="prediccion-valor"),
                     html.Div(f"≈ {y * p.cap_MW_ultima * 24:,.0f} MWh con {p.cap_MW_ultima:g} MW".replace(",", "\u00a0"), className="small"),
                     html.Div("Energía pronosticada; el error típico de una planta es de ±0.04 de FC (≈ ±1 h de capacidad plena al día).", className="small text-muted")], xs=6, md=4),
            dbc.Col([html.Div(f"Intervalo al {nivel:.0%}", className="small text-muted"), html.Div(f"[{iv['inferior']:.3f}, {iv['superior']:.3f}]", className="h4 mb-0"),
                     html.Div(f"cobertura empírica en prueba ({iv['ambito_cobertura']}): {iv['cobertura_prueba']:.1%} (nominal {nivel:.0%})", className="small")], xs=6, md=5),
            dbc.Col([html.Div("FC observado en T", className="small text-muted"),
                     html.Div("—" if real is None else f"{real:.3f}", className="h4 mb-0"),
                     html.Div("entrada editada: no comparable" if editado else f"error {real - y:+.3f}", className="small")], xs=12, md=3),
        ], className="mb-2"),
        derivadas,
        html.Small(f"Intervalo: ŷ ± {iv['semiancho']:.4f}; {iv['origen']} (n = {iv['n_calibracion']}); el límite inferior se recorta en 0. "
                   "La cobertura empírica menor que la nominal indica que el periodo de prueba fue más difícil que el de calibración.", className="text-muted d-block mb-2"),
        dcc.Graph(figure=plots.prediccion_contexto(historia, pd.Timestamp(fecha), y, iv["inferior"], iv["superior"], real, nivel,
                                                   f"{nombre} · planta {planta} · T = {pd.Timestamp(fecha).date()}"), config=GRAF),
        dcc.Graph(figure=plots.shap_local(sh, "Explicación local (valores de Shapley exactos frente a 30 filas de entrenamiento)"), config=GRAF),
        html.P(f"La variable que más mueve esta predicción es {sh.loc[sh.shap.abs().idxmax(), 'variable']} "
               f"({sh.shap.abs().max():+.4f} de FC en valor absoluto); azul suma, naranja resta respecto de la predicción media del fondo ({sh.base.iloc[0]:.3f}).",
               className="insight"),
    ])


@callback(Output("diag-real-pred", "figure"), Output("diag-hist", "figure"), Output("diag-tiempo", "figure"), Output("diag-acf", "figure"),
          Output("diag-metricas", "children"), Output("diag-txt", "children"), Input("mod-modelo", "value"), Input("diag-plantas", "value"))
def diagnostico(clave, plantas):
    try:
        d = data.predicciones_prueba()
    except DatosNoDisponibles as error:
        v = plots.vacia(str(error))
        return v, v, v, v, "", ""
    d = d[d.modelo == config.MODELOS[clave]]
    if plantas:
        d = d[d.code.isin(plantas)]
    if d.empty:
        v = plots.vacia("Sin datos")
        return v, v, v, v, "", ""
    nombre = config.MODELOS[clave]
    res = d.fc - d.prediccion
    por_planta = d.assign(a=res.abs(), e=res).groupby("code")
    mae, mbe = por_planta.a.mean().mean(), (-por_planta.e.mean()).mean()
    r2 = 1 - (res ** 2).sum() / ((d.fc - d.fc.mean()) ** 2).sum()
    diario = d.assign(r=res).groupby("date").r.mean()
    a = acf(diario.to_numpy(), 20)
    b = banda_95(len(diario))
    textos = (f"{nombre}: MAE macro {mae:.4f}, sesgo medio (predicho − real) {mbe:+.4f}, R² global {r2:.3f} en {len(d):,} días-planta de "
              f"{d.code.nunique()} plantas.")
    lectura = (f"Los residuos tienen asimetría {float(((res - res.mean()) ** 3).mean() / res.std(ddof=0) ** 3):.2f}: los errores grandes son días de "
               f"generación mucho menor que la esperada (indisponibilidad o nubosidad no pronosticada). La autocorrelación a 1 día de la media diaria de "
               f"residuos es {a[1]:.2f} (banda ±{b:.2f}): {'quedan errores comunes a todas las plantas que persisten de un día a otro' if abs(a[1]) > b else 'no queda estructura temporal apreciable'}.")
    return (plots.real_vs_predicho(d, f"Real frente a predicho · {nombre}"),
            plots.histograma(res, f"Histograma de residuos · {nombre}", "Residuo = real − predicho (FC)", nbins=60, color=plots.COLORES["neutro"]),
            plots.residuos_tiempo(d, f"Residuos en el tiempo · {nombre}"),
            _acf_residuos(a, b, nombre), textos, lectura)


def _acf_residuos(a, b, nombre):
    import plotly.graph_objects as go
    f = go.Figure(go.Bar(x=np.arange(1, len(a)), y=a[1:], marker=dict(color=plots.COLORES["serie"]), width=0.4,
                         hovertemplate="rezago %{x}<br>ACF %{y:.3f}<extra></extra>"))
    for s in (b, -b):
        f.add_hline(y=s, line=dict(color=plots.COLORES["acento"], dash="dash", width=1.5))
    f.update_layout(title=dict(text=f"ACF de la media diaria de los residuos · {nombre} (bandas ±1.96/√n)"), height=320,
                    xaxis_title="Rezago (días)", yaxis_title="Autocorrelación", showlegend=False)
    return f


@callback(Output("shap-barras", "figure"), Output("shap-puntos", "figure"), Output("shap-txt", "children"), Input("mod-modelo", "value"))
def shap_global(clave):
    try:
        s = data.shap_global()
    except DatosNoDisponibles as error:
        v = plots.vacia(str(error))
        return v, v, ""
    nombre = config.MODELOS[clave]
    d = s[s.modelo == nombre]
    imp = d.assign(a=d.shap.abs()).groupby("variable").a.mean().sort_values(ascending=False)
    txt = (f"Para {nombre}, las variables con mayor aporte medio son {imp.index[0]} ({imp.iloc[0]:.4f}) y {imp.index[1]} ({imp.iloc[1]:.4f}) en unidades de FC "
           f"(100 filas de la ventana de prueba, fondo de 30 filas de entrenamiento, Shapley exacto validado contra shap.explainers.Exact). "
           + ("En K1 y W1 las variables son el nivel y las anomalías; en los modelos de librería, el historial y el pronóstico en bruto, por lo que la "
              "comparación de importancias es entre representaciones distintas." if clave in ("k1", "w1") else
              "Compare con K1/W1: allí el nivel y la anomalía de radiación concentran el aporte."))
    return (plots.shap_global_barras(d, f"Importancia global · {nombre}"), plots.shap_global_puntos(d, f"Resumen SHAP por fila · {nombre}"), txt)


@callback(Output("mcs-tabla", "data"), Output("mcs-tabla", "columns"), Output("mcs-txt", "children"), Input("mcs-universo", "value"), Input("mcs-perdida", "value"))
def tabla_mcs(universo, perdida):
    m = data.mcs()
    m = m[(m.universo == universo) & (m.perdida == perdida)].sort_values("perdida_media")
    t = m[["modelo", "perdida_media", "p_mcs", "orden_eliminacion", "en_mcs_90"]].copy()
    t["perdida_media"] = t.perdida_media.round(5)
    t["p_mcs"] = t.p_mcs.round(4)
    t["en_mcs_90"] = np.where(t.en_mcs_90, "sí", "no")
    dentro = t[t.en_mcs_90 == "sí"].modelo.tolist()
    txt = (f"El conjunto de confianza de modelos al 90 % (p_MCS ≥ 0.10) contiene {len(dentro)} de {len(t)} modelos: {', '.join(dentro)}. "
           f"Que el modelo de menor pérdida no quede solo en el MCS significa que los datos no permiten declararlo superior a los demás miembros.")
    return t.to_dict("records"), [{"name": c, "id": c} for c in t.columns], txt


@callback(Output("pruebas-tabla", "data"), Output("pruebas-tabla", "columns"), Output("pruebas-txt", "children"), Output("veredicto-r3", "children"),
          Output("curva", "figure"), Output("curva-txt", "children"), Input("pruebas-familia", "value"))
def tabla_pruebas(familia):
    p = data.pruebas()
    f = p[p.familia == familia].copy()
    t = pd.DataFrame({
        "propuesto": f.propuesto, "referencia": f.referencia, "decisiva": np.where(f.decisiva_prefijada, "sí", "no"),
        "MAE propuesto": f.mae_propuesto.round(4), "MAE referencia": f.mae_referencia.round(4), "mejora %": f.mejora_pct.round(2),
        "IC95 dif.": [f"[{a:+.4f}, {b:+.4f}]" for a, b in zip(f.ic95_inf, f.ic95_sup)], "d de Cohen": f.cohen_d.round(3),
        "DM-HLN p Holm": f.dm_hln_p_holm.round(4), "GW p Holm": f.gw_p_holm.round(4), "Wilcoxon p Holm": f.wilcoxon_p_holm.round(4),
        "P(propuesto mejor) bayes": f.bayes_planta_p_propuesto.round(3), "plantas que mejoran": [f"{a}/{b}" for a, b in zip(f.plantas_mejora, f.plantas)],
    })
    tres = (f.dm_hln_p_holm < 0.05) & (f.gw_p_holm < 0.05) & (f.wilcoxon_p_holm < 0.05)
    txt = (f"{int(tres.sum())} de {len(f)} comparaciones de la familia {familia} son significativas al 5 % en las tres pruebas con Holm. "
           f"El tamaño del efecto (d de Cohen sobre la diferencia diaria de pérdida) va de {f.cohen_d.min():.2f} a {f.cohen_d.max():.2f}: efectos pequeños a moderados.")

    # Veredicto R3 (se calcula con las tablas, no se escribe a mano).
    k1rf = p[(p.familia == "K1") & (p.decisiva_prefijada)].iloc[0]
    w1 = p[p.familia == "W1"]
    w1_tres = (w1.dm_hln_p_holm < 0.05) & (w1.gw_p_holm < 0.05) & (w1.wilcoxon_p_holm < 0.05)
    no_w1 = w1[~w1_tres].referencia.tolist()
    m = data.mcs()
    en_mcs = m[(m.universo == "con W1") & (m.perdida == "absoluta") & m.en_mcs_90]
    veredicto = dbc.Alert([
        html.H3("Veredicto sobre R3 (superioridad estadísticamente verificada): NO se cumple", className="h5"),
        html.Ul([
            html.Li(f"Comparación decisiva prefijada, K1 frente a {k1rf.referencia}: mejora {k1rf.mejora_pct:.1f} % del MAE, pero p ajustados DM-HLN "
                    f"{k1rf.dm_hln_p_holm:.3f}, Giacomini-White {k1rf.gw_p_holm:.3f} y Wilcoxon {k1rf.wilcoxon_p_holm:.3f}: ninguna es significativa al 5 %."),
            html.Li(f"W1 tiene el menor MAE de la prueba ({w1.mae_propuesto.iloc[0]:.4f}), pero no supera con las tres pruebas a: {', '.join(no_w1) or 'ninguna'}; "
                    f"y el MCS al 90 % (pérdida absoluta) conserva {len(en_mcs)} modelos, entre ellos referencias de librería, de modo que no es el único superior."),
            html.Li("Las pruebas se fijaron después de conocer los resultados de la ventana de prueba (resultados/pruebas_guia/PROTOCOLO.md: análisis post hoc) y "
                    "W1 se seleccionó con datos de desarrollo y se evaluó en una ventana reutilizada: por la Sección 6.1 de la guía, no pueden sustentar R3."),
            html.Li("Lo que sí se sostiene: K1 y W1 tienen el menor error medio en desarrollo y en prueba, mejoran en la mayoría de plantas y grupos, y la "
                    "ventaja frente a modelos lineales (Ridge, Lasso, BayesianRidge) es significativa; frente a SVR, Random Forest y XGBoost es pequeña y frágil."),
        ], className="mb-0"),
    ], color="warning", className="veredicto mt-3")

    c = data.curva_aprendizaje()
    g = c.groupby(["modelo", "fraccion"]).MAE_D3.mean().unstack()
    txt_c = (f"Con el 10 % de las filas (≈{int(c[c.fraccion == 0.1].filas.mean())}), K1 obtiene {g.loc['K1 Modelo de anomalías', 0.1]:.4f} y Random Forest "
             f"{g.loc['Random Forest', 0.1]:.4f}; con el 100 %, {g.loc['K1 Modelo de anomalías', 1.0]:.4f} y {g.loc['Random Forest', 1.0]:.4f}. K1 tiene seis parámetros: "
             "converge con pocos datos. Exploratoria: hiperparámetros fijos del entrenamiento final, 3 submuestras aleatorias por fracción.")
    return t.to_dict("records"), [{"name": x, "id": x} for x in t.columns], txt, veredicto, plots.curva_aprendizaje(c, "MAE en el bloque D3 según el tamaño del entrenamiento"), txt_c

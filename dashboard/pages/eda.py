"""Sección 2: Análisis exploratorio con filtros globales que actualizan todos los gráficos de la página."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, callback, dcc, html

from src import data, plots
from src.data import DatosNoDisponibles
from src.features import ANOM, DICCIONARIO, PRON
from src.series import acf, banda_95, descomponer, pacf, prueba_adf, prueba_kpss

dash.register_page(__name__, path="/eda", name="EDA", title="EDA · Pronóstico D+1 del FC solar", order=1)

VARIABLES = ["fc", "fc_D", "media7", "media30"] + PRON + ANOM + ["residuo_nivel"]
ETIQUETAS = {v: f"{v} ({DICCIONARIO[v][1]})" for v in VARIABLES if v in DICCIONARIO}
ETIQUETAS["residuo_nivel"] = "residuo_nivel = fc − media30 (adimensional)"
ETIQUETAS["fc"] = "fc: FC observado en T (adimensional)"
MESES = {1: "ene", 2: "feb", 3: "mar", 4: "abr", 5: "may", 6: "jun", 7: "jul", 8: "ago", 9: "sep", 10: "oct", 11: "nov", 12: "dic"}
GRAF = {"displaylogo": False, "responsive": True, "displayModeBar": "hover", "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


# Alto fijo del contenedor de cada gráfico: con responsive=True Plotly ocupa el ancho real de la tarjeta (pantallas pequeñas).
ALTOS = {"eda-bivariado": 480, "eda-corr": 520, "eda-faltantes": 440, "eda-descomp": 620, "eda-acf": 340}


def _grafico(id_, alt):
    return html.Div(dcc.Loading(dcc.Graph(id=id_, config=GRAF, responsive=True, style={"width": "100%", "height": f"{ALTOS.get(id_, 360)}px"}),
                                type="dot", parent_style={"width": "100%", "minWidth": 0}), role="img", **{"aria-label": alt})


def _tarjeta(titulo, id_graf, id_insight, alt, extra=None):
    return dbc.Card([dbc.CardHeader(titulo), dbc.CardBody([extra, _grafico(id_graf, alt), html.P(id=id_insight, className="insight")])])


def filtrar(df: pd.DataFrame, grupos, plantas, inicio, fin, meses, excluir=False) -> pd.DataFrame:
    """Aplica los filtros globales (vacío = todos)."""
    m = pd.Series(True, index=df.index)
    if grupos:
        m &= df["grupo"].isin([int(g) for g in grupos])
    if plantas:
        m &= df["code"].isin(plantas)
    if inicio:
        m &= df["date"] >= pd.Timestamp(inicio)
    if fin:
        m &= df["date"] <= pd.Timestamp(fin)
    if meses:
        m &= df["date"].dt.month.between(int(meses[0]), int(meses[1]))
    if excluir and "dia_anomalo" in df:
        m &= df["dia_anomalo"] == 0
    return df[m]


def layout(**_):
    try:
        p = data.plantas()
        panel = data.panel()
    except DatosNoDisponibles as error:
        return dbc.Alert(str(error), color="danger", className="mt-3")
    fmin, fmax = panel.date.min().date(), panel.date.max().date()
    opciones_grupo = [{"label": f"Grupo {g} ({n} plantas)", "value": int(g)} for g, n in p.groupby("grupo").size().items()]
    opciones_planta = [{"label": f"{r.code} · {r.planta.title()}", "value": r.code} for r in p.itertuples()]
    opciones_var = [{"label": ETIQUETAS[v], "value": v} for v in VARIABLES]

    filtros = dbc.Card(dbc.CardBody(dbc.Row([
        dbc.Col([html.Label("Grupo espacial", htmlFor="eda-grupos"), dcc.Dropdown(id="eda-grupos", options=opciones_grupo, multi=True, placeholder="Todos")], md=3),
        dbc.Col([html.Label("Planta", htmlFor="eda-plantas"), dcc.Dropdown(id="eda-plantas", options=opciones_planta, multi=True, placeholder="Todas")], md=3),
        dbc.Col([html.Label("Rango de fechas"), dcc.DatePickerRange(id="eda-fechas", min_date_allowed=fmin, max_date_allowed=fmax, start_date=fmin, end_date=fmax,
                                                                     display_format="YYYY-MM-DD", first_day_of_week=1)], md=3),
        dbc.Col([html.Label("Meses del año"), dcc.RangeSlider(id="eda-meses", min=1, max=12, step=1, value=[1, 12], marks={k: v for k, v in MESES.items() if k % 2},
                                                               allowCross=False),
                 dbc.Checklist(id="eda-excluir", options=[{"label": "Excluir días con bandera de calidad", "value": 1}], value=[], switch=True,
                               className="small mt-1")], md=3),
    ], className="g-2")), className="filtros")

    return html.Div([
        html.Div([
            html.H1("Dentro de una planta el FC sigue al sol; entre plantas, el nivel base cambia mucho"),
            html.P("Los filtros controlan todas las figuras de la página. El nivel propio de cada planta depende de factores técnicos "
                   "(montaje, seguimiento, pérdidas, disponibilidad), por eso el historial reciente pesa tanto en el pronóstico.",
                   className="lead mb-0"),
        ], className="hero"),
        html.P("Los filtros de esta barra se aplican a todos los gráficos de la página. Cada gráfico tiene una lectura que se recalcula con los "
               "datos filtrados.", className="text-muted"),
        filtros,
        html.Div(id="eda-alerta"),
        dbc.Row(id="eda-kpis", className="g-2 my-2"),

        html.H2("Variable objetivo", className="seccion"),
        dbc.Row([
            dbc.Col(_tarjeta("Distribución del FC diario", "eda-dist", "eda-dist-txt", "Histograma del factor de capacidad diario"), lg=6),
            dbc.Col(_tarjeta("FC medio mensual", "eda-mensual", "eda-mensual-txt", "Serie del FC medio mensual con el rango entre plantas"), lg=6),
        ]),
        _tarjeta("FC por grupo espacial", "eda-grupo", "eda-grupo-txt", "Diagramas de caja del FC por grupo"),

        html.H2("Explorador bivariado", className="seccion"),
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([html.Label("Eje X", htmlFor="eda-x", className="small fw-semibold"), dcc.Dropdown(id="eda-x", options=opciones_var, value="anom_rad", clearable=False)], md=4),
                dbc.Col([html.Label("Eje Y", htmlFor="eda-y", className="small fw-semibold"), dcc.Dropdown(id="eda-y", options=opciones_var, value="residuo_nivel", clearable=False)], md=4),
                dbc.Col([html.Label("Color", htmlFor="eda-color", className="small fw-semibold"),
                         dcc.Dropdown(id="eda-color", options=[{"label": "Sin color", "value": "ninguno"}, {"label": "Periodo (entrenamiento / prueba)", "value": "periodo"}] + opciones_var,
                                      value="periodo", clearable=False)], md=4),
            ], className="g-2"),
            _grafico("eda-bivariado", "Diagrama de dispersión con histogramas marginales de las variables elegidas"),
            html.P(id="eda-bivariado-txt", className="insight"),
        ])),

        html.H2("Correlación, faltantes y atípicos", className="seccion"),
        _tarjeta("Matriz de correlación (tabla D+1)", "eda-corr", "eda-corr-txt", "Mapa de calor de correlaciones entre variables",
                 extra=dbc.RadioItems(id="eda-metodo", options=[{"label": "Pearson", "value": "pearson"}, {"label": "Spearman", "value": "spearman"}],
                                      value="pearson", inline=True, className="small")),
        dbc.Row([
            dbc.Col(_tarjeta("Días sin dato por planta y mes", "eda-faltantes", "eda-faltantes-txt", "Mapa de calor del porcentaje de días sin FC por planta y mes"), lg=7),
            dbc.Col(_tarjeta("Valores atípicos del FC por planta", "eda-atipicos", "eda-atipicos-txt", "Diagramas de caja del FC por planta con atípicos"), lg=5),
        ]),

        html.H2("Serie de tiempo", className="seccion"),
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([html.Label("Planta", htmlFor="eda-planta-serie", className="small fw-semibold"), dcc.Dropdown(id="eda-planta-serie", clearable=False)], md=6),
                dbc.Col([html.Label("Serie", className="small fw-semibold"),
                         dbc.RadioItems(id="eda-serie-var", options=[{"label": "FC diario", "value": "fc"}, {"label": "Residuo de nivel (fc − media30)", "value": "residuo_nivel"}],
                                        value="fc", inline=True)], md=6),
            ], className="g-2"),
            _grafico("eda-descomp", "Descomposición de la serie en tendencia, componente anual y residuo"),
            html.P(id="eda-descomp-txt", className="insight"),
            _grafico("eda-acf", "Funciones de autocorrelación y autocorrelación parcial"),
            html.P(id="eda-acf-txt", className="insight"),
            html.Div(id="eda-estacionariedad"),
        ])),
    ])


@callback(Output("eda-plantas", "options"), Input("eda-grupos", "value"))
def opciones_plantas(grupos):
    p = data.plantas()
    if grupos:
        p = p[p.grupo.isin([int(g) for g in grupos])]
    return [{"label": f"{r.code} · {r.planta.title()}", "value": r.code} for r in p.itertuples()]


def _kpi(valor, etiqueta):
    return dbc.Col(html.Div([html.Div(valor, className="valor"), html.Div(etiqueta, className="etiqueta")], className="kpi"), xs=6, md=3)


@callback(
    Output("eda-alerta", "children"), Output("eda-kpis", "children"),
    Output("eda-dist", "figure"), Output("eda-dist-txt", "children"),
    Output("eda-mensual", "figure"), Output("eda-mensual-txt", "children"),
    Output("eda-grupo", "figure"), Output("eda-grupo-txt", "children"),
    Output("eda-corr", "figure"), Output("eda-corr-txt", "children"),
    Output("eda-faltantes", "figure"), Output("eda-faltantes-txt", "children"),
    Output("eda-atipicos", "figure"), Output("eda-atipicos-txt", "children"),
    Input("eda-grupos", "value"), Input("eda-plantas", "value"), Input("eda-fechas", "start_date"), Input("eda-fechas", "end_date"),
    Input("eda-meses", "value"), Input("eda-excluir", "value"), Input("eda-metodo", "value"),
)
def actualizar_eda(grupos, plantas, inicio, fin, meses, excluir, metodo):
    vacia = plots.vacia("Sin datos para los filtros elegidos")
    try:
        pan = filtrar(data.panel(), grupos, plantas, inicio, fin, meses)
        tab = filtrar(data.tabla_d1(), grupos, plantas, inicio, fin, meses)
    except DatosNoDisponibles as error:
        return (dbc.Alert(str(error), color="danger"), [], vacia, "", vacia, "", vacia, "", vacia, "", vacia, "", vacia, "")
    con_dato = pan[pan.fc.notna()]
    if excluir:
        con_dato = con_dato[con_dato.dia_anomalo == 0]
        tab = tab.merge(pan[["code", "date", "dia_anomalo"]], on=["code", "date"], how="left")
        tab = tab[tab.dia_anomalo.fillna(0) == 0]
    if con_dato.empty:
        alerta = dbc.Alert("Los filtros no dejan ningún día con dato. Amplíe el rango de fechas o la selección de plantas.", color="warning")
        return (alerta, [], vacia, "", vacia, "", vacia, "", vacia, "", vacia, "", vacia, "")

    kpis = [_kpi(f"{con_dato.code.nunique()}", "plantas"), _kpi(f"{len(con_dato):,}".replace(",", " "), "días-planta con FC"),
            _kpi(f"{con_dato.fc.mean():.3f}", "FC medio"), _kpi(f"{100 * pan.fc.isna().mean():.1f} %", "días del calendario sin FC")]

    # Distribución del objetivo.
    fc = con_dato.fc
    asim = float(((fc - fc.mean()) ** 3).mean() / fc.std(ddof=0) ** 3) if len(fc) > 2 else float("nan")
    f_dist = plots.histograma(fc, "Distribución del FC diario", "FC (adimensional)")
    t_dist = (f"El FC medio es {fc.mean():.3f} (mediana {fc.median():.3f}, desviación {fc.std():.3f}); la asimetría es {asim:.2f}: "
              f"{'hay una cola de días de baja generación' if asim < -0.3 else 'la distribución es aproximadamente simétrica' if abs(asim) <= 0.3 else 'hay una cola de días de alta generación'}. "
              f"El {100 * (fc < 0.1).mean():.1f} % de los días tiene FC menor que 0.10.")

    # Serie mensual.
    f_mes = plots.serie_mensual(con_dato, "FC medio mensual de las plantas filtradas")
    mens = con_dato.assign(m=con_dato.date.dt.month).groupby("m").fc.mean()
    t_mes = (f"Por mes del año, el FC medio más alto es el de {MESES[int(mens.idxmax())]} ({mens.max():.3f}) y el más bajo el de "
             f"{MESES[int(mens.idxmin())]} ({mens.min():.3f}); la banda gris muestra cuánto difieren las plantas en un mismo mes.")

    # Por grupo.
    f_g = plots.caja_por(con_dato.sort_values("grupo"), "grupo", "fc", "FC diario por grupo espacial", "Grupo", "FC (adimensional)")
    med = con_dato.groupby("grupo").fc.median()
    t_g = (f"El grupo {med.idxmax()} tiene la mediana más alta ({med.max():.3f}) y el grupo {med.idxmin()} la más baja ({med.min():.3f}): "
           "el nivel del FC depende de la planta y de su ubicación, no solo del clima, lo que motiva usar el nivel reciente de cada planta (K1).") \
        if len(med) > 1 else f"Un solo grupo seleccionado: mediana {med.iloc[0]:.3f}."

    # Correlación.
    t = tab.assign(residuo_nivel=tab.fc - tab.media30)
    if len(t) > 5:
        cm = t[VARIABLES].corr(method=metodo)
        f_c = plots.correlacion(cm, f"Correlación de {metodo.title()} entre variables de la tabla D+1")
        cfc = cm["fc"].drop(["fc", "residuo_nivel"]).abs().sort_values(ascending=False)
        cres = cm["residuo_nivel"].drop(["fc", "residuo_nivel"]).abs().sort_values(ascending=False)
        t_c = (f"Con el FC, la variable más correlacionada es {cfc.index[0]} (|r| = {cfc.iloc[0]:.2f}); con el residuo de nivel (fc − media30) es "
               f"{cres.index[0]} (|r| = {cres.iloc[0]:.2f}). Las anomalías explican la desviación diaria respecto del nivel, que es lo que K1 modela.")
    else:
        f_c, t_c = vacia, "Muy pocas filas D+1 para estimar correlaciones."

    # Faltantes.
    falt = pan.assign(mes=pan.date.dt.to_period("M").dt.to_timestamp(), sin=pan.fc.isna() * 100.0).pivot_table(index="code", columns="mes", values="sin", aggfunc="mean")
    f_f = plots.faltantes(falt, "Porcentaje de días sin FC por planta y mes")
    sin_pron = 100 * pan.loc[pan.fc.notna(), "rad_pron"].isna().mean()
    peor = pan.groupby("code").fc.apply(lambda s: s.isna().mean() * 100).sort_values(ascending=False)
    t_f = (f"El {100 * pan.fc.isna().mean():.1f} % de los días del calendario no tiene FC (XM no reporta días sin generación) y el {sin_pron:.1f} % de los días "
           f"con FC no tiene pronóstico completo. La planta con más huecos es {peor.index[0]} ({peor.iloc[0]:.1f} %). Las filas incompletas no entran a la tabla D+1.")

    # Atípicos.
    def n_atip(s):
        q1, q3 = s.quantile([0.25, 0.75])
        return int(((s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))).sum())
    atip = con_dato.groupby("code").fc.apply(n_atip).sort_values(ascending=False)
    f_a = plots.caja_por(con_dato, "code", "fc", "FC diario por planta (atípicos: regla 1.5 × RIC)", "Planta", "FC (adimensional)")
    marcados = pan.loc[pan.fc.notna(), "dia_anomalo"].mean() * 100
    t_a = (f"Hay {int(atip.sum())} días atípicos por la regla 1.5 × RIC dentro de cada planta ({100 * atip.sum() / len(con_dato):.1f} %), sobre todo en {atip.index[0]} ({atip.iloc[0]}). "
           f"El {marcados:.1f} % de los días con FC tiene bandera de calidad (pocas horas o rendimiento bajo); no se eliminan del modelado, solo se usan en sensibilidades.")

    return (None, kpis, f_dist, t_dist, f_mes, t_mes, f_g, t_g, f_c, t_c, f_f, t_f, f_a, t_a)


@callback(Output("eda-bivariado", "figure"), Output("eda-bivariado-txt", "children"),
          Input("eda-grupos", "value"), Input("eda-plantas", "value"), Input("eda-fechas", "start_date"), Input("eda-fechas", "end_date"),
          Input("eda-meses", "value"), Input("eda-x", "value"), Input("eda-y", "value"), Input("eda-color", "value"))
def actualizar_bivariado(grupos, plantas, inicio, fin, meses, x, y, color):
    try:
        t = filtrar(data.tabla_d1(), grupos, plantas, inicio, fin, meses)
    except DatosNoDisponibles as error:
        return plots.vacia(str(error)), ""
    t = t.assign(residuo_nivel=t.fc - t.media30)
    if len(t) < 3:
        return plots.vacia("Sin filas D+1 para los filtros elegidos"), "Amplíe los filtros."
    f = plots.dispersion(t, x, y, color, ETIQUETAS, f"{y} frente a {x}")
    r = t[[x, y]].corr().iloc[0, 1]
    rs = t[[x, y]].corr(method="spearman").iloc[0, 1]
    fuerza = "fuerte" if abs(r) >= 0.6 else "moderada" if abs(r) >= 0.3 else "débil"
    return f, (f"Relación {fuerza} y {'positiva' if r > 0 else 'negativa'}: Pearson r = {r:.2f}, Spearman ρ = {rs:.2f} en {len(t):,} filas"
               f"{' (se dibuja una muestra de 4 000)' if len(t) > 4000 else ''}. Una diferencia grande entre r y ρ indica no linealidad o atípicos.")


@callback(Output("eda-planta-serie", "options"), Output("eda-planta-serie", "value"),
          Input("eda-grupos", "value"), Input("eda-plantas", "value"), Input("eda-planta-serie", "value"))
def opciones_serie(grupos, plantas, actual):
    p = data.plantas()
    if grupos:
        p = p[p.grupo.isin([int(g) for g in grupos])]
    if plantas:
        p = p[p.code.isin(plantas)]
    if p.empty:
        p = data.plantas()
    opciones = [{"label": f"{r.code} · {r.planta.title()}", "value": r.code} for r in p.itertuples()]
    valor = actual if actual in set(p.code) else p.code.iloc[0]
    return opciones, valor


@callback(Output("eda-descomp", "figure"), Output("eda-descomp-txt", "children"), Output("eda-acf", "figure"), Output("eda-acf-txt", "children"),
          Output("eda-estacionariedad", "children"),
          Input("eda-planta-serie", "value"), Input("eda-serie-var", "value"), Input("eda-fechas", "start_date"), Input("eda-fechas", "end_date"))
def actualizar_serie(planta, variable, inicio, fin):
    if not planta:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    pan = filtrar(data.panel(), None, [planta], inicio, fin, None)
    s = pan.set_index("date").fc.sort_index()
    if variable == "residuo_nivel":
        nivel = s.asfreq("D").rolling(30, min_periods=15).mean().shift(1)
        s = (s.asfreq("D") - nivel)
    s = s.asfreq("D") if len(s) else s
    if s.notna().sum() < 60:
        v = plots.vacia("Se necesitan al menos 60 días con dato")
        return v, "Serie demasiado corta para descomponer.", v, "", None
    nombre = "FC diario" if variable == "fc" else "residuo de nivel"
    dec = descomponer(s)
    f_d = plots.descomposicion(dec, f"Descomposición aditiva del {nombre} · planta {planta}")
    var_tot = dec.observado.var()
    t_d = (f"La tendencia (media móvil de 30 días) explica {100 * dec.tendencia.var() / var_tot:.0f} % de la varianza del {nombre} y el componente anual por mes "
           f"{100 * dec.anual.var() / var_tot:.0f} %; el resto queda en el residuo diario. El nivel cambia lentamente, por eso la media de 30 días es un buen ancla.")
    x = s.to_numpy()
    a, pa = acf(x, 30), pacf(x, 30)
    b = banda_95(int(np.isfinite(x).sum()))
    f_a = plots.acf_pacf(a, pa, b, f"Autocorrelación del {nombre} · planta {planta} (bandas ±1.96/√n)")
    fuera = int((np.abs(a[1:]) > b).sum())
    t_a = (f"La autocorrelación a 1 día es {a[1]:.2f} y {fuera} de 30 rezagos salen de la banda del 95 %; la PACF a 1 día es {pa[1]:.2f}. "
           f"{'Hay memoria de varios días: usar el historial reciente es útil.' if a[1] > 0.3 else 'La memoria es corta.'} "
           "(La ACF ignora los huecos del calendario al centrar; los rezagos cruzan días sin dato).")
    adf, kpss = prueba_adf(x), prueba_kpss(x)
    tabla = dbc.Table([
        html.Thead(html.Tr([html.Th("Prueba"), html.Th("H0"), html.Th("Estadístico"), html.Th("Valor crítico 5 %"), html.Th("Rezagos"), html.Th("Decisión al 5 %")])),
        html.Tbody([
            html.Tr([html.Td("Dickey-Fuller aumentada (con constante)"), html.Td("raíz unitaria"), html.Td(f"{adf['estadistico']:.2f}"), html.Td("−2.86"),
                     html.Td(adf["rezagos"]), html.Td("se rechaza: estacionaria" if adf["rechaza_raiz_unitaria_5"] else "no se rechaza")]),
            html.Tr([html.Td("KPSS (nivel)"), html.Td("estacionaria en nivel"), html.Td(f"{kpss['estadistico']:.3f}"), html.Td("0.463"),
                     html.Td(kpss["rezagos"]), html.Td("se rechaza: no estacionaria" if kpss["rechaza_estacionariedad_5"] else "no se rechaza")]),
        ]),
    ], bordered=False, size="sm", className="mt-2")
    concl = ("Ambas pruebas apuntan a estacionariedad." if adf["rechaza_raiz_unitaria_5"] and not kpss["rechaza_estacionariedad_5"] else
             "ADF rechaza la raíz unitaria pero KPSS rechaza la estacionariedad en nivel: la serie es estacionaria alrededor de un nivel que cambia lentamente, "
             "coherente con modelar la desviación respecto del nivel reciente." if adf["rechaza_raiz_unitaria_5"] else
             "No se rechaza la raíz unitaria: la serie tiene cambios de nivel persistentes.")
    return f_d, t_d, f_a, t_a, html.Div([html.H3("Pruebas de estacionariedad", className="h6 mt-3"), tabla,
                                          html.P(concl + " Huecos internos interpolados linealmente solo para las pruebas.", className="insight")])

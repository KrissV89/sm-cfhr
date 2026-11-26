"""Generate all 9 real-data EDA figures for the SM-CFHR manuscript (Spanish labels)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 12.0,
    "axes.titlesize": 13.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 12.0,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.0,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUT_DIR = Path(__file__).resolve().parents[1] / "figures"
DEFAULT_BASE = Path(__file__).resolve().parents[3] / "base_generacion_clima" / "base_planta_dia.csv"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, l1, p2, l2 = np.radians([lat1, lon1, lat2, lon2])
    dphi = p2 - p1
    dlam = l2 - l1
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2.0) ** 2
    return float(2.0 * r * np.arcsin(np.sqrt(a)))


def make_cohorte_y_mapa(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    stages = [
        "Registros brutos recurso-día (XM)",
        "Con capacidad neta y energía > 0",
        "Coordenadas verificadas y FC <= 1,05",
        "Post-arranque (>=60 d) y >=100 d útiles",
    ]
    counts = [30964, 19685, 13651, 12756]
    hatches = ["///", "...", "\\\\", ""]
    colors = ["#CBD5E1", "#93C5FD", "#3B82F6", "#1E3A8A"]
    y = np.arange(len(stages))
    bars = ax.barh(y, counts, color=colors, edgecolor="#0F172A", lw=0.9)
    for bar, h in zip(bars, hatches):
        if h:
            bar.set_hatch(h)
    ax.set_yticks(y)
    ax.set_yticklabels(stages)
    ax.invert_yaxis()
    ax.set_xlabel("Registros planta-día")
    ax.set_title("Depuración secuencial de la cohorte (17 plantas FV)")
    ax.set_xlim(0, 37500)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    for bar, val in zip(bars, counts):
        ax.text(val + 550, bar.get_y() + bar.get_height() / 2, f"{val:,}", va="center", fontsize=11.0, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_flujo_cohorte.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    pl = df.groupby(["code", "grupo"]).agg(lat=("lat", "first"), lon=("lon", "first"), cap=("cap_MW", "median")).reset_index()
    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for grp, sub in pl.groupby("grupo"):
        g_idx = int(grp) - 1
        ax.scatter(
            sub["lon"], sub["lat"],
            s=65 + np.sqrt(sub["cap"]) * 12.0,
            color=cmap(g_idx % 10),
            marker=markers[g_idx % len(markers)],
            alpha=0.9, edgecolors="#0F172A", linewidth=0.9,
            label=f"Zona {int(grp)} ({len(sub)} plantas)",
        )
    ax.set_xlabel("Longitud (grados)")
    ax.set_ylabel("Latitud (grados)")
    ax.set_title("Distribución espacial de las 17 plantas (G = 7 zonas, enlace <100 km)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower left", ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_mapa_plantas.pdf", bbox_inches="tight")
    plt.close(fig)


def make_cobertura_y_horas(df: pd.DataFrame) -> None:
    df_v = df[df["horas_xm"] > 6].copy()

    # Use filtered cohort (12,725 records; 251-1,032 valid days) to match text
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    order = df_v.groupby("code")["date"].count().sort_values(ascending=True).index.tolist()
    for idx, code in enumerate(order):
        sub = df_v[df_v["code"] == code].sort_values("date")
        ax.scatter(sub["date"], np.full(len(sub), idx), s=5.0, color="#1D4ED8", marker="s")
        ax.text(pd.Timestamp("2026-11-12"), idx, f"{len(sub)} d", va="center", fontsize=9.8, color="#0F172A")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order, fontsize=10.0)
    ax.set_xlim(pd.Timestamp("2023-12-20"), pd.Timestamp("2027-03-10"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=20)
    ax.set_title("Cobertura temporal válida (>6 h) por planta (n = 12,725; 251–1,032 d)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Código de planta")
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_cobertura_temporal.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.2))
    hrs = df["horas_xm"].dropna().to_numpy()
    bins = np.arange(0.5, 25.5, 1.0)
    counts, edges = np.histogram(hrs, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bar_colors = ["#DC2626" if c <= 6 else "#2563EB" for c in centers]
    bars = ax1.bar(centers, counts, width=0.82, color=bar_colors, edgecolor="#0F172A", lw=0.7)
    for bar, c in zip(bars, centers):
        if c <= 6:
            bar.set_hatch("///")
    ax1.axvline(6.5, color="#0F172A", ls="--", lw=1.4, label="Umbral: >6 h reportadas")
    ax1.set_yscale("log")
    ax1.set_xlabel("Intervalos horarios reportados por día (h)")
    ax1.set_ylabel("Planta-días (escala log)")
    ax1.set_title("(a) Horas reportadas por planta-día")
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(axis="y", linestyle=":", alpha=0.45)

    incomp = df[df["horas_xm"] <= 6]["fc"].dropna().to_numpy()
    comp = df[df["horas_xm"] > 6]["fc"].dropna().to_numpy()
    ax2.boxplot([incomp, comp], tick_labels=[f"<= 6 h\n(n={len(incomp)})", f"> 6 h\n(n={len(comp)})"],
                widths=0.45, patch_artist=True,
                boxprops=dict(facecolor="#DBEAFE", color="#1E3A8A", lw=1.2),
                medianprops=dict(color="#DC2626", lw=1.8),
                whiskerprops=dict(color="#475569", lw=1.1),
                capprops=dict(color="#475569", lw=1.1))
    ax2.set_ylabel("Factor de capacidad observado (FC)")
    ax2.set_title("(b) Factor de capacidad según completitud")
    ax2.grid(axis="y", linestyle=":", alpha=0.45)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_horas_y_anomalos.pdf", bbox_inches="tight")
    plt.close(fig)


def make_distribucion_y_rendimiento(df: pd.DataFrame) -> None:
    df_v = df[df["horas_xm"] > 6].copy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.6, 7.6), gridspec_kw={"height_ratios": [1.0, 1.35]})
    fc = df_v["fc"].dropna().to_numpy()
    med = float(np.median(fc))
    q25, q75 = np.percentile(fc, [25, 75])
    ax1.hist(fc, bins=45, color="#2563EB", alpha=0.8, edgecolor="white")
    ax1.axvspan(q25, q75, color="#93C5FD", alpha=0.4, label=f"RIC: {q25:.3f}–{q75:.3f}")
    ax1.axvline(med, color="#0F172A", lw=1.8, ls="--", label=f"Mediana: {med:.3f}")
    ax1.set_title("(a) Distribución global del factor de capacidad diario")
    ax1.set_xlabel("Factor de capacidad diario (FC)")
    ax1.set_ylabel("Planta-días")
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.45)

    order = df_v.groupby("code")["fc"].median().sort_values().index.tolist()
    data = [df_v.loc[df_v["code"] == c, "fc"].dropna().to_numpy() for c in order]
    ax2.boxplot(data, vert=False, tick_labels=order, showfliers=False,
                boxprops=dict(color="#1E40AF", lw=1.2), medianprops=dict(color="#DC2626", lw=1.6),
                whiskerprops=dict(color="#475569", lw=1.0), capprops=dict(color="#475569", lw=1.0))
    ax2.axvline(med, color="#0F172A", lw=1.4, ls="--", alpha=0.8)
    ax2.set_title("(b) Factor de capacidad diario por planta")
    ax2.set_xlabel("Factor de capacidad diario (FC)")
    ax2.set_ylabel("Código de planta")
    ax2.tick_params(axis="y", labelsize=9.5)
    ax2.grid(axis="x", linestyle=":", alpha=0.45)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_distribucion_fc.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    sub = df_v[["gti10", "fc"]].dropna().copy()
    sub["sy"] = np.where(sub["gti10"] > 0.2, sub["fc"] * 24.0 / sub["gti10"], np.nan)
    anom = sub[sub["sy"] < 0.3]
    norm = sub[sub["sy"] >= 0.3]
    hb = ax.hexbin(norm["gti10"], norm["fc"], gridsize=34, cmap="Blues", mincnt=1, bins="log")
    fig.colorbar(hb, ax=ax, label="Planta-días (log)")
    ax.scatter(anom["gti10"], anom["fc"], color="#DC2626", marker="x", s=34, lw=1.4,
               label=f"Bajo rendimiento < 0,3 ({len(anom)} días; >6 h)", zorder=4)
    xg = np.linspace(0.2, float(sub["gti10"].max()), 100)
    ax.plot(xg, np.minimum(1.0, 1.5 * xg / 24.0), "k--", lw=1.5, label="Razón = 1,5")
    ax.plot(xg, 0.3 * xg / 24.0, color="#DC2626", ls=":", lw=1.6, label="Umbral = 0,3")
    ax.set_title("Rendimiento específico vs. irradiación inclinada (10°)")
    ax.set_xlabel(r"Irradiación plano inclinado 10° ($\mathrm{kWh/m^2/d\acute{\imath}a}$)")
    ax.set_ylabel("Factor de capacidad diario (FC)")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.45)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_rendimiento_especifico.pdf", bbox_inches="tight")
    plt.close(fig)


def make_estacionalidad_y_dependencia(df: pd.DataFrame) -> None:
    df_v = df[df["horas_xm"] > 6].copy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.2, 6.4), gridspec_kw={"height_ratios": [1.0, 1.45]}, sharex=True)
    df_m = df_v.copy()
    df_m["ym"] = pd.to_datetime(df_m["date"]).dt.to_period("M").dt.to_timestamp()
    piv = df_m.pivot_table(index="code", columns="ym", values="fc", aggfunc="mean")
    order = df_v.groupby("code")["fc"].median().sort_values(ascending=False).index
    piv = piv.loc[order]
    months = piv.columns
    med_m = piv.median(axis=0)
    q25_m = piv.quantile(0.25, axis=0)
    q75_m = piv.quantile(0.75, axis=0)

    ax1.fill_between(months, q25_m, q75_m, color="#93C5FD", alpha=0.45, label="RIC entre plantas")
    ax1.plot(months, med_m, "o-", color="#1D4ED8", lw=1.8, ms=4.5, label="Mediana entre plantas")
    ax1.set_title("(a) Evolución mensual del factor de capacidad")
    ax1.set_ylabel("FC mensual")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.45)

    im = ax2.imshow(piv.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0.14, vmax=0.35,
                    extent=[mdates.date2num(months[0]), mdates.date2num(months[-1]), len(piv) - 0.5, -0.5])
    ax2.set_yticks(np.arange(len(piv)))
    ax2.set_yticklabels(piv.index, fontsize=9.2)
    ax2.xaxis_date()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.tick_params(axis="x", rotation=20)
    ax2.set_title("(b) Factor de capacidad medio mensual por planta")
    ax2.set_xlabel("Mes")
    fig.colorbar(im, ax=ax2, orientation="vertical", fraction=0.03, pad=0.02, label="FC medio")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_estacionalidad.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.8, 7.2))
    lags = np.arange(1, 31)
    acfs = []
    for _, grp in df_v.groupby("code"):
        s = grp.sort_values("date").set_index("date")["fc"].asfreq("D")
        vals = [s.autocorr(lag=int(k)) for k in lags]
        acfs.append(vals)
        ax1.plot(lags, vals, color="#94A3B8", lw=1.0, alpha=0.65)
    med_acf = np.nanmedian(np.array(acfs), axis=0)
    ax1.plot(lags, med_acf, color="#1D4ED8", lw=2.4, label="Mediana en 17 plantas")
    ax1.plot([], [], color="#94A3B8", lw=1.2, label="Plantas individuales (17)")
    ax1.axhline(0.0, color="#0F172A", lw=1.0, ls="--")
    ax1.set_title("(a) Autocorrelación temporal del FC diario")
    ax1.set_xlabel("Rezago (días)")
    ax1.set_ylabel("Autocorrelación")
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.45)

    piv_c = df_v.pivot(index="date", columns="code", values="fc")
    corr_mat = piv_c.corr()
    meta = df_v.groupby("code").agg(lat=("lat", "first"), lon=("lon", "first"), grp=("grupo", "first")).to_dict("index")
    codes = sorted(meta.keys())
    same_d, same_c, diff_d, diff_c = [], [], [], []
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            c1, c2 = codes[i], codes[j]
            dist = _haversine_km(meta[c1]["lat"], meta[c1]["lon"], meta[c2]["lat"], meta[c2]["lon"])
            val = float(corr_mat.loc[c1, c2])
            if meta[c1]["grp"] == meta[c2]["grp"]:
                same_d.append(max(dist, 0.8))
                same_c.append(val)
            else:
                diff_d.append(max(dist, 0.8))
                diff_c.append(val)
    ax2.scatter(diff_d, diff_c, color="#DC2626", marker="^", s=42, alpha=0.8, label=f"Zonas distintas ({len(diff_c)} pares)")
    ax2.scatter(same_d, same_c, color="#2563EB", marker="o", s=44, alpha=0.9, label=f"Misma zona espacial ({len(same_c)} pares)")
    ax2.axvline(100.0, color="#0F172A", lw=1.2, ls=":", label="Umbral agrupamiento 100 km")
    ax2.set_xscale("log")
    ax2.set_title("(b) Correlación entre plantas vs. distancia geográfica")
    ax2.set_xlabel("Distancia entre plantas (km, escala log)")
    ax2.set_ylabel("Correlación pareada de FC")
    ax2.legend(frameon=True)
    ax2.grid(True, which="both", linestyle=":", alpha=0.45)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_dependencia.pdf", bbox_inches="tight")
    plt.close(fig)


def make_clima(df: pd.DataFrame) -> None:
    df_v = df[df["horas_xm"] > 6].copy()

    fig, axes = plt.subplots(3, 2, figsize=(10.4, 9.6))
    specs = [
        ("rad_bm", r"Irradiación ($\mathrm{kWh/m^2/d\acute{\imath}a}$)", "(a) Irradiación horizontal"),
        ("temp_bm", "Temperatura a 2 m (°C)", "(b) Temperatura del aire a 2 m"),
        ("nube_bm", "Nubosidad diurna (%)", "(c) Nubosidad diurna"),
        ("humedad_bm", "Humedad relativa (%)", "(d) Humedad relativa"),
        ("viento_bm", "Velocidad del viento a 10 m (m/s)", "(e) Viento a 10 m"),
    ]
    for ax, (col, xlabel, title) in zip(axes.flat[:5], specs):
        vals = df_v[col].dropna().to_numpy()
        ax.hist(vals, bins=32, color="#2563EB", alpha=0.8, edgecolor="white")
        med = float(np.median(vals))
        ax.axvline(med, color="#0F172A", lw=1.6, ls="--", label=f"Mediana: {med:.2f}")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Planta-días")
        ax.legend(frameon=True)
        ax.grid(True, linestyle=":", alpha=0.4)

    ax6 = axes.flat[5]
    sub_r = df_v[["rad_e5", "rad_bm"]].dropna()
    hb = ax6.hexbin(sub_r["rad_e5"], sub_r["rad_bm"], gridsize=30, cmap="Blues", mincnt=1, bins="log")
    fig.colorbar(hb, ax=ax6, label="conteo log")
    ax6.plot([0.5, 8.0], [0.5, 8.0], "k--", lw=1.4, label="Identidad 1:1")
    ax6.set_title("(f) Irradiación Best-match vs. ERA5")
    ax6.set_xlabel(r"ERA5 ($\mathrm{kWh/m^2/d\acute{\imath}a}$)")
    ax6.set_ylabel(r"Best-match ($\mathrm{kWh/m^2/d\acute{\imath}a}$)")
    ax6.legend(frameon=True)
    ax6.grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_clima.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera las 9 figuras EDA en español para SM-CFHR.")
    parser.add_argument("--base-csv", type=Path, default=DEFAULT_BASE)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.base_csv, parse_dates=["date"])
    make_cohorte_y_mapa(df)
    make_cobertura_y_horas(df)
    make_distribucion_y_rendimiento(df)
    make_estacionalidad_y_dependencia(df)
    make_clima(df)
    print("Generated 9 Spanish EDA PDFs in", OUT_DIR)


if __name__ == "__main__":
    main()

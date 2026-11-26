"""Generate all 8 Phase 5 Spanish vector PDF result figures from verified v1/v2/v3 CSVs."""

from __future__ import annotations

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
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.0,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PAPER_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = PAPER_DIR.parents[1]
OUT_DIR = PAPER_DIR / "figures"
EQ_DIR = ROOT_DIR / "resultados" / "experimento_equivalencia_v3"
AB_DIR = ROOT_DIR / "resultados" / "ablaciones_controladas_v3"
OOF_DIR = ROOT_DIR / "resultados" / "evaluacion_ashr_oof"
V2_DIR = ROOT_DIR / "resultados" / "inferencia_revision_independiente_v2"

# Colorblind-safe palette (Okabe-Ito inspired)
C_OBS = "#0F172A"
C_SM = "#0072B2"
C_BAYES = "#009E73"
C_ELASTIC = "#CC79A7"
C_RAW = "#D55E00"
C_MUTE = "#64748B"


def fig01_real_vs_pred_d1() -> None:
    df = pd.read_csv(EQ_DIR / "predicciones_equivalencia" / "D1_espaciotemporal_cinco_variables_equiv_A.csv")
    df["date"] = pd.to_datetime(df["date"])
    plants = [("GYPO", "Guayepo (370-388 MW)"), ("3IRX", "Portón del Sol (102 MW)"), ("3EDL", "Bosques Llanos V (17,9 MW)")]
    models = [
        ("SM-CFHR__CF", "SM-CFHR (FC)", C_SM, "-"),
        ("BayesianRidge__CF", "BayesianRidge (FC)", C_BAYES, "--"),
        ("HuberRegressor__kWh_bruto", "HuberRegressor (kWh bruto)", C_RAW, ":"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 7.6), sharex=False)
    for ax, (code, label) in zip(axes, plants):
        sub = df[df["code"] == code].copy()
        obs = sub[sub["metodo"] == "SM-CFHR__CF"].sort_values("date").tail(90)
        ax.plot(obs["date"], obs["y_kwh"] / 1000.0, color=C_OBS, lw=1.4, label="Observado")
        for m, m_lbl, col, ls in models:
            ms = sub[sub["metodo"] == m].sort_values("date").tail(90)
            ax.plot(ms["date"], ms["prediccion_kwh"] / 1000.0, color=col, ls=ls, lw=1.3, label=m_lbl)
        ax.set_title(f"Central {code}: {label}", loc="left", fontsize=10.5)
        ax.set_ylabel("MWh/día")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.grid(alpha=0.22, ls=":")
    axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.28), frameon=False)
    axes[-1].set_xlabel("Fecha de evaluación (últimos 90 días)")
    fig.suptitle("Pronóstico diario D+1 en transferencia espaciotemporal con meteorología", y=0.99, fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_res_01_real_vs_pred_d1.pdf", bbox_inches="tight")
    plt.close(fig)


def fig02_real_vs_pred_recursivo() -> None:
    df28 = pd.read_csv(EQ_DIR / "predicciones_equivalencia" / "RecGEFSv12_espaciotemporal_28d_equiv_A.csv")
    df7 = pd.read_csv(EQ_DIR / "predicciones_equivalencia" / "RecGEFSv12_espaciotemporal_7d_equiv_A.csv")
    cap_df = pd.read_csv(EQ_DIR / "resultados_por_planta_equivalencia.csv")[["code", "cap_MW"]].drop_duplicates()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    # Left: trajectory for 3IRX on a representative origin in 28d
    sub_p = df28[df28["code"] == "3IRX"].copy()
    origins = sorted(sub_p["origen"].unique())
    orig = origins[len(origins) // 2]
    tr = sub_p[sub_p["origen"] == orig]
    obs = tr[tr["metodo"] == "SM-CFHR__CF"].sort_values("lead_dias")
    ax1.plot(obs["lead_dias"], obs["y_kwh"] / 1000.0, color=C_OBS, lw=1.8, marker="o", ms=3, label="Observado")
    for m, lbl, col, ls in [
        ("SM-CFHR__CF", "SM-CFHR (FC)", C_SM, "-"),
        ("ElasticNet__CF", "ElasticNet (FC)", C_ELASTIC, "--"),
        ("ElasticNet__kWh_bruto", "ElasticNet (kWh bruto)", C_RAW, ":"),
    ]:
        ms = tr[tr["metodo"] == m].sort_values("lead_dias")
        ax1.plot(ms["lead_dias"], ms["prediccion_kwh"] / 1000.0, color=col, ls=ls, lw=1.4, label=lbl)
    ax1.set_title(f"Trayectoria 28d (3IRX, origen {orig})")
    ax1.set_xlabel("Horizonte de paso h (días)")
    ax1.set_ylabel("Generación diaria (MWh/día)")
    ax1.legend(frameon=False, fontsize=8.5)
    ax1.grid(alpha=0.22, ls=":")

    # Right: MAE_CF by lead day h in 28d ET + GEFSv12
    m28 = df28.merge(cap_df, on="code", how="left")
    m28["ae_cf"] = np.abs(m28["prediccion_kwh"] - m28["y_kwh"]) / (24000.0 * m28["cap_MW"])
    for m, lbl, col, ls in [
        ("SM-CFHR__CF", "SM-CFHR (FC, 28d)", C_SM, "-"),
        ("ElasticNet__CF", "ElasticNet (FC, 28d)", C_ELASTIC, "--"),
        ("BayesianRidge__CF", "BayesianRidge (FC, 28d)", C_BAYES, "-."),
        ("ElasticNet__kWh_bruto", "ElasticNet (kWh bruto, 28d)", C_RAW, ":"),
    ]:
        g = m28[m28["metodo"] == m].groupby("lead_dias", as_index=False)["ae_cf"].mean()
        ax2.plot(g["lead_dias"], g["ae_cf"], color=col, ls=ls, lw=1.5, label=lbl)
    # Also mark 7d mean for reference
    m7 = df7.merge(cap_df, on="code", how="left")
    m7_sm = m7[m7["metodo"] == "SM-CFHR__CF"]
    mae7 = (np.abs(m7_sm["prediccion_kwh"] - m7_sm["y_kwh"]) / (24000.0 * m7_sm["cap_MW"])).mean()
    ax2.axhline(mae7, color=C_MUTE, ls="--", lw=1.0, label=f"Media SM-CFHR 7d ({mae7:.3f})")
    ax2.set_title("Degradación del error por paso h (ET + GEFSv12)")
    ax2.set_xlabel("Horizonte h (días)")
    ax2.set_ylabel("MAE medio en factor de capacidad")
    ax2.legend(frameon=False, fontsize=8.2)
    ax2.grid(alpha=0.22, ls=":")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_res_02_real_vs_pred_recursivo.pdf", bbox_inches="tight")
    plt.close(fig)


def fig03_real_vs_pred_directo() -> None:
    df_et = pd.read_csv(EQ_DIR / "predicciones_equivalencia" / "DirectoSinClima_espaciotemporal_28d_equiv_A.csv")
    df_ip = pd.read_csv(EQ_DIR / "predicciones_equivalencia" / "DirectoSinClima_intraplanta_28d_equiv_A.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.1))
    for ax, df, title in [
        (ax1, df_et, "Directo 28d ET sin clima (acumulado)"),
        (ax2, df_ip, "Directo 28d IP sin clima (acumulado)"),
    ]:
        for m, lbl, col, mk in [
            ("SM-CFHR__CF", "SM-CFHR (FC)", C_SM, "o"),
            ("ElasticNet__CF", "ElasticNet (FC)", C_ELASTIC, "s"),
            ("XGBoost__kWh_bruto", "XGBoost (kWh bruto)", C_RAW, "^"),
        ]:
            sub = df[df["metodo"] == m]
            ax.scatter(sub["y_total_kwh"] / 1e6, sub["prediccion_total_kwh"] / 1e6,
                       s=18, alpha=0.55, color=col, marker=mk, label=lbl, edgecolors="none")
        lim = max(df["y_total_kwh"].max(), df["prediccion_total_kwh"].max()) / 1e6 * 1.05
        ax.plot([0, lim], [0, lim], color=C_OBS, lw=1.0, ls="--", label="Identidad y = x")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(title)
        ax.set_xlabel("Energía acumulada observada (GWh / 28d)")
        ax.set_ylabel("Energía acumulada predicha (GWh / 28d)")
        ax.legend(frameon=False, fontsize=8.3)
        ax.grid(alpha=0.22, ls=":")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_res_03_real_vs_pred_directo.pdf", bbox_inches="tight")
    plt.close(fig)


def fig04_equivalencia_representacion() -> None:
    df_a = pd.read_csv(EQ_DIR / "resumen_equivalencia_condicion_A.csv")
    df_b = pd.read_csv(EQ_DIR / "resumen_equivalencia_condicion_B.csv")
    models = ["BayesianRidge", "ElasticNet", "HuberRegressor", "Ridge", "RandomForest", "XGBoost", "SM-CFHR"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.3))
    # Panel A: Mean MAE_CF_macro across 10 Cond A scenarios: kWh_bruto vs CF
    g_a = df_a.groupby(["modelo_base", "representacion"], as_index=False)["MAE_CF_macro"].mean()
    y = np.arange(len(models))
    h = 0.36
    raw_vals = [float(g_a[(g_a["modelo_base"] == m) & (g_a["representacion"] == "kWh_bruto")]["MAE_CF_macro"].iloc[0]) for m in models]
    cf_vals = [float(g_a[(g_a["modelo_base"] == m) & (g_a["representacion"] == "CF")]["MAE_CF_macro"].iloc[0]) for m in models]

    ax1.barh(y + h / 2, raw_vals, height=h, color=C_RAW, alpha=0.85, label="Entrada/objetivo kWh bruto")
    ax1.barh(y - h / 2, cf_vals, height=h, color=C_SM, alpha=0.90, label="Representación causal FC")
    for idx, (rv, cv) in enumerate(zip(raw_vals, cf_vals)):
        pct = 100.0 * (cv - rv) / rv
        ax1.text(max(rv, cv) + 0.001, idx, f"{pct:+.1f}%", va="center", fontsize=8.3, color=C_OBS)
    ax1.set_yticks(y)
    ax1.set_yticklabels(models)
    ax1.set_xlim(0, max(raw_vals) * 1.24)
    ax1.set_xlabel("MAE medio en factor de capacidad (10 escenarios)")
    ax1.set_title("Condición A: efecto de normalizar por FC")
    ax1.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax1.grid(axis="x", alpha=0.22, ls=":")

    # Panel B: Cond B (3-fold causal CV) Delta % of SM-CFHR__CF vs best alternative CF baseline
    sub_b = df_b[df_b["metodo"] == "SM-CFHR__CF"].copy()
    short = {
        "D1_espaciotemporal_cinco_variables": "D1 ET 5var",
        "D1_intraplanta_cinco_variables": "D1 IP 5var",
        "D1_espaciotemporal_sin_meteorologia": "D1 ET s/clima",
        "D1_intraplanta_sin_meteorologia": "D1 IP s/clima",
        "RecGEFSv12_espaciotemporal_28d": "Rec28d ET+GEFS",
        "DirectoSinClima_intraplanta_28d": "Dir28d IP s/clima",
    }
    sub_b["lbl"] = sub_b["escenario"].map(short)
    sub_b = sub_b.sort_values("delta_pct_SMCFHR_vs_mejor_CF_en_CF")
    cols_b = [C_BAYES if v <= 0 else C_RAW for v in sub_b["delta_pct_SMCFHR_vs_mejor_CF_en_CF"]]
    yb = np.arange(len(sub_b))
    ax2.barh(yb, sub_b["delta_pct_SMCFHR_vs_mejor_CF_en_CF"], color=cols_b, alpha=0.88)
    ax2.axvline(0.0, color=C_OBS, lw=1.0)
    for idx, (_, r) in enumerate(sub_b.iterrows()):
        v = r["delta_pct_SMCFHR_vs_mejor_CF_en_CF"]
        offset = 0.25 if v >= 0 else -0.25
        ha = "left" if v >= 0 else "right"
        ax2.text(v + offset, idx, f"{v:+.2f}% ({r['mejor_alt_CF'].replace('__CF','')})", va="center", ha=ha, fontsize=8.0)
    ax2.set_yticks(yb)
    ax2.set_yticklabels(sub_b["lbl"])
    ax2.set_xlim(-3.5, 16.5)
    ax2.set_xlabel("Diferencia % de SM-CFHR (FC) vs. mejor base FC")
    ax2.set_title("Condición B (CV 3 pliegues): SM-CFHR vs. pares en FC")
    ax2.grid(axis="x", alpha=0.22, ls=":")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_res_04_equivalencia_representacion.pdf", bbox_inches="tight")
    plt.close(fig)


def fig05_rankings_y_regimenes() -> None:
    df = pd.read_csv(V2_DIR / "05_metricas_corregidas" / "metricas_completas_28_escenarios_cf_y_kwh.csv")
    cmp_fh = pd.read_csv(OOF_DIR / "metricas" / "comparacion_fixed_vs_hpo.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.1))
    # Panel A: Rank in 23 models (kWh vs CF) by suite/task
    df["grupo"] = df["suite"] + " (" + np.where(df["task"] == "espaciotemporal", "ET", "IP") + ")"
    g = df.groupby("grupo", as_index=False)[["rango_en_23_kWh", "rango_en_23_CF"]].mean().sort_values("rango_en_23_CF")
    y = np.arange(len(g))
    h = 0.36
    ax1.barh(y - h / 2, g["rango_en_23_kWh"], height=h, color=C_RAW, alpha=0.85, label="Rango en MAE (kWh)")
    ax1.barh(y + h / 2, g["rango_en_23_CF"], height=h, color=C_SM, alpha=0.90, label="Rango en MAE (FC)")
    ax1.set_yticks(y)
    ax1.set_yticklabels(g["grupo"], fontsize=8.5)
    ax1.set_xlabel("Rango medio entre 23 modelos (menor = mejor)")
    ax1.set_title("Sensibilidad de rango: kWh vs. Factor de Capacidad")
    ax1.legend(frameon=False, fontsize=8.3)
    ax1.grid(axis="x", alpha=0.22, ls=":")

    # Panel B: Fixed vs HPO relative difference (%) across 28 scenarios
    cmp_fh["rel"] = cmp_fh["rel_macro_HPO_vs_Fixed_pct"]
    et_vals = cmp_fh[cmp_fh["task"] == "espaciotemporal"]["rel"].to_numpy()
    ip_vals = cmp_fh[cmp_fh["task"] == "intraplanta"]["rel"].to_numpy()
    bp = ax2.boxplot([et_vals, ip_vals], tick_labels=["Espaciotemporal (14)", "Intraplanta (14)"], patch_artist=True, widths=0.45)
    for patch, col in zip(bp["boxes"], [C_SM, C_ELASTIC]):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
    ax2.axhline(0.0, color=C_OBS, lw=1.0, ls="--")
    ax2.set_ylabel("Cambio % MAE_macro (HPO - Fixed)")
    ax2.set_title("SM-CFHR-Fixed (B=0) vs. HPO (B=25)")
    ax2.text(0.5, 0.92, "Fixed vence en 19/28 escenarios\n(14/14 ET; coste 59x menor)", transform=ax2.transAxes,
             ha="center", va="top", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="#F8FAFC", ec="#CBD5E1"))
    ax2.grid(axis="y", alpha=0.22, ls=":")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_res_05_rankings_y_regimenes.pdf", bbox_inches="tight")
    plt.close(fig)


def fig06_ablaciones_y_monotonia() -> None:
    df_ab = pd.read_csv(AB_DIR / "ablaciones_controladas_6_escenarios.csv")
    df_mo = pd.read_csv(AB_DIR / "auditoria_monotonia_activacion.csv")
    df_hu = pd.read_csv(AB_DIR / "auditoria_huber_operacional_y_estres.csv")

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0))
    # Panel A: Ablation deltas (mean across scenarios)
    sub_v = df_ab[df_ab["variant"] != "SM_CFHR_completo"].groupby("variant", as_index=False)["delta_pct_CF"].mean()
    sub_v = sub_v.sort_values("delta_pct_CF")
    lbl_map = {
        "sin_L2_anisotropica_010": "L2 isotr. (0,10)",
        "escala_STD_en_target": "Escala STD target",
        "sin_Huber_L2": "Sin Huber (L2)",
        "sin_monotonia": "Sin monotonía",
        "sin_L2_anisotropica_001": "L2 isotr. (0,01)",
        "sin_anomalias_meteo": "Sin anomalías NWP",
        "sin_meteorologia": "Sin meteorología",
        "sin_CF_energia_bruta": "Sin FC (kWh bruto)",
        "sin_CF_y_sin_Huber_L2": "Sin FC ni Huber",
    }
    sub_v["lbl"] = sub_v["variant"].map(lambda x: lbl_map.get(x, x))
    cols_a = [C_BAYES if v < 0 else C_RAW for v in sub_v["delta_pct_CF"]]
    ya = np.arange(len(sub_v))
    axes[0].barh(ya, sub_v["delta_pct_CF"], color=cols_a, alpha=0.88)
    axes[0].axvline(0.0, color=C_OBS, lw=0.9)
    axes[0].set_yticks(ya)
    axes[0].set_yticklabels(sub_v["lbl"], fontsize=8.0)
    axes[0].set_xlabel("Cambio medio % MAE_CF al retirar")
    axes[0].set_title("Ablación factorial (6 escenarios)")
    axes[0].grid(axis="x", alpha=0.22, ls=":")

    # Panel B: Monotonicity activation rate
    short = {
        "D1_espaciotemporal_cinco_variables": "D1 ET 5v",
        "D1_espaciotemporal_sin_meteorologia": "D1 ET s/c",
        "RecGEFSv12_espaciotemporal_7d": "Rec7d ET",
        "RecGEFSv12_espaciotemporal_28d": "Rec28d ET",
        "D1_intraplanta_cinco_variables": "D1 IP 5v",
        "DirectoSinClima_intraplanta_28d": "Dir28d IP",
    }
    df_mo["lbl"] = df_mo["escenario"].map(short)
    df_mo = df_mo.sort_values("pct_cf30_activa")
    ym = np.arange(len(df_mo))
    axes[1].barh(ym - 0.18, df_mo["pct_cf30_activa"], height=0.35, color=C_SM, label="cota cf30 >= 0")
    axes[1].barh(ym + 0.18, df_mo["pct_rad_activa"], height=0.35, color=C_ELASTIC, label="cota rad >= 0")
    axes[1].set_yticks(ym)
    axes[1].set_yticklabels(df_mo["lbl"], fontsize=8.0)
    axes[1].set_xlabel("% de pliegues con cota activa")
    axes[1].set_title("Activación de monotonía")
    axes[1].legend(frameon=False, fontsize=8.0, loc="lower right")
    axes[1].grid(axis="x", alpha=0.22, ls=":")

    # Panel C: Huber vs L2 operational & synthetic stress/leverage
    reg_lbl = {
        "Operacional_completo": "Operacional (4312)",
        "Episodios_bajo_rendimiento_sy<0.3": "Bajo rend. (11)",
        "Cola_superior_5pct_error": "Cola 5% error (216)",
        "Estres_0pct_limpio": "Estrés 0% limpio",
        "Contaminacion_target_5pct": "Contam. target 5%",
        "Contaminacion_target_10pct": "Contam. target 10%",
        "Apalancamiento_covariable_rad_5pct": "Apalanc. rad 5%",
    }
    df_hu["lbl"] = df_hu["regimen"].map(reg_lbl)
    yh = np.arange(len(df_hu))
    axes[2].barh(yh - 0.18, df_hu["MAE_CF_Huber"], height=0.35, color=C_SM, label="Huber (delta=1,35)")
    axes[2].barh(yh + 0.18, df_hu["MAE_CF_L2"], height=0.35, color=C_RAW, label="Pérdida L2")
    axes[2].set_yticks(yh)
    axes[2].set_yticklabels(df_hu["lbl"], fontsize=8.0)
    axes[2].set_xlabel("MAE en factor de capacidad")
    axes[2].set_title("Huber vs. L2: operacional y estrés")
    axes[2].legend(frameon=False, fontsize=8.0, loc="lower right")
    axes[2].grid(axis="x", alpha=0.22, ls=":")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_res_06_ablaciones_y_monotonia.pdf", bbox_inches="tight")
    plt.close(fig)


def fig07_error_por_planta_y_escala() -> None:
    df_p = pd.read_csv(EQ_DIR / "resultados_por_planta_equivalencia.csv")
    df_b = pd.read_csv(AB_DIR / "sensibilidad_sesgo_seleccion_horas_xm.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.1))
    # Panel A: Per-plant MAE_CF vs cap_MW in D1 ET 5var
    sub = df_p[(df_p["escenario"] == "D1_espaciotemporal_cinco_variables") & (df_p["metodo"].isin(["SM-CFHR__CF", "BayesianRidge__CF", "HuberRegressor__kWh_bruto"]))]
    for m, lbl, col, mk in [
        ("SM-CFHR__CF", "SM-CFHR (FC)", C_SM, "o"),
        ("BayesianRidge__CF", "BayesianRidge (FC)", C_BAYES, "s"),
        ("HuberRegressor__kWh_bruto", "HuberRegressor (kWh bruto)", C_RAW, "^"),
    ]:
        ms = sub[sub["metodo"] == m]
        ax1.scatter(ms["cap_MW"], ms["mae_cf"], color=col, marker=mk, s=36, alpha=0.85, label=lbl)
    ax1.set_xscale("log")
    ax1.set_xlabel("Capacidad instalada vigente (MW, escala log)")
    ax1.set_ylabel("MAE_CF por central")
    ax1.set_title("Error por central y escala (D1 ET 5 var.)")
    ax1.legend(frameon=False, fontsize=8.2)
    ax1.grid(alpha=0.22, ls=":")

    # Panel B: Selection bias sensitivity horas_xm <= 6
    sub_b = df_b[df_b["variant"] == "SM_CFHR_completo"].copy()
    sub_b["esc_lbl"] = np.where(sub_b["escenario"].str.contains("espaciotemporal"), "D1 ET 5 var.", "D1 IP 5 var.")
    piv = sub_b.pivot(index="esc_lbl", columns="muestra", values="MAE_CF_macro")
    x = np.arange(len(piv))
    w = 0.34
    ax2.bar(x - w / 2, piv["Filtrado_horas>6"], width=w, color=C_SM, label="Filtrado (horas_xm > 6)")
    ax2.bar(x + w / 2, piv["Con_incompletos_<=6h_observados"], width=w, color=C_RAW, label="Con incompletos (<= 6h)")
    for idx, (v1, v2) in enumerate(zip(piv["Filtrado_horas>6"], piv["Con_incompletos_<=6h_observados"])):
        ax2.text(idx + w / 2, v2 + 0.0005, f"+{100*(v2-v1)/v1:.2f}%", ha="center", fontsize=8.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(piv.index)
    ax2.set_ylim(0, max(piv.max()) * 1.16)
    ax2.set_ylabel("MAE_CF_macro")
    ax2.set_title("Sensibilidad al filtro horas_xm <= 6")
    ax2.legend(frameon=False, fontsize=8.2)
    ax2.grid(axis="y", alpha=0.22, ls=":")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_res_07_error_por_planta_y_escala.pdf", bbox_inches="tight")
    plt.close(fig)


def fig08_incertidumbre_conformal() -> None:
    res = pd.read_csv(OOF_DIR / "incertidumbre" / "tabla_conformal_resumen.csv")
    plt_df = pd.read_csv(OOF_DIR / "incertidumbre" / "tabla_conformal_por_planta.csv")
    p90 = plt_df[plt_df["nominal_coverage"] == 0.9].sort_values("empirical_coverage")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.1))
    y = np.arange(len(p90))
    ax1.barh(y, 100.0 * p90["empirical_coverage"], color=C_SM, alpha=0.88)
    ax1.axvline(90.0, color=C_RAW, ls="--", lw=1.3, label="Nominal 90%")
    r90 = float(res[res["nominal_coverage"] == 0.9]["empirical_coverage"].iloc[0]) * 100.0
    ax1.axvline(r90, color=C_OBS, ls=":", lw=1.2, label=f"Empírica global ({r90:.2f}%)")
    ax1.set_yticks(y)
    ax1.set_yticklabels(p90["plant"], fontsize=8.0)
    ax1.set_xlabel("Cobertura empírica (%)")
    ax1.set_title("Cobertura conforme por planta (nominal 90%)")
    ax1.legend(frameon=False, fontsize=8.2, loc="lower right")
    ax1.grid(axis="x", alpha=0.22, ls=":")

    # Right: Summary nominal vs empirical + halfwidth
    x = np.arange(len(res))
    w = 0.34
    ax2.bar(x - w / 2, 100.0 * res["nominal_coverage"], width=w, color=C_MUTE, alpha=0.7, label="Cobertura nominal (%)")
    ax2.bar(x + w / 2, 100.0 * res["empirical_coverage"], width=w, color=C_SM, alpha=0.9, label="Cobertura empírica (%)")
    for idx, r in res.reset_index(drop=True).iterrows():
        ax2.text(idx + w / 2, 100.0 * r["empirical_coverage"] + 1.5,
                 f"{100*r['empirical_coverage']:.2f}%\n(±{r['conformal_halfwidth']:.4f} FC)",
                 ha="center", fontsize=8.2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{int(100*v)}%" for v in res["nominal_coverage"]])
    ax2.set_ylim(0, 108)
    ax2.set_ylabel("Cobertura (%)")
    ax2.set_title(f"Subcobertura bajo deriva temporal (n_cal={int(res['n_cal'].iloc[0])})")
    ax2.legend(frameon=False, fontsize=8.2, loc="upper left")
    ax2.grid(axis="y", alpha=0.22, ls=":")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_res_08_incertidumbre_conformal.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig01_real_vs_pred_d1()
    fig02_real_vs_pred_recursivo()
    fig03_real_vs_pred_directo()
    fig04_equivalencia_representacion()
    fig05_rankings_y_regimenes()
    fig06_ablaciones_y_monotonia()
    fig07_error_por_planta_y_escala()
    fig08_incertidumbre_conformal()
    print("Generated all 8 Phase 5 Spanish result PDFs in", OUT_DIR)


if __name__ == "__main__":
    main()

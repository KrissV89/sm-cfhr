"""Generate 6 Spanish vector PDF result figures strictly from frozen OOF & verified v2 outputs."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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
OOF_DIR = ROOT_DIR / "resultados" / "evaluacion_ashr_oof"
V2_DIR = ROOT_DIR / "resultados" / "inferencia_revision_independiente_v2"

C_OBS = "#0F172A"
C_SM = "#0072B2"
C_ALT = "#009E73"
C_WARN = "#D55E00"
C_ACC = "#CC79A7"
C_MUTE = "#64748B"


def fig01_rankings_y_fixed_hpo() -> None:
    df = pd.read_csv(V2_DIR / "05_metricas_corregidas" / "metricas_completas_28_escenarios_cf_y_kwh.csv")
    cmp_fh = pd.read_csv(OOF_DIR / "metricas" / "comparacion_fixed_vs_hpo.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.1))
    df["grupo"] = df["suite"] + " (" + np.where(df["task"] == "espaciotemporal", "ET", "IP") + ")"
    g = df.groupby("grupo", as_index=False)[["rango_en_23_kWh", "rango_en_23_CF"]].mean().sort_values("rango_en_23_CF")
    y = np.arange(len(g))
    h = 0.36
    ax1.barh(y - h / 2, g["rango_en_23_kWh"], height=h, color=C_WARN, alpha=0.85, label="Rango en MAE (kWh)")
    ax1.barh(y + h / 2, g["rango_en_23_CF"], height=h, color=C_SM, alpha=0.90, label="Rango en MAE (FC)")
    ax1.set_yticks(y)
    ax1.set_yticklabels(g["grupo"], fontsize=8.5)
    ax1.set_xlabel("Rango medio entre 23 modelos (menor = mejor)")
    ax1.set_title("Sensibilidad de rango: kWh vs. Factor de Capacidad")
    ax1.legend(frameon=False, fontsize=8.3)
    ax1.grid(axis="x", alpha=0.22, ls=":")

    et_vals = cmp_fh[cmp_fh["task"] == "espaciotemporal"]["rel_macro_HPO_vs_Fixed_pct"].to_numpy()
    ip_vals = cmp_fh[cmp_fh["task"] == "intraplanta"]["rel_macro_HPO_vs_Fixed_pct"].to_numpy()
    bp = ax2.boxplot([et_vals, ip_vals], tick_labels=["Espaciotemporal (14)", "Intraplanta (14)"], patch_artist=True, widths=0.45)
    for patch, col in zip(bp["boxes"], [C_SM, C_ACC]):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
    ax2.axhline(0.0, color=C_OBS, lw=1.0, ls="--")
    ax2.set_ylabel("Cambio % MAE_macro (HPO - Fixed)")
    ax2.set_title("SM-CFHR-Fixed (B=0) vs. HPO (B=25)")
    ax2.text(
        0.5, 0.92,
        "Fixed vence en 19/28 escenarios\n(14/14 ET; coste 59x menor)",
        transform=ax2.transAxes, ha="center", va="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#F8FAFC", ec="#CBD5E1"),
    )
    ax2.grid(axis="y", alpha=0.22, ls=":")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_res_oof_rankings_y_fixed_hpo.pdf", bbox_inches="tight")
    plt.close(fig)


def fig02_planta_y_grupos() -> None:
    intra = pd.read_csv(V2_DIR / "05_metricas_corregidas" / "intra_por_planta_capacidad_corregido.csv")
    inter = pd.read_csv(V2_DIR / "05_metricas_corregidas" / "inter_grupo_et_vs_ip_pareado.csv").sort_values("grupo")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax1.scatter(intra["cap_MW_media"], intra["MAE_CF_ASHR"], color=C_SM, s=42, alpha=0.88, label="SM-CFHR-Fixed")
    ax1.scatter(intra["cap_MW_media"], intra["MAE_CF_MejorBib"], color=C_WARN, marker="^", s=38, alpha=0.82, label="Mejor biblioteca (kWh)")
    ax1.set_xscale("log")
    ax1.set_xlabel("Capacidad media instalada (MW, escala log)")
    ax1.set_ylabel("MAE_CF por planta (cap. diaria al origen)")
    ax1.set_title("Error adimensional por planta (D1 ET 5 var.)")
    ax1.legend(frameon=False, fontsize=8.3)
    ax1.grid(alpha=0.22, ls=":")

    x = np.arange(len(inter))
    w = 0.35
    ax2.bar(x - w / 2, inter["MAE_CF_ET"], width=w, color=C_SM, alpha=0.90, label="Espaciotemporal (ET)")
    ax2.bar(x + w / 2, inter["MAE_CF_IP"], width=w, color=C_ALT, alpha=0.85, label="Intraplanta (IP)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"G{int(r.grupo)}\n({int(r.n_pares)})" for r in inter.itertuples(index=False)], fontsize=8.2)
    ax2.set_xlabel("Grupo espacial y pares coincidentes (code, date)")
    ax2.set_ylabel("MAE_CF medio (cap. diaria al origen)")
    ax2.set_title(f"ET vs. IP pareado por grupo (n={int(inter['n_pares'].sum()):,} pares)")
    ax2.legend(frameon=False, fontsize=8.3)
    ax2.grid(axis="y", alpha=0.22, ls=":")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_res_oof_planta_y_grupos.pdf", bbox_inches="tight")
    plt.close(fig)


def fig03_ablaciones_y_conformal() -> None:
    abl = pd.read_csv(OOF_DIR / "ablaciones" / "tabla_ablaciones.csv").sort_values("mae_cf")
    conf_res = pd.read_csv(OOF_DIR / "incertidumbre" / "tabla_conformal_resumen.csv")
    conf_plt = pd.read_csv(OOF_DIR / "incertidumbre" / "tabla_conformal_por_planta.csv")
    p90 = conf_plt[conf_plt["nominal_coverage"] == 0.9].sort_values("empirical_coverage")

    lbl_map = {
        "sin_l2_estructurada": "Sin L2 diferenciada",
        "SM-CFHR_completo": "SM-CFHR completo",
        "sin_monotonia": "Sin monotonía",
        "escala_std_vs_mad": "Escala STD vs. MAD",
        "sin_huber_L2": "Sin Huber (L2)",
        "energia_bruta_vs_cf": "Energía bruta (sin FC)",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.1))
    y1 = np.arange(len(abl))
    cols = [C_WARN if v == "energia_bruta_vs_cf" else C_SM for v in abl["ablation_variant"]]
    ax1.barh(y1, abl["mae_cf"], color=cols, alpha=0.88)
    for idx, v in enumerate(abl["mae_cf"]):
        ax1.text(v + 0.003, idx, f"{v:.5f}", va="center", fontsize=8.0)
    ax1.set_yticks(y1)
    ax1.set_yticklabels([lbl_map.get(v, v) for v in abl["ablation_variant"]], fontsize=8.3)
    ax1.set_xlim(0, max(abl["mae_cf"]) * 1.22)
    ax1.set_xlabel("MAE_CF en corte cronológico D+1")
    ax1.set_title("Ablación diagnóstica D+1 (panel real)")
    ax1.grid(axis="x", alpha=0.22, ls=":")

    y2 = np.arange(len(p90))
    ax2.barh(y2, 100.0 * p90["empirical_coverage"], color=C_SM, alpha=0.88)
    ax2.axvline(90.0, color=C_WARN, ls="--", lw=1.3, label="Nominal 90%")
    r90 = float(conf_res[conf_res["nominal_coverage"] == 0.9]["empirical_coverage"].iloc[0]) * 100.0
    ax2.axvline(r90, color=C_OBS, ls=":", lw=1.2, label=f"Empírica global ({r90:.2f}%)")
    ax2.set_yticks(y2)
    ax2.set_yticklabels(p90["plant"], fontsize=8.0)
    ax2.set_xlabel("Cobertura empírica (%)")
    ax2.set_title("Incertidumbre conforme temporal (90% nom.)")
    ax2.legend(frameon=False, fontsize=8.2, loc="lower right")
    ax2.grid(axis="x", alpha=0.22, ls=":")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_res_oof_ablaciones_y_conformal.pdf", bbox_inches="tight")
    plt.close(fig)


def copy_frozen_real_vs_pred_pdfs() -> None:
    p2_dir = V2_DIR / "06_exploratorio_p2"
    mapping = [
        ("real_vs_predicho_d1.pdf", "fig_res_oof_real_vs_pred_d1.pdf"),
        ("real_vs_predicho_multihorizonte_rec.pdf", "fig_res_oof_real_vs_pred_rec.pdf"),
        ("real_vs_predicho_directo_acumulado.pdf", "fig_res_oof_real_vs_pred_dir.pdf"),
    ]
    for src_name, dst_name in mapping:
        src = p2_dir / src_name
        if not src.exists():
            raise FileNotFoundError(f"Missing verified frozen figure: {src}")
        shutil.copy2(src, OUT_DIR / dst_name)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig01_rankings_y_fixed_hpo()
    fig02_planta_y_grupos()
    fig03_ablaciones_y_conformal()
    copy_frozen_real_vs_pred_pdfs()
    print("Generated 6 frozen-OOF result PDFs in", OUT_DIR)


if __name__ == "__main__":
    main()

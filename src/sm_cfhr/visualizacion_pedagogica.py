"""Scientific visualizations for library-only, intra-dataset, inter-group, and OOF evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from ashr_oof_pipeline.config import SCENARIOS_28
    from ashr_oof_pipeline.evaluacion_estadistica import compute_scenario_metrics, model_confidence_set
    from ashr_oof_pipeline.oof_runner import load_official_benchmarks
    from ashr_oof_pipeline.visualizacion import _save_fig
    from sm_cfhr.inference import spa_test
    from sm_cfhr.persistence import atomic_write_csv
except ImportError:
    from src.ashr_oof_pipeline.config import SCENARIOS_28
    from src.ashr_oof_pipeline.evaluacion_estadistica import compute_scenario_metrics, model_confidence_set
    from src.ashr_oof_pipeline.oof_runner import load_official_benchmarks
    from src.ashr_oof_pipeline.visualizacion import _save_fig
    from src.sm_cfhr.inference import spa_test
    from src.sm_cfhr.persistence import atomic_write_csv

LIBRARIES_9 = sorted(
    [
        "BayesianRidge",
        "DecisionTree",
        "HistGradientBoosting",
        "Lasso",
        "Random Forest",
        "Ridge",
        "SVR",
        "XGBoost",
        "k-NN",
    ]
)


def _short_sc(sc_id: str) -> str:
    return (
        sc_id.replace("espaciotemporal", "ET")
        .replace("intraplanta", "IP")
        .replace("cinco_variables", "5var")
        .replace("sin_meteorologia", "noMet")
        .replace("DirectoSinClima", "DirSinClima")
    )


def compute_library_only_tables(oof_dir: Path, root: Path) -> dict[str, pd.DataFrame]:
    """Compute post-hoc evaluation restricted exclusively to SM-CFHR-Fixed vs 9 library models."""
    pred_dir = oof_dir / "predicciones"
    rank_rows: list[dict[str, Any]] = []
    heat_rows: list[dict[str, Any]] = []
    mcs_rows: list[dict[str, Any]] = []
    spa_rows: list[dict[str, Any]] = []

    for sp in SCENARIOS_28:
        bench_raw = load_official_benchmarks(sp, root=root, exclude_cf_brh=True)
        b_keys = bench_raw[list(sp.join_keys)].drop_duplicates()
        ashr_df = pd.read_csv(pred_dir / f"{sp.scenario_id}.csv")
        ashr_df["code"] = ashr_df["code"].astype(str)
        ashr_df = ashr_df.merge(b_keys, on=list(sp.join_keys), how="inner")

        bench_lib = bench_raw[bench_raw["metodo"].isin(LIBRARIES_9)].merge(
            ashr_df[list(sp.join_keys)].drop_duplicates(), on=list(sp.join_keys), how="inner"
        )
        bench_lib["_ae"] = (bench_lib[sp.pred_col] - bench_lib[sp.target_col]).abs()
        by_m = (
            bench_lib.groupby(["metodo", "code"], as_index=False)["_ae"]
            .mean()
            .groupby("metodo", as_index=False)["_ae"]
            .mean()
            .rename(columns={"_ae": "MAE_macro"})
        )
        ashr_mae = compute_scenario_metrics(ashr_df, sp.pred_col, sp.target_col)["MAE_macro"]

        h_row: dict[str, Any] = {"scenario_id": sp.scenario_id, "suite": sp.suite, "task": sp.task, "unit": sp.unit, "MAE_ASHR": ashr_mae}
        for r in by_m.itertuples(index=False):
            h_row[r.metodo] = 100.0 * (ashr_mae / float(r.MAE_macro) - 1.0)
        heat_rows.append(h_row)

        all10 = (
            pd.concat([pd.DataFrame([{"metodo": "SM-CFHR-Fixed", "MAE_macro": ashr_mae}]), by_m], ignore_index=True)
            .sort_values("MAE_macro")
            .reset_index(drop=True)
        )
        all10["rango_en_10"] = np.arange(1, len(all10) + 1)
        all10["scenario_id"] = sp.scenario_id
        rank_rows.append(all10)

        time_col = "date" if sp.mode_eval == "d1" else "origen"
        ashr_t = ashr_df.assign(_ae=np.abs(ashr_df[sp.pred_col] - ashr_df[sp.target_col])).groupby(time_col)["_ae"].mean()
        piv = bench_lib.groupby([time_col, "metodo"], as_index=False)["_ae"].mean().pivot(index=time_col, columns="metodo", values="_ae")
        piv["ASHR"] = ashr_t
        piv = piv.dropna(axis=0, how="any")

        mcs = model_confidence_set(piv.to_numpy(float), list(piv.columns), alpha=0.10, B=500, seed=2026)
        mcs_rows.append(
            {
                "escenario": sp.scenario_id,
                "in_mcs_lib_90": "ASHR" in mcs["survivors"],
                "sole_survivor_lib": mcs["survivors"] == ["ASHR"],
                "n_survivors_lib": len(mcs["survivors"]),
                "p_mcs_ashr_lib": float(mcs["p_mcs"]["ASHR"]),
            }
        )
        spa_ashr = spa_test(piv["ASHR"].to_numpy(float), piv.drop(columns=["ASHR"]).to_numpy(float), B=500, mean_block=7.0, seed=2026)
        spa_rows.append({"escenario": sp.scenario_id, "p_spa_H0_ASHR_vs_9libs": float(spa_ashr["p_spa"])})

    rank_df = pd.concat(rank_rows, ignore_index=True)
    heat_df = pd.DataFrame(heat_rows)
    mcs_df = pd.DataFrame(mcs_rows)
    spa_df = pd.DataFrame(spa_rows)

    atomic_write_csv(rank_df, oof_dir / "inferencia" / "ranking_bibliotecas_only.csv")
    atomic_write_csv(mcs_df, oof_dir / "inferencia" / "mcs_bibliotecas_only.csv")
    return {"rank_10": rank_df, "heat_vs_libs": heat_df, "mcs_lib": mcs_df, "spa_lib": spa_df}


def generate_pedagogical_figures(oof_dir: Path, root: Path) -> dict[str, tuple[Path, Path]]:
    """Generate all 10 pedagogical PNG (300 dpi) + PDF figures from saved OOF artifacts."""
    fig_dir = oof_dir / "figuras" / "pedagogicas"
    fig_dir.mkdir(parents=True, exist_ok=True)
    lib_res = compute_library_only_tables(oof_dir, root)

    t28 = pd.read_csv(oof_dir / "metricas" / "tabla_28_escenarios.csv")
    fh_df = pd.read_csv(oof_dir / "metricas" / "comparacion_fixed_vs_hpo.csv")
    tmul = pd.read_csv(oof_dir / "multiplicidad" / "tabla_multiplicidad_familias.csv")
    tmcs_all = pd.read_csv(oof_dir / "inferencia" / "tabla_mcs.csv")
    r3_df = pd.read_csv(oof_dir / "conclusiones" / "veredicto_r3.csv")

    sub_bib = tmul[tmul["referencia"] == "Mejor_Bib"].merge(
        t28[["scenario_id", "MAE_ASHR", "MAE_Mejor_Bib", "pct_vs_MejorLib"]],
        left_on="escenario",
        right_on="scenario_id",
        how="left",
    )

    out: dict[str, tuple[Path, Path]] = {}

    # Fig P01: Ranking medio en 10 modelos (SM-CFHR-Fixed + 9 bibliotecas) y victorias 1-vs-1
    rank_df = lib_res["rank_10"]
    heat_df = lib_res["heat_vs_libs"]
    mean_ranks = rank_df.groupby("metodo", as_index=False)["rango_en_10"].mean().sort_values("rango_en_10")
    wins_1v1 = {m: int((heat_df[m] < 0).sum()) for m in LIBRARIES_9}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    colors = ["#1b7837" if m == "SM-CFHR-Fixed" else "#4393c3" for m in mean_ranks["metodo"]]
    y = np.arange(len(mean_ranks))
    axes[0].barh(y, mean_ranks["rango_en_10"], color=colors)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(mean_ranks["metodo"], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Rango ordinal medio (1 = mejor, sobre 10 modelos)")
    axes[0].set_title("Rango ordinal medio entre 10 modelos")
    for idx, val in enumerate(mean_ranks["rango_en_10"]):
        axes[0].text(val + 0.1, idx, f"{val:.2f}", va="center", fontsize=8.5)
    axes[0].grid(axis="x", linestyle=":", alpha=0.5)

    lib_sorted = sorted(LIBRARIES_9, key=lambda m: wins_1v1[m], reverse=True)
    w_vals = [wins_1v1[m] for m in lib_sorted]
    y2 = np.arange(len(lib_sorted))
    axes[1].barh(y2, w_vals, color="#2166ac")
    axes[1].set_yticks(y2)
    axes[1].set_yticklabels(lib_sorted, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 31)
    axes[1].axvline(14, color="#666666", linestyle="--", linewidth=0.9)
    axes[1].set_xlabel("Escenarios donde SM-CFHR-Fixed < biblioteca (de 28)")
    axes[1].set_title("Victorias 1-vs-1 frente a cada biblioteca")
    for idx, val in enumerate(w_vals):
        axes[1].text(val + 0.4, idx, f"{val}/28", va="center", fontsize=8.5)
    axes[1].grid(axis="x", linestyle=":", alpha=0.5)
    fig.suptitle("Rango ordinal y victorias 1-vs-1 frente a nueve bibliotecas", fontsize=11.5)
    fig.tight_layout()
    out["P01"] = _save_fig(fig, fig_dir, "figP01_ranking_bibliotecas")

    # Fig P02: Heatmap 28 escenarios x 9 bibliotecas (% MAE_macro)
    mat = heat_df[LIBRARIES_9].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11.5, 9.2))
    im = ax.imshow(np.clip(mat, -30, 30), cmap="RdBu_r", vmin=-30, vmax=30, aspect="auto")
    ax.set_xticks(np.arange(len(LIBRARIES_9)))
    ax.set_xticklabels(LIBRARIES_9, rotation=30, ha="right", fontsize=8.5)
    ax.set_yticks(np.arange(len(heat_df)))
    ylabels = [f"{_short_sc(r.scenario_id)} [{'kWh/d' if r.unit == 'kWh/dia' else 'kWh_tot'}]" for r in heat_df.itertuples(index=False)]
    ax.set_yticklabels(ylabels, fontsize=7.5)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:+.1f}", ha="center", va="center", fontsize=6.2, color="black" if abs(mat[i, j]) < 18 else "white")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Δ% MAE_macro de SM-CFHR-Fixed vs. biblioteca (< 0 favorece a SM-CFHR)")
    ax.set_title("Diferencia relativa de MAE_macro frente a cada biblioteca (28 escenarios)")
    fig.tight_layout()
    out["P02"] = _save_fig(fig, fig_dir, "figP02_heatmap_vs_bibliotecas")

    # Fig P03: Victorias (18) y derrotas (10) frente al mejor de biblioteca por escenario
    s3 = sub_bib.sort_values("pct_vs_MejorLib").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, 8.8))
    c3 = ["#1b7837" if v < 0 else "#b2182b" for v in s3["pct_vs_MejorLib"]]
    y3 = np.arange(len(s3))
    ax.barh(y3, s3["pct_vs_MejorLib"], color=c3)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_yticks(y3)
    lbls3 = [
        f"{_short_sc(r.escenario)} vs {r.modelo_referencia}{' *' if r.estado_estadistico == 'SIGNIFICATIVO_HOLM' else ''}"
        for r in s3.itertuples(index=False)
    ]
    ax.set_yticklabels(lbls3, fontsize=7.8)
    ax.invert_yaxis()
    for idx, v in enumerate(s3["pct_vs_MejorLib"]):
        ax.text(v + (0.4 if v >= 0 else -0.4), idx, f"{v:+.1f}%", va="center", ha="left" if v >= 0 else "right", fontsize=7.2)
    ax.set_xlabel("Diferencia porcentual en MAE_macro vs. mejor biblioteca (< 0 favorece a SM-CFHR-Fixed)")
    ax.set_title("Diferencia de MAE_macro frente a la mejor biblioteca por escenario (* = p_Holm < 0.05)")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out["P03"] = _save_fig(fig, fig_dir, "figP03_victorias_derrotas_bibliotecas")

    # Fig P04: Comparación pareada MAE_macro SM-CFHR vs Mejor_Bib (2 subpaneles por unidad)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, unit_lbl, title in [
        (axes[0], "kWh/dia", "Escenarios diarios y recursivos (20 esc., kWh/día)"),
        (axes[1], "kWh_total", "Escenarios directos acumulados (8 esc., kWh totales)"),
    ]:
        sub = t28[t28["unit"] == unit_lbl]
        for task, col, mk in [("espaciotemporal", "#1b7837", "o"), ("intraplanta", "#762a83", "s")]:
            st = sub[sub["task"] == task]
            ax.scatter(st["MAE_Mejor_Bib"] / 1000.0, st["MAE_ASHR"] / 1000.0, c=col, marker=mk, s=55, label=task, alpha=0.88)
        lim_lo = min(sub["MAE_Mejor_Bib"].min(), sub["MAE_ASHR"].min()) / 1000.0 * 0.92
        lim_hi = max(sub["MAE_Mejor_Bib"].max(), sub["MAE_ASHR"].max()) / 1000.0 * 1.05
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=1.0, label="Igualdad (y = x)")
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_xlabel("MAE_macro Mejor Biblioteca (MWh)")
        ax.set_ylabel("MAE_macro SM-CFHR-Fixed (MWh)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)
    fig.suptitle("Comparación pareada de MAE_macro sobre observaciones alineadas", fontsize=11.5)
    fig.tight_layout()
    out["P04"] = _save_fig(fig, fig_dir, "figP04_pareado_ashr_vs_mejor_bib")

    # Fig P05: Desglose por horizontes (normalizado /H en Directo para comparar en kWh/día) y regímenes
    fh_norm = fh_df.copy()
    fh_norm["MAE_Fixed_dia"] = np.where(fh_norm["unit"] == "kWh_total", fh_norm["MAE_macro_Fixed"] / fh_norm["horizon_days"], fh_norm["MAE_macro_Fixed"])
    fh_norm["MAE_HPO_dia"] = np.where(fh_norm["unit"] == "kWh_total", fh_norm["MAE_macro_HPO"] / fh_norm["horizon_days"], fh_norm["MAE_macro_HPO"])
    fh_norm = fh_norm.merge(t28[["scenario_id", "MAE_Mejor_Bib"]], on="scenario_id", how="left")
    fh_norm["MAE_Bib_dia"] = np.where(fh_norm["unit"] == "kWh_total", fh_norm["MAE_Mejor_Bib"] / fh_norm["horizon_days"], fh_norm["MAE_Mejor_Bib"])

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    h_agg = fh_norm.groupby("horizon_days", as_index=False)[["MAE_Fixed_dia", "MAE_HPO_dia", "MAE_Bib_dia"]].mean()
    x = np.arange(len(h_agg))
    w = 0.26
    axes[0].bar(x - w, h_agg["MAE_Fixed_dia"] / 1000.0, width=w, label="SM-CFHR-Fixed", color="#1b7837")
    axes[0].bar(x, h_agg["MAE_HPO_dia"] / 1000.0, width=w, label="SM-CFHR-HPO", color="#2166ac")
    axes[0].bar(x + w, h_agg["MAE_Bib_dia"] / 1000.0, width=w, label="Mejor Biblioteca", color="#999999")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{int(h)}d" if h > 1 else "D+1" for h in h_agg["horizon_days"]])
    axes[0].set_ylabel("MAE_macro equivalente (MWh/día; Directo dividido entre H)")
    axes[0].set_title("MAE_macro equivalente por horizonte")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", linestyle=":", alpha=0.5)

    reg_agg = fh_norm.groupby(["suite", "task"], as_index=False)[["pct_Fixed_vs_BestBib", "pct_HPO_vs_BestBib"]].mean()
    y_r = np.arange(len(reg_agg))
    axes[1].barh(y_r - 0.18, reg_agg["pct_Fixed_vs_BestBib"], height=0.35, label="Fixed vs. Mejor Bib.", color="#1b7837")
    axes[1].barh(y_r + 0.18, reg_agg["pct_HPO_vs_BestBib"], height=0.35, label="HPO vs. Mejor Bib.", color="#2166ac")
    axes[1].axvline(0.0, color="black", linewidth=1.0)
    axes[1].set_yticks(y_r)
    axes[1].set_yticklabels([f"{r.suite} | {'ET' if r.task == 'espaciotemporal' else 'IP'}" for r in reg_agg.itertuples(index=False)], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Δ% medio vs. Mejor Biblioteca (< 0 favorece a SM-CFHR)")
    axes[1].set_title("Diferencia media por suite y régimen")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out["P05"] = _save_fig(fig, fig_dir, "figP05_horizontes_y_regimenes")

    # Fig P06: Forest plot de diferencias pareadas (ASHR - Mejor_Bib) con IC 95%
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 9.6), gridspec_kw={"height_ratios": [20, 8]})
    for ax, unit_lbl, ttl in [
        (axes[0], "kWh/dia", "Diferencias pareadas (ASHR - Mejor_Bib) e IC 95% — Escenarios diarios/recursivos (MWh/día)"),
        (axes[1], "kWh_total", "Diferencias pareadas (ASHR - Mejor_Bib) e IC 95% — Escenarios directos acumulados (MWh totales)"),
    ]:
        sub = sub_bib[sub_bib["unit"] == unit_lbl].sort_values("mean_diff").reset_index(drop=True)
        y6 = np.arange(len(sub))
        for idx, r in enumerate(sub.itertuples(index=False)):
            if r.estado_estadistico == "SIGNIFICATIVO_HOLM":
                col = "#1b7837"
            elif bool(r.t_eff_insufficient):
                col = "#999999"
            else:
                col = "#2166ac" if r.mean_diff < 0 else "#b2182b"
            md_m = r.mean_diff / 1000.0
            lo_m = r.ci95_lo / 1000.0
            hi_m = r.ci95_hi / 1000.0
            ax.errorbar(md_m, idx, xerr=[[max(0.0, md_m - lo_m)], [max(0.0, hi_m - md_m)]], fmt="o", color=col, capsize=3, markersize=4.5)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_yticks(y6)
        ax.set_yticklabels([f"{_short_sc(r.escenario)} ({'Teff<2H' if r.t_eff_insufficient else r.estado_estadistico[:6]})" for r in sub.itertuples(index=False)], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("Diferencia media de error temporal (MWh; < 0 favorece a SM-CFHR-Fixed)")
        ax.set_title(ttl, fontsize=9.5)
        ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out["P06"] = _save_fig(fig, fig_dir, "figP06_forest_plot_ic95")

    # Fig P07: Mapa de significancia vs tamaño del efecto + Matriz MCS/SPA
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), gridspec_kw={"width_ratios": [1.15, 1.0]})
    for idx, r in enumerate(sub_bib.itertuples(index=False)):
        y_p = -np.log10(max(float(r.p_dm_hln), 1e-25))
        if r.estado_estadistico == "SIGNIFICATIVO_HOLM":
            col, mk = "#1b7837", "*"
        elif bool(r.t_eff_insufficient):
            col, mk = "#999999", "^"
        else:
            col, mk = ("#2166ac" if r.cohen_dz_pareado < 0 else "#b2182b"), "o"
        axes[0].scatter(r.cohen_dz_pareado, y_p, c=col, marker=mk, s=90 if mk == "*" else 50)
        if r.estado_estadistico == "SIGNIFICATIVO_HOLM" or abs(r.cohen_dz_pareado) >= 0.6:
            axes[0].text(r.cohen_dz_pareado + 0.02, y_p + 0.2, _short_sc(r.escenario)[:18], fontsize=7)
    axes[0].axvline(-0.20, color="#1b7837", linestyle="--", linewidth=0.9, label="Umbral efecto a favor (dz = -0.20)")
    axes[0].axvline(0.20, color="#b2182b", linestyle="--", linewidth=0.9)
    axes[0].axhline(-np.log10(0.05), color="#444444", linestyle=":", linewidth=0.9, label="p bruto = 0.05")
    axes[0].set_xlabel("Tamaño del efecto pareado Cohen's dz (< 0 favorece a SM-CFHR)")
    axes[0].set_ylabel("-log10(p_DM_HLN_HAC)")
    axes[0].set_title("Significancia estadística vs. relevancia práctica")
    axes[0].legend(fontsize=7.5, loc="upper right")
    axes[0].grid(True, linestyle=":", alpha=0.5)

    mcs_lib = lib_res["mcs_lib"]
    spa_lib = lib_res["spa_lib"]
    m_all = tmcs_all[["escenario", "ashr_in_mcs_90"]].merge(mcs_lib[["escenario", "in_mcs_lib_90", "sole_survivor_lib"]], on="escenario").merge(spa_lib, on="escenario")
    mat_m = np.column_stack([
        m_all["ashr_in_mcs_90"].astype(int).to_numpy(),
        m_all["in_mcs_lib_90"].astype(int).to_numpy(),
        m_all["sole_survivor_lib"].astype(int).to_numpy(),
        (m_all["p_spa_H0_ASHR_vs_9libs"] >= 0.05).astype(int).to_numpy(),
    ])
    axes[1].imshow(mat_m, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    axes[1].set_xticks([0, 1, 2, 3])
    axes[1].set_xticklabels(["MCS 23\n(sobrevive)", "MCS Bib\n(sobrevive)", "MCS Bib\n(único)", "SPA Bib\n(H0 no rech.)"], fontsize=8)
    axes[1].set_yticks(np.arange(len(m_all)))
    axes[1].set_yticklabels([_short_sc(s) for s in m_all["escenario"]], fontsize=6.8)
    axes[1].set_title("Supervivencia MCS y SPA por escenario (verde = Sí)")
    fig.tight_layout()
    out["P07"] = _save_fig(fig, fig_dir, "figP07_mapa_significancia_efectos")

    # Fig P08: Fixed vs HPO por régimen ET vs IP + distribución de hiperparámetros HPO
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    s8 = fh_df.sort_values("rel_macro_HPO_vs_Fixed_pct").reset_index(drop=True)
    c8 = ["#1b7837" if v < 0 else "#2166ac" for v in s8["rel_macro_HPO_vs_Fixed_pct"]]
    y8 = np.arange(len(s8))
    axes[0].barh(y8, s8["rel_macro_HPO_vs_Fixed_pct"], color=c8)
    axes[0].axvline(0.0, color="black", linewidth=1.0)
    axes[0].set_yticks(y8)
    axes[0].set_yticklabels([f"{_short_sc(r.scenario_id)}" for r in s8.itertuples(index=False)], fontsize=7.0)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Δ% MAE_macro de HPO respecto a Fixed (< 0 mejora HPO; 9/28 todos IP)")
    axes[0].set_title("SM-CFHR-HPO vs. SM-CFHR-Fixed por escenario")
    axes[0].grid(axis="x", linestyle=":", alpha=0.5)

    trial_files = sorted((oof_dir / "hpo_trials").glob("*.csv"))
    if trial_files:
        t_sample = pd.concat([pd.read_csv(f).assign(scenario_id=f.stem) for f in trial_files], ignore_index=True)
        valid_t = t_sample[t_sample["trial_valid"]] if "trial_valid" in t_sample.columns else t_sample
        best_per_study = valid_t.sort_values("inner_mae_macro_kwh").groupby(["scenario_id", "outer_fold"], as_index=False).first()
        axes[1].boxplot(
            [best_per_study["delta"], np.log10(best_per_study["alpha_lin"]), np.log10(best_per_study["alpha_meteo"])],
            labels=["delta\n(Fixed=1.35)", "log10(alpha_lin)\n(Fixed=-2.0)", "log10(alpha_meteo)\n(Fixed=-1.0)"],
        )
        axes[1].scatter([1, 2, 3], [1.35, -2.0, -1.0], color="#b2182b", zorder=5, s=60, label="SM-CFHR-Fixed")
        axes[1].set_ylabel("Valor seleccionado en 14,552 estudios Optuna")
        axes[1].set_title("Dispersión de hiperparámetros HPO frente a Fixed")
        axes[1].legend(fontsize=8.5)
        axes[1].grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out["P08"] = _save_fig(fig, fig_dir, "figP08_fixed_vs_hpo_regimenes")

    # Fig P09: Rendimiento vs coste computacional + Benchmark SMT 8/10/12
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    axes[0].scatter([499.748 / 60.0], [18], s=140, color="#1b7837", label="Fixed (8 workers, 8.33 min, 2.07 GB RSS)")
    axes[0].scatter([29550.293 / 60.0], [13], s=140, color="#2166ac", label="HPO (12 workers, 492.5 min, 4.20 GB RSS)")
    axes[0].annotate("Fixed: 18/28 vs Mejor Bib\nRango 10 libs = 1.64", (499.748 / 60.0 + 18, 17.7), fontsize=8.5)
    axes[0].annotate("HPO: 13/28 vs Mejor Bib\n(distinto paralelismo: 12w)", (29550.293 / 60.0 - 210, 13.5), fontsize=8.5)
    axes[0].set_xlabel("Tiempo de pared medido (minutos)")
    axes[0].set_ylabel("Victorias vs. Mejor Biblioteca (de 28)")
    axes[0].set_ylim(10, 21)
    axes[0].set_title("Rendimiento científico vs. tiempo de pared")
    axes[0].legend(fontsize=7.8, loc="lower left")
    axes[0].grid(True, linestyle=":", alpha=0.5)

    bench_p = root / "results" / "benchmark_paralelo" / "benchmark_smt_8_10_12.csv"
    if bench_p.exists():
        bdf = pd.read_csv(bench_p)
        axes[1].bar(bdf["workers"].astype(str) + "w", bdf["units_per_min"], color=["#6baed6", "#3182bd", "#08519c"])
        for idx, r in enumerate(bdf.itertuples(index=False)):
            axes[1].text(idx, r.units_per_min + 0.8, f"{r.units_per_min:.2f} u/min\n({r.peak_aggregate_rss_mb/1024:.2f} GB)", ha="center", fontsize=8)
        axes[1].set_ylim(0, 56)
        axes[1].set_ylabel("Rendimiento sostenido (unidades/min)")
        axes[1].set_title("Benchmark SMT separado (48 unidades × 25 trials)")
        axes[1].grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out["P09"] = _save_fig(fig, fig_dir, "figP09_rendimiento_vs_coste")

    # Fig P10: Síntesis visual de criterios R3 frente al desempeño en bibliotecas
    fig, ax = plt.subplots(figsize=(10.0, 3.8))
    labels_c = [
        "R3 C1: Métrica externa\n(18/28 vs Mejor_Bib ex post)",
        "R3 C2: 3 pruebas calibradas\n(Holm=2/15; SPA/MCS mixtos)",
        "R3 C3: Tamaño del efecto\n(2/15 con dz <= -0.20)",
        "R3 C4: Alcance / externos\n(sin datasets externos)",
        "Comparación descriptiva (bibliotecas)\n(#1 en 10 modelos; 23-28/28 1-vs-1)",
    ]
    scores = [18 / 28, 2 / 15, 2 / 15, 0.0, 1.0]
    cols10 = ["#d95f02", "#d95f02", "#d95f02", "#999999", "#1b7837"]
    y10 = np.arange(len(labels_c))
    ax.barh(y10, [s * 100.0 for s in scores], color=cols10)
    ax.set_yticks(y10)
    ax.set_yticklabels(labels_c, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    texts = ["NO_CUMPLE (18/28)", "NO_CUMPLE / MIXTO", "NO_CUMPLE (2/15 real)", "PENDIENTE_AUTORIZACION", "RESPALDADO (rango 1.64/10)"]
    for idx, (s, txt) in enumerate(zip(scores, texts)):
        ax.text(s * 100.0 + 1.5, idx, txt, va="center", fontsize=8.5, fontweight="bold")
    ax.set_xlabel("Grado cuantitativo observado (%)")
    ax.set_title("Resumen de criterios R3 frente al desempeño descriptivo en bibliotecas")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out["P10"] = _save_fig(fig, fig_dir, "figP10_sintesis_r3_criterios")

    # --- Intra-dataset & Inter-plant / Inter-group analyses (from persisted OOF predictions + base_planta_dia.csv) ---
    base_csv = root / "base_generacion_clima" / "base_planta_dia.csv"
    if base_csv.exists():
        base_df = pd.read_csv(base_csv, usecols=["code", "grupo", "cap_MW"])
        base_df["code"] = base_df["code"].astype(str)
        plant_meta = base_df.groupby("code", as_index=False).agg(grupo=("grupo", "first"), cap_MW_media=("cap_MW", "mean"))
        plant_meta["B_i"] = plant_meta["cap_MW_media"] * 24000.0

        t_planta = pd.read_csv(oof_dir / "metricas" / "tabla_por_planta.csv")
        t_planta["code"] = t_planta["code"].astype(str)
        d1_et_p = t_planta[t_planta["escenario"] == "D1_espaciotemporal_cinco_variables"].merge(plant_meta, on="code", how="left")

        # Mejor_Bib per plant on D1_espaciotemporal_cinco_variables
        sp_d1 = next(sp for sp in SCENARIOS_28 if sp.scenario_id == "D1_espaciotemporal_cinco_variables")
        bench_d1 = load_official_benchmarks(sp_d1, root=root, exclude_cf_brh=True)
        bench_d1 = bench_d1[bench_d1["metodo"].isin(LIBRARIES_9)].copy()
        bench_d1["_ae"] = (bench_d1[sp_d1.pred_col] - bench_d1[sp_d1.target_col]).abs()
        by_p_m = bench_d1.groupby(["code", "metodo"], as_index=False)["_ae"].mean()
        best_bib_name_d1 = by_p_m.groupby("metodo")["_ae"].mean().idxmin()
        bib_p = by_p_m[by_p_m["metodo"] == best_bib_name_d1][["code", "_ae"]].rename(columns={"_ae": "MAE_kWh_MejorBib"})
        d1_et_p = d1_et_p.merge(bib_p, on="code", how="left")
        d1_et_p["MAE_CF_ASHR"] = d1_et_p["MAE_ASHR"] / d1_et_p["B_i"]
        d1_et_p["MAE_CF_MejorBib"] = d1_et_p["MAE_kWh_MejorBib"] / d1_et_p["B_i"]

        mm_wins_p = t_planta.groupby("code", as_index=False)["diff_vs_MM30"].apply(lambda s: int((s < 0).sum())).rename(columns={"diff_vs_MM30": "victorias_vs_MM30_en_28"})
        intra_planta_df = d1_et_p[["code", "grupo_x", "cap_MW_media", "MAE_ASHR", "MAE_kWh_MejorBib", "MAE_CF_ASHR", "MAE_CF_MejorBib"]].rename(columns={"grupo_x": "grupo", "MAE_ASHR": "MAE_kWh_ASHR"}).merge(mm_wins_p, on="code", how="left")
        atomic_write_csv(intra_planta_df, oof_dir / "metricas" / "intra_por_planta_capacidad.csv")

        # Fig P11: Intra-dataset per plant & installed capacity
        fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
        axes[0].scatter(intra_planta_df["cap_MW_media"], intra_planta_df["MAE_kWh_ASHR"] / 1000.0, s=65, color="#1b7837", label="SM-CFHR-Fixed")
        axes[0].scatter(intra_planta_df["cap_MW_media"], intra_planta_df["MAE_kWh_MejorBib"] / 1000.0, s=55, marker="^", color="#b2182b", alpha=0.8, label=f"Mejor Bib. D1 ({best_bib_name_d1})")
        axes[0].set_xlabel("Capacidad instalada media (MW)")
        axes[0].set_ylabel("MAE absoluto en D1 ET 5-var (MWh/día)")
        axes[0].set_title("Error absoluto vs. capacidad instalada (17 plantas)")
        axes[0].legend(fontsize=8.5)
        axes[0].grid(True, linestyle=":", alpha=0.5)

        ip_s = intra_planta_df.sort_values("MAE_CF_ASHR").reset_index(drop=True)
        x11 = np.arange(len(ip_s))
        axes[1].bar(x11 - 0.18, ip_s["MAE_CF_ASHR"] * 100.0, width=0.36, color="#1b7837", label="SM-CFHR-Fixed")
        axes[1].bar(x11 + 0.18, ip_s["MAE_CF_MejorBib"] * 100.0, width=0.36, color="#b2182b", alpha=0.75, label=best_bib_name_d1)
        axes[1].set_xticks(x11)
        axes[1].set_xticklabels(ip_s["code"], rotation=45, ha="right", fontsize=7.5)
        axes[1].set_ylabel("MAE_CF normalizado (% de capacidad)")
        axes[1].set_title("Error normalizado por planta (elimina efecto escala MW)")
        axes[1].legend(fontsize=8)
        axes[1].grid(axis="y", linestyle=":", alpha=0.5)
        fig.suptitle("Evaluación intra-dataset: desempeño por planta y relación con capacidad", fontsize=11.5)
        fig.tight_layout()
        out["P11"] = _save_fig(fig, fig_dir, "figP11_intra_plantas_capacidad")

        # Fig P12: Intra-dataset temporal blocks (D1/D2/D3) + Weather & Recursive/Direct impact
        df_d1_et = pd.read_csv(oof_dir / "predicciones" / "D1_espaciotemporal_cinco_variables.csv")
        df_d1_et["code"] = df_d1_et["code"].astype(str)
        df_d1_et = df_d1_et.merge(plant_meta[["code", "B_i"]], on="code", how="left")
        df_d1_et["ae_cf"] = np.abs(df_d1_et["prediccion_kwh"] - df_d1_et["y_kwh"]) / df_d1_et["B_i"]
        blk_agg = df_d1_et.groupby("bloque", as_index=False)["ae_cf"].agg(["mean", "median"]).reset_index()

        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
        axes[0].bar(blk_agg["bloque"], blk_agg["mean"] * 100.0, color=["#6baed6", "#3182bd", "#08519c"])
        for idx, r in enumerate(blk_agg.itertuples(index=False)):
            axes[0].text(idx, r.mean * 100.0 + 0.08, f"{r.mean*100:.2f}%", ha="center", fontsize=8.5)
        axes[0].set_ylabel("MAE_CF medio (% de capacidad)")
        axes[0].set_title("Variación temporal por bloque externo (D1 ET 5-var)")
        axes[0].grid(axis="y", linestyle=":", alpha=0.5)

        # Paired comparison: RecSinClima vs RecGEFSv12 vs Directo/H (ET & IP)
        strat_rows = []
        for h in (7, 14, 21, 28):
            m_rec_nc = float(t28.loc[t28["scenario_id"] == f"RecSinClima_espaciotemporal_{h}d", "MAE_ASHR"].iloc[0])
            m_rec_c = float(t28.loc[t28["scenario_id"] == f"RecGEFSv12_espaciotemporal_{h}d", "MAE_ASHR"].iloc[0])
            m_dir_nc = float(t28.loc[t28["scenario_id"] == f"DirectoSinClima_espaciotemporal_{h}d", "MAE_ASHR"].iloc[0]) / h
            strat_rows.append((h, m_rec_nc / 1000.0, m_rec_c / 1000.0, m_dir_nc / 1000.0))
        st_df = pd.DataFrame(strat_rows, columns=["H", "Rec_SinClima", "Rec_GEFSv12", "Directo_div_H"])
        axes[1].plot(st_df["H"], st_df["Rec_GEFSv12"], marker="o", color="#1b7837", label="Recursivo + GEFSv12 (ET)")
        axes[1].plot(st_df["H"], st_df["Rec_SinClima"], marker="s", color="#2166ac", label="Recursivo sin clima (ET)")
        axes[1].plot(st_df["H"], st_df["Directo_div_H"], marker="^", color="#b2182b", label="Directo sin clima / H (ET)")
        axes[1].set_xticks([7, 14, 21, 28])
        axes[1].set_xlabel("Horizonte H (días)")
        axes[1].set_ylabel("MAE_macro equivalente (MWh/día)")
        axes[1].set_title("Efecto de meteorología y estrategia (Recursivo vs. Directo)")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, linestyle=":", alpha=0.5)
        fig.tight_layout()
        out["P12"] = _save_fig(fig, fig_dir, "figP12_intra_periodos_clima_estrategia")

        # Inter-plant / Inter-group (G=1..7) generalization: ET vs IP
        df_d1_ip = pd.read_csv(oof_dir / "predicciones" / "D1_intraplanta_cinco_variables.csv")
        df_d1_ip["code"] = df_d1_ip["code"].astype(str)
        df_d1_ip = df_d1_ip.merge(plant_meta[["code", "B_i"]], on="code", how="left")
        df_d1_ip["ae_cf"] = np.abs(df_d1_ip["prediccion_kwh"] - df_d1_ip["y_kwh"]) / df_d1_ip["B_i"]

        g_et = df_d1_et.groupby("grupo", as_index=False)["ae_cf"].mean().rename(columns={"ae_cf": "MAE_CF_ET"})
        g_ip = df_d1_ip.groupby("grupo", as_index=False)["ae_cf"].mean().rename(columns={"ae_cf": "MAE_CF_IP"})
        inter_g = g_et.merge(g_ip, on="grupo", how="inner")

        # Library ranks by group in D1 ET 5-var
        b_d1_g = bench_d1.copy() if "grupo" in bench_d1.columns else bench_d1.merge(plant_meta[["code", "grupo"]], on="code", how="left")
        b_d1_g["_ae"] = (b_d1_g["prediccion_kwh"] - b_d1_g["y_kwh"]).abs()
        g_ranks = []
        for grp, sub_b in b_d1_g.groupby("grupo"):
            bm = sub_b.groupby("metodo", as_index=False)["_ae"].mean()
            a_mae = float(df_d1_et.loc[df_d1_et["grupo"] == grp].assign(_ae=np.abs(df_d1_et["prediccion_kwh"] - df_d1_et["y_kwh"]))["_ae"].mean())
            all10_g = pd.concat([pd.DataFrame([{"metodo": "SM-CFHR-Fixed", "_ae": a_mae}]), bm], ignore_index=True).sort_values("_ae").reset_index(drop=True)
            rk = int(all10_g.index[all10_g["metodo"] == "SM-CFHR-Fixed"][0]) + 1
            g_ranks.append({"grupo": int(grp), "rango_en_10_libs_ET": rk})
        inter_g = inter_g.merge(pd.DataFrame(g_ranks), on="grupo", how="left")
        atomic_write_csv(inter_g, oof_dir / "metricas" / "inter_grupo_et_vs_ip.csv")

        # Fig P13: Inter-plant / Inter-group generalization (no external datasets)
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
        xg = np.arange(len(inter_g))
        axes[0].bar(xg - 0.18, inter_g["MAE_CF_ET"] * 100.0, width=0.36, color="#1b7837", label="ET (plantas no vistas)")
        axes[0].bar(xg + 0.18, inter_g["MAE_CF_IP"] * 100.0, width=0.36, color="#762a83", label="IP (misma planta, futuro)")
        axes[0].set_xticks(xg)
        axes[0].set_xticklabels([f"Grupo {int(g)}" for g in inter_g["grupo"]])
        axes[0].set_ylabel("MAE_CF (% de capacidad)")
        axes[0].set_title("Generalización ET vs. IP por grupo geográfico (D+1 5-var)")
        axes[0].legend(fontsize=8.5)
        axes[0].grid(axis="y", linestyle=":", alpha=0.5)

        axes[1].bar(xg, inter_g["rango_en_10_libs_ET"], color="#2166ac")
        axes[1].set_xticks(xg)
        axes[1].set_xticklabels([f"Grupo {int(g)}" for g in inter_g["grupo"]])
        axes[1].set_ylim(0, 10.5)
        axes[1].axhline(1.0, color="#1b7837", linestyle="--", linewidth=0.9, label="Puesto #1 de 10")
        axes[1].set_ylabel("Rango ordinal de SM-CFHR-Fixed (1 = mejor de 10)")
        axes[1].set_title("Estabilidad del ranking frente a 9 bibliotecas por grupo (ET)")
        axes[1].legend(fontsize=8.5)
        axes[1].grid(axis="y", linestyle=":", alpha=0.5)
        fig.suptitle("Transferencia entre plantas y grupos geográficos (panel solar único, sin datasets externos)", fontsize=11)
        fig.tight_layout()
        out["P13"] = _save_fig(fig, fig_dir, "figP13_inter_grupos_et_vs_ip")

        # Fig P14: Error distribution across populations (ET vs IP, with vs without weather)
        df_d1_et_nw = pd.read_csv(oof_dir / "predicciones" / "D1_espaciotemporal_sin_meteorologia.csv")
        df_d1_et_nw["code"] = df_d1_et_nw["code"].astype(str)
        df_d1_et_nw = df_d1_et_nw.merge(plant_meta[["code", "B_i"]], on="code", how="left")
        df_d1_et_nw["ae_cf"] = np.abs(df_d1_et_nw["prediccion_kwh"] - df_d1_et_nw["y_kwh"]) / df_d1_et_nw["B_i"]

        df_d1_ip_nw = pd.read_csv(oof_dir / "predicciones" / "D1_intraplanta_sin_meteorologia.csv")
        df_d1_ip_nw["code"] = df_d1_ip_nw["code"].astype(str)
        df_d1_ip_nw = df_d1_ip_nw.merge(plant_meta[["code", "B_i"]], on="code", how="left")
        df_d1_ip_nw["ae_cf"] = np.abs(df_d1_ip_nw["prediccion_kwh"] - df_d1_ip_nw["y_kwh"]) / df_d1_ip_nw["B_i"]

        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        data_box = [
            df_d1_et["ae_cf"].to_numpy() * 100.0,
            df_d1_ip["ae_cf"].to_numpy() * 100.0,
            df_d1_et_nw["ae_cf"].to_numpy() * 100.0,
            df_d1_ip_nw["ae_cf"].to_numpy() * 100.0,
        ]
        bp = ax.boxplot(data_box, labels=["ET con clima", "IP con clima", "ET sin clima", "IP sin clima"], showfliers=False, patch_artist=True)
        for patch, c in zip(bp["boxes"], ["#1b7837", "#762a83", "#4393c3", "#9970ab"]):
            patch.set_facecolor(c)
            patch.set_alpha(0.65)
        ax.set_ylabel("Error absoluto normalizado |e|/B_i (% de capacidad)")
        ax.set_title("Distribución de errores diarios normalizados entre poblaciones D+1 (sin atípicos extremos)")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        fig.tight_layout()
        out["P14"] = _save_fig(fig, fig_dir, "figP14_distribucion_errores_poblaciones")

    return out


__all__ = ["LIBRARIES_9", "compute_library_only_tables", "generate_pedagogical_figures"]

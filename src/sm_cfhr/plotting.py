"""Publication-grade scientific plotting helpers (PNG 300 dpi + vector PDF)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from ashr_oof_pipeline.visualizacion import (
        _save_fig,
        export_all_figures,
        plot_fig1_comparacion_general,
        plot_fig2_et_vs_ip,
        plot_fig3_horizontes,
        plot_fig4_desempeno_planta,
        plot_fig5_comparaciones_pareadas,
    )
except ImportError:
    from src.ashr_oof_pipeline.visualizacion import (
        _save_fig,
        export_all_figures,
        plot_fig1_comparacion_general,
        plot_fig2_et_vs_ip,
        plot_fig3_horizontes,
        plot_fig4_desempeno_planta,
        plot_fig5_comparaciones_pareadas,
    )


def plot_nested_cv_scheme(audit_df: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """Plot temporal train/validation intervals and causal gaps across inner CV splits."""
    fig, ax = plt.subplots(figsize=(10.5, max(4.0, 0.35 * len(audit_df))))
    tr_s = pd.to_datetime(audit_df["train_min_date"])
    tr_e = pd.to_datetime(audit_df["train_max_date"])
    va_s = pd.to_datetime(audit_df["val_min_date"])
    va_e = pd.to_datetime(audit_df["val_max_date"])
    y = np.arange(len(audit_df))
    for i in y:
        ax.plot([tr_s.iloc[i], tr_e.iloc[i]], [i, i], color="#2166ac", linewidth=5, label="Entrenamiento" if i == 0 else "")
        ax.plot([va_s.iloc[i], va_e.iloc[i]], [i, i], color="#d6604d", linewidth=5, label="Validación" if i == 0 else "")
    labels = [f"{r.scenario_id[:16]} | s{r.split_idx} (gap={r.effective_gap_days}d)" for r in audit_df.itertuples(index=False)]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Fecha")
    ax.set_title("Esquema de particiones causales en Validación Cruzada Anidada")
    ax.legend(fontsize=8)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return _save_fig(fig, out_dir, "fig06_nested_cv_esquema")


def plot_optuna_convergence(trials_df: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """Plot Optuna TPE best-so-far trajectory across trials per study (or median/IQR if >12 studies)."""
    df = trials_df[trials_df["trial_valid"]].copy() if "trial_valid" in trials_df.columns else trials_df.copy()
    group_cols = ["scenario_id", "outer_fold"] if "scenario_id" in df.columns else ["outer_fold"]
    n_studies = int(df.groupby(group_cols, sort=False).ngroups)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    if n_studies <= 12:
        for key, g in df.groupby(group_cols, sort=False):
            gs = g.sort_values("trial_number").reset_index(drop=True)
            lbl = f"{key[0][:14]} | {key[1]}" if isinstance(key, tuple) else f"Fold {key}"
            ax.plot(np.arange(1, len(gs) + 1), gs["best_so_far_kwh"] / 1000.0, marker="o", markersize=3, label=lbl)
        ax.set_xlabel("Trial válido Optuna TPE")
        ax.set_ylabel("Mejor MAE interno acumulado (MWh/día)")
        ax.set_title(f"Convergencia Optuna TPE (N={n_studies} estudios)")
        ax.legend(fontsize=7.5)
    else:
        rel_rows = []
        for _, g in df.groupby(group_cols, sort=False):
            gs = g.sort_values("trial_number").reset_index(drop=True)
            init = float(gs["best_so_far_kwh"].iloc[0])
            if init > 0:
                rel = 100.0 * (gs["best_so_far_kwh"].to_numpy(dtype=float) / init - 1.0)
                for idx, val in enumerate(rel, start=1):
                    rel_rows.append((idx, float(val)))
        rel_df = pd.DataFrame(rel_rows, columns=["trial_idx", "rel_pct"])
        agg = rel_df.groupby("trial_idx")["rel_pct"].agg(
            med="median",
            p25=lambda s: float(np.quantile(s, 0.25)),
            p75=lambda s: float(np.quantile(s, 0.75)),
        ).reset_index()
        ax.plot(agg["trial_idx"], agg["med"], marker="o", markersize=3.5, color="#2166ac", label="Mediana (N=" + f"{n_studies:,} estudios)")
        ax.fill_between(agg["trial_idx"], agg["p25"], agg["p75"], color="#2166ac", alpha=0.22, label="Rango intercuartílico (p25–p75)")
        ax.axhline(0.0, color="#666666", linestyle="--", linewidth=0.9)
        ax.set_xlabel("Número de trial válido Optuna TPE (presupuesto oficial = 25)")
        ax.set_ylabel("Mejora relativa acumulada vs. trial 1 (%)")
        ax.set_title(f"Convergencia Optuna TPE en {n_studies:,} estudios (mediana e IQR)")
        ax.legend(fontsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    return _save_fig(fig, out_dir, "fig07_optuna_convergencia")


def plot_hp_sensitivity(sens_df: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """Plot 1D hyperparameter sensitivity curves for delta, alpha_lin, and alpha_meteo."""
    params = list(sens_df["varied_param"].unique())
    fig, axes = plt.subplots(1, len(params), figsize=(4.5 * max(1, len(params)), 4.0))
    if len(params) == 1:
        axes = [axes]
    for ax, p in zip(axes, params):
        sub = sens_df[sens_df["varied_param"] == p].sort_values("param_value")
        ax.plot(sub["param_value"], sub["mae"], marker="o", color="#2166ac")
        if p in ("alpha_lin", "alpha_meteo"):
            ax.set_xscale("log")
        ax.set_xlabel(p)
        ax.set_ylabel("MAE (CF)")
        ax.set_title(f"Sensibilidad a {p}")
        ax.grid(True, linestyle=":", alpha=0.5)
    fig.suptitle("Sensibilidad unidimensional de hiperparámetros de SM-CFHR")
    fig.tight_layout()
    return _save_fig(fig, out_dir, "fig08_sensibilidad_hp")


def plot_ablation_bar(abl_df: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """Plot controlled ablation MAE across SM-CFHR variants."""
    s = abl_df.sort_values("mae_kwh")
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.barh(np.arange(len(s)), s["mae_kwh"] / 1000.0, color="#4393c3")
    ax.set_yticks(np.arange(len(s)))
    ax.set_yticklabels(s["ablation_variant"], fontsize=8.5)
    ax.set_xlabel("MAE (MWh/día)")
    ax.set_title("Estudio de ablación controlada de componentes de SM-CFHR")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return _save_fig(fig, out_dir, "fig09_ablacion")


def plot_learning_and_scalability(lc_df: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """Plot learning curve (MAE vs fraction) and fit-time scalability."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    s = lc_df.sort_values("fraction")
    axes[0].plot(s["fraction"] * 100.0, s["mae"], marker="o", color="#1b7837")
    axes[0].set_xlabel("Fracción de entrenamiento (%)")
    axes[0].set_ylabel("MAE (CF)")
    axes[0].set_title("Curva de aprendizaje")
    axes[0].grid(True, linestyle=":", alpha=0.5)

    axes[1].plot(s["n_train"], s["fit_time_s"], marker="s", color="#762a83")
    axes[1].set_xlabel("Número de observaciones (n)")
    axes[1].set_ylabel("Tiempo de ajuste (s)")
    axes[1].set_title("Escalabilidad computacional")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    return _save_fig(fig, out_dir, "fig10_aprendizaje_y_escalabilidad")


def plot_robustness_stress(stress_df: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """Plot MAE degradation under noise and outlier contamination (Huber vs L2)."""
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for mdl, g in stress_df.groupby("model"):
        gs = g.sort_values("outlier_frac")
        ax.plot(gs["outlier_frac"] * 100.0, gs["mae"], marker="o", label=mdl)
    ax.set_xlabel("Fracción de valores atípicos (%)")
    ax.set_ylabel("MAE en validación limpia (CF)")
    ax.set_title("Robustez ante contaminación con ruido y atípicos")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    return _save_fig(fig, out_dir, "fig11_robustez_atipicos")


def plot_conformal_coverage(conf_summary_df: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """Plot nominal vs empirical conformal interval coverage."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(conf_summary_df))
    w = 0.35
    ax.bar(x - w / 2, conf_summary_df["nominal_coverage"] * 100.0, width=w, label="Cobertura nominal (%)", color="#999999")
    ax.bar(x + w / 2, conf_summary_df["empirical_coverage"] * 100.0, width=w, label="Cobertura empírica (%)", color="#2166ac")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v * 100)}%" for v in conf_summary_df["nominal_coverage"]])
    ax.set_ylabel("Cobertura (%)")
    ax.set_title("Calibración de intervalos conformes temporales")
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return _save_fig(fig, out_dir, "fig12_incertidumbre_conforme")


def plot_feature_importance(global_imp_df: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """Plot global additive contribution importance across predictors."""
    s = global_imp_df.sort_values("mean_abs_contrib", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.3 * len(s))))
    ax.barh(np.arange(len(s)), s["mean_abs_contrib"], color="#2166ac")
    ax.set_yticks(np.arange(len(s)))
    ax.set_yticklabels(s["feature"], fontsize=8.5)
    ax.set_xlabel("Contribución media absoluta |w_j * (z_j - mean(z_j))| (CF)")
    ax.set_title("Importancia global de variables en SM-CFHR")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return _save_fig(fig, out_dir, "fig13_importancia_variables")


__all__ = [
    "export_all_figures",
    "plot_ablation_bar",
    "plot_conformal_coverage",
    "plot_feature_importance",
    "plot_fig1_comparacion_general",
    "plot_fig2_et_vs_ip",
    "plot_fig3_horizontes",
    "plot_fig4_desempeno_planta",
    "plot_fig5_comparaciones_pareadas",
    "plot_hp_sensitivity",
    "plot_learning_and_scalability",
    "plot_nested_cv_scheme",
    "plot_optuna_convergence",
    "plot_robustness_stress",
]

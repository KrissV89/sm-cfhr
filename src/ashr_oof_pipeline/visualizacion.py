"""Publication-grade scientific figures for ASHR OOF evaluation (PNG + PDF)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_fig1_comparacion_general(tabla_28: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """Fig 1: ASHR MAE macro vs best library (separating daily kWh/dia and cumulative kWh)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    subsets = [
        ("Escenarios diarios y recursivos (MWh/día)", tabla_28[tabla_28["unit"] == "kWh/dia"]),
        ("Escenarios directos acumulados (MWh)", tabla_28[tabla_28["unit"] == "kWh_total"]),
    ]
    for ax, (title, sub) in zip(axes, subsets):
        s = sub.sort_values("pct_vs_MejorLib")
        y_pos = np.arange(len(s))
        h = 0.36
        ax.barh(y_pos - h / 2, s["MAE_ASHR"] / 1000.0, height=h, label="ASHR (SM-CFHR)", color="#2166ac")
        ax.barh(y_pos + h / 2, s["MAE_Mejor_Bib"] / 1000.0, height=h, label="Mejor biblioteca", color="#999999")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(s["scenario_id"], fontsize=8)
        ax.set_xlabel("MAE macro (MWh)")
        ax.set_title(title)
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)
    fig.suptitle("Figura 1. Comparación general de ASHR frente al mejor benchmark de biblioteca")
    fig.tight_layout()
    return _save_fig(fig, out_dir, "fig01_comparacion_general")


def plot_fig2_et_vs_ip(tabla_28: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """Fig 2: Spatiotemporal (ET) vs Intraplant (IP) percentage difference vs best library."""
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    piv = tabla_28.pivot_table(index="suite", columns="task", values="pct_vs_MejorLib", aggfunc="mean")
    suites = list(piv.index)
    x = np.arange(len(suites))
    w = 0.35
    ax.bar(x - w / 2, piv.get("espaciotemporal", np.zeros(len(suites))), width=w, label="Espaciotemporal (ET)", color="#1b7837")
    ax.bar(x + w / 2, piv.get("intraplanta", np.zeros(len(suites))), width=w, label="Intraplanta (IP)", color="#762a83")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(suites, fontsize=8.5)
    ax.set_ylabel("Diferencia media vs. mejor biblioteca (%)")
    ax.set_title("Figura 2. Generalización espaciotemporal (ET) frente a intraplanta (IP)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    return _save_fig(fig, out_dir, "fig02_et_vs_ip")


def plot_fig3_horizontes(tabla_28: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """Fig 3: Performance by horizon (D+1, 7d, 14d, 21d, 28d) distinguishing direct vs recursive."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=False)
    for ax, (title, sub) in zip(
        axes,
        [
            ("Pronóstico diario / recursivo (MWh/día)", tabla_28[tabla_28["unit"] == "kWh/dia"]),
            ("Pronóstico directo acumulado (MWh)", tabla_28[tabla_28["unit"] == "kWh_total"]),
        ],
    ):
        for (suite, task), g in sub.groupby(["suite", "task"]):
            gs = g.sort_values("horizon_days")
            ax.plot(gs["horizon_days"], gs["MAE_ASHR"] / 1000.0, marker="o", label=f"{suite} ({task[:2].upper()})")
        ax.set_xlabel("Horizonte (días)")
        ax.set_ylabel("MAE macro (MWh)")
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=7.5)
    fig.suptitle("Figura 3. Comportamiento de ASHR por horizonte predictivo")
    fig.tight_layout()
    return _save_fig(fig, out_dir, "fig03_horizontes")


def plot_fig4_desempeno_planta(tabla_planta: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """Fig 4: Per-plant MAE distribution across all 17 plants (without hiding poor plants)."""
    daily_p = tabla_planta[tabla_planta["unit"] == "kWh/dia"].copy()
    if daily_p.empty:
        daily_p = tabla_planta.copy()
    by_plant = daily_p.groupby("code")["MAE_ASHR"].median().sort_values()
    order = list(by_plant.index)
    groups = [daily_p.loc[daily_p["code"] == c, "MAE_ASHR"].to_numpy(float) / 1000.0 for c in order]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.boxplot(groups, tick_labels=order, orientation="horizontal")
    ax.set_xlabel("MAE de ASHR (MWh/día)")
    ax.set_ylabel("Planta fotovoltaica (código)")
    ax.set_title("Figura 4. Heterogeneidad del error de ASHR entre las 17 plantas")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return _save_fig(fig, out_dir, "fig04_desempeno_planta")


def plot_fig5_comparaciones_pareadas(tabla_inf: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """Fig 5: Paired mean loss difference vs best library with 95% stationary bootstrap CIs."""
    sub = tabla_inf[tabla_inf["referencia"] == "Mejor_Bib"].copy()
    if sub.empty:
        sub = tabla_inf.copy()
    sub = sub.sort_values("mean_diff")
    y_pos = np.arange(len(sub))
    mean_d = sub["mean_diff"].to_numpy(float) / 1000.0
    err_lo = np.maximum(0.0, mean_d - sub["ci95_lo"].to_numpy(float) / 1000.0)
    err_hi = np.maximum(0.0, sub["ci95_hi"].to_numpy(float) / 1000.0 - mean_d)

    fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.28 * len(sub))))
    ax.errorbar(mean_d, y_pos, xerr=[err_lo, err_hi], fmt="o", color="#2166ac", capsize=3)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    labels = [
        f"{r.escenario} [{'T_eff<2h' if r.t_eff_insufficient else r.estado_estadistico}]"
        for r in sub.itertuples(index=False)
    ]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Diferencia media de pérdida ASHR - Mejor biblioteca (MWh, IC 95 %)")
    ax.set_title("Figura 5. Comparaciones pareadas con intervalos bootstrap estacionarios")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return _save_fig(fig, out_dir, "fig05_comparaciones_pareadas")


def export_all_figures(bundle: dict[str, Any], fig_dir: Path) -> dict[str, tuple[Path, Path]]:
    """Export all 5 publication figures in PNG and PDF formats."""
    return {
        "fig1": plot_fig1_comparacion_general(bundle["tabla_28_escenarios"], fig_dir),
        "fig2": plot_fig2_et_vs_ip(bundle["tabla_28_escenarios"], fig_dir),
        "fig3": plot_fig3_horizontes(bundle["tabla_28_escenarios"], fig_dir),
        "fig4": plot_fig4_desempeno_planta(bundle["tabla_por_planta"], fig_dir),
        "fig5": plot_fig5_comparaciones_pareadas(bundle["tabla_inferencia"], fig_dir),
    }


def build_synthetic_demo_data() -> dict[str, pd.DataFrame]:
    """Generate synthetic demonstration data strictly for unit-testing visualization."""
    rng = np.random.default_rng(2026)
    summary = pd.DataFrame({
        "scenario_id": [f"Sim_{i}" for i in range(1, 9)],
        "suite": ["Rec sin clima (43-46)"] * 4 + ["Directo sin clima (48-51)"] * 4,
        "task": ["espaciotemporal", "intraplanta"] * 4,
        "horizon_days": [7, 7, 14, 14, 7, 7, 14, 14],
        "unit": ["kWh/dia"] * 4 + ["kWh_total"] * 4,
        "MAE_ASHR": rng.uniform(58000, 88000, size=8),
        "MAE_Mejor_Bib": rng.uniform(60000, 90000, size=8),
        "pct_vs_MejorLib": rng.uniform(-22.0, 2.0, size=8),
    })
    plant = pd.DataFrame({
        "escenario": np.repeat(["Sim_ET", "Sim_IP"], 17),
        "unit": ["kWh/dia"] * 34,
        "code": [f"P{i}" for i in range(17)] * 2,
        "MAE_ASHR": rng.uniform(15000, 95000, size=34),
    })
    inf = pd.DataFrame({
        "escenario": [f"Sim_{i}" for i in range(1, 5)],
        "referencia": ["Mejor_Bib"] * 4,
        "t_eff_insufficient": [False, False, True, True],
        "estado_estadistico": ["SIGNIFICATIVO_HOLM", "NO_SIGNIFICATIVO", "T_EFF_INSUFICIENTE", "T_EFF_INSUFICIENTE"],
        "mean_diff": [-12000.0, -1500.0, -500.0, 300.0],
        "ci95_lo": [-16000.0, -2800.0, -1400.0, -400.0],
        "ci95_hi": [-8000.0, -200.0, 400.0, 1000.0],
    })
    return {"summary": summary, "plant": plant, "inf": inf}

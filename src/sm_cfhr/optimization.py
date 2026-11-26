"""Optuna TPE optimization wrappers and trial convergence summaries."""

from __future__ import annotations

import pandas as pd

try:
    from ashr_oof_pipeline.optimizacion_hiperparametros import (
        nested_cv_optimize_and_refit,
        optimize_hyperparams_tpe,
    )
except ImportError:
    from src.ashr_oof_pipeline.optimizacion_hiperparametros import (
        nested_cv_optimize_and_refit,
        optimize_hyperparams_tpe,
    )


def summarize_hpo_trials(trials_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize Optuna TPE convergence, selected hyperparameters, and timing per study."""
    has_sc = trials_df is not None and not trials_df.empty and "scenario_id" in trials_df.columns
    base_cols = (["scenario_id"] if has_sc else []) + [
        "outer_fold",
        "n_rows_logged",
        "n_trials",
        "n_valid_trials",
        "best_inner_mae_kwh",
        "best_delta",
        "best_alpha_lin",
        "best_alpha_meteo",
        "total_wall_time_s",
        "max_peak_memory_kb",
    ]
    if trials_df is None or trials_df.empty:
        return pd.DataFrame(columns=base_cols)
    group_cols = ["scenario_id", "outer_fold"] if has_sc else ["outer_fold"]
    rows = []
    for key, g in trials_df.groupby(group_cols, sort=False):
        sc_id, fold_id = key if has_sc else (None, key)
        valid = g[g["trial_valid"]] if "trial_valid" in g.columns else g
        best_row = valid.sort_values("inner_mae_macro_kwh").iloc[0] if not valid.empty else g.iloc[0]
        row = {
            "outer_fold": fold_id,
            "n_rows_logged": int(len(g)),
            "n_trials": int(len(valid)) if "trial_valid" in g.columns else int(len(g)),
            "n_valid_trials": int(len(valid)),
            "best_inner_mae_kwh": float(best_row["inner_mae_macro_kwh"]),
            "best_delta": float(best_row["delta"]),
            "best_alpha_lin": float(best_row["alpha_lin"]),
            "best_alpha_meteo": float(best_row["alpha_meteo"]),
            "total_wall_time_s": float(g["wall_time_s"].sum()),
            "max_peak_memory_kb": float(g["peak_memory_kb"].max()),
        }
        if has_sc:
            row = {"scenario_id": sc_id, **row}
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "nested_cv_optimize_and_refit",
    "optimize_hyperparams_tpe",
    "summarize_hpo_trials",
]

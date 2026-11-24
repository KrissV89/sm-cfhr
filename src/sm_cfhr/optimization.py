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
    """Summarize Optuna TPE convergence, selected hyperparameters, and timing per outer fold."""
    if trials_df is None or trials_df.empty:
        return pd.DataFrame(
            columns=[
                "outer_fold",
                "n_trials",
                "n_valid_trials",
                "best_inner_mae_kwh",
                "best_delta",
                "best_alpha_lin",
                "best_alpha_meteo",
                "total_wall_time_s",
                "max_peak_memory_kb",
            ]
        )
    rows = []
    for fold_id, g in trials_df.groupby("outer_fold", sort=False):
        valid = g[g["trial_valid"]] if "trial_valid" in g.columns else g
        best_row = valid.sort_values("inner_mae_macro_kwh").iloc[0] if not valid.empty else g.iloc[0]
        rows.append(
            {
                "outer_fold": fold_id,
                "n_trials": int(len(g)),
                "n_valid_trials": int(len(valid)),
                "best_inner_mae_kwh": float(best_row["inner_mae_macro_kwh"]),
                "best_delta": float(best_row["delta"]),
                "best_alpha_lin": float(best_row["alpha_lin"]),
                "best_alpha_meteo": float(best_row["alpha_meteo"]),
                "total_wall_time_s": float(g["wall_time_s"].sum()),
                "max_peak_memory_kb": float(g["peak_memory_kb"].max()),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "nested_cv_optimize_and_refit",
    "optimize_hyperparams_tpe",
    "summarize_hpo_trials",
]

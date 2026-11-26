"""Causal temporal/spatial validation helpers and effective-design diagnostics."""

from __future__ import annotations

from typing import Any

import pandas as pd

try:
    from ashr_oof_pipeline.config import SCENARIOS_28, verify_benchmark_inventory
    from ashr_oof_pipeline.optimizacion_hiperparametros import (
        audit_inner_splits,
        build_causal_inner_splits,
    )
except ImportError:
    from src.ashr_oof_pipeline.config import SCENARIOS_28, verify_benchmark_inventory
    from src.ashr_oof_pipeline.optimizacion_hiperparametros import (
        audit_inner_splits,
        build_causal_inner_splits,
    )


def analyze_effective_design(bundle: dict[str, Any] | None) -> pd.DataFrame:
    """Summarize effective sample size T_eff, horizon overlap, and power constraints."""
    if bundle is None:
        return pd.DataFrame(
            columns=[
                "escenario",
                "suite",
                "task",
                "horizon_days",
                "T_eff",
                "t_eff_insufficient",
                "nota_potencia",
            ]
        )
    t_inf = bundle["tabla_inferencia"]
    sub = t_inf[t_inf["referencia"] == "Mejor_Bib"].copy()
    if sub.empty:
        return sub
    sub["nota_potencia"] = sub.apply(
        lambda r: (
            "T_eff < 2*H: aproximación asintótica poco fiable"
            if bool(r["t_eff_insufficient"])
            else "T_eff >= 2*H; considerar dependencia espacial entre 17 plantas (Nadeau-Bengio no aplica directamente sin repeticiones independientes)"
        ),
        axis=1,
    )
    return sub[
        ["escenario", "suite", "task", "horizon_days", "T_eff", "t_eff_insufficient", "nota_potencia"]
    ].reset_index(drop=True)


__all__ = [
    "SCENARIOS_28",
    "analyze_effective_design",
    "audit_inner_splits",
    "build_causal_inner_splits",
    "verify_benchmark_inventory",
]

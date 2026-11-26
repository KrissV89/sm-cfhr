"""Automated tests for notebook audit fixes, Optuna grouping, alignment, and post-OOF splits."""

from __future__ import annotations

import numpy as np
import pandas as pd

import sm_cfhr
from ashr_oof_pipeline.evaluacion_estadistica import compute_scenario_metrics
from sm_cfhr.inference import evaluate_r3_superiority
from sm_cfhr.optimization import summarize_hpo_trials


def test_evaluate_r3_uses_cohen_dz_pareado_with_sign() -> None:
    t28 = pd.DataFrame([{"pct_vs_MejorLib": -10.0}, {"pct_vs_MejorLib": 2.0}])
    t_inf = pd.DataFrame(
        [
            {"referencia": "Mejor_Bib", "t_eff_insufficient": False, "p_holm": 1e-5, "cohen_dz_pareado": -0.89},
            {"referencia": "Mejor_Bib", "t_eff_insufficient": False, "p_holm": 0.5, "cohen_dz_pareado": 0.30},
        ]
    )
    t_mcs = pd.DataFrame([{"ashr_in_mcs_90": True, "n_survivors": 1}, {"ashr_in_mcs_90": True, "n_survivors": 5}])
    r3 = evaluate_r3_superiority({"tabla_28_escenarios": t28, "tabla_inferencia": t_inf, "tabla_mcs": t_mcs})
    c3 = r3.loc[r3["condicion"] == "Condicion_3_Tamano_Efecto"].iloc[0]
    assert c3["resultado"] == "1/2"


def test_package_version_matches_pyproject() -> None:
    assert sm_cfhr.__version__ == "1.0.3"


def test_summarize_hpo_trials_groups_by_scenario_and_outer_fold() -> None:
    rows = []
    for sc in ("sc_A", "sc_B"):
        for t in range(25):
            rows.append(
                {
                    "scenario_id": sc,
                    "outer_fold": "D1_1",
                    "trial_number": t,
                    "delta": 1.35,
                    "alpha_lin": 0.01,
                    "alpha_meteo": 0.10,
                    "inner_mae_macro_kwh": 50000.0 - t * 10.0,
                    "trial_valid": True,
                    "wall_time_s": 0.1,
                    "peak_memory_kb": 1024.0,
                }
            )
    df = pd.DataFrame(rows)
    summ = summarize_hpo_trials(df)
    assert len(summ) == 2
    assert set(summ["scenario_id"]) == {"sc_A", "sc_B"}
    assert (summ["n_trials"] == 25).all()
    assert (summ["n_valid_trials"] == 25).all()


def test_compute_scenario_metrics_macro_differs_from_micro_when_unbalanced() -> None:
    df = pd.DataFrame(
        {
            "code": ["P1"] * 100 + ["P2"] * 2,
            "y_pred": [0.0] * 100 + [0.0, 0.0],
            "y_true": [1.0] * 100 + [100.0, 100.0],
        }
    )
    m = compute_scenario_metrics(df, "y_pred", "y_true")
    mae_micro = float(np.mean(np.abs(df["y_pred"] - df["y_true"])))
    assert np.isclose(m["MAE_macro"], 50.5)
    assert not np.isclose(m["MAE_macro"], mae_micro)

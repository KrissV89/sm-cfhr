"""Automated verification tests for corrected statistical inference (DM-HLN-HAC, MCS, SPA, Bootstrap)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from arch.bootstrap import MCS as ArchMCS
from arch.bootstrap import SPA as ArchSPA

from src.ashr_oof_pipeline.evaluacion_estadistica import model_confidence_set
from src.sm_cfhr.inference import spa_test
from src.sm_cfhr.inference_v2 import (
    default_block_length,
    dm_hln_hac_v2,
    infer_series_step_and_overlap,
    mcs_t_max_v2,
    spa_test_v2,
    stationary_bootstrap_ci_v2,
)


def _ar1(n: int, phi: float, mean: float = 0.0, sd: float = 1.0, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sd * np.sqrt(max(1e-6, 1.0 - phi ** 2)), size=n)
    x = np.empty(n, dtype=float)
    x[0] = mean + rng.normal(0.0, sd)
    for t in range(1, n):
        x[t] = mean + phi * (x[t - 1] - mean) + eps[t]
    return x


# 1. Step & overlap detection
def test_step_and_overlap_detection() -> None:
    d1_dates = pd.date_range("2025-11-01", periods=100, freq="D")
    info_d1 = infer_series_step_and_overlap(d1_dates, horizon_days=1)
    assert info_d1["h_overlap"] == 1
    assert info_d1["overlapping"] is False

    non_ov_28 = pd.date_range("2024-05-26", periods=31, freq="28D")
    info_28 = infer_series_step_and_overlap(non_ov_28, horizon_days=28)
    assert info_28["median_step_days"] == pytest.approx(28.0)
    assert info_28["h_overlap"] == 1
    assert info_28["overlapping"] is False

    ov_7 = pd.date_range("2025-01-01", periods=120, freq="D")
    info_ov7 = infer_series_step_and_overlap(ov_7, horizon_days=7)
    assert info_ov7["h_overlap"] == 7
    assert info_ov7["overlapping"] is True


# 2. DM-HLN-HAC: 1-step, non-overlapping multi-step, overlapping multi-step
def test_dm_hln_hac_overlap_and_direction() -> None:
    rng = np.random.default_rng(2026)
    d_a_better = rng.normal(-1.5, 1.0, size=31)
    res_non_ov = dm_hln_hac_v2(d_a_better, h_overlap=1)
    assert res_non_ov["hln_factor"] > 0.95
    assert res_non_ov["dm_hln"] < -4.0
    assert res_non_ov["p_dm_hln"] < 1e-3

    # Overlapping 7-day daily forecasts must use L >= 6
    d_ov7 = np.convolve(rng.normal(0.0, 1.0, size=200), np.ones(7) / np.sqrt(7.0), mode="valid")
    res_ov7 = dm_hln_hac_v2(d_ov7, h_overlap=7)
    assert res_ov7["L_hac"] >= 6
    assert res_ov7["T_eff"] < res_ov7["T"]

    # Superiority of B (d > 0)
    d_b_better = rng.normal(1.2, 1.0, size=60)
    res_b = dm_hln_hac_v2(d_b_better, h_overlap=1)
    assert res_b["dm_hln"] > 4.0
    assert res_b["p_dm_hln"] < 1e-3


# 3. DM-HLN-HAC: equivalent models & autocorrelation calibration
def test_dm_hln_hac_null_calibration_and_ar1() -> None:
    rejections_iid = 0
    rejections_ar1 = 0
    n_sim = 200
    for s in range(n_sim):
        rng = np.random.default_rng(5000 + s)
        d_iid = rng.normal(0.0, 1.0, size=80)
        if dm_hln_hac_v2(d_iid, h_overlap=1)["p_dm_hln"] < 0.05:
            rejections_iid += 1
        d_ar = _ar1(120, phi=0.6, mean=0.0, sd=1.0, seed=7000 + s)
        r_ar = dm_hln_hac_v2(d_ar, h_overlap=1)
        assert r_ar["T_eff"] < r_ar["T"]
        if r_ar["p_dm_hln"] < 0.05:
            rejections_ar1 += 1
    assert 0.01 <= rejections_iid / n_sim <= 0.11
    assert 0.01 <= rejections_ar1 / n_sim <= 0.14


# 4. Degenerate & short series handling
def test_dm_hln_hac_degenerate_cases() -> None:
    assert np.isnan(dm_hln_hac_v2(np.array([0.1, 0.2, 0.3]), h_overlap=1)["p_dm_hln"])
    res_zero = dm_hln_hac_v2(np.zeros(25), h_overlap=1)
    assert res_zero["dm_hln"] == 0.0
    assert res_zero["p_dm_hln"] == 1.0


# 5. MCS one-sided T_max vs arch.bootstrap.MCS
def test_mcs_dominance_equivalence_and_arch_agreement() -> None:
    rng = np.random.default_rng(42)
    T = 120
    # Model 0 clearly dominates; Models 1, 2 worse; Models 3, 4 much worse
    base = rng.normal(1.0, 0.2, size=(T, 1))
    L = np.column_stack([
        base[:, 0],
        base[:, 0] + 0.35 + rng.normal(0, 0.15, T),
        base[:, 0] + 0.45 + rng.normal(0, 0.15, T),
        base[:, 0] + 1.20 + rng.normal(0, 0.20, T),
        base[:, 0] + 1.50 + rng.normal(0, 0.20, T),
    ])
    names = ["M0_best", "M1", "M2", "M3_bad", "M4_worst"]
    res = mcs_t_max_v2(L, names, alpha=0.10, B=1000, mean_block=4.0, seed=2026)
    assert res["survivors"] == ["M0_best"]

    # Cross-check with arch.bootstrap.MCS (method='max')
    amcs = ArchMCS(L, size=0.10, reps=1000, block_size=4, method="max", bootstrap="stationary", seed=2026)
    amcs.compute()
    arch_included = [names[i] for i in amcs.included]
    assert arch_included == res["survivors"]

    # Wrapper in evaluacion_estadistica.py must also default to one-sided and agree
    res_wrap = model_confidence_set(L, names, alpha=0.10, B=1000, mean_block=4.0, seed=2026)
    assert res_wrap["survivors"] == ["M0_best"]

    # Equivalent models -> all survive
    L_eq = rng.normal(1.0, 0.3, size=(100, 4))
    res_eq = mcs_t_max_v2(L_eq, ["A", "B", "C", "D"], alpha=0.10, B=500, mean_block=4.0, seed=2026)
    assert len(res_eq["survivors"]) == 4


# 6. SPA studentized consistent vs arch.bootstrap.SPA
def test_spa_dominance_null_and_arch_agreement() -> None:
    rng = np.random.default_rng(99)
    T = 150
    bench = _ar1(T, phi=0.4, mean=2.0, sd=0.5, seed=10)
    # Competitor 0 clearly beats benchmark; others worse
    comp_good = bench - 0.35 + rng.normal(0, 0.2, T)
    comp_bad = bench + 0.50 + rng.normal(0, 0.3, T)
    models = np.column_stack([comp_good, comp_bad])

    res_dom = spa_test_v2(bench, models, B=1000, mean_block=4.0, seed=2026)
    assert res_dom["p_spa_lower"] <= res_dom["p_spa_consistent"] <= res_dom["p_spa_upper"]
    assert res_dom["p_spa_consistent"] < 0.01

    aspa = ArchSPA(bench, models, block_size=4, reps=1000, bootstrap="stationary", studentize=True, seed=2026)
    aspa.compute()
    assert abs(aspa.pvalues["consistent"] - res_dom["p_spa_consistent"]) < 0.03

    # Null case: benchmark is at least as good as all competitors
    models_worse = np.column_stack([bench + 0.15 + rng.normal(0, 0.2, T), bench + 0.30 + rng.normal(0, 0.2, T)])
    res_null = spa_test_v2(bench, models_worse, B=1000, mean_block=4.0, seed=2026)
    assert res_null["p_spa_consistent"] > 0.20

    # Existing wrapper spa_test must also return consistent keys
    wrap = spa_test(bench, models, B=500, mean_block=4.0, seed=2026)
    assert wrap["p_spa"] < 0.02
    assert "p_spa_consistent" in wrap


# 7. Stationary bootstrap block length in observation units
def test_stationary_bootstrap_observation_units() -> None:
    assert default_block_length(31) <= 31 / 4.0
    d_short = _ar1(31, phi=0.3, mean=-2.0, sd=0.5, seed=77)
    lo, hi, blk = stationary_bootstrap_ci_v2(d_short, B=1000, seed=2026)
    assert blk <= 7.75
    assert lo < np.mean(d_short) < hi
    assert hi < 0.0

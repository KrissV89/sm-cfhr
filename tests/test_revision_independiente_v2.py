"""Comprehensive automated tests for post-audit independent scientific corrections (v2)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from arch.bootstrap import MCS as ArchMCS
from arch.bootstrap import SPA as ArchSPA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sm_cfhr.inference_v2 import (
    build_macro_aligned_panel,
    classify_teff_reliability,
    dm_hln_hac_v2,
    hansen_politis_romano_variance,
    macro_synchronized_block_bootstrap,
    mcs_t_max_v2,
    spa_test_v2,
)
from src.sm_cfhr.metrics_audit import detect_capacity_changes


def _make_panel_df(
    rows: list[tuple[str, str, float, float, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build synthetic (df_a, df_b) from (code, date, y, pred_a, pred_b)."""
    da = pd.DataFrame(
        [{"code": c, "date": d, "y_kwh": y, "prediccion_kwh": pa} for c, d, y, pa, _ in rows]
    )
    db = pd.DataFrame(
        [{"code": c, "date": d, "y_kwh": y, "prediccion_kwh": pb} for c, d, y, _, pb in rows]
    )
    return da, db


def test_macro_vs_origin_balanced_and_sign_flip() -> None:
    """Balanced panels agree identically; unbalanced panels can flip sign between macro and origin."""
    # 1) Balanced panel: 3 plants, 10 dates each -> macro == micro == origin
    rng = np.random.default_rng(101)
    bal_rows = []
    for p in ("P1", "P2", "P3"):
        for t in range(10):
            y = 100.0
            pa = y + rng.normal(0, 5)
            pb = y + rng.normal(0, 8)
            bal_rows.append((p, f"2025-01-{t+1:02d}", y, pa, pb))
    da, db = _make_panel_df(bal_rows)
    p_bal = build_macro_aligned_panel(da, db, ("code", "date"), "date", "prediccion_kwh", "y_kwh")
    assert p_bal["diff_mae_macro"] == pytest.approx(p_bal["diff_mae_micro"], rel=1e-12, abs=1e-12)
    assert p_bal["diff_mae_macro"] == pytest.approx(p_bal["diff_temporal_origin"], rel=1e-12, abs=1e-12)
    assert float(np.mean(p_bal["z_macro_series"])) == pytest.approx(p_bal["diff_mae_macro"], rel=1e-12, abs=1e-12)
    assert not p_bal["sign_flip_macro_vs_origin"]

    # 2) Unbalanced panel with sign flip:
    # Plant A (20 dates): A loses by +10 on t=0..19 -> mean_A = +10
    # Plant B (2 dates, shared only on t=0,1): A wins by -30 on t=0,1 -> mean_B = -30
    # MAE_macro = (10 - 30)/2 = -10 (favors A)
    # On t=0,1 origin mean = (10 - 30)/2 = -10; on t=2..19 origin mean = +10
    # MAE_origin = (2*(-10) + 18*(10))/20 = +8 (favors B -> SIGN FLIP!)
    unbal_rows = []
    for t in range(20):
        unbal_rows.append(("A", f"2025-02-{t+1:02d}", 100.0, 120.0, 110.0))  # |e_a|=20, |e_b|=10 -> d=+10
    for t in range(2):
        unbal_rows.append(("B", f"2025-02-{t+1:02d}", 100.0, 100.0, 130.0))  # |e_a|=0, |e_b|=30 -> d=-30
    da_u, db_u = _make_panel_df(unbal_rows)
    p_un = build_macro_aligned_panel(da_u, db_u, ("code", "date"), "date", "prediccion_kwh", "y_kwh")
    assert p_un["diff_mae_macro"] == pytest.approx(-10.0, abs=1e-9)
    assert p_un["diff_temporal_origin"] == pytest.approx(8.0, abs=1e-9)
    assert p_un["sign_flip_macro_vs_origin"] is True
    assert float(np.mean(p_un["z_macro_series"])) == pytest.approx(p_un["diff_mae_macro"], rel=1e-12, abs=1e-12)

    bm = macro_synchronized_block_bootstrap(p_un["panel_sum"], p_un["panel_cnt"], mean_block=2.0, B=500, seed=2026)
    assert bm["estimate"] == pytest.approx(-10.0, abs=1e-9)
    assert p_un["imbalance_ratio_obs"] == pytest.approx(10.0, abs=1e-9)


def test_macro_synchronized_bootstrap_preserves_temporal_and_spatial_dependence() -> None:
    """Synchronized block bootstrap yields wider SE than naive i.i.d. under AR(1) + spatial factor."""
    rng = np.random.default_rng(2026)
    T, M = 80, 6
    f = np.zeros(T)
    for t in range(1, T):
        f[t] = 0.75 * f[t - 1] + rng.normal(0, 1.0)
    S = f[:, None] + rng.normal(0, 0.2, size=(T, M))
    C = np.ones((T, M), dtype=float)
    res_blk = macro_synchronized_block_bootstrap(S, C, mean_block=5.0, B=1000, seed=2026)
    res_iid = macro_synchronized_block_bootstrap(S, C, mean_block=1.0, B=1000, seed=2026)
    assert res_blk["se_boot"] > res_iid["se_boot"]
    assert res_blk["pct_draws_all_plants_present"] == pytest.approx(100.0)


def test_dm_hln_degenerate_cases_and_teff_classification() -> None:
    """Verify explicit policy for d_t=0, d_t=-2, d_t=+3 and T_eff reliability labels."""
    r0 = dm_hln_hac_v2(np.zeros(40), h_overlap=1)
    assert r0["degenerate_case"] == "IDENTICAL_ZERO_DIFF"
    assert r0["dm_hln"] == 0.0
    assert r0["p_dm_hln"] == 1.0

    r_neg = dm_hln_hac_v2(np.full(40, -2.0), h_overlap=1)
    assert r_neg["degenerate_case"] == "CONSTANT_NONZERO_DIFF"
    assert np.isneginf(r_neg["dm_hln"])
    assert np.isnan(r_neg["p_dm_hln"])

    r_pos = dm_hln_hac_v2(np.full(40, 3.0), h_overlap=1)
    assert r_pos["degenerate_case"] == "CONSTANT_NONZERO_DIFF"
    assert np.isposinf(r_pos["dm_hln"])
    assert np.isnan(r_pos["p_dm_hln"])

    assert classify_teff_reliability(35.0, 1) == "FIABILIDAD_ASINTOTICA_ESTANDAR"
    assert classify_teff_reliability(25.0, 1) == "MUESTRA_MODERADA_CAUTELA"
    assert classify_teff_reliability(10.0, 1) == "MUESTRA_PEQUENA_SOLO_SENSIBILIDAD"
    assert classify_teff_reliability(1.5, 1) == "NO_EVALUABLE"


def test_spa_reconciliation_with_arch_and_boundary_cases() -> None:
    """Verify why arch 8.0.0 differs (unstudentized) and match both unstudentized & pre-studentized arch."""
    rng = np.random.default_rng(77)
    T, K = 120, 6
    lb = rng.normal(10.0, 1.0, size=T)
    # 1) Clear superiority of competitor 0
    lm_sup = rng.normal(10.4, 1.0, size=(T, K))
    lm_sup[:, 0] = rng.normal(9.3, 1.0, size=T)

    # Unstudentized match against arch.SPA
    v2_un = spa_test_v2(lb, lm_sup, B=2000, mean_block=5.0, seed=2026, studentize=False, clip_zero=False, finite_sample_floor=False)
    ar_def = ArchSPA(lb, lm_sup, block_size=5, reps=2000, bootstrap="stationary", studentize=True, seed=2026)
    ar_def.compute()
    assert abs(v2_un["p_spa_consistent"] - float(ar_def.pvalues["consistent"])) < 0.02

    # Studentized match against arch.SPA fed with pre-studentized loss diffs
    d_raw = lb[:, None] - lm_sup
    omega = np.sqrt(hansen_politis_romano_variance(d_raw, 5.0))
    scale = np.sqrt(T) / omega
    v2_st = spa_test_v2(lb, lm_sup, B=2000, mean_block=5.0, seed=2026, studentize=True, variance_estimator="pr_kernel")
    ar_pre = ArchSPA(np.zeros(T), -(d_raw * scale[None, :]), block_size=5, reps=2000, bootstrap="stationary", studentize=False, seed=2026)
    ar_pre.compute()
    assert abs(v2_st["p_spa_consistent"] - float(ar_pre.pvalues["consistent"])) < 0.02
    assert v2_st["p_spa_consistent"] < 0.05

    # 2) Benchmark strictly beats all competitors -> t_obs == 0 -> p == 1.0
    lm_worse = lb[:, None] + np.abs(rng.normal(2.0, 0.3, size=(T, K)))
    v2_best = spa_test_v2(lb, lm_worse, B=500, mean_block=5.0, seed=2026)
    assert v2_best["spa_stat"] == 0.0
    assert v2_best["p_spa_consistent"] == 1.0


def test_mcs_reconciliation_with_arch() -> None:
    """Verify full survivor set agreement between mcs_t_max_v2 and arch.MCS."""
    rng = np.random.default_rng(123)
    T = 100
    base = rng.normal(5.0, 0.5, size=(T, 2))
    bad = rng.normal(7.0, 0.5, size=(T, 3))
    mat = np.hstack([base, bad])
    names = ["M1", "M2", "B1", "B2", "B3"]
    res_v2 = mcs_t_max_v2(mat, names, alpha=0.10, B=2000, mean_block=4.0, seed=2026)
    am = ArchMCS(mat, size=0.10, reps=2000, block_size=4, method="max", bootstrap="stationary", seed=2026)
    am.compute()
    arch_surv = [names[i] for i in am.included]
    assert set(res_v2["survivors"]) == set(arch_surv)
    assert set(res_v2["survivors"]) == {"M1", "M2"}


def test_capacity_changes_and_frozen_oof_integrity() -> None:
    """Verify capacity-change detection and confirm frozen OOF manifest remains untouched."""
    cap_df = pd.DataFrame([
        {"code": "P1", "date": "2024-01-01", "cap_MW": 10.0},
        {"code": "P1", "date": "2024-06-01", "cap_MW": 20.0},
        {"code": "P2", "date": "2024-01-01", "cap_MW": 50.0},
    ])
    chg = detect_capacity_changes(cap_df)
    assert list(chg["code"]) == ["P1"]

    manifest_path = ROOT / "resultados" / "evaluacion_ashr_oof" / "manifest_ashr_oof.json"
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert m["estatus"] == "OOF_DEFINITIVO_COMPLETO"

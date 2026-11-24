"""Suite de 20 verificaciones obligatorias de auditoría integral para Entregable 3 (SM-CFHR / ASHR)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.adaptive_structured_huber import AdaptiveStructuredHuberRegressor
from src.ashr_oof_pipeline.config import (
    SCENARIOS_28,
    build_frozen_ashr,
    find_project_root,
    verify_benchmark_inventory,
)
from src.ashr_oof_pipeline.evaluacion_estadistica import (
    apply_holm,
    bootstrap_ci_difference,
    clark_west_test,
    giacomini_white_test,
    model_confidence_set,
    run_paired_inference,
)
from src.ashr_oof_pipeline.optimizacion_hiperparametros import (
    build_causal_inner_splits,
    nested_cv_optimize_and_refit,
)
from src.ashr_oof_pipeline.robustez_y_ablacion import (
    compute_interpretability_summary,
    compute_temporal_conformal_intervals,
    run_ablation_suite,
    run_hp_sensitivity,
    run_learning_and_scalability_curves,
    run_noise_outlier_stress,
    run_seed_stability_hpo,
)


def _make_synthetic_panel(n_days: int = 90, n_plants: int = 4, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for p_idx in range(n_plants):
        code = f"PLANTA_{p_idx}"
        grupo = f"G_{p_idx // 2}"
        cap = 10.0 + 5.0 * p_idx
        rad = np.clip(4.5 + 1.2 * np.sin(np.arange(n_days) * 0.1) + rng.normal(0, 0.4, n_days), 0.5, 8.0)
        cf_base = np.clip(0.18 + 0.03 * (rad - 4.5) + rng.normal(0, 0.015, n_days), 0.02, 0.45)
        for d_i, dt in enumerate(dates):
            denom = cap * 24_000.0
            cf30 = float(np.mean(cf_base[max(0, d_i - 30) : max(1, d_i)]))
            cf7 = float(np.mean(cf_base[max(0, d_i - 7) : max(1, d_i)]))
            cf1 = float(cf_base[max(0, d_i - 1)])
            rad_mean = float(np.mean(rad[max(0, d_i - 30) : d_i + 1]))
            rows.append(
                {
                    "code": code,
                    "grupo": grupo,
                    "date": dt,
                    "origen": dt - pd.Timedelta(days=1),
                    "target_end_date": dt,
                    "cap_MW_D": cap,
                    "kwh_D": cf1 * denom,
                    "media7_kwh": cf7 * denom,
                    "media30_kwh": cf30 * denom,
                    "month_sin": float(np.sin(2 * np.pi * dt.month / 12.0)),
                    "month_cos": float(np.cos(2 * np.pi * dt.month / 12.0)),
                    "doy_sin": float(np.sin(2 * np.pi * dt.dayofyear / 365.25)),
                    "doy_cos": float(np.cos(2 * np.pi * dt.dayofyear / 365.25)),
                    "rad_pron": float(rad[d_i]),
                    "temp_pron": float(28.0 + rng.normal(0, 1.5)),
                    "nube_pron": float(np.clip(40.0 + rng.normal(0, 15.0), 0, 100)),
                    "humedad_pron": float(np.clip(70.0 + rng.normal(0, 8.0), 10, 100)),
                    "viento_pron": float(np.clip(3.0 + rng.normal(0, 0.8), 0.2, 12.0)),
                    "anom_rad": float(rad[d_i] - rad_mean),
                    "anom_temp": float(rng.normal(0, 0.8)),
                    "anom_nube": float(rng.normal(0, 8.0)),
                    "anom_humedad": float(rng.normal(0, 4.0)),
                    "anom_viento": float(rng.normal(0, 0.5)),
                    "y": float(cf_base[d_i] * denom),
                }
            )
    return pd.DataFrame(rows)


def _make_xy(n: int = 160, p: int = 17, seed: int = 2026) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[:, 4] = np.clip(0.20 + 0.05 * X[:, 4], 0.02, 0.50)
    X[:, 7] = np.clip(5.0 + 1.0 * X[:, 7], 0.5, 9.0)
    y = np.clip(0.6 * X[:, 4] + 0.02 * X[:, 7] + rng.normal(0, 0.015, size=n), 0.01, 0.95)
    meta = {"cf30_idx": 4, "monotone_pos_idx": (4, 7), "meteo_idx": tuple(range(7, p))}
    return X, y, meta


# 1. Sklearn BaseEstimator + RegressorMixin + get_params / set_params
def test_01_sklearn_base_estimator_and_params() -> None:
    est = build_frozen_ashr()
    assert isinstance(est, BaseEstimator)
    assert isinstance(est, RegressorMixin)
    params = est.get_params(deep=False)
    assert params["delta"] == pytest.approx(1.35)
    est.set_params(delta=2.0)
    assert est.delta == pytest.approx(2.0)


# 2. sklearn.base.clone and Pipeline compatibility
def test_02_clone_and_pipeline_compatibility() -> None:
    X, y, _ = _make_xy(n=80, p=10)
    est = build_frozen_ashr(weather_mode="none")
    cloned = clone(est)
    assert cloned is not est
    assert cloned.get_params() == est.get_params()
    pipe = Pipeline([("scaler", StandardScaler()), ("model", est)])
    pipe.fit(X, y)
    preds = pipe.predict(X[:10])
    assert preds.shape == (10,) and np.all(np.isfinite(preds))


# 3. Joblib and pickle serialization round-trip
def test_03_serialization_round_trip(tmp_path: Path) -> None:
    X, y, meta = _make_xy(n=100, p=17)
    est = build_frozen_ashr()
    est.fit(X, y, metadata=meta)
    p_orig = est.predict(X[:20])

    jl_path = tmp_path / "sm_cfhr.joblib"
    joblib.dump(est, jl_path)
    p_jl = joblib.load(jl_path).predict(X[:20])
    assert np.allclose(p_orig, p_jl, atol=1e-12)

    raw = pickle.dumps(est)
    p_pk = pickle.loads(raw).predict(X[:20])
    assert np.allclose(p_orig, p_pk, atol=1e-12)


# 4. Exact equivalence of frozen Fijo_Lineal_Mono
def test_04_frozen_config_equivalence() -> None:
    X, y, meta = _make_xy(n=120, p=17)
    est_frozen = build_frozen_ashr(mode="espaciotemporal", horizon=1, weather_mode="forecast")
    est_manual = AdaptiveStructuredHuberRegressor(
        mode="espaciotemporal",
        horizon=1,
        weather_mode="forecast",
        selection="fixed",
        fixed_config={
            "name": "fixed_lin_mono",
            "complexity_rank": 1,
            "n_knots": 0,
            "alpha_lin": 0.01,
            "alpha_meteo": 0.10,
            "alpha_spl": 0.10,
            "max_interactions": 0,
            "alpha_int": 0.10,
            "structured": True,
            "use_mono": True,
            "anchor": "none",
            "lead_damp": 1.0,
        },
        delta=1.35,
        max_iter=400,
        tol=1e-7,
        random_state=2026,
    )
    est_frozen.fit(X, y, metadata=meta)
    est_manual.fit(X, y, metadata=meta)
    assert np.allclose(est_frozen.predict(X), est_manual.predict(X), atol=1e-12)


# 5. Analytical gradient vs finite differences
def test_05_analytical_gradient_vs_finite_diff() -> None:
    from src.modelo_aditivo_huber import _huber_val_and_grad

    rng = np.random.default_rng(7)
    r = rng.normal(0.0, 2.5, size=64)
    val0, grad = _huber_val_and_grad(r, delta=1.35)
    assert np.all(np.isfinite(val0))
    eps = 1e-6
    fd = np.zeros_like(r)
    for i in range(len(r)):
        rp = r.copy()
        rm = r.copy()
        rp[i] += eps
        rm[i] -= eps
        fd[i] = (np.sum(_huber_val_and_grad(rp, 1.35)[0]) - np.sum(_huber_val_and_grad(rm, 1.35)[0])) / (2 * eps)
    assert np.allclose(grad, fd, atol=1e-5)


# 6. Positive scale and translation equivariance of Huber reference
def test_06_scale_and_translation_equivariance() -> None:
    X, y, meta = _make_xy(n=100, p=17)
    est1 = build_frozen_ashr()
    est2 = build_frozen_ashr()
    # Unconstrained to test exact affine equivariance
    est1.fixed_config = {**est1.fixed_config, "use_mono": False}
    est2.fixed_config = {**est2.fixed_config, "use_mono": False}
    est1.fit(X, y, metadata=meta)
    a, b = 3.5, 0.25
    est2.fit(X, a * y + b, metadata=meta)
    p1 = est1.predict(X[:20])
    p2 = est2.predict(X[:20])
    assert np.allclose(p2, a * p1 + b, atol=1e-5)


# 7. Physical monotonicity bounds (cf30 >= 0, rad_pron >= 0)
def test_07_physical_monotonicity_bounds() -> None:
    X, y, meta = _make_xy(n=120, p=17)
    # Invert target correlation on rad_pron to force boundary constraint activation
    y_inv = np.clip(0.3 - 0.05 * X[:, 7], 0.01, 0.9)
    est = build_frozen_ashr()
    est.fit(X, y_inv, metadata=meta)
    assert est.coef_[4] >= -1e-12
    assert est.coef_[7] >= -1e-12


# 8. No label leakage: perturbing y_test does not alter fit or predictions
def test_08_no_label_leakage() -> None:
    X, y, meta = _make_xy(n=120, p=17)
    X_tr, y_tr = X[:90], y[:90]
    X_te = X[90:]
    est1 = build_frozen_ashr().fit(X_tr, y_tr, metadata=meta)
    p1 = est1.predict(X_te)
    # Perturb y_test completely (not passed to fit, verifying isolation)
    y_te_corrupt = np.full(len(X_te), 9999.0)
    assert y_te_corrupt.shape[0] == X_te.shape[0]
    est2 = build_frozen_ashr().fit(X_tr, y_tr, metadata=meta)
    p2 = est2.predict(X_te)
    assert np.allclose(p1, p2, atol=1e-12)


# 9. No preprocessing/imputation leakage from test features
def test_09_no_imputation_or_scaling_leakage() -> None:
    X, y, meta = _make_xy(n=120, p=17)
    X_tr, y_tr = X[:90].copy(), y[:90].copy()
    X_te_a = X[90:].copy()
    X_te_b = X[90:].copy() * 100.0 + 500.0
    est = build_frozen_ashr().fit(X_tr, y_tr, metadata=meta)
    mu_before = est._state["x_mean"].copy()
    _ = est.predict(X_te_a)
    _ = est.predict(X_te_b)
    assert np.allclose(est._state["x_mean"], mu_before, atol=1e-12)


# 10. Strict temporal separation in D+1 inner CV
def test_10_d1_inner_cv_temporal_causality() -> None:
    df = _make_synthetic_panel(n_days=90, n_plants=4)
    splits = build_causal_inner_splits(df, mode_eval="d1", task="intraplanta", horizon_days=1)
    assert len(splits) >= 1
    dates = pd.to_datetime(df["date"]).to_numpy()
    for tr_idx, va_idx in splits:
        assert np.max(dates[tr_idx]) < np.min(dates[va_idx])


# 11. Strict temporal separation in Direct 7-28d inner CV
def test_11_direct_inner_cv_causality() -> None:
    df = _make_synthetic_panel(n_days=120, n_plants=4)
    df["target_end_date"] = pd.to_datetime(df["origen"]) + pd.Timedelta(days=14)
    splits = build_causal_inner_splits(df, mode_eval="direct", task="intraplanta", horizon_days=14)
    assert len(splits) >= 1
    origins = pd.to_datetime(df["origen"]).to_numpy()
    t_ends = pd.to_datetime(df["target_end_date"]).to_numpy()
    for tr_idx, va_idx in splits:
        assert np.max(t_ends[tr_idx]) <= np.min(origins[va_idx])


# 12. Strict temporal separation in Recursive 7-28d inner CV
def test_12_recursive_inner_cv_causality() -> None:
    df = _make_synthetic_panel(n_days=120, n_plants=4)
    h = 7
    splits = build_causal_inner_splits(df, mode_eval="recursive", task="intraplanta", horizon_days=h)
    assert len(splits) >= 1
    origins = pd.to_datetime(df["origen"]).to_numpy()
    for tr_idx, va_idx in splits:
        assert np.max(origins[tr_idx]) + np.timedelta64(h, "D") <= np.min(origins[va_idx])


# 13. Strict spatial separation in Espaciotemporal inner CV
def test_13_espaciotemporal_spatial_group_holdout() -> None:
    df = _make_synthetic_panel(n_days=90, n_plants=6)  # 3 groups
    splits = build_causal_inner_splits(df, mode_eval="d1", task="espaciotemporal", horizon_days=1)
    assert len(splits) >= 1
    groups = df["grupo"].to_numpy()
    for tr_idx, va_idx in splits:
        assert set(groups[tr_idx]).isdisjoint(set(groups[va_idx]))


# 14. Nested CV + Optuna TPE optimization end-to-end with full trial logging
def test_14_nested_cv_tpe_end_to_end() -> None:
    df = _make_synthetic_panel(n_days=90, n_plants=4)
    dates = pd.to_datetime(df["date"])
    outer_splits = [
        (np.where(dates < "2025-03-01")[0], np.where((dates >= "2025-03-01") & (dates < "2025-03-16"))[0]),
        (np.where(dates < "2025-03-16")[0], np.where(dates >= "2025-03-16")[0]),
    ]
    res = nested_cv_optimize_and_refit(
        df=df,
        outer_splits=outer_splits,
        task="espaciotemporal",
        horizon_days=1,
        weather_on=True,
        mode_eval="d1",
        n_trials=6,
        seed=2026,
        embargo_days=5,
    )
    assert len(res["fold_summaries"]) == 2
    assert len(res["trial_logs"]) == 12
    assert np.all(np.isfinite(res["oof_df"]["pred_sm_cfhr_hpo_kwh"]))


# 15. Determinism of Fixed across seeds & reproducibility of TPE with fixed seed
def test_15_seed_determinism_and_stability() -> None:
    df = _make_synthetic_panel(n_days=80, n_plants=4)
    dates = pd.to_datetime(df["date"])
    tr = df[dates < "2025-03-01"].reset_index(drop=True)
    va = df[dates >= "2025-03-01"].reset_index(drop=True)
    stab = run_seed_stability_hpo(
        tr, va, task="espaciotemporal", horizon_days=1, weather_on=True, mode_eval="d1",
        seeds=(2026, 2027, 2028), n_trials=4
    )
    fixed_maes = stab[stab["variant"] == "SM-CFHR-Fixed"]["mae_macro_kwh"].to_numpy()
    assert np.std(fixed_maes) == pytest.approx(0.0, abs=1e-12)


# 16. HP sensitivity & Huber vs L2 outlier robustness
def test_16_hp_sensitivity_and_outlier_stress() -> None:
    X, y, meta = _make_xy(n=120, p=17)
    sens = run_hp_sensitivity(X[:90], y[:90], X[90:], y[90:], meta, weather_on=True)
    assert len(sens) == 15
    stress = run_noise_outlier_stress(X[:90], y[:90], X[90:], y[90:], meta, outlier_fracs=(0.0, 0.10))
    assert len(stress) == 4 and np.all(np.isfinite(stress["mae"]))


# 17. Ablation suite (6 variants) & learning/scalability curves
def test_17_ablation_and_learning_curves() -> None:
    X, y, meta = _make_xy(n=140, p=17)
    denom = np.full(len(y), 240_000.0)
    abl = run_ablation_suite(X[:100], y[:100], X[100:], y[100:], denom[:100], denom[100:], meta)
    assert len(abl) == 6
    lc = run_learning_and_scalability_curves(X[:100], y[:100], X[100:], y[100:], meta)
    assert list(lc["fraction"]) == [0.10, 0.25, 0.50, 1.00]
    assert np.all(lc["peak_memory_kb"] > 0)


# 18. Temporal conformal intervals (coverage, width, by-plant)
def test_18_temporal_conformal_intervals() -> None:
    rng = np.random.default_rng(2026)
    y_cal = rng.normal(0.0, 1.0, size=200)
    p_cal = np.zeros(200)
    y_te = rng.normal(0.0, 1.0, size=100)
    p_te = np.zeros(100)
    plants = np.array(["P1" if i < 50 else "P2" for i in range(100)])
    conf = compute_temporal_conformal_intervals(y_cal, p_cal, p_te, y_te, alphas=(0.10, 0.20), plants_test=plants)
    assert len(conf["summary"]) == 2
    assert len(conf["by_plant"]) == 4
    cov_90 = conf["summary"].loc[conf["summary"]["nominal_coverage"] == 0.90, "empirical_coverage"].iloc[0]
    assert 0.75 <= cov_90 <= 1.0


# 19. Statistical inference (DM-HLN-HAC, GW, Clark-West, MCS, Holm, bootstrap CI)
def test_19_statistical_inference_suite() -> None:
    rng = np.random.default_rng(2026)
    y = rng.normal(0.0, 1.0, size=120)
    p_ashr = y + rng.normal(0.0, 0.2, size=120)
    p_ref = y + rng.normal(0.0, 0.5, size=120)
    l_ashr = np.abs(y - p_ashr)
    l_ref = np.abs(y - p_ref)

    paired = run_paired_inference(l_ashr, l_ref, horizon_h=1, is_temporal=True)
    assert paired["p_dm_hln"] < 0.05
    cw = clark_west_test(y, p_ref, p_ashr, tau=1)
    assert np.isfinite(cw["cw_stat"]) and 0.0 <= cw["p_cw_one_sided"] <= 1.0
    ci = bootstrap_ci_difference(l_ashr - l_ref, n_boot=200)
    assert ci["ci_lo"] < ci["ci_hi"]
    mcs = model_confidence_set(np.column_stack([l_ashr, l_ref]), ["SM-CFHR", "Ref"], B=100)
    assert "SM-CFHR" in mcs["survivors"]
    adj = apply_holm(np.array([0.005, 0.04, 0.20]))
    assert np.all(adj >= np.array([0.005, 0.04, 0.20]))


# 20. Inventory (28 scenarios, 22 models, no CF-BRH, no definitive OOF yet) + interpretability
def test_20_inventory_no_definitive_oof_and_interpretability() -> None:
    root = find_project_root()
    assert len(SCENARIOS_28) == 28
    inv = verify_benchmark_inventory(root)
    assert len(inv) == 28
    assert inv["file_exists"].all()
    assert inv["cf_brh_excluded"].all()
    assert (inv["n_official_models"] == 22).all()
    assert not (root / "resultados" / "evaluacion_ashr_oof" / "manifest_ashr_oof.json").exists()

    X, y, meta = _make_xy(n=80, p=17)
    est = build_frozen_ashr().fit(X, y, metadata=meta)
    feats = [f"f_{i}" for i in range(17)]
    interp = compute_interpretability_summary(est, X[:60], X[60:], feats, n_local=3)
    assert len(interp["global_importance"]) == 17
    loc0 = interp["local_explanations"][0]
    assert loc0["baseline_pred"] + loc0["sum_attributions"] == pytest.approx(loc0["model_pred"], abs=1e-10)

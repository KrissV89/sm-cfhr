"""Robustness, hyperparameter sensitivity, ablation, conformal uncertainty, and interpretability."""

from __future__ import annotations

import time
import tracemalloc
from typing import Any

import numpy as np
import pandas as pd

try:
    from adaptive_structured_huber import AdaptiveStructuredHuberRegressor
except ImportError:
    from src.adaptive_structured_huber import AdaptiveStructuredHuberRegressor
from .optimizacion_hiperparametros import (
    _build_ashr_fixed_with_hp,
    optimize_hyperparams_tpe,
)


def run_hp_sensitivity(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    metadata_tr: dict[str, Any],
    weather_on: bool = True,
    deltas: tuple[float, ...] = (0.5, 1.0, 1.35, 2.0, 3.0),
    alphas_lin: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    alphas_meteo: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0),
) -> pd.DataFrame:
    """1D sensitivity sweeps over delta, alpha_lin, and alpha_meteo."""
    rows: list[dict[str, Any]] = []
    base = {"delta": 1.35, "alpha_lin": 0.01, "alpha_meteo": 0.10 if weather_on else 0.01}

    sweeps: list[tuple[str, tuple[float, ...]]] = [
        ("delta", deltas),
        ("alpha_lin", alphas_lin),
    ]
    if weather_on:
        sweeps.append(("alpha_meteo", alphas_meteo))

    for param_name, grid in sweeps:
        for val in grid:
            hp = dict(base)
            hp[param_name] = float(val)
            est = _build_ashr_fixed_with_hp(
                task="espaciotemporal",
                horizon=1,
                weather_on=weather_on,
                delta=hp["delta"],
                alpha_lin=hp["alpha_lin"],
                alpha_meteo=hp["alpha_meteo"],
            )
            est.fit(X_tr, y_tr, metadata=metadata_tr)
            pred = est.predict(X_va)
            rows.append(
                {
                    "varied_param": param_name,
                    "param_value": float(val),
                    "mae": float(np.mean(np.abs(pred - y_va))),
                    "rmse": float(np.sqrt(np.mean((pred - y_va) ** 2))),
                }
            )
    return pd.DataFrame(rows)


def run_seed_stability_hpo(
    train_outer: pd.DataFrame,
    val_outer: pd.DataFrame,
    task: str,
    horizon_days: int,
    weather_on: bool,
    mode_eval: str,
    seeds: tuple[int, ...] = tuple(range(2026, 2036)),
    n_trials: int = 12,
    target_col: str = "y",
) -> pd.DataFrame:
    """Measure determinism of SM-CFHR-Fixed and seed variability of SM-CFHR-HPO across >=10 seeds."""
    from .optimizacion_hiperparametros import (
        _compute_macro_mae_kwh,
        _fit_and_predict_fold,
    )

    rows: list[dict[str, Any]] = []
    for s in seeds:
        y_f, p_f, c_f, _ = _fit_and_predict_fold(
            train_outer,
            val_outer,
            task=task,
            horizon_days=horizon_days,
            weather_on=weather_on,
            delta=1.35,
            alpha_lin=0.01,
            alpha_meteo=0.10 if weather_on else 0.01,
            target_col=target_col,
            seed=int(s),
        )
        rows.append(
            {
                "variant": "SM-CFHR-Fixed",
                "seed": int(s),
                "mae_macro_kwh": _compute_macro_mae_kwh(y_f, p_f, c_f),
                "delta": 1.35,
                "alpha_lin": 0.01,
                "alpha_meteo": 0.10 if weather_on else 0.01,
            }
        )

        best_hp, _ = optimize_hyperparams_tpe(
            train_outer=train_outer,
            task=task,
            horizon_days=horizon_days,
            weather_on=weather_on,
            mode_eval=mode_eval,
            n_trials=n_trials,
            seed=int(s),
            target_col=target_col,
            n_inner_splits=2,
            embargo_days=5,
        )
        y_h, p_h, c_h, _ = _fit_and_predict_fold(
            train_outer,
            val_outer,
            task=task,
            horizon_days=horizon_days,
            weather_on=weather_on,
            delta=best_hp["delta"],
            alpha_lin=best_hp["alpha_lin"],
            alpha_meteo=best_hp["alpha_meteo"],
            target_col=target_col,
            seed=int(s),
        )
        rows.append(
            {
                "variant": "SM-CFHR-HPO",
                "seed": int(s),
                "mae_macro_kwh": _compute_macro_mae_kwh(y_h, p_h, c_h),
                **best_hp,
            }
        )
    return pd.DataFrame(rows)


def run_noise_outlier_stress(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    metadata_tr: dict[str, Any],
    weather_on: bool = True,
    outlier_fracs: tuple[float, ...] = (0.0, 0.05, 0.10),
    noise_std: float = 0.10,
    seed: int = 2026,
) -> pd.DataFrame:
    """Compare Huber (delta=1.35) vs L2-like (delta=1e6) under target outliers and feature noise."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for frac in outlier_fracs:
        y_corrupt = y_tr.copy()
        X_corrupt = X_tr.copy()
        if frac > 0:
            n_out = max(1, int(frac * len(y_tr)))
            out_idx = rng.choice(len(y_tr), size=n_out, replace=False)
            y_corrupt[out_idx] = np.clip(y_corrupt[out_idx] + rng.choice([-0.6, 0.6], size=n_out), 0.0, 1.0)
            X_corrupt = X_corrupt + rng.normal(0.0, noise_std, size=X_corrupt.shape) * np.maximum(
                np.std(X_tr, axis=0), 1e-3
            )

        for label, d_val in (("SM-CFHR_Huber", 1.35), ("Referencia_L2", 1e6)):
            est = _build_ashr_fixed_with_hp(
                task="espaciotemporal",
                horizon=1,
                weather_on=weather_on,
                delta=d_val,
                alpha_lin=0.01,
                alpha_meteo=0.10 if weather_on else 0.01,
            )
            est.fit(X_corrupt, y_corrupt, metadata=metadata_tr)
            pred = est.predict(X_va)
            rows.append(
                {
                    "model": label,
                    "outlier_frac": float(frac),
                    "mae": float(np.mean(np.abs(pred - y_va))),
                    "rmse": float(np.sqrt(np.mean((pred - y_va) ** 2))),
                }
            )
    return pd.DataFrame(rows)


def run_learning_and_scalability_curves(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    metadata_tr: dict[str, Any],
    weather_on: bool = True,
    fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00),
) -> pd.DataFrame:
    """Evaluate sample efficiency, wall-clock fit/predict time, and peak memory across fractions."""
    rows: list[dict[str, Any]] = []
    n_total = len(X_tr)
    for frac in fractions:
        n_sub = max(20, int(round(frac * n_total)))
        X_sub = X_tr[-n_sub:]
        y_sub = y_tr[-n_sub:]
        est = _build_ashr_fixed_with_hp(
            task="espaciotemporal",
            horizon=1,
            weather_on=weather_on,
            delta=1.35,
            alpha_lin=0.01,
            alpha_meteo=0.10 if weather_on else 0.01,
        )
        tracemalloc.start()
        t0 = time.perf_counter()
        est.fit(X_sub, y_sub, metadata=metadata_tr)
        fit_s = float(time.perf_counter() - t0)
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        t1 = time.perf_counter()
        pred = est.predict(X_va)
        pred_s = float(time.perf_counter() - t1)

        rows.append(
            {
                "fraction": float(frac),
                "n_train": int(n_sub),
                "n_features": int(X_sub.shape[1]),
                "mae": float(np.mean(np.abs(pred - y_va))),
                "fit_time_s": fit_s,
                "predict_time_s": pred_s,
                "peak_memory_kb": float(peak_mem / 1024.0),
            }
        )
    return pd.DataFrame(rows)


def run_ablation_suite(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    denom_tr: np.ndarray,
    denom_va: np.ndarray,
    metadata_tr: dict[str, Any],
    weather_on: bool = True,
) -> pd.DataFrame:
    """Controlled 6-variant ablation isolating Huber, structured L2, monotonicity, MAD, and CF."""
    variants = [
        ("SM-CFHR_completo", {"delta": 1.35, "structured": True, "use_mono": True}),
        ("sin_huber_L2", {"delta": 1e6, "structured": True, "use_mono": True}),
        ("sin_l2_estructurada", {"delta": 1.35, "structured": False, "use_mono": True}),
        ("sin_monotonia", {"delta": 1.35, "structured": True, "use_mono": False}),
    ]
    rows: list[dict[str, Any]] = []
    for name, opts in variants:
        cfg = {
            "name": name,
            "complexity_rank": 1,
            "n_knots": 0,
            "alpha_lin": 0.01,
            "alpha_meteo": 0.10 if (opts["structured"] and weather_on) else 0.01,
            "alpha_spl": 0.10,
            "max_interactions": 0,
            "alpha_int": 0.10,
            "structured": opts["structured"],
            "use_mono": opts["use_mono"],
            "anchor": "none",
            "lead_damp": 1.0,
        }
        est = AdaptiveStructuredHuberRegressor(
            mode="espaciotemporal",
            horizon=1,
            weather_mode="forecast" if weather_on else "none",
            selection="fixed",
            fixed_config=cfg,
            delta=opts["delta"],
            max_iter=400,
            tol=1e-7,
            random_state=2026,
        )
        est.fit(X_tr, y_tr, metadata=metadata_tr)
        pred_cf = np.clip(est.predict(X_va), 0.0, 1.0)
        pred_kwh = pred_cf * denom_va
        y_va_kwh = y_va * denom_va
        rows.append(
            {
                "ablation_variant": name,
                "mae_cf": float(np.mean(np.abs(pred_cf - y_va))),
                "mae_kwh": float(np.mean(np.abs(pred_kwh - y_va_kwh))),
            }
        )

    # Variant 5: std scaling instead of MAD scaling on features
    mu_std = np.mean(X_tr, axis=0)
    sd_std = np.maximum(np.std(X_tr, axis=0), 1e-6)
    est_std = _build_ashr_fixed_with_hp("espaciotemporal", 1, weather_on, 1.35, 0.01, 0.10)
    est_std.fit((X_tr - mu_std) / sd_std, y_tr, metadata=metadata_tr)
    pred_std_cf = np.clip(est_std.predict((X_va - mu_std) / sd_std), 0.0, 1.0)
    rows.append(
        {
            "ablation_variant": "escala_std_vs_mad",
            "mae_cf": float(np.mean(np.abs(pred_std_cf - y_va))),
            "mae_kwh": float(np.mean(np.abs(pred_std_cf * denom_va - y_va * denom_va))),
        }
    )

    # Variant 6: raw energy target (kWh) instead of dimensionless capacity factor (CF)
    y_tr_kwh = y_tr * denom_tr
    y_va_kwh = y_va * denom_va
    est_kwh = _build_ashr_fixed_with_hp("espaciotemporal", 1, weather_on, 1.35, 0.01, 0.10)
    est_kwh.fit(X_tr, y_tr_kwh, metadata=metadata_tr)
    pred_raw_kwh = np.maximum(est_kwh.predict(X_va), 0.0)
    rows.append(
        {
            "ablation_variant": "energia_bruta_vs_cf",
            "mae_cf": float(np.mean(np.abs(pred_raw_kwh / denom_va - y_va))),
            "mae_kwh": float(np.mean(np.abs(pred_raw_kwh - y_va_kwh))),
        }
    )
    return pd.DataFrame(rows)


def compute_temporal_conformal_intervals(
    y_cal: np.ndarray,
    pred_cal: np.ndarray,
    pred_test: np.ndarray,
    y_test: np.ndarray | None = None,
    alphas: tuple[float, ...] = (0.10, 0.20),
    plants_test: np.ndarray | None = None,
) -> dict[str, Any]:
    """Split/temporal conformal prediction intervals with finite-sample quantile correction."""
    res_cal = np.abs(np.asarray(y_cal, dtype=float) - np.asarray(pred_cal, dtype=float))
    res_cal = res_cal[np.isfinite(res_cal)]
    n_cal = res_cal.size
    if n_cal == 0:
        raise ValueError("Empty calibration residuals.")

    p_te = np.asarray(pred_test, dtype=float)
    summary_rows: list[dict[str, Any]] = []
    intervals: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    plant_rows: list[dict[str, Any]] = []

    for alpha in alphas:
        q_level = min(1.0, (1.0 - float(alpha)) * (1.0 + 1.0 / n_cal))
        q = float(np.quantile(res_cal, q_level, method="higher" if hasattr(np, "quantile") else "linear"))
        lo = p_te - q
        hi = p_te + q
        intervals[float(alpha)] = (lo, hi)

        row: dict[str, Any] = {
            "nominal_coverage": float(1.0 - alpha),
            "alpha": float(alpha),
            "conformal_halfwidth": q,
            "mean_width": float(2.0 * q),
            "n_cal": int(n_cal),
        }
        if y_test is not None:
            yt = np.asarray(y_test, dtype=float)
            covered = (yt >= lo) & (yt <= hi)
            row["empirical_coverage"] = float(np.mean(covered))
            if plants_test is not None:
                pt = np.asarray(plants_test)
                for p in np.unique(pt):
                    pm = pt == p
                    plant_rows.append(
                        {
                            "nominal_coverage": float(1.0 - alpha),
                            "plant": str(p),
                            "n_obs": int(np.sum(pm)),
                            "empirical_coverage": float(np.mean(covered[pm])),
                            "mean_width": float(2.0 * q),
                        }
                    )
        summary_rows.append(row)

    return {
        "summary": pd.DataFrame(summary_rows),
        "by_plant": pd.DataFrame(plant_rows),
        "intervals": intervals,
    }


def compute_interpretability_summary(
    est: AdaptiveStructuredHuberRegressor,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    n_local: int = 3,
) -> dict[str, Any]:
    """Compute global standardized importance and exact additive local attributions."""
    st = est._state
    w = np.asarray(st["weights"][: X_train.shape[1]], dtype=float)
    mu = np.asarray(st["x_mean"][: X_train.shape[1]], dtype=float)
    sd = np.asarray(st["x_scale"][: X_train.shape[1]], dtype=float)
    m_r = float(st["m_r"])
    s_r = float(st["s_r"])

    Z_tr = (X_train - mu) / sd
    Z_te = (X_test - mu) / sd
    mean_z_tr = np.mean(Z_tr, axis=0)

    # Exact linear SHAP / additive attribution in CF space relative to training mean baseline
    contrib_te = s_r * (Z_te - mean_z_tr[None, :]) * w[None, :]
    global_df = pd.DataFrame(
        {
            "feature": feature_names,
            "std_coef": w,
            "mean_abs_contrib": np.mean(np.abs(contrib_te), axis=0),
        }
    ).sort_values("mean_abs_contrib", ascending=False).reset_index(drop=True)

    local_idx = np.linspace(0, max(0, len(X_test) - 1), num=min(n_local, len(X_test)), dtype=int)
    baseline_pred = float(m_r + s_r * (est.intercept_ + np.dot(mean_z_tr, w)))
    local_rows = []
    for idx in local_idx:
        local_rows.append(
            {
                "row_idx": int(idx),
                "baseline_pred": baseline_pred,
                "model_pred": float(est.predict(X_test[idx : idx + 1])[0]),
                "sum_attributions": float(np.sum(contrib_te[idx])),
                "attributions": dict(zip(feature_names, contrib_te[idx].tolist())),
            }
        )
    return {"global_importance": global_df, "local_explanations": local_rows}

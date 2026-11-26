"""Controlled representation-equivalence benchmark (kWh_bruto vs. CF) for SM-CFHR v3.0.0.

Ensures all compared models share identical splits, origins, features, causal
capacity-at-origin normalization, inverse reconstruction, physical clipping,
and comparable selection budgets (Condition A: fixed HPs; Condition B: inner 3-fold CV).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.adaptive_structured_huber import AdaptiveStructuredHuberRegressor
from src.ashr_oof_pipeline.config import ScenarioSpec, build_frozen_ashr
from src.ashr_oof_pipeline.oof_runner import (
    ANOM_COLS,
    ENERGY_BOUND,
    METEO_COLS,
    extract_cf_features,
)
from src.ashr_oof_pipeline.optimizacion_hiperparametros import build_causal_inner_splits

BASE_KWH_COLS = [
    "kwh_D",
    "media7_kwh",
    "media30_kwh",
    "cap_MW_D",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
]

MODEL_NAMES_V3 = [
    "SM-CFHR",
    "Ridge",
    "BayesianRidge",
    "HuberRegressor",
    "ElasticNet",
    "RandomForest",
    "XGBoost",
]

FIXED_HPS_A: dict[str, dict[str, Any]] = {
    "SM-CFHR": {"delta": 1.35, "alpha_lin": 0.01, "alpha_meteo": 0.10},
    "Ridge": {"alpha": 1.0},
    "BayesianRidge": {"alpha_1": 1e-6, "lambda_1": 1e-6},
    "HuberRegressor": {"epsilon": 1.35, "alpha": 0.01},
    "ElasticNet": {"alpha": 0.01, "l1_ratio": 0.5},
    "RandomForest": {"n_estimators": 100, "min_samples_leaf": 10, "max_features": 0.8},
    "XGBoost": {"n_estimators": 120, "max_depth": 3, "learning_rate": 0.05},
}

CV_GRIDS_B: dict[str, list[dict[str, Any]]] = {
    "SM-CFHR": [
        {"delta": 1.35, "alpha_lin": 0.01, "alpha_meteo": 0.10},
        {"delta": 1.35, "alpha_lin": 0.01, "alpha_meteo": 0.01},
        {"delta": 1.35, "alpha_lin": 0.10, "alpha_meteo": 0.10},
        {"delta": 2.00, "alpha_lin": 0.01, "alpha_meteo": 0.10},
    ],
    "Ridge": [{"alpha": a} for a in (0.01, 0.1, 1.0, 10.0)],
    "BayesianRidge": [
        {"alpha_1": a, "lambda_1": l}
        for a, l in ((1e-8, 1e-8), (1e-8, 1e-4), (1e-4, 1e-8), (1e-4, 1e-4))
    ],
    "HuberRegressor": [
        {"epsilon": e, "alpha": a}
        for e, a in ((1.20, 0.001), (1.35, 0.01), (1.75, 0.01), (1.35, 0.10))
    ],
    "ElasticNet": [
        {"alpha": a, "l1_ratio": r}
        for a, r in ((0.001, 0.2), (0.001, 0.8), (0.01, 0.2), (0.01, 0.8))
    ],
    "RandomForest": [
        {"n_estimators": 100, "min_samples_leaf": lf, "max_features": mf}
        for lf, mf in ((5, 0.5), (5, 1.0), (20, 0.5), (20, 1.0))
    ],
    "XGBoost": [
        {"n_estimators": 120, "max_depth": d, "learning_rate": lr}
        for d, lr in ((2, 0.03), (2, 0.08), (4, 0.03), (4, 0.08))
    ],
}


def extract_kwh_features(df: pd.DataFrame, weather_on: bool) -> tuple[np.ndarray, np.ndarray]:
    """Extract raw physical kWh predictors and causal origin energy bound."""
    cap = df["cap_MW_D"].to_numpy(dtype=float)
    denom = np.where(np.isfinite(cap) & (cap > 0), cap * ENERGY_BOUND, np.nan)
    cols = list(BASE_KWH_COLS) + (list(METEO_COLS) + list(ANOM_COLS) if weather_on else [])
    X = np.column_stack([df[c].to_numpy(dtype=float) for c in cols])
    return X, denom


def _build_sklearn_estimator(name: str, cfg: dict[str, Any], seed: int = 2026) -> Any:
    if name == "Ridge":
        est = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=float(cfg["alpha"])))
    elif name == "BayesianRidge":
        est = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            BayesianRidge(alpha_1=float(cfg["alpha_1"]), lambda_1=float(cfg["lambda_1"])),
        )
    elif name == "HuberRegressor":
        est = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            HuberRegressor(epsilon=float(cfg["epsilon"]), alpha=float(cfg["alpha"]), max_iter=500, tol=1e-4),
        )
    elif name == "ElasticNet":
        est = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            ElasticNet(alpha=float(cfg["alpha"]), l1_ratio=float(cfg["l1_ratio"]), max_iter=10000, tol=1e-4, random_state=seed),
        )
    elif name == "RandomForest":
        est = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=int(cfg["n_estimators"]),
                min_samples_leaf=int(cfg["min_samples_leaf"]),
                max_features=float(cfg["max_features"]),
                n_jobs=1,
                random_state=seed,
            ),
        )
    elif name == "XGBoost":
        est = make_pipeline(
            SimpleImputer(strategy="median"),
            XGBRegressor(
                n_estimators=int(cfg["n_estimators"]),
                max_depth=int(cfg["max_depth"]),
                learning_rate=float(cfg["learning_rate"]),
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=1.0,
                tree_method="hist",
                device="cpu",
                n_jobs=1,
                objective="reg:squarederror",
                random_state=seed,
                verbosity=0,
            ),
        )
    else:
        raise KeyError(name)
    return est


def fit_model_representation(
    model_name: str,
    representation: str,
    train: pd.DataFrame,
    task: str,
    horizon: int,
    weather_on: bool,
    horizon_direct: int = 1,
    target_col: str = "y",
    cfg: dict[str, Any] | None = None,
    seed: int = 2026,
) -> dict[str, Any]:
    """Fit `model_name` under `representation` ('CF' or 'kWh_bruto') strictly on `train`."""
    hp = dict(cfg or FIXED_HPS_A[model_name])
    if representation not in ("CF", "kWh_bruto"):
        raise ValueError(f"Representación desconocida: {representation}")

    if representation == "CF":
        X_raw, _, denom_tr = extract_cf_features(train, weather_on=weather_on, lead=1)
        y_vals = train[target_col].to_numpy(dtype=float)
        valid = np.isfinite(denom_tr) & (denom_tr > 0) & np.isfinite(y_vals)
        X_v = X_raw[valid]
        y_cf = np.clip(y_vals[valid] / (denom_tr[valid] * float(horizon_direct)), 0.0, 1.0)

        if model_name == "SM-CFHR":
            mu = np.nanmedian(X_v, axis=0)
            mu = np.where(np.isfinite(mu), mu, 0.0)
            X_imp = np.where(np.isfinite(X_v), X_v, mu)
            est = AdaptiveStructuredHuberRegressor(
                mode=task,
                horizon=int(horizon),
                weather_mode="forecast" if weather_on else "none",
                selection="fixed",
                fixed_config={
                    "name": "fixed_lin_mono",
                    "complexity_rank": 1,
                    "n_knots": 0,
                    "alpha_lin": float(hp["alpha_lin"]),
                    "alpha_meteo": float(hp["alpha_meteo"] if weather_on else hp["alpha_lin"]),
                    "alpha_spl": 0.10,
                    "max_interactions": 0,
                    "alpha_int": 0.10,
                    "structured": True,
                    "use_mono": True,
                    "anchor": "none",
                    "lead_damp": 1.0,
                },
                delta=float(hp["delta"]),
                max_iter=400,
                tol=1e-7,
                random_state=seed,
            )
            meta_tr = {
                "cf30_idx": 4,
                "monotone_pos_idx": (4, 7) if weather_on else (4,),
                "meteo_idx": tuple(range(7, 17)) if weather_on else (),
            }
            est.fit(X_imp, y_cf, metadata=meta_tr)
            return {
                "model_name": model_name,
                "representation": representation,
                "est": est,
                "mu": mu,
                "weather_on": weather_on,
                "horizon_direct": horizon_direct,
            }

        est = _build_sklearn_estimator(model_name, hp, seed=seed)
        est.fit(X_v, y_cf)
        return {
            "model_name": model_name,
            "representation": representation,
            "est": est,
            "mu": None,
            "weather_on": weather_on,
            "horizon_direct": horizon_direct,
        }

    # representation == "kWh_bruto"
    X_raw, denom_tr = extract_kwh_features(train, weather_on=weather_on)
    y_vals = train[target_col].to_numpy(dtype=float)
    valid = np.isfinite(denom_tr) & (denom_tr > 0) & np.isfinite(y_vals)
    X_v = X_raw[valid]
    y_v = y_vals[valid]

    if model_name == "SM-CFHR":
        mu = np.nanmedian(X_v, axis=0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        X_imp = np.where(np.isfinite(X_v), X_v, mu)
        est = AdaptiveStructuredHuberRegressor(
            mode=task,
            horizon=int(horizon),
            weather_mode="forecast" if weather_on else "none",
            selection="fixed",
            fixed_config={
                "name": "fixed_kwh_control",
                "complexity_rank": 1,
                "n_knots": 0,
                "alpha_lin": float(hp["alpha_lin"]),
                "alpha_meteo": float(hp["alpha_meteo"] if weather_on else hp["alpha_lin"]),
                "alpha_spl": 0.10,
                "max_interactions": 0,
                "alpha_int": 0.10,
                "structured": True,
                "use_mono": False,
                "anchor": "none",
                "lead_damp": 1.0,
            },
            delta=float(hp["delta"]),
            max_iter=400,
            tol=1e-7,
            random_state=seed,
        )
        est.fit(X_imp, y_v, metadata={"meteo_idx": tuple(range(8, 18)) if weather_on else ()})
        return {
            "model_name": model_name,
            "representation": representation,
            "est": est,
            "mu": mu,
            "weather_on": weather_on,
            "horizon_direct": horizon_direct,
        }

    base_est = _build_sklearn_estimator(model_name, hp, seed=seed)
    est = TransformedTargetRegressor(regressor=base_est, transformer=StandardScaler())
    est.fit(X_v, y_v)
    return {
        "model_name": model_name,
        "representation": representation,
        "est": est,
        "mu": None,
        "weather_on": weather_on,
        "horizon_direct": horizon_direct,
    }


def predict_model_representation(
    fitted: dict[str, Any], query: pd.DataFrame, lead: int = 1
) -> np.ndarray:
    """Predict physical kWh with identical inverse transform and clipping across models."""
    hd = float(fitted["horizon_direct"])
    fallback = query["media30_kwh"].to_numpy(dtype=float) * hd

    if fitted["representation"] == "CF":
        X_raw, _, denom_q = extract_cf_features(query, weather_on=fitted["weather_on"], lead=lead)
        if fitted["mu"] is not None:
            X_in = np.where(np.isfinite(X_raw), X_raw, fitted["mu"])
        else:
            X_in = X_raw
        cf_hat = np.clip(fitted["est"].predict(X_in), 0.0, 1.0)
        pred_kwh = cf_hat * denom_q * hd
    else:
        X_raw, denom_q = extract_kwh_features(query, weather_on=fitted["weather_on"])
        if fitted["mu"] is not None:
            X_in = np.where(np.isfinite(X_raw), X_raw, fitted["mu"])
        else:
            X_in = X_raw
        pred_kwh = fitted["est"].predict(X_in)

    pred_kwh = np.where(np.isfinite(pred_kwh), pred_kwh, fallback)
    upper = np.where(np.isfinite(denom_q), denom_q * hd, np.inf)
    return np.clip(pred_kwh, 0.0, upper)


def select_best_cfg_cv3(
    model_name: str,
    representation: str,
    train: pd.DataFrame,
    sp: ScenarioSpec,
    target_col: str,
    seed: int = 2026,
) -> dict[str, Any]:
    """Select best configuration from a 4-point grid using causal 3-fold inner CV."""
    splits = build_causal_inner_splits(
        train,
        mode_eval=sp.mode_eval,
        task=sp.task,
        horizon_days=sp.horizon_days,
        n_splits=3,
        embargo_days=30,
    )
    grid = CV_GRIDS_B[model_name]
    if not splits:
        return dict(FIXED_HPS_A[model_name])

    best_cfg = dict(grid[0])
    best_score = np.inf
    for cfg in grid:
        fold_maes = []
        for tr_idx, va_idx in splits:
            tr_df = train.iloc[tr_idx]
            va_df = train.iloc[va_idx]
            fitted = fit_model_representation(
                model_name=model_name,
                representation=representation,
                train=tr_df,
                task=sp.task,
                horizon=sp.horizon_days,
                weather_on=sp.weather_on,
                horizon_direct=sp.horizon_days if sp.mode_eval == "direct" else 1,
                target_col=target_col,
                cfg=cfg,
                seed=seed,
            )
            preds = predict_model_representation(fitted, va_df, lead=1)
            ae = np.abs(preds - va_df[target_col].to_numpy(dtype=float))
            codes = va_df["code"].astype(str).to_numpy()
            uniq = np.unique(codes)
            macro = float(np.mean([np.mean(ae[codes == c]) for c in uniq])) if uniq.size >= 2 else float(np.mean(ae))
            fold_maes.append(macro)
        score = float(np.mean(fold_maes))
        if score < best_score:
            best_score = score
            best_cfg = dict(cfg)
    return best_cfg


__all__ = [
    "CV_GRIDS_B",
    "FIXED_HPS_A",
    "MODEL_NAMES_V3",
    "extract_kwh_features",
    "fit_model_representation",
    "predict_model_representation",
    "select_best_cfg_cv3",
]

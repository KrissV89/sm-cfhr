"""Hyperparameter optimization (Optuna TPE) and nested causal CV for SM-CFHR / ASHR-Fixed."""

from __future__ import annotations

import hashlib
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

try:
    from adaptive_structured_huber import AdaptiveStructuredHuberRegressor
except ImportError:
    from src.adaptive_structured_huber import AdaptiveStructuredHuberRegressor
from .config import find_project_root, sha256_file
from .oof_runner import (
    METEO_COLS,
    build_raw_plants_from_panel,
    extract_cf_features,
    rollout_recursive_from_origin,
)

HPO_PROTOCOL_VERSION = "v1.0.3-exact-recursive-state"


def compute_hpo_protocol_hash(root: Path | None = None) -> str:
    """Compute a combined SHA-256 fingerprint for HPO protocol version, estimator, and CV runners."""
    r = root or find_project_root()
    h = hashlib.sha256()
    h.update(HPO_PROTOCOL_VERSION.encode("utf-8"))
    for rel in (
        "src/adaptive_structured_huber.py",
        "src/ashr_oof_pipeline/optimizacion_hiperparametros.py",
        "src/ashr_oof_pipeline/oof_runner.py",
        "experiments/config_ashr_congelada.json",
    ):
        p = r / rel
        if p.exists():
            h.update(sha256_file(p).encode("utf-8"))
    return h.hexdigest()


def _compute_macro_mae_kwh(
    y_true: np.ndarray, y_pred: np.ndarray, codes: np.ndarray | None
) -> float:
    ae = np.abs(y_pred - y_true)
    if codes is not None:
        uniq = np.unique(codes)
        if uniq.size >= 2:
            return float(np.mean([np.mean(ae[codes == c]) for c in uniq]))
    return float(np.mean(ae))


def _build_ashr_fixed_with_hp(
    task: str,
    horizon: int,
    weather_on: bool,
    delta: float,
    alpha_lin: float,
    alpha_meteo: float,
    seed: int = 2026,
) -> AdaptiveStructuredHuberRegressor:
    """Instantiate fixed-structure SM-CFHR with continuous HPs (no multimode selector)."""
    fixed_cfg = {
        "name": "fixed_lin_mono",
        "complexity_rank": 1,
        "n_knots": 0,
        "alpha_lin": float(alpha_lin),
        "alpha_meteo": float(alpha_meteo if weather_on else alpha_lin),
        "alpha_spl": 0.10,
        "max_interactions": 0,
        "alpha_int": 0.10,
        "structured": True,
        "use_mono": True,
        "anchor": "none",
        "lead_damp": 1.0,
    }
    return AdaptiveStructuredHuberRegressor(
        mode=task,
        horizon=int(horizon),
        weather_mode="forecast" if weather_on else "none",
        selection="fixed",
        fixed_config=fixed_cfg,
        delta=float(delta),
        max_iter=400,
        tol=1e-7,
        random_state=int(seed),
    )


def build_causal_inner_splits(
    df: pd.DataFrame,
    mode_eval: str,
    task: str,
    horizon_days: int = 1,
    n_splits: int = 3,
    embargo_days: int = 30,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construct strictly causal inner CV splits respecting D+1, recursive, and direct horizons.

    Invariants enforced:
      - D+1: max(date_train) <= min(date_val) - embargo (at least 1 day gap).
      - Direct: target_end_date_train <= min(origin_val).
      - Recursive: origin_train + horizon_days <= min(origin_val).
      - Espaciotemporal: when >= n_splits spatial groups exist, validation holds out a group
        not present in training for that split.
    """
    n = len(df)
    if n < 30:
        return []

    if mode_eval == "d1":
        time_col = "date"
    else:
        time_col = next((c for c in ("t_origin", "origen", "date") if c in df.columns), "date")

    times = pd.to_datetime(df[time_col]).to_numpy()
    uniq_t = np.sort(np.unique(times))
    if uniq_t.size < max(6, n_splits + 2):
        return []

    end_col = next((c for c in ("t_end_target", "target_end_date") if c in df.columns), None)
    if mode_eval == "direct" and end_col is not None:
        target_ends = pd.to_datetime(df[end_col]).to_numpy()
    elif mode_eval in ("direct", "recursive"):
        target_ends = times + np.timedelta64(int(horizon_days), "D")
    else:
        target_ends = times

    groups = df["grupo"].to_numpy() if "grupo" in df.columns else None
    uniq_g = np.unique(groups) if groups is not None else np.array([])
    use_spatial_holdout = task == "espaciotemporal" and groups is not None and uniq_g.size >= 2

    val_len = max(1, int(0.15 * uniq_t.size))
    gap_days = int(embargo_days) if uniq_t.size >= embargo_days + n_splits * val_len + 5 else 1
    gap_delta = np.timedelta64(max(1, gap_days), "D")

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        val_end_pos = uniq_t.size - k * val_len
        val_start_pos = max(1, val_end_pos - val_len)
        if val_start_pos >= val_end_pos:
            continue
        va_lo = uniq_t[val_start_pos]
        va_hi = uniq_t[val_end_pos - 1]
        va_mask = (times >= va_lo) & (times <= va_hi)

        if mode_eval == "d1":
            tr_mask = times <= (va_lo - gap_delta)
        else:
            tr_mask = target_ends <= va_lo

        if use_spatial_holdout:
            hold_g = uniq_g[k % uniq_g.size]
            tr_sp = tr_mask & (groups != hold_g)
            va_sp = va_mask & (groups == hold_g)
            if np.sum(tr_sp) < 15 or np.sum(va_sp) < 5:
                continue
            tr_mask, va_mask = tr_sp, va_sp

        tr_idx = np.where(tr_mask)[0]
        va_idx = np.where(va_mask)[0]
        if tr_idx.size >= 15 and va_idx.size >= 5:
            splits.append((tr_idx, va_idx))

    return splits


def audit_inner_splits(
    df: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    mode_eval: str,
    task: str,
    horizon_days: int = 1,
    strict: bool = True,
) -> pd.DataFrame:
    """Audit inner CV splits for temporal/spatial leakage and return detailed metadata."""
    time_col = "date" if mode_eval == "d1" else next((c for c in ("t_origin", "origen", "date") if c in df.columns), "date")
    times = pd.to_datetime(df[time_col]).to_numpy()
    end_col = next((c for c in ("t_end_target", "target_end_date") if c in df.columns), None)
    if mode_eval == "direct" and end_col is not None:
        target_ends = pd.to_datetime(df[end_col]).to_numpy()
    elif mode_eval in ("direct", "recursive"):
        target_ends = times + np.timedelta64(int(horizon_days), "D")
    else:
        target_ends = times

    groups = df["grupo"].to_numpy() if "grupo" in df.columns else None
    codes = df["code"].astype(str).to_numpy() if "code" in df.columns else None
    rows: list[dict[str, Any]] = []
    for idx, (tr_idx, va_idx) in enumerate(splits):
        tr_min = pd.Timestamp(np.min(times[tr_idx]))
        tr_max = pd.Timestamp(np.max(times[tr_idx]))
        tr_max_end = pd.Timestamp(np.max(target_ends[tr_idx]))
        va_min = pd.Timestamp(np.min(times[va_idx]))
        va_max = pd.Timestamp(np.max(times[va_idx]))
        gap_days = int((va_min - tr_max).days)

        tr_g = sorted({str(g) for g in groups[tr_idx]}) if groups is not None else []
        va_g = sorted({str(g) for g in groups[va_idx]}) if groups is not None else []
        spatial_disjoint = bool(set(tr_g).isdisjoint(set(va_g))) if groups is not None else True

        if strict:
            if mode_eval == "d1":
                assert tr_max < va_min, f"Fuga temporal D+1 en split {idx}: {tr_max} >= {va_min}"
            else:
                assert tr_max_end <= va_min, f"Fuga temporal {mode_eval} en split {idx}: {tr_max_end} > {va_min}"
            if task == "espaciotemporal" and groups is not None and len(np.unique(groups)) >= 2:
                assert spatial_disjoint, f"Fuga espacial ET en split {idx}: {set(tr_g) & set(va_g)}"

        rows.append(
            {
                "split_idx": idx,
                "mode_eval": mode_eval,
                "task": task,
                "horizon_days": int(horizon_days),
                "train_min_date": str(tr_min.date()),
                "train_max_date": str(tr_max.date()),
                "train_max_target_end": str(tr_max_end.date()),
                "val_min_date": str(va_min.date()),
                "val_max_date": str(va_max.date()),
                "effective_gap_days": gap_days,
                "train_groups": ",".join(tr_g),
                "val_groups": ",".join(va_g),
                "spatial_disjoint": spatial_disjoint,
                "n_train": int(len(tr_idx)),
                "n_val": int(len(va_idx)),
                "n_plants_train": int(len(np.unique(codes[tr_idx]))) if codes is not None else 0,
                "n_plants_val": int(len(np.unique(codes[va_idx]))) if codes is not None else 0,
            }
        )
    return pd.DataFrame(rows)


def _build_local_weather_store(
    history_df: pd.DataFrame, val_df: pd.DataFrame
) -> dict[str, Any]:
    """Build a strictly causal weather store from local DataFrames when external GEFSv12 store is absent."""
    origin_forecasts: dict[tuple[str, pd.Timestamp, pd.Timestamp], np.ndarray] = {}
    origin_baselines: dict[tuple[str, pd.Timestamp], np.ndarray] = {}

    time_col_h = next((c for c in ("date", "t_origin", "origen") if c in history_df.columns), "date")
    time_col_v = next((c for c in ("date", "t_origin", "origen") if c in val_df.columns), "date")

    hist = history_df.copy()
    hist[time_col_h] = pd.to_datetime(hist[time_col_h])
    val = val_df.copy()
    val[time_col_v] = pd.to_datetime(val[time_col_v])
    if "code" not in hist.columns:
        hist["code"] = "PLANT"
    if "code" not in val.columns:
        val["code"] = "PLANT"
    hist["code"] = hist["code"].astype(str)
    val["code"] = val["code"].astype(str)

    for code, v_grp in val.groupby("code", sort=True):
        v_sorted = v_grp.sort_values(time_col_v)
        first_val_date = pd.Timestamp(v_sorted[time_col_v].iloc[0])
        origin_ts = first_val_date - pd.Timedelta(days=1)

        # Strictly causal baseline from history <= origin_ts
        h_sub = hist[(hist["code"] == code) & (hist[time_col_h] <= origin_ts)].sort_values(time_col_h)
        if not h_sub.empty and all(c in h_sub.columns for c in METEO_COLS):
            base_vec = np.nanmean(h_sub[METEO_COLS].tail(30).to_numpy(dtype=float), axis=0)
        else:
            base_vec = np.zeros(len(METEO_COLS), dtype=float)
        origin_baselines[(str(code), origin_ts)] = base_vec

        if all(c in v_sorted.columns for c in METEO_COLS):
            for row in v_sorted.itertuples(index=False):
                t_date = pd.Timestamp(getattr(row, time_col_v))
                w_vec = np.array([float(getattr(row, c)) for c in METEO_COLS], dtype=float)
                origin_forecasts[(str(code), origin_ts, t_date)] = w_vec

    return {
        "origin_forecasts": origin_forecasts,
        "origin_baselines": origin_baselines,
    }


def _rollout_recursive_val(
    est: AdaptiveStructuredHuberRegressor,
    mu: np.ndarray,
    val_df: pd.DataFrame,
    horizon_days: int,
    weather_on: bool,
    target_col: str = "y",
    raw_plants: dict[str, dict[str, Any]] | None = None,
    weather_store: dict[str, Any] | None = None,
    history_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Exact causal recursive rollout on validation block sharing `rollout_recursive_from_origin`."""
    time_col = next((c for c in ("date", "t_origin", "origen") if c in val_df.columns), "date")
    has_code = "code" in val_df.columns
    codes_arr = val_df["code"].astype(str).to_numpy() if has_code else np.full(len(val_df), "PLANT")

    fitted = {
        "est": est,
        "mu": mu,
        "weather_on": bool(weather_on),
        "horizon_direct": 1,
    }

    if raw_plants is None:
        combined = pd.concat([history_df, val_df], ignore_index=True) if history_df is not None else val_df
        if target_col != "y" and target_col in combined.columns and "y" not in combined.columns:
            combined = combined.assign(y=combined[target_col])
        local_raw = build_raw_plants_from_panel(combined)
        # Ensure the day-before-val origin has causal kwh_D and cap_MW_D when history_df lacks origin row
        for code in np.unique(codes_arr):
            sub = val_df[codes_arr == code].sort_values(time_col).reset_index(drop=True)
            if sub.empty:
                continue
            first_dt = pd.Timestamp(sub.loc[0, time_col])
            origin_ts = first_dt - pd.Timedelta(days=1)
            info = local_raw[str(code)]
            if origin_ts < info["min_date"]:
                full_idx = pd.date_range(origin_ts, info["max_date"], freq="D")
                s_kwh = pd.Series(info["kwh_valid"], index=info["dates"]).reindex(full_idx)
                s_cap = pd.Series(info["cap_MW"], index=info["dates"]).reindex(full_idx).bfill()
                if "kwh_D" in sub.columns and np.isfinite(sub.loc[0, "kwh_D"]):
                    s_kwh.loc[origin_ts] = float(sub.loc[0, "kwh_D"])
                if "cap_MW_D" in sub.columns and np.isfinite(sub.loc[0, "cap_MW_D"]):
                    s_cap.loc[origin_ts] = float(sub.loc[0, "cap_MW_D"])
                info["min_date"] = full_idx.min()
                info["dates"] = full_idx
                info["kwh_valid"] = s_kwh.to_numpy(dtype=float)
                info["cap_MW"] = s_cap.to_numpy(dtype=float)
        raw_plants_eff = local_raw
    else:
        raw_plants_eff = raw_plants

    if weather_on and weather_store is None:
        weather_store_eff = _build_local_weather_store(
            history_df if history_df is not None else pd.DataFrame(), val_df
        )
    else:
        weather_store_eff = weather_store

    y_out: list[float] = []
    p_out: list[float] = []
    c_out: list[str] = []

    for code in np.unique(codes_arr):
        sub = val_df[codes_arr == code].sort_values(time_col).reset_index(drop=True)
        if sub.empty or str(code) not in raw_plants_eff:
            continue
        first_dt = pd.Timestamp(sub.loc[0, time_col])
        origin_ts = first_dt - pd.Timedelta(days=1)
        info = raw_plants_eff[str(code)]
        if origin_ts < info["min_date"]:
            continue

        # Ensure PlantRecursiveState sees valid cap_at_origin <= origin_ts
        rows = rollout_recursive_from_origin(
            fitted=fitted,
            pred_infos=[info],
            origin=origin_ts,
            horizon_days=int(horizon_days),
            weather_on=bool(weather_on),
            weather_store=weather_store_eff,
        )
        for r in rows:
            y_out.append(float(r["y_kwh"]))
            p_out.append(float(r["prediccion_kwh"]))
            c_out.append(str(r["code"]))

    c_arr = np.asarray(c_out) if has_code else None
    return np.asarray(y_out, dtype=float), np.asarray(p_out, dtype=float), c_arr


def _fit_and_predict_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    task: str,
    horizon_days: int,
    weather_on: bool,
    delta: float,
    alpha_lin: float,
    alpha_meteo: float,
    horizon_direct: int = 1,
    target_col: str = "y",
    seed: int = 2026,
    mode_eval: str = "d1",
    raw_plants: dict[str, dict[str, Any]] | None = None,
    weather_store: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, AdaptiveStructuredHuberRegressor]:
    """Fit SM-CFHR strictly on train_df (fold-internal imputation) and predict val_df."""
    X_tr_raw, _, denom_tr = extract_cf_features(train_df, weather_on=weather_on, lead=1)
    y_tr = train_df[target_col].to_numpy(dtype=float)
    valid_tr = np.isfinite(denom_tr) & (denom_tr > 0) & np.isfinite(y_tr)
    X_tr_raw = X_tr_raw[valid_tr]
    cf_y_tr = np.clip(y_tr[valid_tr] / (denom_tr[valid_tr] * float(horizon_direct)), 0.0, 1.0)

    mu = np.nanmedian(X_tr_raw, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    X_tr_imp = np.where(np.isfinite(X_tr_raw), X_tr_raw, mu)

    meta_tr = {
        "cf30_idx": 4,
        "monotone_pos_idx": (4, 7) if weather_on else (4,),
        "meteo_idx": tuple(range(7, 17)) if weather_on else (),
    }
    est = _build_ashr_fixed_with_hp(
        task=task,
        horizon=horizon_days,
        weather_on=weather_on,
        delta=delta,
        alpha_lin=alpha_lin,
        alpha_meteo=alpha_meteo,
        seed=seed,
    )
    est.fit(X_tr_imp, cf_y_tr, metadata=meta_tr)

    if mode_eval == "recursive" and int(horizon_days) > 1:
        y_va, pred_kwh, codes_va = _rollout_recursive_val(
            est,
            mu,
            val_df,
            horizon_days=horizon_days,
            weather_on=weather_on,
            target_col=target_col,
            raw_plants=raw_plants,
            weather_store=weather_store,
            history_df=train_df,
        )
        return y_va, pred_kwh, codes_va, est

    X_va_raw, _, denom_va = extract_cf_features(val_df, weather_on=weather_on, lead=1)
    X_va_imp = np.where(np.isfinite(X_va_raw), X_va_raw, mu)
    pred_cf = np.clip(est.predict(X_va_imp), 0.0, 1.0)
    pred_kwh = pred_cf * denom_va * float(horizon_direct)
    y_va = val_df[target_col].to_numpy(dtype=float)
    codes_va = val_df["code"].to_numpy() if "code" in val_df.columns else None
    return y_va, pred_kwh, codes_va, est


def optimize_hyperparams_tpe(
    train_outer: pd.DataFrame,
    task: str,
    horizon_days: int,
    weather_on: bool,
    mode_eval: str,
    n_trials: int = 25,
    seed: int = 2026,
    outer_fold_id: str | int = 0,
    horizon_direct: int = 1,
    target_col: str = "y",
    n_inner_splits: int = 3,
    embargo_days: int = 30,
    storage_url: str | None = None,
    study_name: str | None = None,
    proto_hash: str | None = None,
    raw_plants: dict[str, dict[str, Any]] | None = None,
    weather_store: dict[str, Any] | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Run bounded TPE Bayesian optimization strictly inside `train_outer` with optional SQLite persistence."""
    splits = build_causal_inner_splits(
        train_outer,
        mode_eval=mode_eval,
        task=task,
        horizon_days=horizon_days,
        n_splits=n_inner_splits,
        embargo_days=embargo_days,
    )
    if not splits:
        fallback = {"delta": 1.35, "alpha_lin": 0.01, "alpha_meteo": 0.10 if weather_on else 0.01}
        return fallback, pd.DataFrame()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(
        seed=int(seed),
        n_startup_trials=min(8, max(2, int(n_trials) // 3)),
        multivariate=True,
    )
    storage = (
        optuna.storages.RDBStorage(url=storage_url, engine_kwargs={"connect_args": {"timeout": 30}})
        if storage_url
        else None
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage=storage,
        study_name=study_name,
        load_if_exists=bool(storage is not None and study_name is not None),
    )
    completed_before = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    existing_ver = study.user_attrs.get("hpo_protocol_version")
    if existing_ver is not None and existing_ver != HPO_PROTOCOL_VERSION:
        raise RuntimeError(
            f"Estudio Optuna '{study_name}' incompatible: hpo_protocol_version={existing_ver} != {HPO_PROTOCOL_VERSION}."
        )
    if existing_ver is None and len(completed_before) > 0:
        raise RuntimeError(
            f"Estudio Optuna legado '{study_name}' sin '{HPO_PROTOCOL_VERSION}' es incompatible con v1.0.3; no se reutilizan trials antiguos."
        )
    if proto_hash is not None:
        existing_hash = study.user_attrs.get("proto_hash")
        if existing_hash is not None and existing_hash != proto_hash:
            raise RuntimeError(
                f"Estudio Optuna '{study_name}' incompatible: proto_hash={existing_hash} != {proto_hash}."
            )
        study.set_user_attr("proto_hash", proto_hash)
    study.set_user_attr("hpo_protocol_version", HPO_PROTOCOL_VERSION)

    remaining_trials = max(0, int(n_trials) - len(completed_before))

    def objective(trial: optuna.Trial) -> float:
        delta = trial.suggest_float("delta", 0.5, 3.0)
        alpha_lin = trial.suggest_float("alpha_lin", 1e-4, 1.0, log=True)
        if weather_on:
            alpha_meteo = trial.suggest_float("alpha_meteo", 1e-3, 10.0, log=True)
        else:
            alpha_meteo = alpha_lin

        t0 = time.perf_counter()
        tracemalloc.start()
        fold_maes = []
        for tr_idx, va_idx in splits:
            y_va, pred_va, codes_va, _ = _fit_and_predict_fold(
                train_outer.iloc[tr_idx],
                train_outer.iloc[va_idx],
                task=task,
                horizon_days=horizon_days,
                weather_on=weather_on,
                delta=delta,
                alpha_lin=alpha_lin,
                alpha_meteo=alpha_meteo,
                horizon_direct=horizon_direct,
                target_col=target_col,
                seed=seed,
                mode_eval=mode_eval,
                raw_plants=raw_plants,
                weather_store=weather_store,
            )
            valid_va = np.isfinite(y_va) & np.isfinite(pred_va)
            if np.any(valid_va):
                c_sub = codes_va[valid_va] if codes_va is not None else None
                fold_maes.append(_compute_macro_mae_kwh(y_va[valid_va], pred_va[valid_va], c_sub))
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        wall_s = float(time.perf_counter() - t0)

        mean_mae = float(np.mean(fold_maes)) if (len(fold_maes) == len(splits) and np.all(np.isfinite(fold_maes))) else float("inf")
        trial.set_user_attr("alpha_meteo_eff", float(alpha_meteo))
        trial.set_user_attr("wall_time_s", wall_s)
        trial.set_user_attr("peak_memory_kb", float(peak_mem / 1024.0))
        return mean_mae

    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials, show_progress_bar=False)

    trial_logs: list[dict[str, Any]] = []
    best_so_far = float("inf")
    for t in sorted(study.trials, key=lambda x: x.number):
        if t.state != optuna.trial.TrialState.COMPLETE or t.value is None:
            trial_logs.append(
                {
                    "outer_fold": outer_fold_id,
                    "trial_number": int(t.number),
                    "delta": float(t.params.get("delta", np.nan)),
                    "alpha_lin": float(t.params.get("alpha_lin", np.nan)),
                    "alpha_meteo": float(t.user_attrs.get("alpha_meteo_eff", t.params.get("alpha_meteo", np.nan))),
                    "inner_mae_macro_kwh": float("inf"),
                    "trial_valid": False,
                    "n_inner_splits": len(splits),
                    "seed": int(seed),
                    "wall_time_s": float(t.user_attrs.get("wall_time_s", 0.0)),
                    "peak_memory_kb": float(t.user_attrs.get("peak_memory_kb", 0.0)),
                    "best_so_far_kwh": float(best_so_far) if np.isfinite(best_so_far) else float("nan"),
                }
            )
            continue
        val = float(t.value)
        is_valid = bool(np.isfinite(val))
        if is_valid:
            best_so_far = min(best_so_far, val)
        trial_logs.append(
            {
                "outer_fold": outer_fold_id,
                "trial_number": int(t.number),
                "delta": float(t.params["delta"]),
                "alpha_lin": float(t.params["alpha_lin"]),
                "alpha_meteo": float(t.user_attrs.get("alpha_meteo_eff", t.params.get("alpha_meteo", t.params["alpha_lin"]))),
                "inner_mae_macro_kwh": val,
                "trial_valid": is_valid,
                "n_inner_splits": len(splits),
                "seed": int(seed),
                "wall_time_s": float(t.user_attrs.get("wall_time_s", 0.0)),
                "peak_memory_kb": float(t.user_attrs.get("peak_memory_kb", 0.0)),
                "best_so_far_kwh": float(best_so_far) if np.isfinite(best_so_far) else float("nan"),
            }
        )

    bp = dict(study.best_params)
    best_hp = {
        "delta": float(bp["delta"]),
        "alpha_lin": float(bp["alpha_lin"]),
        "alpha_meteo": float(bp.get("alpha_meteo", bp["alpha_lin"])),
    }
    return best_hp, pd.DataFrame(trial_logs)


def nested_cv_optimize_and_refit(
    df: pd.DataFrame,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    task: str,
    horizon_days: int,
    weather_on: bool,
    mode_eval: str,
    n_trials: int = 25,
    seed: int = 2026,
    horizon_direct: int = 1,
    target_col: str = "y",
    n_inner_splits: int = 3,
    embargo_days: int = 30,
    raw_plants: dict[str, dict[str, Any]] | None = None,
    weather_store: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute full nested causal CV: inner TPE HPO -> outer train refit -> OOF predictions."""
    oof_frames: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []
    all_trials: list[pd.DataFrame] = []

    for fold_idx, (tr_idx, te_idx) in enumerate(outer_splits):
        train_outer = df.iloc[tr_idx].reset_index(drop=True)
        test_outer = df.iloc[te_idx].reset_index(drop=True)

        best_hp, trials_df = optimize_hyperparams_tpe(
            train_outer=train_outer,
            task=task,
            horizon_days=horizon_days,
            weather_on=weather_on,
            mode_eval=mode_eval,
            n_trials=n_trials,
            seed=seed,
            outer_fold_id=fold_idx,
            horizon_direct=horizon_direct,
            target_col=target_col,
            n_inner_splits=n_inner_splits,
            embargo_days=embargo_days,
            raw_plants=raw_plants,
            weather_store=weather_store,
        )
        if not trials_df.empty:
            all_trials.append(trials_df)

        t0 = time.perf_counter()
        y_te, pred_te, codes_te, _ = _fit_and_predict_fold(
            train_outer,
            test_outer,
            task=task,
            horizon_days=horizon_days,
            weather_on=weather_on,
            delta=best_hp["delta"],
            alpha_lin=best_hp["alpha_lin"],
            alpha_meteo=best_hp["alpha_meteo"],
            horizon_direct=horizon_direct,
            target_col=target_col,
            seed=seed,
            mode_eval=mode_eval,
            raw_plants=raw_plants,
            weather_store=weather_store,
        )
        refit_s = float(time.perf_counter() - t0)
        valid_te = np.isfinite(y_te) & np.isfinite(pred_te)
        c_sub = codes_te[valid_te] if codes_te is not None else None
        outer_mae = _compute_macro_mae_kwh(y_te[valid_te], pred_te[valid_te], c_sub)

        out_fold = test_outer.copy()
        out_fold["pred_sm_cfhr_hpo_kwh"] = pred_te
        out_fold["outer_fold"] = fold_idx
        oof_frames.append(out_fold)
        fold_summaries.append(
            {
                "outer_fold": fold_idx,
                "n_train": len(train_outer),
                "n_test": len(test_outer),
                "outer_mae_macro_kwh": outer_mae,
                "refit_time_s": refit_s,
                **best_hp,
            }
        )

    oof_df = pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame()
    trials_all = pd.concat(all_trials, ignore_index=True) if all_trials else pd.DataFrame()
    return {
        "oof_df": oof_df,
        "fold_summaries": pd.DataFrame(fold_summaries),
        "trial_logs": trials_all,
    }

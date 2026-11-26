"""Direct out-of-fold multi-horizon benchmark (7, 14, 21, 28 days) without weather.

Evaluates all 22 procedures from ``produccion_intraplanta.py`` under contiguous
expanding-window temporal cross-validation across the full historical dataset
(2024-01-01 to 2026-10-31). Both intraplanta and leave-one-group-out
spatiotemporal designs are supported, predicting the cumulative H-day energy
Y_H(t) = sum_{l=1..H} y_{t+l} directly from origin t with strict causal
target-end filtering (t' + H <= origin) and automated leakage audits.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path

for _env_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ.setdefault(_env_name, "1")

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from scripts.evaluacion import produccion_intraplanta as daily
from scripts.evaluacion.produccion_multihorizonte import (
    CUSTOM_WX,
    ENERGY_BOUND,
    EWMA_ALPHAS,
    INNER_FRACTIONS,
    INNER_GAP_DAYS,
    INNER_MAX_SPLITS,
    KALMAN_QS,
    MIN_PLANT_HISTORY_DAYS,
    MIN_TRAIN_ROWS,
    PlantRecursiveState,
    RECALIBRATION_DAYS,
    WARMUP_DAYS,
    _file_sha256,
    _harc_design,
    _key_signature,
    _plant_has_history_and_valid_future,
    build_outer_origins,
    load_generation_panel,
)


HORIZONS = (7, 14, 21, 28)


def _clip_direct_physical(pred: np.ndarray, query: pd.DataFrame, horizon: int) -> np.ndarray:
    """Clip direct H-day energy predictions to [0, H * 24000 * cap_MW_D]."""
    cap = query["cap_MW_D"].to_numpy(dtype=float)
    upper = np.where(np.isfinite(cap) & (cap > 0), float(horizon) * ENERGY_BOUND * cap, np.inf)
    return np.clip(np.asarray(pred, dtype=float), 0.0, upper)


def build_direct_panel(
    base_path: Path,
    horizon: int,
    base_data: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Build causal direct-horizon panel with complete H-day targets Y_H(t).

    For each row at origin ``t_origin`` (whose causal features are indexed at
    ``date = t_origin + 1 day`` in ``load_generation_panel``), the direct target
    is ``Y_H(t_origin) = sum_{l=1..H} y_{t_origin + l}`` computed strictly over
    complete windows (all ``H`` days have ``horas_xm > 6`` and finite ``kwh``).
    Incomplete windows are never imputed and are excluded.
    """
    panel, raw_plants = load_generation_panel(base_path, base_data=base_data)
    if panel.empty:
        panel["t_origin"] = pd.Series(dtype="datetime64[ns]")
        panel["t_end_target"] = pd.Series(dtype="datetime64[ns]")
        panel["y_H"] = pd.Series(dtype=float)
        panel["B_H"] = pd.Series(dtype=float)
        return panel, raw_plants

    y_h_vals = np.full(len(panel), np.nan, dtype=float)
    for code, idxs in panel.groupby("code", sort=False).groups.items():
        info = raw_plants[str(code)]
        min_date = info["min_date"]
        kwh_valid = info["kwh_valid"]
        n_total = len(kwh_valid)
        idx_arr = np.asarray(list(idxs), dtype=int)
        dates = panel["date"].iloc[idx_arr]
        start_offsets = (dates - min_date).dt.days.to_numpy(dtype=int)
        for pos, start_idx in zip(idx_arr, start_offsets):
            end_idx = start_idx + horizon
            if 0 <= start_idx and end_idx <= n_total:
                win = kwh_valid[start_idx:end_idx]
                if np.isfinite(win).all():
                    y_h_vals[pos] = float(np.sum(win))

    direct = panel.copy()
    direct["t_origin"] = direct["date"] - pd.Timedelta(days=1)
    direct["t_end_target"] = direct["t_origin"] + pd.Timedelta(days=horizon)
    direct["y_H"] = y_h_vals
    direct["B_H"] = float(horizon) * direct["media30_kwh"]
    direct = direct[np.isfinite(direct["y_H"]) & np.isfinite(direct["B_H"])].reset_index(drop=True)
    return direct, raw_plants


def _build_origin_query_row(plant_info: dict, origin: pd.Timestamp, horizon: int) -> dict:
    """Build strictly causal feature row at `origin` (features through `origin`)."""
    st = PlantRecursiveState(plant_info, origin, horizon)
    row = st.next_features(origin + pd.Timedelta(days=1))
    row["B_H"] = float(horizon) * float(row["media30_kwh"]) if np.isfinite(row["media30_kwh"]) else np.nan
    return row


def _plant_eligible_at_origin(plant_info: dict, origin: pd.Timestamp, horizon: int) -> tuple[bool, str]:
    """Check causal history, valid media30_kwh at origin, and >=1 valid future day."""
    ok, reason = _plant_has_history_and_valid_future(plant_info, origin, horizon)
    if not ok:
        return False, reason
    n_hist = (origin - plant_info["min_date"]).days + 1
    s30 = max(0, n_hist - 30)
    w30 = plant_info["kwh_valid"][s30:n_hist]
    if np.isfinite(w30).sum() < 15:
        return False, "media30_insuficiente_al_origen"
    return True, "elegible"


def fit_direct_model(method: str, cfg: dict, train: pd.DataFrame, horizon: int) -> dict:
    """Fit a procedure once on `train` for direct H-day energy prediction."""
    if method == "Media móvil de 30 días":
        return {"method": method, "horizon": horizon}

    if method in daily.LIBRARIES:
        model = daily._library(method, cfg, seed=daily.SEED)
        model.fit(train[daily.BASE_X], train["y_H"])
        return {"method": method, "horizon": horizon, "model": model}

    X_raw = train[CUSTOM_WX].to_numpy(dtype=float)
    mu = np.nanmedian(X_raw, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    X_imp = np.where(np.isfinite(X_raw), X_raw, mu)
    scaler = StandardScaler().fit(X_imp)
    X = scaler.transform(X_imp)

    y_h = train["y_H"].to_numpy(dtype=float)
    b_h = train["B_H"].to_numpy(dtype=float)
    r = y_h - b_h
    r_center = float(np.mean(r))
    r_scale = float(np.std(r))
    r_scale = r_scale if np.isfinite(r_scale) and r_scale > 1e-10 else 1.0
    r_std = (r - r_center) / r_scale

    state: dict = {
        "method": method,
        "horizon": horizon,
        "cfg": cfg,
        "mu": mu,
        "scaler": scaler,
        "r_center": r_center,
        "r_scale": r_scale,
    }

    if method == "RLAC":
        state["est"] = Ridge(alpha=cfg["alpha"]).fit(X, r_std)
    elif method == "IWLAR":
        state["X_train"] = X
        state["r_std"] = r_std
        state["train_code"] = train["code"].to_numpy()
        state["train_group"] = train["grupo"].to_numpy()
        state["h"] = float(cfg["h"])
    elif method == "PDEC":
        x_pdec = np.column_stack([X, train["deviation_D"].fillna(0.0).to_numpy(dtype=float)])
        state["est"] = Ridge(alpha=cfg["alpha"]).fit(x_pdec, r_std)
    elif method == "ELAC":
        lv = float(horizon) * train[f"ewma_{cfg['alpha']}"].fillna(train["media30_kwh"]).to_numpy(dtype=float)
        target = (y_h - lv - r_center) / r_scale
        state["est"] = Ridge(alpha=1).fit(X, target)
        state["level_col"] = f"ewma_{cfg['alpha']}"
    elif method == "KLAC":
        col = f"kalman_{cfg['q']}"
        lv = float(horizon) * train[col].fillna(train["media30_kwh"]).to_numpy(dtype=float)
        target = (y_h - lv - r_center) / r_scale
        state["est"] = Ridge(alpha=1).fit(X, target)
        state["level_col"] = col
    elif method == "RARC":
        state["est"] = HuberRegressor(epsilon=cfg["epsilon"], max_iter=300).fit(X, r_std)
    elif method == "MARB":
        constraints = tuple(0 for _ in range(X.shape[1]))
        state["est"] = XGBRegressor(
            n_estimators=120,
            max_depth=cfg["depth"],
            learning_rate=0.04,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=1,
            tree_method="hist",
            device="cpu",
            n_jobs=1,
            objective="reg:squarederror",
            random_state=daily.SEED,
            verbosity=0,
            monotone_constraints=constraints,
        ).fit(X, r_std)
    elif method == "RQRF":
        rf = RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=cfg["leaf"],
            max_features=1.0,
            n_jobs=1,
            random_state=daily.SEED,
        ).fit(X, r_std)
        leaf_mat = rf.apply(X)
        leaf_res = []
        for t_i in range(leaf_mat.shape[1]):
            col_ids = leaf_mat[:, t_i]
            leaf_res.append({int(leaf): r_std[col_ids == leaf] for leaf in np.unique(col_ids)})
        state["rf"] = rf
        state["leaf_res"] = leaf_res
    elif method == "ABGPR":
        ix = daily._recent_stratified_indices(train)
        ker = ConstantKernel(1.0) * RBF(cfg["length"]) + WhiteKernel(0.2)
        state["est"] = GaussianProcessRegressor(
            kernel=ker, alpha=1e-4, normalize_y=True, optimizer=None
        ).fit(X[ix], r_std[ix])
    elif method == "SRMOS":
        w = cfg["window"]
        cutoff = train["t_end_target"].max() - pd.Timedelta(days=w - 1)
        mask = train["t_end_target"].ge(cutoff).to_numpy()
        a = np.zeros((int(mask.sum()), 1), dtype=float)
        m = Ridge(alpha=1).fit(a, r_std[mask])
        state["const_r_std"] = float(m.predict([[0.0]])[0])
    elif method == "HARC":
        groups = pd.Index(pd.unique(train["grupo"]))
        plants = pd.Index(pd.unique(train["code"]))
        use_group = len(groups) > 1
        A = _harc_design(train, X, groups, plants, use_group)
        ng = len(groups) if use_group else 0
        penalty = np.r_[
            0.0,
            np.ones(X.shape[1]),
            np.full(2 * ng, cfg["lam_group"]),
            np.full(2 * len(plants), cfg["lam_plant"]),
        ]
        beta = np.linalg.solve(A.T @ A + np.diag(penalty) + np.eye(A.shape[1]) * 1e-9, A.T @ r_std)
        state["groups"] = groups
        state["plants"] = plants
        state["use_group"] = use_group
        state["beta"] = beta
    elif method == "WCFC":
        state["est_ridge"] = Ridge(alpha=1).fit(X, r_std)
        state["est_huber"] = HuberRegressor(max_iter=250).fit(X, r_std)
        state["weight_huber"] = float(cfg["weight_huber"])
    else:
        raise KeyError(method)

    return state


def predict_direct_batch(fitted: dict, query: pd.DataFrame) -> np.ndarray:
    """Predict direct H-day energy for a batch of query rows and clip physically."""
    method = fitted["method"]
    horizon = int(fitted["horizon"])
    base = float(horizon) * query["media30_kwh"].to_numpy(dtype=float)

    if method == "Media móvil de 30 días":
        return _clip_direct_physical(base, query, horizon)

    if method in daily.LIBRARIES:
        raw_pred = fitted["model"].predict(query[daily.BASE_X])
        return _clip_direct_physical(raw_pred, query, horizon)

    Xt_raw = query[CUSTOM_WX].to_numpy(dtype=float)
    Xt_imp = np.where(np.isfinite(Xt_raw), Xt_raw, fitted["mu"])
    Xt = fitted["scaler"].transform(Xt_imp)
    r_center = fitted["r_center"]
    r_scale = fitted["r_scale"]

    if method in ("RLAC", "RARC", "MARB", "ABGPR"):
        pred = base + r_center + r_scale * fitted["est"].predict(Xt)
    elif method == "IWLAR":
        X_tr = fitted["X_train"]
        r_std = fitted["r_std"]
        tr_code = fitted["train_code"]
        tr_group = fitted["train_group"]
        h = fitted["h"]
        te_code = query["code"].to_numpy()
        te_group = query["grupo"].to_numpy()
        pred = np.empty(len(query), dtype=float)
        dists = cdist(Xt, X_tr, "sqeuclidean") / (h ** 2)
        for i in range(len(query)):
            kernel = np.exp(-np.minimum(dists[i], 700.0))
            identity = np.where(te_code[i] == tr_code, 1.0, np.where(te_group[i] == tr_group, 0.5, 0.2))
            w = kernel * identity + 1e-8
            A = np.column_stack([np.ones(len(X_tr)), X_tr])
            Aq = np.r_[1.0, Xt[i]]
            G = A.T @ (w[:, None] * A) + 1e-3 * w.sum() * np.diag([0.0] + [1.0] * X_tr.shape[1])
            b = A.T @ (w * r_std)
            pred[i] = base[i] + r_center + r_scale * (Aq @ np.linalg.solve(G, b))
    elif method == "PDEC":
        xt_pdec = np.column_stack([Xt, query["deviation_D"].fillna(0.0).to_numpy(dtype=float)])
        pred = base + r_center + r_scale * fitted["est"].predict(xt_pdec)
    elif method == "ELAC":
        lvt = float(horizon) * query[fitted["level_col"]].fillna(query["media30_kwh"]).to_numpy(dtype=float)
        pred = lvt + r_center + r_scale * fitted["est"].predict(Xt)
    elif method == "KLAC":
        lvt = float(horizon) * query[fitted["level_col"]].fillna(query["media30_kwh"]).to_numpy(dtype=float)
        pred = lvt + r_center + r_scale * fitted["est"].predict(Xt)
    elif method == "RQRF":
        rf = fitted["rf"]
        leaf_res = fitted["leaf_res"]
        test_leaves = rf.apply(Xt)
        n_trees = test_leaves.shape[1]
        qpred = np.empty(len(Xt), dtype=float)
        for i in range(len(Xt)):
            vals_list = []
            wts_list = []
            row_leaves = test_leaves[i]
            for t_i in range(n_trees):
                arr = leaf_res[t_i][int(row_leaves[t_i])]
                n_leaf = len(arr)
                if n_leaf:
                    vals_list.append(arr)
                    wts_list.append(np.full(n_leaf, 1.0 / (n_trees * n_leaf)))
            vals = np.concatenate(vals_list)
            wts = np.concatenate(wts_list)
            order = np.argsort(vals)
            v = vals[order]
            w = wts[order]
            qpred[i] = v[min(np.searchsorted(np.cumsum(w), 0.5, side="left"), len(v) - 1)]
        pred = base + r_center + r_scale * qpred
    elif method == "SRMOS":
        pred = base + r_center + r_scale * fitted["const_r_std"]
    elif method == "HARC":
        At = _harc_design(query, Xt, fitted["groups"], fitted["plants"], fitted["use_group"])
        pred = base + r_center + r_scale * (At @ fitted["beta"])
    elif method == "WCFC":
        a = fitted["est_ridge"].predict(Xt)
        b = fitted["est_huber"].predict(Xt)
        w = fitted["weight_huber"]
        pred = base + r_center + r_scale * ((1.0 - w) * a + w * b)
    else:
        raise KeyError(method)

    return _clip_direct_physical(pred, query, horizon)


def build_direct_inner_splits(
    train: pd.DataFrame,
    origin: pd.Timestamp,
    horizon: int,
    gap_days: int = INNER_GAP_DAYS,
) -> list[tuple[pd.DataFrame, pd.Timestamp, pd.DataFrame]]:
    """Build up to INNER_MAX_SPLITS direct internal CV splits strictly <= origin.

    Inner training samples satisfy ``t_end_target <= s_m - gap_days``.
    """
    origins_avail = np.sort(train["t_origin"].unique())
    if len(origins_avail) == 0:
        return []
    splits = []
    seen = set()
    for frac in INNER_FRACTIONS[:INNER_MAX_SPLITS]:
        idx = min(int(round(len(origins_avail) * frac)), len(origins_avail) - 1)
        if idx < 0:
            continue
        s_m = pd.Timestamp(origins_avail[idx])
        if s_m + pd.Timedelta(days=horizon) > origin or s_m in seen:
            continue
        cutoff = s_m - pd.Timedelta(days=gap_days)
        inner_train = train[train["t_end_target"] <= cutoff]
        if len(inner_train) < 30:
            continue
        inner_val = train[train["t_origin"] == s_m]
        if inner_val.empty:
            continue
        seen.add(s_m)
        splits.append((inner_train, s_m, inner_val))
    return splits


def choose_hyperparameters_direct(
    method: str,
    horizon: int,
    inner_splits: list[tuple[pd.DataFrame, pd.Timestamp, pd.DataFrame]],
) -> dict:
    """Select hyperparameters for `method` via direct H-day inner CV."""
    if method == "Media móvil de 30 días":
        return {}
    configs = daily.GRIDS[method]
    if not inner_splits:
        return configs[0]
    if len(configs) > daily.MAX_CONFIGURATIONS:
        chosen = np.random.default_rng(daily.SEED).choice(
            len(configs), daily.MAX_CONFIGURATIONS, replace=False
        )
        configs = [configs[i] for i in sorted(chosen)]

    scores = []
    for cfg in configs:
        losses = []
        for inner_train, _, inner_val in inner_splits:
            try:
                fitted = fit_direct_model(method, cfg, inner_train, horizon)
                preds = predict_direct_batch(fitted, inner_val)
                errs = np.abs(preds - inner_val["y_H"].to_numpy(dtype=float))
                plant_maes = pd.Series(errs).groupby(inner_val["code"].to_numpy()).mean()
                losses.append(float(plant_maes.mean()))
            except (ValueError, np.linalg.LinAlgError):
                losses.append(np.inf)
        scores.append(float(np.mean(losses)) if losses else np.inf)
    return configs[int(np.argmin(scores))]


def _filter_causal_train(direct_panel: pd.DataFrame, task: str, unit: str | int, origin: pd.Timestamp) -> pd.DataFrame:
    """Return strictly causal training rows with t_end_target <= origin."""
    if task == "intraplanta":
        mask = (direct_panel["code"] == str(unit)) & (direct_panel["t_end_target"] <= origin)
    else:
        mask = (direct_panel["grupo"] != int(unit)) & (direct_panel["t_end_target"] <= origin)
    return direct_panel.loc[mask]


def _plan_unit_origins(
    task: str,
    unit: str | int,
    origins: list[pd.Timestamp],
    horizon: int,
    direct_panel: pd.DataFrame,
    raw_plants: dict[str, dict],
) -> tuple[list[dict], list[dict], list[tuple]]:
    """Plan eligible OOF origins and causal recalibration anchors for one (task, unit)."""
    cov_inelig: list[dict] = []
    eval_specs: list[dict] = []
    anchor_jobs: list[tuple] = []

    last_anchor: pd.Timestamp | None = None
    last_inner_count: int = 0

    for origin in origins:
        target_start = origin + pd.Timedelta(days=1)
        target_end = origin + pd.Timedelta(days=horizon)
        train = _filter_causal_train(direct_panel, task, unit, origin)

        if task == "intraplanta":
            code = str(unit)
            group_id = raw_plants[code]["grupo"]
            unit_name = code
            ok_plant, reason = _plant_eligible_at_origin(raw_plants[code], origin, horizon)
            if len(train) < MIN_TRAIN_ROWS:
                cov_inelig.append({
                    "tarea": task, "origen": origin, "unidad": unit_name, "code": code, "grupo": group_id,
                    "train_rows": len(train), "inner_splits": 0, "estado": "historia_entrenamiento_menor_60",
                    "train_end": train["t_end_target"].max() if len(train) else pd.NaT,
                    "test_start": target_start, "test_end": target_end,
                })
                continue
            if not ok_plant:
                cov_inelig.append({
                    "tarea": task, "origen": origin, "unidad": unit_name, "code": code, "grupo": group_id,
                    "train_rows": len(train), "inner_splits": 0, "estado": reason,
                    "train_end": train["t_end_target"].max(), "test_start": target_start, "test_end": target_end,
                })
                continue
            pred_codes = [code]
        else:
            group_id = int(unit)
            unit_name = f"grupo_{group_id}"
            group_codes = sorted(c for c, info in raw_plants.items() if info["grupo"] == group_id)
            pred_codes = []
            local_cov = []
            for code in group_codes:
                ok_plant, reason = _plant_eligible_at_origin(raw_plants[code], origin, horizon)
                if ok_plant:
                    pred_codes.append(code)
                else:
                    local_cov.append({
                        "tarea": task, "origen": origin, "unidad": unit_name, "code": code, "grupo": group_id,
                        "train_rows": len(train), "inner_splits": 0, "estado": reason,
                        "train_end": train["t_end_target"].max() if len(train) else pd.NaT,
                        "test_start": target_start, "test_end": target_end,
                    })
            if len(train) < MIN_TRAIN_ROWS or train["grupo"].nunique() < 1 or not pred_codes:
                if not local_cov:
                    local_cov.append({
                        "tarea": task, "origen": origin, "unidad": unit_name, "code": "", "grupo": group_id,
                        "train_rows": len(train), "inner_splits": 0, "estado": "sin_plantas_elegibles_o_train_insuficiente",
                        "train_end": train["t_end_target"].max() if len(train) else pd.NaT,
                        "test_start": target_start, "test_end": target_end,
                    })
                cov_inelig.extend(local_cov)
                continue
            cov_inelig.extend(local_cov)

        need_recal = (
            last_anchor is None
            or (origin - last_anchor).days >= RECALIBRATION_DAYS
            or last_inner_count == 0
        )
        if need_recal:
            inner_splits = build_direct_inner_splits(train, origin, horizon)
            if last_anchor is not None and last_inner_count == 0 and len(inner_splits) == 0:
                recalibrated = False
                anchor_used = last_anchor
                n_inner_used = 0
                inner_sig = "[]"
            else:
                recalibrated = True
                last_anchor = origin
                last_inner_count = len(inner_splits)
                anchor_used = origin
                n_inner_used = len(inner_splits)
                inner_sig = json.dumps([
                    {"train": _key_signature(a), "val_origin": s_m.strftime("%Y-%m-%d")}
                    for a, s_m, _ in inner_splits
                ])
                anchor_jobs.append((task, unit, origin, horizon, direct_panel))
        else:
            recalibrated = False
            anchor_used = last_anchor
            n_inner_used = last_inner_count
            inner_sig = "reutiliza_ancla"

        eval_specs.append({
            "task": task, "unit": unit, "unit_name": unit_name, "origin": origin,
            "pred_codes": pred_codes, "inner_splits_count": n_inner_used,
            "recalibrated": recalibrated, "anchor_origin": anchor_used,
            "train_rows": len(train), "train_end": train["t_end_target"].max(),
            "test_start": target_start, "test_end": target_end,
            "train_key_sha256": _key_signature(train), "inner_key_sha256": inner_sig,
        })

    return cov_inelig, eval_specs, anchor_jobs


def _run_anchor_recalibration_task(args: tuple) -> tuple[tuple, dict[str, dict], dict[str, float]]:
    """Stage 1 worker: select direct H-day hyperparameters for all 22 methods at one anchor."""
    task, unit, anchor_origin, horizon, direct_panel = args
    train = _filter_causal_train(direct_panel, task, unit, anchor_origin).copy()
    inner_splits = build_direct_inner_splits(train, anchor_origin, horizon)
    cfgs: dict[str, dict] = {}
    sel_secs: dict[str, float] = {}
    for method in daily.METHODS:
        t0 = time.perf_counter()
        cfgs[method] = choose_hyperparameters_direct(method, horizon, inner_splits)
        sel_secs[method] = time.perf_counter() - t0
    return (task, unit, anchor_origin), cfgs, sel_secs


def _run_unit_refit_batch_task(args: tuple) -> dict:
    """Stage 2 worker: refit all 22 methods and predict direct H-day energy for a chunk of origins."""
    specs, cfg_lookup, sel_lookup, horizon, direct_panel, raw_plants = args
    cov_rows: list[dict] = []
    pred_rows: list[dict] = []
    choice_rows: list[dict] = []

    for sp in specs:
        task = sp["task"]
        unit = sp["unit"]
        unit_name = sp["unit_name"]
        origin = sp["origin"]
        anchor_origin = sp["anchor_origin"]
        pred_infos = [raw_plants[c] for c in sp["pred_codes"]]

        train = _filter_causal_train(direct_panel, task, unit, origin).copy()
        query_rows = [_build_origin_query_row(info, origin, horizon) for info in pred_infos]
        query = pd.DataFrame(query_rows)

        for info in pred_infos:
            cov_rows.append({
                "tarea": task, "origen": origin, "unidad": unit_name, "code": info["code"], "grupo": info["grupo"],
                "train_rows": sp["train_rows"], "inner_splits": sp["inner_splits_count"], "estado": "evaluado",
                "train_end": sp["train_end"], "test_start": sp["test_start"], "test_end": sp["test_end"],
                "train_key_sha256": sp["train_key_sha256"], "inner_key_sha256": sp["inner_key_sha256"],
            })

        cfgs = cfg_lookup[(task, unit, anchor_origin)]
        sel_secs = sel_lookup[(task, unit, anchor_origin)]

        actuals: list[tuple[int, float, bool]] = []
        for info in pred_infos:
            n_hist = (origin - info["min_date"]).days + 1
            fut_kwh = info["kwh_valid"][n_hist : n_hist + horizon]
            dias_obs = int(np.isfinite(fut_kwh).sum())
            completa = bool(len(fut_kwh) == horizon and np.isfinite(fut_kwh).all())
            y_total = float(np.sum(fut_kwh)) if completa else np.nan
            actuals.append((dias_obs, y_total, completa))

        for method in daily.METHODS:
            cfg = cfgs[method]
            sel_sec = sel_secs[method] if sp["recalibrated"] else 0.0

            t1 = time.perf_counter()
            fitted = fit_direct_model(method, cfg, train, horizon)
            preds = predict_direct_batch(fitted, query)
            fit_pred_sec = time.perf_counter() - t1

            choice_rows.append({
                "tarea": task, "origen": origin, "unidad": unit_name, "metodo": method,
                "hiperparametros": json.dumps(cfg, ensure_ascii=False),
                "recalibrado": bool(sp["recalibrated"]),
                "origen_ancla_hiperparametros": anchor_origin,
                "splits_internos": sp["inner_splits_count"],
                "segundos_seleccion": sel_sec,
                "segundos_ajuste_y_prediccion": fit_pred_sec,
            })

            for i, info in enumerate(pred_infos):
                dias_obs, y_total, completa = actuals[i]
                cap_at_origin = float(query_rows[i]["cap_MW_D"])
                denom_total = cap_at_origin * ENERGY_BOUND * float(horizon)
                pred_val = float(preds[i])
                err_total = abs(pred_val - y_total) if completa else np.nan
                pred_rows.append({
                    "metodo": method,
                    "tarea": task,
                    "origen": origin,
                    "unidad_fold": unit_name,
                    "code": info["code"],
                    "grupo": info["grupo"],
                    "fecha_inicio_ventana": sp["test_start"],
                    "fecha_fin_ventana": sp["test_end"],
                    "horizonte_dias": horizon,
                    "prediccion_total_kwh": pred_val,
                    "y_total_kwh": y_total,
                    "dias_observados": dias_obs,
                    "completa": completa,
                    "error_total_abs_kwh": err_total,
                    "cap_MW_origen": cap_at_origin,
                    "error_total_rel_cap": (err_total / denom_total) if (np.isfinite(err_total) and denom_total > 0) else np.nan,
                })

    return {"coverage": cov_rows, "predictions": pred_rows, "choices": choice_rows}


def audit_direct_leakage(
    base_path: Path,
    direct_panel: pd.DataFrame,
    raw_plants: dict[str, dict],
    pred: pd.DataFrame,
    coverage: pd.DataFrame,
    choices: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Run 4 automated leakage & causal integrity controls on the direct pipeline."""
    records = []
    sample_origin = pd.Timestamp("2025-10-31")

    # (a) t_end_target <= origin for 100% of training samples
    ev = coverage[coverage["estado"] == "evaluado"]
    assert (pd.to_datetime(ev["train_end"]) <= pd.to_datetime(ev["origen"])).all(), "train_end > origin in OOF folds"
    sample_train = _filter_causal_train(direct_panel, "espaciotemporal", 1, sample_origin)
    assert (sample_train["t_end_target"] <= sample_origin).all(), "t_end_target > origin in sample_train"
    inner_sample = build_direct_inner_splits(sample_train, sample_origin, horizon)
    for itr, s_m, ival in inner_sample:
        assert (itr["t_end_target"] <= s_m - pd.Timedelta(days=INNER_GAP_DAYS)).all(), "inner_train leaks past s_m - gap"
        assert (ival["t_end_target"] <= sample_origin).all(), "inner_val leaks past origin"
    records.append({
        "control": "causalidad_t_end_target_train_y_inner",
        "detalle": f"100% muestras train y splits internos con t_end_target <= origen (folds={len(ev)})",
        "estado": "superado",
    })

    # (b) Future perturbation test (date > origin) for kwh (+50%) and cap_MW (+50%)
    base_df = pd.read_csv(
        base_path,
        usecols=["code", "date", "kwh", "horas_xm", "cap_MW", "lat", "lon", "grupo"],
        parse_dates=["date"],
    )
    altered_df = base_df.copy()
    fut_mask = altered_df["date"] > sample_origin
    altered_df.loc[fut_mask, "kwh"] = altered_df.loc[fut_mask, "kwh"] * 1.5 + 100.0
    altered_df.loc[fut_mask, "cap_MW"] = altered_df.loc[fut_mask, "cap_MW"] * 1.5
    panel_alt, plants_alt = build_direct_panel(base_path, horizon, base_data=altered_df)
    train_orig = _filter_causal_train(direct_panel, "espaciotemporal", 1, sample_origin)
    train_alt = _filter_causal_train(panel_alt, "espaciotemporal", 1, sample_origin)
    elig_codes = [c for c in sorted(raw_plants) if raw_plants[c]["grupo"] == 1 and _plant_eligible_at_origin(raw_plants[c], sample_origin, horizon)[0]]
    q_orig = pd.DataFrame([_build_origin_query_row(raw_plants[c], sample_origin, horizon) for c in elig_codes])
    q_alt = pd.DataFrame([_build_origin_query_row(plants_alt[c], sample_origin, horizon) for c in elig_codes])
    for method in ("RLAC", "KLAC", "XGBoost", "RQRF", "Media móvil de 30 días"):
        cfg = daily.GRIDS[method][0]
        p1 = predict_direct_batch(fit_direct_model(method, cfg, train_orig, horizon), q_orig)
        p2 = predict_direct_batch(fit_direct_model(method, cfg, train_alt, horizon), q_alt)
        np.testing.assert_allclose(p1, p2, atol=0.0, rtol=0.0)
    records.append({
        "control": "perturbacion_futuro_kwh_y_capacidad",
        "detalle": "atol=0.0 en RLAC, KLAC, XGBoost, RQRF, Media móvil de 30 días",
        "estado": "superado",
    })

    # (c) Prefix truncation test (only rows <= origin)
    prefix_df = base_df[base_df["date"] <= sample_origin].copy()
    panel_pref, plants_pref = build_direct_panel(base_path, horizon, base_data=prefix_df)
    train_pref = _filter_causal_train(panel_pref, "espaciotemporal", 1, sample_origin)
    q_pref = pd.DataFrame([_build_origin_query_row(plants_pref[c], sample_origin, horizon) for c in elig_codes])
    for method in ("RLAC", "KLAC", "Ridge"):
        cfg = daily.GRIDS[method][0]
        p1 = predict_direct_batch(fit_direct_model(method, cfg, train_orig, horizon), q_orig)
        p3 = predict_direct_batch(fit_direct_model(method, cfg, train_pref, horizon), q_pref)
        np.testing.assert_allclose(p1, p3, atol=0.0, rtol=0.0)
    records.append({
        "control": "reconstruccion_prefijo_temporal",
        "detalle": "atol=0.0 en RLAC, KLAC, Ridge",
        "estado": "superado",
    })

    # (d) OOF integrity: no duplicate (metodo, tarea, code, origen), train_end <= origen < test_start,
    #     and causal hyperparameter anchor <= origen
    assert (pd.to_datetime(ev["origen"]) < pd.to_datetime(ev["test_start"])).all(), "origin >= test_start"
    if not choices.empty:
        assert (
            pd.to_datetime(choices["origen_ancla_hiperparametros"]) <= pd.to_datetime(choices["origen"])
        ).all(), "origen_ancla_hiperparametros > origen"
    dups = int(pred.duplicated(subset=["metodo", "tarea", "code", "origen"]).sum())
    assert dups == 0, f"Duplicate OOF predictions found: {dups}"
    records.append({
        "control": "integridad_oof_y_causalidad_splits",
        "detalle": f"duplicados_oof={dups}, folds_evaluados={len(ev)}, anclas_causales_ok=True",
        "estado": "superado",
    })

    return pd.DataFrame(records)


def _aggregate_direct_metrics(pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Aggregate direct H-day total-energy OOF metrics over complete windows only."""
    complete = pred[pred["completa"]].copy()
    complete["sq_total"] = (complete["prediccion_total_kwh"] - complete["y_total_kwh"]) ** 2

    by_plant = complete.groupby(["tarea", "metodo", "code", "grupo"], as_index=False).agg(
        MAE_total_kWh=("error_total_abs_kwh", "mean"),
        MAE_total_rel_cap=("error_total_rel_cap", "mean"),
        RMSE2_total=("sq_total", "mean"),
        trayectorias=("origen", "size"),
    )
    by_plant["RMSE_total_kWh"] = np.sqrt(by_plant.pop("RMSE2_total"))

    by_group = by_plant.groupby(["tarea", "metodo", "grupo"], as_index=False).agg(
        MAE_macro_planta_total_kWh=("MAE_total_kWh", "mean"),
        MAE_macro_planta_rel_cap=("MAE_total_rel_cap", "mean"),
        RMSE_macro_planta_total_kWh=("RMSE_total_kWh", "mean"),
        plantas=("code", "nunique"),
        trayectorias=("trayectorias", "sum"),
    )

    incomp_counts = pred[~pred["completa"]].groupby(["tarea", "metodo"], as_index=False).agg(
        trayectorias_incompletas_excluidas=("origen", "size")
    )

    energy_total = by_plant.groupby(["tarea", "metodo"], as_index=False).agg(
        MAE_energia_total_macro_kWh=("MAE_total_kWh", "mean"),
        MAE_energia_total_rel_cap=("MAE_total_rel_cap", "mean"),
        RMSE_energia_total_macro_kWh=("RMSE_total_kWh", "mean"),
        trayectorias_completas=("trayectorias", "sum"),
        plantas=("code", "nunique"),
    ).merge(incomp_counts, on=["tarea", "metodo"], how="left")
    energy_total["trayectorias_incompletas_excluidas"] = (
        energy_total["trayectorias_incompletas_excluidas"].fillna(0).astype(int)
    )

    base_tot = energy_total[energy_total["metodo"] == "Media móvil de 30 días"][
        ["tarea", "MAE_energia_total_macro_kWh"]
    ].rename(columns={"MAE_energia_total_macro_kWh": "_base_tot_mae"})
    energy_total = energy_total.merge(base_tot, on="tarea", how="left")
    energy_total["Skill_MAE_energia_total"] = 1.0 - (
        energy_total["MAE_energia_total_macro_kWh"] / energy_total["_base_tot_mae"]
    )
    energy_total = energy_total.drop(columns=["_base_tot_mae"])
    energy_total = energy_total.sort_values(["tarea", "MAE_energia_total_macro_kWh"]).reset_index(drop=True)

    return {
        "energy_total": energy_total,
        "by_plant": by_plant,
        "by_group": by_group,
    }


def run_direct_horizon_benchmark(
    base_path: Path,
    output: Path,
    horizon: int,
    max_workers: int | None = None,
    origins_override: list[pd.Timestamp] | None = None,
) -> dict[str, pd.DataFrame | dict]:
    """Execute full OOF direct multi-horizon benchmark for a given `horizon` (7, 14, 21, or 28)."""
    t_start = time.perf_counter()
    direct_panel, raw_plants = build_direct_panel(base_path, horizon)
    global_min = min(info["min_date"] for info in raw_plants.values())
    global_max = max(info["max_date"] for info in raw_plants.values())
    origins = origins_override or build_outer_origins(global_min, global_max, horizon)

    plants_list = sorted(raw_plants.keys())
    groups_list = sorted({info["grupo"] for info in raw_plants.values()})
    workers = max_workers or min(14, os.cpu_count() or 1)

    all_cov: list[dict] = []
    all_eval_specs: list[dict] = []
    anchor_jobs: list[tuple] = []

    for code in plants_list:
        c_inelig, e_specs, a_jobs = _plan_unit_origins("intraplanta", code, origins, horizon, direct_panel, raw_plants)
        all_cov.extend(c_inelig)
        all_eval_specs.extend(e_specs)
        anchor_jobs.extend(a_jobs)
    for group_id in groups_list:
        c_inelig, e_specs, a_jobs = _plan_unit_origins("espaciotemporal", group_id, origins, horizon, direct_panel, raw_plants)
        all_cov.extend(c_inelig)
        all_eval_specs.extend(e_specs)
        anchor_jobs.extend(a_jobs)

    # Stage 1: parallel causal hyperparameter recalibration across all (task, unit, anchor)
    if workers <= 1:
        anchor_results = [_run_anchor_recalibration_task(job) for job in anchor_jobs]
    else:
        anchor_results = Parallel(n_jobs=workers, backend="loky")(
            delayed(_run_anchor_recalibration_task)(job) for job in anchor_jobs
        )
    cfg_lookup = {key: cfgs for key, cfgs, _ in anchor_results}
    sel_lookup = {key: secs for key, _, secs in anchor_results}

    # Stage 2: parallel OOF refit + direct prediction across balanced chunks
    n_chunks = max(1, min(len(all_eval_specs), workers * 3))
    spec_chunks = [all_eval_specs[i::n_chunks] for i in range(n_chunks) if all_eval_specs[i::n_chunks]]
    if workers <= 1:
        stage2_results = [
            _run_unit_refit_batch_task((chunk, cfg_lookup, sel_lookup, horizon, direct_panel, raw_plants))
            for chunk in spec_chunks
        ]
    else:
        stage2_results = Parallel(n_jobs=workers, backend="loky")(
            delayed(_run_unit_refit_batch_task)((chunk, cfg_lookup, sel_lookup, horizon, direct_panel, raw_plants))
            for chunk in spec_chunks
        )

    all_pred, all_choices = [], []
    for res in stage2_results:
        all_cov.extend(res["coverage"])
        all_pred.extend(res["predictions"])
        all_choices.extend(res["choices"])

    coverage = pd.DataFrame(all_cov).sort_values(["tarea", "origen", "unidad", "code"]).reset_index(drop=True)
    pred = pd.DataFrame(all_pred).sort_values(["tarea", "metodo", "origen", "code"]).reset_index(drop=True)
    choices = pd.DataFrame(all_choices).sort_values(["tarea", "origen", "unidad", "metodo"]).reset_index(drop=True)

    audit = audit_direct_leakage(base_path, direct_panel, raw_plants, pred, coverage, choices, horizon)
    agg = _aggregate_direct_metrics(pred)

    output.mkdir(parents=True, exist_ok=True)
    pred.to_csv(output / "predicciones_oof.csv", index=False)
    agg["energy_total"].to_csv(output / "metricas_energia_total.csv", index=False)
    agg["by_plant"].to_csv(output / "metricas_por_planta.csv", index=False)
    agg["by_group"].to_csv(output / "metricas_por_grupo.csv", index=False)
    coverage.to_csv(output / "cobertura_folds.csv", index=False)
    choices.to_csv(output / "seleccion_hiperparametros.csv", index=False)
    audit.to_csv(output / "auditoria_fuga.csv", index=False)

    elapsed = time.perf_counter() - t_start
    source_path = Path(__file__).resolve()
    root = daily.ROOT
    manifest = {
        "target": f"energía total acumulada en ventana completa de H={horizon} días Y_H(t) = sum_{{l=1..H}} y_{{t+l}}, pronóstico directo OOF desde origen fijo",
        "strategy": "directo",
        "horizon_days": horizon,
        "weather_enabled": False,
        "weather_note": "Sin variables ni pronósticos meteorológicos; el archivo disponible solo cubre D+1.",
        "global_date_range": [global_min.strftime("%Y-%m-%d"), global_max.strftime("%Y-%m-%d")],
        "warmup_days": WARMUP_DAYS,
        "outer_origins_count": len(origins),
        "outer_origins": [o.strftime("%Y-%m-%d") for o in origins],
        "outer_designs": {
            "intraplanta": "validación temporal expansiva contigua por planta; muestras train con t_end_target <= origen",
            "espaciotemporal": "leave-one-group-out cruzado con orígenes temporales contiguos; muestras train con t_end_target <= origen",
        },
        "hyperparameter_selection": (
            f"Validación interna temporal directa específica del horizonte H={horizon} "
            f"(hasta {INNER_MAX_SPLITS} splits con gap={INNER_GAP_DAYS}d y t_end_target <= s_m - gap, máx {daily.MAX_CONFIGURATIONS} configs), "
            f"con recalibración causal cada {RECALIBRATION_DAYS} días y reajuste de parámetros en cada origen OOF (Tashman, 2000)."
        ),
        "methods": daily.METHODS,
        "method_display_names": daily.DISPLAY_LABELS,
        "custom_model_names": daily.MODEL_LABELS,
        "seed": daily.SEED,
        "elapsed_seconds": round(elapsed, 2),
        "leakage_audit": audit.to_dict("records"),
        "provenance": {
            "source_file": str(source_path.relative_to(root)),
            "source_sha256": _file_sha256(source_path),
            "recursive_source_sha256": _file_sha256(Path(__file__).resolve().with_name("produccion_multihorizonte.py")),
            "daily_source_sha256": _file_sha256(Path(daily.__file__).resolve()),
            "generation_file": str(Path(base_path).resolve().relative_to(root)),
            "generation_sha256": _file_sha256(Path(base_path)),
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
                "xgboost": xgboost.__version__,
            },
        },
        "limitations": [
            "No se verificó la hora de publicación XM de kWh(D-1); causalidad por fecha no garantiza disponibilidad operacional.",
            "Sin meteorología: los modelos solo reciben historial de producción, capacidad congelada al origen y calendario.",
            "Cobertura escalonada por planta (e.g., 5F43 inicia en 2026-02-21); unidades sin historia suficiente se registran sin imputación.",
            "La capacidad futura se mantiene fija al último valor observado al origen.",
            "Tanto entrenamiento como evaluación exigen ventanas completas (H/H días válidos); las ventanas incompletas no se imputan.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "predictions": pred,
        "energy_total": agg["energy_total"],
        "by_plant": agg["by_plant"],
        "by_group": agg["by_group"],
        "coverage": coverage,
        "choices": choices,
        "audit": audit,
        "manifest": manifest,
    }

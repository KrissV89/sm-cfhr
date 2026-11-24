"""Recursive out-of-fold multi-horizon benchmark (7, 14, 21, 28 days) WITH causal weather.

Evaluates all 22 procedures from ``produccion_intraplanta.py`` under contiguous
expanding-window temporal cross-validation across the full historical dataset
(2024-01-01 to 2026-10-31), using strictly causal NWP weather forecasts
(NOAA GEFSv12 00Z multi-week forecasts) and causal rolling weather anomalies.
Both intraplanta and leave-one-group-out spatiotemporal designs are supported,
with horizon-specific recursive inner CV (RECALIBRATION_DAYS=168, INNER_MAX_SPLITS=2)
and a 5-part automated leakage audit including future-weather perturbation.
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
from scripts.evaluacion import produccion_multihorizonte as base_mh


HORIZONS = base_mh.HORIZONS
ENERGY_BOUND = base_mh.ENERGY_BOUND
WARMUP_DAYS = base_mh.WARMUP_DAYS
RECALIBRATION_DAYS = base_mh.RECALIBRATION_DAYS  # 168
INNER_GAP_DAYS = base_mh.INNER_GAP_DAYS          # 30
INNER_MAX_SPLITS = base_mh.INNER_MAX_SPLITS      # 2
INNER_FRACTIONS = base_mh.INNER_FRACTIONS        # (0.65, 0.85)
MIN_TRAIN_ROWS = base_mh.MIN_TRAIN_ROWS          # 60
MIN_PLANT_HISTORY_DAYS = base_mh.MIN_PLANT_HISTORY_DAYS  # 15
EWMA_ALPHAS = base_mh.EWMA_ALPHAS
KALMAN_QS = base_mh.KALMAN_QS

METEO = list(daily.METEO)
ANOM = list(daily.ANOM)
CUSTOM_WX = list(ANOM)
LIB_X = list(daily.BASE_X) + METEO + ANOM
FEATURE_COLS = (
    list(daily.BASE_X)
    + ["deviation_D"]
    + [f"ewma_{a}" for a in EWMA_ALPHAS]
    + [f"kalman_{q}" for q in KALMAN_QS]
    + METEO
    + ANOM
)


def load_causal_weather_store(
    forecast_path: Path,
    pron_data: pd.DataFrame | None = None,
) -> dict:
    """Load multi-horizon causal forecasts and precompute strictly causal baselines.

    Causality invariant:
    - Training row `(code, date)` uses the most recent prior forecast (`lead_dias in 1..7`,
      `fecha_emision < date`) and its 30-day rolling baseline over `d < date`.
    - Query `(code, origin, target_date = origin + lead)` uses the forecast issued at
      `fecha_emision == origin` and a 30-day baseline frozen at `origin` (`d <= origin`).
    """
    if pron_data is None:
        pron = pd.read_csv(
            forecast_path,
            usecols=["code", "fecha_emision", "fecha_objetivo", "lead_dias"] + METEO,
            parse_dates=["fecha_emision", "fecha_objetivo"],
        )
    else:
        pron = pron_data.copy()
        pron["fecha_emision"] = pd.to_datetime(pron["fecha_emision"])
        pron["fecha_objetivo"] = pd.to_datetime(pron["fecha_objetivo"])

    pron["code"] = pron["code"].astype(str)
    em_check = pron["fecha_emision"] <= (pron["fecha_objetivo"] - pd.to_timedelta(pron["lead_dias"], unit="D"))
    if not bool(em_check.all()):
        raise ValueError("Weather forecast file violates fecha_emision <= fecha_objetivo - lead_dias.")

    # 1. Origin-issued forecast lookup: (code, fecha_emision, fecha_objetivo) -> np.ndarray(5)
    origin_forecasts: dict[tuple[str, pd.Timestamp, pd.Timestamp], np.ndarray] = {}
    vals_mat = pron[METEO].to_numpy(dtype=float)
    codes_arr = pron["code"].to_numpy()
    em_arr = pron["fecha_emision"].to_numpy()
    obj_arr = pron["fecha_objetivo"].to_numpy()
    for i in range(len(pron)):
        origin_forecasts[(str(codes_arr[i]), pd.Timestamp(em_arr[i]), pd.Timestamp(obj_arr[i]))] = vals_mat[i]

    # 2. Causal daily historical weather series W_hist(code, date) from leads 1..7 (fecha_emision < date)
    short_lead = pron[pron["lead_dias"].between(1, 7)].sort_values(
        ["code", "fecha_objetivo", "lead_dias"]
    )
    daily_hist = short_lead.drop_duplicates(subset=["code", "fecha_objetivo"], keep="first")

    wx_hist_frames: list[pd.DataFrame] = []
    origin_baselines: dict[tuple[str, pd.Timestamp], np.ndarray] = {}

    for code, grp in daily_hist.groupby("code", sort=True):
        g = grp.set_index("fecha_objetivo").sort_index().asfreq("D")
        f = pd.DataFrame(index=g.index)
        f["code"] = code
        for raw, anom in zip(METEO, ANOM):
            f[raw] = g[raw]
            base_30 = g[raw].rolling(30, min_periods=15).mean().shift(1)
            f[anom] = g[raw] - base_30
        wx_hist_frames.append(f.reset_index(names="date"))

        # Precompute frozen 30-day baseline at each date `origin` using d in [origin - 29d, origin]
        # Note: `rolling(30, min_periods=15).mean()` at index `origin` uses `d <= origin` strictly!
        base_at_origin = g[METEO].rolling(30, min_periods=15).mean()
        b_vals = base_at_origin.to_numpy(dtype=float)
        for dt, row_vec in zip(base_at_origin.index, b_vals):
            origin_baselines[(str(code), pd.Timestamp(dt))] = row_vec

    wx_hist = pd.concat(wx_hist_frames, ignore_index=True) if wx_hist_frames else pd.DataFrame(
        columns=["code", "date"] + METEO + ANOM
    )
    return {
        "wx_hist": wx_hist,
        "origin_forecasts": origin_forecasts,
        "origin_baselines": origin_baselines,
    }


def load_generation_weather_panel(
    base_path: Path,
    forecast_path: Path,
    base_data: pd.DataFrame | None = None,
    pron_data: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, dict], dict]:
    """Build causal daily generation + weather panel and weather lookup store."""
    gen_panel, raw_plants = base_mh.load_generation_panel(base_path, base_data=base_data)
    wx_store = load_causal_weather_store(forecast_path, pron_data=pron_data)
    wx_hist = wx_store["wx_hist"]

    panel = gen_panel.merge(wx_hist, on=["code", "date"], how="inner", validate="one_to_one")
    panel = panel.dropna(subset=ANOM).sort_values(["code", "date"]).reset_index(drop=True)
    return panel, raw_plants, wx_store


class PlantRecursiveWeatherState(base_mh.PlantRecursiveState):
    """Causal O(1)-per-step recursive state with origin-issued NWP weather & causal anomalies."""

    def __init__(
        self,
        plant_info: dict,
        origin: pd.Timestamp,
        horizon: int,
        wx_store: dict,
    ) -> None:
        super().__init__(plant_info, origin, horizon)
        self.wx_store = wx_store
        self.baseline = wx_store["origin_baselines"].get(
            (self.code, self.origin), np.full(len(METEO), np.nan, dtype=float)
        )

    def next_features(self, target_date: pd.Timestamp) -> dict:
        row = super().next_features(target_date)
        raw_wx = self.wx_store["origin_forecasts"].get(
            (self.code, self.origin, pd.Timestamp(target_date)),
            np.full(len(METEO), np.nan, dtype=float),
        )
        anom_wx = raw_wx - self.baseline
        for i, col in enumerate(METEO):
            row[col] = float(raw_wx[i]) if np.isfinite(raw_wx[i]) else np.nan
        for i, col in enumerate(ANOM):
            row[col] = float(anom_wx[i]) if np.isfinite(anom_wx[i]) else np.nan
        return row


def fit_model(method: str, cfg: dict, train: pd.DataFrame) -> dict:
    """Fit a procedure once on `train` (weather_on=True), matching daily._fit_predict."""
    if method == "Media móvil de 30 días":
        return {"method": method}

    if method in daily.LIBRARIES:
        model = daily._library(method, cfg, seed=daily.SEED)
        model.fit(train[LIB_X], train["y"])
        return {"method": method, "model": model}

    X_raw = train[CUSTOM_WX].to_numpy(dtype=float)
    mu = np.nanmedian(X_raw, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    X_imp = np.where(np.isfinite(X_raw), X_raw, mu)
    scaler = StandardScaler().fit(X_imp)
    X = scaler.transform(X_imp)

    r = train["y"].to_numpy(dtype=float) - train["media30_kwh"].to_numpy(dtype=float)
    r_center = float(np.mean(r))
    r_scale = float(np.std(r))
    r_scale = r_scale if np.isfinite(r_scale) and r_scale > 1e-10 else 1.0
    r_std = (r - r_center) / r_scale

    state: dict = {
        "method": method,
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
        lv = train[f"ewma_{cfg['alpha']}"].to_numpy(dtype=float)
        target = (train["y"].to_numpy(dtype=float) - lv - r_center) / r_scale
        state["est"] = Ridge(alpha=1).fit(X, target)
        state["level_col"] = f"ewma_{cfg['alpha']}"
    elif method == "KLAC":
        col = f"kalman_{cfg['q']}"
        lv = train[col].fillna(train["media30_kwh"]).to_numpy(dtype=float)
        target = (train["y"].to_numpy(dtype=float) - lv - r_center) / r_scale
        state["est"] = Ridge(alpha=1).fit(X, target)
        state["level_col"] = col
    elif method == "RARC":
        state["est"] = HuberRegressor(epsilon=cfg["epsilon"], max_iter=300).fit(X, r_std)
    elif method == "MARB":
        constraints = (1,) + (0,) * (X.shape[1] - 1)
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
        rad = ANOM.index("anom_rad")
        cutoff = train["date"].max() - pd.Timedelta(days=w - 1)
        mask = train["date"].ge(cutoff).to_numpy()
        a = X[mask, rad : rad + 1]
        state["est"] = Ridge(alpha=1).fit(a, r_std[mask])
        state["rad_idx"] = rad
    elif method == "HARC":
        groups = pd.Index(pd.unique(train["grupo"]))
        plants = pd.Index(pd.unique(train["code"]))
        use_group = len(groups) > 1
        A = base_mh._harc_design(train, X, groups, plants, use_group)
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


def predict_batch(fitted: dict, query: pd.DataFrame) -> np.ndarray:
    """Predict a batch of query rows using a pre-fitted weather-enabled model."""
    method = fitted["method"]
    if method == "Media móvil de 30 días":
        return daily._clip_physical(query["media30_kwh"].to_numpy(dtype=float), query)

    if method in daily.LIBRARIES:
        raw_pred = fitted["model"].predict(query[LIB_X])
        return daily._clip_physical(raw_pred, query)

    Xt_raw = query[CUSTOM_WX].to_numpy(dtype=float)
    Xt_imp = np.where(np.isfinite(Xt_raw), Xt_raw, fitted["mu"])
    Xt = fitted["scaler"].transform(Xt_imp)
    base = query["media30_kwh"].to_numpy(dtype=float)
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
        lvt = query[fitted["level_col"]].to_numpy(dtype=float)
        pred = lvt + r_center + r_scale * fitted["est"].predict(Xt)
    elif method == "KLAC":
        lvt = query[fitted["level_col"]].fillna(query["media30_kwh"]).to_numpy(dtype=float)
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
        rad = fitted["rad_idx"]
        pred = base + r_center + r_scale * fitted["est"].predict(Xt[:, rad : rad + 1])
    elif method == "HARC":
        At = base_mh._harc_design(query, Xt, fitted["groups"], fitted["plants"], fitted["use_group"])
        pred = base + r_center + r_scale * (At @ fitted["beta"])
    elif method == "WCFC":
        a = fitted["est_ridge"].predict(Xt)
        b = fitted["est_huber"].predict(Xt)
        w = fitted["weight_huber"]
        pred = base + r_center + r_scale * ((1.0 - w) * a + w * b)
    else:
        raise KeyError(method)

    return daily._clip_physical(np.asarray(pred, dtype=float), query)


def _plant_has_weather_and_history(
    plant_info: dict,
    origin: pd.Timestamp,
    horizon: int,
    wx_store: dict,
) -> tuple[bool, str]:
    ok, reason = base_mh._plant_has_history_and_valid_future(plant_info, origin, horizon)
    if not ok:
        return False, reason
    code = plant_info["code"]
    base_vec = wx_store["origin_baselines"].get((code, pd.Timestamp(origin)))
    if base_vec is None or not np.isfinite(base_vec).all():
        return False, "sin_linea_base_clima_al_origen"
    return True, "elegible"


def _rollout_recursive_weather_batch(
    fitted: dict,
    plant_infos: list[dict],
    origin: pd.Timestamp,
    horizon: int,
    wx_store: dict,
) -> np.ndarray:
    states = [PlantRecursiveWeatherState(p, origin, horizon, wx_store) for p in plant_infos]
    preds = np.empty((len(states), horizon), dtype=float)
    for lead in range(1, horizon + 1):
        target_date = origin + pd.Timedelta(days=lead)
        query = pd.DataFrame([st.next_features(target_date) for st in states])
        yhat = predict_batch(fitted, query)
        preds[:, lead - 1] = yhat
        for i, st in enumerate(states):
            st.step_update(float(yhat[i]))
    return preds


def build_inner_origins(
    train: pd.DataFrame,
    raw_plants: dict[str, dict],
    origin: pd.Timestamp,
    horizon: int,
    wx_store: dict,
    gap_days: int = INNER_GAP_DAYS,
) -> list[tuple[pd.DataFrame, pd.Timestamp, list[dict]]]:
    """Build up to 2 recursive internal validation splits strictly <= origin."""
    dates = np.sort(train["date"].unique())
    if len(dates) == 0:
        return []
    train_codes = sorted(train["code"].astype(str).unique())
    splits = []
    seen_origins = set()
    for frac in INNER_FRACTIONS:
        idx = min(int(round(len(dates) * frac)), len(dates) - 1)
        if idx < 0:
            continue
        s_m = pd.Timestamp(dates[idx])
        if s_m + pd.Timedelta(days=horizon) > origin or s_m in seen_origins:
            continue
        cutoff = s_m - pd.Timedelta(days=gap_days)
        inner_train = train[train["date"] <= cutoff]
        if len(inner_train) < 30:
            continue
        eligible_infos = []
        for code in train_codes:
            ok, _ = _plant_has_weather_and_history(raw_plants[code], s_m, horizon, wx_store)
            if ok:
                eligible_infos.append(raw_plants[code])
        if not eligible_infos:
            continue
        seen_origins.add(s_m)
        splits.append((inner_train, s_m, eligible_infos))
    return splits


def choose_hyperparameters_recursive(
    method: str,
    train: pd.DataFrame,
    origin: pd.Timestamp,
    horizon: int,
    inner_splits: list[tuple[pd.DataFrame, pd.Timestamp, list[dict]]],
    wx_store: dict,
) -> dict:
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
        for inner_train, s_m, elig_infos in inner_splits:
            try:
                fitted = fit_model(method, cfg, inner_train)
                preds = _rollout_recursive_weather_batch(fitted, elig_infos, s_m, horizon, wx_store)
                losses.append(base_mh._evaluate_recursive_macro_mae(preds, elig_infos, s_m, horizon))
            except (ValueError, np.linalg.LinAlgError):
                losses.append(np.inf)
        scores.append(float(np.mean(losses)) if losses else np.inf)
    return configs[int(np.argmin(scores))]


def _plan_unit_origins(
    task: str,
    unit: str | int,
    origins: list[pd.Timestamp],
    horizon: int,
    panel: pd.DataFrame,
    raw_plants: dict[str, dict],
    wx_store: dict,
) -> tuple[list[dict], list[dict], list[tuple]]:
    cov_inelig: list[dict] = []
    eval_specs: list[dict] = []
    anchor_jobs: list[tuple] = []

    last_anchor: pd.Timestamp | None = None
    last_inner_count: int = 0

    for origin in origins:
        target_start = origin + pd.Timedelta(days=1)
        target_end = origin + pd.Timedelta(days=horizon)

        if task == "intraplanta":
            code = str(unit)
            group_id = raw_plants[code]["grupo"]
            unit_name = code
            train = panel[(panel["code"] == code) & (panel["date"] <= origin)]
            ok_plant, reason = _plant_has_weather_and_history(raw_plants[code], origin, horizon, wx_store)
            if len(train) < MIN_TRAIN_ROWS:
                cov_inelig.append({
                    "tarea": task, "origen": origin, "unidad": unit_name, "code": code, "grupo": group_id,
                    "train_rows": len(train), "inner_splits": 0, "estado": "historia_entrenamiento_menor_60",
                    "train_end": train["date"].max() if len(train) else pd.NaT,
                    "test_start": target_start, "test_end": target_end,
                })
                continue
            if not ok_plant:
                cov_inelig.append({
                    "tarea": task, "origen": origin, "unidad": unit_name, "code": code, "grupo": group_id,
                    "train_rows": len(train), "inner_splits": 0, "estado": reason,
                    "train_end": train["date"].max(), "test_start": target_start, "test_end": target_end,
                })
                continue
            pred_codes = [code]
        else:
            group_id = int(unit)
            unit_name = f"grupo_{group_id}"
            train = panel[(panel["grupo"] != group_id) & (panel["date"] <= origin)]
            group_codes = sorted(c for c, info in raw_plants.items() if info["grupo"] == group_id)
            pred_codes = []
            local_cov = []
            for code in group_codes:
                ok_plant, reason = _plant_has_weather_and_history(raw_plants[code], origin, horizon, wx_store)
                if ok_plant:
                    pred_codes.append(code)
                else:
                    local_cov.append({
                        "tarea": task, "origen": origin, "unidad": unit_name, "code": code, "grupo": group_id,
                        "train_rows": len(train), "inner_splits": 0, "estado": reason,
                        "train_end": train["date"].max() if len(train) else pd.NaT,
                        "test_start": target_start, "test_end": target_end,
                    })
            if len(train) < MIN_TRAIN_ROWS or train["grupo"].nunique() < 1 or not pred_codes:
                if not local_cov:
                    local_cov.append({
                        "tarea": task, "origen": origin, "unidad": unit_name, "code": "", "grupo": group_id,
                        "train_rows": len(train), "inner_splits": 0, "estado": "sin_plantas_elegibles_o_train_insuficiente",
                        "train_end": train["date"].max() if len(train) else pd.NaT,
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
            inner_splits = build_inner_origins(train, raw_plants, origin, horizon, wx_store)
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
                    {"train": base_mh._key_signature(a), "val_origin": s_m.strftime("%Y-%m-%d")}
                    for a, s_m, _ in inner_splits
                ])
                anchor_jobs.append((task, unit, origin, horizon, panel, raw_plants, wx_store))
        else:
            recalibrated = False
            anchor_used = last_anchor
            n_inner_used = last_inner_count
            inner_sig = "reutiliza_ancla"

        eval_specs.append({
            "task": task, "unit": unit, "unit_name": unit_name, "origin": origin,
            "pred_codes": pred_codes, "inner_splits_count": n_inner_used,
            "recalibrated": recalibrated, "anchor_origin": anchor_used,
            "train_rows": len(train), "train_end": train["date"].max(),
            "test_start": target_start, "test_end": target_end,
            "train_key_sha256": base_mh._key_signature(train), "inner_key_sha256": inner_sig,
        })

    return cov_inelig, eval_specs, anchor_jobs


def _run_anchor_recalibration_task(args: tuple) -> tuple[tuple, dict[str, dict], dict[str, float]]:
    task, unit, anchor_origin, horizon, panel, raw_plants, wx_store = args
    if task == "intraplanta":
        train = panel[(panel["code"] == str(unit)) & (panel["date"] <= anchor_origin)].copy()
    else:
        train = panel[(panel["grupo"] != int(unit)) & (panel["date"] <= anchor_origin)].copy()
    inner_splits = build_inner_origins(train, raw_plants, anchor_origin, horizon, wx_store)
    cfgs: dict[str, dict] = {}
    sel_secs: dict[str, float] = {}
    for method in daily.METHODS:
        t0 = time.perf_counter()
        cfgs[method] = choose_hyperparameters_recursive(
            method, train, anchor_origin, horizon, inner_splits, wx_store
        )
        sel_secs[method] = time.perf_counter() - t0
    return (task, unit, anchor_origin), cfgs, sel_secs


def _run_unit_refit_batch_task(args: tuple) -> dict:
    specs, cfg_lookup, sel_lookup, horizon, panel, raw_plants, wx_store = args
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

        if task == "intraplanta":
            train = panel[(panel["code"] == str(unit)) & (panel["date"] <= origin)].copy()
        else:
            train = panel[(panel["grupo"] != int(unit)) & (panel["date"] <= origin)].copy()

        for info in pred_infos:
            cov_rows.append({
                "tarea": task, "origen": origin, "unidad": unit_name, "code": info["code"], "grupo": info["grupo"],
                "train_rows": sp["train_rows"], "inner_splits": sp["inner_splits_count"], "estado": "evaluado",
                "train_end": sp["train_end"], "test_start": sp["test_start"], "test_end": sp["test_end"],
                "train_key_sha256": sp["train_key_sha256"], "inner_key_sha256": sp["inner_key_sha256"],
            })

        cfgs = cfg_lookup[(task, unit, anchor_origin)]
        sel_secs = sel_lookup[(task, unit, anchor_origin)]

        for method in daily.METHODS:
            cfg = cfgs[method]
            sel_sec = sel_secs[method] if sp["recalibrated"] else 0.0

            t1 = time.perf_counter()
            fitted = fit_model(method, cfg, train)
            preds = _rollout_recursive_weather_batch(fitted, pred_infos, origin, horizon, wx_store)
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
                n_hist = (origin - info["min_date"]).days + 1
                cap_at_origin = PlantRecursiveWeatherState(info, origin, horizon, wx_store).cap_at_origin
                denom = cap_at_origin * ENERGY_BOUND
                for lead in range(1, horizon + 1):
                    target_date = origin + pd.Timedelta(days=lead)
                    actual_y = float(info["kwh_valid"][n_hist + lead - 1])
                    pred_val = float(preds[i, lead - 1])
                    err = abs(pred_val - actual_y) if np.isfinite(actual_y) else np.nan
                    pred_rows.append({
                        "metodo": method,
                        "tarea": task,
                        "origen": origin,
                        "unidad_fold": unit_name,
                        "code": info["code"],
                        "grupo": info["grupo"],
                        "fecha_objetivo": target_date,
                        "lead_dias": lead,
                        "semana_lead": (lead - 1) // 7 + 1,
                        "prediccion_kwh": pred_val,
                        "y_kwh": actual_y,
                        "error_abs_kwh": err,
                        "cap_MW_origen": cap_at_origin,
                        "error_capacidad_relativa": (err / denom) if (np.isfinite(err) and denom > 0) else np.nan,
                    })

    return {"coverage": cov_rows, "predictions": pred_rows, "choices": choice_rows}


def audit_multihorizon_weather_leakage(
    base_path: Path,
    forecast_path: Path,
    panel: pd.DataFrame,
    raw_plants: dict[str, dict],
    wx_store: dict,
    pred: pd.DataFrame,
    coverage: pd.DataFrame,
    choices: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Run 5 automated leakage & equivalence controls including future-weather perturbation."""
    records = []
    # Use an origin from the OOF set (2025-11-02 is a valid GEFS origin: 2024-03-31 + 581 days)
    sample_origin = pd.Timestamp("2025-11-02")

    # 1. Equivalence fit_model/predict_batch vs daily._fit_predict(weather_on=True) across all 22 methods
    train_eq = panel[(panel["grupo"] != 1) & (panel["date"] <= sample_origin)].copy()
    test_eq = panel[(panel["grupo"] == 1) & (panel["date"] > sample_origin)].head(10).copy()
    max_diff_methods = 0.0
    for method in daily.METHODS:
        cfg = daily.GRIDS[method][0]
        p_daily = daily._fit_predict(method, cfg, train_eq, test_eq, weather_on=True)
        p_fast = predict_batch(fit_model(method, cfg, train_eq), test_eq)
        diff = float(np.max(np.abs(p_daily - p_fast)))
        max_diff_methods = max(max_diff_methods, diff)
    assert max_diff_methods < 1e-8, f"Equivalence failed: max_diff={max_diff_methods}"
    records.append({
        "control": "equivalencia_fit_predict_22_metodos_clima",
        "detalle": f"max_diff={max_diff_methods:.2e}",
        "estado": "superado",
    })

    # 2. Equivalence O(1) PlantRecursiveWeatherState vs panel generation features at origin + 1
    max_diff_state = 0.0
    gen_only_cols = [c for c in FEATURE_COLS if c not in METEO and c not in ANOM]
    for code in sorted(raw_plants)[:5]:
        info = raw_plants[code]
        if info["min_date"] > sample_origin - pd.Timedelta(days=60):
            continue
        st = PlantRecursiveWeatherState(info, sample_origin, horizon, wx_store)
        feat = st.next_features(sample_origin + pd.Timedelta(days=1))
        ref = panel[(panel["code"] == code) & (panel["date"] == sample_origin + pd.Timedelta(days=1))]
        if not ref.empty:
            ref_row = ref.iloc[0]
            for col in gen_only_cols:
                v1, v2 = float(feat[col]), float(ref_row[col])
                if np.isfinite(v1) and np.isfinite(v2):
                    max_diff_state = max(max_diff_state, abs(v1 - v2))
            assert all(np.isfinite(float(feat[c])) for c in METEO + ANOM), f"Non-finite weather at {code}"
    assert max_diff_state < 1e-9, f"State equivalence failed: max_diff={max_diff_state}"
    records.append({
        "control": "equivalencia_estado_recursivo_O1_clima",
        "detalle": f"max_diff={max_diff_state:.2e}",
        "estado": "superado",
    })

    # 3. Future perturbation test (date > origin for kwh/cap_MW AND fecha_emision > origin for weather)
    base_df = pd.read_csv(
        base_path,
        usecols=["code", "date", "kwh", "horas_xm", "cap_MW", "lat", "lon", "grupo"],
        parse_dates=["date"],
    )
    pron_df = pd.read_csv(
        forecast_path,
        usecols=["code", "fecha_emision", "fecha_objetivo", "lead_dias"] + METEO,
        parse_dates=["fecha_emision", "fecha_objetivo"],
    )
    altered_base = base_df.copy()
    fut_mask = altered_base["date"] > sample_origin
    altered_base.loc[fut_mask, "kwh"] = altered_base.loc[fut_mask, "kwh"] * 1.5 + 100.0
    altered_base.loc[fut_mask, "cap_MW"] = altered_base.loc[fut_mask, "cap_MW"] * 1.5

    altered_pron = pron_df.copy()
    fut_wx_mask = altered_pron["fecha_emision"] > sample_origin
    for col in METEO:
        altered_pron.loc[fut_wx_mask, col] = altered_pron.loc[fut_wx_mask, col] * 5.0 + 50.0

    panel_alt, plants_alt, wx_alt = load_generation_weather_panel(
        base_path, forecast_path, base_data=altered_base, pron_data=altered_pron
    )
    train_alt = panel_alt[(panel_alt["grupo"] != 1) & (panel_alt["date"] <= sample_origin)].copy()
    elig_orig = [raw_plants[c] for c in sorted(raw_plants) if raw_plants[c]["grupo"] == 1]
    elig_alt = [plants_alt[c] for c in sorted(plants_alt) if plants_alt[c]["grupo"] == 1]
    for method in ("RLAC", "KLAC", "XGBoost", "RQRF"):
        cfg = daily.GRIDS[method][0]
        r1 = _rollout_recursive_weather_batch(fit_model(method, cfg, train_eq), elig_orig, sample_origin, horizon, wx_store)
        r2 = _rollout_recursive_weather_batch(fit_model(method, cfg, train_alt), elig_alt, sample_origin, horizon, wx_alt)
        np.testing.assert_allclose(r1, r2, atol=0.0, rtol=0.0)
    records.append({
        "control": "perturbacion_futuro_kwh_cap_y_clima",
        "detalle": "atol=0.0 en RLAC, KLAC, XGBoost, RQRF",
        "estado": "superado",
    })

    # 4. Prefix truncation test (only generation <= origin and weather issued <= origin)
    prefix_base = base_df[base_df["date"] <= sample_origin].copy()
    prefix_pron = pron_df[pron_df["fecha_emision"] <= sample_origin].copy()
    panel_pref, plants_pref, wx_pref = load_generation_weather_panel(
        base_path, forecast_path, base_data=prefix_base, pron_data=prefix_pron
    )
    train_pref = panel_pref[(panel_pref["grupo"] != 1) & (panel_pref["date"] <= sample_origin)].copy()
    elig_pref = [plants_pref[c] for c in sorted(plants_pref) if plants_pref[c]["grupo"] == 1]
    for method in ("RLAC", "KLAC", "Ridge"):
        cfg = daily.GRIDS[method][0]
        r1 = _rollout_recursive_weather_batch(fit_model(method, cfg, train_eq), elig_orig, sample_origin, horizon, wx_store)
        r3 = _rollout_recursive_weather_batch(fit_model(method, cfg, train_pref), elig_pref, sample_origin, horizon, wx_pref)
        np.testing.assert_allclose(r1, r3, atol=0.0, rtol=0.0)
    records.append({
        "control": "reconstruccion_prefijo_temporal_clima",
        "detalle": "atol=0.0 en RLAC, KLAC, Ridge",
        "estado": "superado",
    })

    # 5. OOF integrity & causal hyperparameter anchors
    ev = coverage[coverage["estado"] == "evaluado"]
    assert (pd.to_datetime(ev["train_end"]) <= pd.to_datetime(ev["origen"])).all(), "train_end > origin"
    assert (pd.to_datetime(ev["origen"]) < pd.to_datetime(ev["test_start"])).all(), "origin >= test_start"
    if not choices.empty:
        assert (
            pd.to_datetime(choices["origen_ancla_hiperparametros"]) <= pd.to_datetime(choices["origen"])
        ).all(), "origen_ancla_hiperparametros > origen"
    dups = int(pred.duplicated(subset=["metodo", "tarea", "code", "fecha_objetivo"]).sum())
    assert dups == 0, f"Duplicate OOF predictions found: {dups}"
    records.append({
        "control": "integridad_oof_y_causalidad_splits",
        "detalle": f"duplicados_oof={dups}, folds_evaluados={len(ev)}, anclas_causales_ok=True",
        "estado": "superado",
    })

    return pd.DataFrame(records)


def run_horizon_weather_benchmark(
    base_path: Path,
    forecast_path: Path,
    output: Path,
    horizon: int,
    max_workers: int | None = None,
    origins_override: list[pd.Timestamp] | None = None,
) -> dict[str, pd.DataFrame | dict]:
    """Execute full OOF multi-horizon benchmark WITH causal NWP weather for a given `horizon`."""
    t_start = time.perf_counter()
    panel, raw_plants, wx_store = load_generation_weather_panel(base_path, forecast_path)
    global_min = min(info["min_date"] for info in raw_plants.values())
    global_max = max(info["max_date"] for info in raw_plants.values())
    origins = origins_override or base_mh.build_outer_origins(global_min, global_max, horizon)

    plants_list = sorted(raw_plants.keys())
    groups_list = sorted({info["grupo"] for info in raw_plants.values()})
    workers = max_workers or min(14, os.cpu_count() or 1)

    all_cov: list[dict] = []
    all_eval_specs: list[dict] = []
    anchor_jobs: list[tuple] = []

    for code in plants_list:
        c_inelig, e_specs, a_jobs = _plan_unit_origins(
            "intraplanta", code, origins, horizon, panel, raw_plants, wx_store
        )
        all_cov.extend(c_inelig)
        all_eval_specs.extend(e_specs)
        anchor_jobs.extend(a_jobs)
    for group_id in groups_list:
        c_inelig, e_specs, a_jobs = _plan_unit_origins(
            "espaciotemporal", group_id, origins, horizon, panel, raw_plants, wx_store
        )
        all_cov.extend(c_inelig)
        all_eval_specs.extend(e_specs)
        anchor_jobs.extend(a_jobs)

    if workers <= 1:
        anchor_results = [_run_anchor_recalibration_task(job) for job in anchor_jobs]
    else:
        anchor_results = Parallel(n_jobs=workers, backend="loky")(
            delayed(_run_anchor_recalibration_task)(job) for job in anchor_jobs
        )
    cfg_lookup = {key: cfgs for key, cfgs, _ in anchor_results}
    sel_lookup = {key: secs for key, _, secs in anchor_results}

    n_chunks = max(1, min(len(all_eval_specs), workers * 3))
    spec_chunks = [all_eval_specs[i::n_chunks] for i in range(n_chunks) if all_eval_specs[i::n_chunks]]
    if workers <= 1:
        stage2_results = [
            _run_unit_refit_batch_task((chunk, cfg_lookup, sel_lookup, horizon, panel, raw_plants, wx_store))
            for chunk in spec_chunks
        ]
    else:
        stage2_results = Parallel(n_jobs=workers, backend="loky")(
            delayed(_run_unit_refit_batch_task)((chunk, cfg_lookup, sel_lookup, horizon, panel, raw_plants, wx_store))
            for chunk in spec_chunks
        )

    all_pred, all_choices = [], []
    for res in stage2_results:
        all_cov.extend(res["coverage"])
        all_pred.extend(res["predictions"])
        all_choices.extend(res["choices"])

    coverage = pd.DataFrame(all_cov).sort_values(["tarea", "origen", "unidad", "code"]).reset_index(drop=True)
    pred = pd.DataFrame(all_pred).sort_values(["tarea", "metodo", "origen", "code", "lead_dias"]).reset_index(drop=True)
    choices = pd.DataFrame(all_choices).sort_values(["tarea", "origen", "unidad", "metodo"]).reset_index(drop=True)

    audit = audit_multihorizon_weather_leakage(
        base_path, forecast_path, panel, raw_plants, wx_store, pred, coverage, choices, horizon
    )
    agg = base_mh._aggregate_metrics(pred, horizon)

    output.mkdir(parents=True, exist_ok=True)
    pred.to_csv(output / "predicciones_oof.csv", index=False)
    agg["global_daily"].to_csv(output / "metricas_diarias_globales.csv", index=False)
    agg["by_plant"].to_csv(output / "metricas_por_planta.csv", index=False)
    agg["by_group"].to_csv(output / "metricas_por_grupo.csv", index=False)
    agg["by_lead"].to_csv(output / "metricas_por_lead.csv", index=False)
    agg["energy_total"].to_csv(output / "metricas_energia_total.csv", index=False)
    coverage.to_csv(output / "cobertura_folds.csv", index=False)
    choices.to_csv(output / "seleccion_hiperparametros.csv", index=False)
    audit.to_csv(output / "auditoria_fuga.csv", index=False)

    elapsed = time.perf_counter() - t_start
    source_path = Path(__file__).resolve()
    root = daily.ROOT
    manifest = {
        "target": "generación eléctrica diaria observada (kWh/planta-día), pronóstico recursivo OOF desde origen fijo con clima NWP causal",
        "horizon_days": horizon,
        "weather_enabled": True,
        "weather_source": str(Path(forecast_path).resolve().relative_to(root)),
        "global_date_range": [global_min.strftime("%Y-%m-%d"), global_max.strftime("%Y-%m-%d")],
        "warmup_days": WARMUP_DAYS,
        "outer_origins_count": len(origins),
        "outer_origins": [o.strftime("%Y-%m-%d") for o in origins],
        "outer_designs": {
            "intraplanta": "validación temporal expansiva contigua por planta; train <= origen",
            "espaciotemporal": "leave-one-group-out cruzado con orígenes temporales contiguos; historia previa del grupo retenido solo para rezagos",
        },
        "hyperparameter_selection": (
            f"Validación interna temporal recursiva específica del horizonte H={horizon} "
            f"(hasta {INNER_MAX_SPLITS} splits con gap={INNER_GAP_DAYS}d, máx {daily.MAX_CONFIGURATIONS} configs), "
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
            "source_sha256": base_mh._file_sha256(source_path),
            "daily_source_sha256": base_mh._file_sha256(Path(daily.__file__).resolve()),
            "base_mh_source_sha256": base_mh._file_sha256(Path(base_mh.__file__).resolve()),
            "generation_file": str(Path(base_path).resolve().relative_to(root)),
            "generation_sha256": base_mh._file_sha256(Path(base_path)),
            "weather_file": str(Path(forecast_path).resolve().relative_to(root)),
            "weather_sha256": base_mh._file_sha256(Path(forecast_path)),
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
            "Los pronósticos meteorológicos provienen de NOAA GEFSv12 (00Z, malla 0.5°) interpolados bilinealmente.",
            "Cobertura escalonada por planta y disponibilidad de línea base climática causal (>=15 días) al inicio de 2024.",
            "La capacidad futura se mantiene fija al último valor observado al origen.",
            "El error de energía total acumulada se evalúa exclusivamente en trayectorias con observación real completa (H/H días).",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "predictions": pred,
        "global_daily": agg["global_daily"],
        "by_plant": agg["by_plant"],
        "by_group": agg["by_group"],
        "by_lead": agg["by_lead"],
        "energy_total": agg["energy_total"],
        "coverage": coverage,
        "choices": choices,
        "audit": audit,
        "manifest": manifest,
    }

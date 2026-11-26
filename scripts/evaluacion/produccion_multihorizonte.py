"""Recursive out-of-fold multi-horizon benchmark (7, 14, 21, 28 days) without weather.

Evaluates all 22 procedures from ``produccion_intraplanta.py`` under contiguous
expanding-window temporal cross-validation across the full historical dataset
(2024-01-01 to 2026-10-31). Both intraplanta and leave-one-group-out
spatiotemporal designs are supported, with horizon-specific recursive inner CV
for hyperparameter selection and strict future-masking / leakage audits.
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


HORIZONS = (7, 14, 21, 28)
ENERGY_BOUND = 24_000.0
WARMUP_DAYS = 90
RECALIBRATION_DAYS = 168
INNER_GAP_DAYS = 30
INNER_MAX_SPLITS = 2
INNER_FRACTIONS = (0.65, 0.85)
MIN_TRAIN_ROWS = 60
MIN_PLANT_HISTORY_DAYS = 15
EWMA_ALPHAS = (0.03, 0.07, 0.15, 0.3)
KALMAN_QS = (0.0001, 0.001, 0.01, 0.1)
CUSTOM_WX = ["kwh_D", "media7_kwh", "month_sin", "month_cos", "doy_sin", "doy_cos"]
FEATURE_COLS = (
    daily.BASE_X
    + ["deviation_D"]
    + [f"ewma_{a}" for a in EWMA_ALPHAS]
    + [f"kalman_{q}" for q in KALMAN_QS]
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key_signature(frame: pd.DataFrame) -> str:
    if frame.empty:
        return hashlib.sha256(b"").hexdigest()
    keys = frame[["code", "date"]].copy().sort_values(["code", "date"])
    keys["date"] = pd.to_datetime(keys.date).dt.strftime("%Y-%m-%d")
    return hashlib.sha256(keys.to_csv(index=False).encode("utf-8")).hexdigest()


def _compute_kalman_series(y_unit: np.ndarray, q: float) -> np.ndarray:
    """Exact forward-only local-level Kalman filter on capacity-normalized energy."""
    level = np.nan
    var = 1.0
    out = np.empty(len(y_unit), dtype=float)
    for i, val in enumerate(y_unit):
        if np.isfinite(level):
            var += q
        if np.isfinite(val):
            if not np.isfinite(level):
                level = float(val)
                var = 0.0
            else:
                gain = var / (var + 1.0)
                level += gain * (float(val) - level)
                var *= 1.0 - gain
        out[i] = level
    return out


def load_generation_panel(
    base_path: Path,
    base_data: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Build causal daily panel across the full date range without weather data."""
    if base_data is None:
        base = pd.read_csv(
            base_path,
            usecols=["code", "date", "kwh", "horas_xm", "cap_MW", "lat", "lon", "grupo"],
            parse_dates=["date"],
        )
    else:
        base = base_data.copy()
        base["date"] = pd.to_datetime(base["date"])
    base["code"] = base["code"].astype(str)
    base = base.sort_values(["code", "date"]).reset_index(drop=True)

    raw_plants: dict[str, dict] = {}
    rows: list[pd.DataFrame] = []

    for code, src in base.groupby("code", sort=True):
        g = src.set_index("date").asfreq("D")
        h = g["kwh"].where(g["horas_xm"] > 6)
        cap = g["cap_MW"]
        group_id = int(src["grupo"].iloc[0])

        r = pd.DataFrame(index=g.index)
        r["code"] = code
        r["grupo"] = group_id
        r["lat"] = float(src["lat"].iloc[0])
        r["lon"] = float(src["lon"].iloc[0])
        r["y"] = h
        r["horas_xm"] = g["horas_xm"]
        r["cap_MW_eval"] = cap
        r["kwh_D"] = h.shift(1)
        r["media7_kwh"] = h.rolling(7, min_periods=4).mean().shift(1)
        r["media30_kwh"] = h.rolling(30, min_periods=15).mean().shift(1)
        r["cap_MW_D"] = cap.shift(1)
        r["deviation_D"] = r["kwh_D"] - h.rolling(7, min_periods=4).mean().shift(2)
        for alpha in EWMA_ALPHAS:
            r[f"ewma_{alpha}"] = h.ewm(alpha=alpha, adjust=False, min_periods=15).mean().shift(1)

        denom = (cap * ENERGY_BOUND).where(cap > 0)
        y_unit = (h / denom).to_numpy(dtype=float)
        for q in KALMAN_QS:
            levels = pd.Series(_compute_kalman_series(y_unit, q), index=g.index).shift(1)
            r[f"kalman_{q}"] = levels * r["cap_MW_D"] * ENERGY_BOUND

        r["date"] = r.index
        doy = r["date"].dt.dayofyear
        r["month_sin"] = np.sin(2.0 * np.pi * r["date"].dt.month / 12.0)
        r["month_cos"] = np.cos(2.0 * np.pi * r["date"].dt.month / 12.0)
        r["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
        r["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)

        raw_plants[code] = {
            "code": code,
            "grupo": group_id,
            "min_date": g.index.min(),
            "max_date": g.index.max(),
            "dates": g.index,
            "kwh_valid": h.to_numpy(dtype=float),
            "cap_MW": cap.to_numpy(dtype=float),
        }

        ready = (g["horas_xm"] > 6) & r["media30_kwh"].notna() & r["y"].notna()
        rows.append(r.loc[ready].reset_index(drop=True))

    panel = pd.concat(rows, ignore_index=True).sort_values(["code", "date"]).reset_index(drop=True)
    return panel, raw_plants


def build_outer_origins(
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
    horizon: int,
    warmup_days: int = WARMUP_DAYS,
) -> list[pd.Timestamp]:
    """Contiguous non-overlapping OOF origins: t_k = t0 + k * H with t_k + H <= max_date."""
    t0 = pd.Timestamp(min_date) + pd.Timedelta(days=warmup_days)
    max_d = pd.Timestamp(max_date)
    origins: list[pd.Timestamp] = []
    cur = t0
    while cur + pd.Timedelta(days=horizon) <= max_d:
        origins.append(cur)
        cur = cur + pd.Timedelta(days=horizon)
    return origins


class PlantRecursiveState:
    """O(1)-per-step causal state for recursive multi-step forecasting from a fixed origin.

    Initialized strictly from observations <= origin; never reads any data > origin.
    """

    def __init__(self, plant_info: dict, origin: pd.Timestamp, horizon: int) -> None:
        self.code: str = plant_info["code"]
        self.grupo: int = plant_info["grupo"]
        self.min_date: pd.Timestamp = plant_info["min_date"]
        self.origin: pd.Timestamp = pd.Timestamp(origin)
        self.horizon: int = horizon

        n_hist = (self.origin - self.min_date).days + 1
        if n_hist <= 0:
            raise ValueError(f"Origin {origin} precedes plant {self.code} min_date {self.min_date}")

        dates = plant_info["dates"]
        # Slice strictly <= origin
        mask_len = min(n_hist, len(dates))
        hist_kwh = np.full(n_hist, np.nan, dtype=float)
        hist_cap = np.full(n_hist, np.nan, dtype=float)
        if mask_len > 0:
            hist_kwh[:mask_len] = plant_info["kwh_valid"][:mask_len]
            hist_cap[:mask_len] = plant_info["cap_MW"][:mask_len]

        valid_cap_idx = np.flatnonzero(np.isfinite(hist_cap) & (hist_cap > 0))
        if len(valid_cap_idx) == 0:
            raise ValueError(f"No valid cap_MW <= {origin} for {self.code}")
        self.cap_at_origin: float = float(hist_cap[valid_cap_idx[-1]])

        # Preallocate buffer covering history + horizon
        self.total_len = n_hist + horizon
        self.buf = np.full(self.total_len, np.nan, dtype=float)
        self.buf[:n_hist] = hist_kwh
        self.cur_idx = n_hist  # index of origin + 1

        # Initialize EWMA states matching pandas Series.ewm(alpha=a, adjust=False, min_periods=15)
        self.ewma_state: dict[float, float] = {}
        self.ewma_count: dict[float, int] = {}
        hist_series = pd.Series(hist_kwh)
        n_valid_hist = int(np.isfinite(hist_kwh).sum())
        for alpha in EWMA_ALPHAS:
            ew_series = hist_series.ewm(alpha=alpha, adjust=False, min_periods=1).mean()
            last_ew = ew_series.iloc[-1] if len(ew_series) else np.nan
            self.ewma_state[alpha] = float(last_ew) if np.isfinite(last_ew) else np.nan
            self.ewma_count[alpha] = n_valid_hist

        # Initialize Kalman states matching _compute_kalman_series
        self.kalman_level: dict[float, float] = {}
        self.kalman_var: dict[float, float] = {}
        denom_hist = np.where(hist_cap > 0, hist_cap * ENERGY_BOUND, np.nan)
        y_unit_hist = hist_kwh / denom_hist
        for q in KALMAN_QS:
            lvl = np.nan
            var = 1.0
            for val in y_unit_hist:
                if np.isfinite(lvl):
                    var += q
                if np.isfinite(val):
                    if not np.isfinite(lvl):
                        lvl = float(val)
                        var = 0.0
                    else:
                        gain = var / (var + 1.0)
                        lvl += gain * (float(val) - lvl)
                        var *= 1.0 - gain
            self.kalman_level[q] = lvl
            self.kalman_var[q] = var

    def next_features(self, target_date: pd.Timestamp) -> dict:
        """Return feature dictionary for `target_date` using state up to `target_date - 1`."""
        idx = self.cur_idx  # corresponds to target_date; idx - 1 is prev day
        kwh_d = self.buf[idx - 1] if idx - 1 >= 0 else np.nan

        s7 = max(0, idx - 7)
        w7 = self.buf[s7:idx]
        v7 = w7[np.isfinite(w7)]
        media7 = float(v7.mean()) if len(v7) >= 4 else np.nan

        s30 = max(0, idx - 30)
        w30 = self.buf[s30:idx]
        v30 = w30[np.isfinite(w30)]
        media30 = float(v30.mean()) if len(v30) >= 15 else np.nan

        s_dev = max(0, idx - 8)
        e_dev = max(0, idx - 1)
        w_dev = self.buf[s_dev:e_dev]
        v_dev = w_dev[np.isfinite(w_dev)]
        dev_d = float(kwh_d - v_dev.mean()) if (np.isfinite(kwh_d) and len(v_dev) >= 4) else np.nan

        doy = target_date.dayofyear
        row = {
            "code": self.code,
            "grupo": self.grupo,
            "date": target_date,
            "kwh_D": float(kwh_d) if np.isfinite(kwh_d) else np.nan,
            "media7_kwh": media7,
            "media30_kwh": media30,
            "cap_MW_D": self.cap_at_origin,
            "deviation_D": dev_d,
            "month_sin": float(np.sin(2.0 * np.pi * target_date.month / 12.0)),
            "month_cos": float(np.cos(2.0 * np.pi * target_date.month / 12.0)),
            "doy_sin": float(np.sin(2.0 * np.pi * doy / 365.25)),
            "doy_cos": float(np.cos(2.0 * np.pi * doy / 365.25)),
        }
        for alpha in EWMA_ALPHAS:
            st = self.ewma_state[alpha]
            cnt = self.ewma_count[alpha]
            row[f"ewma_{alpha}"] = float(st) if (cnt >= 15 and np.isfinite(st)) else np.nan
        for q in KALMAN_QS:
            lvl = self.kalman_level[q]
            row[f"kalman_{q}"] = float(lvl * self.cap_at_origin * ENERGY_BOUND) if np.isfinite(lvl) else np.nan
        return row

    def step_update(self, predicted_kwh: float) -> None:
        """Advance recursive state by one day using only the model's prediction."""
        val = float(predicted_kwh)
        self.buf[self.cur_idx] = val
        self.cur_idx += 1

        if np.isfinite(val):
            for alpha in EWMA_ALPHAS:
                self.ewma_count[alpha] += 1
                st = self.ewma_state[alpha]
                self.ewma_state[alpha] = val if not np.isfinite(st) else alpha * val + (1.0 - alpha) * st

        y_unit = val / (self.cap_at_origin * ENERGY_BOUND) if self.cap_at_origin > 0 else np.nan
        for q in KALMAN_QS:
            lvl = self.kalman_level[q]
            var = self.kalman_var[q]
            if np.isfinite(lvl):
                var += q
            if np.isfinite(y_unit):
                if not np.isfinite(lvl):
                    lvl = float(y_unit)
                    var = 0.0
                else:
                    gain = var / (var + 1.0)
                    lvl += gain * (float(y_unit) - lvl)
                    var *= 1.0 - gain
            self.kalman_level[q] = lvl
            self.kalman_var[q] = var


def fit_model(method: str, cfg: dict, train: pd.DataFrame) -> dict:
    """Fit a procedure once on `train` (weather_on=False), matching daily._fit_predict."""
    if method == "Media móvil de 30 días":
        return {"method": method}

    if method in daily.LIBRARIES:
        model = daily._library(method, cfg, seed=daily.SEED)
        model.fit(train[daily.BASE_X], train["y"])
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
        cutoff = train["date"].max() - pd.Timedelta(days=w - 1)
        mask = train["date"].ge(cutoff).to_numpy()
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


def _harc_design(
    frame: pd.DataFrame,
    values: np.ndarray,
    groups: pd.Index,
    plants: pd.Index,
    use_group: bool,
) -> np.ndarray:
    group_idx = groups.get_indexer(frame["grupo"])
    plant_idx = plants.get_indexer(frame["code"])
    G = np.zeros((len(frame), len(groups))) if use_group else np.zeros((len(frame), 0))
    P = np.zeros((len(frame), len(plants)))
    gvalid = group_idx >= 0
    pvalid = plant_idx >= 0
    if use_group:
        G[np.arange(len(frame))[gvalid], group_idx[gvalid]] = 1.0
    P[np.arange(len(frame))[pvalid], plant_idx[pvalid]] = 1.0
    rad = values[:, 0] if values.shape[1] else np.zeros(len(frame))
    return np.column_stack([np.ones(len(frame)), values, G, G * rad[:, None], P, P * rad[:, None]])


def predict_batch(fitted: dict, query: pd.DataFrame) -> np.ndarray:
    """Predict a batch of query rows using a pre-fitted model and clip physically."""
    method = fitted["method"]
    if method == "Media móvil de 30 días":
        return daily._clip_physical(query["media30_kwh"].to_numpy(dtype=float), query)

    if method in daily.LIBRARIES:
        raw_pred = fitted["model"].predict(query[daily.BASE_X])
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

    return daily._clip_physical(np.asarray(pred, dtype=float), query)


def _plant_has_history_and_valid_future(
    plant_info: dict,
    origin: pd.Timestamp,
    horizon: int,
) -> tuple[bool, str]:
    """Check if a plant can be forecasted recursively from `origin` and has >=1 valid target."""
    if plant_info["min_date"] > origin:
        return False, "planta_no_iniciada_al_origen"
    dates = plant_info["dates"]
    n_hist = (origin - plant_info["min_date"]).days + 1
    if n_hist < MIN_PLANT_HISTORY_DAYS:
        return False, "historia_menor_15_dias"
    hist_kwh = plant_info["kwh_valid"][: min(n_hist, len(dates))]
    if np.isfinite(hist_kwh).sum() < MIN_PLANT_HISTORY_DAYS:
        return False, "observaciones_validas_menores_15"
    hist_cap = plant_info["cap_MW"][: min(n_hist, len(dates))]
    if not (np.isfinite(hist_cap) & (hist_cap > 0)).any():
        return False, "sin_capacidad_valida_al_origen"

    start_fut = n_hist
    end_fut = min(n_hist + horizon, len(dates))
    if end_fut <= start_fut:
        return False, "sin_ventana_futura"
    fut_kwh = plant_info["kwh_valid"][start_fut:end_fut]
    if np.isfinite(fut_kwh).sum() == 0:
        return False, "sin_observaciones_validas_en_ventana"
    return True, "elegible"


def _rollout_recursive_batch(
    fitted: dict,
    plant_infos: list[dict],
    origin: pd.Timestamp,
    horizon: int,
) -> np.ndarray:
    """Run recursive rollout for a list of plants from `origin` for `horizon` days.

    Returns array of shape (len(plant_infos), horizon) with predictions in kWh.
    """
    states = [PlantRecursiveState(p, origin, horizon) for p in plant_infos]
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
    gap_days: int = INNER_GAP_DAYS,
) -> list[tuple[pd.DataFrame, pd.Timestamp, list[dict]]]:
    """Build up to 3 recursive internal validation splits strictly <= origin."""
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
            ok, _ = _plant_has_history_and_valid_future(raw_plants[code], s_m, horizon)
            if ok:
                eligible_infos.append(raw_plants[code])
        if not eligible_infos:
            continue
        seen_origins.add(s_m)
        splits.append((inner_train, s_m, eligible_infos))
    return splits


def _evaluate_recursive_macro_mae(
    preds: np.ndarray,
    plant_infos: list[dict],
    origin: pd.Timestamp,
    horizon: int,
) -> float:
    """Macro MAE across plants on valid observed days in [origin + 1, origin + horizon]."""
    plant_maes = []
    for i, info in enumerate(plant_infos):
        n_hist = (origin - info["min_date"]).days + 1
        actual = info["kwh_valid"][n_hist : n_hist + horizon]
        valid = np.isfinite(actual)
        if valid.any():
            plant_maes.append(float(np.mean(np.abs(preds[i, valid] - actual[valid]))))
    return float(np.mean(plant_maes)) if plant_maes else np.inf


def choose_hyperparameters_recursive(
    method: str,
    train: pd.DataFrame,
    raw_plants: dict[str, dict],
    origin: pd.Timestamp,
    horizon: int,
    inner_splits: list[tuple[pd.DataFrame, pd.Timestamp, list[dict]]],
) -> dict:
    """Select hyperparameters for `method` via horizon-specific recursive inner CV."""
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
                preds = _rollout_recursive_batch(fitted, elig_infos, s_m, horizon)
                losses.append(_evaluate_recursive_macro_mae(preds, elig_infos, s_m, horizon))
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

        if task == "intraplanta":
            code = str(unit)
            group_id = raw_plants[code]["grupo"]
            unit_name = code
            train = panel[(panel["code"] == code) & (panel["date"] <= origin)]
            ok_plant, reason = _plant_has_history_and_valid_future(raw_plants[code], origin, horizon)
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
                ok_plant, reason = _plant_has_history_and_valid_future(raw_plants[code], origin, horizon)
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
            inner_splits = build_inner_origins(train, raw_plants, origin, horizon)
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
                anchor_jobs.append((task, unit, origin, horizon, panel, raw_plants))
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
            "train_key_sha256": _key_signature(train), "inner_key_sha256": inner_sig,
        })

    return cov_inelig, eval_specs, anchor_jobs


def _run_anchor_recalibration_task(args: tuple) -> tuple[tuple, dict[str, dict], dict[str, float]]:
    """Stage 1 worker: select horizon-specific hyperparameters for all 22 methods at one anchor."""
    task, unit, anchor_origin, horizon, panel, raw_plants = args
    if task == "intraplanta":
        train = panel[(panel["code"] == str(unit)) & (panel["date"] <= anchor_origin)].copy()
    else:
        train = panel[(panel["grupo"] != int(unit)) & (panel["date"] <= anchor_origin)].copy()
    inner_splits = build_inner_origins(train, raw_plants, anchor_origin, horizon)
    cfgs: dict[str, dict] = {}
    sel_secs: dict[str, float] = {}
    for method in daily.METHODS:
        t0 = time.perf_counter()
        cfgs[method] = choose_hyperparameters_recursive(
            method, train, raw_plants, anchor_origin, horizon, inner_splits
        )
        sel_secs[method] = time.perf_counter() - t0
    return (task, unit, anchor_origin), cfgs, sel_secs


def _run_unit_refit_batch_task(args: tuple) -> dict:
    """Stage 2 worker: refit all 22 methods and rollout recursively for a list of OOF origins."""
    specs, cfg_lookup, sel_lookup, horizon, panel, raw_plants = args
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
            preds = _rollout_recursive_batch(fitted, pred_infos, origin, horizon)
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
                cap_at_origin = PlantRecursiveState(info, origin, horizon).cap_at_origin
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


def audit_multihorizon_leakage(
    base_path: Path,
    panel: pd.DataFrame,
    raw_plants: dict[str, dict],
    pred: pd.DataFrame,
    coverage: pd.DataFrame,
    choices: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Run 5 automated leakage & equivalence controls on the multi-horizon pipeline."""
    records = []

    # 1. Equivalence fit_model/predict_batch vs daily._fit_predict across all 22 methods
    sample_origin = pd.Timestamp("2025-10-31")
    train_eq = panel[(panel["grupo"] != 1) & (panel["date"] <= sample_origin)].copy()
    test_eq = panel[(panel["grupo"] == 1) & (panel["date"] > sample_origin)].head(10).copy()
    max_diff_methods = 0.0
    for method in daily.METHODS:
        cfg = daily.GRIDS[method][0]
        p_daily = daily._fit_predict(method, cfg, train_eq, test_eq, weather_on=False)
        p_fast = predict_batch(fit_model(method, cfg, train_eq), test_eq)
        diff = float(np.max(np.abs(p_daily - p_fast)))
        max_diff_methods = max(max_diff_methods, diff)
    assert max_diff_methods < 1e-8, f"Equivalence failed: max_diff={max_diff_methods}"
    records.append({
        "control": "equivalencia_fit_predict_22_metodos",
        "detalle": f"max_diff={max_diff_methods:.2e}",
        "estado": "superado",
    })

    # 2. Equivalence O(1) PlantRecursiveState vs full panel features at origin + 1
    max_diff_state = 0.0
    for code in sorted(raw_plants)[:5]:
        info = raw_plants[code]
        if info["min_date"] > sample_origin - pd.Timedelta(days=60):
            continue
        st = PlantRecursiveState(info, sample_origin, horizon)
        feat = st.next_features(sample_origin + pd.Timedelta(days=1))
        ref = panel[(panel["code"] == code) & (panel["date"] == sample_origin + pd.Timedelta(days=1))]
        if not ref.empty:
            ref_row = ref.iloc[0]
            for col in FEATURE_COLS:
                v1, v2 = float(feat[col]), float(ref_row[col])
                if np.isfinite(v1) and np.isfinite(v2):
                    max_diff_state = max(max_diff_state, abs(v1 - v2))
    assert max_diff_state < 1e-9, f"State equivalence failed: max_diff={max_diff_state}"
    records.append({
        "control": "equivalencia_estado_recursivo_O1",
        "detalle": f"max_diff={max_diff_state:.2e}",
        "estado": "superado",
    })

    # 3. Future perturbation test (date > origin) for kwh (+50%) and cap_MW (+50%)
    base_df = pd.read_csv(
        base_path,
        usecols=["code", "date", "kwh", "horas_xm", "cap_MW", "lat", "lon", "grupo"],
        parse_dates=["date"],
    )
    altered_df = base_df.copy()
    fut_mask = altered_df["date"] > sample_origin
    altered_df.loc[fut_mask, "kwh"] = altered_df.loc[fut_mask, "kwh"] * 1.5 + 100.0
    altered_df.loc[fut_mask, "cap_MW"] = altered_df.loc[fut_mask, "cap_MW"] * 1.5
    panel_alt, plants_alt = load_generation_panel(base_path, base_data=altered_df)
    train_alt = panel_alt[(panel_alt["grupo"] != 1) & (panel_alt["date"] <= sample_origin)].copy()
    elig_orig = [raw_plants[c] for c in sorted(raw_plants) if raw_plants[c]["grupo"] == 1]
    elig_alt = [plants_alt[c] for c in sorted(plants_alt) if plants_alt[c]["grupo"] == 1]
    for method in ("RLAC", "KLAC", "XGBoost", "RQRF"):
        cfg = daily.GRIDS[method][0]
        r1 = _rollout_recursive_batch(fit_model(method, cfg, train_eq), elig_orig, sample_origin, horizon)
        r2 = _rollout_recursive_batch(fit_model(method, cfg, train_alt), elig_alt, sample_origin, horizon)
        np.testing.assert_allclose(r1, r2, atol=0.0, rtol=0.0)
    records.append({
        "control": "perturbacion_futuro_kwh_y_capacidad",
        "detalle": "atol=0.0 en RLAC, KLAC, XGBoost, RQRF",
        "estado": "superado",
    })

    # 4. Prefix truncation test (only rows <= origin)
    prefix_df = base_df[base_df["date"] <= sample_origin].copy()
    panel_pref, plants_pref = load_generation_panel(base_path, base_data=prefix_df)
    train_pref = panel_pref[(panel_pref["grupo"] != 1) & (panel_pref["date"] <= sample_origin)].copy()
    elig_pref = [plants_pref[c] for c in sorted(plants_pref) if plants_pref[c]["grupo"] == 1]
    for method in ("RLAC", "KLAC", "Ridge"):
        cfg = daily.GRIDS[method][0]
        r1 = _rollout_recursive_batch(fit_model(method, cfg, train_eq), elig_orig, sample_origin, horizon)
        r3 = _rollout_recursive_batch(fit_model(method, cfg, train_pref), elig_pref, sample_origin, horizon)
        np.testing.assert_allclose(r1, r3, atol=0.0, rtol=0.0)
    records.append({
        "control": "reconstruccion_prefijo_temporal",
        "detalle": "atol=0.0 en RLAC, KLAC, Ridge",
        "estado": "superado",
    })

    # 5. OOF integrity: no duplicate (metodo, tarea, code, fecha_objetivo), train_end <= origin < test_start,
    #    and causal hyperparameter anchor <= origin
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


def _aggregate_metrics(pred: pd.DataFrame, horizon: int) -> dict[str, pd.DataFrame]:
    """Compute daily OOF metrics, by-lead, by-plant, by-group, and total-energy metrics."""
    valid = pred[pred["y_kwh"].notna()].copy()
    valid["sq_error"] = (valid["prediccion_kwh"] - valid["y_kwh"]) ** 2

    by_plant = valid.groupby(["tarea", "metodo", "code", "grupo"], as_index=False).agg(
        MAE_kWh=("error_abs_kwh", "mean"),
        MAE_capacidad_relativa=("error_capacidad_relativa", "mean"),
        RMSE2=("sq_error", "mean"),
        dias_oof=("y_kwh", "size"),
    )
    by_plant["RMSE_kWh"] = np.sqrt(by_plant.pop("RMSE2"))

    by_group = by_plant.groupby(["tarea", "metodo", "grupo"], as_index=False).agg(
        MAE_macro_planta_kWh=("MAE_kWh", "mean"),
        MAE_macro_planta_capacidad_relativa=("MAE_capacidad_relativa", "mean"),
        RMSE_macro_planta_kWh=("RMSE_kWh", "mean"),
        plantas=("code", "nunique"),
        dias_oof=("dias_oof", "sum"),
    )

    global_daily = by_plant.groupby(["tarea", "metodo"], as_index=False).agg(
        MAE_macro_kWh=("MAE_kWh", "mean"),
        MAE_macro_capacidad_relativa=("MAE_capacidad_relativa", "mean"),
        RMSE_macro_kWh=("RMSE_kWh", "mean"),
        plantas=("code", "nunique"),
        dias_oof=("dias_oof", "sum"),
    )
    group_macro = by_group.groupby(["tarea", "metodo"], as_index=False).agg(
        MAE_macro_grupo_kWh=("MAE_macro_planta_kWh", "mean"),
        MAE_macro_grupo_capacidad_relativa=("MAE_macro_planta_capacidad_relativa", "mean"),
        RMSE_macro_grupo_kWh=("RMSE_macro_planta_kWh", "mean"),
        grupos=("grupo", "nunique"),
    )
    global_daily = global_daily.merge(group_macro, on=["tarea", "metodo"], validate="one_to_one")
    base_daily = global_daily[global_daily["metodo"] == "Media móvil de 30 días"][
        ["tarea", "MAE_macro_kWh", "RMSE_macro_kWh"]
    ].rename(columns={"MAE_macro_kWh": "_base_mae", "RMSE_macro_kWh": "_base_rmse"})
    global_daily = global_daily.merge(base_daily, on="tarea", how="left")
    global_daily["Skill_MAE"] = 1.0 - (global_daily["MAE_macro_kWh"] / global_daily["_base_mae"])
    global_daily["Skill_RMSE"] = 1.0 - (global_daily["RMSE_macro_kWh"] / global_daily["_base_rmse"])
    global_daily = global_daily.drop(columns=["_base_mae", "_base_rmse"])
    global_daily = global_daily.sort_values(["tarea", "MAE_macro_kWh"]).reset_index(drop=True)

    lead_plant = valid.groupby(["tarea", "metodo", "lead_dias", "semana_lead", "code"], as_index=False).agg(
        MAE_kWh=("error_abs_kwh", "mean"),
        MAE_capacidad_relativa=("error_capacidad_relativa", "mean"),
        RMSE2=("sq_error", "mean"),
    )
    lead_plant["RMSE_kWh"] = np.sqrt(lead_plant.pop("RMSE2"))
    by_lead = lead_plant.groupby(["tarea", "metodo", "lead_dias", "semana_lead"], as_index=False).agg(
        MAE_macro_kWh=("MAE_kWh", "mean"),
        MAE_macro_capacidad_relativa=("MAE_capacidad_relativa", "mean"),
        RMSE_macro_kWh=("RMSE_kWh", "mean"),
        plantas=("code", "nunique"),
    )

    # Cumulative energy error over complete observed trajectories only (observados == horizon)
    traj = pred.groupby(["tarea", "metodo", "code", "grupo", "origen", "cap_MW_origen"], as_index=False).agg(
        prediccion_total_kwh=("prediccion_kwh", "sum"),
        y_total_kwh=("y_kwh", "sum"),
        dias_observados=("y_kwh", "count"),
    )
    traj["completa"] = traj["dias_observados"] == horizon
    traj_complete = traj[traj["completa"]].copy()
    traj_complete["error_total_abs_kwh"] = (traj_complete["prediccion_total_kwh"] - traj_complete["y_total_kwh"]).abs()
    traj_complete["sq_total"] = (traj_complete["prediccion_total_kwh"] - traj_complete["y_total_kwh"]) ** 2
    traj_complete["denom_total"] = traj_complete["cap_MW_origen"] * ENERGY_BOUND * horizon
    traj_complete["error_total_rel_cap"] = traj_complete["error_total_abs_kwh"] / traj_complete["denom_total"]

    incomp_counts = traj[~traj["completa"]].groupby(["tarea", "metodo"], as_index=False).agg(
        trayectorias_incompletas_excluidas=("origen", "size")
    )
    total_plant = traj_complete.groupby(["tarea", "metodo", "code", "grupo"], as_index=False).agg(
        MAE_total_kWh=("error_total_abs_kwh", "mean"),
        MAE_total_rel_cap=("error_total_rel_cap", "mean"),
        RMSE2_total=("sq_total", "mean"),
        trayectorias=("origen", "size"),
    )
    total_plant["RMSE_total_kWh"] = np.sqrt(total_plant.pop("RMSE2_total"))
    energy_total = total_plant.groupby(["tarea", "metodo"], as_index=False).agg(
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
        "global_daily": global_daily,
        "by_plant": by_plant,
        "by_group": by_group,
        "by_lead": by_lead,
        "energy_total": energy_total,
    }


def run_horizon_benchmark(
    base_path: Path,
    output: Path,
    horizon: int,
    max_workers: int | None = None,
    origins_override: list[pd.Timestamp] | None = None,
) -> dict[str, pd.DataFrame | dict]:
    """Execute full OOF multi-horizon benchmark for a given `horizon` (7, 14, 21, or 28)."""
    t_start = time.perf_counter()
    panel, raw_plants = load_generation_panel(base_path)
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
        c_inelig, e_specs, a_jobs = _plan_unit_origins("intraplanta", code, origins, horizon, panel, raw_plants)
        all_cov.extend(c_inelig)
        all_eval_specs.extend(e_specs)
        anchor_jobs.extend(a_jobs)
    for group_id in groups_list:
        c_inelig, e_specs, a_jobs = _plan_unit_origins("espaciotemporal", group_id, origins, horizon, panel, raw_plants)
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

    # Stage 2: parallel OOF refit + recursive rollout across balanced chunks
    n_chunks = max(1, min(len(all_eval_specs), workers * 3))
    spec_chunks = [all_eval_specs[i::n_chunks] for i in range(n_chunks) if all_eval_specs[i::n_chunks]]
    if workers <= 1:
        stage2_results = [
            _run_unit_refit_batch_task((chunk, cfg_lookup, sel_lookup, horizon, panel, raw_plants))
            for chunk in spec_chunks
        ]
    else:
        stage2_results = Parallel(n_jobs=workers, backend="loky")(
            delayed(_run_unit_refit_batch_task)((chunk, cfg_lookup, sel_lookup, horizon, panel, raw_plants))
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

    audit = audit_multihorizon_leakage(base_path, panel, raw_plants, pred, coverage, choices, horizon)
    agg = _aggregate_metrics(pred, horizon)

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
        "target": "generación eléctrica diaria observada (kWh/planta-día), pronóstico recursivo OOF desde origen fijo",
        "horizon_days": horizon,
        "weather_enabled": False,
        "weather_note": "Sin variables ni pronósticos meteorológicos; el archivo disponible solo cubre D+1.",
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
            "source_sha256": _file_sha256(source_path),
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

"""Benchmark de producción diaria con validación temporal independiente por planta.

El experimento compara las once adaptaciones personalizadas documentadas en el
proyecto, RLAC, nueve referencias de librería y una media móvil. Cada planta se
ajusta por separado. La selección de hiperparámetros usa tres cortes temporales
internos con gap; los tres bloques externos permanecen fuera de esa selección.
"""
from __future__ import annotations

import json
import hashlib
import os
import platform
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
import sklearn
import scipy
import xgboost
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, HuberRegressor, Lasso, QuantileRegressor, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[2]
METEO = ["rad_pron", "temp_pron", "nube_pron", "humedad_pron", "viento_pron"]
ANOM = ["anom_rad", "anom_temp", "anom_nube", "anom_humedad", "anom_viento"]
BLOCKS = {
    "D1": ("2025-11-01", "2026-01-31"),
    "D2": ("2026-02-01", "2026-04-30"),
    "D3": ("2026-05-01", "2026-07-31"),
}
BASE_X = ["kwh_D", "media7_kwh", "media30_kwh", "cap_MW_D", "month_sin", "month_cos", "doy_sin", "doy_cos"]
LIBRARIES = ["XGBoost", "Random Forest", "Ridge", "SVR", "HistGradientBoosting", "k-NN", "Lasso", "BayesianRidge", "DecisionTree"]
CUSTOM = ["RLAC", "IWLAR", "WCFC", "ABGPR", "HARC", "SRMOS", "ELAC", "KLAC", "RARC", "RQRF", "PDEC", "MARB"]
MODEL_LABELS = {
    "RLAC": "Corrección Ridge del residuo sobre nivel móvil",
    "IWLAR": "Regresión local ponderada de residuos",
    "WCFC": "Mezcla temporal de correcciones Ridge–Huber",
    "ABGPR": "Proceso gaussiano sobre residuos",
    "HARC": "Efectos penalizados de grupo y sitio",
    "SRMOS": "MOS con ventana temporal de entrenamiento",
    "ELAC": "Corrección residual sobre nivel exponencial",
    "KLAC": "Corrección residual sobre nivel local de Kalman",
    "RARC": "Corrección residual robusta Huber",
    "RQRF": "Bosque cuantílico de residuos",
    "PDEC": "Corrección por desviación del nivel previo",
    "MARB": "Boosting residual monótono en anomalía de radiación",
}
DISPLAY_LABELS = {
    **MODEL_LABELS,
    "XGBoost":"XGBoost",
    "Random Forest":"Bosque aleatorio",
    "Ridge":"Regresión Ridge",
    "SVR":"Regresión con vectores de soporte",
    "HistGradientBoosting":"Boosting de gradiente histográmico",
    "k-NN":"k vecinos más cercanos",
    "Lasso":"Regresión Lasso",
    "BayesianRidge":"Regresión Ridge bayesiana",
    "DecisionTree":"Árbol de decisión",
    "Media móvil de 30 días":"Media móvil de 30 días",
}
METHODS = CUSTOM + LIBRARIES + ["Media móvil de 30 días"]
SEED = 2026
MAX_CONFIGURATIONS = 4
INNER_VALIDATION_DAYS = 14
INNER_GAP_DAYS = 30
DEVELOPMENT_BLOCKS = BLOCKS
FINAL_BLOCK = {"D4": ("2026-08-01", "2026-10-31")}


def _weather(pron: pd.DataFrame) -> pd.DataFrame:
    out = []
    for code, group in pron.groupby("code", sort=False):
        g = group.sort_values("date").set_index("date").asfreq("D")
        f = pd.DataFrame(index=g.index)
        f["code"] = code
        for raw, anomaly in zip(METEO, ANOM):
            f[raw] = g[raw]
            # The reference is strictly historical: today's forecast is not part of its own baseline.
            f[anomaly] = g[raw] - g[raw].rolling(30, min_periods=15).mean().shift(1)
        out.append(f.reset_index(names="date"))
    return pd.concat(out, ignore_index=True)


def build_panel(base_path: Path, forecast_path: Path, weather_on: bool,
                base_data: pd.DataFrame | None = None, pron_data: pd.DataFrame | None = None):
    if base_data is None:
        base = pd.read_csv(base_path, usecols=["code", "date", "kwh", "horas_xm", "cap_MW", "lat", "lon", "grupo"], parse_dates=["date"])
    else:
        base = base_data.copy()
        base["date"] = pd.to_datetime(base.date)
    if pron_data is None:
        pron = pd.read_csv(forecast_path, usecols=["code", "date"] + METEO, parse_dates=["date"])
    else:
        pron = pron_data.copy()
        pron["date"] = pd.to_datetime(pron.date)
    base["code"] = base.code.astype(str); pron["code"] = pron.code.astype(str)
    end = pd.Timestamp(BLOCKS["D3"][1])
    base = base[base.date <= end].copy(); pron = pron[pron.date <= end].copy()
    wx = _weather(pron)
    common = wx.dropna(subset=ANOM)[["code", "date"]]
    rows = []
    for code, src in base.groupby("code", sort=False):
        src = src.sort_values("date").set_index("date")
        g = src.asfreq("D")
        rec = pd.DataFrame(index=g.index)
        rec["code"] = code; rec["grupo"] = src.grupo.iloc[0]
        rec["lat"] = src.lat.iloc[0]; rec["lon"] = src.lon.iloc[0]
        rec["y"] = g.kwh; rec["horas_xm"] = g.horas_xm; rec["cap_MW_eval"] = g.cap_MW
        history = g.kwh.where(g.horas_xm > 6)
        rec["kwh_D"] = history.shift(1)
        rec["media7_kwh"] = history.rolling(7, min_periods=4).mean().shift(1)
        rec["media30_kwh"] = history.rolling(30, min_periods=15).mean().shift(1)
        rec["cap_MW_D"] = g.cap_MW.shift(1)
        # Yesterday's deviation from its own trailing level; this is not a forecast error.
        rec["deviation_D"] = rec.kwh_D - history.rolling(7, min_periods=4).mean().shift(2)
        # Forward-only exponential and local-level states, shifted so day T uses data through T-1.
        for alpha in (0.03, 0.07, 0.15, 0.3):
            rec[f"ewma_{alpha}"] = history.ewm(alpha=alpha, adjust=False, min_periods=15).mean().shift(1)
        for q in (0.0001, 0.001, 0.01, 0.1):
            # Work in per-unit daily energy. q is the process/observation variance ratio;
            # the fixed observation variance is therefore dimensionless, not 1 kWh^2.
            denom = (g.cap_MW * 24_000).where(g.cap_MW > 0)
            y_unit = history / denom
            h_obs = 1.0
            q_proc = q * h_obs
            level, var, vals = np.nan, h_obs, []
            for val in y_unit.to_numpy(float):
                if np.isfinite(level):
                    var += q_proc
                if np.isfinite(val):
                    if not np.isfinite(level):
                        level, var = val, 0.0
                    else:
                        gain = var / (var + h_obs)
                        level += gain * (val - level)
                        var *= 1 - gain
                vals.append(level)
            level_unit_prior = pd.Series(vals, index=g.index).shift(1)
            rec[f"kalman_{q}"] = level_unit_prior * rec.cap_MW_D * 24_000
        rows.append(rec.reset_index(names="date"))
    panel = pd.concat(rows, ignore_index=True).merge(common, on=["code", "date"], how="inner", validate="one_to_one")
    if weather_on:
        panel = panel.merge(wx, on=["code", "date"], how="left", validate="one_to_one").dropna(subset=ANOM)
    panel = panel[(panel.horas_xm > 6) & panel.media30_kwh.notna() & panel.y.notna()].copy()
    panel["month_sin"] = np.sin(2*np.pi*panel.date.dt.month/12); panel["month_cos"] = np.cos(2*np.pi*panel.date.dt.month/12)
    doy=panel.date.dt.dayofyear; panel["doy_sin"] = np.sin(2*np.pi*doy/365.25); panel["doy_cos"] = np.cos(2*np.pi*doy/365.25)
    if not weather_on:
        # Forecast completeness defines the common origin set, but weather values never enter a no-weather estimator.
        panel = panel.drop(columns=METEO + ANOM, errors="ignore")
    return panel.sort_values(["code", "date"]).reset_index(drop=True)


def leakage_audit(base_path: Path, forecast_path: Path, panel: pd.DataFrame,
                  weather_on: bool) -> pd.DataFrame:
    """Rebuild prefixes and perturb same-day labels to test temporal feature isolation."""
    cols = [c for c in BASE_X + ["deviation_D"] + [f"ewma_{alpha}" for alpha in (0.03, 0.07, 0.15, 0.3)] + [f"kalman_{q}" for q in (0.0001, 0.001, 0.01, 0.1)] + (METEO + ANOM if weather_on else []) if c in panel]
    keys = ["code", "date"]
    base = pd.read_csv(base_path, usecols=["code", "date", "kwh", "horas_xm", "cap_MW", "lat", "lon", "grupo"], parse_dates=["date"])
    pron = pd.read_csv(forecast_path, usecols=["code", "date"] + METEO, parse_dates=["date"])
    base["code"] = base.code.astype(str); pron["code"] = pron.code.astype(str)
    records = []
    for block, (first, _) in BLOCKS.items():
        cutoff = pd.Timestamp(first) - pd.Timedelta(days=1)
        rebuilt = build_panel(base_path, forecast_path, weather_on,
                              base_data=base[base.date <= cutoff], pron_data=pron[pron.date <= cutoff])
        expected = panel[panel.date <= cutoff].set_index(keys)[cols].sort_index()
        actual = rebuilt.set_index(keys)[cols].sort_index()
        pd.testing.assert_frame_equal(expected, actual, check_exact=True)
        records.append({"control": "reconstruccion de prefijo temporal", "fold": block, "filas": len(expected), "estado": "superado"})

    sample = pd.concat([panel[panel.date.between(a, z)].groupby("code", sort=False).nth(0).head(2)
                        for a, z in BLOCKS.values()], ignore_index=False).drop_duplicates(["code", "date"])
    for row in sample.itertuples(index=False):
        altered = base.copy()
        mask = altered.code.eq(row.code) & altered.date.eq(row.date)
        if not mask.any():
            continue
        altered.loc[mask, "kwh"] = altered.loc[mask, "kwh"] + max(1.0, abs(float(row.y)) * .25)
        rebuilt = build_panel(base_path, forecast_path, weather_on, base_data=altered, pron_data=pron)
        before = panel.loc[(panel.code == row.code) & (panel.date == row.date), cols].reset_index(drop=True)
        after = rebuilt.loc[(rebuilt.code == row.code) & (rebuilt.date == row.date), cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(before, after, check_exact=True)
    records.append({"control": "perturbacion del objetivo del mismo dia", "fold": "muestra en D1-D3", "filas": len(sample), "estado": "superado"})
    return pd.DataFrame(records)


def _xcols(weather_on):
    return BASE_X + (METEO + ANOM if weather_on else [])


def _pipeline(est, scale=True):
    steps = [SimpleImputer(strategy="median")]
    if scale: steps.append(StandardScaler())
    steps.append(est)
    return make_pipeline(*steps)


def _library(name, cfg, seed=2026):
    pars = dict(cfg)
    if name == "XGBoost": est=_pipeline(XGBRegressor(n_estimators=180, max_depth=pars.get("depth",3), learning_rate=pars.get("learning_rate",.05), min_child_weight=5, subsample=.8, colsample_bytree=1, tree_method="hist", device="cpu", n_jobs=1, objective="reg:squarederror", random_state=seed, verbosity=0))
    elif name == "Random Forest": est=_pipeline(RandomForestRegressor(n_estimators=140, min_samples_leaf=pars.get("leaf",10), max_features=pars.get("max_features",.8), n_jobs=1, random_state=seed), False)
    elif name == "Ridge": est=_pipeline(Ridge(alpha=pars.get("alpha",1)))
    elif name == "SVR": est=_pipeline(SVR(C=pars.get("C",1), epsilon=pars.get("epsilon",.1)))
    elif name == "HistGradientBoosting": est=_pipeline(HistGradientBoostingRegressor(max_iter=140, learning_rate=pars.get("learning_rate",.05), max_leaf_nodes=pars.get("leaves",15), l2_regularization=1, random_state=seed), False)
    elif name == "k-NN": est=_pipeline(KNeighborsRegressor(n_neighbors=pars.get("k",10), weights=pars.get("weights","distance"), n_jobs=1))
    elif name == "Lasso": est=_pipeline(Lasso(alpha=pars.get("alpha",.01), max_iter=20000, tol=1e-4))
    elif name == "BayesianRidge": est=_pipeline(BayesianRidge(alpha_1=pars.get("alpha_1",1e-6),alpha_2=pars.get("alpha_2",1e-6),lambda_1=pars.get("lambda_1",1e-6),lambda_2=pars.get("lambda_2",1e-6)))
    elif name == "DecisionTree": est=_pipeline(DecisionTreeRegressor(max_depth=pars.get("depth",8), min_samples_leaf=pars.get("leaf",10), random_state=seed), False)
    else: raise KeyError(name)
    # This scaler is fit inside each call to fit: only on that training split.
    return TransformedTargetRegressor(regressor=est, transformer=StandardScaler())


GRIDS = {
    "RLAC": [{"alpha": a} for a in (.01,.1,1.,10.)],
    "IWLAR": [{"h": h} for h in (.25,.5,1.,2.)],
    "WCFC": [{"weight_huber": w} for w in (0.,1/3,2/3,1.)],
    "ABGPR": [{"length": l} for l in (.25,.5,1.,2.)],
    "HARC": [{"lam_group": g,"lam_plant": p} for g,p in ((.1,.1),(.1,10.),(10.,.1),(10.,10.))],
    "SRMOS": [{"window": d} for d in (30,60,90,180)],
    "ELAC": [{"alpha": a} for a in (.03,.07,.15,.3)],
    "KLAC": [{"q": q} for q in (.0001,.001,.01,.1)],
    "RARC": [{"epsilon": e} for e in (1.1,1.35,1.75,2.5)],
    "RQRF": [{"leaf": n} for n in (3,5,10,20)],
    "PDEC": [{"alpha": a} for a in (.01,.1,1.,10.)],
    "MARB": [{"depth": d} for d in (1,2,3,4)],
    "XGBoost": [{"depth": d,"learning_rate": lr} for d,lr in ((2,.03),(2,.08),(4,.03),(4,.08))],
    "Random Forest": [{"leaf": leaf,"max_features": feat} for leaf,feat in ((5,.5),(5,1.),(20,.5),(20,1.))],
    "Ridge": [{"alpha": a} for a in (.01,.1,1.,10.)],
    "SVR": [{"C": c,"epsilon": e} for c,e in ((.1,.05),(.1,.2),(10.,.05),(10.,.2))],
    "HistGradientBoosting": [{"leaves": leaves,"learning_rate": lr} for leaves,lr in ((7,.03),(7,.1),(15,.03),(15,.1))],
    "k-NN": [{"k": k,"weights": w} for k,w in ((5,"uniform"),(5,"distance"),(15,"uniform"),(15,"distance"))],
    "Lasso": [{"alpha": a} for a in (.0001,.001,.01,.1)],
    "BayesianRidge": [{"alpha_1": a,"lambda_1": l,"alpha_2": 1e-6,"lambda_2": 1e-6} for a,l in ((1e-8,1e-8),(1e-8,1e-4),(1e-4,1e-8),(1e-4,1e-4))],
    "DecisionTree": [{"depth": d,"leaf": leaf} for d,leaf in ((4,5),(4,20),(8,5),(8,20))],
    "Media móvil de 30 días": [{}],
}


def _recent_stratified_indices(frame: pd.DataFrame, max_rows: int = 300) -> np.ndarray:
    """Choose a reproducible, recent sample with equal representation by plant."""
    indexed = frame[["code", "date"]].copy().reset_index(drop=True)
    indexed["_pos"] = np.arange(len(indexed))
    codes = sorted(indexed.code.astype(str).unique())
    if not codes:
        return np.array([], dtype=int)
    per_plant = max(1, max_rows // len(codes))
    picked = []
    for code in codes:
        recent = indexed[indexed.code.astype(str).eq(code)].sort_values("date").tail(per_plant)
        picked.extend(recent._pos.tolist())
    return indexed.iloc[picked].sort_values(["date", "code"])["_pos"].to_numpy(dtype=int)


def _macro_mae(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    scored = pd.DataFrame({"code": frame.code.to_numpy(),
                           "abs_error": np.abs(np.asarray(prediction, float) - frame.y.to_numpy(float))})
    return float(scored.groupby("code", sort=False).abs_error.mean().mean())


def _clip_physical(prediction: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    """Apply the same nonnegative and nameplate-energy bounds to every procedure."""
    values=np.asarray(prediction,float)
    upper=frame.cap_MW_D.to_numpy(float)*24_000
    upper=np.where(np.isfinite(upper)&(upper>=0),upper,np.inf)
    return np.minimum(np.maximum(values,0),upper)


def _fit_predict(name, cfg, train, test, weather_on):
    if name == "Media móvil de 30 días": return _clip_physical(test.media30_kwh.to_numpy(float),test)
    if name in LIBRARIES:
        model = _library(name, cfg); model.fit(train[_xcols(weather_on)], train.y)
        return _clip_physical(model.predict(test[_xcols(weather_on)]),test)
    wx = ANOM if weather_on else ["kwh_D", "media7_kwh", "month_sin", "month_cos", "doy_sin", "doy_cos"]
    X = train[wx].to_numpy(float); Xt = test[wx].to_numpy(float)
    mu = np.nanmedian(X, axis=0); mu = np.where(np.isfinite(mu),mu,0)
    X = np.where(np.isfinite(X), X, mu); Xt = np.where(np.isfinite(Xt), Xt, mu)
    scaler=StandardScaler().fit(X); X=scaler.transform(X); Xt=scaler.transform(Xt)
    r=train.y.to_numpy(float)-train.media30_kwh.to_numpy(float)
    # Residual scaling is learned from this model's training rows only.
    r_center=float(np.mean(r)); r_scale=float(np.std(r)); r_scale=r_scale if np.isfinite(r_scale) and r_scale>1e-10 else 1.0
    r_std=(r-r_center)/r_scale
    base=test.media30_kwh.to_numpy(float)
    if name == "RLAC":
        m=Ridge(alpha=cfg["alpha"]).fit(X,r_std); pred=base+r_center+r_scale*m.predict(Xt)
    elif name == "IWLAR":
        pred=np.empty(len(test)); h=cfg["h"]
        train_code=train.code.to_numpy(); train_group=train.grupo.to_numpy()
        test_code=test.code.to_numpy(); test_group=test.grupo.to_numpy()
        # Multiplicative climate-locality and plant/group identity weights.
        for i in range(len(test)):
            kernel=np.exp(-np.minimum(cdist(Xt[i:i+1],X,"sqeuclidean")[0]/h**2,700))
            identity=np.where(test_code[i]==train_code,1.,np.where(test_group[i]==train_group,.5,.2))
            w=kernel*identity+1e-8
            A=np.column_stack([np.ones(len(X)),X]); Aq=np.r_[1.,Xt[i]]
            G=A.T@(w[:,None]*A)+1e-3*w.sum()*np.diag([0]+[1]*X.shape[1])
            b=A.T@(w*r_std); pred[i]=base[i]+r_center+r_scale*(Aq@np.linalg.solve(G,b))
    elif name == "PDEC":
        x=np.column_stack([X, train.deviation_D.fillna(0).to_numpy()]); xt=np.column_stack([Xt,test.deviation_D.fillna(0).to_numpy()])
        mm=Ridge(alpha=cfg["alpha"]).fit(x,r_std); pred=base+r_center+r_scale*mm.predict(xt)
    elif name == "ELAC":
        lv=train[f"ewma_{cfg['alpha']}"] .to_numpy(float); lvt=test[f"ewma_{cfg['alpha']}"] .to_numpy(float)
        mm=Ridge(alpha=1).fit(X,(train.y.to_numpy()-lv-r_center)/r_scale); pred=lvt+r_center+r_scale*mm.predict(Xt)
    elif name == "KLAC":
        state=f"kalman_{cfg['q']}"
        # If the filter has no initialized state, use the same causal 30-day level
        # supplied to all residual procedures; never drop a scored row.
        lv=train[state].fillna(train.media30_kwh).to_numpy(float)
        lvt=test[state].fillna(test.media30_kwh).to_numpy(float)
        mm=Ridge(alpha=1).fit(X,(train.y.to_numpy()-lv-r_center)/r_scale); pred=lvt+r_center+r_scale*mm.predict(Xt)
    elif name == "RARC":
        m=HuberRegressor(epsilon=cfg["epsilon"],max_iter=300).fit(X,r_std); pred=base+r_center+r_scale*m.predict(Xt)
    elif name == "MARB":
        constraints=(1,)+(0,)*(X.shape[1]-1) if weather_on else tuple(0 for _ in range(X.shape[1]))
        m=XGBRegressor(n_estimators=120,max_depth=cfg["depth"],learning_rate=.04,min_child_weight=5,subsample=.8,colsample_bytree=1,tree_method="hist",device="cpu",n_jobs=1,objective="reg:squarederror",random_state=2026,verbosity=0,monotone_constraints=constraints).fit(X,r_std); pred=base+r_center+r_scale*m.predict(Xt)
    elif name == "RQRF":
        m=RandomForestRegressor(n_estimators=120,min_samples_leaf=cfg["leaf"],max_features=1.,n_jobs=1,random_state=2026).fit(X,r_std)
        # Quantile regression forest: average within-leaf weights across trees,
        # then take the weighted median of training residuals for each query.
        train_leaves=[tree.apply(X) for tree in m.estimators_]
        test_leaves=[tree.apply(Xt) for tree in m.estimators_]
        leaf_values=[]
        for leaf_ids in train_leaves:
            leaf_values.append({leaf: np.flatnonzero(leaf_ids == leaf) for leaf in np.unique(leaf_ids)})
        qpred=np.empty(len(Xt),float)
        for i in range(len(Xt)):
            values=[]; weights=[]
            for tree_i in range(len(m.estimators_)):
                ids=leaf_values[tree_i][test_leaves[tree_i][i]]
                if len(ids):
                    values.extend(r_std[ids].tolist())
                    weights.extend([1.0/(len(m.estimators_)*len(ids))]*len(ids))
            order=np.argsort(values); v=np.asarray(values)[order]; w=np.asarray(weights)[order]
            qpred[i]=v[min(np.searchsorted(np.cumsum(w),.5,side="left"),len(v)-1)]
        pred=base+r_center+r_scale*qpred
    elif name == "ABGPR":
        # Cap the GP solve using a recent sample balanced across training plants.
        ix=_recent_stratified_indices(train); ker=ConstantKernel(1.)*RBF(cfg["length"])+WhiteKernel(.2)
        m=GaussianProcessRegressor(kernel=ker,alpha=1e-4,normalize_y=True,optimizer=None).fit(X[ix],r_std[ix]); pred=base+r_center+r_scale*m.predict(Xt)
    elif name == "SRMOS":
        pred=[]; w=cfg["window"]
        rad=0 if not weather_on else ANOM.index("anom_rad")
        cutoff=train.date.max()-pd.Timedelta(days=w-1)
        window_mask=train.date.ge(cutoff).to_numpy()
        for i,row in enumerate(test.itertuples()):
            a=X[window_mask,rad:rad+1] if weather_on else np.zeros((int(window_mask.sum()),1)); aq=Xt[i,rad] if weather_on else 0
            m=Ridge(alpha=1).fit(a,r_std[window_mask]); pred.append(base[i]+r_center+r_scale*m.predict([[aq]])[0])
        pred=np.asarray(pred)
    elif name == "HARC":
        # Ridge-penalized group/site deviations. Unseen categories receive zero deviation.
        groups=pd.Index(pd.unique(train.grupo)); plants=pd.Index(pd.unique(train.code))
        use_group=len(groups)>1
        def design(frame, values):
            group_idx=groups.get_indexer(frame.grupo); plant_idx=plants.get_indexer(frame.code)
            G=np.zeros((len(frame),len(groups))) if use_group else np.zeros((len(frame),0))
            P=np.zeros((len(frame),len(plants)))
            gvalid=group_idx>=0; pvalid=plant_idx>=0
            if use_group: G[np.arange(len(frame))[gvalid],group_idx[gvalid]]=1.
            P[np.arange(len(frame))[pvalid],plant_idx[pvalid]]=1.
            rad=values[:,0] if values.shape[1] else np.zeros(len(frame))
            return np.column_stack([np.ones(len(frame)),values,G,G*rad[:,None],P,P*rad[:,None]])
        A=design(train,X); At=design(test,Xt)
        ng=len(groups) if use_group else 0
        penalty=np.r_[0.,np.ones(X.shape[1]),np.full(2*ng,cfg["lam_group"]),np.full(2*len(plants),cfg["lam_plant"])]
        beta=np.linalg.solve(A.T@A+np.diag(penalty)+np.eye(A.shape[1])*1e-9,A.T@r_std)
        pred=base+r_center+r_scale*(At@beta)
    elif name == "WCFC":
        # Blend two residual forecasts; the robust weight is selected in temporal CV.
        a=Ridge(alpha=1).fit(X,r_std).predict(Xt)
        b=HuberRegressor(max_iter=250).fit(X,r_std).predict(Xt)
        corr=(1-cfg["weight_huber"])*a+cfg["weight_huber"]*b
        pred=base+r_center+r_scale*corr
    else: raise KeyError(name)
    return _clip_physical(np.asarray(pred,float),test)


def _inner_splits(train):
    dates=np.sort(train.date.unique())
    out=[]
    # Tres ventanas internas de 14 calendario días, separadas del entrenamiento por 30 días.
    # Ventanas cortas permiten seleccionar parámetros también en plantas con historia limitada.
    for fraction in (.55,.70,.85):
        start=min(int(round(len(dates)*fraction)),len(dates)-1)
        if start < 0: continue
        valid_start=pd.Timestamp(dates[start])
        valid_end=valid_start+pd.Timedelta(days=INNER_VALIDATION_DAYS)
        cutoff=valid_start-pd.Timedelta(days=INNER_GAP_DAYS)
        a=train[train.date<cutoff]; b=train[(train.date>=valid_start)&(train.date<valid_end)]
        if len(a)>=30 and len(b)>=10: out.append((a,b))
    return out


def _key_signature(frame: pd.DataFrame) -> str:
    keys=frame[["code","date"]].copy().sort_values(["code","date"])
    keys["date"]=pd.to_datetime(keys.date).dt.strftime("%Y-%m-%d")
    return hashlib.sha256(keys.to_csv(index=False).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance(base_path: Path, forecast_path: Path, weather_on: bool) -> dict:
    source=Path(__file__).resolve()
    return {
        "source_file":str(source.relative_to(ROOT)),
        "source_sha256":_file_sha256(source),
        "data_files":{
            "generation":str(Path(base_path).resolve().relative_to(ROOT)),
            "generation_sha256":_file_sha256(Path(base_path)),
            "weather_forecast":str(Path(forecast_path).resolve().relative_to(ROOT)),
            "weather_forecast_sha256":_file_sha256(Path(forecast_path)),
        },
            "runtime":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__,"xgboost":xgboost.__version__},
        "seed":SEED,
        "weather_enabled":bool(weather_on),
    }


def _feature_map(weather_on: bool) -> dict:
    custom_space=ANOM if weather_on else ["kwh_D","media7_kwh","month_sin","month_cos","doy_sin","doy_cos"]
    return {
        "library_models":_xcols(weather_on),
        "residual_custom_models":custom_space,
        "base_level":"media30_kwh",
        "PDEC_additional_input":"deviation_D",
        "ELAC_level":"ewma_alpha (causal, shifted one day)",
        "KLAC_level":"kalman_q (capacity-normalized causal local level, shifted one day)",
        "HARC_identity_inputs":["grupo","code"],
        "IWLAR_identity_inputs":["grupo","code"],
        "physical_prediction_bound":"0 <= prediction <= 24000 * cap_MW_D when cap_MW_D is available",
    }


def _choose(name, train, weather_on):
    if name=="Media móvil de 30 días": return {}
    splits=_inner_splits(train)
    # First grid entry is the declared fallback when history is too short for inner CV.
    if not splits: return GRIDS[name][0]
    configs=GRIDS[name]
    if len(configs)>MAX_CONFIGURATIONS:
        chosen=np.random.default_rng(SEED).choice(len(configs),MAX_CONFIGURATIONS,replace=False)
        configs=[configs[i] for i in sorted(chosen)]
    scores=[]
    for cfg in configs:
        loss=[]
        for a,b in splits:
            try: loss.append(_macro_mae(b,_fit_predict(name,cfg,a,b,weather_on)))
            except (ValueError,np.linalg.LinAlgError): loss.append(np.inf)
        scores.append(np.mean(loss) if loss else np.inf)
    return configs[int(np.argmin(scores))]


def run_intraplanta(base_path: Path, forecast_path: Path, output: Path, weather_on: bool):
    panel=build_panel(base_path,forecast_path,weather_on)
    audit=leakage_audit(base_path,forecast_path,panel,weather_on)
    records=[]; coverage=[]; choices=[]; timings=[]
    for block,(first,last) in BLOCKS.items():
        first,last=pd.Timestamp(first),pd.Timestamp(last)
        for code,plant in panel.groupby("code",sort=True):
            train=plant[plant.date<first].copy(); test=plant[plant.date.between(first,last)].copy()
            if test.empty: continue
            if train.empty:
                coverage.append({"bloque":block,"code":code,"train_rows":0,"test_rows":len(test),"train_end":pd.NaT,"test_start":test.date.min(),"test_end":test.date.max(),"inner_splits":0,"train_key_sha256":_key_signature(train),"test_key_sha256":_key_signature(test),"inner_key_sha256":"[]","estado":"sin historial previo; fold no entrenable"})
                continue
            if train.date.max()>=test.date.min(): raise ValueError(f"Corte temporal inválido {block}/{code}")
            inner=_inner_splits(train)
            for a_inner,b_inner in inner:
                if a_inner.date.max() >= b_inner.date.min() - pd.Timedelta(days=30):
                    raise ValueError(f"Gap interno menor de 30 días en {block}/{code}")
            for method in METHODS:
                tic=time.perf_counter(); cfg=_choose(method,train,weather_on); fit_seconds=time.perf_counter()-tic
                tic=time.perf_counter()
                try:
                    pred=_fit_predict(method,cfg,train,test,weather_on)
                except Exception as exc:
                    raise RuntimeError(
                        f"Falló el ajuste externo de {method} en {block}, planta {code}, "
                        f"configuración {cfg}; filas train={len(train)}, test={len(test)}"
                    ) from exc
                final_fit_predict_seconds=time.perf_counter()-tic
                for row,p in zip(test.itertuples(index=False),pred):
                    err=abs(p-row.y); denom=float(row.cap_MW_D)*24_000 if np.isfinite(row.cap_MW_D) and row.cap_MW_D>0 else np.nan
                    records.append({"metodo":method,"bloque":block,"code":code,"grupo":row.grupo,"date":row.date,"y_kwh":row.y,"prediccion_kwh":p,"error_abs_kwh":err,"error_capacidad_relativa":err/denom if np.isfinite(denom) else np.nan})
                choices.append({"metodo":method,"bloque":block,"code":code,"hiperparametros":json.dumps(cfg,ensure_ascii=False),"splits_internos":len(inner),"segundos_seleccion":fit_seconds,"segundos_ajuste_y_prediccion_externa":final_fit_predict_seconds})
            inner_sig=[{"train":_key_signature(a),"validation":_key_signature(b)} for a,b in inner]
            coverage.append({"bloque":block,"code":code,"train_rows":len(train),"test_rows":len(test),"train_end":train.date.max(),"test_start":test.date.min(),"test_end":test.date.max(),"inner_splits":len(inner),"train_key_sha256":_key_signature(train),"test_key_sha256":_key_signature(test),"inner_key_sha256":json.dumps(inner_sig),"estado":"evaluado"})
            print(f"{output.name}: {block} planta {code}: {len(train)} train, {len(test)} test, {len(inner)} splits internos", flush=True)
    pred=pd.DataFrame(records); coverage=pd.DataFrame(coverage); choices=pd.DataFrame(choices)
    pred["sq_error"]=(pred.prediccion_kwh-pred.y_kwh)**2
    byplant=pred.groupby(["metodo","code"],as_index=False).agg(MAE_kWh=("error_abs_kwh","mean"),MAE_capacidad_relativa=("error_capacidad_relativa","mean"),RMSE2=("sq_error","mean"),dias=("y_kwh","size"))
    byplant["RMSE_kWh"]=np.sqrt(byplant.pop("RMSE2"))
    globalm=byplant.groupby("metodo",as_index=False).agg(MAE_macro_kWh=("MAE_kWh","mean"),MAE_macro_capacidad_relativa=("MAE_capacidad_relativa","mean"),RMSE_macro_kWh=("RMSE_kWh","mean"),plantas=("code","nunique"),dias=("dias","sum")).sort_values("MAE_macro_kWh")
    blockm=(pred.groupby(["metodo","bloque","code"],as_index=False).agg(MAE_kWh=("error_abs_kwh","mean"),sq=("sq_error","mean")))
    blockm["RMSE_kWh"]=np.sqrt(blockm.pop("sq"))
    blockm=blockm.groupby(["metodo","bloque"],as_index=False).agg(MAE_macro_kWh=("MAE_kWh","mean"),RMSE_macro_kWh=("RMSE_kWh","mean"),plantas=("code","nunique"))
    output.mkdir(parents=True,exist_ok=True)
    pred.to_csv(output/"predicciones.csv",index=False); coverage.to_csv(output/"cobertura_temporal_por_planta.csv",index=False)
    choices.to_csv(output/"seleccion_hiperparametros.csv",index=False); byplant.to_csv(output/"metricas_por_planta.csv",index=False)
    globalm.to_csv(output/"metricas_globales.csv",index=False); blockm.to_csv(output/"metricas_por_bloque.csv",index=False); audit.to_csv(output/"auditoria_fuga.csv",index=False)
    manifest={"target":"producción eléctrica diaria observada (kWh/planta-día)","design":"validación temporal por planta; cada modelo se ajusta con la historia de esa misma planta","outer_blocks":BLOCKS,"outer_folds_total":int(len(coverage)),"outer_folds_evaluated":int((coverage.estado=="evaluado").sum()),"outer_folds_without_prior_training_history":int((coverage.estado!="evaluado").sum()),"inner_splits_per_fold":"hasta 3","inner_splits_total":int(coverage.inner_splits.sum()),"inner_validation_window_days":14,"inner_gap_days":30,"seed":SEED,"max_configurations_per_method":MAX_CONFIGURATIONS,"custom_models":CUSTOM,"library_models":LIBRARIES,"baseline":"Media móvil de 30 días","weather_inputs":METEO if weather_on else [],"common_sample_requires_weather_archive":True,"weather_values_passed_to_estimators":bool(weather_on),"hyperparameter_search":"rejilla congelada con hasta cuatro opciones; selección por MAE macro por planta en hasta 3 splits temporales internos","plants":int(panel.code.nunique()),"origins":int(panel[["code","date"]].drop_duplicates().shape[0]),"scope":"exploratorio; no evalúa transferencia a plantas no vistas","limitations":["No se verificó el sello de publicación XM de kWh(T-1).","Los efectos penalizados de grupo y planta de HARC no se separan al ajustar una planta por vez.","La variante sin meteorología conserva los mismos orígenes disponibles en el archivo de pronósticos; no entrega valores meteorológicos al estimador."]}
    manifest["custom_model_names"] = {
        "RLAC": "Rolling-Level Anomaly Correction",
        "IWLAR": "Identity-Weighted Local Anomaly Regression",
        "WCFC": "Window-Calibrated Forecast Combination",
        "ABGPR": "Anomaly-Based Gaussian Process Regression",
        "HARC": "Hierarchical Anomaly Residual Correction",
        "SRMOS": "Site-Specific Rolling Model Output Statistics",
        "ELAC": "Exponential-Level Anomaly Correction",
        "KLAC": "Kalman-Level Anomaly Correction",
        "RARC": "Robust Anomaly Residual Correction",
        "RQRF": "Residual Quantile Regression Forest",
        "PDEC": "Previous-Day Error Correction",
        "MARB": "Monotone Anomaly Residual Boosting",
    }
    manifest["target_and_predictor_scaling"] = "Fit inside each internal training split only; outer scaling uses outer training data only."
    manifest["MARB_monotonic_constraint"] = "Nondecreasing in radiation anomaly in the weather variant; no meteorological variable is used in the reduced variant."
    manifest["leakage_audit"] = audit.to_dict("records")
    manifest["custom_model_names"] = MODEL_LABELS
    manifest["method_display_names"] = DISPLAY_LABELS
    manifest["feature_map"] = _feature_map(weather_on)
    manifest["effective_grid_sizes"] = {name:len(grid) for name,grid in GRIDS.items()}
    manifest["parameter_grids"] = GRIDS
    manifest["no_inner_split_fallback"] = {name:grid[0] for name,grid in GRIDS.items()}
    manifest["inner_selection_metric"] = "MAE macro por planta; cada planta pesa igual dentro de cada split"
    manifest["modeling_notes"] = {
        "PDEC":"Usa desviación del kWh del día previo frente a su nivel móvil de siete días; no usa error de un pronóstico previo.",
        "WCFC":"Mezcla dos predicciones de residuo Ridge/Huber con peso Huber seleccionado por validación temporal interna.",
        "RQRF":"Bosque cuantílico con pesos de hojas y cuantil ponderado 0.5 de residuos de entrenamiento.",
        "HARC":"Efectos de grupo/sitio penalizados; las contribuciones por separado no son identificables con un solo grupo/planta.",
        "IWLAR":"En estos diseños el peso de identidad es constante; esta ronda no identifica su aporte.",
        "SRMOS":"Ventana de días calendario fija en el corte de entrenamiento externo.",
        "ABGPR":"Muestra de hasta 300 filas recientes, balanceadas por planta.",
        "KLAC":"Filtro local sobre energía normalizada por capacidad; q es una razón adimensional de varianzas; si no hay estado inicial, usa el nivel causal de 30 días.",
    }
    manifest["limitations"] = [
        "La disponibilidad de kwh(T-1) al origen no se confirma porque XM no aporta aquí la hora de publicación.",
        "El benchmark intraplanta usa el historial de la misma planta y no evalúa transferencia a plantas no vistas.",
        "El peso de identidad de IWLAR es constante en esta partición y no se identifica su aporte.",
        "HARC se interpreta como efectos penalizados; en una planta sus componentes group/site no se separan.",
        "La ventana de entrenamiento de SRMOS queda fija durante cada bloque externo.",
        "No se dispone de un periodo confirmatorio virgen a nivel de proyecto.",
    ]
    manifest["provenance"] = _provenance(base_path,forecast_path,weather_on)
    (output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return {"panel":panel,"predicciones":pred,"cobertura":coverage,"seleccion":choices,"por_planta":byplant,"global":globalm,"por_bloque":blockm,"auditoria":audit,"manifest":manifest}


def run_spatiotemporal(base_path: Path, forecast_path: Path, output: Path, weather_on: bool):
    """Leave one spatial group out, crossed with the same forward temporal blocks."""
    panel=build_panel(base_path,forecast_path,weather_on)
    audit=leakage_audit(base_path,forecast_path,panel,weather_on)
    records=[]; coverage=[]; choices=[]
    for block,(first,last) in DEVELOPMENT_BLOCKS.items():
        first,last=pd.Timestamp(first),pd.Timestamp(last)
        for group in sorted(panel.grupo.unique()):
            train=panel[(panel.grupo!=group)&(panel.date<first)].copy()
            test=panel[(panel.grupo==group)&panel.date.between(first,last)].copy()
            if test.empty: continue
            if train.empty or train.date.max()>=test.date.min():
                raise ValueError(f"Corte espaciotemporal inválido {block}/grupo {group}")
            inner=_inner_splits(train)
            for a_inner,b_inner in inner:
                if a_inner.date.max()>=b_inner.date.min()-pd.Timedelta(days=INNER_GAP_DAYS):
                    raise ValueError(f"Gap interno menor de {INNER_GAP_DAYS} días en {block}/grupo {group}")
            for method in METHODS:
                tic=time.perf_counter(); cfg=_choose(method,train,weather_on); seconds=time.perf_counter()-tic
                tic=time.perf_counter()
                try:
                    pred=_fit_predict(method,cfg,train,test,weather_on)
                except Exception as exc:
                    raise RuntimeError(
                        f"Falló el ajuste externo de {method} en {block}, grupo {group}, "
                        f"configuración {cfg}; filas train={len(train)}, test={len(test)}"
                    ) from exc
                final_fit_predict_seconds=time.perf_counter()-tic
                for row,value in zip(test.itertuples(index=False),pred):
                    err=abs(value-row.y); denom=float(row.cap_MW_D)*24_000 if np.isfinite(row.cap_MW_D) and row.cap_MW_D>0 else np.nan
                    records.append({"metodo":method,"bloque":block,"grupo_excluido":group,"grupo":row.grupo,
                                    "code":row.code,"date":row.date,"y_kwh":row.y,
                                    "prediccion_kwh":value,"error_abs_kwh":err,
                                    "error_capacidad_relativa":err/denom if np.isfinite(denom) else np.nan})
                choices.append({"metodo":method,"bloque":block,"grupo_excluido":group,
                                "hiperparametros":json.dumps(cfg,ensure_ascii=False),
                                "splits_internos":len(inner),"segundos_seleccion":seconds,
                                "segundos_ajuste_y_prediccion_externa":final_fit_predict_seconds})
            inner_sig=[{"train":_key_signature(a),"validation":_key_signature(b)} for a,b in inner]
            coverage.append({"bloque":block,"grupo_excluido":group,"train_rows":len(train),"test_rows":len(test),
                             "plantas_entrenamiento":train.code.nunique(),"plantas_prueba":test.code.nunique(),
                             "train_end":train.date.max(),"test_start":test.date.min(),"test_end":test.date.max(),
                             "inner_splits":len(inner),"train_key_sha256":_key_signature(train),"test_key_sha256":_key_signature(test),
                             "inner_key_sha256":json.dumps(inner_sig),"estado":"evaluado"})
            print(f"{output.name}: {block}, grupo {group}: {len(train)} train, {len(test)} test, {len(inner)} splits internos",flush=True)
    pred=pd.DataFrame(records); coverage=pd.DataFrame(coverage); choices=pd.DataFrame(choices)
    pred["sq_error"]=(pred.prediccion_kwh-pred.y_kwh)**2
    byplant=pred.groupby(["metodo","code","grupo"],as_index=False).agg(MAE_kWh=("error_abs_kwh","mean"),MAE_capacidad_relativa=("error_capacidad_relativa","mean"),RMSE2=("sq_error","mean"),dias=("y_kwh","size"))
    byplant["RMSE_kWh"]=np.sqrt(byplant.pop("RMSE2"))
    globalm=byplant.groupby("metodo",as_index=False).agg(MAE_macro_kWh=("MAE_kWh","mean"),MAE_macro_capacidad_relativa=("MAE_capacidad_relativa","mean"),RMSE_macro_kWh=("RMSE_kWh","mean"),plantas=("code","nunique"),dias=("dias","sum")).sort_values("MAE_macro_kWh")
    macro_group=byplant.groupby(["metodo","grupo"],as_index=False).agg(MAE_macro_planta_kWh=("MAE_kWh","mean"),MAE_macro_planta_capacidad_relativa=("MAE_capacidad_relativa","mean"),RMSE_macro_planta_kWh=("RMSE_kWh","mean"))
    group_overall=macro_group.groupby("metodo",as_index=False).agg(MAE_macro_grupo_kWh=("MAE_macro_planta_kWh","mean"),MAE_macro_grupo_capacidad_relativa=("MAE_macro_planta_capacidad_relativa","mean"),RMSE_macro_grupo_kWh=("RMSE_macro_planta_kWh","mean"))
    globalm=globalm.merge(group_overall,on="metodo",validate="one_to_one")
    blockm=pred.groupby(["metodo","bloque","grupo_excluido","code"],as_index=False).agg(MAE_kWh=("error_abs_kwh","mean"),sq=("sq_error","mean"))
    blockm["RMSE_kWh"]=np.sqrt(blockm.pop("sq"))
    blockm=blockm.groupby(["metodo","bloque"],as_index=False).agg(MAE_macro_kWh=("MAE_kWh","mean"),RMSE_macro_kWh=("RMSE_kWh","mean"),plantas=("code","nunique"),grupos=("grupo_excluido","nunique"))
    bygroup=macro_group.rename(columns={"grupo":"grupo_excluido","MAE_macro_planta_kWh":"MAE_macro_kWh","MAE_macro_planta_capacidad_relativa":"MAE_macro_capacidad_relativa","RMSE_macro_planta_kWh":"RMSE_macro_kWh"})
    output.mkdir(parents=True,exist_ok=True)
    pred.to_csv(output/"predicciones.csv",index=False); coverage.to_csv(output/"cobertura_espaciotemporal.csv",index=False)
    choices.to_csv(output/"seleccion_hiperparametros.csv",index=False); byplant.to_csv(output/"metricas_por_planta.csv",index=False)
    globalm.to_csv(output/"metricas_globales.csv",index=False); blockm.to_csv(output/"metricas_por_bloque.csv",index=False)
    bygroup.to_csv(output/"metricas_por_grupo_excluido.csv",index=False); audit.to_csv(output/"auditoria_fuga.csv",index=False)
    manifest={"target":"daily observed electricity production in kWh per plant","outer_design":"leave-one-spatial-group-out crossed with forward temporal blocks D1-D3","outer_folds":len(coverage),"inner_splits":"same 3 rolling temporal splits with 14-day validation windows and a 30-day gap for every method","max_configurations_per_method":MAX_CONFIGURATIONS,"seed":SEED,"weather_inputs":METEO if weather_on else [],"same_common_origins":True,"weather_values_passed_to_estimators":bool(weather_on),"custom_models":CUSTOM,"library_models":LIBRARIES,"baseline":"Media móvil de 30 días","final_block_reserved":FINAL_BLOCK,"leakage_audit":audit.to_dict("records"),"scope":"development only; held-out group has its own past production history available for lagged features"}
    manifest["target_and_predictor_scaling"]="Fit inside each inner training split only; final model fit uses outer training data only."
    manifest["MARB_monotonic_constraint"]="Nondecreasing in radiation anomaly in the weather variant; no meteorological variable is used in the reduced variant."
    manifest["custom_model_names"]=MODEL_LABELS
    manifest["method_display_names"]=DISPLAY_LABELS
    manifest["feature_map"]=_feature_map(weather_on)
    manifest["effective_grid_sizes"]={name:len(grid) for name,grid in GRIDS.items()}
    manifest["parameter_grids"]=GRIDS
    manifest["no_inner_split_fallback"]={name:grid[0] for name,grid in GRIDS.items()}
    manifest["inner_selection_metric"]="MAE macro por planta; cada planta pesa igual dentro de cada split"
    manifest["group_macro_definition"]="Promedio del MAE de plantas dentro de grupo; promedio con igual peso entre grupos."
    manifest["modeling_notes"]={
        "PDEC":"Usa desviación del kWh del día previo frente a su nivel móvil de siete días; no usa error de un pronóstico previo.",
        "WCFC":"Mezcla dos predicciones de residuo Ridge/Huber con peso Huber seleccionado por validación temporal interna.",
        "RQRF":"Bosque cuantílico con pesos de hojas y cuantil ponderado 0.5 de residuos de entrenamiento.",
        "HARC":"Efectos de grupo/sitio penalizados; categorías no vistas reciben desviación cero.",
        "IWLAR":"En estos diseños el peso de identidad es constante; esta ronda no identifica su aporte.",
        "SRMOS":"Ventana de días calendario fija en el corte de entrenamiento externo.",
        "ABGPR":"Muestra de hasta 300 filas recientes, balanceadas por planta.",
        "KLAC":"Filtro local sobre energía normalizada por capacidad; q es una razón adimensional de varianzas; si no hay estado inicial, usa el nivel causal de 30 días.",
    }
    manifest["limitations"]=[
        "El grupo de prueba no entra al ajuste, pero sus plantas sí aportan su propia historia previa a las características de nivel.",
        "La disponibilidad operacional de kwh(T-1) no se confirma sin sello de publicación XM.",
        "El peso de identidad de IWLAR es constante en este diseño y no se identifica su aporte.",
        "Las ventanas D1-D3 son desarrollo exploratorio; D4 ya apareció en análisis anteriores del proyecto.",
    ]
    manifest["provenance"]=_provenance(base_path,forecast_path,weather_on)
    (output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return {"panel":panel,"predicciones":pred,"cobertura":coverage,"seleccion":choices,"por_planta":byplant,
            "por_grupo":bygroup,"global":globalm,"por_bloque":blockm,"auditoria":audit,"manifest":manifest}

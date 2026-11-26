#!/usr/bin/env python3
"""Controlled factorial ablations, monotonicity activation, Huber audit, and selection-bias v3.0.0.

Does NOT modify frozen `resultados/evaluacion_ashr_oof/`.
Writes all outputs to `resultados/ablaciones_controladas_v3/`.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adaptive_structured_huber import AdaptiveStructuredHuberRegressor
from src.ashr_oof_pipeline.config import SCENARIOS_28, ScenarioSpec
from src.ashr_oof_pipeline.oof_runner import (
    ANOM_COLS,
    ENERGY_BOUND,
    METEO_COLS,
    extract_cf_features,
    load_official_benchmarks,
)
from src.sm_cfhr.inference_v2 import (
    build_macro_aligned_panel,
    dm_hln_hac_v2,
    infer_series_step_and_overlap,
    macro_synchronized_block_bootstrap,
)
from src.sm_cfhr.metrics_audit import (
    attach_capacity_and_cf_errors,
    compute_audited_metrics_from_df,
    load_d1_plant_capacity_table,
)
from src.sm_cfhr.representation_benchmark import extract_kwh_features

VERSION = "3.0.0-ablaciones-controladas"
SEED = 2026
SCENARIO_MAP: dict[str, ScenarioSpec] = {sp.scenario_id: sp for sp in SCENARIOS_28}

ABL_SCENARIOS = [
    "D1_espaciotemporal_cinco_variables",
    "D1_intraplanta_cinco_variables",
    "D1_espaciotemporal_sin_meteorologia",
    "RecGEFSv12_espaciotemporal_7d",
    "RecGEFSv12_espaciotemporal_28d",
    "DirectoSinClima_intraplanta_28d",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


class _TargetStdRegressor(AdaptiveStructuredHuberRegressor):
    """Variant that scales target y by sample (mean, STD) instead of (median, MAD)."""

    def _fit_single(self, X, y, cfg, metadata=None):  # type: ignore[override]
        import src.adaptive_structured_huber as ash_mod
        orig_fn = ash_mod._robust_target_stats
        ash_mod._robust_target_stats = lambda arr: (float(np.mean(arr)), max(float(np.std(arr)), 1e-6))
        try:
            return super()._fit_single(X, y, cfg, metadata)
        finally:
            ash_mod._robust_target_stats = orig_fn


def _fit_variant(
    variant: str,
    train: pd.DataFrame,
    task: str,
    horizon: int,
    weather_on: bool,
    horizon_direct: int,
    target_col: str,
) -> dict[str, Any]:
    use_cf = variant not in ("sin_CF_energia_bruta", "sin_CF_y_sin_Huber_L2")
    delta = 1e6 if variant in ("sin_Huber_L2", "sin_CF_y_sin_Huber_L2") else 1.35
    alpha_lin = 0.10 if variant == "sin_L2_anisotropica_010" else 0.01
    alpha_meteo = 0.01 if variant == "sin_L2_anisotropica_001" else 0.10
    use_mono = False if (variant == "sin_monotonia" or not use_cf) else True
    drop_anom = variant == "sin_anomalias_meteo"
    drop_wx = variant == "sin_meteorologia"
    eff_wx = weather_on and (not drop_wx)

    if use_cf:
        X_raw, _, denom_tr = extract_cf_features(train, weather_on=eff_wx, lead=1)
        if eff_wx and drop_anom:
            X_raw = X_raw[:, :12]  # 7 base + 5 raw meteo (no 5 anomalies)
        y_vals = train[target_col].to_numpy(dtype=float)
        valid = np.isfinite(denom_tr) & (denom_tr > 0) & np.isfinite(y_vals)
        X_v = X_raw[valid]
        y_v = np.clip(y_vals[valid] / (denom_tr[valid] * float(horizon_direct)), 0.0, 1.0)
        meteo_idx = tuple(range(7, 12)) if (eff_wx and drop_anom) else (tuple(range(7, 17)) if eff_wx else ())
        mono_idx = (4, 7) if eff_wx else (4,)
    else:
        X_raw, denom_tr = extract_kwh_features(train, weather_on=eff_wx)
        y_vals = train[target_col].to_numpy(dtype=float)
        valid = np.isfinite(denom_tr) & (denom_tr > 0) & np.isfinite(y_vals)
        X_v = X_raw[valid]
        y_v = y_vals[valid]
        meteo_idx = tuple(range(8, 18)) if eff_wx else ()
        mono_idx = ()

    mu = np.nanmedian(X_v, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    X_imp = np.where(np.isfinite(X_v), X_v, mu)

    cls = _TargetStdRegressor if variant == "escala_STD_en_target" else AdaptiveStructuredHuberRegressor
    est = cls(
        mode=task,
        horizon=int(horizon),
        weather_mode="forecast" if eff_wx else "none",
        selection="fixed",
        fixed_config={
            "name": variant,
            "complexity_rank": 1,
            "n_knots": 0,
            "alpha_lin": float(alpha_lin),
            "alpha_meteo": float(alpha_meteo if eff_wx else alpha_lin),
            "alpha_spl": 0.10,
            "max_interactions": 0,
            "alpha_int": 0.10,
            "structured": True,
            "use_mono": use_mono,
            "anchor": "none",
            "lead_damp": 1.0,
        },
        delta=float(delta),
        max_iter=400,
        tol=1e-7,
        random_state=SEED,
    )
    est.fit(X_imp, y_v, metadata={"cf30_idx": 4, "monotone_pos_idx": mono_idx, "meteo_idx": meteo_idx})
    return {
        "variant": variant,
        "use_cf": use_cf,
        "eff_wx": eff_wx,
        "drop_anom": drop_anom,
        "est": est,
        "mu": mu,
        "horizon_direct": horizon_direct,
    }


def _predict_variant(fitted: dict[str, Any], query: pd.DataFrame, lead: int = 1) -> np.ndarray:
    hd = float(fitted["horizon_direct"])
    fallback = query["media30_kwh"].to_numpy(dtype=float) * hd
    if fitted["use_cf"]:
        X_raw, _, denom_q = extract_cf_features(query, weather_on=fitted["eff_wx"], lead=lead)
        if fitted["eff_wx"] and fitted["drop_anom"]:
            X_raw = X_raw[:, :12]
        X_imp = np.where(np.isfinite(X_raw), X_raw, fitted["mu"])
        cf_hat = np.clip(fitted["est"].predict(X_imp), 0.0, 1.0)
        pred_kwh = cf_hat * denom_q * hd
    else:
        X_raw, denom_q = extract_kwh_features(query, weather_on=fitted["eff_wx"])
        X_imp = np.where(np.isfinite(X_raw), X_raw, fitted["mu"])
        pred_kwh = fitted["est"].predict(X_imp)
    pred_kwh = np.where(np.isfinite(pred_kwh), pred_kwh, fallback)
    upper = np.where(np.isfinite(denom_q), denom_q * hd, np.inf)
    return np.clip(pred_kwh, 0.0, upper)


def _variants_for_scenario(sp: ScenarioSpec) -> list[str]:
    base = [
        "SM_CFHR_completo",
        "sin_CF_energia_bruta",
        "sin_Huber_L2",
        "sin_CF_y_sin_Huber_L2",
        "sin_monotonia",
        "escala_STD_en_target",
    ]
    if sp.weather_on:
        base.extend([
            "sin_L2_anisotropica_001",
            "sin_L2_anisotropica_010",
            "sin_anomalias_meteo",
            "sin_meteorologia",
        ])
    return base


def _run_ablation_scenario(
    sp: ScenarioSpec,
    d1_cap_df: pd.DataFrame,
    include_incomplete_d1: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    from scripts.evaluacion import produccion_intraplanta as daily
    from scripts.evaluacion import produccion_multihorizonte as mh
    from scripts.evaluacion import produccion_multihorizonte_clima as mhc
    from scripts.evaluacion import produccion_multihorizonte_directo as mhd

    variants = _variants_for_scenario(sp) if not include_incomplete_d1 else ["SM_CFHR_completo", "sin_Huber_L2"]
    pred_rows: list[dict[str, Any]] = []
    mono_rows: list[dict[str, Any]] = []

    if sp.mode_eval == "d1":
        base_csv = ROOT / "base_generacion_clima" / "base_planta_dia.csv"
        pron_csv = ROOT / "datos_pronostico" / "pronostico_diario.csv"
        if include_incomplete_d1:
            # Build panel keeping horas_xm <= 6 observed rows in evaluation (no imputation of missing targets)
            base_df = pd.read_csv(base_csv, parse_dates=["date"])
            base_df["horas_xm"] = 24  # bypass filter ONLY for panel assembly, then restore real horas_xm for reporting
            panel = daily.build_panel(base_csv, pron_csv, sp.weather_on, base_data=base_df)
        else:
            panel = daily.build_panel(base_csv, pron_csv, sp.weather_on)
        bench = load_official_benchmarks(sp, root=ROOT)

        for blk, bdf in bench.groupby("bloque"):
            first_ts = pd.to_datetime(bdf["date"]).min()
            last_ts = pd.to_datetime(bdf["date"]).max()
            units = sorted(panel["grupo"].unique()) if sp.task == "espaciotemporal" else sorted(panel["code"].unique())
            for unit in units:
                train = (
                    panel[(panel["grupo"] != unit) & (panel["date"] < first_ts)].copy()
                    if sp.task == "espaciotemporal"
                    else panel[(panel["code"] == unit) & (panel["date"] < first_ts)].copy()
                )
                test = (
                    panel[(panel["grupo"] == unit) & panel["date"].between(first_ts, last_ts)].copy()
                    if sp.task == "espaciotemporal"
                    else panel[(panel["code"] == unit) & panel["date"].between(first_ts, last_ts)].copy()
                )
                if train.empty or test.empty:
                    continue
                fitted_map: dict[str, dict[str, Any]] = {}
                for var in variants:
                    ft = _fit_variant(var, train, sp.task, 1, sp.weather_on, 1, "y")
                    fitted_map[var] = ft
                    preds = _predict_variant(ft, test, lead=1)
                    for r_t, val in zip(test.itertuples(index=False), preds):
                        pred_rows.append({
                            "variant": var,
                            "bloque": blk,
                            "code": str(r_t.code),
                            "grupo": int(r_t.grupo),
                            "date": str(pd.Timestamp(r_t.date).date()),
                            "prediccion_kwh": float(val),
                            "y_kwh": float(r_t.y),
                        })
                if "SM_CFHR_completo" in fitted_map and "sin_monotonia" in fitted_map:
                    w_con = np.asarray(fitted_map["SM_CFHR_completo"]["est"].coef_, dtype=float)
                    w_sin = np.asarray(fitted_map["sin_monotonia"]["est"].coef_, dtype=float)
                    mono_rows.append({
                        "escenario": sp.scenario_id,
                        "fold_unit": f"{blk}_{unit}",
                        "w_cf30_con": float(w_con[4]),
                        "w_cf30_sin": float(w_sin[4]),
                        "cf30_active": bool(w_con[4] <= 1e-9 and w_sin[4] < -1e-9),
                        "w_rad_con": float(w_con[7]) if sp.weather_on else np.nan,
                        "w_rad_sin": float(w_sin[7]) if sp.weather_on else np.nan,
                        "rad_active": bool(sp.weather_on and w_con[7] <= 1e-9 and w_sin[7] < -1e-9),
                        "max_abs_coef_diff": float(np.max(np.abs(w_con - w_sin))),
                    })
    elif sp.mode_eval == "direct":
        panel, _ = mhd.build_direct_panel(ROOT / "base_generacion_clima" / "base_planta_dia.csv", sp.horizon_days)
        bench = load_official_benchmarks(sp, root=ROOT)
        origins = sorted(pd.Timestamp(o) for o in bench["origen"].unique())
        units = sorted(panel["grupo"].unique()) if sp.task == "espaciotemporal" else sorted(panel["code"].unique())
        for origin in origins:
            for u in units:
                train = (
                    panel[(panel["code"] == str(u)) & (panel["t_end_target"] <= origin)].copy()
                    if sp.task == "intraplanta"
                    else panel[(panel["grupo"] != int(u)) & (panel["t_end_target"] <= origin)].copy()
                )
                query = (
                    panel[(panel["code"] == str(u)) & (panel["t_origin"] == origin)].copy()
                    if sp.task == "intraplanta"
                    else panel[(panel["grupo"] == int(u)) & (panel["t_origin"] == origin)].copy()
                )
                if len(train) < 30 or query.empty:
                    continue
                fitted_map = {}
                for var in variants:
                    ft = _fit_variant(var, train, sp.task, sp.horizon_days, False, sp.horizon_days, "y_H")
                    fitted_map[var] = ft
                    preds = _predict_variant(ft, query, lead=1)
                    for r_q, val in zip(query.itertuples(index=False), preds):
                        pred_rows.append({
                            "variant": var,
                            "tarea": sp.task,
                            "code": str(r_q.code),
                            "grupo": int(r_q.grupo),
                            "origen": str(origin.date()),
                            "prediccion_total_kwh": float(val),
                            "y_total_kwh": float(r_q.y_H),
                        })
                w_con = np.asarray(fitted_map["SM_CFHR_completo"]["est"].coef_, dtype=float)
                w_sin = np.asarray(fitted_map["sin_monotonia"]["est"].coef_, dtype=float)
                mono_rows.append({
                    "escenario": sp.scenario_id,
                    "fold_unit": f"{u}_{origin.date()}",
                    "w_cf30_con": float(w_con[4]),
                    "w_cf30_sin": float(w_sin[4]),
                    "cf30_active": bool(w_con[4] <= 1e-9 and w_sin[4] < -1e-9),
                    "w_rad_con": np.nan,
                    "w_rad_sin": np.nan,
                    "rad_active": False,
                    "max_abs_coef_diff": float(np.max(np.abs(w_con - w_sin))),
                })
    else:
        base_path = ROOT / "base_generacion_clima" / "base_planta_dia.csv"
        gefs_path = ROOT / "datos_pronostico_multihorizonte" / "pronostico_28d_gefs.csv"
        panel, raw_plants, weather_store = mhc.load_generation_weather_panel(base_path, gefs_path)
        bench = load_official_benchmarks(sp, root=ROOT)
        origins = sorted(pd.Timestamp(o) for o in bench["origen"].unique())
        units = sorted(panel["grupo"].unique()) if sp.task == "espaciotemporal" else sorted(panel["code"].unique())
        for u in units:
            _, eval_specs, _ = mhc._plan_unit_origins(sp.task, u, origins, sp.horizon_days, panel, raw_plants, weather_store)
            for spec in eval_specs:
                origin = spec["origin"]
                origin_ts = pd.Timestamp(origin)
                pred_infos = [raw_plants[c] for c in spec["pred_codes"]]
                train = panel[(panel["grupo"] != int(u)) & (panel["date"] <= origin)].copy()
                if train.empty:
                    continue
                fitted_map = {}
                for var in variants:
                    ft = _fit_variant(var, train, sp.task, sp.horizon_days, sp.weather_on, 1, "y")
                    fitted_map[var] = ft
                    states = [mhc.PlantRecursiveWeatherState(p, origin_ts, sp.horizon_days, weather_store) for p in pred_infos]
                    for lead in range(1, sp.horizon_days + 1):
                        t_date = origin_ts + pd.Timedelta(days=lead)
                        q_df = pd.DataFrame([s.next_features(t_date) for s in states])
                        yhat = _predict_variant(ft, q_df, lead=lead)
                        for i, info in enumerate(pred_infos):
                            n_hist = (origin_ts - info["min_date"]).days + 1
                            idx_t = n_hist + lead - 1
                            ay = float(info["kwh_valid"][idx_t]) if 0 <= idx_t < len(info["kwh_valid"]) else float("nan")
                            if np.isfinite(ay):
                                pred_rows.append({
                                    "variant": var,
                                    "tarea": sp.task,
                                    "code": info["code"],
                                    "grupo": int(info["grupo"]),
                                    "origen": str(origin_ts.date()),
                                    "lead_dias": lead,
                                    "prediccion_kwh": float(yhat[i]),
                                    "y_kwh": ay,
                                })
                            states[i].step_update(float(yhat[i]))
                w_con = np.asarray(fitted_map["SM_CFHR_completo"]["est"].coef_, dtype=float)
                w_sin = np.asarray(fitted_map["sin_monotonia"]["est"].coef_, dtype=float)
                mono_rows.append({
                    "escenario": sp.scenario_id,
                    "fold_unit": f"{u}_{origin_ts.date()}",
                    "w_cf30_con": float(w_con[4]),
                    "w_cf30_sin": float(w_sin[4]),
                    "cf30_active": bool(w_con[4] <= 1e-9 and w_sin[4] < -1e-9),
                    "w_rad_con": float(w_con[7]),
                    "w_rad_sin": float(w_sin[7]),
                    "rad_active": bool(w_con[7] <= 1e-9 and w_sin[7] < -1e-9),
                    "max_abs_coef_diff": float(np.max(np.abs(w_con - w_sin))),
                })

    raw_df = pd.DataFrame(pred_rows)
    if not include_incomplete_d1:
        bench_keys = load_official_benchmarks(sp, root=ROOT, exclude_cf_brh=True)[list(sp.join_keys)].drop_duplicates()
        raw_df = raw_df.merge(bench_keys, on=list(sp.join_keys), how="inner")

    bench_raw = load_official_benchmarks(sp, root=ROOT, exclude_cf_brh=True)
    cf_df = attach_capacity_and_cf_errors(raw_df, sp, bench_raw, d1_cap_df=d1_cap_df)

    time_col = "date" if sp.mode_eval == "d1" else "origen"
    h_ov = int(infer_series_step_and_overlap(cf_df[time_col].unique(), sp.horizon_days)["h_overlap"])
    base_df = cf_df[cf_df["variant"] == "SM_CFHR_completo"]
    base_cf_df = base_df.assign(_pred_cf=base_df["_ae_cf"], _zero=0.0)
    m_base = compute_audited_metrics_from_df(base_df, sp)

    summary_rows: list[dict[str, Any]] = []
    for var, grp in cf_df.groupby("variant", sort=False):
        mm = compute_audited_metrics_from_df(grp, sp)
        if var == "SM_CFHR_completo":
            diff_cf, ci_lo, ci_hi, p_dm = 0.0, 0.0, 0.0, 1.0
        else:
            grp_cf = grp.assign(_pred_cf=grp["_ae_cf"], _zero=0.0)
            p_cf = build_macro_aligned_panel(grp_cf, base_cf_df, sp.join_keys, time_col, "_pred_cf", "_zero")
            dm_r = dm_hln_hac_v2(p_cf["z_macro_series"], h_overlap=h_ov)
            bm_r = macro_synchronized_block_bootstrap(p_cf["panel_sum"], p_cf["panel_cnt"], B=1500, seed=SEED)
            diff_cf = float(p_cf["diff_mae_macro"])
            ci_lo, ci_hi = float(bm_r["ci_lo"]), float(bm_r["ci_hi"])
            p_dm = float(dm_r["p_dm_hln"])
        summary_rows.append({
            "escenario": sp.scenario_id,
            "variant": var,
            "MAE_CF_macro": mm["MAE_CF_macro"],
            "MAE_macro_kWh": mm["MAE_macro_kWh"],
            "delta_CF_vs_completo": mm["MAE_CF_macro"] - m_base["MAE_CF_macro"],
            "delta_pct_CF": 100.0 * (mm["MAE_CF_macro"] / m_base["MAE_CF_macro"] - 1.0),
            "ci95_lo_CF": ci_lo,
            "ci95_hi_CF": ci_hi,
            "p_dm_CF": p_dm,
            "diff_CF_aligned": diff_cf,
        })
    return cf_df, summary_rows, mono_rows


def _huber_operational_and_stress_audit(d1_cf_df: pd.DataFrame) -> pd.DataFrame:
    """Compare Huber vs L2 across operational subsets, target outliers, and covariate leverage."""
    from scripts.evaluacion import produccion_intraplanta as daily

    base_df = pd.read_csv(ROOT / "base_generacion_clima" / "base_planta_dia.csv", parse_dates=["date"])
    base_df["code"] = base_df["code"].astype(str)
    base_df["date_str"] = base_df["date"].dt.strftime("%Y-%m-%d")
    base_df["sy"] = np.where(base_df["gti10"] > 0.2, base_df["fc"] * 24.0 / base_df["gti10"], np.nan)
    sy_map = base_df[["code", "date_str", "sy"]].drop_duplicates(subset=["code", "date_str"])

    rows = []
    # 1) Operational subsets from D1 ET 5var
    sub_h = d1_cf_df[d1_cf_df["variant"] == "SM_CFHR_completo"].merge(
        sy_map, left_on=["code", "date"], right_on=["code", "date_str"], how="left"
    )
    sub_l = d1_cf_df[d1_cf_df["variant"] == "sin_Huber_L2"].merge(
        sy_map, left_on=["code", "date"], right_on=["code", "date_str"], how="left"
    )
    for label, mask_fn in [
        ("Operacional_completo", lambda d: np.ones(len(d), dtype=bool)),
        ("Episodios_bajo_rendimiento_sy<0.3", lambda d: (d["sy"] < 0.3).fillna(False).to_numpy()),
        ("Cola_superior_5pct_error", lambda d: d["_ae_cf"].to_numpy() >= np.quantile(d["_ae_cf"].to_numpy(), 0.95)),
    ]:
        mh = sub_h[mask_fn(sub_h)]
        ml = sub_l[mask_fn(sub_l)]
        rows.append({
            "regimen": label,
            "n_obs": int(len(mh)),
            "MAE_CF_Huber": float(mh["_ae_cf"].mean()),
            "MAE_CF_L2": float(ml["_ae_cf"].mean()),
            "diff_Huber_minus_L2": float(mh["_ae_cf"].mean() - ml["_ae_cf"].mean()),
        })

    # 2) Controlled stress on D1 chronological split: target outliers (5%, 10%) vs covariate leverage (5% rad_pron)
    panel = daily.build_panel(
        ROOT / "base_generacion_clima" / "base_planta_dia.csv",
        ROOT / "datos_pronostico" / "pronostico_diario.csv",
        True,
    )
    tr = panel[(panel["date"] < pd.Timestamp("2025-11-01")) & panel["y"].notna() & panel["cap_MW_D"].notna() & (panel["cap_MW_D"] > 0)].copy()
    va = panel[(panel["date"] >= pd.Timestamp("2025-11-01")) & panel["y"].notna() & panel["cap_MW_D"].notna() & (panel["cap_MW_D"] > 0)].copy()
    rng = np.random.default_rng(SEED)
    y_va = va["y"].to_numpy(float) / (va["cap_MW_D"].to_numpy(float) * ENERGY_BOUND)

    for stress_name, frac, kind in [
        ("Estres_0pct_limpio", 0.0, "none"),
        ("Contaminacion_target_5pct", 0.05, "target"),
        ("Contaminacion_target_10pct", 0.10, "target"),
        ("Apalancamiento_covariable_rad_5pct", 0.05, "leverage"),
    ]:
        tr_s = tr.copy()
        n_pert = int(round(frac * len(tr_s)))
        if n_pert > 0:
            idx_p = rng.choice(len(tr_s), size=n_pert, replace=False)
            if kind == "target":
                sign = rng.choice([-1.0, 1.0], size=n_pert)
                cap_d = tr_s.iloc[idx_p]["cap_MW_D"].to_numpy(float) * ENERGY_BOUND
                tr_s.iloc[idx_p, tr_s.columns.get_loc("y")] = np.clip(
                    tr_s.iloc[idx_p]["y"].to_numpy(float) + sign * 0.6 * cap_d, 0.0, cap_d
                )
            elif kind == "leverage":
                tr_s.iloc[idx_p, tr_s.columns.get_loc("rad_pron")] = tr_s.iloc[idx_p]["rad_pron"].to_numpy(float) * 25.0
                tr_s.iloc[idx_p, tr_s.columns.get_loc("anom_rad")] = tr_s.iloc[idx_p]["anom_rad"].to_numpy(float) + 50.0
        fh = _fit_variant("SM_CFHR_completo", tr_s, "espaciotemporal", 1, True, 1, "y")
        fl = _fit_variant("sin_Huber_L2", tr_s, "espaciotemporal", 1, True, 1, "y")
        ph = _predict_variant(fh, va, 1) / (va["cap_MW_D"].to_numpy(float) * ENERGY_BOUND)
        pl = _predict_variant(fl, va, 1) / (va["cap_MW_D"].to_numpy(float) * ENERGY_BOUND)
        rows.append({
            "regimen": stress_name,
            "n_obs": int(len(va)),
            "MAE_CF_Huber": float(np.nanmean(np.abs(ph - y_va))),
            "MAE_CF_L2": float(np.nanmean(np.abs(pl - y_va))),
            "diff_Huber_minus_L2": float(np.nanmean(np.abs(ph - y_va)) - np.nanmean(np.abs(pl - y_va))),
        })
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.perf_counter()
    out_dir = ROOT / "resultados" / "ablaciones_controladas_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    d1_cap_df = load_d1_plant_capacity_table(ROOT)

    all_summary: list[dict[str, Any]] = []
    all_mono: list[dict[str, Any]] = []
    d1_et_cf_df = pd.DataFrame()

    for sc in ABL_SCENARIOS:
        sp = SCENARIO_MAP[sc]
        print(f"[Ablation v3] Running {sc} ...", flush=True)
        cf_df, s_rows, m_rows = _run_ablation_scenario(sp, d1_cap_df, include_incomplete_d1=False)
        all_summary.extend(s_rows)
        all_mono.extend(m_rows)
        if sc == "D1_espaciotemporal_cinco_variables":
            d1_et_cf_df = cf_df

    df_abl = pd.DataFrame(all_summary)
    df_abl.to_csv(out_dir / "ablaciones_controladas_6_escenarios.csv", index=False)

    # Monotonicity activation summary
    df_mono_folds = pd.DataFrame(all_mono)
    mono_sum = df_mono_folds.groupby("escenario", as_index=False).agg(
        n_folds=("fold_unit", "count"),
        pct_cf30_activa=("cf30_active", lambda s: 100.0 * float(np.mean(s))),
        pct_rad_activa=("rad_active", lambda s: 100.0 * float(np.mean(s))),
        w_cf30_min_sin=("w_cf30_sin", "min"),
        w_rad_min_sin=("w_rad_sin", "min"),
        max_abs_coef_diff=("max_abs_coef_diff", "max"),
    )
    mono_sum.to_csv(out_dir / "auditoria_monotonia_activacion.csv", index=False)

    # Huber operational & stress audit
    df_huber = _huber_operational_and_stress_audit(d1_et_cf_df)
    df_huber.to_csv(out_dir / "auditoria_huber_operacional_y_estres.csv", index=False)

    # Selection bias sensitivity (retaining observed horas_xm <= 6 in D+1)
    bias_rows = []
    for sc in ("D1_espaciotemporal_cinco_variables", "D1_intraplanta_cinco_variables"):
        sp = SCENARIO_MAP[sc]
        _, s_clean, _ = _run_ablation_scenario(sp, d1_cap_df, include_incomplete_d1=False)
        _, s_all, _ = _run_ablation_scenario(sp, d1_cap_df, include_incomplete_d1=True)
        for tag, s_list in [("Filtrado_horas>6", s_clean), ("Con_incompletos_<=6h_observados", s_all)]:
            for r in s_list:
                if r["variant"] in ("SM_CFHR_completo", "sin_Huber_L2"):
                    bias_rows.append({"escenario": sc, "muestra": tag, **r})
    pd.DataFrame(bias_rows).to_csv(out_dir / "sensibilidad_sesgo_seleccion_horas_xm.csv", index=False)

    # Verdicts table per component
    verdict_rows = []
    for comp_var, comp_label in [
        ("sin_CF_energia_bruta", "Normalización causal por factor de capacidad"),
        ("sin_Huber_L2", "Pérdida de Huber (delta=1.35) vs. L2"),
        ("sin_L2_anisotropica_001", "Regularización anisotrópica vs. isotrópica (0.01)"),
        ("sin_L2_anisotropica_010", "Regularización anisotrópica vs. isotrópica (0.10)"),
        ("sin_monotonia", "Restricciones de monotonía (cf30 >= 0, rad >= 0)"),
        ("escala_STD_en_target", "Escalado MAD vs. STD en objetivo"),
        ("sin_anomalias_meteo", "Anomalías meteorológicas causales"),
        ("sin_meteorologia", "Variables meteorológicas NWP"),
    ]:
        sub = df_abl[df_abl["variant"] == comp_var]
        if sub.empty:
            continue
        # Removing component -> positive delta_CF_vs_completo means component HELPS (removing it increases error)
        mean_delta_pct = float(sub["delta_pct_CF"].mean())
        n_pos = int((sub["delta_pct_CF"] > 0.25).sum())
        n_neg = int((sub["delta_pct_CF"] < -0.25).sum())
        if n_pos == len(sub):
            verdict = "Mejora observada"
        elif n_neg == len(sub):
            verdict = "Degradación"
        elif n_pos > 0 and n_neg > 0:
            verdict = "Resultado dependiente del escenario"
        elif np.max(np.abs(sub["delta_pct_CF"])) < 0.25:
            verdict = "Sin diferencia apreciable"
        else:
            verdict = "Resultado dependiente del escenario"
        verdict_rows.append({
            "componente": comp_label,
            "variante_ablacion": comp_var,
            "n_escenarios": int(len(sub)),
            "media_delta_pct_CF_al_retirar": round(mean_delta_pct, 3),
            "min_delta_pct_CF": round(float(sub["delta_pct_CF"].min()), 3),
            "max_delta_pct_CF": round(float(sub["delta_pct_CF"].max()), 3),
            "veredicto": verdict,
        })
    pd.DataFrame(verdict_rows).to_csv(out_dir / "veredictos_componentes_v3.csv", index=False)

    meta = {
        "version": VERSION,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed": SEED,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "scenarios": ABL_SCENARIOS,
    }
    (out_dir / "metadatos_ablaciones_v3.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    man_rows = []
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.name != "MANIFIESTO_ABLACIONES_V3.csv":
            man_rows.append({"ruta_relativa": p.relative_to(out_dir).as_posix(), "tamano_bytes": p.stat().st_size, "sha256": _sha256(p)})
    with (out_dir / "MANIFIESTO_ABLACIONES_V3.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ruta_relativa", "tamano_bytes", "sha256"], lineterminator="\n")
        w.writeheader()
        w.writerows(man_rows)
    print(f"Controlled ablations v3 complete in {meta['elapsed_s']:.1f}s ({len(man_rows)} artifacts).")


if __name__ == "__main__":
    main()

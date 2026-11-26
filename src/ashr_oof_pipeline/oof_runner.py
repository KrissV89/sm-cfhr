"""Modular OOF evaluation pipeline for frozen ASHR across all 28 solar scenarios.

Does NOT execute when `execute=False` (dry-run / audit mode by default).
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    SCENARIOS_28,
    ScenarioSpec,
    build_frozen_ashr,
    find_project_root,
    load_frozen_config,
    sha256_file,
    verify_benchmark_inventory,
)

ENERGY_BOUND = 24_000.0
LEAD_DAMPING = 0.85
METEO_COLS = ["rad_pron", "temp_pron", "nube_pron", "humedad_pron", "viento_pron"]
ANOM_COLS = ["anom_rad", "anom_temp", "anom_nube", "anom_humedad", "anom_viento"]


def extract_cf_features(
    df: pd.DataFrame, weather_on: bool, lead: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build strictly causal dimensionless predictors, CF30, and capacity-energy denominator."""
    cap = df["cap_MW_D"].to_numpy(dtype=float)
    denom = np.where(np.isfinite(cap) & (cap > 0), cap * ENERGY_BOUND, np.nan)
    cf30 = df["media30_kwh"].to_numpy(dtype=float) / denom
    cf7 = df["media7_kwh"].to_numpy(dtype=float) / denom
    cf1 = df["kwh_D"].to_numpy(dtype=float) / denom

    cf30 = np.where(np.isfinite(cf30), np.clip(cf30, 0.0, 1.0), 0.15)
    cf7 = np.where(np.isfinite(cf7), np.clip(cf7, 0.0, 1.0), cf30)
    cf1 = np.where(np.isfinite(cf1), np.clip(cf1, 0.0, 1.0), cf7)

    damp = float(LEAD_DAMPING ** max(0, lead - 1))
    cols = [
        df["month_sin"].to_numpy(dtype=float),
        df["month_cos"].to_numpy(dtype=float),
        df["doy_sin"].to_numpy(dtype=float),
        df["doy_cos"].to_numpy(dtype=float),
        cf30,
        (cf7 - cf30) * damp,
        (cf1 - cf7) * damp,
    ]
    if weather_on:
        for col in METEO_COLS + ANOM_COLS:
            cols.append(df[col].to_numpy(dtype=float))
    return np.column_stack(cols), cf30, denom


def build_raw_plants_from_panel(panel_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Reconstruct daily `raw_plants` metadata from a panel DataFrame without inventing history."""
    time_col = next((c for c in ("date", "t_origin", "origen") if c in panel_df.columns), "date")
    val_col = "y" if "y" in panel_df.columns else ("y_kwh" if "y_kwh" in panel_df.columns else "kwh")
    cap_col = "cap_MW_eval" if "cap_MW_eval" in panel_df.columns else ("cap_MW_D" if "cap_MW_D" in panel_df.columns else "cap_MW")

    df = panel_df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    if "code" not in df.columns:
        df["code"] = "PLANT"
    df["code"] = df["code"].astype(str)
    if "grupo" not in df.columns:
        df["grupo"] = 0

    raw_plants: dict[str, dict[str, Any]] = {}
    for code, grp in df.groupby("code", sort=True):
        g = grp.sort_values(time_col).drop_duplicates(subset=[time_col], keep="last").set_index(time_col).asfreq("D")
        kwh_arr = g[val_col].to_numpy(dtype=float)
        if cap_col in g.columns:
            cap_arr = g[cap_col].ffill().to_numpy(dtype=float)
        else:
            cap_arr = np.full(len(g), np.nan, dtype=float)
        # Ensure origin day (`min_date - 1` or first row's `kwh_D`) is populated when available
        raw_plants[str(code)] = {
            "code": str(code),
            "grupo": int(grp["grupo"].iloc[0]),
            "min_date": g.index.min(),
            "max_date": g.index.max(),
            "dates": g.index,
            "kwh_valid": kwh_arr,
            "cap_MW": cap_arr,
        }
    return raw_plants


def rollout_recursive_from_origin(
    fitted: dict[str, Any],
    pred_infos: list[dict[str, Any]],
    origin: pd.Timestamp,
    horizon_days: int,
    weather_on: bool,
    weather_store: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Execute exact causal multi-horizon rollout from `origin` using PlantRecursiveState."""
    from scripts.evaluacion import produccion_multihorizonte as mh
    from scripts.evaluacion import produccion_multihorizonte_clima as mhc

    origin_ts = pd.Timestamp(origin)
    states = (
        [mhc.PlantRecursiveWeatherState(p, origin_ts, int(horizon_days), weather_store) for p in pred_infos]
        if weather_on
        else [mh.PlantRecursiveState(p, origin_ts, int(horizon_days)) for p in pred_infos]
    )
    rows: list[dict[str, Any]] = []
    for lead in range(1, int(horizon_days) + 1):
        target_date = origin_ts + pd.Timedelta(days=lead)
        query = pd.DataFrame([st.next_features(target_date) for st in states])
        yhat = predict_ashr_on_query(fitted, query, lead=lead)
        for i, info in enumerate(pred_infos):
            n_hist = (origin_ts - info["min_date"]).days + 1
            idx_target = n_hist + lead - 1
            ay = float(info["kwh_valid"][idx_target]) if 0 <= idx_target < len(info["kwh_valid"]) else float("nan")
            if np.isfinite(ay):
                rows.append({
                    "code": info["code"],
                    "grupo": int(info["grupo"]),
                    "origen": str(origin_ts.date()),
                    "lead_dias": lead,
                    "prediccion_kwh": float(yhat[i]),
                    "y_kwh": ay,
                })
            states[i].step_update(float(yhat[i]))
    return rows


def fit_ashr_on_train(
    train: pd.DataFrame,
    task: str,
    horizon: int,
    weather_on: bool,
    horizon_direct: int = 1,
    target_col: str = "y",
    root: Path | None = None,
    use_hpo: bool = False,
    mode_eval: str = "d1",
    n_trials: int = 25,
    outer_fold_id: str | int = 0,
    seed: int = 2026,
    storage_url: str | None = None,
    study_name: str | None = None,
    proto_hash: str | None = None,
    raw_plants: dict[str, dict[str, Any]] | None = None,
    weather_store: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit SM-CFHR-Fixed or SM-CFHR-HPO strictly on `train` with fold-internal imputation."""
    trials_df = pd.DataFrame()
    chosen_hp = {"delta": 1.35, "alpha_lin": 0.01, "alpha_meteo": 0.10 if weather_on else 0.01}

    if use_hpo:
        from .optimizacion_hiperparametros import (
            _build_ashr_fixed_with_hp,
            optimize_hyperparams_tpe,
        )

        chosen_hp, trials_df = optimize_hyperparams_tpe(
            train_outer=train,
            task=task,
            horizon_days=horizon,
            weather_on=weather_on,
            mode_eval=mode_eval,
            n_trials=n_trials,
            seed=seed,
            outer_fold_id=outer_fold_id,
            horizon_direct=horizon_direct,
            target_col=target_col,
            storage_url=storage_url,
            study_name=study_name,
            proto_hash=proto_hash,
            raw_plants=raw_plants,
            weather_store=weather_store,
        )
        est = _build_ashr_fixed_with_hp(
            task=task,
            horizon=horizon,
            weather_on=weather_on,
            delta=chosen_hp["delta"],
            alpha_lin=chosen_hp["alpha_lin"],
            alpha_meteo=chosen_hp["alpha_meteo"],
            seed=seed,
        )
    else:
        est = build_frozen_ashr(
            mode=task,
            horizon=horizon,
            weather_mode="forecast" if weather_on else "none",
            root=root,
        )

    X_raw, _, denom_tr = extract_cf_features(train, weather_on=weather_on, lead=1)
    y_vals = train[target_col].to_numpy(dtype=float)
    valid = np.isfinite(denom_tr) & (denom_tr > 0) & np.isfinite(y_vals)
    X_raw = X_raw[valid]
    cf_y = np.clip(y_vals[valid] / (denom_tr[valid] * float(horizon_direct)), 0.0, 1.0)

    mu = np.nanmedian(X_raw, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    X_imp = np.where(np.isfinite(X_raw), X_raw, mu)

    meta_tr = {
        "cf30_idx": 4,
        "monotone_pos_idx": (4, 7) if weather_on else (4,),
        "meteo_idx": tuple(range(7, 17)) if weather_on else (),
    }
    est.fit(X_imp, cf_y, metadata=meta_tr)
    return {
        "est": est,
        "mu": mu,
        "weather_on": weather_on,
        "horizon_direct": horizon_direct,
        "chosen_hp": chosen_hp,
        "trials_df": trials_df,
    }


def predict_ashr_on_query(
    fitted: dict[str, Any], query: pd.DataFrame, lead: int = 1
) -> np.ndarray:
    """Predict kWh (daily or cumulative H-day) using pre-fitted ASHR and physical clipping."""
    X_raw, _, denom_q = extract_cf_features(
        query, weather_on=fitted["weather_on"], lead=lead
    )
    X_imp = np.where(np.isfinite(X_raw), X_raw, fitted["mu"])
    cf_hat = np.clip(fitted["est"].predict(X_imp), 0.0, 1.0)
    hd = float(fitted["horizon_direct"])
    pred_kwh = cf_hat * denom_q * hd
    fallback = query["media30_kwh"].to_numpy(dtype=float) * hd
    pred_kwh = np.where(np.isfinite(pred_kwh), pred_kwh, fallback)
    return np.clip(pred_kwh, 0.0, np.where(np.isfinite(denom_q), denom_q * hd, np.inf))


def load_official_benchmarks(
    scenario: ScenarioSpec, root: Path | None = None, exclude_cf_brh: bool = True
) -> pd.DataFrame:
    """Load official benchmark predictions for `scenario`, excluding CF-BRH by default."""
    r = root or find_project_root()
    df = pd.read_csv(r / scenario.benchmark_relpath)
    if "tarea" in df.columns:
        df = df[df["tarea"] == scenario.task].copy()
    if exclude_cf_brh:
        df = df[df["metodo"] != "CF-BRH"].copy()
    df["code"] = df["code"].astype(str)
    return df


def align_predictions(
    ashr_df: pd.DataFrame, bench_df: pd.DataFrame, join_keys: tuple[str, ...], strict: bool = True
) -> pd.DataFrame:
    """Align ASHR predictions against official benchmark rows on exact `join_keys` without silent drops."""
    a = ashr_df.copy()
    b = bench_df.copy()
    a["code"] = a["code"].astype(str)
    b["code"] = b["code"].astype(str)
    keys = list(join_keys)
    if a[keys].isna().any().any():
        raise ValueError(f"Claves faltantes (NaN) en predicciones ASHR para {keys}.")
    if a.duplicated(subset=keys).any():
        raise ValueError(f"Claves duplicadas en predicciones ASHR para {keys}.")
    keys_df = a[keys].drop_duplicates()
    merged = b.merge(keys_df, on=keys, how="inner")
    if strict:
        matched_keys = merged[keys].drop_duplicates()
        if len(matched_keys) < len(keys_df):
            raise ValueError(
                f"Alineación incompleta: {len(keys_df) - len(matched_keys)} claves de ASHR no existen en benchmark."
            )
    return merged


def artifact_cache_valid(
    out_dir: Path,
    expected_src_hash: str,
    variante: str = "fixed",
    expected_hpo_proto_hash: str | None = None,
) -> bool:
    """Check whether cached OOF artifacts match frozen SHA-256, HPO protocol v1.0.3, and variants."""
    from .optimizacion_hiperparametros import HPO_PROTOCOL_VERSION

    manifest_path = out_dir / "manifest_ashr_oof.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("sha256_src_adaptive_structured_huber") != expected_src_hash
            or manifest.get("estatus") != "OOF_DEFINITIVO_COMPLETO"
        ):
            return False
        req_vars = {"fixed", "hpo"} if variante == "ambos" else {variante}
        done_vars = set(manifest.get("variantes_completadas", [manifest.get("variante", "fixed")]))
        if not req_vars.issubset(done_vars):
            return False
        if "hpo" in req_vars:
            if manifest.get("hpo_protocol_version") != HPO_PROTOCOL_VERSION:
                return False
            if expected_hpo_proto_hash is not None and manifest.get("hpo_protocol_hash") != expected_hpo_proto_hash:
                return False
        return True
    except (json.JSONDecodeError, OSError):
        return False


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_to_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _persist_unit_fragment(
    ckpt_mgr: Any,
    var_label: str,
    scenario_id: str,
    unit_tag: str,
    unit_df: pd.DataFrame,
    trials_df: pd.DataFrame | None = None,
) -> None:
    if ckpt_mgr is None or unit_df.empty:
        return
    try:
        from sm_cfhr.persistence import WorkUnitLock, atomic_write_csv, atomic_write_parquet
    except ImportError:
        from src.sm_cfhr.persistence import WorkUnitLock, atomic_write_csv, atomic_write_parquet

    unit_key = f"{var_label}__{scenario_id}__{unit_tag}"
    frag_p = ckpt_mgr.run_dir / "predicciones" / "fragments" / var_label / scenario_id / f"{unit_tag}.parquet"
    with WorkUnitLock(ckpt_mgr.run_dir / "checkpoints" / f"{unit_key}.lock"):
        sha = atomic_write_parquet(unit_df, frag_p)
        if trials_df is not None and not trials_df.empty:
            t_p = ckpt_mgr.run_dir / "hpo_trials" / "fragments" / scenario_id / f"{unit_tag}.csv"
            atomic_write_csv(trials_df, t_p)
        ckpt_mgr.mark_unit(unit_key, "COMPLETED", artifact_path=frag_p, artifact_sha256=sha, n_rows=len(unit_df))
        ckpt_mgr.check_memory_guard()


def _try_load_unit_fragment(
    ckpt_mgr: Any,
    var_label: str,
    scenario_id: str,
    unit_tag: str,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if ckpt_mgr is None:
        return None, None
    unit_key = f"{var_label}__{scenario_id}__{unit_tag}"
    if not ckpt_mgr.is_unit_completed(unit_key):
        return None, None
    frag_p = ckpt_mgr.run_dir / "predicciones" / "fragments" / var_label / scenario_id / f"{unit_tag}.parquet"
    t_p = ckpt_mgr.run_dir / "hpo_trials" / "fragments" / scenario_id / f"{unit_tag}.csv"
    df_u = pd.read_parquet(frag_p)
    df_t = pd.read_csv(t_p) if t_p.exists() else None
    return df_u, df_t


def _run_d1_scenario(
    sp: ScenarioSpec,
    root: Path,
    use_hpo: bool = False,
    n_trials: int = 25,
    storage_url: str | None = None,
    proto_hash: str | None = None,
    ckpt_mgr: Any = None,
    var_label: str = "fixed",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scripts.evaluacion import produccion_intraplanta as daily

    base_path = root / "base_generacion_clima" / "base_planta_dia.csv"
    forecast_path = root / "datos_pronostico" / "pronostico_diario.csv"
    panel = daily.build_panel(base_path, forecast_path, sp.weather_on)
    bench = load_official_benchmarks(sp, root=root)

    unit_frames, trials_list = [], []
    if "bloque" in bench.columns:
        for blk, bdf in bench.groupby("bloque"):
            first_ts = pd.to_datetime(bdf["date"]).min()
            last_ts = pd.to_datetime(bdf["date"]).max()
            units = sorted(panel["grupo"].unique()) if sp.task == "espaciotemporal" else sorted(panel["code"].unique())
            for unit in units:
                unit_tag = f"{blk}_{unit}"
                cached_u, cached_t = _try_load_unit_fragment(ckpt_mgr, var_label, sp.scenario_id, unit_tag)
                if cached_u is not None:
                    unit_frames.append(cached_u)
                    if cached_t is not None:
                        trials_list.append(cached_t)
                    continue

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
                fitted = fit_ashr_on_train(
                    train, sp.task, 1, sp.weather_on, root=root,
                    use_hpo=use_hpo, mode_eval="d1", n_trials=n_trials,
                    outer_fold_id=unit_tag,
                    storage_url=storage_url,
                    study_name=f"{sp.scenario_id}__{unit_tag}",
                    proto_hash=proto_hash,
                )
                t_df = fitted["trials_df"]
                if not t_df.empty:
                    trials_list.append(t_df)
                p_hat = predict_ashr_on_query(fitted, test, lead=1)
                u_rows = [
                    {
                        "metodo": "ASHR-HPO" if use_hpo else "ASHR",
                        "bloque": blk, "code": str(r_t.code),
                        "grupo": int(r_t.grupo), "date": str(pd.Timestamp(r_t.date).date()),
                        "prediccion_kwh": float(val), "y_kwh": float(r_t.y),
                    }
                    for r_t, val in zip(test.itertuples(index=False), p_hat)
                ]
                u_df = pd.DataFrame(u_rows)
                _persist_unit_fragment(ckpt_mgr, var_label, sp.scenario_id, unit_tag, u_df, t_df)
                unit_frames.append(u_df)
                del fitted
                gc.collect()
    df_all = pd.concat(unit_frames, ignore_index=True) if unit_frames else pd.DataFrame()
    trials_all = pd.concat(trials_list, ignore_index=True) if trials_list else pd.DataFrame()
    return df_all, trials_all


def _run_recursive_scenario(
    sp: ScenarioSpec,
    root: Path,
    use_hpo: bool = False,
    n_trials: int = 25,
    storage_url: str | None = None,
    proto_hash: str | None = None,
    ckpt_mgr: Any = None,
    var_label: str = "fixed",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scripts.evaluacion import produccion_multihorizonte as mh
    from scripts.evaluacion import produccion_multihorizonte_clima as mhc

    base_path = root / "base_generacion_clima" / "base_planta_dia.csv"
    gefs_path = root / "datos_pronostico_multihorizonte" / "pronostico_28d_gefs.csv"
    if sp.weather_on:
        panel, raw_plants, weather_store = mhc.load_generation_weather_panel(base_path, gefs_path)
        plan_fn = lambda u, origs: mhc._plan_unit_origins(sp.task, u, origs, sp.horizon_days, panel, raw_plants, weather_store)
    else:
        panel, raw_plants = mh.load_generation_panel(base_path)
        weather_store = None
        plan_fn = lambda u, origs: mh._plan_unit_origins(sp.task, u, origs, sp.horizon_days, panel, raw_plants)

    bench = load_official_benchmarks(sp, root=root)
    origins = sorted(pd.Timestamp(o) for o in bench["origen"].unique())
    units = sorted(panel["grupo"].unique()) if sp.task == "espaciotemporal" else sorted(panel["code"].unique())

    unit_frames, trials_list = [], []
    for u in units:
        _, eval_specs, _ = plan_fn(u, origins)
        for spec in eval_specs:
            origin = spec["origin"]
            unit_tag = f"{u}_{origin.date()}"
            cached_u, cached_t = _try_load_unit_fragment(ckpt_mgr, var_label, sp.scenario_id, unit_tag)
            if cached_u is not None:
                unit_frames.append(cached_u)
                if cached_t is not None:
                    trials_list.append(cached_t)
                continue

            pred_infos = [raw_plants[c] for c in spec["pred_codes"]]
            train = (
                panel[(panel["code"] == str(u)) & (panel["date"] <= origin)].copy()
                if sp.task == "intraplanta"
                else panel[(panel["grupo"] != int(u)) & (panel["date"] <= origin)].copy()
            )
            fitted = fit_ashr_on_train(
                train, sp.task, sp.horizon_days, sp.weather_on, root=root,
                use_hpo=use_hpo, mode_eval="recursive", n_trials=n_trials,
                outer_fold_id=unit_tag,
                storage_url=storage_url,
                study_name=f"{sp.scenario_id}__{unit_tag}",
                proto_hash=proto_hash,
                raw_plants=raw_plants,
                weather_store=weather_store,
            )
            t_df = fitted["trials_df"]
            if not t_df.empty:
                trials_list.append(t_df)
            raw_rows = rollout_recursive_from_origin(
                fitted=fitted,
                pred_infos=pred_infos,
                origin=origin,
                horizon_days=sp.horizon_days,
                weather_on=sp.weather_on,
                weather_store=weather_store,
            )
            metodo_lbl = "ASHR-HPO" if use_hpo else "ASHR"
            u_rows = [
                {"metodo": metodo_lbl, "tarea": sp.task, **rw}
                for rw in raw_rows
            ]
            u_df = pd.DataFrame(u_rows)
            _persist_unit_fragment(ckpt_mgr, var_label, sp.scenario_id, unit_tag, u_df, t_df)
            unit_frames.append(u_df)
            del fitted
            gc.collect()
    df_all = pd.concat(unit_frames, ignore_index=True) if unit_frames else pd.DataFrame()
    trials_all = pd.concat(trials_list, ignore_index=True) if trials_list else pd.DataFrame()
    return df_all, trials_all


def _run_direct_scenario(
    sp: ScenarioSpec,
    root: Path,
    use_hpo: bool = False,
    n_trials: int = 25,
    storage_url: str | None = None,
    proto_hash: str | None = None,
    ckpt_mgr: Any = None,
    var_label: str = "fixed",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scripts.evaluacion import produccion_multihorizonte_directo as mhd

    base_path = root / "base_generacion_clima" / "base_planta_dia.csv"
    panel, _ = mhd.build_direct_panel(base_path, sp.horizon_days)
    bench = load_official_benchmarks(sp, root=root)
    origins = sorted(pd.Timestamp(o) for o in bench["origen"].unique())
    units = sorted(panel["grupo"].unique()) if sp.task == "espaciotemporal" else sorted(panel["code"].unique())

    unit_frames, trials_list = [], []
    for origin in origins:
        for u in units:
            unit_tag = f"{u}_{origin.date()}"
            cached_u, cached_t = _try_load_unit_fragment(ckpt_mgr, var_label, sp.scenario_id, unit_tag)
            if cached_u is not None:
                unit_frames.append(cached_u)
                if cached_t is not None:
                    trials_list.append(cached_t)
                continue

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
            fitted = fit_ashr_on_train(
                train, sp.task, sp.horizon_days, False,
                horizon_direct=sp.horizon_days, target_col="y_H", root=root,
                use_hpo=use_hpo, mode_eval="direct", n_trials=n_trials,
                outer_fold_id=unit_tag,
                storage_url=storage_url,
                study_name=f"{sp.scenario_id}__{unit_tag}",
                proto_hash=proto_hash,
            )
            t_df = fitted["trials_df"]
            if not t_df.empty:
                trials_list.append(t_df)
            yhat = predict_ashr_on_query(fitted, query, lead=1)
            u_rows = [
                {
                    "metodo": "ASHR-HPO" if use_hpo else "ASHR",
                    "tarea": sp.task, "code": str(r_q.code),
                    "grupo": int(r_q.grupo), "origen": str(origin.date()),
                    "prediccion_total_kwh": float(val), "y_total_kwh": float(r_q.y_H),
                }
                for r_q, val in zip(query.itertuples(index=False), yhat)
            ]
            u_df = pd.DataFrame(u_rows)
            _persist_unit_fragment(ckpt_mgr, var_label, sp.scenario_id, unit_tag, u_df, t_df)
            unit_frames.append(u_df)
            del fitted
            gc.collect()
    df_all = pd.concat(unit_frames, ignore_index=True) if unit_frames else pd.DataFrame()
    trials_all = pd.concat(trials_list, ignore_index=True) if trials_list else pd.DataFrame()
    return df_all, trials_all


def run_oof_definitivo(
    execute: bool = False,
    out_dir: Path | None = None,
    root: Path | None = None,
    variante: str = "fixed",
    n_trials: int = 25,
    run_id: str | None = None,
    max_rss_mb: float = 4096.0,
) -> dict[str, Any]:
    """Run or dry-run the definitive 28-scenario OOF evaluation with incremental persistence and checkpoints."""
    from .optimizacion_hiperparametros import HPO_PROTOCOL_VERSION, compute_hpo_protocol_hash

    r = root or find_project_root()
    cfg = load_frozen_config(r)
    src_hash = sha256_file(r / "src" / "adaptive_structured_huber.py")
    expected_hash = cfg["sha256_src_adaptive_structured_huber"]
    hpo_proto_hash = compute_hpo_protocol_hash(r)
    inv = verify_benchmark_inventory(r)
    target_dir = out_dir or (r / "resultados" / "evaluacion_ashr_oof")
    effective_run_id = run_id or f"oof_{variante}"

    if not execute:
        return {
            "estatus": "DRY_RUN_PREPARADO",
            "ejecutado": False,
            "variante": variante,
            "run_id": effective_run_id,
            "n_trials_hpo": int(n_trials) if variante in ("hpo", "ambos") else 0,
            "n_escenarios": len(SCENARIOS_28),
            "inventario_ok": bool(inv["file_exists"].all() and inv["cols_ok"].all()),
            "cf_brh_excluido_en_todos": bool(inv["cf_brh_excluded"].all()),
            "sha256_src_adaptive_structured_huber": src_hash,
            "config_hash_esperado": expected_hash,
            "hashes_coinciden": bool(src_hash == expected_hash),
            "hpo_protocol_version": HPO_PROTOCOL_VERSION,
            "hpo_protocol_hash": hpo_proto_hash,
            "out_dir": str(target_dir),
        }

    if src_hash != expected_hash:
        raise RuntimeError(
            f"Hash SHA-256 desalineado: actual={src_hash} vs esperado={expected_hash}."
        )

    if artifact_cache_valid(target_dir, src_hash, variante=variante, expected_hpo_proto_hash=hpo_proto_hash):
        return {
            "estatus": "CACHE_VALIDO_REUTILIZADO",
            "ejecutado": False,
            "variante": variante,
            "out_dir": str(target_dir),
        }

    try:
        from sm_cfhr.persistence import RunCheckpointManager, WorkUnitLock, atomic_write_parquet
    except ImportError:
        from src.sm_cfhr.persistence import RunCheckpointManager, WorkUnitLock, atomic_write_parquet

    cfg_hash = sha256_file(r / "experiments" / "config_ashr_congelada.json")
    run_variants = ("fixed", "hpo") if variante == "ambos" else (variante,)
    ckpt_proto = hpo_proto_hash if "hpo" in run_variants else expected_hash
    ckpt_mgr = RunCheckpointManager(
        base_dir=target_dir,
        run_id=effective_run_id,
        config_hash=cfg_hash,
        src_hash=src_hash,
        proto_hash=ckpt_proto,
        max_rss_mb=max_rss_mb,
        total_units=len(run_variants) * len(SCENARIOS_28),
    )

    t0 = time.perf_counter()
    pred_dir = target_dir / "predicciones"
    pred_dir.mkdir(parents=True, exist_ok=True)
    scenario_times: dict[str, float] = {}
    scenario_hashes: dict[str, str] = {}

    for var in run_variants:
        use_hpo = var == "hpo"
        hpo_dir = target_dir / "hpo_trials"
        if use_hpo:
            hpo_dir.mkdir(parents=True, exist_ok=True)
        for sp in SCENARIOS_28:
            key = f"{var}__{sp.scenario_id}"
            out_csv = pred_dir / (f"{sp.scenario_id}.csv" if var == "fixed" else f"{sp.scenario_id}_hpo.csv")
            frag_pq = ckpt_mgr.run_dir / "predicciones" / "fragments" / f"{key}.parquet"

            if ckpt_mgr.is_unit_completed(key):
                scenario_hashes[key] = sha256_file(out_csv)
                continue

            with WorkUnitLock(ckpt_mgr.run_dir / "checkpoints" / f"{key}.lock"):
                ckpt_mgr.mark_unit(key, "RUNNING")
                ckpt_mgr.check_memory_guard()
                ts = time.perf_counter()
                storage_url = (
                    f"sqlite:///{(ckpt_mgr.run_dir / 'optuna' / f'{sp.scenario_id}.db').as_posix()}"
                    if use_hpo
                    else None
                )
                if sp.mode_eval == "d1":
                    df_pred, trials_df = _run_d1_scenario(
                        sp, r, use_hpo=use_hpo, n_trials=n_trials,
                        storage_url=storage_url, proto_hash=hpo_proto_hash if use_hpo else expected_hash,
                        ckpt_mgr=ckpt_mgr, var_label=var,
                    )
                elif sp.mode_eval == "recursive":
                    df_pred, trials_df = _run_recursive_scenario(
                        sp, r, use_hpo=use_hpo, n_trials=n_trials,
                        storage_url=storage_url, proto_hash=hpo_proto_hash if use_hpo else expected_hash,
                        ckpt_mgr=ckpt_mgr, var_label=var,
                    )
                else:
                    df_pred, trials_df = _run_direct_scenario(
                        sp, r, use_hpo=use_hpo, n_trials=n_trials,
                        storage_url=storage_url, proto_hash=hpo_proto_hash if use_hpo else expected_hash,
                        ckpt_mgr=ckpt_mgr, var_label=var,
                    )
                if df_pred.empty:
                    ckpt_mgr.mark_unit(key, "FAILED", error="Escenario vacío")
                    raise RuntimeError(f"Escenario vacío durante ejecución definitiva: {key}")

                _atomic_to_csv(df_pred, out_csv)
                atomic_write_parquet(df_pred, frag_pq)
                if use_hpo and not trials_df.empty:
                    _atomic_to_csv(trials_df, hpo_dir / f"{sp.scenario_id}.csv")
                fhash = sha256_file(out_csv)
                scenario_times[key] = round(time.perf_counter() - ts, 3)
                scenario_hashes[key] = fhash
                ckpt_mgr.mark_unit(
                    key, "COMPLETED", artifact_path=out_csv, artifact_sha256=fhash, n_rows=len(df_pred)
                )
                del df_pred, trials_df
                gc.collect()

    manifest_path = target_dir / "manifest_ashr_oof.json"
    prev_vars: list[str] = []
    prev_times: dict[str, float] = {}
    prev_hashes: dict[str, str] = {}
    if manifest_path.exists():
        try:
            prev_m = json.loads(manifest_path.read_text(encoding="utf-8"))
            if prev_m.get("sha256_src_adaptive_structured_huber") == src_hash:
                prev_vars = list(prev_m.get("variantes_completadas", []))
                prev_times = dict(prev_m.get("tiempos_por_escenario_s", {}))
                prev_hashes = dict(prev_m.get("hashes_sha256_predicciones", {}))
        except (json.JSONDecodeError, OSError):
            pass

    all_vars = sorted(set(prev_vars) | set(run_variants))
    merged_times = {**prev_times, **scenario_times}
    merged_hashes = {**prev_hashes, **scenario_hashes}
    manifest = {
        "estatus": "OOF_DEFINITIVO_COMPLETO",
        "procedencia": "OOF_DEFINITIVO_REAL",
        "run_id": effective_run_id,
        "variante": variante,
        "variantes_completadas": all_vars,
        "n_trials_hpo": int(n_trials) if "hpo" in all_vars else 0,
        "sha256_src_adaptive_structured_huber": src_hash,
        "hpo_protocol_version": HPO_PROTOCOL_VERSION,
        "hpo_protocol_hash": hpo_proto_hash,
        "tiempo_total_s": round(time.perf_counter() - t0, 2),
        "tiempos_por_escenario_s": merged_times,
        "hashes_sha256_predicciones": merged_hashes,
    }
    _atomic_write_text(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False))
    ckpt_mgr.finalize_manifest({"hashes_sha256_predicciones": merged_hashes, "variantes_completadas": all_vars})
    return manifest

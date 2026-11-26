#!/usr/bin/env python3
"""Execute Fundamental Representation Equivalence Experiment (kWh_bruto vs. CF) v3.0.0.

Does NOT modify frozen `resultados/evaluacion_ashr_oof/`.
Writes all outputs to `resultados/experimento_equivalencia_v3/`.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ashr_oof_pipeline.config import SCENARIOS_28, ScenarioSpec
from src.ashr_oof_pipeline.evaluacion_estadistica import apply_holm
from src.ashr_oof_pipeline.oof_runner import load_official_benchmarks
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
from src.sm_cfhr.representation_benchmark import (
    FIXED_HPS_A,
    MODEL_NAMES_V3,
    fit_model_representation,
    predict_model_representation,
    select_best_cfg_cv3,
)

VERSION = "3.0.0-equivalencia-representacion"
SEED = 2026
N_WORKERS = 6

SCENARIOS_COND_A = [
    "D1_espaciotemporal_cinco_variables",
    "D1_intraplanta_cinco_variables",
    "D1_espaciotemporal_sin_meteorologia",
    "D1_intraplanta_sin_meteorologia",
    "RecGEFSv12_espaciotemporal_7d",
    "RecGEFSv12_espaciotemporal_28d",
    "RecGEFSv12_intraplanta_28d",
    "RecSinClima_espaciotemporal_28d",
    "DirectoSinClima_espaciotemporal_28d",
    "DirectoSinClima_intraplanta_28d",
]

SCENARIOS_COND_B = [
    "D1_espaciotemporal_cinco_variables",
    "D1_intraplanta_cinco_variables",
    "D1_espaciotemporal_sin_meteorologia",
    "D1_intraplanta_sin_meteorologia",
    "RecGEFSv12_espaciotemporal_28d",
    "DirectoSinClima_intraplanta_28d",
]

SCENARIO_MAP: dict[str, ScenarioSpec] = {sp.scenario_id: sp for sp in SCENARIOS_28}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def _eval_d1_scenario(sp: ScenarioSpec, condition: str) -> tuple[pd.DataFrame, dict[str, float]]:
    from scripts.evaluacion import produccion_intraplanta as daily

    panel = daily.build_panel(
        ROOT / "base_generacion_clima" / "base_planta_dia.csv",
        ROOT / "datos_pronostico" / "pronostico_diario.csv",
        sp.weather_on,
    )
    bench = load_official_benchmarks(sp, root=ROOT)
    pairs = [(m, r) for m in MODEL_NAMES_V3 for r in ("CF", "kWh_bruto")]
    times_s: dict[str, float] = {f"{m}__{r}": 0.0 for m, r in pairs}
    rows: list[dict[str, Any]] = []

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

            # MM30 baseline
            mm_vals = np.clip(
                test["media30_kwh"].to_numpy(float),
                0.0,
                test["cap_MW_D"].to_numpy(float) * 24000.0,
            )
            for r_t, val in zip(test.itertuples(index=False), mm_vals):
                rows.append({
                    "metodo": "MM30__control",
                    "modelo_base": "MM30",
                    "representacion": "control",
                    "bloque": blk,
                    "code": str(r_t.code),
                    "grupo": int(r_t.grupo),
                    "date": str(pd.Timestamp(r_t.date).date()),
                    "prediccion_kwh": float(val),
                    "y_kwh": float(r_t.y),
                })

            for m_name, rep in pairs:
                t0 = time.perf_counter()
                cfg = (
                    select_best_cfg_cv3(m_name, rep, train, sp, target_col="y", seed=SEED)
                    if condition == "B"
                    else FIXED_HPS_A[m_name]
                )
                fitted = fit_model_representation(
                    m_name, rep, train, sp.task, 1, sp.weather_on, horizon_direct=1, target_col="y", cfg=cfg, seed=SEED
                )
                preds = predict_model_representation(fitted, test, lead=1)
                times_s[f"{m_name}__{rep}"] += time.perf_counter() - t0
                lbl = f"{m_name}__{rep}"
                for r_t, val in zip(test.itertuples(index=False), preds):
                    rows.append({
                        "metodo": lbl,
                        "modelo_base": m_name,
                        "representacion": rep,
                        "bloque": blk,
                        "code": str(r_t.code),
                        "grupo": int(r_t.grupo),
                        "date": str(pd.Timestamp(r_t.date).date()),
                        "prediccion_kwh": float(val),
                        "y_kwh": float(r_t.y),
                    })
    return pd.DataFrame(rows), times_s


def _eval_direct_scenario(sp: ScenarioSpec, condition: str) -> tuple[pd.DataFrame, dict[str, float]]:
    from scripts.evaluacion import produccion_multihorizonte_directo as mhd

    panel, _ = mhd.build_direct_panel(ROOT / "base_generacion_clima" / "base_planta_dia.csv", sp.horizon_days)
    bench = load_official_benchmarks(sp, root=ROOT)
    origins = sorted(pd.Timestamp(o) for o in bench["origen"].unique())
    units = sorted(panel["grupo"].unique()) if sp.task == "espaciotemporal" else sorted(panel["code"].unique())
    pairs = [(m, r) for m in MODEL_NAMES_V3 for r in ("CF", "kWh_bruto")]
    times_s: dict[str, float] = {f"{m}__{r}": 0.0 for m, r in pairs}
    rows: list[dict[str, Any]] = []

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

            mm_vals = np.clip(
                query["media30_kwh"].to_numpy(dtype=float) * float(sp.horizon_days),
                0.0,
                query["cap_MW_D"].to_numpy(dtype=float) * 24000.0 * float(sp.horizon_days),
            )
            for r_q, val in zip(query.itertuples(index=False), mm_vals):
                rows.append({
                    "metodo": "MM30__control",
                    "modelo_base": "MM30",
                    "representacion": "control",
                    "tarea": sp.task,
                    "code": str(r_q.code),
                    "grupo": int(r_q.grupo),
                    "origen": str(origin.date()),
                    "prediccion_total_kwh": float(val),
                    "y_total_kwh": float(r_q.y_H),
                })

            for m_name, rep in pairs:
                t0 = time.perf_counter()
                cfg = (
                    select_best_cfg_cv3(m_name, rep, train, sp, target_col="y_H", seed=SEED)
                    if condition == "B"
                    else FIXED_HPS_A[m_name]
                )
                fitted = fit_model_representation(
                    m_name, rep, train, sp.task, sp.horizon_days, False,
                    horizon_direct=sp.horizon_days, target_col="y_H", cfg=cfg, seed=SEED,
                )
                preds = predict_model_representation(fitted, query, lead=1)
                times_s[f"{m_name}__{rep}"] += time.perf_counter() - t0
                lbl = f"{m_name}__{rep}"
                for r_q, val in zip(query.itertuples(index=False), preds):
                    rows.append({
                        "metodo": lbl,
                        "modelo_base": m_name,
                        "representacion": rep,
                        "tarea": sp.task,
                        "code": str(r_q.code),
                        "grupo": int(r_q.grupo),
                        "origen": str(origin.date()),
                        "prediccion_total_kwh": float(val),
                        "y_total_kwh": float(r_q.y_H),
                    })
    return pd.DataFrame(rows), times_s


def _rec_unit_worker(args: tuple[str, str, Any, str]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    sc, condition, u_val, _ = args
    sp = SCENARIO_MAP[sc]
    from scripts.evaluacion import produccion_multihorizonte as mh
    from scripts.evaluacion import produccion_multihorizonte_clima as mhc

    base_path = ROOT / "base_generacion_clima" / "base_planta_dia.csv"
    gefs_path = ROOT / "datos_pronostico_multihorizonte" / "pronostico_28d_gefs.csv"
    if sp.weather_on:
        panel, raw_plants, weather_store = mhc.load_generation_weather_panel(base_path, gefs_path)
        plan_fn = lambda origs: mhc._plan_unit_origins(sp.task, u_val, origs, sp.horizon_days, panel, raw_plants, weather_store)
    else:
        panel, raw_plants = mh.load_generation_panel(base_path)
        weather_store = None
        plan_fn = lambda origs: mh._plan_unit_origins(sp.task, u_val, origs, sp.horizon_days, panel, raw_plants)

    bench = load_official_benchmarks(sp, root=ROOT)
    origins = sorted(pd.Timestamp(o) for o in bench["origen"].unique())
    _, eval_specs, _ = plan_fn(origins)

    pairs = [(m, r) for m in MODEL_NAMES_V3 for r in ("CF", "kWh_bruto")]
    times_s: dict[str, float] = {f"{m}__{r}": 0.0 for m, r in pairs}
    rows: list[dict[str, Any]] = []

    for spec in eval_specs:
        origin = spec["origin"]
        origin_ts = pd.Timestamp(origin)
        pred_infos = [raw_plants[c] for c in spec["pred_codes"]]
        train = (
            panel[(panel["code"] == str(u_val)) & (panel["date"] <= origin)].copy()
            if sp.task == "intraplanta"
            else panel[(panel["grupo"] != int(u_val)) & (panel["date"] <= origin)].copy()
        )
        if train.empty:
            continue

        # MM30 rollout
        st_mm = (
            [mhc.PlantRecursiveWeatherState(p, origin_ts, sp.horizon_days, weather_store) for p in pred_infos]
            if sp.weather_on
            else [mh.PlantRecursiveState(p, origin_ts, sp.horizon_days) for p in pred_infos]
        )
        for lead in range(1, sp.horizon_days + 1):
            t_date = origin_ts + pd.Timedelta(days=lead)
            q_df = pd.DataFrame([s.next_features(t_date) for s in st_mm])
            yhat = np.clip(q_df["media30_kwh"].to_numpy(float), 0.0, q_df["cap_MW_D"].to_numpy(float) * 24000.0)
            for i, info in enumerate(pred_infos):
                n_hist = (origin_ts - info["min_date"]).days + 1
                idx_t = n_hist + lead - 1
                ay = float(info["kwh_valid"][idx_t]) if 0 <= idx_t < len(info["kwh_valid"]) else float("nan")
                if np.isfinite(ay):
                    rows.append({
                        "metodo": "MM30__control",
                        "modelo_base": "MM30",
                        "representacion": "control",
                        "tarea": sp.task,
                        "code": info["code"],
                        "grupo": int(info["grupo"]),
                        "origen": str(origin_ts.date()),
                        "lead_dias": lead,
                        "prediccion_kwh": float(yhat[i]),
                        "y_kwh": ay,
                    })
                st_mm[i].step_update(float(yhat[i]))

        for m_name, rep in pairs:
            t0 = time.perf_counter()
            cfg = (
                select_best_cfg_cv3(m_name, rep, train, sp, target_col="y", seed=SEED)
                if condition == "B"
                else FIXED_HPS_A[m_name]
            )
            fitted = fit_model_representation(
                m_name, rep, train, sp.task, sp.horizon_days, sp.weather_on, horizon_direct=1, target_col="y", cfg=cfg, seed=SEED
            )
            states = (
                [mhc.PlantRecursiveWeatherState(p, origin_ts, sp.horizon_days, weather_store) for p in pred_infos]
                if sp.weather_on
                else [mh.PlantRecursiveState(p, origin_ts, sp.horizon_days) for p in pred_infos]
            )
            lbl = f"{m_name}__{rep}"
            for lead in range(1, sp.horizon_days + 1):
                t_date = origin_ts + pd.Timedelta(days=lead)
                q_df = pd.DataFrame([s.next_features(t_date) for s in states])
                yhat = predict_model_representation(fitted, q_df, lead=lead)
                for i, info in enumerate(pred_infos):
                    n_hist = (origin_ts - info["min_date"]).days + 1
                    idx_t = n_hist + lead - 1
                    ay = float(info["kwh_valid"][idx_t]) if 0 <= idx_t < len(info["kwh_valid"]) else float("nan")
                    if np.isfinite(ay):
                        rows.append({
                            "metodo": lbl,
                            "modelo_base": m_name,
                            "representacion": rep,
                            "tarea": sp.task,
                            "code": info["code"],
                            "grupo": int(info["grupo"]),
                            "origen": str(origin_ts.date()),
                            "lead_dias": lead,
                            "prediccion_kwh": float(yhat[i]),
                            "y_kwh": ay,
                        })
                    states[i].step_update(float(yhat[i]))
            times_s[lbl] += time.perf_counter() - t0

    return rows, times_s


def _eval_rec_scenario(sp: ScenarioSpec, condition: str) -> tuple[pd.DataFrame, dict[str, float]]:
    from scripts.evaluacion import produccion_multihorizonte as mh

    base_path = ROOT / "base_generacion_clima" / "base_planta_dia.csv"
    panel, _ = mh.load_generation_panel(base_path)
    units = sorted(panel["grupo"].unique()) if sp.task == "espaciotemporal" else sorted(panel["code"].unique())
    tasks = [(sp.scenario_id, condition, u, sp.task) for u in units]

    all_rows: list[dict[str, Any]] = []
    tot_times: dict[str, float] = {f"{m}__{r}": 0.0 for m in MODEL_NAMES_V3 for r in ("CF", "kWh_bruto")}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for r_list, t_map in ex.map(_rec_unit_worker, tasks):
            all_rows.extend(r_list)
            for k, v in t_map.items():
                tot_times[k] += v
    return pd.DataFrame(all_rows), tot_times


def _run_condition(
    condition: str,
    scenario_ids: list[str],
    d1_cap_df: pd.DataFrame,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    plant_rows: list[dict[str, Any]] = []
    pred_dir = out_dir / "predicciones_equivalencia"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for sc in scenario_ids:
        sp = SCENARIO_MAP[sc]
        print(f"[Cond {condition}] Running {sc} ...", flush=True)
        t_sc = time.perf_counter()
        if sp.mode_eval == "d1":
            raw_df, times_s = _eval_d1_scenario(sp, condition)
        elif sp.mode_eval == "direct":
            raw_df, times_s = _eval_direct_scenario(sp, condition)
        else:
            raw_df, times_s = _eval_rec_scenario(sp, condition)

        bench_raw = load_official_benchmarks(sp, root=ROOT, exclude_cf_brh=True)
        bench_keys = bench_raw[list(sp.join_keys)].drop_duplicates()
        raw_df = raw_df.merge(bench_keys, on=list(sp.join_keys), how="inner")

        # Save predictions for Condition A for P2/Phase 5 plotting
        if condition == "A":
            raw_df.to_csv(pred_dir / f"{sc}_equiv_A.csv", index=False)

        cf_df = attach_capacity_and_cf_errors(raw_df, sp, bench_raw, d1_cap_df=d1_cap_df)

        # Compute metrics per (metodo, modelo_base, representacion)
        sc_met = []
        for (lbl, m_base, rep), grp in cf_df.groupby(["metodo", "modelo_base", "representacion"]):
            mm = compute_audited_metrics_from_df(grp, sp)
            sc_met.append({
                "condicion": condition,
                "escenario": sc,
                "suite": sp.suite,
                "task": sp.task,
                "horizon_days": sp.horizon_days,
                "weather_on": sp.weather_on,
                "mode_eval": sp.mode_eval,
                "metodo": lbl,
                "modelo_base": m_base,
                "representacion": rep,
                "n_obs": int(len(grp)),
                "tiempo_s": round(times_s.get(lbl, 0.0), 3),
                **mm,
            })
            if condition == "A":
                pg = grp.groupby(["code", "grupo"], as_index=False).agg(
                    mae_kwh=("_ae_kwh", "mean"),
                    mae_cf=("_ae_cf", "mean"),
                    cap_MW=("cap_MW", "mean"),
                )
                pg["escenario"] = sc
                pg["metodo"] = lbl
                pg["modelo_base"] = m_base
                pg["representacion"] = rep
                plant_rows.extend(pg.to_dict("records"))

        sc_df = pd.DataFrame(sc_met)
        sc_df["rango_kWh"] = sc_df["MAE_macro_kWh"].rank(method="min").astype(int)
        sc_df["rango_CF"] = sc_df["MAE_CF_macro"].rank(method="min").astype(int)

        sm_row = sc_df[sc_df["metodo"] == "SM-CFHR__CF"].iloc[0]
        cf_others = sc_df[(sc_df["representacion"] == "CF") & (sc_df["modelo_base"] != "SM-CFHR")].sort_values("MAE_CF_macro")
        kwh_others = sc_df[(sc_df["representacion"] == "kWh_bruto") & (sc_df["modelo_base"] != "SM-CFHR")].sort_values("MAE_macro_kWh")
        best_cf_name = str(cf_others.iloc[0]["metodo"])
        best_cf_val = float(cf_others.iloc[0]["MAE_CF_macro"])
        best_cf_kwh = float(cf_others.iloc[0]["MAE_macro_kWh"])
        best_kwh_name = str(kwh_others.iloc[0]["metodo"])
        best_kwh_val = float(kwh_others.iloc[0]["MAE_macro_kWh"])

        sc_df["mejor_alt_CF"] = best_cf_name
        sc_df["mejor_alt_kWh"] = best_kwh_name
        sc_df["delta_pct_SMCFHR_vs_mejor_CF_en_CF"] = 100.0 * (sm_row["MAE_CF_macro"] / best_cf_val - 1.0)
        sc_df["delta_pct_SMCFHR_vs_mejor_CF_en_kWh"] = 100.0 * (sm_row["MAE_macro_kWh"] / best_cf_kwh - 1.0)
        sc_df["delta_pct_SMCFHR_vs_mejor_kWh"] = 100.0 * (sm_row["MAE_macro_kWh"] / best_kwh_val - 1.0)
        summary_rows.extend(sc_df.to_dict("records"))

        # Paired inference: SM-CFHR__CF vs each baseline__CF (and SM-CFHR__kWh_bruto)
        time_col = "date" if sp.mode_eval == "d1" else "origen"
        step_info = infer_series_step_and_overlap(cf_df[time_col].unique(), sp.horizon_days)
        h_ov = int(step_info["h_overlap"])
        df_sm = cf_df[cf_df["metodo"] == "SM-CFHR__CF"]
        for comp in [f"{m}__CF" for m in MODEL_NAMES_V3 if m != "SM-CFHR"] + ["SM-CFHR__kWh_bruto"]:
            df_c = cf_df[cf_df["metodo"] == comp]
            # 1) On kWh
            p_kwh = build_macro_aligned_panel(df_sm, df_c, sp.join_keys, time_col, sp.pred_col, sp.target_col)
            dm_kwh = dm_hln_hac_v2(p_kwh["z_macro_series"], h_overlap=h_ov)
            bm_kwh = macro_synchronized_block_bootstrap(p_kwh["panel_sum"], p_kwh["panel_cnt"], B=1500, seed=SEED)
            # 2) On CF
            df_sm_cf = df_sm.assign(_pred_cf=df_sm["_ae_cf"], _zero=0.0)
            df_c_cf = df_c.assign(_pred_cf=df_c["_ae_cf"], _zero=0.0)
            p_cf = build_macro_aligned_panel(df_sm_cf, df_c_cf, sp.join_keys, time_col, "_pred_cf", "_zero")
            dm_cf = dm_hln_hac_v2(p_cf["z_macro_series"], h_overlap=h_ov)
            bm_cf = macro_synchronized_block_bootstrap(p_cf["panel_sum"], p_cf["panel_cnt"], B=1500, seed=SEED)
            paired_rows.append({
                "condicion": condition,
                "escenario": sc,
                "comparador": comp,
                "T": p_kwh["n_times"],
                "h_overlap": h_ov,
                "T_eff_kWh": dm_kwh["T_eff"],
                "diff_MAE_macro_kWh": p_kwh["diff_mae_macro"],
                "ci95_lo_kWh": bm_kwh["ci_lo"],
                "ci95_hi_kWh": bm_kwh["ci_hi"],
                "dm_hln_kWh": dm_kwh["dm_hln"],
                "p_dm_kWh": dm_kwh["p_dm_hln"],
                "T_eff_CF": dm_cf["T_eff"],
                "diff_MAE_CF_macro": p_cf["diff_mae_macro"],
                "ci95_lo_CF": bm_cf["ci_lo"],
                "ci95_hi_CF": bm_cf["ci_hi"],
                "dm_hln_CF": dm_cf["dm_hln"],
                "p_dm_CF": dm_cf["p_dm_hln"],
            })
        print(
            f"  done in {time.perf_counter()-t_sc:.1f}s | SM-CFHR rank CF={sm_row['rango_CF']} "
            f"vs {best_cf_name} ({sc_df['delta_pct_SMCFHR_vs_mejor_CF_en_CF'].iloc[0]:+.2f}% CF)",
            flush=True,
        )

    df_pair = pd.DataFrame(paired_rows)
    if not df_pair.empty:
        df_pair["p_holm_escenario_CF"] = np.nan
        df_pair["p_holm_escenario_kWh"] = np.nan
        for sc_v, idx in df_pair.groupby("escenario", sort=False).groups.items():
            sub_idx = df_pair.loc[idx].index[df_pair.loc[idx, "comparador"] != "SM-CFHR__kWh_bruto"]
            df_pair.loc[sub_idx, "p_holm_escenario_CF"] = apply_holm(df_pair.loc[sub_idx, "p_dm_CF"].fillna(1.0).to_numpy(float))
            df_pair.loc[sub_idx, "p_holm_escenario_kWh"] = apply_holm(df_pair.loc[sub_idx, "p_dm_kWh"].fillna(1.0).to_numpy(float))
    return pd.DataFrame(summary_rows), df_pair, pd.DataFrame(plant_rows)


def main() -> None:
    t0 = time.perf_counter()
    out_dir = ROOT / "resultados" / "experimento_equivalencia_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    d1_cap_df = load_d1_plant_capacity_table(ROOT)

    df_a, pair_a, plant_a = _run_condition("A", SCENARIOS_COND_A, d1_cap_df, out_dir)
    df_b, pair_b, _ = _run_condition("B", SCENARIOS_COND_B, d1_cap_df, out_dir)

    df_a.to_csv(out_dir / "resumen_equivalencia_condicion_A.csv", index=False)
    df_b.to_csv(out_dir / "resumen_equivalencia_condicion_B.csv", index=False)
    pd.concat([pair_a, pair_b], ignore_index=True).to_csv(
        out_dir / "inferencia_pareada_smcfhr_vs_cf_baselines.csv", index=False
    )
    plant_a.to_csv(out_dir / "resultados_por_planta_equivalencia.csv", index=False)

    meta = {
        "version": VERSION,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed": SEED,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "scenarios_cond_A": SCENARIOS_COND_A,
        "scenarios_cond_B": SCENARIOS_COND_B,
        "models": MODEL_NAMES_V3,
    }
    (out_dir / "metadatos_equivalencia_v3.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    man_rows = []
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.name != "MANIFIESTO_EQUIVALENCIA_V3.csv":
            man_rows.append({"ruta_relativa": p.relative_to(out_dir).as_posix(), "tamano_bytes": p.stat().st_size, "sha256": _sha256(p)})
    with (out_dir / "MANIFIESTO_EQUIVALENCIA_V3.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ruta_relativa", "tamano_bytes", "sha256"], lineterminator="\n")
        w.writeheader()
        w.writerows(man_rows)
    print(f"Equivalence experiment v3 complete in {meta['elapsed_s']:.1f}s ({len(man_rows)} artifacts).")


if __name__ == "__main__":
    main()

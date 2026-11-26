#!/usr/bin/env python3
"""Independent post-review scientific recalculation v2 (no retraining, read-only OOF).

Generates `resultados/inferencia_revision_independiente_v2/` covering:
01_referencia_original/
02_estimandos_y_dm_hln/
03_spa_y_mcs_reconciliados/
04_sensibilidad/
05_metricas_corregidas/
06_exploratorio_p2/
plus `metadatos_revision_v2.json` and `MANIFIESTO_REVISION_V2.csv`.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch.bootstrap import MCS as ArchMCS
from arch.bootstrap import SPA as ArchSPA

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ashr_oof_pipeline.config import SCENARIOS_28
from src.ashr_oof_pipeline.evaluacion_estadistica import apply_holm, build_23_model_ranking
from src.ashr_oof_pipeline.oof_runner import align_predictions, load_official_benchmarks
from src.ashr_oof_pipeline.resultados_loader import LIBS_SET
from src.sm_cfhr.inference_v2 import (
    build_macro_aligned_panel,
    build_macro_loss_matrix,
    classify_teff_reliability,
    default_block_length,
    dm_hln_hac_v2,
    hansen_politis_romano_variance,
    infer_series_step_and_overlap,
    macro_synchronized_block_bootstrap,
    mcs_t_max_v2,
    spa_test_v2,
)
from src.sm_cfhr.metrics_audit import (
    attach_capacity_and_cf_errors,
    build_metric_definitions_catalog,
    compute_audited_metrics_from_df,
    detect_capacity_changes,
    load_d1_plant_capacity_table,
)

VERSION = "2.0.0-revision-independiente"
SEED = 2026
B_SPA_MCS = 5000
B_BOOT = 2000


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _acf_k(arr: np.ndarray, k: int) -> float:
    if len(arr) <= k + 2 or np.std(arr) <= 1e-12:
        return np.nan
    return float(np.corrcoef(arr[:-k], arr[k:])[0, 1])


def main() -> None:
    oof_dir = ROOT / "resultados" / "evaluacion_ashr_oof"
    out_dir = ROOT / "resultados" / "inferencia_revision_independiente_v2"
    d01 = out_dir / "01_referencia_original"
    d02 = out_dir / "02_estimandos_y_dm_hln"
    d03 = out_dir / "03_spa_y_mcs_reconciliados"
    d04 = out_dir / "04_sensibilidad"
    d05 = out_dir / "05_metricas_corregidas"
    d06 = out_dir / "06_exploratorio_p2"
    for d in (d01, d02, d03, d04, d05, d06):
        d.mkdir(parents=True, exist_ok=True)

    # 01: Copy frozen reference tables (read-only)
    ref_files = [
        oof_dir / "tablas_paper" / "tabla_28_escenarios.csv",
        oof_dir / "inferencia" / "tabla_inferencia.csv",
        oof_dir / "inferencia" / "tabla_mcs.csv",
        oof_dir / "inferencia" / "mcs_bibliotecas_only.csv",
        oof_dir / "inferencia" / "tabla_spa.csv",
        oof_dir / "conclusiones" / "veredicto_r3.csv",
    ]
    for rf in ref_files:
        shutil.copy2(rf, d01 / rf.name)

    d1_cap_df = load_d1_plant_capacity_table(ROOT)
    cap_changes_df = detect_capacity_changes(d1_cap_df)
    cap_changes_df.to_csv(d05 / "plantas_cambio_capacidad.csv", index=False)
    build_metric_definitions_catalog().to_csv(d05 / "catalogo_definiciones_metricas.csv", index=False)

    reconcil_rows: list[dict[str, Any]] = []
    all9_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    spa_rows: list[dict[str, Any]] = []
    mcs_rows: list[dict[str, Any]] = []
    sens_teff_rows_store: list[dict[str, Any]] = []
    sens_blk_hac_rows: list[dict[str, Any]] = []
    logo_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    factorial_rows: list[dict[str, Any]] = []

    rep_sens = {
        "D1_espaciotemporal_cinco_variables",
        "D1_intraplanta_sin_meteorologia",
        "RecSinClima_intraplanta_14d",
        "DirectoSinClima_intraplanta_7d",
        "RecGEFSv12_espaciotemporal_28d",
    }

    # Store data for P2 plots
    p2_d1_df: pd.DataFrame | None = None
    p2_rec_df: pd.DataFrame | None = None
    p2_dir_df: pd.DataFrame | None = None

    for sp in SCENARIOS_28:
        sc = sp.scenario_id
        time_col = "date" if sp.mode_eval == "d1" else "origen"

        ashr_raw = pd.read_csv(oof_dir / "predicciones" / f"{sc}.csv")
        ashr_raw["code"] = ashr_raw["code"].astype(str)
        hpo_raw = pd.read_csv(oof_dir / "predicciones" / f"{sc}_hpo.csv")
        hpo_raw["code"] = hpo_raw["code"].astype(str)

        bench_raw = load_official_benchmarks(sp, root=ROOT, exclude_cf_brh=True)
        bench_keys = bench_raw[list(sp.join_keys)].drop_duplicates()

        ashr_df = ashr_raw.merge(bench_keys, on=list(sp.join_keys), how="inner")
        hpo_df = hpo_raw.merge(bench_keys, on=list(sp.join_keys), how="inner")
        bench_al = align_predictions(ashr_df, bench_raw, sp.join_keys)

        # Attach CF capacity
        ashr_cf = attach_capacity_and_cf_errors(ashr_df, sp, bench_raw, d1_cap_df=d1_cap_df)
        hpo_cf = attach_capacity_and_cf_errors(hpo_df, sp, bench_raw, d1_cap_df=d1_cap_df)
        bench_cf = attach_capacity_and_cf_errors(bench_al, sp, bench_raw, d1_cap_df=d1_cap_df)

        # Metrics across all 23 models in kWh and CF
        m_fixed = compute_audited_metrics_from_df(ashr_cf, sp)
        m_hpo = compute_audited_metrics_from_df(hpo_cf, sp)
        all_kwh_list = [{"metodo": "ASHR", "MAE_macro": m_fixed["MAE_macro_kWh"]}]
        all_cf_list = [{"metodo": "ASHR", "MAE_macro": m_fixed["MAE_CF_macro"]}]
        bench_metrics_by_m: dict[str, dict[str, float]] = {}
        for m_name, grp in bench_cf.groupby("metodo"):
            mm = compute_audited_metrics_from_df(grp, sp)
            bench_metrics_by_m[str(m_name)] = mm
            all_kwh_list.append({"metodo": str(m_name), "MAE_macro": mm["MAE_macro_kWh"]})
            all_cf_list.append({"metodo": str(m_name), "MAE_macro": mm["MAE_CF_macro"]})

        rank_kwh_df = build_23_model_ranking(pd.DataFrame(all_kwh_list))
        rank_cf_df = build_23_model_ranking(pd.DataFrame(all_cf_list))
        rango_kwh = int(rank_kwh_df.loc[rank_kwh_df["metodo"] == "ASHR", "rango"].iloc[0])
        rango_cf = int(rank_cf_df.loc[rank_cf_df["metodo"] == "ASHR", "rango"].iloc[0])

        libs_sorted_kwh = sorted(
            [(m, bench_metrics_by_m[m]["MAE_macro_kWh"]) for m in LIBS_SET if m in bench_metrics_by_m],
            key=lambda x: x[1],
        )
        best_lib_kwh_name, _ = libs_sorted_kwh[0]
        libs_sorted_cf = sorted(
            [(m, bench_metrics_by_m[m]["MAE_CF_macro"]) for m in LIBS_SET if m in bench_metrics_by_m],
            key=lambda x: x[1],
        )
        best_lib_cf_name, _ = libs_sorted_cf[0]
        m_best_lib = bench_metrics_by_m[best_lib_kwh_name]
        m_mm30 = bench_metrics_by_m["Media móvil de 30 días"]

        skill_mae_fixed = 100.0 * (1.0 - m_fixed["MAE_macro_kWh"] / m_mm30["MAE_macro_kWh"])
        skill_rmse_fixed = 100.0 * (1.0 - m_fixed["RMSE_micro_kWh"] / m_mm30["RMSE_micro_kWh"])

        metrics_rows.append({
            "escenario": sc,
            "suite": sp.suite,
            "task": sp.task,
            "horizon_days": sp.horizon_days,
            "weather_on": sp.weather_on,
            "mode_eval": sp.mode_eval,
            "rango_en_23_kWh": rango_kwh,
            "rango_en_23_CF": rango_cf,
            "cambio_rango_kWh_vs_CF": rango_cf - rango_kwh,
            "mejor_bib_kWh": best_lib_kwh_name,
            "mejor_bib_CF": best_lib_cf_name,
            "MAE_macro_kWh_Fixed": m_fixed["MAE_macro_kWh"],
            "MAE_macro_kWh_por_dia_Fixed": m_fixed["MAE_macro_kWh_por_dia"],
            "MAE_micro_kWh_Fixed": m_fixed["MAE_micro_kWh"],
            "MAE_CF_macro_Fixed": m_fixed["MAE_CF_macro"],
            "MAE_CF_micro_Fixed": m_fixed["MAE_CF_micro"],
            "nMAE_rel_generacion_media_planta_Fixed": m_fixed["nMAE_rel_generacion_media_planta"],
            "MAE_macro_kWh_HPO": m_hpo["MAE_macro_kWh"],
            "MAE_CF_macro_HPO": m_hpo["MAE_CF_macro"],
            "MAE_macro_kWh_MejorBib": m_best_lib["MAE_macro_kWh"],
            "MAE_CF_macro_MejorBib": m_best_lib["MAE_CF_macro"],
            "diff_pct_kWh_vs_MejorBib": 100.0 * (m_fixed["MAE_macro_kWh"] / m_best_lib["MAE_macro_kWh"] - 1.0),
            "diff_pct_CF_vs_MejorBib": 100.0 * (m_fixed["MAE_CF_macro"] / m_best_lib["MAE_CF_macro"] - 1.0),
            "MAE_macro_kWh_MM30": m_mm30["MAE_macro_kWh"],
            "MAE_CF_macro_MM30": m_mm30["MAE_CF_macro"],
            "Skill_MAE_vs_MM30_pct": skill_mae_fixed,
            "Skill_RMSE_vs_MM30_pct": skill_rmse_fixed,
        })

        # Step & overlap audit
        step_info = infer_series_step_and_overlap(ashr_df[time_col].unique(), sp.horizon_days)
        h_ov = int(step_info["h_overlap"])
        ts_sorted = pd.to_datetime(pd.Index(ashr_df[time_col].unique())).sort_values()

        # Reconcile estimands: vs Mejor_Bib (retrospective), vs MM30 (pre-specified), vs HPO
        for ref_tipo, ref_name, df_ref in [
            ("Mejor_Bib_Retrospectivo", best_lib_kwh_name, bench_al[bench_al["metodo"] == best_lib_kwh_name]),
            ("MM30_Preespecificado", "Media móvil de 30 días", bench_al[bench_al["metodo"] == "Media móvil de 30 días"]),
            ("SM_CFHR_HPO_Preespecificado", "SM-CFHR-HPO", hpo_df),
        ]:
            panel = build_macro_aligned_panel(
                ashr_df, df_ref, sp.join_keys, time_col, sp.pred_col, sp.target_col
            )
            z_mac = panel["z_macro_series"]
            d_ori = panel["d_origin_series"]
            dm_mac = dm_hln_hac_v2(z_mac, h_overlap=h_ov, lag=None)
            dm_ori = dm_hln_hac_v2(d_ori, h_overlap=h_ov, lag=None)
            boot_mac = macro_synchronized_block_bootstrap(
                panel["panel_sum"], panel["panel_cnt"], mean_block=None, B=B_BOOT, seed=SEED
            )
            reconcil_rows.append({
                "escenario": sc,
                "suite": sp.suite,
                "task": sp.task,
                "horizon_days": sp.horizon_days,
                "tipo_referencia": ref_tipo,
                "modelo_referencia": ref_name,
                "n_obs": panel["n_obs"],
                "n_plants": panel["n_plants"],
                "T": panel["n_times"],
                "plants_per_origin_min": panel["plants_per_origin_min"],
                "plants_per_origin_med": panel["plants_per_origin_med"],
                "plants_per_origin_max": panel["plants_per_origin_max"],
                "origins_per_plant_min": panel["origins_per_plant_min"],
                "origins_per_plant_max": panel["origins_per_plant_max"],
                "imbalance_ratio_obs": round(panel["imbalance_ratio_obs"], 2),
                "diff_MAE_macro": panel["diff_mae_macro"],
                "diff_MAE_micro": panel["diff_mae_micro"],
                "diff_temporal_origen": panel["diff_temporal_origin"],
                "sign_flip_macro_vs_origin": panel["sign_flip_macro_vs_origin"],
                "h_overlap": h_ov,
                "L_hac": dm_mac["L_hac"],
                "T_eff_macro": dm_mac["T_eff"],
                "T_eff_origen": dm_ori["T_eff"],
                "nivel_fiabilidad_teff": classify_teff_reliability(dm_mac["T_eff"], h_ov),
                "dm_hln_macro_v2": dm_mac["dm_hln"],
                "p_dm_macro_v2": dm_mac["p_dm_hln"],
                "degenerate_case_macro": dm_mac["degenerate_case"],
                "boot_macro_ci95_lo": boot_mac["ci_lo"],
                "boot_macro_ci95_hi": boot_mac["ci_hi"],
                "boot_macro_se": boot_mac["se_boot"],
                "p_boot_macro_two_sided": boot_mac["p_boot_two_sided"],
                "boot_mean_block": boot_mac["mean_block"],
                "pct_draws_all_plants_present": boot_mac["pct_draws_all_plants_present"],
                "dm_hln_origen_v2": dm_ori["dm_hln"],
                "p_dm_origen_v2": dm_ori["p_dm_hln"],
            })

            if ref_tipo == "Mejor_Bib_Retrospectivo":
                overlap_rows.append({
                    "escenario": sc,
                    "horizon_days": sp.horizon_days,
                    "T": panel["n_times"],
                    "fecha_inicio": str(ts_sorted[0].date()),
                    "fecha_fin": str(ts_sorted[-1].date()),
                    "min_step_days": step_info["min_step_days"],
                    "median_step_days": step_info["median_step_days"],
                    "max_step_days": step_info["max_step_days"],
                    "h_overlap_real": h_ov,
                    "solapamiento_temporal": step_info["overlapping"],
                    "L_hac": dm_mac["L_hac"],
                    "acf1_macro": _acf_k(z_mac, 1),
                    "acf2_macro": _acf_k(z_mac, 2),
                    "acf3_macro": _acf_k(z_mac, 3),
                    "T_eff_macro": dm_mac["T_eff"],
                })
                sens_teff_rows_store.append({
                    "escenario": sc,
                    "suite": sp.suite,
                    "horizon_days": sp.horizon_days,
                    "h_overlap": h_ov,
                    "T_eff_macro": dm_mac["T_eff"],
                    "T_eff_origen": dm_ori["T_eff"],
                    "p_dm_macro_v2": dm_mac["p_dm_hln"],
                    "p_dm_origen_v2": dm_ori["p_dm_hln"],
                    "diff_macro": panel["diff_mae_macro"],
                    "diff_origen": panel["diff_temporal_origin"],
                })
                if sc in rep_sens:
                    for blk_s in (1.0, 2.0, 3.0, 4.0, 7.0):
                        if blk_s > panel["n_times"] / 2.0:
                            continue
                        bm_s = macro_synchronized_block_bootstrap(
                            panel["panel_sum"], panel["panel_cnt"], mean_block=blk_s, B=1500, seed=SEED
                        )
                        sens_blk_hac_rows.append({
                            "escenario": sc,
                            "tipo": "BOOTSTRAP_SINCRONIZADO",
                            "parametro": f"block={blk_s}",
                            "valor": bm_s["se_boot"],
                            "ci_lo": bm_s["ci_lo"],
                            "ci_hi": bm_s["ci_hi"],
                            "p_val": bm_s["p_boot_two_sided"],
                        })
                    for lag_s in (0, 1, 2, 3, 4, None):
                        dm_s = dm_hln_hac_v2(z_mac, h_overlap=h_ov, lag=lag_s)
                        sens_blk_hac_rows.append({
                            "escenario": sc,
                            "tipo": "HAC_LAG",
                            "parametro": f"lag={'auto' if lag_s is None else lag_s}",
                            "valor": dm_s["T_eff"],
                            "ci_lo": dm_s["L_hac"],
                            "ci_hi": dm_s["dm_hln"],
                            "p_val": dm_s["p_dm_hln"],
                        })

                # Leave-one-cluster-out (G=7) sensitivity on MAE_macro
                groups = sorted(ashr_df["grupo"].unique())
                for g_out in groups:
                    sub_a = ashr_df[ashr_df["grupo"] != g_out]
                    sub_b = df_ref[df_ref["grupo"] != g_out]
                    p_g = build_macro_aligned_panel(sub_a, sub_b, sp.join_keys, time_col, sp.pred_col, sp.target_col)
                    dm_g = dm_hln_hac_v2(p_g["z_macro_series"], h_overlap=h_ov, lag=None)
                    logo_rows.append({
                        "escenario": sc,
                        "grupo_excluido": g_out,
                        "n_plants_restantes": p_g["n_plants"],
                        "diff_MAE_macro": p_g["diff_mae_macro"],
                        "dm_hln_macro_v2": dm_g["dm_hln"],
                        "p_dm_macro_v2": dm_g["p_dm_hln"],
                    })

        # All 9 libraries paired on MAE_macro
        for lib_name in sorted(LIBS_SET):
            df_lib = bench_al[bench_al["metodo"] == lib_name]
            if df_lib.empty:
                continue
            p_l = build_macro_aligned_panel(ashr_df, df_lib, sp.join_keys, time_col, sp.pred_col, sp.target_col)
            dm_l = dm_hln_hac_v2(p_l["z_macro_series"], h_overlap=h_ov, lag=None)
            all9_rows.append({
                "escenario": sc,
                "suite": sp.suite,
                "task": sp.task,
                "horizon_days": sp.horizon_days,
                "biblioteca": lib_name,
                "es_mejor_bib_retrospectivo": lib_name == best_lib_kwh_name,
                "diff_MAE_macro": p_l["diff_mae_macro"],
                "T": p_l["n_times"],
                "T_eff_macro": dm_l["T_eff"],
                "nivel_fiabilidad_teff": classify_teff_reliability(dm_l["T_eff"], h_ov),
                "dm_hln_macro_v2": dm_l["dm_hln"],
                "p_dm_macro_v2": dm_l["p_dm_hln"],
            })

        # --- SPA & MCS on both MAE_macro and MAE_origen loss matrices ---
        macro_mat_df, origin_mat_df = build_macro_loss_matrix(
            ashr_df, bench_al, sp.join_keys, time_col, sp.pred_col, sp.target_col
        )
        T_mat = len(macro_mat_df)
        blk = default_block_length(T_mat)
        blk_int = int(round(blk))
        cols23 = list(macro_mat_df.columns)
        bib_cols = ["ASHR"] + sorted([c for c in LIBS_SET if c in cols23])
        comp_no_ashr = [c for c in cols23 if c != "ASHR"]
        comp_no_mm = [c for c in cols23 if c != "Media móvil de 30 días"]

        # SPA comparisons (B=5000)
        for est_label, mat_df in [("MAE_macro", macro_mat_df), ("MAE_origen", origin_mat_df)]:
            lb_a = mat_df["ASHR"].to_numpy(float)
            lm_a = mat_df[comp_no_ashr].to_numpy(float)
            s_pr = spa_test_v2(lb_a, lm_a, B=B_SPA_MCS, mean_block=blk, seed=SEED, studentize=True, variance_estimator="pr_kernel")
            s_bt = spa_test_v2(lb_a, lm_a, B=B_SPA_MCS, mean_block=blk, seed=SEED, studentize=True, variance_estimator="bootstrap")
            s_un = spa_test_v2(lb_a, lm_a, B=B_SPA_MCS, mean_block=blk, seed=SEED, studentize=False, variance_estimator="pr_kernel", clip_zero=False, finite_sample_floor=False)

            arch_def = ArchSPA(lb_a, lm_a, block_size=blk_int, reps=B_SPA_MCS, bootstrap="stationary", studentize=True, seed=SEED)
            arch_def.compute()

            # Pre-studentized inputs passed to arch.SPA to verify Hansen (2005) studentized consistency
            d_raw = lb_a[:, None] - lm_a
            omega_pr = np.sqrt(hansen_politis_romano_variance(d_raw, blk))
            scale_pr = np.sqrt(T_mat) / omega_pr
            arch_pre = ArchSPA(
                np.zeros(T_mat),
                -(d_raw * scale_pr[None, :]),
                block_size=blk_int,
                reps=B_SPA_MCS,
                bootstrap="stationary",
                studentize=False,
                seed=SEED,
            )
            arch_pre.compute()

            s_mm = spa_test_v2(
                mat_df["Media móvil de 30 días"].to_numpy(float),
                mat_df[comp_no_mm].to_numpy(float),
                B=B_SPA_MCS,
                mean_block=blk,
                seed=SEED,
                studentize=True,
                variance_estimator="pr_kernel",
            )
            spa_rows.append({
                "escenario": sc,
                "estimando": est_label,
                "T": T_mat,
                "mean_block": blk,
                "spa_stat_ASHR": s_pr["spa_stat"],
                "spa_stat_unclipped_ASHR": s_pr["spa_stat_unclipped"],
                "p_spa_ASHR_stud_pr_cons": s_pr["p_spa_consistent"],
                "p_spa_ASHR_stud_boot_cons": s_bt["p_spa_consistent"],
                "p_spa_ASHR_unstud": s_un["p_spa_consistent"],
                "p_spa_ASHR_arch_default": float(arch_def.pvalues["consistent"]),
                "p_spa_ASHR_arch_prestud": 1.0 if s_pr["spa_stat_unclipped"] <= 0.0 else float(arch_pre.pvalues["consistent"]),
                "p_spa_MM30_stud_pr_cons": s_mm["p_spa_consistent"],
            })

        # MCS comparisons (B=5000)
        for est_label, mat_df in [("MAE_macro", macro_mat_df), ("MAE_origen", origin_mat_df)]:
            m23 = mcs_t_max_v2(mat_df.to_numpy(float), cols23, alpha=0.10, B=B_SPA_MCS, mean_block=blk, seed=SEED)
            am23 = ArchMCS(mat_df.to_numpy(float), size=0.10, reps=B_SPA_MCS, block_size=blk_int, method="max", bootstrap="stationary", seed=SEED)
            am23.compute()
            arch_s23 = [cols23[i] for i in am23.included]
            set_v2 = set(m23["survivors"])
            set_ar = set(arch_s23)
            jacc = len(set_v2 & set_ar) / max(1, len(set_v2 | set_ar))

            mbib = mcs_t_max_v2(mat_df[bib_cols].to_numpy(float), bib_cols, alpha=0.10, B=B_SPA_MCS, mean_block=blk, seed=SEED)

            mcs_rows.append({
                "escenario": sc,
                "estimando": est_label,
                "T": T_mat,
                "mean_block": blk,
                "n_surv_23_v2": len(m23["survivors"]),
                "n_surv_23_arch": len(arch_s23),
                "diff_n_surv_23": len(m23["survivors"]) - len(arch_s23),
                "jaccard_23_v2_vs_arch": round(jacc, 4),
                "ashr_in_mcs23_v2": "ASHR" in set_v2,
                "ashr_in_mcs23_arch": "ASHR" in set_ar,
                "p_mcs23_ashr_v2": m23["p_mcs"]["ASHR"],
                "n_surv_bib10_v2": len(mbib["survivors"]),
                "ashr_in_mcs_bib10_v2": "ASHR" in mbib["survivors"],
                "survivors_23_v2": ", ".join(m23["survivors"]),
                "survivors_bib10_v2": ", ".join(mbib["survivors"]),
            })

        # Capture samples for P2 plots & factorial summary
        if sc == "D1_espaciotemporal_cinco_variables":
            p2_d1_df = ashr_cf.merge(
                bench_al[bench_al["metodo"].isin([best_lib_kwh_name, "Media móvil de 30 días"])][list(sp.join_keys) + ["metodo", sp.pred_col]],
                on=list(sp.join_keys),
                how="left",
            )
            p2_d1_df = p2_d1_df.assign(hpo_pred=hpo_df.set_index(list(sp.join_keys)).loc[pd.MultiIndex.from_frame(p2_d1_df[list(sp.join_keys)]), sp.pred_col].to_numpy()) if False else p2_d1_df
            p2_d1_df = p2_d1_df.merge(
                hpo_df[list(sp.join_keys) + [sp.pred_col]].rename(columns={sp.pred_col: "pred_hpo"}),
                on=list(sp.join_keys),
                how="left",
            )
        if sc == "RecGEFSv12_espaciotemporal_28d":
            p2_rec_df = ashr_cf.merge(
                bench_al[bench_al["metodo"] == best_lib_kwh_name][list(sp.join_keys) + [sp.pred_col]].rename(columns={sp.pred_col: "pred_bib"}),
                on=list(sp.join_keys),
                how="left",
            )
        if sc == "DirectoSinClima_intraplanta_28d":
            p2_dir_df = ashr_cf.merge(
                bench_al[bench_al["metodo"] == best_lib_kwh_name][list(sp.join_keys) + [sp.pred_col]].rename(columns={sp.pred_col: "pred_bib"}),
                on=list(sp.join_keys),
                how="left",
            )

        # Descriptive factorial rows by plant & scenario
        plant_g = ashr_cf.groupby(["code", "grupo"], as_index=False).agg(
            cap_MW_media=("cap_MW", "mean"),
            mae_kwh=("_ae_kwh", "mean"),
            mae_cf=("_ae_cf", "mean"),
        )
        plant_g["escenario"] = sc
        plant_g["suite"] = sp.suite
        plant_g["task"] = sp.task
        plant_g["horizon_days"] = sp.horizon_days
        plant_g["weather_on"] = sp.weather_on
        plant_g["mode_eval"] = sp.mode_eval
        factorial_rows.extend(plant_g.to_dict("records"))

    # Adjust Holm within families for reconcil_rows and all9_rows
    df_rec = pd.DataFrame(reconcil_rows)
    df_rec["p_holm_macro_fam"] = np.nan
    for (ref_t, suite_v), idx in df_rec.groupby(["tipo_referencia", "suite"], sort=False).groups.items():
        mask = (df_rec.loc[idx, "T_eff_macro"] >= 2.0 * df_rec.loc[idx, "h_overlap"]) & df_rec.loc[idx, "p_dm_macro_v2"].notna()
        v_idx = df_rec.loc[idx].index[mask]
        if len(v_idx) > 0:
            df_rec.loc[v_idx, "p_holm_macro_fam"] = apply_holm(df_rec.loc[v_idx, "p_dm_macro_v2"].to_numpy(float))

    df_all9 = pd.DataFrame(all9_rows)
    df_all9["p_holm_por_escenario"] = np.nan
    for sc_v, idx in df_all9.groupby("escenario", sort=False).groups.items():
        df_all9.loc[idx, "p_holm_por_escenario"] = apply_holm(df_all9.loc[idx, "p_dm_macro_v2"].fillna(1.0).to_numpy(float))
    df_all9["p_holm_global_252"] = apply_holm(df_all9["p_dm_macro_v2"].fillna(1.0).to_numpy(float))

    # Sensitivity T_eff across thresholds for BOTH MAE_macro and MAE_origen
    df_ts = pd.DataFrame(sens_teff_rows_store)
    sens_teff_out = []
    for est, teff_col, p_col, d_col in [
        ("MAE_macro", "T_eff_macro", "p_dm_macro_v2", "diff_macro"),
        ("MAE_origen", "T_eff_origen", "p_dm_origen_v2", "diff_origen"),
    ]:
        for label, fn in [
            ("Original_T_eff>=2*H_dias", lambda r, tc=teff_col: r[tc] >= 2.0 * r["horizon_days"]),
            ("Tecnico_T_eff>=2*h_overlap", lambda r, tc=teff_col: r[tc] >= 2.0 * r["h_overlap"]),
            ("Umbral_T_eff>=15", lambda r, tc=teff_col: r[tc] >= 15.0),
            ("Umbral_T_eff>=20", lambda r, tc=teff_col: r[tc] >= 20.0),
            ("Umbral_T_eff>=25", lambda r, tc=teff_col: r[tc] >= 25.0),
            ("Umbral_T_eff>=30", lambda r, tc=teff_col: r[tc] >= 30.0),
            ("Umbral_T_eff>=40", lambda r, tc=teff_col: r[tc] >= 40.0),
        ]:
            sub = df_ts[df_ts.apply(fn, axis=1)].copy()
            n_sig = n_win = n_loss = 0
            for _, grp in sub.groupby("suite"):
                ph = apply_holm(grp[p_col].fillna(1.0).to_numpy(float))
                sig = ph < 0.05
                n_sig += int(sig.sum())
                n_win += int((sig & (grp[d_col].to_numpy(float) < 0)).sum())
                n_loss += int((sig & (grp[d_col].to_numpy(float) > 0)).sum())
            sens_teff_out.append({
                "estimando": est,
                "criterio": label,
                "escenarios_evaluables": int(len(sub)),
                "escenarios_excluidos": int(28 - len(sub)),
                "sig_holm_total": n_sig,
                "sig_holm_favor_SM_CFHR": n_win,
                "sig_holm_favor_Bib": n_loss,
            })

    # Write CSVs
    df_rec.to_csv(d02 / "reconciliacion_estimandos_28_escenarios.csv", index=False)
    df_all9.to_csv(d02 / "comparacion_todas_9_bibliotecas_macro.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(d02 / "verificacion_no_solapamiento_28_escenarios.csv", index=False)

    pd.DataFrame(spa_rows).to_csv(d03 / "spa_reconciliacion_28_escenarios.csv", index=False)
    pd.DataFrame(mcs_rows).to_csv(d03 / "mcs_reconciliacion_28_escenarios.csv", index=False)

    pd.DataFrame(sens_teff_out).to_csv(d04 / "sensibilidad_umbral_teff_macro_y_origen.csv", index=False)
    pd.DataFrame(sens_blk_hac_rows).to_csv(d04 / "sensibilidad_bloques_y_hac.csv", index=False)
    pd.DataFrame(logo_rows).to_csv(d04 / "sensibilidad_clusters_espaciales_logo.csv", index=False)

    pd.DataFrame(metrics_rows).to_csv(d05 / "metricas_completas_28_escenarios_cf_y_kwh.csv", index=False)

    df_fact = pd.DataFrame(factorial_rows)
    df_fact.to_csv(d06 / "analisis_factorial_descriptivo.csv", index=False)

    # --- P2 Figures (real vs predicted + factorial descriptive) ---
    # 1. D+1 representative plant
    if p2_d1_df is not None:
        top_code = str(p2_d1_df.groupby("code")["y_kwh"].mean().idxmax())
        sub_p = p2_d1_df[p2_d1_df["code"] == top_code].sort_values("date").drop_duplicates("date")
        fig, ax = plt.subplots(figsize=(11, 4.2))
        dates = pd.to_datetime(sub_p["date"])
        ax.plot(dates, sub_p["y_kwh"] / 1e3, label="Observado", color="black", lw=1.4)
        ax.plot(dates, sub_p["prediccion_kwh_x"] / 1e3, label="SM-CFHR-Fixed", color="#1b9e77", lw=1.3)
        if "pred_hpo" in sub_p.columns:
            ax.plot(dates, sub_p["pred_hpo"] / 1e3, label="SM-CFHR-HPO", color="#7570b3", lw=1.0, ls="--")
        ax.set_title(f"Real vs. Predicho D+1 (ET + Clima) — Planta {top_code}")
        ax.set_ylabel("Generación diaria (MWh/día)")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        plt.tight_layout()
        fig.savefig(d06 / "real_vs_predicho_d1.png", dpi=180)
        fig.savefig(d06 / "real_vs_predicho_d1.pdf")
        plt.close(fig)

    # 2. Multi-horizon recursive 28d (favorable ET + GeFSv12)
    if p2_rec_df is not None:
        top_code = str(p2_rec_df.groupby("code")["y_kwh"].mean().idxmax())
        sub_r = p2_rec_df[p2_rec_df["code"] == top_code].copy()
        agg_r = sub_r.groupby("origen", as_index=False)[["y_kwh", "prediccion_kwh", "pred_bib"]].mean().sort_values("origen")
        fig, ax = plt.subplots(figsize=(11, 4.2))
        ox = pd.to_datetime(agg_r["origen"])
        ax.plot(ox, agg_r["y_kwh"] / 1e3, "o-", label="Observado medio (H=28d)", color="black", lw=1.3, ms=4)
        ax.plot(ox, agg_r["prediccion_kwh"] / 1e3, "s-", label="SM-CFHR-Fixed", color="#1b9e77", lw=1.3, ms=4)
        ax.plot(ox, agg_r["pred_bib"] / 1e3, "^--", label="Mejor Biblioteca (Lasso)", color="#d95f02", lw=1.1, ms=4)
        ax.set_title(f"Real vs. Predicho Recursivo 28d (ET + GEFSv12) — Promedio por origen en {top_code}")
        ax.set_ylabel("Generación media en ventana 28d (MWh/día)")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        plt.tight_layout()
        fig.savefig(d06 / "real_vs_predicho_multihorizonte_rec.png", dpi=180)
        fig.savefig(d06 / "real_vs_predicho_multihorizonte_rec.pdf")
        plt.close(fig)

    # 3. Direct accumulated 28d (unfavorable IP DirectoSinClima_intraplanta_28d) — respect accumulated total!
    if p2_dir_df is not None:
        top_code = str(p2_dir_df.groupby("code")["y_total_kwh"].mean().idxmax())
        sub_d = p2_dir_df[p2_dir_df["code"] == top_code].sort_values("origen")
        fig, ax = plt.subplots(figsize=(11, 4.2))
        ox = pd.to_datetime(sub_d["origen"])
        ax.plot(ox, sub_d["y_total_kwh"] / 1e6, "o-", label="Observado acumulado 28d", color="black", lw=1.3, ms=4)
        ax.plot(ox, sub_d["prediccion_total_kwh"] / 1e6, "s-", label="SM-CFHR-Fixed", color="#1b9e77", lw=1.3, ms=4)
        ax.plot(ox, sub_d["pred_bib"] / 1e6, "^--", label="Mejor Biblioteca (k-NN)", color="#d95f02", lw=1.1, ms=4)
        ax.set_title(f"Real vs. Predicho Directo Acumulado 28d (IP sin clima, caso desfavorable) — Planta {top_code}")
        ax.set_ylabel("Energía acumulada en 28 días (GWh)")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        plt.tight_layout()
        fig.savefig(d06 / "real_vs_predicho_directo_acumulado.png", dpi=180)
        fig.savefig(d06 / "real_vs_predicho_directo_acumulado.pdf")
        plt.close(fig)

    # 4. Descriptive factorial summary figure (correlational, not causal)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    by_suite = df_fact.groupby(["suite", "task"], as_index=False)["mae_cf"].mean()
    labels_s = [f"{r['suite']}\n({r['task'][:2].upper()})" for _, r in by_suite.iterrows()]
    axes[0].barh(labels_s, by_suite["mae_cf"], color="#2b8cbe")
    axes[0].set_xlabel("MAE_CF_macro promedio (SM-CFHR-Fixed)")
    axes[0].set_title("Desempeño por régimen y estrategia (descriptivo)")
    axes[0].grid(axis="x", alpha=0.25)

    by_plant = df_fact.groupby("code", as_index=False).agg(cap=("cap_MW_media", "mean"), cf=("mae_cf", "mean"), kwh=("mae_kwh", "mean"))
    axes[1].scatter(by_plant["cap"], by_plant["cf"], color="#1b9e77", s=55, alpha=0.85)
    for _, r in by_plant.iterrows():
        axes[1].annotate(r["code"], (r["cap"], r["cf"]), fontsize=7, xytext=(3, 2), textcoords="offset points")
    axes[1].set_xlabel("Capacidad instalada media (MW)")
    axes[1].set_ylabel("MAE_CF_macro medio")
    axes[1].set_title("Error en CF vs. escala instalada (correlación exploratoria)")
    axes[1].grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(d06 / "factorial_descriptivo_regimenes.png", dpi=180)
    fig.savefig(d06 / "factorial_descriptivo_regimenes.pdf")
    plt.close(fig)

    # Metadata & Manifest
    meta = {
        "version": VERSION,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed": SEED,
        "B_spa_mcs": B_SPA_MCS,
        "B_bootstrap_macro": B_BOOT,
        "oof_manifest_sha256": _sha256(oof_dir / "manifest_ashr_oof.json"),
        "n_sign_flips_macro_vs_origin_mejor_bib": int(
            df_rec[df_rec["tipo_referencia"] == "Mejor_Bib_Retrospectivo"]["sign_flip_macro_vs_origin"].sum()
        ),
    }
    (out_dir / "metadatos_revision_v2.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest_rows = []
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.name != "MANIFIESTO_REVISION_V2.csv":
            manifest_rows.append({
                "ruta_relativa": p.relative_to(out_dir).as_posix(),
                "tamano_bytes": p.stat().st_size,
                "sha256": _sha256(p),
            })
    with (out_dir / "MANIFIESTO_REVISION_V2.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ruta_relativa", "tamano_bytes", "sha256"], lineterminator="\n")
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"Revision v2 complete: {len(manifest_rows)} artifacts generated in {out_dir}")
    sub_mb = df_rec[df_rec["tipo_referencia"] == "Mejor_Bib_Retrospectivo"]
    print(f"Sign flips macro vs origin (Mejor_Bib): {int(sub_mb['sign_flip_macro_vs_origin'].sum())}/28")
    print(f"Holm sig on MAE_macro (T_eff>=2*hov): {int((sub_mb['p_holm_macro_fam'] < 0.05).sum())}/28")


if __name__ == "__main__":
    main()

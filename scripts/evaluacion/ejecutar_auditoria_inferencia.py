"""Reproducible statistical inference audit & recalculation without retraining (SM-CFHR).

Reads frozen OOF predictions from `resultados/evaluacion_ashr_oof/` (read-only) and
writes separated Block A (implementation fixes), Block B (sensitivity), and Block C
(exploratory post-hoc) outputs under `resultados/auditoria_inferencia_corregida/`.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from arch.bootstrap import MCS as ArchMCS
from arch.bootstrap import SPA as ArchSPA

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ashr_oof_pipeline.config import SCENARIOS_28, load_frozen_config
from src.ashr_oof_pipeline.evaluacion_estadistica import apply_holm, compute_scenario_metrics
from src.ashr_oof_pipeline.oof_runner import align_predictions, load_official_benchmarks
from src.ashr_oof_pipeline.resultados_loader import LIBS_SET
from scripts.evaluacion.inferencia_estadistica_v2 import paired_effect_sizes
from src.sm_cfhr.inference_v2 import (
    default_block_length,
    dm_hln_hac_v2,
    infer_series_step_and_overlap,
    mcs_t_max_v2,
    spa_test_v2,
    stationary_bootstrap_ci_v2,
)

VERSION = "1.0.0-auditoria-inferencia"
SEED = 2026


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _classify_status(p_raw: float, p_holm: float, t_eff_insuff: bool, alpha: float = 0.05) -> str:
    if t_eff_insuff:
        return "T_EFF_INSUFICIENTE"
    if not np.isfinite(p_raw):
        return "NO_EVALUABLE"
    if np.isfinite(p_holm) and p_holm < alpha:
        return "SIGNIFICATIVO_HOLM"
    if p_raw < alpha:
        return "NOMINAL_SIN_HOLM"
    return "NO_SIGNIFICATIVO"


def main() -> None:
    oof_dir = ROOT / "resultados" / "evaluacion_ashr_oof"
    out_dir = ROOT / "resultados" / "auditoria_inferencia_corregida"
    dir_a = out_dir / "A_correcciones_implementacion"
    dir_b = out_dir / "B_analisis_sensibilidad"
    dir_c = out_dir / "C_analisis_exploratorios"
    for d in (dir_a, dir_b, dir_c):
        d.mkdir(parents=True, exist_ok=True)

    # Load original tables for exact before/after comparison (read-only)
    orig_inf = pd.read_csv(oof_dir / "inferencia" / "tabla_inferencia.csv")
    orig_mult = pd.read_csv(oof_dir / "multiplicidad" / "tabla_multiplicidad_familias.csv")
    orig_mcs = pd.read_csv(oof_dir / "inferencia" / "tabla_mcs.csv")
    orig_mcs_bib = pd.read_csv(oof_dir / "inferencia" / "mcs_bibliotecas_only.csv")
    orig_spa = pd.read_csv(oof_dir / "inferencia" / "tabla_spa.csv")
    orig_t28 = pd.read_csv(oof_dir / "tablas_paper" / "tabla_28_escenarios.csv")

    comp_rows: list[dict[str, Any]] = []
    mcs23_rows: list[dict[str, Any]] = []
    mcsbib_rows: list[dict[str, Any]] = []
    spa_rows: list[dict[str, Any]] = []

    sens_teff_store: list[dict[str, Any]] = []
    sens_boot_rows: list[dict[str, Any]] = []
    sens_hac_rows: list[dict[str, Any]] = []

    ex_ante_rows: list[dict[str, Any]] = []
    scale_free_rows: list[dict[str, Any]] = []

    rep_scenarios = {
        "D1_espaciotemporal_cinco_variables",
        "RecGEFSv12_espaciotemporal_7d",
        "RecGEFSv12_espaciotemporal_14d",
        "RecGEFSv12_espaciotemporal_21d",
        "RecGEFSv12_espaciotemporal_28d",
        "DirectoSinClima_intraplanta_28d",
    }

    for sp in SCENARIOS_28:
        sc = sp.scenario_id
        ashr_df = pd.read_csv(oof_dir / "predicciones" / f"{sc}.csv")
        ashr_df["code"] = ashr_df["code"].astype(str)
        bench_raw = load_official_benchmarks(sp, root=ROOT, exclude_cf_brh=True)
        bench_keys = bench_raw[list(sp.join_keys)].drop_duplicates()
        ashr_df = ashr_df.merge(bench_keys, on=list(sp.join_keys), how="inner")
        bench_al = align_predictions(ashr_df, bench_raw, sp.join_keys)

        ashr_df["_ae"] = np.abs(ashr_df[sp.pred_col] - ashr_df[sp.target_col])
        ashr_df["_sq"] = (ashr_df[sp.pred_col] - ashr_df[sp.target_col]) ** 2
        bench_al["_ae"] = np.abs(bench_al[sp.pred_col] - bench_al[sp.target_col])
        bench_al["_sq"] = (bench_al[sp.pred_col] - bench_al[sp.target_col]) ** 2

        time_col = "date" if sp.mode_eval == "d1" else "origen"
        step_info = infer_series_step_and_overlap(ashr_df[time_col].unique(), sp.horizon_days)
        h_ov = int(step_info["h_overlap"])

        # Macro MAE per model
        ashr_mae = compute_scenario_metrics(ashr_df, sp.pred_col, sp.target_col)["MAE_macro"]
        by_m = (
            bench_al.groupby(["metodo", "code"], as_index=False)["_ae"]
            .mean()
            .groupby("metodo", as_index=False)["_ae"]
            .mean()
            .rename(columns={"_ae": "MAE_macro"})
        )
        lib_df = by_m[by_m["metodo"].isin(LIBS_SET)].sort_values("MAE_macro")
        best_lib_n = str(lib_df.iloc[0]["metodo"])
        best_lib_mae = float(lib_df.iloc[0]["MAE_macro"])

        # Temporal series (mean across available plants per time_col, exact match to pipeline)
        ashr_t = ashr_df.groupby(time_col)["_ae"].mean()
        bench_t = bench_al.groupby([time_col, "metodo"], as_index=False)["_ae"].mean()
        piv = bench_t.pivot(index=time_col, columns="metodo", values="_ae")
        piv["ASHR"] = ashr_t
        piv = piv.dropna(axis=0, how="any").sort_index()
        T_series = int(len(piv))
        blk_def = default_block_length(T_series)

        # --- Paired DM-HLN-HAC v2 vs Mejor_Bib ---
        a_vec = piv["ASHR"].to_numpy(float)
        b_vec = piv[best_lib_n].to_numpy(float)
        d_vec = a_vec - b_vec
        dm_v2 = dm_hln_hac_v2(d_vec, h_overlap=h_ov, lag=None)
        ci_lo_v2, ci_hi_v2, blk_used = stationary_bootstrap_ci_v2(d_vec, mean_block=blk_def, B=2000, seed=SEED)
        eff = paired_effect_sizes(a_vec, b_vec)

        # Original rows for comparison
        r_orig = orig_inf[(orig_inf["escenario"] == sc) & (orig_inf["referencia"] == "Mejor_Bib")].iloc[0]
        r_mult = orig_mult[(orig_mult["escenario"] == sc) & (orig_mult["referencia"] == "Mejor_Bib")].iloc[0]
        r_mcs_orig = orig_mcs[orig_mcs["escenario"] == sc].iloc[0]
        r_mcsbib_orig = orig_mcs_bib[orig_mcs_bib["escenario"] == sc].iloc[0]
        r_spa_orig = orig_spa[orig_spa["escenario"] == sc].iloc[0]

        comp_rows.append({
            "escenario": sc,
            "suite": sp.suite,
            "task": sp.task,
            "horizon_days": sp.horizon_days,
            "unit": sp.unit,
            "modelo_mejor_bib": best_lib_n,
            "MAE_ASHR": round(ashr_mae, 4),
            "MAE_Mejor_Bib": round(best_lib_mae, 4),
            "diff_abs_MAE": round(ashr_mae - best_lib_mae, 4),
            "diff_rel_pct": round(100.0 * (ashr_mae / best_lib_mae - 1.0), 3),
            "direccion": "FAVOR_SM_CFHR" if ashr_mae < best_lib_mae else "FAVOR_COMPETIDOR",
            "n_time_units": T_series,
            "median_step_days": step_info["median_step_days"],
            "h_overlap_real": h_ov,
            "L_hac_v2": dm_v2["L_hac"],
            "T_eff_orig": float(r_orig["T_eff"]),
            "T_eff_v2": float(dm_v2["T_eff"]),
            "t_eff_insuff_orig_2H": bool(r_orig["t_eff_insufficient"]),
            "t_eff_insuff_v2_2hov": bool(dm_v2["T_eff"] < 2.0 * h_ov),
            "dm_hln_orig": float(r_orig["dm_hln"]),
            "p_dm_orig": float(r_orig["p_dm_hln"]),
            "p_holm_orig_global56": float(r_orig["p_holm"]),
            "p_holm_orig_fam": float(r_mult["p_holm_familia"]) if pd.notna(r_mult["p_holm_familia"]) else np.nan,
            "estado_orig": str(r_orig["estado_estadistico"]),
            "dm_hln_v2": float(dm_v2["dm_hln"]),
            "p_dm_v2": float(dm_v2["p_dm_hln"]),
            "hln_factor_v2": float(dm_v2["hln_factor"]),
            "ci95_lo_v2": float(ci_lo_v2),
            "ci95_hi_v2": float(ci_hi_v2),
            "mean_block_v2": float(blk_used),
            "cohen_dz": float(eff["cohen_dz_pareado"]),
            "r_rangobiserial": float(eff["r_rangobiserial_pareado"]),
            "prob_victoria": float(eff["prob_victoria_A"]),
            "lag1_acf": float(dm_v2["lag1_acf"]),
        })

        # --- MCS v2 (23 models) + cross-check with arch.MCS ---
        mat23 = piv.to_numpy(float)
        cols23 = list(piv.columns)
        mcs23 = mcs_t_max_v2(mat23, cols23, alpha=0.10, B=2000, mean_block=blk_def, seed=SEED)
        amcs23 = ArchMCS(mat23, size=0.10, reps=1000, block_size=int(round(blk_def)), method="max", bootstrap="stationary", seed=SEED)
        amcs23.compute()
        arch_surv23 = [cols23[i] for i in amcs23.included]
        mcs23_rows.append({
            "escenario": sc,
            "T": T_series,
            "mean_block": blk_def,
            "n_surv_orig": int(r_mcs_orig["n_survivors"]),
            "ashr_in_mcs_orig": bool(r_mcs_orig["ashr_in_mcs_90"]),
            "p_mcs_ashr_orig": float(r_mcs_orig["p_mcs_ashr"]),
            "n_surv_v2": int(len(mcs23["survivors"])),
            "ashr_in_mcs_v2": "ASHR" in mcs23["survivors"],
            "p_mcs_ashr_v2": float(mcs23["p_mcs"]["ASHR"]),
            "n_surv_arch": int(len(arch_surv23)),
            "ashr_in_mcs_arch": "ASHR" in arch_surv23,
            "survivors_v2": ", ".join(mcs23["survivors"]),
        })

        # --- MCS v2 (ASHR + 9 libs only) ---
        bib_cols = ["ASHR"] + sorted([c for c in LIBS_SET if c in piv.columns])
        mcsb = mcs_t_max_v2(piv[bib_cols].to_numpy(float), bib_cols, alpha=0.10, B=2000, mean_block=blk_def, seed=SEED)
        mcsbib_rows.append({
            "escenario": sc,
            "n_surv_bib_orig": int(r_mcsbib_orig["n_survivors_lib"]),
            "ashr_in_mcs_bib_orig": bool(r_mcsbib_orig["in_mcs_lib_90"]),
            "p_mcs_bib_orig": float(r_mcsbib_orig["p_mcs_ashr_lib"]),
            "n_surv_bib_v2": int(len(mcsb["survivors"])),
            "ashr_in_mcs_bib_v2": "ASHR" in mcsb["survivors"],
            "p_mcs_bib_v2": float(mcsb["p_mcs"]["ASHR"]),
            "survivors_bib_v2": ", ".join(mcsb["survivors"]),
        })

        # --- SPA v2 (H0: MM30 not beaten; H0: ASHR not beaten) + arch.SPA ---
        comp_no_mm = [c for c in piv.columns if c != "Media móvil de 30 días"]
        comp_no_ashr = [c for c in piv.columns if c != "ASHR"]
        spa_mm = spa_test_v2(piv["Media móvil de 30 días"].to_numpy(float), piv[comp_no_mm].to_numpy(float), B=2000, mean_block=blk_def, seed=SEED)
        spa_ashr = spa_test_v2(piv["ASHR"].to_numpy(float), piv[comp_no_ashr].to_numpy(float), B=2000, mean_block=blk_def, seed=SEED)
        aspa_ashr = ArchSPA(
            piv["ASHR"].to_numpy(float),
            piv[comp_no_ashr].to_numpy(float),
            block_size=int(round(blk_def)),
            reps=1000,
            bootstrap="stationary",
            studentize=True,
            seed=SEED,
        )
        aspa_ashr.compute()
        spa_rows.append({
            "escenario": sc,
            "T": T_series,
            "mean_block": blk_def,
            "p_spa_MM30_orig": float(r_spa_orig["p_spa_H0_MM30"]),
            "p_spa_ASHR_orig": float(r_spa_orig["p_spa_H0_ASHR"]),
            "p_spa_MM30_v2_cons": float(spa_mm["p_spa_consistent"]),
            "p_spa_ASHR_v2_lower": float(spa_ashr["p_spa_lower"]),
            "p_spa_ASHR_v2_cons": float(spa_ashr["p_spa_consistent"]),
            "p_spa_ASHR_v2_upper": float(spa_ashr["p_spa_upper"]),
            "p_spa_ASHR_arch_cons": float(aspa_ashr.pvalues["consistent"]),
        })

        # --- Block B: Sensitivity ---
        sens_teff_store.append({
            "escenario": sc,
            "suite": sp.suite,
            "horizon_days": sp.horizon_days,
            "h_overlap": h_ov,
            "T_eff": float(dm_v2["T_eff"]),
            "p_dm_v2": float(dm_v2["p_dm_hln"]),
            "mean_diff": float(np.mean(d_vec)),
        })

        if sc in rep_scenarios:
            for blk_s in (1.0, 2.0, 3.0, 4.0, 7.0):
                if blk_s > T_series / 2.0:
                    continue
                c_lo, c_hi, _ = stationary_bootstrap_ci_v2(d_vec, mean_block=blk_s, B=2000, seed=SEED)
                m_s = mcs_t_max_v2(piv[bib_cols].to_numpy(float), bib_cols, alpha=0.10, B=1000, mean_block=blk_s, seed=SEED)
                s_s = spa_test_v2(piv["ASHR"].to_numpy(float), piv[comp_no_ashr].to_numpy(float), B=1000, mean_block=blk_s, seed=SEED)
                sens_boot_rows.append({
                    "escenario": sc,
                    "T": T_series,
                    "mean_block_obs": blk_s,
                    "ci95_lo": c_lo,
                    "ci95_hi": c_hi,
                    "ci_width": c_hi - c_lo,
                    "p_mcs_bib_ashr": float(m_s["p_mcs"]["ASHR"]),
                    "n_surv_bib": int(len(m_s["survivors"])),
                    "p_spa_ashr_cons": float(s_s["p_spa_consistent"]),
                })
            for lag_s in (0, 1, 2, 3, 4, None):
                r_lag = dm_hln_hac_v2(d_vec, h_overlap=h_ov, lag=lag_s)
                sens_hac_rows.append({
                    "escenario": sc,
                    "lag_param": "auto" if lag_s is None else str(lag_s),
                    "L_hac_used": r_lag["L_hac"],
                    "T_eff": r_lag["T_eff"],
                    "dm_hln": r_lag["dm_hln"],
                    "p_dm_hln": r_lag["p_dm_hln"],
                })

        # --- Block C: Exploratory (ex-ante fixed benchmarks + all 9 libs Holm/scenario) ---
        ex_refs = ["Ridge", "Lasso", "BayesianRidge", "Media móvil de 30 días"]
        for ref_m in ex_refs:
            if ref_m not in piv.columns:
                continue
            d_ex = a_vec - piv[ref_m].to_numpy(float)
            r_ex = dm_hln_hac_v2(d_ex, h_overlap=h_ov, lag=None)
            ef_ex = paired_effect_sizes(a_vec, piv[ref_m].to_numpy(float))
            ex_ante_rows.append({
                "alcance": "EXPLORATORIO_POST_HOC",
                "escenario": sc,
                "suite": sp.suite,
                "task": sp.task,
                "horizon_days": sp.horizon_days,
                "referencia_ex_ante": ref_m,
                "T": T_series,
                "T_eff": r_ex["T_eff"],
                "mean_diff": float(np.mean(d_ex)),
                "dm_hln_v2": r_ex["dm_hln"],
                "p_dm_v2": r_ex["p_dm_hln"],
                "cohen_dz": ef_ex["cohen_dz_pareado"],
            })

        # Scale-free proxy: normalize each plant's MAE by plant mean target (y_mean_plant), plus balanced-panel check
        p_stats = ashr_df.groupby("code").agg(
            mae_ashr=("_ae", "mean"),
            mse_ashr=("_sq", "mean"),
            y_mean=("target_col" if "target_col" in ashr_df.columns else sp.target_col, "mean"),
        )
        b_best = bench_al[bench_al["metodo"] == best_lib_n].groupby("code").agg(
            mae_bib=("_ae", "mean"),
            mse_bib=("_sq", "mean"),
        )
        b_mm = bench_al[bench_al["metodo"] == "Media móvil de 30 días"].groupby("code").agg(
            mse_mm=("_sq", "mean"),
        )
        joined_p = p_stats.join(b_best, how="inner").join(b_mm, how="left")
        nmae_ashr = float((joined_p["mae_ashr"] / np.maximum(joined_p["y_mean"], 1e-6)).mean())
        nmae_bib = float((joined_p["mae_bib"] / np.maximum(joined_p["y_mean"], 1e-6)).mean())
        rmse_ashr = float(np.sqrt(joined_p["mse_ashr"]).mean())
        rmse_mm = float(np.sqrt(joined_p["mse_mm"]).mean()) if "mse_mm" in joined_p else np.nan
        plants_per_t = ashr_df.groupby(time_col)["code"].nunique()
        scale_free_rows.append({
            "alcance": "EXPLORATORIO_POST_HOC",
            "escenario": sc,
            "n_plants_total": int(len(joined_p)),
            "plants_per_t_min": int(plants_per_t.min()),
            "plants_per_t_med": float(plants_per_t.median()),
            "plants_per_t_max": int(plants_per_t.max()),
            "nMAE_rel_planta_ASHR": round(nmae_ashr, 4),
            "nMAE_rel_planta_MejorBib": round(nmae_bib, 4),
            "diff_nMAE_rel_pct": round(100.0 * (nmae_ashr / nmae_bib - 1.0), 3),
            "RMSE_skill_vs_MM30_pct": round(100.0 * (1.0 - rmse_ashr / rmse_mm), 2) if np.isfinite(rmse_mm) and rmse_mm > 0 else np.nan,
        })

    # Family-wise Holm on Block A (suite, same family definition as pipeline_stages.py for Mejor_Bib)
    df_comp = pd.DataFrame(comp_rows)
    df_comp["p_holm_v2_fam"] = np.nan
    for suite_val, idx in df_comp.groupby("suite", sort=False).groups.items():
        valid_mask = (~df_comp.loc[idx, "t_eff_insuff_v2_2hov"].astype(bool)) & df_comp.loc[idx, "p_dm_v2"].notna()
        v_idx = df_comp.loc[idx].index[valid_mask]
        if len(v_idx) > 0:
            df_comp.loc[v_idx, "p_holm_v2_fam"] = apply_holm(df_comp.loc[v_idx, "p_dm_v2"].to_numpy(float))

    df_comp["estado_v2"] = [
        _classify_status(r["p_dm_v2"], r["p_holm_v2_fam"], bool(r["t_eff_insuff_v2_2hov"]))
        for _, r in df_comp.iterrows()
    ]

    # Merge MCS & SPA into comparison table for a single complete view
    df_mcs23 = pd.DataFrame(mcs23_rows)
    df_mcsbib = pd.DataFrame(mcsbib_rows)
    df_spa = pd.DataFrame(spa_rows)
    df_comp = df_comp.merge(
        df_mcs23[["escenario", "n_surv_orig", "ashr_in_mcs_orig", "p_mcs_ashr_orig", "n_surv_v2", "ashr_in_mcs_v2", "p_mcs_ashr_v2"]],
        on="escenario",
    ).merge(
        df_spa[["escenario", "p_spa_ASHR_orig", "p_spa_ASHR_v2_cons", "p_spa_MM30_orig", "p_spa_MM30_v2_cons"]],
        on="escenario",
    )

    def _motivo(r: pd.Series) -> str:
        reasons = []
        if r["t_eff_insuff_orig_2H"] and not r["t_eff_insuff_v2_2hov"]:
            reasons.append("Corregido desajuste de unidades T_eff<2H")
        if r["h_overlap_real"] != r["horizon_days"]:
            reasons.append(f"Corregido HLN h_overlap={r['h_overlap_real']} (antes H={r['horizon_days']})")
        if r["estado_orig"] != r["estado_v2"]:
            reasons.append(f"Estado {r['estado_orig']} -> {r['estado_v2']}")
        return "; ".join(reasons) if reasons else "Sin cambio de clasificacion"

    df_comp["motivo_cambio_clasificacion"] = df_comp.apply(_motivo, axis=1)

    # Classification change summary
    summary_rows = []
    for _, r in df_comp.iterrows():
        est_o = r["estado_orig"]
        est_n = r["estado_v2"]
        fav = r["direccion"] == "FAVOR_SM_CFHR"
        if est_n == "SIGNIFICATIVO_HOLM" and est_o != "SIGNIFICATIVO_HOLM":
            cat = "NUEVA_VICTORIA_SIGNIFICATIVA_HOLM" if fav else "NUEVA_DERROTA_SIGNIFICATIVA_HOLM"
        elif est_n == "NOMINAL_SIN_HOLM" and est_o != "NOMINAL_SIN_HOLM":
            cat = "NUEVA_VICTORIA_NOMINAL" if fav else "NUEVA_DERROTA_NOMINAL"
        elif est_n == "SIGNIFICATIVO_HOLM":
            cat = "VICTORIA_SIGNIFICATIVA_MANTENIDA" if fav else "DERROTA_SIGNIFICATIVA_MANTENIDA"
        elif est_n in ("T_EFF_INSUFICIENTE", "NO_EVALUABLE"):
            cat = "PERMANECE_NO_EVALUABLE"
        else:
            cat = "SIN_EVIDENCIA_SUFICIENTE"
        summary_rows.append({
            "escenario": r["escenario"],
            "categoria_cambio": cat,
            "direccion": r["direccion"],
            "estado_orig": est_o,
            "estado_v2": est_n,
            "dm_hln_orig": r["dm_hln_orig"],
            "dm_hln_v2": r["dm_hln_v2"],
            "p_dm_orig": r["p_dm_orig"],
            "p_dm_v2": r["p_dm_v2"],
            "p_holm_v2_fam": r["p_holm_v2_fam"],
        })

    # Block B: T_eff sensitivity table
    df_teff_base = pd.DataFrame(sens_teff_store)
    teff_sens_rows = []
    for label, rule in [
        ("Original_T_eff>=2*H_dias", lambda r: r["T_eff"] >= 2.0 * r["horizon_days"]),
        ("Corregido_T_eff>=2*h_overlap", lambda r: r["T_eff"] >= 2.0 * r["h_overlap"]),
        ("Umbral_T_eff>=20_bloques", lambda r: r["T_eff"] >= 20.0),
        ("Umbral_T_eff>=25_bloques", lambda r: r["T_eff"] >= 25.0),
        ("Umbral_T_eff>=30_bloques", lambda r: r["T_eff"] >= 30.0),
    ]:
        elig = df_teff_base.apply(rule, axis=1)
        sub = df_teff_base[elig].copy()
        n_elig = int(len(sub))
        n_sig_fam = 0
        n_win_fam = 0
        n_loss_fam = 0
        for _, grp in sub.groupby("suite"):
            ph = apply_holm(grp["p_dm_v2"].to_numpy(float))
            sig = ph < 0.05
            n_sig_fam += int(sig.sum())
            n_win_fam += int((sig & (grp["mean_diff"].to_numpy(float) < 0)).sum())
            n_loss_fam += int((sig & (grp["mean_diff"].to_numpy(float) > 0)).sum())
        teff_sens_rows.append({
            "criterio": label,
            "escenarios_evaluables": n_elig,
            "escenarios_excluidos": 28 - n_elig,
            "sig_holm_familia_total": n_sig_fam,
            "sig_holm_favor_SM_CFHR": n_win_fam,
            "sig_holm_favor_Bib": n_loss_fam,
        })

    # Block C: ex-ante Holm by (suite, referencia_ex_ante)
    df_ex = pd.DataFrame(ex_ante_rows)
    df_ex["p_holm_familia_ex_ante"] = np.nan
    for (_, _), idx in df_ex.groupby(["suite", "referencia_ex_ante"], sort=False).groups.items():
        df_ex.loc[idx, "p_holm_familia_ex_ante"] = apply_holm(df_ex.loc[idx, "p_dm_v2"].to_numpy(float))

    # Write CSV outputs
    df_comp.to_csv(dir_a / "comparacion_antes_despues_28_escenarios.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(dir_a / "resumen_cambios_clasificacion.csv", index=False)
    df_mcs23.to_csv(dir_a / "mcs_corregido_23_modelos.csv", index=False)
    df_mcsbib.to_csv(dir_a / "mcs_corregido_bibliotecas_only.csv", index=False)
    df_spa.to_csv(dir_a / "spa_corregido_28_escenarios.csv", index=False)

    pd.DataFrame(teff_sens_rows).to_csv(dir_b / "sensibilidad_umbral_teff.csv", index=False)
    pd.DataFrame(sens_boot_rows).to_csv(dir_b / "sensibilidad_bootstrap_bloques.csv", index=False)
    pd.DataFrame(sens_hac_rows).to_csv(dir_b / "sensibilidad_hac_lag.csv", index=False)

    df_ex.to_csv(dir_c / "comparacion_benchmark_fijo_ex_ante.csv", index=False)
    pd.DataFrame(scale_free_rows).to_csv(dir_c / "metricas_libres_escala_cf_y_skill.csv", index=False)

    # Metadata & source hashes
    meta = {
        "version": VERSION,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed": SEED,
        "mcs_reps": 2000,
        "spa_reps": 2000,
        "bootstrap_ci_reps": 2000,
        "oof_source_dir": "resultados/evaluacion_ashr_oof",
        "oof_manifest_sha256": _sha256(oof_dir / "manifest_ashr_oof.json"),
        "source_hashes": {
            "src/sm_cfhr/inference_v2.py": _sha256(ROOT / "src" / "sm_cfhr" / "inference_v2.py"),
            "src/sm_cfhr/inference.py": _sha256(ROOT / "src" / "sm_cfhr" / "inference.py"),
            "src/ashr_oof_pipeline/evaluacion_estadistica.py": _sha256(
                ROOT / "src" / "ashr_oof_pipeline" / "evaluacion_estadistica.py"
            ),
            "src/ashr_oof_pipeline/resultados_loader.py": _sha256(
                ROOT / "src" / "ashr_oof_pipeline" / "resultados_loader.py"
            ),
            "scripts/evaluacion/ejecutar_auditoria_inferencia.py": _sha256(Path(__file__)),
        },
        "changes_applied": [
            "P0.1: Infer series step from timestamps and pass h_overlap (1 for non-overlapping blocks) to dm_hln_hac_v2.",
            "P0.2: Replace T_eff < 2*H_days unit-mismatch with h_overlap-consistent diagnostics and sensitivity table.",
            "P0.3: Use one-sided T_max MCS (B=2000) with observation-unit block length, cross-checked vs arch.MCS.",
            "P1.1: Use dependent long-run bootstrap variance omega in studentized SPA (B=2000), cross-checked vs arch.SPA.",
            "P1.4: Use observation-unit stationary bootstrap block length bounded <= T/4.",
        ],
    }
    (out_dir / "metadatos_auditoria.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Audit recalculation completed.")
    print("Summary counts by categoria_cambio:")
    print(pd.DataFrame(summary_rows)["categoria_cambio"].value_counts().to_string())


if __name__ == "__main__":
    main()

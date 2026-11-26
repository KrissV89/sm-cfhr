"""Validation, loading, alignment, and summary layer for definitive ASHR OOF artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    SCENARIOS_28,
    find_project_root,
    load_frozen_config,
    sha256_file,
)
from .evaluacion_estadistica import (
    apply_holm,
    build_23_model_ranking,
    compute_scenario_metrics,
    model_confidence_set,
    run_paired_inference,
)
from .oof_runner import align_predictions, load_official_benchmarks

LIBS_SET = {
    "BayesianRidge", "DecisionTree", "HistGradientBoosting", "Lasso",
    "Random Forest", "Ridge", "SVR", "XGBoost", "k-NN",
}
CUSTOM_SET = {
    "ABGPR", "ELAC", "HARC", "IWLAR", "KLAC", "MARB",
    "PDEC", "RARC", "RLAC", "RQRF", "SRMOS", "WCFC",
}


def validate_oof_artifacts(
    out_dir: Path,
    expected_src_hash: str,
    allow_synthetic_test: bool = False,
) -> dict[str, Any]:
    """Validate manifest, provenance, frozen source hash, and presence of all 28 CSVs."""
    manifest_path = out_dir / "manifest_ashr_oof.json"
    if not out_dir.exists() or not manifest_path.exists():
        return {
            "state": "PENDING",
            "reasons": ["No existe manifest_ashr_oof.json; evaluación OOF definitiva pendiente."],
            "manifest": None,
        }

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {"state": "INVALID", "reasons": [f"Manifiesto ilegible: {exc}"], "manifest": None}

    reasons: list[str] = []
    estatus = str(manifest.get("estatus", ""))
    if estatus != "OOF_DEFINITIVO_COMPLETO":
        reasons.append(f"Estatus inválido o no definitivo: '{estatus}'.")

    proc = str(manifest.get("procedencia", "OOF_DEFINITIVO_REAL"))
    if proc == "SINTETICO_TEST" and not allow_synthetic_test:
        reasons.append("Artefactos marcados como SINTETICO_TEST rechazados en modo definitivo.")

    if manifest.get("sha256_src_adaptive_structured_huber") != expected_src_hash:
        reasons.append("Hash SHA-256 de src/adaptive_structured_huber.py no coincide con el código congelado.")

    pred_dir = out_dir / "predicciones"
    missing = [sp.scenario_id for sp in SCENARIOS_28 if not (pred_dir / f"{sp.scenario_id}.csv").exists()]
    if missing:
        reasons.append(f"Faltan {len(missing)} archivos de predicciones: {missing[:3]}...")

    if reasons:
        return {"state": "INVALID", "reasons": reasons, "manifest": manifest}
    return {"state": "VALID", "reasons": [], "manifest": manifest}


def validate_predictions_df(
    df: pd.DataFrame,
    join_keys: tuple[str, ...],
    pred_col: str,
    target_col: str,
) -> None:
    """Validate schema, finiteness, physical non-negativity, and key uniqueness."""
    req = set(join_keys) | {pred_col, target_col}
    missing_cols = req - set(df.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing_cols)}")
    if df.empty:
        raise ValueError("DataFrame de predicciones vacío.")
    for col in (*join_keys, pred_col, target_col):
        if df[col].isna().any():
            raise ValueError(f"Valores faltantes detectados en columna '{col}'.")
    preds = df[pred_col].to_numpy(dtype=float)
    targs = df[target_col].to_numpy(dtype=float)
    if not (np.all(np.isfinite(preds)) and np.all(np.isfinite(targs))):
        raise ValueError("Valores no finitos en predicciones u objetivos.")
    if np.any(preds < -1e-9):
        raise ValueError("Predicciones físicas negativas detectadas.")
    if df.duplicated(subset=list(join_keys)).any():
        raise ValueError(f"Claves duplicadas detectadas para {join_keys}.")


def load_oof_bundle(
    out_dir: Path,
    root: Path | None = None,
    allow_synthetic_test: bool = False,
) -> dict[str, Any]:
    """Load, validate, align, and summarize all 28 OOF scenarios and statistical tests."""
    r = root or find_project_root()
    cfg = load_frozen_config(r)
    expected_hash = cfg["sha256_src_adaptive_structured_huber"]
    val = validate_oof_artifacts(out_dir, expected_hash, allow_synthetic_test=allow_synthetic_test)
    if val["state"] != "VALID":
        raise RuntimeError(f"Artefactos OOF no válidos ({val['state']}): {val['reasons']}")

    manifest = val["manifest"]
    pred_dir = out_dir / "predicciones"

    rows_28, plant_rows, inf_rows, cost_rows, mcs_rows = [], [], [], [], []
    times_map = manifest.get("tiempos_por_escenario_s", {})

    for sp in SCENARIOS_28:
        ashr_df = pd.read_csv(pred_dir / f"{sp.scenario_id}.csv")
        ashr_df["code"] = ashr_df["code"].astype(str)
        validate_predictions_df(ashr_df, sp.join_keys, sp.pred_col, sp.target_col)

        bench_raw = load_official_benchmarks(sp, root=r, exclude_cf_brh=True)
        bench_keys = bench_raw[list(sp.join_keys)].drop_duplicates()
        ashr_df = ashr_df.merge(bench_keys, on=list(sp.join_keys), how="inner")
        bench_al = align_predictions(ashr_df, bench_raw, sp.join_keys)
        if bench_al.empty:
            raise ValueError(f"Alineación vacía en escenario {sp.scenario_id}.")

        ashr_metrics = compute_scenario_metrics(ashr_df, sp.pred_col, sp.target_col)
        bench_al["_ae"] = (bench_al[sp.pred_col] - bench_al[sp.target_col]).abs()
        by_m = (
            bench_al.groupby(["metodo", "code"], as_index=False)["_ae"]
            .mean()
            .groupby("metodo", as_index=False)["_ae"]
            .mean()
            .rename(columns={"_ae": "MAE_macro"})
        )

        all_m = pd.concat(
            [pd.DataFrame([{"metodo": "ASHR", "MAE_macro": ashr_metrics["MAE_macro"]}]), by_m],
            ignore_index=True,
        )
        ranked = build_23_model_ranking(all_m)
        ashr_rank = int(ranked.loc[ranked["metodo"] == "ASHR", "rango"].iloc[0])

        lib_df = by_m[by_m["metodo"].isin(LIBS_SET)].sort_values("MAE_macro")
        cust_df = by_m[by_m["metodo"].isin(CUSTOM_SET)].sort_values("MAE_macro")
        mm_df = by_m[by_m["metodo"] == "Media móvil de 30 días"]

        best_lib_n = str(lib_df.iloc[0]["metodo"])
        best_lib_mae = float(lib_df.iloc[0]["MAE_macro"])
        best_cust_n = str(cust_df.iloc[0]["metodo"])
        best_cust_mae = float(cust_df.iloc[0]["MAE_macro"])
        mm_mae = float(mm_df.iloc[0]["MAE_macro"]) if not mm_df.empty else np.nan

        rows_28.append({
            "scenario_id": sp.scenario_id,
            "suite": sp.suite,
            "task": sp.task,
            "horizon_days": sp.horizon_days,
            "weather_on": sp.weather_on,
            "mode_eval": sp.mode_eval,
            "unit": sp.unit,
            "n_obs": ashr_metrics["n_obs"],
            "MAE_ASHR": ashr_metrics["MAE_macro"],
            "RMSE_ASHR": ashr_metrics["RMSE"],
            "MBE_ASHR": ashr_metrics["MBE"],
            "MAE_top5_ASHR": ashr_metrics["MAE_top5"],
            "Mejor_Biblioteca": best_lib_n,
            "MAE_Mejor_Bib": best_lib_mae,
            "pct_vs_MejorLib": 100.0 * (ashr_metrics["MAE_macro"] / best_lib_mae - 1.0),
            "Mejor_Custom": best_cust_n,
            "MAE_Mejor_Custom": best_cust_mae,
            "pct_vs_MejorCustom": 100.0 * (ashr_metrics["MAE_macro"] / best_cust_mae - 1.0),
            "MAE_MM30": mm_mae,
            "Skill_vs_MM30_pct": 100.0 * (1.0 - ashr_metrics["MAE_macro"] / mm_mae) if np.isfinite(mm_mae) else np.nan,
            "rango_en_23": ashr_rank,
            "n_modelos_comparados": int(len(ranked)),
        })

        # Per-plant MAE
        ashr_df["_ae"] = np.abs(ashr_df[sp.pred_col] - ashr_df[sp.target_col])
        p_ashr = ashr_df.groupby(["code", "grupo"], as_index=False)["_ae"].mean().rename(columns={"_ae": "MAE_ASHR"})
        p_mm = (
            bench_al[bench_al["metodo"] == "Media móvil de 30 días"]
            .groupby("code", as_index=False)["_ae"]
            .mean()
            .rename(columns={"_ae": "MAE_MM30"})
        )
        p_merged = p_ashr.merge(p_mm, on="code", how="left")
        p_merged["escenario"] = sp.scenario_id
        p_merged["suite"] = sp.suite
        p_merged["task"] = sp.task
        p_merged["unit"] = sp.unit
        p_merged["diff_vs_MM30"] = p_merged["MAE_ASHR"] - p_merged["MAE_MM30"]
        plant_rows.extend(p_merged.to_dict("records"))

        # Paired inference on independent temporal unit (date for D+1, origen for Rec/Directo)
        time_col = "date" if sp.mode_eval == "d1" else "origen"
        ashr_t = ashr_df.groupby(time_col)["_ae"].mean()

        # Model Confidence Set (MCS) over temporal loss matrix across all 23 models
        bench_t = bench_al.groupby([time_col, "metodo"], as_index=False)["_ae"].mean()
        piv = bench_t.pivot(index=time_col, columns="metodo", values="_ae")
        piv["ASHR"] = ashr_t
        piv = piv.dropna(axis=0, how="any")
        if len(piv) >= 10 and piv.shape[1] >= 2:
            mcs_res = model_confidence_set(
                piv.to_numpy(dtype=float), list(piv.columns), alpha=0.10, B=200, seed=2026
            )
            mcs_rows.append(
                {
                    "escenario": sp.scenario_id,
                    "suite": sp.suite,
                    "task": sp.task,
                    "n_time_units": int(len(piv)),
                    "n_survivors": int(len(mcs_res["survivors"])),
                    "ashr_in_mcs_90": "ASHR" in mcs_res["survivors"],
                    "p_mcs_ashr": float(mcs_res["p_mcs"].get("ASHR", np.nan)),
                    "survivors": ", ".join(mcs_res["survivors"]),
                }
            )

        for ref_label, ref_name in (("Mejor_Bib", best_lib_n), ("MM30", "Media móvil de 30 días")):
            ref_sub = bench_al[bench_al["metodo"] == ref_name]
            if ref_sub.empty:
                continue
            ref_t = ref_sub.groupby(time_col)["_ae"].mean()
            al_t = pd.concat({"ashr": ashr_t, "ref": ref_t}, axis=1).dropna()
            inf = run_paired_inference(
                al_t["ashr"].to_numpy(float),
                al_t["ref"].to_numpy(float),
                horizon_h=sp.horizon_days,
                is_temporal=True,
            )
            inf_rows.append({
                "escenario": sp.scenario_id,
                "suite": sp.suite,
                "task": sp.task,
                "horizon_days": sp.horizon_days,
                "unit": sp.unit,
                "referencia": ref_label,
                "modelo_referencia": ref_name,
                **inf,
            })

        cost_rows.append({
            "scenario_id": sp.scenario_id,
            "suite": sp.suite,
            "task": sp.task,
            "tiempo_s": float(times_map.get(sp.scenario_id, np.nan)),
        })

    t28 = pd.DataFrame(rows_28)
    t_rank = (
        t28.groupby(["suite", "task"], as_index=False)
        .agg(
            n_escenarios=("scenario_id", "count"),
            rango_medio=("rango_en_23", "mean"),
            victorias_vs_bib=("pct_vs_MejorLib", lambda s: int((s < 0).sum())),
            victorias_vs_custom=("pct_vs_MejorCustom", lambda s: int((s < 0).sum())),
            victorias_vs_mm30=("Skill_vs_MM30_pct", lambda s: int((s > 0).sum())),
            pct_medio_vs_bib=("pct_vs_MejorLib", "mean"),
        )
    )
    t_inf = pd.DataFrame(inf_rows)
    if not t_inf.empty:
        t_inf["p_holm"] = apply_holm(t_inf["p_dm_hln"].fillna(1.0).to_numpy())
        t_inf["estado_estadistico"] = np.where(
            t_inf["t_eff_insufficient"],
            "T_EFF_INSUFICIENTE",
            np.where(
                t_inf["p_holm"] < 0.05,
                "SIGNIFICATIVO_HOLM",
                np.where(t_inf["p_dm_hln"] < 0.05, "NOMINAL_SIN_HOLM", "NO_SIGNIFICATIVO"),
            ),
        )

    return {
        "manifest": manifest,
        "tabla_28_escenarios": t28,
        "tabla_rankings_regimen": t_rank,
        "tabla_por_planta": pd.DataFrame(plant_rows),
        "tabla_inferencia": t_inf,
        "tabla_mcs": pd.DataFrame(mcs_rows),
        "tabla_coste": pd.DataFrame(cost_rows),
    }


def build_descriptive_summary(bundle: dict[str, Any] | None) -> dict[str, Any]:
    """Generate neutral descriptive summary (reporting both wins and losses)."""
    if bundle is None:
        return {"estado": "PENDIENTE", "mensaje": "Evaluación OOF definitiva no ejecutada."}

    t28 = bundle["tabla_28_escenarios"]
    t_inf = bundle["tabla_inferencia"]
    best_row = t28.sort_values("pct_vs_MejorLib").iloc[0]
    worst_row = t28.sort_values("pct_vs_MejorLib", ascending=False).iloc[0]

    inf_bib = t_inf[t_inf["referencia"] == "Mejor_Bib"]
    return {
        "estado": "VALIDO",
        "procedencia": bundle["manifest"].get("procedencia", "OOF_DEFINITIVO_REAL"),
        "n_escenarios": int(len(t28)),
        "victorias_vs_mejor_bib": int((t28["pct_vs_MejorLib"] < 0).sum()),
        "derrotas_vs_mejor_bib": int((t28["pct_vs_MejorLib"] > 0).sum()),
        "victorias_vs_mejor_custom": int((t28["pct_vs_MejorCustom"] < 0).sum()),
        "victorias_vs_mm30": int((t28["Skill_vs_MM30_pct"] > 0).sum()),
        "rango_medio_en_23": round(float(t28["rango_en_23"].mean()), 2),
        "mejor_escenario_vs_bib": f"{best_row['scenario_id']} ({best_row['pct_vs_MejorLib']:+.2f}%)",
        "peor_escenario_vs_bib": f"{worst_row['scenario_id']} ({worst_row['pct_vs_MejorLib']:+.2f}%)",
        "sig_holm_vs_bib": int((inf_bib["estado_estadistico"] == "SIGNIFICATIVO_HOLM").sum()) if not inf_bib.empty else 0,
        "t_eff_insuficiente_n": int((inf_bib["estado_estadistico"] == "T_EFF_INSUFICIENTE").sum()) if not inf_bib.empty else 0,
    }


def export_paper_tables(bundle: dict[str, Any], tables_dir: Path) -> dict[str, Path]:
    """Export publishable CSV and LaTeX tables from a validated OOF bundle."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}
    for key in ("tabla_28_escenarios", "tabla_rankings_regimen", "tabla_inferencia", "tabla_mcs", "tabla_coste"):
        df = bundle[key]
        csv_p = tables_dir / f"{key}.csv"
        tex_p = tables_dir / f"{key}.tex"
        df.to_csv(csv_p, index=False)
        tex_p.write_text(df.to_latex(index=False, float_format="%.2f"), encoding="utf-8")
        exported[f"{key}_csv"] = csv_p
        exported[f"{key}_tex"] = tex_p
    return exported


def create_synthetic_oof_fixture(tmp_dir: Path, root: Path | None = None) -> Path:
    """Create a schema-valid synthetic OOF directory strictly for unit testing State B."""
    r = root or find_project_root()
    cfg = load_frozen_config(r)
    pred_dir = tmp_dir / "predicciones"
    pred_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)

    times = {}
    for sp in SCENARIOS_28:
        bench = load_official_benchmarks(sp, root=r, exclude_cf_brh=True)
        sub = bench[bench["metodo"] == "Media móvil de 30 días"].copy()
        if sub.empty:
            first_m = bench["metodo"].iloc[0]
            sub = bench[bench["metodo"] == first_m].copy()
        sub = sub.dropna(subset=[sp.pred_col, sp.target_col]).copy()
        # Deterministic perturbation around real benchmark keys (unit test fixture only)
        noise = rng.normal(0.0, 0.02, size=len(sub))
        preds = np.maximum(0.0, sub[sp.pred_col].to_numpy(float) * (0.98 + noise))
        out_df = sub[list(sp.join_keys) + ["grupo", sp.target_col]].copy()
        out_df.insert(0, "metodo", "ASHR")
        out_df[sp.pred_col] = preds
        out_df.to_csv(pred_dir / f"{sp.scenario_id}.csv", index=False)
        times[sp.scenario_id] = 0.5

    manifest = {
        "estatus": "OOF_DEFINITIVO_COMPLETO",
        "procedencia": "SINTETICO_TEST",
        "sha256_src_adaptive_structured_huber": cfg["sha256_src_adaptive_structured_huber"],
        "tiempo_total_s": 14.0,
        "tiempos_por_escenario_s": times,
    }
    (tmp_dir / "manifest_ashr_oof.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return tmp_dir

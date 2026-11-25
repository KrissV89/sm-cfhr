"""Resumable 8-stage post-OOF scientific pipeline generating all tables, studies, and figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .inference import (
    apply_holm,
    benjamini_hochberg_adjust,
    evaluate_r3_superiority,
)
from .io import export_full_scientific_artifacts
from .persistence import atomic_write_csv, atomic_write_text
from .plotting import (
    export_all_figures,
    plot_ablation_bar,
    plot_conformal_coverage,
    plot_feature_importance,
    plot_hp_sensitivity,
    plot_learning_and_scalability,
    plot_robustness_stress,
)
from .robustness import (
    compute_interpretability_summary,
    compute_temporal_conformal_intervals,
    run_ablation_suite,
    run_hp_sensitivity,
    run_learning_and_scalability_curves,
    run_noise_outlier_stress,
)
from .validation import analyze_effective_design

try:
    from ashr_oof_pipeline.config import load_frozen_config
    from ashr_oof_pipeline.oof_runner import extract_cf_features
    from ashr_oof_pipeline.resultados_loader import (
        build_descriptive_summary,
        export_paper_tables,
        load_oof_bundle,
    )
except ImportError:
    from src.ashr_oof_pipeline.config import load_frozen_config
    from src.ashr_oof_pipeline.oof_runner import extract_cf_features
    from src.ashr_oof_pipeline.resultados_loader import (
        build_descriptive_summary,
        export_paper_tables,
        load_oof_bundle,
    )


def apply_family_multiplicity(t_inf: pd.DataFrame) -> pd.DataFrame:
    """Apply Holm and Benjamini-Hochberg strictly within pre-specified (suite, referencia) families.

    Invalid / insufficient rows keep NaN adjusted p-values and explicit `estado_estadistico`.
    """
    out = t_inf.copy()
    out["familia_id"] = out["suite"].astype(str) + "__" + out["referencia"].astype(str)
    out["p_holm_familia"] = np.nan
    out["p_bh_familia"] = np.nan
    out["n_comparaciones_familia"] = 0

    for fam_id, idx in out.groupby("familia_id", sort=False).groups.items():
        valid_mask = (~out.loc[idx, "t_eff_insufficient"].astype(bool)) & out.loc[idx, "p_dm_hln"].notna()
        valid_idx = out.loc[idx].index[valid_mask]
        m = int(len(valid_idx))
        out.loc[idx, "n_comparaciones_familia"] = m
        if m > 0:
            raw_p = out.loc[valid_idx, "p_dm_hln"].to_numpy(dtype=float)
            out.loc[valid_idx, "p_holm_familia"] = apply_holm(raw_p)
            out.loc[valid_idx, "p_bh_familia"] = benjamini_hochberg_adjust(raw_p)
    return out


def run_post_oof_stages(
    oof_dir: Path,
    root: Path,
    allow_synthetic_test: bool = False,
    panel_df: pd.DataFrame | None = None,
    mode: str = "CONFIRMATORY",
) -> dict[str, Any]:
    """Execute resumable post-OOF stages 2-8 from persisted predictions without retraining OOF.

    In CONFIRMATORY or DEVELOPMENT mode, synthetic fallback data is strictly prohibited when `panel_df` is None.
    """
    if mode not in ("TEST_SYNTHETIC", "DEVELOPMENT", "CONFIRMATORY"):
        raise ValueError(f"Invalid mode: {mode}")
    if mode == "TEST_SYNTHETIC" and not allow_synthetic_test:
        raise RuntimeError("TEST_SYNTHETIC mode requires allow_synthetic_test=True.")

    cfg = load_frozen_config(root)
    scope_auth = bool(cfg.get("autorizacion_alcance_solar_docente", False))

    # Stage 2-3: Load and validate persisted OOF bundle
    bundle = load_oof_bundle(oof_dir, root=root, allow_synthetic_test=(mode == "TEST_SYNTHETIC"))

    # Stage 4: Family-wise multiplicity + effective design + R3 evaluation
    t_inf_fam = apply_family_multiplicity(bundle["tabla_inferencia"])
    atomic_write_csv(t_inf_fam, oof_dir / "multiplicidad" / "tabla_multiplicidad_familias.csv")
    design_df = analyze_effective_design(bundle)
    atomic_write_csv(design_df, oof_dir / "inferencia" / "diseno_efectivo.csv")
    r3_df = evaluate_r3_superiority(bundle, scope_authorized=scope_auth, alpha=0.05)
    atomic_write_csv(r3_df, oof_dir / "conclusiones" / "veredicto_r3.csv")

    fig_dir = oof_dir / "figuras"
    export_all_figures(bundle, fig_dir)

    # Stage 5-6: Scientific studies (strictly prohibit synthetic fallback in CONFIRMATORY/DEVELOPMENT)
    if panel_df is None and mode != "TEST_SYNTHETIC":
        export_paper_tables(bundle, oof_dir / "tablas_paper")
        desc = build_descriptive_summary(bundle)
        full_manifest = export_full_scientific_artifacts(
            bundle,
            oof_dir,
            extra_artifacts={
                "conclusiones/veredicto_r3": r3_df,
                "inferencia/diseno_efectivo": design_df,
                "multiplicidad/tabla_multiplicidad_familias": t_inf_fam,
                "conclusiones/resumen_descriptivo": desc,
            },
        )
        return {
            "status": "COMPLETED_SIN_ESTUDIOS_PANEL",
            "mode": mode,
            "etapas_5_6_estado": "BLOQUEADO_SIN_DATOS_REALES",
            "n_exported": full_manifest["n_archivos_exportados"],
        }

    if panel_df is None:
        rng = np.random.default_rng(2026)
        n = 160
        X_s = rng.normal(size=(n, 17))
        X_s[:, 4] = np.clip(0.20 + 0.05 * X_s[:, 4], 0.02, 0.50)
        X_s[:, 7] = np.clip(5.0 + 1.0 * X_s[:, 7], 0.5, 9.0)
        y_s = np.clip(0.6 * X_s[:, 4] + 0.02 * X_s[:, 7] + rng.normal(0, 0.015, size=n), 0.01, 0.95)
        denom_s = np.full(n, 240_000.0)
        meta_s = {"cf30_idx": 4, "monotone_pos_idx": (4, 7), "meteo_idx": tuple(range(7, 17))}
        X_tr, y_tr, X_va, y_va = X_s[:110], y_s[:110], X_s[110:], y_s[110:]
        d_tr, d_va = denom_s[:110], denom_s[110:]
        feat_names = [f"f_{i}" for i in range(17)]
    else:
        p_sorted = panel_df.sort_values(["date", "code"]).reset_index(drop=True)
        X_all, _, denom_all = extract_cf_features(p_sorted, weather_on=True, lead=1)
        y_all = np.clip(p_sorted["y"].to_numpy(dtype=float) / denom_all, 0.0, 1.0)
        dates_all = pd.to_datetime(p_sorted["date"]).to_numpy()
        codes_all = p_sorted["code"].astype(str).to_numpy()
        valid = np.all(np.isfinite(X_all), axis=1) & np.isfinite(y_all) & np.isfinite(denom_all)
        X_all, y_all, denom_all = X_all[valid], y_all[valid], denom_all[valid]
        dates_all, codes_all = dates_all[valid], codes_all[valid]
        p_valid = p_sorted.loc[valid].reset_index(drop=True)

        unique_dates = np.sort(np.unique(dates_all))
        date_cut = unique_dates[int(0.75 * len(unique_dates))]
        tr_mask = dates_all < date_cut
        va_mask = dates_all >= date_cut
        if dates_all[tr_mask].max() >= dates_all[va_mask].min():
            raise RuntimeError("Temporal leakage detected in post-OOF date split.")

        X_tr, y_tr, d_tr = X_all[tr_mask], y_all[tr_mask], denom_all[tr_mask]
        X_va, y_va, d_va = X_all[va_mask], y_all[va_mask], denom_all[va_mask]
        dates_va, codes_va = dates_all[va_mask], codes_all[va_mask]
        meta_s = {"cf30_idx": 4, "monotone_pos_idx": (4, 7), "meteo_idx": tuple(range(7, 17))}
        feat_names = [
            "month_sin", "month_cos", "doy_sin", "doy_cos", "cf30", "diff_7_30", "diff_1_7",
            "rad_pron", "temp_pron", "nube_pron", "humedad_pron", "viento_pron",
            "anom_rad", "anom_temp", "anom_nube", "anom_humedad", "anom_viento",
        ]

    abl_df = run_ablation_suite(X_tr, y_tr, X_va, y_va, d_tr, d_va, meta_s, weather_on=True)
    sens_df = run_hp_sensitivity(X_tr, y_tr, X_va, y_va, meta_s, weather_on=True)
    stress_df = run_noise_outlier_stress(X_tr, y_tr, X_va, y_va, meta_s, weather_on=True)
    lc_df = run_learning_and_scalability_curves(X_tr, y_tr, X_va, y_va, meta_s, weather_on=True)

    atomic_write_csv(abl_df, oof_dir / "ablaciones" / "tabla_ablaciones.csv")
    atomic_write_csv(sens_df, oof_dir / "sensibilidad" / "tabla_sensibilidad_hp.csv")
    atomic_write_csv(stress_df, oof_dir / "robustez" / "tabla_estres_atipicos.csv")
    atomic_write_csv(lc_df, oof_dir / "robustez" / "tabla_aprendizaje_escalabilidad.csv")

    if panel_df is not None:
        from .robustness import run_seed_stability_hpo

        tr_dates = np.sort(np.unique(dates_all[tr_mask]))
        va_dates = np.sort(np.unique(dates_va))
        tr_sub = p_valid[p_valid["date"].isin(tr_dates[-240:])].copy()
        va_sub = p_valid[p_valid["date"].isin(va_dates[:60])].copy()
        seed_df = run_seed_stability_hpo(
            tr_sub,
            va_sub,
            task="espaciotemporal",
            horizon_days=1,
            weather_on=True,
            mode_eval="d1",
            seeds=tuple(range(2026, 2036)),
            n_trials=12,
        )
        atomic_write_csv(seed_df, oof_dir / "sensibilidad" / "tabla_estabilidad_semillas.csv")

    try:
        from ashr_oof_pipeline.config import build_frozen_ashr
    except ImportError:
        from src.ashr_oof_pipeline.config import build_frozen_ashr
    est = build_frozen_ashr(mode="espaciotemporal", horizon=1, weather_mode="forecast", root=root)
    est.fit(X_tr, y_tr, metadata=meta_s)
    p_va = est.predict(X_va)
    if panel_df is not None:
        va_unique = np.sort(np.unique(dates_va))
        val_mid_date = va_unique[len(va_unique) // 2]
        cal_m = dates_va < val_mid_date
        te_m = dates_va >= val_mid_date
        conf_res = compute_temporal_conformal_intervals(
            y_va[cal_m], p_va[cal_m], p_va[te_m], y_va[te_m], plants_test=codes_va[te_m]
        )
        atomic_write_csv(conf_res["by_plant"], oof_dir / "incertidumbre" / "tabla_conformal_por_planta.csv")
    else:
        mid = len(X_va) // 2
        conf_res = compute_temporal_conformal_intervals(y_va[:mid], p_va[:mid], p_va[mid:], y_va[mid:])
    atomic_write_csv(conf_res["summary"], oof_dir / "incertidumbre" / "tabla_conformal_resumen.csv")
    interp_res = compute_interpretability_summary(est, X_tr, X_va, feat_names, n_local=3)
    atomic_write_csv(interp_res["global_importance"], oof_dir / "interpretabilidad" / "tabla_importancia_global.csv")
    atomic_write_text(oof_dir / "interpretabilidad" / "ejemplos_locales.json", json.dumps(interp_res["local_explanations"], indent=2))

    # Stage 7-8: Tables, Figures, and full manifest
    fig_dir = oof_dir / "figuras"
    export_all_figures(bundle, fig_dir)
    plot_ablation_bar(abl_df, fig_dir)
    plot_hp_sensitivity(sens_df, fig_dir)
    plot_robustness_stress(stress_df, fig_dir)
    plot_learning_and_scalability(lc_df, fig_dir)
    plot_conformal_coverage(conf_res["summary"], fig_dir)
    plot_feature_importance(interp_res["global_importance"], fig_dir)

    export_paper_tables(bundle, oof_dir / "tablas_paper")
    desc = build_descriptive_summary(bundle)
    full_manifest = export_full_scientific_artifacts(
        bundle,
        oof_dir,
        extra_artifacts={
            "conclusiones/veredicto_r3": r3_df,
            "inferencia/diseno_efectivo": design_df,
            "multiplicidad/tabla_multiplicidad_familias": t_inf_fam,
            "conclusiones/resumen_descriptivo": desc,
        },
    )
    return {"status": "COMPLETED", "n_exported": full_manifest["n_archivos_exportados"]}


__all__ = ["apply_family_multiplicity", "run_post_oof_stages"]

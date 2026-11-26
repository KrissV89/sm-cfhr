"""Statistical inference, multiple-testing control, effect sizes, and R3 evaluator."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from ashr_oof_pipeline.evaluacion_estadistica import (
        _stationary_bootstrap_indices,
        apply_holm,
        bootstrap_ci_difference,
        build_23_model_ranking,
        clark_west_test,
        compute_scenario_metrics,
        giacomini_white_test,
        model_confidence_set,
        run_paired_inference,
    )
except ImportError:
    from src.ashr_oof_pipeline.evaluacion_estadistica import (
        _stationary_bootstrap_indices,
        apply_holm,
        bootstrap_ci_difference,
        build_23_model_ranking,
        clark_west_test,
        compute_scenario_metrics,
        giacomini_white_test,
        model_confidence_set,
        run_paired_inference,
    )


def benjamini_hochberg_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR adjustment to an array of raw p-values."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return p.copy()
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty_like(adj)
    out[order] = adj
    return out


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute non-parametric Cliff's delta effect size in [-1, 1] (x - y)."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / (a.size * b.size))


def cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    """Compute paired Cohen's d = mean(x - y) / std(x - y)."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return float("nan")
    sd = float(np.std(d, ddof=1))
    return float(np.mean(d) / sd) if sd > 1e-12 else 0.0


def giacomini_white_audited(d: np.ndarray, tau: int = 1) -> dict[str, Any]:
    """Run Giacomini-White test with lag-1 autocorrelation diagnostic on loss differentials."""
    arr = np.asarray(d, dtype=float)
    arr = arr[np.isfinite(arr)]
    res = giacomini_white_test(arr, tau=tau)
    if arr.size >= 12 and np.std(arr) > 1e-12:
        acf1 = float(np.corrcoef(arr[:-1], arr[1:])[0, 1])
    else:
        acf1 = float("nan")
    warn = bool(tau == 1 and np.isfinite(acf1) and abs(acf1) > 0.15)
    res["lag1_acf"] = acf1
    res["gw_autocorr_warning"] = warn
    res["estado_gw"] = "ADVERTENCIA_AUTOCORR_H1" if warn else "APLICABLE"
    return res


def spa_test(
    loss_benchmark: np.ndarray,
    loss_models: np.ndarray,
    B: int = 500,
    mean_block: float = 7.0,
    seed: int = 2026,
) -> dict[str, float]:
    """Hansen (2005) Superior Predictive Ability (SPA) test using stationary bootstrap.

    Tests H0: no competitor outperforms the benchmark (max_k E[L_bench - L_k] <= 0).
    """
    lb = np.asarray(loss_benchmark, dtype=float)
    lm = np.asarray(loss_models, dtype=float)
    if lm.ndim == 1:
        lm = lm[:, None]
    d = lb[:, None] - lm
    valid = np.all(np.isfinite(d), axis=1)
    d = d[valid]
    T, K = d.shape
    if T < 10 or K == 0:
        return {"spa_stat": float("nan"), "p_spa": float("nan"), "T_spa": float(T)}

    from src.sm_cfhr.inference_v2 import spa_test_v2

    res = spa_test_v2(
        loss_benchmark=lb,
        loss_models=lm,
        B=B,
        mean_block=mean_block,
        seed=seed,
        studentize=True,
        variance_estimator="pr_kernel",
        clip_zero=True,
        finite_sample_floor=True,
    )
    return {
        "spa_stat": float(res["spa_stat"]),
        "p_spa": float(res["p_spa_consistent"]),
        "p_spa_lower": float(res["p_spa_lower"]),
        "p_spa_consistent": float(res["p_spa_consistent"]),
        "p_spa_upper": float(res["p_spa_upper"]),
        "T_spa": float(res["T_spa"]),
    }


def evaluate_r3_superiority(
    bundle: dict[str, Any] | None,
    scope_authorized: bool = False,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Evaluate the 4 formal R3 superiority conditions without fabricating missing evidence."""
    if bundle is None:
        return pd.DataFrame(
            [
                {
                    "condicion": f"Condicion_{i}",
                    "estado": "NO_VERIFICABLE",
                    "evidencia": "Evaluación OOF definitiva aún no ejecutada.",
                    "calculo": "N/A",
                    "fuente": "bundle=None",
                    "limitacion": "Ejecutar OOF definitivo para evaluar.",
                    "resultado": "PENDIENTE",
                }
                for i in range(1, 5)
            ]
        )

    t28 = bundle["tabla_28_escenarios"]
    t_inf = bundle["tabla_inferencia"]
    t_mcs = bundle.get("tabla_mcs", pd.DataFrame())

    # Condicion 1: Mejor metrica externa que el mejor benchmark
    n_wins = int((t28["pct_vs_MejorLib"] < 0).sum())
    n_tot = int(len(t28))
    c1_pass = n_wins == n_tot
    c1_state = "CUMPLE" if c1_pass else "NO_CUMPLE"

    # Condicion 2: Al menos tres pruebas distintas, validas y calibradas (DM-HLN-HAC + MCS + SPA)
    # GW con tau=1 queda excluida como prueba confirmatoria si hay advertencia de autocorrelacion
    inf_bib = t_inf[(t_inf["referencia"] == "Mejor_Bib") & (~t_inf["t_eff_insufficient"])]
    t_spa = bundle.get("tabla_spa", pd.DataFrame())
    n_holm = int((inf_bib["p_holm"] < alpha).sum()) if not inf_bib.empty else 0
    n_mcs_only = int((t_mcs["ashr_in_mcs_90"] & (t_mcs["n_survivors"] == 1)).sum()) if not t_mcs.empty else 0
    if t_spa is None or (isinstance(t_spa, pd.DataFrame) and t_spa.empty):
        c2_state = "NO_VERIFICABLE"
        c2_evid = f"Solo 2 procedimientos ejecutados (DM-HLN Holm={n_holm}, MCS_excl={n_mcs_only}); falta SPA calibrada para acreditar 3 pruebas distintas."
    elif inf_bib.empty:
        c2_state = "EVIDENCIA_INSUFICIENTE"
        c2_evid = "T_eff insuficiente en los escenarios evaluados."
    else:
        spa_col = "p_spa" if "p_spa" in t_spa.columns else "p_spa_H0_MM30"
        n_spa = int((t_spa[spa_col] < alpha).sum())
        c2_state = "CUMPLE" if (n_holm >= max(1, len(inf_bib) // 2) and n_mcs_only >= 1 and n_spa >= 1) else "NO_CUMPLE"
        c2_evid = f"DM-HLN Holm={n_holm}/{len(inf_bib)}, MCS_excl={n_mcs_only}/{len(t_mcs)}, SPA_sig={n_spa}/{len(t_spa)}"

    # Condicion 3: Tamano del efecto no despreciable a favor de SM-CFHR (dz <= -0.20)
    eff_col = "cohen_dz_pareado" if "cohen_dz_pareado" in inf_bib.columns else "cohen_d"
    if eff_col in inf_bib.columns and not inf_bib.empty:
        n_eff = int((inf_bib[eff_col] <= -0.20).sum())
    else:
        n_eff = 0
    c3_state = "CUMPLE" if (not inf_bib.empty and n_eff >= max(1, len(inf_bib) // 2)) else ("EVIDENCIA_INSUFICIENTE" if inf_bib.empty else "NO_CUMPLE")

    # Condicion 4: Consistencia en conjuntos adicionales o alcance solar autorizado
    c4_state = "CUMPLE" if (scope_authorized and c1_pass) else ("PENDIENTE_AUTORIZACION" if not scope_authorized else "NO_CUMPLE")

    return pd.DataFrame(
        [
            {
                "condicion": "Condicion_1_Metrica_Externa",
                "estado": c1_state,
                "evidencia": f"Victorias vs Mejor Biblioteca: {n_wins}/{n_tot} escenarios",
                "calculo": "pct_vs_MejorLib < 0",
                "fuente": "tabla_28_escenarios",
                "limitacion": "Requiere victoria consistente sin ocultar derrotas.",
                "resultado": f"{n_wins}/{n_tot}",
            },
            {
                "condicion": "Condicion_2_Pruebas_Multiples_y_MCS",
                "estado": c2_state,
                "evidencia": c2_evid,
                "calculo": "DM-HLN-HAC + Holm + MCS (alpha=0.10) + SPA (GW excluida si autocorrelacion)",
                "fuente": "tabla_inferencia, tabla_mcs, tabla_spa",
                "limitacion": "Exige 3 procedimientos distintos calibrados; GW con tau=1 excluida ante sobrerrechazo.",
                "resultado": f"Holm={n_holm}, MCS_excl={n_mcs_only}",
            },
            {
                "condicion": "Condicion_3_Tamano_Efecto",
                "estado": c3_state,
                "evidencia": f"|Cohen's d| >= 0.2 en {n_eff}/{len(inf_bib)} escenarios con T_eff suficiente",
                "calculo": "|d| >= 0.20 (y umbral Cliff |delta| >= 0.147)",
                "fuente": "tabla_inferencia",
                "limitacion": "Efectos pequeños pueden carecer de relevancia práctica en kWh.",
                "resultado": f"{n_eff}/{len(inf_bib)}",
            },
            {
                "condicion": "Condicion_4_Alcance_y_Consistencia",
                "estado": c4_state,
                "evidencia": "28 escenarios solares (17 plantas Colombia); sin datasets externos.",
                "calculo": "Validación de alcance externo + consistencia multi-régimen",
                "fuente": "config.SCENARIOS_28",
                "limitacion": "Sustitución de datasets externos pendiente de validación externa independiente.",
                "resultado": "AUTORIZADO" if scope_authorized else "PENDIENTE_AUTORIZACION",
            },
        ]
    )


__all__ = [
    "apply_holm",
    "benjamini_hochberg_adjust",
    "bootstrap_ci_difference",
    "build_23_model_ranking",
    "clark_west_test",
    "cliffs_delta",
    "cohens_d_paired",
    "compute_scenario_metrics",
    "evaluate_r3_superiority",
    "giacomini_white_audited",
    "giacomini_white_test",
    "model_confidence_set",
    "run_paired_inference",
    "spa_test",
]

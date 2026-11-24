"""Official statistical evaluation protocol for ASHR OOF comparisons.

Integrates DM-HLN-HAC, Giacomini-White CPA, stationary-bootstrap MCS,
paired effect sizes, and Holm family-wise correction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

try:
    from scripts.evaluacion import inferencia_estadistica_v2 as inf2
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path

    _root = _Path(__file__).resolve().parents[2]
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))
    from scripts.evaluacion import inferencia_estadistica_v2 as inf2


def compute_scenario_metrics(
    df: pd.DataFrame, pred_col: str, target_col: str
) -> dict[str, float]:
    """Compute macro MAE (mean across plants), RMSE, MBE, and 95th-percentile tail MAE."""
    err = df[pred_col].to_numpy(dtype=float) - df[target_col].to_numpy(dtype=float)
    ae = np.abs(err)
    macro_mae = float(df.assign(_ae=ae).groupby("code")["_ae"].mean().mean())
    rmse = float(np.sqrt(np.mean(err * err)))
    mbe = float(np.mean(err))
    q95 = float(np.quantile(ae, 0.95))
    top5 = ae[ae >= q95]
    return {
        "n_obs": int(len(df)),
        "MAE_macro": macro_mae,
        "RMSE": rmse,
        "MBE": mbe,
        "MAE_top5": float(np.mean(top5)) if len(top5) else np.nan,
    }


def build_23_model_ranking(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Rank ASHR + 22 official models (excluding CF-BRH) by MAE_macro ascending."""
    out = metrics_df[metrics_df["metodo"] != "CF-BRH"].sort_values("MAE_macro").reset_index(drop=True)
    out["rango"] = np.arange(1, len(out) + 1)
    return out


def giacomini_white_test(d: np.ndarray, tau: int = 1) -> dict[str, float]:
    """Giacomini-White (2006) unconditional/conditional predictive ability test with HAC."""
    arr = np.asarray(d, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 10:
        return {"gw_stat": np.nan, "p_gw": np.nan, "T_gw": int(arr.size)}

    y_dep = arr[1:]
    Z = np.column_stack([np.ones(len(y_dep)), arr[:-1]])
    g = Z * y_dep[:, None]
    T = g.shape[0]
    g_mean = np.mean(g, axis=0)
    gc = g - g_mean
    L = max(0, int(tau) - 1)
    omega = (gc.T @ gc) / T
    for k in range(1, min(L, T - 2) + 1):
        wk = 1.0 - k / (L + 1.0)
        gk = (gc[k:].T @ gc[:-k]) / T
        omega = omega + wk * (gk + gk.T)
    stat = float(T * g_mean @ np.linalg.pinv(omega) @ g_mean)
    p_val = float(chi2.sf(max(0.0, stat), df=2))
    return {"gw_stat": stat, "p_gw": p_val, "T_gw": int(T)}


def _stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / max(mean_block, 1.0)
    idx = np.empty(n, dtype=int)
    idx[0] = int(rng.integers(0, n))
    for i in range(1, n):
        if rng.random() < p:
            idx[i] = int(rng.integers(0, n))
        else:
            idx[i] = (idx[i - 1] + 1) % n
    return idx


def model_confidence_set(
    loss_matrix: np.ndarray,
    model_names: list[str],
    alpha: float = 0.10,
    B: int = 500,
    mean_block: float = 7.0,
    seed: int = 2026,
) -> dict[str, Any]:
    """Hansen-Lunde-Nason Model Confidence Set (T_max statistic) with stationary bootstrap."""
    L = np.asarray(loss_matrix, dtype=float)
    T, M = L.shape
    rng = np.random.default_rng(seed)
    boot_idx = np.array([_stationary_bootstrap_indices(T, mean_block, rng) for _ in range(B)])

    active = list(range(M))
    eliminated: list[tuple[str, float]] = []
    pvals: dict[str, float] = {m: 1.0 for m in model_names}
    p_running = 0.0

    while len(active) > 1:
        sub = L[:, active]
        d_ij = sub - sub.mean(axis=1, keepdims=True)
        d_bar = d_ij.mean(axis=0)
        # Bootstrap centered relative losses
        boot_means = d_ij[boot_idx].mean(axis=1)  # (B, k)
        centered_boot = boot_means - d_bar[None, :]
        se = np.maximum(np.std(centered_boot, axis=0, ddof=1), 1e-12)
        t_obs = d_bar / se
        t_boot = np.abs(centered_boot) / se[None, :]
        stat_obs = float(np.max(t_obs))
        stat_boot = np.max(t_boot, axis=1)
        p_step = float(np.mean(stat_boot >= stat_obs))
        p_running = max(p_running, p_step)

        worst_local = int(np.argmax(t_obs))
        worst_idx = active[worst_local]
        pvals[model_names[worst_idx]] = p_running
        if p_running >= alpha:
            break
        eliminated.append((model_names[worst_idx], p_running))
        active.pop(worst_local)

    survivors = [model_names[i] for i in active]
    return {"survivors": survivors, "eliminated": eliminated, "p_mcs": pvals}


def run_paired_inference(
    loss_ashr: np.ndarray,
    loss_ref: np.ndarray,
    horizon_h: int = 1,
    is_temporal: bool = True,
    seed: int = 2026,
) -> dict[str, Any]:
    """Run paired statistical inference respecting temporal dependence and T_eff limits."""
    d = np.asarray(loss_ashr, dtype=float) - np.asarray(loss_ref, dtype=float)
    d = d[np.isfinite(d)]
    eff = inf2.paired_effect_sizes(loss_ashr, loss_ref)
    if not is_temporal:
        return {"T_eff": float(len(d)), "t_eff_insufficient": False, **eff}

    dm = inf2.dm_hln_hac(d, horizon_h=horizon_h)
    ci_lo, ci_hi = inf2.stationary_bootstrap_ci(d, mean_block=max(7.0, float(horizon_h)), B=1000, seed=seed)
    gw = giacomini_white_test(d, tau=horizon_h)
    t_eff = float(dm["T_eff"])
    return {
        "T_eff": t_eff,
        "t_eff_insufficient": bool(t_eff < 2.0 * horizon_h),
        "mean_diff": float(np.mean(d)),
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "dm_hln": dm["dm_hln"],
        "p_dm_hln": dm["p_dm_hln"],
        "gw_stat": gw["gw_stat"],
        "p_gw": gw["p_gw"],
        **eff,
    }


def apply_holm(pvalues: np.ndarray) -> np.ndarray:
    return inf2.holm_adjust(np.asarray(pvalues, dtype=float))


def clark_west_test(
    y_true: np.ndarray,
    pred_restricted: np.ndarray,
    pred_unrestricted: np.ndarray,
    tau: int = 1,
) -> dict[str, float]:
    """Clark-West (2007) nested model forecast comparison test with Newey-West HAC variance.

    Tests H0: equal MSPE (restricted nested model) vs H1: unrestricted model has lower MSPE.
    """
    y = np.asarray(y_true, dtype=float)
    p1 = np.asarray(pred_restricted, dtype=float)
    p2 = np.asarray(pred_unrestricted, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p1) & np.isfinite(p2)
    y, p1, p2 = y[mask], p1[mask], p2[mask]
    T = int(y.size)
    if T < 10:
        return {"cw_stat": np.nan, "p_cw_one_sided": np.nan, "T_cw": T}

    f = (y - p1) ** 2 - ((y - p2) ** 2 - (p1 - p2) ** 2)
    f_bar = float(np.mean(f))
    fc = f - f_bar
    L = max(0, int(tau) - 1)
    var_hac = float(np.dot(fc, fc) / T)
    for k in range(1, min(L, T - 2) + 1):
        wk = 1.0 - k / (L + 1.0)
        cov_k = float(np.dot(fc[k:], fc[:-k]) / T)
        var_hac += 2.0 * wk * cov_k
    se = float(np.sqrt(max(var_hac / T, 1e-18)))
    cw_stat = float(f_bar / se)
    p_one = float(norm.sf(cw_stat))
    return {"cw_stat": cw_stat, "p_cw_one_sided": p_one, "T_cw": T}


def bootstrap_ci_difference(
    d: np.ndarray,
    n_boot: int = 1000,
    mean_block: float = 7.0,
    alpha: float = 0.05,
    seed: int = 2026,
) -> dict[str, float]:
    """Stationary bootstrap confidence interval and standardized effect size for paired losses."""
    arr = np.asarray(d, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 5:
        return {"mean_diff": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "cohen_d": np.nan}
    ci_lo, ci_hi = inf2.stationary_bootstrap_ci(
        arr, mean_block=float(mean_block), B=int(n_boot), alpha=float(alpha), seed=int(seed)
    )
    sd = float(np.std(arr, ddof=1))
    cohen_d = float(np.mean(arr) / sd) if sd > 1e-12 else 0.0
    return {
        "mean_diff": float(np.mean(arr)),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "cohen_d": cohen_d,
    }

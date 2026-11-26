"""Verified, unit-consistent statistical inference routines for time-series forecast evaluation.

Implements:
- Step/overlap detection (`infer_series_step_and_overlap`)
- Macro-aligned panel decomposition (`build_macro_aligned_panel`, `build_macro_loss_matrix`)
- Synchronized cross-sectional stationary block bootstrap (`macro_synchronized_block_bootstrap`)
- Diebold-Mariano with HLN finite-sample correction, Newey-West HAC, and degenerate-case policy (`dm_hln_hac_v2`)
- Effective sample size reliability classification (`classify_teff_reliability`)
- One-sided T_max Model Confidence Set (`mcs_t_max_v2`, Hansen et al. 2011)
- Studentized consistent Superior Predictive Ability (`spa_test_v2`, Hansen 2005)
- Observation-unit stationary bootstrap CIs (`stationary_bootstrap_ci_v2`)
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def infer_series_step_and_overlap(timestamps: Sequence[Any], horizon_days: int) -> dict[str, Any]:
    """Infer actual temporal step (in days) and forecast-horizon overlap in series units."""
    ts = pd.to_datetime(pd.Index(timestamps)).dropna().sort_values().unique()
    n_t = int(len(ts))
    if n_t <= 1:
        return {
            "n_time_units": n_t,
            "min_step_days": 0.0,
            "median_step_days": 0.0,
            "max_step_days": 0.0,
            "horizon_days": int(horizon_days),
            "h_overlap": max(1, int(horizon_days)),
            "overlapping": bool(horizon_days > 1),
        }
    diffs = pd.Series(ts).diff().dropna().dt.total_seconds() / 86400.0
    min_step = float(diffs.min())
    med_step = float(diffs.median())
    max_step = float(diffs.max())
    h_ov = max(1, int(np.ceil(float(horizon_days) / med_step))) if med_step > 0.0 else max(1, int(horizon_days))
    return {
        "n_time_units": n_t,
        "min_step_days": min_step,
        "median_step_days": med_step,
        "max_step_days": max_step,
        "horizon_days": int(horizon_days),
        "h_overlap": h_ov,
        "overlapping": bool(h_ov > 1),
    }


def _stationary_bootstrap_indices_batch(
    T: int, mean_block: float, B: int, seed: int = 2026
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p_geom = 1.0 / max(1.0, float(mean_block))
    starts = rng.integers(0, T, size=(B, T))
    jumps = rng.random(size=(B, T)) < p_geom
    idx = np.empty((B, T), dtype=int)
    idx[:, 0] = starts[:, 0]
    for t in range(1, T):
        idx[:, t] = np.where(jumps[:, t], starts[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def default_block_length(T: int) -> float:
    """Observation-unit default block length bounded strictly below T/4 for short series."""
    if T <= 8:
        return 1.0
    raw = float(round(T ** (1.0 / 3.0)))
    return float(max(2.0, min(raw, max(2.0, T / 4.0))))


def classify_teff_reliability(t_eff: float, h_overlap: int) -> str:
    """Separate technical eligibility (T_eff >= 2*h_overlap) from asymptotic reliability."""
    if not np.isfinite(t_eff) or t_eff < 2.0 * max(1, int(h_overlap)):
        return "NO_EVALUABLE"
    if t_eff < 20.0:
        return "MUESTRA_PEQUENA_SOLO_SENSIBILIDAD"
    if t_eff < 30.0:
        return "MUESTRA_MODERADA_CAUTELA"
    return "FIABILIDAD_ASINTOTICA_ESTANDAR"


def dm_hln_hac_v2(d: np.ndarray, h_overlap: int = 1, lag: int | None = None) -> dict[str, Any]:
    """Diebold-Mariano test with Newey-West Bartlett HAC, HLN correction, and degenerate-case policy."""
    arr = np.asarray(d, dtype=float)
    arr = arr[np.isfinite(arr)]
    T = int(arr.size)
    h_ov = max(1, int(h_overlap))
    if T < 5 or h_ov >= T:
        return {
            "T": T,
            "T_eff": float(T),
            "h_overlap": h_ov,
            "L_hac": 0,
            "hln_factor": np.nan,
            "dm_raw": np.nan,
            "dm_hln": np.nan,
            "p_dm_hln": np.nan,
            "lag1_acf": np.nan,
            "gamma0": np.nan,
            "lrv": np.nan,
            "degenerate_case": "INSUFFICIENT_T",
            "warning": "Serie demasiado corta o h_overlap >= T; contraste DM-HLN no aplicable.",
        }

    d_mean = float(np.mean(arr))
    dc = arr - d_mean
    gamma0 = float(np.dot(dc, dc) / T)
    if gamma0 <= 1e-18:
        if abs(d_mean) <= 1e-12:
            return {
                "T": T,
                "T_eff": float(T),
                "h_overlap": h_ov,
                "L_hac": 0,
                "hln_factor": 1.0,
                "dm_raw": 0.0,
                "dm_hln": 0.0,
                "p_dm_hln": 1.0,
                "lag1_acf": 0.0,
                "gamma0": gamma0,
                "lrv": 0.0,
                "degenerate_case": "IDENTICAL_ZERO_DIFF",
                "warning": "Diferencia nula constante (modelos idénticos).",
            }
        signed_inf = float(np.copysign(np.inf, d_mean))
        return {
            "T": T,
            "T_eff": float(T),
            "h_overlap": h_ov,
            "L_hac": 0,
            "hln_factor": 1.0,
            "dm_raw": signed_inf,
            "dm_hln": signed_inf,
            "p_dm_hln": np.nan,
            "lag1_acf": np.nan,
            "gamma0": gamma0,
            "lrv": 0.0,
            "degenerate_case": "CONSTANT_NONZERO_DIFF",
            "warning": "Varianza nula con diferencia constante distinta de cero; contraste DM-HLN asintótico degenerado.",
        }

    if lag is None:
        nw_auto = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
        L = max(h_ov - 1, nw_auto)
    else:
        L = int(lag)
    L = max(0, min(L, max(0, T - 2)))

    lrv = gamma0
    acf_sum = 0.0
    for k in range(1, L + 1):
        wk = 1.0 - k / (L + 1.0)
        gk = float(np.dot(dc[k:], dc[:-k]) / T)
        lrv += 2.0 * wk * gk
        acf_sum += wk * (gk / gamma0)

    if lrv <= 1e-18:
        lrv = max(gamma0, 1e-18)

    t_eff = float(T / max(1.0, 1.0 + 2.0 * acf_sum))
    se = float(np.sqrt(lrv / T))
    dm_raw = float(d_mean / se)
    hln_num = T + 1.0 - 2.0 * h_ov + (h_ov * (h_ov - 1.0)) / T
    hln_factor = float(np.sqrt(max(1e-9, hln_num / T)))
    dm_hln = float(dm_raw * hln_factor)
    p_val = float(2.0 * student_t.sf(abs(dm_hln), df=max(1, T - 1)))
    lag1_acf = float(np.corrcoef(arr[:-1], arr[1:])[0, 1]) if T >= 6 and np.std(arr) > 1e-12 else 0.0
    return {
        "T": T,
        "T_eff": round(t_eff, 4),
        "h_overlap": h_ov,
        "L_hac": int(L),
        "hln_factor": round(hln_factor, 6),
        "dm_raw": dm_raw,
        "dm_hln": dm_hln,
        "p_dm_hln": p_val,
        "lag1_acf": lag1_acf,
        "gamma0": gamma0,
        "lrv": lrv,
        "degenerate_case": "OK",
        "warning": "",
    }


def build_macro_aligned_panel(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    join_keys: Sequence[str],
    time_col: str,
    pred_col: str,
    target_col: str,
    plant_col: str = "code",
) -> dict[str, Any]:
    """Align paired predictions and construct the exact MAE_macro-equivalent temporal series.

    Because MAE_macro = (1/M) * sum_{i=1}^M (1/N_i) * sum_{r in i} ae_r,
    defining z_t^macro = (T/M) * sum_{i in P_t} (S_{t,i} / N_i) yields
    mean_t(z_t^macro) == diff_MAE_macro identically even when the panel is unbalanced.
    """
    keys = list(join_keys)
    a = df_a[keys + [pred_col, target_col]].copy()
    b = df_b[keys + [pred_col]].copy()
    a[plant_col] = a[plant_col].astype(str)
    b[plant_col] = b[plant_col].astype(str)
    m = a.merge(b, on=keys, how="inner", suffixes=("_a", "_b"))
    if m.empty:
        raise ValueError("Alineación vacía al construir el panel macro-alineado.")

    ae_a = np.abs(m[f"{pred_col}_a"].to_numpy(float) - m[target_col].to_numpy(float))
    ae_b = np.abs(m[f"{pred_col}_b"].to_numpy(float) - m[target_col].to_numpy(float))
    m["_d"] = ae_a - ae_b

    plant_means = m.groupby(plant_col)["_d"].mean()
    plant_counts = m.groupby(plant_col)["_d"].count()
    diff_macro = float(plant_means.mean())
    diff_micro = float(m["_d"].mean())

    origin_means = m.groupby(time_col)["_d"].mean().sort_index()
    diff_origin = float(origin_means.mean())

    # Exact (T x M) sum and count matrices
    sum_piv = m.groupby([time_col, plant_col])["_d"].sum().unstack(plant_col, fill_value=0.0).sort_index()
    cnt_piv = m.groupby([time_col, plant_col])["_d"].count().unstack(plant_col, fill_value=0).sort_index()
    plants = list(sum_piv.columns)
    times = list(sum_piv.index)
    S = sum_piv.to_numpy(dtype=float)
    C = cnt_piv.to_numpy(dtype=float)
    T, M = S.shape
    N_i = np.maximum(C.sum(axis=0), 1.0)
    z_macro = (T / float(M)) * np.sum(S / N_i[None, :], axis=1)

    plants_per_t = (C > 0).sum(axis=1)
    origins_per_p = (C > 0).sum(axis=0)
    obs_per_p = C.sum(axis=0)

    return {
        "n_obs": int(len(m)),
        "n_times": T,
        "n_plants": M,
        "times": times,
        "plants": plants,
        "panel_sum": S,
        "panel_cnt": C,
        "z_macro_series": z_macro,
        "d_origin_series": origin_means.loc[times].to_numpy(dtype=float),
        "diff_mae_macro": diff_macro,
        "diff_mae_micro": diff_micro,
        "diff_temporal_origin": diff_origin,
        "sign_flip_macro_vs_origin": bool((diff_macro * diff_origin) < 0.0),
        "plants_per_origin_min": int(plants_per_t.min()),
        "plants_per_origin_med": float(np.median(plants_per_t)),
        "plants_per_origin_max": int(plants_per_t.max()),
        "origins_per_plant_min": int(origins_per_p.min()),
        "origins_per_plant_med": float(np.median(origins_per_p)),
        "origins_per_plant_max": int(origins_per_p.max()),
        "imbalance_ratio_obs": float(obs_per_p.max() / max(1.0, obs_per_p.min())),
    }


def build_macro_loss_matrix(
    ashr_df: pd.DataFrame,
    bench_al: pd.DataFrame,
    join_keys: Sequence[str],
    time_col: str,
    pred_col: str,
    target_col: str,
    plant_col: str = "code",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (T x M_models) loss matrices under both MAE_macro-weighted and origin-mean estimands."""
    keys = list(join_keys)
    a = ashr_df[keys + [pred_col, target_col]].copy()
    a[plant_col] = a[plant_col].astype(str)
    a["_ae"] = np.abs(a[pred_col].to_numpy(float) - a[target_col].to_numpy(float))
    a["metodo"] = "ASHR"

    b = bench_al[keys + ["metodo", pred_col, target_col]].copy()
    b[plant_col] = b[plant_col].astype(str)
    b["_ae"] = np.abs(b[pred_col].to_numpy(float) - b[target_col].to_numpy(float))

    combined = pd.concat([a[keys + ["metodo", "_ae"]], b[keys + ["metodo", "_ae"]]], ignore_index=True)

    # Origin-mean loss matrix
    origin_mat = combined.groupby([time_col, "metodo"])["_ae"].mean().unstack("metodo").dropna(axis=0, how="any").sort_index()

    # Exact MAE_macro-weighted loss matrix on the common time index
    valid_times = origin_mat.index
    sub = combined[combined[time_col].isin(valid_times)]
    # Plant total counts N_{i,m}
    n_im = sub.groupby(["metodo", plant_col])["_ae"].count().rename("_N_im")
    m_count = sub.groupby("metodo")[plant_col].nunique().rename("_M_m")
    st = sub.groupby([time_col, "metodo", plant_col])["_ae"].sum().reset_index()
    st = st.merge(n_im.reset_index(), on=["metodo", plant_col], how="left")
    st = st.merge(m_count.reset_index(), on="metodo", how="left")
    T = float(len(valid_times))
    st["_w_ae"] = (T / st["_M_m"]) * (st["_ae"] / st["_N_im"])
    macro_mat = st.groupby([time_col, "metodo"])["_w_ae"].sum().unstack("metodo").loc[valid_times]
    return macro_mat, origin_mat


def macro_synchronized_block_bootstrap(
    panel_sum: np.ndarray,
    panel_cnt: np.ndarray,
    mean_block: float | None = None,
    B: int = 2000,
    alpha: float = 0.05,
    seed: int = 2026,
) -> dict[str, Any]:
    """Cross-sectionally synchronized stationary bootstrap for unbalanced MAE_macro differences."""
    S = np.asarray(panel_sum, dtype=float)
    C = np.asarray(panel_cnt, dtype=float)
    T, M = S.shape
    N_i = C.sum(axis=0)
    valid_p = N_i > 0
    if T < 5 or not np.any(valid_p):
        return {
            "estimate": np.nan,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
            "se_boot": np.nan,
            "z_boot": np.nan,
            "p_boot_two_sided": np.nan,
            "mean_block": 1.0,
            "pct_draws_all_plants_present": np.nan,
            "degenerate_case": "INSUFFICIENT_T",
        }

    est = float(np.mean(S[:, valid_p].sum(axis=0) / N_i[valid_p]))
    blk = float(mean_block) if mean_block is not None else default_block_length(T)
    blk = max(1.0, min(blk, max(1.0, T / 2.0)))
    idx = _stationary_bootstrap_indices_batch(T, blk, B, seed=seed)

    # Vectorized over B: (B, T, M) -> sum over time -> (B, M)
    S_b = S[idx].sum(axis=1)
    C_b = C[idx].sum(axis=1)
    present = C_b > 0
    pct_all = float(100.0 * np.mean(np.all(present[:, valid_p], axis=1)))
    with np.errstate(divide="ignore", invalid="ignore"):
        plant_means_b = np.where(present, S_b / np.maximum(C_b, 1.0), 0.0)
    m_b = np.maximum(present.sum(axis=1), 1)
    boot_macro = plant_means_b.sum(axis=1) / m_b

    se_boot = float(np.std(boot_macro, ddof=1))
    ci_lo = float(np.quantile(boot_macro, alpha / 2.0))
    ci_hi = float(np.quantile(boot_macro, 1.0 - alpha / 2.0))

    if se_boot <= 1e-15:
        if abs(est) <= 1e-12:
            return {
                "estimate": est,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "se_boot": 0.0,
                "z_boot": 0.0,
                "p_boot_two_sided": 1.0,
                "mean_block": blk,
                "pct_draws_all_plants_present": pct_all,
                "degenerate_case": "IDENTICAL_ZERO_DIFF",
            }
        return {
            "estimate": est,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "se_boot": 0.0,
            "z_boot": float(np.copysign(np.inf, est)),
            "p_boot_two_sided": np.nan,
            "mean_block": blk,
            "pct_draws_all_plants_present": pct_all,
            "degenerate_case": "CONSTANT_NONZERO_DIFF",
        }

    centered = boot_macro - est
    p_boot = float((1.0 + np.sum(np.abs(centered) >= abs(est))) / (B + 1.0))
    return {
        "estimate": est,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "se_boot": se_boot,
        "z_boot": float(est / se_boot),
        "p_boot_two_sided": p_boot,
        "mean_block": blk,
        "pct_draws_all_plants_present": pct_all,
        "degenerate_case": "OK",
    }


def stationary_bootstrap_ci_v2(
    d: np.ndarray,
    mean_block: float | None = None,
    B: int = 2000,
    alpha: float = 0.05,
    seed: int = 2026,
) -> tuple[float, float, float]:
    """Stationary bootstrap confidence interval with observation-unit block length."""
    arr = np.asarray(d, dtype=float)
    arr = arr[np.isfinite(arr)]
    T = int(arr.size)
    if T < 5:
        return np.nan, np.nan, 1.0
    blk = float(mean_block) if mean_block is not None else default_block_length(T)
    blk = max(1.0, min(blk, max(1.0, T / 2.0)))
    idx = _stationary_bootstrap_indices_batch(T, blk, B, seed=seed)
    means = np.mean(arr[idx], axis=1)
    return float(np.quantile(means, alpha / 2.0)), float(np.quantile(means, 1.0 - alpha / 2.0)), blk


def hansen_politis_romano_variance(d: np.ndarray, block_size: float) -> np.ndarray:
    """Asymptotic long-run variance estimator of Politis-Romano / Hansen (2005, Eq. 7)."""
    arr = np.asarray(d, dtype=float)
    T = arr.shape[0]
    dm = arr - np.mean(arr, axis=0, keepdims=True)
    p = 1.0 / max(1.0, float(block_size))
    var = np.sum(dm ** 2, axis=0) / T
    for i in range(1, T):
        kappa = ((1.0 - i / T) * ((1.0 - p) ** i)) + ((i / T) * ((1.0 - p) ** (T - i)))
        var += 2.0 * kappa * np.sum(dm[: T - i, :] * dm[i:, :], axis=0) / T
    return np.maximum(var, 1e-18)


def mcs_t_max_v2(
    loss_matrix: np.ndarray,
    model_names: list[str],
    alpha: float = 0.10,
    B: int = 2000,
    mean_block: float | None = None,
    seed: int = 2026,
    finite_sample_floor: bool = True,
) -> dict[str, Any]:
    """Hansen, Lunde & Nason (2011) Model Confidence Set (one-sided T_max statistic)."""
    L = np.asarray(loss_matrix, dtype=float)
    valid = np.all(np.isfinite(L), axis=1)
    L = L[valid]
    T, M = L.shape
    if T < 5 or M < 2:
        return {
            "survivors": list(model_names),
            "eliminated": [],
            "elimination_order": [],
            "p_mcs": {m: 1.0 for m in model_names},
            "mc_se": {m: 0.0 for m in model_names},
            "mean_block": 1.0,
            "T": T,
        }

    blk = float(mean_block) if mean_block is not None else default_block_length(T)
    blk = max(1.0, min(blk, max(1.0, T / 2.0)))
    boot_idx = _stationary_bootstrap_indices_batch(T, blk, B, seed=seed)

    active = list(range(M))
    eliminated: list[tuple[str, float]] = []
    elimination_order: list[tuple[str, float, float]] = []
    pvals: dict[str, float] = {m: 1.0 for m in model_names}
    mc_se: dict[str, float] = {m: 0.0 for m in model_names}
    p_running = 0.0

    while len(active) > 1:
        sub = L[:, active]
        d_idot = sub - sub.mean(axis=1, keepdims=True)
        d_bar = d_idot.mean(axis=0)

        boot_means = d_idot[boot_idx].mean(axis=1)
        centered_boot = boot_means - d_bar[None, :]
        se = np.maximum(np.sqrt(np.mean(centered_boot ** 2, axis=0)), 1e-12)

        t_obs = d_bar / se
        t_boot = centered_boot / se[None, :]
        stat_obs = float(np.max(t_obs))
        stat_boot = np.max(t_boot, axis=1)
        if finite_sample_floor:
            p_step = float((1.0 + np.sum(stat_boot >= stat_obs)) / (B + 1.0))
        else:
            p_step = float(np.mean(stat_boot > stat_obs))
        p_running = max(p_running, p_step)

        worst_local = int(np.argmax(t_obs))
        worst_idx = active[worst_local]
        name_w = model_names[worst_idx]
        pvals[name_w] = p_running
        mc_se[name_w] = float(np.sqrt(p_running * (1.0 - p_running) / B))
        elimination_order.append((name_w, p_step, p_running))
        if p_running >= alpha:
            break
        eliminated.append((name_w, p_running))
        active.pop(worst_local)

    survivors = [model_names[i] for i in active]
    return {
        "survivors": survivors,
        "eliminated": eliminated,
        "elimination_order": elimination_order,
        "p_mcs": pvals,
        "mc_se": mc_se,
        "mean_block": blk,
        "T": T,
    }


def spa_test_v2(
    loss_benchmark: np.ndarray,
    loss_models: np.ndarray,
    B: int = 2000,
    mean_block: float | None = None,
    seed: int = 2026,
    studentize: bool = True,
    variance_estimator: str = "pr_kernel",
    clip_zero: bool = True,
    finite_sample_floor: bool = True,
) -> dict[str, Any]:
    """Hansen (2005) Superior Predictive Ability (SPA) test.

    Tests H0: max_k E[L_bench - L_k] <= 0 against H1: some competitor outperforms benchmark.
    Notes:
    - Uses Hansen (2005, Eq. 7) Politis-Romano kernel variance (`variance_estimator='pr_kernel'`)
      or bootstrap variance (`'bootstrap'`).
    - When `clip_zero=True` and no competitor beats the benchmark in sample (`max_k d_bar_k <= 0`),
      Hansen (2005) defines T_n^SPA = 0 and p_SPA = 1.0.
    - Setting `studentize=False, clip_zero=False, finite_sample_floor=False` matches `arch 8.0.0`
      (`arch.bootstrap.SPA`, where `studentize=True` is stored but unapplied in `compute()`).
    """
    lb = np.asarray(loss_benchmark, dtype=float)
    lm = np.asarray(loss_models, dtype=float)
    if lm.ndim == 1:
        lm = lm[:, None]
    d = lb[:, None] - lm
    valid = np.all(np.isfinite(d), axis=1)
    d = d[valid]
    T, K = d.shape
    if T < 5 or K == 0:
        return {
            "spa_stat": np.nan,
            "spa_stat_unclipped": np.nan,
            "p_spa_lower": np.nan,
            "p_spa_consistent": np.nan,
            "p_spa_upper": np.nan,
            "T_spa": float(T),
            "mean_block": 1.0,
            "studentize": studentize,
            "variance_estimator": variance_estimator,
            "n_valid_consistent": 0,
        }

    blk = float(mean_block) if mean_block is not None else default_block_length(T)
    blk = max(1.0, min(blk, max(1.0, T / 2.0)))
    boot_idx = _stationary_bootstrap_indices_batch(T, blk, B, seed=seed)

    d_bar = np.mean(d, axis=0)
    boot_means = d[boot_idx].mean(axis=1)  # (B, K)
    centered = boot_means - d_bar[None, :]

    if variance_estimator == "pr_kernel":
        omega2 = hansen_politis_romano_variance(d, blk)
        omega = np.sqrt(omega2)
    elif variance_estimator == "bootstrap":
        omega = np.maximum(np.sqrt(T * np.mean(centered ** 2, axis=0)), 1e-12)
    else:
        raise ValueError(f"Unknown variance_estimator: {variance_estimator}")

    scale = (np.sqrt(T) / omega) if studentize else np.ones_like(omega)
    raw_obs = float(np.max(d_bar * scale))
    t_obs = max(0.0, raw_obs) if clip_zero else raw_obs

    thresh = -np.sqrt(2.0 * np.log(max(np.log(max(T, 3)), 1.0001))) * (omega / np.sqrt(T))
    valid_cons = d_bar >= thresh
    mu_lower = np.minimum(0.0, d_bar)
    mu_cons = np.where(valid_cons, 0.0, d_bar)
    mu_upper = np.zeros_like(d_bar)

    def _pval(mu_vec: np.ndarray) -> float:
        if clip_zero and raw_obs <= 0.0:
            return 1.0
        z_boot = (centered + mu_vec[None, :]) * scale[None, :]
        raw_boot = np.max(z_boot, axis=1)
        t_boot = np.maximum(0.0, raw_boot) if clip_zero else raw_boot
        if finite_sample_floor:
            return float((1.0 + np.sum(t_boot >= t_obs)) / (B + 1.0))
        return float(np.mean(t_boot > t_obs))

    return {
        "spa_stat": t_obs,
        "spa_stat_unclipped": raw_obs,
        "p_spa_lower": _pval(mu_lower),
        "p_spa_consistent": _pval(mu_cons),
        "p_spa_upper": _pval(mu_upper),
        "T_spa": float(T),
        "mean_block": blk,
        "studentize": studentize,
        "variance_estimator": variance_estimator,
        "n_valid_consistent": int(np.sum(valid_cons)),
    }


__all__ = [
    "build_macro_aligned_panel",
    "build_macro_loss_matrix",
    "classify_teff_reliability",
    "default_block_length",
    "dm_hln_hac_v2",
    "hansen_politis_romano_variance",
    "infer_series_step_and_overlap",
    "macro_synchronized_block_bootstrap",
    "mcs_t_max_v2",
    "spa_test_v2",
    "stationary_bootstrap_ci_v2",
]

"""Módulo de inferencia estadística corregida v2 para R-ILWR-PA (v3).

Implementa:
  - MCS (Hansen et al., 2011) con estadístico unilateral T_max, monotonía de p-valores,
    piso de Monte Carlo 1/(B+1), error estándar MC y bootstrap estacionario sobre los 23 modelos.
  - Diebold-Mariano con varianza HAC Newey-West, corrección HLN y tamaño muestral efectivo T_eff.
  - Giacomini-White con covarianza Newey-West sin centrar (bajo H0: E[Z_t]=0).
  - Intervalos de confianza por bootstrap estacionario (Politis-Romano) y BCa pareado.
  - Prueba t corregida de Nadeau-Bengio (2003) para validación cruzada repetida.
  - Verificación de supuestos (ADF, KPSS, Ljung-Box, Shapiro-Wilk).
  - Tamaños del efecto pareados (Cohen d_z, rango-biserial r_rb, probabilidad de victoria)
    junto con Cliff's delta marginal explícitamente etiquetado.
  - Ajuste Holm-Bonferroni por familia y global.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap, chi2, rankdata, shapiro, t as student_t
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss

ROOT = Path(__file__).resolve().parents[2]
SEED = 2026


def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * float(p[idx]))
        adj[idx] = min(1.0, running)
    return adj


def stationary_bootstrap_indices(T: int, mean_block: float, B: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p_geom = 1.0 / max(1.0, float(mean_block))
    starts = rng.integers(0, T, size=(B, T))
    jumps = rng.random(size=(B, T)) < p_geom
    idx = np.empty((B, T), dtype=int)
    idx[:, 0] = starts[:, 0]
    for t in range(1, T):
        idx[:, t] = np.where(jumps[:, t], starts[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def stationary_bootstrap_ci(
    d: np.ndarray, mean_block: float = 7.0, B: int = 2000, alpha: float = 0.05, seed: int = SEED
) -> tuple[float, float]:
    d_arr = np.asarray(d, dtype=float)
    idx = stationary_bootstrap_indices(len(d_arr), mean_block, B, seed=seed)
    means = np.mean(d_arr[idx], axis=1)
    return float(np.quantile(means, alpha / 2.0)), float(np.quantile(means, 1.0 - alpha / 2.0))


def paired_bca_ci(d: np.ndarray, B: int = 2000, alpha: float = 0.05, seed: int = SEED) -> tuple[float, float]:
    d_arr = np.asarray(d, dtype=float)
    if np.allclose(d_arr, d_arr[0]):
        v = float(d_arr[0])
        return v, v
    res = bootstrap(
        (d_arr,),
        np.mean,
        confidence_level=1.0 - alpha,
        n_resamples=B,
        method="BCa",
        random_state=np.random.default_rng(seed),
    )
    lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        res_p = bootstrap(
            (d_arr,),
            np.mean,
            confidence_level=1.0 - alpha,
            n_resamples=B,
            method="percentile",
            random_state=np.random.default_rng(seed),
        )
        return float(res_p.confidence_interval.low), float(res_p.confidence_interval.high)
    return lo, hi


def dm_hln_hac(d: np.ndarray, horizon_h: int = 1, lag: int | None = None) -> dict:
    d_arr = np.asarray(d, dtype=float)
    T = len(d_arr)
    d_mean = float(np.mean(d_arr))
    dc = d_arr - d_mean
    if lag is None:
        L = max(1, int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0))))
    else:
        L = int(lag)
    L = max(0, min(L, max(0, T - 2)))
    gamma0 = float(np.dot(dc, dc) / T)
    lrv = gamma0
    acf_sum = 0.0
    for k in range(1, L + 1):
        wk = 1.0 - k / (L + 1.0)
        gk = float(np.dot(dc[k:], dc[:-k]) / T)
        lrv += 2.0 * wk * gk
        if gamma0 > 1e-18:
            acf_sum += wk * (gk / gamma0)
    if lrv <= 1e-18:
        lrv = max(gamma0, 1e-18)
    t_eff = float(T / max(1.0, 1.0 + 2.0 * acf_sum))
    se = float(np.sqrt(lrv / T))
    dm = d_mean / se
    hln = float(np.sqrt(max(1e-9, (T + 1.0 - 2.0 * horizon_h + horizon_h * (horizon_h - 1.0) / T) / T)))
    dm_hln = dm * hln
    p_val = float(2.0 * student_t.sf(abs(dm_hln), df=max(1, T - 1)))
    return {"dm_hln": float(dm_hln), "p_dm_hln": p_val, "L_hac": int(L), "T_eff": round(t_eff, 2)}


def giacomini_white_hac(d: np.ndarray, lag: int | None = None) -> dict:
    """Prueba de capacidad predictiva condicional (Giacomini & White, 2006) con HAC sin centrar."""
    d_arr = np.asarray(d, dtype=float)
    if len(d_arr) < 10:
        return {"gw_wald": np.nan, "p_gw": np.nan, "L_hac": 0}
    y_dep = d_arr[1:]
    n = len(y_dep)
    H = np.column_stack([np.ones(n), d_arr[:-1]])
    Z = H * y_dep[:, None]
    zbar = np.mean(Z, axis=0)
    if lag is None:
        L = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    else:
        L = int(lag)
    L = max(0, min(L, max(0, n - 2)))
    # Covarianza Newey-West sin centrar bajo H0: E[Z_t] = 0
    Omega = (Z.T @ Z) / n
    for k in range(1, L + 1):
        wk = 1.0 - k / (L + 1.0)
        Gk = (Z[k:].T @ Z[:-k]) / n
        Omega = Omega + wk * (Gk + Gk.T)
    # Garantizar semidefinida positiva si la resta de rezagos produjera autovalores mínimos negativos
    eigvals, eigvecs = np.linalg.eigh(Omega)
    if np.min(eigvals) < 1e-12:
        eigvals = np.maximum(eigvals, 1e-12)
        Omega = (eigvecs * eigvals) @ eigvecs.T
    wald = float(n * (zbar @ np.linalg.solve(Omega, zbar)))
    p_val = float(chi2.sf(wald, df=2))
    return {"gw_wald": wald, "p_gw": p_val, "L_hac": int(L)}


def nadeau_bengio_ttest(d: np.ndarray, n_test_ratio: float = 0.25) -> dict:
    """Prueba t pareada corregida de Nadeau & Bengio (2003) sobre K pliegues."""
    d_arr = np.asarray(d, dtype=float)
    K = len(d_arr)
    mean_d = float(np.mean(d_arr))
    var_d = float(np.var(d_arr, ddof=1))
    corr_var = (1.0 / K + float(n_test_ratio)) * var_d
    se_nb = float(np.sqrt(max(corr_var, 1e-24)))
    t_stat = mean_d / se_nb
    p_val = float(2.0 * student_t.sf(abs(t_stat), df=max(1, K - 1)))
    return {"t_nb": float(t_stat), "p_nb": p_val, "K": int(K), "se_nb": se_nb}


def check_assumptions_ts(d: np.ndarray) -> dict:
    d_arr = np.asarray(d, dtype=float)
    lb_df = acorr_ljungbox(d_arr, lags=[7], return_df=True)
    lb_p = float(lb_df["lb_pvalue"].iloc[0])
    _, shap_p = shapiro(d_arr) if 3 <= len(d_arr) <= 5000 else (np.nan, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            adf_p = float(adfuller(d_arr, autolag="AIC")[1])
        except Exception:
            adf_p = np.nan
        try:
            kpss_p = float(kpss(d_arr, regression="c", nlags="auto")[1])
        except Exception:
            kpss_p = np.nan
    return {
        "p_ljung_box_7": round(lb_p, 4),
        "p_shapiro": round(float(shap_p), 4) if np.isfinite(shap_p) else np.nan,
        "p_adf": round(adf_p, 4) if np.isfinite(adf_p) else np.nan,
        "p_kpss": round(kpss_p, 4) if np.isfinite(kpss_p) else np.nan,
    }


def paired_effect_sizes(loss_a: np.ndarray, loss_b: np.ndarray) -> dict:
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    d = a - b
    sd = float(np.std(d, ddof=1))
    cohen_dz = float(np.mean(d) / sd) if sd > 1e-12 else 0.0
    p_win = float(np.mean(d < 0))
    # Correlación rango-biserial pareada (negativo favorece a A: pérdida menor)
    nonzero = np.abs(d) > 1e-15
    if np.any(nonzero):
        dnz = d[nonzero]
        ranks = rankdata(np.abs(dnz))
        w_pos = float(np.sum(ranks[dnz > 0]))
        w_neg = float(np.sum(ranks[dnz < 0]))
        total_r = float(np.sum(ranks))
        r_rb = (w_pos - w_neg) / total_r
    else:
        r_rb = 0.0
    # Cliff's delta marginal (no pareado; se reporta etiquetado como marginal)
    diff_mat = a[:, None] - b[None, :]
    cliff_marg = float((np.sum(diff_mat > 0) - np.sum(diff_mat < 0)) / (len(a) * len(b)))
    return {
        "cohen_dz_pareado": round(cohen_dz, 3),
        "r_rangobiserial_pareado": round(float(r_rb), 3),
        "prob_victoria_A": round(p_win, 3),
        "cliffs_delta_marginal": round(cliff_marg, 3),
    }


def mcs_stationary_v2(
    loss_matrix: np.ndarray,
    model_names: list[str],
    alpha: float = 0.10,
    B: int = 2000,
    mean_block: float = 7.0,
    seed: int = SEED,
) -> pd.DataFrame:
    """MCS (Hansen et al., 2011) con T_max unilateral, piso MC 1/(B+1) y monotonía."""
    T, M0 = loss_matrix.shape
    idx_boot = stationary_bootstrap_indices(T, mean_block, B, seed=seed)

    surviving = list(range(M0))
    elim_records = {}
    running_p = 0.0

    while len(surviving) > 1:
        L = loss_matrix[:, surviving]
        d_ij = L[:, :, None] - L[:, None, :]
        d_bar_ij = np.mean(d_ij, axis=0)
        d_bar_i = np.mean(d_bar_ij, axis=1)

        boot_means = np.mean(d_ij[idx_boot], axis=1)
        boot_i = np.mean(boot_means, axis=2)
        centered = boot_i - d_bar_i[None, :]
        var_i = np.var(centered, axis=0, ddof=1)
        se_i = np.sqrt(np.maximum(var_i, 1e-18))

        t_i = d_bar_i / se_i
        t_max = float(np.max(t_i))
        # Estadístico unilateral de Hansen et al. (2011) Ec. (12)
        boot_t_max = np.max(centered / se_i[None, :], axis=1)
        raw_p = float((1.0 + np.sum(boot_t_max >= t_max)) / (B + 1.0))
        running_p = max(running_p, raw_p)
        mc_se = float(np.sqrt(running_p * (1.0 - running_p) / B))

        worst_pos = int(np.argmax(t_i))
        worst_idx = surviving[worst_pos]
        elim_records[model_names[worst_idx]] = (running_p, mc_se, False)
        if running_p >= alpha:
            # Todos los restantes sobreviven; se registra el p-valor de parada del último candidato evaluado
            elim_records[model_names[worst_idx]] = (running_p, mc_se, True)
            break
        surviving.pop(worst_pos)

    if len(surviving) == 1:
        only_name = model_names[surviving[0]]
        elim_records[only_name] = (1.0, 0.0, True)

    mean_loss = np.mean(loss_matrix, axis=0)
    rows = []
    for idx, name in enumerate(model_names):
        in_mcs = idx in surviving
        p_mcs, mc_se, _ = elim_records.get(name, (1.0, 0.0, True))
        if in_mcs and name not in elim_records:
            # Miembros que permanecieron cuando otro alcanzó p >= alpha
            p_mcs = max(running_p, alpha)
            mc_se = float(np.sqrt(p_mcs * (1.0 - p_mcs) / B))
        rows.append(
            {
                "modelo": name,
                "MAE_diario_medio": float(mean_loss[idx]),
                "en_MCS_90": bool(in_mcs),
                "p_MCS": round(float(p_mcs), 4),
                "mc_se": round(float(mc_se), 4),
                "piso_mc": round(1.0 / (B + 1.0), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("MAE_diario_medio").reset_index(drop=True)


def run_solar_d1_v2(out_dir: Path) -> None:
    pred_rilwr = pd.read_csv(ROOT / "resultados" / "evaluacion_rilwr_pa_v3" / "predicciones_d1.csv")
    pred_rilwr["date"] = pd.to_datetime(pred_rilwr["date"])

    mcs_all, mcs_sens, tests_all, hac_sens = [], [], [], []
    scenarios = (
        "espaciotemporal_cinco_variables",
        "intraplanta_cinco_variables",
        "espaciotemporal_sin_meteorologia",
        "intraplanta_sin_meteorologia",
    )

    for sc in scenarios:
        ref_df = pd.read_csv(ROOT / "resultados" / "benchmark_produccion_definitivo" / sc / "predicciones.csv")
        ref_df["date"] = pd.to_datetime(ref_df["date"])

        s_rilwr = pred_rilwr[pred_rilwr["escenario"] == sc].groupby("date")["error_abs_kwh"].mean().rename("R-ILWR-PA")
        piv = ref_df.groupby(["date", "metodo"])["error_abs_kwh"].mean().unstack("metodo")
        mat = pd.concat([s_rilwr, piv], axis=1).dropna().sort_index()
        assert mat.shape[1] == 23 and "CF-BRH" not in mat.columns

        # MCS completo 23 modelos (bloque nominal 7)
        mcs_df = mcs_stationary_v2(mat.to_numpy(float), list(mat.columns), alpha=0.10, B=2000, mean_block=7.0)
        mcs_df["escenario"] = sc
        mcs_all.append(mcs_df)

        # Sensibilidad de MCS a longitud de bloque L in {4, 7, 14}
        for blk in (4.0, 7.0, 14.0):
            ms = mcs_stationary_v2(mat.to_numpy(float), list(mat.columns), alpha=0.10, B=2000, mean_block=blk)
            row_r = ms[ms["modelo"] == "R-ILWR-PA"].iloc[0]
            mcs_sens.append(
                {
                    "escenario": sc,
                    "mean_block": blk,
                    "n_sobrevivientes_23": int(ms["en_MCS_90"].sum()),
                    "R_ILWR_PA_en_MCS": bool(row_r["en_MCS_90"]),
                    "R_ILWR_PA_p_MCS": float(row_r["p_MCS"]),
                }
            )

        # Contrastes pareados frente a referencias clave (m=7 por escenario, m=28 global)
        comps = ["BayesianRidge", "Lasso", "SVR", "Random Forest", "XGBoost", "IWLAR", "WCFC"]
        sc_rows = []
        for comp in comps:
            a = mat["R-ILWR-PA"].to_numpy(float)
            b = mat[comp].to_numpy(float)
            d = a - b
            dm = dm_hln_hac(d, horizon_h=1, lag=None)
            gw = giacomini_white_hac(d, lag=None)
            ci_lo, ci_hi = stationary_bootstrap_ci(d, mean_block=7.0, B=2000)
            ci_lo_4, ci_hi_4 = stationary_bootstrap_ci(d, mean_block=4.0, B=2000)
            ci_lo_14, ci_hi_14 = stationary_bootstrap_ci(d, mean_block=14.0, B=2000)
            assump = check_assumptions_ts(d)
            eff = paired_effect_sizes(a, b)
            sc_rows.append(
                {
                    "alcance": "EXPLORATORIO_SOLAR_V2",
                    "escenario": sc,
                    "referencia": comp,
                    "T_dias": len(d),
                    "T_eff": dm["T_eff"],
                    "L_hac": dm["L_hac"],
                    "diff_media_kWh": round(float(np.mean(d)), 2),
                    "IC95_L7_inf": round(ci_lo, 2),
                    "IC95_L7_sup": round(ci_hi, 2),
                    "IC95_L4": f"[{ci_lo_4:.1f}, {ci_hi_4:.1f}]",
                    "IC95_L14": f"[{ci_lo_14:.1f}, {ci_hi_14:.1f}]",
                    **eff,
                    "dm_hln_stat": round(dm["dm_hln"], 3),
                    "p_dm_crudo": dm["p_dm_hln"],
                    "gw_wald_hac": round(gw["gw_wald"], 3),
                    "p_gw_crudo": gw["p_gw"],
                    **assump,
                }
            )
            for lag_s in (2, 4, 7):
                dm_s = dm_hln_hac(d, horizon_h=1, lag=lag_s)
                gw_s = giacomini_white_hac(d, lag=lag_s)
                hac_sens.append(
                    {
                        "escenario": sc,
                        "referencia": comp,
                        "L_hac": lag_s,
                        "T_eff": dm_s["T_eff"],
                        "dm_hln_stat": round(dm_s["dm_hln"], 3),
                        "p_dm_crudo": dm_s["p_dm_hln"],
                        "gw_wald_hac": round(gw_s["gw_wald"], 3),
                        "p_gw_crudo": gw_s["p_gw"],
                    }
                )
        sc_df = pd.DataFrame(sc_rows)
        sc_df["p_dm_holm_escenario"] = holm_adjust(sc_df["p_dm_crudo"].to_numpy())
        sc_df["p_gw_holm_escenario"] = holm_adjust(sc_df["p_gw_crudo"].to_numpy())
        tests_all.append(sc_df)

    full_tests = pd.concat(tests_all, ignore_index=True)
    full_tests["p_dm_holm_global28"] = holm_adjust(full_tests["p_dm_crudo"].to_numpy())
    full_tests["p_gw_holm_global28"] = holm_adjust(full_tests["p_gw_crudo"].to_numpy())

    pd.concat(mcs_all, ignore_index=True).to_csv(out_dir / "mcs_23_modelos_d1.csv", index=False)
    pd.DataFrame(mcs_sens).to_csv(out_dir / "mcs_sensibilidad_bloques.csv", index=False)
    full_tests.to_csv(out_dir / "pruebas_solar_d1_v2.csv", index=False)
    pd.DataFrame(hac_sens).to_csv(out_dir / "sensibilidad_hac_d1.csv", index=False)


def main() -> None:
    t0 = time.perf_counter()
    out_dir = ROOT / "resultados" / "evaluacion_rilwr_pa_v3" / "inferencia_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_solar_d1_v2(out_dir)
    print(f"Inferencia solar v2 completada en {time.perf_counter() - t0:.2f}s -> {out_dir}")


if __name__ == "__main__":
    main()

"""Multimode Adaptive Structured Huber Regressor in pure NumPy/SciPy.

Unifies linear, quantile-spline, structured-L2, physically-monotone,
train-screened interaction, and causal-anchor residual formulations under a
single objective and analytical-gradient L-BFGS-B solver, with internal CV
structure selection and parsimony protection.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError

try:
    from modelo_aditivo_huber import (
        _as_1d_float,
        _as_2d_float,
        _huber_val_and_grad,
        _robust_target_stats,
    )
except ImportError:
    from src.modelo_aditivo_huber import (
        _as_1d_float,
        _as_2d_float,
        _huber_val_and_grad,
        _robust_target_stats,
    )


def _candidate_grid(resolved_mode: str, horizon: int) -> list[dict[str, Any]]:
    """Return a compact, context-aware candidate grid (5-6 configs, no Cartesian blowup)."""
    if resolved_mode == "tabular":
        return [
            {
                "name": "tab_lin_unif", "complexity_rank": 1,
                "n_knots": 0, "alpha_lin": 1e-2, "alpha_meteo": 1e-2, "alpha_spl": 1e-2,
                "max_interactions": 0, "alpha_int": 0.1, "structured": False, "use_mono": False,
                "anchor": "none", "lead_damp": 1.0,
            },
            {
                "name": "tab_spline_med", "complexity_rank": 2,
                "n_knots": 6, "alpha_lin": 1e-2, "alpha_meteo": 1e-2, "alpha_spl": 5e-2,
                "max_interactions": 0, "alpha_int": 0.1, "structured": False, "use_mono": False,
                "anchor": "none", "lead_damp": 1.0,
            },
            {
                "name": "tab_spline_flex", "complexity_rank": 3,
                "n_knots": 8, "alpha_lin": 1e-3, "alpha_meteo": 1e-3, "alpha_spl": 1e-2,
                "max_interactions": 0, "alpha_int": 0.1, "structured": False, "use_mono": False,
                "anchor": "none", "lead_damp": 1.0,
            },
            {
                "name": "tab_spline_inter_med", "complexity_rank": 4,
                "n_knots": 6, "alpha_lin": 1e-2, "alpha_meteo": 1e-2, "alpha_spl": 5e-2,
                "max_interactions": 4, "alpha_int": 5e-2, "structured": False, "use_mono": False,
                "anchor": "none", "lead_damp": 1.0,
            },
            {
                "name": "tab_spline_inter_flex", "complexity_rank": 5,
                "n_knots": 6, "alpha_lin": 1e-3, "alpha_meteo": 1e-3, "alpha_spl": 1e-2,
                "max_interactions": 6, "alpha_int": 2e-2, "structured": False, "use_mono": False,
                "anchor": "none", "lead_damp": 1.0,
            },
        ]

    if horizon > 1:
        return [
            {
                "name": "mh_lin_mm30_damp", "complexity_rank": 1,
                "n_knots": 0, "alpha_lin": 1e-2, "alpha_meteo": 1e-1, "alpha_spl": 1e-1,
                "max_interactions": 0, "alpha_int": 0.1, "structured": True, "use_mono": True,
                "anchor": "mm30", "lead_damp": 0.90,
            },
            {
                "name": "mh_lin_kalman_damp", "complexity_rank": 2,
                "n_knots": 0, "alpha_lin": 1e-2, "alpha_meteo": 1e-1, "alpha_spl": 1e-1,
                "max_interactions": 0, "alpha_int": 0.1, "structured": True, "use_mono": True,
                "anchor": "kalman", "lead_damp": 0.90,
            },
            {
                "name": "mh_lin_struct_mono", "complexity_rank": 3,
                "n_knots": 0, "alpha_lin": 1e-2, "alpha_meteo": 1e-1, "alpha_spl": 1e-1,
                "max_interactions": 0, "alpha_int": 0.1, "structured": True, "use_mono": True,
                "anchor": "none", "lead_damp": 1.0,
            },
            {
                "name": "mh_spline_mm30_damp", "complexity_rank": 4,
                "n_knots": 6, "alpha_lin": 1e-2, "alpha_meteo": 5e-1, "alpha_spl": 5e-1,
                "max_interactions": 0, "alpha_int": 0.1, "structured": True, "use_mono": True,
                "anchor": "mm30", "lead_damp": 0.90,
            },
            {
                "name": "mh_spline_kalman_damp", "complexity_rank": 5,
                "n_knots": 6, "alpha_lin": 1e-2, "alpha_meteo": 5e-1, "alpha_spl": 5e-1,
                "max_interactions": 0, "alpha_int": 0.1, "structured": True, "use_mono": True,
                "anchor": "kalman", "lead_damp": 0.90,
            },
        ]

    if resolved_mode == "intraplanta":
        return [
            {
                "name": "ip_lin_struct_light", "complexity_rank": 1,
                "n_knots": 0, "alpha_lin": 1e-2, "alpha_meteo": 1e-1, "alpha_spl": 1e-1,
                "max_interactions": 0, "alpha_int": 0.1, "structured": True, "use_mono": True,
                "anchor": "none", "lead_damp": 1.0,
            },
            {
                "name": "ip_lin_struct_strong", "complexity_rank": 2,
                "n_knots": 0, "alpha_lin": 1e-1, "alpha_meteo": 1.0, "alpha_spl": 1.0,
                "max_interactions": 0, "alpha_int": 0.1, "structured": True, "use_mono": True,
                "anchor": "none", "lead_damp": 1.0,
            },
            {
                "name": "ip_lin_mm30_res", "complexity_rank": 3,
                "n_knots": 0, "alpha_lin": 1e-2, "alpha_meteo": 1e-1, "alpha_spl": 1e-1,
                "max_interactions": 0, "alpha_int": 0.1, "structured": True, "use_mono": True,
                "anchor": "mm30", "lead_damp": 1.0,
            },
            {
                "name": "ip_lin_kalman_res", "complexity_rank": 4,
                "n_knots": 0, "alpha_lin": 1e-2, "alpha_meteo": 1e-1, "alpha_spl": 1e-1,
                "max_interactions": 0, "alpha_int": 0.1, "structured": True, "use_mono": True,
                "anchor": "kalman", "lead_damp": 1.0,
            },
            {
                "name": "ip_spline_struct", "complexity_rank": 5,
                "n_knots": 6, "alpha_lin": 1e-2, "alpha_meteo": 5e-1, "alpha_spl": 5e-1,
                "max_interactions": 0, "alpha_int": 0.1, "structured": True, "use_mono": True,
                "anchor": "none", "lead_damp": 1.0,
            },
            {
                "name": "ip_spline_kalman_res", "complexity_rank": 6,
                "n_knots": 6, "alpha_lin": 1e-2, "alpha_meteo": 5e-1, "alpha_spl": 5e-1,
                "max_interactions": 0, "alpha_int": 0.1, "structured": True, "use_mono": True,
                "anchor": "kalman", "lead_damp": 1.0,
            },
        ]

    # espaciotemporal (horizon == 1)
    return [
        {
            "name": "et_lin_unif", "complexity_rank": 1,
            "n_knots": 0, "alpha_lin": 1e-2, "alpha_meteo": 1e-2, "alpha_spl": 1e-2,
            "max_interactions": 0, "alpha_int": 0.1, "structured": False, "use_mono": False,
            "anchor": "none", "lead_damp": 1.0,
        },
        {
            "name": "et_lin_struct_mono", "complexity_rank": 2,
            "n_knots": 0, "alpha_lin": 1e-2, "alpha_meteo": 1e-1, "alpha_spl": 1e-1,
            "max_interactions": 0, "alpha_int": 0.1, "structured": True, "use_mono": True,
            "anchor": "none", "lead_damp": 1.0,
        },
        {
            "name": "et_lin_kalman_res", "complexity_rank": 3,
            "n_knots": 0, "alpha_lin": 1e-2, "alpha_meteo": 1e-1, "alpha_spl": 1e-1,
            "max_interactions": 0, "alpha_int": 0.1, "structured": True, "use_mono": True,
            "anchor": "kalman", "lead_damp": 1.0,
        },
        {
            "name": "et_spline_unif", "complexity_rank": 4,
            "n_knots": 6, "alpha_lin": 1e-2, "alpha_meteo": 1e-2, "alpha_spl": 1e-1,
            "max_interactions": 0, "alpha_int": 0.1, "structured": False, "use_mono": False,
            "anchor": "none", "lead_damp": 1.0,
        },
        {
            "name": "et_spline_struct_mono", "complexity_rank": 5,
            "n_knots": 6, "alpha_lin": 1e-2, "alpha_meteo": 1e-1, "alpha_spl": 5e-1,
            "max_interactions": 0, "alpha_int": 0.1, "structured": True, "use_mono": True,
            "anchor": "none", "lead_damp": 1.0,
        },
        {
            "name": "et_spline_kalman_res", "complexity_rank": 6,
            "n_knots": 6, "alpha_lin": 1e-2, "alpha_meteo": 1e-1, "alpha_spl": 5e-1,
            "max_interactions": 0, "alpha_int": 0.1, "structured": True, "use_mono": True,
            "anchor": "kalman", "lead_damp": 1.0,
        },
    ]


def _candidate_grid_v2(resolved_mode: str, horizon: int) -> list[dict[str, Any]]:
    """Candidate grid for internal_cv_v2 with true mathematical complexity ordering."""
    if horizon > 1 and resolved_mode != "tabular":
        return [
            {
                "name": "mh_lin_struct_mono", "complexity_rank": 1,
                "n_knots": 0, "alpha_lin": 1e-2, "alpha_meteo": 1e-1, "alpha_spl": 1e-1,
                "max_interactions": 0, "alpha_int": 0.1, "structured": True, "use_mono": True,
                "anchor": "none", "lead_damp": 1.0,
            },
            {
                "name": "mh_lin_mm30_damp", "complexity_rank": 2,
                "n_knots": 0, "alpha_lin": 1e-2, "alpha_meteo": 1e-1, "alpha_spl": 1e-1,
                "max_interactions": 0, "alpha_int": 0.1, "structured": True, "use_mono": True,
                "anchor": "mm30", "lead_damp": 0.90,
            },
            {
                "name": "mh_lin_kalman_damp", "complexity_rank": 3,
                "n_knots": 0, "alpha_lin": 1e-2, "alpha_meteo": 1e-1, "alpha_spl": 1e-1,
                "max_interactions": 0, "alpha_int": 0.1, "structured": True, "use_mono": True,
                "anchor": "kalman", "lead_damp": 0.90,
            },
            {
                "name": "mh_spline_struct_mono", "complexity_rank": 4,
                "n_knots": 6, "alpha_lin": 1e-2, "alpha_meteo": 5e-1, "alpha_spl": 5e-1,
                "max_interactions": 0, "alpha_int": 0.1, "structured": True, "use_mono": True,
                "anchor": "none", "lead_damp": 1.0,
            },
            {
                "name": "mh_spline_mm30_damp", "complexity_rank": 5,
                "n_knots": 6, "alpha_lin": 1e-2, "alpha_meteo": 5e-1, "alpha_spl": 5e-1,
                "max_interactions": 0, "alpha_int": 0.1, "structured": True, "use_mono": True,
                "anchor": "mm30", "lead_damp": 0.90,
            },
            {
                "name": "mh_spline_kalman_damp", "complexity_rank": 6,
                "n_knots": 6, "alpha_lin": 1e-2, "alpha_meteo": 5e-1, "alpha_spl": 5e-1,
                "max_interactions": 0, "alpha_int": 0.1, "structured": True, "use_mono": True,
                "anchor": "kalman", "lead_damp": 0.90,
            },
        ]
    return _candidate_grid(resolved_mode, horizon)


class AdaptiveStructuredHuberRegressor(BaseEstimator, RegressorMixin):
    """Unified multimode regressor with internal structure selection and exact L-BFGS-B solver."""

    def __init__(
        self,
        mode: str = "auto",
        horizon: int = 1,
        weather_mode: str = "auto",
        selection: str = "internal_cv",
        fixed_config: dict[str, Any] | None = None,
        delta: float = 1.35,
        parsimony_tol: float = 0.005,
        stability_weight: float = 0.25,
        random_state: int = 42,
        max_iter: int = 400,
        tol: float = 1e-7,
        store_train_diagnostics: bool = False,
    ) -> None:
        self.mode = mode
        self.horizon = horizon
        self.weather_mode = weather_mode
        self.selection = selection
        self.fixed_config = fixed_config
        self.delta = delta
        self.parsimony_tol = parsimony_tol
        self.stability_weight = stability_weight
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.store_train_diagnostics = store_train_diagnostics

    def _resolve_mode(self, metadata: dict[str, Any] | None) -> str:
        if self.mode in ("espaciotemporal", "intraplanta", "tabular"):
            return self.mode
        if metadata is None:
            return "tabular"
        groups = metadata.get("groups")
        plants = metadata.get("plant_ids")
        if groups is not None and len(np.unique(groups)) >= 3:
            return "espaciotemporal"
        if plants is not None:
            return "intraplanta" if len(np.unique(plants)) == 1 else "espaciotemporal"
        if metadata.get("dates") is not None:
            return "intraplanta"
        return "tabular"

    @staticmethod
    def _extract_anchor(
        n_samples: int, cfg: dict[str, Any], metadata: dict[str, Any] | None
    ) -> np.ndarray:
        anchor_type = cfg.get("anchor", "none")
        if anchor_type == "none" or metadata is None:
            return np.zeros(n_samples, dtype=np.float64)
        key = "anchor_mm30" if anchor_type == "mm30" else "anchor_kalman"
        arr = metadata.get(key)
        if arr is None:
            return np.zeros(n_samples, dtype=np.float64)
        arr_f = np.asarray(arr, dtype=np.float64).ravel()
        return np.where(np.isfinite(arr_f), arr_f, 0.0)

    def _objective_and_grad_debug(
        self,
        params: np.ndarray,
        Z_design: np.ndarray,
        z_y: np.ndarray,
        penalties: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Exposed analytical objective and gradient for mathematical verification."""
        n_samples = Z_design.shape[0]
        b = params[0]
        w = params[1:]
        r = z_y - (b + Z_design @ w)
        h_val, h_grad_r = _huber_val_and_grad(r, self.delta)
        loss = float(np.mean(h_val) + 0.5 * np.sum(penalties * (w * w)))
        g_pred = -h_grad_r / n_samples
        g_b = float(np.sum(g_pred))
        g_w = Z_design.T @ g_pred + penalties * w
        return loss, np.concatenate(([g_b], g_w))

    @staticmethod
    def _build_spline_basis_fit(
        Z_lin: np.ndarray, n_knots: int
    ) -> tuple[np.ndarray, list[tuple[int, float, float, float]]]:
        if n_knots <= 0:
            return np.empty((Z_lin.shape[0], 0), dtype=np.float64), []
        qs = np.linspace(0.0, 1.0, n_knots + 2)[1:-1]
        cols = []
        specs: list[tuple[int, float, float, float]] = []
        for j in range(Z_lin.shape[1]):
            col = Z_lin[:, j]
            if np.unique(col).size <= 3:
                continue
            for t in np.unique(np.quantile(col, qs)):
                hinge = np.maximum(0.0, col - float(t))
                h_std = float(np.std(hinge))
                if h_std <= 1e-8:
                    continue
                h_mean = float(np.mean(hinge))
                cols.append((hinge - h_mean) / h_std)
                specs.append((j, float(t), h_mean, h_std))
        if not cols:
            return np.empty((Z_lin.shape[0], 0), dtype=np.float64), []
        return np.column_stack(cols), specs

    @staticmethod
    def _transform_splines(
        Z_lin: np.ndarray, specs: list[tuple[int, float, float, float]]
    ) -> np.ndarray:
        if not specs:
            return np.empty((Z_lin.shape[0], 0), dtype=np.float64)
        return np.column_stack(
            [(np.maximum(0.0, Z_lin[:, j] - t) - m) / s for j, t, m, s in specs]
        )

    @staticmethod
    def _pair_terms(zj: np.ndarray, zl: np.ndarray) -> list[np.ndarray]:
        pj, pl = np.maximum(0.0, zj), np.maximum(0.0, zl)
        return [zj * zl, pj * zl, zj * pl, pj * pl]

    def _screen_interactions(
        self, Z_lin: np.ndarray, residuals: np.ndarray, active_mask: np.ndarray, max_int: int
    ) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[float, float]]]:
        n_samples, n_features = Z_lin.shape
        if max_int <= 0 or n_features < 2:
            return np.empty((n_samples, 0), dtype=np.float64), [], []
        w_h = np.where(
            np.abs(residuals) <= self.delta,
            1.0,
            self.delta / np.maximum(np.abs(residuals), 1e-12),
        )
        rw = w_h * residuals
        rw_c = rw - np.mean(rw)
        rw_norm = float(np.linalg.norm(rw_c))
        if rw_norm <= 1e-12:
            return np.empty((n_samples, 0), dtype=np.float64), [], []

        active = [j for j in range(n_features) if active_mask[j]]
        scored: list[tuple[float, int, int]] = []
        for ia in range(len(active)):
            j = active[ia]
            zj, pj = Z_lin[:, j], np.maximum(0.0, Z_lin[:, j])
            for ib in range(ia + 1, len(active)):
                l = active[ib]
                zl = Z_lin[:, l]
                best_c = 0.0
                for term in (zj * zl, pj * np.maximum(0.0, zl)):
                    tc = term - np.mean(term)
                    tn = float(np.linalg.norm(tc))
                    if tn > 1e-12:
                        c = abs(float(np.dot(tc, rw_c)) / (tn * rw_norm))
                        if c > best_c:
                            best_c = c
                if best_c > 0.0:
                    scored.append((best_c, j, l))
        scored.sort(key=lambda x: (-x[0], x[1], x[2]))

        cols, stats, kept = [], [], []
        for _, j, l in scored[:max_int]:
            p_cols, p_stats, ok = [], [], True
            for term in self._pair_terms(Z_lin[:, j], Z_lin[:, l]):
                ts = float(np.std(term))
                if ts <= 1e-8:
                    ok = False
                    break
                tm = float(np.mean(term))
                p_cols.append((term - tm) / ts)
                p_stats.append((tm, ts))
            if ok:
                kept.append((j, l))
                cols.extend(p_cols)
                stats.extend(p_stats)
        if not cols:
            return np.empty((n_samples, 0), dtype=np.float64), [], []
        return np.column_stack(cols), kept, stats

    def _transform_interactions(
        self, Z_lin: np.ndarray, pairs: list[tuple[int, int]], stats: list[tuple[float, float]]
    ) -> np.ndarray:
        if not pairs:
            return np.empty((Z_lin.shape[0], 0), dtype=np.float64)
        cols, idx = [], 0
        for j, l in pairs:
            for term in self._pair_terms(Z_lin[:, j], Z_lin[:, l]):
                tm, ts = stats[idx]
                cols.append((term - tm) / ts)
                idx += 1
        return np.column_stack(cols)

    def _fit_single(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cfg: dict[str, Any],
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        n_samples, n_features = X.shape
        anchor = self._extract_anchor(n_samples, cfg, metadata)
        r_target = y - anchor

        x_mean = np.mean(X, axis=0)
        x_std = np.std(X, axis=0)
        active_mask = x_std > 1e-8
        x_scale = np.where(active_mask, x_std, 1.0)
        Z_lin = (X - x_mean) / x_scale

        m_r, s_r = _robust_target_stats(r_target)
        z_r = (r_target - m_r) / s_r

        Z_spl, spline_specs = self._build_spline_basis_fit(Z_lin, int(cfg.get("n_knots", 0)))
        Z_add = np.hstack([Z_lin, Z_spl]) if Z_spl.shape[1] > 0 else Z_lin

        structured = bool(cfg.get("structured", False))
        use_mono = bool(cfg.get("use_mono", False))
        meteo_idx = set(metadata.get("meteo_idx", ()) if metadata else ())
        mono_idx = set(metadata.get("monotone_pos_idx", ()) if metadata else ())
        if cfg.get("anchor", "none") != "none" and metadata and metadata.get("cf30_idx") is not None:
            mono_idx = mono_idx - {int(metadata["cf30_idx"])}

        alpha_lin = float(cfg.get("alpha_lin", 1e-2))
        alpha_meteo = float(cfg.get("alpha_meteo", alpha_lin)) if structured else alpha_lin
        alpha_spl = float(cfg.get("alpha_spl", alpha_lin))
        alpha_int = float(cfg.get("alpha_int", alpha_spl))

        pen_lin = np.array(
            [alpha_meteo if (structured and j in meteo_idx) else alpha_lin for j in range(n_features)],
            dtype=np.float64,
        )
        pen_add = np.concatenate([pen_lin, np.full(Z_spl.shape[1], alpha_spl, dtype=np.float64)])

        bounds_add: list[tuple[float | None, float | None]] = [(None, None)]
        bounds_add.extend(
            (0.0, None) if (use_mono and j in mono_idx) else (None, None)
            for j in range(n_features)
        )
        bounds_add.extend((None, None) for _ in range(Z_spl.shape[1]))

        init = np.zeros(Z_add.shape[1] + 1, dtype=np.float64)
        res = minimize(
            lambda p: self._objective_and_grad_debug(p, Z_add, z_r, pen_add),
            init,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds_add,
            options={"maxiter": self.max_iter, "ftol": self.tol, "gtol": self.tol},
        )
        b0, w_add = float(res.x[0]), res.x[1:].copy()
        n_iter, converged = int(res.nit), bool(res.success)

        max_int = int(cfg.get("max_interactions", 0))
        int_pairs, int_stats = [], []
        if max_int > 0:
            res_add = z_r - (b0 + Z_add @ w_add)
            Z_int, int_pairs, int_stats = self._screen_interactions(Z_lin, res_add, active_mask, max_int)
            if Z_int.shape[1] > 0:
                Z_full = np.hstack([Z_add, Z_int])
                pen_full = np.concatenate([pen_add, np.full(Z_int.shape[1], alpha_int, dtype=np.float64)])
                bounds_full = bounds_add + [(None, None)] * Z_int.shape[1]
                res2 = minimize(
                    lambda p: self._objective_and_grad_debug(p, Z_full, z_r, pen_full),
                    np.zeros(Z_full.shape[1] + 1, dtype=np.float64),
                    method="L-BFGS-B",
                    jac=True,
                    bounds=bounds_full,
                    options={"maxiter": self.max_iter, "ftol": self.tol, "gtol": self.tol},
                )
                b0, w_full = float(res2.x[0]), res2.x[1:].copy()
                n_iter += int(res2.nit)
                converged = bool(res2.success)
                pen_used, Z_used = pen_full, Z_full
            else:
                w_full, pen_used, Z_used = w_add, pen_add, Z_add
        else:
            w_full, pen_used, Z_used = w_add, pen_add, Z_add

        state: dict[str, Any] = {
            "cfg": cfg, "x_mean": x_mean, "x_scale": x_scale, "m_r": m_r, "s_r": s_r,
            "spline_specs": spline_specs, "int_pairs": int_pairs, "int_stats": int_stats,
            "intercept": b0, "weights": w_full, "n_iter": n_iter, "converged": converged,
            "n_expanded": int(n_features + len(spline_specs) + 4 * len(int_pairs)),
            "penalties": pen_used,
        }
        if self.store_train_diagnostics:
            state["Z_design"] = Z_used
            state["z_r"] = z_r
        return state

    def _predict_single(
        self, state: dict[str, Any], X: np.ndarray, metadata: dict[str, Any] | None
    ) -> np.ndarray:
        Z_lin = (X - state["x_mean"]) / state["x_scale"]
        blocks = [Z_lin]
        if state["spline_specs"]:
            blocks.append(self._transform_splines(Z_lin, state["spline_specs"]))
        if state["int_pairs"]:
            blocks.append(self._transform_interactions(Z_lin, state["int_pairs"], state["int_stats"]))
        Z_full = np.hstack(blocks) if len(blocks) > 1 else Z_lin
        z_hat = state["intercept"] + Z_full @ state["weights"]
        r_hat = state["m_r"] + state["s_r"] * z_hat

        cfg = state["cfg"]
        anchor = self._extract_anchor(X.shape[0], cfg, metadata)
        if cfg.get("anchor", "none") != "none" and metadata is not None and metadata.get("lead") is not None:
            lead_arr = np.asarray(metadata["lead"], dtype=np.float64)
            gamma = float(cfg.get("lead_damp", 1.0)) ** np.maximum(0.0, lead_arr - 1.0)
        else:
            gamma = 1.0
        return anchor + gamma * r_hat

    def _build_internal_splits(
        self, n_samples: int, resolved_mode: str, metadata: dict[str, Any] | None
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if metadata is not None and metadata.get("dates") is not None:
            dates = np.asarray(metadata["dates"])
            uniq_dates = np.sort(np.unique(dates))
            if uniq_dates.size >= 25:
                n_val = max(5, int(0.15 * uniq_dates.size))
                gap = max(1, min(14, int(0.05 * uniq_dates.size)))
                groups = np.asarray(metadata["groups"]) if metadata.get("groups") is not None else None
                uniq_g = np.unique(groups) if groups is not None else np.array([])
                splits = []
                for k in range(3):
                    val_end_idx = uniq_dates.size - k * n_val
                    val_start_idx = max(0, val_end_idx - n_val)
                    tr_end_idx = val_start_idx - gap
                    if tr_end_idx < 5 or val_start_idx >= val_end_idx:
                        continue
                    tr_cut = uniq_dates[tr_end_idx]
                    va_lo, va_hi = uniq_dates[val_start_idx], uniq_dates[val_end_idx - 1]
                    tr_mask = dates <= tr_cut
                    va_mask = (dates >= va_lo) & (dates <= va_hi)
                    if resolved_mode == "espaciotemporal" and groups is not None and uniq_g.size >= 3:
                        hold_g = uniq_g[k % uniq_g.size]
                        tr_mask = tr_mask & (groups != hold_g)
                        va_mask = va_mask & (groups == hold_g)
                    tr_idx, va_idx = np.where(tr_mask)[0], np.where(va_mask)[0]
                    if tr_idx.size >= 15 and va_idx.size >= 5:
                        splits.append((tr_idx, va_idx))
                if splits:
                    return splits

        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n_samples)
        folds = np.array_split(perm, 3)
        out = []
        for k in range(3):
            va_idx = folds[k]
            tr_idx = np.concatenate([folds[j] for j in range(3) if j != k])
            if tr_idx.size >= 10 and va_idx.size >= 3:
                out.append((tr_idx, va_idx))
        return out

    @staticmethod
    def _slice_metadata(metadata: dict[str, Any] | None, idx: np.ndarray) -> dict[str, Any] | None:
        if metadata is None:
            return None
        out = {}
        n_ref = None
        for k, v in metadata.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1:
                if n_ref is None or v.shape[0] == n_ref:
                    n_ref = v.shape[0] if n_ref is None else n_ref
                    out[k] = v[idx]
                else:
                    out[k] = v
            else:
                out[k] = v
        return out

    def _build_internal_splits_v2(
        self, n_samples: int, resolved_mode: str, metadata: dict[str, Any] | None
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if metadata is not None and metadata.get("dates") is not None:
            dates = np.asarray(metadata["dates"])
            uniq_dates = np.sort(np.unique(dates))
            if uniq_dates.size >= 25:
                n_val = max(self.horizon, int(0.15 * uniq_dates.size))
                gap = max(self.horizon if uniq_dates.size >= 3 * self.horizon + 10 else 1, min(14, int(0.05 * uniq_dates.size)))
                groups = np.asarray(metadata["groups"]) if metadata.get("groups") is not None else None
                uniq_g = np.unique(groups) if groups is not None else np.array([])
                splits = []
                for k in range(3):
                    val_end_idx = uniq_dates.size - k * n_val
                    val_start_idx = max(0, val_end_idx - n_val)
                    tr_end_idx = val_start_idx - gap
                    if tr_end_idx < 5 or val_start_idx >= val_end_idx:
                        continue
                    tr_cut = uniq_dates[tr_end_idx]
                    va_lo, va_hi = uniq_dates[val_start_idx], uniq_dates[val_end_idx - 1]
                    tr_mask = dates <= tr_cut
                    va_mask = (dates >= va_lo) & (dates <= va_hi)
                    if resolved_mode == "espaciotemporal" and groups is not None and uniq_g.size >= 3:
                        hold_g = uniq_g[k % uniq_g.size]
                        tr_mask = tr_mask & (groups != hold_g)
                        va_mask = va_mask & (groups == hold_g)
                    tr_idx, va_idx = np.where(tr_mask)[0], np.where(va_mask)[0]
                    if tr_idx.size >= 15 and va_idx.size >= 5:
                        splits.append((tr_idx, va_idx))
                if splits:
                    return splits
        return self._build_internal_splits(n_samples, resolved_mode, metadata)

    def _prepare_val_metadata_v2(
        self, meta_va: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Freeze multihorizon anchors at validation block origin and set block-relative leads."""
        if meta_va is None or self.horizon <= 1 or meta_va.get("dates") is None:
            return meta_va
        out = dict(meta_va)
        dates_va = np.asarray(out["dates"])
        va_lo = np.min(dates_va)
        if np.issubdtype(dates_va.dtype, np.datetime64):
            diff_days = ((dates_va - va_lo) / np.timedelta64(1, "D")).astype(np.float64)
        else:
            diff_days = (dates_va - va_lo).astype(np.float64)
        out["lead"] = np.clip(diff_days + 1.0, 1.0, float(self.horizon))

        plants = np.asarray(out["plant_ids"]) if out.get("plant_ids") is not None else np.zeros(len(dates_va), dtype=int)
        for key in ("anchor_mm30", "anchor_kalman"):
            if out.get(key) is not None:
                arr = np.asarray(out[key], dtype=np.float64).copy()
                for p in np.unique(plants):
                    pmask = plants == p
                    p_idx = np.where(pmask)[0]
                    first_pos = p_idx[int(np.argmin(dates_va[p_idx]))]
                    arr[pmask] = arr[first_pos]
                out[key] = arr
        return out

    @staticmethod
    def _eval_val_mae_v2(
        pred_va: np.ndarray, y_va: np.ndarray, resolved_mode: str, meta_va: dict[str, Any] | None
    ) -> float:
        ae = np.abs(pred_va - y_va)
        if resolved_mode == "espaciotemporal" and meta_va is not None and meta_va.get("plant_ids") is not None:
            plants = np.asarray(meta_va["plant_ids"])
            uniq_p = np.unique(plants)
            if uniq_p.size >= 2:
                return float(np.mean([np.mean(ae[plants == p]) for p in uniq_p]))
        return float(np.mean(ae))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> "AdaptiveStructuredHuberRegressor":
        t0 = time.perf_counter()
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        X_arr = _as_2d_float(X)
        y_arr = _as_1d_float(y)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X and y must have the same number of rows: got {X_arr.shape[0]} and {y_arr.shape[0]}."
            )
        self.n_features_in_ = int(X_arr.shape[1])
        resolved_mode = self._resolve_mode(metadata)

        cv_records: list[dict[str, Any]] = []
        if self.selection == "fixed" and self.fixed_config is not None:
            chosen_cfg = dict(self.fixed_config)
        elif self.selection == "internal_cv_v2":
            candidates = _candidate_grid_v2(resolved_mode, self.horizon)
            splits = self._build_internal_splits_v2(X_arr.shape[0], resolved_mode, metadata)
            if not splits:
                chosen_cfg = candidates[0]
            else:
                for cfg in candidates:
                    losses = []
                    for tr_idx, va_idx in splits:
                        meta_tr = self._slice_metadata(metadata, tr_idx)
                        meta_va = self._prepare_val_metadata_v2(self._slice_metadata(metadata, va_idx))
                        st = self._fit_single(X_arr[tr_idx], y_arr[tr_idx], cfg, meta_tr)
                        pred_va = self._predict_single(st, X_arr[va_idx], meta_va)
                        losses.append(self._eval_val_mae_v2(pred_va, y_arr[va_idx], resolved_mode, meta_va))
                    cv_mean = float(np.mean(losses))
                    cv_std = float(np.std(losses))
                    score = cv_mean + self.stability_weight * cv_std
                    cv_records.append({
                        "name": cfg["name"],
                        "complexity_rank": int(cfg["complexity_rank"]),
                        "cv_mae": cv_mean,
                        "cv_std": cv_std,
                        "score": score,
                    })
                min_score = min(r["score"] for r in cv_records)
                tol_cut = min_score * (1.0 + self.parsimony_tol)
                eligible = [r for r in cv_records if r["score"] <= tol_cut]
                best_rec = min(eligible, key=lambda r: (r["complexity_rank"], r["score"]))
                chosen_cfg = next(c for c in candidates if c["name"] == best_rec["name"])
        else:
            candidates = _candidate_grid(resolved_mode, self.horizon)
            splits = self._build_internal_splits(X_arr.shape[0], resolved_mode, metadata)
            if not splits:
                chosen_cfg = candidates[0]
            else:
                for cfg in candidates:
                    losses = []
                    for tr_idx, va_idx in splits:
                        meta_tr = self._slice_metadata(metadata, tr_idx)
                        meta_va = self._slice_metadata(metadata, va_idx)
                        st = self._fit_single(X_arr[tr_idx], y_arr[tr_idx], cfg, meta_tr)
                        pred_va = self._predict_single(st, X_arr[va_idx], meta_va)
                        losses.append(float(np.mean(np.abs(pred_va - y_arr[va_idx]))))
                    cv_records.append({
                        "name": cfg["name"],
                        "complexity_rank": int(cfg["complexity_rank"]),
                        "cv_mae": float(np.mean(losses)),
                    })
                min_mae = min(r["cv_mae"] for r in cv_records)
                tol_cut = min_mae * (1.0 + self.parsimony_tol)
                eligible = [r for r in cv_records if r["cv_mae"] <= tol_cut]
                best_rec = min(eligible, key=lambda r: (r["complexity_rank"], r["cv_mae"]))
                chosen_cfg = next(c for c in candidates if c["name"] == best_rec["name"])

        final_state = self._fit_single(X_arr, y_arr, chosen_cfg, metadata)
        self._state = final_state
        self.resolved_mode_ = resolved_mode
        self.selected_config_ = dict(chosen_cfg)
        self.cv_results_ = cv_records
        self.intercept_ = float(final_state["intercept"])
        self.coef_ = final_state["weights"][: X_arr.shape[1]].copy()
        self.n_iter_ = int(final_state["n_iter"])
        self.converged_ = bool(final_state["converged"])
        self.n_features_expanded_ = int(final_state["n_expanded"])
        self.fit_time_s_ = float(time.perf_counter() - t0)
        return self

    def predict(
        self, X: np.ndarray, metadata: dict[str, Any] | None = None
    ) -> np.ndarray:
        if not hasattr(self, "_state"):
            raise NotFittedError("This AdaptiveStructuredHuberRegressor instance is not fitted yet.")
        X_arr = _as_2d_float(X)
        if hasattr(self, "n_features_in_") and X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, but estimator expects {self.n_features_in_}."
            )
        return self._predict_single(self._state, X_arr, metadata)

"""Structured Additive Huber Regressor in pure NumPy/SciPy.

Implements:
- Scale- and shift-equivariant Huber loss + group-differentiated L2 regularization.
- Quantile-based piecewise-linear spline basis per continuous covariate.
- Train-only Huber-weighted residual screening of pairwise bilinear interactions.
- Optional non-negativity box constraints on physically monotone linear slopes.
- Exact analytical gradient solved via L-BFGS-B.
- Causal Kalman feature builder for time-series residual anchoring (Prototype F).
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

try:
    from prototipos_mecanismos import causal_kalman_cf
except ImportError:
    from src.prototipos_mecanismos import causal_kalman_cf


def _as_2d_float(X: np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for X, got shape {arr.shape}")
    return arr


def _as_1d_float(y: np.ndarray) -> np.ndarray:
    return np.asarray(y, dtype=np.float64).ravel()


def _robust_target_stats(y: np.ndarray) -> tuple[float, float]:
    m_y = float(np.median(y))
    mad = float(np.median(np.abs(y - m_y)))
    if mad > 0.0:
        sigma_y = max(1.4826 * mad, 1e-6)
    else:
        sigma_y = max(float(np.std(y)), 1e-6)
    return m_y, sigma_y


def _huber_val_and_grad(r: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
    abs_r = np.abs(r)
    quad = abs_r <= delta
    val = np.where(quad, 0.5 * r * r, delta * (abs_r - 0.5 * delta))
    grad_r = np.where(quad, r, delta * np.sign(r))
    return val, grad_r


def build_causal_kalman_feature(
    y_series: np.ndarray,
    q: float = 0.01,
    r: float = 0.05,
    huber_k: float = 2.0,
    init_mean: float = 0.0,
    init_var: float = 1.0,
) -> np.ndarray:
    """Build strictly causal robust Kalman level feature for time series."""
    return causal_kalman_cf(
        y_series,
        q=q,
        r=r,
        robust=True,
        huber_k=huber_k,
        init_mean=init_mean,
        init_var=init_var,
    )


class StructuredAdditiveHuberRegressor:
    """Equivariant Huber regressor with quantile splines, group L2, monotonicity, and interactions."""

    def __init__(
        self,
        delta: float = 1.35,
        n_knots: int = 0,
        alpha_lin: float = 1e-2,
        alpha_meteo: float | None = None,
        alpha_spl: float = 1e-1,
        max_interactions: int = 0,
        alpha_int: float = 1e-1,
        monotone_pos_idx: tuple[int, ...] | list[int] | None = None,
        meteo_idx: tuple[int, ...] | list[int] | None = None,
        max_iter: int = 400,
        tol: float = 1e-7,
    ) -> None:
        self.delta = float(delta)
        self.n_knots = int(n_knots)
        self.alpha_lin = float(alpha_lin)
        self.alpha_meteo = float(alpha_meteo) if alpha_meteo is not None else float(alpha_lin)
        self.alpha_spl = float(alpha_spl)
        self.max_interactions = int(max_interactions)
        self.alpha_int = float(alpha_int)
        self.monotone_pos_idx = (
            tuple(int(i) for i in monotone_pos_idx)
            if monotone_pos_idx is not None
            else ()
        )
        self.meteo_idx = (
            tuple(int(i) for i in meteo_idx) if meteo_idx is not None else ()
        )
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def _solve_lbfgs(
        self,
        Z_design: np.ndarray,
        z_y: np.ndarray,
        penalties: np.ndarray,
        bounds: list[tuple[float | None, float | None]],
    ) -> tuple[float, np.ndarray, int, bool]:
        n_samples, n_cols = Z_design.shape
        delta = self.delta

        def obj_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
            b = params[0]
            w = params[1:]
            r = z_y - (b + Z_design @ w)
            h_val, h_grad_r = _huber_val_and_grad(r, delta)
            loss = float(np.mean(h_val) + 0.5 * np.sum(penalties * (w * w)))
            g_pred = -h_grad_r / n_samples
            g_b = float(np.sum(g_pred))
            g_w = Z_design.T @ g_pred + penalties * w
            return loss, np.concatenate(([g_b], g_w))

        init = np.zeros(n_cols + 1, dtype=np.float64)
        res = minimize(
            obj_and_grad,
            init,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": self.tol, "gtol": self.tol},
        )
        return float(res.x[0]), res.x[1:].copy(), int(res.nit), bool(res.success)

    def _build_spline_basis_fit(
        self, Z_lin: np.ndarray
    ) -> tuple[np.ndarray, list[tuple[int, float, float, float]]]:
        if self.n_knots <= 0:
            return np.empty((Z_lin.shape[0], 0), dtype=np.float64), []

        qs = np.linspace(0.0, 1.0, self.n_knots + 2)[1:-1]
        cols = []
        specs: list[tuple[int, float, float, float]] = []
        n_features = Z_lin.shape[1]

        for j in range(n_features):
            col = Z_lin[:, j]
            if np.unique(col).size <= 3:
                continue
            knots = np.unique(np.quantile(col, qs))
            for t in knots:
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
        cols = [
            (np.maximum(0.0, Z_lin[:, j] - t) - mean) / scale
            for j, t, mean, scale in specs
        ]
        return np.column_stack(cols)

    @staticmethod
    def _pair_terms(z_j: np.ndarray, z_l: np.ndarray) -> list[np.ndarray]:
        pj = np.maximum(0.0, z_j)
        pl = np.maximum(0.0, z_l)
        return [z_j * z_l, pj * z_l, z_j * pl, pj * pl]

    def _screen_and_build_interactions(
        self,
        Z_lin: np.ndarray,
        residuals: np.ndarray,
        active_mask: np.ndarray,
    ) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[float, float]]]:
        n_samples, n_features = Z_lin.shape
        if self.max_interactions <= 0 or n_features < 2:
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

        active_idx = [j for j in range(n_features) if active_mask[j]]
        scored: list[tuple[float, int, int]] = []
        for idx_a in range(len(active_idx)):
            j = active_idx[idx_a]
            zj = Z_lin[:, j]
            pj = np.maximum(0.0, zj)
            for idx_b in range(idx_a + 1, len(active_idx)):
                l = active_idx[idx_b]
                zl = Z_lin[:, l]
                best_corr = 0.0
                for term in (zj * zl, pj * np.maximum(0.0, zl)):
                    tc = term - np.mean(term)
                    t_norm = float(np.linalg.norm(tc))
                    if t_norm > 1e-12:
                        c = abs(float(np.dot(tc, rw_c)) / (t_norm * rw_norm))
                        if c > best_corr:
                            best_corr = c
                if best_corr > 0.0:
                    scored.append((best_corr, j, l))

        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        selected_pairs = [(j, l) for _, j, l in scored[: self.max_interactions]]

        cols = []
        norm_stats: list[tuple[float, float]] = []
        kept_pairs: list[tuple[int, int]] = []
        for j, l in selected_pairs:
            terms = self._pair_terms(Z_lin[:, j], Z_lin[:, l])
            pair_cols = []
            pair_stats = []
            valid_pair = True
            for term in terms:
                t_std = float(np.std(term))
                if t_std <= 1e-8:
                    valid_pair = False
                    break
                t_mean = float(np.mean(term))
                pair_cols.append((term - t_mean) / t_std)
                pair_stats.append((t_mean, t_std))
            if valid_pair:
                kept_pairs.append((j, l))
                cols.extend(pair_cols)
                norm_stats.extend(pair_stats)

        if not cols:
            return np.empty((n_samples, 0), dtype=np.float64), [], []
        return np.column_stack(cols), kept_pairs, norm_stats

    def _transform_interactions(
        self,
        Z_lin: np.ndarray,
        pairs: list[tuple[int, int]],
        norm_stats: list[tuple[float, float]],
    ) -> np.ndarray:
        if not pairs:
            return np.empty((Z_lin.shape[0], 0), dtype=np.float64)
        cols = []
        stat_idx = 0
        for j, l in pairs:
            for term in self._pair_terms(Z_lin[:, j], Z_lin[:, l]):
                t_mean, t_std = norm_stats[stat_idx]
                cols.append((term - t_mean) / t_std)
                stat_idx += 1
        return np.column_stack(cols)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StructuredAdditiveHuberRegressor":
        t0 = time.perf_counter()
        X_arr = _as_2d_float(X)
        y_arr = _as_1d_float(y)
        n_samples, n_features = X_arr.shape

        x_mean = np.mean(X_arr, axis=0)
        x_std = np.std(X_arr, axis=0)
        active_mask = x_std > 1e-8
        x_scale = np.where(active_mask, x_std, 1.0)
        Z_lin = (X_arr - x_mean) / x_scale

        m_y, sigma_y = _robust_target_stats(y_arr)
        z_y = (y_arr - m_y) / sigma_y

        Z_spl, spline_specs = self._build_spline_basis_fit(Z_lin)
        Z_add = np.hstack([Z_lin, Z_spl]) if Z_spl.shape[1] > 0 else Z_lin

        meteo_set = {i for i in self.meteo_idx if 0 <= i < n_features}
        mono_set = {i for i in self.monotone_pos_idx if 0 <= i < n_features}

        pen_lin = np.array(
            [self.alpha_meteo if j in meteo_set else self.alpha_lin for j in range(n_features)],
            dtype=np.float64,
        )
        pen_spl = np.full(Z_spl.shape[1], self.alpha_spl, dtype=np.float64)
        pen_add = np.concatenate([pen_lin, pen_spl])

        bounds_add: list[tuple[float | None, float | None]] = [(None, None)]
        bounds_add.extend(
            (0.0, None) if j in mono_set else (None, None) for j in range(n_features)
        )
        bounds_add.extend((None, None) for _ in range(Z_spl.shape[1]))

        b0, w_add, n_iter, converged = self._solve_lbfgs(Z_add, z_y, pen_add, bounds_add)

        int_pairs: list[tuple[int, int]] = []
        int_stats: list[tuple[float, float]] = []
        if self.max_interactions > 0:
            res_add = z_y - (b0 + Z_add @ w_add)
            Z_int, int_pairs, int_stats = self._screen_and_build_interactions(
                Z_lin, res_add, active_mask
            )
            if Z_int.shape[1] > 0:
                Z_full = np.hstack([Z_add, Z_int])
                pen_int = np.full(Z_int.shape[1], self.alpha_int, dtype=np.float64)
                pen_full = np.concatenate([pen_add, pen_int])
                bounds_full = bounds_add + [(None, None)] * Z_int.shape[1]
                b0, w_full, n_iter2, converged = self._solve_lbfgs(
                    Z_full, z_y, pen_full, bounds_full
                )
                n_iter += n_iter2
            else:
                w_full = w_add
        else:
            w_full = w_add

        self.x_mean_ = x_mean
        self.x_scale_ = x_scale
        self.m_y_ = m_y
        self.sigma_y_ = sigma_y
        self.n_features_in_ = n_features
        self.spline_specs_ = spline_specs
        self.selected_interactions_ = int_pairs
        self.interaction_stats_ = int_stats
        self.intercept_ = b0
        self.coef_ = w_full[:n_features].copy()
        self.weights_full_ = w_full.copy()
        self.n_features_expanded_ = int(n_features + len(spline_specs) + 4 * len(int_pairs))
        self.n_iter_ = n_iter
        self.converged_ = converged
        self.fit_time_s_ = float(time.perf_counter() - t0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = _as_2d_float(X)
        Z_lin = (X_arr - self.x_mean_) / self.x_scale_
        blocks = [Z_lin]
        if self.spline_specs_:
            blocks.append(self._transform_splines(Z_lin, self.spline_specs_))
        if self.selected_interactions_:
            blocks.append(
                self._transform_interactions(
                    Z_lin, self.selected_interactions_, self.interaction_stats_
                )
            )
        Z_full = np.hstack(blocks) if len(blocks) > 1 else Z_lin
        z_hat = self.intercept_ + Z_full @ self.weights_full_
        return self.m_y_ + self.sigma_y_ * z_hat

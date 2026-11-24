"""Pure NumPy/SciPy prototypes for mechanism-diagnostic experiments.

Implements:
- GlobalHuberEquivariant: shift- and positive-scale-equivariant Huber + L2 reference.
- StudentTARDRegressor: Student-t EM regression with optional ARD feature precisions.
- AsymmetricMonotoneHuberRegressor: asymmetric Huber loss + structured L2 + non-negativity
  constraints via FISTA proximal gradient.
- causal_ewma_cf / causal_kalman_cf: strictly causal 1D capacity-factor level filters
  (conventional and Huber-robust Kalman) supporting NaN gaps.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _as_2d_float(X: np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for X, got shape {arr.shape}")
    return arr


def _as_1d_float(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64).ravel()
    return arr


def _robust_target_stats(y: np.ndarray) -> tuple[float, float]:
    m_y = float(np.median(y))
    mad = float(np.median(np.abs(y - m_y)))
    if mad > 0.0:
        sigma_y = max(1.4826 * mad, 1e-6)
    else:
        std_y = float(np.std(y))
        sigma_y = max(std_y, 1e-6)
    return m_y, sigma_y


def _standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_mean = np.mean(X, axis=0)
    x_std = np.std(X, axis=0)
    x_scale = np.where(x_std > 1e-8, x_std, 1.0)
    Z = (X - x_mean) / x_scale
    return Z, x_mean, x_scale


def _huber_val_and_grad(r: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """Return rho_delta(r) and d/dr rho_delta(r) for residual r = z - z_hat."""
    abs_r = np.abs(r)
    quad = abs_r <= delta
    val = np.where(quad, 0.5 * r * r, delta * (abs_r - 0.5 * delta))
    grad_r = np.where(quad, r, delta * np.sign(r))
    return val, grad_r


class GlobalHuberEquivariant:
    """Shift- and positive-scale-equivariant symmetric Huber + L2 regressor."""

    def __init__(
        self,
        delta: float = 1.35,
        alpha: float = 1e-2,
        max_iter: int = 500,
        tol: float = 1e-7,
    ) -> None:
        self.delta = float(delta)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GlobalHuberEquivariant":
        X_arr = _as_2d_float(X)
        y_arr = _as_1d_float(y)
        n_samples, n_features = X_arr.shape

        Z, x_mean, x_scale = _standardize_fit(X_arr)
        m_y, sigma_y = _robust_target_stats(y_arr)
        z_y = (y_arr - m_y) / sigma_y

        delta = self.delta
        alpha = self.alpha

        def obj_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
            b = params[0]
            w = params[1:]
            r = z_y - (b + Z @ w)
            h_val, h_grad_r = _huber_val_and_grad(r, delta)
            loss = float(np.mean(h_val) + 0.5 * alpha * np.dot(w, w))
            g_pred = -h_grad_r / n_samples
            g_b = float(np.sum(g_pred))
            g_w = Z.T @ g_pred + alpha * w
            return loss, np.concatenate(([g_b], g_w))

        init = np.zeros(n_features + 1, dtype=np.float64)
        res = minimize(
            obj_and_grad,
            init,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iter, "ftol": self.tol, "gtol": self.tol},
        )

        self.x_mean_ = x_mean
        self.x_scale_ = x_scale
        self.m_y_ = m_y
        self.sigma_y_ = sigma_y
        self.intercept_ = float(res.x[0])
        self.coef_ = res.x[1:].copy()
        self.n_iter_ = int(res.nit)
        self.converged_ = bool(res.success)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = _as_2d_float(X)
        Z = (X_arr - self.x_mean_) / self.x_scale_
        z_hat = self.intercept_ + Z @ self.coef_
        return self.m_y_ + self.sigma_y_ * z_hat


class StudentTARDRegressor:
    """Student-t EM linear regressor with optional ARD feature precisions."""

    def __init__(
        self,
        nu: float = 4.0,
        use_ard: bool = True,
        tau: float = 1.0,
        tau_0: float = 1e-8,
        alpha_init: float = 1e-2,
        max_iter: int = 200,
        tol: float = 1e-6,
    ) -> None:
        self.nu = float(nu)
        self.use_ard = bool(use_ard)
        self.tau = float(tau)
        self.tau_0 = float(tau_0)
        self.alpha_init = float(alpha_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StudentTARDRegressor":
        X_arr = _as_2d_float(X)
        y_arr = _as_1d_float(y)
        n_samples, n_features = X_arr.shape

        Z, x_mean, x_scale = _standardize_fit(X_arr)
        m_y, sigma_y = _robust_target_stats(y_arr)
        z_y = (y_arr - m_y) / sigma_y

        Z_aug = np.column_stack([np.ones(n_samples, dtype=np.float64), Z])
        alphas = np.full(n_features, self.alpha_init, dtype=np.float64)
        tau = max(self.tau, 1e-4)
        nu = self.nu

        theta = np.zeros(n_features + 1, dtype=np.float64)
        weights = np.ones(n_samples, dtype=np.float64)
        converged = False
        n_iter = 0

        for it in range(1, self.max_iter + 1):
            n_iter = it
            theta_prev = theta.copy()

            # E-step: Student-t scale mixture weights
            r = z_y - Z_aug @ theta
            weights = (nu + 1.0) / (nu + tau * (r * r))

            # M-step for theta and posterior covariance
            Zw = Z_aug * weights[:, None]
            prior_diag = np.concatenate(([self.tau_0], alphas))
            precision_mat = tau * (Z_aug.T @ Zw) + np.diag(prior_diag)
            rhs = tau * (Zw.T @ z_y)
            cov = np.linalg.pinv(precision_mat)
            theta = cov @ rhs

            # M-step for feature precisions (EM update)
            w_coef = theta[1:]
            diag_cov = np.maximum(np.diag(cov)[1:], 0.0)
            if self.use_ard:
                alphas = 1.0 / np.maximum(w_coef * w_coef + diag_cov, 1e-10)
            else:
                pooled = float(n_features) / max(
                    float(np.sum(w_coef * w_coef + diag_cov)), 1e-10
                )
                alphas = np.full(n_features, pooled, dtype=np.float64)
            alphas = np.clip(alphas, 1e-6, 1e8)

            # M-step for noise precision tau
            r_new = z_y - Z_aug @ theta
            var_proj = np.sum((Z_aug @ cov) * Z_aug, axis=1)
            denom = float(np.sum(weights * (r_new * r_new + var_proj)))
            tau = float(np.clip(n_samples / max(denom, 1e-10), 1e-6, 1e6))

            if np.max(np.abs(theta - theta_prev)) < self.tol and it >= 3:
                converged = True
                break

        self.x_mean_ = x_mean
        self.x_scale_ = x_scale
        self.m_y_ = m_y
        self.sigma_y_ = sigma_y
        self.intercept_ = float(theta[0])
        self.coef_ = theta[1:].copy()
        self.feature_precisions_ = alphas.copy()
        self.obs_weights_ = weights.copy()
        self.tau_ = float(tau)
        self.n_iter_ = n_iter
        self.converged_ = converged
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = _as_2d_float(X)
        Z = (X_arr - self.x_mean_) / self.x_scale_
        z_hat = self.intercept_ + Z @ self.coef_
        return self.m_y_ + self.sigma_y_ * z_hat


class AsymmetricMonotoneHuberRegressor:
    """Asymmetric Huber + structured L2 + non-negativity constraints via FISTA."""

    def __init__(
        self,
        tau_asym: float = 0.5,
        tau_mem: float = 1e-2,
        tau_meteo: float = 1e-2,
        delta: float = 1.35,
        monotone_pos_idx: tuple[int, ...] | list[int] | None = None,
        meteo_idx: tuple[int, ...] | list[int] | None = None,
        max_iter: int = 800,
        tol: float = 1e-7,
    ) -> None:
        self.tau_asym = float(tau_asym)
        self.tau_mem = float(tau_mem)
        self.tau_meteo = float(tau_meteo)
        self.delta = float(delta)
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

    def fit(
        self, X: np.ndarray, y: np.ndarray
    ) -> "AsymmetricMonotoneHuberRegressor":
        X_arr = _as_2d_float(X)
        y_arr = _as_1d_float(y)
        n_samples, n_features = X_arr.shape

        Z, x_mean, x_scale = _standardize_fit(X_arr)
        m_y, sigma_y = _robust_target_stats(y_arr)
        z_y = (y_arr - m_y) / sigma_y

        l2_vec = np.full(n_features, self.tau_mem, dtype=np.float64)
        for idx in self.meteo_idx:
            if 0 <= idx < n_features:
                l2_vec[idx] = self.tau_meteo

        mono_idx = np.array(
            [i for i in self.monotone_pos_idx if 0 <= i < n_features], dtype=int
        )

        delta = self.delta
        w_pos = 2.0 * self.tau_asym
        w_neg = 2.0 * (1.0 - self.tau_asym)

        def loss_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
            b = params[0]
            w = params[1:]
            r = z_y - (b + Z @ w)
            h_val, h_grad_r = _huber_val_and_grad(r, delta)
            asym_w = np.where(r >= 0.0, w_pos, w_neg)
            loss = float(np.mean(asym_w * h_val) + 0.5 * np.sum(l2_vec * w * w))
            g_pred = -(asym_w * h_grad_r) / n_samples
            g_b = float(np.sum(g_pred))
            g_w = Z.T @ g_pred + l2_vec * w
            return loss, np.concatenate(([g_b], g_w))

        def proj(params: np.ndarray) -> np.ndarray:
            out = params.copy()
            if mono_idx.size > 0:
                out[1 + mono_idx] = np.maximum(out[1 + mono_idx], 0.0)
            return out

        # Upper bound on Lipschitz constant of gradient
        Z_aug = np.column_stack([np.ones(n_samples, dtype=np.float64), Z])
        op_norm_sq = float(np.linalg.norm(Z_aug, ord=2) ** 2)
        L_lip = max(w_pos, w_neg) * op_norm_sq / n_samples + float(
            np.max(l2_vec) if l2_vec.size > 0 else 0.0
        )
        step = 1.0 / max(L_lip, 1e-6)

        x_cur = np.zeros(n_features + 1, dtype=np.float64)
        y_ext = x_cur.copy()
        t_fista = 1.0
        converged = False
        n_iter = 0

        for it in range(1, self.max_iter + 1):
            n_iter = it
            _, grad = loss_and_grad(y_ext)
            x_next = proj(y_ext - step * grad)
            diff = x_next - x_cur

            if np.max(np.abs(diff)) < self.tol and it >= 3:
                x_cur = x_next
                converged = True
                break

            # Adaptive restart if momentum opposes gradient
            if float(np.dot(y_ext - x_next, x_next - x_cur)) > 0.0:
                t_next = 1.0
                y_ext = x_next.copy()
            else:
                t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_fista * t_fista))
                y_ext = x_next + ((t_fista - 1.0) / t_next) * diff

            x_cur = x_next
            t_fista = t_next

        self.x_mean_ = x_mean
        self.x_scale_ = x_scale
        self.m_y_ = m_y
        self.sigma_y_ = sigma_y
        self.intercept_ = float(x_cur[0])
        self.coef_ = x_cur[1:].copy()
        self.n_iter_ = n_iter
        self.converged_ = converged
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = _as_2d_float(X)
        Z = (X_arr - self.x_mean_) / self.x_scale_
        z_hat = self.intercept_ + Z @ self.coef_
        return self.m_y_ + self.sigma_y_ * z_hat


def causal_ewma_cf(
    y: np.ndarray, alpha: float = 0.25, init: float = 0.0
) -> np.ndarray:
    """Strictly causal 1D EWMA filter: pred[t] depends only on y[:t]."""
    arr = _as_1d_float(y)
    n = arr.shape[0]
    preds = np.empty(n, dtype=np.float64)
    state = float(init)
    initialized = False
    for t in range(n):
        preds[t] = state
        val = arr[t]
        if np.isfinite(val):
            if not initialized:
                state = float(val)
                initialized = True
            else:
                state = alpha * float(val) + (1.0 - alpha) * state
    return preds


def causal_kalman_cf(
    y: np.ndarray,
    q: float = 0.01,
    r: float = 0.05,
    robust: bool = False,
    huber_k: float = 2.0,
    init_mean: float = 0.0,
    init_var: float = 1.0,
) -> np.ndarray:
    """Strictly causal 1D random-walk Kalman filter (conventional or Huber-robust)."""
    arr = _as_1d_float(y)
    n = arr.shape[0]
    preds = np.empty(n, dtype=np.float64)
    mean = float(init_mean)
    var = float(init_var)
    q_val = max(float(q), 1e-10)
    r_val = max(float(r), 1e-10)

    for t in range(n):
        m_pred = mean
        p_pred = var + q_val
        preds[t] = m_pred

        val = arr[t]
        if np.isfinite(val):
            innov = float(val) - m_pred
            s_var = p_pred + r_val
            if robust:
                z = abs(innov) / np.sqrt(max(s_var, 1e-12))
                w = 1.0 if z <= huber_k else huber_k / max(z, 1e-12)
                r_eff = r_val / max(w, 1e-6)
                k_gain = p_pred / (p_pred + r_eff)
            else:
                k_gain = p_pred / s_var
            mean = m_pred + k_gain * innov
            var = (1.0 - k_gain) * p_pred
        else:
            mean = m_pred
            var = p_pred

    return preds

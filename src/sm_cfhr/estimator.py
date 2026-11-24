"""Scikit-learn compatible estimators for SM-CFHR / ASHR.

Mathematical formulation
------------------------
Given standardized predictors z_i in R^p and robust MAD-scaled capacity factor target
c_tilde_i = (c_i - m_c) / s_c, SM-CFHR solves the strictly convex box-constrained problem:

    min_{theta_0, theta} (1/n) sum_{i=1}^n rho_delta(c_tilde_i - theta_0 - z_i^T theta)
                         + (alpha_lin / 2) sum_{j in G_lin} theta_j^2
                         + (alpha_meteo / 2) sum_{j in G_meteo} theta_j^2
    subject to theta_j >= 0 for j in M_pos

where rho_delta(r) is the symmetric Huber loss with threshold delta > 0, solved via
analytical-gradient L-BFGS-B in pure NumPy/SciPy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

try:
    from adaptive_structured_huber import AdaptiveStructuredHuberRegressor
except ImportError:
    from src.adaptive_structured_huber import AdaptiveStructuredHuberRegressor


class SMCFHRRegressor(BaseEstimator, RegressorMixin):
    """Public scikit-learn wrapper for fixed-structure SM-CFHR.

    Operates in dimensionless capacity-factor space [0, 1] during ``fit`` and ``predict``.
    Use ``predict_energy(X, cap_mw, horizon_direct=1)`` when physical kWh output is needed.

    Parameters
    ----------
    delta : float, default=1.35
        Huber transition threshold in MAD-standardized residual units.
    alpha_lin : float, default=0.01
        L2 regularization weight for autoregressive and calendar predictors.
    alpha_meteo : float, default=0.10
        L2 regularization weight for numerical weather predictors when ``structured=True``.
    structured : bool, default=True
        Whether to apply differentiated L2 regularization across predictor families.
    use_mono : bool, default=True
        Whether to enforce non-negative box constraints on ``monotone_pos_idx``.
    monotone_pos_idx : tuple[int, ...] | None, default=None
        Column indices constrained to theta_j >= 0 when ``use_mono=True``.
    meteo_idx : tuple[int, ...] | None, default=None
        Column indices receiving ``alpha_meteo`` regularization.
    mode : str, default="espaciotemporal"
        Generalization regime ("espaciotemporal", "intraplanta", or "tabular").
    horizon : int, default=1
        Forecast horizon in days.
    weather_mode : str, default="forecast"
        Weather availability regime ("forecast" or "none").
    max_iter : int, default=400
        Maximum L-BFGS-B iterations.
    tol : float, default=1e-7
        Optimization convergence tolerance.
    random_state : int, default=2026
        Deterministic seed stored for reproducibility.
    """

    def __init__(
        self,
        delta: float = 1.35,
        alpha_lin: float = 0.01,
        alpha_meteo: float = 0.10,
        structured: bool = True,
        use_mono: bool = True,
        monotone_pos_idx: tuple[int, ...] | None = None,
        meteo_idx: tuple[int, ...] | None = None,
        mode: str = "espaciotemporal",
        horizon: int = 1,
        weather_mode: str = "forecast",
        max_iter: int = 400,
        tol: float = 1e-7,
        random_state: int = 2026,
    ) -> None:
        self.delta = delta
        self.alpha_lin = alpha_lin
        self.alpha_meteo = alpha_meteo
        self.structured = structured
        self.use_mono = use_mono
        self.monotone_pos_idx = monotone_pos_idx
        self.meteo_idx = meteo_idx
        self.mode = mode
        self.horizon = horizon
        self.weather_mode = weather_mode
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _build_backend(self) -> AdaptiveStructuredHuberRegressor:
        fixed_cfg = {
            "name": "sm_cfhr_fixed",
            "complexity_rank": 1,
            "n_knots": 0,
            "alpha_lin": float(self.alpha_lin),
            "alpha_meteo": float(self.alpha_meteo),
            "alpha_spl": 0.10,
            "max_interactions": 0,
            "alpha_int": 0.10,
            "structured": bool(self.structured),
            "use_mono": bool(self.use_mono),
            "anchor": "none",
            "lead_damp": 1.0,
        }
        return AdaptiveStructuredHuberRegressor(
            mode=self.mode,
            horizon=int(self.horizon),
            weather_mode=self.weather_mode,
            selection="fixed",
            fixed_config=fixed_cfg,
            delta=float(self.delta),
            random_state=int(self.random_state),
            max_iter=int(self.max_iter),
            tol=float(self.tol),
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> "SMCFHRRegressor":
        """Fit SM-CFHR on capacity-factor target ``y`` (dimensionless)."""
        meta = dict(metadata) if metadata is not None else {}
        if self.monotone_pos_idx is not None and "monotone_pos_idx" not in meta:
            meta["monotone_pos_idx"] = self.monotone_pos_idx
        if self.meteo_idx is not None and "meteo_idx" not in meta:
            meta["meteo_idx"] = self.meteo_idx

        backend = self._build_backend()
        backend.fit(X, y, metadata=meta if meta else None)
        self.estimator_ = backend
        self.n_features_in_ = backend.n_features_in_
        if hasattr(backend, "feature_names_in_"):
            self.feature_names_in_ = backend.feature_names_in_
        self.coef_ = backend.coef_
        self.intercept_ = backend.intercept_
        self.n_iter_ = backend.n_iter_
        self.converged_ = backend.converged_
        return self

    def predict(
        self,
        X: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Predict dimensionless capacity factor clipped to physical bounds [0, 1]."""
        if not hasattr(self, "estimator_"):
            from sklearn.exceptions import NotFittedError

            raise NotFittedError("This SMCFHRRegressor instance is not fitted yet.")
        return np.clip(self.estimator_.predict(X, metadata=metadata), 0.0, 1.0)

    def predict_energy(
        self,
        X: np.ndarray,
        cap_mw: np.ndarray | float,
        horizon_direct: int = 1,
    ) -> np.ndarray:
        """Predict physical energy in kWh (daily or cumulative H-day) from installed capacity."""
        cf = self.predict(X)
        cap = np.asarray(cap_mw, dtype=float)
        if np.any(~np.isfinite(cap)) or np.any(cap <= 0):
            raise ValueError("cap_mw must be strictly positive and finite.")
        return cf * cap * 24_000.0 * float(horizon_direct)


__all__ = ["AdaptiveStructuredHuberRegressor", "SMCFHRRegressor"]

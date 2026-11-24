"""Comprehensive mathematical and software unit tests for AdaptiveStructuredHuberRegressor."""

from __future__ import annotations

import numpy as np

from src.adaptive_structured_huber import AdaptiveStructuredHuberRegressor
from src.modelo_aditivo_huber import build_causal_kalman_feature
from src.prototipos_mecanismos import GlobalHuberEquivariant


def test_gradiente_analitico_vs_diferencias_finitas() -> None:
    rng = np.random.default_rng(11)
    Z = rng.normal(size=(80, 6))
    z_y = 0.5 + 1.2 * Z[:, 0] - 0.7 * Z[:, 1] + 0.3 * rng.normal(size=80)
    pen = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
    params = rng.normal(size=7)

    model = AdaptiveStructuredHuberRegressor(delta=1.35)
    _, grad_ana = model._objective_and_grad_debug(params, Z, z_y, pen)

    eps = 1e-6
    grad_fd = np.zeros_like(params)
    for k in range(len(params)):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[k] += eps
        p_minus[k] -= eps
        f_plus, _ = model._objective_and_grad_debug(p_plus, Z, z_y, pen)
        f_minus, _ = model._objective_and_grad_debug(p_minus, Z, z_y, pen)
        grad_fd[k] = (f_plus - f_minus) / (2.0 * eps)

    np.testing.assert_allclose(grad_ana, grad_fd, rtol=1e-6, atol=1e-6)


def test_condiciones_kkt_y_precision_optimizador() -> None:
    rng = np.random.default_rng(22)
    X = rng.normal(size=(120, 4))
    y = 1.0 - 2.0 * X[:, 0] + 0.9 * X[:, 1] + 0.1 * rng.normal(size=120)

    cfg = {
        "name": "kkt_check", "complexity_rank": 1, "n_knots": 4,
        "alpha_lin": 0.02, "alpha_meteo": 0.1, "alpha_spl": 0.1,
        "max_interactions": 0, "alpha_int": 0.1, "structured": True,
        "use_mono": True, "anchor": "none", "lead_damp": 1.0,
    }
    m = AdaptiveStructuredHuberRegressor(
        mode="intraplanta", selection="fixed", fixed_config=cfg, tol=1e-9,
        store_train_diagnostics=True,
    ).fit(X, y, metadata={"monotone_pos_idx": (0,), "meteo_idx": (2, 3)})

    st = m._state
    params = np.concatenate(([st["intercept"]], st["weights"]))
    _, grad = m._objective_and_grad_debug(params, st["Z_design"], st["z_r"], st["penalties"])

    # KKT for unconstrained coords: |grad_k| ~ 0; for constrained coord j=0 (idx 1): w >= 0, grad >= 0, w*grad ~ 0
    assert st["weights"][0] >= -1e-10
    assert grad[1] >= -1e-5
    assert abs(st["weights"][0] * grad[1]) < 1e-6
    unconstrained_idx = [0] + list(range(2, len(grad)))
    assert np.max(np.abs(grad[unconstrained_idx])) < 1e-4


def test_equivarianza_traslacion_y_escala_con_y_sin_anclaje() -> None:
    rng = np.random.default_rng(33)
    X_tr = rng.normal(size=(140, 4))
    b_tr = rng.uniform(0.2, 0.8, size=140)
    y_tr = b_tr + 0.3 * np.sin(X_tr[:, 0]) + 0.05 * rng.normal(size=140)
    X_te = rng.normal(size=(40, 4))
    b_te = rng.uniform(0.2, 0.8, size=40)

    scale, shift = 10.0, 250.0
    for anchor in ("none", "kalman"):
        cfg = {
            "name": f"eq_{anchor}", "complexity_rank": 1, "n_knots": 5,
            "alpha_lin": 0.05, "alpha_meteo": 0.05, "alpha_spl": 0.1,
            "max_interactions": 0, "alpha_int": 0.1, "structured": False,
            "use_mono": False, "anchor": anchor, "lead_damp": 0.9,
        }
        m1 = AdaptiveStructuredHuberRegressor(selection="fixed", fixed_config=cfg).fit(
            X_tr, y_tr, metadata={"anchor_kalman": b_tr}
        )
        m2 = AdaptiveStructuredHuberRegressor(selection="fixed", fixed_config=cfg).fit(
            X_tr, scale * y_tr + shift, metadata={"anchor_kalman": scale * b_tr + shift}
        )
        p1 = m1.predict(X_te, metadata={"anchor_kalman": b_te, "lead": 3})
        p2 = m2.predict(X_te, metadata={"anchor_kalman": scale * b_te + shift, "lead": 3})
        np.testing.assert_allclose(p2, scale * p1 + shift, rtol=1e-4, atol=1e-4)


def test_causalidad_temporal_exacta() -> None:
    rng = np.random.default_rng(44)
    y = 0.5 + 0.1 * np.sin(np.linspace(0, 8, 90)) + 0.02 * rng.normal(size=90)
    t_split = 50
    y_fut = y.copy()
    y_fut[t_split:] = 500.0

    k1 = build_causal_kalman_feature(y, q=0.01, r=0.05, huber_k=2.0)
    k2 = build_causal_kalman_feature(y_fut, q=0.01, r=0.05, huber_k=2.0)
    np.testing.assert_array_equal(k1[: t_split + 1], k2[: t_split + 1])


def test_reproducibilidad_selector_interno() -> None:
    rng = np.random.default_rng(55)
    X = rng.normal(size=(180, 5))
    y = 1.0 + np.sin(X[:, 0]) + 0.5 * X[:, 1] + 0.1 * rng.normal(size=180)
    meta = {"dates": np.arange(180), "groups": np.tile([1, 2, 3], 60)}

    m1 = AdaptiveStructuredHuberRegressor(mode="espaciotemporal", random_state=42).fit(X, y, metadata=meta)
    m2 = AdaptiveStructuredHuberRegressor(mode="espaciotemporal", random_state=42).fit(X, y, metadata=meta)
    assert m1.selected_config_["name"] == m2.selected_config_["name"]
    np.testing.assert_array_equal(m1.predict(X), m2.predict(X))


def test_ausencia_fuga_test_en_transformaciones() -> None:
    rng = np.random.default_rng(66)
    X_tr = rng.normal(size=(120, 4))
    y_tr = 2.0 * X_tr[:, 0] + 0.1 * rng.normal(size=120)
    X_te1 = rng.normal(size=(30, 4))
    X_te2 = X_te1.copy()
    X_te2[0, :] = 1e6  # Extreme test row must not alter predictions on rows 1..29

    m = AdaptiveStructuredHuberRegressor(mode="tabular", random_state=7).fit(X_tr, y_tr)
    p1 = m.predict(X_te1)
    p2 = m.predict(X_te2)
    np.testing.assert_array_equal(p1[1:], p2[1:])


def test_plantas_no_vistas_y_modo_auto() -> None:
    rng = np.random.default_rng(77)
    X_tr = rng.normal(size=(150, 5))
    y_tr = 0.5 + 0.3 * X_tr[:, 0] + 0.05 * rng.normal(size=150)
    meta_tr = {
        "dates": np.tile(np.arange(50), 3),
        "groups": np.repeat([1, 2, 3], 50),
        "plant_ids": np.repeat(["P1", "P2", "P3"], 50),
    }
    m = AdaptiveStructuredHuberRegressor(mode="auto", horizon=1).fit(X_tr, y_tr, metadata=meta_tr)
    assert m.resolved_mode_ == "espaciotemporal"

    X_unseen = rng.normal(size=(20, 5))
    meta_unseen = {"plant_ids": np.array(["UNSEEN_PLANT"] * 20), "groups": np.array([99] * 20)}
    pred = m.predict(X_unseen, metadata=meta_unseen)
    assert pred.shape == (20,)
    assert np.all(np.isfinite(pred))


def test_sin_meteorologia_y_horizonte_mayor_uno() -> None:
    rng = np.random.default_rng(88)
    X_tr = rng.normal(size=(180, 4))
    b_tr = rng.uniform(0.3, 0.7, size=180)
    y_tr = b_tr + 0.15 * X_tr[:, 0] + 0.03 * rng.normal(size=180)
    meta_tr = {"dates": np.arange(180), "anchor_kalman": b_tr, "anchor_mm30": b_tr}

    m = AdaptiveStructuredHuberRegressor(
        mode="intraplanta", horizon=14, weather_mode="none"
    ).fit(X_tr, y_tr, metadata=meta_tr)
    assert m.converged_

    X_te = rng.normal(size=(10, 4))
    b_te = np.full(10, 0.5)
    p_h1 = m.predict(X_te, metadata={"anchor_kalman": b_te, "anchor_mm30": b_te, "lead": 1})
    p_h14 = m.predict(X_te, metadata={"anchor_kalman": b_te, "anchor_mm30": b_te, "lead": 14})
    assert np.all(np.isfinite(p_h1)) and np.all(np.isfinite(p_h14))


def test_muestras_pequenas_y_matrices_casi_singulares() -> None:
    rng = np.random.default_rng(99)
    X = rng.normal(size=(24, 3))
    X_sing = np.column_stack([X, X[:, 0] + 1e-10 * rng.normal(size=24), np.ones(24)])
    y = 0.4 + 0.2 * X[:, 0] + 0.05 * rng.normal(size=24)

    m = AdaptiveStructuredHuberRegressor(mode="intraplanta", random_state=1).fit(X_sing, y)
    pred = m.predict(X_sing)
    assert np.all(np.isfinite(pred))


def test_compatibilidad_variantes_historicas_modo_fijo() -> None:
    rng = np.random.default_rng(123)
    X = rng.normal(size=(120, 5))
    y = 0.5 + 0.8 * X[:, 0] - 0.4 * X[:, 1] + 0.1 * rng.normal(size=120)
    X_te = rng.normal(size=(25, 5))

    ref = GlobalHuberEquivariant(delta=1.35, alpha=0.05).fit(X, y)
    cfg_a = {
        "name": "hist_A", "complexity_rank": 1, "n_knots": 0,
        "alpha_lin": 0.05, "alpha_meteo": 0.05, "alpha_spl": 0.05,
        "max_interactions": 0, "alpha_int": 0.1, "structured": False,
        "use_mono": False, "anchor": "none", "lead_damp": 1.0,
    }
    ashr = AdaptiveStructuredHuberRegressor(selection="fixed", fixed_config=cfg_a).fit(X, y)
    np.testing.assert_allclose(ashr.predict(X_te), ref.predict(X_te), rtol=1e-6, atol=1e-6)


def test_modo_tabular_general() -> None:
    rng = np.random.default_rng(456)
    X_tr = rng.uniform(-2.0, 2.0, size=(240, 4))
    y_tr = np.sin(2.0 * X_tr[:, 0]) + 0.7 * (X_tr[:, 1] ** 2) + 0.1 * rng.normal(size=240)
    X_te = rng.uniform(-2.0, 2.0, size=(80, 4))
    y_te = np.sin(2.0 * X_te[:, 0]) + 0.7 * (X_te[:, 1] ** 2)

    m = AdaptiveStructuredHuberRegressor(mode="auto", random_state=42).fit(X_tr, y_tr)
    assert m.resolved_mode_ == "tabular"
    assert m.selected_config_["n_knots"] > 0
    mae = float(np.mean(np.abs(m.predict(X_te) - y_te)))
    assert mae < 0.35

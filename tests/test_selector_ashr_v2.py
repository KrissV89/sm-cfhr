"""Unit tests for the revised ASHR v2 selector (internal_cv_v2)."""

from __future__ import annotations

import numpy as np

from src.adaptive_structured_huber import (
    AdaptiveStructuredHuberRegressor,
    _candidate_grid_v2,
)


def test_orden_complejidad_verdadero_v2() -> None:
    cands = _candidate_grid_v2("intraplanta", horizon=14)
    ranks = {c["name"]: c["complexity_rank"] for c in cands}
    assert ranks["mh_lin_struct_mono"] == 1
    assert ranks["mh_lin_struct_mono"] < ranks["mh_lin_mm30_damp"] < ranks["mh_lin_kalman_damp"]
    assert ranks["mh_lin_kalman_damp"] < ranks["mh_spline_struct_mono"]


def test_congelamiento_anclaje_bloque_validacion_y_causalidad() -> None:
    dates = np.array(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"], dtype="datetime64[D]")
    meta_va = {
        "dates": dates,
        "plant_ids": np.array(["P1", "P1", "P1", "P1"]),
        "anchor_mm30": np.array([0.50, 0.99, 0.10, 0.80]),
        "anchor_kalman": np.array([0.45, 0.90, 0.20, 0.70]),
    }
    ashr = AdaptiveStructuredHuberRegressor(mode="intraplanta", horizon=14, selection="internal_cv_v2")
    prep = ashr._prepare_val_metadata_v2(meta_va)
    np.testing.assert_array_equal(prep["lead"], np.array([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(prep["anchor_mm30"], np.full(4, 0.50))
    np.testing.assert_array_equal(prep["anchor_kalman"], np.full(4, 0.45))


def test_ausencia_fuga_en_selector_v2() -> None:
    rng = np.random.default_rng(17)
    X_tr = rng.normal(size=(160, 5))
    y_tr = 0.5 + 0.2 * X_tr[:, 0] + 0.05 * rng.normal(size=160)
    meta_tr = {
        "dates": np.arange(160),
        "anchor_mm30": np.full(160, 0.5),
        "anchor_kalman": np.full(160, 0.5),
    }
    m = AdaptiveStructuredHuberRegressor(
        mode="intraplanta", horizon=14, selection="internal_cv_v2", random_state=2026
    ).fit(X_tr, y_tr, metadata=meta_tr)

    X_te1 = rng.normal(size=(20, 5))
    X_te2 = X_te1.copy()
    X_te2[0, :] = 1e5
    np.testing.assert_array_equal(m.predict(X_te1)[1:], m.predict(X_te2)[1:])


def test_macro_plant_mae_y_compatibilidad_v1() -> None:
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(180, 4))
    y = 0.4 + 0.3 * X[:, 0] + 0.05 * rng.normal(size=180)
    meta = {
        "dates": np.tile(np.arange(60), 3),
        "groups": np.repeat([1, 2, 3], 60),
        "plant_ids": np.repeat(["A", "B", "C"], 60),
    }
    m_v1_a = AdaptiveStructuredHuberRegressor(mode="espaciotemporal", selection="internal_cv", random_state=42).fit(X, y, metadata=meta)
    m_v1_b = AdaptiveStructuredHuberRegressor(mode="espaciotemporal", selection="internal_cv", random_state=42).fit(X, y, metadata=meta)
    assert m_v1_a.selected_config_["name"] == m_v1_b.selected_config_["name"]
    np.testing.assert_array_equal(m_v1_a.predict(X), m_v1_b.predict(X))

    m_v2 = AdaptiveStructuredHuberRegressor(mode="espaciotemporal", selection="internal_cv_v2", stability_weight=0.25, random_state=42).fit(X, y, metadata=meta)
    assert m_v2.converged_
    assert all("score" in r for r in m_v2.cv_results_)

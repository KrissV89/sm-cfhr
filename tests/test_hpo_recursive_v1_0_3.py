"""Mandatory 12-test suite (A-L) for v1.0.3 exact recursive HPO rollout and compatibility."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

from scripts.evaluacion import produccion_multihorizonte as mh
from scripts.evaluacion import produccion_multihorizonte_clima as mhc
from src.ashr_oof_pipeline.config import (
    build_frozen_ashr,
    find_project_root,
    load_frozen_config,
    sha256_file,
)
from src.ashr_oof_pipeline.oof_runner import (
    artifact_cache_valid,
    build_raw_plants_from_panel,
    fit_ashr_on_train,
    rollout_recursive_from_origin,
)
from src.ashr_oof_pipeline.optimizacion_hiperparametros import (
    HPO_PROTOCOL_VERSION,
    _fit_and_predict_fold,
    _rollout_recursive_val,
    audit_inner_splits,
    build_causal_inner_splits,
    compute_hpo_protocol_hash,
    optimize_hyperparams_tpe,
)


def _make_synthetic_panel(
    n_days: int = 120,
    n_plants: int = 4,
    weather_on: bool = True,
    seed: int = 2026,
) -> tuple[pd.DataFrame, dict[str, dict], dict]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    base_rows = []
    pron_rows = []
    for i in range(n_plants):
        code = f"P{i+1}"
        grupo = i // 2
        cap = 20.0 + 10.0 * i
        # Smooth seasonal + noise generation
        t = np.arange(n_days, dtype=float)
        cf = np.clip(0.24 + 0.04 * np.sin(2.0 * np.pi * t / 30.0) + rng.normal(0, 0.02, size=n_days), 0.05, 0.45)
        kwh = cf * cap * 24_000.0
        for dt, y_val in zip(dates, kwh):
            base_rows.append({
                "code": code,
                "date": dt,
                "kwh": float(y_val),
                "horas_xm": 12.0,
                "cap_MW": cap,
                "lat": 5.0 + 0.5 * grupo,
                "lon": -74.0,
                "grupo": grupo,
            })
        if weather_on:
            # Build causal weather forecasts emitted at origin for leads 1..28
            for em_idx in range(n_days):
                em_dt = dates[em_idx]
                for lead in range(1, 29):
                    obj_dt = em_dt + pd.Timedelta(days=lead)
                    pron_rows.append({
                        "code": code,
                        "fecha_emision": em_dt,
                        "fecha_objetivo": obj_dt,
                        "lead_dias": lead,
                        "rad_pron": float(5.5 + rng.normal(0, 0.3)),
                        "temp_pron": 28.0,
                        "nube_pron": 30.0,
                        "humedad_pron": 70.0,
                        "viento_pron": 3.0,
                    })

    base_df = pd.DataFrame(base_rows)
    if weather_on:
        pron_df = pd.DataFrame(pron_rows)
        panel, raw_plants, wx_store = mhc.load_generation_weather_panel(
            Path("unused"), Path("unused"), base_data=base_df, pron_data=pron_df
        )
    else:
        panel, raw_plants = mh.load_generation_panel(Path("unused"), base_data=base_df)
        wx_store = {}
    return panel, raw_plants, wx_store


# A. Internal rollout and external rollout produce identical trajectory for same estimator, state, origin
def test_a_internal_and_external_rollout_exact_equivalence() -> None:
    panel, raw_plants, wx_store = _make_synthetic_panel(n_days=110, n_plants=2, weather_on=True)
    origin = pd.Timestamp("2025-03-20")
    train = panel[panel["date"] <= origin].copy()
    val_df = panel[(panel["date"] > origin) & (panel["date"] <= origin + pd.Timedelta(days=14))].copy()

    fitted = fit_ashr_on_train(
        train, task="intraplanta", horizon=14, weather_on=True, use_hpo=False, mode_eval="recursive"
    )
    ext_rows = rollout_recursive_from_origin(
        fitted=fitted,
        pred_infos=[raw_plants["P1"], raw_plants["P2"]],
        origin=origin,
        horizon_days=14,
        weather_on=True,
        weather_store=wx_store,
    )
    ext_df = pd.DataFrame(ext_rows).sort_values(["code", "lead_dias"]).reset_index(drop=True)

    y_int, p_int, c_int = _rollout_recursive_val(
        fitted["est"],
        fitted["mu"],
        val_df,
        horizon_days=14,
        weather_on=True,
        target_col="y",
        raw_plants=raw_plants,
        weather_store=wx_store,
        history_df=train,
    )
    np.testing.assert_allclose(p_int, ext_df["prediccion_kwh"].to_numpy(), atol=1e-12)
    np.testing.assert_allclose(y_int, ext_df["y_kwh"].to_numpy(), atol=1e-12)


# B. Sliding windows of 7 and 30 days update exactly (not exponential approximation)
def test_b_exact_sliding_windows_7_and_30_days() -> None:
    panel, raw_plants, _ = _make_synthetic_panel(n_days=80, n_plants=1, weather_on=False)
    info = raw_plants["P1"]
    origin = pd.Timestamp("2025-02-20")
    st = mh.PlantRecursiveState(info, origin, horizon=7)

    # Record initial history slice at origin
    n_hist = (origin - info["min_date"]).days + 1
    buf = list(info["kwh_valid"][:n_hist])
    preds = [100_000.0, 120_000.0, 90_000.0, 110_000.0]
    for step, p_val in enumerate(preds):
        target_dt = origin + pd.Timedelta(days=step + 1)
        feat = st.next_features(target_dt)
        expected_m7 = float(np.nanmean(buf[-7:]))
        expected_m30 = float(np.nanmean(buf[-30:]))
        assert feat["media7_kwh"] == pytest.approx(expected_m7, abs=1e-9)
        assert feat["media30_kwh"] == pytest.approx(expected_m30, abs=1e-9)
        st.step_update(p_val)
        buf.append(p_val)


# C & D. Causal dependence on <= origin + own predictions; invariance to future real perturbations
def test_cd_causal_origin_and_future_perturbation_invariance() -> None:
    panel, raw_plants, wx_store = _make_synthetic_panel(n_days=100, n_plants=2, weather_on=True)
    origin = pd.Timestamp("2025-03-10")
    train = panel[panel["date"] <= origin].copy()
    val_clean = panel[(panel["date"] > origin) & (panel["date"] <= origin + pd.Timedelta(days=7))].copy()

    val_corrupt = val_clean.copy()
    val_corrupt["y"] = 99_999_999.0
    val_corrupt["kwh_D"] = 99_999_999.0
    val_corrupt["media7_kwh"] = 99_999_999.0
    val_corrupt["media30_kwh"] = 99_999_999.0

    # Copy raw_plants with corrupted future y > origin
    raw_corrupt = {}
    for k, v in raw_plants.items():
        vc = dict(v)
        arr = vc["kwh_valid"].copy()
        n_hist = (origin - vc["min_date"]).days + 1
        arr[n_hist:] = 99_999_999.0
        vc["kwh_valid"] = arr
        raw_corrupt[k] = vc

    _, p_clean, _, _ = _fit_and_predict_fold(
        train, val_clean, task="intraplanta", horizon_days=7, weather_on=True,
        delta=1.35, alpha_lin=0.01, alpha_meteo=0.10, mode_eval="recursive",
        raw_plants=raw_plants, weather_store=wx_store,
    )
    _, p_corrupt, _, _ = _fit_and_predict_fold(
        train, val_corrupt, task="intraplanta", horizon_days=7, weather_on=True,
        delta=1.35, alpha_lin=0.01, alpha_meteo=0.10, mode_eval="recursive",
        raw_plants=raw_corrupt, weather_store=wx_store,
    )
    np.testing.assert_allclose(p_clean, p_corrupt, atol=1e-12)


# E & F. All horizons H = 7, 14, 21, 28 work for both weather_on=False and weather_on=True
@pytest.mark.parametrize("horizon", [7, 14, 21, 28])
@pytest.mark.parametrize("weather_on", [False, True])
def test_ef_horizons_and_weather_regimes(horizon: int, weather_on: bool) -> None:
    panel, raw_plants, wx_store = _make_synthetic_panel(n_days=130, n_plants=2, weather_on=weather_on)
    origin = pd.Timestamp("2025-03-20")
    train = panel[panel["date"] <= origin].copy()
    val_df = panel[(panel["date"] > origin) & (panel["date"] <= origin + pd.Timedelta(days=horizon))].copy()

    y_va, p_va, _, _ = _fit_and_predict_fold(
        train, val_df, task="intraplanta", horizon_days=horizon, weather_on=weather_on,
        delta=1.35, alpha_lin=0.01, alpha_meteo=0.10 if weather_on else 0.01,
        mode_eval="recursive", raw_plants=raw_plants, weather_store=wx_store if weather_on else None,
    )
    assert len(p_va) == len(y_va) and len(p_va) > 0
    assert np.all(np.isfinite(p_va))


# G & H. ET preserves strict spatial exclusion; IP preserves temporal order and embargo
def test_gh_et_spatial_disjoint_and_ip_temporal_order() -> None:
    panel, _, _ = _make_synthetic_panel(n_days=110, n_plants=4, weather_on=False)
    et_splits = build_causal_inner_splits(
        panel, mode_eval="recursive", task="espaciotemporal", horizon_days=7, n_splits=2, embargo_days=7
    )
    assert len(et_splits) >= 1
    et_audit = audit_inner_splits(panel, et_splits, mode_eval="recursive", task="espaciotemporal", horizon_days=7, strict=True)
    assert bool(et_audit["spatial_disjoint"].all())

    p1 = panel[panel["code"] == "P1"].reset_index(drop=True)
    ip_splits = build_causal_inner_splits(
        p1, mode_eval="recursive", task="intraplanta", horizon_days=7, n_splits=2, embargo_days=7
    )
    assert len(ip_splits) >= 1
    ip_audit = audit_inner_splits(p1, ip_splits, mode_eval="recursive", task="intraplanta", horizon_days=7, strict=True)
    assert (pd.to_datetime(ip_audit["train_max_target_end"]) <= pd.to_datetime(ip_audit["val_min_date"])).all()


# I. Direct cumulative scenario preserves H-day cumulative target and units
def test_i_direct_cumulative_target_and_units() -> None:
    panel, _, _ = _make_synthetic_panel(n_days=110, n_plants=2, weather_on=False)
    horizon = 7
    panel_dir = panel.copy()
    panel_dir["t_origin"] = panel_dir["date"] - pd.Timedelta(days=1)
    panel_dir["t_end_target"] = panel_dir["t_origin"] + pd.Timedelta(days=horizon)
    panel_dir["y_H"] = panel_dir["y"] * float(horizon)

    train = panel_dir.iloc[:140].copy()
    val = panel_dir.iloc[140:160].copy()
    y_va, p_va, _, _ = _fit_and_predict_fold(
        train, val, task="espaciotemporal", horizon_days=horizon, weather_on=False,
        delta=1.35, alpha_lin=0.01, alpha_meteo=0.01,
        horizon_direct=horizon, target_col="y_H", mode_eval="direct",
    )
    # Predictions must be in cumulative H-day kWh scale (around 7 * daily kWh)
    assert np.median(p_va) > 3.0 * np.median(panel["y"])
    np.testing.assert_allclose(y_va, val["y_H"].to_numpy())


# J. SM-CFHR-Fixed numerical equivalence and untouched core estimator SHA-256
def test_j_fixed_equivalence_and_core_sha256_unchanged() -> None:
    root = find_project_root()
    cfg = load_frozen_config(root)
    actual_sha = sha256_file(root / "src" / "adaptive_structured_huber.py")
    assert actual_sha == cfg["sha256_src_adaptive_structured_huber"]

    rng = np.random.default_rng(2026)
    X = rng.normal(size=(60, 17))
    X[:, 4] = np.clip(0.24 + 0.04 * X[:, 4], 0.05, 0.45)
    X[:, 7] = np.clip(5.5 + 0.5 * X[:, 7], 1.0, 8.0)
    y = np.clip(0.6 * X[:, 4] + 0.02 * X[:, 7], 0.05, 0.90)
    meta = {"cf30_idx": 4, "monotone_pos_idx": (4, 7), "meteo_idx": tuple(range(7, 17))}

    est1 = build_frozen_ashr(mode="espaciotemporal", horizon=1, weather_mode="forecast", root=root)
    est2 = build_frozen_ashr(mode="espaciotemporal", horizon=1, weather_mode="forecast", root=root)
    est1.fit(X[:45], y[:45], metadata=meta)
    est2.fit(X[:45], y[:45], metadata=meta)
    np.testing.assert_allclose(est1.predict(X[45:]), est2.predict(X[45:]), atol=1e-12)


# K & L. SQLite resume under v1.0.3 and explicit rejection of v1.0.2 legacy studies / manifests
def test_kl_sqlite_resume_and_legacy_v1_0_2_rejection(tmp_path: Path) -> None:
    panel, raw_plants, _ = _make_synthetic_panel(n_days=110, n_plants=4, weather_on=False)
    db_url = f"sqlite:///{(tmp_path / 'v103.db').as_posix()}"
    h_hash = compute_hpo_protocol_hash()

    _, t1 = optimize_hyperparams_tpe(
        panel, task="espaciotemporal", horizon_days=7, weather_on=False, mode_eval="recursive",
        n_trials=2, seed=2026, storage_url=db_url, study_name="s_v103",
        proto_hash=h_hash, raw_plants=raw_plants, embargo_days=7,
    )
    assert len(t1) == 2

    _, t2 = optimize_hyperparams_tpe(
        panel, task="espaciotemporal", horizon_days=7, weather_on=False, mode_eval="recursive",
        n_trials=4, seed=2026, storage_url=db_url, study_name="s_v103",
        proto_hash=h_hash, raw_plants=raw_plants, embargo_days=7,
    )
    assert len(t2) == 4 and bool(t2["trial_valid"].all())

    # L1: Legacy v1.0.2 SQLite study (has COMPLETE trial, no hpo_protocol_version) must be rejected
    legacy_db = f"sqlite:///{(tmp_path / 'legacy_v102.db').as_posix()}"
    st_old = optuna.create_study(direction="minimize", storage=legacy_db, study_name="s_old")
    tr = st_old.ask({"delta": optuna.distributions.FloatDistribution(0.5, 3.0)})
    st_old.tell(tr, 1000.0)
    with pytest.raises(RuntimeError, match="incompatible|legado"):
        optimize_hyperparams_tpe(
            panel, task="espaciotemporal", horizon_days=7, weather_on=False, mode_eval="recursive",
            n_trials=3, seed=2026, storage_url=legacy_db, study_name="s_old",
            proto_hash=h_hash, raw_plants=raw_plants, embargo_days=7,
        )

    # L2: Legacy v1.0.2 manifest (no hpo_protocol_version) valid for Fixed only if src_hash matches, invalid for HPO
    m_dir = tmp_path / "cache_v102"
    m_dir.mkdir()
    src_hash = load_frozen_config()["sha256_src_adaptive_structured_huber"]
    (m_dir / "manifest_ashr_oof.json").write_text(
        json.dumps({
            "estatus": "OOF_DEFINITIVO_COMPLETO",
            "sha256_src_adaptive_structured_huber": src_hash,
            "variantes_completadas": ["fixed", "hpo"],
        }),
        encoding="utf-8",
    )
    assert artifact_cache_valid(m_dir, src_hash, variante="fixed") is True
    assert artifact_cache_valid(m_dir, src_hash, variante="hpo", expected_hpo_proto_hash=h_hash) is False
    assert artifact_cache_valid(m_dir, src_hash, variante="ambos", expected_hpo_proto_hash=h_hash) is False

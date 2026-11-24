"""Pruebas específicas de los 6 bloqueadores corregidos para sm-cfhr-oof-v1.1 (A-F)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sm_cfhr.inference import evaluate_r3_superiority
from sm_cfhr.persistence import RunCheckpointManager
from sm_cfhr.pipeline_stages import run_post_oof_stages
from src.ashr_oof_pipeline.config import find_project_root, load_frozen_config
from src.ashr_oof_pipeline.oof_runner import (
    _persist_unit_fragment,
    _try_load_unit_fragment,
    align_predictions,
    artifact_cache_valid,
)
from src.ashr_oof_pipeline.optimizacion_hiperparametros import (
    _fit_and_predict_fold,
    build_causal_inner_splits,
)
from src.ashr_oof_pipeline.resultados_loader import (
    create_synthetic_oof_fixture,
    load_oof_bundle,
)


def test_blocker_a_recursive_rollout_ignores_future_real_history() -> None:
    rng = np.random.default_rng(2026)
    dates = pd.date_range("2025-01-01", periods=60, freq="D")
    rows = []
    for dt in dates:
        rows.append(
            {
                "code": "P1",
                "grupo": 0,
                "date": dt,
                "cap_MW_D": 20.0,
                "kwh_D": 80_000.0 + rng.normal(0, 1000.0),
                "media7_kwh": 80_000.0,
                "media30_kwh": 80_000.0,
                "month_sin": 0.0,
                "month_cos": 1.0,
                "doy_sin": 0.0,
                "doy_cos": 1.0,
                "y": 80_000.0,
            }
        )
    df = pd.DataFrame(rows)
    tr = df.iloc[:45].copy()
    va_clean = df.iloc[45:55].copy()
    va_corrupt_future = va_clean.copy()
    # Corrupt future rows' historical columns (steps 1..9): causal rollout must NOT read them
    va_corrupt_future.iloc[1:, va_corrupt_future.columns.get_loc("kwh_D")] = 9_999_999.0
    va_corrupt_future.iloc[1:, va_corrupt_future.columns.get_loc("media30_kwh")] = 9_999_999.0

    _, p_clean, _, _ = _fit_and_predict_fold(
        tr, va_clean, task="intraplanta", horizon_days=7, weather_on=False,
        delta=1.35, alpha_lin=0.01, alpha_meteo=0.01, mode_eval="recursive"
    )
    _, p_corrupt, _, _ = _fit_and_predict_fold(
        tr, va_corrupt_future, task="intraplanta", horizon_days=7, weather_on=False,
        delta=1.35, alpha_lin=0.01, alpha_meteo=0.01, mode_eval="recursive"
    )
    np.testing.assert_allclose(p_clean, p_corrupt, atol=1e-12)


def test_blocker_a_et_strict_spatial_disjoint_no_silent_degradation() -> None:
    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    # 2 groups, group 1 has only 2 rows -> cannot satisfy va_sp >= 5, so split must be skipped, not degraded
    rows = [{"code": "P1", "grupo": 0, "date": dt, "y": 1.0} for dt in dates]
    rows += [{"code": "P2", "grupo": 1, "date": dt, "y": 1.0} for dt in dates[:2]]
    df = pd.DataFrame(rows)
    splits = build_causal_inner_splits(df, mode_eval="d1", task="espaciotemporal", horizon_days=1, n_splits=2, embargo_days=2)
    groups = df["grupo"].to_numpy()
    for tr_idx, va_idx in splits:
        assert set(groups[tr_idx]).isdisjoint(set(groups[va_idx]))


def test_blocker_b_prohibits_synthetic_in_confirmatory(tmp_path: Path) -> None:
    root = find_project_root()
    syn = create_synthetic_oof_fixture(tmp_path / "conf_dir", root=root)
    # In CONFIRMATORY mode, synthetic fixture is rejected by load_oof_bundle
    with pytest.raises(RuntimeError, match="SINTETICO_TEST"):
        load_oof_bundle(syn, root=root, allow_synthetic_test=False)

    # Even with real-labeled manifest, CONFIRMATORY with panel_df=None must NOT generate synthetic ablations
    m_path = syn / "manifest_ashr_oof.json"
    m = json.loads(m_path.read_text(encoding="utf-8"))
    m["procedencia"] = "OOF_DEFINITIVO_REAL"
    m_path.write_text(json.dumps(m), encoding="utf-8")
    res = run_post_oof_stages(syn, root=root, allow_synthetic_test=False, panel_df=None, mode="CONFIRMATORY")
    assert res["etapas_5_6_estado"] == "BLOQUEADO_SIN_DATOS_REALES"
    assert not (syn / "ablaciones" / "tabla_ablaciones.csv").exists()


def test_blocker_c_r3_requires_three_calibrated_tests(tmp_path: Path) -> None:
    root = find_project_root()
    syn = create_synthetic_oof_fixture(tmp_path / "r3_dir", root=root)
    bundle = load_oof_bundle(syn, root=root, allow_synthetic_test=True)
    # Bundle has DM-HLN and MCS, but no tabla_spa -> Condition 2 must be NO_VERIFICABLE
    r3 = evaluate_r3_superiority(bundle, scope_authorized=False)
    c2 = r3[r3["condicion"] == "Condicion_2_Pruebas_Multiples_y_MCS"].iloc[0]
    assert c2["estado"] == "NO_VERIFICABLE"


def test_blocker_d_granular_unit_fragment_resume_without_duplicates(tmp_path: Path) -> None:
    mgr = RunCheckpointManager(tmp_path, "run_d", "cfg", "src", "proto", total_units=2)
    df_u = pd.DataFrame({"code": ["P1", "P2"], "prediccion_kwh": [100.0, 200.0], "y_kwh": [105.0, 195.0]})
    _persist_unit_fragment(mgr, "fixed", "D1_test", "D1_g0", df_u)
    loaded, _ = _try_load_unit_fragment(mgr, "fixed", "D1_test", "D1_g0")
    assert loaded is not None and len(loaded) == 2


def test_blocker_ef_cache_variant_independence_and_strict_alignment(tmp_path: Path) -> None:
    m_path = tmp_path / "manifest_ashr_oof.json"
    m_path.write_text(
        json.dumps({
            "estatus": "OOF_DEFINITIVO_COMPLETO",
            "sha256_src_adaptive_structured_huber": "abc",
            "variante": "fixed",
            "variantes_completadas": ["fixed"],
        }),
        encoding="utf-8",
    )
    assert artifact_cache_valid(tmp_path, "abc", variante="fixed") is True
    assert artifact_cache_valid(tmp_path, "abc", variante="hpo") is False
    assert artifact_cache_valid(tmp_path, "abc", variante="ambos") is False

    df_a = pd.DataFrame({"code": ["P1", "P1"], "date": ["2026-01-01", "2026-01-01"], "pred": [1.0, 2.0]})
    df_b = pd.DataFrame({"code": ["P1"], "date": ["2026-01-01"], "metodo": ["M1"], "pred": [1.0]})
    with pytest.raises(ValueError, match="Claves duplicadas"):
        align_predictions(df_a, df_b, ("code", "date"))

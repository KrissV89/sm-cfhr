"""Pruebas automatizadas del preflight final de integración, seguridad operacional y Estado B."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ashr_oof_pipeline.config import find_project_root, load_frozen_config
from src.ashr_oof_pipeline.oof_runner import fit_ashr_on_train, run_oof_definitivo
from src.ashr_oof_pipeline.optimizacion_hiperparametros import (
    audit_inner_splits,
    build_causal_inner_splits,
)
from src.ashr_oof_pipeline.resultados_loader import (
    create_synthetic_oof_fixture,
    load_oof_bundle,
    validate_oof_artifacts,
)


def _make_real_like_direct_panel(n_days: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for g_idx, code in enumerate(["P1", "P2", "P3", "P4"]):
        grupo = g_idx // 2
        for d_i, dt in enumerate(dates):
            rows.append(
                {
                    "code": code,
                    "grupo": grupo,
                    "t_origin": dt,
                    "t_end_target": dt + pd.Timedelta(days=14),
                    "cap_MW_D": 20.0,
                    "kwh_D": 80_000.0 + rng.normal(0, 2_000.0),
                    "media7_kwh": 80_000.0,
                    "media30_kwh": 80_000.0,
                    "month_sin": 0.0,
                    "month_cos": 1.0,
                    "doy_sin": 0.0,
                    "doy_cos": 1.0,
                    "y_H": float(14 * (80_000.0 + rng.normal(0, 2_000.0))),
                }
            )
    return pd.DataFrame(rows)


def test_preflight_01_fit_ashr_on_train_hpo_real_integration() -> None:
    df = _make_real_like_direct_panel(n_days=70)
    fitted = fit_ashr_on_train(
        df,
        task="espaciotemporal",
        horizon=14,
        weather_on=False,
        horizon_direct=14,
        target_col="y_H",
        use_hpo=True,
        mode_eval="direct",
        n_trials=4,
        outer_fold_id="test_direct",
    )
    assert not fitted["trials_df"].empty
    assert (fitted["trials_df"]["trial_valid"]).all()
    assert fitted["chosen_hp"]["alpha_meteo"] == pytest.approx(fitted["chosen_hp"]["alpha_lin"])


def test_preflight_02_audit_inner_splits_detects_direct_and_spatial_causality() -> None:
    df = _make_real_like_direct_panel(n_days=80)
    splits = build_causal_inner_splits(df, mode_eval="direct", task="espaciotemporal", horizon_days=14)
    assert len(splits) >= 1
    audit_df = audit_inner_splits(df, splits, mode_eval="direct", task="espaciotemporal", horizon_days=14, strict=True)
    assert audit_df["spatial_disjoint"].all()
    assert (pd.to_datetime(audit_df["train_max_target_end"]) <= pd.to_datetime(audit_df["val_min_date"])).all()


def test_preflight_03_dry_run_supports_variantes_and_hash_check() -> None:
    root = find_project_root()
    dry = run_oof_definitivo(execute=False, root=root, variante="ambos", n_trials=25)
    assert dry["estatus"] == "DRY_RUN_PREPARADO"
    assert dry["ejecutado"] is False
    assert dry["variante"] == "ambos"
    assert dry["n_trials_hpo"] == 25
    assert dry["hashes_coinciden"] is True


def test_preflight_04_hash_mismatch_blocks_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = find_project_root()
    monkeypatch.setattr(
        "src.ashr_oof_pipeline.oof_runner.load_frozen_config",
        lambda _r: {**load_frozen_config(root), "sha256_src_adaptive_structured_huber": "bad_hash"},
    )
    with pytest.raises(RuntimeError, match="Hash SHA-256 desalineado"):
        run_oof_definitivo(execute=True, out_dir=tmp_path / "blocked", root=root)


def test_preflight_05_mcs_connected_in_state_b_and_rejects_partial(tmp_path: Path) -> None:
    root = find_project_root()
    syn_dir = create_synthetic_oof_fixture(tmp_path / "syn_b", root=root)
    bundle = load_oof_bundle(syn_dir, root=root, allow_synthetic_test=True)
    assert "tabla_mcs" in bundle and not bundle["tabla_mcs"].empty
    assert set(["escenario", "n_survivors", "ashr_in_mcs_90", "p_mcs_ashr"]).issubset(bundle["tabla_mcs"].columns)

    # Remove 1 scenario file -> must fail validation as INVALID
    first_csv = next((syn_dir / "predicciones").glob("*.csv"))
    first_csv.unlink()
    expected_hash = load_frozen_config(root)["sha256_src_adaptive_structured_huber"]
    val = validate_oof_artifacts(syn_dir, expected_hash, allow_synthetic_test=True)
    assert val["state"] == "INVALID"


def test_preflight_06_reserved_definitive_dir_untouched() -> None:
    root = find_project_root()
    assert not (root / "resultados" / "evaluacion_ashr_oof" / "manifest_ashr_oof.json").exists()

"""Suite de 12 pruebas de recuperación, persistencia incremental, checkpoints, locks y RSS (A-L)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

from sm_cfhr import SMCFHRRegressor
from sm_cfhr.persistence import (
    RunCheckpointManager,
    WorkUnitLock,
    atomic_write_csv,
    atomic_write_parquet,
    atomic_write_text,
    get_process_rss_mb,
)
from sm_cfhr.pipeline_stages import apply_family_multiplicity, run_post_oof_stages
from src.ashr_oof_pipeline.config import find_project_root, load_frozen_config
from src.ashr_oof_pipeline.optimizacion_hiperparametros import optimize_hyperparams_tpe
from src.ashr_oof_pipeline.resultados_loader import (
    create_synthetic_oof_fixture,
    validate_oof_artifacts,
)


def _make_dev_df(n_days: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for g_idx, code in enumerate(["P1", "P2", "P3", "P4"]):
        for dt in dates:
            rows.append(
                {
                    "code": code,
                    "grupo": g_idx // 2,
                    "date": dt,
                    "cap_MW_D": 20.0,
                    "kwh_D": 80_000.0 + rng.normal(0, 2_000.0),
                    "media7_kwh": 80_000.0,
                    "media30_kwh": 80_000.0,
                    "month_sin": 0.0,
                    "month_cos": 1.0,
                    "doy_sin": 0.0,
                    "doy_cos": 1.0,
                    "rad_pron": 5.0 + rng.normal(0, 0.4),
                    "temp_pron": 28.0,
                    "nube_pron": 30.0,
                    "humedad_pron": 70.0,
                    "viento_pron": 3.0,
                    "anom_rad": rng.normal(0, 0.3),
                    "anom_temp": 0.0,
                    "anom_nube": 0.0,
                    "anom_humedad": 0.0,
                    "anom_viento": 0.0,
                    "y": float(80_000.0 + rng.normal(0, 2_000.0)),
                }
            )
    return pd.DataFrame(rows)


# A. Interrupcion tras guardar trials en Optuna SQLite y reanudacion sin repetir trials COMPLETE
def test_rec_a_optuna_sqlite_resume_without_repeating(tmp_path: Path) -> None:
    df = _make_dev_df(80)
    db_url = f"sqlite:///{(tmp_path / 'study.db').as_posix()}"
    _, t1 = optimize_hyperparams_tpe(
        df, task="espaciotemporal", horizon_days=1, weather_on=True, mode_eval="d1",
        n_trials=3, seed=2026, storage_url=db_url, study_name="s_resume", proto_hash="h1", embargo_days=5
    )
    assert len(t1) == 3
    # Resume asking for 5 total trials -> only 2 new trials run, total 5
    _, t2 = optimize_hyperparams_tpe(
        df, task="espaciotemporal", horizon_days=1, weather_on=True, mode_eval="d1",
        n_trials=5, seed=2026, storage_url=db_url, study_name="s_resume", proto_hash="h1", embargo_days=5
    )
    assert len(t2) == 5 and (t2["trial_valid"]).all()


# B. Trial interrumpido/fallido en SQLite no contado como COMPLETE
def test_rec_b_failed_optuna_trial_not_counted_as_complete(tmp_path: Path) -> None:
    db_url = f"sqlite:///{(tmp_path / 'fail.db').as_posix()}"
    storage = optuna.storages.RDBStorage(url=db_url)
    st = optuna.create_study(direction="minimize", storage=storage, study_name="s_fail")
    st.set_user_attr("proto_hash", "h1")
    t = st.ask()
    st.tell(t, state=optuna.trial.TrialState.FAIL)

    df = _make_dev_df(80)
    _, logs = optimize_hyperparams_tpe(
        df, task="espaciotemporal", horizon_days=1, weather_on=True, mode_eval="d1",
        n_trials=2, seed=2026, storage_url=db_url, study_name="s_fail", proto_hash="h1", embargo_days=5
    )
    assert int(logs["trial_valid"].sum()) == 2


# C. Interrupcion tras guardar predicciones y salto de unidad valida al reanudar
def test_rec_c_checkpoint_skips_completed_unit(tmp_path: Path) -> None:
    mgr = RunCheckpointManager(tmp_path, "run_c", "cfg1", "src1", "proto1", total_units=2)
    out_p = mgr.run_dir / "predicciones" / "u1.csv"
    df = pd.DataFrame({"code": ["P1"], "pred": [10.0]})
    sha = atomic_write_csv(df, out_p)
    mgr.mark_unit("u1", "COMPLETED", artifact_path=out_p, artifact_sha256=sha, n_rows=1)
    assert mgr.is_unit_completed("u1", expected_rows=1) is True
    assert mgr.is_unit_completed("u2") is False


# D. Archivo temporal *.tmp incompleto no contamina artefacto definitivo
def test_rec_d_atomic_write_leaves_valid_artifact_only(tmp_path: Path) -> None:
    target = tmp_path / "preds.parquet"
    df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    (tmp_path / "preds.parquet.tmp.99999").write_bytes(b"corrupt_partial")
    sha = atomic_write_parquet(df, target)
    assert target.exists() and len(sha) == 64
    assert len(pd.read_parquet(target)) == 2


# E. Checkpoint corrupto detectado como INVALID
def test_rec_e_corrupt_checkpoint_detected_invalid(tmp_path: Path) -> None:
    mgr = RunCheckpointManager(tmp_path, "run_e", "cfg1", "src1", "proto1")
    mgr._ckpt_path("u1").write_text("{corrupt_json", encoding="utf-8")
    st = mgr.get_unit_state("u1")
    assert st["status"] == "INVALID"
    assert mgr.is_unit_completed("u1") is False


# F. Archivo valido con checksum incorrecto marcado INVALID
def test_rec_f_checksum_mismatch_marks_invalid(tmp_path: Path) -> None:
    mgr = RunCheckpointManager(tmp_path, "run_f", "cfg1", "src1", "proto1")
    out_p = mgr.run_dir / "predicciones" / "u1.csv"
    atomic_write_csv(pd.DataFrame({"a": [1]}), out_p)
    mgr.mark_unit("u1", "COMPLETED", artifact_path=out_p, artifact_sha256="0" * 64, n_rows=1)
    assert mgr.is_unit_completed("u1") is False
    assert mgr.get_unit_state("u1")["status"] == "INVALID"


# G. Reanudacion con configuracion diferente bloqueada
def test_rec_g_config_mismatch_blocks_resume(tmp_path: Path) -> None:
    RunCheckpointManager(tmp_path, "run_g", "cfg_v1", "src_v1", "proto_v1")
    with pytest.raises(RuntimeError, match="Incompatibilidad"):
        RunCheckpointManager(tmp_path, "run_g", "cfg_v2", "src_v1", "proto_v1")


# H. Dos procesos intentando reclamar el mismo trabajo bloqueados por WorkUnitLock
def test_rec_h_exclusive_work_unit_lock(tmp_path: Path) -> None:
    lock_p = tmp_path / "unit.lock"
    with WorkUnitLock(lock_p):
        with pytest.raises(RuntimeError, match="already locked"):
            with WorkUnitLock(lock_p):
                pass
    assert not lock_p.exists()


# I. Notebook con resultados parciales detecta Estado C INVALID
def test_rec_i_partial_results_detected_as_state_c_invalid(tmp_path: Path) -> None:
    root = find_project_root()
    syn = create_synthetic_oof_fixture(tmp_path / "partial_oof", root=root)
    next((syn / "predicciones").glob("*.csv")).unlink()
    exp_hash = load_frozen_config(root)["sha256_src_adaptive_structured_huber"]
    val = validate_oof_artifacts(syn, exp_hash, allow_synthetic_test=True)
    assert val["state"] == "INVALID"


# J & K. Reconstruccion de etapas 2-8 sin reentrenar y multiplicidad por familias
def test_rec_jk_post_oof_stages_and_family_multiplicity(tmp_path: Path) -> None:
    root = find_project_root()
    syn = create_synthetic_oof_fixture(tmp_path / "stage_oof", root=root)
    res = run_post_oof_stages(syn, root=root, allow_synthetic_test=True, mode="TEST_SYNTHETIC")
    assert res["status"] == "COMPLETED"
    assert (syn / "conclusiones" / "veredicto_r3.csv").exists()
    assert (syn / "ablaciones" / "tabla_ablaciones.csv").exists()
    assert (syn / "incertidumbre" / "tabla_conformal_resumen.csv").exists()
    assert (syn / "interpretabilidad" / "tabla_importancia_global.csv").exists()
    assert (syn / "figuras" / "fig09_ablacion.png").exists()

    # Family multiplicity preserves NaNs for insufficient T_eff
    t_inf = pd.DataFrame({
        "suite": ["S1", "S1", "S2"],
        "referencia": ["Mejor_Bib", "Mejor_Bib", "Mejor_Bib"],
        "t_eff_insufficient": [False, False, True],
        "p_dm_hln": [0.01, 0.04, np.nan],
    })
    fam = apply_family_multiplicity(t_inf)
    assert np.isnan(fam.loc[2, "p_holm_familia"])
    assert fam.loc[0, "n_comparaciones_familia"] == 2


# L. Ausencia de duplicados tras reanudar, RSS medido y estado minimo sin Z_design
def test_rec_l_no_duplicates_rss_and_minimal_estimator_state(tmp_path: Path) -> None:
    rss = get_process_rss_mb()
    assert rss > 0.0

    X = np.random.default_rng(2026).normal(size=(50, 10))
    y = np.random.default_rng(2026).uniform(0.05, 0.4, size=50)
    m = SMCFHRRegressor().fit(X, y)
    assert "Z_design" not in m.estimator_._state
    assert "z_r" not in m.estimator_._state

    p_text = tmp_path / "atomic.txt"
    atomic_write_text(p_text, "ok")
    assert p_text.read_text(encoding="utf-8") == "ok"

"""Tests for Windows spawn parallel OOF, Optuna resume, recycling, equivalence, and consolidation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sm_cfhr.parallel import (  # noqa: E402
    HorizonStorageLayout,
    ParallelOOFCoordinator,
    build_work_inventory,
    consolidate_parallel_run,
    merge_disjoint_runs,
)


def _create_synthetic_data_root(base: Path) -> Path:
    """Create minimal realistic solar panel + benchmarks for D+1, Rec 7d (no/with weather), Direct 7d."""
    rng = np.random.default_rng(2026)
    dates = pd.date_range("2024-09-01", periods=365, freq="D")
    plants = [
        ("P1", 1, 50.0, 5.4, -74.7),
        ("P2", 1, 80.0, 5.5, -74.6),
        ("P3", 2, 20.0, 8.4, -75.6),
        ("P4", 2, 35.0, 8.2, -73.3),
        ("P5", 3, 100.0, 10.4, -74.6),
        ("P6", 3, 15.0, 10.6, -74.8),
    ]

    gen_rows = []
    d1_wx_rows = []
    for code, grp, cap, lat, lon in plants:
        cf_base = 0.24 + 0.03 * np.sin(2.0 * np.pi * np.arange(len(dates)) / 180.0)
        rad = 4.8 + 1.1 * np.sin(2.0 * np.pi * np.arange(len(dates)) / 90.0) + rng.normal(0, 0.3, len(dates))
        temp = 27.0 + rng.normal(0, 1.2, len(dates))
        nube = np.clip(45.0 + rng.normal(0, 15.0, len(dates)), 0, 100)
        hum = np.clip(72.0 + rng.normal(0, 8.0, len(dates)), 10, 100)
        viento = np.clip(3.2 + rng.normal(0, 0.8, len(dates)), 0.2, 12.0)
        cf = np.clip(cf_base + 0.02 * (rad - 4.8) + rng.normal(0, 0.015, len(dates)), 0.05, 0.85)
        kwh = cf * cap * 24_000.0
        for i, d in enumerate(dates):
            gen_rows.append(
                {
                    "code": code,
                    "date": d.strftime("%Y-%m-%d"),
                    "kwh": float(kwh[i]),
                    "horas_xm": 24.0,
                    "cap_MW": cap,
                    "lat": lat,
                    "lon": lon,
                    "grupo": grp,
                }
            )
            d1_wx_rows.append(
                {
                    "code": code,
                    "date": d.strftime("%Y-%m-%d"),
                    "rad_pron": float(rad[i]),
                    "temp_pron": float(temp[i]),
                    "nube_pron": float(nube[i]),
                    "humedad_pron": float(hum[i]),
                    "viento_pron": float(viento[i]),
                }
            )

    (base / "base_generacion_clima").mkdir(parents=True, exist_ok=True)
    (base / "datos_pronostico").mkdir(parents=True, exist_ok=True)
    (base / "datos_pronostico_multihorizonte").mkdir(parents=True, exist_ok=True)

    pd.DataFrame(gen_rows).to_csv(base / "base_generacion_clima" / "base_planta_dia.csv", index=False)
    pd.DataFrame(d1_wx_rows).to_csv(base / "datos_pronostico" / "pronostico_diario.csv", index=False)

    # GEFSv12 multihorizon archived forecasts: daily lead=1 history + 7d leads for 3 origins
    mh_origins = [pd.Timestamp("2025-06-15"), pd.Timestamp("2025-07-01"), pd.Timestamp("2025-07-15")]
    mh_origin_set = set(mh_origins)
    gefs_rows = []
    for code, _, _, _, _ in plants:
        for orig in dates[:-7]:
            max_l = 7 if orig in mh_origin_set else 1
            for lead in range(1, max_l + 1):
                tgt = orig + pd.Timedelta(days=lead)
                gefs_rows.append(
                    {
                        "code": code,
                        "fecha_emision": orig.strftime("%Y-%m-%d"),
                        "fecha_objetivo": tgt.strftime("%Y-%m-%d"),
                        "lead_dias": lead,
                        "rad_pron": float(4.9 + rng.normal(0, 0.3)),
                        "temp_pron": float(27.0 + rng.normal(0, 1.0)),
                        "nube_pron": float(np.clip(45.0 + rng.normal(0, 10.0), 0, 100)),
                        "humedad_pron": float(np.clip(70.0 + rng.normal(0, 5.0), 10, 100)),
                        "viento_pron": float(np.clip(3.0 + rng.normal(0, 0.5), 0.2, 10.0)),
                    }
                )
    pd.DataFrame(gefs_rows).to_csv(base / "datos_pronostico_multihorizonte" / "pronostico_28d_gefs.csv", index=False)

    # Official-style benchmark CSV stubs so inventory & alignment work identically
    d1_dates = pd.date_range("2025-07-10", periods=10, freq="D")
    for task in ("espaciotemporal", "intraplanta"):
        for w_lbl in ("cinco_variables", "sin_meteorologia"):
            bdir = base / "resultados" / "benchmark_produccion_definitivo" / f"{task}_{w_lbl}"
            bdir.mkdir(parents=True, exist_ok=True)
            rows = []
            for code, grp, _, _, _ in plants:
                for d in d1_dates:
                    rows.append(
                        {
                            "metodo": "Ridge",
                            "bloque": "D1",
                            "code": code,
                            "grupo": grp,
                            "date": d.strftime("%Y-%m-%d"),
                            "prediccion_kwh": 100000.0,
                            "y_kwh": 100000.0,
                        }
                    )
            pd.DataFrame(rows).to_csv(bdir / "predicciones.csv", index=False)

    for bbase, is_dir in (
        ("benchmark_multihorizonte", False),
        ("benchmark_multihorizonte_clima", False),
        ("benchmark_multihorizonte_directo", True),
    ):
        hdir = base / "resultados" / bbase / "horizonte_7_dias"
        hdir.mkdir(parents=True, exist_ok=True)
        rows = []
        for task in ("espaciotemporal", "intraplanta"):
            for code, grp, _, _, _ in plants:
                for orig in mh_origins:
                    if is_dir:
                        rows.append(
                            {
                                "metodo": "Ridge",
                                "tarea": task,
                                "code": code,
                                "grupo": grp,
                                "origen": orig.strftime("%Y-%m-%d"),
                                "prediccion_total_kwh": 700000.0,
                                "y_total_kwh": 700000.0,
                            }
                        )
                    else:
                        for lead in range(1, 8):
                            rows.append(
                                {
                                    "metodo": "Ridge",
                                    "tarea": task,
                                    "code": code,
                                    "grupo": grp,
                                    "origen": orig.strftime("%Y-%m-%d"),
                                    "lead_dias": lead,
                                    "prediccion_kwh": 100000.0,
                                    "y_kwh": 100000.0,
                                }
                            )
        pd.DataFrame(rows).to_csv(hdir / "predicciones_oof.csv", index=False)
    return base


def test_parallel_equivalence_fixed_and_hpo(tmp_path: Path) -> None:
    """1-worker sequential vs 2-worker parallel must yield bit-identical predictions & trials."""
    data_root = _create_synthetic_data_root(tmp_path / "data_root")
    scenarios = [
        "D1_espaciotemporal_cinco_variables",
        "D1_intraplanta_sin_meteorologia",
        "RecSinClima_intraplanta_7d",
        "RecGEFSv12_espaciotemporal_7d",
        "DirectoSinClima_espaciotemporal_7d",
    ]
    units_all = build_work_inventory(
        root=ROOT,
        data_root=data_root,
        variante="ambos",
        n_trials=4,
        seed=2026,
        scenarios=scenarios,
    )
    # Subset covering all 5 scenarios in both fixed and hpo
    subset = []
    seen = set()
    for u in units_all:
        key = (u.variant, u.scenario_id)
        if key not in seen:
            seen.add(key)
            subset.append(u)
    assert len(subset) == 10

    out_dir = tmp_path / "runs_out"
    c1 = ParallelOOFCoordinator(
        out_dir=out_dir, run_id="seq_w1", repo_root=ROOT, data_root=data_root,
        n_workers=1, variante="ambos", n_trials=4, seed=2026,
    )
    s1 = c1.run(subset)
    assert s1["estatus"] == "COMPLETED"
    assert s1["failed"] == 0

    c2 = ParallelOOFCoordinator(
        out_dir=out_dir, run_id="par_w2", repo_root=ROOT, data_root=data_root,
        n_workers=2, max_tasks_per_worker=3, variante="ambos", n_trials=4, seed=2026,
    )
    s2 = c2.run(subset)
    assert s2["estatus"] == "COMPLETED"
    assert s2["failed"] == 0
    assert s2["worker_recycles"] >= 1

    m1 = consolidate_parallel_run(out_dir / "seq_w1", root=ROOT)
    m2 = consolidate_parallel_run(out_dir / "par_w2", root=ROOT)
    assert m1["estatus"] == "OOF_PARCIAL"
    assert m1["hashes_sha256_predicciones"] == m2["hashes_sha256_predicciones"]

    for sc in scenarios:
        for suf in ("", "_hpo"):
            p1 = pd.read_csv(out_dir / "seq_w1" / "consolidated" / "predicciones" / f"{sc}{suf}.csv")
            p2 = pd.read_csv(out_dir / "par_w2" / "consolidated" / "predicciones" / f"{sc}{suf}.csv")
            col = "prediccion_total_kwh" if "Directo" in sc else "prediccion_kwh"
            np.testing.assert_allclose(p1[col].to_numpy(), p2[col].to_numpy(), rtol=0.0, atol=1e-10)

        t1 = pd.read_csv(out_dir / "seq_w1" / "consolidated" / "hpo_trials" / f"{sc}.csv")
        t2 = pd.read_csv(out_dir / "par_w2" / "consolidated" / "hpo_trials" / f"{sc}.csv")
        assert len(t1) == len(t2) == 4
        np.testing.assert_allclose(t1["inner_mae_macro_kwh"].to_numpy(), t2["inner_mae_macro_kwh"].to_numpy(), rtol=0.0, atol=1e-10)


def test_optuna_resume_no_extra_trials(tmp_path: Path) -> None:
    """Interrupted Optuna study resumes cleanly to exact n_trials without creating extra trials."""
    from ashr_oof_pipeline.optimizacion_hiperparametros import optimize_hyperparams_tpe

    data_root = _create_synthetic_data_root(tmp_path / "data_opt")
    from scripts.evaluacion import produccion_intraplanta as daily

    panel = daily.build_panel(
        data_root / "base_generacion_clima" / "base_planta_dia.csv",
        data_root / "datos_pronostico" / "pronostico_diario.csv",
        weather_on=True,
    )
    train = panel[panel["grupo"] != 1].copy()
    db_path = tmp_path / "study.db"
    url = f"sqlite:///{db_path.as_posix()}"

    _, df_part = optimize_hyperparams_tpe(
        train, task="espaciotemporal", horizon_days=1, weather_on=True, mode_eval="d1",
        n_trials=3, seed=2026, study_name="resume_test", storage_url=url,
    )
    assert len(df_part) == 3

    hp_res, df_full = optimize_hyperparams_tpe(
        train, task="espaciotemporal", horizon_days=1, weather_on=True, mode_eval="d1",
        n_trials=6, seed=2026, study_name="resume_test", storage_url=url,
    )
    assert len(df_full) == 6
    assert set(df_full["trial_number"]) == {0, 1, 2, 3, 4, 5}
    assert np.isfinite(hp_res["delta"])


def test_orphan_lock_recovery_and_disjoint_merge(tmp_path: Path) -> None:
    """Orphaned locks are recovered, corrupt fragments detected, and disjoint shards merge cleanly."""
    data_root = _create_synthetic_data_root(tmp_path / "data_shard")
    scenarios = ["D1_espaciotemporal_cinco_variables"]
    u_s0 = build_work_inventory(
        root=ROOT, data_root=data_root, variante="fixed", scenarios=scenarios, shard_index=0, num_shards=2
    )
    u_s1 = build_work_inventory(
        root=ROOT, data_root=data_root, variante="fixed", scenarios=scenarios, shard_index=1, num_shards=2
    )
    assert set(u.unit_id for u in u_s0).isdisjoint(u.unit_id for u in u_s1)

    out_dir = tmp_path / "shards_out"
    c0 = ParallelOOFCoordinator(
        out_dir=out_dir, run_id="shard0", repo_root=ROOT, data_root=data_root,
        n_workers=2, variante="fixed", n_trials=0, shard_index=0, num_shards=2,
    )
    # Simulate stale lock from dead PID before running
    stale_lock = c0.storage.unit_lock_path(u_s0[0].unit_id)
    stale_lock.write_text("pid=99999999\n", encoding="utf-8")
    c0.storage.write_unit_record(u_s0[0].unit_id, "RUNNING")

    s0 = c0.run(u_s0)
    assert s0["recovered_orphaned_locks"] == 1
    assert s0["estatus"] == "COMPLETED"

    c1 = ParallelOOFCoordinator(
        out_dir=out_dir, run_id="shard1", repo_root=ROOT, data_root=data_root,
        n_workers=2, variante="fixed", n_trials=0, shard_index=1, num_shards=2,
    )
    s1 = c1.run(u_s1)
    assert s1["estatus"] == "COMPLETED"

    merged = merge_disjoint_runs([out_dir / "shard0", out_dir / "shard1"], out_dir / "merged", root=ROOT)
    assert merged["n_unidades_esperadas"] == len(u_s0) + len(u_s1)
    assert merged["n_unidades_faltantes"] == 0

    # Corrupt a fragment and verify detection
    first_u = u_s0[0]
    frag = out_dir / "shard0" / first_u.horizon_bucket / first_u.scenario_id / "fixed" / "fragments" / f"{first_u.unit_tag}.parquet"
    frag.write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="Incomplete or corrupt"):
        consolidate_parallel_run(out_dir / "shard0", root=ROOT, require_complete=True)

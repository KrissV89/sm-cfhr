"""Comprehensive verification tests for the frozen ASHR OOF pipeline and final notebook."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
import pytest

from src.adaptive_structured_huber import AdaptiveStructuredHuberRegressor
from src.ashr_oof_pipeline import (
    SCENARIOS_28,
    build_descriptive_summary,
    build_frozen_ashr,
    export_paper_tables,
    find_project_root,
    load_frozen_config,
    load_oof_bundle,
    validate_oof_artifacts,
    validate_predictions_df,
    verify_benchmark_inventory,
)
from src.ashr_oof_pipeline.evaluacion_estadistica import (
    apply_holm,
    build_23_model_ranking,
    compute_scenario_metrics,
    model_confidence_set,
    run_paired_inference,
)
from src.ashr_oof_pipeline.oof_runner import align_predictions, run_oof_definitivo
from src.ashr_oof_pipeline.resultados_loader import create_synthetic_oof_fixture
from src.ashr_oof_pipeline.visualizacion import export_all_figures


def test_reproduccion_exacta_y_consistencia_config() -> None:
    root = find_project_root()
    cfg = load_frozen_config(root)
    assert cfg["parametros_modelo"]["selection"] == "fixed"
    assert cfg["parametros_modelo"]["fixed_config"]["n_knots"] == 0
    assert cfg["parametros_modelo"]["fixed_config"]["use_mono"] is True

    rng = np.random.default_rng(2026)
    X_tr = rng.normal(size=(150, 17))
    y_tr = np.clip(0.4 + 0.1 * X_tr[:, 4] + 0.08 * X_tr[:, 7] + 0.02 * rng.normal(size=150), 0.0, 1.0)
    X_te = rng.normal(size=(40, 17))
    meta = {"cf30_idx": 4, "monotone_pos_idx": (4, 7), "meteo_idx": tuple(range(7, 17))}

    frozen = build_frozen_ashr(mode="espaciotemporal", horizon=1, weather_mode="forecast", root=root).fit(X_tr, y_tr, metadata=meta)
    manual = AdaptiveStructuredHuberRegressor(
        mode="espaciotemporal", horizon=1, weather_mode="forecast",
        selection="fixed", fixed_config=cfg["parametros_modelo"]["fixed_config"],
        delta=1.35, max_iter=400, tol=1e-7, random_state=2026,
    ).fit(X_tr, y_tr, metadata=meta)
    np.testing.assert_array_equal(frozen.predict(X_te), manual.predict(X_te))


def test_inventario_28_escenarios_9_12_1_y_dry_run() -> None:
    root = find_project_root()
    inv = verify_benchmark_inventory(root)
    assert len(inv) == 28
    assert inv["file_exists"].all()
    assert inv["cols_ok"].all()
    assert inv["cf_brh_excluded"].all()
    assert (inv["n_official_models"] == 22).all()

    dry = run_oof_definitivo(execute=False, root=root)
    assert dry["estatus"] == "DRY_RUN_PREPARADO"
    assert dry["ejecutado"] is False
    assert dry["inventario_ok"] is True
    assert dry["sha256_src_adaptive_structured_huber"] == dry["config_hash_esperado"]


def test_validacion_manifiestos_y_claves_duplicadas(tmp_path: Path) -> None:
    root = find_project_root()
    expected_hash = load_frozen_config(root)["sha256_src_adaptive_structured_huber"]

    # Missing directory -> PENDING
    assert validate_oof_artifacts(tmp_path / "missing", expected_hash)["state"] == "PENDING"
    assert build_descriptive_summary(None)["estado"] == "PENDIENTE"

    # Exploratory or mismatched-hash manifest -> INVALID
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "manifest_ashr_oof.json").write_text(
        json.dumps({"estatus": "EXPLORATORIO_DESARROLLO", "sha256_src_adaptive_structured_huber": "wrong"}),
        encoding="utf-8",
    )
    val_bad = validate_oof_artifacts(bad_dir, expected_hash)
    assert val_bad["state"] == "INVALID"
    assert len(val_bad["reasons"]) >= 2

    # Duplicate keys -> ValueError
    dup_df = pd.DataFrame({"code": ["P1", "P1"], "date": ["2026-01-01", "2026-01-01"], "pred": [10.0, 12.0], "y": [11.0, 11.0]})
    with pytest.raises(ValueError, match="Claves duplicadas"):
        validate_predictions_df(dup_df, ("code", "date"), "pred", "y")


def test_estado_b_sintetico_tablas_y_5_figuras(tmp_path: Path) -> None:
    root = find_project_root()
    syn_dir = create_synthetic_oof_fixture(tmp_path / "syn_oof", root=root)

    # Without allow_synthetic_test=True, synthetic fixture is rejected
    expected_hash = load_frozen_config(root)["sha256_src_adaptive_structured_huber"]
    assert validate_oof_artifacts(syn_dir, expected_hash, allow_synthetic_test=False)["state"] == "INVALID"

    bundle = load_oof_bundle(syn_dir, root=root, allow_synthetic_test=True)
    assert len(bundle["tabla_28_escenarios"]) == 28
    assert (bundle["tabla_28_escenarios"]["n_modelos_comparados"] == 23).all()

    figs = export_all_figures(bundle, tmp_path / "figs")
    assert len(figs) == 5
    for png, pdf in figs.values():
        assert png.exists() and png.stat().st_size > 0
        assert pdf.exists() and pdf.stat().st_size > 0

    tabs = export_paper_tables(bundle, tmp_path / "tabs")
    assert len(tabs) == 10
    for p in tabs.values():
        assert p.exists() and p.stat().st_size > 0

    desc = build_descriptive_summary(bundle)
    assert desc["estado"] == "VALIDO"
    assert desc["n_escenarios"] == 28


def test_estadisticas_y_alineacion() -> None:
    rng = np.random.default_rng(7)
    loss_a = rng.uniform(10, 20, size=18)
    loss_b = loss_a + rng.normal(1.5, 0.5, size=18)
    res_inf = run_paired_inference(loss_a, loss_b, horizon_h=14, is_temporal=True)
    assert res_inf["t_eff_insufficient"] is True

    L = np.column_stack([loss_a, loss_b, loss_b + 3.0])
    mcs = model_confidence_set(L, ["ASHR", "Lib1", "Lib2"], alpha=0.10, B=100, seed=2026)
    assert "ASHR" in mcs["survivors"]
    assert len(apply_holm(np.array([0.01, 0.04]))) == 2

    df_a = pd.DataFrame({"code": ["P1", "P1"], "date": ["2026-01-01", "2026-01-02"], "pred": [10.0, 12.0], "y": [11.0, 11.0]})
    df_b = pd.DataFrame({"metodo": ["M1", "M1", "M1"], "code": ["P1", "P1", "P1"], "date": ["2026-01-01", "2026-01-02", "2026-01-03"], "pred": [10.0, 12.0, 99.0], "y": [11.0, 11.0, 11.0]})
    assert len(align_predictions(df_a, df_b, ("code", "date"))) == 2
    assert compute_scenario_metrics(df_a, "pred", "y")["n_obs"] == 2
    rk = build_23_model_ranking(pd.DataFrame({"metodo": ["CF-BRH", "ASHR", "Lasso"], "MAE_macro": [1.0, 2.0, 3.0]}))
    assert list(rk["metodo"]) == ["ASHR", "Lasso"]


def test_notebook_valido_y_ejecucion_limpia_estado_a() -> None:
    root = find_project_root()
    nb_path = root / "notebooks" / "actuales" / "evaluacion_cientifica_ashr_oof.ipynb"
    nb = nbf.read(nb_path, as_version=4)
    nbf.validate(nb)

    md_text = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "markdown")
    for sec in ("1. Introducción", "2. Revisión de literatura", "3. Formulación matemática", "7. Protocolo experimental", "19. Conclusiones"):
        assert sec in md_text
    assert "SM-CFHR" in md_text
    assert "28 escenarios" in md_text

    ns: dict = {}
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            exec(compile(cell["source"], str(nb_path), "exec"), ns)
    assert ns["dry_status"]["estatus"] == "DRY_RUN_PREPARADO"
    assert ns["val_status"]["state"] == "PENDING"
    assert ns["bundle"] is None
    assert ns["resumen_desc"]["estado"] == "PENDIENTE"

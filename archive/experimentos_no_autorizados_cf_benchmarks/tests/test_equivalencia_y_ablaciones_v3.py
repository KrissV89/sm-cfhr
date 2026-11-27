"""Automated tests for v3.0.0 representation equivalence and controlled ablations."""

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

from src.sm_cfhr.representation_benchmark import (
    MODEL_NAMES_V3,
    extract_kwh_features,
    fit_model_representation,
    predict_model_representation,
)


def _synthetic_solar_panel(n_days: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    rows = []
    for code, grupo, cap in [("P_SMALL", 1, 10.0), ("P_LARGE", 2, 200.0)]:
        for t in range(n_days):
            cf30 = 0.25 + 0.03 * np.sin(t / 10.0)
            cf7 = cf30 + rng.normal(0, 0.01)
            cf1 = cf7 + rng.normal(0, 0.01)
            rad = 4.5 + 1.0 * np.sin(t / 7.0) + rng.normal(0, 0.2)
            cf_true = np.clip(0.55 * cf30 + 0.03 * rad + rng.normal(0, 0.015), 0.02, 0.95)
            cap_t = cap * (1.2 if (code == "P_LARGE" and t >= n_days // 2) else 1.0)
            denom = cap_t * 24000.0
            rows.append({
                "code": code,
                "grupo": grupo,
                "date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=t),
                "cap_MW_D": cap_t,
                "media30_kwh": cf30 * denom,
                "media7_kwh": cf7 * denom,
                "kwh_D": cf1 * denom,
                "month_sin": np.sin(2 * np.pi * ((t // 30) % 12 + 1) / 12.0),
                "month_cos": np.cos(2 * np.pi * ((t // 30) % 12 + 1) / 12.0),
                "doy_sin": np.sin(2 * np.pi * (t + 1) / 365.25),
                "doy_cos": np.cos(2 * np.pi * (t + 1) / 365.25),
                "rad_pron": rad,
                "temp_pron": 26.0,
                "nube_pron": 45.0,
                "humedad_pron": 70.0,
                "viento_pron": 2.5,
                "anom_rad": rad - 4.5,
                "anom_temp": 0.0,
                "anom_nube": 0.0,
                "anom_humedad": 0.0,
                "anom_viento": 0.0,
                "y": cf_true * denom,
            })
    return pd.DataFrame(rows)


def test_cf_vs_kwh_representation_wrappers_and_clipping() -> None:
    """All 7 models fit cleanly in both CF and kWh_bruto and respect [0, 24000*cap_MW_D]."""
    df = _synthetic_solar_panel(80)
    tr = df[df["grupo"] == 1].copy()
    te = df[df["grupo"] == 2].copy()
    upper = te["cap_MW_D"].to_numpy(float) * 24000.0

    X_kwh, denom = extract_kwh_features(tr, weather_on=True)
    assert X_kwh.shape == (len(tr), 18)
    assert np.all(denom > 0)

    for m_name in MODEL_NAMES_V3:
        for rep in ("CF", "kWh_bruto"):
            ft = fit_model_representation(
                m_name, rep, tr, "espaciotemporal", 1, True, horizon_direct=1, target_col="y", seed=2026
            )
            preds = predict_model_representation(ft, te, lead=1)
            assert preds.shape == (len(te),)
            assert np.all(np.isfinite(preds))
            assert np.all(preds >= -1e-9)
            assert np.all(preds <= upper + 1e-6)


def test_cf_normalization_reduces_scale_transfer_error_for_ridge() -> None:
    """On a 20x scale shift between train (10 MW) and test (200 MW), Ridge-CF beats Ridge-kWh."""
    df = _synthetic_solar_panel(90)
    tr = df[df["grupo"] == 1].copy()
    te = df[df["grupo"] == 2].copy()
    y_te = te["y"].to_numpy(float)

    f_cf = fit_model_representation("Ridge", "CF", tr, "espaciotemporal", 1, True, 1, "y", seed=2026)
    f_kw = fit_model_representation("Ridge", "kWh_bruto", tr, "espaciotemporal", 1, True, 1, "y", seed=2026)
    mae_cf = float(np.mean(np.abs(predict_model_representation(f_cf, te) - y_te)))
    mae_kw = float(np.mean(np.abs(predict_model_representation(f_kw, te) - y_te)))
    assert mae_cf < mae_kw


def test_frozen_oof_manifest_untouched_v3() -> None:
    """Confirm frozen OOF manifest remains untouched."""
    p = ROOT / "resultados" / "evaluacion_ashr_oof" / "manifest_ashr_oof.json"
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))
        assert m["estatus"] == "OOF_DEFINITIVO_COMPLETO"


def test_v3_equivalence_artifacts_and_schemas() -> None:
    """Validate Condition A/B equivalence summary CSVs, paired inference, and manifest."""
    eq_dir = ROOT / "resultados" / "experimento_equivalencia_v3"
    if not eq_dir.exists():
        pytest.skip("experimento_equivalencia_v3 not present")
    req_files = [
        "resumen_equivalencia_condicion_A.csv",
        "resumen_equivalencia_condicion_B.csv",
        "inferencia_pareada_smcfhr_vs_cf_baselines.csv",
        "resultados_por_planta_equivalencia.csv",
        "MANIFIESTO_EQUIVALENCIA_V3.csv",
    ]
    for fn in req_files:
        fp = eq_dir / fn
        assert fp.exists(), f"Missing {fn}"
        df = pd.read_csv(fp)
        assert len(df) > 0, f"Empty {fn}"

    df_a = pd.read_csv(eq_dir / "resumen_equivalencia_condicion_A.csv")
    df_b = pd.read_csv(eq_dir / "resumen_equivalencia_condicion_B.csv")
    assert df_a["escenario"].nunique() == 10
    assert df_b["escenario"].nunique() == 6
    for df_sub in (df_a, df_b):
        assert np.all(np.isfinite(df_sub["MAE_CF_macro"].to_numpy(float)))
        assert np.all(np.isfinite(df_sub["MAE_macro_kWh"].to_numpy(float)))


def test_v3_ablation_huber_monotonicity_and_bias_artifacts() -> None:
    """Validate controlled ablation, monotonicity, Huber stress/leverage, and bias CSVs."""
    ab_dir = ROOT / "resultados" / "ablaciones_controladas_v3"
    if not ab_dir.exists():
        pytest.skip("ablaciones_controladas_v3 not present")

    df_ab = pd.read_csv(ab_dir / "ablaciones_controladas_6_escenarios.csv")
    assert df_ab["escenario"].nunique() == 6
    assert np.all(np.isfinite(df_ab["MAE_CF_macro"].to_numpy(float)))

    df_mo = pd.read_csv(ab_dir / "auditoria_monotonia_activacion.csv")
    assert len(df_mo) == 6
    assert np.all(np.isfinite(df_mo["pct_cf30_activa"].to_numpy(float)))

    df_hu = pd.read_csv(ab_dir / "auditoria_huber_operacional_y_estres.csv")
    assert len(df_hu) == 7
    assert np.all(np.isfinite(df_hu["MAE_CF_Huber"].to_numpy(float)))
    assert np.all(np.isfinite(df_hu["MAE_CF_L2"].to_numpy(float)))

    df_bias = pd.read_csv(ab_dir / "sensibilidad_sesgo_seleccion_horas_xm.csv")
    assert len(df_bias) == 8
    assert np.all(np.isfinite(df_bias["MAE_CF_macro"].to_numpy(float)))

    df_ver = pd.read_csv(ab_dir / "veredictos_componentes_v3.csv")
    assert len(df_ver) == 8
    assert (ab_dir / "MANIFIESTO_ABLACIONES_V3.csv").exists()


"""Audited metric definitions, true capacity-factor MAE, and scale-normalization checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ashr_oof_pipeline.config import ScenarioSpec


def load_d1_plant_capacity_table(root: Path) -> pd.DataFrame:
    """Load time-varying installed capacity (cap_MW) and causal origin capacity (cap_MW_origen) per (code, date)."""
    base_path = root / "base_generacion_clima" / "base_planta_dia.csv"
    df = pd.read_csv(base_path, usecols=["code", "date", "cap_MW"])
    df["code"] = df["code"].astype(str)
    df["date"] = df["date"].astype(str)
    df = df.drop_duplicates(subset=["code", "date"]).sort_values(["code", "date"]).reset_index(drop=True)
    df["cap_MW_origen"] = df.groupby("code")["cap_MW"].shift(1).fillna(df["cap_MW"])
    return df


def detect_capacity_changes(base_cap_df: pd.DataFrame) -> pd.DataFrame:
    """Identify plants whose installed capacity changed during the 2024-2026 study window."""
    rows = []
    for code, grp in base_cap_df.sort_values("date").groupby("code"):
        caps = grp["cap_MW"].to_numpy(float)
        uniq = pd.unique(caps)
        if len(uniq) > 1:
            rows.append({
                "code": str(code),
                "n_distinct_caps": int(len(uniq)),
                "cap_min_MW": float(np.min(caps)),
                "cap_max_MW": float(np.max(caps)),
                "fecha_inicio": str(grp["date"].iloc[0]),
                "fecha_fin": str(grp["date"].iloc[-1]),
                "capacidades_MW": ", ".join(f"{c:.2f}" for c in uniq),
            })
    return pd.DataFrame(rows)


def attach_capacity_and_cf_errors(
    df: pd.DataFrame,
    sp: ScenarioSpec,
    bench_raw: pd.DataFrame,
    d1_cap_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach installed capacity and exact dimensionless capacity-factor absolute error."""
    out = df.copy()
    out["code"] = out["code"].astype(str)
    ae_kwh = np.abs(out[sp.pred_col].to_numpy(float) - out[sp.target_col].to_numpy(float))
    out["_ae_kwh"] = ae_kwh
    out["_sq_kwh"] = (out[sp.pred_col].to_numpy(float) - out[sp.target_col].to_numpy(float)) ** 2

    if sp.mode_eval == "d1":
        if d1_cap_df is None:
            raise ValueError("d1_cap_df es obligatorio para escenarios D+1.")
        out["date"] = out["date"].astype(str)
        out = out.merge(d1_cap_df, on=["code", "date"], how="left")
        cap_col = "cap_MW_origen" if "cap_MW_origen" in out.columns else "cap_MW"
        if out[cap_col].isna().any():
            raise ValueError(f"Falta {cap_col} en D+1 para escenario {sp.scenario_id}")
        denom = 24000.0 * out[cap_col].to_numpy(float)
        out["_ae_cf"] = ae_kwh / denom
        out["cap_MW_usada"] = out[cap_col]
    else:
        if "cap_MW_origen" not in out.columns:
            cap_keys = ["code", "origen"]
            b_caps = bench_raw[cap_keys + ["cap_MW_origen"]].drop_duplicates(subset=cap_keys)
            b_caps["code"] = b_caps["code"].astype(str)
            b_caps["origen"] = b_caps["origen"].astype(str)
            out["origen"] = out["origen"].astype(str)
            out = out.merge(b_caps, on=cap_keys, how="left")
        if out["cap_MW_origen"].isna().any():
            raise ValueError(f"Falta cap_MW_origen para escenario {sp.scenario_id}")
        mult_h = float(sp.horizon_days) if sp.mode_eval == "direct" else 1.0
        denom = 24000.0 * out["cap_MW_origen"].to_numpy(float) * mult_h
        out["_ae_cf"] = ae_kwh / denom
        out["cap_MW"] = out["cap_MW_origen"]
    return out


def compute_audited_metrics_from_df(df_with_cf: pd.DataFrame, sp: ScenarioSpec) -> dict[str, float]:
    """Compute explicitly named kWh and true capacity-factor metrics for one model."""
    ae_kwh = df_with_cf["_ae_kwh"].to_numpy(float)
    ae_cf = df_with_cf["_ae_cf"].to_numpy(float)
    sq_kwh = df_with_cf["_sq_kwh"].to_numpy(float)
    y_kwh = df_with_cf[sp.target_col].to_numpy(float)

    by_plant = df_with_cf.assign(_y=y_kwh).groupby("code").agg(
        mae_kwh=("_ae_kwh", "mean"),
        mae_cf=("_ae_cf", "mean"),
        y_mean=("_y", "mean"),
    )
    mae_macro_kwh = float(by_plant["mae_kwh"].mean())
    mae_macro_cf = float(by_plant["mae_cf"].mean())
    nmae_rel_gen = float((by_plant["mae_kwh"] / np.maximum(by_plant["y_mean"], 1e-6)).mean())

    h_div = float(sp.horizon_days) if sp.mode_eval == "direct" else 1.0
    return {
        "MAE_macro_kWh": mae_macro_kwh,
        "MAE_macro_kWh_por_dia": mae_macro_kwh / h_div,
        "MAE_micro_kWh": float(np.mean(ae_kwh)),
        "MAE_CF_macro": mae_macro_cf,
        "MAE_CF_micro": float(np.mean(ae_cf)),
        "nMAE_rel_generacion_media_planta": nmae_rel_gen,
        "RMSE_micro_kWh": float(np.sqrt(np.mean(sq_kwh))),
        "RMSE_macro_planta_kWh": float(np.sqrt(df_with_cf.groupby("code")["_sq_kwh"].mean()).mean()),
    }


def build_metric_definitions_catalog() -> pd.DataFrame:
    """Return authoritative catalog of metric definitions, units, and denominators."""
    rows: list[dict[str, Any]] = [
        {
            "metrica": "MAE_macro_kWh",
            "unidad": "kWh/día (D+1/Rec) o kWh totales (Directo)",
            "formula": "(1/M) * sum_i (1/N_i) * sum_t |y_hat_{i,t} - y_{i,t}|",
            "denominador": "Ninguno (escala física absoluta; pondera igual cada planta i)",
            "alcance": "Métrica primaria congelada del protocolo",
        },
        {
            "metrica": "MAE_macro_kWh_por_dia",
            "unidad": "kWh/día",
            "formula": "MAE_macro_kWh / H (en directos) o MAE_macro_kWh (D+1/Rec)",
            "denominador": "H días en directos acumulados",
            "alcance": "Comparabilidad diaria entre estrategias directa y recursiva",
        },
        {
            "metrica": "MAE_micro_kWh",
            "unidad": "kWh/día o kWh totales",
            "formula": "(1/N) * sum_{i,t} |y_hat_{i,t} - y_{i,t}|",
            "denominador": "Total de observaciones N (mayor peso a plantas con series más largas)",
            "alcance": "Métrica secundaria diagnóstica",
        },
        {
            "metrica": "MAE_CF_macro",
            "unidad": "adimensional (factor de capacidad in [0,1])",
            "formula": "(1/M) * sum_i (1/N_i) * sum_t |y_hat_{i,t} - y_{i,t}| / (24000 * cap_MW_{i,t} * H_mult)",
            "denominador": "Capacidad instalada vigente 24000 * cap_MW_{i,t} (* H en directos)",
            "alcance": "Métrica adicional libre de escala entre plantas (9.8 a 388.0 MW)",
        },
        {
            "metrica": "nMAE_rel_generacion_media_planta",
            "unidad": "adimensional (razón respecto a generación media)",
            "formula": "(1/M) * sum_i (MAE_kWh_i / mean_t(y_{i,t}))",
            "denominador": "Generación media observada de la planta (NO es factor de capacidad)",
            "alcance": "Métrica renombrada de auditoría previa (antes ambigua como CF)",
        },
        {
            "metrica": "Skill_MAE_vs_MM30_pct",
            "unidad": "%",
            "formula": "100 * (1 - MAE_macro(m) / MAE_macro(MM30))",
            "denominador": "MAE_macro de Media móvil de 30 días (MM30)",
            "alcance": "Ganancia porcentual frente a persistencia causal MM30",
        },
        {
            "metrica": "Skill_RMSE_vs_MM30_pct",
            "unidad": "%",
            "formula": "100 * (1 - RMSE(m) / RMSE(MM30))",
            "denominador": "RMSE de Media móvil de 30 días (MM30; NO climatología-persistencia)",
            "alcance": "Ganancia cuadrática frente a MM30",
        },
    ]
    return pd.DataFrame(rows)


__all__ = [
    "attach_capacity_and_cf_errors",
    "build_metric_definitions_catalog",
    "compute_audited_metrics_from_df",
    "detect_capacity_changes",
    "load_d1_plant_capacity_table",
]

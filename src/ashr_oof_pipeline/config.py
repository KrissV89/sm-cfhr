"""Configuration, frozen ASHR builder, and 28-scenario official solar inventory."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from adaptive_structured_huber import AdaptiveStructuredHuberRegressor
except ImportError:
    from src.adaptive_structured_huber import AdaptiveStructuredHuberRegressor


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    suite: str
    task: str
    horizon_days: int
    weather_on: bool
    mode_eval: str  # "d1" | "recursive" | "direct"
    unit: str
    benchmark_relpath: str
    target_col: str
    pred_col: str
    join_keys: tuple[str, ...]


def find_project_root(start: Path | None = None) -> Path:
    """Locate repository root cleanly from notebooks, scripts, or tests."""
    cur = (start or Path(__file__)).resolve()
    for candidate in (cur, *cur.parents):
        if (candidate / "experiments" / "config_ashr_congelada.json").exists():
            return candidate
        if (candidate / "src" / "adaptive_structured_huber.py").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root containing config_ashr_congelada.json")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_frozen_config(root: Path | None = None) -> dict[str, Any]:
    r = root or find_project_root()
    cfg_path = r / "experiments" / "config_ashr_congelada.json"
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def build_frozen_ashr(
    mode: str = "espaciotemporal",
    horizon: int = 1,
    weather_mode: str = "forecast",
    root: Path | None = None,
) -> AdaptiveStructuredHuberRegressor:
    """Instantiate the exact frozen Fijo_Lineal_Mono (ASHR-Fixed / SM-CFHR) configuration."""
    cfg = load_frozen_config(root)
    p = cfg["parametros_modelo"]
    return AdaptiveStructuredHuberRegressor(
        mode=mode,
        horizon=horizon,
        weather_mode=weather_mode,
        selection=p["selection"],
        fixed_config=p["fixed_config"],
        delta=p["delta"],
        max_iter=p["max_iter"],
        tol=p["tol"],
        random_state=p["random_state"],
    )


def _build_scenarios_28() -> tuple[ScenarioSpec, ...]:
    specs: list[ScenarioSpec] = []
    # 1-4: D+1
    for task in ("espaciotemporal", "intraplanta"):
        for w_on, w_lbl in ((True, "cinco_variables"), (False, "sin_meteorologia")):
            specs.append(
                ScenarioSpec(
                    scenario_id=f"D1_{task}_{w_lbl}",
                    suite="D+1 (37-40)",
                    task=task,
                    horizon_days=1,
                    weather_on=w_on,
                    mode_eval="d1",
                    unit="kWh/dia",
                    benchmark_relpath=f"resultados/benchmark_produccion_definitivo/{task}_{w_lbl}/predicciones.csv",
                    target_col="y_kwh",
                    pred_col="prediccion_kwh",
                    join_keys=("code", "date"),
                )
            )
    # 5-12: Rec sin clima
    for h in (7, 14, 21, 28):
        for task in ("espaciotemporal", "intraplanta"):
            specs.append(
                ScenarioSpec(
                    scenario_id=f"RecSinClima_{task}_{h}d",
                    suite="Rec sin clima (43-46)",
                    task=task,
                    horizon_days=h,
                    weather_on=False,
                    mode_eval="recursive",
                    unit="kWh/dia",
                    benchmark_relpath=f"resultados/benchmark_multihorizonte/horizonte_{h}_dias/predicciones_oof.csv",
                    target_col="y_kwh",
                    pred_col="prediccion_kwh",
                    join_keys=("code", "origen", "lead_dias"),
                )
            )
    # 13-20: Directo sin clima
    for h in (7, 14, 21, 28):
        for task in ("espaciotemporal", "intraplanta"):
            specs.append(
                ScenarioSpec(
                    scenario_id=f"DirectoSinClima_{task}_{h}d",
                    suite="Directo sin clima (48-51)",
                    task=task,
                    horizon_days=h,
                    weather_on=False,
                    mode_eval="direct",
                    unit="kWh_total",
                    benchmark_relpath=f"resultados/benchmark_multihorizonte_directo/horizonte_{h}_dias/predicciones_oof.csv",
                    target_col="y_total_kwh",
                    pred_col="prediccion_total_kwh",
                    join_keys=("code", "origen"),
                )
            )
    # 21-28: Rec GEFSv12
    for h in (7, 14, 21, 28):
        for task in ("espaciotemporal", "intraplanta"):
            specs.append(
                ScenarioSpec(
                    scenario_id=f"RecGEFSv12_{task}_{h}d",
                    suite="Rec con clima (53-56)",
                    task=task,
                    horizon_days=h,
                    weather_on=True,
                    mode_eval="recursive",
                    unit="kWh/dia",
                    benchmark_relpath=f"resultados/benchmark_multihorizonte_clima/horizonte_{h}_dias/predicciones_oof.csv",
                    target_col="y_kwh",
                    pred_col="prediccion_kwh",
                    join_keys=("code", "origen", "lead_dias"),
                )
            )
    assert len(specs) == 28
    return tuple(specs)


SCENARIOS_28: tuple[ScenarioSpec, ...] = _build_scenarios_28()


def verify_benchmark_inventory(root: Path | None = None) -> pd.DataFrame:
    """Audit existence, columns, and 22-model universe (CF-BRH excluded) across all 28 scenarios."""
    r = root or find_project_root()
    rows = []
    cache: dict[Path, pd.DataFrame] = {}
    for sp in SCENARIOS_28:
        fpath = r / sp.benchmark_relpath
        exists = fpath.exists()
        n_models, has_cf_brh, cols_ok = 0, False, False
        if exists:
            if fpath not in cache:
                cache[fpath] = pd.read_csv(fpath, usecols=lambda c: c in {"metodo", "tarea", sp.target_col, sp.pred_col, *sp.join_keys})
            df = cache[fpath]
            if "tarea" in df.columns:
                df = df[df["tarea"] == sp.task]
            methods = set(df["metodo"].unique())
            has_cf_brh = "CF-BRH" in methods
            n_models = len(methods - {"CF-BRH"})
            cols_ok = all(c in df.columns for c in (*sp.join_keys, sp.target_col, sp.pred_col, "metodo"))
        rows.append({
            "scenario_id": sp.scenario_id,
            "suite": sp.suite,
            "task": sp.task,
            "horizon_days": sp.horizon_days,
            "weather_on": sp.weather_on,
            "mode_eval": sp.mode_eval,
            "unit": sp.unit,
            "file_exists": exists,
            "cols_ok": cols_ok,
            "n_official_models": n_models,
            "cf_brh_excluded": not has_cf_brh,
        })
    return pd.DataFrame(rows)

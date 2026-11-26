"""Deterministic work-unit inventory and disjoint sharding across the 28 official scenarios."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    from ashr_oof_pipeline.config import SCENARIOS_28, ScenarioSpec, find_project_root
    from ashr_oof_pipeline.oof_runner import load_official_benchmarks
except ImportError:
    from src.ashr_oof_pipeline.config import SCENARIOS_28, ScenarioSpec, find_project_root
    from src.ashr_oof_pipeline.oof_runner import load_official_benchmarks

HORIZON_BUCKETS: tuple[str, ...] = ("D1", "7d", "14d", "21d", "28d")


def horizon_bucket(horizon_days: int) -> str:
    """Map integer horizon days to canonical storage bucket."""
    h = int(horizon_days)
    return "D1" if h == 1 else f"{h}d"


@dataclass(frozen=True)
class WorkUnitSpec:
    """Immutable descriptor of a single independent OOF work unit."""

    unit_id: str
    variant: str  # "fixed" | "hpo"
    horizon_bucket: str  # "D1" | "7d" | "14d" | "21d" | "28d"
    scenario_id: str
    suite: str
    task: str
    horizon_days: int
    weather_on: bool
    mode_eval: str  # "d1" | "recursive" | "direct"
    unit_tag: str
    unit_entity: str
    block_or_origin: str
    pred_codes: tuple[str, ...]
    n_trials: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["pred_codes"] = list(self.pred_codes)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkUnitSpec":
        d = dict(data)
        d["pred_codes"] = tuple(str(c) for c in d.get("pred_codes", ()))
        return cls(**d)


def _select_scenarios(
    scenarios: Iterable[str] | None = None,
    horizons: Iterable[int] | None = None,
) -> list[ScenarioSpec]:
    sc_filter = set(scenarios) if scenarios is not None else None
    h_filter = {int(h) for h in horizons} if horizons is not None else None
    out: list[ScenarioSpec] = []
    for sp in SCENARIOS_28:
        if sc_filter is not None and sp.scenario_id not in sc_filter:
            continue
        if h_filter is not None and sp.horizon_days not in h_filter:
            continue
        out.append(sp)
    return out


def build_work_inventory(
    root: Path | None = None,
    data_root: Path | None = None,
    variante: str = "fixed",
    n_trials: int = 25,
    seed: int = 2026,
    scenarios: Iterable[str] | None = None,
    horizons: Iterable[int] | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
) -> list[WorkUnitSpec]:
    """Build deterministic inventory of independent OOF work units using exact frozen partition rules."""
    if num_shards < 1 or not (0 <= shard_index < num_shards):
        raise ValueError(f"Invalid shard specification: shard_index={shard_index}, num_shards={num_shards}")
    if variante not in ("fixed", "hpo", "ambos"):
        raise ValueError(f"Invalid variante: {variante}")

    repo_root = root or find_project_root()
    d_root = data_root or repo_root
    selected = _select_scenarios(scenarios=scenarios, horizons=horizons)
    if not selected:
        return []

    from scripts.evaluacion import produccion_intraplanta as daily
    from scripts.evaluacion import produccion_multihorizonte as mh
    from scripts.evaluacion import produccion_multihorizonte_clima as mhc
    from scripts.evaluacion import produccion_multihorizonte_directo as mhd

    base_path = d_root / "base_generacion_clima" / "base_planta_dia.csv"
    forecast_path = d_root / "datos_pronostico" / "pronostico_diario.csv"
    gefs_path = d_root / "datos_pronostico_multihorizonte" / "pronostico_28d_gefs.csv"

    run_variants = ("fixed", "hpo") if variante == "ambos" else (variante,)
    d1_panel_cache: dict[bool, pd.DataFrame] = {}
    rec_cache: dict[bool, tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, Any] | None]] = {}
    dir_panel_cache: dict[int, pd.DataFrame] = {}

    raw_units: list[WorkUnitSpec] = []
    for sp in selected:
        h_bkt = horizon_bucket(sp.horizon_days)
        bench = load_official_benchmarks(sp, root=d_root)

        if sp.mode_eval == "d1":
            if sp.weather_on not in d1_panel_cache:
                d1_panel_cache[sp.weather_on] = daily.build_panel(base_path, forecast_path, sp.weather_on)
            panel = d1_panel_cache[sp.weather_on]
            if "bloque" not in bench.columns:
                continue
            for blk, bdf in bench.groupby("bloque", sort=True):
                first_ts = pd.to_datetime(bdf["date"]).min()
                last_ts = pd.to_datetime(bdf["date"]).max()
                units = (
                    sorted(panel["grupo"].unique())
                    if sp.task == "espaciotemporal"
                    else sorted(panel["code"].unique())
                )
                for unit in units:
                    train = (
                        panel[(panel["grupo"] != unit) & (panel["date"] < first_ts)]
                        if sp.task == "espaciotemporal"
                        else panel[(panel["code"] == unit) & (panel["date"] < first_ts)]
                    )
                    test = (
                        panel[(panel["grupo"] == unit) & panel["date"].between(first_ts, last_ts)]
                        if sp.task == "espaciotemporal"
                        else panel[(panel["code"] == unit) & panel["date"].between(first_ts, last_ts)]
                    )
                    if train.empty or test.empty:
                        continue
                    unit_tag = f"{blk}_{unit}"
                    for var in run_variants:
                        raw_units.append(
                            WorkUnitSpec(
                                unit_id=f"{var}__{sp.scenario_id}__{unit_tag}",
                                variant=var,
                                horizon_bucket=h_bkt,
                                scenario_id=sp.scenario_id,
                                suite=sp.suite,
                                task=sp.task,
                                horizon_days=sp.horizon_days,
                                weather_on=sp.weather_on,
                                mode_eval=sp.mode_eval,
                                unit_tag=unit_tag,
                                unit_entity=str(unit),
                                block_or_origin=str(blk),
                                pred_codes=(),
                                n_trials=int(n_trials) if var == "hpo" else 0,
                                seed=int(seed),
                            )
                        )

        elif sp.mode_eval == "recursive":
            if sp.weather_on not in rec_cache:
                if sp.weather_on:
                    p_rec, rp, ws = mhc.load_generation_weather_panel(base_path, gefs_path)
                else:
                    p_rec, rp = mh.load_generation_panel(base_path)
                    ws = None
                rec_cache[sp.weather_on] = (p_rec, rp, ws)
            panel, raw_plants, weather_store = rec_cache[sp.weather_on]
            origins = sorted(pd.Timestamp(o) for o in bench["origen"].unique())
            units = (
                sorted(panel["grupo"].unique())
                if sp.task == "espaciotemporal"
                else sorted(panel["code"].unique())
            )
            for u in units:
                if sp.weather_on:
                    _, eval_specs, _ = mhc._plan_unit_origins(
                        sp.task, u, origins, sp.horizon_days, panel, raw_plants, weather_store
                    )
                else:
                    _, eval_specs, _ = mh._plan_unit_origins(
                        sp.task, u, origins, sp.horizon_days, panel, raw_plants
                    )
                for spec in eval_specs:
                    origin = spec["origin"]
                    unit_tag = f"{u}_{origin.date()}"
                    pred_codes = tuple(str(c) for c in spec["pred_codes"])
                    for var in run_variants:
                        raw_units.append(
                            WorkUnitSpec(
                                unit_id=f"{var}__{sp.scenario_id}__{unit_tag}",
                                variant=var,
                                horizon_bucket=h_bkt,
                                scenario_id=sp.scenario_id,
                                suite=sp.suite,
                                task=sp.task,
                                horizon_days=sp.horizon_days,
                                weather_on=sp.weather_on,
                                mode_eval=sp.mode_eval,
                                unit_tag=unit_tag,
                                unit_entity=str(u),
                                block_or_origin=str(origin.date()),
                                pred_codes=pred_codes,
                                n_trials=int(n_trials) if var == "hpo" else 0,
                                seed=int(seed),
                            )
                        )

        else:  # direct
            if sp.horizon_days not in dir_panel_cache:
                p_dir, _ = mhd.build_direct_panel(base_path, sp.horizon_days)
                dir_panel_cache[sp.horizon_days] = p_dir
            panel = dir_panel_cache[sp.horizon_days]
            origins = sorted(pd.Timestamp(o) for o in bench["origen"].unique())
            units = (
                sorted(panel["grupo"].unique())
                if sp.task == "espaciotemporal"
                else sorted(panel["code"].unique())
            )
            for origin in origins:
                for u in units:
                    train = (
                        panel[(panel["code"] == str(u)) & (panel["t_end_target"] <= origin)]
                        if sp.task == "intraplanta"
                        else panel[(panel["grupo"] != int(u)) & (panel["t_end_target"] <= origin)]
                    )
                    query = (
                        panel[(panel["code"] == str(u)) & (panel["t_origin"] == origin)]
                        if sp.task == "intraplanta"
                        else panel[(panel["grupo"] == int(u)) & (panel["t_origin"] == origin)]
                    )
                    if len(train) < 30 or query.empty:
                        continue
                    unit_tag = f"{u}_{origin.date()}"
                    for var in run_variants:
                        raw_units.append(
                            WorkUnitSpec(
                                unit_id=f"{var}__{sp.scenario_id}__{unit_tag}",
                                variant=var,
                                horizon_bucket=h_bkt,
                                scenario_id=sp.scenario_id,
                                suite=sp.suite,
                                task=sp.task,
                                horizon_days=sp.horizon_days,
                                weather_on=sp.weather_on,
                                mode_eval=sp.mode_eval,
                                unit_tag=unit_tag,
                                unit_entity=str(u),
                                block_or_origin=str(origin.date()),
                                pred_codes=(),
                                n_trials=int(n_trials) if var == "hpo" else 0,
                                seed=int(seed),
                            )
                        )

    # Clear temporary caches immediately
    d1_panel_cache.clear()
    rec_cache.clear()
    dir_panel_cache.clear()

    if num_shards == 1:
        return raw_units
    return [u for idx, u in enumerate(raw_units) if (idx % num_shards) == shard_index]

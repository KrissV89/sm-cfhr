"""Isolated Windows-spawn worker with single-thread BLAS, 1-scenario cache, and process recycling."""

from __future__ import annotations

import os

# Prevent BLAS/OpenMP oversubscription across worker processes before importing numpy/scipy
for _env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
):
    os.environ[_env] = "1"

import gc
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from ashr_oof_pipeline.oof_runner import (
        fit_ashr_on_train,
        load_official_benchmarks,
        predict_ashr_on_query,
        rollout_recursive_from_origin,
    )
    from sm_cfhr.persistence import WorkUnitLock, get_process_rss_mb
except ImportError:
    from src.ashr_oof_pipeline.oof_runner import (
        fit_ashr_on_train,
        load_official_benchmarks,
        predict_ashr_on_query,
        rollout_recursive_from_origin,
    )
    from src.sm_cfhr.persistence import WorkUnitLock, get_process_rss_mb

from .storage import HorizonStorageLayout
from .work_units import WorkUnitSpec


class _ScenarioContextCache:
    """Keep only ONE scenario dataset in worker RAM at a time; release immediately on switch."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = Path(data_root)
        self._key: tuple[str, str, int, bool] | None = None
        self._ctx: dict[str, Any] = {}

    def clear(self) -> None:
        self._ctx.clear()
        self._key = None
        gc.collect()

    def get(self, unit: WorkUnitSpec) -> dict[str, Any]:
        key = (unit.scenario_id, unit.mode_eval, unit.horizon_days, unit.weather_on)
        if self._key == key:
            return self._ctx

        self.clear()

        from ashr_oof_pipeline.config import ScenarioSpec
        from scripts.evaluacion import produccion_intraplanta as daily
        from scripts.evaluacion import produccion_multihorizonte as mh
        from scripts.evaluacion import produccion_multihorizonte_clima as mhc
        from scripts.evaluacion import produccion_multihorizonte_directo as mhd

        base_path = self.data_root / "base_generacion_clima" / "base_planta_dia.csv"
        forecast_path = self.data_root / "datos_pronostico" / "pronostico_diario.csv"
        gefs_path = self.data_root / "datos_pronostico_multihorizonte" / "pronostico_28d_gefs.csv"

        if unit.mode_eval == "d1":
            w_lbl = "cinco_variables" if unit.weather_on else "sin_meteorologia"
            sp_stub = ScenarioSpec(
                scenario_id=unit.scenario_id,
                suite=unit.suite,
                task=unit.task,
                horizon_days=1,
                weather_on=unit.weather_on,
                mode_eval="d1",
                unit="kWh/dia",
                benchmark_relpath=f"resultados/benchmark_produccion_definitivo/{unit.task}_{w_lbl}/predicciones.csv",
                target_col="y_kwh",
                pred_col="prediccion_kwh",
                join_keys=("code", "date"),
            )
            bench = load_official_benchmarks(sp_stub, root=self.data_root)
            block_bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
            for blk, bdf in bench.groupby("bloque"):
                block_bounds[str(blk)] = (pd.to_datetime(bdf["date"]).min(), pd.to_datetime(bdf["date"]).max())
            panel = daily.build_panel(base_path, forecast_path, unit.weather_on)
            del bench
            self._ctx = {"panel": panel, "block_bounds": block_bounds}
        elif unit.mode_eval == "recursive":
            if unit.weather_on:
                panel, raw_plants, weather_store = mhc.load_generation_weather_panel(base_path, gefs_path)
            else:
                panel, raw_plants = mh.load_generation_panel(base_path)
                weather_store = None
            self._ctx = {"panel": panel, "raw_plants": raw_plants, "weather_store": weather_store}
        else:  # direct
            panel, _ = mhd.build_direct_panel(base_path, unit.horizon_days)
            self._ctx = {"panel": panel}

        self._key = key
        return self._ctx


def execute_single_unit(
    unit: WorkUnitSpec,
    storage: HorizonStorageLayout,
    ctx_cache: _ScenarioContextCache,
    repo_root: Path,
    hpo_proto_hash: str,
    expected_src_hash: str,
) -> dict[str, Any]:
    """Execute one independent unit, persist immediately, and return lightweight metadata."""
    use_hpo = unit.variant == "hpo"
    proto_hash = hpo_proto_hash if use_hpo else expected_src_hash
    rss_before = get_process_rss_mb()

    t_load0 = time.perf_counter()
    ctx = ctx_cache.get(unit)
    load_time_s = time.perf_counter() - t_load0

    t_comp0 = time.perf_counter()
    storage_url = storage.unit_optuna_storage_url(unit) if use_hpo else None
    study_name = f"{unit.scenario_id}__{unit.unit_tag}" if use_hpo else None

    if unit.mode_eval == "d1":
        panel: pd.DataFrame = ctx["panel"]
        first_ts, last_ts = ctx["block_bounds"][unit.block_or_origin]
        u_val = int(unit.unit_entity) if unit.task == "espaciotemporal" else str(unit.unit_entity)
        train = (
            panel[(panel["grupo"] != u_val) & (panel["date"] < first_ts)].copy()
            if unit.task == "espaciotemporal"
            else panel[(panel["code"] == u_val) & (panel["date"] < first_ts)].copy()
        )
        test = (
            panel[(panel["grupo"] == u_val) & panel["date"].between(first_ts, last_ts)].copy()
            if unit.task == "espaciotemporal"
            else panel[(panel["code"] == u_val) & panel["date"].between(first_ts, last_ts)].copy()
        )
        fitted = fit_ashr_on_train(
            train,
            unit.task,
            1,
            unit.weather_on,
            root=repo_root,
            use_hpo=use_hpo,
            mode_eval="d1",
            n_trials=unit.n_trials,
            outer_fold_id=unit.unit_tag,
            seed=unit.seed,
            storage_url=storage_url,
            study_name=study_name,
            proto_hash=proto_hash,
        )
        t_df = fitted["trials_df"]
        p_hat = predict_ashr_on_query(fitted, test, lead=1)
        u_df = pd.DataFrame(
            [
                {
                    "metodo": "ASHR-HPO" if use_hpo else "ASHR",
                    "bloque": unit.block_or_origin,
                    "code": str(r_t.code),
                    "grupo": int(r_t.grupo),
                    "date": str(pd.Timestamp(r_t.date).date()),
                    "prediccion_kwh": float(val),
                    "y_kwh": float(r_t.y),
                }
                for r_t, val in zip(test.itertuples(index=False), p_hat)
            ]
        )
        del train, test, fitted, p_hat

    elif unit.mode_eval == "recursive":
        panel = ctx["panel"]
        raw_plants = ctx["raw_plants"]
        weather_store = ctx["weather_store"]
        origin = pd.Timestamp(unit.block_or_origin)
        pred_infos = [raw_plants[c] for c in unit.pred_codes]
        train = (
            panel[(panel["code"] == str(unit.unit_entity)) & (panel["date"] <= origin)].copy()
            if unit.task == "intraplanta"
            else panel[(panel["grupo"] != int(unit.unit_entity)) & (panel["date"] <= origin)].copy()
        )
        fitted = fit_ashr_on_train(
            train,
            unit.task,
            unit.horizon_days,
            unit.weather_on,
            root=repo_root,
            use_hpo=use_hpo,
            mode_eval="recursive",
            n_trials=unit.n_trials,
            outer_fold_id=unit.unit_tag,
            seed=unit.seed,
            storage_url=storage_url,
            study_name=study_name,
            proto_hash=proto_hash,
            raw_plants=raw_plants,
            weather_store=weather_store,
        )
        t_df = fitted["trials_df"]
        raw_rows = rollout_recursive_from_origin(
            fitted=fitted,
            pred_infos=pred_infos,
            origin=origin,
            horizon_days=unit.horizon_days,
            weather_on=unit.weather_on,
            weather_store=weather_store,
        )
        metodo_lbl = "ASHR-HPO" if use_hpo else "ASHR"
        u_df = pd.DataFrame([{"metodo": metodo_lbl, "tarea": unit.task, **rw} for rw in raw_rows])
        del train, fitted, raw_rows, pred_infos

    else:  # direct
        panel = ctx["panel"]
        origin = pd.Timestamp(unit.block_or_origin)
        train = (
            panel[(panel["code"] == str(unit.unit_entity)) & (panel["t_end_target"] <= origin)].copy()
            if unit.task == "intraplanta"
            else panel[(panel["grupo"] != int(unit.unit_entity)) & (panel["t_end_target"] <= origin)].copy()
        )
        query = (
            panel[(panel["code"] == str(unit.unit_entity)) & (panel["t_origin"] == origin)].copy()
            if unit.task == "intraplanta"
            else panel[(panel["grupo"] == int(unit.unit_entity)) & (panel["t_origin"] == origin)].copy()
        )
        fitted = fit_ashr_on_train(
            train,
            unit.task,
            unit.horizon_days,
            False,
            horizon_direct=unit.horizon_days,
            target_col="y_H",
            root=repo_root,
            use_hpo=use_hpo,
            mode_eval="direct",
            n_trials=unit.n_trials,
            outer_fold_id=unit.unit_tag,
            seed=unit.seed,
            storage_url=storage_url,
            study_name=study_name,
            proto_hash=proto_hash,
        )
        t_df = fitted["trials_df"]
        yhat = predict_ashr_on_query(fitted, query, lead=1)
        u_df = pd.DataFrame(
            [
                {
                    "metodo": "ASHR-HPO" if use_hpo else "ASHR",
                    "tarea": unit.task,
                    "code": str(r_q.code),
                    "grupo": int(r_q.grupo),
                    "origen": str(origin.date()),
                    "prediccion_total_kwh": float(val),
                    "y_total_kwh": float(r_q.y_H),
                }
                for r_q, val in zip(query.itertuples(index=False), yhat)
            ]
        )
        del train, query, fitted, yhat

    compute_time_s = time.perf_counter() - t_comp0

    t_write0 = time.perf_counter()
    rss_before_write = get_process_rss_mb()
    rec = storage.persist_unit_outputs(
        unit,
        pred_df=u_df,
        trials_df=t_df,
        metrics={
            "load_time_s": round(load_time_s, 4),
            "compute_time_s": round(compute_time_s, 4),
            "rss_before_mb": round(rss_before, 2),
        },
    )
    write_time_s = time.perf_counter() - t_write0

    n_rows = len(u_df)
    del u_df, t_df
    gc.collect()
    rss_after = get_process_rss_mb()

    return {
        "unit_id": unit.unit_id,
        "status": "COMPLETED",
        "artifact_path": rec["artifact_path"],
        "artifact_sha256": rec["artifact_sha256"],
        "n_rows": n_rows,
        "load_time_s": round(load_time_s, 4),
        "compute_time_s": round(compute_time_s, 4),
        "write_time_s": round(write_time_s, 4),
        "elapsed_s": round(load_time_s + compute_time_s + write_time_s, 4),
        "rss_before_mb": round(rss_before, 2),
        "rss_before_write_mb": round(rss_before_write, 2),
        "rss_after_mb": round(rss_after, 2),
        "pid": os.getpid(),
    }


def worker_process_loop(
    worker_slot: int,
    task_queue: Any,
    result_queue: Any,
    base_dir_str: str,
    run_id: str,
    repo_root_str: str,
    data_root_str: str,
    hpo_proto_hash: str,
    expected_src_hash: str,
    max_tasks_per_worker: int,
    worker_max_rss_mb: float,
) -> None:
    """Entry point for spawned worker processes; recycles cleanly on task count or RSS cap."""
    import sys

    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    repo_root = Path(repo_root_str)
    data_root = Path(data_root_str)
    storage = HorizonStorageLayout(base_dir=Path(base_dir_str), run_id=run_id)
    ctx_cache = _ScenarioContextCache(data_root=data_root)
    tasks_done = 0

    while True:
        item = task_queue.get()
        if item is None:
            ctx_cache.clear()
            result_queue.put(
                {
                    "kind": "WORKER_EXIT",
                    "worker_slot": worker_slot,
                    "pid": os.getpid(),
                    "reason": "SHUTDOWN",
                    "tasks_done": tasks_done,
                    "rss_mb": round(get_process_rss_mb(), 2),
                }
            )
            return

        unit = WorkUnitSpec.from_dict(item)
        try:
            with WorkUnitLock(storage.unit_lock_path(unit.unit_id)):
                if storage.is_unit_completed(unit):
                    res = {
                        "kind": "UNIT_DONE",
                        "worker_slot": worker_slot,
                        "unit_id": unit.unit_id,
                        "status": "COMPLETED",
                        "skipped_cached": True,
                        "n_rows": storage.get_unit_record(unit.unit_id).get("n_rows", 0),
                        "load_time_s": 0.0,
                        "compute_time_s": 0.0,
                        "write_time_s": 0.0,
                        "elapsed_s": 0.0,
                        "rss_after_mb": round(get_process_rss_mb(), 2),
                        "pid": os.getpid(),
                    }
                else:
                    storage.write_unit_record(unit.unit_id, "RUNNING")
                    res = execute_single_unit(
                        unit=unit,
                        storage=storage,
                        ctx_cache=ctx_cache,
                        repo_root=repo_root,
                        hpo_proto_hash=hpo_proto_hash,
                        expected_src_hash=expected_src_hash,
                    )
                    res["kind"] = "UNIT_DONE"
                    res["worker_slot"] = worker_slot
                    res["skipped_cached"] = False
            tasks_done += 1
            rss_now = float(res.get("rss_after_mb", get_process_rss_mb()))
            recycle = (max_tasks_per_worker > 0 and tasks_done >= max_tasks_per_worker) or (
                worker_max_rss_mb > 0 and rss_now >= worker_max_rss_mb
            )
            res["recycle_requested"] = recycle
            result_queue.put(res)
            if recycle:
                ctx_cache.clear()
                return
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            try:
                storage.write_unit_record(unit.unit_id, "FAILED", error=err_msg)
            except OSError:
                pass
            result_queue.put(
                {
                    "kind": "UNIT_FAILED",
                    "worker_slot": worker_slot,
                    "unit_id": unit.unit_id,
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                    "pid": os.getpid(),
                    "rss_after_mb": round(get_process_rss_mb(), 2),
                    "recycle_requested": True,
                }
            )
            ctx_cache.clear()
            return

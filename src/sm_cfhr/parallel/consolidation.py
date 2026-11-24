"""Deterministic, idempotent, memory-bounded consolidation and disjoint-run merging."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from ashr_oof_pipeline.config import SCENARIOS_28, find_project_root, sha256_file
    from ashr_oof_pipeline.optimizacion_hiperparametros import HPO_PROTOCOL_VERSION
    from sm_cfhr.persistence import atomic_write_csv, atomic_write_text
except ImportError:
    from src.ashr_oof_pipeline.config import SCENARIOS_28, find_project_root, sha256_file
    from src.ashr_oof_pipeline.optimizacion_hiperparametros import HPO_PROTOCOL_VERSION
    from src.sm_cfhr.persistence import atomic_write_csv, atomic_write_text

from .storage import HorizonStorageLayout
from .work_units import WorkUnitSpec


def _check_unique_keys(df: pd.DataFrame, join_keys: tuple[str, ...], scenario_id: str) -> None:
    keys = list(join_keys)
    if df[keys].isna().any().any():
        raise ValueError(f"Missing join keys in consolidated predictions for {scenario_id}: {keys}")
    if df.duplicated(subset=keys).any():
        n_dup = int(df.duplicated(subset=keys).sum())
        raise ValueError(f"Duplicate/contradictory rows ({n_dup}) detected in {scenario_id} on keys {keys}")


def consolidate_parallel_run(
    run_dir: Path,
    out_dir: Path | None = None,
    root: Path | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Consolidate persisted unit fragments scenario-by-scenario into standard OOF CSVs and manifest."""
    repo_root = Path(root or find_project_root())
    run_path = Path(run_dir)
    manifest_p = run_path / "manifest_run.json"
    inv_p = run_path / "inventory.json"
    if not manifest_p.exists() or not inv_p.exists():
        raise FileNotFoundError(f"Missing manifest_run.json or inventory.json in {run_path}")

    run_manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    inv_payload = json.loads(inv_p.read_text(encoding="utf-8"))
    units = [WorkUnitSpec.from_dict(u) for u in inv_payload.get("units", [])]
    if not units:
        raise ValueError(f"Empty inventory in {run_path}")

    storage = HorizonStorageLayout(base_dir=run_path.parent, run_id=run_path.name)
    target_dir = Path(out_dir or (run_path / "consolidated"))
    pred_dir = target_dir / "predicciones"
    hpo_dir = target_dir / "hpo_trials"
    pred_dir.mkdir(parents=True, exist_ok=True)

    sc_map = {sp.scenario_id: sp for sp in SCENARIOS_28}
    grouped: dict[tuple[str, str], list[WorkUnitSpec]] = {}
    for u in units:
        grouped.setdefault((u.variant, u.scenario_id), []).append(u)

    missing_units: list[str] = []
    invalid_units: list[str] = []
    pred_hashes: dict[str, str] = {}
    scenario_times: dict[str, float] = {}
    variants_seen: set[str] = set()
    scenarios_complete: set[str] = set()
    total_rows = 0

    for (var, sc_id), sc_units in sorted(grouped.items()):
        sp = sc_map[sc_id]
        dfs: list[pd.DataFrame] = []
        t_dfs: list[pd.DataFrame] = []
        sc_ok = True
        sc_time_s = 0.0
        for u in sc_units:
            rec = storage.get_unit_record(u.unit_id)
            st = rec.get("status", "PENDING")
            if st != "COMPLETED":
                missing_units.append(u.unit_id)
                sc_ok = False
                continue
            if not storage.is_unit_completed(u):
                invalid_units.append(u.unit_id)
                sc_ok = False
                continue
            sc_time_s += float(rec.get("load_time_s", 0.0)) + float(rec.get("compute_time_s", 0.0))
            dfs.append(pd.read_parquet(rec["artifact_path"]))
            if var == "hpo" and rec.get("trials_path"):
                t_dfs.append(pd.read_csv(rec["trials_path"]))

        if require_complete and not sc_ok:
            raise RuntimeError(
                f"Incomplete or corrupt units in {var}/{sc_id}: missing={len(missing_units)}, invalid={len(invalid_units)}"
            )
        if not dfs:
            continue

        variants_seen.add(var)
        df_sc = pd.concat(dfs, ignore_index=True)
        sort_cols = [c for c in sp.join_keys if c in df_sc.columns]
        if sort_cols:
            df_sc = df_sc.sort_values(sort_cols).reset_index(drop=True)
        _check_unique_keys(df_sc, sp.join_keys, sc_id)
        total_rows += len(df_sc)

        out_csv = pred_dir / (f"{sc_id}.csv" if var == "fixed" else f"{sc_id}_hpo.csv")
        pred_hashes[f"{var}__{sc_id}"] = atomic_write_csv(df_sc, out_csv)
        scenario_times[f"{var}__{sc_id}"] = round(sc_time_s, 3)
        del df_sc, dfs

        if var == "hpo" and t_dfs:
            hpo_dir.mkdir(parents=True, exist_ok=True)
            t_all = pd.concat(t_dfs, ignore_index=True)
            if {"outer_fold", "trial_number"}.issubset(t_all.columns):
                t_all = t_all.sort_values(["outer_fold", "trial_number"]).reset_index(drop=True)
            atomic_write_csv(t_all, hpo_dir / f"{sc_id}.csv")
            del t_all, t_dfs

        if sc_ok:
            scenarios_complete.add(f"{var}__{sc_id}")
        gc.collect()

    all_28_present = len(scenarios_complete) == (len(variants_seen) * len(SCENARIOS_28)) and not missing_units and not invalid_units
    estatus = "OOF_DEFINITIVO_COMPLETO" if all_28_present else "OOF_PARCIAL"

    oof_manifest = {
        "estatus": estatus,
        "procedencia": "OOF_DEFINITIVO_REAL",
        "run_id": run_manifest.get("run_id"),
        "variante": run_manifest.get("variante"),
        "variantes_completadas": sorted(variants_seen),
        "n_escenarios_28_completos": len(scenarios_complete),
        "n_unidades_esperadas": len(units),
        "n_unidades_faltantes": len(missing_units),
        "n_unidades_invalidas": len(invalid_units),
        "unidades_faltantes": missing_units[:50],
        "unidades_invalidas": invalid_units[:50],
        "total_filas_prediccion": int(total_rows),
        "sha256_src_adaptive_structured_huber": run_manifest.get("src_hash"),
        "config_hash": run_manifest.get("config_hash"),
        "hpo_protocol_version": run_manifest.get("hpo_protocol_version", HPO_PROTOCOL_VERSION),
        "hpo_protocol_hash": run_manifest.get("hpo_protocol_hash"),
        "seed": run_manifest.get("seed"),
        "n_trials": run_manifest.get("n_trials"),
        "tiempos_por_escenario_s": scenario_times,
        "hashes_sha256_predicciones": pred_hashes,
        "consolidated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    atomic_write_text(target_dir / "manifest_ashr_oof.json", json.dumps(oof_manifest, indent=2, ensure_ascii=False))
    return oof_manifest


def merge_disjoint_runs(
    run_dirs: list[Path],
    out_dir: Path,
    root: Path | None = None,
) -> dict[str, Any]:
    """Merge multiple disjoint sharded runs after verifying strict scientific identity and zero overlap."""
    if len(run_dirs) < 2:
        raise ValueError("merge_disjoint_runs requires at least two run directories.")

    repo_root = Path(root or find_project_root())
    manifests = [json.loads((Path(d) / "manifest_run.json").read_text(encoding="utf-8")) for d in run_dirs]
    ref = manifests[0]
    for m in manifests[1:]:
        for key in ("config_hash", "src_hash", "hpo_protocol_hash", "variante", "n_trials", "seed"):
            if m.get(key) != ref.get(key):
                raise RuntimeError(
                    f"Incompatible runs for merge on '{key}': {m.get(key)!r} != {ref.get(key)!r}"
                )

    seen_units: set[str] = set()
    merged_units: list[WorkUnitSpec] = []
    unit_sources: dict[str, HorizonStorageLayout] = {}

    for d in run_dirs:
        rp = Path(d)
        st = HorizonStorageLayout(base_dir=rp.parent, run_id=rp.name)
        for u in st.load_inventory():
            if u.unit_id in seen_units:
                raise RuntimeError(f"Overlapping unit_id across disjoint runs: {u.unit_id}")
            seen_units.add(u.unit_id)
            merged_units.append(u)
            unit_sources[u.unit_id] = st

    merged_dir = Path(out_dir)
    merged_storage = HorizonStorageLayout(
        base_dir=merged_dir.parent,
        run_id=merged_dir.name,
        config_hash=ref["config_hash"],
        src_hash=ref["src_hash"],
        hpo_protocol_hash=ref["hpo_protocol_hash"],
        hpo_protocol_version=ref.get("hpo_protocol_version", HPO_PROTOCOL_VERSION),
        variante=ref["variante"],
        n_trials=ref["n_trials"],
        seed=ref["seed"],
        shard_index=0,
        num_shards=1,
    )
    merged_storage.save_inventory(merged_units)

    for u in merged_units:
        src_st = unit_sources[u.unit_id]
        rec = src_st.get_unit_record(u.unit_id)
        if rec.get("status") != "COMPLETED" or not src_st.is_unit_completed(u):
            continue
        df_p = pd.read_parquet(rec["artifact_path"])
        df_t = pd.read_csv(rec["trials_path"]) if rec.get("trials_path") else None
        merged_storage.persist_unit_outputs(u, df_p, df_t)
        del df_p, df_t

    return consolidate_parallel_run(merged_storage.run_dir, out_dir=merged_storage.run_dir / "consolidated", root=repo_root)

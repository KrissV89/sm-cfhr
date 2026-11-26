"""Horizon-organized incremental storage, atomic ledger, lock recovery, and identity verification."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import psutil

try:
    from sm_cfhr.persistence import (
        VALID_STATES,
        atomic_write_csv,
        atomic_write_parquet,
        atomic_write_text,
        get_process_rss_mb,
        sha256_file,
    )
except ImportError:
    from src.sm_cfhr.persistence import (
        VALID_STATES,
        atomic_write_csv,
        atomic_write_parquet,
        atomic_write_text,
        get_process_rss_mb,
        sha256_file,
    )

from .work_units import HORIZON_BUCKETS, WorkUnitSpec


def _safe_id(unit_id: str) -> str:
    return unit_id.replace(":", "__").replace("/", "_").replace("\\", "_")


class HorizonStorageLayout:
    """Isolated, horizon-organized incremental storage for parallel OOF runs."""

    def __init__(
        self,
        base_dir: Path,
        run_id: str,
        config_hash: str | None = None,
        src_hash: str | None = None,
        hpo_protocol_hash: str | None = None,
        hpo_protocol_version: str | None = None,
        variante: str | None = None,
        n_trials: int | None = None,
        seed: int | None = None,
        shard_index: int = 0,
        num_shards: int = 1,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.run_id = run_id
        self.run_dir = self.base_dir / run_id
        self.ledger_dir = self.run_dir / "ledger"
        self.units_dir = self.ledger_dir / "units"
        self.locks_dir = self.ledger_dir / "locks"
        self.events_path = self.ledger_dir / "events.jsonl"
        self.progress_path = self.ledger_dir / "progress.json"
        self.manifest_path = self.run_dir / "manifest_run.json"
        self.inventory_path = self.run_dir / "inventory.json"
        self.optuna_dir = self.run_dir / "optuna"
        self.consolidated_dir = self.run_dir / "consolidated"

        self.config_hash = config_hash
        self.src_hash = src_hash
        self.hpo_protocol_hash = hpo_protocol_hash
        self.hpo_protocol_version = hpo_protocol_version
        self.variante = variante
        self.n_trials = int(n_trials) if n_trials is not None else None
        self.seed = int(seed) if seed is not None else None
        self.shard_index = int(shard_index)
        self.num_shards = int(num_shards)

        self._ensure_dirs()
        if config_hash is not None and src_hash is not None:
            self.verify_or_init_manifest()

    def _ensure_dirs(self) -> None:
        for p in (
            self.run_dir,
            self.units_dir,
            self.locks_dir,
            self.optuna_dir,
            self.consolidated_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)
        for h_bkt in HORIZON_BUCKETS:
            (self.run_dir / h_bkt).mkdir(parents=True, exist_ok=True)

    def verify_or_init_manifest(self) -> dict[str, Any]:
        """Ensure a resumed run matches exact code, config, protocol, seed, and shard identity."""
        if self.manifest_path.exists():
            m = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            checks = {
                "config_hash": self.config_hash,
                "src_hash": self.src_hash,
                "hpo_protocol_hash": self.hpo_protocol_hash,
                "hpo_protocol_version": self.hpo_protocol_version,
                "variante": self.variante,
                "n_trials": self.n_trials,
                "seed": self.seed,
                "shard_index": self.shard_index,
                "num_shards": self.num_shards,
            }
            for k, expected in checks.items():
                if expected is not None and m.get(k) != expected:
                    raise RuntimeError(
                        f"Run identity mismatch on '{k}': stored={m.get(k)!r} != current={expected!r}."
                    )
            return m

        m = {
            "run_id": self.run_id,
            "estatus": "INITIALIZED",
            "config_hash": self.config_hash,
            "src_hash": self.src_hash,
            "hpo_protocol_hash": self.hpo_protocol_hash,
            "hpo_protocol_version": self.hpo_protocol_version,
            "variante": self.variante,
            "n_trials": self.n_trials,
            "seed": self.seed,
            "shard_index": self.shard_index,
            "num_shards": self.num_shards,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        atomic_write_text(self.manifest_path, json.dumps(m, indent=2, ensure_ascii=False))
        return m

    def update_manifest_fields(self, updates: dict[str, Any]) -> dict[str, Any]:
        m = json.loads(self.manifest_path.read_text(encoding="utf-8")) if self.manifest_path.exists() else {}
        m.update(updates)
        m["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        atomic_write_text(self.manifest_path, json.dumps(m, indent=2, ensure_ascii=False))
        return m

    def save_inventory(self, units: list[WorkUnitSpec]) -> None:
        payload = {
            "run_id": self.run_id,
            "n_units": len(units),
            "units": [u.to_dict() for u in units],
        }
        atomic_write_text(self.inventory_path, json.dumps(payload, indent=2, ensure_ascii=False))

    def load_inventory(self) -> list[WorkUnitSpec]:
        if not self.inventory_path.exists():
            return []
        payload = json.loads(self.inventory_path.read_text(encoding="utf-8"))
        return [WorkUnitSpec.from_dict(u) for u in payload.get("units", [])]

    def unit_scenario_dir(self, unit: WorkUnitSpec) -> Path:
        return self.run_dir / unit.horizon_bucket / unit.scenario_id / unit.variant

    def unit_pred_path(self, unit: WorkUnitSpec) -> Path:
        return self.unit_scenario_dir(unit) / "fragments" / f"{_safe_id(unit.unit_tag)}.parquet"

    def unit_trials_path(self, unit: WorkUnitSpec) -> Path:
        return self.unit_scenario_dir(unit) / "hpo_trials" / f"{_safe_id(unit.unit_tag)}.csv"

    def unit_optuna_db_path(self, unit: WorkUnitSpec) -> Path:
        return self.optuna_dir / unit.horizon_bucket / unit.scenario_id / f"{_safe_id(unit.unit_tag)}.db"

    def unit_optuna_storage_url(self, unit: WorkUnitSpec) -> str:
        db_p = self.unit_optuna_db_path(unit)
        db_p.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_p.as_posix()}"

    def unit_record_path(self, unit_id: str) -> Path:
        return self.units_dir / f"{_safe_id(unit_id)}.json"

    def unit_lock_path(self, unit_id: str) -> Path:
        return self.locks_dir / f"{_safe_id(unit_id)}.lock"

    def get_unit_record(self, unit_id: str) -> dict[str, Any]:
        p = self.unit_record_path(unit_id)
        if not p.exists():
            return {"unit_id": unit_id, "status": "PENDING"}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"unit_id": unit_id, "status": "INVALID", "error": "Corrupt unit record JSON"}

    def is_unit_completed(self, unit: WorkUnitSpec) -> bool:
        rec = self.get_unit_record(unit.unit_id)
        if rec.get("status") != "COMPLETED":
            return False
        pred_p = Path(rec.get("artifact_path", ""))
        exp_sha = rec.get("artifact_sha256")
        if not pred_p.exists() or not exp_sha:
            return False
        if sha256_file(pred_p) != exp_sha:
            self.write_unit_record(unit.unit_id, "INVALID", error="Prediction fragment SHA-256 mismatch")
            return False
        if unit.variant == "hpo":
            t_sha = rec.get("trials_sha256")
            t_p_str = rec.get("trials_path")
            if t_p_str:
                t_p = Path(t_p_str)
                if not t_p.exists() or (t_sha and sha256_file(t_p) != t_sha):
                    self.write_unit_record(unit.unit_id, "INVALID", error="Trials CSV SHA-256 mismatch")
                    return False
        return True

    def recover_orphaned_locks(self) -> list[str]:
        """Detect stale lock files whose owning PID no longer exists and mark units INTERRUPTED."""
        recovered: list[str] = []
        for lock_p in self.locks_dir.glob("*.lock"):
            owner_pid: int | None = None
            try:
                txt = lock_p.read_text(encoding="utf-8", errors="ignore")
                for line in txt.splitlines():
                    if line.startswith("pid="):
                        owner_pid = int(line.split("=", 1)[1].strip())
                        break
            except (OSError, ValueError):
                owner_pid = None

            alive = bool(owner_pid is not None and owner_pid != os.getpid() and psutil.pid_exists(owner_pid))
            if not alive:
                unit_id = lock_p.stem
                try:
                    lock_p.unlink()
                except OSError:
                    continue
                rec = self.get_unit_record(unit_id)
                if rec.get("status") == "RUNNING":
                    self.write_unit_record(
                        unit_id,
                        "INTERRUPTED",
                        error=f"Recovered orphaned lock (stale pid={owner_pid})",
                    )
                recovered.append(unit_id)
        return recovered

    def write_unit_record(
        self,
        unit_id: str,
        status: str,
        artifact_path: Path | None = None,
        artifact_sha256: str | None = None,
        trials_path: Path | None = None,
        trials_sha256: str | None = None,
        n_rows: int | None = None,
        error: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if status not in VALID_STATES:
            raise ValueError(f"Invalid status: {status}")
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "unit_id": unit_id,
            "status": status,
            "pid": os.getpid(),
            "rss_mb": round(get_process_rss_mb(), 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "artifact_path": str(artifact_path) if artifact_path is not None else None,
            "artifact_sha256": artifact_sha256,
            "trials_path": str(trials_path) if trials_path is not None else None,
            "trials_sha256": trials_sha256,
            "n_rows": int(n_rows) if n_rows is not None else None,
            "error": error,
        }
        if metrics:
            payload.update(metrics)
        atomic_write_text(self.unit_record_path(unit_id), json.dumps(payload, indent=2, ensure_ascii=False))
        return payload

    def persist_unit_outputs(
        self,
        unit: WorkUnitSpec,
        pred_df: pd.DataFrame,
        trials_df: pd.DataFrame | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Atomically persist unit predictions and optional HPO trials, then commit COMPLETED record."""
        pred_p = self.unit_pred_path(unit)
        pred_sha = atomic_write_parquet(pred_df, pred_p)

        t_p: Path | None = None
        t_sha: str | None = None
        if trials_df is not None and not trials_df.empty:
            t_p = self.unit_trials_path(unit)
            t_sha = atomic_write_csv(trials_df, t_p)

        return self.write_unit_record(
            unit_id=unit.unit_id,
            status="COMPLETED",
            artifact_path=pred_p,
            artifact_sha256=pred_sha,
            trials_path=t_p,
            trials_sha256=t_sha,
            n_rows=len(pred_df),
            metrics=metrics,
        )

    def append_event(self, event: dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def write_progress(self, progress: dict[str, Any]) -> None:
        atomic_write_text(self.progress_path, json.dumps(progress, indent=2, ensure_ascii=False))

"""Incremental persistence, atomic writes, exclusive work-unit locks, checkpoints, and RSS memory guard."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import psutil

VALID_STATES = ("PENDING", "RUNNING", "COMPLETED", "FAILED", "INTERRUPTED", "INVALID")

RUN_SUBDIRS = (
    "checkpoints",
    "logs",
    "optuna",
    "predicciones",
    "predicciones/fragments",
    "folds",
    "hiperparametros",
    "metricas",
    "inferencia",
    "multiplicidad",
    "ablaciones",
    "sensibilidad",
    "robustez",
    "incertidumbre",
    "interpretabilidad",
    "coste",
    "tablas",
    "figuras",
    "reproducibilidad",
)


def get_process_rss_mb() -> float:
    """Return real process Resident Set Size (RSS) in megabytes."""
    proc = psutil.Process(os.getpid())
    return float(proc.memory_info().rss / (1024.0 * 1024.0))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> str:
    """Write bytes atomically with flush, fsync, checksum, and os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return sha256_bytes(data)


def atomic_write_text(path: Path, text: str) -> str:
    """Write UTF-8 text atomically and return its SHA-256."""
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_csv(df: pd.DataFrame, path: Path) -> str:
    """Write DataFrame to CSV atomically and return its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return sha256_file(path)


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> str:
    """Write DataFrame to Parquet atomically and return its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return sha256_file(path)


class WorkUnitLock:
    """Exclusive file lock preventing two processes from claiming the same work unit."""

    def __init__(self, lock_path: Path) -> None:
        self.lock_path = Path(lock_path)
        self._fd: int | None = None

    def __enter__(self) -> "WorkUnitLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self._fd, f"pid={os.getpid()}\n".encode("utf-8"))
        except FileExistsError as exc:
            raise RuntimeError(f"Work unit already locked by another process: {self.lock_path}") from exc
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        if self.lock_path.exists():
            try:
                self.lock_path.unlink()
            except OSError:
                pass


class RunCheckpointManager:
    """Durable checkpoint, structured logging, identity verification, and RSS guard."""

    def __init__(
        self,
        base_dir: Path,
        run_id: str,
        config_hash: str,
        src_hash: str,
        proto_hash: str,
        max_rss_mb: float = 4096.0,
        total_units: int = 0,
    ) -> None:
        self.run_dir = Path(base_dir) / "runs" / run_id
        self.run_id = run_id
        self.config_hash = config_hash
        self.src_hash = src_hash
        self.proto_hash = proto_hash
        self.max_rss_mb = float(max_rss_mb)
        self.total_units = int(total_units)

        self.run_dir.mkdir(parents=True, exist_ok=True)
        for sub in RUN_SUBDIRS:
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.run_dir / "manifest.json"
        self.log_path = self.run_dir / "logs" / "run.jsonl"
        self.progress_path = self.run_dir / "logs" / "progress.json"
        self._verify_or_init_manifest()

    def _verify_or_init_manifest(self) -> None:
        if self.manifest_path.exists():
            try:
                m = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise RuntimeError(f"Manifiesto de corrida corrupto: {exc}") from exc
            if (
                m.get("config_hash") != self.config_hash
                or m.get("src_hash") != self.src_hash
                or m.get("proto_hash") != self.proto_hash
            ):
                raise RuntimeError(
                    "Incompatibilidad de configuración/código/protocolo al intentar reanudar la corrida."
                )
        else:
            m = {
                "run_id": self.run_id,
                "estatus": "RUNNING",
                "config_hash": self.config_hash,
                "src_hash": self.src_hash,
                "proto_hash": self.proto_hash,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            atomic_write_text(self.manifest_path, json.dumps(m, indent=2, ensure_ascii=False))

    def _ckpt_path(self, unit_id: str) -> Path:
        safe = unit_id.replace(":", "__").replace("/", "_")
        return self.run_dir / "checkpoints" / f"{safe}.json"

    def get_unit_state(self, unit_id: str) -> dict[str, Any]:
        p = self._ckpt_path(unit_id)
        if not p.exists():
            return {"unit_id": unit_id, "status": "PENDING"}
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"unit_id": unit_id, "status": "INVALID", "error": "Corrupt checkpoint JSON"}
        return data

    def is_unit_completed(self, unit_id: str, expected_rows: int | None = None) -> bool:
        st = self.get_unit_state(unit_id)
        if st.get("status") != "COMPLETED":
            return False
        art = st.get("artifact_path")
        exp_sha = st.get("artifact_sha256")
        if not art or not exp_sha:
            return False
        art_p = Path(art)
        if not art_p.exists():
            return False
        if sha256_file(art_p) != exp_sha:
            self.mark_unit(unit_id, "INVALID", error="Artifact SHA-256 mismatch")
            return False
        if expected_rows is not None and st.get("n_rows") != int(expected_rows):
            self.mark_unit(unit_id, "INVALID", error="Row count mismatch")
            return False
        return True

    def mark_unit(
        self,
        unit_id: str,
        status: str,
        artifact_path: Path | None = None,
        artifact_sha256: str | None = None,
        n_rows: int | None = None,
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if status not in VALID_STATES:
            raise ValueError(f"Invalid checkpoint status: {status}")
        rss = get_process_rss_mb()
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "unit_id": unit_id,
            "status": status,
            "config_hash": self.config_hash,
            "src_hash": self.src_hash,
            "proto_hash": self.proto_hash,
            "artifact_path": str(artifact_path) if artifact_path is not None else None,
            "artifact_sha256": artifact_sha256,
            "n_rows": int(n_rows) if n_rows is not None else None,
            "rss_mb": round(rss, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "error": error,
        }
        if extra:
            payload.update(extra)
        atomic_write_text(self._ckpt_path(unit_id), json.dumps(payload, indent=2, ensure_ascii=False))
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._update_progress()

    def _update_progress(self) -> None:
        ckpts = list((self.run_dir / "checkpoints").glob("*.json"))
        counts = {s: 0 for s in VALID_STATES}
        for c in ckpts:
            try:
                st = json.loads(c.read_text(encoding="utf-8")).get("status", "INVALID")
            except (json.JSONDecodeError, OSError):
                st = "INVALID"
            counts[st] = counts.get(st, 0) + 1
        total = max(self.total_units, len(ckpts), 1)
        prog = {
            "run_id": self.run_id,
            "completed_units": counts["COMPLETED"],
            "total_units": total,
            "progress_pct": round(100.0 * counts["COMPLETED"] / total, 2),
            "counts": counts,
            "rss_mb": round(get_process_rss_mb(), 2),
            "max_rss_mb": self.max_rss_mb,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        atomic_write_text(self.progress_path, json.dumps(prog, indent=2, ensure_ascii=False))

    def check_memory_guard(self) -> float:
        rss = get_process_rss_mb()
        if rss > self.max_rss_mb:
            raise MemoryError(
                f"Proceso superó el presupuesto de RAM RSS ({rss:.1f} MB > {self.max_rss_mb:.1f} MB)."
            )
        return rss

    def finalize_manifest(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        m = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        m["estatus"] = "COMPLETED"
        m["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        m["peak_rss_mb_end"] = round(get_process_rss_mb(), 2)
        if extra:
            m.update(extra)
        atomic_write_text(self.manifest_path, json.dumps(m, indent=2, ensure_ascii=False))
        return m


__all__ = [
    "RUN_SUBDIRS",
    "RunCheckpointManager",
    "VALID_STATES",
    "WorkUnitLock",
    "atomic_write_bytes",
    "atomic_write_csv",
    "atomic_write_parquet",
    "atomic_write_text",
    "get_process_rss_mb",
    "sha256_bytes",
    "sha256_file",
]

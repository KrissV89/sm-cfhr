"""Windows-compatible spawn multiprocessing coordinator with zero result accumulation in RAM."""

from __future__ import annotations

import multiprocessing as mp
import queue as py_queue
import signal
import time
from pathlib import Path
from typing import Any

import psutil

try:
    from ashr_oof_pipeline.config import find_project_root, load_frozen_config, sha256_file
    from ashr_oof_pipeline.optimizacion_hiperparametros import (
        HPO_PROTOCOL_VERSION,
        compute_hpo_protocol_hash,
    )
    from sm_cfhr.persistence import get_process_rss_mb
except ImportError:
    from src.ashr_oof_pipeline.config import find_project_root, load_frozen_config, sha256_file
    from src.ashr_oof_pipeline.optimizacion_hiperparametros import (
        HPO_PROTOCOL_VERSION,
        compute_hpo_protocol_hash,
    )
    from src.sm_cfhr.persistence import get_process_rss_mb

from .storage import HorizonStorageLayout
from .work_units import WorkUnitSpec
from .worker import worker_process_loop


class ParallelOOFCoordinator:
    """Distribute independent work units across recyclable spawn workers with strict RSS control."""

    def __init__(
        self,
        out_dir: Path,
        run_id: str,
        repo_root: Path | None = None,
        data_root: Path | None = None,
        n_workers: int = 4,
        max_tasks_per_worker: int = 60,
        worker_max_rss_mb: float = 1536.0,
        variante: str = "fixed",
        n_trials: int = 25,
        seed: int = 2026,
        shard_index: int = 0,
        num_shards: int = 1,
    ) -> None:
        self.repo_root = Path(repo_root or find_project_root())
        self.data_root = Path(data_root or self.repo_root)
        self.n_workers = max(1, int(n_workers))
        self.max_tasks_per_worker = int(max_tasks_per_worker)
        self.worker_max_rss_mb = float(worker_max_rss_mb)
        self.variante = variante
        self.n_trials = int(n_trials)
        self.seed = int(seed)

        cfg = load_frozen_config(self.repo_root)
        self.expected_src_hash = cfg["sha256_src_adaptive_structured_huber"]
        self.src_hash = sha256_file(self.repo_root / "src" / "adaptive_structured_huber.py")
        if self.src_hash != self.expected_src_hash:
            raise RuntimeError(
                f"SHA-256 mismatch in src/adaptive_structured_huber.py: {self.src_hash} != {self.expected_src_hash}"
            )
        self.cfg_hash = sha256_file(self.repo_root / "experiments" / "config_ashr_congelada.json")
        self.hpo_proto_hash = compute_hpo_protocol_hash(self.repo_root)

        self.storage = HorizonStorageLayout(
            base_dir=Path(out_dir),
            run_id=run_id,
            config_hash=self.cfg_hash,
            src_hash=self.src_hash,
            hpo_protocol_hash=self.hpo_proto_hash,
            hpo_protocol_version=HPO_PROTOCOL_VERSION,
            variante=variante,
            n_trials=n_trials,
            seed=seed,
            shard_index=shard_index,
            num_shards=num_shards,
        )

    def _spawn_worker(
        self,
        ctx: mp.context.SpawnContext,
        slot: int,
        task_q: Any,
        result_q: Any,
    ) -> mp.Process:
        p = ctx.Process(
            target=worker_process_loop,
            args=(
                slot,
                task_q,
                result_q,
                str(self.storage.base_dir),
                self.storage.run_id,
                str(self.repo_root),
                str(self.data_root),
                self.hpo_proto_hash,
                self.expected_src_hash,
                self.max_tasks_per_worker,
                self.worker_max_rss_mb,
            ),
            daemon=False,
        )
        p.start()
        return p

    def _measure_cluster_rss_mb(self, procs: dict[int, mp.Process]) -> tuple[float, float, dict[int, float]]:
        coord_rss = get_process_rss_mb()
        worker_rss: dict[int, float] = {}
        total_workers = 0.0
        for slot, proc in procs.items():
            if proc.is_alive() and proc.pid is not None:
                try:
                    r = psutil.Process(proc.pid).memory_info().rss / (1024.0 * 1024.0)
                    worker_rss[slot] = round(r, 2)
                    total_workers += r
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        return round(coord_rss, 2), round(coord_rss + total_workers, 2), worker_rss

    def run(self, units: list[WorkUnitSpec], fail_fast: bool = False) -> dict[str, Any]:
        """Execute pending work units dynamically across N recyclable workers."""
        t0 = time.perf_counter()
        recovered_locks = self.storage.recover_orphaned_locks()
        self.storage.save_inventory(units)

        pending: list[WorkUnitSpec] = []
        completed_init = 0
        failed_init = 0
        for u in units:
            if self.storage.is_unit_completed(u):
                completed_init += 1
            else:
                st = self.storage.get_unit_record(u.unit_id).get("status", "PENDING")
                if st == "FAILED":
                    failed_init += 1
                pending.append(u)

        # Group pending units by scenario so each worker reuses its 1-scenario cache maximally
        pending.sort(key=lambda u: (u.variant, u.horizon_days, u.scenario_id, u.unit_tag))

        total_units = len(units)
        if not pending:
            summary = {
                "run_id": self.storage.run_id,
                "estatus": "ALL_UNITS_ALREADY_COMPLETED" if failed_init == 0 else "NO_PENDING_WITH_FAILURES",
                "total_units": total_units,
                "completed": completed_init,
                "executed_now": 0,
                "failed": 0,
                "recovered_orphaned_locks": len(recovered_locks),
                "n_workers": self.n_workers,
                "elapsed_s": round(time.perf_counter() - t0, 3),
                "peak_coord_rss_mb": round(get_process_rss_mb(), 2),
                "peak_aggregate_rss_mb": round(get_process_rss_mb(), 2),
                "worker_recycles": 0,
            }
            self.storage.update_manifest_fields(
                {"estatus": "COMPLETED" if completed_init == total_units else "PARTIAL", **summary}
            )
            return summary

        ctx = mp.get_context("spawn")
        n_active_workers = min(self.n_workers, len(pending))
        task_queues: dict[int, Any] = {s: ctx.Queue(maxsize=2) for s in range(n_active_workers)}
        result_q: Any = ctx.Queue()
        procs: dict[int, mp.Process] = {}
        inflight: dict[int, str] = {}

        stop_requested = False
        prev_sigint = signal.getsignal(signal.SIGINT)

        def _handle_sigint(signum: int, frame: Any) -> None:
            nonlocal stop_requested
            stop_requested = True

        signal.signal(signal.SIGINT, _handle_sigint)

        completed_now = 0
        failed_now = 0
        recycles = 0
        spawn_count = 0
        spawn_overhead_s = 0.0
        load_time_sum = 0.0
        compute_time_sum = 0.0
        write_time_sum = 0.0
        peak_coord_rss = get_process_rss_mb()
        peak_agg_rss = peak_coord_rss
        peak_worker_rss = 0.0
        errors: list[dict[str, str]] = []

        next_idx = 0
        cpu_t0 = time.process_time()

        try:
            self.storage.update_manifest_fields({"estatus": "RUNNING", "n_workers": n_active_workers})
            for slot in range(n_active_workers):
                ts = time.perf_counter()
                procs[slot] = self._spawn_worker(ctx, slot, task_queues[slot], result_q)
                spawn_overhead_s += time.perf_counter() - ts
                spawn_count += 1
                if next_idx < len(pending):
                    u = pending[next_idx]
                    next_idx += 1
                    inflight[slot] = u.unit_id
                    task_queues[slot].put(u.to_dict())

            while inflight or any(p.is_alive() for p in procs.values()):
                coord_rss, agg_rss, w_rss = self._measure_cluster_rss_mb(procs)
                peak_coord_rss = max(peak_coord_rss, coord_rss)
                peak_agg_rss = max(peak_agg_rss, agg_rss)
                if w_rss:
                    peak_worker_rss = max(peak_worker_rss, max(w_rss.values()))

                if stop_requested and not inflight:
                    break

                try:
                    msg = result_q.get(timeout=0.5)
                except py_queue.Empty:
                    # Check for unexpected worker crashes
                    for slot, proc in list(procs.items()):
                        if slot in inflight and not proc.is_alive() and proc.exitcode not in (0, None):
                            crashed_uid = inflight.pop(slot)
                            failed_now += 1
                            err = f"Worker slot {slot} crashed (exitcode={proc.exitcode}) on {crashed_uid}"
                            errors.append({"unit_id": crashed_uid, "error": err})
                            self.storage.write_unit_record(crashed_uid, "FAILED", error=err)
                            self.storage.append_event({"event": "WORKER_CRASH", "unit_id": crashed_uid, "error": err})
                    if not inflight and (stop_requested or next_idx >= len(pending)):
                        break
                    continue

                kind = msg.get("kind")
                slot = int(msg["worker_slot"])

                if kind == "WORKER_EXIT":
                    procs[slot].join(timeout=10.0)
                    continue

                uid = msg["unit_id"]
                inflight.pop(slot, None)
                self.storage.append_event(msg)

                if kind == "UNIT_DONE":
                    completed_now += 1
                    load_time_sum += float(msg.get("load_time_s", 0.0))
                    compute_time_sum += float(msg.get("compute_time_s", 0.0))
                    write_time_sum += float(msg.get("write_time_s", 0.0))
                    peak_worker_rss = max(peak_worker_rss, float(msg.get("rss_after_mb", 0.0)))
                elif kind == "UNIT_FAILED":
                    failed_now += 1
                    errors.append({"unit_id": uid, "error": str(msg.get("error", "unknown"))})
                    if fail_fast:
                        stop_requested = True

                self.storage.write_progress(
                    {
                        "run_id": self.storage.run_id,
                        "total_units": total_units,
                        "completed": completed_init + completed_now,
                        "failed": failed_now,
                        "pending": max(0, len(pending) - next_idx),
                        "inflight": len(inflight),
                        "peak_coord_rss_mb": peak_coord_rss,
                        "peak_aggregate_rss_mb": peak_agg_rss,
                        "peak_worker_rss_mb": round(peak_worker_rss, 2),
                        "worker_recycles": recycles,
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                )

                recycled = bool(msg.get("recycle_requested", False))
                if recycled:
                    # Wait for old worker process to exit completely so OS frees its RSS before spawning replacement
                    procs[slot].join(timeout=15.0)
                    recycles += 1
                    if not stop_requested and next_idx < len(pending):
                        ts = time.perf_counter()
                        procs[slot] = self._spawn_worker(ctx, slot, task_queues[slot], result_q)
                        spawn_overhead_s += time.perf_counter() - ts
                        spawn_count += 1

                if not stop_requested and next_idx < len(pending):
                    u_next = pending[next_idx]
                    next_idx += 1
                    inflight[slot] = u_next.unit_id
                    task_queues[slot].put(u_next.to_dict())
                elif not recycled and procs[slot].is_alive():
                    task_queues[slot].put(None)
        finally:
            signal.signal(signal.SIGINT, prev_sigint)
            for slot, proc in procs.items():
                if proc.is_alive():
                    try:
                        task_queues[slot].put_nowait(None)
                    except Exception:
                        pass
                    proc.join(timeout=5.0)
                    if proc.is_alive():
                        proc.terminate()
                        proc.join(timeout=2.0)

        elapsed_s = max(time.perf_counter() - t0, 1e-6)
        cpu_s = time.process_time() - cpu_t0
        total_done = completed_init + completed_now
        status = (
            "INTERRUPTED"
            if stop_requested
            else ("COMPLETED" if (total_done == total_units and failed_now == 0) else "PARTIAL")
        )
        summary = {
            "run_id": self.storage.run_id,
            "estatus": status,
            "n_workers": n_active_workers,
            "total_units": total_units,
            "completed_before": completed_init,
            "executed_now": completed_now,
            "completed_total": total_done,
            "failed": failed_now,
            "recovered_orphaned_locks": len(recovered_locks),
            "elapsed_s": round(elapsed_s, 3),
            "units_per_s": round(completed_now / elapsed_s, 4),
            "load_time_sum_s": round(load_time_sum, 3),
            "compute_time_sum_s": round(compute_time_sum, 3),
            "write_time_sum_s": round(write_time_sum, 3),
            "spawn_count": spawn_count,
            "spawn_overhead_s": round(spawn_overhead_s, 3),
            "worker_recycles": recycles,
            "coordinator_cpu_s": round(cpu_s, 3),
            "peak_coord_rss_mb": round(peak_coord_rss, 2),
            "peak_worker_rss_mb": round(peak_worker_rss, 2),
            "peak_aggregate_rss_mb": round(peak_agg_rss, 2),
            "errors": errors,
        }
        self.storage.update_manifest_fields(summary)
        return summary

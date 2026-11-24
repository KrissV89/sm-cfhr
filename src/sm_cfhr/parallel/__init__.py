"""Parallel multiprocessing execution, horizon-organized persistence, and consolidation for SM-CFHR."""

from __future__ import annotations

from .consolidation import consolidate_parallel_run, merge_disjoint_runs
from .coordinator import ParallelOOFCoordinator
from .storage import HorizonStorageLayout
from .work_units import HORIZON_BUCKETS, WorkUnitSpec, build_work_inventory, horizon_bucket

__all__ = [
    "HORIZON_BUCKETS",
    "HorizonStorageLayout",
    "ParallelOOFCoordinator",
    "WorkUnitSpec",
    "build_work_inventory",
    "consolidate_parallel_run",
    "horizon_bucket",
    "merge_disjoint_runs",
]

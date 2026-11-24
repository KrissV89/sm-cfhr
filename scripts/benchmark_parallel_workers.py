"""Controlled benchmark comparing 1, 2, 4, 6, and 8 spawn workers on Ryzen 7 8845HS."""

from __future__ import annotations

import os

for _env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
):
    os.environ[_env] = "1"

import argparse
import json
import multiprocessing as mp
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sm_cfhr.parallel import ParallelOOFCoordinator, build_work_inventory  # noqa: E402
from tests.test_parallel_oof import _create_synthetic_data_root  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark 1, 2, 4, 6, 8 spawn workers for SM-CFHR.")
    parser.add_argument("--workers-list", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    parser.add_argument("--n-trials", type=int, default=6)
    parser.add_argument("--max-units", type=int, default=16)
    args = parser.parse_args()

    tmp_base = Path(tempfile.mkdtemp(prefix="smcfhr_bench_"))
    try:
        data_root = _create_synthetic_data_root(tmp_base / "data")
        scenarios = [
            "D1_espaciotemporal_cinco_variables",
            "D1_intraplanta_cinco_variables",
            "RecSinClima_intraplanta_7d",
            "RecGEFSv12_espaciotemporal_7d",
        ]
        all_units = build_work_inventory(
            root=ROOT,
            data_root=data_root,
            variante="hpo",
            n_trials=args.n_trials,
            seed=2026,
            scenarios=scenarios,
        )
        bench_units = all_units[: args.max_units]

        rows = []
        t_seq = None
        for nw in args.workers_list:
            out_dir = tmp_base / "runs"
            run_id = f"bench_w{nw}"
            if (out_dir / run_id).exists():
                shutil.rmtree(out_dir / run_id)

            coord = ParallelOOFCoordinator(
                out_dir=out_dir,
                run_id=run_id,
                repo_root=ROOT,
                data_root=data_root,
                n_workers=nw,
                max_tasks_per_worker=8,
                worker_max_rss_mb=1536.0,
                variante="hpo",
                n_trials=args.n_trials,
                seed=2026,
            )
            t0 = time.perf_counter()
            summary = coord.run(bench_units)
            wall_s = time.perf_counter() - t0
            if t_seq is None:
                t_seq = wall_s
            speedup = t_seq / max(wall_s, 1e-6)
            rows.append(
                {
                    "workers": nw,
                    "n_units": len(bench_units),
                    "n_trials_per_unit": args.n_trials,
                    "wall_time_s": round(wall_s, 3),
                    "units_per_s": summary["units_per_s"],
                    "speedup_vs_1w": round(speedup, 3),
                    "efficiency_pct": round(100.0 * speedup / nw, 1),
                    "peak_coord_rss_mb": summary["peak_coord_rss_mb"],
                    "peak_worker_rss_mb": summary["peak_worker_rss_mb"],
                    "peak_aggregate_rss_mb": summary["peak_aggregate_rss_mb"],
                    "load_time_sum_s": summary["load_time_sum_s"],
                    "compute_time_sum_s": summary["compute_time_sum_s"],
                    "write_time_sum_s": summary["write_time_sum_s"],
                    "spawn_overhead_s": summary["spawn_overhead_s"],
                    "worker_recycles": summary["worker_recycles"],
                    "errors": len(summary["errors"]),
                }
            )
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

        out_bench = ROOT / "results" / "benchmark_paralelo"
        out_bench.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_bench / "benchmark_workers_ryzen8845hs.csv", index=False)
        (out_bench / "benchmark_workers_ryzen8845hs.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

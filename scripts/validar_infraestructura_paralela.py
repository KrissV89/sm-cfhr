"""Final local validation of parallel OOF infrastructure on real D1-D3 development data (25 trials)."""

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

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ashr_oof_pipeline.config import SCENARIOS_28, load_frozen_config  # noqa: E402
from ashr_oof_pipeline.oof_runner import align_predictions, load_official_benchmarks  # noqa: E402
from ashr_oof_pipeline.resultados_loader import (  # noqa: E402
    validate_oof_artifacts,
    validate_predictions_df,
)
from sm_cfhr.persistence import get_process_rss_mb  # noqa: E402
from sm_cfhr.parallel import (  # noqa: E402
    ParallelOOFCoordinator,
    WorkUnitSpec,
    build_work_inventory,
    consolidate_parallel_run,
)

DEV_START = "2025-11-01"
DEV_END = "2026-07-31"
TARGET_SCENARIOS = [
    "D1_espaciotemporal_cinco_variables",
    "DirectoSinClima_espaciotemporal_14d",
    "RecSinClima_intraplanta_14d",
    "RecGEFSv12_espaciotemporal_14d",
]


def _select_dev_units(data_root: Path, n_per_scenario: int = 4, n_trials: int = 25) -> list[WorkUnitSpec]:
    all_units = build_work_inventory(
        root=ROOT,
        data_root=data_root,
        variante="hpo",
        n_trials=n_trials,
        seed=2026,
        scenarios=TARGET_SCENARIOS,
    )
    selected: list[WorkUnitSpec] = []
    for sc in TARGET_SCENARIOS:
        sc_units = [u for u in all_units if u.scenario_id == sc]
        if sc.startswith("D1_"):
            dev_units = [u for u in sc_units if u.block_or_origin == "D1"]
        else:
            dev_units = [u for u in sc_units if DEV_START <= u.block_or_origin <= DEV_END]
        if len(dev_units) < n_per_scenario:
            raise RuntimeError(f"Insufficient dev units for {sc}: found {len(dev_units)}")
        # Evenly spaced across development range
        idx = np.linspace(0, len(dev_units) - 1, num=n_per_scenario, dtype=int)
        for i in idx:
            selected.append(dev_units[int(i)])
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate parallel OOF on 25-trial real dev units.")
    parser.add_argument("--data-root", type=Path, required=True, help="Root containing real solar dev data.")
    parser.add_argument("--workers-list", type=int, nargs="+", default=[4, 6, 8])
    args = parser.parse_args()

    out_val = ROOT / "results" / "validacion_final_paralelo"
    out_val.mkdir(parents=True, exist_ok=True)
    tmp_base = Path(tempfile.mkdtemp(prefix="smcfhr_val25_"))

    try:
        units_16 = _select_dev_units(args.data_root, n_per_scenario=4, n_trials=25)
        units_4 = [next(u for u in units_16 if u.scenario_id == sc) for sc in TARGET_SCENARIOS]

        # ------------------------------------------------------------------
        # 1. Representative 25-trial benchmark: 4, 6, 8 workers with recycling
        # ------------------------------------------------------------------
        bench_rows = []
        for nw in args.workers_list:
            runs_dir = tmp_base / f"bench_w{nw}"
            psutil.cpu_percent(interval=None)
            rss_start = get_process_rss_mb()
            coord = ParallelOOFCoordinator(
                out_dir=runs_dir,
                run_id=f"val_w{nw}",
                repo_root=ROOT,
                data_root=args.data_root,
                n_workers=nw,
                max_tasks_per_worker=2,
                worker_max_rss_mb=1536.0,
                variante="hpo",
                n_trials=25,
                seed=2026,
            )
            t0 = time.perf_counter()
            summary = coord.run(units_16)
            wall_s = time.perf_counter() - t0
            cpu_pct = psutil.cpu_percent(interval=None)
            rss_end = get_process_rss_mb()

            bench_rows.append(
                {
                    "workers": nw,
                    "n_units": len(units_16),
                    "n_trials": 25,
                    "wall_time_s": round(wall_s, 3),
                    "units_per_s": summary["units_per_s"],
                    "cpu_percent_sys": round(cpu_pct, 1),
                    "rss_coord_start_mb": round(rss_start, 2),
                    "rss_coord_end_mb": round(rss_end, 2),
                    "peak_coord_rss_mb": summary["peak_coord_rss_mb"],
                    "peak_worker_rss_mb": summary["peak_worker_rss_mb"],
                    "peak_aggregate_rss_mb": summary["peak_aggregate_rss_mb"],
                    "worker_recycles": summary["worker_recycles"],
                    "load_time_sum_s": summary["load_time_sum_s"],
                    "compute_time_sum_s": summary["compute_time_sum_s"],
                    "write_time_sum_s": summary["write_time_sum_s"],
                    "spawn_overhead_s": summary["spawn_overhead_s"],
                    "errors": len(summary["errors"]),
                }
            )
            print("[BENCH_25T]", json.dumps(bench_rows[-1], ensure_ascii=False), flush=True)

        pd.DataFrame(bench_rows).to_csv(out_val / "benchmark_25trials_dev.csv", index=False)

        # ------------------------------------------------------------------
        # 2. Scientific equivalence (1w vs 4w) + interruption/resume (25 trials)
        # ------------------------------------------------------------------
        eq_dir = tmp_base / "equiv"
        c_seq = ParallelOOFCoordinator(
            out_dir=eq_dir, run_id="seq_w1", repo_root=ROOT, data_root=args.data_root,
            n_workers=1, max_tasks_per_worker=10, variante="hpo", n_trials=25, seed=2026,
        )
        s_seq = c_seq.run(units_4)

        c_par = ParallelOOFCoordinator(
            out_dir=eq_dir, run_id="par_w4", repo_root=ROOT, data_root=args.data_root,
            n_workers=4, max_tasks_per_worker=1, variante="hpo", n_trials=25, seed=2026,
        )
        s_par = c_par.run(units_4)

        m_seq = consolidate_parallel_run(eq_dir / "seq_w1", root=ROOT)
        m_par = consolidate_parallel_run(eq_dir / "par_w4", root=ROOT)

        # Simulated interruption after 2/4 units + stale lock on unit 3, then resume
        c_res_1 = ParallelOOFCoordinator(
            out_dir=eq_dir, run_id="res_w4", repo_root=ROOT, data_root=args.data_root,
            n_workers=2, variante="hpo", n_trials=25, seed=2026,
        )
        c_res_1.run(units_4[:2])
        stale_lock = c_res_1.storage.unit_lock_path(units_4[2].unit_id)
        stale_lock.write_text("pid=99999999\n", encoding="utf-8")
        c_res_1.storage.write_unit_record(units_4[2].unit_id, "RUNNING")

        c_res_2 = ParallelOOFCoordinator(
            out_dir=eq_dir, run_id="res_w4", repo_root=ROOT, data_root=args.data_root,
            n_workers=4, variante="hpo", n_trials=25, seed=2026,
        )
        s_res = c_res_2.run(units_4)
        m_res = consolidate_parallel_run(eq_dir / "res_w4", root=ROOT)

        sc_map = {sp.scenario_id: sp for sp in SCENARIOS_28}
        cfg = load_frozen_config(ROOT)
        expected_hash = cfg["sha256_src_adaptive_structured_huber"]

        # Contract check on consolidated partial run
        val_chk = validate_oof_artifacts(eq_dir / "par_w4" / "consolidated", expected_hash)
        eq_details = []
        for sc in TARGET_SCENARIOS:
            sp = sc_map[sc]
            p_seq = pd.read_csv(eq_dir / "seq_w1" / "consolidated" / "predicciones" / f"{sc}_hpo.csv")
            p_par = pd.read_csv(eq_dir / "par_w4" / "consolidated" / "predicciones" / f"{sc}_hpo.csv")
            p_res = pd.read_csv(eq_dir / "res_w4" / "consolidated" / "predicciones" / f"{sc}_hpo.csv")

            validate_predictions_df(p_par, sp.join_keys, sp.pred_col, sp.target_col)
            bench_raw = load_official_benchmarks(sp, root=args.data_root, exclude_cf_brh=True)
            aligned = align_predictions(p_par, bench_raw, sp.join_keys, strict=True)

            max_diff_par = float(np.max(np.abs(p_seq[sp.pred_col].to_numpy() - p_par[sp.pred_col].to_numpy())))
            max_diff_res = float(np.max(np.abs(p_seq[sp.pred_col].to_numpy() - p_res[sp.pred_col].to_numpy())))
            mae_seq = float(np.mean(np.abs(p_seq[sp.pred_col] - p_seq[sp.target_col])))
            mae_par = float(np.mean(np.abs(p_par[sp.pred_col] - p_par[sp.target_col])))

            t_seq = pd.read_csv(eq_dir / "seq_w1" / "consolidated" / "hpo_trials" / f"{sc}.csv")
            t_par = pd.read_csv(eq_dir / "par_w4" / "consolidated" / "hpo_trials" / f"{sc}.csv")
            max_trial_diff = float(np.max(np.abs(t_seq["inner_mae_macro_kwh"].to_numpy() - t_par["inner_mae_macro_kwh"].to_numpy())))

            eq_details.append(
                {
                    "scenario_id": sc,
                    "n_trials_seq": len(t_seq),
                    "n_trials_par": len(t_par),
                    "max_pred_diff_1w_vs_4w": max_diff_par,
                    "max_pred_diff_1w_vs_resume": max_diff_res,
                    "max_trial_mae_diff": max_trial_diff,
                    "mae_seq": round(mae_seq, 6),
                    "mae_par": round(mae_par, 6),
                    "benchmark_aligned_rows": len(aligned),
                }
            )

        report = {
            "benchmark_25trials_dev": bench_rows,
            "equivalence_summary": {
                "seq_status": s_seq["estatus"],
                "par_status": s_par["estatus"],
                "resume_status": s_res["estatus"],
                "resume_recovered_orphaned_locks": s_res["recovered_orphaned_locks"],
                "resume_executed_now": s_res["executed_now"],
                "hashes_seq_eq_par": m_seq["hashes_sha256_predicciones"] == m_par["hashes_sha256_predicciones"],
                "hashes_seq_eq_res": m_seq["hashes_sha256_predicciones"] == m_res["hashes_sha256_predicciones"],
                "partial_manifest_state": val_chk["state"],
                "partial_manifest_estatus": m_par["estatus"],
                "partial_manifest_procedencia": m_par["procedencia"],
                "per_scenario": eq_details,
            },
        }
        (out_val / "validacion_final_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("[VALIDACION_OK]", json.dumps(report["equivalence_summary"], indent=2, ensure_ascii=False))
    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

"""Sustained 25-trial SMT benchmark comparing 8, 10, and 12 workers on identical 48 dev units."""

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
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sm_cfhr.parallel import ParallelOOFCoordinator, build_work_inventory, consolidate_parallel_run  # noqa: E402


def _select_smt_dev_units(data_root: Path, n_per_scenario: int = 12, n_trials: int = 25):
    target_scenarios = [
        "D1_espaciotemporal_cinco_variables",
        "DirectoSinClima_espaciotemporal_14d",
        "RecSinClima_intraplanta_14d",
        "RecGEFSv12_espaciotemporal_14d",
    ]
    all_units = build_work_inventory(
        root=ROOT,
        data_root=data_root,
        variante="hpo",
        n_trials=n_trials,
        seed=2026,
        scenarios=target_scenarios,
    )
    selected = []
    for sc in target_scenarios:
        sc_units = [u for u in all_units if u.scenario_id == sc]
        if sc.startswith("D1_"):
            dev_units = [u for u in sc_units if u.block_or_origin in ("D1", "D2", "D3")]
        else:
            dev_units = [u for u in sc_units if "2025-11-01" <= u.block_or_origin <= "2026-07-31"]
        if len(dev_units) < n_per_scenario:
            raise RuntimeError(f"Insufficient dev units for {sc}: found {len(dev_units)}")
        step = max(1, len(dev_units) // n_per_scenario)
        selected.extend(dev_units[i * step] for i in range(n_per_scenario))
    return selected


def _read_cpu_freq_and_temp() -> tuple[float | None, float | None]:
    freq_mhz = None
    try:
        cf = psutil.cpu_freq()
        if cf is not None and cf.current:
            freq_mhz = round(float(cf.current), 1)
    except Exception:
        freq_mhz = None

    temp_c = None
    try:
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "try { (Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature -ErrorAction Stop | Select-Object -First 1 -ExpandProperty CurrentTemperature) } catch { '' }",
            ],
            text=True,
            timeout=3,
        ).strip()
        if out.isdigit():
            temp_c = round((int(out) / 10.0) - 273.15, 1)
    except Exception:
        temp_c = None
    return freq_mhz, temp_c


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare 8, 10, 12 workers with 25-trial sustained load.")
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--workers-list", type=int, nargs="+", default=[8, 10, 12])
    parser.add_argument("--n-per-scenario", type=int, default=12, help="Units per scenario (default 12 => 48 total).")
    args = parser.parse_args()

    units = _select_smt_dev_units(args.data_root, n_per_scenario=args.n_per_scenario, n_trials=25)
    out_dir_res = ROOT / "results" / "benchmark_paralelo"
    out_dir_res.mkdir(parents=True, exist_ok=True)
    tmp_base = Path(tempfile.mkdtemp(prefix="smcfhr_smt_"))

    rows = []
    hashes_by_w: dict[int, dict[str, str]] = {}
    base_upm = None

    try:
        for nw in args.workers_list:
            runs_dir = tmp_base / f"w{nw}"
            psutil.cpu_percent(interval=None)
            coord = ParallelOOFCoordinator(
                out_dir=runs_dir,
                run_id=f"smt_w{nw}",
                repo_root=ROOT,
                data_root=args.data_root,
                n_workers=nw,
                max_tasks_per_worker=12,
                worker_max_rss_mb=1536.0,
                variante="hpo",
                n_trials=25,
                seed=2026,
            )
            t0 = time.perf_counter()
            summary = coord.run(units)
            wall_s = time.perf_counter() - t0
            cpu_pct = psutil.cpu_percent(interval=None)
            freq_mhz, temp_c = _read_cpu_freq_and_temp()

            man = consolidate_parallel_run(runs_dir / f"smt_w{nw}", root=ROOT)
            hashes_by_w[nw] = man["hashes_sha256_predicciones"]

            upm = (len(units) / wall_s) * 60.0
            if nw == args.workers_list[0]:
                base_upm = upm
            gain_pct = 100.0 * (upm / base_upm - 1.0) if base_upm else 0.0

            row = {
                "workers": nw,
                "n_units": len(units),
                "n_trials": 25,
                "wall_time_s": round(wall_s, 3),
                "units_per_min": round(upm, 3),
                "gain_vs_8w_pct": round(gain_pct, 2),
                "cpu_percent_sys": round(cpu_pct, 1),
                "cpu_freq_mhz": freq_mhz,
                "cpu_temp_c": temp_c,
                "peak_coord_rss_mb": summary["peak_coord_rss_mb"],
                "peak_worker_rss_mb": summary["peak_worker_rss_mb"],
                "peak_aggregate_rss_mb": summary["peak_aggregate_rss_mb"],
                "worker_recycles": summary["worker_recycles"],
                "errors": len(summary["errors"]),
                "hashes_match_8w": bool(hashes_by_w[nw] == hashes_by_w[args.workers_list[0]]),
            }
            rows.append(row)
            print("[SMT_BENCH]", json.dumps(row, ensure_ascii=False), flush=True)

        # Selection rule: >= +10.0% sustained throughput vs 8w, 0 errors, exact hash match, < 6500 MB RSS
        best_w = args.workers_list[0]
        best_gain = 0.0
        for r in rows[1:]:
            if (
                r["gain_vs_8w_pct"] >= 10.0
                and r["errors"] == 0
                and r["hashes_match_8w"]
                and r["peak_aggregate_rss_mb"] < 6500.0
                and r["gain_vs_8w_pct"] > best_gain
            ):
                best_w = int(r["workers"])
                best_gain = float(r["gain_vs_8w_pct"])

        payload = {
            "recommended_workers": best_w,
            "selection_criterion": ">= +10.0% units/min vs 8w, 0 errors, exact hash match, < 6500 MB aggregate RSS",
            "rows": rows,
        }
        pd.DataFrame(rows).to_csv(out_dir_res / "benchmark_smt_8_10_12.csv", index=False)
        (out_dir_res / "benchmark_smt_8_10_12.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("[RECOMMENDATION]", json.dumps({"recommended_workers": best_w, "best_gain_pct": best_gain}), flush=True)
    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

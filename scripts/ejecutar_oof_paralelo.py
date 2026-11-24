"""Windows-safe parallel OOF coordinator CLI for SM-CFHR v1.0.0 (dry-run by default)."""

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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ashr_oof_pipeline.config import load_frozen_config, sha256_file  # noqa: E402
from ashr_oof_pipeline.optimizacion_hiperparametros import (  # noqa: E402
    HPO_PROTOCOL_VERSION,
    compute_hpo_protocol_hash,
)
from sm_cfhr.parallel import (  # noqa: E402
    ParallelOOFCoordinator,
    build_work_inventory,
    consolidate_parallel_run,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SM-CFHR v1.0.0 parallel OOF coordinator (dry-run unless --confirm-oof)."
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of spawn worker processes (default: 4).")
    parser.add_argument(
        "--variante",
        choices=("fixed", "hpo", "ambos"),
        default="fixed",
        help="Variant to evaluate: fixed, hpo, or ambos.",
    )
    parser.add_argument("--n-trials", type=int, default=25, help="Optuna TPE trials per HPO unit (default: 25).")
    parser.add_argument("--seed", type=int, default=2026, help="Deterministic seed (default: 2026).")
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=None,
        help="Optional horizon filter in days (e.g. 1 7 14).",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=None,
        help="Optional scenario_id filter.",
    )
    parser.add_argument("--max-tasks-per-worker", type=int, default=60, help="Recycle worker after N units.")
    parser.add_argument("--worker-max-rss-mb", type=float, default=1536.0, help="Recycle worker above RSS MB.")
    parser.add_argument("--run-id", type=str, default="oof_v1_0_0_local", help="Isolated run identifier.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "oof_v1_0_0",
        help="Isolated output base directory (never touches resultados/evaluacion_ashr_oof).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT,
        help="Root containing base_generacion_clima/, datos_pronostico*, and resultados/benchmark_*.",
    )
    parser.add_argument("--shard-index", type=int, default=0, help="Disjoint shard index [0..num-shards-1].")
    parser.add_argument("--num-shards", type=int, default=1, help="Total disjoint shards across machines.")
    parser.add_argument("--status", action="store_true", help="Inspect current progress.json and exit.")
    parser.add_argument("--consolidate", action="store_true", help="Consolidate persisted fragments and exit.")
    parser.add_argument("--confirm-oof", action="store_true", help="Authorize real parallel execution.")
    args = parser.parse_args()

    run_dir = args.out_dir / args.run_id
    if args.status:
        prog_p = run_dir / "ledger" / "progress.json"
        if not prog_p.exists():
            print(json.dumps({"run_id": args.run_id, "status": "NOT_FOUND", "path": str(prog_p)}, indent=2))
            return 1
        print(prog_p.read_text(encoding="utf-8"))
        return 0

    if args.consolidate:
        res = consolidate_parallel_run(run_dir=run_dir, root=ROOT, require_complete=False)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    cfg = load_frozen_config(ROOT)
    src_hash = sha256_file(ROOT / "src" / "adaptive_structured_huber.py")
    expected_hash = cfg["sha256_src_adaptive_structured_huber"]

    if not args.confirm_oof:
        data_ok = (args.data_root / "base_generacion_clima" / "base_planta_dia.csv").exists()
        preview = {
            "estatus": "DRY_RUN_PREPARADO",
            "ejecutado": False,
            "workers": args.workers,
            "variante": args.variante,
            "n_trials": args.n_trials if args.variante in ("hpo", "ambos") else 0,
            "run_id": args.run_id,
            "out_dir": str(run_dir),
            "data_root": str(args.data_root),
            "data_root_present": data_ok,
            "shard": f"{args.shard_index}/{args.num_shards}",
            "hashes_coinciden": bool(src_hash == expected_hash),
            "hpo_protocol_version": HPO_PROTOCOL_VERSION,
            "hpo_protocol_hash": compute_hpo_protocol_hash(ROOT),
        }
        if data_ok:
            units = build_work_inventory(
                root=ROOT,
                data_root=args.data_root,
                variante=args.variante,
                n_trials=args.n_trials,
                seed=args.seed,
                scenarios=args.scenarios,
                horizons=args.horizons,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
            preview["n_units"] = len(units)
        print(json.dumps(preview, indent=2, ensure_ascii=False))
        return 0

    units = build_work_inventory(
        root=ROOT,
        data_root=args.data_root,
        variante=args.variante,
        n_trials=args.n_trials,
        seed=args.seed,
        scenarios=args.scenarios,
        horizons=args.horizons,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    coord = ParallelOOFCoordinator(
        out_dir=args.out_dir,
        run_id=args.run_id,
        repo_root=ROOT,
        data_root=args.data_root,
        n_workers=args.workers,
        max_tasks_per_worker=args.max_tasks_per_worker,
        worker_max_rss_mb=args.worker_max_rss_mb,
        variante=args.variante,
        n_trials=args.n_trials,
        seed=args.seed,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    summary = coord.run(units)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

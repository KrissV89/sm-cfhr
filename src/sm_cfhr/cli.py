"""Master CLI entry point for SM-CFHR: doctor, evaluate, status, resume, report."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    from ashr_oof_pipeline.config import (
        SCENARIOS_28,
        find_project_root,
        load_frozen_config,
        sha256_file,
        verify_benchmark_inventory,
    )
    from ashr_oof_pipeline.oof_runner import run_oof_definitivo
except ImportError:
    from src.ashr_oof_pipeline.config import (
        SCENARIOS_28,
        find_project_root,
        load_frozen_config,
        sha256_file,
        verify_benchmark_inventory,
    )
    from src.ashr_oof_pipeline.oof_runner import run_oof_definitivo

from .persistence import get_process_rss_mb
from .pipeline_stages import run_post_oof_stages
from .reproducibility import compare_reproducibility_runs, generate_reproducibility_bundle


def cmd_doctor(root: Path) -> int:
    """Verify Python, dependencies, data files, benchmarks, hashes, disk space, and write permissions."""
    cfg = load_frozen_config(root)
    src_file = root / "src" / "adaptive_structured_huber.py"
    src_hash = sha256_file(src_file) if src_file.exists() else "MISSING"
    expected_hash = cfg.get("sha256_src_adaptive_structured_huber", "")

    manifest_p = root / "data" / "manifests" / "data_manifest.json"
    data_checks: list[dict[str, Any]] = []
    if manifest_p.exists():
        dm = json.loads(manifest_p.read_text(encoding="utf-8"))
        for item in dm.get("datasets", []):
            p = root / item["expected_path"]
            exists = p.exists()
            actual_sha = sha256_file(p) if exists else None
            data_checks.append(
                {
                    "name": item["name"],
                    "expected_path": item["expected_path"],
                    "classification": item["classification"],
                    "exists": exists,
                    "sha256_ok": bool(exists and actual_sha == item["sha256"]),
                }
            )

    inv = verify_benchmark_inventory(root)
    bench_ok = bool(inv["file_exists"].all() and inv["cols_ok"].all())
    disk = shutil.disk_usage(root)
    free_gb = round(disk.free / (1024**3), 2)

    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    probe = results_dir / ".write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        writable = True
    except OSError:
        writable = False

    report = {
        "python_version": sys.version.split()[0],
        "project_root": str(root),
        "frozen_version": cfg.get("version_codigo"),
        "core_sha256_match": bool(src_hash == expected_hash),
        "n_scenarios": len(SCENARIOS_28),
        "benchmarks_present_and_valid": bench_ok,
        "n_benchmarks_found": int(inv["file_exists"].sum()),
        "datasets": data_checks,
        "writable_results_dir": writable,
        "free_disk_gb": free_gb,
        "rss_mb": round(get_process_rss_mb(), 2),
        "ready_for_full_oof": bool(
            src_hash == expected_hash
            and bench_ok
            and all(d["exists"] and d["sha256_ok"] for d in data_checks)
            and writable
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def cmd_status(root: Path, run_id: str, out_dir: Path | None) -> int:
    """Inspect live progress of a run without interrupting it."""
    base = out_dir or (root / "resultados" / "evaluacion_ashr_oof")
    prog_p = base / "runs" / run_id / "logs" / "progress.json"
    if not prog_p.exists():
        print(json.dumps({"run_id": run_id, "status": "NOT_FOUND", "path": str(prog_p)}, indent=2))
        return 1
    print(prog_p.read_text(encoding="utf-8"))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sm-cfhr", description="SM-CFHR Master Scientific CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Diagnose environment, hashes, datasets, and benchmarks.")

    for name in ("evaluate", "resume"):
        p = sub.add_parser(name, help=f"{name.capitalize()} 28-scenario OOF (dry-run unless --confirm-oof).")
        p.add_argument("--variante", choices=("fixed", "hpo", "ambos"), default="ambos")
        p.add_argument("--n-trials", type=int, default=25)
        p.add_argument("--run-id", type=str, default="oof_definitivo_v1_1")
        p.add_argument("--max-rss-mb", type=float, default=4096.0)
        p.add_argument("--out-dir", type=Path, default=None)
        p.add_argument("--confirm-oof", action="store_true", help="Explicitly authorize definitive OOF execution.")

    p_st = sub.add_parser("status", help="Inspect live run progress.")
    p_st.add_argument("--run-id", type=str, default="oof_definitivo_v1_1")
    p_st.add_argument("--out-dir", type=Path, default=None)

    p_rep = sub.add_parser("report", help="Run post-OOF stages 2-8 in CONFIRMATORY mode.")
    p_rep.add_argument("--oof-dir", type=Path, default=None)

    p_rb = sub.add_parser("repro-bundle", help="Generate results/runs/<run_id>/reproducibilidad/ evidence bundle.")
    p_rb.add_argument("--run-id", type=str, default="oof_definitivo_v1_1")

    p_cmp = sub.add_parser("compare-runs", help="Compare two independent OOF result directories within tolerances.")
    p_cmp.add_argument("--dir-a", type=Path, required=True)
    p_cmp.add_argument("--dir-b", type=Path, required=True)
    p_cmp.add_argument("--out-md", type=Path, default=None)

    args = parser.parse_args(argv)
    root = find_project_root(Path.cwd())

    if args.command == "doctor":
        return cmd_doctor(root)
    if args.command == "status":
        return cmd_status(root, args.run_id, args.out_dir)
    if args.command in ("evaluate", "resume"):
        res = run_oof_definitivo(
            execute=bool(args.confirm_oof),
            out_dir=args.out_dir,
            root=root,
            variante=args.variante,
            n_trials=args.n_trials,
            run_id=args.run_id,
            max_rss_mb=args.max_rss_mb,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    if args.command == "report":
        oof_dir = args.oof_dir or (root / "resultados" / "evaluacion_ashr_oof")
        res = run_post_oof_stages(oof_dir=oof_dir, root=root, allow_synthetic_test=False, mode="CONFIRMATORY")
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    if args.command == "repro-bundle":
        paths = generate_reproducibility_bundle(root=root, run_id=args.run_id)
        print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2, ensure_ascii=False))
        return 0
    if args.command == "compare-runs":
        res = compare_reproducibility_runs(args.dir_a, args.dir_b, out_md=args.out_md)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI wrapper for ASHR 28-scenario OOF evaluation (dry-run by default)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ashr_oof_pipeline.oof_runner import run_oof_definitivo  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="ASHR OOF definitive pipeline (dry-run by default).")
    parser.add_argument(
        "--ejecutar-oof-definitivo",
        action="store_true",
        help="Execute the full 28-scenario OOF evaluation (omit for dry-run audit).",
    )
    parser.add_argument(
        "--variante",
        choices=("fixed", "hpo", "ambos"),
        default="fixed",
        help="Variant to run: SM-CFHR-Fixed, SM-CFHR-HPO, or both.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=25,
        help="Optuna TPE trials per outer fold when variante includes hpo (default: 25).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Deterministic run identifier for checkpoints/resume.",
    )
    parser.add_argument(
        "--max-rss-mb",
        type=float,
        default=4096.0,
        help="Process RSS memory guard in MB (default: 4096).",
    )
    args = parser.parse_args()
    res = run_oof_definitivo(
        execute=args.ejecutar_oof_definitivo,
        variante=args.variante,
        n_trials=args.n_trials,
        run_id=args.run_id,
        max_rss_mb=args.max_rss_mb,
        root=ROOT,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

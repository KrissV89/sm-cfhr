"""CLI for resumable post-OOF stages 2-8 (metrics, inference, studies, tables, and figures)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sm_cfhr.pipeline_stages import run_post_oof_stages  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable post-OOF stages 2-8 from persisted results.")
    parser.add_argument(
        "--oof-dir",
        type=Path,
        default=ROOT / "resultados" / "evaluacion_ashr_oof",
        help="Directory containing persisted OOF predictions and manifest_ashr_oof.json.",
    )
    args = parser.parse_args()
    res = run_post_oof_stages(oof_dir=args.oof_dir, root=ROOT, allow_synthetic_test=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

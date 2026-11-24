"""Modular OOF evaluation, results loading, statistical inference, and visualization for ASHR."""

from .config import (
    SCENARIOS_28,
    build_frozen_ashr,
    find_project_root,
    load_frozen_config,
    sha256_file,
    verify_benchmark_inventory,
)
from .resultados_loader import (
    build_descriptive_summary,
    export_paper_tables,
    load_oof_bundle,
    validate_oof_artifacts,
    validate_predictions_df,
)

__all__ = [
    "SCENARIOS_28",
    "build_descriptive_summary",
    "build_frozen_ashr",
    "export_paper_tables",
    "find_project_root",
    "load_frozen_config",
    "load_oof_bundle",
    "sha256_file",
    "validate_oof_artifacts",
    "validate_predictions_df",
    "verify_benchmark_inventory",
]

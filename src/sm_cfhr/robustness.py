"""Robustness, sensitivity, ablation, conformal uncertainty, and interpretability wrappers."""

from __future__ import annotations

try:
    from ashr_oof_pipeline.robustez_y_ablacion import (
        compute_interpretability_summary,
        compute_temporal_conformal_intervals,
        run_ablation_suite,
        run_hp_sensitivity,
        run_learning_and_scalability_curves,
        run_noise_outlier_stress,
        run_seed_stability_hpo,
    )
except ImportError:
    from src.ashr_oof_pipeline.robustez_y_ablacion import (
        compute_interpretability_summary,
        compute_temporal_conformal_intervals,
        run_ablation_suite,
        run_hp_sensitivity,
        run_learning_and_scalability_curves,
        run_noise_outlier_stress,
        run_seed_stability_hpo,
    )

__all__ = [
    "compute_interpretability_summary",
    "compute_temporal_conformal_intervals",
    "run_ablation_suite",
    "run_hp_sensitivity",
    "run_learning_and_scalability_curves",
    "run_noise_outlier_stress",
    "run_seed_stability_hpo",
]

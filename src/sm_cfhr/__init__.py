"""SM-CFHR: Structured Monotone Capacity-Factor Huber Regressor reusable scientific package."""

from __future__ import annotations

from .estimator import AdaptiveStructuredHuberRegressor, SMCFHRRegressor
from .inference import (
    apply_holm,
    benjamini_hochberg_adjust,
    bootstrap_ci_difference,
    build_23_model_ranking,
    clark_west_test,
    cliffs_delta,
    cohens_d_paired,
    compute_scenario_metrics,
    evaluate_r3_superiority,
    giacomini_white_audited,
    giacomini_white_test,
    model_confidence_set,
    run_paired_inference,
    spa_test,
)
from .io import (
    ensure_artifact_tree,
    export_full_scientific_artifacts,
)
from .optimization import (
    nested_cv_optimize_and_refit,
    optimize_hyperparams_tpe,
    summarize_hpo_trials,
)
from .robustness import (
    compute_interpretability_summary,
    compute_temporal_conformal_intervals,
    run_ablation_suite,
    run_hp_sensitivity,
    run_learning_and_scalability_curves,
    run_noise_outlier_stress,
    run_seed_stability_hpo,
)
from .validation import (
    SCENARIOS_28,
    analyze_effective_design,
    audit_inner_splits,
    build_causal_inner_splits,
    verify_benchmark_inventory,
)

__version__ = "1.0.3"

__all__ = [
    "AdaptiveStructuredHuberRegressor",
    "SCENARIOS_28",
    "SMCFHRRegressor",
    "__version__",
    "analyze_effective_design",
    "apply_holm",
    "audit_inner_splits",
    "benjamini_hochberg_adjust",
    "bootstrap_ci_difference",
    "build_23_model_ranking",
    "build_causal_inner_splits",
    "clark_west_test",
    "cliffs_delta",
    "cohens_d_paired",
    "compute_interpretability_summary",
    "compute_scenario_metrics",
    "compute_temporal_conformal_intervals",
    "ensure_artifact_tree",
    "evaluate_r3_superiority",
    "export_full_scientific_artifacts",
    "giacomini_white_audited",
    "giacomini_white_test",
    "model_confidence_set",
    "nested_cv_optimize_and_refit",
    "optimize_hyperparams_tpe",
    "run_ablation_suite",
    "run_hp_sensitivity",
    "run_learning_and_scalability_curves",
    "run_noise_outlier_stress",
    "run_paired_inference",
    "run_seed_stability_hpo",
    "spa_test",
    "summarize_hpo_trials",
    "verify_benchmark_inventory",
]

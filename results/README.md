# Resultados OOF y Runs Resumibles

Las corridas ejecutadas con `python -m sm_cfhr.cli evaluate --confirm-oof` se almacenan en:
- `resultados/evaluacion_ashr_oof/runs/<run_id>/` (checkpoints, logs JSONL, bases SQLite Optuna, fragmentos Parquet)
- `resultados/evaluacion_ashr_oof/predicciones/` (consolidados CSV por escenario)
- `resultados/evaluacion_ashr_oof/manifest_ashr_oof.json`

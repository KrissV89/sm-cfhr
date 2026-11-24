# Provenance and Release Lineage — SM-CFHR (`KrissV89/sm-cfhr`)

## 1. Official Distribution Identity
- **Repository:** `https://github.com/KrissV89/sm-cfhr`
- **Owner:** `KrissV89`
- **Release Tag:** `v1.0.3-polyform`
- **License (from initial commit):** PolyForm Noncommercial License 1.0.0 (`PolyForm-Noncommercial-1.0.0`)
- **Copyright Notice:** `Copyright (c) 2026 Enmanuel David Diaz Molinares`
- **Commercial Licensing Contact:** `edmolinares@soulspacers.com`

## 2. Scientific Equivalence with Historical `v1.0.3`
This repository starts with an independent, clean Git history and carries forward the exact scientific implementation of `v1.0.3` from the historical archive repository (`KrissV89/structured-monotone-capacity-factor-huber-regressor`, now private):

- **Core estimator (`src/adaptive_structured_huber.py`) SHA-256:**
  `27a9fda4fa8b7ef0d95faf9e925e47ade9ed8f25e9ea5d97d6d7cf1c00ddde83` (unchanged from `v1.0.2` and `v1.0.3`).
- **Frozen protocol configuration (`experiments/config_ashr_congelada.json`):**
  `version_codigo = "1.0.3-oof-definitivo"`.
- **HPO recursive rollout protocol (`src/ashr_oof_pipeline/optimizacion_hiperparametros.py`):**
  `HPO_PROTOCOL_VERSION = "v1.0.3-exact-recursive-state"`,
  `HPO_PROTOCOL_HASH = "e2a1d2b68fb9fbf90f49ab4caf5c33ae011bc03d63b2486f7ab1cf593b604d5d"`.
- **Historical origin commit:** `d522cf3a288ca08f6080a17247794f706358c145` (tag `v1.0.3` in the historical private repository).

No equations, loss function, gradient, physical box constraints, Fixed configuration, Optuna TPE search space, seeds, 25-trial budget, 28 scenarios, or 22 benchmarks were modified.

## 3. Licensing and Rights Separation
- **Original software code** in `KrissV89/sm-cfhr` (estimator, OOF pipeline, CLI, scripts, tests, dashboard, notebook Python code, scientific utilities, and Lean 4 formalization in `formal/sm_cfhr_lean/`) is licensed under **PolyForm Noncommercial License 1.0.0** from its initial commit.
- **Historical releases (`v1.0.0`, `v1.0.1`, `v1.0.2`, `v1.0.3`)** previously published in the historical repository remain governed by the MIT terms under which those specific releases were originally distributed (non-retroactivity).
- **External datasets** (XM, Open-Meteo, Global Energy Monitor) and **third-party dependencies** (NumPy, SciPy, Pandas, scikit-learn, Statsmodels, Optuna, Lean 4, Mathlib) are NOT relicensed under PolyForm and remain subject to their respective licenses and terms of use (see `data/README.md` and `data/manifests/data_manifest.json`).
- **Academic manuscript and figures** (`docs/paper_sm_cfhr/`) retain their separate academic authorship and copyright.

## 4. Collaborator Migration Instructions (Without Interrupting Active Runs)
If you are running an evaluation from the historical repository (`v1.0.2` or `v1.0.3`):
1. **Do NOT stop or modify your active run** and do NOT alter your local data directory or existing checkpoints.
2. When ready to use the new official distribution, clone into a separate folder:
   ```bash
   git clone https://github.com/KrissV89/sm-cfhr.git
   cd sm-cfhr
   pip install -e .
   ```
3. Import your existing local data into the new clone without touching the original data files:
   ```bash
   python scripts/setup/import_local_data.py --source-root <PATH_TO_EXTRACTED_DATA_FOLDER>
   python -m sm_cfhr.cli doctor
   ```
4. Note that `v1.0.2` Optuna SQLite studies cannot be reused with `v1.0.3` / `v1.0.3-polyform` due to the exact recursive state fix (`HPO_PROTOCOL_VERSION = "v1.0.3-exact-recursive-state"`).

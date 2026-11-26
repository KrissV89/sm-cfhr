# Provenance and Release Lineage — SM-CFHR (`KrissV89/sm-cfhr`)

## 1. Distribution Identity
- **Repository:** `https://github.com/KrissV89/sm-cfhr`
- **Owner:** `KrissV89`
- **Current Release Tag:** `v3.0.0`
- **License:** PolyForm Noncommercial License 1.0.0 (`PolyForm-Noncommercial-1.0.0`)
- **Copyright Notice:** `Copyright (c) 2026 Enmanuel David Diaz Molinares`
- **Commercial Licensing Contact:** `edmolinares@soulspacers.com`

## 2. Cryptographic & Protocol Anchors
- **Core estimator (`src/adaptive_structured_huber.py`) SHA-256:**
  `27a9fda4fa8b7ef0d95faf9e925e47ade9ed8f25e9ea5d97d6d7cf1c00ddde83`
- **Frozen OOF protocol configuration (`experiments/config_ashr_congelada.json`):**
  `version_codigo = "1.0.3-oof-definitivo"`, `HPO_PROTOCOL_VERSION = "v1.0.3-exact-recursive-state"`, `HPO_PROTOCOL_HASH = "e2a1d2b68fb9fbf90f49ab4caf5c33ae011bc03d63b2486f7ab1cf593b604d5d"`.
- **Frozen OOF integrity:** All `57` artifact hashes under `resultados/evaluacion_ashr_oof/` are preserved unmodified across the `v2.0.0` inference reconciliation and `v3.0.0` representation-equivalence / ablation experiments.

## 3. Licensing and Rights Separation
- **Original software code** in `KrissV89/sm-cfhr` (estimator, OOF pipeline, CLI, scripts, tests, dashboard, notebook Python code, scientific utilities, and Lean 4 formalization in `formal/sm_cfhr_lean/`) is licensed under **PolyForm Noncommercial License 1.0.0**.
- **External datasets** (XM, Open-Meteo, Global Energy Monitor) and **third-party dependencies** (NumPy, SciPy, Pandas, scikit-learn, Statsmodels, Optuna, Lean 4, Mathlib) retain their respective licenses and terms of use (see `data/README.md` and `data/manifests/data_manifest.json`).
- **Academic manuscript, supplementary material, and figures** (`docs/paper_sm_cfhr/`) retain their separate academic authorship and copyright.

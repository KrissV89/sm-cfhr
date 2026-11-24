# Matriz GO / NO-GO — Preflight Final `SM-CFHR` / `ASHR`

**Decisión técnica global:** **`GO TÉCNICO`**

| # | Criterio de Preflight | Clasificación | Evidencia verificada | Limitación / Nota |
|---|---|---|---|---|
| 1 | Pruebas unitarias y anti-fuga (`41` previas + `6` integración) | `PASS` | `pytest`: **47/47 PASSED** (`208.03 s`) | — |
| 2 | Integración end-to-end real en desarrollo (`5` regímenes) | `PASS` | `resultados/preflight_ashr/e2e_desarrollo_resumen.csv` | Ejecutado en subconjuntos de desarrollo (`D1` / primeros orígenes). |
| 3 | Ejecución real HPO TPE (`B=25` + selección + refit) | `PASS` | `resultados/preflight_ashr/hpo_trials_d1_et_b25.csv` | `50` trials reales (`2` grupos $\times 25$), `wall_time=3.652 s`. |
| 4 | Ausencia de fugas en Nested CV (`D+1`, `Rec`, `Directo`, `ET`, `IP`) | `PASS` | `resultados/preflight_ashr/nested_cv_audit.csv` (`27` splits auditados) | En slice de 2 grupos (`Rec28d_ET`/`Directo28d_ET`), `spatial_disjoint` cae a temporal por tamaño mínimo; con 7 grupos reales es `True`. |
| 5 | Reproducción causal de `D+1`, `Recursivo` y `Directo` | `PASS` | `oof_runner.py` + `preflight_ashr_desarrollo.py` | — |
| 6 | Separación `IP` (temporal) y `ET` (espacial + temporal) | `PASS` | `audit_inner_splits(..., strict=True)` | — |
| 7 | Alineación de claves, unidades y métricas (`MAE_macro_kWh`) | `PASS` | `resultados_loader.py` + `optimizacion_hiperparametros.py` | — |
| 8 | Integridad y seguridad operacional del comando definitivo | `PASS` | Hash SHA-256 estricto, escritura atómica y checkpointing en `oof_runner.py` | — |
| 9 | Inferencia estadística conectada al flujo definitivo (`DM-HLN-HAC`, `GW`, `MCS`, `Holm`) | `PASS CON LIMITACIONES` | `load_oof_bundle` + `auditoria_estadistica_h0.json` | `GW` con `tau=1` sobrerrechaza bajo errores dependientes `AR(1)`; usar junto a `DM-HLN-HAC` y `MCS`. |
| 10 | Notebook funcional en Estado A y Estado B (fixture aislado) | `PASS` | `nbconvert --execute` (Estado A) + `estado_b_fixture_resumen.json` (Estado B: 28 escenarios, 10 tablas, 5 figuras) | — |
| 11 | Equidad experimental frente a los 22 benchmarks | `PASS CON LIMITACIONES` | `AUDITORIA_EQUIDAD_BENCHMARKS.md` | Benchmarks históricos usaron rejillas de 4 puntos; comparación principal justa es `SM-CFHR-Fixed` ($B=0$). |
| 12 | Coste computacional caracterizado | `PASS` | `estimacion_coste_28_escenarios.csv`: `19,497` fits (`Fixed`, ~14.3 min) vs `1,481,772` fits (`HPO B=25`, ~9.9 h) | `HPO B=25` en 28 escenarios requiere ejecución prolongada con checkpointing. |
| 13 | Congelamiento y trazabilidad SHA-256 | `PASS` | `MANIFIESTO_PREFLIGHT.json`, `config_ashr_congelada.json` | Tag git pendiente de crear tras commit final. |
| 14 | Conjuntos confirmatorios reservados intactos | `PASS` | `resultados/evaluacion_ashr_oof/` no existe (`test_preflight_06`) | — |
| 15 | Jupyter Book y Dashboard | `PASS CON LIMITACIONES` / `NO VERIFICABLE` | `jupyter-book v2.1.6` verificado; `dash` ausente en `solarpv-eda` | Actualización de `book/` y `app.py` depende del OOF definitivo; arranque de `dash` requiere entorno de Render. |

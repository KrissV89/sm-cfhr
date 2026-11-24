# Informe Final de Preparación y Auditoría Integral — Entregable 3 (`SM-CFHR` / `ASHR`)

**Fecha:** 2026-11-23  
**Entorno verificado:** Python `3.11.15` (`solarpv-eda`)  
**Estado de ejecución OOF definitiva:** `NO EJECUTADO` (`"oof_definitivo_ejecutado": false`)

---

## Respuestas Explícitas A–T

- **A. Estado general de cumplimiento Entregable 3:** `DOCUMENTADO` e `IMPLEMENTADO` en preparación (`docs/auditoria_entregable3/MATRIZ_CUMPLIMIENTO_ENTREGABLE3.md`); veredicto empírico confirmatorio `PENDIENTE` de ejecución `--ejecutar-oof-definitivo`. Guías PDF externas (`Entregable3MachineLearningProject*.pdf`, `9. Evaluación de modelos*.pdf`) marcadas `NO ACCESIBLE EN REPOSITORIO`.
- **B. Alcance y dominio exclusivo (28 escenarios solares, sin datasets externos):** `DOCUMENTADO`; sustitución de datasets externos por los **28 escenarios solares** (17 plantas XM Colombia) registrada como `PENDIENTE DE VALIDACIÓN DOCENTE`. No se ejecutó ni tocó `cpu_act`, `sulfur`, `trafico_i94` ni `44136/44138/44147`.
- **C. R1 — Novedad y revisión formal de literatura:** `DOCUMENTADO` (`docs/auditoria_entregable3/AUDITORIA_R1_R2_R3.md`). Basado en 24 exports reales de **WoS Core Collection** (`fuentes/bibliografia/wos/`), 8 trabajos cercanos comparados, *snowballing* y afirmación acotada al dominio fotovoltaico multi-planta/multi-horizonte. Exportaciones de Scopus/IEEE Xplore en disco: `PENDIENTE` (si el docente exige archivo adicional).
- **D. R2 — Implementación propia en NumPy/SciPy:** `IMPLEMENTADO` y `PROBADO` (`src/adaptive_structured_huber.py`, `src/modelo_aditivo_huber.py`). Pérdida Huber, escalado MAD, L2 estructurada, cotas monótonas y L-BFGS-B con gradiente analítico 100% NumPy/SciPy.
- **E. Compatibilidad `scikit-learn` (`BaseEstimator`, `RegressorMixin`, `clone`, `Pipeline`, serialización):** `IMPLEMENTADO` y `PROBADO` (tests 01–04 en `tests/test_auditoria_integral_entregable3.py`). Equivalencia exacta `< 1e-12` preservada respecto a `Fijo_Lineal_Mono`.
- **F. Optimización de hiperparámetros (`delta`, `alpha_lin`, `alpha_meteo`):** `IMPLEMENTADO`, `PROBADO` y `DOCUMENTADO` (`src/ashr_oof_pipeline/optimizacion_hiperparametros.py`, `docs/auditoria_entregable3/PROTOCOLO_OPTIMIZACION_ASHR.md`). Optuna `TPESampler` con arquitectura estructural fija y registro completo de ensayos.
- **G. Validación cruzada causal anidada (`D+1`, `Rec 7–28d`, `Directo 7–28d`, `ET`/`IP`):** `IMPLEMENTADO` y `PROBADO` (tests 10–14). Garantiza embargo temporal, `target_end_date_train <= origin_val`, `origin_train + h <= origin_val` y exclusión espacial de grupo en `ET`.
- **H. Auditoría de fuga de datos (etiqueta, imputación, escalado, tiempo, espacio):** `IMPLEMENTADO` y `PROBADO` (tests 08–13). Imputación `nanmedian` y escalado ajustados estrictamente dentro de `train`.
- **I. Benchmarks oficiales y equidad de presupuestos (22 referencias + `SM-CFHR`):** `DOCUMENTADO` (`docs/auditoria_entregable3/AUDITORIA_BENCHMARKS_Y_PRESUPUESTOS.md`) y `PROBADO` (test 20). Inventario exacto: 9 librerías + 12 custom + `Media móvil de 30 días` (`CF-BRH` excluido). Política de doble reporte (`SM-CFHR-Fixed` B=0 vs `SM-CFHR-HPO` B=25 TPE).
- **J. R3 — Inferencia estadística (`DM-HLN-HAC`, `GW`, `Clark-West`, `MCS`, Holm, IC bootstrap):** `IMPLEMENTADO` y `PROBADO` en sintético (`src/ashr_oof_pipeline/evaluacion_estadistica.py`, test 19); veredicto confirmatorio `PENDIENTE` de OOF definitivo.
- **K. Estudios de ablación controlada (6 variantes):** `IMPLEMENTADO` y `PROBADO` en sintético (`src/ashr_oof_pipeline/robustez_y_ablacion.py::run_ablation_suite`, test 17); ejecución en 28 escenarios `PENDIENTE` de OOF definitivo.
- **L. Robustez, sensibilidad HP, estabilidad (>=10 semillas), ruido/outliers, aprendizaje y escalabilidad:** `IMPLEMENTADO` y `PROBADO` en sintético (`src/ashr_oof_pipeline/robustez_y_ablacion.py`, tests 15–17); ejecución en 28 escenarios `PENDIENTE` de OOF definitivo.
- **M. Incertidumbre predictiva (intervalos conformes temporales):** `IMPLEMENTADO` y `PROBADO` en sintético (`compute_temporal_conformal_intervals`, test 18); calibración definitiva `PENDIENTE` de OOF definitivo.
- **N. Interpretabilidad global y local:** `IMPLEMENTADO` y `PROBADO` (`compute_interpretability_summary`, test 20). Descomposición aditiva exacta verificada (`baseline + sum(contrib) == pred` a `< 1e-10`).
- **O. Notebook científico definitivo (16 secciones, modo dual A/B):** `IMPLEMENTADO` y `PROBADO` (`notebooks/actuales/evaluacion_cientifica_ashr_oof.ipynb`, validado con `jupyter nbconvert --execute` en kernel `solarpv-eda`).
- **P. Jupyter Book (`book/`):** `DOCUMENTADO` (`docs/auditoria_entregable3/AUDITORIA_DASHBOARD_Y_E2.md`). `jupyter-book v2.1.6` verificado en sistema; actualización de capítulo `SM-CFHR` `PENDIENTE` de resultados OOF definitivos.
- **Q. Dashboard predictivo (`app.py`, `pages/`, `render.yaml`):** `DOCUMENTADO` (`docs/auditoria_entregable3/AUDITORIA_DASHBOARD_Y_E2.md`). Integración de artefactos `SM-CFHR` `PENDIENTE` de OOF definitivo; arranque en vivo desde `solarpv-eda` `BLOQUEADO / NO PROBADO EN ESTE ENTORNO` por ausencia del paquete `dash` en `solarpv-eda`.
- **R. Protocolo pre-registrable y configuración congelada:** `IMPLEMENTADO` y `DOCUMENTADO` (`experiments/protocolo.md`, `experiments/config_ashr_congelada.json`).
- **S. Suite de pruebas unitarias e integrales:** `PROBADO` (**41/41 PASS**, `0` fallos, `107.77 s` en `solarpv-eda`).
- **T. Manifiesto SHA-256 y bloqueadores residuales:** `IMPLEMENTADO` (`resultados/manifiesto_preparacion_entregable3.json`). Bloqueadores únicos para el cierre final del curso: (1) autorización para ejecutar `--ejecutar-oof-definitivo`, (2) aval docente sobre el alcance exclusivo solar (sin datasets externos), y (3) lectura/cotejo con los PDFs de rúbrica si se incorporan al repositorio.

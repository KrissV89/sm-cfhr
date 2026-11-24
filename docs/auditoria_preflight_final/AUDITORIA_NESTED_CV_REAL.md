# Auditoría de Validación Cruzada Anidada Real (`Nested CV`)

**Módulos auditados:** `src/ashr_oof_pipeline/optimizacion_hiperparametros.py` (`build_causal_inner_splits`, `audit_inner_splits`, `nested_cv_optimize_and_refit`) y `src/ashr_oof_pipeline/oof_runner.py`.  
**Evidencia medida:** `resultados/preflight_ashr/nested_cv_audit.csv` (27 splits internos reales auditados con `strict=True`).

---

## 1. Garantías Verificadas por Fold Externo

1. **Aislamiento de `test_outer`:** `optimize_hyperparams_tpe` recibe únicamente `train_outer`; la imputación (`nanmedian`), estandarización (`x_mean`, `x_scale`), escalado robusto (`MAD`) y optimización L-BFGS-B se ejecutan dentro de cada split interno de `train_outer`.
2. **Validación temporal por régimen:**
   - **`D+1` (`d1`):** $\max(\text{date}_{\text{train}}) \le \min(\text{date}_{\text{val}}) - 30\text{d}$ (`effective_gap_days = 30` en `D1_ET` y `D1_IP`).
   - **`Recursivo` (`7–28d`):** $\max(\text{origin}_{\text{train}}) + H \le \min(\text{origin}_{\text{val}})$ (`effective_gap_days = 14` en `Rec14d_IP`, `28` en `Rec28d_ET`).
   - **`Directo` (`7–28d`):** $\max(\text{t\_end\_target}_{\text{train}}) \le \min(\text{t\_origin}_{\text{val}})$ (`effective_gap_days = 28` en `Directo28d_ET`).
3. **Validación espacial (`ET`):**
   - En `D1_espaciotemporal_cinco_variables`, los 6 splits internos auditados cumplen `spatial_disjoint = True` (retienen el grupo de validación completo y lo excluyen de `train_inner`).
   - *Nota técnica observada:* en los ensayos reducidos de `Rec28d_ET` y `Directo28d_ET` (donde la muestra de preflight se acotó a 2 grupos), el filtro espacial dejaba `< 15` filas en `train_inner`, activando el respaldo temporal seguro (`spatial_disjoint=False`). Con los **7 grupos completos** en la corrida oficial, el filtro espacial se mantiene activo.
4. **Corrección aplicada durante el preflight:** Se añadió soporte explícito para columnas reales `t_origin` y `t_end_target` en `build_causal_inner_splits` y se implementó `audit_inner_splits(..., strict=True)`.

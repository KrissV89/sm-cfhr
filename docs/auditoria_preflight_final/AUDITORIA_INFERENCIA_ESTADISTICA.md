# Auditoría de Inferencia Estadística y Calibración bajo $H_0$

**Módulos:** `src/ashr_oof_pipeline/evaluacion_estadistica.py` y `src/ashr_oof_pipeline/resultados_loader.py`  
**Evidencia medida:** `resultados/preflight_ashr/auditoria_estadistica_h0.json`

---

## 1. Conexión al Flujo Definitivo

- `load_oof_bundle` ejecuta automáticamente sobre las series temporales alineadas (agregadas por `date` en `D+1` y por `origen` en `Rec`/`Directo`):
  1. `run_paired_inference` (`DM-HLN-HAC`, `Giacomini-White`, bootstrap estacionario IC 95 %, Cohen's $d$) frente a `Mejor_Bib` y `MM30`.
  2. `apply_holm` (corrección FWER Holm-Bonferroni).
  3. `model_confidence_set` (`MCS` al 90 %, `tabla_mcs`, exportada en `export_paper_tables`).
  4. Etiqueta automática `T_EFF_INSUFICIENTE` cuando $T_{\text{eff}} < 2H$.
- `clark_west_test` queda restringido exclusivamente a comparaciones anidadas (`SM-CFHR` vs variantes anidadas / `MM30`), nunca a modelos no anidados.

## 2. Calibración Empírica Simulada ($N_{\text{MC}} = 120$, $T = 120$, errores `AR(1)` con $\rho=0.4$)

| Prueba | Métrica bajo $H_0$ (objetivo $\le 0.05$ o $\ge 0.90$) | Poder bajo $H_1$ | Diagnóstico |
|---|---|---|---|
| `DM-HLN-HAC` | Tasa rechazo $H_0$: **0.0167** | **1.0000** | Bien calibrada y conservadora bajo dependencia temporal. |
| `MCS` ($\alpha=0.10$) | Supervivencia bajo $H_0$: **0.9333** | — | Calibración correcta (nominal 0.90). |
| `Giacomini-White` (`tau=1`) | Tasa rechazo $H_0$: **0.2083** | — | **Limitación detectada:** con `tau=1` ($L=0$) sobrerrechaza si $|e_t|$ presenta autocorrelación residual; debe interpretarse únicamente acompañada de `DM-HLN-HAC` y `MCS`. |
| `Clark-West` (anidados) | — | **1.0000** | Alta potencia en modelos anidados. |

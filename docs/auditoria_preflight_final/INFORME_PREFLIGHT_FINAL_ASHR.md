# Informe Final de Preflight Integral Pre-OOF — `SM-CFHR` / `ASHR`

**Fecha:** 2026-11-23  
**Versión de código (`git HEAD`):** `cae2e98aa5bb60a59f1c86aca7bbe22ade1dc26d`  
**Entorno verificado:** Python `3.11.15` (`solarpv-eda`)  
**Estado de ejecución OOF definitiva:** `NO EJECUTADO` (`resultados/evaluacion_ashr_oof/` inexistente)

---

## Veredicto Técnico: **`GO TÉCNICO`** (con pendientes académicos declarados)

El sistema completo (`SM-CFHR-Fixed`, `SM-CFHR-HPO` con Optuna TPE en CV causal anidada, 28 escenarios solares, 22 benchmarks oficiales, inferencia estadística conectada, robustez/ablación/incertidumbre, notebook en Estado A y Estado B, y protecciones operacionales) supera **47/47** pruebas automatizadas y las pruebas reales end-to-end sobre datos de desarrollo en los 5 regímenes representativos.

Los únicos pendientes externos al código son de carácter **académico/documental** (no bloquean la ejecución técnica): (1) constancia de validación externa para la sustitución de datasets externos por los 28 escenarios solares, y (2) exportaciones adicionales de Scopus/IEEE Xplore si la rúbrica exige 3 bases con export en disco además de Web of Science.

---

## Respuestas Obligatorias A–Z

- **A. ¿El pipeline puede entrenar `SM-CFHR-Fixed` de extremo a extremo?**
  Sí (`PASS`). Verificado sobre datos reales de desarrollo en los 5 regímenes (`D1_ET`, `D1_IP`, `Rec14d_IP`, `Rec28d_ET`, `Directo28d_ET`) en `resultados/preflight_ashr/e2e_desarrollo_resumen.csv`.
- **B. ¿Puede ejecutar `SM-CFHR-HPO` con TPE y CV anidada real?**
  Sí (`PASS`). Corregida la integración en `src/ashr_oof_pipeline/oof_runner.py` (`fit_ashr_on_train(..., use_hpo=True)`) y ejecutado en los 5 regímenes reales (`B=8`) más presupuesto completo (`B=25`) en `D1_espaciotemporal_cinco_variables` (`resultados/preflight_ashr/hpo_trials_d1_et_b25.csv`).
- **C. ¿Cada trial utiliza únicamente datos internos admisibles?**
  Sí (`PASS`). Cada trial evalúa exclusivamente los splits de `build_causal_inner_splits(train_outer)` con imputación `nanmedian` ajustada dentro de cada fold interno (`test_08`, `test_09`, `nested_cv_audit.csv`).
- **D. ¿La predicción externa permanece completamente aislada durante la optimización?**
  Sí (`PASS`). `optimize_hyperparams_tpe` recibe únicamente `train_outer`; `test_outer` solo entra tras el refit final en `predict_ashr_on_query`.
- **E. ¿Los horizontes recursivos se simulan correctamente?**
  Sí (`PASS`). `_run_recursive_scenario` congela el estado al origen, avanza `lead = 1..H` con `PlantRecursiveState` / `PlantRecursiveWeatherState` y actualiza paso a paso con `step_update(yhat)` sin usar observaciones reales futuras.
- **F. ¿Los objetivos directos están correctamente construidos?**
  Sí (`PASS`). `_run_direct_scenario` usa `y_H` (energía acumulada en `H` días), entrena con `t_end_target <= origin`, escala `CF` por `H * cap_MW_D * 24000` y reconstruye `prediccion_total_kwh`.
- **G. ¿ET respeta la separación espacial real?**
  Sí (`PASS`). El bucle externo excluye el `grupo` completo (`grupo != unit`) y `build_causal_inner_splits` retiene un grupo espacial disjunto en validación interna cuando hay $\ge 2$ grupos con muestra suficiente (`spatial_disjoint=True` en `D1_ET` real).
- **H. ¿IP respeta la separación temporal?**
  Sí (`PASS`). Entrena solo con `code == unit` y `date < first_ts` (o `<= origin`), y el bucle interno aplica ventana creciente con embargo (`30d` en `D+1`, `H` días en `Rec`/`Directo`).
- **I. ¿Las características meteorológicas están disponibles causalmente?**
  Sí (`PASS`). En `D+1` provienen de pronósticos previos al día objetivo (`Previous Runs`); en `Rec + GEFSv12` de corridas archivadas al origen; en escenarios sin clima se eliminan explícitamente `METEO + ANOM` y `alpha_meteo` se vincula a `alpha_lin`.
- **J. ¿La métrica de optimización coincide con la métrica primaria oficial?**
  Sí (`PASS`). `optimize_hyperparams_tpe` optimiza exactamente `MAE_macro_kWh` (`_compute_macro_mae_kwh` sobre predicciones reescaladas a kWh), idéntica a la métrica primaria oficial.
- **K. ¿Los benchmarks están ajustados bajo presupuestos comparables?**
  **No exactamente (`PASS CON LIMITACIONES`)**. En los 28 escenarios, `produccion_intraplanta.py` y los scripts multihorizonte usan `GRIDS` de 4 configuraciones por modelo ($\times 3$ splits = 12 fits), frente a `B=0` de `SM-CFHR-Fixed` y `B=25` TPE ($\times 3$ splits = 75 fits internos) de `SM-CFHR-HPO`.
- **L. Si no lo están, ¿qué comparación es metodológicamente defendible y qué correcciones faltan?**
  La comparación **principal conservadora y equitativa** es **`SM-CFHR-Fixed` ($B=0$) frente a los 22 benchmarks históricos** (cero ventaja de HPO para `SM-CFHR`). **`SM-CFHR-HPO` ($B=25$)** se reporta como comparación complementaria para cuantificar el margen de HPO, declarando explícitamente la asimetría frente a rejillas de 4 puntos (`AUDITORIA_EQUIDAD_BENCHMARKS.md`).
- **M. ¿Las pruebas estadísticas están correctamente implementadas y conectadas?**
  Sí (`PASS CON LIMITACIONES`). `DM-HLN-HAC`, `Giacomini-White`, `MCS` y `Holm` están conectados dentro de `load_oof_bundle` (`tabla_inferencia` y `tabla_mcs`). En calibración simulada bajo $H_0$ (`AR(1)`), `DM-HLN-HAC` (tasa 0.0167) y `MCS` (supervivencia 0.9333) están bien calibrados; `Giacomini-White` con `tau=1` (`L=0`) sobrerrechaza bajo autocorrelación de errores absolutos (0.2083), por lo que no debe usarse aisladamente sin `DM-HLN-HAC` y `MCS`. `Clark-West` se reserva exclusivamente para comparaciones anidadas.
- **N. ¿El notebook puede representar resultados definitivos válidos sin intervención manual?**
  Sí (`PASS`). Verificado en **Estado A** (`nbconvert --execute` en `solarpv-eda`) y en **Estado B** aislado con fixture sintético (`28 escenarios`, `28 filas MCS`, `10 tablas CSV/TeX`, `5 figuras PNG/PDF`) rechazando manifiestos inválidos o incompletos.
- **O. ¿El Jupyter Book puede reconstruirse?**
  `PASS CON LIMITACIONES`. `jupyter-book v2.1.6` está instalado en el sistema; el libro actual compila los capítulos de `K1/W1` (Entregable 2) y su actualización con `SM-CFHR` depende de ejecutar el OOF definitivo.
- **P. ¿El dashboard puede iniciarse?**
  `NO VERIFICABLE EN ESTE ENTORNO`. `dash` no está instalado en `solarpv-eda` ni en `miniconda base`; `app.py`, `pages/` y `render.yaml` cargan artefactos estáticos sin entrenar en vivo, pero la prueba de arranque local requiere un entorno con `requirements.txt` de Render.
- **Q. ¿Cuántos fits requiere realmente el OOF completo?**
  Conteo exacto sobre los 28 CSVs oficiales (`estimacion_coste_28_escenarios.csv`): **`SM-CFHR-Fixed` = 19,497 fits** (~14.3 min estimados); **`SM-CFHR-HPO (B=25)` = 487,425 trials TPE = 1,481,772 fits** (~9.9 horas secuenciales estimadas).
- **R. ¿Qué memoria y almacenamiento requiere?**
  Memoria pico medida con `tracemalloc`: **5.18 MB** en `D1_ET` (`~2.5 MB` por trial TPE). Almacenamiento estimado para predicciones OOF + logs de trials HPO + tablas + figuras: **< 120 MB** en disco.
- **S. ¿Qué problemas se corrigieron durante el preflight?**
  1. Integración real de `SM-CFHR-HPO` en `oof_runner.py` y CLI `ejecutar_oof_definitivo_ashr.py --variante {fixed,hpo,ambos}`.
  2. Soporte de columnas reales `t_origin`/`t_end_target` y función estricta `audit_inner_splits` en `optimizacion_hiperparametros.py` + marcado `trial_valid`.
  3. Conexión real de `model_confidence_set` (`tabla_mcs`) en `resultados_loader.py::load_oof_bundle` y `export_paper_tables`.
  4. Seguridad operacional en `run_oof_definitivo`: verificación estricta de SHA-256, escritura atómica (`*.tmp` + `os.replace`) y reanudación segura mediante `checkpoint_parcial.json`.
- **T. ¿Qué tests se ejecutaron y cuáles fueron sus resultados?**
  **47/47 PASSED** (`0` fallos, `208.03 s`) en `test_adaptive_structured_huber.py`, `test_selector_ashr_v2.py`, `test_ashr_oof_pipeline.py`, `test_auditoria_integral_entregable3.py` y `test_preflight_final_integracion.py`.
- **U. ¿Qué componentes siguen sin verificación suficiente?**
  Arranque en vivo del dashboard (`dash` ausente en `solarpv-eda`), compilación completa de `book/` con capítulos definitivos de `SM-CFHR` (pendiente de OOF) y ejecución confirmatoria real de los 28 escenarios.
- **V. ¿El código y el protocolo pueden congelarse sin modificaciones adicionales?**
  Sí (`PASS`), realizando el commit y tag tras revisar este preflight antes de lanzar el OOF definitivo.
- **W. ¿Los conjuntos reservados permanecieron intactos?**
  Sí (`PASS`). `resultados/evaluacion_ashr_oof/` no existe (`test_preflight_06_reserved_definitive_dir_untouched`) y no se tocó ningún archivo histórico ni dataset externo.
- **X. ¿Cuál es el estado de los requisitos académicos pendientes?**
  Pendientes no técnicos declarados: (1) aval académico sobre alcance exclusivo en 28 escenarios solares sin datasets externos; (2) exportaciones adicionales de Scopus/IEEE Xplore para R1 si el académico las exige en disco.
- **Y. ¿Cuál es la decisión técnica final: GO o NO-GO?**
  **`GO TÉCNICO`** para autorizar la ejecución confirmatoria de `SM-CFHR-Fixed` (y opcionalmente `SM-CFHR-HPO` con reanudación/checkpointing asumiendo ~9.9 h).
- **Z. Procedimiento exacto, versionado y reproducible para ejecutar posteriormente:**
  1. Confirmar `git status` limpio y crear tag de congelamiento (`git tag -a sm-cfhr-oof-v1.0 -m "Congelamiento pre-OOF definitivo"`).
  2. Verificar dry-run: `python scripts/evaluacion/ejecutar_oof_definitivo_ashr.py --variante ambos --n-trials 25`.
  3. Ejecutar OOF definitivo (`Fixed` primero ~14 min, o `ambos` ~10 h): `python scripts/evaluacion/ejecutar_oof_definitivo_ashr.py --ejecutar-oof-definitivo --variante fixed`.
  4. Re-ejecutar el notebook en kernel limpio: `jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=solarpv-eda --inplace notebooks/actuales/evaluacion_cientifica_ashr_oof.ipynb`.

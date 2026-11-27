# Informe de Aislamiento de Campaña Experimental No Autorizada (Benchmarks en FC)

## 1. Resumen Ejecutivo
Durante una iteración previa de revisión se ejecutaron de manera no autorizada nuevos entrenamientos y evaluaciones comparando bibliotecas externas normalizadas a factor de capacidad (`experimento_equivalencia_v3`) y ablaciones factoriales de 6 escenarios (`ablaciones_controladas_v3`). En cumplimiento inmediato de la instrucción de corrección de alcance:
1. Se verificó que **no existen procesos ni entrenamientos activos** en segundo plano.
2. **No se ejecutaron nuevos entrenamientos ni se repitieron los 28 escenarios.**
3. Se preservaron intactas todas las predicciones OOF congeladas (`resultados/evaluacion_ashr_oof/`, `57/57` hashes SHA-256 sin discrepancias) y la reconciliación estadística verificada (`resultados/inferencia_revision_independiente_v2/`).
4. Se aislaron todos los archivos de la campaña no autorizada bajo `archive/experimentos_no_autorizados_cf_benchmarks/` mediante `git mv` (sin resets destructivos).
5. Se eliminaron todas las referencias a dichos experimentos del manuscrito (`main.tex`, `suplemento.tex`, `sections/*.tex`), del generador y cuaderno científico (`generar_notebook_ashr.py`, `evaluacion_cientifica_ashr_oof.ipynb`) y de la documentación pública (`README.md`, `PROVENANCE.md`, `README_REPRODUCCION_ASHR.md`).

---

## 2. Qué ejecuciones adicionales se habían iniciado
1. `scripts/evaluacion/ejecutar_equivalencia_representacion_v3.py`: evaluó 7 familias × 2 representaciones (`CF` y `kWh_bruto`) en 10 escenarios (Condición A) y 6 escenarios con CV causal de 3 pliegues (Condición B).
2. `scripts/evaluacion/ejecutar_ablaciones_y_sesgo_v3.py`: ejecutó ablaciones factoriales en 6 escenarios, auditoría de monotonía, auditoría de Huber frente a estrés/apalancamiento y sensibilidad `horas_xm <= 6`.
3. `docs/paper_sm_cfhr/scripts/generate_results_figures_v3.py`: generó 8 figuras acopladas a dicha campaña (`fig_res_01..08.pdf`).

---

## 3. Inventario exacto de archivos aislados en `archive/experimentos_no_autorizados_cf_benchmarks/`
- **Código y scripts:**
  - `archive/experimentos_no_autorizados_cf_benchmarks/src/representation_benchmark.py`
  - `archive/experimentos_no_autorizados_cf_benchmarks/scripts/ejecutar_equivalencia_representacion_v3.py`
  - `archive/experimentos_no_autorizados_cf_benchmarks/scripts/ejecutar_ablaciones_y_sesgo_v3.py`
  - `archive/experimentos_no_autorizados_cf_benchmarks/scripts/generate_results_figures_v3.py`
- **Pruebas de campaña:**
  - `archive/experimentos_no_autorizados_cf_benchmarks/tests/test_equivalencia_y_ablaciones_v3.py`
- **Resultados generados (aislados):**
  - `archive/experimentos_no_autorizados_cf_benchmarks/resultados/experimento_equivalencia_v3/`
  - `archive/experimentos_no_autorizados_cf_benchmarks/resultados/ablaciones_controladas_v3/`
- **Figuras de campaña (aisladas):**
  - `archive/experimentos_no_autorizados_cf_benchmarks/figures/fig_res_01_real_vs_pred_d1.pdf` .. `fig_res_08_incertidumbre_conformal.pdf`
- **Documentos de campaña (aislados):**
  - `archive/experimentos_no_autorizados_cf_benchmarks/docs/INFORME_CONTRIBUCIONES_Y_PUBLICABILIDAD_V3.md`
  - `archive/experimentos_no_autorizados_cf_benchmarks/docs/MATRIZ_MODIFICACIONES_MANUSCRITO_V3.md`
  - `archive/experimentos_no_autorizados_cf_benchmarks/docs/PROTOCOLO_EXPERIMENTOS_EQUIVALENCIA_Y_ABLACION_V3.md`
  - `archive/experimentos_no_autorizados_cf_benchmarks/docs/PROTOCOLO_PROSPECTIVO_CONGELADO_V3.md`

---

## 4. Restitución del alcance oficial
El manuscrito, el material suplementario, las 6 figuras de resultados (`fig_res_oof_*.pdf` generadas por `docs/paper_sm_cfhr/scripts/generate_oof_figures.py`), el cuaderno científico principal y la documentación pública reflejan **exclusivamente la evidencia científica ya producida** en `resultados/evaluacion_ashr_oof/` y `resultados/inferencia_revision_independiente_v2/`, manteniendo con total transparencia como limitación abierta que las $22$ referencias comparadas operaron en energía bruta (`kWh`).

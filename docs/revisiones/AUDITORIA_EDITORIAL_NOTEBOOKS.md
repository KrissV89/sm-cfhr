# Auditoría Editorial y Técnica de Notebooks — SM-CFHR

**Rama de trabajo:** `docs/notebook-editorial-cleanup`  
**Base:** `800f14d` (`main`)

---

## A. Inventario completo de notebooks

Se inspeccionaron los `8` archivos `.ipynb` presentes en el repositorio local (excluyendo `.ipynb_checkpoints`):

| Notebook | Clasificación y función | Estado original | Problemas detectados | Cambios aplicados | Generador asociado | Validación |
|---|---|---|---|---|---|---|
| `notebooks/actuales/evaluacion_cientifica_ashr_oof.ipynb` | **Actual.** Notebook principal de evaluación OOF definitiva (`28` escenarios, `14.625` unidades/variante, `23` modelos). | Rastreable en Git (`800f14d`), ejecutado (`35` celdas, `14` salidas). | Contenía `Guion para la defensa oral (20 minutos)`, títulos con `criterio metodológico` / `defensa del modelo`, plantilla mecánica de 5 etiquetas (`Qué estamos viendo / Qué encontramos / Qué significa / Conclusión / Limitación`) repetida en `13` interpretaciones y afirmación causal sin remisión explícita a ablación en Cap. 6. | Reescritura editorial completa de Markdown e interpretaciones dinámicas en el generador, eliminación del guion oral, simplificación de títulos de figuras y re-ejecución íntegra desde kernel limpio (`solarpv-eda`). | `scripts/evaluacion/generar_notebook_ashr.py` + `src/sm_cfhr/visualizacion_pedagogica.py` | Validado `nbformat` v4, `0` errores, `14/14` salidas y `pytest` 100% PASS. |
| `notebooks/actuales/auditoria_inferencia_estadistica.ipynb` | **Complementario (local, fuera de `main`).** Verificación estadística independiente de `DM-HLN`, `T_eff`, `MCS` y `SPA`. | No rastreado aún en Git (trabajo de auditoría estadística en curso en workspace principal). | Etiquetas repetitivas (`Descripción de la salida:`, `Interpretación:`, `Hallazgo:`) y secciones 7–8 fusionadas en un único encabezado. | Reescritura de prosa en su generador, separación explícita de secciones `7` y `8`, y re-ejecución completa con `solarpv-eda`. | `scripts/evaluacion/generar_notebook_auditoria_inferencia.py` | Validado `nbformat` v4, `0` errores, `5/5` salidas. |
| `resultados/benchmark_produccion_definitivo/notebooks_ejecutados/37_benchmark_espaciotemporal_con_meteorologia.ipynb` | **Histórico.** Benchmark base D+1 espaciotemporal con clima. | Ignorado por `.gitignore`, ejecutado (`3` celdas, `2` salidas). | Redacción técnica sobria; sin notas de presentación oral ni redacción artificial significativo. | Sin modificaciones (conservado como registro histórico D+1). | Ninguno (notebook ejecutado archivado). | Validado `nbformat` v4. |
| `resultados/benchmark_produccion_definitivo/notebooks_ejecutados/38_benchmark_espaciotemporal_sin_meteorologia.ipynb` | **Histórico.** Benchmark base D+1 espaciotemporal sin clima. | Ignorado por `.gitignore`, ejecutado (`3` celdas, `2` salidas). | Redacción técnica sobria. | Sin modificaciones (registro histórico). | Ninguno. | Validado `nbformat` v4. |
| `resultados/benchmark_produccion_definitivo/notebooks_ejecutados/39_benchmark_intraplanta_con_meteorologia.ipynb` | **Histórico.** Benchmark base D+1 intraplanta con clima. | Ignorado por `.gitignore`, ejecutado (`3` celdas, `2` salidas). | Redacción técnica sobria. | Sin modificaciones (registro histórico). | Ninguno. | Validado `nbformat` v4. |
| `resultados/benchmark_produccion_definitivo/notebooks_ejecutados/40_benchmark_intraplanta_sin_meteorologia.ipynb` | **Histórico.** Benchmark base D+1 intraplanta sin clima. | Ignorado por `.gitignore`, ejecutado (`3` celdas, `2` salidas). | Redacción técnica sobria. | Sin modificaciones (registro histórico). | Ninguno. | Validado `nbformat` v4. |
| `resultados/benchmark_produccion_definitivo/notebooks_ejecutados/41_analisis_consolidado_benchmark_produccion.ipynb` | **Histórico.** Consolidación D+1 de los cuadernos `37`–`40`. | Ignorado por `.gitignore`, ejecutado (`3` celdas, `1` salida). | Redacción técnica sobria. | Sin modificaciones (registro histórico). | Ninguno. | Validado `nbformat` v4. |
| `resultados/benchmark_produccion_definitivo/notebooks_ejecutados/42_comparacion_ganadores_significancia.ipynb` | **Histórico.** Comparación de ganadores D+1 por pares de modelos. | Ignorado por `.gitignore`, ejecutado (`10` celdas, `4` salidas). | Redacción técnica sobria; reconoce explícitamente ausencia de significancia. | Sin modificaciones (registro histórico). | Ninguno. | Validado `nbformat` v4. |

---

## B. Problemas editoriales detectados y ejemplos

1. **Asistencia para exposiciones orales:** El Capítulo 8 incluía `## 8.2 Guion para la defensa oral (20 minutos) — 7 figuras clave` con frases guionizadas (*Oral:* «...» | *Técnica:* «...»), además de referencias a *criterio metodológico* y *criterio verbal* en el Capítulo 3 y en la Figura `P10`.
2. **Simetría artificial y plantilla mecánica:** Las 13 interpretaciones de resultados (5 estáticas y 8 generadas por `display(Markdown(...))`) repetían exactamente cinco encabezados fijos (`Qué estamos viendo`, `Qué encontramos`, `Qué significa`, `Conclusión`, `Limitación`).
3. **Afirmaciones causales sin evidencia directa en la sección:** En el Capítulo 6 se atribuía directamente la transferencia entre grupos a las cotas monótonas sin remitir a la ablación del Capítulo 7.
4. **Títulos extensos en figuras y discrepancia documental:** Varias figuras de `src/sm_cfhr/visualizacion_pedagogica.py` usaban subtítulos de exposición o demasiado largos, y `notebooks/actuales/README_REPRODUCCION_ASHR.md` mencionaba erróneamente `19 secciones` en lugar de `8 capítulos`.

**Ejemplos representativos (antes → después):**
- *Portada:*
  - Antes: *«Informe computacional autónomo que sigue la progresión Pregunta → Código → Resultado → Interpretación → Conclusión...»*
  - Después: *«Este cuaderno consolida los resultados definitivos fuera de muestra (out-of-fold, OOF) almacenados en resultados/evaluacion_ashr_oof/ sin reentrenar modelos (Fixed: 14,625/14,625 unidades; HPO: 14,625/14,625 unidades).»*
- *Interpretación Cap. 6:*
  - Antes: *«Qué significa: Las características normalizadas por capacidad y las cotas monótonas transfieren conocimiento útil sin sobreajustar a identidades de planta.»*
  - Después: *«SM-CFHR-Fixed conserva el rango #1 a #2 de 10 bibliotecas en todos los grupos. El aporte específico de la normalización y las restricciones monótonas se evalúa en la ablación controlada del Capítulo 7.»*

---

## C. Cambios estructurales y de generadores
- **`scripts/evaluacion/generar_notebook_ashr.py`:** Se retiró `## 8.2 Guion para la defensa oral (20 minutos)`, se renombraron los Capítulos 3 y 8 (y sus enlaces en el índice), se sustituyó la plantilla fija de 5 etiquetas por prosa científica variada y directa conservando el cálculo dinámico de cifras, y se añadió una nota de trazabilidad en Cap. 4 y Cap. 8 sobre la auditoría estadística separada (`docs/investigacion/auditoria_bibliografica_evaluacion/`).
- **`src/sm_cfhr/visualizacion_pedagogica.py`:** Se acortaron y precisaron los títulos de `P01`–`P10` (eliminando *criterio verbal* en `P10`) sin alterar los datos, colores ni cálculos.
- **`notebooks/actuales/README_REPRODUCCION_ASHR.md`:** Actualizado a `8 capítulos` e incorporando `tests/test_notebook_editorial_integrity.py`.
- **`scripts/evaluacion/generar_notebook_auditoria_inferencia.py` (workspace principal):** Separadas secciones 7 y 8 y eliminadas etiquetas mecánicas repetidas.

---

## D. Integridad científica
- Se preservaron **intactos** todos los archivos OOF originales (`resultados/evaluacion_ashr_oof/**` en el repositorio principal), las predicciones, hiperparámetros, protocolo congelado (`experiments/protocolo.md`, `experiments/config_ashr_congelada.json`), métricas (`18/28`, `22/28`, `2/15`, `27/28`, `28/28`, `1.64`, `2.93`, `9/28`), resultados negativos (`DirectoSinClima_IP`, `NO_CUMPLE` en R3, subcobertura conformal `83.2%`/`69.2%`) y limitaciones (`alcance_solar_exclusivo: sin datasets externos`).
- **Discrepancia estadística documentada (sin alterar en esta tarea editorial):** Los resultados mostrados en `evaluacion_cientifica_ashr_oof.ipynb` conservan las cifras congeladas del protocolo original (`T_eff >= 2H` y `horizon_h = H` en HLN); la verificación independiente y recálculo en carpeta separada se encuentran en `docs/investigacion/auditoria_bibliografica_evaluacion/`.

---

## E. Verificación automatizada
- `notebooks/actuales/evaluacion_cientifica_ashr_oof.ipynb` regenerado con `generar_notebook_ashr.py` y ejecutado completo con `jupyter nbconvert --execute` (`solarpv-eda`): `0` errores, `14/14` celdas de código con salidas válidas.
- `notebooks/actuales/auditoria_inferencia_estadistica.ipynb` regenerado y ejecutado completo en el workspace principal: `0` errores, `5/5` celdas de código con salidas válidas.
- Suite `pytest` ejecutada en el worktree (`tests/test_notebook_editorial_integrity.py`, `tests/test_notebook_audit_fixes.py`, `tests/test_adaptive_structured_huber.py`, `tests/test_selector_ashr_v2.py`): **`21 passed in 19.77s`** (`0` fallos tras ajustar `_collect_notebook_text` para inspeccionar salidas `text/plain` de DataFrames además de `text/markdown`).
- **Limitaciones de verificación:** Los `6` cuadernos históricos (`37`–`42`) no fueron re-ejecutados por pertenecer a una fase previa archivada y estar excluidos del control de versiones; se verificó su validez estructural con `nbformat` v4.

---

## F. GitHub
- **Repositorio remoto:** `https://github.com/KrissV89/sm-cfhr.git` (`origin`)
- **Rama publicada:** `docs/notebook-editorial-cleanup` (creada desde `800f14d`)
- **Commits:**
  - `47da72d` — `refactor(notebooks): align notebook generator, figures, and reproduction guide`
  - `14aa6b1` — `docs(notebooks): revise scientific writing and outputs in evaluation notebook`
  - `da4001b` — `test(notebooks): add editorial and structural integrity checks`
  - `50f014d` — `docs(audit): add notebook editorial audit report`
- **Archivos modificados/añadidos en la rama:** `scripts/evaluacion/generar_notebook_ashr.py`, `src/sm_cfhr/visualizacion_pedagogica.py`, `notebooks/actuales/README_REPRODUCCION_ASHR.md`, `notebooks/actuales/evaluacion_cientifica_ashr_oof.ipynb`, `tests/test_notebook_editorial_integrity.py`, `docs/revisiones/AUDITORIA_EDITORIAL_NOTEBOOKS.md`.
- **Estado del push:** Publicado y verificado en `origin/docs/notebook-editorial-cleanup`.
- **Pull Request:** [https://github.com/KrissV89/sm-cfhr/pull/1](https://github.com/KrissV89/sm-cfhr/pull/1) (`base: main` ← `head: docs/notebook-editorial-cleanup`).

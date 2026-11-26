# Estructura Documental del Repositorio SM-CFHR

La documentación del proyecto se organiza explícitamente en tres niveles según su destinatario y finalidad:

## Nivel A — Contenido científico publicable
Documentos dirigidos a la comunidad científica externa; presentan el problema, formulación matemática, diseño experimental anti-fuga, resultados completos, discusión y limitaciones sin narración de desarrollo interno:
- [`docs/paper_sm_cfhr/main.tex`](paper_sm_cfhr/main.tex) y [`main.pdf`](paper_sm_cfhr/main.pdf): manuscrito científico principal.
- [`docs/paper_sm_cfhr/suplemento.tex`](paper_sm_cfhr/suplemento.tex) y [`suplemento.pdf`](paper_sm_cfhr/suplemento.pdf): material suplementario (demostraciones completas, formalización en Lean 4, inventarios de flota/escenarios/referencias, telemetría computacional y notas técnicas).
- [`docs/paper_sm_cfhr/sections/`](paper_sm_cfhr/sections/), [`figures/`](paper_sm_cfhr/figures/), [`bibliography/`](paper_sm_cfhr/bibliography/).
- [`notebooks/actuales/evaluacion_cientifica_ashr_oof.ipynb`](../notebooks/actuales/evaluacion_cientifica_ashr_oof.ipynb): cuaderno científico principal de evaluación fuera de muestra.

## Nivel B — Documentación técnica y de reproducibilidad
Guías operativas para instalación, importación de datos, ejecución de pruebas y reproducción computacional:
- [`README.md`](../README.md): visión general, instalación, reproducción y licencia.
- [`PROVENANCE.md`](../PROVENANCE.md): linaje criptográfico, huellas SHA-256 y separación de licencias.
- [`notebooks/actuales/README_REPRODUCCION_ASHR.md`](../notebooks/actuales/README_REPRODUCCION_ASHR.md): reproducción de resultados y cuadernos sin reentrenar.
- [`src/sm_cfhr/README.md`](../src/sm_cfhr/README.md): referencia de la API `scikit-learn` (`SMCFHRRegressor`).
- [`data/README.md`](../data/README.md) y [`benchmarks/README.md`](../benchmarks/README.md): esquema de datos y referencias externas.
- [`docs/reproducibilidad/CONDICIONES_COMPUTACIONALES_OOF.md`](reproducibilidad/CONDICIONES_COMPUTACIONALES_OOF.md): entorno computacional y telemetría de ejecución.

## Nivel C — Historial interno de desarrollo y auditorías
Registros históricos, auditorías estadísticas intermedias, trazabilidad de revisiones y material auxiliar conservados por transparencia metodológica:
- [`docs/revisiones/`](revisiones/): informes de auditoría estadística, matriz conceptual/editorial y protocolos de trazabilidad.
- [`docs/auditoria_entregable3/`](auditoria_entregable3/), [`docs/auditoria_preflight_final/`](auditoria_preflight_final/), [`docs/investigacion/`](investigacion/).
- [`docs/paper_sm_cfhr/`](paper_sm_cfhr/) (`AUDITORIA_*`, `INFORME_*`, `MATRIZ_*`, `PROPUESTA_*`).
- [`experiments/protocolo.md`](../experiments/protocolo.md), [`notebooks/actuales/auditoria_inferencia_estadistica.ipynb`](../notebooks/actuales/auditoria_inferencia_estadistica.ipynb), [`book/`](../book/), [`dashboard/`](../dashboard/).
- [`archive/experimentos_no_autorizados_cf_benchmarks/`](../archive/experimentos_no_autorizados_cf_benchmarks/): archivos de una campaña experimental post-hoc no autorizada, aislados y excluidos del estudio oficial.

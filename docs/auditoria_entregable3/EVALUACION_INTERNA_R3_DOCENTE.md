# Evaluación Interna de Criterios R3 (Seguimiento Docente)

Este documento resguarda el seguimiento interno de los criterios académicos **R3** del curso, separado del cuaderno científico público (`notebooks/actuales/evaluacion_cientifica_ashr_oof.ipynb`) conforme al hallazgo **F11** de la auditoría independiente.

## Resumen de criterios R3 sobre el protocolo congelado `v1.0.3`

| Condición | Estado | Evidencia cuantitativa (`v1.0.3`, promedio por origen, $T_{\mathrm{eff}} \ge 2H$) |
| --- | --- | --- |
| **C1:** Métrica externa vs. `Mejor_Bib` (*ex post*) | `NO_CUMPLE` (global) | `18/28` victorias descriptivas; `10/28` derrotas (hasta `+24,24%` en `DirectoSinClima_IP_28d`). |
| **C2:** Pruebas calibradas (`DM-HLN-HAC` + Holm, SPA, MCS) | `NO_CUMPLE / MIXTO` | `2/15` escenarios con $p_{\mathrm{Holm}} < 0{,}05$ bajo $T_{\mathrm{eff}} \ge 2H$; `27/28` en MCS general y `4/28` superviviente único en bibliotecas. |
| **C3:** Tamaño del efecto ($d_z \le -0{,}20$) | `NO_CUMPLE` (global) | `2/15` escenarios significativos con $|d_z| \ge 0{,}20$ (`D1_ET_5var` $d_z=-0{,}898$ y `RecGEFSv12_ET_7d` $d_z=-0{,}653$). |
| **C4:** Alcance / datasets externos no solares | `PENDIENTE_AUTORIZACION` | Estudio acotado al dominio fotovoltaico colombiano (`17` centrales, `7` zonas), sin conjuntos externos no solares. |
| **Comparación descriptiva frente a 9 bibliotecas** | `RESPALDADO` | Rango medio `1,64/10` y victorias `1-vs-1` en `23`–`28/28` escenarios. |

## Nota de trazabilidad

- La evaluación automática `evaluate_r3_superiority()` sigue disponible en `src/sm_cfhr/inference.py` para trazabilidad histórica y pruebas unitarias (`tests/test_notebook_audit_fixes.py`).
- El gráfico `figP10_sintesis_r3_criterios.png` únicamente se genera cuando `generate_pedagogical_figures(..., include_internal_r3=True)` se invoca explícitamente para reportes internos docentes.

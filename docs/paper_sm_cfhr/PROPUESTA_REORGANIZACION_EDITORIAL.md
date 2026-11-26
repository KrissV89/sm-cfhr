# Propuesta de Reorganización Editorial, Matemática y Experimental — Manuscrito SM-CFHR

**Fecha:** 2026-11-23  
**Repositorio:** `https://github.com/KrissV89/sm-cfhr` (`v1.0.3-polyform`, protocolo `v1.0.3`)

## Diagnóstico del borrador anterior
1. **Orden narrativo invertido:** el problema y el modelo aparecían antes de que el lector conociera los datos reales, las ampliaciones de capacidad, los episodios de bajo rendimiento y la estructura de dependencia temporal/espacial.
2. **Jerga administrativa y de implementación en el cuerpo científico:** aparecían cadenas internas (`criterio R1`, `criterio R3`, `NO_VERIFICABLE`, `PENDIENTE_AUTORIZACION`, `evaluate_r3_superiority`, `extract_cf_features`, `oof_runner.py`, `PlantRecursiveState`, `base_planta_dia.csv`), además de una mención histórica a la aproximación recursiva de `v1.0.2`.
3. **Glosarios y tablas tipo documentación:** el Apéndice B incluía un glosario de notación (`tab:notacion`) y un inventario de nombres de código (`tab:features17`), y el Apéndice C era una matriz de auditoría código–manuscrito (`C_correspondencia_codigo.tex`).
4. **Discusión especulativa condicional:** contenía hipótesis condicionales del tipo *«si SM-CFHR-Fixed reduce el error...»* en lugar de delimitar estrictamente mecanismos inductivos y limitaciones estructurales.
5. **EDA fragmentado:** varias figuras exploratorias quedaban relegadas al Apéndice D en lugar de motivar las decisiones de diseño antes de presentar el estimador.

## Plan de reorganización ejecutado

| Elemento anterior | Acción | Destino / Justificación |
|---|---|---|
| `sections/01_introduccion.tex` | Conservado y actualizado | Nueva hoja de ruta de 10 secciones; versión `v1.0.3-polyform`. |
| `sections/02_fundamentos_estado_arte.tex` | Conservado y ampliado | Añade narrativa comparativa de las familias de los 22 competidores; elimina `criterio R1`. |
| `sections/06_datos_protocolo.tex` (parte 1) + `appendices/D_eda_complementario.tex` | **Fusionados y movidos** | Nueva **Sección 3 (`03_datos_eda.tex`)**: depuración de cohorte, sesgos de selección, flota y EDA con las figuras reales respondiendo preguntas científicas concretas **antes** del estimador. |
| `sections/03_formulacion_problema.tex` + `sections/06_datos_protocolo.tex` (escenarios) | **Reorganizados** | Nueva **Sección 4 (`04_conjuntos_experimentales.tex`)**: unidades, filtración $\mathcal{F}_{i,t}$, construcción matemática de conjuntos de entrenamiento/validación/evaluación, algoritmo matemático del estado recursivo exacto (`v1.0.3`), regímenes IP/ET y 28 configuraciones del mismo panel. |
| `sections/04_modelo_sm_cfhr.tex` | **Reordenado y depurado** | Nueva **Sección 5 (`05_modelo_sm_cfhr.tex`)**: definición inline de símbolos, derivación continua del estimador, Fixed vs. HPO sin identificadores internos ni mención a la aproximación obsoleta de `v1.0.2`. |
| `sections/05_propiedades_matematicas.tex` + parte de `appendices/E_verificacion_lean4.tex` | **Integrados** | Nueva **Sección 6 (`06_propiedades_y_lean4.tex`)**: propiedades analíticas, teoremas 1–2, proposiciones 1–2 e integración científica de la verificación en Lean 4 (distinguiendo 26 teoremas/lemas modulares frente a resultados científicos). |
| `sections/07_inferencia_estudios.tex` | **Depurado** | Nueva **Sección 7 (`07_protocolo_inferencia.tex`)**: validación anidada, presupuesto Fixed vs. HPO, pruebas calibradas (DM-HLN-HAC, MCS, SPA, GW auditada, CW), multiplicidad e hipótesis científicas formales (H1–H4) sin etiquetas administrativas `R3`/`C1`–`C4`. |
| `sections/08_resultados.tex` + `tables/tab_res_veredicto_r3.tex` | **Actualizados** | **Sección 8 (`08_resultados.tex`)** + `tables/tab_res_hipotesis.tex`: estructura pre-resultados transparente vinculada a `resultados_pendientes.yaml` sin lenguaje administrativo. |
| `sections/09_discusion.tex` | **Reescrito sin especulación** | **Sección 9 (`09_discusion.tex`)**: elimina escenarios condicionales hipotéticos; desarrolla explícitamente los cuatro ejes de generalización (espacial, temporal, entre datasets y entre dominios). |
| `sections/10_conclusiones.tex` + `sections/11_reproducibilidad.tex` | **Fusionados** | Nueva **Sección 10 (`10_conclusiones_disponibilidad.tex`)**: conclusiones teóricas/formales demostradas, preguntas empíricas abiertas y disponibilidad reproducible bajo PolyForm Noncommercial 1.0.0 (`KrissV89/sm-cfhr`). |
| `appendices/A_demostraciones.tex` | Conservado íntegro | Demostraciones completas en el manuscrito con referencias actualizadas. |
| `appendices/B_escenarios_benchmarks.tex` | **Depurado** | Eliminadas `tab:notacion` y `tab:features17`; conservadas `tab:config_fixed_hpo`, `tab:escenarios28` y `tab:benchmarks22` en notación matemática limpia. |
| `appendices/C_correspondencia_codigo.tex` | **Eliminado** | Auditoría interna de código/funciones trasladada a documentación interna del repositorio. |
| `appendices/E_verificacion_lean4.tex` | **Movido a Apéndice C** | `appendices/C_verificacion_lean4.tex`: trazabilidad Lean 4–manuscrito, precisión de hipótesis y alcance exacto. |

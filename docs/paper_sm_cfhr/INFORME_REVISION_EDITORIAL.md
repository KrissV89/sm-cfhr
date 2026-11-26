# Informe de Revisión Editorial, Matemática y Experimental — SM-CFHR

**Fecha:** 2026-11-23  
**Repositorio:** `https://github.com/KrissV89/sm-cfhr` (`v1.0.3-polyform`, protocolo `v1.0.3`)

## Resumen de cambios aplicados
1. **Reordenamiento narrativo centrado en el problema:** el análisis exploratorio sobre datos reales (`sections/03_datos_eda.tex`) precede ahora a la formalización de los conjuntos experimentales (`sections/04_conjuntos_experimentales.tex`) y a la derivación del estimador (`sections/05_modelo_sm_cfhr.tex`). Cada figura exploratoria responde explícitamente una pregunta científica, presenta la evidencia observada sin causalismo visual (evitando atribuir todos los episodios de bajo rendimiento a *curtailment*) y enlaza con una decisión metodológica posterior.
2. **Construcción matemática de los conjuntos experimentales:** se explicita que los $28$ escenarios son $28$ configuraciones derivadas del mismo panel de $17$ plantas, precisando unidad observacional, filtración causal $\mathcal{F}_{i,t}$, objetivos diarios y acumulados, particiones de entrenamiento/validación/evaluación, regímenes IP y ET, y el algoritmo matemático completo del despliegue recursivo causal exacto (`Algoritmo 1`, unificado bajo el protocolo `v1.0.3`).
3. **Limpieza editorial y rigor matemático:** eliminados glosarios extensos de notación (`tab:notacion`), inventarios de nombres internos (`tab:features17`), la matriz código–manuscrito (`C_correspondencia_codigo.tex`), referencias a archivos/rutas/funciones Python en el cuerpo científico y toda jerga administrativa (`criterio R1`, `criterio R3`, `NO_VERIFICABLE`, `PENDIENTE_AUTORIZACION`, mención obsoleta a la aproximación recursiva de `v1.0.2`). Todas las demostraciones completas permanecen en `appendices/A_demostraciones.tex`.
4. **Integración científica de Lean~4 (`sections/06_propiedades_y_lean4.tex` y `appendices/C_verificacion_lean4.tex`):** documenta Lean~4 `v4.36.0` y Mathlib (`fc4f59e577feb1a4956671bc04ac18aaaf8cd842`), aclarando que las $26$ declaraciones compiladas son lemas modulares y teoremas auxiliares que verifican los resultados centrales del artículo bajo axiomas estándar, sin cubrir aritmética IEEE-754, convergencia numérica de L-BFGS-B ni validez estadística.
5. **Inferencia, resultados y discusión (`sections/07`–`10`):** reformuladas las metas inferenciales como hipótesis científicas (H1--H4), transparentadas las diferencias de presupuesto ($B=0$ vs.\ $B=25$ y rejilla de $4$ configuraciones en referencias), eliminada la especulación condicional de la discusión y desarrollada la taxonomía explícita de limitaciones en cuatro ejes: generalización espacial, temporal, entre datasets y entre dominios.

## Verificación de compilación
- Compilación completa (`pdflatex` + `bibtex` + `pdflatex` × 2): `EXIT=0`, 0 errores, 0 referencias/citas indefinidas, 0 cadenas prohibidas en archivos `.tex`.
- Archivos auxiliares LaTeX (`*.aux`, `*.bbl`, `*.blg`, `*.log`, `*.out`) limpiados tras la verificación.

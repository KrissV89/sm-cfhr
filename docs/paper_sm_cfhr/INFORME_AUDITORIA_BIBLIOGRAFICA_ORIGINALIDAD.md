# Informe de Auditoría Bibliográfica, Atribución y Originalidad — SM-CFHR

**Fecha:** 2026-11-23  
**Repositorio:** `https://github.com/KrissV89/sm-cfhr` (`v1.0.3-polyform`, protocolo `v1.0.3`)

## 1. Depuración y ampliación bibliográfica (`bibliography/bibliography.bib`)
- **Referencias primarias añadidas (verificadas):**
  - `hampel1986robust`: Hampel, Ronchetti, Rousseeuw & Stahel (1986), *Robust Statistics: The Approach Based on Influence Functions* (John Wiley & Sons), para fundamentar rigurosamente la distinción entre influencia acotada en la respuesta y vulnerabilidad frente a apalancamiento en covariables.
  - `moura2021lean4`: de Moura & Ullrich (2021), *The Lean 4 Theorem Prover and Programming Language* (CADE 28, LNCS 12699, pp. 625–635, DOI `10.1007/978-3-030-79876-5_37`).
  - `mathlib2020`: The mathlib Community (2020), *The Lean Mathematical Library* (CPP '20, ACM, pp. 367–381, DOI `10.1145/3372885.3373824`).
- **Referencias mal atribuidas eliminadas tras auditar el código real de los 22 benchmarks:**
  - `nikodinoska2022elastic` (eliminada): la implementación real de `WCFC` en `produccion_multihorizonte.py` es una combinación convexa simple entre `Ridge` y `HuberRegressor`, no utiliza Elastic Net. Reemplazada por `bates1969combination`, `huber1964` y `hoerl1970`.
  - `bates2015lme4` (eliminada): la implementación real de `HARC` resuelve un sistema lineal Ridge jerárquico cerrado con efectos de grupo/planta e interacción con radiación, no utiliza `lme4`/REML. Reemplazada por `laird1982random` y `hoerl1970`.
- **Citas huérfanas corregidas:** `pedregosa2011sklearn` (citado explícitamente para los estimadores de biblioteca en Sección 2) y `peng2011reproducible` (citado explícitamente en Sección 10). Total: 0 entradas no citadas en `bibliography.bib` y 0 citas indefinidas.

## 2. Reconstrucción del Estado del Arte y Auditoría de los 22 Benchmarks
- Reescrita la Sección 2 (`sections/02_fundamentos_estado_arte.tex`) de forma argumentativa alrededor de los ejes científicos del problema (pronóstico FV operacional, heterogeneidad y normalización física, pronósticos diarios/subestacionales y MOS, estimación robusta frente a apalancamiento, regularización anisotrópica y restricciones de caja, estrategias multihorizonte directo vs. recursivo, familias de competidores y evaluación bajo dependencia espaciotemporal).
- Documentada en `AUDITORIA_22_BENCHMARKS.md` y reflejada en `appendices/B_escenarios_benchmarks.tex` la distinción exacta entre los 9 estimadores de biblioteca, las 12 adaptaciones residuales propias del proyecto y la referencia causal `MM30`.

## 3. Atribución matemática y calibración de originalidad
- Documentada en `MATRIZ_ATRIBUCION_TEOREMAS.md` y aplicada en `sections/06_propiedades_y_lean4.tex` y `appendices/A_demostraciones.tex`:
  - **Teoría clásica / corolarios:** convexidad del poliedro $\Omega$, regularidad $C^1$ y Lipschitz de $\nabla J$, unicidad de $\mathbf{w}^*$ por $D_\alpha \succ 0$, condiciones KKT por Slater (Boyd & Vandenberghe 2004; Huber 1964) y cotas al gradiente en respuesta (Hampel et al. 1986; Huber & Ronchetti 2009).
  - **Caracterizaciones específicas de SM-CFHR:** coercividad y existencia bajo $n \ge 1, \alpha_j > 0$ con intercepto no penalizado; unicidad condicional de $b^*$ cuando existe al menos un residuo interior $|r_{k_0}^*| < \delta$ junto al contraejemplo explícito $n=2$ de no unicidad; y monotonía parcial *ceteris paribus* tras truncamiento físico.
  - **Verificación formal en Lean 4:** explícitamente separada de descubrimiento matemático, aritmética IEEE-754 de Python, convergencia numérica de L-BFGS-B y validez estadística empírica.
- Revisados Resumen, Introducción, Discusión y Conclusiones para mantener un lenguaje prudente y verificable en los seis ejes (teoría establecida, métodos adaptados, decisiones de modelado propias, resultados específicos de SM-CFHR, verificación en Lean 4 y evidencia empírica pendiente).

## 4. Afirmaciones todavía no respaldadas (pendientes de ejecución OOF)
- Cualquier afirmación de superioridad predictiva, significancia estadística (DM--HLN--HAC, MCS, SPA), tamaño del efecto, ganancia de HPO frente a Fixed, ablaciones empíricas y cobertura conforme permanece **abierta y sin concluir** hasta integrar los artefactos reales de la evaluación OOF en curso.

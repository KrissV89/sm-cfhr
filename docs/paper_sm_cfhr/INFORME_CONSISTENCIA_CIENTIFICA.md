# Informe de Consistencia Científica — Manuscrito SM-CFHR

**Fecha:** 2026-11-23  
**Código auditado:** `v1.0.3-polyform` (protocolo científico `v1.0.3`, repositorio oficial `https://github.com/KrissV89/sm-cfhr`; SHA-256 del núcleo congelado `src/adaptive_structured_huber.py` inalterado: `27a9fda4fa8b7ef0d95faf9e925e47ade9ed8f25e9ea5d97d6d7cf1c00ddde83`). Tag científico histórico `v1.0.2` conservado en el repositorio de archivo privado.

## 1. Auditoría Matemática y Verificación Formal en Lean 4
- **Unidades y dimensiones:** Verificada la derivación de $24\,000\,C_{i,t}$ ($\text{kWh/día}$), la separación explícita entre objetivos diarios ($\text{kWh/día}$) y acumulados directos ($Y_{i,t}^{(H)}$ en $\text{kWh}$ o $\text{MWh}$), y la normalización por $H$ al contrastar directo vs. recursivo.
- **Regularidad diferencial:** Precisado que $J(b,\mathbf{w})$ es $C^1$ con gradiente Lipschitz continuo, pero no $C^2$ en los hiperplanos $|r_k|=\delta$.
- **Convexidad y unicidad:** Corregida la afirmación simplificada del docstring de `sm_cfhr/estimator.py`: se demuestra unicidad global de $\mathbf{w}^*$ (por $D_\alpha \succ 0$) y unicidad de $b^*$ bajo la condición suficiente de al menos un residuo en $(-\delta, \delta)$, exhibiendo el contraejemplo degenerado cuando todo $|r_k^*| \ge \delta$.
- **KKT, robustez y monotonía:** Demostradas condiciones KKT bajo cualificación de Slater, acotación de influencia en $y$ (aclarando falta de acotación frente a apalancamiento en $X$) y monotonía parcial tras clipping $[0,1]$.
- **Formalización Lean 4 (`formal/sm-cfhr-lean4`):** 26 teoremas compilados con `lake build` (`leanprover/lean4:v4.36.0`, Mathlib `fc4f59e577feb1a4956671bc04ac18aaaf8cd842`), 0 `sorry`, 0 `admit`, 0 axiomas personalizados (auditado con `#print axioms` en `SMCFHR/AxiomsAudit.lean`). Documentado en el Apéndice E (`appendices/E_verificacion_lean4.tex`) y `THEOREMS_MANIFEST.json`.

## 2. Auditoría Código–Manuscrito (`v1.0.3`)
- Verificada concordancia en `MATRIZ_CODIGO_MANUSCRITO.csv` y `MATRIZ_LATEX_LEAN.csv`.
- **Corrección HPO `v1.0.3`:** Eliminada la aproximación exponencial en `_rollout_recursive_val`; Optuna TPE valida ahora mediante el rollout recursivo exacto `rollout_recursive_from_origin` con `PlantRecursiveState` / `PlantRecursiveWeatherState` (`HPO_PROTOCOL_VERSION = "v1.0.3-exact-recursive-state"`, `HPO_PROTOCOL_HASH = e2a1d2b68fb9fbf90f49ab4caf5c33ae011bc03d63b2486f7ab1cf593b604d5d`), verificado con 30 tests en verde (`tests/test_hpo_recursive_v1_0_3.py`, `tests/test_adaptive_structured_huber.py`, `tests/test_selector_ashr_v2.py`).
- Documentadas explícitamente: (i) la estandarización externa de $X$ en `escala_std_vs_mad`, y (ii) la exclusión de `giacomini_white_audited` cuando $\tau=1$ y $|\hat{\rho}_1|>0{,}15$.

## 3. Auditoría Bibliográfica
- 0 citas indefinidas, 0 entradas no citadas en `bibliography/bibliography.bib`.
- Todas las claves nuevas (Byrd et al. 1995, Zhu et al. 1997, Boyd & Vandenberghe 2004, Huber & Ronchetti 2009, Rousseeuw & Croux 1993, Hansen 2005, Clark & West 2007, Benjamini & Hochberg 1995, Lei et al. 2018, Barber et al. 2023, Cliff 1993, Peng 2011, Virtanen et al. 2020) verificadas contra registros oficiales y DOIs reales.
- Ningún PDF con copyright fue incluido en el worktree del repositorio público.

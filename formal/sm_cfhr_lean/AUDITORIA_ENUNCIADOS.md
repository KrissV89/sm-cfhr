# Auditoría de Enunciados Matemáticos — SM-CFHR (`v1.0.2`)

**Fuentes auditadas:**
- `docs/paper_sm_cfhr/sections/04_modelo_sm_cfhr.tex`
- `docs/paper_sm_cfhr/sections/05_propiedades_matematicas.tex`
- `docs/paper_sm_cfhr/appendices/A_demostraciones.tex`
- `src/adaptive_structured_huber.py`, `src/modelo_aditivo_huber.py`, `src/sm_cfhr/estimator.py`

---

## Matriz de Auditoría por Enunciado

| ID | Enunciado en el manuscrito | Hipótesis explícitas requeridas | Hipótesis faltantes o matices detectados | Vínculo al código Python | Alcance Lean 4 vs. fuera de alcance |
|---|---|---|---|---|---|
| **D1** | Operador $\operatorname{clip}(u,0,1)$ y rango físico $[0, 24000\,C_{i,t}\,H_{\mathrm{dir}}]$ | $C_{i,t} > 0$, $H_{\mathrm{dir}} > 0$ | Ninguna; equivalente a $\max(0, \min(u, 1))$. | `sm_cfhr/estimator.py::predict`, `predict_energy`; `oof_runner.py::predict_ashr_on_query` | **Formalizable en Lean 4** (`Clipping.lean`, `Monotonicity.lean`). |
| **L1** | Pérdida de Huber $\rho_\delta$ y $\psi_\delta$: no negatividad, paridad, cota lineal, Lipschitz-1 de $\psi_\delta$, derivabilidad $C^1$, convexidad global y estricta en $(-\delta,\delta)$ | $\delta > 0$ | El manuscrito menciona $C^1$ y no $C^2$ en $|r|=\delta$, lo cual es exacto. | `modelo_aditivo_huber.py::_huber_val_and_grad` | **Formalizable en Lean 4** (`Huber.lean`). |
| **T1** | Teorema 1: $\Omega = \mathbb{R} \times \mathcal{C}$ cerrado/convexo; $J \in C^1$ convexa con $\nabla J$ Lipschitz; $J$ coerciva y existe minimizador global | $\delta > 0$, $n \ge 1$, $\alpha_j > 0$ para todo $j$ | En la prueba de coercividad debe explicitarse $n \ge 1$ para que $\frac{1}{n}\sum_{k=1}^n \rho_\delta(r_k) \to +\infty$ cuando $|b|\to\infty$. | `adaptive_structured_huber.py::_fit_single` | **Formalizable en Lean 4** (`Model.lean`, `Gradient.lean`, `Optimality.lean`). |
| **T2** | Teorema 2: Unicidad global de $\mathbf{w}^*$; unicidad de $b^*$ si $\exists k$ con $|r_k(b^*,\mathbf{w}^*)| < \delta$; no unicidad en caso degenerado | $\delta > 0$, $n \ge 1$, $\alpha_j > 0$ | Corrige el docstring de `src/sm_cfhr/estimator.py` que llamaba «estrictamente convexo» al problema completo. | `sm_cfhr/estimator.py` (docstring) | **Formalizable en Lean 4** (`Optimality.lean`), incluyendo contraejemplo constructivo. |
| **P1** | Proposición 1 (KKT): equivalencia entre minimizador global en $\Omega$ y condiciones KKT con $\lambda_j^* \ge 0$, $w_j^* \ge 0$, $\lambda_j^* w_j^* = 0$ | $\delta > 0$, $n \ge 1$, $\alpha_j > 0$ | Las restricciones de caja son afines con punto interior estricto (Slater); la equivalencia se demuestra directamente vía convexidad y derivada direccional de primer orden. | `adaptive_structured_huber.py::_fit_single` (`bounds`) | **Formalizable en Lean 4** (`Optimality.lean`). |
| **P2a** | Proposición 2(1): Acotación de contribuciones al gradiente: $|\psi_\delta(r_k)|/n \le \delta/n$ y $|Z_{kj}\psi_\delta(r_k)|/n \le (\delta/n)|Z_{kj}|$ | $\delta > 0$, $n \ge 1$ | **Advertencia crítica:** esto acota la contribución de cada punto al gradiente en $y$, **no** implica función de influencia estadística globalmente acotada frente a puntos de apalancamiento arbitrarios en $Z$. | `adaptive_structured_huber.py::_objective_and_grad_debug` | **Formalizable en Lean 4** como cota algebraica del gradiente (`Gradient.lean`). |
| **P2b** | Proposición 2(2): Monotonía parcial de la predicción física $\hat{y}$ respecto a $j \in \mathcal{M}_+$ | $w_j \ge 0$, $\sigma_y > 0$, $\tilde{s}_j > 0$, $C > 0$, $H_{\mathrm{dir}} > 0$ | El manuscrito exige implícitamente $\sigma_y > 0$ y $\tilde{s}_j > 0$ (garantizados en código por pisos $10^{-6}$ y $10^{-8}$); además solo vale *ceteris paribus* (no monotonía conjunta). | `oof_runner.py::predict_ashr_on_query`, `sm_cfhr/estimator.py::predict_energy` | **Formalizable en Lean 4** (`Monotonicity.lean`). |
| **E1** | Convergencia de L-BFGS-B en punto flotante, complejidad $O(np)$, calibración estadística y validación causal | Hardware/software IEEE-754 | No es un teorema puramente algebraico sobre $\mathbb{R}$: depende de aritmética finita, implementación SciPy y diseño muestral. | `scipy.optimize.minimize(..., method="L-BFGS-B")` | **FUERA DE ALCANCE DE LEAN 4** (se trata aparte; no se declara verificado formalmente). |

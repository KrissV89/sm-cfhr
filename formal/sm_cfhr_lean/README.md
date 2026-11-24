# Formalización Matemática de SM-CFHR en Lean 4

Proyecto autocontenido en **Lean 4** + **Mathlib** que verifica formalmente las propiedades analíticas, convexas, de optimalidad y de monotonía parcial del estimador **SM-CFHR** (*Structured Monotone Capacity-Factor Huber Regressor*, referencia científica `v1.0.2`).

## Estructura
- `SMCFHR/Clipping.lean`: truncamiento físico `[0, 1]`.
- `SMCFHR/Huber.lean`: pérdida $\rho_\delta$, derivada $\psi_\delta$, Lipschitz-1, diferenciabilidad $C^1$, convexidad global y estricta en $(-\delta, \delta)$.
- `SMCFHR/Model.lean`: residuos, conjunto factible $\mathcal{C}$, función objetivo, gradientes y predicción física.
- `SMCFHR/Gradient.lean`: derivada direccional exacta, desigualdad de subgradiente, Lipschitz del gradiente y cotas por observación.
- `SMCFHR/Optimality.lean`: coercividad, existencia de minimizador global, unicidad de $\mathbf{w}^*$, unicidad condicional de $b^*$, contraejemplo degenerado y equivalencia KKT.
- `SMCFHR/Monotonicity.lean`: cotas físicas y monotonía parcial *ceteris paribus*.
- `SMCFHR/AxiomsAudit.lean`: verificación mediante `#print axioms`.

## Reproducción
```bash
cd formal/sm_cfhr_lean
lake exe cache get
lake build
lake env lean SMCFHR/AxiomsAudit.lean
```

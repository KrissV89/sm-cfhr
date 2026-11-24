import SMCFHR

/-!
# Auditoría formal de dependencias axiomáticas (`#print axioms`)

Verifica que todos los teoremas centrales de `SMCFHR` dependen únicamente
de los axiomas estándar de Lean 4 (`propext`, `Classical.choice`, `Quot.sound`)
y no contienen `sorryAx` ni axiomas personalizados.
-/

#print axioms SMCFHR.clip01_nonneg
#print axioms SMCFHR.clip01_le_one
#print axioms SMCFHR.clip01_mono
#print axioms SMCFHR.huber_nonneg
#print axioms SMCFHR.psi_abs_le
#print axioms SMCFHR.psi_lipschitz
#print axioms SMCFHR.hasDerivAt_huber
#print axioms SMCFHR.convexOn_huber
#print axioms SMCFHR.strictConvexOn_huber_band
#print axioms SMCFHR.convex_feasibleSet
#print axioms SMCFHR.isClosed_feasibleSet
#print axioms SMCFHR.convexOn_objective
#print axioms SMCFHR.hasDerivAt_objective_line
#print axioms SMCFHR.objective_subgradient
#print axioms SMCFHR.gradB_lipschitz
#print axioms SMCFHR.gradB_term_bound
#print axioms SMCFHR.gradW_term_bound
#print axioms SMCFHR.objective_coercive
#print axioms SMCFHR.exists_global_minimizer
#print axioms SMCFHR.unique_weights
#print axioms SMCFHR.unique_intercept_of_interior_residual
#print axioms SMCFHR.intercept_nonunique_counterexample
#print axioms SMCFHR.kkt_iff_isMinOn
#print axioms SMCFHR.predictCF_bounds
#print axioms SMCFHR.predictEnergy_bounds
#print axioms SMCFHR.predictEnergy_monotone_in_restricted

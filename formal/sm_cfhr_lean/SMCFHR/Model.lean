import Mathlib.Analysis.Convex.Basic
import Mathlib.Analysis.Convex.Function
import Mathlib.Data.Real.Basic
import Mathlib.Tactic
import SMCFHR.Clipping
import SMCFHR.Huber

/-!
# Formulación del modelo SM-CFHR

Define los espacios finito-dimensionales `Params p := ℝ × (Fin p → ℝ)`,
los residuos `resid Z y θ k`, el conjunto factible `FeasibleSet M`,
la función objetivo `objective δ Z y α θ`, las componentes del gradiente
`gradB` y `gradW`, y la predicción física `predictCF` y `predictEnergy`.
-/

namespace SMCFHR

/-- Parámetros del modelo: intercepto `b` y vector de pesos `w ∈ ℝ^p`. -/
abbrev Params (p : ℕ) := ℝ × (Fin p → ℝ)

/-- Residuo estandarizado de la observación `k`: `y_k - (b + ∑_j Z_{kj} w_j)`. -/
def resid {n p : ℕ} (Z : Fin n → Fin p → ℝ) (y : Fin n → ℝ)
    (θ : Params p) (k : Fin n) : ℝ :=
  y k - (θ.1 + ∑ j : Fin p, Z k j * θ.2 j)

/-- Conjunto factible de parámetros con restricciones de no negatividad en `M`. -/
def FeasibleSet {p : ℕ} (M : Finset (Fin p)) : Set (Params p) :=
  {θ : Params p | ∀ j ∈ M, 0 ≤ θ.2 j}

/-- Función objetivo SM-CFHR: pérdida Huber promedio + penalización cuadrática diagonal. -/
noncomputable def objective {n p : ℕ} (δ : ℝ) (Z : Fin n → Fin p → ℝ)
    (y : Fin n → ℝ) (α : Fin p → ℝ) (θ : Params p) : ℝ :=
  ((1 : ℝ) / n) * ∑ k : Fin n, huber δ (resid Z y θ k) +
    (1 / 2 : ℝ) * ∑ j : Fin p, α j * (θ.2 j) ^ 2

/-- Derivada parcial del objetivo respecto al intercepto `b`. -/
noncomputable def gradB {n p : ℕ} (δ : ℝ) (Z : Fin n → Fin p → ℝ) (y : Fin n → ℝ)
    (θ : Params p) : ℝ :=
  -((1 : ℝ) / n) * ∑ k : Fin n, psi δ (resid Z y θ k)

/-- Derivada parcial del objetivo respecto al peso `w_j`. -/
noncomputable def gradW {n p : ℕ} (δ : ℝ) (Z : Fin n → Fin p → ℝ) (y : Fin n → ℝ)
    (α : Fin p → ℝ) (θ : Params p) (j : Fin p) : ℝ :=
  -((1 : ℝ) / n) * ∑ k : Fin n, Z k j * psi δ (resid Z y θ k) + α j * θ.2 j

/-- Predicción en factor de capacidad acotado al intervalo `[0, 1]`. -/
noncomputable def predictCF {p : ℕ} (m_y s_y : ℝ) (x_mean x_scale : Fin p → ℝ)
    (θ : Params p) (x : Fin p → ℝ) : ℝ :=
  clip01 (m_y + s_y * (θ.1 + ∑ j : Fin p, ((x j - x_mean j) / x_scale j) * θ.2 j))

/-- Predicción física de energía en kWh: `cf * 24000 * cap_MW * H_dir`. -/
noncomputable def predictEnergy {p : ℕ} (m_y s_y cap_MW H_dir : ℝ) (x_mean x_scale : Fin p → ℝ)
    (θ : Params p) (x : Fin p → ℝ) : ℝ :=
  predictCF m_y s_y x_mean x_scale θ x * (24000 : ℝ) * cap_MW * H_dir

theorem mem_feasibleSet_iff {p : ℕ} {M : Finset (Fin p)} {θ : Params p} :
    θ ∈ FeasibleSet M ↔ ∀ j ∈ M, 0 ≤ θ.2 j := Iff.rfl

theorem zero_mem_feasibleSet {p : ℕ} (M : Finset (Fin p)) :
    ((0 : ℝ), fun _ => (0 : ℝ)) ∈ FeasibleSet M := by
  intro j _
  exact le_rfl

theorem convex_feasibleSet {p : ℕ} (M : Finset (Fin p)) :
    Convex ℝ (FeasibleSet M) := by
  intro θ₁ h₁ θ₂ h₂ a b ha hb _ j hj
  change 0 ≤ a * θ₁.2 j + b * θ₂.2 j
  exact add_nonneg (mul_nonneg ha (h₁ j hj)) (mul_nonneg hb (h₂ j hj))

theorem isClosed_feasibleSet {p : ℕ} (M : Finset (Fin p)) :
    IsClosed (FeasibleSet M) := by
  have h_eq : FeasibleSet M = ⋂ j ∈ M, {θ : Params p | 0 ≤ θ.2 j} := by
    ext θ; simp [FeasibleSet]
  rw [h_eq]
  refine isClosed_biInter fun j _ => ?_
  exact isClosed_le continuous_const ((continuous_apply j).comp continuous_snd)

theorem objective_nonneg {n p : ℕ} {δ : ℝ} (hδ : 0 < δ)
    (Z : Fin n → Fin p → ℝ) (y : Fin n → ℝ)
    {α : Fin p → ℝ} (hα : ∀ j, 0 ≤ α j) (θ : Params p) :
    0 ≤ objective δ Z y α θ := by
  unfold objective
  have h1 : 0 ≤ ((1 : ℝ) / n) * ∑ k : Fin n, huber δ (resid Z y θ k) := by
    refine mul_nonneg (by positivity) ?_
    exact Finset.sum_nonneg fun k _ => huber_nonneg hδ _
  have h2 : 0 ≤ (1 / 2 : ℝ) * ∑ j : Fin p, α j * (θ.2 j) ^ 2 := by
    refine mul_nonneg (by norm_num) ?_
    exact Finset.sum_nonneg fun j _ => mul_nonneg (hα j) (sq_nonneg _)
  exact add_nonneg h1 h2

theorem resid_affine {n p : ℕ} (Z : Fin n → Fin p → ℝ) (y : Fin n → ℝ)
    (θ₁ θ₂ : Params p) {a b : ℝ} (hab : a + b = 1) (k : Fin n) :
    resid Z y (a • θ₁ + b • θ₂) k = a * resid Z y θ₁ k + b * resid Z y θ₂ k := by
  unfold resid
  simp only [Prod.smul_fst, Prod.smul_snd, Prod.fst_add, Prod.snd_add, Pi.add_apply,
    Pi.smul_apply, smul_eq_mul]
  have hsum : ∑ j : Fin p, Z k j * (a * θ₁.2 j + b * θ₂.2 j) =
      a * (∑ j : Fin p, Z k j * θ₁.2 j) + b * (∑ j : Fin p, Z k j * θ₂.2 j) := by
    simp_rw [mul_add, Finset.sum_add_distrib, ← mul_assoc, mul_comm _ a, mul_comm _ b, mul_assoc,
      ← Finset.mul_sum]
  rw [hsum]
  have hy : y k = a * y k + b * y k := by
    calc y k = (a + b) * y k := by rw [hab, one_mul]
    _ = a * y k + b * y k := add_mul a b (y k)
  linarith

theorem convexOn_objective {n p : ℕ} {δ : ℝ} (hδ : 0 < δ)
    (Z : Fin n → Fin p → ℝ) (y : Fin n → ℝ)
    {α : Fin p → ℝ} (hα : ∀ j, 0 ≤ α j) :
    ConvexOn ℝ Set.univ (objective δ Z y α) := by
  refine ⟨convex_univ, fun θ₁ _ θ₂ _ a b ha hb hab => ?_⟩
  unfold objective
  have hhub : ∀ k : Fin n, huber δ (resid Z y (a • θ₁ + b • θ₂) k) ≤
      a * huber δ (resid Z y θ₁ k) + b * huber δ (resid Z y θ₂ k) := by
    intro k
    rw [resid_affine Z y θ₁ θ₂ hab k]
    exact (convexOn_huber hδ).2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab
  have hsum1 : ∑ k : Fin n, huber δ (resid Z y (a • θ₁ + b • θ₂) k) ≤
      a * ∑ k : Fin n, huber δ (resid Z y θ₁ k) + b * ∑ k : Fin n, huber δ (resid Z y θ₂ k) := by
    calc ∑ k : Fin n, huber δ (resid Z y (a • θ₁ + b • θ₂) k)
        ≤ ∑ k : Fin n, (a * huber δ (resid Z y θ₁ k) + b * huber δ (resid Z y θ₂ k)) :=
          Finset.sum_le_sum fun k _ => hhub k
    _ = a * ∑ k : Fin n, huber δ (resid Z y θ₁ k) + b * ∑ k : Fin n, huber δ (resid Z y θ₂ k) := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
  have hsq : ∀ j : Fin p, (a * θ₁.2 j + b * θ₂.2 j) ^ 2 ≤
      a * (θ₁.2 j) ^ 2 + b * (θ₂.2 j) ^ 2 := by
    intro j
    have hid : a * (θ₁.2 j) ^ 2 + b * (θ₂.2 j) ^ 2 - (a * θ₁.2 j + b * θ₂.2 j) ^ 2 =
        a * b * (θ₁.2 j - θ₂.2 j) ^ 2 := by
      rw [show a = 1 - b by linarith]; ring
    have hnn : 0 ≤ a * b * (θ₁.2 j - θ₂.2 j) ^ 2 :=
      mul_nonneg (mul_nonneg ha hb) (sq_nonneg _)
    linarith
  have hsum2 : ∑ j : Fin p, α j * ((a • θ₁ + b • θ₂).2 j) ^ 2 ≤
      a * ∑ j : Fin p, α j * (θ₁.2 j) ^ 2 + b * ∑ j : Fin p, α j * (θ₂.2 j) ^ 2 := by
    calc ∑ j : Fin p, α j * ((a • θ₁ + b • θ₂).2 j) ^ 2
        ≤ ∑ j : Fin p, α j * (a * (θ₁.2 j) ^ 2 + b * (θ₂.2 j) ^ 2) := by
          refine Finset.sum_le_sum fun j _ => ?_
          exact mul_le_mul_of_nonneg_left (by simpa [smul_eq_mul] using hsq j) (hα j)
    _ = a * ∑ j : Fin p, α j * (θ₁.2 j) ^ 2 + b * ∑ j : Fin p, α j * (θ₂.2 j) ^ 2 := by
          simp_rw [mul_add, Finset.sum_add_distrib, ← mul_assoc, mul_comm _ a, mul_comm _ b, mul_assoc,
            ← Finset.mul_sum]
  have hc1 : (0 : ℝ) ≤ (1 : ℝ) / n := by positivity
  have hc2 : (0 : ℝ) ≤ (1 / 2 : ℝ) := by norm_num
  have hl1 := mul_le_mul_of_nonneg_left hsum1 hc1
  have hl2 := mul_le_mul_of_nonneg_left hsum2 hc2
  simp only [smul_eq_mul]
  linarith

end SMCFHR

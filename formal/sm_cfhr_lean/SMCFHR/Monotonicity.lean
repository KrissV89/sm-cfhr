import Mathlib.Data.Real.Basic
import Mathlib.Tactic
import SMCFHR.Clipping
import SMCFHR.Model

/-!
# Cotas físicas y monotonía parcial de la predicción

Demuestra que:
- `predictCF` pertenece siempre a `[0, 1]`;
- `predictEnergy` pertenece a `[0, 24000 * cap_MW * H_dir]` cuando `cap_MW > 0` y `H_dir > 0`;
- para todo predictor restringido `j ∈ M` con `w_j ≥ 0`, `s_y > 0`, `x_scale j > 0`,
  `cap_MW > 0` y `H_dir > 0`, incrementar la coordenada `j` sin alterar las demás
  no disminuye la energía predicha (`ceteris paribus`).
-/

namespace SMCFHR

theorem predictCF_bounds {p : ℕ} (m_y s_y : ℝ) (x_mean x_scale : Fin p → ℝ)
    (θ : Params p) (x : Fin p → ℝ) :
    0 ≤ predictCF m_y s_y x_mean x_scale θ x ∧
      predictCF m_y s_y x_mean x_scale θ x ≤ 1 :=
  ⟨clip01_nonneg _, clip01_le_one _⟩

theorem predictEnergy_bounds {p : ℕ} (m_y s_y cap_MW H_dir : ℝ)
    (hcap : 0 ≤ cap_MW) (hH : 0 ≤ H_dir)
    (x_mean x_scale : Fin p → ℝ) (θ : Params p) (x : Fin p → ℝ) :
    0 ≤ predictEnergy m_y s_y cap_MW H_dir x_mean x_scale θ x ∧
      predictEnergy m_y s_y cap_MW H_dir x_mean x_scale θ x ≤ 24000 * cap_MW * H_dir := by
  unfold predictEnergy
  have hcf := predictCF_bounds m_y s_y x_mean x_scale θ x
  have hscale : 0 ≤ (24000 : ℝ) * cap_MW * H_dir := by positivity
  constructor
  · exact mul_nonneg (mul_nonneg (mul_nonneg hcf.1 (by norm_num)) hcap) hH
  · have := mul_le_mul_of_nonneg_right hcf.2 hscale
    linarith

theorem predictCF_monotone_in_restricted {p : ℕ} {M : Finset (Fin p)}
    {θ : Params p} (hθ : θ ∈ FeasibleSet M)
    (m_y : ℝ) {s_y : ℝ} (hs : 0 < s_y)
    (x_mean : Fin p → ℝ) {x_scale : Fin p → ℝ} {j : Fin p} (hj : j ∈ M) (hscale_j : 0 < x_scale j)
    (x x' : Fin p → ℝ) (h_incr : x j ≤ x' j) (h_other : ∀ l ≠ j, x' l = x l) :
    predictCF m_y s_y x_mean x_scale θ x ≤ predictCF m_y s_y x_mean x_scale θ x' := by
  unfold predictCF
  apply clip01_mono
  have hwj : 0 ≤ θ.2 j := hθ j hj
  have hsum : ∑ l : Fin p, ((x l - x_mean l) / x_scale l) * θ.2 l ≤
      ∑ l : Fin p, ((x' l - x_mean l) / x_scale l) * θ.2 l := by
    refine Finset.sum_le_sum fun l _ => ?_
    by_cases hlj : l = j
    · subst hlj
      have hdiff : (x l - x_mean l) / x_scale l ≤ (x' l - x_mean l) / x_scale l :=
        div_le_div_of_nonneg_right (by linarith) (le_of_lt hscale_j)
      exact mul_le_mul_of_nonneg_right hdiff hwj
    · rw [h_other l hlj]
  have hmul : s_y * (θ.1 + ∑ l : Fin p, ((x l - x_mean l) / x_scale l) * θ.2 l) ≤
      s_y * (θ.1 + ∑ l : Fin p, ((x' l - x_mean l) / x_scale l) * θ.2 l) :=
    mul_le_mul_of_nonneg_left (by linarith) (le_of_lt hs)
  linarith

theorem predictEnergy_monotone_in_restricted {p : ℕ} {M : Finset (Fin p)}
    {θ : Params p} (hθ : θ ∈ FeasibleSet M)
    {m_y s_y cap_MW H_dir : ℝ} (hs : 0 < s_y) (hcap : 0 < cap_MW) (hH : 0 < H_dir)
    {x_mean x_scale : Fin p → ℝ} {j : Fin p} (hj : j ∈ M) (hscale_j : 0 < x_scale j)
    (x x' : Fin p → ℝ) (h_incr : x j ≤ x' j) (h_other : ∀ l ≠ j, x' l = x l) :
    predictEnergy m_y s_y cap_MW H_dir x_mean x_scale θ x ≤
      predictEnergy m_y s_y cap_MW H_dir x_mean x_scale θ x' := by
  unfold predictEnergy
  have hcf := predictCF_monotone_in_restricted hθ m_y hs x_mean hj hscale_j x x' h_incr h_other
  have hpos : 0 ≤ (24000 : ℝ) * cap_MW * H_dir := by positivity
  calc predictCF m_y s_y x_mean x_scale θ x * 24000 * cap_MW * H_dir
      = predictCF m_y s_y x_mean x_scale θ x * (24000 * cap_MW * H_dir) := by ring
  _ ≤ predictCF m_y s_y x_mean x_scale θ x' * (24000 * cap_MW * H_dir) :=
      mul_le_mul_of_nonneg_right hcf hpos
  _ = predictCF m_y s_y x_mean x_scale θ x' * 24000 * cap_MW * H_dir := by ring

end SMCFHR

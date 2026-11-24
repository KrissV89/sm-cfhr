import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Tactic
import SMCFHR.Huber
import SMCFHR.Model

/-!
# Gradiente analítico exacto, regularidad Lipschitz y cotas por observación

Formaliza:
- derivada direccional exacta `hasDerivAt_objective_line` a lo largo de cualquier rayo `θ + t • d`;
- desigualdad de subgradiente de primer orden `objective_subgradient`;
- continuidad Lipschitz explícita de `gradB`;
- cotas exactas por observación sobre las contribuciones al gradiente.
-/

namespace SMCFHR

private lemma hasDerivAt_huber_affine {δ : ℝ} (hδ : 0 < δ) (c s : ℝ) :
    HasDerivAt (fun t : ℝ => huber δ (c + t * s)) (psi δ c * s) 0 := by
  have h_inner : HasDerivAt (fun t : ℝ => c + t * s) s 0 := by
    have h1 := ((hasDerivAt_id (0 : ℝ)).mul_const s).const_add c
    convert h1 using 1 <;> simp
  have h_hub_at : HasDerivAt (huber δ) (psi δ c) (c + (0 : ℝ) * s) := by
    simpa using hasDerivAt_huber hδ c
  simpa only [Function.comp_def] using HasDerivAt.comp (0 : ℝ) h_hub_at h_inner

theorem hasDerivAt_objective_line {n p : ℕ} {δ : ℝ} (hδ : 0 < δ)
    (Z : Fin n → Fin p → ℝ) (y : Fin n → ℝ) (α : Fin p → ℝ)
    (θ d : Params p) :
    HasDerivAt (fun t : ℝ => objective δ Z y α (θ + t • d))
      (gradB δ Z y θ * d.1 + ∑ j : Fin p, gradW δ Z y α θ j * d.2 j) 0 := by
  unfold objective gradB gradW
  have h_hub : ∀ k : Fin n, HasDerivAt
      (fun t : ℝ => huber δ (resid Z y (θ + t • d) k))
      (psi δ (resid Z y θ k) * (-(d.1 + ∑ j : Fin p, Z k j * d.2 j))) 0 := by
    intro k
    have h_lin : (fun t : ℝ => resid Z y (θ + t • d) k) =
        fun t : ℝ => resid Z y θ k + t * (-(d.1 + ∑ j : Fin p, Z k j * d.2 j)) := by
      ext t
      unfold resid
      simp only [Prod.fst_add, Prod.snd_add, Prod.smul_fst, Prod.smul_snd, Pi.add_apply,
        Pi.smul_apply, smul_eq_mul]
      have hs : ∑ j : Fin p, Z k j * (θ.2 j + t * d.2 j) =
          (∑ j : Fin p, Z k j * θ.2 j) + t * (∑ j : Fin p, Z k j * d.2 j) := by
        simp_rw [mul_add, Finset.sum_add_distrib, ← mul_assoc, mul_comm _ t, mul_assoc,
          ← Finset.mul_sum]
      rw [hs]; ring
    have h_eq_hub : (fun t : ℝ => huber δ (resid Z y (θ + t • d) k)) =
        fun t : ℝ => huber δ (resid Z y θ k + t * (-(d.1 + ∑ j : Fin p, Z k j * d.2 j))) := by
      ext t; rw [congrFun h_lin t]
    rw [h_eq_hub]
    exact hasDerivAt_huber_affine hδ (resid Z y θ k) (-(d.1 + ∑ j : Fin p, Z k j * d.2 j))
  have h_sum1 : HasDerivAt
      (fun t : ℝ => ((1 : ℝ) / n) * ∑ k : Fin n, huber δ (resid Z y (θ + t • d) k))
      (((1 : ℝ) / n) * ∑ k : Fin n, psi δ (resid Z y θ k) * (-(d.1 + ∑ j : Fin p, Z k j * d.2 j))) 0 := by
    simpa using (HasDerivAt.sum fun k _ => h_hub k).const_mul ((1 : ℝ) / n)
  have h_pen : ∀ j : Fin p, HasDerivAt
      (fun t : ℝ => α j * (θ.2 j + t * d.2 j) ^ 2)
      (α j * (2 * θ.2 j * d.2 j)) 0 := by
    intro j
    have h_in : HasDerivAt (fun t : ℝ => θ.2 j + t * d.2 j) (d.2 j) 0 := by
      simpa using ((hasDerivAt_id (0 : ℝ)).mul_const (d.2 j)).const_add (θ.2 j)
    have h_sq : HasDerivAt (fun t : ℝ => (θ.2 j + t * d.2 j) ^ 2) (2 * (θ.2 j + 0 * d.2 j) ^ 1 * d.2 j) 0 :=
      h_in.pow 2
    simpa [pow_one, mul_assoc] using h_sq.const_mul (α j)
  have h_sum2 : HasDerivAt
      (fun t : ℝ => (1 / 2 : ℝ) * ∑ j : Fin p, α j * ((θ + t • d).2 j) ^ 2)
      ((1 / 2 : ℝ) * ∑ j : Fin p, α j * (2 * θ.2 j * d.2 j)) 0 := by
    have h_eq : (fun t : ℝ => (1 / 2 : ℝ) * ∑ j : Fin p, α j * ((θ + t • d).2 j) ^ 2) =
        fun t : ℝ => (1 / 2 : ℝ) * ∑ j : Fin p, α j * (θ.2 j + t * d.2 j) ^ 2 := by
      ext t; simp [smul_eq_mul]
    rw [h_eq]
    simpa using (HasDerivAt.sum fun j _ => h_pen j).const_mul (1 / 2 : ℝ)
  have h_tot := h_sum1.add h_sum2
  convert h_tot using 1
  have h_reorg1 : ((1 : ℝ) / n) * ∑ k : Fin n, psi δ (resid Z y θ k) * (-(d.1 + ∑ j : Fin p, Z k j * d.2 j)) =
      (-((1 : ℝ) / n) * ∑ k : Fin n, psi δ (resid Z y θ k)) * d.1 +
        ∑ j : Fin p, (-((1 : ℝ) / n) * ∑ k : Fin n, Z k j * psi δ (resid Z y θ k)) * d.2 j := by
    have hs1 : ∑ k : Fin n, psi δ (resid Z y θ k) * (-(d.1 + ∑ j : Fin p, Z k j * d.2 j)) =
        -(∑ k : Fin n, psi δ (resid Z y θ k)) * d.1 -
          ∑ j : Fin p, (∑ k : Fin n, Z k j * psi δ (resid Z y θ k)) * d.2 j := by
      calc ∑ k : Fin n, psi δ (resid Z y θ k) * (-(d.1 + ∑ j : Fin p, Z k j * d.2 j))
          = ∑ k : Fin n, (-(psi δ (resid Z y θ k) * d.1) - ∑ j : Fin p, (Z k j * psi δ (resid Z y θ k)) * d.2 j) := by
            refine Finset.sum_congr rfl fun k _ => ?_
            rw [neg_add, mul_add]
            have : psi δ (resid Z y θ k) * -∑ j : Fin p, Z k j * d.2 j =
                -∑ j : Fin p, (Z k j * psi δ (resid Z y θ k)) * d.2 j := by
              rw [mul_neg, Finset.mul_sum]
              congr 1
              refine Finset.sum_congr rfl fun j _ => by ring
            rw [this]; ring
      _ = -(∑ k : Fin n, psi δ (resid Z y θ k)) * d.1 -
            ∑ k : Fin n, ∑ j : Fin p, (Z k j * psi δ (resid Z y θ k)) * d.2 j := by
            rw [Finset.sum_sub_distrib, ← Finset.sum_neg_distrib]
            congr 1
            simp_rw [← neg_mul, ← Finset.sum_mul]
      _ = -(∑ k : Fin n, psi δ (resid Z y θ k)) * d.1 -
            ∑ j : Fin p, (∑ k : Fin n, Z k j * psi δ (resid Z y θ k)) * d.2 j := by
            rw [Finset.sum_comm]
            congr 1
            refine Finset.sum_congr rfl fun j _ => ?_
            rw [Finset.sum_mul]
    rw [hs1, mul_sub]
    have h_f1 : ((1 : ℝ) / n) * (-(∑ k : Fin n, psi δ (resid Z y θ k)) * d.1) =
        (-((1 : ℝ) / n) * ∑ k : Fin n, psi δ (resid Z y θ k)) * d.1 := by ring
    have h_f2 : ((1 : ℝ) / n) * ∑ j : Fin p, (∑ k : Fin n, Z k j * psi δ (resid Z y θ k)) * d.2 j =
        -∑ j : Fin p, (-((1 : ℝ) / n) * ∑ k : Fin n, Z k j * psi δ (resid Z y θ k)) * d.2 j := by
      rw [Finset.mul_sum, ← Finset.sum_neg_distrib]
      refine Finset.sum_congr rfl fun j _ => by ring
    rw [h_f1, h_f2]; ring
  have h_reorg2 : (1 / 2 : ℝ) * ∑ j : Fin p, α j * (2 * θ.2 j * d.2 j) =
      ∑ j : Fin p, (α j * θ.2 j) * d.2 j := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun j _ => by ring
  rw [h_reorg1, h_reorg2]
  simp_rw [add_mul, Finset.sum_add_distrib]
  ring

theorem objective_subgradient {n p : ℕ} {δ : ℝ} (hδ : 0 < δ)
    (Z : Fin n → Fin p → ℝ) (y : Fin n → ℝ)
    {α : Fin p → ℝ} (hα : ∀ j, 0 ≤ α j) (θ θ' : Params p) :
    objective δ Z y α θ +
      (gradB δ Z y θ * (θ'.1 - θ.1) + ∑ j : Fin p, gradW δ Z y α θ j * (θ'.2 j - θ.2 j)) ≤
      objective δ Z y α θ' := by
  have h_hub_sum : ∑ k : Fin n, (huber δ (resid Z y θ k) +
      psi δ (resid Z y θ k) * (resid Z y θ' k - resid Z y θ k)) ≤
      ∑ k : Fin n, huber δ (resid Z y θ' k) :=
    Finset.sum_le_sum fun k _ => huber_subgradient hδ (resid Z y θ k) (resid Z y θ' k)
  have h_res_diff : ∀ k : Fin n, resid Z y θ' k - resid Z y θ k =
      -(θ'.1 - θ.1) - ∑ j : Fin p, Z k j * (θ'.2 j - θ.2 j) := by
    intro k; unfold resid; simp_rw [mul_sub, Finset.sum_sub_distrib]; ring
  have h_lin_hub : ((1 : ℝ) / n) * ∑ k : Fin n, psi δ (resid Z y θ k) * (resid Z y θ' k - resid Z y θ k) =
      gradB δ Z y θ * (θ'.1 - θ.1) +
        ∑ j : Fin p, (-((1 : ℝ) / n) * ∑ k : Fin n, Z k j * psi δ (resid Z y θ k)) * (θ'.2 j - θ.2 j) := by
    have h_inner : ∑ k : Fin n, psi δ (resid Z y θ k) * (resid Z y θ' k - resid Z y θ k) =
        -(∑ k : Fin n, psi δ (resid Z y θ k)) * (θ'.1 - θ.1) -
          ∑ j : Fin p, (∑ k : Fin n, Z k j * psi δ (resid Z y θ k)) * (θ'.2 j - θ.2 j) := by
      calc ∑ k : Fin n, psi δ (resid Z y θ k) * (resid Z y θ' k - resid Z y θ k)
          = ∑ k : Fin n, (-(psi δ (resid Z y θ k) * (θ'.1 - θ.1)) -
              ∑ j : Fin p, (Z k j * psi δ (resid Z y θ k)) * (θ'.2 j - θ.2 j)) := by
            refine Finset.sum_congr rfl fun k _ => ?_
            rw [h_res_diff k, mul_sub, Finset.mul_sum]
            congr 1; ring
            refine Finset.sum_congr rfl fun j _ => by ring
      _ = -(∑ k : Fin n, psi δ (resid Z y θ k)) * (θ'.1 - θ.1) -
            ∑ j : Fin p, (∑ k : Fin n, Z k j * psi δ (resid Z y θ k)) * (θ'.2 j - θ.2 j) := by
            rw [Finset.sum_sub_distrib, ← Finset.sum_neg_distrib, Finset.sum_comm]
            congr 1
            · simp_rw [← neg_mul, ← Finset.sum_mul]
            · refine Finset.sum_congr rfl fun j _ => by rw [Finset.sum_mul]
    rw [h_inner, mul_sub]
    unfold gradB
    have hf1 : ((1 : ℝ) / n) * (-(∑ k : Fin n, psi δ (resid Z y θ k)) * (θ'.1 - θ.1)) =
        (-((1 : ℝ) / n) * ∑ k : Fin n, psi δ (resid Z y θ k)) * (θ'.1 - θ.1) := by ring
    have hf2 : ((1 : ℝ) / n) * ∑ j : Fin p, (∑ k : Fin n, Z k j * psi δ (resid Z y θ k)) * (θ'.2 j - θ.2 j) =
        -∑ j : Fin p, (-((1 : ℝ) / n) * ∑ k : Fin n, Z k j * psi δ (resid Z y θ k)) * (θ'.2 j - θ.2 j) := by
      rw [Finset.mul_sum, ← Finset.sum_neg_distrib]
      refine Finset.sum_congr rfl fun j _ => by ring
    rw [hf1, hf2]; ring
  have h_quad : ∑ j : Fin p, ((1 / 2 : ℝ) * α j * (θ.2 j) ^ 2 + α j * θ.2 j * (θ'.2 j - θ.2 j)) ≤
      ∑ j : Fin p, (1 / 2 : ℝ) * α j * (θ'.2 j) ^ 2 := by
    refine Finset.sum_le_sum fun j _ => ?_
    have hsq : 0 ≤ (1 / 2 : ℝ) * α j * (θ'.2 j - θ.2 j) ^ 2 :=
      mul_nonneg (mul_nonneg (by norm_num) (hα j)) (sq_nonneg _)
    linarith
  have h_hub_avg := mul_le_mul_of_nonneg_left h_hub_sum (by positivity : (0 : ℝ) ≤ (1 : ℝ) / n)
  rw [Finset.sum_add_distrib, mul_add] at h_hub_avg
  unfold objective gradW
  simp_rw [add_mul, Finset.sum_add_distrib]
  have h_pen_eq : (1 / 2 : ℝ) * ∑ j : Fin p, α j * (θ.2 j) ^ 2 =
      ∑ j : Fin p, (1 / 2 : ℝ) * α j * (θ.2 j) ^ 2 := by
    rw [Finset.mul_sum]; refine Finset.sum_congr rfl fun j _ => by ring
  have h_pen_eq' : (1 / 2 : ℝ) * ∑ j : Fin p, α j * (θ'.2 j) ^ 2 =
      ∑ j : Fin p, (1 / 2 : ℝ) * α j * (θ'.2 j) ^ 2 := by
    rw [Finset.mul_sum]; refine Finset.sum_congr rfl fun j _ => by ring
  rw [h_pen_eq, h_pen_eq']
  have h_quad2 : ∑ j : Fin p, (1 / 2 : ℝ) * α j * (θ.2 j) ^ 2 +
      ∑ j : Fin p, α j * θ.2 j * (θ'.2 j - θ.2 j) ≤
      ∑ j : Fin p, (1 / 2 : ℝ) * α j * (θ'.2 j) ^ 2 := by
    rw [← Finset.sum_add_distrib]; exact h_quad
  linarith

theorem gradB_lipschitz {n p : ℕ} {δ : ℝ} (hδ : 0 < δ)
    (Z : Fin n → Fin p → ℝ) (y : Fin n → ℝ) (θ₁ θ₂ : Params p) :
    |gradB δ Z y θ₁ - gradB δ Z y θ₂| ≤
      ((1 : ℝ) / n) * ∑ k : Fin n, (|θ₁.1 - θ₂.1| + ∑ j : Fin p, |Z k j| * |θ₁.2 j - θ₂.2 j|) := by
  unfold gradB
  have h1 : -((1 : ℝ) / n) * ∑ k : Fin n, psi δ (resid Z y θ₁ k) -
      (-((1 : ℝ) / n) * ∑ k : Fin n, psi δ (resid Z y θ₂ k)) =
      -((1 : ℝ) / n) * ∑ k : Fin n, (psi δ (resid Z y θ₁ k) - psi δ (resid Z y θ₂ k)) := by
    rw [Finset.sum_sub_distrib]; ring
  rw [h1, abs_mul, abs_neg, abs_of_nonneg (by positivity : (0 : ℝ) ≤ (1 : ℝ) / n)]
  refine mul_le_mul_of_nonneg_left ?_ (by positivity)
  calc |∑ k : Fin n, (psi δ (resid Z y θ₁ k) - psi δ (resid Z y θ₂ k))|
      ≤ ∑ k : Fin n, |psi δ (resid Z y θ₁ k) - psi δ (resid Z y θ₂ k)| :=
        Finset.abs_sum_le_sum_abs _ _
  _ ≤ ∑ k : Fin n, |resid Z y θ₁ k - resid Z y θ₂ k| := by
        refine Finset.sum_le_sum fun k _ => ?_
        have hl := (psi_lipschitz hδ).dist_le_mul (resid Z y θ₁ k) (resid Z y θ₂ k)
        simpa [Real.dist_eq] using hl
  _ ≤ ∑ k : Fin n, (|θ₁.1 - θ₂.1| + ∑ j : Fin p, |Z k j| * |θ₁.2 j - θ₂.2 j|) := by
        refine Finset.sum_le_sum fun k _ => ?_
        have hr : resid Z y θ₁ k - resid Z y θ₂ k =
            (θ₂.1 - θ₁.1) + ∑ j : Fin p, Z k j * (θ₂.2 j - θ₁.2 j) := by
          unfold resid
          simp_rw [mul_sub, Finset.sum_sub_distrib]
          ring
        rw [hr]
        calc |(θ₂.1 - θ₁.1) + ∑ j : Fin p, Z k j * (θ₂.2 j - θ₁.2 j)|
            ≤ |θ₂.1 - θ₁.1| + |∑ j : Fin p, Z k j * (θ₂.2 j - θ₁.2 j)| := abs_add_le _ _
        _ ≤ |θ₁.1 - θ₂.1| + ∑ j : Fin p, |Z k j * (θ₂.2 j - θ₁.2 j)| := by
              rw [abs_sub_comm θ₂.1 θ₁.1]
              exact add_le_add le_rfl (Finset.abs_sum_le_sum_abs (fun j => Z k j * (θ₂.2 j - θ₁.2 j)) Finset.univ)
        _ = |θ₁.1 - θ₂.1| + ∑ j : Fin p, |Z k j| * |θ₁.2 j - θ₂.2 j| := by
              congr 1
              refine Finset.sum_congr rfl fun j _ => ?_
              rw [abs_mul, abs_sub_comm]

theorem gradB_term_bound {n p : ℕ} (_hn : 0 < n) {δ : ℝ} (hδ : 0 < δ)
    (Z : Fin n → Fin p → ℝ) (y : Fin n → ℝ) (θ : Params p) (k : Fin n) :
    |((1 : ℝ) / n) * psi δ (resid Z y θ k)| ≤ δ / n := by
  rw [abs_mul, abs_of_nonneg (by positivity : (0 : ℝ) ≤ (1 : ℝ) / n)]
  have h := mul_le_mul_of_nonneg_left (psi_abs_le hδ (resid Z y θ k))
    (by positivity : (0 : ℝ) ≤ (1 : ℝ) / n)
  simpa [one_div, div_eq_mul_inv, mul_comm] using h

theorem gradW_term_bound {n p : ℕ} (_hn : 0 < n) {δ : ℝ} (hδ : 0 < δ)
    (Z : Fin n → Fin p → ℝ) (y : Fin n → ℝ) (θ : Params p) (k : Fin n) (j : Fin p) :
    |((1 : ℝ) / n) * (Z k j * psi δ (resid Z y θ k))| ≤ (δ / n) * |Z k j| := by
  rw [abs_mul, abs_mul, abs_of_nonneg (by positivity : (0 : ℝ) ≤ (1 : ℝ) / n)]
  have h := psi_abs_le hδ (resid Z y θ k)
  calc ((1 : ℝ) / n) * (|Z k j| * |psi δ (resid Z y θ k)|)
      ≤ ((1 : ℝ) / n) * (|Z k j| * δ) := by
        refine mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left h (abs_nonneg _)) (by positivity)
  _ = (δ / n) * |Z k j| := by ring

end SMCFHR

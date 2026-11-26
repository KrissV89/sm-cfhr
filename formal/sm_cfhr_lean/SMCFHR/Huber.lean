import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.Deriv.Slope
import Mathlib.Analysis.Convex.Function
import Mathlib.Topology.MetricSpace.Lipschitz
import Mathlib.Tactic

/-!
# Pérdida robusta de Huber y función de influencia

Formaliza la función de pérdida de Huber `huber δ r` y su derivada `psi δ r = max (-δ) (min r δ)`
para `δ > 0`, demostrando:
- no negatividad, paridad y cota inferior lineal;
- acotación `|psi δ r| ≤ δ`, monotonía y propiedad 1-Lipschitz;
- derivabilidad exacta `HasDerivAt (huber δ) (psi δ r) r` en todo `ℝ`;
- desigualdad de subgradiente, convexidad global en `ℝ` y convexidad estricta en `(-δ, δ)`.
-/

namespace SMCFHR

/-- Pérdida de Huber con umbral `δ`. -/
noncomputable def huber (δ r : ℝ) : ℝ :=
  if |r| ≤ δ then (1 / 2 : ℝ) * r ^ 2 else δ * (|r| - (1 / 2 : ℝ) * δ)

/-- Derivada de la pérdida de Huber (función de influencia acotada). -/
noncomputable def psi (δ r : ℝ) : ℝ := max (-δ) (min r δ)

theorem huber_zero (δ : ℝ) (hδ : 0 ≤ δ) : huber δ 0 = 0 := by
  unfold huber
  simp [abs_zero, hδ]

theorem huber_even (δ r : ℝ) : huber δ (-r) = huber δ r := by
  unfold huber
  simp [abs_neg]

theorem huber_nonneg {δ : ℝ} (hδ : 0 < δ) (r : ℝ) : 0 ≤ huber δ r := by
  unfold huber
  split_ifs with h
  · positivity
  · have hr : δ < |r| := lt_of_not_ge h
    have h1 : 0 ≤ |r| - (1 / 2 : ℝ) * δ := by linarith
    exact mul_nonneg (le_of_lt hδ) h1

theorem huber_linear_lower_bound {δ : ℝ} (hδ : 0 < δ) (r : ℝ) :
    δ * (|r| - (1 / 2 : ℝ) * δ) ≤ huber δ r := by
  unfold huber
  split_ifs with h
  · have hsq : 0 ≤ (|r| - δ) ^ 2 := sq_nonneg (|r| - δ)
    have hr2 : |r| ^ 2 = r ^ 2 := sq_abs r
    linarith
  · exact le_rfl

theorem psi_abs_le {δ : ℝ} (hδ : 0 < δ) (r : ℝ) : |psi δ r| ≤ δ := by
  unfold psi
  rw [abs_le]
  constructor
  · exact le_max_left (-δ) (min r δ)
  · exact max_le (by linarith) (min_le_right r δ)

theorem psi_mono (δ : ℝ) : Monotone (psi δ) := by
  intro a b hab
  unfold psi
  exact max_le_max_left (-δ) (min_le_min_right δ hab)

theorem psi_eq_of_abs_le {δ r : ℝ} (h : |r| ≤ δ) : psi δ r = r := by
  rw [abs_le] at h
  unfold psi
  rw [min_eq_left h.2, max_eq_right h.1]

theorem psi_eq_pos {δ r : ℝ} (hδ : 0 ≤ δ) (h : δ ≤ r) : psi δ r = δ := by
  unfold psi
  rw [min_eq_right h]
  exact max_eq_right (by linarith)

theorem psi_eq_neg {δ r : ℝ} (_hδ : 0 ≤ δ) (h : r ≤ -δ) : psi δ r = -δ := by
  unfold psi
  rw [min_eq_left (by linarith)]
  exact max_eq_left h

theorem psi_lipschitz {δ : ℝ} (hδ : 0 < δ) : LipschitzWith 1 (psi δ) := by
  rw [lipschitzWith_iff_dist_le_mul]
  intro x y
  simp only [NNReal.coe_one, one_mul, Real.dist_eq]
  rw [abs_le]
  have h1 : ∀ a b : ℝ, psi δ a - psi δ b ≤ |a - b| := by
    intro a b
    unfold psi
    rcases le_total a (-δ) with ha1 | ha1
    · rw [min_eq_left (by linarith), max_eq_left ha1]
      have hb : -δ ≤ max (-δ) (min b δ) := le_max_left _ _
      linarith [abs_nonneg (a - b)]
    · rcases le_total a δ with ha2 | ha2
      · rw [min_eq_left ha2, max_eq_right ha1]
        rcases le_total b (-δ) with hb1 | hb1
        · rw [min_eq_left (by linarith), max_eq_left hb1]; linarith [le_abs_self (a - b)]
        · rcases le_total b δ with hb2 | hb2
          · rw [min_eq_left hb2, max_eq_right hb1]; exact le_abs_self (a - b)
          · rw [min_eq_right hb2, max_eq_right (by linarith)]; linarith [abs_nonneg (a - b)]
      · rw [min_eq_right ha2, max_eq_right (by linarith)]
        rcases le_total b (-δ) with hb1 | hb1
        · rw [min_eq_left (by linarith), max_eq_left hb1]; linarith [le_abs_self (a - b)]
        · rcases le_total b δ with hb2 | hb2
          · rw [min_eq_left hb2, max_eq_right hb1]; linarith [le_abs_self (a - b)]
          · rw [min_eq_right hb2, max_eq_right (by linarith)]; linarith [abs_nonneg (a - b)]
  constructor
  · have := h1 y x
    rw [abs_sub_comm] at this
    linarith
  · exact h1 x y

theorem hasDerivAt_huber {δ : ℝ} (hδ : 0 < δ) (r : ℝ) :
    HasDerivAt (huber δ) (psi δ r) r := by
  have huber_at_neg : huber δ (-δ) = (1 / 2 : ℝ) * δ ^ 2 := by
    unfold huber; simp [abs_neg, abs_of_pos hδ]
  have huber_at_pos : huber δ δ = (1 / 2 : ℝ) * δ ^ 2 := by
    unfold huber; simp [abs_of_pos hδ]
  rcases lt_trichotomy r (-δ) with hlt | heq | hgt
  · have h_ev : huber δ =ᶠ[nhds r] fun t => δ * (-t - (1 / 2 : ℝ) * δ) := by
      filter_upwards [eventually_lt_nhds hlt] with t ht
      unfold huber
      split_ifs with h
      · linarith [(abs_le.mp h).1]
      · rw [abs_of_neg (by linarith)]
    have h_d : HasDerivAt (fun t : ℝ => δ * (-t - (1 / 2 : ℝ) * δ)) (-δ) r := by
      have h_sub : HasDerivAt (fun t : ℝ => -t - (1 / 2 : ℝ) * δ) (-1) r :=
        (hasDerivAt_id r).neg.sub_const ((1 / 2 : ℝ) * δ)
      have h_mul := h_sub.const_mul δ
      convert h_mul using 1
      ring
    rw [psi_eq_neg (le_of_lt hδ) (le_of_lt hlt)]
    exact h_d.congr_of_eventuallyEq h_ev
  · rw [heq, psi_eq_neg (le_of_lt hδ) le_rfl, hasDerivAt_iff_tendsto_slope_zero]
    have h_eq : (fun t : ℝ => t⁻¹ • (huber δ (-δ + t) - huber δ (-δ))) =ᶠ[nhdsWithin 0 ({0}ᶜ : Set ℝ)]
        fun t => -δ + (if 0 ≤ t then (1 / 2 : ℝ) * t else 0) := by
      filter_upwards [self_mem_nhdsWithin, nhdsWithin_le_nhds (Metric.ball_mem_nhds (0 : ℝ) hδ)] with t ht0 ht
      simp only [Set.mem_compl_iff, Set.mem_singleton_iff] at ht0
      simp only [Metric.mem_ball, dist_zero_right, Real.norm_eq_abs, abs_lt] at ht
      rw [smul_eq_mul, huber_at_neg]
      unfold huber
      rcases le_or_gt 0 t with ht_pos | ht_neg
      · have habs : |-δ + t| ≤ δ := by rw [abs_le]; constructor <;> linarith
        simp only [habs, ht_pos, ↓reduceIte]
        field_simp [ht0]; ring
      · have hnot : ¬ |-δ + t| ≤ δ := fun h => by linarith [(abs_le.mp h).1]
        have hnot2 : ¬ 0 ≤ t := not_le_of_gt ht_neg
        simp only [hnot, hnot2, ↓reduceIte]
        rw [abs_of_neg (by linarith)]; field_simp [ht0]; ring
    have h_lim : Filter.Tendsto (fun t : ℝ => -δ + (if 0 ≤ t then (1 / 2 : ℝ) * t else 0))
        (nhdsWithin 0 ({0}ᶜ : Set ℝ)) (nhds (-δ)) := by
      have h_abs : ∀ t : ℝ, |-δ + (if 0 ≤ t then (1 / 2 : ℝ) * t else 0) - (-δ)| ≤ |t| := by
        intro t
        split_ifs <;> simp [abs_mul] <;> linarith [abs_nonneg t]
      rw [Metric.tendsto_nhdsWithin_nhds]
      intro ε hε
      refine ⟨ε, hε, fun {t} _ ht => ?_⟩
      rw [Real.dist_eq, sub_zero] at ht
      rw [Real.dist_eq]
      exact lt_of_le_of_lt (h_abs t) ht
    exact h_lim.congr' h_eq.symm
  · rcases lt_trichotomy r δ with hlt2 | heq2 | hgt2
    · have h_ev : huber δ =ᶠ[nhds r] fun t => (1 / 2 : ℝ) * t ^ 2 := by
        filter_upwards [eventually_gt_nhds hgt, eventually_lt_nhds hlt2] with t ht1 ht2
        unfold huber
        have habs : |t| ≤ δ := by rw [abs_le]; constructor <;> linarith
        simp only [habs, ↓reduceIte]
      have h_d : HasDerivAt (fun t : ℝ => (1 / 2 : ℝ) * t ^ 2) r r := by
        have h_p := (hasDerivAt_pow 2 r).const_mul (1 / 2 : ℝ)
        convert h_p using 1
        ring
      rw [psi_eq_of_abs_le (by rw [abs_le]; constructor <;> linarith)]
      exact h_d.congr_of_eventuallyEq h_ev
    · rw [heq2, psi_eq_pos (le_of_lt hδ) le_rfl, hasDerivAt_iff_tendsto_slope_zero]
      have h_eq : (fun t : ℝ => t⁻¹ • (huber δ (δ + t) - huber δ δ)) =ᶠ[nhdsWithin 0 ({0}ᶜ : Set ℝ)]
          fun t => δ + (if t ≤ 0 then (1 / 2 : ℝ) * t else 0) := by
        filter_upwards [self_mem_nhdsWithin, nhdsWithin_le_nhds (Metric.ball_mem_nhds (0 : ℝ) hδ)] with t ht0 ht
        simp only [Set.mem_compl_iff, Set.mem_singleton_iff] at ht0
        simp only [Metric.mem_ball, dist_zero_right, Real.norm_eq_abs, abs_lt] at ht
        rw [smul_eq_mul, huber_at_pos]
        unfold huber
        rcases le_or_gt t 0 with ht_neg | ht_pos
        · have habs : |δ + t| ≤ δ := by rw [abs_le]; constructor <;> linarith
          simp only [habs, ht_neg, ↓reduceIte]
          field_simp [ht0]; ring
        · have hnot : ¬ |δ + t| ≤ δ := fun h => by linarith [(abs_le.mp h).2]
          have hnot2 : ¬ t ≤ 0 := not_le_of_gt ht_pos
          simp only [hnot, hnot2, ↓reduceIte]
          rw [abs_of_pos (by linarith)]; field_simp [ht0]; ring
      have h_lim : Filter.Tendsto (fun t : ℝ => δ + (if t ≤ 0 then (1 / 2 : ℝ) * t else 0))
          (nhdsWithin 0 ({0}ᶜ : Set ℝ)) (nhds δ) := by
        have h_abs : ∀ t : ℝ, |δ + (if t ≤ 0 then (1 / 2 : ℝ) * t else 0) - δ| ≤ |t| := by
          intro t
          split_ifs <;> simp [abs_mul] <;> linarith [abs_nonneg t]
        rw [Metric.tendsto_nhdsWithin_nhds]
        intro ε hε
        refine ⟨ε, hε, fun {t} _ ht => ?_⟩
        rw [Real.dist_eq, sub_zero] at ht
        rw [Real.dist_eq]
        exact lt_of_le_of_lt (h_abs t) ht
      exact h_lim.congr' h_eq.symm
    · have h_ev : huber δ =ᶠ[nhds r] fun t => δ * (t - (1 / 2 : ℝ) * δ) := by
        filter_upwards [eventually_gt_nhds hgt2] with t ht
        unfold huber
        split_ifs with h
        · linarith [(abs_le.mp h).2]
        · rw [abs_of_pos (by linarith)]
      have h_d : HasDerivAt (fun t : ℝ => δ * (t - (1 / 2 : ℝ) * δ)) δ r := by
        have h_sub : HasDerivAt (fun t : ℝ => t - (1 / 2 : ℝ) * δ) 1 r :=
          (hasDerivAt_id r).sub_const ((1 / 2 : ℝ) * δ)
        have h_mul := h_sub.const_mul δ
        convert h_mul using 1
        ring
      rw [psi_eq_pos (le_of_lt hδ) (le_of_lt hgt2)]
      exact h_d.congr_of_eventuallyEq h_ev

theorem huber_subgradient {δ : ℝ} (hδ : 0 < δ) (x y : ℝ) :
    huber δ x + psi δ x * (y - x) ≤ huber δ y := by
  rcases le_total (|x|) δ with hx | hx
  · rw [psi_eq_of_abs_le hx]
    have hx_val : huber δ x = (1 / 2 : ℝ) * x ^ 2 := by
      unfold huber; split_ifs with h; rfl; exact absurd hx h
    rw [hx_val]
    rcases le_total (|y|) δ with hy | hy
    · have hy_val : huber δ y = (1 / 2 : ℝ) * y ^ 2 := by
        unfold huber; split_ifs with h; rfl; exact absurd hy h
      rw [hy_val]
      nlinarith [sq_nonneg (y - x)]
    · have h_lin := huber_linear_lower_bound hδ y
      have h_aux : (1 / 2 : ℝ) * x ^ 2 + x * (y - x) ≤ δ * (|y| - (1 / 2 : ℝ) * δ) := by
        have hx_b := abs_le.mp hx
        rcases le_total 0 y with hy0 | hy0
        · rw [abs_of_nonneg hy0]
          have hy_ge : δ ≤ y := by rwa [abs_of_nonneg hy0] at hy
          have h_prod : 0 ≤ (δ - x) * (y - δ) := mul_nonneg (by linarith) (by linarith)
          nlinarith [sq_nonneg (x - δ)]
        · rw [abs_of_nonpos hy0]
          have hy_le : y ≤ -δ := by rw [abs_of_nonpos hy0] at hy; linarith
          have h_prod : (x + δ) * (y + δ) ≤ 0 := mul_nonpos_of_nonneg_of_nonpos (by linarith) (by linarith)
          nlinarith [sq_nonneg (x + δ)]
      linarith
  · have h_lin := huber_linear_lower_bound hδ y
    rcases le_total x 0 with hx0 | hx0
    · have hx_neg : x ≤ -δ := by rw [abs_of_nonpos hx0] at hx; linarith
      rw [psi_eq_neg (le_of_lt hδ) hx_neg]
      have hx_val : huber δ x = δ * (-x - (1 / 2 : ℝ) * δ) := by
        unfold huber
        split_ifs with h
        · have heq : |x| = δ := le_antisymm h hx
          rw [abs_of_nonpos hx0] at heq
          nlinarith [sq_nonneg (x + δ)]
        · rw [abs_of_nonpos hx0]
      rw [hx_val]
      have h_abs : -y ≤ |y| := neg_le_abs y
      nlinarith [le_of_lt hδ]
    · have hx_pos : δ ≤ x := by rw [abs_of_nonneg hx0] at hx; exact hx
      rw [psi_eq_pos (le_of_lt hδ) hx_pos]
      have hx_val : huber δ x = δ * (x - (1 / 2 : ℝ) * δ) := by
        unfold huber
        split_ifs with h
        · have heq : |x| = δ := le_antisymm h hx
          rw [abs_of_nonneg hx0] at heq
          nlinarith [sq_nonneg (x - δ)]
        · rw [abs_of_nonneg hx0]
      rw [hx_val]
      have h_abs : y ≤ |y| := le_abs_self y
      nlinarith [le_of_lt hδ]

theorem convexOn_huber {δ : ℝ} (hδ : 0 < δ) : ConvexOn ℝ Set.univ (huber δ) := by
  refine ⟨convex_univ, fun x _ y _ a b ha hb hab => ?_⟩
  let z := a * x + b * y
  have hx := huber_subgradient hδ z x
  have hy := huber_subgradient hδ z y
  have h1 : a * (huber δ z + psi δ z * (x - z)) ≤ a * huber δ x :=
    mul_le_mul_of_nonneg_left hx ha
  have h2 : b * (huber δ z + psi δ z * (y - z)) ≤ b * huber δ y :=
    mul_le_mul_of_nonneg_left hy hb
  have hz : a * (x - z) + b * (y - z) = 0 := by
    dsimp [z]
    calc a * (x - (a * x + b * y)) + b * (y - (a * x + b * y))
        = (a * x + b * y) - (a + b) * (a * x + b * y) := by ring
    _ = 0 := by rw [hab, one_mul, sub_self]
  have hcomb : (a + b) * huber δ z + psi δ z * (a * (x - z) + b * (y - z)) ≤
      a * huber δ x + b * huber δ y := by linarith
  rw [hab, one_mul, hz, mul_zero, add_zero] at hcomb
  simpa [smul_eq_mul, z] using hcomb

theorem strictConvexOn_huber_band {δ : ℝ} (hδ : 0 < δ) :
    StrictConvexOn ℝ (Set.Ioo (-δ) δ) (huber δ) := by
  refine ⟨convex_Ioo (-δ) δ, fun x hx y hy hxy a b ha hb hab => ?_⟩
  have h_eq : ∀ r ∈ Set.Ioo (-δ) δ, huber δ r = (1 / 2 : ℝ) * r ^ 2 := by
    intro r hr
    unfold huber
    have habs : |r| ≤ δ := by rw [abs_le]; constructor <;> linarith [hr.1, hr.2]
    simp only [habs, ↓reduceIte]
  have h_mid : a • x + b • y ∈ Set.Ioo (-δ) δ :=
    (convex_Ioo (-δ) δ) hx hy (le_of_lt ha) (le_of_lt hb) hab
  rw [h_eq _ h_mid, h_eq x hx, h_eq y hy]
  simp only [smul_eq_mul]
  have h_diff : 0 < a * b * (x - y) ^ 2 :=
    mul_pos (mul_pos ha hb) (sq_pos_of_ne_zero (sub_ne_zero.mpr hxy))
  have h_id : a * ((1 / 2 : ℝ) * x ^ 2) + b * ((1 / 2 : ℝ) * y ^ 2) -
      (1 / 2 : ℝ) * (a * x + b * y) ^ 2 = (1 / 2 : ℝ) * (a * b * (x - y) ^ 2) := by
    calc a * ((1 / 2 : ℝ) * x ^ 2) + b * ((1 / 2 : ℝ) * y ^ 2) - (1 / 2 : ℝ) * (a * x + b * y) ^ 2
        = (1 / 2 : ℝ) * (a * (1 - a) * x ^ 2 + b * (1 - b) * y ^ 2 - 2 * a * b * x * y) := by ring
    _ = (1 / 2 : ℝ) * (a * b * x ^ 2 + b * a * y ^ 2 - 2 * a * b * x * y) := by
          rw [show 1 - a = b by linarith, show 1 - b = a by linarith]
    _ = (1 / 2 : ℝ) * (a * b * (x - y) ^ 2) := by ring
  linarith

end SMCFHR

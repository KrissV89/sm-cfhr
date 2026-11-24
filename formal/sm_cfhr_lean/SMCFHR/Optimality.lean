import Mathlib.Topology.Compactness.Compact
import Mathlib.Tactic
import SMCFHR.Huber
import SMCFHR.Model
import SMCFHR.Gradient

/-!
# Existencia, unicidad, contraejemplo del intercepto y condiciones KKT

Demuestra:
- coercividad de `objective` cuando `n ≥ 1`, `δ > 0` y `α j > 0` para todo `j`;
- existencia de al menos un minimizador global en `FeasibleSet M`;
- unicidad del vector de coeficientes `w*` entre cualesquiera dos minimizadores;
- unicidad de `b*` si existe al menos una observación con residuo interior `|r_k| < δ`;
- contraejemplo explícito de no unicidad de `b*` cuando todos los residuos cumplen `|r_k| ≥ δ`;
- caracterización KKT exacta (necesidad y suficiencia) para optimalidad en `FeasibleSet M`.
-/

namespace SMCFHR

private lemma continuous_sum_finset {α β : Type*} [TopologicalSpace α] [TopologicalSpace β]
    [AddCommMonoid β] [ContinuousAdd β] {ι : Type*} (s : Finset ι) (f : ι → α → β)
    (hf : ∀ i ∈ s, Continuous (f i)) : Continuous (fun x => ∑ i ∈ s, f i x) := by
  classical
  induction s using Finset.induction_on with
  | empty => simp [continuous_const]
  | @insert a s ha ih =>
    have h1 : (fun x => ∑ i ∈ insert a s, f i x) = fun x => f a x + ∑ i ∈ s, f i x := by
      ext x; rw [Finset.sum_insert ha]
    rw [h1]
    exact (hf a (Finset.mem_insert_self a s)).add (ih fun i hi => hf i (Finset.mem_insert_of_mem hi))

/-- Continuidad de `objective` respecto a los parámetros `θ`. -/
theorem continuous_objective {n p : ℕ} (δ : ℝ) (hδ : 0 < δ)
    (Z : Fin n → Fin p → ℝ) (y : Fin n → ℝ) (α : Fin p → ℝ) :
    Continuous (objective δ Z y α) := by
  have hhub_cont : Continuous (huber δ) :=
    continuous_iff_continuousAt.mpr fun r => (hasDerivAt_huber hδ r).continuousAt
  unfold objective
  refine Continuous.add ?_ ?_
  · refine continuous_const.mul (continuous_sum_finset Finset.univ (fun k θ => huber δ (resid Z y θ k)) fun k _ => hhub_cont.comp ?_)
    unfold resid
    refine continuous_const.sub (continuous_fst.add ?_)
    exact continuous_sum_finset Finset.univ (fun j (θ : Params p) => Z k j * θ.2 j) fun j _ =>
      continuous_const.mul ((continuous_apply j).comp continuous_snd)
  · refine continuous_const.mul (continuous_sum_finset Finset.univ (fun j (θ : Params p) => α j * (θ.2 j) ^ 2) fun j _ => ?_)
    exact continuous_const.mul (((continuous_apply j).comp continuous_snd).pow 2)

/-- Coercividad: fuera de una caja suficientemente grande, `objective > C`. -/
theorem objective_coercive {n p : ℕ} (hn : 0 < n) {δ : ℝ} (hδ : 0 < δ)
    (Z : Fin n → Fin p → ℝ) (y : Fin n → ℝ)
    {α : Fin p → ℝ} (hα : ∀ j, 0 < α j) (C : ℝ) :
    ∃ R : ℝ, 0 < R ∧ ∀ θ : Params p,
      (R < |θ.1| ∨ ∃ j : Fin p, R < |θ.2 j|) → C < objective δ Z y α θ := by
  let k0 : Fin n := ⟨0, hn⟩
  let M_z : ℝ := ∑ j : Fin p, |Z k0 j|
  let α_min : ℝ := if hp : p = 0 then 1 else Finset.inf' Finset.univ (by
    rw [Finset.univ_nonempty_iff]; exact ⟨⟨0, Nat.pos_of_ne_zero hp⟩⟩) α
  have hα_min_pos : 0 < α_min := by
    dsimp [α_min]
    split_ifs with hp
    · norm_num
    · rw [Finset.lt_inf'_iff]
      intro j _
      exact hα j
  have hα_le : ∀ j : Fin p, α_min ≤ α j := by
    intro j
    dsimp [α_min]
    split_ifs with hp
    · exact absurd j.isLt (by omega)
    · exact Finset.inf'_le _ (Finset.mem_univ j)
  -- Cota para w_j: si |w_j| > R_w entonces (1/2) * α_min * R_w^2 > C
  let C_pos : ℝ := max C 0 + 1
  let R_w : ℝ := Real.sqrt (2 * C_pos / α_min) + 1
  have hRw_pos : 0 < R_w := by positivity
  -- Cota para b: si |w_j| ≤ R_w para todo j y |b| > R_b entonces el término k0 excede C
  let R_b : ℝ := |y k0| + M_z * R_w + (n : ℝ) * (C_pos / δ + (1 / 2 : ℝ) * δ) + 1
  let R : ℝ := max R_w R_b
  refine ⟨R, lt_of_lt_of_le hRw_pos (le_max_left _ _), fun θ h_out => ?_⟩
  have h_nonneg1 : 0 ≤ ((1 : ℝ) / n) * ∑ k : Fin n, huber δ (resid Z y θ k) := by
    refine mul_nonneg (by positivity) (Finset.sum_nonneg fun k _ => huber_nonneg hδ _)
  have h_nonneg2 : 0 ≤ (1 / 2 : ℝ) * ∑ j : Fin p, α j * (θ.2 j) ^ 2 := by
    refine mul_nonneg (by norm_num) (Finset.sum_nonneg fun j _ => mul_nonneg (le_of_lt (hα j)) (sq_nonneg _))
  by_cases h_w_big : ∃ j : Fin p, R_w < |θ.2 j|
  · rcases h_w_big with ⟨j0, hj0⟩
    have hj0_sq : R_w ^ 2 < (θ.2 j0) ^ 2 := by
      have := sq_lt_sq' (by linarith) hj0
      simpa [sq_abs] using this
    have h_term : C_pos ≤ (1 / 2 : ℝ) * (α j0 * (θ.2 j0) ^ 2) := by
      have hsq_bound : 2 * C_pos / α_min < (θ.2 j0) ^ 2 := by
        have h1 : Real.sqrt (2 * C_pos / α_min) < R_w := by linarith
        have h2 : 2 * C_pos / α_min ≤ (Real.sqrt (2 * C_pos / α_min)) ^ 2 :=
          le_of_eq (Real.sq_sqrt (by positivity)).symm
        have h3 : (Real.sqrt (2 * C_pos / α_min)) ^ 2 < R_w ^ 2 := by
          nlinarith [Real.sqrt_nonneg (2 * C_pos / α_min)]
        linarith
      have hα_pos : α_min ≤ α j0 := hα_le j0
      calc C_pos = (1 / 2 : ℝ) * (α_min * (2 * C_pos / α_min)) := by field_simp [ne_of_gt hα_min_pos]
      _ ≤ (1 / 2 : ℝ) * (α j0 * (θ.2 j0) ^ 2) := by
          refine mul_le_mul_of_nonneg_left ?_ (by norm_num)
          exact mul_le_mul hα_pos (le_of_lt hsq_bound) (by positivity) (le_of_lt (hα j0))
    have h_sum_ge : α j0 * (θ.2 j0) ^ 2 ≤ ∑ j : Fin p, α j * (θ.2 j) ^ 2 :=
      Finset.single_le_sum (fun j _ => mul_nonneg (le_of_lt (hα j)) (sq_nonneg _)) (Finset.mem_univ j0)
    unfold objective
    have h_obj_ge : C_pos ≤ (1 / 2 : ℝ) * ∑ j : Fin p, α j * (θ.2 j) ^ 2 := by linarith
    have hC_lt : C < C_pos := by dsimp [C_pos]; linarith [le_max_left C 0]
    linarith
  · push_neg at h_w_big
    have hb_big : R_b < |θ.1| := by
      rcases h_out with hb | ⟨j, hj⟩
      · exact lt_of_le_of_lt (le_max_right R_w R_b) hb
      · exact absurd (lt_of_le_of_lt (le_max_left R_w R_b) hj) (not_lt_of_ge (h_w_big j))
    have h_lin_abs : |∑ j : Fin p, Z k0 j * θ.2 j| ≤ M_z * R_w := by
      calc |∑ j : Fin p, Z k0 j * θ.2 j|
          ≤ ∑ j : Fin p, |Z k0 j * θ.2 j| := Finset.abs_sum_le_sum_abs _ _
      _ = ∑ j : Fin p, |Z k0 j| * |θ.2 j| := by
            refine Finset.sum_congr rfl fun j _ => abs_mul _ _
      _ ≤ ∑ j : Fin p, |Z k0 j| * R_w := by
            refine Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (h_w_big j) (abs_nonneg _)
      _ = M_z * R_w := by rw [← Finset.sum_mul]
    have hr_big : (n : ℝ) * (C_pos / δ + (1 / 2 : ℝ) * δ) < |resid Z y θ k0| := by
      unfold resid
      have h_tri : |θ.1| - (|y k0| + |∑ j : Fin p, Z k0 j * θ.2 j|) ≤
          |y k0 - (θ.1 + ∑ j : Fin p, Z k0 j * θ.2 j)| := by
        have h1 : |θ.1| ≤ |θ.1 + ∑ j : Fin p, Z k0 j * θ.2 j| + |∑ j : Fin p, Z k0 j * θ.2 j| := by
          have := abs_add_le (θ.1 + ∑ j : Fin p, Z k0 j * θ.2 j) (-∑ j : Fin p, Z k0 j * θ.2 j)
          simpa [abs_neg] using this
        have h2 : |θ.1 + ∑ j : Fin p, Z k0 j * θ.2 j| ≤
            |y k0 - (θ.1 + ∑ j : Fin p, Z k0 j * θ.2 j)| + |y k0| := by
          have := abs_add_le (y k0) (-(y k0 - (θ.1 + ∑ j : Fin p, Z k0 j * θ.2 j)))
          simpa [abs_neg, abs_sub_comm, add_comm] using this
        linarith
      dsimp [R_b] at hb_big
      linarith
    have hhub_big : (n : ℝ) * C_pos < huber δ (resid Z y θ k0) := by
      have h_lb := huber_linear_lower_bound hδ (resid Z y θ k0)
      have hn_pos : (0 : ℝ) < n := Nat.cast_pos.mpr hn
      have h_calc : (n : ℝ) * C_pos < δ * (|resid Z y θ k0| - (1 / 2 : ℝ) * δ) := by
        have h_mul : δ * ((n : ℝ) * (C_pos / δ + (1 / 2 : ℝ) * δ)) < δ * |resid Z y θ k0| :=
          mul_lt_mul_of_pos_left hr_big hδ
        have h_eq : δ * ((n : ℝ) * (C_pos / δ + (1 / 2 : ℝ) * δ)) =
            (n : ℝ) * C_pos + (n : ℝ) * ((1 / 2 : ℝ) * δ ^ 2) := by
          field_simp [ne_of_gt hδ]
        have hn1 : (1 : ℝ) ≤ n := Nat.one_le_cast.mpr hn
        have h_half : δ * ((1 / 2 : ℝ) * δ) ≤ (n : ℝ) * ((1 / 2 : ℝ) * δ ^ 2) := by
          nlinarith [sq_nonneg δ]
        linarith
      linarith
    have h_sum_ge : huber δ (resid Z y θ k0) ≤ ∑ k : Fin n, huber δ (resid Z y θ k) :=
      Finset.single_le_sum (fun k _ => huber_nonneg hδ _) (Finset.mem_univ k0)
    have h_avg : C_pos < ((1 : ℝ) / n) * ∑ k : Fin n, huber δ (resid Z y θ k) := by
      have hn_pos : (0 : ℝ) < n := Nat.cast_pos.mpr hn
      rw [one_div, lt_inv_mul_iff₀ hn_pos]
      linarith
    unfold objective
    have hC_lt : C < C_pos := by dsimp [C_pos]; linarith [le_max_left C 0]
    linarith

/-- Existencia de al menos un minimizador global sobre `FeasibleSet M`. -/
theorem exists_global_minimizer {n p : ℕ} (hn : 0 < n) {δ : ℝ} (hδ : 0 < δ)
    (Z : Fin n → Fin p → ℝ) (y : Fin n → ℝ)
    {α : Fin p → ℝ} (hα : ∀ j, 0 < α j) (M : Finset (Fin p)) :
    ∃ θ_star ∈ FeasibleSet M, IsMinOn (objective δ Z y α) (FeasibleSet M) θ_star := by
  let θ0 : Params p := ((0 : ℝ), fun _ => (0 : ℝ))
  have hθ0 : θ0 ∈ FeasibleSet M := zero_mem_feasibleSet M
  obtain ⟨R, hR_pos, h_coer⟩ := objective_coercive hn hδ Z y hα (objective δ Z y α θ0)
  let Box : Set (Params p) :=
    (Set.Icc (-R) R) ×ˢ (Set.pi Set.univ fun _ => Set.Icc (-R) R)
  have hBox_compact : IsCompact Box :=
    isCompact_Icc.prod (isCompact_univ_pi fun _ => isCompact_Icc)
  have hS_compact : IsCompact (FeasibleSet M ∩ Box) :=
    hBox_compact.inter_left (isClosed_feasibleSet M)
  have hθ0_box : θ0 ∈ Box := by
    refine ⟨⟨by linarith, by linarith⟩, fun _ _ => ⟨by linarith, by linarith⟩⟩
  have h_ne : (FeasibleSet M ∩ Box).Nonempty := ⟨θ0, hθ0, hθ0_box⟩
  obtain ⟨θ_star, ⟨hθ_feas, hθ_box⟩, hmin⟩ :=
    hS_compact.exists_isMinOn h_ne (continuous_objective δ hδ Z y α).continuousOn
  refine ⟨θ_star, hθ_feas, fun θ hθ => ?_⟩
  by_cases h_in : θ ∈ Box
  · exact hmin ⟨hθ, h_in⟩
  · have h_out : R < |θ.1| ∨ ∃ j : Fin p, R < |θ.2 j| := by
      by_contra h_neg
      push_neg at h_neg
      apply h_in
      refine ⟨abs_le.mp h_neg.1, fun j _ => abs_le.mp (h_neg.2 j)⟩
    have h_gt := h_coer θ h_out
    have h_le0 : objective δ Z y α θ_star ≤ objective δ Z y α θ0 := hmin ⟨hθ0, hθ0_box⟩
    exact le_trans h_le0 (le_of_lt h_gt)

/-- Unicidad del vector de coeficientes `w*` entre cualesquiera dos minimizadores globales. -/
theorem unique_weights {n p : ℕ} {δ : ℝ} (hδ : 0 < δ)
    (Z : Fin n → Fin p → ℝ) (y : Fin n → ℝ)
    {α : Fin p → ℝ} (hα : ∀ j, 0 < α j) (M : Finset (Fin p))
    {θ₁ θ₂ : Params p} (h₁ : θ₁ ∈ FeasibleSet M) (h₂ : θ₂ ∈ FeasibleSet M)
    (hm₁ : IsMinOn (objective δ Z y α) (FeasibleSet M) θ₁)
    (hm₂ : IsMinOn (objective δ Z y α) (FeasibleSet M) θ₂) :
    θ₁.2 = θ₂.2 := by
  by_contra h_ne
  have h_ex : ∃ j : Fin p, θ₁.2 j ≠ θ₂.2 j := by
    by_contra h_all; push_neg at h_all; exact h_ne (funext h_all)
  rcases h_ex with ⟨j0, hj0⟩
  let θ_mid : Params p := ((1 / 2 : ℝ) • θ₁ + (1 / 2 : ℝ) • θ₂)
  have h_mid_feas : θ_mid ∈ FeasibleSet M :=
    convex_feasibleSet M h₁ h₂ (by norm_num) (by norm_num) (by norm_num)
  have h_val_eq : objective δ Z y α θ₁ = objective δ Z y α θ₂ :=
    le_antisymm (hm₁ h₂) (hm₂ h₁)
  -- Desigualdad estricta en el término cuadrático para j0
  have h_sq_strict : (1 / 2 : ℝ) * (((1 / 2 : ℝ) * θ₁.2 j0 + (1 / 2 : ℝ) * θ₂.2 j0) ^ 2) <
      (1 / 2 : ℝ) * ((1 / 2 : ℝ) * (θ₁.2 j0) ^ 2 + (1 / 2 : ℝ) * (θ₂.2 j0) ^ 2) := by
    have hpos : 0 < (θ₁.2 j0 - θ₂.2 j0) ^ 2 := sq_pos_of_ne_zero (sub_ne_zero.mpr hj0)
    linarith
  have h_hub_mid : ∑ k : Fin n, huber δ (resid Z y θ_mid k) ≤
      (1 / 2 : ℝ) * ∑ k : Fin n, huber δ (resid Z y θ₁ k) +
        (1 / 2 : ℝ) * ∑ k : Fin n, huber δ (resid Z y θ₂ k) := by
    calc ∑ k : Fin n, huber δ (resid Z y θ_mid k)
        ≤ ∑ k : Fin n, ((1 / 2 : ℝ) * huber δ (resid Z y θ₁ k) +
            (1 / 2 : ℝ) * huber δ (resid Z y θ₂ k)) := by
          refine Finset.sum_le_sum fun k _ => ?_
          rw [show θ_mid = (1 / 2 : ℝ) • θ₁ + (1 / 2 : ℝ) • θ₂ from rfl]
          rw [resid_affine Z y θ₁ θ₂ (by norm_num) k]
          exact (convexOn_huber hδ).2 (Set.mem_univ _) (Set.mem_univ _) (by norm_num) (by norm_num) (by norm_num)
    _ = (1 / 2 : ℝ) * ∑ k : Fin n, huber δ (resid Z y θ₁ k) +
          (1 / 2 : ℝ) * ∑ k : Fin n, huber δ (resid Z y θ₂ k) := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
  have h_pen_mid : ∑ j : Fin p, α j * (θ_mid.2 j) ^ 2 <
      (1 / 2 : ℝ) * ∑ j : Fin p, α j * (θ₁.2 j) ^ 2 +
        (1 / 2 : ℝ) * ∑ j : Fin p, α j * (θ₂.2 j) ^ 2 := by
    have h_le : ∀ j : Fin p, α j * (θ_mid.2 j) ^ 2 ≤
        (1 / 2 : ℝ) * (α j * (θ₁.2 j) ^ 2) + (1 / 2 : ℝ) * (α j * (θ₂.2 j) ^ 2) := by
      intro j
      have hsq_j : (θ_mid.2 j) ^ 2 ≤ (1 / 2 : ℝ) * (θ₁.2 j) ^ 2 + (1 / 2 : ℝ) * (θ₂.2 j) ^ 2 := by
        have := sq_nonneg (θ₁.2 j - θ₂.2 j)
        change ((1 / 2 : ℝ) * θ₁.2 j + (1 / 2 : ℝ) * θ₂.2 j) ^ 2 ≤ _
        linarith
      nlinarith [hα j]
    have h_lt_j0 : α j0 * (θ_mid.2 j0) ^ 2 <
        (1 / 2 : ℝ) * (α j0 * (θ₁.2 j0) ^ 2) + (1 / 2 : ℝ) * (α j0 * (θ₂.2 j0) ^ 2) := by
      change α j0 * (((1 / 2 : ℝ) * θ₁.2 j0 + (1 / 2 : ℝ) * θ₂.2 j0) ^ 2) < _
      nlinarith [hα j0]
    have h_sum_lt := Finset.sum_lt_sum (fun j _ => h_le j) ⟨j0, Finset.mem_univ j0, h_lt_j0⟩
    simpa [Finset.sum_add_distrib, ← Finset.mul_sum] using h_sum_lt
  have h_mid_lt : objective δ Z y α θ_mid <
      (1 / 2 : ℝ) * objective δ Z y α θ₁ + (1 / 2 : ℝ) * objective δ Z y α θ₂ := by
    unfold objective
    have hc1 : (0 : ℝ) ≤ (1 : ℝ) / n := by positivity
    have hl1 := mul_le_mul_of_nonneg_left h_hub_mid hc1
    linarith
  have h_ge : objective δ Z y α θ₁ ≤ objective δ Z y α θ_mid := hm₁ h_mid_feas
  linarith

/-- Unicidad del intercepto si existe al menos un residuo interior `|r_k| < δ`. -/
theorem unique_intercept_of_interior_residual {n p : ℕ} (hn : 0 < n) {δ : ℝ} (hδ : 0 < δ)
    (Z : Fin n → Fin p → ℝ) (y : Fin n → ℝ)
    {α : Fin p → ℝ} (hα : ∀ j, 0 < α j) (M : Finset (Fin p))
    {θ₁ θ₂ : Params p} (h₁ : θ₁ ∈ FeasibleSet M) (h₂ : θ₂ ∈ FeasibleSet M)
    (hm₁ : IsMinOn (objective δ Z y α) (FeasibleSet M) θ₁)
    (hm₂ : IsMinOn (objective δ Z y α) (FeasibleSet M) θ₂)
    (h_int : ∃ k0 : Fin n, |resid Z y θ₁ k0| < δ) :
    θ₁.1 = θ₂.1 := by
  have hw_eq : θ₁.2 = θ₂.2 := unique_weights hδ Z y hα M h₁ h₂ hm₁ hm₂
  rcases h_int with ⟨k0, hk0⟩
  by_contra hb_ne
  -- Existe t ∈ (0, 1) suficientemente pequeño tal que r_{k0}(θ_t) ∈ (-δ, δ) también
  have h_gap : 0 < δ - |resid Z y θ₁ k0| := by linarith
  let t : ℝ := min (1 / 2 : ℝ) ((δ - |resid Z y θ₁ k0|) / (2 * (|θ₁.1 - θ₂.1| + 1)))
  have ht_pos : 0 < t := by
    refine lt_min (by norm_num) ?_
    exact div_pos h_gap (by positivity)
  have ht_lt1 : t < 1 := lt_of_le_of_lt (min_le_left _ _) (by norm_num)
  let θ_t : Params p := (1 - t) • θ₁ + t • θ₂
  have h_t_feas : θ_t ∈ FeasibleSet M :=
    convex_feasibleSet M h₁ h₂ (by linarith) (le_of_lt ht_pos) (by ring)
  have h_val_eq : objective δ Z y α θ₁ = objective δ Z y α θ₂ :=
    le_antisymm (hm₁ h₂) (hm₂ h₁)
  -- Si t_eff = t/2, en θ_{t} ambos r_{k0}(θ₁) y r_{k0}(θ_t) están en (-δ, δ) y difieren
  have hr_diff : resid Z y θ₁ k0 ≠ resid Z y θ₂ k0 := by
    unfold resid
    rw [hw_eq]
    intro heq
    apply hb_ne
    linarith
  have hr_k0_t : |resid Z y θ_t k0| < δ := by
    have h_aff := resid_affine Z y θ₁ θ₂ (show (1 - t) + t = 1 by ring) k0
    rw [show (1 - t) • θ₁ + t • θ₂ = θ_t from rfl] at h_aff
    have h_step : resid Z y θ_t k0 - resid Z y θ₁ k0 = t * (θ₁.1 - θ₂.1) := by
      unfold resid at h_aff ⊢
      rw [hw_eq] at h_aff ⊢
      linarith
    have h_bound : t * |θ₁.1 - θ₂.1| ≤ (δ - |resid Z y θ₁ k0|) / 2 := by
      have ht_le : t ≤ (δ - |resid Z y θ₁ k0|) / (2 * (|θ₁.1 - θ₂.1| + 1)) := min_le_right _ _
      calc t * |θ₁.1 - θ₂.1|
          ≤ ((δ - |resid Z y θ₁ k0|) / (2 * (|θ₁.1 - θ₂.1| + 1))) * |θ₁.1 - θ₂.1| :=
            mul_le_mul_of_nonneg_right ht_le (abs_nonneg _)
      _ ≤ ((δ - |resid Z y θ₁ k0|) / (2 * (|θ₁.1 - θ₂.1| + 1))) * (|θ₁.1 - θ₂.1| + 1) :=
            mul_le_mul_of_nonneg_left (by linarith) (div_nonneg (le_of_lt h_gap) (by positivity))
      _ = (δ - |resid Z y θ₁ k0|) / 2 := by
            have : |θ₁.1 - θ₂.1| + 1 ≠ 0 := by positivity
            field_simp [this]
    have := abs_add_le (resid Z y θ₁ k0) (t * (θ₁.1 - θ₂.1))
    rw [show resid Z y θ₁ k0 + t * (θ₁.1 - θ₂.1) = resid Z y θ_t k0 by linarith] at this
    rw [abs_mul, abs_of_pos ht_pos] at this
    linarith
  -- En el punto medio de θ₁ y θ_t, el término k0 de Huber es estrictamente convexo
  let θ_mid : Params p := (1 / 2 : ℝ) • θ₁ + (1 / 2 : ℝ) • θ_t
  have h_mid_feas : θ_mid ∈ FeasibleSet M :=
    convex_feasibleSet M h₁ h_t_feas (by norm_num) (by norm_num) (by norm_num)
  have h_k0_ne : resid Z y θ₁ k0 ≠ resid Z y θ_t k0 := by
    unfold resid
    rw [show θ_t.2 = θ₁.2 by
      ext j; simp [θ_t, hw_eq]; ring]
    intro heq
    have h_t1 : θ_t.1 = (1 - t) * θ₁.1 + t * θ₂.1 := rfl
    have : t * (θ₁.1 - θ₂.1) = 0 := by linarith
    rcases mul_eq_zero.mp this with ht0 | hb0
    · linarith
    · exact hb_ne (sub_eq_zero.mp hb0)
  have h_k0_strict : huber δ (resid Z y θ_mid k0) <
      (1 / 2 : ℝ) * huber δ (resid Z y θ₁ k0) + (1 / 2 : ℝ) * huber δ (resid Z y θ_t k0) := by
    rw [show θ_mid = (1 / 2 : ℝ) • θ₁ + (1 / 2 : ℝ) • θ_t from rfl]
    rw [resid_affine Z y θ₁ θ_t (by norm_num) k0]
    exact (strictConvexOn_huber_band hδ).2 (abs_lt.mp hk0) (abs_lt.mp hr_k0_t)
      h_k0_ne (by norm_num) (by norm_num) (by norm_num)
  have h_other_le : ∀ k : Fin n, huber δ (resid Z y θ_mid k) ≤
      (1 / 2 : ℝ) * huber δ (resid Z y θ₁ k) + (1 / 2 : ℝ) * huber δ (resid Z y θ_t k) := by
    intro k
    rw [show θ_mid = (1 / 2 : ℝ) • θ₁ + (1 / 2 : ℝ) • θ_t from rfl]
    rw [resid_affine Z y θ₁ θ_t (by norm_num) k]
    exact (convexOn_huber hδ).2 (Set.mem_univ _) (Set.mem_univ _) (by norm_num) (by norm_num) (by norm_num)
  have h_sum_lt : ∑ k : Fin n, huber δ (resid Z y θ_mid k) <
      (1 / 2 : ℝ) * ∑ k : Fin n, huber δ (resid Z y θ₁ k) +
        (1 / 2 : ℝ) * ∑ k : Fin n, huber δ (resid Z y θ_t k) := by
    have hs := Finset.sum_lt_sum (fun k _ => h_other_le k) ⟨k0, Finset.mem_univ k0, h_k0_strict⟩
    simpa [Finset.sum_add_distrib, ← Finset.mul_sum] using hs
  have h_pen_eq : θ_t.2 = θ₁.2 ∧ θ_mid.2 = θ₁.2 := by
    constructor <;> ext j <;> simp [θ_t, θ_mid, hw_eq] <;> ring
  have h_t_le : objective δ Z y α θ_t ≤ objective δ Z y α θ₁ := by
    have h_conv := (convexOn_objective hδ Z y (fun j => le_of_lt (hα j))).2
      (Set.mem_univ θ₁) (Set.mem_univ θ₂) (show (0 : ℝ) ≤ 1 - t by linarith)
      (le_of_lt ht_pos) (show (1 - t) + t = 1 by ring)
    change objective δ Z y α θ_t ≤ (1 - t) * objective δ Z y α θ₁ + t * objective δ Z y α θ₂ at h_conv
    rw [← h_val_eq] at h_conv
    linarith
  have h_mid_lt : objective δ Z y α θ_mid <
      (1 / 2 : ℝ) * objective δ Z y α θ₁ + (1 / 2 : ℝ) * objective δ Z y α θ_t := by
    unfold objective
    rw [h_pen_eq.1, h_pen_eq.2]
    have hn_pos : (0 : ℝ) < (1 : ℝ) / n := by positivity
    have hl := mul_lt_mul_of_pos_left h_sum_lt hn_pos
    linarith
  have h_ge : objective δ Z y α θ₁ ≤ objective δ Z y α θ_mid := hm₁ h_mid_feas
  linarith

/-- Contraejemplo explícito de no unicidad del intercepto cuando todo `|r_k| ≥ δ`. -/
theorem intercept_nonunique_counterexample {δ : ℝ} (hδ : 0 < δ) :
    let Z : Fin 2 → Fin 1 → ℝ := fun _ _ => 0
    let y : Fin 2 → ℝ := fun k => if k = 0 then -2 * δ else 2 * δ
    let α : Fin 1 → ℝ := fun _ => 1
    let θ₁ : Params 1 := (0, fun _ => 0)
    let θ₂ : Params 1 := (δ, fun _ => 0)
    θ₁ ≠ θ₂ ∧
      objective δ Z y α θ₁ = (3 / 2 : ℝ) * δ ^ 2 ∧
      objective δ Z y α θ₂ = (3 / 2 : ℝ) * δ ^ 2 ∧
      IsMinOn (objective δ Z y α) (FeasibleSet ∅) θ₁ ∧
      IsMinOn (objective δ Z y α) (FeasibleSet ∅) θ₂ := by
  intro Z y α θ₁ θ₂
  have h_ne : θ₁ ≠ θ₂ := by
    intro h
    have : (0 : ℝ) = δ := congrArg Prod.fst h
    linarith
  have h_eval : ∀ b : ℝ, |b| ≤ δ → objective δ Z y α (b, fun _ => 0) = (3 / 2 : ℝ) * δ ^ 2 := by
    intro b hb
    rw [abs_le] at hb
    unfold objective resid
    simp only [Z, y, α, Fin.sum_univ_two, Fin.sum_univ_one, add_zero,
      Fin.isValue, one_ne_zero, ↓reduceIte, mul_zero, zero_pow (by norm_num : 2 ≠ 0),
      Nat.cast_ofNat]
    have h0 : huber δ (-2 * δ - b) = δ * (2 * δ + b - (1 / 2 : ℝ) * δ) := by
      unfold huber
      by_cases h : |-2 * δ - b| ≤ δ
      · have h_abs := abs_le.mp h
        have hb_eq : b = -δ := by linarith
        subst hb_eq
        simp only [h, ↓reduceIte]
        ring
      · simp only [h, ↓reduceIte]
        rw [abs_of_neg (show -2 * δ - b < 0 by linarith)]
        ring
    have h1 : huber δ (2 * δ - b) = δ * (2 * δ - b - (1 / 2 : ℝ) * δ) := by
      unfold huber
      by_cases h : |2 * δ - b| ≤ δ
      · have h_abs := abs_le.mp h
        have hb_eq : b = δ := by linarith
        subst hb_eq
        simp only [h, ↓reduceIte]
        ring
      · simp only [h, ↓reduceIte]
        rw [abs_of_pos (show 0 < 2 * δ - b by linarith)]
    rw [h0, h1]
    ring
  have h_lb : ∀ θ : Params 1, (3 / 2 : ℝ) * δ ^ 2 ≤ objective δ Z y α θ := by
    intro θ
    have hl0 := huber_linear_lower_bound hδ (resid Z y θ 0)
    have hl1 := huber_linear_lower_bound hδ (resid Z y θ 1)
    have hr0 : resid Z y θ 0 = -2 * δ - θ.1 := by unfold resid; simp [Z, y]
    have hr1 : resid Z y θ 1 = 2 * δ - θ.1 := by unfold resid; simp [Z, y]
    rw [hr0] at hl0
    rw [hr1] at hl1
    have h_sum_abs : 4 * δ ≤ |-2 * δ - θ.1| + |2 * δ - θ.1| := by
      have := abs_add_le (2 * δ - θ.1) (-(-2 * δ - θ.1))
      rw [abs_neg] at this
      have h_eq : (2 * δ - θ.1) + (-(-2 * δ - θ.1)) = 4 * δ := by ring
      rw [h_eq, abs_of_pos (by positivity)] at this
      linarith
    have h_pen : 0 ≤ (1 / 2 : ℝ) * ∑ j : Fin 1, α j * (θ.2 j) ^ 2 := by
      refine mul_nonneg (by norm_num) (Finset.sum_nonneg fun j _ => by simp [α]; positivity)
    unfold objective
    simp only [Nat.cast_ofNat, Fin.sum_univ_two]
    have h_tot : 3 * δ ^ 2 ≤ huber δ (resid Z y θ 0) + huber δ (resid Z y θ 1) := by
      rw [hr0, hr1]
      nlinarith
    linarith
  refine ⟨h_ne, h_eval 0 (by simp [le_of_lt hδ]), h_eval δ (by rw [abs_of_pos hδ]), ?_, ?_⟩
  · intro θ _
    rw [h_eval 0 (by simp [le_of_lt hδ])]
    exact h_lb θ
  · intro θ _
    rw [h_eval δ (by rw [abs_of_pos hδ])]
    exact h_lb θ

private lemma deriv_nonneg_of_local_min_right {f : ℝ → ℝ} {L : ℝ}
    (hf : HasDerivAt f L 0) (hmin : ∃ ε > 0, ∀ t ∈ Set.Ioo (0 : ℝ) ε, f 0 ≤ f t) :
    0 ≤ L := by
  rcases hmin with ⟨ε, hε, hle⟩
  rw [hasDerivAt_iff_tendsto_slope_zero] at hf
  have h_ev : ∀ᶠ t in nhdsWithin (0 : ℝ) (Set.Ioi 0), 0 ≤ t⁻¹ • (f t - f 0) := by
    filter_upwards [Ioo_mem_nhdsGT hε] with t ht
    rw [smul_eq_mul]
    exact mul_nonneg (inv_nonneg.mpr (le_of_lt ht.1)) (sub_nonneg.mpr (hle t ht))
  have h_tendsto : Filter.Tendsto (fun t : ℝ => t⁻¹ • (f t - f 0))
      (nhdsWithin 0 (Set.Ioi 0)) (nhds L) := by
    simpa [slope_def_module, sub_zero] using
      hf.mono_left (nhdsWithin_mono 0 (fun _ ht => ne_of_gt ht))
  exact ge_of_tendsto h_tendsto h_ev

/-- Condiciones KKT exactas: equivalencia con ser minimizador global en `FeasibleSet M`. -/
theorem kkt_iff_isMinOn {n p : ℕ} {δ : ℝ} (hδ : 0 < δ)
    (Z : Fin n → Fin p → ℝ) (y : Fin n → ℝ)
    {α : Fin p → ℝ} (hα : ∀ j, 0 ≤ α j) (M : Finset (Fin p))
    {θ : Params p} (hθ : θ ∈ FeasibleSet M) :
    IsMinOn (objective δ Z y α) (FeasibleSet M) θ ↔
      (gradB δ Z y θ = 0 ∧
        (∀ j ∉ M, gradW δ Z y α θ j = 0) ∧
        (∀ j ∈ M, 0 ≤ gradW δ Z y α θ j ∧ θ.2 j * gradW δ Z y α θ j = 0)) := by
  constructor
  · intro hmin
    -- Helper: si θ + t • d ∈ FeasibleSet M para todo t ∈ (0, ε), entonces ⟨∇J(θ), d⟩ ≥ 0
    have h_dir_nonneg : ∀ d : Params p, (∃ ε > 0, ∀ t ∈ Set.Ioo (0 : ℝ) ε, θ + t • d ∈ FeasibleSet M) →
        0 ≤ gradB δ Z y θ * d.1 + ∑ j : Fin p, gradW δ Z y α θ j * d.2 j := by
      intro d ⟨ε, hε, hd⟩
      exact deriv_nonneg_of_local_min_right (hasDerivAt_objective_line hδ Z y α θ d)
        ⟨ε, hε, fun t ht => by
          simpa using hmin (hd t ht)⟩
    -- 1. gradB = 0 probando d = (±1, 0)
    have hb0 : gradB δ Z y θ = 0 := by
      have hpos := h_dir_nonneg (1, fun _ => 0) ⟨1, one_pos, fun t _ j hj => by
        simpa using hθ j hj⟩
      have hneg := h_dir_nonneg (-1, fun _ => 0) ⟨1, one_pos, fun t _ j hj => by
        simpa using hθ j hj⟩
      simp only [mul_one, mul_neg, mul_zero, Finset.sum_const_zero, add_zero] at hpos hneg
      linarith
    -- 2. Para j ∉ M: gradW j = 0
    have h_free : ∀ j ∉ M, gradW δ Z y α θ j = 0 := by
      intro j hj
      let ej : Fin p → ℝ := fun l => if l = j then 1 else 0
      have hpos := h_dir_nonneg (0, ej) ⟨1, one_pos, fun t _ l hl => by
        have hlj : l ≠ j := fun h => hj (h ▸ hl)
        simpa [ej, hlj] using hθ l hl⟩
      have hneg := h_dir_nonneg (0, -ej) ⟨1, one_pos, fun t _ l hl => by
        have hlj : l ≠ j := fun h => hj (h ▸ hl)
        simpa [ej, hlj] using hθ l hl⟩
      simp only [mul_zero, zero_add, ej, mul_ite, mul_one, mul_neg, Pi.neg_apply,
        Finset.sum_neg_distrib, Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte] at hpos hneg
      linarith
    -- 3. Para j ∈ M: gradW j ≥ 0 y complementariedad θ.2 j * gradW j = 0
    have h_rest : ∀ j ∈ M, 0 ≤ gradW δ Z y α θ j ∧ θ.2 j * gradW δ Z y α θ j = 0 := by
      intro j hj
      let ej : Fin p → ℝ := fun l => if l = j then 1 else 0
      have hpos := h_dir_nonneg (0, ej) ⟨1, one_pos, fun t ht l hl => by
        by_cases hlj : l = j
        · subst hlj
          simp only [Prod.snd_add, Prod.smul_snd, Pi.add_apply, Pi.smul_apply, smul_eq_mul, ej,
            ↓reduceIte]
          linarith [hθ l hl, ht.1]
        · simpa [ej, hlj] using hθ l hl⟩
      simp only [mul_zero, zero_add, ej, mul_ite, mul_one,
        Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte] at hpos
      refine ⟨hpos, ?_⟩
      rcases eq_or_lt_of_le (hθ j hj) with hw0 | hw_pos
      · rw [← hw0, zero_mul]
      · have hneg := h_dir_nonneg (0, -ej) ⟨θ.2 j, hw_pos, fun t ht l hl => by
          by_cases hlj : l = j
          · subst hlj
            simp only [Prod.snd_add, Prod.smul_snd, Pi.add_apply, Pi.smul_apply, Pi.neg_apply,
              smul_eq_mul, ej, ↓reduceIte]
            linarith [ht.2]
          · simpa [ej, hlj] using hθ l hl⟩
        simp only [mul_zero, zero_add, ej, mul_ite, mul_neg, mul_one, Pi.neg_apply,
          Finset.sum_neg_distrib, Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte] at hneg
        have hg0 : gradW δ Z y α θ j = 0 := by linarith
        rw [hg0, mul_zero]
    exact ⟨hb0, h_free, h_rest⟩
  · rintro ⟨hb0, h_free, h_rest⟩ θ' hθ'
    have h_sub := objective_subgradient hδ Z y hα θ θ'
    have h_term_b : gradB δ Z y θ * (θ'.1 - θ.1) = 0 := by rw [hb0, zero_mul]
    have h_term_w : 0 ≤ ∑ j : Fin p, gradW δ Z y α θ j * (θ'.2 j - θ.2 j) := by
      refine Finset.sum_nonneg fun j _ => ?_
      by_cases hj : j ∈ M
      · obtain ⟨hg_nonneg, h_comp⟩ := h_rest j hj
        have h1 : gradW δ Z y α θ j * (θ'.2 j - θ.2 j) =
            gradW δ Z y α θ j * θ'.2 j - θ.2 j * gradW δ Z y α θ j := by ring
        rw [h1, h_comp, sub_zero]
        exact mul_nonneg hg_nonneg (hθ' j hj)
      · rw [h_free j hj, zero_mul]
    exact le_trans (le_add_of_nonneg_right (by rw [h_term_b, zero_add]; exact h_term_w)) h_sub

end SMCFHR

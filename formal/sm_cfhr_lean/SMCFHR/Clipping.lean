import Mathlib.Data.Real.Basic
import Mathlib.Tactic

/-!
# Clipping físico en factor de capacidad

Formaliza el operador `clip01(u) = max 0 (min u 1)` utilizado en SM-CFHR para proyectar
factores de capacidad y predicciones al intervalo físico `[0, 1]`.
-/

namespace SMCFHR

/-- Operador de truncamiento al intervalo `[0, 1]`. -/
def clip01 (u : ℝ) : ℝ := max 0 (min u 1)

theorem clip01_nonneg (u : ℝ) : 0 ≤ clip01 u :=
  le_max_left 0 (min u 1)

theorem clip01_le_one (u : ℝ) : clip01 u ≤ 1 := by
  unfold clip01
  exact max_le (by norm_num) (min_le_right u 1)

theorem clip01_mem_Icc (u : ℝ) : clip01 u ∈ Set.Icc (0 : ℝ) 1 :=
  ⟨clip01_nonneg u, clip01_le_one u⟩

theorem clip01_eq_self {u : ℝ} (h0 : 0 ≤ u) (h1 : u ≤ 1) : clip01 u = u := by
  unfold clip01
  rw [min_eq_left h1, max_eq_right h0]

theorem clip01_mono {u v : ℝ} (h : u ≤ v) : clip01 u ≤ clip01 v := by
  unfold clip01
  exact max_le_max_left 0 (min_le_min_right 1 h)

theorem clip01_eq_min_max (u : ℝ) : clip01 u = min (max u 0) 1 := by
  unfold clip01
  rcases le_total u 0 with hu0 | hu0 <;> rcases le_total u 1 with hu1 | hu1
  · rw [min_eq_left hu1, max_eq_left hu0, max_eq_right hu0, min_eq_left (by norm_num)]
  · linarith
  · rw [min_eq_left hu1, max_eq_right hu0, max_eq_left hu0, min_eq_left hu1]
  · rw [min_eq_right hu1, max_eq_right (by norm_num), max_eq_left (by linarith),
        min_eq_right hu1]

end SMCFHR

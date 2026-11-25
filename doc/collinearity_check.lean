/-
Machine check of the linear algebra behind the collinearity maps of `doc/sweep_writeup.pdf`, section 5.

Setting. `a` and `b` are the sensitivity vectors of `m` statistics to two parameters (one entry per
statistic). A change `(x, y)` of the two parameters moves the statistics, to first order, by
`x • a + y • b`. The statistics cannot separate the two parameters locally when some non-zero
`(x, y)` gives `x • a + y • b = 0`.

Compiled against Lean 4.29 / Mathlib v4.29.0. This file is not part of any Lean project in this
repository; it needs a Mathlib project to compile. These are textbook facts, checked because the
write-up leans on them, not claimed as results.
-/
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Tactic

open RealInnerProductSpace

namespace CollinearityCheck

/-- One statistic. Whatever the two slopes `a` and `b` are, some non-zero change of the two
parameters leaves the statistic unchanged to first order. -/
theorem one_statistic (a b : ℝ) : ∃ x y : ℝ, (x, y) ≠ (0, 0) ∧ a * x + b * y = 0 := by
  by_cases hb : b = 0
  · exact ⟨0, 1, by simp, by simp [hb]⟩
  · exact ⟨b, -a, by simp [hb], by ring⟩

/-- Two statistics. Such a change exists exactly when `a₁ b₂ − a₂ b₁ = 0`. -/
theorem two_statistics (a₁ a₂ b₁ b₂ : ℝ) :
    (∃ x y : ℝ, (x, y) ≠ (0, 0) ∧ a₁ * x + b₁ * y = 0 ∧ a₂ * x + b₂ * y = 0) ↔
      a₁ * b₂ - a₂ * b₁ = 0 := by
  constructor
  · rintro ⟨x, y, hne, h1, h2⟩
    have hx : x * (a₁ * b₂ - a₂ * b₁) = 0 := by linear_combination b₂ * h1 - b₁ * h2
    have hy : y * (a₁ * b₂ - a₂ * b₁) = 0 := by linear_combination a₁ * h2 - a₂ * h1
    by_contra hd
    have hx0 : x = 0 := (mul_eq_zero.mp hx).resolve_right hd
    have hy0 : y = 0 := (mul_eq_zero.mp hy).resolve_right hd
    exact hne (by simp [hx0, hy0])
  · intro hd
    by_cases h1 : (b₁, -a₁) = ((0 : ℝ), (0 : ℝ))
    · by_cases h2 : (b₂, -a₂) = ((0 : ℝ), (0 : ℝ))
      · simp only [Prod.mk.injEq, neg_eq_zero] at h1 h2
        exact ⟨1, 0, by simp, by simp [h1.2], by simp [h2.2]⟩
      · exact ⟨b₂, -a₂, h2, by linear_combination hd, by ring⟩
    · exact ⟨b₁, -a₁, h1, by ring, by linear_combination (-1 : ℝ) * hd⟩

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Any number of statistics, with `a ≠ 0`. Such a change exists exactly when `b` is a multiple
of `a`. -/
theorem unseen_iff_multiple (a b : E) (ha : a ≠ 0) :
    (∃ x y : ℝ, (x, y) ≠ (0, 0) ∧ x • a + y • b = 0) ↔ ∃ l : ℝ, b = l • a := by
  constructor
  · rintro ⟨x, y, hne, h⟩
    have hy : y ≠ 0 := by
      intro hy0
      rw [hy0, zero_smul, add_zero] at h
      rcases smul_eq_zero.mp h with hx | ha'
      · exact hne (by simp [hx, hy0])
      · exact ha ha'
    have hyb : y • b = -(x • a) := eq_neg_of_add_eq_zero_right h
    refine ⟨-(x / y), ?_⟩
    calc b = y⁻¹ • (y • b) := by rw [smul_smul, inv_mul_cancel₀ hy, one_smul]
      _ = y⁻¹ • (-(x • a)) := by rw [hyb]
      _ = (-(x / y)) • a := by rw [smul_neg, smul_smul, ← neg_smul, div_eq_inv_mul]
  · rintro ⟨l, rfl⟩
    exact ⟨l, -1, by simp, by simp⟩

/-- With both vectors non-zero: `b` is a multiple of `a` exactly when the cosine of the angle
between them is `+1` or `−1`. -/
theorem abs_cos_eq_one_iff (a b : E) (ha : a ≠ 0) (hb : b ≠ 0) :
    |⟪a, b⟫ / (‖a‖ * ‖b‖)| = 1 ↔ ∃ l : ℝ, b = l • a := by
  rw [abs_real_inner_div_norm_mul_norm_eq_one_iff]
  constructor
  · rintro ⟨_, r, _, h⟩
    exact ⟨r, h⟩
  · rintro ⟨l, h⟩
    refine ⟨ha, l, ?_, h⟩
    rintro rfl
    exact hb (by simpa using h)

/-- The part of `b` that `a` cannot reproduce. With `l = ⟪a, b⟫ / ‖a‖²`,
`‖b − l a‖² = ‖b‖² − ⟪a, b⟫² / ‖a‖²`. -/
theorem residual_sq (a b : E) (ha : a ≠ 0) :
    ‖b - (⟪a, b⟫ / ‖a‖ ^ 2) • a‖ ^ 2 = ‖b‖ ^ 2 - ⟪a, b⟫ ^ 2 / ‖a‖ ^ 2 := by
  have hn : ‖a‖ ≠ 0 := norm_ne_zero_iff.mpr ha
  rw [norm_sub_sq_real, inner_smul_right, norm_smul, mul_pow, Real.norm_eq_abs, sq_abs,
    real_inner_comm a b]
  field_simp
  ring

/-- The same in terms of the cosine: `‖b − l a‖² = ‖b‖² (1 − cos²)`. -/
theorem residual_sq_cos (a b : E) (ha : a ≠ 0) (hb : b ≠ 0) :
    ‖b - (⟪a, b⟫ / ‖a‖ ^ 2) • a‖ ^ 2 = ‖b‖ ^ 2 * (1 - (⟪a, b⟫ / (‖a‖ * ‖b‖)) ^ 2) := by
  have hna : ‖a‖ ≠ 0 := norm_ne_zero_iff.mpr ha
  have hnb : ‖b‖ ≠ 0 := norm_ne_zero_iff.mpr hb
  rw [residual_sq a b ha]
  field_simp

end CollinearityCheck

#check @CollinearityCheck.one_statistic
#check @CollinearityCheck.two_statistics
#check @CollinearityCheck.unseen_iff_multiple
#check @CollinearityCheck.abs_cos_eq_one_iff
#check @CollinearityCheck.residual_sq
#check @CollinearityCheck.residual_sq_cos
#print axioms CollinearityCheck.one_statistic
#print axioms CollinearityCheck.two_statistics
#print axioms CollinearityCheck.unseen_iff_multiple
#print axioms CollinearityCheck.abs_cos_eq_one_iff
#print axioms CollinearityCheck.residual_sq
#print axioms CollinearityCheck.residual_sq_cos

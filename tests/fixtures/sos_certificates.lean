import Mathlib
set_option linter.style.longLine true

theorem two_squares (x y : ℝ) : 0 ≤ x ^ 2 + y ^ 2 - 2 * x - 2 * y + 2 := by
  have h : x ^ 2 + y ^ 2 - 2 * x - 2 * y + 2 = (2 : ℝ) * (-(x / 2) - y / 2 + 1) ^ 2 + (1 / 2 : ℝ) *
    (-x + y) ^ 2 := by ring
  rw [h]
  positivity

theorem pd_quadratic (x y : ℝ) : 0 ≤ x ^ 2 - x * y + y ^ 2 + 1 := by
  have h : x ^ 2 - x * y + y ^ 2 + 1 = (1 : ℝ) ^ 2 + (-(x / 2) + y) ^ 2 + (3 / 4 : ℝ) * x ^ 2 := by
    ring
  rw [h]
  positivity

theorem square_of_quadratic (x y : ℝ) : 0 ≤ x ^ 4 - 2 * x ^ 2 * y ^ 2 + y ^ 4 := by
  have h : x ^ 4 - 2 * x ^ 2 * y ^ 2 + y ^ 4 = (-x ^ 2 + y ^ 2) ^ 2 := by ring
  rw [h]
  positivity

theorem product_of_squares (x y : ℝ) : 0 ≤ x ^ 4 + x ^ 2 * y ^ 2 - 2 * y * x ^ 3 + 2 * x * y ^ 2 + 2
    * x ^ 3 - 4 * y * x ^ 2 + x ^ 2 - 2 * x * y + y ^ 2 := by
  have h : x ^ 4 + x ^ 2 * y ^ 2 - 2 * y * x ^ 3 + 2 * x * y ^ 2 + 2 * x ^ 3 - 4 * y * x ^ 2 + x ^ 2
    - 2 * x * y + y ^ 2 = (-x ^ 2 + x * y - x + y) ^ 2 := by ring
  rw [h]
  positivity

theorem univariate (x : ℝ) : 0 ≤ x ^ 4 - 2 * x ^ 3 + 2 * x ^ 2 - 2 * x + 1 := by
  have h : x ^ 4 - 2 * x ^ 3 + 2 * x ^ 2 - 2 * x + 1 = (-x + 1) ^ 2 + (-x ^ 2 + x) ^ 2 := by ring
  rw [h]
  positivity

theorem quartic_pd (x y : ℝ) : 0 ≤ x ^ 4 + y ^ 4 + x * y / 2 + 1 := by
  have h : x ^ 4 + y ^ 4 + x * y / 2 + 1 = (-(2 * x ^ 2 / 5) + x * y / 8 - 2 * y ^ 2 / 5 + 1) ^ 2 +
    (4 / 5 : ℝ) * (5 * x / 32 + y) ^ 2 + (999 / 1280 : ℝ) * x ^ 2 + (21 / 25 : ℝ) *
    (-(2 * x ^ 2 / 3) + 5 * x * y / 84 + y ^ 2) ^ 2 + (5251 / 6720 : ℝ) *
    (560 * x ^ 2 / 5251 + x * y) ^ 2 + (12019 / 26255 : ℝ) * (x ^ 2) ^ 2 := by ring
  rw [h]
  positivity

theorem three_vars (x y z : ℝ) : 0 ≤ x ^ 2 - x * y - x * z + y ^ 2 - y * z + z ^ 2 := by
  have h : x ^ 2 - x * y - x * z + y ^ 2 - y * z + z ^ 2 = (-(x / 2) - y / 2 + z) ^ 2 + (3 / 4 : ℝ)
    * (-x + y) ^ 2 := by ring
  rw [h]
  positivity

theorem amgm3 (x y z : ℝ) : 0 ≤ x ^ 4 + y ^ 4 + z ^ 4 - 4 * x * y * z + 1 := by
  have h : x ^ 4 + y ^ 4 + z ^ 4 - 4 * x * y * z + 1 =
    (-(x ^ 2 / 3) - y ^ 2 / 3 - z ^ 2 / 3 + 1) ^ 2 + (2 / 3 : ℝ) * (-(x * y) + z) ^ 2 + (2 / 3 : ℝ)
    * (-(x * z) + y) ^ 2 + (2 / 3 : ℝ) * (-(y * z) + x) ^ 2 + (8 / 9 : ℝ) *
    (-(x ^ 2 / 2) - y ^ 2 / 2 + z ^ 2) ^ 2 + (2 / 3 : ℝ) * (-x ^ 2 + y ^ 2) ^ 2 := by ring
  rw [h]
  positivity


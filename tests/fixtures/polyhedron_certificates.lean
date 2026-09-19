import Mathlib
set_option linter.style.longLine true

theorem cell_goal (r t j : ℝ) (_hj : (2 : ℝ) ≤ j) (_h0 : 0 ≤ r) (h1 : 0 ≤ -r + (1 / 2 : ℝ))
    (_h2 : 0 ≤ t) (_h3 : 0 ≤ -t + 1) (h4 : 0 ≤ -(j * r) + 2 * j * t + t - 1) :
    0 ≤ -(4 * j * r) + 8 * j * t - r + 4 * t - 3 := by
  linarith only [h1, h4]

theorem needs_lambda (r t j : ℝ) (hj : (0 : ℝ) ≤ j) (h0 : 0 ≤ -r + t) (h1 : 0 ≤ j * r - j + t - 1) :
    0 ≤ t - 1 := by
  have hJ0 : (0 : ℝ) ≤ j := by linarith
  have h0J := mul_nonneg hJ0 h0
  have hg : (0 : ℝ) ≤ (j + 1) * (t - 1) := by
    linarith only [h0J, h1]
  have hg' := nonneg_of_mul_nonneg_right hg (by linarith only [hJ0])
  linarith only [hg']

theorem needs_lambda_sq (r t j : ℝ) (hj : (0 : ℝ) ≤ j) (h0 : 0 ≤ -r + t)
    (h1 : 0 ≤ r * j ^ 2 - j ^ 2 + t - 1) :
    0 ≤ t - 1 := by
  have hJ0 : (0 : ℝ) ≤ j := by linarith
  have h0J := mul_nonneg hJ0 h0
  have h0JJ := mul_nonneg hJ0 h0J
  have pJJ := mul_nonneg hJ0 hJ0
  have hg : (0 : ℝ) ≤ (j ^ 2 + 1) * (t - 1) := by
    linarith only [h0JJ, h1]
  have hg' := nonneg_of_mul_nonneg_right hg (by linarith only [hJ0, pJJ])
  linarith only [hg']

theorem k_chain (r t j : ℝ) (hj : (2 : ℝ) ≤ j) (h0 : 0 ≤ -r + t) (h1 : 0 ≤ r) :
    0 ≤ j * t - 2 * t := by
  have hK0 : (0 : ℝ) ≤ j - 2 := by linarith
  have h0K := mul_nonneg hK0 h0
  have h1K := mul_nonneg hK0 h1
  linarith only [h0K, h1K]

theorem neg_j0 (r t j : ℝ) (hj : (-1 : ℝ) ≤ j) (h0 : 0 ≤ -r + t) (h1 : 0 ≤ r) :
    0 ≤ j * t + t := by
  have hK0 : (0 : ℝ) ≤ j + 1 := by linarith
  have h0K := mul_nonneg hK0 h0
  have h1K := mul_nonneg hK0 h1
  linarith only [h0K, h1K]

theorem jk_chain (r t j : ℝ) (hj : (2 : ℝ) ≤ j) (h0 : 0 ≤ -r + t) (h1 : 0 ≤ r) :
    0 ≤ t * j ^ 2 - 2 * j * t := by
  have hJ0 : (0 : ℝ) ≤ j := by linarith
  have hK0 : (0 : ℝ) ≤ j - 2 := by linarith
  have h0K := mul_nonneg hK0 h0
  have h0JK := mul_nonneg hJ0 h0K
  have h1K := mul_nonneg hK0 h1
  have h1JK := mul_nonneg hJ0 h1K
  linarith only [h0JK, h1JK]

theorem pairwise (r : ℝ) (h0 : 0 ≤ r) (h1 : 0 ≤ -r + 1) :
    0 ≤ -r ^ 2 + r := by
  have h0xh1 := mul_nonneg h0 h1
  linarith only [h0xh1]

theorem pairwise_j (r j : ℝ) (hj : (1 : ℝ) ≤ j) (h0 : 0 ≤ r) (h1 : 0 ≤ -r + 1) :
    0 ≤ -(j * r ^ 2) + j * r := by
  have hJ0 : (0 : ℝ) ≤ j := by linarith
  have h0xh1 := mul_nonneg h0 h1
  have h0xh1J := mul_nonneg hJ0 h0xh1
  linarith only [h0xh1J]

theorem cell_empty (r t j : ℝ) (hj : (2 : ℝ) ≤ j) (h0 : 0 ≤ r - (1 / 2 : ℝ))
    (h1 : 0 ≤ -(j * r) + 2 * j * t + t - 1) (h2 : 0 ≤ -t + (1 / 4 : ℝ)) :
    False := by
  have hJ0 : (0 : ℝ) ≤ j := by linarith
  have h0J := mul_nonneg hJ0 h0
  have h2J := mul_nonneg hJ0 h2
  linarith only [h0J, h1, h2, h2J]

theorem cell_empty_lambda (r t j : ℝ) (hj : (0 : ℝ) ≤ j) (h0 : 0 ≤ -r + t)
    (h1 : 0 ≤ j * r - j + t - 1) (h2 : 0 ≤ -t + (1 / 2 : ℝ)) :
    False := by
  have hJ0 : (0 : ℝ) ≤ j := by linarith
  have h0J := mul_nonneg hJ0 h0
  have h2J := mul_nonneg hJ0 h2
  linarith only [h0J, h1, h2, h2J, hJ0]

theorem pure_power (t j : ℝ) (hj : (0 : ℝ) ≤ j) (h0 : 0 ≤ t) :
    0 ≤ j ^ 2 + t := by
  have hJ0 : (0 : ℝ) ≤ j := by linarith
  have pJJ := mul_nonneg hJ0 hJ0
  linarith only [h0, pJJ]

theorem farkas (r t : ℝ) (h0 : 0 ≤ r) (_h1 : 0 ≤ -r + 1) (h2 : 0 ≤ -r + t) :
    0 ≤ -r + 2 * t := by
  linarith only [h0, h2]

theorem pure_jk (t j : ℝ) (hj : (2 : ℝ) ≤ j) (h0 : 0 ≤ t) :
    0 ≤ j ^ 2 - 2 * j + t := by
  have hJ0 : (0 : ℝ) ≤ j := by linarith
  have hK0 : (0 : ℝ) ≤ j - 2 := by linarith
  have pJK := mul_nonneg hJ0 hK0
  linarith only [h0, pJK]

theorem pure_kk (t j : ℝ) (hj : (2 : ℝ) ≤ j) (h0 : 0 ≤ t) :
    0 ≤ j ^ 2 - 4 * j + t + 4 := by
  have hK0 : (0 : ℝ) ≤ j - 2 := by linarith
  have pKK := mul_nonneg hK0 hK0
  linarith only [h0, pKK]

theorem neg_lambda (r t j : ℝ) (hj : (-1 : ℝ) ≤ j) (h0 : 0 ≤ -r + t)
    (h1 : 0 ≤ j * r - j + r + t - 2) :
    0 ≤ t - 1 := by
  have hK0 : (0 : ℝ) ≤ j + 1 := by linarith
  have h0K := mul_nonneg hK0 h0
  have hg : (0 : ℝ) ≤ (j + 2) * (t - 1) := by
    linarith only [h0K, h1]
  have hg' := nonneg_of_mul_nonneg_right hg (by linarith only [hK0])
  linarith only [hg']

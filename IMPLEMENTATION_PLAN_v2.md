# Symplex v0.2 Implementation Plan

**Generated from:** Full mathematical correctness audit of the v0.1.0 codebase
**Scope:** Bug fixes, safety hardening, test coverage, and targeted feature gaps
**Philosophy:** Fix what's broken, warn about what's limited, test what's untested. No hacking tests to pass — make the math right.

---

## Table of Contents

1. [Tier 0 — Genuine Bugs (Ship-Blockers)](#tier-0--genuine-bugs-ship-blockers)
2. [Tier 1 — Safety Hardening (Silent Failures → Loud Failures)](#tier-1--safety-hardening-silent-failures--loud-failures)
3. [Tier 2 — Test Coverage Gaps (Under-Tested Modules)](#tier-2--test-coverage-gaps-under-tested-modules)
4. [Tier 3 — Feature Gaps (Missing Capabilities)](#tier-3--feature-gaps-missing-capabilities)
5. [Tier 4 — Quality-of-Life & Polish](#tier-4--quality-of-life--polish)
6. [Appendix A — Test Matrix](#appendix-a--test-matrix)
7. [Appendix B — Files Touched Per Tier](#appendix-b--files-touched-per-tier)

---

## Tier 0 — Genuine Bugs (Ship-Blockers)

These are mathematically incorrect results. They must be fixed before any release.

### T0-1: `limit_at_infinity` ignores leading coefficient sign

**File:** `src/calculus/limit.rs` lines 258–264 (Strategy 0)
**Bug:** When numerator degree > denominator degree, the code returns `+∞` or `-∞` based solely on the direction of approach, without checking the sign of the leading coefficient ratio. Example: `lim(x→+∞) -x³/(x+1)` should be `-∞` but currently returns `+∞`.

**Fix:**
```
fn limit_at_infinity(arena, expr, var, positive) {
    // ... existing degree extraction ...
    if nd > dd {
        // NEW: compute sign of leading coefficient ratio
        let lc_numer = extract_leading_coeff(arena, numer_poly, var);
        let lc_denom = extract_leading_coeff(arena, denom_poly, var);
        let ratio_positive = sign(lc_numer) * sign(lc_denom) > 0;

        // For x → -∞, odd degree difference flips the sign
        let degree_diff = nd - dd;
        let flip = !positive && (degree_diff % 2 == 1);
        let result_positive = ratio_positive ^ flip;

        if result_positive {
            return Ok(arena.infinity());
        } else {
            return Ok(arena.neg_infinity());
        }
    }
}
```

**Implementation details:**
- Add helper `extract_leading_coeff(arena, expr, var) -> Option<Ratio<BigInt>>` that extracts the coefficient of the highest-degree term in `var`. This can reuse `as_coeff_term` after collecting powers of `var`.
- If `extract_leading_coeff` returns `None` (expression too complex), fall through to Strategy 1 as before — do not guess.
- The same fix must be applied to **Strategy 2** (lines 318–348), which has the same bug in the substituted form.

**Tests to add** (in `tests/test_limits_infinity.rs`):
| Test name | Expression | Expected | What it catches |
|-----------|-----------|----------|-----------------|
| `limit_neg_leading_coeff_at_pos_inf` | `lim(x→+∞) -x³/(x+1)` | `-∞` | Sign of leading coeff |
| `limit_neg_leading_coeff_at_neg_inf` | `lim(x→-∞) -x³/(x+1)` | `+∞` | Sign + odd degree flip |
| `limit_neg_over_neg_at_pos_inf` | `lim(x→+∞) -x²/(-x+1)` | `+∞` | Double negative → positive |
| `limit_even_degree_diff_neg_inf` | `lim(x→-∞) x⁴/(x²+1)` | `+∞` | Even degree: no flip at -∞ |
| `limit_odd_degree_diff_neg_inf` | `lim(x→-∞) x³/(x²+1)` | `-∞` | Odd degree: flip at -∞ |
| `limit_rational_coeffs` | `lim(x→+∞) (3x²+1)/(-2x+5)` | `-∞` | Rational leading coefficients |

---

### T0-2: Heaviside integral incorrect for negative linear coefficient

**File:** `src/transforms/integrate.rs` lines 1884–1895
**Bug:** `∫H(ax+b)dx = (ax+b)·H(ax+b)/a` is only correct for `a > 0`. For `a < 0`, the Heaviside step function goes from 1 to 0 (instead of 0 to 1), and dividing by a negative `a` flips the sign incorrectly. The correct general formula is `∫H(ax+b)dx = (ax+b)·H(ax+b)/a` which is actually algebraically correct for both signs — BUT the issue is that `a` could be a symbolic expression with unknown sign, in which case we should use `|a|` or guard the transformation.

**Detailed analysis:** On re-examination, `(ax+b)·H(ax+b)/a` IS algebraically correct for any nonzero `a`:
- For `a > 0`: H turns on at `x = -b/a`, antiderivative is `(ax+b)/a` for `x > -b/a`, zero otherwise. Derivative = `a/a · H(ax+b) = H(ax+b)` ✓
- For `a < 0`: H is 1 for `x < -b/a`, zero otherwise. `(ax+b)/a` is `x + b/a`, derivative is 1 in that region. The product `(ax+b)·H(ax+b)/a` gives `(ax+b)/a` where `ax+b > 0`, which for negative `a` means `x < -b/a`. The derivative of this piecewise function is indeed `H(ax+b)` ✓

**Revised verdict:** The formula is actually correct. However, compare with the DiracDelta case (line 1874) which uses `|a|`: `∫δ(ax+b)dx = H(ax+b)/|a|`. The Heaviside formula uses bare `a` (not `|a|`), which is correct but might look inconsistent. The real concern is when `a_expr` is **symbolic** — we can't determine its sign. In that case, dividing by `a_expr` is the right thing algebraically.

**Action:** Downgrade from bug to **documentation + test coverage**. Add tests that verify correctness for negative `a`:

**Tests to add** (in `tests/test_integrate.rs` or a new `tests/test_heaviside_integ.rs`):
| Test name | Integrand | Expected behavior |
|-----------|-----------|-------------------|
| `heaviside_negative_coeff` | `∫H(-2x+3)dx` | Verify FTC numerically at x=0,1,2,3 |
| `heaviside_symbolic_coeff` | `∫H(a·x)dx` with `a` symbolic | Returns `(a·x)·H(a·x)/a` — verify structure |
| `dirac_negative_coeff` | `∫δ(-3x+6)dx` | `H(-3x+6)/3` — verify FTC |
| `dirac_sifting_negative` | `∫x²·δ(-x+2)dx` | Sifting gives `4·H(-x+2)` — verify |

---

### T0-3: Piecewise `evalf` last-branch fallback is unsafe

**File:** `src/transforms/evalf.rs` lines 874–878
**Bug:** When no Piecewise condition can be evaluated to true or false, the code unconditionally evaluates the last branch's value. If the last branch is NOT an else clause (i.e., its condition is not `BoolTrue`), this silently returns a wrong answer.

**Fix:**
```rust
// BEFORE (unsafe):
if let Some(&(value_id, _)) = pairs.last() {
    debug!("evalf: Piecewise — falling back to last branch");
    return eval_node_or_subtree(arena, value_id, cache, prec, rm, cc);
}

// AFTER (safe):
if let Some(&(value_id, cond_id)) = pairs.last() {
    let cond_node = arena.node(cond_id);
    if matches!(cond_node, ExprNode::BoolTrue) {
        debug!("evalf: Piecewise — using explicit else branch");
        return eval_node_or_subtree(arena, value_id, cache, prec, rm, cc);
    }
    warn!("evalf: Piecewise — no condition resolved and last branch is not an else; refusing to guess");
}
Err(SymplexError::Unevaluable {
    reason: "cannot evaluate piecewise: no condition is definitively true and no else branch exists".into(),
})
```

**Tests to add** (in `tests/test_evalf_gaps.rs`):
| Test name | Setup | Expected |
|-----------|-------|----------|
| `piecewise_else_branch_used` | `Piecewise((1, x>0), (2, True))` at x=−1 | `Ok(2.0)` |
| `piecewise_no_else_returns_error` | `Piecewise((1, x>0), (2, x<-10))` at x=−1 where x>0 can't be resolved symbolically | `Err(Unevaluable)` |
| `piecewise_all_false_returns_error` | `Piecewise((1, False), (2, False))` | `Err(Unevaluable)` |
| `piecewise_first_true_wins` | `Piecewise((1, True), (2, True))` | `Ok(1.0)` |

---

## Tier 1 — Safety Hardening (Silent Failures → Loud Failures)

These are not bugs per se — the math that IS returned is correct — but the system silently drops information without telling anyone. Users get incomplete results with no indication that anything went wrong.

### T1-1: Add `warn!` logging at all silent bailout points in polynomial factoring

**Files and locations:**

| File | Line | Current behavior | Fix |
|------|------|-----------------|-----|
| `src/poly/dense.rs` ~L838 | Kronecker degree cap `min(6)` | Silent | Add `debug!("factoring: Kronecker method capped at degree {}; irreducible factors of degree >{} will be missed", max_trial, max_trial)` |
| `src/poly/dense.rs` ~L898 | Divisor product cap `>500` | Silent | Add `warn!("factoring: rational root search abandoned — divisor product {} exceeds cap 500", count)` |
| `src/poly/dense.rs` ~L993 | Kronecker combination cap `>100_000` | Silent | Add `warn!("factoring: Kronecker search abandoned — {} combinations exceeds cap", total)` |
| `src/poly/dense.rs` ~L1119 | Divisor magnitude cap `>10⁹` | Silent | Add `debug!("factoring: coefficient {} exceeds divisor magnitude cap 10^9", n_abs)` |
| `src/poly/polysys.rs` ~L256 | Separate divisor cap `>10_000` | Silent | Add `warn!` and unify with the main cap |

**Tests:** No new test functions needed — these are logging improvements. But add one integration test that triggers a cap and verifies the computation still returns a valid (if incomplete) result without panicking.

---

### T1-2: Add `warn!` when eigenvalue computation is incomplete

**File:** `src/domains/matrix.rs`

**Current behavior:** `eigenvals()` returns `Ok(vec![])` when the characteristic polynomial can't be solved — indistinguishable from "has no eigenvalues" (impossible for a nonzero square matrix).

**Fix:** After `solve_or_empty()`, check if the number of returned eigenvalues (counting multiplicity) equals the matrix dimension. If not, log a warning:

```rust
pub fn eigenvals(&self, var: &Ex) -> Result<Vec<Ex>, SymplexError> {
    let cp = self.char_poly(var)?;
    let roots = cp.solve_or_empty(var);
    let n = self.nrows();
    if roots.len() < n {
        tracing::warn!(
            "eigenvals: found {} eigenvalue(s) for a {}×{} matrix — \
             characteristic polynomial may have irreducible factors beyond solver capability",
            roots.len(), n, n
        );
    }
    Ok(roots)
}
```

Also add the same check in `eigvals_via_poly_factor` (line ~2030) when individual factors yield no roots:

```rust
if factor_roots.is_empty() && factor_degree > 0 {
    tracing::warn!(
        "eigenvals: irreducible factor of degree {} yielded no roots — \
         eigenvalues from this factor are missing",
        factor_degree
    );
}
```

**Tests to add** (in `tests/test_matrix_linalg.rs`):
| Test name | Matrix | Expected |
|-----------|--------|----------|
| `eigenvals_warns_on_incomplete` | A 5×5 with char poly having an irreducible quintic | Returns `Ok(vec![])`, verify with `tracing-test` that a warning was emitted |
| `eigenvals_irrational_via_quadratic` | `[[0,2],[1,0]]` (eigenvalues ±√2) | Returns the √2 eigenvalues (verifies the quadratic path works for irrational roots) |

---

### T1-3: Unify divisor caps between `dense.rs` and `polysys.rs`

**Problem:** `dense.rs` uses a cap of 10⁹ for coefficient magnitude and 500 for divisor-product count. `polysys.rs` uses a separate cap of 10,000 for coefficient magnitude. These should be unified.

**Fix:**
- Add constants to `src/poly/mod.rs`:
  ```rust
  /// Maximum coefficient magnitude for rational root divisor enumeration.
  pub(crate) const MAX_DIVISOR_COEFFICIENT: i64 = 1_000_000_000;
  /// Maximum number of divisor-product combinations to try.
  pub(crate) const MAX_DIVISOR_COMBINATIONS: usize = 500;
  /// Maximum Kronecker trial factor degree.
  pub(crate) const MAX_KRONECKER_DEGREE: usize = 6;
  /// Maximum Kronecker evaluation combinations.
  pub(crate) const MAX_KRONECKER_COMBINATIONS: usize = 100_000;
  ```
- Replace all magic numbers in `dense.rs` and `polysys.rs` with these constants.
- This makes the limits discoverable, documentable, and overridable in the future.

---

### T1-4: Digamma precision — upgrade from f64 to arbitrary precision

**File:** `src/transforms/evalf.rs` lines 236–246 (dispatch) and 1374–1426 (f64 impl)

**Problem:** `digamma_f64` gives only ~15 significant digits regardless of requested precision. Every other special function (Gamma, erf, LambertW, Bessel) has a proper arbitrary-precision implementation.

**Fix:** Write `arb_digamma(x: &BigFloat, prec: usize, rm: RoundingMode, cc: &mut Consts) -> BigFloat` using the same algorithm as the existing f64 version but with `BigFloat` arithmetic:

1. **Reflection formula** for negative arguments: `ψ(1-x) - ψ(x) = π·cot(πx)`
2. **Recurrence** to shift argument: while `x < 8`, compute `result -= 1/x; x += 1`
3. **Asymptotic series**: `ψ(x) ~ ln(x) - 1/(2x) - Σ B_{2k}/(2k·x^{2k})` with Bernoulli numbers computed from `src/base/bernoulli.rs`

The Bernoulli number infrastructure already exists (used by `stirling_log_gamma`). The number of series terms should be `prec / 3 + 5` (each term gains about 3 bits of precision from the `x^{-2k}` decay).

**Wire up:**
```rust
ExprNode::Digamma(inner) => {
    let val = get_cached(cache, *inner)?;
    if !val.1.is_zero() {
        return Err(SymplexError::Unevaluable {
            reason: "Digamma of complex argument not yet supported in evalf".into(),
        });
    }
    let result = arb_digamma(&val.0, prec, rm, cc)?;
    Ok((result, BigFloat::new(prec)))
}
```

Keep `digamma_f64` as an internal helper for `eval_f64()` fast-path.

**Tests to add** (in `tests/test_evalf_arb_prec.rs` or `tests/test_arb_prec_special.rs`):
| Test name | Input | Precision | Expected |
|-----------|-------|-----------|----------|
| `digamma_arb_prec_at_1` | `ψ(1)` | 50 digits | `-γ` (Euler-Mascheroni, verify first 40 digits) |
| `digamma_arb_prec_at_5` | `ψ(5)` | 50 digits | `1 + 1/2 + 1/3 + 1/4 - γ` |
| `digamma_arb_prec_at_half` | `ψ(1/2)` | 50 digits | `-γ - 2·ln(2)` |
| `digamma_arb_prec_negative` | `ψ(-1/2)` | 30 digits | Verify via reflection formula |
| `digamma_f64_matches_arb` | `ψ(3.7)` | 15 digits | f64 and arb agree to 12 digits |

---

## Tier 2 — Test Coverage Gaps (Under-Tested Modules)

These modules have working implementations but insufficient test coverage to build confidence.

### T2-1: ODE solver — test all 13 classes

**File:** `tests/test_ode_coverage.rs` (extend) and `tests/test_ode_advanced.rs` (extend)

**Current state:** Only 3 of 13 ODE classes have coverage in `test_ode_coverage.rs`. `test_ode_advanced.rs` covers more but is inconsistent.

**Plan:** Create a new comprehensive file `tests/test_ode_comprehensive.rs` with systematic coverage:

| ODE Class | Test function | Equation | Known solution |
|-----------|--------------|----------|----------------|
| `SimpleSeparable` | `ode_simple_sep_polynomial` | y' = x² | y = x³/3 + C₁ |
| `SimpleSeparable` | `ode_simple_sep_trig` | y' = cos(x) | y = sin(x) + C₁ |
| `FullSeparable` | `ode_full_sep_basic` | y' = x·y | y = C₁·exp(x²/2) |
| `FullSeparable` | `ode_full_sep_rational` | y' = y/x | y = C₁·x |
| `FirstOrderLinearCC` | `ode_focc_basic` | y' + 2y = exp(x) | y = exp(x)/3 + C₁·exp(-2x) |
| `FirstOrderLinearCC` | `ode_focc_zero_rhs` | y' + 3y = 0 | y = C₁·exp(-3x) |
| `FirstOrderLinearVC` | `ode_fovc_basic` | y' + y/x = x | Integrating factor solution |
| `FirstOrderLinearVC` | `ode_fovc_sinusoidal` | y' + (tan x)·y = cos(x) | y = sin(x)·cos(x) + C₁·cos(x) |
| `ExactFirstOrder` | `ode_exact_basic` | (2xy+3)+(x²+4y)y'=0 | x²y + 3x + 2y² = C₁ |
| `ExactFirstOrder` | `ode_exact_trig` | cos(y)+(−x·sin(y)+1)y'=0 | x·cos(y) + y = C₁ |
| `IntegratingFactor` | `ode_ifactor_mu_x` | (y−x²)+(x)y'=0 | Non-exact, μ=μ(x) |
| `IntegratingFactor` | `ode_ifactor_mu_y` | (2y²+3x)+(2xy)y'=0 | Non-exact, μ=μ(y) |
| `Bernoulli` | `ode_bernoulli_n2` | y' + y = y² | Standard n=2 case |
| `Bernoulli` | `ode_bernoulli_n3` | y' + y/x = x·y³ | n=3, verify substitution |
| `HomogeneousCoeff` | `ode_homogeneous_basic` | y' = (x+y)/(x−y) | Substitution v=y/x |
| `HomogeneousCoeff` | `ode_homogeneous_linear` | y' = y/x + 1 | Verify v=y/x reduction |
| `SecondOrderCCHomogeneous` | `ode_socc_distinct` | y''−3y'+2y=0 | y = C₁eˣ + C₂e²ˣ |
| `SecondOrderCCHomogeneous` | `ode_socc_repeated` | y''−2y'+y=0 | y = (C₁+C₂x)eˣ |
| `SecondOrderCCHomogeneous` | `ode_socc_complex` | y''+y=0 | y = C₁cos(x) + C₂sin(x) |
| `SecondOrderCCNonHomogeneous` | `ode_socc_nonhom_poly` | y''+y=x | y_p = x |
| `SecondOrderCCNonHomogeneous` | `ode_socc_nonhom_exp` | y''−y=eˣ | Variation or undetermined |
| `EulerCauchy` | `ode_euler_distinct` | x²y''+xy'−y=0 | y = C₁x + C₂/x |
| `EulerCauchy` | `ode_euler_complex` | x²y''+xy'+y=0 | y = C₁cos(ln x)+C₂sin(ln x) |
| `EulerCauchy` | `ode_euler_repeated` | x²y''−xy'+y=0 | y = x(C₁+C₂·ln x) |
| `VariationOfParameters` | `ode_vop_tan` | y''+y=tan(x) | Classic textbook example |
| `VariationOfParameters` | `ode_vop_sec` | y''+y=sec(x) | Another classic |
| `NthOrderReducible` | `ode_reducible_missing_x` | y·y''+y'²=0 | p=y' substitution |

**Verification strategy:** Each test should:
1. Assert `dsolve` returns `Some(result)`
2. Numerically verify the solution satisfies the ODE at 5+ points (reuse existing `verify_first_order_numerically` / `verify_second_order_numerically` helpers)
3. For tests where `dsolve` might legitimately fail (integrating factor, homogeneous coeff), accept `None` but log it — don't assert `Some`

---

### T2-2: ODE system solver — test defective matrix handling

**File:** New `tests/test_ode_systems.rs` (extend existing)

| Test name | System | Expected |
|-----------|--------|----------|
| `system_diagonal` | `[[1,0],[0,2]]` | `[C₁·eᵗ, C₂·e²ᵗ]` |
| `system_2x2_real` | `[[0,1],[-2,3]]` | Eigenvalue-based |
| `system_2x2_complex` | `[[0,-1],[1,0]]` | `[C₁cos(t)−C₂sin(t), ...]` |
| `system_defective_2x2` | `[[1,1],[0,1]]` | Should use Jordan form: `e^t·[C₁+(C₁t+C₂), C₁]` — **currently returns None, so test accepts None but documents the gap** |
| `system_3x3` | `[[1,0,0],[0,2,0],[0,0,3]]` | Diagonal, exact |
| `system_nonhomogeneous` | `A·x + b(t)` | Variation of parameters |

**For the defective matrix gap (T3-2), the test should be written NOW as:**
```rust
#[test]
fn system_defective_2x2_known_limitation() {
    // Currently returns None — will be fixed in T3-2
    let result = solve_ode_system(&a_matrix, &t);
    if let Some(solution) = result {
        // If someone fixes this, verify the solution is correct
        verify_system_solution(&a_matrix, &solution, &t);
    }
    // No assert — this documents a known limitation
}
```

---

### T2-3: Limits — expand test coverage

**File:** `tests/test_limits_infinity.rs` (extend) and `tests/test_limits.rs` (extend)

**Infinity limits to add:**

| Test name | Expression | Expected | What it covers |
|-----------|-----------|----------|----------------|
| `limit_exp_over_poly` | `lim(x→∞) eˣ/x²` | `+∞` | Exponential dominance |
| `limit_poly_over_exp` | `lim(x→∞) x¹⁰/eˣ` | `0` | Exponential wins |
| `limit_ln_over_poly` | `lim(x→∞) ln(x)/x` | `0` | Logarithmic growth |
| `limit_ln_over_sqrt` | `lim(x→∞) ln(x)/√x` | `0` | ln vs power |
| `limit_exp_neg_x_squared` | `lim(x→∞) e^{-x²}` | `0` | Gaussian decay |
| `limit_rational_neg_inf_even` | `lim(x→-∞) x⁴/(x²+1)` | `+∞` | Even degree at -∞ |
| `limit_rational_neg_inf_odd` | `lim(x→-∞) x³/(x²+1)` | `-∞` | Odd degree at -∞ |
| `limit_oscillating` | `lim(x→∞) sin(x)/x` | `0` | Bounded numerator |

**Finite-point limits to add:**

| Test name | Expression | Expected | What it covers |
|-----------|-----------|----------|----------------|
| `limit_lhopital_0_over_0` | `lim(x→0) (eˣ-1)/x` | `1` | L'Hôpital 0/0 |
| `limit_lhopital_inf_over_inf` | `lim(x→∞) x/eˣ` | `0` | L'Hôpital ∞/∞ |
| `limit_lhopital_repeated` | `lim(x→0) (eˣ-1-x)/x²` | `1/2` | Needs 2 applications |
| `limit_cube_root` | `lim(x→8) (x^{1/3}-2)/(x-8)` | `1/12` | Algebraic |
| `limit_one_sided_concept` | `lim(x→0⁺) 1/x` | `+∞` | (If one-sided limits are supported) |
| `limit_series_fallback` | `lim(x→0) sin(x)/x` via series | `1` | Verify series path works |

---

### T2-4: Series — expand test coverage

**File:** New `tests/test_series_comprehensive.rs`

| Test name | Function | Point | Order | Known coefficients |
|-----------|----------|-------|-------|-------------------|
| `taylor_exp_at_0` | eˣ | 0 | 8 | 1, 1, 1/2, 1/6, 1/24, ... |
| `taylor_sin_at_0` | sin(x) | 0 | 8 | 0, 1, 0, -1/6, 0, 1/120, ... |
| `taylor_cos_at_0` | cos(x) | 0 | 8 | 1, 0, -1/2, 0, 1/24, ... |
| `taylor_ln_at_1` | ln(x) | 1 | 6 | 0, 1, -1/2, 1/3, -1/4, ... |
| `taylor_atan_at_0` | atan(x) | 0 | 8 | 0, 1, 0, -1/3, 0, 1/5, ... |
| `taylor_sinh_at_0` | sinh(x) | 0 | 8 | 0, 1, 0, 1/6, 0, 1/120, ... |
| `taylor_cosh_at_0` | cosh(x) | 0 | 8 | 1, 0, 1/2, 0, 1/24, ... |
| `taylor_exp_at_1` | eˣ | 1 | 5 | e, e, e/2, e/6, e/24, ... |
| `taylor_sin_at_pi` | sin(x) | π | 5 | 0, -1, 0, 1/6, 0, ... |
| `taylor_composition` | exp(sin(x)) | 0 | 5 | Verify numerically |
| `laurent_1_over_x` | 1/x | 0 | 3 | Laurent pole order 1 |
| `laurent_1_over_sin` | 1/sin(x) | 0 | 3 | Laurent pole order 1 |
| `laurent_cot` | cos(x)/sin(x) | 0 | 5 | 1/x - x/3 - x³/45 ... |
| `series_numerical_accuracy` | All above | — | — | Evaluate at x=0.1, compare to f64 |

**Verification strategy:** For each series, evaluate the truncated polynomial at `x = point ± 0.1` and compare against the function's `eval_f64()`. Tolerance should be `O(h^{order})` where `h = 0.1`.

---

### T2-5: Polynomial factoring — test at boundaries

**File:** New `tests/test_factoring_boundaries.rs`

| Test name | Polynomial | Expected | What it covers |
|-----------|-----------|----------|----------------|
| `factor_degree_4_into_2_2` | `x⁴-5x²+4` = `(x²-1)(x²-4)` | Both quadratics found | Within Kronecker cap |
| `factor_degree_6_into_2_3` | `(x²+1)(x³-1)` | Both factors found | At Kronecker boundary |
| `factor_degree_8_into_4_4` | `(x⁴+1)(x⁴-1)` | At least `(x⁴-1)` splits | Tests degree > 6 boundary |
| `factor_irreducible_degree_5` | `x⁵+x+1` (irreducible over ℤ) | Returns unchanged | Graceful non-factoring |
| `factor_large_coefficients` | `999999x² - 1000001x + 2` | Finds rational roots | Tests near divisor cap |
| `factor_cyclotomic_12` | Φ₁₂(x) = x⁴-x²+1 | Irreducible (correctly) | Degree 4, no rational roots |
| `factor_product_of_linears` | `(x-1)(x-2)(x-3)(x-4)(x-5)` | All 5 linear factors | Rational Root Theorem |
| `factor_content_extraction` | `6x³ + 12x² + 6x` | `6x(x²+2x+1)` = `6x(x+1)²` | Content + SFD |

---

### T2-6: Matrix eigenvalues — test at boundaries

**File:** Extend `tests/test_matrix_linalg.rs`

| Test name | Matrix | Expected |
|-----------|--------|----------|
| `eigenvals_irrational_2x2` | `[[0,2],[1,0]]` | ±√2 |
| `eigenvals_repeated_3x3` | `[[2,1,0],[0,2,0],[0,0,3]]` | {2 (mult 2), 3} — 2 is defective |
| `eigenvals_complex_3x3` | `[[0,-1,0],[1,0,0],[0,0,1]]` | {±i, 1} |
| `eigenvals_4x4_factorable` | Block diagonal `[[A,0],[0,B]]` | Union of 2×2 eigenvalues |
| `eigenvals_5x5_known_roots` | Companion matrix of `(x-1)(x-2)(x-3)(x-4)(x-5)` | {1,2,3,4,5} |
| `eigenvals_incomplete_warns` | Companion matrix of x⁵+x+1 (irreducible) | Returns `Ok(vec![])`, logs warning |
| `det_transpose_identity` | Random 4×4 | `det(A) == det(Aᵀ)` |
| `cayley_hamilton` | 3×3 | `A³ - tr·A² + ... = 0` (numerically) |

---

### T2-7: Parser round-trip coverage

**File:** Extend `src/output/parse.rs` unit tests

**Phase 1: Add parsing for missing function names** (no grammar changes, just extend the function dispatch table):

| Function | Current parse support | Action |
|----------|--------------------|--------|
| `gamma(x)` | ❌ | Add to function table → `ExprNode::Gamma` |
| `erf(x)` | ❌ | Add → `ExprNode::Erf` |
| `erfc(x)` | ❌ | Add → `ExprNode::Erfc` |
| `floor(x)` | ❌ | Add → `ExprNode::Floor` |
| `ceil(x)` / `ceiling(x)` | ❌ | Add → `ExprNode::Ceiling` |
| `sign(x)` | ❌ | Add → `ExprNode::Sign` |
| `heaviside(x)` | ❌ | Add → `ExprNode::Heaviside` |
| `lambertw(x)` | ❌ | Add → `ExprNode::LambertW` |
| `factorial(x)` / `x!` | ❌ | Add → `ExprNode::Factorial` (postfix `!` requires lexer change) |
| `binomial(n,k)` / `C(n,k)` | ❌ | Add → `ExprNode::Binomial` |

**Phase 2: Add round-trip tests:**

```rust
fn assert_round_trip(input: &str) {
    let ctx = Context::new();
    let expr = ctx.parse(input).unwrap();
    let displayed = format!("{}", expr);
    let reparsed = ctx.parse(&displayed).unwrap();
    // Numerical equality at test point, not string equality
    // (canonical forms may differ from input)
    assert_numerically_equal(&expr, &reparsed, ...);
}
```

Test cases: `"sin(x)^2 + cos(x)^2"`, `"exp(x) * ln(x)"`, `"gamma(5)"`, `"erf(x)"`, `"floor(3.7)"`, `"binomial(10, 3)"`.

---

## Tier 3 — Feature Gaps (Missing Capabilities)

These are non-trivial features that would meaningfully expand symplex's capability.

### T3-1: Generalized eigenvectors for defective matrices in ODE systems

**File:** `src/calculus/ode.rs` lines 3134–3274 (`solve_ode_system_eigen`)

**Current state:** For eigenvalue λ with algebraic multiplicity m > geometric multiplicity g, the code only finds g eigenvectors from `null(A−λI)` and silently drops the remaining m−g modes, ultimately returning `None`.

**Fix:** Implement Jordan chain computation:
1. For each eigenvalue λ with multiplicity m, compute the **nullity chain**: `nullity((A−λI)^k)` for k = 1, 2, ..., until it equals m.
2. For each generalized eigenvector of rank k (in `null((A−λI)^k)` but not `null((A−λI)^{k-1})`), the ODE solution mode is:
   ```
   e^{λt} · [ v_k + t·v_{k-1} + t²/2!·v_{k-2} + ... + t^{k-1}/(k-1)!·v_1 ]
   ```
   where `(A−λI)·v_j = v_{j-1}` and `(A−λI)·v_1 = 0`.
3. The Matrix module already has Jordan form computation (`jordan_form` in matrix.rs). Consider reusing its nullity chain logic.

**Implementation steps:**
1. Add `fn generalized_eigenvectors(a: &Matrix, eigenvalue: &Ex, multiplicity: usize) -> Vec<Vec<Ex>>` to `matrix.rs`
2. In `solve_ode_system_eigen`, when `null_basis.len() < multiplicity`, call the generalized eigenvector function
3. Build the solution modes using the Jordan chain formula

**Tests:** The `system_defective_2x2_known_limitation` test from T2-2 should start passing.

**Estimated effort:** Medium. The matrix module already has the infrastructure (null space, matrix powers). The main work is building the solution modes from the Jordan chain.

---

### T3-2: Series fast-paths for common functions

**File:** `src/calculus/series.rs` lines 115–186 (`try_known_maclaurin`)

**Current state:** Only sin, cos, exp have fast-path Maclaurin series. Everything else goes through the generic n-derivatives path.

**Add fast paths for:**

| Function | Series | Complexity |
|----------|--------|------------|
| `sinh(x)` | `Σ x^{2k+1}/(2k+1)!` | Trivial (same as sin without alternating sign) |
| `cosh(x)` | `Σ x^{2k}/(2k)!` | Trivial |
| `ln(1+x)` | `Σ (-1)^{k+1} x^k / k` | Trivial (convergent for |x|<1) |
| `atan(x)` | `Σ (-1)^k x^{2k+1} / (2k+1)` | Trivial |
| `1/(1-x)` | `Σ x^k` | Geometric series |
| `(1+x)^α` | `Σ C(α,k) x^k` | Binomial series (needs generalized binomial coeff) |

**Implementation:** Follow the existing pattern — match on the expression node type in `try_known_maclaurin`, compute coefficients via closed-form formulas, build the polynomial.

**Tests:** Add to `tests/test_series_comprehensive.rs` (see T2-4).

---

### T3-3: Asymptotic series (expansion at infinity)

**File:** `src/calculus/series.rs`

**Current state:** `series()` only expands at finite points. No expansion at x → ∞.

**Fix:** Add `asymptotic_series(arena, expr, var, order)`:
1. Substitute `x = 1/t`
2. Compute `series(result, t, 0, order)` (Taylor around t=0)
3. Substitute back `t = 1/x`
4. This gives an expansion in powers of `1/x`

**Implementation:** ~30 lines wrapping the existing `series()`.

**Tests:**
| Expression | Expected leading terms |
|-----------|----------------------|
| `1/(x+1)` at ∞ | `1/x - 1/x² + 1/x³ - ...` |
| `exp(1/x)` at ∞ | `1 + 1/x + 1/(2x²) + ...` |
| `x/(x+1)` at ∞ | `1 - 1/x + 1/x² - ...` |

---

### T3-4: `oo` / `inf` display-parse consistency

**File:** `src/output/parse.rs` and `src/output/display.rs`

**Current state:** Display outputs `oo` for infinity, and the parser recognizes `oo`, `inf`, `Inf`, `infinity`. These are already consistent — but `-oo`, `zoo` (complex infinity), and `nan` need to be checked.

**Audit and fix:**
1. Verify `parse("-oo")` → `NegInfinity` (currently: unary minus + `oo`, should work)
2. Add `zoo` to parser → `ComplexInfinity`
3. Add `nan` / `NaN` to parser → `NaN`
4. Add `true` / `True` to parser → `BoolTrue`
5. Add `false` / `False` to parser → `BoolFalse`

---

## Tier 4 — Quality-of-Life & Polish

### T4-1: Un-ignore the passing test

**File:** `tests/test_arb_prec_special.rs` line 240

The test `integrate_sqrt_x2_plus_one_ftc` (line 277) now passes. Remove its `#[ignore]` annotation.

The test `integrate_sqrt_one_minus_x2_ftc` (line 240) still fails due to evaluation point issues. Investigate:
- The FTC checker uses points that may be outside the domain `[-1, 1]` for `√(1-x²)`
- Fix the test helper to use domain-appropriate points (e.g., x = 0.1, 0.3, 0.5, 0.7, 0.9)
- If the antiderivative itself is the issue (arena index problem), document it as a known limitation with a `// TODO` and keep the ignore

---

### T4-2: LaTeX `LogGamma` rendering

**File:** `src/output/latex.rs`

**Current:** `LogGamma(x)` renders as `\ln \Gamma\left(x\right)` which is misleading (LogGamma is a specific function, not the composition ln∘Γ).

**Fix:** Render as `\operatorname{LogGamma}\left(x\right)` or `\log\Gamma\left(x\right)`.

---

### T4-3: LaTeX multi-character symbol names

**File:** `src/output/latex.rs`

**Current:** Multi-character non-Greek symbols like `foo` render as `foo` in LaTeX, which typesets as *f*·*o*·*o* (italic product of three variables).

**Fix:** Detect multi-character, non-Greek symbol names and wrap them in `\operatorname{}`:
```rust
if name.len() > 1 && !GREEK_LETTERS.contains_key(name) && !name.contains('_') {
    write!(out, "\\operatorname{{{}}}", name)
} else {
    // existing logic
}
```

**Exception:** Names with subscripts (e.g., `x_1`) should NOT be wrapped — the existing subscript logic handles them correctly.

---

### T4-4: Compilation warnings cleanup

**Files:**
- `src/units/inference.rs:342` — unused import `crate::prelude::*`
- `src/units/inference.rs:501` — variable `F` should be snake_case
- `tests/test_dim_macro.rs:7` — unused import
- `tests/test_dim_macro.rs:146` — unused variable `m`
- `tests/test_units_proptest.rs:145` — variables `I`, `R` should be snake_case

**Fix:** Remove unused imports; rename variables or add `#[allow(non_snake_case)]` where physics convention dictates uppercase (e.g., `F` for force, `I` for current).

---

## Appendix A — Test Matrix

### Summary of new tests by tier

| Tier | New test functions | New test files |
|------|-------------------|----------------|
| T0 | ~14 | 1 new (heaviside), extend 1 |
| T1 | ~4 | extend 1 |
| T2 | ~65 | 3 new, extend 4 |
| T3 | ~15 | extend 2 |
| T4 | ~5 | extend 2 |
| **Total** | **~103** | **4 new files, 10 extended** |

### New test files to create

1. `tests/test_ode_comprehensive.rs` — Systematic coverage of all 13 ODE classes
2. `tests/test_series_comprehensive.rs` — Taylor, Laurent, fast-path, and numerical accuracy
3. `tests/test_factoring_boundaries.rs` — Polynomial factoring at capability boundaries
4. `tests/test_heaviside_integ.rs` — Heaviside and DiracDelta integration edge cases

### Test files to extend

1. `tests/test_limits_infinity.rs` — +8 tests (sign, exp/poly growth races)
2. `tests/test_limits.rs` — +6 tests (L'Hôpital, algebraic, series fallback)
3. `tests/test_matrix_linalg.rs` — +8 tests (irrational eigenvalues, boundaries, Cayley-Hamilton)
4. `tests/test_evalf_gaps.rs` — +4 tests (Piecewise safety)
5. `tests/test_arb_prec_special.rs` — +5 tests (digamma precision), un-ignore 1
6. `tests/test_ode_systems.rs` — +6 tests (defective matrices, nonhomogeneous)
7. `tests/test_parser.rs` — +10 tests (new function names, round-trips)
8. `src/output/parse.rs` — +6 unit tests (round-trip assertions)

---

## Appendix B — Files Touched Per Tier

### Tier 0 (Bugs)
| File | Change type |
|------|-------------|
| `src/calculus/limit.rs` | Bug fix (leading coeff sign) |
| `src/transforms/evalf.rs` | Bug fix (Piecewise fallback) |
| `tests/test_limits_infinity.rs` | Add 6 tests |
| `tests/test_evalf_gaps.rs` | Add 4 tests |
| `tests/test_heaviside_integ.rs` | New file, ~4 tests |

### Tier 1 (Safety)
| File | Change type |
|------|-------------|
| `src/poly/dense.rs` | Add `warn!` at 4 locations |
| `src/poly/polysys.rs` | Add `warn!` at 2 locations |
| `src/poly/mod.rs` | Add shared constants |
| `src/domains/matrix.rs` | Add `warn!` in eigenvals |
| `src/transforms/evalf.rs` | Add `arb_digamma` (~80 lines) |
| `tests/test_arb_prec_special.rs` | Add 5 digamma tests |
| `tests/test_matrix_linalg.rs` | Add 2 tests |

### Tier 2 (Test Coverage)
| File | Change type |
|------|-------------|
| `tests/test_ode_comprehensive.rs` | New file, ~26 tests |
| `tests/test_ode_systems.rs` | Extend, +6 tests |
| `tests/test_limits_infinity.rs` | Extend, +8 tests |
| `tests/test_limits.rs` | Extend, +6 tests |
| `tests/test_series_comprehensive.rs` | New file, ~14 tests |
| `tests/test_factoring_boundaries.rs` | New file, ~8 tests |
| `tests/test_matrix_linalg.rs` | Extend, +8 tests |
| `src/output/parse.rs` | Extend function table + tests |

### Tier 3 (Features)
| File | Change type |
|------|-------------|
| `src/calculus/ode.rs` | Generalized eigenvectors (~100 lines) |
| `src/domains/matrix.rs` | `generalized_eigenvectors` method (~60 lines) |
| `src/calculus/series.rs` | 6 new fast paths (~80 lines) + asymptotic series (~30 lines) |
| `src/output/parse.rs` | New function names + `zoo`/`nan`/`true`/`false` |

### Tier 4 (Polish)
| File | Change type |
|------|-------------|
| `tests/test_arb_prec_special.rs` | Un-ignore 1 test, fix 1 test |
| `src/output/latex.rs` | LogGamma + operatorname fixes |
| `src/units/inference.rs` | Warning cleanup |
| `tests/test_dim_macro.rs` | Warning cleanup |
| `tests/test_units_proptest.rs` | Warning cleanup |

---

## Execution Order

The tiers are ordered by priority, but within each tier, items are independent and can be parallelized. The recommended execution order:

```
Week 1: T0-1, T0-2, T0-3 (bugs — ship blockers)
         T4-4 (warning cleanup — 5 min)
         T4-1 (un-ignore passing test — 5 min)

Week 2: T1-1, T1-2, T1-3 (safety logging — all independent)
         T1-4 (digamma arb prec — standalone)

Week 3: T2-1 (ODE tests — largest single block of new tests)
         T2-3, T2-4 (limits + series tests — independent of ODE)

Week 4: T2-5, T2-6 (factoring + eigenvalue boundary tests)
         T2-7 (parser round-trip)

Week 5: T3-1 (generalized eigenvectors — depends on T2-6 tests existing)
         T3-2 (series fast paths — depends on T2-4 tests existing)

Week 6: T3-3, T3-4 (asymptotic series, parse consistency)
         T4-2, T4-3 (LaTeX polish)
```

Each week's items are independent of each other but ordered so that **tests are written before the features they'll eventually validate**. This ensures:
1. Known-limitation tests exist BEFORE features are implemented (test-first)
2. Bugs are fixed BEFORE test coverage is expanded (correct foundation)
3. Safety logging is added BEFORE boundary tests exercise the limits (observable failures)

---

## Success Criteria

After all tiers are complete:

- [ ] `cargo test` passes with 0 failures, 0 ignored (currently: 0 failures, 2 ignored)
- [ ] `cargo test` emits 0 warnings (currently: 5 warnings)
- [ ] Total test count increases from 5,616 to ≥5,720 (+104 new tests minimum)
- [ ] Every `warn!` added in T1 is exercised by at least one test (via `tracing-test`)
- [ ] `lim(x→+∞) -x³/x` returns `-∞` (not `+∞`)
- [ ] `Piecewise((1, False), (2, False)).evalf()` returns `Err`, not `Ok(2.0)`
- [ ] `ψ(1).evalf(50)` returns 40+ correct digits of `-γ`
- [ ] All 13 ODE classes have at least one test each
- [ ] All proptest regression seeds still pass
- [ ] All 263 SymPy cross-validation fixtures still pass
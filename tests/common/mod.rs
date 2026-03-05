//! Shared test infrastructure for symplex integration tests.
//!
//! Include in any test file with `mod common;` at the top,
//! then use `common::assert_math_eq(...)` etc.
//!
//! **Policy:** All helpers use `expect()`/`unwrap()` by default — no silent
//! bailouts. If evaluation can legitimately fail for some inputs (e.g. in
//! proptests), callers should handle the `Result` themselves.

#![allow(dead_code)]

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

/// Default integer evaluation points for numerical comparison.
/// Chosen to avoid common singularities (0, ±1) while covering
/// both signs and a range of magnitudes.
pub const DEFAULT_POINTS_I64: &[i64] = &[-3, -2, 2, 3, 5, 7];

/// Default tolerance for floating-point comparison.
pub const DEFAULT_TOL: f64 = 1e-9;

/// Evaluation points used by `assert_ftc` — chosen to avoid zeros,
/// poles, and branch cuts of common elementary functions.
pub const FTC_POINTS: &[f64] = &[0.3, 0.7, 1.4];

// ═══════════════════════════════════════════════════════════════════════════
// Float comparison
// ═══════════════════════════════════════════════════════════════════════════

/// Approximate equality for `f64` values with NaN/Inf handling.
///
/// Uses mixed absolute + relative tolerance:
/// `|a - b| < tol * max(|a|, |b|, 1.0)`
///
/// Special cases:
/// - `NaN == NaN` → true
/// - `+Inf == +Inf` → true
/// - `+Inf == -Inf` → false
pub fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_infinite() && b.is_infinite() {
        return a.signum() == b.signum();
    }
    if a.is_nan() || b.is_nan() || a.is_infinite() || b.is_infinite() {
        return false;
    }
    let scale = a.abs().max(b.abs()).max(1.0);
    (a - b).abs() < tol * scale
}

// ═══════════════════════════════════════════════════════════════════════════
// Numerical equality
// ═══════════════════════════════════════════════════════════════════════════

/// Assert two expressions are numerically equal at multiple integer points
/// using the default points and tolerance.
///
/// # Panics
/// - If evaluation fails for one side but succeeds for the other at any point.
/// - If no point could be evaluated (vacuous test detection).
/// - If numerical values differ beyond tolerance.
pub fn assert_math_eq(a: &Ex, b: &Ex, var: &Ex, label: &str) {
    assert_math_eq_tol(a, b, var, DEFAULT_POINTS_I64, DEFAULT_TOL, label);
}

/// Assert two expressions are numerically equal at specified integer points.
pub fn assert_math_eq_tol(
    a: &Ex,
    b: &Ex,
    var: &Ex,
    points: &[i64],
    tol: f64,
    label: &str,
) {
    let mut checked = 0usize;
    for &pt in points {
        let va = a.subs_i64(var, pt).eval().evalf_f64();
        let vb = b.subs_i64(var, pt).eval().evalf_f64();
        match (va, vb) {
            (Ok(av), Ok(bv)) => {
                checked += 1;
                assert!(
                    approx_eq(av, bv, tol),
                    "{label} at {var}={pt}: {av} vs {bv} (diff={}, tol={})",
                    (av - bv).abs(),
                    tol
                );
            }
            // Both fail at same point — skip (e.g. singularity)
            (Err(_), Err(_)) => {}
            (Ok(av), Err(e)) => {
                panic!("{label} at {var}={pt}: a evaluated to {av} but b failed: {e}");
            }
            (Err(e), Ok(bv)) => {
                panic!("{label} at {var}={pt}: a failed: {e} but b evaluated to {bv}");
            }
        }
    }
    assert!(
        checked > 0,
        "{label}: no evaluation points succeeded — test is vacuous (tried {points:?})"
    );
}

/// Assert two expressions are numerically equal at specified rational points.
/// Useful when integer points hit singularities or are otherwise unsuitable.
pub fn assert_math_eq_rational(
    a: &Ex,
    b: &Ex,
    var: &Ex,
    points: &[(i64, i64)],
    tol: f64,
    label: &str,
) {
    let ctx = symplex::default_context();
    let mut checked = 0usize;
    for &(p, q) in points {
        let pt = ctx.rational(p, q);
        let va = a.subs(var, &pt).eval().evalf_f64();
        let vb = b.subs(var, &pt).eval().evalf_f64();
        match (va, vb) {
            (Ok(av), Ok(bv)) => {
                checked += 1;
                assert!(
                    approx_eq(av, bv, tol),
                    "{label} at {var}={p}/{q}: {av} vs {bv} (diff={})",
                    (av - bv).abs(),
                );
            }
            (Err(_), Err(_)) => {}
            (Ok(av), Err(e)) => {
                panic!("{label} at {var}={p}/{q}: a={av} but b failed: {e}");
            }
            (Err(e), Ok(bv)) => {
                panic!("{label} at {var}={p}/{q}: a failed: {e} but b={bv}");
            }
        }
    }
    assert!(
        checked > 0,
        "{label}: no rational evaluation points succeeded — test is vacuous"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// FTC verification (Fundamental Theorem of Calculus)
// ═══════════════════════════════════════════════════════════════════════════

/// Assert the Fundamental Theorem of Calculus:
///
///     d/dx( ∫ f(x) dx ) ≈ f(x)
///
/// 1. Integrates `integrand` with respect to `var`.
/// 2. Asserts the result is not an unevaluated `Integral` node.
/// 3. Differentiates the antiderivative.
/// 4. Asserts numerical equality with the original integrand at 3 points.
///
/// # Panics
/// - If integration returns an unevaluated Integral.
/// - If numerical FTC check fails at any test point.
pub fn assert_ftc(integrand: &Ex, var: &Ex, label: &str) {
    assert_ftc_tol(integrand, var, DEFAULT_TOL, label);
}

/// FTC assertion with configurable tolerance.
pub fn assert_ftc_tol(integrand: &Ex, var: &Ex, tol: f64, label: &str) {
    let antideriv = integrand.integrate(var);
    let s = format!("{antideriv}");
    assert!(
        !s.contains("Integral"),
        "{label}: integration returned unevaluated Integral: {s}"
    );
    let deriv = antideriv.diff(var);

    let ctx = symplex::default_context();
    let mut checked = 0usize;
    for &pt_f in FTC_POINTS {
        // Convert to rational to avoid float contamination
        let numer = (pt_f * 1000.0).round() as i64;
        let pt = ctx.rational(numer, 1000);

        let original_val = integrand.subs(var, &pt).eval().evalf_f64();
        let derived_val = deriv.subs(var, &pt).eval().evalf_f64();

        match (original_val, derived_val) {
            (Ok(o), Ok(d)) => {
                checked += 1;
                let scale = o.abs().max(d.abs()).max(1.0);
                assert!(
                    (o - d).abs() < tol * scale,
                    "FTC failed for {label} at {var}={pt_f}: \
                     integrand={o}, d/dx(antideriv)={d}, diff={}, \
                     antideriv='{antideriv}', deriv='{deriv}'",
                    (o - d).abs(),
                );
            }
            (Err(_), Err(_)) => {}
            (Ok(o), Err(e)) => {
                panic!(
                    "FTC {label} at {var}={pt_f}: integrand={o} but derivative failed: {e}"
                );
            }
            (Err(e), Ok(d)) => {
                panic!(
                    "FTC {label} at {var}={pt_f}: integrand failed: {e} but derivative={d}"
                );
            }
        }
    }
    assert!(
        checked > 0,
        "FTC {label}: no evaluation points succeeded — test is vacuous"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Root verification
// ═══════════════════════════════════════════════════════════════════════════

/// Verify that every root satisfies `poly(root) ≈ 0`.
///
/// Tries exact symbolic check first (`eval()` → "0"), then falls back
/// to numerical evaluation via `evalf_f64()` or `evalf_complex64()`.
///
/// # Panics
/// - If any root produces a nonzero residual beyond tolerance.
/// - If evaluation fails for any root.
pub fn verify_roots(poly: &Ex, var: &Ex, roots: &[Ex], tol: f64) {
    assert!(
        !roots.is_empty(),
        "verify_roots: empty roots list for poly '{poly}'"
    );
    for (i, root) in roots.iter().enumerate() {
        let substituted = poly.subs(var, root).eval();
        let s = format!("{substituted}");
        if s == "0" {
            continue; // Exact zero — perfect
        }

        // Try real evaluation first
        if let Ok(v) = substituted.evalf_f64() {
            assert!(
                v.abs() < tol,
                "root {i} ({root}) doesn't satisfy poly '{poly}': residual = {v}"
            );
            continue;
        }

        // Fall back to complex evaluation
        match substituted.evalf_complex64() {
            Ok((re, im)) => {
                let mag = (re * re + im * im).sqrt();
                assert!(
                    mag < tol,
                    "root {i} ({root}) doesn't satisfy poly '{poly}': \
                     complex residual = ({re}, {im}), |r| = {mag}"
                );
            }
            Err(e) => {
                panic!(
                    "root {i} ({root}) for poly '{poly}': \
                     residual '{substituted}' can't be evaluated: {e}"
                );
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Simplification value-preservation
// ═══════════════════════════════════════════════════════════════════════════

/// Assert that `expr.simplify()` produces the same numerical value as the
/// original at multiple evaluation points.
pub fn assert_simplify_preserves_value(expr: &Ex, var: &Ex, label: &str) {
    let simplified = expr.simplify();
    assert_math_eq(expr, &simplified, var, &format!("simplify preserves value: {label}"));
}

/// Assert that `expr.full_simplify()` produces the same numerical value as
/// the original at multiple evaluation points.
pub fn assert_full_simplify_preserves_value(expr: &Ex, var: &Ex, label: &str) {
    let simplified = expr.full_simplify();
    assert_math_eq(
        expr,
        &simplified,
        var,
        &format!("full_simplify preserves value: {label}"),
    );
}

/// Assert that `expr.expand()` produces the same numerical value as the
/// original at multiple evaluation points.
pub fn assert_expand_preserves_value(expr: &Ex, var: &Ex, label: &str) {
    let expanded = expr.expand();
    assert_math_eq(expr, &expanded, var, &format!("expand preserves value: {label}"));
}

// ═══════════════════════════════════════════════════════════════════════════
// ODE verification
// ═══════════════════════════════════════════════════════════════════════════

/// Verify a first-order ODE solution by numerical substitution.
///
/// Given an ODE of the form `F(x, y, y') = 0` and a solution `y = sol(x)`,
/// substitutes the solution and its derivative into the ODE expression
/// and checks the residual is near zero.
///
/// `ode_expr` should be an expression that equals zero when the ODE is satisfied.
/// For example, for y' + 2y = 0, pass `diff(y,x) + 2*y` (which should be zero).
pub fn verify_ode_first_order(
    ode_expr: &Ex,
    solution: &Ex,
    func_var: &Ex,
    indep_var: &Ex,
    points: &[(i64, i64)],
    tol: f64,
    label: &str,
) {
    let ctx = symplex::default_context();
    let dsol = solution.diff(indep_var);

    // Build the derivative symbol (formal derivative node)
    let deriv_sym = func_var.diff(indep_var);

    let mut checked = 0usize;
    for &(p, q) in points {
        let pt = ctx.rational(p, q);

        // Substitute y = sol and y' = dsol into the ODE
        let residual = ode_expr
            .subs(func_var, solution)
            .subs(&deriv_sym, &dsol)
            .subs(indep_var, &pt)
            .eval();

        if let Ok(v) = residual.evalf_f64() {
            checked += 1;
            assert!(
                v.abs() < tol,
                "ODE {label}: residual at {indep_var}={p}/{q} is {v} (expected ~0)"
            );
        }
    }
    assert!(
        checked > 0,
        "ODE {label}: no points could be evaluated — test is vacuous"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Display helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Assert that `format!("{expr}")` exactly equals `expected`.
/// Thin wrapper with a descriptive error message.
pub fn assert_display_eq(expr: &Ex, expected: &str) {
    let actual = format!("{expr}");
    assert_eq!(
        actual, expected,
        "display mismatch: expected '{expected}', got '{actual}'"
    );
}

/// Assert that `format!("{expr}")` contains all of the given substrings.
pub fn assert_display_contains(expr: &Ex, substrings: &[&str], context: &str) {
    let s = format!("{expr}");
    for sub in substrings {
        assert!(
            s.contains(sub),
            "{context}: display '{s}' does not contain '{sub}'"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Bail counter for proptests
// ═══════════════════════════════════════════════════════════════════════════

/// A counter that tracks how many times a proptest evaluation was skipped.
/// Assert `counter.assert_not_vacuous()` at the end to ensure at least
/// one iteration actually ran the real assertion.
pub struct BailCounter {
    pub checked: usize,
    pub skipped: usize,
    pub label: String,
}

impl BailCounter {
    pub fn new(label: &str) -> Self {
        Self {
            checked: 0,
            skipped: 0,
            label: label.to_string(),
        }
    }

    pub fn check(&mut self) {
        self.checked += 1;
    }

    pub fn skip(&mut self) {
        self.skipped += 1;
    }

    /// Panics if zero iterations were checked.
    pub fn assert_not_vacuous(&self) {
        assert!(
            self.checked > 0,
            "{}: all {} iterations were skipped — test is vacuous",
            self.label, self.skipped
        );
    }

    /// Panics if the skip rate exceeds the given fraction (0.0–1.0).
    pub fn assert_skip_rate_below(&self, max_rate: f64) {
        let total = self.checked + self.skipped;
        if total == 0 {
            panic!("{}: no iterations at all", self.label);
        }
        let rate = self.skipped as f64 / total as f64;
        assert!(
            rate <= max_rate,
            "{}: skip rate {:.1}% exceeds max {:.1}% ({} checked, {} skipped)",
            self.label,
            rate * 100.0,
            max_rate * 100.0,
            self.checked,
            self.skipped
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Expression evaluation helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluate an expression at an integer point after substitution + eval.
/// Returns the f64 value or panics with a descriptive message.
pub fn eval_at_i64(expr: &Ex, var: &Ex, pt: i64) -> f64 {
    expr.subs_i64(var, pt)
        .eval()
        .evalf_f64()
        .unwrap_or_else(|e| panic!("eval_at_i64({expr}, {var}={pt}) failed: {e}"))
}

/// Evaluate an expression at a rational point after substitution + eval.
/// Returns the f64 value or panics with a descriptive message.
pub fn eval_at_rational(expr: &Ex, var: &Ex, p: i64, q: i64) -> f64 {
    let ctx = symplex::default_context();
    let pt = ctx.rational(p, q);
    expr.subs(var, &pt)
        .eval()
        .evalf_f64()
        .unwrap_or_else(|e| panic!("eval_at_rational({expr}, {var}={p}/{q}) failed: {e}"))
}

// ═══════════════════════════════════════════════════════════════════════════
// Spot-check helpers for inequality solutions
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluate a polynomial at a point and assert it's positive.
pub fn assert_positive_at(poly: &Ex, var: &Ex, pt: i64, label: &str) {
    let val = eval_at_i64(poly, var, pt);
    assert!(
        val > 0.0,
        "{label}: expected {poly} > 0 at {var}={pt}, got {val}"
    );
}

/// Evaluate a polynomial at a point and assert it's negative.
pub fn assert_negative_at(poly: &Ex, var: &Ex, pt: i64, label: &str) {
    let val = eval_at_i64(poly, var, pt);
    assert!(
        val < 0.0,
        "{label}: expected {poly} < 0 at {var}={pt}, got {val}"
    );
}

/// Evaluate a polynomial at a rational point and assert it's positive.
pub fn assert_positive_at_rational(poly: &Ex, var: &Ex, p: i64, q: i64, label: &str) {
    let val = eval_at_rational(poly, var, p, q);
    assert!(
        val > 0.0,
        "{label}: expected {poly} > 0 at {var}={p}/{q}, got {val}"
    );
}

/// Evaluate a polynomial at a rational point and assert it's negative.
pub fn assert_negative_at_rational(poly: &Ex, var: &Ex, p: i64, q: i64, label: &str) {
    let val = eval_at_rational(poly, var, p, q);
    assert!(
        val < 0.0,
        "{label}: expected {poly} < 0 at {var}={p}/{q}, got {val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Macros
// ═══════════════════════════════════════════════════════════════════════════

/// Assert that `simplify(expr)` produces the given display string.
#[macro_export]
macro_rules! assert_simplifies_to {
    ($expr:expr, $expected:expr) => {
        let __expr = &$expr;
        let __result = __expr.simplify();
        let __actual = format!("{__result}");
        assert_eq!(
            __actual, $expected,
            "simplify({__expr}) expected '{}', got '{__actual}'",
            $expected
        );
    };
}

/// Assert that `simplify(expr)` leaves the display string unchanged.
#[macro_export]
macro_rules! assert_simplify_unchanged {
    ($expr:expr) => {
        let __expr = &$expr;
        let __before = format!("{__expr}");
        let __result = __expr.simplify();
        let __after = format!("{__result}");
        assert_eq!(
            __before, __after,
            "simplify({__before}) should be unchanged, got '{__after}'"
        );
    };
}

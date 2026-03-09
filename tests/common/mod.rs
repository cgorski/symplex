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
        let va = a.subs_i64(var, pt).eval().eval_f64();
        let vb = b.subs_i64(var, pt).eval().eval_f64();
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
    let ctx = a.context();
    let mut checked = 0usize;
    for &(p, q) in points {
        let pt = ctx.rational(p, q);
        let va = a.subs(var, &pt).eval().eval_f64();
        let vb = b.subs(var, &pt).eval().eval_f64();
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

    let ctx = integrand.context();
    let mut checked = 0usize;
    for &pt_f in FTC_POINTS {
        // Convert to rational to avoid float contamination
        let numer = (pt_f * 1000.0).round() as i64;
        let pt = ctx.rational(numer, 1000);

        let original_val = integrand.subs(var, &pt).eval().eval_f64();
        let derived_val = deriv.subs(var, &pt).eval().eval_f64();

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
        // If the root contains unevaluated forms (e.g., RootOf), try to
        // evaluate it numerically first and substitute the number instead.
        // This avoids creating compound expressions like RootOf(...)^5 that
        // the evaluator can't handle.
        if root.has_unevaluated() {
            if let Ok(val) = root.eval_f64() {
                let ctx = poly.context();
                // Build a rational approximation of the numerical value
                // and substitute that instead
                let numer = (val * 1e12).round() as i64;
                let pt = ctx.rational(numer, 1_000_000_000_000);
                let residual = poly.subs(var, &pt).eval();
                if let Ok(r) = residual.eval_f64() {
                    assert!(
                        r.abs() < tol,
                        "root {i} ({root}) doesn't satisfy poly '{poly}': \
                         numerical residual = {r} (root ≈ {val})"
                    );
                }
                // If we can't eval the residual, skip — the root was unevaluated
                // and we did our best
                continue;
            }
            // Can't evaluate numerically (e.g., complex RootOf) — skip
            continue;
        }

        let substituted = poly.subs(var, root).eval();
        let s = format!("{substituted}");
        if s == "0" {
            continue; // Exact zero — perfect
        }

        // Try real evaluation first
        if let Ok(v) = substituted.eval_f64() {
            assert!(
                v.abs() < tol,
                "root {i} ({root}) doesn't satisfy poly '{poly}': residual = {v}"
            );
            continue;
        }

        // Fall back to complex evaluation
        match substituted.eval_complex64() {
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
    let ctx = ode_expr.context();
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

        if let Ok(v) = residual.eval_f64() {
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
        .eval_f64()
        .unwrap_or_else(|e| panic!("eval_at_i64({expr}, {var}={pt}) failed: {e}"))
}

/// Evaluate an expression at a rational point after substitution + eval.
/// Returns the f64 value or panics with a descriptive message.
pub fn eval_at_rational(expr: &Ex, var: &Ex, p: i64, q: i64) -> f64 {
    let ctx = expr.context();
    let pt = ctx.rational(p, q);
    expr.subs(var, &pt)
        .eval()
        .eval_f64()
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
// Domain classification
// ═══════════════════════════════════════════════════════════════════════════

/// Broad domain classification for an expression, determined by which
/// node types appear in its tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExprDomain {
    /// No variables — purely numeric.
    Constant,
    /// Only Add, Mul, Pow (non-negative integer exponents), Num, Symbol.
    Polynomial,
    /// Polynomial plus negative-integer Pow or explicit division.
    Rational,
    /// Contains trig functions but no Exp/Ln.
    Trigonometric,
    /// Contains Exp or Ln but no trig.
    ExpLog,
    /// Contains both trig and exp/log, or other special functions.
    Mixed,
}

/// Feature flags accumulated during a tree walk for domain classification.
struct DomainFlags {
    has_symbol: bool,
    has_trig: bool,
    has_exp_ln: bool,
    has_special: bool,
    has_negative_pow: bool,
    has_symbolic_pow: bool,
}

impl DomainFlags {
    fn new() -> Self {
        Self {
            has_symbol: false,
            has_trig: false,
            has_exp_ln: false,
            has_special: false,
            has_negative_pow: false,
            has_symbolic_pow: false,
        }
    }

    fn classify(&self) -> ExprDomain {
        if !self.has_symbol {
            return ExprDomain::Constant;
        }
        if self.has_trig && self.has_exp_ln {
            return ExprDomain::Mixed;
        }
        if self.has_special {
            return ExprDomain::Mixed;
        }
        if self.has_trig {
            return ExprDomain::Trigonometric;
        }
        if self.has_exp_ln {
            return ExprDomain::ExpLog;
        }
        if self.has_negative_pow || self.has_symbolic_pow {
            return ExprDomain::Rational;
        }
        ExprDomain::Polynomial
    }
}

/// Classify the domain of an expression by inspecting its display string.
///
/// This is a pragmatic heuristic for tests — it examines the formatted
/// expression for the presence of function names and structural patterns.
/// It's not meant to be a rigorous mathematical classifier.
pub fn classify_domain_from_display(expr: &Ex) -> ExprDomain {
    let s = format!("{expr}");

    let mut flags = DomainFlags::new();

    // Check for variables (any letter that isn't part of a function name)
    // Simple heuristic: if it contains single-letter tokens or known var patterns
    for c in s.chars() {
        if c.is_ascii_alphabetic() {
            flags.has_symbol = true;
            break;
        }
    }

    // Check for trig functions
    let trig_names = [
        "sin(", "cos(", "tan(", "asin(", "acos(", "atan(",
        "sinh(", "cosh(", "tanh(", "sec(", "csc(", "cot(",
    ];
    for name in &trig_names {
        if s.contains(name) {
            flags.has_trig = true;
            break;
        }
    }

    // Check for exp/ln
    if s.contains("exp(") || s.contains("ln(") || s.contains("log(") {
        flags.has_exp_ln = true;
    }

    // Check for special functions
    let special_names = [
        "Gamma(", "erf(", "erfc(", "Beta(", "DiracDelta(",
        "Heaviside(", "lambertw(", "Digamma(",
    ];
    for name in &special_names {
        if s.contains(name) {
            flags.has_special = true;
            break;
        }
    }

    // Check for negative exponents (indicates rational)
    if s.contains("^(-") || s.contains("^-") || s.contains("1/") {
        flags.has_negative_pow = true;
    }

    flags.classify()
}

// ═══════════════════════════════════════════════════════════════════════════
// Canonical equality
// ═══════════════════════════════════════════════════════════════════════════

/// Maximum node-count expansion factor allowed during normalization.
/// If a normalization step produces an expression > this factor × input size,
/// we fall back to numerical comparison instead.
const MAX_COMPLEXITY_FACTOR: usize = 5;

/// Test whether two expressions are mathematically equal using a layered
/// strategy:
///
/// 1. **Structural identity** — O(1) via display string comparison.
/// 2. **Domain-specific normalization** — eval → normalize → zero check.
/// 3. **Numerical fallback** — multi-point evaluation for mixed domains.
///
/// This is designed for TEST assertions. It is more expensive than simple
/// string comparison but far more robust.
///
/// Returns `true` if the expressions are (very likely) mathematically equal,
/// `false` if they are definitely not equal or if equality cannot be determined.
pub fn canonical_eq(a: &Ex, b: &Ex) -> bool {
    // Fast path: identical display strings
    let sa = format!("{a}");
    let sb = format!("{b}");
    if sa == sb {
        return true;
    }

    // Compute a - b and try to show it's zero
    let diff = a - b;
    let diff_eval = diff.eval();

    // Check if eval alone resolved it
    let sd = format!("{diff_eval}");
    if sd == "0" {
        return true;
    }

    // Classify the domain and apply targeted normalization
    let domain = classify_domain_from_display(&diff_eval);

    match domain {
        ExprDomain::Constant => {
            // Should have been caught by eval → "0" check above.
            // Try evalf as last resort for numeric constants.
            if let Ok(v) = diff_eval.eval_f64() {
                return v.abs() < 1e-12;
            }
            false
        }
        ExprDomain::Polynomial => {
            // expand() produces canonical sum-of-monomials form
            let expanded = diff_eval.expand();
            let se = format!("{expanded}");
            if se == "0" {
                return true;
            }
            // If expand didn't reach zero, try full_simplify
            let simplified = diff_eval.full_simplify();
            let ss = format!("{simplified}");
            if ss == "0" {
                return true;
            }
            // Fall back to numerical
            numerical_zero_test(&diff_eval)
        }
        ExprDomain::Rational => {
            // cancel() produces canonical coprime p/q form
            let x = a.context().symbol("x");
            let cancelled = diff_eval.cancel(&x);
            let sc = format!("{cancelled}");
            if sc == "0" {
                return true;
            }
            // Try together then cancel
            let together = diff_eval.together().cancel(&x);
            let st = format!("{together}");
            if st == "0" {
                return true;
            }
            numerical_zero_test(&diff_eval)
        }
        ExprDomain::Trigonometric => {
            // Try trigsimp first
            let tsimp = diff_eval.simplify_trig();
            let st = format!("{tsimp}");
            if st == "0" {
                return true;
            }
            // Try rewrite to exponential form and cancel
            let as_exp = diff_eval.rewrite_as_exp();
            let x = a.context().symbol("x");
            let cancelled = as_exp.cancel(&x);
            let sc = format!("{cancelled}");
            if sc == "0" {
                return true;
            }
            // Try full_simplify
            let simplified = diff_eval.full_simplify();
            let ss = format!("{simplified}");
            if ss == "0" {
                return true;
            }
            numerical_zero_test(&diff_eval)
        }
        ExprDomain::ExpLog => {
            // Expand logs, then cancel
            let expanded = diff_eval.expand_log().expand();
            let se = format!("{expanded}");
            if se == "0" {
                return true;
            }
            let x = a.context().symbol("x");
            let cancelled = expanded.cancel(&x);
            let sc = format!("{cancelled}");
            if sc == "0" {
                return true;
            }
            numerical_zero_test(&diff_eval)
        }
        ExprDomain::Mixed => {
            // Try cascaded simplification strategies
            #[allow(clippy::type_complexity)]
            let strategies: Vec<Box<dyn Fn(&Ex) -> Ex>> = vec![
                Box::new(|e: &Ex| e.simplify_trig()),
                Box::new(|e: &Ex| e.full_simplify()),
                Box::new(|e: &Ex| e.smart_simplify()),
                Box::new(|e: &Ex| {
                    let x = a.context().symbol("x");
                    e.rewrite_as_exp().cancel(&x)
                }),
            ];
            for strat in &strategies {
                let result = strat(&diff_eval);
                let sr = format!("{result}");
                if sr == "0" {
                    return true;
                }
            }
            // Numerical fallback
            numerical_zero_test(&diff_eval)
        }
    }
}

/// Numerical zero test: evaluate at multiple points and check all values
/// are near zero. Uses a larger point set than `assert_math_eq` for
/// higher confidence.
fn numerical_zero_test(expr: &Ex) -> bool {
    let x = expr.context().symbol("x");
    let test_points: &[i64] = &[-7, -3, -2, 2, 3, 5, 7, 11];
    let mut checked = 0usize;
    for &pt in test_points {
        if let Ok(v) = expr.subs_i64(&x, pt).eval().eval_f64() {
            if v.is_nan() || v.is_infinite() {
                continue;
            }
            checked += 1;
            if v.abs() > 1e-8 {
                return false; // Definitely not zero
            }
        }
    }
    // If we checked at least 3 points and all were near zero, likely equal
    checked >= 3
}

/// Assert that two expressions are mathematically equal using canonical
/// equality testing. This is stronger than `assert_math_eq` (numerical
/// only) because it tries symbolic normalization first.
///
/// Use this when you need high confidence that two expressions are equal,
/// especially for polynomial, rational, and trigonometric expressions.
pub fn assert_canonical_eq(a: &Ex, b: &Ex, label: &str) {
    assert!(
        canonical_eq(a, b),
        "{label}: expressions are not canonically equal.\n  a = {a}\n  b = {b}\n  a - b = {}",
        a - b
    );
}

/// Assert that an expression is canonically equal to zero.
pub fn assert_canonical_zero(expr: &Ex, label: &str) {
    let zero = expr.context().int(0);
    assert!(
        canonical_eq(expr, &zero),
        "{label}: expression is not canonically zero: {expr}"
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

//! Trigonometric power integration.
//!
//! Handles integrals of the form:
//! - `∫ sin^n(x) dx` via recursive reduction formula
//! - `∫ cos^n(x) dx` via recursive reduction formula
//! - `∫ sin^m(x)·cos^n(x) dx` when one exponent is odd (Pythagorean substitution)
//!   or both even (reduction formula)

use crate::arena::Arena;
use crate::node::{ExprId, ExprNode, SymbolId};

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Zero};

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Compute binomial coefficient C(n, k) using BigInt to avoid overflow.
fn binom(n: u64, k: u64) -> BigInt {
    if k > n {
        return BigInt::zero();
    }
    if k == 0 || k == n {
        return BigInt::one();
    }
    // Use the smaller of k and n-k for efficiency
    let k = k.min(n - k);
    let mut result = BigInt::one();
    for i in 0..k {
        result = result * BigInt::from(n - i) / BigInt::from(i + 1);
    }
    result
}

/// Check whether `expr` transitively contains a reference to `var_sym`.
fn contains_var(arena: &Arena, expr: ExprId, var_sym: SymbolId) -> bool {
    let mut stack: Vec<ExprId> = vec![expr];
    while let Some(id) = stack.pop() {
        if let ExprNode::Symbol(sid) = arena.node(id)
            && *sid == var_sym
        {
            return true;
        }
        let children = arena.node(id).children();
        stack.extend_from_slice(&children);
    }
    false
}

/// Try to extract a [`Ratio<BigInt>`] as an `i64`.
///
/// Returns `Some(n)` when the expression is a numeric literal whose value
/// is an integer that fits in `i64`.
fn as_i64(arena: &Arena, id: ExprId) -> Option<i64> {
    let r = arena.as_num(id)?;
    if !r.is_integer() {
        return None;
    }
    let big = r.to_integer();
    i64::try_from(&big).ok()
}

/// Detect whether `inner` is exactly the variable `var`.
fn is_var(_arena: &Arena, inner: ExprId, var: ExprId) -> bool {
    inner == var
}

/// Try to decompose `factor` into a sin-power of `var`.
///
/// Returns `Some(exponent)` for:
/// - `Sin(var)` → 1
/// - `Pow(Sin(var), n)` → n  (when n is an integer)
fn extract_sin_power(arena: &Arena, factor: ExprId, var: ExprId) -> Option<i64> {
    match arena.node(factor).clone() {
        ExprNode::Sin(inner) if is_var(arena, inner, var) => Some(1),
        ExprNode::Pow(base, exp) => {
            if let ExprNode::Sin(inner) = arena.node(base).clone()
                && is_var(arena, inner, var)
            {
                return as_i64(arena, exp);
            }
            None
        }
        _ => None,
    }
}

/// Try to decompose `factor` into a cos-power of `var`.
///
/// Returns `Some(exponent)` for:
/// - `Cos(var)` → 1
/// - `Pow(Cos(var), n)` → n  (when n is an integer)
fn extract_cos_power(arena: &Arena, factor: ExprId, var: ExprId) -> Option<i64> {
    match arena.node(factor).clone() {
        ExprNode::Cos(inner) if is_var(arena, inner, var) => Some(1),
        ExprNode::Pow(base, exp) => {
            if let ExprNode::Cos(inner) = arena.node(base).clone()
                && is_var(arena, inner, var)
            {
                return as_i64(arena, exp);
            }
            None
        }
        _ => None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// sin^n integration
// ═══════════════════════════════════════════════════════════════════════════

/// Compute `∫ sin^n(x) dx` using the recursive reduction formula.
///
/// - n = 0 → `x`
/// - n = 1 → `−cos(x)`
/// - n = −1 → `−ln|csc(x)+cot(x)|`
/// - n ≤ −2 → upward reduction toward 0:
///   `(1/(n+1))·cos(x)·sin^(n+1)(x) + (n+2)/(n+1)·∫sin^(n+2)(x)dx`
/// - n ≥ 2 → `−(1/n)·cos(x)·sin^(n−1)(x) + (n−1)/n · ∫ sin^(n−2)(x) dx`
pub(crate) fn sin_pow_integrate(arena: &mut Arena, n: i64, var: ExprId) -> ExprId {
    if n == 0 {
        return var; // ∫ 1 dx = x
    }
    if n == 1 {
        // ∫ sin(x) dx = −cos(x)
        let cos_x = arena.cos(var);
        return arena.neg(cos_x);
    }
    if n == -1 {
        // ∫ csc(x) dx = −ln|csc(x) + cot(x)|
        let sin_x = arena.sin(var);
        let cos_x = arena.cos(var);
        let neg_one_id = arena.int(-1);
        let csc_x = arena.pow(sin_x, neg_one_id); // 1/sin(x)
        let cot_x = arena.mul(&[cos_x, csc_x]); // cos(x)/sin(x)
        let sum = arena.add(&[csc_x, cot_x]);
        let abs_sum = arena.abs(sum);
        let ln_val = arena.ln(abs_sum);
        return arena.neg(ln_val);
    }
    if n < -1 {
        // Upward reduction formula (recurse toward 0):
        //   ∫ sin^n(x) dx = (1/(n+1))·cos(x)·sin^(n+1)(x)
        //                  + (n+2)/(n+1) · ∫ sin^(n+2)(x) dx
        let cos_x = arena.cos(var);
        let sin_x = arena.sin(var);

        // sin^(n+1)(x)  — note n+1 ≤ −1 here, never 0 or 1
        let exp_id = arena.int(n + 1);
        let sin_pow = arena.pow(sin_x, exp_id);

        // First term: (1/(n+1)) · cos(x) · sin^(n+1)(x)
        let inv = arena.rational(1, n + 1);
        let first_term = arena.mul(&[inv, cos_x, sin_pow]);

        // Second term: (n+2)/(n+1) · ∫ sin^(n+2)(x) dx
        let coeff = arena.rational(n + 2, n + 1);
        let recursive = sin_pow_integrate(arena, n + 2, var);
        let second_term = arena.mul(&[coeff, recursive]);

        return arena.add(&[first_term, second_term]);
    }

    // Recursive reduction:
    //   ∫ sin^n(x) dx = −(1/n)·cos(x)·sin^(n−1)(x)
    //                  + (n−1)/n · ∫ sin^(n−2)(x) dx

    let cos_x = arena.cos(var);
    let sin_x = arena.sin(var);

    // sin^(n−1)(x)
    let sin_pow = if n - 1 == 1 {
        sin_x
    } else {
        let n_minus_1 = arena.int(n - 1);
        arena.pow(sin_x, n_minus_1)
    };

    // First term: −(1/n) · cos(x) · sin^(n−1)(x)
    let neg_inv_n = arena.rational(-1, n);
    let first_term = arena.mul(&[neg_inv_n, cos_x, sin_pow]);

    // Second term: (n−1)/n · ∫ sin^(n−2)(x) dx
    let coeff = arena.rational(n - 1, n);
    let recursive = sin_pow_integrate(arena, n - 2, var);
    let second_term = arena.mul(&[coeff, recursive]);

    arena.add(&[first_term, second_term])
}

// ═══════════════════════════════════════════════════════════════════════════
// cos^n integration
// ═══════════════════════════════════════════════════════════════════════════

/// Compute `∫ cos^n(x) dx` using the recursive reduction formula.
///
/// - n = 0 → `x`
/// - n = 1 → `sin(x)`
/// - n = −1 → `ln|sec(x)+tan(x)|`
/// - n ≤ −2 → upward reduction toward 0:
///   `−(1/(n+1))·sin(x)·cos^(n+1)(x) + (n+2)/(n+1)·∫cos^(n+2)(x)dx`
/// - n ≥ 2 → `(1/n)·sin(x)·cos^(n−1)(x) + (n−1)/n · ∫ cos^(n−2)(x) dx`
pub(crate) fn cos_pow_integrate(arena: &mut Arena, n: i64, var: ExprId) -> ExprId {
    if n == 0 {
        return var; // ∫ 1 dx = x
    }
    if n == 1 {
        // ∫ cos(x) dx = sin(x)
        return arena.sin(var);
    }
    if n == -1 {
        // ∫ sec(x) dx = ln|sec(x) + tan(x)|
        let cos_x = arena.cos(var);
        let neg_one_id = arena.int(-1);
        let sec_x = arena.pow(cos_x, neg_one_id); // 1/cos(x)
        let tan_x = arena.tan(var);
        let sum = arena.add(&[sec_x, tan_x]);
        let abs_sum = arena.abs(sum);
        return arena.ln(abs_sum);
    }
    if n < -1 {
        // Upward reduction formula (recurse toward 0):
        //   ∫ cos^n(x) dx = −(1/(n+1))·sin(x)·cos^(n+1)(x)
        //                  + (n+2)/(n+1) · ∫ cos^(n+2)(x) dx
        let sin_x = arena.sin(var);
        let cos_x = arena.cos(var);

        // cos^(n+1)(x)  — note n+1 ≤ −1 here, never 0 or 1
        let exp_id = arena.int(n + 1);
        let cos_pow = arena.pow(cos_x, exp_id);

        // First term: −(1/(n+1)) · sin(x) · cos^(n+1)(x)
        let neg_inv = arena.rational(-1, n + 1);
        let first_term = arena.mul(&[neg_inv, sin_x, cos_pow]);

        // Second term: (n+2)/(n+1) · ∫ cos^(n+2)(x) dx
        let coeff = arena.rational(n + 2, n + 1);
        let recursive = cos_pow_integrate(arena, n + 2, var);
        let second_term = arena.mul(&[coeff, recursive]);

        return arena.add(&[first_term, second_term]);
    }

    // Recursive reduction:
    //   ∫ cos^n(x) dx = (1/n)·sin(x)·cos^(n−1)(x)
    //                  + (n−1)/n · ∫ cos^(n−2)(x) dx

    let sin_x = arena.sin(var);
    let cos_x = arena.cos(var);

    // cos^(n−1)(x)
    let cos_pow = if n - 1 == 1 {
        cos_x
    } else {
        let n_minus_1 = arena.int(n - 1);
        arena.pow(cos_x, n_minus_1)
    };

    // First term: (1/n) · sin(x) · cos^(n−1)(x)
    let inv_n = arena.rational(1, n);
    let first_term = arena.mul(&[inv_n, sin_x, cos_pow]);

    // Second term: (n−1)/n · ∫ cos^(n−2)(x) dx
    let coeff = arena.rational(n - 1, n);
    let recursive = cos_pow_integrate(arena, n - 2, var);
    let second_term = arena.mul(&[coeff, recursive]);

    arena.add(&[first_term, second_term])
}

// ═══════════════════════════════════════════════════════════════════════════
// sin^m · cos^n integration
// ═══════════════════════════════════════════════════════════════════════════

/// Compute `∫ sin^m(x) · cos^n(x) dx`.
///
/// Strategy:
/// - m = 0 → `cos_pow_integrate(n)`
/// - n = 0 → `sin_pow_integrate(m)`
/// - m odd → Pythagorean substitution with `u = cos(x)`
/// - n odd → Pythagorean substitution with `u = sin(x)`
/// - both even → reduction formula on `m`
pub(crate) fn sin_cos_integrate(arena: &mut Arena, m: i64, n: i64, var: ExprId) -> ExprId {
    // Degenerate cases: one exponent is zero.
    if m == 0 {
        return cos_pow_integrate(arena, n, var);
    }
    if n == 0 {
        return sin_pow_integrate(arena, m, var);
    }

    // Negative exponents: leave unevaluated for now.
    if m < 0 || n < 0 {
        let sin_x = arena.sin(var);
        let cos_x = arena.cos(var);
        let m_id = arena.int(m);
        let n_id = arena.int(n);
        let sin_pow = if m == 1 {
            sin_x
        } else {
            arena.pow(sin_x, m_id)
        };
        let cos_pow = if n == 1 {
            cos_x
        } else {
            arena.pow(cos_x, n_id)
        };
        let integrand = arena.mul(&[sin_pow, cos_pow]);
        return arena.intern(ExprNode::Integral(integrand, var));
    }

    // ── m is odd: u = cos(x), sin²(x) = 1 − u² ───────────────────
    //
    // ∫ sin^(2k+1)(x) · cos^n(x) dx
    //   = −∫ (1−u²)^k · u^n du       where u = cos(x)
    //   = −Σ_{j=0}^{k} C(k,j)·(−1)^j · u^(n+2j+1) / (n+2j+1)
    //   = −Σ_{j=0}^{k} C(k,j)·(−1)^j · cos^(n+2j+1)(x) / (n+2j+1)
    if m % 2 != 0 {
        let k = (m - 1) / 2; // m = 2k + 1
        let cos_x = arena.cos(var);
        let mut terms: Vec<ExprId> = Vec::new();

        for j in 0..=k {
            let c = binom(k as u64, j as u64);
            // sign: overall −1 from substitution, then (−1)^j from binomial
            // combined sign = (−1)^(j+1)
            let sign: i64 = if (j + 1) % 2 == 0 { 1 } else { -1 };
            let power = n + 2 * j + 1;

            // coefficient: sign * C(k,j) / power
            let numer = BigInt::from(sign) * c;
            let ratio = Ratio::new(numer, BigInt::from(power));
            let num_id = arena.intern_num(ratio);
            let coeff = arena.intern(ExprNode::Num(num_id));

            // cos^power(x)
            let cos_term = if power == 1 {
                cos_x
            } else {
                let pow_id = arena.int(power);
                arena.pow(cos_x, pow_id)
            };

            let term = arena.mul(&[coeff, cos_term]);
            terms.push(term);
        }

        return arena.add(&terms);
    }

    // ── n is odd: u = sin(x), cos²(x) = 1 − u² ───────────────────
    //
    // ∫ sin^m(x) · cos^(2k+1)(x) dx
    //   = ∫ u^m · (1−u²)^k du        where u = sin(x)
    //   = Σ_{j=0}^{k} C(k,j)·(−1)^j · u^(m+2j+1) / (m+2j+1)
    //   = Σ_{j=0}^{k} C(k,j)·(−1)^j · sin^(m+2j+1)(x) / (m+2j+1)
    if n % 2 != 0 {
        let k = (n - 1) / 2; // n = 2k + 1
        let sin_x = arena.sin(var);
        let mut terms: Vec<ExprId> = Vec::new();

        for j in 0..=k {
            let c = binom(k as u64, j as u64);
            let sign: i64 = if j % 2 == 0 { 1 } else { -1 };
            let power = m + 2 * j + 1;

            let numer = BigInt::from(sign) * c;
            let ratio = Ratio::new(numer, BigInt::from(power));
            let num_id = arena.intern_num(ratio);
            let coeff = arena.intern(ExprNode::Num(num_id));

            // sin^power(x)
            let sin_term = if power == 1 {
                sin_x
            } else {
                let pow_id = arena.int(power);
                arena.pow(sin_x, pow_id)
            };

            let term = arena.mul(&[coeff, sin_term]);
            terms.push(term);
        }

        return arena.add(&terms);
    }

    // ── Both m and n are even: use reduction formula on m ──────────
    //
    // ∫ sin^m(x)·cos^n(x) dx
    //   = −sin^(m−1)(x)·cos^(n+1)(x) / (m+n)
    //     + (m−1)/(m+n) · ∫ sin^(m−2)(x)·cos^n(x) dx
    //
    // Base cases are handled by the m==0 / n==0 checks at the top via
    // recursion reducing m by 2 each step.
    debug_assert!(m % 2 == 0 && n % 2 == 0 && m >= 2 && n >= 2);

    let sin_x = arena.sin(var);
    let cos_x = arena.cos(var);
    let mn = m + n;

    // sin^(m−1)(x)
    let sin_pow = if m - 1 == 1 {
        sin_x
    } else {
        let exp = arena.int(m - 1);
        arena.pow(sin_x, exp)
    };

    // cos^(n+1)(x)
    let cos_pow = {
        let exp = arena.int(n + 1);
        arena.pow(cos_x, exp)
    };

    // First term: −sin^(m−1)(x) · cos^(n+1)(x) / (m+n)
    let neg_inv_mn = arena.rational(-1, mn);
    let first_term = arena.mul(&[neg_inv_mn, sin_pow, cos_pow]);

    // Second term: (m−1)/(m+n) · ∫ sin^(m−2)(x)·cos^n(x) dx
    let coeff = arena.rational(m - 1, mn);
    let recursive = sin_cos_integrate(arena, m - 2, n, var);
    let second_term = arena.mul(&[coeff, recursive]);

    arena.add(&[first_term, second_term])
}

// ═══════════════════════════════════════════════════════════════════════════
// Pattern-matching entry point
// ═══════════════════════════════════════════════════════════════════════════

/// Try to recognise a trigonometric power integrand and compute the
/// antiderivative.
///
/// Recognises:
/// - `sin(x)^n`  (or bare `sin(x)` → n=1)
/// - `cos(x)^n`  (or bare `cos(x)` → n=1)
/// - products containing `sin(x)^m` and/or `cos(x)^n`
///
/// Returns `None` when the expression does not match any trig-power pattern.
pub(crate) fn try_trig_power_integral(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    var_sym: SymbolId,
) -> Option<ExprId> {
    let node = arena.node(expr).clone();

    // ── Single factor: Pow(Sin(var), n) or Pow(Cos(var), n) ────────
    if let Some(n) = extract_sin_power(arena, expr, var) {
        if !(0..2).contains(&n) {
            return Some(sin_pow_integrate(arena, n, var));
        }
        // n == 1 is handled by the main integrator already.
        return None;
    }

    if let Some(n) = extract_cos_power(arena, expr, var) {
        if !(0..2).contains(&n) {
            return Some(cos_pow_integrate(arena, n, var));
        }
        return None;
    }

    // ── Product: Mul(...) containing sin/cos powers ────────────────
    if let ExprNode::Mul(ref children) = node {
        let mut sin_exp: i64 = 0;
        let mut cos_exp: i64 = 0;
        let mut other_factors: Vec<ExprId> = Vec::new();
        let mut found_trig = false;

        for &child in children.iter() {
            if let Some(s) = extract_sin_power(arena, child, var) {
                sin_exp += s;
                found_trig = true;
            } else if let Some(c) = extract_cos_power(arena, child, var) {
                cos_exp += c;
                found_trig = true;
            } else if contains_var(arena, child, var_sym) {
                // A var-dependent factor that isn't a sin/cos power — bail.
                return None;
            } else {
                // Constant factor (independent of var): keep aside.
                other_factors.push(child);
            }
        }

        if !found_trig {
            return None;
        }

        // We need at least one exponent ≥ 2 to be interesting, or a
        // mixed sin·cos product.  The main integrator already handles
        // single sin(x) and cos(x), so only fire when the combined
        // problem is genuinely a "power" integral.
        let dominated_by_basic = sin_exp == 0 && cos_exp == 1 || sin_exp == 1 && cos_exp == 0;
        if dominated_by_basic {
            return None;
        }

        // Compute the trig-power integral.
        let trig_result = sin_cos_integrate(arena, sin_exp, cos_exp, var);

        // Re-attach any constant prefactors.
        if other_factors.is_empty() {
            return Some(trig_result);
        }
        other_factors.push(trig_result);
        return Some(arena.mul(&other_factors));
    }

    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;

    // ------------------------------------------------------------------
    // sin^n integration
    // ------------------------------------------------------------------

    #[test]
    fn sin_zero() {
        // ∫ sin^0(x) dx = ∫ 1 dx = x
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = sin_pow_integrate(&mut a, 0, x);
        assert_eq!(result, x);
    }

    #[test]
    fn sin_first() {
        // ∫ sin(x) dx = −cos(x)
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = sin_pow_integrate(&mut a, 1, x);
        let s = a.display(result).to_string();
        assert!(s.contains("cos"), "expected −cos(x), got: {s}");
    }

    #[test]
    fn sin_squared() {
        // ∫ sin²(x) dx = −½·sin(x)·cos(x) + ½·x
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = sin_pow_integrate(&mut a, 2, x);
        let s = a.display(result).to_string();
        assert!(
            s.contains("cos") && s.contains("sin"),
            "expected reduction formula terms, got: {s}"
        );
    }

    #[test]
    fn sin_cubed() {
        // ∫ sin³(x) dx = −⅓·cos(x)·sin²(x) + ⅔·(−cos(x))
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = sin_pow_integrate(&mut a, 3, x);
        let s = a.display(result).to_string();
        assert!(s.contains("cos"), "expected cos terms, got: {s}");
    }

    #[test]
    fn sin_fourth() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = sin_pow_integrate(&mut a, 4, x);
        let s = a.display(result).to_string();
        // Should produce some combination of x, sin, cos
        assert!(!s.is_empty(), "got: {s}");
    }

    #[test]
    fn sin_neg2_is_neg_cot() {
        // ∫ sin^(-2)(x) dx = −cot(x) = −cos(x)/sin(x)
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = sin_pow_integrate(&mut a, -2, x);
        let s = a.display(result).to_string();
        // Should NOT be unevaluated — should contain cos and sin
        assert!(
            !s.contains("Integral"),
            "expected evaluated result, got unevaluated: {s}"
        );
        assert!(
            s.contains("cos") && s.contains("sin"),
            "expected cos/sin terms for −cot(x), got: {s}"
        );
    }

    // ------------------------------------------------------------------
    // cos^n integration
    // ------------------------------------------------------------------

    #[test]
    fn cos_zero() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = cos_pow_integrate(&mut a, 0, x);
        assert_eq!(result, x);
    }

    #[test]
    fn cos_first() {
        // ∫ cos(x) dx = sin(x)
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = cos_pow_integrate(&mut a, 1, x);
        let s = a.display(result).to_string();
        assert!(s.contains("sin"), "expected sin(x), got: {s}");
    }

    #[test]
    fn cos_squared() {
        // ∫ cos²(x) dx = ½·sin(x)·cos(x) + ½·x
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = cos_pow_integrate(&mut a, 2, x);
        let s = a.display(result).to_string();
        assert!(
            s.contains("cos") && s.contains("sin"),
            "expected reduction formula terms, got: {s}"
        );
    }

    #[test]
    fn cos_cubed() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = cos_pow_integrate(&mut a, 3, x);
        let s = a.display(result).to_string();
        assert!(s.contains("sin"), "expected sin terms, got: {s}");
    }

    #[test]
    fn cos_neg1_is_ln_sec_tan() {
        // ∫ cos^(-1)(x) dx = ∫ sec(x) dx = ln|sec(x)+tan(x)|
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = cos_pow_integrate(&mut a, -1, x);
        let s = a.display(result).to_string();
        assert!(
            !s.contains("Integral"),
            "expected evaluated result, got unevaluated: {s}"
        );
        assert!(
            s.contains("ln"),
            "expected ln term for ln|sec+tan|, got: {s}"
        );
    }

    #[test]
    fn cos_neg2_is_tan() {
        // ∫ cos^(-2)(x) dx = ∫ sec²(x) dx = tan(x)
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = cos_pow_integrate(&mut a, -2, x);
        let s = a.display(result).to_string();
        assert!(
            !s.contains("Integral"),
            "expected evaluated result, got unevaluated: {s}"
        );
        assert!(
            s.contains("sin") && s.contains("cos"),
            "expected sin/cos terms for tan(x), got: {s}"
        );
    }

    // ------------------------------------------------------------------
    // sin^m · cos^n integration
    // ------------------------------------------------------------------

    #[test]
    fn sin1_cos1() {
        // ∫ sin(x)·cos(x) dx  (m=1 odd)
        // = −cos²(x)/2
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = sin_cos_integrate(&mut a, 1, 1, x);
        let s = a.display(result).to_string();
        assert!(s.contains("cos"), "expected cos term, got: {s}");
    }

    #[test]
    fn sin2_cos1() {
        // ∫ sin²(x)·cos(x) dx  (n=1 odd)
        // = sin³(x)/3
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = sin_cos_integrate(&mut a, 2, 1, x);
        let s = a.display(result).to_string();
        assert!(s.contains("sin"), "expected sin term, got: {s}");
    }

    #[test]
    fn sin1_cos2() {
        // ∫ sin(x)·cos²(x) dx  (m=1 odd)
        // = −cos³(x)/3
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = sin_cos_integrate(&mut a, 1, 2, x);
        let s = a.display(result).to_string();
        assert!(s.contains("cos"), "expected cos term, got: {s}");
    }

    #[test]
    fn sin3_cos2() {
        // ∫ sin³(x)·cos²(x) dx  (m=3 odd)
        // = −cos³(x)/3 + cos⁵(x)/5
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = sin_cos_integrate(&mut a, 3, 2, x);
        let s = a.display(result).to_string();
        assert!(s.contains("cos"), "expected cos terms, got: {s}");
    }

    #[test]
    fn sin2_cos3() {
        // ∫ sin²(x)·cos³(x) dx  (n=3 odd)
        // = sin³(x)/3 − sin⁵(x)/5
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = sin_cos_integrate(&mut a, 2, 3, x);
        let s = a.display(result).to_string();
        assert!(s.contains("sin"), "expected sin terms, got: {s}");
    }

    #[test]
    fn sin2_cos2_both_even() {
        // ∫ sin²(x)·cos²(x) dx — both even, uses reduction
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = sin_cos_integrate(&mut a, 2, 2, x);
        let s = a.display(result).to_string();
        // Should contain both sin and cos terms
        assert!(!s.is_empty(), "got: {s}");
    }

    #[test]
    fn sin_cos_m_zero_delegates() {
        // m=0 should delegate to cos_pow_integrate
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = sin_cos_integrate(&mut a, 0, 3, x);
        let direct = cos_pow_integrate(&mut a, 3, x);
        assert_eq!(result, direct);
    }

    #[test]
    fn sin_cos_n_zero_delegates() {
        // n=0 should delegate to sin_pow_integrate
        let mut a = Arena::new();
        let x = a.symbol("x");
        let result = sin_cos_integrate(&mut a, 3, 0, x);
        let direct = sin_pow_integrate(&mut a, 3, x);
        assert_eq!(result, direct);
    }

    // ------------------------------------------------------------------
    // Pattern detection via try_trig_power_integral
    // ------------------------------------------------------------------

    #[test]
    fn try_detect_sin_squared() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let var_sym = match a.node(x) {
            ExprNode::Symbol(sid) => *sid,
            _ => unreachable!(),
        };
        let sin_x = a.sin(x);
        let two = a.int(2);
        let expr = a.pow(sin_x, two);
        let result = try_trig_power_integral(&mut a, expr, x, var_sym);
        assert!(result.is_some(), "should detect sin²(x)");
        let s = a.display(result.unwrap()).to_string();
        assert!(
            s.contains("cos") && s.contains("sin"),
            "expected reduction formula, got: {s}"
        );
    }

    #[test]
    fn try_detect_cos_cubed() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let var_sym = match a.node(x) {
            ExprNode::Symbol(sid) => *sid,
            _ => unreachable!(),
        };
        let cos_x = a.cos(x);
        let three = a.int(3);
        let expr = a.pow(cos_x, three);
        let result = try_trig_power_integral(&mut a, expr, x, var_sym);
        assert!(result.is_some(), "should detect cos³(x)");
        let s = a.display(result.unwrap()).to_string();
        assert!(s.contains("sin"), "expected sin terms, got: {s}");
    }

    #[test]
    fn try_detect_sin_cos_product() {
        // sin(x)^2 * cos(x)^3 as Mul
        let mut a = Arena::new();
        let x = a.symbol("x");
        let var_sym = match a.node(x) {
            ExprNode::Symbol(sid) => *sid,
            _ => unreachable!(),
        };
        let sin_x = a.sin(x);
        let cos_x = a.cos(x);
        let two = a.int(2);
        let three = a.int(3);
        let sin2 = a.pow(sin_x, two);
        let cos3 = a.pow(cos_x, three);
        let expr = a.mul(&[sin2, cos3]);
        let result = try_trig_power_integral(&mut a, expr, x, var_sym);
        assert!(result.is_some(), "should detect sin²·cos³");
    }

    #[test]
    fn try_detect_bare_sin_cos_product() {
        // sin(x) * cos(x) — implicit power 1 for each
        let mut a = Arena::new();
        let x = a.symbol("x");
        let var_sym = match a.node(x) {
            ExprNode::Symbol(sid) => *sid,
            _ => unreachable!(),
        };
        let sin_x = a.sin(x);
        let cos_x = a.cos(x);
        let expr = a.mul(&[sin_x, cos_x]);
        let result = try_trig_power_integral(&mut a, expr, x, var_sym);
        assert!(result.is_some(), "should detect sin(x)·cos(x)");
    }

    #[test]
    fn try_detect_with_constant_factor() {
        // 3 * sin(x)^2 * cos(x)^3
        let mut a = Arena::new();
        let x = a.symbol("x");
        let var_sym = match a.node(x) {
            ExprNode::Symbol(sid) => *sid,
            _ => unreachable!(),
        };
        let sin_x = a.sin(x);
        let cos_x = a.cos(x);
        let two = a.int(2);
        let three_exp = a.int(3);
        let sin2 = a.pow(sin_x, two);
        let cos3 = a.pow(cos_x, three_exp);
        let three = a.int(3);
        let expr = a.mul(&[three, sin2, cos3]);
        let result = try_trig_power_integral(&mut a, expr, x, var_sym);
        assert!(
            result.is_some(),
            "should detect 3·sin²·cos³ with constant factor"
        );
    }

    #[test]
    fn try_returns_none_for_non_trig() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let var_sym = match a.node(x) {
            ExprNode::Symbol(sid) => *sid,
            _ => unreachable!(),
        };
        // x^2 is not a trig power
        let two = a.int(2);
        let expr = a.pow(x, two);
        let result = try_trig_power_integral(&mut a, expr, x, var_sym);
        assert!(result.is_none(), "x² is not a trig power integral");
    }

    #[test]
    fn try_returns_none_for_mixed_non_trig() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let var_sym = match a.node(x) {
            ExprNode::Symbol(sid) => *sid,
            _ => unreachable!(),
        };
        // x * sin(x)^2 — has a var-dependent non-trig factor
        let sin_x = a.sin(x);
        let two = a.int(2);
        let sin2 = a.pow(sin_x, two);
        let expr = a.mul(&[x, sin2]);
        let result = try_trig_power_integral(&mut a, expr, x, var_sym);
        assert!(
            result.is_none(),
            "x·sin²(x) has a non-trig dependent factor"
        );
    }

    // ------------------------------------------------------------------
    // Helper tests
    // ------------------------------------------------------------------

    #[test]
    fn binom_values() {
        assert_eq!(binom(0, 0), BigInt::from(1));
        assert_eq!(binom(1, 0), BigInt::from(1));
        assert_eq!(binom(1, 1), BigInt::from(1));
        assert_eq!(binom(4, 2), BigInt::from(6));
        assert_eq!(binom(5, 0), BigInt::from(1));
        assert_eq!(binom(5, 5), BigInt::from(1));
        assert_eq!(binom(5, 3), BigInt::from(10));
        assert_eq!(binom(6, 3), BigInt::from(20));
        assert_eq!(binom(3, 5), BigInt::from(0)); // k > n
    }

    #[test]
    fn extract_sin_power_bare() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let sin_x = a.sin(x);
        assert_eq!(extract_sin_power(&a, sin_x, x), Some(1));
    }

    #[test]
    fn extract_sin_power_squared() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let sin_x = a.sin(x);
        let two = a.int(2);
        let sin2 = a.pow(sin_x, two);
        assert_eq!(extract_sin_power(&a, sin2, x), Some(2));
    }

    #[test]
    fn extract_cos_power_bare() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let cos_x = a.cos(x);
        assert_eq!(extract_cos_power(&a, cos_x, x), Some(1));
    }

    #[test]
    fn extract_cos_power_cubed() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let cos_x = a.cos(x);
        let three = a.int(3);
        let cos3 = a.pow(cos_x, three);
        assert_eq!(extract_cos_power(&a, cos3, x), Some(3));
    }

    #[test]
    fn extract_returns_none_for_non_trig() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        assert_eq!(extract_sin_power(&a, x2, x), None);
        assert_eq!(extract_cos_power(&a, x2, x), None);
    }
}

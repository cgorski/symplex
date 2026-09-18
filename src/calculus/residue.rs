//! Compute residues of functions at poles of any order.
//!
//! The residue of `f(x)` at `x = a` is the coefficient of `1/(x−a)` in the
//! Laurent expansion of `f` around `a`.
//!
//! # Algorithm
//!
//! 1. Shift: `g(t) = f(a + t)` and split `g = N(t)/D(t)`.
//! 2. If `D` is a polynomial in `t`, its multiplicity `m` at `t = 0` is the
//!    index of the first non-zero coefficient.  With `D̃(t) = D(t)/tᵐ`
//!    (exact, non-vanishing at 0), the residue is the Taylor coefficient
//!    `Res = 1/(m−1)! · dᵐ⁻¹/dtᵐ⁻¹ [N(t)/D̃(t)]` at `t = 0`, which is a plain
//!    substitution — no limits needed.  This handles algebraic points such
//!    as `a = i` where polynomial cancellation over ℚ is unavailable.
//! 3. Otherwise (transcendental denominators such as `1/sin t`), the order
//!    is probed with `lim tᵏ g(t)` for `k = 1, …, 6` and the same derivative
//!    formula is evaluated as a limit.
//! 4. If `N(0)` is not finite (essential singularity like `e^{1/t}`) or
//!    nothing above succeeds, an error is returned and the caller keeps a
//!    formal `Residue` node.

use num_bigint::BigInt;
use num_traits::One;

use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;
use crate::calculus::calculus_util;
use crate::transforms::{diff, eval, subs};

/// Maximum pole order probed by the limit-based fallback.
const MAX_PROBE_ORDER: usize = 6;

fn failed(reason: impl Into<String>) -> SymplexError {
    SymplexError::ComputationFailed {
        operation: "residue",
        reason: reason.into(),
    }
}

/// Compute the residue of `expr` at `var = point`.
///
/// See the module documentation for the algorithm.  Returns `Err` when the
/// residue cannot be determined (essential singularity, unknown pole
/// order, or a non-symbol `var`).
pub(crate) fn residue(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Result<ExprId, SymplexError> {
    if !matches!(arena.node(var), ExprNode::Symbol(_)) {
        return Err(failed("residue variable must be a symbol"));
    }
    if point == arena.infinity() {
        return residue_at_infinity(arena, expr, var);
    }
    let result = residue_inner(arena, expr, var, point)?;
    // The limit engine can leak internal dummy symbols into a half-finished
    // result; a residue may only mention symbols of the input.
    let mut allowed: Vec<ExprId> = walk::free_symbols(arena, expr);
    allowed.extend(walk::free_symbols(arena, point));
    if walk::free_symbols(arena, result)
        .iter()
        .any(|s| !allowed.contains(s))
    {
        return Err(failed("limit engine returned an incomplete result"));
    }
    Ok(result)
}

/// [`residue`] without the final symbol-leak check.
fn residue_inner(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Result<ExprId, SymplexError> {
    // Shift the pole to the origin: g(t) = f(point + t).
    let t = arena.symbol("__res_t");
    let shifted_var = arena.add(&[point, t]);
    let g = subs::subs(arena, expr, var, shifted_var);
    let g = eval::eval(arena, g);

    if !walk::contains(arena, g, t) {
        // Constant in t: analytic, residue 0 (unless it is itself non-finite).
        return Ok(arena.zero());
    }

    // Clear nested fractions (e.g. from f(1/t)) so that the denominator is a
    // genuine polynomial whenever f is rational.
    let g = crate::transforms::integrate::clear_nested_fractions(arena, g, t);
    let (numer, denom) = crate::poly::polybridge::as_numer_denom(arena, g);

    if let Some(coeffs) = calculus_util::poly_coeffs_symbolic(arena, denom, t) {
        return residue_polynomial_denominator(arena, numer, &coeffs, t);
    }

    residue_by_limits(arena, g, t)
}

/// Residue when the shifted denominator is a polynomial in `t` with the
/// given coefficients.
fn residue_polynomial_denominator(
    arena: &mut Arena,
    numer: ExprId,
    coeffs: &[ExprId],
    t: ExprId,
) -> Result<ExprId, SymplexError> {
    // Multiplicity of t = 0: index of the first coefficient that is not
    // provably zero.
    let mut m = coeffs.len();
    for (k, &c) in coeffs.iter().enumerate() {
        if !is_provably_zero(arena, c) {
            m = k;
            break;
        }
    }
    if m == coeffs.len() {
        return Err(failed("denominator is identically zero"));
    }

    // Numerator must be analytic at t = 0.
    let n0 = subs::subs(arena, numer, t, arena.zero());
    let n0 = eval::eval(arena, n0);
    if !is_finite_constant(arena, n0, t) {
        return Err(failed(
            "numerator is not analytic at the point (essential singularity?)",
        ));
    }

    if m == 0 {
        // No pole: the function is analytic here.
        return Ok(arena.zero());
    }

    // D̃(t) = Σ_{k ≥ m} c_k t^{k−m}
    let mut terms: Vec<ExprId> = Vec::with_capacity(coeffs.len() - m);
    for (k, &c) in coeffs.iter().enumerate().skip(m) {
        let e = arena.int((k - m) as i64);
        let tp = arena.pow(t, e);
        terms.push(arena.mul(&[c, tp]));
    }
    let d_tilde = arena.add(&terms);

    // h(t) = N(t)/D̃(t); Res = h^{(m−1)}(0) / (m−1)!
    let mut h = arena.div(numer, d_tilde);
    for _ in 1..m {
        h = diff::diff(arena, h, t);
    }
    let h0 = subs::subs(arena, h, t, arena.zero());
    let h0 = eval::eval(arena, h0);
    if !is_finite_constant(arena, h0, t) {
        return Err(failed("could not evaluate the Laurent coefficient"));
    }
    let res = divide_by_factorial(arena, h0, m - 1);
    Ok(eval::eval(arena, res))
}

/// Limit-based fallback for non-polynomial denominators.
fn residue_by_limits(arena: &mut Arena, g: ExprId, t: ExprId) -> Result<ExprId, SymplexError> {
    let zero = arena.zero();
    for m in 1..=MAX_PROBE_ORDER {
        let e = arena.int(m as i64);
        let tm = arena.pow(t, e);
        let scaled = arena.mul(&[tm, g]);
        let scaled = eval::eval(arena, scaled);
        let Ok(l) = crate::calculus::limit::limit(arena, scaled, t, zero) else {
            continue;
        };
        if !is_finite_constant(arena, l, t) {
            continue;
        }
        if m == 1 {
            return Ok(eval::eval(arena, l));
        }
        // Order ≤ m established: Res = lim d^{m−1}/dt^{m−1}[tᵐ g] / (m−1)!
        let mut h = scaled;
        for _ in 1..m {
            h = diff::diff(arena, h, t);
        }
        let h = eval::eval(arena, h);
        let l2 = crate::calculus::limit::limit(arena, h, t, zero)?;
        if !is_finite_constant(arena, l2, t) {
            return Err(failed("Laurent coefficient limit is not finite"));
        }
        let res = divide_by_factorial(arena, l2, m - 1);
        return Ok(eval::eval(arena, res));
    }
    Err(failed(
        "could not determine the pole order (essential singularity or order > 6)",
    ))
}

/// Residue at infinity: `Res_{z=∞} f = −Res_{t=0} f(1/t)/t²`.
pub(crate) fn residue_at_infinity(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
) -> Result<ExprId, SymplexError> {
    if !matches!(arena.node(var), ExprNode::Symbol(_)) {
        return Err(failed("residue variable must be a symbol"));
    }
    let t = arena.symbol("__res_inf_t");
    let one = arena.one();
    let inv_t = arena.div(one, t);
    let f_inv = subs::subs(arena, expr, var, inv_t);
    let neg_two = arena.int(-2);
    let t_m2 = arena.pow(t, neg_two);
    let h = arena.mul(&[f_inv, t_m2]);
    let h = eval::eval(arena, h);
    let zero = arena.zero();
    let r = residue(arena, h, t, zero)?;
    let neg = arena.neg(r);
    Ok(eval::eval(arena, neg))
}

/// `v / n!`
fn divide_by_factorial(arena: &mut Arena, v: ExprId, n: usize) -> ExprId {
    let mut fact = BigInt::one();
    for k in 2..=n {
        fact *= BigInt::from(k);
    }
    let f = arena.big_int(fact);
    arena.div(v, f)
}

/// Is `c` provably zero (structurally after `eval`, after full
/// simplification, or numerically when constant)?
fn is_provably_zero(arena: &mut Arena, c: ExprId) -> bool {
    let c = eval::eval(arena, c);
    if arena.is_zero_structural(c) {
        return true;
    }
    let expanded = crate::transforms::expand::expand(arena, c);
    let expanded = eval::eval(arena, expanded);
    if arena.is_zero_structural(expanded) {
        return true;
    }
    if walk::free_symbols(arena, expanded).is_empty() {
        // Constant: decide numerically (real or complex).
        if let Ok(s) = crate::transforms::evalf::evalf(arena, expanded, 20) {
            let s = s.trim();
            if s == "0" {
                return true;
            }
            if let Ok(v) = s.parse::<f64>() {
                return v.abs() < 1e-18;
            }
            return false;
        }
    }
    let opts = crate::simplify::simplify_engine::SimplifyOpts::default();
    let r = crate::simplify::simplify_engine::unified_simplify(arena, expanded, &opts);
    arena.is_zero_structural(r.expr)
}

/// Is `v` a finite value free of `t`, infinities and formal nodes?
fn is_finite_constant(arena: &mut Arena, v: ExprId, t: ExprId) -> bool {
    if walk::contains(arena, v, t) || walk::has_unevaluated(arena, v) {
        return false;
    }
    let mut stack = vec![v];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Infinity
            | ExprNode::NegInfinity
            | ExprNode::ComplexInfinity
            | ExprNode::NaN => {
                return false;
            }
            ExprNode::Ln(inner) if arena.is_zero_structural(*inner) => return false,
            ExprNode::Pow(base, exp)
                if arena.is_zero_structural(*base)
                    && arena
                        .as_num(*exp)
                        .is_some_and(num_traits::Signed::is_negative) =>
            {
                return false;
            }
            node => node.for_each_child(|c| stack.push(c)),
        }
    }
    if walk::free_symbols(arena, v).is_empty() && arena.as_num(v).is_none() {
        // Constant: make sure it evaluates to a finite (possibly complex) number.
        return match crate::transforms::evalf::evalf(arena, v, 16) {
            Ok(s) => !s.contains("inf") && !s.contains("nan") && !s.contains("NaN"),
            Err(_) => false,
        };
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn show(a: &Arena, id: ExprId) -> String {
        a.display(id).to_string()
    }

    #[test]
    fn simple_pole() {
        let mut a = Arena::new();
        let z = a.symbol("z");
        let f = a.pow(z, a.neg_one());
        let zero = a.zero();
        let r = residue(&mut a, f, z, zero).unwrap();
        assert_eq!(show(&a, r), "1");
    }

    #[test]
    fn double_pole_exp() {
        let mut a = Arena::new();
        let z = a.symbol("z");
        let m2 = a.int(-2);
        let zm2 = a.pow(z, m2);
        let ez = a.exp(z);
        let f = a.mul(&[zm2, ez]);
        let zero = a.zero();
        let r = residue(&mut a, f, z, zero).unwrap();
        assert_eq!(show(&a, r), "1");
    }

    #[test]
    fn triple_pole_zero_residue() {
        let mut a = Arena::new();
        let z = a.symbol("z");
        let one = a.one();
        let zm1 = a.sub(z, one);
        let m3 = a.int(-3);
        let f = a.pow(zm1, m3);
        let r = residue(&mut a, f, z, one).unwrap();
        assert_eq!(show(&a, r), "0");
    }

    #[test]
    fn cos_over_z_cubed() {
        let mut a = Arena::new();
        let z = a.symbol("z");
        let c = a.cos(z);
        let m3 = a.int(-3);
        let zm3 = a.pow(z, m3);
        let f = a.mul(&[c, zm3]);
        let zero = a.zero();
        let r = residue(&mut a, f, z, zero).unwrap();
        assert_eq!(show(&a, r), "-1/2");
    }

    #[test]
    fn double_pole_at_i() {
        let mut a = Arena::new();
        let z = a.symbol("z");
        let two = a.int(2);
        let z2 = a.pow(z, two);
        let one = a.one();
        let d = a.add(&[z2, one]);
        let m2 = a.int(-2);
        let f = a.pow(d, m2);
        let i = a.i_unit();
        let r = residue(&mut a, f, z, i).unwrap();
        // −i/4
        let s = show(&a, r);
        assert!(s == "-1/4*I" || s == "-1/4*i", "{s}");
    }

    #[test]
    fn analytic_point_is_zero() {
        let mut a = Arena::new();
        let z = a.symbol("z");
        let two = a.int(2);
        let f = a.pow(z, two);
        let zero = a.zero();
        let r = residue(&mut a, f, z, zero).unwrap();
        assert_eq!(show(&a, r), "0");
    }

    #[test]
    fn essential_singularity_fails() {
        let mut a = Arena::new();
        let z = a.symbol("z");
        let inv = a.pow(z, a.neg_one());
        let f = a.exp(inv);
        let zero = a.zero();
        assert!(residue(&mut a, f, z, zero).is_err());
    }

    #[test]
    fn transcendental_denominator() {
        let mut a = Arena::new();
        let z = a.symbol("z");
        let s = a.sin(z);
        let f = a.pow(s, a.neg_one());
        let zero = a.zero();
        let r = residue(&mut a, f, z, zero).unwrap();
        assert_eq!(show(&a, r), "1");
    }

    #[test]
    fn at_infinity() {
        let mut a = Arena::new();
        let z = a.symbol("z");
        let f = a.pow(z, a.neg_one());
        let r = residue_at_infinity(&mut a, f, z).unwrap();
        assert_eq!(show(&a, r), "-1");
    }
}

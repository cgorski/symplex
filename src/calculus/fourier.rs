//! Fourier series expansion.
//!
//! Computes the Fourier trigonometric series of a function f(x)
//! over an interval [a, b] = [-L, L] (or [-π, π] by default).

use crate::base::arena::Arena;
use crate::base::node::ExprId;

/// `∫_{-π}^{π} g(var) dvar`: the antiderivative's endpoint difference when
/// that is a closed, finite expression; otherwise the definite integrator
/// (which takes endpoint values as one-sided limits), and failing that the
/// unevaluated `DefiniteIntegral` node (which still evaluates numerically).
/// 0.22 always substituted `±π` into the antiderivative: a tan-half-angle
/// antiderivative is `zoo` there, and an unevaluated `Integral(g, var)` has
/// no endpoint value at all — `ln(2 + cos x)` came out as the literal `nan`.
fn integral_over_period(arena: &mut Arena, g: ExprId, var: ExprId) -> ExprId {
    let pi = arena.pi;
    let neg_pi = arena.neg(pi);
    let anti = crate::transforms::integrate::integrate(arena, g, var);
    if !crate::base::walk::has_unevaluated(arena, anti) {
        let at_pi = crate::transforms::subs::subs(arena, anti, var, pi);
        let at_neg_pi = crate::transforms::subs::subs(arena, anti, var, neg_pi);
        let diff = arena.sub(at_pi, at_neg_pi);
        let diff = crate::transforms::eval::eval(arena, diff);
        if !has_non_finite(arena, diff) {
            return diff;
        }
    }
    match crate::calculus::definite::integrate_definite(arena, g, var, neg_pi, pi) {
        Ok(v) if !has_non_finite(arena, v) => v,
        _ => crate::calculus::definite::unevaluated_definite(arena, g, var, neg_pi, pi),
    }
}

/// Does `e` contain `nan`, `zoo` or `±∞` anywhere?
fn has_non_finite(arena: &Arena, e: ExprId) -> bool {
    use crate::base::node::ExprNode;
    crate::base::walk::post_order_ids(arena, e)
        .into_iter()
        .any(|id| {
            matches!(
                arena.node(id),
                ExprNode::NaN
                    | ExprNode::ComplexInfinity
                    | ExprNode::Infinity
                    | ExprNode::NegInfinity
            )
        })
}

/// Compute the Fourier series of `expr` over [-pi, pi] up to `n_terms` harmonics.
///
/// F(x) ≈ a₀/2 + Σ_{n=1}^{N} [aₙ cos(nx) + bₙ sin(nx)]
///
/// where:
/// - aₙ = (1/π) ∫_{-π}^{π} f(x) cos(nx) dx
/// - bₙ = (1/π) ∫_{-π}^{π} f(x) sin(nx) dx
///
/// A coefficient whose integrand has no closed antiderivative is kept as a
/// `DefiniteIntegral` node, so the result reports `has_unevaluated()` and
/// still evaluates numerically.  Beyond [`MAX_TERMS`] harmonics (each costs
/// two integrations) the harmonics stay a formal `Sum` ([`formal_harmonics`]).
///
/// [`MAX_TERMS`]: crate::base::cast::MAX_TERMS
pub(crate) fn fourier_series(arena: &mut Arena, expr: ExprId, var: ExprId, n_terms: u32) -> ExprId {
    let pi = arena.pi;
    let two = arena.int(2);

    // a₀ = (1/π) ∫_{-π}^{π} f(x) dx
    let definite = integral_over_period(arena, expr, var);
    let a0 = arena.div(definite, pi);
    let a0_half = arena.div(a0, two);

    if crate::base::cast::bounded_len(n_terms, crate::base::cast::MAX_TERMS).is_none() {
        let harmonics = formal_harmonics(arena, expr, var, n_terms);
        return arena.add(&[a0_half, harmonics]);
    }

    let mut terms = vec![a0_half];

    for n in 1..=n_terms {
        let n_expr = arena.int(n as i64);
        let nx = arena.mul(&[n_expr, var]);

        // aₙ = (1/π) ∫_{-π}^{π} f(x) cos(nx) dx
        let cos_nx = arena.cos(nx);
        let f_cos = arena.mul(&[expr, cos_nx]);
        let an_num = integral_over_period(arena, f_cos, var);
        let an = arena.div(an_num, pi);

        // bₙ = (1/π) ∫_{-π}^{π} f(x) sin(nx) dx
        let sin_nx = arena.sin(nx);
        let f_sin = arena.mul(&[expr, sin_nx]);
        let bn_num = integral_over_period(arena, f_sin, var);
        let bn = arena.div(bn_num, pi);

        // aₙ cos(nx) + bₙ sin(nx)
        let an_cos = arena.mul(&[an, cos_nx]);
        let bn_sin = arena.mul(&[bn, sin_nx]);
        let term = arena.add(&[an_cos, bn_sin]);
        terms.push(term);
    }

    let result = arena.add(&terms);
    crate::transforms::eval::eval(arena, result)
}

/// `Σ_{n=1}^{N} (aₙ cos(nx) + bₙ sin(nx))` as a formal `Sum` over a dummy
/// harmonic index `_n`, with the coefficients formal integrals over
/// `[−π, π]`: the series [`fourier_series`] does not compute term by term
/// (up to 0.57 `fourier_series(x, x, 2³¹ − 1)` asked for 34 GB for its
/// coefficient lists and aborted the process).
fn formal_harmonics(arena: &mut Arena, expr: ExprId, var: ExprId, n_terms: u32) -> ExprId {
    let pi = arena.pi;
    let neg_pi = arena.neg(pi);
    let n = arena.symbol("_n");
    let nx = arena.mul(&[n, var]);
    let cos_nx = arena.cos(nx);
    let sin_nx = arena.sin(nx);
    let mut parts = Vec::with_capacity(2);
    for trig in [cos_nx, sin_nx] {
        let integrand = arena.mul(&[expr, trig]);
        let integral =
            crate::calculus::definite::unevaluated_definite(arena, integrand, var, neg_pi, pi);
        let coefficient = arena.div(integral, pi);
        parts.push(arena.mul(&[coefficient, trig]));
    }
    let body = arena.add(&parts);
    let one = arena.one;
    let hi = arena.int(i64::from(n_terms));
    arena.make(crate::base::node::ExprNode::Sum(body, n, one, hi))
}

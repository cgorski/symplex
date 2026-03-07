//! Fourier series expansion.
//!
//! Computes the Fourier trigonometric series of a function f(x)
//! over an interval [a, b] = [-L, L] (or [-π, π] by default).

use crate::base::arena::Arena;
use crate::base::node::ExprId;

/// Compute the Fourier series of `expr` over [-pi, pi] up to `n_terms` harmonics.
///
/// F(x) ≈ a₀/2 + Σ_{n=1}^{N} [aₙ cos(nx) + bₙ sin(nx)]
///
/// where:
/// - aₙ = (1/π) ∫_{-π}^{π} f(x) cos(nx) dx
/// - bₙ = (1/π) ∫_{-π}^{π} f(x) sin(nx) dx
pub(crate) fn fourier_series(arena: &mut Arena, expr: ExprId, var: ExprId, n_terms: u32) -> ExprId {
    let pi = arena.pi;
    let neg_pi = arena.neg(pi);
    let two = arena.int(2);

    // a₀ = (1/π) ∫_{-π}^{π} f(x) dx
    let integral_f = crate::transforms::integrate::integrate(arena, expr, var);
    let f_at_pi = crate::transforms::subs::subs(arena, integral_f, var, pi);
    let f_at_neg_pi = crate::transforms::subs::subs(arena, integral_f, var, neg_pi);
    let definite = arena.sub(f_at_pi, f_at_neg_pi);
    let a0 = arena.div(definite, pi);
    let a0_half = arena.div(a0, two);

    let mut terms = vec![a0_half];

    for n in 1..=n_terms {
        let n_expr = arena.int(n as i64);
        let nx = arena.mul(&[n_expr, var]);

        // aₙ = (1/π) ∫_{-π}^{π} f(x) cos(nx) dx
        let cos_nx = arena.cos(nx);
        let f_cos = arena.mul(&[expr, cos_nx]);
        let integral_cos = crate::transforms::integrate::integrate(arena, f_cos, var);
        let at_pi = crate::transforms::subs::subs(arena, integral_cos, var, pi);
        let at_neg_pi = crate::transforms::subs::subs(arena, integral_cos, var, neg_pi);
        let an_num = arena.sub(at_pi, at_neg_pi);
        let an = arena.div(an_num, pi);

        // bₙ = (1/π) ∫_{-π}^{π} f(x) sin(nx) dx
        let sin_nx = arena.sin(nx);
        let f_sin = arena.mul(&[expr, sin_nx]);
        let integral_sin = crate::transforms::integrate::integrate(arena, f_sin, var);
        let at_pi_s = crate::transforms::subs::subs(arena, integral_sin, var, pi);
        let at_neg_pi_s = crate::transforms::subs::subs(arena, integral_sin, var, neg_pi);
        let bn_num = arena.sub(at_pi_s, at_neg_pi_s);
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

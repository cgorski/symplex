//! Infinite sums of hypergeometric terms.
//!
//! `Σ_{k ≥ a} t(k)` is evaluated when the term ratio `t(k+1)/t(k) =
//! C·P(k)/Q(k)` is a constant times a rational function of `k` with
//! rational coefficients, recognised structurally by [`term_ratio`]:
//! products and integer powers of polynomials in `k`, factorials, `Γ` and
//! binomials of arguments linear in `k` with integer slopes, and
//! `c^(m·k + b)` for a constant `c` and an integer `m`.  For a rational `c`
//! the constant is part of `P/Q`; otherwise (`√2^k/k!`, `i^k/k!`) it is
//! recovered numerically from two consecutive terms ([`scaled_series`]).
//!
//! Convergence is classified from the ratio as SymPy's `check_convergence`
//! does (`sympy/core/evalf.py`, BSD-3): with `d = deg Q − deg P`, the terms
//! decay like `1/(k!)^d` when `d > 0` and grow like `(k!)^(−d)` when
//! `d < 0`; when `d = 0`, `g = lc P / lc Q` gives geometric convergence
//! (`|g| < 1`) or divergence (`|g| > 1`), and `|g| = 1` polynomial
//! convergence or divergence according to `p = q₁/q₀ − p₁/p₀`, with
//! `|t(k)| ≈ C·k^(−p)` (computed here from the normalised second
//! coefficients, which does not depend on common factors of `P` and `Q`).
//!
//! Geometric or faster convergence is summed directly, as SymPy's `hypsum`
//! does: the first term `t(N₀)` comes from the summand, and the series is
//! `t(N₀)·F` with `F = Σ_j u_j`, `u₀ = 1`, `u_{j+1} = u_j·P(k)/Q(k)` at
//! `k = N₀ + j`, where `P(k)` and `Q(k)` are exact integers.  `N₀` lies
//! beyond every root of `P` and `Q` (a Cauchy bound; the terms before it
//! are summed one by one), so the recurrence never divides by zero and the
//! tail has a rigorous geometric bound: for `k ≥ N > 0`,
//!
//! ```text
//! |P(k)/Q(k)| ≤ ρ(N) = Σ|pᵢ|·N^(i−e) / (|q_e| − Σ_{i<e} |qᵢ|·N^(i−e)),   e = deg Q ≥ deg P,
//! ```
//!
//! (the numerator does not increase with `N`, the denominator does not
//! decrease), so once `ρ(N) < 1` the terms after `u_j` add up to at most
//! `|u_j|·ρ/(1 − ρ)`.  The summation stops when that is below the working
//! precision; the reported bound is the first term's relative error, the
//! roundings of the recurrence and of the partial sums, and the tail.
//!
//! Polynomial convergence of a rational term (`Σ 1/(k² + 1)`) is handled
//! before this module, by Euler–Maclaurin with a proven remainder
//! (`emsum.rs`); a polynomially convergent term of any other kind is
//! refused ([`SymplexError::Unevaluable`]), as are sums whose term is not
//! recognised as hypergeometric; divergent sums are
//! [`SymplexError::Divergent`].

use astro_float::{BigFloat, RoundingMode};
use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};
use rustc_hash::FxHashMap;

use super::accuracy::{self, Bound, ErrExp};
use crate::base::arena::Arena;
use crate::base::bigcomplex::{Complex, c_mul, c_sub};
use crate::base::errors::SymplexError;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::{Q, bigint_to_bigfloat};
use crate::base::walk;
use crate::poly::dense::Poly;

/// Highest degree of `P` or `Q` [`term_ratio`] builds.
const MAX_DEGREE: usize = 64;

/// Largest slope of a factorial, `Γ` or binomial argument, and largest
/// exponent of a power, that [`term_ratio`] expands.
const MAX_SHIFT: i64 = 8;

/// Most terms summed before the ratio is regular (below the Cauchy bound).
const MAX_DIRECT_TERMS: u64 = 10_000;

/// Most terms of the recurrence per evaluation.
const MAX_SERIES_TERMS: u64 = 100_000;

/// The summand at an integer index, at a given precision: its value and
/// error bound.
pub(super) type TermFn<'a> =
    dyn FnMut(&BigInt, usize) -> Result<(Complex, Bound), SymplexError> + 'a;

/// The shape of a sub-expression of the summand, as a function of the
/// index `k`.
#[derive(Clone, Debug)]
enum Shape {
    /// Free of `k` (ratio 1); `Some` when it is a rational number.
    Free(Option<Q>),
    /// A polynomial in `k` with rational coefficients.
    Poly(Poly),
    /// A term with ratio `t(k+1)/t(k) = C·P(k)/Q(k)`, where `C` is 1 when
    /// the flag is false, and otherwise a constant that is not a rational
    /// number (`c^m` of `c^(m·k + b)` for an irrational or complex `c`),
    /// recovered numerically from two consecutive terms.
    Ratio(Poly, Poly, bool),
}

fn capped(p: Poly) -> Option<Poly> {
    (p.degree().unwrap_or(0) <= MAX_DEGREE).then_some(p)
}

fn shifted(p: &Poly) -> Poly {
    p.taylor_shift(&Q::one())
}

/// `(k-slope, polynomial)` of a shape that is linear in `k` with an integer
/// slope (a rational constant has slope 0).
fn linear(shape: &Shape) -> Option<(i64, Poly)> {
    match shape {
        Shape::Free(Some(q)) => Some((0, Poly::constant(q.clone()))),
        Shape::Poly(p) if p.degree().unwrap_or(0) <= 1 => {
            let slope = p.coeff(1);
            if !slope.is_integer() {
                return None;
            }
            let slope = slope.to_integer().to_i64()?;
            (slope.abs() <= MAX_SHIFT).then(|| (slope, p.clone()))
        }
        _ => None,
    }
}

/// `x + j` as a polynomial.
fn plus(x: &Poly, j: i64) -> Poly {
    x.add(&Poly::constant(Q::from_integer(BigInt::from(j))))
}

/// `Γ(x(k+1) + 1)/Γ(x(k) + 1)` for `x(k+1) = x(k) + s`, as `(num, den)`:
/// `(x+1)⋯(x+s)` for `s > 0`, `1/(x(x−1)⋯(x+s+1))` for `s < 0`.
fn factorial_step(x: &Poly, s: i64) -> (Poly, Poly) {
    let mut prod = Poly::one();
    if s > 0 {
        for j in 1..=s {
            prod = prod.mul(&plus(x, j));
        }
        (prod, Poly::one())
    } else {
        for j in 0..-s {
            prod = prod.mul(&plus(x, -j));
        }
        (Poly::one(), prod)
    }
}

/// The shape of node `id` from its children's shapes (`None`: not a
/// hypergeometric term, or not recognised).
fn shape_of(
    arena: &Arena,
    id: ExprId,
    var: ExprId,
    shapes: &FxHashMap<ExprId, Shape>,
) -> Option<Shape> {
    if id == var {
        return Some(Shape::Poly(Poly::x()));
    }
    let node = arena.node(id);
    let kids = node.children();
    let mut free = true;
    for c in &kids {
        match shapes.get(c)? {
            Shape::Free(_) => {}
            _ => free = false,
        }
    }
    if free {
        return Some(Shape::Free(arena.as_num(id).cloned()));
    }
    let shape = match node {
        ExprNode::Add(children) => {
            let mut acc = Poly::zero();
            for c in children.iter() {
                match shapes.get(c)? {
                    Shape::Poly(p) => acc = acc.add(p),
                    Shape::Free(Some(q)) => acc = acc.add(&Poly::constant(q.clone())),
                    _ => return None,
                }
            }
            Shape::Poly(acc)
        }
        ExprNode::Mul(children) => {
            let mut poly = Some(Poly::one());
            let (mut num, mut den) = (Poly::one(), Poly::one());
            let mut constant = false;
            for c in children.iter() {
                match shapes.get(c)? {
                    Shape::Free(Some(q)) => poly = poly.map(|p| p.scale(q)),
                    Shape::Free(None) => poly = None,
                    Shape::Poly(p) => {
                        poly = match poly {
                            Some(acc) => Some(capped(acc.mul(p))?),
                            None => None,
                        };
                        num = capped(num.mul(&shifted(p)))?;
                        den = capped(den.mul(p))?;
                    }
                    Shape::Ratio(p, q, c) => {
                        poly = None;
                        num = capped(num.mul(p))?;
                        den = capped(den.mul(q))?;
                        constant |= *c;
                    }
                }
            }
            match poly {
                Some(p) => Shape::Poly(p),
                None => Shape::Ratio(num, den, constant),
            }
        }
        ExprNode::Neg(c) => match shapes.get(c)? {
            Shape::Poly(p) => Shape::Poly(p.neg()),
            other => other.clone(),
        },
        ExprNode::Pow(b, e) => match (shapes.get(b)?, shapes.get(e)?) {
            // c^(m·k + b0): ratio c^m for a rational c and an integer m.
            (Shape::Free(Some(c)), Shape::Poly(ep)) => {
                if c.is_zero() || ep.degree()? > 1 {
                    return None;
                }
                let m = ep.coeff(1);
                if !m.is_integer() {
                    return None;
                }
                let m = m.to_integer().to_i32()?;
                if i64::from(m).abs() > 8 * MAX_SHIFT {
                    return None;
                }
                Shape::Ratio(Poly::constant(c.pow(m)), Poly::one(), false)
            }
            // c^(m·k + b0) for a constant c that is not a rational number:
            // ratio c^m, a constant `infinite_sum` recovers from the terms.
            (Shape::Free(None), Shape::Poly(ep)) => {
                if ep.degree()? > 1 || !ep.coeff(1).is_integer() {
                    return None;
                }
                Shape::Ratio(Poly::one(), Poly::one(), true)
            }
            (base, Shape::Free(Some(n))) if n.is_integer() => {
                let n = n.to_integer().to_i64()?;
                if n == 0 || n.abs() > 4 * MAX_SHIFT {
                    return None;
                }
                let m = usize::try_from(n.unsigned_abs()).ok()?;
                match base {
                    Shape::Poly(p) if n > 0 => Shape::Poly(capped(p.pow(m))?),
                    Shape::Poly(p) => {
                        Shape::Ratio(capped(p.pow(m))?, capped(shifted(p).pow(m))?, false)
                    }
                    Shape::Ratio(p, q, c) if n > 0 => {
                        Shape::Ratio(capped(p.pow(m))?, capped(q.pow(m))?, *c)
                    }
                    Shape::Ratio(p, q, c) => Shape::Ratio(capped(q.pow(m))?, capped(p.pow(m))?, *c),
                    Shape::Free(_) => return None,
                }
            }
            _ => return None,
        },
        // (x)! with x = s·k + c: ratio (x+1)⋯(x+s).
        ExprNode::Factorial(a) => {
            let (s, x) = linear(shapes.get(a)?)?;
            if s <= 0 {
                return None;
            }
            let (num, den) = factorial_step(&x, s);
            Shape::Ratio(num, den, false)
        }
        // Γ(x) = (x − 1)!.
        ExprNode::Gamma(a) => {
            let (s, x) = linear(shapes.get(a)?)?;
            if s <= 0 {
                return None;
            }
            let (num, den) = factorial_step(&plus(&x, -1), s);
            Shape::Ratio(num, den, false)
        }
        // C(n, m) = n!/(m!·(n − m)!).
        ExprNode::Binomial(n, m) => {
            let (sn, xn) = linear(shapes.get(n)?)?;
            let (sm, xm) = linear(shapes.get(m)?)?;
            let xd = xn.sub(&xm);
            let sd = sn - sm;
            if sd.abs() > MAX_SHIFT {
                return None;
            }
            let (n1, d1) = factorial_step(&xn, sn);
            let (n2, d2) = factorial_step(&xm, sm);
            let (n3, d3) = factorial_step(&xd, sd);
            Shape::Ratio(
                capped(n1.mul(&d2).mul(&d3))?,
                capped(d1.mul(&n2).mul(&n3))?,
                false,
            )
        }
        _ => return None,
    };
    Some(shape)
}

/// The ratio `t(k+1)/t(k) = C·P(k)/Q(k)` of the summand `body` in the
/// index `var`, when it is recognised as a hypergeometric term (see the
/// module documentation): `(P, Q, constant)`, `constant` when `C` is not 1
/// (and not a rational number: it is then recovered from the terms).
fn term_ratio(arena: &Arena, body: ExprId, var: ExprId) -> Option<(Poly, Poly, bool)> {
    let mut shapes: FxHashMap<ExprId, Shape> = FxHashMap::default();
    for id in walk::post_order_ids(arena, body) {
        let shape = shape_of(arena, id, var, &shapes)?;
        shapes.insert(id, shape);
    }
    match shapes.remove(&body)? {
        Shape::Free(_) => Some((Poly::one(), Poly::one(), false)),
        Shape::Poly(p) => Some((shifted(&p), p, false)),
        Shape::Ratio(p, q, c) => Some((p, q, c)),
    }
}

/// The coefficients of `P` and `Q` scaled to integers by a common factor
/// (the ratio is unchanged).
fn integer_coefficients(p: &Poly, q: &Poly) -> (Vec<BigInt>, Vec<BigInt>) {
    let mut lcm = BigInt::one();
    for c in p.coeffs().iter().chain(q.coeffs()) {
        lcm = lcm.lcm(c.denom());
    }
    let scale = |poly: &Poly| -> Vec<BigInt> {
        poly.coeffs()
            .iter()
            .map(|c| (c * Q::from_integer(lcm.clone())).to_integer())
            .collect()
    };
    (scale(p), scale(q))
}

fn eval_int(coeffs: &[BigInt], k: &BigInt) -> BigInt {
    let mut acc = BigInt::zero();
    for c in coeffs.iter().rev() {
        acc = acc * k + c;
    }
    acc
}

/// An integer beyond the modulus of every root (Cauchy: `1 + max |cᵢ/c_d|`).
fn root_bound(coeffs: &[BigInt]) -> BigInt {
    let Some((lc, rest)) = coeffs.split_last() else {
        return BigInt::zero();
    };
    let max = rest.iter().map(BigInt::abs).max().unwrap_or_default();
    BigInt::one() + max.div_ceil(&lc.abs())
}

/// `ρ(N)/(1 − ρ(N))` (see the module documentation) as the exponent of an
/// upper bound, when `ρ(N) < 1`.
fn tail_factor_exp(p: &[BigInt], q: &[BigInt], n: &BigInt) -> Option<i64> {
    let (q_lc, q_rest) = q.split_last()?;
    let mut power = BigInt::one();
    let mut p_plus = BigInt::zero();
    let mut q_rest_sum = BigInt::zero();
    for i in 0..q.len() {
        if let Some(c) = p.get(i) {
            p_plus += c.abs() * &power;
        }
        if let Some(c) = q_rest.get(i) {
            q_rest_sum += c.abs() * &power;
        }
        if i + 1 < q.len() {
            power *= n;
        }
    }
    let q_minus = q_lc.abs() * power - q_rest_sum;
    let gap = &q_minus - &p_plus;
    if !q_minus.is_positive() || !gap.is_positive() {
        return None;
    }
    // ρ/(1 − ρ) = P⁺/(Q⁻ − P⁺) < 2^bits(P⁺) / 2^(bits(gap) − 1).
    let bits = |x: &BigInt| i64::try_from(x.bits()).unwrap_or(i64::MAX / 8);
    Some(bits(&p_plus) - bits(&gap) + 1)
}

fn divergent(reason: String) -> SymplexError {
    SymplexError::Divergent {
        operation: "evalf",
        reason,
    }
}

fn unevaluable(reason: impl Into<String>) -> SymplexError {
    SymplexError::Unevaluable {
        reason: reason.into(),
    }
}

/// `Σ_{k ≥ lo} body` for the index `var` at working precision `prec`: the
/// value and its error bound (see the module documentation).  `term(k, p)`
/// evaluates the summand at the integer `k` at `p` bits with its bound.
pub(super) fn infinite_sum(
    arena: &Arena,
    body: ExprId,
    var: ExprId,
    lo: &BigInt,
    prec: usize,
    rm: RoundingMode,
    term: &mut TermFn<'_>,
) -> Result<(Complex, Bound), SymplexError> {
    let (p_ratio, q_ratio, constant) = term_ratio(arena, body, var).ok_or_else(|| {
        unevaluable(format!(
            "infinite sum of `{}`: the term is not recognised as hypergeometric \
             (a rational term ratio), so no error bound is available",
            arena.display(body)
        ))
    })?;
    if q_ratio.is_zero() {
        return Err(unevaluable(
            "infinite sum: the term ratio has a zero denominator",
        ));
    }
    let (p, q) = integer_coefficients(&p_ratio, &q_ratio);

    // Terms before the ratio is regular, one by one.
    let bound = root_bound(&p).max(root_bound(&q));
    let start = lo.clone().max(bound + 1).max(BigInt::one());
    let direct = (&start - lo).to_u64().unwrap_or(u64::MAX);
    if direct > MAX_DIRECT_TERMS {
        return Err(unevaluable(format!(
            "infinite sum: {direct} terms before the term ratio is regular \
             (more than {MAX_DIRECT_TERMS})"
        )));
    }
    let mut sum = super::SumAcc::new(prec);
    let mut k = lo.clone();
    while k < start {
        let (t, e) = term(&k, prec)?;
        sum.add(&t, e, rm);
        k += 1;
    }

    let (t0, b0) = term(&start, prec)?;
    if accuracy::mag(&t0).is_none() && b0.is_exact() {
        // t(N₀) = 0, and P, Q do not vanish beyond N₀: every later term is 0.
        return Ok(sum.finish());
    }
    if constant {
        let (series, bound) =
            scaled_series(arena, body, &p, &q, &start, (&t0, b0), prec, rm, term)?;
        sum.add(&series, bound, rm);
        return Ok(sum.finish());
    }
    if !p.is_empty() {
        classify(&p, &q, arena, body)?;
    }

    // F = Σ u_j at wp bits.
    let wp = prec + 64;
    let (f, f_err) = normalised_series(&p, &q, &start, prec, wp, rm)?;

    // Sanity check of the ratio: t(N₀ + 1) = t(N₀)·P(N₀)/Q(N₀).
    let e0 = b0.joint();
    let (t1, b1) = term(&(&start + 1), prec)?;
    let e1 = b1.joint();
    let r0 = bigint_to_bigfloat(&eval_int(&p, &start), wp).div(
        &bigint_to_bigfloat(&eval_int(&q, &start), wp),
        wp,
        rm,
    );
    let r0 = (r0, BigFloat::new(wp));
    let predicted = c_mul(&t0, &r0, wp, rm);
    let e_pred = accuracy::product_error([(&t0, e0), (&r0, accuracy::rounding(&r0, wp))]);
    let slack = accuracy::shift(
        e1.max(e_pred).max(accuracy::rounding(&predicted, prec)),
        4.0,
    );
    if !accuracy::is_unknown(slack)
        && !accuracy::contains_zero(&c_sub(&t1, &predicted, wp, rm), slack)
    {
        return Err(unevaluable(format!(
            "infinite sum of `{}`: the term ratio does not match the terms",
            arena.display(body)
        )));
    }

    let f = (f, BigFloat::new(wp));
    let series = c_mul(&t0, &f, prec, rm);
    let series_err = accuracy::lsum(
        accuracy::product_error([(&t0, e0), (&f, f_err)]),
        accuracy::rounding(&series, prec),
    );
    // `F` is real: the series is real when its first term is.
    let series_bound = if accuracy::exactly_real(&t0, b0) {
        Bound::real(series_err)
    } else {
        Bound::both(series_err)
    };
    sum.add(&series, series_bound, rm);
    Ok(sum.finish())
}

/// `|x|` of a positive integer ratio `a/b` as an `f64`, rounded up (with a
/// margin for the conversions); `+∞` when it does not fit.
fn ratio_up(a: &BigInt, b: &BigInt) -> f64 {
    let shift = a.bits().max(b.bits()).saturating_sub(900);
    let (a, b) = (a.abs() >> shift, b.abs() >> shift);
    let (Some(a), Some(b)) = (a.to_f64(), b.to_f64()) else {
        return f64::INFINITY;
    };
    if b == 0.0 {
        return f64::INFINITY;
    }
    // A shift truncates both: `a/2ˢ < a' + 1` and `b' ≤ b/2ˢ`, so
    // `(a' + 1)/b' ≥ a/b`; the conversions and the division are within
    // 2⁻⁵³ each, which the margin covers.
    let a = if shift > 0 { a + 1.0 } else { a };
    a / b * (1.0 + 1e-12)
}

/// `Σ_{j ≥ 0} t(N₀ + j)` for a ratio `C·P(k)/Q(k)` whose constant `C` is not a
/// rational number (`√2^k/k!`, `i^k/(2k)!`): `C = t(N₀+1)·Q(N₀)/(t(N₀)·P(N₀))`
/// from the first two terms, checked against the third, and the series
/// `t(N₀)·Σ uⱼ`, `uⱼ₊₁ = uⱼ·C·P(k)/Q(k)`, summed until the geometric tail bound
/// `|uⱼ|·ρ/(1 − ρ)`, `ρ = |C|·P⁺(k)/Q⁻(k)` (see the module documentation), is
/// below the precision.  `C` carries a relative error `ε` from the terms, and
/// `uⱼ` then `j·ε` (to first order).  Before 0.29 such a term was "not
/// recognised as hypergeometric" and the sum refused.
#[allow(clippy::too_many_arguments)]
fn scaled_series(
    arena: &Arena,
    body: ExprId,
    p: &[BigInt],
    q: &[BigInt],
    start: &BigInt,
    first: (&Complex, Bound),
    prec: usize,
    rm: RoundingMode,
    term: &mut TermFn<'_>,
) -> Result<(Complex, Bound), SymplexError> {
    use crate::base::bigcomplex::{c_abs, c_add, c_div, c_from_real};
    let (t0, b0) = first;
    let wp = prec + 64;
    let (t1, b1) = term(&(start + 1), prec)?;
    let (t2, b2) = term(&(start + 2), prec)?;
    let unknown = Ok((t0.clone(), Bound::UNKNOWN));
    // Relative error exponents of the first terms.
    let rel = |t: &Complex, b: Bound| -> Option<i64> {
        let m = accuracy::mag(t)?;
        let e = b.joint();
        if accuracy::is_unknown(e) {
            None
        } else if accuracy::is_exact(e) {
            Some(-(wp as i64))
        } else {
            Some(((e - m as f64 + 1.0).ceil() as i64).max(-(wp as i64)))
        }
    };
    let (Some(r0), Some(r1)) = (rel(t0, b0), rel(&t1, b1)) else {
        return unknown; // a term indistinguishable from 0
    };
    let p0 = eval_int(p, start);
    let q0 = eval_int(q, start);
    if p0.is_zero() || q0.is_zero() {
        return Err(unevaluable("infinite sum: the term ratio vanishes"));
    }
    let ratio0 = c_from_real(
        bigint_to_bigfloat(&q0, wp).div(&bigint_to_bigfloat(&p0, wp), wp, rm),
        wp,
    );
    let c = c_mul(&c_div(&t1, t0, wp, rm), &ratio0, wp, rm);
    // ε(C) ≤ ε(t₀) + ε(t₁) + roundings.
    let eps_c = r0.max(r1) + 2;
    if eps_c > -16 {
        return unknown;
    }
    let c_real = accuracy::exactly_real(t0, b0) && accuracy::exactly_real(&t1, b1);
    // The third term must follow: t₂ = t₁·C·P(N₀+1)/Q(N₀+1).
    let k1 = start + 1;
    let r1q = c_from_real(
        bigint_to_bigfloat(&eval_int(p, &k1), wp).div(
            &bigint_to_bigfloat(&eval_int(q, &k1), wp),
            wp,
            rm,
        ),
        wp,
    );
    let predicted = c_mul(&c_mul(&t1, &c, wp, rm), &r1q, wp, rm);
    let slack = accuracy::shift(
        accuracy::mag(&predicted)
            .map_or(accuracy::UNKNOWN, |m| (m + eps_c + 4) as f64)
            .max(b2.joint()),
        2.0,
    );
    if !accuracy::is_unknown(slack)
        && !accuracy::contains_zero(&c_sub(&t2, &predicted, wp, rm), slack)
    {
        return Err(unevaluable(format!(
            "infinite sum of `{}`: the term ratio does not match the terms",
            arena.display(body)
        )));
    }

    // |C| with its relative error; convergence from the leading ratio.
    let c_mag = super::bigfloat_to_f64_rounded(&c_abs(&c, 64, rm), rm).unwrap_or(f64::INFINITY);
    let eps = 2f64.powi(i32::try_from(eps_c).unwrap_or(0));
    let c_hi = c_mag * (1.0 + eps) * (1.0 + 1e-15);
    let c_lo = c_mag * (1.0 - eps) * (1.0 - 1e-15);
    let (dp, dq) = (p.len() - 1, q.len() - 1);
    let shown = || arena.display(body).to_string();
    if dp > dq {
        return Err(divergent(format!(
            "the terms of the infinite sum of `{}` grow like (k!)^{}",
            shown(),
            dp - dq
        )));
    }
    if dp == dq {
        let g = ratio_up(&p[dp], &q[dq]);
        let g_lo = (p[dp].abs().to_f64().unwrap_or(0.0)
            / q[dq].abs().to_f64().unwrap_or(f64::INFINITY))
            * (1.0 - 1e-12);
        if c_lo * g_lo > 1.0 {
            return Err(divergent(format!(
                "the terms of the infinite sum of `{}` grow geometrically",
                shown()
            )));
        }
        if c_hi * g >= 1.0 {
            return Err(unevaluable(format!(
                "the infinite sum of `{}` converges at best polynomially at this precision; \
                 no rigorous error bound is available",
                shown()
            )));
        }
    }

    // F = Σ uⱼ at wp bits.
    let mag = |x: &BigFloat| -> Option<i64> {
        if x.is_zero() {
            None
        } else {
            x.exponent().map(i64::from)
        }
    };
    let mut u = c_one_at(wp);
    let mut f = u.clone();
    let mut abs_sum = BigFloat::from_i32(1, 64);
    let mut weighted = BigFloat::new(64); // Σ j·|uⱼ|
    let mut peak = 0i64;
    let mut k = start.clone();
    let mut steps: u64 = 0;
    let tail = loop {
        let um = match accuracy::mag(&u) {
            Some(m) => m,
            None => break accuracy::EXACT,
        };
        // ρ = |C|·P⁺(k)/Q⁻(k) at the current k (non-increasing in k).
        if let Some((pp, qm)) = tail_parts(p, q, &k) {
            let rho = c_hi * ratio_up(&pp, &qm);
            if rho < 1.0 {
                let factor = (rho / (1.0 - rho)).log2().max(-1e6);
                let tail = um as f64 + factor + 1.0;
                if tail <= (peak - prec as i64 - 8) as f64 {
                    break tail;
                }
            }
        }
        if steps >= MAX_SERIES_TERMS {
            break accuracy::UNKNOWN;
        }
        let pk = bigint_to_bigfloat(&eval_int(p, &k), wp);
        let qk = bigint_to_bigfloat(&eval_int(q, &k), wp);
        let r = c_from_real(pk.div(&qk, wp, rm), wp);
        u = c_mul(&c_mul(&u, &c, wp, rm), &r, wp, rm);
        f = c_add(&f, &u, wp, rm);
        steps += 1;
        let ua = c_abs(&u, 64, rm);
        abs_sum = abs_sum.add(&ua, 64, RoundingMode::Up);
        weighted = weighted.add(
            &ua.mul(&BigFloat::from_u64(steps, 64), 64, RoundingMode::Up),
            64,
            RoundingMode::Up,
        );
        peak = peak.max(accuracy::mag(&f).unwrap_or(peak));
        k += 1;
    };
    if accuracy::is_unknown(tail) {
        return unknown;
    }
    let wp_i = wp as i64;
    let n = accuracy::ceil_log2(usize::try_from(steps).unwrap_or(usize::MAX).max(1));
    let recurrence = (mag(&abs_sum).unwrap_or(0) + n + 3 - wp_i) as f64;
    let rounding = (peak + n - wp_i) as f64;
    let from_c = mag(&weighted).map_or(accuracy::EXACT, |m| (m + eps_c + 1) as f64);
    let f_err = accuracy::shift(recurrence.max(rounding).max(tail).max(from_c), 1.0);
    let series = c_mul(t0, &f, prec, rm);
    let err = accuracy::lsum(
        accuracy::product_error([(t0, b0.joint()), (&f, f_err)]),
        accuracy::rounding(&series, prec),
    );
    tracing::debug!(
        steps,
        err,
        "evalf: hypergeometric series with a constant ratio factor"
    );
    let bound = if c_real && accuracy::exactly_real(t0, b0) {
        Bound::real(err)
    } else {
        Bound::both(err)
    };
    Ok((series, bound))
}

/// `1 + 0i` at `wp` bits.
fn c_one_at(wp: usize) -> Complex {
    crate::base::bigcomplex::c_one(wp)
}

/// `(P⁺(k), Q⁻(k))` of the tail bound (see the module documentation):
/// `Σ|pᵢ|·k^(i−e)` and `|qₑ| − Σ_{i<e} |qᵢ|·k^(i−e)` scaled by `k^e`, when the
/// latter is positive.
fn tail_parts(p: &[BigInt], q: &[BigInt], n: &BigInt) -> Option<(BigInt, BigInt)> {
    let (q_lc, q_rest) = q.split_last()?;
    let mut power = BigInt::one();
    let mut p_plus = BigInt::zero();
    let mut q_rest_sum = BigInt::zero();
    for i in 0..q.len() {
        if let Some(c) = p.get(i) {
            p_plus += c.abs() * &power;
        }
        if let Some(c) = q_rest.get(i) {
            q_rest_sum += c.abs() * &power;
        }
        if i + 1 < q.len() {
            power *= n;
        }
    }
    let q_minus = q_lc.abs() * power - q_rest_sum;
    q_minus.is_positive().then_some((p_plus, q_minus))
}

/// Refuse a sum that diverges or converges only polynomially (see the
/// module documentation), for a non-zero ratio `P/Q`.
fn classify(p: &[BigInt], q: &[BigInt], arena: &Arena, body: ExprId) -> Result<(), SymplexError> {
    let shown = || arena.display(body).to_string();
    let (dp, dq) = (p.len() - 1, q.len() - 1);
    if dp > dq {
        return Err(divergent(format!(
            "the terms of the infinite sum of `{}` grow like (k!)^{}",
            shown(),
            dp - dq
        )));
    }
    if dp < dq {
        return Ok(()); // factorial convergence
    }
    let g = Q::new(p[dp].clone(), q[dq].clone());
    let one = Q::one();
    if g.abs() < one {
        return Ok(()); // geometric convergence
    }
    if g.abs() > one {
        return Err(divergent(format!(
            "the terms of the infinite sum of `{}` grow geometrically (ratio → {g})",
            shown()
        )));
    }
    // |t(k)| ≈ C·k^(−e): |P/Q| = 1 − e/k + O(1/k²).
    let e = if dp == 0 {
        Q::zero()
    } else {
        Q::new(q[dq - 1].clone(), q[dq].clone()) - Q::new(p[dp - 1].clone(), p[dp].clone())
    };
    let converges = if g.is_positive() {
        e > one
    } else {
        e.is_positive()
    };
    if converges {
        Err(unevaluable(format!(
            "the infinite sum of `{}` converges only polynomially (terms ~ k^(−{e})); \
             an extrapolation with a rigorous error bound is not implemented",
            shown()
        )))
    } else {
        Err(divergent(format!(
            "the terms of the infinite sum of `{}` decay only like k^(−{e})",
            shown()
        )))
    }
}

/// `F = Σ_{j ≥ 0} u_j`, `u₀ = 1`, `u_{j+1} = u_j·P(N₀ + j)/Q(N₀ + j)`, at
/// `wp` bits, summed until a rigorous bound on the tail is below
/// `2^(−prec)` relative to the partial sums: `F` and its error bound.
fn normalised_series(
    p: &[BigInt],
    q: &[BigInt],
    start: &BigInt,
    prec: usize,
    wp: usize,
    rm: RoundingMode,
) -> Result<(BigFloat, ErrExp), SymplexError> {
    let mag = |x: &BigFloat| -> Option<i64> {
        if x.is_zero() {
            None
        } else {
            x.exponent().map(i64::from)
        }
    };
    let mut u = BigFloat::from_i32(1, wp);
    let mut f = u.clone();
    let mut abs_sum = BigFloat::from_i32(1, 64);
    let mut peak = mag(&f).unwrap_or(0);
    let mut k = start.clone();
    let mut steps: u64 = 0;
    let mut tail_factor: Option<i64> = tail_factor_exp(p, q, &k);
    let mut next_check: u64 = 1;
    let tail = loop {
        let Some(um) = mag(&u) else {
            break accuracy::EXACT; // every later term is 0
        };
        if let Some(tf) = tail_factor {
            // |u_j| < 2^um carries a relative error far below 1: +1.
            let tail = um + tf + 1;
            if tail <= peak - prec as i64 - 8 {
                break tail as f64;
            }
        }
        if steps >= MAX_SERIES_TERMS {
            break tail_factor.map_or(accuracy::UNKNOWN, |tf| (um + tf + 1) as f64);
        }
        let pk = bigint_to_bigfloat(&eval_int(p, &k), wp);
        let qk = bigint_to_bigfloat(&eval_int(q, &k), wp);
        u = u.mul(&pk, wp, rm).div(&qk, wp, rm);
        f = f.add(&u, wp, rm);
        abs_sum = abs_sum.add(&u.abs(), 64, rm);
        peak = peak.max(mag(&f).unwrap_or(peak));
        k += 1;
        steps += 1;
        if steps == next_check {
            next_check *= 2;
            if let Some(tf) = tail_factor_exp(p, q, &k) {
                tail_factor = Some(tail_factor.map_or(tf, |old| old.min(tf)));
            }
        }
    };
    if accuracy::is_unknown(tail) {
        return Ok((f, accuracy::UNKNOWN));
    }
    // Each u_j carries a relative error below 3j·2^(−wp) (two roundings per
    // step, the exact integers converted exactly or with one more), and
    // each partial sum one rounding.
    let wp_i = wp as i64;
    let n = accuracy::ceil_log2(usize::try_from(steps).unwrap_or(usize::MAX).max(1));
    let recurrence = (mag(&abs_sum).unwrap_or(0) + n + 2 - wp_i) as f64;
    let rounding = (peak + n - wp_i) as f64;
    let err = accuracy::shift(recurrence.max(rounding).max(tail), 1.0);
    tracing::debug!(steps, err, "evalf: hypergeometric series summed");
    Ok((f, err))
}

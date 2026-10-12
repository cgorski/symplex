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
//! (`emsum.rs`); a smooth term that is not hypergeometric (`ln k/k²`,
//! `H_k/k³`) by its expansion at infinity (`asymsum.rs`, tried after the
//! dominated series below; before 0.44 refused); a polynomially convergent
//! term of any other kind is refused ([`SymplexError::Unevaluable`]);
//! divergent sums are
//! [`SymplexError::Divergent`].  A term that is not hypergeometric but a sum
//! of hypergeometric terms of geometric decay times factors of polynomial
//! growth (`λᵏ·ln(k!)/k!`, `ln(k)/2ᵏ`) is summed term by term against a
//! dominating series (see "Dominated series" below; before 0.44 refused),
//! and so is an infinite product of `1 + a(k)` with such an `a`.

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

/// A sub-expression of the summand (the summand itself, a factor, a
/// constant) at an integer index, at a given precision: its value and error
/// bound.
pub(super) type PartFn<'a> =
    dyn FnMut(ExprId, &BigInt, usize) -> Result<(Complex, Bound), SymplexError> + 'a;

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
    part: &mut PartFn<'_>,
) -> Result<(Complex, Bound), SymplexError> {
    let Some((p_ratio, q_ratio, constant)) = term_ratio(arena, body, var) else {
        if let Some(result) = dominated_sum(arena, body, var, lo, prec, rm, part) {
            return result;
        }
        // A smooth term of polynomial decay: its expansion at infinity
        // (`asymsum.rs`; before 0.44 refused).
        if let Some(result) = super::asymsum::asymptotic_sum(arena, body, var, lo, prec, rm, part) {
            return result;
        }
        return Err(unevaluable(format!(
            "infinite sum of `{}`: the term is neither hypergeometric (a rational term \
             ratio) nor a hypergeometric term times factors of polynomial growth, so no \
             error bound is available",
            arena.display(body)
        )));
    };
    let term = &mut |k: &BigInt, p: usize| part(body, k, p);
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

// ── Dominated series ──────────────────────────────────────────────────────────────
//
// A summand that is not hypergeometric, `λᵏ·ln(k!)/k!` (the entropy of a
// Poisson law), `ln(k)/2ᵏ`, `sin(k)/k!`, `harmonic(k)·xᵏ`, is summed term
// by term with a rigorous bound on the rest when each of its addends is
// `H(k)·S(k)`: a hypergeometric factor `H` whose ratio `C·P(k)/Q(k)` tends
// to a limit of modulus below 1 (or to 0) — `C` the ratio `c^m` of factors
// `c^(m·k + b)` with an irrational or complex constant `c`, or `e^a` of
// `exp(a·k + b)`, bounded in modulus from the value of `c` — times factors
// `S` of at most polynomial growth.  For `S` an envelope
//
//     |S(k)| ≤ E(k) = 2^a·k^p·(ln k)^q        for every integer k ≥ N ≥ 3
//
// is derived from its expression tree: `|k| = k`, a polynomial below
// `Σ|cᵢ|·k^d`, sums and products of envelopes (`k^p ≤ k^p'` and `(ln k)^q ≤
// (ln k)^q'` for `p ≤ p'`, `q ≤ q'` since `k, ln k ≥ 1`), real powers,
// `|sin|, |cos|, |tanh| ≤ 1` and `|atan| ≤ π/2` of real arguments,
// `0 ≤ H(x) ≤ 1 + ln(x + 1)` for `harmonic` of a positive polynomial,
// `|Log z| ≤ |ln|z|| + π`, and for a polynomial `c(k)` without roots beyond
// `N` (`L·k^d ≤ |c(k)| ≤ U·k^d`) `|ln c(k)| ≤ (max(|ln L|, |ln U|) + d)·ln k`;
// for a hypergeometric `G(k)` without zeros beyond `N`, whose ratio has
// `|ln|r(j)|| ≤ α + β·ln j` there, `|ln G(k)| ≤ (|ln|G(N)|| + α + β + π)·k·ln k`
// (so `|ln(λᵏ/k!)| ≤ C·k·ln k`); a hypergeometric factor whose ratio is at most
// 1 beyond `N` is at most its value at `N`.  Then for `k ≥ M ≥ N`
//
//     |H(k + 1)|·E(k + 1) ≤ ρ(M)·γ(M)·|H(k)|·E(k),
//
// with `ρ(M)` `|C|` times the bound of `|P/Q|` of the module documentation (which does
// not increase with `M`) and `γ(M) = (1 + 1/M)^p·(ln(M + 1)/ln M)^q` the
// largest growth of `E` from `M` on: the rest of the series is at most
// `|H(M)|·E(M)/(1 − ρ(M)γ(M))` once `ργ < 1`, with `|H(M)|` from the factors
// of `H` evaluated at `M`.  Terms are summed (each evaluated with its own
// bound) until that is below the precision.  An infinite product `∏(1 +
// a(k))` whose `a` has such addends is multiplied out until the bound `T` on
// `Σ_{j≥M} |a(j)|` is at most `¼`: then every `|a(j)| ≤ ¼`, `|ln(1 + a(j))| ≤
// 2|a(j)|`, and the rest of the product is `e^w` with `|w| ≤ 2T`, within
// `e^(2T) − 1` of 1.  The value is real when every node is real for real
// arguments and every logarithm's argument is positive beyond `N`.
//
// mpmath's `nsum` (BSD) extrapolates such sums (Richardson, Shanks,
// Euler–Maclaurin) with an error estimate, not a bound.

/// Most terms (or factors) [`dominated_sum`] and [`dominated_product`]
/// evaluate.
const MAX_DOMINATED_TERMS: u64 = 20_000;

/// Precision of the evaluations behind the bounds (magnitudes, signs).
const AUX_PREC: usize = 128;

/// `x` rounded up by more than the `f64` rounding of the arithmetic it came
/// from (infinities unchanged).
fn up(x: f64) -> f64 {
    if x.is_finite() {
        x + x.abs() * 1e-12 + 1e-12
    } else {
        x
    }
}

/// A non-negative `x` (not a logarithm) rounded up likewise.
fn up_linear(x: f64) -> f64 {
    x * (1.0 + 1e-12)
}

/// `x` rounded down likewise.
fn down(x: f64) -> f64 {
    if x.is_finite() {
        x - x.abs() * 1e-12 - 1e-12
    } else {
        x
    }
}

/// `log₂ π` (rounded up).
const LOG2_PI: f64 = 1.651_496_129_472_319;

/// An upper bound `|S(k)| ≤ 2^a·k^p·(ln k)^q` for every integer `k ≥ N ≥ 3`
/// (`a = −∞`: `S` vanishes there).
#[derive(Clone, Copy, Debug)]
struct Envelope {
    a: f64,
    p: f64,
    q: f64,
}

impl Envelope {
    fn constant(a: f64) -> Envelope {
        Envelope { a, p: 0.0, q: 0.0 }
    }

    fn plus(self, o: Envelope) -> Envelope {
        Envelope {
            a: accuracy::lsum(self.a, o.a),
            p: self.p.max(o.p),
            q: self.q.max(o.q),
        }
    }

    fn times(self, o: Envelope) -> Envelope {
        Envelope {
            a: up(self.a + o.a),
            p: self.p + o.p,
            q: self.q + o.q,
        }
    }

    /// `|S|^r` for `r > 0`.
    fn power(self, r: f64) -> Envelope {
        Envelope {
            a: up(self.a * r),
            p: self.p * r,
            q: self.q * r,
        }
    }

    /// `log₂ E(k)` for `k ≥ 3`, rounded up.
    fn at(self, k: f64) -> f64 {
        up(self.a + self.p * k.log2() + self.q * k.ln().log2())
    }

    /// `log₂` of the largest `E(j + 1)/E(j)` over `j ≥ k ≥ 3`, rounded up.
    fn growth(self, k: f64) -> f64 {
        up(
            self.p.max(0.0) * (1.0 / k).ln_1p() * std::f64::consts::LOG2_E
                + self.q.max(0.0) * ((k + 1.0).ln() / k.ln()).log2(),
        )
    }
}

/// `log₂ x` of a positive integer, as `(lower, upper)` bounds.
fn lg_big(x: &BigInt) -> Option<(f64, f64)> {
    if !x.is_positive() {
        return None;
    }
    let bits = x.bits();
    let shift = bits.saturating_sub(53);
    let top = (x >> shift).to_f64()?;
    let s = shift as f64;
    // Exact below 2⁵³; above, x lies in [top, top + 1)·2^shift.
    let next = if shift == 0 { top } else { top + 1.0 };
    Some((down(top.log2() + s), up(next.log2() + s)))
}

/// `log₂` of an upper bound on `|x ± 2^e|` (the true value of an evaluated
/// node), `None` without a bound.
fn lg_upper(v: &Complex, b: Bound) -> Option<f64> {
    let e = b.joint();
    (!accuracy::is_unknown(e)).then(|| accuracy::lsum(accuracy::lg_abs(v), e))
}

/// A polynomial with rational coefficients of degree `d ≥ 1` on `k ≥ N`,
/// where it has no root: `(L, U, d, lc > 0)` with `L·k^d ≤ |c(k)| ≤ U·k^d`
/// (`L` rounded down, `U` up).
fn poly_bounds(p: &Poly, n: &BigInt) -> Option<(f64, f64, usize, bool)> {
    let d = p.degree()?;
    let lc = p.coeff(d);
    if lc.is_zero() {
        return None;
    }
    let nq = Q::from_integer(n.clone());
    let mut upper = Q::zero();
    let mut rest = Q::zero();
    for i in 0..=d {
        let c = p.coeff(i).abs();
        upper += &c;
        if i < d {
            // |cᵢ|·N^(i−d), which does not increase with k ≥ N.
            let mut t = c;
            for _ in i..d {
                t /= &nq;
            }
            rest += t;
        }
    }
    let low = lc.abs() - rest;
    if !low.is_positive() {
        return None;
    }
    let f = |x: &Q| crate::base::numeric::ratio_to_f64(x);
    let low = f(&low)? * (1.0 - 1e-12);
    (low > 0.0).then_some((low, up_linear(f(&upper)?), d, lc.is_positive()))
}

/// One addend `H(k)·S(k)` of a dominated summand (see above).
struct Dominated {
    /// The factors that make up `H`.
    factors: Vec<ExprId>,
    /// `H(k+1)/H(k) = C·P(k)/Q(k)`, integer coefficients, and `log₂` of an
    /// upper bound on `|C|` (the exponential factors).
    p: Vec<BigInt>,
    q: Vec<BigInt>,
    log_c: f64,
    /// `−log₂` of the limit of `|C·P/Q|`: bits gained per term in the end.
    rate: f64,
    /// The envelope of the product of the other factors.
    env: Envelope,
}

/// What the analysis knows of a node for `k ≥ N`.
#[derive(Clone, Copy, Debug, Default)]
struct Facts {
    /// Real for every integer `k ≥ N`.
    real: bool,
    /// Real and positive for every integer `k ≥ N`.
    positive: bool,
}

/// The analysis of the addends of a summand (see above).
struct Analysis<'s, 'p, 'q> {
    arena: &'s Arena,
    var: ExprId,
    n: BigInt,
    shapes: FxHashMap<ExprId, Shape>,
    facts: FxHashMap<ExprId, Facts>,
    part: &'p mut PartFn<'q>,
}

impl Analysis<'_, '_, '_> {
    /// A node at `N` (its value and bound), at [`AUX_PREC`] bits.
    fn value(&mut self, id: ExprId) -> Option<(Complex, Bound)> {
        let (v, b) = (self.part)(id, &self.n, AUX_PREC).ok()?;
        (!b.is_unknown()).then_some((v, b))
    }

    /// [`Facts`] of every node of `root`, bottom-up.
    fn facts_of(&mut self, root: ExprId) -> Option<()> {
        for id in walk::post_order_ids(self.arena, root) {
            if self.facts.contains_key(&id) {
                continue;
            }
            let f = self.node_facts(id)?;
            self.facts.insert(id, f);
        }
        Some(())
    }

    fn node_facts(&mut self, id: ExprId) -> Option<Facts> {
        let arena = self.arena;
        let fact = |s: &Self, c: &ExprId| s.facts.get(c).copied().unwrap_or_default();
        let real_only = |r: bool| Facts {
            real: r,
            positive: false,
        };
        if id == self.var {
            return Some(Facts {
                real: true,
                positive: true,
            });
        }
        match self.shapes.get(&id) {
            Some(Shape::Free(Some(q))) => {
                return Some(Facts {
                    real: true,
                    positive: q.is_positive(),
                });
            }
            Some(Shape::Free(None)) => {
                let Some((v, b)) = self.value(id) else {
                    return Some(Facts::default());
                };
                let real = accuracy::exactly_real(&v, b);
                let positive =
                    real && v.0.is_positive() && !accuracy::part_contains_zero(&v.0, b.re);
                return Some(Facts { real, positive });
            }
            Some(Shape::Poly(p)) => {
                // Beyond its roots a polynomial has the sign of its leading
                // coefficient (`N` is beyond them, see `analyse`).
                let lc = p.degree().map(|d| p.coeff(d));
                return Some(Facts {
                    real: true,
                    positive: lc.is_some_and(|c| c.is_positive()),
                });
            }
            _ => {}
        }
        let node = arena.node(id);
        let kids = node.children();
        let all = |s: &Self, pred: fn(Facts) -> bool| kids.iter().all(|c| pred(fact(s, c)));
        Some(match node {
            ExprNode::Num(_)
            | ExprNode::Pi
            | ExprNode::E
            | ExprNode::EulerGamma
            | ExprNode::Catalan
            | ExprNode::GoldenRatio => Facts {
                real: true,
                positive: true,
            },
            ExprNode::Add(_) => Facts {
                real: all(self, |f| f.real),
                positive: all(self, |f| f.positive),
            },
            ExprNode::Mul(_) => Facts {
                real: all(self, |f| f.real),
                positive: all(self, |f| f.positive),
            },
            ExprNode::Neg(c) => real_only(fact(self, c).real),
            ExprNode::Abs(c) => Facts {
                real: true,
                positive: fact(self, c).positive,
            },
            ExprNode::Exp(c) => {
                let r = fact(self, c).real;
                Facts {
                    real: r,
                    positive: r,
                }
            }
            ExprNode::Pow(b, e) => {
                let (fb, fe) = (fact(self, b), fact(self, e));
                let integer_exponent = match self.shapes.get(e) {
                    Some(Shape::Free(Some(q))) => q.is_integer(),
                    Some(Shape::Poly(p)) => p.coeffs().iter().all(Q::is_integer),
                    _ => false,
                };
                if fb.positive && fe.real {
                    Facts {
                        real: true,
                        positive: true,
                    }
                } else {
                    real_only(fb.real && integer_exponent)
                }
            }
            ExprNode::Ln(c) => real_only(fact(self, c).positive),
            ExprNode::Sin(c)
            | ExprNode::Cos(c)
            | ExprNode::Tan(c)
            | ExprNode::Atan(c)
            | ExprNode::Sinh(c)
            | ExprNode::Tanh(c)
            | ExprNode::Erf(c) => real_only(fact(self, c).real),
            ExprNode::Cosh(c) | ExprNode::Erfc(c) => {
                let r = fact(self, c).real;
                Facts {
                    real: r,
                    positive: r,
                }
            }
            ExprNode::Factorial(c) | ExprNode::Gamma(c) => {
                // Γ is positive on (0, ∞); a positive argument here is a
                // polynomial beyond its roots.
                let f = fact(self, c);
                Facts {
                    real: f.real,
                    positive: f.positive,
                }
            }
            ExprNode::Binomial(a, b) => {
                // C(a, b) > 0 for integers a ≥ b ≥ 0: `b` and `a − b`
                // polynomials with integer coefficients, non-negative beyond N.
                let poly = |s: &Self, id: &ExprId| match s.shapes.get(id) {
                    Some(Shape::Poly(p)) => Some(p.clone()),
                    Some(Shape::Free(Some(q))) => Some(Poly::constant(q.clone())),
                    _ => None,
                };
                let n = self.n.clone();
                let nonnegative = |p: &Poly| {
                    let (c, _) = integer_coefficients(p, &Poly::one());
                    p.coeffs().iter().all(Q::is_integer)
                        && p.degree().is_none_or(|d| !p.coeff(d).is_negative())
                        && n > root_bound(&c)
                };
                let positive = match (poly(self, a), poly(self, b)) {
                    (Some(pa), Some(pb)) => nonnegative(&pb) && nonnegative(&pa.sub(&pb)),
                    _ => false,
                };
                Facts {
                    real: all(self, |f| f.real),
                    positive,
                }
            }
            ExprNode::Apply(sid, args) => {
                if arena.lib_fn(*sid) == Some(crate::base::libfn::LibFn::Harmonic)
                    && args.len() == 1
                {
                    let f = fact(self, &args[0]);
                    real_only(f.positive)
                } else {
                    Facts::default()
                }
            }
            _ => Facts::default(),
        })
    }

    /// The envelope of the factor `root` (see above), `None` when a node has
    /// no rule.
    fn envelope(&mut self, root: ExprId) -> Option<Envelope> {
        // Post-order over the children whose envelopes the rules use (not
        // into the argument of `ln`, `sin`, …, which have rules of their own).
        let mut env: FxHashMap<ExprId, Envelope> = FxHashMap::default();
        let mut stack = vec![(root, false)];
        while let Some((id, expanded)) = stack.pop() {
            if env.contains_key(&id) {
                continue;
            }
            if expanded {
                let e = self.node_envelope(id, &env)?;
                env.insert(id, e);
            } else {
                stack.push((id, true));
                for c in self.enveloped_children(id) {
                    if !env.contains_key(&c) {
                        stack.push((c, false));
                    }
                }
            }
        }
        env.remove(&root)
    }

    /// The children of `id` whose envelopes [`Self::node_envelope`] uses.
    fn enveloped_children(&self, id: ExprId) -> Vec<ExprId> {
        if id == self.var || self.shapes.contains_key(&id) {
            return Vec::new();
        }
        match self.arena.node(id) {
            ExprNode::Add(children) | ExprNode::Mul(children) => children.to_vec(),
            ExprNode::Neg(c) | ExprNode::Abs(c) => vec![*c],
            ExprNode::Pow(b, e) if self.arena.as_num(*e).is_some_and(|r| r.is_positive()) => {
                vec![*b]
            }
            _ => Vec::new(),
        }
    }

    fn node_envelope(&mut self, id: ExprId, env: &FxHashMap<ExprId, Envelope>) -> Option<Envelope> {
        let arena = self.arena;
        if id == self.var {
            return Some(Envelope {
                a: 0.0,
                p: 1.0,
                q: 0.0,
            });
        }
        match self.shapes.get(&id).cloned() {
            Some(Shape::Free(Some(q))) => {
                let a = if q.is_zero() {
                    f64::NEG_INFINITY
                } else {
                    let (_, hi) = lg_q(&q.abs())?;
                    hi
                };
                return Some(Envelope::constant(a));
            }
            Some(Shape::Free(None)) => {
                let (v, b) = self.value(id)?;
                return Some(Envelope::constant(lg_upper(&v, b)?));
            }
            Some(Shape::Poly(p)) => {
                let d = p.degree().unwrap_or(0);
                let sum = p.coeffs().iter().fold(Q::zero(), |acc, c| acc + c.abs());
                if sum.is_zero() {
                    return Some(Envelope::constant(f64::NEG_INFINITY));
                }
                return Some(Envelope {
                    a: lg_q(&sum)?.1,
                    p: d as f64,
                    q: 0.0,
                });
            }
            Some(Shape::Ratio(p, q, false)) => {
                // At most its value at N when the ratio is at most 1 beyond
                // (`tail_parts` bounds it for deg P ≤ deg Q).
                let (pi, qi) = integer_coefficients(&p, &q);
                if pi.len() > qi.len() {
                    return None;
                }
                let (pp, qm) = tail_parts(&pi, &qi, &self.n)?;
                if pp > qm {
                    return None;
                }
                let (v, b) = self.value(id)?;
                return Some(Envelope::constant(lg_upper(&v, b)?));
            }
            Some(Shape::Ratio(_, _, true)) => return None,
            None => {}
        }
        let child = |c: &ExprId| env.get(c).copied();
        let node = arena.node(id);
        Some(match node {
            ExprNode::Add(children) => {
                let mut acc = Envelope::constant(f64::NEG_INFINITY);
                for c in children.iter() {
                    acc = acc.plus(child(c)?);
                }
                acc
            }
            ExprNode::Mul(children) => {
                let mut acc = Envelope::constant(0.0);
                for c in children.iter() {
                    acc = acc.times(child(c)?);
                }
                acc
            }
            ExprNode::Neg(c) | ExprNode::Abs(c) => child(c)?,
            ExprNode::Pow(b, e)
                if self.arena.as_num(*b).is_some_and(Q::is_zero)
                    && matches!(self.shapes.get(e), Some(Shape::Poly(p))
                        if p.degree().is_some_and(|d| p.coeff(d).is_positive())) =>
            {
                // 0^e(k) = 0 once e(k) > 0, beyond the roots of e.
                Envelope::constant(f64::NEG_INFINITY)
            }
            ExprNode::Pow(b, e) => {
                let r = self.arena.as_num(*e)?;
                let rf = crate::base::numeric::ratio_to_f64(r)?;
                if r.is_positive() {
                    child(b)?.power(rf)
                } else if r.is_negative() {
                    // A polynomial without roots beyond N: |c(k)|^r ≤ (L k^d)^r.
                    let Some(Shape::Poly(p)) = self.shapes.get(b) else {
                        return None;
                    };
                    let (low, _, d, _) = poly_bounds(p, &self.n)?;
                    Envelope {
                        a: up(rf * low.log2()),
                        p: rf * d as f64,
                        q: 0.0,
                    }
                } else {
                    Envelope::constant(0.0)
                }
            }
            ExprNode::Ln(x) => self.log_envelope(*x)?,
            ExprNode::Sin(c) | ExprNode::Cos(c) | ExprNode::Tanh(c) | ExprNode::Erf(c) => {
                self.facts.get(c)?.real.then_some(Envelope::constant(0.0))?
            }
            ExprNode::Atan(c) => self
                .facts
                .get(c)?
                .real
                .then_some(Envelope::constant(up(LOG2_PI - 1.0)))?,
            ExprNode::Apply(sid, args)
                if arena.lib_fn(*sid) == Some(crate::base::libfn::LibFn::Harmonic)
                    && args.len() == 1 =>
            {
                // 0 ≤ H(x) ≤ ψ(x + 1) + γ ≤ 1 + ln(x + 1) for x ≥ 0, and
                // x = c(k) ≤ U·k^d.
                let Some(Shape::Poly(p)) = self.shapes.get(&args[0]) else {
                    return None;
                };
                let (_, upper, d, positive) = poly_bounds(p, &self.n)?;
                if !positive {
                    return None;
                }
                let c = up(1.0 + up(upper + 1.0).ln() + d as f64);
                Envelope {
                    a: up(c.log2()),
                    p: 0.0,
                    q: 1.0,
                }
            }
            _ => return None,
        })
    }

    /// The envelope of `ln x` (see above): `x` a polynomial or a
    /// hypergeometric term without zeros beyond `N`.
    fn log_envelope(&mut self, x: ExprId) -> Option<Envelope> {
        let positive = self.facts.get(&x).is_some_and(|f| f.positive);
        let branch = if positive { 0.0 } else { std::f64::consts::PI };
        let c = match self.shapes.get(&x).cloned()? {
            Shape::Poly(p) => {
                // L k^d ≤ |c(k)| ≤ U k^d: |ln|c(k)|| ≤ max(|ln L|, |ln U|) + d ln k.
                let (low, upper, d, _) = poly_bounds(&p, &self.n)?;
                let c = up(low.ln().abs().max(upper.ln().abs()) + d as f64 + branch);
                return Some(Envelope {
                    a: up(c.log2()),
                    p: 0.0,
                    q: 1.0,
                });
            }
            Shape::Ratio(p, q, false) => {
                let (pi, qi) = integer_coefficients(&p, &q);
                // |G(N)| within [lo, hi], away from 0.
                let (v, b) = self.value(x)?;
                let e = b.joint();
                let lv = accuracy::lg_abs_low(&v);
                if lv.is_nan() || lv <= e + 1.0 {
                    return None;
                }
                let lo = down(lv + (-(e - lv).exp2()).ln_1p() * std::f64::consts::LOG2_E);
                let hi = lg_upper(&v, b)?;
                let at_n = (lo.abs().max(hi.abs())) * std::f64::consts::LN_2;
                // α: |ln|P(j)/Q(j)|| ≤ α + β ln j for j ≥ N.
                let (lp, up_p) = ratio_bounds(&pi, &self.n)?;
                let (lq, up_q) = ratio_bounds(&qi, &self.n)?;
                let alpha = (lp - up_q).abs().max((up_p - lq).abs()) * std::f64::consts::LN_2;
                let beta = (pi.len() as f64 - qi.len() as f64).abs();
                up(at_n + alpha + beta + branch)
            }
            _ => return None,
        };
        Some(Envelope {
            a: up(up(c).log2()),
            p: 1.0,
            q: 1.0,
        })
    }
}

/// An exponential factor of a summand: `c^(m·k + b)` for a constant `c`
/// that is not a rational number (its ratio `c^m` is not part of `P/Q`), or
/// `exp(a·k + b)` for a rational `a`.
#[derive(Clone, Copy)]
enum Exponential {
    /// `c^(m·k + b)`: the base and the slope `m`.
    Power(ExprId, i64),
    /// `exp(a·k + b)`: the slope `a` as an `f64` rounded up.
    Exp(f64),
}

fn exponential_factor(
    arena: &Arena,
    f: ExprId,
    shapes: &FxHashMap<ExprId, Shape>,
) -> Option<Exponential> {
    let linear = |e: &ExprId| match shapes.get(e) {
        Some(Shape::Poly(p)) if p.degree() == Some(1) => Some(p.coeff(1)),
        _ => None,
    };
    match arena.node(f) {
        ExprNode::Pow(b, e) if matches!(shapes.get(b), Some(Shape::Free(None))) => {
            let m = linear(e)?;
            if !m.is_integer() {
                return None;
            }
            Some(Exponential::Power(*b, m.to_integer().to_i64()?))
        }
        ExprNode::Exp(e) => {
            let a = crate::base::numeric::ratio_to_f64(&linear(e)?)?;
            Some(Exponential::Exp(up(a)))
        }
        _ => None,
    }
}

impl Analysis<'_, '_, '_> {
    /// `log₂` of an upper bound on the modulus of the ratio of an exponential
    /// factor: `m·log₂|c|`, or `a·log₂ e`.
    fn exponential_ratio(&mut self, e: Exponential) -> Option<f64> {
        match e {
            Exponential::Exp(a) => Some(up(a * std::f64::consts::LOG2_E)),
            Exponential::Power(base, m) => {
                let (v, b) = self.value(base)?;
                let hi = lg_upper(&v, b)?;
                let lv = accuracy::lg_abs_low(&v);
                let err = b.joint();
                if m >= 0 {
                    return Some(up(m as f64 * hi));
                }
                // |c| from below, away from 0.
                if lv.is_nan() || lv <= err + 1.0 {
                    return None;
                }
                let lo = down(lv + (-(err - lv).exp2()).ln_1p() * std::f64::consts::LOG2_E);
                Some(up(m as f64 * lo))
            }
        }
    }
}

/// `log₂` bounds of a positive rational.
fn lg_q(x: &Q) -> Option<(f64, f64)> {
    let (nl, nh) = lg_big(x.numer())?;
    let (dl, dh) = lg_big(x.denom())?;
    Some((down(nl - dh), up(nh - dl)))
}

/// For integer coefficients `c` of degree `d` and `k ≥ N` beyond the roots:
/// `log₂` bounds `(lower, upper)` of `L`, `U` with `L·k^d ≤ |c(k)| ≤ U·k^d`.
fn ratio_bounds(c: &[BigInt], n: &BigInt) -> Option<(f64, f64)> {
    let (lc, rest) = c.split_last()?;
    let d = rest.len();
    // Scaled by N^d: U·N^d = Σ|cᵢ|N^i, L·N^d = |c_d|N^d − Σ_{i<d}|cᵢ|N^i.
    let mut power = BigInt::one();
    let mut rest_sum = BigInt::zero();
    for ci in rest {
        rest_sum += ci.abs() * &power;
        power *= n;
    }
    let top = lc.abs() * &power;
    let upper = &top + &rest_sum;
    let lower = top - rest_sum;
    let scale = d as f64 * lg_big(n)?.1;
    let scale_low = d as f64 * lg_big(n)?.0;
    let (ll, _) = lg_big(&lower)?;
    let (_, uh) = lg_big(&upper)?;
    Some((down(ll - scale), up(uh - scale_low)))
}

/// The addends of a summand (`list`) as dominated terms, and the index `N`
/// from which their envelopes and ratio bounds hold; `None` when an addend
/// is not of that form.
fn analyse(
    arena: &Arena,
    list: &[ExprId],
    var: ExprId,
    lo: &BigInt,
    part: &mut PartFn<'_>,
) -> Option<(Vec<Dominated>, BigInt, bool)> {
    // The shapes of every node (see `shape_of`).
    let mut shapes: FxHashMap<ExprId, Shape> = FxHashMap::default();
    for &root in list {
        for id in walk::post_order_ids(arena, root) {
            if shapes.contains_key(&id) {
                continue;
            }
            if let Some(s) = shape_of(arena, id, var, &shapes) {
                shapes.insert(id, s);
            }
        }
    }
    // N: beyond the roots of every polynomial and ratio in sight.
    let mut n = lo.clone().max(BigInt::from(3));
    for s in shapes.values() {
        let (p, q) = match s {
            Shape::Poly(p) => (p.clone(), Poly::one()),
            Shape::Ratio(p, q, _) => (p.clone(), q.clone()),
            Shape::Free(_) => continue,
        };
        let (pi, qi) = integer_coefficients(&p, &q);
        n = n.max(root_bound(&pi) + 1).max(root_bound(&qi) + 1);
    }
    // A product H·S per addend.
    let mut split: Vec<(Vec<ExprId>, Vec<ExprId>)> = Vec::with_capacity(list.len());
    for &addend in list {
        let mut a = addend;
        while let ExprNode::Neg(c) = arena.node(a) {
            a = *c;
        }
        let factors: Vec<ExprId> = match arena.node(a) {
            ExprNode::Mul(children) => children.to_vec(),
            _ => vec![a],
        };
        let (mut h, mut s) = (Vec::new(), Vec::new());
        for f in factors {
            if exponential_factor(arena, f, &shapes).is_some() {
                h.push(f);
                continue;
            }
            match shapes.get(&f) {
                Some(Shape::Ratio(_, _, true)) => return None,
                Some(_) => h.push(f),
                None => s.push(f),
            }
        }
        split.push((h, s));
    }
    let mut analysis = Analysis {
        arena,
        var,
        n,
        shapes,
        facts: FxHashMap::default(),
        part,
    };
    let mut real = true;
    let mut out = Vec::with_capacity(split.len());
    for (h, s) in split {
        // H's ratio, as in `shape_of` for a product, and the modulus of the
        // constant ratio of its exponential factors.
        let (mut num, mut den) = (Poly::one(), Poly::one());
        let mut log_c = 0.0f64;
        for f in &h {
            if let Some(exp) = exponential_factor(arena, *f, &analysis.shapes) {
                log_c = up(log_c + analysis.exponential_ratio(exp)?);
                continue;
            }
            match analysis.shapes.get(f)? {
                Shape::Free(_) => {}
                Shape::Poly(p) => {
                    num = capped(num.mul(&shifted(p)))?;
                    den = capped(den.mul(p))?;
                }
                Shape::Ratio(p, q, _) => {
                    num = capped(num.mul(p))?;
                    den = capped(den.mul(q))?;
                }
            }
        }
        let (p, q) = integer_coefficients(&num, &den);
        let (dp, dq) = (p.len().checked_sub(1)?, q.len().checked_sub(1)?);
        // Geometric decay or faster: deg P < deg Q, or |C|·|lc P/lc Q| < 1.
        let limit = lg_big(&p[dp].abs())?.1 - lg_big(&q[dq].abs())?.0 + log_c;
        if dp > dq || (dp == dq && limit >= -1e-9) {
            return None;
        }
        // Bits gained per term in the end (factorial decay: unbounded).
        let rate = if dp < dq { f64::INFINITY } else { -limit };
        let mut env = Envelope::constant(0.0);
        for &f in &s {
            analysis.facts_of(f)?;
            env = env.times(analysis.envelope(f)?);
        }
        for &f in h.iter().chain(&s) {
            analysis.facts_of(f)?;
            real &= analysis.facts.get(&f).is_some_and(|x| x.real);
        }
        out.push(Dominated {
            factors: h,
            p,
            q,
            log_c,
            rate,
            env,
        });
    }
    Some((out, analysis.n, real))
}

/// `log₂` of a bound on `Σ_{k ≥ m} |Hᵢ(k)·Sᵢ(k)|` over the addends, for
/// `m ≥ N`; `None` while `ργ ≥ 1` or without a value of `H(m)`.
fn dominated_tail(addends: &[Dominated], m: &BigInt, part: &mut PartFn<'_>) -> Option<f64> {
    let mf = m.to_f64()?;
    let mut total = f64::NEG_INFINITY;
    for d in addends {
        let (pp, qm) = tail_parts(&d.p, &d.q, m)?;
        let lr = if pp.is_zero() {
            f64::NEG_INFINITY
        } else {
            up(lg_big(&pp)?.1 - lg_big(&qm)?.0 + d.log_c)
        };
        let lrg = up(lr + d.env.growth(mf));
        if lrg.is_nan() || lrg >= -1e-9 {
            return None;
        }
        // 1/(1 − 2^lrg).
        let geometric = up(-(-(lrg.exp2())).ln_1p() * std::f64::consts::LOG2_E);
        let mut lh = 0.0f64;
        for &f in &d.factors {
            let (v, b) = part(f, m, AUX_PREC).ok()?;
            lh = up(lh + lg_upper(&v, b)?);
        }
        total = accuracy::lsum(total, up(lh + d.env.at(mf) + geometric));
    }
    Some(total)
}

/// The refusal of a series whose slowest addend gains too few bits per term
/// to reach `prec` within [`MAX_DOMINATED_TERMS`] (`λᵏ` with `λ` within
/// `10⁻³` of 1), decided before summing rather than after the last term.
fn too_slow(
    addends: &[Dominated],
    prec: usize,
    what: impl FnOnce() -> String,
) -> Option<SymplexError> {
    let rate = addends.iter().map(|d| d.rate).fold(f64::INFINITY, f64::min);
    let needed = prec as f64 / rate;
    (needed > MAX_DOMINATED_TERMS as f64).then(|| {
        unevaluable(format!(
            "{}: its terms decay by a factor of only 2^({:.3e}) each; more than \
             {MAX_DOMINATED_TERMS} terms would be needed",
            what(),
            -rate
        ))
    })
}

/// `Σ_{k ≥ lo} body` for a summand whose addends are dominated (see above):
/// `None` when they are not.
pub(super) fn dominated_sum(
    arena: &Arena,
    body: ExprId,
    var: ExprId,
    lo: &BigInt,
    prec: usize,
    rm: RoundingMode,
    part: &mut PartFn<'_>,
) -> Option<Result<(Complex, Bound), SymplexError>> {
    let list: Vec<ExprId> = match arena.node(body) {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![body],
    };
    let (addends, n, real) = analyse(arena, &list, var, lo, part)?;
    if let Some(err) = too_slow(&addends, prec, || {
        format!("infinite sum of `{}`", arena.display(body))
    }) {
        return Some(Err(err));
    }
    let mut sum = super::SumAcc::new(prec);
    let mut k = lo.clone();
    let mut peak = f64::NEG_INFINITY;
    let mut steps: u64 = 0;
    // The bound on the rest costs an evaluation of the factors of `H`: it is
    // checked every few terms, less often as the sum grows.
    let mut check: u64 = 0;
    loop {
        if k >= n && steps >= check {
            check = steps + (steps / 16).max(4);
            if let Some(t) = dominated_tail(&addends, &k, part)
                && t <= peak - prec as f64 - 8.0
            {
                let tail = if real { Bound::real(t) } else { Bound::both(t) };
                sum.add(&crate::base::bigcomplex::c_zero(prec), tail, rm);
                tracing::debug!(steps, tail = t, "evalf: dominated series summed");
                return Some(Ok(sum.finish()));
            }
        }
        if steps >= MAX_DOMINATED_TERMS {
            return Some(Err(unevaluable(format!(
                "infinite sum of `{}`: more than {MAX_DOMINATED_TERMS} terms before the \
                 bound on the rest is below the precision",
                arena.display(body)
            ))));
        }
        let (t, e) = match part(body, &k, prec) {
            Ok(v) => v,
            Err(err) => return Some(Err(err)),
        };
        sum.add(&t, e, rm);
        peak = peak
            .max(accuracy::lg_abs(&t))
            .max(accuracy::lg_abs(&sum.value));
        k += 1;
        steps += 1;
    }
}

/// `∏_{k ≥ lo} body` for `body = 1 + a(k)` whose `a` has dominated addends
/// (see above): `None` when it has not.
pub(super) fn dominated_product(
    arena: &Arena,
    body: ExprId,
    var: ExprId,
    lo: &BigInt,
    prec: usize,
    rm: RoundingMode,
    part: &mut PartFn<'_>,
) -> Option<Result<(Complex, Bound), SymplexError>> {
    let ExprNode::Add(children) = arena.node(body) else {
        return None;
    };
    let one = children
        .iter()
        .position(|c| arena.as_num(*c).is_some_and(|q| q.is_one()))?;
    let rest: Vec<ExprId> = children
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != one)
        .map(|(_, &c)| c)
        .collect();
    let (addends, n, real) = analyse(arena, &rest, var, lo, part)?;
    if let Some(err) = too_slow(&addends, prec, || {
        format!("infinite product of `{}`", arena.display(body))
    }) {
        return Some(Err(err));
    }
    // The partial product at `wp` bits and a bound on its relative error:
    // `Σ rᵢ` over the factors' relative errors `rᵢ = eᵢ/(|vᵢ| − eᵢ)` and a
    // rounding of `2^(3−wp)` per complex product, so that the product of the
    // true factors is within `e^R − 1` of it.  (The bound of products of
    // `evalf`, per step from the factors' binary exponents, doubles at
    // every factor in `[1, 2)`: a product of hundreds of factors near 1 had
    // a bound of `2^9` around a value of 2.)
    let wp = prec + 32;
    let mut product = c_one_at(wp);
    let mut rel = 0.0f64;
    let mut unknown = false;
    let mut k = lo.clone();
    let mut steps: u64 = 0;
    let mut check: u64 = 0;
    loop {
        let due = k >= n && steps >= check;
        if due {
            check = steps + (steps / 16).max(4);
        }
        if due
            && let Some(t) = dominated_tail(&addends, &k, part)
            && t <= -2.0
        {
            // The rest of the product is within e^(2T) − 1 of 1.
            let tail = up_linear((2.0 * t.exp2()).exp_m1());
            let total = up_linear(rel + steps as f64 * (3.0 - wp as f64).exp2() + tail);
            if tail.log2() <= -(prec as f64) - 8.0 {
                if unknown || total >= 0.5 {
                    return Some(Ok((product, Bound::UNKNOWN)));
                }
                // e^R − 1 ≤ R·e^R.
                let factor = up(total.log2() + total * std::f64::consts::LOG2_E);
                let e = up(accuracy::lg_abs(&product) + factor);
                let bound = if real { Bound::real(e) } else { Bound::both(e) };
                tracing::debug!(steps, tail = t, "evalf: dominated product multiplied");
                return Some(Ok((product, bound)));
            }
        }
        if steps >= MAX_DOMINATED_TERMS {
            return Some(Err(unevaluable(format!(
                "infinite product of `{}`: more than {MAX_DOMINATED_TERMS} factors before \
                 the bound on the rest is below the precision",
                arena.display(body)
            ))));
        }
        let (f, e) = match part(body, &k, prec) {
            Ok(v) => v,
            Err(err) => return Some(Err(err)),
        };
        if accuracy::mag(&f).is_none() && e.is_exact() {
            // An exact zero factor; the rest converges.
            return Some(Ok((crate::base::bigcomplex::c_zero(prec), Bound::EXACT)));
        }
        let je = e.joint();
        if !accuracy::is_exact(je) {
            let low = accuracy::lg_abs_low(&f);
            if accuracy::is_unknown(je) || !low.is_finite() || je >= low - 1.0 {
                unknown = true;
            } else {
                // e/(|v| − e) ≤ 2e/|v| once e ≤ |v|/2.
                rel = up_linear(rel + (je - low + 1.0).exp2());
            }
        }
        product = c_mul(&product, &f, wp, rm);
        k += 1;
        steps += 1;
    }
}

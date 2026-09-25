//! Infinite sums of hypergeometric terms.
//!
//! `Σ_{k ≥ a} t(k)` is evaluated when the term ratio `t(k+1)/t(k) =
//! P(k)/Q(k)` is a rational function of `k` with rational coefficients,
//! recognised structurally by [`term_ratio`]: products and integer powers
//! of polynomials in `k`, factorials, `Γ` and binomials of arguments linear
//! in `k` with integer slopes, and `c^(m·k + b)` for a rational `c` and an
//! integer `m`.
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
//! Polynomial convergence (`Σ 1/k²`) needs an extrapolation with a
//! rigorous remainder (Euler–Maclaurin, or SymPy's unbounded Richardson
//! extrapolation) that is not implemented: such sums are refused
//! ([`SymplexError::Unevaluable`]), as are sums whose term is not
//! recognised as hypergeometric; divergent sums are
//! [`SymplexError::Divergent`].

use astro_float::{BigFloat, RoundingMode};
use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};
use rustc_hash::FxHashMap;

use super::accuracy::{self, ErrExp};
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
    dyn FnMut(&BigInt, usize) -> Result<(Complex, ErrExp), SymplexError> + 'a;

/// The shape of a sub-expression of the summand, as a function of the
/// index `k`.
#[derive(Clone, Debug)]
enum Shape {
    /// Free of `k` (ratio 1); `Some` when it is a rational number.
    Free(Option<Q>),
    /// A polynomial in `k` with rational coefficients.
    Poly(Poly),
    /// A term with ratio `t(k+1)/t(k) = P(k)/Q(k)`.
    Ratio(Poly, Poly),
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
                    Shape::Ratio(p, q) => {
                        poly = None;
                        num = capped(num.mul(p))?;
                        den = capped(den.mul(q))?;
                    }
                }
            }
            match poly {
                Some(p) => Shape::Poly(p),
                None => Shape::Ratio(num, den),
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
                Shape::Ratio(Poly::constant(c.pow(m)), Poly::one())
            }
            (base, Shape::Free(Some(n))) if n.is_integer() => {
                let n = n.to_integer().to_i64()?;
                if n == 0 || n.abs() > 4 * MAX_SHIFT {
                    return None;
                }
                let m = usize::try_from(n.unsigned_abs()).ok()?;
                match base {
                    Shape::Poly(p) if n > 0 => Shape::Poly(capped(p.pow(m))?),
                    Shape::Poly(p) => Shape::Ratio(capped(p.pow(m))?, capped(shifted(p).pow(m))?),
                    Shape::Ratio(p, q) if n > 0 => {
                        Shape::Ratio(capped(p.pow(m))?, capped(q.pow(m))?)
                    }
                    Shape::Ratio(p, q) => Shape::Ratio(capped(q.pow(m))?, capped(p.pow(m))?),
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
            Shape::Ratio(num, den)
        }
        // Γ(x) = (x − 1)!.
        ExprNode::Gamma(a) => {
            let (s, x) = linear(shapes.get(a)?)?;
            if s <= 0 {
                return None;
            }
            let (num, den) = factorial_step(&plus(&x, -1), s);
            Shape::Ratio(num, den)
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
            Shape::Ratio(capped(n1.mul(&d2).mul(&d3))?, capped(d1.mul(&n2).mul(&n3))?)
        }
        _ => return None,
    };
    Some(shape)
}

/// The ratio `t(k+1)/t(k) = P(k)/Q(k)` of the summand `body` in the index
/// `var`, when it is recognised as a hypergeometric term (see the module
/// documentation).
fn term_ratio(arena: &Arena, body: ExprId, var: ExprId) -> Option<(Poly, Poly)> {
    let mut shapes: FxHashMap<ExprId, Shape> = FxHashMap::default();
    for id in walk::post_order_ids(arena, body) {
        let shape = shape_of(arena, id, var, &shapes)?;
        shapes.insert(id, shape);
    }
    match shapes.remove(&body)? {
        Shape::Free(_) => Some((Poly::one(), Poly::one())),
        Shape::Poly(p) => Some((shifted(&p), p)),
        Shape::Ratio(p, q) => Some((p, q)),
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
) -> Result<(Complex, ErrExp), SymplexError> {
    let (p_ratio, q_ratio) = term_ratio(arena, body, var).ok_or_else(|| {
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

    let (t0, e0) = term(&start, prec)?;
    if accuracy::mag(&t0).is_none() && accuracy::is_exact(e0) {
        // t(N₀) = 0, and P, Q do not vanish beyond N₀: every later term is 0.
        return Ok(sum.finish());
    }
    if !p.is_empty() {
        classify(&p, &q, arena, body)?;
    }

    // F = Σ u_j at wp bits.
    let wp = prec + 64;
    let (f, f_err) = normalised_series(&p, &q, &start, prec, wp, rm)?;

    // Sanity check of the ratio: t(N₀ + 1) = t(N₀)·P(N₀)/Q(N₀).
    let (t1, e1) = term(&(&start + 1), prec)?;
    let r0 = bigint_to_bigfloat(&eval_int(&p, &start), wp).div(
        &bigint_to_bigfloat(&eval_int(&q, &start), wp),
        wp,
        rm,
    );
    let r0 = (r0, BigFloat::new(wp));
    let predicted = c_mul(&t0, &r0, wp, rm);
    let e_pred = accuracy::product_error([(&t0, e0), (&r0, accuracy::rounding(&r0, wp))]);
    let slack = e1
        .max(e_pred)
        .max(accuracy::rounding(&predicted, prec))
        .saturating_add(4);
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
    let series_err = accuracy::product_error([(&t0, e0), (&f, f_err)])
        .max(accuracy::rounding(&series, prec).saturating_add(1));
    sum.add(&series, series_err, rm);
    Ok(sum.finish())
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
                break tail;
            }
        }
        if steps >= MAX_SERIES_TERMS {
            break tail_factor.map_or(accuracy::UNKNOWN, |tf| um + tf + 1);
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
    let recurrence = mag(&abs_sum).unwrap_or(0) + n + 2 - wp_i;
    let rounding = peak + n - wp_i;
    let err = recurrence.max(rounding).max(tail).saturating_add(1);
    tracing::debug!(steps, err, "evalf: hypergeometric series summed");
    Ok((f, err))
}

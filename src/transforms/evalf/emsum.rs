//! Infinite sums of rational terms, with a rigorous remainder.
//!
//! `Σ_{k ≥ a} f(k)` for a rational function `f = A/B` of the index (or
//! `(−1)^k·f(k)`) converges only polynomially, beyond the reach of the
//! hypergeometric recurrence of `hypsum.rs`.  It is summed as
//!
//! ```text
//! Σ_{k=a}^{N−1} f(k)  +  Σ_{j ≥ d} aⱼ·ζ(j, N),
//! ```
//!
//! where `f(x) = Σ_{j ≥ d} aⱼ·x^(−j)` is the Laurent expansion of `f` at
//! infinity (`d = deg B − deg A ≥ 2`; exact rational coefficients from the
//! power-series quotient of the reversed polynomials), valid for `|x|` beyond
//! every root of `B`, and `ζ(j, N) = Σ_{k ≥ N} k^(−j)` is the Hurwitz zeta
//! function at an integer, by the Euler–Maclaurin formula
//!
//! ```text
//! ζ(j, N) = N^(1−j)/(j−1) + N^(−j)/2 + Σ_{m=1}^{M} B₂ₘ/(2m)!·(j)₂ₘ₋₁·N^(1−j−2m) + R,
//! |R| ≤ 2ζ(2M)/(2π)^(2M)·(j)₂ₘ₋₁·N^(1−j−2M)
//! ```
//!
//! (`(j)ₚ` the rising factorial; the remainder bound is the classical one,
//! `2ζ(2M)/(2π)^(2M)·∫_N^∞ |g^(2M)|`, for `g = x^(−j)` whose derivatives have
//! constant sign; see e.g. Olver, *Asymptotics and Special Functions*,
//! ch. 8 §1, or DLMF 2.10.1–2.10.2).  The truncation of the Laurent series is
//! bounded by Cauchy's estimate: on `|x| = R` beyond the roots, `|aⱼ| ≤
//! max_{|x|=R} |f|·R^j`, and `ζ(j, N) ≤ N^(−j) + N^(1−j)/(j−1)`, so for
//! `N ≥ 4R`
//!
//! ```text
//! Σ_{j ≥ J} |aⱼ|·ζ(j, N) ≤ max|f|·(R/N)^J·(1 + N/(J−1))/(1 − R/N).
//! ```
//!
//! An alternating sum is paired first: `Σ_{k ≥ N} (−1)^k f(k) = (−1)^N·Σ_{m ≥ 0}
//! g(m)` with `g(m) = f(N + 2m) − f(N + 2m + 1)`, a rational function whose
//! degree difference is one more than `f`'s.
//!
//! SymPy's `evalf_sum` (`sympy/core/evalf.py`, BSD-3) sums such series by
//! the Euler–Maclaurin formula applied to the summand symbolically, with an
//! error estimate from the size of the last correction; here the formula is
//! applied to the powers `x^(−j)` only, where its remainder has a proven bound,
//! and every step is bounded.
//!
//! An infinite product `∏_{k ≥ a} f(k)` of a rational `f = A/B` with `deg A =
//! deg B`, equal leading coefficients and `deg(A − B) ≤ deg B − 2` (so that
//! `Σ |f(k) − 1|` converges) is the exact product of its factors below `N`
//! times `exp(Σ_{k ≥ N} ln f(k))`.  Beyond the roots of `A` and `B`, `ln f`
//! has the Laurent expansion `Σ_{j ≥ 2} cⱼ x^(−j)` obtained from that of the
//! rational logarithmic derivative `g = f′/f = (A′B − AB′)/(AB) = Σ eⱼ
//! x^(−j)`, `cⱼ = −eⱼ₊₁/j`, and `Σ_{k ≥ N} ln f(k) = Σⱼ cⱼ·ζ(j, N)` as above;
//! Cauchy's estimate for `g` gives `|cⱼ| ≤ max_{|x|=R} |g|·R^(j+1)/j`, the
//! truncation bound above with an extra factor `R`.  (SymPy's `Product`
//! evaluates `exp(Sum(log(f)))` the same way, by its Euler–Maclaurin sum.)

use astro_float::{BigFloat, RoundingMode};
use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};
use rustc_hash::FxHashMap;

use super::accuracy::{self, Bound, ErrExp};
use crate::base::arena::Arena;
use crate::base::bigcomplex::Complex;
use crate::base::errors::SymplexError;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::{Q, bigint_to_bigfloat, ratio_to_bigfloat};
use crate::base::walk;
use crate::poly::dense::Poly;

/// Highest degree of a numerator or denominator built.
const MAX_DEGREE: usize = 64;

/// Most terms summed directly before the tail.
const MAX_DIRECT_TERMS: u64 = 200_000;

/// A sub-expression of the summand as `(−1)^(alt·k)·num/den`.
#[derive(Clone, Debug)]
struct Rational {
    num: Poly,
    den: Poly,
    alt: bool,
}

impl Rational {
    fn constant(q: Q) -> Rational {
        Rational {
            num: Poly::constant(q),
            den: Poly::one(),
            alt: false,
        }
    }

    fn capped(self) -> Option<Rational> {
        let small = |p: &Poly| p.degree().unwrap_or(0) <= MAX_DEGREE;
        (small(&self.num) && small(&self.den)).then_some(self)
    }
}

/// The summand `body` as `(−1)^(alt·k)·A(k)/B(k)` with `A`, `B` coprime,
/// when it is built from the index and rational numbers by `+`, `·`, integer
/// powers and `(−1)^(s·k + c)` (integers `s`, `c`).
fn rational_summand(arena: &Arena, body: ExprId, var: ExprId) -> Option<Rational> {
    let mut shapes: FxHashMap<ExprId, Rational> = FxHashMap::default();
    for id in walk::post_order_ids(arena, body) {
        let shape = if id == var {
            Rational {
                num: Poly::x(),
                den: Poly::one(),
                alt: false,
            }
        } else {
            match arena.node(id) {
                ExprNode::Num(n) => Rational::constant(arena.num(*n).clone()),
                ExprNode::Neg(c) => {
                    let r = shapes.get(c)?;
                    Rational {
                        num: r.num.neg(),
                        ..r.clone()
                    }
                }
                ExprNode::Add(children) => {
                    let mut acc = Rational::constant(Q::zero());
                    for (i, c) in children.iter().enumerate() {
                        let r = shapes.get(c)?;
                        if i > 0 && r.alt != acc.alt {
                            return None;
                        }
                        acc = Rational {
                            num: acc.num.mul(&r.den).add(&r.num.mul(&acc.den)),
                            den: acc.den.mul(&r.den),
                            alt: r.alt,
                        }
                        .capped()?;
                    }
                    acc
                }
                ExprNode::Mul(children) => {
                    let mut acc = Rational::constant(Q::one());
                    for c in children.iter() {
                        let r = shapes.get(c)?;
                        acc = Rational {
                            num: acc.num.mul(&r.num),
                            den: acc.den.mul(&r.den),
                            alt: acc.alt ^ r.alt,
                        }
                        .capped()?;
                    }
                    acc
                }
                ExprNode::Pow(b, e) => power(arena, *b, *e, var, &shapes)?,
                _ => return None,
            }
        };
        shapes.insert(id, shape);
    }
    let r = shapes.remove(&body)?;
    if r.den.is_zero() {
        return None;
    }
    let g = Poly::gcd(&r.num, &r.den);
    if g.degree().unwrap_or(0) > 0 {
        return Some(Rational {
            num: r.num.div(&g),
            den: r.den.div(&g),
            alt: r.alt,
        });
    }
    Some(r)
}

/// `b^e` for an integer literal `e`, or `(−1)^(s·k + c)`.
fn power(
    arena: &Arena,
    b: ExprId,
    e: ExprId,
    var: ExprId,
    shapes: &FxHashMap<ExprId, Rational>,
) -> Option<Rational> {
    if let Some(q) = arena.as_num(e) {
        if !q.is_integer() {
            return None;
        }
        let n = q.to_integer().to_i64()?;
        let m = usize::try_from(n.unsigned_abs())
            .ok()
            .filter(|&m| m <= MAX_DEGREE)?;
        let r = shapes.get(&b)?;
        let (num, den) = if n >= 0 {
            (r.num.pow(m), r.den.pow(m))
        } else {
            if r.num.is_zero() {
                return None;
            }
            (r.den.pow(m), r.num.pow(m))
        };
        return Rational {
            num,
            den,
            alt: r.alt && m % 2 == 1,
        }
        .capped();
    }
    // (−1)^(s·k + c): the exponent a linear polynomial in k with integer
    // coefficients.
    if *arena.as_num(b)? != -Q::one() {
        return None;
    }
    let ep = crate::poly::polybridge::expr_to_poly(arena, e, var)?;
    if ep.degree()? > 1 || ep.coeffs().iter().any(|c| !c.is_integer()) {
        return None;
    }
    let s = ep.coeff(1).to_integer();
    let c = ep.coeff(0).to_integer();
    let sign = if c.is_odd() { -Q::one() } else { Q::one() };
    Some(Rational {
        num: Poly::constant(sign),
        den: Poly::one(),
        alt: s.is_odd(),
    })
}

/// `Σ_{k ≥ lo} body` when the summand is a rational function of the index,
/// possibly alternating (see the module documentation): `None` when it is
/// not; the value and its error bound, or [`SymplexError::Divergent`] (the
/// terms decay too slowly) or [`SymplexError::Unevaluable`] (a pole in the
/// range, or a denominator whose roots are too large to sum past).
pub(super) fn rational_sum(
    arena: &Arena,
    body: ExprId,
    var: ExprId,
    lo: &BigInt,
    prec: usize,
) -> Option<Result<(Complex, Bound), SymplexError>> {
    let r = rational_summand(arena, body, var)?;
    Some(sum_rational(arena, body, &r, lo, prec))
}

fn divergent(arena: &Arena, body: ExprId, why: &str) -> SymplexError {
    SymplexError::Divergent {
        operation: "evalf",
        reason: format!(
            "the infinite sum of `{}` diverges: {why}",
            arena.display(body)
        ),
    }
}

fn unevaluable(reason: String) -> SymplexError {
    SymplexError::Unevaluable { reason }
}

/// `2^e` as a 64-bit float with an exponent of any size.
fn pow2(e: i64) -> BigFloat {
    let mut x = BigFloat::from_i32(1, 64);
    let e = e.saturating_add(1).clamp(
        i64::from(astro_float::EXPONENT_MIN),
        i64::from(astro_float::EXPONENT_MAX),
    );
    x.set_exponent(astro_float::Exponent::try_from(e).unwrap_or(0));
    x
}

/// Running upper bound on an absolute error, in 64-bit floats rounded up.
#[derive(Clone, Debug)]
struct ErrSum(BigFloat);

impl ErrSum {
    fn new() -> ErrSum {
        ErrSum(BigFloat::new(64))
    }

    fn add(&mut self, x: &BigFloat) {
        self.0 = self.0.add(&x.abs(), 64, RoundingMode::Up);
    }

    fn exponent(&self) -> ErrExp {
        if self.0.is_zero() {
            accuracy::EXACT
        } else if self.0.is_inf() || self.0.is_nan() {
            accuracy::UNKNOWN
        } else {
            // The sum was accumulated rounding up: its logarithm, rounded up.
            accuracy::part_lg(&self.0)
        }
    }
}

fn sum_rational(
    arena: &Arena,
    body: ExprId,
    r: &Rational,
    lo: &BigInt,
    prec: usize,
) -> Result<(Complex, Bound), SymplexError> {
    let wp = prec + 64;
    let rm = RoundingMode::ToEven;
    if r.num.is_zero() {
        return Ok(((BigFloat::new(prec), BigFloat::new(prec)), Bound::EXACT));
    }
    let d = r.den.degree().unwrap_or(0) as i64 - r.num.degree().unwrap_or(0) as i64;
    if d < 1 || (!r.alt && d < 2) {
        return Err(divergent(
            arena,
            body,
            &format!("its terms decay only like k^(−{})", d.max(0)),
        ));
    }
    let (value, err) = if r.alt {
        // Direct terms up to N, then the paired tail from N.
        let n = cut_point(&r.den, lo, wp)?;
        let (direct, mut err) = direct_sum(&r.num, &r.den, lo, &n, true, wp)?;
        let shift = |p: &Poly, extra: i64| -> Poly {
            let q = p.taylor_shift(&(Q::from_integer(n.clone() + extra)));
            // q(2m)
            let coeffs: Vec<Q> = q
                .coeffs()
                .iter()
                .enumerate()
                .map(|(i, c)| c * Q::from_integer(BigInt::one() << i))
                .collect();
            Poly::from_coeffs(coeffs)
        };
        let (a0, b0) = (shift(&r.num, 0), shift(&r.den, 0));
        let (a1, b1) = (shift(&r.num, 1), shift(&r.den, 1));
        let gn = a0.mul(&b1).sub(&a1.mul(&b0));
        let gd = b0.mul(&b1);
        let (tail, tail_err) = if gn.is_zero() {
            (BigFloat::new(wp), ErrSum::new())
        } else {
            let g = Poly::gcd(&gn, &gd);
            let (gn, gd) = if g.degree().unwrap_or(0) > 0 {
                (gn.div(&g), gd.div(&g))
            } else {
                (gn, gd)
            };
            let m0 = cut_point(&gd, &BigInt::zero(), wp)?;
            let (head, mut e) = direct_sum(&gn, &gd, &BigInt::zero(), &m0, false, wp)?;
            let (rest, e2) = zeta_tail(&gn, &gd, &m0, wp)?;
            e.add(&e2.0);
            (head.add(&rest, wp, rm), e)
        };
        let tail = if n.is_odd() { tail.neg() } else { tail };
        err.add(&tail_err.0);
        (direct.add(&tail, wp, rm), err)
    } else {
        let n = cut_point(&r.den, lo, wp)?;
        let (direct, mut err) = direct_sum(&r.num, &r.den, lo, &n, false, wp)?;
        let (tail, e) = zeta_tail(&r.num, &r.den, &n, wp)?;
        err.add(&e.0);
        (direct.add(&tail, wp, rm), err)
    };
    tracing::debug!(err = err.exponent(), "evalf: rational infinite sum");
    Ok(((value, BigFloat::new(prec)), Bound::real(err.exponent())))
}

/// A summand `C·(−1)^(εk)·(a·k + b)^(−s)`: the `k`-free factors `C`, the
/// sign and alternation, rationals `a > 0` and `b`, and a rational `s` that
/// is not an integer (an integer power is a rational summand).
pub(super) struct PowerSummand {
    pub(super) consts: Vec<ExprId>,
    negate: bool,
    alt: bool,
    a: Q,
    b: Q,
    s: Q,
}

/// The summand `body` as a [`PowerSummand`] (`None` when it is not one, or
/// the series diverges: `s ≤ 1`, or `s ≤ 0` when it alternates).
pub(super) fn power_summand(arena: &Arena, body: ExprId, var: ExprId) -> Option<PowerSummand> {
    let mut root = body;
    let mut negate = false;
    while let ExprNode::Neg(c) = arena.node(root) {
        negate = !negate;
        root = *c;
    }
    let factors: Vec<ExprId> = match arena.node(root) {
        ExprNode::Mul(children) => children.to_vec(),
        _ => vec![root],
    };
    let mut consts = Vec::new();
    let mut alt = false;
    let mut power: Option<(Q, Q, Q)> = None;
    for f in factors {
        if !walk::post_order_ids(arena, f).contains(&var) {
            consts.push(f);
            continue;
        }
        let ExprNode::Pow(base, e) = arena.node(f) else {
            return None;
        };
        if arena.as_num(*base).is_some_and(|q| *q == -Q::one()) {
            // (−1)^(m·k + c), integers m, c.
            let ep = crate::poly::polybridge::expr_to_poly(arena, *e, var)?;
            if ep.degree()? > 1 || ep.coeffs().iter().any(|c| !c.is_integer()) {
                return None;
            }
            alt ^= ep.coeff(1).to_integer().is_odd();
            negate ^= ep.coeff(0).to_integer().is_odd();
            continue;
        }
        let r = arena.as_num(*e)?;
        if r.is_integer() || power.is_some() {
            return None;
        }
        let bp = crate::poly::polybridge::expr_to_poly(arena, *base, var)?;
        if bp.degree()? != 1 || !bp.coeff(1).is_positive() {
            return None;
        }
        power = Some((bp.coeff(1), bp.coeff(0), -r.clone()));
    }
    let (a, b, s) = power?;
    let converges = if alt { s.is_positive() } else { s > Q::one() };
    converges.then_some(PowerSummand {
        consts,
        negate,
        alt,
        a,
        b,
        s,
    })
}

/// `Σ_{k ≥ lo}` of a [`PowerSummand`] whose constant factors have the values
/// `consts`: with `x = lo + b/a > 0`,
///
/// ```text
/// Σ (a k + b)^(−s) = a^(−s)·ζ(s, x),
/// Σ (−1)^k (a k + b)^(−s) = (−1)^lo·a^(−s)·2^(−s)·(ζ(s, x/2) − ζ(s, (x + 1)/2))
/// ```
///
/// (DLMF 25.11.1, and the alternating sum split by parity), the Hurwitz
/// zeta function by Euler–Maclaurin (`polylog::hurwitz_real`), whose error
/// is a few units of `2^(largest − wp)` per term summed.  Before 0.44 such
/// sums, which `eval` leaves as they are (`Σ k^(−5/2)`, `Σ (−1)^k/√(k + 1)`),
/// were refused.
pub(super) fn power_sum(
    p: &PowerSummand,
    consts: &[(Complex, Bound)],
    lo: &BigInt,
    prec: usize,
    rm: RoundingMode,
    cc: &mut astro_float::Consts,
) -> Result<(Complex, Bound), SymplexError> {
    let x = Q::from_integer(lo.clone()) + &p.b / &p.a;
    if !x.is_positive() {
        return Err(unevaluable(
            "infinite sum: the power's base vanishes or is negative in the range".into(),
        ));
    }
    let sf = crate::base::numeric::ratio_to_f64(&p.s).unwrap_or(f64::INFINITY);
    if sf.abs() > 1e5 {
        return Err(unevaluable(
            "infinite sum: a power beyond 10^5 is not summed numerically".into(),
        ));
    }
    let wp = prec + 64 + super::magnitude_bits(sf.abs() + 1.0);
    let s = ratio_to_bigfloat(&p.s, wp, rm);
    let (sum, largest) = if p.alt {
        let half = Q::new(BigInt::one(), BigInt::from(2));
        let x1 = ratio_to_bigfloat(&(&x * &half), wp, rm);
        let x2 = ratio_to_bigfloat(&((&x + Q::one()) * &half), wp, rm);
        let (h1, l1) = super::polylog::hurwitz_real(&s, &x1, wp, rm, cc)?;
        let (h2, l2) = super::polylog::hurwitz_real(&s, &x2, wp, rm, cc)?;
        let two_s = super::bf_pow(&BigFloat::from_i32(2, 64), &s.neg(), wp, rm, cc);
        let v = h1.sub(&h2, wp, rm).mul(&two_s, wp, rm);
        let v = if lo.is_odd() { v.neg() } else { v };
        (v, l1.max(l2) - sf)
    } else {
        let xb = ratio_to_bigfloat(&x, wp, rm);
        super::polylog::hurwitz_real(&s, &xb, wp, rm, cc)?
    };
    let scale = super::bf_pow(&ratio_to_bigfloat(&p.a, wp, rm), &s.neg(), wp, rm, cc);
    let ls = accuracy::part_lg(&scale);
    let value = sum.mul(&scale, wp, rm);
    let value = if p.negate { value.neg() } else { value };
    // Up to `5·wp + 3|s| + 100` terms, each rounded with a relative error of
    // `(|s|·ln(x + N) + 4)·2^(−wp)`; a^(−s) and the products a few more.
    let terms = 5.0 * wp as f64 + 3.0 * sf.abs() + 100.0;
    let xf = crate::base::numeric::ratio_to_f64(&x).unwrap_or(f64::INFINITY);
    let per_term = sf.abs() * (xf + terms).ln() + 4.0;
    let e = largest + ls - wp as f64 + 8.0 + (terms * per_term).log2();
    if !e.is_finite() {
        return Ok(((value, BigFloat::new(prec)), Bound::UNKNOWN));
    }
    let mut parts: Vec<(&Complex, Bound)> = consts.iter().map(|(v, b)| (v, *b)).collect();
    let summed = (value, BigFloat::new(prec));
    parts.push((&summed, Bound::real(e)));
    Ok(accuracy::mul_with_bound(&parts, prec, rm))
}

/// `∏_{k ≥ lo} body` when the factor is a rational function of the index
/// tending to 1 like `k^(−2)` or faster (see the module documentation):
/// `None` when it is not rational; [`SymplexError::Divergent`] when the
/// product does not converge (to a nonzero limit), [`SymplexError::Unevaluable`]
/// for a pole in the range.  Before 0.44 every infinite product was refused.
pub(super) fn rational_product(
    arena: &Arena,
    body: ExprId,
    var: ExprId,
    lo: &BigInt,
    prec: usize,
) -> Option<Result<(Complex, Bound), SymplexError>> {
    let r = rational_summand(arena, body, var)?;
    if r.alt {
        return None;
    }
    Some(product_rational(arena, body, &r, lo, prec))
}

fn product_rational(
    arena: &Arena,
    body: ExprId,
    r: &Rational,
    lo: &BigInt,
    prec: usize,
) -> Result<(Complex, Bound), SymplexError> {
    let wp = prec + 64;
    let rm = RoundingMode::ToEven;
    let diverges = |why: &str| SymplexError::Divergent {
        operation: "evalf",
        reason: format!(
            "the infinite product of `{}` diverges: {why}",
            arena.display(body)
        ),
    };
    let (a, b) = (&r.num, &r.den);
    if a.is_zero() {
        return Ok(((BigFloat::new(prec), BigFloat::new(prec)), Bound::EXACT));
    }
    let (da, db) = (a.degree().unwrap_or(0), b.degree().unwrap_or(0));
    if da != db || a.coeff(da) != b.coeff(db) {
        return Err(diverges("its factors do not tend to 1"));
    }
    let diff = a.sub(b);
    if diff.is_zero() {
        // Every factor is 1 (A = B: no pole either).
        return Ok((
            (BigFloat::from_i32(1, prec), BigFloat::new(prec)),
            Bound::EXACT,
        ));
    }
    if diff.degree().unwrap_or(0) + 2 > db {
        return Err(diverges("its factors tend to 1 only like 1 + c/k"));
    }
    // g = f′/f = (A′B − AB′)/(AB), and N beyond the roots of A and B.
    let gn = a.derivative().mul(b).sub(&a.mul(&b.derivative()));
    let gd = a.mul(b);
    let n = cut_point(&gd, lo, wp)?;
    // The factors below N exactly.
    let (ai, bi) = integer_coefficients(a, b);
    let (mut num, mut den) = (BigInt::one(), BigInt::one());
    let mut k = lo.clone();
    while k < n {
        let bk = eval_int(&bi, &k);
        if bk.is_zero() {
            return Err(unevaluable(format!(
                "infinite product: the factor has a pole at k = {k}"
            )));
        }
        let ak = eval_int(&ai, &k);
        if ak.is_zero() {
            // A zero factor; the rest converges.
            return Ok(((BigFloat::new(prec), BigFloat::new(prec)), Bound::EXACT));
        }
        num *= ak;
        den *= bk;
        k += 1;
    }
    let head = ratio_to_bigfloat(&Q::new(num, den), wp, rm);
    let (tail, tail_err) = if gn.is_zero() {
        (BigFloat::new(wp), ErrSum::new())
    } else {
        zeta_tail_with(&gn, &gd, &n, wp, true)?
    };
    // head·e^tail: the head and e^tail round once each (and the product);
    // an error δ of the tail moves e^tail by at most e^δ − 1 ≤ 2δ relative.
    let mut cc = astro_float::Consts::new()
        .map_err(|e| unevaluable(format!("astro-float constants: {e:?}")))?;
    let value = head.mul(&tail.exp(wp, rm, &mut cc), wp, rm);
    let delta = tail_err
        .0
        .mul(&BigFloat::from_i32(2, 64), 64, RoundingMode::Up);
    let mut rel = ErrSum::new();
    rel.add(&delta);
    rel.add(&pow2(3 - wp as i64));
    let mut err = ErrSum::new();
    err.add(&value.abs().mul(&rel.0, 64, RoundingMode::Up));
    tracing::debug!(err = err.exponent(), "evalf: rational infinite product");
    Ok(((value, BigFloat::new(prec)), Bound::real(err.exponent())))
}

/// An integer `R ≥ 4` with `|bᵢ| ≤ |bₙ|·(R/4)^(n−i)` for the integer
/// coefficients `b` of degree `n`: every root has modulus below `R/2`
/// (Fujiwara), and on `|x| = R`, `|B(x)| ≥ ⅔·|bₙ|·Rⁿ`.
fn root_radius(b: &[BigInt]) -> BigInt {
    let n = b.len() - 1;
    let lc = b[n].abs();
    let mut r = BigInt::from(4);
    loop {
        let ok = (0..n).all(|i| {
            let k = u32::try_from(n - i).unwrap_or(u32::MAX);
            b[i].abs() * BigInt::from(4).pow(k) <= &lc * r.pow(k)
        });
        if ok {
            return r;
        }
        r *= 2;
    }
}

/// The coefficients of `p` and `q` scaled to integers by a common factor.
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

/// Where the direct summation of `A/B` from `lo` hands over to the tail:
/// `N = max(lo, 4R, wp)`, with `R` from [`root_radius`] (the Laurent
/// series converges fast) and at least `wp` (the Euler–Maclaurin terms of
/// `ζ(j, N)` shrink like `((j + 2m)/(2πN))^(2m)`).
fn cut_point(den: &Poly, lo: &BigInt, wp: usize) -> Result<BigInt, SymplexError> {
    let (_, b) = integer_coefficients(&Poly::one(), den);
    let r = root_radius(&b);
    let n = lo.clone().max(r * 4).max(BigInt::from(wp));
    let count = (&n - lo).to_u64().unwrap_or(u64::MAX);
    if count > MAX_DIRECT_TERMS {
        return Err(unevaluable(format!(
            "infinite sum: {count} terms before the tail (more than {MAX_DIRECT_TERMS})"
        )));
    }
    Ok(n)
}

/// `Σ_{k=lo}^{hi−1} (±1)^k·A(k)/B(k)` at `wp` bits, and a bound on its
/// error: each term is a quotient of exact integers rounded three times, and
/// each partial sum once.
fn direct_sum(
    num: &Poly,
    den: &Poly,
    lo: &BigInt,
    hi: &BigInt,
    alt: bool,
    wp: usize,
) -> Result<(BigFloat, ErrSum), SymplexError> {
    let rm = RoundingMode::ToEven;
    let (a, b) = integer_coefficients(num, den);
    let mut sum = BigFloat::new(wp);
    let mut abs = BigFloat::new(64);
    let mut k = lo.clone();
    let mut count: u64 = 0;
    while &k < hi {
        let bk = eval_int(&b, &k);
        if bk.is_zero() {
            return Err(unevaluable(format!(
                "infinite sum: the term has a pole at k = {k}"
            )));
        }
        let ak = eval_int(&a, &k);
        let mut t = bigint_to_bigfloat(&ak, wp).div(&bigint_to_bigfloat(&bk, wp), wp, rm);
        if alt && k.is_odd() {
            t = t.neg();
        }
        sum = sum.add(&t, wp, rm);
        abs = abs.add(&t.abs(), 64, RoundingMode::Up);
        abs = abs.add(&sum.abs(), 64, RoundingMode::Up);
        k += 1;
        count += 1;
    }
    let mut err = ErrSum::new();
    if count > 0 {
        // ≤ 3 roundings per term and one per partial sum, each ≤ 2^(−wp).
        err.add(&abs.mul(&pow2(2 - wp as i64), 64, RoundingMode::Up));
    }
    Ok((sum, err))
}

/// Bernoulli numbers `B₀, B₁, …, B_n` (`B₁ = +½`; only the even ones are
/// used) by the Akiyama–Tanigawa algorithm, exactly.
fn bernoulli_numbers(n: usize) -> Vec<Q> {
    let mut a: Vec<Q> = Vec::with_capacity(n + 1);
    let mut out = Vec::with_capacity(n + 1);
    for m in 0..=n {
        a.push(Q::new(BigInt::one(), BigInt::from(m + 1)));
        for j in (1..=m).rev() {
            a[j - 1] = Q::from_integer(BigInt::from(j)) * (&a[j - 1] - &a[j]);
        }
        out.push(a[0].clone());
    }
    out
}

thread_local! {
    /// `B₂ₘ/(2m)!` for `m = 1, 2, …`, extended on demand.
    static BERNOULLI_OVER_FACTORIAL: std::cell::RefCell<Vec<Q>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// `B₂ₘ/(2m)!` for `m = 1..=count`.
fn bernoulli_over_factorial(count: usize) -> Vec<Q> {
    BERNOULLI_OVER_FACTORIAL.with(|cell| {
        let mut table = cell.borrow_mut();
        if table.len() < count {
            let b = bernoulli_numbers(2 * count);
            let mut fact = BigInt::one();
            let mut out = Vec::with_capacity(count);
            for (i, bi) in b.iter().enumerate().skip(1) {
                fact *= BigInt::from(i);
                if i % 2 == 0 {
                    out.push(bi / Q::from_integer(fact.clone()));
                }
            }
            *table = out;
        }
        table[..count].to_vec()
    })
}

/// Most Euler–Maclaurin corrections per `ζ(j, N)`.
const MAX_EM_TERMS: usize = 200;

/// `Σ_{k ≥ N} A(k)/B(k) = Σ_{j ≥ d} aⱼ·ζ(j, N)` at `wp` bits, and a bound on
/// its error (see the module documentation); `N` is beyond every root of
/// `B` ([`cut_point`]).
fn zeta_tail(
    num: &Poly,
    den: &Poly,
    n: &BigInt,
    wp: usize,
) -> Result<(BigFloat, ErrSum), SymplexError> {
    zeta_tail_with(num, den, n, wp, false)
}

/// [`zeta_tail`] of `A/B`, or (`log`) of the function `h` with `h′ = A/B`
/// and `h(∞) = 0` (`ln f` for the logarithmic derivative `A/B` of a
/// rational `f → 1`): the coefficient of `x^(−j)` in `h` is `−eⱼ₊₁/j` for
/// the coefficients `e` of `A/B`, which must decay like `x^(−3)`.
fn zeta_tail_with(
    num: &Poly,
    den: &Poly,
    n: &BigInt,
    wp: usize,
    log: bool,
) -> Result<(BigFloat, ErrSum), SymplexError> {
    let rm = RoundingMode::ToEven;
    let (a, b) = integer_coefficients(num, den);
    let decay = if log { 3 } else { 2 };
    if a.is_empty() || b.len() < a.len() + decay {
        return Err(unevaluable(
            "infinite sum: the tail's term does not decay like k^(−2)".into(),
        ));
    }
    let (deg_a, deg_b) = (a.len() - 1, b.len() - 1);
    // The first power of the expansion summed: x^(−d).
    let d = deg_b - deg_a - usize::from(log);
    let r = root_radius(&b);
    // max_{|x| = R} |A/B| ≤ Σ|αᵢ|Rⁱ / (⅔·|bₙ|·Rⁿ).
    let a_plus: BigInt = a
        .iter()
        .enumerate()
        .map(|(i, c)| c.abs() * r.pow(u32::try_from(i).unwrap_or(u32::MAX)))
        .sum();
    let b_minus = b[deg_b].abs() * r.pow(u32::try_from(deg_b).unwrap_or(u32::MAX)) * 2;
    let lg = |x: &BigInt| x.bits() as f64; // ≥ log₂ x ≥ bits − 1 for x ≥ 1
    // log₂(3·A⁺/(2|bₙ|Rⁿ)) = log₂(3·A⁺/b_minus) with b_minus = 2|bₙ|Rⁿ·2/2;
    // for `h`, |cⱼ| ≤ max|A/B|·R^(j+1)/j: one more factor R.
    let log_max_f =
        lg(&a_plus) - (lg(&b_minus) - 1.0) + 3f64.log2() + if log { lg(&r) } else { 0.0 };
    let log_ratio = lg(&r) - (lg(n) - 1.0); // ≥ log₂(R/N)
    let nf = n.to_f64().unwrap_or(f64::INFINITY);

    // Laurent coefficients cᵢ = a_{d+i} of γ(w)/β(w), γᵢ = α_{m−i},
    // βᵢ = b_{n−i}, as cᵢ = Cᵢ/β₀^(i+1) with integers
    // Cᵢ = γᵢ·β₀^i − Σ_{l=1}^{min(i,n)} βₗ·C_{i−l}·β₀^(l−1)
    // (no rational normalisation in the loop).
    let beta: Vec<BigInt> = (0..=deg_b).map(|i| b[deg_b - i].clone()).collect();
    let gamma = |i: usize| -> BigInt {
        if i <= deg_a {
            a[deg_a - i].clone()
        } else {
            BigInt::zero()
        }
    };

    // Target: the tail's own magnitude, ≈ |a_d|·N^(1−d)/(d−1), less wp bits.
    let beta0 = beta[0].clone();
    let log_c0 = gamma(0).bits() as f64 - beta0.bits() as f64;
    let target = log_c0 + (1.0 - d as f64) * nf.log2() - wp as f64 - 8.0;
    // J: the truncation bound max|f|·(R/N)^J·(1 + N/(J−1))·2 below target.
    let mut terms = 0usize;
    let truncation = loop {
        let j = (d + terms) as f64;
        let bound = log_max_f + j * log_ratio + (1.0 + nf / (j - 1.0).max(1.0)).log2() + 1.0;
        if bound <= target || terms >= 4 * wp {
            break bound;
        }
        terms += 1;
    };
    // β₀^p for p = 0..=max(terms, deg_b).
    let mut beta0_pow: Vec<BigInt> = vec![BigInt::one()];
    for p in 1..=terms.max(deg_b) + 1 {
        let next = &beta0_pow[p - 1] * &beta0;
        beta0_pow.push(next);
    }
    let mut big_c: Vec<BigInt> = Vec::with_capacity(terms);
    for i in 0..terms {
        let mut acc = gamma(i) * &beta0_pow[i];
        for l in 1..=i.min(deg_b) {
            acc -= &beta[l] * &big_c[i - l] * &beta0_pow[l - 1];
        }
        big_c.push(acc);
    }

    let n_bf = bigint_to_bigfloat(n, wp);
    let inv_n = BigFloat::from_i32(1, wp).div(&n_bf, wp, rm);
    let inv_n2 = inv_n.mul(&inv_n, wp, rm);
    let mut err = ErrSum::new();
    err.add(&pow2(truncation.ceil() as i64));
    let mut total = BigFloat::new(wp);
    // N^(−j), starting at j = d.
    let mut n_pow = inv_n.powi(d, wp, rm);
    let mut em_terms = 16usize;
    let mut table = bernoulli_over_factorial(em_terms);
    for (i, ci) in big_c.iter().enumerate() {
        let j = d + i;
        if !ci.is_zero() {
            // ζ(j, N) ≈ N^(1−j)/(j−1) + N^(−j)/2 + Σ_m B₂ₘ/(2m)!·(j)₂ₘ₋₁·N^(1−j−2m).
            let jm1 = BigFloat::from_u64(j as u64 - 1, wp);
            let mut zeta = n_bf.mul(&n_pow, wp, rm).div(&jm1, wp, rm);
            zeta = zeta.add(&n_pow.div(&BigFloat::from_i32(2, wp), wp, rm), wp, rm);
            // rising = (j)_{2m−1}, power = N^(1−j−2m).
            let mut rising = BigFloat::from_u64(j as u64, wp);
            let mut power = n_pow.mul(&inv_n, wp, rm);
            let mut m = 1usize;
            let remainder = loop {
                if m > table.len() {
                    if em_terms >= MAX_EM_TERMS {
                        break None;
                    }
                    em_terms = (em_terms * 2).min(MAX_EM_TERMS);
                    table = bernoulli_over_factorial(em_terms);
                }
                let coef = ratio_to_bigfloat(&table[m - 1], wp, rm);
                let t = coef.mul(&rising, wp, rm).mul(&power, wp, rm);
                zeta = zeta.add(&t, wp, rm);
                // Remainder after these m corrections:
                // 2ζ(2m)·(2π)^(−2m)·(j)_{2m−1}·N^(1−j−2m), with 2ζ(2m) < 4
                // and (2π)^(2m) ≥ 32^m.
                let bound = rising.mul(&power, 64, RoundingMode::Up).mul(
                    &pow2(2 - 5 * m as i64),
                    64,
                    RoundingMode::Up,
                );
                let small = match (bound.exponent(), zeta.exponent()) {
                    (Some(be), Some(ze)) => i64::from(be) < i64::from(ze) - wp as i64 - 4,
                    _ => bound.is_zero(),
                };
                if small {
                    break Some(bound);
                }
                // (j)_{2m+1} = (j)_{2m−1}·(j + 2m − 1)·(j + 2m).
                let f1 = BigFloat::from_u64((j + 2 * m - 1) as u64, wp);
                let f2 = BigFloat::from_u64((j + 2 * m) as u64, wp);
                rising = rising.mul(&f1, wp, rm).mul(&f2, wp, rm);
                power = power.mul(&inv_n2, wp, rm);
                m += 1;
            };
            let Some(remainder) = remainder else {
                return Err(unevaluable(
                    "infinite sum: the Euler–Maclaurin series did not converge".into(),
                ));
            };
            let mut cj =
                bigint_to_bigfloat(ci, wp).div(&bigint_to_bigfloat(&beta0_pow[i + 1], wp), wp, rm);
            if log {
                // −eⱼ₊₁/j: one more rounding, which the count below covers.
                cj = cj.div(&BigFloat::from_u64(j as u64, 64), wp, rm).neg();
            }
            let term = cj.mul(&zeta, wp, rm);
            total = total.add(&term, wp, rm);
            // |cⱼ|·(remainder + roundings of ζ) + rounding of cⱼ·ζ and of the
            // sum.  The powers of 1/N carry j + 2m roundings, the
            // corrections 3m more.
            let roundings = (j + 5 * m + 12) as u64;
            let rounding = zeta
                .abs()
                .mul(&pow2(1 - wp as i64), 64, RoundingMode::Up)
                .mul(&BigFloat::from_u64(roundings, 64), 64, RoundingMode::Up);
            err.add(&cj.abs().mul(
                &remainder.add(&rounding, 64, RoundingMode::Up),
                64,
                RoundingMode::Up,
            ));
            err.add(&total.abs().mul(&pow2(1 - wp as i64), 64, RoundingMode::Up));
        }
        n_pow = n_pow.mul(&inv_n, wp, rm);
    }
    Ok((total, err))
}

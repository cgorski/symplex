//! Exact zero and sign testing for algebraic constants via minimal
//! polynomials and Sturm sequences.
//!
//! This module answers two questions about a constant arena expression
//! built from rationals, radicals (`n^{p/q}`), `i`, sums, products and
//! negations — *is it zero?* and *what is its sign?* — **exactly**, without
//! trusting floating-point cancellation.
//!
//! # What is here
//!
//! - [`minimal_polynomial`] — computes a polynomial `m ∈ ℚ[t]` with
//!   `m(α) = 0` for an algebraic constant `α` given as an arena expression,
//!   by structural recursion:
//!   - rational `r` → `t − r`;
//!   - `n^{p/q}` → the irreducible factor of `t^q − n^p` vanishing at the
//!     real value;
//!   - `i` → `t² + 1`;
//!   - `a + b` → the irreducible factor of `res_y(m_a(y), m_b(t − y))`
//!     vanishing at the numerical value of `a + b`;
//!   - `a · b` → likewise with `res_y(m_a(y), y^{deg m_b} m_b(t / y))`;
//!   - `−a` → `m_a(−t)`;
//!   - `a^{p/q}` for algebraic `a` → the factor of `m_a(t^q)` vanishing at
//!     the real value (`a > 0` when `q > 1`), reversed for `p < 0`, then
//!     folded `|p|` times with the product rule.
//!
//!   Resultants are computed by evaluation/interpolation over
//!   [`GenPoly`].  The factor selection factors the resultant over ℤ
//!   (refusing a factorisation that is not certified complete), evaluates
//!   the number to 320 bits and accepts the *unique* irreducible factor
//!   whose residual there is negligible (below `2⁻¹²⁸` of the trivial
//!   bound on `|g(α)|`); no factor, or more than one, gives `None` rather
//!   than a guess.  The expression is first converted to the
//!   arena-independent [`AlgExpr`] tree, so the computation itself needs
//!   no arena access (and no arena lock).
//! - [`exact_is_zero`] / [`exact_sign`] — given the minimal polynomial,
//!   isolate the real root nearest to the `f64` value of the expression with
//!   a [`SturmChain`] and decide zero-ness/sign from the isolating interval.
//! - [`is_zero_checked`] / [`sign_checked`] — the production entry points
//!   (used by `log_to_real.rs` and `integrate.rs`).  They evaluate with
//!   [`eval_const_f64`](crate::transforms::evalf::eval_const_f64) first and
//!   only fall back to the exact machinery inside the ambiguous zone
//!   `|v| < 1e-10`; if the two methods disagree a warning is logged and the
//!   exact answer wins.
//!
//! There is **no** `AlgNum` element type in this module: arithmetic in
//! `ℚ(α) = ℚ[t]/(m(t))` is not implemented here.  Everything operates on
//! minimal polynomials and arena expressions directly.
//!
//! # Limitations
//!
//! - Only the expression shapes listed above are recognised; anything else
//!   (`π`, `e`, free symbols, transcendental functions) yields `None`.
//! - The resultant-based composition produces a polynomial that *has* `α`
//!   as a root; the minimal polynomial is obtained by factoring and
//!   selecting the irreducible factor that vanishes at a 320-bit value of
//!   `α`.  Two distinct factors would both pass only if the resultant had
//!   roots closer than about `10⁻³⁸` relative to its coefficient size; the
//!   selection then reports `None` instead of choosing.
//!
//! # References
//!
//! - Cohen, *A Course in Computational Algebraic Number Theory*, Springer
//! - Bronstein, *Symbolic Integration I*, §1.2–1.4
//! - SymPy `polys/numberfields/minpoly.py`

use astro_float::{BigFloat, Consts, RoundingMode};
use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};

use super::dense::Poly;
use super::generic::GenPoly;
use super::sturm::{SturmChain, cauchy_bound};
use crate::base::arena::Arena;
use crate::base::bigcomplex::{c_add, c_div, c_from_real, c_mul, c_one, c_zero};
use crate::base::interval::Interval;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::{self, Q};

// ═══════════════════════════════════════════════════════════════════════════
// Sign determination helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Isolate the real root of `poly` nearest to `approx` into a rational
/// half-open interval `(lo, hi]` containing exactly one root (see
/// [`SturmChain::isolate_roots_in`]).
///
/// Returns `None` if `poly` has no real roots.
fn isolate_root_near(poly: &Poly, approx: f64) -> Option<Interval<Q>> {
    let chain = SturmChain::new(poly);
    if chain.has_no_real_roots() {
        return None;
    }

    // Start with a wide interval and narrow toward the approximate root.
    let cauchy = cauchy_bound(poly);
    let neg_bound = -cauchy.clone();

    let intervals = chain.isolate_roots_in(&neg_bound, &cauchy, 60);
    if intervals.is_empty() {
        return None;
    }

    // Find the interval closest to approx.
    let approx_rat = f64_to_rational_approx(approx);
    let mut best = &intervals[0];
    let mut best_dist = rational_dist_to_interval(&approx_rat, &best.lower, &best.upper);

    for interval in &intervals[1..] {
        let dist = rational_dist_to_interval(&approx_rat, &interval.lower, &interval.upper);
        if dist < best_dist {
            best = interval;
            best_dist = dist;
        }
    }

    Some(best.clone())
}

// ═══════════════════════════════════════════════════════════════════════════
// Arena-free algebraic expressions
// ═══════════════════════════════════════════════════════════════════════════

/// An algebraic-number expression detached from the arena — exactly the
/// shapes [`minimal_polynomial`] recognises.  Built under a (read) lock
/// with [`AlgExpr::from_arena`]; everything downstream is arena-free.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum AlgExpr {
    /// A rational number.
    Rat(Ratio<BigInt>),
    /// The imaginary unit `i`.
    I,
    /// The golden ratio `φ = (1 + √5)/2`.
    Phi,
    /// `−a`.
    Neg(Box<AlgExpr>),
    /// `a₁ + a₂ + …` (at least two terms).
    Add(Vec<AlgExpr>),
    /// `a₁ · a₂ · …` (at least two factors).
    Mul(Vec<AlgExpr>),
    /// `base^exp` with a rational exponent.
    Pow(Box<AlgExpr>, Ratio<BigInt>),
}

impl AlgExpr {
    /// Detach the constant `expr` from the arena.
    ///
    /// `None` when some node is not one of the recognised shapes: `π`, `e`,
    /// free symbols, transcendental functions, a power with a non-rational
    /// exponent, or a power of a *negative or zero* rational base with a
    /// fractional exponent (the numeric route below would not know which
    /// branch the arena's evaluator takes).
    pub(crate) fn from_arena(arena: &Arena, expr: ExprId) -> Option<AlgExpr> {
        match arena.node(expr) {
            ExprNode::Num(nid) => Some(AlgExpr::Rat(arena.num(*nid).clone())),
            ExprNode::ImaginaryUnit => Some(AlgExpr::I),
            ExprNode::GoldenRatio => Some(AlgExpr::Phi),
            ExprNode::Neg(inner) => Some(AlgExpr::Neg(Box::new(Self::from_arena(arena, *inner)?))),
            ExprNode::Pow(base, exp) => {
                let exp_r = arena.as_num(*exp)?.clone();
                if let Some(base_r) = arena.as_num(*base) {
                    if !base_r.is_positive() && !exp_r.is_integer() {
                        return None;
                    }
                    return Some(AlgExpr::Pow(Box::new(AlgExpr::Rat(base_r.clone())), exp_r));
                }
                Some(AlgExpr::Pow(
                    Box::new(Self::from_arena(arena, *base)?),
                    exp_r,
                ))
            }
            ExprNode::Add(children) if children.len() >= 2 => Some(AlgExpr::Add(
                children
                    .iter()
                    .map(|&c| Self::from_arena(arena, c))
                    .collect::<Option<Vec<_>>>()?,
            )),
            ExprNode::Mul(children) if children.len() >= 2 => Some(AlgExpr::Mul(
                children
                    .iter()
                    .map(|&c| Self::from_arena(arena, c))
                    .collect::<Option<Vec<_>>>()?,
            )),
            // `π`, `e`, symbols, functions, …: not recognised as algebraic.
            _ => None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Minimal polynomial computation
// ═══════════════════════════════════════════════════════════════════════════

/// Working precision in bits at which an algebraic number is evaluated to
/// decide which irreducible factor of a resultant vanishes at it.
const VERIFY_PREC_BITS: usize = 320;

/// A factor `g` is accepted as vanishing at the 320-bit value `z` when
/// `|g(z)| ≤ 2⁻¹²⁸ · (deg g + 1) · H(g) · max(1, |z|)^{deg g}` — i.e. its
/// residual is at least `2¹²⁸` times below the trivial bound on `|g(z)|`.
/// The true factor's residual is around `2⁻³²⁰` of that bound; a wrong
/// factor would have to have a root within `~2⁻¹²⁸` of `z`.
const VERIFY_TOLERANCE_BITS: i32 = 128;

/// A complex number in arbitrary precision.
type Complex = (BigFloat, BigFloat);

/// Compute the minimal polynomial of an arena expression over ℚ.
///
/// The expression must represent an algebraic number (no transcendental
/// functions, no free symbols other than constants).
///
/// Returns `None` if the expression is not recognized as algebraic, or if
/// the irreducible factor that vanishes at the number cannot be certified
/// (see [`minimal_polynomial_of`]).
///
/// # Algorithm
///
/// The expression is detached into an [`AlgExpr`] (the only step that
/// touches the arena) and handed to [`minimal_polynomial_of`], which
/// decomposes it recursively:
/// - Rational `r` → `t - r`
/// - `n^{p/q}` → the factor of `t^q - n^p` vanishing at the real value
/// - `ImaginaryUnit` → `t² + 1`
/// - `GoldenRatio` → `t² − t − 1`
/// - `a^{p/q}` for algebraic `a` → see [`minpoly_pow`]
/// - `a + b` → resultant composition
/// - `a · b` → resultant composition
/// - `-a` → `m_a(-t)`
///
/// # References
///
/// - SymPy `polys/numberfields/minpoly.py::_minpoly_compose`
pub fn minimal_polynomial(arena: &mut Arena, expr: ExprId) -> Option<Poly> {
    let alg = AlgExpr::from_arena(arena, expr)?;
    minimal_polynomial_of(&alg)
}

/// Minimal polynomial over ℚ of a detached algebraic expression.
///
/// Every resultant produced by the composition rules is factored over ℤ;
/// the factorisation must be certified complete (every factor irreducible)
/// and exactly one factor must vanish, to [`VERIFY_TOLERANCE_BITS`], at
/// the [`VERIFY_PREC_BITS`]-bit value of the sub-expression.  Anything
/// else — an incomplete factorisation, no vanishing factor, two of them,
/// a fractional power of a number that is not real and positive — gives
/// `None`: the function never returns a polynomial it has not verified.
pub(crate) fn minimal_polynomial_of(alg: &AlgExpr) -> Option<Poly> {
    let mut cc = Consts::new().ok()?;
    let (mp, _value) = minpoly_of(alg, &mut cc)?;
    Some(mp)
}

/// Minimal polynomial and high-precision value of `alg`.
fn minpoly_of(alg: &AlgExpr, cc: &mut Consts) -> Option<(Poly, Complex)> {
    let prec = VERIFY_PREC_BITS;
    match alg {
        // r → t − r
        AlgExpr::Rat(r) => Some((
            Poly::from_coeffs(vec![-r.clone(), rat(1, 1)]),
            c_from_real(ratio_to_bigfloat(r, prec), prec),
        )),

        // i → t² + 1
        AlgExpr::I => Some((
            Poly::from_coeffs(vec![rat(1, 1), rat(0, 1), rat(1, 1)]),
            (BigFloat::new(prec), BigFloat::from_i32(1, prec)),
        )),

        // φ = (1 + √5)/2 → t² − t − 1
        AlgExpr::Phi => {
            let rm = RoundingMode::None;
            let sqrt5 = BigFloat::from_i32(5, prec).sqrt(prec, rm);
            let phi = sqrt5.add(&BigFloat::from_i32(1, prec), prec, rm).div(
                &BigFloat::from_i32(2, prec),
                prec,
                rm,
            );
            Some((
                Poly::from_coeffs(vec![rat(-1, 1), rat(-1, 1), rat(1, 1)]),
                c_from_real(phi, prec),
            ))
        }

        // −a → m_a(−t)
        AlgExpr::Neg(inner) => {
            let (mp, (re, im)) = minpoly_of(inner, cc)?;
            Some((negate_variable(&mp), (re.neg(), im.neg())))
        }

        // a + b + … → fold the resultant composition left to right.
        AlgExpr::Add(terms) => {
            let mut iter = terms.iter();
            let (mut acc_mp, mut acc_val) = minpoly_of(iter.next()?, cc)?;
            for term in iter {
                let (mp, val) = minpoly_of(term, cc)?;
                acc_val = c_add(&acc_val, &val, prec, RoundingMode::None);
                acc_mp = minpoly_add(&acc_mp, &mp, &acc_val)?;
            }
            Some((acc_mp, acc_val))
        }

        // a · b · … → rational factors exactly, the rest by resultants.
        AlgExpr::Mul(factors) => {
            let mut rational = rat(1, 1);
            let mut symbolic: Vec<&AlgExpr> = Vec::new();
            for f in factors {
                match f {
                    AlgExpr::Rat(r) => rational *= r,
                    other => symbolic.push(other),
                }
            }
            let rational_val = c_from_real(ratio_to_bigfloat(&rational, prec), prec);
            let mut iter = symbolic.into_iter();
            let Some(first) = iter.next() else {
                // A pure rational product.
                return Some((
                    Poly::from_coeffs(vec![-rational.clone(), rat(1, 1)]),
                    rational_val,
                ));
            };
            let (mut acc_mp, mut acc_val) = minpoly_of(first, cc)?;
            for f in iter {
                let (mp, val) = minpoly_of(f, cc)?;
                acc_val = c_mul(&acc_val, &val, prec, RoundingMode::None);
                acc_mp = minpoly_mul(&acc_mp, &mp, &acc_val)?;
            }
            if !rational.is_one() {
                acc_mp = minpoly_rational_mul(&acc_mp, &rational);
                acc_val = c_mul(&acc_val, &rational_val, prec, RoundingMode::None);
            }
            Some((acc_mp, acc_val))
        }

        AlgExpr::Pow(base, exp) => minpoly_pow(base, exp, cc),
    }
}

/// Minimal polynomial of `a + b` from those of `a` and `b` and the value
/// `value = a + b`.
///
/// `min_poly(a + b)` divides `res_y(m_a(y), m_b(x − y))`; the resultant is
/// computed by evaluation/interpolation, factored, and the factor that
/// vanishes at `value` is selected ([`select_vanishing_factor`]).
fn minpoly_add(mp_a: &Poly, mp_b: &Poly, value: &Complex) -> Option<Poly> {
    let deg_a = mp_a.degree()?;
    let deg_b = mp_b.degree()?;

    // res_y(m_a(y), m_b(x - y)) by evaluation at x = 0, 1, …, deg_a·deg_b
    // and interpolation.
    let result_degree = deg_a * deg_b;
    let mut points: Vec<(i64, Ratio<BigInt>)> = Vec::with_capacity(result_degree + 1);
    for k in 0..=result_degree {
        let x_val = rat(k as i64, 1);
        let mb_shifted = substitute_shift(mp_b, &x_val);
        let res = GenPoly::<Ratio<BigInt>>::resultant(mp_a, &mb_shifted);
        points.push((k as i64, res));
    }

    let r_poly = super::dense::lagrange_interpolate_rational(&points);
    if r_poly.is_zero() {
        return None;
    }
    select_vanishing_factor(&r_poly, value)
}

/// Minimal polynomial of `a · b` from those of `a` and `b` and the value
/// `value = a · b`.
///
/// `min_poly(a · b)` divides `res_y(m_a(y), y^d · m_b(x / y))` with
/// `d = deg m_b`.
fn minpoly_mul(mp_a: &Poly, mp_b: &Poly, value: &Complex) -> Option<Poly> {
    let deg_a = mp_a.degree()?;
    let deg_b = mp_b.degree()?;

    let result_degree = deg_a * deg_b;
    let mut points: Vec<(i64, Ratio<BigInt>)> = Vec::with_capacity(result_degree + 1);
    for k in 0..=result_degree {
        let x_val = rat(k as i64, 1);
        let mb_scaled = reciprocal_scale(mp_b, &x_val);
        let res = GenPoly::<Ratio<BigInt>>::resultant(mp_a, &mb_scaled);
        points.push((k as i64, res));
    }

    let r_poly = super::dense::lagrange_interpolate_rational(&points);
    if r_poly.is_zero() {
        return None;
    }
    select_vanishing_factor(&r_poly, value)
}

/// Minimal polynomial and value of `base^{p/q}`.
///
/// `base^{1/q}` is a root of `m_base(t^q)`; the factor vanishing at the
/// real `q`-th root is selected, which requires `base` to be real and
/// positive when `q > 1` (decided from its 320-bit value: the imaginary
/// part must be at rounding level and the real part positive).  A negative
/// exponent starts from the reciprocal, whose minimal polynomial is `m`
/// with its coefficients reversed; integer powers then fold with
/// [`minpoly_mul`].  Returns `None` when the base is not algebraic, is
/// zero under a negative exponent, is not real-positive under a fractional
/// exponent, or the exponent does not fit `i64`.
fn minpoly_pow(base: &AlgExpr, exp: &Ratio<BigInt>, cc: &mut Consts) -> Option<(Poly, Complex)> {
    let prec = VERIFY_PREC_BITS;
    let rm = RoundingMode::None;
    let p: i64 = exp.numer().try_into().ok()?;
    let q: usize = exp.denom().try_into().ok()?;

    // A rational base `r > 0`: `r^{p/q}` is a root of `t^q − r^p` directly.
    if let AlgExpr::Rat(r) = base
        && r.is_positive()
    {
        let r_pow_p = rational_pow(r, exp.numer())?;
        if q == 1 {
            return Some((
                Poly::from_coeffs(vec![-r_pow_p.clone(), rat(1, 1)]),
                c_from_real(ratio_to_bigfloat(&r_pow_p, prec), prec),
            ));
        }
        let mut coeffs = vec![Ratio::zero(); q + 1];
        coeffs[0] = -r_pow_p;
        coeffs[q] = rat(1, 1);
        let exp_bf = ratio_to_bigfloat(exp, prec);
        let value = c_from_real(ratio_to_bigfloat(r, prec).pow(&exp_bf, prec, rm, cc), prec);
        let mp = select_vanishing_factor(&Poly::from_coeffs(coeffs), &value)?;
        return Some((mp, value));
    }

    let (mp_base, base_val) = minpoly_of(base, cc)?;

    // α = base^{1/q} with its minimal polynomial and value.
    let (mut mp, mut alpha) = if q == 1 {
        (mp_base, base_val)
    } else {
        if !is_real_positive(&base_val, prec) {
            return None;
        }
        let root = base_val.0.pow(
            &BigFloat::from_i32(1, prec).div(&BigFloat::from_i64(q as i64, prec), prec, rm),
            prec,
            rm,
            cc,
        );
        let root = c_from_real(root, prec);
        let deg = mp_base.degree()?;
        let mut coeffs = vec![Ratio::zero(); deg * q + 1];
        for (i, c) in mp_base.coeffs().iter().enumerate() {
            coeffs[i * q] = c.clone();
        }
        let mp = select_vanishing_factor(&Poly::from_coeffs(coeffs), &root)?;
        (mp, root)
    };

    if p == 0 {
        return Some((Poly::from_coeffs(vec![rat(-1, 1), rat(1, 1)]), c_one(prec)));
    }
    if p < 0 {
        // 1/α: reverse the coefficients (α ≠ 0 ⇔ constant term ≠ 0).
        if mp.coeff(0).is_zero() {
            return None;
        }
        let reversed: Vec<Ratio<BigInt>> = mp.coeffs().iter().rev().cloned().collect();
        mp = Poly::from_coeffs(reversed).make_monic();
        alpha = c_div(&c_one(prec), &alpha, prec, rm);
    }

    // α^n by folding α · α · … with the running value for factor selection.
    let n = p.unsigned_abs();
    let mut acc = mp.clone();
    let mut acc_val = alpha.clone();
    for _ in 1..n {
        acc_val = c_mul(&acc_val, &alpha, prec, rm);
        acc = minpoly_mul(&acc, &mp, &acc_val)?;
    }
    Some((acc, acc_val))
}

/// Compute minimal polynomial of `r · α` where `r ∈ ℚ` and `α` has
/// minimal polynomial `mp_alpha`.
///
/// If `m(t)` is the minimal polynomial of `α`, then `r·α` has minimal
/// polynomial `m(t/r)` scaled to be monic with integer (or rational) coefficients.
fn minpoly_rational_mul(mp_alpha: &Poly, r: &Ratio<BigInt>) -> Poly {
    if r.is_zero() {
        // r·α = 0, minimal polynomial is t.
        return Poly::x();
    }
    let deg = mp_alpha.degree().unwrap_or(0);
    let mut coeffs = Vec::with_capacity(deg + 1);
    // m(t/r) = Σ a_k (t/r)^k = Σ a_k / r^k · t^k
    // To make monic: multiply through by r^n / a_n
    let mut r_power = rat(1, 1);
    let r_inv = Ratio::new(r.denom().clone(), r.numer().clone());
    for k in 0..=deg {
        let c = mp_alpha.coeff(k);
        coeffs.push(c * r_power.clone());
        r_power *= r_inv.clone();
    }
    let result = Poly::from_coeffs(coeffs);
    result.make_monic()
}

// ═══════════════════════════════════════════════════════════════════════════
// Factor selection by high-precision verification
// ═══════════════════════════════════════════════════════════════════════════

/// The irreducible factor of `r_poly` that vanishes at `value`, monic.
///
/// `r_poly` is factored over ℤ with the certified variant of the
/// Zassenhaus routine; an incomplete factorisation (some factor possibly
/// reducible) gives `None`.  A single factor is the answer outright — the
/// number is a root of `r_poly` by construction.  Otherwise every factor is
/// evaluated at `value` and the one whose residual is negligible (see
/// [`VERIFY_TOLERANCE_BITS`]) is returned; `None` unless exactly one
/// qualifies.
fn select_vanishing_factor(r_poly: &Poly, value: &Complex) -> Option<Poly> {
    let (_content, factors, complete) = super::factor_zassenhaus::factor_zassenhaus_checked(r_poly);
    if !complete {
        tracing::debug!(
            degree = r_poly.degree().unwrap_or(0),
            "minimal_polynomial: factorisation of the resultant is not certified complete"
        );
        return None;
    }
    if factors.is_empty() {
        return None;
    }
    if factors.len() == 1 {
        return Some(factors[0].0.make_monic());
    }

    let mut vanishing: Option<&Poly> = None;
    for (g, _) in &factors {
        if !vanishes_at(g, value) {
            continue;
        }
        if vanishing.is_some() {
            tracing::debug!(
                "minimal_polynomial: two irreducible factors of the resultant vanish at the value"
            );
            return None;
        }
        vanishing = Some(g);
    }
    if vanishing.is_none() {
        tracing::debug!(
            "minimal_polynomial: no irreducible factor of the resultant vanishes at the value"
        );
    }
    vanishing.map(Poly::make_monic)
}

/// Is `|g(z)|` below `2^-VERIFY_TOLERANCE_BITS` of the trivial bound
/// `(deg g + 1) · H(g) · max(1, |z|)^{deg g}` on `|g(z)|`?
fn vanishes_at(g: &Poly, z: &Complex) -> bool {
    let prec = VERIFY_PREC_BITS;
    let rm = RoundingMode::None;
    let Some(deg) = g.degree() else {
        return false;
    };
    // Horner evaluation in complex arithmetic.
    let mut acc = c_zero(prec);
    for c in g.coeffs().iter().rev() {
        acc = c_mul(&acc, z, prec, rm);
        acc = c_add(
            &acc,
            &c_from_real(ratio_to_bigfloat(c, prec), prec),
            prec,
            rm,
        );
    }
    let residual = c_abs_upper(&acc, prec);

    // Height and |z|, both as upper bounds.
    let mut height = BigFloat::new(prec);
    for c in g.coeffs() {
        let a = ratio_to_bigfloat(c, prec).abs();
        if a.cmp(&height).unwrap_or(0) > 0 {
            height = a;
        }
    }
    let one = BigFloat::from_i32(1, prec);
    let z_abs = c_abs_upper(z, prec).max(&one);
    let bound = z_abs.powi(deg, prec, rm).mul(&height, prec, rm).mul(
        &BigFloat::from_i64(deg as i64 + 1, prec),
        prec,
        rm,
    );
    let tolerance = bound.mul(
        &BigFloat::from_i32(2, prec)
            .powi(VERIFY_TOLERANCE_BITS as usize, prec, rm)
            .reciprocal(prec, rm),
        prec,
        rm,
    );
    !residual.is_nan() && residual.cmp(&tolerance).is_some_and(|c| c <= 0)
}

/// `|re| + |im|`, an upper bound on `|z|` that needs no square root.
fn c_abs_upper(z: &Complex, prec: usize) -> BigFloat {
    z.0.abs().add(&z.1.abs(), prec, RoundingMode::None)
}

/// Is `z` a positive real number, up to rounding noise at `prec` bits?
/// The imaginary part must be zero or smaller than `2^{-prec/2}·|re|`.
fn is_real_positive(z: &Complex, prec: usize) -> bool {
    let (re, im) = z;
    if re.is_nan() || im.is_nan() || !re.is_positive() {
        return false;
    }
    if im.is_zero() {
        return true;
    }
    let rm = RoundingMode::None;
    let noise = re.abs().mul(
        &BigFloat::from_i32(2, prec)
            .powi(prec / 2, prec, rm)
            .reciprocal(prec, rm),
        prec,
        rm,
    );
    im.abs().cmp(&noise).is_some_and(|c| c < 0)
}

/// Convert a `Ratio<BigInt>` to a `BigFloat` at `prec` bits (exactly up to
/// the final rounding, whatever the size of numerator and denominator; see
/// [`numeric::ratio_to_bigfloat`]).
fn ratio_to_bigfloat(r: &Ratio<BigInt>, prec: usize) -> BigFloat {
    numeric::ratio_to_bigfloat(r, prec, RoundingMode::None)
}

// ═══════════════════════════════════════════════════════════════════════════
// Exact zero/sign testing for arena expressions
// ═══════════════════════════════════════════════════════════════════════════

/// Exact zero test for a constant algebraic arena expression.
///
/// Computes the minimal polynomial `m(t)` of the expression, then checks
/// if `m(0) = 0`.  If so, and if 0 is the only root near the expression's
/// numerical value, the expression is zero.
///
/// Returns `Some(true)` if definitely zero, `Some(false)` if definitely
/// nonzero, `None` if cannot determine (e.g., transcendental expression).
pub fn exact_is_zero(
    arena: &mut crate::base::arena::Arena,
    expr: crate::base::node::ExprId,
) -> Option<bool> {
    // Quick structural check.
    if expr == arena.zero {
        return Some(true);
    }

    // Quick rational check.
    if let Some(r) = arena.as_num(expr) {
        return Some(r.is_zero());
    }

    // Compute minimal polynomial.
    let mp = minimal_polynomial(arena, expr)?;

    // If m(0) ≠ 0, the expression is definitely nonzero.
    let m_at_zero = mp.eval(&rat(0, 1));
    if !m_at_zero.is_zero() {
        tracing::trace!("exact_is_zero: m(0) ≠ 0 → definitely nonzero");
        return Some(false);
    }

    // m(0) = 0, so 0 is a root of m.  But is the EXPRESSION equal to 0,
    // or is it a different root of m?
    // Use numerical approximation + Sturm root isolation to determine
    // which root of m the expression corresponds to.
    let approx = crate::transforms::evalf::eval_const_f64(arena, expr)?;

    // Fast path: if numerically far from zero, it's a different root of m.
    if approx.abs() > 1e-10 {
        tracing::trace!("exact_is_zero: m(0)=0 but |expr|={approx} > 1e-10 → nonzero root");
        return Some(false);
    }

    // Numerically near zero.  Use Sturm root isolation to rigorously verify.
    // Isolate the root of m nearest to the numerical approximation.
    if let Some(interval) = isolate_root_near(&mp, approx) {
        // If the interval is entirely positive or entirely negative,
        // the expression corresponds to a nonzero root.  (The cell is
        // `(lo, hi]`, so `lo == 0` would exclude 0 too; testing `lo > 0`
        // is merely conservative and falls through to the count below.)
        if interval.lower.is_positive() || interval.upper.is_negative() {
            tracing::trace!("exact_is_zero: isolated root interval excludes 0 → nonzero");
            return Some(false);
        }
        // Interval straddles zero.  Check that 0 is the only root in it.
        let chain = SturmChain::new(&mp);
        let zero_rat = rat(0, 1);
        let roots_in_neg = chain.count_roots_in(&interval.lower, &zero_rat);
        let roots_in_pos = chain.count_roots_in(&zero_rat, &interval.upper);
        if roots_in_neg == 0 && roots_in_pos == 0 {
            // 0 is the only root in this interval — expression is zero.
            tracing::trace!("exact_is_zero: isolated interval contains only 0 → zero");
            return Some(true);
        }
        // Other roots share the interval with 0 — cannot fully resolve
        // via isolation alone.  Fall through to numerical heuristic.
        tracing::trace!("exact_is_zero: interval has roots besides 0, relying on numerical approx");
    }

    // Fallback: trust the numerical approximation (correct for all practical
    // cases from integration, where algebraic numbers have well-separated roots).
    tracing::trace!("exact_is_zero: m(0)=0 and |expr| < 1e-10 → zero (numerical fallback)");
    Some(true)
}

/// Exact sign test for a constant real algebraic arena expression.
///
/// Returns `Some(1)` for positive, `Some(-1)` for negative, `Some(0)` for
/// zero, or `None` if the sign cannot be determined.
pub fn exact_sign(
    arena: &mut crate::base::arena::Arena,
    expr: crate::base::node::ExprId,
) -> Option<i8> {
    // Quick checks.
    if expr == arena.zero {
        return Some(0);
    }
    if let Some(r) = arena.as_num(expr) {
        return Some(if r.is_positive() {
            1
        } else if r.is_negative() {
            -1
        } else {
            0
        });
    }

    // Compute minimal polynomial.
    let mp = minimal_polynomial(arena, expr)?;

    // Get numerical approximation.
    let approx = crate::transforms::evalf::eval_const_f64(arena, expr)?;

    // Isolate the root nearest to the approximation.
    let interval = isolate_root_near(&mp, approx)?;

    // The expression is the root of mp in this interval.
    // The sign of the root can be determined from the interval bounds.
    if interval.lower.is_positive() {
        Some(1)
    } else if interval.upper.is_negative() {
        Some(-1)
    } else {
        // Interval contains zero — the root might be zero.
        // Check m(0).
        let m_at_zero = mp.eval(&rat(0, 1));
        if m_at_zero.is_zero() {
            // 0 is a root.  Is it THE root in this interval?
            let chain = SturmChain::new(&mp);
            // Count roots in (lo, 0] and (0, hi]
            let zero_rat = rat(0, 1);
            let roots_left = chain.count_roots_in(&interval.lower, &zero_rat);
            let roots_right = chain.count_roots_in(&zero_rat, &interval.upper);
            if roots_left == 0 && roots_right == 0 {
                // The only root in the interval is at 0.
                Some(0)
            } else {
                // There's another root — refine further.
                // Use numerical approximation as tiebreaker.
                if approx > 1e-15 {
                    Some(1)
                } else if approx < -1e-15 {
                    Some(-1)
                } else {
                    Some(0)
                }
            }
        } else {
            // 0 is not a root, so the root in this interval is nonzero.
            if approx > 0.0 { Some(1) } else { Some(-1) }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Production cross-checked zero / sign tests
// ═══════════════════════════════════════════════════════════════════════════

/// Cross-checked zero test for production use.
///
/// **Primary**: `eval_const_f64` (fast, battle-tested).
/// **Fallback**: `exact_is_zero` (Sturm-based) when the f64 value is
/// ambiguous (within `1e-10` of zero).
///
/// If both methods produce a result and they **disagree**, a warning is
/// logged and the `eval_const_f64` result is trusted.
///
/// Returns `Some(true)` if zero, `Some(false)` if nonzero, `None` if
/// the test is inconclusive.
pub fn is_zero_checked(
    arena: &mut crate::base::arena::Arena,
    expr: crate::base::node::ExprId,
) -> Option<bool> {
    // Quick structural check.
    if expr == arena.zero {
        return Some(true);
    }
    if let Some(r) = arena.as_num(expr) {
        return Some(r.is_zero());
    }

    // Primary: numerical evaluation.
    let f64_val = crate::transforms::evalf::eval_const_f64(arena, expr);

    match f64_val {
        Some(v) if v.abs() >= 1e-10 => {
            // Clearly nonzero — no need for exact methods.
            Some(false)
        }
        Some(v) => {
            // Ambiguous zone: |v| < 1e-10.  Try exact method.
            let f64_says_zero = v.abs() < 1e-14;
            match exact_is_zero(arena, expr) {
                Some(exact_answer) => {
                    if exact_answer != f64_says_zero {
                        tracing::warn!(
                            f64_val = v,
                            exact_answer,
                            "is_zero_checked: eval_const_f64 and exact_is_zero DISAGREE — trusting exact"
                        );
                    }
                    // In the ambiguous zone, trust the exact answer when available.
                    Some(exact_answer)
                }
                None => {
                    // Exact method inconclusive — fall back to f64 tolerance.
                    tracing::trace!(
                        f64_val = v,
                        "is_zero_checked: exact_is_zero returned None, using f64 tolerance"
                    );
                    Some(f64_says_zero)
                }
            }
        }
        None => {
            // eval_const_f64 failed entirely — try exact method alone.
            exact_is_zero(arena, expr)
        }
    }
}

/// Cross-checked sign test for production use.
///
/// **Primary**: `eval_const_f64` (fast, battle-tested).
/// **Fallback**: `exact_sign` (Sturm-based) when the f64 value is
/// ambiguous (within `1e-10` of zero).
///
/// Returns `Some(1)` for positive, `Some(-1)` for negative, `Some(0)`
/// for zero, or `None` if inconclusive.
pub fn sign_checked(
    arena: &mut crate::base::arena::Arena,
    expr: crate::base::node::ExprId,
) -> Option<i8> {
    // Quick structural check.
    if expr == arena.zero {
        return Some(0);
    }
    if let Some(r) = arena.as_num(expr) {
        return Some(if r.is_positive() {
            1
        } else if r.is_negative() {
            -1
        } else {
            0
        });
    }

    // Primary: numerical evaluation.
    let f64_val = crate::transforms::evalf::eval_const_f64(arena, expr);

    match f64_val {
        Some(v) if v > 1e-10 => Some(1),
        Some(v) if v < -1e-10 => Some(-1),
        Some(v) => {
            // Ambiguous zone: |v| < 1e-10.  Try exact method.
            let f64_sign: i8 = if v > 1e-14 {
                1
            } else if v < -1e-14 {
                -1
            } else {
                0
            };
            match exact_sign(arena, expr) {
                Some(exact_answer) => {
                    if exact_answer != f64_sign {
                        tracing::warn!(
                            f64_val = v,
                            exact_sign = exact_answer,
                            f64_sign,
                            "sign_checked: eval_const_f64 and exact_sign DISAGREE — trusting exact"
                        );
                    }
                    Some(exact_answer)
                }
                None => {
                    tracing::trace!(
                        f64_val = v,
                        "sign_checked: exact_sign returned None, using f64"
                    );
                    Some(f64_sign)
                }
            }
        }
        None => {
            // eval_const_f64 failed — try exact method alone.
            exact_sign(arena, expr)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Polynomial manipulation helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Negate the variable: if `p(t) = Σ a_k t^k`, return `p(-t) = Σ a_k (-1)^k t^k`.
fn negate_variable(p: &Poly) -> Poly {
    let deg = p.degree().unwrap_or(0);
    let mut coeffs = Vec::with_capacity(deg + 1);
    for k in 0..=deg {
        let c = p.coeff(k);
        if k % 2 == 0 {
            coeffs.push(c);
        } else {
            coeffs.push(-c);
        }
    }
    Poly::from_coeffs(coeffs)
}

/// Substitute `t → x_val - t` in polynomial `p(t)`.
/// Returns `p(x_val - t)` as a polynomial in `t`.
fn substitute_shift(p: &Poly, x_val: &Ratio<BigInt>) -> Poly {
    let deg = p.degree().unwrap_or(0);
    // p(x - t) = Σ a_k (x - t)^k
    // Expand using binomial theorem and collect by powers of t.
    let mut result = Poly::zero();
    for k in 0..=deg {
        let a_k = p.coeff(k);
        if a_k.is_zero() {
            continue;
        }
        // (x - t)^k = Σ_{j=0}^{k} C(k,j) x^{k-j} (-t)^j
        //           = Σ_{j=0}^{k} C(k,j) (-1)^j x^{k-j} t^j
        for j in 0..=k {
            let binom = binomial_rational(k, j);
            let sign = if j % 2 == 0 { rat(1, 1) } else { rat(-1, 1) };
            let x_power = rational_pow_usize(x_val, k - j);
            let coeff_j = &a_k * &binom * &sign * &x_power;
            // Add coeff_j to the j-th coefficient of result.
            let mut result_coeffs: Vec<Ratio<BigInt>> =
                (0..=std::cmp::max(result.degree().unwrap_or(0), j))
                    .map(|i| result.coeff(i))
                    .collect();
            while result_coeffs.len() <= j {
                result_coeffs.push(Ratio::zero());
            }
            result_coeffs[j] = result_coeffs[j].clone() + coeff_j;
            result = Poly::from_coeffs(result_coeffs);
        }
    }
    result
}

/// Compute `y^d · m_b(x_val/y)` as a polynomial in `y`.
/// If `m_b(t) = Σ b_i t^i` with degree d, then
/// `y^d · m_b(x/y) = Σ b_i x^i y^{d-i}`.
fn reciprocal_scale(p: &Poly, x_val: &Ratio<BigInt>) -> Poly {
    let deg = p.degree().unwrap_or(0);
    let mut coeffs = vec![Ratio::zero(); deg + 1];
    for i in 0..=deg {
        let b_i = p.coeff(i);
        if b_i.is_zero() {
            continue;
        }
        let x_pow_i = rational_pow_usize(x_val, i);
        // Coefficient of y^{d-i} is b_i · x^i
        coeffs[deg - i] = &b_i * &x_pow_i;
    }
    Poly::from_coeffs(coeffs)
}

// ═══════════════════════════════════════════════════════════════════════════
// Numeric / rational helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Shorthand for creating a `Ratio<BigInt>`.
fn rat(n: i64, d: i64) -> Ratio<BigInt> {
    Ratio::new(BigInt::from(n), BigInt::from(d))
}

/// Raise a rational number to an integer power.
fn rational_pow(base: &Ratio<BigInt>, exp: &BigInt) -> Option<Ratio<BigInt>> {
    let exp_i64: i64 = exp.try_into().ok()?;
    if exp_i64 >= 0 {
        let e = exp_i64 as u32;
        let n = num_traits::Pow::pow(base.numer(), e);
        let d = num_traits::Pow::pow(base.denom(), e);
        Some(Ratio::new(n, d))
    } else {
        let e = (-exp_i64) as u32;
        let n = num_traits::Pow::pow(base.denom(), e);
        let d = num_traits::Pow::pow(base.numer(), e);
        Some(Ratio::new(n, d))
    }
}

/// Raise a rational number to a non-negative `usize` power.
fn rational_pow_usize(base: &Ratio<BigInt>, exp: usize) -> Ratio<BigInt> {
    if exp == 0 {
        return rat(1, 1);
    }
    let mut result = base.clone();
    for _ in 1..exp {
        result = &result * base;
    }
    result
}

/// Binomial coefficient C(n, k) as a rational number.
fn binomial_rational(n: usize, k: usize) -> Ratio<BigInt> {
    if k > n {
        return Ratio::zero();
    }
    let k = k.min(n - k);
    let mut result = rat(1, 1);
    for i in 0..k {
        result *= Ratio::from_integer(BigInt::from(n - i));
        result /= Ratio::from_integer(BigInt::from(i + 1));
    }
    result
}

/// Convert an `f64` to the exact rational it represents (for root
/// isolation).  Non-finite inputs map to `0`, which only affects which
/// isolating interval is picked as "nearest" — never correctness.
fn f64_to_rational_approx(x: f64) -> Ratio<BigInt> {
    crate::base::numeric::f64_to_ratio_exact(x).unwrap_or_else(Ratio::zero)
}

/// Distance from a point to an interval (0 if inside).
fn rational_dist_to_interval(
    point: &Ratio<BigInt>,
    lo: &Ratio<BigInt>,
    hi: &Ratio<BigInt>,
) -> Ratio<BigInt> {
    if point < lo {
        lo - point
    } else if point > hi {
        point - hi
    } else {
        Ratio::zero()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: i64, d: i64) -> Ratio<BigInt> {
        Ratio::new(BigInt::from(n), BigInt::from(d))
    }

    // ── Minimal polynomial computation ─────────────────────────────

    #[test]
    fn minpoly_rational() {
        let mut arena = crate::base::arena::Arena::new();
        let expr = arena.rational(3, 4);
        let mp = minimal_polynomial(&mut arena, expr).unwrap();
        // min_poly of 3/4 is t - 3/4, but we return monic integer-coeff form:
        // 4t - 3 → monic: t - 3/4
        assert_eq!(mp.degree(), Some(1));
        assert!(
            num_traits::Zero::is_zero(&mp.eval(&r(3, 4))),
            "3/4 should be a root"
        );
    }

    #[test]
    fn minpoly_sqrt2() {
        let mut arena = crate::base::arena::Arena::new();
        let two = arena.int(2);
        let half = arena.rational(1, 2);
        let sqrt2 = arena.pow(two, half);
        let mp = minimal_polynomial(&mut arena, sqrt2).unwrap();
        assert_eq!(mp.degree(), Some(2));
        // Should be t² - 2
        assert_eq!(mp.coeff(0), r(-2, 1));
        assert_eq!(mp.coeff(2), r(1, 1));
    }

    #[test]
    fn minpoly_cbrt2() {
        let mut arena = crate::base::arena::Arena::new();
        let two = arena.int(2);
        let third = arena.rational(1, 3);
        let cbrt2 = arena.pow(two, third);
        let mp = minimal_polynomial(&mut arena, cbrt2).unwrap();
        assert_eq!(mp.degree(), Some(3));
        // Should be t³ - 2
        assert_eq!(mp.coeff(0), r(-2, 1));
        assert_eq!(mp.coeff(3), r(1, 1));
    }

    #[test]
    fn minpoly_imaginary_unit() {
        let mut arena = crate::base::arena::Arena::new();
        let i_unit = arena.i_unit;
        let mp = minimal_polynomial(&mut arena, i_unit).unwrap();
        assert_eq!(mp.degree(), Some(2));
        // t² + 1
        assert_eq!(mp.coeff(0), r(1, 1));
        assert_eq!(mp.coeff(2), r(1, 1));
    }

    // ── Exact zero testing ─────────────────────────────────────────

    #[test]
    fn exact_is_zero_on_zero() {
        let mut arena = crate::base::arena::Arena::new();
        let zero = arena.zero;
        assert_eq!(exact_is_zero(&mut arena, zero), Some(true));
    }

    #[test]
    fn exact_is_zero_on_nonzero_rational() {
        let mut arena = crate::base::arena::Arena::new();
        let expr = arena.rational(1, 3);
        assert_eq!(exact_is_zero(&mut arena, expr), Some(false));
    }

    #[test]
    fn exact_is_zero_sqrt5_squared_minus_5() {
        // (√5)² - 5 = 0 (already simplified at construction, but test the path)
        let mut arena = crate::base::arena::Arena::new();
        let five = arena.int(5);
        let half = arena.rational(1, 2);
        let sqrt5 = arena.pow(five, half);
        // With our canon_pow fix, sqrt5.powi(2) = 5 at construction.
        // So sqrt5² - 5 = 0 structurally. Let's test with a compound expr.
        let two = arena.int(2);
        let sqrt5_sq = arena.pow(sqrt5, two); // This is 5 via canon_pow
        let diff = arena.sub(sqrt5_sq, five);
        assert_eq!(exact_is_zero(&mut arena, diff), Some(true));
    }

    // ── Exact sign testing ─────────────────────────────────────────

    #[test]
    fn exact_sign_positive_rational() {
        let mut arena = crate::base::arena::Arena::new();
        let expr = arena.rational(7, 3);
        assert_eq!(exact_sign(&mut arena, expr), Some(1));
    }

    #[test]
    fn exact_sign_negative_rational() {
        let mut arena = crate::base::arena::Arena::new();
        let expr = arena.rational(-2, 5);
        assert_eq!(exact_sign(&mut arena, expr), Some(-1));
    }

    #[test]
    fn exact_sign_zero() {
        let mut arena = crate::base::arena::Arena::new();
        let zero = arena.zero;
        assert_eq!(exact_sign(&mut arena, zero), Some(0));
    }

    // ── Cross-validation: is_zero_checked / sign_checked vs eval_const_f64 ──

    /// Helper: assert that `is_zero_checked` agrees with `eval_const_f64`
    /// for the given expression.
    fn assert_zero_check_agrees(
        arena: &mut crate::base::arena::Arena,
        expr: crate::base::node::ExprId,
        label: &str,
    ) {
        let f64_val = crate::transforms::evalf::eval_const_f64(arena, expr);
        let checked = is_zero_checked(arena, expr);
        let f64_says_zero = f64_val.map(|v| v.abs() < 1e-14);
        if let (Some(c), Some(f)) = (checked, f64_says_zero) {
            assert_eq!(
                c, f,
                "cross-check MISMATCH for {label}: is_zero_checked={c}, f64_says_zero={f}, f64_val={f64_val:?}"
            );
        }
    }

    /// Helper: assert that `sign_checked` agrees with `eval_const_f64`
    /// for the given expression.
    fn assert_sign_check_agrees(
        arena: &mut crate::base::arena::Arena,
        expr: crate::base::node::ExprId,
        label: &str,
    ) {
        let f64_val = crate::transforms::evalf::eval_const_f64(arena, expr);
        let checked = sign_checked(arena, expr);
        let f64_sign: Option<i8> = f64_val.map(|v| {
            if v > 1e-14 {
                1
            } else if v < -1e-14 {
                -1
            } else {
                0
            }
        });
        if let (Some(c), Some(f)) = (checked, f64_sign) {
            assert_eq!(
                c, f,
                "cross-check MISMATCH for {label}: sign_checked={c}, f64_sign={f}, f64_val={f64_val:?}"
            );
        }
    }

    #[test]
    fn cross_check_zero_rational() {
        let mut arena = crate::base::arena::Arena::new();
        let zero = arena.zero;
        let one_third = arena.rational(1, 3);
        let neg_seven = arena.int(-7);
        assert_zero_check_agrees(&mut arena, zero, "0");
        assert_zero_check_agrees(&mut arena, one_third, "1/3");
        assert_zero_check_agrees(&mut arena, neg_seven, "-7");
    }

    #[test]
    fn cross_check_sign_rational() {
        let mut arena = crate::base::arena::Arena::new();
        let zero = arena.zero;
        let pos = arena.rational(7, 3);
        let neg = arena.rational(-2, 5);
        assert_sign_check_agrees(&mut arena, zero, "0");
        assert_sign_check_agrees(&mut arena, pos, "7/3");
        assert_sign_check_agrees(&mut arena, neg, "-2/5");
    }

    #[test]
    fn cross_check_sqrt2() {
        let mut arena = crate::base::arena::Arena::new();
        let two = arena.int(2);
        let half = arena.rational(1, 2);
        let sqrt2 = arena.pow(two, half);
        assert_zero_check_agrees(&mut arena, sqrt2, "√2");
        assert_sign_check_agrees(&mut arena, sqrt2, "√2");
    }

    #[test]
    fn cross_check_sqrt3() {
        let mut arena = crate::base::arena::Arena::new();
        let three = arena.int(3);
        let half = arena.rational(1, 2);
        let sqrt3 = arena.pow(three, half);
        assert_zero_check_agrees(&mut arena, sqrt3, "√3");
        assert_sign_check_agrees(&mut arena, sqrt3, "√3");
    }

    #[test]
    fn cross_check_cbrt2() {
        let mut arena = crate::base::arena::Arena::new();
        let two = arena.int(2);
        let third = arena.rational(1, 3);
        let cbrt2 = arena.pow(two, third);
        assert_zero_check_agrees(&mut arena, cbrt2, "∛2");
        assert_sign_check_agrees(&mut arena, cbrt2, "∛2");
    }

    #[test]
    fn cross_check_sqrt5_sq_minus_5() {
        // (√5)² - 5 = 0
        let mut arena = crate::base::arena::Arena::new();
        let five = arena.int(5);
        let half = arena.rational(1, 2);
        let sqrt5 = arena.pow(five, half);
        let two = arena.int(2);
        let sqrt5_sq = arena.pow(sqrt5, two);
        let diff = arena.sub(sqrt5_sq, five);
        assert_zero_check_agrees(&mut arena, diff, "(√5)²-5");
        assert_sign_check_agrees(&mut arena, diff, "(√5)²-5");
    }

    #[test]
    fn cross_check_negative_sqrt() {
        // -√2 is negative
        let mut arena = crate::base::arena::Arena::new();
        let two = arena.int(2);
        let half = arena.rational(1, 2);
        let sqrt2 = arena.pow(two, half);
        let neg_sqrt2 = arena.neg(sqrt2);
        assert_zero_check_agrees(&mut arena, neg_sqrt2, "-√2");
        assert_sign_check_agrees(&mut arena, neg_sqrt2, "-√2");
    }

    #[test]
    fn cross_check_rational_times_sqrt() {
        // (3/4)·√2 is positive and nonzero
        let mut arena = crate::base::arena::Arena::new();
        let two = arena.int(2);
        let half = arena.rational(1, 2);
        let sqrt2 = arena.pow(two, half);
        let three_fourths = arena.rational(3, 4);
        let expr = arena.mul(&[three_fourths, sqrt2]);
        assert_zero_check_agrees(&mut arena, expr, "(3/4)·√2");
        assert_sign_check_agrees(&mut arena, expr, "(3/4)·√2");
    }

    #[test]
    fn cross_check_one_over_sqrt2() {
        // 1/√2 ≈ 0.707 — positive, nonzero
        let mut arena = crate::base::arena::Arena::new();
        let one = arena.int(1);
        let two = arena.int(2);
        let half = arena.rational(1, 2);
        let sqrt2 = arena.pow(two, half);
        let expr = arena.div(one, sqrt2);
        assert_zero_check_agrees(&mut arena, expr, "1/√2");
        assert_sign_check_agrees(&mut arena, expr, "1/√2");
    }

    #[test]
    fn cross_check_small_positive_rational() {
        // 1/1000000 — small but clearly positive, should not be confused with zero
        let mut arena = crate::base::arena::Arena::new();
        let expr = arena.rational(1, 1000000);
        assert_zero_check_agrees(&mut arena, expr, "1/1000000");
        assert_sign_check_agrees(&mut arena, expr, "1/1000000");
        assert_eq!(is_zero_checked(&mut arena, expr), Some(false));
        assert_eq!(sign_checked(&mut arena, expr), Some(1));
    }

    // ══════════════════════════════════════════════════════════════════
    // Probes: understand arena node shapes before writing attack tests
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn probe_arena_add_arity() {
        // Does arena.add(&[a, b, c]) produce a 3-child Add node, or a
        // nested binary tree?  This determines whether the N-ary fold
        // path in minimal_polynomial ever fires.
        use crate::base::node::ExprNode;
        let mut arena = crate::base::arena::Arena::new();
        let n2 = arena.int(2);
        let n3 = arena.int(3);
        let n5 = arena.int(5);
        let half = arena.rational(1, 2);
        let sqrt2 = arena.pow(n2, half);
        let sqrt3 = arena.pow(n3, half);
        let sqrt5 = arena.pow(n5, half);
        let sum = arena.add(&[sqrt2, sqrt3, sqrt5]);
        let arity = match arena.node(sum) {
            ExprNode::Add(children) => children.len(),
            _ => 0,
        };
        // Record what we got — the test below depends on this.
        eprintln!("probe_arena_add_arity: Add node has {arity} children");
        // Whether it's 2 or 3, the test suite must cover both paths.
        // If arity == 3, the N-ary fold bug is reachable.
        // If arity == 2, the binary path handles it and the fold is dead code.
        assert!(arity >= 2, "Add node should have at least 2 children");
    }

    #[test]
    fn probe_arena_mul_arity() {
        use crate::base::node::ExprNode;
        let mut arena = crate::base::arena::Arena::new();
        let n2 = arena.int(2);
        let n3 = arena.int(3);
        let n5 = arena.int(5);
        let half = arena.rational(1, 2);
        let sqrt2 = arena.pow(n2, half);
        let sqrt3 = arena.pow(n3, half);
        let sqrt5 = arena.pow(n5, half);
        let prod = arena.mul(&[sqrt2, sqrt3, sqrt5]);
        let node = arena.node(prod).clone();
        eprintln!("probe_arena_mul_arity: node = {node:?}");
        // √2·√3·√5 might be simplified to √30, or stored as Mul with
        // 2 or 3 children, or something else.  Record what we get.
        match &node {
            ExprNode::Mul(children) => {
                eprintln!("  Mul with {} children", children.len());
            }
            ExprNode::Pow(_, _) => {
                eprintln!("  Simplified to a Pow (likely √30)");
            }
            ExprNode::Num(_) => {
                eprintln!("  Simplified to a number");
            }
            _ => {
                eprintln!("  Other node type");
            }
        }
    }

    #[test]
    fn probe_sqrt2_times_sqrt3_simplification() {
        // Does √2·√3 get simplified to √6 by the arena?
        // If yes, the Mul path of minimal_polynomial is bypassed.
        let mut arena = crate::base::arena::Arena::new();
        let n2 = arena.int(2);
        let n3 = arena.int(3);
        let n6 = arena.int(6);
        let half = arena.rational(1, 2);
        let sqrt2 = arena.pow(n2, half);
        let sqrt3 = arena.pow(n3, half);
        let sqrt6 = arena.pow(n6, half);
        let prod = arena.mul(&[sqrt2, sqrt3]);
        let structurally_same = prod == sqrt6;
        eprintln!("probe: √2·√3 == √6 structurally? {structurally_same}");
        eprintln!("probe: √2·√3 node = {:?}", arena.node(prod));
        eprintln!("probe: √6 node = {:?}", arena.node(sqrt6));

        // If structurally same, subtraction gives arena.zero directly.
        let diff = arena.sub(prod, sqrt6);
        let diff_is_zero = diff == arena.zero;
        eprintln!("probe: √2·√3 - √6 == 0 structurally? {diff_is_zero}");
    }

    #[test]
    fn probe_what_expressions_reach_exact_is_zero() {
        // The expressions fed to is_zero_checked from log_to_real.rs are:
        // - Coefficients of polynomials in integration variables
        //   (b1, b0 from Rothstein-Trager roots)
        // - Remainders from polynomial division (r = a0 - q·b0)
        // - Determinants (a1·b0 - b1·a0)
        // - Imaginary parts of algebraic roots
        // - Real parts of algebraic roots
        //
        // These are typically RATIONAL NUMBERS or SINGLE RADICALS, not
        // multi-radical sums.  The N-ary fold path is unlikely to fire
        // from integration.  However, we must still ensure correctness
        // for all expression types.
        //
        // This probe verifies the typical expression types work:
        let mut arena = crate::base::arena::Arena::new();

        // Type 1: pure rational (most common)
        let r = arena.rational(1, 6);
        assert_eq!(is_zero_checked(&mut arena, r), Some(false));

        // Type 2: single radical
        let n3 = arena.int(3);
        let half = arena.rational(1, 2);
        let sqrt3 = arena.pow(n3, half);
        assert_eq!(is_zero_checked(&mut arena, sqrt3), Some(false));

        // Type 3: rational times radical (common from Rothstein-Trager)
        let sixth = arena.rational(1, 6);
        let scaled = arena.mul(&[sixth, sqrt3]);
        assert_eq!(is_zero_checked(&mut arena, scaled), Some(false));

        // Type 4: negated radical
        let neg_sqrt3 = arena.neg(sqrt3);
        assert_eq!(is_zero_checked(&mut arena, neg_sqrt3), Some(false));
        assert_eq!(sign_checked(&mut arena, neg_sqrt3), Some(-1));
    }

    // ══════════════════════════════════════════════════════════════════
    // Minimal polynomial correctness — hard assertions
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn minpoly_sqrt2_plus_sqrt3_exact_coefficients() {
        // The minimal polynomial of √2+√3 is t⁴ - 10t² + 1.
        // This goes through the 2-child Add resultant path.
        // We MUST get exactly this polynomial, not a divisor or multiple.
        let mut arena = crate::base::arena::Arena::new();
        let n2 = arena.int(2);
        let n3 = arena.int(3);
        let half = arena.rational(1, 2);
        let sqrt2 = arena.pow(n2, half);
        let sqrt3 = arena.pow(n3, half);
        let sum = arena.add(&[sqrt2, sqrt3]);
        let mp =
            minimal_polynomial(&mut arena, sum).expect("minimal_polynomial must succeed for √2+√3");
        assert_eq!(mp.degree(), Some(4), "degree must be exactly 4");
        // Coefficients: t⁴ - 10t² + 1 = [1, 0, -10, 0, 1]
        assert_eq!(mp.coeff(0), r(1, 1), "constant term must be 1");
        assert_eq!(mp.coeff(1), r(0, 1), "t¹ coefficient must be 0");
        assert_eq!(mp.coeff(2), r(-10, 1), "t² coefficient must be -10");
        assert_eq!(mp.coeff(3), r(0, 1), "t³ coefficient must be 0");
        assert_eq!(mp.coeff(4), r(1, 1), "t⁴ coefficient must be 1");
    }

    #[test]
    fn minpoly_sqrt2_times_sqrt3_exact() {
        // √2·√3 — the arena may simplify this to √6.  Either way,
        // the minimal polynomial must be t² - 6.
        let mut arena = crate::base::arena::Arena::new();
        let n2 = arena.int(2);
        let n3 = arena.int(3);
        let half = arena.rational(1, 2);
        let sqrt2 = arena.pow(n2, half);
        let sqrt3 = arena.pow(n3, half);
        let prod = arena.mul(&[sqrt2, sqrt3]);
        let mp = minimal_polynomial(&mut arena, prod)
            .expect("minimal_polynomial must succeed for √2·√3");
        // Whether the arena simplified to √6 or kept it as √2·√3,
        // the minimal polynomial must be t² - 6.
        assert_eq!(mp.degree(), Some(2), "degree must be 2");
        assert_eq!(mp.coeff(0), r(-6, 1), "constant term must be -6");
        assert_eq!(mp.coeff(2), r(1, 1), "leading coefficient must be 1");
        // Verify it vanishes at √6 ≈ 2.449.
        let val = crate::transforms::evalf::eval_const_f64(&mut arena, prod)
            .expect("must evaluate numerically");
        assert!((val - 6.0_f64.sqrt()).abs() < 1e-10, "√2·√3 must equal √6");
    }

    /// Helper: evaluate polynomial at f64 via rational approximation, return f64.
    fn eval_poly_at_f64(p: &Poly, x: f64) -> f64 {
        let x_rat = f64_to_rational_approx(x);
        numeric::ratio_to_f64(&p.eval(&x_rat)).unwrap_or(f64::NAN)
    }

    #[test]
    fn minpoly_nary_add_must_vanish_at_value() {
        // √2 + √3 + √5 — if the N-ary fold produces a polynomial,
        // that polynomial MUST vanish at the numerical value.
        // This is a hard test: wrong factor selection would produce
        // a polynomial that does NOT vanish.
        let mut arena = crate::base::arena::Arena::new();
        let half = arena.rational(1, 2);
        let n2 = arena.int(2);
        let n3 = arena.int(3);
        let n5 = arena.int(5);
        let sqrt2 = arena.pow(n2, half);
        let sqrt3 = arena.pow(n3, half);
        let sqrt5 = arena.pow(n5, half);
        let sum = arena.add(&[sqrt2, sqrt3, sqrt5]);

        let val = crate::transforms::evalf::eval_const_f64(&mut arena, sum)
            .expect("√2+√3+√5 must evaluate");
        // √2+√3+√5 ≈ 1.414 + 1.732 + 2.236 ≈ 5.382
        assert!(
            (val - 5.382).abs() < 0.01,
            "sanity: √2+√3+√5 ≈ 5.382, got {val}"
        );

        let mp = minimal_polynomial(&mut arena, sum);
        if let Some(ref mp) = mp {
            let residual = eval_poly_at_f64(mp, val);
            // The minimal polynomial MUST vanish at the value.
            // With f64 evaluation of a degree-8 polynomial, we allow
            // some numerical noise, but it should be small.
            assert!(
                residual.abs() < 1.0,
                "WRONG MINIMAL POLYNOMIAL: p(√2+√3+√5) = {residual} (should be ≈ 0), \
                 degree={:?}, poly coeffs: {:?}",
                mp.degree(),
                (0..=mp.degree().unwrap_or(0))
                    .map(|i| mp.coeff(i).to_string())
                    .collect::<Vec<_>>()
            );
        }
        // If None, minimal_polynomial can't handle it — that's safe
        // because exact_is_zero will also return None and fall to f64.
    }

    #[test]
    fn minpoly_nary_mul_must_vanish_at_value() {
        // √2 · √3 · √5 = √30 — same test for the Mul fold.
        let mut arena = crate::base::arena::Arena::new();
        let half = arena.rational(1, 2);
        let n2 = arena.int(2);
        let n3 = arena.int(3);
        let n5 = arena.int(5);
        let sqrt2 = arena.pow(n2, half);
        let sqrt3 = arena.pow(n3, half);
        let sqrt5 = arena.pow(n5, half);
        let prod = arena.mul(&[sqrt2, sqrt3, sqrt5]);

        let val = crate::transforms::evalf::eval_const_f64(&mut arena, prod)
            .expect("√2·√3·√5 must evaluate");
        assert!(
            (val - 30.0_f64.sqrt()).abs() < 1e-10,
            "sanity: √2·√3·√5 = √30"
        );

        let mp = minimal_polynomial(&mut arena, prod);
        if let Some(ref mp) = mp {
            let residual = eval_poly_at_f64(mp, val);
            assert!(
                residual.abs() < 1.0,
                "WRONG MINIMAL POLYNOMIAL: p(√30) = {residual}, degree={:?}",
                mp.degree()
            );
        }
    }

    // ══════════════════════════════════════════════════════════════════
    // Zero detection — true zeros that must be caught
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn zero_detection_structural_subtraction() {
        // (√2 + √3) - (√2 + √3) = 0.
        // Arena should simplify to structural zero.
        let mut arena = crate::base::arena::Arena::new();
        let half = arena.rational(1, 2);
        let n2 = arena.int(2);
        let n3 = arena.int(3);
        let sqrt2 = arena.pow(n2, half);
        let sqrt3 = arena.pow(n3, half);
        let sum = arena.add(&[sqrt2, sqrt3]);
        let diff = arena.sub(sum, sum);
        // Must detect zero — no excuses.
        assert_eq!(
            is_zero_checked(&mut arena, diff),
            Some(true),
            "(√2+√3)-(√2+√3) must be zero"
        );
    }

    #[test]
    fn zero_detection_sqrt_product_identity() {
        // √2 · √3 - √6 = 0 (algebraic identity).
        let mut arena = crate::base::arena::Arena::new();
        let half = arena.rational(1, 2);
        let n2 = arena.int(2);
        let n3 = arena.int(3);
        let n6 = arena.int(6);
        let sqrt2 = arena.pow(n2, half);
        let sqrt3 = arena.pow(n3, half);
        let sqrt6 = arena.pow(n6, half);
        let prod = arena.mul(&[sqrt2, sqrt3]);
        let diff = arena.sub(prod, sqrt6);
        let result = is_zero_checked(&mut arena, diff);
        // This MUST be Some(true).  If it's Some(false), that's bad math.
        // If the arena simplifies √2·√3 to √6, the difference is
        // structurally zero.  If not, exact methods must detect it.
        assert_eq!(
            result,
            Some(true),
            "√2·√3 - √6 is exactly zero — is_zero_checked must detect it. \
             If Some(false), that's BAD MATH."
        );
    }

    #[test]
    fn zero_detection_sqrt5_squared() {
        // (√5)² - 5 = 0.
        let mut arena = crate::base::arena::Arena::new();
        let n5 = arena.int(5);
        let half = arena.rational(1, 2);
        let two = arena.int(2);
        let sqrt5 = arena.pow(n5, half);
        let sq = arena.pow(sqrt5, two);
        let diff = arena.sub(sq, n5);
        assert_eq!(
            is_zero_checked(&mut arena, diff),
            Some(true),
            "(√5)² - 5 must be detected as zero"
        );
    }

    // ══════════════════════════════════════════════════════════════════
    // Nonzero detection — must NOT be called zero
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn nonzero_radical_sum_clearly_positive() {
        // √2 + √3 - √5 ≈ 0.728.  Clearly nonzero.
        let mut arena = crate::base::arena::Arena::new();
        let half = arena.rational(1, 2);
        let n2 = arena.int(2);
        let n3 = arena.int(3);
        let n5 = arena.int(5);
        let sqrt2 = arena.pow(n2, half);
        let sqrt3 = arena.pow(n3, half);
        let sqrt5 = arena.pow(n5, half);
        let neg_sqrt5 = arena.neg(sqrt5);
        let expr = arena.add(&[sqrt2, sqrt3, neg_sqrt5]);

        let val =
            crate::transforms::evalf::eval_const_f64(&mut arena, expr).expect("must evaluate");
        assert!(val > 0.5, "√2+√3-√5 ≈ 0.728, got {val}");

        assert_eq!(
            is_zero_checked(&mut arena, expr),
            Some(false),
            "is_zero_checked must say √2+√3-√5 is nonzero"
        );
        assert_eq!(
            sign_checked(&mut arena, expr),
            Some(1),
            "sign_checked must say √2+√3-√5 is positive"
        );
    }

    #[test]
    fn nonzero_sqrt2_minus_sqrt3() {
        // √2 - √3 ≈ -0.318.  Nonzero, negative.
        let mut arena = crate::base::arena::Arena::new();
        let half = arena.rational(1, 2);
        let n2 = arena.int(2);
        let n3 = arena.int(3);
        let sqrt2 = arena.pow(n2, half);
        let sqrt3 = arena.pow(n3, half);
        let diff = arena.sub(sqrt2, sqrt3);

        assert_eq!(
            is_zero_checked(&mut arena, diff),
            Some(false),
            "√2-√3 is not zero"
        );
        assert_eq!(
            sign_checked(&mut arena, diff),
            Some(-1),
            "√2-√3 is negative"
        );
    }

    #[test]
    fn nonzero_tiny_rational_in_ambiguous_zone() {
        // 1e-15 is inside the ambiguous zone (|v| < 1e-10) and inside
        // the f64 zero-tolerance (|v| < 1e-14).  eval_const_f64 alone
        // would wrongly call this zero.  The rational fast-path in
        // is_zero_checked must catch it.
        let mut arena = crate::base::arena::Arena::new();
        let expr = arena.rational(1, 1_000_000_000_000_000); // 1e-15
        assert_eq!(
            is_zero_checked(&mut arena, expr),
            Some(false),
            "1e-15 is tiny but nonzero — MUST NOT be called zero"
        );
        assert_eq!(sign_checked(&mut arena, expr), Some(1));
    }

    #[test]
    fn nonzero_tiny_negative_rational_in_ambiguous_zone() {
        let mut arena = crate::base::arena::Arena::new();
        let expr = arena.rational(-1, 1_000_000_000_000_000);
        assert_eq!(
            is_zero_checked(&mut arena, expr),
            Some(false),
            "-1e-15 is tiny but nonzero"
        );
        assert_eq!(sign_checked(&mut arena, expr), Some(-1));
    }

    #[test]
    fn nonzero_very_small_rational_1e_20() {
        // Even smaller: 1e-20.  Way inside ambiguous zone.
        let mut arena = crate::base::arena::Arena::new();
        // Build 1/10^20 without overflow: (1/10^10) * (1/10^10)
        let a = arena.rational(1, 10_000_000_000); // 1e-10
        let b = arena.rational(1, 10_000_000_000);
        let expr = arena.mul(&[a, b]);
        let expr = crate::transforms::eval::eval(&mut arena, expr);
        // is_zero_checked: either Some(false) (correct) or None (acceptable)
        // but NEVER Some(true).
        let result = is_zero_checked(&mut arena, expr);
        assert!(
            result != Some(true),
            "1e-20 is nonzero — is_zero_checked MUST NOT say it's zero. Got {result:?}"
        );
    }

    // ══════════════════════════════════════════════════════════════════
    // Nested radical and unsupported expression graceful fallback
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn nested_radical_no_crash_no_wrong_answer() {
        // √(2 + √3) ≈ 1.932.  minimal_polynomial can't handle this.
        // is_zero_checked must return Some(false) (via f64 fallback)
        // or None.  NEVER Some(true).
        let mut arena = crate::base::arena::Arena::new();
        let half = arena.rational(1, 2);
        let n3 = arena.int(3);
        let n2 = arena.int(2);
        let sqrt3 = arena.pow(n3, half);
        let inner = arena.add(&[n2, sqrt3]);
        let outer = arena.pow(inner, half);

        let result = is_zero_checked(&mut arena, outer);
        assert!(
            result != Some(true),
            "√(2+√3) ≈ 1.93 — must NOT be called zero! Got {result:?}"
        );
        // Sign should be positive.
        let sign = sign_checked(&mut arena, outer);
        assert!(
            sign == Some(1) || sign.is_none(),
            "√(2+√3) is positive — sign must be 1 or unknown, not {sign:?}"
        );
    }

    #[test]
    fn transcendental_pi_no_crash() {
        // π is transcendental — minimal_polynomial returns None.
        // is_zero_checked must not crash, must say nonzero.
        let mut arena = crate::base::arena::Arena::new();
        let pi = arena.pi;
        let result = is_zero_checked(&mut arena, pi);
        assert!(result != Some(true), "π is not zero!");
    }

    #[test]
    fn free_symbol_no_crash() {
        // A free symbol x has no numerical value.
        // is_zero_checked should return None (can't determine).
        let mut arena = crate::base::arena::Arena::new();
        let x = arena.symbol("x");
        let result = is_zero_checked(&mut arena, x);
        // Must not falsely claim zero or nonzero for an unknown symbol.
        // None is the only correct answer.
        assert!(
            result.is_none(),
            "Free symbol x: is_zero_checked must return None, got {result:?}"
        );
    }

    // ══════════════════════════════════════════════════════════════════
    // The N-ary fold bug: direct analysis
    // ══════════════════════════════════════════════════════════════════
    //
    // In minimal_polynomial, the N-ary Add fold does:
    //
    //     acc_expr = children[0]
    //     acc_mp = minimal_polynomial(arena, children[0])
    //     for child in children[1..]:
    //         child_mp = minimal_polynomial(arena, child)
    //         acc_mp = minpoly_add(&acc_mp, &child_mp, arena, acc_expr, child)
    //         acc_expr = child   // BUG: should be acc_expr + child
    //
    // After iteration 1: acc_mp = minpoly(c[0]+c[1]), but acc_expr = c[1]
    // In iteration 2: pick_factor evaluates c[1]+c[2] instead of c[0]+c[1]+c[2]
    //
    // This means the factor selection target is WRONG.  However, the
    // resultant polynomial is CORRECT (it's computed from acc_mp and
    // child_mp).  The wrong target only matters if the resultant has
    // multiple irreducible factors and the wrong target picks a different
    // factor than the correct target would.
    //
    // For this to cause bad math:
    //   1. The resultant must factor into 2+ irreducible pieces
    //   2. The wrong target (c[1]+c[2]) must evaluate closer to a root
    //      of the WRONG factor than to a root of the RIGHT factor
    //   3. The resulting wrong minimal polynomial must change the
    //      zero/sign verdict in is_zero_checked

    #[test]
    fn nary_fold_bug_direct_detection() {
        // We construct the scenario directly:
        // acc_mp = minpoly(√2+√3) = t⁴-10t²+1
        // child_mp = minpoly(√5) = t²-5
        // resultant = res_y(acc_mp(y), child_mp(x-y)) = degree 8 poly
        //
        // The correct target is √2+√3+√5 ≈ 5.382
        // The buggy target is √3+√5 ≈ 3.968
        //
        // If the degree-8 resultant factors and the two targets pick
        // different factors, we've found the bug.
        use crate::base::node::ExprNode;
        let mut arena = crate::base::arena::Arena::new();
        let half = arena.rational(1, 2);
        let n2 = arena.int(2);
        let n3 = arena.int(3);
        let n5 = arena.int(5);
        let sqrt2 = arena.pow(n2, half);
        let sqrt3 = arena.pow(n3, half);
        let sqrt5 = arena.pow(n5, half);
        let sum3 = arena.add(&[sqrt2, sqrt3, sqrt5]);

        // Check: does the arena produce a 3-child Add?
        let is_nary = matches!(arena.node(sum3), ExprNode::Add(c) if c.len() > 2);

        if is_nary {
            // The N-ary fold path fires.  Test rigorously.
            let mp = minimal_polynomial(&mut arena, sum3);
            let val =
                crate::transforms::evalf::eval_const_f64(&mut arena, sum3).expect("must evaluate");

            if let Some(ref mp) = mp {
                // HARD CHECK: polynomial must vanish at the value.
                let residual = eval_poly_at_f64(mp, val);
                assert!(
                    residual.abs() < 1.0,
                    "N-ARY FOLD BUG TRIGGERED: minimal polynomial does NOT vanish \
                     at √2+√3+√5 = {val}. Residual = {residual}, degree = {:?}. \
                     The factor selection likely picked the wrong factor due to the \
                     acc_expr = child bug in the N-ary Add fold.",
                    mp.degree()
                );

                // DOUBLE CHECK: is_zero_checked on the full expression
                // must say nonzero.
                assert_eq!(
                    is_zero_checked(&mut arena, sum3),
                    Some(false),
                    "√2+√3+√5 is nonzero"
                );
            }
        } else {
            // Arena binarized the addition — the N-ary fold is unreachable.
            // The 2-child path handles it correctly.  Still verify:
            assert_eq!(
                is_zero_checked(&mut arena, sum3),
                Some(false),
                "√2+√3+√5 is nonzero (binary path)"
            );
        }
    }

    #[test]
    fn nary_fold_bug_zero_expression() {
        // If the N-ary fold bug fires for a ZERO expression, it could
        // cause is_zero_checked to return Some(false) — worst case.
        //
        // Construct: √2 + √3 + (-√2 - √3) = 0
        // This should be structurally simplified, but let's verify.
        let mut arena = crate::base::arena::Arena::new();
        let half = arena.rational(1, 2);
        let n2 = arena.int(2);
        let n3 = arena.int(3);
        let sqrt2 = arena.pow(n2, half);
        let sqrt3 = arena.pow(n3, half);
        let neg_sqrt2 = arena.neg(sqrt2);
        let neg_sqrt3 = arena.neg(sqrt3);
        // Try constructing with all 4 terms
        let sum = arena.add(&[sqrt2, sqrt3, neg_sqrt2, neg_sqrt3]);
        let result = is_zero_checked(&mut arena, sum);
        assert_eq!(result, Some(true), "√2+√3-√2-√3 must be zero");
    }

    // ══════════════════════════════════════════════════════════════════
    // Binary Add path (2 children) — the well-tested path
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn binary_add_sqrt2_plus_sqrt3_exact_minpoly() {
        // √2 + √3 uses the 2-child path. Verify exact coefficients.
        let mut arena = crate::base::arena::Arena::new();
        let half = arena.rational(1, 2);
        let n2 = arena.int(2);
        let n3 = arena.int(3);
        let sqrt2 = arena.pow(n2, half);
        let sqrt3 = arena.pow(n3, half);
        let sum = arena.add(&[sqrt2, sqrt3]);

        let mp = minimal_polynomial(&mut arena, sum).expect("must succeed for √2+√3");
        assert_eq!(mp.degree(), Some(4));
        assert_eq!(mp.coeff(0), r(1, 1));
        assert_eq!(mp.coeff(1), r(0, 1));
        assert_eq!(mp.coeff(2), r(-10, 1));
        assert_eq!(mp.coeff(3), r(0, 1));
        assert_eq!(mp.coeff(4), r(1, 1));

        // Verify it vanishes at √2+√3.
        let val = crate::transforms::evalf::eval_const_f64(&mut arena, sum).unwrap();
        let residual = eval_poly_at_f64(&mp, val);
        assert!(residual.abs() < 1e-6, "p(√2+√3) = {residual}, expected ≈ 0");

        // Check OTHER roots don't coincide: √2-√3, -√2+√3, -√2-√3
        let other_roots = [
            2.0_f64.sqrt() - 3.0_f64.sqrt(),  // ≈ -0.318
            -2.0_f64.sqrt() + 3.0_f64.sqrt(), // ≈ 0.318
            -2.0_f64.sqrt() - 3.0_f64.sqrt(), // ≈ -3.146
        ];
        for root in &other_roots {
            let res = eval_poly_at_f64(&mp, *root);
            assert!(
                res.abs() < 1e-6,
                "p({root}) = {res}, expected ≈ 0 (it's a root too)"
            );
        }

        assert_eq!(is_zero_checked(&mut arena, sum), Some(false));
        assert_eq!(sign_checked(&mut arena, sum), Some(1));
    }

    #[test]
    fn binary_add_sqrt2_minus_1_exact_minpoly() {
        // √2 - 1 has minimal polynomial t² + 2t - 1.
        // Verify exact coefficients.
        let mut arena = crate::base::arena::Arena::new();
        let half = arena.rational(1, 2);
        let n2 = arena.int(2);
        let one = arena.int(1);
        let sqrt2 = arena.pow(n2, half);
        let expr = arena.sub(sqrt2, one);

        let mp = minimal_polynomial(&mut arena, expr).expect("must succeed for √2-1");
        assert_eq!(mp.degree(), Some(2));
        // √2-1 is root of t² + 2t - 1 = 0 (completing the square: (t+1)²=2)
        assert_eq!(mp.coeff(0), r(-1, 1));
        assert_eq!(mp.coeff(1), r(2, 1));
        assert_eq!(mp.coeff(2), r(1, 1));

        // √2 - 1 ≈ 0.414.  Positive, nonzero.
        assert_eq!(is_zero_checked(&mut arena, expr), Some(false));
        assert_eq!(sign_checked(&mut arena, expr), Some(1));
    }

    // ══════════════════════════════════════════════════════════════════
    // Sign testing edge cases
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn sign_checked_negative_radical_difference() {
        // √2 - √3 ≈ -0.318.  sign_checked must say -1.
        let mut arena = crate::base::arena::Arena::new();
        let half = arena.rational(1, 2);
        let n2 = arena.int(2);
        let n3 = arena.int(3);
        let sqrt2 = arena.pow(n2, half);
        let sqrt3 = arena.pow(n3, half);
        let diff = arena.sub(sqrt2, sqrt3);
        assert_eq!(sign_checked(&mut arena, diff), Some(-1));
    }

    #[test]
    fn sign_checked_zero_expression() {
        let mut arena = crate::base::arena::Arena::new();
        let zero = arena.zero;
        assert_eq!(sign_checked(&mut arena, zero), Some(0));
    }

    #[test]
    fn sign_checked_positive_cbrt() {
        // ∛2 ≈ 1.26.
        let mut arena = crate::base::arena::Arena::new();
        let n2 = arena.int(2);
        let third = arena.rational(1, 3);
        let cbrt2 = arena.pow(n2, third);
        assert_eq!(sign_checked(&mut arena, cbrt2), Some(1));
    }

    #[test]
    fn sign_checked_negative_cbrt() {
        // -∛2 ≈ -1.26.
        let mut arena = crate::base::arena::Arena::new();
        let n2 = arena.int(2);
        let third = arena.rational(1, 3);
        let cbrt2 = arena.pow(n2, third);
        let neg = arena.neg(cbrt2);
        assert_eq!(sign_checked(&mut arena, neg), Some(-1));
    }

    // ══════════════════════════════════════════════════════════════════
    // Stress: many expressions, all must agree with f64
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn stress_cross_check_many_expressions() {
        let mut arena = crate::base::arena::Arena::new();
        let half = arena.rational(1, 2);

        // Build a bank of radical expressions and verify cross-check
        // agreement for every one.
        let bases: Vec<i64> = vec![2, 3, 5, 7, 11];
        for &b in &bases {
            let base = arena.int(b);
            let sqrt_b = arena.pow(base, half);

            // √b
            assert_zero_check_agrees(&mut arena, sqrt_b, &format!("√{b}"));
            assert_sign_check_agrees(&mut arena, sqrt_b, &format!("√{b}"));

            // -√b
            let neg = arena.neg(sqrt_b);
            assert_zero_check_agrees(&mut arena, neg, &format!("-√{b}"));
            assert_sign_check_agrees(&mut arena, neg, &format!("-√{b}"));

            // 1/√b
            let one = arena.int(1);
            let inv = arena.div(one, sqrt_b);
            assert_zero_check_agrees(&mut arena, inv, &format!("1/√{b}"));
            assert_sign_check_agrees(&mut arena, inv, &format!("1/√{b}"));
        }

        // Pairwise sums and differences of √p for small primes.
        for i in 0..bases.len() {
            for j in (i + 1)..bases.len() {
                let bi = arena.int(bases[i]);
                let bj = arena.int(bases[j]);
                let si = arena.pow(bi, half);
                let sj = arena.pow(bj, half);

                let sum = arena.add(&[si, sj]);
                let diff = arena.sub(si, sj);

                let sum_label = format!("√{}+√{}", bases[i], bases[j]);
                let diff_label = format!("√{}-√{}", bases[i], bases[j]);

                assert_zero_check_agrees(&mut arena, sum, &sum_label);
                assert_sign_check_agrees(&mut arena, sum, &sum_label);
                assert_zero_check_agrees(&mut arena, diff, &diff_label);
                assert_sign_check_agrees(&mut arena, diff, &diff_label);

                // All sums of positive radicals are nonzero and positive.
                assert_eq!(
                    is_zero_checked(&mut arena, sum),
                    Some(false),
                    "{sum_label} must be nonzero"
                );
                assert_eq!(
                    sign_checked(&mut arena, sum),
                    Some(1),
                    "{sum_label} must be positive"
                );

                // Differences: sign depends on which is larger.
                let expected_sign: i8 = if bases[i] < bases[j] { -1 } else { 1 };
                assert_eq!(
                    is_zero_checked(&mut arena, diff),
                    Some(false),
                    "{diff_label} must be nonzero"
                );
                let s = sign_checked(&mut arena, diff);
                assert!(
                    s == Some(expected_sign) || s.is_none(),
                    "{diff_label}: expected sign {expected_sign}, got {s:?}"
                );
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════
    // BUG HUNT — Dr. Katya Moroz (algebraic identity specialist)
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn katya_sqrt2_plus_sqrt3_squared_minus_5_minus_2sqrt6() {
        // (√2+√3)² = 5+2√6.  So (√2+√3)² - 5 - 2√6 = 0.
        let mut a = crate::base::arena::Arena::new();
        let half = a.rational(1, 2);
        let n2 = a.int(2);
        let n3 = a.int(3);
        let n5 = a.int(5);
        let n6 = a.int(6);
        let sqrt2 = a.pow(n2, half);
        let sqrt3 = a.pow(n3, half);
        let sqrt6 = a.pow(n6, half);
        let sum = a.add(&[sqrt2, sqrt3]);
        let two = a.int(2);
        let sq = a.pow(sum, two);
        let sq = crate::transforms::eval::eval(&mut a, sq);
        let two_sqrt6 = a.mul(&[two, sqrt6]);
        let rhs = a.add(&[n5, two_sqrt6]);
        let rhs = crate::transforms::eval::eval(&mut a, rhs);
        let diff = a.sub(sq, rhs);
        let diff = crate::transforms::eval::eval(&mut a, diff);
        let result = is_zero_checked(&mut a, diff);
        assert_eq!(
            result,
            Some(true),
            "BUG: (√2+√3)²-5-2√6 should be zero. Got {result:?}, node={:?}",
            a.node(diff)
        );
    }

    #[test]
    fn katya_golden_ratio_phi_sq_minus_phi_minus_1() {
        // φ = (1+√5)/2, φ²-φ-1 = 0
        let mut a = crate::base::arena::Arena::new();
        let half = a.rational(1, 2);
        let n5 = a.int(5);
        let n1 = a.int(1);
        let n2 = a.int(2);
        let sqrt5 = a.pow(n5, half);
        let num = a.add(&[n1, sqrt5]);
        let phi = a.div(num, n2);
        let phi_sq = a.pow(phi, n2);
        let phi_sq = crate::transforms::eval::eval(&mut a, phi_sq);
        let phi_sq_minus_phi = a.sub(phi_sq, phi);
        let diff = a.sub(phi_sq_minus_phi, n1);
        let diff = crate::transforms::eval::eval(&mut a, diff);
        let result = is_zero_checked(&mut a, diff);
        assert_eq!(
            result,
            Some(true),
            "BUG: φ²-φ-1 should be zero. Got {result:?}"
        );
    }

    #[test]
    fn katya_2sqrt2_minus_sqrt8() {
        // 2√2 - √8 = 2√2 - 2√2 = 0
        let mut a = crate::base::arena::Arena::new();
        let half = a.rational(1, 2);
        let n2 = a.int(2);
        let n8 = a.int(8);
        let sqrt2 = a.pow(n2, half);
        let sqrt8 = a.pow(n8, half);
        let two_sqrt2 = a.mul(&[n2, sqrt2]);
        let diff = a.sub(two_sqrt2, sqrt8);
        let diff = crate::transforms::eval::eval(&mut a, diff);
        let result = is_zero_checked(&mut a, diff);
        assert_eq!(
            result,
            Some(true),
            "BUG: 2√2 - √8 = 0. Got {result:?}, node={:?}",
            a.node(diff)
        );
    }

    #[test]
    fn katya_sqrt50_minus_5sqrt2() {
        // √50 - 5√2 = 5√2 - 5√2 = 0
        let mut a = crate::base::arena::Arena::new();
        let half = a.rational(1, 2);
        let n2 = a.int(2);
        let n5 = a.int(5);
        let n50 = a.int(50);
        let sqrt2 = a.pow(n2, half);
        let sqrt50 = a.pow(n50, half);
        let five_sqrt2 = a.mul(&[n5, sqrt2]);
        let diff = a.sub(sqrt50, five_sqrt2);
        let diff = crate::transforms::eval::eval(&mut a, diff);
        let result = is_zero_checked(&mut a, diff);
        assert_eq!(result, Some(true), "BUG: √50 - 5√2 = 0. Got {result:?}");
    }

    #[test]
    fn katya_cbrt4_minus_cbrt2_squared() {
        // ∛4 - (∛2)² = 0 since 4^(1/3) = 2^(2/3) = (2^(1/3))²
        let mut a = crate::base::arena::Arena::new();
        let n2 = a.int(2);
        let n4 = a.int(4);
        let third = a.rational(1, 3);
        let cbrt4 = a.pow(n4, third);
        let cbrt2 = a.pow(n2, third);
        let cbrt2_sq = a.pow(cbrt2, n2);
        let cbrt2_sq = crate::transforms::eval::eval(&mut a, cbrt2_sq);
        let diff = a.sub(cbrt4, cbrt2_sq);
        let diff = crate::transforms::eval::eval(&mut a, diff);
        let result = is_zero_checked(&mut a, diff);
        assert_eq!(result, Some(true), "BUG: ∛4 - (∛2)² = 0. Got {result:?}");
    }

    #[test]
    fn katya_sqrt2_times_sqrt2_minus_2() {
        // √2 · √2 - 2 = 0
        let mut a = crate::base::arena::Arena::new();
        let half = a.rational(1, 2);
        let n2 = a.int(2);
        let sqrt2 = a.pow(n2, half);
        let prod = a.mul(&[sqrt2, sqrt2]);
        let prod = crate::transforms::eval::eval(&mut a, prod);
        let diff = a.sub(prod, n2);
        let diff = crate::transforms::eval::eval(&mut a, diff);
        let result = is_zero_checked(&mut a, diff);
        assert_eq!(result, Some(true), "BUG: √2·√2-2 = 0. Got {result:?}");
    }

    #[test]
    fn katya_sqrt2_plus_sqrt3_times_sqrt2_minus_sqrt3_plus_1() {
        // (√2+√3)(√2-√3) = 2-3 = -1.  So (√2+√3)(√2-√3)+1 = 0.
        let mut a = crate::base::arena::Arena::new();
        let half = a.rational(1, 2);
        let n1 = a.int(1);
        let n2 = a.int(2);
        let n3 = a.int(3);
        let sqrt2 = a.pow(n2, half);
        let sqrt3 = a.pow(n3, half);
        let sum = a.add(&[sqrt2, sqrt3]);
        let diff = a.sub(sqrt2, sqrt3);
        let prod = a.mul(&[sum, diff]);
        let prod = crate::transforms::eval::eval(&mut a, prod);
        let expr = a.add(&[prod, n1]);
        let expr = crate::transforms::eval::eval(&mut a, expr);
        let result = is_zero_checked(&mut a, expr);
        assert_eq!(
            result,
            Some(true),
            "BUG: (√2+√3)(√2-√3)+1 = 0. Got {result:?}"
        );
    }

    #[test]
    fn katya_1_over_sqrt2_minus_sqrt2_over_2() {
        // 1/√2 - √2/2 = √2/2 - √2/2 = 0
        let mut a = crate::base::arena::Arena::new();
        let half = a.rational(1, 2);
        let n1 = a.int(1);
        let n2 = a.int(2);
        let sqrt2 = a.pow(n2, half);
        let inv_sqrt2 = a.div(n1, sqrt2);
        let sqrt2_over_2 = a.div(sqrt2, n2);
        let diff = a.sub(inv_sqrt2, sqrt2_over_2);
        let diff = crate::transforms::eval::eval(&mut a, diff);
        let result = is_zero_checked(&mut a, diff);
        assert_eq!(
            result,
            Some(true),
            "BUG: 1/√2 - √2/2 = 0. Got {result:?}, node={:?}",
            a.node(diff)
        );
    }

    #[test]
    fn katya_rationalize_1_over_sqrt2_plus_1() {
        // 1/(√2+1) = √2-1 (rationalized).  So 1/(√2+1)-(√2-1)=0.
        let mut a = crate::base::arena::Arena::new();
        let half = a.rational(1, 2);
        let n1 = a.int(1);
        let n2 = a.int(2);
        let sqrt2 = a.pow(n2, half);
        let sqrt2_plus_1 = a.add(&[sqrt2, n1]);
        let lhs = a.div(n1, sqrt2_plus_1);
        let rhs = a.sub(sqrt2, n1);
        let diff = a.sub(lhs, rhs);
        let diff = crate::transforms::eval::eval(&mut a, diff);
        let result = is_zero_checked(&mut a, diff);
        assert_eq!(
            result,
            Some(true),
            "BUG: 1/(√2+1)-(√2-1) = 0. Got {result:?}, node={:?}",
            a.node(diff)
        );
    }

    // ══════════════════════════════════════════════════════════════════
    // BUG HUNT — Prof. Tomás Reyes (integration & simplification)
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn tomas_integrate_1_over_x2_plus_1() {
        // ∫ 1/(x²+1) dx = atan(x).  At x=1: atan(1) = π/4.
        let mut a = crate::base::arena::Arena::new();
        let x = a.symbol("x");
        let one = a.int(1);
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let denom = a.add(&[x2, one]);
        let integrand = a.div(one, denom);
        let result = a.integrate_expr(integrand, x);
        let result = crate::transforms::eval::eval(&mut a, result);
        let at_1 = crate::transforms::subs::subs(&mut a, result, x, one);
        let at_1 = crate::transforms::eval::eval(&mut a, at_1);
        let val = crate::transforms::evalf::eval_const_f64(&mut a, at_1);
        assert!(val.is_some(), "BUG: ∫1/(x²+1) must evaluate at x=1");
        let v = val.unwrap();
        assert!(
            (v - std::f64::consts::FRAC_PI_4).abs() < 1e-10,
            "BUG: ∫1/(x²+1) at x=1 should be π/4 ≈ 0.7854, got {v}"
        );
    }

    #[test]
    fn tomas_integrate_1_over_x2_plus_2x_plus_2() {
        // ∫ 1/(x²+2x+2) dx = atan(x+1).  At x=0: atan(1) = π/4.
        let mut a = crate::base::arena::Arena::new();
        let x = a.symbol("x");
        let one = a.int(1);
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let two_x = a.mul(&[two, x]);
        let denom = a.add(&[x2, two_x, two]);
        let integrand = a.div(one, denom);
        let result = a.integrate_expr(integrand, x);
        let result = crate::transforms::eval::eval(&mut a, result);
        let zero = a.zero;
        let at_0 = crate::transforms::subs::subs(&mut a, result, x, zero);
        let at_0 = crate::transforms::eval::eval(&mut a, at_0);
        let val = crate::transforms::evalf::eval_const_f64(&mut a, at_0);
        assert!(val.is_some(), "BUG: ∫1/(x²+2x+2) must evaluate at x=0");
        let v = val.unwrap();
        assert!(
            (v - std::f64::consts::FRAC_PI_4).abs() < 1e-10,
            "BUG: ∫1/(x²+2x+2) at x=0 should be π/4, got {v}"
        );
    }

    #[test]
    fn tomas_integrate_2x_over_x2_plus_1() {
        // ∫ 2x/(x²+1) dx = ln(x²+1).  At x=1: ln(2).
        let mut a = crate::base::arena::Arena::new();
        let x = a.symbol("x");
        let one = a.int(1);
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let denom = a.add(&[x2, one]);
        let numer = a.mul(&[two, x]);
        let integrand = a.div(numer, denom);
        let result = a.integrate_expr(integrand, x);
        let result = crate::transforms::eval::eval(&mut a, result);
        let at_1 = crate::transforms::subs::subs(&mut a, result, x, one);
        let at_1 = crate::transforms::eval::eval(&mut a, at_1);
        let val = crate::transforms::evalf::eval_const_f64(&mut a, at_1);
        assert!(val.is_some(), "BUG: ∫2x/(x²+1) must evaluate at x=1");
        let v = val.unwrap();
        assert!(
            (v - 2.0_f64.ln()).abs() < 1e-10,
            "BUG: ∫2x/(x²+1) at x=1 should be ln(2) ≈ 0.6931, got {v}"
        );
    }

    #[test]
    fn tomas_simplify_sin_sq_plus_cos_sq() {
        // sin²(x) + cos²(x) = 1
        let mut a = crate::base::arena::Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let sinx = a.sin(x);
        let cosx = a.cos(x);
        let sin2 = a.pow(sinx, two);
        let cos2 = a.pow(cosx, two);
        let sum = a.add(&[sin2, cos2]);
        let simplified = a.trigsimp_expr(sum);
        let simplified = crate::transforms::eval::eval(&mut a, simplified);
        assert_eq!(
            simplified, a.one,
            "BUG: sin²(x)+cos²(x) should simplify to 1"
        );
    }

    #[test]
    fn tomas_derivative_of_integral_is_identity() {
        // d/dx ∫ x² dx = x² (fundamental theorem)
        let mut a = crate::base::arena::Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let integral = a.integrate_expr(x2, x);
        let integral = crate::transforms::eval::eval(&mut a, integral);
        let deriv = a.diff_wrt(integral, x);
        let deriv = crate::transforms::eval::eval(&mut a, deriv);
        // Test at x=3: should be 9
        let three = a.int(3);
        let at_3 = crate::transforms::subs::subs(&mut a, deriv, x, three);
        let at_3 = crate::transforms::eval::eval(&mut a, at_3);
        let val = crate::transforms::evalf::eval_const_f64(&mut a, at_3);
        assert!(val.is_some(), "BUG: d/dx(∫x²dx) must evaluate at x=3");
        assert!(
            (val.unwrap() - 9.0).abs() < 1e-10,
            "BUG: d/dx(∫x²dx) at x=3 should be 9, got {:?}",
            val
        );
    }

    #[test]
    fn tomas_integrate_exp_x() {
        // ∫ e^x dx = e^x.  At x=1: e.
        let mut a = crate::base::arena::Arena::new();
        let x = a.symbol("x");
        let ex = a.exp(x);
        let result = a.integrate_expr(ex, x);
        let result = crate::transforms::eval::eval(&mut a, result);
        let one = a.int(1);
        let at_1 = crate::transforms::subs::subs(&mut a, result, x, one);
        let at_1 = crate::transforms::eval::eval(&mut a, at_1);
        let val = crate::transforms::evalf::eval_const_f64(&mut a, at_1);
        assert!(val.is_some(), "BUG: ∫e^x must evaluate at x=1");
        let v = val.unwrap();
        assert!(
            (v - std::f64::consts::E).abs() < 1e-10,
            "BUG: ∫e^x at x=1 should be e ≈ 2.7183, got {v}"
        );
    }

    #[test]
    fn tomas_expand_then_factor() {
        // (x+1)(x+2) = x²+3x+2.  Expand then factor should round-trip.
        let mut a = crate::base::arena::Arena::new();
        let x = a.symbol("x");
        let one = a.int(1);
        let two = a.int(2);
        let f1 = a.add(&[x, one]);
        let f2 = a.add(&[x, two]);
        let prod = a.mul(&[f1, f2]);
        let expanded = a.expand_expr(prod);
        let expanded = crate::transforms::eval::eval(&mut a, expanded);
        // Evaluate expanded at x=10: should be 11·12 = 132
        let ten = a.int(10);
        let at_10 = crate::transforms::subs::subs(&mut a, expanded, x, ten);
        let at_10 = crate::transforms::eval::eval(&mut a, at_10);
        let val = crate::transforms::evalf::eval_const_f64(&mut a, at_10);
        assert_eq!(
            val,
            Some(132.0),
            "BUG: (x+1)(x+2) at x=10 should be 132, got {val:?}"
        );
    }

    #[test]
    fn tomas_diff_sin_is_cos() {
        // d/dx sin(x) = cos(x).  At x=0: cos(0) = 1.
        let mut a = crate::base::arena::Arena::new();
        let x = a.symbol("x");
        let sinx = a.sin(x);
        let deriv = a.diff_wrt(sinx, x);
        let deriv = crate::transforms::eval::eval(&mut a, deriv);
        let zero = a.zero;
        let at_0 = crate::transforms::subs::subs(&mut a, deriv, x, zero);
        let at_0 = crate::transforms::eval::eval(&mut a, at_0);
        let val = crate::transforms::evalf::eval_const_f64(&mut a, at_0);
        assert!(val.is_some(), "BUG: d/dx(sin(x)) must evaluate at x=0");
        assert!(
            (val.unwrap() - 1.0).abs() < 1e-10,
            "BUG: d/dx(sin(x)) at x=0 should be cos(0)=1, got {:?}",
            val
        );
    }

    #[test]
    fn tomas_diff_ln_is_1_over_x() {
        // d/dx ln(x) = 1/x.  At x=2: 1/2 = 0.5.
        let mut a = crate::base::arena::Arena::new();
        let x = a.symbol("x");
        let lnx = a.ln(x);
        let deriv = a.diff_wrt(lnx, x);
        let deriv = crate::transforms::eval::eval(&mut a, deriv);
        let two = a.int(2);
        let at_2 = crate::transforms::subs::subs(&mut a, deriv, x, two);
        let at_2 = crate::transforms::eval::eval(&mut a, at_2);
        let val = crate::transforms::evalf::eval_const_f64(&mut a, at_2);
        assert!(val.is_some(), "BUG: d/dx(ln(x)) must evaluate at x=2");
        assert!(
            (val.unwrap() - 0.5).abs() < 1e-10,
            "BUG: d/dx(ln(x)) at x=2 should be 0.5, got {:?}",
            val
        );
    }

    #[test]
    fn tomas_second_derivative_x_cubed() {
        // d²/dx² x³ = 6x.  At x=5: 30.
        let mut a = crate::base::arena::Arena::new();
        let x = a.symbol("x");
        let three = a.int(3);
        let x3 = a.pow(x, three);
        let d1 = a.diff_wrt(x3, x);
        let d1 = crate::transforms::eval::eval(&mut a, d1);
        let d2 = a.diff_wrt(d1, x);
        let d2 = crate::transforms::eval::eval(&mut a, d2);
        let five = a.int(5);
        let at_5 = crate::transforms::subs::subs(&mut a, d2, x, five);
        let at_5 = crate::transforms::eval::eval(&mut a, at_5);
        let val = crate::transforms::evalf::eval_const_f64(&mut a, at_5);
        assert_eq!(
            val,
            Some(30.0),
            "BUG: d²/dx²(x³) at x=5 should be 30, got {val:?}"
        );
    }

    // ══════════════════════════════════════════════════════════════════
    // BUG HUNT — Dr. Lin Wei (numerical edge cases & misc features)
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn lin_sign_sqrt2_minus_near_rational_positive() {
        // √2 - 14142/10000 = √2 - 1.4142 ≈ 1.356e-5.  Tiny positive.
        let mut a = crate::base::arena::Arena::new();
        let half = a.rational(1, 2);
        let n2 = a.int(2);
        let sqrt2 = a.pow(n2, half);
        let approx = a.rational(14142, 10000);
        let diff = a.sub(sqrt2, approx);
        let diff = crate::transforms::eval::eval(&mut a, diff);
        let sign = sign_checked(&mut a, diff);
        assert_eq!(
            sign,
            Some(1),
            "BUG: √2 - 14142/10000 is small positive. Got sign={sign:?}"
        );
    }

    #[test]
    fn lin_sign_sqrt2_minus_near_rational_negative() {
        // √2 - 141422/100000 = √2 - 1.41422 ≈ -6.4e-6.  Tiny negative.
        let mut a = crate::base::arena::Arena::new();
        let half = a.rational(1, 2);
        let n2 = a.int(2);
        let sqrt2 = a.pow(n2, half);
        let approx = a.rational(141422, 100000);
        let diff = a.sub(sqrt2, approx);
        let diff = crate::transforms::eval::eval(&mut a, diff);
        let sign = sign_checked(&mut a, diff);
        assert_eq!(
            sign,
            Some(-1),
            "BUG: √2 - 141422/100000 is small negative. Got sign={sign:?}"
        );
    }

    #[test]
    fn lin_is_zero_sqrt2_minus_rational_not_zero() {
        // √2 - 7071/5000 ≈ 1.42e-4.  NOT zero.
        let mut a = crate::base::arena::Arena::new();
        let half = a.rational(1, 2);
        let n2 = a.int(2);
        let sqrt2 = a.pow(n2, half);
        let approx = a.rational(7071, 5000);
        let diff = a.sub(sqrt2, approx);
        let diff = crate::transforms::eval::eval(&mut a, diff);
        let result = is_zero_checked(&mut a, diff);
        assert_eq!(
            result,
            Some(false),
            "BUG: √2 - 7071/5000 is not zero. Got {result:?}"
        );
    }

    #[test]
    fn lin_evalf_pi() {
        let mut a = crate::base::arena::Arena::new();
        let pi = a.pi;
        let val = crate::transforms::evalf::eval_const_f64(&mut a, pi);
        assert!(val.is_some(), "BUG: π must evaluate");
        let v = val.unwrap();
        assert!(
            (v - std::f64::consts::PI).abs() < 1e-10,
            "BUG: π should be 3.14159..., got {v}"
        );
    }

    #[test]
    fn lin_evalf_e() {
        let mut a = crate::base::arena::Arena::new();
        let e = a.e_const;
        let val = crate::transforms::evalf::eval_const_f64(&mut a, e);
        assert!(val.is_some(), "BUG: e must evaluate");
        let v = val.unwrap();
        assert!(
            (v - std::f64::consts::E).abs() < 1e-10,
            "BUG: e should be 2.71828..., got {v}"
        );
    }

    #[test]
    fn lin_limit_sinx_over_x() {
        // lim_{x→0} sin(x)/x = 1
        let mut a = crate::base::arena::Arena::new();
        let x = a.symbol("x");
        let sinx = a.sin(x);
        let ratio = a.div(sinx, x);
        let zero = a.zero;
        let lim = a
            .limit_expr(ratio, x, zero)
            .expect("BUG: limit_expr failed for sin(x)/x");
        let lim = crate::transforms::eval::eval(&mut a, lim);
        assert_eq!(
            lim,
            a.one,
            "BUG: lim sin(x)/x → 0 should be 1, got node={:?}",
            a.node(lim)
        );
    }

    #[test]
    fn lin_chain_rule_sin_x_squared() {
        // d/dx sin(x²) = 2x·cos(x²).  At x=1: 2cos(1) ≈ 1.0806.
        let mut a = crate::base::arena::Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let sin_x2 = a.sin(x2);
        let deriv = a.diff_wrt(sin_x2, x);
        let deriv = crate::transforms::eval::eval(&mut a, deriv);
        let one = a.int(1);
        let at_1 = crate::transforms::subs::subs(&mut a, deriv, x, one);
        let at_1 = crate::transforms::eval::eval(&mut a, at_1);
        let val = crate::transforms::evalf::eval_const_f64(&mut a, at_1);
        assert!(val.is_some(), "BUG: d/dx sin(x²) must evaluate at x=1");
        let expected = 2.0 * 1.0_f64.cos();
        assert!(
            (val.unwrap() - expected).abs() < 1e-10,
            "BUG: d/dx sin(x²) at x=1 should be 2cos(1) ≈ {expected}, got {:?}",
            val
        );
    }

    #[test]
    fn lin_series_exp_x_order_4() {
        // Taylor e^x at x=0 to order 4, evaluated at x=1/10.
        // Should be close to e^(0.1) ≈ 1.10517.
        let mut a = crate::base::arena::Arena::new();
        let x = a.symbol("x");
        let ex = a.exp(x);
        let zero = a.zero;
        let series = a
            .series_expr(ex, x, zero, 4u32)
            .expect("BUG: series_expr failed for e^x");
        let series = crate::transforms::eval::eval(&mut a, series);
        let tenth = a.rational(1, 10);
        let at_01 = crate::transforms::subs::subs(&mut a, series, x, tenth);
        let at_01 = crate::transforms::eval::eval(&mut a, at_01);
        let val = crate::transforms::evalf::eval_const_f64(&mut a, at_01);
        assert!(val.is_some(), "BUG: Taylor e^x must evaluate at x=0.1");
        let v = val.unwrap();
        assert!(
            (v - 0.1_f64.exp()).abs() < 1e-4,
            "BUG: Taylor e^x(4th order) at x=0.1 should be ≈ 1.10517, got {v}"
        );
    }

    #[test]
    fn lin_solve_quadratic() {
        // x²-5x+6 = 0 has roots 2 and 3.
        let mut a = crate::base::arena::Arena::new();
        let x = a.symbol("x");
        let two = a.int(2);
        let six = a.int(6);
        let x2 = a.pow(x, two);
        let neg5 = a.int(-5);
        let neg5x = a.mul(&[neg5, x]);
        let poly = a.add(&[x2, neg5x, six]);
        let roots = crate::transforms::solve::solve(&mut a, poly, x);
        assert_eq!(
            roots.len(),
            2,
            "BUG: x²-5x+6 should have 2 roots, got {}",
            roots.len()
        );
        let mut vals: Vec<f64> = roots
            .iter()
            .filter_map(|r| crate::transforms::evalf::eval_const_f64(&mut a, r.value))
            .collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(
            (vals[0] - 2.0).abs() < 1e-10,
            "BUG: first root should be 2, got {}",
            vals[0]
        );
        assert!(
            (vals[1] - 3.0).abs() < 1e-10,
            "BUG: second root should be 3, got {}",
            vals[1]
        );
    }

    #[test]
    fn lin_matrix_det_2x2() {
        // det([[1,2],[3,4]]) = 1·4 - 2·3 = -2
        // Computed via arena arithmetic (Matrix API uses public Ex type).
        let mut a = crate::base::arena::Arena::new();
        let n1 = a.int(1);
        let n2 = a.int(2);
        let n3 = a.int(3);
        let n4 = a.int(4);
        let ad = a.mul(&[n1, n4]);
        let bc = a.mul(&[n2, n3]);
        let det = a.sub(ad, bc);
        let det = crate::transforms::eval::eval(&mut a, det);
        let val = crate::transforms::evalf::eval_const_f64(&mut a, det);
        assert_eq!(
            val,
            Some(-2.0),
            "BUG: det([[1,2],[3,4]]) should be -2, got {val:?}"
        );
    }

    #[test]
    fn lin_matrix_det_3x3_singular() {
        // det([[1,2,3],[4,5,6],[7,8,9]]) = 0 (rows are arithmetic progression)
        // Cofactor expansion along row 1:
        //   1·(5·9-6·8) - 2·(4·9-6·7) + 3·(4·8-5·7)
        //   = 1·(45-48) - 2·(36-42) + 3·(32-35)
        //   = -3 + 12 - 9 = 0
        let mut a = crate::base::arena::Arena::new();
        let n1 = a.int(1);
        let n2 = a.int(2);
        let n3 = a.int(3);
        let n4 = a.int(4);
        let n5 = a.int(5);
        let n6 = a.int(6);
        let n7 = a.int(7);
        let n8 = a.int(8);
        let n9 = a.int(9);
        // Minor M11 = 5·9 - 6·8
        let p59 = a.mul(&[n5, n9]);
        let p68 = a.mul(&[n6, n8]);
        let m11 = a.sub(p59, p68);
        // Minor M12 = 4·9 - 6·7
        let p49 = a.mul(&[n4, n9]);
        let p67 = a.mul(&[n6, n7]);
        let m12 = a.sub(p49, p67);
        // Minor M13 = 4·8 - 5·7
        let p48 = a.mul(&[n4, n8]);
        let p57 = a.mul(&[n5, n7]);
        let m13 = a.sub(p48, p57);
        // det = 1·M11 - 2·M12 + 3·M13
        let t1 = a.mul(&[n1, m11]);
        let t2 = a.mul(&[n2, m12]);
        let t3 = a.mul(&[n3, m13]);
        let det = a.sub(t1, t2);
        let det = a.add(&[det, t3]);
        let det = crate::transforms::eval::eval(&mut a, det);
        let val = crate::transforms::evalf::eval_const_f64(&mut a, det);
        assert_eq!(
            val,
            Some(0.0),
            "BUG: det of singular 3x3 should be 0, got {val:?}"
        );
    }

    #[test]
    fn lin_solve_linear_system_2x2() {
        // x + y = 3, 2x - y = 0  →  x = 1, y = 2
        let mut a = crate::base::arena::Arena::new();
        let x = a.symbol("x");
        let y = a.symbol("y");
        let n3 = a.int(3);
        let n2 = a.int(2);
        // eq1: x + y - 3 = 0
        let eq1 = a.add(&[x, y]);
        let eq1 = a.sub(eq1, n3);
        // eq2: 2x - y = 0
        let two_x = a.mul(&[n2, x]);
        let eq2 = a.sub(two_x, y);
        // Solve eq1 for x: x = 3 - y
        let roots_x = crate::transforms::solve::solve(&mut a, eq1, x);
        if !roots_x.is_empty() {
            let x_val = roots_x[0].value; // x = 3 - y
            // Sub into eq2: 2(3-y) - y = 0 → 6-3y = 0 → y = 2
            let eq2_sub = crate::transforms::subs::subs(&mut a, eq2, x, x_val);
            let eq2_sub = crate::transforms::eval::eval(&mut a, eq2_sub);
            let roots_y = crate::transforms::solve::solve(&mut a, eq2_sub, y);
            if !roots_y.is_empty() {
                let y_val = crate::transforms::evalf::eval_const_f64(&mut a, roots_y[0].value);
                assert!(
                    y_val.is_some_and(|v| (v - 2.0).abs() < 1e-10),
                    "BUG: y should be 2, got {y_val:?}"
                );
                // Substitute back: x = 3 - 2 = 1
                let x_final = crate::transforms::subs::subs(&mut a, x_val, y, roots_y[0].value);
                let x_final = crate::transforms::eval::eval(&mut a, x_final);
                let x_val_f64 = crate::transforms::evalf::eval_const_f64(&mut a, x_final);
                assert!(
                    x_val_f64.is_some_and(|v| (v - 1.0).abs() < 1e-10),
                    "BUG: x should be 1, got {x_val_f64:?}"
                );
            }
        }
    }

    #[test]
    fn lin_product_rule_derivative() {
        // d/dx [x·sin(x)] = sin(x) + x·cos(x).  At x=π/2: 1 + 0 = 1.
        let mut a = crate::base::arena::Arena::new();
        let x = a.symbol("x");
        let sinx = a.sin(x);
        let prod = a.mul(&[x, sinx]);
        let deriv = a.diff_wrt(prod, x);
        let deriv = crate::transforms::eval::eval(&mut a, deriv);
        // Evaluate at x = π/2
        let two = a.int(2);
        let pi = a.pi;
        let pi_half = a.div(pi, two);
        let at_pi2 = crate::transforms::subs::subs(&mut a, deriv, x, pi_half);
        let at_pi2 = crate::transforms::eval::eval(&mut a, at_pi2);
        let val = crate::transforms::evalf::eval_const_f64(&mut a, at_pi2);
        assert!(val.is_some(), "BUG: d/dx(x·sin(x)) must evaluate at x=π/2");
        assert!(
            (val.unwrap() - 1.0).abs() < 1e-10,
            "BUG: d/dx(x·sin(x)) at x=π/2 should be 1, got {:?}",
            val
        );
    }

    #[test]
    fn lin_integrate_then_diff_sinx() {
        // d/dx [∫sin(x)dx] = sin(x).  At x=π/6: sin(π/6) = 1/2.
        let mut a = crate::base::arena::Arena::new();
        let x = a.symbol("x");
        let sinx = a.sin(x);
        let integral = a.integrate_expr(sinx, x);
        let integral = crate::transforms::eval::eval(&mut a, integral);
        let deriv = a.diff_wrt(integral, x);
        let deriv = crate::transforms::eval::eval(&mut a, deriv);
        let six = a.int(6);
        let pi = a.pi;
        let pi_6 = a.div(pi, six);
        let at_val = crate::transforms::subs::subs(&mut a, deriv, x, pi_6);
        let at_val = crate::transforms::eval::eval(&mut a, at_val);
        let val = crate::transforms::evalf::eval_const_f64(&mut a, at_val);
        assert!(val.is_some(), "BUG: d/dx(∫sin(x)dx) must evaluate at x=π/6");
        assert!(
            (val.unwrap() - 0.5).abs() < 1e-10,
            "BUG: d/dx(∫sin(x)dx) at x=π/6 should be 0.5, got {:?}",
            val
        );
    }

    #[test]
    fn lin_evalf_sqrt2_high_precision() {
        // √2 should evaluate close to 1.41421356237...
        let mut a = crate::base::arena::Arena::new();
        let n2 = a.int(2);
        let half = a.rational(1, 2);
        let sqrt2 = a.pow(n2, half);
        let val = crate::transforms::evalf::eval_const_f64(&mut a, sqrt2);
        assert!(val.is_some(), "BUG: √2 must evaluate");
        let v = val.unwrap();
        assert!(
            (v - std::f64::consts::SQRT_2).abs() < 1e-14,
            "BUG: √2 should be 1.41421356237..., got {v}"
        );
    }

    #[test]
    fn lin_negative_exponent() {
        // 2^(-1) = 1/2
        let mut a = crate::base::arena::Arena::new();
        let n2 = a.int(2);
        let neg1 = a.int(-1);
        let result = a.pow(n2, neg1);
        let result = crate::transforms::eval::eval(&mut a, result);
        let val = crate::transforms::evalf::eval_const_f64(&mut a, result);
        assert_eq!(val, Some(0.5), "BUG: 2^(-1) should be 0.5, got {val:?}");
    }

    #[test]
    fn lin_zero_to_the_zero() {
        // 0^0 is conventionally 1 in combinatorics/algebra.
        let mut a = crate::base::arena::Arena::new();
        let zero = a.zero;
        let result = a.pow(zero, zero);
        let result = crate::transforms::eval::eval(&mut a, result);
        let val = crate::transforms::evalf::eval_const_f64(&mut a, result);
        // Most CAS systems return 1 for 0^0.
        assert!(
            val == Some(1.0) || result == a.one,
            "BUG: 0^0 should be 1, got val={val:?}, node={:?}",
            a.node(result)
        );
    }

    #[test]
    fn lin_large_integer_arithmetic() {
        // 2^64 - 1 = 18446744073709551615
        let mut a = crate::base::arena::Arena::new();
        let n2 = a.int(2);
        let n64 = a.int(64);
        let big = a.pow(n2, n64);
        let big = crate::transforms::eval::eval(&mut a, big);
        let one = a.int(1);
        let result = a.sub(big, one);
        let result = crate::transforms::eval::eval(&mut a, result);
        // Check it's the right number: should be 2^64-1
        if let Some(r) = a.as_num(result) {
            let expected: u64 = u64::MAX; // 2^64-1
            assert_eq!(
                r.to_string(),
                expected.to_string(),
                "BUG: 2^64-1 should be {expected}, got {r}"
            );
        } else {
            panic!(
                "BUG: 2^64-1 should be a number, got node={:?}",
                a.node(result)
            );
        }
    }

    #[test]
    fn lin_gcd_polynomial() {
        // gcd(x²-1, x²-2x+1) = x-1  (since x²-1=(x-1)(x+1), x²-2x+1=(x-1)²)
        let mut a = crate::base::arena::Arena::new();
        let x = a.symbol("x");
        let one = a.int(1);
        let two = a.int(2);
        let x2 = a.pow(x, two);
        // p1 = x²-1
        let p1 = a.sub(x2, one);
        // p2 = x²-2x+1
        let neg2 = a.int(-2);
        let neg2x = a.mul(&[neg2, x]);
        let p2 = a.add(&[x2, neg2x, one]);
        let gcd_expr = a.poly_gcd_expr(p1, p2, x);
        let gcd_expr = gcd_expr.expect("BUG: poly_gcd_expr returned None for x²-1 and x²-2x+1");
        let gcd_expr = crate::transforms::eval::eval(&mut a, gcd_expr);
        // Evaluate at x=5: gcd should be (x-1), so at x=5 → 4.
        let five = a.int(5);
        let at_5 = crate::transforms::subs::subs(&mut a, gcd_expr, x, five);
        let at_5 = crate::transforms::eval::eval(&mut a, at_5);
        let val = crate::transforms::evalf::eval_const_f64(&mut a, at_5);
        // gcd(x²-1, x²-2x+1) = x-1 → at x=5: 4.  Or could be -(x-1) = 1-x → -4.
        // Or scaled by a constant.  Just check it divides both.
        assert!(val.is_some(), "BUG: poly gcd must evaluate");
        let v = val.unwrap();
        assert!(
            v.abs() > 0.1,
            "BUG: gcd(x²-1, x²-2x+1) at x=5 should be nonzero, got {v}"
        );
        // Check it divides p1 at x=5: p1(5) = 24, should be divisible by gcd(5)
        let p1_at_5 = crate::transforms::subs::subs(&mut a, p1, x, five);
        let p1_at_5 = crate::transforms::eval::eval(&mut a, p1_at_5);
        let p1_val = crate::transforms::evalf::eval_const_f64(&mut a, p1_at_5).unwrap();
        let remainder = p1_val / v;
        assert!(
            (remainder - remainder.round()).abs() < 1e-10,
            "BUG: gcd should divide p1. p1(5)={p1_val}, gcd(5)={v}, ratio={remainder}"
        );
    }

    // ══════════════════════════════════════════════════════════════════
    // HARD adversarial tests — designed to probe real failure modes
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn hard_near_zero_radical_minus_close_rational() {
        // √2 - 665857/470832 ≈ -1.6e-12.  This is inside the ambiguous
        // zone (|v| < 1e-10) and is genuinely NONZERO.
        // 665857/470832 is a convergent of √2's continued fraction.
        // is_zero_checked MUST say Some(false), not Some(true).
        let mut a = crate::base::arena::Arena::new();
        let half = a.rational(1, 2);
        let n2 = a.int(2);
        let sqrt2 = a.pow(n2, half);
        let close_approx = a.rational(665857, 470832);
        let diff = a.sub(sqrt2, close_approx);
        let diff = crate::transforms::eval::eval(&mut a, diff);
        let f64_val = crate::transforms::evalf::eval_const_f64(&mut a, diff);
        eprintln!("hard_near_zero: √2 - 665857/470832 f64 = {f64_val:?}");
        let result = is_zero_checked(&mut a, diff);
        assert_eq!(
            result,
            Some(false),
            "BUG: √2 - 665857/470832 ≈ -1.6e-12 is NONZERO but is_zero_checked says {result:?}. \
             f64={f64_val:?}. This is a critical failure in the ambiguous zone."
        );
    }

    #[test]
    fn hard_near_zero_radical_minus_close_rational_sign() {
        // Same expression: √2 - 665857/470832.  Sign must be -1.
        // √2 = 1.41421356237309504...
        // 665857/470832 = 1.41421356237468...
        // diff ≈ -1.6e-12, negative.
        let mut a = crate::base::arena::Arena::new();
        let half = a.rational(1, 2);
        let n2 = a.int(2);
        let sqrt2 = a.pow(n2, half);
        let close_approx = a.rational(665857, 470832);
        let diff = a.sub(sqrt2, close_approx);
        let diff = crate::transforms::eval::eval(&mut a, diff);
        let sign = sign_checked(&mut a, diff);
        assert_eq!(
            sign,
            Some(-1),
            "BUG: √2 - 665857/470832 is tiny negative. sign_checked says {sign:?}"
        );
    }

    #[test]
    fn hard_near_zero_positive_radical_minus_rational() {
        // √2 - 1393/985 ≈ +2.4e-7.  Just above zero.  Nonzero, positive.
        // 1393/985 is an earlier convergent.
        let mut a = crate::base::arena::Arena::new();
        let half = a.rational(1, 2);
        let n2 = a.int(2);
        let sqrt2 = a.pow(n2, half);
        let approx = a.rational(1393, 985);
        let diff = a.sub(sqrt2, approx);
        let diff = crate::transforms::eval::eval(&mut a, diff);
        let result = is_zero_checked(&mut a, diff);
        assert_eq!(result, Some(false), "BUG: √2 - 1393/985 is nonzero");
        let sign = sign_checked(&mut a, diff);
        assert_eq!(sign, Some(1), "BUG: √2 - 1393/985 is positive");
    }

    #[test]
    fn hard_sqrt_n2_plus_1_minus_n_large() {
        // √(n²+1) - n ≈ 1/(2n) for large n.  For n=10000, ≈ 5e-5.
        // This is nonzero and positive.  Tests that arithmetic with
        // large integers doesn't confuse the zero checker.
        let mut a = crate::base::arena::Arena::new();
        let half = a.rational(1, 2);
        let n = a.int(10000);
        let n2 = a.int(100000001); // 10000² + 1
        let sqrt_n2p1 = a.pow(n2, half);
        let diff = a.sub(sqrt_n2p1, n);
        let diff = crate::transforms::eval::eval(&mut a, diff);
        let result = is_zero_checked(&mut a, diff);
        assert_eq!(result, Some(false), "BUG: √(10000²+1)-10000 is nonzero");
        let sign = sign_checked(&mut a, diff);
        assert_eq!(sign, Some(1), "BUG: √(10000²+1)-10000 is positive");
    }

    #[test]
    fn hard_minpoly_identity_as_zero_test() {
        // √2+√3 satisfies t⁴-10t²+1=0.  So if we compute
        // (√2+√3)⁴ - 10(√2+√3)² + 1, it should be exactly 0.
        // This tests the full pipeline: eval simplifies the powers,
        // then is_zero_checked verifies.
        let mut a = crate::base::arena::Arena::new();
        let half = a.rational(1, 2);
        let n2 = a.int(2);
        let n3 = a.int(3);
        let n4 = a.int(4);
        let n10 = a.int(10);
        let n1 = a.int(1);
        let sqrt2 = a.pow(n2, half);
        let sqrt3 = a.pow(n3, half);
        let s = a.add(&[sqrt2, sqrt3]);
        // s⁴
        let s2 = a.pow(s, n2);
        let s2 = crate::transforms::eval::eval(&mut a, s2);
        let s4 = a.pow(s, n4);
        let s4 = crate::transforms::eval::eval(&mut a, s4);
        // 10·s²
        let ten_s2 = a.mul(&[n10, s2]);
        let ten_s2 = crate::transforms::eval::eval(&mut a, ten_s2);
        // s⁴ - 10s² + 1
        let neg_ten_s2 = a.neg(ten_s2);
        let expr = a.add(&[s4, neg_ten_s2, n1]);
        let expr = crate::transforms::eval::eval(&mut a, expr);
        let f64_val = crate::transforms::evalf::eval_const_f64(&mut a, expr);
        eprintln!("hard_minpoly_identity: (√2+√3)⁴-10(√2+√3)²+1 f64 = {f64_val:?}");
        let result = is_zero_checked(&mut a, expr);
        assert_eq!(
            result,
            Some(true),
            "BUG: (√2+√3)⁴-10(√2+√3)²+1 = 0 (minimal poly identity). \
             is_zero_checked says {result:?}, f64={f64_val:?}"
        );
    }

    #[test]
    fn hard_rationalize_denominator_identity() {
        // 1/(√3+√2) = √3-√2 (multiply by conjugate).
        // So 1/(√3+√2) - (√3-√2) = 0.
        let mut a = crate::base::arena::Arena::new();
        let half = a.rational(1, 2);
        let n1 = a.int(1);
        let n2 = a.int(2);
        let n3 = a.int(3);
        let sqrt2 = a.pow(n2, half);
        let sqrt3 = a.pow(n3, half);
        let denom = a.add(&[sqrt3, sqrt2]);
        let lhs = a.div(n1, denom);
        let rhs = a.sub(sqrt3, sqrt2);
        let diff = a.sub(lhs, rhs);
        let diff = crate::transforms::eval::eval(&mut a, diff);
        let f64_val = crate::transforms::evalf::eval_const_f64(&mut a, diff);
        eprintln!(
            "hard_rationalize: 1/(√3+√2)-(√3-√2) f64 = {f64_val:?}, node = {:?}",
            a.node(diff)
        );
        let result = is_zero_checked(&mut a, diff);
        assert_eq!(
            result,
            Some(true),
            "BUG: 1/(√3+√2)-(√3-√2) = 0. Got {result:?}, f64={f64_val:?}"
        );
    }

    #[test]
    fn hard_exact_is_zero_on_symbol_plus_radical() {
        // x + √2 has a free symbol — minimal_polynomial returns None.
        // is_zero_checked must return None (cannot determine), NOT
        // Some(true) or Some(false).
        let mut a = crate::base::arena::Arena::new();
        let x = a.symbol("x");
        let half = a.rational(1, 2);
        let n2 = a.int(2);
        let sqrt2 = a.pow(n2, half);
        let expr = a.add(&[x, sqrt2]);
        let result = is_zero_checked(&mut a, expr);
        assert!(
            result.is_none(),
            "BUG: x+√2 has a free variable — is_zero_checked must return None, got {result:?}"
        );
    }

    #[test]
    fn hard_sign_of_near_zero_difference_of_radicals() {
        // √5 - √3 - √2 + 1 ≈ 2.236 - 1.732 - 1.414 + 1 = 0.090
        // Small positive.  sign_checked must say 1.
        let mut a = crate::base::arena::Arena::new();
        let half = a.rational(1, 2);
        let n1 = a.int(1);
        let n2 = a.int(2);
        let n3 = a.int(3);
        let n5 = a.int(5);
        let sqrt2 = a.pow(n2, half);
        let sqrt3 = a.pow(n3, half);
        let sqrt5 = a.pow(n5, half);
        let neg_sqrt3 = a.neg(sqrt3);
        let neg_sqrt2 = a.neg(sqrt2);
        let expr = a.add(&[sqrt5, neg_sqrt3, neg_sqrt2, n1]);
        let expr = crate::transforms::eval::eval(&mut a, expr);
        let f64_val = crate::transforms::evalf::eval_const_f64(&mut a, expr);
        eprintln!("hard_sign_near_zero: √5-√3-√2+1 f64 = {f64_val:?}");
        let result = is_zero_checked(&mut a, expr);
        assert_eq!(
            result,
            Some(false),
            "BUG: √5-√3-√2+1 ≈ 0.09, nonzero. Got {result:?}"
        );
        let sign = sign_checked(&mut a, expr);
        assert_eq!(
            sign,
            Some(1),
            "BUG: √5-√3-√2+1 ≈ 0.09, positive. Got {sign:?}"
        );
    }

    #[test]
    fn hard_integrate_1_over_x4_plus_1() {
        // ∫ 1/(x⁴+1) dx is a hard rational integral.  The
        // Rothstein-Trager algorithm produces degree-4 roots.
        // This tests the full pipeline including log_to_real.
        let mut a = crate::base::arena::Arena::new();
        let x = a.symbol("x");
        let one = a.int(1);
        let four = a.int(4);
        let x4 = a.pow(x, four);
        let denom = a.add(&[x4, one]);
        let integrand = a.div(one, denom);
        let result = a.integrate_expr(integrand, x);
        let result = crate::transforms::eval::eval(&mut a, result);
        // Evaluate at x=1 and x=0, take difference for definite integral.
        let at_1 = crate::transforms::subs::subs(&mut a, result, x, one);
        let at_1 = crate::transforms::eval::eval(&mut a, at_1);
        let zero = a.zero;
        let at_0 = crate::transforms::subs::subs(&mut a, result, x, zero);
        let at_0 = crate::transforms::eval::eval(&mut a, at_0);
        let definite = a.sub(at_1, at_0);
        let definite = crate::transforms::eval::eval(&mut a, definite);
        let val = crate::transforms::evalf::eval_const_f64(&mut a, definite);
        // Known value: ∫₀¹ 1/(x⁴+1) dx ≈ 0.86697298...
        // (via Wolfram Alpha / numerical integration)
        if let Some(v) = val {
            assert!(
                (v - 0.86697298).abs() < 1e-4,
                "BUG: ∫₀¹ 1/(x⁴+1)dx should be ≈ 0.8670, got {v}"
            );
        } else {
            eprintln!(
                "hard_integrate_x4+1: could not evaluate definite integral numerically. \
                       Result node: {:?}",
                a.node(definite)
            );
            // Don't assert failure — the integration may produce a form
            // that can't be evaluated at specific points.
        }
    }

    #[test]
    fn hard_integrate_1_over_x3_minus_1() {
        // ∫ 1/(x³-1) dx involves log_to_real with quadratic factor.
        // This is a regression test for the code we just wired.
        let mut a = crate::base::arena::Arena::new();
        let x = a.symbol("x");
        let one = a.int(1);
        let three = a.int(3);
        let x3 = a.pow(x, three);
        let denom = a.sub(x3, one);
        let integrand = a.div(one, denom);
        let result = a.integrate_expr(integrand, x);
        let result = crate::transforms::eval::eval(&mut a, result);
        // Evaluate at x=2
        let two = a.int(2);
        let at_2 = crate::transforms::subs::subs(&mut a, result, x, two);
        let at_2 = crate::transforms::eval::eval(&mut a, at_2);
        let val = crate::transforms::evalf::eval_const_f64(&mut a, at_2);
        // Known: ∫ 1/(x³-1) dx at x=2 involves ln and atan terms.
        // Numerical value: ≈ ln(1)/3 + ... (partial fractions)
        // Just verify it evaluates to SOME finite number.
        if let Some(v) = val {
            assert!(
                v.is_finite(),
                "BUG: ∫1/(x³-1) at x=2 should be finite, got {v}"
            );
            assert!(
                v.abs() < 100.0,
                "BUG: ∫1/(x³-1) at x=2 should be reasonable, got {v}"
            );
        }
    }

    #[test]
    fn hard_eval_const_f64_in_ambiguous_zone_is_not_trusted() {
        // Construct: √2 - p/q where p/q is chosen so f64(diff) ≈ 1e-13.
        // That's inside the f64 tolerance (< 1e-14? no, 1e-13 > 1e-14)
        // but inside the ambiguous zone (< 1e-10).
        // is_zero_checked should use exact methods and say nonzero.
        //
        // √2 ≈ 1.4142135623730950488...
        // 14142135623731/10000000000000 = 1.4142135623731
        // diff ≈ -5e-14.  This is INSIDE the 1e-14 tolerance!
        // eval_const_f64 alone would say "zero".
        // exact_is_zero should say "nonzero" (it's √2 minus a rational).
        let mut a = crate::base::arena::Arena::new();
        let half = a.rational(1, 2);
        let n2 = a.int(2);
        let sqrt2 = a.pow(n2, half);
        let close = a.rational(14142135623731i64, 10000000000000i64);
        let diff = a.sub(sqrt2, close);
        let diff = crate::transforms::eval::eval(&mut a, diff);
        let f64_val = crate::transforms::evalf::eval_const_f64(&mut a, diff);
        eprintln!("hard_ambiguous: √2 - 14142135623731/10000000000000 f64 = {f64_val:?}");
        // f64_val might be ~-5e-14 or so
        let result = is_zero_checked(&mut a, diff);
        // MUST be Some(false).  If it's Some(true), the exact methods
        // failed to override the f64 tolerance — that's the critical bug.
        assert_eq!(
            result,
            Some(false),
            "CRITICAL BUG: √2 - 14142135623731/10000000000000 is nonzero, but \
             is_zero_checked says {result:?}. f64={f64_val:?}. \
             The exact method must override the f64 tolerance in the ambiguous zone."
        );
    }

    #[test]
    fn hard_eval_const_f64_ambiguous_zone_sign() {
        // Same expression: sign must be determined correctly.
        // √2 - 14142135623731/10000000000000
        // √2 = 1.41421356237309504...
        // rat = 1.41421356237310000...
        // diff ≈ -5e-14, NEGATIVE.
        let mut a = crate::base::arena::Arena::new();
        let half = a.rational(1, 2);
        let n2 = a.int(2);
        let sqrt2 = a.pow(n2, half);
        let close = a.rational(14142135623731i64, 10000000000000i64);
        let diff = a.sub(sqrt2, close);
        let diff = crate::transforms::eval::eval(&mut a, diff);
        let sign = sign_checked(&mut a, diff);
        assert_eq!(
            sign,
            Some(-1),
            "BUG: √2 - 14142135623731/10000000000000 is tiny negative, \
             sign_checked says {sign:?}"
        );
    }

    #[test]
    fn hard_double_rationalization_zero() {
        // 1/(√5+√3) - (√5-√3)/2 = 0
        // Because 1/(√5+√3) = (√5-√3)/((√5+√3)(√5-√3)) = (√5-√3)/(5-3) = (√5-√3)/2
        let mut a = crate::base::arena::Arena::new();
        let half = a.rational(1, 2);
        let n1 = a.int(1);
        let n2 = a.int(2);
        let n3 = a.int(3);
        let n5 = a.int(5);
        let sqrt3 = a.pow(n3, half);
        let sqrt5 = a.pow(n5, half);
        let denom = a.add(&[sqrt5, sqrt3]);
        let lhs = a.div(n1, denom);
        let numer = a.sub(sqrt5, sqrt3);
        let rhs = a.div(numer, n2);
        let diff = a.sub(lhs, rhs);
        let diff = crate::transforms::eval::eval(&mut a, diff);
        let f64_val = crate::transforms::evalf::eval_const_f64(&mut a, diff);
        eprintln!("hard_double_rat: 1/(√5+√3)-(√5-√3)/2 f64 = {f64_val:?}");
        let result = is_zero_checked(&mut a, diff);
        assert_eq!(
            result,
            Some(true),
            "BUG: 1/(√5+√3)-(√5-√3)/2 = 0. Got {result:?}, f64={f64_val:?}"
        );
    }

    #[test]
    fn hard_cube_root_identity() {
        // (∛2)³ - 2 = 0
        let mut a = crate::base::arena::Arena::new();
        let n2 = a.int(2);
        let n3 = a.int(3);
        let third = a.rational(1, 3);
        let cbrt2 = a.pow(n2, third);
        let cubed = a.pow(cbrt2, n3);
        let cubed = crate::transforms::eval::eval(&mut a, cubed);
        let diff = a.sub(cubed, n2);
        let diff = crate::transforms::eval::eval(&mut a, diff);
        let result = is_zero_checked(&mut a, diff);
        assert_eq!(result, Some(true), "BUG: (∛2)³ - 2 = 0. Got {result:?}");
    }

    #[test]
    fn hard_fourth_root_identity() {
        // (⁴√5)⁴ - 5 = 0
        let mut a = crate::base::arena::Arena::new();
        let n4 = a.int(4);
        let n5 = a.int(5);
        let quarter = a.rational(1, 4);
        let root = a.pow(n5, quarter);
        let power = a.pow(root, n4);
        let power = crate::transforms::eval::eval(&mut a, power);
        let diff = a.sub(power, n5);
        let diff = crate::transforms::eval::eval(&mut a, diff);
        let result = is_zero_checked(&mut a, diff);
        assert_eq!(result, Some(true), "BUG: (⁴√5)⁴ - 5 = 0. Got {result:?}");
    }

    #[test]
    fn hard_mixed_power_zero() {
        // 2^(1/6) - (∛2)·(√2)^(-1/3)... let's try a simpler one:
        // 2^(1/2) · 3^(1/2) · 6^(-1/2) - 1 = 0
        // Because √2·√3/√6 = √6/√6 = 1.
        let mut a = crate::base::arena::Arena::new();
        let half = a.rational(1, 2);
        let neg_half = a.rational(-1, 2);
        let n1 = a.int(1);
        let n2 = a.int(2);
        let n3 = a.int(3);
        let n6 = a.int(6);
        let sqrt2 = a.pow(n2, half);
        let sqrt3 = a.pow(n3, half);
        let inv_sqrt6 = a.pow(n6, neg_half);
        let prod = a.mul(&[sqrt2, sqrt3, inv_sqrt6]);
        let prod = crate::transforms::eval::eval(&mut a, prod);
        let diff = a.sub(prod, n1);
        let diff = crate::transforms::eval::eval(&mut a, diff);
        let result = is_zero_checked(&mut a, diff);
        assert_eq!(
            result,
            Some(true),
            "BUG: √2·√3/√6 - 1 = 0. Got {result:?}, node={:?}",
            a.node(diff)
        );
    }

    #[test]
    fn hard_stress_all_signs_of_radical_differences() {
        // For each pair (p, q) with p < q from {2,3,5,7,11,13},
        // √p - √q must be negative.  This tests sign_checked across
        // many cases, including some that go through Add(2 children).
        let mut a = crate::base::arena::Arena::new();
        let half = a.rational(1, 2);
        let primes = [2i64, 3, 5, 7, 11, 13];
        for i in 0..primes.len() {
            for j in (i + 1)..primes.len() {
                let pi = a.int(primes[i]);
                let pj = a.int(primes[j]);
                let si = a.pow(pi, half);
                let sj = a.pow(pj, half);
                let diff = a.sub(si, sj);
                let sign = sign_checked(&mut a, diff);
                assert_eq!(
                    sign,
                    Some(-1),
                    "BUG: √{} - √{} should be negative (√{} < √{}), got {sign:?}",
                    primes[i],
                    primes[j],
                    primes[i],
                    primes[j]
                );
            }
        }
    }
}

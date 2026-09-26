//! Error bounds for numerical evaluation.
//!
//! `evalf` evaluates an expression bottom-up at a fixed binary working
//! precision.  Alongside every value `re + i·im` it keeps a [`Bound`]: an
//! estimate `2^e` of the absolute error of *each part*, propagated from the
//! children's bounds to first order ([`node_error`]).  Two bounds rather than
//! one because a part can be exactly zero: the imaginary part of a real
//! expression is 0 with no error at all, while an imaginary part that
//! cancelled to 0 (or to a rounding residue) at the working precision is only
//! known to lie within its error of 0.  That difference decides branch cuts
//! (below).
//!
//! | node | error of each part |
//! |---|---|
//! | exact rational (dyadic, fits the precision), `i` | 0 |
//! | other rational, `π`, `e`, constants | rounding: `2^(mag − prec)`; the imaginary part is exact |
//! | `a₁ + … + aₙ` | per part `Σ err(aᵢ)` plus the rounding of the partial sums — absolute errors add, so cancellation shows as a result much smaller than its error; exact when every term's part is exact and no partial sum was rounded |
//! | `a₁ · … · aₙ` | per part, from `(a + bi)(c + di) = (ac − bd) + (ad + bc)i`: each product `x·y` contributes `abs(x)·err(y) + abs(y)·err(x) + err(x)·err(y)` and its rounding, and none at all when a factor is an exact 0 |
//! | `b^n`, `b^p`, `b^e` | relative errors scale by the exponent, plus `abs(ln b)·err(e)`; a positive power `p` of a value indistinguishable from 0 is bounded by `2·(abs(b) + err(b))^p`, a negative one has no bound |
//! | `exp z` | `abs(exp z)·err(z)` |
//! | `ln z`, `arg z` | `err(z)/abs(z)`; no bound when `z` is indistinguishable from 0 |
//! | `sin`, `cos`, `sinh`, `cosh` | `max(1, abs(f(z)))·err(z)` |
//! | `tan`, `tanh` | `(1 + abs(f(z))²)·err(z)` |
//! | `atan`, `asin`, `acos`, `asinh`, `acosh`, `atanh` | `err(z)·abs(f′(z))`, with `abs(f′)` from the distances to the two branch points; within a few error radii of a square-root branch point `p` the Hölder bound `4·√(abs(z − p) + err(z))`, and no bound near a logarithmic one |
//! | `sign`, `floor`, `ceiling`, `heaviside`, `KroneckerDelta` | exact, or unknown when the argument (difference) is within its error of the threshold |
//! | special functions `f(x₁, …, xₙ)` | `2^(⌈log₂ k⌉ + 1)·maxᵢ abs(∂f/∂xᵢ)·err(xᵢ)` over the `k` distinct inexact arguments, `abs(∂f/∂xᵢ)` bounded over the argument's error ball (next table, `sensitivity.rs`); no bound when the ball reaches a pole or branch point (its radius above an eighth of the distance); exact arguments cost nothing |
//!
//! The sensitivity `abs(∂f/∂x)` of a special function — its condition number
//! `abs(∂ ln f/∂ ln x)` times `abs(f/x)`, which can be huge: `I_x(a, b)` near
//! `x = 1` for a small `b`, `Γ` near a pole, any function near a zero — comes
//! from a closed-form bound where one is cheap, from a bound on the change
//! over the ball where the derivative is unbounded but integrable, and
//! otherwise from the function evaluated again at the working precision with
//! the argument moved by its error (doubled):
//!
//! | function | argument | bound on `abs(∂f/∂x)` (`d`: distance to the nearest pole) |
//! |---|---|---|
//! | `Γ`, `x!`, `ln Γ`, `B(a, b)`, `C(n, k)` | each | `abs(f)·abs(ψ)` (`abs(ψ)` for `ln Γ`) with `abs(ψ(y)) ≤ 1/d + ln(1 + abs(y)) + γ`, and `abs(ψ(u) − ψ(v)) ≤ abs(u − v)·(1/m + 1/m²)` for `u, v > 0`, `m = min(u, v)` |
//! | `ψ⁽ⁿ⁾` | `x` | `(n+1)!·Σₖ abs(x + k)^(−n−2)`, at most `x^(−p) + x^(1−p)/(p−1)` (`p = n + 2`), or `d^(−p) + 2^(p+2)` left of 0; the order is discrete |
//! | `erf`, `erfc`; `erfi`; `erf⁻¹`, `erfc⁻¹` | `x` | `(2/√π)·e^(−x²)`; `(2/√π)·e^(x²)`; `(√π/2)·e^(f²)` |
//! | `Si`, `Ci`, `Shi`, `Chi`, `Ei`, `li`, Fresnel | `x` | `min(1, 1/abs(x))`, `1/abs(x)`, `max(1.18, e^abs(x)/(2·abs(x)))`, `e^abs(x)/abs(x)`, `eˣ/abs(x)`, `1/abs(ln x)`, 1 |
//! | `W` | `x` | `e^(−W)/abs(1 + W)`, with the first-order change below an eighth of `1 + W` |
//! | `ζ`, `η` | `s ≥ 0` | `1/(s − 1)² + 1`, and `η = (1 − 2^(1−s))·ζ` |
//! | `Li_s(z)` | `z` | `1/abs(1 − z)`, `abs(ln(1 − z)/z)`, 2 for an exact `s = 1`, 2, `≥ 3`; otherwise `m!/(1 − abs(z))^(m+1)`, `m = max(0, ⌈1 − s⌉)` |
//! |  | `s` | `abs(z)·m!/(1 − abs(z))^(m+1)`, `m = max(0, ⌈2 − s⌉)` |
//! | `Γ(s, x)`, `γ(s, x)` | `x` | `x^(s−1)·e^(−x)` |
//! |  | `s` | `abs(f)·max(abs(ln x), ln E[T given T > x])`, resp. `abs(f)·e·(abs(ln x) + 1/s)` (`T ~ Gamma(s)`) |
//! | `E_ν(x)` | `x`, `ν` | `abs(f)·E[T]`, `abs(f)·ln E[T]`, with `E[T] ≤ 1 + (max(0, −ν) + 1)/x` |
//! | `I_(x₁, x₂)(a, b)`, `B_(x₁, x₂)(a, b)` | limits | the integrand `t^(a−1)·(1−t)^(b−1)` (over `B(a, b)`); within 8 radii of 0 or 1, its integral over the ball |
//! |  | shapes | `abs(f)·abs(E[ln X given x₁ < X < x₂])` (minus `E[ln X]` regularised), and for a tail `abs(∂I_x/∂a) ≤ (1 − I_x)·(ψ(a+b) − ψ(a))` |
//! | `J_ν`, `I_ν`, `K_ν` (`ν ≥ 0`) | `x` | `(ν/x)·abs(f) + min(1, (x/2)^(ν+1)/Γ(ν+2))`, `(1 + ν/x)·abs(f)`, `(1 + (ν+1)/x)·abs(f)` |
//! |  | `ν` | from the series (`I`, `J` for `x ≤ 16`) and `K_ν = ∫ e^(−x cosh t)·cosh(νt) dt`: `(ν/x)·abs(f)` for `K` |
//! | `Ai`, `Bi`, `Ai′`, `Bi′`, `K(m)`, `E(m)`, `F(φ, m)`, `Π(n, m)` | each | envelopes (`(√x + 1)·abs(f)`, `0.6·(abs(x)^(1/4) + 1)`, …) and the derivative formulas with `E ≤ π/2`, `K ≤ π/(2√(1 − m))` |
//! | `Y_ν`; `ν` of `J_ν` beyond 16, orthogonal polynomials, `ζ` left of 0, the rest | — | `Y_(ν−1)` at 128 bits; numerically |
//!
//! Before 0.29 a special function carried its arguments' relative error
//! over, plus 4 bits, whatever its condition number, and a rational argument
//! rounded to the working precision printed wrong digits as certified:
//! `betainc_regularized(27/11, 1/26562500, 0, 1 − 3·10⁻³⁰)` was
//! `2.5118502017106955·10⁻⁶` in `eval_f64` (truly `…7194424·10⁻⁶`),
//! `besselj(0, x)` at a 36-digit rational `x` next to its first zero
//! `6.719·10⁻³⁸` (truly `6.450·10⁻³⁸`), and `betainc_regularized(1/3, 1/7, 0,
//! 1 − 10⁻⁶⁰)` exactly `1` (the limit rounded to 1, where the integrand is
//! infinite).
//!
//! `err(z)` of a complex argument is the joint bound `err(re) + err(im)`
//! ([`Bound::joint`]).  A function that is real on an exactly real argument
//! keeps an exact imaginary part; every other result carries the propagated
//! bound on both parts, and the rounding of the result (at its magnitude).
//!
//! Nodes whose evaluator sees more than its children's values report their
//! own bound ([`reported`] adds the rounding, as [`node_error`] does):
//!
//! | node | error of the result |
//! |---|---|
//! | finite `Sum` / `Product` | the terms' bounds (evaluated with the index bound exactly), as for `+` / `·`, plus the rounding of every partial result |
//! | infinite `Sum` of a hypergeometric term | the first term's relative error carried by the recurrence, the recurrence's roundings, and a rigorous geometric bound on the tail (`hypsum.rs`) |
//! | `RootOf` | the radius of a certified inclusion disk of the root (`poly::roots::root_balls`), on both parts of a non-real root and on the real part of a root certified real (whose imaginary part is exactly 0); unknown when the disks do not isolate the roots |
//! | `RootSum` | the body's bound at each root, the root bounded by its inclusion disk, as for `+` |
//! | `Piecewise` | the chosen branch's bound when every condition up to it is decided with certainty (a comparison whose difference is outside its error ball, or of exact values); unknown otherwise, like `sign` at its threshold |
//! | physical constants | their value's bound |
//! | definite integrals (`f64` quadrature, see `evalf.rs`) | the Gauss–Kronrod error estimate plus the rounding of the `f64` integrand values times the length of the interval |
//!
//! # Branch cuts
//!
//! `ln`, `arg`, `√` and non-integer powers are discontinuous across the
//! negative real axis, `asin`, `acos` and `atanh` across the real axis beyond
//! ±1, `acosh` across the real axis left of 1, and `atan` and `asinh` across
//! the imaginary axis beyond ±i ([`Cut`]).  Which side of its cut an argument
//! lies on is decided by the part perpendicular to the cut — the imaginary
//! part for the real-axis cuts, the real part for the imaginary-axis ones:
//!
//! * exactly 0: the argument is on the cut, and the value is the principal
//!   one, the limit from the side of counter-clockwise continuity (mpmath's
//!   and SymPy's convention): `ln(−2) = ln 2 + iπ`, `√−4 = 2i`,
//!   `arg(−1) = π`;
//! * outside its error ball: its sign picks the side, as the computed value
//!   has it;
//! * within its error ball but not exactly 0: the side is undecidable at this
//!   precision and the value has no bound ([`cut_undecided`]), so the
//!   expression is evaluated again at a higher precision and refused when
//!   the budget is spent — never guessed.
//!
//! Before 0.29 a value had a single bound, so "exactly real" and "real to
//! within its error" could not be told apart, the rounding residue of an
//! imaginary part that cancelled picked the side, and nothing noticed:
//! `arg(−1 − i·(exp(10⁻³⁰) − 1 − 10⁻³⁰))` came out `+π` (the imaginary part
//! is `−5·10⁻⁶¹`; the value is `−π`), `√(−4 − i·(…))` came out `2i` instead of
//! `−2i`, `ln(−1 − i·(sin²1 + cos²1 − 1))` came out `−iπ` for an argument
//! that is exactly `−1`.
//!
//! These are estimates, not proofs (interval arithmetic would need a rigorous
//! bound for every special function); they catch the loss that matters in
//! practice — catastrophic cancellation, the amplification by `exp`, `tan` or
//! a division, the side of a branch cut — and the evaluator re-evaluates at a
//! higher precision until the requested digits are covered
//! (`evaluate_adaptive` in `evalf.rs`).

use astro_float::{BigFloat, RoundingMode};
use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};
use rustc_hash::FxHashMap;

use crate::base::arena::Arena;
use crate::base::bigcomplex::{Complex, c_add, c_mul, c_one, c_sub, c_zero};
use crate::base::libfn::LibFn;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;

/// Base-2 exponent `e` of an absolute error bound: `|error| ≤ 2^e`.
pub(super) type ErrExp = i64;

/// The value is exact.
pub(super) const EXACT: ErrExp = i64::MIN / 4;

/// No bound is known: a division by a value indistinguishable from zero,
/// a decision (`sign`, `floor`, the side of a branch cut) at its threshold.
pub(super) const UNKNOWN: ErrExp = i64::MAX / 4;

/// The bound of a value that underflowed to 0: below the smallest positive
/// float.
const UNDERFLOW: ErrExp = astro_float::EXPONENT_MIN as ErrExp;

/// Is `e` the bound of an exact value (possibly after harmless arithmetic
/// on [`EXACT`])?
pub(super) fn is_exact(e: ErrExp) -> bool {
    e < EXACT / 2
}

/// Is `e` [`UNKNOWN`] (possibly after arithmetic)?
pub(super) fn is_unknown(e: ErrExp) -> bool {
    e > UNKNOWN / 2
}

fn clamp(e: ErrExp) -> ErrExp {
    e.clamp(EXACT, UNKNOWN)
}

/// `e + k`: exact stays exact, unknown stays unknown.
fn shift(e: ErrExp, k: i64) -> ErrExp {
    if is_exact(e) {
        EXACT
    } else if is_unknown(e) {
        UNKNOWN
    } else {
        clamp(e.saturating_add(k))
    }
}

/// The error bounds of the two parts of a complex value `re + i·im`:
/// `|Δre| ≤ 2^re`, `|Δim| ≤ 2^im` (see the module documentation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Bound {
    /// Exponent of the bound on the real part's error.
    pub(super) re: ErrExp,
    /// Exponent of the bound on the imaginary part's error.
    pub(super) im: ErrExp,
}

impl Bound {
    /// Both parts exact.
    pub(super) const EXACT: Bound = Bound {
        re: EXACT,
        im: EXACT,
    };

    /// No bound.
    pub(super) const UNKNOWN: Bound = Bound {
        re: UNKNOWN,
        im: UNKNOWN,
    };

    /// A real value: the imaginary part is exactly 0.
    pub(super) fn real(re: ErrExp) -> Bound {
        Bound { re, im: EXACT }
    }

    /// The same bound on both parts.
    pub(super) fn both(e: ErrExp) -> Bound {
        Bound { re: e, im: e }
    }

    /// The exponent of a bound on the error of the complex value,
    /// `|Δz| ≤ |Δre| + |Δim|`.
    pub(super) fn joint(self) -> ErrExp {
        if self.is_unknown() {
            return UNKNOWN;
        }
        match (is_exact(self.re), is_exact(self.im)) {
            (true, true) => EXACT,
            (true, false) => self.im,
            (false, true) => self.re,
            (false, false) => clamp(self.re.max(self.im).saturating_add(1)),
        }
    }

    /// Is either part unbounded?
    pub(super) fn is_unknown(self) -> bool {
        is_unknown(self.re) || is_unknown(self.im)
    }

    /// Are both parts exact?
    pub(super) fn is_exact(self) -> bool {
        is_exact(self.re) && is_exact(self.im)
    }
}

/// `m` with `2^(m−1) ≤ |x| < 2^m`; `None` for zero.
fn part_mag(x: &BigFloat) -> Option<i64> {
    if x.is_zero() {
        None
    } else {
        x.exponent().map(i64::from)
    }
}

fn is_finite(z: &Complex) -> bool {
    !(z.0.is_inf() || z.0.is_nan() || z.1.is_inf() || z.1.is_nan())
}

/// The magnitude exponent of a complex value (the larger of its parts'),
/// `None` for zero.
pub(super) fn mag(z: &Complex) -> Option<i64> {
    match (part_mag(&z.0), part_mag(&z.1)) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
    }
}

/// An exponent bounding the *true* value: its magnitude or its error.
fn upper(z: &Complex, err: ErrExp) -> ErrExp {
    mag(z).map_or(err, |m| m.max(err))
}

/// Does the ball `z ± 2^err` contain 0?
pub(super) fn contains_zero(z: &Complex, err: ErrExp) -> bool {
    match mag(z) {
        None => true,
        Some(m) => m <= err.saturating_add(1),
    }
}

/// Does the interval `x ± 2^e` contain 0?  (An exact part only when it is
/// 0.)
pub(super) fn part_contains_zero(x: &BigFloat, e: ErrExp) -> bool {
    match part_mag(x) {
        None => true,
        Some(m) => m <= e.saturating_add(1),
    }
}

/// Is the part `x ± 2^e` exactly 0?
pub(super) fn exact_zero(x: &BigFloat, e: ErrExp) -> bool {
    x.is_zero() && is_exact(e)
}

/// Is `z ± b` exactly real: its imaginary part exactly 0?
pub(super) fn exactly_real(z: &Complex, b: Bound) -> bool {
    exact_zero(&z.1, b.im)
}

/// The accurate bits of `z ± 2^err` relative to `|z|` (`None` for a value
/// that is zero or indistinguishable from zero).
pub(super) fn accurate_bits(z: &Complex, err: ErrExp) -> Option<i64> {
    let m = mag(z)?;
    if is_exact(err) {
        return Some(i64::MAX / 8);
    }
    (m > err).then_some(m - err)
}

/// The rounding of `z` to `prec` bits, at its magnitude.
pub(super) fn rounding(z: &Complex, prec: usize) -> ErrExp {
    mag(z).map_or(EXACT, |m| m - prec as i64)
}

/// The rounding of one part to `prec` bits, at its own magnitude.
fn part_rounding(x: &BigFloat, prec: usize) -> ErrExp {
    part_mag(x).map_or(EXACT, |m| m - prec as i64)
}

/// `⌈log₂ n⌉` (0 for `n ≤ 1`).
pub(super) fn ceil_log2(n: usize) -> i64 {
    i64::from(usize::BITS - n.saturating_sub(1).leading_zeros())
}

/// `⌈log₂|q|⌉`, for a non-zero rational (approximately).
fn log2_abs(q: &Q) -> i64 {
    i64::try_from(q.numer().bits()).unwrap_or(i64::MAX / 8)
        - i64::try_from(q.denom().bits()).unwrap_or(0)
        + 1
}

/// Is the rational exactly representable as a `prec`-bit binary float?
fn exactly_representable(q: &Q, prec: usize) -> bool {
    let d = q.denom();
    let power_of_two = (d & (d - BigInt::one())).is_zero();
    power_of_two && q.numer().bits() <= prec as u64
}

/// The error bound of `value`, computed at working precision `prec` by a
/// routine that bounds its own error by `err`: that bound plus the rounding
/// of each part, finished as [`node_error`] finishes a propagated bound.  A
/// part that is 0 with an exact bound stays exact.
pub(super) fn reported(value: &Complex, err: Bound, prec: usize) -> Bound {
    if !is_finite(value) || err.is_unknown() {
        return Bound::UNKNOWN;
    }
    let part = |x: &BigFloat, e: ErrExp| -> ErrExp {
        if exact_zero(x, e) {
            EXACT
        } else {
            clamp(e.max(part_rounding(x, prec)).saturating_add(1))
        }
    };
    Bound {
        re: part(&value.0, err.re),
        im: part(&value.1, err.im),
    }
}

/// The propagated bound of a product of factors `v ± 2^e` (joint bounds):
/// `Σ err(aᵢ)·Π_{j≠i} abs(aⱼ)`, times the number of factors (the roundings of
/// the partial products are the caller's).  An exact zero factor makes the
/// product exact.
pub(super) fn product_error<'a>(parts: impl IntoIterator<Item = (&'a Complex, ErrExp)>) -> ErrExp {
    let mut bounds: Vec<(ErrExp, ErrExp)> = Vec::new();
    for (v, e) in parts {
        if is_unknown(e) {
            return UNKNOWN;
        }
        if mag(v).is_none() && is_exact(e) {
            return EXACT; // an exact zero factor
        }
        bounds.push((e, upper(v, e)));
    }
    let total: ErrExp = bounds.iter().map(|&(_, u)| u).sum();
    let worst = bounds
        .iter()
        .map(|&(e, u)| if is_exact(e) { EXACT } else { e + total - u })
        .max()
        .unwrap_or(EXACT);
    worst.saturating_add(ceil_log2(bounds.len()))
}

// ── Sums and products: value and bound together ────────────────────────────

/// One part of a running sum: the worst term bound, the largest partial
/// sum, and whether any term's part was not an exact 0.
#[derive(Clone, Copy, Debug)]
pub(super) struct PartSum {
    worst: ErrExp,
    peak: Option<i64>,
    nonzero: bool,
}

impl PartSum {
    pub(super) const fn new() -> PartSum {
        PartSum {
            worst: EXACT,
            peak: None,
            nonzero: false,
        }
    }

    /// Account for a term's part `term ± 2^err`, after which the partial
    /// sum's part is `partial`.
    pub(super) fn add(&mut self, term: &BigFloat, err: ErrExp, partial: &BigFloat) {
        self.worst = self.worst.max(err);
        self.nonzero |= !exact_zero(term, err);
        self.peak = self.peak.max(part_mag(partial));
    }

    /// The bound of this part of a sum of `count` terms: the terms' bounds
    /// plus the rounding of the partial sums (at the largest of them), and
    /// exact when every term's part was an exact 0.
    pub(super) fn bound(&self, count: usize, prec: usize) -> ErrExp {
        if !self.nonzero {
            return EXACT;
        }
        if is_unknown(self.worst) {
            return UNKNOWN;
        }
        let n = ceil_log2(count);
        let rounding = self.peak.map_or(EXACT, |m| m - prec as i64 + n);
        shift(self.worst, n).max(rounding)
    }
}

/// `x` marked exact (its bound says it is; astro-float's flag is sticky and
/// also set by `0·y` for an inexact `y`).
fn clean(x: &BigFloat) -> BigFloat {
    let mut x = x.clone();
    x.set_inexact(false);
    x
}

/// Is the sum of the exact values `parts` at `prec` bits free of rounding?
fn exact_sum<'a>(parts: impl Iterator<Item = &'a BigFloat>, prec: usize, rm: RoundingMode) -> bool {
    let mut s = BigFloat::new(prec);
    for x in parts {
        if x.is_zero() {
            continue;
        }
        s = s.add(&clean(x), prec, rm);
        if s.inexact() {
            return false;
        }
    }
    true
}

/// `a₁ + … + aₙ` summed left to right from 0 at `prec` bits, as `evalf`
/// sums, with the bound of each part ([`PartSum`]); a part whose terms are
/// all exact and whose partial sums were never rounded is exact.
pub(super) fn add_with_bound(
    terms: &[(&Complex, Bound)],
    prec: usize,
    rm: RoundingMode,
) -> (Complex, Bound) {
    let mut sum = c_zero(prec);
    let (mut re, mut im) = (PartSum::new(), PartSum::new());
    let mut unknown = false;
    for &(v, b) in terms {
        unknown |= b.is_unknown();
        sum = c_add(&sum, v, prec, rm);
        re.add(&v.0, b.re, &sum.0);
        im.add(&v.1, b.im, &sum.1);
    }
    if unknown {
        return (sum, Bound::UNKNOWN);
    }
    let finish = |acc: &PartSum, exact_terms: bool, parts: &mut dyn Iterator<Item = &BigFloat>| {
        let e = acc.bound(terms.len(), prec);
        if is_exact(e) || (exact_terms && exact_sum(parts, prec, rm)) {
            EXACT
        } else {
            clamp(e.saturating_add(1))
        }
    };
    let re_exact = terms.iter().all(|&(_, b)| is_exact(b.re));
    let im_exact = terms.iter().all(|&(_, b)| is_exact(b.im));
    let bound = Bound {
        re: finish(&re, re_exact, &mut terms.iter().map(|(v, _)| &v.0)),
        im: finish(&im, im_exact, &mut terms.iter().map(|(v, _)| &v.1)),
    };
    (sum, bound)
}

/// `log₂` of an error bound (`−∞` for an exact one).
fn lg(e: ErrExp) -> f64 {
    if is_exact(e) {
        f64::NEG_INFINITY
    } else {
        e as f64
    }
}

/// The error exponent of a `log₂` bound.
fn from_lg(x: f64) -> ErrExp {
    if x == f64::NEG_INFINITY {
        EXACT
    } else if x.is_nan() || x > 1e15 {
        UNKNOWN
    } else {
        clamp(x.max(-1e15).ceil() as i64)
    }
}

/// `log₂(2^a + 2^b)`.
fn lsum(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    hi + (1.0 + (lo - hi).exp2()).log2()
}

/// A factor's part in a product step: its value and the `log₂` of its
/// error bound.
type Factor<'a> = (&'a BigFloat, f64);

/// One part of a product step, `x₁·y₁ ∓ x₂·y₂` (computed as `result` at
/// `prec` bits; `subtract` for the real part): the `log₂` of its error
/// bound.  A product with an exact-zero factor is exactly 0 and contributes
/// nothing; the others contribute `abs(x)·err(y) + abs(y)·err(x) +
/// err(x)·err(y)` and three roundings (two products and their sum) at the
/// largest magnitude involved — none when every factor is exact and the
/// exact recomputation rounds nothing.
fn product_part(
    terms: [(Factor<'_>, Factor<'_>); 2],
    result: &BigFloat,
    subtract: bool,
    prec: usize,
    rm: RoundingMode,
) -> f64 {
    let p = prec as f64;
    let lmag = |x: &BigFloat| part_mag(x).map_or(f64::NEG_INFINITY, |m| m as f64);
    let mut err = f64::NEG_INFINITY;
    let mut round = f64::NEG_INFINITY;
    let mut exact = true;
    let mut live: Vec<(&BigFloat, &BigFloat)> = Vec::with_capacity(2);
    for ((x, ex), (y, ey)) in terms {
        if (x.is_zero() && ex == f64::NEG_INFINITY) || (y.is_zero() && ey == f64::NEG_INFINITY) {
            continue;
        }
        let (mx, my) = (lmag(x), lmag(y));
        err = lsum(err, lsum(lsum(mx + ey, my + ex), ex + ey));
        round = round.max(mx + my - p);
        exact &= ex == f64::NEG_INFINITY && ey == f64::NEG_INFINITY;
        live.push((x, y));
    }
    if live.is_empty() {
        return f64::NEG_INFINITY;
    }
    if exact {
        let mut acc: Option<BigFloat> = None;
        let mut rounded = false;
        for &(x, y) in &live {
            let t = clean(x).mul(&clean(y), prec, rm);
            rounded |= t.inexact();
            let next = match acc {
                None => t,
                Some(a) if subtract => a.sub(&t, prec, rm),
                Some(a) => a.add(&t, prec, rm),
            };
            rounded |= next.inexact();
            acc = Some(next);
        }
        if !rounded {
            return f64::NEG_INFINITY;
        }
    }
    round = round.max(lmag(result) - p);
    lsum(err, round + 3f64.log2())
}

/// `a₁ · … · aₙ` multiplied left to right from 1 at `prec` bits, as `evalf`
/// multiplies, with the bound of each part ([`product_part`] for every
/// step).  An exact-zero factor makes the product exact; a product of
/// non-zero factors that came out 0 underflowed.
pub(super) fn mul_with_bound(
    factors: &[(&Complex, Bound)],
    prec: usize,
    rm: RoundingMode,
) -> (Complex, Bound) {
    let mut product = c_one(prec);
    let mut err = [f64::NEG_INFINITY; 2];
    let mut unknown = false;
    let mut zero_factor = false;
    let mut all_real = true;
    let mut all_nonzero = true;
    for &(v, b) in factors {
        unknown |= b.is_unknown();
        zero_factor |= exact_zero(&v.0, b.re) && exact_zero(&v.1, b.im);
        all_real &= exactly_real(v, b);
        all_nonzero &= mag(v).is_some();
        let next = c_mul(&product, v, prec, rm);
        if !unknown && !zero_factor {
            let (pr, pi) = ((&product.0, err[0]), (&product.1, err[1]));
            let (fr, fi) = ((&v.0, lg(b.re)), (&v.1, lg(b.im)));
            err = [
                product_part([(pr, fr), (pi, fi)], &next.0, true, prec, rm),
                product_part([(pr, fi), (pi, fr)], &next.1, false, prec, rm),
            ];
        }
        product = next;
    }
    if unknown {
        return (product, Bound::UNKNOWN);
    }
    if zero_factor {
        return (product, Bound::EXACT);
    }
    if mag(&product).is_none() && all_nonzero && !factors.is_empty() {
        let im = if all_real { EXACT } else { UNDERFLOW };
        return (product, Bound { re: UNDERFLOW, im });
    }
    let bound = Bound {
        re: from_lg(err[0]),
        im: from_lg(err[1]),
    };
    (product, bound)
}

// ── Branch cuts ────────────────────────────────────────────────────────────

/// A branch cut of an elementary function (see the module documentation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Cut {
    /// `(−∞, 0]`: `ln`, `arg`, `√`, non-integer powers, `atan2(y, x)` at
    /// `y = 0`.
    NegativeReal,
    /// `(−∞, −1] ∪ [1, ∞)`: `asin`, `acos`, `atanh`.
    RealBeyondOne,
    /// `(−∞, 1]`: `acosh`.
    RealBelowOne,
    /// `i·((−∞, −1] ∪ [1, ∞))`: `atan`, `asinh`.
    ImaginaryBeyondOne,
}

/// The sign of `x` (0 for either zero).
fn sign_of(x: &BigFloat) -> i32 {
    if x.is_zero() {
        0
    } else if x.is_negative() {
        -1
    } else {
        1
    }
}

/// `x − c`, keeping every bit of `x`.
fn minus(x: &BigFloat, c: i32) -> BigFloat {
    let p = x.mantissa_max_bit_len().unwrap_or(64).max(64) + 64;
    x.sub(&BigFloat::from_i32(c, 64), p, RoundingMode::ToEven)
}

/// Does the interval `x ± 2^e` meet `(−∞, c]`?
fn meets_at_most(x: &BigFloat, e: ErrExp, c: i32) -> bool {
    let d = minus(x, c);
    sign_of(&d) <= 0 || part_contains_zero(&d, e)
}

/// Does the interval `x ± 2^e` meet `[c, ∞)`?
fn meets_at_least(x: &BigFloat, e: ErrExp, c: i32) -> bool {
    let d = minus(x, c);
    sign_of(&d) >= 0 || part_contains_zero(&d, e)
}

/// Does the interval `x ± 2^e` lie in `[lo, hi]`?
fn within(x: &BigFloat, e: ErrExp, lo: i32, hi: i32) -> bool {
    !meets_at_most_strict(x, e, lo) && !meets_at_least_strict(x, e, hi)
}

/// Does the interval `x ± 2^e` lie in `[c, ∞)`?
fn at_least(x: &BigFloat, e: ErrExp, c: i32) -> bool {
    !meets_at_most_strict(x, e, c)
}

/// Does the interval `x ± 2^e` meet `(−∞, c)`?
fn meets_at_most_strict(x: &BigFloat, e: ErrExp, c: i32) -> bool {
    let d = minus(x, c);
    sign_of(&d) < 0 || (!is_exact(e) && part_contains_zero(&d, e))
}

/// Does the interval `x ± 2^e` meet `(c, ∞)`?
fn meets_at_least_strict(x: &BigFloat, e: ErrExp, c: i32) -> bool {
    let d = minus(x, c);
    sign_of(&d) > 0 || (!is_exact(e) && part_contains_zero(&d, e))
}

/// Is the side of `cut` on which the argument `z ± b` lies undecidable?
///
/// The box `re ± 2^b.re`, `im ± 2^b.im` must meet the cut, and the part
/// perpendicular to it (`im` for the real-axis cuts, `re` for the
/// imaginary-axis ones) must be within its error of 0 without being exactly
/// 0: an exact 0 is on the cut, where the principal value is the
/// convention, and a part outside its error ball has a certain sign.
pub(super) fn cut_undecided(z: &Complex, b: Bound, cut: Cut) -> bool {
    let (across, across_err, along, along_err) = match cut {
        Cut::ImaginaryBeyondOne => (&z.0, b.re, &z.1, b.im),
        _ => (&z.1, b.im, &z.0, b.re),
    };
    if exact_zero(across, across_err) || !part_contains_zero(across, across_err) {
        return false;
    }
    match cut {
        Cut::NegativeReal => meets_at_most(along, along_err, 0),
        Cut::RealBelowOne => meets_at_most(along, along_err, 1),
        Cut::RealBeyondOne | Cut::ImaginaryBeyondOne => {
            meets_at_most(along, along_err, -1) || meets_at_least(along, along_err, 1)
        }
    }
}

/// How a function behaves at one of its branch points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BranchPoint {
    /// Like `√(z − p)`, and continuous there (the jump across the cut
    /// vanishes at `p`).
    Root,
    /// Like `√(z − p)` on either side of a cut that jumps at `p` (`acosh`
    /// at −1): continuous along the real axis.
    RootOnCut,
    /// Like `ln(z − p)`: unbounded.
    Log,
}

/// The Hölder bound near a square-root branch point `p`: every point `w`
/// of the ball (radius `2^j`, centre within `2^(j+3)` of `p`) has
/// `|w − p| < 2^(j+4)`, and `|f(w) − f(p)| ≤ 1.03·√(2·|w − p|)` for the
/// inverse trigonometric and hyperbolic functions while `|w − p| ≤ 1/4`, so
/// two values differ by at most `2^(2 + ⌈(j + 4)/2⌉)`.
fn holder(j: ErrExp) -> ErrExp {
    if j > -8 {
        UNKNOWN
    } else {
        (j + 5).div_euclid(2) + 2
    }
}

/// The propagated bound of an inverse trigonometric or hyperbolic function
/// (node `node`) of `z ± b`, at working precision `prec`: see the module
/// documentation.  `|f′(z)| = |(z − p₁)(z − p₂)|^(−k)` with the branch
/// points `p₁, p₂` (`k = ½` for `asin`, `acos`, `asinh`, `acosh`, 1 for
/// `atan`, `atanh`), computed from the distances at high precision — the
/// 64-bit `1 − z²` before 0.29 cancelled to 0 for `z` within `2⁻⁶⁴` of ±1
/// and `acos(1 + 5·10⁻⁶¹)` was refused.
fn inverse_error(node: &ExprNode, z: &Complex, b: Bound, prec: usize) -> Bound {
    use BranchPoint::{Log, Root, RootOnCut};
    // The two branch points `re + i·im` and how the function behaves there.
    type Points = [((i32, i32), BranchPoint); 2];
    let (points, cut, halve): (Points, Cut, bool) = match node {
        ExprNode::Asin(_) | ExprNode::Acos(_) => {
            ([((1, 0), Root), ((-1, 0), Root)], Cut::RealBeyondOne, true)
        }
        ExprNode::Atanh(_) => ([((1, 0), Log), ((-1, 0), Log)], Cut::RealBeyondOne, false),
        ExprNode::Acosh(_) => (
            [((1, 0), Root), ((-1, 0), RootOnCut)],
            Cut::RealBelowOne,
            true,
        ),
        ExprNode::Atan(_) => (
            [((0, 1), Log), ((0, -1), Log)],
            Cut::ImaginaryBeyondOne,
            false,
        ),
        _ => (
            [((0, 1), Root), ((0, -1), Root)],
            Cut::ImaginaryBeyondOne,
            true,
        ),
    };
    let real = exactly_real(z, b)
        && match node {
            ExprNode::Asin(_) | ExprNode::Acos(_) | ExprNode::Atanh(_) => within(&z.0, b.re, -1, 1),
            ExprNode::Acosh(_) => at_least(&z.0, b.re, 1),
            _ => true,
        };
    let result = |e: ErrExp| {
        if is_unknown(e) {
            Bound::UNKNOWN
        } else if real {
            Bound::real(e)
        } else {
            Bound::both(e)
        }
    };
    if b.is_exact() {
        return result(EXACT);
    }
    let j = b.joint();
    let hp = z.0.mantissa_max_bit_len().unwrap_or(prec).max(prec) + 64;
    let mut distance = [0i64; 2];
    for (slot, &((pr, pi), kind)) in distance.iter_mut().zip(points.iter()) {
        let p = (BigFloat::from_i32(pr, 64), BigFloat::from_i32(pi, 64));
        match mag(&c_sub(z, &p, hp, RoundingMode::ToEven)) {
            Some(m) if m > j.saturating_add(3) => *slot = m,
            _ => {
                return match kind {
                    Log => Bound::UNKNOWN,
                    Root => result(holder(j)),
                    RootOnCut if !exactly_real(z, b) && cut_undecided(z, b, cut) => Bound::UNKNOWN,
                    RootOnCut => result(holder(j)),
                };
            }
        }
    }
    if cut_undecided(z, b, cut) {
        return Bound::UNKNOWN;
    }
    // |z − pᵢ| ≥ 2^(mᵢ − 1), and over the ball (radius at most 1/8 of each
    // distance) |f′| grows by less than 2.
    let log_w = distance[0] + distance[1] - 2;
    let amplification = if halve {
        (1 - log_w).div_euclid(2)
    } else {
        -log_w
    };
    result(shift(j, amplification.max(0) + 2))
}

/// The branch cut of a special function on the real axis, in one of its
/// arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RealCut {
    /// `(−∞, 0]`.
    BelowZero,
    /// `(−∞, 1]`.
    BelowOne,
    /// `[1, ∞)`.
    AboveOne,
    /// `(−∞, −1] ∪ [1, ∞)`.
    OutsideUnit,
    /// `(−∞, 0] ∪ [2, ∞)`.
    OutsideZeroTwo,
    /// `(−∞, −1/e]` (the principal branch of Lambert's W).
    BelowMinusInvE,
}

/// Where a special function is not analytic, as far as the error bound of
/// a real routine applied to an argument whose imaginary part is only
/// within its error of 0 is concerned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Analytic {
    /// Entire or meromorphic in every argument (for the arguments `evalf`
    /// accepts): the propagated bound holds on both parts.
    Everywhere,
    /// Off the cut in argument number `.0`: the propagated bound holds when
    /// that argument's interval misses the cut.
    OffCut(usize, RealCut),
    /// No rule: an argument that may be off the real axis has no bound.
    Unknown,
}

fn analyticity(arena: &Arena, node: &ExprNode) -> Analytic {
    use Analytic::{Everywhere, OffCut, Unknown};
    use RealCut::{AboveOne, BelowMinusInvE, BelowOne, BelowZero, OutsideUnit, OutsideZeroTwo};
    match node {
        ExprNode::Gamma(_)
        | ExprNode::Digamma(_)
        | ExprNode::Polygamma(_, _)
        | ExprNode::Zeta(_)
        | ExprNode::Erf(_)
        | ExprNode::Erfc(_)
        | ExprNode::Beta(_, _)
        | ExprNode::Factorial(_)
        | ExprNode::Binomial(_, _)
        | ExprNode::Si(_)
        | ExprNode::DiracDelta(_) => Everywhere,
        ExprNode::LogGamma(_) | ExprNode::Ci(_) | ExprNode::Ei(_) => OffCut(0, BelowZero),
        ExprNode::Li(_) => OffCut(0, BelowOne),
        ExprNode::LambertW(_) => OffCut(0, BelowMinusInvE),
        ExprNode::Apply(sid, _) => match arena.lib_fn(*sid) {
            Some(
                LibFn::Erfi
                | LibFn::Shi
                | LibFn::FresnelS
                | LibFn::FresnelC
                | LibFn::AiryAi
                | LibFn::AiryBi
                | LibFn::AiryAiPrime
                | LibFn::AiryBiPrime
                | LibFn::DirichletEta
                | LibFn::Legendre
                | LibFn::ChebyshevT
                | LibFn::ChebyshevU
                | LibFn::Hermite
                | LibFn::Laguerre
                | LibFn::Gegenbauer
                | LibFn::Jacobi
                | LibFn::AssocLaguerre,
            ) => Everywhere,
            Some(LibFn::Chi) => OffCut(0, BelowZero),
            Some(
                LibFn::BesselJ
                | LibFn::BesselY
                | LibFn::BesselI
                | LibFn::BesselK
                | LibFn::ExpInt
                | LibFn::UpperGamma
                | LibFn::LowerGamma,
            ) => OffCut(1, BelowZero),
            Some(LibFn::PolyLog) => OffCut(1, AboveOne),
            Some(LibFn::EllipticK | LibFn::EllipticE) => OffCut(0, AboveOne),
            Some(LibFn::ErfInv) => OffCut(0, OutsideUnit),
            Some(LibFn::ErfcInv) => OffCut(0, OutsideZeroTwo),
            Some(LibFn::AssocLegendre) => OffCut(2, OutsideUnit),
            _ => Unknown,
        },
        _ => Unknown,
    }
}

/// Does the interval `x ± 2^e` meet the cut?
fn meets_cut(x: &BigFloat, e: ErrExp, cut: RealCut) -> bool {
    match cut {
        RealCut::BelowZero => meets_at_most(x, e, 0),
        RealCut::BelowOne => meets_at_most(x, e, 1),
        RealCut::AboveOne => meets_at_least(x, e, 1),
        RealCut::OutsideUnit => meets_at_most(x, e, -1) || meets_at_least(x, e, 1),
        RealCut::OutsideZeroTwo => meets_at_most(x, e, 0) || meets_at_least(x, e, 2),
        RealCut::BelowMinusInvE => {
            // −1/e < −0.3678: (−∞, −0.3678] covers the cut.
            let p = x.mantissa_max_bit_len().unwrap_or(64).max(64) + 64;
            let c = BigFloat::from_i32(3678, 64).div(
                &BigFloat::from_i32(10_000, 64),
                64,
                RoundingMode::Up,
            );
            let d = x.add(&c, p, RoundingMode::ToEven);
            sign_of(&d) <= 0 || part_contains_zero(&d, e.saturating_add(1))
        }
    }
}

/// Is the exact real `x` an integer?
fn is_integer_value(x: &BigFloat) -> bool {
    x.is_zero() || x.int() == *x
}

// ── Node bounds ─────────────────────────────────────────────────────────────

/// The error bound of the value `value` of node `id`, from the values
/// (`cache`) and error bounds (`errs`) of its children, at working
/// precision `prec`.  See the module documentation.  `special` computes the
/// propagated error of a special function, `Σ |∂f/∂xᵢ|·err(xᵢ)`
/// (`sensitivity::special_error`); it is called only for such a node whose
/// children all have a bound.
pub(super) fn node_error(
    arena: &Arena,
    id: ExprId,
    value: &Complex,
    cache: &FxHashMap<ExprId, Complex>,
    errs: &FxHashMap<ExprId, Bound>,
    prec: usize,
    special: impl FnOnce() -> ErrExp,
) -> Bound {
    if !is_finite(value) {
        return Bound::UNKNOWN;
    }
    let child = |c: ExprId| -> Option<(&Complex, Bound)> {
        Some((
            cache.get(&c)?,
            errs.get(&c).copied().unwrap_or(Bound::UNKNOWN),
        ))
    };
    // The child's value and bound, or an early `UNKNOWN`.
    macro_rules! known_child {
        ($c:expr) => {
            match child($c) {
                Some((z, b)) if !b.is_unknown() => (z, b),
                _ => return Bound::UNKNOWN,
            }
        };
    }
    let ub_out = mag(value).unwrap_or(EXACT);
    let node = arena.node(id);
    let propagated: Bound = match node {
        ExprNode::Num(nid) => {
            if exactly_representable(arena.num(*nid), prec) {
                return Bound::EXACT;
            }
            Bound::real(rounding(value, prec))
        }
        ExprNode::ImaginaryUnit | ExprNode::BoolTrue | ExprNode::BoolFalse => return Bound::EXACT,
        ExprNode::Pi
        | ExprNode::E
        | ExprNode::EulerGamma
        | ExprNode::Catalan
        | ExprNode::GoldenRatio
        | ExprNode::PhysicalConstant(..) => Bound::real(rounding(value, prec)),

        // Evaluated with their bound by `add_with_bound` / `mul_with_bound`
        // (`eval_node_with_error`); recomputed here for completeness.
        ExprNode::Add(children) | ExprNode::Mul(children) => {
            let mut parts: Vec<(&Complex, Bound)> = Vec::with_capacity(children.len());
            for &c in children.iter() {
                match child(c) {
                    Some(part) => parts.push(part),
                    None => return Bound::UNKNOWN,
                }
            }
            let rm = RoundingMode::ToEven;
            return if matches!(node, ExprNode::Add(_)) {
                add_with_bound(&parts, prec, rm).1
            } else {
                mul_with_bound(&parts, prec, rm).1
            };
        }
        ExprNode::Neg(c) | ExprNode::Conjugate(c) => match child(*c) {
            Some((_, b)) => b,
            None => return Bound::UNKNOWN,
        },
        ExprNode::Re(c) => match child(*c) {
            Some((_, b)) => Bound::real(b.re),
            None => return Bound::UNKNOWN,
        },
        ExprNode::Im(c) => match child(*c) {
            Some((_, b)) => Bound::real(b.im),
            None => return Bound::UNKNOWN,
        },
        ExprNode::Pow(base, exp) => {
            let (b, eb) = known_child!(*base);
            let (x, ex) = known_child!(*exp);
            pow_bound(arena.as_num(*exp), b, eb, x, ex, ub_out)
        }

        // `d exp z = exp z · dz`: the argument's absolute error is the
        // result's relative error.  Before 0.29 the magnitude was clamped
        // at 1 (`ub_out.max(0)`), so `exp(−260.1) ≈ 2⁻³⁷⁵` carried the
        // absolute error of its argument, `2^(9 − prec)`: at the 384-bit cap
        // that ball contained 0, `ln(exp(−260.1))` had no bound and was
        // refused.
        ExprNode::Exp(c) => {
            let (z, b) = known_child!(*c);
            let e = shift(b.joint(), ub_out);
            if exactly_real(z, b) {
                Bound::real(e)
            } else {
                Bound::both(e)
            }
        }
        ExprNode::Ln(c) | ExprNode::Arg(c) => {
            let (z, b) = known_child!(*c);
            let is_arg = matches!(node, ExprNode::Arg(_));
            let real = is_arg || (exactly_real(z, b) && sign_of(&z.0) > 0);
            if b.is_exact() {
                if real {
                    Bound::real(EXACT)
                } else {
                    Bound::EXACT
                }
            } else {
                let j = b.joint();
                if contains_zero(z, j) || cut_undecided(z, b, Cut::NegativeReal) {
                    return Bound::UNKNOWN;
                }
                let e = j - mag(z).unwrap_or(0) + 1;
                if real { Bound::real(e) } else { Bound::both(e) }
            }
        }
        ExprNode::Sin(c) | ExprNode::Cos(c) | ExprNode::Sinh(c) | ExprNode::Cosh(c) => {
            let (z, b) = known_child!(*c);
            let e = shift(b.joint(), ub_out.max(0) + 1);
            if exactly_real(z, b) {
                Bound::real(e)
            } else {
                Bound::both(e)
            }
        }
        ExprNode::Tan(c) | ExprNode::Tanh(c) => {
            let (z, b) = known_child!(*c);
            let e = shift(b.joint(), (2 * ub_out).max(0) + 1);
            if exactly_real(z, b) {
                Bound::real(e)
            } else {
                Bound::both(e)
            }
        }
        ExprNode::Atan(c)
        | ExprNode::Asin(c)
        | ExprNode::Acos(c)
        | ExprNode::Asinh(c)
        | ExprNode::Acosh(c)
        | ExprNode::Atanh(c) => {
            let (z, b) = known_child!(*c);
            inverse_error(node, z, b, prec)
        }
        ExprNode::Abs(c) => {
            let (z, b) = known_child!(*c);
            Bound::real(if exactly_real(z, b) { b.re } else { b.joint() })
        }
        // `atan2(y, x) = arg(x + iy)` for real `x`, `y`: the cut is `y = 0`,
        // `x < 0`.  Arguments whose imaginary parts are only within their
        // error of 0 may be complex, where `atan2` is not the real function.
        ExprNode::Atan2(y, x) => {
            let (yv, ey) = known_child!(*y);
            let (xv, ex) = known_child!(*x);
            if !exactly_real(yv, ey) || !exactly_real(xv, ex) {
                return Bound::UNKNOWN;
            }
            let z = (xv.0.clone(), yv.0.clone());
            let b = Bound {
                re: ex.re,
                im: ey.re,
            };
            if b.is_exact() {
                Bound::real(EXACT)
            } else {
                let j = b.joint();
                if contains_zero(&z, j) || cut_undecided(&z, b, Cut::NegativeReal) {
                    return Bound::UNKNOWN;
                }
                Bound::real(j - mag(&z).unwrap_or(0) + 1)
            }
        }

        // Decisions: exact, unless the argument is within its error of the
        // threshold.
        ExprNode::Sign(c) | ExprNode::Heaviside(c) => {
            let (z, b) = known_child!(*c);
            if exactly_real(z, b) {
                if is_exact(b.re) || !part_contains_zero(&z.0, b.re) {
                    return Bound::EXACT; // −1, 0 or 1
                }
                return Bound::UNKNOWN;
            }
            // An argument that may be complex: `heaviside` is undefined
            // there, `sign(z) = z/|z|`.
            let j = b.joint();
            if matches!(node, ExprNode::Heaviside(_)) || contains_zero(z, j) {
                return Bound::UNKNOWN;
            }
            Bound::both(j - mag(z).unwrap_or(0) + 1)
        }
        ExprNode::Floor(c) | ExprNode::Ceiling(c) => {
            let (z, b) = known_child!(*c);
            if !exactly_real(z, b) {
                return Bound::UNKNOWN; // `floor` of a value that may be complex
            }
            if is_exact(b.re) {
                return Bound::EXACT;
            }
            // Distance from the argument to the integer it rounded to, and
            // to the next one.
            let p = 64 + prec;
            let rm = RoundingMode::ToEven;
            let d1 = part_mag(&z.0.sub(&value.0, p, rm));
            let one = BigFloat::from_i32(1, p);
            let next = match node {
                ExprNode::Floor(_) => value.0.add(&one, p, rm),
                _ => value.0.sub(&one, p, rm),
            };
            let d2 = part_mag(&next.sub(&z.0, p, rm));
            let margin = d1.unwrap_or(i64::MIN).min(d2.unwrap_or(i64::MIN));
            if d1.is_none() || d2.is_none() || margin <= b.re + 1 {
                return Bound::UNKNOWN;
            }
            return Bound::EXACT; // an integer
        }
        ExprNode::Min(children) | ExprNode::Max(children) => {
            let mut worst = EXACT;
            for &c in children.iter() {
                let (z, b) = known_child!(c);
                if !exactly_real(z, b) {
                    return Bound::UNKNOWN;
                }
                worst = worst.max(b.re);
            }
            Bound::real(worst)
        }
        // A decision on `i − j = 0`: exact when both are exact or their
        // difference is outside its error box.
        ExprNode::KroneckerDelta(i, j) => {
            let (iv, ei) = known_child!(*i);
            let (jv, ej) = known_child!(*j);
            if ei.is_exact() && ej.is_exact() {
                return Bound::EXACT;
            }
            let d = c_sub(iv, jv, prec + 64, RoundingMode::ToEven);
            let re = shift(ei.re.max(ej.re), 1);
            let im = shift(ei.im.max(ej.im), 1);
            if part_contains_zero(&d.0, re) && part_contains_zero(&d.1, im) {
                return Bound::UNKNOWN;
            }
            return Bound::EXACT;
        }

        // Special functions: each argument's error times the function's
        // sensitivity to it (`sensitivity.rs`).  Before 0.29 the arguments'
        // relative error carried over plus 4 bits, whatever the condition
        // number: `betainc_regularized(27/11, 1/26562500, 0, 1 − 3·10⁻³⁰)`
        // lost 11 of the 16 digits it certified to the rounding of `x`.
        ExprNode::Gamma(_)
        | ExprNode::LogGamma(_)
        | ExprNode::Digamma(_)
        | ExprNode::Erf(_)
        | ExprNode::Erfc(_)
        | ExprNode::LambertW(_)
        | ExprNode::Beta(_, _)
        | ExprNode::Si(_)
        | ExprNode::Ci(_)
        | ExprNode::Ei(_)
        | ExprNode::Li(_)
        | ExprNode::Zeta(_)
        | ExprNode::Polygamma(_, _)
        | ExprNode::Factorial(_)
        | ExprNode::Binomial(_, _)
        | ExprNode::Apply(_, _)
        | ExprNode::DiracDelta(_) => {
            let mut all_real = true;
            for c in node.children() {
                let (v, b) = known_child!(c);
                all_real &= exactly_real(v, b);
            }
            let worst = special();
            if is_unknown(worst) {
                return Bound::UNKNOWN;
            }
            if all_real && value.1.is_zero() {
                Bound::real(worst)
            } else if all_real {
                Bound::both(worst)
            } else {
                // A real routine applied to arguments that may be off the
                // real axis: the propagated bound holds where the function
                // is analytic, not across its cut, where the side is
                // undecidable.
                match analyticity(arena, node) {
                    Analytic::Everywhere => Bound::both(worst),
                    Analytic::OffCut(i, cut) => {
                        let kids = node.children();
                        let Some((v, b)) = kids.get(i).and_then(|&c| child(c)) else {
                            return Bound::UNKNOWN;
                        };
                        if !exactly_real(v, b) && meets_cut(&v.0, b.re, cut) {
                            return Bound::UNKNOWN;
                        }
                        Bound::both(worst)
                    }
                    Analytic::Unknown => return Bound::UNKNOWN,
                }
            }
        }

        // Evaluated with their own bounds by `eval_node_with_error`.
        _ => return Bound::UNKNOWN,
    };
    finish(arena, id, value, cache, errs, propagated, prec)
}

/// The final bound of node `id`: the propagated bound plus the rounding of
/// the result (at its magnitude) on each part, except a part that is 0 with
/// an exact propagated bound, which is exact — or, for a whole value of 0
/// computed exactly from non-zero children, an underflow.
fn finish(
    arena: &Arena,
    id: ExprId,
    value: &Complex,
    cache: &FxHashMap<ExprId, Complex>,
    errs: &FxHashMap<ExprId, Bound>,
    propagated: Bound,
    prec: usize,
) -> Bound {
    if propagated.is_unknown() {
        return Bound::UNKNOWN;
    }
    if propagated.is_exact() && mag(value).is_none() {
        if !underflowed(arena, id, cache) {
            return Bound::EXACT;
        }
        let real = |c: &ExprId| {
            cache
                .get(c)
                .is_some_and(|v| exactly_real(v, errs.get(c).copied().unwrap_or(Bound::UNKNOWN)))
        };
        let positive = |c: &ExprId| cache.get(c).is_some_and(|v| sign_of(&v.0) > 0);
        let im = match arena.node(id) {
            ExprNode::Exp(c) if real(c) => EXACT,
            ExprNode::Pow(b, e) if real(b) && real(e) && positive(b) => EXACT,
            _ => UNDERFLOW,
        };
        return Bound { re: UNDERFLOW, im };
    }
    let round = rounding(value, prec);
    let part = |e: ErrExp, x: &BigFloat| -> ErrExp {
        if is_exact(e) && x.is_zero() {
            EXACT
        } else {
            clamp(e.max(round).saturating_add(1))
        }
    };
    Bound {
        re: part(propagated.re, &value.0),
        im: part(propagated.im, &value.1),
    }
}

/// Is the zero value of node `id`, computed from exact children, an
/// underflow?  `exp` is never 0, nor a product or a power of non-zero
/// values.  (astro-float's `exp` returns 0 below its exponent range without
/// flagging it inexact, and before 0.29 that zero counted as exact:
/// `Piecewise((1, exp(−4·10⁹) > 0), (0, True))` came out `0`.)
fn underflowed(arena: &Arena, id: ExprId, cache: &FxHashMap<ExprId, Complex>) -> bool {
    let nonzero = |c: &ExprId| cache.get(c).is_some_and(|v| mag(v).is_some());
    match arena.node(id) {
        ExprNode::Exp(_) => true,
        ExprNode::Mul(children) => children.iter().all(nonzero),
        ExprNode::Pow(base, _) => nonzero(base),
        _ => false,
    }
}

/// The bound of `b^x` for `b ± eb`, `x ± ex` (`x_literal`: the exponent's
/// exact value when it is a rational literal), the result's magnitude
/// exponent being `ub_out`.
///
/// A non-integer power has the cut of `ln` on the negative real axis: an
/// undecidable side has no bound.  The result is real (exact imaginary
/// part) for an exactly real base and exponent when the exponent is an
/// integer or the base is positive; `(−r)^(n/2) = ±i·r^(n/2)` (the half-
/// integer power of an exactly real negative base, evaluated exactly so)
/// has an exact real part.
fn pow_bound(
    x_literal: Option<&Q>,
    b: &Complex,
    eb: Bound,
    x: &Complex,
    ex: Bound,
    ub_out: ErrExp,
) -> Bound {
    let integer_exponent = match x_literal {
        Some(q) => q.is_integer(),
        None => ex.is_exact() && x.1.is_zero() && is_integer_value(&x.0),
    };
    if !integer_exponent && !contains_zero(b, eb.joint()) && cut_undecided(b, eb, Cut::NegativeReal)
    {
        return Bound::UNKNOWN;
    }
    let e = pow_error(x_literal, b, eb.joint(), x, ex.joint(), ub_out);
    let base_real = exactly_real(b, eb);
    let base_sign = if part_contains_zero(&b.0, eb.re) {
        0
    } else {
        sign_of(&b.0)
    };
    if base_real && exactly_real(x, ex) && (integer_exponent || base_sign > 0) {
        Bound::real(e)
    } else if base_real && base_sign < 0 && x_literal.is_some_and(|q| *q.denom() == BigInt::from(2))
    {
        Bound { re: EXACT, im: e }
    } else {
        Bound::both(e)
    }
}

/// The joint error bound of `b^x`, with `b ± 2^eb`, `x ± 2^ex`, the
/// result's magnitude exponent `ub_out` and `x`'s exact value when it is a
/// rational literal.
fn pow_error(
    x_literal: Option<&Q>,
    b: &Complex,
    eb: ErrExp,
    x: &Complex,
    ex: ErrExp,
    ub_out: ErrExp,
) -> ErrExp {
    if let Some(q) = x_literal {
        if q.is_zero() {
            return EXACT;
        }
        if is_exact(eb) {
            return EXACT; // only the rounding of the result
        }
        // Relative error scales by |q|: rel_out = |q|·rel_b.
        let log_q = log2_abs(q);
        if contains_zero(b, eb) {
            // A positive power of a value near 0 stays near 0: every point
            // of the ball has |w| < 2^(eb + 2), so |w^q| < 2^(q·(eb + 2)),
            // and two values differ by at most twice that.  (Before 0.29
            // this was 2^(eb/2 + 1), too small for q < ½: a cube root.)  A
            // negative power is unbounded.
            if !q.is_positive() {
                return UNKNOWN;
            }
            let qf = q.numer().to_f64().unwrap_or(f64::INFINITY)
                / q.denom().to_f64().unwrap_or(f64::INFINITY);
            let bound = qf * (eb.saturating_add(2)) as f64;
            return if bound.is_finite() && bound < 1e15 {
                clamp(bound.max(-1e15).ceil() as i64 + 1)
            } else {
                UNKNOWN
            };
        }
        let rel_b = eb - mag(b).unwrap_or(0);
        return ub_out.saturating_add(rel_b + log_q + 1);
    }
    // General exponent: b^x = exp(x·ln b), so
    // rel_out ≈ |x|·rel_b + |ln b|·err(x).
    if contains_zero(b, eb) && !is_exact(eb) {
        return UNKNOWN;
    }
    let rel_b = if is_exact(eb) {
        EXACT
    } else {
        eb - mag(b).unwrap_or(0)
    };
    let term_b = rel_b.saturating_add(upper(x, ex).max(0));
    // |ln b| ≤ |ln|b|| + π; ln|b| ≈ mag(b)·ln 2.
    let log_ln_b = {
        let m = mag(b).unwrap_or(0).unsigned_abs();
        let bound = (m as f64 * std::f64::consts::LN_2).max(std::f64::consts::PI) + 1.0;
        bound.log2().ceil() as i64
    };
    let term_x = if is_exact(ex) {
        EXACT
    } else {
        ex.saturating_add(log_ln_b)
    };
    ub_out.saturating_add(term_b.max(term_x) + 1)
}

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
//! Bounds are `log₂` values with a fractional part ([`ErrExp`]), and the
//! error of a node is its propagated bound **plus** its own rounding, the two
//! added exactly (`log₂(2^a + 2^b)`, [`lsum`]) — not their maximum rounded up
//! to the next power of two, which before 0.30 cost about a bit per node of
//! a perfectly conditioned chain.  The rounding of a result is two units in
//! its last place (a faithful rounding, with a margin for the transcendental
//! functions).  `r` below is the error radius of the argument, and every
//! derivative is bounded over the whole ball, not only at its centre.
//!
//! | node | propagated error of each part |
//! |---|---|
//! | exact rational (dyadic, fits the precision), `i` | 0, and no rounding |
//! | other rational, `π`, `e`, constants | 0 (the rounding only); the imaginary part is exact |
//! | `−a`, `conj a`, `re a`, `im a`, `min`, `max` | the argument's bound, no rounding |
//! | `a₁ + … + aₙ` | per part `Σ err(aᵢ)` plus one ulp of every partial sum after the first — absolute errors add, so cancellation shows as a result much smaller than its error; exact when every term's part is exact and no partial sum was rounded |
//! | `a₁ · … · aₙ` | per part, from `(a + bi)(c + di) = (ac − bd) + (ad + bc)i`: each product `x·y` contributes `abs(x)·err(y) + abs(y)·err(x) + err(x)·err(y)` and its rounding, and none at all when a factor is an exact 0 |
//! | `b^q` (rational literal `q`) | `abs(b^q)·abs(q)·ρ·(1 − ρ)^(−abs(q − 1))`, `ρ = r/abs(b)` (the mean value theorem, for every `q`); a positive power of a value indistinguishable from 0 is bounded by `2·(2r)^q`, a negative one has no bound |
//! | `b^e` | `abs(b^e)·(e^(r_w) − 1)` with `r_w` the error of `e·ln b` |
//! | `exp z` | `abs(exp z)·(e^r − 1)`, no bound beyond `r = 512` |
//! | `ln z`, `arg z`, `atan2` | `r/(abs(z) − r)`; no bound when `z` is indistinguishable from 0; `arg` of an exactly real, certainly positive `z` is exactly 0 |
//! | `sin`, `cos` / `sinh`, `cosh` | `r·cosh(abs(im z) + r)` / `r·cosh(abs(re z) + r)` — exactly `r` for `sin`, `cos` of a real argument; `r` of a rational `abs(q) ≥ 1` under `sin`, `cos`, `tan` is its rounding at `prec + log₂ abs(q)` bits (it is converted again, `evalf::trig_arg`) |
//! | `tan`, `tanh` | `r·(1 + w²)`, `w = (T + t)/(1 − T·t)`, `T = abs(f(z))`, `t = tan r`; no bound when `r·T > 1/8` (the ball reaches an eighth of the distance to a pole); `r` for a real `tanh` |
//! | `atan`, `asin`, `acos`, `asinh`, `acosh`, `atanh` | `r·Π(dᵢ − r)^(−k)` with the distances `dᵢ` to the two branch points (`abs(f′) = Π abs(z − pᵢ)^(−k)`); within a few error radii of a square-root branch point `p` the Hölder bound `2.06·√(2·(abs(z − p) + r))`, and no bound near a logarithmic one |
//! | `sign`, `floor`, `ceiling`, `heaviside`, `KroneckerDelta` | exact, or unknown when the argument (difference) is within its error of the threshold |
//! | special functions `f(x₁, …, xₙ)` | `2·Σᵢ abs(∂f/∂xᵢ)·err(xᵢ)` over the distinct inexact arguments, `abs(∂f/∂xᵢ)` bounded over the argument's error ball (next table, `sensitivity.rs`); no bound when the ball reaches a pole or branch point (its radius above an eighth of the distance); exact arguments cost nothing; a 0 from exact nonzero arguments is an underflow for the functions that vanish at no rational point (`erfc`, `Γ`, `Ei`, `Ai`, `Bi`, `I_ν`, `K_ν`, `Γ(s, x)`, `E_ν`), exact otherwise |
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
//! | `Si`, `Ci`, `Shi`, `Chi`, `Ei`, `li`, Fresnel | `x` | `min(1, 1/abs(x))`, `1/abs(x)`, `max(1.18, e^abs(x)/(2·abs(x)))`, `e^abs(x)/abs(x)`, `eˣ/abs(x)`, `1/abs(ln x)`, 1, and a change below `2/abs(x)` over a ball around `abs(x) > 2` (`abs(S − 1/2) < 0.43/abs(x)`, DLMF 7.12) |
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
//! | Lambert W, `W_k(z)` on every branch (`lambertw.rs`) | `2·abs(W′)·r` over the ball, `abs(W′) = abs(e^(−W)/(1 + W))`, while the first-order change is below a sixteenth of `min(1, abs(1 + W))`; unknown when the side of the branch's cut is undecidable |
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

/// Base-2 logarithm `e` of an absolute error bound, `|error| ≤ 2^e`, with a
/// fractional part: the bounds of a long computation are combined exactly
/// (`log₂(2^a + 2^b)`, [`lsum`]) and multiplied by the exact factors that
/// are known, not rounded up to a power of two at every node.  Before 0.30
/// the exponent was an integer, every node added about a bit (`max(e,
/// rounding) + 1`, `2^⌈log₂ n⌉` for a sum of `n` terms, `2·|f(z)|` for
/// `sin`), and `sin(⋯ sin(0) + 1 ⋯) + 1` nested 800 deep — a computation
/// whose true error grows like the logarithm of its depth — had a bound of
/// `2^1595` at 3,200 bits.
pub(super) type ErrExp = f64;

/// The value is exact.
pub(super) const EXACT: ErrExp = f64::NEG_INFINITY;

/// No bound is known: a division by a value indistinguishable from zero,
/// a decision (`sign`, `floor`, the side of a branch cut) at its threshold.
pub(super) const UNKNOWN: ErrExp = f64::INFINITY;

/// The bound of a value that underflowed to 0: below the smallest positive
/// float.
const UNDERFLOW: ErrExp = astro_float::EXPONENT_MIN as ErrExp;

/// Bounds beyond this are [`UNKNOWN`].  Only near the end of the `f64`
/// range: a bound of `2^(10¹¹²)` (`(1 + 10⁻¹⁵⁰)^(10¹⁵⁰)` at 128 bits, whose
/// rounded base is raised to a huge power) is useless as a bound but still
/// says how it shrinks with the precision, and the evaluation is repeated.
const LIMIT: ErrExp = 1e300;

/// Is `e` the bound of an exact value?
pub(super) fn is_exact(e: ErrExp) -> bool {
    e == EXACT
}

/// Is `e` [`UNKNOWN`] (or NaN from arithmetic on it)?
pub(super) fn is_unknown(e: ErrExp) -> bool {
    e.is_nan() || e > LIMIT
}

/// Is `e` the bound of a value that underflowed ([`UNDERFLOW`], possibly
/// after arithmetic): at the bottom of the exponent range, where it cannot
/// shrink with the precision?
pub(super) fn is_underflow(e: ErrExp) -> bool {
    !is_exact(e) && e < UNDERFLOW / 2.0
}

/// `e` normalised: NaN or beyond [`LIMIT`] is [`UNKNOWN`], below `−LIMIT`
/// (but not exact) is kept finite.
fn clamp(e: ErrExp) -> ErrExp {
    if is_unknown(e) {
        UNKNOWN
    } else if is_exact(e) {
        EXACT
    } else {
        e.max(-LIMIT)
    }
}

/// `e + k` (`log₂` of `2^e·2^k`): exact stays exact, unknown stays unknown.
pub(super) fn shift(e: ErrExp, k: f64) -> ErrExp {
    if is_exact(e) {
        EXACT
    } else if is_unknown(e) || k.is_nan() {
        UNKNOWN
    } else {
        clamp(e + k)
    }
}

/// `log₂(2^a + 2^b)`, exactly (to `f64` rounding, rounded up by a relative
/// `2⁻⁴⁰` so that the sum stays an upper bound); exact and unknown operands
/// behave as 0 and ∞.
pub(super) fn lsum(a: ErrExp, b: ErrExp) -> ErrExp {
    if is_unknown(a) || is_unknown(b) {
        return UNKNOWN;
    }
    if is_exact(a) {
        return b;
    }
    if is_exact(b) {
        return a;
    }
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    clamp(up(hi + (lo - hi).exp2().ln_1p() * std::f64::consts::LOG2_E))
}

/// The error bounds of the two parts of a complex value `re + i·im`:
/// `|Δre| ≤ 2^re`, `|Δim| ≤ 2^im` (see the module documentation).
#[derive(Clone, Copy, Debug, PartialEq)]
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
        lsum(self.re, self.im)
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
pub(super) fn part_mag(x: &BigFloat) -> Option<i64> {
    if x.is_zero() {
        None
    } else {
        x.exponent().map(i64::from)
    }
}

/// `log₂|x|`, rounded up by a hair (`−∞` for 0, `+∞` for a non-finite
/// value): an upper bound on the magnitude, a fraction of a bit tighter than
/// the binary exponent.
pub(super) fn part_lg(x: &BigFloat) -> f64 {
    if x.is_zero() {
        return f64::NEG_INFINITY;
    }
    if x.is_nan() || x.is_inf() {
        return f64::INFINITY;
    }
    let e = f64::from(x.exponent().unwrap_or(0));
    // The leading 53 bits of the (normalised) mantissa, read in place: this
    // runs for every node, and a copy of the number cost a fifth of the
    // evaluation time.  The mantissa is below `(top + 1)·2⁻⁵³`.
    if let Some(&top) = x.mantissa_digits().and_then(|w| w.last()) {
        // (`Word` is `u64` here, `u32` on 32-bit targets.)
        #[allow(clippy::useless_conversion)]
        let top = u64::from(top) << (64 - astro_float::Word::BITS);
        if top >> 63 == 1 {
            let m_up = ((top >> 11) as f64 + 1.0) * f64::EPSILON / 2.0;
            return up(e + m_up.log2());
        }
    }
    // A subnormal mantissa (not normalised): at most 1.
    up(e)
}

/// `x` rounded up by more than the `f64` rounding of the logarithms it came
/// from (`|x|·2⁻⁵⁰ + 10⁻¹²`).
fn up(x: f64) -> f64 {
    x + x.abs() * 1e-15 + 1e-12
}

/// `x` rounded down likewise.
fn down(x: f64) -> f64 {
    x - x.abs() * 1e-15 - 1e-12
}

/// `log₂|x|` rounded down (a lower bound on the magnitude).
pub(super) fn part_lg_low(x: &BigFloat) -> f64 {
    let l = part_lg(x);
    if l.is_finite() { down(down(l)) } else { l }
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

/// `log₂ |z|` rounded up (`−∞` for 0), `|z|² = re² + im²`.
pub(super) fn lg_abs(z: &Complex) -> f64 {
    let (a, b) = (part_lg(&z.0), part_lg(&z.1));
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    up(hi + 0.5 * (2.0 * (lo - hi)).exp2().ln_1p() * std::f64::consts::LOG2_E)
}

/// `log₂ |z|` rounded down: a lower bound on the magnitude.
pub(super) fn lg_abs_low(z: &Complex) -> f64 {
    let l = lg_abs(z);
    if l.is_finite() {
        down(down(down(l)))
    } else {
        l
    }
}

/// An exponent bounding the *true* value: its magnitude plus its error.
fn upper(z: &Complex, err: ErrExp) -> ErrExp {
    lsum(lg_abs(z), err)
}

/// Does the ball `z ± 2^err` contain 0?
pub(super) fn contains_zero(z: &Complex, err: ErrExp) -> bool {
    mag(z).is_none() || lg_abs_low(z) <= err
}

/// Does the interval `x ± 2^e` contain 0?  (An exact part only when it is
/// 0.)
pub(super) fn part_contains_zero(x: &BigFloat, e: ErrExp) -> bool {
    x.is_zero() || part_lg_low(x) <= e
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
/// that is zero or indistinguishable from zero): `log₂(|z|/2^err)`, from a
/// lower bound on `|z|`.
pub(super) fn accurate_bits(z: &Complex, err: ErrExp) -> Option<f64> {
    mag(z)?;
    if is_exact(err) {
        return Some(f64::INFINITY);
    }
    let l = lg_abs_low(z);
    (l > err).then_some(l - err)
}

/// The rounding of a `prec`-bit result at the magnitude `m` of its
/// exponent: two units in the last place, the error of a faithfully
/// rounded operation with a margin for astro-float's transcendental
/// functions (the same allowance as before 0.30, when it came from the
/// `+1` of every node).
fn ulp2(m: i64, prec: usize) -> ErrExp {
    (m - prec as i64 + 1) as f64
}

/// The rounding of `z` to `prec` bits, at its magnitude.
pub(super) fn rounding(z: &Complex, prec: usize) -> ErrExp {
    mag(z).map_or(EXACT, |m| ulp2(m, prec))
}

/// The rounding of one part to `prec` bits, at its own magnitude.
fn part_rounding(x: &BigFloat, prec: usize) -> ErrExp {
    part_mag(x).map_or(EXACT, |m| ulp2(m, prec))
}

/// `⌈log₂ n⌉` (0 for `n ≤ 1`).
pub(super) fn ceil_log2(n: usize) -> i64 {
    i64::from(usize::BITS - n.saturating_sub(1).leading_zeros())
}

/// Is the rational exactly representable as a `prec`-bit binary float?
fn exactly_representable(q: &Q, prec: usize) -> bool {
    let d = q.denom();
    let power_of_two = (d & (d - BigInt::one())).is_zero();
    power_of_two && q.numer().bits() <= prec as u64
}

/// The error bound of `value`, computed at working precision `prec` by a
/// routine that bounds its own error by `err`: that bound plus the rounding
/// of each part.  A part that is 0 with an exact bound stays exact.
pub(super) fn reported(value: &Complex, err: Bound, prec: usize) -> Bound {
    if !is_finite(value) || err.is_unknown() {
        return Bound::UNKNOWN;
    }
    let part = |x: &BigFloat, e: ErrExp| -> ErrExp {
        if exact_zero(x, e) {
            EXACT
        } else {
            lsum(e, part_rounding(x, prec))
        }
    };
    Bound {
        re: part(&value.0, err.re),
        im: part(&value.1, err.im),
    }
}

/// The propagated bound of a product of factors `v ± 2^e` (joint bounds):
/// `|Π(vᵢ + δᵢ) − Π vᵢ| ≤ Π(|vᵢ| + 2^eᵢ) − Π|vᵢ|` (the roundings of the
/// partial products are the caller's).  An exact zero factor makes the
/// product exact.
pub(super) fn product_error<'a>(parts: impl IntoIterator<Item = (&'a Complex, ErrExp)>) -> ErrExp {
    // Π(|vᵢ| + rᵢ) − Π|vᵢ| = Π|vᵢ|·(Π(1 + ρᵢ) − 1), `ρᵢ = rᵢ/|vᵢ|`, with the
    // last factor from `expm1(Σ ln(1 + ρᵢ))` — accurate for a tiny relative
    // error, where a difference of logarithms is not.
    let mut lmag = 0.0; // log₂ Π|vᵢ|, an upper bound
    let mut log1p_sum = 0.0; // Σ ln(1 + ρᵢ)
    let mut high = 0.0; // log₂ Π(|vᵢ| + rᵢ), for an inexact zero factor
    let mut any_inexact = false;
    let mut zero_factor = false;
    for (v, e) in parts {
        if is_unknown(e) {
            return UNKNOWN;
        }
        if mag(v).is_none() && is_exact(e) {
            return EXACT; // an exact zero factor
        }
        high += upper(v, e);
        if is_exact(e) {
            lmag += lg_abs(v);
            continue;
        }
        any_inexact = true;
        if mag(v).is_none() {
            zero_factor = true;
            continue;
        }
        lmag += lg_abs(v);
        let rho = e - lg_abs_low(v);
        log1p_sum += if rho > 60.0 {
            rho * std::f64::consts::LN_2
        } else {
            rho.exp2().ln_1p()
        };
    }
    if !any_inexact {
        return EXACT;
    }
    if zero_factor {
        return clamp(high);
    }
    let factor = if log1p_sum > 40.0 {
        log1p_sum * std::f64::consts::LOG2_E
    } else {
        log1p_sum.exp_m1().log2()
    };
    clamp(up(lmag + factor))
}

// ── Sums and products: value and bound together ────────────────────────────

/// One part of a running sum: the sum of the terms' bounds, the sum of the
/// roundings of the partial sums, and whether any term's part was not an
/// exact 0.
#[derive(Clone, Copy, Debug)]
pub(super) struct PartSum {
    terms: ErrExp,
    roundings: ErrExp,
    nonzero: bool,
}

impl PartSum {
    pub(super) const fn new() -> PartSum {
        PartSum {
            terms: EXACT,
            roundings: EXACT,
            nonzero: false,
        }
    }

    /// Account for a term's part `term ± 2^err`, after which the partial
    /// sum's part is `partial` (rounded to `prec` bits unless the term's
    /// part is an exact 0 or the partial sum is still the first term).
    pub(super) fn add(&mut self, term: &BigFloat, err: ErrExp, partial: &BigFloat, prec: usize) {
        let first = !self.nonzero;
        self.terms = lsum(self.terms, err);
        let exact_zero_term = exact_zero(term, err);
        self.nonzero |= !exact_zero_term;
        if !first && !term.is_zero() {
            // The addition rounds to within an ulp of the partial sum.
            let r = part_mag(partial).map_or(EXACT, |m| (m - prec as i64) as f64);
            self.roundings = lsum(self.roundings, r);
        }
    }

    /// The bound of this part of the sum: the terms' bounds plus the
    /// roundings of the partial sums, exact when every term's part was an
    /// exact 0.
    pub(super) fn bound(&self) -> ErrExp {
        if !self.nonzero {
            return EXACT;
        }
        lsum(self.terms, self.roundings)
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
        re.add(&v.0, b.re, &sum.0, prec);
        im.add(&v.1, b.im, &sum.1, prec);
    }
    if unknown {
        return (sum, Bound::UNKNOWN);
    }
    let finish = |acc: &PartSum, exact_terms: bool, parts: &mut dyn Iterator<Item = &BigFloat>| {
        let e = acc.bound();
        if is_exact(e) || (exact_terms && exact_sum(parts, prec, rm)) {
            EXACT
        } else {
            e
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
    e
}

/// The error exponent of a `log₂` bound.
fn from_lg(x: f64) -> ErrExp {
    clamp(x)
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
/// two values differ by at most `2·1.03·2^((j + 5)/2)`.
fn holder(j: ErrExp) -> ErrExp {
    if j > -8.0 {
        UNKNOWN
    } else {
        up((j + 5.0) / 2.0 + 2.06f64.log2())
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
    let mut distance = [0.0f64; 2];
    for (slot, &((pr, pi), kind)) in distance.iter_mut().zip(points.iter()) {
        let p = (BigFloat::from_i32(pr, 64), BigFloat::from_i32(pi, 64));
        let d = c_sub(z, &p, hp, RoundingMode::ToEven);
        let ld = lg_abs_low(&d);
        if mag(&d).is_some() && ld > j + 3.0 {
            *slot = ld;
        } else {
            return match kind {
                Log => Bound::UNKNOWN,
                Root => result(holder(j)),
                RootOnCut if !exactly_real(z, b) && cut_undecided(z, b, cut) => Bound::UNKNOWN,
                RootOnCut => result(holder(j)),
            };
        }
    }
    if cut_undecided(z, b, cut) {
        return Bound::UNKNOWN;
    }
    // `|f′(w)| = Π|w − pᵢ|^(−k)` exactly, and over the ball `|w − pᵢ| ≥
    // dᵢ − r`: the error is at most `r·Π(dᵢ − r)^(−k)`.  Far from both points
    // it is small (`|atan′ z| = 1/|z² + 1|`): before 0.30 the amplification
    // was at least 1, `atan(polygamma(1792, E))` (an argument of
    // `−1.2·10⁴²⁷⁵` with its relative error) had a ball of `2^10341` around
    // `−π/2`, which contains 0, and came out `0`.
    let k = if halve { 0.5 } else { 1.0 };
    let log_w: f64 = distance
        .iter()
        .map(|&ld| ld + (-(j - ld).exp2()).ln_1p() * std::f64::consts::LOG2_E)
        .sum();
    result(shift(j, up(-k * log_w)))
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
    use RealCut::{AboveOne, BelowOne, BelowZero, OutsideUnit, OutsideZeroTwo};
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
    // `log₂` of an upper bound on the value's magnitude.
    let lv = lg_abs(value);
    let node = arena.node(id);
    let propagated: Bound = match node {
        // Exact inputs: only the node's own rounding ([`finish`]).
        ExprNode::Num(nid) => {
            if exactly_representable(arena.num(*nid), prec) {
                return Bound::EXACT;
            }
            Bound::real(EXACT)
        }
        ExprNode::ImaginaryUnit | ExprNode::BoolTrue | ExprNode::BoolFalse => return Bound::EXACT,
        ExprNode::Pi
        | ExprNode::E
        | ExprNode::EulerGamma
        | ExprNode::Catalan
        | ExprNode::GoldenRatio
        | ExprNode::PhysicalConstant(..) => Bound::real(EXACT),

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
        // Exact operations: the child's bound, no rounding.
        ExprNode::Neg(c) | ExprNode::Conjugate(c) => {
            return child(*c).map_or(Bound::UNKNOWN, |(_, b)| b);
        }
        ExprNode::Re(c) => return child(*c).map_or(Bound::UNKNOWN, |(_, b)| Bound::real(b.re)),
        ExprNode::Im(c) => return child(*c).map_or(Bound::UNKNOWN, |(_, b)| Bound::real(b.im)),
        ExprNode::Pow(base, exp) => {
            let (b, eb) = known_child!(*base);
            let (x, ex) = known_child!(*exp);
            pow_bound(arena.as_num(*exp), b, eb, x, ex, lv)
        }

        // `d exp z = exp z · dz`: the argument's absolute error is the
        // result's relative error.  Before 0.29 the magnitude was clamped
        // at 1 (`ub_out.max(0)`), so `exp(−260.1) ≈ 2⁻³⁷⁵` carried the
        // absolute error of its argument, `2^(9 − prec)`: at the 384-bit cap
        // that ball contained 0, `ln(exp(−260.1))` had no bound and was
        // refused.
        ExprNode::Exp(c) => {
            let (z, b) = known_child!(*c);
            // `|e^(z+δ) − e^z| ≤ |e^z|·(e^|δ| − 1)`: the first-order `|e^z|·|δ|`
            // only for a small `δ`; `(e^r − 1)/r` for a radius `r` from 1/8
            // (before 0.30 always the first order, `e^1000·10` for
            // `exp(1000 ± 10)`, whose values reach `e^1010`).
            let j = b.joint();
            let e = if is_exact(j) {
                EXACT
            } else if j < -30.0 {
                // `e^r − 1 ≤ r·e^r`.
                shift(j, up(lv + j.exp2() * std::f64::consts::LOG2_E))
            } else if j <= 9.0 {
                clamp(up(lv + j.exp2().exp_m1().log2()))
            } else {
                return Bound::UNKNOWN;
            };
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
            // The argument of an exactly real number that is certainly
            // positive is exactly 0.  Before 0.30 it carried the child's
            // relative error: `arg(expint(1, √22))` was a 0 with a bound,
            // only zero to the precision reached (and the deep zero search
            // would pursue it for seconds).
            if is_arg
                && exactly_real(z, b)
                && !part_contains_zero(&z.0, b.re)
                && sign_of(&z.0) > 0
                && value.0.is_zero()
            {
                return Bound::EXACT;
            }
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
                // `|ln(z + δ) − ln z| ≤ r/(|z| − r)` (and `arg` likewise).
                let e = log_ball(j, lg_abs_low(z));
                if real { Bound::real(e) } else { Bound::both(e) }
            }
        }
        ExprNode::Sin(c) | ExprNode::Cos(c) | ExprNode::Sinh(c) | ExprNode::Cosh(c) => {
            let (z, b) = known_child!(*c);
            let j = trig_arg_error(arena, node, *c, z, b, prec);
            // The derivative over the ball: `|cos w|, |sin w| ≤ cosh(Im w)`,
            // `|cosh w|, |sinh w| ≤ cosh(Re w)` — exactly 1 for `sin`, `cos`
            // of a real argument, which are Lipschitz with constant 1
            // (before 0.30 `2·max(1, |f(z)|)`, a bit per `sin` of a chain).
            let across = if matches!(node, ExprNode::Sin(_) | ExprNode::Cos(_)) {
                (&z.1, b.im)
            } else {
                (&z.0, b.re)
            };
            let factor = if exact_zero(across.0, across.1) {
                0.0
            } else {
                let t = part_lg(across.0).exp2()
                    + if is_exact(across.1) {
                        0.0
                    } else {
                        across.1.exp2()
                    };
                if !j.is_finite() && !is_exact(j) {
                    UNKNOWN
                } else {
                    log2_cosh(t + if is_exact(j) { 0.0 } else { j.exp2() })
                }
            };
            let e = shift(j, factor);
            if exactly_real(z, b) {
                Bound::real(e)
            } else {
                Bound::both(e)
            }
        }
        ExprNode::Tan(c) | ExprNode::Tanh(c) => {
            let (z, b) = known_child!(*c);
            let arg_err = trig_arg_error(arena, node, *c, z, b, prec);
            // The first-order bound holds on a ball well inside the
            // distance to the nearest pole, about `1/|tan z|` there: no
            // bound beyond an eighth of it.  Before 0.30 the ball of
            // `tan(π/2·(1 − 10⁻⁸⁰))` at 128 bits (its argument rounded onto
            // the pole) was `±3·10³⁹` around a value of `10³⁹` — the true
            // value is `6.4·10⁷⁹` — and `Piecewise` decided `3.8·10⁵⁰ > tan(…)`
            // on it.
            // (A real `tanh` has no pole and a derivative below 1.)
            let poles = matches!(node, ExprNode::Tan(_)) || !exactly_real(z, b);
            if poles && !is_exact(arg_err) && arg_err + lv.max(0.0) > -3.0 {
                return Bound::UNKNOWN;
            }
            // `f(z + δ) = (f(z) ± g(δ))/(1 ∓ f(z)·g(δ))` with `|g(δ)| ≤ tan r`,
            // and `|f′(w)| ≤ 1 + |f(w)|²` over the ball; 1 for a real `tanh`.
            let factor = if !poles || is_exact(arg_err) {
                0.0
            } else {
                let r = arg_err.exp2();
                let t = r * (1.0 + r * r);
                let big_t = lv.exp2();
                let w = (big_t + t) / (1.0 - big_t * t);
                (1.0 + w * w).log2()
            };
            let e = shift(arg_err, up(factor));
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
                Bound::real(log_ball(j, lg_abs_low(&z)))
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
            // `|w/|w| − z/|z|| ≤ 2r/|z|` for `|w − z| ≤ r`.
            Bound::both(shift(log_ball(j, lg_abs_low(z)), 1.0))
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
            let d1 = z.0.sub(&value.0, p, rm);
            let one = BigFloat::from_i32(1, p);
            let next = match node {
                ExprNode::Floor(_) => value.0.add(&one, p, rm),
                _ => value.0.sub(&one, p, rm),
            };
            let d2 = next.sub(&z.0, p, rm);
            if part_contains_zero(&d1, b.re) || part_contains_zero(&d2, b.re) {
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
            // A copy of one of the arguments.
            return Bound::real(worst);
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
            let re = lsum(ei.re, ej.re);
            let im = lsum(ei.im, ej.im);
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

/// The joint error of the argument `c` (value `z ± b`) of the node `node`
/// as the node uses it.  `sin`, `cos` and `tan` of a rational literal
/// `|r| ≥ 1` convert it again with `log₂|r|` extra bits before reducing it
/// modulo 2π (`evalf::trig_arg`), so its error is the rounding at
/// `prec + log₂|r|` bits, about `2^−prec`, not the child's rounding at
/// `prec` bits.  Before 0.30 the bound used the latter (`sin(10¹⁰⁰)` had
/// a ball of `2^(333−prec)`) and only the agreement of two precisions,
/// which proves nothing, accepted the value.
fn trig_arg_error(
    arena: &Arena,
    node: &ExprNode,
    c: ExprId,
    z: &Complex,
    b: Bound,
    prec: usize,
) -> ErrExp {
    let reconverted = matches!(node, ExprNode::Sin(_) | ExprNode::Cos(_) | ExprNode::Tan(_))
        && matches!(arena.node(c), ExprNode::Num(_))
        && z.1.is_zero()
        && z.0.inexact()
        && z.0
            .exponent()
            .is_some_and(|e| e > 0 && i64::from(e) <= i64::from(arena.config.max_evalf_precision));
    if reconverted {
        1.0 - prec as f64
    } else {
        b.joint()
    }
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
            lsum(e, round)
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
///
/// Nor are the special functions that decay exponentially and vanish at
/// no nonzero rational point — `erfc`, `Γ`, `x!`, `Ei`, `Ai`, `Bi` and
/// their derivatives, `I_ν`, `K_ν`, `Γ(s, x)`, `E_ν(x)` — at nonzero
/// arguments (their zeros, where they have any, are irrational).  Before
/// 0.30 their underflow was an exact 0 as well: `sign(airyai(10¹⁰))`,
/// `sign(erfc(10⁵))` and `sign(besselk(0, 10¹⁰))` were `0`, truly `1`.
fn underflowed(arena: &Arena, id: ExprId, cache: &FxHashMap<ExprId, Complex>) -> bool {
    let nonzero = |c: &ExprId| cache.get(c).is_some_and(|v| mag(v).is_some());
    match arena.node(id) {
        ExprNode::Exp(_) => true,
        ExprNode::Mul(children) => children.iter().all(nonzero),
        ExprNode::Pow(base, _) => nonzero(base),
        ExprNode::Erfc(c) | ExprNode::Gamma(c) | ExprNode::Factorial(c) | ExprNode::Ei(c) => {
            nonzero(c)
        }
        // The variable is the last argument (the order or parameter may be 0).
        ExprNode::Apply(sid, args) => {
            matches!(
                arena.lib_fn(*sid),
                Some(
                    LibFn::AiryAi
                        | LibFn::AiryAiPrime
                        | LibFn::AiryBi
                        | LibFn::AiryBiPrime
                        | LibFn::BesselI
                        | LibFn::BesselK
                        | LibFn::UpperGamma
                        | LibFn::ExpInt
                )
            ) && args.last().is_some_and(nonzero)
        }
        _ => false,
    }
}

/// The bound of `b^x` for `b ± eb`, `x ± ex` (`x_literal`: the exponent's
/// exact value when it is a rational literal), `lv` bounding the result's
/// magnitude (`log₂`).
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
    lv: f64,
) -> Bound {
    let integer_exponent = match x_literal {
        Some(q) => q.is_integer(),
        None => ex.is_exact() && x.1.is_zero() && is_integer_value(&x.0),
    };
    if !integer_exponent && !contains_zero(b, eb.joint()) && cut_undecided(b, eb, Cut::NegativeReal)
    {
        return Bound::UNKNOWN;
    }
    let e = pow_error(x_literal, b, eb.joint(), x, ex.joint(), lv);
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

/// `log₂ |q|` of a rational (to `f64` precision, beyond its range from the
/// bit lengths).
fn lg_q(q: &Q) -> f64 {
    let (n, d) = (q.numer().abs(), q.denom().clone());
    match (n.to_f64(), d.to_f64()) {
        (Some(a), Some(b)) if a.is_finite() && b.is_finite() && a > 0.0 => a.log2() - b.log2(),
        _ => n.bits() as f64 - d.bits() as f64 + 1.0,
    }
}

/// The joint error bound of `b^x`, with `b ± 2^eb`, `x ± 2^ex`, `lv`
/// bounding the result's magnitude and `x`'s exact value when it is a
/// rational literal.
///
/// A literal `q`: `(b + δ)^q = b^q·(1 + ε)^q` with `|ε| ≤ ρ = 2^eb/|b|`, and by
/// the mean value theorem `|(1 + ε)^q − 1| ≤ |q|·ρ·(1 − ρ)^(−|q−1|)`, an
/// upper bound for every `q` (not only to first order: `(1 + 10⁻¹⁵⁰)^(10¹⁵⁰)`
/// at 128 bits has none, `ρ·q` being huge).
fn pow_error(
    x_literal: Option<&Q>,
    b: &Complex,
    eb: ErrExp,
    x: &Complex,
    ex: ErrExp,
    lv: f64,
) -> ErrExp {
    if let Some(q) = x_literal {
        if q.is_zero() {
            return EXACT;
        }
        if is_exact(eb) {
            return EXACT; // only the rounding of the result
        }
        let log_q = lg_q(q);
        if contains_zero(b, eb) {
            // A positive power of a value near 0 stays near 0: every point
            // of the ball has |w| ≤ |b| + 2^eb ≤ 2^(eb + 1), so |w^q| ≤
            // 2^(q·(eb + 1)), and two values differ by at most twice that.
            // A negative power is unbounded.
            if !q.is_positive() {
                return UNKNOWN;
            }
            let qf = q.numer().to_f64().unwrap_or(f64::INFINITY)
                / q.denom().to_f64().unwrap_or(f64::INFINITY);
            let bound = qf * (eb + 1.0 + 1e-9) + 1.0;
            return if bound.is_finite() && bound < LIMIT {
                clamp(up(bound))
            } else {
                UNKNOWN
            };
        }
        let rho = eb - lg_abs_low(b); // log₂ ρ < 0
        let q1 = (q - Q::one()).abs();
        let q1f = q1.numer().to_f64().unwrap_or(f64::INFINITY) / q1.denom().to_f64().unwrap_or(1.0);
        let growth = if q1f == 0.0 {
            0.0
        } else {
            -q1f * (-rho.exp2()).ln_1p() * std::f64::consts::LOG2_E
        };
        if !growth.is_finite() || growth > LIMIT {
            return UNKNOWN;
        }
        return clamp(up(lv + log_q + rho + growth));
    }
    // General exponent: b^x = exp(x·ln b).  The error of `w = x·ln b` is
    // `r_w ≤ |x|·r_ln + |ln b|·2^ex + r_ln·2^ex` with `r_ln = 2^eb/(|b| −
    // 2^eb)`, and `|e^(w+δ) − e^w| ≤ |e^w|·(e^r_w − 1)`.
    if contains_zero(b, eb) && !is_exact(eb) {
        return UNKNOWN;
    }
    let r_ln = if is_exact(eb) {
        EXACT
    } else {
        log_ball(eb, lg_abs_low(b))
    };
    let lx = upper(x, EXACT);
    // |ln b| ≤ |ln|b|| + π.
    let ln_b = (lg_abs(b).abs().max(lg_abs_low(b).abs()) * std::f64::consts::LN_2
        + std::f64::consts::PI)
        .log2();
    let r_w = lsum(lsum(shift(r_ln, lx), shift(ex, ln_b)), shift(r_ln, ex));
    if is_exact(r_w) {
        return EXACT;
    }
    if r_w > 9.0 {
        return UNKNOWN;
    }
    let factor = if r_w < -30.0 {
        r_w + r_w.exp2() * std::f64::consts::LOG2_E
    } else {
        r_w.exp2().exp_m1().log2()
    };
    clamp(up(lv + factor))
}

/// `log₂(r/(|z| − r))` for `r = 2^j < |z|` (`lz` a lower bound on
/// `log₂|z|`): the change of `ln` (and `arg`, and `sign`'s `1/|z|`) over the
/// ball.
fn log_ball(j: ErrExp, lz: f64) -> ErrExp {
    if is_exact(j) {
        return EXACT;
    }
    let rho = j - lz;
    if rho >= 0.0 {
        return UNKNOWN;
    }
    clamp(up(rho - (-rho.exp2()).ln_1p() * std::f64::consts::LOG2_E))
}

/// `log₂ cosh t` for `t ≥ 0` (`+∞` beyond `f64`).
fn log2_cosh(t: f64) -> f64 {
    if !t.is_finite() {
        return f64::INFINITY;
    }
    let t = t.abs();
    up(t * std::f64::consts::LOG2_E - 1.0 + (-2.0 * t).exp().ln_1p() * std::f64::consts::LOG2_E)
}

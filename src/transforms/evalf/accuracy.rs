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
//! | `+`, `·`, `−`, integer powers, `conj`, `re`, `im` of rationals and `i` only | their exact complex-rational value rounded once (`exact.rs`): the rounding of each part only, and an exact 0 part where the value has one |
//! | `−a`, `conj a`, `re a`, `im a`, `min`, `max` | the argument's bound, no rounding |
//! | `a₁ + … + aₙ` | per part `Σ err(aᵢ)` plus one ulp of every partial sum after the first — absolute errors add, so cancellation shows as a result much smaller than its error; exact when every term's part is exact and no partial sum was rounded |
//! | `a₁ · … · aₙ` | per part, from `(a + bi)(c + di) = (ac − bd) + (ad + bc)i`: each product `x·y` contributes `abs(x)·err(y) + abs(y)·err(x) + err(x)·err(y)` and its rounding, and none at all when a factor is an exact 0 |
//! | `b^q` (rational literal `q`; a real result) | `abs(b^q)·abs(q)·ρ·(1 − ρ)^(−abs(q − 1))`, `ρ = r/abs(b)` (the mean value theorem, for every `q`); a positive power of a value indistinguishable from 0 is bounded by `2·(2r)^q`, a negative one has no bound |
//! | `b^e` (a real result) | `abs(b^e)·(e^(r_w) − 1)` with `r_w` the error of `e·ln b`, plus `abs(b^e)·abs(e)·abs(ln b)·2^(2 − 32 − prec)` for `exp(e·ln b)` formed with 32 guard bits |
//! | `bⁿ` (`n` in `i32`, `b` not real), `b^e` (a complex result) | per part along the route `evalf` takes: binary powering, each product bounded as for `·` (and `1/b^abs(n)` for `n < 0`); `exp(e·Log b)` with `Log b = ln c_abs(b) + i·atan2`, the rounding of `c_abs` an absolute error of the real part, the base's error carried by `Log′ = 1/w` part by part (next row), then `exp` as below (`accuracy::power_parts`) |
//! | `exp z` | `abs(exp z)·(e^r − 1)` for a real `z`; otherwise per part, `abs(ΔRe f) ≤ abs(Re f′)·r_re + abs(Im f′)·r_im` and `abs(ΔIm f) ≤ abs(Im f′)·r_re + abs(Re f′)·r_im` (Cauchy–Riemann) with `f′ = exp` bounded over the ball, and each part rounded at its own magnitude; no bound beyond `r = 512` unless the whole ball lies below the exponent range (`e^(Re z + r)`) |
//! | `ln z`, `arg z`, `atan2` | `r/(abs(z) − r)`; no bound when `z` is indistinguishable from 0; `arg` of an exactly real, certainly positive `z` is exactly 0; off the positive reals the real part `ln c_abs(z)` carries the rounding of `c_abs`, `2^(1.5 − prec)` absolute (exact for `z ∈ {−1, ±i}`) |
//! | `sin`, `cos` / `sinh`, `cosh` | `r` for `sin`, `cos` of a real argument (`r` of a rational `abs(q) ≥ 1` under `sin`, `cos`, `tan` is its rounding at `prec + log₂ abs(q)` bits: it is converted again, `evalf::trig_arg`), `r·cosh(abs(re z) + r)` for `sinh`, `cosh`; at an argument that is not exactly real per part as for `exp`, with `abs(Re f′) ≤ cosh B`, `abs(Im f′) ≤ sinh B` over the ball, `B = abs(im z) + r_im` for `sin`, `cos` (`abs(re z) + r_re` for the hyperbolic ones, whose `Re`/`Im` roles follow `f′`), each part rounded at its own magnitude |
//! | `tan`, `tanh` | `r·(1 + w²)`, `w = (T + t)/(1 − T·t)`, `T = abs(f(z))`, `t = tan r`; no bound when `r·T > 1/8` (the ball reaches an eighth of the distance to a pole); `r` for a real `tanh` |
//! | `atan`, `asin`, `acos`, `asinh`, `acosh`, `atanh` | `r·Π(dᵢ − r)^(−k)` with the distances `dᵢ` to the two branch points (`abs(f′) = Π abs(z − pᵢ)^(−k)`); within a few error radii of a square-root branch point `p` the Hölder bound `2.06·√(2·(abs(z − p) + r))`, and no bound near a logarithmic one |
//! | `sign`, `floor`, `ceiling`, `heaviside`, `DiracDelta`, `KroneckerDelta` | exact, or unknown when the argument (difference) is within its error of the threshold (0 for `DiracDelta`, whose value at an exact 0 or a non-real argument is refused, and unknown at an argument not exactly real) |
//! | special functions `f(x₁, …, xₙ)` | `2·Σᵢ abs(∂f/∂xᵢ)·err(xᵢ)` over the distinct inexact arguments, `abs(∂f/∂xᵢ)` bounded over the argument's error ball (next table, `sensitivity.rs`); no bound when the ball reaches a pole or branch point (its radius above an eighth of the distance); exact arguments cost nothing; a 0 from exact nonzero arguments is an underflow for the functions that vanish at no rational point (`erfc`, `Γ`, `Ei`, `Ai`, `Bi`, `I_ν`, `K_ν`, `Γ(s, x)`, `E_ν`) and for `J_ν` with `ν > abs(x)`, exact otherwise |
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
//! bound on both parts, and the rounding of the result (at its magnitude) —
//! except `exp`, `sin`, `cos`, `sinh`, `cosh` at an argument that is not
//! exactly real and the complex routes of powers, which bound each part on
//! its own ([`analytic_parts`], [`power_parts`]): next to an axis the small
//! part's error is small too (since 0.55; up to 0.54 `(1 + 10⁻⁴⁵i)²`
//! printed `1`).
//!
//! Nodes whose evaluator sees more than its children's values report their
//! own bound ([`reported`] adds the rounding, as [`node_error`] does):
//!
//! | node | error of the result |
//! |---|---|
//! | finite `Sum` / `Product` | the terms' bounds (evaluated with the index bound exactly), as for `+` / `·`, plus the rounding of every partial result |
//! | infinite `Sum` of a hypergeometric term | the first term's relative error carried by the recurrence, the recurrence's roundings, and a rigorous geometric bound on the tail (`hypsum.rs`) |
//! | infinite `Sum` of a smooth term (`asymsum.rs`) | the terms below `N` as for a finite sum, the balls of the expansion's coefficients times the power sums, the Euler–Maclaurin (Euler–Boole) remainders, and the sum of the expansion's remainder |
//! | `ζ`, `η`, `ψ`, `ψ⁽ᵐ⁾`, `harmonic`, `Γ`, `x!`, `B` at a non-real argument (`zeta.rs`) | the routine's own `2^(−prec−12)·abs(f)`, plus twice the radius times a bound on `abs(f′)` over the ball: the Euler–Maclaurin formula differentiated term by term for `ζ` (and `η`), `(m+1)!·Σ (abs(z + k) − r)^(−m−2)` for `ψ⁽ᵐ⁾`, `abs(Γ)·(e^(Ψr) − 1)` for `Γ` (`Ψ` the bound on `abs(ψ)` of `loggamma.rs`); unknown within eight radii of a pole |
//! | `RootOf` | the radius of a certified inclusion disk of the root (`poly::roots::root_balls`), on both parts of a non-real root and on the real part of a root certified real (whose imaginary part is exactly 0); unknown when the disks do not isolate the roots |
//! | `RootSum` | the body's bound at each root, the root bounded by its inclusion disk, as for `+` |
//! | `Piecewise` | the chosen branch's bound when every condition up to it is decided with certainty (a comparison whose difference is outside its error ball, or of exact values); unknown otherwise, like `sign` at its threshold |
//! | physical constants | their value's bound |
//! | Lambert W, `W_k(z)` on every branch (`lambertw.rs`) | `2·abs(W′)·r` over the ball, `abs(W′) = abs(e^(−W)/(1 + W))`, while the first-order change is below a sixteenth of `min(1, abs(1 + W))`; unknown when the side of the branch's cut is undecidable |
//! | `Li_s(z)` for a non-real `s` (`polylog.rs`) | its own error, and for `abs(z) + r < 3/4` the series' derivative bounds in `z` and `s`; beyond, `2·abs(Li_(s−1)(z)/z)·r` at 128 bits (grown over the ball as for a real order) and a forward difference in `s` |
//! | `Li_s(z)` for a real `s` and `z` off `[−1, 1]` (`polylog.rs`) | the routine's own `2^(−prec−12)·abs(Li_s)` (every truncation bounded), plus `2·abs(Li_(s−1)(z)/z)·r` (at 128 bits, grown by `(1 − r/d)^(−max(1, abs(s − 2)))`, `d = abs(z − 1)`) and `2·abs(∂Li_s/∂s)·r_s` (a forward difference at 128 bits); unknown when `r > d/8` or the ball meets the cut `[1, ∞)` with an imaginary part only within its error of 0; the imaginary part is exact for an exactly real `z < −1` |
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

use super::ball::{Ball, CBall, Cache};
use super::wp::RM;
use std::borrow::Borrow;

use astro_float::BigFloat;
use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};
use rustc_hash::FxHashMap;

use crate::base::arena::Arena;
use crate::base::bigcomplex::{Complex, c_add, c_mul, c_one, c_sub, c_zero};
use crate::base::func::Func;
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
pub(super) const UNDERFLOW: ErrExp = astro_float::EXPONENT_MIN as ErrExp;

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
pub(super) fn exactly_representable(q: &Q, prec: usize) -> bool {
    let d = q.denom();
    let power_of_two = (d & (d - BigInt::one())).is_zero();
    power_of_two && q.numer().bits() <= prec as u64
}

/// The bound of the part `x`, the rational `q` rounded to `prec` bits: exact
/// when `q` is 0 or exactly representable, otherwise its rounding.
pub(super) fn rational_part(q: &Q, x: &BigFloat, prec: usize) -> ErrExp {
    if q.is_zero() || exactly_representable(q, prec) {
        EXACT
    } else {
        part_rounding(x, prec)
    }
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
fn exact_sum<'a>(parts: impl Iterator<Item = &'a BigFloat>, prec: usize) -> bool {
    let mut s = BigFloat::new(prec);
    for x in parts {
        if x.is_zero() {
            continue;
        }
        s = s.add(&clean(x), prec, RM);
        if s.inexact() {
            return false;
        }
    }
    true
}

/// `a₁ + … + aₙ` summed left to right from 0 at `prec` bits, as `evalf`
/// sums, with the bound of each part ([`PartSum`]); a part whose terms are
/// all exact and whose partial sums were never rounded is exact.
pub(super) fn add_with_bound(terms: &[(&Complex, Bound)], prec: usize) -> CBall {
    let mut sum = c_zero(prec);
    let (mut re, mut im) = (PartSum::new(), PartSum::new());
    let mut unknown = false;
    for &(v, b) in terms {
        unknown |= b.is_unknown();
        sum = c_add(&sum, v, prec, RM);
        re.add(&v.0, b.re, &sum.0, prec);
        im.add(&v.1, b.im, &sum.1, prec);
    }
    if unknown {
        return CBall::unknown(sum);
    }
    let finish = |acc: &PartSum, exact_terms: bool, parts: &mut dyn Iterator<Item = &BigFloat>| {
        let e = acc.bound();
        if is_exact(e) || (exact_terms && exact_sum(parts, prec)) {
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
    CBall::new(sum, bound)
}

/// A term certainly not 0 that a sum lost below its error bound: the sum's
/// error ball is wider than the term, so the term changed nothing the
/// evaluation can see (SymPy's `add_terms` drops it the same way).  Harmless
/// for a value with certified digits — the bound covers the term — but not
/// for a value that is zero to the precision reached: if the rest of the sum
/// cancels exactly, the value is the lost term, not 0.
/// `sin²1 + cos²1 − 1 + e^(−10¹⁰)` is `e^(−10¹⁰)`, and `e^(−10¹⁰)·(sin²1 +
/// cos²1) − e^(−10¹⁰) + e^(−2·10¹⁰)` is `e^(−2·10¹⁰)`; before 0.35 both
/// printed `0` (see `evaluate_adaptive` in `evalf.rs`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Absorbed {
    /// The fewest bits by which an error bound exceeded a term in range it
    /// lost (the precision that keeps the nearest one), when one was lost.
    pub(super) nearest: Option<f64>,
    /// A term below the exponent range was lost: no precision keeps it.
    pub(super) below_range: bool,
}

impl Absorbed {
    /// Record the term of magnitude `2^term` (absolute `log₂`) of a value
    /// with the error bound `2^radius`, when it is lost below that bound.
    pub(super) fn note(slot: &mut Option<Absorbed>, term: f64, radius: ErrExp) {
        if !term.is_finite() || !radius.is_finite() || is_unknown(radius) || term >= radius {
            return;
        }
        let mut a = slot.unwrap_or(Absorbed {
            nearest: None,
            below_range: false,
        });
        if term < UNDERFLOW / 2.0 {
            a.below_range = true;
        } else {
            let d = radius - term;
            a.nearest = Some(a.nearest.map_or(d, |n| n.min(d)));
        }
        *slot = Some(a);
    }
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
///
/// `abs(x)` in the propagated part is `log₂ abs(x)` rounded up by a hair
/// ([`part_lg`]), not the binary exponent: before 0.44 it was `2^m ≥ abs(x)`
/// with `2^(m−1) ≤ abs(x) < 2^m`, up to twice `abs(x)` for every factor in
/// `[1, 2)`, so that the bound of a product of `n` factors near 1 grew like
/// `2^n` (`Π_{k≤400} (1 + 1/(k² + 1/3))` lost 400 bits of its bound).  The
/// roundings stay at the exponents (units in the last place).
fn product_part(
    terms: [(Factor<'_>, Factor<'_>); 2],
    result: &BigFloat,
    subtract: bool,
    prec: usize,
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
        let (lx, ly) = (part_lg(x), part_lg(y));
        err = lsum(err, product_rule(lx, ex, ly, ey));
        round = round.max(lmag(x) + lmag(y) - p);
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
            let t = clean(x).mul(&clean(y), prec, RM);
            rounded |= t.inexact();
            let next = match acc {
                None => t,
                Some(a) if subtract => a.sub(&t, prec, RM),
                Some(a) => a.add(&t, prec, RM),
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

/// The propagated error of a product `x·y` of `x ± 2^ex`, `y ± 2^ey` with
/// `abs(x) ≤ 2^lx`, `abs(y) ≤ 2^ly`: `abs(x)·err(y) + abs(y)·err(x) +
/// err(x)·err(y)` (the rounding of the product is not included).
pub(super) fn product_rule(lx: f64, ex: ErrExp, ly: f64, ey: ErrExp) -> ErrExp {
    lsum(lsum(lx + ey, ly + ex), ex + ey)
}

/// `a₁ · … · aₙ` multiplied left to right from 1 at `prec` bits, as `evalf`
/// multiplies, with the bound of each part ([`product_part`] for every
/// step).  An exact-zero factor makes the product exact; a product of
/// non-zero factors that came out 0 underflowed.
pub(super) fn mul_with_bound(factors: &[(&Complex, Bound)], prec: usize) -> CBall {
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
        let next = c_mul(&product, v, prec, RM);
        if !unknown && !zero_factor {
            let (pr, pi) = ((&product.0, err[0]), (&product.1, err[1]));
            let (fr, fi) = ((&v.0, lg(b.re)), (&v.1, lg(b.im)));
            err = [
                product_part([(pr, fr), (pi, fi)], &next.0, true, prec),
                product_part([(pr, fi), (pi, fr)], &next.1, false, prec),
            ];
        }
        product = next;
    }
    if unknown {
        return CBall::unknown(product);
    }
    if zero_factor {
        return CBall::exact(product);
    }
    if mag(&product).is_none() && all_nonzero && !factors.is_empty() {
        let im = if all_real { EXACT } else { UNDERFLOW };
        return CBall::new(product, Bound { re: UNDERFLOW, im });
    }
    let bound = Bound {
        re: from_lg(err[0]),
        im: from_lg(err[1]),
    };
    CBall::new(product, bound)
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
    x.sub(&BigFloat::from_i32(c, 64), p, RM)
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

/// Where a ball lies relative to a branch cut ([`cut_side`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CutSide {
    /// The ball does not meet the cut: its part perpendicular to the cut
    /// has a certain sign (the side the value takes), or the ball misses
    /// the cut's segment of the line.
    Off,
    /// Exactly on the cut: the perpendicular part is an exact 0 and the
    /// ball meets the cut's segment; the value is the principal one by
    /// convention.
    On,
    /// The perpendicular part is within its error of 0 without being
    /// exactly 0 and the ball meets the cut's segment: the side is
    /// undecidable at this precision.
    Undecided,
}

/// Where the argument `z ± b` lies relative to `cut`.
///
/// The part perpendicular to the cut is `im` for the real-axis cuts and
/// `re` for the imaginary-axis ones; the box `re ± 2^b.re`, `im ± 2^b.im`
/// meets the cut when that part may be 0 and the other part meets the
/// cut's segment of its axis.
pub(super) fn cut_side(z: &Complex, b: Bound, cut: Cut) -> CutSide {
    let (across, across_err, along, along_err) = match cut {
        Cut::ImaginaryBeyondOne => (&z.0, b.re, &z.1, b.im),
        _ => (&z.1, b.im, &z.0, b.re),
    };
    let exactly_on_line = exact_zero(across, across_err);
    if !exactly_on_line && !part_contains_zero(across, across_err) {
        return CutSide::Off;
    }
    let meets = match cut {
        Cut::NegativeReal => meets_at_most(along, along_err, 0),
        Cut::RealBelowOne => meets_at_most(along, along_err, 1),
        Cut::RealBeyondOne | Cut::ImaginaryBeyondOne => {
            meets_at_most(along, along_err, -1) || meets_at_least(along, along_err, 1)
        }
    };
    match (meets, exactly_on_line) {
        (false, _) => CutSide::Off,
        (true, true) => CutSide::On,
        (true, false) => CutSide::Undecided,
    }
}

/// Is the side of `cut` on which the argument `z ± b` lies undecidable
/// ([`CutSide::Undecided`])?
///
/// The box `re ± 2^b.re`, `im ± 2^b.im` must meet the cut, and the part
/// perpendicular to it (`im` for the real-axis cuts, `re` for the
/// imaginary-axis ones) must be within its error of 0 without being exactly
/// 0: an exact 0 is on the cut, where the principal value is the
/// convention, and a part outside its error ball has a certain sign.
pub(super) fn cut_undecided(z: &Complex, b: Bound, cut: Cut) -> bool {
    // `cut_side(z, b, cut) == CutSide::Undecided`, without testing the
    // segment for an exactly real argument (every real `ln`).
    let (across, across_err) = match cut {
        Cut::ImaginaryBeyondOne => (&z.0, b.re),
        _ => (&z.1, b.im),
    };
    if exact_zero(across, across_err) || !part_contains_zero(across, across_err) {
        return false;
    }
    cut_side(z, b, cut) == CutSide::Undecided
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

fn analyticity(node: &ExprNode) -> Analytic {
    use Analytic::{Everywhere, OffCut, Unknown};
    use RealCut::{AboveOne, BelowOne, BelowZero, OutsideUnit, OutsideZeroTwo};
    match node {
        n if n.func_head().is_some_and(entire_or_meromorphic) => Everywhere,
        ExprNode::LogGamma(_) | ExprNode::Ci(_) | ExprNode::Ei(_) => OffCut(0, BelowZero),
        ExprNode::Li(_) => OffCut(0, BelowOne),
        ExprNode::Lib(f, _) => match f {
            LibFn::Chi => OffCut(0, BelowZero),
            LibFn::BesselJ
            | LibFn::BesselY
            | LibFn::BesselI
            | LibFn::BesselK
            | LibFn::ExpInt
            | LibFn::UpperGamma
            | LibFn::LowerGamma => OffCut(1, BelowZero),
            LibFn::PolyLog => OffCut(1, AboveOne),
            LibFn::EllipticK | LibFn::EllipticE => OffCut(0, AboveOne),
            LibFn::ErfInv => OffCut(0, OutsideUnit),
            LibFn::ErfcInv => OffCut(0, OutsideZeroTwo),
            LibFn::AssocLegendre => OffCut(2, OutsideUnit),
            _ => Unknown,
        },
        _ => Unknown,
    }
}

/// The functions [`analyticity`] calls entire or meromorphic in every
/// argument (for the arguments `evalf` accepts): `Γ`, `ψ`, `ψ⁽ⁿ⁾`, `ζ`,
/// `erf`, `erfc`, `B`, `n!`, `C(n, k)`, `Si`, `δ`, and of the library
/// functions `erfi`, `Shi`, the Fresnel and Airy functions, `η` and the
/// orthogonal polynomials.
fn entire_or_meromorphic(f: Func) -> bool {
    matches!(
        f,
        Func::Gamma
            | Func::Digamma
            | Func::Polygamma
            | Func::Zeta
            | Func::Erf
            | Func::Erfc
            | Func::Beta
            | Func::Factorial
            | Func::Binomial
            | Func::Si
            | Func::DiracDelta
            | Func::Erfi
            | Func::Shi
            | Func::FresnelS
            | Func::FresnelC
            | Func::AiryAi
            | Func::AiryBi
            | Func::AiryAiPrime
            | Func::AiryBiPrime
            | Func::DirichletEta
            | Func::Legendre
            | Func::ChebyshevT
            | Func::ChebyshevU
            | Func::Hermite
            | Func::Laguerre
            | Func::Gegenbauer
            | Func::Jacobi
            | Func::AssocLaguerre
    )
}

/// The functions whose error [`node_error`] bounds by their sensitivity
/// to each argument: the special functions but `W` (bounded by its own
/// evaluation), and every library and internal function.
fn error_by_sensitivity(f: Func) -> bool {
    f.family() != crate::base::func::Family::Node || (f.is_special() && f != Func::LambertW0)
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

// ── Node bounds, from the arguments' balls ──────────────────────────────────

/// What a node's bound rule gives: the final bound, or the propagated one
/// to which [`finish`] adds the node's rounding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum NodeBound {
    /// The node's bound as it stands.
    Final(Bound),
    /// The propagated part; the rounding of the result is added.
    Propagated(Bound),
}

/// The bound of a value of `e^w` computed as 0 for every `w` within `2^j`
/// of a `z` whose real part is `re`, when the whole ball lies below the
/// exponent range: `|e^w| ≤ e^(Re z + 2^j)`.  `None` when part of the ball
/// may reach the range, or the radius is exact (another rule) or unknown.
/// Up to 0.52 a radius beyond `2⁹` gave no bound even there: the nodes of
/// `∫₀^∞ e^{tx}·f(x) dx` for a Weibull density `f` where the exponent is
/// `−10⁷⁸ ± 2⁹³` (`x ≈ 1/2`, shape `31/2`, scale `6·10⁻⁶`) refused the
/// quadrature of its moment generating function.
fn exp_ball_below_range(re: &BigFloat, j: ErrExp) -> Option<ErrExp> {
    if is_exact(j) || is_unknown(j) || !re.is_negative() || re.is_inf() || re.is_nan() {
        return None;
    }
    // `abs(Re z) ≥ 2^lo`; `Re z + 2^j` read no further than `−10³⁰⁰` (an
    // `f64`), and at most `−2^(lo − 1)` for a radius below half of it.
    let lo = part_lg_low(re);
    let below = if lo < 996.0 && j < 996.0 {
        let (depth, radius) = (lo.exp2(), j.exp2());
        (radius < depth).then_some(radius - depth)
    } else {
        (j < lo - 1.0).then(|| -(lo - 1.0).min(996.0).exp2())
    }?;
    let e = up(below * std::f64::consts::LOG2_E);
    (e < UNDERFLOW).then_some(e.max(-LIMIT))
}

/// The bound rules of the nodes (the module documentation's table), one
/// method per node kind, on the argument's ball.
impl<Z: Borrow<Complex>> Ball<Z> {
    /// The propagated bound of `exp` at this argument, for a value `value`
    /// of magnitude `2^lv` computed at `prec` bits (see [`node_error`]).
    ///
    /// At an argument that is not exactly real each part has its own bound
    /// ([`analytic_parts`]; `exp′ = exp`, so over the ball `|Re e^w| ≤
    /// |Re e^z| + |e^z|·(e^r − 1)`, and the imaginary part likewise) and its
    /// own rounding ([`own_roundings`]: `e^a·cos b`, `e^a·sin b`).  Up to
    /// 0.54 both parts carried `|e^z|·(e^r − 1)` and the rounding at `|e^z|`:
    /// `exp(1/10⁵⁰ + i)·(cos 1 − i sin 1)`-type values lost the smaller
    /// part, and `exp(i(π/2 + 10⁻⁴⁵))` printed `i` at 10 digits (its real
    /// part `−10⁻⁴⁵` needs the argument to about 190 bits, which the
    /// precision search now pursues: `evalf::evaluate_adaptive`).
    fn exp_error(&self, lv: f64, value: &Complex, prec: usize) -> NodeBound {
        let (z, b) = (self.z(), self.err);
        // `|e^(z+δ) − e^z| ≤ |e^z|·(e^|δ| − 1)`: the first-order `|e^z|·|δ|`
        // only for a small `δ`; `(e^r − 1)/r` for a radius `r` from 1/8
        // (before 0.30 always the first order, `e^1000·10` for
        // `exp(1000 ± 10)`, whose values reach `e^1010`).
        let j = b.joint();
        // `e^z` is never 0: a 0 underflowed, and so does every value in
        // the ball, `|e^(z+δ)| ≤ |e^z|·e^|δ|`.  Up to 0.31 its bound was
        // unknown (`lv` is `−∞`) and `exp(−10¹⁰ − √2)` was refused as
        // lacking precision.
        if lv == f64::NEG_INFINITY && j <= 9.0 {
            let e = if is_exact(j) {
                UNDERFLOW
            } else {
                up(UNDERFLOW + j.exp2() * std::f64::consts::LOG2_E)
            };
            let im = if exactly_real(z, b) { EXACT } else { e };
            return NodeBound::Final(Bound { re: e, im });
        }
        if lv == f64::NEG_INFINITY
            && let Some(e) = exp_ball_below_range(&z.0, j)
        {
            let im = if exactly_real(z, b) { EXACT } else { e };
            return NodeBound::Final(Bound { re: e, im });
        }
        let e = if is_exact(j) {
            EXACT
        } else if j < -30.0 {
            // `e^r − 1 ≤ r·e^r`.
            shift(j, up(lv + j.exp2() * std::f64::consts::LOG2_E))
        } else if j <= 9.0 {
            clamp(up(lv + j.exp2().exp_m1().log2()))
        } else {
            return NodeBound::Final(Bound::UNKNOWN);
        };
        if exactly_real(z, b) {
            return NodeBound::Propagated(Bound::real(e));
        }
        // `|e^z|·(e^r − 1)`, and the rounding of the value (`e^z` is `value`
        // to within it, a part that came out 0 included).
        let slack = lsum(e, rounding(value, prec));
        let d_re = lsum(part_lg(&value.0), slack);
        let d_im = lsum(part_lg(&value.1), slack);
        NodeBound::Final(own_roundings(
            value,
            analytic_parts(d_re, d_im, b.re, b.im),
            prec,
        ))
    }

    /// The propagated bound of `ln` (or `arg`, `is_arg`) at this argument,
    /// whose value is `value` at `prec` bits (see [`node_error`]).
    ///
    /// Off the positive reals `ln z` is formed as `ln c_abs(z) + i·arg z`,
    /// and `c_abs` (two squares, their sum and a square root) is within
    /// `2^(1 − prec)` of `|z|` relative: an *absolute* error of the real
    /// part, which the rounding at the value's magnitude does not cover
    /// when `|ln|z||` and `|arg z|` are both small.  Up to 0.54 it was
    /// left out: the real part of `ln(1 + 10⁻¹⁵i)` printed as
    /// `5.00000000000000000000000000063e-31` at 30 digits (it is
    /// `4.9999999999999999999999999999975e-31`) and `eval_complex64` gave
    /// `4.999999957e-31`.  Exact for `z ∈ {−1, ±i}` (`ln(−1) = iπ` keeps an
    /// exact real part).
    fn ln_error(&self, is_arg: bool, value: &Complex, prec: usize) -> NodeBound {
        let (z, b) = (self.z(), self.err);
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
            return NodeBound::Final(Bound::EXACT);
        }
        let mut propagated = if b.is_exact() {
            if real {
                Bound::real(EXACT)
            } else {
                Bound::EXACT
            }
        } else {
            let j = b.joint();
            if contains_zero(z, j) || cut_undecided(z, b, Cut::NegativeReal) {
                return NodeBound::Final(Bound::UNKNOWN);
            }
            // `|ln(z + δ) − ln z| ≤ r/(|z| − r)` (and `arg` likewise).
            let e = log_ball(j, lg_abs_low(z));
            if real { Bound::real(e) } else { Bound::both(e) }
        };
        // The route of `eval_node`: the real `ln` of a positive real centre,
        // `ln c_abs(z) + i·atan2` otherwise.
        let modulus_route = !is_arg && !(z.1.is_zero() && !z.0.is_negative());
        let one = BigFloat::from_i32(1, 64);
        let is_one = |v: &BigFloat| v.abs().cmp(&one) == Some(0);
        let unit = (z.1.is_zero() && is_one(&z.0)) || (z.0.is_zero() && is_one(&z.1));
        if modulus_route && !unit {
            propagated.re = lsum(propagated.re, 1.5 - prec as f64);
        }
        NodeBound::Propagated(propagated)
    }

    /// The propagated bound of `abs` at this argument (see [`node_error`]).
    fn abs_error(&self) -> NodeBound {
        let (z, b) = (self.z(), self.err);
        NodeBound::Propagated(Bound::real(if exactly_real(z, b) {
            b.re
        } else {
            b.joint()
        }))
    }

    /// The propagated bound of `DiracDelta` at this argument (see [`node_error`]):
    /// exact for a real argument (an exact 0 imaginary part; `δ` has no
    /// value off the reals, and up to 0.50 one only real to the working
    /// precision was a certified 0) away from 0.
    fn delta_error(&self) -> NodeBound {
        if exactly_real(self.z(), self.err)
            && (self.err.is_exact() || !contains_zero(self.z(), self.err.joint()))
        {
            NodeBound::Final(Bound::EXACT)
        } else {
            NodeBound::Final(Bound::UNKNOWN)
        }
    }

    /// The propagated bound of `sign` (or `heaviside`) at this argument (see
    /// [`node_error`]).
    fn sign_error(&self, heaviside: bool) -> NodeBound {
        let (z, b) = (self.z(), self.err);
        if exactly_real(z, b) {
            if is_exact(b.re) || !part_contains_zero(&z.0, b.re) {
                return NodeBound::Final(Bound::EXACT); // −1, 0 or 1
            }
            return NodeBound::Final(Bound::UNKNOWN);
        }
        // An argument that may be complex: `heaviside` is undefined
        // there, `sign(z) = z/|z|`.
        let j = b.joint();
        if heaviside || contains_zero(z, j) {
            return NodeBound::Final(Bound::UNKNOWN);
        }
        // `|w/|w| − z/|z|| ≤ 2r/|z|` for `|w − z| ≤ r`.
        NodeBound::Propagated(Bound::both(shift(log_ball(j, lg_abs_low(z)), 1.0)))
    }

    /// The propagated bound of `sin`, `cos` (`circular`) or `sinh`, `cosh` at
    /// this argument, whose joint error as the node uses it is `2^j`
    /// ([`trig_arg_error`]; see [`node_error`]).
    fn trig_error(&self, circular: bool, j: ErrExp) -> NodeBound {
        let (z, b) = (self.z(), self.err);
        // The derivative over the ball: `|cos w|, |sin w| ≤ cosh(Im w)`,
        // `|cosh w|, |sinh w| ≤ cosh(Re w)` — exactly 1 for `sin`, `cos`
        // of a real argument, which are Lipschitz with constant 1
        // (before 0.30 `2·max(1, |f(z)|)`, a bit per `sin` of a chain).
        let across = if circular { (&z.1, b.im) } else { (&z.0, b.re) };
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
        NodeBound::Propagated(if exactly_real(z, b) {
            Bound::real(e)
        } else {
            Bound::both(e)
        })
    }

    /// The bound of `sin`, `cos`, `sinh` or `cosh` (`node`, value `value`
    /// at `prec` bits) at an argument that is not exactly real, part by part
    /// ([`analytic_parts`]).  Over the ball, with `B = |Im z| + r_im` and
    /// `A = |Re z| + r_re`: `|Re f′| ≤ cosh B`, `|Im f′| ≤ sinh B` for
    /// `sin` and `cos` (`f′ = cos z = cos a·cosh b − i·sin a·sinh b`, and
    /// `−sin z`); `cosh A`, `sinh A` for `sinh`; `sinh A`, `cosh A` for
    /// `cosh`.  Each part is a product of two correctly rounded factors
    /// (`sin a·cosh b`, …: [`own_roundings`]).  Up to 0.54 both parts
    /// carried `r·cosh B` and the rounding at `|f(z)|`: `cos(1 + 10⁻⁵⁰i)`
    /// printed `0.5403023059`, its imaginary part `−8.4·10⁻⁵¹` lying within
    /// the error of the real part.
    fn trig_parts(&self, node: &ExprNode, value: &Complex, prec: usize) -> Bound {
        let (z, b) = (self.z(), self.err);
        let circular = matches!(node, ExprNode::Sin(_) | ExprNode::Cos(_));
        let (across, r) = if circular { (&z.1, b.im) } else { (&z.0, b.re) };
        let t = part_lg(across).exp2() + if is_exact(r) { 0.0 } else { r.exp2() };
        if !t.is_finite() {
            return Bound::UNKNOWN;
        }
        let (c, s) = (log2_cosh(t), log2_sinh(t));
        let (d_re, d_im) = if matches!(node, ExprNode::Cosh(_)) {
            (s, c)
        } else {
            (c, s)
        };
        let propagated = analytic_parts(d_re, d_im, b.re, b.im);
        if propagated.is_unknown() {
            return Bound::UNKNOWN;
        }
        own_roundings(value, propagated, prec)
    }

    /// The propagated bound of `tan` (`tan`) or `tanh` at this argument,
    /// whose joint error as the node uses it is `2^arg_err`, for a value of
    /// magnitude `2^lv` (see [`node_error`]).
    fn tan_error(&self, tan: bool, arg_err: ErrExp, lv: f64) -> NodeBound {
        let (z, b) = (self.z(), self.err);
        // The first-order bound holds on a ball well inside the
        // distance to the nearest pole, about `1/|tan z|` there: no
        // bound beyond an eighth of it.  Before 0.30 the ball of
        // `tan(π/2·(1 − 10⁻⁸⁰))` at 128 bits (its argument rounded onto
        // the pole) was `±3·10³⁹` around a value of `10³⁹` — the true
        // value is `6.4·10⁷⁹` — and `Piecewise` decided `3.8·10⁵⁰ > tan(…)`
        // on it.
        // (A real `tanh` has no pole and a derivative below 1.)
        let poles = tan || !exactly_real(z, b);
        if poles && !is_exact(arg_err) && arg_err + lv.max(0.0) > -3.0 {
            return NodeBound::Final(Bound::UNKNOWN);
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
        NodeBound::Propagated(if exactly_real(z, b) {
            Bound::real(e)
        } else {
            Bound::both(e)
        })
    }

    /// The bound of `atan2(y, x)` for `y` this ball (see [`node_error`]):
    /// `atan2(y, x) = arg(x + iy)` for real `x`, `y`, whose cut is `y = 0`,
    /// `x < 0`.  Arguments whose imaginary parts are only within their
    /// error of 0 may be complex, where `atan2` is not the real function.
    fn atan2_error<W: Borrow<Complex>>(&self, x: &Ball<W>) -> NodeBound {
        let (yv, ey) = (self.z(), self.err);
        let (xv, ex) = (x.z(), x.err);
        if !exactly_real(yv, ey) || !exactly_real(xv, ex) {
            return NodeBound::Final(Bound::UNKNOWN);
        }
        let z = (xv.0.clone(), yv.0.clone());
        let b = Bound {
            re: ex.re,
            im: ey.re,
        };
        if b.is_exact() {
            return NodeBound::Propagated(Bound::real(EXACT));
        }
        let j = b.joint();
        if contains_zero(&z, j) || cut_undecided(&z, b, Cut::NegativeReal) {
            return NodeBound::Final(Bound::UNKNOWN);
        }
        NodeBound::Propagated(Bound::real(log_ball(j, lg_abs_low(&z))))
    }

    /// The bound of `floor` (`floor`) or `ceiling` at this argument, whose
    /// value is `value` (see [`node_error`]): exact, unless the argument may
    /// be complex or is within its error of the integer it rounded to or of
    /// the next one.
    fn floor_error(&self, floor: bool, value: &Complex, prec: usize) -> NodeBound {
        let (z, b) = (self.z(), self.err);
        if !exactly_real(z, b) {
            return NodeBound::Final(Bound::UNKNOWN); // `floor` of a value that may be complex
        }
        if is_exact(b.re) {
            return NodeBound::Final(Bound::EXACT);
        }
        // Distance from the argument to the integer it rounded to, and to
        // the next one.
        let p = 64 + prec;
        let d1 = z.0.sub(&value.0, p, RM);
        let one = BigFloat::from_i32(1, p);
        let next = if floor {
            value.0.add(&one, p, RM)
        } else {
            value.0.sub(&one, p, RM)
        };
        let d2 = next.sub(&z.0, p, RM);
        if part_contains_zero(&d1, b.re) || part_contains_zero(&d2, b.re) {
            return NodeBound::Final(Bound::UNKNOWN);
        }
        NodeBound::Final(Bound::EXACT) // an integer
    }

    /// The bound of `min` or `max` of the arguments `args` (see
    /// [`node_error`]): a copy of one of them, so the largest of their
    /// bounds; none for an argument that may be complex.
    fn extremum_error(args: &[Ball<Z>]) -> NodeBound {
        let mut worst = EXACT;
        for x in args {
            if !exactly_real(x.z(), x.err) {
                return NodeBound::Final(Bound::UNKNOWN);
            }
            worst = worst.max(x.err.re);
        }
        NodeBound::Final(Bound::real(worst))
    }

    /// The bound of `KroneckerDelta(i, j)` for `i` this ball (see
    /// [`node_error`]): a decision on `i − j = 0`, exact when both are exact
    /// or their difference is outside its error box.
    fn kronecker_error<W: Borrow<Complex>>(&self, j: &Ball<W>, prec: usize) -> NodeBound {
        let (iv, ei) = (self.z(), self.err);
        let (jv, ej) = (j.z(), j.err);
        if ei.is_exact() && ej.is_exact() {
            return NodeBound::Final(Bound::EXACT);
        }
        let d = c_sub(iv, jv, prec + 64, RM);
        let re = lsum(ei.re, ej.re);
        let im = lsum(ei.im, ej.im);
        if part_contains_zero(&d.0, re) && part_contains_zero(&d.1, im) {
            return NodeBound::Final(Bound::UNKNOWN);
        }
        NodeBound::Final(Bound::EXACT)
    }

    /// The bound of a special function of the arguments `args` whose value
    /// is `value`, from `worst`, the propagated error of its real routine
    /// (`sensitivity::special_error`; see [`node_error`]).  On arguments
    /// that may be off the real axis the bound holds where the function is
    /// analytic (`analytic`), not across its cut, where the side is
    /// undecidable.
    fn special_error(
        args: &[Ball<Z>],
        worst: ErrExp,
        value: &Complex,
        analytic: impl FnOnce() -> Analytic,
    ) -> NodeBound {
        if is_unknown(worst) {
            return NodeBound::Final(Bound::UNKNOWN);
        }
        let all_real = args.iter().all(|x| exactly_real(x.z(), x.err));
        if all_real && value.1.is_zero() {
            return NodeBound::Propagated(Bound::real(worst));
        }
        if all_real {
            return NodeBound::Propagated(Bound::both(worst));
        }
        match analytic() {
            Analytic::Everywhere => NodeBound::Propagated(Bound::both(worst)),
            Analytic::OffCut(i, cut) => {
                let Some(x) = args.get(i) else {
                    return NodeBound::Final(Bound::UNKNOWN);
                };
                if !exactly_real(x.z(), x.err) && meets_cut(&x.z().0, x.err.re, cut) {
                    return NodeBound::Final(Bound::UNKNOWN);
                }
                NodeBound::Propagated(Bound::both(worst))
            }
            Analytic::Unknown => NodeBound::Final(Bound::UNKNOWN),
        }
    }

    /// The propagated bound of `b^x` for the base `b ± eb` (this ball) and the
    /// exponent `x ± ex` (`x_literal`: its exact value when it is a rational
    /// literal), `lv` bounding the result's magnitude (`log₂`).
    ///
    /// A non-integer power has the cut of `ln` on the negative real axis: an
    /// undecidable side has no bound.  The result is real (exact imaginary
    /// part) for an exactly real base and exponent when the exponent is an
    /// integer or the base is positive; `(−r)^(n/2) = ±i·r^(n/2)` (the half-
    /// integer power of an exactly real negative base, evaluated exactly so)
    /// has an exact real part.
    pub(super) fn power_error<W: Borrow<Complex>>(
        &self,
        exp: &Ball<W>,
        x_literal: Option<&Q>,
        lv: f64,
    ) -> Bound {
        self.power_bound(exp, x_literal, lv, Bound::EXACT)
    }

    /// [`power_error`](Self::power_error) of the power `evalf` computes at
    /// `prec` bits: plus the error of its own route ([`pow_rounding`]).
    fn evaluated_power_error<W: Borrow<Complex>>(
        &self,
        exp: &Ball<W>,
        x_literal: Option<&Q>,
        lv: f64,
        prec: usize,
    ) -> Bound {
        let r = pow_rounding(self.z(), exp.z(), x_literal, prec);
        let own = Bound {
            re: shift(r.re, lv),
            im: shift(r.im, lv),
        };
        self.power_bound(exp, x_literal, lv, own)
    }

    /// The bound of `b^x` with `own` (absolute, `log₂`, per part) added to
    /// the propagated error.
    fn power_bound<W: Borrow<Complex>>(
        &self,
        exp: &Ball<W>,
        x_literal: Option<&Q>,
        lv: f64,
        own: Bound,
    ) -> Bound {
        let (b, eb) = (self.z(), self.err);
        let (x, ex) = (exp.z(), exp.err);
        let integer_exponent = match x_literal {
            Some(q) => q.is_integer(),
            None => ex.is_exact() && x.1.is_zero() && is_integer_value(&x.0),
        };
        if !integer_exponent
            && !contains_zero(b, eb.joint())
            && cut_undecided(b, eb, Cut::NegativeReal)
        {
            return Bound::UNKNOWN;
        }
        let e = pow_error(x_literal, b, eb.joint(), x, ex.joint(), lv);
        let (e_re, e_im) = (lsum(e, own.re), lsum(e, own.im));
        let base_real = exactly_real(b, eb);
        let base_sign = if part_contains_zero(&b.0, eb.re) {
            0
        } else {
            sign_of(&b.0)
        };
        if base_real && exactly_real(x, ex) && (integer_exponent || base_sign > 0) {
            Bound::real(e_re)
        } else if base_real
            && base_sign < 0
            && x_literal.is_some_and(|q| *q.denom() == BigInt::from(2))
        {
            Bound {
                re: EXACT,
                im: e_im,
            }
        } else {
            Bound { re: e_re, im: e_im }
        }
    }

    /// The propagated bound of an inverse trigonometric or hyperbolic function
    /// (node `node`) at this argument, at working precision `prec`: see the
    /// module documentation.  `|f′(z)| = |(z − p₁)(z − p₂)|^(−k)` with the branch
    /// points `p₁, p₂` (`k = ½` for `asin`, `acos`, `asinh`, `acosh`, 1 for
    /// `atan`, `atanh`), computed from the distances at high precision — the
    /// 64-bit `1 − z²` before 0.29 cancelled to 0 for `z` within `2⁻⁶⁴` of ±1
    /// and `acos(1 + 5·10⁻⁶¹)` was refused.
    fn inverse_error(&self, node: &ExprNode, prec: usize) -> Bound {
        use BranchPoint::{Log, Root, RootOnCut};
        let (z, b) = (self.z(), self.err);
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
                ExprNode::Asin(_) | ExprNode::Acos(_) | ExprNode::Atanh(_) => {
                    within(&z.0, b.re, -1, 1)
                }
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
            let d = c_sub(z, &p, hp, RM);
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
    cache: Cache<'_>,
    prec: usize,
    special: impl FnOnce() -> ErrExp,
) -> Bound {
    if !is_finite(value) {
        return Bound::UNKNOWN;
    }
    // The child's value and bound (a view of the cached value).
    let child = |c: ExprId| cache.ball(&c);
    // The child's ball, or an early `UNKNOWN`.
    macro_rules! known_ball {
        ($c:expr) => {
            match child($c) {
                Some(x) if !x.err.is_unknown() => x,
                _ => return Bound::UNKNOWN,
            }
        };
    }
    // The balls of the children, or an early `UNKNOWN`.
    macro_rules! known_balls {
        ($children:expr) => {{
            let children = $children;
            let mut balls = Vec::with_capacity(children.len());
            for &c in children.iter() {
                balls.push(known_ball!(c));
            }
            balls
        }};
    }
    // A rule's final bound returns; a propagated one gets the rounding.
    macro_rules! settle {
        ($rule:expr) => {
            match $rule {
                NodeBound::Final(b) => return b,
                NodeBound::Propagated(p) => p,
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
        // (`eval_ball`); recomputed here for completeness.
        ExprNode::Add(children) | ExprNode::Mul(children) => {
            let mut parts: Vec<(&Complex, Bound)> = Vec::with_capacity(children.len());
            for &c in children.iter() {
                match child(c) {
                    Some(x) => parts.push((x.mid, x.err)),
                    None => return Bound::UNKNOWN,
                }
            }
            return if matches!(node, ExprNode::Add(_)) {
                add_with_bound(&parts, prec).err
            } else {
                mul_with_bound(&parts, prec).err
            };
        }
        // Exact operations: the child's bound, no rounding.
        ExprNode::Neg(c) | ExprNode::Conjugate(c) => {
            return child(*c).map_or(Bound::UNKNOWN, |x| x.err);
        }
        ExprNode::Re(c) => return child(*c).map_or(Bound::UNKNOWN, |x| Bound::real(x.err.re)),
        ExprNode::Im(c) => return child(*c).map_or(Bound::UNKNOWN, |x| Bound::real(x.err.im)),
        ExprNode::Pow(base, exp) => {
            let (b, x) = (known_ball!(*base), known_ball!(*exp));
            if let Some(bound) = power_parts(&b, &x, arena.as_num(*exp), value, prec) {
                return bound;
            }
            b.evaluated_power_error(&x, arena.as_num(*exp), lv, prec)
        }

        // `d exp z = exp z · dz`: the argument's absolute error is the
        // result's relative error.  Before 0.29 the magnitude was clamped
        // at 1 (`ub_out.max(0)`), so `exp(−260.1) ≈ 2⁻³⁷⁵` carried the
        // absolute error of its argument, `2^(9 − prec)`: at the 384-bit cap
        // that ball contained 0, `ln(exp(−260.1))` had no bound and was
        // refused.
        ExprNode::Exp(c) => settle!(known_ball!(*c).exp_error(lv, value, prec)),
        ExprNode::Ln(c) | ExprNode::Arg(c) => {
            settle!(known_ball!(*c).ln_error(matches!(node, ExprNode::Arg(_)), value, prec))
        }
        ExprNode::Sin(c) | ExprNode::Cos(c) | ExprNode::Sinh(c) | ExprNode::Cosh(c) => {
            let x = known_ball!(*c);
            if !exactly_real(x.z(), x.err) {
                return x.trig_parts(node, value, prec);
            }
            let j = trig_arg_error(arena, node, *c, x.z(), x.err, prec);
            let circular = matches!(node, ExprNode::Sin(_) | ExprNode::Cos(_));
            settle!(x.trig_error(circular, j))
        }
        ExprNode::Tan(c) | ExprNode::Tanh(c) => {
            let x = known_ball!(*c);
            let arg_err = trig_arg_error(arena, node, *c, x.z(), x.err, prec);
            settle!(x.tan_error(matches!(node, ExprNode::Tan(_)), arg_err, lv))
        }
        ExprNode::Atan(c)
        | ExprNode::Asin(c)
        | ExprNode::Acos(c)
        | ExprNode::Asinh(c)
        | ExprNode::Acosh(c)
        | ExprNode::Atanh(c) => known_ball!(*c).inverse_error(node, prec),
        ExprNode::Abs(c) => settle!(known_ball!(*c).abs_error()),
        // `atan2(y, x) = arg(x + iy)` for real `x`, `y`: the cut is `y = 0`,
        // `x < 0`.  Arguments whose imaginary parts are only within their
        // error of 0 may be complex, where `atan2` is not the real function.
        ExprNode::Atan2(y, x) => settle!(known_ball!(*y).atan2_error(&known_ball!(*x))),

        // Decisions: exact, unless the argument is within its error of the
        // threshold.
        // `δ(z)` is exactly 0 for a `z` certainly not 0 (complex too), and
        // has no bound for a `z` within its error of 0 (at an exact 0 it has
        // no value, refused in `eval_node`).
        ExprNode::DiracDelta(c) => settle!(known_ball!(*c).delta_error()),
        ExprNode::Sign(c) | ExprNode::Heaviside(c) => {
            settle!(known_ball!(*c).sign_error(matches!(node, ExprNode::Heaviside(_))))
        }
        ExprNode::Floor(c) | ExprNode::Ceiling(c) => {
            let floor = matches!(node, ExprNode::Floor(_));
            settle!(known_ball!(*c).floor_error(floor, value, prec))
        }
        ExprNode::Min(children) | ExprNode::Max(children) => {
            settle!(Ball::extremum_error(&known_balls!(children)))
        }
        ExprNode::KroneckerDelta(i, j) => {
            settle!(known_ball!(*i).kronecker_error(&known_ball!(*j), prec))
        }

        // Special functions: each argument's error times the function's
        // sensitivity to it (`sensitivity.rs`).  Before 0.29 the arguments'
        // relative error carried over plus 4 bits, whatever the condition
        // number: `betainc_regularized(27/11, 1/26562500, 0, 1 − 3·10⁻³⁰)`
        // lost 11 of the 16 digits it certified to the rounding of `x`.
        n if matches!(n, ExprNode::Apply(..))
            || n.func_head().is_some_and(error_by_sensitivity) =>
        {
            let args = known_balls!(node.children());
            let worst = special();
            settle!(Ball::special_error(&args, worst, value, || {
                analyticity(node)
            }))
        }

        // Evaluated with their own bounds by `eval_ball`.
        _ => return Bound::UNKNOWN,
    };
    finish(arena, id, value, cache, propagated, prec)
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
    cache: Cache<'_>,
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
        let real = |c: &ExprId| cache.get(c).is_some_and(|v| exactly_real(v, cache.err(c)));
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
///
/// Nor is `J_ν(x)` of a real order `ν > abs(x) > 0`: it has no zero there
/// (the first positive zero `j_{ν,1}` exceeds `ν`, DLMF 10.21.3, and
/// `J_n(−x) = (−1)ⁿ J_n(x)`).  Before 0.35 a `J_ν` below the range was
/// refused where it arose: `1 + besselj(10⁹, 1)` was refused, truly `1`.
fn underflowed(arena: &Arena, id: ExprId, cache: Cache<'_>) -> bool {
    let nonzero = |c: &ExprId| cache.get(c).is_some_and(|v| mag(v).is_some());
    match arena.node(id) {
        ExprNode::Lib(LibFn::BesselJ, args) => bessel_j_zero_free(args, cache),
        ExprNode::Exp(_) => true,
        ExprNode::Mul(children) => children.iter().all(nonzero),
        ExprNode::Pow(base, _) => nonzero(base),
        ExprNode::Erfc(c) | ExprNode::Gamma(c) | ExprNode::Factorial(c) | ExprNode::Ei(c) => {
            nonzero(c)
        }
        // The variable is the last argument (the order or parameter may be 0).
        ExprNode::Lib(f, args) => {
            matches!(
                f,
                LibFn::AiryAi
                    | LibFn::AiryAiPrime
                    | LibFn::AiryBi
                    | LibFn::AiryBiPrime
                    | LibFn::BesselI
                    | LibFn::BesselK
                    | LibFn::UpperGamma
                    | LibFn::ExpInt
            ) && args.last().is_some_and(nonzero)
        }
        _ => false,
    }
}

/// Do the values of `(ν, x)` lie where `J_ν(x)` has no zero: a real
/// order `ν > abs(x) > 0` with a real `x`, or a negative integer order
/// `−n` with `n > abs(x) > 0` (`J_{−n} = (−1)ⁿ J_n`; see [`underflowed`])?
fn bessel_j_zero_free(args: &[ExprId], cache: Cache<'_>) -> bool {
    let (Some(nu), Some(x)) = (
        args.first().and_then(|c| cache.get(c)),
        args.get(1).and_then(|c| cache.get(c)),
    ) else {
        return false;
    };
    let order_ok = super::bf_strictly_positive(&nu.0) || (nu.0.is_negative() && nu.0.is_int());
    args.len() == 2
        && nu.1.is_zero()
        && x.1.is_zero()
        && is_finite(nu)
        && is_finite(x)
        && order_ok
        && mag(x).is_some()
        && x.0.abs().cmp(&nu.0.abs()).is_some_and(|c| c < 0)
}

/// What is known of a value that came out 0 with an underflow bound
/// ([`is_underflow`]) besides its size: that it is not 0, and its sign when
/// it is real ([`underflow_nonzero`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Nonzero {
    /// A positive real number.
    Positive,
    /// A negative real number.
    Negative,
    /// Not 0; its sign, or its argument, is not known.
    Unsigned,
}

impl Nonzero {
    fn negated(self) -> Nonzero {
        match self {
            Nonzero::Positive => Nonzero::Negative,
            Nonzero::Negative => Nonzero::Positive,
            Nonzero::Unsigned => Nonzero::Unsigned,
        }
    }

    fn times(self, other: Nonzero) -> Nonzero {
        match (self, other) {
            (Nonzero::Unsigned, _) | (_, Nonzero::Unsigned) => Nonzero::Unsigned,
            (a, b) if a == b => Nonzero::Positive,
            _ => Nonzero::Negative,
        }
    }
}

/// Is the value of node `id`, which came out 0 with an underflow bound, a
/// number that is certainly not 0 — below the exponent range, not a zero
/// to the precision reached?  `nonzero` holds the nodes already known to
/// be such numbers; any other child counts as nonzero when its ball
/// excludes 0.
///
/// `exp` is never 0; a product, a power or a quotient of nonzero values is
/// not; nor `erfc`, `Γ` and `x!` of a real argument, `Ei` of a negative
/// one (there it is negative), the functions of [`underflowed`] at exact
/// nonzero arguments (with their sign at a positive real variable), or
/// `sin`, `tan`, `asin`, `atan`, their hyperbolic
/// counterparts and `erf` of a nonzero number that underflowed (they
/// vanish simply at 0, and keep its sign).  A sum is nonzero when its
/// terms are and have one sign.  Anything else — `K₀(10¹⁰) − Ai(10⁷)` — is
/// not known to be nonzero, and is refused (`evalf::undecided_error`).  The
/// values held scaled (`extended`: `exp`, `Ei`, `erfc` and what is built
/// from them) have their own status, decided from the scaled value —
/// `exp(−10¹⁰) − exp(−2·10¹⁰)` is positive, `exp(−10¹⁰)·(sin²1 +
/// cos²1) − exp(−10¹⁰)` a zero to the precision reached.
pub(super) fn underflow_nonzero(
    arena: &Arena,
    id: ExprId,
    cache: Cache<'_>,
    nonzero: &FxHashMap<ExprId, Nonzero>,
) -> Option<Nonzero> {
    let bound = |c: ExprId| cache.err(&c);
    let real = |c: ExprId| cache.get(&c).is_some_and(|v| exactly_real(v, bound(c)));
    let exact = |c: ExprId| bound(c).is_exact();
    let finite = |c: ExprId| cache.get(&c).is_some_and(is_finite) && !bound(c).is_unknown();
    let status = |c: ExprId| -> Option<Nonzero> {
        if let Some(&s) = nonzero.get(&c) {
            return Some(s);
        }
        let (v, b) = (cache.get(&c)?, bound(c));
        if b.is_unknown() || !is_finite(v) || contains_zero(v, b.joint()) {
            return None;
        }
        Some(match (exactly_real(v, b), sign_of(&v.0)) {
            (true, 1) => Nonzero::Positive,
            (true, _) => Nonzero::Negative,
            (false, _) => Nonzero::Unsigned,
        })
    };
    // `Positive` and `Negative` are real signs: a child's is its value's
    // when that is exactly real, or what `nonzero` recorded.
    let real_sign = |c: ExprId| status(c).filter(|s| *s != Nonzero::Unsigned);
    match arena.node(id) {
        ExprNode::Exp(c) => finite(*c).then(|| {
            if real(*c) {
                Nonzero::Positive
            } else {
                Nonzero::Unsigned
            }
        }),
        ExprNode::Mul(children) => children
            .iter()
            .try_fold(Nonzero::Positive, |acc, &c| Some(acc.times(status(c)?))),
        ExprNode::Pow(base, exp) => {
            let s = status(*base)?;
            if !finite(*exp) {
                return None;
            }
            let odd = arena
                .as_num(*exp)
                .filter(|q| q.is_integer())
                .map(|q| !(q.to_integer() % BigInt::from(2)).is_zero());
            Some(match (s, odd) {
                (Nonzero::Positive, _) if real(*exp) => Nonzero::Positive,
                (s, Some(true)) => s,
                (Nonzero::Positive | Nonzero::Negative, Some(false)) => Nonzero::Positive,
                _ => Nonzero::Unsigned,
            })
        }
        ExprNode::Neg(c) => status(*c).map(Nonzero::negated),
        ExprNode::Conjugate(c) => status(*c),
        ExprNode::Abs(c) => status(*c).map(|_| Nonzero::Positive),
        ExprNode::Add(children) => {
            let first = real_sign(*children.first()?)?;
            (first != Nonzero::Unsigned && children.iter().all(|&c| real_sign(c) == Some(first)))
                .then_some(first)
        }
        ExprNode::Sin(c)
        | ExprNode::Tan(c)
        | ExprNode::Sinh(c)
        | ExprNode::Tanh(c)
        | ExprNode::Asin(c)
        | ExprNode::Atan(c)
        | ExprNode::Asinh(c)
        | ExprNode::Atanh(c)
        | ExprNode::Erf(c) => nonzero.get(c).copied(),
        ExprNode::Erfc(c) => {
            if (real(*c) || real_sign(*c).is_some()) && finite(*c) {
                Some(Nonzero::Positive)
            } else {
                (exact(*c) && status(*c).is_some()).then_some(Nonzero::Unsigned)
            }
        }
        ExprNode::Gamma(c) | ExprNode::Factorial(c) => finite(*c).then_some(Nonzero::Unsigned),
        ExprNode::Ei(c) => match real_sign(*c) {
            Some(Nonzero::Negative) => Some(Nonzero::Negative),
            _ => (exact(*c) && status(*c).is_some()).then_some(Nonzero::Unsigned),
        },
        ExprNode::Lib(f, args) => {
            if !underflowed(arena, id, cache) || !args.iter().all(|&c| exact(c)) {
                return None;
            }
            // At a positive real variable and a real order or parameter:
            // `Ai`, `K_ν`, `Γ(s, x)`, `E_ν(x)` are positive, `Ai′` is
            // negative, `I_ν` (`ν ≥ 0`) is positive.
            let positive_x = args
                .last()
                .is_some_and(|&c| real_sign(c) == Some(Nonzero::Positive));
            let real_params = args.iter().all(|&c| real(c));
            let nonnegative_order = args
                .first()
                .is_some_and(|&c| cache.get(&c).is_some_and(|v| sign_of(&v.0) >= 0) && real(c));
            Some(match f {
                LibFn::AiryAi | LibFn::BesselK | LibFn::UpperGamma | LibFn::ExpInt
                    if positive_x && real_params =>
                {
                    Nonzero::Positive
                }
                // `J_ν(x) > 0` for `0 < x < ν` (no zero there, and
                // `(x/2)^ν/Γ(ν + 1) > 0` near 0); `J_n(−x) = (−1)ⁿ J_n(x)`
                // and `J_{−n}(x) = (−1)ⁿ J_n(x)`: for an integer order the
                // sign `(−1)ⁿ` once for a negative `x`, once for a negative
                // order.
                LibFn::BesselJ if real_params => {
                    let order = args.first().and_then(|c| cache.get(c));
                    match order {
                        Some(n) if n.0.is_int() => {
                            let flips = usize::from(!positive_x) + usize::from(n.0.is_negative());
                            if flips % 2 == 1 && super::bessel_order::is_odd(&n.0) {
                                Nonzero::Negative
                            } else {
                                Nonzero::Positive
                            }
                        }
                        Some(_) if positive_x => Nonzero::Positive,
                        _ => Nonzero::Unsigned,
                    }
                }
                LibFn::AiryAiPrime if positive_x => Nonzero::Negative,
                LibFn::BesselI if positive_x && real_params && nonnegative_order => {
                    Nonzero::Positive
                }
                _ => Nonzero::Unsigned,
            })
        }
        _ => None,
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

/// `log₂` of the relative error that `evalf`'s own route for `b^x` (centre
/// values; `x_literal` the exponent's rational literal) adds at `prec` bits
/// beyond the rounding of the result:
///
/// * none for the correctly rounded routes: `√b` of a non-negative real,
///   an integer literal exponent (`powi`; a complex base by repeated
///   multiplication), a real integer exponent below `10⁹` of a positive
///   base;
/// * `|x·ln b|·2^(2 − 32 − prec)` for `exp(x·ln b)` of a positive real base
///   and a real exponent (and `(−r)^(n/2)`, taken from `r^(n/2)`), formed
///   with 32 guard bits;
/// * `|x|·(|ln|b|| + |arg b|)·2^(3 − prec)` for the general route
///   `exp(x·Log b)` at `prec` bits: the roundings of `Log b` (`π` for a
///   negative base) and of the product, which exact `b` and `x` do not
///   avoid.  Up to 0.51 that route counted as exact for an exact base and
///   exponent, and `(−1)^(8 + 2i)` (`= e^(−2π)`) kept the rounding residue
///   of `8π` as a certified imaginary part: it was called non-real.  The
///   imaginary part alone stays exact where the route cannot make one: for
///   `|b| = 1` and `Re x = 0` exactly, `Im(x·Log b) = Re x·arg b + Im x·ln|b|`
///   is an exact 0 (`(−1)^i = e^(−π)`, `i^i = e^(−π/2)` are real).
///
/// Since 0.55 the routes with a complex value (a non-real base to an
/// integer power, `exp(x·Log b)`) are bounded part by part along the
/// route ([`power_parts`]); this serves the routes with a real value (and
/// one exact part) that [`Ball::power_error`] bounds.
fn pow_rounding(b: &Complex, x: &Complex, x_literal: Option<&Q>, prec: usize) -> Bound {
    if mag(b).is_none() || mag(x).is_none() {
        return Bound::EXACT; // `0^x` and `b^0`: exact, or refused, in `eval_node`.
    }
    let b_real = b.1.is_zero();
    let b_sign = sign_of(&b.0);
    if let Some(q) = x_literal {
        let small_integer = q.is_integer() && q.to_integer().to_i32().is_some();
        let square_root = b_real && b_sign >= 0 && *q == Q::new(BigInt::one(), BigInt::from(2));
        if small_integer || square_root {
            return Bound::EXACT;
        }
    }
    let ln_abs = lg_abs(b).abs().max(lg_abs_low(b).abs()) * std::f64::consts::LN_2;
    let lx = lg_abs(x);
    let x_real = x.1.is_zero();
    let half_integer = x_literal.is_some_and(|q| *q.denom() == BigInt::from(2));
    let (lw, guard) = if b_real && ((x_real && b_sign > 0) || (b_sign < 0 && half_integer)) {
        let powi = x_real
            && x.0.is_int()
            && crate::base::bigfloat::to_f64(&x.0).is_some_and(|v| v.abs() < 1.0e9);
        if powi {
            return Bound::EXACT;
        }
        (lx + ln_abs.log2(), 2.0 - 32.0)
    } else {
        let arg = if b_real && b_sign > 0 {
            0.0
        } else {
            std::f64::consts::PI
        };
        (lx + (ln_abs + arg).log2(), 3.0)
    };
    let e = lw + guard - prec as f64;
    let e = if e.is_finite() { up(e) } else { EXACT };
    // `b ∈ {±1, ±i}`: `|b|` and `ln|b|` are computed exactly (1 and 0).
    let one = BigFloat::from_i32(1, prec);
    let unit = |v: &BigFloat| v.abs().cmp(&one) == Some(0);
    let unit_modulus = (b.1.is_zero() && unit(&b.0)) || (b.0.is_zero() && unit(&b.1));
    let im_exact = unit_modulus && x.0.is_zero();
    Bound {
        re: e,
        im: if im_exact { EXACT } else { e },
    }
}

/// A complex value as a route of `evalf` computes it, for bounding the
/// route part by part: upper bounds (`log₂`, `−∞` for an exact 0) on the
/// magnitudes of the computed parts, and the error of each computed part
/// against the true value (the propagated error of the inputs and the
/// roundings so far).
#[derive(Clone, Copy, Debug)]
struct Parts {
    /// `log₂` of an upper bound on `|re|` of the computed value.
    re: f64,
    /// `log₂` of an upper bound on `|im|` of the computed value.
    im: f64,
    /// The error of each part.
    e: Bound,
}

impl Parts {
    /// The exact `1` (`c_powi` starts from it).
    const ONE: Parts = Parts {
        re: 0.0,
        im: f64::NEG_INFINITY,
        e: Bound::EXACT,
    };

    /// The computed value `z` with the error `e`; `None` for a value that
    /// is not finite or an error that is unknown.
    fn of(z: &Complex, e: Bound) -> Option<Parts> {
        let (re, im) = (part_lg(&z.0), part_lg(&z.1));
        (re < f64::INFINITY && im < f64::INFINITY && !e.is_unknown()).then_some(Parts { re, im, e })
    }

    /// `log₂` bounds on the parts of the true value.
    fn true_re(&self) -> f64 {
        lsum(self.re, self.e.re)
    }

    /// See [`Parts::true_re`].
    fn true_im(&self) -> f64 {
        lsum(self.im, self.e.im)
    }

    /// The product `self·v` as `c_mul` forms it at `prec` bits, `(ac −
    /// bd) + (ad + bc)i` with each of the four products and the two sums
    /// rounded.  With computed `u = t_u + E_u`, `v = t_v + E_v`, the
    /// difference `u·v − t_u·t_v = u·E_v + E_u·t_v` is bounded part by
    /// part, and the roundings add `2^(1 − prec)·(|ac| + |bd|)` to the
    /// real part (two products and the difference, each within half a
    /// unit; `2^(2 − prec)` with a margin) and likewise `(|ad| + |bc|)` to
    /// the imaginary part.  A part both of whose products have an exact
    /// zero factor is an exact 0.
    fn mul(&self, v: &Parts, prec: usize) -> Parts {
        let u = self;
        if u.e.is_unknown() || v.e.is_unknown() {
            return Parts {
                e: Bound::UNKNOWN,
                ..*u
            };
        }
        let rnd = 2.0 - prec as f64;
        let re_abs = lsum(u.re + v.re, u.im + v.im);
        let im_abs = lsum(u.re + v.im, u.im + v.re);
        let (tre, tim) = (v.true_re(), v.true_im());
        let e_re = lsum(
            lsum(u.re + v.e.re, u.im + v.e.im),
            lsum(u.e.re + tre, u.e.im + tim),
        );
        let e_im = lsum(
            lsum(u.re + v.e.im, u.im + v.e.re),
            lsum(u.e.re + tim, u.e.im + tre),
        );
        let mag = |m: f64| if m == f64::NEG_INFINITY { m } else { up(m) };
        Parts {
            re: mag(re_abs),
            im: mag(im_abs),
            e: Bound {
                re: lsum(e_re, re_abs + rnd),
                im: lsum(e_im, im_abs + rnd),
            },
        }
    }

    /// The bound of `1/p` for `p` this value, as `c_div(1, p)` forms it:
    /// `(re p, −im p)/(re² p + im² p)`, each part within `2^(3 − prec)` of
    /// itself, and the error of `p` carried by `(1/w)′ = −1/w²`
    /// ([`analytic_parts`]: `|Re w⁻²| ≤ |w|⁻²`, `|Im w⁻²| = 2|w_re||w_im|·
    /// |w|⁻⁴` over the ball).  `|p|` is read from the value `1/p` (`value`).
    fn recip_bound(&self, value: &Complex, prec: usize) -> Bound {
        let r = self.e.joint();
        // `log₂|p| ≥ −log₂|value|`, less the rounding of the quotient (far
        // below the margin of `down`).
        let lp = down(-lg_abs(value));
        if is_unknown(r) || !lp.is_finite() {
            return Bound::UNKNOWN;
        }
        let d = if is_exact(r) {
            lp
        } else if r < lp - 1.0 {
            down(lp + (-(r - lp).exp2()).ln_1p() * std::f64::consts::LOG2_E)
        } else {
            return Bound::UNKNOWN;
        };
        let s_re = -2.0 * d;
        let s_im = 1.0 + self.true_re() + self.true_im() - 4.0 * d;
        let p = analytic_parts(s_re, s_im, self.e.re, self.e.im);
        let part = |x: &BigFloat, e: ErrExp| -> ErrExp {
            if !x.is_zero() {
                lsum(e, part_lg(x) + 3.0 - prec as f64)
            } else if is_exact(e) {
                EXACT
            } else {
                lsum(e, rounding(value, prec))
            }
        };
        Bound {
            re: part(&value.0, p.re),
            im: part(&value.1, p.im),
        }
    }
}

/// The bound of `b^x` (value `value` at `prec` bits) part by part, for the
/// two routes of `evalf`'s power (`eval_node`) whose value is complex: a
/// non-real base to a literal integer power in `i32` (binary powering,
/// `c_powi`, and `1/bⁿ` by `c_div` for a negative one: [`powi_parts`]) and
/// the general `exp(x·Log b)` ([`general_power_parts`]).  `None` for the
/// other routes, whose bound [`Ball::power_error`] gives with an exact
/// part where the route has one (`√b` and integer powers of a real base,
/// a positive real base to a real power, a negative real base to an
/// integer or half-integer power), and for `0^x`, `b^0` and a value that
/// came out 0 (an underflow, [`finish`]).
///
/// Up to 0.54 these routes had one bound for both parts, the rounding at
/// the magnitude of the value and `|x|·(|ln|b|| + π)·2^(3 − prec)` for
/// `exp(x·Log b)` of every non-real base: `(1 + 10⁻⁴⁵i)²`, `(1 +
/// 10⁻⁴⁵i)^(1/3)` and `2^(10⁻⁴⁵i)` printed `1` at 10 digits.
fn power_parts<Z: Borrow<Complex>, W: Borrow<Complex>>(
    base: &Ball<Z>,
    exp: &Ball<W>,
    x_literal: Option<&Q>,
    value: &Complex,
    prec: usize,
) -> Option<Bound> {
    let (b, eb) = (base.z(), base.err);
    let (x, ex) = (exp.z(), exp.err);
    if mag(value).is_none() || (mag(b).is_none()) || mag(x).is_none() {
        return None;
    }
    let (b_real, x_real) = (b.1.is_zero(), x.1.is_zero());
    let half = Q::new(BigInt::one(), BigInt::from(2));
    if b_real && !b.0.is_negative() && x_literal == Some(&half) {
        return None;
    }
    if let Some(n) = x_literal
        .filter(|q| q.is_integer())
        .and_then(|q| q.to_integer().to_i32())
    {
        return (!b_real).then(|| powi_parts(b, eb, n, value, prec));
    }
    if b_real && x_real && !b.0.is_negative() {
        return None;
    }
    if b_real && b.0.is_negative() {
        let integer = x_real && x.0.is_int() && ex.is_exact();
        let half_integer = x_literal.is_some_and(|q| *q.denom() == BigInt::from(2));
        if integer || half_integer {
            return None;
        }
    }
    Some(general_power_parts(b, eb, x, ex, x_literal, value, prec))
}

/// The bound of `bⁿ` for a non-real base `b ± eb` and `n` in `i32`, as
/// `c_powi` forms it by repeated squaring (and `c_div(1, b^|n|)` for `n <
/// 0`), every product bounded by [`Parts::mul`].  Next to the real axis no
/// part cancels and each keeps its own magnitude: `Im (1 + εi)ⁿ ≈ nε`.
fn powi_parts(b: &Complex, eb: Bound, n: i32, value: &Complex, prec: usize) -> Bound {
    if n == 0 {
        return Bound::EXACT;
    }
    let Some(base) = Parts::of(b, eb) else {
        return Bound::UNKNOWN;
    };
    let mut result = Parts::ONE;
    let mut square = base;
    let mut k = n.unsigned_abs();
    loop {
        if k & 1 == 1 {
            result = result.mul(&square, prec);
        }
        k >>= 1;
        if k == 0 {
            break;
        }
        square = square.mul(&square, prec);
    }
    if result.e.is_unknown() {
        Bound::UNKNOWN
    } else if n > 0 {
        result.e
    } else {
        result.recip_bound(value, prec)
    }
}

/// The bound of `b^x = exp(x·Log b)` as `eval_node` forms it, part by
/// part: `Log b = ln c_abs(b) + i·atan2(im b, re b)` (`c_abs` within
/// `2^(1 − prec)` of `|b|`, an absolute error of the logarithm, exact for
/// `b ∈ {±1, ±i}`; `atan2` within four units of itself, exactly 0 for a
/// positive real base), the base's error carried by `Log′ = 1/w`
/// ([`analytic_parts`]); the product `x·Log b` ([`Parts::mul`]); and `exp`
/// of it as [`Ball::exp_error`] bounds it (exactly real when `x·Log b`
/// is: `(−1)^i = e^(−π)`).  A base whose ball meets the cut of `Log`
/// with an undecidable side has no bound unless the exponent is an exact
/// integer (where the power is continuous across it), nor does one that
/// reaches 0.
fn general_power_parts(
    b: &Complex,
    eb: Bound,
    x: &Complex,
    ex: Bound,
    x_literal: Option<&Q>,
    value: &Complex,
    prec: usize,
) -> Bound {
    let r = eb.joint();
    if is_unknown(r) || ex.is_unknown() {
        return Bound::UNKNOWN;
    }
    let integer_exponent = match x_literal {
        Some(q) => q.is_integer(),
        None => ex.is_exact() && x.1.is_zero() && is_integer_value(&x.0),
    };
    if !integer_exponent && cut_undecided(b, eb, Cut::NegativeReal) {
        return Bound::UNKNOWN;
    }
    let (lb, lb_low) = (lg_abs(b), lg_abs_low(b));
    // `log₂` of a lower bound on `|w|` over the ball.
    let d = if is_exact(r) {
        lb_low
    } else if r < lb_low - 1.0 {
        down(lb_low + (-(r - lb_low).exp2()).ln_1p() * std::f64::consts::LOG2_E)
    } else {
        return Bound::UNKNOWN;
    };
    // `Re(1/w) = w_re/|w|²`, `Im(1/w) = −w_im/|w|²`.
    let s_re = lsum(part_lg(&b.0), eb.re) - 2.0 * d;
    let s_im = lsum(part_lg(&b.1), eb.im) - 2.0 * d;
    let propagated = analytic_parts(s_re, s_im, eb.re, eb.im);
    let one = BigFloat::from_i32(1, 64);
    let is_one = |v: &BigFloat| v.abs().cmp(&one) == Some(0);
    let unit = (b.1.is_zero() && is_one(&b.0)) || (b.0.is_zero() && is_one(&b.1));
    let lg_pi = up(std::f64::consts::PI.log2());
    // `|ln|b||` and `|arg b|` (`≤ |im b|/re b` in the right half-plane).
    let l_re = if unit {
        f64::NEG_INFINITY
    } else {
        up((lb.abs().max(lb_low.abs()) * std::f64::consts::LN_2).log2())
    };
    let l_im = if b.1.is_zero() {
        if sign_of(&b.0) > 0 {
            f64::NEG_INFINITY
        } else {
            lg_pi
        }
    } else if sign_of(&b.0) > 0 {
        (part_lg(&b.1) - part_lg_low(&b.0)).min(lg_pi - 1.0)
    } else {
        lg_pi
    };
    let route_re = if unit {
        EXACT
    } else {
        lsum(1.5 - prec as f64, l_re + 1.0 - prec as f64)
    };
    let route_im = if l_im == f64::NEG_INFINITY {
        EXACT
    } else {
        l_im + 2.0 - prec as f64
    };
    let log = Parts {
        re: l_re,
        im: l_im,
        e: Bound {
            re: lsum(propagated.re, route_re),
            im: lsum(propagated.im, route_im),
        },
    };
    let Some(xp) = Parts::of(x, ex) else {
        return Bound::UNKNOWN;
    };
    let w = xp.mul(&log, prec);
    if w.e.is_unknown() {
        return Bound::UNKNOWN;
    }
    // `exp` of `w`: `|e^(w+δ) − e^w| ≤ |e^w|·(e^r − 1)` (see
    // [`Ball::exp_error`]).
    let rw = w.e.joint();
    let lv = lg_abs(value);
    let g = if is_exact(rw) {
        EXACT
    } else if rw < -30.0 {
        shift(rw, up(lv + rw.exp2() * std::f64::consts::LOG2_E))
    } else if rw <= 9.0 {
        clamp(up(lv + rw.exp2().exp_m1().log2()))
    } else {
        return Bound::UNKNOWN;
    };
    if w.im == f64::NEG_INFINITY && is_exact(w.e.im) {
        // An exactly real exponent of `e`: `c_exp` multiplies by `sin 0`.
        return Bound {
            re: lsum(g, part_rounding(&value.0, prec)),
            im: EXACT,
        };
    }
    let slack = lsum(g, rounding(value, prec));
    let d_re = lsum(part_lg(&value.0), slack);
    let d_im = lsum(part_lg(&value.1), slack);
    own_roundings(value, analytic_parts(d_re, d_im, w.e.re, w.e.im), prec)
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

/// `log₂ sinh t` for `t ≥ 0`, rounded up (`−∞` at 0, `+∞` beyond
/// `f64`): `sinh t = e^t·(1 − e^(−2t))/2`.
fn log2_sinh(t: f64) -> f64 {
    if !t.is_finite() {
        return f64::INFINITY;
    }
    let t = t.abs();
    // `t` is a sum of magnitudes, 0 only for an exact 0 part and radius.
    if t <= 0.0 {
        return f64::NEG_INFINITY;
    }
    up(t * std::f64::consts::LOG2_E - 1.0 + (-(-2.0 * t).exp_m1()).log2())
}

/// The propagated bound of an analytic function `f` part by part, at an
/// argument whose real and imaginary parts have the errors `2^rx` and
/// `2^ry`, from bounds `2^d_re ≥ |Re f′|` and `2^d_im ≥ |Im f′|` over the
/// ball.  By Cauchy–Riemann (`∂f/∂y = i·f′`) a change `δ = δx + i·δy`
/// of the argument changes `f` by `∫ f′·δ`, whose real part is `Re f′·δx
/// − Im f′·δy` and imaginary part `Im f′·δx + Re f′·δy`:
///
/// `|ΔRe f| ≤ |Re f′|·rx + |Im f′|·ry`,  `|ΔIm f| ≤ |Im f′|·rx + |Re f′|·ry`.
///
/// Next to the axis, where one part of `f` and of `f′` is small, so is
/// that part's error: `sin(1 + εi) = sin 1·cosh ε + i·cos 1·sinh ε` moves
/// its imaginary part by about `ε·rx + ry`, not by the `rx + ry` of a
/// joint bound (up to 0.54), which buried it for `ε` below the precision.
fn analytic_parts(d_re: f64, d_im: f64, rx: ErrExp, ry: ErrExp) -> Bound {
    Bound {
        re: lsum(shift(rx, d_re), shift(ry, d_im)),
        im: lsum(shift(rx, d_im), shift(ry, d_re)),
    }
}

/// `propagated` plus the rounding of each part of `value` at its own
/// magnitude: for a route that computes each part to its own precision
/// (`c_exp`, `c_sin`, …: each part a product of two correctly rounded
/// factors, within two units in the last place of that part).  A part
/// that came out 0 keeps the rounding at the magnitude of the whole value
/// (as [`finish`] rounds every part), since a route may give an exact 0
/// for a residue of that size (`sin_cos` at the rounded `π`), unless its
/// propagated bound is exact.
fn own_roundings(value: &Complex, propagated: Bound, prec: usize) -> Bound {
    let joint = rounding(value, prec);
    let part = |x: &BigFloat, e: ErrExp| -> ErrExp {
        if !x.is_zero() {
            lsum(e, part_rounding(x, prec))
        } else if is_exact(e) {
            EXACT
        } else {
            lsum(e, joint)
        }
    };
    Bound {
        re: part(&value.0, propagated.re),
        im: part(&value.1, propagated.im),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `analyticity` and `node_error` select exactly the functions of their
    /// 0.49 or-patterns (W6d: the lists became `entire_or_meromorphic`,
    /// which took in the library functions of the `Lib` arm's
    /// `Everywhere` group, and `error_by_sensitivity`).
    #[test]
    fn function_predicates_are_the_old_lists() {
        use crate::base::func::{Family, Func::*, selected};
        let entire = [
            Gamma,
            Digamma,
            Polygamma,
            Zeta,
            Erf,
            Erfc,
            Beta,
            Factorial,
            Binomial,
            Si,
            DiracDelta,
            Erfi,
            Shi,
            FresnelS,
            FresnelC,
            AiryAi,
            AiryBi,
            AiryAiPrime,
            AiryBiPrime,
            DirichletEta,
            Legendre,
            ChebyshevT,
            ChebyshevU,
            Hermite,
            Laguerre,
            Gegenbauer,
            Jacobi,
            AssocLaguerre,
        ];
        assert_eq!(
            selected(entire_or_meromorphic),
            entire.into_iter().collect()
        );
        let mut sensitivity: std::collections::HashSet<Func> = [
            Gamma, LogGamma, Digamma, Erf, Erfc, Beta, Si, Ci, Ei, Li, Zeta, Polygamma, Factorial,
            Binomial,
        ]
        .into_iter()
        .collect();
        sensitivity.extend(selected(|f| f.family() != Family::Node));
        assert_eq!(selected(error_by_sensitivity), sensitivity);
    }
}

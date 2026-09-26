//! How much a special function's value moves when its arguments move.
//!
//! A special-function node is evaluated from its children's values, each of
//! which carries an error bound (`accuracy`).  To first order the value's
//! error is
//!
//! ```text
//! Σᵢ |∂f/∂xᵢ| · err(xᵢ)
//! ```
//!
//! over the distinct inexact children; [`special_error`] computes it.  The
//! derivative `|∂f/∂xᵢ|` is the function's sensitivity to that argument, its
//! condition number `|∂ ln f/∂ ln xᵢ|` times `|f|/|xᵢ|`, and it can be huge:
//! `I_x(a, b)` near `x = 1` for a small `b` (`b/(1 − x)`), `Γ` near a pole,
//! `J₀` or `Ci` near a zero (where the *relative* error is unbounded even for
//! a modest derivative).  Before 0.29 the rule was "the arguments' relative
//! error carries over, plus 4 bits", and a rational argument rounded to the
//! working precision gave certified wrong digits:
//! `betainc_regularized(27/11, 1/26562500, 0, 1 − 3·10⁻³⁰)` was
//! `2.5118502017106955·10⁻⁶` in `eval_f64`, truly `2.5118502017194425·10⁻⁶`.
//!
//! Each (function, argument) pair gets one of three treatments ([`Sens`]):
//!
//! * a closed-form upper bound on `|∂f/∂x|` over the argument's error ball,
//!   from the exact distances to the function's singular points, the node's
//!   value, `f64` bounds of `ln Γ` and, for `Y_ν`, the companion `Y_{ν−1}`
//!   at 128 bits — see the table in the `accuracy` module documentation
//!   (the bounds that are not textbook identities were checked against
//!   mpmath over wide grids: largest ratio of the true derivative to the
//!   bound 0.99);
//! * a bound on the change of the value over the ball directly, where the
//!   derivative is unbounded but integrable (the incomplete beta function at
//!   a limit within its error of 0 or 1);
//! * numerically: the function is evaluated again at the working precision
//!   with the argument moved by its error radius (at least a few units in
//!   its last place), and the difference, doubled, is the contribution.  One
//!   more evaluation of the node, for the arguments without a closed form
//!   (shape parameters of Bessel functions, `polylog`'s order, `ζ` left of
//!   0, the orthogonal polynomials' variables, the elliptic integrals of two
//!   arguments).
//!
//! A ball that is not well inside the function's domain of analyticity — its
//! radius more than an eighth of the distance to a pole or branch point —
//! has no bound: the evaluation is repeated at a higher precision, where
//! the ball is smaller.  Exact arguments cost nothing.  These are estimates
//! in the sense of the `accuracy` module (upper bounds of the first-order
//! term, not interval arithmetic), made conservative by a factor of two.

use astro_float::{BigFloat, Consts, RoundingMode};
use rustc_hash::FxHashMap;

use super::accuracy::{self, EXACT, ErrExp, UNKNOWN};
use crate::base::arena::Arena;
use crate::base::bigcomplex::{Complex, c_sub};
use crate::base::libfn::LibFn;
use crate::base::node::{ExprId, ExprNode};

/// Precision of the companion values and auxiliary quantities: only their
/// magnitude matters, but the routines build integers with
/// `BigFloat::from_i128`, which astro-float refuses (NaN) below 128 bits.
const LP: usize = 128;

const LOG2E: f64 = std::f64::consts::LOG2_E;
const LN2: f64 = std::f64::consts::LN_2;

/// `log₂(8/7)`: a quantity like `1/d` grows by at most this over a ball
/// whose radius is an eighth of `d`.
const BALL: f64 = 0.192_645_077_942_396_2;

/// `log₂(1/(1 − 2^x))`: how much `1/d` grows over a ball of radius `2^x·d`
/// (`x = e − log₂ d`), at most [`BALL`] for the widest ball admitted
/// (`x ≤ −3`), 0 for an exact argument.  The worst case `BALL` itself is
/// only for single factors: raised to a power `p` it inflates the bound by
/// `(8/7)^p` whatever the radius (before 0.30 `x^(s−1)` in the sensitivity
/// of `Γ(s, x)` to `x` was taken at `9x/8`: `2^(1.16·10⁶)` for `s ≈ 6.8·10⁶`).
fn shrink(x: f64) -> f64 {
    if x == f64::NEG_INFINITY {
        0.0
    } else {
        -(-x.exp2()).ln_1p() * LOG2E
    }
}

/// `log₂(1 + 2^x)`: how much `d` grows over a ball of radius `2^x·d`.
fn swell(x: f64) -> f64 {
    if x == f64::NEG_INFINITY {
        0.0
    } else {
        x.exp2().ln_1p() * LOG2E
    }
}

/// `log₂(2/√π)`.
const LOG2_TWO_OVER_SQRT_PI: f64 = 0.176_656_383_560_998_7;

/// Euler's constant, rounded up.
const EULER_GAMMA_UP: f64 = 0.577_215_664_901_533;

/// What is known about the sensitivity of a function to one argument.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Sens {
    /// The function does not depend on it within its error (a discrete
    /// parameter within its error of an integer), or its derivative is 0.
    Zero,
    /// `log₂` of an upper bound on `|∂f/∂x|` over the argument's error ball.
    Deriv(f64),
    /// `log₂` of an upper bound on `|f(y) − f(x)|` for every `y` of the ball.
    Change(f64),
    /// Evaluate the function again with the argument moved.
    Numeric,
    /// No bound: the ball reaches a singular point.
    Unbounded,
}

// ── Small helpers ──────────────────────────────────────────────────────────

/// `log₂|x|` (`−∞` for 0, `+∞` for a non-finite value).
fn lg(x: &BigFloat) -> f64 {
    if x.is_zero() {
        return f64::NEG_INFINITY;
    }
    if x.is_nan() || x.is_inf() {
        return f64::INFINITY;
    }
    let e = x.exponent().unwrap_or(0);
    let mut m = x.abs();
    m.set_exponent(0);
    let mf = super::bigfloat_to_f64_rounded(&m, RoundingMode::ToEven).unwrap_or(0.75);
    f64::from(e) + mf.abs().max(0.5).log2()
}

/// `log₂(2^a + 2^b)`.
fn lsum(a: f64, b: f64) -> f64 {
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    if hi == f64::NEG_INFINITY || !hi.is_finite() {
        return hi;
    }
    hi + (lo - hi).exp2().ln_1p() * LOG2E
}

/// `x` as an `f64`, `±∞` beyond the range.
fn f(x: &BigFloat) -> f64 {
    if x.is_zero() {
        return 0.0;
    }
    let l = lg(x);
    if l > 1000.0 {
        return if neg(x) {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }
    super::bigfloat_to_f64_rounded(x, RoundingMode::ToEven).unwrap_or(0.0)
}

/// `x > 0`.  astro-float's `is_positive` is the sign bit alone, true for
/// `+0`: a betainc lower limit of `0` was taken for a positive one, and its
/// shape sensitivity bounded by `|ln 0|` (`ln(betainc_regularized(23 + 1/3,
/// 1/2, 0, 2⁻²⁰))` was `PrecisionExhausted`).
fn pos(x: &BigFloat) -> bool {
    x.is_positive() && !x.is_zero()
}

/// `x < 0` (not `−0`).
fn neg(x: &BigFloat) -> bool {
    x.is_negative() && !x.is_zero()
}

/// `a − b` exactly.  (astro-float's `sub_full_prec` rounds to precision 0
/// when an operand is 0.)
fn diff(a: &BigFloat, b: &BigFloat) -> BigFloat {
    if b.is_zero() {
        a.clone()
    } else if a.is_zero() {
        b.neg()
    } else {
        a.sub_full_prec(b)
    }
}

/// `a + b` exactly.
fn sum(a: &BigFloat, b: &BigFloat) -> BigFloat {
    diff(a, &b.neg())
}

fn int(n: i32) -> BigFloat {
    BigFloat::from_i32(n, LP)
}

/// `1 − x` exactly.
fn one_minus(x: &BigFloat) -> BigFloat {
    diff(&int(1), x)
}

/// `ln(1 + |x|)`.
fn ln1p_abs(x: &BigFloat) -> f64 {
    let a = f(x).abs();
    if a.is_finite() {
        a.ln_1p()
    } else {
        lg(x) * LN2
    }
}

/// Is the ball of radius `2^e` (`e` a log₂) well inside, at most an eighth
/// of the distance `2^ld` to a singular point?
fn inside(ld: f64, e: f64) -> bool {
    ld > e + 3.0
}

/// `log₂ n!` (exact to a fraction of a bit).
fn log2_factorial(n: f64) -> f64 {
    if n < 2.0 {
        return 0.0;
    }
    if n <= 64.0 {
        let mut s = 0.0;
        let mut k = 2.0;
        while k <= n {
            s += f64::log2(k);
            k += 1.0;
        }
        return s;
    }
    // Stirling, with the 1/(12n) term: within 10⁻⁶ bits for n > 64.
    ((n + 0.5) * n.ln() - n + 0.5 * (2.0 * std::f64::consts::PI).ln() + 1.0 / (12.0 * n)) * LOG2E
}

/// `ln Γ(x)` for `x > 0` in `f64`, and a bound on its absolute error: the
/// recurrence up to 16 and Stirling's series with three terms (error below
/// `10⁻⁹`); beyond the `f64` range from the binary exponent (`ln Γ(x) ≈
/// −ln x` for a tiny `x`, `x ln x − x` for a huge one, up to `ln x`).
fn ln_gamma_f64(x: &BigFloat) -> (f64, f64) {
    let xf = f(x);
    if !(1e-300..=1e300).contains(&xf) {
        let ln_x = lg(x) * LN2;
        return if xf < 1.0 {
            (-ln_x, 1.0)
        } else {
            (xf * (ln_x - 1.0), ln_x.abs() + 1.0 + xf * 1e-12)
        };
    }
    let mut z = xf;
    let mut shift = 0.0;
    while z < 16.0 {
        shift += z.ln();
        z += 1.0;
    }
    let z2 = z * z;
    let series = 1.0 / (12.0 * z) - 1.0 / (360.0 * z * z2) + 1.0 / (1260.0 * z * z2 * z2);
    let v = (z - 0.5) * z.ln() - z + 0.5 * (2.0 * std::f64::consts::PI).ln() + series - shift;
    (v, 1e-9 + 1e-14 * (v.abs() + shift.abs() + z * z.ln().abs()))
}

/// Lower and upper bounds on `log₂ B(p, q)` for `p, q > 0`.
fn log2_beta_bounds(p: &BigFloat, q: &BigFloat) -> (f64, f64) {
    let s = sum(p, q);
    let (a, ea) = ln_gamma_f64(p);
    let (b, eb) = ln_gamma_f64(q);
    let (c, ec) = ln_gamma_f64(&s);
    let (v, err) = (a + b - c, ea + eb + ec);
    ((v - err) * LOG2E, (v + err) * LOG2E)
}

/// `log₂` of the distance from `x` to the nearest pole of `Γ` (the
/// non-positive integers); `None` at a pole.
fn pole_distance(x: &BigFloat) -> Option<f64> {
    if pos(x) {
        return Some(lg(x));
    }
    let frac = diff(x, &x.floor());
    let d = frac.clone().min(&one_minus(&frac));
    (!d.is_zero()).then(|| lg(&d))
}

/// `log₂` of an upper bound on `|ψ(y)|` over the ball `x ± 2^e`:
/// `|ψ(y)| ≤ 1/d + ln(1 + |y|) + γ` with `d` the distance to the nearest
/// pole (`ψ(y + 1) ∈ (−γ, ln(y + 1))` for `y > 0`, and the reflection
/// `ψ(y) = ψ(1 − y) − π cot(πy)` with `|π cot(πd)| ≤ 1/d` for `d ≤ ½`).
/// `None` when the ball is not well inside.
fn psi_abs(x: &BigFloat, e: f64) -> Option<f64> {
    let ld = pole_distance(x)?;
    if !inside(ld, e) {
        return None;
    }
    Some(lsum(
        -ld + shrink(e - ld),
        (ln1p_abs(x) + 1.0 + EULER_GAMMA_UP).log2(),
    ))
}

/// `log₂` of an upper bound on `|ψ(u) − ψ(v)|` (`h = u − v`) over balls of
/// radius `2^e`: the sum of the two bounds, or for `u, v > 0` the mean
/// value `|h|·ψ′(m) ≤ |h|·(1/m + 1/m²)` with `m = min(u, v)` (`ψ′` decreases
/// on `(0, ∞)`, and `ψ′(m) = Σ 1/(m + k)² ≤ 1/m² + 1/m`), whichever is
/// smaller — the difference cancels for large arguments (`B(10⁸, ½)`).
fn psi_diff(u: &BigFloat, v: &BigFloat, h: &BigFloat, e: f64) -> Option<f64> {
    let crude = lsum(psi_abs(u, e)?, psi_abs(v, e)?);
    if pos(u) && pos(v) {
        let lm = lg(u).min(lg(v));
        let mvt = lg(h) + lsum(-lm, -2.0 * lm) + 2.0 * shrink(e - lm);
        return Some(crude.min(mvt));
    }
    Some(crude)
}

/// `log₂` of an upper bound on `|ψ⁽ⁿ⁺¹⁾(y)| = (n+1)!·Σₖ 1/|y + k|^{n+2}` over
/// the ball `x ± 2^e`: `Σ_k 1/|y + k|^p ≤ 1/y^p + 1/((p−1)y^{p−1})` for
/// `y > 0`, `≤ 1/d^p + 2^{p+2}` otherwise (`d` the distance to the nearest
/// pole: the other terms are at distances `j ± d ≥ j − ½`).
fn polygamma_deriv(n: f64, x: &BigFloat, e: f64) -> Sens {
    let Some(ld) = pole_distance(x) else {
        return Sens::Unbounded;
    };
    if !inside(ld, e) {
        return Sens::Unbounded;
    }
    let p = n + 2.0;
    let sum = if pos(x) {
        lsum(-p * ld, -(p - 1.0) * ld - (p - 1.0).log2())
    } else {
        lsum(-p * ld, p + 2.0)
    };
    // Over the ball `|x + k|^(−p)` grows by at most `(1 − r/d)^(−p)` (`r`
    // the radius, `d` the distance), `(8/7)^p` only for the widest ball
    // admitted: 329 bits for `polygamma(1708, √22)`, whose argument is
    // known to `2^−265`, and three evaluations instead of one.
    let growth = -p * (1.0 - (e - ld).exp2()).log2();
    Sens::Deriv(log2_factorial(n + 1.0) + sum + growth.min(p * BALL))
}

/// A discrete parameter (an order or degree the routine takes as an
/// integer, to within `10⁻¹²`): no contribution when its error is below
/// that tolerance, no bound otherwise.
fn discrete(e: f64) -> Sens {
    if e <= -40.0 {
        Sens::Zero
    } else {
        Sens::Unbounded
    }
}

fn deriv_if(ok: bool, d: f64) -> Sens {
    if ok { Sens::Deriv(d) } else { Sens::Unbounded }
}

// ── The arguments ──────────────────────────────────────────────────────────

/// The real parts of a node's arguments, their error radii (`log₂`, `−∞`
/// when exact) and the node's value.
struct Args<'a> {
    x: Vec<&'a BigFloat>,
    e: Vec<f64>,
    v: &'a BigFloat,
    rm: RoundingMode,
}

fn radius(e: ErrExp) -> f64 {
    // An exact bound is already `−∞`.
    e
}

/// The sensitivity of node `node` to its argument number `i`.
fn analytic(arena: &Arena, node: &ExprNode, i: usize, a: &Args<'_>, cc: &mut Consts) -> Sens {
    let x = a.x[i];
    let e = a.e[i];
    let lv = lg(a.v);
    match node {
        // Γ′ = Γ·ψ.
        ExprNode::Gamma(_) => psi_abs(x, e).map_or(Sens::Unbounded, |s| Sens::Deriv(lv + s)),
        ExprNode::Factorial(_) => {
            let x1 = sum(x, &int(1));
            psi_abs(&x1, e).map_or(Sens::Unbounded, |s| Sens::Deriv(lv + s))
        }
        ExprNode::LogGamma(_) => psi_abs(x, e).map_or(Sens::Unbounded, Sens::Deriv),
        ExprNode::Digamma(_) => polygamma_deriv(0.0, x, e),
        ExprNode::Polygamma(..) => {
            if i == 0 {
                discrete(e)
            } else {
                polygamma_deriv(f(a.x[0]).round().max(0.0), x, e)
            }
        }
        // ∂ ln B/∂a = ψ(a) − ψ(a + b).
        ExprNode::Beta(..) => {
            let (p, q) = (a.x[i], a.x[1 - i]);
            let s = sum(p, q);
            psi_diff(p, &s, q, e).map_or(Sens::Unbounded, |d| Sens::Deriv(lv + d))
        }
        // C(n, k) = Γ(n+1)/(Γ(k+1)Γ(n−k+1)).
        ExprNode::Binomial(..) => {
            if a.v.is_zero() {
                // C(n, k) = 0 on the whole ball when an exact negative
                // integer k puts a pole of Γ(k + 1) below and the ball of n
                // meets no pole of Γ(n + 1) above (`binomial(√2, −3)`).
                let (n, k) = (a.x[0], a.x[1]);
                let k_pole = a.e[1] == f64::NEG_INFINITY && k.is_int() && neg(k);
                let n1 = sum(n, &int(1));
                let n_clear = pole_distance(&n1).is_some_and(|ld| inside(ld, a.e[0]));
                return if k_pole && n_clear {
                    Sens::Zero
                } else {
                    Sens::Unbounded
                };
            }
            let (n, k) = (a.x[0], a.x[1]);
            let one = int(1);
            let n1 = sum(n, &one);
            let nk1 = sum(&diff(n, k), &one);
            let k1 = sum(k, &one);
            let d = if i == 0 {
                psi_diff(&n1, &nk1, k, e)
            } else {
                psi_diff(&nk1, &k1, &diff(&nk1, &k1), e)
            };
            d.map_or(Sens::Unbounded, |d| Sens::Deriv(lv + d))
        }
        // erf′ = −erfc′ = (2/√π)·e^{−x²}.
        ExprNode::Erf(_) | ExprNode::Erfc(_) => {
            let lo = (f(x).abs() - e.exp2()).max(0.0);
            Sens::Deriv(LOG2_TWO_OVER_SQRT_PI - lo * lo * LOG2E)
        }
        ExprNode::Si(_) => Sens::Deriv((-lg(x)).min(0.0)),
        // Ci′ = cos x/x, Ei′ = eˣ/x: logarithmic singularities at 0.
        ExprNode::Ci(_) => deriv_if(inside(lg(x), e), -lg(x) + shrink(e - lg(x))),
        ExprNode::Ei(_) => deriv_if(
            inside(lg(x), e),
            (f(x) + e.exp2()) * LOG2E - lg(x) + shrink(e - lg(x)),
        ),
        // li′ = 1/ln x: singular at 1, and the ball must not reach 0.
        ExprNode::Li(_) => {
            let u = diff(x, &int(1));
            if !inside(lg(&u), e) || !inside(lg(x), e) {
                return Sens::Unbounded;
            }
            if lg(&u) < -1.0 {
                // |ln(1 + u)| ≥ |u|/(1 + |u|).
                Sens::Deriv((1.0 + f(&u).abs()).log2() - lg(&u) + BALL)
            } else {
                Sens::Deriv(-(lg(x).abs() * LN2).max(0.28).log2() + 1.0)
            }
        }
        // ζ(s) = 1/(s − 1) + γ + O(s − 1): |ζ′(s)| ≤ 1/(s − 1)² + 1 for
        // s ≥ 0; left of 0, numerically.
        ExprNode::Zeta(_) => {
            if neg(x) {
                return Sens::Numeric;
            }
            // For `σ ≥ 3` on the ball, `|ζ′(σ)| ≤ ln 2·2^−σ + ∫₂^∞ ln t·t^−σ dt
            // ≤ 1.9·2^−σ < 2^(1−σ)`.  Before 0.30 the bound was at least 1:
            // `zeta(720·polygamma(1669, E))` (an argument of `10⁴⁰⁰⁰` known to
            // its relative error, and a value of 1) had a ball containing 0
            // and came out `0`.
            // (The low end of the ball: at least `7x/8` when its radius is
            // at most `x/8`, which also holds beyond the `f64` range.)
            let low = if pos(x) && inside(lg(x), e) {
                (f(x) - e.exp2()).max(0.875 * lg(x).exp2())
            } else {
                f(x) - e.exp2()
            };
            if low >= 3.0 {
                return Sens::Deriv(1.0 - low);
            }
            let lu = lg(&diff(x, &int(1)));
            deriv_if(inside(lu, e), lsum(-2.0 * (lu - shrink(e - lu)), 0.0))
        }
        ExprNode::DiracDelta(_) => Sens::Zero,
        ExprNode::Apply(sid, _) => match arena.lib_fn(*sid) {
            Some(fun) => lib_sens(fun, i, a, cc),
            None => Sens::Numeric,
        },
        _ => Sens::Numeric,
    }
}

/// [`analytic`] for the library functions.
fn lib_sens(fun: LibFn, i: usize, a: &Args<'_>, cc: &mut Consts) -> Sens {
    let x = a.x[i];
    let e = a.e[i];
    let lv = lg(a.v);
    match fun {
        LibFn::Erfi => {
            let hi = f(x).abs() + e.exp2();
            Sens::Deriv(LOG2_TWO_OVER_SQRT_PI + hi * hi * LOG2E)
        }
        // (erf⁻¹)′(y) = (√π/2)·e^{w²}, w = erf⁻¹(y): singular at y = ±1
        // (erfc⁻¹ at 0 and 2).
        LibFn::ErfInv | LibFn::ErfcInv => {
            let (lo, hi) = if fun == LibFn::ErfInv {
                (sum(x, &int(1)), one_minus(x))
            } else {
                (x.clone(), diff(&int(2), x))
            };
            let w = f(a.v);
            deriv_if(
                inside(lg(&lo), e) && inside(lg(&hi), e),
                -LOG2_TWO_OVER_SQRT_PI + w * w * LOG2E + 0.5,
            )
        }
        // Shi′ = sinh x/x ≤ max(1.18, e^{|x|}/(2|x|)); Chi′ = cosh x/x.
        LibFn::Shi => {
            let ax = f(x).abs() + e.exp2();
            Sens::Deriv(if ax < 1.0 {
                0.25
            } else {
                ax * LOG2E - (2.0 * ax).log2()
            })
        }
        LibFn::Chi => deriv_if(
            inside(lg(x), e),
            (f(x).abs() + e.exp2()) * LOG2E - lg(x) + shrink(e - lg(x)),
        ),
        // |S′|, |C′| ≤ 1; and for |y| ≥ 1, |S(y) − ½|, |C(y) − ½| ≤ f + g ≤
        // 1/(π|y|) + 1/(π²|y|³) < 0.43/|y| (DLMF 7.12.2, 7.12.3: `f`, `g`
        // positive and below their first terms), so over a ball of
        // radius at most |x|/8 the value moves by less than `2/|x|`.  Before
        // 0.30 only the derivative bound: `fresnels(erfi(335))` had a ball
        // of `2^(161900 − prec)` around its value ½.
        LibFn::FresnelS | LibFn::FresnelC => {
            let lx = lg(x);
            if lx > 1.0 && inside(lx, e) && 1.0 - lx < e {
                Sens::Change(1.0 - lx)
            } else {
                Sens::Deriv(0.0)
            }
        }
        LibFn::UpperGamma | LibFn::LowerGamma => incomplete_gamma(fun, i, a),
        LibFn::ExpInt => {
            // E_ν(x) = ∫₁^∞ e^{−xt} t^{−ν} dt: E_ν′ = −E_{ν−1} = −E_ν·E[T] and
            // ∂E_ν/∂ν = −E_ν·E[ln T] for T of density ∝ e^{−xt} t^{−ν} on
            // (1, ∞), with E[T] ≤ 1 + (max(0, −ν) + 1)/x (T − 1 is
            // stochastically below a Gamma(max(0, −ν) + 1, x) variable).
            let (nu, xv) = (a.x[0], a.x[1]);
            if !pos(xv) || a.v.is_zero() {
                return Sens::Numeric;
            }
            let c = f(nu).min(0.0).abs() + 1.0;
            let mean_t = 1.0 + c * (-lg(xv)).exp2();
            if i == 1 {
                deriv_if(inside(lg(xv), a.e[1]), lv + mean_t.log2() + BALL)
            } else {
                Sens::Deriv(lv + mean_t.ln().max(f64::MIN_POSITIVE).log2())
            }
        }
        LibFn::PolyLog => polylog_sens(i, a),
        LibFn::DirichletEta => eta_sens(x),
        // Ai″ = x·Ai.  For x ≥ 0: |Ai′| ≤ (√x + 1)·Ai (Bi likewise),
        // x·Ai ≤ √x·|Ai′| and x·Bi ≤ (√x + 1)·Bi′; for x < 0: |Ai′|, |Bi′| ≤
        // 0.6·(|x|^{1/4} + 1) and |Ai|, |Bi| ≤ 0.71 (mpmath over x ≤ 40 and
        // decades to 10⁻⁶: largest ratios 0.91 and 0.87; the asymptotic
        // envelopes decrease beyond).
        LibFn::AiryAi | LibFn::AiryBi | LibFn::AiryAiPrime | LibFn::AiryBiPrime => {
            let ax = f(x).abs() + e.exp2();
            let prime = matches!(fun, LibFn::AiryAiPrime | LibFn::AiryBiPrime);
            Sens::Deriv(if neg(x) {
                if prime {
                    ax.log2()
                } else {
                    (ax.sqrt().sqrt() + 1.0).log2()
                }
            } else {
                let plus = if fun == LibFn::AiryAiPrime { 0.0 } else { 1.0 };
                (ax.sqrt() + plus).log2() + lv + 0.2
            })
        }
        LibFn::BesselJ | LibFn::BesselY | LibFn::BesselI | LibFn::BesselK => {
            if i == 0 {
                return bessel_order(fun, a);
            }
            bessel_x(fun, a, cc)
        }
        // K′ = (E − (1 − m)K)/(2m(1 − m)), E′ = (E − K)/(2m): at most 1 in
        // magnitude for |m| ≤ ½; beyond, with E ≤ π/2 (m > 0) or
        // E ≤ √(1 − m)·π/2 (m < 0) and K ≤ π/(2√(1 − m)) (m > 0):
        // |K′| ≤ (π/2·max(1, √(1−m)) + (1−m)K)/(2|m|(1−m)),
        // |E′| ≤ (E + π/2·max(1, 1/√(1−m)))/(2|m|) (mpmath: ratios ≤ 0.83).
        LibFn::EllipticK | LibFn::EllipticE => {
            let u = one_minus(x);
            if !inside(lg(&u), e) {
                return Sens::Unbounded;
            }
            if lg(x) < -1.0 {
                return Sens::Deriv(0.0);
            }
            let half_pi = std::f64::consts::FRAC_PI_2.log2();
            let lu = lg(&u);
            Sens::Deriv(if fun == LibFn::EllipticK {
                lsum(half_pi + 0.5 * lu.max(0.0), lu + lv) - 1.0 - lg(x) - lu + BALL
            } else {
                lsum(lv, half_pi - 0.5 * lu.min(0.0)) - 1.0 - lg(x) + BALL
            })
        }
        LibFn::EllipticF => elliptic_f_sens(i, a, cc),
        LibFn::EllipticPi => elliptic_pi_sens(i, a),
        LibFn::BetaInc | LibFn::BetaIncRegularized => {
            betainc_sens(fun == LibFn::BetaIncRegularized, i, a)
        }
        LibFn::Legendre
        | LibFn::ChebyshevT
        | LibFn::ChebyshevU
        | LibFn::Hermite
        | LibFn::Laguerre
        | LibFn::Gegenbauer
        | LibFn::Jacobi
        | LibFn::AssocLaguerre => {
            if i == 0 {
                discrete(e)
            } else {
                Sens::Numeric
            }
        }
        LibFn::AssocLegendre => {
            if i < 2 {
                discrete(e)
            } else {
                Sens::Numeric
            }
        }
        // H′(z) = ψ′(z + 1).
        LibFn::Harmonic => polygamma_deriv(0.0, &sum(x, &int(1)), e),
        // ∂ ln rf(x, n)/∂x = ψ(x + n) − ψ(x), ∂/∂n = ψ(x + n); ff(x, n) =
        // Γ(x + 1)/Γ(x − n + 1) likewise.  At a zero (a pole of the lower
        // Γ) numerically.
        LibFn::RisingFactorial | LibFn::FallingFactorial => {
            if a.v.is_zero() {
                return Sens::Numeric;
            }
            let (xv, nv) = (a.x[0], a.x[1]);
            let one = int(1);
            let (upper, lower) = if fun == LibFn::RisingFactorial {
                (sum(xv, nv), xv.clone())
            } else {
                (sum(xv, &one), sum(&diff(xv, nv), &one))
            };
            let d = if i == 0 {
                psi_diff(&upper, &lower, nv, e)
            } else if fun == LibFn::RisingFactorial {
                psi_abs(&upper, e)
            } else {
                psi_abs(&lower, e)
            };
            d.map_or(Sens::Unbounded, |d| Sens::Deriv(lv + d))
        }
        _ => Sens::Numeric,
    }
}

/// Bessel functions in the order `ν ≥ 0`, `x > 0`, from the integral and
/// series representations: `∂K_ν/∂ν = ∫₀^∞ e^{−x cosh t} t sinh(νt) dt ≤
/// (ν/x)·K_ν` (`t ≤ sinh t`, then by parts); `∂I_ν/∂ν = Σ T_k (ln(x/2) −
/// ψ(ν+k+1))` over the (positive) terms `T_k` of `I_ν`, so `|∂I_ν/∂ν| ≤
/// I_ν·(|ln(x/2)| + γ + ln(ν + 1 + x/2))` (`ψ(y) ∈ [−γ, ln y)` for
/// `y ≥ 1`, Jensen, and the mean of `k` is `(x/2)I_{ν+1}/I_ν ≤ x/2`); `J_ν`
/// has the same terms with alternating signs, bounded by that of
/// `I_ν(|x|) ≤ (|x|/2)^ν e^{|x|}/Γ(ν+1)` — used for `|x| ≤ 16`, where it
/// overestimates by at most `e^{16}`.  Otherwise (and `Y`) numerically.
fn bessel_order(fun: LibFn, a: &Args<'_>) -> Sens {
    let (nu, x) = (a.x[0], a.x[1]);
    if neg(nu) || !pos(x) || a.v.is_zero() {
        return Sens::Numeric;
    }
    let (nf, xf) = (f(nu), f(x));
    let lv = lg(a.v);
    let log_terms =
        || ((xf / 2.0).ln().abs() + EULER_GAMMA_UP + (nf + 1.0 + xf / 2.0).ln()).log2() + 0.2;
    match fun {
        LibFn::BesselK => Sens::Deriv(if nu.is_zero() {
            f64::NEG_INFINITY
        } else {
            lv + nf.log2() - lg(x) + 0.2
        }),
        LibFn::BesselI => Sens::Deriv(lv + log_terms()),
        LibFn::BesselJ if xf <= 16.0 => {
            let (lg_g, err) = ln_gamma_f64(&sum(nu, &int(1)));
            Sens::Deriv(nf * (lg(x) - 1.0) + xf * LOG2E - (lg_g - err) * LOG2E + log_terms())
        }
        _ => Sens::Numeric,
    }
}

/// `Li_s(z)`: `∂/∂z = Li_{s−1}(z)/z`, which for an exact integer `s` is
/// `1/(1 − z)` (`s = 1`), `−ln(1 − z)/z` (`s = 2`) and at most `ζ(2) < 2`
/// for `s ≥ 3`, `|z| ≤ 1`; for any real `s` and `|z| < 1`, `|Li_{s−1}(z)/z|
/// ≤ Σ |z|^{k−1} k^m ≤ m!/(1 − |z|)^{m+1}` with `m = max(0, ⌈1 − s⌉)`
/// (`k^m ≤ k(k+1)⋯(k+m−1)`).  In `s`: `|∂Li_s/∂s| = |Σ z^k ln k/k^s| ≤
/// Σ |z|^k k^{1−s} ≤ |z|·m′!/(1 − |z|)^{m′+1}`, `m′ = max(0, ⌈2 − s⌉)`
/// (`ln k ≤ k`).  For `|z| ≥ 1`, numerically.
fn polylog_sens(i: usize, a: &Args<'_>) -> Sens {
    let (s, z) = (a.x[0], a.x[1]);
    let e = a.e[1];
    let sf = f(s);
    if i == 0 {
        let w = one_minus(&z.abs());
        if !pos(&w) || !inside(lg(&w), a.e[0].max(e)) {
            return Sens::Numeric;
        }
        let m = (2.0 - sf).ceil().max(0.0);
        return Sens::Deriv(lg(z) + log2_factorial(m) - (m + 1.0) * (lg(&w) - shrink(e - lg(&w))));
    }
    let u = one_minus(z);
    let exact_int = s.is_int() && a.e[0] == f64::NEG_INFINITY;
    if exact_int && sf == 1.0 {
        return deriv_if(inside(lg(&u), e), -lg(&u) + BALL);
    }
    if exact_int && sf == 2.0 && (!pos(z) || inside(lg(&u), e)) {
        if lg(z) < -1.0 {
            return Sens::Deriv(1.0);
        }
        // |ln(1 − z)/z|, and ln u moves by at most 8/7·r/u.
        let ln_u = (lg(&u) * LN2).abs() + shrink(e - lg(&u)) * LN2;
        return Sens::Deriv(ln_u.max(f64::MIN_POSITIVE).log2() - lg(z) + 0.2);
    }
    if exact_int && sf >= 3.0 && lg(z) <= 0.0 {
        return Sens::Deriv(1.0);
    }
    let w = one_minus(&z.abs());
    if !pos(&w) || !inside(lg(&w), e) {
        return Sens::Numeric;
    }
    let m = (1.0 - sf).ceil().max(0.0);
    Sens::Deriv(log2_factorial(m) - (m + 1.0) * (lg(&w) - shrink(e - lg(&w))))
}

/// `η(s) = (1 − 2^{1−s})ζ(s)` for `s ≥ 0`: `|η′| ≤ 2^{1−s} ln 2·|ζ| + |1 −
/// 2^{1−s}|·|ζ′|` with `|ζ(s)| ≤ 1/|s − 1| + 1` and `|ζ′(s)| ≤ 1/(s − 1)² + 1`,
/// and at most 1 for `|s − 1| ≤ ¼` (`η′(1) = 0.16`); left of 0,
/// numerically.
fn eta_sens(s: &BigFloat) -> Sens {
    if neg(s) {
        return Sens::Numeric;
    }
    let d = f(&diff(s, &int(1))).abs();
    if d <= 0.25 {
        return Sens::Deriv(0.0);
    }
    let sf = f(s);
    let t = (1.0 - sf).exp2();
    let bound = t * LN2 * (1.0 / d + 1.0) + (1.0 - t).abs() * (1.0 / (d * d) + 1.0);
    Sens::Deriv(bound.log2() + 0.2)
}

/// `F(φ|m)`: `∂F/∂φ = (1 − m sin²φ)^{−1/2}`, and `|∂F/∂m| ≤ |φ|/(2·D^{3/2})`
/// with `D = 1 − max(m, 0)·(sin²φ for |φ| < π/2, else 1)` the smallest
/// `1 − m sin²θ` on `[0, φ]` (the integrand's derivative in `m` is
/// `sin²θ/(2(1 − m sin²θ)^{3/2})`).
fn elliptic_f_sens(i: usize, a: &Args<'_>, cc: &mut Consts) -> Sens {
    let (phi, m) = (a.x[0], a.x[1]);
    let e = a.e[i];
    let rm = a.rm;
    let small = lg(phi) < 0.6; // |φ| < π/2
    // 1 − m sin²φ (when φ is not huge), and its minimum on [0, φ].
    let at_phi = (lg(phi) <= 20.0).then(|| {
        let p = LP + 32;
        let s = phi.sin(p, rm, cc);
        one_minus(&m.mul(&s.mul(&s, p, rm), p, rm))
    });
    let d_min = if !pos(m) {
        Some(int(1))
    } else if small {
        at_phi.clone()
    } else {
        Some(one_minus(m))
    };
    let Some(d_min) = d_min.filter(pos) else {
        return Sens::Numeric;
    };
    if i == 0 {
        let d = at_phi.filter(pos).unwrap_or(d_min);
        // Over the ball 1 − m sin²φ moves by at most |m|·r.
        return deriv_if(
            lg(&d) > lg(m) + e + 3.0,
            -0.5 * (lg(&d) - shrink(lg(m) + e - lg(&d))),
        );
    }
    let lphi = lg(phi);
    deriv_if(lg(&d_min) > e + 3.0, lphi - 1.0 - 1.5 * (lg(&d_min) - BALL))
}

/// `Π(n|m)` for `n, m < 1`: `|∂Π/∂n| ≤ π(2 − n)/(4(1 − n)^{3/2}√c)`,
/// `|∂Π/∂m| ≤ π/(4√(1 − n)·c^{3/2})`, `c = min(1, 1 − m)` (the integrand
/// `(1 − n sin²θ)^{−1}(1 − m sin²θ)^{−1/2}` differentiated under the integral
/// and bounded by the integrals of `Π(n|0)`; mpmath: ratios ≤ 0.99);
/// otherwise numerically.
fn elliptic_pi_sens(i: usize, a: &Args<'_>) -> Sens {
    let (n, m) = (a.x[0], a.x[1]);
    let (un, um) = (one_minus(n), one_minus(m));
    if !pos(&un) || !pos(&um) {
        return Sens::Numeric;
    }
    let (lun, lum) = (lg(&un), lg(&um));
    let lc = lum.min(0.0);
    let quarter_pi = std::f64::consts::FRAC_PI_4.log2();
    if i == 0 {
        let two_minus_n = (1.0 + f(&un)).log2();
        return deriv_if(
            inside(lun, a.e[0]),
            quarter_pi + two_minus_n - 1.5 * (lun - shrink(a.e[0] - lun)) - 0.5 * lc,
        );
    }
    deriv_if(
        lc == 0.0 || inside(lum, a.e[1]),
        quarter_pi - 0.5 * lun - 1.5 * (lc - BALL),
    )
}

/// `Γ(s, x)` and `γ(s, x)`: in `x` the integrand `x^{s−1}e^{−x}` (singular at
/// 0 for `s < 1`); in `s`, `∂ ln Γ(s, x)/∂s = E[ln T | T > x]` for `T` of
/// density `∝ t^{s−1}e^{−t}`, which lies between `ln x` and
/// `ln E[T | T > x] = ln(s + x^s e^{−x}/Γ(s, x))` (Jensen), and
/// `∂ ln γ(s, x)/∂s = E[ln T | T < x]`, at most `e·(|ln x| + 1/s)` in
/// magnitude (the density against `t^{s−1}` on `(0, min(x, 1))`, and
/// `γ(s, x) ≥ γ(s, 1) ≥ e^{−1}/s` beyond).
fn incomplete_gamma(fun: LibFn, i: usize, a: &Args<'_>) -> Sens {
    let (s, x) = (a.x[0], a.x[1]);
    let lv = lg(a.v);
    let sf = f(s);
    let lx = lg(x);
    if i == 1 {
        if !inside(lx, a.e[1]) {
            return if sf >= 1.0 {
                Sens::Numeric
            } else {
                Sens::Unbounded
            };
        }
        let r = a.e[1].exp2();
        let xf = f(x);
        // x^{s−1} over the ball: at x − r when s < 1, x + r otherwise.
        let lxb = if sf < 1.0 {
            lx - shrink(a.e[1] - lx)
        } else {
            lx + swell(a.e[1] - lx)
        };
        return Sens::Deriv((sf - 1.0) * lxb - (xf - r) * LOG2E);
    }
    if a.v.is_zero() {
        return Sens::Numeric;
    }
    let ln_x = lx * LN2;
    if fun == LibFn::LowerGamma {
        if !pos(s) || x.is_zero() {
            return Sens::Numeric;
        }
        let bound = std::f64::consts::E * (ln_x.abs() + (-lg(s)).exp2());
        return Sens::Deriv(lv + bound.log2());
    }
    if x.is_zero() {
        return psi_abs(s, a.e[0]).map_or(Sens::Unbounded, |p| Sens::Deriv(lv + p));
    }
    // log₂(x^s e^{−x}/Γ(s, x)).
    let l_rho = sf * lx - f(x) * LOG2E - lv;
    let l_mean = lsum(lg(s), l_rho);
    let bound = ln_x.abs().max((l_mean * LN2).max(0.0)) + 1e-300;
    Sens::Deriv(lv + bound.log2() + 0.1)
}

/// Bessel functions in `x`.  For `ν ≥ 0` (`K_{−ν} = K_ν`, and an integer
/// order's sign does not matter for `J`, `I`): `|K′_ν| ≤ (1 + (ν+1)/x)·K_ν`,
/// `|I′_ν| ≤ (1 + ν/x)·I_ν`, and `J′_ν = (ν/x)J_ν − J_{ν+1}` with `|J_{ν+1}| ≤
/// min(1, (|x|/2)^{ν+1}/Γ(ν+2))` (mpmath over `ν ≤ 20`, `x` from `10⁻⁸` to
/// 30: ratios ≤ 1).  Otherwise, and for `Y`, the recurrence
/// `B′_ν = B_{ν−1} ∓ (ν/x)·B_ν` with `B_{ν−1}` evaluated at 128 bits.  Singular
/// at 0 except `J_n`, `I_n` of an integer order.
fn bessel_x(fun: LibFn, a: &Args<'_>, cc: &mut Consts) -> Sens {
    let (nu, x) = (a.x[0], a.x[1]);
    let int_order = nu.is_int();
    let entire = matches!(fun, LibFn::BesselJ | LibFn::BesselI) && int_order;
    if !entire && !inside(lg(x), a.e[1]) {
        return Sens::Unbounded;
    }
    let lv = lg(a.v);
    let lx = lg(x);
    let nf = f(nu);
    let usable = !neg(nu) || int_order || fun == LibFn::BesselK;
    if usable && fun != LibFn::BesselY {
        let an = nf.abs();
        // log₂(ν/|x|), and the ball: |x| may shrink by an eighth.
        let ratio = if an == 0.0 {
            f64::NEG_INFINITY
        } else {
            an.log2() - lx + shrink(a.e[1] - lx)
        };
        return Sens::Deriv(match fun {
            LibFn::BesselK => lsum(0.0, (an + 1.0).log2() - lx + shrink(a.e[1] - lx)) + lv + 0.2,
            LibFn::BesselI => lsum(0.0, ratio) + lv + 0.2,
            _ => {
                let (lg_g, err) = ln_gamma_f64(&sum(&int(2), &nu.abs()));
                let small = (an + 1.0) * (lx + swell(a.e[1] - lx) - 1.0) - (lg_g - err) * LOG2E;
                lsum(small.min(0.0), ratio + lv) + 0.2
            }
        });
    }
    let nu1 = diff(nu, &int(1));
    let rm = a.rm;
    let companion = match fun {
        LibFn::BesselJ => super::arb_bessel_j(&nu1, x, LP, rm, cc),
        LibFn::BesselY => super::arb_bessel_y(&nu1, x, LP, rm, cc),
        LibFn::BesselI => super::arb_bessel_i(&nu1, x, LP, rm, cc),
        _ => super::arb_bessel_k(&nu1, x, LP, rm, cc),
    };
    match companion {
        Ok(c) => {
            let ratio = if nu.is_zero() {
                f64::NEG_INFINITY
            } else {
                lg(nu) - lg(x) + lg(a.v)
            };
            Sens::Deriv(lsum(lg(&c), ratio) + 1.0)
        }
        Err(_) => Sens::Numeric,
    }
}

/// `B_{(x₁, x₂)}(a, b) = ∫_{x₁}^{x₂} t^{a−1}(1−t)^{b−1} dt` and its regularised
/// form `I = B/B(a, b)`.
///
/// * A limit `t`: the integrand `w(t)` (over `B(a, b)`), times the largest
///   ratio of `w` over the ball, `exp(r·8/7·(|a−1|/t + |b−1|/(1−t)))`.  A limit
///   within 8 radii of 0 or 1 bounds the change directly by the integral of
///   `w` over the reach `s` of the ball: `∫₀^s w ≤ s^a/a·max(1, (1−s)^{b−1})`
///   (and symmetrically at 1) — an `x` that rounds to 1 had a derivative of
///   `∞` there and was taken as exact.
/// * A shape: see [`beta_shape`].
fn betainc_sens(regularized: bool, i: usize, a: &Args<'_>) -> Sens {
    let (p, q, x1, x2) = (a.x[0], a.x[1], a.x[2], a.x[3]);
    if !pos(p) || !pos(q) {
        return Sens::Numeric;
    }
    let (ln_b_lo, ln_b_hi) = log2_beta_bounds(p, q);
    // The integrand over B(a, b), bounded above.
    let norm = if regularized { ln_b_lo } else { 0.0 };
    if i >= 2 {
        let t = a.x[i];
        let e = a.e[i];
        let u = one_minus(t);
        let (pf, qf) = (f(p), f(q));
        let (lt, lu) = (lg(t), lg(&u));
        if !inside(lt, e) || !inside(lu, e) {
            // The reach of the ball from the nearer end.
            let near_zero = lt <= lu;
            let (first, other, l_edge) = if near_zero {
                (pf, qf, lt)
            } else {
                (qf, pf, lu)
            };
            let ls = lsum(l_edge, e + 3.0);
            if ls >= -1.0 {
                return Sens::Unbounded;
            }
            let rest = if other < 1.0 {
                (other - 1.0) * (1.0 - ls.exp2()).log2()
            } else {
                0.0
            };
            return Sens::Change(first * ls - first.log2() + rest - norm + 1.0);
        }
        // ln of the largest ratio of the integrand over the ball:
        // `|p − 1|·|ln(1 − r/t)| + |q − 1|·|ln(1 − r/(1 − t))|` exactly
        // (before 0.30 `r·(8/7)·(|p − 1|/t + …)`, the widest ball's factor
        // multiplied into an exponent).
        let variation =
            ((pf - 1.0).abs() * shrink(e - lt) + (qf - 1.0).abs() * shrink(e - lu)) * LN2;
        return Sens::Deriv((pf - 1.0) * lt + (qf - 1.0) * lu - norm + variation * LOG2E);
    }
    let (lo, hi) = if x1.cmp(x2).is_some_and(|c| c > 0) {
        (x2, x1)
    } else {
        (x1, x2)
    };
    let bound = if i == 0 {
        beta_shape(p, q, lo, hi, a.e[0], a.v, regularized, ln_b_hi)
    } else {
        let (lo1, hi1) = (one_minus(hi), one_minus(lo));
        beta_shape(q, p, &lo1, &hi1, a.e[1], a.v, regularized, ln_b_hi)
    };
    bound.map_or(Sens::Unbounded, |d| Sens::Deriv(lg(a.v) + d))
}

/// `log₂` of an upper bound on `|∂ ln B/∂p|` for the first shape `p` of
/// `B_{(lo, hi)}(p, q)` (or of `I`): with `X ~ Beta(p, q)`,
/// `∂ ln B_{(lo,hi)}/∂p = μ = E[ln X | lo < X < hi] ∈ [ln lo, ln hi]`, so
/// `|μ| ≤ |ln lo|`, and for `lo = 0`, `|μ| ≤ R·(|ln hi| + 1/p)` (the density
/// against `t^{p−1}` on `(0, hi)`; `R = (1−hi)^{1−q}` for `q > 1`, else 1).
/// Regularised, `E[ln X] = ψ(p) − ψ(p + q)` is subtracted: `|μ| + |ψ(p) −
/// ψ(p+q)|`, or for a tail the sharper `|∂I_x/∂p| ≤ (1 − I_x)(ψ(p+q) − ψ(p))`
/// (so `(1 − I)/I·…` relative for the lower tail, `ψ(p+q) − ψ(p)` for the
/// upper one).  The second shape is the first of `B_{(1−hi, 1−lo)}(q, p)`.
/// `ln_b_hi`: an upper bound on `log₂ B(p, q)`.
#[allow(clippy::too_many_arguments)]
fn beta_shape(
    p: &BigFloat,
    q: &BigFloat,
    lo: &BigFloat,
    hi: &BigFloat,
    e: f64,
    value: &BigFloat,
    regularized: bool,
    ln_b_hi: f64,
) -> Option<f64> {
    let pq = sum(p, q);
    // |ψ(p) − ψ(p + q)|, natural-log units, as a log₂.
    let m = psi_diff(p, &pq, q, e)?;
    let one = int(1);
    let hi_one = hi.cmp(&one) == Some(0);
    let mu = if pos(lo) {
        (-lg(lo) * LN2).max(1e-300).log2()
    } else if hi_one {
        m
    } else {
        let qf = f(q);
        let r = if qf > 1.0 {
            (1.0 - qf) * lg(&one_minus(hi))
        } else {
            0.0
        };
        r + (-lg(hi) * LN2 + (-lg(p)).exp2()).log2()
    };
    let v = value.abs();
    if !regularized {
        let mut d = mu;
        if lo.is_zero() && !hi_one {
            // |μ| ≤ |∂ ln I/∂p| + |E ln X| ≤ ψ-difference/I, I = B/B(p, q).
            d = d.min(m + ln_b_hi - lg(&v));
        }
        return Some(d);
    }
    if lo.is_zero() && hi_one {
        return Some(f64::NEG_INFINITY); // I = 1
    }
    let mut d = lsum(mu, m);
    if lo.is_zero() {
        let one_minus_v = one_minus(&v);
        d = d.min(m + lg(&one_minus_v) - lg(&v));
    } else if hi_one {
        d = d.min(m);
    }
    Some(d)
}

// ── Numerical sensitivity ──────────────────────────────────────────────────

/// The change of node `id`'s value when child `child` (value `v`, error
/// radius `2^e`) moves by `δ = max(2^e, 4 ulps)`: node re-evaluated at the
/// working precision, the difference scaled back to `2^e` and doubled;
/// `None` when the node evaluates on neither side.
#[allow(clippy::too_many_arguments)]
fn numeric_change(
    arena: &Arena,
    id: ExprId,
    child: ExprId,
    e: ErrExp,
    value: &Complex,
    cache: &FxHashMap<ExprId, Complex>,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Option<ErrExp> {
    let v = cache.get(&child)?;
    let floor = accuracy::mag(v).map_or(e, |m| (m - prec as i64 + 2) as f64);
    // The move: `2^pe` with `pe` an integer at least the radius.
    let pe = e.max(floor).ceil();
    if !pe.is_finite() || pe.abs() > 2e9 {
        return None;
    }
    let mut delta = BigFloat::from_i32(1, LP);
    delta.set_exponent(i32::try_from(pe as i64 + 1).ok()?);
    let mut local: FxHashMap<ExprId, Complex> = FxHashMap::default();
    for c in arena.node(id).children() {
        local.insert(c, cache.get(&c)?.clone());
    }
    for moved in [sum(&v.0, &delta), diff(&v.0, &delta)] {
        local.insert(child, (moved, v.1.clone()));
        if let Ok(w) = super::eval_node(arena, id, &local, prec, rm, cc) {
            if [&w.0, &w.1].iter().any(|p| p.is_nan() || p.is_inf()) {
                continue;
            }
            let hp = prec + 64;
            let d = c_sub(&w, value, hp, rm);
            // Scaled back from the move to the radius, doubled.
            return Some(if accuracy::mag(&d).is_none() {
                EXACT
            } else {
                accuracy::lg_abs(&d) - (pe - e) + 1.0
            });
        }
    }
    None
}

/// The propagated error bound (exponent) of special-function node `id` with
/// value `value`: `Σ |∂f/∂xᵢ|·err(xᵢ)` over its distinct inexact children
/// (see the module documentation), doubled for the number of terms and
/// once more for the first-order approximation.  [`EXACT`] when every
/// argument is exact, [`UNKNOWN`] when an argument has no bound or the
/// sensitivity to it cannot be bounded.
#[allow(clippy::too_many_arguments)]
pub(super) fn special_error(
    arena: &Arena,
    id: ExprId,
    value: &Complex,
    cache: &FxHashMap<ExprId, Complex>,
    errs: &FxHashMap<ExprId, accuracy::Bound>,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> ErrExp {
    let node = arena.node(id);
    let kids = node.children();
    let mut x = Vec::with_capacity(kids.len());
    let mut e = Vec::with_capacity(kids.len());
    let mut joint = Vec::with_capacity(kids.len());
    for c in &kids {
        let (Some(v), Some(b)) = (cache.get(c), errs.get(c)) else {
            return UNKNOWN;
        };
        if b.is_unknown() {
            return UNKNOWN;
        }
        x.push(&v.0);
        joint.push(b.joint());
        e.push(radius(b.joint()));
    }
    if joint.iter().all(|&j| accuracy::is_exact(j)) {
        return EXACT;
    }
    let args = Args {
        x,
        e,
        v: &value.0,
        rm,
    };
    let mut contributions: Vec<ErrExp> = Vec::new();
    let mut seen: Vec<ExprId> = Vec::new();
    for (i, &c) in kids.iter().enumerate() {
        if accuracy::is_exact(joint[i]) || seen.contains(&c) {
            continue;
        }
        seen.push(c);
        // Every position the child occupies.
        let mut total = f64::NEG_INFINITY;
        let mut change = f64::NEG_INFINITY;
        let mut numeric = false;
        for (j, _) in kids.iter().enumerate().filter(|&(_, &k)| k == c) {
            match analytic(arena, node, j, &args, cc) {
                Sens::Zero => {}
                Sens::Deriv(d) if d.is_nan() => numeric = true,
                Sens::Deriv(d) => total = lsum(total, d),
                Sens::Change(d) if d.is_nan() => numeric = true,
                Sens::Change(d) => change = lsum(change, d),
                Sens::Numeric => numeric = true,
                Sens::Unbounded => return UNKNOWN,
            }
        }
        if numeric {
            match numeric_change(arena, id, c, joint[i], value, cache, prec, rm, cc) {
                Some(k) => contributions.push(k),
                None => return UNKNOWN,
            }
            continue;
        }
        let l = lsum(total + args.e[i], change);
        if l == f64::NEG_INFINITY {
            continue;
        }
        if !l.is_finite() || l > 1e15 {
            return UNKNOWN;
        }
        contributions.push(l);
    }
    // Twice the sum of the contributions (the factor 2 of the module
    // documentation; before 0.30 `2^(⌈log₂ k⌉ + 1)` times the largest).
    let total = contributions.into_iter().fold(EXACT, accuracy::lsum);
    accuracy::shift(total, 1.0)
}

//! Which special-function nodes are evaluated by the complex routines
//! (`cexpint.rs`, `cbessel.rs`, `binomial` and the Fresnel integrals here),
//! and the error bounds of those nodes.
//!
//! A node goes to them when an argument is not real, or when its real
//! arguments lie where the value is not real and the real routine refuses:
//! `expint(n, x)`, `uppergamma(a, x)`, `lowergamma(a, x)` for `x < 0`,
//! `lowergamma(a, x)` for `a < 0`, `li(x)` for `x < 0`, `bessely`,
//! `besselk` of a negative argument and `besselj`, `besseli` of one with a
//! non-integer order (refused before 0.45: "expint of negative argument is
//! complex").  Every other real case stays with the real routine.
//!
//! # Error bounds
//!
//! The node's bound is its own `2^−(prec+12)` relative plus, for each
//! inexact argument of radius `r`, twice `r` times a bound on the derivative
//! over the argument's ball:
//!
//! | function | derivative in `z`, bounded over `abs(u − z) ≤ r` |
//! |---|---|
//! | `Ei`, `E1`-like | `abs(e^u/u) ≤ e^(x+r)/(abs(z) − r)` |
//! | `li` | `1/abs(ln u) ≤ 1/(abs(ln z) − r/(abs(z) − r))` |
//! | `Si`, `Shi` | `abs(sin u/u), abs(sinh u/u) ≤ min(sinh(abs(z)+r)/(abs(z)+r), cosh(abs(y or x)+r)/(abs(z)−r))` |
//! | `Ci`, `Chi` | `cosh(abs(y), resp. abs(x), + r)/(abs(z) − r)` |
//! | `E_n` | `(abs(n − 1)·abs(E_n) + e^(−x+r))/(abs(z) − r)` (from `E_n′ = ((n−1)E_n − e^(−z))/z`) |
//! | `Γ(a, z)`, `γ(a, z)` | `abs(z^(a−1)e^(−z))·e^(abs(a−1)·r/(abs(z)−r) + r)` |
//! | Fresnel | `abs(sin, cos(πu²/2)) ≤ cosh(π(abs(x)+r)(abs(y)+r))` |
//!
//! and numerically (`sensitivity::numeric_change`: the node evaluated again
//! with the argument moved) for the orders and parameters (`E_n` in `n`,
//! `Γ(a, z)` in `a`, the Bessel orders), the Bessel and Airy functions and
//! `binomial`.  A function with a cut on `(−∞, 0]` (`(−∞, 1]` for `li`) has
//! no bound when the argument's ball reaches the cut with an undecided side,
//! and every function singular at 0 none when the ball reaches within eight
//! radii of 0.  A value whose arguments are all exactly real and whose
//! imaginary part came out exactly 0 is real where the function is real
//! there ([`real_valued`]: off its cut, or entire in its argument, or `E_n`
//! of an integer `n ≤ 0`) — not on its cut, where a route may yield an
//! exact 0 that is not the value (below).
//!
//! # On the cut
//!
//! At exactly real arguments whose last one is negative (the value from
//! above), the parts of some of these functions are real functions of
//! `x = −z > 0` with a closed form or a real routine ([`CutRule`]):
//!
//! | function | real part | imaginary part |
//! |---|---|---|
//! | `E_n(−x)`, `n ≥ 1` an integer | the general route | `−π·x^(n−1)/(n−1)!` |
//! | `Γ(−m, −x)`, `m ≥ 0` an integer | the general route | `(−1)^(m+1)·π/m!` |
//! | `E_n(−x)`, `n = j + 3/2`, `j` an integer | the general route | `(−1)^j·x^(n−1)·Γ(1 − n)` |
//! | `Chi(−x)`, `Ci(−x)` | `Chi(x)`, `Ci(x)` (real routines) | `π` |
//! | `K_n(−x)`, `n` an integer | `(−1)^n·K_n(x)` | `−π·I_n(x)` (real routines) |
//!
//! (`E₁(−x) = −Ei(x) − iπ`, DLMF 6.2.4, and DLMF 8.19.7 for `E_n`; DLMF
//! 8.4.15 for `Γ(−m, ·)`; for a half-integer `n`, `E_n(z) = z^(n−1)·Γ(1 −
//! n, z)` and `Γ(a, z) = Γ(a) − z^a·Γ(a)·γ*(a, z)` with `γ*` entire and real
//! on the real axis, DLMF 8.2.6, where on the cut `z^a = ±i·x^a`; the `ln z`
//! of the series of `Chi`, `Ci`, DLMF 6.6.5, 6.6.6, taken from above; DLMF
//! 10.34.2 for `K_n`.  (The half-integer `Γ(a, z)`, `γ(a, z)` reach `evalf`
//! as `erfc` of an imaginary argument, `cerf.rs`.))  Before 0.54 the general routes carried one
//! error bound, `abs(f)·2^−(prec+12)`, for both parts, so an imaginary part
//! `O(1)` next to a real part `e^x/x` lay inside it: `Chi(−200)` printed no
//! `πi` at 10, 16, 30 or 60 digits; and on the negative axis the asymptotic
//! expansion of `Γ(a, z)` is a real sum (the `∓iπ` is exponentially small
//! next to the leading term, inside its remainder bound), whose exact-0
//! imaginary part the rule above certified: `im(E₁(−160))` and
//! `im(Γ(0, −160))` were `0` at 10, 16 and 30 digits (they are `−π`) and
//! `im(E₂(−530))` at every count (`−1665.04…`).  Each part now carries its
//! own bound: the route's error (relative to `abs(f)`) on a part it
//! computes, the rounding of a closed form or a real routine (relative to
//! the part) on the others.  An argument's error moves each part by that
//! part's derivative along the cut: `0` for the constant imaginary parts,
//! `(n − 1)·abs(Im)/(x − r)·(1 + r/x)^(n−1)` for `E_n`'s, the closed forms
//! above for the real parts of `E_n`, `Γ`, `Chi`, `Ci`, and the change of
//! each part between the ends of the ball for `K_n` (measured, doubled, as
//! the other Bessel bounds).  Where a real routine refuses or a part it
//! gives is 0 (an underflow), `K_n` takes its general route.
//!
//! # `erfinv`, `erfcinv` next to the ends of their domains
//!
//! `erfinv(1 − u) = erfcinv(u)`: an argument `1 − 10⁻¹⁰⁰⁰` rounds onto 1 below
//! 3,322 bits, and before 0.45 it was refused at every precision the
//! configuration allows.  When the argument is an exact rational within
//! `2^−10` of `±1` (of 2 for `erfcinv`), or a sum `±1 + t` (`2 + t`) whose
//! term `t` is that small, the complement `u` is formed exactly from the
//! rational, or is `∓t` with `t`'s own bound, and `erfcinv(u)` evaluated:
//! `erfinv(±(1 − u)) = ±erfcinv(u)`, `erfcinv(2 − u) = −erfcinv(u)`.  The
//! derivative in `u` is `(√π/2)·e^(x²) < 1/(2ux)` for `x = erfcinv(u)` (from
//! `erfc x < e^(−x²)/(x√π)`), over the ball `1/(2(u − r)x)`, with no bound
//! when the ball reaches within eight radii of 0.

use super::ball::{CBall, Cache};
use super::wp::{RM, Wp};
use crate::base::bigfloat::{is_strictly_positive, to_f64};
use astro_float::{BigFloat, Consts};

use super::accuracy::{self, Bound, Cut};
use super::cexpint::{self, TrigInt};
use crate::base::arena::Arena;
use crate::base::bigcomplex::{Complex, c_div, c_mul, c_one, c_sub, c_zero};
use crate::base::errors::SymplexError;
use crate::base::libfn::LibFn;
use crate::base::node::{ExprId, ExprNode};

const LOG2E: f64 = std::f64::consts::LOG2_E;
const PI: f64 = std::f64::consts::PI;

/// A special function evaluated by the complex routines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Fun {
    Ei,
    Li,
    Trig(TrigInt),
    ExpInt,
    UpperGamma,
    LowerGamma,
    Bessel(super::cbessel::Kind),
    Airy(super::AiryKind),
    Binomial,
    FresnelS,
    FresnelC,
}

fn nonreal(cache: Cache<'_>, c: ExprId) -> bool {
    cache.get(&c).is_some_and(|v| !v.1.is_zero())
}

fn negative_real(cache: Cache<'_>, c: ExprId) -> bool {
    cache
        .get(&c)
        .is_some_and(|v| v.1.is_zero() && v.0.is_negative() && !v.0.is_zero())
}

fn integer_real(cache: Cache<'_>, c: ExprId) -> bool {
    cache
        .get(&c)
        .is_some_and(|v| v.1.is_zero() && (v.0.is_zero() || v.0.is_int()))
}

/// The function of `node` and its arguments, when the complex routines
/// evaluate it (module documentation).
pub(super) fn classify(node: &ExprNode, cache: Cache<'_>) -> Option<(Fun, Vec<ExprId>)> {
    use super::cbessel::Kind as B;
    let (f, args): (Fun, Vec<ExprId>) = match node {
        ExprNode::Ei(c) if nonreal(cache, *c) => (Fun::Ei, vec![*c]),
        ExprNode::Li(c) if nonreal(cache, *c) || negative_real(cache, *c) => (Fun::Li, vec![*c]),
        ExprNode::Si(c) if nonreal(cache, *c) => (Fun::Trig(TrigInt::Si), vec![*c]),
        ExprNode::Ci(c) if nonreal(cache, *c) || negative_real(cache, *c) => {
            (Fun::Trig(TrigInt::Ci), vec![*c])
        }
        ExprNode::Binomial(n, k) if nonreal(cache, *n) || nonreal(cache, *k) => {
            (Fun::Binomial, vec![*n, *k])
        }
        ExprNode::Lib(f, a) => {
            let f = *f;
            let any_nonreal = a.iter().any(|&c| nonreal(cache, c));
            let f = match (f, a.len()) {
                (LibFn::Shi, 1) if any_nonreal => Fun::Trig(TrigInt::Shi),
                (LibFn::Chi, 1) if any_nonreal || negative_real(cache, a[0]) => {
                    Fun::Trig(TrigInt::Chi)
                }
                (LibFn::ExpInt, 2) if any_nonreal || negative_real(cache, a[1]) => Fun::ExpInt,
                (LibFn::UpperGamma, 2) if any_nonreal || negative_real(cache, a[1]) => {
                    Fun::UpperGamma
                }
                (LibFn::LowerGamma, 2)
                    if any_nonreal
                        || negative_real(cache, a[1])
                        || (negative_real(cache, a[0]) && !integer_real(cache, a[0])) =>
                {
                    Fun::LowerGamma
                }
                (LibFn::BesselJ | LibFn::BesselY | LibFn::BesselI | LibFn::BesselK, 2) => {
                    let kind = match f {
                        LibFn::BesselJ => B::J,
                        LibFn::BesselY => B::Y,
                        LibFn::BesselI => B::I,
                        _ => B::K,
                    };
                    let negative_cut = negative_real(cache, a[1])
                        && (matches!(kind, B::Y | B::K) || !integer_real(cache, a[0]));
                    if !(any_nonreal || negative_cut) {
                        return None;
                    }
                    Fun::Bessel(kind)
                }
                (LibFn::AiryAi, 1) if any_nonreal => Fun::Airy(super::AiryKind::Ai),
                (LibFn::AiryBi, 1) if any_nonreal => Fun::Airy(super::AiryKind::Bi),
                (LibFn::AiryAiPrime, 1) if any_nonreal => Fun::Airy(super::AiryKind::AiPrime),
                (LibFn::AiryBiPrime, 1) if any_nonreal => Fun::Airy(super::AiryKind::BiPrime),
                (LibFn::FresnelS, 1) if any_nonreal => Fun::FresnelS,
                (LibFn::FresnelC, 1) if any_nonreal => Fun::FresnelC,
                _ => return None,
            };
            (f, a.to_vec())
        }
        _ => return None,
    };
    args.iter()
        .all(|a| cache.contains_key(a))
        .then_some((f, args))
}

/// The value of `f` at `args` (rounded to `prec + 16` bits, its error below
/// `2^−(prec+12)` of its modulus, or of each part where a [`CutRule`] gave
/// it).
pub(super) fn eval(f: Fun, args: &[&Complex], w: &mut Wp<'_>) -> Result<Complex, SymplexError> {
    eval_routed(f, args, w).map(|r| r.value)
}

/// A value of [`eval`] with the [`CutRule`] that gave its parts (`None`:
/// the general route, one bound for both parts).
pub(super) struct Routed {
    pub(super) value: Complex,
    cut: Option<CutRule>,
}

/// [`eval`], saying which route gave the value ([`bound`] needs it).
pub(super) fn eval_routed(
    f: Fun,
    args: &[&Complex],
    w: &mut Wp<'_>,
) -> Result<Routed, SymplexError> {
    if let Some(rule) = cut_rule(f, args)
        && let Some(value) = on_cut(rule, f, args, w)?
    {
        return Ok(Routed {
            value,
            cut: Some(rule),
        });
    }
    Ok(Routed {
        value: general(f, args, w)?,
        cut: None,
    })
}

/// The general route of `f` (the module documentation of `cexpint.rs`,
/// `cbessel.rs`, `cerf.rs`).
fn general(f: Fun, args: &[&Complex], w: &mut Wp<'_>) -> Result<Complex, SymplexError> {
    match f {
        Fun::Ei => cexpint::ei(args[0], w),
        Fun::Li => cexpint::li(args[0], w),
        Fun::Trig(kind) => cexpint::trig_integral(kind, args[0], w),
        Fun::ExpInt => cexpint::expint(args[0], args[1], w),
        Fun::UpperGamma => cexpint::upper_gamma(args[0], args[1], w),
        Fun::LowerGamma => cexpint::lower_gamma(args[0], args[1], w),
        Fun::Bessel(kind) => super::cbessel::bessel(kind, args[0], args[1], w),
        Fun::Airy(kind) => super::cbessel::airy(kind, args[0], w),
        Fun::Binomial => binomial(args[0], args[1], w),
        Fun::FresnelS => super::cerf::fresnel(true, args[0], w),
        Fun::FresnelC => super::cerf::fresnel(false, args[0], w),
    }
}

// ── binomial ────────────────────────────────────────────────────────────────

/// The parts of `z` in `f64` (a zero imaginary part as `+0`).
fn parts(z: &Complex) -> Option<(f64, f64)> {
    cexpint::f64_parts(z).map(|p| (p.x, p.y))
}

/// `x` as an integer exactly (`None` beyond `2^53` or not an integer).
fn exact_int(z: &Complex) -> Option<i64> {
    if !z.1.is_zero() || !(z.0.is_zero() || z.0.is_int()) {
        return None;
    }
    let v = to_f64(&z.0)?;
    (v.abs() < 9.0e15).then_some(v as i64)
}

/// Most factors of the falling factorial `a(a − 1)⋯(a − m + 1)/m!`.
const MAX_FALLING: i64 = 100_000;

/// `a(a − 1)⋯(a − m + 1)/m!`, `0 ≤ m ≤ MAX_FALLING`, to `p` bits relative.
fn falling_over_factorial(a: &Complex, m: i64, p: usize) -> Complex {
    let q = p + cexpint::guard(m as usize + 2);
    let mut v = c_one(q);
    for j in 0..m {
        let f = (super::zeta::plus_int(&a.0, -j, q), a.1.clone());
        v = c_mul(&v, &f, q, RM);
        let d = BigFloat::from_u64(j as u64 + 1, 64);
        v = (v.0.div(&d, q, RM), v.1.div(&d, q, RM));
    }
    v
}

/// `binomial(a, b)` with an argument not real: `Γ(a + 1)/(Γ(b + 1)Γ(a − b +
/// 1))`, or for an integer `b` (or `a − b`) `m ≥ 0` the falling factorial
/// `a(a−1)⋯(a−m+1)/m!`, `0` for a negative one (`1/Γ` vanishes at its poles:
/// mpmath's `binomial`).  Rounded to `prec + 16` bits, its error below
/// `2^−(prec+12)` of its modulus.
fn binomial(a: &Complex, b: &Complex, w: &mut Wp<'_>) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    let p = prec + 24;
    let exact = super::exact_bits(&a.0, p).max(super::exact_bits(&b.0, p));
    let amb = (a.0.sub(&b.0, exact, RM), a.1.sub(&b.1, p + 64, RM));
    for m in [exact_int(b), exact_int(&amb)].into_iter().flatten() {
        if m < 0 {
            return Ok(c_zero(prec + 16));
        }
        if m <= MAX_FALLING {
            let v = falling_over_factorial(a, m, p);
            return Ok((
                super::round_to(v.0, prec + 16),
                super::round_to(v.1, prec + 16),
            ));
        }
    }
    let gamma = |z: &Complex, cc: &mut Consts| -> Result<Complex, SymplexError> {
        if z.1.is_zero() {
            let zf = to_f64(&z.0).unwrap_or(f64::INFINITY).abs();
            let g = super::magnitude_bits((zf + 2.0) * ((zf + 2.0).ln() + 1.0));
            let v = super::arb_gamma_real(&z.0, &mut Wp::new(p + g + 16, cc))?;
            Ok((v, BigFloat::new(p)))
        } else {
            super::zeta::gamma_complex(z, &mut Wp::new(p, cc))
        }
    };
    let one = BigFloat::from_i32(1, 64);
    let a1 = (a.0.add(&one, super::exact_bits(&a.0, p), RM), a.1.clone());
    let b1 = (b.0.add(&one, super::exact_bits(&b.0, p), RM), b.1.clone());
    let amb1 = (
        amb.0.add(&one, super::exact_bits(&amb.0, p), RM),
        amb.1.clone(),
    );
    let ga = gamma(&a1, w.cc)?;
    let gb = gamma(&b1, w.cc)?;
    let gab = gamma(&amb1, w.cc)?;
    let den = c_mul(&gb, &gab, p, RM);
    if den.0.is_zero() && den.1.is_zero() {
        return Err(super::unevaluable(
            "binomial: Gamma(b + 1)·Gamma(a − b + 1) is zero",
        ));
    }
    let v = c_div(&ga, &den, p, RM);
    if [&v.0, &v.1].iter().any(|x| x.is_nan() || x.is_inf()) {
        return Err(super::unevaluable("binomial beyond the exponent range"));
    }
    Ok((
        super::round_to(v.0, prec + 16),
        super::round_to(v.1, prec + 16),
    ))
}

// ── On the cut ────────────────────────────────────────────────────────────────────────

/// How the parts of a function at real arguments whose last one is
/// negative are formed (module documentation, "On the cut").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CutRule {
    /// `E_n(−x)`, `1 ≤ n ≤ MAX_CUT_ORDER`: `Im = −π·x^(n−1)/(n−1)!`.
    ExpInt(u64),
    /// `E_n(−x)`, `n = k/2` for an odd `k`: `Im = (−1)^j·x^(n−1)·Γ(1 − n)`,
    /// `j = n − 3/2`.
    ExpIntHalf(i64),
    /// `Γ(−m, −x)`, `0 ≤ m ≤ MAX_CUT_ORDER`: `Im = (−1)^(m+1)·π/m!`.
    Gamma(u64),
    /// `Chi(−x) = Chi(x) + iπ`, `Ci(−x) = Ci(x) + iπ`.
    Trig(TrigInt),
    /// `K_n(−x) = (−1)^n·K_n(x) − iπ·I_n(x)`, `abs(n) ≤ MAX_CUT_ORDER`.
    BesselK(u64),
}

/// Largest integer order of a [`CutRule`] (as `cexpint.rs`'s finite sums).
const MAX_CUT_ORDER: u64 = 5_000;

/// The [`CutRule`] of `f` at `vals`: every argument real (an exact-0
/// imaginary part) and the last one negative, the order an integer.
fn cut_rule(f: Fun, vals: &[&Complex]) -> Option<CutRule> {
    use super::cbessel::Kind as B;
    let z = vals.last()?;
    if vals.iter().any(|v| !v.1.is_zero()) || !z.0.is_negative() || z.0.is_zero() {
        return None;
    }
    let order = || {
        exact_int(vals[0])
            .map(i64::unsigned_abs)
            .filter(|&n| n <= MAX_CUT_ORDER)
    };
    // `2a` for a half-integer order `a` (exactly: a binary exponent).
    let half = || {
        let mut t = vals[0].0.clone();
        if let Some(e) = t.exponent() {
            t.set_exponent(e.checked_add(1)?);
        }
        exact_int(&(t, BigFloat::new(64)))
            .filter(|k| k % 2 != 0 && k.unsigned_abs() <= 2 * MAX_CUT_ORDER)
    };
    let int = exact_int(vals[0]);
    match f {
        Fun::ExpInt if int.is_some_and(|n| n >= 1) => order().map(CutRule::ExpInt),
        Fun::UpperGamma if int.is_some_and(|a| a <= 0) => order().map(CutRule::Gamma),
        Fun::ExpInt if int.is_none() => half().map(CutRule::ExpIntHalf),
        Fun::Trig(kind @ (TrigInt::Chi | TrigInt::Ci)) => Some(CutRule::Trig(kind)),
        Fun::Bessel(B::K) => order().map(CutRule::BesselK),
        _ => None,
    }
}

/// `k/2` exactly.
fn half_of(k: i64) -> BigFloat {
    let mut b = BigFloat::from_i64(k, 64);
    if let Some(e) = b.exponent() {
        b.set_exponent(e - 1);
    }
    b
}

/// A part from a real routine, kept only when it is a nonzero finite number
/// (a 0 may be an underflow).
fn nonzero_part(x: Result<BigFloat, SymplexError>) -> Option<BigFloat> {
    x.ok()
        .filter(|v| !(v.is_zero() || v.is_nan() || v.is_inf()))
}

/// The value of `f` at `vals` by `rule` (rounded to `prec + 16` bits, each
/// part's error below `2^−(prec+12)` of that part, or of the modulus for a
/// part of the general route), or `None` where `K_n`'s real routines
/// refuse or give 0 (the general route then evaluates it).
fn on_cut(
    rule: CutRule,
    f: Fun,
    vals: &[&Complex],
    w: &mut Wp<'_>,
) -> Result<Option<Complex>, SymplexError> {
    let prec = w.bits();
    let p = prec + 16;
    let Some(z) = vals.last() else {
        return Ok(None);
    };
    let x = z.0.neg();
    let value = match rule {
        CutRule::ExpInt(n) => {
            let v = general(f, vals, w)?;
            // −π·x^(n−1)/(n−1)!, every factor to `q` bits.
            let q = p + cexpint::guard(2 * n as usize + 2) + 8;
            let mut t = w.cc.pi(q, RM).clone();
            for k in 1..n {
                t = t.mul(&x, q, RM).div(&BigFloat::from_u64(k, 64), q, RM);
            }
            (v.0, super::round_to(t.neg(), p))
        }
        CutRule::Gamma(m) => {
            let v = general(f, vals, w)?;
            let q = p + cexpint::guard(m as usize + 2) + 8;
            let mut t = w.cc.pi(q, RM).clone();
            for k in 2..=m {
                t = t.div(&BigFloat::from_u64(k, 64), q, RM);
            }
            (
                v.0,
                super::round_to(if m % 2 == 0 { t.neg() } else { t }, p),
            )
        }
        CutRule::ExpIntHalf(k) => {
            let v = general(f, vals, w)?;
            // (−1)^j·√x^(k−2)·Γ((2 − k)/2), every factor to `q` bits.
            let steps = (k - 2).unsigned_abs();
            let q = p + cexpint::guard(steps as usize + 4) + 8;
            let s = x.sqrt(q, RM);
            let mut t = BigFloat::from_i32(1, 64);
            for _ in 0..steps {
                t = t.mul(&s, q, RM);
            }
            if k < 2 {
                t = BigFloat::from_i32(1, 64).div(&t, q, RM);
            }
            let g = cexpint::gamma_of(&(half_of(2 - k), BigFloat::new(64)), &mut w.at(q))?;
            t = t.mul(&g.0, q, RM);
            if ((k - 3) / 2).rem_euclid(2) == 1 {
                t = t.neg();
            }
            (v.0, super::round_to(t, p))
        }

        CutRule::Trig(kind) => {
            // The real routines' errors are the node's, as on the real path.
            let re = match kind {
                TrigInt::Chi => super::arb_shi_chi(&x, false, &mut w.at(p))?,
                _ => super::arb_si_ci(&x, false, &mut w.at(p))?.1,
            };
            (re, w.cc.pi(p, RM).clone())
        }
        CutRule::BesselK(n) => {
            let order = BigFloat::from_u64(n, 64);
            let k = nonzero_part(super::arb_bessel_k(&order, &x, &mut w.at(p)));
            let i = nonzero_part(super::arb_bessel_i(&order, &x, &mut w.at(p)));
            let (Some(k), Some(i)) = (k, i) else {
                return Ok(None);
            };
            let q = p + 8;
            let im = i.mul(&w.cc.pi(q, RM), q, RM).neg();
            let re = if n % 2 == 1 { k.neg() } else { k };
            (super::round_to(re, p), super::round_to(im, p))
        }
    };
    Ok(Some(value))
}

/// `log₂` of the own error of each part of `v`, computed by `rule` at
/// `prec` bits (see [`on_cut`]): the general route's bound (relative to
/// the modulus, a hair more for the modulus it was relative to) on the part
/// it gave, the part's rounding on the others.
fn own_on_cut(rule: CutRule, v: &Complex, prec: usize) -> Bound {
    let own = |l: f64| l - prec as f64 - 12.0;
    let (re, im) = (accuracy::part_lg(&v.0), accuracy::part_lg(&v.1));
    let route = own(accuracy::lg_abs(v) + 1e-6);
    match rule {
        CutRule::ExpInt(_) | CutRule::ExpIntHalf(_) | CutRule::Gamma(_) => Bound {
            re: route,
            im: own(im),
        },
        CutRule::Trig(_) | CutRule::BesselK(_) => Bound {
            re: own(re),
            im: own(im),
        },
    }
}

/// The error of each part of `v` (value of `f` at `vals` by `rule`) from
/// the error `2^e` of its last argument, which is exactly real: each part's
/// derivative along the cut (module documentation), `None` without a
/// bound.
fn along_cut(
    rule: CutRule,
    f: Fun,
    vals: &[&Complex],
    v: &Complex,
    e: f64,
    w: &mut Wp<'_>,
) -> Option<Bound> {
    let z = vals.last()?;
    let x = -to_f64(&z.0)?;
    let r = e.exp2();
    if !(x.is_finite() && 8.0 * r < x) {
        return None;
    }
    let up = |l: f64| l + 1e-9 * l.abs() + 1e-9;
    // `abs(P′) ≤ c·abs(P)/(x − r)·(1 + r/(x − r))^c` over the ball for a part
    // `P = C·x^(±c)`.
    let power = |c: f64| {
        let d =
            accuracy::part_lg(&v.1) + c.log2() - (x - r).log2() + c * (r / (x - r)).ln_1p() * LOG2E;
        up(d) + e + 1.0
    };
    let closed = match rule {
        CutRule::ExpInt(n) if n >= 2 => power((n - 1) as f64),
        // `n − 1 = (k − 2)/2`.
        CutRule::ExpIntHalf(k) => power((k - 2).unsigned_abs() as f64 / 2.0),
        CutRule::ExpInt(_) | CutRule::Gamma(_) | CutRule::Trig(_) => accuracy::EXACT,
        CutRule::BesselK(_) => return cut_change(rule, f, vals, v, e, w),
    };
    let d = derivative(f, vals, v, r)?;
    if !d.is_finite() {
        return None;
    }
    Some(Bound {
        re: d + e + 1.0,
        im: closed,
    })
}

/// The change of each part of `v` (value of `f` at `vals` by `rule`) when
/// the last argument moves by `δ = max(2^e, 4 ulps)` either way, scaled
/// back to `2^e` and doubled (as `sensitivity::numeric_change`, per part,
/// the larger of the two sides); `None` when a side does not evaluate by
/// `rule` or `δ` reaches 0.
fn cut_change(
    rule: CutRule,
    f: Fun,
    vals: &[&Complex],
    v: &Complex,
    e: f64,
    w: &mut Wp<'_>,
) -> Option<Bound> {
    let prec = w.bits();
    let z = vals.last()?;
    let floor = accuracy::mag(z).map_or(e, |m| (m - prec as i64 + 2) as f64);
    let pe = e.max(floor).ceil();
    if !pe.is_finite() || pe.abs() > 2e9 || accuracy::mag(z).is_none_or(|m| pe + 3.0 > m as f64) {
        return None;
    }
    let mut delta = BigFloat::from_i32(1, 64);
    delta.set_exponent(i32::try_from(pe as i64 + 1).ok()?);
    let hp = prec + 64;
    let mut out = Bound::EXACT;
    for moved in [z.0.add(&delta, hp, RM), z.0.sub(&delta, hp, RM)] {
        let mz = (moved, BigFloat::new(64));
        let mut mv = vals.to_vec();
        *mv.last_mut()? = &mz;
        let u = on_cut(rule, f, &mv, w).ok()??;
        let d = c_sub(&u, v, hp, RM);
        let scale = |x: &BigFloat| {
            if x.is_zero() {
                accuracy::EXACT
            } else {
                accuracy::part_lg(x) - (pe - e) + 1.0
            }
        };
        // (`EXACT` is `−∞`: the larger exponent is the larger error.)
        out = Bound {
            re: out.re.max(scale(&d.0)),
            im: out.im.max(scale(&d.1)),
        };
    }
    Some(out)
}

/// Is `f` real at the real values `vals` (an exact-0 imaginary part of its
/// route is then its value): off its cut, entire in its last argument, or
/// `E_n` of an integer `n ≤ 0` (`z^(n−1)` times an entire function).
fn real_valued(f: Fun, vals: &[&Complex]) -> bool {
    let Some(z) = vals.last() else {
        return false;
    };
    if vals.iter().any(|v| !v.1.is_zero()) {
        return false;
    }
    match cut_of(f, vals) {
        None => true,
        Some(Cut::NegativeReal) => {
            is_strictly_positive(&z.0)
                || (f == Fun::ExpInt && exact_int(vals[0]).is_some_and(|n| n <= 0))
        }
        Some(_) => false,
    }
}

// ── Error bounds ────────────────────────────────────────────────────────────────────────

/// `log₂ cosh t`, also for large `t`.
fn log2_cosh(t: f64) -> f64 {
    let t = t.abs();
    if t > 30.0 {
        t * LOG2E - 1.0
    } else {
        t.cosh().log2()
    }
}

/// `log₂(sinh t/t)` for `t ≥ 0` (below `10⁻⁸` it is under `3·10⁻¹⁷`,
/// inside the margin `derivative` adds).
fn log2_sinhc(t: f64) -> f64 {
    if t < 1e-8 {
        0.0
    } else if t > 30.0 {
        t * LOG2E - 1.0 - t.log2()
    } else {
        (t.sinh() / t).log2()
    }
}

/// The bound on `log₂ abs(f′)` over the ball of radius `r` around `z = x + iy`
/// (the closed forms of the module documentation), `None` without one
/// (then the derivative is measured numerically), `Some(+∞)` where the ball
/// reaches a singular point.
fn derivative(f: Fun, vals: &[&Complex], v: &Complex, r: f64) -> Option<f64> {
    let z = vals.last()?;
    let (x, y) = parts(z)?;
    let az = x.hypot(y);
    // Within eight radii of 0 (or a NaN): no bound.
    let near0 = (r * 8.0).partial_cmp(&az) != Some(std::cmp::Ordering::Less);
    let up = |l: f64| l + 1e-9 * l.abs() + 1e-9;
    let d = match f {
        Fun::Ei => {
            if near0 {
                return Some(f64::INFINITY);
            }
            (x + r) * LOG2E - (az - r).log2()
        }
        Fun::Li => {
            if near0 {
                return Some(f64::INFINITY);
            }
            let lnz = (az.ln()).hypot(y.atan2(x));
            let move_ln = r / (az - r) * 1.0001;
            let low = lnz - move_ln;
            let well_inside = low > 0.0 && 8.0 * move_ln <= lnz;
            if !well_inside {
                return Some(f64::INFINITY);
            }
            -low.log2()
        }
        Fun::Trig(kind) => {
            let across = match kind {
                TrigInt::Si | TrigInt::Ci => y.abs(),
                _ => x.abs(),
            };
            let far = if near0 {
                f64::INFINITY
            } else {
                log2_cosh(across + r) - (az - r).log2()
            };
            match kind {
                TrigInt::Si | TrigInt::Shi => far.min(log2_sinhc(az + r)),
                _ => far,
            }
        }
        Fun::ExpInt => {
            if near0 {
                return Some(f64::INFINITY);
            }
            // `abs(n − 1)` from the exact difference (an `f64` `n` of 1 only
            // says `abs(n − 1) < 2⁻⁵³`).
            let n = vals[0];
            let lnm1 = accuracy::lg_abs(&(super::zeta::plus_int(&n.0, -1, 128), n.1.clone()));
            let lv = accuracy::lg_abs(v);
            let a = if lnm1 == f64::NEG_INFINITY || lv == f64::NEG_INFINITY {
                f64::NEG_INFINITY
            } else {
                let nm1 = lnm1.exp2();
                lnm1 + lv + (r * (1.0 + nm1 / (az - r)) * 2.0) * LOG2E
            };
            let b = (r - x) * LOG2E;
            lsum(a, b) - (az - r).log2()
        }
        Fun::UpperGamma | Fun::LowerGamma => {
            if near0 {
                return Some(f64::INFINITY);
            }
            let (ax, ay) = parts(vals[0])?;
            let am1 = (ax - 1.0).hypot(ay);
            let theta = y.atan2(x);
            // log₂ abs(z^(a−1) e^(−z)).
            let base = ((ax - 1.0) * az.ln() - ay * theta - x) * LOG2E;
            base + (am1 * r / (az - r) * 1.0001 + r) * LOG2E
        }
        Fun::FresnelS | Fun::FresnelC => log2_cosh(PI * (x.abs() + r) * (y.abs() + r)),
        Fun::Bessel(_) | Fun::Airy(_) | Fun::Binomial => return None,
    };
    Some(up(d))
}

// ── erfinv, erfcinv next to the ends ─────────────────────────────────────────────────

/// The complement of the argument of an `erfinv`/`erfcinv` node next to an
/// end of its domain (module documentation): `u`, its error exponent, and
/// whether the value is `−erfcinv(u)`.
struct EndComplement {
    u: BigFloat,
    err: f64,
    negate: bool,
}

/// Largest complement taken this way (`2^−10`).
const END_COMPLEMENT: f64 = 1.0 / 1024.0;

fn end_complement(
    arena: &Arena,
    node: &ExprNode,
    cache: Cache<'_>,
    prec: usize,
) -> Option<EndComplement> {
    use crate::base::numeric::Q;
    let ExprNode::Lib(f, args) = node else {
        return None;
    };
    let inverse_erfc = match f {
        LibFn::ErfInv => false,
        LibFn::ErfcInv => true,
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }
    let small =
        |u: &BigFloat| is_strictly_positive(u) && to_f64(u).is_some_and(|f| f < END_COMPLEMENT);
    match arena.node(args[0]) {
        ExprNode::Num(n) => {
            let q = arena.num(*n);
            let one = Q::from_integer(1.into());
            let (c, negate) = if inverse_erfc {
                (Q::from_integer(2.into()) - q, true)
            } else if *q.numer() < 0.into() {
                (one + q, true)
            } else {
                (one - q, false)
            };
            let u = super::ratio_to_bigfloat(&c, prec + 64);
            if !small(&u) {
                return None;
            }
            let err = accuracy::part_lg(&u) - (prec + 64) as f64 + 1.0;
            Some(EndComplement { u, err, negate })
        }
        ExprNode::Add(children) if children.len() == 2 => {
            // ±1 + t (erfinv) or 2 + t (erfcinv), with the constant exact.
            let (k, t) = match (arena.as_num(children[0]), arena.as_num(children[1])) {
                (Some(k), None) => (k.clone(), children[1]),
                (None, Some(k)) => (k.clone(), children[0]),
                _ => return None,
            };
            let tv = cache.get(&t)?;
            let tb = cache.bound(&t)?;
            if tb.is_unknown() || !accuracy::exactly_real(tv, tb) {
                return None;
            }
            let (u, negate) = if inverse_erfc && k == Q::from_integer(2.into()) {
                (tv.0.neg(), true)
            } else if !inverse_erfc && k == Q::from_integer(1.into()) {
                (tv.0.neg(), false)
            } else if !inverse_erfc && k == Q::from_integer((-1).into()) {
                (tv.0.clone(), true)
            } else {
                return None;
            };
            small(&u).then_some(EndComplement {
                u,
                err: tb.re,
                negate,
            })
        }
        _ => None,
    }
}

/// `erfinv`/`erfcinv` of node `node` next to an end of its domain, with its
/// bound (module documentation), or `None` when the argument is not of that
/// form.  (`erfcinv(u) > 2.3` for `u < 2^−10`.)
pub(super) fn inverse_erf_end(
    arena: &Arena,
    node: &ExprNode,
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Option<Result<CBall, SymplexError>> {
    let prec = w.bits();
    let end = end_complement(arena, node, cache, prec)?;
    let lu = accuracy::part_lg(&end.u);
    let well_inside = end.err + 3.0 <= lu;
    if !accuracy::is_exact(end.err) && !well_inside {
        // The ball reaches within eight radii of 0: no bound yet.
        return Some(Ok(CBall::new(super::c_zero(prec), Bound::UNKNOWN)));
    }
    let x = match super::arb_erfcinv_small(&end.u, &mut w.at(prec + 8)) {
        Ok(x) => x,
        Err(e) => return Some(Err(e)),
    };
    let lx = accuracy::part_lg(&x);
    let own = lx - prec as f64 + 1.0;
    let total = if accuracy::is_exact(end.err) {
        own
    } else {
        // 1/(2(u − r)x), with a quarter more for the change of x over the
        // ball.
        // In log₂ (u may lie far below the `f64` range).
        let xf = to_f64(&x).unwrap_or(0.0);
        let shrink = (1.0 - (end.err - lu).exp2()).log2();
        if !(xf > 1.0 && shrink.is_finite()) {
            return Some(Ok(CBall::new(super::c_zero(prec), Bound::UNKNOWN)));
        }
        let d = -(1.0 + lu + shrink + xf.log2()) + 0.33;
        accuracy::lsum(own, d + end.err + 1.0)
    };
    let v = super::round_to(x, prec);
    let v = if end.negate { v.neg() } else { v };
    Some(Ok(CBall::new((v, BigFloat::new(prec)), Bound::real(total))))
}

/// `log₂(2^a + 2^b)`.
fn lsum(a: f64, b: f64) -> f64 {
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    if hi == f64::NEG_INFINITY || !hi.is_finite() {
        return hi;
    }
    hi + (lo - hi).exp2().ln_1p() * LOG2E
}

/// The cut of `f` in its last argument, given the other arguments (none for
/// an entire function or an integer order of `J`, `I`).
fn cut_of(f: Fun, vals: &[&Complex]) -> Option<Cut> {
    use super::cbessel::Kind as B;
    let int = |z: &Complex| exact_int(z).is_some();
    match f {
        Fun::Trig(TrigInt::Si | TrigInt::Shi)
        | Fun::Airy(_)
        | Fun::Binomial
        | Fun::FresnelS
        | Fun::FresnelC => None,
        Fun::Li => Some(Cut::RealBelowOne),
        Fun::Bessel(B::J | B::I) if int(vals[0]) => None,
        Fun::UpperGamma | Fun::LowerGamma if exact_int(vals[0]).is_some_and(|n| n > 0) => None,
        _ => Some(Cut::NegativeReal),
    }
}

/// The error bound of node `id` (function `f` of `args`, value `routed`).
#[allow(clippy::too_many_arguments)]
pub(super) fn bound(
    arena: &Arena,
    id: ExprId,
    f: Fun,
    args: &[ExprId],
    routed: &Routed,
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Bound {
    let prec = w.bits();
    let v = &routed.value;
    if [&v.0, &v.1].iter().any(|x| x.is_nan() || x.is_inf()) {
        return Bound::UNKNOWN;
    }
    let mut vals: Vec<&Complex> = Vec::with_capacity(args.len());
    let mut bs: Vec<Bound> = Vec::with_capacity(args.len());
    for a in args {
        let (Some(val), Some(b)) = (cache.get(a), cache.bound(a)) else {
            return Bound::UNKNOWN;
        };
        if b.is_unknown() {
            return Bound::UNKNOWN;
        }
        vals.push(val);
        bs.push(b);
    }
    let lv = accuracy::lg_abs(v);
    let own = lv - prec as f64 - 12.0;
    let all_real = vals
        .iter()
        .zip(&bs)
        .all(|(z, b)| accuracy::exactly_real(z, *b));
    // An exact-0 imaginary part is the value only where `f` is real.
    let real = all_real && v.1.is_zero() && real_valued(f, &vals);
    let finish = |b: Bound| if real { Bound::real(b.re) } else { b };
    let mut total = match routed.cut {
        Some(rule) => own_on_cut(rule, v, prec),
        None => Bound::both(own),
    };
    if bs.iter().all(|b| b.is_exact()) {
        return finish(total);
    }
    // The side of the cut must be decided.
    let last = vals.len() - 1;
    if let Some(cut) = cut_of(f, &vals)
        && accuracy::cut_undecided(vals[last], bs[last], cut)
    {
        return Bound::UNKNOWN;
    }
    for (i, a) in args.iter().enumerate() {
        if bs[i].is_exact() || args[..i].contains(a) {
            continue;
        }
        let e = bs[i].joint();
        // Each part along the cut (the last argument is exactly real: its
        // side was decided above).
        if i == last
            && let Some(rule) = routed.cut
            && accuracy::exactly_real(vals[last], bs[last])
        {
            let Some(c) = along_cut(rule, f, &vals, v, e, w) else {
                return Bound::UNKNOWN;
            };
            total = Bound {
                re: accuracy::lsum(total.re, c.re),
                im: accuracy::lsum(total.im, c.im),
            };
            continue;
        }
        let closed = if i == last {
            derivative(f, &vals, v, e.exp2())
        } else {
            None
        };
        let c = match closed {
            Some(d) if d.is_finite() => d + e + 1.0,
            Some(_) => return Bound::UNKNOWN,
            None => match super::sensitivity::numeric_change(arena, id, *a, e, v, cache, w) {
                Some(k) => k,
                None => return Bound::UNKNOWN,
            },
        };
        total = Bound {
            re: accuracy::lsum(total.re, c),
            im: accuracy::lsum(total.im, c),
        };
    }

    if total.is_unknown() {
        return Bound::UNKNOWN;
    }
    finish(total)
}

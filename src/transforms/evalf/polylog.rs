//! `polylog(s, z)` of a real order `s` at a non-real argument or a real one
//! outside `[−1, 1]`, and the error bound of the node.  (`arb_polylog` in
//! `evalf.rs` serves a real `z ∈ [−1, 1]`.)
//!
//! The branch is the principal one, analytic in `ℂ ∖ [1, ∞)`; on the cut
//! (an exactly real `z > 1`) the value is the limit from below,
//! `Li_s(x − i0)`: SymPy's and mpmath's convention (`polylog(3/2, 3) =
//! 0.87749… − 3.71558…i`, `polylog(1, 3) = −ln 2 − iπ`), the continuity
//! counter-clockwise around the branch point 1.  A real `z < −1` gives an
//! exactly real value.
//!
//! Three expansions, chosen by the cost of the defining series:
//!
//! * the series `Σ_{k≥1} z^k k^{−s}` where it converges fast — `|z| ≤ 3/4`,
//!   or `|z| ≤ 1` for a large order — stopped by a bound on its tail: the
//!   next term over `1 − q`, `q` the largest ratio of consecutive terms
//!   beyond (and `2·k^{1−s}/(s − 1)` on the unit circle).  `k^{−s}` is a
//!   power only for a prime `k`, a product of two earlier ones otherwise;
//! * for `3/4 < |z| < 2` the expansion in `μ = ln z` (`|μ| ≤ 3.3 < 2π`):
//!   `Li_s(e^μ) = Γ(1−s)(−μ)^{s−1} + Σ_{k≥0} ζ(s−k) μ^k/k!`, and for an
//!   integer `s = n ≥ 1` `μ^{n−1}/(n−1)!·(H_{n−1} − ln(−μ)) + Σ_{k≠n−1}
//!   ζ(n−k) μ^k/k!` (DLMF 25.12.12; principal branches of `(−μ)^{s−1}` and
//!   `ln(−μ)`, which put an exactly real `z > 1` below the cut).  Beyond
//!   `σ = s − k ≤ −1` the functional equation bounds `|ζ(σ)| ≤
//!   2(2π)^{σ−1}Γ(1−σ)ζ(1−σ)`, and the tail by a geometric series of ratio
//!   `|μ|/2π·max(1, (k+1−s)/(k+1))`.  `ln|z|` comes from `|z|² − 1`
//!   formed exactly and `ln(1 + d) = 2 atanh(d/(2 + d))`: next to `z = 1`
//!   `ln|z|` of a rounded `|z|` would lose `log₂(1/|z − 1|)` bits of `μ`,
//!   and `(−μ)^{s−1}` with them;
//! * beyond, the inversion formula (DLMF 25.12.13, written with a Hurwitz
//!   argument `a`, `e^{2πia} = z`, in the strip `0 < Re a ≤ 1`)
//!
//!   ```text
//!   Li_s(z) = −e^{iπs} Li_s(1/z) + (2π)^s/Γ(s)·e^{iπs/2}·ζ(1 − s, a),  a = ½ + ln(−z)/(2πi)
//!   ```
//!
//!   for `|z| > 1` with the principal `ln(−z)` (the Hurwitz argument of
//!   mpmath's `polylog_general`; checked against mpmath in every quadrant,
//!   on the cut and on both sides of it): an exactly real `z > 1` has
//!   `Re a = 1`, the value below the cut.  `Re a` is taken from `arg z` itself (`arg z/2π` above
//!   the real axis, `1 + arg z/2π` below and on the cut).  Then the series
//!   for `Li_s(1/z)`, and the Hurwitz zeta function `ζ(σ, a)` of complex
//!   `a` by Euler–Maclaurin from `w = a + N`, `|w| ≳ 0.15·wp`:
//!   `Σ_{k<N} (a+k)^{−σ} + w^{1−σ}/(σ−1) + w^{−σ}/2 + Σ_{j=1}^{M}
//!   B₂ⱼ/(2j)!·(σ)_{2j−1}·w^{1−σ−2j}`, with the remainder bounded by
//!   `4|(σ)_{2M}|/(2π)^{2M}·∫_N^∞ |a + t|^{−σ−2M} dt` (`|B̃₂ₘ| ≤ |B₂ₘ| ≤
//!   4(2M)!/(2π)^{2M}`) and the integral by `min((Re w)^{1−p}/(p−1),
//!   (π/2)|w|^{1−p})`, `p = σ + 2M ≥ 2` (`|a + N + τ|² ≥ |w|² + τ²`).  The
//!   formula holds for an integer order too, where `ζ(1 − n, a) = −B_n(a)/n`
//!   and the Euler–Maclaurin sum ends by itself.
//!
//! A large order (`s ≥ 24`) takes the expansion in `μ` wherever `abs(μ) ≤
//! 0.85·2π`, `abs(z) > 1` included: its terms do not cancel (they are about
//! `μ^k/k!`), and the sum stops at the first `k` where a bound on all the
//! rest ([`mu_tail`]) is below the precision, long before `k = s`; the
//! `ζ(s − k)` of large arguments are direct sums ([`zeta_direct`]).  Before
//! 0.37 the inversion took `abs(z) ≥ 2`, whose Hurwitz sum cancels about
//! `1.77·s` bits (`polylog(300, 3)` 0.36 s, `polylog(1000, 3)` refused), and
//! the expansion summed every `ζ(s − k)`, `k < s`.  On the cut the imaginary
//! part is its closed form `−π·(ln x)^(s−1)/Γ(s)` ([`cut_imaginary`]),
//! correct relative to itself, with a bound of its own.  A complex order
//! inside the unit disc is the defining series ([`polylog_complex_order`]).
//!
//! A non-positive integer order is the finite Stirling sum ([`nonpositive`])
//! down to −1000, and beyond the sum over the poles `n!·Σ_k (2πik − Log
//! z)^(−n−1)` ([`nonpositive_poles`]), here and for `z ∈ [−1, 1]` (whose
//! `arb_polylog` comes here for those orders).
//!
//! Every truncation is below `2^−wp` of the largest term summed; the loss —
//! the largest term over the result — is measured and the evaluation
//! repeated with that many more bits, up to `cancellation_cap` (the poles
//! of `Γ(1 − s)` and `ζ(s − n + 1)` next to an integer order cancel in the
//! `μ`-expansion; `Li_s` has zeros off the real axis).
//!
//! The regimes and formulas follow mpmath's `polylog`, `polylog_series` and
//! `polylog_general` (`mpmath/functions/zeta.py`, BSD), which use the
//! series for `|z| < 0.9`, the `μ`-expansion for `|ln z| < 5` and Jonquière's
//! two-Hurwitz form beyond; here the inversion takes one Hurwitz function
//! and the series of `1/z`, and every truncation has a bound.

use super::ball::Cache;
use super::wp::{RM, Wp};
use crate::base::bigfloat::{self, is_strictly_positive, to_f64};
use astro_float::{BigFloat, Consts};
use num_bigint::BigInt;
use num_rational::Ratio;

use super::accuracy::{self, Bound};
use crate::base::bigcomplex::{Complex, c_add, c_div, c_mul, c_neg, c_one, c_sub, c_zero};
use crate::base::errors::SymplexError;

const LN2: f64 = std::f64::consts::LN_2;
const LOG2E: f64 = std::f64::consts::LOG2_E;
/// `log₂ 2π`.
const LOG2_TWO_PI: f64 = 2.651_496_129_472_319;
/// `log₂(3/4)`.
const LOG2_THREE_QUARTERS: f64 = -0.415_037_499_278_843_8;
/// `log₂(π/2)`.
const LOG2_HALF_PI: f64 = 0.651_496_129_472_318_8;
/// Precision of the auxiliary evaluations of the error bound (`Li_{s−1}`,
/// and `Li_s` at a moved order).
const LP: usize = 128;
/// Largest `|s|` evaluated (the Hurwitz sum needs `|w| ≳ |s|/5` terms).
const MAX_ORDER: f64 = 1.0e6;

/// A sum and `log₂` of the largest term that went into it (its rounding
/// error is a few units of `2^−wp` of that term).
struct Sum {
    value: Complex,
    largest: f64,
}

impl Sum {
    fn new(prec: usize) -> Sum {
        Sum {
            value: c_zero(prec),
            largest: f64::NEG_INFINITY,
        }
    }

    fn add(&mut self, t: &Complex, wp: usize) {
        self.largest = self.largest.max(accuracy::lg_abs(t));
        self.value = c_add(&self.value, t, wp, RM);
    }
}

/// Is the value `z` of `polylog`'s argument outside the real interval
/// `[−1, 1]` (non-real, or `|z| > 1`), where this module evaluates it?
pub(super) fn off_unit_interval(z: &Complex) -> bool {
    !z.1.is_zero() || super::bf_gt(&z.0.abs(), &BigFloat::from_i32(1, 64))
}

/// Is `node` a `polylog(s, z)` with a real order and an argument this module
/// evaluates, both values in `cache`?
pub(super) fn is_general(node: &crate::base::node::ExprNode, cache: Cache<'_>) -> bool {
    let crate::base::node::ExprNode::Lib(crate::base::libfn::LibFn::PolyLog, args) = node else {
        return false;
    };
    args.len() == 2
        && cache.get(&args[0]).is_some_and(|s| s.1.is_zero())
        && cache.get(&args[1]).is_some_and(off_unit_interval)
}

/// `Li_s(z)` for a real `s` and `z` off `[−1, 1]` (see the module
/// documentation), rounded to `prec + 16` bits: its absolute error is
/// below `2^−(prec+12)·|Li_s(z)|` (on the real part, for a real value).
pub(super) fn polylog_general(
    s: &BigFloat,
    z: &Complex,
    w: &mut Wp<'_>,
) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    if [s, &z.0, &z.1].iter().any(|x| x.is_nan() || x.is_inf()) {
        return Err(super::unevaluable("polylog of special float value"));
    }
    if z.0.is_zero() && z.1.is_zero() {
        return Ok(c_zero(prec));
    }
    if z.1.is_zero() && z.0 == BigFloat::from_i32(1, 64) {
        return Err(super::unevaluable("polylog(s, 1) is zeta(s)"));
    }
    let s_f = to_f64(s).unwrap_or(f64::INFINITY);
    if s_f.abs() > MAX_ORDER {
        return Err(super::unevaluable(
            "polylog of an order beyond 10^6 off [-1, 1] not supported in evalf",
        ));
    }
    let s_int = super::bf_as_int(s)?;
    let real = z.1.is_zero() && (z.0.is_negative() || s_int.is_some_and(|n| n <= 0));
    tracing::debug!(prec, s = s_f, "evalf: polylog off [-1, 1]");
    let cap = super::cancellation_cap(prec);
    let mut extra =
        (32 + accuracy::ceil_log2(prec) as usize + near_integer_bits(s, s_int, prec)).min(cap);
    loop {
        let wp = prec + extra;
        let sum = value_at(s, s_int, s_f, z, &mut w.at(wp)).map_err(super::requested_at(prec))?;
        let lv = if real {
            accuracy::part_lg(&sum.value.0)
        } else {
            accuracy::lg_abs(&sum.value)
        };
        let lost = if lv.is_finite() && sum.largest.is_finite() {
            ((sum.largest - lv).max(0.0).ceil() as usize).min(cap + 1)
        } else if lv.is_finite() {
            0
        } else {
            wp
        };
        if lost + 24 <= extra {
            let re = super::round_to(sum.value.0, prec + 16);
            let im = if real {
                BigFloat::new(prec + 16)
            } else if let Some(im) = cut_imaginary(s, z, &mut w.at(prec + 16)) {
                im
            } else {
                super::round_to(sum.value.1, prec + 16)
            };
            return Ok((re, im));
        }
        // The measured loss is at most the true one (a value swamped by
        // rounding noise measures the noise): beyond the cap it cannot be
        // paid for (`polylog(1000.5, 3)` spent 1.9 s on a second pass).
        if extra >= cap || lost + 24 > cap {
            return Err(super::special_exhausted(prec));
        }
        extra = (lost + 40).max(2 * extra).min(cap);
    }
}

/// Is `z` real and above 1 (on the cut, the value below it), with `s > 0`?
fn on_cut(s: &BigFloat, z: &Complex) -> bool {
    z.1.is_zero()
        && super::bf_gt(&z.0, &BigFloat::from_i32(1, 64))
        && super::bf_strictly_positive(s)
}

/// The imaginary part of `Li_s(x − i0)` for a real `x > 1` and a real `s > 0`,
/// `−π·(ln x)^(s−1)/Γ(s)` (the jump across the cut is `2πi·(ln x)^(s−1)/Γ(s)`,
/// the imaginary parts of `Γ(1 − s)(−μ)^(s−1)` and of `−μ^(n−1)/(n−1)!·ln(−μ)` in
/// the expansion in `μ = ln x`), to `prec` bits relative.  The expansions
/// give it only to `2^−wp` of the real part: before 0.37 `im(polylog(300,
/// 3))` (`−5.02·10⁻⁶⁰⁰`, mpmath at 200 digits) printed `0`.  `None` off the cut.
fn cut_imaginary(s: &BigFloat, z: &Complex, w: &mut Wp<'_>) -> Option<BigFloat> {
    let prec = w.bits();
    if !on_cut(s, z) {
        return None;
    }
    let sf = to_f64(s)?;
    let wp = prec + 16;
    // ln x relative to 2^(−wp−32) (next to 1 from x² − 1 formed exactly).
    let (lnx, _) = ln_near_one(z, &mut w.at(wp + 32));
    let lnln = lnx.ln(wp + 32, RM, w.cc);
    let size = (sf * (sf.ln().abs() + 2.0) + to_f64(&lnln)?.abs() * sf).max(1.0);
    let lp = wp + 32 + super::magnitude_bits(size) + super::magnitude_bits(sf.max(1.0));
    let lnln = lnx.ln(lp, RM, w.cc);
    let s1 = minus_int(s, 1, lp);
    let lgs = super::arb_log_gamma(s, &mut w.at(lp)).ok()?.0;
    let e = s1.mul(&lnln, lp, RM).sub(&lgs, lp, RM);
    let pi = w.cc.pi(wp, RM).clone();
    let v = e.exp(wp, RM, w.cc).mul(&pi, wp, RM).neg();
    (!v.is_zero() && !v.is_inf() && !v.is_nan()).then(|| super::round_to(v, prec))
}

/// `log₂(1/|s − n|)` for a non-integer `s` within `2⁻⁸` of an integer `n`
/// (else 0): the bits the `μ`-expansion loses to the cancelling poles of
/// `Γ(1 − s)` and `ζ(s − n + 1)`, taken as guard bits from the start (the
/// measured loss would cost a second pass).
fn near_integer_bits(s: &BigFloat, s_int: Option<i64>, prec: usize) -> usize {
    if s_int.is_some() {
        return 0;
    }
    let exact = super::exact_bits(s, prec);
    let n = s.add(&BigFloat::from_f64(0.5, 64), exact, RM).floor();
    let d = s.sub(&n, exact, RM);
    let l = -lg2(&d);
    if l.is_finite() && l > 8.0 {
        (l.ceil() as usize).min(4 * prec)
    } else {
        0
    }
}

/// One evaluation at working precision `wp`, by the regime of the module
/// documentation.
fn value_at(
    s: &BigFloat,
    s_int: Option<i64>,
    s_f: f64,
    z: &Complex,
    w: &mut Wp<'_>,
) -> Result<Sum, SymplexError> {
    let wp = w.bits();
    if let Some(n) = s_int
        && n <= 0
    {
        if n.unsigned_abs() > super::MAX_NONPOSITIVE_ORDER {
            return nonpositive_poles(n.unsigned_abs(), z, w);
        }
        return nonpositive(n.unsigned_abs(), z, wp);
    }
    let r2 = z.0.mul(&z.0, wp, RM).add(&z.1.mul(&z.1, wp, RM), wp, RM);
    let lz = 0.5 * lg2(&r2);
    let cheap = |r: f64| series_terms(r, s_f, wp).is_some_and(|k| k <= 3 * wp + 64);
    if lz <= LOG2_THREE_QUARTERS {
        return series(s, s_int, s_f, z, w);
    }
    if !super::bf_gt(&r2, &BigFloat::from_i32(1, 64)) {
        return if cheap(lz.exp2().min(1.0)) {
            series(s, s_int, s_f, z, w)
        } else {
            mu_expansion(s, s_int, s_f, z, w)
        };
    }
    // A large order: the expansion in `ln z` wherever it converges well (its
    // terms do not cancel, and they stop early, see `mu_tail`), rather than
    // the inversion, whose Hurwitz sum cancels about `1.77·s` bits.
    if s_f >= LARGE_ORDER && mu_ratio(z).is_some_and(|r| r <= MU_RATIO) {
        return mu_expansion(s, s_int, s_f, z, w);
    }
    if lz >= 1.0 || cheap((-lz).exp2()) {
        inversion(s, s_int, s_f, z, w)
    } else {
        mu_expansion(s, s_int, s_f, z, w)
    }
}

/// The order from which `abs(ln z) ≤ MU_RATIO·2π` takes the expansion in
/// `ln z` for `abs(z) > 1`.
const LARGE_ORDER: f64 = 24.0;

/// The largest `abs(ln z)/2π` the expansion in `ln z` takes for a large order.
const MU_RATIO: f64 = 0.85;

/// `abs(ln z)/2π` (principal logarithm) in `f64`, `None` beyond its range.
fn mu_ratio(z: &Complex) -> Option<f64> {
    let (x, y) = (to_f64(&z.0)?, to_f64(&z.1)?);
    let r = x.hypot(y);
    (r.is_finite() && r > 0.0).then(|| r.ln().hypot(y.atan2(x)) / (2.0 * std::f64::consts::PI))
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// `log₂|x|` (`−∞` for 0), also beyond the `f64` range.
fn lg2(x: &BigFloat) -> f64 {
    if x.is_zero() {
        return f64::NEG_INFINITY;
    }
    match to_f64(x) {
        Some(v) if v != 0.0 => v.abs().log2(),
        _ => f64::from(x.exponent().unwrap_or(0)),
    }
}

/// `z·x` for a real `x`.
fn scale(z: &Complex, x: &BigFloat, wp: usize) -> Complex {
    (z.0.mul(x, wp, RM), z.1.mul(x, wp, RM))
}

/// `z/n` for a positive integer `n`.
fn div_int(z: &Complex, n: u64, wp: usize) -> Complex {
    let d = BigFloat::from_u64(n, 64);
    (z.0.div(&d, wp, RM), z.1.div(&d, wp, RM))
}

/// `w^e = exp(e·Ln w)` (principal branch) for a real `e`, to `wp` bits
/// relative: the exponent is formed with the bits of its magnitude as guard.
fn pow_real(w: &Complex, e: &BigFloat, wpc: &mut Wp<'_>) -> Complex {
    let wp = wpc.bits();
    let ef = to_f64(e).unwrap_or(f64::INFINITY).abs();
    let lw = (accuracy::lg_abs(w) * LN2).abs() + 4.0;
    let p = wp + super::magnitude_bits(ef * lw) + 8;
    let l = super::c_ln(w, &mut wpc.at(p));
    let x = scale(&l, e, p);
    let v = super::c_exp(&x, &mut wpc.at(p));
    (super::round_to(v.0, wp), super::round_to(v.1, wp))
}

/// `x − c` exactly for a small integer `c`.
fn minus_int(x: &BigFloat, c: i64, wp: usize) -> BigFloat {
    x.sub(
        &BigFloat::from_i128(i128::from(c), 128),
        super::exact_bits(x, wp),
        RM,
    )
}

/// `e^{iπx}`, exact where `x` is an integer or half-integer.
fn exp_i_pi(x: &BigFloat, w: &mut Wp<'_>) -> Complex {
    let wp = w.bits();
    // x′ = x − 2·round(x/2), exactly (a binary x/2 is exact).
    let exact = super::exact_bits(x, wp);
    let half_x = x.div(&BigFloat::from_i32(2, 64), exact, RM);
    let n = half_x.add(&BigFloat::from_f64(0.5, 64), exact, RM).floor();
    let xr = x.sub(&n.mul(&BigFloat::from_i32(2, 64), exact, RM), exact, RM);
    let twice = xr.mul(&BigFloat::from_i32(2, 64), exact, RM);
    if twice.is_int() || twice.is_zero() {
        let q = to_f64(&twice).unwrap_or(0.0).round() as i64;
        let one = BigFloat::from_i32(1, wp);
        let zero = BigFloat::new(wp);
        return match q.rem_euclid(4) {
            0 => (one, zero),
            1 => (zero, one),
            2 => (one.neg(), zero),
            _ => (zero, one.neg()),
        };
    }
    let t = w.cc.pi(wp + 8, RM).clone().mul(&xr, wp + 8, RM);
    (t.cos(wp, RM, w.cc), t.sin(wp, RM, w.cc))
}

/// `ln Γ(x)` for `x ≥ 1` in `f64`: the recurrence up to 16 and Stirling's
/// series with three terms (error below `10⁻⁹`, far inside the margins it
/// is used with).
fn ln_gamma(x: f64) -> f64 {
    let mut z = x;
    let mut shift = 0.0;
    while z < 16.0 {
        shift += z.ln();
        z += 1.0;
    }
    let z2 = z * z;
    let series = 1.0 / (12.0 * z) - 1.0 / (360.0 * z * z2) + 1.0 / (1260.0 * z * z2 * z2);
    (z - 0.5) * z.ln() - z + 0.5 * (2.0 * std::f64::consts::PI).ln() + series - shift
}

/// An estimate of the number of terms of `Σ r^k k^{−s}` before they fall
/// below `2^−wp`, `None` when that is beyond reach (`r ≥ 1` with `s ≤ 1`).
/// It chooses the regime and sizes the term budget; the series itself
/// stops on a rigorous bound of its tail.
fn series_terms(r: f64, s: f64, wp: usize) -> Option<usize> {
    let target = wp as f64 * LN2 + 16.0;
    if r <= 0.0 {
        return Some(2);
    }
    let a = -r.ln();
    // (Next to the unit circle the term count only chooses another route.)
    if a <= 1e-12 {
        if s <= 1.25 {
            return None;
        }
        // Σ_{j>k} j^{−s} ≤ k^{1−s}/(s − 1).
        let k = ((target + (1.0 / (s - 1.0)).ln().max(0.0)) / (s - 1.0)).exp();
        return (k < 1e6).then(|| k.ceil() as usize + 2);
    }
    // r^k k^{−s} = exp(−(ka + s ln k)), increasing in k beyond −s/a.
    let mut k = if s < 0.0 { (-s / a).max(1.0) } else { 1.0 };
    while k * a + s * k.ln() < target {
        k *= 1.25;
        if k > 1e7 {
            return None;
        }
    }
    Some(k.ceil() as usize + 2)
}

/// Smallest prime factor of every `k ≤ n` (`spf[k] == k` for a prime).
fn smallest_prime_factors(n: usize) -> Vec<u32> {
    let mut spf = vec![0u32; n + 1];
    for p in 2..=n {
        if spf[p] == 0 {
            for m in (p..=n).step_by(p) {
                if spf[m] == 0 {
                    spf[m] = p as u32;
                }
            }
        }
    }
    spf
}

// ── Regimes ─────────────────────────────────────────────────────────────────

/// `Li_{−n}(z) = Σ_{k=0}^{n} k! S(n+1, k+1) w^{k+1}`, `w = z/(1 − z)`.
fn nonpositive(n: u64, z: &Complex, wp: usize) -> Result<Sum, SymplexError> {
    // The row of Stirling numbers at once (`O(n²)`; before 0.34 one number
    // at a time, `O(n³)`, up to an order of −5000).
    let row = super::stirling2_row(n + 1)?;
    let one = BigFloat::from_i32(1, 64);
    let omz = (one.sub(&z.0, super::exact_bits(&z.0, wp), RM), z.1.neg());
    if omz.0.is_zero() && omz.1.is_zero() {
        return Err(super::infinite("polylog(s, 1) diverges for s ≤ 1"));
    }
    let w = c_div(z, &omz, wp, RM);
    let mut k_fact = BigInt::from(1);
    let mut w_pow = w.clone();
    let mut sum = Sum::new(wp);
    for k in 0..=n {
        if k > 0 {
            k_fact *= BigInt::from(k);
            w_pow = c_mul(&w_pow, &w, wp, RM);
        }
        let c = super::ratio_to_bigfloat(&Ratio::from_integer(&k_fact * &row[k as usize + 1]), wp);
        sum.add(&scale(&w_pow, &c, wp), wp);
    }
    Ok(sum)
}

/// `Li_{−n}(z)` for an order below `−MAX_NONPOSITIVE_ORDER` by its poles:
///
/// ```text
/// Li_{−n}(e^μ) = n!·Σ_{k∈ℤ} (2πik − μ)^(−n−1),   μ = Log z,  n ≥ 1,
/// ```
///
/// Jonquière's formula (DLMF 25.12.13 with `ζ(n + 1, a)` as its sum; the
/// two-Hurwitz form of mpmath's `polylog_general`): both sides are
/// `2πi`-periodic in `μ` with the poles `n!·(2πik − μ)^(−n−1)` and vanish
/// as `Re μ → ±∞`, so they agree for every `z ∉ {0, 1}`.  For a large `n`
/// the terms nearest `μ` dominate: `abs(2πik − μ) ≥ π(2abs(k) − 1)`
/// (`abs(Im μ) ≤ π`), so the terms beyond `abs(k) ≤ K` add at most
/// `2·n!·(π(2K + 1))^(−n−1)·(1 + (2K + 1)/(2n))` (the first plus the
/// integral), taken below `2^(−wp−12)` of the term at `k = 0`.  Each term is
/// `exp(ln n! − (n + 1)·Log(2πik − μ))` with the bits of its exponent as
/// guard.  A real `z < 0` has two dominant conjugate terms (`k = 0, 1`),
/// whose cancellation the caller measures.  Before 0.35 the Stirling sum of
/// [`nonpositive`] was refused beyond `n = 1000`: `polylog(−1500, 1/2)`.
fn nonpositive_poles(n: u64, z: &Complex, w: &mut Wp<'_>) -> Result<Sum, SymplexError> {
    let wp = w.bits();
    if z.0.is_zero() && z.1.is_zero() {
        return Ok(Sum::new(wp));
    }
    let nf = n as f64;
    // ln n! is about n·ln n, and (n + 1)·Log(…) as large.
    let p = wp + 24 + super::magnitude_bits(nf * (nf.ln() + 64.0));
    let mu = super::c_ln(z, &mut w.at(p));
    let mu_abs = accuracy::lg_abs(&mu).exp2();
    // (Refuses.)
    if !mu_abs.is_finite() || mu_abs == 0.0 {
        return Err(super::infinite("polylog(s, 1) diverges for s ≤ 1"));
    }
    // The fewest K ≥ 1 whose tail is below 2^(−wp−12) of n!/abs(μ)^(n+1).
    let mut kmax: u64 = 1;
    loop {
        let width = (2 * kmax + 1) as f64;
        let margin = (nf + 1.0) * (std::f64::consts::PI * width / mu_abs).log2()
            - 1.0
            - (1.0 + width / (2.0 * nf)).log2();
        if margin >= (wp + 12) as f64 {
            break;
        }
        kmax += 1;
        if kmax > 100_000 {
            return Err(super::special_exhausted(wp));
        }
    }
    let n1 = BigFloat::from_u64(n + 1, 64);
    let ln_fact = super::arb_log_gamma(&n1, &mut w.at(p))?.0;
    let two_pi =
        w.cc.pi(p, RM)
            .clone()
            .mul(&BigFloat::from_i32(2, 64), p, RM);
    let neg_re = mu.0.neg();
    let mut sum = Sum::new(wp);
    let kmax = i64::try_from(kmax).unwrap_or(i64::MAX);
    for k in -kmax..=kmax {
        let im = two_pi
            .mul(&BigFloat::from_i64(k, 64), p, RM)
            .sub(&mu.1, p, RM);
        let l = super::c_ln(&(neg_re.clone(), im), &mut w.at(p));
        let e = (
            ln_fact.sub(&l.0.mul(&n1, p, RM), p, RM),
            l.1.mul(&n1, p, RM).neg(),
        );
        let t = super::c_exp(&e, w);
        if [&t.0, &t.1].iter().any(|v| v.is_inf() || v.is_nan()) {
            return Err(super::unevaluable(
                "polylog of a large negative order: the value overflows the arbitrary-precision \
                 exponent range",
            ));
        }
        sum.add(&t, wp);
    }
    Ok(sum)
}

/// The defining series `Σ_{k≥1} z^k k^{−s}` for `|z| ≤ 1` (`s > 1` on the
/// unit circle), stopped by a bound on its tail.
#[allow(clippy::too_many_arguments)]
fn series(
    s: &BigFloat,
    s_int: Option<i64>,
    s_f: f64,
    z: &Complex,
    w: &mut Wp<'_>,
) -> Result<Sum, SymplexError> {
    let wp = w.bits();
    let lr = accuracy::lg_abs(z);
    // (`lg_abs` is an upper bound: "on the circle" includes a margin, and
    // it only refuses this route.)
    let unit = lr >= -1e-9;
    if unit && s_f <= 1.25 {
        return Err(super::unevaluable("polylog series on the unit circle"));
    }
    let r = lr.exp2().min(1.0);
    let max_terms = series_terms(r, s_f, wp)
        .map(|k| (2 * k + 64).min(1 << 21))
        .ok_or_else(|| super::series_did_not_converge("polylog", 0))?;
    let spf = if s_int.is_none() {
        smallest_prime_factors(max_terms)
    } else {
        Vec::new()
    };
    let neg_s = s.neg();
    let one = BigFloat::from_i32(1, wp);
    let mut powers: Vec<BigFloat> = Vec::with_capacity(max_terms.min(4096));
    let mut zk = z.clone();
    let mut sum = Sum::new(wp);
    let target = |largest: f64| largest - wp as f64 - 4.0;
    for k in 1..=max_terms {
        if k > 1 {
            zk = c_mul(&zk, z, wp, RM);
        }
        let p = if k == 1 {
            one.clone()
        } else {
            match s_int {
                Some(n) => {
                    // n ≥ 1 here (the non-positive orders are rational).
                    let kn =
                        BigFloat::from_u64(k as u64, 64).powi(n.unsigned_abs() as usize, wp, RM);
                    one.div(&kn, wp, RM)
                }
                None => {
                    let q = spf[k] as usize;
                    if q == k {
                        super::bf_pow(&BigFloat::from_u64(k as u64, 64), &neg_s, w)
                    } else {
                        powers[q - 1].mul(&powers[k / q - 1], wp, RM)
                    }
                }
            }
        };
        let term = scale(&zk, &p, wp);
        if s_int.is_none() {
            powers.push(p);
        }
        sum.add(&term, wp);
        // The tail Σ_{j>k} |z|^j j^{−s}.
        let kf = k as f64;
        let next = (kf + 1.0) * lr.min(0.0) - s_f * (kf + 1.0).log2();
        let q = r * ((kf + 2.0) / (kf + 1.0)).powf(-s_f).max(1.0);
        let mut tail = if q < 1.0 {
            next - (1.0 - q).log2()
        } else {
            f64::INFINITY
        };
        if s_f > 1.0 {
            // |z|^j ≤ 2 for j ≤ 2^(wp−1) when |z| ≤ 1 + 2^−wp.
            tail = tail.min(1.0 + (1.0 - s_f) * kf.log2() - (s_f - 1.0).log2());
        }
        if tail < target(sum.largest) {
            return Ok(sum);
        }
    }
    Err(super::series_did_not_converge("polylog", max_terms))
}

/// Is `node` a `polylog(s, z)` with a non-real order and `abs(z) < 1`, both
/// values in `cache` ([`polylog_complex_order`])?
pub(super) fn is_complex_order(node: &crate::base::node::ExprNode, cache: Cache<'_>) -> bool {
    let crate::base::node::ExprNode::Lib(crate::base::libfn::LibFn::PolyLog, args) = node else {
        return false;
    };
    args.len() == 2
        && cache.get(&args[0]).is_some_and(|s| !s.1.is_zero())
        && cache.get(&args[1]).is_some()
}

/// `Li_s(z)` for a non-real order `s = σ + iτ`, by the regime of `abs(z)`:
///
/// * `abs(z) ≤ 3/4`, or below 1 while at most 200,000 terms do: the defining series
///   `Σ_{k≥1} z^k·k^(−s)`, `k^(−s) = e^(−s·ln k)` for a prime `k` and a product of
///   two earlier ones otherwise, stopped by the bound on its tail of the real
///   order `σ` (`abs(k^(−s)) = k^(−σ)`; see [`series`]);
/// * otherwise `abs(z) < 4/3`: the expansion in `μ = ln z`, `Γ(1 − s)(−μ)^(s−1) +
///   Σ_{k≥0} ζ(s − k)μ^k/k!` ([`complex_mu_expansion`]; `s` is never an integer);
/// * `abs(z) ≥ 4/3`: the inversion formula of the module documentation with
///   the series of `1/z` and the Hurwitz function of the complex order `1 − s`
///   ([`complex_inversion`]);
///
/// the loss to cancellation measured and made up as in [`polylog_general`].
/// `Li_s(1) = ζ(s)` for `σ > 1`.  Rounded to `prec + 16` bits, its absolute
/// error below `2^−(prec+12)·abs(Li_s(z))`.  Before 0.37 every complex order
/// was refused (`polylog(3/2 + i/2, 1/2)`), and before 0.44 every `abs(z) ≥ 1`
/// (`polylog(2 + i, 2)`, "not yet supported in evalf").
pub(super) fn polylog_complex_order(
    s: &Complex,
    z: &Complex,
    w: &mut Wp<'_>,
) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    if [&s.0, &s.1, &z.0, &z.1]
        .iter()
        .any(|x| x.is_nan() || x.is_inf())
    {
        return Err(super::unevaluable("polylog of special float value"));
    }
    if z.0.is_zero() && z.1.is_zero() {
        return Ok(c_zero(prec));
    }
    let sigma = to_f64(&s.0).ok_or_else(|| super::unevaluable("polylog order beyond f64"))?;
    let tau = to_f64(&s.1).ok_or_else(|| super::unevaluable("polylog order beyond f64"))?;
    if z.1.is_zero() && z.0 == BigFloat::from_i32(1, 64) {
        if super::bf_gt(&s.0, &BigFloat::from_i32(1, 64)) {
            return super::zeta::zeta_complex(s, w);
        }
        // Infinite for `Re s < 1` (`Li_s(z) ~ Γ(1 − s)(−ln z)^(s−1)`); at `Re s
        // = 1` the term `(−ln z)^(iτ)` oscillates: no limit.
        let reason = "polylog(s, 1) diverges for Re s <= 1";
        return Err(if super::bf_gt(&BigFloat::from_i32(1, 64), &s.0) {
            super::infinite(reason)
        } else {
            super::unevaluable(reason)
        });
    }
    if sigma.abs().max(tau.abs()) > 1.0e4 {
        return Err(super::unevaluable(
            "polylog of a complex order beyond 10^4 not supported in evalf",
        ));
    }
    let lr = accuracy::lg_abs(z);
    let r = lr.exp2();
    if r.is_nan() {
        return Err(super::unevaluable(
            "polylog argument beyond the exponent range",
        ));
    }
    // The series while it takes up to 200,000 terms (a power at the primes
    // only, a product elsewhere: cheaper than the ζ values of the expansion).
    let cheap = |r: f64| series_terms(r, sigma, prec + 64).is_some_and(|k| k <= 200_000);
    let regime = if r <= 0.75 || (r < 1.0 && cheap(r)) {
        Regime::Series
    } else if r < 4.0 / 3.0 {
        Regime::Mu
    } else {
        Regime::Inversion
    };
    let cap = super::cancellation_cap(prec);
    let mut extra = 32 + accuracy::ceil_log2(prec) as usize;
    loop {
        let wp = prec + extra;
        let sum = match regime {
            Regime::Series => complex_series(s, sigma, z, &mut w.at(wp)),
            Regime::Mu => complex_mu_expansion(s, sigma, tau, z, &mut w.at(wp)),
            Regime::Inversion => complex_inversion(s, sigma, z, &mut w.at(wp)),
        }
        .map_err(super::requested_at(prec))?;
        let lv = accuracy::lg_abs(&sum.value);
        let lost = if lv.is_finite() && sum.largest.is_finite() {
            ((sum.largest - lv).max(0.0).ceil() as usize).min(cap + 1)
        } else {
            wp
        };

        if lost + 24 <= extra {
            return Ok((
                super::round_to(sum.value.0, prec + 16),
                super::round_to(sum.value.1, prec + 16),
            ));
        }
        if extra >= cap || lost + 24 > cap {
            return Err(super::special_exhausted(prec));
        }
        extra = (lost + 40).max(2 * extra).min(cap);
    }
}

/// The factors of `ζ(s − k)` by the functional equation, carried from one `k`
/// to the next in [`complex_mu_expansion`]: `A_k = 2^(s−k) π^(s−k−1)`, `Γ(1 − s +
/// k)`, and `sin(πs/2)`, `cos(πs/2)`.
struct LeftZeta {
    a: Complex,
    g: Complex,
    sn: Complex,
    cs: Complex,
}

/// The expansion [`polylog_complex_order`] takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Regime {
    Series,
    Mu,
    Inversion,
}

/// The expansion in `μ = ln z` for a non-real order (see
/// [`polylog_complex_order`]) at `wp` bits.  `ζ(s − k)` by `zeta.rs` while `σ − k
/// ≥ 0`, beyond by the functional equation as a recurrence in `k`:
/// `ζ(s − k) = A_k·sin(π(s − k)/2)·Γ(1 − s + k)·ζ(1 − s + k)` with `A_k = 2^(s−k)
/// π^(s−k−1)` (`A_{k+1} = A_k/2π`), the sine one of `±sin(πs/2)`, `±cos(πs/2)`
/// and `Γ(2 − s + k) = (1 − s + k)·Γ(1 − s + k)`.  The tail from `j` with `σ − j ≤ −1`
/// is that of [`mu_expansion`] times `cosh(πτ/2)`: `abs(ζ(s − j)) ≤ 2(2π)^(σ−j−1)·
/// cosh(πτ/2)·Γ(1 − σ + j)·ζ(1 − σ + j)` (`abs(sin(x + iy)) ≤ cosh y`, `abs(Γ(x + iy)) ≤
/// Γ(x)`).
#[allow(clippy::too_many_arguments)]
fn complex_mu_expansion(
    s: &Complex,
    sigma: f64,
    tau: f64,
    z: &Complex,
    w: &mut Wp<'_>,
) -> Result<Sum, SymplexError> {
    let wp = w.bits();
    use super::zeta;
    let mu = ln_near_one(z, w);
    let lmu = accuracy::lg_abs(&mu);
    let rho = (lmu - LOG2_TWO_PI).exp2();
    if rho.is_nan() || rho >= 0.9 {
        return Err(super::unevaluable(
            "polylog: ln z outside the expansion's disc",
        ));
    }
    let mut sum = Sum::new(wp);
    let one_minus =
        |k: i64, wp: usize| -> Complex { (zeta::plus_int(&s.0.neg(), 1 + k, wp), s.1.neg()) };
    if !(mu.0.is_zero() && mu.1.is_zero()) {
        // Γ(1 − s)·(−μ)^(s−1), principal branch.
        let t = one_minus(0, wp);
        let g = zeta::gamma_complex(&t, w)?;
        let pw = zeta::pow_neg(&c_neg(&mu), &t, w);
        sum.add(&c_mul(&g, &pw, wp, RM), wp);
    }
    let lcosh = std::f64::consts::FRAC_PI_2 * tau.abs() * LOG2E;
    let max_terms = 8 * wp + 256;
    let mut mk = c_one(wp); // μ^k/k!
    let mut left: Option<LeftZeta> = None;
    // m^(−(1−s+k)), m = 1..=64, once ζ(1 − s + k) is that direct sum
    // (`zeta::direct_terms_upto`): each step divides by m.
    let mut direct: Option<Vec<Complex>> = None;
    let dp = wp + 16;
    let pi = w.cc.pi(wp + 16, RM).clone();
    let two_pi = pi.mul(&BigFloat::from_i32(2, 64), wp + 16, RM);
    for k in 0..max_terms {
        if k > 0 {
            mk = div_int(&c_mul(&mk, &mu, wp, RM), k as u64, wp);
        }
        let sk = (zeta::plus_int(&s.0, -(k as i64), wp), s.1.clone());
        let (zv, zlargest) = if sigma - (k as f64) >= 0.0 {
            let v = zeta::zeta_complex(&sk, w)?;
            let l = accuracy::lg_abs(&v);
            (v, l)
        } else {
            let ki = k as i64;
            let LeftZeta { a, g, sn, cs } = match left.take() {
                Some(LeftZeta { a, g, sn, cs }) => {
                    // A_k = A_{k−1}/2π, Γ(1 − s + k) = (k − s)·Γ(k − s).
                    let a = (a.0.div(&two_pi, wp, RM), a.1.div(&two_pi, wp, RM));
                    let g = c_mul(&g, &one_minus(ki - 1, wp), wp, RM);
                    LeftZeta { a, g, sn, cs }
                }
                None => {
                    let p = wp
                        + super::magnitude_bits((sigma.abs() + tau.abs() + k as f64 + 2.0) * 3.0)
                        + 16;
                    let ln2 = BigFloat::from_i32(2, 64).ln(p, RM, w.cc);
                    let lnpi = pi.clone().ln(p, RM, w.cc);
                    let sk1 = (zeta::plus_int(&sk.0, -1, p), sk.1.clone());
                    let e = c_add(
                        &(sk.0.mul(&ln2, p, RM), sk.1.mul(&ln2, p, RM)),
                        &(sk1.0.mul(&lnpi, p, RM), sk1.1.mul(&lnpi, p, RM)),
                        p,
                        RM,
                    );
                    let a = super::c_exp(&e, &mut w.at(p));
                    let a = (super::round_to(a.0, wp), super::round_to(a.1, wp));
                    let g = zeta::gamma_complex(&one_minus(ki, wp), w)?;
                    let sn = zeta::sin_half_pi(s, w);
                    let s1 = (zeta::plus_int(&s.0, 1, wp), s.1.clone());
                    let cs = zeta::sin_half_pi(&s1, w);
                    LeftZeta { a, g, sn, cs }
                }
            };
            // sin(πs/2 − πk/2): S, −C, −S, C.
            let trig = match k % 4 {
                0 => sn.clone(),
                1 => c_neg(&cs),
                2 => c_neg(&sn),
                _ => cs.clone(),
            };
            let one = (BigFloat::from_i32(1, 64), BigFloat::new(64));
            let n_direct = zeta::direct_terms_upto(1.0 - sigma + k as f64, 1.0, 0.0, wp, 64);
            let zr = match n_direct {
                Some(n) => {
                    let powers = match direct.take() {
                        Some(mut p) => {
                            for (m, x) in p.iter_mut().enumerate().skip(1) {
                                let d = BigFloat::from_u64(m as u64 + 1, 64);
                                *x = (x.0.div(&d, dp, RM), x.1.div(&d, dp, RM));
                            }
                            p
                        }
                        None => {
                            let t = one_minus(ki, dp);
                            (1..=64u64)
                                .map(|m| {
                                    let b = (BigFloat::from_u64(m, 64), BigFloat::new(64));
                                    zeta::pow_neg(&b, &t, &mut w.at(dp))
                                })
                                .collect()
                        }
                    };
                    let mut acc = c_zero(dp);
                    for x in powers.iter().take(n) {
                        acc = c_add(&acc, x, dp, RM);
                    }
                    direct = Some(powers);
                    zeta::Sum {
                        value: acc,
                        largest: 0.0,
                    }
                }
                None => zeta::hurwitz(&one_minus(ki, wp), &one, w)?,
            };
            let v = c_mul(
                &c_mul(&c_mul(&a, &trig, wp, RM), &g, wp, RM),
                &zr.value,
                wp,
                RM,
            );
            let l =
                accuracy::lg_abs(&v).max(zr.largest + accuracy::lg_abs(&a) + accuracy::lg_abs(&g));
            left = Some(LeftZeta { a, g, sn, cs });
            (v, l)
        };
        let term = c_mul(&mk, &zv, wp, RM);
        sum.add(&term, wp);
        sum.largest = sum.largest.max(zlargest + accuracy::lg_abs(&mk));
        // The tail from j = k + 1 once σ − j ≤ −1 (see `mu_expansion`).
        let j = k as f64 + 1.0;
        if j >= sigma + 1.0 && k >= 1 {
            let lb = 1.73
                + lcosh
                + (sigma - j - 1.0) * LOG2_TWO_PI
                + ln_gamma(j + 1.0 - sigma) * LOG2E
                + j * lmu
                - ln_gamma(j + 1.0) * LOG2E;
            let q = rho * ((j + 1.0 - sigma) / (j + 1.0)).max(1.0);
            if q < 1.0 && lb - (1.0 - q).log2() < sum.largest - wp as f64 - 4.0 {
                return Ok(sum);
            }
        }
    }
    Err(super::series_did_not_converge("polylog", max_terms))
}

/// The inversion formula of the module documentation for a non-real order
/// and `abs(z) > 1`: `Li_s(z) = −e^(iπs)·Li_s(1/z) + (2π)^s/Γ(s)·e^(iπs/2)·ζ(1 − s,
/// a)`, `e^(iπs) = e^(−πτ)·e^(iπσ)`, `Γ(s)` from `zeta.rs`, the Hurwitz function
/// of the complex order `1 − s` by `zeta::hurwitz` (`0 < Re a ≤ 1`).
fn complex_inversion(
    s: &Complex,
    sigma: f64,
    z: &Complex,
    w: &mut Wp<'_>,
) -> Result<Sum, SymplexError> {
    let wp = w.bits();
    use super::zeta;
    let pi = w.cc.pi(wp, RM).clone();
    let two_pi = pi.mul(&BigFloat::from_i32(2, 64), wp, RM);
    let arg = super::atan2_bf(&z.1, &z.0, w);
    let upper = is_strictly_positive(&z.1) || (z.1.is_zero() && z.0.is_negative());
    let turn = arg.div(&two_pi, wp, RM);
    let re_a = if upper {
        turn
    } else {
        BigFloat::from_i32(1, 64).add(&turn, wp, RM)
    };
    let ln_abs = super::c_abs(z, wp + 8, RM).ln(wp, RM, w.cc);
    let a = (re_a, ln_abs.div(&two_pi, wp, RM).neg());
    let inv_z = c_div(&c_one(wp), z, wp, RM);
    let li = complex_series(s, sigma, &inv_z, w)?;
    let t = (zeta::plus_int(&s.0.neg(), 1, wp), s.1.neg());
    let hz = zeta::hurwitz(&t, &a, w)?;
    let gamma = zeta::gamma_complex(s, w)?;
    // (2π)^s/Γ(s) = exp(s·ln 2π)/Γ(s).
    let ls = accuracy::lg_abs(s).exp2();
    let p = wp + super::magnitude_bits(ls * 2.0) + 8;
    let ln2pi =
        w.cc.pi(p, RM)
            .clone()
            .mul(&BigFloat::from_i32(2, 64), p, RM)
            .ln(p, RM, w.cc);
    let ex = super::c_exp(
        &(s.0.mul(&ln2pi, p, RM), s.1.mul(&ln2pi, p, RM)),
        &mut w.at(p),
    );
    let c = c_div(&ex, &gamma, wp, RM);
    // e^(iπs), e^(iπs/2): `e^(−πτ)` with the bits of `πτ` as guard.
    let tp = wp + super::magnitude_bits(4.0 * ls) + 8;
    let pt = w.cc.pi(tp, RM).clone().mul(&s.1, tp, RM);
    let e1 = scale(&exp_i_pi(&s.0, w), &pt.neg().exp(wp, RM, w.cc), wp);
    let half_s =
        s.0.div(&BigFloat::from_i32(2, 64), super::exact_bits(&s.0, wp), RM);
    let half_pt = pt.div(&BigFloat::from_i32(2, 64), tp, RM);
    let e2 = scale(&exp_i_pi(&half_s, w), &half_pt.neg().exp(wp, RM, w.cc), wp);
    let t1 = c_neg(&c_mul(&e1, &li.value, wp, RM));
    let ce2 = c_mul(&c, &e2, wp, RM);
    let t2 = c_mul(&ce2, &hz.value, wp, RM);
    let mut sum = Sum::new(wp);
    sum.add(&t1, wp);
    sum.add(&t2, wp);
    sum.largest = sum
        .largest
        .max(li.largest + accuracy::lg_abs(&e1))
        .max(hz.largest + accuracy::lg_abs(&ce2));
    Ok(sum)
}

/// The series of [`polylog_complex_order`] at `wp` bits.
fn complex_series(
    s: &Complex,
    sigma: f64,
    z: &Complex,
    w: &mut Wp<'_>,
) -> Result<Sum, SymplexError> {
    let wp = w.bits();
    let lr = accuracy::lg_abs(z);
    let r = lr.exp2().min(1.0);
    let max_terms = series_terms(r, sigma, wp)
        .map(|k| (2 * k + 64).min(1 << 21))
        .ok_or_else(|| super::series_did_not_converge("polylog", 0))?;
    // The tail is measured against `2^−target`; the powers `z^k` and the
    // partial sums carry `log₂` of the number of terms more, for their
    // roundings.
    let target = wp;
    let wp = wp + accuracy::ceil_log2(max_terms) as usize + 4;
    let spf = smallest_prime_factors(max_terms);
    let neg_s = c_neg(s);
    let mut powers: Vec<Complex> = Vec::with_capacity(max_terms.min(4096));
    let mut zk = z.clone();
    let mut sum = Sum::new(wp);
    for k in 1..=max_terms {
        if k > 1 {
            zk = c_mul(&zk, z, wp, RM);
        }
        let p = if k == 1 {
            c_one(wp)
        } else {
            let q = spf[k] as usize;
            if q == k {
                pow_complex(k as u64, &neg_s, &mut w.at(wp))
            } else {
                c_mul(&powers[q - 1], &powers[k / q - 1], wp, RM)
            }
        };
        sum.add(&c_mul(&zk, &p, wp, RM), wp);
        powers.push(p);
        // The tail Σ_{j>k} |z|^j j^{−σ}, as in `series`.
        let kf = k as f64;
        let next = (kf + 1.0) * lr.min(0.0) - sigma * (kf + 1.0).log2();
        let q = r * ((kf + 2.0) / (kf + 1.0)).powf(-sigma).max(1.0);
        if q < 1.0 && next - (1.0 - q).log2() < sum.largest - target as f64 - 4.0 {
            return Ok(sum);
        }
    }
    Err(super::series_did_not_converge("polylog", max_terms))
}

/// `k^w = e^(w·ln k)` for a positive integer `k` and a complex `w`, to `wp` bits
/// relative: the exponent with the bits of its magnitude as guard.
fn pow_complex(k: u64, w: &Complex, wpc: &mut Wp<'_>) -> Complex {
    let wp = wpc.bits();
    let size = (to_f64(&w.0).unwrap_or(f64::INFINITY).abs()
        + to_f64(&w.1).unwrap_or(f64::INFINITY).abs())
        * ((k as f64).ln() + 1.0);
    let p = wp + super::magnitude_bits(size) + 8;
    let lk = BigFloat::from_u64(k, 64).ln(p, RM, wpc.cc);
    let v = super::c_exp(&scale(w, &lk, p), &mut wpc.at(p));
    (super::round_to(v.0, wp), super::round_to(v.1, wp))
}

/// The error bound of `Li_s(z)` (value `v`) for a complex order (see
/// [`polylog_complex_order`]): its own error, and twice the first-order
/// changes over the balls of `z` and `s`, `abs(∂Li/∂z) ≤ Σ k^(1−σ′) r′^(k−1)` and
/// `abs(∂Li/∂s) ≤ Σ ln k·r′^k k^(−σ′)` with `r′ = abs(z) + 2^rz < 3/4`, `σ′ = σ −
/// 2^rs` (summed in `f64` with a geometric bound on their tails, times 2).
/// Beyond, as for a real order ([`error_bound`]): `abs(∂Li_s/∂z) = abs(Li_(s−1)(z)/z)`
/// at 128 bits grown by `(1 − r/d)^(−max(1, abs(s − 2)))` (`d = abs(z − 1)`, no bound
/// for `r > d/8` or a ball meeting the cut `[1, ∞)` with an undecided side),
/// and `abs(∂Li_s/∂s)` from a forward difference at 128 bits.
#[allow(clippy::too_many_arguments)]
pub(super) fn complex_order_bound(
    s: &Complex,
    bs: Bound,
    z: &Complex,
    bz: Bound,
    v: &Complex,
    w: &mut Wp<'_>,
) -> Bound {
    let prec = w.bits();
    if bs.is_unknown() || bz.is_unknown() {
        return Bound::UNKNOWN;
    }
    let lv = accuracy::lg_abs(v);
    if !lv.is_finite() {
        return Bound::UNKNOWN;
    }
    let mut total = lv - prec as f64 - 12.0;
    if !bz.is_exact() || !bs.is_exact() {
        let (Some(sigma), Some(r)) = (to_f64(&s.0), Some(accuracy::lg_abs(z).exp2())) else {
            return Bound::UNKNOWN;
        };
        let rz = if bz.is_exact() {
            0.0
        } else {
            bz.joint().exp2()
        };
        let rs = if bs.is_exact() {
            0.0
        } else {
            bs.joint().exp2()
        };
        let (rp, sp) = (r + rz, sigma - rs);
        if rp.is_nan() || !sp.is_finite() {
            return Bound::UNKNOWN;
        }
        if rp >= 0.75 {
            if !bz.is_exact() {
                match complex_z_change(s, z, bz, w.cc) {
                    Some(c) => total = accuracy::lsum(total, c),
                    None => return Bound::UNKNOWN,
                }
            }
            if !bs.is_exact() {
                match complex_s_change(s, bs.joint(), z, v, w.cc) {
                    Some(c) => total = accuracy::lsum(total, c),
                    None => return Bound::UNKNOWN,
                }
            }
            return Bound::both(total);
        }
        let (Some(dz), Some(ds)) = (
            magnitude_sum(rp, 1.0 - sp, true),
            magnitude_sum(rp, -sp, false),
        ) else {
            return Bound::UNKNOWN;
        };
        // In log₂: a radius below the `f64` range underflows to 0, and before
        // 0.45 the argument's error was then dropped past 1,074 bits.
        if !bz.is_exact() {
            total = accuracy::lsum(total, dz.log2() + bz.joint() + 1.0);
        }
        if !bs.is_exact() {
            total = accuracy::lsum(total, ds.log2() + bs.joint() + 1.0);
        }
    }

    Bound::both(total)
}

/// Twice the change of `Li_s` (a non-real order) over the ball of `z` (see
/// [`complex_order_bound`]).
fn complex_z_change(s: &Complex, z: &Complex, bz: Bound, cc: &mut Consts) -> Option<f64> {
    let r = bz.joint();
    let undecided_side =
        !accuracy::exact_zero(&z.1, bz.im) && accuracy::part_contains_zero(&z.1, bz.im);
    if undecided_side {
        let hi = z.0.add(&bigfloat::from_f64(r.exp2(), 64), LP, RM);
        if !super::bf_lt(&hi, &BigFloat::from_i32(1, 64)) {
            return None;
        }
    }
    let one = BigFloat::from_i32(1, 64);
    let d = (z.0.sub(&one, super::exact_bits(&z.0, LP), RM), z.1.clone());
    let ld = accuracy::lg_abs_low(&d);
    if ld.is_nan() || ld <= r + 3.0 {
        return None;
    }
    let s1 = (super::zeta::plus_int(&s.0, -1, LP), s.1.clone());
    let li = polylog_complex_order(&s1, z, &mut Wp::new(LP, cc)).ok()?;
    let dl = accuracy::lg_abs(&li) - accuracy::lg_abs_low(z);
    if !dl.is_finite() {
        return None;
    }
    let s2 = (super::zeta::plus_int(&s.0, -2, LP), s.1.clone());
    let p = accuracy::lg_abs(&s2).exp2().max(1.0);
    Some(dl + p * shrink(r - ld) + r + 1.0)
}

/// Twice the change of `Li_s(z)` over the ball `s ± 2^es` of a non-real
/// order, from a forward difference along the real axis (`Li_s` is analytic
/// in `s`; see [`complex_order_bound`]).
fn complex_s_change(
    s: &Complex,
    es: f64,
    z: &Complex,
    v: &Complex,
    cc: &mut Consts,
) -> Option<f64> {
    let ls = accuracy::lg_abs(s);
    let lh = (ls.max(0.0) - 40.0).floor();
    if !lh.is_finite() || es > lh - 8.0 {
        return None;
    }
    let mut h = BigFloat::from_i32(1, 64);
    h.set_exponent(lh as i32 + 1);
    let moved = (s.0.add(&h, super::exact_bits(&s.0, LP), RM), s.1.clone());
    let w = polylog_complex_order(&moved, z, &mut Wp::new(LP, cc)).ok()?;
    let diff = c_sub(&w, v, LP + 64, RM);
    let noise = accuracy::lg_abs(v) - (LP as f64 - 4.0) - lh;
    let deriv = accuracy::lsum(accuracy::lg_abs(&diff) - lh, noise);
    Some(deriv + es + 2.0)
}

/// `Σ_{k≥1} k^e·r^(k−1)` (`derivative`) or `Σ_{k≥2} ln k·k^e·r^k` in `f64`,
/// rounded up generously: the terms until the rest, bounded by a geometric
/// series, is below `10⁻¹⁸` of the sum, then that bound.
fn magnitude_sum(r: f64, e: f64, derivative: bool) -> Option<f64> {
    let mut sum = 0.0f64;
    for k in 1..2_000_000u64 {
        let kf = k as f64;
        let t = if derivative {
            kf.powf(e) * r.powf(kf - 1.0)
        } else {
            kf.ln() * kf.powf(e) * r.powf(kf)
        };
        sum += t;
        // The ratio of consecutive terms from here on is at most
        // r·(1 + 1/k)^max(e, 0) (and ln(k + 2)/ln(k + 1) for the logarithms),
        // decreasing in k.
        let lnq = if derivative {
            1.0
        } else {
            (kf + 2.0).ln() / (kf + 1.0).ln()
        };
        let q = r * (1.0 + 1.0 / kf).powf(e.max(0.0)) * lnq;
        if k > 2 && q < 1.0 {
            // (A geometric bound on the rest, returned with a margin.)
            let rest = t * q / (1.0 - q);
            if rest < sum * 1e-18 {
                return Some((sum + rest) * 1.001);
            }
        }
        if !sum.is_finite() {
            return None;
        }
    }
    None
}

/// `ln z` accurate next to `z = 1`: `ln|z| = ½ ln(1 + d)` with `d = |z|² −
/// 1 = (x − 1)(x + 1) + y²` formed with `x − 1` exact, and `ln(1 + d) =
/// 2 atanh(t)`, `t = d/(2 + d)`, summed for `|d| < 1/4`.
fn ln_near_one(z: &Complex, w: &mut Wp<'_>) -> Complex {
    let wp = w.bits();
    let p = wp + 16;
    let one = BigFloat::from_i32(1, 64);
    let xm1 = z.0.sub(&one, super::exact_bits(&z.0, p), RM);
    let xp1 = z.0.add(&one, super::exact_bits(&z.0, p), RM);
    let d = xm1.mul(&xp1, p, RM).add(&z.1.mul(&z.1, p, RM), p, RM);
    let arg = super::atan2_bf(&z.1, &z.0, w);
    let small = d.is_zero() || d.exponent().is_some_and(|e| e <= -2);
    let ln_abs = if !small {
        one.add(&d, p, RM)
            .ln(p, RM, w.cc)
            .div(&BigFloat::from_i32(2, 64), wp, RM)
    } else if d.is_zero() {
        BigFloat::new(wp)
    } else {
        // ½ ln(1 + d) = t + t³/3 + t⁵/5 + …, |t| < 1/7.
        let t = d.div(&BigFloat::from_i32(2, 64).add(&d, p, RM), p, RM);
        let t2 = t.mul(&t, p, RM);
        let mut pw = t.clone();
        let mut acc = t.clone();
        for k in 1..(p / 2 + 8) {
            pw = pw.mul(&t2, p, RM);
            let term = pw.div(&BigFloat::from_u64((2 * k + 1) as u64, 64), p, RM);
            acc = acc.add(&term, p, RM);
            if super::negligible(&term, &acc, p) {
                break;
            }
        }
        super::round_to(acc, wp)
    };
    (ln_abs, arg)
}

/// The expansion in `μ = ln z` for `|μ| < 2π` (see the module
/// documentation), stopped by the bound on its tail.
#[allow(clippy::too_many_arguments)]
fn mu_expansion(
    s: &BigFloat,
    s_int: Option<i64>,
    s_f: f64,
    z: &Complex,
    w: &mut Wp<'_>,
) -> Result<Sum, SymplexError> {
    let wp = w.bits();
    let mu = ln_near_one(z, w);
    let neg_mu = c_neg(&mu);
    let lmu = accuracy::lg_abs(&mu);
    // |μ|/2π: below 0.6 for 3/4 < |z| < 2, below 0.85 for a large order (`value_at`).
    let rho = (lmu - LOG2_TWO_PI).exp2();
    if rho.is_nan() || rho >= 0.9 {
        return Err(super::unevaluable(
            "polylog: ln z outside the expansion's disc",
        ));
    }
    let mut sum = Sum::new(wp);
    if s_int.is_none() {
        // Γ(1 − s)(−μ)^{s−1}, `1 − s` exact (see `polylog_near_one_at`).
        let one_minus_s = BigFloat::from_i32(1, 64).sub(s, super::exact_bits(s, wp), RM);
        let g = super::arb_gamma_real(&one_minus_s, w)?;
        let s_minus_1 = minus_int(s, 1, wp);
        let lead = scale(&pow_real(&neg_mu, &s_minus_1, w), &g, wp);
        sum.add(&lead, wp);
    }
    let max_terms = 8 * wp + 256 + s_f.max(0.0) as usize;
    let minus_one = BigFloat::from_i32(-1, 64);
    let mut left: Option<super::ZetaLeft> = None;
    let mut mk = c_one(wp); // μ^k/k!
    let mu_abs = lmu.exp2();
    for k in 0..max_terms {
        // Before the terms of order `s − k ≤ 2`: stop where the rest of the
        // sum, bounded term by term ([`mu_tail`]), is below the precision.
        // Before 0.37 every `ζ(s − k)`, `k < s`, was summed: `polylog(300,
        // 19/10)` took 0.7 s, `polylog(1000, 19/10)` was refused.
        if k >= 1
            && (k as f64) < s_f - 2.0
            && (k as f64) > 2.0 * mu_abs + 1.0
            && mu_tail(k, s_f, s_int, lmu, rho) < sum.largest - wp as f64 - 4.0
        {
            return Ok(sum);
        }
        if k > 0 {
            mk = div_int(&c_mul(&mk, &mu, wp, RM), k as u64, wp);
        }
        let term = match s_int {
            Some(n) if k as i64 + 1 == n => {
                // μ^{n−1}/(n−1)!·(H_{n−1} − ln(−μ))
                let one = BigFloat::from_i32(1, wp);
                let mut h = BigFloat::new(wp);
                for j in 1..n {
                    h = h.add(&one.div(&BigFloat::from_i64(j, 64), wp, RM), wp, RM);
                }
                let l = super::c_ln(&neg_mu, w);
                c_mul(&mk, &(h.sub(&l.0, wp, RM), l.1.neg()), wp, RM)
            }
            Some(n) => {
                let m = n - k as i64;
                let zeta = if m <= 0 {
                    super::zeta_nonpositive_int(m, wp)
                } else {
                    let mb = BigFloat::from_i64(m, 64);
                    match zeta_direct(&mb, m as f64, w) {
                        Some(z) => z,
                        None => super::arb_zeta(&mb, w)?,
                    }
                };
                scale(&mk, &zeta, wp)
            }
            None => {
                let arg = minus_int(s, k as i64, wp);
                let direct = if left.is_none() {
                    zeta_direct(&arg, s_f - k as f64, w)
                } else {
                    None
                };
                let zeta = if let Some(z) = direct {
                    z
                } else if left.is_some() || super::bf_lt(&arg, &minus_one) {
                    if left.is_none() {
                        left = Some(super::ZetaLeft::new(&arg, &mut w.at(wp))?);
                    }
                    match left.as_mut() {
                        Some(l) => l.next(),
                        None => super::arb_zeta(&arg, w)?,
                    }
                } else {
                    super::arb_zeta(&arg, w)?
                };
                scale(&mk, &zeta, wp)
            }
        };
        sum.add(&term, wp);
        // The tail from j = k + 1 once σ = s − j ≤ −1: the terms are below
        // b_j = 3.3·(2π)^{s−j−1}Γ(j+1−s)|μ|^j/j!, with b_{i+1}/b_i =
        // ρ(i+1−s)/(i+1) ≤ ρ·max(1, (j+1−s)/(j+1)) for i ≥ j.
        let j = k as f64 + 1.0;
        if j >= s_f + 1.0 && k >= 1 {
            let lb =
                1.73 + (s_f - j - 1.0) * LOG2_TWO_PI + ln_gamma(j + 1.0 - s_f) * LOG2E + j * lmu
                    - ln_gamma(j + 1.0) * LOG2E;
            let q = rho * ((j + 1.0 - s_f) / (j + 1.0)).max(1.0);
            if q < 1.0 && lb - (1.0 - q).log2() < sum.largest - wp as f64 - 4.0 {
                return Ok(sum);
            }
        }
    }
    Err(super::series_did_not_converge("polylog", max_terms))
}

/// `ζ(m) = Σ_{j≤J} j^{−m} + r`, `0 < r ≤ J^{1−m}/(m − 1) < 2^(−wp−8)`, for a
/// real `m ≥ max(8, wp/4)` (`mf` its value), where `J ≤ 64` terms do: the
/// terms of the expansion in `ln z` at a large order (Borwein's sum takes
/// `0.39·wp` powers whatever `m`).  `None` otherwise.
fn zeta_direct(m: &BigFloat, mf: f64, w: &mut Wp<'_>) -> Option<BigFloat> {
    let wp = w.bits();
    if mf.is_nan() || mf < 8.0 || mf < wp as f64 / 4.0 {
        return None;
    }
    let j_max = ((wp as f64 + 8.0) / (mf - 1.0)).exp2().ceil() + 1.0;
    if j_max.is_nan() || j_max > 64.0 {
        return None;
    }
    let neg = m.neg();
    let mut sum = BigFloat::from_i32(1, wp + 8);
    for j in 2..=(j_max as u64) {
        let t = super::bf_pow(&BigFloat::from_u64(j, 64), &neg, &mut w.at(wp + 8));
        sum = sum.add(&t, wp + 8, RM);
    }
    Some(super::round_to(sum, wp))
}

/// `log₂` of a bound on `Σ_{j≥k} abs(ζ(s − j)·μ^j/j!)` (the log term at `j = n − 1`
/// for an integer order `n`) for `2·abs(μ) + 1 < k < s − 2`, `log₂ abs(μ) = lmu`,
/// `abs(μ)/2π = rho < 1`: the terms with `σ = s − j ≥ 2` by `ζ(σ) ≤ ζ(2) < 1.65`
/// and a geometric series of ratio `abs(μ)/(k + 1) < ½`; the at most three with
/// `−1 < σ < 2` by `abs(ζ(σ)) ≤ 1 + 1/abs(σ − 1)` (`ζ(σ) − 1/(σ − 1)` lies in
/// `[0.42, 0.65]` there), the log term by `H_{n−1} + abs(ln(−μ)) ≤ ln n + 1 +
/// abs(log abs(μ)) + π`; and those with `σ ≤ −1` as in [`mu_expansion`].
fn mu_tail(k: usize, s_f: f64, s_int: Option<i64>, lmu: f64, rho: f64) -> f64 {
    let mu_abs = lmu.exp2();
    let kf = k as f64;
    let term = |j: f64| j * lmu - ln_gamma(j + 1.0) * LOG2E;
    let mut total = 0.73 + term(kf) - (1.0 - mu_abs / (kf + 1.0)).log2();
    // The integers j with −1 < s − j < 2.
    let first = (s_f - 2.0).floor() + 1.0;
    let mut j = first.max(kf);
    while s_f - j > -1.0 {
        let sigma = s_f - j;
        let z = match s_int {
            Some(n) if j as i64 == n - 1 => {
                (n as f64).ln() + 1.0 + (lmu * LN2).abs() + std::f64::consts::PI
            }
            _ if sigma == 1.0 => f64::INFINITY,
            _ => 1.0 + 1.0 / (sigma - 1.0).abs(),
        };
        total = lsum2(total, term(j) + z.log2());
        j += 1.0;
    }
    // σ ≤ −1: b_j = 3.3·(2π)^{s−j−1}Γ(j+1−s)|μ|^j/j! and a geometric tail.
    let jc = j;
    let q = rho * ((jc + 1.0 - s_f) / (jc + 1.0)).max(1.0);
    if q >= 1.0 {
        return f64::INFINITY;
    }
    let lb = 1.73 + (s_f - jc - 1.0) * LOG2_TWO_PI + ln_gamma(jc + 1.0 - s_f) * LOG2E + term(jc);
    lsum2(total, lb - (1.0 - q).log2())
}

/// `log₂` of a lower bound on `abs(Li_s(z) − z)` for a real order `s ≥ 4` (`s_f`;
/// `s_int`: an integer; `near_int`: a non-integer within `2⁻⁸` of one) and
/// `z = x + iy` (in `f64`): the distance of `Li_s` from its limit `z` as `s →
/// ∞`, which a cancellation against `z` leaves (`evalf::note_limit_tail`).
/// `−∞`-like (`−10³⁰⁰`, below the exponent range: such a cancellation is
/// refused) where no bound is proved.
///
/// * `abs(z) ≤ 1`: `abs(Σ_{k≥2} z^k/k^s) ≥ abs(z)²/2^s·(1 − Σ_{k≥3} (2/k)^s) ≥
///   abs(z)²·2^(−s−1)` (`Σ_{k≥3} (2/k)⁴ < 0.32`).
/// * `abs(z) > 1`, `m = abs(ln z) < 0.85·2π`, `s ≥ 8m + 40`, `0.585·s ≥ 7.22·m + 5`
///   and `s` an integer or at least `2⁻⁸` from one: in the expansion in `μ =
///   ln z`, `Li_s(z) − z = Σ_{k<s−1} (ζ(s − k) − 1)μ^k/k! + R` with `R` the terms
///   from `k = s − 1` on (with the `Γ(1 − s)` or logarithmic term, less the
///   tail of `e^μ`), and the sum is `Σ_{j≥2} j^(−s)·Σ_{k<s−1} (jμ)^k/k!`:
///   `2^(−s)·(z² − T)` (`T` the tail of `e^(2μ)` from `s − 1`, at most
///   `2·(2m)^(s−1)/(s − 1)!`) and the rest at most `4·3^(−s)·e^(3m)`.  When
///   `T`, the rest and `R` (bounded as in [`mu_tail`]) are each below
///   `2^(−s)·abs(z)²/8` (the conditions make the first two so), `abs(Li_s(z) − z) ≥
///   abs(z)²·2^(−s−1)`.
pub(super) fn tail_lower_bound(s_f: f64, s_int: bool, near_int: bool, x: f64, y: f64) -> f64 {
    const UNKNOWN_TAIL: f64 = -1e300;
    let r = x.hypot(y);
    // (No bound claimed.)
    if s_f.is_nan() || s_f < 4.0 || !r.is_finite() || r == 0.0 {
        return UNKNOWN_TAIL;
    }
    let lz = r.log2();
    let main = 2.0 * lz - s_f - 1.0;
    if r <= 1.0 {
        return main;
    }
    let m = r.ln().hypot(y.atan2(x));
    if m >= 0.85 * 2.0 * std::f64::consts::PI
        || s_f < 8.0 * m + 40.0
        || 0.585 * s_f < 7.22 * m + 5.0
        || (!s_int && near_int)
    {
        return UNKNOWN_TAIL;
    }
    // R: the terms from k0 = ⌊s⌋ − 3 on (an over-estimate), the tail of e^μ
    // from there, and |Γ(1 − s)|·m^(s−1) ≤ π·2⁷/Γ(s)·m^(s−1) off the integers.
    let k0 = s_f.floor() - 3.0;
    let lm = m.log2();
    let n = s_int.then_some(s_f as i64);
    let mut rb = mu_tail(k0 as usize, s_f, n, lm, m / (2.0 * std::f64::consts::PI));
    rb = lsum2(rb, 1.0 + k0 * lm - ln_gamma(k0 + 1.0) * LOG2E);
    // The last term of the main sum, `1 < s − k < 2`, has `Σ_{j≥3} j^(k−s) ≤
    // 769`, not 4.
    rb = lsum2(rb, 9.6 + (s_f - 2.0) * lm - ln_gamma(s_f - 1.0) * LOG2E);
    if !s_int {
        rb = lsum2(rb, 8.66 + (s_f - 1.0) * lm - ln_gamma(s_f) * LOG2E);
    }
    if rb.is_finite() && rb <= main - 3.0 {
        main
    } else {
        UNKNOWN_TAIL
    }
}

/// `log₂(2^a + 2^b)`, rounded up.
fn lsum2(a: f64, b: f64) -> f64 {
    accuracy::lsum(a, b)
}

/// The inversion formula for `|z| > 1` (see the module documentation).
#[allow(clippy::too_many_arguments)]
fn inversion(
    s: &BigFloat,
    s_int: Option<i64>,
    s_f: f64,
    z: &Complex,
    w: &mut Wp<'_>,
) -> Result<Sum, SymplexError> {
    let wp = w.bits();
    // a = ½ + ln(−z)/(2πi) = (½ + arg(−z)/2π) − i·ln|z|/2π with the
    // principal arg(−z) = arg z ∓ π: Re a = arg z/2π above the real axis
    // (and for z < 0), 1 + arg z/2π below it and on the cut z > 1 (arg(−z)
    // = π there).  From arg z itself, not ½ + arg(−z)/2π: just above the
    // cut the latter cancels to 0 (`z = 73/8 + 10⁻⁵⁸·i`).
    let pi = w.cc.pi(wp, RM).clone();
    let two_pi = pi.mul(&BigFloat::from_i32(2, 64), wp, RM);
    let arg = super::atan2_bf(&z.1, &z.0, w);
    let upper = is_strictly_positive(&z.1) || (z.1.is_zero() && z.0.is_negative());
    let turn = arg.div(&two_pi, wp, RM);
    let re_a = if upper {
        turn
    } else {
        BigFloat::from_i32(1, 64).add(&turn, wp, RM)
    };
    let ln_abs = super::c_abs(z, wp + 8, RM).ln(wp, RM, w.cc);
    let a = (re_a, ln_abs.div(&two_pi, wp, RM).neg());
    // For a large order the Hurwitz sum cancels about `1.77·s` bits (terms
    // of `|w|^(s−1)`, `|w| ≈ s/5`, against `|ζ(1 − s, a)| ≈ 2Γ(s)/(2π)^s`):
    // beyond the cap, refused before the sum (`polylog(1000.5, 3)` spent a
    // second on it).
    if 1.5 * s_f > super::cancellation_cap(wp) as f64 {
        return Err(super::special_exhausted(wp));
    }
    let inv_z = c_div(&c_one(wp), z, wp, RM);
    let li = series(s, s_int, s_f, &inv_z, w)?;
    let sigma = BigFloat::from_i32(1, 64).sub(s, super::exact_bits(s, wp), RM);
    let hz = hurwitz(&sigma, &a, w)?;
    let gamma = super::arb_gamma_real(s, w)?;
    let c = super::bf_pow(&two_pi, s, w).div(&gamma, wp, RM);
    let e1 = exp_i_pi(s, w);
    let half_s = s.div(&BigFloat::from_i32(2, 64), super::exact_bits(s, wp), RM);
    let e2 = exp_i_pi(&half_s, w);
    let t1 = c_neg(&c_mul(&e1, &li.value, wp, RM));
    let t2 = scale(&c_mul(&e2, &hz.value, wp, RM), &c, wp);
    let lc = lg2(&c);
    let mut sum = Sum::new(wp);
    sum.add(&t1, wp);
    sum.add(&t2, wp);
    sum.largest = sum.largest.max(li.largest).max(hz.largest + lc);
    Ok(sum)
}

/// `ζ(σ, a)` for a real `σ ≠ 1` and a real `0 < a ≤ 1` at working precision
/// `wp` (see [`hurwitz`]): the value and `log₂` of the largest term summed,
/// which bounds its absolute error by a few units of `2^(largest − wp)`.  The
/// caller measures the cancellation (`largest` against the value) and
/// repeats with more bits.
pub(super) fn hurwitz_real(
    sigma: &BigFloat,
    a: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<(BigFloat, f64), SymplexError> {
    let wp = w.bits();
    if *sigma == BigFloat::from_i32(1, 64) {
        return Err(super::unevaluable("Hurwitz zeta: pole at order 1"));
    }
    let sum = hurwitz(sigma, &(a.clone(), BigFloat::new(wp)), w)?;
    Ok((sum.value.0, sum.largest))
}

/// The Hurwitz zeta function `ζ(σ, a)` for a real `σ ≠ 1` and `0 < Re a ≤
/// 1` by Euler–Maclaurin (see the module documentation).
fn hurwitz(sigma: &BigFloat, a: &Complex, wpc: &mut Wp<'_>) -> Result<Sum, SymplexError> {
    let wp = wpc.bits();
    let sf = to_f64(sigma).ok_or_else(|| super::unevaluable("Hurwitz zeta order"))?;
    let (Some(ar), Some(ai)) = (to_f64(&a.0), to_f64(&a.1)) else {
        return Err(super::unevaluable("Hurwitz zeta argument beyond f64"));
    };
    if (a.0.is_negative() && !a.0.is_zero()) || (a.0.is_zero() && a.1.is_zero()) {
        return Err(super::unevaluable("Hurwitz zeta: Re a < 0"));
    }
    let target = 0.15 * wp as f64 + (-sf).max(0.0) / 5.0 + 2.0;
    let n: usize = if ar.hypot(ai) >= target {
        0
    } else {
        ((target * target - ai * ai).max(0.0).sqrt() - ar)
            .ceil()
            .max(0.0) as usize
    };
    let exact = super::exact_bits(&a.0, wp);
    let neg_sigma = sigma.neg();
    let mut sum = Sum::new(wp);
    for k in 0..n {
        let ak = (
            a.0.add(&BigFloat::from_u64(k as u64, 64), exact, RM),
            a.1.clone(),
        );
        sum.add(&pow_real(&ak, &neg_sigma, wpc), wp);
    }
    let w = (
        a.0.add(&BigFloat::from_u64(n as u64, 64), exact, RM),
        a.1.clone(),
    );
    let base = pow_real(&w, &neg_sigma, wpc); // w^{−σ}
    let sigma_minus_1 = minus_int(sigma, 1, wp);
    let integral = c_div(
        &c_mul(&base, &w, wp, RM),
        &(sigma_minus_1, BigFloat::new(wp)),
        wp,
        RM,
    );
    sum.add(&integral, wp);
    sum.add(&div_int(&base, 2, wp), wp);
    let inv_w = c_div(&c_one(wp), &w, wp, RM);
    let inv_w2 = c_mul(&inv_w, &inv_w, wp, RM);
    let lw = accuracy::lg_abs_low(&w);
    let lre = lg2(&w.0) - 1e-9;
    let mut pw = inv_w; // w^{−(2j−1)}
    // c_j = (σ)_{2j−1}/(2j)!
    let mut c = sigma.div(&BigFloat::from_i32(2, 64), wp, RM);
    let mut lpoch = 0.0f64; // log₂|(σ)_{2j}|
    let max_m = 4 * wp + 2 * sf.abs() as usize + 64;
    for j in 1..=max_m {
        if j > 1 {
            let f1 = minus_int(sigma, -(2 * j as i64 - 3), wp);
            let f2 = minus_int(sigma, -(2 * j as i64 - 2), wp);
            c = c.mul(&f1, wp, RM).mul(&f2, wp, RM).div(
                &BigFloat::from_u64(((2 * j - 1) * (2 * j)) as u64, 64),
                wp,
                RM,
            );
            pw = c_mul(&pw, &inv_w2, wp, RM);
        }
        let b = super::ratio_to_bigfloat(&super::bernoulli::even(j), wp);
        let t = scale(&c_mul(&base, &pw, wp, RM), &b.mul(&c, wp, RM), wp);
        sum.add(&t, wp);
        let jf = j as f64;
        // The factors σ + 2j − 2 and σ + 2j − 1 exactly: next to an integer
        // order one of them is tiny (`σ = −1 − 10⁻²⁰`), and in `f64` it was 0
        // — an "exact" sum after one term, wrong from the 21st digit.
        let g1 = minus_int(sigma, -(2 * j as i64 - 2), wp);
        let g2 = minus_int(sigma, -(2 * j as i64 - 1), wp);
        lpoch += lg2(&g1) + lg2(&g2);
        if lpoch == f64::NEG_INFINITY {
            // (σ)_{2j} = 0: the sum is exact (an integer σ ≤ 0).
            return Ok(sum);
        }
        let p = sf + 2.0 * jf;
        if p >= 2.5 {
            let int = ((1.0 - p) * lre - (p - 1.0).log2()).min(LOG2_HALF_PI + (1.0 - p) * lw);
            let rem = 2.0 + lpoch - 2.0 * jf * LOG2_TWO_PI + int;
            // (A bound on the remainder: the stop is certified.)
            if rem < sum.largest - wp as f64 - 4.0 {
                return Ok(sum);
            }
        }
    }
    Err(super::series_did_not_converge("Hurwitz zeta", max_m))
}

// ── Error bound ─────────────────────────────────────────────────────────────

/// `−log₂(1 − 2^x)`: how much `1/d` grows over a ball of radius `2^x·d`.
fn shrink(x: f64) -> f64 {
    if x == f64::NEG_INFINITY {
        0.0
    } else {
        -(-x.exp2()).ln_1p() * LOG2E
    }
}

/// The error bound of `Li_s(z)` (value `v`) off `[−1, 1]` for `s ± bs`,
/// `z ± bz`: the routine's own error (see [`polylog_general`]) plus twice
/// the first-order change over the arguments' balls,
///
/// * in `z`: `|∂Li_s/∂z| = |Li_{s−1}(z)/z|`, evaluated at 128 bits, grown by
///   `(1 − r/d)^{−max(1, |s − 2|)}` over the ball (`r` its radius, `d` the
///   distance to the branch point 1, where `Li_{s−1}` behaves like
///   `|z − 1|^{s−2}`).  No bound when the ball is not well inside (`r >
///   d/8`), nor when it meets the cut `[1, ∞)` with an imaginary part only
///   within its error of 0 (the side is undecidable; an exactly real `z > 1`
///   is on the cut, by convention below it);
/// * in `s`: `|∂Li_s/∂s|` from a forward difference at 128 bits with a step
///   of `2^−40·max(1, |s|)`, plus the difference's rounding.
#[allow(clippy::too_many_arguments)]
pub(super) fn error_bound(
    s: &BigFloat,
    bs: Bound,
    z: &Complex,
    bz: Bound,
    v: &Complex,
    w: &mut Wp<'_>,
) -> Bound {
    let prec = w.bits();
    if bs.is_unknown() || bz.is_unknown() {
        return Bound::UNKNOWN;
    }
    // Real on the real axis left of 1, and a rational function of `z` for
    // an exact order `s ≤ 0`.
    let nonpositive_order = bs.is_exact() && (s.is_zero() || (s.is_int() && s.is_negative()));
    let real = accuracy::is_exact(bs.im)
        && accuracy::exactly_real(z, bz)
        && (z.0.is_negative() || nonpositive_order);
    let lv = if real {
        accuracy::part_lg(&v.0)
    } else {
        accuracy::lg_abs(v)
    };
    if !lv.is_finite() {
        return Bound::UNKNOWN;
    }
    let mut total = lv - prec as f64 - 12.0;
    if !bz.is_exact() {
        match z_change(s, z, bz, w.cc) {
            Some(c) => total = accuracy::lsum(total, c),
            None => return Bound::UNKNOWN,
        }
    }
    if !bs.is_exact() {
        match s_change(s, bs.joint(), z, v, w.cc) {
            Some(c) => total = accuracy::lsum(total, c),
            None => return Bound::UNKNOWN,
        }
    }
    if real {
        Bound::real(total)
    } else if let Some(im) = cut_imaginary_bound(s, bs, z, bz, v, prec) {
        Bound { re: total, im }
    } else {
        Bound::both(total)
    }
}

/// The bound of the imaginary part `−π(ln x)^(s−1)/Γ(s)` on the cut (see
/// [`cut_imaginary`]) relative to itself: its rounding, and twice the
/// first-order change over the balls of `x` (`abs(∂ ln Im/∂x) = (s − 1)/(x·ln x)`,
/// at the lower end of the ball, which must stay within a quarter of `x − 1`)
/// and of `s` (`abs(∂ ln Im/∂s) = abs(ln ln x − ψ(s)) ≤ abs(ln ln x) + abs(ln s) + 1/s +
/// 1` at the lower end of its ball, which must stay above `s/2`).  `None` off
/// the cut, or for a zero imaginary part.
fn cut_imaginary_bound(
    s: &BigFloat,
    bs: Bound,
    z: &Complex,
    bz: Bound,
    v: &Complex,
    prec: usize,
) -> Option<f64> {
    if !on_cut(s, z)
        || !accuracy::exactly_real(z, bz)
        || !accuracy::is_exact(bs.im)
        || v.1.is_zero()
    {
        return None;
    }
    let li = accuracy::part_lg(&v.1);
    let mut total = li - prec as f64 - 12.0;
    let sf = to_f64(s)?;
    let xf = to_f64(&z.0)?;
    let lnx = (xf - 1.0).ln_1p();
    if lnx.is_nan() || lnx <= 0.0 {
        return None;
    }
    if !bz.is_exact() {
        let r = bz.re.exp2();
        if r.is_nan() || r > (xf - 1.0) / 4.0 {
            return None;
        }
        let lo = xf - r;
        let d = (sf + 1.0) / (lo * (lo - 1.0).ln_1p());
        // A change of at most a quarter: twice the first order covers it.
        if (d * r).is_nan() || d * r > 0.25 {
            return None;
        }
        let rel = d.log2() + bz.re + 1.0;
        total = accuracy::lsum(total, li + rel);
    }
    if !bs.is_exact() {
        let r = bs.re.exp2();
        if r.is_nan() || r > sf / 2.0 {
            return None;
        }
        let lo = sf - r;
        let d = lnx.ln().abs() + lo.ln().abs().max((sf + r).ln().abs()) + 1.0 / lo + 1.0;
        if (d * r).is_nan() || d * r > 0.25 {
            return None;
        }
        let rel = d.log2() + bs.re + 1.0;
        total = accuracy::lsum(total, li + rel);
    }
    total.is_finite().then_some(total)
}

/// Twice the change of `Li_s` over the ball of `z` (see [`error_bound`]).
fn z_change(s: &BigFloat, z: &Complex, bz: Bound, cc: &mut Consts) -> Option<f64> {
    let r = bz.joint();
    let undecided_side =
        !accuracy::exact_zero(&z.1, bz.im) && accuracy::part_contains_zero(&z.1, bz.im);
    if undecided_side {
        // Does the real part's interval reach [1, ∞)?
        let hi = z.0.add(&bigfloat::from_f64(r.exp2(), 64), LP, RM);
        if !super::bf_lt(&hi, &BigFloat::from_i32(1, 64)) {
            return None;
        }
    }
    let one = BigFloat::from_i32(1, 64);
    let d = (z.0.sub(&one, super::exact_bits(&z.0, LP), RM), z.1.clone());
    let ld = accuracy::lg_abs_low(&d);
    if ld.is_nan() || ld <= r + 3.0 {
        return None;
    }
    let s1 = minus_int(s, 1, LP);
    let li = polylog_general(&s1, z, &mut Wp::new(LP, cc)).ok()?;
    let lz = accuracy::lg_abs_low(z);
    let dl = accuracy::lg_abs(&li) - lz;
    if !dl.is_finite() {
        // Li_{s−1}(z) = 0 at 128 bits: its size is 2^−128 of its terms at
        // most; take the value's own size over |z| instead.
        return None;
    }
    let p = (to_f64(s)? - 2.0).abs().max(1.0);
    Some(dl + p * shrink(r - ld) + r + 1.0)
}

/// Twice the change of `Li_s(z)` over the ball `s ± 2^es` (see
/// [`error_bound`]).
fn s_change(s: &BigFloat, es: f64, z: &Complex, v: &Complex, cc: &mut Consts) -> Option<f64> {
    let sf = to_f64(s)?;
    let lh = (sf.abs().max(1.0).log2() - 40.0).floor();
    if es > lh - 8.0 {
        // A ball wider than the step: no estimate.
        return None;
    }
    let mut h = BigFloat::from_i32(1, 64);
    h.set_exponent(lh as i32 + 1);
    let moved = s.add(&h, super::exact_bits(s, LP), RM);
    let w = polylog_general(&moved, z, &mut Wp::new(LP, cc)).ok()?;
    let diff = c_sub(&w, v, LP + 64, RM);
    // |Δ|/h plus the rounding of both values (2^−(LP−4) of |Li|).
    let noise = accuracy::lg_abs(v) - (LP as f64 - 4.0) - lh;
    let deriv = accuracy::lsum(accuracy::lg_abs(&diff) - lh, noise);
    Some(deriv + es + 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use astro_float::Radix;

    fn close(x: &BigFloat, want: &str, bits: i32, cc: &mut Consts) -> bool {
        let w = BigFloat::parse(want, Radix::Dec, 400, RM, cc);
        let d = x.sub(&w, 400, RM);
        d.is_zero()
            || d.exponent()
                .is_some_and(|e| e < w.exponent().unwrap_or(0) - bits)
    }

    /// `ζ(−1, a) = −B₂(a)/2`: an integer `σ ≤ 0` ends the Euler–Maclaurin
    /// sum by itself (`(σ)_{2j} = 0`).  `a = 1/2 + i`: `13/24`, real.
    #[test]
    fn hurwitz_at_a_non_positive_integer_is_a_bernoulli_polynomial() {
        let mut cc = Consts::new().unwrap();
        let a = (BigFloat::from_f64(0.5, 64), BigFloat::from_i32(1, 64));
        let v = hurwitz(&BigFloat::from_i32(-1, 64), &a, &mut Wp::new(256, &mut cc)).unwrap();
        let want = BigFloat::from_i32(13, 64).div(&BigFloat::from_i32(24, 64), 400, RM);
        let d = v.value.0.sub(&want, 400, RM);
        assert!(d.is_zero() || d.exponent().is_some_and(|e| e < -200));
        assert!(v.value.1.is_zero() || v.value.1.exponent().is_some_and(|e| e < -200));
    }

    /// Next to an integer `σ` one factor of the Pochhammer symbol in the
    /// remainder bound is tiny; formed in `f64` it was 0, the sum stopped
    /// after one Bernoulli term, and `polylog(2 + 10⁻²⁰, 3 + i)` was wrong
    /// from its 21st digit.
    ///
    /// mpmath: `zeta(-1 - mpf(2)**-70, mpc(mpf(1)/4, mpf(1)/2))` (dps 100
    /// and 200) → `0.135416666666666666666373504570848456900621708517242889…
    /// + 0.1250000000000000000001218632544294143230559564251799939…i`.
    #[test]
    fn hurwitz_next_to_an_integer_order() {
        let mut cc = Consts::new().unwrap();
        let mut eps = BigFloat::from_i32(1, 64);
        eps.set_exponent(-69);
        let sigma = BigFloat::from_i32(-1, 64).sub(&eps, 200, RM);
        let a = (BigFloat::from_f64(0.25, 64), BigFloat::from_f64(0.5, 64));
        let v = hurwitz(&sigma, &a, &mut Wp::new(256, &mut cc)).unwrap();
        assert!(close(
            &v.value.0,
            "0.135416666666666666666373504570848456900621708517242889",
            170,
            &mut cc
        ));
        assert!(close(
            &v.value.1,
            "0.1250000000000000000001218632544294143230559564251799939",
            170,
            &mut cc
        ));
    }

    /// `e^{iπx}` is exact at the integers and half-integers.
    #[test]
    fn exp_i_pi_is_exact_at_quarter_turns() {
        let mut cc = Consts::new().unwrap();
        let e = exp_i_pi(&BigFloat::from_f64(-2.5, 64), &mut Wp::new(128, &mut cc));
        assert!(e.0.is_zero() && e.1 == BigFloat::from_i32(-1, 64));
        let e = exp_i_pi(&BigFloat::from_i32(7, 64), &mut Wp::new(128, &mut cc));
        assert!(e.1.is_zero() && e.0 == BigFloat::from_i32(-1, 64));
    }
}

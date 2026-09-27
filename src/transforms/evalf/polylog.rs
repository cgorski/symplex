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

use astro_float::{BigFloat, Consts, RoundingMode};
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

    fn add(&mut self, t: &Complex, wp: usize, rm: RoundingMode) {
        self.largest = self.largest.max(accuracy::lg_abs(t));
        self.value = c_add(&self.value, t, wp, rm);
    }
}

/// Is the value `z` of `polylog`'s argument outside the real interval
/// `[−1, 1]` (non-real, or `|z| > 1`), where this module evaluates it?
pub(super) fn off_unit_interval(z: &Complex) -> bool {
    !z.1.is_zero() || super::bf_gt(&z.0.abs(), &BigFloat::from_i32(1, 64))
}

/// Is `node` a `polylog(s, z)` with a real order and an argument this module
/// evaluates, both values in `cache`?
pub(super) fn is_general(
    arena: &crate::base::arena::Arena,
    node: &crate::base::node::ExprNode,
    cache: &rustc_hash::FxHashMap<crate::base::node::ExprId, Complex>,
) -> bool {
    let crate::base::node::ExprNode::Apply(sid, args) = node else {
        return false;
    };
    arena.lib_fn(*sid) == Some(crate::base::libfn::LibFn::PolyLog)
        && args.len() == 2
        && cache.get(&args[0]).is_some_and(|s| s.1.is_zero())
        && cache.get(&args[1]).is_some_and(off_unit_interval)
}

/// `Li_s(z)` for a real `s` and `z` off `[−1, 1]` (see the module
/// documentation), rounded to `prec + 16` bits: its absolute error is
/// below `2^−(prec+12)·|Li_s(z)|` (on the real part, for a real value).
pub(super) fn polylog_general(
    s: &BigFloat,
    z: &Complex,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<Complex, SymplexError> {
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
    let s_int = super::bf_as_int(s, rm, cc)?;
    let real = z.1.is_zero() && (z.0.is_negative() || s_int.is_some_and(|n| n <= 0));
    tracing::debug!(prec, s = s_f, "evalf: polylog off [-1, 1]");
    let cap = super::cancellation_cap(prec);
    let mut extra =
        (32 + accuracy::ceil_log2(prec) as usize + near_integer_bits(s, s_int, prec, rm)).min(cap);
    loop {
        let wp = prec + extra;
        let sum = value_at(s, s_int, s_f, z, wp, rm, cc).map_err(super::requested_at(prec))?;
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
            let re = super::round_to(sum.value.0, prec + 16, rm);
            let im = if real {
                BigFloat::new(prec + 16)
            } else {
                super::round_to(sum.value.1, prec + 16, rm)
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

/// `log₂(1/|s − n|)` for a non-integer `s` within `2⁻⁸` of an integer `n`
/// (else 0): the bits the `μ`-expansion loses to the cancelling poles of
/// `Γ(1 − s)` and `ζ(s − n + 1)`, taken as guard bits from the start (the
/// measured loss would cost a second pass).
fn near_integer_bits(s: &BigFloat, s_int: Option<i64>, prec: usize, rm: RoundingMode) -> usize {
    if s_int.is_some() {
        return 0;
    }
    let exact = super::exact_bits(s, prec);
    let n = s.add(&BigFloat::from_f64(0.5, 64), exact, rm).floor();
    let d = s.sub(&n, exact, rm);
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
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<Sum, SymplexError> {
    if let Some(n) = s_int
        && n <= 0
    {
        return nonpositive(n.unsigned_abs(), z, wp, rm);
    }
    let r2 = z.0.mul(&z.0, wp, rm).add(&z.1.mul(&z.1, wp, rm), wp, rm);
    let lz = 0.5 * lg2(&r2);
    let cheap = |r: f64| series_terms(r, s_f, wp).is_some_and(|k| k <= 3 * wp + 64);
    if lz <= LOG2_THREE_QUARTERS {
        return series(s, s_int, s_f, z, wp, rm, cc);
    }
    if !super::bf_gt(&r2, &BigFloat::from_i32(1, 64)) {
        return if cheap(lz.exp2().min(1.0)) {
            series(s, s_int, s_f, z, wp, rm, cc)
        } else {
            mu_expansion(s, s_int, s_f, z, wp, rm, cc)
        };
    }
    if lz >= 1.0 || cheap((-lz).exp2()) {
        inversion(s, s_int, s_f, z, wp, rm, cc)
    } else {
        mu_expansion(s, s_int, s_f, z, wp, rm, cc)
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// A finite `f64` value of `x` (`None` beyond the range or for NaN).
fn to_f64(x: &BigFloat) -> Option<f64> {
    super::bigfloat_to_f64_rounded(x, RoundingMode::ToEven)
        .ok()
        .filter(|v| v.is_finite())
}

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
fn scale(z: &Complex, x: &BigFloat, wp: usize, rm: RoundingMode) -> Complex {
    (z.0.mul(x, wp, rm), z.1.mul(x, wp, rm))
}

/// `z/n` for a positive integer `n`.
fn div_int(z: &Complex, n: u64, wp: usize, rm: RoundingMode) -> Complex {
    let d = BigFloat::from_u64(n, 64);
    (z.0.div(&d, wp, rm), z.1.div(&d, wp, rm))
}

/// `w^e = exp(e·Ln w)` (principal branch) for a real `e`, to `wp` bits
/// relative: the exponent is formed with the bits of its magnitude as guard.
fn pow_real(w: &Complex, e: &BigFloat, wp: usize, rm: RoundingMode, cc: &mut Consts) -> Complex {
    let ef = to_f64(e).unwrap_or(f64::INFINITY).abs();
    let lw = (accuracy::lg_abs(w) * LN2).abs() + 4.0;
    let p = wp + super::magnitude_bits(ef * lw) + 8;
    let l = super::c_ln(w, p, rm, cc);
    let x = scale(&l, e, p, rm);
    let v = super::c_exp(&x, p, rm, cc);
    (super::round_to(v.0, wp, rm), super::round_to(v.1, wp, rm))
}

/// `x − c` exactly for a small integer `c`.
fn minus_int(x: &BigFloat, c: i64, wp: usize, rm: RoundingMode) -> BigFloat {
    x.sub(
        &BigFloat::from_i128(i128::from(c), 128),
        super::exact_bits(x, wp),
        rm,
    )
}

/// `e^{iπx}`, exact where `x` is an integer or half-integer.
fn exp_i_pi(x: &BigFloat, wp: usize, rm: RoundingMode, cc: &mut Consts) -> Complex {
    // x′ = x − 2·round(x/2), exactly (a binary x/2 is exact).
    let exact = super::exact_bits(x, wp);
    let half_x = x.div(&BigFloat::from_i32(2, 64), exact, rm);
    let n = half_x.add(&BigFloat::from_f64(0.5, 64), exact, rm).floor();
    let xr = x.sub(&n.mul(&BigFloat::from_i32(2, 64), exact, rm), exact, rm);
    let twice = xr.mul(&BigFloat::from_i32(2, 64), exact, rm);
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
    let t = cc.pi(wp + 8, rm).clone().mul(&xr, wp + 8, rm);
    (t.cos(wp, rm, cc), t.sin(wp, rm, cc))
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
fn nonpositive(n: u64, z: &Complex, wp: usize, rm: RoundingMode) -> Result<Sum, SymplexError> {
    if n > 5000 {
        return Err(super::unevaluable(
            "polylog of an order below -5000 off [-1, 1] not supported in evalf",
        ));
    }
    let one = BigFloat::from_i32(1, 64);
    let omz = (one.sub(&z.0, super::exact_bits(&z.0, wp), rm), z.1.neg());
    if omz.0.is_zero() && omz.1.is_zero() {
        return Err(super::unevaluable("polylog(s, 1) diverges for s ≤ 1"));
    }
    let w = c_div(z, &omz, wp, rm);
    let mut k_fact = BigInt::from(1);
    let mut w_pow = w.clone();
    let mut sum = Sum::new(wp);
    for k in 0..=n {
        if k > 0 {
            k_fact *= BigInt::from(k);
            w_pow = c_mul(&w_pow, &w, wp, rm);
        }
        let s2 = super::stirling2(n + 1, k + 1)
            .ok_or_else(|| super::unevaluable("polylog: Stirling number overflow"))?;
        let c = super::ratio_to_bigfloat(&Ratio::from_integer(&k_fact * s2), wp, rm);
        sum.add(&scale(&w_pow, &c, wp, rm), wp, rm);
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
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<Sum, SymplexError> {
    let lr = accuracy::lg_abs(z);
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
            zk = c_mul(&zk, z, wp, rm);
        }
        let p = if k == 1 {
            one.clone()
        } else {
            match s_int {
                Some(n) => {
                    // n ≥ 1 here (the non-positive orders are rational).
                    let kn =
                        BigFloat::from_u64(k as u64, 64).powi(n.unsigned_abs() as usize, wp, rm);
                    one.div(&kn, wp, rm)
                }
                None => {
                    let q = spf[k] as usize;
                    if q == k {
                        super::bf_pow(&BigFloat::from_u64(k as u64, 64), &neg_s, wp, rm, cc)
                    } else {
                        powers[q - 1].mul(&powers[k / q - 1], wp, rm)
                    }
                }
            }
        };
        let term = scale(&zk, &p, wp, rm);
        if s_int.is_none() {
            powers.push(p);
        }
        sum.add(&term, wp, rm);
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

/// `ln z` accurate next to `z = 1`: `ln|z| = ½ ln(1 + d)` with `d = |z|² −
/// 1 = (x − 1)(x + 1) + y²` formed with `x − 1` exact, and `ln(1 + d) =
/// 2 atanh(t)`, `t = d/(2 + d)`, summed for `|d| < 1/4`.
fn ln_near_one(z: &Complex, wp: usize, rm: RoundingMode, cc: &mut Consts) -> Complex {
    let p = wp + 16;
    let one = BigFloat::from_i32(1, 64);
    let xm1 = z.0.sub(&one, super::exact_bits(&z.0, p), rm);
    let xp1 = z.0.add(&one, super::exact_bits(&z.0, p), rm);
    let d = xm1.mul(&xp1, p, rm).add(&z.1.mul(&z.1, p, rm), p, rm);
    let arg = super::atan2_bf(&z.1, &z.0, wp, rm, cc);
    let small = d.is_zero() || d.exponent().is_some_and(|e| e <= -2);
    let ln_abs = if !small {
        one.add(&d, p, rm)
            .ln(p, rm, cc)
            .div(&BigFloat::from_i32(2, 64), wp, rm)
    } else if d.is_zero() {
        BigFloat::new(wp)
    } else {
        // ½ ln(1 + d) = t + t³/3 + t⁵/5 + …, |t| < 1/7.
        let t = d.div(&BigFloat::from_i32(2, 64).add(&d, p, rm), p, rm);
        let t2 = t.mul(&t, p, rm);
        let mut pw = t.clone();
        let mut acc = t.clone();
        for k in 1..(p / 2 + 8) {
            pw = pw.mul(&t2, p, rm);
            let term = pw.div(&BigFloat::from_u64((2 * k + 1) as u64, 64), p, rm);
            acc = acc.add(&term, p, rm);
            if super::negligible(&term, &acc, p) {
                break;
            }
        }
        super::round_to(acc, wp, rm)
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
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<Sum, SymplexError> {
    let mu = ln_near_one(z, wp, rm, cc);
    let neg_mu = c_neg(&mu);
    let lmu = accuracy::lg_abs(&mu);
    // |μ|/2π, below 0.6 for 3/4 < |z| < 2.
    let rho = (lmu - LOG2_TWO_PI).exp2();
    if rho.is_nan() || rho >= 0.7 {
        return Err(super::unevaluable(
            "polylog: ln z outside the expansion's disc",
        ));
    }
    let mut sum = Sum::new(wp);
    if s_int.is_none() {
        // Γ(1 − s)(−μ)^{s−1}, `1 − s` exact (see `polylog_near_one_at`).
        let one_minus_s = BigFloat::from_i32(1, 64).sub(s, super::exact_bits(s, wp), rm);
        let g = super::arb_gamma_real(&one_minus_s, wp, rm, cc)?;
        let s_minus_1 = minus_int(s, 1, wp, rm);
        let lead = scale(&pow_real(&neg_mu, &s_minus_1, wp, rm, cc), &g, wp, rm);
        sum.add(&lead, wp, rm);
    }
    let max_terms = 8 * wp + 256 + s_f.max(0.0) as usize;
    let minus_one = BigFloat::from_i32(-1, 64);
    let mut left: Option<super::ZetaLeft> = None;
    let mut mk = c_one(wp); // μ^k/k!
    for k in 0..max_terms {
        if k > 0 {
            mk = div_int(&c_mul(&mk, &mu, wp, rm), k as u64, wp, rm);
        }
        let term = match s_int {
            Some(n) if k as i64 + 1 == n => {
                // μ^{n−1}/(n−1)!·(H_{n−1} − ln(−μ))
                let one = BigFloat::from_i32(1, wp);
                let mut h = BigFloat::new(wp);
                for j in 1..n {
                    h = h.add(&one.div(&BigFloat::from_i64(j, 64), wp, rm), wp, rm);
                }
                let l = super::c_ln(&neg_mu, wp, rm, cc);
                c_mul(&mk, &(h.sub(&l.0, wp, rm), l.1.neg()), wp, rm)
            }
            Some(n) => {
                let m = n - k as i64;
                let zeta = if m <= 0 {
                    super::zeta_nonpositive_int(m, wp, rm)
                } else {
                    super::arb_zeta(&BigFloat::from_i64(m, 64), wp, rm, cc)?
                };
                scale(&mk, &zeta, wp, rm)
            }
            None => {
                let arg = minus_int(s, k as i64, wp, rm);
                let zeta = if left.is_some() || super::bf_lt(&arg, &minus_one) {
                    if left.is_none() {
                        left = Some(super::ZetaLeft::new(&arg, wp, rm, cc)?);
                    }
                    match left.as_mut() {
                        Some(l) => l.next(rm),
                        None => super::arb_zeta(&arg, wp, rm, cc)?,
                    }
                } else {
                    super::arb_zeta(&arg, wp, rm, cc)?
                };
                scale(&mk, &zeta, wp, rm)
            }
        };
        sum.add(&term, wp, rm);
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

/// The inversion formula for `|z| > 1` (see the module documentation).
#[allow(clippy::too_many_arguments)]
fn inversion(
    s: &BigFloat,
    s_int: Option<i64>,
    s_f: f64,
    z: &Complex,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<Sum, SymplexError> {
    // a = ½ + ln(−z)/(2πi) = (½ + arg(−z)/2π) − i·ln|z|/2π with the
    // principal arg(−z) = arg z ∓ π: Re a = arg z/2π above the real axis
    // (and for z < 0), 1 + arg z/2π below it and on the cut z > 1 (arg(−z)
    // = π there).  From arg z itself, not ½ + arg(−z)/2π: just above the
    // cut the latter cancels to 0 (`z = 73/8 + 10⁻⁵⁸·i`).
    let pi = cc.pi(wp, rm).clone();
    let two_pi = pi.mul(&BigFloat::from_i32(2, 64), wp, rm);
    let arg = super::atan2_bf(&z.1, &z.0, wp, rm, cc);
    let upper = (z.1.is_positive() && !z.1.is_zero()) || (z.1.is_zero() && z.0.is_negative());
    let turn = arg.div(&two_pi, wp, rm);
    let re_a = if upper {
        turn
    } else {
        BigFloat::from_i32(1, 64).add(&turn, wp, rm)
    };
    let ln_abs = super::c_abs(z, wp + 8, rm).ln(wp, rm, cc);
    let a = (re_a, ln_abs.div(&two_pi, wp, rm).neg());
    // For a large order the Hurwitz sum cancels about `1.77·s` bits (terms
    // of `|w|^(s−1)`, `|w| ≈ s/5`, against `|ζ(1 − s, a)| ≈ 2Γ(s)/(2π)^s`):
    // beyond the cap, refused before the sum (`polylog(1000.5, 3)` spent a
    // second on it).
    if 1.5 * s_f > super::cancellation_cap(wp) as f64 {
        return Err(super::special_exhausted(wp));
    }
    let inv_z = c_div(&c_one(wp), z, wp, rm);
    let li = series(s, s_int, s_f, &inv_z, wp, rm, cc)?;
    let sigma = BigFloat::from_i32(1, 64).sub(s, super::exact_bits(s, wp), rm);
    let hz = hurwitz(&sigma, &a, wp, rm, cc)?;
    let gamma = super::arb_gamma_real(s, wp, rm, cc)?;
    let c = super::bf_pow(&two_pi, s, wp, rm, cc).div(&gamma, wp, rm);
    let e1 = exp_i_pi(s, wp, rm, cc);
    let half_s = s.div(&BigFloat::from_i32(2, 64), super::exact_bits(s, wp), rm);
    let e2 = exp_i_pi(&half_s, wp, rm, cc);
    let t1 = c_neg(&c_mul(&e1, &li.value, wp, rm));
    let t2 = scale(&c_mul(&e2, &hz.value, wp, rm), &c, wp, rm);
    let lc = lg2(&c);
    let mut sum = Sum::new(wp);
    sum.add(&t1, wp, rm);
    sum.add(&t2, wp, rm);
    sum.largest = sum.largest.max(li.largest).max(hz.largest + lc);
    Ok(sum)
}

/// The Hurwitz zeta function `ζ(σ, a)` for a real `σ ≠ 1` and `0 < Re a ≤
/// 1` by Euler–Maclaurin (see the module documentation).
fn hurwitz(
    sigma: &BigFloat,
    a: &Complex,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<Sum, SymplexError> {
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
            a.0.add(&BigFloat::from_u64(k as u64, 64), exact, rm),
            a.1.clone(),
        );
        sum.add(&pow_real(&ak, &neg_sigma, wp, rm, cc), wp, rm);
    }
    let w = (
        a.0.add(&BigFloat::from_u64(n as u64, 64), exact, rm),
        a.1.clone(),
    );
    let base = pow_real(&w, &neg_sigma, wp, rm, cc); // w^{−σ}
    let sigma_minus_1 = minus_int(sigma, 1, wp, rm);
    let integral = c_div(
        &c_mul(&base, &w, wp, rm),
        &(sigma_minus_1, BigFloat::new(wp)),
        wp,
        rm,
    );
    sum.add(&integral, wp, rm);
    sum.add(&div_int(&base, 2, wp, rm), wp, rm);
    let inv_w = c_div(&c_one(wp), &w, wp, rm);
    let inv_w2 = c_mul(&inv_w, &inv_w, wp, rm);
    let lw = accuracy::lg_abs_low(&w);
    let lre = lg2(&w.0) - 1e-9;
    let mut pw = inv_w; // w^{−(2j−1)}
    // c_j = (σ)_{2j−1}/(2j)!
    let mut c = sigma.div(&BigFloat::from_i32(2, 64), wp, rm);
    let mut lpoch = 0.0f64; // log₂|(σ)_{2j}|
    let max_m = 4 * wp + 2 * sf.abs() as usize + 64;
    for j in 1..=max_m {
        if j > 1 {
            let f1 = minus_int(sigma, -(2 * j as i64 - 3), wp, rm);
            let f2 = minus_int(sigma, -(2 * j as i64 - 2), wp, rm);
            c = c.mul(&f1, wp, rm).mul(&f2, wp, rm).div(
                &BigFloat::from_u64(((2 * j - 1) * (2 * j)) as u64, 64),
                wp,
                rm,
            );
            pw = c_mul(&pw, &inv_w2, wp, rm);
        }
        let b = super::ratio_to_bigfloat(&super::bernoulli::even(j), wp, rm);
        let t = scale(&c_mul(&base, &pw, wp, rm), &b.mul(&c, wp, rm), wp, rm);
        sum.add(&t, wp, rm);
        let jf = j as f64;
        // The factors σ + 2j − 2 and σ + 2j − 1 exactly: next to an integer
        // order one of them is tiny (`σ = −1 − 10⁻²⁰`), and in `f64` it was 0
        // — an "exact" sum after one term, wrong from the 21st digit.
        let g1 = minus_int(sigma, -(2 * j as i64 - 2), wp, rm);
        let g2 = minus_int(sigma, -(2 * j as i64 - 1), wp, rm);
        lpoch += lg2(&g1) + lg2(&g2);
        if lpoch == f64::NEG_INFINITY {
            // (σ)_{2j} = 0: the sum is exact (an integer σ ≤ 0).
            return Ok(sum);
        }
        let p = sf + 2.0 * jf;
        if p >= 2.5 {
            let int = ((1.0 - p) * lre - (p - 1.0).log2()).min(LOG2_HALF_PI + (1.0 - p) * lw);
            let rem = 2.0 + lpoch - 2.0 * jf * LOG2_TWO_PI + int;
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
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Bound {
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
        match z_change(s, z, bz, rm, cc) {
            Some(c) => total = accuracy::lsum(total, c),
            None => return Bound::UNKNOWN,
        }
    }
    if !bs.is_exact() {
        match s_change(s, bs.joint(), z, v, rm, cc) {
            Some(c) => total = accuracy::lsum(total, c),
            None => return Bound::UNKNOWN,
        }
    }
    if real {
        Bound::real(total)
    } else {
        Bound::both(total)
    }
}

/// Twice the change of `Li_s` over the ball of `z` (see [`error_bound`]).
fn z_change(
    s: &BigFloat,
    z: &Complex,
    bz: Bound,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Option<f64> {
    let r = bz.joint();
    let undecided_side =
        !accuracy::exact_zero(&z.1, bz.im) && accuracy::part_contains_zero(&z.1, bz.im);
    if undecided_side {
        // Does the real part's interval reach [1, ∞)?
        let hi = z.0.add(&BigFloat::from_f64(r.exp2(), 64), LP, rm);
        if !super::bf_lt(&hi, &BigFloat::from_i32(1, 64)) {
            return None;
        }
    }
    let one = BigFloat::from_i32(1, 64);
    let d = (z.0.sub(&one, super::exact_bits(&z.0, LP), rm), z.1.clone());
    let ld = accuracy::lg_abs_low(&d);
    if ld.is_nan() || ld <= r + 3.0 {
        return None;
    }
    let s1 = minus_int(s, 1, LP, rm);
    let li = polylog_general(&s1, z, LP, rm, cc).ok()?;
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
fn s_change(
    s: &BigFloat,
    es: f64,
    z: &Complex,
    v: &Complex,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Option<f64> {
    let sf = to_f64(s)?;
    let lh = (sf.abs().max(1.0).log2() - 40.0).floor();
    if es > lh - 8.0 {
        // A ball wider than the step: no estimate.
        return None;
    }
    let mut h = BigFloat::from_i32(1, 64);
    h.set_exponent(lh as i32 + 1);
    let moved = s.add(&h, super::exact_bits(s, LP), rm);
    let w = polylog_general(&moved, z, LP, rm, cc).ok()?;
    let diff = c_sub(&w, v, LP + 64, rm);
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
        let rm = RoundingMode::ToEven;
        let w = BigFloat::parse(want, Radix::Dec, 400, rm, cc);
        let d = x.sub(&w, 400, rm);
        d.is_zero()
            || d.exponent()
                .is_some_and(|e| e < w.exponent().unwrap_or(0) - bits)
    }

    /// `ζ(−1, a) = −B₂(a)/2`: an integer `σ ≤ 0` ends the Euler–Maclaurin
    /// sum by itself (`(σ)_{2j} = 0`).  `a = 1/2 + i`: `13/24`, real.
    #[test]
    fn hurwitz_at_a_non_positive_integer_is_a_bernoulli_polynomial() {
        let rm = RoundingMode::ToEven;
        let mut cc = Consts::new().unwrap();
        let a = (BigFloat::from_f64(0.5, 64), BigFloat::from_i32(1, 64));
        let v = hurwitz(&BigFloat::from_i32(-1, 64), &a, 256, rm, &mut cc).unwrap();
        let want = BigFloat::from_i32(13, 64).div(&BigFloat::from_i32(24, 64), 400, rm);
        let d = v.value.0.sub(&want, 400, rm);
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
        let rm = RoundingMode::ToEven;
        let mut cc = Consts::new().unwrap();
        let mut eps = BigFloat::from_i32(1, 64);
        eps.set_exponent(-69);
        let sigma = BigFloat::from_i32(-1, 64).sub(&eps, 200, rm);
        let a = (BigFloat::from_f64(0.25, 64), BigFloat::from_f64(0.5, 64));
        let v = hurwitz(&sigma, &a, 256, rm, &mut cc).unwrap();
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
        let rm = RoundingMode::ToEven;
        let mut cc = Consts::new().unwrap();
        let e = exp_i_pi(&BigFloat::from_f64(-2.5, 64), 128, rm, &mut cc);
        assert!(e.0.is_zero() && e.1 == BigFloat::from_i32(-1, 64));
        let e = exp_i_pi(&BigFloat::from_i32(7, 64), 128, rm, &mut cc);
        assert!(e.1.is_zero() && e.0 == BigFloat::from_i32(-1, 64));
    }
}

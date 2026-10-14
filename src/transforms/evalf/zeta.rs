//! `ζ(s)`, `η(s)`, `ψ(z)`, `ψ⁽ᵐ⁾(z)` and `Γ(z)` at non-real arguments, and
//! the error bounds of those nodes.  (The real routines are `arb_zeta`,
//! `arb_dirichlet_eta`, `arb_digamma`, `arb_polygamma` and `arb_gamma_real`
//! in `evalf.rs`.)  Before 0.44 every one of them refused a non-real
//! `zeta(1/2 + 14i)`, `polygamma(5, 5 + i)`, `gamma(1 + i)`,
//! `dirichlet_eta(1/2 + i)`, `harmonic(1/2 + i)`, `beta(1 + i, 2)`.
//!
//! # The Hurwitz zeta function by Euler–Maclaurin
//!
//! `ζ(s, a) = Σ_{k≥0} (a + k)^(−s)` (principal powers) for a complex `s ≠ 1`
//! and `Re a > 0` — or any `a` off the poles `0, −1, −2, …` for an integer
//! `s`, where the powers are single-valued — is summed directly up to `N`
//! and its tail by the Euler–Maclaurin formula at `w = a + N`:
//!
//! ```text
//! ζ(s, a) = Σ_{k<N} (a+k)^(−s) + w^(1−s)/(s−1) + w^(−s)/2
//!         + Σ_{j=1}^{M} B₂ⱼ/(2j)!·(s)₂ⱼ₋₁·w^(1−s−2j) + R_M,
//! |R_M| ≤ 4/(2π)^(2M)·abs((s)₂ₘ)·e^(abs(τ)·abs(arg w))·∫_N^∞ abs(a + t)^(−p) dt,
//! ```
//!
//! `s = σ + iτ`, `p = σ + 2M > 1`, `(s)ₙ` the rising factorial (DLMF 25.11.5,
//! 2.10.1).  The remainder is `−∫_N^∞ B̃₂ₘ(t)/(2M)!·f^(2M)(t) dt` for `f(t) =
//! (a + t)^(−s)`, with `abs(B̃₂ₘ) ≤ abs(B₂ₘ) ≤ 4(2M)!/(2π)^(2M)` and
//! `abs((a + t)^(−s−2M)) = abs(a + t)^(−p)·e^(τ·arg(a + t))`, where `abs(arg(a + t))`
//! falls from `abs(arg w) ≤ abs(Im a)/Re w` as `t` grows; the integral is at most
//! `min((Re w)^(1−p)/(p − 1), (π/2)·abs(w)^(1−p))` (the second for `p ≥ 2`:
//! `abs(w + t)² ≥ abs(w)² + t²` for `Re w ≥ 0`).  `N` is the smallest (in steps
//! of a quarter) for which an `f64` simulation of the bound reaches the
//! precision; the sum stops at the first `M` whose bound is below `2^−wp` of
//! the largest term.  `k^(−s)` is a power only for a prime `k` when `a = 1`,
//! a product of two earlier ones otherwise.  The structure — direct terms,
//! then the Euler–Maclaurin tail with `N` grown until the corrections
//! converge — is mpmath's `_hurwitz_em` (`mpmath/functions/zeta.py`, BSD),
//! with the remainder bound in place of its size test.
//!
//! * `ζ(s)` far up the critical strip (`abs(t) > 500·wp` or `abs(t) ≥ 10⁵`):
//!   the Riemann–Siegel formula with Arias de Reyna's bounds (`rszeta.rs`,
//!   since 0.45), about `√(t/2π)` terms instead of `t/2π`; where its
//!   conditions fail, the sum below.
//! * `ζ(s) = ζ(s, 1)` for `σ ≥ 0`; left of 0 the functional equation
//!   `ζ(s) = 2^s π^(s−1) sin(πs/2) Γ(1−s) ζ(1−s)` (DLMF 25.4.1), the factors
//!   but `sin` as one exponential of `s ln 2 + (s − 1) ln π + ln Γ(1 − s)`
//!   (`loggamma.rs`), `sin(πs/2)` with `s/2` reduced exactly.
//! * `η(s) = (1 − 2^(1−s))·ζ(s)` (DLMF 25.2.3), the first factor `−expm1((1 −
//!   s) ln 2)` with as many guard bits as it is small.
//! * `ψ⁽ᵐ⁾(z) = (−1)^(m+1)·m!·ζ(m + 1, z)` for `m ≥ 1` (DLMF 25.11.12), and
//!   `ψ(z) = ψ(w) − Σ_{k<N} 1/(z + k)` with `ψ(w) = ln w − 1/(2w) − Σ_{j=1}^M
//!   B₂ⱼ/(2j)·w^(−2j) − R_M`: the same formula for `f(t) = 1/(w + t)`, whose
//!   remainder is the bound above with `s = 1`.
//! * `Γ(z) = exp(ln Γ(z))` with `ln Γ` from `loggamma.rs` at as many more bits
//!   as its magnitude has; `z! = Γ(z + 1)`, `B(a, b) = Γ(a)Γ(b)/Γ(a + b)`,
//!   `harmonic(z) = ψ(z + 1) + γ`.
//!
//! Every truncation is below `2^−wp` of the largest term summed; the loss —
//! the largest term over the result (the zeros of `ζ` on the critical line,
//! of `ψ`) — is measured and the evaluation repeated with that many more bits,
//! up to `cancellation_cap`, as in `polylog.rs`.
//!
//! # Error bounds
//!
//! The node's bound is its own `2^−(prec+12)` relative, plus twice the
//! argument's radius `r` times a bound on the derivative over the argument's
//! ball (`u` within `r` of the argument; no bound when the ball reaches
//! within eight radii of a pole):
//!
//! | function | bound on the derivative over the ball |
//! |---|---|
//! | `ζ(u)` | the Euler–Maclaurin formula at `N = ⌈abs(s)⌉ + 2` with `M = max(1, ⌈(2 − σ′)/2⌉)` differentiated term by term: `Σ_{k<N} ln k·k^(−σ′)`, `N^(1−σ′)(ln N/δ + 1/δ²)`, `ln N·N^(−σ′)/2`, `abs(B₂ⱼ)/(2j)!·N^(1−σ′−2j)·(P′₂ⱼ₋₁(x) + ln N·P₂ⱼ₋₁(x))` and the remainder's derivative, with `σ′ = σ − r`, `x = abs(s) + r`, `δ = abs(s − 1) − r` and `Pₙ(x) = x(x+1)⋯(x+n−1) ≥ abs((u)ₙ)` (non-negative coefficients) |
//! | `η(u)` | `ln 2·2^(1−σ′)·abs(ζ) + (1 + 2^(1−σ′))·abs(ζ′)`, `abs(ζ)` bounded from the same formula |
//! | `ψ⁽ᵐ⁾(u)` (`ψ` for `m = 0`) | `(m+1)!·Σ_{k≥0} (abs(z + k) − r)^(−m−2)`, summed to `Re(z + K) ≥ 1` and bounded by an integral beyond |
//! | `Γ(u)` | `abs(Γ(z))·(e^(Ψr) − 1)/r`, `Ψ` the bound on `abs(ψ)` of `loggamma.rs` |
//! | `B(a, b)` | `abs(B)·(Ψ(a) + Ψ(a + b))` in `a` (and in `b`), the change `abs(B)·(e^t − 1)` |

use super::wp::{RM, Wp};
use crate::base::bigfloat::to_f64;
use astro_float::{BigFloat, Consts};

use super::accuracy::{self, Bound};
use crate::base::bigcomplex::{Complex, c_add, c_div, c_mul, c_neg, c_one, c_sub, c_zero};
use crate::base::errors::SymplexError;

const LN2: f64 = std::f64::consts::LN_2;
const LOG2E: f64 = std::f64::consts::LOG2_E;
/// `log₂ 2π`.
const LOG2_TWO_PI: f64 = 2.651_496_129_472_319;
/// `log₂(π/2)`.
const LOG2_HALF_PI: f64 = 0.651_496_129_472_318_8;

/// Most terms summed directly (`ζ(s)` for `abs(Im s)` up to about `5·10⁵`).
const MAX_TERMS: usize = 100_000;

/// Most terms of the derivative bound summed one by one (`zeta_ball`).
const DIRECT_BALL_TERMS: usize = 100_000;

/// Largest order of `ψ⁽ᵐ⁾` at a non-real argument (as for a real one,
/// beyond which `eval` leaves it to `polygamma_large`).
pub(super) const MAX_POLYGAMMA_ORDER: u32 = 10_000;

/// A sum and `log₂` of the largest term that went into it (its rounding
/// error is a few units of `2^−wp` of that term).
pub(super) struct Sum {
    pub(super) value: Complex,
    pub(super) largest: f64,
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

/// `log₂ √(2^(2a) + 2^(2b))` for `log₂` magnitudes `a`, `b`.
fn lg_hypot(a: f64, b: f64) -> f64 {
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    if hi == f64::NEG_INFINITY {
        return hi;
    }
    hi + 0.5 * (2.0 * (lo - hi)).exp2().ln_1p() * LOG2E
}

/// `log₂(2^a + 2^b)`.
fn lsum(a: f64, b: f64) -> f64 {
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    if hi == f64::NEG_INFINITY || !hi.is_finite() {
        return hi;
    }
    hi + (lo - hi).exp2().ln_1p() * LOG2E
}

/// `x + c` exactly for a small integer `c`.
pub(super) fn plus_int(x: &BigFloat, c: i64, wp: usize) -> BigFloat {
    x.add(
        &BigFloat::from_i128(i128::from(c), 128),
        super::exact_bits(x, wp),
        RM,
    )
}

/// `log₂ abs(s + c)` for a complex `s` and an integer `c`, the real part
/// `σ + c` formed exactly (next to an integer order it is tiny).
fn lg_shifted(s: &Complex, c: i64, wp: usize) -> f64 {
    lg_hypot(lg2(&plus_int(&s.0, c, wp)), lg2(&s.1))
}

/// `z/n` for a positive integer `n`.
fn div_int(z: &Complex, n: u64, wp: usize) -> Complex {
    let d = BigFloat::from_u64(n, 64);
    (z.0.div(&d, wp, RM), z.1.div(&d, wp, RM))
}

/// `z·x` for a real `x`.
fn scale(z: &Complex, x: &BigFloat, wp: usize) -> Complex {
    (z.0.mul(x, wp, RM), z.1.mul(x, wp, RM))
}

fn round_c(v: Complex, prec: usize) -> Complex {
    (super::round_to(v.0, prec), super::round_to(v.1, prec))
}

/// `b^(−s) = exp(−s·Ln b)` (principal branch) for a complex `b ≠ 0`, to `wp`
/// bits relative: the exponent with the bits of its magnitude as guard.
pub(super) fn pow_neg(b: &Complex, s: &Complex, w: &mut Wp<'_>) -> Complex {
    let wp = w.bits();
    let ls = accuracy::lg_abs(s).exp2();
    let lb = (accuracy::lg_abs(b) * LN2).abs() + 4.0;
    let p = wp + super::magnitude_bits(ls * lb) + 8;
    let l = super::c_ln(b, &mut w.at(p));
    let x = c_neg(&c_mul(s, &l, p, RM));
    round_c(super::c_exp(&x, &mut w.at(p)), wp)
}

/// `b^(−m)` for a positive integer `m` by binary powering, `log₂ m` guard bits
/// for its roundings.
fn pow_neg_int(b: &Complex, m: u64, wp: usize) -> Complex {
    let p = wp + accuracy::ceil_log2(m as usize) as usize + 8;
    let v = crate::base::bigcomplex::c_powi(b, m as usize, p, RM);
    round_c(c_div(&c_one(p), &v, p, RM), wp)
}

/// `s` as a positive integer, when it is one (real, integral, `≤ 2⁵³`).
fn positive_integer(s: &Complex) -> Option<u64> {
    if !s.1.is_zero() || !s.0.is_int() || !super::bf_strictly_positive(&s.0) {
        return None;
    }
    let v = to_f64(&s.0)?;
    (v <= 9.0e15).then_some(v as u64)
}

thread_local! {
    /// `B₂ⱼ` rounded to the working precision it was last asked at (the
    /// expansion in `ln z` of `polylog.rs` sums a Hurwitz function per term).
    static BERNOULLI: std::cell::RefCell<(usize, Vec<BigFloat>)> =
        const { std::cell::RefCell::new((0, Vec::new())) };
}

/// `B₂ⱼ` (`j ≥ 1`) rounded to `wp` bits.
fn bernoulli(j: usize, wp: usize) -> BigFloat {
    BERNOULLI.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.0 != wp {
            *cache = (wp, Vec::new());
        }
        while cache.1.len() < j {
            let k = cache.1.len() + 1;
            cache
                .1
                .push(super::ratio_to_bigfloat(&super::bernoulli::even(k), wp));
        }
        cache
            .1
            .get(j.wrapping_sub(1))
            .cloned()
            .unwrap_or_else(|| super::ratio_to_bigfloat(&super::bernoulli::even(j), wp))
    })
}

/// The smallest prime factor of every integer up to `n`.
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

// ── Euler–Maclaurin ─────────────────────────────────────────────────────────

/// `log₂` of the Euler–Maclaurin remainder bound after `M` corrections (see
/// the module documentation): `lpoch = log₂ abs((s)₂ₘ)`, `p = σ + 2M`,
/// `lre = log₂ Re w`, `lw = log₂ abs(w)` (lower bounds), `arg = log₂ e^(abs(τ)·abs(arg w))`.
/// `None` while `p ≤ 1`.
fn em_remainder(m: usize, lpoch: f64, p: f64, lre: f64, lw: f64, arg: f64) -> Option<f64> {
    if p <= 1.0 + 1e-9 {
        return None;
    }
    let mut int = (1.0 - p) * lre - (p - 1.0).log2();
    if p >= 2.0 {
        int = int.min(LOG2_HALF_PI + (1.0 - p) * lw);
    }
    Some(2.0 + lpoch - 2.0 * m as f64 * LOG2_TWO_PI + arg + int)
}

/// The number of direct terms `N ≥ min_n` for which the remainder bound of
/// `ζ(s, a)` at `w = a + N` (`s = sr + i·si`, `a = ar + i·ai`) falls below
/// `2^−wp` of the leading terms (`log₂` magnitude `lead`, or the larger of
/// `w^(1−s)/(s − 1)` and the first direct term, `l_s1 = log₂ abs(s − 1)` formed
/// exactly: next to the pole `sr − 1` is 0 in `f64`) within
/// [`max_corrections`] corrections, by an `f64` simulation (`N` grown by a
/// quarter at a time).  Only a choice of `N`: the sum stops on the bound
/// itself.  `None` beyond [`MAX_TERMS`].
#[allow(clippy::too_many_arguments)]
fn choose_terms(
    sr: f64,
    si: f64,
    ar: f64,
    ai: f64,
    min_n: usize,
    lead: Option<f64>,
    l_s1: f64,
    wp: usize,
) -> Option<usize> {
    let jmax = max_corrections(wp, sr.hypot(si));
    // The corrections shrink only while abs(s + 2j) < 2π·abs(w): no N below
    // abs(s)/2π − abs(a) can do.
    let least = (sr.hypot(si) / (2.0 * std::f64::consts::PI) - ar.hypot(ai)).max(0.0);
    if least > MAX_TERMS as f64 {
        return None;
    }
    let mut n = min_n.max(1).max(least as usize);
    loop {
        if n > MAX_TERMS {
            return None;
        }
        let wr = ar + n as f64;
        let lw = wr.hypot(ai).log2();
        let lre = wr.log2();
        let arg = si.abs() * ai.abs() / wr * LOG2E * (1.0 + 1e-9);
        let lead = lead.unwrap_or_else(|| {
            let lead_int = (1.0 - sr) * lw + si * ai.atan2(wr) * LOG2E - l_s1;
            let a0 = ar + min_n as f64;
            let lead_first = -sr * a0.hypot(ai).log2() + si * ai.atan2(a0) * LOG2E;
            lead_int.max(lead_first)
        });
        let target = lead.clamp(-1e6, 1e6) - wp as f64 - 8.0;
        let mut lpoch = 0.0;
        let mut least_rem = f64::INFINITY;
        for j in 1..=jmax {
            lpoch += (sr + (2 * j - 2) as f64).hypot(si).max(1e-300).log2()
                + (sr + (2 * j - 1) as f64).hypot(si).max(1e-300).log2();
            let p = sr + 2.0 * j as f64;
            if let Some(rem) = em_remainder(j, lpoch, p, lre, lw, arg) {
                if rem < target {
                    return Some(n);
                }
                // Past the smallest remainder: this N does not do.
                if rem > least_rem + 64.0 {
                    break;
                }
                least_rem = least_rem.min(rem);
            }
        }
        n = (n + n / 4).max(n + 1);
    }
}

/// For a real `a = ar > 0` (`ai = 0`) and `σ = sr > 2`, the number `n ≤ 16`
/// of direct terms whose tail `Σ_{k≥n} abs((a + k)^(−s)) ≤ (a + n)^(−σ) + (a +
/// n)^(1−σ)/(σ − 1)` (the moduli `(a + k)^(−σ)` decrease) is below `2^−wp` of the
/// first term, when there is one: no Euler–Maclaurin tail is needed (the
/// values `ζ(1 − s + k)` of the expansion in `ln z` in `polylog.rs`).
pub(super) fn direct_terms(sr: f64, ar: f64, ai: f64, wp: usize) -> Option<usize> {
    direct_terms_upto(sr, ar, ai, wp, 16)
}

/// [`direct_terms`] with at most `max` terms.
pub(super) fn direct_terms_upto(sr: f64, ar: f64, ai: f64, wp: usize, max: usize) -> Option<usize> {
    if ai != 0.0 || ar <= 0.0 || sr <= 2.0 || !sr.is_finite() {
        return None;
    }
    let target = -sr * ar.log2() - wp as f64 - 8.0;
    (1..=max).find(|&n| {
        let x = (ar + n as f64).log2();
        lsum(-sr * x, (1.0 - sr) * x - (sr - 1.0).log2()) < target
    })
}

/// The most Euler–Maclaurin corrections taken at `wp` bits for an order of
/// modulus `abs_s`.
fn max_corrections(wp: usize, abs_s: f64) -> usize {
    wp / 2 + 16 + (abs_s.min(1e6) / 2.0) as usize
}

/// `ζ(s, a)` at working precision `wp` (see the module documentation):
/// `s ≠ 1`, and `Re a > 0` or `s` an integer with `a` off the poles.
pub(super) fn hurwitz(s: &Complex, a: &Complex, w: &mut Wp<'_>) -> Result<Sum, SymplexError> {
    let wp = w.bits();
    let (Some(sr), Some(si), Some(ar), Some(ai)) =
        (to_f64(&s.0), to_f64(&s.1), to_f64(&a.0), to_f64(&a.1))
    else {
        return Err(super::unevaluable("Hurwitz zeta argument beyond f64"));
    };
    if s.1.is_zero() && s.0 == BigFloat::from_i32(1, 64) {
        return Err(super::unevaluable("Hurwitz zeta: pole at order 1"));
    }
    let int_order = positive_integer(s);
    let right = super::bf_strictly_positive(&a.0);
    if !right && int_order.is_none() {
        return Err(super::unevaluable(
            "Hurwitz zeta of a non-integer order needs Re a > 0",
        ));
    }
    if a.1.is_zero() && !right && (a.0.is_zero() || a.0.is_int()) {
        return Err(super::unevaluable("Hurwitz zeta at a pole of its argument"));
    }
    // Direct terms up to Re(a + N) ≥ 1 at least (saturating: a huge `−Re a`
    // casts to `usize::MAX`, refused below).
    let min_n = if ar >= 1.0 {
        0
    } else {
        ((1.0 - ar).ceil() as usize).saturating_add(1)
    };
    if min_n > MAX_TERMS {
        return Err(super::unevaluable(
            "Hurwitz zeta: the argument is too far left of 0",
        ));
    }
    let exact = super::exact_bits(&a.0, wp);
    if let Some(n) = direct_terms(sr, ar, ai, wp) {
        // A large order: the direct sum alone.
        let mut sum = Sum::new(wp);
        for k in 0..n {
            let ak = (
                a.0.add(&BigFloat::from_u64(k as u64, 64), exact, RM),
                a.1.clone(),
            );
            let t = match int_order {
                Some(m) => pow_neg_int(&ak, m, wp),
                None => pow_neg(&ak, s, w),
            };
            sum.add(&t, wp);
        }
        return Ok(sum);
    }
    let l_s1 = lg_shifted(s, -1, wp);
    let mut n = choose_terms(sr, si, ar, ai, min_n, None, l_s1, wp).ok_or_else(|| {
        super::unevaluable(
            "Hurwitz zeta: an order or argument this large is not supported in evalf",
        )
    })?;
    // The simulation only chooses N: if the corrections do not reach the
    // bound there, more direct terms.
    let mut tries = 0;
    loop {
        match hurwitz_at(s, a, int_order, sr, si, ai, n, w) {
            Err(SymplexError::ComputationFailed { .. }) if tries < 3 && 2 * n <= MAX_TERMS => {
                tries += 1;
                n *= 2;
            }
            other => return other,
        }
    }
}

/// The direct terms below `N = n` and the Euler–Maclaurin tail of
/// [`hurwitz`].
#[allow(clippy::too_many_arguments)]
fn hurwitz_at(
    s: &Complex,
    a: &Complex,
    int_order: Option<u64>,
    sr: f64,
    si: f64,
    ai: f64,
    n: usize,
    wpc: &mut Wp<'_>,
) -> Result<Sum, SymplexError> {
    let wp = wpc.bits();
    // The truncation is measured against `2^−target`; the arithmetic carries
    // `log₂` of the number of terms more, for their roundings.
    let target = wp;
    let wp = wp + accuracy::ceil_log2(n + max_corrections(wp, sr.hypot(si))) as usize + 4;
    let exact = super::exact_bits(&a.0, wp);
    let one_a = a.1.is_zero() && a.0 == BigFloat::from_i32(1, 64);
    let mut sum = Sum::new(wp);
    if one_a {
        // k^(−s), k = 1..n: powers at the primes only.
        let spf = smallest_prime_factors(n);
        let mut powers: Vec<Complex> = Vec::with_capacity(n);
        for k in 1..=n {
            let p = if k == 1 {
                c_one(wp)
            } else if spf[k] as usize == k {
                let b = (BigFloat::from_u64(k as u64, 64), BigFloat::new(64));
                match int_order {
                    Some(m) => pow_neg_int(&b, m, wp),
                    None => pow_neg(&b, s, &mut wpc.at(wp)),
                }
            } else {
                let q = spf[k] as usize;
                c_mul(&powers[q - 1], &powers[k / q - 1], wp, RM)
            };
            sum.add(&p, wp);
            powers.push(p);
        }
    } else {
        for k in 0..n {
            let ak = (
                a.0.add(&BigFloat::from_u64(k as u64, 64), exact, RM),
                a.1.clone(),
            );
            let t = match int_order {
                Some(m) => pow_neg_int(&ak, m, wp),
                None => pow_neg(&ak, s, &mut wpc.at(wp)),
            };
            sum.add(&t, wp);
        }
    }
    let w = (
        a.0.add(&BigFloat::from_u64(n as u64, 64), exact, RM),
        a.1.clone(),
    );
    let base = match int_order {
        Some(m) => pow_neg_int(&w, m, wp),
        None => pow_neg(&w, s, &mut wpc.at(wp)),
    };
    let s1 = (plus_int(&s.0, -1, wp), s.1.clone());
    sum.add(&c_div(&c_mul(&base, &w, wp, RM), &s1, wp, RM), wp);
    sum.add(&div_int(&base, 2, wp), wp);
    em_corrections(s, sr, si, ai, &w, &base, &mut sum, target, wp)?;
    Ok(sum)
}

/// The corrections `Σ_{j≥1} B₂ⱼ/(2j)!·(s)₂ⱼ₋₁·w^(−s)·w^(1−2j)` added to `sum`
/// (at `wp` bits) until the remainder bound is below `2^−target` of its
/// largest term.
#[allow(clippy::too_many_arguments)]
fn em_corrections(
    s: &Complex,
    sr: f64,
    si: f64,
    ai: f64,
    w: &Complex,
    base: &Complex,
    sum: &mut Sum,
    target: usize,
    wp: usize,
) -> Result<(), SymplexError> {
    let inv_w = c_div(&c_one(wp), w, wp, RM);
    let inv_w2 = c_mul(&inv_w, &inv_w, wp, RM);
    let lw = accuracy::lg_abs_low(w);
    let lre = accuracy::part_lg_low(&w.0);
    let wr = to_f64(&w.0).unwrap_or(f64::INFINITY);
    let arg = si.abs() * ai.abs() / wr * LOG2E * (1.0 + 1e-9);
    let mut pw = inv_w; // w^(−(2j−1))
    // c_j = (s)_{2j−1}/(2j)!
    let mut c = div_int(s, 2, wp);
    let mut lpoch = 0.0f64; // log₂ abs((s)_{2j})
    let max_m = 2 * max_corrections(target, sr.hypot(si)) + 16;
    for j in 1..=max_m {
        if j > 1 {
            let f1 = (plus_int(&s.0, 2 * j as i64 - 3, wp), s.1.clone());
            let f2 = (plus_int(&s.0, 2 * j as i64 - 2, wp), s.1.clone());
            c = div_int(
                &c_mul(&c_mul(&c, &f1, wp, RM), &f2, wp, RM),
                ((2 * j - 1) * (2 * j)) as u64,
                wp,
            );
            pw = c_mul(&pw, &inv_w2, wp, RM);
        }
        let b = bernoulli(j, wp);
        let t = scale(&c_mul(&c_mul(base, &pw, wp, RM), &c, wp, RM), &b, wp);
        sum.add(&t, wp);
        lpoch += lg_shifted(s, 2 * j as i64 - 2, wp) + lg_shifted(s, 2 * j as i64 - 1, wp);
        if lpoch == f64::NEG_INFINITY {
            // (s)_{2j} = 0: the sum is exact (an integer s ≤ 0).
            return Ok(());
        }
        let p = sr + 2.0 * j as f64;
        if let Some(rem) = em_remainder(j, lpoch, p, lre, lw, arg)
            && rem < sum.largest - target as f64 - 4.0
        {
            return Ok(());
        }
    }
    Err(super::series_did_not_converge("Hurwitz zeta", max_m))
}

/// `ψ(z)` for `z` off the poles at working precision `wp`: `ψ(w) − Σ_{k<N}
/// 1/(z + k)`, `ψ(w)` by Euler–Maclaurin (see the module documentation).
fn digamma_em(z: &Complex, wpc: &mut Wp<'_>) -> Result<Sum, SymplexError> {
    let wp = wpc.bits();
    let (Some(zr), Some(zi)) = (to_f64(&z.0), to_f64(&z.1)) else {
        return Err(super::unevaluable("digamma argument beyond f64"));
    };
    let min_n = if zr >= 1.0 {
        0
    } else {
        ((1.0 - zr).ceil() as usize).saturating_add(1)
    };
    if min_n > MAX_TERMS {
        return Err(super::unevaluable(
            "digamma: the argument is too far left of 0",
        ));
    }
    // The leading term is ln w, of modulus at least ln 2 (Re w ≥ 2).
    let n = choose_terms(1.0, 0.0, zr, zi, min_n, Some(-1.0), 0.0, wp)
        .ok_or_else(|| super::unevaluable("digamma: argument not supported in evalf"))?;
    let target = wp;
    let wp = wp + accuracy::ceil_log2(n + max_corrections(wp, 1.0)) as usize + 4;
    let exact = super::exact_bits(&z.0, wp);
    let mut sum = Sum::new(wp);
    for k in 0..n {
        let zk = (
            z.0.add(&BigFloat::from_u64(k as u64, 64), exact, RM),
            z.1.clone(),
        );
        sum.add(&c_neg(&c_div(&c_one(wp), &zk, wp, RM)), wp);
    }
    let w = (
        z.0.add(&BigFloat::from_u64(n as u64, 64), exact, RM),
        z.1.clone(),
    );
    sum.add(&super::c_ln(&w, &mut wpc.at(wp)), wp);
    let inv_w = c_div(&c_one(wp), &w, wp, RM);
    sum.add(&c_neg(&div_int(&inv_w, 2, wp)), wp);
    let inv_w2 = c_mul(&inv_w, &inv_w, wp, RM);
    let lw = accuracy::lg_abs_low(&w);
    let lre = accuracy::part_lg_low(&w.0);
    let mut pw = inv_w2.clone(); // w^(−2j)
    let mut lpoch = 0.0f64; // log₂ (2j)!
    let max_m = 2 * max_corrections(target, 1.0) + 16;
    for j in 1..=max_m {
        if j > 1 {
            pw = c_mul(&pw, &inv_w2, wp, RM);
        }
        let b = bernoulli(j, wp);
        let coeff = b.div(&BigFloat::from_u64(2 * j as u64, 64), wp, RM);
        sum.add(&c_neg(&scale(&pw, &coeff, wp)), wp);
        lpoch += ((2 * j - 1) as f64).log2() + ((2 * j) as f64).log2();
        if let Some(rem) = em_remainder(j, lpoch, 1.0 + 2.0 * j as f64, lre, lw, 0.0)
            && rem < sum.largest - target as f64 - 4.0
        {
            return Ok(sum);
        }
    }
    Err(super::series_did_not_converge("digamma", max_m))
}

// ── Values with the loss made up ────────────────────────────────────────────

/// Evaluate `f` at working precisions `prec + extra`, the loss to
/// cancellation (largest term over the value) measured and made up up to
/// `cancellation_cap`; the value rounded to `prec + 16` bits, its error below
/// `2^−(prec+12)` of its modulus.
pub(super) fn with_loss(
    w: &mut Wp<'_>,
    f: &mut dyn FnMut(usize, &mut Consts) -> Result<Sum, SymplexError>,
) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    let cap = super::cancellation_cap(prec);
    let mut extra = 32 + accuracy::ceil_log2(prec) as usize;
    loop {
        let wp = prec + extra;
        let sum = f(wp, w.cc).map_err(super::requested_at(prec))?;
        let lv = accuracy::lg_abs(&sum.value);
        let lost = if lv.is_finite() && sum.largest.is_finite() {
            ((sum.largest - lv).max(0.0).ceil() as usize).min(cap + 1)
        } else if lv.is_finite() {
            0
        } else {
            wp
        };
        if lost + 24 <= extra {
            return Ok(round_c(sum.value, prec + 16));
        }
        if extra >= cap || lost + 24 > cap {
            return Err(super::special_exhausted(prec));
        }
        extra = (lost + 40).max(2 * extra).min(cap);
    }
}

/// `ζ(s)` for a non-real `s` (see the module documentation), rounded to
/// `prec + 16` bits, its error below `2^−(prec+12)·abs(ζ(s))`.
pub(super) fn zeta_complex(s: &Complex, w: &mut Wp<'_>) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    if [&s.0, &s.1].iter().any(|x| x.is_nan() || x.is_inf()) {
        return Err(super::unevaluable("zeta of special float value"));
    }
    tracing::debug!(prec, "evalf: zeta of a complex argument");
    with_loss(w, &mut |wp, cc| zeta_at(s, &mut Wp::new(wp, cc)))
}

/// `ζ(s)` at working precision `wp`: Euler–Maclaurin for `σ ≥ 0`, the
/// functional equation left of 0.
fn zeta_at(s: &Complex, wpc: &mut Wp<'_>) -> Result<Sum, SymplexError> {
    let wp = wpc.bits();
    let one = (BigFloat::from_i32(1, 64), BigFloat::new(64));
    // Far up the critical strip the Riemann–Siegel formula (`rszeta.rs`;
    // mpmath's criterion `abs(t) > 500·wp`, `10·abs(σ) < wp`, and from `abs(t)
    // = 10⁵` on, where Euler–Maclaurin needs `abs(t)/(2π)` terms): its
    // absolute error is below `2^−wp`, the scale of the value 1.
    if let (Some(sig), Some(t)) = (to_f64(&s.0), to_f64(&s.1)) {
        let (t, w) = (t.abs(), wp as f64);
        if (t > 500.0 * w || t >= 1.0e5)
            && 10.0 * sig.abs() < w.max(t.sqrt())
            && let Some(v) = super::rszeta::zeta_rs(s, &mut wpc.at(wp + 4))?
        {
            return Ok(Sum {
                value: v,
                largest: 0.0,
            });
        }
    }
    if !s.0.is_negative() || s.0.is_zero() {
        return hurwitz(s, &one, wpc);
    }
    // ζ(s) = 2^s π^(s−1) sin(πs/2) Γ(1 − s) ζ(1 − s), `1 − s` exact.
    let t = (plus_int(&s.0.neg(), 1, wp), s.1.neg());
    let z1 = hurwitz(&t, &one, wpc)?;
    let p = prefactor(s, &t, wpc)?;
    let value = c_mul(&p, &z1.value, wp, RM);
    let largest = accuracy::lg_abs(&p) + z1.largest;
    Ok(Sum { value, largest })
}

/// `2^s π^(s−1) sin(πs/2) Γ(t)` for `t = 1 − s` (non-real), to `wp` bits
/// relative.
fn prefactor(s: &Complex, t: &Complex, w: &mut Wp<'_>) -> Result<Complex, SymplexError> {
    let wp = w.bits();
    let abs_t = accuracy::lg_abs(t).exp2();
    // |ln Γ(t)| ≲ |t|(ln|t| + π + 1) + 4; the exponent's absolute error is
    // the result's relative one.
    let size = (abs_t + 1.0) * ((abs_t + 1.0).ln() + 5.0) + 8.0;
    let p = wp + super::magnitude_bits(size) + 16;
    let lg = super::loggamma::loggamma_complex(t, &mut w.at(p))?;
    let ln2 = BigFloat::from_i32(2, 64).ln(p, RM, w.cc);
    let lnpi = w.cc.pi(p, RM).clone().ln(p, RM, w.cc);
    let s1 = (plus_int(&s.0, -1, p), s.1.clone());
    let e = c_add(
        &c_add(&scale(s, &ln2, p), &scale(&s1, &lnpi, p), p, RM),
        &lg,
        p,
        RM,
    );
    let ex = super::c_exp(&e, &mut w.at(p));
    let sn = sin_half_pi(s, &mut w.at(wp + 8));
    Ok(round_c(c_mul(&ex, &sn, wp + 8, RM), wp))
}

/// `sin(πs/2)` for a complex `s`, with `σ/2 = n + d`, `n` the nearest integer
/// and `d` exact: `(−1)ⁿ·sin(πd + iπτ/2)`.
pub(super) fn sin_half_pi(s: &Complex, w: &mut Wp<'_>) -> Complex {
    let wp = w.bits();
    let exact = super::exact_bits(&s.0, wp);
    let half = s.0.div(&BigFloat::from_i32(2, 64), exact, RM);
    let n = half.add(&BigFloat::from_f64(0.5, 64), exact, RM).floor();
    let d = half.sub(&n, exact, RM);
    let pi = w.cc.pi(wp + 8, RM).clone();
    let y =
        s.1.mul(&pi, wp + 8, RM)
            .div(&BigFloat::from_i32(2, 64), wp + 8, RM);
    let v = super::c_sin(&(pi.mul(&d, wp + 8, RM), y), &mut w.at(wp + 8));
    let odd = !n
        .div(&BigFloat::from_i32(2, 64), exact, RM)
        .fract()
        .is_zero();
    round_c(if odd { c_neg(&v) } else { v }, wp)
}

/// `η(s) = (1 − 2^(1−s))·ζ(s)` for a non-real `s`, rounded to `prec + 16`
/// bits, its error below `2^−(prec+12)·abs(η(s))`.
pub(super) fn eta_complex(s: &Complex, w: &mut Wp<'_>) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    let z = zeta_complex(s, &mut w.at(prec + 8))?;
    let f = one_minus_pow2(s, &mut w.at(prec + 24));
    Ok(round_c(c_mul(&f, &z, prec + 24, RM), prec + 16))
}

/// `1 − 2^(1−s) = −expm1((1 − s)·ln 2)` to `wp` bits relative: `e^u` with as
/// many guard bits as `u` is small (`1 − s` exact).
fn one_minus_pow2(s: &Complex, w: &mut Wp<'_>) -> Complex {
    let wp = w.bits();
    let t = (plus_int(&s.0.neg(), 1, wp), s.1.neg());
    let small = (-accuracy::lg_abs(&t)).clamp(0.0, 1e6) as usize;
    let p = wp + small + super::magnitude_bits(accuracy::lg_abs(&t).exp2()) + 16;
    let ln2 = BigFloat::from_i32(2, 64).ln(p, RM, w.cc);
    let u = scale(&t, &ln2, p);
    let e = super::c_exp(&u, &mut w.at(p));
    round_c(c_sub(&c_one(p), &e, p, RM), wp)
}

/// `ψ⁽ᵐ⁾(z)` for an integer `m ≥ 0` and a non-real `z`, rounded to `prec +
/// 16` bits, its error below `2^−(prec+12)·abs(ψ⁽ᵐ⁾(z))`.
pub(super) fn polygamma_complex(
    m: u32,
    z: &Complex,
    w: &mut Wp<'_>,
) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    if [&z.0, &z.1].iter().any(|x| x.is_nan() || x.is_inf()) {
        return Err(super::unevaluable("polygamma of special float value"));
    }
    if m > MAX_POLYGAMMA_ORDER {
        return Err(super::unevaluable(
            "polygamma of an order beyond 10^4 at a complex argument not supported in evalf",
        ));
    }
    tracing::debug!(prec, m, "evalf: polygamma of a complex argument");
    if m == 0 {
        return with_loss(w, &mut |wp, cc| digamma_em(z, &mut Wp::new(wp, cc)));
    }
    let s = (BigFloat::from_u64(u64::from(m) + 1, 64), BigFloat::new(64));
    let h = with_loss(&mut w.at(prec + 8), &mut |wp, cc| {
        hurwitz(&s, z, &mut Wp::new(wp, cc))
    })?;
    // (−1)^(m+1)·m!, with log₂ m guard bits for the roundings of the product.
    let p = prec + 24 + accuracy::ceil_log2(m as usize) as usize;
    let mut f = BigFloat::from_i32(1, p);
    for k in 2..=u64::from(m) {
        f = f.mul(&BigFloat::from_u64(k, 64), p, RM);
    }
    if m.is_multiple_of(2) {
        f = f.neg();
    }
    Ok(round_c(scale(&h, &f, p), prec + 16))
}

/// `Γ(z)` for a non-real `z`: `exp(ln Γ(z))`, rounded to `prec + 16` bits, its
/// error below `2^−(prec+12)·abs(Γ(z))`.
pub(super) fn gamma_complex(z: &Complex, w: &mut Wp<'_>) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    if [&z.0, &z.1].iter().any(|x| x.is_nan() || x.is_inf()) {
        return Err(super::unevaluable("Gamma of special float value"));
    }
    // The magnitude of ln Γ first (its absolute error is Γ's relative one).
    let rough = super::loggamma::loggamma_complex(z, &mut w.at(64))?;
    // |Γ(z)| = e^{Re ln Γ}: a value beyond the exponent range is refused
    // here, before `exp` is asked for it at a working precision that holds
    // all the bits of `Im ln Γ` (that took minutes: `Γ` of a factorial with a
    // 10⁵-digit modulus, found by `fuzz_parser`).
    let limit = f64::from(astro_float::EXPONENT_MAX) * std::f64::consts::LN_2;
    if to_f64(&rough.0).is_none_or(|re_lg| re_lg.abs() > limit) {
        return Err(super::unevaluable(
            "Gamma of a complex argument beyond the exponent range",
        ));
    }
    let size = accuracy::lg_abs(&rough).exp2() + 2.0;
    if !size.is_finite() || size > limit {
        return Err(super::unevaluable(
            "Gamma of a complex argument beyond the exponent range",
        ));
    }
    let p = prec + 32 + super::magnitude_bits(size);
    let lg = super::loggamma::loggamma_complex(z, &mut w.at(p))?;
    let v = super::c_exp(&lg, &mut w.at(p));
    if [&v.0, &v.1].iter().any(|x| x.is_nan() || x.is_inf()) {
        return Err(super::unevaluable(
            "Gamma of a complex argument beyond the exponent range",
        ));
    }
    Ok(round_c(v, prec + 16))
}

/// `harmonic(z) = ψ(z + 1) + γ` for a non-real `z`, rounded to `prec + 16`
/// bits, its error below `2^−(prec+12)` of its modulus (relative to the larger
/// of `abs(ψ(z + 1))` and `γ`).
pub(super) fn harmonic_complex(z: &Complex, w: &mut Wp<'_>) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    let z1 = (plus_int(&z.0, 1, prec), z.1.clone());
    // The imaginary part is never 0: no cancellation can swallow it, but the
    // real part may cancel against γ — the bits the sum loses are added.
    let mut p = prec + 24;
    loop {
        let psi = polygamma_complex(0, &z1, &mut w.at(p))?;
        let g = super::arb_euler_gamma(&mut w.at(p + 16))?;
        let v = (psi.0.add(&g, p + 16, RM), psi.1.clone());
        let lost = (accuracy::lg_abs(&psi).max(0.0) - accuracy::lg_abs(&v)).max(0.0);
        if (lost as usize) + prec + 20 <= p || p > prec + super::cancellation_cap(prec) {
            return Ok(round_c(v, prec + 16));
        }
        p = prec + 40 + lost.ceil() as usize;
    }
}

/// `B(a, b) = Γ(a)Γ(b)/Γ(a + b)` with an argument not real (`Γ` of a non-real
/// argument by [`gamma_complex`], of a real one by `arb_gamma_real`), rounded
/// to `prec + 16` bits, its error below `2^−(prec+12)·abs(B)`.
pub(super) fn beta_complex(
    a: &Complex,
    b: &Complex,
    w: &mut Wp<'_>,
) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    let p = prec + 24;
    let gamma = |z: &Complex, cc: &mut Consts| -> Result<Complex, SymplexError> {
        if z.1.is_zero() {
            Ok((
                super::arb_gamma_real(&z.0, &mut Wp::new(p, cc))?,
                BigFloat::new(p),
            ))
        } else {
            gamma_complex(z, &mut Wp::new(p, cc))
        }
    };
    let exact = super::exact_bits(&a.0, p).max(super::exact_bits(&b.0, p));
    let ab = (a.0.add(&b.0, exact, RM), a.1.add(&b.1, p + 64, RM));
    let ga = gamma(a, w.cc)?;
    let gb = gamma(b, w.cc)?;
    let gab = gamma(&ab, w.cc)?;
    if gab.0.is_zero() && gab.1.is_zero() {
        return Err(super::unevaluable("Beta function: Gamma(a+b) is zero"));
    }
    let v = c_div(&c_mul(&ga, &gb, p, RM), &gab, p, RM);
    Ok(round_c(v, prec + 16))
}

// ── Error bounds ────────────────────────────────────────────────────────────

/// The error bound of `B(a, b)` (value `v`) for `a ± ba`, `b ± bb`, an argument
/// not real: `abs(∂B/∂a) = abs(B)·abs(ψ(a) − ψ(a + b)) ≤ abs(B)·(Ψ(a) + Ψ(a + b))`
/// (and for `b`), `Ψ` the bound on `abs(ψ)` over the ball (`loggamma.rs`), the
/// first-order change doubled.
pub(super) fn beta_bound(
    a: &Complex,
    ba: Bound,
    b: &Complex,
    bb: Bound,
    v: &Complex,
    prec: usize,
) -> Bound {
    let (Some((ax, ay, ra)), Some((bx, by, rb))) = (ball(a, ba), ball(b, bb)) else {
        return Bound::UNKNOWN;
    };
    let lv = accuracy::lg_abs(v);
    if !lv.is_finite() {
        return Bound::UNKNOWN;
    }
    let own = lv - prec as f64 - 12.0;
    if ra == 0.0 && rb == 0.0 {
        return Bound::both(own);
    }
    let r = ra + rb;
    let psi = |x: f64, y: f64, rad: f64| super::loggamma::log2_psi_bound(x, y, rad.max(1e-300));
    let (Some(pa), Some(pb), Some(pab)) =
        (psi(ax, ay, ra), psi(bx, by, rb), psi(ax + bx, ay + by, r))
    else {
        return Bound::UNKNOWN;
    };
    let da = lsum(pa, pab);
    let db = lsum(pb, pab);
    let mut d = f64::NEG_INFINITY;
    if ra > 0.0 {
        d = lsum(d, da + log2_radius(ba));
    }
    if rb > 0.0 {
        d = lsum(d, db + log2_radius(bb));
    }
    // |B(u) − B(z)| ≤ |B|·(e^t − 1), t the bound on |Δ ln B|.
    if !(d.is_finite() && d < 0.0) {
        return Bound::UNKNOWN;
    }
    Bound::both(accuracy::lsum(own, lv + log2_expm1_exp2(d) + 1.0 + 1e-9))
}

/// `(x, y, r)`: the argument's parts in `f64` and its radius (`0` when exact),
/// `None` when the bound is unknown or beyond `f64`.  A radius below the
/// `f64` range is taken as the smallest positive `f64` for the geometric
/// tests (a negligible overestimate there); the error itself is formed from
/// the exponent ([`log2_radius`]).  Before 0.45 it underflowed to 0, which
/// stands for an exact argument, and past 1,074 bits the argument's error
/// was dropped (`ζ(√2/2 + 200000i)` was "certified" at 1,162 bits without a
/// bound on `ζ′`).
fn ball(z: &Complex, bz: Bound) -> Option<(f64, f64, f64)> {
    if bz.is_unknown() {
        return None;
    }
    let r = if bz.is_exact() {
        0.0
    } else {
        bz.joint().exp2().max(f64::MIN_POSITIVE)
    };
    Some((to_f64(&z.0)?, to_f64(&z.1)?, r))
}

/// `log₂` of the radius of a known bound (`−∞` when exact).
fn log2_radius(bz: Bound) -> f64 {
    if bz.is_exact() {
        f64::NEG_INFINITY
    } else {
        bz.joint()
    }
}

/// The node's own error `2^−(prec+12)·abs(v)` plus twice the radius `2^lr`
/// times the derivative bound `2^d`.
fn finish(v: &Complex, prec: usize, lr: f64, d: Option<f64>) -> Bound {
    let lv = accuracy::lg_abs(v);
    if !lv.is_finite() {
        return Bound::UNKNOWN;
    }
    let own = lv - prec as f64 - 12.0;
    if lr == f64::NEG_INFINITY {
        return Bound::both(own);
    }
    match d {
        Some(d) if d.is_finite() => Bound::both(accuracy::lsum(own, d + lr + 1.0)),
        _ => Bound::UNKNOWN,
    }
}

/// `log₂(e^(2^l) − 1)` for `l < 0`, also far below the `f64` range of `2^l`.
fn log2_expm1_exp2(l: f64) -> f64 {
    if l < -30.0 {
        // e^x − 1 = x(1 + x/2 + …) < x·e^x, x < 2^−30.
        l + 1e-9
    } else {
        l.exp2().exp_m1().log2()
    }
}

/// The error bound of `ζ(s)` (value `v`) for a non-real `s ± bs`.
pub(super) fn zeta_bound(s: &Complex, bs: Bound, v: &Complex, prec: usize) -> Bound {
    let Some((x, y, r)) = ball(s, bs) else {
        return Bound::UNKNOWN;
    };
    let d = if r == 0.0 {
        None
    } else {
        zeta_ball(x, y, r).map(|(_, d)| d)
    };
    finish(v, prec, log2_radius(bs), d)
}

/// The error bound of `η(s)` (value `v`) for a non-real `s ± bs`:
/// `abs(η′) ≤ ln 2·2^(1−σ′)·abs(ζ) + (1 + 2^(1−σ′))·abs(ζ′)` over the ball.
pub(super) fn eta_bound(s: &Complex, bs: Bound, v: &Complex, prec: usize) -> Bound {
    let Some((x, y, r)) = ball(s, bs) else {
        return Bound::UNKNOWN;
    };
    let d = if r == 0.0 {
        None
    } else {
        zeta_ball(x, y, r).map(|(lz, ld)| {
            let p2 = 1.0 - (x - r); // log₂ 2^(1−σ′)
            lsum(LN2.log2() + p2 + lz, lsum(0.0, p2) + ld)
        })
    };
    finish(v, prec, log2_radius(bs), d)
}

/// The error bound of `ψ⁽ᵐ⁾(z)` (value `v`; `ψ` for `m = 0`) for a non-real
/// `z ± bz`.
pub(super) fn polygamma_bound(m: u32, z: &Complex, bz: Bound, v: &Complex, prec: usize) -> Bound {
    let Some((x, y, r)) = ball(z, bz) else {
        return Bound::UNKNOWN;
    };
    let d = if r == 0.0 {
        None
    } else {
        let p = f64::from(m) + 2.0;
        pole_sum(x, y, r, p).map(|s| s + log2_factorial(u64::from(m) + 1))
    };
    finish(v, prec, log2_radius(bz), d)
}

/// The error bound of `Γ(z)` (value `v`) for a non-real `z ± bz`:
/// `abs(Γ(u) − Γ(z)) ≤ abs(Γ(z))·(e^(Ψr) − 1)` with `Ψ` the bound on `abs(ψ)`
/// over the ball (`loggamma.rs`), doubled.
pub(super) fn gamma_bound(z: &Complex, bz: Bound, v: &Complex, prec: usize) -> Bound {
    let Some((x, y, r)) = ball(z, bz) else {
        return Bound::UNKNOWN;
    };
    let lv = accuracy::lg_abs(v);
    if !lv.is_finite() {
        return Bound::UNKNOWN;
    }
    let own = lv - prec as f64 - 12.0;
    if r == 0.0 {
        return Bound::both(own);
    }
    let Some(lpsi) = super::loggamma::log2_psi_bound(x, y, r) else {
        return Bound::UNKNOWN;
    };
    let lpr = lpsi + log2_radius(bz);
    if !(lpr.is_finite() && lpr < 0.0) {
        return Bound::UNKNOWN;
    }
    let change = lv + log2_expm1_exp2(lpr) + 1e-9;
    Bound::both(accuracy::lsum(own, change + 1.0))
}

/// `log₂ n!`.
pub(super) fn log2_factorial(n: u64) -> f64 {
    if n <= 64 {
        return (2..=n).map(|k| (k as f64).log2()).sum();
    }
    let n = n as f64;
    ((n + 0.5) * n.ln() - n + 0.5 * (2.0 * std::f64::consts::PI).ln() + 1.0 / (12.0 * n)) * LOG2E
        + 1e-9
}

/// `log₂` of an upper bound on `Σ_{k≥0} abs(u + k)^(−p)` over `abs(u − z) ≤ r`,
/// `z = x + iy`: the terms to `Re(z + K) ≥ 1` one by one, `(abs(z + k) − r)^(−p)`,
/// and beyond `g(K) + (x + K − r)^(1−p)/(p − 1)` (the terms decrease there).
/// `None` when the ball comes within eight radii of a pole `−k`.
fn pole_sum(x: f64, y: f64, r: f64, p: f64) -> Option<f64> {
    if !(r.is_finite() && r <= 0.0625) {
        return None;
    }
    let k_direct = if x >= 1.0 + r {
        0
    } else {
        (1.0 + r - x).ceil() as usize
    };
    if k_direct > MAX_TERMS {
        return None;
    }
    let mut acc = f64::NEG_INFINITY;
    for k in 0..k_direct {
        let d = (x + k as f64).hypot(y);
        if d <= 8.0 * r {
            return None;
        }
        acc = lsum(acc, -p * (d - r).log2());
    }
    let xk = x + k_direct as f64 - r; // ≥ 1
    let g = -p * ((x + k_direct as f64).hypot(y) - r).log2();
    let tail = lsum(g, (1.0 - p) * xk.log2() - (p - 1.0).log2());
    Some(lsum(acc, tail) + 1e-9)
}

/// `log₂` of upper bounds on `abs(ζ(u))` and `abs(ζ′(u))` over `abs(u − s) ≤ r`, `s
/// = x + iy`, from the Euler–Maclaurin formula at `N = ⌈abs(s) + r⌉ + 2`
/// differentiated term by term (see the module documentation).  `None` when
/// the ball comes within eight radii of the pole at 1.  Beyond
/// [`DIRECT_BALL_TERMS`] the partial sums are bounded in closed form (since
/// 0.45; before, `N` beyond [`MAX_TERMS`] had no bound): `Σ_{1≤k<N} k^(−σ′) ≤
/// [σ′ ≥ 0] + ∫_1^N x^(−σ′) dx` (monotone terms) and `Σ ln k·k^(−σ′) ≤ ln N·Σ
/// k^(−σ′)`.
fn zeta_ball(x: f64, y: f64, r: f64) -> Option<(f64, f64)> {
    let big_x = x.hypot(y) + r; // abs(u) ≤ X
    let sig = x - r; // Re u ≥ σ′
    let delta = (x - 1.0).hypot(y) - r; // abs(u − 1) ≥ δ
    if !(big_x.is_finite() && delta > 7.0 * r && r <= 0.0625) {
        return None;
    }
    let n = big_x.ceil() + 2.0;
    if n > 1.0e15 {
        return None;
    }
    let ln_n = n.ln();
    let lg_n = n.log2();
    let (mut lz, mut ld) = if n > DIRECT_BALL_TERMS as f64 {
        // log₂ ∫_1^N x^(−σ′) dx, then the sums.
        let e = 1.0 - sig;
        let li = if e.abs() < 1e-9 {
            ln_n.log2()
        } else if e > 0.0 {
            // (N^e − 1)/e
            e * lg_n + (-(-e * lg_n).exp2()).ln_1p() * LOG2E - e.log2()
        } else {
            // (1 − N^e)/(−e)
            (-(e * lg_n).exp2()).ln_1p() * LOG2E - (-e).log2()
        };
        let lz = if sig >= 0.0 { lsum(0.0, li) } else { li };
        (lz, lz + ln_n.log2())
    } else {
        // Σ_{k<N} k^(−σ′) and Σ ln k·k^(−σ′), scaled by the largest k^(−σ′).
        let shift = if sig < 0.0 {
            -sig * (n - 1.0).log2()
        } else {
            0.0
        };
        let (mut zs, mut ds) = (0.0f64, 0.0f64);
        for k in 1..n as usize {
            let kf = k as f64;
            let t = (-sig * kf.log2() - shift).exp2();
            zs += t;
            ds += t * kf.ln();
        }
        let lz = zs.log2() + shift;
        let ld = if ds > 0.0 {
            ds.log2() + shift
        } else {
            f64::NEG_INFINITY
        };
        (lz, ld)
    };
    // w^(1−u)/(u − 1) and its derivative.
    let l1 = (1.0 - sig) * lg_n;
    lz = lsum(lz, l1 - delta.log2());
    ld = lsum(ld, l1 + (ln_n / delta + 1.0 / (delta * delta)).log2());
    // w^(−u)/2.
    lz = lsum(lz, -sig * lg_n - 1.0);
    ld = lsum(ld, -sig * lg_n - 1.0 + ln_n.log2());
    // The corrections and the remainder: P_n(X) = X(X+1)⋯(X+n−1) and
    // Σ 1/(X + i) for P′/P.
    let m = ((2.0 - sig) / 2.0).ceil().max(1.0) as usize;
    let (mut lp, mut inv) = (0.0f64, 0.0f64);
    let mut deg = 0usize;
    let mut extend = |to: usize, lp: &mut f64, inv: &mut f64| {
        while deg < to {
            let f = big_x + deg as f64;
            *lp += f.log2();
            *inv += 1.0 / f;
            deg += 1;
        }
    };
    for j in 1..=m {
        extend(2 * j - 1, &mut lp, &mut inv);
        let e = 2.0 - 2.0 * j as f64 * LOG2_TWO_PI + (1.0 - sig - 2.0 * j as f64) * lg_n;
        lz = lsum(lz, e + lp);
        ld = lsum(ld, e + lp + (inv + ln_n).log2());
    }
    extend(2 * m, &mut lp, &mut inv);
    let p = sig + 2.0 * m as f64;
    let e = 2.0 - 2.0 * m as f64 * LOG2_TWO_PI + (1.0 - p) * lg_n - (p - 1.0).log2();
    lz = lsum(lz, e + lp);
    ld = lsum(ld, e + lp + (inv + ln_n + 1.0 / (p - 1.0)).log2());
    // f64 rounding of the sums.
    let margin = 1e-6;
    (lz.is_finite() && ld.is_finite()).then_some((lz + margin, ld + margin))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: &str, im: &str, p: usize) -> Complex {
        let mut cc = Consts::new().unwrap();
        (
            BigFloat::parse(re, astro_float::Radix::Dec, p, RM, &mut cc),
            BigFloat::parse(im, astro_float::Radix::Dec, p, RM, &mut cc),
        )
    }

    fn close(v: &Complex, re: f64, im: f64, tol: f64) -> bool {
        let (a, b) = (to_f64(&v.0).unwrap(), to_f64(&v.1).unwrap());
        (a - re).abs() <= tol * re.abs().max(1e-300) && (b - im).abs() <= tol * im.abs().max(1e-300)
    }

    /// mpmath 1.3 at 30 digits: `zeta(0.5 + 14j)` =
    /// `0.022241142609993589246213199204 − 0.103258123266450057902363095553i`;
    /// `zeta(-2.5 + 3j)` = `0.0687636790336464816284776659999 +
    /// 0.13398028393783442697393013743i` (the functional equation).
    #[test]
    fn zeta_complex_matches_mpmath() {
        let mut cc = Consts::new().unwrap();
        let s = c("0.5", "14", 256);
        let v = zeta_complex(&s, &mut Wp::new(128, &mut cc)).unwrap();
        assert!(
            close(
                &v,
                0.022_241_142_609_993_59,
                -0.103_258_123_266_450_06,
                1e-14
            ),
            "{v:?}"
        );
        let s = c("-2.5", "3", 256);
        let v = zeta_complex(&s, &mut Wp::new(128, &mut cc)).unwrap();
        assert!(
            close(
                &v,
                0.068_763_679_033_646_48,
                0.133_980_283_937_834_43,
                1e-14
            ),
            "{v:?}"
        );
    }

    #[test]
    fn zeta_ball_is_finite_on_the_critical_line() {
        assert!(zeta_ball(0.5, 14.0, 1e-30).is_some());
        assert!(zeta_ball(1.0, 1e-10, 1e-8).is_none());
    }
}

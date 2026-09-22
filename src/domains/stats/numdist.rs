//! Numeric (`f64`) reference distributions: the regularised incomplete beta
//! and gamma functions and, on top of them, the distribution function,
//! survival function and quantile of the normal, Student-t, χ², F, beta,
//! gamma, binomial and Poisson families — `scipy.stats.<dist>.{cdf, sf,
//! ppf, isf}` in plain floating point.
//!
//! This is the *numeric* kernel behind
//! [`Distribution::quantile_f64`](super::Distribution::quantile_f64) and the
//! critical values of the statistics-on-data modules.  The exact tails
//! (`chi_squared_sf`, `f_sf`, `Ols::p_values`, `PValue`) are untouched:
//! they build an [`Ex`](crate::api::expr::Ex) and stay the source of
//! truth; this module only replaces the routes that used to evaluate those
//! expressions in 128-bit arithmetic once per root-finding probe.
//!
//! # Algorithms
//!
//! | function | method | relative accuracy |
//! |---|---|---|
//! | [`betainc_regularized_f64`] | the dispatch of TOMS 708 (DiDonato & Morris 1992): power series `bpser`, the recurrence `bup`, the asymptotic expansion `bgrat` for a large first and small second parameter, the continued fraction `bfrac`, and the two-large-parameter expansion `basym` (both shapes above 100 and `x` within `3 %` of the mean, in `λ = a − (a + b)x`); each tail computed directly where it is the smaller one | ≈ 1e-15 typical, ≤ 1e-13 for `a + b ≤ 1e6` away from the far tail; in the far tail (`1e-100` and below) with shapes ≥ 1e4 expect ≈ 1e-11 — the conditioning of the `f64` argument, one ulp of which moves such a tail by that much; for `a, b` both ≈ 1e7 or more the conditioning of `(a + b)·x` in double limits it to ≈ 1e-12 |
//! | [`gammainc_lower_regularized_f64`], [`gammainc_upper_regularized_f64`] | a dispatch on `a`: below `GAMMA_TEMME_MIN_A = 1e6`, or for `x` more than 40 standard deviations from `a`, the power series for `x < a + 1` and Lentz's continued fraction otherwise, with the prefactor `xᵃe⁻ˣ/Γ(a)` through Loader's `bd0` and the Stirling remainder (no cancellation for large `a`); from `a = 1e6` on and within those 40σ, Temme's uniform asymptotic expansion (DLMF 8.12) truncated after `c₁/a`, its `½ erfc(z)` and correction term sharing one exponential so the tail stays correct into the subnormal range | ≈ 1e-15; ≈ 1e-12 at `a = 5e7` (the conditioning of the `f64` argument) |
//! | `norm` | Cody's `erfc`, continued by `erfcx(x)·e^{−x²}` where Cody's approximation stops (`x > 26.5`, the subnormal range: `Φ(x)` is non-zero down to `x ≈ −38.5`) / the crate's `erfcinv` | ≈ 1e-16 (the precision of a subnormal result in the subnormal range) |
//! | `t` | `½ I_{ν/(ν + x²)}(ν/2, ½)`; once `ν/x² < 1e-290` (so for `|x|` beyond `≈ 1e145`, where `x²` would soon overflow) the power law `½ (ν/x²)^{ν/2} / ((ν/2) B(ν/2, ½))` from `ln(ν/x²)`, so the tail is right up to `|x| = f64::MAX` (`P(T > 1e200) = 3.2e-101` for `ν = ½`) | ≈ 1e-15; ≈ 1e-13 in the power-law region (`|ν/2 · ln(ν/x²)| · ε`) |
//! | quantiles | safeguarded Newton on the logarithm of the relevant tail, in a log or logit variable, from a Cornish–Fisher / Wilson–Hilferty start | ≈ 1e-15 (the CDF's accuracy) |
//!
//! # Conventions
//!
//! * A distribution function returns an `f64`: `0`/`1` at `∓∞`, `NaN` for a
//!   `NaN` argument or an invalid parameter (a non-positive shape, a
//!   probability outside `[0, 1]`); no error is raised.
//! * A quantile (`ppf`, and `isf` for the upper tail) returns
//!   [`SymplexError::InvalidArgument`] for a level outside `(0, 1)` or an
//!   invalid parameter, and [`SymplexError::ComputationFailed`] should the
//!   iteration not converge.  Every error names the function that raised
//!   it (`"numdist::t::ppf"`).
//! * Discrete families take the argument `k` as an `f64` and floor it, as
//!   scipy does; their quantile is the smallest lattice point `k` with
//!   `F(k) ≥ p`, and their `isf` the smallest `k` with `P(X > k) ≤ q`.  The
//!   lattice parameter (`n`, `λ`) of a quantile must not exceed `2⁵³`,
//!   beyond which consecutive integers are no longer distinct doubles
//!   ([`SymplexError::InvalidArgument`]; scipy returns `NaN` there).
//! * `sf(x) = 1 − cdf(x)` is computed directly, not by subtraction, so a
//!   tail probability of `1e-300` keeps its relative accuracy.
//!
//! ```
//! use symplex::stats::numdist::{norm, t};
//!
//! // scipy: stats.norm.ppf(0.975) = 1.959963984540054
//! assert!((norm::ppf(0.975)? - 1.959963984540054).abs() < 1e-14);
//! // scipy: stats.t.ppf(0.975, 5) = 2.5705818356363146
//! assert!((t::ppf(0.975, 5.0)? - 2.5705818356363146).abs() < 1e-13);
//! // The tails are consistent and the quantile round-trips.
//! assert!((t::cdf(2.5705818356363146, 5.0) - 0.975).abs() < 1e-14);
//! assert!(t::sf(40.0, 30.0) < 1e-27 && t::sf(40.0, 30.0) > 0.0);
//! # Ok::<(), symplex::prelude::SymplexError>(())
//! ```

use crate::base::errors::SymplexError;
use crate::output::codegen::numeric_rt::{erfc as erfc_cody, erfcinv, lgamma};

/// `ln √(2π)`.
const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_7;
/// `√(2π)`.
const SQRT_2PI: f64 = 2.506_628_274_631_000_2;
/// `√π`.
const SQRT_PI: f64 = 1.772_453_850_905_516;
const EPS: f64 = f64::EPSILON;
/// Budget for the incomplete-gamma series and continued fraction (they need
/// about `8.5·√a` terms near `x ≈ a`, so this covers `a ≈ 10¹²`).
const GAMMA_MAX_ITER: usize = 10_000_000;
/// `2⁵³`: the largest lattice parameter (`n`, `λ`) of a discrete family for
/// which every integer `k` up to it is an exact `f64`.
const MAX_LATTICE_PARAM: f64 = 9_007_199_254_740_992.0;

// ═══════════════════════════════════════════════════════════════════════════
// erfc into the subnormal range
// ═══════════════════════════════════════════════════════════════════════════

/// From this argument on `erfc` is `erfcx(x)·e^{−x²}` rather than Cody's
/// rational approximation, which stops at `XBIG = 26.543` — the point where
/// `erfc` reaches the smallest normal double — and returns `0` beyond.  The
/// product form stays correct through the subnormal range, until `e^{−x²}`
/// itself underflows at `x ≈ 27.3` (`Φ(x)` for `x` down to `≈ −38.5`).
const ERFC_ASYMPTOTIC_MIN_X: f64 = 26.0;

/// `e^{−x²}` for `x ≥ 0` with the exponent split as Cody does it:
/// `xₛ = ⌊16x⌋/16` (so `xₛ²` is exact) and `δ = (x − xₛ)(x + xₛ)`, then
/// `e^{−xₛ²} e^{−δ}`.  The rounding of `x²` itself — `x²ε ≈ 8·10⁻¹⁴` in the
/// exponent at `x = 27` — never enters.
fn exp_neg_square(x: f64) -> f64 {
    let xs = (x * 16.0).floor() / 16.0;
    let del = (x - xs) * (x + xs);
    (-xs * xs).exp() * (-del).exp()
}

/// `e^{x²} erfc(x)` for `x ≥ 0`, without overflow.  Below
/// [`ERFC_ASYMPTOTIC_MIN_X`] it is Cody's `erfc` times `e^{x²}` with the
/// exponent split exactly as inside `erfc`, so the two exponentials cancel
/// to rounding; from there on the asymptotic series
/// `(1/(x√π)) Σ_{k≥0} (−1)^k (2k − 1)!! / (2x²)^k`, whose eleventh term is
/// below `2·10⁻²¹` at `x = 26`.
fn erfcx(x: f64) -> f64 {
    if x < ERFC_ASYMPTOTIC_MIN_X {
        let xs = (x * 16.0).floor() / 16.0;
        let del = (x - xs) * (x + xs);
        return erfc_cody(x) * (xs * xs).exp() * del.exp();
    }
    let ratio = -0.5 / (x * x);
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut odd = 1.0;
    for _ in 0..10 {
        term *= odd * ratio;
        sum += term;
        odd += 2.0;
    }
    sum / (x * SQRT_PI)
}

/// The complementary error function, Cody's approximation continued into
/// the subnormal range by `erfcx(x)·e^{−x²}` (see [`ERFC_ASYMPTOTIC_MIN_X`]).
fn erfc(x: f64) -> f64 {
    if x < ERFC_ASYMPTOTIC_MIN_X {
        erfc_cody(x)
    } else if x.is_infinite() {
        0.0
    } else {
        erfcx(x) * exp_neg_square(x)
    }
}

/// Both tails of a distribution function at one point, each computed
/// directly where it is the smaller one.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Tails {
    lower: f64,
    upper: f64,
}

impl Tails {
    const NAN: Tails = Tails {
        lower: f64::NAN,
        upper: f64::NAN,
    };
    const ZERO: Tails = Tails {
        lower: 0.0,
        upper: 1.0,
    };
    const ONE: Tails = Tails {
        lower: 1.0,
        upper: 0.0,
    };

    fn flipped(self) -> Tails {
        Tails {
            lower: self.upper,
            upper: self.lower,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Stirling remainder, log-beta and the prefactors x^a y^b / B(a,b), x^a e^{-x} / Γ(a)
// ═══════════════════════════════════════════════════════════════════════════

/// `ln Γ(z) − [(z − ½) ln z − z + ln √(2π)]`, the Stirling remainder
/// (Loader's `stirlerr`): the asymptotic series through `z⁻¹³` for
/// `z ≥ 10` (truncation below `3e-17`), the definition below.
fn stirlerr(z: f64) -> f64 {
    if z >= 10.0 {
        let inv = 1.0 / z;
        let inv2 = inv * inv;
        inv * (1.0 / 12.0
            - inv2
                * (1.0 / 360.0
                    - inv2
                        * (1.0 / 1260.0
                            - inv2
                                * (1.0 / 1680.0
                                    - inv2
                                        * (1.0 / 1188.0
                                            - inv2 * (691.0 / 360_360.0 - inv2 / 156.0))))))
    } else {
        lgamma(z) - ((z - 0.5) * z.ln() - z + LN_SQRT_2PI)
    }
}

/// `ln B(a, b)` for `a, b > 0`, without the cancellation of
/// `ln Γ(a) + ln Γ(b) − ln Γ(a + b)` for large arguments (R's `lbeta`).
fn lbeta(a: f64, b: f64) -> f64 {
    let (p, q) = if a < b { (a, b) } else { (b, a) };
    if p >= 10.0 {
        let corr = stirlerr(p) + stirlerr(q) - stirlerr(p + q);
        -0.5 * q.ln()
            + LN_SQRT_2PI
            + corr
            + (p - 0.5) * (p / (p + q)).ln()
            + q * (-p / (p + q)).ln_1p()
    } else if q >= 10.0 {
        let corr = stirlerr(q) - stirlerr(p + q);
        lgamma(p) + corr + p - p * (p + q).ln() + (q - 0.5) * (-p / (p + q)).ln_1p()
    } else {
        lgamma(p) + lgamma(q) - lgamma(p + q)
    }
}

/// `x − ln(1 + x)` without cancellation for small `x`: with
/// `r = x/(2 + x)`, `2r²[1/(1 − r) − r Σ_{k≥1} r^{2k−2}/(2k + 1)]`.
fn rlog1(x: f64) -> f64 {
    if !(-0.5..=0.5).contains(&x) {
        return x - x.ln_1p();
    }
    let r = x / (2.0 + x);
    let t = r * r;
    let mut w = 1.0 / 3.0;
    let mut term = 1.0;
    let mut k = 1.0;
    loop {
        term *= t;
        k += 1.0;
        let add = term / (2.0 * k + 1.0);
        w += add;
        if add <= EPS * w {
            break;
        }
    }
    2.0 * t * (1.0 / (1.0 - r) - r * w)
}

/// `k ln(k/m) + m − k` (Loader's `bd0`) given `diff = k − m` exactly and
/// `ln_ratio = ln(k/m)`: the series in `v = diff/(k + m)` when `k ≈ m`
/// (where the direct formula cancels), the direct formula otherwise.
fn bd0_with(k: f64, m: f64, diff: f64, ln_ratio: f64) -> f64 {
    if diff.abs() < 0.1 * (k + m) {
        let v = diff / (k + m);
        let v2 = v * v;
        let mut s = diff * v;
        let mut ej = 2.0 * k * v;
        let mut j = 1.0;
        loop {
            ej *= v2;
            let s1 = s + ej / (2.0 * j + 1.0);
            if s1 == s || j > 1000.0 {
                return s1;
            }
            s = s1;
            j += 1.0;
        }
    }
    k * ln_ratio - diff
}

/// `k ln(k/m) + m − k` for `k, m > 0`.
fn bd0(k: f64, m: f64) -> f64 {
    bd0_with(k, m, k - m, (k / m).ln())
}

/// `ln(xᵃ yᵇ / B(a, b))` for `x + y = 1`, the smaller of `x`, `y` taken
/// as exact (the other is `1 −` it, never rounded): with `n = a + b`,
/// `−bd0(a, nx) − bd0(b, ny) − [stirlerr(a) + stirlerr(b) − stirlerr(n)]
/// + ½ ln(ab/(2πn))`, in which the two `bd0` differences `nx − a` and
/// `ny − b` are the same number up to sign and are formed once.  `−∞`
/// when `x` or `y` is `0`.
fn log_beta_pref(a: f64, b: f64, x: f64, y: f64) -> f64 {
    if x <= 0.0 || y <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let n = a + b;
    let (s, small, large) = if y <= x { (y, b, a) } else { (x, a, b) };
    let ln_l = (-s).ln_1p();
    let ns = n * s;
    let d = ns - small;
    let t_small = bd0_with(small, ns, -d, (small / n).ln() - s.ln());
    let nl = n - ns;
    let t_large = bd0_with(large, nl, d, (large / n).ln() - ln_l);
    -t_small - t_large - stirlerr(a) - stirlerr(b) + stirlerr(n) + 0.5 * (a * b / n).ln()
        - LN_SQRT_2PI
}

/// `ln(xᵃ e⁻ˣ / Γ(a))` for `a, x > 0`: `−stirlerr(a) − bd0(a, x) + ½ ln(a/(2π))`.
fn log_gamma_pref(a: f64, x: f64) -> f64 {
    -stirlerr(a) - bd0(a, x) + 0.5 * a.ln() - LN_SQRT_2PI
}

// ═══════════════════════════════════════════════════════════════════════════
// Regularised incomplete gamma
// ═══════════════════════════════════════════════════════════════════════════

/// `P(a, x) / r` with `r = xᵃe⁻ˣ/Γ(a)`: `Σ_{n≥0} xⁿ / (a(a+1)⋯(a+n))`.
fn gamma_p_series_over_r(a: f64, x: f64) -> f64 {
    let mut term = 1.0 / a;
    let mut sum = term;
    let mut ap = a;
    for _ in 0..GAMMA_MAX_ITER {
        ap += 1.0;
        term *= x / ap;
        sum += term;
        if term <= sum * EPS {
            break;
        }
    }
    sum
}

/// `Q(a, x) / r` by Lentz's algorithm on Legendre's continued fraction
/// (`x ≥ a + 1`, where it converges fast).
fn gamma_q_cf_over_r(a: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    let mut i = 1.0;
    for _ in 0..GAMMA_MAX_ITER {
        let an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
        i += 1.0;
    }
    h
}

/// Above this `a`, `x` within a few standard deviations of `a` goes to
/// Temme's uniform asymptotic expansion: the power series needs `O(√a)`
/// terms there and the continued fraction converges slowly, and both lose
/// digits to the accumulated rounding (`3·10⁻¹¹` relative at `a = 5·10⁷`).
/// The expansion is truncated after `C₁/a`, so its own error is
/// `≈ |C₂|/a² ≈ 4·10⁻³/a²`: below `4·10⁻¹⁵` from this threshold on.
const GAMMA_TEMME_MIN_A: f64 = 1e6;

/// `η` of Temme's expansion from `d = λ − 1 = (x − a)/a`: `½η² = d − ln(1 + d)`,
/// with the sign of `d`.  The caller forms `d` as `(x − a)/a` — exact in the
/// difference by Sterbenz's lemma — never as `x/a − 1`, which loses four
/// digits at `a = 5·10⁷` and, amplified through `e^{−aη²/2}`, costs
/// `10⁻¹⁰` in the tail.
fn temme_eta(d: f64) -> f64 {
    // φ = d − ln(1 + d) = d²/2 − d³/3 + d⁴/4 − …  The subtraction loses
    // ≈ log₁₀(1/|d|) digits, which a√φ then carries into the exponent, so
    // the alternating series (24 terms suffice to double precision for
    // |d| < 0.1; its terms are all `O(d^k)`) is used well beyond the tiny
    // range where the direct form is catastrophic.
    let phi = if d.abs() < 0.1 {
        let mut term = d * d; // d^k
        let mut sum = 0.0;
        let mut k = 2.0;
        let mut positive = true; // +d²/2 − d³/3 + d⁴/4 − …
        loop {
            let contribution = term / k;
            let next = if positive {
                sum + contribution
            } else {
                sum - contribution
            };
            positive = !positive;
            if next == sum || k > 60.0 {
                break next;
            }
            sum = next;
            term *= d;
            k += 1.0;
        }
    } else {
        d - d.ln_1p()
    };
    let eta = (2.0 * phi).sqrt();
    if d < 0.0 { -eta } else { eta }
}

/// `c₀(η)` and `c₁(η)` of Temme's expansion, DLMF §8.12 (<https://dlmf.nist.gov/8.12>):
/// with `μ = λ − 1`,
///
/// ```text
/// c₀(η) = 1/μ − 1/η                                            (DLMF 8.12.9)
/// c_k(η) = (1/η) dc_{k−1}/dη + (−1)^k g_k / μ                   (DLMF 8.12.10)
/// ```
///
/// where `g_k` are the Stirling coefficients of `Γ*(a) ~ Σ g_k a⁻ᵏ`
/// (DLMF 5.11.3–5.11.4: `g₀ = 1`, `g₁ = 1/12`, `g₂ = 1/288`, …), so
/// `c₁ = c₀′/η − (1/12)/μ`; `dλ/dη = ηλ/μ` is the reciprocal of 8.12.2.
/// Both closed forms cancel catastrophically as `η → 0`, where the Taylor
/// series `c_k(η) = Σ d_{k,n} ηⁿ` (DLMF 8.12.11–8.12.14, exact rationals;
/// DiDonato & Morris 1986, TOMS 654, tabulate them to 30 digits) take over:
///
/// ```text
/// c₀ = −1/3 + η/12 − 2η²/135 + η³/864 + η⁴/2835 − 139η⁵/777600 + η⁶/25515
///      − 571η⁷/261273600 − 281η⁸/151559100 + …
/// c₁ = −1/540 − η/288 + η²/378 − 77η³/77760 + η⁴/4860 − η⁵/2488320
///      − 2743η⁶/151559100 + …
/// ```
///
/// (`c₀(0) = −1/3`, `c₁(0) = −1/540`: DLMF 8.12.17.)  The switch at
/// `|η| = 0.05` keeps the omitted series terms below `10⁻¹⁶` and the
/// closed forms' cancellation below two digits.
fn temme_c0_c1(eta: f64, d: f64) -> (f64, f64) {
    if eta.abs() < 0.05 {
        let e = eta;
        let c0 = -1.0 / 3.0
            + e * (1.0 / 12.0
                + e * (-2.0 / 135.0
                    + e * (1.0 / 864.0
                        + e * (1.0 / 2835.0
                            + e * (-139.0 / 777_600.0
                                + e * (1.0 / 25_515.0
                                    + e * (-571.0 / 261_273_600.0
                                        + e * (-281.0 / 151_559_100.0))))))));
        let c1 = -1.0 / 540.0
            + e * (-1.0 / 288.0
                + e * (1.0 / 378.0
                    + e * (-77.0 / 77_760.0
                        + e * (1.0 / 4860.0
                            + e * (-1.0 / 2_488_320.0 + e * (-2743.0 / 151_559_100.0))))));
        return (c0, c1);
    }
    let lm1 = d;
    let c0 = 1.0 / lm1 - 1.0 / eta;
    let dlambda = eta * (1.0 + d) / lm1;
    let dc0 = -dlambda / (lm1 * lm1) + 1.0 / (eta * eta);
    let c1 = dc0 / eta - (1.0 / 12.0) / lm1;
    (c0, c1)
}

/// `(P, Q)` by Temme's uniform asymptotic expansion for large `a`
/// (Temme 1979; DLMF 8.12.3–8.12.4, 8.12.7):
///
/// ```text
/// Q(a, x) = ½ erfc(η √(a/2)) + S(a, η),   P(a, x) = ½ erfc(−η √(a/2)) − S(a, η),
/// S(a, η) ~ e^{−aη²/2} / √(2πa) · Σ_k c_k(η) a⁻ᵏ,
/// ```
///
/// truncated after `c₁/a`.  On `|η| ≤ 1.5`, `|c₂(η)| ≤ 0.0089` (maximum at
/// `η = −1.5`; `c₂(0) = 25/6048`), so the truncation error is below
/// `0.009/a²` — `9·10⁻¹⁵` at `a = 10⁶` — and the result is limited by the
/// `f64` argument itself: one ulp of `x` moves the tail by `≈ |η|√a · 2⁻⁵³`
/// relative.  Verified against 50-digit quadrature of the density at
/// `a ∈ {10⁶, 5·10⁷, 10⁹}`, `|x − a| ≤ 8√a`: worst relative error `9·10⁻¹³`
/// (at `a = 10⁹`, `x = a + 8√a`, where that ulp effect is `3·10⁻¹¹` per ulp).
///
/// With `z = η√(a/2)` both pieces of the smaller tail carry the same
/// exponential, `½ erfc(|z|) = ½ erfcx(|z|) e^{−z²}` and
/// `S = e^{−z²} (c₀ + c₁/a)/√(2πa)`, so that exponential is factored out
/// and the two mantissas are summed before the one multiplication.  Summing
/// the finished pieces instead let `erfc` underflow to `0` at `z > 26.5`
/// while `S` survived and returned a *negative* tail (`−6·10⁻³²¹` for
/// `Q(5·10⁷, 5.027·10⁷)`, truly `3.6·10⁻³¹⁸`).  The larger tail is `1 −` the
/// smaller one, and the smaller is clamped into `[0, 1]`.
fn gammainc_tails_temme(a: f64, x: f64) -> Tails {
    // (x − a) is exact for a/2 ≤ x ≤ 2a (Sterbenz), which the caller's
    // 40σ window guarantees; x/a − 1 would not be.
    let d = (x - a) / a;
    let eta = temme_eta(d);
    let z = eta * (a / 2.0).sqrt();
    let (c0, c1) = temme_c0_c1(eta, d);
    let correction = (c0 + c1 / a) / (2.0 * std::f64::consts::PI * a).sqrt();
    // Upper tail Q = ½ erfc(z) + S for z ≥ 0; lower tail P = ½ erfc(−z) − S for z < 0.
    let mantissa = if z >= 0.0 {
        0.5 * erfcx(z) + correction
    } else {
        0.5 * erfcx(-z) - correction
    };
    let small = (mantissa * exp_neg_square(z.abs())).clamp(0.0, 1.0);
    if z >= 0.0 {
        Tails {
            lower: 1.0 - small,
            upper: small,
        }
    } else {
        Tails {
            lower: small,
            upper: 1.0 - small,
        }
    }
}

/// `(P(a, x), Q(a, x))`.
fn gammainc_tails(a: f64, x: f64) -> Tails {
    if a.is_nan() || x.is_nan() || a <= 0.0 || a.is_infinite() || x < 0.0 {
        return Tails::NAN;
    }
    if x == 0.0 {
        return Tails::ZERO;
    }
    if x.is_infinite() {
        return Tails::ONE;
    }
    if a >= GAMMA_TEMME_MIN_A {
        // Within ~40 standard deviations of the mean the uniform expansion
        // is both faster and more accurate than the series / fraction.
        let sd = a.sqrt();
        if (x - a).abs() < 40.0 * sd {
            return gammainc_tails_temme(a, x);
        }
    }
    let r = log_gamma_pref(a, x).exp();
    if x < a + 1.0 {
        let p = r * gamma_p_series_over_r(a, x);
        Tails {
            lower: p,
            upper: 1.0 - p,
        }
    } else {
        let q = r * gamma_q_cf_over_r(a, x);
        Tails {
            lower: 1.0 - q,
            upper: q,
        }
    }
}

/// The regularised lower incomplete gamma function
/// `P(a, x) = γ(a, x)/Γ(a)` for `a > 0`, `x ≥ 0` — `scipy.special.gammainc`.
/// `NaN` for `a ≤ 0`, `x < 0` or a `NaN` argument.
///
/// ```
/// use symplex::stats::numdist::gammainc_lower_regularized_f64;
///
/// // P(1, 2) = 1 − e⁻²
/// let p = gammainc_lower_regularized_f64(1.0, 2.0);
/// assert!((p - (1.0 - (-2.0f64).exp())).abs() < 1e-15);
/// ```
pub fn gammainc_lower_regularized_f64(a: f64, x: f64) -> f64 {
    gammainc_tails(a, x).lower
}

/// The regularised upper incomplete gamma function
/// `Q(a, x) = Γ(a, x)/Γ(a) = 1 − P(a, x)`, computed directly where it is
/// small — `scipy.special.gammaincc`.
///
/// ```
/// use symplex::stats::numdist::gammainc_upper_regularized_f64;
///
/// // Q(1, 40) = e⁻⁴⁰, far below what 1 − P could resolve.
/// let q = gammainc_upper_regularized_f64(1.0, 40.0);
/// assert!((q / (-40.0f64).exp() - 1.0).abs() < 1e-14);
/// ```
pub fn gammainc_upper_regularized_f64(a: f64, x: f64) -> f64 {
    gammainc_tails(a, x).upper
}

// ═══════════════════════════════════════════════════════════════════════════
// Regularised incomplete beta (the TOMS 708 dispatch)
// ═══════════════════════════════════════════════════════════════════════════

/// `I_x(a, b)` by the power series
/// `xᵃ/(a B(a, b)) · [1 + a Σ_{n≥1} (1−b)(2−b)⋯(n−b)/n! · xⁿ/(a + n)]`,
/// for `b ≤ 1` or `b x ≤ 0.7` and `x ≤ 0.7`.
fn bpser(a: f64, b: f64, x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let lead = (a * x.ln() - lbeta(a, b)).exp() / a;
    if lead == 0.0 {
        return 0.0;
    }
    let tol = EPS / a;
    let mut n = 0.0;
    let mut sum = 0.0;
    let mut c = 1.0;
    loop {
        n += 1.0;
        c *= (1.0 - b / n) * x;
        let w = c / (a + n);
        sum += w;
        if w.abs() <= tol || n >= 1e7 {
            break;
        }
    }
    lead * (1.0 + a * sum)
}

/// `I_x(a, b) − I_x(a + n, b)` for an integer `n ≥ 1`: the sum of the `n`
/// positive terms `x^{a+j} yᵇ / ((a + j) B(a + j, b))`, accumulated
/// relative to the largest so far so that no term over- or underflows.
fn bup(a: f64, b: f64, x: f64, y: f64, n: usize) -> f64 {
    let log_t0 = log_beta_pref(a, b, x, y) - a.ln();
    if log_t0 == f64::NEG_INFINITY {
        return 0.0;
    }
    let apb = a + b;
    let ap1 = a + 1.0;
    let lnx = x.ln();
    let mut log_t = log_t0;
    let mut max = log_t;
    let mut sum = 1.0;
    for j in 1..n {
        let jf = (j - 1) as f64;
        log_t += lnx + ((apb + jf) / (ap1 + jf)).ln();
        if log_t > max {
            sum = sum * (max - log_t).exp() + 1.0;
            max = log_t;
        } else {
            // The term ratio is monotone in `j`, so once the terms decrease
            // they keep decreasing and a negligible one ends the sum.
            let term = (log_t - max).exp();
            sum += term;
            if term <= EPS * sum {
                break;
            }
        }
    }
    max.exp() * sum
}

/// `1/Γ(a + 1) − 1` for `0 ≤ a ≤ 1`.
fn gam1(a: f64) -> f64 {
    let lg = lgamma(a + 1.0);
    -lg.exp_m1() * (-lg).exp()
}

/// `Q(a, x) / r` with `r = e⁻ˣxᵃ/Γ(a) = exp(log_r)`, for `a ≤ 1`
/// (TOMS 708 `grat_r`): a Taylor expansion of `P(a, x)/xᵃ` for `x < 1.1`
/// arranged so that neither tail cancels, the continued fraction beyond.
fn grat_r(a: f64, x: f64, log_r: f64) -> f64 {
    if a * x == 0.0 {
        return if x <= a { (-log_r).exp() } else { 0.0 };
    }
    if x >= 1.1 {
        return gamma_q_cf_over_r(a, x);
    }
    let mut an = 3.0;
    let mut c = x;
    let mut sum = x / (a + 3.0);
    let tol = 0.1 * EPS / (a + 1.0);
    loop {
        an += 1.0;
        c *= -(x / an);
        let t = c / (a + an);
        sum += t;
        if t.abs() <= tol {
            break;
        }
    }
    // j = 1 − Γ(a + 1) P(a, x) / xᵃ
    let j = a * x * ((sum / 6.0 - 0.5 / (a + 2.0)) * x + 1.0 / (a + 1.0));
    let z = a * x.ln();
    let h = gam1(a);
    let g = h + 1.0;
    if (x >= 0.25 && a < x / 2.59) || z > -0.13394 {
        // Q directly, through eᶻ − 1: no `1 − P` cancellation.
        let l = z.exp_m1();
        let q = ((l + 1.0) * j - l) * g - h;
        if q <= 0.0 { 0.0 } else { q * (-log_r).exp() }
    } else {
        let p = z.exp() * g * (1.0 - j);
        (1.0 - p) * (-log_r).exp()
    }
}

/// `I_x(a, b)` for `a ≥ 15`, `b ≤ 1` by the asymptotic expansion of
/// DiDonato & Morris (1992, §9) in the incomplete gamma function of
/// `−(a + (b−1)/2) ln x`; `None` when it cannot be evaluated (the leading
/// term underflows or the partial sums turn negative).
fn bgrat(a: f64, b: f64, x: f64, y: f64) -> Option<f64> {
    const N_TERMS: usize = 30;
    let bm1 = b - 1.0;
    let nu = a + 0.5 * bm1;
    let lnx = if y > 0.375 { x.ln() } else { (-y).ln_1p() };
    let z = -nu * lnx;
    if b * z == 0.0 {
        return None;
    }
    // r = e⁻ᶻ zᵇ / Γ(b);  u = r · Γ(a + b) / (Γ(a) νᵇ).
    let log_r = b * z.ln() - z - lgamma(b);
    let log_u = log_r - ((lbeta(a, b) - lgamma(b)) + b * nu.ln());
    if log_u == f64::NEG_INFINITY {
        // The whole expansion underflows: the tail is below the smallest double.
        return Some(0.0);
    }
    let u = log_u.exp();
    let v = 0.25 / (nu * nu);
    let t2 = lnx * 0.25 * lnx;
    let mut j = grat_r(b, z, log_r);
    let mut sum = j;
    let mut t = 1.0;
    let mut cn = 1.0;
    let mut n2 = 0.0;
    let mut c = [0.0; N_TERMS];
    let mut d = [0.0; N_TERMS];
    for n in 1..=N_TERMS {
        let bp2n = b + n2;
        j = (bp2n * (bp2n + 1.0) * j + (z + bp2n + 1.0) * t) * v;
        n2 += 2.0;
        t *= t2;
        cn /= n2 * (n2 + 1.0);
        c[n - 1] = cn;
        let mut s = 0.0;
        if n > 1 {
            let mut coef = b - n as f64;
            for i in 1..n {
                s += coef * c[i - 1] * d[n - 1 - i];
                coef += b;
            }
        }
        d[n - 1] = bm1 * cn + s / n as f64;
        let dj = d[n - 1] * j;
        sum += dj;
        if sum <= 0.0 {
            return None;
        }
        if dj.abs() <= 15.0 * EPS * sum {
            break;
        }
    }
    Some(if u == 0.0 {
        (log_u + sum.ln()).exp()
    } else {
        u * sum
    })
}

/// `I_x(a, b)` for `a, b > 1` by the continued fraction of DiDonato &
/// Morris, `λ = (a + b) y − b ≥ 0` (so `x` is below the mean and the value
/// is the smaller tail).
fn bfrac(a: f64, b: f64, x: f64, y: f64, lambda: f64) -> f64 {
    let brc = log_beta_pref(a, b, x, y).exp();
    if brc == 0.0 {
        return 0.0;
    }
    let c = lambda + 1.0;
    let c0 = b / a;
    let c1 = 1.0 / a + 1.0;
    let yp1 = y + 1.0;
    let mut n = 0.0;
    let mut p = 1.0;
    let mut s = a + 1.0;
    let mut an = 0.0;
    let mut bn = 1.0;
    let mut anp1 = 1.0;
    let mut bnp1 = c / c1;
    let mut r = c1 / c;
    for _ in 0..10_000 {
        n += 1.0;
        let w = n * x * (b - n);
        let t = n / a;
        let e = a / s;
        let alpha = p * (p + c0) * e * e * (w * x);
        let e = (t + 1.0) / (c1 + t + t);
        let beta = w / s + n + e * (c + n * yp1);
        p = t + 1.0;
        s += 2.0;
        let t = alpha * an + beta * anp1;
        an = anp1;
        anp1 = t;
        let t = alpha * bn + beta * bnp1;
        bn = bnp1;
        bnp1 = t;
        let r0 = r;
        r = anp1 / bnp1;
        if (r - r0).abs() <= 15.0 * EPS * r {
            break;
        }
        an /= bnp1;
        bn /= bnp1;
        anp1 = r;
        bnp1 = 1.0;
    }
    brc * r
}

/// `I_x(a, b)` for large `a, b` (both `≥ 15`, neither below `100` unless
/// `λ` is small) by the asymptotic expansion of DiDonato & Morris,
/// `λ = (a + b) y − b ≥ 0`.
fn basym(a: f64, b: f64, lambda: f64) -> f64 {
    const NUM: usize = 20;
    /// `2/√π`.
    const E0: f64 = std::f64::consts::FRAC_2_SQRT_PI;
    /// `2^{−3/2}`.
    const E1: f64 = 0.353_553_390_593_273_7;
    let f = a * rlog1(-lambda / a) + b * rlog1(lambda / b);
    let t = (-f).exp();
    if t == 0.0 {
        return 0.0;
    }
    let z0 = f.sqrt();
    let z = 0.5 * z0 / E1;
    let z2 = f + f;
    let (h, r0, r1, w0) = if a < b {
        let h = a / b;
        (
            h,
            1.0 / (h + 1.0),
            (b - a) / b,
            1.0 / (a * (h + 1.0)).sqrt(),
        )
    } else {
        let h = b / a;
        (
            h,
            1.0 / (h + 1.0),
            (b - a) / a,
            1.0 / (b * (h + 1.0)).sqrt(),
        )
    };
    let mut a0 = [0.0; NUM + 1];
    let mut b0 = [0.0; NUM + 1];
    let mut c = [0.0; NUM + 1];
    let mut d = [0.0; NUM + 1];
    a0[0] = r1 * 2.0 / 3.0;
    c[0] = -0.5 * a0[0];
    d[0] = -c[0];
    let mut j0 = 0.5 / E0 * erfcx(z0);
    let mut j1 = E1;
    let mut sum = j0 + d[0] * w0 * j1;
    let mut s = 1.0;
    let h2 = h * h;
    let mut hn = 1.0;
    let mut w = w0;
    let mut znm1 = z;
    let mut zn = z2;
    let mut n = 2;
    while n <= NUM {
        hn *= h2;
        a0[n - 1] = r0 * 2.0 * (h * hn + 1.0) / (n as f64 + 2.0);
        let np1 = n + 1;
        s += hn;
        a0[np1 - 1] = r1 * 2.0 * s / (n as f64 + 3.0);
        for i in n..=np1 {
            let r = -0.5 * (i as f64 + 1.0);
            b0[0] = r * a0[0];
            for m in 2..=i {
                let mut bsum = 0.0;
                for j in 1..m {
                    let mmj = m - j;
                    bsum += (j as f64 * r - mmj as f64) * a0[j - 1] * b0[mmj - 1];
                }
                b0[m - 1] = r * a0[m - 1] + bsum / m as f64;
            }
            c[i - 1] = b0[i - 1] / (i as f64 + 1.0);
            let mut dsum = 0.0;
            for j in 1..i {
                dsum += d[i - j - 1] * c[j - 1];
            }
            d[i - 1] = -(dsum + c[i - 1]);
        }
        j0 = E1 * znm1 + (n as f64 - 1.0) * j0;
        j1 = E1 * zn + n as f64 * j1;
        znm1 *= z2;
        zn *= z2;
        w *= w0;
        let t0 = d[n - 1] * w * j0;
        w *= w0;
        let t1 = d[np1 - 1] * w * j1;
        sum += t0 + t1;
        if t0.abs() + t1.abs() <= 100.0 * EPS * sum {
            break;
        }
        n += 2;
    }
    let bcorr = stirlerr(a) + stirlerr(b) - stirlerr(a + b);
    E0 * t * (-bcorr).exp() * sum
}

/// A tail probability: the lower tail `P(X ≤ x)` or the upper tail
/// `P(X > x)`.  Names which tail a route of [`bratio`] computed directly,
/// and which tail a quantile inversion targets.
#[derive(Clone, Copy, Debug)]
enum Side {
    Lower(f64),
    Upper(f64),
}

/// The TOMS 708 dispatch for `I_x(a, b)`, `a, b > 0`, `0 < x < 1`,
/// `y = 1 − x`: chooses the route, and the tail it computes directly, from
/// the sizes of `a`, `b` and the position of `x` relative to the mean.
/// `None` when `bgrat` fails.
fn bratio_route(a: f64, b: f64, x: f64, y: f64) -> Option<(Side, bool)> {
    let (side, swap) = if a.min(b) <= 1.0 {
        let swap = x > 0.5;
        let (a0, b0, x0, y0) = if swap { (b, a, y, x) } else { (a, b, x, y) };
        // Now x0 ≤ ½ ≤ y0.
        let side = if a0.max(b0) > 1.0 {
            if b0 <= 1.0 {
                Side::Lower(bpser(a0, b0, x0))
            } else if x0 >= 0.29 {
                Side::Upper(bpser(b0, a0, y0))
            } else if x0 < 0.1 && (x0 * b0).powf(a0) <= 0.7 {
                Side::Lower(bpser(a0, b0, x0))
            } else if b0 > 15.0 {
                Side::Upper(bgrat(b0, a0, y0, x0)?)
            } else {
                Side::Upper(bup(b0, a0, y0, x0, 20) + bgrat(b0 + 20.0, a0, y0, x0)?)
            }
        } else if a0 >= 0.2f64.min(b0) || x0.powf(a0) <= 0.9 {
            Side::Lower(bpser(a0, b0, x0))
        } else if x0 >= 0.3 {
            Side::Upper(bpser(b0, a0, y0))
        } else {
            Side::Upper(bup(b0, a0, y0, x0, 20) + bgrat(b0 + 20.0, a0, y0, x0)?)
        };
        (side, swap)
    } else {
        // λ = a y − b x: positive when x is below the mean a/(a + b).
        let lambda = if a > b {
            (a + b) * y - b
        } else {
            a - (a + b) * x
        };
        let swap = lambda < 0.0;
        let (a0, b0, x0, y0, lambda) = if swap {
            (b, a, y, x, -lambda)
        } else {
            (a, b, x, y, lambda)
        };
        let side = if b0 < 40.0 {
            if b0 * x0 <= 0.7 {
                Side::Lower(bpser(a0, b0, x0))
            } else {
                // Reduce b0 to its fractional part bf ∈ (0, 1] by the
                // recurrence, then finish with the series or bgrat.
                let mut n = b0.floor();
                let mut bf = b0 - n;
                if bf == 0.0 {
                    n -= 1.0;
                    bf = 1.0;
                }
                let mut w = bup(bf, a0, y0, x0, n as usize);
                if x0 <= 0.7 {
                    w += bpser(a0, bf, x0);
                } else {
                    let mut aa = a0;
                    if aa <= 15.0 {
                        w += bup(aa, bf, x0, y0, 20);
                        aa += 20.0;
                    }
                    w += bgrat(aa, bf, x0, y0)?;
                }
                Side::Lower(w)
            }
        } else if a0 > b0 {
            if b0 <= 100.0 || lambda > 0.03 * b0 {
                Side::Lower(bfrac(a0, b0, x0, y0, lambda))
            } else {
                Side::Lower(basym(a0, b0, lambda))
            }
        } else if a0 <= 100.0 || lambda > 0.03 * a0 {
            Side::Lower(bfrac(a0, b0, x0, y0, lambda))
        } else {
            Side::Lower(basym(a0, b0, lambda))
        };
        (side, swap)
    };
    Some((side, swap))
}

/// `(I_x(a, b), 1 − I_x(a, b))` for `a, b > 0` and `x + y = 1`, each tail
/// computed directly where it is the smaller one.  `NaN` for invalid
/// arguments.
fn bratio(a: f64, b: f64, x: f64, y: f64) -> Tails {
    if a.is_nan()
        || b.is_nan()
        || x.is_nan()
        || y.is_nan()
        || a <= 0.0
        || b <= 0.0
        || !(0.0..=1.0).contains(&x)
        || !(0.0..=1.0).contains(&y)
    {
        return Tails::NAN;
    }
    if x == 0.0 {
        return Tails::ZERO;
    }
    if y == 0.0 {
        return Tails::ONE;
    }
    let Some((side, swap)) = bratio_route(a, b, x, y) else {
        return Tails::NAN;
    };
    let tails = match side {
        Side::Lower(w) => Tails {
            lower: w,
            upper: 1.0 - w,
        },
        Side::Upper(w1) => Tails {
            lower: 1.0 - w1,
            upper: w1,
        },
    };
    if swap { tails.flipped() } else { tails }
}

/// The regularised incomplete beta function `I_x(a, b)` for `a, b > 0`
/// and `0 ≤ x ≤ 1` — `scipy.special.betainc`.  `NaN` outside that domain.
///
/// ```
/// use symplex::stats::numdist::betainc_regularized_f64;
///
/// // I_{0.4}(2, 3) = 0.5248 exactly (a polynomial for integer shapes).
/// assert!((betainc_regularized_f64(2.0, 3.0, 0.4) - 0.5248).abs() < 1e-15);
/// assert_eq!(betainc_regularized_f64(2.0, 3.0, 0.0), 0.0);
/// assert_eq!(betainc_regularized_f64(2.0, 3.0, 1.0), 1.0);
/// ```
pub fn betainc_regularized_f64(a: f64, b: f64, x: f64) -> f64 {
    bratio(a, b, x, 1.0 - x).lower
}

// ═══════════════════════════════════════════════════════════════════════════
// Argument checks and the quantile solver
// ═══════════════════════════════════════════════════════════════════════════

fn invalid(op: &'static str, reason: String) -> SymplexError {
    SymplexError::invalid_argument(op, reason)
}

/// `0 < p < 1` for a probability level named `name`.
fn check_level(op: &'static str, name: &str, p: f64) -> Result<(), SymplexError> {
    if p > 0.0 && p < 1.0 {
        Ok(())
    } else {
        Err(invalid(
            op,
            format!("{name} must lie strictly between 0 and 1, got {p}"),
        ))
    }
}

/// `v` finite and positive, for a parameter named `name`.
fn check_positive(op: &'static str, name: &str, v: f64) -> Result<(), SymplexError> {
    if v.is_finite() && v > 0.0 {
        Ok(())
    } else {
        Err(invalid(
            op,
            format!("{name} must be finite and positive, got {v}"),
        ))
    }
}

/// A lower-tail level `p` split into the tail it is nearest to: `Lower(p)`
/// for `p ≤ ½`, `Upper(1 − p)` otherwise, so an inversion always targets
/// the smaller tail.
fn nearest_tail(p: f64) -> Side {
    if p <= 0.5 {
        Side::Lower(p)
    } else {
        Side::Upper(1.0 - p)
    }
}

/// The same for an upper-tail level `q` (`isf`): `Upper(q)` for `q ≤ ½`,
/// so a tiny `q` is never rounded through `1 − q`.
fn nearest_tail_upper(q: f64) -> Side {
    if q <= 0.5 {
        Side::Upper(q)
    } else {
        Side::Lower(1.0 - q)
    }
}

/// A monotone objective and its derivative at one point.
struct Eval {
    g: f64,
    dg: f64,
}

/// The root of an increasing `g` by Newton's method safeguarded by a
/// bracket: a Newton step is taken when it lands inside the current
/// bracket and at least halves it, a bisection (or, while an end is still
/// infinite, a doubling step outwards) otherwise.  `g` may return `±∞`
/// (a tail that underflowed): only its sign is used then.  Converged when
/// the step is below `16ε(1 + |v|)`, the resolution of a log or logit
/// variable, or when a step already below `10⁻⁶(1 + |v|)` fails to
/// reduce `|g|` — the iteration has reached the rounding noise of `g`,
/// and the best iterate is the answer.
fn solve_increasing(
    op: &'static str,
    g: impl Fn(f64) -> Eval,
    v0: f64,
    lo: f64,
    hi: f64,
) -> Result<f64, SymplexError> {
    const MAX_ITER: usize = 200;
    const TOL: f64 = 16.0 * EPS;
    const QUADRATIC_PHASE: f64 = 1e-6;
    if v0.is_nan() {
        return Err(SymplexError::computation_failed(
            op,
            "the starting point of the quantile iteration is not a number",
        ));
    }
    let (mut lo, mut hi) = (lo, hi);
    let mut v = v0.clamp(lo, hi);
    let mut step = 1.0;
    let mut width = f64::INFINITY;
    let mut best = (v, f64::INFINITY);
    for _ in 0..MAX_ITER {
        let Eval { g: gv, dg } = g(v);
        if gv.is_nan() {
            return Err(SymplexError::computation_failed(
                op,
                format!("the distribution function is not a number at {v}"),
            ));
        }
        if gv == 0.0 {
            return Ok(v);
        }
        if gv.abs() < best.1 {
            best = (v, gv.abs());
        } else if width <= QUADRATIC_PHASE * (1.0 + v.abs()) {
            return Ok(best.0);
        }
        if gv < 0.0 {
            lo = v;
        } else {
            hi = v;
        }
        let newton = v - gv / dg;
        let newton_ok = gv.is_finite()
            && dg.is_finite()
            && dg > 0.0
            && newton > lo
            && newton < hi
            && 2.0 * gv.abs() <= (width * dg).abs();
        let next = if newton_ok {
            newton
        } else if gv < 0.0 {
            if hi.is_finite() {
                0.5 * (lo + hi)
            } else {
                step *= 2.0;
                v + step
            }
        } else if lo.is_finite() {
            0.5 * (lo + hi)
        } else {
            step *= 2.0;
            v - step
        };
        let tol = TOL * (1.0 + v.abs());
        if (next - v).abs() <= tol {
            return Ok(next);
        }
        if hi - lo <= tol {
            return Ok(0.5 * (lo + hi));
        }
        width = (next - v).abs();
        v = next;
    }
    Err(SymplexError::computation_failed(
        op,
        format!("the quantile iteration did not converge in {MAX_ITER} steps"),
    ))
}

/// A lattice parameter (`n` of the binomial, `λ` of the Poisson) at most
/// [`MAX_LATTICE_PARAM`]`= 2⁵³`: beyond it consecutive integers `k` are no
/// longer distinct doubles and a quantile has no exact answer (scipy
/// returns `NaN` there; the search here would run on a lattice coarser
/// than the integers).
fn check_lattice_param(op: &'static str, name: &str, v: f64) -> Result<(), SymplexError> {
    if v <= MAX_LATTICE_PARAM {
        Ok(())
    } else {
        Err(invalid(
            op,
            format!(
                "{name} must not exceed 2^53 = {MAX_LATTICE_PARAM}, above which the lattice points k are not representable as f64, got {v}"
            ),
        ))
    }
}

/// The smallest lattice point `k ∈ [kmin, kmax]` at which a monotone
/// predicate `done` holds (`done` is `false` below some threshold and `true`
/// from it on), from a guess `k0`: a doubling search for a bracket, then
/// bisection on the integers.  `done` returns `None` when the distribution
/// function is not a number.
///
/// The bisection ends as soon as the midpoint no longer separates the
/// bracket — which happens only when `lo` and `hi` are adjacent doubles
/// beyond `2⁵³`, where the lattice is coarser than the integers; `hi`,
/// the smallest representable point known to satisfy the predicate, is
/// then the answer.  (The callers reject parameters above `2⁵³`, so this
/// is reached only within a few standard deviations above that limit.)
fn discrete_search(
    op: &'static str,
    done: impl Fn(f64) -> Option<bool>,
    k0: f64,
    kmin: f64,
    kmax: f64,
) -> Result<f64, SymplexError> {
    const MAX_STEPS: usize = 2000;
    let nan = || SymplexError::computation_failed(op, "the distribution function is not a number");
    let k0 = k0.round().clamp(kmin, kmax);
    let Some(d0) = done(k0) else {
        return Err(nan());
    };
    let mut lo;
    let mut hi;
    let mut step = 1.0;
    if d0 {
        hi = k0;
        lo = k0 - 1.0;
        for _ in 0..MAX_STEPS {
            if lo < kmin {
                lo = kmin - 1.0;
                break;
            }
            let Some(d) = done(lo) else {
                return Err(nan());
            };
            if !d {
                break;
            }
            hi = lo;
            lo -= step;
            step *= 2.0;
        }
    } else {
        lo = k0;
        hi = k0 + 1.0;
        for _ in 0..MAX_STEPS {
            if hi >= kmax {
                hi = kmax;
                break;
            }
            let Some(d) = done(hi) else {
                return Err(nan());
            };
            if d {
                break;
            }
            lo = hi;
            hi += step;
            step *= 2.0;
        }
    }
    // !done(lo) and done(hi) (or lo is below the support).
    while hi - lo > 1.0 {
        let mid = (0.5 * (lo + hi)).floor();
        if mid <= lo || mid >= hi {
            break;
        }
        let Some(d) = done(mid) else {
            return Err(nan());
        };
        if d {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Ok(hi)
}

/// The smallest lattice point `k ∈ [kmin, kmax]` with `cdf(k) ≥ p` for a
/// non-decreasing `cdf`, from a guess `k0`.
fn discrete_ppf(
    op: &'static str,
    cdf: impl Fn(f64) -> f64,
    p: f64,
    k0: f64,
    kmin: f64,
    kmax: f64,
) -> Result<f64, SymplexError> {
    let done = |k: f64| {
        let c = cdf(k);
        if c.is_nan() { None } else { Some(c >= p) }
    };
    discrete_search(op, done, k0, kmin, kmax)
}

/// The smallest lattice point `k ∈ [kmin, kmax]` with `sf(k) ≤ q` for a
/// non-increasing `sf`, from a guess `k0` — scipy's `isf` for a discrete
/// distribution.
fn discrete_isf(
    op: &'static str,
    sf: impl Fn(f64) -> f64,
    q: f64,
    k0: f64,
    kmin: f64,
    kmax: f64,
) -> Result<f64, SymplexError> {
    let done = |k: f64| {
        let s = sf(k);
        if s.is_nan() { None } else { Some(s <= q) }
    };
    discrete_search(op, done, k0, kmin, kmax)
}

// ═══════════════════════════════════════════════════════════════════════════
// Families
// ═══════════════════════════════════════════════════════════════════════════

/// The standard normal distribution.
pub mod norm {
    use super::{SQRT_2PI, check_level, erfc, erfcinv};
    use crate::base::errors::SymplexError;
    use std::f64::consts::SQRT_2;

    /// `Φ(x) = ½ erfc(−x/√2)`.
    pub fn cdf(x: f64) -> f64 {
        0.5 * erfc(-x / SQRT_2)
    }

    /// `1 − Φ(x) = ½ erfc(x/√2)`, accurate in the upper tail.
    pub fn sf(x: f64) -> f64 {
        0.5 * erfc(x / SQRT_2)
    }

    /// `φ(x) = e^{−x²/2}/√(2π)`.
    pub fn pdf(x: f64) -> f64 {
        (-0.5 * x * x).exp() / SQRT_2PI
    }

    /// `Φ⁻¹(p)` for `0 < p < 1`.
    ///
    /// ```
    /// use symplex::stats::numdist::norm;
    ///
    /// // scipy: stats.norm.ppf(0.975) = 1.959963984540054
    /// assert!((norm::ppf(0.975)? - 1.959963984540054).abs() < 1e-14);
    /// assert!(norm::ppf(1.0).is_err());
    /// # Ok::<(), symplex::prelude::SymplexError>(())
    /// ```
    pub fn ppf(p: f64) -> Result<f64, SymplexError> {
        check_level("numdist::norm::ppf", "p", p)?;
        Ok(-SQRT_2 * erfcinv(2.0 * p))
    }

    /// `Φ⁻¹(1 − q)`: the upper-tail quantile, accurate for small `q`.
    pub fn isf(q: f64) -> Result<f64, SymplexError> {
        check_level("numdist::norm::isf", "q", q)?;
        Ok(SQRT_2 * erfcinv(2.0 * q))
    }
}

/// Student's t distribution with `df > 0` degrees of freedom.
pub mod t {
    use super::{
        Eval, Side, Tails, bratio, check_level, check_positive, lbeta, log_beta_pref, nearest_tail,
        nearest_tail_upper, norm, solve_increasing,
    };
    use crate::base::errors::SymplexError;

    /// Below this value of `r = ν/x²` the incomplete beta function of the
    /// tail is replaced by the leading term of its power series,
    /// `I_r(a, ½) = rᵃ/(a B(a, ½)) · (1 + O(r))`, evaluated from `ln r`:
    /// the relative error `O(r)` is far below `ε`, and neither `x²` (which
    /// overflows for `|x| > 1.34·10¹⁵⁴`) nor `r` (subnormal or `0`) is ever
    /// formed.  For `ν < 2` the tail is still a normal double out there:
    /// `P(T > 10²⁰⁰) = 3.2·10⁻¹⁰¹` for `ν = ½`.
    const POWER_LAW_MAX_R: f64 = 1e-290;

    /// The incomplete-beta arguments `(ν/(ν + x²), x²/(ν + x²))` of the
    /// tail at `|x|`, or `None` when `ν/x²` is below [`POWER_LAW_MAX_R`]
    /// (including an overflowing `x²`) and the power law takes over.
    fn beta_args(ax: f64, df: f64) -> Option<(f64, f64)> {
        let x2 = ax * ax;
        // `df / ∞ = 0`, so an overflowing `x²` also lands here.
        if df / x2 < POWER_LAW_MAX_R {
            return None;
        }
        Some((df / (df + x2), x2 / (df + x2)))
    }

    /// `ln(ν/x²)` from the logarithms, finite for every finite `x ≠ 0`.
    fn ln_r(ax: f64, df: f64) -> f64 {
        df.ln() - 2.0 * ax.ln()
    }

    /// `ln(x₀ᵃ y₀^½ / B(a, ½))` with `a = ν/2`, `x₀ = ν/(ν + x²)`: the
    /// logarithm of `|x|·f(x)`, `f` the density.  In the power-law region
    /// `y₀ = 1` and it is `a ln(ν/x²) − ln B(a, ½)`.
    fn log_pref(ax: f64, df: f64) -> f64 {
        let a = 0.5 * df;
        match beta_args(ax, df) {
            Some((x0, y0)) => log_beta_pref(a, 0.5, x0, y0),
            None => a * ln_r(ax, df) - lbeta(a, 0.5),
        }
    }

    /// Both tails: `P(T > |x|) = ½ I_{ν/(ν + x²)}(ν/2, ½)`, with the
    /// complementary argument `x²/(ν + x²)` formed directly, and the
    /// power law `½ (ν/x²)^{ν/2} / ((ν/2) B(ν/2, ½))` once `ν/x²` is
    /// below [`POWER_LAW_MAX_R`].
    fn tails(x: f64, df: f64) -> Tails {
        if x.is_nan() || df.is_nan() || df <= 0.0 {
            return Tails::NAN;
        }
        if df.is_infinite() {
            return Tails {
                lower: norm::cdf(x),
                upper: norm::sf(x),
            };
        }
        if x == 0.0 {
            return Tails {
                lower: 0.5,
                upper: 0.5,
            };
        }
        let ax = x.abs();
        let a = 0.5 * df;
        let (tail, body) = match beta_args(ax, df) {
            Some((x0, y0)) => {
                let i = bratio(a, 0.5, x0, y0);
                (0.5 * i.lower, 0.5 + 0.5 * i.upper)
            }
            None => {
                let tail = 0.5 * (a * ln_r(ax, df) - a.ln() - lbeta(a, 0.5)).exp();
                (tail, 1.0 - tail)
            }
        };
        if x > 0.0 {
            Tails {
                lower: body,
                upper: tail,
            }
        } else {
            Tails {
                lower: tail,
                upper: body,
            }
        }
    }

    /// `P(T ≤ x)`.  `scipy.stats.t.cdf(x, df)`.
    pub fn cdf(x: f64, df: f64) -> f64 {
        tails(x, df).lower
    }

    /// `P(T > x)`, accurate in the tail.  `scipy.stats.t.sf(x, df)`.
    pub fn sf(x: f64, df: f64) -> f64 {
        tails(x, df).upper
    }

    /// The density.  `scipy.stats.t.pdf(x, df)`.
    pub fn pdf(x: f64, df: f64) -> f64 {
        if x.is_nan() || df.is_nan() || df <= 0.0 {
            return f64::NAN;
        }
        if df.is_infinite() {
            return norm::pdf(x);
        }
        if x == 0.0 {
            return (-lbeta(0.5 * df, 0.5)).exp() / df.sqrt();
        }
        if x.is_infinite() {
            return 0.0;
        }
        let ax = x.abs();
        log_pref(ax, df).exp() / ax
    }

    /// `t > 0` with `P(T > t) = q` for `0 < q < ½`: Newton on
    /// `ln P(T > eᵘ)` in `u = ln t` from the Cornish–Fisher expansion
    /// around the normal quantile (or, in the far tail and for small `ν`,
    /// the power-law tail `P(T > t) ≈ ν^{ν/2 − 1} t^{−ν}/B(ν/2, ½)`).
    /// `+∞` when the quantile lies beyond the largest double, i.e. when
    /// `q < P(T > f64::MAX)` — `6.2·10⁻³²` for `ν = 0.1`, `2.4·10⁻¹⁵⁵` for
    /// `ν = ½` (the heavy tails of small `ν`).
    fn upper_quantile(op: &'static str, q: f64, df: f64) -> Result<f64, SymplexError> {
        if q < tails(f64::MAX, df).upper {
            return Ok(f64::INFINITY);
        }
        let a = 0.5 * df;
        let z = norm::isf(q)?;
        let t0 = z
            + (z * z * z + z) / (4.0 * df)
            + (5.0 * z.powi(5) + 16.0 * z.powi(3) + 3.0 * z) / (96.0 * df * df);
        let mut u0 = t0.ln();
        if q < 0.05 || df < 3.0 {
            let ln_tail = ((a - 1.0) * df.ln() - lbeta(a, 0.5) - q.ln()) / df;
            u0 = u0.max(ln_tail);
        }
        // Keep e^{u₀} finite; the bracket search takes it from there.
        let u0 = u0.min(700.0);
        let lnq = q.ln();
        let g = |u: f64| {
            let t = u.exp();
            let sf = tails(t, df).upper;
            let ln_sf = sf.ln();
            Eval {
                g: lnq - ln_sf,
                dg: (log_pref(t, df) - ln_sf).exp(),
            }
        };
        solve_increasing(op, g, u0, f64::NEG_INFINITY, f64::INFINITY).map(f64::exp)
    }

    fn quantile(op: &'static str, side: Side, df: f64) -> Result<f64, SymplexError> {
        if df.is_infinite() && df > 0.0 {
            return match side {
                Side::Lower(p) => norm::ppf(p),
                Side::Upper(q) => norm::isf(q),
            };
        }
        check_positive(op, "df", df)?;
        Ok(match side {
            Side::Lower(0.5) => 0.0,
            Side::Lower(pl) => -upper_quantile(op, pl, df)?,
            Side::Upper(q) => upper_quantile(op, q, df)?,
        })
    }

    /// `t` with `P(T ≤ t) = p`.  `scipy.stats.t.ppf(p, df)`.  Exact in the
    /// far tails of small `ν`, where scipy's own `sf` underflows
    /// (`t::ppf(1e-100, 0.5) = -1.0285e199`), and `∓∞` when the quantile
    /// lies beyond the largest double (`P(T > f64::MAX) = 2.4e-155` for
    /// `ν = ½`, so `t::ppf(1e-300, 0.5) = -∞`).
    ///
    /// ```
    /// use symplex::stats::numdist::t;
    ///
    /// // scipy: stats.t.ppf(0.3, 3.5) = -0.575338701772231
    /// assert!((t::ppf(0.3, 3.5)? + 0.575338701772231).abs() < 1e-13);
    /// # Ok::<(), symplex::prelude::SymplexError>(())
    /// ```
    pub fn ppf(p: f64, df: f64) -> Result<f64, SymplexError> {
        const OP: &str = "numdist::t::ppf";
        check_level(OP, "p", p)?;
        quantile(OP, nearest_tail(p), df)
    }

    /// `t` with `P(T > t) = q`.  `scipy.stats.t.isf(q, df)`; a small `q`
    /// keeps its relative accuracy.
    pub fn isf(q: f64, df: f64) -> Result<f64, SymplexError> {
        const OP: &str = "numdist::t::isf";
        check_level(OP, "q", q)?;
        quantile(OP, nearest_tail_upper(q), df)
    }
}

/// The gamma distribution with shape `k > 0` and scale `θ > 0` (the
/// crate's `Gamma(k, θ)`; the rate is `1/θ`).
pub mod gamma {
    use super::{
        Eval, Side, Tails, check_level, check_positive, gammainc_tails, lgamma, log_gamma_pref,
        nearest_tail, nearest_tail_upper, norm, solve_increasing,
    };
    use crate::base::errors::SymplexError;

    pub(super) fn tails(x: f64, shape: f64, scale: f64) -> Tails {
        if x.is_nan() || !(shape.is_finite() && shape > 0.0) || !(scale.is_finite() && scale > 0.0)
        {
            return Tails::NAN;
        }
        if x <= 0.0 {
            return Tails::ZERO;
        }
        gammainc_tails(shape, x / scale)
    }

    /// `P(X ≤ x) = P(k, x/θ)`.  `scipy.stats.gamma.cdf(x, k, scale=θ)`.
    pub fn cdf(x: f64, shape: f64, scale: f64) -> f64 {
        tails(x, shape, scale).lower
    }

    /// `P(X > x) = Q(k, x/θ)`, accurate in the tail.
    pub fn sf(x: f64, shape: f64, scale: f64) -> f64 {
        tails(x, shape, scale).upper
    }

    /// The quantile of the standard gamma (`θ = 1`): Newton on the
    /// logarithm of the nearer tail in `u = ln x`, from the Wilson–Hilferty
    /// start `k(1 − 1/(9k) + z/(3√k))³` (or `(p Γ(k + 1))^{1/k}` deep in
    /// the lower tail).
    pub(super) fn standard_quantile(
        op: &'static str,
        side: Side,
        shape: f64,
    ) -> Result<f64, SymplexError> {
        let (z, target, lower) = match side {
            Side::Lower(pl) => (norm::ppf(pl)?, pl, true),
            Side::Upper(q) => (norm::isf(q)?, q, false),
        };
        let mut x0 = shape * (1.0 - 1.0 / (9.0 * shape) + z / (3.0 * shape.sqrt())).powi(3);
        if !(x0.is_finite() && x0 > 0.0) {
            x0 = if lower {
                ((target.ln() + lgamma(shape + 1.0)) / shape).exp()
            } else {
                (-target.ln()).max(shape)
            };
        }
        let ln_target = target.ln();
        let g = |u: f64| {
            let x = u.exp();
            let tl = gammainc_tails(shape, x);
            let ln_tail = if lower { tl.lower.ln() } else { tl.upper.ln() };
            let gv = if lower {
                ln_tail - ln_target
            } else {
                ln_target - ln_tail
            };
            Eval {
                g: gv,
                dg: (log_gamma_pref(shape, x) - ln_tail).exp(),
            }
        };
        solve_increasing(op, g, x0.ln(), f64::NEG_INFINITY, f64::INFINITY).map(f64::exp)
    }

    fn quantile(op: &'static str, side: Side, shape: f64, scale: f64) -> Result<f64, SymplexError> {
        check_positive(op, "the shape", shape)?;
        check_positive(op, "the scale", scale)?;
        standard_quantile(op, side, shape).map(|x| scale * x)
    }

    /// `x` with `P(X ≤ x) = p`.  `scipy.stats.gamma.ppf(p, k, scale=θ)`.
    ///
    /// ```
    /// use symplex::stats::numdist::gamma;
    ///
    /// // scipy: stats.gamma.ppf(0.3, 3.5, scale=0.75) = 1.7517489183679027
    /// assert!((gamma::ppf(0.3, 3.5, 0.75)? - 1.7517489183679027).abs() < 1e-13);
    /// # Ok::<(), symplex::prelude::SymplexError>(())
    /// ```
    pub fn ppf(p: f64, shape: f64, scale: f64) -> Result<f64, SymplexError> {
        const OP: &str = "numdist::gamma::ppf";
        check_level(OP, "p", p)?;
        quantile(OP, nearest_tail(p), shape, scale)
    }

    /// `x` with `P(X > x) = q`.  `scipy.stats.gamma.isf(q, k, scale=θ)`.
    pub fn isf(q: f64, shape: f64, scale: f64) -> Result<f64, SymplexError> {
        const OP: &str = "numdist::gamma::isf";
        check_level(OP, "q", q)?;
        quantile(OP, nearest_tail_upper(q), shape, scale)
    }
}

/// The χ² distribution with `df > 0` degrees of freedom (`Gamma(df/2, 2)`).
pub mod chi2 {
    use super::{Side, check_level, check_positive, gamma, nearest_tail, nearest_tail_upper};
    use crate::base::errors::SymplexError;

    /// `P(X ≤ x)`.  `scipy.stats.chi2.cdf(x, df)`.
    pub fn cdf(x: f64, df: f64) -> f64 {
        gamma::tails(x, 0.5 * df, 2.0).lower
    }

    /// `P(X > x)`, accurate in the tail.  `scipy.stats.chi2.sf(x, df)`.
    pub fn sf(x: f64, df: f64) -> f64 {
        gamma::tails(x, 0.5 * df, 2.0).upper
    }

    fn quantile(op: &'static str, side: Side, df: f64) -> Result<f64, SymplexError> {
        check_positive(op, "df", df)?;
        gamma::standard_quantile(op, side, 0.5 * df).map(|x| 2.0 * x)
    }

    /// `x` with `P(X ≤ x) = p`.  `scipy.stats.chi2.ppf(p, df)`.
    ///
    /// ```
    /// use symplex::stats::numdist::chi2;
    ///
    /// // scipy: stats.chi2.ppf(0.3, 7) = 4.671330448981074
    /// assert!((chi2::ppf(0.3, 7.0)? - 4.671330448981074).abs() < 1e-13);
    /// # Ok::<(), symplex::prelude::SymplexError>(())
    /// ```
    pub fn ppf(p: f64, df: f64) -> Result<f64, SymplexError> {
        const OP: &str = "numdist::chi2::ppf";
        check_level(OP, "p", p)?;
        quantile(OP, nearest_tail(p), df)
    }

    /// `x` with `P(X > x) = q`.  `scipy.stats.chi2.isf(q, df)`.
    pub fn isf(q: f64, df: f64) -> Result<f64, SymplexError> {
        const OP: &str = "numdist::chi2::isf";
        check_level(OP, "q", q)?;
        quantile(OP, nearest_tail_upper(q), df)
    }
}

/// The beta distribution with shapes `α, β > 0`.
pub mod beta {
    use super::{
        Eval, Side, Tails, bratio, check_level, check_positive, lbeta, log_beta_pref, nearest_tail,
        nearest_tail_upper, norm, solve_increasing,
    };
    use crate::base::errors::SymplexError;

    fn tails(x: f64, alpha: f64, beta: f64) -> Tails {
        if x.is_nan() || !(alpha.is_finite() && alpha > 0.0) || !(beta.is_finite() && beta > 0.0) {
            return Tails::NAN;
        }
        if x <= 0.0 {
            return Tails::ZERO;
        }
        if x >= 1.0 {
            return Tails::ONE;
        }
        bratio(alpha, beta, x, 1.0 - x)
    }

    /// `P(X ≤ x) = I_x(α, β)`.  `scipy.stats.beta.cdf(x, a, b)`.
    pub fn cdf(x: f64, alpha: f64, beta: f64) -> f64 {
        tails(x, alpha, beta).lower
    }

    /// `P(X > x) = I_{1−x}(β, α)`, accurate in the tail.
    pub fn sf(x: f64, alpha: f64, beta: f64) -> f64 {
        tails(x, alpha, beta).upper
    }

    /// The logit of a starting point for the quantile: the normal
    /// approximation `μ + zσ` when it lands inside `(0, 1)`, else the
    /// power-law tail `I_x ≈ xᵅ/(α B(α, β))` (or its mirror image).
    pub(super) fn start_logit(side: Side, alpha: f64, beta: f64) -> Result<f64, SymplexError> {
        let s = alpha + beta;
        let mean = alpha / s;
        let sd = (alpha * beta / (s * s * (s + 1.0))).sqrt();
        let z = match side {
            Side::Lower(pl) => norm::ppf(pl)?,
            Side::Upper(q) => norm::isf(q)?,
        };
        let x0 = mean + z * sd;
        Ok(match side {
            Side::Lower(pl) => {
                let x = if x0 > 0.0 && x0 < 1.0 && pl >= 1e-3 {
                    x0
                } else {
                    ((pl.ln() + alpha.ln() + lbeta(alpha, beta)) / alpha)
                        .exp()
                        .min(0.5)
                };
                x.ln() - (-x).ln_1p()
            }
            Side::Upper(q) => {
                let y = if x0 > 0.0 && x0 < 1.0 && q >= 1e-3 {
                    1.0 - x0
                } else {
                    ((q.ln() + beta.ln() + lbeta(alpha, beta)) / beta)
                        .exp()
                        .min(0.5)
                };
                (-y).ln_1p() - y.ln()
            }
        })
    }

    /// Newton on the logarithm of the nearer tail in the logit
    /// `v = ln(x/(1 − x))`; returns the logit of the quantile.
    fn quantile_logit(
        op: &'static str,
        side: Side,
        alpha: f64,
        beta: f64,
    ) -> Result<f64, SymplexError> {
        let v0 = start_logit(side, alpha, beta)?;
        let (target, lower) = match side {
            Side::Lower(pl) => (pl, true),
            Side::Upper(q) => (q, false),
        };
        let ln_target = target.ln();
        let g = |v: f64| {
            let x = 1.0 / (1.0 + (-v).exp());
            let y = 1.0 / (1.0 + v.exp());
            let tl = bratio(alpha, beta, x, y);
            let ln_tail = if lower { tl.lower.ln() } else { tl.upper.ln() };
            let gv = if lower {
                ln_tail - ln_target
            } else {
                ln_target - ln_tail
            };
            Eval {
                g: gv,
                dg: (log_beta_pref(alpha, beta, x, y) - ln_tail).exp(),
            }
        };
        solve_increasing(op, g, v0, f64::NEG_INFINITY, f64::INFINITY)
    }

    fn quantile(op: &'static str, side: Side, alpha: f64, beta: f64) -> Result<f64, SymplexError> {
        check_positive(op, "α", alpha)?;
        check_positive(op, "β", beta)?;
        quantile_logit(op, side, alpha, beta).map(|v| 1.0 / (1.0 + (-v).exp()))
    }

    /// `x` with `P(X ≤ x) = p`.  `scipy.stats.beta.ppf(p, a, b)`.
    ///
    /// ```
    /// use symplex::stats::numdist::beta;
    ///
    /// // scipy: stats.beta.ppf(0.3, 3/7, 5/2) = 0.02083250526531868
    /// assert!((beta::ppf(0.3, 3.0 / 7.0, 2.5)? - 0.02083250526531868).abs() < 1e-14);
    /// # Ok::<(), symplex::prelude::SymplexError>(())
    /// ```
    pub fn ppf(p: f64, alpha: f64, beta: f64) -> Result<f64, SymplexError> {
        const OP: &str = "numdist::beta::ppf";
        check_level(OP, "p", p)?;
        quantile(OP, nearest_tail(p), alpha, beta)
    }

    /// `x` with `P(X > x) = q`.  `scipy.stats.beta.isf(q, a, b)`.
    pub fn isf(q: f64, alpha: f64, beta: f64) -> Result<f64, SymplexError> {
        const OP: &str = "numdist::beta::isf";
        check_level(OP, "q", q)?;
        quantile(OP, nearest_tail_upper(q), alpha, beta)
    }
}

/// The F distribution with `d₁, d₂ > 0` degrees of freedom.
pub mod f {
    use super::{
        Eval, Side, Tails, beta, bratio, check_level, check_positive, log_beta_pref, nearest_tail,
        nearest_tail_upper, solve_increasing,
    };
    use crate::base::errors::SymplexError;

    fn tails(x: f64, d1: f64, d2: f64) -> Tails {
        if x.is_nan() || !(d1.is_finite() && d1 > 0.0) || !(d2.is_finite() && d2 > 0.0) {
            return Tails::NAN;
        }
        if x <= 0.0 {
            return Tails::ZERO;
        }
        let num = d1 * x;
        let den = num + d2;
        if den.is_infinite() {
            return Tails::ONE;
        }
        bratio(0.5 * d1, 0.5 * d2, num / den, d2 / den)
    }

    /// `P(X ≤ x) = I_{d₁x/(d₁x + d₂)}(d₁/2, d₂/2)`.  `scipy.stats.f.cdf(x, d1, d2)`.
    pub fn cdf(x: f64, d1: f64, d2: f64) -> f64 {
        tails(x, d1, d2).lower
    }

    /// `P(X > x) = I_{d₂/(d₁x + d₂)}(d₂/2, d₁/2)`, accurate in the tail.
    pub fn sf(x: f64, d1: f64, d2: f64) -> f64 {
        tails(x, d1, d2).upper
    }

    /// Newton on the logarithm of the nearer tail in `u = ln x`, from the
    /// beta start transported by `x = d₂ b / (d₁ (1 − b))`.
    fn quantile(op: &'static str, side: Side, d1: f64, d2: f64) -> Result<f64, SymplexError> {
        check_positive(op, "d1", d1)?;
        check_positive(op, "d2", d2)?;
        let (a, b) = (0.5 * d1, 0.5 * d2);
        // logit(b) = ln(b/(1 − b)); x = (d₂/d₁) e^{logit}.
        let u0 = beta::start_logit(side, a, b)? + (d2 / d1).ln();
        let (target, lower) = match side {
            Side::Lower(pl) => (pl, true),
            Side::Upper(q) => (q, false),
        };
        let ln_target = target.ln();
        let g = |u: f64| {
            let x = u.exp();
            let num = d1 * x;
            let den = num + d2;
            let (xb, yb) = (num / den, d2 / den);
            let tl = bratio(a, b, xb, yb);
            let ln_tail = if lower { tl.lower.ln() } else { tl.upper.ln() };
            let gv = if lower {
                ln_tail - ln_target
            } else {
                ln_target - ln_tail
            };
            Eval {
                g: gv,
                dg: (log_beta_pref(a, b, xb, yb) - ln_tail).exp(),
            }
        };
        solve_increasing(op, g, u0, f64::NEG_INFINITY, f64::INFINITY).map(f64::exp)
    }

    /// `x` with `P(X ≤ x) = p`.  `scipy.stats.f.ppf(p, d1, d2)`.
    ///
    /// ```
    /// use symplex::stats::numdist::f;
    ///
    /// // scipy: stats.f.ppf(0.3, 5, 9) = 0.6031760598027508
    /// assert!((f::ppf(0.3, 5.0, 9.0)? - 0.6031760598027508).abs() < 1e-13);
    /// # Ok::<(), symplex::prelude::SymplexError>(())
    /// ```
    pub fn ppf(p: f64, d1: f64, d2: f64) -> Result<f64, SymplexError> {
        const OP: &str = "numdist::f::ppf";
        check_level(OP, "p", p)?;
        quantile(OP, nearest_tail(p), d1, d2)
    }

    /// `x` with `P(X > x) = q`.  `scipy.stats.f.isf(q, d1, d2)`.
    pub fn isf(q: f64, d1: f64, d2: f64) -> Result<f64, SymplexError> {
        const OP: &str = "numdist::f::isf";
        check_level(OP, "q", q)?;
        quantile(OP, nearest_tail_upper(q), d1, d2)
    }
}

/// The binomial distribution with `n` trials and success probability `p`.
pub mod binom {
    use super::{
        Tails, bratio, check_lattice_param, check_level, discrete_isf, discrete_ppf, invalid, norm,
    };
    use crate::base::errors::SymplexError;

    fn tails(k: f64, n: f64, p: f64) -> Tails {
        if k.is_nan() || !(n.is_finite() && n >= 0.0) || !(0.0..=1.0).contains(&p) {
            return Tails::NAN;
        }
        let kf = k.floor();
        if kf < 0.0 {
            return Tails::ZERO;
        }
        if kf >= n {
            return Tails::ONE;
        }
        // P(X ≤ k) = I_{1−p}(n − k, k + 1)
        bratio(n - kf, kf + 1.0, 1.0 - p, p)
    }

    /// `P(X ≤ ⌊k⌋)`.  `scipy.stats.binom.cdf(k, n, p)`.
    pub fn cdf(k: f64, n: f64, p: f64) -> f64 {
        tails(k, n, p).lower
    }

    /// `P(X > ⌊k⌋)`, accurate in the tail.  `scipy.stats.binom.sf(k, n, p)`.
    pub fn sf(k: f64, n: f64, p: f64) -> f64 {
        tails(k, n, p).upper
    }

    /// `n` a non-negative count at most `2⁵³`, `p ∈ [0, 1]`; returns `⌊n⌋`.
    fn check_params(op: &'static str, n: f64, p: f64) -> Result<f64, SymplexError> {
        if !(n.is_finite() && n >= 0.0) {
            return Err(invalid(
                op,
                format!("n must be a non-negative count, got {n}"),
            ));
        }
        check_lattice_param(op, "n", n)?;
        if !(0.0..=1.0).contains(&p) {
            return Err(invalid(op, format!("p must lie in [0, 1], got {p}")));
        }
        Ok(n.floor())
    }

    /// The smallest `k ∈ 0..=n` with `P(X ≤ k) ≥ q`.  `scipy.stats.binom.ppf(q, n, p)`.
    /// `n` must not exceed `2⁵³` (an [`InvalidArgument`](SymplexError::InvalidArgument)
    /// beyond; scipy returns `NaN`).
    ///
    /// ```
    /// use symplex::stats::numdist::binom;
    ///
    /// // scipy: stats.binom.ppf([0.3, 0.9], 9, 3/7) = (3, 6)
    /// assert_eq!(binom::ppf(0.3, 9.0, 3.0 / 7.0)?, 3.0);
    /// assert_eq!(binom::ppf(0.9, 9.0, 3.0 / 7.0)?, 6.0);
    /// # Ok::<(), symplex::prelude::SymplexError>(())
    /// ```
    pub fn ppf(q: f64, n: f64, p: f64) -> Result<f64, SymplexError> {
        const OP: &str = "numdist::binom::ppf";
        check_level(OP, "q", q)?;
        let n = check_params(OP, n, p)?;
        let k0 = n * p + norm::ppf(q)? * (n * p * (1.0 - p)).sqrt();
        discrete_ppf(OP, |k| cdf(k, n, p), q, k0, 0.0, n)
    }

    /// The smallest `k ∈ 0..=n` with `P(X > k) ≤ q`.  `scipy.stats.binom.isf(q, n, p)`;
    /// a small `q` is never rounded through `1 − q`.
    ///
    /// ```
    /// use symplex::stats::numdist::binom;
    ///
    /// // scipy: stats.binom.isf([0.3, 0.9], 9, 3/7) = (5, 2)
    /// assert_eq!(binom::isf(0.3, 9.0, 3.0 / 7.0)?, 5.0);
    /// assert_eq!(binom::isf(0.9, 9.0, 3.0 / 7.0)?, 2.0);
    /// # Ok::<(), symplex::prelude::SymplexError>(())
    /// ```
    pub fn isf(q: f64, n: f64, p: f64) -> Result<f64, SymplexError> {
        const OP: &str = "numdist::binom::isf";
        check_level(OP, "q", q)?;
        let n = check_params(OP, n, p)?;
        let k0 = n * p + norm::isf(q)? * (n * p * (1.0 - p)).sqrt();
        discrete_isf(OP, |k| sf(k, n, p), q, k0, 0.0, n)
    }
}

/// The Poisson distribution with rate `λ > 0`.
pub mod poisson {
    use super::{
        Tails, check_lattice_param, check_level, check_positive, discrete_isf, discrete_ppf,
        gammainc_tails, norm,
    };
    use crate::base::errors::SymplexError;

    fn tails(k: f64, rate: f64) -> Tails {
        if k.is_nan() || !(rate.is_finite() && rate > 0.0) {
            return Tails::NAN;
        }
        let kf = k.floor();
        if kf < 0.0 {
            return Tails::ZERO;
        }
        if kf == f64::INFINITY {
            return Tails::ONE;
        }
        // P(X ≤ k) = Q(k + 1, λ)
        gammainc_tails(kf + 1.0, rate).flipped()
    }

    /// `P(X ≤ ⌊k⌋)`.  `scipy.stats.poisson.cdf(k, mu)`.
    pub fn cdf(k: f64, rate: f64) -> f64 {
        tails(k, rate).lower
    }

    /// `P(X > ⌊k⌋)`, accurate in the tail.  `scipy.stats.poisson.sf(k, mu)`.
    pub fn sf(k: f64, rate: f64) -> f64 {
        tails(k, rate).upper
    }

    /// The smallest `k ≥ 0` with `P(X ≤ k) ≥ q`.  `scipy.stats.poisson.ppf(q, mu)`.
    /// The rate must not exceed `2⁵³` (an [`InvalidArgument`](SymplexError::InvalidArgument)
    /// beyond; scipy returns `NaN`).
    ///
    /// ```
    /// use symplex::stats::numdist::poisson;
    ///
    /// // scipy: poisson(7/3).cdf(0) = 0.0969… < 0.1, so ppf(0.1) = 1; ppf(0.95) = 5
    /// assert_eq!(poisson::ppf(0.1, 7.0 / 3.0)?, 1.0);
    /// assert_eq!(poisson::ppf(0.95, 7.0 / 3.0)?, 5.0);
    /// # Ok::<(), symplex::prelude::SymplexError>(())
    /// ```
    pub fn ppf(q: f64, rate: f64) -> Result<f64, SymplexError> {
        const OP: &str = "numdist::poisson::ppf";
        check_level(OP, "q", q)?;
        check_positive(OP, "the rate", rate)?;
        check_lattice_param(OP, "the rate", rate)?;
        let k0 = rate + norm::ppf(q)? * rate.sqrt();
        discrete_ppf(OP, |k| cdf(k, rate), q, k0, 0.0, f64::INFINITY)
    }

    /// The smallest `k ≥ 0` with `P(X > k) ≤ q`.  `scipy.stats.poisson.isf(q, mu)`;
    /// a small `q` is never rounded through `1 − q`.
    ///
    /// ```
    /// use symplex::stats::numdist::poisson;
    ///
    /// // scipy: stats.poisson.isf([0.1, 0.95], 7/3) = (4, 0)
    /// assert_eq!(poisson::isf(0.1, 7.0 / 3.0)?, 4.0);
    /// assert_eq!(poisson::isf(0.95, 7.0 / 3.0)?, 0.0);
    /// # Ok::<(), symplex::prelude::SymplexError>(())
    /// ```
    pub fn isf(q: f64, rate: f64) -> Result<f64, SymplexError> {
        const OP: &str = "numdist::poisson::isf";
        check_level(OP, "q", q)?;
        check_positive(OP, "the rate", rate)?;
        check_lattice_param(OP, "the rate", rate)?;
        let k0 = rate + norm::isf(q)? * rate.sqrt();
        discrete_isf(OP, |k| sf(k, rate), q, k0, 0.0, f64::INFINITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        ((a - b) / b).abs()
    }

    #[test]
    fn stirling_pieces() {
        // stirlerr(10) = lgamma(10) − Stirling: 1/(12·10) − 1/(360·1000) + … = 0.008330563433362871
        assert!((stirlerr(10.0) - 0.00833056343336287).abs() < 1e-16);
        // Continuity of the two branches at z = 10.
        assert!((stirlerr(10.0) - stirlerr(9.999_999_999)).abs() < 1e-12);
        // lbeta against ln Γ for moderate arguments.
        let direct = lgamma(3.5) + lgamma(12.0) - lgamma(15.5);
        assert!((lbeta(3.5, 12.0) - direct).abs() < 1e-13);
        // rlog1 against the definition away from zero, and its series near zero.
        assert!((rlog1(0.7) - (0.7 - 1.7f64.ln())).abs() < 1e-16);
        assert!((rlog1(1e-3) - (1e-3 - 1e-3f64.ln_1p())).abs() < 1e-16);
        // bd0(k, k) = 0 and bd0 against the definition away from k = m.
        assert_eq!(bd0(5.0, 5.0), 0.0);
        assert!((bd0(2.0, 5.0) - (2.0 * (2.0f64 / 5.0).ln() + 3.0)).abs() < 1e-15);
    }

    #[test]
    fn prefactors_match_definitions() {
        let (a, b, x): (f64, f64, f64) = (2.5, 4.0, 0.3);
        let direct = a * x.ln() + b * (1.0f64 - x).ln() - (lgamma(a) + lgamma(b) - lgamma(a + b));
        assert!((log_beta_pref(a, b, x, 1.0 - x) - direct).abs() < 1e-14);
        let direct = a * x.ln() - x - lgamma(a);
        assert!((log_gamma_pref(a, x) - direct).abs() < 1e-14);
    }

    #[test]
    fn grat_r_is_continuous_at_the_branch_point() {
        // Q(a, x)/r from the Taylor branch (x < 1.1) and the continued fraction agree.
        for a in [0.05, 0.5, 0.9] {
            let lo = grat_r(a, 1.099_999_999, log_gamma_pref(a, 1.099_999_999));
            let hi = grat_r(a, 1.1, log_gamma_pref(a, 1.1));
            assert!(rel(lo, hi) < 1e-9, "a = {a}: {lo} vs {hi}");
            // and both agree with Q/r from the generic route
            let q = gammainc_tails(a, 1.1).upper / log_gamma_pref(a, 1.1).exp();
            assert!(rel(hi, q) < 1e-13, "a = {a}: {hi} vs {q}");
        }
    }

    #[test]
    fn incomplete_beta_identities() {
        // I_x(1, 1) = x; I_x(a, b) + I_{1−x}(b, a) = 1; polynomial cases.
        assert!((betainc_regularized_f64(1.0, 1.0, 0.37) - 0.37).abs() < 1e-15);
        let t = bratio(3.0, 7.0, 0.2, 0.8);
        assert!((t.lower + t.upper - 1.0).abs() < 1e-15);
        assert!((betainc_regularized_f64(7.0, 3.0, 0.8) - t.upper).abs() < 1e-15);
        // I_{0.4}(2, 3) = 0.5248
        assert!((betainc_regularized_f64(2.0, 3.0, 0.4) - 0.5248).abs() < 1e-15);
        assert!(betainc_regularized_f64(-1.0, 3.0, 0.4).is_nan());
        assert!(betainc_regularized_f64(1.0, 3.0, 1.4).is_nan());
    }

    #[test]
    fn incomplete_gamma_identities() {
        // P(1, x) = 1 − e^{−x}; Q(½, x) = erfc(√x).
        assert!((gammainc_lower_regularized_f64(1.0, 0.7) - (1.0 - (-0.7f64).exp())).abs() < 1e-15);
        assert!(
            rel(
                gammainc_upper_regularized_f64(0.5, 3.0),
                erfc(3.0f64.sqrt())
            ) < 1e-14
        );
        assert_eq!(gammainc_lower_regularized_f64(2.0, 0.0), 0.0);
        assert!(gammainc_lower_regularized_f64(0.0, 1.0).is_nan());
    }

    #[test]
    fn quantiles_round_trip() -> Result<(), SymplexError> {
        for (p, df) in [
            (1e-10, 1.0),
            (1e-3, 2.5),
            (0.3, 30.0),
            (0.999, 1e4),
            (0.5, 7.0),
        ] {
            let x = t::ppf(p, df)?;
            assert!(
                rel(t::cdf(x, df), p) < 1e-13 || p == 0.5,
                "t: p = {p}, df = {df}"
            );
        }
        for (p, df) in [(1e-10, 1.0), (1e-3, 2.5), (0.3, 30.0), (0.999, 1e4)] {
            let x = chi2::ppf(p, df)?;
            assert!(rel(chi2::cdf(x, df), p) < 1e-13, "chi2: p = {p}, df = {df}");
        }
        for (p, a, b) in [
            (1e-10, 0.5, 0.5),
            (1e-3, 2.0, 30.0),
            (0.3, 5.0, 9.0),
            (0.999, 0.05, 3.0),
        ] {
            let x = beta::ppf(p, a, b)?;
            assert!(
                rel(beta::cdf(x, a, b), p) < 1e-13,
                "beta: p = {p}, a = {a}, b = {b}"
            );
            let x = f::ppf(p, 2.0 * a, 2.0 * b)?;
            assert!(rel(f::cdf(x, 2.0 * a, 2.0 * b), p) < 1e-13, "f: p = {p}");
        }
        assert!(t::ppf(0.0, 3.0).is_err());
        assert!(t::ppf(0.5, -3.0).is_err());
        assert!(chi2::ppf(1.0, 3.0).is_err());
        assert!(beta::ppf(0.5, 0.0, 1.0).is_err());
        Ok(())
    }

    #[test]
    fn discrete_quantiles() -> Result<(), SymplexError> {
        // binom(9, 3/7): ppf(0.3) = 3, ppf(0.9) = 6; poisson(7/3): ppf(0.1) = 1
        assert_eq!(binom::ppf(0.3, 9.0, 3.0 / 7.0)?, 3.0);
        assert_eq!(binom::ppf(0.9, 9.0, 3.0 / 7.0)?, 6.0);
        assert_eq!(poisson::ppf(0.1, 7.0 / 3.0)?, 1.0);
        assert_eq!(poisson::ppf(0.3, 7.0 / 3.0)?, 1.0);
        assert_eq!(poisson::ppf(0.95, 7.0 / 3.0)?, 5.0);
        // Far out on a big lattice the doubling search still lands.
        let k = poisson::ppf(1e-10, 1e6)?;
        assert!(poisson::cdf(k, 1e6) >= 1e-10 && poisson::cdf(k - 1.0, 1e6) < 1e-10);
        assert_eq!(binom::ppf(0.5, 0.0, 0.3)?, 0.0);
        Ok(())
    }
}

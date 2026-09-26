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
//! | [`betainc_regularized_f64`] | the dispatch of TOMS 708 (DiDonato & Morris 1992): power series `bpser`, the recurrence `bup`, the asymptotic expansion `bgrat` for a large first and small second parameter, the continued fraction `bfrac`, and the two-large-parameter expansion `basym` (both shapes above 100 and `x` within `3 %` of the mean, in `λ = a − (a + b)x`); each tail computed directly where it is the smaller one.  A shape below 1 is kept as a factor, never as `ln` of it in an exponent: `1/(a B(a, b))` from `ln Γ(1 + t)` (through `gam1`, the Taylor series of `1/Γ(1 + t)`, DLMF 5.7.1) and `ln Γ(a + b) − ln Γ(a)` without cancellation (the `algdiv` of DiDonato & Morris).  The distance from the mean, `(a + b)x − a`, is formed without rounding `a + b` or the product (two-sum and a fused multiply-add) | ≈ 1e-15 typical, ≤ 3e-14 for a tail ≥ 1e-10 (shapes from `1e-300` to `1e16`: largest measured 2e-14); in a smaller tail `P` the rounding of its logarithm, up to ≈ 8·ε·\|ln P\| (1e-13 at 1e-100, 3.5e-13 at 1e-277).  Measured on 4600 random and extreme cases against mpmath at ≥ 35 digits (positive-term series where its hypergeometric route fails).  `a + b` must be a double (`NaN` beyond) |
//! | [`gammainc_lower_regularized_f64`], [`gammainc_upper_regularized_f64`] | a dispatch on `a`: below 1, the Taylor route of DiDonato & Morris (1986, §2) for `x < 1.1` (each tail formed where it is the smaller one) and Legendre's continued fraction beyond, with the prefactor `a(1 + gam1(a))·e^{a ln x − x}`; below `GAMMA_TEMME_MIN_A = 1e6`, or for `x` beyond both 40 standard deviations and `a/2` from `a`, the power series for `x < a + 1` and Lentz's continued fraction otherwise (rescaled by `x` beyond `10¹⁰⁰`), with the prefactor `xᵃe⁻ˣ/Γ(a)` directly for `a ≤ 10` and through Loader's `bd0` and the Stirling remainder above (no cancellation for large `a`); from `a = 1e6` on and within that window, Temme's uniform asymptotic expansion (DLMF 8.12) truncated after `c₁/a`, its `½ erfc(z)` and correction term sharing one exponential so the tail stays correct into the subnormal range | ≈ 1e-15 for `a ≤ 1e4`, also for `a` down to `1e-300`; rising as `√a·ε` to ≈ 3e-14 just below `a = 1e6` (the rounding of the `O(√a)` terms of the series and fraction), ≈ 6e-15 in Temme's range (to `a = 1e15`), all for a tail ≥ 1e-10; in a smaller tail `P` the rounding of its logarithm, up to ≈ 5·ε·\|ln P\| (2.3e-13 at 1e-241) |
//! | `norm` | Cody's `erfc`, continued by `erfcx(x)·e^{−x²}` where Cody's approximation stops (`x > 26.5`, the subnormal range: `Φ(x)` is non-zero down to `x ≈ −38.5`) / the crate's `erfcinv` | ≈ 1e-16 (the precision of a subnormal result in the subnormal range) |
//! | `t` | `½ I_{ν/(ν + x²)}(ν/2, ½)`; once `ν/x² < 1e-290` (so for `|x|` beyond `≈ 1e145`, where `x²` would soon overflow) the power law `½ (ν/x²)^{ν/2} / ((ν/2) B(ν/2, ½))`, the power as `ν^{ν/2}·|x|^{−ν}` while that is a normal double and from `ln(ν/x²)` below, so the tail is right up to `|x| = f64::MAX` (`P(T > 1e200) = 3.2e-101` for `ν = ½`) | ≈ 1e-15; ≈ 1e-13 in the power-law region where the tail is subnormal (`|ν/2 · ln(ν/x²)| · ε`) |
//! | `f` | `I_z(d₁/2, d₂/2)` with `z = d₁x/(d₁x + d₂)`, `y = 1 − z` formed directly; where `z` or `y` is below the normal range (`d₁x` under- or overflows, or a shape is extreme) the power series of the small tail from `ln z` (`ln y`) and `d₂z` (`d₁y`), never from the subnormal argument, and the other tail as `−expm1` of its logarithm | as the incomplete beta |
//! | quantiles | safeguarded Newton on the logarithm of the relevant tail, in a log or logit variable, from a Cornish–Fisher / Wilson–Hilferty start; the root's last Newton correction applied in `x` itself (the log variable cannot resolve it); the gamma for shapes from `1e10` on directly in `x` from the Cornish–Fisher expansion `k + z√k + (z² − 1)/3 + (z³ − 7z)/(36√k)`; where one float step moves the tail by more than `1e-9` (shapes ≳ 1e20), the float next to the crossing of the level | ≈ 1e-15 (the CDF's accuracy); at a subnormal level ≈ `ε·|ln p|·∂ln x/∂ln p` (the logarithm of the level is rounded) |
//!
//! Every tail is carried with its logarithm (a mantissa and an exponent
//! until the end), so a tail below the smallest double still has one: the
//! quantile iterations and the discrete searches compare in logarithms
//! where a value would be subnormal.
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
//! * A quantile outside the positive floats of the support is the float
//!   the definition `inf{x : F(x) ≥ p}` gives over the floats: the smallest
//!   positive double (`5e-324`) when that already meets the level (a tiny
//!   shape piles its mass against 0), `+∞` (`1` for the beta) when even the
//!   largest double falls short.  The discrete searches compare the smaller
//!   tail with its level (`sf(k) ≤ 1 − p` for `p > ½`).
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

/// `m·eˢ`: a non-negative quantity held as a mantissa and a separate
/// natural exponent.  A tail probability is carried this way until the
/// end, so that
///
/// * a tail below the smallest double keeps its logarithm `ln m + s` —
///   the quantile solvers work on `ln P`, and at a subnormal level the
///   rounded `P` has only a few bits (0.26 returned `t::ppf(4.4e-323,
///   1.12e8)` with a relative error of `1.2·10⁻⁵`);
/// * a small factor — `b` in `I_x(a, b) ≈ b·(…)` for a tiny shape —
///   multiplies the mantissa instead of entering the exponent as `ln b`,
///   where its rounding would be amplified `|ln b|` times.
#[derive(Clone, Copy, Debug)]
struct Scaled {
    m: f64,
    s: f64,
}

impl Scaled {
    const ZERO: Scaled = Scaled { m: 0.0, s: 0.0 };

    /// A plain value.
    fn of(v: f64) -> Scaled {
        Scaled { m: v, s: 0.0 }
    }

    /// `eˢ`.
    fn exp(s: f64) -> Scaled {
        Scaled { m: 1.0, s }
    }

    /// The value, rounded once: `m·eˢ` while `eˢ` is a normal double (so an
    /// in-range value is exactly what the direct product gives), otherwise
    /// `e^{s + ln m}`, which neither under- nor overflows on the way.
    fn value(self) -> f64 {
        if (-708.0..=709.0).contains(&self.s) || self.m <= 0.0 || self.m.is_nan() {
            self.m * self.s.exp()
        } else {
            (self.s + self.m.ln()).exp()
        }
    }

    /// `ln(m·eˢ)`, finite whenever `m > 0`.
    fn ln(self) -> f64 {
        self.m.ln() + self.s
    }

    /// The product of two scaled quantities.
    fn mul(self, o: Scaled) -> Scaled {
        Scaled {
            m: self.m,
            s: self.s + o.s,
        }
        .times(o.m)
    }

    /// `f·m·eˢ`, the mantissa kept a normal double: a product that would be
    /// subnormal (or overflow) first moves a factor `2⁶⁰⁰` into the exponent
    /// — at the price of one rounding of `s`, `½ ulp(s)`, where the product
    /// alone would have lost its digits.  Only then: `s − 600 ln 2` rounds at
    /// the magnitude of `s` (a few `10⁻¹⁴` relative near `|s| = 400`).  0.26
    /// multiplied `(b/a)·yᵃ` into a subnormal mantissa and lost the tail:
    /// `f::sf(3.6e123, 3.3e-121, 1956)` was `0`, truly `6.6e-324`.
    fn times(self, f: f64) -> Scaled {
        /// `2⁶⁰⁰`.
        const BIG: f64 = 4.149_515_568_880_993e180;
        let (mut m, mut s) = (self.m, self.s);
        for _ in 0..4 {
            let p = (m * f).abs();
            if p < f64::MIN_POSITIVE && m != 0.0 && f != 0.0 {
                m *= BIG;
                s -= 600.0 * std::f64::consts::LN_2;
            } else if p.is_infinite() && m.is_finite() && f.is_finite() {
                m /= BIG;
                s += 600.0 * std::f64::consts::LN_2;
            } else {
                break;
            }
        }
        Scaled { m: m * f, s }
    }

    /// `m·e^{s + t}`.
    fn times_exp(self, t: f64) -> Scaled {
        Scaled {
            m: self.m,
            s: self.s + t,
        }
    }

    /// The sum of two non-negative quantities.  While the larger is a normal
    /// double the values are added (the smaller one's lost bits are below
    /// its rounding); only a sum below that is combined in logarithms.
    fn plus(self, other: Scaled) -> Scaled {
        if self.m.is_nan() || other.m.is_nan() {
            return Scaled::of(f64::NAN);
        }
        if self.m == 0.0 {
            return other;
        }
        if other.m == 0.0 {
            return self;
        }
        let (u, v) = (self.value(), other.value());
        if u.max(v) >= 1e-290 {
            return Scaled::of(u + v);
        }
        let s = self.s.max(other.s);
        Scaled {
            m: self.m * (self.s - s).exp() + other.m * (other.s - s).exp(),
            s,
        }
    }
}

/// Both tails of a distribution function at one point, each computed
/// directly where it is the smaller one, with their logarithms (`ln_lower`,
/// `ln_upper`) — finite, and accurate, where the tail itself is subnormal
/// or below the smallest double.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Tails {
    lower: f64,
    upper: f64,
    ln_lower: f64,
    ln_upper: f64,
}

impl Tails {
    const NAN: Tails = Tails {
        lower: f64::NAN,
        upper: f64::NAN,
        ln_lower: f64::NAN,
        ln_upper: f64::NAN,
    };
    const ZERO: Tails = Tails {
        lower: 0.0,
        upper: 1.0,
        ln_lower: f64::NEG_INFINITY,
        ln_upper: 0.0,
    };
    const ONE: Tails = Tails {
        lower: 1.0,
        upper: 0.0,
        ln_lower: 0.0,
        ln_upper: f64::NEG_INFINITY,
    };

    /// Two tails given as plain values (their logarithms taken from them).
    fn of(lower: f64, upper: f64) -> Tails {
        Tails {
            lower,
            upper,
            ln_lower: lower.ln(),
            ln_upper: upper.ln(),
        }
    }

    /// The lower tail computed directly as `w`, the upper one as `1 − w`.
    fn from_lower(w: Scaled) -> Tails {
        if w.m.is_nan() || w.s.is_nan() {
            return Tails::NAN;
        }
        let lower = w.value().clamp(0.0, 1.0);
        Tails {
            lower,
            upper: 1.0 - lower,
            ln_lower: if w.m > 0.0 {
                w.ln().min(0.0)
            } else {
                lower.ln()
            },
            ln_upper: (-lower).ln_1p(),
        }
    }

    /// The upper tail computed directly as `w`, the lower one as `1 − w`.
    fn from_upper(w: Scaled) -> Tails {
        Tails::from_lower(w).flipped()
    }

    fn flipped(self) -> Tails {
        Tails {
            lower: self.upper,
            upper: self.lower,
            ln_lower: self.ln_upper,
            ln_upper: self.ln_lower,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Stirling remainder, log-beta and the prefactors x^a y^b / B(a,b), x^a e^{-x} / Γ(a)
// ═══════════════════════════════════════════════════════════════════════════

/// `ln Γ(z) − [(z − ½) ln z − z + ln √(2π)]`, the Stirling remainder
/// (`stirlerr` in C. Loader, *Fast and accurate computation of binomial
/// probabilities*, 2000): the series `Σ B₂ₖ/(2k(2k − 1) z^(2k−1))` (DLMF
/// 5.11.1) through `z⁻¹³` for `z ≥ 10` (truncation below `3e-17`), the
/// definition below.
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
/// `ln Γ(a) + ln Γ(b) − ln Γ(a + b)` for large arguments: each large
/// `ln Γ` is written as Stirling's formula plus its remainder (DLMF 5.11.1)
/// and the leading terms are combined analytically.
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

/// `k ln(k/m) + m − k` (`bd0` in Loader 2000) given `diff = k − m` exactly and
/// `ln_ratio = ln(k/m)`: the series in `v = diff/(k + m)`,
/// `diff·v + 2k Σ_{j≥1} v^{2j+1}/(2j + 1)`, for `|v| < ½` (`k/m` between ⅓
/// and 3), the direct formula otherwise.
///
/// The series has no cancellation there (its correction terms are at most
/// `|v|/3` of `diff·v` against it, and of one sign) and needs 27 terms at
/// `|v| = ½`.  Loader switches at `|v| = 0.1`, and just beyond that the
/// direct formula cancels a factor `1/v`: its error `|k ln(k/m)|·ε` was
/// `10⁻¹²` of `e^{−bd0}` in a far tail — `Q(24128.47, 30411.25) = 2.7·10⁻³⁰⁶`
/// (the exponent 698.6 from `−5584 + 6283`) came out `10⁻¹²` off.  From
/// `k/m = 3` on the direct formula loses less than two bits.
fn bd0_with(k: f64, m: f64, diff: f64, ln_ratio: f64) -> f64 {
    // `0.5k + 0.5m` and the halved quotient: `k + m` overflows near
    // f64::MAX, and 0.26 then took the series branch for any pair
    // (`bd0(1e300, 1.8e308) = 0`, and `gamma::sf(1.8e308, 1e300)` came out
    // `2.2e-159` instead of 0).
    if diff.abs() < 0.5 * k + 0.5 * m {
        let v = (0.5 * diff) / (0.5 * k + 0.5 * m);
        let v2 = v * v;
        let mut s = diff * v;
        let mut ej = 2.0 * (k * v);
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
    // `k/m` overflows for a subnormal `m` (0.005 / 5e-324 > f64::MAX) and
    // underflows the other way; the logarithm of the ratio does neither.
    let ratio = k / m;
    let ln_ratio = if ratio.is_finite() && ratio > 0.0 {
        ratio.ln()
    } else {
        k.ln() - m.ln()
    };
    bd0_with(k, m, k - m, ln_ratio)
}

/// `(a + b)·s − c` without rounding `a + b` or the product first: `a + b`
/// as the exact pair `n + n_lo` (Knuth's two-sum), `n·s` as the exact pair
/// `p + p_lo` (a fused multiply-add), and then `(p − c) + (p_lo + n_lo·s)`,
/// whose leading difference is exact near the mean (Sterbenz).  This is the
/// distance `λ` of the argument from the mean in the units of the shapes;
/// formed as `(a + b)·x − a` in plain doubles it carries an absolute error
/// `ε·a`, which the tail's exponent (`≈ λ²/2a`) turns into a relative error
/// `ε·|λ|` — `3·10⁻⁸` for `I_{0.107}(3.7·10¹⁴, 3.1·10¹⁵)` 31 standard
/// deviations out (0.28), `10⁻¹¹` at shapes near `3·10⁷`.
fn mean_offset(a: f64, b: f64, s: f64, c: f64) -> f64 {
    let n = a + b;
    let bb = n - a;
    let n_lo = (a - (n - bb)) + (b - bb);
    let p = n * s;
    let p_lo = n.mul_add(s, -p);
    (p - c) + (p_lo + n_lo * s)
}

/// `ln(xᵃ yᵇ / B(a, b))` for `x + y = 1`, the smaller of `x`, `y` taken
/// as exact (the other is `1 −` it, never rounded): with `n = a + b`,
/// `−bd0(a, nx) − bd0(b, ny) − [stirlerr(a) + stirlerr(b) − stirlerr(n)]
/// + ½ ln(ab/(2πn))`, in which the two `bd0` differences `nx − a` and
/// `ny − b` are the same number up to sign and are formed once, without
/// rounding ([`mean_offset`]).  `−∞` when `x` or `y` is `0`.
fn log_beta_pref(a: f64, b: f64, x: f64, y: f64) -> f64 {
    if x <= 0.0 || y <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let n = a + b;
    let (s, small, large) = if y <= x { (y, b, a) } else { (x, a, b) };
    let ln_l = (-s).ln_1p();
    let ns = n * s;
    let d = mean_offset(a, b, s, small);
    let t_small = bd0_with(small, ns, -d, (small / n).ln() - s.ln());
    let nl = n - ns;
    let t_large = bd0_with(large, nl, d, (large / n).ln() - ln_l);
    // ln(ab/n): the product underflows for two tiny shapes (1e-111·5e-283),
    // where 0.26 returned −∞ and dropped the whole term.
    let ab_n = a * (b / n);
    let ln_ab_n = if ab_n >= f64::MIN_POSITIVE {
        ab_n.ln()
    } else {
        a.ln() + b.ln() - n.ln()
    };
    -t_small - t_large - stirlerr(a) - stirlerr(b) + stirlerr(n) + 0.5 * ln_ab_n - LN_SQRT_2PI
}

/// `xᵃ yᵇ / B(a, b)` for `x + y = 1`, from whichever form rounds less.
/// Loader's [`log_beta_pref`] is made for `x` near the mean `a/(a + b)`,
/// where the factors are astronomically large and small and their product
/// is not; its error is about `ε` times its own size, so far from the mean
/// (`x = 1.6·10⁻⁵²` for shapes 6.4 and 1.5e5: an exponent of `−698`) it
/// loses `10⁻¹³`.  There the direct product `a·inv_a_beta(a, b)·xᵃyᵇ`
/// ([`power_pair`]) carries only the exponent of `1/(aB)` (here 69) and
/// that of the larger argument's power.
fn beta_pref(a: f64, b: f64, x: f64, y: f64) -> Scaled {
    let lbp = log_beta_pref(a, b, x, y);
    let iab = inv_a_beta(a, b);
    let big_power = if x <= y {
        b * (-x).ln_1p()
    } else {
        a * (-y).ln_1p()
    };
    if lbp.is_finite() && iab.s.abs() + big_power.abs() + 1.0 < 0.5 * lbp.abs() {
        let direct = power_pair(iab.times(a), x, a, y, b);
        if direct.m.is_finite() && direct.m > 0.0 && direct.s.is_finite() {
            return direct;
        }
    }
    Scaled::exp(lbp)
}

/// `ln(xᵃ e⁻ˣ / Γ(a))` for `a, x > 0`: `−stirlerr(a) − bd0(a, x) + ½ ln(a/(2π))`.
fn log_gamma_pref(a: f64, x: f64) -> f64 {
    -stirlerr(a) - bd0(a, x) + 0.5 * a.ln() - LN_SQRT_2PI
}

/// `xᵃ e⁻ˣ / Γ(a)` for `a ≥ 1`: for `a ≤ 10` as the product itself while
/// `xᵃ` (through `powf`) and `e⁻ˣ` are normal doubles, otherwise
/// Loader's form [`log_gamma_pref`].  Far from the mean — `x = 10⁻¹⁷⁴`,
/// `a = 1.2` — `bd0 ≈ a ln(a/x)` is several hundred, and its rounding in
/// the exponent cost `6·10⁻¹⁴`; Loader's form is for `x ≈ a`, which the
/// product cannot do for large `a`.
fn gamma_pref(a: f64, x: f64) -> Scaled {
    if a <= 10.0 {
        let (xa, ex) = (x.powf(a), (-x).exp());
        if xa.is_finite() && xa >= f64::MIN_POSITIVE && ex >= f64::MIN_POSITIVE {
            return Scaled {
                m: xa * ex,
                s: -lgamma(a),
            };
        }
    }
    Scaled::exp(log_gamma_pref(a, x))
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
/// (`x ≥ a + 1`, where it converges fast).  Beyond `x = 10¹⁰⁰` the
/// fraction is evaluated for `x·Q/r` (every partial numerator divided by
/// `x²`, every denominator by `x`), which is of order 1: the unscaled
/// value `≈ 1/x` is subnormal near `f64::MAX`, where 0.26's iteration
/// never met its tolerance and ran its ten million steps (48 ms).
fn gamma_q_cf_over_r(a: f64, x: f64) -> Scaled {
    if x > 1e100 {
        return Scaled {
            m: gamma_q_cf_scaled_by(a, x, x),
            s: -x.ln(),
        };
    }
    Scaled::of(gamma_q_cf_scaled_by(a, x, 1.0))
}

/// `s·Q(a, x)/r`: Legendre's fraction with its denominators divided by `s`
/// and its partial numerators by `s²`.
fn gamma_q_cf_scaled_by(a: f64, x: f64, s: f64) -> f64 {
    const TINY: f64 = 1e-300;
    // Divisions, not a reciprocal: 1/s is subnormal for s near f64::MAX,
    // and dividing by s = 1 leaves the unscaled fraction bit for bit.
    let step = 2.0 / s;
    let mut b = (x + 1.0 - a) / s;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    let mut i = 1.0;
    for _ in 0..GAMMA_MAX_ITER {
        let an = -i * (i - a) / s / s;
        b += step;
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
    // window guarantees; x/a − 1 would not be.
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
    // e^{−z²} split as in `exp_neg_square`: the exact square xₛ² is the
    // exponent, e^{−δ} joins the mantissa.
    let az = z.abs();
    let xs = (az * 16.0).floor() / 16.0;
    let del = (az - xs) * (az + xs);
    let small = Scaled {
        m: (mantissa * (-del).exp()).max(0.0),
        s: -(xs * xs),
    };
    if z >= 0.0 {
        Tails::from_upper(small)
    } else {
        Tails::from_lower(small)
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
        // is both faster and more accurate than the series / fraction; so
        // it is for `|x − a| ≤ a/2` (`|η| ≤ 0.6`, inside the `|η| ≤ 1.5` of
        // its error bound), where the series needs `a/|x − a|` terms — up to
        // its ten-million cap once `40√a ≪ a` (0.26: 9 ms and a wrong
        // tail for `gamma::cdf(0.9999999999999998e300, 1e300)`).
        let sd = a.sqrt();
        if (x - a).abs() < 40.0 * sd || (x - a).abs() <= 0.5 * a {
            return gammainc_tails_temme(a, x);
        }
    }
    if a < 1.0 {
        return gammainc_tails_small_shape(a, x);
    }
    let r = gamma_pref(a, x);
    if x < a + 1.0 {
        Tails::from_lower(r.times(gamma_p_series_over_r(a, x)))
    } else {
        Tails::from_upper(r.mul(gamma_q_cf_over_r(a, x)))
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
fn bpser(a: f64, b: f64, x: f64) -> Scaled {
    if x == 0.0 {
        return Scaled::ZERO;
    }
    let lead = times_power(inv_a_beta(a, b), x, a);
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
    lead.times(1.0 + a * sum)
}

/// `f·xᵃyᵇ` for `x + y = 1` with the smaller of `x`, `y` exact (the other
/// is `1 −` it, rounded): the larger one's power as `e^{b ln(1 − s)}` from
/// the exact `s`, since `powf` of the rounded complement would carry its
/// rounding times the exponent.
fn power_pair(f: Scaled, x: f64, a: f64, y: f64, b: f64) -> Scaled {
    if x <= y {
        times_power(f, x, a).times_exp(b * (-x).ln_1p())
    } else {
        times_power(f, y, b).times_exp(a * (-y).ln_1p())
    }
}

/// `f·xᵖ` for `0 < x ≤ 1`, `p > 0`: through `powf` (one rounding) while
/// `xᵖ` is a normal double, as `p ln x` in the exponent below.
fn times_power(f: Scaled, x: f64, p: f64) -> Scaled {
    let xp = x.powf(p);
    if xp >= f64::MIN_POSITIVE {
        return f.times(xp);
    }
    // xᵖ underflows, but f·xᵖ may be a normal double: fold eˢ into the
    // base, (e^{s/p}·x)ᵖ, whose rounding is amplified p-fold rather than
    // |p ln x|-fold (`I_{9.2e-65}(4.91, 517)` lost 1.4·10⁻¹³ through the
    // exponent ≈ −723).
    let base = x * (f.s / p).exp();
    let v = f.m * base.powf(p);
    if base.is_finite() && v.is_finite() && v >= f64::MIN_POSITIVE {
        return Scaled::of(v);
    }
    f.times_exp(p * x.ln())
}

/// `I_x(a, b) − I_x(a + n, b)` for an integer `n ≥ 1`: the sum of the `n`
/// positive terms `x^{a+j} yᵇ / ((a + j) B(a + j, b))`, accumulated
/// relative to the largest so far so that no term over- or underflows.
fn bup(a: f64, b: f64, x: f64, y: f64, n: usize) -> Scaled {
    // The first term xᵃyᵇ/(a B(a, b)): Loader's form, except for a tiny
    // shape, where its stirlerr cancels (|ln a|·ε) and `inv_a_beta` keeps
    // the shape as a factor.
    let t0 = if a.min(b) < 1e-3 {
        power_pair(inv_a_beta(a, b), x, a, y, b)
    } else {
        beta_pref(a, b, x, y).times(1.0 / a)
    };
    if t0.m == 0.0 || t0.s == f64::NEG_INFINITY {
        return Scaled::ZERO;
    }
    let apb = a + b;
    let ap1 = a + 1.0;
    let lnx = x.ln();
    let log_t0 = 0.0;
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
    Scaled {
        m: t0.m * sum,
        s: t0.s + max,
    }
}

/// `1/Γ(1 + a) − 1` for `−½ ≤ a ≤ ½`, and through `1/Γ(1 + a) =
/// (1 + gam1(a − 1))/a` on `(½, 1½]`: the Taylor series
/// `1/Γ(1 + t) = 1 + Σ_{k≥1} cₖ tᵏ` (DLMF 5.7.1, whose coefficients are these
/// shifted by one) through `t²²`, where the next term is below `10⁻²³` for
/// `|t| ≤ ½`.  Relative error below `1.2·10⁻¹⁵` on `(0, 1½]` against
/// mpmath's `rgamma(1 + a) − 1` at 700 digits.  0.26 formed `lgamma(1 + a)`,
/// which loses `a` in the rounding of `1 + a` (all of it for `a < ε`), so
/// the incomplete gamma and beta functions of a shape below `10⁻³` lost
/// digits — up to all of them.
fn gam1(a: f64) -> f64 {
    // mpmath 1.3: mp.dps = 40; taylor(lambda z: rgamma(1 + z), 0, 22)[1:]
    const C: [f64; 22] = [
        0.577_215_664_901_532_9,
        -0.655_878_071_520_253_9,
        -0.042_002_635_034_095_24,
        0.166_538_611_382_291_48,
        -0.042_197_734_555_544_33,
        -0.009_621_971_527_876_973,
        0.007_218_943_246_663_1,
        -0.001_165_167_591_859_065_2,
        -0.000_215_241_674_114_950_98,
        0.000_128_050_282_388_116_2,
        -2.013_485_478_078_824e-5,
        -1.250_493_482_142_670_6e-6,
        1.133_027_231_981_696e-6,
        -2.056_338_416_977_607e-7,
        6.116_095_104_481_416e-9,
        5.002_007_644_469_223e-9,
        -1.181_274_570_487_02e-9,
        1.043_426_711_691_100_5e-10,
        7.782_263_439_905_071e-12,
        -3.696_805_618_642_206e-12,
        5.100_370_287_454_476e-13,
        -2.058_326_053_566_506_6e-14,
    ];
    let series = |t: f64| C.iter().rev().fold(0.0, |acc, &c| acc * t + c) * t;
    if a <= 0.5 {
        series(a)
    } else {
        // 1/Γ(1 + a) − 1 = (1 + S(t))/a − 1 = (S(t) − t)/a, t = a − 1 exact.
        let t = a - 1.0;
        (series(t) - t) / a
    }
}

/// `ln Γ(1 + a)` for `0 ≤ a ≤ 1`, from [`gam1`]: no rounding of `1 + a`.
fn lgamma1p(a: f64) -> f64 {
    -gam1(a).ln_1p()
}

/// `ln Γ(a + b) − ln Γ(a)` for `a, b > 0`, without the cancellation of the
/// two logarithms when `b` is small beside `a` (the `algdiv` of DiDonato &
/// Morris 1992): for `a ≥ 10` the Stirling forms with the leading terms
/// combined analytically,
/// `(a − ½) ln(1 + b/a) + b ln(a + b) − b + stirlerr(a + b) − stirlerr(a)`;
/// below, `a` is first shifted past 10 by the recurrence
/// `Γ(a + b)/Γ(a) = [Γ(a + n + b)/Γ(a + n)] · Π_{k<n} (a + k)/(a + k + b)`,
/// each factor through `ln(1 + b/(a + k))`.
fn ln_gamma_ratio(a: f64, b: f64) -> f64 {
    let mut a = a;
    let mut shift = 0.0;
    while a < 10.0 {
        shift += (b / a).ln_1p();
        a += 1.0;
    }
    (a - 0.5) * (b / a).ln_1p() + b * (a + b).ln() - b + stirlerr_diff(a, b) - shift
}

/// `stirlerr(a + b) − stirlerr(a)` for `a ≥ 10`, `b > 0`, term by term in
/// the series of [`stirlerr`] (DLMF 5.11.1):
/// `Σ cₖ a^{−m}·expm1(−m ln(1 + b/a))`, `m = 2k − 1`, so that the difference
/// keeps its relative accuracy as `b → 0` — the direct difference of two
/// `≈ 1/(12a)` values has an absolute error `ε/(12a)`, which swamps a
/// result of order `b/a²`.
fn stirlerr_diff(a: f64, b: f64) -> f64 {
    // B₂ₖ/(2k(2k − 1)), k = 1..7, as in `stirlerr`.
    const C: [f64; 7] = [
        1.0 / 12.0,
        -1.0 / 360.0,
        1.0 / 1260.0,
        -1.0 / 1680.0,
        1.0 / 1188.0,
        -691.0 / 360_360.0,
        1.0 / 156.0,
    ];
    let l = (b / a).ln_1p();
    let inv = 1.0 / a;
    let inv2 = inv * inv;
    let mut pow = inv; // a^{−m}
    let mut m = 1.0;
    let mut sum = 0.0;
    for c in C {
        sum += c * pow * (-m * l).exp_m1();
        pow *= inv2;
        m += 2.0;
    }
    sum
}

/// `1/(a·B(a, b)) = Γ(a + b)/(Γ(1 + a)Γ(b))`, the prefactor of the power
/// series of `I_x(a, b)`, with a small shape kept out of the exponent:
///
/// * `b < a ≤ 1`: `(b/(a + b))·Γ(1 + a + b)/(Γ(1 + a)Γ(1 + b))`, the
///   small `b` as a mantissa;
/// * `a ≤ 1`, `a ≤ b`: `e^{ln Γ(b + a) − ln Γ(b) − ln Γ(1 + a)}` — an
///   exponent of order `a` computed to its own relative accuracy, so that
///   `1 − I_x(a, b)` can be taken from it through `expm1` when `a` is tiny;
/// * `b ≤ 1 < a`: `(b/a)·e^{ln Γ(a + b) − ln Γ(a) − ln Γ(1 + b)}`;
/// * `a, b > 1`: `e^{−ln a − ln B(a, b)}`.
///
/// `ln Γ(1 + t)` is [`lgamma1p`], the differences [`ln_gamma_ratio`].  The
/// `ln B(a, b)` of 0.26 carried `−ln b` (690 at `b = 10⁻³⁰⁰`) into an
/// exponent that then cancelled, costing `|ln b|·ε` relative.
fn inv_a_beta(a: f64, b: f64) -> Scaled {
    // u/v as a mantissa, or in the exponent where the quotient is not a
    // normal double (b = 10⁻³⁰⁰ over a = 10¹⁰).
    let ratio = |u: f64, v: f64| {
        let r = u / v;
        if r >= f64::MIN_POSITIVE {
            Scaled::of(r)
        } else {
            Scaled::exp(u.ln() - v.ln())
        }
    };
    if a <= 1.0 && b < a {
        ratio(b, a + b).times_exp(lgamma_small(a + b) - lgamma1p(a) - lgamma1p(b))
    } else if a <= 1.0 {
        Scaled::exp(ln_gamma_ratio(b, a) - lgamma1p(a))
    } else if b <= 1.0 {
        ratio(b, a).times_exp(ln_gamma_ratio(a, b) - lgamma1p(b))
    } else {
        Scaled::exp(-a.ln() - lbeta(a, b))
    }
}

/// `ln Γ(1 + t)` for `0 ≤ t ≤ 2`: [`lgamma1p`] up to 1, `lgamma(1 + t)`
/// beyond (where `1 + t` loses nothing).
fn lgamma_small(t: f64) -> f64 {
    if t <= 1.0 {
        lgamma1p(t)
    } else {
        lgamma(1.0 + t)
    }
}

/// The Taylor route for a shape `0 < a ≤ 1` and `x < 1.1` (DiDonato &
/// Morris 1986, §2): from the series
/// `Γ(1 + a) P(a, x)/xᵃ = 1 − j`, `j = −a Σ_{n≥1} (−x)ⁿ/(n!(a + n))`, and
/// `1/Γ(1 + a) = 1 + h` (`h =` [`gam1`]), `P = eᶻ(1 + h)(1 − j)` with
/// `z = a ln x`, and `Q = ((l + 1) j − l)(1 + h) − h` with `l = eᶻ − 1`.
/// Where `Q` is the smaller tail (`xᵃ` near 1) it is formed by the second
/// expression, which does not cancel; elsewhere `P` by the first.
fn gamma_taylor_small_shape(a: f64, x: f64) -> Tails {
    let mut an = 3.0;
    let mut c = x;
    let mut sum = x / (a + 3.0);
    let tol = 0.1 * EPS / (a + 1.0);
    loop {
        an += 1.0;
        c *= -(x / an);
        let t = c / (a + an);
        sum += t;
        if t.abs() <= tol || an > 200.0 {
            break;
        }
    }
    let j = a * x * ((sum / 6.0 - 0.5 / (a + 2.0)) * x + 1.0 / (a + 1.0));
    let z = a * x.ln();
    let h = gam1(a);
    let g = h + 1.0;
    if (x >= 0.25 && a < x / 2.59) || z > -0.13394 {
        let l = z.exp_m1();
        let q = ((l + 1.0) * j - l) * g - h;
        Tails::from_upper(Scaled::of(q.max(0.0)))
    } else {
        Tails::from_lower(Scaled::of(z.exp() * g * (1.0 - j)))
    }
}

/// `(P(a, x), Q(a, x))` for `0 < a < 1`: the Taylor route below `x = 1.1`,
/// Legendre's continued fraction for `Q` above it with the prefactor
/// `xᵃe⁻ˣ/Γ(a) = a(1 + h)e^{a ln x − x}` — `Γ(a)` never formed, whose
/// logarithm `≈ −ln a` would carry `|ln a|·ε` into the result.  (0.26 took
/// `P` from the power series and `Q = 1 − P` for every `x < a + 1`:
/// `gammaincc(1e-300, 0.9) = −4.6·10⁻¹⁴`, truly `2.6·10⁻³⁰¹`.)
fn gammainc_tails_small_shape(a: f64, x: f64) -> Tails {
    if x < 1.1 {
        return gamma_taylor_small_shape(a, x);
    }
    let r = Scaled {
        m: a * (1.0 + gam1(a)),
        s: a * x.ln() - x,
    };
    Tails::from_upper(r.mul(gamma_q_cf_over_r(a, x)))
}

/// `Q(a, x)/r` with `r = xᵃe⁻ˣ/Γ(a)` for `0 < a ≤ 1` (the `j` of
/// [`bgrat`]): the continued fraction from `x = 1.1` on, the Taylor route
/// divided by `r` below.  Infinite when `r` underflows.
fn grat_r(a: f64, x: f64, r: Scaled) -> f64 {
    if x >= 1.1 {
        return gamma_q_cf_over_r(a, x).value();
    }
    gamma_taylor_small_shape(a, x).upper / r.value()
}

/// `I_x(a, b)` for `a ≥ 15`, `b ≤ 1` by the asymptotic expansion of
/// DiDonato & Morris (1992, §9) in the incomplete gamma function of
/// `−(a + (b−1)/2) ln x`; `None` when it cannot be evaluated (the partial
/// sums turn negative).  The prefactors keep `b` as a factor —
/// `r = e⁻ᶻzᵇ/Γ(b) = b(1 + gam1(b))·zᵇe⁻ᶻ` and `u = r·Γ(a + b)/(Γ(a)νᵇ)`
/// with `ln Γ(a + b) − ln Γ(a)` from [`ln_gamma_ratio`] — where 0.26 took
/// `lgamma(b) ≈ −ln b` into the exponent and out again, and `gam1(b)` from
/// `lgamma(1 + b)`, which is `0` for `b < ε`: `I_{0.99}(50, 10⁻²⁰)` came out
/// `1.14·10⁻²⁰`, truly `5.63·10⁻²¹`.
fn bgrat(a: f64, b: f64, x: f64, y: f64) -> Option<Scaled> {
    const N_TERMS: usize = 30;
    let bm1 = b - 1.0;
    let nu = a + 0.5 * bm1;
    let lnx = if y > 0.375 { x.ln() } else { (-y).ln_1p() };
    let z = -nu * lnx;
    if z.is_nan() || z <= 0.0 {
        return None;
    }
    let r = Scaled {
        m: b * (1.0 + gam1(b)),
        s: b * z.ln() - z,
    };
    let u = Scaled {
        m: r.m,
        s: r.s + ln_gamma_ratio(a, b) - b * nu.ln(),
    };
    let v = 0.25 / (nu * nu);
    let t2 = lnx * 0.25 * lnx;
    let mut j = grat_r(b, z, r);
    if !j.is_finite() {
        return None;
    }
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
    Some(u.times(sum))
}

/// `I_x(a, b)` for `a, b > 1` by the continued fraction of DiDonato &
/// Morris, `λ = (a + b) y − b ≥ 0` (so `x` is below the mean and the value
/// is the smaller tail).
fn bfrac(a: f64, b: f64, x: f64, y: f64, lambda: f64) -> Scaled {
    let brc = beta_pref(a, b, x, y);
    if brc.s == f64::NEG_INFINITY || brc.m == 0.0 {
        return Scaled::ZERO;
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
    brc.times(r)
}

/// `I_x(a, b)` for large `a, b` (both `≥ 15`, neither below `100` unless
/// `λ` is small) by the asymptotic expansion of DiDonato & Morris,
/// `λ = (a + b) y − b ≥ 0`.
fn basym(a: f64, b: f64, lambda: f64) -> Scaled {
    const NUM: usize = 20;
    /// `2/√π`.
    const E0: f64 = std::f64::consts::FRAC_2_SQRT_PI;
    /// `2^{−3/2}`.
    const E1: f64 = 0.353_553_390_593_273_7;
    let f = a * rlog1(-lambda / a) + b * rlog1(lambda / b);
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
        // Far out (f of order 10³⁰⁰ for shapes near 10³⁰⁰) the powers of
        // z overflow while those of w₀ underflow; their product is small, and
        // the tail itself is e^{−f} — keep the terms so far.
        if !(t0.is_finite() && t1.is_finite()) {
            break;
        }
        sum += t0 + t1;
        if t0.abs() + t1.abs() <= 100.0 * EPS * sum {
            break;
        }
        n += 2;
    }
    let bcorr = stirlerr(a) + stirlerr(b) - stirlerr(a + b);
    Scaled {
        m: E0 * (-bcorr).exp() * sum,
        s: -f,
    }
}

/// A tail probability: the lower tail `P(X ≤ x)` or the upper tail
/// `P(X > x)` — the tail a quantile inversion targets.
#[derive(Clone, Copy, Debug)]
enum Side {
    Lower(f64),
    Upper(f64),
}

/// The tail a route of [`bratio`] computed directly.
#[derive(Clone, Copy, Debug)]
enum Direct {
    Lower(Scaled),
    Upper(Scaled),
}

/// The TOMS 708 dispatch for `I_x(a, b)`, `a, b > 0`, `0 < x < 1`,
/// `y = 1 − x`: chooses the route, and the tail it computes directly, from
/// the sizes of `a`, `b` and the position of `x` relative to the mean.
/// `None` when `bgrat` fails.
fn bratio_route(a: f64, b: f64, x: f64, y: f64) -> Option<(Direct, bool)> {
    let (side, swap) = if a.min(b) <= 1.0 {
        let swap = x > 0.5;
        let (a0, b0, x0, y0) = if swap { (b, a, y, x) } else { (a, b, x, y) };
        // Now x0 ≤ ½ ≤ y0.
        let side = if a0.max(b0) > 1.0 {
            if b0 <= 1.0 {
                Direct::Lower(bpser(a0, b0, x0))
            } else if x0 >= 0.29 {
                Direct::Upper(bpser(b0, a0, y0))
            } else if x0 < 0.1 && (x0 * b0).powf(a0) <= 0.7 {
                Direct::Lower(bpser(a0, b0, x0))
            } else if b0 > 15.0 {
                Direct::Upper(bgrat(b0, a0, y0, x0)?)
            } else {
                Direct::Upper(bup(b0, a0, y0, x0, 20).plus(bgrat(b0 + 20.0, a0, y0, x0)?))
            }
        } else if a0 >= 0.2f64.min(b0) || x0.powf(a0) <= 0.9 {
            Direct::Lower(bpser(a0, b0, x0))
        } else if x0 >= 0.3 {
            Direct::Upper(bpser(b0, a0, y0))
        } else {
            Direct::Upper(bup(b0, a0, y0, x0, 20).plus(bgrat(b0 + 20.0, a0, y0, x0)?))
        };
        (side, swap)
    } else {
        // λ = a y − b x = a − (a + b)x = (a + b)y − b: positive when x is
        // below the mean a/(a + b); from the exact one of x, y (the smaller)
        // without rounding (a + b) or its product ([`mean_offset`]) — basym's
        // exponent is ≈ λ²/2 (1/a + 1/b), so an error ε·a in λ was 10⁻¹¹ of
        // the tail at shapes near 3·10⁷ (0.28 chose by a > b).
        let lambda = if x <= y {
            -mean_offset(a, b, x, a)
        } else {
            mean_offset(a, b, y, b)
        };
        let swap = lambda < 0.0;
        let (a0, b0, x0, y0, lambda) = if swap {
            (b, a, y, x, -lambda)
        } else {
            (a, b, x, y, lambda)
        };
        let side = if b0 < 40.0 {
            if b0 * x0 <= 0.7 {
                Direct::Lower(bpser(a0, b0, x0))
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
                    w = w.plus(bpser(a0, bf, x0));
                } else {
                    let mut aa = a0;
                    if aa <= 15.0 {
                        w = w.plus(bup(aa, bf, x0, y0, 20));
                        aa += 20.0;
                    }
                    w = w.plus(bgrat(aa, bf, x0, y0)?);
                }
                Direct::Lower(w)
            }
        } else if a0 > b0 {
            if b0 <= 100.0 || lambda > 0.03 * b0 {
                Direct::Lower(bfrac(a0, b0, x0, y0, lambda))
            } else {
                Direct::Lower(basym(a0, b0, lambda))
            }
        } else if a0 <= 100.0 || lambda > 0.03 * a0 {
            Direct::Lower(bfrac(a0, b0, x0, y0, lambda))
        } else {
            Direct::Lower(basym(a0, b0, lambda))
        };
        (side, swap)
    };
    Some((side, swap))
}

/// `(I_x(a, b), 1 − I_x(a, b))` for `a, b > 0` and `x + y = 1`, each tail
/// computed directly where it is the smaller one.  `NaN` for invalid
/// arguments.
fn bratio(a: f64, b: f64, x: f64, y: f64) -> Tails {
    // `a + b` must be a double: every route forms it (0.26 returned 0 and
    // 1 for both tails of Beta(f64::MAX, f64::MAX)).
    if a.is_nan()
        || b.is_nan()
        || x.is_nan()
        || y.is_nan()
        || a <= 0.0
        || b <= 0.0
        || !(a + b).is_finite()
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
        Direct::Lower(w) => Tails::from_lower(w),
        Direct::Upper(w1) => Tails::from_upper(w1),
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

/// The quantile when it lies outside the positive floats of a family's
/// support, from the tails at the smallest positive float `lo_x` and at
/// the largest float of the support `hi_x`: `lo_x` when that end already
/// meets the level — `P(X ≤ lo_x) ≥ p`, or `P(X > lo_x) ≤ q` — the
/// definition `inf{x : F(x) ≥ p}` over the floats; `beyond` when even
/// `hi_x` falls short (`+∞`, or `1` for the beta).  Compared in
/// logarithms, which a subnormal tail keeps.  `None` otherwise.
///
/// Tiny shapes pile the mass against an end: 0.26 had only the first of
/// the four cases for most families and answered, say, `f::isf(1e-10,
/// 1e-300, 1e10)` — where every positive `x` has `P(X > x) < 10⁻²⁹⁷` —
/// with `+∞`.
fn outside_float_range(
    side: Side,
    at_lo: Tails,
    lo_x: f64,
    at_hi: Tails,
    beyond: f64,
) -> Option<f64> {
    match side {
        Side::Lower(p) => {
            let lp = p.ln();
            if lp <= at_lo.ln_lower {
                Some(lo_x)
            } else if lp > at_hi.ln_lower {
                Some(beyond)
            } else {
                None
            }
        }
        Side::Upper(q) => {
            let lq = q.ln();
            if lq >= at_lo.ln_upper {
                Some(lo_x)
            } else if lq < at_hi.ln_upper {
                Some(beyond)
            } else {
                None
            }
        }
    }
}

/// `ln(1 − eˡ)` for `l ≤ 0`, without cancellation at either end (Mächler
/// 2012, "Accurately computing log(1 − exp(−|a|))"): `ln(−expm1(l))` for
/// `l > −ln 2`, where `eˡ` is near 1; `ln1p(−eˡ)` below.  The quantile
/// iterations of 0.28 took `ln1p(−eˡ)` throughout, which is `ln` of a
/// rounded difference near `l = 0` and `−∞` for `|l| < ε/2` — the upper
/// tail of a tiny shape, whose lower tail is `1 − O(shape)`:
/// `f::isf(3.27e-36, 7.97e-39, 1.37e7)` came out `0` (truly `2.503e-319`).
fn ln_1m_exp(l: f64) -> f64 {
    if l > -std::f64::consts::LN_2 {
        (-l.exp_m1()).ln()
    } else {
        (-l.exp()).ln_1p()
    }
}

/// The objective and slope of an upper-tail quantile iteration where the
/// lower tail is its leading term `ln P = s·u + c`, exactly linear in the
/// log variable `u` with slope `s`: `g = −ln(Q/q)` for `Q = 1 − P`, and
/// `dg/du = s·P/Q`.  `Q = −expm1(ln P)` keeps its relative accuracy, so
/// `g` is compared through [`log_ratio`] (resolution `ε`) rather than as
/// `ln q − ln Q` (resolution `ε·|ln q|`, which a slope `s·P/Q` of
/// `1.4·10⁻³` turns into `10⁻¹¹` of the quantile).
fn upper_from_leading_lower(ln_lower: f64, slope: f64, target: f64, ln_target: f64) -> Eval {
    let upper = -ln_lower.exp_m1();
    let ln_upper = ln_1m_exp(ln_lower);
    Eval {
        g: -log_ratio(upper, ln_upper, target, ln_target),
        dg: slope * (ln_lower - ln_upper).exp(),
    }
}

/// `ln(value/target)` for a tail and its level, each given with its
/// logarithm: `ln(1 + (value − target)/target)` while both are normal
/// doubles, so that near the root the objective keeps the relative
/// resolution of the values (`ε`) rather than that of their logarithms
/// (`ε·|ln p|`: `1.1·10⁻¹³` at `p = 10⁻³⁰⁶`, which let 0.26 stop 45 ulps
/// from `gamma::ppf(1.46e-306, 2.98, 0.0284)`); the logarithms below.
fn log_ratio(value: f64, ln_value: f64, target: f64, ln_target: f64) -> f64 {
    if value >= f64::MIN_POSITIVE && target >= f64::MIN_POSITIVE && value.is_finite() {
        ((value - target) / target).ln_1p()
    } else {
        ln_value - ln_target
    }
}

/// The float nearest the quantile where one float step changes the tail by
/// more than `10⁻⁹` relative (or the tail misses its level): from the
/// solver's `x`, step by single floats against the family's own `tails`
/// (its rounding of `x/θ` included) to the adjacent pair between which the
/// level is crossed, and return the one whose tail is nearer the level in
/// logarithm — linear interpolation of `ln tail` across the ulp.  Elsewhere
/// the last ulp is below the tails' own rounding and `x` is returned as it
/// is.
///
/// A shape of `10²⁴` puts one ulp of the quantile at `10⁻⁴` of the tail,
/// and `10³⁰⁰` puts the whole distribution between two floats, where the
/// crossing pair is the only information there is; the audit's build
/// returned floats a few ulps off, whose tails missed the level by up to
/// that much (`gamma::isf(3.78e-3, 2.7e24, 1.75)`: 3.7762e-3).
fn snap_to_floats(x: f64, side: Side, tails: impl Fn(f64) -> Tails) -> f64 {
    use std::cmp::Ordering::{Greater, Less};
    if !x.is_finite() {
        return x;
    }
    let tail = |t: Tails| match side {
        Side::Lower(_) => t.ln_lower,
        Side::Upper(_) => t.ln_upper,
    };
    let ln_level = match side {
        Side::Lower(p) | Side::Upper(p) => p.ln(),
    };
    let here = tail(tails(x));
    let per_ulp = (tail(tails(x.next_up())) - here).abs();
    // Neighbouring floats can share one `x/θ` and so one tail value; a tail
    // that misses its level although it does not move within an ulp is the
    // same regime.
    let misses = (here - ln_level).abs() > 1e-6;
    if !misses && (per_ulp.is_nan() || per_ulp <= 1e-9) {
        return x;
    }
    let done = |x: f64| {
        let t = tails(x);
        match side {
            Side::Lower(p) => tail_cmp(t.lower, t.ln_lower, p).map(|o| o != Less),
            Side::Upper(q) => tail_cmp(t.upper, t.ln_upper, q).map(|o| o != Greater),
        }
    };
    // The crossing pair (lo, hi): the level not yet met at lo, met at hi.
    let start = x;
    let mut x = x;
    let mut crossing = None;
    for _ in 0..64 {
        match done(x) {
            Some(true) => {
                let prev = x.next_down();
                if done(prev) == Some(true) {
                    x = prev;
                } else {
                    crossing = Some((prev, x));
                    break;
                }
            }
            Some(false) => x = x.next_up(),
            None => return start,
        }
    }
    let Some((lo, hi)) = crossing else {
        return start;
    };
    let (d_lo, d_hi) = (
        (tail(tails(lo)) - ln_level).abs(),
        (tail(tails(hi)) - ln_level).abs(),
    );
    if d_lo < d_hi { lo } else { hi }
}

/// A monotone objective and its derivative at one point.
struct Eval {
    g: f64,
    dg: f64,
}

/// The root `v + dv` found by [`solve_increasing`]: `v` is the evaluated
/// iterate with the smallest `|g|`, `dv` the Newton correction from it.
/// The correction is below the resolution of `v` — a log or logit variable
/// near `|v| = 100` has a spacing of `1.4·10⁻¹⁴`, 64 ulps of the quantile
/// — so the caller applies it in the quantile's own variable
/// (`x = eᵛ·e^{dv}`), where it is not rounded away.
#[derive(Clone, Copy, Debug)]
struct Root {
    v: f64,
    dv: f64,
}

impl Root {
    /// `e^{v + dv}` for a log variable.
    fn exp(self) -> f64 {
        let x = self.v.exp();
        if x.is_finite() && x >= f64::MIN_POSITIVE {
            x * self.dv.exp()
        } else {
            (self.v + self.dv).exp()
        }
    }
}

/// The root of an increasing `g` by Newton's method safeguarded by a
/// bracket: a Newton step is taken when it lands inside the current
/// bracket and at least halves it, a bisection (or, while an end is still
/// infinite, a doubling step outwards) otherwise.  `g` may return `±∞`
/// (a tail that underflowed): only its sign is used then.  Converged when
/// the step is below `16ε(1 + |v|)`, the resolution of a log or logit
/// variable, when the bracket is that narrow, or when a step already below
/// `10⁻⁶(1 + |v|)` fails to reduce `|g|` — the iteration has reached the
/// rounding noise of `g`.  The answer is then the best *evaluated* iterate
/// with its Newton correction ([`Root`]); 0.26 returned the unevaluated
/// last step, which after a rejected Newton step was a bisection midpoint
/// half a bracket away from a point that already solved the equation
/// (`gamma::ppf(1.3e-235, 2.4e10, 0.118)`, 254 ulps).
fn solve_increasing(
    op: &'static str,
    g: impl Fn(f64) -> Eval,
    v0: f64,
    lo: f64,
    hi: f64,
) -> Result<Root, SymplexError> {
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
    // An infinite start (a start formula at its limit) begins at the edge
    // of the doubles' logarithms instead; the bracket search goes from there.
    let mut v = v0.clamp(-745.0, 710.0).clamp(lo, hi);
    let mut step = 1.0;
    let mut width = f64::INFINITY;
    // (iterate, |g|, Newton correction)
    let mut best = (v, f64::INFINITY, 0.0);
    // The answer once the iteration stops: `best` with its Newton
    // correction, if that is within the resolution of `v` (otherwise `g` is
    // at its rounding noise and the correction means nothing).  But if even
    // the best `|g|` is large — the tail is a factor `e^{1/2}` off its level
    // at every point tried — then `g` jumps across zero between adjacent
    // values of `v`: the distribution is narrower than a float's spacing
    // (Beta(10¹⁰⁰, 10¹⁰⁰) lies within 10⁻⁴⁹ of ½), and the smallest `|g|`
    // picks an arbitrary float (0.26 answered `beta::isf(1e-300, 1e100,
    // 1e100)` with 0.29).  The quantile over the floats, `inf{x : F(x) ≥ p}`,
    // is then the upper end of the bracket: `g > 0` there, and `g` increases.
    let finish = |best: (f64, f64, f64), hi: f64| {
        let (v, g_abs, dv) = best;
        if g_abs >= 0.5 && hi.is_finite() {
            Root { v: hi, dv: 0.0 }
        } else if dv.abs() <= 64.0 * TOL * (1.0 + v.abs()) {
            Root { v, dv }
        } else {
            Root { v, dv: 0.0 }
        }
    };
    for _ in 0..MAX_ITER {
        let Eval { g: gv, dg } = g(v);

        if gv.is_nan() {
            return Err(SymplexError::computation_failed(
                op,
                format!("the distribution function is not a number at {v}"),
            ));
        }
        if gv == 0.0 {
            return Ok(Root { v, dv: 0.0 });
        }
        if gv.abs() < best.1 {
            let corr = if dg.is_finite() && dg > 0.0 {
                -gv / dg
            } else {
                0.0
            };
            best = (v, gv.abs(), corr);
        } else if gv.is_finite()
            && best.1 < 0.5
            && best.2.abs() <= QUADRATIC_PHASE * (1.0 + best.0.abs())
            && width <= QUADRATIC_PHASE * (1.0 + v.abs())
        {
            // Only at the root: the best point's own Newton correction is
            // already that small (0.26 also stopped here far from it, when a
            // meaningless derivative at an overflowed `x` gave a tiny step:
            // `f::isf(0.93, 3.6e6, 2.1e-4)` returned 2.1e219, whose tail is
            // 0.948).
            return Ok(finish(best, hi));
        }
        if gv < 0.0 {
            lo = v;
        } else {
            hi = v;
        }
        let newton = v - gv / dg;
        // Toward a side not yet bracketed a Newton step may go no further
        // than the outward doubling step would: far from the root `dg` can
        // be meaningless (0.26 took `t::ppf(1e-300, 1e100)` from `u = 700`
        // to `u = −5.8e102` in one step and then bisected past its budget).
        let open = if newton < v { lo } else { hi };
        let newton_ok = gv.is_finite()
            && dg.is_finite()
            && dg > 0.0
            && newton > lo
            && newton < hi
            && 2.0 * gv.abs() <= (width * dg).abs()
            && (open.is_finite() || (newton - v).abs() <= 2.0 * step);
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
        // While `g` only jumps (see `finish`) the bracket is the answer, and
        // it is narrowed to a quarter ulp of the quantile variable.
        let tol = if best.1 >= 0.5 {
            0.25 * EPS * (1.0 + v.abs())
        } else {
            TOL * (1.0 + v.abs())
        };
        if (next - v).abs() <= tol || hi - lo <= tol {
            // No finite |g| seen (every tail underflowed): only the
            // bracket is known.
            if best.1.is_infinite() {
                let v = if hi.is_finite() { hi } else { next };
                return Ok(Root { v, dv: 0.0 });
            }
            return Ok(finish(best, hi));
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
/// [`Distribution::quantile_f64`](super::Distribution::quantile_f64)
/// searches the lattice of any discrete family with it too, deciding each
/// point exactly.
pub(crate) fn discrete_search(
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

/// A tail, given with its logarithm, against a level `> 0`: the values
/// while both are normal doubles, the logarithms otherwise — a subnormal
/// tail has too few bits to be compared (0.26 answered
/// `poisson::ppf(2e-323, 8.81e7)` with 87782519, whose cdf `1.73e-323`
/// rounds up to the level; the answer is 87782552).
fn tail_cmp(value: f64, ln_value: f64, level: f64) -> Option<std::cmp::Ordering> {
    if value >= f64::MIN_POSITIVE && level >= f64::MIN_POSITIVE {
        value.partial_cmp(&level)
    } else {
        ln_value.partial_cmp(&level.ln())
    }
}

/// The smallest lattice point `k ∈ [kmin, kmax]` at which the tail named by
/// `target` has crossed its level — `P(X ≤ k) ≥ p` for `Side::Lower(p)`,
/// `P(X > k) ≤ q` for `Side::Upper(q)` — from a guess `k0`.  Callers name
/// the *smaller* tail ([`nearest_tail`], [`nearest_tail_upper`]): `cdf(k) ≥
/// p` and `sf(k) ≤ 1 − p` are the same condition, and `1 − p` is exact
/// for `p ≥ ½`, but only the small tail is resolved near its level (0.26
/// compared the rounded `sf = 1 − 1.67·10⁻¹⁶` with `q = 1 − 2⁻⁵²` and
/// returned `poisson::isf(1 − 2⁻⁵², 1229036.68) = 1220000`; the answer is
/// 1220039).
fn discrete_quantile(
    op: &'static str,
    tails: impl Fn(f64) -> Tails,
    target: Side,
    k0: f64,
    kmin: f64,
    kmax: f64,
) -> Result<f64, SymplexError> {
    let done = |k: f64| {
        let t = tails(k);
        if t.lower.is_nan() || t.upper.is_nan() {
            return None;
        }
        use std::cmp::Ordering::{Greater, Less};
        Some(match target {
            Side::Lower(p) => tail_cmp(t.lower, t.ln_lower, p)? != Less,
            Side::Upper(q) => tail_cmp(t.upper, t.ln_upper, q)? != Greater,
        })
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
        Eval, Side, Tails, bratio, check_level, check_positive, inv_a_beta, lbeta, log_beta_pref,
        log_ratio, nearest_tail, nearest_tail_upper, norm, power_pair, snap_to_floats,
        solve_increasing,
    };
    use crate::base::errors::SymplexError;
    use std::f64::consts::LN_2;

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
            // A tiny ν: 1/B(a, ½) = a·inv_a_beta(a, ½) (Loader's form cancels).
            Some((x0, y0)) if a < 1e-3 => {
                power_pair(inv_a_beta(a, 0.5).times(a), x0, a, y0, 0.5).ln()
            }
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
            return Tails::of(norm::cdf(x), norm::sf(x));
        }
        if x == 0.0 {
            return Tails::of(0.5, 0.5);
        }
        let ax = x.abs();
        let a = 0.5 * df;
        let (tail, ln_tail, body) = match beta_args(ax, df) {
            Some((x0, y0)) => {
                let i = bratio(a, 0.5, x0, y0);
                (0.5 * i.lower, i.ln_lower - LN_2, 0.5 + 0.5 * i.upper)
            }
            None => {
                // (ν/x²)^{ν/2} = ν^{ν/2}·|x|^{−ν}: two `powf` roundings while
                // both are normal, instead of `(ν/2)·ln(ν/x²)` (several hundred)
                // rounded in the exponent.
                let pw = df.powf(a) * ax.powf(-df);
                let w = if pw.is_finite() && pw >= f64::MIN_POSITIVE {
                    inv_a_beta(a, 0.5).times(pw)
                } else {
                    inv_a_beta(a, 0.5).times_exp(a * ln_r(ax, df))
                }
                .times(0.5);
                let tail = w.value();
                (tail, w.ln(), 1.0 - tail)
            }
        };
        let t = Tails {
            lower: tail,
            upper: body,
            ln_lower: ln_tail,
            ln_upper: body.ln(),
        };
        if x > 0.0 { t.flipped() } else { t }
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
        // In logarithms: at a subnormal level the rounded tail has too few
        // bits to decide.
        if q.ln() < tails(f64::MAX, df).ln_upper {
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
            let tl = tails(t, df);
            Eval {
                g: -log_ratio(tl.upper, tl.ln_upper, q, lnq),
                dg: (log_pref(t, df) - tl.ln_upper).exp(),
            }
        };
        let t = solve_increasing(op, g, u0, f64::NEG_INFINITY, f64::INFINITY)?.exp();
        Ok(snap_to_floats(t, Side::Upper(q), |x| tails(x, df)))
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
            // The median, for `ppf(½)` and `isf(½)` alike (0.26 answered
            // `t::isf(0.5, 1)` with 5.5e-30).
            Side::Lower(0.5) | Side::Upper(0.5) => 0.0,
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
        Eval, Root, Side, Tails, check_level, check_positive, gammainc_tails, lgamma, lgamma1p,
        log_gamma_pref, log_ratio, nearest_tail, nearest_tail_upper, norm, outside_float_range,
        snap_to_floats, solve_increasing, upper_from_leading_lower,
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
        let u = x / scale;
        if u < f64::MIN_POSITIVE && scale != 1.0 {
            // `x/θ` is subnormal or underflowed (θ > 1), keeping few or no
            // significant bits, but a tiny shape still has real mass there
            // (χ² with 0.01 df: P(X ≤ 5e-324) = 0.0242).  For u < 1e-300 the
            // series is its leading term, P(k, u) = uᵏ/Γ(k + 1), formed in
            // logarithms from `x` and `θ` themselves.
            // The upper tail as −expm1 of that logarithm: a tiny shape has
            // P near 1 here, and 0.26's `1 − P` gave `chi2::sf(5e-324, 5e-144)
            // = 0` (truly 1.9e-141), so `chi2::isf(4.9e-308, 5e-144)` answered
            // 5e-324.
            let lg = if shape <= 1.0 {
                lgamma1p(shape)
            } else {
                lgamma(shape + 1.0)
            };
            let ln_lower = shape * (x.ln() - scale.ln()) - lg;
            let upper = -ln_lower.exp_m1();
            return Tails {
                lower: ln_lower.exp(),
                upper,
                ln_lower,
                ln_upper: upper.ln(),
            };
        }
        gammainc_tails(shape, u)
    }

    /// `P(X ≤ x) = P(k, x/θ)`.  `scipy.stats.gamma.cdf(x, k, scale=θ)`.
    pub fn cdf(x: f64, shape: f64, scale: f64) -> f64 {
        tails(x, shape, scale).lower
    }

    /// `P(X > x) = Q(k, x/θ)`, accurate in the tail.
    pub fn sf(x: f64, shape: f64, scale: f64) -> f64 {
        tails(x, shape, scale).upper
    }

    /// The logarithm `ln y` of the standard gamma (`θ = 1`) quantile:
    /// Newton on the logarithm of the nearer tail in `u = ln y`, from the
    /// Wilson–Hilferty start `k(1 − 1/(9k) + z/(3√k))³` (or
    /// `(p Γ(k + 1))^{1/k}` deep in the lower tail).  Callers scale in
    /// logarithms, `x = e^{u + ln θ}`: a tiny shape puts lower quantiles
    /// where `y` itself is below every float although `θy` is not.  Below
    /// `y = e^{−690}` the lower tail is its series' leading term
    /// `yᵏ/Γ(k + 1)`, so `u` never has to be exponentiated there.
    pub(super) fn standard_log_quantile(
        op: &'static str,
        side: Side,
        shape: f64,
    ) -> Result<StdQuantile, SymplexError> {
        const LOG_TINY: f64 = -690.0;
        let (z, target, lower) = match side {
            Side::Lower(pl) => (norm::ppf(pl)?, pl, true),
            Side::Upper(q) => (norm::isf(q)?, q, false),
        };
        let ln_target = target.ln();
        if shape >= DIRECT_MIN_SHAPE {
            return direct_quantile(op, z, target, lower, shape).map(StdQuantile::Value);
        }
        let x0 = shape * (1.0 - 1.0 / (9.0 * shape) + z / (3.0 * shape.sqrt())).powi(3);
        let u0 = if x0.is_finite() && x0 > 0.0 {
            x0.ln()
        } else if lower {
            (ln_target + lgamma(shape + 1.0)) / shape
        } else {
            (-ln_target).max(shape).ln()
        };
        // ln Γ(k + 1) to its own relative accuracy: `lgamma(1 + k)` rounds
        // `1 + k` to 1 for a tiny shape and returns 0 for a value of −γk.
        let ln_gamma_k1 = if shape <= 1.0 {
            lgamma1p(shape)
        } else {
            lgamma(shape + 1.0)
        };
        let g = |u: f64| {
            if u < LOG_TINY {
                // ln P(k, y) = k·u − ln Γ(k + 1) + O(y), exactly linear.
                let ln_lower = shape * u - ln_gamma_k1;
                return if lower {
                    Eval {
                        g: ln_lower - ln_target,
                        dg: shape,
                    }
                } else {
                    upper_from_leading_lower(ln_lower, shape, target, ln_target)
                };
            }
            let x = u.exp();
            let tl = gammainc_tails(shape, x);
            let (tail, ln_tail) = if lower {
                (tl.lower, tl.ln_lower)
            } else {
                (tl.upper, tl.ln_upper)
            };
            let r = log_ratio(tail, ln_tail, target, ln_target);
            Eval {
                g: if lower { r } else { -r },
                dg: (log_gamma_pref(shape, x) - ln_tail).exp(),
            }
        };
        solve_increasing(op, g, u0, f64::NEG_INFINITY, f64::INFINITY).map(StdQuantile::Log)
    }

    /// A standard gamma quantile `y`: as `ln y` from the log-variable
    /// iteration (a tiny shape puts `y` below every float although `θy`
    /// is not), or as `y` itself from [`direct_quantile`].
    pub(super) enum StdQuantile {
        Log(Root),
        Value(f64),
    }

    /// From this shape on the quantile is found in `y` itself, not `ln y`:
    /// the distribution's relative width `1/√k` falls below the resolution
    /// of `ln y` (`ε ln k`) near `k = 10³⁰`, and 0.26's log-variable
    /// iteration returned `gamma::ppf(0.3, 1e30) = e²·10³⁰` (and failed
    /// outright at `k = 10³⁰⁰`); it was also slow (1.4 ms at `k = 10²⁰`).
    const DIRECT_MIN_SHAPE: f64 = 1e10;

    /// The standard gamma quantile for `k ≥` [`DIRECT_MIN_SHAPE`]: the
    /// Cornish–Fisher start `k + z√k + (z² − 1)/3 + (z³ − 7z)/(36√k)`
    /// (standardised skewness `2/√k`, excess kurtosis `6/k`), whose
    /// neglected term is `O(z⁴/k)` — below a unit at `|z| ≤ 38.5`, the
    /// subnormal levels — then Newton on the logarithm of the tail in `y`
    /// (Temme's expansion evaluates every tail within `38.5` standard
    /// deviations), until a step no longer moves `y`.
    fn direct_quantile(
        op: &'static str,
        z: f64,
        target: f64,
        lower: bool,
        shape: f64,
    ) -> Result<f64, SymplexError> {
        let ln_target = target.ln();
        let sd = shape.sqrt();
        // `z` is already signed as the quantile's standard score: `ppf` of a
        // lower level, `isf` of an upper one.
        let mut y = (shape + z * sd + (z * z - 1.0) / 3.0 + (z * z * z - 7.0 * z) / (36.0 * sd))
            .min(f64::MAX);
        for _ in 0..8 {
            let tl = gammainc_tails(shape, y);
            let (tail, ln_tail) = if lower {
                (tl.lower, tl.ln_lower)
            } else {
                (tl.upper, tl.ln_upper)
            };
            let r = log_ratio(tail, ln_tail, target, ln_target);
            let g = if lower { r } else { -r };
            // d ln P/dy = f(y)/P, f(y) = yᵏ⁻¹e⁻ʸ/Γ(k).
            let dg = (log_gamma_pref(shape, y) - y.ln() - ln_tail).exp();
            if !g.is_finite() || !dg.is_finite() || dg <= 0.0 {
                return Err(SymplexError::computation_failed(
                    op,
                    format!("the distribution function is not usable at {y}"),
                ));
            }
            // `g` increases with `y`: still negative at the largest double,
            // the quantile lies beyond it (a shape within a few standard
            // deviations of f64::MAX).
            if y == f64::MAX && g < 0.0 {
                return Ok(f64::INFINITY);
            }
            let next = (y - g / dg).min(f64::MAX);
            if (next - y).abs() <= 0.5 * f64::EPSILON * y {
                return Ok(next);
            }
            y = next;
        }
        Ok(y)
    }

    fn quantile(op: &'static str, side: Side, shape: f64, scale: f64) -> Result<f64, SymplexError> {
        check_positive(op, "the shape", shape)?;
        check_positive(op, "the scale", scale)?;
        // A tiny shape puts P(X ≤ 5e-324) above small lower levels (shape
        // 0.005: 0.024); those quantiles are below every positive float, and
        // the answer is the smallest float with F(x) ≥ p (as for `beta`),
        // not the 0 the logarithmic solve underflows to.
        let tiniest = f64::from_bits(1);
        if let Some(x) = outside_float_range(
            side,
            tails(tiniest, shape, scale),
            tiniest,
            tails(f64::MAX, shape, scale),
            f64::INFINITY,
        ) {
            return Ok(x);
        }
        let x = scale_log_quantile(standard_log_quantile(op, side, shape)?, scale);
        Ok(snap_to_floats(x, side, |x| tails(x, shape, scale)))
    }

    /// `θy`; for `y = e^{v + dv}` as `θ·eᵛ·e^{dv}` whenever `eᵛ` is a normal
    /// float, and `e^{v + dv + ln θ}` where `eᵛ` alone would under- or
    /// overflow.
    pub(super) fn scale_log_quantile(q: StdQuantile, scale: f64) -> f64 {
        let u = match q {
            StdQuantile::Value(y) => return scale * y,
            StdQuantile::Log(u) => u,
        };
        let y = u.v.exp();
        if y.is_finite() && y >= f64::MIN_POSITIVE {
            scale * (y * u.dv.exp())
        } else {
            (u.v + u.dv + scale.ln()).exp()
        }
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
    use super::{
        Side, check_level, check_positive, gamma, nearest_tail, nearest_tail_upper,
        outside_float_range, snap_to_floats,
    };
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
        // Outside the positive floats (see `gamma::quantile`).
        let tiniest = f64::from_bits(1);
        if let Some(x) = outside_float_range(
            side,
            gamma::tails(tiniest, 0.5 * df, 2.0),
            tiniest,
            gamma::tails(f64::MAX, 0.5 * df, 2.0),
            f64::INFINITY,
        ) {
            return Ok(x);
        }
        let x = gamma::scale_log_quantile(gamma::standard_log_quantile(op, side, 0.5 * df)?, 2.0);
        Ok(snap_to_floats(x, side, |x| gamma::tails(x, 0.5 * df, 2.0)))
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
        Eval, Root, Side, Tails, bratio, check_level, check_positive, lbeta, log_beta_pref,
        log_ratio, nearest_tail, nearest_tail_upper, norm, outside_float_range, snap_to_floats,
        solve_increasing,
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
        // The deep-tail start (p·a·B(a, b))^{1/a} underflows for tiny shapes
        // (a = 0.005, p = 0.02: e^{−741}); its logit is then its logarithm,
        // taken without exponentiating (0.22 started Newton at −∞).
        let deep = |ln_x: f64| -> f64 {
            let x = ln_x.exp().min(0.5);
            if x > 0.0 { x.ln() - (-x).ln_1p() } else { ln_x }
        };
        // logit(x₀) from whichever of x₀, 1 − x₀ is not rounded: 0.26 took
        // `1 − x₀` for every upper level, which is 1 for x₀ < ε/2, and
        // started from −∞ (`beta::isf(0.1, 2, 1e20)` came out 0).
        let logit = |x0: f64| {
            if x0 <= 0.5 {
                x0.ln() - (-x0).ln_1p()
            } else {
                let y = 1.0 - x0;
                (-y).ln_1p() - y.ln()
            }
        };
        Ok(match side {
            Side::Lower(pl) => {
                if x0 > 0.0 && x0 < 1.0 && pl >= 1e-3 {
                    logit(x0)
                } else {
                    deep((pl.ln() + alpha.ln() + lbeta(alpha, beta)) / alpha)
                }
            }
            Side::Upper(q) => {
                if x0 > 0.0 && x0 < 1.0 && q >= 1e-3 {
                    logit(x0)
                } else {
                    -deep((q.ln() + beta.ln() + lbeta(alpha, beta)) / beta)
                }
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
    ) -> Result<Root, SymplexError> {
        let v0 = start_logit(side, alpha, beta)?;
        let (target, lower) = match side {
            Side::Lower(pl) => (pl, true),
            Side::Upper(q) => (q, false),
        };
        let ln_target = target.ln();
        let g = |v: f64| {
            let (x, y) = (logistic(v), logistic(-v));
            let tl = bratio(alpha, beta, x, y);
            let (tail, ln_tail) = if lower {
                (tl.lower, tl.ln_lower)
            } else {
                (tl.upper, tl.ln_upper)
            };
            let r = log_ratio(tail, ln_tail, target, ln_target);
            let gv = if lower { r } else { -r };
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
        check_positive(op, "α + β", alpha + beta)?;
        // With tiny shapes the mass piles up within e^-700 of an end of
        // [0, 1]: Beta(10⁻³, 10⁻³) has P(X ≤ 5e-324) = 0.2375, so every
        // lower level below that has a quantile no f64 can hold.  Answer
        // with the smallest float whose tail reaches the level (the
        // quantile's definition, inf{x : F(x) ≥ p}, over the floats) rather
        // than letting the logit Newton wander into the subnormals (0.22
        // returned 5.6e-309 for the 0.2055 quantile, whose cdf is 0.246).
        let tiniest = f64::from_bits(1);
        let largest = 1.0 - f64::EPSILON / 2.0;
        if let Some(x) = outside_float_range(
            side,
            tails(tiniest, alpha, beta),
            tiniest,
            tails(largest, alpha, beta),
            1.0,
        ) {
            return Ok(x);
        }
        let x = logistic_root(quantile_logit(op, side, alpha, beta)?);
        Ok(snap_to_floats(x, side, |x| tails(x, alpha, beta)))
    }

    /// `logistic(v + dv)` with the correction applied to `x` itself:
    /// `d ln x/dv = 1 − x`.
    fn logistic_root(r: Root) -> f64 {
        if r.v < -700.0 {
            // 1 + eᵛ = 1: x = e^{v + dv}, rounded once.
            return (r.v + r.dv).exp();
        }
        logistic(r.v) * (r.dv * logistic(-r.v)).exp()
    }

    /// `1/(1 + e^{−v})` without overflow: `1/(1 + e^{−v})` stops at
    /// `1/f64::MAX ≈ 5.6e-309` once `e^{−v}` overflows (`v < −709.8`), so
    /// the quantile of a tail heavier than that was pinned there (0.22:
    /// `beta::isf(0.781, 1.07e-3, 1e-3) = 5.6e-309` with cdf 0.775); for
    /// `v < 0` the form `e^{v}/(1 + e^{v})` reaches the subnormals.
    fn logistic(v: f64) -> f64 {
        if v < 0.0 {
            let e = v.exp();
            e / (1.0 + e)
        } else {
            1.0 / (1.0 + (-v).exp())
        }
    }

    /// `x` with `P(X ≤ x) = p`.  `scipy.stats.beta.ppf(p, a, b)`.  `α + β`
    /// must be a finite double ([`SymplexError::InvalidArgument`] beyond, as
    /// for a non-positive shape).
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
        EPS, Eval, Scaled, Side, Tails, beta, bratio, check_level, check_positive, inv_a_beta,
        log_beta_pref, log_ratio, nearest_tail, nearest_tail_upper, outside_float_range,
        snap_to_floats, solve_increasing, upper_from_leading_lower,
    };
    use crate::base::errors::SymplexError;
    use std::f64::consts::LN_2;

    fn tails(x: f64, d1: f64, d2: f64) -> Tails {
        if x.is_nan() || !(d1.is_finite() && d1 > 0.0) || !(d2.is_finite() && d2 > 0.0) {
            return Tails::NAN;
        }
        if x <= 0.0 {
            return Tails::ZERO;
        }
        if x.is_infinite() {
            return Tails::ONE;
        }
        let (a, b) = (0.5 * d1, 0.5 * d2);
        let args = BetaArgs::new(x, d1, d2);
        if args.y < f64::MIN_POSITIVE {
            // The upper argument y = 1/(1 + t) is below the normal range (a
            // tiny d₂ keeps real mass there: F(6331, 0.01) has
            // P(X > 2.8e304) ≈ 0.027).  0.26 formed y = d₂/(d₁x + d₂), which
            // underflows to 0 long before the tail does, and returned
            // sf = 0 (`f::sf(7.2e239, 1.03e5, 6.4e-154)`: truly 1 − 2.9·10⁻¹⁵¹).
            let upper = small_arg_series(b, a, args.ln_y, (d1.ln() - LN_2 + args.ln_y).exp());
            return tails_from_small_arg(upper).flipped();
        }
        if args.z < f64::MIN_POSITIVE {
            // The lower argument z = t/(1 + t) is (`d₁x` underflowed, or d₂
            // is huge); a tiny d₁ still has mass there (F(0.01, 0.01):
            // P(X ≤ 5e-324) = 0.0121).
            let lower = small_arg_series(a, b, args.ln_z, (d2.ln() - LN_2 + args.ln_z).exp());
            return tails_from_small_arg(lower);
        }
        bratio(a, b, args.z, args.y)
    }

    /// `I_w(p, q)` for an argument `w` below the normal range, given `ln w`
    /// and `v = q·w` (which may be of order 1 when `q` is huge): the power
    /// series `wᵖ/(p B(p, q))·[1 + p Σ_{n≥1} ((1 − q)ₙ/n!) wⁿ/(p + n)]`
    /// with the terms `((1 − q)ₙ/n!) wⁿ = Π_{k≤n} (k/q − 1) v/k` built from
    /// `v`, never from the subnormal `w`.
    fn small_arg_series(p: f64, q: f64, ln_w: f64, v: f64) -> Scaled {
        let mut c = 1.0;
        let mut sum = 0.0;
        let mut n = 0.0;
        while n < 1000.0 {
            n += 1.0;
            c *= (n / q - 1.0) * v / n;
            let w = c / (p + n);
            sum += w;
            if (p * w).abs() <= EPS * (1.0 + p * sum).abs() {
                break;
            }
        }
        // The correction as ln(1 + p·Σ) in the exponent: the complement
        // `1 − I` is taken from the logarithm when `I` is near 1.
        inv_a_beta(p, q).times_exp(p * ln_w + (p * sum).ln_1p())
    }

    /// Both tails from the leading term `w` of the tail whose beta argument
    /// is below the normal range: that tail is `w`, the other `1 − w` —
    /// through `−expm1(ln w)` when `w` is near 1 (a tiny shape), where 0.26
    /// subtracted: `f::sf(2.2e-279, 4.2e-283, 1.89)` was `1.1·10⁻¹³`, truly
    /// `2.7·10⁻²⁸⁰`.
    fn tails_from_small_arg(w: Scaled) -> Tails {
        let mut t = Tails::from_lower(w);
        let ln_w = t.ln_lower;
        if ln_w > -1.0 {
            let other = -ln_w.exp_m1();
            t.upper = other;
            t.ln_upper = other.ln();
        }
        t
    }

    /// The beta arguments of the F distribution function at `x`,
    /// `z = d₁x/(d₁x + d₂)` and `y = d₂/(d₁x + d₂)`, with their logarithms
    /// `ln z = −ln(1 + 1/t)`, `ln y = −ln(1 + t)` from `t = d₁x/d₂` — finite
    /// where `d₁x` or `d₁x + d₂` under- or overflows and `z` or `y` is not a
    /// normal double.
    pub(super) struct BetaArgs {
        pub(super) z: f64,
        pub(super) y: f64,
        pub(super) ln_z: f64,
        pub(super) ln_y: f64,
    }

    impl BetaArgs {
        pub(super) fn new(x: f64, d1: f64, d2: f64) -> BetaArgs {
            let num = d1 * x;
            let den = num + d2;
            let in_range = |v: f64| v.is_finite() && v >= f64::MIN_POSITIVE;
            if in_range(num) && den.is_finite() {
                let (z, y) = (num / den, d2 / den);
                if in_range(z) && in_range(y) {
                    return BetaArgs {
                        z,
                        y,
                        ln_z: z.ln(),
                        ln_y: y.ln(),
                    };
                }
            }
            // t = d₁x/d₂ in whichever order keeps it in range, else only its
            // logarithm.
            let ratio = d1 / d2;
            let t = if in_range(num) && in_range(num / d2) {
                num / d2
            } else if in_range(ratio) && in_range(x * ratio) {
                x * ratio
            } else if in_range(x / d2) && in_range(x / d2 * d1) {
                x / d2 * d1
            } else {
                f64::NAN
            };
            let ln_t = if t.is_nan() {
                x.ln() + d1.ln() - d2.ln()
            } else {
                t.ln()
            };
            // ln y = −ln(1 + t), ln z = ln t − ln(1 + t), each from the
            // form that neither cancels nor overflows.
            let (ln_y, ln_z) = if ln_t <= 0.0 {
                let l1p = ln_t.exp().ln_1p();
                (-l1p, ln_t - l1p)
            } else {
                let l1p = (-ln_t).exp().ln_1p();
                (-ln_t - l1p, -l1p)
            };
            BetaArgs {
                z: ln_z.exp(),
                y: ln_y.exp(),
                ln_z,
                ln_y,
            }
        }
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
        // Below every positive float: the smallest float with F(x) ≥ p (see
        // `beta::quantile`).
        // Beyond the largest float a tiny d₂ has a tail heavier than any
        // power we can hold: +∞, as for `t`.
        let tiniest = f64::from_bits(1);
        if let Some(x) = outside_float_range(
            side,
            tails(tiniest, d1, d2),
            tiniest,
            tails(f64::MAX, d1, d2),
            f64::INFINITY,
        ) {
            return Ok(x);
        }
        let (a, b) = (0.5 * d1, 0.5 * d2);
        // logit(b) = ln(b/(1 − b)); x = (d₂/d₁) e^{logit}.
        let u0 = beta::start_logit(side, a, b)? + (d2 / d1).ln();
        let (target, lower) = match side {
            Side::Lower(pl) => (pl, true),
            Side::Upper(q) => (q, false),
        };
        let ln_target = target.ln();
        // ln(a·B(a, b)) to its own relative accuracy: of order `a` for a tiny
        // `a`, where `ln a + ln B(a, b)` cancels to an absolute error
        // `ε·|ln a|` (88·ε at a = 4e-39, beside a value of 6.5e-38).
        let (ln_r, ln_ab) = ((d2 / d1).ln(), -inv_a_beta(a, b).ln());
        let g = |u: f64| {
            if u < -690.0 && ln_r > u + 40.0 {
                // `x = eᵘ` is at or below the subnormals and negligible
                // beside r = d₂/d₁: ln I_z(a, b) = a·(u − ln r) − ln(a·B(a, b))
                // + O(z), exactly linear in u (as for `gamma`).
                let ln_lower = a * (u - ln_r) - ln_ab;
                return if lower {
                    Eval {
                        g: ln_lower - ln_target,
                        dg: a,
                    }
                } else {
                    upper_from_leading_lower(ln_lower, a, target, ln_target)
                };
            }
            let x = u.exp();
            let num = d1 * x;
            let den = num + d2;
            let (xb, yb) =
                if (num < f64::MIN_POSITIVE || den.is_infinite()) && (d2 / d1).is_finite() {
                    // Subnormal or huge `x`: form the beta arguments without
                    // `d₁x`, which under- or overflows.
                    let r = d2 / d1;
                    (x / (x + r), r / (x + r))
                } else {
                    (num / den, d2 / den)
                };
            // `tails` handles an underflowed argument (leading term).
            let tl = tails(x, d1, d2);
            let (tail, ln_tail) = if lower {
                (tl.lower, tl.ln_lower)
            } else {
                (tl.upper, tl.ln_upper)
            };
            let r = log_ratio(tail, ln_tail, target, ln_target);
            let gv = if lower { r } else { -r };
            // d ln I/du = x_b^a y_b^b/(B·I); where x_b (or y_b) is below
            // every float the leading term I ∝ x_b^a (y_b^b) gives the slope
            // a (b).
            let dg = if !x.is_finite() {
                // `eᵘ` overflowed: no slope to speak of, bisect.
                f64::NAN
            } else if xb > 0.0 && yb > 0.0 {
                (log_beta_pref(a, b, xb, yb) - ln_tail).exp()
            } else if xb > 0.0 {
                b
            } else {
                a
            };
            Eval { g: gv, dg }
        };
        let x = solve_increasing(op, g, u0, f64::NEG_INFINITY, f64::INFINITY)?.exp();
        Ok(snap_to_floats(x, side, |x| tails(x, d1, d2)))
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
        Tails, bratio, check_lattice_param, check_level, discrete_quantile, invalid, nearest_tail,
        nearest_tail_upper, norm,
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
        discrete_quantile(OP, |k| tails(k, n, p), nearest_tail(q), k0, 0.0, n)
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
        discrete_quantile(OP, |k| tails(k, n, p), nearest_tail_upper(q), k0, 0.0, n)
    }
}

/// The Poisson distribution with rate `λ > 0`.
pub mod poisson {
    use super::{
        Tails, check_lattice_param, check_level, check_positive, discrete_quantile, gammainc_tails,
        nearest_tail, nearest_tail_upper, norm,
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
        discrete_quantile(
            OP,
            |k| tails(k, rate),
            nearest_tail(q),
            k0,
            0.0,
            f64::INFINITY,
        )
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
        discrete_quantile(
            OP,
            |k| tails(k, rate),
            nearest_tail_upper(q),
            k0,
            0.0,
            f64::INFINITY,
        )
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
            let lo = grat_r(
                a,
                1.099_999_999,
                Scaled::exp(log_gamma_pref(a, 1.099_999_999)),
            );
            let hi = grat_r(a, 1.1, Scaled::exp(log_gamma_pref(a, 1.1)));
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

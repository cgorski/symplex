//! Shared `f64` runtime for special functions.
//!
//! This module is used in **two** ways:
//!
//! 1. It is compiled into symplex and called by the stack VM behind
//!    [`Ex::compile`](crate::api::expr::Expr::compile).
//! 2. Its source text is embedded (via `include_str!`) into generated Rust
//!    code as a `mod symplex_rt { … }` preamble, so generated code and
//!    compiled closures use *identical* algorithms.
//!
//! Sections are delimited by `// @@begin NAME` / `// @@end NAME` markers.
//! The code generator extracts only the sections that are actually used
//! (plus their transitive dependencies).  The `prim` section is regenerated
//! per math backend (std / libm / cfg-gated); everything else is emitted
//! verbatim.  For that reason this file must stay free of crate-internal
//! imports, `std`-only APIs outside the `prim` section, heap allocation, and
//! recursion over data structures.
//!
//! Accuracy notes (relative error unless stated):
//!
//! | function | method | accuracy |
//! |----------|--------|----------|
//! | `gamma` | exact products (integers, half-integers), Lanczos (g=607/128, n=15), reflection | ≤ 2e-15 |
//! | `lgamma` | ln Γ for 0.5 ≤ x < 20, Stirling for x ≥ 20, reflection | ≈ 1e-15 absolute; relative accuracy degrades near the zeros x = 1, 2 |
//! | `digamma` | recurrence to x ≥ 10, asymptotic series through x⁻¹⁴ | ≈ 1e-15 absolute |
//! | `erf`, `erfc` | W. J. Cody's rational Chebyshev approximations | ≈ 1e-16, `erfc` keeps relative accuracy for large x |
//! | `erfinv`, `erfcinv` | Maclaurin / Winitzki start + Halley on `erf`/`erfc`, Newton on `ln erfc` via erfcx in the deep tail | ≤ 3e-16 (relative accuracy kept down to subnormal `erfcinv` arguments) |
//! | `lambert_w0` | branch-point series + Halley iteration | ≈ 1e-15 (ill-conditioned near −1/e) |
//! | `bessel_j/y` | power series, Miller backward recurrence, Hankel asymptotics | ≈ 1e-14 (absolute near zeros) |
//! | `bessel_i` | power series / asymptotic expansion | ≈ 1e-14 |
//! | `bessel_k` | trapezoidal integral representation / asymptotic expansion | ≈ 1e-14 |
//! | orthogonal polynomials | three-term recurrences | a few ulps per step |
//! | `fibonacci`, `lucas` | fast doubling (exact ≤ 2⁵³), Binet beyond | exact / ≈ 1e-15 |
//! | `harmonic` | direct sum (n ≤ 100), ψ(n+1)+γ beyond | ≈ 1e-15 |

#![allow(clippy::excessive_precision)]
#![allow(clippy::manual_range_contains)]

// @@begin prim
#[inline(always)]
fn p_exp(x: f64) -> f64 {
    x.exp()
}
#[inline(always)]
fn p_ln(x: f64) -> f64 {
    x.ln()
}
#[inline(always)]
fn p_sin(x: f64) -> f64 {
    x.sin()
}
#[inline(always)]
fn p_cos(x: f64) -> f64 {
    x.cos()
}
#[inline(always)]
fn p_sqrt(x: f64) -> f64 {
    x.sqrt()
}
#[inline(always)]
fn p_powf(x: f64, y: f64) -> f64 {
    x.powf(y)
}
#[inline(always)]
fn p_abs(x: f64) -> f64 {
    x.abs()
}
#[inline(always)]
fn p_floor(x: f64) -> f64 {
    x.floor()
}
// @@end prim

// @@begin consts
const PI: f64 = core::f64::consts::PI;
const E: f64 = core::f64::consts::E;
const LN_PI: f64 = 1.1447298858494002;
const LN_SQRT_2PI: f64 = 0.9189385332046727;
const SQRT_2PI: f64 = 2.5066282746310002;
const EULER_GAMMA: f64 = 0.5772156649015329;
const EPS: f64 = f64::EPSILON;
// @@end consts

// Named mathematical constants exposed to the code generators (outside the
// embeddable sections so they are not duplicated into generated code).
/// Euler–Mascheroni constant γ as an `f64`.
pub(crate) const EULER_GAMMA_F64: f64 = EULER_GAMMA;
/// Catalan's constant G as an `f64`.
pub(crate) const CATALAN_F64: f64 = 0.915_965_594_177_219_0;
/// Golden ratio φ = (1 + √5)/2 as an `f64`.
pub(crate) const GOLDEN_RATIO_F64: f64 = 1.618_033_988_749_895;

// @@begin util
/// True when `x` is a finite integer value.
fn is_int(x: f64) -> bool {
    x.is_finite() && x == p_floor(x)
}
/// True when the integer-valued `n` is even.
fn is_even(n: f64) -> bool {
    let h = n * 0.5;
    h == p_floor(h)
}
/// sin(πx) with exact argument reduction (accurate for large |x|).
fn sin_pi(x: f64) -> f64 {
    let n = p_floor(x + 0.5);
    let r = x - n;
    let s = p_sin(PI * r);
    if is_even(n) { s } else { -s }
}
/// cos(πx) with exact argument reduction.
fn cos_pi(x: f64) -> f64 {
    let n = p_floor(x + 0.5);
    let r = x - n;
    let c = p_cos(PI * r);
    if is_even(n) { c } else { -c }
}
/// Sign of Γ(x) for non-pole `x`: +1 for x > 0, alternating on negative unit intervals.
fn gamma_sign(x: f64) -> f64 {
    if x > 0.0 || is_even(p_floor(x)) {
        1.0
    } else {
        -1.0
    }
}
/// True when `x` is a pole of Γ (a non-positive integer).
fn is_gamma_pole(x: f64) -> bool {
    x <= 0.0 && is_int(x)
}
// @@end util

// @@begin gamma
/// Γ(x) for real x.
///
/// Integers and half-integers use exact products; otherwise the Lanczos
/// approximation (Godfrey's g = 607/128, n = 15 coefficients) with the
/// reflection formula for x < 0.5.
/// Poles (non-positive integers) return NaN; ±0 returns ±∞.
pub fn gamma(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }
    if is_int(x) {
        if x < 0.0 {
            return f64::NAN;
        }
        if x > 171.0 {
            return f64::INFINITY;
        }
        let mut acc = 1.0;
        let mut k = 2.0;
        while k < x {
            acc *= k;
            k += 1.0;
        }
        return acc;
    }
    if is_int(2.0 * x) && p_abs(x) < 170.0 {
        // Half-integers: Γ(k + 1/2) = √π · ∏_{j<k} (j + 1/2), extended downwards by
        // Γ(x) = Γ(x + 1)/x.  Products of exactly representable factors.
        const SQRT_PI: f64 = 1.7724538509055159;
        let mut acc = SQRT_PI;
        if x > 0.0 {
            let mut j = 0.5;
            while j < x {
                acc *= j;
                j += 1.0;
            }
        } else {
            let mut j = x;
            while j < 0.0 {
                acc /= j;
                j += 1.0;
            }
        }
        return acc;
    }
    if x < 0.5 {
        return PI / (sin_pi(x) * lanczos_gamma(1.0 - x));
    }
    if x > 171.7 {
        return f64::INFINITY;
    }
    lanczos_gamma(x)
}
/// Lanczos approximation (Godfrey's g = 607/128, n = 15 coefficients),
/// valid for x ≥ 0.5 with relative error ≤ 2e-15 over the whole range.
fn lanczos_gamma(x: f64) -> f64 {
    const G: f64 = 607.0 / 128.0;
    const C: [f64; 15] = [
        0.99999999999999709182,
        57.156235665862923517,
        -59.597960355475491248,
        14.136097974741747174,
        -0.49191381609762019978,
        0.33994649984811888699e-4,
        0.46523628927048575665e-4,
        -0.98374475304879564677e-4,
        0.15808870322491248884e-3,
        -0.21026444172410488319e-3,
        0.21743961811521264320e-3,
        -0.16431810653676389022e-3,
        0.84418223983852743293e-4,
        -0.26190838401581408670e-4,
        0.36899182659531622704e-5,
    ];
    if x > 171.7 {
        return f64::INFINITY;
    }
    let z = x - 1.0;
    let mut a = C[0];
    let mut i = 1;
    while i < 15 {
        a += C[i] / (z + i as f64);
        i += 1;
    }
    let t = z + G + 0.5;
    let half = p_powf(t, (z + 0.5) * 0.5);
    SQRT_2PI * half * (half * p_exp(-t)) * a
}
// @@end gamma

// @@begin lgamma
/// ln|Γ(x)|.
///
/// Poles return +∞.  Uses ln Γ(x) for 0.5 ≤ x < 20 (absolute error ≈ 1e-15,
/// so relative accuracy degrades near the zeros x = 1 and x = 2), the
/// Stirling series for x ≥ 20 and the reflection formula for x < 0.5.
pub fn lgamma(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }
    if is_gamma_pole(x) {
        return f64::INFINITY;
    }
    if x < 0.5 {
        return LN_PI - p_ln(p_abs(sin_pi(x))) - lgamma_pos(1.0 - x);
    }
    lgamma_pos(x)
}
/// ln Γ(x) for x ≥ 0.5.
fn lgamma_pos(x: f64) -> f64 {
    if x < 20.0 {
        return p_ln(gamma(x));
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    let series = inv
        * (1.0 / 12.0
            - inv2
                * (1.0 / 360.0
                    - inv2 * (1.0 / 1260.0 - inv2 * (1.0 / 1680.0 - inv2 * (1.0 / 1188.0)))));
    (x - 0.5) * p_ln(x) - x + LN_SQRT_2PI + series
}
// @@end lgamma

// @@begin digamma
/// Digamma function ψ(x) = Γ'(x)/Γ(x).
///
/// Reflection for x < 0, recurrence up to x ≥ 10, then the asymptotic series
/// through x⁻¹⁴ (truncation error < 5e-17).  Poles return NaN.
pub fn digamma(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    if is_gamma_pole(x) {
        return f64::NAN;
    }
    let mut result = 0.0;
    let mut x = x;
    if x < 0.0 {
        // ψ(x) = ψ(1 − x) − π cot(πx)
        result -= PI * cos_pi(x) / sin_pi(x);
        x = 1.0 - x;
    }
    while x < 10.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    let series = inv2
        * (1.0 / 12.0
            - inv2
                * (1.0 / 120.0
                    - inv2
                        * (1.0 / 252.0
                            - inv2
                                * (1.0 / 240.0
                                    - inv2
                                        * (1.0 / 132.0
                                            - inv2 * (691.0 / 32760.0 - inv2 * (1.0 / 12.0)))))));
    result + p_ln(x) - 0.5 * inv - series
}
// @@end digamma

// @@begin erf
/// Error function erf(x) (W. J. Cody's rational approximations, ≈ 1e-16).
pub fn erf(x: f64) -> f64 {
    calerf(x, false)
}
/// Complementary error function erfc(x) = 1 − erf(x).
///
/// Retains full relative accuracy for large x (down to the underflow
/// threshold near x ≈ 26.5).
pub fn erfc(x: f64) -> f64 {
    calerf(x, true)
}
/// Cody's asymptotic rational approximation for the scaled complementary
/// error function on |x| > 4 (shared by `calerf` and `erfcx_large`).
const ERFC_P: [f64; 6] = [
    3.05326634961232344e-1,
    3.60344899949804439e-1,
    1.25781726111229246e-1,
    1.60837851487422766e-2,
    6.58749161529837803e-4,
    1.63153871373020978e-2,
];
const ERFC_Q: [f64; 5] = [
    2.56852019228982242e00,
    1.87295284992346725e00,
    5.27905102951428412e-1,
    6.05183413124413191e-2,
    2.33520497626869185e-3,
];
/// 1/√π.
const INV_SQRT_PI: f64 = 5.6418958354775628695e-1;
/// Scaled complementary error function erfcx(x) = e^{x²}·erfc(x) for x > 4.
///
/// Does not underflow (erfcx(x) ~ 1/(x√π)), which is what the deep tail of
/// `erfcinv` needs where `erfc` itself is subnormal or zero.
fn erfcx_large(x: f64) -> f64 {
    let ysq = 1.0 / (x * x);
    let mut xnum = ERFC_P[5] * ysq;
    let mut xden = ysq;
    let mut i = 0;
    while i < 4 {
        xnum = (xnum + ERFC_P[i]) * ysq;
        xden = (xden + ERFC_Q[i]) * ysq;
        i += 1;
    }
    let result = ysq * (xnum + ERFC_P[4]) / (xden + ERFC_Q[4]);
    (INV_SQRT_PI - result) / x
}
/// Cody's CALERF: `complement == false` → erf, `true` → erfc.
fn calerf(x: f64, complement: bool) -> f64 {
    const A: [f64; 5] = [
        3.16112374387056560e00,
        1.13864154151050156e02,
        3.77485237685302021e02,
        3.20937758913846947e03,
        1.85777706184603153e-1,
    ];
    const B: [f64; 4] = [
        2.36012909523441209e01,
        2.44024637934444173e02,
        1.28261652607737228e03,
        2.84423683343917062e03,
    ];
    const C: [f64; 9] = [
        5.64188496988670089e-1,
        8.88314979438837594e00,
        6.61191906371416295e01,
        2.98635138197400131e02,
        8.81952221241769090e02,
        1.71204761263407058e03,
        2.05107837782607147e03,
        1.23033935479799725e03,
        2.15311535474403846e-8,
    ];
    const D: [f64; 8] = [
        1.57449261107098347e01,
        1.17693950891312499e02,
        5.37181101862009858e02,
        1.62138957456669019e03,
        3.29079923573345963e03,
        4.36261909014324716e03,
        3.43936767414372164e03,
        1.23033935480374942e03,
    ];
    const THRESH: f64 = 0.46875;
    const XSMALL: f64 = 1.11e-16;
    const XBIG: f64 = 26.543;

    if x.is_nan() {
        return f64::NAN;
    }
    let y = p_abs(x);
    if y <= THRESH {
        let ysq = if y > XSMALL { y * y } else { 0.0 };
        let mut xnum = A[4] * ysq;
        let mut xden = ysq;
        let mut i = 0;
        while i < 3 {
            xnum = (xnum + A[i]) * ysq;
            xden = (xden + B[i]) * ysq;
            i += 1;
        }
        let result = x * (xnum + A[3]) / (xden + B[3]);
        return if complement { 1.0 - result } else { result };
    }
    let mut result;
    if y <= 4.0 {
        let mut xnum = C[8] * y;
        let mut xden = y;
        let mut i = 0;
        while i < 7 {
            xnum = (xnum + C[i]) * y;
            xden = (xden + D[i]) * y;
            i += 1;
        }
        result = (xnum + C[7]) / (xden + D[7]);
        let ysq = p_floor(y * 16.0) / 16.0;
        let del = (y - ysq) * (y + ysq);
        result *= p_exp(-ysq * ysq) * p_exp(-del);
    } else {
        result = 0.0;
        if y < XBIG {
            result = erfcx_large(y);
            let ysq = p_floor(y * 16.0) / 16.0;
            let del = (y - ysq) * (y + ysq);
            result *= p_exp(-ysq * ysq) * p_exp(-del);
        }
    }
    // `result` now holds erfc(|x|).
    if complement {
        if x < 0.0 { 2.0 - result } else { result }
    } else {
        let r = (0.5 - result) + 0.5;
        if x < 0.0 { -r } else { r }
    }
}
// @@end erf

// @@begin erfinv
/// √π/2.
const HALF_SQRT_PI: f64 = 0.88622692545275801365;
/// Maclaurin coefficients of erfinv: `erfinv(x) = Σ ERFINV_SERIES[k]·w^{2k+1}`
/// with w = (√π/2)·x (OEIS A092676/A092677: 1, 1/3, 7/30, 127/630, …).
const ERFINV_SERIES: [f64; 8] = [
    1.0,
    0.33333333333333333333,
    0.23333333333333333333,
    0.20158730158730158730,
    0.19263668430335097002,
    0.19532547699214365881,
    0.20593586454697565891,
    0.22320975741875211778,
];
/// Inverse error function: erf(erfinv(x)) = x for −1 < x < 1.
///
/// ±1 map to ±∞, |x| > 1 to NaN.  For |x| ≤ 1/2 an 8-term Maclaurin series
/// (relative error ≤ 7e-7) is polished by Halley iterations on
/// `erf(z) − x`; for |x| > 1/2 the problem is handed to the `erfc`-based
/// tail solver with the *exact* complement 1 − |x| (Sterbenz), so that the
/// accuracy near ±1 is set by `erfc`'s relative accuracy rather than by
/// cancellation in `erf(z) − x`.  Relative error ≤ 3e-16 across the range.
pub fn erfinv(x: f64) -> f64 {
    if x.is_nan() || x < -1.0 || x > 1.0 {
        return f64::NAN;
    }
    if x == 1.0 {
        return f64::INFINITY;
    }
    if x == -1.0 {
        return f64::NEG_INFINITY;
    }
    let a = p_abs(x);
    if a <= 0.5 {
        return erfinv_central(x);
    }
    let z = erfcinv_tail(1.0 - a);
    if x < 0.0 { -z } else { z }
}
/// Inverse complementary error function: erfc(erfcinv(y)) = y for 0 < y < 2.
///
/// 0 maps to +∞, 2 to −∞, values outside [0, 2] to NaN.  Keeps full relative
/// accuracy for tiny y (down to the smallest subnormal, erfcinv ≈ 27.2)
/// by iterating on `ln erfc(z)` with the scaled function erfcx.
pub fn erfcinv(y: f64) -> f64 {
    if y.is_nan() || y < 0.0 || y > 2.0 {
        return f64::NAN;
    }
    if y == 0.0 {
        return f64::INFINITY;
    }
    if y == 2.0 {
        return f64::NEG_INFINITY;
    }
    if y > 1.5 {
        // 2 − y is exact for y ∈ [1, 2].
        return -erfcinv_tail(2.0 - y);
    }
    if y >= 0.5 {
        // 1 − y is exact for y ∈ [1/2, 3/2].
        return erfinv_central(1.0 - y);
    }
    erfcinv_tail(y)
}
/// erfinv(x) for |x| ≤ 1/2: Maclaurin start, then Halley on erf(z) − x.
fn erfinv_central(x: f64) -> f64 {
    let w = HALF_SQRT_PI * x;
    let w2 = w * w;
    let mut p = ERFINV_SERIES[7];
    let mut i = 7;
    while i > 0 {
        i -= 1;
        p = ERFINV_SERIES[i] + p * w2;
    }
    let mut z = w * p;
    // Halley: f = erf(z) − x, f' = (2/√π)e^{−z²}, f'' = −2z f'
    //   z ← z − d / (1 + z·d)  with  d = f / f'.
    let mut iter = 0;
    while iter < 8 {
        let f = erf(z) - x;
        let d = f * HALF_SQRT_PI * p_exp(z * z);
        let dz = d / (1.0 + z * d);
        z -= dz;
        if p_abs(dz) <= 2.0 * EPS * p_abs(z) {
            break;
        }
        iter += 1;
    }
    z
}
/// erfcinv(y) for 0 < y < 1/2 (so z > 0.47).
///
/// Starts from Winitzki's closed-form approximation (relative error ≤ 2e-3)
/// and refines with Halley on erfc(z) − y while z < 4; beyond that
/// (y < erfc(4) ≈ 1.5e-8) Newton on ln erfc(z) − ln y using erfcx, which
/// neither underflows nor cancels.
fn erfcinv_tail(y: f64) -> f64 {
    // Winitzki (2008): erfinv(x) ≈ sqrt(sqrt((2/(πa) + L/2)² − L/a) − (2/(πa) + L/2))
    // with L = ln(1 − x²) = ln y + ln(2 − y), a = 0.147.
    let l = p_ln(y) + p_ln(2.0 - y);
    let t = 2.0 / (PI * 0.147) + 0.5 * l;
    let mut z = p_sqrt(p_sqrt(t * t - l / 0.147) - t);
    if y > 1.5e-8 {
        // Halley on f = erfc(z) − y (f' = −(2/√π)e^{−z²}, f'' = −2z f').
        let mut iter = 0;
        while iter < 12 {
            let f = erfc(z) - y;
            let d = -f * HALF_SQRT_PI * p_exp(z * z);
            let dz = d / (1.0 + z * d);
            z -= dz;
            if p_abs(dz) <= 2.0 * EPS * p_abs(z) {
                break;
            }
            iter += 1;
        }
    } else {
        // Newton on g = ln erfcx(z) − z² − ln y,  g' = −(2/√π)/erfcx(z).
        let ln_y = p_ln(y);
        let mut iter = 0;
        while iter < 12 {
            let ex = erfcx_large(z);
            let g = p_ln(ex) - z * z - ln_y;
            let dz = -g * ex * HALF_SQRT_PI;
            z -= dz;
            if p_abs(dz) <= 4.0 * EPS * p_abs(z) {
                break;
            }
            iter += 1;
        }
    }
    z
}
// @@end erfinv

// @@begin lambert_w0
/// Principal branch of the Lambert W function, W₀(x)·exp(W₀(x)) = x.
///
/// Defined for x ≥ −1/e (NaN otherwise).  Uses the branch-point series near
/// −1/e and Halley iteration elsewhere.  Note that W₀ is ill-conditioned
/// near the branch point: the achievable accuracy there is limited by the
/// input rounding, not by this routine.
pub fn lambert_w0(x: f64) -> f64 {
    const NEG_INV_E: f64 = -0.36787944117144233;
    if x.is_nan() || x < NEG_INV_E {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    if x == 0.0 {
        return x;
    }
    if x == NEG_INV_E {
        return -1.0;
    }
    let mut w;
    if x < -0.25 {
        // Branch-point series in p = sqrt(2(e·x + 1)).
        let p = p_sqrt(2.0 * (E * x + 1.0));
        w = -1.0
            + p * (1.0
                - p * (1.0 / 3.0
                    - p * (11.0 / 72.0
                        - p * (43.0 / 540.0
                            - p * (769.0 / 17280.0
                                - p * (221.0 / 8505.0
                                    - p * (680863.0 / 43545600.0 - p * (1963.0 / 204120.0))))))));
        if p < 0.02 {
            return w;
        }
    } else if x < E {
        w = p_ln(1.0 + x);
    } else {
        let l1 = p_ln(x);
        let l2 = p_ln(l1);
        w = l1 - l2 + l2 / l1;
    }
    let mut iter = 0;
    while iter < 40 {
        let ew = p_exp(w);
        let f = w * ew - x;
        let wp1 = w + 1.0;
        let denom = ew * wp1 - (w + 2.0) * f / (2.0 * wp1);
        let dw = f / denom;
        w -= dw;
        if p_abs(dw) <= 2.0 * EPS * (p_abs(w) + 1e-300) {
            break;
        }
        iter += 1;
    }
    w
}
// @@end lambert_w0

// @@begin beta
/// Beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b).
///
/// Poles of Γ(a) or Γ(b) return NaN; a pole of Γ(a+b) alone returns 0.
pub fn beta(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    let s = a + b;
    if is_gamma_pole(a) || is_gamma_pole(b) {
        return f64::NAN;
    }
    if is_gamma_pole(s) {
        return 0.0;
    }
    if a > 0.0 && b > 0.0 && s < 171.0 {
        return gamma(a) * gamma(b) / gamma(s);
    }
    let sign = gamma_sign(a) * gamma_sign(b) * gamma_sign(s);
    sign * p_exp(lgamma(a) + lgamma(b) - lgamma(s))
}
// @@end beta

// @@begin factorial
/// Factorial x! = Γ(x + 1) (exact for integers ≤ 22, ≈ 1e-15 beyond).
pub fn factorial(x: f64) -> f64 {
    gamma(x + 1.0)
}
// @@end factorial

// @@begin binomial
/// Generalised binomial coefficient C(n, k) = Γ(n+1)/(Γ(k+1)Γ(n−k+1)).
///
/// Integer k ≥ 0 uses the (exact for integer n) multiplicative formula,
/// negative integer k gives 0, other cases go through Γ/ln Γ.
pub fn binomial(n: f64, k: f64) -> f64 {
    if n.is_nan() || k.is_nan() {
        return f64::NAN;
    }
    if is_int(k) {
        if k < 0.0 {
            return 0.0;
        }
        let mut kk = k;
        if is_int(n) && n >= 0.0 {
            if k > n {
                return 0.0;
            }
            if n - k < k {
                kk = n - k;
            }
        }
        if kk <= 2000.0 {
            let mut acc = 1.0;
            let mut i = 1.0;
            while i <= kk {
                acc = acc * (n - kk + i) / i;
                i += 1.0;
            }
            return acc;
        }
    }
    let a = n + 1.0;
    let b = k + 1.0;
    let c = n - k + 1.0;
    if is_gamma_pole(b) || is_gamma_pole(c) {
        return if is_gamma_pole(a) { f64::NAN } else { 0.0 };
    }
    if is_gamma_pole(a) {
        return f64::NAN;
    }
    if a > 0.0 && b > 0.0 && c > 0.0 && a < 171.0 {
        return gamma(a) / (gamma(b) * gamma(c));
    }
    let sign = gamma_sign(a) * gamma_sign(b) * gamma_sign(c);
    sign * p_exp(lgamma(a) - lgamma(b) - lgamma(c))
}
// @@end binomial

// @@begin bessel_core
/// Power series for J_n(x), n ≥ 0, x ≥ 0; terms decrease monotonically
/// (no cancellation) when x ≤ 2·sqrt(n+1).
fn bessel_j_series(n: i32, x: f64) -> f64 {
    let half = 0.5 * x;
    let mut term = 1.0;
    let mut i = 1;
    while i <= n {
        term *= half / i as f64;
        i += 1;
    }
    if term == 0.0 {
        return 0.0;
    }
    let h2 = half * half;
    let mut sum = term;
    let mut k = 1.0;
    while k < 500.0 {
        term *= -h2 / (k * (n as f64 + k));
        sum += term;
        if p_abs(term) <= EPS * p_abs(sum) {
            break;
        }
        k += 1.0;
    }
    sum
}
/// Miller's backward recurrence for x > 0.  Returns `(J_n, Y_0, Y_1)`, with
/// Y_0 and Y_1 obtained from the Neumann series over the same J values.
fn bessel_miller(n: i32, x: f64) -> (f64, f64, f64) {
    let nmax = if n as f64 > x { n } else { x as i32 };
    let mut m = nmax + 20 + p_sqrt(40.0 * nmax as f64) as i32;
    if m % 2 == 1 {
        m += 1;
    }
    let two_over_x = 2.0 / x;
    let mut bjp = 0.0; // J_{j+1}
    let mut bj = 1.0; // J_j
    let mut sum = 2.0; // J_0 + 2 Σ J_{2k}  (starts with 2·J_m)
    let mk = (m / 2) as f64;
    let mut s0 = if (m / 2) % 2 == 0 {
        1.0 / mk
    } else {
        -1.0 / mk
    }; // Σ (−1)^k J_{2k}/k
    let mut s1 = 0.0; // Σ (−1)^k (J_{2k−1} − J_{2k+1})/k
    let mut last_odd = 0.0; // J_{2k+1} (the previously seen odd order)
    let mut jn = if n == m { 1.0 } else { 0.0 };
    let mut j0 = 0.0;
    let mut j1 = 0.0;
    let mut j = m;
    while j >= 1 {
        let bjm = (j as f64 * two_over_x) * bj - bjp;
        bjp = bj;
        bj = bjm;
        if p_abs(bj) > 1e250 {
            let scale = 1e-250;
            bj *= scale;
            bjp *= scale;
            sum *= scale;
            s0 *= scale;
            s1 *= scale;
            last_odd *= scale;
            jn *= scale;
            j0 *= scale;
            j1 *= scale;
        }
        let order = j - 1;
        if order == n {
            jn = bj;
        }
        if order % 2 == 0 {
            if order == 0 {
                sum += bj;
                j0 = bj;
            } else {
                sum += 2.0 * bj;
                let k = (order / 2) as f64;
                if (order / 2) % 2 == 0 {
                    s0 += bj / k;
                } else {
                    s0 -= bj / k;
                }
            }
        } else {
            let kk = (order + 1) / 2;
            let k = kk as f64;
            if kk % 2 == 0 {
                s1 += (bj - last_odd) / k;
            } else {
                s1 -= (bj - last_odd) / k;
            }
            last_odd = bj;
            if order == 1 {
                j1 = bj;
            }
        }
        j -= 1;
    }
    let inv = 1.0 / sum;
    let jn = jn * inv;
    let j0 = j0 * inv;
    let j1 = j1 * inv;
    let s0 = s0 * inv;
    let s1 = s1 * inv;
    let lg = p_ln(0.5 * x) + EULER_GAMMA;
    let two_over_pi = 2.0 / PI;
    let y0 = two_over_pi * (lg * j0 - 2.0 * s0);
    let y1 = two_over_pi * (lg * j1 - j0 / x + s1);
    (jn, y0, y1)
}
/// Hankel asymptotic expansion for J_n(x), Y_n(x) with x ≥ 25.
/// Returns `None` when the series does not converge to ≈1e-16 (n too large).
fn bessel_hankel(n: i32, x: f64) -> Option<(f64, f64)> {
    let mu = 4.0 * (n as f64) * (n as f64);
    let mut p = 1.0;
    let mut q = 0.0;
    let mut term = 1.0;
    let mut prev = f64::INFINITY;
    let mut k = 1;
    let mut converged = false;
    while k < 80 {
        let kf = k as f64;
        term *= (mu - (2.0 * kf - 1.0) * (2.0 * kf - 1.0)) / (kf * 8.0 * x);
        if p_abs(term) >= prev {
            break;
        }
        prev = p_abs(term);
        if k % 2 == 1 {
            if (k / 2) % 2 == 0 {
                q += term;
            } else {
                q -= term;
            }
        } else if (k / 2) % 2 == 0 {
            p += term;
        } else {
            p -= term;
        }
        if prev < 1e-17 {
            converged = true;
            break;
        }
        k += 1;
    }
    if !converged && prev > 1e-15 {
        return None;
    }
    let chi = x - (0.5 * n as f64 + 0.25) * PI;
    let c = p_cos(chi);
    let s = p_sin(chi);
    let pref = p_sqrt(2.0 / (PI * x));
    Some((pref * (p * c - q * s), pref * (p * s + q * c)))
}
/// Power series for Y_0(x) and Y_1(x), 0 < x ≤ 2.
fn bessel_y01_series(x: f64) -> (f64, f64) {
    let half = 0.5 * x;
    let h2 = half * half;
    let lg = p_ln(half);
    let j0 = bessel_j_series(0, x);
    let j1 = bessel_j_series(1, x);
    // Y_0 = (2/π)[(ln(x/2)+γ) J_0 + Σ_{k≥1} (−1)^{k+1} H_k (x²/4)^k/(k!)²]
    let mut term = 1.0;
    let mut hk = 0.0;
    let mut s0 = 0.0;
    // Y_1 = −2/(πx) + (2/π) ln(x/2) J_1 − (1/π)(x/2) Σ_{k≥0} (−1)^k (H_k + H_{k+1} − 2γ)(x²/4)^k/(k!(k+1)!)
    let mut term1 = 1.0;
    let mut s1 = 0.0;
    let mut k = 0.0;
    while k < 200.0 {
        // Y_1 term for index k (H_k, H_{k+1} known)
        let hk1 = hk + 1.0 / (k + 1.0);
        let t1 = term1 * (hk + hk1 - 2.0 * EULER_GAMMA);
        s1 += t1;
        // advance to k+1
        let kn = k + 1.0;
        term *= h2 / (kn * kn);
        hk = hk1;
        let t0 = term * hk;
        s0 += t0;
        term1 *= -h2 / (kn * (kn + 1.0));
        if p_abs(t0) <= EPS * p_abs(s0) && p_abs(term1) <= EPS * p_abs(s1) {
            break;
        }
        // alternate the sign of the Y_0 series via `term` sign handling
        term = -term;
        k = kn;
    }
    let two_over_pi = 2.0 / PI;
    let y0 = two_over_pi * ((lg + EULER_GAMMA) * j0 + s0);
    let y1 = -two_over_pi / x + two_over_pi * lg * j1 - half * s1 / PI;
    (y0, y1)
}
// @@end bessel_core

// @@begin bessel_j
/// Bessel function of the first kind J_n(x) for integer order n.
pub fn bessel_j(n: i32, x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let mut sign = 1.0;
    let n = if n < 0 {
        if n % 2 != 0 {
            sign = -sign;
        }
        -n
    } else {
        n
    };
    let ax = if x < 0.0 {
        if n % 2 != 0 {
            sign = -sign;
        }
        -x
    } else {
        x
    };
    if ax == 0.0 {
        return if n == 0 { 1.0 } else { 0.0 };
    }
    if ax.is_infinite() {
        return 0.0;
    }
    let v = if ax <= 2.0 * p_sqrt(n as f64 + 1.0) {
        bessel_j_series(n, ax)
    } else if ax < 25.0 {
        bessel_miller(n, ax).0
    } else if let Some((j, _)) = bessel_hankel(n, ax) {
        j
    } else {
        bessel_miller(n, ax).0
    };
    sign * v
}
// @@end bessel_j

// @@begin bessel_y
/// Bessel function of the second kind Y_n(x) for integer order n (x > 0).
pub fn bessel_y(n: i32, x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x.is_infinite() {
        return 0.0;
    }
    let mut sign = 1.0;
    let n = if n < 0 {
        if n % 2 != 0 {
            sign = -sign;
        }
        -n
    } else {
        n
    };
    let (y0, y1) = if x <= 2.0 {
        bessel_y01_series(x)
    } else if x < 25.0 {
        let (_, y0, y1) = bessel_miller(0, x);
        (y0, y1)
    } else {
        match (bessel_hankel(0, x), bessel_hankel(1, x)) {
            (Some((_, y0)), Some((_, y1))) => (y0, y1),
            _ => {
                let (_, y0, y1) = bessel_miller(0, x);
                (y0, y1)
            }
        }
    };
    if n == 0 {
        return sign * y0;
    }
    let mut ym = y0;
    let mut y = y1;
    let mut k = 1;
    while k < n {
        let yp = (2.0 * k as f64 / x) * y - ym;
        ym = y;
        y = yp;
        k += 1;
    }
    sign * y
}
// @@end bessel_y

// @@begin bessel_i
/// Modified Bessel function of the first kind I_n(x) for integer order n.
pub fn bessel_i(n: i32, x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let n = if n < 0 { -n } else { n };
    let mut sign = 1.0;
    let ax = if x < 0.0 {
        if n % 2 != 0 {
            sign = -1.0;
        }
        -x
    } else {
        x
    };
    if ax == 0.0 {
        return if n == 0 { 1.0 } else { 0.0 };
    }
    if ax.is_infinite() {
        return sign * f64::INFINITY;
    }
    if ax > 30.0 {
        // Asymptotic expansion: e^x/sqrt(2πx) Σ (−1)^k a_k(n)/x^k
        let mu = 4.0 * (n as f64) * (n as f64);
        let mut sum = 1.0;
        let mut term = 1.0;
        let mut prev = f64::INFINITY;
        let mut k = 1;
        let mut ok = false;
        while k < 80 {
            let kf = k as f64;
            term *= -(mu - (2.0 * kf - 1.0) * (2.0 * kf - 1.0)) / (kf * 8.0 * ax);
            if p_abs(term) >= prev {
                break;
            }
            prev = p_abs(term);
            sum += term;
            if prev < 1e-17 * p_abs(sum) {
                ok = true;
                break;
            }
            k += 1;
        }
        if ok {
            return sign * p_exp(ax - 0.5 * p_ln(2.0 * PI * ax)) * sum;
        }
    }
    // Power series (all terms positive).
    let half = 0.5 * ax;
    let mut term = 1.0;
    let mut i = 1;
    while i <= n {
        term *= half / i as f64;
        i += 1;
    }
    if term == 0.0 {
        return 0.0;
    }
    let h2 = half * half;
    let mut sum = term;
    let mut k = 1.0;
    while k < 2000.0 {
        term *= h2 / (k * (n as f64 + k));
        sum += term;
        if term <= EPS * sum {
            break;
        }
        k += 1.0;
    }
    sign * sum
}
// @@end bessel_i

// @@begin bessel_k
/// Modified Bessel function of the second kind K_n(x) for integer order n (x > 0).
///
/// Uses the asymptotic expansion for x ≥ 20 (when it converges) and the
/// trapezoidal rule on K_n(x) = ∫₀^∞ e^{−x cosh t} cosh(nt) dt otherwise,
/// which is spectrally accurate for this integrand.
pub fn bessel_k(n: i32, x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x.is_infinite() {
        return 0.0;
    }
    let n = if n < 0 { -n } else { n };
    let nf = n as f64;
    if x >= 20.0 {
        let mu = 4.0 * nf * nf;
        let mut sum = 1.0;
        let mut term = 1.0;
        let mut prev = f64::INFINITY;
        let mut k = 1;
        let mut ok = false;
        while k < 80 {
            let kf = k as f64;
            term *= (mu - (2.0 * kf - 1.0) * (2.0 * kf - 1.0)) / (kf * 8.0 * x);
            if p_abs(term) >= prev {
                break;
            }
            prev = p_abs(term);
            sum += term;
            if prev < 1e-17 * p_abs(sum) {
                ok = true;
                break;
            }
            k += 1;
        }
        if ok {
            return p_sqrt(PI / (2.0 * x)) * p_exp(-x) * sum;
        }
    }
    // Trapezoidal rule on ∫₀^∞ e^{−x cosh t} cosh(nt) dt.  The discretisation
    // error decays like exp(−π²/h) times a factor growing with x and n, so
    // start from a step tuned for sqrt(x² + n²) and halve it until two
    // successive results agree to working precision (a reliable stopping
    // rule because the error is exponentially small in 1/h).
    let scale = p_sqrt(x * x + nf * nf);
    let mut h = PI * PI / (scale + 45.0);
    let mut prev = bessel_k_trapezoid(nf, x, h);
    let mut refinements = 0;
    while refinements < 8 {
        h *= 0.5;
        let cur = bessel_k_trapezoid(nf, x, h);
        if p_abs(cur - prev) <= 4.0 * EPS * p_abs(cur) {
            return cur;
        }
        prev = cur;
        refinements += 1;
    }
    prev
}
/// One trapezoidal-rule evaluation of ∫₀^∞ e^{−x cosh t} cosh(nt) dt with step `h`.
fn bessel_k_trapezoid(nf: f64, x: f64, h: f64) -> f64 {
    // Peak of e^{−x cosh t + n t} lies at sinh t = n/x.
    let r = nf / x;
    let t_peak = p_ln(r + p_sqrt(r * r + 1.0));
    let mut sum = 0.5 * p_exp(-x);
    let mut j = 1;
    while j < 100000 {
        let t = h * j as f64;
        let ch = 0.5 * (p_exp(t) + p_exp(-t));
        let a = -x * ch;
        let f = 0.5 * (p_exp(a + nf * t) + p_exp(a - nf * t));
        sum += f;
        if t > t_peak + 1.0 && f <= 1e-17 * sum {
            break;
        }
        j += 1;
    }
    h * sum
}
// @@end bessel_k

// @@begin orthopoly
/// Legendre polynomial P_n(x) (P_{−n} = P_{n−1}).
pub fn legendre_p(n: i32, x: f64) -> f64 {
    let n = if n < 0 { -n - 1 } else { n };
    if n == 0 {
        return 1.0;
    }
    let mut pm = 1.0;
    let mut p = x;
    let mut k = 1;
    while k < n {
        let kf = k as f64;
        let pn = ((2.0 * kf + 1.0) * x * p - kf * pm) / (kf + 1.0);
        pm = p;
        p = pn;
        k += 1;
    }
    p
}
/// Chebyshev polynomial of the first kind T_n(x) (T_{−n} = T_n).
pub fn chebyshev_t(n: i32, x: f64) -> f64 {
    let n = if n < 0 { -n } else { n };
    if n == 0 {
        return 1.0;
    }
    let mut tm = 1.0;
    let mut t = x;
    let mut k = 1;
    while k < n {
        let tn = 2.0 * x * t - tm;
        tm = t;
        t = tn;
        k += 1;
    }
    t
}
/// Chebyshev polynomial of the second kind U_n(x) (U_{−n} = −U_{n−2}).
pub fn chebyshev_u(n: i32, x: f64) -> f64 {
    if n == -1 {
        return 0.0;
    }
    let (n, sign) = if n < 0 { (-n - 2, -1.0) } else { (n, 1.0) };
    if n == 0 {
        return sign;
    }
    let mut um = 1.0;
    let mut u = 2.0 * x;
    let mut k = 1;
    while k < n {
        let un = 2.0 * x * u - um;
        um = u;
        u = un;
        k += 1;
    }
    sign * u
}
/// Physicists' Hermite polynomial H_n(x), n ≥ 0 (NaN for n < 0).
pub fn hermite_h(n: i32, x: f64) -> f64 {
    if n < 0 {
        return f64::NAN;
    }
    if n == 0 {
        return 1.0;
    }
    let mut hm = 1.0;
    let mut h = 2.0 * x;
    let mut k = 1;
    while k < n {
        let hn = 2.0 * x * h - 2.0 * k as f64 * hm;
        hm = h;
        h = hn;
        k += 1;
    }
    h
}
/// Laguerre polynomial L_n(x), n ≥ 0 (NaN for n < 0).
pub fn laguerre_l(n: i32, x: f64) -> f64 {
    if n < 0 {
        return f64::NAN;
    }
    if n == 0 {
        return 1.0;
    }
    let mut lm = 1.0;
    let mut l = 1.0 - x;
    let mut k = 1;
    while k < n {
        let kf = k as f64;
        let ln = ((2.0 * kf + 1.0 - x) * l - kf * lm) / (kf + 1.0);
        lm = l;
        l = ln;
        k += 1;
    }
    l
}
// @@end orthopoly

// @@begin fibonacci
/// `(F_m, F_{m+1})` by fast doubling.
///
/// Exact 128-bit integer arithmetic for m ≤ 185 (F_186 < 2¹²⁸), then
/// correctly rounded to `f64`; `f64` doubling beyond that (O(log m) roundings).
fn fib_pair(m: u32) -> (f64, f64) {
    if m <= 185 {
        let mut a: u128 = 0;
        let mut b: u128 = 1;
        let mut bit = 31;
        while bit >= 0 {
            let d = a * (2 * b - a);
            let e = a * a + b * b;
            a = d;
            b = e;
            if (m >> bit) & 1 == 1 {
                let t = a + b;
                a = b;
                b = t;
            }
            bit -= 1;
        }
        return (a as f64, b as f64);
    }
    let mut a = 0.0;
    let mut b = 1.0;
    let mut bit = 31;
    while bit >= 0 {
        let d = a * (2.0 * b - a);
        let e = a * a + b * b;
        a = d;
        b = e;
        if (m >> bit) & 1 == 1 {
            let t = a + b;
            a = b;
            b = t;
        }
        bit -= 1;
    }
    (a, b)
}
/// Fibonacci number F_n for integer n (NaN for non-integers).
///
/// Exact for |n| ≤ 185 (128-bit doubling), ≈ 1e-15 beyond;
/// F_{−n} = (−1)^{n+1} F_n.
pub fn fibonacci(n: f64) -> f64 {
    if n == f64::INFINITY {
        return f64::INFINITY;
    }
    if !is_int(n) {
        return f64::NAN;
    }
    let m = p_abs(n);
    if m > 4_000_000_000.0 {
        return if n < 0.0 && is_even(m) {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }
    let v = fib_pair(m as u32).0;
    if n < 0.0 && is_even(m) { -v } else { v }
}
/// Lucas number L_n for integer n (NaN for non-integers); L_{−n} = (−1)^n L_n.
pub fn lucas(n: f64) -> f64 {
    if n == f64::INFINITY {
        return f64::INFINITY;
    }
    if !is_int(n) {
        return f64::NAN;
    }
    let m = p_abs(n);
    if m > 4_000_000_000.0 {
        return if n < 0.0 && !is_even(m) {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }
    let (f, f1) = fib_pair(m as u32);
    let v = 2.0 * f1 - f;
    if n < 0.0 && !is_even(m) { -v } else { v }
}
// @@end fibonacci

// @@begin harmonic
/// Harmonic number H_n = Σ_{k=1}^{n} 1/k, extended to real n as ψ(n+1) + γ.
///
/// Direct (reverse-order) summation for integer 0 ≤ n ≤ 100, digamma beyond.
/// Negative integers are poles and return NaN.
pub fn harmonic(n: f64) -> f64 {
    if n.is_nan() {
        return f64::NAN;
    }
    if n == f64::INFINITY {
        return f64::INFINITY;
    }
    if is_int(n) {
        if n < 0.0 {
            return f64::NAN;
        }
        if n <= 100.0 {
            let mut sum = 0.0;
            let mut k = n;
            while k >= 1.0 {
                sum += 1.0 / k;
                k -= 1.0;
            }
            return sum;
        }
    }
    digamma(n + 1.0) + EULER_GAMMA
}
// @@end harmonic

// @@begin factorial2
/// Double factorial n!! for integer n ≥ −1 (odd negative n via
/// n!! = (n+2)!!/(n+2); even negative n and non-integers return NaN).
pub fn factorial2(n: f64) -> f64 {
    if !is_int(n) {
        return f64::NAN;
    }
    if n >= -1.0 {
        let mut acc = 1.0;
        let mut k = n;
        while k > 1.0 {
            acc *= k;
            k -= 2.0;
        }
        return acc;
    }
    if is_even(n) {
        return f64::NAN;
    }
    // n odd, n ≤ −3: n!! = 1 / ((n+2)(n+4)···(−1))
    let mut acc = 1.0;
    let mut k = n + 2.0;
    while k <= -1.0 {
        acc *= k;
        k += 2.0;
    }
    1.0 / acc
}
// @@end factorial2

// @@begin pochhammer
/// Rising factorial (Pochhammer symbol) (x)_n = Γ(x+n)/Γ(x).
///
/// Integer |n| ≤ 1000 uses a direct product; otherwise Γ/ln Γ with sign
/// handling.  Poles of Γ(x+n) not cancelled by a pole of Γ(x) return NaN.
pub fn rising_factorial(x: f64, n: f64) -> f64 {
    if x.is_nan() || n.is_nan() {
        return f64::NAN;
    }
    if is_int(n) && p_abs(n) <= 1000.0 {
        let mut acc = 1.0;
        if n >= 0.0 {
            let mut i = 0.0;
            while i < n {
                acc *= x + i;
                i += 1.0;
            }
            return acc;
        }
        let mut i = 1.0;
        while i <= -n {
            acc *= x - i;
            i += 1.0;
        }
        return 1.0 / acc;
    }
    let top = x + n;
    if is_gamma_pole(x) {
        return if is_gamma_pole(top) { f64::NAN } else { 0.0 };
    }
    if is_gamma_pole(top) {
        return f64::NAN;
    }
    let sign = gamma_sign(top) * gamma_sign(x);
    sign * p_exp(lgamma(top) - lgamma(x))
}
/// Falling factorial x^(n) = x(x−1)···(x−n+1) = Γ(x+1)/Γ(x−n+1).
pub fn falling_factorial(x: f64, n: f64) -> f64 {
    rising_factorial(x - n + 1.0, n)
}
// @@end pochhammer

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        if b == 0.0 {
            a.abs()
        } else {
            ((a - b) / b).abs()
        }
    }

    fn assert_rel(got: f64, want: f64, tol: f64, what: &str) {
        assert!(
            rel(got, want) <= tol,
            "{what}: got {got:.17e}, want {want:.17e}, rel err {:.3e} > {tol:.0e}",
            rel(got, want)
        );
    }

    #[test]
    fn gamma_known_values() {
        assert_rel(gamma(0.5), 1.7724538509055160273, 1e-15, "Γ(1/2)");
        assert_rel(gamma(1.5), 0.88622692545275801365, 1e-15, "Γ(3/2)");
        assert_rel(gamma(-0.5), -3.5449077018110320546, 1e-15, "Γ(-1/2)");
        assert_rel(gamma(-1.5), 2.3632718012073547031, 1e-15, "Γ(-3/2)");
        assert_rel(gamma(-10.5), -2.6401218205477163162e-7, 1e-15, "Γ(-21/2)");
        assert_rel(gamma(0.75), 1.2254167024651776451, 2e-15, "Γ(3/4)");
        assert_rel(gamma(-0.75), -4.8341465442958777492, 2e-15, "Γ(-3/4)");
        assert_rel(gamma(1.0 / 3.0), 2.6789385347077476337, 1e-15, "Γ(1/3)");
        assert_rel(gamma(10.5), 1133278.3889487855673, 2e-15, "Γ(10.5)");
        assert_rel(gamma(170.5), 5.5620924145599996107e305, 2e-15, "Γ(170.5)");
        // Reference computed at the exact double nearest 170.3.
        assert_rel(gamma(170.3), 1.9915875572358899762e305, 3e-15, "Γ(170.3)");
        assert_rel(gamma(55.25), 6.275774141473366285e71, 3e-15, "Γ(55.25)");
        assert_eq!(gamma(5.0), 24.0);
        assert_eq!(gamma(1.0), 1.0);
        assert_eq!(gamma(21.0), 2432902008176640000.0);
        assert!(gamma(-2.0).is_nan());
        assert!(gamma(0.0).is_infinite() && gamma(0.0) > 0.0);
        assert!(gamma(-0.0).is_infinite() && gamma(-0.0) < 0.0);
        assert_eq!(gamma(200.0), f64::INFINITY);
        assert_rel(
            gamma(-100.5),
            -3.3536908198076786422e-159,
            5e-14,
            "Γ(-100.5)",
        );
    }

    #[test]
    fn lgamma_known_values() {
        assert_rel(lgamma(100.0), 359.13420536957539878, 1e-15, "lnΓ(100)");
        assert_rel(lgamma(0.5), 0.57236494292470008707, 1e-14, "lnΓ(1/2)");
        assert_rel(lgamma(1000.5), 5908.6741758486774887, 1e-15, "lnΓ(1000.5)");
        assert_rel(lgamma(-0.5), 1.2655121234846453965, 1e-14, "ln|Γ(-1/2)|");
        assert_eq!(lgamma(1.0), 0.0);
        assert_eq!(lgamma(2.0), 0.0);
        assert!(lgamma(-3.0).is_infinite());
        assert_rel(lgamma(1e-300), 690.77552789821370521, 1e-14, "lnΓ(1e-300)");
    }

    #[test]
    fn digamma_known_values() {
        assert_rel(digamma(1.0), -0.57721566490153286061, 1e-14, "ψ(1)");
        assert_rel(digamma(0.5), -1.9635100260214234794, 1e-14, "ψ(1/2)");
        assert_rel(digamma(2.0), 0.42278433509846713939, 1e-14, "ψ(2)");
        assert_rel(digamma(10.0), 2.2517525890667211076, 1e-15, "ψ(10)");
        assert_rel(digamma(-0.5), 0.036489973978576520559, 1e-12, "ψ(-1/2)");
        assert_rel(digamma(1000.0), 6.9072551956488120521, 1e-15, "ψ(1000)");
        assert_rel(digamma(-2.5), 1.1031566406452431872, 1e-14, "ψ(-5/2)");
        assert!(digamma(-1.0).is_nan());
    }

    #[test]
    fn erf_known_values() {
        assert_rel(erf(0.1), 0.11246291601828489220, 1e-15, "erf(0.1)");
        assert_rel(erf(0.5), 0.52049987781304653768, 1e-15, "erf(0.5)");
        assert_rel(erf(1.0), 0.84270079294971486934, 1e-15, "erf(1)");
        assert_rel(erf(2.0), 0.99532226501895273416, 1e-15, "erf(2)");
        assert_rel(erf(-1.0), -0.84270079294971486934, 1e-15, "erf(-1)");
        assert_eq!(erf(0.0), 0.0);
        assert_eq!(erf(30.0), 1.0);
        assert_rel(erfc(0.5), 0.47950012218695346232, 1e-15, "erfc(0.5)");
        assert_rel(erfc(3.0), 2.2090496998585441373e-5, 1e-15, "erfc(3)");
        assert_rel(erfc(5.0), 1.5374597944280348502e-12, 1e-15, "erfc(5)");
        assert_rel(erfc(10.0), 2.0884875837625447570e-45, 1e-15, "erfc(10)");
        assert_rel(erfc(20.0), 5.3958656116079009289e-176, 1e-15, "erfc(20)");
        assert_rel(erfc(-1.0), 1.8427007929497148693, 1e-15, "erfc(-1)");
        assert_eq!(erfc(30.0), 0.0);
    }

    #[test]
    fn erfinv_known_values() {
        // mpmath 1.3 (dps 60), evaluated at the exact double nearest each
        // literal: erfinv(mpf(x)).
        let cases: [(f64, f64); 15] = [
            (0.1, 0.088855990494257691974),
            (0.3, 0.27246271472675434502),
            (0.46875, 0.44271885732435436322),
            (0.47, 0.4440673114347423053),
            (0.5, 0.47693627620446987338),
            (0.6, 0.59511608144999482198),
            (0.75, 0.81341984759761854169),
            (0.9, 1.1630871536766741628),
            (0.99, 1.8213863677184494559),
            (0.999, 2.3267537655135244939),
            (0.99999, 3.123413274341570864),
            (1.0 - 1e-9, 4.3200053881053620459),
            (1.0 - 1e-12, 5.0420318985726961301),
            (1.0 - 1e-15, 5.6759157397447131788),
            (1.0 - f64::EPSILON, 5.8050186831934533002),
        ];
        for &(x, want) in &cases {
            assert_rel(erfinv(x), want, 1e-15, &format!("erfinv({x})"));
            assert_rel(erfinv(-x), -want, 1e-15, &format!("erfinv(-{x})"));
        }
        // Tiny arguments: erfinv(x) ≈ (√π/2)x.
        assert_rel(
            erfinv(1e-300),
            8.8622692545275803586e-301,
            1e-15,
            "erfinv(1e-300)",
        );
        assert_rel(
            erfinv(1e-10),
            8.8622692545275804594e-11,
            1e-15,
            "erfinv(1e-10)",
        );
        assert_eq!(erfinv(0.0), 0.0);
        assert_eq!(erfinv(1.0), f64::INFINITY);
        assert_eq!(erfinv(-1.0), f64::NEG_INFINITY);
        assert!(erfinv(1.0000001).is_nan());
        assert!(erfinv(-2.0).is_nan());
        assert!(erfinv(f64::NAN).is_nan());
        // Round trip through the runtime's own erf.
        for &x in &[0.001, 0.25, 0.5, 0.7, 0.95, 0.9999, 1.0 - 1e-13] {
            assert_rel(erf(erfinv(x)), x, 4e-16, &format!("erf(erfinv({x}))"));
        }
    }

    #[test]
    fn erfcinv_known_values() {
        // mpmath 1.3 (dps 60): erfcinv(y) solved from erfc(z) = y at the
        // exact double nearest each literal.
        let cases: [(f64, f64); 11] = [
            (0.5, 0.47693627620446987338),
            (1.5, -0.47693627620446987338),
            (0.1, 1.1630871536766740677),
            (1.9, -1.1630871536766737823),
            (0.999, 0.00088622715746655289169),
            (1e-5, 3.1234132743408750177),
            (1e-10, 4.5728249673894852748),
            (1e-20, 6.6015806223551425656),
            (1e-100, 15.065574702592645704),
            (1e-300, 26.209469960516123886),
            (1.9999999999, -4.5728249585449249378),
        ];
        for &(y, want) in &cases {
            assert_rel(erfcinv(y), want, 1e-15, &format!("erfcinv({y})"));
        }
        // Subnormal arguments keep relative accuracy (erfc underflows here).
        assert_rel(
            erfcinv(2e-308),
            26.545265807344898846,
            1e-15,
            "erfcinv(2e-308)",
        );
        assert_rel(
            erfcinv(5e-324),
            27.213293210812948815,
            1e-15,
            "erfcinv(5e-324)",
        );
        assert_eq!(erfcinv(1.0), 0.0);
        assert_eq!(erfcinv(0.0), f64::INFINITY);
        assert_eq!(erfcinv(2.0), f64::NEG_INFINITY);
        assert!(erfcinv(-0.1).is_nan());
        assert!(erfcinv(2.5).is_nan());
        // erfcinv(y) = erfinv(1 − y) where 1 − y is exact.
        assert_eq!(erfcinv(0.75), erfinv(0.25));
        assert_eq!(erfcinv(0.5), erfinv(0.5));
    }

    #[test]
    fn lambert_w_known_values() {
        assert_rel(lambert_w0(1.0), 0.56714329040978387300, 1e-15, "W(1)");
        assert_rel(lambert_w0(E), 1.0, 1e-15, "W(e)");
        assert_rel(lambert_w0(10.0), 1.7455280027406993831, 1e-15, "W(10)");
        assert_rel(lambert_w0(1e10), 20.028685413304950781, 1e-15, "W(1e10)");
        assert_eq!(lambert_w0(-1.0 / E), -1.0);
        assert!(lambert_w0(-0.5).is_nan());
        for &x in &[-0.36, -0.3, -0.1, -1e-3, 1e-3, 0.3, 2.5, 100.0, 1e100] {
            let w = lambert_w0(x);
            assert_rel(w * w.exp(), x, 1e-14, &format!("W({x}) e^W"));
        }
        // Branch-point series region.
        let x = -0.3678;
        let w = lambert_w0(x);
        assert_rel(w * w.exp(), x, 1e-12, "W(-0.3678) e^W");
    }

    #[test]
    fn beta_binomial_factorial() {
        assert_rel(beta(2.0, 3.0), 1.0 / 12.0, 1e-15, "B(2,3)");
        assert_rel(beta(0.5, 0.5), PI, 1e-15, "B(1/2,1/2)");
        assert_rel(beta(2.5, 1.5), PI / 16.0, 4e-15, "B(2.5,1.5)");
        assert_rel(beta(-0.5, 1.5), -PI, 1e-14, "B(-1/2,3/2)");
        assert!(beta(0.0, 1.0).is_nan());
        assert_eq!(beta(0.5, -0.5), 0.0);
        assert_eq!(factorial(5.0), 120.0);
        assert_eq!(factorial(0.0), 1.0);
        assert_rel(factorial(0.5), 0.88622692545275801365, 1e-15, "0.5!");
        assert_eq!(binomial(10.0, 3.0), 120.0);
        assert_eq!(binomial(50.0, 25.0), 126410606437752.0);
        assert_eq!(binomial(5.0, 7.0), 0.0);
        assert_eq!(binomial(5.0, -1.0), 0.0);
        assert_eq!(binomial(0.5, 2.0), -0.125);
        assert_eq!(binomial(-3.0, 2.0), 6.0);
        assert_rel(
            binomial(0.5, 0.25),
            1.0787052023767587133,
            1e-14,
            "C(1/2,1/4)",
        );
        assert_rel(
            binomial(200.0, 100.0),
            9.0548514656103281165e58,
            1e-13,
            "C(200,100)",
        );
    }

    #[test]
    fn bessel_j_known_values() {
        assert_rel(bessel_j(0, 1.0), 0.76519768655796655145, 1e-15, "J0(1)");
        assert_rel(bessel_j(1, 1.0), 0.44005058574493351596, 1e-15, "J1(1)");
        assert_rel(bessel_j(0, 5.0), -0.17759677131433830435, 1e-14, "J0(5)");
        assert_rel(bessel_j(2, 10.0), 0.25463031368512062253, 1e-14, "J2(10)");
        assert_rel(bessel_j(5, 3.0), 0.043028434877047583925, 1e-14, "J5(3)");
        assert_rel(bessel_j(0, 30.0), -0.086367983581040211336, 1e-13, "J0(30)");
        assert_rel(
            bessel_j(0, 100.0),
            0.019985850304223122424,
            1e-13,
            "J0(100)",
        );
        assert_rel(bessel_j(10, 1.0), 2.630615123687453207e-10, 1e-14, "J10(1)");
        assert_rel(bessel_j(1, 24.0), -0.15403806518312122128, 1e-13, "J1(24)");
        assert_rel(bessel_j(3, 40.0), -0.12614481550582080316, 1e-13, "J3(40)");
        assert_rel(
            bessel_j(20, 30.0),
            0.0048310199934040645386,
            1e-12,
            "J20(30)",
        );
        assert_eq!(bessel_j(0, 0.0), 1.0);
        assert_eq!(bessel_j(3, 0.0), 0.0);
        assert_rel(bessel_j(1, -1.0), -0.44005058574493351596, 1e-15, "J1(-1)");
        assert_rel(bessel_j(-1, 1.0), -0.44005058574493351596, 1e-15, "J-1(1)");
    }

    #[test]
    fn bessel_y_known_values() {
        assert_rel(bessel_y(0, 1.0), 0.088256964215676957983, 1e-14, "Y0(1)");
        assert_rel(bessel_y(1, 1.0), -0.78121282130028871655, 1e-14, "Y1(1)");
        assert_rel(bessel_y(0, 10.0), 0.055671167283599391424, 1e-13, "Y0(10)");
        assert_rel(bessel_y(1, 10.0), 0.24901542420695388392, 1e-13, "Y1(10)");
        assert_rel(bessel_y(0, 0.1), -1.5342386513503668083, 1e-14, "Y0(0.1)");
        assert_rel(bessel_y(1, 0.1), -6.4589510947020266377, 1e-14, "Y1(0.1)");
        assert_rel(bessel_y(2, 5.0), 0.36766288260552451799, 1e-13, "Y2(5)");
        assert_rel(bessel_y(0, 30.0), -0.11729573168666402525, 1e-13, "Y0(30)");
        assert_rel(bessel_y(3, 2.5), -0.75605549675367099684, 1e-13, "Y3(2.5)");
        assert!(bessel_y(0, -1.0).is_nan());
        assert_eq!(bessel_y(0, 0.0), f64::NEG_INFINITY);
    }

    #[test]
    fn bessel_i_k_known_values() {
        assert_rel(bessel_i(0, 1.0), 1.2660658777520083356, 1e-15, "I0(1)");
        assert_rel(bessel_i(1, 1.0), 0.56515910399248502721, 1e-15, "I1(1)");
        assert_rel(bessel_i(0, 10.0), 2815.7166284662544715, 1e-14, "I0(10)");
        assert_rel(bessel_i(2, 3.0), 2.2452124409299511546, 1e-14, "I2(3)");
        assert_rel(bessel_i(0, 50.0), 2.9325537838493363267e20, 1e-14, "I0(50)");
        assert_rel(bessel_i(1, -1.0), -0.56515910399248502721, 1e-15, "I1(-1)");
        assert_rel(
            bessel_i(0, 700.0),
            1.5295933476718737363e302,
            1e-13,
            "I0(700)",
        );
        assert_rel(bessel_k(0, 1.0), 0.42102443824070833334, 1e-14, "K0(1)");
        assert_rel(bessel_k(1, 1.0), 0.60190723019723457474, 1e-14, "K1(1)");
        assert_rel(bessel_k(0, 10.0), 1.7780062316167651811e-5, 1e-14, "K0(10)");
        assert_rel(bessel_k(2, 3.0), 0.061510458471742037657, 1e-14, "K2(3)");
        assert_rel(bessel_k(0, 0.1), 2.4270690247020165578, 1e-14, "K0(0.1)");
        assert_rel(
            bessel_k(1, 25.0),
            3.5327780731999337702e-12,
            1e-14,
            "K1(25)",
        );
        assert_rel(bessel_k(5, 2.0), 9.4310491005964674428, 1e-13, "K5(2)");
        assert_rel(
            bessel_k(10, 30.0),
            1.0842816942222973911e-13,
            1e-13,
            "K10(30)",
        );
        assert!(bessel_k(0, -1.0).is_nan());
    }

    #[test]
    fn orthogonal_polynomials() {
        assert_eq!(legendre_p(3, 0.5), -0.4375);
        assert_eq!(legendre_p(0, 0.3), 1.0);
        assert_eq!(legendre_p(-3, 0.5), legendre_p(2, 0.5));
        assert_rel(chebyshev_t(4, 0.3), 0.3448, 1e-15, "T4(0.3)");
        assert_eq!(chebyshev_t(-4, 0.3), chebyshev_t(4, 0.3));
        assert_eq!(chebyshev_u(2, 0.5), 0.0);
        assert_eq!(chebyshev_u(3, 0.5), -1.0);
        assert_eq!(chebyshev_u(-1, 0.5), 0.0);
        assert_eq!(chebyshev_u(-3, 0.5), -chebyshev_u(1, 0.5));
        assert_eq!(hermite_h(3, 2.0), 40.0);
        assert_eq!(hermite_h(0, 2.0), 1.0);
        assert!(hermite_h(-1, 2.0).is_nan());
        assert_eq!(laguerre_l(2, 1.0), -0.5);
        assert_eq!(laguerre_l(1, 3.0), -2.0);
    }

    #[test]
    fn integer_sequences() {
        assert_eq!(fibonacci(0.0), 0.0);
        assert_eq!(fibonacci(1.0), 1.0);
        assert_eq!(fibonacci(10.0), 55.0);
        assert_eq!(fibonacci(50.0), 12586269025.0);
        assert_eq!(fibonacci(78.0), 8944394323791464.0);
        assert_rel(fibonacci(100.0), 3.5422484817926191508e20, 1e-15, "F100");
        assert_rel(fibonacci(185.0), 2.05697230343233228174e38, 1e-15, "F185");
        assert_rel(
            fibonacci(300.0),
            2.2223224462942044552973989e62,
            5e-15,
            "F300",
        );
        assert_eq!(fibonacci(-5.0), 5.0);
        assert_eq!(fibonacci(-6.0), -8.0);
        assert!(fibonacci(2.5).is_nan());
        assert_eq!(lucas(0.0), 2.0);
        assert_eq!(lucas(10.0), 123.0);
        assert_eq!(lucas(-3.0), -4.0);
        assert_rel(lucas(100.0), 7.9207083984837225313e20, 1e-15, "L100");
        assert_eq!(harmonic(0.0), 0.0);
        assert_eq!(harmonic(1.0), 1.0);
        assert_rel(harmonic(5.0), 137.0 / 60.0, 1e-15, "H5");
        assert_rel(harmonic(100.0), 5.1873775176396202608, 1e-15, "H100");
        assert_rel(harmonic(1000.0), 7.4854708605503449127, 1e-15, "H1000");
        assert_rel(harmonic(0.5), 0.61370563888010938117, 1e-14, "H(1/2)");
        assert_eq!(factorial2(7.0), 105.0);
        assert_eq!(factorial2(8.0), 384.0);
        assert_eq!(factorial2(0.0), 1.0);
        assert_eq!(factorial2(-1.0), 1.0);
        assert_eq!(factorial2(-3.0), -1.0);
        assert_rel(factorial2(-5.0), 1.0 / 3.0, 1e-15, "(-5)!!");
        assert!(factorial2(-2.0).is_nan());
        assert_eq!(rising_factorial(3.0, 4.0), 360.0);
        assert_eq!(rising_factorial(0.5, 3.0), 1.875);
        assert_eq!(rising_factorial(5.0, -2.0), 1.0 / 12.0);
        assert_rel(
            rising_factorial(2.5, 1.5),
            4.5135166683820502956,
            1e-14,
            "(2.5)_{1.5}",
        );
        assert_eq!(falling_factorial(5.0, 2.0), 20.0);
        assert_eq!(falling_factorial(0.5, 2.0), -0.25);
    }
}

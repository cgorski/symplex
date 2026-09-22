//! Machinery shared by the statistics-on-data modules: exact conversions,
//! argument checks with one wording, the reference tail functions
//! (`χ²`, `F`, Student-t, normal) each module used to carry its own copy
//! of, and the Wald summary of a maximum-likelihood fit.
//!
//! Everything here is `pub(crate)`.  The public vocabulary it standardises:
//!
//! - a probability *level* (`confidence`, `alpha`, `beta`, `power`) is an
//!   `f64` in `(0, 1)` because it only ever feeds a numeric quantile;
//! - a distribution or hypothesis *parameter* (`p0`, `mu0`, `sigma`, prior
//!   shapes) is a `&Q` because it enters the exact statistic;
//! - counts are `usize`; data are `&[Q]`; a `ctx: &Context` is passed
//!   exactly when an `Ex` is built from non-`Ex` inputs.

use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive};

use super::data::Q;
use super::family::Distribution;
use crate::api::context::Context;
use crate::api::expr::Ex;
use crate::base::dense_f64;
use crate::base::errors::SymplexError;
use crate::domains::optimize::{RootOpts, brent_root};
use crate::output::codegen::numeric_rt::{erfc, erfcinv};

// ── Exact conversions ───────────────────────────────────────────────────

/// `n` as an exact rational.
pub(crate) fn qi(n: i64) -> Q {
    Q::from_integer(BigInt::from(n))
}

/// `n` as an exact rational (no overflow: `BigInt`).
pub(crate) fn qu(n: usize) -> Q {
    Q::from_integer(BigInt::from(n))
}

/// A rational as an expression in `ctx`.
pub(crate) fn ex(ctx: &Context, q: &Q) -> Ex {
    ctx.from_ratio(q.clone())
}

/// A count as an expression in `ctx` (no overflow).
pub(crate) fn ex_usize(ctx: &Context, n: usize) -> Ex {
    ctx.from_bigint(BigInt::from(n))
}

/// The nearest `f64` to a rational (`Ratio::to_f64`, which scales rather
/// than dividing two possibly-overflowing conversions).
pub(crate) fn q_to_f64(q: &Q) -> f64 {
    q.to_f64().unwrap_or(f64::NAN)
}

// ── Argument checks (one wording each) ──────────────────────────────────

pub(crate) fn invalid(op: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::invalid_argument(op, reason)
}

/// `0 < v < 1`, for a level named `name` (`"confidence"`, `"alpha"`, …).
pub(crate) fn check_unit_open(op: &'static str, name: &str, v: f64) -> Result<(), SymplexError> {
    if v > 0.0 && v < 1.0 {
        Ok(())
    } else {
        Err(invalid(
            op,
            format!("{name} must lie strictly between 0 and 1, got {v}"),
        ))
    }
}

/// `0 < confidence < 1`.
pub(crate) fn check_confidence(op: &'static str, confidence: f64) -> Result<(), SymplexError> {
    check_unit_open(op, "confidence", confidence)
}

/// `0 < alpha < 1`.
pub(crate) fn check_alpha(op: &'static str, alpha: f64) -> Result<(), SymplexError> {
    check_unit_open(op, "alpha", alpha)
}

/// `v` finite.
pub(crate) fn check_finite(op: &'static str, name: &str, v: f64) -> Result<(), SymplexError> {
    if v.is_finite() {
        Ok(())
    } else {
        Err(invalid(op, format!("{name} must be finite, got {v}")))
    }
}

/// At least `min` observations in `x` (named `name`).
pub(crate) fn check_sample(
    op: &'static str,
    name: &str,
    x: &[Q],
    min: usize,
) -> Result<(), SymplexError> {
    if x.len() < min {
        return Err(invalid(
            op,
            format!(
                "{name} needs at least {min} observation{}, got {}",
                if min == 1 { "" } else { "s" },
                x.len()
            ),
        ));
    }
    Ok(())
}

/// `n` as an `i64`, or an error naming the operation.
pub(crate) fn usize_to_i64(op: &'static str, n: usize) -> Result<i64, SymplexError> {
    i64::try_from(n).map_err(|_| invalid(op, format!("{n} does not fit in an i64")))
}

// ── Reference distributions: exact tails ────────────────────────────────

/// `P(χ²_df ≥ x) = Γ(df/2, x/2) / Γ(df/2)` as an expression.
pub(crate) fn chi_squared_sf(ctx: &Context, df: usize, x: &Ex) -> Ex {
    let half_df = ex_usize(ctx, df) / ctx.int(2);
    (x / ctx.int(2)).uppergamma(&half_df) / half_df.gamma()
}

/// [`chi_squared_sf`] at a rational statistic (`1` for `x ≤ 0`).
pub(crate) fn chi_squared_sf_q(ctx: &Context, df: usize, x: &Q) -> Ex {
    if !x.is_positive() {
        return ctx.one();
    }
    chi_squared_sf(ctx, df, &ex(ctx, x))
}

/// `P(F_{d₁,d₂} ≥ f) = I_{d₂/(d₂ + d₁f)}(d₂/2, d₁/2)` as an expression
/// (`1` for `f ≤ 0`).
pub(crate) fn f_sf(ctx: &Context, d1: usize, d2: usize, f: &Q) -> Ex {
    if !f.is_positive() {
        return ctx.one();
    }
    let z = qu(d2) / (qu(d2) + qu(d1) * f);
    ex(ctx, &z).betainc_regularized(
        &(ex_usize(ctx, d2) / ctx.int(2)),
        &(ex_usize(ctx, d1) / ctx.int(2)),
        &ctx.zero(),
    )
}

/// [`f_sf`] for rational (not necessarily integer) degrees of freedom —
/// the Greenhouse–Geisser and Huynh–Feldt corrections scale both by `ε`.
pub(crate) fn f_sf_rational(ctx: &Context, d1: &Q, d2: &Q, f: &Q) -> Ex {
    if !f.is_positive() {
        return ctx.one();
    }
    let z = d2 / (d2 + d1 * f);
    ex(ctx, &z).betainc_regularized(
        &ex(ctx, &(d2 / qi(2))),
        &ex(ctx, &(d1 / qi(2))),
        &ctx.zero(),
    )
}

// ── Reference distributions: numeric quantiles ──────────────────────────

/// `Φ(x) = ½ erfc(−x/√2)`.
pub(crate) fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// `1 − Φ(x) = ½ erfc(x/√2)`.
pub(crate) fn norm_sf(x: f64) -> f64 {
    0.5 * erfc(x / std::f64::consts::SQRT_2)
}

/// `φ(x) = e^{−x²/2} / √(2π)`.
pub(crate) fn norm_pdf(x: f64) -> f64 {
    const SQRT_2PI: f64 = 2.506_628_274_631_000_5;
    (-0.5 * x * x).exp() / SQRT_2PI
}

/// `P(|Z| ≥ |z|) = erfc(|z|/√2)`.
pub(crate) fn normal_two_sided(z: f64) -> f64 {
    erfc(z.abs() / std::f64::consts::SQRT_2)
}

/// `Φ⁻¹(1 − alpha)`: the upper-tail standard normal quantile.
pub(crate) fn norm_isf(alpha: f64) -> f64 {
    std::f64::consts::SQRT_2 * erfcinv(2.0 * alpha)
}

/// `Φ⁻¹(p)`.
pub(crate) fn norm_ppf(p: f64) -> f64 {
    -norm_isf(p)
}

/// The two-sided critical value `z_{α/2}` for a confidence level.
pub(crate) fn z_two_sided(confidence: f64) -> f64 {
    norm_isf((1.0 - confidence) / 2.0)
}

/// `P(T ≤ t)` for Student's t with `ν` degrees of freedom, as an `f64`:
/// the exact expression `½ I_{ν/(t²+ν)}(ν/2, ½)` evaluated by `eval_f64`
/// (arbitrary precision, then rounded).
pub(crate) fn student_t_cdf_f64(ctx: &Context, df: f64, t: f64) -> Result<f64, SymplexError> {
    let nu = ctx.from_f64(df)?;
    let z = &nu / (ctx.from_f64(t * t)? + &nu);
    let tail = ctx.rational(1, 2)
        * z.betainc_regularized(&(&nu / ctx.int(2)), &ctx.rational(1, 2), &ctx.zero());
    let tail = tail.eval_f64()?;
    Ok(if t < 0.0 { tail } else { 1.0 - tail })
}

/// The Student-t quantile `t_{p, ν}` by Brent's method on
/// [`student_t_cdf_f64`] over a bracket grown from `0` (the distribution
/// is symmetric, so `p < ½` is mirrored).
pub(crate) fn student_t_quantile_f64(
    op: &'static str,
    ctx: &Context,
    df: f64,
    p: f64,
) -> Result<f64, SymplexError> {
    if p < 0.5 {
        return student_t_quantile_f64(op, ctx, df, 1.0 - p).map(|t| -t);
    }
    if p == 0.5 {
        return Ok(0.0);
    }
    let g = |t: f64| student_t_cdf_f64(ctx, df, t).unwrap_or(f64::NAN) - p;
    let mut hi = 1.0;
    for _ in 0..64 {
        let v = g(hi);
        if v.is_nan() {
            return Err(SymplexError::computation_failed(
                op,
                "the Student-t distribution function could not be evaluated",
            ));
        }
        if v >= 0.0 {
            break;
        }
        hi *= 2.0;
    }
    let root = brent_root(g, 0.0, hi, &RootOpts::default())
        .map_err(|e| SymplexError::computation_failed(op, e.to_string()))?;
    if root.is_finite() {
        Ok(root)
    } else {
        Err(SymplexError::computation_failed(
            op,
            "the Student-t quantile did not converge",
        ))
    }
}

/// The two-sided Student-t critical value `t_{α/2, ν}` for a confidence
/// level, without a caller-supplied context (a private one is made: the
/// result is an `f64`).
pub(crate) fn t_two_sided(op: &'static str, df: f64, confidence: f64) -> Result<f64, SymplexError> {
    let ctx = Context::new();
    student_t_quantile_f64(op, &ctx, df, 1.0 - (1.0 - confidence) / 2.0)
}

/// A standard normal in a private context, for numeric quantiles.
pub(crate) fn standard_normal() -> Distribution {
    let ctx = Context::new();
    Distribution::normal(ctx.int(0), ctx.int(1))
}

// ── Maximum-likelihood fits: information matrices ────────────────────────────

/// A Cholesky pivot below this fraction of its diagonal entry marks an
/// information matrix as numerically singular (collinear regressors, a
/// likelihood without a finite maximiser).
const INFORMATION_PIVOT_REL_TOL: f64 = 1e-12;

/// Lower Cholesky factor (flat row-major, see [`dense_f64`]) of a
/// `p×p` information matrix given as rows; `None` when it is not
/// positive definite to [`INFORMATION_PIVOT_REL_TOL`].
pub(crate) fn information_cholesky(info: &[Vec<f64>]) -> Option<Vec<f64>> {
    dense_f64::cholesky(
        &dense_f64::flatten(info),
        info.len(),
        INFORMATION_PIVOT_REL_TOL,
    )
}

/// `I⁻¹` from the observed information at the estimate, with the standard
/// errors, `z` and two-sided normal p-values of `params`.
pub(crate) struct WaldSummary {
    pub cov: Vec<Vec<f64>>,
    pub se: Vec<f64>,
    pub z: Vec<f64>,
    pub p: Vec<f64>,
}

/// The Wald summary of a fit, or `None` when `info` is singular (the
/// standard errors are undefined).
pub(crate) fn wald_summary(info: &[Vec<f64>], params: &[f64]) -> Option<WaldSummary> {
    let n = info.len();
    let l = information_cholesky(info)?;
    let cov = dense_f64::to_rows(&dense_f64::spd_inverse(&l, n), n, n);
    let se: Vec<f64> = (0..params.len()).map(|j| cov[j][j].sqrt()).collect();
    let z: Vec<f64> = params.iter().zip(&se).map(|(b, s)| b / s).collect();
    let p: Vec<f64> = z.iter().map(|z| normal_two_sided(*z)).collect();
    Some(WaldSummary { cov, se, z, p })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_quantiles() {
        // scipy.stats.norm.ppf(0.975) = 1.959963984540054
        assert!((z_two_sided(0.95) - 1.959963984540054).abs() < 1e-12);
        assert!((norm_ppf(0.975) - 1.959963984540054).abs() < 1e-12);
        assert!((norm_isf(0.025) - 1.959963984540054).abs() < 1e-12);
    }

    #[test]
    fn student_t_quantile() {
        // scipy.stats.t.ppf(0.975, 5) = 2.5705818356363146
        assert!((t_two_sided("test", 5.0, 0.95).unwrap() - 2.5705818356363146).abs() < 1e-9);
    }

    #[test]
    fn checks() {
        assert!(check_confidence("op", 0.95).is_ok());
        assert!(check_confidence("op", 1.0).is_err());
        assert!(check_alpha("op", 0.0).is_err());
        assert!(check_sample("op", "x", &[qi(1)], 2).is_err());
        assert!(check_sample("op", "x", &[qi(1), qi(2)], 2).is_ok());
        assert_eq!(q_to_f64(&qi(3)), 3.0);
    }
}

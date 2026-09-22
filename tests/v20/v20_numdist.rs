//! 0.21 — `stats::numdist`, the `f64` reference-distribution kernel behind
//! `Distribution::quantile_f64` and the critical values of the data modules.
//!
//! Reference values cite scipy 1.18 (`symplex/.venv/bin/python`) by the call
//! that produced them; where scipy itself is only good to ~1e-13 the value
//! is mpmath 1.3 at 50 digits (`mpmath.betainc` / `mpmath.gammainc` with
//! `regularized=True`, quantiles by `mpmath.findroot` from the scipy value)
//! and the scipy number is quoted alongside.  Nothing below was computed by
//! hand.

use std::time::Instant;

use symplex::prelude::*;
use symplex::stats::Distribution;
use symplex::stats::data::from_i64;
use symplex::stats::estimation::confidence_interval_mean;
use symplex::stats::numdist::{
    beta, betainc_regularized_f64, binom, chi2, f, gamma, gammainc_lower_regularized_f64,
    gammainc_upper_regularized_f64, norm, poisson, t,
};

/// Relative closeness, absolute when the reference is (essentially) zero.
fn close(actual: f64, expected: f64, rel: f64, label: &str) {
    // A quantile at the median is exactly 0; anything below 1e-15 is "zero".
    if expected.abs() < 1e-15 && actual.abs() < 1e-15 {
        return;
    }
    let scale = expected.abs().max(1e-300);
    assert!(
        (actual - expected).abs() <= rel * scale,
        "{label}: got {actual:e}, expected {expected:e} (relative error {:e})",
        (actual - expected).abs() / scale
    );
}

fn err_is_invalid<T: std::fmt::Debug>(r: Result<T, SymplexError>, label: &str) {
    assert!(
        matches!(r, Err(SymplexError::InvalidArgument { .. })),
        "{label}: expected an InvalidArgument error, got {r:?}"
    );
}

/// A small deterministic generator for the consistency sweeps
/// (SplitMix64, values in `[0, 1)`).
struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// The scipy / mpmath grid
// ═══════════════════════════════════════════════════════════════════════════

/// Reference values already cited in the crate's earlier test files
/// (`tests/v13`, `tests/v17`) and its doctests, reproduced here through the
/// kernel directly.
#[test]
fn known_scipy_values_from_earlier_tracks() -> Result<(), SymplexError> {
    // scipy: stats.norm.ppf(0.975) = 1.959963984540054
    close(
        norm::ppf(0.975)?,
        1.959963984540054,
        1e-14,
        "norm.ppf(0.975)",
    );
    // scipy: stats.norm.ppf(0.3) = -0.5244005127080407 (v17: normal(-3/2, 7/4).ppf(0.3) = -2.4177008972390714)
    close(
        -1.5 + 1.75 * norm::ppf(0.3)?,
        -2.4177008972390714,
        1e-14,
        "normal(-3/2, 7/4).ppf(0.3)",
    );
    // scipy: stats.t.ppf(0.975, 5) = 2.5705818356363146
    close(
        t::ppf(0.975, 5.0)?,
        2.5705818356363146,
        1e-14,
        "t.ppf(0.975, 5)",
    );
    // scipy: stats.t.ppf(0.3, 3.5) = -0.575338701772231
    close(
        t::ppf(0.3, 3.5)?,
        -0.575338701772231,
        1e-14,
        "t.ppf(0.3, 3.5)",
    );
    // scipy: stats.chi2.ppf(0.3, 7) = 4.671330448981074
    close(
        chi2::ppf(0.3, 7.0)?,
        4.671330448981074,
        1e-14,
        "chi2.ppf(0.3, 7)",
    );
    // scipy: stats.chi2.cdf(5, 7) = 0.34003677030571744
    close(
        chi2::cdf(5.0, 7.0),
        0.34003677030571744,
        1e-14,
        "chi2.cdf(5, 7)",
    );
    // scipy: stats.beta.ppf(0.3, 3/7, 5/2) = 0.02083250526531868
    close(
        beta::ppf(0.3, 3.0 / 7.0, 2.5)?,
        0.02083250526531868,
        1e-14,
        "beta.ppf(0.3, 3/7, 5/2)",
    );
    // scipy: stats.beta.cdf(1/3, 3/7, 5/2) = 0.8522768492555847
    close(
        beta::cdf(1.0 / 3.0, 3.0 / 7.0, 2.5),
        0.8522768492555847,
        1e-14,
        "beta.cdf(1/3, 3/7, 5/2)",
    );
    // scipy: stats.beta.ppf([0.025, 0.975], 9, 6) = (0.3513801106159917, 0.8233889100178821)
    close(
        beta::ppf(0.025, 9.0, 6.0)?,
        0.3513801106159917,
        1e-14,
        "beta.ppf(0.025, 9, 6)",
    );
    close(
        beta::ppf(0.975, 9.0, 6.0)?,
        0.8233889100178821,
        1e-14,
        "beta.ppf(0.975, 9, 6)",
    );
    // scipy: stats.f.ppf(0.3, 5, 9) = 0.6031760598027508; stats.f.cdf(2, 5, 9) = 0.827290123585564
    close(
        f::ppf(0.3, 5.0, 9.0)?,
        0.6031760598027508,
        1e-14,
        "f.ppf(0.3, 5, 9)",
    );
    close(
        f::cdf(2.0, 5.0, 9.0),
        0.827290123585564,
        1e-14,
        "f.cdf(2, 5, 9)",
    );
    // scipy: stats.gamma.ppf(0.3, 3.5, scale=0.75) = 1.7517489183679027; cdf(2) = 0.3806444747602957
    close(
        gamma::ppf(0.3, 3.5, 0.75)?,
        1.7517489183679027,
        1e-14,
        "gamma.ppf(0.3, 3.5, 0.75)",
    );
    close(
        gamma::cdf(2.0, 3.5, 0.75),
        0.3806444747602957,
        1e-14,
        "gamma.cdf(2, 3.5, 0.75)",
    );
    // scipy: stats.binom(9, 3/7).ppf(0.3) = 3, ppf(0.9) = 6
    assert_eq!(binom::ppf(0.3, 9.0, 3.0 / 7.0)?, 3.0);
    assert_eq!(binom::ppf(0.9, 9.0, 3.0 / 7.0)?, 6.0);
    // scipy: stats.poisson(7/3).cdf(0) = 0.09697196786440504; ppf(0.1) = 1, ppf(0.95) = 5
    close(
        poisson::cdf(0.0, 7.0 / 3.0),
        0.09697196786440504,
        1e-14,
        "poisson.cdf(0, 7/3)",
    );
    assert_eq!(poisson::ppf(0.1, 7.0 / 3.0)?, 1.0);
    assert_eq!(poisson::ppf(0.95, 7.0 / 3.0)?, 5.0);
    // scipy: stats.poisson.sf(2, 2) = 0.3233235838169366
    close(
        poisson::sf(2.0, 2.0),
        0.3233235838169366,
        1e-14,
        "poisson.sf(2, 2)",
    );
    Ok(())
}

// Generated with mpmath 1.3 at 40–50 dps (t: bisection on the exact CDF to
// 1e-35; χ²: 400-step bisection in log x so tiny quantiles keep relative
// accuracy) —
// except the χ²(1e8) row, where mpmath's hypergeometric series does not
// converge: those four values come from bisection on a 60-dps `mp.quad` of the
// gamma density itself.  scipy 1.18 is *wrong* there — its
// `gammaincinv(5e7, 1e-10)` gives 99910424.0, and the mass below that point
// by quadrature is 1.18e-10, 18 % too much; the crate's 99910063.36 carries
// mass 9.99999999995e-11.  (On the χ²(1e4) row scipy and mpmath agree to the
// last digit, so scipy is fine as an oracle at moderate `df`.)
const T_PPF: &[(f64, f64, f64)] = &[
    (1.0, 1e-10, -3183098861.837907),
    (1.0, 0.001, -318.30883898555044),
    (1.0, 0.5, 0.0),
    (1.0, 0.999, 318.30883898555044),
    (2.5, 1e-10, -8765.437771364572),
    (2.5, 0.001, -13.822193110865966),
    (2.5, 0.5, 0.0),
    (2.5, 0.999, 13.822193110865966),
    (30.0, 1e-10, -9.377489780407144),
    (30.0, 0.001, -3.385184866829305),
    (30.0, 0.5, 0.0),
    (30.0, 0.999, 3.385184866829305),
    (10000.0, 1e-10, -6.367941351482803),
    (10000.0, 0.001, -3.091047516030612),
    (10000.0, 0.5, 0.0),
    (10000.0, 0.999, 3.091047516030612),
    (100000000.0, 1e-10, -6.361341561862985),
    (100000000.0, 0.001, -3.0902323876691056),
    (100000000.0, 0.5, 0.0),
    (100000000.0, 0.999, 3.0902323876691056),
];
const CHI2_PPF: &[(f64, f64, f64)] = &[
    (1.0, 1e-10, 1.5707963267948965e-20),
    (1.0, 0.001, 1.57079714926249e-06),
    (1.0, 0.5, 0.4549364231195728),
    (1.0, 0.999, 10.827566170662733),
    (2.5, 1e-10, 2.2101150154935887e-08),
    (2.5, 0.001, 0.008815875243366963),
    (2.5, 0.5, 1.8738477677808791),
    (2.5, 0.999, 15.082186971981958),
    (30.0, 1e-10, 3.043040379553346),
    (30.0, 0.001, 11.587951045645056),
    (30.0, 0.5, 29.336031516661585),
    (30.0, 0.999, 59.70306430442993),
    (10000.0, 1e-10, 9126.511802423256),
    (10000.0, 0.001, 9568.668495093969),
    (10000.0, 0.5, 9999.333341235144),
    (10000.0, 0.999, 10442.730565410178),
    (100000000.0, 1e-10, 99910063.3636419),
    (100000000.0, 0.001, 99956303.21524589),
    (100000000.0, 0.5, 99999999.33333333),
    (100000000.0, 0.999, 100043708.18413502),
];

/// Relative tolerance for a quantile: `1e-12` as the kernel promises, loosened
/// to `1e-9` only for the Cauchy (`df = 1`) far tail, where the quantile is
/// `~1/(πp)` and the level itself is known to one ulp (`≈ 1e-16 · 1e10`).
fn quantile_tol(df: f64, p: f64) -> f64 {
    if df <= 1.0 && (p <= 1e-9 || p >= 1.0 - 1e-9) {
        1e-9
    } else {
        1e-12
    }
}

#[test]
fn t_and_chi2_quantiles_match_the_reference_grid() -> Result<(), SymplexError> {
    for &(df, p, expected) in T_PPF {
        close(
            t::ppf(p, df)?,
            expected,
            quantile_tol(df, p),
            &format!("t.ppf({p}, {df})"),
        );
    }
    for &(df, p, expected) in CHI2_PPF {
        close(
            chi2::ppf(p, df)?,
            expected,
            1e-12,
            &format!("chi2.ppf({p}, {df})"),
        );
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Round trips, consistency with the exact CDFs, the Distribution route
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quantiles_round_trip_through_the_cdf() -> Result<(), SymplexError> {
    let levels = [1e-10, 1e-3, 0.5, 0.999];
    for df in [1.0, 2.5, 30.0, 1e4, 1e8] {
        for p in levels {
            let x = t::ppf(p, df)?;
            let back = if p <= 0.5 {
                t::cdf(x, df)
            } else {
                1.0 - t::sf(x, df)
            };
            close(back, p, 1e-12, &format!("t({df}) round trip at p = {p}"));
            let x = chi2::ppf(p, df)?;
            let back = if p <= 0.5 {
                chi2::cdf(x, df)
            } else {
                1.0 - chi2::sf(x, df)
            };
            // The round trip is limited by the f64 quantile itself: one ulp of
            // x moves the χ²(k) tail by ≈ |x − k|/√(2k)·√k·2⁻⁵³ ≈ |z|·√(k/2)·2⁻⁵³
            // relative (z = standardised distance), and the quantile solver
            // stops at 16ε in its log variable (~30 ulps).  At k = 1e8 and
            // z ≈ 3 that is ≈ 30 · 3 · 7000 · 1.1e-16 ≈ 7e-11; elsewhere it is
            // far below 1e-12.
            let z = ((chi2::ppf(p, df)? - df) / (2.0 * df).sqrt())
                .abs()
                .max(1.0);
            let ulp_limit = 30.0 * z * (df / 2.0).sqrt() * f64::EPSILON;
            let tol = f64::max(1e-12, 4.0 * ulp_limit);
            close(back, p, tol, &format!("chi2({df}) round trip at p = {p}"));
            // `isf(q) = ppf(1 − q)` only where `1 − q` is representable: at
            // q = 1e-10 the f64 `1 − q` is off by ~1e-17 relative, which the
            // Cauchy quantile (df = 1) amplifies to 1e-7.  Compare on the
            // smaller tail instead: `isf(q)` against `−ppf(q)` (symmetry).
            close(
                t::isf(p, df)?,
                -t::ppf(p, df)?,
                1e-12,
                &format!("t({df}) isf(q) = −ppf(q) at q = {p}"),
            );
        }
    }
    for (a, b) in [
        (0.5, 0.5),
        (3.0 / 7.0, 2.5),
        (9.0, 6.0),
        (50.0, 50.0),
        (0.05, 3.0),
    ] {
        for p in levels {
            let x = beta::ppf(p, a, b)?;
            let back = if p <= 0.5 {
                beta::cdf(x, a, b)
            } else {
                1.0 - beta::sf(x, a, b)
            };
            close(
                back,
                p,
                1e-12,
                &format!("beta({a}, {b}) round trip at p = {p}"),
            );
            let x = f::ppf(p, 2.0 * a, 2.0 * b)?;
            let back = if p <= 0.5 {
                f::cdf(x, 2.0 * a, 2.0 * b)
            } else {
                1.0 - f::sf(x, 2.0 * a, 2.0 * b)
            };
            close(
                back,
                p,
                1e-12,
                &format!("f({a}, {b}) round trip at p = {p}"),
            );
        }
    }
    for (k, th) in [(0.5, 1.0), (3.5, 0.75), (50.0, 0.1), (1e4, 2.0)] {
        for p in levels {
            let x = gamma::ppf(p, k, th)?;
            let back = if p <= 0.5 {
                gamma::cdf(x, k, th)
            } else {
                1.0 - gamma::sf(x, k, th)
            };
            close(
                back,
                p,
                1e-12,
                &format!("gamma({k}, {th}) round trip at p = {p}"),
            );
        }
    }
    for p in [1e-300, 1e-10, 0.3, 0.5, 0.7, 1.0 - 1e-10] {
        close(
            norm::cdf(norm::ppf(p)?),
            p,
            1e-13,
            &format!("norm round trip at p = {p}"),
        );
        close(
            norm::sf(norm::isf(p)?),
            p,
            1e-13,
            &format!("norm isf round trip at q = {p}"),
        );
    }
    Ok(())
}

/// The kernel and the exact `Ex` distribution functions (evaluated in
/// arbitrary precision by `eval_f64`) agree to `1e-12` at 30 random points
/// per family, so the two routes cannot drift apart.
#[test]
fn numeric_cdfs_match_the_exact_expression_cdfs() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let mut rng = Lcg(20_240_921);
    let exact = |d: &Distribution, x: f64| -> Result<f64, SymplexError> {
        d.cdf(&ctx.from_f64(x)?).eval_f64()
    };
    struct Case {
        name: &'static str,
        dist: Distribution,
        numeric: fn(f64) -> f64,
        // The sampling window inside the support.
        lo: f64,
        hi: f64,
    }
    let cases = [
        Case {
            name: "Normal(1/2, 3/2)",
            dist: Distribution::normal(ctx.rational(1, 2), ctx.rational(3, 2)),
            numeric: |x| norm::cdf((x - 0.5) / 1.5),
            lo: -5.0,
            hi: 6.0,
        },
        Case {
            name: "StudentT(5/2)",
            dist: Distribution::student_t(ctx.rational(5, 2)),
            numeric: |x| t::cdf(x, 2.5),
            lo: -8.0,
            hi: 8.0,
        },
        Case {
            name: "StudentT(30)",
            dist: Distribution::student_t(ctx.int(30)),
            numeric: |x| t::cdf(x, 30.0),
            lo: -6.0,
            hi: 6.0,
        },
        Case {
            name: "ChiSquared(7)",
            dist: Distribution::chi_squared(ctx.int(7)),
            numeric: |x| chi2::cdf(x, 7.0),
            lo: 0.01,
            hi: 30.0,
        },
        Case {
            name: "FDistribution(5, 9)",
            dist: Distribution::f_distribution(ctx.int(5), ctx.int(9)),
            numeric: |x| f::cdf(x, 5.0, 9.0),
            lo: 0.01,
            hi: 8.0,
        },
        Case {
            name: "Beta(3/7, 5/2)",
            dist: Distribution::beta(ctx.rational(3, 7), ctx.rational(5, 2)),
            numeric: |x| beta::cdf(x, 3.0 / 7.0, 2.5),
            lo: 0.001,
            hi: 0.999,
        },
        Case {
            name: "Gamma(7/2, 3/4)",
            dist: Distribution::gamma(ctx.rational(7, 2), ctx.rational(3, 4)),
            numeric: |x| gamma::cdf(x, 3.5, 0.75),
            lo: 0.01,
            hi: 12.0,
        },
    ];
    for case in &cases {
        for _ in 0..30 {
            let x = case.lo + (case.hi - case.lo) * rng.next_f64();
            let expected = exact(&case.dist, x)?;
            close(
                (case.numeric)(x),
                expected,
                1e-12,
                &format!("{} at x = {x}", case.name),
            );
        }
    }
    // Lattice families: the exact CDF is a rational sum / an incomplete gamma.
    let bin = Distribution::binomial(ctx.int(20), ctx.rational(3, 7));
    for k in 0..=20 {
        let expected = bin.cdf(&ctx.int(k)).eval_f64()?;
        close(
            binom::cdf(k as f64, 20.0, 3.0 / 7.0),
            expected,
            1e-12,
            &format!("Binomial(20, 3/7) at k = {k}"),
        );
    }
    let poi = Distribution::poisson(ctx.rational(7, 3));
    for k in 0..12 {
        let expected = poi.cdf(&ctx.int(k)).eval_f64()?;
        close(
            poisson::cdf(k as f64, 7.0 / 3.0),
            expected,
            1e-12,
            &format!("Poisson(7/3) at k = {k}"),
        );
    }
    Ok(())
}

/// `Distribution::quantile_f64` routes the eight families through the
/// kernel (bit-identical to calling it directly), leaves symbolic
/// parameters to the old error, and is fast.
#[test]
fn distribution_quantile_f64_goes_through_the_kernel() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let n = Distribution::normal(ctx.int(2), ctx.int(3));
    assert_eq!(n.quantile_f64(0.3)?, 2.0 + 3.0 * norm::ppf(0.3)?);
    let d = Distribution::student_t(ctx.rational(5, 2));
    assert_eq!(d.quantile_f64(0.999)?, t::ppf(0.999, 2.5)?);
    let d = Distribution::chi_squared(ctx.int(7));
    assert_eq!(d.quantile_f64(1e-3)?, chi2::ppf(1e-3, 7.0)?);
    let d = Distribution::f_distribution(ctx.int(5), ctx.int(9));
    assert_eq!(d.quantile_f64(0.3)?, f::ppf(0.3, 5.0, 9.0)?);
    let d = Distribution::beta(ctx.int(9), ctx.int(6));
    assert_eq!(d.quantile_f64(0.975)?, beta::ppf(0.975, 9.0, 6.0)?);
    let d = Distribution::gamma(ctx.rational(7, 2), ctx.rational(3, 4));
    assert_eq!(d.quantile_f64(0.3)?, gamma::ppf(0.3, 3.5, 0.75)?);
    // Lattice families keep the generic route's 1e-12 slack on F(k) ≥ p.
    let d = Distribution::binomial(ctx.int(9), ctx.rational(3, 7));
    assert_eq!(d.quantile_f64(0.3)?, 3.0);
    assert_eq!(d.quantile_f64(0.9)?, 6.0);
    let d = Distribution::poisson(ctx.rational(7, 3));
    assert_eq!(d.quantile_f64(0.1)?, 1.0);
    assert_eq!(d.quantile_f64(0.95)?, 5.0);
    // Bernoulli(1/2): F(0) = 1/2 exactly, so p = 1/2 picks 0 in both routes.
    let d = Distribution::binomial(ctx.int(1), ctx.rational(1, 2));
    assert_eq!(d.quantile_f64(0.5)?, 0.0);
    // A symbolic parameter still cannot be inverted numerically.
    let sym = Distribution::student_t(ctx.symbol("nu"));
    assert!(sym.quantile_f64(0.5).is_err());
    // Invalid numeric parameters are reported, not iterated on.
    err_is_invalid(
        Distribution::student_t(ctx.int(-1)).quantile_f64(0.5),
        "student_t(-1)",
    );
    Ok(())
}

/// The switch from Brent on 128-bit expressions (≈ 670 ms for one Student-t
/// quantile in debug) to the kernel: measured at well under a millisecond,
/// bounded here at five times the measurement.
#[test]
fn student_t_quantile_is_fast() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let t10 = Distribution::student_t(ctx.int(10));
    // Warm the arena (symbol interning, the parameter's evaluation).
    let q = t10.quantile_f64(0.975)?;
    // scipy: t.ppf(0.975, 10) = 2.228138851986274; mpmath (40 dps) 2.22813885198627475
    close(q, 2.228138851986274, 1e-14, "t.ppf(0.975, 10)");
    let start = Instant::now();
    for _ in 0..10 {
        let _ = t10.quantile_f64(0.975)?;
    }
    let per_call = start.elapsed() / 10;
    assert!(
        per_call.as_secs_f64() < 5e-3,
        "student_t(10).quantile_f64(0.975) took {per_call:?} per call"
    );
    // The same 20 points through `confidence_interval_mean` (Brent on the
    // 128-bit expression took ≈ 407 ms in debug and gave
    // (13.603630327955615, 19.59636967204439), which the kernel reproduces).
    let x = from_i64(&[
        5, 7, 8, 9, 10, 12, 13, 15, 16, 18, 20, 22, 19, 20, 22, 20, 21, 23, 25, 27,
    ]);
    let start = Instant::now();
    let ci = confidence_interval_mean(&x, 0.95)?;
    let elapsed = start.elapsed();
    close(
        ci.lower,
        13.603630327955615,
        1e-11,
        "confidence_interval_mean lower",
    );
    close(
        ci.upper,
        19.59636967204439,
        1e-11,
        "confidence_interval_mean upper",
    );
    assert!(
        elapsed.as_secs_f64() < 2e-2,
        "confidence_interval_mean on 20 points took {elapsed:?}"
    );
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Special functions and conventions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn incomplete_beta_and_gamma_identities() {
    // I_x(1, 1) = x; I_x(a, b) = 1 − I_{1−x}(b, a); I_{0.4}(2, 3) = 0.5248 (a polynomial).
    close(
        betainc_regularized_f64(1.0, 1.0, 0.37),
        0.37,
        1e-15,
        "I_x(1, 1)",
    );
    close(
        betainc_regularized_f64(2.0, 3.0, 0.4),
        0.5248,
        1e-15,
        "I_0.4(2, 3)",
    );
    close(
        betainc_regularized_f64(3.0, 7.0, 0.2) + betainc_regularized_f64(7.0, 3.0, 0.8),
        1.0,
        1e-15,
        "symmetry",
    );
    assert_eq!(betainc_regularized_f64(2.0, 3.0, 0.0), 0.0);
    assert_eq!(betainc_regularized_f64(2.0, 3.0, 1.0), 1.0);
    assert!(betainc_regularized_f64(0.0, 3.0, 0.5).is_nan());
    assert!(betainc_regularized_f64(2.0, 3.0, 1.5).is_nan());
    assert!(betainc_regularized_f64(2.0, f64::NAN, 0.5).is_nan());
    // P(1, x) = 1 − e^{−x}; Q(½, x) = erfc(√x); P + Q = 1.
    close(
        gammainc_lower_regularized_f64(1.0, 0.7),
        1.0 - (-0.7f64).exp(),
        1e-15,
        "P(1, 0.7)",
    );
    // scipy: special.erfc(sqrt(3)) = 0.014305878435429631
    close(
        gammainc_upper_regularized_f64(0.5, 3.0),
        0.014305878435429631,
        1e-14,
        "Q(1/2, 3)",
    );
    close(
        gammainc_lower_regularized_f64(3.5, 2.0) + gammainc_upper_regularized_f64(3.5, 2.0),
        1.0,
        1e-15,
        "P + Q",
    );
    assert_eq!(gammainc_lower_regularized_f64(2.0, 0.0), 0.0);
    assert_eq!(gammainc_upper_regularized_f64(2.0, f64::INFINITY), 0.0);
    assert!(gammainc_lower_regularized_f64(0.0, 1.0).is_nan());
    assert!(gammainc_lower_regularized_f64(1.0, -1.0).is_nan());
}

#[test]
fn conventions_at_the_edges_and_out_of_domain() {
    // ±∞ and NaN arguments.
    assert_eq!(t::cdf(f64::NEG_INFINITY, 3.0), 0.0);
    assert_eq!(t::sf(f64::INFINITY, 3.0), 0.0);
    assert_eq!(t::cdf(0.0, 3.0), 0.5);
    assert!(t::cdf(f64::NAN, 3.0).is_nan());
    assert!(t::cdf(1.0, -3.0).is_nan());
    assert_eq!(chi2::cdf(-1.0, 3.0), 0.0);
    assert_eq!(chi2::sf(f64::INFINITY, 3.0), 0.0);
    assert!(chi2::cdf(1.0, 0.0).is_nan());
    assert_eq!(f::cdf(0.0, 2.0, 3.0), 0.0);
    assert_eq!(f::sf(f64::INFINITY, 2.0, 3.0), 0.0);
    assert_eq!(beta::cdf(1.0, 2.0, 3.0), 1.0);
    assert_eq!(beta::sf(0.0, 2.0, 3.0), 1.0);
    assert!(beta::cdf(0.5, 2.0, -3.0).is_nan());
    assert_eq!(gamma::cdf(0.0, 2.0, 3.0), 0.0);
    assert_eq!(norm::cdf(f64::NEG_INFINITY), 0.0);
    assert_eq!(norm::sf(f64::INFINITY), 0.0);
    assert!(norm::cdf(f64::NAN).is_nan());
    // Lattice families floor their argument; below the support is 0, at or
    // above the last atom is 1; p = 0 and p = 1 are degenerate but valid.
    assert_eq!(binom::cdf(-0.5, 9.0, 0.3), 0.0);
    assert_eq!(binom::cdf(9.0, 9.0, 0.3), 1.0);
    assert_eq!(binom::cdf(3.7, 9.0, 0.3), binom::cdf(3.0, 9.0, 0.3));
    assert_eq!(binom::cdf(3.0, 9.0, 0.0), 1.0);
    assert_eq!(binom::cdf(3.0, 9.0, 1.0), 0.0);
    assert!(binom::cdf(3.0, 9.0, 1.5).is_nan());
    assert_eq!(poisson::cdf(-1.0, 2.0), 0.0);
    assert_eq!(poisson::cdf(2.9, 2.0), poisson::cdf(2.0, 2.0));
    assert!(poisson::cdf(2.0, 0.0).is_nan());
    // Quantiles: the level must lie strictly inside (0, 1), parameters must be valid.
    for p in [0.0, 1.0, -0.1, 1.5, f64::NAN] {
        err_is_invalid(norm::ppf(p), "norm::ppf");
        err_is_invalid(norm::isf(p), "norm::isf");
        err_is_invalid(t::ppf(p, 3.0), "t::ppf");
        err_is_invalid(t::isf(p, 3.0), "t::isf");
        err_is_invalid(chi2::ppf(p, 3.0), "chi2::ppf");
        err_is_invalid(chi2::isf(p, 3.0), "chi2::isf");
        err_is_invalid(f::ppf(p, 3.0, 4.0), "f::ppf");
        err_is_invalid(f::isf(p, 3.0, 4.0), "f::isf");
        err_is_invalid(beta::ppf(p, 3.0, 4.0), "beta::ppf");
        err_is_invalid(beta::isf(p, 3.0, 4.0), "beta::isf");
        err_is_invalid(gamma::ppf(p, 3.0, 4.0), "gamma::ppf");
        err_is_invalid(gamma::isf(p, 3.0, 4.0), "gamma::isf");
        err_is_invalid(binom::ppf(p, 9.0, 0.3), "binom::ppf");
        err_is_invalid(poisson::ppf(p, 2.0), "poisson::ppf");
    }
    err_is_invalid(t::ppf(0.3, 0.0), "t::ppf df = 0");
    err_is_invalid(t::ppf(0.3, f64::NAN), "t::ppf df = NaN");
    err_is_invalid(chi2::ppf(0.3, -1.0), "chi2::ppf df < 0");
    err_is_invalid(f::ppf(0.3, 2.0, 0.0), "f::ppf d2 = 0");
    err_is_invalid(beta::ppf(0.3, 2.0, f64::INFINITY), "beta::ppf β = ∞");
    err_is_invalid(gamma::ppf(0.3, 2.0, -1.0), "gamma::ppf θ < 0");
    err_is_invalid(binom::ppf(0.3, -1.0, 0.5), "binom::ppf n < 0");
    err_is_invalid(binom::ppf(0.3, 9.0, 1.5), "binom::ppf p > 1");
    err_is_invalid(poisson::ppf(0.3, 0.0), "poisson::ppf λ = 0");
    // Errors name the raising function.
    match t::ppf(2.0, 3.0) {
        Err(SymplexError::InvalidArgument { operation, .. }) => {
            assert_eq!(operation, "numdist::t::ppf");
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
    // Infinite degrees of freedom are the normal.
    assert_eq!(t::cdf(1.3, f64::INFINITY), norm::cdf(1.3));
    assert_eq!(t::ppf(0.3, f64::INFINITY).ok(), norm::ppf(0.3).ok());
}

//! 0.22 — `stats::numdist` regressions from the 0.21 differential audit.
//!
//! Reference values cite scipy 1.18 (`symplex/.venv/bin/python`) by the call
//! that produced them; where scipy is itself inaccurate — it is in the far
//! tails: `scipy.stats.t.sf(1e160, 1) = 0.0`, `scipy.stats.norm.cdf(-38) =
//! 0.0` — the value is mpmath 1.3 at `mp.dps = 30` and the scipy number is
//! quoted alongside.  For the two-large-shape incomplete beta (`basym`)
//! mpmath's `betainc` does not converge; those values are the finite sum
//! `I_x(a, b) = P(Bin(a + b − 1, x) ≥ a)` (integer `a`, `b`) accumulated in
//! mpmath at 30 digits, cross-checked against 40-digit `mp.quad` of the
//! density where the two agree.  Nothing below was computed by hand.

// Reference values are quoted at the 17 digits mpmath printed them.
#![allow(clippy::excessive_precision)]

use std::time::{Duration, Instant};

use symplex::prelude::*;
use symplex::stats::numdist::{
    beta, betainc_regularized_f64, binom, chi2, gamma, norm, poisson, t,
};

/// Relative closeness, absolute when the reference is (essentially) zero.
fn close(actual: f64, expected: f64, rel: f64, label: &str) {
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

/// Closeness for a subnormal reference: relative `rel` or `ulps` units of
/// the subnormal spacing `2⁻¹⁰⁷⁴`, whichever is larger — a subnormal carries
/// fewer than 53 bits, so a relative tolerance alone would be unmeetable.
fn close_subnormal(actual: f64, expected: f64, rel: f64, ulps: f64, label: &str) {
    let spacing = f64::from_bits(1);
    let tol = (rel * expected.abs()).max(ulps * spacing);
    assert!(
        (actual - expected).abs() <= tol,
        "{label}: got {actual:e}, expected {expected:e} (difference {:e}, tolerance {tol:e})",
        (actual - expected).abs()
    );
}

fn err_is_invalid<T: std::fmt::Debug>(r: Result<T, SymplexError>, label: &str) {
    assert!(
        matches!(r, Err(SymplexError::InvalidArgument { .. })),
        "{label}: expected an InvalidArgument error, got {r:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// A1 — Student t beyond |x| = 1.34e154, where x² overflows
// ═══════════════════════════════════════════════════════════════════════════

/// `t::sf(1e160, 1)`: mpmath `betainc(1/2, 1/2, 0, 1/(1 + 1e320), regularized=True)/2`
/// = 3.1830988618379067e-161 (= 1/(π·1e160)); scipy `stats.t.sf(1e160, 1) = 0.0`.
#[test]
fn t_sf_cauchy_far_beyond_the_x_squared_overflow() {
    close(
        t::sf(1e160, 1.0),
        3.1830988618379067e-161,
        1e-12,
        "t.sf(1e160, 1)",
    );
    // and it is continuous across the power-law switch:
    // mpmath t.sf(1e154, 1) = 3.1830988618379067e-155 (scipy 3.1830988618379068e-155)
    close(
        t::sf(1e154, 1.0),
        3.1830988618379067e-155,
        1e-12,
        "t.sf(1e154, 1)",
    );
    // mpmath t.sf(1e155, 1) = 3.1830988618379067e-156 (scipy 0.0)
    close(
        t::sf(1e155, 1.0),
        3.1830988618379067e-156,
        1e-12,
        "t.sf(1e155, 1)",
    );
    // mpmath t.sf(1e300, 1) = 3.1830988618379067e-301 (scipy 0.0)
    close(
        t::sf(1e300, 1.0),
        3.1830988618379067e-301,
        1e-12,
        "t.sf(1e300, 1)",
    );
}

/// `t::sf(1e200, 0.5)`: mpmath `betainc(1/4, 1/2, 0, 0.5/(0.5 + 1e400), regularized=True)/2`
/// = 3.207009754142229e-101; scipy `stats.t.sf(1e200, 0.5) = 0.0`.
#[test]
fn t_sf_half_degree_of_freedom_at_1e200() {
    close(
        t::sf(1e200, 0.5),
        3.207009754142229e-101,
        1e-12,
        "t.sf(1e200, 0.5)",
    );
}

/// `t::cdf(-1e170, 1.5)`: mpmath 3.7708524320162464e-256; scipy `stats.t.cdf(-1e170, 1.5) = 0.0`.
#[test]
fn t_cdf_lower_tail_at_minus_1e170() {
    close(
        t::cdf(-1e170, 1.5),
        3.7708524320162464e-256,
        1e-12,
        "t.cdf(-1e170, 1.5)",
    );
    // The body on the other side is exactly 1 at that distance.
    assert_eq!(t::sf(-1e170, 1.5), 1.0);
    // ±∞ keep their exact values.
    assert_eq!(t::sf(f64::INFINITY, 1.5), 0.0);
    assert_eq!(t::cdf(f64::NEG_INFINITY, 1.5), 0.0);
    assert_eq!(t::cdf(f64::INFINITY, 1.5), 1.0);
}

/// `t::ppf(1e-200, 1)`: the Cauchy quantile is `−cot(πp)`; mpmath
/// `-cot(pi*mpf('1e-200'))` = −3.1830988618379067e199 (scipy −3.183098861837907e199).
/// The kernel used to return −1.34e154, where its `sf` first became 0.
#[test]
fn t_ppf_cauchy_at_1e_minus_200() -> Result<(), SymplexError> {
    close(
        t::ppf(1e-200, 1.0)?,
        -3.1830988618379067e199,
        1e-11,
        "t.ppf(1e-200, 1)",
    );
    Ok(())
}

/// `t::ppf(1e-160, 1)`: mpmath `-cot(pi*mpf('1e-160'))` = −3.1830988618379067e159
/// (scipy −3.1830988618379068e159).
#[test]
fn t_ppf_cauchy_at_1e_minus_160() -> Result<(), SymplexError> {
    close(
        t::ppf(1e-160, 1.0)?,
        -3.1830988618379067e159,
        1e-11,
        "t.ppf(1e-160, 1)",
    );
    Ok(())
}

/// `t::isf(3.120729288632323e-231, 1)`: mpmath `cot(pi*mpf('3.120729288632323e-231'))`
/// = 1.0199855762666673e230 (scipy 1.0199855762666672e230).
#[test]
fn t_isf_cauchy_at_3e_minus_231() -> Result<(), SymplexError> {
    close(
        t::isf(3.120729288632323e-231, 1.0)?,
        1.0199855762666673e230,
        1e-11,
        "t.isf(3.12e-231, 1)",
    );
    Ok(())
}

/// `t::ppf(1e-100, 0.5)`: mpmath `findroot` on `ln sf(eᵘ, ½) − ln 1e-100`
/// (`sf` by `betainc` at 30 digits) gives `x = −1.02849115631634e199`, and
/// `sf` at that point reproduces 1e-100.  scipy `stats.t.ppf(1e-100, 0.5)`
/// = −4.740375954054589e153 is wrong (its own `sf` underflows there).
#[test]
fn t_ppf_half_degree_of_freedom_at_1e_minus_100() -> Result<(), SymplexError> {
    let x = t::ppf(1e-100, 0.5)?;
    close(x, -1.02849115631634e199, 1e-11, "t.ppf(1e-100, 0.5)");
    // and the round trip holds
    close(
        t::cdf(x, 0.5),
        1e-100,
        1e-11,
        "t.cdf(t.ppf(1e-100, 0.5), 0.5)",
    );
    Ok(())
}

/// A quantile beyond the largest double is `∓∞`, not a clamp: mpmath
/// `t.sf(1.7976931348623157e308, 0.1)` = 6.2382159555909198e-32 and
/// `t.sf(1.7976931348623157e308, 0.5)` = 2.3918971474675349e-155, so
/// levels below those have no finite quantile (scipy returns
/// `-2.1199605744342963e153` and `-4.740375954054589e153` there, its
/// `sf` having underflowed).  Just above them the quantile is finite
/// and round-trips.
#[test]
fn t_ppf_beyond_the_largest_double_is_infinite() -> Result<(), SymplexError> {
    assert_eq!(t::ppf(1e-50, 0.1)?, f64::NEG_INFINITY);
    assert_eq!(t::isf(1e-50, 0.1)?, f64::INFINITY);
    assert_eq!(t::ppf(1e-300, 0.5)?, f64::NEG_INFINITY);
    assert_eq!(t::isf(1e-300, 0.5)?, f64::INFINITY);
    let x = t::ppf(1e-150, 0.5)?;
    assert!(x.is_finite() && x < -1e290, "t.ppf(1e-150, 0.5) = {x:e}");
    close(
        t::cdf(x, 0.5),
        1e-150,
        1e-11,
        "t.cdf(t.ppf(1e-150, 0.5), 0.5)",
    );
    Ok(())
}

/// `t::pdf` against `scipy.stats.t.pdf` (moderate arguments) and mpmath
/// `Γ((ν+1)/2)/(√(νπ) Γ(ν/2)) (1 + x²/ν)^{−(ν+1)/2}` where scipy overflows.
#[test]
fn t_pdf_matches_scipy_and_mpmath() {
    // scipy: stats.t.pdf(0.7, 3.5) = 0.2768478555390718
    close(
        t::pdf(0.7, 3.5),
        0.2768478555390718,
        1e-14,
        "t.pdf(0.7, 3.5)",
    );
    // scipy: stats.t.pdf(-2.3, 1) = 0.050605705275642406
    close(
        t::pdf(-2.3, 1.0),
        0.050605705275642406,
        1e-14,
        "t.pdf(-2.3, 1)",
    );
    // scipy: stats.t.pdf(5, 30) = 3.288890205292232e-05
    close(
        t::pdf(5.0, 30.0),
        3.288890205292232e-05,
        1e-13,
        "t.pdf(5, 30)",
    );
    // scipy: stats.t.pdf(0, 2) = 0.3535533905932738 (= 1/(2√2))
    close(t::pdf(0.0, 2.0), 0.3535533905932738, 1e-14, "t.pdf(0, 2)");
    // mpmath t.pdf(1e160, 0.5) = 1.6035048770711145e-241; scipy 0.0 (overflow in x²)
    close(
        t::pdf(1e160, 0.5),
        1.6035048770711145e-241,
        1e-12,
        "t.pdf(1e160, 0.5)",
    );
    // mpmath t.pdf(1e200, 1) = 3.18e-401: below the smallest double
    assert_eq!(t::pdf(1e200, 1.0), 0.0);
    assert_eq!(t::pdf(f64::INFINITY, 1.0), 0.0);
    assert!(t::pdf(f64::NAN, 1.0).is_nan());
}

// ═══════════════════════════════════════════════════════════════════════════
// A2 — Temme's expansion where Cody's erfc underflows (z ≳ 26.5)
// ═══════════════════════════════════════════════════════════════════════════

/// `gamma::sf(5.027e7, 5e7)`: mpmath `gammainc(mpf(5e7), mpf(5.027e7), inf, regularized=True)`
/// = 3.5745492319883549e-318 (scipy `stats.gamma.sf(5.027e7, 5e7) = 3.57455e-318`,
/// a subnormal).  The kernel returned −6.433e-321.
#[test]
fn gamma_sf_deep_temme_tail_is_positive_and_right() {
    let q = gamma::sf(5.027e7, 5e7, 1.0);
    assert!(q > 0.0, "gamma.sf(5.027e7, 5e7) = {q:e} must be positive");
    close_subnormal(
        q,
        3.5745492319883549e-318,
        1e-12,
        8.0,
        "gamma.sf(5.027e7, 5e7)",
    );
    // χ²(1e8) is the same tail: scipy stats.chi2.sf(1.0054e8, 1e8) = 3.57455e-318
    close_subnormal(
        chi2::sf(1.0054e8, 1e8),
        3.5745492319883549e-318,
        1e-12,
        8.0,
        "chi2.sf(1.0054e8, 1e8)",
    );
    // deeper still: mpmath gamma.sf(5.0275e7, 5e7) = 5.955e-330, below the smallest double
    assert_eq!(gamma::sf(5.0275e7, 5e7, 1.0), 0.0);
}

/// `gamma::cdf(4.973e7, 5e7)`: mpmath (the power series
/// `xᵃe⁻ˣ/Γ(a)·Σ xⁿ/(a(a+1)⋯(a+n))`, 13 580 terms at 30 digits)
/// = 1.8778449525391997e-320; scipy `stats.gamma.cdf(4.973e7, 5e7) = 1.8774e-320`
/// (one subnormal step low).  The kernel returned 3.5e-323.
#[test]
fn gamma_cdf_deep_temme_lower_tail() {
    close_subnormal(
        gamma::cdf(4.973e7, 5e7, 1.0),
        1.8778449525391997e-320,
        1e-12,
        8.0,
        "gamma.cdf(4.973e7, 5e7)",
    );
    // Just inside the normal range: mpmath gamma.sf(5.0265e7, 5e7) = 1.3088239274803528e-306
    // (scipy 1.3088239274801196e-306)
    close(
        gamma::sf(5.0265e7, 5e7, 1.0),
        1.3088239274803528e-306,
        1e-11,
        "gamma.sf(5.0265e7, 5e7)",
    );
    // mpmath gamma.cdf(4.9735e7, 5e7) = 9.1544625064343368e-309 (scipy 9.154252168220994e-309)
    close_subnormal(
        gamma::cdf(4.9735e7, 5e7, 1.0),
        9.1544625064343368e-309,
        1e-11,
        8.0,
        "gamma.cdf(4.9735e7, 5e7)",
    );
}

/// The deep *lower* Temme tail against the mpmath power series (`mp.dps =
/// 30`, 40 646 terms): `P(5e7, 49929289.32188135)` (`x = a − 10√a`) =
/// 7.2687198106979244e-24, where scipy `special.gammainc` gives
/// 6.8684720299268562e-24 — 5.5 % off; at `a = 1e9`, `x = a − 10√a =
/// 999683772.2339832`, mpmath 7.5399586583055715e-24 against scipy's
/// 3.5684120028617318e-24.  One ulp of `x` moves these tails by
/// `≈ |z|/√a · ulp(x)` ≈ 4e-11 relative, which bounds what any `f64` kernel
/// can promise here.
#[test]
fn gamma_cdf_deep_lower_tail_where_scipy_is_wrong() {
    close(
        gamma::cdf(49929289.32188135, 5e7, 1.0),
        7.2687198106979244e-24,
        1e-10,
        "gamma.cdf(5e7 − 10√5e7, 5e7)",
    );
    close(
        gamma::cdf(999683772.2339832, 1e9, 1.0),
        7.5399586583055715e-24,
        1e-10,
        "gamma.cdf(1e9 − 10√1e9, 1e9)",
    );
}

/// `chi2::isf(1e-310, 1e8)`: scipy `stats.chi2.isf(1e-310, 1e8) = 100533581.52867371`;
/// mpmath `findroot` on `ln Q(5e7, x) − ln 1e-310` gives `x = 50266790.764336854`,
/// i.e. the same 100533581.52867371 for χ².  The kernel failed with
/// `ComputationFailed("not a number at 17.73…")`.
#[test]
fn chi2_and_gamma_isf_at_1e_minus_310() -> Result<(), SymplexError> {
    close(
        chi2::isf(1e-310, 1e8)?,
        100533581.52867371,
        1e-12,
        "chi2.isf(1e-310, 1e8)",
    );
    close(
        gamma::isf(1e-310, 5e7, 1.0)?,
        50266790.764336854,
        1e-12,
        "gamma.isf(1e-310, 5e7)",
    );
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// A3 — Φ(x) in the subnormal band x ∈ (−38.5, −37.5)
// ═══════════════════════════════════════════════════════════════════════════

/// `norm::cdf(-37.6)`: mpmath `ncdf(mpf('-37.6'))` = 1.0748112495871029e-309
/// (scipy 1.074811249586866e-309).  The kernel returned exactly 0.
#[test]
fn norm_cdf_at_minus_37_6_is_subnormal_not_zero() {
    close_subnormal(
        norm::cdf(-37.6),
        1.0748112495871029e-309,
        1e-13,
        4.0,
        "norm.cdf(-37.6)",
    );
    // the upper tail is the mirror image
    close_subnormal(
        norm::sf(37.6),
        1.0748112495871029e-309,
        1e-13,
        4.0,
        "norm.sf(37.6)",
    );
}

/// `norm::cdf(-38.0)`: mpmath `ncdf(mpf('-38'))` = 2.8854283600687843e-316; scipy 0.0.
#[test]
fn norm_cdf_at_minus_38_is_subnormal_not_zero() {
    close_subnormal(
        norm::cdf(-38.0),
        2.8854283600687843e-316,
        1e-13,
        4.0,
        "norm.cdf(-38.0)",
    );
    // mpmath ncdf(-38.4) = 6.6015998543264075e-323: 13 or 14 subnormal steps
    close_subnormal(
        norm::cdf(-38.4),
        6.6015998543264075e-323,
        1e-13,
        1.0,
        "norm.cdf(-38.4)",
    );
    // below the smallest double it is 0, and −∞ stays exact
    assert_eq!(norm::cdf(-39.0), 0.0);
    assert_eq!(norm::cdf(f64::NEG_INFINITY), 0.0);
}

/// The normal range on either side of the switch is unchanged:
/// mpmath `ncdf(-37.4)` = 1.9536815616488883e-306 (scipy 1.9536815616486757e-306),
/// `ncdf(-37.5)` = 4.6053530095819548e-308 (scipy 4.605353009581956e-308),
/// `ncdf(-30)` = 4.9067139271481871e-198, `ncdf(-26)` = 2.4760633155033893e-149.
#[test]
fn norm_cdf_normal_range_around_the_erfc_switch() {
    close(
        norm::cdf(-37.4),
        1.9536815616488883e-306,
        1e-13,
        "norm.cdf(-37.4)",
    );
    close(
        norm::cdf(-37.5),
        4.6053530095819548e-308,
        1e-13,
        "norm.cdf(-37.5)",
    );
    close(
        norm::cdf(-30.0),
        4.9067139271481871e-198,
        1e-14,
        "norm.cdf(-30)",
    );
    close(
        norm::cdf(-26.0),
        2.4760633155033893e-149,
        1e-14,
        "norm.cdf(-26)",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// A4 — discrete quantiles above 2⁵³ used to hang
// ═══════════════════════════════════════════════════════════════════════════

/// `poisson::ppf(0.5, 1e17)`, `poisson::ppf(0.5, 1e300)`, `binom::ppf(0.5, 1e17, 0.5)`
/// looped for ever once the bisection bracket was two adjacent doubles.
/// scipy `stats.poisson.ppf(0.5, 1e17) = nan`, `stats.binom.ppf(0.5, 1e17, 0.5) = nan`.
/// Now an `InvalidArgument` naming the 2⁵³ limit, within the time guard.
#[test]
fn discrete_ppf_above_2_53_is_an_error_not_a_hang() {
    let start = Instant::now();
    let r = poisson::ppf(0.5, 1e17);
    err_is_invalid(r, "poisson::ppf(0.5, 1e17)");
    err_is_invalid(poisson::ppf(0.5, 1e300), "poisson::ppf(0.5, 1e300)");
    err_is_invalid(binom::ppf(0.5, 1e17, 0.5), "binom::ppf(0.5, 1e17, 0.5)");
    err_is_invalid(poisson::isf(0.5, 1e17), "poisson::isf(0.5, 1e17)");
    err_is_invalid(binom::isf(0.5, 1e17, 0.5), "binom::isf(0.5, 1e17, 0.5)");
    // one step past the limit is rejected, the limit itself accepted
    err_is_invalid(
        poisson::ppf(0.5, 9_007_199_254_740_994.0),
        "poisson::ppf(0.5, 2^53 + 2)",
    );
    match poisson::ppf(0.5, 1e17) {
        Err(SymplexError::InvalidArgument { reason, .. }) => {
            assert!(
                reason.contains("2^53"),
                "message should name the limit: {reason}"
            );
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
    assert!(
        start.elapsed() < Duration::from_secs(2),
        "the rejections took {:?}",
        start.elapsed()
    );
}

/// Below the limit the search terminates and lands on the exact lattice
/// point.  mpmath (`mp.dps = 30`): `poisson.cdf(k, 1e15) = Q(k + 1, 1e15)`
/// is 0.97499999962290784662 at `k = 1000000061979503` and
/// 0.97500000147110316045 at `k = 1000000061979504`, so
/// `poisson.ppf(0.975, 1e15) = 1000000061979504` (scipy says
/// 1000000061979503 — one short, its own `cdf` there is 0.9749999996);
/// `Q(1e15 + 1, 1e15)` = 0.50000000841044174072 and `Q(1e15, 1e15)` =
/// 0.49999999579477913062, so `poisson.ppf(0.5, 1e15) = 1e15` (scipy `nan`).
/// `binom.cdf(k, 1e15, 0.3) = I_{0.7}(1e15 − k, k + 1)` by 40-digit
/// quadrature of the density is 0.024999999272034544 at
/// `k = 299999971597423` and 0.025000003305127770 at `k = 299999971597424`,
/// so `binom.ppf(0.025, 1e15, 0.3) = 299999971597424` (scipy says
/// 299999971597427; its `binom.cdf` oscillates between 0.0244 and 0.0254
/// over these `k`).  `binom.ppf(0.5, 1e15, 0.5) = 5e14` by symmetry
/// (`P(X ≤ n/2) = ½ + ½P(X = n/2)`), scipy agrees.
#[test]
fn discrete_ppf_on_a_1e15_lattice_terminates_and_matches_mpmath() -> Result<(), SymplexError> {
    let start = Instant::now();
    assert_eq!(poisson::ppf(0.975, 1e15)?, 1000000061979504.0);
    assert_eq!(poisson::ppf(0.5, 1e15)?, 1e15);
    assert_eq!(binom::ppf(0.5, 1e15, 0.5)?, 500000000000000.0);
    assert_eq!(binom::ppf(0.025, 1e15, 0.3)?, 299999971597424.0);
    // the defining inequalities hold in the kernel's own CDF
    assert!(poisson::cdf(1000000061979504.0, 1e15) >= 0.975);
    assert!(poisson::cdf(1000000061979503.0, 1e15) < 0.975);
    assert!(
        start.elapsed() < Duration::from_secs(2),
        "the four quantiles took {:?}",
        start.elapsed()
    );
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// A5 — poisson::cdf(±∞)
// ═══════════════════════════════════════════════════════════════════════════

/// scipy `stats.poisson.cdf(inf, 3.5) = 1.0`, `sf(inf) = 0.0`, `cdf(-inf) = 0.0`,
/// `sf(-inf) = 1.0`.  The kernel returned `NaN` at `+∞`.
#[test]
fn poisson_cdf_at_infinity() {
    assert_eq!(poisson::cdf(f64::INFINITY, 3.5), 1.0);
    assert_eq!(poisson::sf(f64::INFINITY, 3.5), 0.0);
    assert_eq!(poisson::cdf(f64::NEG_INFINITY, 3.5), 0.0);
    assert_eq!(poisson::sf(f64::NEG_INFINITY, 3.5), 1.0);
    // the binomial already did this: stats.binom.cdf(inf, 9, 0.3) = 1.0
    assert_eq!(binom::cdf(f64::INFINITY, 9.0, 0.3), 1.0);
    assert_eq!(binom::sf(f64::NEG_INFINITY, 9.0, 0.3), 1.0);
    // a NaN argument is still NaN
    assert!(poisson::cdf(f64::NAN, 3.5).is_nan());
}

// ═══════════════════════════════════════════════════════════════════════════
// A6 — binom::isf, poisson::isf (new), binom::sf, the basym branch
// ═══════════════════════════════════════════════════════════════════════════

/// `scipy.stats.binom.isf(q, n, p)`: (0.3, 9, 3/7) = 5; (0.9, 9, 3/7) = 2;
/// (0.5, 1, 0.5) = 0; (0.5, 20, 0.5) = 10; (1e-6, 100, 0.2) = 41;
/// (0.999, 100, 0.2) = 9; (0.05, 1000, 0.01) = 15; (0.5, 0, 0.3) = 0;
/// (0.2, 5, 0) = 0; (0.2, 5, 1) = 5.
#[test]
fn binom_isf_matches_scipy() -> Result<(), SymplexError> {
    assert_eq!(binom::isf(0.3, 9.0, 3.0 / 7.0)?, 5.0);
    assert_eq!(binom::isf(0.9, 9.0, 3.0 / 7.0)?, 2.0);
    assert_eq!(binom::isf(0.5, 1.0, 0.5)?, 0.0);
    assert_eq!(binom::isf(0.5, 20.0, 0.5)?, 10.0);
    assert_eq!(binom::isf(1e-6, 100.0, 0.2)?, 41.0);
    assert_eq!(binom::isf(0.999, 100.0, 0.2)?, 9.0);
    assert_eq!(binom::isf(0.05, 1000.0, 0.01)?, 15.0);
    assert_eq!(binom::isf(0.5, 0.0, 0.3)?, 0.0);
    assert_eq!(binom::isf(0.2, 5.0, 0.0)?, 0.0);
    assert_eq!(binom::isf(0.2, 5.0, 1.0)?, 5.0);
    // the defining property: sf(k − 1) > q ≥ sf(k)
    // scipy: binom.sf(40, 100, 0.2) = 1.2921842431095618e-06, sf(41) = 4.4478087547615315e-07
    assert!(binom::sf(40.0, 100.0, 0.2) > 1e-6 && binom::sf(41.0, 100.0, 0.2) <= 1e-6);
    // invalid levels and parameters are rejected like ppf's
    for q in [0.0, 1.0, -0.1, f64::NAN] {
        err_is_invalid(binom::isf(q, 9.0, 0.3), "binom::isf level");
    }
    err_is_invalid(binom::isf(0.3, -1.0, 0.5), "binom::isf n < 0");
    err_is_invalid(binom::isf(0.3, 9.0, 1.5), "binom::isf p > 1");
    Ok(())
}

/// `scipy.stats.poisson.isf(q, mu)`: (0.1, 7/3) = 4; (0.95, 7/3) = 0;
/// (0.5, 7/3) = 2; (1e-10, 1e6) = 1006368; (1e-6, 0.5) = 7; (0.999, 100) = 71;
/// (0.3, 1e4) = 10052.
#[test]
fn poisson_isf_matches_scipy() -> Result<(), SymplexError> {
    assert_eq!(poisson::isf(0.1, 7.0 / 3.0)?, 4.0);
    assert_eq!(poisson::isf(0.95, 7.0 / 3.0)?, 0.0);
    assert_eq!(poisson::isf(0.5, 7.0 / 3.0)?, 2.0);
    assert_eq!(poisson::isf(1e-10, 1e6)?, 1006368.0);
    assert_eq!(poisson::isf(1e-6, 0.5)?, 7.0);
    assert_eq!(poisson::isf(0.999, 100.0)?, 71.0);
    assert_eq!(poisson::isf(0.3, 1e4)?, 10052.0);
    // scipy: poisson.sf(1006367, 1e6) = 1.0026993934898231e-10, sf(1006368) = 9.962050727877579e-11
    assert!(poisson::sf(1006367.0, 1e6) > 1e-10 && poisson::sf(1006368.0, 1e6) <= 1e-10);
    for q in [0.0, 1.0, 1.5, f64::NAN] {
        err_is_invalid(poisson::isf(q, 2.0), "poisson::isf level");
    }
    err_is_invalid(poisson::isf(0.3, 0.0), "poisson::isf λ = 0");
    match poisson::isf(2.0, 3.0) {
        Err(SymplexError::InvalidArgument { operation, .. }) => {
            assert_eq!(operation, "numdist::poisson::isf");
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
    Ok(())
}

/// `binom::sf` against `scipy.stats.binom.sf(k, n, p)`: (3, 9, 3/7) =
/// 0.5878990445637241; (0, 9, 3/7) = 0.9935038273034675; (8, 9, 3/7) =
/// 0.0004877630889352714; (50, 100, 0.2) = 5.17989263752495e-12; (150,
/// 1000, 0.1) = 2.774440900996004e-07; (2, 20, 0.05) = 0.07548367378849626;
/// (10.7, 20, 0.5) = 0.41190147399902344; (−1, 5, 0.5) = 1; (5, 5, 0.5) = 0.
#[test]
fn binom_sf_matches_scipy() {
    close(
        binom::sf(3.0, 9.0, 3.0 / 7.0),
        0.5878990445637241,
        1e-14,
        "binom.sf(3, 9, 3/7)",
    );
    close(
        binom::sf(0.0, 9.0, 3.0 / 7.0),
        0.9935038273034675,
        1e-14,
        "binom.sf(0, 9, 3/7)",
    );
    close(
        binom::sf(8.0, 9.0, 3.0 / 7.0),
        0.0004877630889352714,
        1e-14,
        "binom.sf(8, 9, 3/7)",
    );
    close(
        binom::sf(50.0, 100.0, 0.2),
        5.17989263752495e-12,
        1e-13,
        "binom.sf(50, 100, 0.2)",
    );
    close(
        binom::sf(150.0, 1000.0, 0.1),
        2.774440900996004e-07,
        1e-13,
        "binom.sf(150, 1000, 0.1)",
    );
    close(
        binom::sf(2.0, 20.0, 0.05),
        0.07548367378849626,
        1e-14,
        "binom.sf(2, 20, 0.05)",
    );
    // the argument is floored
    close(
        binom::sf(10.7, 20.0, 0.5),
        0.41190147399902344,
        1e-14,
        "binom.sf(10.7, 20, 0.5)",
    );
    assert_eq!(binom::sf(-1.0, 5.0, 0.5), 1.0);
    assert_eq!(binom::sf(5.0, 5.0, 0.5), 0.0);
}

/// The two-large-shape expansion `basym` of `betainc_regularized_f64`.
/// Entry condition (`bratio_route`): both shapes above 1, the smaller
/// above 100, and `λ = a − (a + b)·x` (or its mirror image when `x` is
/// above the mean) at most `0.03 × min(a, b)`.  For `(2e5, 3e5)` the
/// mean is `0.4` and that window is `|x − 0.4| ≤ 0.012`; for `(1e6, 1e6)`
/// it is `|x − 0.5| ≤ 0.015`.
///
/// Reference: the finite sum `I_x(a, b) = P(Bin(a + b − 1, x) ≥ a)` in
/// mpmath at 30 digits (`mp.dps = 30`, terms until `< 1e-32` relative),
/// which agrees to 17 digits with 40-digit `mp.quad` of the density for
/// the central cases (`mpmath.betainc` itself raises `NoConvergence`).
#[test]
fn basym_two_large_shapes_near_the_mean() {
    // I_0.4(2e5, 3e5) = 0.50007677650083536  (scipy special.betainc = 0.500076776499733)
    close(
        betainc_regularized_f64(2e5, 3e5, 0.4),
        0.50007677650083536,
        1e-12,
        "I_0.4(2e5, 3e5)",
    );
    // I_0.5001(1e6, 1e6) = 0.6113512821376899  (scipy 0.6113512821376825)
    close(
        betainc_regularized_f64(1e6, 1e6, 0.5001),
        0.6113512821376899,
        1e-12,
        "I_0.5001(1e6, 1e6)",
    );
    // I_0.4999(1e6, 1e6) = 0.3886487178623101  (scipy 0.38864871786231747)
    close(
        beta::cdf(0.4999, 1e6, 1e6),
        0.3886487178623101,
        1e-12,
        "I_0.4999(1e6, 1e6)",
    );
}

/// `basym` on the tails of its window, where the value is small and must be
/// computed directly: `I_0.39(2e5, 3e5)` (`λ = 5000 ≤ 6000`) =
/// 8.6699388658990661e-48 by the binomial sum (scipy 8.669938865898479e-48;
/// the 40-digit quadrature is off here, 8.6699396e-48, so it is *not* the
/// oracle for this one), and the upper tail at `x = 0.405` (`λ = −2500`,
/// mirrored to `I_0.595(3e5, 2e5)`) = 2.8545525000973192e-13 (scipy
/// `special.betaincc(2e5, 3e5, 0.405)` = 2.8545525000629e-13).
#[test]
fn basym_small_tails_inside_its_window() {
    close(
        betainc_regularized_f64(2e5, 3e5, 0.39),
        8.6699388658990661e-48,
        1e-11,
        "I_0.39(2e5, 3e5)",
    );
    close(
        beta::sf(0.405, 2e5, 3e5),
        2.8545525000973192e-13,
        1e-11,
        "1 − I_0.405(2e5, 3e5)",
    );
    // and the complementary tails are consistent
    let lower = beta::cdf(0.405, 2e5, 3e5);
    close(
        lower,
        1.0 - 2.8545525000973192e-13,
        1e-15,
        "I_0.405(2e5, 3e5)",
    );
}

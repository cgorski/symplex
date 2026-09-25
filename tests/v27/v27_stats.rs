//! Discrete quantiles decided on the smaller tail, the remaining
//! non-cancelling tail forms, and the two-sample t-test sample size at a
//! tiny effect.
//!
//! 1. `Distribution::quantile_f64` on a lattice compared `F(k)` with
//!    `p − 10⁻¹²` — in the `numdist` kernels (binomial, Poisson), in the
//!    walk of a family without a closed CDF, in the bracketing of the
//!    others and in the walk of a table.  Every level below `10⁻¹²` became
//!    the smallest double and every level within `10⁻¹²` of `1` a smaller
//!    one.  Now the kernels get `p` as given, and every other discrete
//!    family decides `F(k) ≥ p` exactly (sign of `F(k) − p` in arbitrary
//!    precision), on the smaller tail `S(k) ≤ 1 − p` for `p > ½`.
//! 2. Tail forms that relied on `eval` not folding `γ(1, y)` into
//!    `1 − e^{−y}` (it does at an irrational `y`), and families without
//!    one: they evaluated to `0` in a far tail.
//! 3. `sample_size_t_test_two_sample(1e-4, 0.05, 0.8)`, verified.
//!
//! Reference values: scipy 1.18.1 / statsmodels 0.15.0 by the call quoted,
//! adjudicated by mpmath 1.3 at `mp.dps = 50` with exact inputs where scipy
//! clamps (scripts `target/scratch/q_oracle*.py`, `tails_oracle*.py`,
//! `nct_ss.py` of this session).

// Reference values are quoted at the 20 digits mpmath printed them.
#![allow(clippy::excessive_precision)]

use std::time::{Duration, Instant};

use symplex::prelude::*;
use symplex::stats::hypothesis::{power_t_test_two_sample, sample_size_t_test_two_sample};
use symplex::stats::{Distribution, Family, Kind, Piece, Support, same_family};

/// `1 − 2⁻⁵³`, the largest double below `1`; `1 − p = 2⁻⁵³` exactly.
const TOP: f64 = 1.0 - f64::EPSILON / 2.0;

/// `actual` within `rel` of `expected`, relatively.
fn close(actual: f64, expected: f64, rel: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= rel * expected.abs(),
        "{label}: got {actual:e}, expected {expected:e}"
    );
}

/// A value possibly below the `f64` range, compared through its decimal
/// expansion: `eval_decimal(20)` against mpmath's digits, relatively.
fn close_decimal(e: &Ex, expected: &str, rel: f64, label: &str) {
    let got = e.eval_decimal(20).unwrap();
    let split = |s: &str| -> (f64, i64) {
        let (m, x) = s.split_once('e').unwrap_or((s, "0"));
        (m.parse().unwrap(), x.parse().unwrap())
    };
    let ((gm, gx), (em, ex)) = (split(&got), split(expected));
    let norm = |m: f64, x: i64| -> (f64, i64) {
        let shift = m.abs().log10().floor() as i64;
        (m / 10f64.powi(shift as i32), x + shift)
    };
    let ((gm, gx), (em, ex)) = (norm(gm, gx), norm(em, ex));
    assert!(
        gx == ex && (gm - em).abs() <= rel * em.abs(),
        "{label}: got {got}, expected {expected}"
    );
}

// ── 1. Discrete quantiles ──────────────────────────────────────────────────

/// The binomial and Poisson kernels already invert on the smaller tail;
/// the slack subtracted before them made `poisson(10⁶).quantile_f64(10⁻¹³)`
/// 962716 (the level became `2.2·10⁻³⁰⁸`) and `poisson(7/3)` at `1 − 2⁻⁵³`
/// 20 (the level became `1 − 10⁻¹²`).
#[test]
fn lattice_kernels_get_the_level_as_given() {
    let ctx = Context::new();
    // scipy: poisson.ppf(1e-13, 10**6) = 992660.0; mpmath:
    //   gammainc(992660, 10**6, inf, regularized=True) = 9.9593070953051930731e-14 (F(992659)),
    //   gammainc(992661, 10**6, inf, regularized=True) = 1.0034269405045407338e-13 (F(992660)).
    let big = Distribution::poisson(ctx.int(10).powi(6));
    assert_eq!(big.quantile_f64(1e-13).unwrap(), 992_660.0);
    // scipy: poisson.ppf(1 - 2**-53, 7/3) = 23.0, which compares F(k) next to 1; mpmath:
    //   gammainc(24, 0, mpf(7)/3, regularized=True) = 1.1688573927404834375e-16 > 2**-53 (S(23)),
    //   gammainc(25, 0, mpf(7)/3, regularized=True) = 1.0866816836726622497e-17 ≤ 2**-53 (S(24)).
    let small = Distribution::poisson(ctx.rational(7, 3));
    assert_eq!(small.quantile_f64(TOP).unwrap(), 24.0);
}

/// `NegativeBinomial` (no closed CDF) was walked comparing the summed pmf
/// with `p − 10⁻¹²`, `Geometric` bracketed with the same slack: at
/// `1 − 2⁻⁵³` they gave 48, 72 and 90, and below `10⁻¹²` the first atom.
#[test]
fn negative_binomial_and_geometric_decide_on_the_small_tail() {
    let ctx = Context::new();
    // scipy: nbinom.ppf(1 - 2**-53, 3, 0.5) = 62.0; mpmath:
    //   betainc(62, 3, 0, mpf(1)/2, regularized=True) = 1.1281123604711673636e-16 (S(61)),
    //   betainc(63, 3, 0, mpf(1)/2, regularized=True) = 5.8167446553847312884e-17 (S(62)).
    let nb = Distribution::negative_binomial(ctx.int(3), ctx.rational(1, 2));
    assert_eq!(nb.quantile_f64(TOP).unwrap(), 62.0);
    // scipy: nbinom.ppf(1 - 2**-53, 1.5, 1/3) = 95.0; mpmath:
    //   betainc(95, 1.5, 0, mpf(2)/3, regularized=True) = 1.2029584881706111447e-16 (S(94)),
    //   betainc(96, 1.5, 0, mpf(2)/3, regularized=True) = 8.0606525139115573521e-17 (S(95)).
    let nbr = Distribution::negative_binomial(ctx.rational(3, 2), ctx.rational(1, 3));
    assert_eq!(nbr.quantile_f64(TOP).unwrap(), 95.0);
    // scipy: nbinom.ppf(1e-13, 50, 0.1) = 110.0; mpmath:
    //   betainc(50, 110, 0, mpf(1)/10, regularized=True) = 9.03161958777698e-14 (F(109)),
    //   betainc(50, 111, 0, mpf(1)/10, regularized=True) = 1.18564331089814e-13 (F(110)).
    let nb50 = Distribution::negative_binomial(ctx.int(50), ctx.rational(1, 10));
    assert_eq!(nb50.quantile_f64(1e-13).unwrap(), 110.0);
    // scipy: geom.ppf(1 - 2**-53, 1/3) = 90.0; mpmath: (mpf(2)/3)**90 = 1.4183606858897689309e-16
    //   > 2**-53, (mpf(2)/3)**91 = 9.4557379059317928724e-17 ≤ 2**-53, so 91.
    let g = Distribution::geometric(ctx.rational(1, 3));
    assert_eq!(g.quantile_f64(TOP).unwrap(), 91.0);
    // scipy: geom.ppf(1e-13, 1e-15) = 101.0; mpmath: 1 - (1 - mpf(10)**-15)**100
    //   = 9.999999999999505e-14 < 1e-13 ≤ 1 - (1 - mpf(10)**-15)**101 = 1.0099999999999495e-13.
    let tiny = Distribution::geometric(ctx.one() / ctx.int(10).powi(15));
    assert_eq!(tiny.quantile_f64(1e-13).unwrap(), 101.0);
    // Unchanged in the body: scipy nbinom.ppf([0.9, 0.5], 1.5, 1/3) = [7, 2], geom.ppf(0.3, 1/3) = 1.
    assert_eq!(nbr.quantile_f64(0.9).unwrap(), 7.0);
    assert_eq!(nbr.quantile_f64(0.5).unwrap(), 2.0);
    assert_eq!(g.quantile_f64(0.3).unwrap(), 1.0);
}

/// A family defined outside the crate by its mass function alone:
/// `Poisson(λ)` on `0..∞`.
#[derive(Debug, PartialEq)]
struct PmfOnlyPoisson {
    rate: Ex,
}

impl Family for PmfOnlyPoisson {
    fn name(&self) -> &str {
        "PmfOnlyPoisson"
    }
    fn context(&self) -> Context {
        self.rate.context()
    }
    fn support(&self) -> Support {
        let ctx = self.context();
        Support::integers(&ctx, Some(ctx.zero()), None)
    }
    fn density(&self, k: &Ex) -> Ex {
        self.rate.pow(k) * (-&self.rate).exp() / k.factorial()
    }
    fn parameters(&self) -> Vec<(&'static str, Ex)> {
        vec![("rate", self.rate.clone())]
    }
    fn eq_family(&self, other: &dyn Family) -> bool {
        same_family(self, other)
    }
}

/// A family defined outside the crate by its mass function and survival
/// function: `Geometric(p)` on `1..∞`, `S(k) = (1−p)^⌊k⌋`.
#[derive(Debug, PartialEq)]
struct SurvivalGeometric {
    p: Ex,
}

impl Family for SurvivalGeometric {
    fn name(&self) -> &str {
        "SurvivalGeometric"
    }
    fn context(&self) -> Context {
        self.p.context()
    }
    fn support(&self) -> Support {
        let ctx = self.context();
        Support::integers(&ctx, Some(ctx.one()), None)
    }
    fn density(&self, k: &Ex) -> Ex {
        let ctx = self.context();
        (ctx.one() - &self.p).pow(&(k - ctx.one())) * &self.p
    }
    fn sf(&self, k: &Ex) -> Option<Ex> {
        Some((self.context().one() - &self.p).pow(&k.floor()))
    }
    fn parameters(&self) -> Vec<(&'static str, Ex)> {
        vec![("p", self.p.clone())]
    }
    fn eq_family(&self, other: &dyn Family) -> bool {
        same_family(self, other)
    }
}

/// Custom lattice families take the generic route: a mass function alone
/// is walked (the walk compared with `p − 10⁻¹²`, negative at `10⁻¹³`, so
/// it stopped at the first atom), a survival function is searched exactly
/// near `1` (the walk ignored it and stopped where the summed pmf reached
/// `1 − 10⁻¹²`, far short of `1 − 2⁻⁵³`).
#[test]
fn custom_lattice_families_take_the_generic_route() {
    let ctx = Context::new();
    // scipy: poisson.ppf(1e-13, 100) = 36.0; mpmath:
    //   gammainc(36, 100, inf, regularized=True) = 5.4951389784287683542e-14 (F(35)),
    //   gammainc(37, 100, inf, regularized=True) = 1.5495522609294018764e-13 (F(36)).
    let pmf_only = Distribution::from_family(PmfOnlyPoisson { rate: ctx.int(100) });
    assert_eq!(pmf_only.quantile_f64(1e-13).unwrap(), 36.0);
    // The same S(90) > 2⁻⁵³ ≥ S(91) as `Geometric(1/3)` above (mpmath (2/3)**90, (2/3)**91).
    let with_sf = Distribution::from_family(SurvivalGeometric {
        p: ctx.rational(1, 3),
    });
    assert_eq!(with_sf.quantile_f64(TOP).unwrap(), 91.0);
    assert_eq!(with_sf.quantile_f64(0.9).unwrap(), 6.0); // scipy: geom.ppf(0.9, 1/3) = 6.0
}

/// The table walk compared its running sum with `p − 10⁻¹²` from the
/// bottom: a first atom of mass `5·10⁻¹⁴` answered a level of `10⁻¹³`,
/// and `F(0) = 1 − 10⁻¹⁵` answered `1 − 2⁻⁵³`.  Exact ties still count.
#[test]
fn finite_tables_compare_the_small_tail() {
    let ctx = Context::new();
    let half_e13 = ctx.one() / (ctx.int(2) * ctx.int(10).powi(13));
    let lower = Distribution::finite(
        &ctx,
        vec![
            (ctx.int(0), half_e13.clone()),
            (ctx.int(1), ctx.one() - &half_e13),
        ],
    );
    // scipy: rv_discrete(values=([0, 1], [5e-14, 1 - 5e-14])).ppf(1e-13) = 1.0
    assert_eq!(lower.quantile_f64(1e-13).unwrap(), 1.0);
    let e15 = ctx.one() / ctx.int(10).powi(15);
    let upper = Distribution::finite(
        &ctx,
        vec![(ctx.int(0), ctx.one() - &e15), (ctx.int(1), e15.clone())],
    );
    // scipy: rv_discrete(values=([0, 1], [1 - 1e-15, 1e-15])).ppf(1 - 2**-53) = 1.0 (S(0) = 1e-15 > 2**-53)
    assert_eq!(upper.quantile_f64(TOP).unwrap(), 1.0);
    // Ties: F(0) = 1/2 and F(1) = 3/4 exactly (on the upper side S(1) = 1/4 = 1 − 0.75).
    let quarters = Distribution::finite(
        &ctx,
        vec![
            (ctx.int(0), ctx.rational(1, 2)),
            (ctx.int(1), ctx.rational(1, 4)),
            (ctx.int(2), ctx.rational(1, 4)),
        ],
    );
    assert_eq!(quarters.quantile_f64(0.5).unwrap(), 0.0);
    assert_eq!(quarters.quantile_f64(0.75).unwrap(), 1.0);
    assert_eq!(quarters.quantile_f64(0.75 + 1e-16).unwrap(), 2.0);
}

/// Wrapped and mixed lattices go through the same exact search; they were
/// bracketed with the same `10⁻¹²` slack, and the mixture's support (a
/// lattice and a listed half-integer) by a Brent root floored to an
/// integer.
#[test]
fn wrapped_and_mixed_lattices() {
    let ctx = Context::new();
    let at_least_5 = Support::from_pieces(
        Kind::Continuous,
        vec![Piece::Interval(Interval::closed(
            ctx.int(5),
            ctx.infinity(),
        ))],
    );
    let truncated = Distribution::poisson(ctx.int(20))
        .truncated(&at_least_5)
        .unwrap();
    // mpmath: S(k) = gammainc(k+1, 0, 20, reg)/(1 - gammainc(5, 20, inf, reg)):
    //   S(66) = 1.178649253e-16 > 2**-53 ≥ S(67) = 3.446254639e-17.
    assert_eq!(truncated.quantile_f64(TOP).unwrap(), 67.0);
    // mpmath: F(5) = 5.496502797e-5 ≥ 1e-13, and F reaches 1/2 at 20.
    assert_eq!(truncated.quantile_f64(1e-13).unwrap(), 5.0);
    assert_eq!(truncated.quantile_f64(0.5).unwrap(), 20.0);
    let negated = Distribution::poisson(ctx.int(2))
        .affine(ctx.int(-1), ctx.int(0))
        .unwrap();
    // P(−X ≤ y) = P(X ≥ −y); mpmath: gammainc(19, 0, 2, reg) = P(X ≥ 19) = 6.477297338e-13,
    //   gammainc(20, 0, 2, reg) = P(X ≥ 20) = 6.443731393e-14 < 1e-13, so −19.
    assert_eq!(negated.quantile_f64(1e-13).unwrap(), -19.0);
    let mixed = Distribution::mixture(&[
        (ctx.rational(1, 2), Distribution::poisson(ctx.int(3))),
        (
            ctx.rational(1, 2),
            Distribution::finite(&ctx, vec![(ctx.rational(1, 2), ctx.one())]),
        ),
    ])
    .unwrap();
    // F(0) = e⁻³/2 = 0.0249 < 0.3 ≤ F(1/2) = 0.0249 + 1/2.
    assert_eq!(mixed.quantile_f64(0.3).unwrap(), 0.5);
    // mpmath: gammainc(26, 0, 3, reg)/2 = 1.764146695e-16 > 2**-53 ≥ gammainc(27, 0, 3, reg)/2 = 1.951567712e-17.
    assert_eq!(mixed.quantile_f64(TOP).unwrap(), 26.0);
}

/// `quantile_f64` of a lattice family without a closed CDF built the
/// symbolic CDF `Σ pmf` over the whole line before walking, and closing
/// that sum for a hypergeometric took 4.5 s (debug build); the walk itself
/// takes about a millisecond.  `cdf(4)` took the same 4.5 s: the sum ran
/// up to an unevaluated `floor(4)`, and the summation looked for a closed
/// form instead of adding five terms.  Each is timed against evaluating
/// the eleven masses one by one, in interleaved rounds: about 2× that
/// now, about 5000× before; bound 100.
#[test]
fn hypergeometric_quantile_and_cdf_add_the_masses() {
    let ctx = Context::new();
    let h = Distribution::hypergeometric(ctx.int(50), ctx.int(20), ctx.int(10));
    // scipy: hypergeom.ppf([1e-13, 0.5, 1 - 2**-53], 50, 20, 10) = [0, 4, 10]
    assert_eq!(h.quantile_f64(1e-13).unwrap(), 0.0);
    assert_eq!(h.quantile_f64(TOP).unwrap(), 10.0);
    let (mut quantile, mut cdf, mut reference) = (Duration::ZERO, Duration::ZERO, Duration::ZERO);
    for _ in 0..3 {
        let t = Instant::now();
        assert_eq!(h.quantile_f64(0.5).unwrap(), 4.0);
        quantile += t.elapsed();
        let t = Instant::now();
        // Python: sum(Fraction(comb(20, k)*comb(30, 10 - k), comb(50, 10)) for k in range(5))
        //   = 13522236/20963833 (scipy: hypergeom.cdf(4, 50, 20, 10) = 0.645026889882208)
        assert_eq!(h.cdf(&ctx.int(4)), ctx.rational(13_522_236, 20_963_833));
        cdf += t.elapsed();
        let t = Instant::now();
        let total: f64 = (0..=10)
            .map(|k| h.density(&ctx.int(k)).eval_f64().unwrap())
            .sum();
        reference += t.elapsed();
        close(total, 1.0, 1e-14, "Σ pmf");
    }
    assert!(
        quantile < 100 * reference && cdf < 100 * reference,
        "quantile {quantile:?}, cdf(4) {cdf:?} vs the eleven masses {reference:?}"
    );
}

// ── 2. Tail forms ──────────────────────────────────────────────────────────

/// The exponential and Weibull lower tails were `γ(1, y)`, which `eval`
/// folds into `1 − e^{−y}` at an irrational `y`; Pareto and Geometric had
/// no lower-tail form at all.  At these points they evaluated to `0`
/// (`Pareto(√2, 3)` to "precision exhausted"); now `2 e^{−y/2} sinh(y/2)`
/// (with `y = α ln(x/x_m)` through `atanh` for Pareto, `y = −k ln(1 − p)`
/// for Geometric).
#[test]
fn lower_tails_that_do_not_rely_on_the_eval_fold() {
    let ctx = Context::new();
    let s2 = ctx.int(2).sqrt();
    let tiny = |n: i64| ctx.one() / ctx.int(10).powi(n);
    // mpmath: -expm1(-sqrt(2)*mpf(10)**-100) = 1.4142135623730950488e-100
    let e = Distribution::exponential(s2.clone());
    close(
        e.cdf(&tiny(100)).eval_f64().unwrap(),
        1.414_213_562_373_095_048_8e-100,
        1e-14,
        "Exponential(√2).cdf(1e-100)",
    );
    // mpmath: -expm1(-sqrt(2*mpf(10)**-200)) = 1.4142135623730950488e-100
    let w = Distribution::weibull(ctx.one(), ctx.rational(1, 2));
    close(
        w.cdf(&(ctx.int(2) * tiny(200))).eval_f64().unwrap(),
        1.414_213_562_373_095_048_8e-100,
        1e-14,
        "Weibull(1, 1/2).cdf(2e-200)",
    );
    // mpmath: -expm1(-sqrt(2)*log1p(mpf(10)**-100)) = 1.4142135623730950488e-100
    let pa = Distribution::pareto(ctx.one(), s2.clone());
    close(
        pa.cdf(&(ctx.one() + tiny(100))).eval_f64().unwrap(),
        1.414_213_562_373_095_048_8e-100,
        1e-14,
        "Pareto(1, √2).cdf(1 + 1e-100)",
    );
    // mpmath: -expm1(-3*log1p(mpf(10)**-100/sqrt(2))) = 2.1213203435596425732e-100
    // (the classic form did not evaluate: "precision exhausted")
    let pa3 = Distribution::pareto(s2.clone(), ctx.int(3));
    close(
        pa3.cdf(&(&s2 + tiny(100))).eval_f64().unwrap(),
        2.121_320_343_559_642_573_2e-100,
        1e-14,
        "Pareto(√2, 3).cdf(√2 + 1e-100)",
    );
    // The body keeps the classic form: Pareto(1, 3).cdf(2) = 7/8.
    assert_eq!(
        Distribution::pareto(ctx.one(), ctx.int(3)).cdf(&ctx.int(2)),
        ctx.rational(7, 8)
    );
    // mpmath: -expm1(5*log1p(-pi*mpf(10)**-100)) = 1.5707963267948966192e-99
    let g = Distribution::geometric(ctx.pi() * tiny(100));
    close(
        g.cdf(&ctx.int(5)).eval_f64().unwrap(),
        1.570_796_326_794_896_619_2e-99,
        1e-14,
        "Geometric(π·1e-100).cdf(5)",
    );
}

/// Cauchy decided `x > x₀` symbolically only (`10¹⁰⁰ − √2 > 0` is not
/// decided), and took the classic `½ − atan(z)/π`; Triangular and Binomial
/// had no survival function (`1 − F`); a Triangular with the mode at its
/// lower end had no lower-tail form, and `Distribution::cdf` left an
/// undecided `Piecewise` there (`3 − (√2 + 10⁻¹⁰⁰) > 0`) that `evalf`
/// evaluated to its first branch.  The Cauchy, the first Triangular and
/// the Binomial value were `0`.
#[test]
fn cauchy_triangular_and_binomial_tails() {
    let ctx = Context::new();
    let s2 = ctx.int(2).sqrt();
    let tiny = |n: i64| ctx.one() / ctx.int(10).powi(n);
    // mpmath: atan(pi/(mpf(10)**100 - sqrt(2)))/pi = 1.0e-100
    let c = Distribution::cauchy(s2.clone(), ctx.pi());
    close(
        c.sf(&ctx.int(10).powi(100)).eval_f64().unwrap(),
        1e-100,
        1e-14,
        "Cauchy(√2, π).sf(1e100)",
    );
    // mpmath: (mpf(10)**-30)**2/(sqrt(2)*(sqrt(2) - 1)) = 1.7071067811865475244e-60
    let t = Distribution::triangular(ctx.int(0), s2.clone(), ctx.one());
    close(
        t.sf(&(&s2 - tiny(30))).eval_f64().unwrap(),
        1.707_106_781_186_547_524_4e-60,
        1e-14,
        "Triangular(0, √2, 1).sf(√2 − 1e-30)",
    );
    // mpmath: t*(6 - sqrt(2) - (sqrt(2) + t))/(3 - sqrt(2))**2, t = mpf(10)**-100
    //   = 1.2612038749637414425e-100
    let t_lo = Distribution::triangular(s2.clone(), ctx.int(3), s2.clone());
    close(
        t_lo.cdf(&(&s2 + tiny(100))).eval_f64().unwrap(),
        1.261_203_874_963_741_442_5e-100,
        1e-14,
        "Triangular(√2, 3, √2).cdf(√2 + 1e-100)",
    );
    // mpmath: t*(2*sqrt(2) - t)/sqrt(2)**2, t = mpf(10)**-100 = 1.4142135623730950488e-100
    let t_hi = Distribution::triangular(ctx.int(0), s2.clone(), s2.clone());
    close(
        t_hi.sf(&(&s2 - tiny(100))).eval_f64().unwrap(),
        1.414_213_562_373_095_048_8e-100,
        1e-14,
        "Triangular(0, √2, √2).sf(√2 − 1e-100)",
    );
    // mpmath: fsum(binomial(1000, j)*(1/e)**j*(1 - 1/e)**(1000 - j) for j in 991..1000)
    //   = 1.7710195112531803248e-411
    let b = Distribution::binomial(ctx.int(1000), ctx.one() / ctx.e());
    close_decimal(
        &b.sf(&ctx.int(990)),
        "1.7710195112531803248e-411",
        1e-15,
        "Binomial(1000, 1/e).sf(990)",
    );
    // A rational p stays exact: P(3) + P(4) + P(5) = 17/81 (scipy: binom.sf(2, 5, 1/3) = 0.20987654320987653).
    let b5 = Distribution::binomial(ctx.int(5), ctx.rational(1, 3));
    assert_eq!(b5.sf(&ctx.int(2)), ctx.rational(17, 81));
}

// ── 3. Sample size at a tiny effect ────────────────────────────────────────

/// 0.27 made `power_t_test_two_sample` accurate at large `n` (its χ²
/// density cancelled terms of size `ν ln ν`), which moved this sample
/// size; the new value is the smallest `n` with power ≥ 0.8.
#[test]
fn sample_size_t_test_at_a_tiny_effect() {
    // statsmodels: TTestIndPower().solve_power(effect_size=1e-4, alpha=0.05, power=0.8)
    //   = 1569772102.8256037 → 1569772103.
    // mpmath (dps 40, the two-sided power as the χ²_ν mixture of normal tails, t_c from the same
    //   mixture by findroot): power(1569772102) − 0.8 = −2.0625e-10,
    //   power(1569772103) − 0.8 = +4.3568e-11.
    let n = sample_size_t_test_two_sample(1e-4, 0.05, 0.8).unwrap();
    assert_eq!(n, 1_569_772_103);
    assert!(power_t_test_two_sample(1e-4, n - 1, 0.05).unwrap() < 0.8);
    assert!(power_t_test_two_sample(1e-4, n, 0.05).unwrap() >= 0.8);
}

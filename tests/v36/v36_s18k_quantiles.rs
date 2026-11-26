//! Tests of session 18, second block of 0.60.1. Lane DQ: quantiles of discrete laws, of
//! mixtures and of truncated laws at extreme parameters and levels.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use std::time::Duration;

use symplex::prelude::*;
use symplex::stats::{Distribution, Support};

use crate::perf_assert::CpuTimer;

/// `|a/b − 1|`.
fn rel(a: f64, b: f64) -> f64 {
    (a / b - 1.0).abs()
}

/// `w·a + (1 − w)·b`.
fn mixture(w: Ex, a: Distribution, b: Distribution) -> Distribution {
    let rest = w.context().one() - &w;
    Distribution::mixture(&[(w, a), (rest, b)]).unwrap()
}

// ── Class G: the quantile of a mixture of discrete laws ─────────────────────

/// `½ Hypergeometric(10, 6, 4) + ½ NegativeBinomial(36, ½)` (the reproducer of class G: two
/// saved `fuzz_stats` inputs, 25 s and 76 s).  Up to 0.60.1 every quantile first built the
/// mixture's CDF at a symbol to ask whether it has a closed form — a symbolic summation
/// (Gosper) for the component without one: `quantile_f64(0.3)` took 0.95 s,
/// `quantile_f64(1e-300)` 0.95 s and `isf_f64(1e-300)` 1.3 s in a release build, either
/// component alone under 0.02 s.  Now that question is not asked of a mixture, and each probe
/// is the exact weighted tails of the components: 2 ms, 2 ms and 15 ms.
///
/// Oracle (exact fractions, `oracle/o_dq_repro.py`: the masses `C(6, j) C(4, 4−j)/210` and
/// `P(NB > k) = Σ_{j<36} C(k+36, j)/2^{k+36}`): `F(2) = 0.27380952… < 0.3 ≤ F(3) =
/// 0.46428572…`; `F(0) ≥ 10⁻³⁰⁰`; `S(1184) > q ≥ S(1185)` for `q` the double `1e-300`.
#[test]
fn the_quantile_of_a_mixture_of_discrete_laws_asks_its_components() {
    let ctx = Context::new();
    let half = ctx.rational(1, 2);
    let d = mixture(
        half.clone(),
        Distribution::hypergeometric(ctx.int(10), ctx.int(6), ctx.int(4)),
        Distribution::negative_binomial(ctx.int(36), half),
    );
    let timer = CpuTimer::start();
    assert_eq!(d.quantile_f64(0.3).unwrap(), 3.0);
    assert_eq!(d.quantile_f64(1e-300).unwrap(), 0.0);
    assert_eq!(d.isf_f64(1e-300).unwrap(), 1185.0);
    timer.assert_within(
        Duration::from_secs(10),
        "three quantiles of a mixture of two discrete laws",
    );
}

/// A mixture whose components have different quantiles: `1/10 Hypergeometric(10, 6, 4) +
/// 9/10 Hypergeometric(20, 6, 4)`.  (Up to 0.60.1: 0.3 s per call for the symbolic sums.)
///
/// Oracle (exact fractions, Python: `Fraction(1,10)*F(10,6,4,k) + Fraction(9,10)*F(20,6,4,k)`
/// with `F` the sum of `comb(m,j)*comb(N-m,n-j)/comb(N,n)`): `F(0..=4)` = 0.18642046,
/// 0.60354563, 0.89996314, 0.99007077, 1.
#[test]
fn the_quantile_of_a_mixture_lies_between_those_of_its_components() {
    let ctx = Context::new();
    let hyp = |n: i64| Distribution::hypergeometric(ctx.int(n), ctx.int(6), ctx.int(4));
    let d = mixture(ctx.rational(1, 10), hyp(10), hyp(20));
    assert_eq!(d.quantile_f64(0.1).unwrap(), 0.0);
    assert_eq!(d.quantile_f64(0.5).unwrap(), 1.0);
    assert_eq!(d.quantile_f64(0.99).unwrap(), 3.0);
    // S(k) ≤ 0.4 from k = 1 on (S(0) = 0.81357954, S(1) = 0.39645437).
    assert_eq!(d.isf_f64(0.4).unwrap(), 1.0);
}

// ── Class F: a large hypergeometric law ─────────────────────────────────────

/// `Hypergeometric(90000, 36562, 78750)` (five saved inputs, 5 to 13 s each).  Up to 0.60.1
/// each probe of the search formed both tails at its point as reduced rationals of 49,000
/// bits, and the search started at the mean after 64 exact masses (the compiled mass function
/// overflows): `quantile_f64(0.3)` took 1.2 s and `isf_f64(1e-300)` 8.5 s in a release build.
/// Now one pass over the integer terms `C(m, j) C(N−m, n−j)` decides against `⌈p·C(N, n)⌉`:
/// 0.10 s and 0.12 s.
///
/// Oracle: SciPy 1.18.1 `hypergeom.ppf(0.3, 90000, 36562, 78750)` = 31966; exact Python
/// integers (`oracle/o_dq.py`: the smallest `k` with `Σ_{j≤k} T_j ≥ (1 − q)·C(N, n)`, `q` the
/// double `1e-300` as a fraction) = 33749 — and all 32 quantiles of the benchmark for this law.
#[test]
fn quantiles_of_a_large_hypergeometric_law_are_decided_in_integers() {
    let ctx = Context::new();
    let d = Distribution::hypergeometric(ctx.int(90000), ctx.int(36562), ctx.int(78750));
    let timer = CpuTimer::start();
    assert_eq!(d.quantile_f64(0.3).unwrap(), 31966.0);
    assert_eq!(d.isf_f64(1e-300).unwrap(), 33749.0);
    timer.assert_within(
        Duration::from_secs(120),
        "two quantiles of Hypergeometric(90000, 36562, 78750)",
    );
}

/// `Hypergeometric(60000, 46875, 10312)` truncated to `[12, 6573]`.  Up to 0.60.1 every
/// quantile of it was the refusal `FreeSymbol { name: "_t" }` (after 0.4 s): the hypergeometric
/// tails are closed forms at numbers only, the truncated law therefore had "no closed form",
/// its tail became the sum of the truncated masses over an internal index, and a sum too long
/// to expand reached the evaluator with that index free.  Now the truncated tails are formed
/// from the inner exact tails at a numeric point, and the quantile is decided in integers
/// over the window (0.07 s).
///
/// Oracle: exact Python integers (`oracle/o_dq.py`: the smallest `k ≥ 12` with
/// `Σ_{12≤j≤k} T_j ≥ p·Σ_{12≤j≤6573} T_j`) = 5927 at `p` the double `1e-300`, 6573 at 0.5
/// (the truncation point: the law's mass lies above it), and S(k) ≤ 0.9 from 6571 on.
#[test]
fn a_truncated_hypergeometric_law_has_quantiles() {
    let ctx = Context::new();
    let d = Distribution::hypergeometric(ctx.int(60000), ctx.int(46875), ctx.int(10312))
        .truncated(&Support::interval(ctx.int(12), ctx.int(6573)))
        .unwrap();
    let timer = CpuTimer::start();
    assert_eq!(d.quantile_f64(1e-300).unwrap(), 5927.0);
    assert_eq!(d.quantile_f64(0.5).unwrap(), 6573.0);
    assert_eq!(d.isf_f64(0.9).unwrap(), 6571.0);
    timer.assert_within(
        Duration::from_secs(120),
        "three quantiles of a truncated Hypergeometric(60000, 46875, 10312)",
    );
    // The tails behind them are numbers now, not sums over an internal index: 6000 lies
    // between the quantiles at 1e-300 (5927) and at 1e-100 (6333, same oracle).
    let f = d.cdf(&ctx.int(6000)).eval_f64().unwrap();
    assert!(f > 1e-300 && f < 1e-100, "P(X ≤ 6000) = {f}");
}

// ── Class K and the search side of class E ──────────────────────────────────

/// `Geometric(1/(25·10²⁵⁰))`.  Up to 0.60.1 the search started at the mean `2.5·10²⁵¹` and
/// stepped by 1, which does not move a double of that size: the same point was probed once
/// per doubling of the stride until the stride reached the spacing of the doubles —
/// `quantile_f64(1e-4)` took 0.78 s and `isf_f64(1e-4)` 1.26 s in a release build.  Now the
/// search starts at the closed form `⌈ln(1 − p)/ln(1 − π)⌉` and its first stride is the
/// spacing of the doubles there: 2 ms and 4 ms.  The values are unchanged.
///
/// Oracle (mpmath at 600 and at 1200 digits, `oracle/o_dq_repro.py`): `k* = ceil(log(1 −
/// mpf(1e-4))/log(1 − π))` and `ceil(log(mpf(1e-4))/log(1 − π))`; each result is the smallest
/// double `≥ k*` (`k*/result − 1` = −3.3·10⁻¹⁷ and −5.7·10⁻¹⁷, and the double below is `< k*`).
#[test]
fn quantiles_of_a_geometric_law_with_a_tiny_success_probability() {
    let ctx = Context::new();
    let d = Distribution::geometric(ctx.parse("1/(25*10^250)").unwrap());
    let timer = CpuTimer::start();
    assert_eq!(d.quantile_f64(1e-4).unwrap(), 2.500_125_008_333_958_6e247);
    assert_eq!(d.isf_f64(1e-4).unwrap(), 2.302_585_092_994_046e252);
    timer.assert_within(
        Duration::from_secs(20),
        "two quantiles of Geometric(1/(25e250))",
    );
}

/// `Poisson(3974955008)` truncated to `[0, 3974720536]` (3.7 standard deviations below the
/// mean).  Up to 0.60.1 `quantile_f64` refused at every level whose answer lies below the
/// truncation point — "factorial: the value (about 2^1.210e11) overflows …", after 0.7 s at
/// `1e-300`: the truncated tail was built from the Poisson closed form `Γ(k + 1, λ)/k!` as it
/// is, and the evaluator formed `k!`.  Now a tail that the evaluator refuses is folded first
/// (as the whole-line `cdf` folds every numeric value), and the search starts at the inner
/// law's quantile at the corresponding level: 7 ms.
///
/// Oracle (mpmath at 40 and at 70 digits, `oracle/o_dq_repro.py`: `F(k) = pmf(k)·Σ_i
/// Π_{t<i} (k − t)/λ`, `pmf` through `loggamma`): with `K = 3972603912` and `b = 3974720536`,
/// `F(K)/F(b)/1e-300 − 1 = +5.6323641·10⁻⁴` and `F(K − 1)/F(b)/1e-300 − 1 = −2.8998828·10⁻⁵`.
#[test]
fn a_truncated_poisson_law_of_a_large_rate_has_quantiles() {
    let ctx = Context::new();
    let d = Distribution::poisson(ctx.parse("3974955008").unwrap())
        .truncated(&Support::interval(
            ctx.int(0),
            ctx.parse("3974720536").unwrap(),
        ))
        .unwrap();
    let timer = CpuTimer::start();
    assert_eq!(d.quantile_f64(1e-300).unwrap(), 3_972_603_912.0);
    timer.assert_within(
        Duration::from_secs(20),
        "a far quantile of a truncated Poisson(3974955008)",
    );
}

/// `NegativeBinomial(21590, 1/(1.25·10¹⁸¹))` and `NegativeBinomial(1000, 10⁻¹⁰⁰)`.  Up to
/// 0.60.1 the search walked 100,000 atoms of the compiled mass function, then started at the
/// mean with a stride of 1, which probes the same double until the stride has doubled up to
/// the spacing of the doubles there (`2⁵⁶²` at `2·10¹⁸⁵`; the probes were not counted), each
/// probe an incomplete beta function — 11.7 s (`quantile_f64(1e-300)`), 18.1 s (`isf_f64(1e-300)`) and 2.0 s
/// (`quantile_f64(0.3)` of the second law) in a release build.  Now the search starts at the
/// quantile of the limit law `Gamma(r, (1 − π)/π)` with the spacing of the doubles as its
/// first stride: 0.06 s, 0.03 s and 0.01 s.  The values are unchanged.
///
/// Oracle (mpmath at 40 and at 80 digits, `oracle/o_dq_repro.py`): `π·X → Gamma(r, 1)` with a
/// relative error of order `π`; `findroot` of `gammainc(r, 0, x, regularized=True) = level`
/// (or of the upper function), divided by `π`: 2.0742177585715660539e185,
/// 3.4374920102996475614e185, 9.8317844675789870858e102 — the results exceed them by 3.3, 3.1
/// and 4.4·10⁻¹⁷ (relative).  SciPy 1.18.1 `nbinom.ppf(0.3, 1000, 1e-100)` is
/// 9.831784467578986e102, two doubles below the limit.
#[test]
fn quantiles_of_a_negative_binomial_law_with_a_tiny_success_probability() {
    let ctx = Context::new();
    let d = Distribution::negative_binomial(ctx.int(21590), ctx.parse("1/(125*10^179)").unwrap());
    let timer = CpuTimer::start();
    assert!(rel(d.quantile_f64(1e-300).unwrap(), 2.074_217_758_571_566e185) < 4e-16);
    assert!(rel(d.isf_f64(1e-300).unwrap(), 3.437_492_010_299_647_7e185) < 4e-16);
    let d = Distribution::negative_binomial(ctx.int(1000), ctx.parse("1/10^100").unwrap());
    assert!(rel(d.quantile_f64(0.3).unwrap(), 9.831_784_467_578_988e102) < 4e-16);
    timer.assert_within(
        Duration::from_secs(60),
        "three quantiles of negative binomial laws with a tiny p",
    );
}

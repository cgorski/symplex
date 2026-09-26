//! Audit round 3b-ii of `symplex::stats`: `sequential`, `markov`,
//! `multivariate`.
//!
//! 1. Wald's OC and ASN approximations for the Bernoulli SPRT
//!    (`operating_characteristic_bernoulli`,
//!    `expected_sample_size_bernoulli`) decided "drift = 0" by an
//!    absolute `|E_p[Z]| < 1e-13` in log-likelihood units — so for close
//!    hypotheses `L = 0.562` for a true `0.00117` —, overflowed to `NaN` or
//!    failed to bracket Wald's root for small `p`, and cancelled near the
//!    `p` where the drift vanishes (`E_p[N] = −2.4·10⁷` for a true `23.57`).
//! 2. The per-observation increments were logarithms of rounded ratios:
//!    `ln(1 + 2·10⁻¹²)` was `2.2e-5` off, and `p₁ = 1 − 10⁻⁴⁰⁰` made the
//!    failure increment `−∞`, so a failure-free run had `Λ = 0·(−∞) = NaN`
//!    and the test never stopped.
//! 3. `MarkovChain::limiting_distribution` was `None` for every reducible
//!    chain, though `Pⁿ` converges to one row when there is a single
//!    aperiodic closed class; `sample_path` turned entries with huge
//!    numerators and denominators into `inf/inf = NaN` (the chain froze)
//!    and panicked for `steps = usize::MAX`; duplicate labels were
//!    accepted.
//! 4. `MultivariateNormal::new` (unchecked) with an indefinite or
//!    asymmetric covariance gave an imaginary or meaningless density;
//!    `sample(usize::MAX)` panicked; `correlation_matrix` gave `[[1]]` for a
//!    constant variable; the exact `pca` ordered eigenvalues by their `f64`
//!    values (`1 − 10⁻²⁰` before `1`); `pca_f64` returned the unrotated
//!    diagonal for a matrix of order `10⁻¹⁷⁰` or `10¹⁷⁰`, let a `NaN` entry
//!    through, and gave `NaN` ratios for the zero matrix.
//!
//! Reference values: mpmath 1.3 (`target/scratch/oracle_3b2.py` of this
//! session, output in `oracle_3b2.out`: Wald's formulas with `h` found by
//! bisection of Wald's identity at 60 and 90 digits, independently with a
//! doubling and an analytic bracket, and relative condition numbers by
//! `10⁻³⁰` perturbations of `p`, `ln(p₁/p₀)` and `ln((1−p₁)/(1−p₀))`);
//! SymPy 1.14 exact matrix powers, nullspaces and eigenvalues; numpy
//! `linalg.eigh` / `corrcoef`; simulation `target/scratch/sprt_sim.py`
//! (numpy, seed 20260926, 200 000 runs) for the documented accuracy of
//! Wald's approximations.  Differential sweeps (1 500 SPRT designs, 1 100
//! Markov chains, 350 MVN parameter sets, 350 `pca_f64` matrices against
//! mpmath / `fractions` oracles) were run from `examples/` probes during
//! the session and are not part of the suite.

// Reference values are quoted at the digits mpmath printed them.
#![allow(clippy::excessive_precision)]

use num_bigint::BigInt;
use symplex::linprog::{Q, q, qi};
use symplex::prelude::*;
use symplex::stats::Rng;
use symplex::stats::markov::MarkovChain;
use symplex::stats::multivariate::{MultivariateNormal, correlation_matrix, pca, pca_f64};
use symplex::stats::sequential::{
    Decision, Sprt, expected_sample_size_bernoulli, operating_characteristic_bernoulli,
};

/// `actual` within `rel` of `expected`, relatively.
fn close(actual: f64, expected: f64, rel: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= rel * expected.abs(),
        "{label}: got {actual:e}, expected {expected:e} (rel {:.1e})",
        (actual / expected - 1.0).abs()
    );
}

fn is_invalid<T: std::fmt::Debug>(r: Result<T, SymplexError>) {
    match r {
        Err(SymplexError::InvalidArgument { .. }) => {}
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
}

/// `n / 10ᵏ` exactly.
fn over_pow10(n: BigInt, k: u32) -> Q {
    Q::new(n, BigInt::from(10).pow(k))
}

const EPS: f64 = f64::EPSILON;

// ═══════════════════════════════════════════════════════════════════════════
// 1. Wald's approximations
// ═══════════════════════════════════════════════════════════════════════════

/// For `p₀ = 1/2`, `p₁ = 1/2 + 10⁻⁷` the drift `E_p[Z]` is of order `10⁻¹³`
/// for every `p` near `1/2`, and the old absolute cut-off `|E_p[Z]| < 1e-13`
/// replaced `L(p)` by its `h = 0` limit `B/(B − A) = 0.5621`: at
/// `p = 0.5000002` it returned `L = 0.5621` (true `0.001166`) and
/// `E[N] = 1.63e14` (true `4.81e13`).  The functions are intrinsically
/// ill-conditioned here — they vary on a `p`-scale of `10⁻⁷` — so the
/// tolerance is `16ε` times the relative condition number mpmath measured.
#[test]
fn wald_approximations_for_close_hypotheses_had_an_absolute_cutoff() {
    let (p0, p1) = (q(1, 2), q(5_000_001, 10_000_000));
    // oracle_3b2.py: L = 0.0011661510213548326865 (condL 4.5e7),
    // ASN = 48072930026294.409332 (condN 6.57e6).
    let l = operating_characteristic_bernoulli(0.5000002, &p0, &p1, 0.05, 0.10).unwrap();
    close(
        l,
        0.0011661510213548326865,
        16.0 * EPS * 4.5e7,
        "L(0.5000002)",
    );
    let n = expected_sample_size_bernoulli(0.5000002, &p0, &p1, 0.05, 0.10).unwrap();
    close(
        n,
        48072930026294.409332,
        16.0 * EPS * 6.57e6,
        "E[N](0.5000002)",
    );
    // oracle_3b2.py: L = 0.91594517710915158838 (condL 4.64e6),
    // ASN = 113694386330340.15933 (condN 1.3e7); the old L was 0.5621 again.
    let l = operating_characteristic_bernoulli(0.50000001, &p0, &p1, 0.05, 0.10).unwrap();
    close(
        l,
        0.91594517710915158838,
        16.0 * EPS * 4.64e6,
        "L(0.50000001)",
    );
    let n = expected_sample_size_bernoulli(0.50000001, &p0, &p1, 0.05, 0.10).unwrap();
    close(
        n,
        113694386330340.15933,
        16.0 * EPS * 1.3e7,
        "E[N](0.50000001)",
    );
}

/// The root of Wald's identity grows like `ln(1/p)/ln(p₁/p₀)`: the old
/// doubling search from `h = 1` overflowed `e^{h·ln(p₁/p₀)}` before
/// bracketing it (`p = 10⁻³⁰⁰`: "could not bracket"), and
/// `(e^{Bh} − 1)/(e^{Bh} − e^{Ah})` was `inf/inf = NaN` once `Bh > 709`
/// (`p = 10⁻³` with `p₁ − p₀ = 10⁻⁴`: `L = E[N] = NaN`).  Both are now
/// well-conditioned (condition number `≤ 1`).
#[test]
fn wald_approximations_for_tiny_p_no_longer_overflow() {
    // oracle_3b2.py: L = 1.0, ASN = 2.0492141056749178546 (h = 2748.65).
    let (p0, p1) = (q(7, 10), q(9, 10));
    assert_eq!(
        operating_characteristic_bernoulli(1e-300, &p0, &p1, 0.05, 0.10).unwrap(),
        1.0
    );
    let n = expected_sample_size_bernoulli(1e-300, &p0, &p1, 0.05, 0.10).unwrap();
    close(n, 2.0492141056749178546, 1e-14, "E[N](1e-300)");
    // oracle_3b2.py: L = 1.0, ASN = 11277.886827911867911 (h = 34537.2).
    let (p0, p1) = (q(1, 2), q(5001, 10000));
    assert_eq!(
        operating_characteristic_bernoulli(1e-3, &p0, &p1, 0.05, 0.10).unwrap(),
        1.0
    );
    let n = expected_sample_size_bernoulli(1e-3, &p0, &p1, 0.05, 0.10).unwrap();
    close(n, 11277.886827911867911, 1e-14, "E[N](1e-3)");
}

/// Near `p* = 0.8138310582896…`, where `E_p[Z] = 0`, the old
/// `(L·A + (1 − L)·B)/E_p[Z]` divided two vanishing differences:
/// `E[N] = 20202.05` at `p = 0.8138310583` and `−24160910.9` at
/// `p = 0.81383105829` (true `23.568048271…` both), and `L` was `4e-6`
/// off.  Both functions are smooth there (condition numbers 12.3 and
/// 3.97), so the tolerance is `1e-14`.
#[test]
fn wald_expected_sample_size_near_zero_drift_does_not_cancel() {
    let (p0, p1) = (q(7, 10), q(9, 10));
    // oracle_3b2.py: L = 0.56214719732683278376, ASN = 23.568048271027389391.
    let l = operating_characteristic_bernoulli(0.81383105829, &p0, &p1, 0.05, 0.10).unwrap();
    close(l, 0.56214719732683278376, 1e-14, "L near p*");
    let n = expected_sample_size_bernoulli(0.81383105829, &p0, &p1, 0.05, 0.10).unwrap();
    close(n, 23.568048271027389391, 1e-14, "E[N] near p*");
    // oracle_3b2.py: ASN = 23.568048271598315538.
    let n = expected_sample_size_bernoulli(0.8138310583, &p0, &p1, 0.05, 0.10).unwrap();
    close(n, 23.568048271598315538, 1e-14, "E[N] nearer p*");
}

/// `ln(p₁/p₀)` was the logarithm of the rounded quotient: for
/// `p₁ = 1/2 + 10⁻¹²` the success increment came out as
/// `1.999955756557757e-12` (true `1.999999999998e-12`, a `2.2e-5` error),
/// and for `p₁ = 1/2 + 5·10⁻¹⁰` Wald's `E[N]` at `p = 0.6` was
/// `14451854420.90` (true `14451858825.61`, `3e-7` off, though its
/// condition number is `11`).
#[test]
fn wald_increments_are_logarithms_of_exact_ratios() {
    let (p0, p1) = (
        q(1, 2),
        q(1, 2) + Q::new(BigInt::from(1), BigInt::from(10).pow(12)),
    );
    let mut test = Sprt::bernoulli(&p0, &p1, 0.05, 0.10).unwrap();
    test.update(true);
    // mpmath: log((1/2 + 1e-12)/(1/2)) = 1.999999999998e-12
    close(
        test.log_likelihood_ratio_f64(),
        1.999999999998e-12,
        1e-15,
        "ln(p1/p0)",
    );
    test.reset();
    test.update(false);
    // mpmath: log((1/2 - 1e-12)/(1/2)) = -2.000000000002e-12
    close(
        test.log_likelihood_ratio_f64(),
        -2.000000000002e-12,
        1e-15,
        "ln((1-p1)/(1-p0))",
    );
    // oracle_3b2.py: ASN = 14451858825.610473421 (condN 11.0).
    let n =
        expected_sample_size_bernoulli(0.6, &q(1, 2), &q(1_000_000_001, 2_000_000_000), 0.05, 0.10)
            .unwrap();
    close(n, 14451858825.610473421, 1e-14, "E[N] with p1 - p0 = 5e-10");
}

/// With `p₁ = 1 − 10⁻⁴⁰⁰` the failure ratio `10⁻⁴⁰⁰/(1/2)` rounded to `0`,
/// its logarithm was `−∞`, and `Λ = s·ln 2 + 0·(−∞)` was `NaN` after any
/// failure-free run: 50 successes left the test at `Continue` for ever.
/// Symmetrically for `p₀ = 10⁻⁴⁰⁰` and failures.  Wald's OC for that
/// design was "could not bracket".
#[test]
fn sprt_decides_when_an_increment_underflows() {
    let p1 = over_pow10(BigInt::from(10).pow(400) - 1, 400);
    let mut test = Sprt::bernoulli(&q(1, 2), &p1, 0.05, 0.10).unwrap();
    let mut decision = Decision::Continue;
    for _ in 0..50 {
        let before = test.is_decided();
        decision = test.update(true);
        // The open continuation region is exactly "continue" (a NaN ratio
        // was outside it while the test said Continue).
        if !before {
            let inside = test.boundaries().contains(&test.log_likelihood_ratio_f64());
            assert_eq!(inside, decision == Decision::Continue);
        }
    }
    assert_eq!(decision, Decision::AcceptH1);
    assert_eq!(test.stopped_at(), Some(5));
    // mpmath (900 dps): 50·log(2(1 − 10⁻⁴⁰⁰)) = 34.657359027997265471
    close(
        test.log_likelihood_ratio_f64(),
        34.657359027997265471,
        1e-15,
        "Λ after 50 successes",
    );
    test.reset();
    assert_eq!(test.update(false), Decision::AcceptH0);
    // mpmath (900 dps): log(10⁻⁴⁰⁰/(1/2)) = −920.3408900170583283
    close(
        test.log_likelihood_ratio_f64(),
        -920.3408900170583283,
        1e-15,
        "Λ after a failure",
    );
    // oracle_3b2.py (900 dps): L(0.9) = 0.65561433688628503775, E[N] = 0.0052573595500916935247
    let l = operating_characteristic_bernoulli(0.9, &q(1, 2), &p1, 0.05, 0.10).unwrap();
    close(l, 0.65561433688628503775, 1e-14, "L(0.9)");
    let n = expected_sample_size_bernoulli(0.9, &q(1, 2), &p1, 0.05, 0.10).unwrap();
    close(n, 0.0052573595500916935247, 1e-13, "E[N](0.9)");
    let p0 = over_pow10(BigInt::from(1), 400);
    let mut test = Sprt::bernoulli(&p0, &q(1, 2), 0.05, 0.10).unwrap();
    for _ in 0..50 {
        decision = test.update(false);
    }
    assert_eq!(decision, Decision::AcceptH0);
    // mpmath (900 dps): 50·log((1/2)/(1 − 10⁻⁴⁰⁰)) = −34.657359027997265471
    close(
        test.log_likelihood_ratio_f64(),
        -34.657359027997265471,
        1e-15,
        "Λ after 50 failures",
    );
}

/// `wald_boundaries` divided the rounded `1 − α` and took `ln` of a
/// quotient near `1`: for `α = 0.3`, `β = 0.7 − 2⁻⁵³` (a legal design,
/// `α + β = 0.9999999999999998` in `f64`) the boundaries were
/// `(−1.11e-16, 6.66e-16)`, 53 % and 20 % off.  They are now formed from
/// the exact binary values of `α` and `β` (hugely ill-conditioned in them,
/// but those are exact inputs), so to a few ulps.  (Minor: only designs
/// with `α + β` within about `10⁻⁸` of `1` lost more than `10⁻⁸`.)
#[test]
fn wald_boundaries_keep_their_accuracy_as_alpha_plus_beta_nears_one() {
    use symplex::stats::sequential::wald_boundaries;
    let b = wald_boundaries(0.3, 0.7 - 2f64.powi(-53)).unwrap();
    // mpmath (40 dps) on the exact binary values: log(b/(1−a)), log((1−b)/a)
    close(b.lower, -2.3790493384824785462e-16, 4.0 * EPS, "A");
    close(b.upper, 5.5511151231257813668e-16, 4.0 * EPS, "B");
    // A sum of 1 is still refused, including the decimal 0.3 + 0.7.
    is_invalid(wald_boundaries(0.25, 0.75));
    is_invalid(wald_boundaries(0.3, 0.7));
}

/// Oracle-free sweep over designs that broke the old code (close
/// hypotheses, extreme `p₀`/`p₁`, tiny and loose error rates) and `p` from
/// `0` through `10⁻³⁰⁰`, the zero-drift point and `1 − 2⁻⁵³` to `1`: `L`
/// must be finite, in `[0, 1]` and monotone in `p` (the OC of an SPRT in a
/// monotone-likelihood-ratio family is), and `E[N]` finite and positive.
/// The old code returned `NaN`, errors and a negative `E[N]` on this grid.
#[test]
fn wald_approximations_are_finite_and_monotone_on_a_hard_grid() {
    let tiny = over_pow10(BigInt::from(1), 30);
    let designs: Vec<(Q, Q)> = vec![
        (q(7, 10), q(9, 10)),
        (q(9, 10), q(7, 10)),
        (q(1, 2), q(5_000_001, 10_000_000)),
        (tiny.clone(), q(1, 2)),
        (q(1, 2), qi(1) - tiny),
    ];
    let rates = [(0.05, 0.10), (1e-9, 1e-12), (0.45, 0.45)];
    for (p0, p1) in &designs {
        let (f0, f1): (f64, f64) = (
            num_traits::ToPrimitive::to_f64(p0).unwrap(),
            num_traits::ToPrimitive::to_f64(p1).unwrap(),
        );
        let ls = (f1 / f0).ln();
        let lf = ((1.0 - f1) / (1.0 - f0)).ln();
        let p_star = -lf / (ls - lf);
        let mut ps = vec![
            0.0,
            1e-300,
            1e-20,
            1e-3,
            0.3,
            0.5,
            0.7,
            0.9,
            1.0 - 1e-9,
            1.0 - EPS / 2.0,
            1.0,
        ];
        for k in 1..=12 {
            let d = 10f64.powi(-k);
            ps.extend([p_star * (1.0 - d), p_star * (1.0 + d)]);
        }
        ps.extend([f0, f1, p_star]);
        ps.retain(|p| (0.0..=1.0).contains(p));
        ps.sort_by(f64::total_cmp);
        for &(alpha, beta) in &rates {
            let mut last: Option<f64> = None;
            for &p in &ps {
                let label = format!("p0={p0} p1={p1} alpha={alpha} beta={beta} p={p:e}");
                let l = operating_characteristic_bernoulli(p, p0, p1, alpha, beta)
                    .unwrap_or_else(|e| panic!("{label}: {e}"));
                let n = expected_sample_size_bernoulli(p, p0, p1, alpha, beta)
                    .unwrap_or_else(|e| panic!("{label}: {e}"));
                assert!((0.0..=1.0).contains(&l), "{label}: L = {l}");
                assert!(n.is_finite() && n > 0.0, "{label}: E[N] = {n}");
                if let Some(prev) = last {
                    // Non-increasing in p when p1 > p0, non-decreasing otherwise;
                    // rounding may wobble by a few ulps.
                    let step = if p1 > p0 { l - prev } else { prev - l };
                    assert!(step <= 1e-12, "{label}: L not monotone ({prev} then {l})");
                }
                last = Some(l);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Markov chains
// ═══════════════════════════════════════════════════════════════════════════

/// `limiting_distribution` returned `None` for every reducible chain, with
/// the justification that the limit row depends on the start — false with
/// a single closed class: `P = [[1/2, 1/2], [0, 1]]` has
/// `P⁶⁰ = [[2⁻⁶⁰, 1 − 2⁻⁶⁰], [0, 1]]` (SymPy `P**60`) and `Pⁿ → [[0, 1],
/// [0, 1]]`.  A single *periodic* closed class, and two closed classes,
/// still give `None`.
#[test]
fn limiting_distribution_of_a_chain_with_one_aperiodic_closed_class() {
    let chain =
        MarkovChain::new(QMatrix::new(vec![vec![q(1, 2), q(1, 2)], vec![qi(0), qi(1)]]).unwrap())
            .unwrap();
    // SymPy: limit((1/2)**n) = 0, limit(1 - (1/2)**n) = 1
    assert_eq!(
        chain.limiting_distribution().unwrap(),
        Some(vec![qi(0), qi(1)])
    );
    // Transient state 0 feeding the aperiodic class {1, 2}: SymPy
    // (P3.T - I).nullspace() = [(0, 3/4, 1)], i.e. π = (0, 3/7, 4/7), and
    // float(P3**64) row 0 = [0, 0.428571…, 0.571428…].
    let p3 = QMatrix::new(vec![
        vec![qi(0), q(1, 2), q(1, 2)],
        vec![qi(0), q(1, 3), q(2, 3)],
        vec![qi(0), q(1, 2), q(1, 2)],
    ])
    .unwrap();
    let chain = MarkovChain::new(p3).unwrap();
    assert!(!chain.is_irreducible());
    assert_eq!(
        chain.limiting_distribution().unwrap(),
        Some(vec![qi(0), q(3, 7), q(4, 7)])
    );
    // A single closed class {1, 2} of period 2: SymPy Pp**100 row 1 =
    // (0, 1, 0), Pp**101 row 1 = (0, 0, 1) — no limit.
    let periodic = MarkovChain::new(
        QMatrix::new(vec![
            vec![q(1, 2), q(1, 4), q(1, 4)],
            vec![qi(0), qi(0), qi(1)],
            vec![qi(0), qi(1), qi(0)],
        ])
        .unwrap(),
    )
    .unwrap();
    assert_eq!(periodic.limiting_distribution().unwrap(), None);
    // Two absorbing states: start-dependent limit.
    let two = MarkovChain::new(
        QMatrix::new(vec![
            vec![qi(1), qi(0), qi(0)],
            vec![q(1, 2), qi(0), q(1, 2)],
            vec![qi(0), qi(0), qi(1)],
        ])
        .unwrap(),
    )
    .unwrap();
    assert_eq!(two.limiting_distribution().unwrap(), None);
}

/// `sample_path` converted each entry as `numer.to_f64() / denom.to_f64()`:
/// for `(10⁴⁰⁰ ∓ 1)/(2·10⁴⁰⁰)` that is `inf/inf = NaN`, every threshold
/// test failed, and the chain never left its initial state.  Each step now
/// switches with probability `(10⁴⁰⁰ + 1)/(2·10⁴⁰⁰) ≈ 1/2`, so the number of
/// switches in 4 000 steps is `Binomial(4000, ½)`: `2000 ± 190` is six
/// standard deviations.
#[test]
fn sample_path_with_huge_rationals_moves() {
    let d = BigInt::from(10).pow(400);
    let stay = Q::new(d.clone() - 1, d.clone() * 2);
    let switch = Q::new(d.clone() + 1, d * 2);
    let chain = MarkovChain::new(
        QMatrix::new(vec![vec![stay.clone(), switch.clone()], vec![switch, stay]]).unwrap(),
    )
    .unwrap();
    let path = chain.sample_path(0, 4000, &mut Rng::new(7)).unwrap();
    assert_eq!(path.len(), 4001);
    let switches = path.windows(2).filter(|w| w[0] != w[1]).count();
    assert!(
        (1810..=2190).contains(&switches),
        "{switches} switches in 4000 steps"
    );
}

/// `sample_path(_, usize::MAX, _)` computed `steps + 1` (overflow panic) and
/// `usize::MAX − 1` asked `Vec::with_capacity` for more than `isize::MAX`
/// bytes ("capacity overflow" panic).
#[test]
fn sample_path_with_huge_step_counts_is_an_error() {
    let chain = MarkovChain::new(QMatrix::new(vec![vec![qi(1)]]).unwrap()).unwrap();
    is_invalid(chain.sample_path(0, usize::MAX, &mut Rng::new(1)));
    is_invalid(chain.sample_path(0, usize::MAX - 1, &mut Rng::new(1)));
    assert_eq!(
        chain.sample_path(0, 3, &mut Rng::new(1)).unwrap(),
        vec![0; 4]
    );
}

/// Two states with one label were accepted, and `state_index` silently
/// answered with the first.
#[test]
fn with_labels_rejects_a_repeated_label() {
    let p = || QMatrix::new(vec![vec![q(1, 2), q(1, 2)], vec![qi(0), qi(1)]]).unwrap();
    is_invalid(MarkovChain::with_labels(p(), vec!["a".into(), "a".into()]));
    let chain = MarkovChain::with_labels(p(), vec!["a".into(), "b".into()]).unwrap();
    assert_eq!(chain.state_index("b"), Some(1));
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. The multivariate normal and principal components
// ═══════════════════════════════════════════════════════════════════════════

/// An unchecked `MultivariateNormal::new` with the indefinite
/// `Σ = [[1, 2], [2, 1]]` had the "density" `exp(1/6)/√(−12π²)` at
/// `(1, 0)` (imaginary), and with the asymmetric `[[1, 2], [0, 1]]` the
/// meaningless real `0.0965`; `entropy` took the logarithm of a negative
/// determinant.  They are now refused like `try_new` refuses them; a
/// singular positive-semidefinite `Σ` still fails in the inverse or the
/// Cholesky factor (`ComputationFailed`), as before.
#[test]
fn unchecked_multivariate_normal_refuses_an_invalid_covariance() {
    let ctx = Context::new();
    let zero = || vec![ctx.int(0), ctx.int(0)];
    let point = [ctx.int(1), ctx.int(0)];
    for cov in [matrix![ctx, [1, 2], [2, 1]], matrix![ctx, [1, 2], [0, 1]]] {
        let mvn = MultivariateNormal::new(zero(), cov);
        is_invalid(mvn.density(&point));
        is_invalid(mvn.mahalanobis_squared(&point));
        is_invalid(mvn.precision());
        is_invalid(mvn.entropy());
        is_invalid(mvn.sample(1, &mut Rng::new(1)));
    }
    let singular = MultivariateNormal::new(zero(), matrix![ctx, [1, 1], [1, 1]]);
    assert!(matches!(
        singular.density(&point),
        Err(SymplexError::ComputationFailed { .. })
    ));
    assert!(matches!(
        singular.sample(1, &mut Rng::new(1)),
        Err(SymplexError::ComputationFailed { .. })
    ));
    // A valid unchecked one is unaffected: scipy.stats.multivariate_normal([0, 0], [[2, 1], [1, 3]]).pdf([1, 0])
    // = 0.05272866609622072 (exactly exp(−3/10)/(2π√5) here).
    let ok = MultivariateNormal::new(zero(), matrix![ctx, [2, 1], [1, 3]]);
    let d = ok.density(&point).unwrap().eval_f64().unwrap();
    close(d, 0.05272866609622072, 1e-15, "pdf");
}

/// `MultivariateNormal::sample(usize::MAX, _)` panicked in
/// `Vec::with_capacity` ("capacity overflow").
#[test]
fn multivariate_normal_sample_with_a_huge_count_is_an_error() {
    let ctx = Context::new();
    let mvn = MultivariateNormal::try_new(vec![ctx.int(0)], matrix![ctx, [1]]).unwrap();
    is_invalid(mvn.sample(usize::MAX, &mut Rng::new(1)));
    assert_eq!(mvn.sample(2, &mut Rng::new(1)).unwrap().len(), 2);
}

/// A single constant variable got the correlation matrix `[[1]]` although
/// the docs promise an error for a constant variable (its correlation,
/// even with itself, is `0/0`; `numpy.corrcoef([[1], [1]], rowvar=False)`
/// is `nan`).  With two or more variables the constant one was already
/// refused (through `pearson`).
#[test]
fn correlation_matrix_refuses_a_single_constant_variable() {
    let ctx = Context::new();
    is_invalid(correlation_matrix(&ctx, &[vec![qi(1)], vec![qi(1)]]));
    is_invalid(correlation_matrix(&ctx, &[vec![qi(3)]]));
    let r = correlation_matrix(&ctx, &[vec![qi(1)], vec![qi(2)]]).unwrap();
    assert_eq!(r[(0, 0)], ctx.one());
}

/// The exact `pca` sorted eigenvalues by their `f64` values, which tie for
/// `1` and `1 − 10⁻²⁰`, leaving `eigenvects`' ascending order: eigenvalues
/// `[1 − 10⁻²⁰, 1]`, not decreasing as documented.  SymPy:
/// `Matrix([[1, 0], [0, 1 - 10**-20]]).eigenvals()` = `{1, 1 − 10⁻²⁰}`,
/// explained ratios `λ / (2 − 10⁻²⁰)`.
#[test]
fn pca_orders_eigenvalues_exactly() {
    let ctx = Context::new();
    let e = over_pow10(BigInt::from(1), 20);
    let small = qi(1) - e.clone();
    for cov in [
        QMatrix::new(vec![vec![qi(1), qi(0)], vec![qi(0), small.clone()]]).unwrap(),
        QMatrix::new(vec![vec![small.clone(), qi(0)], vec![qi(0), qi(1)]]).unwrap(),
    ] {
        let p = pca(&ctx, &cov).unwrap();
        assert_eq!(
            p.eigenvalues,
            vec![ctx.one(), ctx.from_ratio(small.clone())]
        );
        let total = qi(2) - e.clone();
        assert_eq!(
            p.explained_variance_ratio,
            vec![
                ctx.from_ratio(qi(1) / total.clone()),
                ctx.from_ratio(small.clone() / total)
            ]
        );
        // The first component is the axis of the variance-1 coordinate.
        let first = if cov[(0, 0)] == qi(1) { [1, 0] } else { [0, 1] };
        assert_eq!(p.components[0], vec![ctx.int(first[0]), ctx.int(first[1])]);
    }
}

/// `pca_f64` handed the matrix to Jacobi sweeps whose convergence test
/// squares the entries: for `[[2, 1], [1, 2]]·10⁻¹⁷⁰` the squares
/// underflowed to `0`, the matrix counted as diagonal, and the result was
/// eigenvalues `[2e-170, 2e-170]` with the identity as components (for
/// `10¹⁷⁰` they overflowed, with the same result).  `numpy.linalg.eigh`
/// gives `[1e-170, 2.9999999999999998e-170]` and `[1e+170, 3e+170]`
/// (LAPACK scales first; mpmath `eigsy` agrees), components `(1, ±1)/√2`.
#[test]
fn pca_f64_scales_before_the_jacobi_sweeps() {
    for s in [1e-170, 1e170] {
        let m = vec![vec![2.0 * s, s], vec![s, 2.0 * s]];
        let p = pca_f64(&m).unwrap();
        close(p.eigenvalues[0], 3.0 * s, 4.0 * EPS, "λ₁");
        close(p.eigenvalues[1], s, 4.0 * EPS, "λ₂");
        close(p.explained_variance_ratio[0], 0.75, 4.0 * EPS, "ratio₁");
        let r = std::f64::consts::FRAC_1_SQRT_2;
        close(p.components[0][0], r, 4.0 * EPS, "v₁");
        close(p.components[0][1], r, 4.0 * EPS, "v₁");
        close(p.components[1][1], -r, 4.0 * EPS, "v₂");
    }
}

/// `pca_f64`'s finiteness test took `f64::max` over the entries, which
/// skips `NaN`, so a `NaN` diagonal came back as an eigenvalue; its
/// symmetry test was absolute below `1` (`1e-12 · max(1, max|a|)`),
/// letting a 10 % asymmetry of a `10⁻²⁰`-scale matrix through; and the
/// zero matrix had `NaN` explained variance ratios.
#[test]
fn pca_f64_rejects_nan_asymmetric_and_zero_matrices() {
    is_invalid(pca_f64(&[vec![f64::NAN, 0.0], vec![0.0, 1.0]]));
    is_invalid(pca_f64(&[vec![1e-20, 1e-20], vec![1.1e-20, 1e-20]]));
    is_invalid(pca_f64(&[vec![0.0, 0.0], vec![0.0, 0.0]]));
    // Symmetric to rounding at that scale is fine.
    let p = pca_f64(&[vec![2e-20, 1e-20], vec![1e-20 * (1.0 + EPS), 2e-20]]).unwrap();
    close(p.eigenvalues[0], 3e-20, 1e-15, "λ₁");
}

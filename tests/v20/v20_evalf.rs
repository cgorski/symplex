//! 0.21 track: the `f64` route of `evalf`, bracketing and integer search in
//! `optimize`, and the cached Markov-chain classes.
//!
//! * `Ex::eval_f64` / `Ex::eval_complex64` now round the 128-bit result
//!   straight to `f64` (no 16-digit decimal string in between), so the
//!   values are the correctly rounded ones: exact rationals to the bit,
//!   transcendental constants equal to `std`'s, and tiny / huge magnitudes
//!   through the subnormals and up to the overflow threshold.
//! * `optimize::grow_bracket` and `optimize::partition_point_by` replace
//!   the hand-rolled doubling loops in `stats`.
//! * `MarkovChain` computes its communication classes once (Tarjan SCC in
//!   `base::graph`) in the order the Warshall scan produced.

use symplex::linprog::{q, qi};
use symplex::optimize::{RootOpts, brent_root, grow_bracket, partition_point_by};
use symplex::prelude::*;
use symplex::stats::markov::MarkovChain;

// ═══════════════════════════════════════════════════════════════════════════
// eval_f64: correctly rounded, exact where the value is exact
// ═══════════════════════════════════════════════════════════════════════════

fn bits(v: f64) -> u64 {
    v.to_bits()
}

#[test]
fn eval_f64_exact_rationals_are_bitwise_exact() {
    let ctx = Context::new();
    let cases: [(Ex, f64); 8] = [
        (ctx.int(2), 2.0),
        (ctx.int(-7), -7.0),
        (ctx.rational(1, 3), 1.0 / 3.0),
        (ctx.rational(-7, 11), -7.0 / 11.0),
        (ctx.rational(1, 10), 0.1),
        (ctx.int(2).powi(60), 2f64.powi(60)),
        (ctx.int(10).powi(300), 1e300),
        (ctx.int(1) / ctx.int(10).powi(300), 1e-300),
    ];
    for (e, want) in cases {
        let got = e.eval_f64().unwrap();
        assert_eq!(bits(got), bits(want), "{e}: got {got:e}, want {want:e}");
    }
    // The 9 of `x² at x = 3` comes out of `eval` exactly.
    let x = ctx.symbol("x");
    assert_eq!(x.powi(2).subs_i64(&x, 3).eval_f64().unwrap(), 9.0);
}

#[test]
fn eval_f64_constants_match_std() {
    let ctx = Context::new();
    let cases: [(Ex, f64); 8] = [
        (ctx.pi(), std::f64::consts::PI),
        (ctx.e(), std::f64::consts::E),
        (ctx.int(2).sqrt(), std::f64::consts::SQRT_2),
        (ctx.int(2).ln(), std::f64::consts::LN_2),
        (ctx.int(1).sin() + ctx.int(2).exp(), 1f64.sin() + 2f64.exp()),
        (ctx.int(3).cos(), 3f64.cos()),
        (ctx.int(700).exp(), 700f64.exp()),
        (ctx.int(2).pow(&ctx.rational(1, 3)), 2f64.cbrt()),
    ];
    for (e, want) in cases {
        let got = e.eval_f64().unwrap();
        // `std` is itself within an ulp of the true value on every platform we build for.
        assert!(
            (got - want).abs() <= 2.0 * f64::EPSILON * want.abs(),
            "{e}: got {got:.17e}, want {want:.17e}"
        );
    }
    // The classical ones are correctly rounded, so they are bit-identical.
    assert_eq!(
        bits(ctx.pi().eval_f64().unwrap()),
        bits(std::f64::consts::PI)
    );
    assert_eq!(
        bits(ctx.int(2).sqrt().eval_f64().unwrap()),
        bits(std::f64::consts::SQRT_2)
    );
    assert_eq!(bits(ctx.e().eval_f64().unwrap()), bits(std::f64::consts::E));
}

#[test]
fn eval_f64_special_functions_and_magnitudes() {
    let ctx = Context::new();
    // erf(1/2) = 0.5204998778130465376827466538919645287364515757579637…
    let erf_half = ctx.rational(1, 2).erf().eval_f64().unwrap();
    assert!((erf_half - 0.520_499_877_813_046_5).abs() < 1e-16);
    // Γ(1/2) = √π.
    let g = ctx.rational(1, 2).gamma().eval_f64().unwrap();
    assert!((g - std::f64::consts::PI.sqrt()).abs() <= 2.0 * f64::EPSILON * g);
    // Tiny: below the normal range, through the subnormals.
    let tiny = ctx.int(10).pow(&ctx.int(-320)).eval_f64().unwrap();
    assert_eq!(bits(tiny), bits(1e-320));
    assert!(tiny > 0.0 && tiny < f64::MIN_POSITIVE);
    let below = ctx.int(10).pow(&ctx.int(-400)).eval_f64().unwrap();
    assert_eq!(below, 0.0);
    // Huge: past the overflow threshold the value is ±∞, as a decimal parse gives.
    let huge = ctx.int(10).pow(&ctx.int(400)).eval_f64().unwrap();
    assert_eq!(huge, f64::INFINITY);
    let neg_huge = (-ctx.int(10).pow(&ctx.int(400))).eval_f64().unwrap();
    assert_eq!(neg_huge, f64::NEG_INFINITY);
    // Negative and large-ish.
    let v = (-ctx.pi() * ctx.int(10).powi(10)).eval_f64().unwrap();
    assert!((v + std::f64::consts::PI * 1e10).abs() <= 2.0 * f64::EPSILON * v.abs());
    // Sign of zero: an exact zero is +0.0.
    assert_eq!(bits(ctx.int(0).eval_f64().unwrap()), bits(0.0));
}

#[test]
fn eval_f64_errors_are_unchanged() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert!(matches!(
        x.eval_f64(),
        Err(SymplexError::FreeSymbol { name }) if name == "x"
    ));
    // A non-real value is refused with the same message as before.
    let err = ctx.int(-1).sqrt().eval_f64().unwrap_err();
    assert!(
        err.to_string().contains("nonzero imaginary part"),
        "unexpected error: {err}"
    );
    // A negligible imaginary part (rounding noise) is dropped.
    let v = (ctx.i_unit() * ctx.pi()).exp().eval_f64().unwrap();
    assert_eq!(v, -1.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// eval_complex64
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_complex64_cases() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let close = |z: Complex64, re: f64, im: f64| {
        assert!(
            (z.re - re).abs() <= 4.0 * f64::EPSILON * re.abs().max(1.0)
                && (z.im - im).abs() <= 4.0 * f64::EPSILON * im.abs().max(1.0),
            "got {z}, want {re} + {im}i"
        );
    };
    // Exact parts are exact.
    let z = (ctx.int(3) + ctx.int(4) * &i).eval_complex64().unwrap();
    assert_eq!((z.re, z.im), (3.0, 4.0));
    assert_eq!(
        ctx.int(-1).sqrt().eval_complex64().unwrap(),
        Complex64::new(0.0, 1.0)
    );
    assert_eq!((-&i).eval_complex64().unwrap(), Complex64::new(0.0, -1.0));
    // (1 + i)^10 = 32i: the real part is rounding noise and is dropped.
    let z = (ctx.int(1) + &i).powi(10).eval_complex64().unwrap();
    assert_eq!((z.re, z.im), (0.0, 32.0));
    // ln(−1) = iπ, e^{iπ} = −1.
    let z = ctx.int(-1).ln().eval_complex64().unwrap();
    assert_eq!((z.re, bits(z.im)), (0.0, bits(std::f64::consts::PI)));
    let z = (&i * ctx.pi()).exp().eval_complex64().unwrap();
    assert_eq!((z.re, z.im), (-1.0, 0.0));
    // exp(1 + i) = e·(cos 1 + i sin 1).
    let z = (ctx.int(1) + &i).exp().eval_complex64().unwrap();
    close(
        z,
        std::f64::consts::E * 1f64.cos(),
        std::f64::consts::E * 1f64.sin(),
    );
    // Real results have im == 0.0 exactly; a real `eval_complex64` agrees with `eval_f64`.
    let e = ctx.int(1).sin() + ctx.int(2).exp();
    let z = e.eval_complex64().unwrap();
    assert_eq!(z.im, 0.0);
    assert_eq!(bits(z.re), bits(e.eval_f64().unwrap()));
    // Parts of very different size are both kept while within the printed precision.
    let z = (ctx.int(10).pow(&ctx.int(-10)) + ctx.int(10).pow(&ctx.int(-20)) * &i)
        .eval_complex64()
        .unwrap();
    close(z, 1e-10, 1e-20);
    // Free symbols are still an error.
    assert!(ctx.symbol("y").eval_complex64().is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// grow_bracket / partition_point_by
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn grow_bracket_grows_the_side_the_root_is_on() {
    // Root at 20: only the upper end moves (0 is a bound); doubling widths 1, 2, 4, … → [0, 32].
    let f = |x: f64| x - 20.0;
    let iv = grow_bracket(f, 0.0, 1.0, Bounds::at_least(0.0), 10).unwrap();
    assert_eq!((iv.lower, iv.upper), (0.0, 32.0));
    // Unbounded: the end where |f| is smaller grows — here the upper one.
    let iv = grow_bracket(f, 0.0, 1.0, Bounds::free(), 10).unwrap();
    assert_eq!((iv.lower, iv.upper), (0.0, 32.0));
    // Root at −20: the lower end grows instead → [−31, 1].
    let g = |x: f64| x + 20.0;
    let iv = grow_bracket(g, 0.0, 1.0, Bounds::free(), 10).unwrap();
    assert_eq!((iv.lower, iv.upper), (-31.0, 1.0));
    // Already bracketed (or an exact zero at an end): returned as is, ordered.
    let iv = grow_bracket(f, 25.0, 15.0, Bounds::free(), 0).unwrap();
    assert_eq!((iv.lower, iv.upper), (15.0, 25.0));
    let iv = grow_bracket(f, 20.0, 21.0, Bounds::free(), 0).unwrap();
    assert_eq!((iv.lower, iv.upper), (20.0, 21.0));
    // The bracket feeds Brent.
    let iv = grow_bracket(f, 0.0, 1.0, Bounds::at_least(0.0), 10).unwrap();
    let r = brent_root(f, iv.lower, iv.upper, &RootOpts::default()).unwrap();
    assert!((r - 20.0).abs() < 1e-12);
}

#[test]
fn grow_bracket_respects_bounds_and_budget() {
    let f = |x: f64| x - 20.0;
    // The upper end is clipped to the bound and the root found there.
    let iv = grow_bracket(f, 0.0, 1.0, Bounds::closed(0.0, 24.0), 10).unwrap();
    assert_eq!((iv.lower, iv.upper), (0.0, 24.0));
    // No root inside the bounds: both ends pinned → error, not a bogus bracket.
    let err = grow_bracket(f, 0.0, 1.0, Bounds::closed(0.0, 8.0), 10).unwrap_err();
    assert!(
        matches!(err, SymplexError::ComputationFailed { .. }),
        "{err}"
    );
    // Budget exhausted before the sign change.
    let err = grow_bracket(f, 0.0, 1.0, Bounds::at_least(0.0), 2).unwrap_err();
    assert!(
        matches!(err, SymplexError::ComputationFailed { .. }),
        "{err}"
    );
    // Never a sign change (x² + 1): error after the budget.
    assert!(grow_bracket(|x| x * x + 1.0, -1.0, 1.0, Bounds::free(), 8).is_err());
    // Bad input.
    assert!(matches!(
        grow_bracket(f, 1.0, 1.0, Bounds::free(), 4),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        grow_bracket(f, 0.0, 1.0, Bounds::closed(2.0, 3.0), 4),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        grow_bracket(|x: f64| x.ln(), 0.0, 1.0, Bounds::free(), 4),
        Err(SymplexError::InvalidArgument { .. })
    ));
    // Non-finite value met while growing.
    let h = |x: f64| if x > 4.0 { f64::NAN } else { x - 20.0 };
    assert!(matches!(
        grow_bracket(h, 0.0, 1.0, Bounds::at_least(0.0), 10),
        Err(SymplexError::ComputationFailed { .. })
    ));
}

#[test]
fn partition_point_by_finds_the_first_true() {
    // Every threshold in a range, against the definition.
    for lo in 0..6usize {
        for hi in lo..40usize {
            for t in 0..45usize {
                let mut calls = 0;
                let got = partition_point_by(lo, hi, |n| {
                    calls += 1;
                    n >= t
                });
                let want = (lo..hi).find(|&n| n >= t).unwrap_or(hi);
                assert_eq!(got, want, "lo={lo} hi={hi} t={t}");
                assert!(
                    calls <= 2 * (usize::BITS as usize),
                    "lo={lo} hi={hi} t={t}: {calls} calls"
                );
            }
        }
    }
    // A huge cap costs logarithmically many probes.
    let mut calls = 0;
    let n = partition_point_by(1, usize::MAX / 2, |n| {
        calls += 1;
        n * n >= 1_000_000_007
    });
    assert_eq!(n, 31_623);
    assert!(calls < 40, "{calls} probes");
    // Cap hit: `hi` is returned and the predicate is never called past it.
    let mut max_probe = 0;
    let n = partition_point_by(3, 100, |n| {
        max_probe = max_probe.max(n);
        false
    });
    assert_eq!(n, 100);
    assert!(max_probe < 100);
    // Empty and inverted ranges.
    assert_eq!(partition_point_by(5, 5, |_| true), 5);
    assert_eq!(partition_point_by(9, 5, |_| true), 5);
}

#[test]
fn stats_users_of_the_new_helpers_agree_with_the_oracles() {
    use symplex::stats::estimation::{IntervalMethod, proportion_interval};
    use symplex::stats::hypothesis::sample_size_two_proportions;
    use symplex::stats::sequential::operating_characteristic_bernoulli;
    // statsmodels: proportion_confint(3, 10, alpha=0.05, method='beta') = (0.06673951117773447, 0.6524528500599972)
    let ci = proportion_interval(3, 10, 0.95, IntervalMethod::ClopperPearson).unwrap();
    assert!((ci.lower - 0.066_739_511_177_734_47).abs() < 1e-12);
    assert!((ci.upper - 0.652_452_850_059_997_2).abs() < 1e-12);
    // Edge counts keep their closed forms.
    let ci = proportion_interval(0, 10, 0.95, IntervalMethod::ClopperPearson).unwrap();
    assert_eq!(ci.lower, 0.0);
    assert!((ci.upper - (1.0 - 0.025f64.powf(0.1))).abs() < 1e-12);
    // Wald's OC at p₀ is 1 − α and at p₁ is β (h = ±1).
    let oc0 = operating_characteristic_bernoulli(0.7, &q(7, 10), &q(9, 10), 0.05, 0.10).unwrap();
    let oc1 = operating_characteristic_bernoulli(0.9, &q(7, 10), &q(9, 10), 0.05, 0.10).unwrap();
    assert!(
        (oc0 - 0.95).abs() < 1e-9 && (oc1 - 0.10).abs() < 1e-9,
        "{oc0} {oc1}"
    );
    // statsmodels: ceil(387.1677468578098) = 388.
    assert_eq!(
        sample_size_two_proportions(0.5, 0.4, 0.05, 0.8).unwrap(),
        388
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// MarkovChain: cached classes in the Warshall order
// ═══════════════════════════════════════════════════════════════════════════

/// A 6-state reducible chain.  Positive transitions:
///
/// * `5 → 5` (absorbing),
/// * `2 → 4`, `4 → 2` (a closed pair, listed as `[2, 4]`),
/// * `0 → 1`, `1 → 0`, `1 → 5` (transient pair `[0, 1]` leaking to 5),
/// * `3 → 2`, `3 → 3` (transient singleton `[3]` feeding the pair).
///
/// Classes by smallest member: `[0, 1]`, `[2, 4]`, `[3]`, `[5]`; closed:
/// `[2, 4]`, `[5]`; transient: `0, 1, 3`.
fn six_state_chain() -> MarkovChain {
    let h = q(1, 2);
    let z = qi(0);
    let o = qi(1);
    let p = QMatrix::new(vec![
        vec![
            z.clone(),
            o.clone(),
            z.clone(),
            z.clone(),
            z.clone(),
            z.clone(),
        ],
        vec![
            h.clone(),
            z.clone(),
            z.clone(),
            z.clone(),
            z.clone(),
            h.clone(),
        ],
        vec![
            z.clone(),
            z.clone(),
            z.clone(),
            z.clone(),
            o.clone(),
            z.clone(),
        ],
        vec![
            z.clone(),
            z.clone(),
            h.clone(),
            h.clone(),
            z.clone(),
            z.clone(),
        ],
        vec![
            z.clone(),
            z.clone(),
            o.clone(),
            z.clone(),
            z.clone(),
            z.clone(),
        ],
        vec![
            z.clone(),
            z.clone(),
            z.clone(),
            z.clone(),
            z.clone(),
            o.clone(),
        ],
    ])
    .unwrap();
    MarkovChain::new(p).unwrap()
}

#[test]
fn markov_classes_keep_the_warshall_order() {
    let chain = six_state_chain();
    assert_eq!(
        chain.communication_classes(),
        vec![vec![0, 1], vec![2, 4], vec![3], vec![5]]
    );
    assert_eq!(chain.closed_classes(), vec![vec![2, 4], vec![5]]);
    assert_eq!(chain.transient_states(), vec![0, 1, 3]);
    assert!(!chain.is_irreducible());
    // Periods: the 2 ⇄ 4 pair has period 2; 5 has period 1; 3 has a self-loop (period 1);
    // 0 ⇄ 1 has period 2 as well (the leak does not change the cycle lengths).
    assert_eq!(chain.period_of(2).unwrap(), Some(2));
    assert_eq!(chain.period_of(5).unwrap(), Some(1));
    assert_eq!(chain.period_of(3).unwrap(), Some(1));
    assert_eq!(chain.period_of(0).unwrap(), Some(2));
    assert!(!chain.is_aperiodic());
    // One stationary distribution per closed class, in class order.
    let pis = chain.stationary_distributions();
    assert_eq!(pis.len(), 2);
    assert_eq!(pis[0], vec![qi(0), qi(0), q(1, 2), qi(0), q(1, 2), qi(0)]);
    assert_eq!(pis[1], vec![qi(0), qi(0), qi(0), qi(0), qi(0), qi(1)]);
    assert!(chain.stationary_distribution().is_err());
    assert_eq!(chain.limiting_distribution().unwrap(), None);
    // Repeated queries hit the cache and agree; a clone carries it.
    let again = chain.clone();
    assert_eq!(again.communication_classes(), chain.communication_classes());
    assert_eq!(again, chain);
}

#[test]
fn markov_classes_on_the_walkthrough_chains() {
    // Irreducible periodic 3-cycle: one class, period 3, stationary but no limit.
    let cyc = MarkovChain::new(
        QMatrix::new(vec![
            vec![qi(0), qi(1), qi(0)],
            vec![qi(0), qi(0), qi(1)],
            vec![qi(1), qi(0), qi(0)],
        ])
        .unwrap(),
    )
    .unwrap();
    assert_eq!(cyc.communication_classes(), vec![vec![0, 1, 2]]);
    assert!(cyc.is_irreducible() && !cyc.is_aperiodic());
    assert_eq!(cyc.period_of(1).unwrap(), Some(3));
    assert_eq!(
        cyc.stationary_distribution().unwrap(),
        vec![q(1, 3), q(1, 3), q(1, 3)]
    );
    assert_eq!(cyc.limiting_distribution().unwrap(), None);
    // Gambler's ruin on 0..=4: two absorbing singletons, the middle three transient.
    let h = q(1, 2);
    let ruin = MarkovChain::new(
        QMatrix::new(vec![
            vec![qi(1), qi(0), qi(0), qi(0), qi(0)],
            vec![h.clone(), qi(0), h.clone(), qi(0), qi(0)],
            vec![qi(0), h.clone(), qi(0), h.clone(), qi(0)],
            vec![qi(0), qi(0), h.clone(), qi(0), h.clone()],
            vec![qi(0), qi(0), qi(0), qi(0), qi(1)],
        ])
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        ruin.communication_classes(),
        vec![vec![0], vec![1, 2, 3], vec![4]]
    );
    assert_eq!(ruin.closed_classes(), vec![vec![0], vec![4]]);
    assert_eq!(ruin.transient_states(), vec![1, 2, 3]);
    assert_eq!(ruin.period_of(2).unwrap(), Some(2));
    assert_eq!(ruin.period_of(0).unwrap(), Some(1));
    let pis = ruin.stationary_distributions();
    assert_eq!(pis[0], vec![qi(1), qi(0), qi(0), qi(0), qi(0)]);
    assert_eq!(pis[1], vec![qi(0), qi(0), qi(0), qi(0), qi(1)]);
    // A state that never returns has no period; states are labelled in reverse
    // to check the order does not depend on the input labelling.
    let rev = MarkovChain::new(
        QMatrix::new(vec![
            vec![qi(1), qi(0), qi(0)],
            vec![qi(1), qi(0), qi(0)],
            vec![qi(0), qi(1), qi(0)],
        ])
        .unwrap(),
    )
    .unwrap();
    assert_eq!(rev.communication_classes(), vec![vec![0], vec![1], vec![2]]);
    assert_eq!(rev.period_of(2).unwrap(), None);
    assert_eq!(rev.period_of(1).unwrap(), None);
    assert!(rev.is_aperiodic());
}

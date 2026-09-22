//! 0.21 track: `base` — the shared fundamentals lifted out of the domain
//! modules in 0.21: the deterministic generators (`base::rng`), the call
//! budget (`base::budget`), `Extended<T>` behind the set normal form, and
//! the integer combinatorics kernels.
//!
//! Every pin here was captured from the 0.20 code **before** the move:
//! the random streams decide which factor a randomised split finds first
//! and which Monte-Carlo sample a test sees, so they are output.

use std::f64::consts::PI;
use std::time::{Duration, Instant};

use num_bigint::BigInt;
use symplex::certificates::{PolyhedronOpts, SosOpts};
use symplex::combinatorics::{binomial, factorial, multinomial};
use symplex::linprog::{Budget, BudgetHit, LpProblem, LpStatus, qi};
use symplex::optimize::{DeOpts, differential_evolution};
use symplex::prelude::*;
use symplex::stats::hypothesis::{BootstrapMethod, bootstrap_ci};
use symplex::stats::{Distribution, Rng};
use symplex::{SplitMix64, XorShift64Star};

// ═══════════════════════════════════════════════════════════════════════════
// RNG streams — bit-identical to 0.20
// ═══════════════════════════════════════════════════════════════════════════

/// The first ten `next_u64` of `SplitMix64::new(42)` in 0.20
/// (`stats::sample::Rng`, `optimize::SplitMix64`, `expr_ops::SplitMix64`
/// — three copies of the same constants).
const SPLITMIX64_SEED_42: [u64; 10] = [
    0xBDD7_3226_2FEB_6E95,
    0x28EF_E333_B266_F103,
    0x4752_6757_130F_9F52,
    0x581C_E1FF_0E4A_E394,
    0x09BC_585A_2448_23F2,
    0xDE44_31FA_3C80_DB06,
    0x37E9_671C_4537_6D5D,
    0xCCF6_35EE_9E9E_2FA4,
    0x5705_B877_0B3D_7DD5,
    0x9E54_D738_297F_77AE,
];

/// The first ten `next_u64` of `XorShift::new(42)` in 0.20
/// (`ntheory::XorShift`, `factor_zassenhaus::XorShift`).
const XORSHIFT64STAR_SEED_42: [u64; 10] = [
    0x1C28_3E14_F85F_D6CB,
    0x4662_E80E_A35A_B5C5,
    0x2424_CC93_9F3B_2C9C,
    0xB940_308D_A56D_A42C,
    0x2303_AA76_50D0_3476,
    0x2F95_D8FA_1D95_7332,
    0xBB99_E78C_1FBF_492C,
    0x1B17_2B52_75BA_3D78,
    0x166E_5106_DF8E_24EF,
    0x768B_89B6_037C_625F,
];

#[test]
fn splitmix64_stream_is_pinned() {
    let mut rng = SplitMix64::new(42);
    let got: Vec<u64> = (0..10).map(|_| rng.next_u64()).collect();
    assert_eq!(got, SPLITMIX64_SEED_42);
    // `next_f64` is the top 53 bits over 2⁵³; `below` is plain modulo.
    let mut rng = SplitMix64::new(42);
    assert_eq!(
        rng.next_f64(),
        (SPLITMIX64_SEED_42[0] >> 11) as f64 / (1u64 << 53) as f64
    );
    let mut rng = SplitMix64::new(42);
    let below: Vec<usize> = (0..8).map(|_| rng.below(7)).collect();
    assert_eq!(below, [5, 5, 0, 2, 6, 4, 2, 6]);
    assert_eq!(SplitMix64::new(0).below(0), 0);
}

#[test]
fn xorshift64star_stream_is_pinned() {
    let mut rng = XorShift64Star::new(42);
    let got: Vec<u64> = (0..10).map(|_| rng.next_u64()).collect();
    assert_eq!(got, XORSHIFT64STAR_SEED_42);
    // Seed 0 is replaced by 1 (the all-zero xorshift state is a fixed point).
    assert_eq!(
        XorShift64Star::new(0).next_u64(),
        XorShift64Star::new(1).next_u64()
    );
    let n = BigInt::from(1_000_000_007u64);
    let mut rng = XorShift64Star::new(5);
    for _ in 0..20 {
        let x = rng.next_big_below(&n);
        assert!(x >= BigInt::from(0) && x < n);
    }
}

#[test]
fn stats_rng_is_the_shared_splitmix64() {
    // `symplex::stats::Rng` is a re-export: same type, same stream.
    let mut a = Rng::new(42);
    let mut b = SplitMix64::new(42);
    for _ in 0..10 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
    let _: SplitMix64 = Rng::new(1);
}

fn rastrigin(p: &[f64]) -> f64 {
    10.0 * p.len() as f64
        + p.iter()
            .map(|x| x * x - 10.0 * (2.0 * PI * x).cos())
            .sum::<f64>()
}

/// `differential_evolution` with `DeOpts { seed: 12345 }` on the 2-D
/// Rastrigin function (the seed of `tests/v03/v03_optimize.rs`), captured
/// bit-for-bit from 0.20.
#[test]
fn differential_evolution_seed_12345_is_unchanged() {
    let bounds = [Interval::closed(-5.12, 5.12), Interval::closed(-5.12, 5.12)];
    let opts = DeOpts {
        seed: 12345,
        ..DeOpts::default()
    };
    let r = differential_evolution(rastrigin, &bounds, &opts).unwrap();
    assert_eq!(r.x.len(), 2);
    assert_eq!(r.x[0].to_bits(), 0x3E21_1991_4B41_41B1, "x[0] = {}", r.x[0]);
    assert_eq!(r.x[1].to_bits(), 0x3E03_9E1D_C1E4_B3E8, "x[1] = {}", r.x[1]);
    assert_eq!(r.fun, 0.0);
    assert_eq!(r.iterations, 85);
    assert_eq!(r.evaluations, 2637);
    assert!(r.converged);
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

/// The `bootstrap_ci` call of
/// `v13_hypothesis::bootstrap_ci_is_deterministic_and_covers_the_true_mean`
/// (data: `Normal(10, 2)`, 400 draws from `Rng::new(2024)`; 2000
/// resamples from `Rng::new(1)`), captured bit-for-bit from 0.20 — the
/// quantile sampler, the gamma/Poisson kernels and the resampling all
/// read the shared stream.
#[test]
fn sampling_and_bootstrap_with_fixed_seeds_are_unchanged() {
    let ctx = Context::new();
    let normal = Distribution::normal(ctx.int(10), ctx.int(2));
    let data = normal.sample(400, &mut Rng::new(2024)).unwrap();
    assert_eq!(data.len(), 400);
    assert_eq!(data[0].to_bits(), 0x4025_4042_1D77_328E, "{}", data[0]);
    assert_eq!(data[1].to_bits(), 0x401D_9EBF_5A14_2D04, "{}", data[1]);
    assert_eq!(data[2].to_bits(), 0x4021_E2D0_E2F9_0E26, "{}", data[2]);

    let ci = bootstrap_ci(
        &data,
        mean,
        2000,
        0.95,
        &mut Rng::new(1),
        BootstrapMethod::Percentile,
    )
    .unwrap();
    assert_eq!(ci.lower.to_bits(), 0x4023_D960_620C_439F, "{}", ci.lower);
    assert_eq!(ci.upper.to_bits(), 0x4024_A32A_31B7_91A4, "{}", ci.upper);
    assert!(ci.lower < 10.0 && 10.0 < ci.upper, "{ci}");

    // Marsaglia–Tsang gamma and Hörmann PTRS Poisson kernels.
    let gamma = Distribution::gamma(ctx.int(3), ctx.int(2));
    let g = gamma.sample(3, &mut Rng::new(1)).unwrap();
    assert_eq!(g[0].to_bits(), 0x401B_7329_BCF0_01AA, "{}", g[0]);
    assert_eq!(g[1].to_bits(), 0x3FF3_2ADE_24BC_1556, "{}", g[1]);
    assert_eq!(g[2].to_bits(), 0x4023_274C_CDAF_1EA7, "{}", g[2]);
    let poisson = Distribution::poisson(ctx.int(50));
    let p = poisson.sample(5, &mut Rng::new(7)).unwrap();
    assert_eq!(p, [48.0, 61.0, 49.0, 49.0, 41.0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Budget
// ═══════════════════════════════════════════════════════════════════════════

/// min x + y  s.t.  x + 2y ≥ 1,  3x + y ≥ 1: two pivots.
fn two_pivot_lp() -> LpProblem {
    LpProblem::minimize(vec![qi(1), qi(1)])
        .ge(vec![qi(1), qi(2)], qi(1))
        .ge(vec![qi(3), qi(1)], qi(1))
}

#[test]
fn budget_builders_and_defaults() {
    assert!(Budget::default().is_unlimited());
    assert_eq!(Budget::default(), Budget::unlimited());
    let b = Budget::max_pivots(1);
    assert_eq!(b.max_pivots, Some(1));
    assert!(b.deadline.is_none() && b.time_limit.is_none());
    assert_eq!(b, Budget::default().with_max_pivots(1));
    assert_eq!(b, Budget::default().with_max_steps(1));
    let now = Instant::now();
    let b = Budget::deadline(now).with_time_limit(Duration::from_secs(5));
    assert_eq!(b.deadline, Some(now));
    assert_eq!(b.time_limit, Some(Duration::from_secs(5)));
    let b = Budget::time_limit(Duration::from_secs(5)).with_deadline(now);
    assert_eq!(b.deadline, Some(now));
    // The crate-root and `linprog` paths name the same type.
    let _: symplex::Budget = Budget::default();
    let _: symplex::BudgetHit = BudgetHit::Deadline;
    assert_eq!(BudgetHit::Deadline.to_string(), "deadline");
    assert_eq!(BudgetHit::MaxPivots.to_string(), "max_pivots");
}

#[test]
fn budget_deadline_in_the_past_is_hit_before_the_first_pivot() {
    let past = Instant::now() - Duration::from_secs(1);
    assert!(Budget::deadline(past).deadline_passed());
    assert_eq!(
        Budget::deadline(past).exhausted(0),
        Some(BudgetHit::Deadline)
    );
    assert!(!Budget::default().deadline_passed());
    let sol = two_pivot_lp()
        .with_budget(Budget::deadline(past))
        .solve()
        .unwrap();
    assert_eq!(sol.status, LpStatus::BudgetExhausted);
    assert!(sol.x.is_empty() && sol.objective.is_none());
}

#[test]
fn budget_max_pivots_caps_the_solve() {
    let sol = two_pivot_lp()
        .with_budget(Budget::max_pivots(1))
        .solve()
        .unwrap();
    assert_eq!(sol.status, LpStatus::BudgetExhausted);
    let sol = two_pivot_lp()
        .with_budget(Budget::max_pivots(1000))
        .solve()
        .unwrap();
    assert_eq!(sol.status, LpStatus::Optimal);
    assert_eq!(Budget::max_pivots(2).exhausted(1), None);
    assert_eq!(
        Budget::max_pivots(2).exhausted(2),
        Some(BudgetHit::MaxPivots)
    );
}

#[test]
fn budget_within_is_eager_and_time_limit_is_per_call() {
    // `within` fixes the deadline now: zero means already passed.
    assert!(Budget::within(Duration::ZERO).deadline_passed());
    let sol = two_pivot_lp()
        .with_budget(Budget::within(Duration::ZERO))
        .solve()
        .unwrap();
    assert_eq!(sol.status, LpStatus::BudgetExhausted);
    // A generous eager window changes nothing.
    let sol = two_pivot_lp()
        .with_budget(Budget::within(Duration::from_secs(60)).with_max_pivots(1000))
        .solve()
        .unwrap();
    assert_eq!(sol.status, LpStatus::Optimal);
    // `time_limit` is relative: not a deadline until the call starts …
    let relative = Budget::time_limit(Duration::ZERO);
    assert!(!relative.deadline_passed());
    let started = relative.start();
    assert!(started.time_limit.is_none() && started.deadline_passed());
    // … and the simplex starts it at `solve`.
    let sol = two_pivot_lp().with_budget(relative).solve().unwrap();
    assert_eq!(sol.status, LpStatus::BudgetExhausted);
    let sol = two_pivot_lp()
        .with_budget(Budget::time_limit(Duration::from_secs(60)))
        .solve()
        .unwrap();
    assert_eq!(sol.status, LpStatus::Optimal);
    // The earlier of deadline and now + limit wins.
    let far = Instant::now() + Duration::from_secs(3600);
    let b = Budget::deadline(far)
        .with_time_limit(Duration::ZERO)
        .start();
    assert!(b.deadline.is_some_and(|d| d < far));
}

#[test]
fn prover_options_expose_their_budget() {
    let far = Instant::now() + Duration::from_secs(3600);
    let opts = PolyhedronOpts::default()
        .with_deadline(far)
        .with_time_limit(Duration::from_millis(250))
        .with_max_pivots(500);
    let b = opts.budget();
    assert_eq!(b.deadline, Some(far));
    assert_eq!(b.time_limit, Some(Duration::from_millis(250)));
    assert_eq!(b.max_pivots, Some(500));
    // Round trip through `with_budget`.
    let again = PolyhedronOpts::default().with_budget(b.clone());
    assert_eq!(again.deadline, opts.deadline);
    assert_eq!(again.time_limit, opts.time_limit);
    assert_eq!(again.max_pivots, opts.max_pivots);
    assert_eq!(again.budget(), b);
    assert!(PolyhedronOpts::default().budget().is_unlimited());

    let sos = SosOpts::default()
        .with_deadline(far)
        .with_time_limit(Duration::from_millis(250));
    let b = sos.budget();
    assert_eq!(b.deadline, Some(far));
    assert_eq!(b.time_limit, Some(Duration::from_millis(250)));
    assert_eq!(b.max_pivots, None, "the SDP search has no pivot cap");
    let again = SosOpts::default().with_budget(Budget::max_pivots(7).with_deadline(far));
    assert_eq!(again.deadline, Some(far));
    assert_eq!(again.time_limit, None);
    assert!(SosOpts::default().budget().is_unlimited());
}

// ═══════════════════════════════════════════════════════════════════════════
// Sets on `Extended<usize>`
// ═══════════════════════════════════════════════════════════════════════════

fn iv(ctx: &Context, a: i64, b: i64, lo_open: bool, hi_open: bool) -> SetEx {
    ctx.interval(
        &ctx.int(a),
        &ctx.int(b),
        IntervalKind::from_open_ends(lo_open, hi_open),
    )
}

/// The normal form is unchanged by the move of the rank positions onto
/// `Extended<usize>` and of the pieces onto `Interval<Extended<usize>>`:
/// the `Display` strings are those of `tests/v02/v02_sets_algebra.rs`.
#[test]
fn set_normal_form_over_extended_ranks() {
    let ctx = Context::new();
    let s = |e: &SetEx| e.to_string();

    let a = iv(&ctx, 0, 2, false, false);
    let b = iv(&ctx, 1, 3, false, false);
    assert_eq!(s(&a.intersection(&b).simplify()), "[1, 2]");

    let a = iv(&ctx, 3, 4, false, false);
    let b = iv(&ctx, 0, 1, false, true);
    let c = iv(&ctx, 1, 2, false, false);
    assert_eq!(s(&a.union(&b).union(&c).simplify()), "[0, 2] ∪ [3, 4]");

    let a = iv(&ctx, 0, 3, false, false);
    let b = iv(&ctx, 1, 2, true, true);
    assert_eq!(s(&a.difference(&b)), "[0, 1] ∪ [2, 3]");
    assert_eq!(s(&b.difference(&a)), "EmptySet");
    let c = iv(&ctx, 2, 5, false, false);
    assert_eq!(s(&a.symmetric_difference(&c)), "[0, 2) ∪ (3, 5]");
    let b = iv(&ctx, 1, 2, false, false);
    assert_eq!(s(&a.complement(&b).simplify()), "[0, 1) ∪ (2, 3]");

    // The infinities are `Extended::{NegInf, PosInf}`, forced open.
    let a = iv(&ctx, 0, 1, false, false);
    assert_eq!(s(&a.absolute_complement()), "(-oo, 0) ∪ (1, oo)");
    assert_eq!(s(&ctx.empty_set().absolute_complement()), "(-oo, oo)");
    let p = ctx.finite_set(&[ctx.int(0)]);
    assert_eq!(s(&p.absolute_complement()), "(-oo, 0) ∪ (0, oo)");
    assert_eq!(s(&p.absolute_complement().absolute_complement()), "{0}");

    // Topology and accessors.
    let a = iv(&ctx, 0, 1, true, false);
    let u = a.union(&ctx.finite_set(&[ctx.int(2)]));
    assert_eq!(s(&u.closure().unwrap()), "[0, 1] ∪ {2}");
    assert_eq!(s(&u.interior().unwrap()), "(0, 1)");
    assert_eq!(s(&u.boundary().unwrap()), "{0, 1, 2}");
    assert_eq!(u.is_open(), Some(false));
    assert_eq!(u.is_closed(), Some(false));
    let half = ctx.interval(&ctx.neg_infinity(), &ctx.int(0), IntervalKind::LeftOpen);
    assert_eq!(half.is_closed(), Some(true));
    assert_eq!(s(&half.boundary().unwrap()), "{0}");
    let x = ctx.symbol("x");
    let parts = (&x.powi(2) - 4).solve_ge(&x).as_intervals().unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].kind, IntervalKind::LeftOpen);
    assert_eq!(parts[1].kind, IntervalKind::RightOpen);
    assert_eq!(s(&parts[0].to_set()), "(-oo, -2]");
    assert_eq!(s(&parts[1].to_set()), "[2, oo)");
    assert_eq!(
        iv(&ctx, 0, 1, false, false)
            .union(&ctx.finite_set(&[ctx.int(5)]))
            .measure()
            .unwrap()
            .to_string(),
        "1"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Combinatorics kernels
// ═══════════════════════════════════════════════════════════════════════════

/// Reference values from CPython 3.14.2: `math.factorial`, `math.comb`
/// (`python3 -c 'import math; print(math.factorial(52), math.comb(70, 35))'`).
#[test]
fn factorial_and_binomial_match_python_math() {
    assert_eq!(factorial(0), BigInt::from(1));
    assert_eq!(factorial(1), BigInt::from(1));
    assert_eq!(factorial(5), BigInt::from(120));
    assert_eq!(factorial(20), BigInt::from(2_432_902_008_176_640_000u64));
    assert_eq!(factorial(25).to_string(), "15511210043330985984000000");
    assert_eq!(
        factorial(52).to_string(),
        "80658175170943878571660636856403766975289505440883277824000000000000"
    );
    assert_eq!(
        factorial(100).to_string(),
        "93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000"
    );
    // Binary splitting agrees with the running product at every size,
    // including across the leaf width.
    let mut running = BigInt::from(1);
    for n in 1..=120u64 {
        running *= n;
        assert_eq!(factorial(n), running, "{n}!");
    }

    assert_eq!(binomial(10, 3), BigInt::from(120));
    assert_eq!(binomial(64, 32).to_string(), "1832624140942590534");
    assert_eq!(binomial(66, 33).to_string(), "7219428434016265740");
    assert_eq!(binomial(70, 35).to_string(), "112186277816662845432");
    assert_eq!(
        binomial(200, 100).to_string(),
        "90548514656103281165404177077484163874504589675413336841320"
    );
    assert_eq!(binomial(5, 7), BigInt::from(0));
    assert_eq!(binomial(-3, 2), BigInt::from(6));
    assert_eq!(multinomial(12, &[3, 4, 5]), Some(BigInt::from(27_720)));
}

/// The symbolic `binomial` node keeps SymPy's semantics through the shared
/// kernel: `C(1, 2) = 0`, a negative upper index stays unevaluated.
#[test]
fn eval_binomial_keeps_the_negative_n_guard() {
    let ctx = Context::new();
    let b = |n: i64, k: i64| ctx.int(n).binomial(&ctx.int(k)).eval().to_string();
    assert_eq!(b(10, 3), "120");
    assert_eq!(b(1, 2), "0");
    assert_eq!(b(70, 35), "112186277816662845432");
    assert_eq!(b(-3, 2), "C(-3, 2)");
    assert_eq!(
        ctx.int(20).factorial().eval().to_string(),
        "2432902008176640000"
    );
}

/// `fu::binomial_coeff` (behind `trig_power_linearize`) is `BigInt`-backed
/// now; the linearisation of the largest accepted power is exact.  (The
/// old `i64` version overflowed at `C(66, k)`; `tr_power` only accepts
/// exponents `3..=12`, so no public path reached it — the kernel is pinned
/// at `C(70, 35)` above.)
#[test]
fn trig_power_linearisation_of_the_largest_power_is_exact() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (name, e) in [("sin", x.sin().powi(12)), ("cos", x.cos().powi(12))] {
        let lin = e.trig_power_linearize();
        assert!(!lin.to_string().contains("^12"), "{name}: {lin}");
        // The constant term is C(12, 6) / 2¹² = 924/4096 = 231/1024.
        assert!(lin.to_string().contains("231/1024"), "{name}: {lin}");
        for t in [0.3_f64, 1.1, 2.7] {
            let want = e.eval_f64_with(&[(&x, t)]).unwrap();
            let got = lin.eval_f64_with(&[(&x, t)]).unwrap();
            assert!(
                (want - got).abs() < 1e-12,
                "{name}^12 at {t}: {want} vs {got}"
            );
        }
    }
}

//! Performance regressions, pinned so that they cannot come back.
//!
//! Timing assertions here never compare against an absolute wall-clock
//! bound (a loaded machine makes any such bound flaky, and a fast one makes
//! it toothless).  Each compares the workload under test with a reference
//! workload of about the same duration when the fast path is taken, summed
//! over many interleaved rounds, so that both sides see the same machine
//! load.  The two differ by a large factor the other way when the fast
//! path is not taken, and the bound sits near the geometric mean.
//! (Checked with 64 timing-test processes at once on 16 cores: best-of-N
//! rounds, or a short reference against one long run, failed there.)
//!
//! Reference values: SymPy 1.14 by the call quoted next to each.

use std::time::{Duration, Instant};

use num_bigint::BigInt;
use num_rational::BigRational;
use symplex::prelude::*;
use symplex::stats::hypothesis::{Alternative, pearson_test};

fn big(s: &str) -> BigInt {
    s.parse().expect("integer literal")
}

/// Wall-clock time of `f` and its result.
fn timed<T>(f: impl FnOnce() -> T) -> (Duration, T) {
    let t = Instant::now();
    let out = f();
    (t.elapsed(), out)
}

/// `f` run on a fresh thread, whose radical memo (thread-local) is empty.
fn on_fresh_thread<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::spawn(f).join().expect("worker thread")
}

/// Total times of `a` and `b` over `rounds` interleaved rounds (`a b a b …`).
fn interleaved_totals(
    rounds: usize,
    mut a: impl FnMut(usize) -> Duration,
    mut b: impl FnMut(usize) -> Duration,
) -> (Duration, Duration) {
    let (mut total_a, mut total_b) = (Duration::ZERO, Duration::ZERO);
    for round in 0..rounds {
        total_a += a(round);
        total_b += b(round);
    }
    (total_a, total_b)
}

// ── radicals of mid-size integers ──────────────────────────────────────────

/// `√n.simplify()` for `n = 3·5·29·111288545009·1616249891189` (87 bits)
/// took 0.99 s in 0.28.0: building `√n` factored the 77-bit cofactor
/// once (66 ms, debug build) and `simplify` factored it again 14 times
/// (`eval` and every rebuilt node re-canonicalised the radical: 925 ms).
/// Now the decomposition is memoised per thread and `simplify` costs
/// 1.3 ms against a 17 ms build, so the comparison has an order of
/// magnitude of margin on both sides.
#[test]
fn sqrt_of_an_87_bit_integer_is_decomposed_once() {
    // sympy.factorint(78243492961199594876179935)
    //   → {3: 1, 5: 1, 29: 1, 111288545009: 1, 1616249891189: 1}: squarefree,
    // sympy.sqrt(78243492961199594876179935) → sqrt(78243492961199594876179935)
    let n = big("78243492961199594876179935");
    let ctx = Context::new();
    let mut trials = Vec::new();
    // n·m for three m: distinct keys, so every build is cold (same cofactor).
    for m in [1u32, 7, 11] {
        let x = ctx.from_bigint(&n * m);
        let (build, root) = timed(|| x.sqrt());
        let (simplify, simplified) = timed(|| root.simplify());
        assert_eq!(simplified, root, "√({m}·n) is already canonical");
        trials.push((build, simplify));
    }
    assert_eq!(
        format!("{}", ctx.from_bigint(n).sqrt().simplify()),
        "sqrt(78243492961199594876179935)"
    );
    assert!(
        trials.iter().any(|(build, simplify)| simplify < build),
        "simplify re-factors the radicand: (build, simplify) = {trials:?}"
    );
}

/// The splitting of a 65- to 84-bit cofactor itself: in 0.28.0 an 84-bit
/// semiprime with 42-bit factors cost 95× a 62-bit one (three wasted rho
/// passes of 2¹⁴ steps, then ECM without stage 2: 136 ms against 1.4 ms,
/// debug build); with one rho pass and ECM stage 2 it is 7× (9.1 ms).  So
/// one 84-bit radical is timed against six 62-bit ones, in 10 interleaved
/// rounds: 1.0× now, 16× in 0.28.0, bound 4.  Every radicand is multiplied
/// by a distinct square — a new memo key with the same cofactor — so every
/// run is cold.
#[test]
fn an_84_bit_radicand_costs_a_bounded_multiple_of_a_62_bit_one() {
    const BLOCK: u32 = 6;
    let ctx = Context::new();
    let pq = big("2199023267911") * big("4398046512107"); // sympy.isprime: both True; 84 bits
    let ab = big("1073741827") * big("2147483659"); // nextprime(2**30) · nextprime(2**31); 62 bits
    let cold_sqrts = |n: &BigInt, squares: std::ops::Range<u32>| {
        let xs: Vec<Ex> = squares.map(|m| ctx.from_bigint(n * (m * m))).collect();
        timed(|| xs.iter().map(Ex::sqrt).count()).0
    };
    let (t84, t62_block) = interleaved_totals(
        10,
        |round| {
            let m = round as u32 + 1;
            cold_sqrts(&pq, m..m + 1)
        },
        |round| {
            let first = round as u32 * BLOCK + 1;
            cold_sqrts(&ab, first..first + BLOCK)
        },
    );
    // sympy.sqrt(4*2199023267911*4398046512107) → 2*sqrt(9671406613478110566098477)
    assert_eq!(
        format!("{}", ctx.from_bigint(&pq * 4u32).sqrt()),
        "2*sqrt(9671406613478110566098477)"
    );
    // sympy.sqrt(1073741827*2147483659) → sqrt(2305843027467304993)
    assert_eq!(
        format!("{}", ctx.from_bigint(ab).sqrt()),
        "sqrt(2305843027467304993)"
    );
    assert!(
        t84 < t62_block * 4,
        "one 84-bit radicand {t84:?} vs six 62-bit ones {t62_block:?} (0.28.0: 16×, now 1.1×)"
    );
}

/// The square part is still extracted in full while the cofactor left by
/// trial division has at most 84 bits (here 80: 1073741827² · 1048583).
#[test]
fn sqrt_pulls_a_30_bit_square_out_of_an_85_bit_integer() {
    let ctx = Context::new();
    // sympy.factorint(18134008452309089554269105)
    //   → {3: 1, 5: 1, 1048583: 1, 1073741827: 2}, so √N = 1073741827·√15728745.
    // (sympy.sqrt(N) leaves N whole: SymPy's `sqrt` only trial-divides.)
    let n = big("18134008452309089554269105");
    let x = ctx.from_bigint(n.clone());
    assert_eq!(format!("{}", x.sqrt()), "1073741827*sqrt(15728745)");
    assert_eq!(
        format!("{}", x.sqrt().simplify()),
        "1073741827*sqrt(15728745)"
    );
    // No cube, a square: N^(3/2) = 1073741827³ · 15728745 · √15728745.
    assert_eq!(
        format!("{}", x.pow(&ctx.rational(1, 3))),
        "cbrt(18134008452309089554269105)"
    );
    let three_halves = x.pow(&ctx.rational(3, 2)) / (x.clone() * x.sqrt());
    assert_eq!(three_halves.simplify(), ctx.one());
}

/// `pearson_test` on 11 pairs of 28-bit integers builds `r = s_xy /
/// √(s_xx·s_yy)` with a 36-digit radicand whose rough part
/// 56819205559·114966123911 (73 bits) was factored 45 times in 0.28.0
/// (0.62–0.78 s, debug build).  Now, factoring once and faster, it takes
/// 38 ms, about as long as 25 runs on 11 pairs of small integers (1.3 ms
/// each, nothing to factor): 1.2× against 21× in 0.28.0, bound 5, over 6
/// interleaved rounds.  Every run is on a fresh thread, so the large one is
/// cold (the radical memo is per thread).
#[test]
fn pearson_test_on_large_integers_factors_its_radicand_once() {
    const BLOCK: usize = 25;
    let x: [i64; 11] = [
        206214717, 246186485, 185595057, 97847563, 55638536, 118155337, 7720303, 207999684,
        52790158, 255094875, 230379092,
    ];
    let y: [i64; 11] = [
        189601890, 139244734, 144094144, 154115592, 81105119, 90153629, 45522501, 8688385,
        267365736, 12952309, 56400124,
    ];
    let q = |v: &[i64]| -> Vec<BigRational> {
        v.iter()
            .map(|&k| BigRational::from_integer(BigInt::from(k)))
            .collect()
    };
    let (xs, ys) = (q(&x), q(&y));
    let (small_x, small_y) = (
        q(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        q(&[3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]),
    );
    // `times` runs on one fresh thread, each on a fresh context.
    let run = |x: &[BigRational], y: &[BigRational], times: usize| {
        let (x, y) = (x.to_vec(), y.to_vec());
        on_fresh_thread(move || {
            let t = Instant::now();
            let mut last = None;
            for _ in 0..times {
                let ctx = Context::new();
                let r = pearson_test(&ctx, &x, &y, Alternative::TwoSided);
                last = Some(r.map(|r| format!("{}", r.statistic)));
            }
            (t.elapsed(), last)
        })
    };
    let mut statistic = None;
    let (t_large, t_small_block) = interleaved_totals(
        6,
        |_| {
            let (t, s) = run(&xs, &ys, 1);
            statistic = s;
            t
        },
        |_| run(&small_x, &small_y, BLOCK).0,
    );
    // With Rational means: r = sxy/sqrt(sxx*syy) →
    //   -46536503346332749*sqrt(36600984596002834250215066106942813)/36600984596002834250215066106942813
    //   (≈ -0.24324692788514152324 by sympy.N(r, 20)); sxx*syy = 585615753536045348003441057711085008/121.
    assert_eq!(
        statistic.expect("ran").expect("pearson_test"),
        "-46536503346332749/36600984596002834250215066106942813*sqrt(36600984596002834250215066106942813)"
    );
    // The small case, with Rational means: sxy/sqrt(sxx*syy) for x = 1..11,
    // y = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5] → sqrt(385)/55 (≈ 0.356753034006338).
    assert_eq!(
        run(&small_x, &small_y, 1)
            .1
            .expect("ran")
            .expect("pearson_test"),
        "1/55*sqrt(385)"
    );
    assert!(
        t_large < t_small_block * 5,
        "pearson_test on large integers {t_large:?} vs {BLOCK} on small ones {t_small_block:?} \
         (0.28.0: 21×, now 1.15×)"
    );
}

// ── panics: removed where an error path exists, documented otherwise ───────

/// `RandomVariable::transform` returns a `Result` but panicked on an
/// expression from another context (`checked_id`) and on an empty name
/// (`Context::symbol`); both are `InvalidArgument` errors now, as is an
/// event from another context in `probability` / `given` / `event_region`.
#[test]
fn random_variable_transform_reports_bad_input_as_errors() {
    use symplex::stats::{Distribution, RandomVariable};
    let ctx = Context::new();
    let z = RandomVariable::new(&ctx, "Z", Distribution::normal(ctx.int(0), ctx.int(1)));
    let other = Context::new();
    let foreign = other.symbol("Z") * 2;
    let invalid = |r: Result<RandomVariable, SymplexError>| {
        matches!(r, Err(SymplexError::InvalidArgument { .. }))
    };
    assert!(invalid(z.transform("Y", &foreign)));
    assert!(invalid(z.transform("", &(z.symbol() * 2))));
    let event = other.symbol("Z").gt(&other.int(0));
    assert!(matches!(
        z.probability(&event),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(z.given(&event).is_err());
    // The valid call is unchanged: Y = 2Z + 1 has mean 1, variance 4.
    let y = z
        .transform("Y", &(2 * z.symbol() + 1))
        .expect("affine transform");
    assert_eq!((y.mean(), y.variance()), (ctx.int(1), ctx.int(4)));
}

/// `symbol_with` reports an empty name and contradictory assumptions
/// (it panicked on both before 0.29).
#[test]
fn symbol_with_reports_an_empty_name_and_contradictions() {
    let ctx = Context::new();
    let t = ctx
        .symbol_with("t", &[Assumption::Positive])
        .expect("consistent");
    assert_eq!(
        t.is_real(),
        Some(true),
        "a declared sign makes a symbol finite, hence real"
    );
    for bad in [
        vec![Assumption::Positive, Assumption::Negative],
        vec![Assumption::Integer, Assumption::Irrational],
        vec![Assumption::Positive, Assumption::Zero],
    ] {
        assert!(matches!(
            ctx.symbol_with("u", &bad),
            Err(SymplexError::ContradictoryAssumptions { .. })
        ));
    }
    assert!(matches!(
        ctx.symbol_with("", &[Assumption::Real]),
        Err(SymplexError::InvalidArgument { .. })
    ));
    // A refused declaration leaves the name free for a consistent one.
    let u = ctx
        .symbol_with("u", &[Assumption::Negative])
        .expect("consistent");
    assert_eq!(u.is_negative(), Some(true));
}

/// `assume` reports an assumption that contradicts the declared ones and
/// leaves the symbol alone (it panicked before 0.29).
#[test]
fn assume_reports_a_contradiction_and_keeps_the_symbol() {
    let ctx = Context::new();
    let t = ctx
        .symbol("t")
        .assume(Assumption::Positive)
        .expect("consistent");
    assert!(matches!(
        t.clone().assume(Assumption::Negative),
        Err(SymplexError::ContradictoryAssumptions { .. })
    ));
    assert_eq!(t.is_positive(), Some(true));
    // Not a symbol: ignored.
    let e = (&t + 1).assume(Assumption::Negative).expect("ignored");
    assert_eq!(e, &t + 1);
}

#[test]
#[should_panic(expected = "different context")]
fn replace_with_a_foreign_expression_panics_as_documented() {
    let (ctx, other) = (Context::new(), Context::new());
    let x = ctx.symbol("x");
    let y = other.symbol("y");
    let _ = x.powi(2).replace(|e| (e == x).then(|| y.clone()));
}

#[test]
#[should_panic(expected = "incompatible variable counts")]
fn s_polynomial_of_different_rings_panics_as_documented() {
    use symplex::multipoly::{GrevLex, MultiPoly, s_polynomial};
    let f = MultiPoly::<GrevLex>::var(2, 0);
    let g = MultiPoly::<GrevLex>::var(3, 1);
    let _ = s_polynomial(&f, &g);
}

// ── powers whose result is refused by the digit guard ─────────────────────

/// `(999999999999999/10¹⁵)¹⁰⁰⁰` exceeds `max_result_digits`, so it stays a
/// power.  Before 0.29 every construction of it computed two 50,000-bit
/// integers, their gcd and their decimal strings (0.55 s, debug build) only
/// to reject the result; the guard now refuses from a lower bound on the
/// digit count.  Compared with building the same power at exponent 10,
/// which is evaluated: interleaved, so machine load affects both sides.
#[test]
fn a_power_refused_by_the_digit_guard_is_not_computed() {
    let ctx = Context::new();
    let q = ctx.rational(999_999_999_999_999, 1_000_000_000_000_000);
    let (refused, evaluated) =
        interleaved_totals(5, |_| timed(|| q.powi(1000)).0, |_| timed(|| q.powi(10)).0);
    let kept = q.powi(1000);
    assert!(
        kept.as_rational().is_none(),
        "the power stays unevaluated: {kept}"
    );
    assert!(q.powi(10).as_rational().is_some());
    // 0.28: refused/evaluated ≈ 10⁴; now ≈ 1.
    assert!(
        refused < evaluated * 50,
        "refused power {refused:?} vs evaluated power {evaluated:?}"
    );
}

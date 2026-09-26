//! Smaller known bugs: `Distribution::isf_f64`, the studentized range at
//! `df = ∞`, `tukey_hsd` at many groups, `DawidSkenePriors` at huge sizes,
//! `spearman_brown` at its pole and `kappa_test` at a degenerate null, and
//! the parser round trip of undefined functions.

use symplex::prelude::*;
use symplex::stats::Distribution;
use symplex::stats::estimation::credible_interval;
use symplex::stats::order::order_statistic;

/// `|x/reference − 1| ≤ tol`.
fn assert_rel(x: f64, reference: f64, tol: f64, what: &str) {
    assert!(
        (x / reference - 1.0).abs() <= tol,
        "{what}: {x:e} vs {reference:e}"
    );
}

const Q53: f64 = f64::EPSILON / 2.0; // 2⁻⁵³
const Q54: f64 = f64::EPSILON / 4.0; // 2⁻⁵⁴
const SUB: f64 = 5e-324; // 2⁻¹⁰⁷⁴, the least subnormal

/// Before: there was no public upper-tail quantile.  `credible_interval`
/// kept a private mirror of the `numdist` kernels and, for any other
/// discrete law, asked for the quantile at the rounded level `1 − q`: at
/// `c = 1 − 2⁻⁵³` (`q = 2⁻⁵⁴`, `1 − q` rounds to 1) a table, a die, a
/// geometric, negative-binomial or hypergeometric law was an error.
/// `Distribution::isf_f64(q)` takes the level itself on every route.
#[test]
fn isf_f64_takes_the_upper_level_itself() {
    let ctx = Context::new();
    // The kernels' isf.  scipy: stats.t.isf(1e-12, 5) = 393.9569595776037;
    //   stats.norm.isf(1e-300) = 37.0470962993612;
    //   stats.norm.isf(5e-324) = 38.467405617144344
    let t5 = Distribution::student_t(ctx.int(5));
    assert_rel(
        t5.isf_f64(1e-12).unwrap(),
        393.956_959_577_603_7,
        1e-14,
        "t5",
    );
    let n = Distribution::normal(ctx.int(1), ctx.int(2));
    assert_rel(
        n.isf_f64(1e-300).unwrap(),
        1.0 + 2.0 * 37.047_096_299_361_2,
        1e-15,
        "N",
    );
    assert_rel(
        n.isf_f64(SUB).unwrap(),
        1.0 + 2.0 * 38.467_405_617_144_344,
        1e-15,
        "N sub",
    );
    assert_eq!(n.isf_f64(0.5).unwrap(), 1.0, "N median");
    // mpmath: the smallest k with gammainc(k+1, 0, 7/3, regularized=True)
    //   <= q: 210 at q = 2**-1074 (scipy's poisson.isf returns nan there)
    let pois = Distribution::poisson(ctx.rational(7, 3));
    assert_eq!(pois.isf_f64(SUB).unwrap(), 210.0);

    // A closed quantile at the exact 1 − q.  scipy: stats.expon.isf(0.5) =
    //   0.6931471805599453 (= ln 2, the double `LN_2`);
    //   stats.expon.isf(5e-324) = 744.4400719213812
    let e1 = Distribution::exponential(ctx.int(1));
    assert_rel(
        e1.isf_f64(0.5).unwrap(),
        std::f64::consts::LN_2,
        1e-15,
        "Exp q=1/2",
    );
    assert_rel(
        e1.isf_f64(SUB).unwrap(),
        744.440_071_921_381_2,
        1e-15,
        "Exp sub",
    );

    // Densities solved on their upper tail.  mpmath (dps 50-80): findroot of
    //   log(S(x)) - log(q) with S(x) = exp(-x)/4 + 3*exp(-3*x)/4 (mixture),
    //   -expm1(5*log1p(-exp(-x))) (maximum of five Exp(1)), exp(-5x) (minimum):
    //   mixture  q = 2**-53: 35.35050620855721078, q = 2**-1074: 743.05377756026137170,
    //            q = 1/2: 0.29113435754190506612
    //   max of 5 q = 2**-53: 38.346238482111201729, q = 2**-1074: 746.04950983381536269,
    //            q = 1/2: 2.0444649242511776791
    //   min of 5 q = 2**-1074: log(2**1074)/5 = 148.88801438427625246
    let e3 = Distribution::exponential(ctx.int(3));
    let mix = Distribution::mixture(&[(ctx.rational(1, 4), e1.clone()), (ctx.rational(3, 4), e3)])
        .unwrap();
    assert_rel(
        mix.isf_f64(Q53).unwrap(),
        35.350_506_208_557_21,
        1e-14,
        "mixture 2^-53",
    );
    assert_rel(
        mix.isf_f64(SUB).unwrap(),
        743.053_777_560_261_4,
        1e-14,
        "mixture sub",
    );
    assert_rel(
        mix.isf_f64(0.5).unwrap(),
        0.291_134_357_541_905_1,
        1e-14,
        "mixture 1/2",
    );
    let max5 = order_statistic(&e1, 5, 5).unwrap();
    assert_rel(
        max5.isf_f64(Q53).unwrap(),
        38.346_238_482_111_2,
        1e-14,
        "max5 2^-53",
    );
    assert_rel(
        max5.isf_f64(SUB).unwrap(),
        746.049_509_833_815_4,
        1e-14,
        "max5 sub",
    );
    assert_rel(
        max5.isf_f64(0.5).unwrap(),
        2.044_464_924_251_178,
        1e-14,
        "max5 1/2",
    );
    let min5 = order_statistic(&e1, 5, 1).unwrap();
    assert_rel(
        min5.isf_f64(SUB).unwrap(),
        148.888_014_384_276_25,
        1e-14,
        "min5 sub",
    );

    // Affine maps: `aX + b` takes the inner isf (a > 0) or ppf (a < 0) at q.
    //   scipy: 2*stats.t.isf(2.0**-53, 3) + 1 = 429906.9961251591;
    //   -3*stats.gamma.ppf(2.0**-53, 2.5) = -2.013334396711627e-06
    //   mpmath: -3*gamma quantile at 2**-1074 (findroot of
    //   log(gammainc(5/2, 0, x, regularized=True)) - log(q)) = -2.3081583541743855e-129
    let t3 = Distribution::student_t(ctx.int(3))
        .affine(ctx.int(2), ctx.int(1))
        .unwrap();
    assert_rel(
        t3.isf_f64(Q53).unwrap(),
        429_906.996_125_159_1,
        1e-14,
        "2 T3 + 1",
    );
    let g = Distribution::gamma(ctx.rational(5, 2), ctx.int(1))
        .affine(ctx.int(-3), ctx.int(0))
        .unwrap();
    assert_rel(
        g.isf_f64(Q53).unwrap(),
        -2.013_334_396_711_627e-6,
        1e-14,
        "-3 Gamma",
    );
    // (The kernel inverts in logarithms at a subnormal level: ln q = -744.44
    // carries 1.1e-13 absolute, 4.5e-14 relative in x after the 1/2.5 power;
    // scipy's -2.308158354174363e-129 is as far.)
    assert_rel(
        g.isf_f64(SUB).unwrap(),
        -2.308_158_354_174_385_5e-129,
        1e-13,
        "-3 Gamma sub",
    );
    // ... and so does the quantile of a decreasing map (it used the closed
    // or searched route below p = 1/2).  mpmath (v28_quantiles): the
    // Gamma(5/2) upper quantile at 2**-53, times 3 = 126.29254835478196827
    assert_rel(
        g.quantile_f64(Q53).unwrap(),
        -126.292_548_354_781_97,
        1e-14,
        "-3 Gamma low",
    );

    // Lattices, decided exactly on the upper tail.  mpmath: the smallest k
    //   with (2/3)**k <= q is 91 at q = 2**-53, 93 at 2**-54, 1837 at
    //   2**-1074, 2 at 1/2 (scipy: stats.geom.isf(2**-53, 1/3) = 90.0,
    //   geom.isf(2**-54, 1/3) = inf, geom.isf(0.5, 1/3) = 2.0)
    let geo = Distribution::geometric(ctx.rational(1, 3));
    assert_eq!(geo.isf_f64(Q53).unwrap(), 91.0);
    assert_eq!(geo.isf_f64(Q54).unwrap(), 93.0);
    assert_eq!(geo.isf_f64(SUB).unwrap(), 1837.0);
    assert_eq!(geo.isf_f64(0.5).unwrap(), 2.0);
    // mpmath: smallest k with betainc(k+1, 3, 0, 3/5, regularized=True) <= q:
    //   85 at 2**-54, 1480 at 2**-1074 (scipy: stats.nbinom.isf(0.5, 3, 0.4) = 4.0)
    let nb = Distribution::negative_binomial(ctx.int(3), ctx.rational(2, 5));
    assert_eq!(nb.isf_f64(Q54).unwrap(), 85.0);
    assert_eq!(nb.isf_f64(SUB).unwrap(), 1480.0);
    assert_eq!(nb.isf_f64(0.5).unwrap(), 4.0);
    // scipy: stats.hypergeom.isf(0.5, 20, 7, 12) = 4.0; the top atom 7 has
    //   no mass above it
    let hyp = Distribution::hypergeometric(ctx.int(20), ctx.int(7), ctx.int(12));
    assert_eq!(hyp.isf_f64(0.5).unwrap(), 4.0);
    assert_eq!(hyp.isf_f64(Q54).unwrap(), 7.0);
    // scipy: stats.randint(1, 7).isf(0.25) = 5.0, .isf(0.5) = 3.0
    let die = Distribution::die(ctx.int(6));
    assert_eq!(die.isf_f64(0.25).unwrap(), 5.0);
    assert_eq!(die.isf_f64(0.5).unwrap(), 3.0);
    assert_eq!(die.isf_f64(Q54).unwrap(), 6.0);

    // A finite table whose top tail 2⁻⁶⁰ lies below 2⁻⁵⁴ (exact
    // arithmetic): S(0) = 1/2, S(1) = 2⁻⁶⁰, S(2) = 0.
    let tiny = ctx.int(2).pow(&ctx.int(-60));
    let table = Distribution::finite(
        &ctx,
        vec![
            (ctx.int(0), ctx.rational(1, 2)),
            (ctx.int(1), ctx.rational(1, 2) - &tiny),
            (ctx.int(2), tiny.clone()),
        ],
    );
    assert_eq!(table.isf_f64(Q54).unwrap(), 1.0);
    assert_eq!(table.isf_f64(1e-300).unwrap(), 2.0);
    assert_eq!(table.isf_f64(SUB).unwrap(), 2.0);
    assert_eq!(table.isf_f64(0.5).unwrap(), 0.0);
    assert!(
        table.quantile_f64(1.0 - Q54).is_err(),
        "1 - 2^-54 rounds to 1"
    );
    let ci = credible_interval(&table, 1.0 - Q53).unwrap();
    assert_eq!((ci.lower, ci.upper), (0.0, 1.0));
    let ci = credible_interval(&die, 1.0 - Q53).unwrap();
    assert_eq!((ci.lower, ci.upper), (1.0, 6.0));

    // S(k) ≤ q and F(k) ≥ 1 − q are one condition where 1 − q is exact.
    for d in [&geo, &nb, &hyp, &die, &table, &pois] {
        for (q, p) in [(0.75, 0.25), (0.25, 0.75), (0.5, 0.5)] {
            assert_eq!(
                d.isf_f64(q).unwrap(),
                d.quantile_f64(p).unwrap(),
                "{d}: {q}"
            );
        }
    }

    // The level must lie in (0, 1).
    for bad in [0.0, 1.0, -0.5, f64::NAN] {
        let err = e1.isf_f64(bad).unwrap_err().to_string();
        assert!(err.contains("isf_f64"), "{err}");
    }
}

/// Before: `studentized_range_{cdf,sf,quantile}(…, df = f64::INFINITY)` was
/// an `InvalidArgument` ("the degrees of freedom must be finite"), while
/// scipy's `studentized_range` accepts `np.inf` and the quadrature already
/// took every `df ≥ 1e30` as the infinite-df limit, the range of `k`
/// standard normals.
#[test]
fn studentized_range_accepts_infinite_degrees_of_freedom() {
    use symplex::stats::anova::{
        studentized_range_cdf, studentized_range_quantile, studentized_range_sf,
    };
    let inf = f64::INFINITY;
    // mpmath (dps 40): k*quad(lambda z: npdf(z)*(ncdf(z) - ncdf(z - q))**(k-1), [-inf, ..., inf])
    //   for the cdf, and k*quad(npdf(z)*(ncdf(z)**(k-1) - (ncdf(z) - ncdf(z - q))**(k-1)))
    //   for the sf (z the maximum):
    //   q = 3,   k = 3:  cdf 0.91445742834504199696, sf 0.085542571654958003039
    //   q = 0.5, k = 3:  cdf 0.066578054864713036275
    //   q = 8,   k = 5:  sf 1.5380313803521567617e-7
    // scipy: studentized_range.cdf(3.0, 3, np.inf) = 0.9144574283450421,
    //   .sf(8.0, 5, np.inf) = 1.5380313811430568e-07 (1 - cdf, 5e-10 off)
    assert_rel(
        studentized_range_cdf(3.0, 3, inf).unwrap(),
        0.914_457_428_345_042,
        1e-14,
        "cdf",
    );
    assert_rel(
        studentized_range_sf(3.0, 3, inf).unwrap(),
        0.085_542_571_654_958,
        1e-13,
        "sf",
    );
    assert_rel(
        studentized_range_cdf(0.5, 3, inf).unwrap(),
        0.066_578_054_864_713_03,
        1e-13,
        "cdf 0.5",
    );
    assert_rel(
        studentized_range_sf(8.0, 5, inf).unwrap(),
        1.538_031_380_352_156_8e-7,
        1e-13,
        "sf 8",
    );
    // mpmath: findroot of the above against the level (sf for p > 1/2):
    //   ppf(0.95, 3) = 3.3144931553981204871, ppf(0.99, 5) = 4.6028210422015743997,
    //   ppf(0.05, 4) = 0.75953319190440863561
    //   (scipy: studentized_range.ppf(0.95, 3, np.inf) = 3.314493155398121)
    assert_rel(
        studentized_range_quantile(0.95, 3, inf).unwrap(),
        3.314_493_155_398_120_5,
        1e-13,
        "ppf .95",
    );
    assert_rel(
        studentized_range_quantile(0.99, 5, inf).unwrap(),
        4.602_821_042_201_574,
        1e-13,
        "ppf .99",
    );
    assert_rel(
        studentized_range_quantile(0.05, 4, inf).unwrap(),
        0.759_533_191_904_408_6,
        1e-13,
        "ppf .05",
    );
    // The same route as any df ≥ 1e30.
    assert_eq!(
        studentized_range_sf(3.0, 3, inf).unwrap(),
        studentized_range_sf(3.0, 3, 1e30).unwrap()
    );
    // NaN and −∞ are still refused.
    for bad in [f64::NAN, f64::NEG_INFINITY] {
        assert!(matches!(
            studentized_range_cdf(3.0, 3, bad),
            Err(SymplexError::InvalidArgument { .. })
        ));
    }
}

/// Thirty groups of 6–9 observations with drifting means (`N = 223`,
/// `df = 193`); the same data as in the oracle scripts quoted below.
fn thirty_groups() -> Vec<Vec<symplex::stats::data::Q>> {
    (0..30i64)
        .map(|g| {
            let n = 6 + g % 4;
            let v: Vec<i64> = (0..n)
                .map(|i| g * 3 / 2 + (i * 7919 + g * 104_729) % 11 - 5)
                .collect();
            symplex::stats::data::from_i64(&v)
        })
        .collect()
}

/// Before: `tukey_hsd` integrated the studentized range tail afresh for
/// every pair (about 4 ms each, release): 30 groups, 435 pairs, took 1.8 s
/// in release.  The inner integral, the range tail `P(W > w)` of `k`
/// normals, depends on `k` alone; it is now tabulated once per call and
/// shared by every pair and by the critical value (30 ms in release), and
/// the p-values are those of the per-pair integral to its own accuracy.
#[test]
fn tukey_hsd_shares_one_range_tail_across_its_pairs() {
    use std::time::{Duration, Instant};
    use symplex::stats::anova::{studentized_range_quantile, studentized_range_sf, tukey_hsd};
    let ctx = Context::new();
    let groups = thirty_groups();
    let pairs = tukey_hsd(&ctx, &groups, 0.95).unwrap();
    assert_eq!(pairs.len(), 435);
    // The p-value of each pair against the per-pair integral (0.28's
    // p_adj), from p ~ 1 to p ~ 1e-55.
    let stat = |p: &symplex::stats::anova::PairwiseComparison| {
        (&p.statistic * &p.statistic).eval_f64().unwrap().sqrt()
    };
    for p in pairs.iter().step_by(29) {
        let direct = studentized_range_sf(stat(p), 30, 193.0).unwrap();
        assert_rel(p.p_adj, direct, 1e-14, &format!("pair ({}, {})", p.i, p.j));
    }
    let far = pairs.iter().find(|p| (p.i, p.j) == (0, 29)).unwrap();
    assert_rel(
        far.p_adj,
        studentized_range_sf(stat(far), 30, 193.0).unwrap(),
        1e-14,
        "(0, 29)",
    );
    // scipy: groups as in `thirty_groups`; r = stats.tukey_hsd(*groups);
    //   r.pvalue[18, 25] = 4.3305665692217055e-05 (1 - cdf: a relative 1e-10
    //   of rounding noise at this size), r.pvalue[0, 29] = 0.0 (ours 1.1e-55);
    //   r.confidence_interval(0.95).low[18, 25] = -15.321499917399835,
    //   .high[18, 25] = -2.8213572254573034
    let p = pairs.iter().find(|p| (p.i, p.j) == (18, 25)).unwrap();
    assert_rel(p.p_adj, 4.330_566_569_221_705_5e-5, 1e-8, "scipy (18, 25)");
    // mpmath (dps 20): q = |x̄₁₈ − x̄₂₅|/se = 7.8136919159952495337 exactly from
    //   the data; quad over s of the χ₁₉₃ density times k*quad(npdf(z)*(ncdf(z)**29
    //   - (ncdf(z) - ncdf(z - q*s))**29)) = 4.3305665731818712429e-5
    assert_rel(p.p_adj, 4.330_566_573_181_871e-5, 1e-14, "mpmath (18, 25)");
    assert_rel(p.ci.lower, -15.321_499_917_399_835, 1e-12, "ci low");
    assert_rel(p.ci.upper, -2.821_357_225_457_303_4, 1e-12, "ci high");
    // The critical value is the studentized range quantile itself.
    let q_crit = studentized_range_quantile(0.95, 30, 193.0).unwrap();
    let se = p.se.eval_f64().unwrap();
    let d = (p.ci.upper + p.ci.lower) / 2.0;
    assert_rel(p.ci.upper - d, q_crit * se, 1e-13, "half-width");

    // Load-robust timing (see v28_perf): twelve groups (66 pairs) against six
    // single tail integrals at the same k, in interleaved rounds.  0.28: each
    // pair was one such integral, plus about fifteen for the critical value
    // (~13x); now the table costs a few integrals.
    let twelve: Vec<_> = groups[..12].to_vec();
    let (mut t_tukey, mut t_single) = (Duration::ZERO, Duration::ZERO);
    for round in 0..3 {
        let t = Instant::now();
        let r = tukey_hsd(&ctx, &twelve, 0.95).unwrap();
        t_tukey += t.elapsed();
        assert_eq!(r.len(), 66);
        let t = Instant::now();
        for j in 0..6 {
            let q = 1.0 + 0.7 * j as f64 + 0.1 * round as f64;
            studentized_range_sf(q, 12, 78.0).unwrap();
        }
        t_single += t.elapsed();
    }
    eprintln!("tukey(12 groups) {t_tukey:?} vs 6 single tails {t_single:?}");
    assert!(
        t_tukey < t_single * 4,
        "tukey_hsd of 12 groups {t_tukey:?} vs six single tails {t_single:?} (0.28: ~13x)"
    );
}

/// Before: `studentized_range_sf(1.0, 12, 1.0)` was `NaN`, as were other
/// upper tails at `k = 12` (`q = e⁻²⁰, df = 2`), and at `q = e^{−34.9}` the
/// quadrature found no window, so `studentized_range_quantile(0.95, 12, 1)`
/// failed: the range tail's ratio `r = Φ̄(z + w)/Φ̄(z)` of two rounded tails
/// came out `1 + ε` at some node, and `ln1p(−r)` was NaN.  (Found while
/// checking the `tukey_hsd` table against the per-pair integral.)
#[test]
fn studentized_range_upper_tail_survives_a_rounded_tail_ratio() {
    use symplex::stats::anova::{
        studentized_range_cdf, studentized_range_quantile, studentized_range_sf,
    };
    // scipy: studentized_range.sf(1.0, 12, 1) = 0.992072121050315,
    //   .cdf(1.0, 12, 1) = 0.007927878949685057,
    //   .ppf(0.95, 12, 1) = 51.95735217609894
    let sf = studentized_range_sf(1.0, 12, 1.0).unwrap();
    assert_rel(sf, 0.992_072_121_050_315, 1e-14, "sf");
    let cdf = studentized_range_cdf(1.0, 12, 1.0).unwrap();
    assert_rel(cdf, 0.007_927_878_949_685_057, 1e-12, "cdf");
    assert_rel(
        studentized_range_quantile(0.95, 12, 1.0).unwrap(),
        51.957_352_176_098_94,
        1e-13,
        "ppf",
    );
    for (q, df) in [
        ((-20.0f64).exp(), 2.0),
        ((-34.898f64).exp(), 1.0),
        ((-34.898f64).exp(), 10.0),
    ] {
        let s = studentized_range_sf(q, 12, df).unwrap();
        assert!(
            (s - 1.0).abs() <= 2.0 * f64::EPSILON,
            "sf({q:e}, 12, {df}) = {s}"
        );
    }
}

/// Before: `DawidSkenePriors::symmetric(1 << 61, 1.0, 1.0, 1.0)` panicked
/// (capacity overflow in `vec!`), and there was no checked constructor.
/// The fields are the materialised vectors (read and built by callers), so
/// the allocation cannot be made lazy: `try_symmetric` checks the size (and
/// the α's), and `symmetric` documents its panic.
#[test]
fn dawid_skene_priors_have_a_checked_constructor() {
    use symplex::stats::aggregation::DawidSkenePriors;
    for k in [1usize << 61, 1 << 40, usize::MAX] {
        assert!(
            matches!(
                DawidSkenePriors::try_symmetric(k, 1.0, 1.0, 1.0),
                Err(SymplexError::InvalidArgument { .. })
            ),
            "{k} categories"
        );
    }
    for bad in [0.5, f64::NAN, f64::INFINITY] {
        assert!(DawidSkenePriors::try_symmetric(3, bad, 2.0, 1.0).is_err());
        assert!(DawidSkenePriors::try_symmetric(3, 1.0, bad, 1.0).is_err());
        assert!(DawidSkenePriors::try_symmetric(3, 1.0, 2.0, bad).is_err());
    }
    let p = DawidSkenePriors::try_symmetric(3, 2.0, 5.0, 1.5).unwrap();
    assert_eq!(p, DawidSkenePriors::symmetric(3, 2.0, 5.0, 1.5));
    assert_eq!(p.class_prior_alpha, vec![2.0; 3]);
    assert_eq!(p.confusion_alpha[2], vec![1.5, 1.5, 5.0]);
    assert_eq!(
        DawidSkenePriors::try_symmetric(0, 1.0, 1.0, 1.0).unwrap(),
        DawidSkenePriors::symmetric(0, 1.0, 1.0, 1.0)
    );
}

/// Before: `spearman_brown(&ctx.int(-1), 2)` returned `zoo`, the pole of
/// `kρ/(1 + (k − 1)ρ)`, and so did `split_half` for halves with `r = −1`
/// and `standardized_alpha` for two reversed items.  The lengthened test
/// (`k` parallel parts, pairwise correlation `ρ`) has variance
/// `kσ²(1 + (k − 1)ρ)`: zero at the pole, impossible below it.
/// `try_spearman_brown` refuses that and a `ρ` outside `[−1, 1]` or `k = 0`
/// (neither statsmodels nor pingouin has the formula); `spearman_brown`
/// returns `nan` there; the two scale functions report the error.
#[test]
fn spearman_brown_refuses_its_pole() {
    use symplex::stats::agreement::RatingTable;
    use symplex::stats::reliability::{
        SplitHalf, spearman_brown, split_half, standardized_alpha, try_spearman_brown,
    };
    let ctx = Context::new();
    let invalid = |r: Result<Ex, SymplexError>| {
        assert!(
            matches!(r, Err(SymplexError::InvalidArgument { .. })),
            "{r:?}"
        );
    };
    // The pole (k = 2: ρ = −1; k = 3: ρ = −1/2), beyond it, outside [−1, 1], k = 0.
    for (r, k) in [
        (ctx.int(-1), 2),
        (ctx.rational(-1, 2), 3),
        (ctx.rational(-3, 4), 3),
        (ctx.rational(3, 2), 2),
        (ctx.rational(-3, 2), 1),
        (ctx.rational(3, 5), 0),
    ] {
        invalid(try_spearman_brown(&r, k));
        assert_eq!(spearman_brown(&r, k), ctx.nan(), "{r}, {k}");
    }
    // Inside the domain (exact arithmetic): 2·(3/5)/(1 + 3/5) = 3/4;
    // 3·(−1/3)/(1 − 2/3) = −3 (−1/3 lies above the pole −1/2 of k = 3);
    // 5·(1/3)/(1 + 4/3) = 5/7.
    assert_eq!(
        try_spearman_brown(&ctx.rational(3, 5), 2).unwrap(),
        ctx.rational(3, 4)
    );
    assert_eq!(
        try_spearman_brown(&ctx.rational(-1, 3), 3).unwrap(),
        ctx.int(-3)
    );
    assert_eq!(
        try_spearman_brown(&ctx.rational(1, 3), 5).unwrap(),
        ctx.rational(5, 7)
    );
    assert_eq!(try_spearman_brown(&ctx.int(-1), 1).unwrap(), ctx.int(-1));
    assert_eq!(spearman_brown(&ctx.rational(3, 5), 2), ctx.rational(3, 4));
    // A symbolic ρ gets the formula; an assumption can refute the domain.
    let rho = ctx.symbol("rho");
    assert!(try_spearman_brown(&rho, 2).is_ok());
    let big = ctx.symbol_with("big", &[Assumption::Positive]).unwrap();
    invalid(try_spearman_brown(&(ctx.int(1) + &big), 2));

    // Two items that are exact reversals: the halves (and the items)
    // correlate at −1 and their sum is constant.
    let t = RatingTable::from_i64(&[&[1, 3], &[2, 2], &[3, 1]]).unwrap();
    invalid(split_half(&ctx, &t, &SplitHalf::OddEven));
    invalid(standardized_alpha(&ctx, &t));
}

/// `kappa_test` at a zero null variance: kept an error, now documented.
/// `Var₀ = 0` exactly when one rater uses a single category or the raters
/// share none (checked exhaustively on all 2×2 and 3×3 tables with cells
/// below 3 and 4×4 with cells below 2, exact arithmetic); `p_o = p_e` for
/// every table with those margins, so `κ̂ = 0` identically and `z = 0/0`.
/// statsmodels: `cohens_kappa([[5, 3], [0, 0]], return_results=True)` gives
/// `var_kappa0 = 0.0`, `z_value = nan`, `pvalue_two_sided = nan`; its
/// `z = 0, p = 1` appears only where rounding leaves `var_kappa0 = 3.3e-16`
/// (`[[4, 0, 2], [0, 0, 0], [0, 0, 0]]`).
#[test]
fn kappa_test_refuses_a_degenerate_null() {
    use num_traits::Zero;
    use symplex::stats::agreement::{Weights, kappa_from_confusion, kappa_test_from_confusion};
    use symplex::stats::hypothesis::Alternative;
    let ctx = Context::new();
    for table in [
        vec![vec![5, 3], vec![0, 0]],
        vec![vec![0, 4], vec![0, 0]],
        vec![vec![4, 0, 2], vec![0, 0, 0], vec![0, 0, 0]],
        vec![vec![0, 2, 1], vec![0, 0, 0], vec![0, 0, 0]],
    ] {
        let r = kappa_test_from_confusion(&ctx, &table, Alternative::TwoSided);
        let Err(SymplexError::InvalidArgument { reason, .. }) = r else {
            panic!("{table:?}: {r:?}");
        };
        assert!(reason.contains("null variance"), "{reason}");
        // κ itself is 0 (statsmodels: kappa 0.0 for the first and third).
        let k = kappa_from_confusion(&table, &Weights::Unweighted).unwrap();
        assert!(k.kappa.is_zero(), "{table:?}: {}", k.kappa);
    }
}

/// Before: the Display of an expression with an undefined function did
/// not parse back — `ctx.parse("f(x)")` was "unknown function 'f'" even
/// in the context that built `f(x)`, and `Derivative(…)` (every
/// `diff`/`formal_diff`/ODE display) was an unknown 2-argument function.
/// A name the context already knows as an undefined function (an `f(…)`
/// built by `Context::apply` or present in any of its expressions) now
/// parses as that function in any arity, and `Derivative(f, x)` as the
/// unevaluated derivative; an unknown name (`sni(x)`, or `f` in a fresh
/// context) is still an error.  SymPy: `sympify("f(x)")` is
/// `Function('f')(x)` (type `UndefinedFunction`), `sympify("sni(x)")` an
/// undefined `sni` too, `sympify("Derivative(f(x), x)")` stays a
/// `Derivative` and `sympify("Subs(Derivative(f(x), x), x, 0)")` a `Subs`.
#[test]
fn display_of_undefined_functions_parses_back() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // A fresh context does not know `f`.
    assert!(ctx.parse("f(x)").is_err());
    let f = ctx.apply("f", &[&x]).unwrap();
    let fxy = ctx.apply("f", &[&x, &y]).unwrap();
    let g = ctx.apply("g", &[x.powi(2)]).unwrap();
    let round_trip = |e: &Ex, shown: &str| {
        let s = e.to_string();
        assert_eq!(s, shown);
        assert_eq!(&ctx.parse(&s).unwrap(), e, "{s}");
    };
    round_trip(&f, "f(x)");
    round_trip(&fxy, "f(x, y)");
    round_trip(&f.diff(&x), "Derivative(f(x), x)");
    round_trip(
        &f.diff(&x).subs_i64(&x, 0),
        "Subs(Derivative(f(x), x), x, 0)",
    );
    round_trip(&g.diff(&x), "2*x*Subs(Derivative(g(_xi), _xi), _xi, x^2)");
    round_trip(&fxy.diff(&y), "Derivative(f(x, y), y)");
    round_trip(&(f.powi(2) + fxy.sin()), "f(x)^2 + sin(f(x, y))");
    // An ODE written with a formal derivative (the dsolve input form).
    round_trip(
        &(&y.formal_diff(&x).formal_diff(&x) + &y),
        "y + Derivative(Derivative(y, x), x)",
    );
    // Any arity, the name's own case.
    assert_eq!(ctx.parse("f(1, 2, 3)").unwrap().to_string(), "f(1, 2, 3)");
    let big_f = ctx.apply("F", &[&x]).unwrap();
    assert_eq!(ctx.parse("F(y)").unwrap(), ctx.apply("F", &[&y]).unwrap());
    assert_eq!(ctx.parse("F(x)").unwrap(), big_f);
    assert_ne!(big_f, f);
    // Still errors: a misspelt name, a variable that was never applied, a
    // library function in the wrong arity, a non-symbol derivative variable.
    let _ = ctx.symbol("h");
    for bad in ["sni(x)", "h(x)", "besselj(x)", "Derivative(f(x), 2)"] {
        assert!(ctx.parse(bad).is_err(), "{bad}");
    }
    let err = ctx.parse("sni(x)").unwrap_err().to_string();
    assert!(
        err.contains("unknown function 'sni'") && err.contains("Context::apply"),
        "{err}"
    );
    // Another context does not share the declaration.
    assert!(Context::new().parse(&f.to_string()).is_err());
}

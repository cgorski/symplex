//! 0.18 track: real roots of high-degree polynomials (`root_of`,
//! `real_roots`) and clean `nroots` output.
//!
//! The degree-40/50 polynomials are the Clopper–Pearson tails
//! `Σ_{j=k}^{n} C(n,j) p^j (1−p)^{n−j} − 1/40` whose root in `(0, 1)` is
//! the exact lower confidence bound; `stats::aggregation::
//! proportion_interval_exact` names it via `Ex::root_of`.  Before 0.18
//! `root_of` returned `None` for them: the Aberth iteration behind the
//! `RootOf` index started on the Cauchy-bound circle (radius > 6·10⁷ for
//! these polynomials, whose roots all lie in `|z| < 1.5`) and had not
//! converged after its 200 iterations, so the verified index lookup
//! found no real root in the isolating interval.
//!
//! Oracle: `scipy.stats.beta.ppf(alpha/2, k, n-k+1)` with `alpha/2 = 1/40`
//! (`symplex/.venv/bin/python`, scipy 1.18.1):
//!
//! ```text
//! >>> from scipy.stats import beta
//! >>> beta.ppf(1/40, 12, 29)   # n = 40, k = 12
//! 0.1656272043932356
//! >>> beta.ppf(1/40, 15, 36)   # n = 50, k = 15
//! 0.17861784566414687
//! ```

use std::time::{Duration, Instant};

use symplex::num_bigint::BigInt;
use symplex::num_rational::Ratio;
use symplex::prelude::*;
use symplex::stats::Rng;

/// Generous bound for a debug build (measured: ≈ 1.3 s for degree 40 and
/// ≈ 2.3 s for degree 50 on a laptop).
const BUDGET: Duration = Duration::from_secs(20);

/// `Σ_{j=k}^{n} C(n,j) p^j (1−p)^{n−j} − 1/denom`, expanded, in a fresh
/// symbol `p`.
fn binomial_tail(ctx: &Context, n: usize, k: usize, denom: i64) -> (Ex, Ex) {
    let p = ctx.symbol("p");
    let one_minus_p = ctx.one() - &p;
    let tail = (k..=n).fold(ctx.zero(), |acc, j| {
        acc + ctx.from_ratio(symplex::stats::data::binomial_q(n, j))
            * p.powi(j as i64)
            * one_minus_p.powi((n - j) as i64)
    });
    let poly = (tail - ctx.rational(1, denom)).expand();
    (poly, p)
}

/// The Clopper–Pearson lower bound as `root_of`: the unique root in
/// `(0, 1)`, whose index among the ascending real roots is the number of
/// real roots below `0`.  Returns the root and the time `root_of` took.
fn tail_root(ctx: &Context, n: usize, k: usize) -> (Ex, Duration) {
    let (poly, p) = binomial_tail(ctx, n, k, 40);
    assert_eq!(poly.degree(&p), Some(n));
    assert_eq!(
        poly.count_real_roots_in(&p, &ctx.zero(), &ctx.one()),
        Some(1),
        "the tail is monotone on (0, 1)"
    );
    let below = poly
        .count_real_roots_in(&p, &ctx.neg_infinity(), &ctx.zero())
        .expect("polynomial");
    let t = Instant::now();
    let root = poly
        .root_of(&p, below)
        .expect("root_of names the root of the degree-n tail polynomial");
    (root, t.elapsed())
}

// ═══════════════════════════════════════════════════════════════════════════
// Problem 1 — root_of at degree 40 / 50
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn root_of_degree_40_binomial_tail_matches_scipy() {
    let ctx = Context::new();
    let (root, took) = tail_root(&ctx, 40, 12);
    assert!(took < BUDGET, "root_of took {took:?}");
    assert!(format!("{root}").starts_with("RootOf("), "{root}");
    // scipy.stats.beta.ppf(1/40, 12, 29)
    let v = root.eval_f64().expect("RootOf evaluates");
    assert!((v - 0.1656272043932356).abs() < 1e-12, "{v}");
}

#[test]
fn root_of_degree_50_binomial_tail_matches_scipy() {
    let ctx = Context::new();
    let (root, took) = tail_root(&ctx, 50, 15);
    assert!(took < BUDGET, "root_of took {took:?}");
    assert!(format!("{root}").starts_with("RootOf("), "{root}");
    // scipy.stats.beta.ppf(1/40, 15, 36)
    let v = root.eval_f64().expect("RootOf evaluates");
    assert!((v - 0.17861784566414687).abs() < 1e-12, "{v}");
}

#[test]
fn real_roots_and_isolation_at_degree_40() {
    let ctx = Context::new();
    let (poly, p) = binomial_tail(&ctx, 40, 12, 40);
    // Isolation to width 1/1024 used to take ~17 s here (rational Horner
    // over a Cauchy-bound interval of width > 10⁸).
    let t = Instant::now();
    let iv = poly.real_roots_isolate(&p);
    assert!(
        t.elapsed() < BUDGET,
        "real_roots_isolate took {:?}",
        t.elapsed()
    );
    assert_eq!(iv.len(), 2);
    let roots = poly.real_roots(&p).expect("all real roots are named");
    assert_eq!(roots.len(), 2);
    let lo = roots[0].eval_f64().expect("evaluates");
    let hi = roots[1].eval_f64().expect("evaluates");
    assert!(
        lo < 0.0 && (hi - 0.1656272043932356).abs() < 1e-12,
        "{lo} {hi}"
    );
    for (r, cell) in roots.iter().zip(&iv) {
        let v = r.eval_f64().expect("evaluates");
        let a = cell.lower.eval_f64().expect("rational");
        let b = cell.upper.eval_f64().expect("rational");
        assert!(a - 1e-12 <= v && v <= b + 1e-12, "{v} ∉ [{a}, {b}]");
    }
}

/// `real_roots_isolate`, `count_real_roots_in` and `root_of` agree on a
/// random degree-30 integer polynomial.
#[test]
fn isolation_counting_and_root_of_agree_on_random_degree_30() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut rng = Rng::new(0x5EED_0018);
    // Coefficients in −9..=9 with a non-zero leading coefficient; degree
    // 30 is even, so the seed is chosen to give a polynomial with real
    // roots (checked below).
    let mut f = ctx.zero();
    for d in 0..=30 {
        let mut c = rng.below(19) as i64 - 9;
        if d == 30 && c == 0 {
            c = 1;
        }
        f += ctx.int(c) * x.powi(d);
    }
    let f = f.expand();
    assert_eq!(f.degree(&x), Some(30));

    let count = f.count_real_roots(&x).expect("polynomial");
    assert!(count >= 2, "seed gives {count} real roots; pick another");
    let iv = f.real_roots_isolate(&x);
    assert_eq!(iv.len(), count);
    let roots = f.real_roots(&x).expect("all roots named");
    assert_eq!(roots.len(), count);

    for (i, cell) in iv.iter().enumerate() {
        // Exactly one root in the closed cell.
        assert_eq!(
            f.count_real_roots_in(&x, &cell.lower, &cell.upper),
            Some(1),
            "cell {i}"
        );
        // `root_of(i)` is that root.
        let r = f.root_of(&x, i).expect("root_of");
        assert_eq!(r, roots[i]);
        let v = r.eval_f64().expect("evaluates");
        let a = cell.lower.eval_f64().expect("rational");
        let b = cell.upper.eval_f64().expect("rational");
        assert!(
            a - 1e-12 <= v && v <= b + 1e-12,
            "root {i}: {v} ∉ [{a}, {b}]"
        );
        // `f` vanishes there (relative to the size of its terms).
        let fv = f
            .subs(&x, &ctx.from_f64(v).expect("finite"))
            .eval_f64()
            .expect("evaluates");
        assert!(fv.abs() < 1e-6 * v.abs().max(1.0).powi(30), "f({v}) = {fv}");
    }
    assert!(f.root_of(&x, count).is_none());
    // The count below each cell is its index.
    for (i, cell) in iv.iter().enumerate() {
        assert_eq!(
            f.count_real_roots_in(&x, &ctx.neg_infinity(), &cell.lower),
            Some(i),
            "cell {i}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Problem 2 — nroots without noise on zero components
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn nroots_x2_plus_1_is_exactly_plus_minus_i() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let roots = (&x.powi(2) + 1).nroots(&x, 10).expect("polynomial");
    assert_eq!(roots.len(), 2);
    assert_eq!(roots[0].re, 0.0, "{roots:?}");
    assert_eq!(roots[1].re, 0.0, "{roots:?}");
    assert_eq!(roots[0].im, -1.0, "{roots:?}");
    assert_eq!(roots[1].im, 1.0, "{roots:?}");
    // Higher requested precision behaves the same.
    let roots = (&x.powi(2) + 1).nroots(&x, 50).expect("polynomial");
    assert!(roots.iter().all(|z| z.re == 0.0), "{roots:?}");
}

#[test]
fn nroots_x3_minus_1_has_clean_real_and_complex_roots() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let roots = (&x.powi(3) - 1).nroots(&x, 15).expect("polynomial");
    assert_eq!(roots.len(), 3);
    // Sorted by (re, im): the conjugate pair first, then 1.
    let half_sqrt3 = 3f64.sqrt() / 2.0;
    for z in &roots[..2] {
        assert!((z.re + 0.5).abs() < 1e-15, "{z:?}");
        assert!((z.im.abs() - half_sqrt3).abs() < 1e-15, "{z:?}");
    }
    assert!(roots[0].im < 0.0 && roots[1].im > 0.0, "{roots:?}");
    assert_eq!(roots[2].re, 1.0, "{roots:?}");
    assert_eq!(roots[2].im, 0.0, "no spurious imaginary part: {roots:?}");
}

#[test]
fn nroots_keeps_genuinely_tiny_roots() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x² − 10⁻⁴⁰: roots ±10⁻²⁰, far above the noise tolerance.
    let tiny = ctx.from_ratio(Ratio::new(BigInt::from(1), BigInt::from(10).pow(40)));
    let roots = (&x.powi(2) - tiny).nroots(&x, 10).expect("polynomial");
    assert_eq!(roots.len(), 2);
    assert!((roots[0].re + 1e-20).abs() < 1e-33, "{roots:?}");
    assert!((roots[1].re - 1e-20).abs() < 1e-33, "{roots:?}");
    assert!(roots.iter().all(|z| z.im == 0.0), "{roots:?}");
    // And a tiny root off the axis keeps its real part too: (x − 10⁻²⁰)² + 1.
    let shift = ctx.from_ratio(Ratio::new(BigInt::from(1), BigInt::from(10).pow(20)));
    let roots = ((&x - shift).powi(2) + 1)
        .expand()
        .nroots(&x, 10)
        .expect("polynomial");
    assert_eq!(roots.len(), 2);
    assert!(
        roots.iter().all(|z| (z.re - 1e-20).abs() < 1e-33),
        "{roots:?}"
    );
}

#[test]
fn nroots_of_mixed_polynomial_snaps_only_noise() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x² + 1)(x² + 4)(x − 3)(x + 1/2): purely imaginary ±i, ±2i and two
    // real roots.
    let f = ((&x.powi(2) + 1) * (&x.powi(2) + 4) * (&x - 3) * (&x * 2 + 1)).expand();
    let roots = f.nroots(&x, 15).expect("polynomial");
    assert_eq!(roots.len(), 6);
    let imag: Vec<_> = roots.iter().filter(|z| z.im != 0.0).collect();
    assert_eq!(imag.len(), 4);
    assert!(imag.iter().all(|z| z.re == 0.0), "{roots:?}");
    let mut ims: Vec<f64> = imag.iter().map(|z| z.im).collect();
    ims.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    for (got, want) in ims.iter().zip([-2.0, -1.0, 1.0, 2.0]) {
        assert!((got - want).abs() < 1e-14, "{ims:?}");
    }
    let real: Vec<f64> = roots.iter().filter(|z| z.im == 0.0).map(|z| z.re).collect();
    assert_eq!(real.len(), 2);
    assert!(
        (real[0] + 0.5).abs() < 1e-14 && (real[1] - 3.0).abs() < 1e-14,
        "{real:?}"
    );
}

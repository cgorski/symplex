//! Independent verification of `symplex::stats` (0.17 audit, track c):
//! distribution machinery, wrappers, joint algebra, estimation, Markov
//! chains, information theory, multivariate normal and order statistics.
//!
//! Every numeric oracle is quoted in a comment above its assertion
//! (`scipy.stats` 1.18, `mpmath`, SymPy 1.14 or `fractions.Fraction`).

use symplex::prelude::*;
use symplex::stats::{Distribution, RandomVariable, Rng, Support};

// ── helpers ────────────────────────────────────────────────────────────────────────

fn f64_of(e: &Ex) -> f64 {
    e.eval_f64()
        .unwrap_or_else(|err| panic!("`{e}` does not evaluate: {err}"))
}

fn assert_close(actual: &Ex, expected: f64, label: &str) {
    let v = f64_of(actual);
    assert!(
        (v - expected).abs() <= 1e-9 * expected.abs().max(1.0),
        "{label}: got {v} (`{actual}`), expected {expected}"
    );
}

fn assert_f64(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= 1e-9 * expected.abs().max(1.0),
        "{label}: got {actual}, expected {expected}"
    );
}

/// `P(event)` of a variable named `X` with distribution `d`.
fn prob(ctx: &Context, d: &Distribution, event: impl Fn(&Ex) -> BoolEx) -> Ex {
    let x = RandomVariable::new(ctx, "X", d.clone());
    x.probability(&event(x.symbol()))
        .unwrap_or_else(|e| panic!("probability of an event on {d}: {e}"))
}

/// Mean of `n` samples from a fixed seed within `4σ/√n` of the exact mean,
/// and every sample inside the support.
fn check_sampling(d: &Distribution, n: usize, seed: u64) {
    let s = d
        .sample(n, &mut Rng::new(seed))
        .unwrap_or_else(|e| panic!("sampling {d}: {e}"));
    let mean = s.iter().sum::<f64>() / n as f64;
    let exact = f64_of(&d.mean());
    let sd = f64_of(&d.variance()).sqrt();
    let tol = 4.0 * sd / (n as f64).sqrt();
    assert!(
        (mean - exact).abs() <= tol,
        "{d}: sample mean {mean} vs exact {exact} (4σ/√n = {tol})"
    );
    let ctx = d.context();
    let support = d.support();
    for v in s.iter().take(200) {
        let point = ctx.from_f64(*v).expect("f64 to Ex");
        // A lattice value read back from f64 is an integer expression.
        let point = if d.kind() == symplex::stats::Kind::Discrete {
            ctx.int(*v as i64)
        } else {
            point
        };
        assert_ne!(
            support.contains(&point),
            Some(false),
            "{d}: sample {v} outside {support}"
        );
    }
}

/// `quantile_f64(cdf(x)) ≈ x` at a numeric `x` inside the support.
fn check_quantile_roundtrip(d: &Distribution, x: f64) {
    let ctx = d.context();
    let p = f64_of(&d.cdf(&ctx.from_f64(x).expect("x")));
    let back = d
        .quantile_f64(p)
        .unwrap_or_else(|e| panic!("{d}: quantile_f64({p}): {e}"));
    assert!(
        (back - x).abs() <= 1e-7 * x.abs().max(1.0),
        "{d}: quantile(cdf({x})) = {back}"
    );
}

// ── regression tests for the bugs found in this audit ────────────────────

#[test]
fn regression_affine_decreasing_lattice_cdf_at_non_integer_point() -> Result<(), SymplexError> {
    // Y = 7 − D for a fair die: P(Y ≤ 7/2) = P(D ≥ 7/2) = P(D ≥ 4) = 1/2.
    // Before the fix `Affine::cdf` used 1 − F(x − 1) with x = 7/2, i.e.
    // 1 − F(2) = 2/3.
    let ctx = Context::new();
    let y = Distribution::die(ctx.int(6)).affine(ctx.int(-1), ctx.int(7))?;
    assert_eq!(y.cdf(&ctx.rational(7, 2)), ctx.rational(1, 2));
    assert_eq!(y.cdf(&ctx.int(3)), ctx.rational(1, 2));
    assert_eq!(y.cdf(&ctx.int(4)), ctx.rational(2, 3));
    // Poisson(2): P(−X ≤ −5/2) = P(X ≥ 3); scipy.stats.poisson.sf(2, 2) = 0.3233235838169366
    let neg = Distribution::poisson(ctx.int(2)).affine(ctx.int(-1), ctx.int(0))?;
    assert_close(
        &neg.cdf(&ctx.rational(-5, 2)),
        0.3233235838169366,
        "P(−X ≤ −5/2)",
    );
    Ok(())
}

#[test]
fn regression_affine_decreasing_lattice_quantile_is_left_to_the_numeric_route()
-> Result<(), SymplexError> {
    // Y = 7 − D: the smallest y with P(Y ≤ y) ≥ 1/6 is 1 (P(Y ≤ 1) = P(D = 6) = 1/6).
    // The reflected closed form −Q_D(1 − p) + 7 gave 2.
    let ctx = Context::new();
    let y = Distribution::die(ctx.int(6)).affine(ctx.int(-1), ctx.int(7))?;
    if let Some(q) = y.quantile(&ctx.rational(1, 6)) {
        assert_eq!(q, ctx.int(1), "closed-form quantile");
    }
    assert_eq!(y.quantile_f64(1.0 / 6.0)?, 1.0);
    assert_eq!(y.quantile_f64(0.2)?, 2.0);
    assert_eq!(y.quantile_f64(0.5)?, 3.0);
    Ok(())
}

#[test]
fn regression_truncated_lattice_with_closed_inner_cdf() -> Result<(), SymplexError> {
    // D | D ≥ 3 for a fair die: P(D ≤ 4 | D ≥ 3) = 2/4, P(D ≤ 3 | D ≥ 3) = 1/4,
    // quantile(1/4) = 3.  Before the fix the transported cdf subtracted
    // F(3) instead of F(2): cdf(4) = 1/4, cdf(3) = 0, and the inverse-transform
    // sampler never produced a 3.
    let ctx = Context::new();
    let t = Distribution::die(ctx.int(6)).truncated(&Support::half_line(ctx.int(3)))?;
    assert_eq!(t.cdf(&ctx.int(4)), ctx.rational(1, 2));
    assert_eq!(t.cdf(&ctx.int(3)), ctx.rational(1, 4));
    assert_eq!(t.cdf(&ctx.rational(9, 2)), ctx.rational(1, 2));
    assert_eq!(t.quantile(&ctx.rational(1, 4)), Some(ctx.int(3)));
    assert_eq!(t.quantile(&ctx.rational(1, 2)), Some(ctx.int(4)));
    let s = t.sample(4000, &mut Rng::new(7))?;
    let threes = s.iter().filter(|v| **v == 3.0).count() as f64 / 4000.0;
    assert!((threes - 0.25).abs() < 0.03, "fraction of 3s {threes}");
    let mean = s.iter().sum::<f64>() / 4000.0;
    assert!((mean - 4.5).abs() < 0.08, "sample mean {mean}");
    // Zero-truncated Poisson(2): scipy (poisson.cdf(2, 2) − poisson.cdf(0, 2)) / (1 − poisson.cdf(0, 2))
    // = 0.6260705709986628 = 4e⁻²/(1 − e⁻²); the old code gave (F(2) − F(1))/mass = 0.313.
    let zt = Distribution::poisson(ctx.int(2)).truncated(&Support::half_line(ctx.int(1)))?;
    assert_close(
        &zt.cdf(&ctx.int(2)),
        0.6260705709986628,
        "zero-truncated Poisson cdf(2)",
    );
    Ok(())
}

#[test]
fn regression_truncated_at_the_supports_own_lower_end_folds_f_lo_to_zero()
-> Result<(), SymplexError> {
    // LogNormal(0, 1) | X ≤ 1: the closed form Φ(ln x) does not fold at x = 0,
    // so the transported cdf must read F(0) as 0.
    // scipy: lognorm.cdf(0.5, 1) / lognorm.cdf(1, 1) = 0.4882171915711655
    let ctx = Context::new();
    let t = Distribution::log_normal(ctx.int(0), ctx.int(1))
        .truncated(&Support::interval(ctx.int(0), ctx.int(1)))?;
    assert_close(&t.cdf(&ctx.rational(1, 2)), 0.4882171915711655, "cdf(1/2)");
    // quantile(1/2) = exp(√2·erfinv(2·(1/4) − 1)) = exp(Φ⁻¹(1/4)) ;
    // scipy: lognorm.ppf(0.25, 1) = 0.5094162838632775
    let q = t
        .quantile(&ctx.rational(1, 2))
        .ok_or_else(|| SymplexError::computation_failed("q", "none"))?;
    assert_close(&q, 0.5094162838632775, "quantile(1/2)");
    Ok(())
}

#[test]
fn regression_even_power_on_a_one_sided_support_uses_one_branch() -> Result<(), SymplexError> {
    // X ~ Exponential(2), Y = X²: f_Y(y) = f_X(√y)/(2√y) = e^{−2√y}/√y on [0, ∞).
    // Before the fix the dead branch −√y was added: f_X(−√y) = 2e^{2√y} is not 0
    // off the support, and P(Y < 1) came out as e² − e⁻².
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let sq = Distribution::exponential(ctx.int(2)).transformed(&x, &x.powi(2))?;
    let expected = (-2 * y.sqrt()).exp() / y.sqrt();
    assert_eq!((sq.density(&y) - expected).simplify(), ctx.int(0));
    let r = RandomVariable::new(&ctx, "S", sq.clone());
    // P(X² < 1) = P(X < 1) = 1 − e⁻²
    assert_eq!(
        r.probability(&r.symbol().lt(&ctx.int(1)))?
            .equals(&(ctx.one() - ctx.int(-2).exp())),
        Some(true)
    );
    // E[X²] = 2/λ² = 1/2
    assert_eq!(sq.mean(), ctx.rational(1, 2));
    // Uniform(1, 3)²: density 1/(4√y) on [1, 9]; total mass 1.
    let u2 = Distribution::uniform(ctx.int(1), ctx.int(3)).transformed(&x, &x.powi(2))?;
    assert_eq!(
        (u2.density(&y) - ctx.rational(1, 4) / y.sqrt()).simplify(),
        ctx.int(0)
    );
    let r = RandomVariable::new(&ctx, "U2", u2.clone());
    assert_eq!(r.probability(&r.symbol().le(&ctx.int(9)))?, ctx.int(1));
    assert_eq!(
        r.probability(&r.symbol().le(&ctx.int(4)))?,
        ctx.rational(1, 2)
    );
    // |X| for X on [−2, 0]: the other single branch.
    let a = Distribution::uniform(ctx.int(-2), ctx.int(0)).transformed(&x, &x.abs())?;
    let r = RandomVariable::new(&ctx, "A", a.clone());
    assert_eq!(
        r.probability(&r.symbol().le(&ctx.rational(1, 2)))?,
        ctx.rational(1, 4)
    );
    assert_eq!(a.mean(), ctx.int(1));
    Ok(())
}

#[test]
fn regression_even_power_on_an_asymmetric_straddling_support_is_refused() {
    // Uniform(−1, 3)²: the two-branch formula is right only on [0, 1]; the
    // old code returned a density integrating to 3/2.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sq = Distribution::uniform(ctx.int(-1), ctx.int(3)).transformed(&x, &x.powi(2));
    assert!(
        matches!(sq, Err(SymplexError::NotImplemented(_))),
        "expected NotImplemented, got {sq:?}"
    );
    // Symmetric supports keep both branches: Uniform(−2, 2)² has E = 4/3.
    let sym = Distribution::uniform(ctx.int(-2), ctx.int(2))
        .transformed(&x, &x.powi(2))
        .expect("symmetric support");
    assert_eq!(sym.mean(), ctx.rational(4, 3));
    let r = RandomVariable::new(&ctx, "Q", sym);
    assert_eq!(
        r.probability(&r.symbol().le(&ctx.int(1))).expect("P"),
        ctx.rational(1, 2)
    );
}

#[test]
fn regression_transformed_affine_map_of_a_finite_lattice_is_enumerated() -> Result<(), SymplexError>
{
    // 2D for a die: the affine route refuses a slope of 2 on a lattice; the
    // finite range must then be enumerated instead of surfacing that error.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let d = Distribution::die(ctx.int(6));
    let twice = d.transformed(&x, &(2 * &x))?;
    assert_eq!(twice.name(), "Finite");
    assert_eq!(twice.mean(), ctx.int(7));
    let r = RandomVariable::new(&ctx, "T", twice.clone());
    assert_eq!(
        r.probability(&r.symbol().eq_expr(&ctx.int(4)))?,
        ctx.rational(1, 6)
    );
    assert_eq!(r.probability(&r.symbol().eq_expr(&ctx.int(3)))?, ctx.int(0));
    assert_eq!(
        r.probability(&r.symbol().le(&ctx.int(7)))?,
        ctx.rational(1, 2)
    );
    // A half-integer shift of a binomial: P(B + 1/2 ≤ 3) = P(B ≤ 2) = 64/81·… exactly
    // Fraction: sum_{k≤2} C(5,k)(1/3)^k(2/3)^(5−k) = 192/243 = 64/81
    let b = Distribution::binomial(ctx.int(5), ctx.rational(1, 3));
    let shifted = b.transformed(&x, &(&x + ctx.rational(1, 2)))?;
    let r = RandomVariable::new(&ctx, "H", shifted.clone());
    assert_eq!(
        r.probability(&r.symbol().le(&ctx.int(3)))?,
        ctx.rational(64, 81)
    );
    assert_eq!(shifted.mean(), ctx.rational(13, 6));
    Ok(())
}

#[test]
fn regression_affine_of_a_lattice_rejects_a_non_integer_intercept() {
    // Binomial(5, 1/3) + 1/2 is not on the integer lattice; the old code
    // accepted it and evaluated the pmf at half-integers.
    let ctx = Context::new();
    let b = Distribution::binomial(ctx.int(5), ctx.rational(1, 3));
    assert!(matches!(
        b.affine(ctx.int(1), ctx.rational(1, 2)),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(b.affine(ctx.int(-1), ctx.int(5)).is_ok());
    // A finite table may be shifted by anything.
    assert!(
        Distribution::finite(
            &ctx,
            vec![
                (ctx.int(1), ctx.rational(1, 2)),
                (ctx.int(2), ctx.rational(1, 2))
            ]
        )
        .affine(ctx.int(3), ctx.rational(1, 2))
        .is_ok()
    );
}

#[test]
fn regression_mixture_quantile_f64_over_overlapping_supports() -> Result<(), SymplexError> {
    let ctx = Context::new();
    // Two normals: the support used to be listed as ℝ ∪ ℝ and quantile_f64 gave
    // "unsupported support shape".  Median of ¼N(−1,1) + ¾N(3,2): solve F(m) = ½;
    // mpmath.findroot(lambda m: 0.25*ncdf(m+1) + 0.75*ncdf((m-3)/2) - 0.5, 2) = 2.1400934935657373
    let a = Distribution::normal(ctx.int(-1), ctx.int(1));
    let b = Distribution::normal(ctx.int(3), ctx.int(2));
    let m = Distribution::mixture(&[(ctx.rational(1, 4), a), (ctx.rational(3, 4), b)])?;
    assert!((m.quantile_f64(0.5)? - 2.1400934935657373).abs() < 1e-9);
    // Two finite tables sharing the value 2: the point must not be counted twice.
    // Masses: 1 ↦ 1/4, 2 ↦ 1/2, 3 ↦ 1/4, so quantile(0.9) = 3.
    let f1 = Distribution::finite(
        &ctx,
        vec![
            (ctx.int(1), ctx.rational(1, 2)),
            (ctx.int(2), ctx.rational(1, 2)),
        ],
    );
    let f2 = Distribution::finite(
        &ctx,
        vec![
            (ctx.int(2), ctx.rational(1, 2)),
            (ctx.int(3), ctx.rational(1, 2)),
        ],
    );
    let mf = Distribution::mixture(&[(ctx.rational(1, 2), f1), (ctx.rational(1, 2), f2)])?;
    assert_eq!(mf.quantile_f64(0.9)?, 3.0);
    assert_eq!(mf.quantile_f64(0.3)?, 2.0);
    assert_eq!(mf.quantile_f64(0.2)?, 1.0);
    assert_eq!(mf.support().pieces().len(), 3);
    Ok(())
}

#[test]
fn regression_quantile_f64_of_a_lattice_family_when_the_first_atom_already_covers_p()
-> Result<(), SymplexError> {
    let ctx = Context::new();
    // scipy.stats.poisson.ppf(0.1, 2) = 1.0 ; F(0) = e⁻² = 0.135 ≥ 0.1 → the
    // bracket search used to stop with "f(a) and f(b) must have opposite signs".
    let p = Distribution::poisson(ctx.int(2));
    assert_eq!(p.quantile_f64(0.1)?, 0.0);
    // scipy.stats.geom.ppf(0.3, 1/3) = 1.0
    let g = Distribution::geometric(ctx.rational(1, 3));
    assert_eq!(g.quantile_f64(0.3)?, 1.0);
    // scipy.stats.geom.ppf(0.9, 1/3) = 6.0
    assert_eq!(g.quantile_f64(0.9)?, 6.0);
    Ok(())
}

#[test]
fn regression_triangular_with_the_mode_at_an_end() -> Result<(), SymplexError> {
    // Triangular(0, 1, 0) is accepted by `try_triangular` (c ∈ [a, b]) but its
    // density divided by c − a = 0: `zoo` in the density and `nan` moments.
    // scipy: triang(c=0).pdf(1/4) = 1.5, cdf(1/4) = 0.4375, ppf(0.3) = 0.16333997346592444,
    // moment(2) = 1/6, var = 1/18.
    let ctx = Context::new();
    let t = Distribution::try_triangular(ctx.int(0), ctx.int(1), ctx.int(0))?;
    assert_eq!(
        t.density(&ctx.rational(1, 4)).simplify(),
        ctx.rational(3, 2)
    );
    assert_eq!(t.cdf(&ctx.rational(1, 4)), ctx.rational(7, 16));
    assert_close(
        &t.quantile(&ctx.rational(3, 10))
            .ok_or_else(|| SymplexError::computation_failed("q", "none"))?,
        0.16333997346592444,
        "ppf(0.3)",
    );
    assert_eq!(t.moment(2), ctx.rational(1, 6));
    assert_eq!(t.variance(), ctx.rational(1, 18));
    assert_eq!(t.mean(), ctx.rational(1, 3));
    let x = ctx.symbol("x");
    assert_eq!(
        t.density(&x)
            .integrate_definite(&x, &ctx.int(0), &ctx.int(1))
            .simplify(),
        ctx.int(1)
    );
    // mgf(1) = 2(e − 1 − 1)/1 = 2e − 4 ; sympy: integrate(2*(1-x)*exp(x), (x, 0, 1)) = -4 + 2*E
    assert_close(
        &t.mgf(&ctx.int(1)),
        2.0 * std::f64::consts::E - 4.0,
        "mgf(1)",
    );
    // scipy: triang(c=1).pdf(1/4) = 0.5, cdf(1/4) = 0.0625, ppf(0.3) = 0.5477225575051661, moment(2) = 1/2
    let u = Distribution::try_triangular(ctx.int(0), ctx.int(1), ctx.int(1))?;
    assert_eq!(
        u.density(&ctx.rational(1, 4)).simplify(),
        ctx.rational(1, 2)
    );
    assert_eq!(u.cdf(&ctx.rational(1, 4)), ctx.rational(1, 16));
    assert_close(
        &u.quantile(&ctx.rational(3, 10))
            .ok_or_else(|| SymplexError::computation_failed("q", "none"))?,
        0.5477225575051661,
        "ppf(0.3) mode at b",
    );
    assert_eq!(u.moment(2), ctx.rational(1, 2));
    // mgf(1) = 2(e − e + 1)... = 2(1·e − e + 1) = 2 ; sympy: integrate(2*x*exp(x), (x, 0, 1)) = 2
    assert_close(&u.mgf(&ctx.int(1)), 2.0, "mgf(1) mode at b");
    let s = t.sample(20_000, &mut Rng::new(5))?;
    let mean = s.iter().sum::<f64>() / 20_000.0;
    assert!(
        (mean - 1.0 / 3.0).abs() < 4.0 * (1.0f64 / 18.0).sqrt() / 20_000f64.sqrt(),
        "sample mean {mean}"
    );
    Ok(())
}

// ── continuous families at fresh parameters ──────────────────────────────

#[test]
fn normal_minus_three_halves_seven_quarters() -> Result<(), SymplexError> {
    // scipy.stats.norm(loc=-1.5, scale=1.75)
    let ctx = Context::new();
    let d = Distribution::try_normal(ctx.rational(-3, 2), ctx.rational(7, 4))?;
    assert_eq!(d.mean(), ctx.rational(-3, 2));
    assert_eq!(d.variance(), ctx.rational(49, 16));
    // norm.pdf(1/3) = 0.1316896638818252
    assert_close(
        &d.density(&ctx.rational(1, 3)),
        0.1316896638818252,
        "pdf(1/3)",
    );
    // norm.cdf(1/3) = 0.8525929209684728
    assert_close(&d.cdf(&ctx.rational(1, 3)), 0.8525929209684728, "cdf(1/3)");
    // norm.ppf(0.3) = -2.4177008972390714
    assert_f64(d.quantile_f64(0.3)?, -2.4177008972390714, "ppf(0.3)");
    // norm.entropy() = 1.9785543211400953
    assert_close(&d.entropy(), 1.9785543211400953, "entropy");
    // exp(μt + σ²t²/2) at t = 1/2: 0.6926797554134794
    assert_close(&d.mgf(&ctx.rational(1, 2)), 0.6926797554134794, "mgf(1/2)");
    // norm.cdf(1/3) − norm.cdf(−1) = 0.24014140206646517
    let p = prob(&ctx, &d, |x| {
        x.gt(&ctx.int(-1)).and(&x.le(&ctx.rational(1, 3)))
    });
    assert_close(&p, 0.24014140206646517, "P(−1 < X ≤ 1/3)");
    // E[X⁴] = μ⁴ + 6μ²σ² + 3σ⁴ = 81/16 + 6·(9/4)(49/16) + 3·(2401/256) = (1296 + 10584 + 7203)/256
    assert_eq!(d.moment(4), ctx.rational(19083, 256));
    check_sampling(&d, 20_000, 11);
    check_quantile_roundtrip(&d, 0.7);
    Ok(())
}

#[test]
fn uniform_minus_two_to_five_halves() -> Result<(), SymplexError> {
    // scipy.stats.uniform(loc=-2, scale=4.5)
    let ctx = Context::new();
    let d = Distribution::try_uniform(ctx.int(-2), ctx.rational(5, 2))?;
    assert_eq!(d.mean(), ctx.rational(1, 4));
    // Fraction(9, 2)**2 / 12 = 27/16
    assert_eq!(d.variance(), ctx.rational(27, 16));
    assert_eq!(d.density(&ctx.int(0)), ctx.rational(2, 9));
    // uniform.cdf(1/3) = (1/3 + 2)/(9/2) = 14/27
    assert_eq!(d.cdf(&ctx.rational(1, 3)), ctx.rational(14, 27));
    // uniform.ppf(0.3) = -0.65
    assert_f64(d.quantile_f64(0.3)?, -0.65, "ppf(0.3)");
    // uniform.entropy() = ln(9/2) = 1.5040773967762742
    assert_close(&d.entropy(), 1.5040773967762742, "entropy");
    // endpoints outside the support: P(−5 < X ≤ 0) = 2/(9/2) = 4/9, P(X > 3) = 0, P(X ≤ −2) = 0
    assert_eq!(
        prob(&ctx, &d, |x| x.gt(&ctx.int(-5)).and(&x.le(&ctx.int(0)))),
        ctx.rational(4, 9)
    );
    assert_eq!(prob(&ctx, &d, |x| x.gt(&ctx.int(3))), ctx.int(0));
    assert_eq!(prob(&ctx, &d, |x| x.le(&ctx.int(-2))), ctx.int(0));
    assert_eq!(prob(&ctx, &d, |x| x.ge(&ctx.int(-2))), ctx.int(1));
    check_sampling(&d, 20_000, 12);
    check_quantile_roundtrip(&d, 1.25);
    Ok(())
}

#[test]
fn exponential_rate_three_sevenths() -> Result<(), SymplexError> {
    // scipy.stats.expon(scale=7/3)
    let ctx = Context::new();
    let d = Distribution::try_exponential(ctx.rational(3, 7))?;
    assert_eq!(d.mean(), ctx.rational(7, 3));
    assert_eq!(d.variance(), ctx.rational(49, 9));
    // expon.pdf(2) = 0.18187407671869285
    assert_close(&d.density(&ctx.int(2)), 0.18187407671869285, "pdf(2)");
    // expon.cdf(2) = 0.57562715432305
    assert_close(&d.cdf(&ctx.int(2)), 0.57562715432305, "cdf(2)");
    // expon.ppf(0.3) = 0.8322415358570424
    assert_f64(d.quantile_f64(0.3)?, 0.8322415358570424, "ppf(0.3)");
    // expon.entropy() = 1 − ln(3/7) = 1.8472978603872037
    assert_close(&d.entropy(), 1.8472978603872037, "entropy");
    // P(1 < X ≤ 2) = e^{−3/7} − e^{−6/7} exactly
    let p = prob(&ctx, &d, |x| x.gt(&ctx.int(1)).and(&x.le(&ctx.int(2))));
    assert_eq!(
        p.equals(&(ctx.rational(-3, 7).exp() - ctx.rational(-6, 7).exp())),
        Some(true)
    );
    // mgf(t) = λ/(λ − t) at t = 1/7: (3/7)/(2/7) = 3/2
    assert_eq!(d.mgf(&ctx.rational(1, 7)).simplify(), ctx.rational(3, 2));
    // E[X³] = 3!/λ³ = 6·343/27 = 686/9
    assert_eq!(d.moment(3), ctx.rational(686, 9));
    check_sampling(&d, 20_000, 13);
    check_quantile_roundtrip(&d, 3.0);
    Ok(())
}

#[test]
fn gamma_shape_seven_halves_scale_three_quarters() -> Result<(), SymplexError> {
    // scipy.stats.gamma(a=3.5, scale=0.75)
    let ctx = Context::new();
    let d = Distribution::try_gamma(ctx.rational(7, 2), ctx.rational(3, 4))?;
    assert_eq!(d.mean(), ctx.rational(21, 8));
    assert_eq!(d.variance(), ctx.rational(63, 32));
    // gamma.pdf(2) = 0.32371717404096995
    assert_close(&d.density(&ctx.int(2)), 0.32371717404096995, "pdf(2)");
    // gamma.cdf(2) = 0.3806444747602957
    assert_close(&d.cdf(&ctx.int(2)), 0.3806444747602957, "cdf(2)");
    // gamma.ppf(0.3) = 1.7517489183679027
    assert_f64(d.quantile_f64(0.3)?, 1.7517489183679027, "ppf(0.3)");
    // gamma.entropy() = 1.6553999282821859
    assert_close(&d.entropy(), 1.6553999282821859, "entropy");
    // (1 − θt)^{−k} at t = 1/2 = 5.181075718419873
    assert_close(&d.mgf(&ctx.rational(1, 2)), 5.181075718419873, "mgf(1/2)");
    // gamma.cdf(2) − gamma.cdf(1) = 0.2946775248613497
    let p = prob(&ctx, &d, |x| x.gt(&ctx.int(1)).and(&x.le(&ctx.int(2))));
    assert_close(&p, 0.2946775248613497, "P(1 < X ≤ 2)");
    // E[X³] = θ³ k(k+1)(k+2) = 18711/512 (Fraction); gamma.moment(3) = 36.544921875
    assert_eq!(d.moment(3), ctx.rational(18711, 512));
    check_quantile_roundtrip(&d, 3.0);
    Ok(())
}

#[test]
fn chi_squared_seven() -> Result<(), SymplexError> {
    // scipy.stats.chi2(df=7)
    let ctx = Context::new();
    let d = Distribution::try_chi_squared(ctx.int(7))?;
    assert_eq!(d.mean(), ctx.int(7));
    assert_eq!(d.variance(), ctx.int(14));
    // chi2.pdf(5) = 0.12204152134938738
    assert_close(&d.density(&ctx.int(5)), 0.12204152134938738, "pdf(5)");
    // chi2.cdf(5) = 0.34003677030571744
    assert_close(&d.cdf(&ctx.int(5)), 0.34003677030571744, "cdf(5)");
    // chi2.ppf(0.3) = 4.671330448981074
    assert_f64(d.quantile_f64(0.3)?, 4.671330448981074, "ppf(0.3)");
    // chi2.entropy() = 2.6362291812939125
    assert_close(&d.entropy(), 2.6362291812939125, "entropy");
    // (1 − 2t)^{−7/2} at t = 1/4 = 2^{7/2} = 11.313708498984761
    assert_close(&d.mgf(&ctx.rational(1, 4)), 11.313708498984761, "mgf(1/4)");
    // chi2.cdf(5) − chi2.cdf(3) = 0.22503900194886806
    let p = prob(&ctx, &d, |x| x.gt(&ctx.int(3)).and(&x.le(&ctx.int(5))));
    assert_close(&p, 0.22503900194886806, "P(3 < X ≤ 5)");
    // E[X²] = k² + 2k = 63
    assert_eq!(d.moment(2), ctx.int(63));
    Ok(())
}

#[test]
fn beta_three_sevenths_five_halves() -> Result<(), SymplexError> {
    // scipy.stats.beta(a=3/7, b=2.5)
    let ctx = Context::new();
    let d = Distribution::try_beta(ctx.rational(3, 7), ctx.rational(5, 2))?;
    // Fraction: mean 6/41, var 588/18491
    assert_eq!(d.mean(), ctx.rational(6, 41));
    assert_eq!(d.variance(), ctx.rational(588, 18491));
    // beta.pdf(1/3) = 0.6954494781666041
    assert_close(
        &d.density(&ctx.rational(1, 3)),
        0.6954494781666041,
        "pdf(1/3)",
    );
    // beta.cdf(1/3) = 0.8522768492555847
    assert_close(&d.cdf(&ctx.rational(1, 3)), 0.8522768492555847, "cdf(1/3)");
    // beta.ppf(0.3) = 0.02083250526531868
    assert_f64(d.quantile_f64(0.3)?, 0.02083250526531868, "ppf(0.3)");
    // beta.entropy() = -1.1935490460612206
    assert_close(&d.entropy(), -1.1935490460612206, "entropy");
    // beta.cdf(1/2) − beta.cdf(1/4) = 0.1538404931274442
    let p = prob(&ctx, &d, |x| {
        x.gt(&ctx.rational(1, 4)).and(&x.le(&ctx.rational(1, 2)))
    });
    assert_close(&p, 0.1538404931274442, "P(1/4 < X ≤ 1/2)");
    // E[X²] = α(α+1)/((α+β)(α+β+1)) = 24/451
    assert_eq!(d.moment(2), ctx.rational(24, 451));
    // Endpoints outside the support
    assert_eq!(prob(&ctx, &d, |x| x.gt(&ctx.int(1))), ctx.int(0));
    assert_eq!(prob(&ctx, &d, |x| x.ge(&ctx.int(-1))), ctx.int(1));
    check_quantile_roundtrip(&d, 0.2);
    Ok(())
}

#[test]
fn cauchy_minus_one_three_quarters() -> Result<(), SymplexError> {
    // scipy.stats.cauchy(loc=-1, scale=0.75)
    let ctx = Context::new();
    let d = Distribution::try_cauchy(ctx.int(-1), ctx.rational(3, 4))?;
    // cauchy.pdf(1) = 0.052324912797335456
    assert_close(&d.density(&ctx.int(1)), 0.052324912797335456, "pdf(1)");
    // cauchy.cdf(1) = 0.8857997487800918
    assert_close(&d.cdf(&ctx.int(1)), 0.8857997487800918, "cdf(1)");
    // cauchy.ppf(0.3) = -1.5449068960040209
    assert_f64(d.quantile_f64(0.3)?, -1.5449068960040209, "ppf(0.3)");
    // cauchy.entropy() = ln(4π·3/4) = ln(3π) = 2.24334217451751
    assert_close(&d.entropy(), 2.24334217451751, "entropy");
    // cauchy.cdf(1) − cauchy.cdf(−2) = 0.6809669840809583
    let p = prob(&ctx, &d, |x| x.gt(&ctx.int(-2)).and(&x.le(&ctx.int(1))));
    assert_close(&p, 0.6809669840809583, "P(−2 < X ≤ 1)");
    // `median()` hands back the raw closed form (`3/4·tan(0) − 1`); it equals −1.
    assert_eq!(d.median().map(|m| m.simplify()), Some(ctx.int(-1)));
    // Symmetric quartiles: x₀ ± γ
    assert_eq!(
        d.quantile(&ctx.rational(3, 4)).map(|q| q.simplify()),
        Some(ctx.rational(-1, 4))
    );
    // Sampling exists (inverse transform) and every draw is a finite real.
    let s = d.sample(1000, &mut Rng::new(2))?;
    assert!(s.iter().all(|v| v.is_finite()));
    check_quantile_roundtrip(&d, 2.5);
    Ok(())
}

#[test]
fn laplace_two_three_sevenths() -> Result<(), SymplexError> {
    // scipy.stats.laplace(loc=2, scale=3/7)
    let ctx = Context::new();
    let d = Distribution::try_laplace(ctx.int(2), ctx.rational(3, 7))?;
    assert_eq!(d.mean(), ctx.int(2));
    assert_eq!(d.variance(), ctx.rational(18, 49));
    // laplace.pdf(1) = 0.11313396250847257
    assert_close(&d.density(&ctx.int(1)), 0.11313396250847257, "pdf(1)");
    // laplace.cdf(1) = 0.04848598393220253, laplace.cdf(3) = 0.9515140160677975
    assert_close(&d.cdf(&ctx.int(1)), 0.04848598393220253, "cdf(1)");
    assert_close(&d.cdf(&ctx.int(3)), 0.9515140160677975, "cdf(3)");
    assert_eq!(d.cdf(&ctx.int(2)), ctx.rational(1, 2));
    // laplace.ppf(0.3) = 1.7810747326717182, laplace.ppf(0.8) = 2.3926960279460667
    assert_f64(d.quantile_f64(0.3)?, 1.7810747326717182, "ppf(0.3)");
    assert_f64(d.quantile_f64(0.8)?, 2.3926960279460667, "ppf(0.8)");
    // laplace.entropy() = 1 + ln(6/7) = 0.8458493201727417
    assert_close(&d.entropy(), 0.8458493201727417, "entropy");
    // laplace.cdf(3) − laplace.cdf(1) = 0.903028032135595
    let p = prob(&ctx, &d, |x| x.gt(&ctx.int(1)).and(&x.le(&ctx.int(3))));
    assert_close(&p, 0.903028032135595, "P(1 < X ≤ 3)");
    // E[X⁴] = μ⁴ + 6μ²(2b²) + 24b⁴ = 61528/2401 (Fraction); laplace.moment(4) = 25.625989171178674
    assert_eq!(d.moment(4), ctx.rational(61528, 2401));
    // mgf(1) = e^{2}/(1 − 9/49) = 49e²/40
    assert_eq!(
        d.mgf(&ctx.int(1))
            .simplify()
            .equals(&(ctx.rational(49, 40) * ctx.int(2).exp())),
        Some(true)
    );
    check_sampling(&d, 20_000, 14);
    check_quantile_roundtrip(&d, 1.5);
    Ok(())
}

#[test]
fn logistic_minus_half_five_quarters() -> Result<(), SymplexError> {
    // scipy.stats.logistic(loc=-0.5, scale=1.25)
    let ctx = Context::new();
    let d = Distribution::try_logistic(ctx.rational(-1, 2), ctx.rational(5, 4))?;
    assert_eq!(d.mean(), ctx.rational(-1, 2));
    // logistic.var() = 5.140418958900708 = 25π²/48
    assert_close(&d.variance(), 5.140418958900708, "variance");
    assert_eq!(
        d.variance()
            .equals(&(ctx.rational(25, 48) * ctx.pi().powi(2))),
        Some(true)
    );
    // logistic.pdf(1) = 0.14231555251744454
    assert_close(&d.density(&ctx.int(1)), 0.14231555251744454, "pdf(1)");
    // logistic.cdf(1) = 0.7685247834990175
    assert_close(&d.cdf(&ctx.int(1)), 0.7685247834990175, "cdf(1)");
    // logistic.ppf(0.3) = -1.5591223254840045
    assert_f64(d.quantile_f64(0.3)?, -1.5591223254840045, "ppf(0.3)");
    // logistic.entropy() = ln(5/4) + 2 = 2.2231435513142097
    assert_close(&d.entropy(), 2.2231435513142097, "entropy");
    // logistic.moment(4) = 118.75353814505958
    assert_close(&d.moment(4), 118.75353814505958, "E[X⁴]");
    // logistic.cdf(1) − logistic.cdf(−1) = 0.36721244361146954
    let p = prob(&ctx, &d, |x| x.gt(&ctx.int(-1)).and(&x.le(&ctx.int(1))));
    assert_close(&p, 0.36721244361146954, "P(−1 < X ≤ 1)");
    check_sampling(&d, 20_000, 15);
    check_quantile_roundtrip(&d, -2.0);
    Ok(())
}

#[test]
fn log_normal_quarter_three_fifths() -> Result<(), SymplexError> {
    // scipy.stats.lognorm(s=0.6, scale=exp(0.25))
    let ctx = Context::new();
    let d = Distribution::try_log_normal(ctx.rational(1, 4), ctx.rational(3, 5))?;
    // lognorm.mean() = 1.5372575235482813, lognorm.var() = 1.0240270399155391
    assert_close(&d.mean(), 1.5372575235482813, "mean");
    assert_close(&d.variance(), 1.0240270399155391, "variance");
    // lognorm.pdf(2) = 0.2530902107710365
    assert_close(&d.density(&ctx.int(2)), 0.2530902107710365, "pdf(2)");
    // lognorm.cdf(2) = 0.7699185489845288
    assert_close(&d.cdf(&ctx.int(2)), 0.7699185489845288, "cdf(2)");
    // lognorm.ppf(0.3) = 0.9374045800245144
    assert_f64(d.quantile_f64(0.3)?, 0.9374045800245144, "ppf(0.3)");
    // lognorm.entropy() = 1.158112909438682
    assert_close(&d.entropy(), 1.158112909438682, "entropy");
    // lognorm.cdf(2) − lognorm.cdf(1) = 0.4314574294738392
    let p = prob(&ctx, &d, |x| x.gt(&ctx.int(1)).and(&x.le(&ctx.int(2))));
    assert_close(&p, 0.4314574294738392, "P(1 < X ≤ 2)");
    // lognorm.moment(3) = exp(3/4 + 9·0.36/2) = 10.697392284111054
    assert_close(&d.moment(3), 10.697392284111054, "E[X³]");
    assert_eq!(
        d.moment(3).equals(&ctx.rational(237, 100).exp()),
        Some(true)
    );
    // Below the support the CDF is 0; P(X ≤ 0) = 0
    assert_eq!(d.cdf(&ctx.int(-1)), ctx.int(0));
    assert_eq!(prob(&ctx, &d, |x| x.le(&ctx.int(0))), ctx.int(0));
    check_sampling(&d, 20_000, 16);
    check_quantile_roundtrip(&d, 2.5);
    Ok(())
}

#[test]
fn student_t_seven_halves() -> Result<(), SymplexError> {
    // scipy.stats.t(df=3.5)
    let ctx = Context::new();
    let d = Distribution::try_student_t(ctx.rational(7, 2))?;
    assert_eq!(d.mean(), ctx.int(0));
    // t.var() = ν/(ν − 2) = 7/3
    assert_eq!(d.variance(), ctx.rational(7, 3));
    // t.pdf(1) = 0.21120394361781114
    assert_close(&d.density(&ctx.int(1)), 0.21120394361781114, "pdf(1)");
    // t.cdf(1) = 0.8093313732182772, t.cdf(-1/2) = 0.3234252196612755
    assert_close(&d.cdf(&ctx.int(1)), 0.8093313732182772, "cdf(1)");
    assert_close(
        &d.cdf(&ctx.rational(-1, 2)),
        0.3234252196612755,
        "cdf(−1/2)",
    );
    // t.ppf(0.3) = -0.575338701772231
    assert_f64(d.quantile_f64(0.3)?, -0.575338701772231, "ppf(0.3)");
    // t.entropy() = 1.720890120682467
    assert_close(&d.entropy(), 1.720890120682467, "entropy");
    // t.cdf(1) − t.cdf(−1/2) = 0.4859061535570017
    let p = prob(&ctx, &d, |x| {
        x.gt(&ctx.rational(-1, 2)).and(&x.le(&ctx.int(1)))
    });
    assert_close(&p, 0.4859061535570017, "P(−1/2 < X ≤ 1)");
    // The fourth moment needs ν > 4: with ν = 7/2 it is infinite; the closed form is absent and
    // the generic integral must not report a finite number.
    let m4 = d.moment(4);
    assert!(
        m4.has_unevaluated() || m4.eval_f64().map(|v| !v.is_finite()).unwrap_or(true),
        "E[X⁴] for ν = 7/2 must not be a finite number, got `{m4}`"
    );
    check_quantile_roundtrip(&d, 1.5);
    Ok(())
}

#[test]
fn f_distribution_five_nine() -> Result<(), SymplexError> {
    // scipy.stats.f(dfn=5, dfd=9)
    let ctx = Context::new();
    let d = Distribution::try_f_distribution(ctx.int(5), ctx.int(9))?;
    // Fraction: mean 9/7, var 1944/1225
    assert_eq!(d.mean(), ctx.rational(9, 7));
    assert_eq!(d.variance(), ctx.rational(1944, 1225));
    // f.pdf(2) = 0.16212057450401576
    assert_close(&d.density(&ctx.int(2)), 0.16212057450401576, "pdf(2)");
    // f.cdf(2) = 0.827290123585564
    assert_close(&d.cdf(&ctx.int(2)), 0.827290123585564, "cdf(2)");
    // f.ppf(0.3) = 0.6031760598027508
    assert_f64(d.quantile_f64(0.3)?, 0.6031760598027508, "ppf(0.3)");
    // f.cdf(2) − f.cdf(1) = 0.2968137950295212
    let p = prob(&ctx, &d, |x| x.gt(&ctx.int(1)).and(&x.le(&ctx.int(2))));
    assert_close(&p, 0.2968137950295212, "P(1 < X ≤ 2)");
    // f.moment(2) = 3.24 = 81/25
    assert_eq!(d.moment(2), ctx.rational(81, 25));
    check_quantile_roundtrip(&d, 0.8);
    Ok(())
}

#[test]
fn weibull_scale_three_halves_shape_five_halves() -> Result<(), SymplexError> {
    // scipy.stats.weibull_min(c=2.5, scale=1.5)
    let ctx = Context::new();
    let d = Distribution::try_weibull(ctx.rational(3, 2), ctx.rational(5, 2))?;
    // weibull_min.mean() = 1.3308957262546128, var() = 0.32433005054275243
    assert_close(&d.mean(), 1.3308957262546128, "mean");
    assert_close(&d.variance(), 0.32433005054275243, "variance");
    // weibull_min.pdf(1) = 0.6311199069084408
    assert_close(&d.density(&ctx.int(1)), 0.6311199069084408, "pdf(1)");
    // weibull_min.cdf(1) = 0.304335217702675
    assert_close(&d.cdf(&ctx.int(1)), 0.304335217702675, "cdf(1)");
    // weibull_min.ppf(0.3) = 0.9931167337940499
    assert_f64(d.quantile_f64(0.3)?, 0.9931167337940499, "ppf(0.3)");
    // weibull_min.entropy() = 0.8355037751749289
    assert_close(&d.entropy(), 0.8355037751749289, "entropy");
    // weibull_min.cdf(2) − weibull_min.cdf(1) = 0.5672899551394157
    let p = prob(&ctx, &d, |x| x.gt(&ctx.int(1)).and(&x.le(&ctx.int(2))));
    assert_close(&p, 0.5672899551394157, "P(1 < X ≤ 2)");
    // weibull_min.moment(3) = 3.7185834067190315
    assert_close(&d.moment(3), 3.7185834067190315, "E[X³]");
    check_sampling(&d, 20_000, 17);
    check_quantile_roundtrip(&d, 2.0);
    Ok(())
}

#[test]
fn pareto_scale_three_halves_shape_seven_halves() -> Result<(), SymplexError> {
    // scipy.stats.pareto(b=3.5, scale=1.5)
    let ctx = Context::new();
    let d = Distribution::try_pareto(ctx.rational(3, 2), ctx.rational(7, 2))?;
    // Fraction: mean 21/10, var 21/25
    assert_eq!(d.mean(), ctx.rational(21, 10));
    assert_eq!(d.variance(), ctx.rational(21, 25));
    // pareto.pdf(2) = 0.6393703176377302
    assert_close(&d.density(&ctx.int(2)), 0.6393703176377302, "pdf(2)");
    // pareto.cdf(2) = 0.6346455327784399
    assert_close(&d.cdf(&ctx.int(2)), 0.6346455327784399, "cdf(2)");
    // pareto.ppf(0.3) = 1.660920945451289
    assert_f64(d.quantile_f64(0.3)?, 1.660920945451289, "ppf(0.3)");
    // pareto.entropy() = 0.4384164253270819
    assert_close(&d.entropy(), 0.4384164253270819, "entropy");
    // P(2 < X ≤ 3) = (3/4)^{7/2} − (1/2)^{7/2} = 0.27696611957324163
    let p = prob(&ctx, &d, |x| x.gt(&ctx.int(2)).and(&x.le(&ctx.int(3))));
    assert_close(&p, 0.27696611957324163, "P(2 < X ≤ 3)");
    assert_eq!(
        p.equals(
            &(ctx.rational(3, 4).pow(&ctx.rational(7, 2))
                - ctx.rational(1, 2).pow(&ctx.rational(7, 2)))
        ),
        Some(true)
    );
    // Below the minimum: cdf(1) = 0 and P(X ≤ 3/2) = 0
    assert_eq!(d.cdf(&ctx.int(1)), ctx.int(0));
    assert_eq!(prob(&ctx, &d, |x| x.le(&ctx.rational(3, 2))), ctx.int(0));
    // E[X³] = α x_m³/(α − 3) = (7/2)(27/8)/(1/2) = 189/8; E[X⁴] does not exist (α = 7/2 < 4)
    assert_eq!(d.moment(3), ctx.rational(189, 8));
    let m4 = d.moment(4);
    assert!(
        m4.has_unevaluated() || m4.eval_f64().map(|v| !v.is_finite()).unwrap_or(true),
        "E[X⁴] for α = 7/2 must not be finite, got `{m4}`"
    );
    check_sampling(&d, 20_000, 18);
    check_quantile_roundtrip(&d, 2.0);
    Ok(())
}

#[test]
fn triangular_minus_one_three_mode_two() -> Result<(), SymplexError> {
    // scipy.stats.triang(c=0.75, loc=-1, scale=4)
    let ctx = Context::new();
    let d = Distribution::try_triangular(ctx.int(-1), ctx.int(3), ctx.int(2))?;
    assert_eq!(d.mean(), ctx.rational(4, 3));
    // Fraction: (1 + 9 + 4 + 3 + 2 − 6)/18 = 13/18
    assert_eq!(d.variance(), ctx.rational(13, 18));
    // triang.pdf(0) = 1/6, triang.pdf(5/2) = 1/4
    assert_eq!(d.density(&ctx.int(0)).simplify(), ctx.rational(1, 6));
    assert_eq!(
        d.density(&ctx.rational(5, 2)).simplify(),
        ctx.rational(1, 4)
    );
    // triang.cdf(0) = 1/12, triang.cdf(5/2) = 15/16
    assert_eq!(d.cdf(&ctx.int(0)), ctx.rational(1, 12));
    assert_eq!(d.cdf(&ctx.rational(5, 2)), ctx.rational(15, 16));
    // triang.ppf(0.3) = 0.8973665961010275, triang.ppf(0.9) = 2.367544467966324
    assert_f64(d.quantile_f64(0.3)?, 0.8973665961010275, "ppf(0.3)");
    assert_f64(d.quantile_f64(0.9)?, 2.367544467966324, "ppf(0.9)");
    // triang.entropy() = 1/2 + ln 2 = 1.1931471805599454
    assert_close(&d.entropy(), 1.1931471805599454, "entropy");
    // triang.cdf(5/2) − triang.cdf(0) = 41/48 = 0.8541666666666666
    assert_eq!(
        prob(&ctx, &d, |x| x
            .gt(&ctx.int(0))
            .and(&x.le(&ctx.rational(5, 2)))),
        ctx.rational(41, 48)
    );
    // scipy.integrate.quad(exp(x)·pdf, −1, 3) = 5.1780443025019744
    assert_close(&d.mgf(&ctx.int(1)), 5.1780443025019744, "mgf(1)");
    check_sampling(&d, 20_000, 19);
    check_quantile_roundtrip(&d, 2.5);
    Ok(())
}

// ── discrete families at fresh parameters ────────────────────────────────

#[test]
fn bernoulli_three_sevenths() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let d = Distribution::try_bernoulli(ctx.rational(3, 7))?;
    assert_eq!(d.mean(), ctx.rational(3, 7));
    assert_eq!(d.variance(), ctx.rational(12, 49));
    assert_eq!(d.density(&ctx.int(1)).simplify(), ctx.rational(3, 7));
    assert_eq!(d.density(&ctx.int(0)).simplify(), ctx.rational(4, 7));
    // open vs closed ends on the lattice
    assert_eq!(prob(&ctx, &d, |x| x.gt(&ctx.int(0))), ctx.rational(3, 7));
    assert_eq!(prob(&ctx, &d, |x| x.ge(&ctx.int(0))), ctx.int(1));
    assert_eq!(
        prob(&ctx, &d, |x| x.gt(&ctx.rational(1, 2))),
        ctx.rational(3, 7)
    );
    assert_eq!(
        prob(&ctx, &d, |x| x.le(&ctx.rational(1, 2))),
        ctx.rational(4, 7)
    );
    assert_eq!(
        prob(&ctx, &d, |x| x.eq_expr(&ctx.rational(1, 2))),
        ctx.int(0)
    );
    assert_eq!(d.cdf(&ctx.rational(1, 2)), ctx.rational(4, 7));
    assert_eq!(d.cdf(&ctx.int(-1)), ctx.int(0));
    assert_eq!(d.cdf(&ctx.int(1)), ctx.int(1));
    // scipy.stats.bernoulli(3/7).entropy() = 0.6829081047004717
    assert_close(&d.entropy(), 0.6829081047004717, "entropy");
    // mgf(ln 2) = 4/7 + 3/7·2 = 10/7
    assert_eq!(d.mgf(&ctx.int(2).ln()).simplify(), ctx.rational(10, 7));
    // scipy.stats.bernoulli(3/7).ppf(0.5) = 0.0 ; ppf(0.6) = 1.0
    assert_eq!(d.quantile_f64(0.5)?, 0.0);
    assert_eq!(d.quantile_f64(0.6)?, 1.0);
    check_sampling(&d, 20_000, 21);
    Ok(())
}

#[test]
fn binomial_nine_three_sevenths() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let d = Distribution::try_binomial(ctx.int(9), ctx.rational(3, 7))?;
    assert_eq!(d.mean(), ctx.rational(27, 7));
    assert_eq!(d.variance(), ctx.rational(108, 49));
    // Fraction: C(9,4)(3/7)^4(4/7)^5 = 1492992/5764801
    assert_eq!(
        d.density(&ctx.int(4)).simplify(),
        ctx.rational(1492992, 5764801)
    );
    // Fraction sums of the pmf
    assert_eq!(d.cdf(&ctx.int(4)), ctx.rational(3868672, 5764801));
    assert_eq!(
        prob(&ctx, &d, |x| x.gt(&ctx.int(2))),
        ctx.rational(4716225, 5764801)
    );
    assert_eq!(
        prob(&ctx, &d, |x| x.ge(&ctx.int(2))),
        ctx.rational(38321991, 40353607)
    );
    assert_eq!(
        prob(&ctx, &d, |x| x.gt(&ctx.int(2)).and(&x.le(&ctx.int(5)))),
        ctx.rational(3939840, 5764801)
    );
    // non-integer bound: P(X ≤ 7/2) = P(X ≤ 3)
    assert_eq!(
        prob(&ctx, &d, |x| x.le(&ctx.rational(7, 2))),
        ctx.rational(2375680, 5764801)
    );
    assert_eq!(d.cdf(&ctx.rational(7, 2)), ctx.rational(2375680, 5764801));
    // endpoints outside the support
    assert_eq!(
        prob(&ctx, &d, |x| x.gt(&ctx.int(-3)).and(&x.le(&ctx.int(20)))),
        ctx.int(1)
    );
    assert_eq!(prob(&ctx, &d, |x| x.gt(&ctx.int(9))), ctx.int(0));
    // E[X³] = 4077/49
    assert_eq!(d.moment(3), ctx.rational(4077, 49));
    // scipy.stats.binom(9, 3/7).ppf(0.3) = 3, ppf(0.9) = 6
    assert_eq!(d.quantile_f64(0.3)?, 3.0);
    assert_eq!(d.quantile_f64(0.9)?, 6.0);
    // scipy.stats.binom(9, 3/7).entropy() = 1.811459474644088
    assert_close(&d.entropy(), 1.811459474644088, "entropy");
    // mgf(t) = (4/7 + 3/7 eᵗ)⁹ at t = ln 2: (10/7)⁹
    assert_eq!(
        d.mgf(&ctx.int(2).ln()).simplify(),
        ctx.rational(10, 7).powi(9).simplify()
    );
    check_sampling(&d, 20_000, 22);
    Ok(())
}

#[test]
fn poisson_seven_thirds() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let d = Distribution::try_poisson(ctx.rational(7, 3))?;
    assert_eq!(d.mean(), ctx.rational(7, 3));
    assert_eq!(d.variance(), ctx.rational(7, 3));
    // scipy.stats.poisson(7/3).pmf(3) = 0.2053171912190799
    assert_close(&d.density(&ctx.int(3)), 0.2053171912190799, "pmf(3)");
    // poisson.cdf(3) = 0.7925363299535327 (also at 7/2)
    assert_close(&d.cdf(&ctx.int(3)), 0.7925363299535327, "cdf(3)");
    assert_close(&d.cdf(&ctx.rational(7, 2)), 0.7925363299535327, "cdf(7/2)");
    assert_close(
        &prob(&ctx, &d, |x| x.le(&ctx.rational(7, 2))),
        0.7925363299535327,
        "P(X ≤ 7/2)",
    );
    // poisson.sf(2) = 0.412780861265547 = P(X > 2); P(X ≥ 2) = 1 − P(0) − P(1)
    assert_close(
        &prob(&ctx, &d, |x| x.gt(&ctx.int(2))),
        0.412780861265547,
        "P(X > 2)",
    );
    let ge2 = prob(&ctx, &d, |x| x.ge(&ctx.int(2)));
    assert_eq!(
        ge2.equals(
            &(ctx.one()
                - ctx.rational(-7, 3).exp()
                - ctx.rational(7, 3) * ctx.rational(-7, 3).exp())
        ),
        Some(true)
    );
    // P(X > 2) + P(X = 2) = P(X ≥ 2), exactly
    let p2 = prob(&ctx, &d, |x| x.eq_expr(&ctx.int(2)));
    assert_eq!(
        (prob(&ctx, &d, |x| x.gt(&ctx.int(2))) + p2 - ge2).simplify(),
        ctx.int(0)
    );
    // Touchard: E[X³] = λ³ + 3λ² + λ = 847/27
    assert_eq!(d.moment(3), ctx.rational(847, 27));
    // scipy: poisson(7/3).cdf(0) = 0.09697196786440504 < 0.1, so ppf(0.1) = 1; ppf(0.3) = 1; ppf(0.95) = 5
    assert_eq!(d.quantile_f64(0.1)?, 1.0);
    assert_eq!(d.quantile_f64(0.3)?, 1.0);
    assert_eq!(d.quantile_f64(0.95)?, 5.0);
    // Below the support
    assert_eq!(prob(&ctx, &d, |x| x.lt(&ctx.int(0))), ctx.int(0));
    assert_eq!(d.cdf(&ctx.rational(-1, 2)), ctx.int(0));
    Ok(())
}

#[test]
fn geometric_three_sevenths() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let d = Distribution::try_geometric(ctx.rational(3, 7))?;
    assert_eq!(d.mean(), ctx.rational(7, 3));
    assert_eq!(d.variance(), ctx.rational(28, 9));
    // Fraction: (4/7)²·3/7 = 48/343 ; 1 − (4/7)³ = 279/343
    assert_eq!(d.density(&ctx.int(3)).simplify(), ctx.rational(48, 343));
    assert_eq!(d.cdf(&ctx.int(3)), ctx.rational(279, 343));
    assert_eq!(d.cdf(&ctx.rational(7, 2)), ctx.rational(279, 343));
    assert_eq!(
        prob(&ctx, &d, |x| x.le(&ctx.int(3))),
        ctx.rational(279, 343)
    );
    // P(X > 3) = (4/7)³ ; P(X ≥ 3) = (4/7)²
    assert_eq!(prob(&ctx, &d, |x| x.gt(&ctx.int(3))), ctx.rational(64, 343));
    assert_eq!(prob(&ctx, &d, |x| x.ge(&ctx.int(3))), ctx.rational(16, 49));
    // The support starts at 1: P(X = 0) = 0, P(X ≥ 1) = 1
    assert_eq!(prob(&ctx, &d, |x| x.eq_expr(&ctx.int(0))), ctx.int(0));
    assert_eq!(prob(&ctx, &d, |x| x.ge(&ctx.int(1))), ctx.int(1));
    // E[X²] = Var + mean² = 77/9
    assert_eq!(d.moment(2), ctx.rational(77, 9));
    // scipy.stats.geom(3/7).ppf(0.3) = 1, ppf(1/3) = 1, ppf(0.9) = 5
    assert_eq!(d.quantile_f64(0.3)?, 1.0);
    assert_eq!(d.quantile_f64(1.0 / 3.0)?, 1.0);
    assert_eq!(d.quantile_f64(0.9)?, 5.0);
    Ok(())
}

#[test]
fn negative_binomial_five_halves_three_sevenths() -> Result<(), SymplexError> {
    // scipy.stats.nbinom(n=2.5, p=3/7) — failures before the 2.5-th success
    let ctx = Context::new();
    let d = Distribution::try_negative_binomial(ctx.rational(5, 2), ctx.rational(3, 7))?;
    assert_eq!(d.mean(), ctx.rational(10, 3));
    assert_eq!(d.variance(), ctx.rational(70, 9));
    // nbinom.pmf(3) = 0.14723572768942417
    assert_close(&d.density(&ctx.int(3)), 0.14723572768942417, "pmf(3)");
    // nbinom.cdf(3) = 0.6110282699111106
    assert_close(&d.cdf(&ctx.int(3)), 0.6110282699111106, "cdf(3)");
    assert_close(
        &prob(&ctx, &d, |x| x.le(&ctx.int(3))),
        0.6110282699111106,
        "P(X ≤ 3)",
    );
    // nbinom.sf(3) = 0.3889717300888892
    assert_close(
        &prob(&ctx, &d, |x| x.gt(&ctx.int(3))),
        0.3889717300888892,
        "P(X > 3)",
    );
    // nbinom.ppf(0.3) = 2
    assert_eq!(d.quantile_f64(0.3)?, 2.0);
    // pmf(0) = p^r = (3/7)^{5/2} = 0.1202425109463631 (the symbolic form keeps an
    // unfolded `C(3/2, 0)`, so compare numerically)
    assert_close(&d.density(&ctx.int(0)), 0.1202425109463631, "pmf(0)");
    Ok(())
}

#[test]
fn hypergeometric_thirty_eleven_eight() -> Result<(), SymplexError> {
    // scipy.stats.hypergeom(M=30, n=11, N=8)
    let ctx = Context::new();
    let d = Distribution::try_hypergeometric(ctx.int(30), ctx.int(11), ctx.int(8))?;
    assert_eq!(d.mean(), ctx.rational(44, 15));
    assert_eq!(d.variance(), ctx.rational(9196, 6525));
    // Fraction: C(11,3)C(19,5)/C(30,8) = 14212/43355
    assert_eq!(
        d.density(&ctx.int(3)).simplify(),
        ctx.rational(14212, 43355)
    );
    assert_eq!(d.cdf(&ctx.int(3)), ctx.rational(89794, 130065));
    assert_eq!(
        prob(&ctx, &d, |x| x.gt(&ctx.int(3))),
        ctx.rational(40271, 130065)
    );
    assert_eq!(
        (prob(&ctx, &d, |x| x.ge(&ctx.int(3))) - prob(&ctx, &d, |x| x.gt(&ctx.int(3)))).simplify(),
        ctx.rational(14212, 43355)
    );
    // E[X²] = 1452/145
    assert_eq!(d.moment(2), ctx.rational(1452, 145));
    // hypergeom.ppf(0.3) = 2
    assert_eq!(d.quantile_f64(0.3)?, 2.0);
    // support: max(0, 8 + 11 − 30) ..= min(8, 11) = 0..=8
    assert_eq!(
        d.support(),
        Support::integers(&ctx, Some(ctx.int(0)), Some(ctx.int(8)))
    );
    assert_eq!(prob(&ctx, &d, |x| x.gt(&ctx.int(8))), ctx.int(0));
    check_sampling(&d, 20_000, 23);
    Ok(())
}

#[test]
fn discrete_uniform_minus_three_to_four() -> Result<(), SymplexError> {
    // scipy.stats.randint(-3, 5)
    let ctx = Context::new();
    let d = Distribution::try_discrete_uniform(ctx.int(-3), ctx.int(4))?;
    assert_eq!(d.mean(), ctx.rational(1, 2));
    assert_eq!(d.variance(), ctx.rational(21, 4));
    assert_eq!(d.density(&ctx.int(0)).simplify(), ctx.rational(1, 8));
    assert_eq!(d.cdf(&ctx.int(1)), ctx.rational(5, 8));
    assert_eq!(d.cdf(&ctx.rational(3, 2)), ctx.rational(5, 8));
    assert_eq!(
        prob(&ctx, &d, |x| x.gt(&ctx.int(-1)).and(&x.le(&ctx.int(2)))),
        ctx.rational(3, 8)
    );
    assert_eq!(
        prob(&ctx, &d, |x| x.ge(&ctx.int(-1)).and(&x.lt(&ctx.int(2)))),
        ctx.rational(3, 8)
    );
    assert_eq!(
        prob(&ctx, &d, |x| x.ge(&ctx.int(-1)).and(&x.le(&ctx.int(2)))),
        ctx.rational(1, 2)
    );
    // E[X³] = 8 (Faulhaber through the generic summation)
    assert_eq!(d.moment(3), ctx.int(8));
    // randint.ppf(0.3) = -1, ppf(0.25) = -2 (F(−2) = 2/8 exactly)
    assert_eq!(d.quantile_f64(0.3)?, -1.0);
    assert_eq!(d.quantile_f64(0.25)?, -2.0);
    assert_eq!(d.quantile(&ctx.rational(1, 4)), Some(ctx.int(-2)));
    // entropy = ln 8 = 3 ln 2
    assert_eq!(d.entropy().equals(&(3 * ctx.int(2).ln())), Some(true));
    check_sampling(&d, 20_000, 24);
    Ok(())
}

#[test]
fn finite_table_with_unsorted_values() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let d = Distribution::try_finite(
        &ctx,
        vec![
            (ctx.int(3), ctx.rational(1, 2)),
            (ctx.int(1), ctx.rational(1, 6)),
            (ctx.rational(5, 2), ctx.rational(1, 3)),
        ],
    )?;
    assert_eq!(d.mean(), ctx.rational(5, 2));
    // Var = E[X²] − 25/4 = (9/2 + 1/6 + 25/12) − 25/4 = 1/2
    assert_eq!(d.variance(), ctx.rational(1, 2));
    assert_eq!(d.cdf(&ctx.int(2)), ctx.rational(1, 6));
    assert_eq!(d.cdf(&ctx.rational(5, 2)), ctx.rational(1, 2));
    assert_eq!(d.cdf(&ctx.rational(11, 4)), ctx.rational(1, 2));
    assert_eq!(d.cdf(&ctx.int(3)), ctx.int(1));
    assert_eq!(
        prob(&ctx, &d, |x| x
            .gt(&ctx.int(1))
            .and(&x.le(&ctx.rational(5, 2)))),
        ctx.rational(1, 3)
    );
    assert_eq!(
        prob(&ctx, &d, |x| x.ge(&ctx.rational(5, 2))),
        ctx.rational(5, 6)
    );
    assert_eq!(
        prob(&ctx, &d, |x| x.gt(&ctx.rational(5, 2))),
        ctx.rational(1, 2)
    );
    assert_eq!(prob(&ctx, &d, |x| x.eq_expr(&ctx.int(2))), ctx.int(0));
    assert_eq!(d.quantile_f64(0.4)?, 2.5);
    assert_eq!(d.quantile_f64(0.5)?, 2.5);
    assert_eq!(d.quantile_f64(0.51)?, 3.0);
    // Duplicate values are rejected by the checked constructor.
    assert!(matches!(
        Distribution::try_finite(
            &ctx,
            vec![
                (ctx.int(1), ctx.rational(1, 2)),
                (ctx.int(1), ctx.rational(1, 2))
            ]
        ),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(Distribution::try_finite(&ctx, vec![(ctx.int(1), ctx.rational(1, 2))]).is_err());
    let s = d.sample(20_000, &mut Rng::new(25))?;
    assert!(s.iter().all(|v| [1.0, 2.5, 3.0].contains(v)));
    let mean = s.iter().sum::<f64>() / 20_000.0;
    assert!(
        (mean - 2.5).abs() < 4.0 * 0.5f64.sqrt() / 20_000f64.sqrt(),
        "mean {mean}"
    );
    Ok(())
}

#[test]
fn regression_finite_table_atom_at_a_non_integer_value_inside_a_lattice_rounded_region()
-> Result<(), SymplexError> {
    // Finite {1: 1/6, 5/2: 1/3, 3: 1/2}: P(1 < X ≤ 5/2) = 1/3.  The region used to
    // be rounded to the integer lattice ([2, 2]) before the listed values were
    // tested, dropping the atom at 5/2.
    let ctx = Context::new();
    let d = Distribution::try_finite(
        &ctx,
        vec![
            (ctx.int(1), ctx.rational(1, 6)),
            (ctx.rational(5, 2), ctx.rational(1, 3)),
            (ctx.int(3), ctx.rational(1, 2)),
        ],
    )?;
    assert_eq!(
        prob(&ctx, &d, |x| x
            .gt(&ctx.int(1))
            .and(&x.le(&ctx.rational(5, 2)))),
        ctx.rational(1, 3)
    );
    assert_eq!(
        prob(&ctx, &d, |x| x
            .gt(&ctx.rational(3, 2))
            .and(&x.lt(&ctx.rational(5, 2)))),
        ctx.int(0)
    );
    assert_eq!(
        prob(&ctx, &d, |x| x
            .ge(&ctx.rational(5, 2))
            .and(&x.lt(&ctx.int(3)))),
        ctx.rational(1, 3)
    );
    assert_eq!(
        prob(&ctx, &d, |x| x.gt(&ctx.rational(9, 4))),
        ctx.rational(5, 6)
    );
    // Conditioning on the same region works too: X | 1 < X ≤ 5/2 is the point mass at 5/2.
    let x = RandomVariable::new(&ctx, "X", d.clone());
    let given = x.given(
        &x.symbol()
            .gt(&ctx.int(1))
            .and(&x.symbol().le(&ctx.rational(5, 2))),
    )?;
    assert_eq!(given.mean(), ctx.rational(5, 2));
    assert_eq!(given.variance(), ctx.int(0));
    Ok(())
}

// ── wrappers and algebra ─────────────────────────────────────────────────

#[test]
fn affine_with_a_negative_scale_reverses_the_support() -> Result<(), SymplexError> {
    let ctx = Context::new();
    // Y = −3X + 1 for X ~ Exponential(2): support (−∞, 1], mean −1/2, variance 9/4.
    let y = Distribution::exponential(ctx.int(2)).affine(ctx.int(-3), ctx.int(1))?;
    let iv = y.support().as_interval().cloned().expect("one interval");
    assert_eq!(iv.lower, ctx.neg_infinity());
    assert_eq!(iv.upper, ctx.int(1));
    assert_eq!(iv.kind, IntervalKind::LeftOpen, "(−∞, 1]");
    assert_eq!(y.mean(), ctx.rational(-1, 2));
    assert_eq!(y.variance(), ctx.rational(9, 4));
    // P(Y ≤ −2) = P(X ≥ 1) = e⁻² = 0.1353352832366127
    assert_eq!(y.cdf(&ctx.int(-2)).equals(&ctx.int(-2).exp()), Some(true));
    assert_eq!(
        prob(&ctx, &y, |v| v.le(&ctx.int(-2))).equals(&ctx.int(-2).exp()),
        Some(true)
    );
    assert_eq!(y.cdf(&ctx.int(2)), ctx.int(1));
    assert_eq!(prob(&ctx, &y, |v| v.gt(&ctx.int(1))), ctx.int(0));
    // Q_Y(0.3) = 1 − 3·Q_X(0.7) = -0.8059592064889038
    assert_f64(y.quantile_f64(0.3)?, -0.8059592064889038, "quantile(0.3)");
    // H(Y) = H(X) + ln 3 = 1 − ln 2 + ln 3 = 1.4054651081081646
    assert_close(&y.entropy(), 1.4054651081081646, "entropy");
    // M_Y(t) = eᵗ·2/(2 + 3t) at t = 1/3: e^{1/3}·2/3
    assert_eq!(
        y.mgf(&ctx.rational(1, 3))
            .simplify()
            .equals(&(ctx.rational(2, 3) * ctx.rational(1, 3).exp())),
        Some(true)
    );
    check_sampling(&y, 20_000, 31);
    // A bounded support with a negative slope: −(1/2)·Uniform(1, 3) + 2 is Uniform(1/2, 3/2), closed.
    let u =
        Distribution::uniform(ctx.int(1), ctx.int(3)).affine(ctx.rational(-1, 2), ctx.int(2))?;
    let iv = u.support().as_interval().cloned().expect("one interval");
    assert_eq!(
        (iv.lower.clone(), iv.upper.clone()),
        (ctx.rational(1, 2), ctx.rational(3, 2))
    );
    assert_eq!(iv.kind, IntervalKind::Closed);
    assert_eq!(u.cdf(&ctx.int(1)), ctx.rational(1, 2));
    assert_eq!(
        prob(&ctx, &u, |v| v.gt(&ctx.rational(5, 4))),
        ctx.rational(1, 4)
    );
    // Lattice reflection: 5 − Binomial(5, 1/3) is Binomial(5, 2/3): P(Y = 2) = 40/243.
    let b =
        Distribution::binomial(ctx.int(5), ctx.rational(1, 3)).affine(ctx.int(-1), ctx.int(5))?;
    assert_eq!(
        b.support(),
        Support::integers(&ctx, Some(ctx.int(0)), Some(ctx.int(5)))
    );
    assert_eq!(
        prob(&ctx, &b, |v| v.eq_expr(&ctx.int(2))),
        ctx.rational(40, 243)
    );
    assert_eq!(b.mean(), ctx.rational(10, 3));
    assert_eq!(b.variance(), ctx.rational(10, 9));
    assert_eq!(b.cdf(&ctx.int(2)), ctx.rational(51, 243));
    Ok(())
}

#[test]
fn truncation_to_a_region_excluding_the_mode() -> Result<(), SymplexError> {
    // scipy.stats.truncnorm(1, 2): mean 1.3831690466315525, var 0.07274288610060176,
    // cdf(1.5) = 0.6758248057339608, ppf(0.3) = 1.1856325685206257
    let ctx = Context::new();
    let z = Distribution::normal(ctx.int(0), ctx.int(1));
    let t = z.truncated(&Support::interval(ctx.int(1), ctx.int(2)))?;
    assert_eq!(t.name(), "Truncated");
    assert_close(&t.mean(), 1.3831690466315525, "mean");
    assert_close(&t.variance(), 0.07274288610060176, "variance");
    assert_close(&t.cdf(&ctx.rational(3, 2)), 0.6758248057339608, "cdf(3/2)");
    assert_f64(t.quantile_f64(0.3)?, 1.1856325685206257, "ppf(0.3)");
    assert_eq!(t.cdf(&ctx.int(1)), ctx.int(0));
    assert_eq!(t.cdf(&ctx.int(2)), ctx.int(1));
    assert_eq!(t.cdf(&ctx.int(0)), ctx.int(0));
    assert_eq!(prob(&ctx, &t, |x| x.gt(&ctx.int(2))), ctx.int(0));
    // P(X ≤ 3/2 | 1 ≤ X ≤ 2) through the event route equals the cdf.
    assert_eq!(
        (prob(&ctx, &t, |x| x.le(&ctx.rational(3, 2))) - t.cdf(&ctx.rational(3, 2))).simplify(),
        ctx.int(0)
    );
    let s = t.sample(20_000, &mut Rng::new(32))?;
    assert!(s.iter().all(|v| (1.0..=2.0).contains(v)));
    let mean = s.iter().sum::<f64>() / 20_000.0;
    assert!(
        (mean - 1.3831690466315525).abs() < 4.0 * 0.07274288610060176f64.sqrt() / 20_000f64.sqrt()
    );
    // Beta(2, 3) | X > 1/2: mass 5/16, E[X | X > 1/2] = 16/25 (Fraction integral).
    let b = Distribution::beta(ctx.int(2), ctx.int(3));
    let half = ctx.rational(1, 2);
    let tb = b.truncated(&Support::from_pieces(
        symplex::stats::Kind::Continuous,
        vec![symplex::stats::Piece::Interval(Interval::open(
            half.clone(),
            ctx.infinity(),
        ))],
    ))?;
    assert_eq!(tb.mean(), ctx.rational(16, 25));
    assert_eq!(
        tb.parameters().last().map(|(_, m)| m.clone()),
        Some(ctx.rational(5, 16))
    );
    Ok(())
}

#[test]
fn truncation_to_a_single_lattice_point_and_to_empty_regions() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let b = Distribution::binomial(ctx.int(5), ctx.rational(1, 3));
    // X | X = 2 is the point mass at 2.
    let x = RandomVariable::new(&ctx, "X", b.clone());
    let at2 = x.given(&x.symbol().eq_expr(&ctx.int(2)))?;
    assert_eq!(at2.mean(), ctx.int(2));
    assert_eq!(at2.variance(), ctx.int(0));
    assert_eq!(
        at2.probability(&at2.symbol().eq_expr(&ctx.int(2)))?,
        ctx.int(1)
    );
    assert_eq!(at2.probability(&at2.symbol().gt(&ctx.int(2)))?, ctx.int(0));
    assert_eq!(
        at2.support(),
        Support::points(vec![ctx.int(2)]).with_kind(symplex::stats::Kind::Discrete)
    );
    // A lattice region between two integers is empty.
    assert!(matches!(
        x.given(&x.symbol().gt(&ctx.int(2)).and(&x.symbol().lt(&ctx.int(3)))),
        Err(SymplexError::InvalidArgument { .. })
    ));
    // Outside the support, and a null point of a density.
    let u = Distribution::uniform(ctx.int(0), ctx.int(1));
    assert!(matches!(
        u.truncated(&Support::interval(ctx.int(2), ctx.int(3))),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        u.truncated(&Support::interval(ctx.rational(1, 2), ctx.rational(1, 2))),
        Err(SymplexError::InvalidArgument { .. })
    ));
    // A region covering the whole support changes nothing.
    let e = Distribution::exponential(ctx.int(1));
    let same = e.truncated(&Support::half_line(ctx.int(-1)))?;
    assert_eq!(same.mean(), ctx.int(1));
    assert_eq!(
        same.cdf(&ctx.int(1))
            .equals(&(ctx.one() - ctx.int(-1).exp())),
        Some(true)
    );
    Ok(())
}

#[test]
fn mixture_with_a_zero_weight_component_and_a_discrete_mixture() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let a = Distribution::normal(ctx.int(0), ctx.int(1));
    let b = Distribution::normal(ctx.int(10), ctx.int(1));
    let m = Distribution::mixture(&[(ctx.int(1), a.clone()), (ctx.int(0), b.clone())])?;
    assert_eq!(m.mean(), ctx.int(0));
    assert_eq!(m.variance(), ctx.int(1));
    let v = ctx.symbol("v");
    assert_eq!((m.density(&v) - a.density(&v)).simplify(), ctx.int(0));
    assert_eq!(
        (m.cdf(&ctx.int(1)) - a.cdf(&ctx.int(1))).simplify(),
        ctx.int(0)
    );
    let s = m.sample(2000, &mut Rng::new(33))?;
    assert!(
        s.iter().all(|x| x.abs() < 6.0),
        "a zero-weight component must never be drawn"
    );
    // Weights must sum to one and be non-negative.
    assert!(
        Distribution::mixture(&[
            (ctx.rational(1, 2), a.clone()),
            (ctx.rational(1, 2), b.clone()),
            (ctx.int(1), a.clone())
        ])
        .is_err()
    );
    assert!(Distribution::mixture(&[(ctx.int(2), a.clone()), (ctx.int(-1), b.clone())]).is_err());
    // ½·Poisson(1) + ½·Poisson(3): mean 2, variance ½(1 + 1) + ½(3 + 9) − 4 = 3,
    // P(X = 2) = (e⁻¹/2 + 9e⁻³/2)/2 = 0.20399076412055445
    let p = Distribution::mixture(&[
        (ctx.rational(1, 2), Distribution::poisson(ctx.int(1))),
        (ctx.rational(1, 2), Distribution::poisson(ctx.int(3))),
    ])?;
    assert_eq!(p.mean(), ctx.int(2));
    assert_eq!(p.variance(), ctx.int(3));
    assert_close(
        &prob(&ctx, &p, |x| x.eq_expr(&ctx.int(2))),
        0.20399076412055445,
        "P(X = 2)",
    );
    // P(X ≤ 2) is the weighted sum of the components' cdfs; P(X > 2) is the complement.
    let le2 = prob(&ctx, &p, |x| x.le(&ctx.int(2)));
    let gt2 = prob(&ctx, &p, |x| x.gt(&ctx.int(2)));
    assert_eq!((le2 + gt2).simplify(), ctx.int(1));
    assert_eq!(
        p.support().pieces().len(),
        1,
        "identical lattice supports are listed once"
    );
    Ok(())
}

#[test]
fn transformed_by_decreasing_and_non_monotone_maps() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // e^{−X} for X ~ Exponential(1) is Uniform(0, 1): density 1 on (0, 1], mean 1/2.
    let e = Distribution::exponential(ctx.int(1));
    let u = e.transformed(&x, &(-&x).exp())?;
    assert_eq!(u.density(&y).simplify(), ctx.int(1));
    assert_eq!(u.mean(), ctx.rational(1, 2));
    assert_eq!(
        prob(&ctx, &u, |v| v.le(&ctx.rational(1, 2))),
        ctx.rational(1, 2)
    );
    let iv = u.support().as_interval().cloned().expect("interval");
    assert_eq!(
        (iv.lower.clone(), iv.upper.clone()),
        (ctx.int(0), ctx.int(1))
    );
    assert!(iv.kind.lower_open(), "e^{{−X}} never reaches 0");
    // −X³ for X ~ Uniform(0, 1): support [−1, 0], density (−y)^{−2/3}/3, mean −1/4.
    let cube = Distribution::uniform(ctx.int(0), ctx.int(1)).transformed(&x, &(-x.powi(3)))?;
    let iv = cube.support().as_interval().cloned().expect("interval");
    assert_eq!(
        (iv.lower.clone(), iv.upper.clone()),
        (ctx.int(-1), ctx.int(0))
    );
    assert_eq!(cube.mean(), ctx.rational(-1, 4));
    assert_eq!(
        prob(&ctx, &cube, |v| v.le(&ctx.rational(-1, 8))),
        ctx.rational(1, 2)
    );
    assert_close(
        &cube.density(&ctx.rational(-1, 8)),
        4.0 / 3.0,
        "density(−1/8) = (1/8)^{−2/3}/3",
    );
    // A non-monotone map that is not an even shape is refused honestly.
    assert!(matches!(
        Distribution::normal(ctx.int(0), ctx.int(1)).transformed(&x, &(x.powi(3) - &x)),
        Err(SymplexError::NotImplemented(_))
    ));
    assert!(matches!(
        Distribution::uniform(ctx.int(-1), ctx.int(1)).transformed(&x, &x.powi(3).sin()),
        Err(SymplexError::NotImplemented(_))
    ));
    // The sampler of a transform applies the map to inner samples: mean of e^{−X} ≈ 1/2.
    check_sampling(&u, 20_000, 34);
    check_sampling(&cube, 20_000, 35);
    Ok(())
}

#[test]
fn sums_of_independent_variables_in_closed_families() {
    use symplex::stats::sum_distribution;
    let ctx = Context::new();
    let rv = |name: &str, d: Distribution| RandomVariable::new(&ctx, name, d);
    // Normal(1/2, 3) + Normal(−2, 4) = Normal(−3/2, 5)
    let n1 = rv("N1", Distribution::normal(ctx.rational(1, 2), ctx.int(3)));
    let n2 = rv("N2", Distribution::normal(ctx.int(-2), ctx.int(4)));
    assert_eq!(
        sum_distribution(&n1, &n2),
        Some(Distribution::normal(ctx.rational(-3, 2), ctx.int(5)))
    );
    // Poisson(3/2) + Poisson(5/2) = Poisson(4)
    let p1 = rv("P1", Distribution::poisson(ctx.rational(3, 2)));
    let p2 = rv("P2", Distribution::poisson(ctx.rational(5, 2)));
    assert_eq!(
        sum_distribution(&p1, &p2),
        Some(Distribution::poisson(ctx.int(4)))
    );
    // Binomial(4, 2/5) + Binomial(6, 2/5) = Binomial(10, 2/5); Bernoulli(2/5) + Bernoulli(2/5) = Binomial(2, 2/5)
    let b1 = rv("B1", Distribution::binomial(ctx.int(4), ctx.rational(2, 5)));
    let b2 = rv("B2", Distribution::binomial(ctx.int(6), ctx.rational(2, 5)));
    assert_eq!(
        sum_distribution(&b1, &b2),
        Some(Distribution::binomial(ctx.int(10), ctx.rational(2, 5)))
    );
    let c1 = rv("C1", Distribution::bernoulli(ctx.rational(2, 5)));
    let c2 = rv("C2", Distribution::bernoulli(ctx.rational(2, 5)));
    assert_eq!(
        sum_distribution(&c1, &c2),
        Some(Distribution::binomial(ctx.int(2), ctx.rational(2, 5)))
    );
    // Gamma(3/2, 2/3) + Gamma(5/2, 2/3) = Gamma(4, 2/3); a different scale has no closed family
    let g1 = rv(
        "G1",
        Distribution::gamma(ctx.rational(3, 2), ctx.rational(2, 3)),
    );
    let g2 = rv(
        "G2",
        Distribution::gamma(ctx.rational(5, 2), ctx.rational(2, 3)),
    );
    let g3 = rv("G3", Distribution::gamma(ctx.int(1), ctx.int(1)));
    assert_eq!(
        sum_distribution(&g1, &g2),
        Some(Distribution::gamma(ctx.int(4), ctx.rational(2, 3)))
    );
    assert_eq!(sum_distribution(&g1, &g3), None);
    // Exponential(2) + Exponential(2) = Gamma(2, 1/2); Exponential(2) + Exponential(3/2) is not closed
    let e1 = rv("E1", Distribution::exponential(ctx.int(2)));
    let e2 = rv("E2", Distribution::exponential(ctx.int(2)));
    let e3 = rv("E3", Distribution::exponential(ctx.rational(3, 2)));
    assert_eq!(
        sum_distribution(&e1, &e2),
        Some(Distribution::gamma(ctx.int(2), ctx.rational(1, 2)))
    );
    assert_eq!(sum_distribution(&e1, &e3), None);
    // ChiSquared(2) + ChiSquared(3) = ChiSquared(5); Exponential(1/2) + ChiSquared(2) = Gamma(2, 2)
    let k1 = rv("K1", Distribution::chi_squared(ctx.int(2)));
    let k2 = rv("K2", Distribution::chi_squared(ctx.int(3)));
    assert_eq!(
        sum_distribution(&k1, &k2),
        Some(Distribution::chi_squared(ctx.int(5)))
    );
    let e4 = rv("E4", Distribution::exponential(ctx.rational(1, 2)));
    assert_eq!(
        sum_distribution(&e4, &k1),
        Some(Distribution::gamma(ctx.int(2), ctx.int(2)))
    );
    // The sum's moments agree with the joint expectation of the sum.
    let s = sum_distribution(&g1, &g2).expect("gamma sum");
    let sum = g1.symbol() + g2.symbol();
    assert_eq!(
        symplex::stats::expectation(&[&g1, &g2], &sum).expect("E"),
        s.mean()
    );
    assert_eq!(
        symplex::stats::variance(&[&g1, &g2], &sum).expect("Var"),
        s.variance()
    );
}

#[test]
fn covariance_correlation_and_conditional_expectation() -> Result<(), SymplexError> {
    use symplex::stats::{conditional_expectation, correlation, covariance};
    let ctx = Context::new();
    let u = RandomVariable::new(&ctx, "U", Distribution::uniform(ctx.int(0), ctx.int(1)));
    let us = u.symbol();
    // Cov(U, 2U) = 2·Var U = 1/6; Corr(U, 2U) = 1; Corr(U, 1 − 3U) = −1
    assert_eq!(covariance(&[&u], us, &(2 * us))?, ctx.rational(1, 6));
    assert_eq!(correlation(&[&u], us, &(2 * us))?, ctx.int(1));
    assert_eq!(correlation(&[&u], us, &(1 - 3 * us))?, ctx.int(-1));
    // Independent exponentials are uncorrelated; Corr(X + Y, X − Y) = (Var X − Var Y)/√(Var(X+Y)Var(X−Y))
    let x = RandomVariable::new(&ctx, "X", Distribution::exponential(ctx.int(2)));
    let y = RandomVariable::new(&ctx, "Y", Distribution::exponential(ctx.int(3)));
    let (xs, ys) = (x.symbol(), y.symbol());
    assert_eq!(covariance(&[&x, &y], xs, ys)?, ctx.int(0));
    assert_eq!(correlation(&[&x, &y], xs, ys)?, ctx.int(0));
    // Var X = 1/4, Var Y = 1/9: Cov(X+Y, X−Y) = 5/36, Var(X±Y) = 13/36 → Corr = 5/13
    assert_eq!(
        covariance(&[&x, &y], &(xs + ys), &(xs - ys))?,
        ctx.rational(5, 36)
    );
    assert_eq!(
        correlation(&[&x, &y], &(xs + ys), &(xs - ys))?,
        ctx.rational(5, 13)
    );
    // E[X | X > 1] = 1 + 1/2 (memorylessness), E[X² | X ≤ 1] for Uniform(0, 2) = 1/3
    assert_eq!(
        conditional_expectation(&x, xs, &xs.gt(&ctx.int(1)))?,
        ctx.rational(3, 2)
    );
    let w = RandomVariable::new(&ctx, "W", Distribution::uniform(ctx.int(0), ctx.int(2)));
    assert_eq!(
        conditional_expectation(&w, &w.symbol().powi(2), &w.symbol().le(&ctx.int(1)))?,
        ctx.rational(1, 3)
    );
    // Binomial(5, 1/3): E[B | B ≥ 3] = (3·40 + 4·10 + 5·1)/51 = 55/17
    let b = RandomVariable::new(
        &ctx,
        "B",
        Distribution::binomial(ctx.int(5), ctx.rational(1, 3)),
    );
    assert_eq!(
        conditional_expectation(&b, b.symbol(), &b.symbol().ge(&ctx.int(3)))?,
        ctx.rational(55, 17)
    );
    // E[B | B > 2] is the same event on the lattice; E[B | B > 5] has probability zero.
    assert_eq!(
        conditional_expectation(&b, b.symbol(), &b.symbol().gt(&ctx.int(2)))?,
        ctx.rational(55, 17)
    );
    assert!(conditional_expectation(&b, b.symbol(), &b.symbol().gt(&ctx.int(5))).is_err());
    Ok(())
}

#[test]
fn probability_of_x_at_most_t_equals_cdf_for_every_continuous_family() {
    let ctx = Context::new();
    let t = ctx.rational(7, 5);
    let families = [
        Distribution::normal(ctx.rational(-3, 2), ctx.rational(7, 4)),
        Distribution::uniform(ctx.int(-2), ctx.rational(5, 2)),
        Distribution::exponential(ctx.rational(3, 7)),
        Distribution::gamma(ctx.rational(7, 2), ctx.rational(3, 4)),
        Distribution::chi_squared(ctx.int(7)),
        Distribution::beta(ctx.rational(3, 7), ctx.rational(5, 2)),
        Distribution::cauchy(ctx.int(-1), ctx.rational(3, 4)),
        Distribution::laplace(ctx.int(2), ctx.rational(3, 7)),
        Distribution::logistic(ctx.rational(-1, 2), ctx.rational(5, 4)),
        Distribution::log_normal(ctx.rational(1, 4), ctx.rational(3, 5)),
        Distribution::student_t(ctx.rational(7, 2)),
        Distribution::f_distribution(ctx.int(5), ctx.int(9)),
        Distribution::weibull(ctx.rational(3, 2), ctx.rational(5, 2)),
        Distribution::pareto(ctx.rational(3, 2), ctx.rational(7, 2)),
        Distribution::triangular(ctx.int(-1), ctx.int(3), ctx.int(2)),
    ];
    for d in &families {
        let p = prob(&ctx, d, |x| x.le(&t));
        let c = d.cdf(&t);
        let diff = (&p - &c).simplify();
        assert!(
            diff.is_zero() == Some(true) || f64_of(&diff).abs() < 1e-12,
            "{d}: P(X ≤ 7/5) = `{p}` but cdf(7/5) = `{c}`"
        );
        // P(X < t) = P(X ≤ t) for a density.
        assert_eq!(
            (prob(&ctx, d, |x| x.lt(&t)) - &p).simplify(),
            ctx.int(0),
            "{d}: open end"
        );
        // Support::contains agrees with a positive density at an interior point.
        let inside = d.support().contains(&t);
        if inside == Some(true) {
            let f = f64_of(&d.density(&t));
            assert!(f > 0.0, "{d}: density at 7/5 should be positive, got {f}");
        }
        let p_of_all = prob(&ctx, d, |_| ctx.bool_true());
        assert_eq!(p_of_all, ctx.int(1), "{d}: P(true)");
    }
    // Beta(3/7, 5/2) does not contain 7/5; Pareto(3/2, 7/2) does not contain 1; Uniform(−2, 5/2) contains 5/2.
    assert_eq!(families[5].support().contains(&t), Some(false));
    assert_eq!(families[13].support().contains(&ctx.int(1)), Some(false));
    assert_eq!(
        families[1].support().contains(&ctx.rational(5, 2)),
        Some(true)
    );
}

#[test]
fn regression_transformed_by_a_cube_keeps_only_the_real_inverse_branch() -> Result<(), SymplexError>
{
    // The docs list `x³` among the recognised monotone maps, but `solve` returns
    // the three cube roots (two complex) and the single-branch check refused it.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let u = Distribution::uniform(ctx.int(0), ctx.int(1));
    // U³: density y^{−2/3}/3 on [0, 1], mean 1/4; SymPy: density(U**3)(y) = 1/(3*y**(2/3)), E = 1/4
    let cube = u.transformed(&x, &x.powi(3))?;
    assert_eq!(
        (cube.density(&y) - ctx.rational(1, 3) * y.pow(&ctx.rational(-2, 3))).simplify(),
        ctx.int(0)
    );
    assert_eq!(cube.mean(), ctx.rational(1, 4));
    assert_eq!(
        prob(&ctx, &cube, |v| v.le(&ctx.rational(1, 8))),
        ctx.rational(1, 2)
    );
    // 2U³ + 1 on [1, 3], mean 3/2
    let aff = u.transformed(&x, &(2 * x.powi(3) + 1))?;
    let iv = aff.support().as_interval().cloned().expect("interval");
    assert_eq!(
        (iv.lower.clone(), iv.upper.clone()),
        (ctx.int(1), ctx.int(3))
    );
    assert_eq!(aff.mean(), ctx.rational(3, 2));
    // U⁵: mean 1/6
    assert_eq!(u.transformed(&x, &x.powi(5))?.mean(), ctx.rational(1, 6));
    // Z³ for a standard normal: E = 0, E[(Z³)²] = E[Z⁶] = 15 by LOTUS
    let z = Distribution::normal(ctx.int(0), ctx.int(1)).transformed(&x, &x.powi(3))?;
    assert_eq!(z.mean(), ctx.int(0));
    assert_eq!(z.moment(2), ctx.int(15));
    Ok(())
}

// ── estimation ───────────────────────────────────────────────────────────

mod estimation_audit {
    use super::*;
    use symplex::linprog::{q, qi};
    use symplex::stats::estimation::{
        FamilyKind, aic, beta_binomial_posterior, bic, credible_interval,
        dirichlet_multinomial_posterior, fit, fit_log_normal, gamma_poisson_posterior,
        log_likelihood, method_of_moments, normal_known_variance_posterior,
        posterior_predictive_beta_binomial,
    };
    use symplex::stats::{
        Beta, Binomial, Gamma, Geometric, LogNormal, NegativeBinomial, Normal, Uniform,
    };

    #[test]
    fn mle_normal_exponential_poisson_on_fresh_data() -> Result<(), SymplexError> {
        let ctx = Context::new();
        // scipy.stats.norm.fit([1.5, 2.5, 4, 5.5, 7, 0.5]) = (3.5, 2.254624876411447); Fraction: mean 7/2, var 61/12
        let d = fit(
            &ctx,
            FamilyKind::Normal,
            &[q(3, 2), q(5, 2), qi(4), q(11, 2), qi(7), q(1, 2)],
        )?;
        let n = d.downcast_ref::<Normal>().expect("normal");
        assert_eq!(n.mean, ctx.rational(7, 2));
        assert_eq!(n.std.powi(2).simplify(), ctx.rational(61, 12));
        assert_close(&n.std, 2.254624876411447, "σ̂");
        // scipy: norm.logpdf(data, 3.5, sqrt(61/12)).sum() = -13.391532842383969
        let ll = log_likelihood(&d, &[q(3, 2), q(5, 2), qi(4), q(11, 2), qi(7), q(1, 2)]);
        assert_close(&ll, -13.391532842383969, "ℓ normal");
        // Exponential: λ̂ = n/Σx = 4/(41/6) = 24/41; scipy expon.fit(..., floc=0) = (0, 1.7083333333333335)
        let e = fit(
            &ctx,
            FamilyKind::Exponential,
            &[q(1, 2), q(3, 2), q(5, 2), q(7, 3)],
        )?;
        assert_eq!(e.mean(), ctx.rational(41, 24));
        // scipy: expon.logpdf([1, 2, 3], scale=2/3).sum() = -7.783604675675506
        let ll = log_likelihood(
            &Distribution::exponential(ctx.rational(3, 2)),
            &[qi(1), qi(2), qi(3)],
        );
        assert_close(&ll, -7.783604675675506, "ℓ exponential");
        assert_eq!(ll.equals(&(3 * ctx.rational(3, 2).ln() - 9)), Some(true));
        // Poisson: λ̂ = 17/7; scipy poisson.logpmf(counts, 17/7).sum() = -13.05944508846259
        let counts = [2, 0, 3, 5, 1, 4, 2].map(qi);
        let p = fit(&ctx, FamilyKind::Poisson, &counts)?;
        assert_eq!(p.mean(), ctx.rational(17, 7));
        assert_close(
            &log_likelihood(&p, &counts),
            -13.05944508846259,
            "ℓ poisson",
        );
        Ok(())
    }

    #[test]
    fn mle_bernoulli_binomial_geometric_uniform_log_normal() -> Result<(), SymplexError> {
        let ctx = Context::new();
        assert_eq!(
            fit(&ctx, FamilyKind::Bernoulli, &[1, 0, 1, 1, 0, 1, 1].map(qi))?.mean(),
            ctx.rational(5, 7)
        );
        let b = fit(
            &ctx,
            FamilyKind::Binomial { n: 6 },
            &[2, 3, 5, 1, 4].map(qi),
        )?;
        let b = b.downcast_ref::<Binomial>().expect("binomial");
        assert_eq!((b.n.clone(), b.p.clone()), (ctx.int(6), ctx.rational(1, 2)));
        let g = fit(&ctx, FamilyKind::Geometric, &[1, 3, 2, 5, 1, 4].map(qi))?;
        assert_eq!(
            g.downcast_ref::<Geometric>().expect("geometric").p,
            ctx.rational(3, 8)
        );
        // scipy: geom.logpmf([1, 3, 2], 1/3).sum() = -4.512232190328822
        assert_close(
            &log_likelihood(
                &Distribution::geometric(ctx.rational(1, 3)),
                &[qi(1), qi(3), qi(2)],
            ),
            -4.512232190328822,
            "ℓ geometric",
        );
        let u = fit(
            &ctx,
            FamilyKind::Uniform,
            &[q(-1, 2), qi(3), q(7, 4), q(5, 2)],
        )?;
        let u = u.downcast_ref::<Uniform>().expect("uniform");
        assert_eq!(
            (u.lo.clone(), u.hi.clone()),
            (ctx.rational(-1, 2), ctx.int(3))
        );
        // scipy: lognorm.fit([2, 3, 6, 9], floc=0) = (0.5855234655216107, 0, 4.242640687119285);
        // SymPy: μ̂ = log(2)/2 + log(3), σ̂² = log(2)²/4 − log(2)log(3)/2 + log(3)²/2
        let l = fit_log_normal(&ctx, &[2, 3, 6, 9].map(qi))?;
        let l = l.downcast_ref::<LogNormal>().expect("lognormal");
        assert_eq!(
            l.mu.equals(&(ctx.int(2).ln() / 2 + ctx.int(3).ln())),
            Some(true)
        );
        assert_close(&l.mu, 1.4451858789480825, "μ̂");
        assert_close(&l.sigma, 0.5855234655216107, "σ̂");
        // exp(μ̂) = 3√2 = 4.242640687119285 (scipy's `scale`)
        assert_close(&l.mu.exp(), 4.242640687119285, "e^μ̂");
        Ok(())
    }

    #[test]
    fn method_of_moments_on_fresh_data() -> Result<(), SymplexError> {
        let ctx = Context::new();
        // Fraction: [2, 3, 5, 10] → x̄ = 5, s² = 19/2 → k̂ = 50/19, θ̂ = 19/10
        let g = method_of_moments(&ctx, FamilyKind::Gamma, &[2, 3, 5, 10].map(qi))?;
        let g = g.downcast_ref::<Gamma>().expect("gamma");
        assert_eq!(
            (g.shape.clone(), g.scale.clone()),
            (ctx.rational(50, 19), ctx.rational(19, 10))
        );
        // [1/5, 3/10, 1/2, 4/5] → α̂ = 117/70, β̂ = 143/70
        let b = method_of_moments(
            &ctx,
            FamilyKind::Beta,
            &[q(1, 5), q(3, 10), q(1, 2), q(4, 5)],
        )?;
        let b = b.downcast_ref::<Beta>().expect("beta");
        assert_eq!(
            (b.alpha.clone(), b.beta.clone()),
            (ctx.rational(117, 70), ctx.rational(143, 70))
        );
        // [0, 1, 3, 7, 2, 5] → p̂ = 9/17, r̂ = 27/8
        let nb = method_of_moments(
            &ctx,
            FamilyKind::NegativeBinomial,
            &[0, 1, 3, 7, 2, 5].map(qi),
        )?;
        let nb = nb.downcast_ref::<NegativeBinomial>().expect("nb");
        assert_eq!(
            (nb.r.clone(), nb.p.clone()),
            (ctx.rational(27, 8), ctx.rational(9, 17))
        );
        // [1, 2, 3, 6] → x̄ = 3, 3s² = 21/2: a, b = 3 ∓ √(21/2)
        let u = method_of_moments(&ctx, FamilyKind::Uniform, &[1, 2, 3, 6].map(qi))?;
        let u = u.downcast_ref::<Uniform>().expect("uniform");
        assert_eq!((&u.lo + &u.hi).simplify(), ctx.int(6));
        assert_eq!((&u.hi - &u.lo).powi(2).simplify(), ctx.int(42));
        // [1, 2, 3, 6] → σ² = ln(1 + s²/x̄²) = ln(25/18), μ = ln 3 − σ²/2
        let l = method_of_moments(&ctx, FamilyKind::LogNormal, &[1, 2, 3, 6].map(qi))?;
        let l = l.downcast_ref::<LogNormal>().expect("lognormal");
        assert_eq!(
            l.sigma
                .powi(2)
                .simplify()
                .equals(&ctx.rational(25, 18).ln()),
            Some(true)
        );
        assert_eq!(
            l.mu.equals(&(ctx.int(3).ln() - ctx.rational(25, 18).ln() / 2)),
            Some(true)
        );
        // The moments are matched: the fitted family reproduces x̄ = 3 and s² = 7/2 exactly.
        let fitted = method_of_moments(&ctx, FamilyKind::LogNormal, &[1, 2, 3, 6].map(qi))?;
        assert_eq!(fitted.mean().simplify(), ctx.int(3));
        assert_eq!(fitted.variance().simplify(), ctx.rational(7, 2));
        // Constant data are refused; the Beta needs data in (0, 1).
        assert!(method_of_moments(&ctx, FamilyKind::Gamma, &[qi(2), qi(2)]).is_err());
        assert!(method_of_moments(&ctx, FamilyKind::Beta, &[q(1, 2), qi(1)]).is_err());
        Ok(())
    }

    #[test]
    fn aic_bic_and_conjugate_posteriors() -> Result<(), SymplexError> {
        let ctx = Context::new();
        // ℓ = −8.39444915467244 (Exp(1/3) on [1, 2, 3, 6]); k = 2, n = 4:
        // AIC = 4 + 16.78889830934488 = 20.78889830934488, BIC = 2 ln 4 + 16.78889830934488 = 19.56148703158466
        let ll = log_likelihood(
            &Distribution::exponential(ctx.rational(1, 3)),
            &[1, 2, 3, 6].map(qi),
        );
        assert_close(&aic(&ll, 2), 20.78889830934488, "AIC");
        assert_close(&bic(&ll, 2, 4), 19.56148703158466, "BIC");
        // Beta(3/2, 5/2) prior, 5 successes, 9 failures → Beta(13/2, 23/2);
        // scipy: beta.ppf([0.05, 0.95], 6.5, 11.5) = (0.18878844406386658, 0.5514564871893417)
        let post = beta_binomial_posterior(&ctx, &q(3, 2), &q(5, 2), 5, 9)?;
        let b = post.downcast_ref::<Beta>().expect("beta");
        assert_eq!(
            (b.alpha.clone(), b.beta.clone()),
            (ctx.rational(13, 2), ctx.rational(23, 2))
        );
        let ci = credible_interval(&post, 0.90)?;
        assert_f64(ci.lower, 0.18878844406386658, "beta ci lower");
        assert_f64(ci.upper, 0.5514564871893417, "beta ci upper");
        // Gamma(3/2, 2/3) prior, counts [3, 4, 5, 0] → Gamma(27/2, 2/11);
        // scipy: gamma.ppf([0.1, 0.9], 13.5, scale=2/11) = (1.646717815172362, 3.34011061343615)
        let post = gamma_poisson_posterior(&ctx, &q(3, 2), &q(2, 3), &[3, 4, 5, 0])?;
        let g = post.downcast_ref::<Gamma>().expect("gamma");
        assert_eq!(
            (g.shape.clone(), g.scale.clone()),
            (ctx.rational(27, 2), ctx.rational(2, 11))
        );
        let ci = credible_interval(&post, 0.80)?;
        assert_f64(ci.lower, 1.646717815172362, "gamma ci lower");
        assert_f64(ci.upper, 3.34011061343615, "gamma ci upper");
        // Prior N(1, 2), σ = 3, data [4, 6, 5, 7] → τ = 1/4 + 4/9, mean 97/25, variance 36/25;
        // scipy: norm.ppf([0.025, 0.975], 97/25, 6/5) = (1.5280432185519346, 6.231956781448065)
        let post =
            normal_known_variance_posterior(&ctx, &qi(1), &qi(2), &qi(3), &[4, 6, 5, 7].map(qi))?;
        let n = post.downcast_ref::<Normal>().expect("normal");
        assert_eq!(n.mean, ctx.rational(97, 25));
        assert_eq!(n.std, ctx.rational(6, 5));
        let ci = credible_interval(&post, 0.95)?;
        assert_f64(ci.lower, 1.5280432185519346, "normal ci lower");
        assert_f64(ci.upper, 6.231956781448065, "normal ci upper");
        // Dirichlet(1/2, 1/2, 1/2) with counts (2, 0, 5): (5/17, 1/17, 11/17)
        assert_eq!(
            dirichlet_multinomial_posterior(&[q(1, 2), q(1, 2), q(1, 2)], &[2, 0, 5])?,
            vec![q(5, 17), q(1, 17), q(11, 17)]
        );
        // Beta-binomial predictive, n = 5, α = 3/2, β = 5/2 (Fraction):
        // [429/2048, 495/2048, 225/1024, 175/1024, 225/2048, 99/2048]; mean nα/(α+β) = 15/8
        let pred = posterior_predictive_beta_binomial(&ctx, &q(3, 2), &q(5, 2), 5)?;
        let want = [
            (429, 2048),
            (495, 2048),
            (225, 1024),
            (175, 1024),
            (225, 2048),
            (99, 2048),
        ];
        for (k, (num, den)) in want.iter().enumerate() {
            assert_eq!(
                pred.density(&ctx.int(k as i64)).eval(),
                ctx.rational(*num, *den),
                "P(K = {k})"
            );
        }
        assert_eq!(pred.mean(), ctx.rational(15, 8));
        assert_eq!(
            prob(&ctx, &pred, |k| k.ge(&ctx.int(4))),
            ctx.rational(324, 2048)
        );
        // Validation
        assert!(credible_interval(&post, 1.0).is_err());
        assert!(beta_binomial_posterior(&ctx, &qi(0), &qi(1), 1, 1).is_err());
        assert!(gamma_poisson_posterior(&ctx, &qi(1), &qi(0), &[1]).is_err());
        assert!(normal_known_variance_posterior(&ctx, &qi(0), &qi(1), &qi(1), &[]).is_err());
        Ok(())
    }
}

// ── Markov chains ────────────────────────────────────────────────────────

mod markov_audit {
    use symplex::linprog::{q, qi};
    use symplex::prelude::*;
    use symplex::stats::markov::MarkovChain;

    fn chain(rows: Vec<Vec<symplex::stats::data::Q>>) -> MarkovChain {
        MarkovChain::new(QMatrix::new(rows).expect("matrix")).expect("chain")
    }

    /// States 0, 1 transient; 2, 3 absorbing.
    fn two_absorbing() -> MarkovChain {
        chain(vec![
            vec![q(1, 4), q(1, 4), q(1, 2), qi(0)],
            vec![q(1, 3), q(1, 6), qi(0), q(1, 2)],
            vec![qi(0), qi(0), qi(1), qi(0)],
            vec![qi(0), qi(0), qi(0), qi(1)],
        ])
    }

    #[test]
    fn four_state_chain_with_two_absorbing_states() -> Result<(), SymplexError> {
        let c = two_absorbing();
        assert_eq!(
            c.communication_classes(),
            vec![vec![0, 1], vec![2], vec![3]]
        );
        assert_eq!(c.closed_classes(), vec![vec![2], vec![3]]);
        assert_eq!(c.transient_states(), vec![0, 1]);
        assert_eq!(c.absorbing_states(), vec![2, 3]);
        assert!(c.is_absorbing_chain() && !c.is_irreducible() && c.is_aperiodic());
        assert_eq!(c.period_of(0)?, Some(1));
        assert_eq!(c.period_of(2)?, Some(1));
        // SymPy: N = (I − Q)⁻¹ = [[20/13, 6/13], [8/13, 18/13]], B = N R = [[10/13, 3/13], [4/13, 9/13]], t = (2, 2)
        let n = c.fundamental_matrix()?;
        assert_eq!(n.row(0), &[q(20, 13), q(6, 13)]);
        assert_eq!(n.row(1), &[q(8, 13), q(18, 13)]);
        let b = c.absorption_probabilities()?;
        assert_eq!(b.row(0), &[q(10, 13), q(3, 13)]);
        assert_eq!(b.row(1), &[q(4, 13), q(9, 13)]);
        assert_eq!(c.expected_steps_to_absorption()?, vec![qi(2), qi(2)]);
        assert_eq!(
            c.hitting_probability(&[3])?,
            vec![q(3, 13), q(9, 13), qi(0), qi(1)]
        );
        assert_eq!(
            c.expected_hitting_time(&[2, 3])?,
            vec![qi(2), qi(2), qi(0), qi(0)]
        );
        // State 1 reaches 2 with probability 4/13 < 1: the expected time is infinite.
        assert!(matches!(
            c.expected_hitting_time(&[2]),
            Err(SymplexError::ComputationFailed { .. })
        ));
        // Two closed classes: no unique stationary distribution, two extreme ones.
        assert!(matches!(
            c.stationary_distribution(),
            Err(SymplexError::InvalidArgument { .. })
        ));
        assert_eq!(
            c.stationary_distributions(),
            vec![
                vec![qi(0), qi(0), qi(1), qi(0)],
                vec![qi(0), qi(0), qi(0), qi(1)]
            ]
        );
        // SymPy: (P**3)[0, :] = [41/576, 31/576, 67/96, 17/96]
        assert_eq!(
            c.n_step(3).row(0),
            &[q(41, 576), q(31, 576), q(67, 96), q(17, 96)]
        );
        let after = c.distribution_after(&[q(1, 2), q(1, 2), qi(0), qi(0)], 3)?;
        let total = after.iter().fold(qi(0), |acc, v| acc + v);
        assert_eq!(total, qi(1));
        assert_eq!(after[2], (q(67, 96) + q(17, 72)) / qi(2));
        Ok(())
    }

    #[test]
    fn a_three_cycle_has_period_three() -> Result<(), SymplexError> {
        let c = chain(vec![
            vec![qi(0), qi(1), qi(0)],
            vec![qi(0), qi(0), qi(1)],
            vec![qi(1), qi(0), qi(0)],
        ]);
        assert!(c.is_irreducible() && c.is_ergodic());
        assert!(!c.is_aperiodic() && !c.is_regular());
        for s in 0..3 {
            assert_eq!(c.period_of(s)?, Some(3));
        }
        // SymPy: stationary distribution (1/3, 1/3, 1/3); P³ = I
        assert_eq!(
            c.stationary_distribution()?,
            vec![q(1, 3), q(1, 3), q(1, 3)]
        );
        assert_eq!(c.n_step(3), QMatrix::identity(3));
        assert_eq!(c.n_step(4), *c.transition_matrix());
        assert_eq!(c.mean_recurrence_times()?, vec![qi(3), qi(3), qi(3)]);
        assert_eq!(c.expected_hitting_time(&[2])?, vec![qi(2), qi(1), qi(0)]);
        // A cycle with a shortcut 0 → 2 has gcd(3, 2) = 1.
        let shortcut = chain(vec![
            vec![qi(0), q(1, 2), q(1, 2)],
            vec![qi(0), qi(0), qi(1)],
            vec![qi(1), qi(0), qi(0)],
        ]);
        assert_eq!(shortcut.period_of(0)?, Some(1));
        assert!(shortcut.is_regular());
        Ok(())
    }

    #[test]
    fn reducible_chain_reports_non_uniqueness_and_the_right_extreme_points()
    -> Result<(), SymplexError> {
        // Closed classes {0, 1} and {2}; state 3 transient.
        let c = chain(vec![
            vec![q(1, 2), q(1, 2), qi(0), qi(0)],
            vec![q(1, 3), q(2, 3), qi(0), qi(0)],
            vec![qi(0), qi(0), qi(1), qi(0)],
            vec![q(1, 4), q(1, 4), q(1, 4), q(1, 4)],
        ]);
        assert_eq!(
            c.communication_classes(),
            vec![vec![0, 1], vec![2], vec![3]]
        );
        assert_eq!(c.closed_classes(), vec![vec![0, 1], vec![2]]);
        assert_eq!(c.transient_states(), vec![3]);
        assert!(matches!(
            c.stationary_distribution(),
            Err(SymplexError::InvalidArgument { .. })
        ));
        // SymPy nullspace of P_Cᵀ − I on {0, 1}: (2/5, 3/5)
        assert_eq!(
            c.stationary_distributions(),
            vec![
                vec![q(2, 5), q(3, 5), qi(0), qi(0)],
                vec![qi(0), qi(0), qi(1), qi(0)]
            ]
        );
        // h₃ = 1/4 + h₃/4 → h₃ = 1/3 for reaching state 2; from the class {0, 1}: 0
        assert_eq!(
            c.hitting_probability(&[2])?,
            vec![qi(0), qi(0), qi(1), q(1, 3)]
        );
        assert!(
            !c.is_absorbing_chain(),
            "state 0 cannot reach the absorbing state 2"
        );
        assert!(c.fundamental_matrix().is_err());
        assert!(c.fundamental_matrix_ergodic().is_err());
        assert_eq!(c.period_of(3)?, Some(1));
        Ok(())
    }

    #[test]
    fn irreducible_three_state_chain_mean_first_passage() -> Result<(), SymplexError> {
        let c = chain(vec![
            vec![q(1, 5), q(3, 5), q(1, 5)],
            vec![q(1, 2), qi(0), q(1, 2)],
            vec![q(1, 3), q(1, 3), q(1, 3)],
        ]);
        // SymPy nullspace: π = (15/44, 7/22, 15/44)
        assert_eq!(
            c.stationary_distribution()?,
            vec![q(15, 44), q(7, 22), q(15, 44)]
        );
        assert!(c.is_regular());
        // SymPy Kemeny–Snell: M = [[0, 13/7, 16/5], [7/3, 0, 13/5], [8/3, 17/7, 0]]
        let m = c.mean_first_passage_times()?;
        assert_eq!(m.row(0), &[qi(0), q(13, 7), q(16, 5)]);
        assert_eq!(m.row(1), &[q(7, 3), qi(0), q(13, 5)]);
        assert_eq!(m.row(2), &[q(8, 3), q(17, 7), qi(0)]);
        // Agrees with the hitting-time system: k₀ = 16/5, k₁ = 13/5 for target {2}.
        assert_eq!(
            c.expected_hitting_time(&[2])?,
            vec![q(16, 5), q(13, 5), qi(0)]
        );
        assert_eq!(
            c.mean_recurrence_times()?,
            vec![q(44, 15), q(22, 7), q(44, 15)]
        );
        // SymPy: P² = [[61/150, 14/75, 61/150], [4/15, 7/15, 4/15], [31/90, 14/45, 31/90]]
        assert_eq!(c.n_step(2).row(1), &[q(4, 15), q(7, 15), q(4, 15)]);
        assert_eq!(c.n_step(2).row(2), &[q(31, 90), q(14, 45), q(31, 90)]);
        // π is a fixed point of P and of the stationary check on P²
        let pi = c.stationary_distribution()?;
        assert_eq!(c.distribution_after(&pi, 5)?, pi);
        Ok(())
    }
}

// ── information theory ───────────────────────────────────────────────────

mod information_audit {
    use super::*;
    use symplex::linprog::q;
    use symplex::stats::information::{
        Base, Given, Norm, conditional_entropy, cross_entropy, entropy, joint_entropy,
        js_divergence, kl_divergence, marginals, mutual_information, normalized_mutual_information,
        probability_vector, total_variation,
    };

    #[test]
    fn entropies_in_bits_and_nats_and_kl_with_a_zero_in_q() -> Result<(), SymplexError> {
        let ctx = Context::new();
        let p = [q(1, 2), q(1, 3), q(1, 6)];
        let u = [q(1, 4), q(1, 4), q(1, 2)];
        // scipy.stats.entropy([1/2, 1/3, 1/6]) = 1.0114042647073518; base=2: 1.459147917027245
        assert_close(
            &entropy(&ctx, &p, Base::Nats)?,
            1.0114042647073518,
            "H nats",
        );
        assert_close(&entropy(&ctx, &p, Base::Bits)?, 1.459147917027245, "H bits");
        // exact: H = ½ ln 2 + ⅓ ln 3 + ⅙ ln 6 = (2/3) ln 2 + (1/2) ln 3 (nats)
        assert_eq!(
            entropy(&ctx, &p, Base::Nats)?.equals(
                &(ctx.rational(2, 3) * ctx.int(2).ln() + ctx.rational(1, 2) * ctx.int(3).ln())
            ),
            Some(true)
        );
        // scipy.stats.entropy(p, u) = 0.25936556631921487 ; base=2: 0.37418541630608887
        assert_close(
            &kl_divergence(&ctx, &p, &u, Base::Nats)?,
            0.25936556631921487,
            "KL nats",
        );
        assert_close(
            &kl_divergence(&ctx, &p, &u, Base::Bits)?,
            0.37418541630608887,
            "KL bits",
        );
        // H(p, u) = H(p) + D(p‖u), exactly
        let ce = cross_entropy(&ctx, &p, &u, Base::Nats)?;
        assert_eq!(
            (ce - entropy(&ctx, &p, Base::Nats)? - kl_divergence(&ctx, &p, &u, Base::Nats)?)
                .simplify(),
            ctx.int(0)
        );
        // A zero in q where p > 0 is an error (the divergence is infinite), documented.
        let z = [q(1, 2), q(1, 2), q(0, 1)];
        assert!(matches!(
            kl_divergence(&ctx, &p, &z, Base::Nats),
            Err(SymplexError::InvalidArgument { .. })
        ));
        // …but a zero in p where q > 0 is fine (0·ln 0 = 0): D(z‖p) = ½ ln(1/2 / 1/2) + ½ ln(1/2 / 1/3) = ½ ln(3/2)
        assert_eq!(
            kl_divergence(&ctx, &z, &p, Base::Nats)?
                .equals(&(ctx.rational(1, 2) * ctx.rational(3, 2).ln())),
            Some(true)
        );
        assert_eq!(total_variation(&p, &u)?, q(1, 3));
        // A probability vector must sum to 1.
        assert!(entropy(&ctx, &[q(1, 2), q(1, 3)], Base::Nats).is_err());
        Ok(())
    }

    #[test]
    fn jensen_shannon_lies_in_zero_ln2() -> Result<(), SymplexError> {
        let ctx = Context::new();
        let p = [q(1, 2), q(1, 3), q(1, 6)];
        let u = [q(1, 4), q(1, 4), q(1, 2)];
        // scipy.spatial.distance.jensenshannon(p, u)**2 = 0.06782778870548349
        let js = js_divergence(&ctx, &p, &u, Base::Nats)?;
        assert_close(&js, 0.06782778870548349, "JS");
        assert_eq!(
            js_divergence(&ctx, &p, &p, Base::Nats)?.simplify(),
            ctx.int(0)
        );
        // Disjoint supports reach the bound ln 2 exactly (one bit).
        let a = [q(1, 1), q(0, 1)];
        let b = [q(0, 1), q(1, 1)];
        assert_eq!(js_divergence(&ctx, &a, &b, Base::Nats)?, ctx.int(2).ln());
        assert_eq!(js_divergence(&ctx, &a, &b, Base::Bits)?, ctx.int(1));
        // Symmetric, and finite even where KL is not.
        assert_eq!(
            (js_divergence(&ctx, &p, &u, Base::Nats)? - js_divergence(&ctx, &u, &p, Base::Nats)?)
                .simplify(),
            ctx.int(0)
        );
        let z = [q(1, 2), q(1, 2), q(0, 1)];
        let v = f64_of(&js_divergence(&ctx, &p, &z, Base::Nats)?);
        assert!(v > 0.0 && v < std::f64::consts::LN_2, "JS = {v}");
        Ok(())
    }

    #[test]
    fn mutual_information_matches_the_entropy_identity() -> Result<(), SymplexError> {
        let ctx = Context::new();
        let joint = vec![
            vec![q(1, 4), q(1, 8), q(1, 8)],
            vec![q(1, 8), q(1, 4), q(1, 8)],
        ];
        let m = marginals(&joint)?;
        assert_eq!(m.rows, vec![q(1, 2), q(1, 2)]);
        assert_eq!(m.cols, vec![q(3, 8), q(3, 8), q(1, 4)]);
        // scipy: H(X) = ln 2, H(Y) = 1.0821955300387673, H(X, Y) = 1.7328679513998633,
        // I = 0.04247475919884924 (0.061278124459132687 bits)
        let hx = entropy(&ctx, &m.rows, Base::Nats)?;
        let hy = entropy(&ctx, &m.cols, Base::Nats)?;
        let hxy = joint_entropy(&ctx, &joint, Base::Nats)?;
        assert_eq!(hx, ctx.int(2).ln());
        assert_close(&hy, 1.0821955300387673, "H(Y)");
        assert_close(&hxy, 1.7328679513998633, "H(X,Y)");
        let i = mutual_information(&ctx, &joint, Base::Nats)?;
        assert_close(&i, 0.04247475919884924, "I");
        assert_close(
            &mutual_information(&ctx, &joint, Base::Bits)?,
            0.061278124459132687,
            "I bits",
        );
        assert_eq!(
            (&hx + &hy - &hxy - &i).simplify(),
            ctx.int(0),
            "I = H(X) + H(Y) − H(X,Y)"
        );
        // H(Y | X) = H(X,Y) − H(X) = 1.039720770839918 ; H(X | Y) = 0.6506724213610959
        assert_close(
            &conditional_entropy(&ctx, &joint, Given::Row, Base::Nats)?,
            1.039720770839918,
            "H(Y|X)",
        );
        assert_close(
            &conditional_entropy(&ctx, &joint, Given::Column, Base::Nats)?,
            0.6506724213610959,
            "H(X|Y)",
        );
        // NMI variants are base-free and in [0, 1]: I/min(H) = I/ln 2
        let nmi = normalized_mutual_information(&ctx, &joint, Norm::Min)?;
        assert_close(
            &nmi,
            0.04247475919884924 / std::f64::consts::LN_2,
            "NMI min",
        );
        let nmi_max = f64_of(&normalized_mutual_information(&ctx, &joint, Norm::Max)?);
        assert!(nmi_max > 0.0 && nmi_max < f64_of(&nmi));
        // Independent margins: I = 0 exactly.
        let indep = vec![vec![q(1, 4), q(1, 4)], vec![q(1, 4), q(1, 4)]];
        assert_eq!(mutual_information(&ctx, &indep, Base::Nats)?, ctx.int(0));
        // A distribution's probability vector feeds these: a Binomial(9, 3/7) has 10 entries summing to 1.
        let pv = probability_vector(&Distribution::binomial(ctx.int(9), ctx.rational(3, 7)))?;
        assert_eq!(pv.len(), 10);
        assert_eq!(pv.iter().fold(q(0, 1), |acc, v| acc + v), q(1, 1));
        // scipy.stats.binom(9, 3/7).entropy() = 1.811459474644088
        assert_close(
            &entropy(&ctx, &pv, Base::Nats)?,
            1.811459474644088,
            "H(Binomial)",
        );
        Ok(())
    }
}

// ── multivariate normal and PCA ──────────────────────────────────────────

mod multivariate_audit {
    use super::*;
    use symplex::stats::multivariate::{MultivariateNormal, pca, pca_f64};

    fn mvn(ctx: &Context) -> MultivariateNormal {
        MultivariateNormal::try_new(
            vec![ctx.int(1), ctx.int(-2), ctx.int(3)],
            matrix![ctx, [4, 1, 0], [1, 2, 1], [0, 1, 3]],
        )
        .expect("valid")
    }

    #[test]
    fn density_precision_and_entropy_match_scipy() -> Result<(), SymplexError> {
        let ctx = Context::new();
        let m = mvn(&ctx);
        // scipy.stats.multivariate_normal([1,-2,3], cov).pdf([0,0,0]) = 7.290426264863305e-05
        assert_close(
            &m.density(&[ctx.int(0), ctx.int(0), ctx.int(0)])?,
            7.290426264863305e-05,
            "pdf(0)",
        );
        // pdf([2, -1, 4]) = 0.010819951931727542
        assert_close(
            &m.density(&[ctx.int(2), ctx.int(-1), ctx.int(4)])?,
            0.010819951931727542,
            "pdf(2,−1,4)",
        );
        // entropy = 5.673422271642127 ; det Σ = 17
        assert_close(&m.entropy()?, 5.673422271642127, "entropy");
        assert_eq!(
            m.entropy()?.equals(
                &(ctx.rational(1, 2) * ((ctx.int(2) * ctx.pi() * ctx.e()).powi(3) * 17).ln())
            ),
            Some(true)
        );
        // SymPy: Σ⁻¹ = [[5, −3, 1], [−3, 12, −4], [1, −4, 7]]/17
        let prec = m.precision()?;
        assert_eq!(prec.get(0, 0).simplify(), ctx.rational(5, 17));
        assert_eq!(prec.get(1, 1).simplify(), ctx.rational(12, 17));
        assert_eq!(prec.get(1, 2).simplify(), ctx.rational(-4, 17));
        // d²((2, −1, 4)) = (1, 1, 1) Σ⁻¹ (1, 1, 1)ᵀ = (5 −3 +1 −3 +12 −4 +1 −4 +7)/17 = 12/17
        assert_eq!(
            m.mahalanobis_squared(&[ctx.int(2), ctx.int(-1), ctx.int(4)])?,
            ctx.rational(12, 17)
        );
        assert!(m.density(&[ctx.int(0), ctx.int(0)]).is_err());
        Ok(())
    }

    #[test]
    fn marginals_and_conditionals_by_hand_fraction_linear_algebra() -> Result<(), SymplexError> {
        let ctx = Context::new();
        let m = mvn(&ctx);
        // Marginal (X₂, X₀): mean (3, 1), cov [[3, 0], [0, 4]]
        let mg = m.marginal(&[2, 0])?;
        assert_eq!(mg.mean, vec![ctx.int(3), ctx.int(1)]);
        assert_eq!(mg.cov, matrix![ctx, [3, 0], [0, 4]]);
        let x1 = m.marginal_1d(1)?;
        assert_eq!((x1.mean(), x1.variance()), (ctx.int(-2), ctx.int(2)));
        // SymPy Schur complement, X₁ = x₁: mean (x₁/2 + 2, x₁/2 + 4), cov [[7/2, −1/2], [−1/2, 5/2]]
        let s = ctx.symbol("x1");
        let c = m.conditional(&[(1, s.clone())])?;
        assert_eq!(c.mean[0].equals(&(&s / 2 + 2)), Some(true));
        assert_eq!(c.mean[1].equals(&(&s / 2 + 4)), Some(true));
        assert_eq!(c.cov[(0, 0)], ctx.rational(7, 2));
        assert_eq!(c.cov[(0, 1)], ctx.rational(-1, 2));
        assert_eq!(c.cov[(1, 1)], ctx.rational(5, 2));
        // X₁ | X₀ = 2, X₂ = 1: mean −29/12, variance 17/12
        let c2 = m.conditional(&[(0, ctx.int(2)), (2, ctx.int(1))])?;
        assert_eq!(c2.mean, vec![ctx.rational(-29, 12)]);
        assert_eq!(c2.cov[(0, 0)], ctx.rational(17, 12));
        // Conditioning on everything, or twice on the same index, is refused.
        assert!(
            m.conditional(&[(0, ctx.int(0)), (1, ctx.int(0)), (2, ctx.int(0))])
                .is_err()
        );
        assert!(m.conditional(&[(0, ctx.int(0)), (0, ctx.int(1))]).is_err());
        assert!(m.marginal(&[3]).is_err());
        // A singular covariance is refused by `try_new`; unchecked, its density fails.
        let singular = matrix![ctx, [1, 2], [2, 4]];
        assert!(matches!(
            MultivariateNormal::try_new(vec![ctx.int(0), ctx.int(0)], singular.clone()),
            Err(SymplexError::InvalidArgument { .. })
        ));
        let unchecked = MultivariateNormal::new(vec![ctx.int(0), ctx.int(0)], singular);
        assert!(unchecked.density(&[ctx.int(0), ctx.int(0)]).is_err());
        // Sampling is deterministic and has the right first moments.
        let s = m.sample(4000, &mut Rng::new(41))?;
        let mean1 = s.iter().map(|r| r[1]).sum::<f64>() / 4000.0;
        assert!(
            (mean1 + 2.0).abs() < 4.0 * 2f64.sqrt() / 4000f64.sqrt(),
            "mean of X₁ {mean1}"
        );
        assert_eq!(
            m.sample(3, &mut Rng::new(41))?,
            m.sample(3, &mut Rng::new(41))?
        );
        Ok(())
    }

    #[test]
    fn exact_pca_with_rational_eigenvalues() -> Result<(), SymplexError> {
        let ctx = Context::new();
        // numpy.linalg.eigvalsh([[5, 2], [2, 2]]) = [1, 6]; eigenvectors (2, 1)/√5 and (1, −2)/√5
        let p = pca(&ctx, &QMatrix::from_i64(&[&[5, 2], &[2, 2]])?)?;
        assert_eq!(p.eigenvalues, vec![ctx.int(6), ctx.int(1)]);
        assert_eq!(
            p.explained_variance_ratio,
            vec![ctx.rational(6, 7), ctx.rational(1, 7)]
        );
        let r5 = ctx.int(5).sqrt();
        assert_eq!(p.components[0][0].equals(&(ctx.int(2) / &r5)), Some(true));
        assert_eq!(p.components[0][1].equals(&(ctx.one() / &r5)), Some(true));
        assert_eq!(p.components[1][0].equals(&(ctx.one() / &r5)), Some(true));
        assert_eq!(p.components[1][1].equals(&(ctx.int(-2) / &r5)), Some(true));
        // numpy.linalg.eigvalsh([[2,0,0],[0,3,1],[0,1,3]]) = [2, 2, 4]
        let p3 = pca(
            &ctx,
            &QMatrix::from_i64(&[&[2, 0, 0], &[0, 3, 1], &[0, 1, 3]])?,
        )?;
        assert_eq!(p3.eigenvalues, vec![ctx.int(4), ctx.int(2), ctx.int(2)]);
        assert_eq!(
            p3.explained_variance_ratio,
            vec![ctx.rational(1, 2), ctx.rational(1, 4), ctx.rational(1, 4)]
        );
        let r2 = ctx.int(2).sqrt();
        assert_eq!(p3.components[0][0], ctx.int(0));
        assert_eq!(p3.components[0][1].equals(&(ctx.one() / &r2)), Some(true));
        assert_eq!(p3.components[0][2].equals(&(ctx.one() / &r2)), Some(true));
        // The two components of the eigenvalue 2 are orthonormal and orthogonal to (0, 1, 1)/√2.
        let dot = |a: &[Ex], b: &[Ex]| {
            a.iter()
                .zip(b)
                .fold(ctx.zero(), |acc, (x, y)| acc + x * y)
                .simplify()
        };
        assert_eq!(dot(&p3.components[1], &p3.components[2]), ctx.int(0));
        assert_eq!(dot(&p3.components[1], &p3.components[1]), ctx.int(1));
        assert_eq!(dot(&p3.components[2], &p3.components[2]), ctx.int(1));
        assert_eq!(dot(&p3.components[0], &p3.components[1]), ctx.int(0));
        // Σv = λv for each component
        let cov = matrix![ctx, [2, 0, 0], [0, 3, 1], [0, 1, 3]];
        for (lambda, v) in p3.eigenvalues.iter().zip(&p3.components) {
            let sv = cov.matmul(&Matrix::col_vector(v.clone()))?;
            for (i, vi) in v.iter().enumerate() {
                assert_eq!((sv.get(i, 0) - lambda * vi).simplify(), ctx.int(0));
            }
        }
        // An indefinite matrix is refused.
        assert!(pca(&ctx, &QMatrix::from_i64(&[&[1, 2], &[2, 1]])?).is_err());
        Ok(())
    }

    #[test]
    fn pca_f64_on_a_four_by_four_matches_numpy_eigh() -> Result<(), SymplexError> {
        // numpy.linalg.eigh([[6,2,1,0],[2,5,0,1],[1,0,4,1],[0,1,1,3]]):
        // eigenvalues 1.7824192906913454, 3.9999999999999996, 4.349764732621507, 7.867815976687146
        let cov = vec![
            vec![6.0, 2.0, 1.0, 0.0],
            vec![2.0, 5.0, 0.0, 1.0],
            vec![1.0, 0.0, 4.0, 1.0],
            vec![0.0, 1.0, 1.0, 3.0],
        ];
        let p = pca_f64(&cov)?;
        let want = [
            7.867815976687146,
            4.349764732621507,
            4.0,
            1.7824192906913454,
        ];
        for (got, w) in p.eigenvalues.iter().zip(want) {
            assert!((got - w).abs() < 1e-12, "eigenvalue {got} vs {w}");
        }
        // eigenvectors (numpy, arbitrary sign; here first non-zero coordinate positive)
        let v7 = [
            0.755464751776333,
            0.58596227664125,
            0.23914457990932333,
            0.16950247513508315,
        ];
        let v43 = [
            0.045379786992318226,
            0.3892312411162163,
            -0.85334980715328,
            -0.34385145412390145,
        ];
        let v4 = [
            0.5773502691896258,
            -0.5773502691896265,
            0.0,
            -0.5773502691896247,
        ];
        let v17 = [
            0.30639900525739555,
            -0.414504424535129,
            -0.4632536848546948,
            0.7209034297925253,
        ];
        for (got, w) in [
            (&p.components[0], v7),
            (&p.components[1], v43),
            (&p.components[2], v4),
            (&p.components[3], v17),
        ] {
            for (g, x) in got.iter().zip(w) {
                assert!((g - x).abs() < 1e-9, "component {got:?} vs {w:?}");
            }
        }
        let total: f64 = p.explained_variance_ratio.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
        assert!((p.explained_variance_ratio[0] - 7.867815976687146 / 18.0).abs() < 1e-12);
        // Not symmetric → error
        assert!(pca_f64(&[vec![1.0, 2.0], vec![0.0, 1.0]]).is_err());
        Ok(())
    }
}

// ── order statistics ─────────────────────────────────────────────────────

mod order_audit {
    use super::*;
    use symplex::stats::order::{maximum_of, minimum_of, order_statistic};

    #[test]
    fn kth_of_n_uniforms_is_beta() -> Result<(), SymplexError> {
        let ctx = Context::new();
        let u = Distribution::uniform(ctx.int(0), ctx.int(1));
        // X₍₃₎ of 7 ~ Beta(3, 5): mean 3/8, variance 15/(64·9) = 5/192, P(X ≤ 1/2) = I_{1/2}(3, 5) = 99/128
        let x3 = order_statistic(&u, 7, 3)?;
        let beta = Distribution::beta(ctx.int(3), ctx.int(5));
        assert_eq!(x3.mean(), ctx.rational(3, 8));
        assert_eq!(x3.variance(), ctx.rational(5, 192));
        assert_eq!(x3.cdf(&ctx.rational(1, 2)), ctx.rational(99, 128));
        assert_eq!(x3.cdf(&ctx.rational(1, 2)), beta.cdf(&ctx.rational(1, 2)));
        let t = ctx.rational(1, 3);
        assert_eq!((x3.density(&t) - beta.density(&t)).simplify(), ctx.int(0));
        assert_eq!(x3.moment(3), beta.moment(3));
        // n = 1: the parent itself; k = n: the maximum with mean n/(n+1)
        let same = order_statistic(&u, 1, 1)?;
        assert_eq!(same.mean(), ctx.rational(1, 2));
        assert_eq!(same.variance(), ctx.rational(1, 12));
        assert_eq!(same.cdf(&ctx.rational(1, 3)), ctx.rational(1, 3));
        let mx = maximum_of(&u, 9)?;
        assert_eq!(mx.mean(), ctx.rational(9, 10));
        assert_eq!(mx.cdf(&ctx.rational(1, 2)), ctx.rational(1, 512));
        assert_eq!(
            prob(&ctx, &mx, |x| x.gt(&ctx.rational(1, 2))),
            ctx.rational(511, 512)
        );
        // Outside the parent's support the cdf clamps.
        assert_eq!(mx.cdf(&ctx.int(2)), ctx.int(1));
        assert_eq!(mx.cdf(&ctx.int(-1)), ctx.int(0));
        let s = mx.sample(20_000, &mut Rng::new(51))?;
        let mean = s.iter().sum::<f64>() / 20_000.0;
        // Var = 9/(100·11)
        assert!(
            (mean - 0.9).abs() < 4.0 * (9.0f64 / 1100.0).sqrt() / 20_000f64.sqrt(),
            "mean {mean}"
        );
        Ok(())
    }

    #[test]
    fn minimum_of_exponentials_is_exponential_with_rate_n_lambda() -> Result<(), SymplexError> {
        let ctx = Context::new();
        let e = Distribution::exponential(ctx.rational(3, 7));
        let mn = minimum_of(&e, 4)?;
        let want = Distribution::exponential(ctx.rational(12, 7));
        assert_eq!(mn.mean(), ctx.rational(7, 12));
        assert_eq!(mn.variance(), ctx.rational(49, 144));
        assert_eq!(
            (mn.cdf(&ctx.int(1)) - want.cdf(&ctx.int(1))).simplify(),
            ctx.int(0)
        );
        let t = ctx.rational(2, 3);
        assert_eq!((mn.density(&t) - want.density(&t)).simplify(), ctx.int(0));
        assert_eq!(
            (prob(&ctx, &mn, |x| x.gt(&ctx.int(1))) - ctx.rational(-12, 7).exp()).simplify(),
            ctx.int(0)
        );
        // The maximum of 2 exponentials: E = 1/λ + 1/(2λ) = 7/2, P(max ≤ 1) = (1 − e^{−3/7})².
        // (The integrand x·(1 − e^{−λx})·e^{−λx} closes only after expansion; the generic
        // route used to hand back an unevaluated `Integral`.)
        let mx = maximum_of(&e, 2)?;
        assert_eq!(mx.mean(), ctx.rational(7, 2));
        // Max of 3: E = (7/3)(1 + 1/2 + 1/3) = 77/18, Var = (49/9)(1 + 1/4 + 1/9) = 2401/324
        let mx3 = maximum_of(&e, 3)?;
        assert_eq!(mx3.mean(), ctx.rational(77, 18));
        assert_eq!(mx3.variance(), ctx.rational(2401, 324));
        assert_eq!(
            mx.cdf(&ctx.int(1))
                .equals(&(ctx.one() - ctx.rational(-3, 7).exp()).powi(2)),
            Some(true)
        );
        Ok(())
    }

    #[test]
    fn order_statistics_of_discrete_parents() -> Result<(), SymplexError> {
        let ctx = Context::new();
        // Max of 3 draws from Binomial(4, 1/2) (Fraction enumeration):
        // [1/4096, 31/1024, 603/2048, 511/1024, 721/4096], mean 361/128
        let b = Distribution::binomial(ctx.int(4), ctx.rational(1, 2));
        let mx = maximum_of(&b, 3)?;
        assert_eq!(mx.name(), "Finite");
        let want = [(1, 4096), (31, 1024), (603, 2048), (511, 1024), (721, 4096)];
        for (k, (n, d)) in want.iter().enumerate() {
            assert_eq!(
                mx.density(&ctx.int(k as i64)).eval(),
                ctx.rational(*n, *d),
                "P(max = {k})"
            );
        }
        assert_eq!(mx.mean(), ctx.rational(361, 128));
        // The median of 3: [23/2048, 113/512, 549/1024, 113/512, 23/2048], mean 2 (symmetry)
        let med = order_statistic(&b, 3, 2)?;
        assert_eq!(med.density(&ctx.int(2)).eval(), ctx.rational(549, 1024));
        assert_eq!(med.density(&ctx.int(0)).eval(), ctx.rational(23, 2048));
        assert_eq!(med.mean(), ctx.int(2));
        assert_eq!(
            prob(&ctx, &med, |x| x.ge(&ctx.int(2))),
            ctx.rational(549, 1024) + ctx.rational(113, 512) + ctx.rational(23, 2048)
        );
        // n = 1 reproduces the parent table
        let same = order_statistic(&b, 1, 1)?;
        assert_eq!(same.mean(), ctx.int(2));
        assert_eq!(same.density(&ctx.int(1)).eval(), ctx.rational(1, 4));
        // Infinite lattice: min of 2 Geometric(1/3) is Geometric(5/9): pmf(1) = 5/9, pmf(2) = 20/81, P(min ≤ 2) = 65/81
        let g = Distribution::geometric(ctx.rational(1, 3));
        let mn = minimum_of(&g, 2)?;
        assert_eq!(mn.name(), "OrderStatistic");
        assert_eq!(mn.density(&ctx.int(1)).eval(), ctx.rational(5, 9));
        assert_eq!(mn.density(&ctx.int(2)).eval(), ctx.rational(20, 81));
        assert_eq!(mn.cdf(&ctx.int(2)), ctx.rational(65, 81));
        assert_eq!(prob(&ctx, &mn, |x| x.le(&ctx.int(2))), ctx.rational(65, 81));
        // Bad ranks
        assert!(order_statistic(&b, 3, 0).is_err());
        assert!(order_statistic(&b, 3, 4).is_err());
        assert!(order_statistic(&b, 0, 0).is_err());
        Ok(())
    }
}

// ── cross-module consistency and documented limits ───────────────────────

#[test]
fn sampling_routes_and_quantile_argument_validation() {
    let ctx = Context::new();
    // Families without a closed-form quantile (continuous) or on an infinite
    // lattice (discrete) sample through routes of their own since 0.18
    // (`tests/v18/v18_samplers.rs`); quantile_f64 works numerically for all.
    let own_route = [
        Distribution::gamma(ctx.int(2), ctx.int(3)),
        Distribution::beta(ctx.int(2), ctx.int(3)),
        Distribution::chi_squared(ctx.int(3)),
        Distribution::student_t(ctx.int(5)),
        Distribution::f_distribution(ctx.int(3), ctx.int(7)),
        Distribution::poisson(ctx.int(2)),
        Distribution::geometric(ctx.rational(1, 3)),
        Distribution::negative_binomial(ctx.int(3), ctx.rational(1, 3)),
    ];
    for d in &own_route {
        assert!(
            d.sample(1, &mut Rng::new(1)).is_ok(),
            "{d}: sampling should have a route"
        );
        let q = d
            .quantile_f64(0.4)
            .unwrap_or_else(|e| panic!("{d}: quantile_f64: {e}"));
        let back = f64_of(&d.cdf(&ctx.from_f64(q).expect("q")));
        assert!(back >= 0.4 - 1e-9, "{d}: F(Q(0.4)) = {back}");
    }
    let n = Distribution::normal(ctx.int(0), ctx.int(1));
    for p in [0.0, 1.0, 1.5, -0.1, f64::NAN] {
        assert!(
            matches!(n.quantile_f64(p), Err(SymplexError::InvalidArgument { .. })),
            "p = {p}"
        );
    }
    // A symbolic parameter cannot be sampled or inverted numerically.
    let mu = ctx.symbol("mu");
    let sym = Distribution::normal(mu, ctx.int(1));
    assert!(sym.sample(1, &mut Rng::new(1)).is_err());
    assert!(sym.quantile_f64(0.5).is_err());
    // The generator is deterministic.
    let a = n.sample(5, &mut Rng::new(99)).expect("sample");
    let b = n.sample(5, &mut Rng::new(99)).expect("sample");
    assert_eq!(a, b);
}

#[test]
fn mixture_of_different_supports_has_a_numeric_quantile() -> Result<(), SymplexError> {
    // ½·N(0, 1) + ½·Uniform(2, 3): support ℝ ∪ [2, 3].
    // mpmath: findroot(0.5·ncdf(m) + 0.5·(m − 2) − 0.5, 2.02) = 2.0216084112108001;
    // findroot(0.5·ncdf(m) − 0.2, −0.25) = -0.25334710313579974
    let ctx = Context::new();
    let m = Distribution::mixture(&[
        (
            ctx.rational(1, 2),
            Distribution::normal(ctx.int(0), ctx.int(1)),
        ),
        (
            ctx.rational(1, 2),
            Distribution::uniform(ctx.int(2), ctx.int(3)),
        ),
    ])?;
    assert_eq!(m.support().pieces().len(), 2);
    assert_f64(m.quantile_f64(0.5)?, 2.0216084112108, "median");
    assert_f64(m.quantile_f64(0.2)?, -0.2533471031357997, "q(0.2)");
    assert_eq!(m.mean(), ctx.rational(5, 4));
    // P(2 ≤ X ≤ 3) = ½·(Φ(3) − Φ(2)) + ½
    let p = prob(&ctx, &m, |x| x.ge(&ctx.int(2)).and(&x.le(&ctx.int(3))));
    // scipy: norm.cdf(3) − norm.cdf(2) = 0.021400233916549105
    assert_close(&p, 0.5 + 0.5 * 0.021400233916549105, "P(2 ≤ X ≤ 3)");
    // The density carries the indicator of the uniform piece:
    // scipy: norm.pdf(5) = 1.4867195147342979e-06, norm.pdf(2.5) = 0.01752830049356854
    let f5 = f64_of(&m.density(&ctx.int(5)));
    assert!(
        (f5 - 0.5 * 1.4867195147342979e-06).abs() < 1e-15,
        "density(5) = {f5}"
    );
    assert_close(
        &m.density(&ctx.rational(5, 2)),
        0.5 * 0.01752830049356854 + 0.5,
        "density(5/2)",
    );
    check_sampling(&m, 20_000, 61);
    Ok(())
}

#[test]
fn characteristic_functions_skewness_and_kurtosis() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let t = ctx.symbol("t");
    // Normal(−3/2, 7/4): φ(t) = exp(−3it/2 − 49t²/32)
    let n = Distribution::normal(ctx.rational(-3, 2), ctx.rational(7, 4));
    let want = (ctx.rational(-3, 2) * &i * &t - ctx.rational(49, 32) * t.powi(2)).exp();
    assert_eq!(
        (n.characteristic_function(&t) - want).simplify(),
        ctx.int(0)
    );
    // Exponential(3/7): φ(t) = λ/(λ − it)
    let e = Distribution::exponential(ctx.rational(3, 7));
    let want = ctx.rational(3, 7) / (ctx.rational(3, 7) - &i * &t);
    assert_eq!(
        (e.characteristic_function(&t) - want).simplify(),
        ctx.int(0)
    );
    // Poisson(7/3): φ(t) = exp(λ(e^{it} − 1))
    let p = Distribution::poisson(ctx.rational(7, 3));
    let want = (ctx.rational(7, 3) * ((&i * &t).exp() - 1)).exp();
    assert_eq!(
        (p.characteristic_function(&t) - want).simplify(),
        ctx.int(0)
    );
    // Skewness and kurtosis (not excess) in closed form:
    // Exponential 2 and 9; Poisson(λ) 1/√λ and 3 + 1/λ (scipy poisson(7/3).stats('sk') = (0.6546536707079771, 0.42857142857142855));
    // Uniform 0 and 9/5; Laplace 0 and 6; Logistic 0 and 21/5; Bernoulli(3/7) skew (1 − 2p)/√(p(1−p)) = √3/6
    assert_eq!(e.skewness(), ctx.int(2));
    assert_eq!(e.kurtosis(), ctx.int(9));
    assert_close(&p.skewness(), 0.6546536707079771, "Poisson skewness");
    assert_eq!(p.kurtosis().simplify(), ctx.rational(24, 7));
    let u = Distribution::uniform(ctx.int(-2), ctx.rational(5, 2));
    assert_eq!(u.skewness(), ctx.int(0));
    assert_eq!(u.kurtosis(), ctx.rational(9, 5));
    let l = Distribution::laplace(ctx.int(2), ctx.rational(3, 7));
    assert_eq!(l.skewness(), ctx.int(0));
    assert_eq!(l.kurtosis(), ctx.int(6));
    let lg = Distribution::logistic(ctx.rational(-1, 2), ctx.rational(5, 4));
    assert_eq!(lg.kurtosis().simplify(), ctx.rational(21, 5));
    let b = Distribution::bernoulli(ctx.rational(3, 7));
    assert_eq!(
        b.skewness().simplify().equals(&(ctx.int(3).sqrt() / 6)),
        Some(true)
    );
    // scipy.stats.gamma(3.5).stats('s') = 1.0690449676496976 = 2/√(7/2)
    let g = Distribution::gamma(ctx.rational(7, 2), ctx.rational(3, 4));
    assert_close(&g.skewness(), 1.0690449676496976, "Gamma skewness");
    Ok(())
}

#[test]
fn events_with_symbolic_bounds_and_unusual_shapes() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let e = RandomVariable::new(&ctx, "E", Distribution::exponential(ctx.rational(3, 7)));
    let es = e.symbol();
    // Symbolic bound on a half line: P(E > a) = e^{−3a/7} (the clip to [0, ∞) keeps max(0, a)
    // symbolic, so the answer is read off the CDF and is valid for a ≥ 0)
    let tail = e.probability(&es.gt(&a))?;
    assert_eq!(
        tail.subs(&a, &ctx.int(2))
            .simplify()
            .equals(&ctx.rational(-6, 7).exp()),
        Some(true)
    );
    // Contradictory conjunctions are empty; X = a ∧ X > a + 1 is empty.
    assert_eq!(
        e.probability(&es.gt(&ctx.int(3)).and(&es.lt(&ctx.int(2))))?,
        ctx.int(0)
    );
    assert_eq!(
        e.probability(&es.eq_expr(&ctx.int(1)).and(&es.gt(&ctx.int(2))))?,
        ctx.int(0)
    );
    // A disjunction with numeric bounds goes through the set machinery: P(E < 1 ∨ E > 2)
    let either = es.lt(&ctx.int(1)).or(&es.gt(&ctx.int(2)));
    assert_eq!(
        e.probability(&either)?
            .equals(&(ctx.one() - ctx.rational(-3, 7).exp() + ctx.rational(-6, 7).exp())),
        Some(true)
    );
    // |E − 1| < 1/2 is the interval (1/2, 3/2)
    let near = (es - 1).abs().lt(&ctx.rational(1, 2));
    assert_eq!(
        e.probability(&near)?
            .equals(&(ctx.rational(-3, 14).exp() - ctx.rational(-9, 14).exp())),
        Some(true)
    );
    // A disjunction with a symbolic bound is refused.
    assert!(matches!(
        e.probability(&es.lt(&a).or(&es.gt(&ctx.int(2)))),
        Err(SymplexError::NotImplemented(_))
    ));
    // On a lattice: B² ≤ 4 is B ∈ {0, 1, 2} for Binomial(5, 1/3): 192/243 = 64/81
    let b = RandomVariable::new(
        &ctx,
        "B",
        Distribution::binomial(ctx.int(5), ctx.rational(1, 3)),
    );
    assert_eq!(
        b.probability(&b.symbol().powi(2).le(&ctx.int(4)))?,
        ctx.rational(64, 81)
    );
    Ok(())
}

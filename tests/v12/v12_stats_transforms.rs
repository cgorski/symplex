//! symplex 0.12 — conditioning, affine maps, transformations and mixtures
//! of distributions (`stats::{Truncated, Affine, Transformed, Mixture}`,
//! `RandomVariable::{given, transform}`).  Reference values cite SymPy
//! 1.14 (`symplex/.venv/bin/python`); where SymPy fails or leaves an
//! integral, the value is derived and said so.

use symplex::prelude::*;
use symplex::stats::{Distribution, RandomVariable, Rng, Support};

fn assert_exact(actual: &Ex, expected: &Ex, label: &str) {
    assert_eq!(
        actual.equals(expected),
        Some(true),
        "{label}: got `{actual}`, expected `{expected}`"
    );
}

fn assert_close(actual: &Ex, expected: f64, label: &str) {
    let v = actual
        .eval_f64()
        .unwrap_or_else(|e| panic!("{label}: `{actual}` did not evaluate: {e}"));
    assert!(
        (v - expected).abs() < 1e-12 * expected.abs().max(1.0),
        "{label}: {v} vs {expected}"
    );
}

/// `(lo, ∞)`, open at `lo`.
fn open_half_line(ctx: &Context, lo: Ex) -> Support {
    Support::from_pieces(
        symplex::stats::Kind::Continuous,
        vec![symplex::stats::Piece::Interval {
            lo,
            hi: ctx.infinity(),
            lo_open: true,
            hi_open: true,
        }],
    )
}

fn standard_normal(ctx: &Context) -> RandomVariable {
    RandomVariable::new(ctx, "N", Distribution::normal(ctx.int(0), ctx.int(1)))
}

// ── given / Truncated ────────────────────────────────────────────────────

#[test]
fn half_normal_by_conditioning() -> Result<(), SymplexError> {
    // SymPy: E(given(N, N>0)) = sqrt(2)/sqrt(pi); variance = (pi - 2)/pi;
    // P(N < 1, N > 0) = erf(sqrt(2)/2).
    let ctx = Context::new();
    let n = standard_normal(&ctx);
    let half = n.given(&n.symbol().gt(&ctx.int(0)))?;
    assert_eq!(half.distribution().name(), "Truncated");
    assert_exact(
        &half.mean(),
        &(ctx.int(2) / ctx.pi()).sqrt(),
        "E[N | N > 0]",
    );
    assert_exact(
        &half.variance(),
        &(ctx.one() - ctx.int(2) / ctx.pi()),
        "Var[N | N > 0]",
    );
    assert_exact(
        &half.probability(&n.symbol().lt(&ctx.int(1)))?,
        &(ctx.int(2).sqrt() / 2).erf(),
        "P(N < 1 | N > 0)",
    );
    // The truncated support (open at 0: the event was strict) and density.
    assert_eq!(half.support(), open_half_line(&ctx, ctx.int(0)));
    let x = ctx.symbol("x");
    assert_exact(
        &half.density(&x),
        &(2 * n.density(&x)),
        "density doubles on the half-line",
    );
    // Total mass is one again.
    assert_exact(
        &half.probability(&ctx.bool_true())?,
        &ctx.one(),
        "P(true) = 1",
    );
    Ok(())
}

#[test]
fn conditioned_uniform_and_binomial() -> Result<(), SymplexError> {
    let ctx = Context::new();
    // SymPy: E(given(U, U > 1/2)) = 3/4, variance = 1/48.
    let u = RandomVariable::new(&ctx, "U", Distribution::uniform(ctx.int(0), ctx.int(1)));
    let uh = u.given(&u.symbol().gt(&ctx.rational(1, 2)))?;
    assert_exact(&uh.mean(), &ctx.rational(3, 4), "E[U | U > ½]");
    assert_exact(&uh.variance(), &ctx.rational(1, 48), "Var[U | U > ½]");
    // The conditional cdf and quantile are transported closed forms.
    assert_exact(
        &uh.cdf(&ctx.rational(3, 4)),
        &ctx.rational(1, 2),
        "F(¾ | U > ½)",
    );
    assert_exact(
        &uh.quantile(&ctx.rational(1, 2)).unwrap(),
        &ctx.rational(3, 4),
        "median of U | U > ½",
    );
    // SymPy: E(given(B, B >= 2)) = 325/131, P(Eq(B, 3), B >= 2) = 40/131
    // for B ~ Binomial(5, 1/3).
    let b = RandomVariable::new(
        &ctx,
        "B",
        Distribution::binomial(ctx.int(5), ctx.rational(1, 3)),
    );
    let bg = b.given(&b.symbol().ge(&ctx.int(2)))?;
    assert_exact(&bg.mean(), &ctx.rational(325, 131), "E[B | B ≥ 2]");
    assert_exact(
        &bg.probability(&b.symbol().eq_expr(&ctx.int(3)))?,
        &ctx.rational(40, 131),
        "P(B = 3 | B ≥ 2)",
    );
    // Conditioning on a set of probability zero is an error.
    assert!(matches!(
        standard_normal(&ctx).given(&ctx.symbol("N").eq_expr(&ctx.int(0))),
        Err(SymplexError::InvalidArgument { .. })
    ));
    Ok(())
}

#[test]
fn conditioning_on_a_non_linear_event_uses_the_set_machinery() -> Result<(), SymplexError> {
    // SymPy: E(given(V, V**2 < 1/4)) = 0 for V ~ Uniform(-1, 1).
    let ctx = Context::new();
    let v = RandomVariable::new(&ctx, "V", Distribution::uniform(ctx.int(-1), ctx.int(1)));
    let inner = v.given(&v.symbol().powi(2).lt(&ctx.rational(1, 4)))?;
    assert_eq!(inner.mean(), ctx.int(0));
    assert_exact(
        &inner.variance(),
        &ctx.rational(1, 12),
        "Var of Uniform(−½, ½)",
    );
    // Two half-lines: P(N² < 1) and P(N > 1 ∨ N < −1) sum to one.
    let n = standard_normal(&ctx);
    let inside = n.probability(&n.symbol().powi(2).lt(&ctx.int(1)))?;
    let outside = n.probability(&n.symbol().gt(&ctx.int(1)).or(&n.symbol().lt(&ctx.int(-1))))?;
    assert_exact(
        &(inside + outside).simplify(),
        &ctx.one(),
        "complementary events",
    );
    // SymPy: P(N**2 < 1) = erf(sqrt(2)/2)  (≈ 0.6826894921370859)
    assert_close(
        &n.probability(&n.symbol().powi(2).lt(&ctx.int(1)))?,
        0.682_689_492_137_085_9,
        "P(N² < 1)",
    );
    Ok(())
}

// ── Affine ───────────────────────────────────────────────────────────────

#[test]
fn affine_map_of_a_normal_is_a_normal() -> Result<(), SymplexError> {
    // SymPy: density(2*N + 1)(x) = sqrt(2)*exp(-(x/2 - 1/2)**2/2)/(4*sqrt(pi)),
    // E = 1, variance = 4.
    let ctx = Context::new();
    let n = standard_normal(&ctx);
    let w = n.transform("W", &(2 * n.symbol() + 1))?;
    assert_eq!(w.distribution().name(), "Affine");
    let x = ctx.symbol("x");
    assert_exact(
        &w.density(&x),
        &(ctx.int(2).sqrt() * (-((&x / 2 - ctx.rational(1, 2)).powi(2)) / 2).exp()
            / (4 * ctx.pi().sqrt())),
        "density of 2N + 1",
    );
    assert_eq!(w.mean(), ctx.int(1));
    assert_eq!(w.variance(), ctx.int(4));
    // Transported closed forms agree with Normal(1, 2).
    let direct = Distribution::normal(ctx.int(1), ctx.int(2));
    let t = ctx.symbol("t");
    assert_exact(&w.mgf(&t), &direct.mgf(&t), "mgf");
    assert_exact(&w.cdf(&ctx.int(3)), &direct.cdf(&ctx.int(3)), "cdf(3)");
    assert_exact(
        &w.quantile(&ctx.rational(3, 4)).unwrap(),
        &direct.quantile(&ctx.rational(3, 4)).unwrap(),
        "quantile(¾)",
    );
    assert_exact(&w.moment(3), &direct.moment(3), "E[W³] = 1 + 3·4 = 13");
    assert_eq!(w.moment(3), ctx.int(13));
    // A decreasing map: SymPy P(-N > -1) = 1 - erfc(sqrt(2)/2)/2 = ½ + ½ erf(√2/2).
    let m = n.transform("M", &(-n.symbol()))?;
    assert_exact(
        &m.probability(&m.symbol().gt(&ctx.int(-1)))?,
        &(ctx.rational(1, 2) + (ctx.int(2).sqrt() / 2).erf() / 2),
        "P(−N > −1)",
    );
    // The slope's sign must be known.
    let a = ctx.symbol("a");
    assert!(matches!(
        n.distribution().affine(a, ctx.int(0)),
        Err(SymplexError::InvalidArgument { .. })
    ));
    Ok(())
}

// ── Transformed ──────────────────────────────────────────────────────────

#[test]
fn square_and_absolute_value_of_a_normal() -> Result<(), SymplexError> {
    // SymPy: density(N**2)(y) = sqrt(2)*exp(-y/2)/(2*sqrt(pi)*sqrt(y)); E = 1;
    // variance = 2.  E(Abs(N)) = sqrt(2)/sqrt(pi); variance(Abs(N)) = (pi - 2)/pi
    // (SymPy cannot form density(Abs(N)) — "Can not solve Abs(N) for N").
    let ctx = Context::new();
    let n = standard_normal(&ctx);
    let y = ctx.symbol("y");
    let sq = n.transform("S", &n.symbol().powi(2))?;
    assert_eq!(sq.distribution().name(), "Transformed");
    assert_eq!(sq.support(), Support::half_line(ctx.int(0)));
    assert_exact(
        &sq.density(&y),
        &(ctx.int(2).sqrt() * (-&y / 2).exp() / (2 * ctx.pi().sqrt() * y.sqrt())),
        "χ²(1) density",
    );
    assert_eq!(sq.mean(), ctx.int(1));
    assert_eq!(sq.variance(), ctx.int(2));
    let ab = n.transform("A", &n.symbol().abs())?;
    assert_exact(&ab.density(&y), &(2 * n.density(&y)), "half-normal density");
    assert_exact(&ab.mean(), &(ctx.int(2) / ctx.pi()).sqrt(), "E|N|");
    assert_exact(
        &ab.variance(),
        &(ctx.one() - ctx.int(2) / ctx.pi()),
        "Var|N|",
    );
    Ok(())
}

#[test]
fn monotone_maps_by_change_of_variables() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let y = ctx.symbol("y");
    // SymPy: density(exp(N))(y) = sqrt(2)*exp(-log(y)**2/2)/(2*sqrt(pi)*y);
    // E(exp(N)) = e^{1/2} (SymPy leaves an unsimplified erf expression that
    // evaluates to 1.6487212707001282).
    let n = standard_normal(&ctx);
    let en = n.transform("E", &n.symbol().exp())?;
    assert_exact(
        &en.density(&y),
        &(ctx.int(2).sqrt() * (-(y.ln().powi(2)) / 2).exp() / (2 * ctx.pi().sqrt() * &y)),
        "log-normal density",
    );
    assert_exact(&en.mean(), &ctx.rational(1, 2).exp(), "E[eᴺ] = √e");
    assert_eq!(en.support(), open_half_line(&ctx, ctx.int(0)));
    // SymPy: density(exp(-Ex))(y) = 2*y on (0, 1], E = 2/3 for Ex ~ Exponential(2).
    let ex = RandomVariable::new(&ctx, "Ex", Distribution::exponential(ctx.int(2)));
    let f = ex.transform("F", &(-ex.symbol()).exp())?;
    assert_exact(&f.density(&y), &(2 * &y), "density of e^{−Ex}");
    assert_eq!(f.mean(), ctx.rational(2, 3));
    // SymPy: density(1/U)(y) = 1/y² on [1, ∞); P(1/U > 2) = 1/2.
    let u = RandomVariable::new(&ctx, "U", Distribution::uniform(ctx.int(0), ctx.int(1)));
    let iu = u.transform("I", &(1 / u.symbol()))?;
    assert_eq!(iu.support(), Support::half_line(ctx.int(1)));
    assert_exact(&iu.density(&y), &y.powi(-2), "density of 1/U");
    assert_eq!(
        iu.probability(&iu.symbol().gt(&ctx.int(2)))?,
        ctx.rational(1, 2)
    );
    // A map that is not monotone and not an even shape is refused honestly.
    assert!(matches!(
        n.transform("Z", &n.symbol().sin()),
        Err(SymplexError::NotImplemented(_))
    ));
    Ok(())
}

#[test]
fn transforming_a_finite_range_maps_and_merges_values() -> Result<(), SymplexError> {
    // SymPy: density(D**2) = {1: 1/6, 4: 1/6, 9: 1/6, 16: 1/6, 25: 1/6, 36: 1/6}, E = 91/6.
    let ctx = Context::new();
    let d = RandomVariable::new(&ctx, "D", Distribution::die(ctx.int(6)));
    let d2 = d.transform("D2", &d.symbol().powi(2))?;
    assert_eq!(d2.distribution().name(), "Finite");
    assert_eq!(d2.mean(), ctx.rational(91, 6));
    assert_eq!(
        d2.probability(&d2.symbol().eq_expr(&ctx.int(9)))?,
        ctx.rational(1, 6)
    );
    // Merging: (D − 3)² takes the value 1 for D = 2 and D = 4.
    let m = d.transform("M", &(d.symbol() - 3).powi(2))?;
    assert_eq!(
        m.probability(&m.symbol().eq_expr(&ctx.int(1)))?,
        ctx.rational(1, 3)
    );
    assert_eq!(
        m.probability(&m.symbol().eq_expr(&ctx.int(0)))?,
        ctx.rational(1, 6)
    );
    Ok(())
}

// ── Mixture ──────────────────────────────────────────────────────────────

#[test]
fn mixture_of_two_normals() -> Result<(), SymplexError> {
    // ¼·Normal(−1, 1) + ¾·Normal(3, 2): mean ¼·(−1) + ¾·3 = 2;
    // E[X²] = ¼·(1 + 1) + ¾·(4 + 9) = 41/4; P(X > 0) = ¼·P(N₁ > 0) + ¾·P(N₂ > 0).
    let ctx = Context::new();
    let a = Distribution::normal(ctx.int(-1), ctx.int(1));
    let b = Distribution::normal(ctx.int(3), ctx.int(2));
    let m = Distribution::mixture(&[
        (ctx.rational(1, 4), a.clone()),
        (ctx.rational(3, 4), b.clone()),
    ])?;
    let x = RandomVariable::new(&ctx, "X", m);
    assert_eq!(x.mean(), ctx.int(2));
    assert_eq!(x.moment(2), ctx.rational(41, 4));
    assert_eq!(x.variance(), ctx.rational(25, 4));
    let zero = ctx.int(0);
    let region = Support::half_line(zero.clone());
    let expected = (ctx.rational(1, 4) * a.probability_of(&region)?
        + ctx.rational(3, 4) * b.probability_of(&region)?)
    .simplify();
    assert_exact(
        &x.probability(&x.symbol().gt(&zero))?,
        &expected,
        "P(X > 0) is the weighted sum",
    );
    // P(N(−1,1) > 0) = 1 − Φ(1), P(N(3,2) > 0) = Φ(1.5).  SymPy:
    // N((1+erf(1/sqrt(2)))/2, 20) = 0.84134474606854294859 = Φ(1),
    // N((1+erf(1.5/sqrt(2)))/2, 20) = 0.93319279873114193400 = Φ(1.5).
    assert_close(
        &x.probability(&x.symbol().gt(&zero))?,
        0.25 * (1.0 - 0.8413447460685429) + 0.75 * 0.9331927987311419,
        "P(X > 0)",
    );
    // The cdf sums the components' clamped cdfs; the density is a plain sum
    // when the supports coincide.
    let v = ctx.symbol("v");
    assert_exact(
        &x.density(&v),
        &(ctx.rational(1, 4) * a.density(&v) + ctx.rational(3, 4) * b.density(&v)),
        "density",
    );
    // Sampling picks a component, then samples it.
    let mut rng = Rng::new(11);
    let s = x.sample(20_000, &mut rng)?;
    let mean = s.iter().sum::<f64>() / s.len() as f64;
    assert!((mean - 2.0).abs() < 0.06, "sample mean {mean}");
    // Validation.
    assert!(Distribution::mixture(&[]).is_err());
    assert!(
        Distribution::mixture(&[
            (ctx.rational(1, 2), a.clone()),
            (ctx.rational(1, 3), b.clone())
        ])
        .is_err(),
        "weights must sum to 1"
    );
    assert!(
        Distribution::mixture(&[(ctx.one(), Distribution::die(ctx.int(6)))]).is_ok()
            && Distribution::mixture(&[
                (ctx.rational(1, 2), a),
                (ctx.rational(1, 2), Distribution::die(ctx.int(6)))
            ])
            .is_err(),
        "one kind per mixture"
    );
    Ok(())
}

// ── Sampling through the new routes ──────────────────────────────────────

#[test]
fn sampling_reaches_normals_truncations_and_transforms() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let n = standard_normal(&ctx);
    let mut rng = Rng::new(3);
    // 0.12: erfinv is compiled, so Normal sampling works (mean 0, variance 1).
    let s = n.sample(20_000, &mut rng)?;
    let var = s.iter().map(|z| z * z).sum::<f64>() / s.len() as f64;
    assert!((var - 1.0).abs() < 0.05, "variance {var}");
    // Truncated: inverse transform through the transported quantile.
    let half = n.given(&n.symbol().gt(&ctx.int(0)))?;
    let s = half.sample(20_000, &mut rng)?;
    assert!(s.iter().all(|v| *v > 0.0));
    let mean = s.iter().sum::<f64>() / s.len() as f64;
    assert!(
        (mean - 0.797_884_560_802_865_4).abs() < 0.02,
        "half-normal mean {mean}"
    );
    // Transformed: the map applied to inner samples.
    let sq = n.transform("S", &n.symbol().powi(2))?;
    let s = sq.sample(20_000, &mut rng)?;
    let mean = s.iter().sum::<f64>() / s.len() as f64;
    assert!((mean - 1.0).abs() < 0.05, "χ²(1) mean {mean}");
    Ok(())
}

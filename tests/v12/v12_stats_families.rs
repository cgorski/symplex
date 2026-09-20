//! symplex 0.12 — the `Family` trait and `Distribution` handle: custom
//! families, downcasting, the typed `Support`, whole-line `cdf`, the new
//! `FDistribution`, Beta/StudentT CDFs through the regularised incomplete
//! beta, and the event shapes the set machinery unlocks.  Reference values
//! cite SymPy 1.14 / mpmath 1.3 (`symplex/.venv/bin/python`).

use std::fmt;

use symplex::prelude::*;
use symplex::stats::{
    Distribution, Family, Kind, Normal, Piece, RandomVariable, Support, same_family,
};

fn assert_close(actual: &Ex, expected: f64, label: &str) {
    let v = actual
        .eval_f64()
        .unwrap_or_else(|e| panic!("{label}: `{actual}` did not evaluate: {e}"));
    assert!(
        (v - expected).abs() < 1e-12 * expected.abs().max(1.0),
        "{label}: {v} vs {expected}"
    );
}

// ── A user-defined family ────────────────────────────────────────────────

/// The standard semicircle law on [−1, 1]: density (2/π)√(1 − x²).
#[derive(Clone, Debug)]
struct Semicircle {
    ctx: Context,
}

impl PartialEq for Semicircle {
    fn eq(&self, _other: &Self) -> bool {
        true // no parameters
    }
}

impl Family for Semicircle {
    fn name(&self) -> &str {
        "Semicircle"
    }
    fn context(&self) -> Context {
        self.ctx.clone()
    }
    fn support(&self) -> Support {
        Support::interval(self.ctx.int(-1), self.ctx.int(1))
    }
    fn density(&self, x: &Ex) -> Ex {
        2 * (self.ctx.one() - x.powi(2)).sqrt() / self.ctx.pi()
    }
    fn parameters(&self) -> Vec<(&'static str, Ex)> {
        vec![]
    }
    fn eq_family(&self, other: &dyn Family) -> bool {
        same_family(self, other)
    }
    fn fmt_display(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Semicircle")
    }
}

#[test]
fn a_custom_family_gets_the_generic_machinery() -> Result<(), SymplexError> {
    // SymPy: X = ContinuousRV(x, 2*sqrt(1 - x**2)/pi, Interval(-1, 1)):
    // E(X) = 0, variance(X) = 1/4, E(X**4) = 1/8, P(X > 1/2) = (-sqrt(3)/4 + pi/3)/pi.
    let ctx = Context::new();
    let d = Distribution::from_family(Semicircle { ctx: ctx.clone() });
    assert_eq!(d.name(), "Semicircle");
    assert_eq!(format!("{d}"), "Semicircle");
    let x = RandomVariable::new(&ctx, "X", d.clone());
    assert_eq!(x.mean(), ctx.int(0));
    assert_eq!(x.variance(), ctx.rational(1, 4));
    assert_eq!(x.moment(4), ctx.rational(1, 8));
    assert_eq!(
        x.probability(&x.symbol().gt(&ctx.rational(1, 2)))?
            .simplify(),
        (ctx.rational(1, 3) - ctx.int(3).sqrt() / (4 * ctx.pi())).simplify()
    );
    // Equality goes through the family; downcasting recovers it.
    assert_eq!(
        d,
        Distribution::from_family(Semicircle { ctx: ctx.clone() })
    );
    assert_ne!(d, Distribution::normal(ctx.int(0), ctx.int(1)));
    assert!(d.downcast_ref::<Semicircle>().is_some());
    assert!(d.downcast_ref::<Normal>().is_none());
    Ok(())
}

// ── Support ──────────────────────────────────────────────────────────────

#[test]
fn support_intersections_clip_symbolic_and_numeric_ends() {
    let ctx = Context::new();
    let a = ctx.symbol("a");
    // [0, ∞) ∩ (a, ∞) keeps a symbolic max.
    let s = Support::half_line(ctx.int(0))
        .intersect(&Support::from_pieces(
            Kind::Continuous,
            vec![Piece::Interval {
                lo: a.clone(),
                hi: ctx.infinity(),
                lo_open: true,
                hi_open: true,
            }],
        ))
        .unwrap();
    let (lo, hi, _, _) = s.as_interval().unwrap();
    assert_eq!(*lo, ctx.int(0).max_with(&a));
    assert_eq!(*hi, ctx.infinity());
    // Numeric ends are decided exactly, and an empty result is empty.
    let s = Support::interval(ctx.int(0), ctx.int(5))
        .intersect(&Support::interval(ctx.int(3), ctx.int(9)))
        .unwrap();
    assert_eq!(s, Support::interval(ctx.int(3), ctx.int(5)));
    let empty = Support::interval(ctx.int(0), ctx.int(1))
        .intersect(&Support::interval(ctx.int(2), ctx.int(3)))
        .unwrap();
    assert!(empty.is_empty());
    // A lattice support normalises open and fractional ends inwards.
    let ints = Support::integers(&ctx, Some(ctx.int(0)), Some(ctx.int(10)));
    let clipped = ints
        .intersect(&Support::from_pieces(
            Kind::Continuous,
            vec![Piece::Interval {
                lo: ctx.rational(3, 2),
                hi: ctx.int(7),
                lo_open: false,
                hi_open: true,
            }],
        ))
        .unwrap();
    assert_eq!(
        clipped,
        Support::integers(&ctx, Some(ctx.int(2)), Some(ctx.int(6)))
    );
    assert_eq!(clipped.contains(&ctx.int(6)), Some(true));
    assert_eq!(clipped.contains(&ctx.rational(5, 2)), Some(false));
    // Points: a table value against an interval.
    let table = Support::points(vec![ctx.int(1), ctx.rational(5, 2), ctx.int(7)]);
    let inside = table
        .intersect(&Support::interval(ctx.int(2), ctx.int(8)))
        .unwrap();
    assert_eq!(
        inside.as_points().unwrap(),
        vec![ctx.rational(5, 2), ctx.int(7)]
    );
    // Round trip through SetEx.
    let set = Support::interval(ctx.int(0), ctx.int(1))
        .to_set(&ctx)
        .union(&ctx.finite_set(&[ctx.int(3)]));
    let back = Support::from_set(Kind::Continuous, &set).unwrap();
    assert_eq!(back.pieces().len(), 2);
    assert_eq!(
        format!("{}", Support::integers(&ctx, Some(ctx.int(1)), None)),
        "[1, oo) ∩ ℤ"
    );
}

// ── Whole-line cdf ───────────────────────────────────────────────────────

#[test]
fn cdf_is_clamped_to_the_support_like_sympy() {
    // SymPy: cdf(Uniform(0, 1))(x) = Piecewise((0, x < 0), (x, x <= 1), (1, True)),
    // cdf(Uniform(0,1))(3) = 1; cdf(Geometric(1/4))(k) is 0 below 1.
    let ctx = Context::new();
    let u = Distribution::uniform(ctx.int(0), ctx.int(1));
    assert_eq!(u.cdf(&ctx.int(3)), ctx.int(1));
    assert_eq!(u.cdf(&ctx.int(-1)), ctx.int(0));
    assert_eq!(u.cdf(&ctx.rational(1, 3)), ctx.rational(1, 3));
    let x = ctx.symbol("x");
    let pw = u.cdf(&x);
    assert_eq!(pw.subs(&x, &ctx.int(2)).eval(), ctx.int(1));
    assert_eq!(pw.subs(&x, &ctx.rational(1, 4)).eval(), ctx.rational(1, 4));
    assert_eq!(pw.subs(&x, &ctx.int(-3)).eval(), ctx.int(0));
    let g = Distribution::geometric(ctx.rational(1, 4));
    assert_eq!(g.cdf(&ctx.int(0)), ctx.int(0));
    assert_eq!(g.cdf(&ctx.int(2)), ctx.rational(7, 16));
    // Unbounded support: no clamp.
    let n = Distribution::normal(ctx.int(0), ctx.int(1));
    assert_eq!(n.cdf(&x), n.family().cdf(&x).unwrap());
}

// ── Point masses and event edge cases ────────────────────────────────────

#[test]
fn point_events_respect_the_lattice_and_the_support() -> Result<(), SymplexError> {
    // SymPy: P(Eq(B, 1/2)) = 0, P(Eq(B, -1)) = 0, P(Eq(B, 7)) = 0 for Binomial(5, 1/3);
    // P(Eq(B, 3) & (B > 5)) = 0; P((B > 1) & (B >= 2)) = P(B >= 2) = 131/243.
    let ctx = Context::new();
    let b = RandomVariable::new(
        &ctx,
        "B",
        Distribution::binomial(ctx.int(5), ctx.rational(1, 3)),
    );
    let bs = b.symbol();
    assert_eq!(b.probability(&bs.eq_expr(&ctx.rational(1, 2)))?, ctx.int(0));
    assert_eq!(b.probability(&bs.eq_expr(&ctx.int(-1)))?, ctx.int(0));
    assert_eq!(b.probability(&bs.eq_expr(&ctx.int(7)))?, ctx.int(0));
    assert_eq!(
        b.probability(&bs.eq_expr(&ctx.int(3)).and(&bs.gt(&ctx.int(5))))?,
        ctx.int(0)
    );
    assert_eq!(
        b.probability(&bs.gt(&ctx.int(1)).and(&bs.ge(&ctx.int(2))))?,
        ctx.rational(131, 243)
    );
    assert_eq!(
        b.probability(&bs.eq_expr(&ctx.int(3)))?,
        ctx.rational(40, 243)
    );
    // A Poisson point below the support is 0, not a Gamma pole.
    let p = RandomVariable::new(&ctx, "P", Distribution::poisson(ctx.int(2)));
    assert_eq!(
        p.probability(&p.symbol().eq_expr(&ctx.int(-1)))?,
        ctx.int(0)
    );
    // Continuous point masses are zero; the constant events are 0 and 1.
    let n = RandomVariable::new(&ctx, "N", Distribution::normal(ctx.int(0), ctx.int(1)));
    assert_eq!(n.probability(&n.symbol().eq_expr(&ctx.int(1)))?, ctx.int(0));
    assert_eq!(n.probability(&ctx.bool_true())?, ctx.int(1));
    assert_eq!(n.probability(&ctx.bool_false())?, ctx.int(0));
    Ok(())
}

// ── New CDFs through the regularised incomplete beta; FDistribution ──────

#[test]
fn beta_student_t_and_f_cdfs() -> Result<(), SymplexError> {
    let ctx = Context::new();
    // mpmath (dps 25): betainc(2.5, 1.5, 0, 0.3, regularized=True) = 0.08894372317066559158
    // (SymPy: cdf(Beta(5/2, 3/2))(3/10) is an unevaluated hypergeometric form.)
    let b = Distribution::beta(ctx.rational(5, 2), ctx.rational(3, 2));
    assert_close(
        &b.cdf(&ctx.rational(3, 10)),
        0.0889437231706656,
        "Beta(5/2,3/2) cdf(0.3)",
    );
    // SymPy: N(cdf(StudentT(3))(1), 20) = 0.80449889052211467904
    let t = Distribution::student_t(ctx.int(3));
    assert_close(&t.cdf(&ctx.int(1)), 0.8044988905221147, "t₃ cdf(1)");
    assert_close(&t.cdf(&ctx.int(-1)), 1.0 - 0.8044988905221147, "t₃ cdf(−1)");
    // FDistribution(3, 5): E = d₂/(d₂−2) = 5/3, Var = 2d₂²(d₁+d₂−2)/(d₁(d₂−2)²(d₂−4)) = 100/9.
    // SymPy raises NotImplementedError on E(FDistribution); mpmath quadrature of
    // the density (dps 25): mean 1.666666666666666666666667, E[X²] 13.888888888888…
    // (= 100/9 + 25/9).
    let f = Distribution::f_distribution(ctx.int(3), ctx.int(5));
    assert_eq!(f.mean(), ctx.rational(5, 3));
    assert_eq!(f.variance(), ctx.rational(100, 9));
    assert_eq!(f.moment(2), ctx.rational(125, 9));
    assert_eq!(f.support(), Support::half_line(ctx.int(0)));
    // mpmath: betainc(1.5, 2.5, 0, 6/11, regularized=True) = 0.7673760819999214441584702
    // (F(2) with d1 = 3, d2 = 5: x = d1·2/(d1·2 + d2) = 6/11), and the same by quadrature.
    assert_close(&f.cdf(&ctx.int(2)), 0.7673760819999215, "F₃,₅ cdf(2)");
    let x = ctx.symbol("x");
    // Total mass of the F density integrates to 1 numerically.
    let total = f
        .density(&x)
        .integrate_numeric(&x, &ctx.int(0), &ctx.int(200))?;
    assert!((total - 1.0).abs() < 1e-3, "F density mass {total}");
    Ok(())
}

// ── Cross-context guard ──────────────────────────────────────────────────

#[test]
fn random_variable_rejects_a_distribution_from_another_context() {
    let a = Context::new();
    let b = Context::new();
    let d = Distribution::normal(b.int(0), b.int(1));
    assert!(matches!(
        RandomVariable::try_new(&a, "X", d.clone()),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(RandomVariable::try_new(&b, "X", d).is_ok());
}

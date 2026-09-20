//! symplex 0.11 — `stats::joint`: independence algebra on several random
//! variables (joint expectation, variance/covariance/correlation), sums
//! with closed families, rectangle and ordering probabilities, conditional
//! expectation/probability, entropy.
//!
//! Reference values from SymPy 1.14 (`sympy.stats`), quoted per test:
//! `X = Normal('X', 0, 1); Y = Normal('Y', 0, 1)`.

use symplex::prelude::*;
use symplex::stats::{self, Distribution, RandomVariable};

fn standard_normals() -> (Context, RandomVariable, RandomVariable) {
    let ctx = Context::new();
    let x = RandomVariable::new(&ctx, "X", Distribution::normal(ctx.int(0), ctx.int(1)));
    let y = RandomVariable::new(&ctx, "Y", Distribution::normal(ctx.int(0), ctx.int(1)));
    (ctx, x, y)
}

fn f64_of(e: &Ex) -> f64 {
    e.eval_f64()
        .unwrap_or_else(|err| panic!("`{e}` does not evaluate: {err}"))
}

// ── Joint expectation ───────────────────────────────────────────────────

#[test]
fn expectation_of_product_of_independent_standard_normals_is_zero() {
    // SymPy: E(X*Y) == 0
    let (ctx, x, y) = standard_normals();
    let e = stats::expectation(&[&x, &y], &(x.symbol() * y.symbol())).unwrap();
    assert_eq!(e, ctx.int(0));
}

#[test]
fn expectation_of_square_of_sum_is_two() {
    // SymPy: E((X + Y)**2) == 2
    let (ctx, x, y) = standard_normals();
    let g = (x.symbol() + y.symbol()).powi(2);
    assert_eq!(stats::expectation(&[&x, &y], &g).unwrap(), ctx.int(2));
}

#[test]
fn expectation_of_x2y2_is_one() {
    // SymPy: E(X**2 * Y**2) == 1
    let (ctx, x, y) = standard_normals();
    let g = x.symbol().powi(2) * y.symbol().powi(2);
    assert_eq!(stats::expectation(&[&x, &y], &g).unwrap(), ctx.int(1));
}

#[test]
fn expectation_of_mixed_binomial_normal_polynomial() {
    // B = Binomial('B', 3, 1/2); N = Normal('N', 1, 2)
    // SymPy: E(B**2 * N) == 3, E(X**2 * B) == 3/2  →  E(B²N + X²B) = 9/2
    let (ctx, x, _) = standard_normals();
    let b = RandomVariable::new(
        &ctx,
        "B",
        Distribution::binomial(ctx.int(3), ctx.rational(1, 2)),
    );
    let n = RandomVariable::new(&ctx, "N", Distribution::normal(ctx.int(1), ctx.int(2)));
    let g = b.symbol().powi(2) * n.symbol() + x.symbol().powi(2) * b.symbol();
    assert_eq!(
        stats::expectation(&[&b, &n, &x], &g).unwrap(),
        ctx.rational(9, 2)
    );
}

#[test]
fn expectation_of_constant_is_the_constant() {
    let (ctx, x, y) = standard_normals();
    assert_eq!(
        stats::expectation(&[&x, &y], &ctx.int(7)).unwrap(),
        ctx.int(7)
    );
}

#[test]
fn expectation_treats_unlisted_symbols_as_parameters() {
    // E[aX + b] = b for a standard normal X.
    let (ctx, x, y) = standard_normals();
    let (a, b) = (ctx.symbol("a"), ctx.symbol("b"));
    let g = &a * x.symbol() + &b;
    assert_eq!(stats::expectation(&[&x, &y], &g).unwrap(), b);
}

#[test]
fn expectation_of_non_polynomial_in_one_variable_delegates_to_the_marginal() {
    // E[e^X] = e^{1/2}: the crate's integrator leaves it as an Integral,
    // which still evaluates numerically.  SymPy: E(exp(X)) → e^{1/2} (as an
    // erf/erfc expression).
    let (_, x, y) = standard_normals();
    let g = x.symbol().exp();
    let joint = stats::expectation(&[&x, &y], &g).unwrap();
    assert_eq!(joint, x.expectation(&g));
    assert!((f64_of(&joint) - 0.5f64.exp()).abs() < 1e-8, "{joint}");
}

#[test]
fn expectation_iterates_polynomial_variables_first() {
    // E[e^X · Y²] = E[e^X]·E[Y²] = e^{1/2}.  Y is integrated out by moments
    // first, leaving a single Integral in X (numerically evaluable).
    // SymPy: E(exp(X)*Y**2) = 3e^{1/2}/2 − e^{1/2}(erf + erfc)/2 = e^{1/2}.
    let (_, x, y) = standard_normals();
    let g = x.symbol().exp() * y.symbol().powi(2);
    let e = stats::expectation(&[&x, &y], &g).unwrap();
    assert!((f64_of(&e) - 0.5f64.exp()).abs() < 1e-8, "{e}");
}

#[test]
fn expectation_rejects_duplicate_variables() {
    let (_, x, _) = standard_normals();
    let r = stats::expectation(&[&x, &x], x.symbol());
    assert!(
        matches!(r, Err(SymplexError::InvalidArgument { .. })),
        "{r:?}"
    );
}

#[test]
fn expectation_rejects_empty_variable_list() {
    let (ctx, _, _) = standard_normals();
    let r = stats::expectation(&[], &ctx.int(1));
    assert!(
        matches!(r, Err(SymplexError::InvalidArgument { .. })),
        "{r:?}"
    );
}

// ── Variance, covariance, correlation ───────────────────────────────────

#[test]
fn variance_of_sum_of_independent_standard_normals_is_two() {
    // SymPy: variance(X + Y) == 2
    let (ctx, x, y) = standard_normals();
    let g = x.symbol() + y.symbol();
    assert_eq!(stats::variance(&[&x, &y], &g).unwrap(), ctx.int(2));
}

#[test]
fn variance_of_affine_image_scales_by_the_square() {
    // SymPy: variance(2*X + 1) == 4
    let (ctx, x, _) = standard_normals();
    let g = 2 * x.symbol() + 1;
    assert_eq!(stats::variance(&[&x], &g).unwrap(), ctx.int(4));
}

#[test]
fn variance_of_binomial_plus_normal_adds_variances() {
    // Var(B + N) = 3/4 + 4 = 19/4.  SymPy's variance(B + N) is an
    // unsimplified erf/erfc expression that evaluates to 4.75.
    let ctx = Context::new();
    let b = RandomVariable::new(
        &ctx,
        "B",
        Distribution::binomial(ctx.int(3), ctx.rational(1, 2)),
    );
    let n = RandomVariable::new(&ctx, "N", Distribution::normal(ctx.int(1), ctx.int(2)));
    let g = b.symbol() + n.symbol();
    assert_eq!(stats::variance(&[&b, &n], &g).unwrap(), ctx.rational(19, 4));
}

#[test]
fn covariance_of_x_and_2x_is_two() {
    // SymPy: covariance(X, 2*X) == 2
    let (ctx, x, _) = standard_normals();
    let c = stats::covariance(&[&x], x.symbol(), &(2 * x.symbol())).unwrap();
    assert_eq!(c, ctx.int(2));
}

#[test]
fn covariance_of_sum_and_difference_is_zero() {
    // SymPy: covariance(X + Y, X - Y) == 0
    let (ctx, x, y) = standard_normals();
    let (xs, ys) = (x.symbol(), y.symbol());
    let c = stats::covariance(&[&x, &y], &(xs + ys), &(xs - ys)).unwrap();
    assert_eq!(c, ctx.int(0));
}

#[test]
fn correlation_of_x_and_2x_plus_1_is_one() {
    // SymPy: correlation(X, 2*X + 1) == 1
    let (ctx, x, _) = standard_normals();
    let r = stats::correlation(&[&x], x.symbol(), &(2 * x.symbol() + 1)).unwrap();
    assert_eq!(r, ctx.int(1));
}

#[test]
fn correlation_of_independent_variables_is_zero() {
    // SymPy: correlation(X, Y) == 0
    let (ctx, x, y) = standard_normals();
    let r = stats::correlation(&[&x, &y], x.symbol(), y.symbol()).unwrap();
    assert_eq!(r, ctx.int(0));
}

#[test]
fn correlation_with_a_constant_is_undefined() {
    let (ctx, x, _) = standard_normals();
    let r = stats::correlation(&[&x], x.symbol(), &ctx.int(3));
    assert!(
        matches!(r, Err(SymplexError::InvalidArgument { .. })),
        "{r:?}"
    );
}

// ── Sums with closed families ───────────────────────────────────────────

#[test]
fn sum_of_independent_normals_is_normal() {
    // Normal(0, 1) + Normal(1, 2) = Normal(1, √5)
    let ctx = Context::new();
    let x = RandomVariable::new(&ctx, "X", Distribution::normal(ctx.int(0), ctx.int(1)));
    let y = RandomVariable::new(&ctx, "Y", Distribution::normal(ctx.int(1), ctx.int(2)));
    assert_eq!(
        stats::sum_distribution(&x, &y),
        Some(Distribution::normal(ctx.int(1), ctx.int(5).sqrt()))
    );
}

#[test]
fn sum_of_binomials_with_the_same_p_is_binomial() {
    // Binomial(3, 1/2) + Binomial(2, 1/2) = Binomial(5, 1/2)
    let ctx = Context::new();
    let half = ctx.rational(1, 2);
    let b = RandomVariable::new(&ctx, "B", Distribution::binomial(ctx.int(3), half.clone()));
    let c = RandomVariable::new(&ctx, "C", Distribution::binomial(ctx.int(2), half.clone()));
    assert_eq!(
        stats::sum_distribution(&b, &c),
        Some(Distribution::binomial(ctx.int(5), half))
    );
}

#[test]
fn sum_of_bernoulli_and_binomial_with_the_same_p_is_binomial() {
    // Bernoulli(1/2) + Binomial(3, 1/2) = Binomial(4, 1/2), in either order.
    let ctx = Context::new();
    let half = ctx.rational(1, 2);
    let a = RandomVariable::new(&ctx, "A", Distribution::bernoulli(half.clone()));
    let b = RandomVariable::new(&ctx, "B", Distribution::binomial(ctx.int(3), half.clone()));
    let expected = Some(Distribution::binomial(ctx.int(4), half.clone()));
    assert_eq!(stats::sum_distribution(&a, &b), expected);
    assert_eq!(stats::sum_distribution(&b, &a), expected);
    let c = RandomVariable::new(&ctx, "C", Distribution::bernoulli(half.clone()));
    assert_eq!(
        stats::sum_distribution(&a, &c),
        Some(Distribution::binomial(ctx.int(2), half))
    );
}

#[test]
fn sum_of_poissons_is_poisson_with_added_rates() {
    // Poisson(2) + Poisson(3) = Poisson(5); symbolic rates add too.
    let ctx = Context::new();
    let p = RandomVariable::new(&ctx, "P", Distribution::poisson(ctx.int(2)));
    let q = RandomVariable::new(&ctx, "Q", Distribution::poisson(ctx.int(3)));
    assert_eq!(
        stats::sum_distribution(&p, &q),
        Some(Distribution::poisson(ctx.int(5)))
    );
    let (l1, l2) = (ctx.symbol("l1"), ctx.symbol("l2"));
    let r = RandomVariable::new(&ctx, "R", Distribution::poisson(l1.clone()));
    let s = RandomVariable::new(&ctx, "S", Distribution::poisson(l2.clone()));
    assert_eq!(
        stats::sum_distribution(&r, &s),
        Some(Distribution::poisson(l1 + l2))
    );
}

#[test]
fn sum_of_negative_binomials_with_the_same_p_adds_the_success_counts() {
    let ctx = Context::new();
    let third = ctx.rational(1, 3);
    let a = RandomVariable::new(
        &ctx,
        "A",
        Distribution::negative_binomial(ctx.int(2), third.clone()),
    );
    let b = RandomVariable::new(
        &ctx,
        "B",
        Distribution::negative_binomial(ctx.int(5), third.clone()),
    );
    assert_eq!(
        stats::sum_distribution(&a, &b),
        Some(Distribution::negative_binomial(ctx.int(7), third))
    );
    let c = RandomVariable::new(
        &ctx,
        "C",
        Distribution::negative_binomial(ctx.int(5), ctx.rational(1, 4)),
    );
    assert_eq!(stats::sum_distribution(&a, &c), None);
}

#[test]
fn sum_of_binomials_with_different_p_has_no_closed_family() {
    let ctx = Context::new();
    let b = RandomVariable::new(
        &ctx,
        "B",
        Distribution::binomial(ctx.int(3), ctx.rational(1, 2)),
    );
    let c = RandomVariable::new(
        &ctx,
        "C",
        Distribution::binomial(ctx.int(2), ctx.rational(1, 3)),
    );
    assert_eq!(stats::sum_distribution(&b, &c), None);
}

#[test]
fn sum_of_normal_and_binomial_has_no_closed_family() {
    let ctx = Context::new();
    let x = RandomVariable::new(&ctx, "X", Distribution::normal(ctx.int(0), ctx.int(1)));
    let b = RandomVariable::new(
        &ctx,
        "B",
        Distribution::binomial(ctx.int(3), ctx.rational(1, 2)),
    );
    assert_eq!(stats::sum_distribution(&x, &b), None);
}

#[test]
fn sum_of_a_variable_with_itself_is_not_a_sum_of_independent_variables() {
    // X + X = 2X ~ Normal(0, 2), not Normal(0, √2).
    let (_, x, _) = standard_normals();
    assert_eq!(stats::sum_distribution(&x, &x), None);
}

// ── Joint probabilities ─────────────────────────────────────────────────

#[test]
fn probability_of_positive_quadrant_is_a_quarter() {
    // P(X > 0 ∧ Y > 0) = P(X > 0)·P(Y > 0) = 1/4 (SymPy 1.14's P(And(X > 0,
    // Y > 0)) raises AttributeError on the product domain).
    let (ctx, x, y) = standard_normals();
    let zero = ctx.int(0);
    let event = x.symbol().gt(&zero).and(&y.symbol().gt(&zero));
    assert_eq!(
        stats::probability(&[&x, &y], &event).unwrap(),
        ctx.rational(1, 4)
    );
}

#[test]
fn probability_of_rectangle_with_several_relations_per_variable() {
    // P(0 < X ∧ X < 1 ∧ Y ≥ 0) = P(0 < X < 1)·(1/2)
    let (ctx, x, y) = standard_normals();
    let (xs, ys) = (x.symbol(), y.symbol());
    let (zero, one) = (ctx.int(0), ctx.int(1));
    let event = xs.gt(&zero).and(&xs.lt(&one)).and(&ys.ge(&zero));
    let joint = stats::probability(&[&x, &y], &event).unwrap();
    let marginal = x.probability(&xs.gt(&zero).and(&xs.lt(&one))).unwrap();
    assert!(
        (f64_of(&joint) - f64_of(&marginal) / 2.0).abs() < 1e-12,
        "{joint}"
    );
}

#[test]
fn probability_of_mixed_rectangle_normal_and_binomial() {
    // P(X > 0 ∧ B ≥ 2) = 1/2 · (3/8 + 1/8) = 1/4
    let (ctx, x, _) = standard_normals();
    let b = RandomVariable::new(
        &ctx,
        "B",
        Distribution::binomial(ctx.int(3), ctx.rational(1, 2)),
    );
    let event = x.symbol().gt(&ctx.int(0)).and(&b.symbol().ge(&ctx.int(2)));
    assert_eq!(
        stats::probability(&[&x, &b], &event).unwrap(),
        ctx.rational(1, 4)
    );
}

#[test]
fn probability_x_less_than_y_for_standard_normals_is_exactly_half() {
    // SymPy: P(X < Y) == 1/2
    let (ctx, x, y) = standard_normals();
    let p = stats::probability(&[&x, &y], &x.symbol().lt(y.symbol())).unwrap();
    assert_eq!(p, ctx.rational(1, 2));
    // The reversed ordering, and the non-strict one, are the same event up
    // to a null set.
    let p_ge = stats::probability(&[&x, &y], &x.symbol().ge(y.symbol())).unwrap();
    assert_eq!(p_ge, ctx.rational(1, 2));
}

#[test]
fn probability_x_less_than_y_for_general_normals() {
    // N = Normal('N', 1, 2).  SymPy: P(X < N).evalf() == 0.672639576990712
    // (= Φ(1/√5)); SymPy leaves an unevaluated Integral, ours is an erf.
    let (ctx, x, _) = standard_normals();
    let n = RandomVariable::new(&ctx, "N", Distribution::normal(ctx.int(1), ctx.int(2)));
    let p = stats::probability(&[&x, &n], &x.symbol().lt(n.symbol())).unwrap();
    assert!(!p.has_unevaluated(), "{p}");
    assert!((f64_of(&p) - 0.672_639_576_990_712).abs() < 1e-12, "{p}");
}

#[test]
fn probability_of_equality_of_continuous_variables_is_zero() {
    let (ctx, x, y) = standard_normals();
    let p = stats::probability(&[&x, &y], &x.symbol().eq_expr(y.symbol())).unwrap();
    assert_eq!(p, ctx.int(0));
}

#[test]
fn probability_of_relation_mixing_variables_is_not_implemented() {
    // X + Y > 0 is not a rectangle nor a bare ordering.
    let (ctx, x, y) = standard_normals();
    let event = (x.symbol() + y.symbol()).gt(&ctx.int(0));
    let r = stats::probability(&[&x, &y], &event);
    assert!(matches!(r, Err(SymplexError::NotImplemented(_))), "{r:?}");
}

#[test]
fn probability_of_ordering_combined_with_other_relations_is_not_implemented() {
    let (ctx, x, y) = standard_normals();
    let event = x.symbol().lt(y.symbol()).and(&x.symbol().gt(&ctx.int(0)));
    let r = stats::probability(&[&x, &y], &event);
    assert!(matches!(r, Err(SymplexError::NotImplemented(_))), "{r:?}");
}

#[test]
fn probability_of_event_in_an_unlisted_variable_is_not_implemented() {
    let (ctx, x, y) = standard_normals();
    let r = stats::probability(&[&x], &y.symbol().gt(&ctx.int(0)));
    assert!(matches!(r, Err(SymplexError::NotImplemented(_))), "{r:?}");
}

// ── Conditioning ────────────────────────────────────────────────────────

#[test]
fn conditional_expectation_of_half_normal_is_sqrt_2_over_pi() {
    // SymPy: E(X, X > 0) == sqrt(2)/sqrt(pi) ≈ 0.797884560802865
    let (ctx, x, _) = standard_normals();
    let e = stats::conditional_expectation(&x, x.symbol(), &x.symbol().gt(&ctx.int(0))).unwrap();
    assert!(!e.has_unevaluated(), "{e}");
    let expected = (2.0 / std::f64::consts::PI).sqrt();
    assert!((f64_of(&e) - expected).abs() < 1e-9, "{e}");
    // Exactly: e − √(2/π) = 0.
    let diff = (&e - (ctx.int(2) / ctx.pi()).sqrt()).simplify();
    assert!((f64_of(&diff)).abs() < 1e-15, "{diff}");
}

#[test]
fn conditional_second_moment_of_half_normal_is_one() {
    // SymPy: E(X**2, X > 0) == 1
    let (ctx, x, _) = standard_normals();
    let e = stats::conditional_expectation(&x, &x.symbol().powi(2), &x.symbol().gt(&ctx.int(0)))
        .unwrap();
    assert_eq!(e, ctx.int(1));
}

#[test]
fn conditional_third_moment_of_half_normal() {
    // SymPy: E(X**3, X > 0) == 2*sqrt(2)/sqrt(pi) ≈ 1.595769121605731
    let (ctx, x, _) = standard_normals();
    let e = stats::conditional_expectation(&x, &x.symbol().powi(3), &x.symbol().gt(&ctx.int(0)))
        .unwrap();
    assert!((f64_of(&e) - 1.595_769_121_605_731).abs() < 1e-12, "{e}");
}

#[test]
fn conditional_expectation_on_a_bounded_interval() {
    // E[X | 0 < X < 1] = (φ(0) − φ(1)) / (Φ(1) − 1/2), numerically.
    let (ctx, x, _) = standard_normals();
    let xs = x.symbol();
    let event = xs.gt(&ctx.int(0)).and(&xs.lt(&ctx.int(1)));
    let e = stats::conditional_expectation(&x, xs, &event).unwrap();
    let phi = |t: f64| (-t * t / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let p = f64_of(&x.probability(&event).unwrap());
    let expected = (phi(0.0) - phi(1.0)) / p;
    assert!((f64_of(&e) - expected).abs() < 1e-9, "{e}");
}

#[test]
fn conditional_expectation_of_binomial_given_at_least_two_successes() {
    // B ~ Binomial(3, 1/2): E[B | B ≥ 2] = (2·3/8 + 3·1/8) / (4/8) = 9/4
    let ctx = Context::new();
    let b = RandomVariable::new(
        &ctx,
        "B",
        Distribution::binomial(ctx.int(3), ctx.rational(1, 2)),
    );
    let e = stats::conditional_expectation(&b, b.symbol(), &b.symbol().ge(&ctx.int(2))).unwrap();
    assert_eq!(e, ctx.rational(9, 4));
}

#[test]
fn conditional_expectation_given_a_point_of_a_discrete_variable_is_the_value() {
    let ctx = Context::new();
    let b = RandomVariable::new(
        &ctx,
        "B",
        Distribution::binomial(ctx.int(3), ctx.rational(1, 2)),
    );
    let e =
        stats::conditional_expectation(&b, &b.symbol().powi(2), &b.symbol().eq_expr(&ctx.int(2)))
            .unwrap();
    assert_eq!(e, ctx.int(4));
}

#[test]
fn conditional_expectation_on_a_null_event_is_an_error() {
    // X = 0 has probability zero for a continuous variable.
    let (ctx, x, _) = standard_normals();
    let r = stats::conditional_expectation(&x, x.symbol(), &x.symbol().eq_expr(&ctx.int(0)));
    assert!(
        matches!(r, Err(SymplexError::InvalidArgument { .. })),
        "{r:?}"
    );
}

#[test]
fn conditional_expectation_on_a_non_linear_event_goes_through_the_set_machinery() {
    // 0.12: `X² > 1` is reduced to (−∞, −1) ∪ (1, ∞) by the inequality
    // solver.  E[X | X² > 1] = 0 by symmetry; E[X² | X² > 1] =
    // 2∫₁^∞ x²φ / P(|X| > 1) ≈ 2.5251352761609812091 (SymPy 1.14 itself
    // raises AttributeError on `E(X, X**2 > 1)`; the value is by quadrature
    // in SymPy on the two half-lines).
    let (ctx, x, _) = standard_normals();
    let event = x.symbol().powi(2).gt(&ctx.int(1));
    let r = stats::conditional_expectation(&x, x.symbol(), &event).unwrap();
    assert_eq!(r, ctx.int(0));
    let r2 = stats::conditional_expectation(&x, &x.symbol().powi(2), &event).unwrap();
    assert!(
        (r2.eval_f64().unwrap() - 2.5251352761609813).abs() < 1e-12,
        "{r2}"
    );
    // Symbolic bounds with a non-linear shape are still honestly unsupported.
    let a = ctx.symbol("a");
    let r3 = stats::conditional_expectation(&x, x.symbol(), &x.symbol().powi(2).gt(&a));
    assert!(matches!(r3, Err(SymplexError::NotImplemented(_))), "{r3:?}");
}

#[test]
fn conditional_probability_x_gt_1_given_x_gt_0_is_twice_the_tail() {
    // SymPy: P(X > 1, X > 0) == 1 − erf(√2/2) ≈ 0.317310507862914 = 2·P(X > 1)
    let (ctx, x, _) = standard_normals();
    let xs = x.symbol();
    let p = stats::conditional_probability(&x, &xs.gt(&ctx.int(1)), &xs.gt(&ctx.int(0))).unwrap();
    let tail = f64_of(&x.probability(&xs.gt(&ctx.int(1))).unwrap());
    assert!((f64_of(&p) - 2.0 * tail).abs() < 1e-12, "{p}");
    assert!((f64_of(&p) - 0.317_310_507_862_914).abs() < 1e-12, "{p}");
}

#[test]
fn conditional_probability_given_a_null_event_is_an_error() {
    let (ctx, x, _) = standard_normals();
    let xs = x.symbol();
    let r = stats::conditional_probability(&x, &xs.gt(&ctx.int(1)), &xs.eq_expr(&ctx.int(0)));
    assert!(
        matches!(r, Err(SymplexError::InvalidArgument { .. })),
        "{r:?}"
    );
}

// ── Entropy ─────────────────────────────────────────────────────────────

#[test]
fn entropy_of_standard_normal_is_half_ln_2_pi_e() {
    // SymPy: entropy(X) == log(2)/2 + 1/2 + log(pi)/2 ≈ 1.41893853320467
    let (ctx, x, _) = standard_normals();
    let h = stats::entropy(&x);
    assert!(!h.has_unevaluated(), "{h}");
    assert!((f64_of(&h) - 1.418_938_533_204_67).abs() < 1e-12, "{h}");
    let expected = (2 * ctx.pi() * ctx.e()).ln() / 2;
    assert_eq!((&h - expected).expand_log().simplify(), ctx.int(0));
}

#[test]
fn entropy_of_normal_with_parameters() {
    // N = Normal('N', 1, 2).  SymPy: entropy(N) == 1/2 + log(pi)/2 + 3*log(2)/2
    // ≈ 2.11208571376462
    let ctx = Context::new();
    let n = RandomVariable::new(&ctx, "N", Distribution::normal(ctx.int(1), ctx.int(2)));
    let h = stats::entropy(&n);
    assert!(!h.has_unevaluated(), "{h}");
    assert!((f64_of(&h) - 2.112_085_713_764_62).abs() < 1e-12, "{h}");
}

#[test]
fn entropy_of_symbolic_normal_is_half_ln_2_pi_e_sigma_squared() {
    let ctx = Context::new();
    let mu = ctx.symbol("mu");
    let sigma = ctx.symbol_with("sigma", &[Assumption::Positive]);
    let z = RandomVariable::new(&ctx, "Z", Distribution::normal(mu, sigma.clone()));
    let h = stats::entropy(&z);
    assert!(!h.has_unevaluated(), "{h}");
    let expected = (2 * ctx.pi() * ctx.e() * sigma.powi(2)).ln() / 2;
    assert_eq!((&h - expected).expand_log().simplify(), ctx.int(0));
}

#[test]
fn entropy_of_binomial_is_the_shannon_entropy_of_its_pmf() {
    // Binomial(3, 1/2): pmf (1/8, 3/8, 3/8, 1/8), H = 3 ln 2 − (3/4) ln 3
    // ≈ 1.255482325178754
    let ctx = Context::new();
    let b = RandomVariable::new(
        &ctx,
        "B",
        Distribution::binomial(ctx.int(3), ctx.rational(1, 2)),
    );
    let h = stats::entropy(&b);
    let expected = 3.0 * 2f64.ln() - 0.75 * 3f64.ln();
    assert!((f64_of(&h) - expected).abs() < 1e-12, "{h}");
}

/// Gamma-family closure under independent sums: Exponential(λ) + Exponential(λ)
/// = Gamma(2, 1/λ), Gamma(k₁, θ) + Gamma(k₂, θ) = Gamma(k₁ + k₂, θ), χ²(a) +
/// χ²(b) = χ²(a + b); different scales have no closed family.
#[test]
fn gamma_family_sums() {
    use symplex::stats::{Distribution, RandomVariable, sum_distribution};
    let ctx = Context::new();
    let e1 = RandomVariable::new(&ctx, "E1", Distribution::exponential(ctx.int(3)));
    let e2 = RandomVariable::new(&ctx, "E2", Distribution::exponential(ctx.int(3)));
    assert_eq!(
        sum_distribution(&e1, &e2),
        Some(Distribution::gamma(ctx.int(2), ctx.rational(1, 3)))
    );
    let g1 = RandomVariable::new(
        &ctx,
        "G1",
        Distribution::gamma(ctx.rational(3, 2), ctx.int(2)),
    );
    let g2 = RandomVariable::new(&ctx, "G2", Distribution::gamma(ctx.int(2), ctx.int(2)));
    assert_eq!(
        sum_distribution(&g1, &g2),
        Some(Distribution::gamma(ctx.rational(7, 2), ctx.int(2)))
    );
    let c1 = RandomVariable::new(&ctx, "C1", Distribution::chi_squared(ctx.int(3)));
    let c2 = RandomVariable::new(&ctx, "C2", Distribution::chi_squared(ctx.int(4)));
    assert_eq!(
        sum_distribution(&c1, &c2),
        Some(Distribution::chi_squared(ctx.int(7)))
    );
    // Mixed scale: none.
    let g3 = RandomVariable::new(&ctx, "G3", Distribution::gamma(ctx.int(1), ctx.int(5)));
    assert_eq!(sum_distribution(&g1, &g3), None);
    // The sum's mean is the sum of the means (Exponential(3): 1/3 each).
    let s = RandomVariable::new(&ctx, "S", sum_distribution(&e1, &e2).unwrap());
    assert_eq!(s.mean(), ctx.rational(2, 3));
}

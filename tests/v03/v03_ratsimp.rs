//! symplex 0.3 — `Ex::ratsimp`: rational-function normal form (single
//! fraction, common factors cancelled, integer-primitive numerator and
//! denominator), the multivariate GCD behind it, and its use in `solve`.

use num_bigint::BigInt;
use num_rational::Ratio;
use proptest::prelude::*;
use symplex::prelude::*;

/// Wall-clock hang guard (scaled up on shared CI runners).
fn time_budget(secs: u64) -> std::time::Duration {
    let mult = if std::env::var_os("CI").is_some() {
        5
    } else {
        1
    };
    std::time::Duration::from_secs(secs * mult)
}

fn s(e: &Ex) -> String {
    format!("{e}")
}

fn rat(p: i64, q: i64) -> Ratio<BigInt> {
    Ratio::new(BigInt::from(p), BigInt::from(q))
}

/// Exact value of `e` at rational points, or `None` if undefined there.
fn value_at(e: &Ex, vars: &[&Ex], pts: &[Ratio<BigInt>]) -> Option<Ratio<BigInt>> {
    let ctx = e.context();
    let vals: Vec<Ex> = pts.iter().map(|r| ctx.from_ratio(r.clone())).collect();
    let pairs: Vec<(&Ex, &Ex)> = vars.iter().copied().zip(vals.iter()).collect();
    let v = e.subs_map(&pairs).eval();
    v.as_rational()
}

/// `a` and `b` agree exactly wherever both are defined, at a grid of
/// rational points; at least three points must be defined for both.
fn assert_same_function(a: &Ex, b: &Ex, vars: &[&Ex], label: &str) {
    let grid: Vec<Ratio<BigInt>> = vec![
        rat(2, 1),
        rat(-3, 1),
        rat(1, 2),
        rat(-5, 3),
        rat(7, 4),
        rat(11, 1),
    ];
    let mut checked = 0;
    let n = vars.len();
    let combos: usize = grid.len().pow(n as u32);
    for idx in 0..combos {
        let mut pts = Vec::with_capacity(n);
        let mut k = idx;
        for _ in 0..n {
            pts.push(grid[k % grid.len()].clone());
            k /= grid.len();
        }
        if let (Some(va), Some(vb)) = (value_at(a, vars, &pts), value_at(b, vars, &pts)) {
            assert_eq!(va, vb, "{label}: {a}  vs  {b} at {pts:?}");
            checked += 1;
        }
    }
    assert!(checked >= 3, "{label}: too few defined points");
}

// ═══════════════════════════════════════════════════════════════════════════
// The motivating cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn together_leftover_is_cancelled_to_j_minus_1_over_2j() {
    let ctx = Context::new();
    let j = ctx.symbol("j");
    let e = (j.powi(2) - 1) / (&j * 2) - (&j - 1) / 2;
    let r = e.ratsimp();
    assert_eq!(r, (&j - 1) / (&j * 2), "got {r}");
    assert_same_function(&r, &e, &[&j], "j case");
}

#[test]
fn solve_result_fraction_of_fractions_collapses() {
    let ctx = Context::new();
    let (j, r) = (ctx.symbol("j"), ctx.symbol("r"));
    let expr = (&r * 3 - 1) / (&j + 1) - (&r + 1) / (&j * 2);
    let sol = expr.solve(&r).unwrap();
    assert_eq!(sol.len(), 1);
    // (3r − 1)/(j + 1) = (r + 1)/(2j)  ⟺  6jr − 2j = jr + r + j + 1  ⟺  r = (3j + 1)/(5j − 1)
    assert_eq!(s(&sol[0]), "(3*j + 1)/(5*j - 1)");
    let expected = (&j * 3 + 1) / (&j * 5 - 1);
    assert_same_function(&sol[0], &expected, &[&j], "solve result");
    // The root really satisfies the equation.
    let residual = expr.subs(&r, &sol[0]).ratsimp();
    assert!(residual.is_zero_structural(), "residual {residual}");
}

#[test]
fn the_old_unsimplified_solve_form_ratsimps_to_the_same_thing() {
    let ctx = Context::new();
    let j = ctx.symbol("j");
    // "(1/2*1/j + 1/(j + 1))/(-1/2*1/j + 3/(j + 1))" as previously printed.
    let half = ctx.rational(1, 2);
    let num = &half / &j + ctx.int(1) / (&j + 1);
    let den = -&half / &j + ctx.int(3) / (&j + 1);
    let e = num / den;
    assert_eq!(s(&e.ratsimp()), "(3*j + 1)/(5*j - 1)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Basic identities
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn one_over_x_plus_one_over_y() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = ctx.int(1) / &x + ctx.int(1) / &y;
    let r = e.ratsimp();
    assert_eq!(r, (&x + &y) / (&x * &y), "got {r}");
}

#[test]
fn difference_of_squares_cancels() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = (x.powi(2) - y.powi(2)) / (&x - &y);
    assert_eq!(e.ratsimp(), &x + &y);
}

#[test]
fn univariate_cancel_matches_cancel() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = (x.powi(3) - 1) / (x.powi(2) - 1);
    let r = e.ratsimp();
    assert_eq!(r, (x.powi(2) + &x + 1) / (&x + 1), "got {r}");
    assert_eq!(r, e.cancel(&x));
}

#[test]
fn nested_fraction_two_variables() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // 1/(x + 1/y) + 1/(y + 1/x) = y/(xy + 1) + x/(xy + 1) = (x + y)/(xy + 1)
    let e = ctx.int(1) / (&x + ctx.int(1) / &y) + ctx.int(1) / (&y + ctx.int(1) / &x);
    let r = e.ratsimp();
    assert_eq!(r, (&x + &y) / (&x * &y + 1), "got {r}");
    assert_same_function(&r, &e, &[&x, &y], "nested");
}

#[test]
fn deeply_nested_continued_fraction() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 1/(1 + 1/(1 + 1/(1 + 1/x)))  =  (2x + 1)/(3x + 2)
    let mut e = x.clone();
    for _ in 0..3 {
        e = ctx.int(1) + ctx.int(1) / &e;
    }
    let e = ctx.int(1) / e;
    let r = e.ratsimp();
    assert_eq!(r, (&x * 2 + 1) / (&x * 3 + 2), "got {r}");
}

#[test]
fn sum_of_three_fractions_over_lcm_not_product() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 1/(x(x+1)) + 1/((x+1)(x+2)) + 1/(x(x+2))
    //   = [(x+2) + x + (x+1)] / (x(x+1)(x+2)) = 3(x+1)/(x(x+1)(x+2)) = 3/(x(x+2))
    let a = ctx.int(1) / (&x * (&x + 1));
    let b = ctx.int(1) / ((&x + 1) * (&x + 2));
    let c = ctx.int(1) / (&x * (&x + 2));
    let e = a + b + c;
    let r = e.ratsimp();
    assert_eq!(r, ctx.int(3) / (x.powi(2) + &x * 2), "got {r}");
    assert_same_function(&r, &e, &[&x], "three fractions");
}

#[test]
fn polynomial_input_is_a_fixed_point() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    for e in [
        &x + 1,
        x.powi(2) + &x * &y * 2 + y.powi(2),
        &x * 2 + 2,
        ctx.int(5).clone(),
        x.clone(),
    ] {
        let r = e.ratsimp();
        assert_eq!(r.id(), e.id(), "{e} → {r}");
    }
}

#[test]
fn simple_fraction_input_is_a_fixed_point() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    for e in [
        ctx.int(1) / &x,
        &x / &y,
        (&x + 1) / (&x - 1),
        (&x + &y) / (&x * &y + 1),
    ] {
        let r = e.ratsimp();
        assert_eq!(r.id(), e.id(), "{e} → {r}");
    }
}

#[test]
fn idempotent() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = (x.powi(2) - y.powi(2)) / ((&x - &y) * (&x + 2)) + ctx.int(1) / (&x + 2);
    let r1 = e.ratsimp();
    let r2 = r1.ratsimp();
    assert_eq!(r1.id(), r2.id());
    assert_eq!(r1, (&x + &y + 1) / (&x + 2), "got {r1}");
}

#[test]
fn zero_result_is_structural_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = ctx.int(1) / (&x + 1) - ctx.int(1) / (&x + 1);
    // Canonicalisation already gives 0 here; ratsimp must agree.
    assert!(e.ratsimp().is_zero_structural());
    let e = (&x + 1) / (x.powi(2) - 1) - ctx.int(1) / (&x - 1);
    assert!(e.ratsimp().is_zero_structural(), "got {}", e.ratsimp());
}

#[test]
fn constant_fraction_of_polynomials_reduces_to_number() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = (&x * 2 + 2) / (&x + 1);
    assert_eq!(e.ratsimp(), ctx.int(2));
    let e = (&x * 3 + 3) / (&x * 6 + 6);
    assert_eq!(e.ratsimp(), ctx.rational(1, 2));
}

// ═══════════════════════════════════════════════════════════════════════════
// Normalisation: integer-primitive parts, positive denominator
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rational_coefficients_are_cleared_to_integers() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x/2 + 1/3)/(x/4 − 1)  =  (6x + 4)/(3x − 12)
    let e = (&x / 2 + ctx.rational(1, 3)) / (&x / 4 - 1);
    let r = e.ratsimp();
    let (n, d) = r.as_numer_denom();
    let n_coeffs = n.coeffs(&x).unwrap();
    let d_coeffs = d.coeffs(&x).unwrap();
    for c in n_coeffs.iter().chain(d_coeffs.iter()) {
        let c = c.as_rational().unwrap();
        assert!(c.is_integer(), "non-integer coefficient {c} in {r}");
    }
    assert_same_function(&r, &e, &[&x], "cleared");
}

#[test]
fn denominator_leading_coefficient_is_positive() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = (&x + 1) / (ctx.int(1) - &x);
    let r = e.ratsimp();
    let (_n, d) = r.as_numer_denom();
    let lc = d.leading_coeff(&x).unwrap().as_rational().unwrap();
    assert!(lc > Ratio::from_integer(BigInt::from(0)), "{r}");
    assert_same_function(&r, &e, &[&x], "sign");
}

#[test]
fn common_integer_content_is_removed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (4x + 6)/(2x² + 10)  →  (2x + 3)/(x² + 5)
    let e = (&x * 4 + 6) / (x.powi(2) * 2 + 10);
    let r = e.ratsimp();
    assert_eq!(r, (&x * 2 + 3) / (x.powi(2) + 5), "got {r}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Opaque subexpressions are independent indeterminates
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sin_squared_over_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = x.sin().powi(2) / x.sin();
    assert_eq!(e.ratsimp(), x.sin());
}

#[test]
fn trig_rational_function_cancels_but_no_identities() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (sn, cs) = (x.sin(), x.cos());
    // (sin² − cos²)/(sin − cos) = sin + cos
    let e = (sn.powi(2) - cs.powi(2)) / (&sn - &cs);
    assert_eq!(e.ratsimp(), &sn + &cs);
    // sin² + cos² is NOT turned into 1.
    let pyth = sn.powi(2) + cs.powi(2);
    assert_eq!(pyth.ratsimp().id(), pyth.id());
}

#[test]
fn pi_is_a_generator() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi = ctx.pi();
    let e = (&x * &pi + &pi) / (&x + 1);
    assert_eq!(e.ratsimp(), pi);
    let e = ctx.int(1) / &pi + ctx.int(1) / (&pi * 2);
    let r = e.ratsimp();
    assert_eq!(r, ctx.int(3) / (&pi * 2), "got {r}");
}

#[test]
fn sqrt_is_a_generator() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sq = x.sqrt();
    // (√x·y + √x)/(y + 1) = √x
    let y = ctx.symbol("y");
    let e = (&sq * &y + &sq) / (&y + 1);
    assert_eq!(e.ratsimp(), sq);
}

#[test]
fn exp_terms_cancel_as_indeterminates() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let ex = x.exp();
    let e = (ex.powi(2) - 1) / (&ex + 1);
    assert_eq!(e.ratsimp(), &ex - 1);
}

#[test]
fn symbolic_power_is_a_generator() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    let xa = x.pow(&a);
    let e = (xa.powi(2) + &xa) / &xa;
    assert_eq!(e.ratsimp(), &xa + 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// Things that must be left alone
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn infinity_and_nan_are_unchanged() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for e in [
        &x + ctx.infinity(),
        ctx.int(1) / (&x - ctx.neg_infinity()),
        &x * ctx.nan(),
        ctx.complex_infinity() + &x,
    ] {
        assert_eq!(e.ratsimp().id(), e.id(), "{e}");
    }
}

#[test]
fn unevaluated_nodes_are_unchanged() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let deriv = y.formal_diff(&x);
    assert!(deriv.has_unevaluated());
    let e = (&deriv * &x + &deriv) / (&x + 1);
    assert_eq!(e.ratsimp().id(), e.id());
}

// ═══════════════════════════════════════════════════════════════════════════
// simplify_rational delegates to ratsimp
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simplify_rational_is_ratsimp() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = (x.powi(2) - y.powi(2)) / (&x - &y) + ctx.int(1) / &x;
    assert_eq!(e.simplify_rational().id(), e.ratsimp().id());
    assert_eq!(
        (x.powi(-1) + x.powi(-1)).simplify_rational(),
        ctx.int(2) / &x
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// solve integration
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_linear_with_fractional_parameter_coefficients() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    // x/a + x/(a+1) = 1  →  x = a(a + 1)/(2a + 1)
    let e = &x / &a + &x / (&a + 1) - 1;
    let sol = e.solve(&x).unwrap();
    assert_eq!(sol.len(), 1);
    let expected = (a.powi(2) + &a) / (&a * 2 + 1);
    assert_eq!(sol[0], expected, "got {}", sol[0]);
}

#[test]
fn solve_quadratic_discriminant_is_a_single_fraction() {
    let ctx = Context::new();
    let (r, j) = (ctx.symbol("r"), ctx.symbol("j"));
    // r²/(j + 1) − r/j + 1 = 0
    let e = r.powi(2) / (&j + 1) - &r / &j + 1;
    let sols = e.solve(&r).unwrap();
    assert_eq!(sols.len(), 2);
    let shown = s(&sols[0]);
    assert!(
        shown.contains("sqrt((-4*j^2 + j + 1)/(j^3 + j^2))"),
        "discriminant not normalised: {shown}"
    );
    // Both roots satisfy the equation.
    for sol in &sols {
        let residual = e.subs(&r, sol);
        let pts = [(3i64, 1i64), (5, 1), (7, 2)];
        for (p, q) in pts {
            let v = residual
                .subs(&j, &ctx.rational(p, q))
                .eval()
                .eval_complex64();
            let Complex64 { re, im } = v.unwrap();
            assert!(
                re.abs() < 1e-9 && im.abs() < 1e-9,
                "residual {re}+{im}i at j={p}/{q}"
            );
        }
    }
}

#[test]
fn solve_plain_symbolic_linear_unchanged() {
    let ctx = Context::new();
    let (x, a, b) = (ctx.symbol("x"), ctx.symbol("a"), ctx.symbol("b"));
    let sol = (&a * &x + &b).solve(&x).unwrap();
    assert_eq!(s(&sol[0]), "-b/a");
}

// ═══════════════════════════════════════════════════════════════════════════
// Performance guard
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn moderately_large_rational_expression_is_fast() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let start = std::time::Instant::now();
    // Σ_{k=1..6} (x + k y)/(x + k z + y)
    let mut e = ctx.zero();
    for k in 1..=6 {
        e += (&x + &y * k) / (&x + &z * k + &y);
    }
    let r = e.ratsimp();
    assert!(
        start.elapsed() < time_budget(5),
        "ratsimp took {:?}",
        start.elapsed()
    );
    assert_same_function(&r, &e, &[&x, &y, &z], "six fractions");
    let (n, d) = r.as_numer_denom();
    assert_eq!(d.expand().degree(&z), Some(6), "denominator {d}");
    assert!(n.expand().degree(&z).unwrap() <= 5);
}

// ═══════════════════════════════════════════════════════════════════════════
// Property: value preservation on random rational expressions
// ═══════════════════════════════════════════════════════════════════════════

/// A small random rational expression in `x`, `y` built from integers and
/// the four operations (division always by a non-constant or non-zero).
#[derive(Debug, Clone)]
enum Tree {
    X,
    Y,
    Int(i8),
    Add(Box<Tree>, Box<Tree>),
    Mul(Box<Tree>, Box<Tree>),
    Div(Box<Tree>, Box<Tree>),
    Pow(Box<Tree>, u8),
}

fn tree_strategy() -> impl Strategy<Value = Tree> {
    let leaf = prop_oneof![Just(Tree::X), Just(Tree::Y), (-4i8..=4).prop_map(Tree::Int),];
    leaf.prop_recursive(4, 24, 2, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone()).prop_map(|(a, b)| Tree::Add(Box::new(a), Box::new(b))),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| Tree::Mul(Box::new(a), Box::new(b))),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| Tree::Div(Box::new(a), Box::new(b))),
            (inner, 2u8..=3).prop_map(|(a, n)| Tree::Pow(Box::new(a), n)),
        ]
    })
}

fn build(ctx: &Context, x: &Ex, y: &Ex, t: &Tree) -> Ex {
    match t {
        Tree::X => x.clone(),
        Tree::Y => y.clone(),
        Tree::Int(n) => ctx.int(i64::from(*n)),
        Tree::Add(a, b) => build(ctx, x, y, a) + build(ctx, x, y, b),
        Tree::Mul(a, b) => build(ctx, x, y, a) * build(ctx, x, y, b),
        Tree::Div(a, b) => build(ctx, x, y, a) / build(ctx, x, y, b),
        Tree::Pow(a, n) => build(ctx, x, y, a).powi(i64::from(*n)),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(40))]

    #[test]
    fn ratsimp_preserves_value_at_random_rational_points(
        t in tree_strategy(),
        px in -6i64..=6, qx in 1i64..=5,
        py in -6i64..=6, qy in 1i64..=5,
    ) {
        let ctx = Context::new();
        let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
        let e = build(&ctx, &x, &y, &t);
        // Skip expressions that canonicalise to an infinity / NaN.
        if e.has_unevaluated() || s(&e).contains("zoo") || s(&e).contains("nan") {
            return Ok(());
        }
        let start = std::time::Instant::now();
        let r = e.ratsimp();
        prop_assert!(start.elapsed() < time_budget(5), "ratsimp took {:?} on {e}", start.elapsed());
        let pts = [rat(px, qx), rat(py, qy)];
        if let (Some(ve), Some(vr)) = (value_at(&e, &[&x, &y], &pts), value_at(&r, &[&x, &y], &pts)) {
            prop_assert_eq!(ve, vr, "{} → {} at {:?}", e, r, pts);
        }
        // Idempotence.
        prop_assert_eq!(r.ratsimp().id(), r.id(), "not idempotent: {}", r);
    }
}

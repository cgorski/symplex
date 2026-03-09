//! Integration tests for Cycle 7–8 math features.
//!
//! Covers:
//!   • Complex numbers: sqrt of negatives, i-unit properties
//!   • Eval: sqrt simplification (radical denesting)
//!   • Eval: trig ↔ hyperbolic bridge (sin(ix), cos(ix))
//!   • Integration: inverse-trig forms, linear-substitution power rule
//!   • Solver: factored products, transcendental equation boundaries
//!   • Convenience API: global constructors, diff_n, args, assumption queries

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Complex: sqrt of negative numbers
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sqrt_neg_one_is_i() {
    let ctx = Context::new();
    let result = ctx.int(-1).sqrt();
    // Canonicalization immediately reduces (-1)^(1/2) to I.
    assert_eq!(format!("{result}"), "I");
}

#[test]
fn sqrt_neg_four_without_eval() {
    let ctx = Context::new();
    let result = ctx.int(-4).sqrt();
    // Canonicalization now fully simplifies sqrt(-4) → 2*I at construction.
    // Previously it only partially simplified to sqrt(4)*I; the improved
    // eval pipeline now reduces sqrt(4) → 2 as well.
    let s = format!("{result}");
    assert!(
        s.contains("I"),
        "sqrt(-4) should contain I, got: {s}"
    );
    assert!(
        s == "2*I" || (s.contains("I") && s.contains("sqrt(4)")),
        "sqrt(-4) should be 2*I (fully simplified) or sqrt(4)*I (partial), got: {s}"
    );
}

#[test]
fn sqrt_neg_four_eval_gives_2i() {
    let ctx = Context::new();
    let result = ctx.int(-4).sqrt().eval();
    // After eval, sqrt(4) collapses to 2 → 2*I.
    assert_eq!(format!("{result}"), "2*I");
}

#[test]
fn sqrt_neg_nine_eval_gives_3i() {
    let ctx = Context::new();
    let result = ctx.int(-9).sqrt().eval();
    assert_eq!(format!("{result}"), "3*I");
}

#[test]
fn sqrt_neg_two_contains_i() {
    let ctx = Context::new();
    let result = ctx.int(-2).sqrt();
    let s = format!("{result}");
    assert!(s.contains("I"), "sqrt(-2) should contain I: {s}");
    assert!(
        s.contains("sqrt(2)"),
        "sqrt(-2) should contain sqrt(2): {s}"
    );
}

#[test]
fn sqrt_neg_three_contains_i() {
    let ctx = Context::new();
    let result = ctx.int(-3).sqrt();
    let s = format!("{result}");
    assert!(s.contains("I"), "sqrt(-3) should contain I: {s}");
    assert!(
        s.contains("sqrt(3)"),
        "sqrt(-3) should contain sqrt(3): {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// i^n canonicalization (verify cycle-7 power rules)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn i_squared_is_neg_one() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    assert_eq!(format!("{}", i.powi(2)), "-1");
}

#[test]
fn i_cubed_is_neg_i() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    assert_eq!(format!("{}", i.powi(3)), "-I");
}

#[test]
fn i_fourth_is_one() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    assert_eq!(format!("{}", i.powi(4)), "1");
}

#[test]
fn i_to_the_fifth_is_i() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    assert_eq!(format!("{}", i.powi(5)), "I");
}

#[test]
fn i_to_neg_one_is_neg_i() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    assert_eq!(format!("{}", i.powi(-1)), "-I");
}

#[test]
fn i_to_100_is_one() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    assert_eq!(format!("{}", i.powi(100)), "1"); // 100 mod 4 = 0
}

// ═══════════════════════════════════════════════════════════════════════════
// Complex algebra
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn one_plus_i_squared_is_2i() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let expr = (&ctx.int(1) + &i).powi(2).expand();
    assert_eq!(format!("{expr}"), "2*I");
}

#[test]
fn i_times_i_is_neg_one() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let result = &i * &i;
    assert_eq!(format!("{result}"), "-1");
}

#[test]
fn complex_addition() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let z1 = &ctx.int(2) + &(&ctx.int(3) * &i);
    let z2 = &ctx.int(4) + &(&ctx.int(5) * &i);
    let sum = &z1 + &z2;
    assert_eq!(format!("{sum}"), "8*I + 6");
}

// ═══════════════════════════════════════════════════════════════════════════
// Eval: sqrt simplification (radical denesting)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sqrt_perfect_square_4() {
    let ctx = Context::new();
    let result = ctx.int(4).sqrt().eval();
    assert_eq!(format!("{result}"), "2");
}

#[test]
fn sqrt_perfect_square_9() {
    let ctx = Context::new();
    let result = ctx.int(9).sqrt().eval();
    assert_eq!(format!("{result}"), "3");
}

#[test]
fn sqrt_perfect_square_16() {
    let ctx = Context::new();
    let result = ctx.int(16).sqrt().eval();
    assert_eq!(format!("{result}"), "4");
}

#[test]
fn sqrt_8_simplified() {
    let ctx = Context::new();
    let result = ctx.int(8).sqrt().eval();
    assert_eq!(format!("{result}"), "2*sqrt(2)");
}

#[test]
fn sqrt_12_simplified() {
    let ctx = Context::new();
    let result = ctx.int(12).sqrt().eval();
    assert_eq!(format!("{result}"), "2*sqrt(3)");
}

#[test]
fn sqrt_50_simplified() {
    let ctx = Context::new();
    let result = ctx.int(50).sqrt().eval();
    assert_eq!(format!("{result}"), "5*sqrt(2)");
}

#[test]
fn sqrt_18_simplified() {
    let ctx = Context::new();
    let result = ctx.int(18).sqrt().eval();
    let s = format!("{result}");
    // 18 = 9 * 2, so sqrt(18) = 3*sqrt(2)
    assert!(
        s.contains("3") && s.contains("sqrt(2)"),
        "sqrt(18) should simplify to 3*sqrt(2), got: {s}"
    );
}

#[test]
fn sqrt_prime_stays_unevaluated() {
    let ctx = Context::new();
    let result = ctx.int(7).sqrt().eval();
    assert_eq!(format!("{result}"), "sqrt(7)");
}

#[test]
fn sqrt_one_is_one() {
    let ctx = Context::new();
    let result = ctx.int(1).sqrt().eval();
    assert_eq!(format!("{result}"), "1");
}

#[test]
fn sqrt_zero_is_zero() {
    let ctx = Context::new();
    let result = ctx.int(0).sqrt().eval();
    assert_eq!(format!("{result}"), "0");
}

// ═══════════════════════════════════════════════════════════════════════════
// Eval: trig-hyperbolic bridge
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sin_of_ix_gives_i_sinh_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let i = ctx.i_unit();
    let result = (&i * &x).sin().eval();
    let s = format!("{result}");
    // sin(ix) = i·sinh(x)
    assert!(
        s.contains("sinh") && s.contains("I"),
        "sin(ix) should → I*sinh(x), got: {s}"
    );
}

#[test]
fn cos_of_ix_gives_cosh_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let i = ctx.i_unit();
    let result = (&i * &x).cos().eval();
    let s = format!("{result}");
    // cos(ix) = cosh(x)
    assert!(s.contains("cosh"), "cos(ix) should → cosh(x), got: {s}");
    // Should NOT contain I (cosh is real-valued for real x).
    assert!(
        !s.contains("I"),
        "cos(ix) = cosh(x) should not contain I, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Integration: inverse-trig antiderivatives
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_asin_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.asin().integrate(&x);
    let s = format!("{result}");
    // ∫ asin(x) dx = x·asin(x) + sqrt(1-x²)
    assert!(s.contains("asin"), "∫ asin(x) dx should contain asin: {s}");
    assert!(s.contains("sqrt"), "∫ asin(x) dx should contain sqrt: {s}");
    assert!(
        !s.contains("Integral"),
        "∫ asin(x) dx should not be unevaluated: {s}"
    );
}

#[test]
fn integrate_asin_x_exact() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.asin().integrate(&x);
    assert_eq!(format!("{result}"), "x*asin(x) + sqrt(-x^2 + 1)");
}

#[test]
fn integrate_acos_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.acos().integrate(&x);
    let s = format!("{result}");
    // ∫ acos(x) dx = x·acos(x) - sqrt(1-x²)
    assert!(s.contains("acos"), "∫ acos(x) dx should contain acos: {s}");
    assert!(s.contains("sqrt"), "∫ acos(x) dx should contain sqrt: {s}");
    assert!(
        !s.contains("Integral"),
        "∫ acos(x) dx should not be unevaluated: {s}"
    );
}

#[test]
fn integrate_acos_x_exact() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.acos().integrate(&x);
    assert_eq!(format!("{result}"), "x*acos(x) - sqrt(-x^2 + 1)");
}

#[test]
fn integrate_atan_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.atan().integrate(&x);
    let s = format!("{result}");
    // ∫ atan(x) dx = x·atan(x) - (1/2)·ln(1+x²)
    assert!(s.contains("atan"), "∫ atan(x) dx should contain atan: {s}");
    assert!(s.contains("ln"), "∫ atan(x) dx should contain ln: {s}");
    assert!(
        !s.contains("Integral"),
        "∫ atan(x) dx should not be unevaluated: {s}"
    );
}

#[test]
fn integrate_atan_x_exact() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.atan().integrate(&x);
    assert_eq!(format!("{result}"), "x*atan(x) - 1/2*ln(x^2 + 1)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Integration: linear-substitution power rule
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_x_plus_1_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = (&x + 1).powi(2).integrate(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("Integral"),
        "∫ (x+1)² dx should be evaluated: {s}"
    );
    assert_eq!(s, "1/3*(x + 1)^3");
}

#[test]
fn integrate_2x_plus_1_cubed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let base = &x * 2 + 1;
    let result = base.powi(3).integrate(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("Integral"),
        "∫ (2x+1)³ dx should not be unevaluated: {s}"
    );
    // (2x+1)^4 / (4·2) = 1/8·(1+2x)^4
    assert_eq!(s, "1/8*(2*x + 1)^4");
}

#[test]
fn integrate_x_plus_1_squared_verify_by_diff() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = (&x + 1).powi(2);
    let anti = integrand.integrate(&x);
    let back = anti.diff(&x);
    // Differentiating should give back (1+x)^2.
    let s = format!("{back}");
    assert!(
        s.contains("x + 1") || s.contains("x + 1"),
        "d/dx(∫(x+1)² dx) should recover (1+x)², got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Integration: basic forms (verify cycle-7 antiderivatives still work)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_sin_x() {
    let x = symplex::default_context().symbol("x");
    let result = x.sin().integrate(&x);
    assert_eq!(format!("{result}"), "-cos(x)");
}

#[test]
fn integrate_cos_x() {
    let x = symplex::default_context().symbol("x");
    let result = x.cos().integrate(&x);
    assert_eq!(format!("{result}"), "sin(x)");
}

#[test]
fn integrate_exp_x() {
    let x = symplex::default_context().symbol("x");
    let result = x.exp().integrate(&x);
    assert_eq!(format!("{result}"), "exp(x)");
}

#[test]
fn integrate_sinh_x() {
    let x = symplex::default_context().symbol("x");
    let result = x.sinh().integrate(&x);
    assert_eq!(format!("{result}"), "cosh(x)");
}

#[test]
fn integrate_cosh_x() {
    let x = symplex::default_context().symbol("x");
    let result = x.cosh().integrate(&x);
    assert_eq!(format!("{result}"), "sinh(x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Integration by parts (cycle-8)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_x_sin_x() {
    let x = symplex::default_context().symbol("x");
    let expr = &x * &x.sin();
    let result = expr.integrate(&x);
    let s = format!("{result}");
    assert!(
        s.contains("sin(x)"),
        "∫ x·sin(x) dx should contain sin(x): {s}"
    );
    assert!(
        s.contains("cos(x)"),
        "∫ x·sin(x) dx should contain cos(x): {s}"
    );
}

#[test]
fn integrate_x_exp_x() {
    let x = symplex::default_context().symbol("x");
    let expr = &x * &x.exp();
    let result = expr.integrate(&x);
    let s = format!("{result}");
    assert!(
        s.contains("exp(x)"),
        "∫ x·exp(x) dx should contain exp(x): {s}"
    );
}

#[test]
fn integrate_x_cos_x() {
    let x = symplex::default_context().symbol("x");
    let expr = &x * &x.cos();
    let result = expr.integrate(&x);
    let s = format!("{result}");
    assert!(
        s.contains("sin(x)") && s.contains("cos(x)"),
        "∫ x·cos(x) dx should contain sin(x) and cos(x): {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Solver: polynomial / factored products
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_linear() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let eq = &x * 2 - 6;
    let roots = eq.solve(&x).unwrap();
    assert_eq!(roots.len(), 1);
    assert_eq!(format!("{}", roots[0]), "3");
}

#[test]
fn solve_quadratic_two_roots() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let eq = x.powi(2) - &x * 5 + 6;
    let roots = eq.solve(&x).unwrap();
    assert_eq!(roots.len(), 2);
    let vals: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    assert!(vals.contains(&"2".into()), "should have root 2: {vals:?}");
    assert!(vals.contains(&"3".into()), "should have root 3: {vals:?}");
}

#[test]
fn solve_quadratic_complex_roots() {
    // x² + 1 = 0 has roots ±I
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let eq = x.powi(2) + 1;
    let roots = eq.solve(&x).unwrap();
    assert_eq!(roots.len(), 2, "x²+1=0 should have 2 complex roots");
    let strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    let joined = strs.join(", ");
    assert!(joined.contains("I"), "roots should contain I: {joined}");
}

#[test]
fn solve_mul_factors_three_roots() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x·(x-1)·(x+2) = 0
    let eq = &x * &(&x - 1) * &(&x + 2);
    let roots = eq.solve_or_empty(&x);
    assert!(
        roots.len() >= 3,
        "x(x-1)(x+2)=0 should have 3 roots, got {}",
        roots.len()
    );
    let vals: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    assert!(vals.contains(&"0".into()), "should have root 0: {vals:?}");
    assert!(vals.contains(&"1".into()), "should have root 1: {vals:?}");
    assert!(vals.contains(&"-2".into()), "should have root -2: {vals:?}");
}

#[test]
fn solve_mul_factors_two_roots() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x·(x - 5) = 0
    let eq = &x * &(&x - 5);
    let roots = eq.solve_or_empty(&x);
    assert!(
        roots.len() >= 2,
        "x(x-5)=0 should have 2 roots, got {}",
        roots.len()
    );
    let vals: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    assert!(vals.contains(&"0".into()), "should have root 0: {vals:?}");
    assert!(vals.contains(&"5".into()), "should have root 5: {vals:?}");
}

#[test]
fn solve_cubic_all_rational() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x³ - 6x² + 11x - 6 = 0 → roots 1, 2, 3
    let eq = x.powi(3) - &x.powi(2) * 6 + &x * 11 - 6;
    let roots = eq.solve(&x).unwrap();
    assert_eq!(roots.len(), 3, "expected 3 roots, got {}", roots.len());
    let vals: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    assert!(vals.contains(&"1".into()), "should have root 1: {vals:?}");
    assert!(vals.contains(&"2".into()), "should have root 2: {vals:?}");
    assert!(vals.contains(&"3".into()), "should have root 3: {vals:?}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Solver: transcendental equations (public API boundary)
//
// The public `Ex::solve()` guards with a polynomial check, so
// transcendental forms like exp(x)=c currently return Err.
// `solve_or_empty` wraps that as an empty Vec.
// We document these boundaries here so regressions are caught if the
// public API is later extended.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_exp_x_minus_5_is_transcendental() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let eq = &x.exp() - 5;
    // The public API now forwards to the internal solver which handles
    // transcendental equations via inversion peeling: exp(x)=5 → x=ln(5).
    let roots = eq.solve(&x).expect("exp(x)-5 should be solvable now");
    assert!(
        !roots.is_empty(),
        "exp(x)-5 should have at least one root (ln(5))"
    );
    // Verify numerically: ln(5) ≈ 1.6094
    let val = roots[0].eval_f64().expect("root should evaluate");
    assert!(
        (val - 5.0_f64.ln()).abs() < 1e-9,
        "root should be ln(5), got {val}"
    );
}

#[test]
fn solve_ln_x_minus_2_is_transcendental() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let eq = &x.ln() - 2;
    // The internal solver handles ln(x)=2 → x=e² via inversion peeling.
    let roots = eq.solve_or_empty(&x);
    assert!(
        !roots.is_empty(),
        "ln(x)-2 should have at least one root (e²)"
    );
    let val = roots[0].eval_f64().expect("root should evaluate");
    assert!(
        (val - std::f64::consts::E.powi(2)).abs() < 1e-9,
        "root should be e², got {val}"
    );
}

#[test]
fn solve_sqrt_x_minus_3_is_transcendental() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let eq = &x.sqrt() - 3;
    // The internal solver handles sqrt(x)=3 → x=9 via inversion peeling.
    let roots = eq.solve_or_empty(&x);
    assert!(
        !roots.is_empty(),
        "sqrt(x)-3 should have at least one root (9)"
    );
    let val = roots[0].eval_f64().expect("root should evaluate");
    assert!(
        (val - 9.0).abs() < 1e-9,
        "root should be 9, got {val}"
    );
}

#[test]
fn solve_non_polynomial_returns_ok_via_inversion() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin();
    // sin(x)=0 is now handled by the internal solver via inversion
    // peeling: x = asin(0) = 0. The solver finds at least the principal root.
    let result = expr.solve(&x);
    assert!(
        result.is_ok(),
        "sin(x) should be solvable via inversion peeling, got: {:?}",
        result.err()
    );
}

#[test]
fn solve_constant_nonzero_no_solutions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let roots = ctx.int(5).solve(&x).unwrap();
    assert!(roots.is_empty(), "5=0 has no solutions");
}

// ═══════════════════════════════════════════════════════════════════════════
// Convenience API: global constructors
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn global_pi() {
    let pi = symplex::default_context().pi();
    assert_eq!(format!("{pi}"), "pi");
}

#[test]
fn global_e() {
    let e = symplex::default_context().e();
    assert_eq!(format!("{e}"), "E");
}

#[test]
fn global_i_unit() {
    let i = symplex::default_context().i_unit();
    assert_eq!(format!("{i}"), "I");
}

#[test]
fn global_infinity() {
    let inf = symplex::default_context().infinity();
    assert_eq!(format!("{inf}"), "oo");
}

#[test]
fn global_convenience_all_four() {
    let pi = symplex::default_context().pi();
    let e = symplex::default_context().e();
    let i = symplex::default_context().i_unit();
    let inf = symplex::default_context().infinity();
    assert_eq!(format!("{pi}"), "pi");
    assert_eq!(format!("{e}"), "E");
    assert_eq!(format!("{i}"), "I");
    assert_eq!(format!("{inf}"), "oo");
}

#[test]
fn global_var() {
    let x = symplex::default_context().symbol("x");
    assert_eq!(format!("{x}"), "x");
}

#[test]
fn global_int() {
    let five = symplex::default_context().int(5);
    assert_eq!(format!("{five}"), "5");
}

#[test]
fn global_rational() {
    let half = symplex::default_context().rational(1, 2);
    assert_eq!(format!("{half}"), "1/2");
}

// ═══════════════════════════════════════════════════════════════════════════
// Convenience API: diff_n (nth derivative)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_n_third_derivative_x5() {
    let x = symplex::default_context().symbol("x");
    let f = x.powi(5);
    let d3 = f.diff_n(&x, 3);
    assert_eq!(format!("{d3}"), "60*x^2");
}

#[test]
fn diff_n_fourth_derivative_x4() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(4);
    let d4 = f.diff_n(&x, 4);
    assert_eq!(format!("{d4}"), "24");
}

#[test]
fn diff_n_zeroth_derivative_is_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(3);
    let d0 = f.diff_n(&x, 0);
    assert_eq!(format!("{d0}"), "x^3");
}

#[test]
fn diff_n_first_derivative() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(4);
    let d1 = f.diff_n(&x, 1);
    assert_eq!(format!("{d1}"), "4*x^3");
}

#[test]
fn diff_n_high_order_vanishes() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(3);
    // Fourth derivative of a cubic is 0.
    let d4 = f.diff_n(&x, 4);
    assert_eq!(format!("{d4}"), "0");
}

// ═══════════════════════════════════════════════════════════════════════════
// Convenience API: args (child sub-expressions)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn args_of_sum() {
    let x = symplex::default_context().symbol("x");
    let expr = &x + 1;
    assert_eq!(expr.args().len(), 2, "x + 1 should have 2 children");
}

#[test]
fn args_of_function() {
    let x = symplex::default_context().symbol("x");
    assert_eq!(x.sin().args().len(), 1, "sin(x) should have 1 child");
    assert_eq!(x.cos().args().len(), 1, "cos(x) should have 1 child");
    assert_eq!(x.exp().args().len(), 1, "exp(x) should have 1 child");
}

#[test]
fn args_of_atom() {
    let x = symplex::default_context().symbol("x");
    assert_eq!(x.args().len(), 0, "a symbol has 0 children");
}

#[test]
fn args_of_integer() {
    let n = symplex::default_context().int(42);
    assert_eq!(n.args().len(), 0, "an integer has 0 children");
}

#[test]
fn args_of_product() {
    let x = symplex::default_context().symbol("x");
    let y = symplex::default_context().symbol("y");
    let expr = &x * &y;
    assert!(
        expr.args().len() >= 2,
        "x*y should have at least 2 children, got {}",
        expr.args().len()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Convenience API: assumption queries on i
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn i_is_imaginary() {
    let i = symplex::default_context().i_unit();
    assert_eq!(i.is_imaginary(), Some(true));
}

#[test]
fn i_is_complex() {
    let i = symplex::default_context().i_unit();
    assert_eq!(i.is_complex(), Some(true));
}

#[test]
fn i_is_not_real() {
    let i = symplex::default_context().i_unit();
    assert_eq!(i.is_real(), Some(false));
}

#[test]
fn i_is_not_zero() {
    let i = symplex::default_context().i_unit();
    assert_eq!(i.is_nonzero(), Some(true));
}

#[test]
fn i_query_via_props() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    assert_eq!(i.query(Props::IMAGINARY), Some(true));
    assert_eq!(i.query(Props::REAL), Some(false));
    assert_eq!(i.query(Props::COMPLEX), Some(true));
}

// ═══════════════════════════════════════════════════════════════════════════
// Convenience API: assumption queries on reals
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integer_is_real() {
    let ctx = Context::new();
    let two = ctx.int(2);
    assert_eq!(two.is_real(), Some(true));
    assert_eq!(two.is_positive(), Some(true));
    assert_eq!(two.is_nonnegative(), Some(true));
}

#[test]
fn negative_integer_is_negative() {
    let ctx = Context::new();
    let neg = ctx.int(-3);
    assert_eq!(neg.is_negative(), Some(true));
    assert_eq!(neg.is_nonnegative(), Some(false));
}

#[test]
fn i_squared_becomes_neg_one_which_is_real() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let i2 = i.powi(2);
    assert_eq!(i2.is_real(), Some(true));
    assert_eq!(i2.is_negative(), Some(true));
}

#[test]
fn assume_positive() {
    let t = symplex::default_context().symbol("t").assume(Assumption::Positive);
    assert_eq!(t.is_positive(), Some(true));
}

#[test]
fn assume_integer_implies_real() {
    let n = symplex::default_context().symbol("n").assume(Assumption::Integer);
    assert_eq!(n.is_integer(), Some(true));
    assert_eq!(n.is_real(), Some(true));
}

// ═══════════════════════════════════════════════════════════════════════════
// Convenience API: sum_of / product_of
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sum_of_integers() {
    let ctx = Context::new();
    let terms: Vec<Ex> = (1..=4).map(|n| ctx.int(n)).collect();
    let total = Ex::sum_of(&ctx, terms);
    assert_eq!(format!("{total}"), "10");
}

#[test]
fn product_of_integers() {
    let ctx = Context::new();
    let factors: Vec<Ex> = (1..=4).map(|n| ctx.int(n)).collect();
    let total = Ex::product_of(&ctx, factors);
    assert_eq!(format!("{total}"), "24");
}

#[test]
fn sum_of_empty_is_zero() {
    let ctx = Context::new();
    let total = Ex::sum_of(&ctx, vec![]);
    assert_eq!(format!("{total}"), "0");
}

#[test]
fn product_of_empty_is_one() {
    let ctx = Context::new();
    let total = Ex::product_of(&ctx, vec![]);
    assert_eq!(format!("{total}"), "1");
}

// ═══════════════════════════════════════════════════════════════════════════
// Convenience API: evalf_f64
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn evalf_f64_integer() {
    let five = symplex::default_context().int(5);
    let val = five.eval_f64().unwrap();
    assert!((val - 5.0).abs() < 1e-10);
}

#[test]
fn evalf_f64_pi() {
    let ctx = Context::new();
    let val = ctx.pi().eval_f64().unwrap();
    assert!((val - std::f64::consts::PI).abs() < 1e-10);
}

#[test]
fn evalf_f64_free_symbol_errors() {
    let x = symplex::default_context().symbol("x");
    assert!(x.eval_f64().is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// Eval: trig / exp special values (cycle-7 regressions)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_sin_zero() {
    let ctx = Context::new();
    let result = ctx.int(0).sin().eval();
    assert_eq!(format!("{result}"), "0");
}

#[test]
fn eval_cos_zero() {
    let ctx = Context::new();
    let result = ctx.int(0).cos().eval();
    assert_eq!(format!("{result}"), "1");
}

#[test]
fn eval_exp_zero() {
    let ctx = Context::new();
    let result = ctx.int(0).exp().eval();
    assert_eq!(format!("{result}"), "1");
}

#[test]
fn eval_ln_one() {
    let ctx = Context::new();
    let result = ctx.int(1).ln().eval();
    assert_eq!(format!("{result}"), "0");
}

#[test]
fn eval_ln_e() {
    let ctx = Context::new();
    let result = ctx.e().ln().eval();
    assert_eq!(format!("{result}"), "1");
}

// ═══════════════════════════════════════════════════════════════════════════
// Eval: inverse-trig special values
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_asin_zero() {
    let ctx = Context::new();
    let result = ctx.int(0).asin().eval();
    assert_eq!(format!("{result}"), "0");
}

#[test]
fn eval_asin_one_is_pi_over_2() {
    let ctx = Context::new();
    let result = ctx.int(1).asin().eval();
    let s = format!("{result}");
    assert!(s.contains("pi"), "asin(1) should contain pi: {s}");
}

#[test]
fn eval_acos_one_is_zero() {
    let ctx = Context::new();
    let result = ctx.int(1).acos().eval();
    assert_eq!(format!("{result}"), "0");
}

#[test]
fn eval_atan_zero() {
    let ctx = Context::new();
    let result = ctx.int(0).atan().eval();
    assert_eq!(format!("{result}"), "0");
}

// ═══════════════════════════════════════════════════════════════════════════
// Eval: hyperbolic special values
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_sinh_zero() {
    let ctx = Context::new();
    let result = ctx.int(0).sinh().eval();
    assert_eq!(format!("{result}"), "0");
}

#[test]
fn eval_cosh_zero() {
    let ctx = Context::new();
    let result = ctx.int(0).cosh().eval();
    assert_eq!(format!("{result}"), "1");
}

#[test]
fn eval_tanh_zero() {
    let ctx = Context::new();
    let result = ctx.int(0).tanh().eval();
    assert_eq!(format!("{result}"), "0");
}

// ═══════════════════════════════════════════════════════════════════════════
// Euler's formula (cycle-8 complex + eval)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn euler_exp_i_pi() {
    // exp(i·π) should ideally be -1, or remain unevaluated.
    let ctx = Context::new();
    let i = ctx.i_unit();
    let expr = (&i * &ctx.pi()).exp().eval();
    let s = format!("{expr}");
    assert!(
        s == "-1" || (s.contains("exp") && s.contains("I")),
        "exp(iπ) should be -1 or unevaluated: {s}"
    );
}

#[test]
fn euler_exp_i_pi_plus_1() {
    // exp(i·π) + 1 should ideally be 0.
    let ctx = Context::new();
    let i = ctx.i_unit();
    let expr = &(&i * &ctx.pi()).exp() + 1;
    let evald = expr.eval();
    let s = format!("{evald}");
    assert!(
        s == "0" || s.contains("exp"),
        "exp(iπ)+1 should be 0 or contain exp: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Diff + complex
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_complex_expression() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let i = ctx.i_unit();
    // d/dx(i·x²) = 2·i·x
    let expr = &i * &x.powi(2);
    let result = expr.diff(&x);
    let s = format!("{result}");
    assert!(
        s.contains("I") && s.contains("x"),
        "d/dx(i·x²) should be 2·I·x, got: {s}"
    );
}

#[test]
fn integrate_complex_expression() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let i = ctx.i_unit();
    // ∫ i·x dx = i·x²/2
    let expr = &i * &x;
    let result = expr.integrate(&x);
    let s = format!("{result}");
    assert!(
        s.contains("I") && s.contains("x"),
        "∫ i·x dx should involve I and x, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// End-to-end workflows combining multiple cycle-7/8 features
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn workflow_diff_subs_evalf() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // f(x) = x^3, f'(2) = 12
    let val = x.powi(3).diff(&x).subs_i64(&x, 2).eval_f64().unwrap();
    assert!((val - 12.0).abs() < 1e-10);
}

#[test]
fn workflow_expand_diff_subs() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x+1)^3 expanded, differentiated, evaluated at x=1
    let expanded = (&x + 1).powi(3).expand();
    let d = expanded.diff(&x);
    let val = d.subs_i64(&x, 1).eval_f64().unwrap();
    // d/dx((x+1)^3) = 3(x+1)^2 → at x=1: 3·4 = 12
    assert!((val - 12.0).abs() < 1e-10);
}

#[test]
fn workflow_definite_integral() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // ∫₀¹ x² dx = 1/3
    let result = x.powi(2).definite_integral(&x, &ctx.int(0), &ctx.int(1));
    assert_eq!(format!("{result}"), "1/3");
}

#[test]
fn workflow_solve_then_verify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let eq = x.powi(2) - &x * 5 + 6;
    let roots = eq.solve(&x).unwrap();
    for root in &roots {
        let substituted = eq.subs(&x, root);
        assert!(
            substituted.is_zero_structural(),
            "substituting x={root} into x²-5x+6 should give 0, got {substituted}"
        );
    }
}

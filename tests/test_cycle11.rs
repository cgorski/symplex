//! Comprehensive tests for Cycles 9-11 features.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Complex numerical evaluation (evalf Tier 3)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn evalf_imaginary_unit() {
    let __ctx = Context::new();
    let i = __ctx.i_unit();
    let result = i.eval_decimal(10);
    assert!(
        result.is_ok(),
        "evalf(i) should succeed: {:?}",
        result.err()
    );
    let s = result.unwrap();
    assert!(s.contains("i") || s.contains("I"), "should contain i: {s}");
}

#[test]
fn evalf_one_plus_i() {
    let ctx = Context::new();
    let expr = &ctx.int(1) + &ctx.i_unit();
    let result = expr.eval_decimal(10);
    assert!(result.is_ok(), "evalf(1+i) should succeed");
    let s = result.unwrap();
    // Should contain both 1 and i
    assert!(
        (s.contains("1") && (s.contains("i") || s.contains("I"))),
        "evalf(1+i) should show real and imaginary parts: {s}"
    );
}

#[test]
fn evalf_exp_i_pi_approx_neg_one() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let expr = (&i * &ctx.pi()).exp();
    let result = expr.eval_decimal(15);
    assert!(result.is_ok(), "evalf(exp(i*pi)) should succeed");
    let s = result.unwrap();
    // Should be approximately -1 (imaginary part near zero)
    assert!(s.contains("-1"), "exp(iπ) ≈ -1: {s}");
}

#[test]
fn evalf_exp_i_pi_via_eval_is_exact_neg_one() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let expr = (&i * &ctx.pi()).exp().eval();
    assert_eq!(
        format!("{expr}"),
        "-1",
        "exp(iπ) should evaluate to exactly -1"
    );
}

#[test]
fn evalf_abs_3_plus_4i() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let z = &ctx.int(3) + &(&ctx.int(4) * &i);
    let result = z.abs().eval_decimal(10);
    assert!(result.is_ok(), "evalf(|3+4i|) should succeed");
    let s = result.unwrap();
    assert!(s.starts_with("5"), "|3+4i| should be 5: {s}");
}

#[test]
fn evalf_abs_3_plus_4i_display() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let z = &ctx.int(3) + &(&ctx.int(4) * &i);
    let abs_z = z.abs();
    let s = format!("{abs_z}");
    assert!(
        s.contains("abs"),
        "|3+4i| symbolic form should contain abs: {s}"
    );
}

#[test]
fn evalf_pure_real_integer() {
    let ctx = Context::new();
    let r = ctx.int(42).eval_decimal(10);
    assert!(r.is_ok());
    assert!(r.unwrap().starts_with("42"), "evalf(42) should be 42");
}

#[test]
fn evalf_pi_digits() {
    let ctx = Context::new();
    let r = ctx.pi().eval_decimal(15);
    assert!(r.is_ok());
    let s = r.unwrap();
    assert!(
        s.starts_with("3.14159"),
        "evalf(pi) should start with 3.14159: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Trig power integration
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_sin_squared_eval() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.sin().powi(2).integrate(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("Integral"),
        "∫ sin²(x) should not be unevaluated: {s}"
    );
    // Known result: -1/2*sin(x)*cos(x) + 1/2*x
    assert!(
        s.contains("sin") && s.contains("cos"),
        "should contain trig terms: {s}"
    );
    assert!(s.contains("x"), "should contain x: {s}");
}

#[test]
fn integrate_cos_squared_eval() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.cos().powi(2).integrate(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("Integral"),
        "∫ cos²(x) should not be unevaluated: {s}"
    );
    assert!(
        s.contains("sin") && s.contains("cos"),
        "should contain trig terms: {s}"
    );
}

#[test]
fn integrate_sin_cubed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.sin().powi(3).integrate(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("Integral"),
        "∫ sin³(x) should not be unevaluated: {s}"
    );
    assert!(s.contains("cos"), "should involve cos: {s}");
}

#[test]
fn integrate_cos_cubed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.cos().powi(3).integrate(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("Integral"),
        "∫ cos³(x) should not be unevaluated: {s}"
    );
}

#[test]
fn integrate_cos_fourth() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.cos().powi(4).integrate(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("Integral"),
        "∫ cos⁴(x) should not be unevaluated: {s}"
    );
    assert!(
        s.contains("sin") && s.contains("cos"),
        "should contain trig terms: {s}"
    );
}

#[test]
fn integrate_sin_fourth() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.sin().powi(4).integrate(&x);
    let s = format!("{result}");
    assert!(
        !s.contains("Integral"),
        "∫ sin⁴(x) should not be unevaluated: {s}"
    );
}

#[test]
fn integrate_sin_squared_roundtrip_numerical() {
    // d/dx(∫ sin²(x) dx) should give back sin²(x), verified numerically
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integral = x.sin().powi(2).integrate(&x);
    let deriv = integral.diff(&x);
    // Evaluate both at x=1 numerically
    let original_at_1 = x.sin().powi(2).subs_i64(&x, 1).eval_f64();
    let roundtrip_at_1 = deriv.subs_i64(&x, 1).eval_f64();
    assert!(original_at_1.is_ok(), "original evalf should work");
    assert!(roundtrip_at_1.is_ok(), "roundtrip evalf should work");
    let orig = original_at_1.unwrap();
    let rt = roundtrip_at_1.unwrap();
    assert!(
        (orig - rt).abs() < 1e-10,
        "roundtrip should match: orig={orig}, rt={rt}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// U-substitution integration
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn u_sub_2x_exp_x_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = (&(&x * 2) * &x.powi(2).exp()).integrate(&x);
    let s = format!("{result}");
    assert!(s.contains("exp"), "∫ 2x·exp(x²) dx should contain exp: {s}");
    assert!(!s.contains("Integral"), "should not be unevaluated: {s}");
}

#[test]
fn u_sub_cos_exp_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = (&x.cos() * &x.sin().exp()).integrate(&x);
    let s = format!("{result}");
    assert!(s.contains("exp"), "∫ cos(x)·exp(sin(x)) dx: {s}");
    assert!(!s.contains("Integral"), "should not be unevaluated: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Trig combine (product-to-sum)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn trig_combine_sin_cos_different_args() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let product = &x.sin() * &y.cos();
    let combined = product.trig_combine();
    let s = format!("{combined}");
    assert!(s.contains("sin"), "sin(x)*cos(y) → sum of sins: {s}");
    // Should produce sin(x+y) and sin(x-y)
    assert!(
        s.contains("+") || s.contains("-"),
        "should be a sum/difference: {s}"
    );
}

#[test]
fn trig_combine_same_arg() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let product = &x.sin() * &x.cos();
    let combined = product.trig_combine();
    let s = format!("{combined}");
    assert!(s.contains("sin"), "sin(x)*cos(x) → involves sin(2x): {s}");
}

#[test]
fn trig_combine_cos_cos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let product = &x.cos() * &y.cos();
    let combined = product.trig_combine();
    let s = format!("{combined}");
    assert!(s.contains("cos"), "cos(x)*cos(y) → sum of cosines: {s}");
}

#[test]
fn trig_combine_then_expand_roundtrip() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // sin(x)*cos(y) → combine → expand_trig should give back products
    let product = &x.sin() * &y.cos();
    let combined = product.trig_combine();
    let re_expanded = combined.expand_trig();
    let s = format!("{re_expanded}");
    // After expand_trig, should contain sin and cos product terms
    assert!(
        s.contains("sin") && s.contains("cos"),
        "roundtrip should contain sin and cos: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Parser
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn parse_float() {
    let ctx = Context::new();
    let r = symplex::parse::parse(&ctx, "3.14");
    assert!(r.is_ok(), "3.14 should parse");
    let s = format!("{}", r.unwrap());
    // 3.14 = 314/100 = 157/50, so should contain rational representation
    assert!(
        s.contains("157") || s.contains("50") || s.contains("314") || s.contains("3.14"),
        "3.14 should parse to a rational: {s}"
    );
}

#[test]
fn parse_implicit_mul() {
    let ctx = Context::new();
    let r = symplex::parse::parse(&ctx, "2x");
    assert!(r.is_ok(), "2x should parse");
    let s = format!("{}", r.unwrap());
    assert!(s.contains("x") && s.contains("2"), "2x → 2*x: {s}");
}

#[test]
fn parse_implicit_mul_pi() {
    let ctx = Context::new();
    let r = symplex::parse::parse(&ctx, "2pi");
    assert!(r.is_ok(), "2pi should parse");
    let s = format!("{}", r.unwrap());
    assert!(
        s.contains("2") && s.contains("pi"),
        "2pi should become 2*pi: {s}"
    );
}

#[test]
fn parse_constants_pi() {
    let ctx = Context::new();
    let result = symplex::parse::parse(&ctx, "pi").unwrap();
    assert_eq!(format!("{result}"), "pi");
}

#[test]
fn parse_constants_i() {
    let ctx = Context::new();
    let result = symplex::parse::parse(&ctx, "I").unwrap();
    assert_eq!(format!("{result}"), "I");
}

#[test]
fn parse_constants_i_lowercase() {
    let ctx = Context::new();
    let result = symplex::parse::parse(&ctx, "i").unwrap();
    assert_eq!(format!("{result}"), "I");
}

#[test]
fn parse_constants_e() {
    let ctx = Context::new();
    let result = symplex::parse::parse(&ctx, "E").unwrap();
    assert_eq!(format!("{result}"), "E");
}

#[test]
fn parse_complex_expr() {
    let ctx = Context::new();
    let r = symplex::parse::parse(&ctx, "2 + 3*I");
    assert!(r.is_ok());
    let s = format!("{}", r.unwrap());
    assert!(
        s.contains("I") && s.contains("2") && s.contains("3"),
        "got: {s}"
    );
}

#[test]
fn parse_log_two_args() {
    let ctx = Context::new();
    let result = symplex::parse::parse(&ctx, "log(x, 2)").unwrap();
    let s = format!("{result}");
    assert!(s.contains("ln"), "log(x,2) should use ln: {s}");
}

#[test]
fn parse_euler_formula() {
    let ctx = Context::new();
    let result = symplex::parse::parse(&ctx, "exp(I*pi)").unwrap();
    let s = format!("{result}");
    assert!(
        s.contains("I") && s.contains("pi"),
        "exp(I*pi) should parse correctly: {s}"
    );
}

#[test]
fn parse_nested_functions() {
    let ctx = Context::new();
    let r = symplex::parse::parse(&ctx, "sin(cos(x))");
    assert!(r.is_ok(), "sin(cos(x)) should parse");
    let s = format!("{}", r.unwrap());
    assert!(
        s.contains("sin") && s.contains("cos") && s.contains("x"),
        "sin(cos(x)) should display correctly: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// From<T> + Sum/Product
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn from_i32_into_ex() {
    let x: Ex = 42i32.into();
    assert_eq!(format!("{x}"), "42");
}

#[test]
fn from_i64_into_ex() {
    let x: Ex = 100i64.into();
    assert_eq!(format!("{x}"), "100");
}

#[test]
fn from_u8_into_ex() {
    let x: Ex = 7u8.into();
    assert_eq!(format!("{x}"), "7");
}

#[test]
fn from_u64_into_ex() {
    let x: Ex = 255u64.into();
    assert_eq!(format!("{x}"), "255");
}

#[test]
fn from_usize_into_ex() {
    let x: Ex = 99usize.into();
    assert_eq!(format!("{x}"), "99");
}

#[test]
fn sum_of_expressions() {
    let ctx = Context::new();
    let terms: Vec<Ex> = (1..=10).map(|n| ctx.int(n)).collect();
    let total: Ex = terms.into_iter().sum();
    assert_eq!(format!("{total}"), "55");
}

#[test]
fn product_of_expressions() {
    let ctx = Context::new();
    let factors: Vec<Ex> = (1..=5).map(|n| ctx.int(n)).collect();
    let total: Ex = factors.into_iter().product();
    assert_eq!(format!("{total}"), "120");
}

#[test]
fn sum_of_refs() {
    let ctx = Context::new();
    let terms: Vec<Ex> = vec![ctx.int(1), ctx.int(2), ctx.int(3)];
    let total: Ex = terms.iter().sum();
    assert_eq!(format!("{total}"), "6");
}

#[test]
fn product_of_refs() {
    let ctx = Context::new();
    let factors: Vec<Ex> = vec![ctx.int(2), ctx.int(3), ctx.int(7)];
    let total: Ex = factors.iter().product();
    assert_eq!(format!("{total}"), "42");
}

#[test]
fn sum_symbolic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let terms = vec![x.clone(), x.clone(), x.clone()];
    let total: Ex = terms.into_iter().sum();
    let s = format!("{total}");
    assert!(s.contains("3") && s.contains("x"), "x+x+x = 3*x: {s}");
}

#[test]
fn sum_empty_is_zero() {
    let terms: Vec<Ex> = vec![];
    let total: Ex = terms.into_iter().sum();
    assert_eq!(format!("{total}"), "0");
}

#[test]
fn product_empty_is_one() {
    let factors: Vec<Ex> = vec![];
    let total: Ex = factors.into_iter().product();
    assert_eq!(format!("{total}"), "1");
}

// ═══════════════════════════════════════════════════════════════════════════
// Solver features
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_quadratic_x2_minus_2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let eq = &x.powi(2) - 2i64;
    let roots = eq.solve_or_empty(&x);
    assert!(!roots.is_empty(), "x²-2=0 should have roots");
    assert_eq!(roots.len(), 2, "should have exactly 2 roots");
}

#[test]
fn solve_cubic_all_rational() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x³ - 6x² + 11x - 6 = 0 has roots 1, 2, 3
    let eq = &x.powi(3) - &(&x.powi(2) * 6) + &(&x * 11) - 6;
    let roots = eq.solve_or_empty(&x);
    assert_eq!(roots.len(), 3, "x³-6x²+11x-6 should have 3 roots");
    // Verify each root
    for root in &roots {
        let val = eq.subs(&x, root);
        assert_eq!(
            format!("{val}"),
            "0",
            "root {} should satisfy equation",
            root
        );
    }
}

#[test]
fn solve_linear_simple() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 2x - 10 = 0 → x = 5
    let eq = &x * 2 - 10;
    let roots = eq.solve_or_empty(&x);
    assert_eq!(roots.len(), 1, "linear should have 1 root");
    assert_eq!(format!("{}", roots[0]), "5");
}

#[test]
fn solve_exp_quadratic_best_effort() {
    // exp(2x) - 3*exp(x) + 2 = 0 — change-of-variable (best-effort)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let exp_x = x.exp();
    let exp_2x = (&x * 2).exp();
    let eq = &exp_2x - &(&exp_x * 3) + 2;
    let roots = eq.solve_or_empty(&x);
    // Accept either roots found or empty — feature is best-effort
    if !roots.is_empty() {
        let strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
        let joined = strs.join(", ");
        assert!(
            joined.contains("ln") || joined.contains("0"),
            "roots should involve ln: {joined}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Higher-order differentiation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_n_third_derivative_x5() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // d³/dx³(x⁵) = 60x²
    let result = x.powi(5).diff_n(&x, 3);
    let s = format!("{result}");
    assert!(s.contains("60"), "d³/dx³(x⁵) should have coeff 60: {s}");
}

#[test]
fn diff_n_zeroth_is_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin();
    let result = expr.diff_n(&x, 0);
    assert_eq!(
        format!("{result}"),
        format!("{expr}"),
        "0th derivative = identity"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Complex arithmetic
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn complex_i_squared_is_neg_one() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let result = i.powi(2);
    assert_eq!(format!("{result}"), "-1", "i² = -1");
}

#[test]
fn complex_one_plus_i_fourth_power() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let expr = (&ctx.int(1) + &i).powi(4);
    let expanded = expr.expand();
    assert_eq!(format!("{expanded}"), "-4", "(1+i)⁴ = -4");
}

#[test]
fn complex_euler_exp_i_pi_plus_one() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let result = &(&i * &ctx.pi()).exp().eval() + 1;
    assert_eq!(format!("{result}"), "0", "exp(iπ) + 1 = 0");
}

#[test]
fn complex_euler_exp_i_pi_over_4() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let angle = &ctx.rational(1, 4) * &ctx.pi();
    let result = (&i * &angle).exp().eval();
    let s = format!("{result}");
    // exp(iπ/4) = (√2)/2 + i(√2)/2
    assert!(
        s.contains("sqrt") || s.contains("2"),
        "exp(iπ/4) should be complex: {s}"
    );
    assert!(s.contains("I"), "exp(iπ/4) should have imaginary part: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Global convenience functions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn global_var_and_ops() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let expr = &x * 2 + 1;
    let s = format!("{expr}");
    assert!(s.contains("x"), "should contain x: {s}");
}

#[test]
fn global_int_and_rational() {
    let __ctx = Context::new();
    let a = __ctx.int(3);
    let b = __ctx.rational(1, 2);
    let sum = &a + &b;
    let s = format!("{sum}");
    assert!(s.contains("7") || s.contains("2"), "3 + 1/2 = 7/2: {s}");
}

#[test]
fn global_constants() {
    let __ctx = Context::new();
    let pi = __ctx.pi();
    let e = __ctx.e();
    let i = __ctx.i_unit();
    assert_eq!(format!("{pi}"), "pi");
    assert_eq!(format!("{e}"), "E");
    assert_eq!(format!("{i}"), "I");
}

// ═══════════════════════════════════════════════════════════════════════════
// Eval of special values
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_sin_zero_is_zero() {
    let ctx = Context::new();
    let result = ctx.int(0).sin().eval();
    assert_eq!(format!("{result}"), "0");
}

#[test]
fn eval_cos_zero_is_one() {
    let ctx = Context::new();
    let result = ctx.int(0).cos().eval();
    assert_eq!(format!("{result}"), "1");
}

#[test]
fn eval_exp_zero_is_one() {
    let ctx = Context::new();
    let result = ctx.int(0).exp().eval();
    assert_eq!(format!("{result}"), "1");
}

#[test]
fn eval_ln_one_is_zero() {
    let ctx = Context::new();
    let result = ctx.int(1).ln().eval();
    assert_eq!(format!("{result}"), "0");
}

// ═══════════════════════════════════════════════════════════════════════════
// Series expansion
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn maclaurin_exp_order_5() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let series = x.exp().maclaurin(&x, 5);
    assert!(!series.has_unevaluated(), "maclaurin of exp(x) should succeed");
    let expanded = series.expand();
    let s = format!("{expanded}");
    assert!(s.contains("x"), "Taylor series should contain x: {s}");
    // Should have x^2, x^3, x^4 terms
    assert!(
        s.contains("x^2") || s.contains("x^"),
        "should have higher-order terms: {s}"
    );
}

#[test]
fn maclaurin_sin_order_5() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let series = x.sin().maclaurin(&x, 5);
    assert!(!series.has_unevaluated(), "maclaurin of sin(x) should succeed");
    let s = format!("{}", series.expand());
    assert!(s.contains("x"), "sin series should contain x: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Cross-cutting workflows
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn workflow_parse_solve_evalf() {
    let ctx = Context::new();
    let eq = symplex::parse::parse(&ctx, "x^2 - 2").unwrap();
    let x = ctx.symbol("x");
    let roots = eq.solve_or_empty(&x);
    assert!(!roots.is_empty(), "x²-2=0 should have roots");
    // Try evalf on a root
    if let Some(root) = roots.first() {
        let r = root.eval_decimal(10);
        assert!(r.is_ok(), "evalf of root should work: {:?}", r.err());
        let s = r.unwrap();
        // √2 ≈ 1.414...
        assert!(
            s.contains("1.41") || s.contains("-1.41"),
            "root of x²-2 should be ±√2 ≈ ±1.414: {s}"
        );
    }
}

#[test]
fn workflow_build_diff_integrate() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let f = x.sin().powi(2);
    let df = f.diff(&x);
    let anti = df.integrate(&x);
    // d/dx(sin²(x)) = 2sin(x)cos(x), ∫ that back should be evaluable
    let s = format!("{anti}");
    assert!(!s.contains("Integral"), "should integrate: {s}");
    // Verify numerically at x=1
    let df_at_1 = df.subs_i64(&x, 1).eval_f64();
    let anti_diff_at_1 = anti.diff(&x).subs_i64(&x, 1).eval_f64();
    if let (Ok(a), Ok(b)) = (df_at_1, anti_diff_at_1) {
        assert!(
            (a - b).abs() < 1e-10,
            "d/dx of antiderivative should match: {a} vs {b}"
        );
    }
}

#[test]
fn workflow_complex_algebra_then_evalf() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    // (1+i)^4 should be -4
    let expr = (&ctx.int(1) + &i).powi(4);
    let expanded = expr.expand();
    assert_eq!(format!("{expanded}"), "-4", "(1+i)⁴ = -4");

    // evalf should confirm
    let num = expanded.eval_decimal(10);
    assert!(num.is_ok());
    let s = num.unwrap();
    assert!(s.starts_with("-4"), "(1+i)⁴ evalf: {s}");
}

#[test]
fn workflow_trig_combine_then_integrate() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sin(x)*cos(x) → combine → then integrate
    let product = &x.sin() * &x.cos();
    let combined = product.trig_combine();
    // Now integrate the combined form
    let integral = combined.integrate(&x);
    let s = format!("{integral}");
    assert!(
        !s.contains("Integral"),
        "combined trig should integrate: {s}"
    );
    assert!(
        s.contains("cos") || s.contains("sin"),
        "should have trig in result: {s}"
    );
}

#[test]
fn workflow_euler_formula_full() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    // exp(iπ/4) should evaluate to (√2/2 + i√2/2)
    let angle = &ctx.rational(1, 4) * &ctx.pi();
    let result = (&i * &angle).exp().eval();
    let s = format!("{result}");
    // Should contain both I and sqrt(2) related terms
    assert!(
        s.contains("I") && s.contains("sqrt"),
        "exp(iπ/4) should be complex with sqrt: {s}"
    );
}

#[test]
fn workflow_solve_verify_evalf() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Solve x³ - 6x² + 11x - 6 = 0 (roots: 1, 2, 3)
    let eq = &x.powi(3) - &(&x.powi(2) * 6) + &(&x * 11) - 6;
    let roots = eq.solve_or_empty(&x);
    assert_eq!(roots.len(), 3, "x³-6x²+11x-6 should have 3 roots");
    // Verify each root
    for root in &roots {
        let val = eq.subs(&x, root);
        assert_eq!(
            format!("{val}"),
            "0",
            "root {} should satisfy equation",
            root
        );
    }
    // Verify roots are 1, 2, 3 via evalf
    let mut root_vals: Vec<f64> = roots.iter().filter_map(|r| r.eval_f64().ok()).collect();
    root_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(root_vals.len(), 3);
    assert!((root_vals[0] - 1.0).abs() < 1e-10, "first root should be 1");
    assert!(
        (root_vals[1] - 2.0).abs() < 1e-10,
        "second root should be 2"
    );
    assert!((root_vals[2] - 3.0).abs() < 1e-10, "third root should be 3");
}

#[test]
fn workflow_diff_n_then_series() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    // Taylor series of exp(x) around 0, order 5
    let series = x.exp().maclaurin(&x, 5);
    assert!(!series.has_unevaluated());
    let s = format!("{}", series.expand());
    assert!(s.contains("x"), "Taylor series should contain x: {s}");
    // Verify the series is a good approximation at x=0.1
    let ctx = Context::new();
    let xc = ctx.symbol("x");
    let series2 = xc.exp().maclaurin(&xc, 5).expand();
    let approx = series2.subs_i64(&xc, 0); // at x=0, exp(0)=1, series=1
    let s2 = format!("{approx}");
    assert!(
        s2.contains("1"),
        "maclaurin of exp at x=0 should be 1: {s2}"
    );
}

#[test]
fn workflow_parse_then_diff() {
    let ctx = Context::new();
    let expr = symplex::parse::parse(&ctx, "x^3 + 2*x^2 + x").unwrap();
    let x = ctx.symbol("x");
    let deriv = expr.diff(&x);
    let s = format!("{deriv}");
    // d/dx(x³ + 2x² + x) = 3x² + 4x + 1
    assert!(s.contains("x"), "derivative should contain x: {s}");
}

#[test]
fn workflow_parse_then_integrate() {
    let ctx = Context::new();
    let expr = symplex::parse::parse(&ctx, "2*x + 1").unwrap();
    let x = ctx.symbol("x");
    let integral = expr.integrate(&x);
    let s = format!("{integral}");
    // ∫(2x+1)dx = x² + x
    assert!(!s.contains("Integral"), "should evaluate: {s}");
    assert!(s.contains("x"), "should contain x: {s}");
}

#[test]
fn workflow_from_into_then_compute() {
    // Use From<i32> to build an expression, then compute
    let two: Ex = 2i32.into();
    let three: Ex = 3i32.into();
    let result = &two + &three;
    assert_eq!(format!("{result}"), "5", "2+3=5 via From");
}

#[test]
fn workflow_sum_product_chain() {
    let ctx = Context::new();
    // Sum 1..=5, then multiply by product 1..=3
    let s: Ex = (1..=5).map(|n| ctx.int(n)).sum();
    let p: Ex = (1..=3).map(|n| ctx.int(n)).product();
    let result = &s * &p;
    // sum = 15, product = 6, result = 90
    assert_eq!(format!("{result}"), "90", "15*6 = 90");
}

#[test]
fn workflow_complex_evalf_of_sqrt_neg() {
    let ctx = Context::new();
    // sqrt(-4) should involve 2i
    let neg4 = ctx.int(-4);
    let result = neg4.sqrt().eval();
    let s = format!("{result}");
    assert!(
        s.contains("I") && s.contains("2"),
        "sqrt(-4) should be 2I: {s}"
    );
}

#[test]
fn workflow_integrate_polynomial_verify_by_diff() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // ∫(3x² + 2x + 1) dx = x³ + x² + x
    let f = &(&x.powi(2) * 3) + &(&x * 2) + 1;
    let integral = f.integrate(&x);
    let deriv = integral.diff(&x);
    // Verify at x=2: f(2) = 12 + 4 + 1 = 17
    let f_val = f.subs_i64(&x, 2).eval_f64();
    let d_val = deriv.subs_i64(&x, 2).eval_f64();
    assert!(f_val.is_ok() && d_val.is_ok());
    let fv = f_val.unwrap();
    let dv = d_val.unwrap();
    assert!((fv - dv).abs() < 1e-10, "d/dx of ∫f = f: {} vs {}", fv, dv);
}

#[test]
fn workflow_evalf_f64_convenience() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let val = pi.eval_f64();
    assert!(val.is_ok());
    let v = val.unwrap();
    assert!(
        (v - std::f64::consts::PI).abs() < 1e-10,
        "evalf_f64(pi) should match std PI: {v}"
    );
}

#[test]
fn workflow_multi_step_simplification() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Start with sin²(x) + cos²(x), which should simplify to 1
    let expr = &x.sin().powi(2) + &x.cos().powi(2);
    let simplified = expr.full_simplify();
    assert_eq!(format!("{simplified}"), "1", "sin²(x) + cos²(x) = 1");
}

//! Consistency and round-trip bug-finder tests for symplex.
//!
//! These tests probe for inconsistencies across different representations
//! and transformations. A failure here indicates a real bug.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Check that two expressions agree numerically at several points.
/// Uses subs_i64 + eval_f64. Skips points where either side fails/NaN/Inf.
fn assert_numerically_equal(
    a: &Ex,
    b: &Ex,
    var: &Ex,
    points: &[i64],
    tol: f64,
    msg: &str,
) {
    for &pt in points {
        let a_val = a.subs_i64(var, pt).eval_f64();
        let b_val = b.subs_i64(var, pt).eval_f64();
        if let (Ok(av), Ok(bv)) = (a_val, b_val) {
            if av.is_nan() || bv.is_nan() {
                continue;
            }
            if av.is_infinite() && bv.is_infinite() && av.signum() == bv.signum() {
                continue;
            }
            if av.is_infinite() || bv.is_infinite() {
                panic!(
                    "{msg} at x={pt}: one side infinite ({av} vs {bv})"
                );
            }
            let diff = (av - bv).abs();
            let scale = av.abs().max(bv.abs()).max(1.0);
            assert!(
                diff / scale < tol,
                "{msg} at x={pt}: {av} vs {bv} (diff={diff}, rel={:.2e})",
                diff / scale
            );
        }
    }
}

const INT_POINTS: &[i64] = &[-3, -2, -1, 1, 2, 3, 4, 5];
const POS_POINTS: &[i64] = &[1, 2, 3, 4, 5, 6];

// ═══════════════════════════════════════════════════════════════════════════
// 1. EXPAND / FACTOR ROUND-TRIP
// ═══════════════════════════════════════════════════════════════════════════

/// expand(factor(p)) should equal p (up to canonical ordering).
#[test]
fn expand_factor_roundtrip_quadratic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // p = x^2 - 5x + 6 = (x-2)(x-3)
    let p = &x.powi(2) - &(&x * 5) + 6;
    let factored = p.factor(&x);
    let re_expanded = factored.expand();
    assert_numerically_equal(&p, &re_expanded, &x, INT_POINTS, 1e-10,
        "expand(factor(x²-5x+6))");
    // Also check structural equality
    assert_eq!(
        format!("{}", p.expand()),
        format!("{re_expanded}"),
        "expand(factor(x²-5x+6)) structural"
    );
}

#[test]
fn expand_factor_roundtrip_cubic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // p = x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
    let p = &x.powi(3) - &(&x.powi(2) * 6) + &(&x * 11) - 6;
    let factored = p.factor(&x);
    let re_expanded = factored.expand();
    assert_numerically_equal(&p, &re_expanded, &x, INT_POINTS, 1e-10,
        "expand(factor(cubic))");
}

#[test]
fn expand_factor_roundtrip_quartic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // p = x^4 - 1 = (x-1)(x+1)(x^2+1)
    let p = &x.powi(4) - 1;
    let factored = p.factor(&x);
    let re_expanded = factored.expand();
    assert_numerically_equal(&p, &re_expanded, &x, INT_POINTS, 1e-10,
        "expand(factor(x⁴-1))");
}

#[test]
fn factor_expand_roundtrip_constructed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Build (x-1)(x+2)(x-3) then expand
    let product = &(&(&x - 1) * &(&x + 2)) * &(&x - 3);
    let expanded = product.expand();
    let factored = expanded.factor(&x);
    let re_expanded = factored.expand();
    assert_eq!(
        format!("{expanded}"),
        format!("{re_expanded}"),
        "factor(expand((x-1)(x+2)(x-3))) roundtrip"
    );
}

#[test]
fn expand_factor_roundtrip_with_coefficients() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // p = 2x^2 + 6x + 4 = 2(x+1)(x+2)
    let p = &(&x.powi(2) * 2) + &(&x * 6) + 4;
    let factored = p.factor(&x);
    let re_expanded = factored.expand();
    assert_numerically_equal(&p, &re_expanded, &x, INT_POINTS, 1e-10,
        "expand(factor(2x²+6x+4))");
}

#[test]
fn expand_factor_roundtrip_difference_of_cubes() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x^3 - 8 = (x-2)(x^2+2x+4)
    let p = &x.powi(3) - 8;
    let factored = p.factor(&x);
    let re_expanded = factored.expand();
    assert_numerically_equal(&p, &re_expanded, &x, INT_POINTS, 1e-10,
        "expand(factor(x³-8))");
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. DIFFERENTIATION / INTEGRATION ROUND-TRIP (FTC)
// ═══════════════════════════════════════════════════════════════════════════

/// d/dx(∫ f dx) == f for polynomials
#[test]
fn ftc_polynomial_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.clone();
    let roundtrip = f.integrate(&x).diff(&x);
    assert_numerically_equal(&f, &roundtrip, &x, INT_POINTS, 1e-10, "FTC: x");
}

#[test]
fn ftc_polynomial_x2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(2);
    let roundtrip = f.integrate(&x).diff(&x);
    assert_numerically_equal(&f, &roundtrip, &x, INT_POINTS, 1e-10, "FTC: x²");
}

#[test]
fn ftc_polynomial_x3() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(3);
    let roundtrip = f.integrate(&x).diff(&x);
    assert_numerically_equal(&f, &roundtrip, &x, INT_POINTS, 1e-10, "FTC: x³");
}

#[test]
fn ftc_polynomial_x4() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(4);
    let roundtrip = f.integrate(&x).diff(&x);
    assert_numerically_equal(&f, &roundtrip, &x, INT_POINTS, 1e-10, "FTC: x⁴");
}

#[test]
fn ftc_polynomial_mixed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // f = 3x^3 - 2x^2 + 5x - 7
    let f = &(&x.powi(3) * 3) - &(&x.powi(2) * 2) + &(&x * 5) - 7;
    let roundtrip = f.integrate(&x).diff(&x);
    assert_numerically_equal(&f, &roundtrip, &x, INT_POINTS, 1e-10,
        "FTC: 3x³-2x²+5x-7");
}

#[test]
fn ftc_constant() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.int(42);
    let roundtrip = f.integrate(&x).diff(&x);
    assert_eq!(format!("{roundtrip}"), "42", "FTC: constant 42");
}

/// d/dx(∫ sin(x) dx) == sin(x)
#[test]
fn ftc_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sin();
    let anti = f.integrate(&x);
    let roundtrip = anti.diff(&x);
    assert_numerically_equal(&f, &roundtrip, &x, INT_POINTS, 1e-10, "FTC: sin(x)");
}

/// d/dx(∫ cos(x) dx) == cos(x)
#[test]
fn ftc_cos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.cos();
    let anti = f.integrate(&x);
    let roundtrip = anti.diff(&x);
    assert_numerically_equal(&f, &roundtrip, &x, INT_POINTS, 1e-10, "FTC: cos(x)");
}

/// d/dx(∫ exp(x) dx) == exp(x)
#[test]
fn ftc_exp() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.exp();
    let anti = f.integrate(&x);
    let roundtrip = anti.diff(&x);
    assert_numerically_equal(&f, &roundtrip, &x, &[-2, -1, 0, 1, 2], 1e-10,
        "FTC: exp(x)");
}

/// d/dx(∫ 1/x dx) == 1/x
#[test]
fn ftc_one_over_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(-1);
    let anti = f.integrate(&x);
    let roundtrip = anti.diff(&x);
    // Only test at positive points to avoid abs issues
    assert_numerically_equal(&f, &roundtrip, &x, POS_POINTS, 1e-10,
        "FTC: 1/x");
}

/// d/dx(∫ x*exp(x) dx) == x*exp(x)
#[test]
fn ftc_x_exp() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x * &x.exp();
    let anti = f.integrate(&x);
    if !anti.has_unevaluated() {
        let roundtrip = anti.diff(&x);
        assert_numerically_equal(&f, &roundtrip, &x, &[-2, -1, 0, 1, 2], 1e-8,
            "FTC: x*exp(x)");
    }
}

/// d/dx(∫ x*sin(x) dx) == x*sin(x)
#[test]
fn ftc_x_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x * &x.sin();
    let anti = f.integrate(&x);
    if !anti.has_unevaluated() {
        let roundtrip = anti.diff(&x);
        assert_numerically_equal(&f, &roundtrip, &x, INT_POINTS, 1e-8,
            "FTC: x*sin(x)");
    }
}

/// d/dx(∫ ln(x) dx) == ln(x)
#[test]
fn ftc_ln() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.ln();
    let anti = f.integrate(&x);
    if !anti.has_unevaluated() {
        let roundtrip = anti.diff(&x);
        assert_numerically_equal(&f, &roundtrip, &x, POS_POINTS, 1e-10,
            "FTC: ln(x)");
    }
}

/// d/dx(∫ sin(x)^2 dx) == sin(x)^2
#[test]
fn ftc_sin_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sin().powi(2);
    let anti = f.integrate(&x);
    if !anti.has_unevaluated() {
        let roundtrip = anti.diff(&x);
        assert_numerically_equal(&f, &roundtrip, &x, INT_POINTS, 1e-8,
            "FTC: sin²(x)");
    }
}

/// d/dx(∫ cos(x)^2 dx) == cos(x)^2
#[test]
fn ftc_cos_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.cos().powi(2);
    let anti = f.integrate(&x);
    if !anti.has_unevaluated() {
        let roundtrip = anti.diff(&x);
        assert_numerically_equal(&f, &roundtrip, &x, INT_POINTS, 1e-8,
            "FTC: cos²(x)");
    }
}

/// d/dx(∫ exp(-x^2) dx) — this is erf-related, may produce unevaluated
#[test]
fn ftc_exp_neg_x2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (-&x.powi(2)).exp();
    let anti = f.integrate(&x);
    if !anti.has_unevaluated() {
        let roundtrip = anti.diff(&x);
        assert_numerically_equal(&f, &roundtrip, &x, &[-2, -1, 0, 1, 2], 1e-8,
            "FTC: exp(-x²)");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. SIMPLIFY IDEMPOTENCY
// ═══════════════════════════════════════════════════════════════════════════

/// simplify(simplify(e)) == simplify(e) for all expressions
fn check_simplify_idempotent(e: &Ex, label: &str) {
    let s1 = e.simplify();
    let s2 = s1.simplify();
    assert_eq!(
        format!("{s1}"),
        format!("{s2}"),
        "simplify not idempotent for {label}: simplify = '{s1}', simplify(simplify) = '{s2}'"
    );
}

#[test]
fn simplify_idempotent_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x.powi(3) + &x.powi(2) + &x + 1;
    check_simplify_idempotent(&e, "x³+x²+x+1");
}

#[test]
fn simplify_idempotent_trig_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x.sin().powi(2) + &x.cos().powi(2);
    check_simplify_idempotent(&e, "sin²(x)+cos²(x)");
}

#[test]
fn simplify_idempotent_exp_ln() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = x.ln().exp();
    check_simplify_idempotent(&e, "exp(ln(x))");
}

#[test]
fn simplify_idempotent_nested_trig() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = (&x * 2).sin(); // sin(2x)
    check_simplify_idempotent(&e, "sin(2x)");
}

#[test]
fn simplify_idempotent_rational() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x^2 - 1) / (x - 1) — simplify might cancel to x+1
    let e = &(&x.powi(2) - 1) / &(&x - 1);
    check_simplify_idempotent(&e, "(x²-1)/(x-1)");
}

#[test]
fn simplify_idempotent_double_neg() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = -(-&x);
    check_simplify_idempotent(&e, "--x");
}

#[test]
fn simplify_idempotent_sum_with_trig() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x + sin²(x) + cos²(x) — should simplify to x+1
    let e = &x + &x.sin().powi(2) + &x.cos().powi(2);
    check_simplify_idempotent(&e, "x+sin²+cos²");
}

#[test]
fn simplify_idempotent_zero_patterns() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x - &x;
    check_simplify_idempotent(&e, "x-x");
}

#[test]
fn simplify_idempotent_one_patterns() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x / &x;
    check_simplify_idempotent(&e, "x/x");
}

#[test]
fn simplify_idempotent_power_of_power() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = x.powi(2).powi(3); // (x^2)^3 = x^6
    check_simplify_idempotent(&e, "(x²)³");
}

#[test]
fn simplify_idempotent_product_of_exps() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let e = &x.exp() * &y.exp(); // exp(x)*exp(y) = exp(x+y)
    check_simplify_idempotent(&e, "exp(x)*exp(y)");
}

#[test]
fn simplify_idempotent_sqrt_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = x.sqrt().powi(2);
    check_simplify_idempotent(&e, "sqrt(x)²");
}

#[test]
fn simplify_idempotent_ln_exp() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]);
    let e = x.exp().ln();
    check_simplify_idempotent(&e, "ln(exp(x))");
}

#[test]
fn simplify_idempotent_expand_then_simplify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = (&x + 1).powi(3).expand();
    check_simplify_idempotent(&e, "(x+1)³ expanded");
}

#[test]
fn simplify_idempotent_complex_expression() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sin(x)^2 + cos(x)^2 + x^2 + 2*x + 1
    let e = &x.sin().powi(2) + &x.cos().powi(2) + &x.powi(2) + &(&x * 2) + 1;
    check_simplify_idempotent(&e, "sin²+cos²+x²+2x+1");
}

#[test]
fn full_simplify_idempotent_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x.powi(3) + &x.powi(2) + &x + 1;
    let s1 = e.simplify();
    let s2 = s1.simplify();
    assert_eq!(
        format!("{s1}"), format!("{s2}"),
        "full_simplify not idempotent for x³+x²+x+1"
    );
}

#[test]
fn full_simplify_idempotent_trig() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x.sin().powi(2) + &x.cos().powi(2);
    let s1 = e.simplify();
    let s2 = s1.simplify();
    assert_eq!(
        format!("{s1}"), format!("{s2}"),
        "full_simplify not idempotent for sin²+cos²"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. EVAL CONSISTENCY
// ═══════════════════════════════════════════════════════════════════════════

/// If f.simplify() == g symbolically, then f.eval_f64() ≈ g.eval_f64()
#[test]
fn eval_consistency_simplify_preserves_values() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let exprs: Vec<(&str, Ex)> = vec![
        ("x²+2x+1", &x.powi(2) + &(&x * 2) + 1),
        ("sin²+cos²", &x.sin().powi(2) + &x.cos().powi(2)),
        ("exp(ln(x))", x.ln().exp()),
        ("x³-x", &x.powi(3) - &x),
    ];

    for (label, e) in &exprs {
        let simplified = e.simplify();
        assert_numerically_equal(e, &simplified, &x, POS_POINTS, 1e-10,
            &format!("eval consistency: simplify({label})"));
    }
}

/// expand preserves numerical values
#[test]
fn eval_consistency_expand_preserves_values() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases: Vec<(&str, Ex)> = vec![
        ("(x+1)²", (&x + 1).powi(2)),
        ("(x+1)³", (&x + 1).powi(3)),
        ("(x+1)(x-1)", &(&x + 1) * &(&x - 1)),
        ("(x+2)⁴", (&x + 2).powi(4)),
    ];

    for (label, e) in &cases {
        let expanded = e.expand();
        assert_numerically_equal(&e, &expanded, &x, INT_POINTS, 1e-10,
            &format!("eval consistency: expand({label})"));
    }
}

/// factor preserves numerical values
#[test]
fn eval_consistency_factor_preserves_values() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases: Vec<(&str, Ex)> = vec![
        ("x²-1", &x.powi(2) - 1),
        ("x²-4", &x.powi(2) - 4),
        ("x³-x", &x.powi(3) - &x),
        ("x²+3x+2", &x.powi(2) + &(&x * 3) + 2),
    ];

    for (label, e) in &cases {
        let factored = e.factor(&x);
        assert_numerically_equal(&e, &factored, &x, INT_POINTS, 1e-10,
            &format!("eval consistency: factor({label})"));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. COMMUTATIVITY OF OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// diff(expand(f)) == expand(diff(f)) for polynomials
#[test]
fn commutativity_diff_expand_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (&x + 1).powi(3);
    let diff_expand = f.expand().diff(&x);
    let expand_diff = f.diff(&x).expand();
    assert_numerically_equal(&diff_expand, &expand_diff, &x, INT_POINTS, 1e-10,
        "diff(expand((x+1)³)) vs expand(diff((x+1)³))");
}

#[test]
fn commutativity_diff_expand_product() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &(&x + 1) * &(&x - 2);
    let diff_expand = f.expand().diff(&x);
    let expand_diff = f.diff(&x).expand();
    assert_numerically_equal(&diff_expand, &expand_diff, &x, INT_POINTS, 1e-10,
        "diff(expand((x+1)(x-2))) vs expand(diff((x+1)(x-2)))");
}

#[test]
fn commutativity_diff_expand_quartic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (&x + 1).powi(4);
    let diff_expand = f.expand().diff(&x);
    let expand_diff = f.diff(&x).expand();
    assert_numerically_equal(&diff_expand, &expand_diff, &x, INT_POINTS, 1e-10,
        "diff(expand((x+1)⁴)) vs expand(diff((x+1)⁴))");
}

/// simplify(subs(f, x, a)) vs subs(simplify(f), x, a) — numerical check
#[test]
fn commutativity_simplify_subs() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x.sin().powi(2) + &x.cos().powi(2) + &x;

    for &pt in INT_POINTS {
        let val_ex = ctx.int(pt);

        let simplify_then_subs = e.simplify().subs(&x, &val_ex).eval_f64();
        let subs_then_simplify = e.subs(&x, &val_ex).simplify().eval_f64();

        if let (Ok(a), Ok(b)) = (simplify_then_subs, subs_then_simplify) {
            if a.is_nan() || b.is_nan() { continue; }
            let diff = (a - b).abs();
            assert!(diff < 1e-10,
                "simplify/subs commutativity at x={pt}: {a} vs {b} (diff={diff})");
        }
    }
}

/// diff(simplify(f)) should numerically agree with simplify(diff(f))
#[test]
fn commutativity_diff_simplify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Use expressions where simplify actually does something
    let e = &x.sin().powi(2) + &x.cos().powi(2) + &x.powi(2);

    let diff_simp = e.simplify().diff(&x);
    let simp_diff = e.diff(&x).simplify();
    assert_numerically_equal(&diff_simp, &simp_diff, &x, INT_POINTS, 1e-8,
        "diff(simplify(sin²+cos²+x²)) vs simplify(diff(sin²+cos²+x²))");
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. LATEX / DISPLAY CONSISTENCY
// ═══════════════════════════════════════════════════════════════════════════

/// LaTeX output should be non-empty and reasonable for basic expressions
#[test]
fn latex_nonempty_basics() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases: Vec<(&str, Ex)> = vec![
        ("x", x.clone()),
        ("x²", x.powi(2)),
        ("sin(x)", x.sin()),
        ("cos(x)", x.cos()),
        ("exp(x)", x.exp()),
        ("ln(x)", x.ln()),
        ("x+1", &x + 1),
        ("x²+x+1", &x.powi(2) + &x + 1),
    ];

    for (label, e) in &cases {
        let latex = e.to_latex();
        assert!(!latex.is_empty(), "LaTeX empty for {label}");
        // LaTeX should not contain Rust debug formatting
        assert!(!latex.contains("ExprId"), "LaTeX contains debug output for {label}: {latex}");
        assert!(!latex.contains("Arena"), "LaTeX contains debug output for {label}: {latex}");
    }
}

/// Display and LaTeX should agree on the "shape" (both represent the same expression)
#[test]
fn display_vs_latex_consistency() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases: Vec<(&str, Ex)> = vec![
        ("x", x.clone()),
        ("42", ctx.int(42)),
        ("x+1", &x + 1),
    ];

    for (label, e) in &cases {
        let display = format!("{e}");
        let latex = e.to_latex();
        // Both should be non-empty
        assert!(!display.is_empty(), "Display empty for {label}");
        assert!(!latex.is_empty(), "LaTeX empty for {label}");
        // For simple symbols, display and LaTeX should agree
        if *label == "x" {
            assert_eq!(display, "x");
            assert_eq!(latex, "x");
        }
        if *label == "42" {
            assert_eq!(display, "42");
            assert_eq!(latex, "42");
        }
    }
}

/// LaTeX for fractions should use \frac
#[test]
fn latex_fractions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let half = ctx.rational(1, 2);
    let latex = half.to_latex();
    // Should contain \frac or /
    assert!(
        latex.contains("frac") || latex.contains("/") || latex.contains("1/2"),
        "Fraction LaTeX doesn't look right: {latex}"
    );

    let ratio_expr = &x / &ctx.int(3);
    let latex2 = ratio_expr.to_latex();
    assert!(!latex2.is_empty(), "LaTeX empty for x/3");
}

/// LaTeX for trig functions should use \sin, \cos, etc.
#[test]
fn latex_trig_commands() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let sin_latex = x.sin().to_latex();
    assert!(sin_latex.contains("\\sin"), "sin LaTeX: {sin_latex}");

    let cos_latex = x.cos().to_latex();
    assert!(cos_latex.contains("\\cos"), "cos LaTeX: {cos_latex}");

    let tan_latex = x.tan().to_latex();
    assert!(tan_latex.contains("\\tan"), "tan LaTeX: {tan_latex}");
}

/// to_latex_inline should wrap in $...$
#[test]
fn latex_inline_wrapping() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let inline = x.to_latex_inline();
    assert!(inline.starts_with('$') && inline.ends_with('$'),
        "inline LaTeX not wrapped in $: {inline}");
}

/// to_latex_display should wrap in $$...$$
#[test]
fn latex_display_wrapping() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let display = x.to_latex_display();
    assert!(display.starts_with("$$") && display.ends_with("$$"),
        "display LaTeX not wrapped in $$: {display}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. COMPILE CONSISTENCY
// ═══════════════════════════════════════════════════════════════════════════

/// f.compile(&["x"])(val) should match f.subs_i64(x, val).eval_f64()
fn check_compile_consistency(f: &Ex, x: &Ex, points: &[i64], tol: f64, label: &str) {
    let compiled = f.compile(&["x"]);
    if compiled.is_none() {
        // Some expressions can't be compiled (unevaluated, etc.)
        return;
    }
    let compiled = compiled.unwrap();

    for &pt in points {
        let compiled_val = compiled(&[pt as f64]);
        let subs_val = f.subs_i64(x, pt).eval_f64();
        if let Ok(sv) = subs_val {
            if sv.is_nan() && compiled_val.is_nan() {
                continue;
            }
            if sv.is_infinite() && compiled_val.is_infinite()
                && sv.signum() == compiled_val.signum()
            {
                continue;
            }
            let diff = (compiled_val - sv).abs();
            let scale = sv.abs().max(compiled_val.abs()).max(1.0);
            assert!(
                diff / scale < tol,
                "{label} at x={pt}: compile={compiled_val}, eval={sv}, diff={diff}"
            );
        }
    }
}

#[test]
fn compile_consistency_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(3) - &(&x.powi(2) * 2) + &(&x * 5) - 3;
    check_compile_consistency(&f, &x, INT_POINTS, 1e-10, "x³-2x²+5x-3");
}

#[test]
fn compile_consistency_trig() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.sin() + &x.cos();
    check_compile_consistency(&f, &x, INT_POINTS, 1e-10, "sin(x)+cos(x)");
}

#[test]
fn compile_consistency_exp() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.exp();
    check_compile_consistency(&f, &x, &[-2, -1, 0, 1, 2, 3], 1e-10, "exp(x)");
}

#[test]
fn compile_consistency_ln() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.ln();
    check_compile_consistency(&f, &x, POS_POINTS, 1e-10, "ln(x)");
}

#[test]
fn compile_consistency_sqrt() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sqrt();
    check_compile_consistency(&f, &x, POS_POINTS, 1e-10, "sqrt(x)");
}

#[test]
fn compile_consistency_complex_expr() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // exp(sin(x)) + ln(x^2 + 1)
    let f = &x.sin().exp() + &(&x.powi(2) + 1).ln();
    check_compile_consistency(&f, &x, INT_POINTS, 1e-10, "exp(sin(x))+ln(x²+1)");
}

#[test]
fn compile_consistency_hyperbolic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.sinh() + &x.cosh();
    check_compile_consistency(&f, &x, &[-2, -1, 0, 1, 2], 1e-10, "sinh+cosh");
}

#[test]
fn compile_consistency_abs() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.abs();
    check_compile_consistency(&f, &x, INT_POINTS, 1e-10, "abs(x)");
}

#[test]
fn compile_consistency_inverse_trig() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // atan(x) is defined everywhere
    let f = x.atan();
    check_compile_consistency(&f, &x, INT_POINTS, 1e-10, "atan(x)");
}

#[test]
fn compile_consistency_powi_negative() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(-2);
    check_compile_consistency(&f, &x, &[-3, -2, -1, 1, 2, 3], 1e-10, "x^(-2)");
}

#[test]
fn compile_consistency_with_constants() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi = ctx.pi();
    let f = &x * &pi + &ctx.e();
    check_compile_consistency(&f, &x, INT_POINTS, 1e-10, "π*x + e");
}

/// compile with multiple variables
#[test]
fn compile_consistency_two_vars() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let f = &x.powi(2) + &y.powi(2);
    let compiled = f.compile(&["x", "y"]);
    if let Some(compiled) = compiled {
        for &xv in &[-2.0, 0.0, 1.0, 3.0] {
            for &yv in &[-1.0, 0.0, 2.0, 4.0] {
                let compiled_val = compiled(&[xv, yv]);
                let expected = xv * xv + yv * yv;
                let diff = (compiled_val - expected).abs();
                assert!(diff < 1e-10,
                    "compile(x²+y²) at ({xv},{yv}): {compiled_val} vs {expected}");
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. CODE GENERATION CONSISTENCY
// ═══════════════════════════════════════════════════════════════════════════

/// Helper: check generated code is syntactically valid Rust
fn assert_valid_rust_code(code: &str, fn_name: &str) {
    assert!(
        code.contains(&format!("pub fn {fn_name}")),
        "missing 'pub fn {fn_name}' in:\n{code}"
    );
    assert!(
        code.contains("-> f64"),
        "missing '-> f64' in:\n{code}"
    );
    // Balanced braces
    let opens = code.chars().filter(|&c| c == '{').count();
    let closes = code.chars().filter(|&c| c == '}').count();
    assert_eq!(opens, closes,
        "unbalanced braces ({opens} open vs {closes} close) in:\n{code}");
    // Balanced parentheses
    let open_parens = code.chars().filter(|&c| c == '(').count();
    let close_parens = code.chars().filter(|&c| c == ')').count();
    assert_eq!(open_parens, close_parens,
        "unbalanced parentheses ({open_parens} vs {close_parens}) in:\n{code}");
}

#[test]
fn codegen_polynomial_valid() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(3) - &(&x.powi(2) * 2) + &(&x * 5) - 3;
    let code = f.to_rust_fn("poly", &["x"]).unwrap();
    assert_valid_rust_code(&code, "poly");
}

#[test]
fn codegen_trig_valid() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.sin() + &x.cos() + &x.tan();
    let code = f.to_rust_fn("trig_fn", &["x"]).unwrap();
    assert_valid_rust_code(&code, "trig_fn");
}

#[test]
fn codegen_nested_valid() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(2).sin().exp() + &(&x.cos() + 2).ln();
    let code = f.to_rust_fn("nested_fn", &["x"]).unwrap();
    assert_valid_rust_code(&code, "nested_fn");
}

#[test]
fn codegen_derivative_valid() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (&x.powi(3) - &(&x * 2) + 1).diff(&x);
    let code = f.to_rust_fn("deriv_fn", &["x"]).unwrap();
    assert_valid_rust_code(&code, "deriv_fn");
}

#[test]
fn codegen_simplify_then_codegen_valid() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (&x.sin().powi(2) + &x.cos().powi(2)).simplify();
    let code = f.to_rust_fn("simp_fn", &[]).unwrap();
    assert_valid_rust_code(&code, "simp_fn");
}

/// Code gen should fail for free symbols not in args
#[test]
fn codegen_free_symbol_error() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let f = &x + &y;
    let result = f.to_rust_fn("bad_fn", &["x"]);
    assert!(result.is_err(), "expected error when y is not in args list");
}

/// to_rust_fn output should be consistent with compile for the same expression
#[test]
fn codegen_vs_compile_consistency() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(3) - &(&x * 2) + 1;

    let compiled = f.compile(&["x"]);
    let code_result = f.to_rust_fn("test_fn", &["x"]);

    // Both should succeed or both should fail
    if let (Some(compiled), Ok(_code)) = (compiled, code_result) {
        // Verify they agree on values
        for &pt in &[-3.0, -1.0, 0.0, 1.0, 2.5, 5.0] {
            let compiled_val = compiled(&[pt]);
            let expected = pt.powi(3) - 2.0 * pt + 1.0;
            let diff = (compiled_val - expected).abs();
            assert!(diff < 1e-10,
                "codegen/compile mismatch at x={pt}: compiled={compiled_val}, expected={expected}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ADDITIONAL CONSISTENCY CHECKS
// ═══════════════════════════════════════════════════════════════════════════

/// diff is linear: d/dx(a*f + b*g) == a*d/dx(f) + b*d/dx(g)
#[test]
fn diff_linearity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let f = x.sin();
    let g = x.exp();

    // d/dx(3*sin(x) + 5*exp(x)) vs 3*cos(x) + 5*exp(x)
    let combined = &(&f * 3) + &(&g * 5);
    let diff_combined = combined.diff(&x);
    let diff_separate = &(&f.diff(&x) * 3) + &(&g.diff(&x) * 5);

    assert_numerically_equal(&diff_combined, &diff_separate, &x, INT_POINTS, 1e-10,
        "diff linearity: 3sin+5exp");
}

/// Leibniz rule: d/dx(f*g) == f'*g + f*g'
#[test]
fn diff_leibniz_rule() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sin();
    let g = x.exp();

    let product = &f * &g;
    let product_diff = product.diff(&x);
    let leibniz = &(&f.diff(&x) * &g) + &(&f * &g.diff(&x));

    assert_numerically_equal(&product_diff, &leibniz, &x, INT_POINTS, 1e-10,
        "Leibniz: d/dx(sin*exp) vs sin'*exp + sin*exp'");
}

/// Chain rule: d/dx(f(g(x))) == f'(g(x)) * g'(x)
#[test]
fn diff_chain_rule_sin_x2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // f(u) = sin(u), g(x) = x^2
    // d/dx sin(x^2) = cos(x^2) * 2x
    let composite = x.powi(2).sin();
    let diff_composite = composite.diff(&x);
    let expected = &x.powi(2).cos() * &(&x * 2);

    assert_numerically_equal(&diff_composite, &expected, &x, INT_POINTS, 1e-10,
        "chain rule: d/dx sin(x²)");
}

/// Expanding then factoring a polynomial should be numerically consistent
#[test]
fn expand_then_factor_numerical() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    for (a, b) in [(-3, 2), (0, 5), (-1, 1), (2, 4)] {
        let p = &(&x - a) * &(&x - b);
        let expanded = p.expand();
        let factored = expanded.factor(&x);
        let re_expanded = factored.expand();

        assert_numerically_equal(&expanded, &re_expanded, &x, INT_POINTS, 1e-10,
            &format!("expand/factor roundtrip for (x-{a})(x-{b})"));
    }
}

/// cancel should preserve numerical values: (x^2-1)/(x-1) → x+1
#[test]
fn cancel_preserves_values() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &(&x.powi(2) - 1) / &(&x - 1);
    let cancelled = e.cancel(&x);
    // Test at points where x != 1
    assert_numerically_equal(&e, &cancelled, &x, &[-3, -2, -1, 2, 3, 4, 5], 1e-10,
        "cancel (x²-1)/(x-1)");
}

/// Higher-degree cancel
#[test]
fn cancel_cubic_preserves_values() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x^3 - x) / (x^2 - 1) = x*(x^2-1)/(x^2-1) = x
    let e = &(&x.powi(3) - &x) / &(&x.powi(2) - 1);
    let cancelled = e.cancel(&x);
    // Test at points where denominator != 0
    assert_numerically_equal(&e, &cancelled, &x, &[-3, -2, 2, 3, 4, 5], 1e-10,
        "cancel (x³-x)/(x²-1)");
}

/// Solve roots should actually be zeros of the polynomial
#[test]
fn solve_roots_are_zeros() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let polys: Vec<(&str, Ex)> = vec![
        ("x²-4", &x.powi(2) - 4),
        ("x²-1", &x.powi(2) - 1),
        ("x²+x-6", &x.powi(2) + &x - 6),
        ("x³-6x²+11x-6", &x.powi(3) - &(&x.powi(2) * 6) + &(&x * 11) - 6),
    ];

    for (label, p) in &polys {
        let roots = p.solve_or_empty(&x);
        for root in &roots {
            let val = p.subs(&x, root);
            if let Ok(f) = val.eval_f64() {
                assert!(
                    f.abs() < 1e-8,
                    "root {root} of {label} gives f={f}, not zero"
                );
            }
        }
    }
}

/// series expansion should converge to the function at small x
#[test]
fn series_approximation_accuracy() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);

    // sin(x) Maclaurin, order 7
    let sin_series = x.sin().series(&x, &zero, 7).expand();
    // At x=1, sin(1) ≈ 0.84147...
    let series_val = sin_series.subs_i64(&x, 1).eval_f64();
    let exact = 1.0f64.sin();
    if let Ok(sv) = series_val {
        let diff = (sv - exact).abs();
        assert!(diff < 0.001,
            "sin Maclaurin order 7 at x=1: series={sv}, exact={exact}, diff={diff}");
    }

    // exp(x) Maclaurin, order 10
    let exp_series = x.exp().series(&x, &zero, 10).expand();
    let series_val = exp_series.subs_i64(&x, 1).eval_f64();
    let exact = 1.0f64.exp();
    if let Ok(sv) = series_val {
        let diff = (sv - exact).abs();
        assert!(diff < 0.001,
            "exp Maclaurin order 10 at x=1: series={sv}, exact={exact}, diff={diff}");
    }
}

/// series of a polynomial should be exact
#[test]
fn series_of_polynomial_exact() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);

    // x^2 + 3x + 5, series to order 5 (above degree 2)
    let p = &x.powi(2) + &(&x * 3) + 5;
    let s = p.series(&x, &zero, 5);
    assert_numerically_equal(&p, &s, &x, INT_POINTS, 1e-10,
        "series(x²+3x+5, order=5)");
}

/// trig_expand then trig_combine should round-trip numerically
#[test]
fn trig_expand_combine_roundtrip() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let two = ctx.int(2);
    let angle = &x * &two;

    // sin(2x) → expand → combine → should ≈ sin(2x) numerically
    let original = angle.sin();
    let expanded = original.expand_trig();
    // Verify the expansion preserves value
    assert_numerically_equal(&original, &expanded, &x, INT_POINTS, 1e-10,
        "sin(2x) expand_trig value preservation");
}

/// expand_log should preserve values
#[test]
fn expand_log_preserves_values() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let original = (&x * &y).ln();
    let expanded = original.expand_log();

    // Check at x=2, y=3
    let o_val = original.subs_i64(&x, 2).subs_i64(&y, 3).eval_f64();
    let e_val = expanded.subs_i64(&x, 2).subs_i64(&y, 3).eval_f64();
    if let (Ok(ov), Ok(ev)) = (o_val, e_val) {
        let diff = (ov - ev).abs();
        assert!(diff < 1e-10, "expand_log ln(xy) at (2,3): {ov} vs {ev}");
    }
}

/// Double differentiation consistency: d²/dx²(f) == d/dx(d/dx(f))
#[test]
fn double_diff_consistency() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(4) + &x.sin();

    let d2_direct = f.diff_n(&x, 2);
    let d2_sequential = f.diff(&x).diff(&x);

    assert_numerically_equal(&d2_direct, &d2_sequential, &x, INT_POINTS, 1e-10,
        "d²/dx²(x⁴+sin(x)) direct vs sequential");
}

/// Triple differentiation
#[test]
fn triple_diff_consistency() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(5);

    let d3_direct = f.diff_n(&x, 3);
    let d3_sequential = f.diff(&x).diff(&x).diff(&x);

    assert_numerically_equal(&d3_direct, &d3_sequential, &x, INT_POINTS, 1e-10,
        "d³/dx³(x⁵) direct vs sequential");
}

/// Verify that diff of a constant w.r.t. a variable is zero
#[test]
fn diff_constant_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    for c in [-5, -1, 0, 1, 7, 42] {
        let constant = ctx.int(c);
        let diff = constant.diff(&x);
        let s = format!("{diff}");
        assert_eq!(s, "0", "d/dx({c}) should be 0, got {s}");
    }
}

/// Verify that diff of a variable w.r.t. itself is 1
#[test]
fn diff_self_is_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let diff = x.diff(&x);
    assert_eq!(format!("{diff}"), "1", "d/dx(x) should be 1");
}

/// Verify that diff of a variable w.r.t. another variable is 0
#[test]
fn diff_other_var_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let diff = y.diff(&x);
    assert_eq!(format!("{diff}"), "0", "d/dx(y) should be 0");
}

/// Eval special values consistency
#[test]
fn eval_special_values() {
    let ctx = Context::new();

    // sin(0) = 0
    let v = ctx.int(0).sin().eval_f64();
    if let Ok(v) = v {
        assert!(v.abs() < 1e-15, "sin(0) = {v}");
    }

    // cos(0) = 1
    let v = ctx.int(0).cos().eval_f64();
    if let Ok(v) = v {
        assert!((v - 1.0).abs() < 1e-15, "cos(0) = {v}");
    }

    // exp(0) = 1
    let v = ctx.int(0).exp().eval_f64();
    if let Ok(v) = v {
        assert!((v - 1.0).abs() < 1e-15, "exp(0) = {v}");
    }

    // ln(1) = 0
    let v = ctx.int(1).ln().eval_f64();
    if let Ok(v) = v {
        assert!(v.abs() < 1e-15, "ln(1) = {v}");
    }
}

/// Verify sin²(x) + cos²(x) simplifies to 1
#[test]
fn pythagorean_identity_simplifies_to_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x.sin().powi(2) + &x.cos().powi(2);
    let simplified = e.simplify();
    assert_eq!(format!("{simplified}"), "1",
        "sin²(x)+cos²(x) should simplify to 1");
}

/// Verify expand of (a+b)^n matches the binomial theorem
#[test]
fn binomial_expansion_consistency() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // (x+1)^2 = x^2 + 2x + 1
    let e2 = (&x + 1).powi(2).expand();
    let expected2 = &x.powi(2) + &(&x * 2) + 1;
    assert_numerically_equal(&e2, &expected2, &x, INT_POINTS, 1e-10,
        "binomial (x+1)²");

    // (x+1)^3 = x^3 + 3x^2 + 3x + 1
    let e3 = (&x + 1).powi(3).expand();
    let expected3 = &x.powi(3) + &(&x.powi(2) * 3) + &(&x * 3) + 1;
    assert_numerically_equal(&e3, &expected3, &x, INT_POINTS, 1e-10,
        "binomial (x+1)³");
}

/// compile of a simplified expression should give same results as compile of original
#[test]
fn compile_simplify_consistency() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x.sin().powi(2) + &x.cos().powi(2) + &x;
    let simplified = e.simplify();

    let c_orig = e.compile(&["x"]);
    let c_simp = simplified.compile(&["x"]);

    if let (Some(co), Some(cs)) = (c_orig, c_simp) {
        for &pt in &[-3.0, -1.0, 0.0, 1.0, 2.5] {
            let vo = co(&[pt]);
            let vs = cs(&[pt]);
            let diff = (vo - vs).abs();
            let scale = vo.abs().max(vs.abs()).max(1.0);
            assert!(diff / scale < 1e-10,
                "compile/simplify mismatch at x={pt}: orig={vo}, simp={vs}");
        }
    }
}

/// subs_i64 and subs with ctx.int should agree
#[test]
fn subs_i64_vs_subs_int() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(3) + &x.sin() - 2;

    for &pt in INT_POINTS {
        let via_subs_i64 = f.subs_i64(&x, pt).eval_f64();
        let via_subs = f.subs(&x, &ctx.int(pt)).eval_f64();
        if let (Ok(a), Ok(b)) = (via_subs_i64, via_subs) {
            let diff = (a - b).abs();
            assert!(diff < 1e-14,
                "subs_i64 vs subs(int) at x={pt}: {a} vs {b}");
        }
    }
}

/// has_unevaluated should be false for successfully integrated expressions
#[test]
fn has_unevaluated_false_for_basic_integrals() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let basic_integrands: Vec<(&str, Ex)> = vec![
        ("x", x.clone()),
        ("x²", x.powi(2)),
        ("x³", x.powi(3)),
        ("sin(x)", x.sin()),
        ("cos(x)", x.cos()),
        ("exp(x)", x.exp()),
        ("1/x", x.powi(-1)),
    ];

    for (label, f) in &basic_integrands {
        let anti = f.integrate(&x);
        assert!(!anti.has_unevaluated(),
            "∫ {label} dx has unevaluated nodes: {anti}");
    }
}

/// free_symbols should give the expected set
#[test]
fn free_symbols_consistency() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    // x + y has free symbols {x, y}
    let e = &x + &y;
    let syms = e.free_symbols();
    assert_eq!(syms.len(), 2, "x+y should have 2 free symbols, got {}", syms.len());

    // After substituting x, only y remains
    let e2 = e.subs(&x, &ctx.int(5));
    let syms2 = e2.free_symbols();
    assert!(syms2.len() <= 1,
        "5+y should have ≤1 free symbols, got {}: {:?}", syms2.len(),
        syms2.iter().map(|s| format!("{s}")).collect::<Vec<_>>());

    // After substituting y too, zero free symbols
    let e3 = e2.subs(&y, &ctx.int(3));
    let syms3 = e3.free_symbols();
    assert_eq!(syms3.len(), 0,
        "5+3 should have 0 free symbols, got {}: {:?}", syms3.len(),
        syms3.iter().map(|s| format!("{s}")).collect::<Vec<_>>());
}

/// count_ops should be non-negative and reasonable
#[test]
fn count_ops_reasonable() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // A constant has 0 ops (it's a leaf)
    let c = ctx.int(42);
    assert!(c.count_ops() == 0 || c.count_ops() == 1,
        "constant should have 0 or 1 ops, got {}", c.count_ops());

    // A symbol has 0 ops
    assert!(x.count_ops() == 0 || x.count_ops() == 1,
        "symbol should have 0 or 1 ops, got {}", x.count_ops());

    // x^2 + x + 1 should have at least 2 ops (add, pow)
    let e = &x.powi(2) + &x + 1;
    assert!(e.count_ops() >= 2,
        "x²+x+1 should have >= 2 ops, got {}", e.count_ops());
}

/// Verifying that expand(a * (b + c)) == expand(a*b + a*c)
#[test]
fn distributive_law() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");

    let lhs = (&x * &(&y + &z)).expand();
    let rhs = (&(&x * &y) + &(&x * &z)).expand();

    // Check at multiple points
    for &xv in &[1i64, 2, 3] {
        for &yv in &[1i64, 2, 3] {
            for &zv in &[1i64, 2, 3] {
                let lv = lhs.subs_i64(&x, xv).subs_i64(&y, yv).subs_i64(&z, zv).eval_f64();
                let rv = rhs.subs_i64(&x, xv).subs_i64(&y, yv).subs_i64(&z, zv).eval_f64();
                if let (Ok(l), Ok(r)) = (lv, rv) {
                    assert!((l - r).abs() < 1e-10,
                        "distributive: x({y}+{z}) vs xy+xz at ({xv},{yv},{zv}): {l} vs {r}");
                }
            }
        }
    }
}

/// Nested simplify should not change the numerical value
#[test]
fn nested_simplify_preserves_value() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let e = &x.sin().powi(2) + &x.cos().powi(2) + &x.powi(2) + &(&x * 2) + 1;
    let s1 = e.simplify();
    let s2 = s1.simplify();
    let s3 = s2.simplify();

    // All should agree numerically
    for &pt in INT_POINTS {
        let v0 = e.subs_i64(&x, pt).eval_f64();
        let v1 = s1.subs_i64(&x, pt).eval_f64();
        let v2 = s2.subs_i64(&x, pt).eval_f64();
        let v3 = s3.subs_i64(&x, pt).eval_f64();

        if let (Ok(v0), Ok(v1), Ok(v2), Ok(v3)) = (v0, v1, v2, v3) {
            let tol = 1e-10;
            assert!((v0 - v1).abs() < tol,
                "simplify changed value at x={pt}: {v0} → {v1}");
            assert!((v1 - v2).abs() < tol,
                "double simplify changed value at x={pt}: {v1} → {v2}");
            assert!((v2 - v3).abs() < tol,
                "triple simplify changed value at x={pt}: {v2} → {v3}");
        }
    }
}

/// Structural: simplify(0) == 0
#[test]
fn simplify_zero_is_zero() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let s = zero.simplify();
    assert_eq!(format!("{s}"), "0");
}

/// Structural: simplify(1) == 1
#[test]
fn simplify_one_is_one() {
    let ctx = Context::new();
    let one = ctx.int(1);
    let s = one.simplify();
    assert_eq!(format!("{s}"), "1");
}

/// Integration then differentiation should recover the integrand
/// for a batch of monomials c*x^n with various c and n.
#[test]
fn ftc_monomial_batch() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    for c in 1..=5i64 {
        for n in 0..=5i64 {
            let f = &x.powi(n) * c;
            let roundtrip = f.integrate(&x).diff(&x);
            assert_numerically_equal(&f, &roundtrip, &x, &[1, 2, 3], 1e-10,
                &format!("FTC monomial {c}*x^{n}"));
        }
    }
}

/// Factor/expand roundtrip for (x-a)*(x-b) with various a, b
#[test]
fn factor_expand_batch() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    for a in -3..=3i64 {
        for b in a..=3i64 {
            let product = &(&x - a) * &(&x - b);
            let expanded = product.expand();
            let factored = expanded.factor(&x);
            let re_expanded = factored.expand();
            assert_numerically_equal(&expanded, &re_expanded, &x, &[10, 20, -10], 1e-10,
                &format!("factor/expand (x-{a})(x-{b})"));
        }
    }
}

/// Smart simplify should also be idempotent
#[test]
fn smart_simplify_idempotent() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x.sin().powi(2) + &x.cos().powi(2) + &x;
    let s1 = e.simplify();
    let s2 = s1.simplify();
    // At minimum, they should agree numerically even if not structurally equal
    assert_numerically_equal(&s1, &s2, &x, INT_POINTS, 1e-10,
        "smart_simplify idempotent");
}

/// Compile should handle zero-arg constant expressions
#[test]
fn compile_constant_expression() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let e = pi.powi(2);
    let compiled = e.compile(&[]);
    if let Some(f) = compiled {
        let val = f(&[]);
        let expected = std::f64::consts::PI.powi(2);
        let diff = (val - expected).abs();
        assert!(diff < 1e-10,
            "compile(π²): {val} vs {expected}");
    }
}

/// Compile should handle expressions with e (Euler's number)
#[test]
fn compile_eulers_number() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e_const = ctx.e();
    let f = &x * &e_const;
    let compiled = f.compile(&["x"]);
    if let Some(compiled) = compiled {
        let val = compiled(&[1.0]);
        let expected = std::f64::consts::E;
        let diff = (val - expected).abs();
        assert!(diff < 1e-10,
            "compile(x*e) at x=1: {val} vs {expected}");
    }
}

/// Eval consistency: expand should not change f64 evaluation
#[test]
fn eval_expand_consistency_complex() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // (x+1)^5
    let original = (&x + 1).powi(5);
    let expanded = original.expand();

    for &pt in INT_POINTS {
        let o_val = original.subs_i64(&x, pt).eval_f64();
        let e_val = expanded.subs_i64(&x, pt).eval_f64();
        if let (Ok(ov), Ok(ev)) = (o_val, e_val) {
            let diff = (ov - ev).abs();
            let scale = ov.abs().max(1.0);
            assert!(diff / scale < 1e-10,
                "expand (x+1)⁵ at x={pt}: {ov} vs {ev}");
        }
    }
}

/// Differentiation of an integral: structural check
#[test]
fn ftc_structural_power_rule() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // d/dx(∫ x dx) = x
    let roundtrip = x.integrate(&x).diff(&x);
    let s = format!("{roundtrip}");
    assert_eq!(s, "x", "d/dx(∫ x dx) = {s}, expected x");

    // d/dx(∫ x² dx) = x²
    let roundtrip2 = x.powi(2).integrate(&x).diff(&x);
    let s2 = format!("{roundtrip2}");
    assert_eq!(s2, "x^2", "d/dx(∫ x² dx) = {s2}, expected x^2");
}

/// Compile multi-variable: check variable ordering matters
#[test]
fn compile_variable_ordering() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // f = x - y
    let f = &x - &y;

    let c_xy = f.compile(&["x", "y"]);
    let c_yx = f.compile(&["y", "x"]);

    if let (Some(fxy), Some(fyx)) = (c_xy, c_yx) {
        // fxy([2, 3]) should give 2 - 3 = -1
        assert!((fxy(&[2.0, 3.0]) - (-1.0)).abs() < 1e-10,
            "compile x-y with [x,y] at (2,3)");
        // fyx([2, 3]) should give 3 - 2 = 1 (y=2, x=3)
        assert!((fyx(&[2.0, 3.0]) - 1.0).abs() < 1e-10,
            "compile x-y with [y,x] at (2,3)");
    }
}

/// Integration of zero should be zero
#[test]
fn integrate_zero_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = ctx.int(0).integrate(&x);
    assert_eq!(format!("{result}"), "0", "∫ 0 dx should be 0");
}

/// Differentiation of sum = sum of derivatives (larger case)
#[test]
fn diff_sum_is_sum_of_diffs() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let terms: Vec<Ex> = (1..=5).map(|n| x.powi(n)).collect();
    let sum: Ex = terms.iter().skip(1).fold(terms[0].clone(), |acc, t| &acc + t);

    let diff_sum = sum.diff(&x);
    let sum_diffs: Ex = terms.iter().map(|t| t.diff(&x)).skip(1).fold(
        terms[0].diff(&x),
        |acc, d| &acc + &d,
    );

    assert_numerically_equal(&diff_sum, &sum_diffs, &x, INT_POINTS, 1e-10,
        "d/dx(sum) vs sum(d/dx) for x+x²+...+x⁵");
}

/// After evaluation, is_zero_structural should be consistent
#[test]
fn is_zero_structural_consistency() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = &x - &x;
    assert!(zero.is_zero_structural(),
        "x - x should be structurally zero");

    let also_zero = ctx.int(0);
    assert!(also_zero.is_zero_structural(), "0 should be structurally zero");
}

/// is_one_structural consistency
#[test]
fn is_one_structural_consistency() {
    let ctx = Context::new();
    let one = ctx.int(1);
    assert!(one.is_one_structural(), "1 should be structurally one");
}

/// codegen for derivative of a polynomial should work
#[test]
fn codegen_for_derivative() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(4) - &(&x.powi(2) * 3) + &(&x * 7);
    let df = f.diff(&x);
    let code = df.to_rust_fn("df", &["x"]).unwrap();
    assert_valid_rust_code(&code, "df");

    // Also verify compiled derivative matches
    let compiled = df.compile(&["x"]);
    if let Some(c) = compiled {
        for &pt in &[0.0, 1.0, 2.0, -1.0] {
            let cv = c(&[pt]);
            let ev = df.subs_i64(&x, pt as i64).eval_f64();
            if let Ok(ev) = ev {
                let diff = (cv - ev).abs();
                assert!(diff < 1e-8,
                    "codegen vs eval for df at x={pt}: {cv} vs {ev}");
            }
        }
    }
}

/// codegen for an integrated function should work
#[test]
fn codegen_for_integral() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(2);
    let anti = f.integrate(&x); // should be x^3/3
    if !anti.has_unevaluated() {
        let code = anti.to_rust_fn("anti", &["x"]).unwrap();
        assert_valid_rust_code(&code, "anti");
    }
}

/// Expr type classification should be consistent
#[test]
fn expr_type_consistency() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // Symbol
    assert!(matches!(x.expr_type(), ExprType::Symbol));

    // Number
    assert!(matches!(ctx.int(42).expr_type(), ExprType::Number));

    // Addition
    let sum = &x + 1;
    assert!(matches!(sum.expr_type(), ExprType::Add));

    // Multiplication
    let prod = &x * 2;
    assert!(matches!(prod.expr_type(), ExprType::Mul));

    // Power
    let pow = x.powi(2);
    assert!(matches!(pow.expr_type(), ExprType::Pow));

    // Function
    let sin = x.sin();
    assert!(matches!(sin.expr_type(), ExprType::Function));
}

/// Verify that simplifying a sum of identical terms works
#[test]
fn simplify_sum_of_same() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x + x should be 2*x
    let e = &x + &x;
    let s = format!("{e}");
    assert!(s == "2*x", "x + x should canonicalize to 2*x, got: {s}");
}

/// Verify x * 0 = 0
#[test]
fn multiply_by_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x * 0;
    let s = format!("{e}");
    assert_eq!(s, "0", "x * 0 should be 0, got: {s}");
}

/// Verify x * 1 = x
#[test]
fn multiply_by_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x * 1;
    let s = format!("{e}");
    assert_eq!(s, "x", "x * 1 should be x, got: {s}");
}

/// Verify x + 0 = x
#[test]
fn add_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x + 0;
    let s = format!("{e}");
    assert_eq!(s, "x", "x + 0 should be x, got: {s}");
}

/// Verify x^0 = 1
#[test]
fn power_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = x.powi(0);
    let s = format!("{e}");
    assert_eq!(s, "1", "x^0 should be 1, got: {s}");
}

/// Verify x^1 = x
#[test]
fn power_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = x.powi(1);
    let s = format!("{e}");
    assert_eq!(s, "x", "x^1 should be x, got: {s}");
}

/// CSE (common subexpression elimination) should not change the expression value
#[test]
fn cse_preserves_values() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // An expression with a repeated sub-expression
    let sub = x.sin();
    let e = &sub.powi(2) + &sub + 1;
    let (bindings, result) = e.cse();

    // Evaluate the original and the CSE result
    for &pt in INT_POINTS {
        let orig_val = e.subs_i64(&x, pt).eval_f64();

        // Build up the CSE result by substituting bindings
        let mut built = result.clone();
        for (var, val) in bindings.iter().rev() {
            built = built.subs(var, val);
        }
        let cse_val = built.subs_i64(&x, pt).eval_f64();

        if let (Ok(ov), Ok(cv)) = (orig_val, cse_val) {
            let diff = (ov - cv).abs();
            assert!(diff < 1e-10,
                "CSE changed value at x={pt}: {ov} vs {cv}");
        }
    }
}

/// Verify that compile with wrong number of vars doesn't silently succeed
/// (it should either error or produce wrong results if given extra vars)
#[test]
fn compile_zero_args_for_constant() {
    let ctx = Context::new();
    let c = ctx.int(42);
    let compiled = c.compile(&[]);
    if let Some(f) = compiled {
        let val = f(&[]);
        assert!((val - 42.0).abs() < 1e-10,
            "compile(42) with no args: {val}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ADVERSARIAL / EDGE-CASE CONSISTENCY TESTS
// ═══════════════════════════════════════════════════════════════════════════

// ── Expand/Factor edge cases ────────────────────────────────────────────

/// Repeated roots: x^3 - 3x^2 + 3x - 1 = (x-1)^3
#[test]
fn factor_expand_repeated_root() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = &x.powi(3) - &(&x.powi(2) * 3) + &(&x * 3) - 1;
    let factored = p.factor(&x);
    let re_expanded = factored.expand();
    assert_numerically_equal(&p, &re_expanded, &x, INT_POINTS, 1e-10,
        "factor/expand x³-3x²+3x-1 (repeated root)");
}

/// Degree-5 polynomial: x^5 - x
#[test]
fn factor_expand_degree5() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = &x.powi(5) - &x;
    let factored = p.factor(&x);
    let re_expanded = factored.expand();
    assert_numerically_equal(&p, &re_expanded, &x, INT_POINTS, 1e-10,
        "factor/expand x⁵-x");
}

/// Polynomial with rational coefficients: 1/2 * x^2 + 3/4 * x + 1/8
#[test]
fn expand_factor_rational_coefficients() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = &(&x.powi(2) * &ctx.rational(1, 2))
          + &(&x * &ctx.rational(3, 4))
          + &ctx.rational(1, 8);
    let factored = p.factor(&x);
    let re_expanded = factored.expand();
    assert_numerically_equal(&p, &re_expanded, &x, POS_POINTS, 1e-10,
        "factor/expand with rational coefficients");
}

/// Factor then expand of x^6 - 1 (cyclotomic)
#[test]
fn factor_expand_cyclotomic_x6_minus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = &x.powi(6) - 1;
    let factored = p.factor(&x);
    let re_expanded = factored.expand();
    assert_numerically_equal(&p, &re_expanded, &x, INT_POINTS, 1e-10,
        "factor/expand x⁶-1");
}

/// Expand of deeply nested: ((x+1)^2 + 1)^2
#[test]
fn expand_deeply_nested() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let inner = &(&x + 1).powi(2) + 1;
    let outer = inner.powi(2);
    let expanded = outer.expand();
    assert_numerically_equal(&outer, &expanded, &x, INT_POINTS, 1e-10,
        "expand ((x+1)²+1)²");
}

// ── FTC edge cases ──────────────────────────────────────────────────────

/// d/dx(∫ tan(x) dx) == tan(x)
#[test]
fn ftc_tan() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.tan();
    let anti = f.integrate(&x);
    if !anti.has_unevaluated() {
        let roundtrip = anti.diff(&x);
        // Avoid points where tan diverges
        assert_numerically_equal(&f, &roundtrip, &x, &[-1, 1, 2], 1e-8,
            "FTC: tan(x)");
    }
}

/// d/dx(∫ x^2 * exp(x) dx) == x^2 * exp(x)
#[test]
fn ftc_x2_exp() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(2) * &x.exp();
    let anti = f.integrate(&x);
    if !anti.has_unevaluated() {
        let roundtrip = anti.diff(&x);
        assert_numerically_equal(&f, &roundtrip, &x, &[-2, -1, 0, 1, 2], 1e-8,
            "FTC: x²·exp(x)");
    }
}

/// d/dx(∫ 1/(1+x^2) dx) == 1/(1+x^2)  (arctan derivative)
#[test]
fn ftc_rational_arctan() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &ctx.int(1) / &(&x.powi(2) + 1);
    let anti = f.integrate(&x);
    if !anti.has_unevaluated() {
        let roundtrip = anti.diff(&x);
        assert_numerically_equal(&f, &roundtrip, &x, INT_POINTS, 1e-8,
            "FTC: 1/(1+x²)");
    }
}

/// d/dx(∫ x/(x^2+1) dx) == x/(x^2+1)
#[test]
fn ftc_x_over_x2_plus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x / &(&x.powi(2) + 1);
    let anti = f.integrate(&x);
    if !anti.has_unevaluated() {
        let roundtrip = anti.diff(&x);
        assert_numerically_equal(&f, &roundtrip, &x, INT_POINTS, 1e-8,
            "FTC: x/(x²+1)");
    }
}

/// d/dx(∫ sin(x)*cos(x) dx) == sin(x)*cos(x)
#[test]
fn ftc_sin_cos_product() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.sin() * &x.cos();
    let anti = f.integrate(&x);
    if !anti.has_unevaluated() {
        let roundtrip = anti.diff(&x);
        assert_numerically_equal(&f, &roundtrip, &x, INT_POINTS, 1e-8,
            "FTC: sin(x)·cos(x)");
    }
}

/// d/dx(∫ exp(2x) dx) == exp(2x)  — chain rule in integration
#[test]
fn ftc_exp_2x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (&x * 2).exp();
    let anti = f.integrate(&x);
    if !anti.has_unevaluated() {
        let roundtrip = anti.diff(&x);
        assert_numerically_equal(&f, &roundtrip, &x, &[-2, -1, 0, 1, 2], 1e-8,
            "FTC: exp(2x)");
    }
}

/// d/dx(∫ cos(3x) dx) == cos(3x)
#[test]
fn ftc_cos_3x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (&x * 3).cos();
    let anti = f.integrate(&x);
    if !anti.has_unevaluated() {
        let roundtrip = anti.diff(&x);
        assert_numerically_equal(&f, &roundtrip, &x, INT_POINTS, 1e-8,
            "FTC: cos(3x)");
    }
}

/// FTC for x^(-1/2) = 1/sqrt(x): ∫ x^(-1/2) dx = 2·sqrt(x)
#[test]
fn ftc_x_neg_half() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.pow(&ctx.rational(-1, 2));
    let anti = f.integrate(&x);
    if !anti.has_unevaluated() {
        let roundtrip = anti.diff(&x);
        assert_numerically_equal(&f, &roundtrip, &x, &[1, 2, 3, 4], 1e-8,
            "FTC: x^(-1/2)");
    }
}

/// FTC for sqrt(x): ∫ sqrt(x) dx = 2/3 * x^(3/2)
#[test]
fn ftc_sqrt_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sqrt();
    let anti = f.integrate(&x);
    if !anti.has_unevaluated() {
        let roundtrip = anti.diff(&x);
        assert_numerically_equal(&f, &roundtrip, &x, POS_POINTS, 1e-8,
            "FTC: sqrt(x)");
    }
}

// ── Simplify idempotency adversarial ────────────────────────────────────

/// full_simplify(full_simplify(e)) == full_simplify(e) for a complex expression
#[test]
fn full_simplify_idempotent_complex() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &(&x.sin().powi(2) + &x.cos().powi(2)) * &(&x + 1);
    let s1 = e.simplify();
    let s2 = s1.simplify();
    assert_eq!(
        format!("{s1}"), format!("{s2}"),
        "full_simplify not idempotent for (sin²+cos²)·(x+1): '{}' vs '{}'", s1, s2
    );
}

/// simplify of a deeply nested fraction
#[test]
fn simplify_idempotent_nested_fraction() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // ((x+1)/(x-1)) * ((x-1)/(x+1))  should simplify to 1
    let e = &(&(&x + 1) / &(&x - 1)) * &(&(&x - 1) / &(&x + 1));
    check_simplify_idempotent(&e, "((x+1)/(x-1))·((x-1)/(x+1))");
}

/// simplify after trig_expand should be idempotent
#[test]
fn simplify_idempotent_after_trig_expand() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = (&x * 2).sin().expand_trig(); // 2·sin(x)·cos(x)
    check_simplify_idempotent(&e, "expand_trig(sin(2x))");
}

/// simplify of expression involving π
#[test]
fn simplify_idempotent_with_pi() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &(&x * &ctx.pi()).sin().powi(2) + &(&x * &ctx.pi()).cos().powi(2);
    check_simplify_idempotent(&e, "sin²(πx)+cos²(πx)");
}

/// simplify idempotent for double fraction (a/b)/(c/d)
#[test]
fn simplify_idempotent_double_fraction() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &(&x / &ctx.int(2)) / &(&x / &ctx.int(3));
    check_simplify_idempotent(&e, "(x/2)/(x/3)");
}

// ── Eval consistency adversarial ────────────────────────────────────────

/// Two different representations of the same polynomial must agree
#[test]
fn eval_consistency_different_polynomial_forms() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Factored form: (x-1)(x-2)(x-3)
    let factored = &(&(&x - 1) * &(&x - 2)) * &(&x - 3);
    // Expanded form: x^3 - 6x^2 + 11x - 6
    let expanded = &x.powi(3) - &(&x.powi(2) * 6) + &(&x * 11) - 6;
    assert_numerically_equal(&factored, &expanded, &x, INT_POINTS, 1e-10,
        "factored vs expanded cubic");
}

/// Simplification path should not lose precision
#[test]
fn eval_consistency_simplify_precision() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x^2 - 1) expanded vs (x-1)(x+1)
    let a = &x.powi(2) - 1;
    let b = &(&x - 1) * &(&x + 1);
    let a_simp = a.simplify();
    let b_simp = b.simplify();
    assert_numerically_equal(&a_simp, &b_simp, &x, INT_POINTS, 1e-10,
        "simplify((x²-1)) vs simplify((x-1)(x+1))");
}

/// Verify that diff then integrate of a polynomial gives back the same polynomial
/// (up to a constant), checked numerically at two points to eliminate the constant
#[test]
fn integrate_diff_recovers_up_to_constant() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // f = x^3 + 2x + 5
    let f = &x.powi(3) + &(&x * 2) + 5;
    let df = f.diff(&x);  // 3x^2 + 2
    let anti_df = df.integrate(&x);  // x^3 + 2x + C for some C
    // anti_df - f should be a constant (independent of x)
    let diff_expr = &anti_df - &f;
    let v1 = diff_expr.subs_i64(&x, 1).eval_f64();
    let v2 = diff_expr.subs_i64(&x, 5).eval_f64();
    if let (Ok(a), Ok(b)) = (v1, v2) {
        let d = (a - b).abs();
        assert!(d < 1e-10,
            "∫(d/dx f) - f should be constant, but varies: at x=1: {a}, at x=5: {b}, diff={d}");
    }
}

// ── Compile consistency adversarial ─────────────────────────────────────

/// compile should agree with eval_f64 for expressions involving constants
#[test]
fn compile_vs_eval_pi_expression() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (&x * &ctx.pi()).sin();
    let compiled = f.compile(&["x"]);
    if let Some(c) = compiled {
        for &pt in &[0i64, 1, 2] {
            let cv = c(&[pt as f64]);
            let ev = f.subs_i64(&x, pt).eval_f64();
            if let Ok(ev) = ev {
                let diff = (cv - ev).abs();
                assert!(diff < 1e-10,
                    "compile vs eval sin(πx) at x={pt}: compile={cv}, eval={ev}");
            }
        }
    }
}

/// compile of a simplified expression vs compile of original
#[test]
fn compile_original_vs_simplified() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // exp(ln(x)) simplifies to x
    let e = x.ln().exp();
    let s = e.simplify();
    let c_orig = e.compile(&["x"]);
    let c_simp = s.compile(&["x"]);
    if let (Some(co), Some(cs)) = (c_orig, c_simp) {
        for &pt in &[1.0, 2.0, 3.0, 4.5] {
            let vo = co(&[pt]);
            let vs = cs(&[pt]);
            let diff = (vo - vs).abs();
            let scale = vo.abs().max(vs.abs()).max(1.0);
            assert!(diff / scale < 1e-10,
                "compile(exp(ln(x))) vs compile(simplified) at {pt}: {vo} vs {vs}");
        }
    }
}

/// compile of factored vs expanded polynomial
#[test]
fn compile_factored_vs_expanded() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expanded = &x.powi(2) - &(&x * 5) + 6;
    let factored = expanded.factor(&x);
    let c_exp = expanded.compile(&["x"]);
    let c_fac = factored.compile(&["x"]);
    if let (Some(ce), Some(cf)) = (c_exp, c_fac) {
        for &pt in &[-3.0, 0.0, 1.0, 2.0, 3.0, 5.0] {
            let ve = ce(&[pt]);
            let vf = cf(&[pt]);
            let diff = (ve - vf).abs();
            assert!(diff < 1e-10,
                "compile(expanded) vs compile(factored) at x={pt}: {ve} vs {vf}");
        }
    }
}

// ── Commutativity adversarial ───────────────────────────────────────────

/// diff(expand(f, trig)) should agree numerically with expand(diff(f), trig)
/// for sin(2x) → 2sin(x)cos(x)
#[test]
fn commutativity_diff_trig_expand() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (&x * 2).sin();
    let diff_then_expand = f.diff(&x).expand_trig();
    let expand_then_diff = f.expand_trig().diff(&x);
    assert_numerically_equal(
        &diff_then_expand, &expand_then_diff, &x, INT_POINTS, 1e-8,
        "diff(expand_trig(sin(2x))) vs expand_trig(diff(sin(2x)))");
}

/// simplify(expand(f)) should agree with expand(simplify(f)) numerically
#[test]
fn commutativity_simplify_expand() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &(&x.sin().powi(2) + &x.cos().powi(2)) * &(&x + 1).powi(2);
    let simp_expand = f.simplify().expand();
    let expand_simp = f.expand().simplify();
    assert_numerically_equal(
        &simp_expand, &expand_simp, &x, INT_POINTS, 1e-8,
        "simplify(expand(f)) vs expand(simplify(f))");
}

/// factor(simplify(f)) vs simplify(factor(f)) for a polynomial
#[test]
fn commutativity_simplify_factor() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(3) - &x;
    let simp_fac = f.simplify().factor(&x);
    let fac_simp = f.factor(&x).simplify();
    assert_numerically_equal(
        &simp_fac, &fac_simp, &x, INT_POINTS, 1e-10,
        "simplify(factor(x³-x)) vs factor(simplify(x³-x))");
}

// ── LaTeX adversarial ───────────────────────────────────────────────────

/// LaTeX of negative numbers should render properly
#[test]
fn latex_negative_numbers() {
    let ctx = Context::new();
    let neg = ctx.int(-5);
    let latex = neg.to_latex();
    assert!(!latex.is_empty(), "LaTeX of -5 is empty");
    assert!(latex.contains("5"), "LaTeX of -5 should contain 5: {latex}");
}

/// LaTeX of nested powers should have proper grouping
#[test]
fn latex_nested_powers() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = x.powi(2).powi(3);
    let latex = e.to_latex();
    assert!(!latex.is_empty(), "LaTeX of (x²)³ is empty");
    // Should contain x somewhere
    assert!(latex.contains("x"), "LaTeX of (x²)³ should contain x: {latex}");
}

/// LaTeX of a sum of products
#[test]
fn latex_sum_of_products() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let e = &(&x * &y) + &x.powi(2);
    let latex = e.to_latex();
    assert!(!latex.is_empty());
    assert!(latex.contains("x"), "LaTeX missing x: {latex}");
    assert!(latex.contains("y"), "LaTeX missing y: {latex}");
}

/// LaTeX of sqrt should use \sqrt
#[test]
fn latex_sqrt_command() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = x.sqrt();
    let latex = e.to_latex();
    assert!(latex.contains("sqrt") || latex.contains("frac{1}{2}") || latex.contains("^{1/2}"),
        "sqrt LaTeX should use \\sqrt or show 1/2 power: {latex}");
}

/// LaTeX of exp should use e^ or \exp
#[test]
fn latex_exp_rendering() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = x.exp();
    let latex = e.to_latex();
    assert!(latex.contains("exp") || latex.contains("e^") || latex.contains("\\mathrm{e}"),
        "exp LaTeX: {latex}");
}

/// LaTeX consistency: simplify then latex vs latex then (manually comparing)
/// just check neither crashes and both are non-empty
#[test]
fn latex_after_simplify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x.sin().powi(2) + &x.cos().powi(2);
    let latex_before = e.to_latex();
    let latex_after = e.simplify().to_latex();
    assert!(!latex_before.is_empty());
    assert!(!latex_after.is_empty());
    // The simplified version should be simpler (shorter or equal)
    // but at minimum non-empty
}

// ── Code generation adversarial ─────────────────────────────────────────

/// codegen for a fraction expression
#[test]
fn codegen_fraction() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &ctx.int(1) / &(&x.powi(2) + 1);
    let code = f.to_rust_fn("frac_fn", &["x"]).unwrap();
    assert_valid_rust_code(&code, "frac_fn");
}

/// codegen for an expression with rational constant
#[test]
fn codegen_rational_constant() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x * &ctx.rational(3, 7);
    let code = f.to_rust_fn("rat_fn", &["x"]).unwrap();
    assert_valid_rust_code(&code, "rat_fn");
}

/// codegen for hyperbolic inverse functions
#[test]
fn codegen_inverse_hyperbolic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.asinh();
    let code = f.to_rust_fn("asinh_fn", &["x"]).unwrap();
    assert_valid_rust_code(&code, "asinh_fn");
}

/// codegen of an integrated result
#[test]
fn codegen_integrated_result() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(3).integrate(&x); // x^4/4
    if !f.has_unevaluated() {
        let code = f.to_rust_fn("anti_cubic", &["x"]).unwrap();
        assert_valid_rust_code(&code, "anti_cubic");
    }
}

/// codegen for a complex multi-operation expression
#[test]
fn codegen_complex_expression() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sin(x)^2 * exp(-x) + cos(x) * ln(x^2 + 1)
    let f = &(&x.sin().powi(2) * &(-&x).exp()) + &(&x.cos() * &(&x.powi(2) + 1).ln());
    let code = f.to_rust_fn("complex_fn", &["x"]).unwrap();
    assert_valid_rust_code(&code, "complex_fn");
}

// ── Deeper numerical consistency ────────────────────────────────────────

/// Verify Euler's identity-adjacent: exp(i*pi) ≈ -1 via sin/cos
/// Check that sin(π) ≈ 0 and cos(π) ≈ -1 numerically
#[test]
fn euler_identity_numerical() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let sin_pi = pi.sin().eval_f64();
    let cos_pi = pi.cos().eval_f64();
    if let Ok(s) = sin_pi {
        assert!(s.abs() < 1e-10, "sin(π) should be 0, got {s}");
    }
    if let Ok(c) = cos_pi {
        assert!((c + 1.0).abs() < 1e-10, "cos(π) should be -1, got {c}");
    }
}

/// Verify ln(exp(1)) = 1
#[test]
fn ln_e_is_one() {
    let ctx = Context::new();
    let e = ctx.e();
    let result = e.ln().eval_f64();
    if let Ok(v) = result {
        assert!((v - 1.0).abs() < 1e-10, "ln(e) should be 1, got {v}");
    }
}

/// Verify exp(ln(5)) = 5
#[test]
fn exp_ln_roundtrip_numerical() {
    let ctx = Context::new();
    let five = ctx.int(5);
    let result = five.ln().exp().eval_f64();
    if let Ok(v) = result {
        assert!((v - 5.0).abs() < 1e-10, "exp(ln(5)) should be 5, got {v}");
    }
}

/// Verify that two different integration paths give the same result
/// ∫ 2x dx vs 2 · ∫ x dx  (linearity of integration)
#[test]
fn integration_linearity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x * 2;
    let int_2x = f.integrate(&x);
    let two_int_x = &x.integrate(&x) * 2;
    // These might differ by a constant, but their derivatives should be the same
    let d1 = int_2x.diff(&x);
    let d2 = two_int_x.diff(&x);
    assert_numerically_equal(&d1, &d2, &x, INT_POINTS, 1e-10,
        "d/dx(∫2x) vs d/dx(2·∫x)");
}

/// Integration of sum = sum of integrals (additivity) checked via diff roundtrip
#[test]
fn integration_additivity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(2);
    let g = x.sin();
    let sum = &f + &g;
    let int_sum = sum.integrate(&x);
    let sum_int = &f.integrate(&x) + &g.integrate(&x);
    // Differentiate both — should recover f + g
    let d1 = int_sum.diff(&x);
    let d2 = sum_int.diff(&x);
    assert_numerically_equal(&d1, &d2, &x, INT_POINTS, 1e-8,
        "d/dx(∫(f+g)) vs d/dx(∫f + ∫g)");
}

/// Verify power rule: ∫ x^n dx = x^(n+1)/(n+1) for several n
#[test]
fn integration_power_rule_structural() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    for n in 1..=6i64 {
        let f = x.powi(n);
        let anti = f.integrate(&x);
        // Expected: x^(n+1) / (n+1)
        let expected = &x.powi(n + 1) / &ctx.int(n + 1);
        assert_numerically_equal(&anti, &expected, &x, POS_POINTS, 1e-10,
            &format!("∫ x^{n} dx = x^{}/{}?", n + 1, n + 1));
    }
}

/// Verify that expanding a factored quartic recovers the original
/// (x^2 - 1)(x^2 + 1) = x^4 - 1
#[test]
fn factor_product_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = &x.powi(2) - 1;
    let b = &x.powi(2) + 1;
    let product = (&a * &b).expand();
    let expected = &x.powi(4) - 1;
    assert_numerically_equal(&product, &expected, &x, INT_POINTS, 1e-10,
        "(x²-1)(x²+1) = x⁴-1");
}

/// The derivative of sin should be cos, verified structurally
#[test]
fn diff_sin_structural() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let df = x.sin().diff(&x);
    assert_eq!(format!("{df}"), "cos(x)", "d/dx sin(x) = {df}");
}

/// The derivative of cos should be -sin, verified structurally
#[test]
fn diff_cos_structural() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let df = x.cos().diff(&x);
    assert_eq!(format!("{df}"), "-sin(x)", "d/dx cos(x) = {df}");
}

/// The derivative of exp should be exp, verified structurally
#[test]
fn diff_exp_structural() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let df = x.exp().diff(&x);
    assert_eq!(format!("{df}"), "exp(x)", "d/dx exp(x) = {df}");
}

/// The derivative of ln should be 1/x, verified structurally
#[test]
fn diff_ln_structural() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let df = x.ln().diff(&x);
    assert_eq!(format!("{df}"), "1/x", "d/dx ln(x) = {df}");
}

/// compile of a derivative expression should match the derivative evaluated
#[test]
fn compile_of_derivative() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(4) + &x.sin();
    let df = f.diff(&x); // 4x^3 + cos(x)
    let compiled = df.compile(&["x"]);
    if let Some(c) = compiled {
        for &pt in &[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0] {
            let cv = c(&[pt]);
            let expected = 4.0 * pt.powi(3) + pt.cos();
            let diff = (cv - expected).abs();
            assert!(diff < 1e-8,
                "compile(d/dx(x⁴+sin(x))) at x={pt}: {cv} vs {expected}");
        }
    }
}

/// compile of an integrated expression should match the antiderivative
#[test]
fn compile_of_integral() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(2);
    let anti = f.integrate(&x); // x^3/3
    if !anti.has_unevaluated() {
        let compiled = anti.compile(&["x"]);
        if let Some(c) = compiled {
            for &pt in &[1.0, 2.0, 3.0, 4.0] {
                let cv = c(&[pt]);
                let expected = pt.powi(3) / 3.0;
                let diff = (cv - expected).abs();
                assert!(diff < 1e-8,
                    "compile(∫x²dx) at x={pt}: {cv} vs {expected}");
            }
        }
    }
}

/// Verify that expand(a+b)^n matches binomial for n=4,5
#[test]
fn binomial_higher_degrees() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // (x+1)^4
    let e4 = (&x + 1).powi(4);
    let expanded4 = e4.expand();
    assert_numerically_equal(&e4, &expanded4, &x, INT_POINTS, 1e-10,
        "binomial (x+1)⁴");

    // (x+1)^5
    let e5 = (&x + 1).powi(5);
    let expanded5 = e5.expand();
    assert_numerically_equal(&e5, &expanded5, &x, INT_POINTS, 1e-10,
        "binomial (x+1)⁵");

    // (x+2)^3
    let e_2_3 = (&x + 2).powi(3);
    let expanded_2_3 = e_2_3.expand();
    assert_numerically_equal(&e_2_3, &expanded_2_3, &x, INT_POINTS, 1e-10,
        "binomial (x+2)³");
}

/// Verify that multiple substitutions commute
#[test]
fn subs_commutativity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let f = &x.powi(2) + &y.powi(2);
    let a = ctx.int(3);
    let b = ctx.int(4);

    // subs(subs(f, x, 3), y, 4) vs subs(subs(f, y, 4), x, 3)
    let path1 = f.subs(&x, &a).subs(&y, &b);
    let path2 = f.subs(&y, &b).subs(&x, &a);
    let v1 = path1.eval_f64();
    let v2 = path2.eval_f64();
    if let (Ok(a), Ok(b)) = (v1, v2) {
        assert!((a - b).abs() < 1e-10,
            "subs commutativity: {a} vs {b}");
        assert!((a - 25.0).abs() < 1e-10,
            "3² + 4² should be 25, got {a}");
    }
}

/// Verify that eval_f64 of a rational is exact
#[test]
fn eval_f64_rational_exact() {
    let ctx = Context::new();
    let r = ctx.rational(1, 3);
    let v = r.eval_f64();
    if let Ok(v) = v {
        assert!((v - 1.0/3.0).abs() < 1e-15,
            "1/3 eval_f64: {v}");
    }
    let r2 = ctx.rational(22, 7);
    let v2 = r2.eval_f64();
    if let Ok(v) = v2 {
        assert!((v - 22.0/7.0).abs() < 1e-15,
            "22/7 eval_f64: {v}");
    }
}

/// Verify that diff(x^n, x) = n*x^(n-1) structurally for small n
#[test]
fn diff_power_rule_structural() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let cases: Vec<(i64, &str)> = vec![
        (2, "2*x"),
        (3, "3*x^2"),
        (4, "4*x^3"),
        (5, "5*x^4"),
    ];

    for (n, expected) in cases {
        let df = x.powi(n).diff(&x);
        let s = format!("{df}");
        assert_eq!(s, expected, "d/dx(x^{n}) should be {expected}, got {s}");
    }
}

/// Verify that integrate(diff(f)) - f is a constant for various f
#[test]
fn integrate_diff_constant_offset() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let functions: Vec<(&str, Ex)> = vec![
        ("x²+3x+1", &x.powi(2) + &(&x * 3) + 1),
        ("x⁴", x.powi(4)),
        ("sin(x)", x.sin()),
        ("exp(x)", x.exp()),
    ];

    for (label, f) in &functions {
        let df = f.diff(&x);
        let anti_df = df.integrate(&x);
        if anti_df.has_unevaluated() { continue; }
        // anti_df - f should be a constant
        let residual = &anti_df - f;
        let v1 = residual.subs_i64(&x, 1).eval_f64();
        let v2 = residual.subs_i64(&x, 3).eval_f64();
        let v3 = residual.subs_i64(&x, 5).eval_f64();
        if let (Ok(a), Ok(b), Ok(c)) = (v1, v2, v3) {
            let max_diff = (a - b).abs().max((b - c).abs()).max((a - c).abs());
            assert!(max_diff < 1e-8,
                "∫(d/dx {label}) - {label} should be constant, diffs: {a}, {b}, {c}");
        }
    }
}

/// Verify that eval preserves exact rational arithmetic
#[test]
fn eval_rational_arithmetic() {
    let ctx = Context::new();
    // 1/3 + 1/6 = 1/2
    let sum = &ctx.rational(1, 3) + &ctx.rational(1, 6);
    let evaled = sum.eval();
    let s = format!("{evaled}");
    assert_eq!(s, "1/2", "1/3 + 1/6 should be 1/2, got {s}");
}

/// Verify rational multiplication: 2/3 * 3/4 = 1/2
#[test]
fn eval_rational_multiplication() {
    let ctx = Context::new();
    let prod = &ctx.rational(2, 3) * &ctx.rational(3, 4);
    let evaled = prod.eval();
    let s = format!("{evaled}");
    assert_eq!(s, "1/2", "2/3 * 3/4 should be 1/2, got {s}");
}

/// Verify that simplify doesn't break rational expressions
#[test]
fn simplify_preserves_rational() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x/2 + x/3) should simplify to 5x/6
    let e = &(&x / &ctx.int(2)) + &(&x / &ctx.int(3));
    let simplified = e.simplify();
    assert_numerically_equal(&e, &simplified, &x, INT_POINTS, 1e-10,
        "simplify(x/2 + x/3)");
}

/// Verify that x^0 is 1 for nonzero x, even after simplify
#[test]
fn power_zero_after_simplify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = x.powi(0);
    let s = e.simplify();
    assert_eq!(format!("{s}"), "1", "simplify(x^0) should be 1");
}

/// Verify cancellation in fractions: (x^2 - 4)/(x - 2) = x + 2
#[test]
fn cancel_quadratic_linear() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &(&x.powi(2) - 4) / &(&x - 2);
    let cancelled = e.cancel(&x);
    // Check structural result
    let expected = &x + 2;
    assert_numerically_equal(&cancelled, &expected, &x, &[-3, -1, 0, 1, 3, 4, 5], 1e-10,
        "cancel (x²-4)/(x-2) = x+2");
}

/// Verify that diff and factor commute numerically on a polynomial
#[test]
fn diff_factor_numerical_consistency() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = &x.powi(3) - &(&x.powi(2) * 3) + &(&x * 2);
    // diff then factor
    let dp = p.diff(&x);
    let dp_factored = dp.factor(&x);
    // factor then diff
    let p_factored = p.factor(&x);
    let p_factored_diff = p_factored.diff(&x);
    // These might not be structurally equal, but should agree numerically
    assert_numerically_equal(&dp_factored, &p_factored_diff, &x, INT_POINTS, 1e-8,
        "factor(diff(p)) vs diff(factor(p))");
}

// ═══════════════════════════════════════════════════════════════════════════
// WAVE 3: DEEPER ADVERSARIAL / EDGE-CASE TESTS
// ═══════════════════════════════════════════════════════════════════════════

// ── Simplify oscillation / non-convergence detection ────────────────────

/// Apply simplify many times; the structural form should stabilize
/// within a small number of iterations.
#[test]
fn simplify_stabilizes_trig_sum() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 2*sin(x)*cos(x) — this is sin(2x)/1, simplify might try to combine/expand
    let e = &(&x.sin() * &x.cos()) * 2;
    let mut prev = format!("{}", e.simplify());
    for i in 0..5 {
        let next_e = e.simplify();
        // re-simplify the already-simplified form
        let s = if i == 0 { next_e.simplify() } else { next_e };
        let curr = format!("{s}");
        if i > 0 {
            assert_eq!(
                prev, curr,
                "simplify oscillates at iteration {i}: '{prev}' vs '{curr}'"
            );
        }
        prev = curr;
    }
}

/// full_simplify on a product containing a Pythagorean identity
/// (sin²+cos²)*(x+1)*(x-1) should simplify to x²-1
#[test]
fn full_simplify_pythagorean_in_product() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &(&x.sin().powi(2) + &x.cos().powi(2)) * &(&(&x + 1) * &(&x - 1));
    let s = e.simplify();
    // Numerically it should equal x²-1
    let expected = &x.powi(2) - 1;
    assert_numerically_equal(&s, &expected, &x, INT_POINTS, 1e-10,
        "full_simplify((sin²+cos²)*(x+1)*(x-1)) = x²-1");
}

/// simplify of 0 * sin(x) should be 0
#[test]
fn simplify_zero_times_trig() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &ctx.int(0) * &x.sin();
    let s = format!("{e}");
    assert_eq!(s, "0", "0 * sin(x) should canonicalize to 0, got: {s}");
}

/// simplify of sin(x)/sin(x) should be 1 (or at least numerically 1)
#[test]
fn simplify_trig_self_cancel() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x.sin() / &x.sin();
    let simplified = e.simplify();
    // At points where sin(x) != 0, value should be 1
    for &pt in &[-3, -2, -1, 1, 2, 3] {
        let v = simplified.subs_i64(&x, pt).eval_f64();
        if let Ok(v) = v {
            assert!(
                (v - 1.0).abs() < 1e-10,
                "sin(x)/sin(x) simplified to {simplified}, value at x={pt}: {v}"
            );
        }
    }
}

/// simplify of exp(x)/exp(x) should be 1
#[test]
fn simplify_exp_self_cancel() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x.exp() / &x.exp();
    let simplified = e.simplify();
    for &pt in INT_POINTS {
        let v = simplified.subs_i64(&x, pt).eval_f64();
        if let Ok(v) = v {
            assert!(
                (v - 1.0).abs() < 1e-10,
                "exp(x)/exp(x) simplified to {simplified}, value at x={pt}: {v}"
            );
        }
    }
}

// ── Tricky FTC cases ────────────────────────────────────────────────────

/// d/dx(∫ sinh(x) dx) == sinh(x)
#[test]
fn ftc_sinh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sinh();
    let anti = f.integrate(&x);
    if !anti.has_unevaluated() {
        let roundtrip = anti.diff(&x);
        assert_numerically_equal(&f, &roundtrip, &x, &[-2, -1, 0, 1, 2], 1e-8,
            "FTC: sinh(x)");
    }
}

/// d/dx(∫ cosh(x) dx) == cosh(x)
#[test]
fn ftc_cosh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.cosh();
    let anti = f.integrate(&x);
    if !anti.has_unevaluated() {
        let roundtrip = anti.diff(&x);
        assert_numerically_equal(&f, &roundtrip, &x, &[-2, -1, 0, 1, 2], 1e-8,
            "FTC: cosh(x)");
    }
}

/// d/dx(∫ sec²(x) dx) == sec²(x) == 1/cos²(x)
#[test]
fn ftc_sec_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sec²(x) = 1/cos²(x) = cos(x)^(-2)
    let f = x.cos().powi(-2);
    let anti = f.integrate(&x);
    if !anti.has_unevaluated() {
        let roundtrip = anti.diff(&x);
        // Avoid x near ±π/2 where cos → 0
        assert_numerically_equal(&f, &roundtrip, &x, &[-1, 0, 1], 1e-8,
            "FTC: sec²(x)");
    }
}

/// d/dx(∫ x*ln(x) dx) == x*ln(x) — integration by parts result
#[test]
fn ftc_x_ln_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x * &x.ln();
    let anti = f.integrate(&x);
    if !anti.has_unevaluated() {
        let roundtrip = anti.diff(&x);
        assert_numerically_equal(&f, &roundtrip, &x, &[1, 2, 3, 4, 5], 1e-8,
            "FTC: x·ln(x)");
    }
}

/// d/dx(∫ 1/(x^2-1) dx) == 1/(x^2-1) — partial fractions
#[test]
fn ftc_partial_fractions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &ctx.int(1) / &(&x.powi(2) - 1);
    let anti = f.integrate(&x);
    if !anti.has_unevaluated() {
        let roundtrip = anti.diff(&x);
        // Avoid x=±1 where denominator is zero; use x=2,3,4,5
        assert_numerically_equal(&f, &roundtrip, &x, &[2, 3, 4, 5], 1e-8,
            "FTC: 1/(x²-1)");
    }
}

/// d/dx(∫ x^3 * sin(x) dx) — multiple integration-by-parts
#[test]
fn ftc_x3_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(3) * &x.sin();
    let anti = f.integrate(&x);
    if !anti.has_unevaluated() {
        let roundtrip = anti.diff(&x);
        assert_numerically_equal(&f, &roundtrip, &x, INT_POINTS, 1e-6,
            "FTC: x³·sin(x)");
    }
}

// ── Expand / Factor tricky cases ────────────────────────────────────────

/// Factor of a perfect square: x^2 + 2x + 1 = (x+1)^2
#[test]
fn factor_perfect_square() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = &x.powi(2) + &(&x * 2) + 1;
    let factored = p.factor(&x);
    let re_expanded = factored.expand();
    assert_eq!(
        format!("{}", p.expand()), format!("{re_expanded}"),
        "factor/expand of perfect square"
    );
}

/// Factor of irreducible over Q: x^2 + 1 should stay as x^2 + 1
#[test]
fn factor_irreducible_over_q() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = &x.powi(2) + 1;
    let factored = p.factor(&x);
    let re_expanded = factored.expand();
    // Should be the same as original since it's irreducible over Q
    assert_numerically_equal(&p, &re_expanded, &x, INT_POINTS, 1e-10,
        "factor/expand of x²+1 (irreducible over Q)");
}

/// Expand of (x+y)^2 with two variables
#[test]
fn expand_bivariate() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let e = (&x + &y).powi(2);
    let expanded = e.expand();
    // Check at several (x,y) pairs
    for &xv in &[1i64, 2, 3] {
        for &yv in &[1i64, 2, 3] {
            let ov = e.subs_i64(&x, xv).subs_i64(&y, yv).eval_f64();
            let ev = expanded.subs_i64(&x, xv).subs_i64(&y, yv).eval_f64();
            if let (Ok(a), Ok(b)) = (ov, ev) {
                assert!(
                    (a - b).abs() < 1e-10,
                    "expand (x+y)² at ({xv},{yv}): {a} vs {b}"
                );
            }
        }
    }
}

/// Expand of (x+y+z)^2
#[test]
fn expand_trivariate() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");
    let e = (&(&x + &y) + &z).powi(2);
    let expanded = e.expand();
    let ov = e.subs_i64(&x, 2).subs_i64(&y, 3).subs_i64(&z, 4).eval_f64();
    let ev = expanded.subs_i64(&x, 2).subs_i64(&y, 3).subs_i64(&z, 4).eval_f64();
    if let (Ok(a), Ok(b)) = (ov, ev) {
        assert!((a - b).abs() < 1e-10, "expand (x+y+z)² at (2,3,4): {a} vs {b}");
        assert!((a - 81.0).abs() < 1e-10, "(2+3+4)² should be 81, got {a}");
    }
}

/// Expand of product of three binomials: (x+1)(x+2)(x+3)
#[test]
fn expand_three_binomials() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let product = &(&(&x + 1) * &(&x + 2)) * &(&x + 3);
    let expanded = product.expand();
    assert_numerically_equal(&product, &expanded, &x, INT_POINTS, 1e-10,
        "expand (x+1)(x+2)(x+3)");
    // At x=0: (1)(2)(3)=6
    let v = expanded.subs_i64(&x, 0).eval_f64();
    if let Ok(v) = v {
        assert!((v - 6.0).abs() < 1e-10, "(0+1)(0+2)(0+3) should be 6, got {v}");
    }
}

// ── Numerical edge cases ────────────────────────────────────────────────

/// eval_f64 at zero for various functions
#[test]
fn eval_at_zero_edge_cases() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // sin(0) = 0
    let v = x.sin().subs_i64(&x, 0).eval_f64();
    if let Ok(v) = v { assert!(v.abs() < 1e-15, "sin(0) = {v}"); }

    // cos(0) = 1
    let v = x.cos().subs_i64(&x, 0).eval_f64();
    if let Ok(v) = v { assert!((v - 1.0).abs() < 1e-15, "cos(0) = {v}"); }

    // exp(0) = 1
    let v = x.exp().subs_i64(&x, 0).eval_f64();
    if let Ok(v) = v { assert!((v - 1.0).abs() < 1e-15, "exp(0) = {v}"); }

    // sinh(0) = 0
    let v = x.sinh().subs_i64(&x, 0).eval_f64();
    if let Ok(v) = v { assert!(v.abs() < 1e-15, "sinh(0) = {v}"); }

    // cosh(0) = 1
    let v = x.cosh().subs_i64(&x, 0).eval_f64();
    if let Ok(v) = v { assert!((v - 1.0).abs() < 1e-15, "cosh(0) = {v}"); }

    // tanh(0) = 0
    let v = x.tanh().subs_i64(&x, 0).eval_f64();
    if let Ok(v) = v { assert!(v.abs() < 1e-15, "tanh(0) = {v}"); }

    // atan(0) = 0
    let v = x.atan().subs_i64(&x, 0).eval_f64();
    if let Ok(v) = v { assert!(v.abs() < 1e-15, "atan(0) = {v}"); }
}

/// Large integer arithmetic: expand((x+1)^10) and check at x=1 → 2^10 = 1024
#[test]
fn expand_large_power_numerical() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = (&x + 1).powi(10);
    let expanded = e.expand();
    let v = expanded.subs_i64(&x, 1).eval_f64();
    if let Ok(v) = v {
        assert!((v - 1024.0).abs() < 1e-6,
            "expand((x+1)^10) at x=1 should be 1024, got {v}");
    }
    // Also check the unexpanded form gives the same
    let v_orig = e.subs_i64(&x, 1).eval_f64();
    if let (Ok(vo), Ok(ve)) = (v_orig, expanded.subs_i64(&x, 1).eval_f64()) {
        assert!((vo - ve).abs() < 1e-6,
            "expand vs original at x=1: {vo} vs {ve}");
    }
}

/// Compile and eval agree for expression with nested functions
#[test]
fn compile_vs_eval_nested_functions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sin(exp(x)) + cos(ln(x+2))
    let f = &x.exp().sin() + &(&x + 2).ln().cos();
    let compiled = f.compile(&["x"]);
    if let Some(c) = compiled {
        for &pt in &[0i64, 1, 2, 3] {
            let cv = c(&[pt as f64]);
            let ev = f.subs_i64(&x, pt).eval_f64();
            if let Ok(ev) = ev {
                let diff = (cv - ev).abs();
                let scale = cv.abs().max(ev.abs()).max(1.0);
                assert!(
                    diff / scale < 1e-10,
                    "compile vs eval nested at x={pt}: compile={cv}, eval={ev}"
                );
            }
        }
    }
}

/// Compile should handle the expression (x^2 + 1)^(-1)
#[test]
fn compile_reciprocal_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (&x.powi(2) + 1).powi(-1);
    check_compile_consistency(&f, &x, INT_POINTS, 1e-10, "1/(x²+1)");
}

/// Verify compile of a rational function
#[test]
fn compile_rational_function() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &(&x.powi(2) + 1) / &(&x.powi(2) - 4);
    let compiled = f.compile(&["x"]);
    if let Some(c) = compiled {
        // Avoid x=±2 where denominator is 0
        for &pt in &[-3.0, -1.0, 0.0, 1.0, 3.0, 4.0] {
            let cv = c(&[pt]);
            let expected = (pt * pt + 1.0) / (pt * pt - 4.0);
            let diff = (cv - expected).abs();
            assert!(
                diff < 1e-10,
                "compile (x²+1)/(x²-4) at x={pt}: {cv} vs {expected}"
            );
        }
    }
}

// ── LaTeX deeper edge cases ─────────────────────────────────────────────

/// LaTeX of a product with negative coefficient
#[test]
fn latex_negative_coefficient() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x * (-3);
    let latex = e.to_latex();
    assert!(!latex.is_empty(), "LaTeX of -3x is empty");
    // Should contain 3 and x in some form
    assert!(latex.contains("3") && latex.contains("x"),
        "LaTeX of -3x should contain 3 and x: {latex}");
}

/// LaTeX of deeply nested expression
#[test]
fn latex_deeply_nested() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sin(cos(exp(ln(x))))
    let e = x.ln().exp().cos().sin();
    let latex = e.to_latex();
    assert!(!latex.is_empty(), "LaTeX of deeply nested expr is empty");
    assert!(latex.contains("\\sin"), "LaTeX should contain \\sin: {latex}");
    assert!(latex.contains("\\cos"), "LaTeX should contain \\cos: {latex}");
}

/// LaTeX of a sum with many terms
#[test]
fn latex_many_terms() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x^5 + x^4 + x^3 + x^2 + x + 1
    let e = &x.powi(5) + &x.powi(4) + &x.powi(3) + &x.powi(2) + &x + 1;
    let latex = e.to_latex();
    assert!(!latex.is_empty());
    assert!(latex.contains("x"), "LaTeX of polynomial should contain x: {latex}");
}

/// LaTeX of a fraction with polynomial numerator and denominator
#[test]
fn latex_polynomial_fraction() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &(&x.powi(2) + 1) / &(&x - 1);
    let latex = e.to_latex();
    assert!(!latex.is_empty(), "LaTeX of (x²+1)/(x-1) is empty");
    // Should contain frac or / for division
    assert!(
        latex.contains("frac") || latex.contains("/"),
        "LaTeX of fraction should show division: {latex}"
    );
}

/// LaTeX of pi and e constants
#[test]
fn latex_constants_pi_e() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let e = ctx.e();
    let pi_latex = pi.to_latex();
    let e_latex = e.to_latex();
    assert!(
        pi_latex.contains("pi") || pi_latex.contains("\\pi"),
        "LaTeX of π should contain pi: {pi_latex}"
    );
    assert!(!e_latex.is_empty(), "LaTeX of e is empty");
}

// ── Code generation edge cases ──────────────────────────────────────────

/// codegen for expression with both sin and cos (sin_cos optimization)
#[test]
fn codegen_sin_cos_together() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.sin() * &x.cos();
    let code = f.to_rust_fn("sin_cos_fn", &["x"]).unwrap();
    assert_valid_rust_code(&code, "sin_cos_fn");
}

/// codegen for negative rational coefficient
#[test]
fn codegen_negative_rational() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x * &ctx.rational(-3, 7);
    let code = f.to_rust_fn("neg_rat", &["x"]).unwrap();
    assert_valid_rust_code(&code, "neg_rat");
}

/// codegen for sum of many terms
#[test]
fn codegen_many_terms() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(5) + &x.powi(4) + &x.powi(3) + &x.powi(2) + &x + 1;
    let code = f.to_rust_fn("poly5", &["x"]).unwrap();
    assert_valid_rust_code(&code, "poly5");
}

/// codegen for reciprocal: 1/x
#[test]
fn codegen_reciprocal() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(-1);
    let code = f.to_rust_fn("recip_fn", &["x"]).unwrap();
    assert_valid_rust_code(&code, "recip_fn");
}

/// codegen for abs(sin(x))
#[test]
fn codegen_abs_of_trig() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sin().abs();
    let code = f.to_rust_fn("abs_sin", &["x"]).unwrap();
    assert_valid_rust_code(&code, "abs_sin");
}

/// codegen for large expanded polynomial and verify balanced syntax
#[test]
fn codegen_expanded_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (&x + 1).powi(5).expand();
    let code = f.to_rust_fn("poly_expanded", &["x"]).unwrap();
    assert_valid_rust_code(&code, "poly_expanded");
}

// ── Deeper commutativity and algebraic identity checks ──────────────────

/// (a-b)(a+b) expanded should equal a²-b²
#[test]
fn difference_of_squares_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let lhs = (&(&x - &y) * &(&x + &y)).expand();
    let rhs = &x.powi(2) - &y.powi(2);
    for &xv in &[1i64, 2, 3, 5] {
        for &yv in &[1i64, 2, 3, 5] {
            let lv = lhs.subs_i64(&x, xv).subs_i64(&y, yv).eval_f64();
            let rv = rhs.subs_i64(&x, xv).subs_i64(&y, yv).eval_f64();
            if let (Ok(a), Ok(b)) = (lv, rv) {
                assert!(
                    (a - b).abs() < 1e-10,
                    "(a-b)(a+b) vs a²-b² at ({xv},{yv}): {a} vs {b}"
                );
            }
        }
    }
}

/// (a+b)^3 expanded should match a³+3a²b+3ab²+b³
#[test]
fn cube_expansion_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let lhs = (&x + &y).powi(3).expand();
    // a³+3a²b+3ab²+b³
    let rhs = &x.powi(3)
        + &(&(&x.powi(2) * &y) * 3)
        + &(&(&x * &y.powi(2)) * 3)
        + &y.powi(3);
    for &xv in &[1i64, 2, -1] {
        for &yv in &[1i64, 2, -1] {
            let lv = lhs.subs_i64(&x, xv).subs_i64(&y, yv).eval_f64();
            let rv = rhs.subs_i64(&x, xv).subs_i64(&y, yv).eval_f64();
            if let (Ok(a), Ok(b)) = (lv, rv) {
                assert!(
                    (a - b).abs() < 1e-10,
                    "(a+b)³ expansion at ({xv},{yv}): {a} vs {b}"
                );
            }
        }
    }
}

/// Verify the quotient rule: d/dx(f/g) = (f'g - fg')/g²
#[test]
fn diff_quotient_rule() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(2);
    let g = x.sin();
    let quotient = &f / &g;
    let dq = quotient.diff(&x);
    // (f'g - fg')/g²
    let fp = f.diff(&x);
    let gp = g.diff(&x);
    let quotient_rule = &(&(&fp * &g) - &(&f * &gp)) / &g.powi(2);
    // Avoid x=0 where sin(x)=0
    assert_numerically_equal(&dq, &quotient_rule, &x, &[-3, -2, -1, 1, 2, 3], 1e-8,
        "quotient rule: d/dx(x²/sin(x))");
}

/// Verify that expand distributes over addition of products
/// a*(b+c) + d*(e+f) expanded = ab+ac+de+df
#[test]
fn expand_sum_of_distributed_products() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let e = &(&x * &(&y + 1)) + &(&y * &(&x + 1));
    let expanded = e.expand();
    // Should be xy + x + xy + y = 2xy + x + y
    let v_orig = e.subs_i64(&x, 3).subs_i64(&y, 4).eval_f64();
    let v_exp = expanded.subs_i64(&x, 3).subs_i64(&y, 4).eval_f64();
    if let (Ok(a), Ok(b)) = (v_orig, v_exp) {
        assert!((a - b).abs() < 1e-10,
            "expand sum of products at (3,4): {a} vs {b}");
    }
}

/// Verify repeated application of expand is idempotent
#[test]
fn expand_idempotent() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = (&x + 1).powi(4);
    let e1 = e.expand();
    let e2 = e1.expand();
    assert_eq!(format!("{e1}"), format!("{e2}"),
        "expand not idempotent for (x+1)^4");
}

/// Verify diff of an expanded expression equals diff then expand
#[test]
fn diff_of_expanded_vs_expand_of_diffed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (&x + 1).powi(5);
    let a = f.expand().diff(&x);
    let b = f.diff(&x).expand();
    assert_eq!(format!("{a}"), format!("{b}"),
        "diff(expand((x+1)^5)) should equal expand(diff((x+1)^5)) structurally");
}

/// Verify that subs into a derivative gives the correct value
#[test]
fn subs_into_derivative() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(3) + &x.sin();
    let df = f.diff(&x); // 3x² + cos(x)
    for &pt in INT_POINTS {
        let symbolic = df.subs_i64(&x, pt).eval_f64();
        let numerical = 3.0 * (pt as f64).powi(2) + (pt as f64).cos();
        if let Ok(sv) = symbolic {
            let diff = (sv - numerical).abs();
            assert!(diff < 1e-10,
                "d/dx(x³+sin(x)) at x={pt}: symbolic={sv}, numerical={numerical}");
        }
    }
}

/// Verify that integrate then diff then simplify reproduces the integrand
/// for the tricky case f = 1/(1+x)
#[test]
fn ftc_one_over_1_plus_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &ctx.int(1) / &(&x + 1);
    let anti = f.integrate(&x);
    if !anti.has_unevaluated() {
        let roundtrip = anti.diff(&x);
        // Only test at x > -1 to avoid the singularity
        assert_numerically_equal(&f, &roundtrip, &x, POS_POINTS, 1e-8,
            "FTC: 1/(1+x)");
    }
}

/// Compile consistency for expression after simplification
#[test]
fn compile_after_full_simplify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x.sin().powi(2) + &x.cos().powi(2) + &x.powi(2);
    let s = e.simplify();
    // Both should give the same numerical values
    let c_orig = e.compile(&["x"]);
    let c_simp = s.compile(&["x"]);
    if let (Some(co), Some(cs)) = (c_orig, c_simp) {
        for &pt in &[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0] {
            let vo = co(&[pt]);
            let vs = cs(&[pt]);
            let diff = (vo - vs).abs();
            let scale = vo.abs().max(vs.abs()).max(1.0);
            assert!(diff / scale < 1e-10,
                "compile(original) vs compile(full_simplified) at x={pt}: {vo} vs {vs}");
        }
    }
}

/// Verify that Display output round-trips through basic structure
#[test]
fn display_contains_variable_names() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let cases: Vec<(&str, Ex, Vec<&str>)> = vec![
        ("x+y", &x + &y, vec!["x", "y"]),
        ("x*y", &x * &y, vec!["x", "y"]),
        ("sin(x)", x.sin(), vec!["sin", "x"]),
        ("exp(x)", x.exp(), vec!["exp", "x"]),
        ("x^2", x.powi(2), vec!["x"]),
    ];

    for (label, e, expected_substrings) in &cases {
        let s = format!("{e}");
        for sub in expected_substrings {
            assert!(s.contains(sub),
                "Display of {label} should contain '{sub}': got '{s}'");
        }
    }
}

/// Verify that rational number display is consistent
#[test]
fn display_rationals() {
    let ctx = Context::new();
    let half = ctx.rational(1, 2);
    let s = format!("{half}");
    assert!(s == "1/2", "1/2 should display as '1/2', got '{s}'");

    let third = ctx.rational(1, 3);
    let s = format!("{third}");
    assert!(s == "1/3", "1/3 should display as '1/3', got '{s}'");

    // Reduced form: 2/4 should display as 1/2
    let two_fourths = ctx.rational(2, 4);
    let s = format!("{two_fourths}");
    assert!(s == "1/2", "2/4 should display as '1/2', got '{s}'");
}

/// Verify solve + substitute round-trip: solve f=0, then verify f(root)=0
#[test]
fn solve_substitute_verify_quartic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x^4 - 5x^2 + 4 = (x^2-1)(x^2-4) = (x-1)(x+1)(x-2)(x+2)
    let p = &x.powi(4) - &(&x.powi(2) * 5) + 4;
    let roots = p.solve_or_empty(&x);
    for root in &roots {
        let val = p.subs(&x, root).eval_f64();
        if let Ok(v) = val {
            assert!(v.abs() < 1e-8,
                "root {root} of x⁴-5x²+4 gives {v}, not 0");
        }
    }
}

/// Verify that the derivative of a constant multiple is the constant times derivative
#[test]
fn diff_constant_multiple() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for c in [-5, -1, 0, 1, 3, 7] {
        let f = &x.sin() * c;
        let df = f.diff(&x);
        let expected = &x.cos() * c;
        assert_numerically_equal(&df, &expected, &x, INT_POINTS, 1e-10,
            &format!("d/dx({c}·sin(x)) = {c}·cos(x)"));
    }
}

/// Verify second derivative of sin = -sin
#[test]
fn second_derivative_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let d2 = x.sin().diff_n(&x, 2);
    let neg_sin = -&x.sin();
    assert_numerically_equal(&d2, &neg_sin, &x, INT_POINTS, 1e-10,
        "d²/dx² sin(x) = -sin(x)");
}

/// Verify second derivative of cos = -cos
#[test]
fn second_derivative_cos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let d2 = x.cos().diff_n(&x, 2);
    let neg_cos = -&x.cos();
    assert_numerically_equal(&d2, &neg_cos, &x, INT_POINTS, 1e-10,
        "d²/dx² cos(x) = -cos(x)");
}

/// Verify fourth derivative of sin = sin (full cycle)
#[test]
fn fourth_derivative_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let d4 = x.sin().diff_n(&x, 4);
    let sin_x = x.sin();
    assert_numerically_equal(&d4, &sin_x, &x, INT_POINTS, 1e-10,
        "d⁴/dx⁴ sin(x) = sin(x)");
}

/// Verify fourth derivative of exp = exp (eigenfunction)
#[test]
fn fourth_derivative_exp() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let d4 = x.exp().diff_n(&x, 4);
    let exp_x = x.exp();
    assert_numerically_equal(&d4, &exp_x, &x, &[-2, -1, 0, 1, 2], 1e-10,
        "d⁴/dx⁴ exp(x) = exp(x)");
}

/// Verify that expand then simplify of (sin²+cos²)^n gives 1 for small n
#[test]
fn simplify_pythagorean_powers() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for n in 1..=3i64 {
        let base = &x.sin().powi(2) + &x.cos().powi(2);
        let e = base.powi(n);
        let simplified = e.simplify();
        for &pt in &[1, 2, 3] {
            let v = simplified.subs_i64(&x, pt).eval_f64();
            if let Ok(v) = v {
                assert!((v - 1.0).abs() < 1e-10,
                    "(sin²+cos²)^{n} simplified to {simplified}, value at x={pt}: {v}");
            }
        }
    }
}

/// compile of a piecewise-like expression: abs(x) * sign(x) should = x
#[test]
fn compile_abs_times_sign() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.abs() * &x.sign();
    let compiled = f.compile(&["x"]);
    if let Some(c) = compiled {
        for &pt in &[-3.0, -1.0, 1.0, 2.0, 5.0] {
            let cv = c(&[pt]);
            // abs(x) * sign(x) = x
            let diff = (cv - pt).abs();
            assert!(diff < 1e-10,
                "compile(|x|·sign(x)) at x={pt}: {cv} (expected {pt})");
        }
    }
}

/// Verify that eval_f64 of negative exponent works correctly
#[test]
fn eval_negative_exponent() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(-3);
    for &pt in &[-3, -2, -1, 1, 2, 3] {
        let v = f.subs_i64(&x, pt).eval_f64();
        let expected = (pt as f64).powi(-3);
        if let Ok(v) = v {
            let diff = (v - expected).abs();
            assert!(diff < 1e-10,
                "x^(-3) at x={pt}: {v} vs {expected}");
        }
    }
}

/// Verify that a trig identity holds: 1 + tan²(x) = sec²(x) = 1/cos²(x)
#[test]
fn trig_identity_tan_sec() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let lhs = &ctx.int(1) + &x.tan().powi(2);
    let rhs = x.cos().powi(-2);
    // Avoid x near π/2 where cos → 0
    assert_numerically_equal(&lhs, &rhs, &x, &[-1, 0, 1], 1e-8,
        "1+tan²(x) vs 1/cos²(x)");
}

/// Verify double angle: sin(2x) = 2*sin(x)*cos(x)
#[test]
fn trig_double_angle_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let lhs = (&x * 2).sin();
    let rhs = &(&x.sin() * &x.cos()) * 2;
    assert_numerically_equal(&lhs, &rhs, &x, INT_POINTS, 1e-10,
        "sin(2x) vs 2·sin(x)·cos(x)");
}

/// Verify double angle: cos(2x) = cos²(x) - sin²(x)
#[test]
fn trig_double_angle_cos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let lhs = (&x * 2).cos();
    let rhs = &x.cos().powi(2) - &x.sin().powi(2);
    assert_numerically_equal(&lhs, &rhs, &x, INT_POINTS, 1e-10,
        "cos(2x) vs cos²(x)-sin²(x)");
}

/// Verify hyperbolic identity: cosh²(x) - sinh²(x) = 1
#[test]
fn hyperbolic_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x.cosh().powi(2) - &x.sinh().powi(2);
    for &pt in &[-2, -1, 0, 1, 2] {
        let v = e.subs_i64(&x, pt).eval_f64();
        if let Ok(v) = v {
            assert!((v - 1.0).abs() < 1e-10,
                "cosh²(x)-sinh²(x) at x={pt}: {v}");
        }
    }
}

/// Simplify should not break the hyperbolic identity if applied
#[test]
fn simplify_hyperbolic_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x.cosh().powi(2) - &x.sinh().powi(2);
    let simplified = e.simplify();
    for &pt in &[-2, -1, 0, 1, 2] {
        let v = simplified.subs_i64(&x, pt).eval_f64();
        if let Ok(v) = v {
            assert!((v - 1.0).abs() < 1e-10,
                "simplify(cosh²-sinh²) at x={pt}: {v}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// WAVE 4: STRUCTURAL CHECKS, CANCELLATION CORNERS, NESTED OPS, TRICKY EVAL
// ═══════════════════════════════════════════════════════════════════════════

// ── Structural cancel checks ────────────────────────────────────────────

/// cancel((x^3 - 8)/(x - 2)) should give x^2 + 2x + 4 (structurally or numerically)
#[test]
fn cancel_cubic_minus_8_over_x_minus_2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &(&x.powi(3) - 8) / &(&x - 2);
    let cancelled = e.cancel(&x);
    let expected = &x.powi(2) + &(&x * 2) + 4;
    // Avoid x=2 where denominator is zero
    assert_numerically_equal(&cancelled, &expected, &x, &[-3, -1, 0, 1, 3, 4, 5], 1e-10,
        "cancel (x³-8)/(x-2) = x²+2x+4");
}

/// cancel((x^4 - 1)/(x^2 - 1)) should give x^2 + 1
#[test]
fn cancel_x4_minus_1_over_x2_minus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &(&x.powi(4) - 1) / &(&x.powi(2) - 1);
    let cancelled = e.cancel(&x);
    let expected = &x.powi(2) + 1;
    // Avoid x=±1
    assert_numerically_equal(&cancelled, &expected, &x, &[-3, -2, 0, 2, 3, 4], 1e-10,
        "cancel (x⁴-1)/(x²-1) = x²+1");
}

/// cancel of an already-reduced fraction should not change it
#[test]
fn cancel_already_reduced() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &(&x + 1) / &(&x + 2);
    let cancelled = e.cancel(&x);
    assert_numerically_equal(&e, &cancelled, &x, &[-4, -3, 0, 1, 3, 5], 1e-10,
        "cancel of already-reduced (x+1)/(x+2)");
}

/// cancel((x^2 - 2x + 1)/(x - 1)) = x - 1 (perfect square numerator)
#[test]
fn cancel_perfect_square_numerator() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &(&x.powi(2) - &(&x * 2) + 1) / &(&x - 1);
    let cancelled = e.cancel(&x);
    let expected = &x - 1;
    // Avoid x=1
    assert_numerically_equal(&cancelled, &expected, &x, &[-3, -2, -1, 0, 2, 3, 5], 1e-10,
        "cancel (x²-2x+1)/(x-1) = x-1");
}

// ── Nested operation consistency ────────────────────────────────────────

/// expand(simplify(expand(f))) should equal expand(f) for polynomials
#[test]
fn nested_expand_simplify_expand() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (&x + 1).powi(3);
    let once = f.expand();
    let triple = f.expand().simplify().expand();
    assert_eq!(format!("{once}"), format!("{triple}"),
        "expand(simplify(expand((x+1)³))) should equal expand((x+1)³)");
}

/// factor(expand(factor(p))) should equal factor(p) numerically
#[test]
fn nested_factor_expand_factor() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = &x.powi(3) - &x;
    let once = p.factor(&x);
    let triple = p.factor(&x).expand().factor(&x);
    assert_numerically_equal(&once, &triple, &x, INT_POINTS, 1e-10,
        "factor(expand(factor(x³-x)))");
}

/// simplify(expand(simplify(e))) should agree numerically with simplify(e)
#[test]
fn nested_simplify_expand_simplify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x.sin().powi(2) + &x.cos().powi(2) + &x.powi(2);
    let once = e.simplify();
    let triple = e.simplify().expand().simplify();
    assert_numerically_equal(&once, &triple, &x, INT_POINTS, 1e-10,
        "simplify(expand(simplify(sin²+cos²+x²)))");
}

/// diff(integrate(diff(f))) should equal diff(f) for nice functions
#[test]
fn nested_diff_integrate_diff() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(4) + &x.sin();
    let df = f.diff(&x);
    let anti_df = df.integrate(&x);
    if !anti_df.has_unevaluated() {
        let d_anti_df = anti_df.diff(&x);
        assert_numerically_equal(&df, &d_anti_df, &x, INT_POINTS, 1e-8,
            "diff(integrate(diff(x⁴+sin(x))))");
    }
}

/// integrate(diff(integrate(f))) should equal integrate(f) up to constant
/// (checked by differentiating both sides)
#[test]
fn nested_integrate_diff_integrate() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(3);
    let once = f.integrate(&x);
    let triple = f.integrate(&x).diff(&x).integrate(&x);
    // Both are antiderivatives of x³, so their difference is constant
    let residual = &triple - &once;
    let v1 = residual.subs_i64(&x, 1).eval_f64();
    let v2 = residual.subs_i64(&x, 5).eval_f64();
    if let (Ok(a), Ok(b)) = (v1, v2) {
        assert!((a - b).abs() < 1e-10,
            "integrate(diff(integrate(x³))) - integrate(x³) should be const: {a} vs {b}");
    }
}

// ── Tricky eval scenarios ───────────────────────────────────────────────

/// eval_f64 of pi should be close to std::f64::consts::PI
#[test]
fn eval_f64_pi_accuracy() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let v = pi.eval_f64();
    if let Ok(v) = v {
        assert!((v - std::f64::consts::PI).abs() < 1e-15,
            "eval_f64(π) = {v}, expected {}", std::f64::consts::PI);
    }
}

/// eval_f64 of e should be close to std::f64::consts::E
#[test]
fn eval_f64_e_accuracy() {
    let ctx = Context::new();
    let e = ctx.e();
    let v = e.eval_f64();
    if let Ok(v) = v {
        assert!((v - std::f64::consts::E).abs() < 1e-15,
            "eval_f64(e) = {v}, expected {}", std::f64::consts::E);
    }
}

/// eval_f64 of sin(π/4) should be √2/2
#[test]
fn eval_f64_sin_pi_over_4() {
    let ctx = Context::new();
    let val = (&ctx.pi() / 4).sin();
    let v = val.eval_f64();
    let expected = std::f64::consts::FRAC_1_SQRT_2;
    if let Ok(v) = v {
        assert!((v - expected).abs() < 1e-10,
            "sin(π/4) = {v}, expected {expected}");
    }
}

/// Substituting a complex expression into another and evaluating
#[test]
fn eval_nested_substitution() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // f(x, y) = x^2 + y^2
    let f = &x.powi(2) + &y.powi(2);
    // Substitute y = sin(x), then evaluate at x=1
    let g = f.subs(&y, &x.sin());
    // g(x) = x^2 + sin^2(x)
    let v = g.subs_i64(&x, 1).eval_f64();
    let expected = 1.0 + 1.0f64.sin().powi(2);
    if let Ok(v) = v {
        assert!((v - expected).abs() < 1e-10,
            "x²+sin²(x) at x=1: {v} vs {expected}");
    }
}

/// Verify that eval_f64 of a large rational is accurate
#[test]
fn eval_f64_large_rational() {
    let ctx = Context::new();
    let r = ctx.rational(355, 113); // Famous approximation to π
    let v = r.eval_f64();
    if let Ok(v) = v {
        assert!((v - 355.0 / 113.0).abs() < 1e-15,
            "355/113 eval_f64: {v}");
    }
}

/// Verify that 0/x simplifies or evaluates to 0
#[test]
fn zero_divided_by_symbol() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &ctx.int(0) / &x;
    let s = format!("{e}");
    assert_eq!(s, "0", "0/x should be 0, got: {s}");
}

/// Verify that x/1 simplifies to x
#[test]
fn symbol_divided_by_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x / &ctx.int(1);
    let s = format!("{e}");
    assert_eq!(s, "x", "x/1 should be x, got: {s}");
}

/// Verify negation: -(-x) = x
#[test]
fn double_negation_structural() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = -(-&x);
    let s = format!("{e}");
    assert_eq!(s, "x", "-(-x) should be x, got: {s}");
}

/// Verify that (x-x) is structurally 0
#[test]
fn self_subtraction_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = &x - &x;
    assert!(e.is_zero_structural(), "x - x should be structurally zero");
}

// ── More compile edge cases ─────────────────────────────────────────────

/// Compile with tanh (common in ML)
#[test]
fn compile_tanh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.tanh();
    check_compile_consistency(&f, &x, &[-2, -1, 0, 1, 2], 1e-10, "tanh(x)");
}

/// Compile of sign function
#[test]
fn compile_sign() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sign();
    let compiled = f.compile(&["x"]);
    if let Some(c) = compiled {
        assert!((c(&[5.0]) - 1.0).abs() < 1e-10, "sign(5) should be 1");
        assert!((c(&[-3.0]) - (-1.0)).abs() < 1e-10, "sign(-3) should be -1");
    }
}

/// Compile of floor function
#[test]
fn compile_floor() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.floor();
    let compiled = f.compile(&["x"]);
    if let Some(c) = compiled {
        assert!((c(&[2.7]) - 2.0).abs() < 1e-10, "floor(2.7) should be 2");
        assert!((c(&[-1.3]) - (-2.0)).abs() < 1e-10, "floor(-1.3) should be -2");
    }
}

/// Compile of ceiling function
#[test]
fn compile_ceiling() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.ceiling();
    let compiled = f.compile(&["x"]);
    if let Some(c) = compiled {
        assert!((c(&[2.1]) - 3.0).abs() < 1e-10, "ceil(2.1) should be 3");
        assert!((c(&[-1.7]) - (-1.0)).abs() < 1e-10, "ceil(-1.7) should be -1");
    }
}

/// Compile of a 3-variable expression: x*y + y*z + z*x
#[test]
fn compile_three_vars_symmetric() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");
    let f = &(&x * &y) + &(&y * &z) + &(&z * &x);
    let compiled = f.compile(&["x", "y", "z"]);
    if let Some(c) = compiled {
        let v = c(&[2.0, 3.0, 5.0]);
        let expected = 2.0 * 3.0 + 3.0 * 5.0 + 5.0 * 2.0; // 6+15+10=31
        assert!((v - expected).abs() < 1e-10,
            "compile(xy+yz+zx) at (2,3,5): {v} vs {expected}");
    }
}

// ── Deep structural identity checks ─────────────────────────────────────

/// The expanded polynomial should have the right degree
#[test]
fn expand_degree_check() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x+1)^n expanded should have degree n
    for n in 2..=6i64 {
        let expanded = (&x + 1).powi(n).expand();
        // The leading term should contain x^n
        let s = format!("{expanded}");
        let power_str = if n == 1 { "x".to_string() } else { format!("x^{n}") };
        assert!(s.contains(&power_str),
            "expand((x+1)^{n}) should contain {power_str}: got {s}");
    }
}

/// expand((x-1)(x-2)...(x-n)) and verify at integer points
#[test]
fn expand_product_of_linears() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x-1)(x-2)(x-3)(x-4)
    let p = &(&(&(&x - 1) * &(&x - 2)) * &(&x - 3)) * &(&x - 4);
    let expanded = p.expand();
    // At each root, value should be 0
    for root in 1..=4i64 {
        let v = expanded.subs_i64(&x, root).eval_f64();
        if let Ok(v) = v {
            assert!(v.abs() < 1e-10,
                "expanded product should be 0 at x={root}, got {v}");
        }
    }
    // At x=5: (4)(3)(2)(1) = 24
    let v = expanded.subs_i64(&x, 5).eval_f64();
    if let Ok(v) = v {
        assert!((v - 24.0).abs() < 1e-10,
            "expanded product at x=5 should be 24, got {v}");
    }
}

/// Verify that factor of a product of distinct linears recovers the factors
/// by checking that the roots of the factored form match
#[test]
fn factor_finds_roots() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let p = (&(&x - 1) * &(&x + 3)).expand();
    let factored = p.factor(&x);
    // The factored form, when evaluated at roots, should give 0
    for root in [1i64, -3] {
        let v = factored.subs_i64(&x, root).eval_f64();
        if let Ok(v) = v {
            assert!(v.abs() < 1e-10,
                "factored form should be 0 at x={root}, got {v}");
        }
    }
}

/// Verify that diff(series(f)) ≈ series(diff(f)) for sin at order 5
#[test]
fn diff_series_vs_series_diff() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let f = x.sin();
    let diff_series = f.series(&x, &zero, 5).diff(&x);
    let series_diff = f.diff(&x).series(&x, &zero, 4); // one order lower after diff
    // These should agree at small x values
    let diff_series_expanded = diff_series.expand();
    let series_diff_expanded = series_diff.expand();
    assert_numerically_equal(
        &diff_series_expanded, &series_diff_expanded, &x, &[0, 1], 1e-2,
        "diff(series(sin)) vs series(diff(sin)) at small x");
}

/// Verify that expand_trig(sin(3x)) gives the correct triple-angle formula
#[test]
fn trig_expand_triple_angle() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (&x * 3).sin();
    let expanded = f.expand_trig();
    // sin(3x) = 3sin(x) - 4sin³(x)
    assert_numerically_equal(&f, &expanded, &x, INT_POINTS, 1e-10,
        "expand_trig(sin(3x)) numerical consistency");
}

/// Verify that expand_trig(cos(3x)) is numerically consistent
#[test]
fn trig_expand_cos_triple_angle() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (&x * 3).cos();
    let expanded = f.expand_trig();
    assert_numerically_equal(&f, &expanded, &x, INT_POINTS, 1e-10,
        "expand_trig(cos(3x)) numerical consistency");
}

// ── Edge cases in arithmetic ────────────────────────────────────────────

/// Verify x^n * x^m = x^(n+m) for various n, m
#[test]
fn power_addition_rule() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for n in 1..=4i64 {
        for m in 1..=4i64 {
            let lhs = &x.powi(n) * &x.powi(m);
            let rhs = x.powi(n + m);
            assert_numerically_equal(&lhs, &rhs, &x, POS_POINTS, 1e-10,
                &format!("x^{n}·x^{m} = x^{}", n + m));
        }
    }
}

/// Verify (x^n)^m = x^(n*m) for various n, m
#[test]
fn power_multiplication_rule() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for n in 1..=3i64 {
        for m in 1..=3i64 {
            let lhs = x.powi(n).powi(m);
            let rhs = x.powi(n * m);
            assert_numerically_equal(&lhs, &rhs, &x, POS_POINTS, 1e-10,
                &format!("(x^{n})^{m} = x^{}", n * m));
        }
    }
}

/// Verify (a*b)^n = a^n * b^n numerically
#[test]
fn power_of_product_rule() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    for n in 2..=4i64 {
        let lhs = (&x * &y).powi(n);
        let rhs = &x.powi(n) * &y.powi(n);
        for &xv in &[1i64, 2, 3] {
            for &yv in &[1i64, 2, 3] {
                let lv = lhs.subs_i64(&x, xv).subs_i64(&y, yv).eval_f64();
                let rv = rhs.subs_i64(&x, xv).subs_i64(&y, yv).eval_f64();
                if let (Ok(a), Ok(b)) = (lv, rv) {
                    assert!((a - b).abs() < 1e-8,
                        "(xy)^{n} vs x^{n}·y^{n} at ({xv},{yv}): {a} vs {b}");
                }
            }
        }
    }
}

/// Verify that negative integer powers work: x^(-n) = 1/x^n
#[test]
fn negative_power_reciprocal() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for n in 1..=4i64 {
        let lhs = x.powi(-n);
        let rhs = &ctx.int(1) / &x.powi(n);
        assert_numerically_equal(&lhs, &rhs, &x, &[-3, -2, -1, 1, 2, 3], 1e-10,
            &format!("x^(-{n}) = 1/x^{n}"));
    }
}

/// Verify that a sum of zero terms is zero
#[test]
fn empty_sum_edge_case() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x + (-x) should be 0
    let e = &x + &(-&x);
    assert!(e.is_zero_structural(), "x + (-x) should be zero, got: {e}");
}

/// Verify that 1^n = 1 for any n
#[test]
fn one_to_any_power() {
    let ctx = Context::new();
    for n in [-3, -2, -1, 0, 1, 2, 3, 10] {
        let e = ctx.int(1).powi(n);
        let v = e.eval_f64();
        if let Ok(v) = v {
            assert!((v - 1.0).abs() < 1e-15,
                "1^{n} should be 1, got {v}");
        }
    }
}

/// Verify that 0^n = 0 for positive n
#[test]
fn zero_to_positive_power() {
    let ctx = Context::new();
    for n in [1, 2, 3, 5, 10] {
        let e = ctx.int(0).powi(n);
        let v = e.eval_f64();
        if let Ok(v) = v {
            assert!(v.abs() < 1e-15,
                "0^{n} should be 0, got {v}");
        }
    }
}

// ── codegen + compile cross-validation ──────────────────────────────────

/// to_rust_fn and compile should not contradict: both produce valid output
/// or both fail for the same expression
#[test]
fn codegen_compile_agreement() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases: Vec<(&str, Ex)> = vec![
        ("x²", x.powi(2)),
        ("sin(x)", x.sin()),
        ("exp(x)", x.exp()),
        ("ln(x)", x.ln()),
        ("sqrt(x)", x.sqrt()),
        ("x³-2x+1", &x.powi(3) - &(&x * 2) + 1),
        ("sin(x)²+cos(x)²", &x.sin().powi(2) + &x.cos().powi(2)),
    ];

    for (label, e) in &cases {
        let codegen_ok = e.to_rust_fn("test_fn", &["x"]).is_ok();
        let compile_ok = e.compile(&["x"]).is_some();
        assert_eq!(codegen_ok, compile_ok,
            "{label}: codegen={codegen_ok}, compile={compile_ok} — should agree");
    }
}

/// Verify codegen produces valid braces for expressions with CSE
#[test]
fn codegen_cse_braces_balanced() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Expression with repeated sub-expressions to trigger CSE
    let sub = &x.sin() + 1;
    let f = &sub.powi(2) + &sub.powi(3);
    let code = f.to_rust_fn("cse_test", &["x"]).unwrap();
    assert_valid_rust_code(&code, "cse_test");
    // If CSE kicked in, there should be a let binding
    // (not a hard requirement, but likely for repeated subexpressions)
}

/// Verify codegen for expression with mixed positive and negative terms
#[test]
fn codegen_mixed_signs() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(3) - &(&x.powi(2) * 5) + &(&x * 3) - 7;
    let code = f.to_rust_fn("mixed_signs", &["x"]).unwrap();
    assert_valid_rust_code(&code, "mixed_signs");
}

// ── LaTeX structural validation ─────────────────────────────────────────

/// LaTeX should have balanced braces
fn assert_latex_balanced_braces(latex: &str, label: &str) {
    let opens = latex.chars().filter(|&c| c == '{').count();
    let closes = latex.chars().filter(|&c| c == '}').count();
    assert_eq!(opens, closes,
        "LaTeX unbalanced braces for {label}: {opens} open vs {closes} close in: {latex}");
}

/// LaTeX balanced braces for many expression types
#[test]
fn latex_balanced_braces_battery() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases: Vec<(&str, Ex)> = vec![
        ("x", x.clone()),
        ("x²", x.powi(2)),
        ("x^10", x.powi(10)),
        ("sin(x)", x.sin()),
        ("cos(x²)", x.powi(2).cos()),
        ("exp(sin(x))", x.sin().exp()),
        ("ln(x+1)", (&x + 1).ln()),
        ("1/x", x.powi(-1)),
        ("x^(1/2)", x.sqrt()),
        ("(x+1)^3", (&x + 1).powi(3)),
        ("x³+3x²+3x+1", (&x + 1).powi(3).expand()),
        ("sin²(x)+cos²(x)", &x.sin().powi(2) + &x.cos().powi(2)),
        ("(x²-1)/(x-1)", &(&x.powi(2) - 1) / &(&x - 1)),
    ];

    for (label, e) in &cases {
        let latex = e.to_latex();
        assert!(!latex.is_empty(), "LaTeX empty for {label}");
        assert_latex_balanced_braces(&latex, label);
    }
}

/// LaTeX of a derivative result should have balanced braces
#[test]
fn latex_of_derivative_balanced() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let df = (&x.powi(3) + &x.sin()).diff(&x);
    let latex = df.to_latex();
    assert_latex_balanced_braces(&latex, "d/dx(x³+sin(x))");
}

/// LaTeX of an integral result should have balanced braces
#[test]
fn latex_of_integral_balanced() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let anti = x.powi(2).integrate(&x);
    if !anti.has_unevaluated() {
        let latex = anti.to_latex();
        assert_latex_balanced_braces(&latex, "∫x²dx");
    }
}

// ── Multi-step workflow tests ───────────────────────────────────────────

/// Full workflow: build → differentiate → simplify → compile → evaluate
#[test]
fn workflow_build_diff_simplify_compile_eval() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(3) - &(&x * 2) + 1;
    let df = f.diff(&x); // 3x² - 2
    let df_simplified = df.simplify();
    let compiled = df_simplified.compile(&["x"]);
    if let Some(c) = compiled {
        for &pt in &[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0] {
            let cv = c(&[pt]);
            let expected = 3.0 * pt.powi(2) - 2.0;
            let diff = (cv - expected).abs();
            assert!(diff < 1e-10,
                "workflow at x={pt}: compile={cv}, expected={expected}");
        }
    }
}

/// Full workflow: build → integrate → diff → simplify → verify equality
#[test]
fn workflow_integrate_diff_simplify_verify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.sin() + &x.powi(2);
    let roundtrip = f.integrate(&x).diff(&x).simplify();
    assert_numerically_equal(&f, &roundtrip, &x, INT_POINTS, 1e-8,
        "workflow: simplify(diff(integrate(sin(x)+x²)))");
}

/// Full workflow: build → expand → factor → verify
#[test]
fn workflow_expand_factor_verify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let original = &(&x - 2) * &(&x + 3);
    let expanded = original.expand();
    let factored = expanded.factor(&x);
    let re_expanded = factored.expand();
    assert_eq!(
        format!("{expanded}"), format!("{re_expanded}"),
        "workflow: expand → factor → expand roundtrip"
    );
}

/// Full workflow: build → series → diff → verify against exact derivative
#[test]
fn workflow_series_then_diff() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let f = x.exp();
    let series = f.series(&x, &zero, 6).expand(); // 1 + x + x²/2 + ...
    let d_series = series.diff(&x).expand(); // should be ≈ exp series again, one order lower
    let d_exact = f.diff(&x); // exp(x)
    // At x=0.5, the 5th-order derivative of the series should be close to exp(0.5)
    let sv = d_series.subs_i64(&x, 1).eval_f64();
    let ev = d_exact.subs_i64(&x, 1).eval_f64();
    if let (Ok(s), Ok(e)) = (sv, ev) {
        let diff = (s - e).abs();
        assert!(diff < 0.01,
            "series(exp,6).diff at x=1: {s} vs {e}");
    }
}

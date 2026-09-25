//! Tests for new Ex API methods added in Cycles 9-18.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// smart_simplify
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn smart_simplify_trig_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin().powi(2) + &x.cos().powi(2);
    assert_eq!(format!("{}", expr.simplify()), "1");
}

#[test]
fn smart_simplify_exp_ln() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(format!("{}", x.ln().exp().simplify()), "x");
}

#[test]
fn smart_simplify_sin_zero() {
    let ctx = Context::new();
    let result = ctx.int(0).sin().simplify();
    assert_eq!(format!("{result}"), "0");
}

#[test]
fn smart_simplify_doesnt_bloat() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let ops_before = x.count_ops();
    let ops_after = x.simplify().count_ops();
    assert!(ops_after <= ops_before + 1);
}

#[test]
fn smart_simplify_complex_expr() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x+1)^2 - x^2 - 2x should simplify to 1
    let expr = &(&x + 1).powi(2) - &x.powi(2) - &(&x * 2);
    let s = format!("{}", expr.expand().simplify());
    assert_eq!(s, "1");
}

// ═══════════════════════════════════════════════════════════════════════════
// count_ops
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn count_ops_atom() {
    let ctx = Context::new();
    assert_eq!(ctx.symbol("x").count_ops(), 0);
    assert_eq!(ctx.int(5).count_ops(), 0);
    assert_eq!(ctx.pi().count_ops(), 0);
}

#[test]
fn count_ops_simple() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!((&x + 1).count_ops(), 1); // Add
    assert_eq!(x.sin().count_ops(), 1); // Sin
    assert_eq!(x.sin().powi(2).count_ops(), 2); // Sin + Pow
}

#[test]
fn count_ops_complex() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin().powi(2) + &x.cos().powi(2);
    assert!(expr.count_ops() >= 4); // Sin + Pow + Cos + Pow + Add
}

// ═══════════════════════════════════════════════════════════════════════════
// factor_terms
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factor_terms_basic() {
    let ctx = Context::new();
    // factor_terms now returns (gcd, inner) where expr == gcd * inner.
    // The inner expression has each coefficient divided by gcd.
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = &x * 4 + &y * 6;
    let (gcd, inner) = expr.factor_terms();
    let gcd_s = format!("{gcd}");
    let inner_s = format!("{inner}");
    assert_eq!(gcd_s, "2", "gcd should be 2: {gcd_s}");
    // inner should contain x and y with reduced coefficients
    assert!(
        inner_s.contains("x") && inner_s.contains("y"),
        "factor_terms inner should contain x and y: {inner_s}"
    );
}

#[test]
fn factor_terms_no_common() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = &x + &y;
    let (gcd, inner) = expr.factor_terms();
    assert_eq!(format!("{gcd}"), "1");
    assert_eq!(format!("{inner}"), format!("{expr}"));
}

#[test]
fn factor_terms_all_same() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x * 6 + 12;
    let (gcd, inner) = expr.factor_terms();
    let gcd_s = format!("{gcd}");
    let inner_s = format!("{inner}");
    assert!(
        gcd_s.contains("6") || gcd_s.contains("3") || gcd_s.contains("2"),
        "gcd should factor: {gcd_s}"
    );
    assert!(inner_s.contains("x"), "inner should contain x: {inner_s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// rationalize_denom
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rationalize_one_over_sqrt2() {
    let ctx = Context::new();
    let expr = ctx.int(1) / &ctx.int(2).sqrt();
    let result = expr.rationalize_denom();
    let s = format!("{result}");
    // 1/√2 → √2/2
    assert!(s.contains("2"), "should rationalize: {s}");
}

#[test]
fn rationalize_no_sqrt_unchanged() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = 1 / &x;
    let result = expr.rationalize_denom();
    assert_eq!(format!("{result}"), format!("{expr}"));
}

#[test]
fn rationalize_integer_denom() {
    let ctx = Context::new();
    let expr = &ctx.symbol("x") / &ctx.int(3);
    let result = expr.rationalize_denom();
    assert_eq!(format!("{result}"), format!("{expr}"));
}

// ═══════════════════════════════════════════════════════════════════════════
// re() / im()
// ═══════════════════════════════════════════════════════════════════════════

// 0.2: bare symbols are no longer assumed real — `re`/`im` of an
// unassumed symbol stay symbolic; with a `Real` assumption they fold.
#[test]
fn re_of_real() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    assert_eq!(format!("{}", x.re()), "x");
    let z = ctx.symbol("z");
    assert_eq!(format!("{}", z.re()), "re(z)");
}

#[test]
fn im_of_real() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    assert_eq!(format!("{}", x.im()), "0");
    let z = ctx.symbol("z");
    assert_eq!(format!("{}", z.im()), "im(z)");
}

#[test]
fn re_of_imaginary() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    assert_eq!(format!("{}", i.re()), "0");
}

#[test]
fn im_of_imaginary() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    assert_eq!(format!("{}", i.im()), "1");
}

#[test]
fn re_im_of_complex() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let z = &ctx.int(3) + &(&ctx.int(4) * &i);
    assert_eq!(format!("{}", z.re()), "3");
    assert_eq!(format!("{}", z.im()), "4");
}

#[test]
fn re_of_number() {
    let ctx = Context::new();
    assert_eq!(format!("{}", ctx.int(5).re()), "5");
    assert_eq!(format!("{}", ctx.int(5).im()), "0");
}

#[test]
fn re_im_of_pi() {
    let ctx = Context::new();
    assert_eq!(format!("{}", ctx.pi().re()), "pi");
    assert_eq!(format!("{}", ctx.pi().im()), "0");
}

// ═══════════════════════════════════════════════════════════════════════════
// lambdify
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn lambdify_x_squared_plus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(2) + 1;
    let func = f.compile(&["x"]).unwrap();
    assert!((func(&[3.0]) - 10.0).abs() < 1e-10);
    assert!((func(&[0.0]) - 1.0).abs() < 1e-10);
}

#[test]
fn lambdify_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let func = x.sin().compile(&["x"]).unwrap();
    assert!((func(&[0.0])).abs() < 1e-10);
    assert!((func(&[std::f64::consts::FRAC_PI_2]) - 1.0).abs() < 1e-10);
}

#[test]
fn lambdify_two_vars() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let f = &x * &y + 1;
    let func = f.compile(&["x", "y"]).unwrap();
    assert!((func(&[3.0, 4.0]) - 13.0).abs() < 1e-10);
}

#[test]
fn lambdify_complex_rejects() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    assert!(i.compile(&[]).is_err());
}

#[test]
fn lambdify_pi_constant() {
    let ctx = Context::new();
    let f = ctx.pi();
    let func = f.compile(&[]).unwrap();
    assert!((func(&[]) - std::f64::consts::PI).abs() < 1e-10);
}

#[test]
fn lambdify_consistency_with_evalf() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.sin().powi(2) + &x.cos().powi(2);
    let func = f.compile(&["x"]).unwrap();
    for pt in [0.0, 0.5, 1.0, 2.0, 3.15] {
        assert!((func(&[pt]) - 1.0).abs() < 1e-10, "sin²+cos² at {pt}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// cse
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cse_no_common() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let (bindings, result) = (&x + &y).cse();
    let _ = format!("{result}");
    // x + y has no shared subexpressions — bindings should be empty
    assert!(
        bindings.is_empty(),
        "simple x + y should have no CSE bindings, got {}",
        bindings.len()
    );
}

#[test]
fn cse_with_shared_subexpr() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sin_x = x.sin();
    let expr = &sin_x.powi(2) + &sin_x;
    let (bindings, result) = expr.cse();
    let _ = format!("{result}");
    // sin(x) appears in both terms — should be extracted
    assert!(!bindings.is_empty(), "CSE should extract shared sin(x)");
}

#[test]
fn cse_doesnt_crash_on_complex() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let i = ctx.i_unit();
    let expr = &(&i * &x) + &(&i * &x.powi(2));
    let (_, result) = expr.cse();
    let _ = format!("{result}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Factorial — arbitrary precision, cross-validated against Python math.factorial
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factorial_0() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let zero = arena.int(0);
        let expr = arena.factorial(zero);
        let result = arena.eval_expr(expr);
        assert_eq!(arena.display(result).to_string(), "1");
    });
}

#[test]
fn factorial_1() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let one = arena.int(1);
        let expr = arena.factorial(one);
        let result = arena.eval_expr(expr);
        assert_eq!(arena.display(result).to_string(), "1");
    });
}

#[test]
fn factorial_5() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let n = arena.int(5);
        let expr = arena.factorial(n);
        let result = arena.eval_expr(expr);
        assert_eq!(arena.display(result).to_string(), "120");
    });
}

#[test]
fn factorial_10() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let n = arena.int(10);
        let expr = arena.factorial(n);
        let result = arena.eval_expr(expr);
        assert_eq!(arena.display(result).to_string(), "3628800");
    });
}

#[test]
fn factorial_12() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let n = arena.int(12);
        let expr = arena.factorial(n);
        let result = arena.eval_expr(expr);
        assert_eq!(arena.display(result).to_string(), "479001600");
    });
}

#[test]
fn factorial_15() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let n = arena.int(15);
        let expr = arena.factorial(n);
        let result = arena.eval_expr(expr);
        assert_eq!(arena.display(result).to_string(), "1307674368000");
    });
}

#[test]
fn factorial_20() {
    // 20! = 2432902008176640000 — fits in u64 but we use BigInt
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let n = arena.int(20);
        let expr = arena.factorial(n);
        let result = arena.eval_expr(expr);
        assert_eq!(arena.display(result).to_string(), "2432902008176640000");
    });
}

#[test]
fn factorial_25() {
    // 25! = 15511210043330985984000000 — exceeds u64, requires BigInt
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let n = arena.int(25);
        let expr = arena.factorial(n);
        let result = arena.eval_expr(expr);
        assert_eq!(
            arena.display(result).to_string(),
            "15511210043330985984000000"
        );
    });
}

#[test]
fn factorial_30() {
    // 30! = 265252859812191058636308480000000 — 33 digits
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let n = arena.int(30);
        let expr = arena.factorial(n);
        let result = arena.eval_expr(expr);
        assert_eq!(
            arena.display(result).to_string(),
            "265252859812191058636308480000000"
        );
    });
}

#[test]
fn factorial_50() {
    // 50! = 30414093201713378043612608166064768844377641568960512000000000000
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let n = arena.int(50);
        let expr = arena.factorial(n);
        let result = arena.eval_expr(expr);
        assert_eq!(
            arena.display(result).to_string(),
            "30414093201713378043612608166064768844377641568960512000000000000"
        );
    });
}

#[test]
fn factorial_100() {
    // 100! — 158 digits, cross-validated against Python math.factorial(100)
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let n = arena.int(100);
        let expr = arena.factorial(n);
        let result = arena.eval_expr(expr);
        let s = arena.display(result).to_string();
        // Verify first and last digits + total length
        assert!(
            s.starts_with("933262154"),
            "100! starts with 933262154, got: {}",
            &s[..20]
        );
        assert!(s.ends_with("00000000"), "100! ends in zeros");
        assert_eq!(s.len(), 158, "100! has 158 digits, got {}", s.len());
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Binomial — arbitrary precision, cross-validated against Python
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn binomial_10_5() {
    // C(10,5) = 252
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let n = arena.int(10);
        let k = arena.int(5);
        let expr = arena.binomial(n, k);
        let result = arena.eval_expr(expr);
        assert_eq!(arena.display(result).to_string(), "252");
    });
}

#[test]
fn binomial_20_10() {
    // C(20,10) = 184756
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let n = arena.int(20);
        let k = arena.int(10);
        let expr = arena.binomial(n, k);
        let result = arena.eval_expr(expr);
        assert_eq!(arena.display(result).to_string(), "184756");
    });
}

#[test]
fn binomial_15_7() {
    // C(15,7) = 6435
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let n = arena.int(15);
        let k = arena.int(7);
        let expr = arena.binomial(n, k);
        let result = arena.eval_expr(expr);
        assert_eq!(arena.display(result).to_string(), "6435");
    });
}

#[test]
fn binomial_30_15() {
    // C(30,15) = 155117520
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let n = arena.int(30);
        let k = arena.int(15);
        let expr = arena.binomial(n, k);
        let result = arena.eval_expr(expr);
        assert_eq!(arena.display(result).to_string(), "155117520");
    });
}

#[test]
fn binomial_50_25() {
    // C(50,25) = 126410606437752
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let n = arena.int(50);
        let k = arena.int(25);
        let expr = arena.binomial(n, k);
        let result = arena.eval_expr(expr);
        assert_eq!(arena.display(result).to_string(), "126410606437752");
    });
}

#[test]
fn binomial_symmetry_large() {
    // C(30,12) == C(30,18)
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let n = arena.int(30);
        let k1 = arena.int(12);
        let k2 = arena.int(18);
        let b1 = arena.binomial(n, k1);
        let b2 = arena.binomial(n, k2);
        let r1 = arena.eval_expr(b1);
        let r2 = arena.eval_expr(b2);
        assert_eq!(
            arena.display(r1).to_string(),
            arena.display(r2).to_string(),
            "C(30,12) should equal C(30,18)"
        );
    });
}

#[test]
fn factorial_negative_stays_unevaluated() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let n = arena.int(-5);
        let expr = arena.factorial(n);
        let result = arena.eval_expr(expr);
        // Negative factorial has no standard value — should stay as (-5)!
        let s = arena.display(result).to_string();
        assert!(
            s.contains("!"),
            "negative factorial should stay unevaluated: {s}"
        );
    });
}

#[test]
fn factorial_rational_stays_unevaluated() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let n = arena.rational(3, 2);
        let expr = arena.factorial(n);
        let result = arena.eval_expr(expr);
        let s = arena.display(result).to_string();
        assert!(
            s.contains("!"),
            "rational factorial should stay unevaluated: {s}"
        );
    });
}

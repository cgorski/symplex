//! 0.2: derivative rules and arbitrary-precision evaluation for the
//! `Apply`-based special functions (Bessel J/Y/I/K, Legendre, Chebyshev,
//! Hermite, Laguerre), plus the removal of the `n ≤ 20` cutoff for exact
//! orthogonal-polynomial expansion.

use symplex::prelude::*;

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

fn rat(ctx: &Context, v: f64) -> Ex {
    ctx.rational((v * 1e9).round() as i64, 1_000_000_000)
}

/// Compare `f.diff(x)` against a central finite difference of `f` itself.
fn check_derivative(f: &Ex, x: &Ex, points: &[f64]) {
    let ctx = f.context();
    let d = f.diff(x);
    assert!(
        !d.has_unevaluated(),
        "derivative of {f} should be closed form, got {d}"
    );
    for &x0 in points {
        let h = 1e-4;
        let fp = f.subs(x, &rat(&ctx, x0 + h)).eval_f64().unwrap();
        let fm = f.subs(x, &rat(&ctx, x0 - h)).eval_f64().unwrap();
        let numeric = (fp - fm) / (2.0 * h);
        let symbolic = d.subs(x, &rat(&ctx, x0)).eval_f64().unwrap();
        assert!(
            approx(numeric, symbolic, 1e-5 * (1.0 + symbolic.abs())),
            "d/dx {f} at {x0}: finite diff {numeric} vs symbolic {symbolic} ({d})"
        );
    }
}

fn assert_prefix(e: &Ex, digits: u32, expected: &str) {
    let s = e.eval_decimal(digits).unwrap();
    assert!(
        s.starts_with(expected),
        "{e} → {s}, expected prefix {expected}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Derivatives
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bessel_derivatives_symbolic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let nu = ctx.symbol("nu");
    let half = ctx.rational(1, 2);
    let nm1 = &nu - 1;
    let np1 = &nu + 1;
    let d = x.bessel_j(&nu).diff(&x);
    assert_eq!(d, &half * &(&x.bessel_j(&nm1) - &x.bessel_j(&np1)));
    let d = x.bessel_y(&nu).diff(&x);
    assert_eq!(d, &half * &(&x.bessel_y(&nm1) - &x.bessel_y(&np1)));
    let d = x.bessel_i(&nu).diff(&x);
    assert_eq!(d, &half * &(&x.bessel_i(&nm1) + &x.bessel_i(&np1)));
    let d = x.bessel_k(&nu).diff(&x);
    assert_eq!(d, -&(&half * &(&x.bessel_k(&nm1) + &x.bessel_k(&np1))));
    // Order depending on the variable → formal derivative.
    let d = x.bessel_j(&x).diff(&x);
    assert!(d.has_unevaluated());
    // Argument constant → zero.
    let y = ctx.symbol("y");
    assert!(y.bessel_j(&nu).diff(&x).is_zero_structural());
}

#[test]
fn bessel_derivatives_numeric() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for order in [0i64, 1, 2, 3] {
        let n = ctx.int(order);
        check_derivative(&x.bessel_j(&n), &x, &[0.7, 2.5, 6.0]);
        check_derivative(&x.bessel_y(&n), &x, &[0.7, 2.5, 6.0]);
        check_derivative(&x.bessel_i(&n), &x, &[0.7, 2.5]);
        check_derivative(&x.bessel_k(&n), &x, &[0.7, 2.5]);
    }
    // Non-integer order for I and K
    let third = ctx.rational(1, 3);
    check_derivative(&x.bessel_i(&third), &x, &[0.8, 2.0]);
    check_derivative(&x.bessel_k(&third), &x, &[0.8, 2.0]);
    // Chain rule through a composite argument
    check_derivative(&x.powi(2).bessel_j(&ctx.int(1)), &x, &[0.9, 1.7]);
}

#[test]
fn orthogonal_polynomial_derivatives_symbolic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let n = ctx.symbol("n");
    assert_eq!(
        format!("{}", x.chebyshev_t(&n).diff(&x)),
        "n*chebyshev_u(n - 1, x)"
    );
    assert_eq!(
        format!("{}", x.hermite(&n).diff(&x)),
        "2*n*hermite(n - 1, x)"
    );
    let d = x.legendre(&n).diff(&x);
    assert!(format!("{d}").contains("legendre(n - 1, x)"), "{d}");
    let d = x.laguerre(&n).diff(&x);
    assert!(format!("{d}").contains("laguerre(n - 1, x)"), "{d}");
    let d = x.chebyshev_u(&n).diff(&x);
    assert!(format!("{d}").contains("chebyshev_t(n + 1, x)"), "{d}");
}

#[test]
fn orthogonal_polynomial_derivatives_numeric() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for deg in [2i64, 3, 5, 8] {
        let n = ctx.int(deg);
        check_derivative(&x.legendre(&n), &x, &[0.3, -0.6]);
        check_derivative(&x.chebyshev_t(&n), &x, &[0.3, -0.6]);
        check_derivative(&x.chebyshev_u(&n), &x, &[0.3, -0.6]);
        check_derivative(&x.hermite(&n), &x, &[0.3, 1.4]);
        check_derivative(&x.laguerre(&n), &x, &[0.3, 2.2]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// evalf: besseli / besselk
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bessel_i_reference_values() {
    let ctx = Context::new();
    let one = ctx.int(1);
    assert_prefix(
        &one.bessel_i(&ctx.int(0)),
        30,
        "1.26606587775200833559824462521",
    );
    assert_prefix(&one.bessel_i(&one), 30, "0.56515910399248502720769602760");
    assert_prefix(
        &ctx.int(2).bessel_i(&ctx.int(2)),
        20,
        "0.68894844769873820405",
    );
    assert_prefix(&ctx.int(30).bessel_i(&ctx.int(0)), 15, "781672297823.977");
    // non-integer order
    assert_prefix(
        &ctx.int(2).bessel_i(&ctx.rational(1, 3)),
        20,
        "2.1587825813728630239",
    );
    // I_{-n} = I_n; I_n(-x) = (-1)^n I_n(x)
    assert_eq!(
        ctx.int(2).bessel_i(&ctx.int(-3)).eval_decimal(20).unwrap(),
        ctx.int(2).bessel_i(&ctx.int(3)).eval_decimal(20).unwrap()
    );
    let v_neg = ctx.int(-2).bessel_i(&ctx.int(1)).eval_f64().unwrap();
    let v_pos = ctx.int(2).bessel_i(&ctx.int(1)).eval_f64().unwrap();
    assert!(approx(v_neg, -v_pos, 1e-14));
}

#[test]
fn bessel_k_reference_values() {
    let ctx = Context::new();
    let one = ctx.int(1);
    assert_prefix(
        &one.bessel_k(&ctx.int(0)),
        30,
        "0.42102443824070833333562737921",
    );
    assert_prefix(&one.bessel_k(&one), 30, "0.60190723019723457473754000153");
    assert_prefix(
        &ctx.int(3).bessel_k(&ctx.int(2)),
        20,
        "0.061510458471742037656",
    );
    // Large argument (asymptotic regime)
    assert_prefix(&ctx.int(50).bessel_k(&ctx.int(0)), 15, "3.41016774978949");
    // Half-integer order has a closed form: K_{1/2}(x) = √(π/(2x)) e^{-x}
    let k_half = one.bessel_k(&ctx.rational(1, 2)).eval_f64().unwrap();
    let expected = (std::f64::consts::PI / 2.0).sqrt() * (-1.0f64).exp();
    assert!(approx(k_half, expected, 1e-14), "{k_half} vs {expected}");
    assert_prefix(
        &ctx.int(2).bessel_k(&ctx.rational(1, 3)),
        20,
        "0.11654496129616524875",
    );
    // Undefined for x <= 0
    assert!(ctx.int(0).bessel_k(&ctx.int(0)).eval_f64().is_err());
    assert!(ctx.int(-1).bessel_k(&ctx.int(0)).eval_f64().is_err());
}

#[test]
fn bessel_y_small_argument_all_orders() {
    // Regression: Y_n for n >= 1 at small x used a Wronskian with the wrong
    // sign (and divided by J_0).  Now computed via the Neumann series.
    let ctx = Context::new();
    let one = ctx.int(1);
    assert_prefix(&one.bessel_y(&ctx.int(0)), 20, "0.08825696421567695798");
    assert_prefix(&one.bessel_y(&one), 20, "-0.7812128213002887165");
    assert_prefix(
        &ctx.int(3).bessel_y(&ctx.int(2)),
        20,
        "-0.1604003934849237296",
    );
    assert_prefix(
        &ctx.rational(1, 2).bessel_y(&ctx.int(3)),
        15,
        "-42.059494304723",
    );
    // Y_{-1}(1) = -Y_1(1)
    let y_m1 = one.bessel_y(&ctx.int(-1)).eval_f64().unwrap();
    assert!(approx(y_m1, 0.781_212_821_300_288_7, 1e-14));
    // Near the first zero of J_0 (x ≈ 2.4048) the old formula blew up.
    // Check the Wronskian J_1 Y_0 − J_0 Y_1 = 2/(πx) there and elsewhere.
    // (Points have at most 9 decimals so `rat` represents them exactly.)
    for xv in [0.3f64, 1.0, 2.4048, 2.404825558, 5.0] {
        let x = rat(&ctx, xv);
        let j0 = x.bessel_j(&ctx.int(0)).eval_f64().unwrap();
        let j1 = x.bessel_j(&one).eval_f64().unwrap();
        let y0 = x.bessel_y(&ctx.int(0)).eval_f64().unwrap();
        let y1 = x.bessel_y(&one).eval_f64().unwrap();
        let w = j1 * y0 - j0 * y1;
        let expected = 2.0 / (std::f64::consts::PI * xv);
        assert!(
            approx(w, expected, 1e-12),
            "Wronskian at {xv}: {w} vs {expected}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// evalf: orthogonal polynomials for any n
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn orthogonal_polynomials_numeric_any_degree() {
    let ctx = Context::new();
    let third = ctx.rational(1, 3);
    // P_5(1/3) = 1/3 exactly (63x⁵ − 70x³ + 15x)/8
    assert_prefix(
        &third.legendre(&ctx.int(5)),
        25,
        "0.333333333333333333333333",
    );
    // T_7(1/3) = cos(7 acos(1/3))
    let t7 = third.chebyshev_t(&ctx.int(7)).eval_f64().unwrap();
    assert!(approx(t7, (7.0 * (1.0f64 / 3.0).acos()).cos(), 1e-13));
    // U_7(1/3) = sin(8θ)/sin(θ)
    let u7 = third.chebyshev_u(&ctx.int(7)).eval_f64().unwrap();
    let theta = (1.0f64 / 3.0).acos();
    assert!(approx(u7, (8.0 * theta).sin() / theta.sin(), 1e-12));
    // H_5(1/3) = 32x⁵ − 160x³ + 120x
    let h5 = third.hermite(&ctx.int(5)).eval_f64().unwrap();
    let x = 1.0f64 / 3.0;
    assert!(approx(
        h5,
        32.0 * x.powi(5) - 160.0 * x.powi(3) + 120.0 * x,
        1e-12
    ));
    // L_5(1/3)
    let l5 = third.laguerre(&ctx.int(5)).eval_f64().unwrap();
    let expected = (-x.powi(5) + 25.0 * x.powi(4) - 200.0 * x.powi(3) + 600.0 * x * x - 600.0 * x
        + 120.0)
        / 120.0;
    assert!(approx(l5, expected, 1e-12));
    // Degrees far beyond the old cutoff of 20: compare Chebyshev against
    // its trigonometric closed form.
    for n in [21i64, 50, 200, 1000] {
        let tn = third.chebyshev_t(&ctx.int(n)).eval_f64().unwrap();
        assert!(
            approx(tn, (n as f64 * theta).cos(), 1e-10),
            "T_{n}(1/3) = {tn}"
        );
    }
    // P_n(1) = 1 for all n
    for n in [30i64, 100] {
        assert_prefix(&ctx.int(1).legendre(&ctx.int(n)), 10, "1");
    }
    // P_30(1/3)
    assert_prefix(&third.legendre(&ctx.int(30)), 20, "0.087544975147035257678");
}

#[test]
fn exact_orthogonal_polynomials_beyond_degree_20() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Exact expansion used to stop at n = 20.
    let t25 = x.chebyshev_t(&ctx.int(25)).eval();
    assert!(!format!("{t25}").contains("chebyshev_t"), "{t25}");
    // Leading coefficient of T_25 is 2^24.
    assert_eq!(t25.degree(&x), Some(25));
    let lc = t25.coeff(&x, 25).unwrap();
    assert_eq!(format!("{lc}"), format!("{}", 1u64 << 24));
    // The expansion agrees numerically with the recurrence-based evalf.
    let v_exact = t25.subs(&x, &ctx.rational(1, 3)).eval_f64().unwrap();
    let v_num = ctx
        .rational(1, 3)
        .chebyshev_t(&ctx.int(25))
        .eval_f64()
        .unwrap();
    assert!(approx(v_exact, v_num, 1e-10));
    let p22 = x.legendre(&ctx.int(22)).eval();
    assert_eq!(p22.degree(&x), Some(22));
}

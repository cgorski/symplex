// Real-world workflow integration tests for symplex.
//
// Each test represents a multi-step pipeline that an actual user would run,
// chaining 3–7 operations together.  Every test contains strong numerical
// assertions — not just "doesn't panic" checks.

use super::common;

use symplex::eq::Equation;
use symplex::matrix::{Matrix, jacobian};
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// 1. Calculus optimisation: diff → solve → classify via second derivative
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn workflow_calculus_optimization() {
    let ctx = Context::new();
    // f(x) = x³ - 6x² + 9x + 1
    let x = ctx.symbol("x");
    let f = &x.powi(3) - &(&x.powi(2) * 6) + &(&x * 9) + 1;

    // Step 1: first derivative
    let fp = f.diff(&x);

    // Step 2: solve f'(x) = 0  →  critical points
    let crits = fp.solve(&x).expect("should solve f'(x)=0");
    assert_eq!(crits.len(), 2, "f'(x) = 3x²-12x+9 has 2 roots");

    let mut crit_vals: Vec<f64> = crits
        .iter()
        .map(|r| r.eval_f64().expect("critical point evaluable"))
        .collect();
    crit_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!(
        (crit_vals[0] - 1.0).abs() < 1e-9,
        "first critical point should be 1, got {}",
        crit_vals[0]
    );
    assert!(
        (crit_vals[1] - 3.0).abs() < 1e-9,
        "second critical point should be 3, got {}",
        crit_vals[1]
    );

    // Step 3: second derivative for classification
    let fpp = fp.diff(&x);

    // f''(1) = 6(1)-12 = -6 < 0  → local max
    let fpp_at1 = fpp.subs_i64(&x, 1).eval().eval_f64().expect("f''(1)");
    assert!(
        fpp_at1 < 0.0,
        "f''(1) should be negative (local max), got {fpp_at1}"
    );

    // f''(3) = 6(3)-12 = 6 > 0  → local min
    let fpp_at3 = fpp.subs_i64(&x, 3).eval().eval_f64().expect("f''(3)");
    assert!(
        fpp_at3 > 0.0,
        "f''(3) should be positive (local min), got {fpp_at3}"
    );

    // Step 4: verify function values  f(1)=5, f(3)=1
    let f1 = f.subs_i64(&x, 1).eval().eval_f64().expect("f(1)");
    assert!((f1 - 5.0).abs() < 1e-9, "f(1) should be 5, got {f1}");
    let f3 = f.subs_i64(&x, 3).eval().eval_f64().expect("f(3)");
    assert!((f3 - 1.0).abs() < 1e-9, "f(3) should be 1, got {f3}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Differentiate → integrate round-trip
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn workflow_diff_integrate_roundtrip() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // f(x) = x³ + 2x² - x + 3
    let f = &x.powi(3) + &(&x.powi(2) * 2) - &x + 3;

    // differentiate
    let fp = f.diff(&x);
    // integrate back
    let g = fp.integrate(&x);

    // g should equal f up to a constant.  Check at several points that
    // g(x) - f(x) is the same constant.
    let pts: &[i64] = &[-3, -1, 0, 2, 5];
    let diffs: Vec<f64> = pts
        .iter()
        .map(|&p| {
            let gv = common::eval_at_i64(&g, &x, p);
            let fv = common::eval_at_i64(&f, &x, p);
            gv - fv
        })
        .collect();

    let c = diffs[0];
    for (i, &d) in diffs.iter().enumerate() {
        assert!(
            (d - c).abs() < 1e-9,
            "g(x)-f(x) should be constant; at x={} diff={}, expected {}",
            pts[i],
            d,
            c
        );
    }
}

#[test]
fn workflow_diff_integrate_roundtrip_trig() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sin();
    let fp = f.diff(&x); // cos(x)
    let g = fp.integrate(&x); // should give sin(x) (+ C)

    // Numerically: g(x) - f(x) should be constant
    let pts: &[(i64, i64)] = &[(1, 10), (3, 10), (7, 10), (11, 10)];
    let mut diffs = Vec::new();
    for &(p, q) in pts {
        let pt = ctx.rational(p, q);
        let gv = g.subs(&x, &pt).eval().eval_f64().expect("g eval");
        let fv = f.subs(&x, &pt).eval().eval_f64().expect("f eval");
        diffs.push(gv - fv);
    }
    let c = diffs[0];
    for (i, &d) in diffs.iter().enumerate() {
        assert!(
            (d - c).abs() < 1e-9,
            "sin round-trip: diff at pt {} = {}, expected const {}",
            i,
            d,
            c
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Partial fractions pipeline
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn workflow_partial_fractions_pipeline() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 1 / (x² + 3x + 2) = 1/((x+1)(x+2))
    let original = &ctx.int(1) / &(&x.powi(2) + &(&x * 3) + 2);

    // Step 1: partial fraction decomposition
    let decomposed = original.partial_fractions(&x);

    // Step 2: verify numerically that decomposed == original at several points
    for &pt in &[3i64, 5, 7] {
        let ov = common::eval_at_i64(&original, &x, pt);
        let dv = common::eval_at_i64(&decomposed, &x, pt);
        assert!(
            common::approx_eq(ov, dv, 1e-12),
            "apart mismatch at x={pt}: original={ov}, decomposed={dv}"
        );
    }

    // Step 3: recombine with together
    let recombined = decomposed.together();
    for &pt in &[3i64, 5, 7] {
        let ov = common::eval_at_i64(&original, &x, pt);
        let rv = common::eval_at_i64(&recombined, &x, pt);
        assert!(
            common::approx_eq(ov, rv, 1e-12),
            "together mismatch at x={pt}: original={ov}, recombined={rv}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Taylor / Maclaurin convergence
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn workflow_taylor_convergence() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sin_x = x.sin();

    // Maclaurin expansion to order 7:  x - x³/6 + x⁵/120 - x⁷/5040
    let series = sin_x.maclaurin(&x, 7);
    let series_expanded = series.expand().eval();

    for &(num, denom, expected_sin) in &[
        (1i64, 10i64, 0.1_f64.sin()),
        (3, 10, 0.3_f64.sin()),
        (5, 10, 0.5_f64.sin()),
    ] {
        let pt = ctx.rational(num, denom);
        let sv = series_expanded
            .subs(&x, &pt)
            .eval()
            .eval_f64()
            .expect("series eval");
        let x_val = num as f64 / denom as f64;
        assert!(
            (sv - expected_sin).abs() < 1e-4,
            "Maclaurin sin({x_val}) ≈ {sv}, actual sin = {expected_sin}"
        );
    }
}

#[test]
fn workflow_taylor_exp_convergence() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let exp_x = x.exp();

    let series = exp_x.maclaurin(&x, 8);
    let series_expanded = series.expand().eval();

    for &(num, denom) in &[(1i64, 10i64), (5, 10), (1, 1)] {
        let pt = ctx.rational(num, denom);
        let sv = series_expanded
            .subs(&x, &pt)
            .eval()
            .eval_f64()
            .expect("series eval");
        let x_val = num as f64 / denom as f64;
        let expected = x_val.exp();
        assert!(
            (sv - expected).abs() < 1e-4,
            "Maclaurin exp({x_val}) ≈ {sv}, actual = {expected}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Solve → verify by substitution
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn workflow_solve_verify_substitute() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x³ - 6x² + 11x - 6 = 0  roots are 1, 2, 3
    let poly = &x.powi(3) - &(&x.powi(2) * 6) + &(&x * 11) - 6;

    let roots = poly.solve(&x).expect("should solve cubic");
    assert_eq!(roots.len(), 3, "cubic should have 3 roots");

    // Verify each root numerically
    common::verify_roots(&poly, &x, &roots, 1e-9);

    // Also verify expected values
    let mut root_vals: Vec<f64> = roots
        .iter()
        .map(|r| r.eval_f64().expect("root to f64"))
        .collect();
    root_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((root_vals[0] - 1.0).abs() < 1e-9, "root 1");
    assert!((root_vals[1] - 2.0).abs() < 1e-9, "root 2");
    assert!((root_vals[2] - 3.0).abs() < 1e-9, "root 3");
}

#[test]
fn workflow_solve_verify_substitute_quadratic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x² - 5x + 6 = 0  roots are 2, 3
    let poly = &x.powi(2) - &(&x * 5) + 6;
    let roots = poly.solve(&x).expect("should solve quadratic");
    assert_eq!(roots.len(), 2);
    common::verify_roots(&poly, &x, &roots, 1e-9);
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Matrix eigenvalue properties
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn workflow_matrix_eigenvalue_properties() {
    let ctx = Context::new();

    // [[2, 1], [1, 2]]
    let m = Matrix::new(vec![
        vec![ctx.int(2), ctx.int(1)],
        vec![ctx.int(1), ctx.int(2)],
    ])
    .unwrap();

    // Step 1: trace and determinant
    let tr = m.trace().unwrap();
    let tr_val = tr.eval_f64().expect("trace eval");
    assert!(
        (tr_val - 4.0).abs() < 1e-9,
        "trace should be 4, got {tr_val}"
    );

    let det = m.det().unwrap();
    let det_val = det.eval_f64().expect("det eval");
    assert!(
        (det_val - 3.0).abs() < 1e-9,
        "det should be 3, got {det_val}"
    );

    // Step 2: eigenvalues
    let eigs = m.eigenvals().unwrap();
    assert_eq!(eigs.len(), 2, "2x2 matrix should have 2 eigenvalues");

    let mut eig_vals: Vec<f64> = eigs
        .iter()
        .map(|e| e.eval_f64().expect("eigenvalue eval"))
        .collect();
    eig_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // eigenvalues of [[2,1],[1,2]] are 1 and 3
    assert!(
        (eig_vals[0] - 1.0).abs() < 1e-9,
        "λ₁ should be 1, got {}",
        eig_vals[0]
    );
    assert!(
        (eig_vals[1] - 3.0).abs() < 1e-9,
        "λ₂ should be 3, got {}",
        eig_vals[1]
    );

    // Step 3: verify sum(eigenvalues) = trace
    let eig_sum = eig_vals[0] + eig_vals[1];
    assert!(
        (eig_sum - tr_val).abs() < 1e-9,
        "sum of eigenvalues ({eig_sum}) should equal trace ({tr_val})"
    );

    // Step 4: verify product(eigenvalues) = determinant
    let eig_prod = eig_vals[0] * eig_vals[1];
    assert!(
        (eig_prod - det_val).abs() < 1e-9,
        "product of eigenvalues ({eig_prod}) should equal determinant ({det_val})"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Jacobian → lambdify → finite-difference check
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn workflow_jacobian_to_lambdify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    // Functions: [x²y, xy²]
    let f1 = &x.powi(2) * &y;
    let f2 = &x * &y.powi(2);

    // Step 1: compute Jacobian
    let jac = jacobian(&[&f1, &f2], &[&x, &y]);
    assert_eq!(jac.shape(), (2, 2));

    // J = [[2xy, x²], [y², 2xy]]
    // Step 2: lambdify J[0][0] = 2xy
    let j00 = jac.get(0, 0);
    let j00_fn = j00.compile(&["x", "y"]).expect("lambdify J[0][0]");

    // Step 3: evaluate at (x, y) = (3, 4)
    let val = j00_fn(&[3.0, 4.0]);
    // 2 * 3 * 4 = 24
    assert!(
        (val - 24.0).abs() < 1e-9,
        "J[0][0](3,4) = 2*3*4 = 24, got {val}"
    );

    // Step 4: finite-difference approximation for ∂f1/∂x at (3,4)
    // f1(x,y) = x²y  →  ∂f1/∂x ≈ (f1(3+h,4) - f1(3-h,4)) / (2h)
    let h = 1e-6;
    let fd = ((3.0_f64 + h).powi(2) * 4.0 - (3.0_f64 - h).powi(2) * 4.0) / (2.0 * h);
    assert!(
        (val - fd).abs() < 1e-4,
        "J[0][0] vs finite diff: {val} vs {fd}"
    );

    // Step 5: also check J[1][1] = 2xy at (3,4) = 24
    let j11 = jac.get(1, 1);
    let j11_fn = j11.compile(&["x", "y"]).expect("lambdify J[1][1]");
    let val11 = j11_fn(&[3.0, 4.0]);
    assert!(
        (val11 - 24.0).abs() < 1e-9,
        "J[1][1](3,4) should be 24, got {val11}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. Trig simplify chain
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn workflow_trig_simplify_chain() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // (sin(x) + cos(x))²
    let expr = (&x.sin() + &x.cos()).powi(2);

    // Step 1: expand → sin²(x) + 2sin(x)cos(x) + cos²(x)
    let expanded = expr.expand();

    // Step 2: simplify / trigsimp
    let simplified = expanded.simplify_trig();

    // Step 3: numerical verification that result ≈ 1 + sin(2x) at several points
    // (sin(x)+cos(x))² = sin²+2sincos+cos² = 1 + 2sin(x)cos(x) = 1 + sin(2x)
    for &(num, denom) in &[(1i64, 10i64), (5, 10), (8, 10), (12, 10)] {
        let pt = ctx.rational(num, denom);
        let x_val = num as f64 / denom as f64;

        let orig_val = expr.subs(&x, &pt).eval().eval_f64().expect("orig eval");
        let simp_val = simplified
            .subs(&x, &pt)
            .eval()
            .eval_f64()
            .expect("simp eval");
        let expected = 1.0 + (2.0 * x_val).sin();

        assert!(
            common::approx_eq(orig_val, expected, 1e-9),
            "original at x={x_val}: got {orig_val}, expected {expected}"
        );
        assert!(
            common::approx_eq(simp_val, expected, 1e-9),
            "simplified at x={x_val}: got {simp_val}, expected {expected}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. Complex / Euler
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn workflow_complex_euler() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let pi = ctx.pi();

    // exp(i*π) should evaluate to -1
    let expr = (&i * &pi).exp().eval();
    let Complex64 { re, im } = expr.eval_complex64().expect("exp(iπ) complex eval");
    assert!(
        (re - (-1.0)).abs() < 1e-9,
        "Re(exp(iπ)) should be -1, got {re}"
    );
    assert!(im.abs() < 1e-9, "Im(exp(iπ)) should be 0, got {im}");

    // exp(i*π) + 1 should be 0  (Euler's identity)
    let euler = &(&i * &pi).exp() + 1;
    let euler_evaled = euler.eval();
    let Complex64 { re: re2, im: im2 } =
        euler_evaled.eval_complex64().expect("euler identity eval");
    let mag = (re2 * re2 + im2 * im2).sqrt();
    assert!(
        mag < 1e-9,
        "exp(iπ)+1 should be 0, got ({re2}, {im2}), |z|={mag}"
    );
}

#[test]
fn workflow_complex_euler_i_squared() {
    let ctx = Context::new();
    // i² = -1
    let i = ctx.i_unit();
    let i_sq = i.powi(2);
    assert_eq!(format!("{i_sq}"), "-1");

    // |i| = 1
    let Complex64 { re, im } = i.eval_complex64().expect("i eval");
    let mag = (re * re + im * im).sqrt();
    assert!((mag - 1.0).abs() < 1e-9, "|i| should be 1, got {mag}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. Polynomial algebra: cancel
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn workflow_polynomial_algebra_chain() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // (x² - 1) / (x - 1)  should cancel to  x + 1
    let expr = &(&x.powi(2) - 1) / &(&x - 1);
    let cancelled = expr.cancel(&x);

    // Display check
    let s = format!("{cancelled}");
    assert_eq!(s, "x + 1", "cancelled form should be x + 1, got: {s}");

    // Numerical verification at several points (avoid x=1, pole of original)
    for &pt in &[2i64, 3, 5, -2, -3] {
        let ov = common::eval_at_i64(&expr, &x, pt);
        let cv = common::eval_at_i64(&cancelled, &x, pt);
        let expected = pt as f64 + 1.0;
        assert!(
            common::approx_eq(ov, expected, 1e-9),
            "original at x={pt}: {ov} vs {expected}"
        );
        assert!(
            common::approx_eq(cv, expected, 1e-9),
            "cancelled at x={pt}: {cv} vs {expected}"
        );
    }
}

#[test]
fn workflow_polynomial_algebra_factor_expand() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // Start with x² - 4, factor, then expand back
    let poly = &x.powi(2) - 4;
    let factored = poly.factor(&x);
    let re_expanded = factored.expand();

    // Numerically should be the same
    common::assert_math_eq(&poly, &re_expanded, &x, "factor then expand x²-4");
}

// ═══════════════════════════════════════════════════════════════════════════
// 11. Definite integral verification
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn workflow_definite_integral_verification() {
    let ctx = Context::new();
    let ctx = ctx.clone();
    let x = ctx.symbol("x");

    // ∫₀¹ x² dx = 1/3
    let result1 = x.powi(2).integrate_definite(&x, &ctx.int(0), &ctx.int(1));
    let val1 = result1.eval().eval_f64().expect("∫₀¹ x² dx");
    assert!(
        (val1 - 1.0 / 3.0).abs() < 1e-9,
        "∫₀¹ x² dx should be 1/3, got {val1}"
    );

    // ∫₀^π sin(x) dx = 2
    let result2 = x.sin().integrate_definite(&x, &ctx.int(0), &ctx.pi());
    let val2 = result2.eval().eval_f64().expect("∫₀^π sin(x) dx");
    assert!(
        (val2 - 2.0).abs() < 1e-9,
        "∫₀^π sin(x) dx should be 2, got {val2}"
    );
}

#[test]
fn workflow_definite_integral_polynomial() {
    let ctx = Context::new();
    let ctx = ctx.clone();
    let x = ctx.symbol("x");

    // ∫₁² (x² + x) dx = [x³/3 + x²/2]₁² = (8/3 + 2) - (1/3 + 1/2) = 23/6
    let expr = &x.powi(2) + &x;
    let result = expr.integrate_definite(&x, &ctx.int(1), &ctx.int(2));
    let val = result.eval().eval_f64().expect("∫₁² (x²+x) dx");
    assert!(
        (val - 23.0 / 6.0).abs() < 1e-9,
        "∫₁² (x²+x) dx should be 23/6 ≈ 3.833, got {val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 12. Substitution chain
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn workflow_substitution_chain() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);

    // expr = x² + 2*y + 1
    let expr = &x.powi(2) + &(&y * 2) + 1;

    // Step 1: subs x → y²
    let step1 = expr.subs(&x, &y.powi(2));
    // Now expr = (y²)² + 2y + 1 = y⁴ + 2y + 1

    // Step 2: subs y → 3
    let step2 = step1.subs_i64(&y, 3).eval();
    // y⁴ + 2y + 1 = 81 + 6 + 1 = 88
    let val = step2.eval_f64().expect("final substitution");
    assert!(
        (val - 88.0).abs() < 1e-9,
        "x²+2y+1 with x→y², y→3 should be 88, got {val}"
    );
}

#[test]
fn workflow_substitution_chain_trig() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);

    // expr = sin(x) + y
    let expr = &x.sin() + &y;

    // subs x → π/2
    let pi_half = &ctx.pi() / &ctx.int(2);
    let step1 = expr.subs(&x, &pi_half);
    // sin(π/2) + y = 1 + y

    // subs y → 4
    let step2 = step1.subs_i64(&y, 4).eval();
    let val = step2.eval_f64().expect("sin(π/2)+4 eval");
    assert!(
        (val - 5.0).abs() < 1e-9,
        "sin(π/2) + 4 should be 5, got {val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 13. Lambdify vs evalf — bulk comparison
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn workflow_lambdify_vs_evalf() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // f(x) = x³ + sin(x)
    let f = &x.powi(3) + &x.sin();

    let func = f.compile(&["x"]).expect("lambdify x³+sin(x)");

    // 20 points from -3 to 3
    let n = 20;
    for i in 0..n {
        let x_val = -3.0 + 6.0 * (i as f64) / ((n - 1) as f64);
        // lambdify result
        let lam_val = func(&[x_val]);

        // evalf result — substitute a rational approximation
        let numer = (x_val * 10000.0).round() as i64;
        let pt = ctx.rational(numer, 10000);
        let evalf_val = f.subs(&x, &pt).eval().eval_f64().expect("evalf");

        assert!(
            common::approx_eq(lam_val, evalf_val, 1e-3),
            "lambdify vs evalf at x={x_val}: lambdify={lam_val}, evalf={evalf_val}"
        );
    }
}

#[test]
fn workflow_lambdify_vs_evalf_multivar() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let f = &x.powi(2) + &y.powi(2);
    let func = f.compile(&["x", "y"]).expect("lambdify x²+y²");

    for &(xv, yv) in &[(1.0, 2.0), (3.0, 4.0), (-1.0, 0.5), (0.0, 0.0)] {
        let lam_val = func(&[xv, yv]);
        let expected = xv * xv + yv * yv;
        assert!(
            (lam_val - expected).abs() < 1e-9,
            "x²+y² at ({xv},{yv}): got {lam_val}, expected {expected}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 14. Equation solve & check
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn workflow_equation_solve_check() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x² - 5x + 6 = 0
    let equation = eq!(ctx, x ^ 2 - 5 * x + 6 = 0);
    let roots = equation.solve(&x).expect("should solve x²-5x+6=0");
    assert_eq!(roots.len(), 2, "quadratic should have 2 roots");

    // Check each root satisfies the polynomial
    let poly = equation.to_expr();
    for root in &roots {
        assert!(
            poly.check_solution(&x, root).unwrap_or(false),
            "root {root} should satisfy x²-5x+6=0"
        );
    }

    // Also verify specific values
    let mut vals: Vec<f64> = roots
        .iter()
        .map(|r| r.eval_f64().expect("root eval"))
        .collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!(
        (vals[0] - 2.0).abs() < 1e-9,
        "root 1 should be 2, got {}",
        vals[0]
    );
    assert!(
        (vals[1] - 3.0).abs() < 1e-9,
        "root 2 should be 3, got {}",
        vals[1]
    );
}

#[test]
fn workflow_equation_solve_check_cubic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x³ - x = 0  →  x(x²-1) = 0  →  roots 0, 1, -1
    let equation = Equation::new(&x.powi(3) - &x, ctx.int(0));
    let roots = equation.solve(&x).expect("should solve x³-x=0");
    assert_eq!(roots.len(), 3, "should have 3 roots");

    let poly = equation.to_expr();
    for root in &roots {
        assert!(
            poly.check_solution(&x, root).unwrap_or(false),
            "root {root} should satisfy x³-x=0"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 15. Series then integrate (FTC verification)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn workflow_series_then_integrate() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // Taylor expand exp(x) to order 4:  1 + x + x²/2 + x³/6
    let series = x.exp().maclaurin(&x, 4);
    let poly = series.expand().eval();

    // Integrate the polynomial
    let anti = poly.integrate(&x);
    // Differentiate back
    let back = anti.diff(&x);

    // FTC: d/dx(∫ poly dx) should numerically equal poly
    for &(num, denom) in &[(3i64, 10i64), (7, 10), (14, 10)] {
        let pt = ctx.rational(num, denom);
        let pv = poly.subs(&x, &pt).eval().eval_f64().expect("poly eval");
        let bv = back.subs(&x, &pt).eval().eval_f64().expect("back eval");
        assert!(
            common::approx_eq(pv, bv, 1e-9),
            "FTC: poly at {num}/{denom}={pv}, d/dx(∫poly)={bv}"
        );
    }

    // Also verify the definite integral over [0,1] against numerical
    // expectation.  The polynomial is 1 + x + x²/2 + x³/6, so
    // ∫₀¹ (1 + x + x²/2 + x³/6) dx = 1 + 1/2 + 1/6 + 1/24 = 41/24 ≈ 1.70833
    let def_int = poly.integrate_definite(&x, &ctx.int(0), &ctx.int(1));
    let def_val = def_int.eval().eval_f64().expect("definite integral");
    let expected = 1.0 + 0.5 + 1.0 / 6.0 + 1.0 / 24.0;
    assert!(
        (def_val - expected).abs() < 1e-9,
        "∫₀¹ series dx = {def_val}, expected {expected}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Additional workflow tests
// ═══════════════════════════════════════════════════════════════════════════

/// Build a polynomial, factor, then solve — all three should agree on roots.
#[test]
fn workflow_factor_then_solve() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let poly = &x.powi(2) - &(&x * 5) + 6; // (x-2)(x-3)

    // Factor
    let factored = poly.factor(&x);
    // Solve
    let roots = poly.solve(&x).expect("solve x²-5x+6");
    assert_eq!(roots.len(), 2);

    // Verify factored form evaluates the same as original
    common::assert_math_eq(&poly, &factored, &x, "factor preserves value");

    // Verify roots
    common::verify_roots(&poly, &x, &roots, 1e-9);
}

/// Differentiation of a product using the product rule, verified numerically.
#[test]
fn workflow_product_rule_verification() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sin();
    let g = x.exp();

    // d/dx(f*g) by symplex
    let fg = &f * &g;
    let dfg = fg.diff(&x);

    // product rule: f'g + fg'
    let fp = f.diff(&x);
    let gp = g.diff(&x);
    let manual = &(&fp * &g) + &(&f * &gp);

    // They should be numerically equal
    for &(num, denom) in &[(1i64, 10i64), (5, 10), (10, 10), (15, 10)] {
        let pt = ctx.rational(num, denom);
        let auto_val = dfg.subs(&x, &pt).eval().eval_f64().expect("auto diff");
        let man_val = manual.subs(&x, &pt).eval().eval_f64().expect("manual diff");
        assert!(
            common::approx_eq(auto_val, man_val, 1e-9),
            "product rule at {num}/{denom}: auto={auto_val}, manual={man_val}"
        );
    }
}

/// Chain rule: d/dx sin(x²) = 2x cos(x²), verified numerically.
#[test]
fn workflow_chain_rule_verification() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(2).sin(); // sin(x²)
    let deriv = expr.diff(&x);

    // Expected: 2x cos(x²)
    let expected = &(&x * 2) * &x.powi(2).cos();

    for &(num, denom) in &[(3i64, 10i64), (7, 10), (12, 10)] {
        let pt = ctx.rational(num, denom);
        let dv = deriv.subs(&x, &pt).eval().eval_f64().expect("deriv eval");
        let ev = expected
            .subs(&x, &pt)
            .eval()
            .eval_f64()
            .expect("expected eval");
        assert!(
            common::approx_eq(dv, ev, 1e-9),
            "chain rule at {num}/{denom}: computed={dv}, expected={ev}"
        );
    }
}

/// Power-of-a-sum expansion and simplification round-trip.
#[test]
fn workflow_expand_simplify_roundtrip() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // (x + 1)³
    let expr = (&x + 1).powi(3);

    // Expand: x³ + 3x² + 3x + 1
    let expanded = expr.expand();

    // Verify expansion is numerically correct
    common::assert_math_eq(&expr, &expanded, &x, "expand (x+1)³");

    // Simplify the expanded form (should stay the same or reduce)
    let simplified = expanded.simplify();
    common::assert_math_eq(&expr, &simplified, &x, "simplify expanded (x+1)³");
}

/// Matrix determinant and inverse verification for 3×3.
#[test]
fn workflow_matrix_inverse_verify() {
    let ctx = Context::new();
    // [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
    let m = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2), ctx.int(3)],
        vec![ctx.int(0), ctx.int(1), ctx.int(4)],
        vec![ctx.int(5), ctx.int(6), ctx.int(0)],
    ])
    .unwrap();

    let det = m.det().unwrap();
    let det_val = det.eval_f64().expect("det eval");
    // det = 1(0-24) - 2(0-20) + 3(0-5) = -24 + 40 - 15 = 1
    assert!(
        (det_val - 1.0).abs() < 1e-9,
        "det should be 1, got {det_val}"
    );

    // Invert and verify M * M⁻¹ = I
    let inv = m.inv().expect("should be invertible (det=1)");
    let product = m.matmul(&inv).unwrap();

    // Check diagonal entries are 1, off-diagonal are 0
    for i in 0..3 {
        for j in 0..3 {
            let v = product.get(i, j).eval().eval_f64().expect("product entry");
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (v - expected).abs() < 1e-8,
                "M*M⁻¹ at ({i},{j}) should be {expected}, got {v}"
            );
        }
    }
}

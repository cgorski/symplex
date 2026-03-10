//! Comprehensive integration tests for the Poly → GenPoly<Ratio<BigInt>> refactor.
//!
//! These tests verify that the type alias `type Poly = GenPoly<Ratio<BigInt>>`
//! preserves all behavior of the former standalone `struct Poly` by exercising
//! every downstream code path through the public API.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helper
// ═══════════════════════════════════════════════════════════════════════════

fn ctx() -> Context {
    Context::new()
}

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol || (a.is_nan() && b.is_nan())
}

// ═══════════════════════════════════════════════════════════════════════════
// Polynomial arithmetic via the public API (exercises polybridge → Poly)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expand_polynomial_product() {
    let c = ctx();
    let x = c.symbol("x");
    // (x+1)(x-1) = x²-1
    let expr = (&x + 1) * (&x - 1);
    let expanded = expr.expand();
    let s = format!("{expanded}");
    assert!(s.contains("x^2"), "should have x^2 term, got: {s}");
}

#[test]
fn expand_cube() {
    let c = ctx();
    let x = c.symbol("x");
    // (x+1)^3 = x³ + 3x² + 3x + 1
    let expr = (&x + 1).powi(3);
    let expanded = expr.expand();
    // Verify numerically at x=2: (3)^3 = 27
    let val = expanded.subs_i64(&x, 2).eval_f64().unwrap();
    assert!(approx(val, 27.0, 1e-10), "expected 27, got {val}");
}

#[test]
fn expand_preserves_value_at_multiple_points() {
    let c = ctx();
    let x = c.symbol("x");
    let expr = (&x + 2) * (&x - 3) * (&x + 5);
    let expanded = expr.expand();
    for pt in [-5, -2, 0, 1, 3, 7] {
        let v1 = expr.subs_i64(&x, pt).eval_f64().unwrap();
        let v2 = expanded.subs_i64(&x, pt).eval_f64().unwrap();
        assert!(
            approx(v1, v2, 1e-10),
            "value mismatch at x={pt}: {v1} vs {v2}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Polynomial solving (exercises solve.rs → Poly)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_linear() {
    let c = ctx();
    let x = c.symbol("x");
    // 2x + 6 = 0  →  x = -3
    let expr = &x * 2 + 6;
    let roots = expr.solve(&x).unwrap();
    assert_eq!(roots.len(), 1, "linear should have 1 root: {roots:?}");
    let val = roots[0].eval_f64().unwrap();
    assert!(approx(val, -3.0, 1e-10), "expected -3, got {val}");
}

#[test]
fn solve_quadratic_two_real_roots() {
    let c = ctx();
    let x = c.symbol("x");
    // x² - 5x + 6 = (x-2)(x-3)
    let expr = x.powi(2) - &x * 5 + 6;
    let roots = expr.solve(&x).unwrap();
    assert_eq!(roots.len(), 2, "quadratic should have 2 roots: {roots:?}");
    let mut vals: Vec<f64> = roots.iter().map(|r| r.eval_f64().unwrap()).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!(approx(vals[0], 2.0, 1e-10), "expected 2, got {}", vals[0]);
    assert!(approx(vals[1], 3.0, 1e-10), "expected 3, got {}", vals[1]);
}

#[test]
fn solve_cubic() {
    let c = ctx();
    let x = c.symbol("x");
    // x³ - 6x² + 11x - 6 = (x-1)(x-2)(x-3)
    let expr = x.powi(3) - &x.powi(2) * 6 + &x * 11 - 6;
    let roots = expr.solve(&x).unwrap();
    assert!(roots.len() >= 3, "cubic should have 3 roots: {roots:?}");
    // Verify each root satisfies the equation
    for r in &roots {
        let val = expr.subs(&x, r).eval();
        let v = val.eval_f64().unwrap_or(999.0);
        assert!(approx(v, 0.0, 1e-8), "root {r} gives residual {v}");
    }
}

#[test]
fn solve_quartic() {
    let c = ctx();
    let x = c.symbol("x");
    // x⁴ - 5x² + 4 = (x²-1)(x²-4) = (x-1)(x+1)(x-2)(x+2)
    let expr = x.powi(4) - &x.powi(2) * 5 + 4;
    let roots = expr.solve(&x).unwrap();
    assert!(roots.len() >= 4, "quartic should have 4 roots: {roots:?}");
    for r in &roots {
        let val = expr.subs(&x, r).eval();
        let v = val.eval_f64().unwrap_or(999.0);
        assert!(approx(v, 0.0, 1e-6), "root {r} gives residual {v}");
    }
}

#[test]
fn solve_returns_result_for_no_real_solutions() {
    let c = ctx();
    let x = c.symbol("x");
    // x² + 1 = 0  →  complex roots
    let expr = x.powi(2) + 1;
    let roots = expr.solve(&x);
    // Whether it returns Ok with complex roots or an error, it shouldn't panic
    if let Ok(roots) = roots {
        for r in &roots {
            let val = expr.subs(&x, r).eval();
            let v = val.eval_f64().unwrap_or(999.0);
            if v.is_finite() {
                assert!(
                    approx(v, 0.0, 1e-6),
                    "returned root {r} doesn't satisfy equation: residual {v}"
                );
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Polynomial factoring (exercises factor.rs → Poly → factor_over_z)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factor_difference_of_squares() {
    let c = ctx();
    let x = c.symbol("x");
    // x² - 1 should factor
    let expr = x.powi(2) - 1;
    let factored = expr.factor(&x);
    for pt in [-3, -1, 0, 1, 3] {
        let v1 = expr.subs_i64(&x, pt).eval_f64().unwrap();
        let v2 = factored.subs_i64(&x, pt).eval_f64().unwrap();
        assert!(
            approx(v1, v2, 1e-10),
            "value mismatch at x={pt}: {v1} vs {v2}"
        );
    }
}

#[test]
fn factor_perfect_square() {
    let c = ctx();
    let x = c.symbol("x");
    // x² + 2x + 1 = (x+1)²
    let expr = x.powi(2) + &x * 2 + 1;
    let factored = expr.factor(&x);
    for pt in [-3, -1, 0, 1, 3] {
        let v1 = expr.subs_i64(&x, pt).eval_f64().unwrap();
        let v2 = factored.subs_i64(&x, pt).eval_f64().unwrap();
        assert!(
            approx(v1, v2, 1e-10),
            "value mismatch at x={pt}: {v1} vs {v2}"
        );
    }
}

#[test]
fn factor_cubic_three_roots() {
    let c = ctx();
    let x = c.symbol("x");
    // x³ - 6x² + 11x - 6 = (x-1)(x-2)(x-3)
    let expr = x.powi(3) - &x.powi(2) * 6 + &x * 11 - 6;
    let factored = expr.factor(&x);
    for pt in [-2, 0, 1, 2, 3, 5] {
        let v1 = expr.subs_i64(&x, pt).eval_f64().unwrap();
        let v2 = factored.subs_i64(&x, pt).eval_f64().unwrap();
        assert!(
            approx(v1, v2, 1e-10),
            "factor value mismatch at x={pt}: {v1} vs {v2}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Polynomial cancellation (exercises polybridge → cancel → Poly::gcd)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cancel_common_factor() {
    let c = ctx();
    let x = c.symbol("x");
    // (x² - 1) / (x - 1) should simplify to (x + 1)
    let numer = x.powi(2) - 1;
    let denom = &x - 1;
    let frac = &numer / &denom;
    let simplified = frac.cancel(&x);
    // Evaluate at x=5: should be 6
    let val = simplified.subs_i64(&x, 5).eval_f64().unwrap();
    assert!(approx(val, 6.0, 1e-10), "expected 6, got {val}");
    // Evaluate at x=0: should be 1
    let val0 = simplified.subs_i64(&x, 0).eval_f64().unwrap();
    assert!(approx(val0, 1.0, 1e-10), "expected 1, got {val0}");
}

#[test]
fn cancel_no_common_factor() {
    let c = ctx();
    let x = c.symbol("x");
    // (x² + 1) / (x + 2) — no common factor
    let numer = x.powi(2) + 1;
    let denom = &x + 2;
    let frac = &numer / &denom;
    let simplified = frac.cancel(&x);
    let v1 = frac.subs_i64(&x, 3).eval_f64().unwrap();
    let v2 = simplified.subs_i64(&x, 3).eval_f64().unwrap();
    assert!(approx(v1, v2, 1e-10), "cancel changed value: {v1} vs {v2}");
}

#[test]
fn cancel_higher_degree_common_factor() {
    let c = ctx();
    let x = c.symbol("x");
    // (x³ - x) / (x² - 1) = x(x²-1)/(x²-1) = x
    let numer = x.powi(3) - &x;
    let denom = x.powi(2) - 1;
    let frac = &numer / &denom;
    let simplified = frac.cancel(&x);
    for pt in [2, 3, 5, 7] {
        let val = simplified.subs_i64(&x, pt).eval_f64().unwrap();
        assert!(approx(val, pt as f64, 1e-10), "expected {pt}, got {val}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Collect (exercises polybridge → collect → Poly)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn collect_polynomial() {
    let c = ctx();
    let x = c.symbol("x");
    // 3x + 2x + x²  should collect to x² + 5x
    let expr = &x * 3 + &x * 2 + x.powi(2);
    let collected = expr.collect(&x);
    for pt in [0, 1, 2, 5] {
        let v1 = expr.subs_i64(&x, pt).eval_f64().unwrap();
        let v2 = collected.subs_i64(&x, pt).eval_f64().unwrap();
        assert!(
            approx(v1, v2, 1e-10),
            "collect value mismatch at x={pt}: {v1} vs {v2}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Together (exercises polybridge → together → Poly LCM)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn together_simple_fractions() {
    let c = ctx();
    let x = c.symbol("x");
    // 1/x + 1/(x+1) = (2x+1)/(x(x+1))
    let one = c.int(1);
    let frac1 = &one / &x;
    let frac2 = &one / &(&x + 1);
    let sum = &frac1 + &frac2;
    let combined = sum.together();
    // Verify value at x=2: 1/2 + 1/3 = 5/6
    let val = combined.subs_i64(&x, 2).eval_f64().unwrap();
    let expected = 1.0 / 2.0 + 1.0 / 3.0;
    assert!(approx(val, expected, 1e-10), "expected {expected}, got {val}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Differentiation of polynomials (exercises derivative path)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_polynomial() {
    let c = ctx();
    let x = c.symbol("x");
    // d/dx(x³ + 2x² + 3x + 4) = 3x² + 4x + 3
    let expr = x.powi(3) + &x.powi(2) * 2 + &x * 3 + 4;
    let deriv = expr.diff(&x);
    let val = deriv.subs_i64(&x, 1).eval_f64().unwrap();
    // 3(1)² + 4(1) + 3 = 10
    assert!(approx(val, 10.0, 1e-10), "expected 10, got {val}");
}

#[test]
fn diff_high_degree_polynomial() {
    let c = ctx();
    let x = c.symbol("x");
    // d/dx(x^10) = 10x^9
    let expr = x.powi(10);
    let deriv = expr.diff(&x);
    let val = deriv.subs_i64(&x, 2).eval_f64().unwrap();
    // 10 * 2^9 = 10 * 512 = 5120
    assert!(approx(val, 5120.0, 1e-6), "expected 5120, got {val}");
}

#[test]
fn diff_derivative_correct_at_many_points() {
    let c = ctx();
    let x = c.symbol("x");
    // p(x) = x^5 - 3x^3 + 2x
    // p'(x) = 5x^4 - 9x^2 + 2
    let expr = x.powi(5) - &x.powi(3) * 3 + &x * 2;
    let deriv = expr.diff(&x);
    for pt in [-3, -1, 0, 1, 2, 4] {
        let xf = pt as f64;
        let expected = 5.0 * xf.powi(4) - 9.0 * xf.powi(2) + 2.0;
        let val = deriv.subs_i64(&x, pt).eval_f64().unwrap();
        assert!(
            approx(val, expected, 1e-6),
            "derivative mismatch at x={pt}: expected {expected}, got {val}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Integration of polynomials (exercises integration → Poly path)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_polynomial_roundtrip() {
    let c = ctx();
    let x = c.symbol("x");
    // ∫(3x² + 2x + 1)dx then d/dx should give back 3x² + 2x + 1
    let expr = &x.powi(2) * 3 + &x * 2 + 1;
    let integral = expr.integrate(&x);
    let roundtrip = integral.diff(&x);
    for pt in [-2, 0, 1, 3, 5] {
        let v1 = expr.subs_i64(&x, pt).eval_f64().unwrap();
        let v2 = roundtrip.subs_i64(&x, pt).eval_f64().unwrap();
        assert!(
            approx(v1, v2, 1e-10),
            "integrate-diff roundtrip failed at x={pt}: {v1} vs {v2}"
        );
    }
}

#[test]
fn integrate_rational_function() {
    let c = ctx();
    let x = c.symbol("x");
    // ∫ 1/x dx = ln(x)
    let one = c.int(1);
    let integrand = &one / &x;
    let integral = integrand.integrate(&x);
    // d/dx(result) should equal 1/x
    let roundtrip = integral.diff(&x);
    for pt in [1, 2, 3, 5] {
        let v1 = integrand.subs_i64(&x, pt).eval_f64().unwrap();
        let v2 = roundtrip.subs_i64(&x, pt).eval_f64().unwrap();
        assert!(
            approx(v1, v2, 1e-8),
            "integrate-diff roundtrip for 1/x failed at x={pt}: {v1} vs {v2}"
        );
    }
}

#[test]
fn integrate_higher_rational() {
    let c = ctx();
    let x = c.symbol("x");
    // ∫ 1/(x²+1) dx = atan(x)
    let one = c.int(1);
    let integrand = &one / &(x.powi(2) + 1);
    let integral = integrand.integrate(&x);
    // d/dx(result) should equal 1/(x²+1)
    let roundtrip = integral.diff(&x);
    for pt in [0, 1, 2, 3] {
        let v1 = integrand.subs_i64(&x, pt).eval_f64().unwrap();
        let v2 = roundtrip.subs_i64(&x, pt).eval_f64().unwrap();
        assert!(
            approx(v1, v2, 1e-8),
            "integrate-diff roundtrip for 1/(x²+1) failed at x={pt}: {v1} vs {v2}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Polynomial degree query (exercises polybridge → Poly → degree)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn degree_of_polynomial() {
    let c = ctx();
    let x = c.symbol("x");
    let expr = x.powi(5) + &x.powi(2) + 1;
    assert_eq!(expr.degree(&x), Some(5));
}

#[test]
fn degree_of_constant() {
    let c = ctx();
    let x = c.symbol("x");
    let expr = c.int(42);
    assert_eq!(expr.degree(&x), Some(0));
}

#[test]
fn degree_of_non_polynomial() {
    let c = ctx();
    let x = c.symbol("x");
    let expr = x.sin();
    assert_eq!(expr.degree(&x), None);
}

// ═══════════════════════════════════════════════════════════════════════════
// eval_f64 on polynomial expressions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_polynomial_at_point() {
    let c = ctx();
    let x = c.symbol("x");
    // p(x) = x³ - 2x + 1
    let expr = x.powi(3) - &x * 2 + 1;
    // p(3) = 27 - 6 + 1 = 22
    let val = expr.subs_i64(&x, 3).eval_f64().unwrap();
    assert!(approx(val, 22.0, 1e-10), "expected 22, got {val}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Gosper summation (exercises gosper.rs → Poly)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gosper_sum_of_k() {
    let c = ctx();
    let k = c.symbol("k");
    let n = c.symbol("n");
    // Build a proper Sum node: Σ_{k=0}^{n} k
    let sum_node = Ex::symbolic_sum(&k, &k, &c.int(0), &n);
    let result = sum_node.gosper_sum(&k);
    // Σ_{k=0}^{n} k = n(n+1)/2
    // At n=10: 10*11/2 = 55
    let val = result.subs_i64(&n, 10).eval_f64().unwrap();
    assert!(
        approx(val, 55.0, 1e-10),
        "sum of k from 0 to 10 should be 55, got {val}"
    );
    // At n=0: 0
    let val0 = result.subs_i64(&n, 0).eval_f64().unwrap();
    assert!(
        approx(val0, 0.0, 1e-10),
        "sum of k from 0 to 0 should be 0, got {val0}"
    );
    // At n=100: 100*101/2 = 5050
    let val100 = result.subs_i64(&n, 100).eval_f64().unwrap();
    assert!(
        approx(val100, 5050.0, 1e-10),
        "sum of k from 0 to 100 should be 5050, got {val100}"
    );
}

#[test]
fn gosper_sum_of_k_squared() {
    let c = ctx();
    let k = c.symbol("k");
    let n = c.symbol("n");
    // Build a proper Sum node: Σ_{k=0}^{n} k²
    let sum_node = Ex::symbolic_sum(&k.powi(2), &k, &c.int(0), &n);
    let result = sum_node.gosper_sum(&k);
    // Σ_{k=0}^{n} k² = n(n+1)(2n+1)/6
    // At n=5: 0+1+4+9+16+25 = 55
    let val5 = result.subs_i64(&n, 5).eval_f64().unwrap();
    assert!(
        approx(val5, 55.0, 1e-10),
        "sum of k² from 0 to 5 should be 55, got {val5}"
    );
    // At n=10: 10*11*21/6 = 385
    let val10 = result.subs_i64(&n, 10).eval_f64().unwrap();
    assert!(
        approx(val10, 385.0, 1e-10),
        "sum of k² from 0 to 10 should be 385, got {val10}"
    );
}

#[test]
fn gosper_sum_geometric() {
    let c = ctx();
    let k = c.symbol("k");
    let n = c.symbol("n");
    // Σ_{k=0}^{n} 2^k = 2^(n+1) - 1
    let body = c.int(2).pow(&k);
    let sum_node = Ex::symbolic_sum(&body, &k, &c.int(0), &n);
    let result = sum_node.gosper_sum(&k);
    // At n=5: 1+2+4+8+16+32 = 63 = 2^6 - 1
    let val5 = result.subs_i64(&n, 5).eval_f64().unwrap();
    assert!(
        approx(val5, 63.0, 1e-10),
        "sum of 2^k from 0 to 5 should be 63, got {val5}"
    );
    // At n=10: 2^11 - 1 = 2047
    let val10 = result.subs_i64(&n, 10).eval_f64().unwrap();
    assert!(
        approx(val10, 2047.0, 1e-10),
        "sum of 2^k from 0 to 10 should be 2047, got {val10}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Gröbner basis and polynomial system solving (MultiPoly path)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn groebner_simple_system() {
    use symplex::groebner::*;
    use symplex::multipoly::*;

    // System: x + y = 3, x - y = 1
    let vars = multipoly_vars::<GrevLex>(2);
    let (x, y) = (&vars[0], &vars[1]);
    let p1 = x + y - 3i64;
    let p2 = x - y - 1i64;
    let basis = groebner_basis(&[p1, p2]);
    assert!(
        is_groebner_basis(&basis),
        "result should be a valid Gröbner basis"
    );
}

#[test]
fn polysys_solve_linear() {
    use num_bigint::BigInt;
    use num_rational::Ratio;
    use symplex::multipoly::*;
    use symplex::polysys::solve_polynomial_system;

    let vars = multipoly_vars::<GrevLex>(2);
    let (x, y) = (&vars[0], &vars[1]);
    let p1 = x + y - 3i64;
    let p2 = x - y - 1i64;
    let solutions = solve_polynomial_system(&[p1, p2]);
    assert!(solutions.is_ok(), "should solve: {:?}", solutions);
    let sols = solutions.unwrap();
    assert_eq!(sols.len(), 1, "should have exactly 1 solution");
    let sol = &sols[0];
    assert_eq!(sol[0], Ratio::from_integer(BigInt::from(2)));
    assert_eq!(sol[1], Ratio::from_integer(BigInt::from(1)));
}

// ═══════════════════════════════════════════════════════════════════════════
// Series expansion (exercises polynomial creation paths)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn maclaurin_of_polynomial_is_exact() {
    let c = ctx();
    let x = c.symbol("x");
    let poly = x.powi(3) + &x * 2 + 7;
    let series = poly.maclaurin(&x, 5);
    for pt in [-2, -1, 0, 1, 2] {
        let v1 = poly.subs_i64(&x, pt).eval_f64().unwrap();
        let v2 = series.subs_i64(&x, pt).eval_f64().unwrap();
        assert!(
            approx(v1, v2, 1e-10),
            "series should match polynomial at x={pt}: {v1} vs {v2}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Risch integration (exercises risch → Poly → GenPoly pathway)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_rational_via_risch() {
    let c = ctx();
    let x = c.symbol("x");
    // ∫ (2x+1)/(x²+x) dx should involve logarithms
    let numer = &x * 2 + 1;
    let denom = x.powi(2) + &x;
    let integrand = &numer / &denom;
    let integral = integrand.integrate(&x);
    // d/dx should give back the integrand
    let roundtrip = integral.diff(&x);
    for pt in [1, 2, 3, 5] {
        let v1 = integrand.subs_i64(&x, pt).eval_f64().unwrap();
        let v2 = roundtrip.subs_i64(&x, pt).eval_f64().unwrap();
        assert!(
            approx(v1, v2, 1e-8),
            "Risch roundtrip failed at x={pt}: {v1} vs {v2}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Lambdify / compile (exercises codegen → Poly simplification path)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn compile_polynomial() {
    let c = ctx();
    let x = c.symbol("x");
    let poly = x.powi(4) - &x.powi(2) * 3 + &x * 2 + 7;
    let f = poly.compile(&["x"]).expect("compile should succeed");
    // Evaluate at x=2.5
    let compiled_val = f(&[2.5]);
    let symbolic_val = poly.subs(&x, &c.rational(5, 2)).eval_f64().unwrap();
    assert!(
        approx(compiled_val, symbolic_val, 1e-10),
        "compile mismatch: {compiled_val} vs {symbolic_val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Stress: large polynomial through the full pipeline
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn stress_large_polynomial_expand_diff_integrate() {
    let c = ctx();
    let x = c.symbol("x");
    // Build (x+1)^8 via repeated squaring
    let base = &x + 1;
    let p2 = &base * &base;
    let p4 = &p2 * &p2;
    let p8 = &p4 * &p4;
    let expanded = p8.expand();
    let degree = expanded.degree(&x);
    assert_eq!(degree, Some(8), "degree should be 8, got {degree:?}");

    // Differentiate
    let deriv = expanded.diff(&x);
    assert_eq!(deriv.degree(&x), Some(7), "derivative degree should be 7");

    // Integrate and differentiate should roundtrip
    let integral = expanded.integrate(&x);
    let roundtrip = integral.diff(&x);
    for pt in [-1, 0, 1, 2] {
        let v1 = expanded.subs_i64(&x, pt).eval_f64().unwrap();
        let v2 = roundtrip.subs_i64(&x, pt).eval_f64().unwrap();
        assert!(
            approx(v1, v2, 1e-8),
            "large poly roundtrip failed at x={pt}: {v1} vs {v2}"
        );
    }
}

#[test]
fn stress_product_of_many_linears() {
    let c = ctx();
    let x = c.symbol("x");
    // (x-1)(x-2)(x-3)(x-4)(x-5)
    let mut product = &x - 1;
    for k in 2..=5 {
        product = &product * &(&x - k);
    }
    let expanded = product.expand();
    assert_eq!(expanded.degree(&x), Some(5));
    // All roots should evaluate to 0
    for k in 1..=5 {
        let val = expanded.subs_i64(&x, k).eval_f64().unwrap();
        assert!(approx(val, 0.0, 1e-8), "root at x={k} gives {val}");
    }
}

#[test]
fn stress_factor_then_solve() {
    let c = ctx();
    let x = c.symbol("x");
    // x⁶ - 1 = (x-1)(x+1)(x²+x+1)(x²-x+1)
    let expr = x.powi(6) - 1;
    let factored = expr.factor(&x);
    // Factor preserves value
    for pt in [-3, -1, 0, 1, 3] {
        let v1 = expr.subs_i64(&x, pt).eval_f64().unwrap();
        let v2 = factored.subs_i64(&x, pt).eval_f64().unwrap();
        assert!(
            approx(v1, v2, 1e-10),
            "factor value mismatch at x={pt}: {v1} vs {v2}"
        );
    }
    // Solve finds at least the real roots
    let roots = expr.solve(&x);
    if let Ok(roots) = roots {
        // x=1 and x=-1 should be among them
        let has_one = roots.iter().any(|r| {
            r.eval_f64()
                .map(|v| approx(v, 1.0, 1e-8))
                .unwrap_or(false)
        });
        let has_neg_one = roots.iter().any(|r| {
            r.eval_f64()
                .map(|v| approx(v, -1.0, 1e-8))
                .unwrap_or(false)
        });
        assert!(has_one, "x=1 should be a root of x⁶-1");
        assert!(has_neg_one, "x=-1 should be a root of x⁶-1");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Cancel + Together round-trip
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn together_then_cancel_roundtrip() {
    let c = ctx();
    let x = c.symbol("x");
    let one = c.int(1);
    // 1/(x-1) + 1/(x+1) → together → cancel
    let sum = &(&one / &(&x - 1)) + &(&one / &(&x + 1));
    let combined = sum.together();
    let cancelled = combined.cancel(&x);
    for pt in [2, 3, 5, 10] {
        let v1 = sum.subs_i64(&x, pt).eval_f64().unwrap();
        let v2 = cancelled.subs_i64(&x, pt).eval_f64().unwrap();
        assert!(
            approx(v1, v2, 1e-10),
            "together-cancel roundtrip failed at x={pt}: {v1} vs {v2}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Verify Poly (as type alias) preserves all arithmetic identities
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn polynomial_arithmetic_identities() {
    let c = ctx();
    let x = c.symbol("x");
    let p = x.powi(3) + &x * 2 - 5;
    let q = x.powi(2) - &x + 1;

    for pt in [-3, -1, 0, 1, 2, 5] {
        let pv = p.subs_i64(&x, pt).eval_f64().unwrap();
        let qv = q.subs_i64(&x, pt).eval_f64().unwrap();

        // p + q
        let sum_v = (&p + &q).subs_i64(&x, pt).eval_f64().unwrap();
        assert!(
            approx(sum_v, pv + qv, 1e-10),
            "p+q mismatch at x={pt}"
        );

        // p - q
        let diff_v = (&p - &q).subs_i64(&x, pt).eval_f64().unwrap();
        assert!(
            approx(diff_v, pv - qv, 1e-10),
            "p-q mismatch at x={pt}"
        );

        // p * q
        let prod_v = (&p * &q).subs_i64(&x, pt).eval_f64().unwrap();
        assert!(
            approx(prod_v, pv * qv, 1e-10),
            "p*q mismatch at x={pt}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Verify Ex is Send + Sync (compile-time check)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ex_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Ex>();
}

//! Comprehensive correctness probe — verifies symbolic results are numerically correct.
//!
//! For every integration we claim works, this probe:
//!   1. Computes the symbolic antiderivative F(x) via `integrate()`
//!   2. If unevaluated (contains "Integral("), skips it
//!   3. Computes F(hi) - F(lo) as the symbolic definite integral value
//!   4. Computes numerical quadrature of the integrand from lo to hi
//!   5. Compares |symbolic - numeric| < tolerance
//!   6. Reports PASS, WRONG, or SKIP for each test
//!
//! Also verifies simplification identities, series accuracy, and ODE solutions.
//!
//! Run with: cargo run --example probe_correctness
//!
//! Exit code 1 if any result is WRONG.

use symplex::prelude::*;

// ── Numerical helpers ──────────────────────────────────────────────────

/// Evaluate a symbolic expression at a rational point x = p/q, returning f64.
/// Uses subs with a rational expression for best accuracy.
fn eval_at(expr: &Ex, var: &Ex, p: i64, q: i64) -> Option<f64> {
    let pt = symplex::rational(p, q);
    let substituted = expr.subs(var, &pt);
    substituted.eval_f64().ok()
}

/// Numerical integration via composite Simpson's rule with n panels (n must be even).
/// Evaluates the integrand at n+1 points in [a, b].
fn numerical_quadrature(
    integrand: &Ex,
    var: &Ex,
    a_num: i64,
    a_den: i64,
    b_num: i64,
    b_den: i64,
    n: usize,
) -> Option<f64> {
    // Convert bounds to f64 for step computation
    let a_f = a_num as f64 / a_den as f64;
    let b_f = b_num as f64 / b_den as f64;
    let h = (b_f - a_f) / n as f64;

    // Use trapezoidal rule for robustness (Simpson requires even n)
    let mut sum = 0.0;
    for i in 0..=n {
        let xi_f = a_f + i as f64 * h;
        // Approximate xi as rational: multiply by 10000, round
        let xi_num = (xi_f * 10000.0).round() as i64;
        let xi_den: i64 = 10000;

        let val = eval_at(integrand, var, xi_num, xi_den)?;
        if val.is_nan() || val.is_infinite() {
            return None;
        }
        let weight = if i == 0 || i == n { 0.5 } else { 1.0 };
        sum += weight * val;
    }
    Some(sum * h)
}

/// Compute definite integral symbolically: F(hi) - F(lo)
fn symbolic_definite(
    antideriv: &Ex,
    var: &Ex,
    hi_num: i64,
    hi_den: i64,
    lo_num: i64,
    lo_den: i64,
) -> Option<f64> {
    let f_hi = eval_at(antideriv, var, hi_num, hi_den)?;
    let f_lo = eval_at(antideriv, var, lo_num, lo_den)?;
    if f_hi.is_nan() || f_hi.is_infinite() || f_lo.is_nan() || f_lo.is_infinite() {
        return None;
    }
    Some(f_hi - f_lo)
}

/// Check FTC: compute numerical derivative of antiderivative and compare to integrand.
/// Uses central difference: F'(x) ≈ (F(x+h) - F(x-h)) / (2h)
fn ftc_check_at_point(
    integrand: &Ex,
    antideriv: &Ex,
    var: &Ex,
    pt_num: i64,
    pt_den: i64,
) -> Option<(f64, f64, f64)> {
    let f_val = eval_at(integrand, var, pt_num, pt_den)?;

    // h = 1/10000
    let h_den: i64 = 10000;
    // x + h
    let xph_num = pt_num * h_den + pt_den;
    let xph_den = pt_den * h_den;
    // x - h
    let xmh_num = pt_num * h_den - pt_den;
    let xmh_den = pt_den * h_den;

    let f_plus = eval_at(antideriv, var, xph_num, xph_den)?;
    let f_minus = eval_at(antideriv, var, xmh_num, xmh_den)?;

    let h_f = 1.0 / h_den as f64;
    let numerical_deriv = (f_plus - f_minus) / (2.0 * h_f);

    let error = (f_val - numerical_deriv).abs();
    Some((f_val, numerical_deriv, error))
}

// ── Test result tracking ───────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    Pass,
    Wrong,
    Skip,
}

struct TestResult {
    label: String,
    verdict: Verdict,
    detail: String,
}

// ── Integration test definitions ───────────────────────────────────────

struct IntegralTest {
    label: &'static str,
    integrand: Ex,
    /// Integration bounds as (lo_num, lo_den, hi_num, hi_den)
    bounds: (i64, i64, i64, i64),
}

fn build_integration_tests(x: &Ex) -> Vec<IntegralTest> {
    vec![
        // ── Basic ──
        IntegralTest {
            label: "x^2",
            integrand: x.powi(2),
            bounds: (1, 2, 1, 1),
        },
        IntegralTest {
            label: "x^5",
            integrand: x.powi(5),
            bounds: (1, 2, 1, 1),
        },
        IntegralTest {
            label: "sin(x)",
            integrand: x.sin(),
            bounds: (1, 2, 1, 1),
        },
        IntegralTest {
            label: "cos(x)",
            integrand: x.cos(),
            bounds: (1, 2, 1, 1),
        },
        IntegralTest {
            label: "exp(x)",
            integrand: x.exp(),
            bounds: (1, 2, 1, 1),
        },
        IntegralTest {
            label: "1/x",
            integrand: &symplex::int(1) / x,
            bounds: (1, 2, 2, 1),
        },
        IntegralTest {
            label: "tan(x)",
            integrand: x.tan(),
            bounds: (1, 4, 1, 1),
        },
        IntegralTest {
            label: "ln(x)",
            integrand: x.ln(),
            bounds: (1, 2, 2, 1),
        },
        // ── By-parts ──
        IntegralTest {
            label: "x*exp(x)",
            integrand: x * &x.exp(),
            bounds: (1, 2, 1, 1),
        },
        IntegralTest {
            label: "x*sin(x)",
            integrand: x * &x.sin(),
            bounds: (1, 2, 1, 1),
        },
        IntegralTest {
            label: "x*cos(x)",
            integrand: x * &x.cos(),
            bounds: (1, 2, 1, 1),
        },
        IntegralTest {
            label: "x^2*exp(x)",
            integrand: &x.powi(2) * &x.exp(),
            bounds: (1, 2, 1, 1),
        },
        IntegralTest {
            label: "x*ln(x)",
            integrand: x * &x.ln(),
            bounds: (1, 2, 2, 1),
        },
        IntegralTest {
            label: "ln(x)^2",
            integrand: x.ln().powi(2),
            bounds: (1, 2, 2, 1),
        },
        IntegralTest {
            label: "x^2*sin(x)",
            integrand: &x.powi(2) * &x.sin(),
            bounds: (1, 2, 1, 1),
        },
        IntegralTest {
            label: "x^3*exp(x)",
            integrand: &x.powi(3) * &x.exp(),
            bounds: (1, 2, 1, 1),
        },
        // ── U-substitution ──
        IntegralTest {
            label: "2x*exp(x^2)",
            integrand: &(&symplex::int(2) * x) * &x.powi(2).exp(),
            bounds: (1, 2, 1, 1),
        },
        IntegralTest {
            label: "cos(x)*exp(sin(x))",
            integrand: &x.cos() * &x.sin().exp(),
            bounds: (1, 2, 1, 1),
        },
        IntegralTest {
            label: "x/(x^2+1)",
            integrand: x / &(x.powi(2) + 1),
            bounds: (1, 2, 1, 1),
        },
        // ── Trig powers ──
        IntegralTest {
            label: "sin^2(x)",
            integrand: x.sin().powi(2),
            bounds: (1, 2, 1, 1),
        },
        IntegralTest {
            label: "cos^2(x)",
            integrand: x.cos().powi(2),
            bounds: (1, 2, 1, 1),
        },
        IntegralTest {
            label: "sin^3(x)",
            integrand: x.sin().powi(3),
            bounds: (1, 2, 1, 1),
        },
        IntegralTest {
            label: "sin^4(x)",
            integrand: x.sin().powi(4),
            bounds: (1, 2, 1, 1),
        },
        IntegralTest {
            label: "tan^2(x)",
            integrand: x.tan().powi(2),
            bounds: (1, 4, 1, 1),
        },
        IntegralTest {
            label: "sin(x)*cos(x)",
            integrand: &x.sin() * &x.cos(),
            bounds: (1, 2, 1, 1),
        },
        IntegralTest {
            label: "sin^2(x)*cos^2(x)",
            integrand: &x.sin().powi(2) * &x.cos().powi(2),
            bounds: (1, 2, 1, 1),
        },
        // ── Rational functions ──
        IntegralTest {
            label: "1/(x^2+1)",
            integrand: symplex::int(1) / &(x.powi(2) + 1),
            bounds: (1, 2, 1, 1),
        },
        IntegralTest {
            label: "1/(x^2-1)",
            integrand: symplex::int(1) / &(x.powi(2) - 1),
            bounds: (2, 1, 3, 1), // away from singularity at x=1
        },
        IntegralTest {
            label: "x/(x^2+1)^2",
            integrand: x / &(x.powi(2) + 1).powi(2),
            bounds: (1, 2, 1, 1),
        },
        // ── Sqrt forms ──
        IntegralTest {
            label: "1/sqrt(1-x^2)",
            integrand: symplex::int(1) / &(&symplex::int(1) - &x.powi(2)).sqrt(),
            bounds: (1, 4, 1, 2), // (0.25, 0.5) — inside (-1,1)
        },
        IntegralTest {
            label: "1/sqrt(x^2+1)",
            integrand: symplex::int(1) / &(x.powi(2) + 1).sqrt(),
            bounds: (1, 2, 1, 1),
        },
        IntegralTest {
            label: "x/sqrt(x^2+1)",
            integrand: x / &(x.powi(2) + 1).sqrt(),
            bounds: (1, 2, 1, 1),
        },
        // ── Cyclic IBP ──
        IntegralTest {
            label: "exp(x)*sin(x)",
            integrand: &x.exp() * &x.sin(),
            bounds: (1, 2, 1, 1),
        },
        IntegralTest {
            label: "exp(x)*cos(x)",
            integrand: &x.exp() * &x.cos(),
            bounds: (1, 2, 1, 1),
        },
        // ── Inverse trig ──
        IntegralTest {
            label: "asin(x)",
            integrand: x.asin(),
            bounds: (1, 4, 1, 2), // (0.25, 0.5) — inside (-1,1)
        },
        IntegralTest {
            label: "acos(x)",
            integrand: x.acos(),
            bounds: (1, 4, 1, 2),
        },
        IntegralTest {
            label: "atan(x)",
            integrand: x.atan(),
            bounds: (1, 2, 1, 1),
        },
        // ── Parametric (specific a values) ──
        IntegralTest {
            label: "sin(2x)",
            integrand: (&symplex::int(2) * x).sin(),
            bounds: (1, 2, 1, 1),
        },
        IntegralTest {
            label: "exp(2x)",
            integrand: (&symplex::int(2) * x).exp(),
            bounds: (1, 2, 1, 1),
        },
        IntegralTest {
            label: "sin(3x)",
            integrand: (&symplex::int(3) * x).sin(),
            bounds: (1, 2, 1, 1),
        },
        // ── Hyperbolic ──
        IntegralTest {
            label: "sinh(x)",
            integrand: x.sinh(),
            bounds: (1, 2, 1, 1),
        },
        IntegralTest {
            label: "cosh(x)",
            integrand: x.cosh(),
            bounds: (1, 2, 1, 1),
        },
        IntegralTest {
            label: "tanh(x)",
            integrand: x.tanh(),
            bounds: (1, 2, 1, 1),
        },
        IntegralTest {
            label: "sinh^2(x)",
            integrand: x.sinh().powi(2),
            bounds: (1, 2, 1, 1),
        },
        // ── Linear substitution ──
        IntegralTest {
            label: "(2x+1)^5",
            integrand: (&symplex::int(2) * x + 1).powi(5),
            bounds: (1, 2, 1, 1),
        },
        IntegralTest {
            label: "1/(3x+2)",
            integrand: symplex::int(1) / &(&symplex::int(3) * x + 2),
            bounds: (1, 2, 1, 1),
        },
        IntegralTest {
            label: "sqrt(2x+1)",
            integrand: (&symplex::int(2) * x + 1).sqrt(),
            bounds: (1, 2, 1, 1),
        },
        // ── Completing the square ──
        IntegralTest {
            label: "1/(x^2+2x+5)",
            integrand: symplex::int(1) / &(x.powi(2) + &symplex::int(2) * x + 5),
            bounds: (1, 2, 1, 1),
        },
    ]
}

// ── Simplification identity tests ──────────────────────────────────────

struct SimplifyTest {
    label: &'static str,
    original: Ex,
    /// Test points as (num, den) pairs
    test_points: Vec<(i64, i64)>,
}

fn build_simplify_tests(x: &Ex) -> Vec<SimplifyTest> {
    vec![
        SimplifyTest {
            label: "sin^2+cos^2 = 1",
            original: &x.sin().powi(2) + &x.cos().powi(2),
            test_points: vec![(1, 2), (1, 1), (3, 2), (2, 1), (7, 3)],
        },
        SimplifyTest {
            label: "tan^2+1 = sec^2",
            original: &x.tan().powi(2) + 1,
            test_points: vec![(1, 4), (1, 2), (1, 1), (5, 4)],
        },
        SimplifyTest {
            label: "cosh^2-sinh^2 = 1",
            original: &x.cosh().powi(2) - &x.sinh().powi(2),
            test_points: vec![(1, 2), (1, 1), (3, 2), (2, 1)],
        },
        SimplifyTest {
            label: "2*sin*cos = sin(2x)",
            original: &symplex::int(2) * &(&x.sin() * &x.cos()),
            test_points: vec![(1, 4), (1, 2), (1, 1), (3, 2)],
        },
    ]
}

fn main() {
    let x = symplex::var("x");

    let mut results: Vec<TestResult> = Vec::new();
    let tolerance = 1e-3;
    let ftc_tolerance = 1e-2; // FTC numerical derivative is less precise

    // ═══════════════════════════════════════════════════════════════════
    // 1. Integration correctness via definite integral comparison
    // ═══════════════════════════════════════════════════════════════════
    println!("=== Integration Correctness Probe ===\n");
    println!("Method: compare symbolic ∫ₐᵇ f(x)dx with numerical quadrature\n");

    let tests = build_integration_tests(&x);

    for test in &tests {
        let (lo_num, lo_den, hi_num, hi_den) = test.bounds;
        let lo_f = lo_num as f64 / lo_den as f64;
        let hi_f = hi_num as f64 / hi_den as f64;

        // Step 1: symbolic integration
        let antideriv = test.integrand.integrate(&x);
        let antideriv_str = format!("{antideriv}");

        // Step 2: check if unevaluated
        if antideriv_str.contains("Integral(") || antideriv_str.contains("integral(") {
            println!("  ⏭  SKIP  {} — unevaluated", test.label);
            results.push(TestResult {
                label: test.label.to_string(),
                verdict: Verdict::Skip,
                detail: "unevaluated integral".to_string(),
            });
            continue;
        }

        // Step 3: symbolic definite integral
        let sym_val = symbolic_definite(&antideriv, &x, hi_num, hi_den, lo_num, lo_den);

        // Step 4: numerical quadrature (200 panels — trapezoidal error O(h²) ≈ 6e-6)
        let num_val = numerical_quadrature(
            &test.integrand,
            &x,
            lo_num,
            lo_den,
            hi_num,
            hi_den,
            200,
        );

        match (sym_val, num_val) {
            (Some(sv), Some(nv)) => {
                let error = (sv - nv).abs();
                let rel_error = if nv.abs() > 1e-10 {
                    error / nv.abs()
                } else {
                    error
                };
                if error < tolerance || rel_error < tolerance {
                    println!(
                        "  ✅ PASS  {} — sym={:.8}, num={:.8}, err={:.2e}",
                        test.label, sv, nv, error
                    );
                    results.push(TestResult {
                        label: test.label.to_string(),
                        verdict: Verdict::Pass,
                        detail: format!(
                            "[{},{}] sym={:.8} num={:.8} err={:.2e}",
                            lo_f, hi_f, sv, nv, error
                        ),
                    });
                } else {
                    println!(
                        "  ❌ WRONG {} — sym={:.8}, num={:.8}, err={:.2e}",
                        test.label, sv, nv, error
                    );
                    results.push(TestResult {
                        label: test.label.to_string(),
                        verdict: Verdict::Wrong,
                        detail: format!(
                            "[{},{}] sym={:.8} num={:.8} err={:.2e}",
                            lo_f, hi_f, sv, nv, error
                        ),
                    });
                }
            }
            (None, _) => {
                println!(
                    "  ⏭  SKIP  {} — cannot evaluate antiderivative at bounds",
                    test.label
                );
                results.push(TestResult {
                    label: test.label.to_string(),
                    verdict: Verdict::Skip,
                    detail: "eval failed for antiderivative".to_string(),
                });
            }
            (_, None) => {
                println!(
                    "  ⏭  SKIP  {} — numerical quadrature failed",
                    test.label
                );
                results.push(TestResult {
                    label: test.label.to_string(),
                    verdict: Verdict::Skip,
                    detail: "numerical quadrature failed".to_string(),
                });
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 2. FTC verification: d/dx(F(x)) ≈ f(x) at test points
    // ═══════════════════════════════════════════════════════════════════
    println!("\n=== FTC Verification (d/dx F(x) ≈ f(x)) ===\n");

    // Pick a subset of integrals for FTC verification at specific points
    let ftc_tests: Vec<(&str, Ex, Vec<(i64, i64)>)> = vec![
        ("x^2", x.powi(2), vec![(1, 2), (1, 1), (3, 2)]),
        ("sin(x)", x.sin(), vec![(1, 2), (1, 1), (3, 2)]),
        ("exp(x)", x.exp(), vec![(1, 2), (1, 1), (3, 2)]),
        ("x*exp(x)", &x * &x.exp(), vec![(1, 2), (1, 1), (3, 2)]),
        (
            "sin^2(x)",
            x.sin().powi(2),
            vec![(1, 2), (1, 1), (3, 2)],
        ),
        (
            "tan^2(x)",
            x.tan().powi(2),
            vec![(1, 4), (1, 2), (3, 4)],
        ),
        (
            "ln(x)^2",
            x.ln().powi(2),
            vec![(1, 2), (1, 1), (3, 2)],
        ),
        (
            "exp(x)*sin(x)",
            &x.exp() * &x.sin(),
            vec![(1, 2), (1, 1), (3, 2)],
        ),
        (
            "sinh^2(x)",
            x.sinh().powi(2),
            vec![(1, 2), (1, 1), (3, 2)],
        ),
    ];

    for (label, integrand, points) in &ftc_tests {
        let antideriv = integrand.integrate(&x);
        let s = format!("{antideriv}");
        if s.contains("Integral(") {
            println!("  ⏭  SKIP  {} — unevaluated", label);
            results.push(TestResult {
                label: format!("FTC:{}", label),
                verdict: Verdict::Skip,
                detail: "unevaluated".to_string(),
            });
            continue;
        }

        let mut all_ok = true;
        let mut worst_error = 0.0f64;
        let mut point_count = 0;

        for &(pn, pd) in points {
            match ftc_check_at_point(integrand, &antideriv, &x, pn, pd) {
                Some((f_val, fp_val, error)) => {
                    point_count += 1;
                    worst_error = worst_error.max(error);
                    if error > ftc_tolerance {
                        all_ok = false;
                        println!(
                            "  ❌ WRONG FTC {} at x={}/{}: f={:.6} F'≈{:.6} err={:.2e}",
                            label, pn, pd, f_val, fp_val, error
                        );
                    }
                }
                None => {} // Skip this point if eval fails
            }
        }

        if point_count == 0 {
            println!("  ⏭  SKIP  FTC {} — no evaluable points", label);
            results.push(TestResult {
                label: format!("FTC:{}", label),
                verdict: Verdict::Skip,
                detail: "no evaluable points".to_string(),
            });
        } else if all_ok {
            println!(
                "  ✅ PASS  FTC {} — {} points, worst err={:.2e}",
                label, point_count, worst_error
            );
            results.push(TestResult {
                label: format!("FTC:{}", label),
                verdict: Verdict::Pass,
                detail: format!("{} points, worst={:.2e}", point_count, worst_error),
            });
        } else {
            results.push(TestResult {
                label: format!("FTC:{}", label),
                verdict: Verdict::Wrong,
                detail: format!("worst err={:.2e}", worst_error),
            });
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 3. Simplification identity checks
    // ═══════════════════════════════════════════════════════════════════
    println!("\n=== Simplification Identity Verification ===\n");

    let simp_tests = build_simplify_tests(&x);

    for test in &simp_tests {
        let simplified = test.original.full_simplify();

        let mut all_ok = true;
        let mut point_count = 0;

        for &(pn, pd) in &test.test_points {
            let orig_val = eval_at(&test.original, &x, pn, pd);
            let simp_val = eval_at(&simplified, &x, pn, pd);

            match (orig_val, simp_val) {
                (Some(ov), Some(sv)) => {
                    point_count += 1;
                    let error = (ov - sv).abs();
                    if error > tolerance {
                        all_ok = false;
                        println!(
                            "  ❌ WRONG SIMP {} at x={}/{}: orig={:.6} simp={:.6} err={:.2e}",
                            test.label, pn, pd, ov, sv, error
                        );
                    }
                }
                _ => {}
            }
        }

        if point_count == 0 {
            println!("  ⏭  SKIP  {} — no evaluable points", test.label);
            results.push(TestResult {
                label: format!("SIMP:{}", test.label),
                verdict: Verdict::Skip,
                detail: "no evaluable points".to_string(),
            });
        } else if all_ok {
            println!("  ✅ PASS  {} — {} points all match", test.label, point_count);
            results.push(TestResult {
                label: format!("SIMP:{}", test.label),
                verdict: Verdict::Pass,
                detail: format!("{} points", point_count),
            });
        } else {
            results.push(TestResult {
                label: format!("SIMP:{}", test.label),
                verdict: Verdict::Wrong,
                detail: "mismatch".to_string(),
            });
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 4. ODE solution verification
    // ═══════════════════════════════════════════════════════════════════
    println!("\n=== ODE Solution Verification ===\n");

    let y = symplex::var("y");
    let dy = y.formal_diff(&x);
    let ddy = dy.formal_diff(&x);

    let ode_cases: Vec<(&str, Ex)> = vec![
        ("y' - x = 0", &dy - &x),
        ("y' + 2y = 0", &dy + &(&symplex::int(2) * &y)),
        ("y'' + y = 0", &ddy + &y),
        ("y'' - 4y = 0", &ddy - &(&symplex::int(4) * &y)),
    ];

    for (label, ode) in &ode_cases {
        match ode.solve_ode(&y, &x) {
            Some((sol, _consts)) => {
                // Verify: check_ode_solution
                let ok = ode.check_ode_solution(&sol, &y, &x);
                if ok {
                    println!("  ✅ PASS  {} — solution verified: y = {}", label, sol);
                    results.push(TestResult {
                        label: format!("ODE:{}", label),
                        verdict: Verdict::Pass,
                        detail: format!("y = {}", sol),
                    });
                } else {
                    println!(
                        "  ❌ WRONG {} — solution FAILS verification: y = {}",
                        label, sol
                    );
                    results.push(TestResult {
                        label: format!("ODE:{}", label),
                        verdict: Verdict::Wrong,
                        detail: format!("fails verification: y = {}", sol),
                    });
                }
            }
            None => {
                println!("  ⏭  SKIP  {} — no solution found", label);
                results.push(TestResult {
                    label: format!("ODE:{}", label),
                    verdict: Verdict::Skip,
                    detail: "unsolvable".to_string(),
                });
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 5. Series expansion accuracy check
    // ═══════════════════════════════════════════════════════════════════
    println!("\n=== Series Expansion Accuracy ===\n");

    let series_cases: Vec<(&str, Ex, Vec<(i64, i64)>)> = vec![
        ("sin(x)", x.sin(), vec![(1, 10), (1, 5), (3, 10)]),
        ("cos(x)", x.cos(), vec![(1, 10), (1, 5), (3, 10)]),
        ("exp(x)", x.exp(), vec![(1, 10), (1, 5), (3, 10)]),
        (
            "ln(1+x)",
            (&x + 1).ln(),
            vec![(1, 10), (1, 5), (3, 10)],
        ),
        ("sinh(x)", x.sinh(), vec![(1, 10), (1, 5), (3, 10)]),
        ("cosh(x)", x.cosh(), vec![(1, 10), (1, 5), (3, 10)]),
    ];

    let series_tol = 1e-5; // Series at small x with order 8 should be very accurate

    for (label, expr, points) in &series_cases {
        match expr.maclaurin(&x, 8) {
            Ok(series) => {
                let expanded = series.expand().eval();
                let mut all_ok = true;
                let mut worst_error = 0.0f64;
                let mut point_count = 0;

                for &(pn, pd) in points {
                    let exact = eval_at(expr, &x, pn, pd);
                    let approx = eval_at(&expanded, &x, pn, pd);

                    match (exact, approx) {
                        (Some(ev), Some(av)) => {
                            point_count += 1;
                            let error = (ev - av).abs();
                            worst_error = worst_error.max(error);
                            if error > series_tol {
                                all_ok = false;
                            }
                        }
                        _ => {}
                    }
                }

                if point_count == 0 {
                    println!("  ⏭  SKIP  series {} — no evaluable points", label);
                    results.push(TestResult {
                        label: format!("SERIES:{}", label),
                        verdict: Verdict::Skip,
                        detail: "no evaluable points".to_string(),
                    });
                } else if all_ok {
                    println!(
                        "  ✅ PASS  series {} — {} points, worst err={:.2e}",
                        label, point_count, worst_error
                    );
                    results.push(TestResult {
                        label: format!("SERIES:{}", label),
                        verdict: Verdict::Pass,
                        detail: format!("{} points, worst={:.2e}", point_count, worst_error),
                    });
                } else {
                    println!(
                        "  ❌ WRONG series {} — worst err={:.2e}",
                        label, worst_error
                    );
                    results.push(TestResult {
                        label: format!("SERIES:{}", label),
                        verdict: Verdict::Wrong,
                        detail: format!("worst={:.2e}", worst_error),
                    });
                }
            }
            Err(e) => {
                println!("  ⏭  SKIP  series {} — {}", label, e);
                results.push(TestResult {
                    label: format!("SERIES:{}", label),
                    verdict: Verdict::Skip,
                    detail: format!("{}", e),
                });
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════════
    println!("\n{}", "=".repeat(60));
    println!("=== CORRECTNESS PROBE SUMMARY ===\n");

    let pass_count = results.iter().filter(|r| r.verdict == Verdict::Pass).count();
    let wrong_count = results.iter().filter(|r| r.verdict == Verdict::Wrong).count();
    let skip_count = results.iter().filter(|r| r.verdict == Verdict::Skip).count();
    let total = results.len();

    println!("  ✅ PASS:  {}", pass_count);
    println!("  ❌ WRONG: {}", wrong_count);
    println!("  ⏭  SKIP:  {}", skip_count);
    println!("  TOTAL:    {}", total);

    if wrong_count > 0 {
        println!("\n--- WRONG RESULTS ---");
        for r in &results {
            if r.verdict == Verdict::Wrong {
                println!("  ❌ {} — {}", r.label, r.detail);
            }
        }
        println!();
        std::process::exit(1);
    } else {
        println!("\nAll evaluated results are numerically correct!");
    }
}

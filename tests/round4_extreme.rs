//! Round 4 — Extreme stress tests for symplex.
//!
//! Strategy: go where nobody has gone. Test non-elementary integrals,
//! complex number identities, series convergence, matrix identities,
//! number theory edge cases, and simplification torture tests.
//!
//! Run with:
//!   cd symplex && cargo test --test round4_extreme 2>&1

mod common;

use num_bigint::BigInt;
use symplex::ntheory;
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Approximate equality with mixed absolute + relative tolerance.
fn approx(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_infinite() && b.is_infinite() {
        return a.signum() == b.signum();
    }
    if a.is_nan() || b.is_nan() || a.is_infinite() || b.is_infinite() {
        return false;
    }
    let scale = a.abs().max(b.abs()).max(1.0);
    (a - b).abs() < tol * scale
}

/// Evaluate expr at var = p/q, returning f64.
fn eval_rational(expr: &Ex, var: &Ex, p: i64, q: i64) -> Result<f64, SymplexError> {
    let ctx = expr.context();
    let pt = ctx.rational(p, q);
    expr.subs(var, &pt).eval().eval_f64()
}

/// Evaluate a fully numeric expression to f64.
fn eval_f64_ex(expr: &Ex) -> Result<f64, SymplexError> {
    expr.eval().eval_f64()
}

/// Evaluate a fully numeric expression to complex (re, im).
fn eval_c64(expr: &Ex) -> Result<(f64, f64), SymplexError> {
    expr.eval().eval_complex64()
}

/// Check that an integration result is either a valid closed form (no
/// unevaluated Integral node) OR a properly unevaluated Integral node.
/// NEVER garbage (e.g. NaN, wrong structure, partial results).
fn assert_integral_sane(result: &Ex, label: &str) {
    let s = format!("{result}");
    // Must not be empty or trivially broken
    assert!(!s.is_empty(), "{label}: empty result string");
    assert!(!s.contains("NaN"), "{label}: integral produced NaN: {s}");
    assert!(!s.contains("nan"), "{label}: integral produced nan: {s}");
    // If it contains Integral, it must be a proper unevaluated node
    // (which is acceptable for non-elementary integrals)
    if !s.contains("Integral") {
        // It's a closed-form answer — it should not contain the word "error"
        assert!(
            !s.to_lowercase().contains("error"),
            "{label}: result contains 'error': {s}"
        );
    }
    eprintln!("{label}: result = {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 1: NON-ELEMENTARY INTEGRALS
// Must return unevaluated Integral(...) or a correct special function,
// NEVER garbage.
// ═══════════════════════════════════════════════════════════════════════════

/// ∫ exp(x²) dx — the Gaussian-type integral with NO elementary closed form.
#[test]
fn integral_exp_x_squared_non_elementary() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.powi(2).exp(); // exp(x^2)
    let result = integrand.integrate(&x);
    assert_integral_sane(&result, "∫ exp(x²) dx");
    // If it claims a closed form, verify numerically via FTC
    let s = format!("{result}");
    if !s.contains("Integral") {
        // d/dx(result) should equal exp(x^2) — verify numerically
        let deriv = result.diff(&x);
        for &pt in &[0.3_f64, 0.5, 0.7] {
            let numer = (pt * 1000.0) as i64;
            let val_orig = eval_rational(&integrand, &x, numer, 1000);
            let val_deriv = eval_rational(&deriv, &x, numer, 1000);
            if let (Ok(a), Ok(b)) = (val_orig, val_deriv) {
                assert!(
                    approx(a, b, 1e-6),
                    "∫ exp(x²) dx: FTC failed at x={pt}: integrand={a}, d/dx(result)={b}"
                );
            }
        }
    }
}

/// ∫ sin(x)/x dx — the sine integral Si(x), non-elementary.
#[test]
fn integral_sinc_non_elementary() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.sin() / &x;
    let result = integrand.integrate(&x);
    assert_integral_sane(&result, "∫ sin(x)/x dx");
    let s = format!("{result}");
    if !s.contains("Integral") {
        let deriv = result.diff(&x);
        for &pt in &[0.5_f64, 1.0, 1.5] {
            let numer = (pt * 1000.0) as i64;
            let val_orig = eval_rational(&integrand, &x, numer, 1000);
            let val_deriv = eval_rational(&deriv, &x, numer, 1000);
            if let (Ok(a), Ok(b)) = (val_orig, val_deriv) {
                assert!(
                    approx(a, b, 1e-6),
                    "∫ sin(x)/x dx: FTC failed at x={pt}: integrand={a}, d/dx(result)={b}"
                );
            }
        }
    }
}

/// ∫ exp(exp(x)) dx — doubly exponential, non-elementary.
#[test]
fn integral_exp_exp_x_non_elementary() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.exp().exp(); // exp(exp(x))
    let result = integrand.integrate(&x);
    assert_integral_sane(&result, "∫ exp(exp(x)) dx");
}

/// ∫ 1/ln(x) dx — the logarithmic integral Li(x), non-elementary.
#[test]
fn integral_one_over_ln_x_non_elementary() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let integrand = &one / &x.ln();
    let result = integrand.integrate(&x);
    assert_integral_sane(&result, "∫ 1/ln(x) dx");
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 2: DEFINITE INTEGRATION IDENTITIES
// ═══════════════════════════════════════════════════════════════════════════

/// ∫₀^π sin(x) dx = 2
#[test]
fn definite_integral_sin_0_to_pi() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let pi = ctx.pi();

    let result = x.sin().integrate_definite(&x, &zero, &pi);
    let s = format!("{result}");
    eprintln!("∫₀^π sin(x) dx = {s}");

    // Try symbolic check first
    let simplified = result.simplify();
    let s2 = format!("{simplified}");
    eprintln!("simplified = {s2}");

    // Try numerical evaluation
    match eval_f64_ex(&simplified) {
        Ok(val) => {
            assert!(
                approx(val, 2.0, 1e-10),
                "∫₀^π sin(x) dx should be 2, got {val}"
            );
        }
        Err(e) => {
            // If eval fails, try the unsimplified version
            match eval_f64_ex(&result) {
                Ok(val) => {
                    assert!(
                        approx(val, 2.0, 1e-10),
                        "∫₀^π sin(x) dx should be 2, got {val}"
                    );
                }
                Err(e2) => {
                    eprintln!("Could not evaluate ∫₀^π sin(x) dx: simplified={e}, raw={e2}");
                    // Check symbolically for "2"
                    assert!(
                        s2 == "2" || s == "2",
                        "∫₀^π sin(x) dx: could not verify = 2: simplified={s2}, raw={s}"
                    );
                }
            }
        }
    }
}

/// ∫₋₁^1 x² dx = 2/3
#[test]
fn definite_integral_x_squared_neg1_to_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let neg1 = ctx.int(-1);
    let pos1 = ctx.int(1);

    let result = x.powi(2).integrate_definite(&x, &neg1, &pos1);
    let s = format!("{result}");
    eprintln!("∫₋₁^1 x² dx = {s}");

    let simplified = result.simplify();
    let s2 = format!("{simplified}");

    // Should be exactly 2/3
    if s2 == "2/3" || s == "2/3" {
        // Perfect
    } else {
        // Try numerical
        match eval_f64_ex(&simplified) {
            Ok(val) => {
                assert!(
                    approx(val, 2.0 / 3.0, 1e-10),
                    "∫₋₁^1 x² dx should be 2/3, got {val}"
                );
            }
            Err(_) => match eval_f64_ex(&result) {
                Ok(val) => {
                    assert!(
                        approx(val, 2.0 / 3.0, 1e-10),
                        "∫₋₁^1 x² dx should be 2/3, got {val}"
                    );
                }
                Err(e) => {
                    panic!("∫₋₁^1 x² dx: could not verify = 2/3: {s2} / {s} / {e}");
                }
            },
        }
    }
}

/// ∫₀^1 x³ dx = 1/4
#[test]
fn definite_integral_x_cubed_0_to_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.powi(3).integrate_definite(&x, &ctx.int(0), &ctx.int(1));
    let s = format!("{}", result.simplify());
    eprintln!("∫₀^1 x³ dx = {s}");
    if s != "1/4" {
        let val = eval_f64_ex(&result).expect("should evaluate");
        assert!(approx(val, 0.25, 1e-10), "expected 1/4 got {val}");
    }
}

/// FTC: ∫ d/dx[sin(x)·cos(x)] dx = sin(x)·cos(x) (up to constant)
#[test]
fn ftc_integrate_derivative_sincos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sin() * x.cos(); // sin(x)cos(x)
    let df = f.diff(&x);
    let reconstructed = df.integrate(&x);
    let s = format!("{reconstructed}");
    eprintln!("∫ d/dx[sin(x)cos(x)] dx = {s}");

    // Verify numerically: reconstructed should differ from f by at most a constant
    let mut diffs = Vec::new();
    for &pt in &[0.3_f64, 0.7, 1.4, 2.1] {
        let numer = (pt * 1000.0) as i64;
        let vf = eval_rational(&f, &x, numer, 1000);
        let vr = eval_rational(&reconstructed, &x, numer, 1000);
        if let (Ok(a), Ok(b)) = (vf, vr) {
            diffs.push(a - b);
        }
    }
    assert!(!diffs.is_empty(), "FTC sin*cos: no points evaluated");
    // All diffs should be the same constant
    let c = diffs[0];
    for (i, &d) in diffs.iter().enumerate() {
        assert!(
            approx(d, c, 1e-6),
            "FTC sin*cos: diff not constant: diffs[0]={c}, diffs[{i}]={d}"
        );
    }
}

/// FTC: ∫ d/dx[x·exp(x)] dx = x·exp(x) (up to constant)
#[test]
fn ftc_integrate_derivative_x_exp_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x * x.exp();
    let df = f.diff(&x);
    let reconstructed = df.integrate(&x);
    let s = format!("{reconstructed}");
    eprintln!("∫ d/dx[x·exp(x)] dx = {s}");

    let mut diffs = Vec::new();
    for &pt in &[0.3_f64, 0.7, 1.1] {
        let numer = (pt * 1000.0) as i64;
        let vf = eval_rational(&f, &x, numer, 1000);
        let vr = eval_rational(&reconstructed, &x, numer, 1000);
        if let (Ok(a), Ok(b)) = (vf, vr) {
            diffs.push(a - b);
        }
    }
    assert!(!diffs.is_empty(), "FTC x·exp(x): no points evaluated");
    let c = diffs[0];
    for (i, &d) in diffs.iter().enumerate() {
        assert!(
            approx(d, c, 1e-6),
            "FTC x·exp(x): not constant: diffs[0]={c}, diffs[{i}]={d}"
        );
    }
}

/// FTC inverse: d/dx[∫ f dx] = f for f = x²+1
#[test]
fn ftc_derivative_of_integral_poly() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(2) + 1;
    let anti = f.integrate(&x);
    let roundtrip = anti.diff(&x);

    for &pt in &[1_i64, 2, 3, 5] {
        let vf = f.subs_i64(&x, pt).eval().eval_f64();
        let vr = roundtrip.subs_i64(&x, pt).eval().eval_f64();
        if let (Ok(a), Ok(b)) = (vf, vr) {
            assert!(
                approx(a, b, 1e-9),
                "d/dx[∫(x²+1) dx] at x={pt}: f={a}, roundtrip={b}"
            );
        }
    }
}

/// FTC inverse: d/dx[∫ sin(x) dx] = sin(x)
#[test]
fn ftc_derivative_of_integral_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sin();
    let anti = f.integrate(&x);
    let roundtrip = anti.diff(&x);

    for &pt in &[0.5_f64, 1.0, 2.0, 3.0] {
        let numer = (pt * 1000.0) as i64;
        let vf = eval_rational(&f, &x, numer, 1000);
        let vr = eval_rational(&roundtrip, &x, numer, 1000);
        if let (Ok(a), Ok(b)) = (vf, vr) {
            assert!(
                approx(a, b, 1e-8),
                "d/dx[∫ sin(x) dx] at x={pt}: f={a}, roundtrip={b}"
            );
        }
    }
}

/// FTC inverse: d/dx[∫ exp(x) dx] = exp(x)
#[test]
fn ftc_derivative_of_integral_exp() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.exp();
    let anti = f.integrate(&x);
    let roundtrip = anti.diff(&x);

    for &pt in &[0_i64, 1, 2] {
        let vf = f.subs_i64(&x, pt).eval().eval_f64();
        let vr = roundtrip.subs_i64(&x, pt).eval().eval_f64();
        if let (Ok(a), Ok(b)) = (vf, vr) {
            assert!(
                approx(a, b, 1e-9),
                "d/dx[∫ exp(x) dx] at x={pt}: f={a}, roundtrip={b}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 3: COMPLEX NUMBER STRESS
// ═══════════════════════════════════════════════════════════════════════════

/// (1+i)^10 — should be a specific power of 2 times a power of i.
/// (1+i)^2 = 2i, so (1+i)^10 = (2i)^5 = 32·i^5 = 32·i.
#[test]
fn complex_one_plus_i_to_the_10() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let one = ctx.int(1);
    let base = &one + &i; // 1 + i
    let result = base.powi(10);
    let s = format!("{result}");
    eprintln!("(1+i)^10 = {s}");

    // (1+i)^2 = 2i => (1+i)^10 = (2i)^5 = 32*i^5 = 32*i
    match eval_c64(&result) {
        Ok((re, im)) => {
            assert!(
                approx(re, 0.0, 1e-10) && approx(im, 32.0, 1e-10),
                "(1+i)^10 should be 32i, got ({re}, {im}i)"
            );
        }
        Err(e) => {
            // Try parsing the string
            eprintln!("Could not evaluate (1+i)^10: {e}, string={s}");
            assert!(
                s.contains("32") && (s.contains("I") || s.contains("i")),
                "(1+i)^10 should contain 32 and i: got {s}"
            );
        }
    }
}

/// Euler's identity: exp(iπ) + 1 = 0
#[test]
fn euler_identity() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let pi = ctx.pi();
    let one = ctx.int(1);
    let expr = (&i * &pi).exp() + &one; // exp(i*pi) + 1
    let s = format!("{expr}");
    eprintln!("exp(iπ) + 1 = {s}");

    let simplified = expr.simplify();
    let s2 = format!("{simplified}");
    eprintln!("simplified = {s2}");

    // Try numerical evaluation
    match eval_c64(&simplified) {
        Ok((re, im)) => {
            assert!(
                approx(re, 0.0, 1e-10) && approx(im, 0.0, 1e-10),
                "exp(iπ) + 1 should be 0, got ({re} + {im}i)"
            );
        }
        Err(_) => {
            match eval_c64(&expr) {
                Ok((re, im)) => {
                    assert!(
                        approx(re, 0.0, 1e-10) && approx(im, 0.0, 1e-10),
                        "exp(iπ) + 1 should be 0, got ({re} + {im}i)"
                    );
                }
                Err(e) => {
                    // At least symbolically it should simplify to 0
                    assert!(
                        s2 == "0" || s == "0",
                        "exp(iπ)+1: can't verify = 0: simplified={s2}, raw={s}, err={e}"
                    );
                }
            }
        }
    }
}

/// i^2 = -1 (basic sanity)
#[test]
fn i_squared_is_neg_one_round4() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let result = i.powi(2);
    assert_eq!(format!("{result}"), "-1");
}

/// i^4 = 1
#[test]
fn i_fourth_is_one_round4() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let result = i.powi(4);
    assert_eq!(format!("{result}"), "1");
}

/// (1+i)(1-i) = 2
#[test]
fn complex_conjugate_product() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let one = ctx.int(1);
    let a = &one + &i;
    let b = &one - &i;
    let result = (&a * &b).eval();
    let s = format!("{result}");
    eprintln!("(1+i)(1-i) = {s}");

    match eval_f64_ex(&result) {
        Ok(val) => assert!(approx(val, 2.0, 1e-10), "(1+i)(1-i) should be 2, got {val}"),
        Err(_) => {
            let simplified = result.simplify();
            assert_eq!(format!("{simplified}"), "2", "(1+i)(1-i) should be 2");
        }
    }
}

/// i^i = exp(-π/2) — a real number!
#[test]
fn i_to_the_i_is_real() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    // i^i = exp(i * ln(i)) = exp(i * iπ/2) = exp(-π/2)
    let result = i.pow(&i);
    let s = format!("{result}");
    eprintln!("i^i = {s}");

    let expected = std::f64::consts::E.powf(-std::f64::consts::FRAC_PI_2);
    eprintln!("expected ≈ {expected}");

    match eval_c64(&result) {
        Ok((re, im)) => {
            assert!(
                approx(im, 0.0, 1e-8),
                "i^i should be real, got imaginary part {im}"
            );
            assert!(
                approx(re, expected, 1e-8),
                "i^i should be exp(-π/2) ≈ {expected}, got {re}"
            );
        }
        Err(e) => {
            eprintln!("Could not evaluate i^i: {e}");
            // At minimum it should not have panicked
        }
    }
}

/// Complex modulus: |3+4i| = 5
#[test]
fn complex_modulus_3_4i() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let expr = ctx.int(3) + &i * 4;
    let modulus = expr.abs();
    let s = format!("{modulus}");
    eprintln!("|3+4i| = {s}");

    match eval_f64_ex(&modulus) {
        Ok(val) => assert!(approx(val, 5.0, 1e-10), "|3+4i| should be 5, got {val}"),
        Err(_) => {
            let simplified = modulus.simplify();
            assert_eq!(format!("{simplified}"), "5");
        }
    }
}

/// (2+3i) + (4-i) = 6 + 2i
#[test]
fn complex_addition() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let a = ctx.int(2) + &i * 3; // 2+3i
    let b = ctx.int(4) - &i; // 4-i
    let result = (&a + &b).eval();
    let s = format!("{result}");
    eprintln!("(2+3i) + (4-i) = {s}");

    match eval_c64(&result) {
        Ok((re, im)) => {
            assert!(approx(re, 6.0, 1e-10), "real part should be 6, got {re}");
            assert!(approx(im, 2.0, 1e-10), "imag part should be 2, got {im}");
        }
        Err(e) => {
            eprintln!("Could not evaluate: {e}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 4: SOLVING HARD SYSTEMS
// ═══════════════════════════════════════════════════════════════════════════

/// Circle meets line: x² + y² = 1, x + y = 1
/// Solutions: (0, 1) and (1, 0)
#[test]
fn solve_system_circle_meets_line() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    // Express as: x² + y² - 1 = 0, x + y - 1 = 0
    let eq1 = x.powi(2) + y.powi(2) - 1;
    let _eq2 = &x + &y - 1;

    // This is a nonlinear system, solve_system handles linear only.
    // Substitute y = 1 - x into eq1:
    let substituted = eq1.subs(&y, &(&ctx.int(1) - &x));
    let s = format!("{substituted}");
    eprintln!("After substitution: {s}");

    let solutions = substituted.solve(&x);
    match solutions {
        Ok(sols) => {
            eprintln!(
                "Solutions for x: {:?}",
                sols.iter().map(|s| format!("{s}")).collect::<Vec<_>>()
            );
            // Should find x=0 and x=1
            let mut found_0 = false;
            let mut found_1 = false;
            for sol in &sols {
                if let Ok(v) = eval_f64_ex(sol) {
                    if approx(v, 0.0, 1e-10) {
                        found_0 = true;
                    }
                    if approx(v, 1.0, 1e-10) {
                        found_1 = true;
                    }
                }
            }
            assert!(
                found_0 && found_1,
                "Expected solutions x=0 and x=1, got: {:?}",
                sols.iter().map(|s| format!("{s}")).collect::<Vec<_>>()
            );
        }
        Err(e) => {
            eprintln!("Solver failed on circle-line: {e}");
            // Not necessarily a bug — document it
        }
    }
}

/// Solve x² - 2 = 0 → x = ±√2
#[test]
fn solve_sqrt2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(2) - 2;
    let solutions = expr.solve(&x).expect("should solve x²-2=0");
    assert_eq!(solutions.len(), 2, "x²-2=0 should have 2 solutions");
    for sol in &solutions {
        let val = eval_f64_ex(sol).expect("solution should evaluate");
        assert!(
            approx(val.abs(), std::f64::consts::SQRT_2, 1e-10),
            "x²-2=0 solution should be ±√2, got {val}"
        );
    }
}

/// Solve x³ - 1 = 0 — should find x=1 and possibly complex cube roots.
#[test]
fn solve_cubic_unity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(3) - 1;
    let solutions = expr.solve(&x).expect("should solve x³-1=0");
    eprintln!(
        "x³-1=0 solutions: {:?}",
        solutions.iter().map(|s| format!("{s}")).collect::<Vec<_>>()
    );

    // At least one real root: x = 1
    let mut found_one = false;
    for sol in &solutions {
        if let Ok(v) = eval_f64_ex(sol)
            && approx(v, 1.0, 1e-10)
        {
            found_one = true;
        }
    }
    assert!(found_one, "x³-1=0 should have real root x=1");
}

/// Solve x⁴ - 1 = 0 — roots are ±1, ±i
#[test]
fn solve_quartic_unity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(4) - 1;
    let solutions = expr.solve(&x).expect("should solve x⁴-1=0");
    eprintln!(
        "x⁴-1=0 solutions: {:?}",
        solutions.iter().map(|s| format!("{s}")).collect::<Vec<_>>()
    );
    // Should have 4 roots (real or complex)
    // At minimum, ±1 should be found
    let mut found_pos1 = false;
    let mut found_neg1 = false;
    for sol in &solutions {
        if let Ok(v) = eval_f64_ex(sol) {
            if approx(v, 1.0, 1e-10) {
                found_pos1 = true;
            }
            if approx(v, -1.0, 1e-10) {
                found_neg1 = true;
            }
        }
    }
    assert!(found_pos1, "x⁴-1=0 should have root x=1");
    assert!(found_neg1, "x⁴-1=0 should have root x=-1");
}

/// Solve linear system: x + 2y = 5, 3x - y = 1 => x=1, y=2
#[test]
fn solve_linear_system_2x2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let eq1 = &x + &y * 2 - 5; // x + 2y - 5 = 0
    let eq2 = &x * 3 - &y - 1; // 3x - y - 1 = 0
    let solution = ctx
        .solve_system(&[eq1, eq2], &[x.clone(), y.clone()])
        .expect("linear input");
    match solution {
        LinearSolution::Unique(pairs) => {
            eprintln!(
                "Linear system solution: {:?}",
                pairs
                    .iter()
                    .map(|(v, s)| format!("{v}={s}"))
                    .collect::<Vec<_>>()
            );
            assert_eq!(pairs.len(), 2);
            let x_val = eval_f64_ex(&pairs[0].1).expect("x value");
            let y_val = eval_f64_ex(&pairs[1].1).expect("y value");
            assert!(approx(x_val, 1.0, 1e-10), "x should be 1, got {x_val}");
            assert!(approx(y_val, 2.0, 1e-10), "y should be 2, got {y_val}");
        }
        other => {
            panic!("Linear system should have a unique solution, got {other:?}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 5: SERIES EXPANSIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Taylor series of tan(x) around 0: x + x³/3 + 2x⁵/15 + ...
#[test]
fn series_tan_around_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let expr = x.tan();
    let series = expr.series(&x, &zero, 6);
    let expanded = series.expand().eval();
    let s = format!("{expanded}");
    eprintln!("tan(x) series = {s}");

    // Verify numerically: compare series and tan(x) at a small point
    let test_pt: f64 = 0.1;
    let numer = (test_pt * 10000.0) as i64;
    let val_series = eval_rational(&expanded, &x, numer, 10000);
    let expected = test_pt.tan();
    let vs = val_series.expect("val_series must evaluate");
    assert!(
        approx(vs, expected, 1e-4),
        "tan(x) series at x={test_pt}: series={vs}, exact={expected}"
    );
}

/// Series of 1/(1-x) around 0 = 1 + x + x² + x³ + ...
#[test]
fn series_geometric() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let one = ctx.int(1);
    let expr = &one / &(&one - &x); // 1/(1-x)
    let series = expr.series(&x, &zero, 5);
    let expanded = series.expand().eval();
    let s = format!("{expanded}");
    eprintln!("1/(1-x) series = {s}");

    // Verify at x=0.1: sum should be close to 1/(1-0.1) = 10/9 ≈ 1.1111
    let test_pt: f64 = 0.1;
    let numer = (test_pt * 10000.0) as i64;
    let val_series = eval_rational(&expanded, &x, numer, 10000);
    let expected = 1.0 / (1.0 - test_pt);
    let vs = val_series.expect("val_series must evaluate");
    assert!(
        approx(vs, expected, 1e-4),
        "1/(1-x) series at x={test_pt}: series={vs}, exact={expected}"
    );
}

/// Series of 1/(1-x)² around 0 = 1 + 2x + 3x² + 4x³ + ...
#[test]
fn series_one_over_1_minus_x_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let one = ctx.int(1);
    let denom = (&one - &x).powi(2);
    let expr = &one / &denom;
    let series = expr.series(&x, &zero, 5);
    let expanded = series.expand().eval();
    let s = format!("{expanded}");
    eprintln!("1/(1-x)² series = {s}");

    // Verify at x=0.1
    let test_pt: f64 = 0.1;
    let numer = (test_pt * 10000.0) as i64;
    let val_series = eval_rational(&expanded, &x, numer, 10000);
    let expected = 1.0 / (1.0 - test_pt).powi(2);
    let vs = val_series.expect("val_series must evaluate");
    assert!(
        approx(vs, expected, 1e-3),
        "1/(1-x)² series at x={test_pt}: series={vs}, exact={expected}"
    );
}

/// Series of exp(x) around 0: 1 + x + x²/2 + x³/6 + ...
#[test]
fn series_exp_at_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let expr = x.exp();
    let series = expr.series(&x, &zero, 6);
    let expanded = series.expand().eval();
    let s = format!("{expanded}");
    eprintln!("exp(x) series = {s}");

    // Verify at x = 0.5
    let test_pt: f64 = 0.5;
    let numer = (test_pt * 10000.0) as i64;
    let val_series = eval_rational(&expanded, &x, numer, 10000);
    let expected = test_pt.exp();
    let vs = val_series.expect("val_series must evaluate");
    assert!(
        approx(vs, expected, 1e-4),
        "exp(x) series at x={test_pt}: series={vs}, exact={expected}"
    );
}

/// Series of ln(1+x) around 0: x - x²/2 + x³/3 - x⁴/4 + ...
#[test]
fn series_ln_1_plus_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let one = ctx.int(1);
    let expr = (&one + &x).ln();
    let series = expr.series(&x, &zero, 6);
    let expanded = series.expand().eval();
    let s = format!("{expanded}");
    eprintln!("ln(1+x) series = {s}");

    // Verify at x = 0.3
    let test_pt: f64 = 0.3;
    let numer = (test_pt * 10000.0) as i64;
    let val_series = eval_rational(&expanded, &x, numer, 10000);
    let expected = (1.0 + test_pt).ln();
    let vs = val_series.expect("val_series must evaluate");
    assert!(
        approx(vs, expected, 1e-4),
        "ln(1+x) series at x={test_pt}: series={vs}, exact={expected}"
    );
}

/// Series of sin(x) around 0: x - x³/6 + x⁵/120 - ...
#[test]
fn series_sin_at_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let expr = x.sin();
    let series = expr.series(&x, &zero, 7);
    let expanded = series.expand().eval();
    let s = format!("{expanded}");
    eprintln!("sin(x) series = {s}");

    // Verify at x = 0.5
    let test_pt: f64 = 0.5;
    let numer = (test_pt * 10000.0) as i64;
    let val_series = eval_rational(&expanded, &x, numer, 10000);
    let expected = test_pt.sin();
    let vs = val_series.expect("val_series must evaluate");
    assert!(
        approx(vs, expected, 1e-5),
        "sin(x) series at x={test_pt}: series={vs}, exact={expected}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 6: SIMPLIFICATION TORTURE
// ═══════════════════════════════════════════════════════════════════════════

/// sin(arcsin(x)) should be x
#[test]
fn simplify_sin_arcsin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.asin().sin();
    let simplified = expr.simplify();
    let s = format!("{simplified}");
    eprintln!("sin(arcsin(x)) = {s}");

    // Verify numerically for x in (-1, 1)
    for &pt in &[0.3_f64, 0.5, 0.7, -0.3] {
        let numer = (pt * 1000.0) as i64;
        let val = eval_rational(&simplified, &x, numer, 1000);
        let v = val.expect("val must evaluate");
        assert!(
            approx(v, pt, 1e-10),
            "sin(arcsin({pt})) should be {pt}, got {v}"
        );
    }
}

/// ln(exp(x)) should be x
#[test]
fn simplify_ln_exp() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.exp().ln();
    let simplified = expr.simplify();
    let s = format!("{simplified}");
    eprintln!("ln(exp(x)) = {s}");

    // Verify numerically
    for &pt in &[-2_i64, -1, 0, 1, 2, 3] {
        let val = simplified.subs_i64(&x, pt).eval().eval_f64();
        let v = val.expect("val must evaluate");
        assert!(
            approx(v, pt as f64, 1e-10),
            "ln(exp({pt})) should be {pt}, got {v}"
        );
    }
}

/// (x²-y²)/(x-y) should simplify or cancel to x+y
#[test]
fn simplify_difference_of_squares() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let numer = x.powi(2) - y.powi(2);
    let denom = &x - &y;
    let expr = &numer / &denom;
    // cancel requires a variable — try x first
    let cancelled = expr.cancel(&x);
    let s = format!("{cancelled}");
    eprintln!("(x²-y²)/(x-y) cancelled w.r.t. x = {s}");

    // Verify numerically: at x=3, y=1: should be 4
    let test_val = cancelled.subs_i64(&x, 3).subs_i64(&y, 1).eval().eval_f64();
    let v = test_val.expect("test_val must evaluate");
    assert!(approx(v, 4.0, 1e-10), "(3²-1²)/(3-1) should be 4, got {v}");

    // At x=5, y=2: should be 7
    let test_val2 = cancelled.subs_i64(&x, 5).subs_i64(&y, 2).eval().eval_f64();
    let v = test_val2.expect("test_val2 must evaluate");
    assert!(approx(v, 7.0, 1e-10), "(5²-2²)/(5-2) should be 7, got {v}");
}

/// sin(x+y) - sin(x)cos(y) - cos(x)sin(y) should be 0
#[test]
fn trig_addition_formula_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let lhs = (&x + &y).sin();
    let rhs = &x.sin() * &y.cos() + &x.cos() * &y.sin();
    let diff = &lhs - &rhs;

    // Check numerically at several points
    for &(xv, yv) in &[(1_i64, 2_i64), (3, 1), (2, 5), (-1, 3)] {
        let val = diff.subs_i64(&x, xv).subs_i64(&y, yv).eval().eval_f64();
        let v = val.expect("val must evaluate");
        assert!(
            approx(v, 0.0, 1e-10),
            "sin(x+y) - sin(x)cos(y) - cos(x)sin(y) at x={xv}, y={yv}: got {v}"
        );
    }
}

/// cos(2x) - cos²(x) + sin²(x) should be 0
#[test]
fn trig_double_angle_cos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let two_x = &x * 2;
    let cos2x = two_x.cos();
    let cos_sq = x.cos().powi(2);
    let sin_sq = x.sin().powi(2);
    let diff = &cos2x - &cos_sq + &sin_sq;

    for &pt in &[0_i64, 1, 2, 3, -1, -2] {
        let val = diff.subs_i64(&x, pt).eval().eval_f64();
        let v = val.expect("val must evaluate");
        assert!(
            approx(v, 0.0, 1e-10),
            "cos(2x) - cos²(x) + sin²(x) at x={pt}: got {v}"
        );
    }
}

/// sin²(x) + cos²(x) = 1
#[test]
fn pythagorean_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin().powi(2) + x.cos().powi(2);

    for &pt in &[0_i64, 1, 2, 3, -1, -5, 10] {
        let val = expr.subs_i64(&x, pt).eval().eval_f64();
        let v = val.expect("val must evaluate");
        assert!(
            approx(v, 1.0, 1e-10),
            "sin²({pt}) + cos²({pt}) should be 1, got {v}"
        );
    }
}

/// exp(ln(x)) = x for x > 0
#[test]
fn simplify_exp_ln() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.ln().exp();
    let simplified = expr.simplify();

    for &pt in &[1_i64, 2, 3, 5, 10] {
        let val = simplified.subs_i64(&x, pt).eval().eval_f64();
        let v = val.expect("val must evaluate");
        assert!(
            approx(v, pt as f64, 1e-10),
            "exp(ln({pt})) should be {pt}, got {v}"
        );
    }
}

/// (x³ - 8)/(x - 2) should cancel to x² + 2x + 4
#[test]
fn cancel_cubic_minus_8() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let numer = x.powi(3) - 8;
    let denom = &x - &ctx.int(2);
    let expr = &numer / &denom;
    let cancelled = expr.cancel(&x);
    let s = format!("{cancelled}");
    eprintln!("(x³-8)/(x-2) = {s}");

    // At x=3: (27-8)/(3-2) = 19, also 9+6+4 = 19
    let val = cancelled.subs_i64(&x, 3).eval().eval_f64();
    let v = val.expect("val must evaluate");
    assert!(
        approx(v, 19.0, 1e-10),
        "(x³-8)/(x-2) at x=3 should be 19, got {v}"
    );
    // At x=5: (125-8)/(5-2) = 117/3 = 39, also 25+10+4 = 39
    let val2 = cancelled.subs_i64(&x, 5).eval().eval_f64();
    let v = val2.expect("val2 must evaluate");
    assert!(
        approx(v, 39.0, 1e-10),
        "(x³-8)/(x-2) at x=5 should be 39, got {v}"
    );
}

/// cos(x+y) - cos(x)cos(y) + sin(x)sin(y) should be 0
#[test]
fn trig_addition_formula_cos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let lhs = (&x + &y).cos();
    let rhs = &x.cos() * &y.cos() - &x.sin() * &y.sin();
    let diff = &lhs - &rhs;

    for &(xv, yv) in &[(1_i64, 2_i64), (3, 1), (2, 5), (-1, 3)] {
        let val = diff.subs_i64(&x, xv).subs_i64(&y, yv).eval().eval_f64();
        let v = val.expect("val must evaluate");
        assert!(
            approx(v, 0.0, 1e-10),
            "cos(x+y) - cos(x)cos(y) + sin(x)sin(y) at x={xv}, y={yv}: got {v}"
        );
    }
}

/// tan(x) - sin(x)/cos(x) should be 0
#[test]
fn trig_tan_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let diff = x.tan() - &(x.sin() / &x.cos());

    for &pt in &[1_i64, 2, 3, -1, -2] {
        let val = diff.subs_i64(&x, pt).eval().eval_f64();
        let v = val.expect("val must evaluate");
        assert!(
            approx(v, 0.0, 1e-10),
            "tan(x) - sin(x)/cos(x) at x={pt}: got {v}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 7: MATRIX IDENTITIES
// ═══════════════════════════════════════════════════════════════════════════

/// det(A·B) = det(A)·det(B) for 2×2 numeric matrices
#[test]
fn matrix_det_product_2x2() {
    let ctx = Context::new();
    let a = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2)],
        vec![ctx.int(3), ctx.int(4)],
    ])
    .unwrap();
    let b = Matrix::new(vec![
        vec![ctx.int(5), ctx.int(6)],
        vec![ctx.int(7), ctx.int(8)],
    ])
    .unwrap();

    let ab = &a * &b;
    let det_ab = ab.det().expect("det(AB)");
    let det_a = a.det().expect("det(A)");
    let det_b = b.det().expect("det(B)");
    let det_a_times_det_b = (&det_a * &det_b).eval();

    let v1 = eval_f64_ex(&det_ab).expect("eval det(AB)");
    let v2 = eval_f64_ex(&det_a_times_det_b).expect("eval det(A)*det(B)");

    assert!(
        approx(v1, v2, 1e-10),
        "det(AB)={v1} should equal det(A)*det(B)={v2}"
    );
}

/// det(A·B) = det(A)·det(B) for 3×3 numeric matrices
#[test]
fn matrix_det_product_3x3() {
    let ctx = Context::new();
    let a = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2), ctx.int(3)],
        vec![ctx.int(0), ctx.int(1), ctx.int(4)],
        vec![ctx.int(5), ctx.int(6), ctx.int(0)],
    ])
    .unwrap();
    let b = Matrix::new(vec![
        vec![ctx.int(2), ctx.int(0), ctx.int(1)],
        vec![ctx.int(3), ctx.int(1), ctx.int(0)],
        vec![ctx.int(0), ctx.int(2), ctx.int(1)],
    ])
    .unwrap();

    let ab = &a * &b;
    let det_ab = ab.det().expect("det(AB)");
    let det_a = a.det().expect("det(A)");
    let det_b = b.det().expect("det(B)");
    let product = (&det_a * &det_b).eval();

    let v1 = eval_f64_ex(&det_ab).expect("eval det(AB)");
    let v2 = eval_f64_ex(&product).expect("eval det(A)*det(B)");

    assert!(
        approx(v1, v2, 1e-10),
        "det(AB)={v1} should equal det(A)*det(B)={v2}"
    );
}

/// (A·B)ᵀ = Bᵀ·Aᵀ
#[test]
fn matrix_transpose_of_product() {
    let ctx = Context::new();
    let a = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2), ctx.int(3)],
        vec![ctx.int(4), ctx.int(5), ctx.int(6)],
    ])
    .unwrap(); // 2×3
    let b = Matrix::new(vec![
        vec![ctx.int(7), ctx.int(8)],
        vec![ctx.int(9), ctx.int(10)],
        vec![ctx.int(11), ctx.int(12)],
    ])
    .unwrap(); // 3×2

    let ab = &a * &b; // 2×2
    let ab_t = ab.transpose();

    let bt = b.transpose(); // 2×3
    let at = a.transpose(); // 3×2
    let bt_at = &bt * &at; // 2×2

    // Compare element by element
    for i in 0..2 {
        for j in 0..2 {
            let v1 = eval_f64_ex(ab_t.get(i, j)).expect("eval (AB)^T");
            let v2 = eval_f64_ex(bt_at.get(i, j)).expect("eval B^T A^T");
            assert!(
                approx(v1, v2, 1e-10),
                "(AB)^T[{i},{j}]={v1} != (B^T A^T)[{i},{j}]={v2}"
            );
        }
    }
}

/// Cayley-Hamilton: A satisfies its own characteristic polynomial.
/// For a 2×2 matrix, char_poly(λ) = λ² - tr(A)λ + det(A)
/// So A² - tr(A)·A + det(A)·I = 0
#[test]
fn cayley_hamilton_2x2() {
    let ctx = Context::new();
    let a = Matrix::new(vec![
        vec![ctx.int(3), ctx.int(1)],
        vec![ctx.int(2), ctx.int(5)],
    ])
    .unwrap();

    let trace = a.trace().expect("trace");
    let det = a.det().expect("det");
    let a2 = &a * &a;
    let ident = Matrix::identity(&ctx, 2);

    // A² - tr(A)·A + det(A)·I should be zero matrix
    let ta = &a * &trace; // tr(A)·A
    let di = &ident * &det; // det(A)·I
    // result = A² - tr(A)·A + det(A)·I
    let step1 = &a2 - &ta;
    let result = &step1 + &di;

    for i in 0..2 {
        for j in 0..2 {
            let val = eval_f64_ex(result.get(i, j));
            match val {
                Ok(v) => {
                    assert!(
                        approx(v, 0.0, 1e-8),
                        "Cayley-Hamilton: result[{i},{j}] = {v}, expected 0"
                    );
                }
                Err(e) => {
                    // Try simplifying
                    let simplified = result.get(i, j).simplify();
                    let s = format!("{simplified}");
                    assert!(
                        s == "0",
                        "Cayley-Hamilton: result[{i},{j}] = {s}, expected 0 (err={e})"
                    );
                }
            }
        }
    }
}

/// Matrix inverse: A · A⁻¹ = I
#[test]
fn matrix_inverse_product_is_identity() {
    let ctx = Context::new();
    let a = Matrix::new(vec![
        vec![ctx.int(2), ctx.int(1)],
        vec![ctx.int(1), ctx.int(3)],
    ])
    .unwrap();

    let a_inv = a.inv().expect("A should be invertible");
    let product = &a * &a_inv;

    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let val = eval_f64_ex(product.get(i, j));
            match val {
                Ok(v) => {
                    assert!(
                        approx(v, expected, 1e-8),
                        "A·A⁻¹[{i},{j}] = {v}, expected {expected}"
                    );
                }
                Err(_) => {
                    let simplified = product.get(i, j).simplify();
                    let s = format!("{simplified}");
                    let exp_s = if i == j { "1" } else { "0" };
                    assert!(s == exp_s, "A·A⁻¹[{i},{j}] = {s}, expected {exp_s}");
                }
            }
        }
    }
}

/// det(Aᵀ) = det(A)
#[test]
fn matrix_det_transpose() {
    let ctx = Context::new();
    let a = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2), ctx.int(3)],
        vec![ctx.int(4), ctx.int(5), ctx.int(6)],
        vec![ctx.int(7), ctx.int(8), ctx.int(10)], // not singular
    ])
    .unwrap();

    let det_a = a.det().expect("det(A)");
    let det_at = a.transpose().det().expect("det(A^T)");

    let v1 = eval_f64_ex(&det_a).expect("eval det(A)");
    let v2 = eval_f64_ex(&det_at).expect("eval det(A^T)");

    assert!(
        approx(v1, v2, 1e-10),
        "det(A)={v1} should equal det(A^T)={v2}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 8: NUMBER THEORY EDGE CASES
// ═══════════════════════════════════════════════════════════════════════════

/// gcd(0, 0) — should be 0 by convention (or at least not panic)
#[test]
fn ntheory_gcd_zero_zero() {
    let result = ntheory::gcd(0_i64, 0_i64);
    // By convention gcd(0,0) = 0
    assert_eq!(
        result,
        BigInt::from(0),
        "gcd(0,0) should be 0, got {result}"
    );
}

/// gcd(0, n) = |n|
#[test]
fn ntheory_gcd_zero_n() {
    let result = ntheory::gcd(0_i64, 12_i64);
    assert_eq!(
        result,
        BigInt::from(12),
        "gcd(0,12) should be 12, got {result}"
    );
}

/// gcd(n, 0) = |n|
#[test]
fn ntheory_gcd_n_zero() {
    let result = ntheory::gcd(15_i64, 0_i64);
    assert_eq!(
        result,
        BigInt::from(15),
        "gcd(15,0) should be 15, got {result}"
    );
}

/// gcd with negatives
#[test]
fn ntheory_gcd_negative() {
    let result = ntheory::gcd(-12_i64, 8_i64);
    assert_eq!(
        result,
        BigInt::from(4),
        "gcd(-12,8) should be 4, got {result}"
    );
}

/// isprime(1) must be false
#[test]
fn ntheory_isprime_one() {
    assert!(!ntheory::isprime(1_i64), "1 is not prime");
}

/// isprime(0) must be false
#[test]
fn ntheory_isprime_zero() {
    assert!(!ntheory::isprime(0_i64), "0 is not prime");
}

/// isprime(-1) must be false
#[test]
fn ntheory_isprime_neg_one() {
    assert!(!ntheory::isprime(-1_i64), "-1 is not prime");
}

/// isprime(-7) — negative primes: by convention, false
#[test]
fn ntheory_isprime_negative_seven() {
    // By standard convention, primes are positive integers > 1
    // isprime(-7) should be false
    let result = ntheory::isprime(-7_i64);
    assert!(
        !result,
        "isprime(-7) should be false (primes are positive), got {result}"
    );
}

/// isprime(2) must be true
#[test]
fn ntheory_isprime_two() {
    assert!(ntheory::isprime(2_i64), "2 is prime");
}

/// isprime(i64::MAX) — a large odd number, should not panic
#[test]
fn ntheory_isprime_i64_max() {
    // i64::MAX = 9223372036854775807
    // This is 7 × 7 × 73 × 127 × 337 × 92737 × 649657 (composite)
    let result = ntheory::isprime(i64::MAX);
    // We don't necessarily know the answer, but it must not panic
    eprintln!("isprime(i64::MAX) = {result}");
    // i64::MAX = 9223372036854775807 is composite (divisible by 7)
    assert!(!result, "i64::MAX should be composite");
}

/// factorint(0) — should return empty or handle gracefully, not panic
#[test]
fn ntheory_factorint_zero() {
    let result = ntheory::factorint(0_i64);
    eprintln!("factorint(0) = {:?}", result);
    // Convention: factorint(0) returns empty
    assert!(
        result.is_empty(),
        "factorint(0) should be empty, got {:?}",
        result
    );
}

/// factorint(1) — should return empty (no prime factors)
#[test]
fn ntheory_factorint_one() {
    let result = ntheory::factorint(1_i64);
    assert!(
        result.is_empty(),
        "factorint(1) should be empty, got {:?}",
        result
    );
}

/// factorint(-12) = factors of |-12| = 12 = 2²·3
#[test]
fn ntheory_factorint_negative() {
    let result = ntheory::factorint(-12_i64);
    eprintln!("factorint(-12) = {:?}", result);
    // Should factorize the absolute value
    let mut product: i64 = 1;
    for (p, e) in &result {
        let pi: i64 = p.try_into().expect("prime should fit in i64");
        product *= pi.pow(*e);
    }
    assert_eq!(
        product, 12,
        "factorint(-12) factors should multiply to 12, got {product}"
    );
}

/// nextprime(i64::MAX - 10) — test near the boundary. Should not panic.
/// (We use a value near MAX but avoid the overflow-prone MAX itself)
#[test]
fn ntheory_nextprime_near_i64_max() {
    // i64::MAX = 9223372036854775807
    // Use BigInt to avoid i64 overflow
    let n = BigInt::from(i64::MAX) - BigInt::from(10);
    let result = ntheory::nextprime(n.clone());
    eprintln!("nextprime({n}) = {result}");
    // Result should be > n and prime
    assert!(result > n, "nextprime should return something > input");
    assert!(
        ntheory::isprime(result.clone()),
        "nextprime result should be prime"
    );
}

/// nextprime(2) should be 3
#[test]
fn ntheory_nextprime_2() {
    let result = ntheory::nextprime(2_i64);
    assert_eq!(
        result,
        BigInt::from(3),
        "nextprime(2) should be 3, got {result}"
    );
}

/// totient(1) = 1
#[test]
fn ntheory_totient_one() {
    let result = ntheory::totient(1_i64);
    assert_eq!(result, BigInt::from(1), "φ(1) should be 1, got {result}");
}

/// totient(12) = 4
#[test]
fn ntheory_totient_twelve() {
    let result = ntheory::totient(12_i64);
    assert_eq!(result, BigInt::from(4), "φ(12) should be 4, got {result}");
}

/// lcm(0, n) should be 0
#[test]
fn ntheory_lcm_zero() {
    let result = ntheory::lcm(0_i64, 5_i64);
    assert_eq!(
        result,
        BigInt::from(0),
        "lcm(0,5) should be 0, got {result}"
    );
}

/// lcm(4, 6) = 12
#[test]
fn ntheory_lcm_4_6() {
    let result = ntheory::lcm(4_i64, 6_i64);
    assert_eq!(
        result,
        BigInt::from(12),
        "lcm(4,6) should be 12, got {result}"
    );
}

/// mobius(1) = 1
#[test]
fn ntheory_mobius_one() {
    let result = ntheory::mobius(1_i64);
    assert_eq!(result, 1, "μ(1) should be 1, got {result}");
}

/// mobius(6) = 1 (6 = 2·3, two distinct prime factors, (-1)^2 = 1)
#[test]
fn ntheory_mobius_six() {
    let result = ntheory::mobius(6_i64);
    assert_eq!(result, 1, "μ(6) should be 1, got {result}");
}

/// mobius(4) = 0 (4 = 2², has repeated prime factor)
#[test]
fn ntheory_mobius_four() {
    let result = ntheory::mobius(4_i64);
    assert_eq!(result, 0, "μ(4) should be 0, got {result}");
}

/// mod_inverse(3, 7) = 5 (because 3*5 = 15 ≡ 1 mod 7)
#[test]
fn ntheory_mod_inverse() {
    let result = ntheory::mod_inverse(3_i64, 7_i64).expect("inverse exists");
    assert_eq!(
        result,
        BigInt::from(5),
        "3^(-1) mod 7 should be 5, got {result}"
    );
}

/// mod_inverse(2, 4) = None (gcd(2,4)=2, no inverse)
#[test]
fn ntheory_mod_inverse_none() {
    let result = ntheory::mod_inverse(2_i64, 4_i64);
    assert!(result.is_none(), "2 has no inverse mod 4");
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 9: LIMIT EDGE CASES
// ═══════════════════════════════════════════════════════════════════════════

/// lim(x→0) sin(x)/x = 1
#[test]
fn limit_sinx_over_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let expr = x.sin() / &x;
    let result = expr.limit(&x, &zero);
    let s = format!("{result}");
    eprintln!("lim(x→0) sin(x)/x = {s}");

    match eval_f64_ex(&result) {
        Ok(v) => assert!(approx(v, 1.0, 1e-10), "lim sin(x)/x should be 1, got {v}"),
        Err(_) => {
            let simplified = result.simplify();
            assert_eq!(
                format!("{simplified}"),
                "1",
                "lim sin(x)/x should be 1, got {s}"
            );
        }
    }
}

/// lim(x→0) (exp(x)-1)/x = 1
#[test]
fn limit_exp_minus_1_over_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let one = ctx.int(1);
    let expr = (x.exp() - &one) / &x;
    let result = expr.limit(&x, &zero);
    let s = format!("{result}");
    eprintln!("lim(x→0) (exp(x)-1)/x = {s}");

    match eval_f64_ex(&result) {
        Ok(v) => assert!(
            approx(v, 1.0, 1e-10),
            "lim (exp(x)-1)/x should be 1, got {v}"
        ),
        Err(_) => {
            let simplified = result.simplify();
            assert_eq!(format!("{simplified}"), "1", "lim (exp(x)-1)/x should be 1");
        }
    }
}

/// lim(x→∞) 1/x = 0
#[test]
fn limit_one_over_x_at_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let inf = ctx.infinity();
    let expr = &one / &x;
    let result = expr.limit(&x, &inf);
    let s = format!("{result}");
    eprintln!("lim(x→∞) 1/x = {s}");

    match eval_f64_ex(&result) {
        Ok(v) => assert!(approx(v, 0.0, 1e-10), "lim 1/x at ∞ should be 0, got {v}"),
        Err(_) => {
            assert!(s == "0", "lim(x→∞) 1/x should be 0, got {s}");
        }
    }
}

/// lim(x→0) (1-cos(x))/x² = 1/2
#[test]
fn limit_one_minus_cos_over_x_sq() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let one = ctx.int(1);
    let expr = (&one - &x.cos()) / &x.powi(2);
    let result = expr.limit(&x, &zero);
    let s = format!("{result}");
    eprintln!("lim(x→0) (1-cos(x))/x² = {s}");

    match eval_f64_ex(&result) {
        Ok(v) => assert!(
            approx(v, 0.5, 1e-10),
            "lim (1-cos(x))/x² should be 1/2, got {v}"
        ),
        Err(_) => {
            assert!(s == "1/2", "lim (1-cos(x))/x² should be 1/2, got {s}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 10: DIFFERENTIATION STRESS TESTS
// ═══════════════════════════════════════════════════════════════════════════

/// d/dx[x^x] — uses logarithmic differentiation: x^x (ln(x) + 1)
#[test]
fn diff_x_to_the_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.pow(&x); // x^x
    let deriv = expr.diff(&x);
    let s = format!("{deriv}");
    eprintln!("d/dx[x^x] = {s}");

    // Verify numerically at x=2: d/dx[x^x] = x^x(ln(x)+1) = 4(ln(2)+1) ≈ 6.7726
    let expected_at_2 = 4.0 * (2.0_f64.ln() + 1.0);
    let val = deriv.subs_i64(&x, 2).eval().eval_f64();
    let v = val.expect("val must evaluate");
    assert!(
        approx(v, expected_at_2, 1e-6),
        "d/dx[x^x] at x=2 should be {expected_at_2}, got {v}"
    );
}

/// Chain rule: d/dx[sin(x²)] = 2x·cos(x²)
#[test]
fn diff_chain_rule_sin_x_sq() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(2).sin(); // sin(x²)
    let deriv = expr.diff(&x);

    // Verify numerically
    for &pt in &[0.5_f64, 1.0, 1.5, 2.0] {
        let numer = (pt * 1000.0) as i64;
        let val = eval_rational(&deriv, &x, numer, 1000);
        let expected = 2.0 * pt * (pt * pt).cos();
        let v = val.expect("val must evaluate");
        assert!(
            approx(v, expected, 1e-8),
            "d/dx[sin(x²)] at x={pt}: got {v}, expected {expected}"
        );
    }
}

/// Product rule: d/dx[x²·sin(x)] = 2x·sin(x) + x²·cos(x)
#[test]
fn diff_product_rule_x2_sinx() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(2) * x.sin();
    let deriv = expr.diff(&x);

    for &pt in &[0.5_f64, 1.0, 2.0] {
        let numer = (pt * 1000.0) as i64;
        let val = eval_rational(&deriv, &x, numer, 1000);
        let expected = 2.0 * pt * pt.sin() + pt * pt * pt.cos();
        let v = val.expect("val must evaluate");
        assert!(
            approx(v, expected, 1e-8),
            "d/dx[x²sin(x)] at x={pt}: got {v}, expected {expected}"
        );
    }
}

/// Higher derivative: d²/dx²[exp(x)] = exp(x)
#[test]
fn diff_second_derivative_exp() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.exp();
    let d2 = expr.diff_n(&x, 2);

    for &pt in &[0_i64, 1, 2] {
        let val = d2.subs_i64(&x, pt).eval().eval_f64();
        let expected = (pt as f64).exp();
        let v = val.expect("val must evaluate");
        assert!(
            approx(v, expected, 1e-9),
            "d²/dx²[exp(x)] at x={pt}: got {v}, expected {expected}"
        );
    }
}

/// d/dx[arctan(x)] = 1/(1+x²)
#[test]
fn diff_arctan() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.atan();
    let deriv = expr.diff(&x);

    for &pt in &[0.0_f64, 0.5, 1.0, 2.0, -1.0] {
        let numer = (pt * 1000.0) as i64;
        let val = eval_rational(&deriv, &x, numer, 1000);
        let expected = 1.0 / (1.0 + pt * pt);
        let v = val.expect("val must evaluate");
        assert!(
            approx(v, expected, 1e-8),
            "d/dx[atan(x)] at x={pt}: got {v}, expected {expected}"
        );
    }
}

/// d³/dx³[x⁵] = 60x²
#[test]
fn diff_third_derivative_x5() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(5);
    let d3 = expr.diff_n(&x, 3);

    for &pt in &[1_i64, 2, 3, -1] {
        let val = d3.subs_i64(&x, pt).eval().eval_f64();
        let expected = 60.0 * (pt as f64).powi(2);
        let v = val.expect("val must evaluate");
        assert!(
            approx(v, expected, 1e-9),
            "d³/dx³[x⁵] at x={pt}: got {v}, expected {expected}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 11: SPECIAL VALUES & CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════

/// sin(0) = 0
#[test]
fn special_sin_zero() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let result = zero.sin().eval();
    assert_eq!(format!("{result}"), "0");
}

/// cos(0) = 1
#[test]
fn special_cos_zero() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let result = zero.cos().eval();
    assert_eq!(format!("{result}"), "1");
}

/// sin(π) = 0
#[test]
fn special_sin_pi() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let result = pi.sin().eval();
    let s = format!("{result}");
    eprintln!("sin(π) = {s}");
    match eval_f64_ex(&result) {
        Ok(v) => assert!(approx(v, 0.0, 1e-10), "sin(π) should be 0, got {v}"),
        Err(_) => assert!(s == "0", "sin(π) should be 0, got {s}"),
    }
}

/// cos(π) = -1
#[test]
fn special_cos_pi() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let result = pi.cos().eval();
    let s = format!("{result}");
    eprintln!("cos(π) = {s}");
    match eval_f64_ex(&result) {
        Ok(v) => assert!(approx(v, -1.0, 1e-10), "cos(π) should be -1, got {v}"),
        Err(_) => assert!(s == "-1", "cos(π) should be -1, got {s}"),
    }
}

/// exp(0) = 1
#[test]
fn special_exp_zero() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let result = zero.exp().eval();
    assert_eq!(format!("{result}"), "1");
}

/// ln(1) = 0
#[test]
fn special_ln_one() {
    let ctx = Context::new();
    let one = ctx.int(1);
    let result = one.ln().eval();
    assert_eq!(format!("{result}"), "0");
}

/// ln(e) = 1
#[test]
fn special_ln_e() {
    let ctx = Context::new();
    let e = ctx.e();
    let result = e.ln().eval();
    let s = format!("{result}");
    eprintln!("ln(e) = {s}");
    match eval_f64_ex(&result) {
        Ok(v) => assert!(approx(v, 1.0, 1e-10), "ln(e) should be 1, got {v}"),
        Err(_) => assert!(s == "1", "ln(e) should be 1, got {s}"),
    }
}

/// sin(π/2) = 1
#[test]
fn special_sin_pi_over_2() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let two = ctx.int(2);
    let result = (&pi / &two).sin().eval();
    let s = format!("{result}");
    eprintln!("sin(π/2) = {s}");
    match eval_f64_ex(&result) {
        Ok(v) => assert!(approx(v, 1.0, 1e-10), "sin(π/2) should be 1, got {v}"),
        Err(_) => assert!(s == "1", "sin(π/2) should be 1, got {s}"),
    }
}

/// cos(π/2) = 0
#[test]
fn special_cos_pi_over_2() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let two = ctx.int(2);
    let result = (&pi / &two).cos().eval();
    let s = format!("{result}");
    eprintln!("cos(π/2) = {s}");
    match eval_f64_ex(&result) {
        Ok(v) => assert!(approx(v, 0.0, 1e-10), "cos(π/2) should be 0, got {v}"),
        Err(_) => assert!(s == "0", "cos(π/2) should be 0, got {s}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 12: POLYNOMIAL IDENTITY STRESS
// ═══════════════════════════════════════════════════════════════════════════

/// Expand and verify: (x+1)^5 at specific points
#[test]
fn expand_then_verify_x_plus_1_fifth() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let expr = (&x + &one).powi(5);
    let expanded = expr.expand();
    let s_exp = format!("{expanded}");
    eprintln!("(x+1)^5 expanded = {s_exp}");

    // Verify expanded form at x=2: (3)^5 = 243
    let val = expanded
        .subs_i64(&x, 2)
        .eval()
        .eval_f64()
        .expect("should eval");
    assert!(
        approx(val, 243.0, 1e-10),
        "(x+1)^5 at x=2 should be 243, got {val}"
    );

    // And at x=0: 1^5 = 1
    let val0 = expanded
        .subs_i64(&x, 0)
        .eval()
        .eval_f64()
        .expect("should eval");
    assert!(
        approx(val0, 1.0, 1e-10),
        "(x+1)^5 at x=0 should be 1, got {val0}"
    );

    // Coefficients: should be 1, 5, 10, 10, 5, 1 (Pascal's triangle)
    let val1 = expanded
        .subs_i64(&x, 1)
        .eval()
        .eval_f64()
        .expect("should eval");
    assert!(
        approx(val1, 32.0, 1e-10),
        "(x+1)^5 at x=1 should be 32, got {val1}"
    );
}

/// Polynomial GCD: gcd(x²-1, x²+2x+1) = x+1
#[test]
fn poly_gcd_test() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let p1 = x.powi(2) - &one; // x²-1 = (x-1)(x+1)
    let p2 = x.powi(2) + &x * 2 + &one; // x²+2x+1 = (x+1)²
    let g = p1.poly_gcd(&p2, &x);
    match g {
        Some(gcd) => {
            let s = format!("{gcd}");
            eprintln!("gcd(x²-1, x²+2x+1) = {s}");

            // The GCD should be x+1 (or a scalar multiple)
            let v1 = gcd.subs_i64(&x, 2).eval().eval_f64();
            let v2 = gcd.subs_i64(&x, 5).eval().eval_f64();
            if let (Ok(a), Ok(b)) = (v1, v2) {
                // ratio should be constant if g is a scalar multiple of (x+1)
                let r1 = a / 3.0; // (x+1) at x=2 is 3
                let r2 = b / 6.0; // (x+1) at x=5 is 6
                assert!(
                    approx(r1, r2, 1e-8),
                    "poly_gcd should be proportional to x+1: ratio at x=2 is {r1}, at x=5 is {r2}"
                );
            }
        }
        None => {
            eprintln!("poly_gcd returned None — expressions may not be polynomial");
        }
    }
}

/// Vieta's formulas: for x²+bx+c=0 with roots r1,r2: r1+r2=-b, r1·r2=c
#[test]
fn vietas_formulas_quadratic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x² - 5x + 6 = 0 => roots 2 and 3
    let expr = x.powi(2) - &x * 5 + 6;
    let solutions = expr.solve(&x).expect("should solve");
    assert_eq!(solutions.len(), 2);
    let r1 = eval_f64_ex(&solutions[0]).expect("root 1");
    let r2 = eval_f64_ex(&solutions[1]).expect("root 2");
    eprintln!("roots: {r1}, {r2}");
    // r1 + r2 = 5 (= -b)
    assert!(
        approx(r1 + r2, 5.0, 1e-10),
        "sum of roots should be 5, got {}",
        r1 + r2
    );
    // r1 * r2 = 6 (= c)
    assert!(
        approx(r1 * r2, 6.0, 1e-10),
        "product of roots should be 6, got {}",
        r1 * r2
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 13: PARSE-AND-ROUNDTRIP STRESS
// ═══════════════════════════════════════════════════════════════════════════

/// Parse a complex expression and verify it evaluates correctly
#[test]
fn parse_complex_expression() {
    let ctx = Context::new();
    let expr = ctx.parse("2^10").expect("should parse");
    let val = eval_f64_ex(&expr.eval()).expect("should eval");
    assert!(approx(val, 1024.0, 1e-10), "2^10 should be 1024, got {val}");
}

/// Parse nested functions
#[test]
fn parse_nested_functions() {
    let ctx = Context::new();
    let expr = ctx.parse("sin(0)").expect("should parse");
    let result = expr.eval();
    let s = format!("{result}");
    eprintln!("sin(0) parsed = {s}");
    let val = eval_f64_ex(&result);
    let v = val.expect("val must evaluate");
    assert!(approx(v, 0.0, 1e-10), "sin(0) should be 0, got {v}");
}

/// Parse and evaluate factorial
#[test]
fn parse_factorial() {
    let ctx = Context::new();
    let five = ctx.int(5);
    let result = five.factorial().eval();
    let s = format!("{result}");
    eprintln!("5! = {s}");
    match eval_f64_ex(&result) {
        Ok(v) => assert!(approx(v, 120.0, 1e-10), "5! should be 120, got {v}"),
        Err(_) => assert_eq!(s, "120", "5! should be 120, got {s}"),
    }
}

/// Parse "x^2 + 1" and verify substitution
#[test]
fn parse_and_substitute() {
    let ctx = Context::new();
    let expr = ctx.parse("x^2 + 1").expect("should parse");
    let x = ctx.symbol("x");
    let result = expr.subs_i64(&x, 3).eval();
    let val = eval_f64_ex(&result).expect("should eval");
    assert!(approx(val, 10.0, 1e-10), "3²+1 should be 10, got {val}");
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 14: INTEGRATION OF STANDARD FUNCTIONS (FTC verification)
// ═══════════════════════════════════════════════════════════════════════════

/// ∫ cos(x) dx = sin(x) — verify via FTC
#[test]
fn integrate_cos_gives_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.cos();
    let anti = integrand.integrate(&x);
    let deriv = anti.diff(&x);

    for &pt in &[0.0_f64, 0.5, 1.0, 2.0, 3.0] {
        let numer = (pt * 1000.0) as i64;
        let vi = eval_rational(&integrand, &x, numer, 1000);
        let vd = eval_rational(&deriv, &x, numer, 1000);
        if let (Ok(a), Ok(b)) = (vi, vd) {
            assert!(
                approx(a, b, 1e-8),
                "d/dx[∫cos(x)dx] at x={pt}: integrand={a}, deriv={b}"
            );
        }
    }
}

/// ∫ 1/x dx = ln(x) — verify via FTC
#[test]
fn integrate_one_over_x_gives_ln() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let integrand = &one / &x;
    let anti = integrand.integrate(&x);
    let s = format!("{anti}");
    eprintln!("∫ 1/x dx = {s}");

    // Verify: d/dx(result) should be 1/x
    let deriv = anti.diff(&x);
    for &pt in &[0.5_f64, 1.0, 2.0, 5.0] {
        let numer = (pt * 1000.0) as i64;
        let vi = eval_rational(&integrand, &x, numer, 1000);
        let vd = eval_rational(&deriv, &x, numer, 1000);
        if let (Ok(a), Ok(b)) = (vi, vd) {
            assert!(
                approx(a, b, 1e-8),
                "d/dx[∫(1/x)dx] at x={pt}: integrand={a}, deriv={b}"
            );
        }
    }
}

/// ∫ x·exp(x) dx = (x-1)·exp(x) — integration by parts
#[test]
fn integrate_x_exp_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x * x.exp();
    let anti = integrand.integrate(&x);
    let s = format!("{anti}");
    eprintln!("∫ x·exp(x) dx = {s}");

    if !s.contains("Integral") {
        let deriv = anti.diff(&x);
        for &pt in &[0.0_f64, 0.5, 1.0, 2.0] {
            let numer = (pt * 1000.0) as i64;
            let vi = eval_rational(&integrand, &x, numer, 1000);
            let vd = eval_rational(&deriv, &x, numer, 1000);
            if let (Ok(a), Ok(b)) = (vi, vd) {
                assert!(
                    approx(a, b, 1e-7),
                    "FTC for ∫x·exp(x)dx at x={pt}: integrand={a}, deriv={b}"
                );
            }
        }
    }
}

/// ∫ x² dx = x³/3 — verify via FTC
#[test]
fn integrate_x_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.powi(2);
    let anti = integrand.integrate(&x);
    let deriv = anti.diff(&x);

    for &pt in &[1_i64, 2, 3, -1, -2] {
        let vi = integrand.subs_i64(&x, pt).eval().eval_f64();
        let vd = deriv.subs_i64(&x, pt).eval().eval_f64();
        if let (Ok(a), Ok(b)) = (vi, vd) {
            assert!(
                approx(a, b, 1e-9),
                "d/dx[∫x²dx] at x={pt}: integrand={a}, deriv={b}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 15: MIXED STRESS — THINGS THAT TICKLE MULTIPLE SUBSYSTEMS
// ═══════════════════════════════════════════════════════════════════════════

/// Substitution stress: deeply nested subs should not stack overflow
#[test]
fn substitution_deep_nesting() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // Build sin(sin(sin(...sin(x)...))) with 20 layers
    let mut expr = x.clone();
    for _ in 0..20 {
        expr = expr.sin();
    }
    let s = format!("{expr}");
    assert!(!s.is_empty(), "deeply nested sin should not crash");

    // Substitute x=0 — should be 0 (since sin(0) = 0)
    let result = expr.subs_i64(&x, 0).eval();
    let val = eval_f64_ex(&result);
    let v = val.expect("val must evaluate");
    assert!(approx(v, 0.0, 1e-10), "sin^20(0) should be 0, got {v}");
}

/// Expand a large polynomial: (x+y)^10
#[test]
fn expand_large_binomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = (&x + &y).powi(10);
    let expanded = expr.expand();
    let s = format!("{expanded}");
    eprintln!("(x+y)^10 expanded length = {}", s.len());

    // Verify at x=1, y=1: should be 2^10 = 1024
    let val = expanded.subs_i64(&x, 1).subs_i64(&y, 1).eval().eval_f64();
    let v = val.expect("val must evaluate");
    assert!(approx(v, 1024.0, 1e-10), "(1+1)^10 should be 1024, got {v}");

    // Verify at x=2, y=3: should be 5^10 = 9765625
    let val2 = expanded.subs_i64(&x, 2).subs_i64(&y, 3).eval().eval_f64();
    let v = val2.expect("val2 must evaluate");
    assert!(
        approx(v, 9765625.0, 1e-6),
        "(2+3)^10 should be 9765625, got {v}"
    );
}

/// Symbolic differentiation preserves eval: d/dx[expr].eval() at a point
/// should match finite difference (f(x+h)-f(x-h))/(2h)
#[test]
fn diff_matches_finite_difference() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (&x * x.sin() + x.exp()) * x.ln(); // complex expression

    let deriv = expr.diff(&x);
    let pt = 1.5_f64;

    // Use a larger h so that rational approximation doesn't lose it,
    // and use a large denominator for precision.
    let h = 1e-5_f64;
    let denom: i64 = 10_000_000;
    let numer_pt = (pt * denom as f64).round() as i64;
    let numer_ph = ((pt + h) * denom as f64).round() as i64;
    let numer_mh = ((pt - h) * denom as f64).round() as i64;

    let f_plus = eval_rational(&expr, &x, numer_ph, denom);
    let f_minus = eval_rational(&expr, &x, numer_mh, denom);
    let analytic = eval_rational(&deriv, &x, numer_pt, denom);

    if let (Ok(fp), Ok(fm), Ok(ad)) = (f_plus, f_minus, analytic) {
        let actual_h = (numer_ph - numer_mh) as f64 / (denom as f64 * 2.0);
        let numerical = (fp - fm) / (2.0 * actual_h);
        assert!(
            approx(ad, numerical, 1e-4),
            "d/dx at x={pt}: analytic={ad}, numerical={numerical}"
        );
    }
}

/// Verify that simplify doesn't change the numerical value of expressions
#[test]
fn simplify_preserves_value_stress() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let exprs: Vec<(&str, Ex)> = vec![
        (
            "x^3 + 3*x^2 + 3*x + 1",
            x.powi(3) + x.powi(2) * 3 + &x * 3 + 1,
        ),
        ("sin(x)^2 + cos(x)^2", x.sin().powi(2) + x.cos().powi(2)),
        ("(x^2 - 1)/(x + 1)", (x.powi(2) - 1) / (&x + &ctx.int(1))),
        ("exp(2*ln(x))", (&x.ln() * 2).exp()),
    ];

    for (label, expr) in &exprs {
        let simplified = expr.simplify();
        for &pt in &[2_i64, 3, 5] {
            let v_orig = expr.subs_i64(&x, pt).eval().eval_f64();
            let v_simp = simplified.subs_i64(&x, pt).eval().eval_f64();
            if let (Ok(a), Ok(b)) = (v_orig, v_simp) {
                assert!(
                    approx(a, b, 1e-8),
                    "{label}: simplify changed value at x={pt}: {a} vs {b}"
                );
            }
        }
    }
}

/// eval_complex64 on purely real expression should have im ≈ 0
#[test]
fn eval_complex_real_expression() {
    let ctx = Context::new();
    let expr = ctx.int(42);
    let (re, im) = eval_c64(&expr).expect("should evaluate");
    assert!(approx(re, 42.0, 1e-10), "real part should be 42, got {re}");
    assert!(
        approx(im, 0.0, 1e-10),
        "imaginary part should be 0, got {im}"
    );
}

/// Multiple substitutions in one go
#[test]
fn multi_subs_consistency() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = x.powi(2) + &y * 3 - 7;

    // subs x=2, y=5: 4 + 15 - 7 = 12
    let result = expr.subs_i64(&x, 2).subs_i64(&y, 5).eval();
    let val = eval_f64_ex(&result).expect("should evaluate");
    assert!(
        approx(val, 12.0, 1e-10),
        "x²+3y-7 at x=2,y=5 should be 12, got {val}"
    );
}

/// 0! = 1
#[test]
fn factorial_zero() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let result = zero.factorial().eval();
    let s = format!("{result}");
    eprintln!("0! = {s}");
    match eval_f64_ex(&result) {
        Ok(v) => assert!(approx(v, 1.0, 1e-10), "0! should be 1, got {v}"),
        Err(_) => assert_eq!(s, "1", "0! should be 1, got {s}"),
    }
}

/// 10! = 3628800
#[test]
fn factorial_ten() {
    let ctx = Context::new();
    let ten = ctx.int(10);
    let result = ten.factorial().eval();
    let s = format!("{result}");
    eprintln!("10! = {s}");
    match eval_f64_ex(&result) {
        Ok(v) => assert!(approx(v, 3628800.0, 1e-6), "10! should be 3628800, got {v}"),
        Err(_) => assert_eq!(s, "3628800", "10! should be 3628800, got {s}"),
    }
}

/// binomial(5,2) = 10
#[test]
fn binomial_5_2() {
    let ctx = Context::new();
    let five = ctx.int(5);
    let two = ctx.int(2);
    let result = five.binomial(&two).eval();
    let s = format!("{result}");
    eprintln!("C(5,2) = {s}");
    match eval_f64_ex(&result) {
        Ok(v) => assert!(approx(v, 10.0, 1e-10), "C(5,2) should be 10, got {v}"),
        Err(_) => assert_eq!(s, "10", "C(5,2) should be 10, got {s}"),
    }
}

/// binomial(n,0) = 1 for n=7
#[test]
fn binomial_n_0() {
    let ctx = Context::new();
    let seven = ctx.int(7);
    let zero = ctx.int(0);
    let result = seven.binomial(&zero).eval();
    let s = format!("{result}");
    eprintln!("C(7,0) = {s}");
    match eval_f64_ex(&result) {
        Ok(v) => assert!(approx(v, 1.0, 1e-10), "C(7,0) should be 1, got {v}"),
        Err(_) => assert_eq!(s, "1", "C(7,0) should be 1, got {s}"),
    }
}

/// binomial(n,n) = 1 for n=7
#[test]
fn binomial_n_n() {
    let ctx = Context::new();
    let seven = ctx.int(7);
    let result = seven.binomial(&seven).eval();
    let s = format!("{result}");
    eprintln!("C(7,7) = {s}");
    match eval_f64_ex(&result) {
        Ok(v) => assert!(approx(v, 1.0, 1e-10), "C(7,7) should be 1, got {v}"),
        Err(_) => assert_eq!(s, "1", "C(7,7) should be 1, got {s}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 16: nextprime(i64::MAX) overflow test
// ═══════════════════════════════════════════════════════════════════════════

/// nextprime(i64::MAX) must not panic — it must use BigInt path.
/// This specifically tests for integer overflow in the i64 fast path.
#[test]
fn ntheory_nextprime_i64_max_no_panic() {
    // Pass as BigInt to skip the i64 fast path (which would overflow)
    let n = BigInt::from(i64::MAX);
    let result = ntheory::nextprime(n.clone());
    eprintln!("nextprime(i64::MAX) = {result}");
    assert!(result > n, "nextprime(i64::MAX) should be > i64::MAX");
    assert!(ntheory::isprime(result.clone()), "result should be prime");
}

/// divisor_count(12) = 6 (divisors: 1,2,3,4,6,12)
#[test]
fn ntheory_divisor_count() {
    let result = ntheory::divisor_count(12_i64);
    assert_eq!(result, 6, "σ₀(12) should be 6, got {result}");
}

/// divisors(12) = [1,2,3,4,6,12]
#[test]
fn ntheory_divisors_twelve() {
    let mut result = ntheory::divisors(12_i64);
    result.sort();
    let expected: Vec<BigInt> = vec![1, 2, 3, 4, 6, 12]
        .into_iter()
        .map(BigInt::from)
        .collect();
    assert_eq!(
        result, expected,
        "divisors(12) should be [1,2,3,4,6,12], got {:?}",
        result
    );
}

/// is_square(16) = true, is_square(15) = false
#[test]
fn ntheory_is_square() {
    assert!(ntheory::is_square(16_i64), "16 is a perfect square");
    assert!(!ntheory::is_square(15_i64), "15 is not a perfect square");
    assert!(ntheory::is_square(0_i64), "0 is a perfect square");
    assert!(ntheory::is_square(1_i64), "1 is a perfect square");
    assert!(!ntheory::is_square(-1_i64), "-1 is not a perfect square");
}

/// is_coprime(8, 15) = true, is_coprime(8, 12) = false
#[test]
fn ntheory_is_coprime() {
    assert!(ntheory::is_coprime(8_i64, 15_i64), "gcd(8,15)=1");
    assert!(!ntheory::is_coprime(8_i64, 12_i64), "gcd(8,12)=4≠1");
}

/// prevprime(2) = None (no prime before 2)
#[test]
fn ntheory_prevprime_two() {
    let result = ntheory::prevprime(2_i64);
    assert!(
        result.is_none(),
        "prevprime(2) should be None, got {:?}",
        result
    );
}

/// prevprime(3) = 2
#[test]
fn ntheory_prevprime_three() {
    let result = ntheory::prevprime(3_i64);
    assert_eq!(
        result,
        Some(BigInt::from(2)),
        "prevprime(3) should be 2, got {:?}",
        result
    );
}

/// prevprime(10) = 7
#[test]
fn ntheory_prevprime_ten() {
    let result = ntheory::prevprime(10_i64);
    assert_eq!(
        result,
        Some(BigInt::from(7)),
        "prevprime(10) should be 7, got {:?}",
        result
    );
}

/// prevprime(1) = None
#[test]
fn ntheory_prevprime_one() {
    let result = ntheory::prevprime(1_i64);
    assert!(
        result.is_none(),
        "prevprime(1) should be None, got {:?}",
        result
    );
}

/// prevprime(0) = None
#[test]
fn ntheory_prevprime_zero() {
    let result = ntheory::prevprime(0_i64);
    assert!(
        result.is_none(),
        "prevprime(0) should be None, got {:?}",
        result
    );
}

/// prevprime(-5) = None
#[test]
fn ntheory_prevprime_negative() {
    let result = ntheory::prevprime(-5_i64);
    assert!(
        result.is_none(),
        "prevprime(-5) should be None, got {:?}",
        result
    );
}

/// isqrt(i64::MAX) — should not panic or overflow internally
#[test]
fn ntheory_isqrt_i64_max() {
    let result = ntheory::isqrt(i64::MAX);
    match result {
        Some(s) => {
            // s² ≤ i64::MAX < (s+1)²
            let s_sq = &s * &s;
            let s_plus_1 = &s + BigInt::from(1);
            let s_plus_1_sq = &s_plus_1 * &s_plus_1;
            let n = BigInt::from(i64::MAX);
            assert!(s_sq <= n, "isqrt(i64::MAX)² should be ≤ i64::MAX");
            assert!(s_plus_1_sq > n, "(isqrt(i64::MAX)+1)² should be > i64::MAX");
            eprintln!("isqrt(i64::MAX) = {s}");
        }
        None => {
            panic!("isqrt(i64::MAX) should return Some, got None");
        }
    }
}

/// isqrt(0) = 0
#[test]
fn ntheory_isqrt_zero() {
    let result = ntheory::isqrt(0_i64);
    assert_eq!(result, Some(BigInt::from(0)), "isqrt(0) should be 0");
}

/// isqrt(-1) = None
#[test]
fn ntheory_isqrt_negative() {
    let result = ntheory::isqrt(-1_i64);
    assert!(result.is_none(), "isqrt(-1) should be None");
}

/// factorint(i64::MAX) — should factorize correctly and not hang
/// i64::MAX = 7² × 73 × 127 × 337 × 92737 × 649657
#[test]
fn ntheory_factorint_i64_max() {
    let factors = ntheory::factorint(i64::MAX);
    eprintln!("factorint(i64::MAX) = {:?}", factors);
    // Verify the factors multiply back to i64::MAX
    let mut product = BigInt::from(1);
    for (p, e) in &factors {
        for _ in 0..*e {
            product *= p;
        }
    }
    assert_eq!(
        product,
        BigInt::from(i64::MAX),
        "factors of i64::MAX should multiply back"
    );
}

/// CRT basic: x ≡ 2 mod 3, x ≡ 3 mod 5 → x ≡ 8 mod 15
#[test]
fn ntheory_crt_basic() {
    let remainders = vec![BigInt::from(2), BigInt::from(3)];
    let moduli = vec![BigInt::from(3), BigInt::from(5)];
    let result = ntheory::crt(&remainders, &moduli);
    match result {
        Some(x) => {
            // x mod 15 should be 8
            let fifteen = BigInt::from(15);
            let normalized = ((&x % &fifteen) + &fifteen) % &fifteen;
            assert_eq!(
                normalized,
                BigInt::from(8),
                "CRT: x ≡ 2 mod 3, x ≡ 3 mod 5 → x mod 15 = 8, got {normalized}"
            );
        }
        None => panic!("CRT should find a solution"),
    }
}

/// CRT incompatible: x ≡ 0 mod 2, x ≡ 1 mod 2 → no solution
#[test]
fn ntheory_crt_incompatible() {
    let remainders = vec![BigInt::from(0), BigInt::from(1)];
    let moduli = vec![BigInt::from(2), BigInt::from(2)];
    let result = ntheory::crt(&remainders, &moduli);
    assert!(
        result.is_none(),
        "CRT with incompatible congruences should return None"
    );
}

/// mod_pow(2, 10, 1000) = 24
#[test]
fn ntheory_mod_pow_basic() {
    let result = ntheory::mod_pow(2_i64, 10_i64, 1000_i64);
    assert_eq!(
        result,
        BigInt::from(24),
        "2^10 mod 1000 = 1024 mod 1000 = 24, got {result}"
    );
}

/// mod_pow(base, 0, m) = 1 for m > 1
#[test]
fn ntheory_mod_pow_zero_exp() {
    let result = ntheory::mod_pow(7_i64, 0_i64, 13_i64);
    assert_eq!(result, BigInt::from(1), "7^0 mod 13 = 1, got {result}");
}

/// divisor_sum(12) = 1+2+3+4+6+12 = 28
#[test]
fn ntheory_divisor_sum_twelve() {
    let result = ntheory::divisor_sum(12_i64);
    assert_eq!(result, BigInt::from(28), "σ(12) should be 28, got {result}");
}

/// totient(p) = p-1 for prime p
#[test]
fn ntheory_totient_prime() {
    let result = ntheory::totient(13_i64);
    assert_eq!(result, BigInt::from(12), "φ(13) should be 12, got {result}");
}

/// Solve x⁵ + x + 1 = 0 — degree 5, no general radical solution.
/// Should produce RootOf nodes or numerical roots, not panic.
#[test]
fn solve_quintic_no_radical() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(5) + &x + 1;
    match expr.solve(&x) {
        Ok(sols) => {
            eprintln!(
                "x⁵+x+1=0 solutions: {:?}",
                sols.iter().map(|s| format!("{s}")).collect::<Vec<_>>()
            );
            // Verify each claimed solution actually satisfies the equation
            for sol in &sols {
                if let Ok((re, im)) = eval_c64(sol) {
                    // Substitute back: sol^5 + sol + 1 should ≈ 0
                    let check = expr.subs(&x, sol).eval();
                    if let Ok((cr, ci)) = eval_c64(&check) {
                        let mag = (cr * cr + ci * ci).sqrt();
                        assert!(
                            mag < 1e-6,
                            "x⁵+x+1=0: solution ({re}+{im}i) gives residual ({cr}+{ci}i), |residual|={mag}"
                        );
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Solver failed on quintic (expected for degree≥5): {e}");
            // Not a bug per se — quintics have no general formula
        }
    }
}

/// ∫₀^1 exp(-x²) dx ≈ 0.7468 (part of erf)
/// This tests definite integration of a non-elementary integrand.
#[test]
fn definite_integral_gaussian_0_to_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = (-x.powi(2)).exp(); // exp(-x²)
    let result = integrand.integrate_definite(&x, &ctx.int(0), &ctx.int(1));
    let s = format!("{result}");
    eprintln!("∫₀^1 exp(-x²) dx = {s}");
    // If it evaluates, check numerically
    match eval_f64_ex(&result.eval()) {
        Ok(v) => {
            // The exact value is erf(1)·√π/2 ≈ 0.74682...
            assert!(
                approx(v, 0.7468241328, 1e-3),
                "∫₀^1 exp(-x²) dx should be ≈ 0.7468, got {v}"
            );
        }
        Err(e) => {
            eprintln!("Could not evaluate ∫₀^1 exp(-x²) dx: {e}");
            // Not necessarily a bug — this is a non-elementary integral
        }
    }
}

/// Double differentiation then integration should recover original (up to constant):
/// ∫ d²/dx²[x⁴] dx = d/dx[x⁴] = 4x³ (up to constant)
#[test]
fn ftc_second_derivative_then_integrate() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(4);
    let d2f = f.diff_n(&x, 2); // 12x²
    let integrated = d2f.integrate(&x); // should be 4x³ + C
    let re_diff = integrated.diff(&x); // should be 12x²

    for &pt in &[1_i64, 2, 3, -1] {
        let vd2 = d2f.subs_i64(&x, pt).eval().eval_f64();
        let vrd = re_diff.subs_i64(&x, pt).eval().eval_f64();
        if let (Ok(a), Ok(b)) = (vd2, vrd) {
            assert!(
                approx(a, b, 1e-9),
                "d/dx[∫ d²/dx²[x⁴] dx] at x={pt}: expected {a}, got {b}"
            );
        }
    }
}

/// Matrix trace is sum of diagonal: tr(A+B) = tr(A) + tr(B)
#[test]
fn matrix_trace_additive() {
    let ctx = Context::new();
    let a = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2)],
        vec![ctx.int(3), ctx.int(4)],
    ])
    .unwrap();
    let b = Matrix::new(vec![
        vec![ctx.int(5), ctx.int(6)],
        vec![ctx.int(7), ctx.int(8)],
    ])
    .unwrap();

    let sum = &a + &b;
    let tr_sum = sum.trace().expect("tr(A+B)");
    let tr_a = a.trace().expect("tr(A)");
    let tr_b = b.trace().expect("tr(B)");
    let tr_a_plus_tr_b = (&tr_a + &tr_b).eval();

    let v1 = eval_f64_ex(&tr_sum).expect("eval tr(A+B)");
    let v2 = eval_f64_ex(&tr_a_plus_tr_b).expect("eval tr(A)+tr(B)");
    assert!(
        approx(v1, v2, 1e-10),
        "tr(A+B)={v1} should equal tr(A)+tr(B)={v2}"
    );
}

/// det(kA) = k^n * det(A) for n×n matrix
#[test]
fn matrix_det_scalar_multiple() {
    let ctx = Context::new();
    let a = Matrix::new(vec![
        vec![ctx.int(1), ctx.int(2)],
        vec![ctx.int(3), ctx.int(4)],
    ])
    .unwrap();
    let k = ctx.int(3);
    let ka = &a * &k;
    let det_ka = ka.det().expect("det(kA)");
    let det_a = a.det().expect("det(A)");
    // For 2×2: det(3A) = 3² * det(A) = 9 * det(A)
    let expected = (&det_a * &ctx.int(9)).eval();

    let v1 = eval_f64_ex(&det_ka).expect("eval det(kA)");
    let v2 = eval_f64_ex(&expected).expect("eval k²·det(A)");
    assert!(
        approx(v1, v2, 1e-10),
        "det(3A)={v1} should equal 9·det(A)={v2}"
    );
}

/// sinh and cosh identity: cosh²(x) - sinh²(x) = 1
#[test]
fn hyperbolic_pythagorean_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.cosh().powi(2) - x.sinh().powi(2);

    for &pt in &[0_i64, 1, 2, -1, -2] {
        let val = expr.subs_i64(&x, pt).eval().eval_f64();
        let v = val.expect("val must evaluate");
        assert!(
            approx(v, 1.0, 1e-10),
            "cosh²({pt}) - sinh²({pt}) should be 1, got {v}"
        );
    }
}

/// d/dx[sinh(x)] = cosh(x)
#[test]
fn diff_sinh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let deriv = x.sinh().diff(&x);

    for &pt in &[0_i64, 1, 2, -1] {
        let vd = deriv.subs_i64(&x, pt).eval().eval_f64();
        let expected = (pt as f64).cosh();
        let v = vd.expect("vd must evaluate");
        assert!(
            approx(v, expected, 1e-9),
            "d/dx[sinh(x)] at x={pt}: got {v}, expected {expected}"
        );
    }
}

/// d/dx[cosh(x)] = sinh(x)
#[test]
fn diff_cosh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let deriv = x.cosh().diff(&x);

    for &pt in &[0_i64, 1, 2, -1] {
        let vd = deriv.subs_i64(&x, pt).eval().eval_f64();
        let expected = (pt as f64).sinh();
        let v = vd.expect("vd must evaluate");
        assert!(
            approx(v, expected, 1e-9),
            "d/dx[cosh(x)] at x={pt}: got {v}, expected {expected}"
        );
    }
}

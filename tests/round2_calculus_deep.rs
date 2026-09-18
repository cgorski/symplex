//! Round 2 deep calculus tests for symplex.
//!
//! Targets areas that weren't thoroughly tested in round 1:
//! - Harder indefinite integrals (tan, sec, IBP, completing the square)
//! - Definite integrals with known values
//! - ODE solving and verification
//! - Series expansion edge cases (geometric, binomial, non-zero point)
//! - Laplace / inverse Laplace roundtrips
//! - Z-transform / inverse Z-transform roundtrips
//! - Fourier transform pairs
//! - Convergence tests for known series

mod common;

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helper: FTC numerical verification (integrate then differentiate)
// ═══════════════════════════════════════════════════════════════════════════

/// Numerically verify the Fundamental Theorem of Calculus:
///   d/dx(∫ f dx) ≈ f   at several test points.
///
/// Returns `(antiderivative_display, is_unevaluated)`.
fn ftc_check(integrand: &Ex, var: &Ex, points: &[f64], tol: f64, label: &str) -> (String, bool) {
    let antideriv = integrand.integrate(var);
    let s = format!("{antideriv}");
    let is_uneval = s.contains("Integral") || antideriv.has_unevaluated();

    if is_uneval {
        return (s, true);
    }

    let deriv = antideriv.diff(var);
    let ctx = integrand.context();
    let mut checked = 0usize;

    for &pt_f in points {
        let numer = (pt_f * 10000.0).round() as i64;
        let pt = ctx.rational(numer, 10000);

        let orig_val = integrand.subs(var, &pt).eval().eval_f64();
        let deriv_val = deriv.subs(var, &pt).eval().eval_f64();

        if let (Ok(o), Ok(d)) = (orig_val, deriv_val)
            && o.is_finite()
            && d.is_finite()
        {
            checked += 1;
            let scale = o.abs().max(d.abs()).max(1.0);
            assert!(
                (o - d).abs() < tol * scale,
                "FTC FAILED for {label} at {var}={pt_f}: \
                     integrand={o}, d/dx(antideriv)={d}, diff={}, \
                     antideriv='{antideriv}', deriv='{deriv}'",
                (o - d).abs(),
            );
        }
    }

    if checked == 0 {
        eprintln!("WARNING: FTC {label}: no evaluation points succeeded");
    }

    (s, false)
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 1: HARDER INDEFINITE INTEGRALS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_tan_x() {
    // ∫ tan(x) dx = -ln(cos(x)) + C   (or ln|sec(x)| + C)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // tan(x) = sin(x)/cos(x)
    let integrand = &x.sin() / &x.cos();
    let (s, is_uneval) = ftc_check(&integrand, &x, &[0.3, 0.7, 1.0], 1e-6, "∫tan(x)dx");

    if is_uneval {
        panic!(
            "BUG: ∫tan(x)dx returned unevaluated form: {s}\n\
             Expected: -ln(cos(x)) or equivalent"
        );
    }
}

#[test]
fn integrate_sec_x() {
    // ∫ sec(x) dx = ln|sec(x) + tan(x)| + C
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sec(x) = 1/cos(x)
    let integrand = &ctx.int(1) / &x.cos();
    let (s, is_uneval) = ftc_check(&integrand, &x, &[0.2, 0.5, 0.8], 1e-6, "∫sec(x)dx");

    if is_uneval {
        panic!(
            "BUG: ∫sec(x)dx returned unevaluated form: {s}\n\
             Expected: ln|sec(x)+tan(x)| or equivalent"
        );
    }
}

#[test]
fn integrate_x_squared_sin_x() {
    // ∫ x²·sin(x) dx = -x²·cos(x) + 2x·sin(x) + 2·cos(x) + C
    // Requires integration by parts twice.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x.powi(2) * &x.sin();
    let (s, is_uneval) = ftc_check(&integrand, &x, &[0.5, 1.0, 1.5, 2.0], 1e-6, "∫x²sin(x)dx");

    if is_uneval {
        panic!(
            "BUG: ∫x²·sin(x)dx returned unevaluated form: {s}\n\
             Expected: -x²·cos(x) + 2x·sin(x) + 2·cos(x) or equivalent"
        );
    }
}

#[test]
fn integrate_x_cos_x() {
    // ∫ x·cos(x) dx = x·sin(x) + cos(x) + C
    // Requires integration by parts once.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x * &x.cos();
    let (s, is_uneval) = ftc_check(&integrand, &x, &[0.5, 1.0, 1.5, 2.0], 1e-6, "∫x·cos(x)dx");

    if is_uneval {
        panic!(
            "BUG: ∫x·cos(x)dx returned unevaluated form: {s}\n\
             Expected: x·sin(x) + cos(x) or equivalent"
        );
    }
}

#[test]
fn integrate_x_exp_x() {
    // ∫ x·exp(x) dx = (x-1)·exp(x) + C
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x * &x.exp();
    let (s, is_uneval) = ftc_check(&integrand, &x, &[0.3, 0.7, 1.0, 1.5], 1e-6, "∫x·exp(x)dx");

    if is_uneval {
        panic!(
            "BUG: ∫x·exp(x)dx returned unevaluated form: {s}\n\
             Expected: (x-1)·exp(x) or equivalent"
        );
    }
}

#[test]
fn integrate_x_squared_exp_x() {
    // ∫ x²·exp(x) dx = (x²-2x+2)·exp(x) + C
    // Requires IBP twice.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x.powi(2) * &x.exp();
    let (s, is_uneval) = ftc_check(&integrand, &x, &[0.3, 0.7, 1.0, 1.5], 1e-6, "∫x²·exp(x)dx");

    if is_uneval {
        panic!(
            "BUG: ∫x²·exp(x)dx returned unevaluated form: {s}\n\
             Expected: (x²-2x+2)·exp(x) or equivalent"
        );
    }
}

#[test]
fn integrate_one_over_one_plus_x_squared() {
    // ∫ 1/(1+x²) dx = arctan(x) + C
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &ctx.int(1) / &(&x.powi(2) + &ctx.int(1));
    let (s, is_uneval) = ftc_check(&integrand, &x, &[0.0, 0.5, 1.0, 2.0], 1e-6, "∫1/(1+x²)dx");

    if is_uneval {
        panic!(
            "BUG: ∫1/(1+x²)dx returned unevaluated form: {s}\n\
             Expected: arctan(x) or equivalent"
        );
    }
}

#[test]
fn integrate_completing_the_square() {
    // ∫ 1/(x²+2x+2) dx = arctan(x+1) + C
    // x²+2x+2 = (x+1)² + 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let denom = &(&x.powi(2) + &(&x * 2)) + &ctx.int(2);
    let integrand = &ctx.int(1) / &denom;
    let (s, is_uneval) = ftc_check(
        &integrand,
        &x,
        &[0.0, 0.5, 1.0, 2.0],
        1e-6,
        "∫1/(x²+2x+2)dx",
    );

    if is_uneval {
        panic!(
            "BUG: ∫1/(x²+2x+2)dx returned unevaluated form: {s}\n\
             Expected: arctan(x+1) or equivalent"
        );
    }
}

#[test]
fn integrate_sin_squared_x() {
    // ∫ sin²(x) dx = x/2 - sin(2x)/4 + C
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.sin().powi(2);
    let (s, is_uneval) = ftc_check(&integrand, &x, &[0.3, 0.7, 1.0, 1.5], 1e-6, "∫sin²(x)dx");

    if is_uneval {
        panic!(
            "BUG: ∫sin²(x)dx returned unevaluated form: {s}\n\
             Expected: x/2 - sin(2x)/4 or equivalent"
        );
    }
}

#[test]
fn integrate_cos_squared_x() {
    // ∫ cos²(x) dx = x/2 + sin(2x)/4 + C
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.cos().powi(2);
    let (s, is_uneval) = ftc_check(&integrand, &x, &[0.3, 0.7, 1.0, 1.5], 1e-6, "∫cos²(x)dx");

    if is_uneval {
        panic!(
            "BUG: ∫cos²(x)dx returned unevaluated form: {s}\n\
             Expected: x/2 + sin(2x)/4 or equivalent"
        );
    }
}

#[test]
fn integrate_exp_neg_x_squared_no_elementary() {
    // ∫ exp(-x²) dx has no elementary closed form (it's erf).
    // The integrator should either return an unevaluated Integral
    // or an erf expression.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = (-&x.powi(2)).exp();
    let antideriv = integrand.integrate(&x);
    let s = format!("{antideriv}");
    // This integral has no elementary form, so if the integrator returns
    // something concrete it must involve erf or be validated via FTC.
    if !antideriv.has_unevaluated() && !s.contains("Integral") {
        // The integrator claims it found a closed form — verify via FTC
        let deriv = antideriv.diff(&x);
        let pt = ctx.rational(7, 10); // x = 0.7
        let orig_val = integrand.subs(&x, &pt).eval().eval_f64();
        let deriv_val = deriv.subs(&x, &pt).eval().eval_f64();
        if let (Ok(o), Ok(d)) = (orig_val, deriv_val) {
            let scale = o.abs().max(d.abs()).max(1.0);
            assert!(
                (o - d).abs() < 1e-5 * scale,
                "BUG: ∫exp(-x²)dx claims closed form '{s}' but FTC fails: \
                 integrand={o}, d/dx(result)={d}"
            );
        }
    }
    // If unevaluated, that's acceptable — this integral is non-elementary.
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 2: DEFINITE INTEGRALS WITH KNOWN VALUES
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn definite_sin_0_to_pi_equals_2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // ∫₀^π sin(x) dx = [-cos(x)]₀^π = -cos(π) + cos(0) = 1 + 1 = 2
    let result = x.sin().integrate_definite(&x, &ctx.int(0), &ctx.pi());
    let evaled = result.eval();
    let s = format!("{evaled}");
    assert_eq!(s, "2", "∫₀^π sin(x)dx should be exactly 2, got: {s}");
}

#[test]
fn definite_x_squared_0_to_1_equals_one_third() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // ∫₀¹ x² dx = 1/3
    let result = x.powi(2).integrate_definite(&x, &ctx.int(0), &ctx.int(1));
    let s = format!("{result}");
    assert_eq!(s, "1/3", "∫₀¹ x²dx should be 1/3, got: {s}");
}

#[test]
fn definite_x_cubed_0_to_1_equals_one_quarter() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // ∫₀¹ x³ dx = 1/4
    let result = x.powi(3).integrate_definite(&x, &ctx.int(0), &ctx.int(1));
    let s = format!("{result}");
    assert_eq!(s, "1/4", "∫₀¹ x³dx should be 1/4, got: {s}");
}

#[test]
fn definite_cos_0_to_pi_half_equals_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // ∫₀^{π/2} cos(x) dx = sin(π/2) - sin(0) = 1 - 0 = 1
    let pi_half = &ctx.pi() / &ctx.int(2);
    let result = x.cos().integrate_definite(&x, &ctx.int(0), &pi_half);
    let evaled = result.eval();
    let s = format!("{evaled}");
    assert_eq!(s, "1", "∫₀^{{π/2}} cos(x)dx should be 1, got: {s}");
}

#[test]
fn definite_exp_0_to_1_is_e_minus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // ∫₀¹ exp(x) dx = e - 1
    let result = x.exp().integrate_definite(&x, &ctx.int(0), &ctx.int(1));
    let evaled = result.eval();
    if let Ok(v) = evaled.eval_f64() {
        let expected = std::f64::consts::E - 1.0;
        assert!(
            (v - expected).abs() < 1e-8,
            "∫₀¹ exp(x)dx should be e-1 ≈ {expected}, got: {v}"
        );
    } else {
        let s = format!("{evaled}");
        assert!(
            s.contains("e") || s.contains("E") || s.contains("exp"),
            "∫₀¹ exp(x)dx should be e-1, got: {s}"
        );
    }
}

#[test]
fn definite_sin_0_to_2pi_equals_0() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // ∫₀^{2π} sin(x) dx = 0 (full period)
    let two_pi = &ctx.int(2) * &ctx.pi();
    let result = x.sin().integrate_definite(&x, &ctx.int(0), &two_pi);
    let evaled = result.eval();
    let s = format!("{evaled}");
    assert_eq!(s, "0", "∫₀^{{2π}} sin(x)dx should be 0, got: {s}");
}

#[test]
fn definite_integral_additivity() {
    // ∫₀² x dx = ∫₀¹ x dx + ∫₁² x dx = 2
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let full = x.integrate_definite(&x, &ctx.int(0), &ctx.int(2));
    let part1 = x.integrate_definite(&x, &ctx.int(0), &ctx.int(1));
    let part2 = x.integrate_definite(&x, &ctx.int(1), &ctx.int(2));

    let full_v = full.eval_f64().expect("full integral evals");
    let sum_v = {
        let p1 = part1.eval_f64().expect("part1 evals");
        let p2 = part2.eval_f64().expect("part2 evals");
        p1 + p2
    };

    assert!(
        (full_v - sum_v).abs() < 1e-10,
        "∫₀² x dx = {full_v}, ∫₀¹ + ∫₁² = {sum_v} — additivity failed"
    );
}

#[test]
fn definite_polynomial_high_degree() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // ∫₀¹ (x⁴ + x³ + x² + x + 1) dx = 1/5 + 1/4 + 1/3 + 1/2 + 1 = 137/60
    let poly = &(&(&(&x.powi(4) + &x.powi(3)) + &x.powi(2)) + &x) + &ctx.int(1);
    let result = poly.integrate_definite(&x, &ctx.int(0), &ctx.int(1));
    let v = result.eval_f64().expect("should evaluate to f64");
    let expected = 137.0 / 60.0;
    assert!(
        (v - expected).abs() < 1e-10,
        "∫₀¹ (x⁴+x³+x²+x+1)dx should be 137/60 ≈ {expected}, got: {v}"
    );
}

#[test]
fn definite_integral_reversed_bounds_negates() {
    // ∫_b^a f dx = -∫_a^b f dx
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let forward = x.powi(2).integrate_definite(&x, &ctx.int(0), &ctx.int(3));
    let reversed = x.powi(2).integrate_definite(&x, &ctx.int(3), &ctx.int(0));
    let fv = forward.eval_f64().expect("forward evals");
    let rv = reversed.eval_f64().expect("reversed evals");
    assert!(
        (fv + rv).abs() < 1e-10,
        "Reversing bounds should negate: forward={fv}, reversed={rv}, sum={}",
        fv + rv
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 3: ODE SOLVING CORRECTNESS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_y_prime_equals_y() {
    // y' - y = 0  →  solution: C·e^x
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let dy = y.formal_diff(&x);
    let ode = &dy - &y;

    let sol = ode.solve_ode(&y, &x);
    let s = format!("{sol}");

    assert!(
        !sol.has_unevaluated(),
        "BUG: y' - y = 0 should be solvable, got unevaluated: {s}"
    );
    assert!(
        s.contains("exp"),
        "Solution of y' = y should contain exp, got: {s}"
    );

    // Verify: y = exp(x) satisfies y' - y = 0
    let particular = x.exp();
    assert!(
        ode.check_ode_solution(&particular, &y, &x),
        "y = exp(x) should satisfy y' - y = 0"
    );
}

#[test]
fn ode_y_prime_plus_y_equals_0() {
    // y' + y = 0  →  solution: C·e^(-x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let dy = y.formal_diff(&x);
    let ode = &dy + &y;

    let sol = ode.solve_ode(&y, &x);
    let s = format!("{sol}");

    assert!(
        !sol.has_unevaluated(),
        "BUG: y' + y = 0 should be solvable, got unevaluated: {s}"
    );
    assert!(
        s.contains("exp"),
        "Solution of y' + y = 0 should contain exp, got: {s}"
    );

    // Verify: y = exp(-x) should satisfy y' + y = 0
    let particular = (-&x).exp();
    assert!(
        ode.check_ode_solution(&particular, &y, &x),
        "y = exp(-x) should satisfy y' + y = 0"
    );
}

#[test]
fn ode_y_prime_plus_y_wrong_solution_rejected() {
    // y' + y = 0: y = exp(x) should NOT satisfy this
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let dy = y.formal_diff(&x);
    let ode = &dy + &y;

    let wrong = x.exp(); // exp(x) does NOT satisfy y' + y = 0
    assert!(
        !ode.check_ode_solution(&wrong, &y, &x),
        "BUG: y = exp(x) should NOT satisfy y' + y = 0 but check_ode_solution says it does"
    );
}

#[test]
fn ode_second_order_y_pp_plus_y_eq_0() {
    // y'' + y = 0  →  C₁·cos(x) + C₂·sin(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &d2y + &y;

    let sol = ode.solve_ode(&y, &x);
    let s = format!("{sol}");

    assert!(
        !sol.has_unevaluated(),
        "BUG: y'' + y = 0 should be solvable, got unevaluated: {s}"
    );
    assert!(
        s.contains("sin") && s.contains("cos"),
        "Solution of y'' + y = 0 should have sin and cos, got: {s}"
    );

    // Verify particular solutions
    assert!(
        ode.check_ode_solution(&x.cos(), &y, &x),
        "y = cos(x) should satisfy y'' + y = 0"
    );
    assert!(
        ode.check_ode_solution(&x.sin(), &y, &x),
        "y = sin(x) should satisfy y'' + y = 0"
    );
    // Wrong solution must be rejected
    assert!(
        !ode.check_ode_solution(&x.exp(), &y, &x),
        "BUG: y = exp(x) should NOT satisfy y'' + y = 0"
    );
}

#[test]
fn ode_second_order_y_pp_minus_y_eq_0() {
    // y'' - y = 0  →  r = ±1  →  C₁·e^x + C₂·e^(-x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &d2y - &y;

    let sol = ode.solve_ode(&y, &x);
    let s = format!("{sol}");

    assert!(
        !sol.has_unevaluated(),
        "BUG: y'' - y = 0 should be solvable, got unevaluated: {s}"
    );

    assert!(
        ode.check_ode_solution(&x.exp(), &y, &x),
        "y = exp(x) should satisfy y'' - y = 0"
    );
    assert!(
        ode.check_ode_solution(&(-&x).exp(), &y, &x),
        "y = exp(-x) should satisfy y'' - y = 0"
    );
}

#[test]
fn ode_overdamped_y_pp_plus_3yp_plus_2y() {
    // y'' + 3y' + 2y = 0  →  r = -1, -2  →  C₁·e^(-x) + C₂·e^(-2x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &(&d2y + &(&dy * 3)) + &(&y * 2);

    let sol = ode.solve_ode(&y, &x);
    let s = format!("{sol}");

    assert!(
        !sol.has_unevaluated(),
        "BUG: y'' + 3y' + 2y = 0 should be solvable, got unevaluated: {s}"
    );

    let exp_neg_x = (-&x).exp();
    assert!(
        ode.check_ode_solution(&exp_neg_x, &y, &x),
        "y = exp(-x) should satisfy y'' + 3y' + 2y = 0"
    );
    let exp_neg_2x = (-&x * 2).exp();
    assert!(
        ode.check_ode_solution(&exp_neg_2x, &y, &x),
        "y = exp(-2x) should satisfy y'' + 3y' + 2y = 0"
    );
}

#[test]
fn ode_classification_first_order_linear() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let dy = y.formal_diff(&x);
    let ode = &dy + &y; // y' + y = 0
    let cls = ode.classify_ode(&y, &x);
    let cls_s = format!("{cls:?}");

    assert!(
        cls_s != "Unknown",
        "y' + y = 0 should not classify as Unknown, got: {cls_s}"
    );
}

#[test]
fn ode_classification_second_order() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &d2y + &y; // y'' + y = 0
    let cls = ode.classify_ode(&y, &x);
    let cls_s = format!("{cls:?}");

    assert!(
        cls_s != "Unknown",
        "y'' + y = 0 should not classify as Unknown, got: {cls_s}"
    );
    assert!(
        cls_s.contains("SecondOrder"),
        "y'' + y = 0 should classify as SecondOrder*, got: {cls_s}"
    );
}

#[test]
fn ode_simple_separable_y_prime_eq_x() {
    // y' = x  →  y = x²/2 + C
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let dy = y.formal_diff(&x);
    let ode = &dy - &x;

    let sol = ode.solve_ode(&y, &x);
    let s = format!("{sol}");

    assert!(
        !sol.has_unevaluated(),
        "y' - x = 0 should be solvable, got unevaluated: {s}"
    );

    let particular = &x.powi(2) / 2;
    assert!(
        ode.check_ode_solution(&particular, &y, &x),
        "y = x²/2 should satisfy y' = x"
    );
}

#[test]
fn ode_full_separable_y_prime_eq_xy() {
    // y' = x·y  →  y = C·exp(x²/2)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let dy = y.formal_diff(&x);
    let ode = &dy - &(&x * &y);

    let sol = ode.solve_ode(&y, &x);
    let s = format!("{sol}");

    assert!(
        !sol.has_unevaluated(),
        "BUG: y' = x·y should be solvable, got unevaluated: {s}"
    );

    // Verify: y = exp(x²/2) should satisfy y' - x·y = 0
    let particular = (&x.powi(2) / 2).exp();
    assert!(
        ode.check_ode_solution(&particular, &y, &x),
        "y = exp(x²/2) should satisfy y' = x·y, solution was: {s}"
    );
}

#[test]
fn ode_general_solution_has_right_number_of_constants() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    // First-order: should have 1 constant
    let dy = y.formal_diff(&x);
    let ode1 = &dy + &y;
    let sol1 = ode1.solve_ode(&y, &x);
    let s1 = format!("{sol1}");
    if !sol1.has_unevaluated() {
        assert!(
            s1.contains("C1"),
            "First-order solution should have C1: {s1}"
        );
    }

    // Second-order: should have 2 constants
    let d2y = dy.formal_diff(&x);
    let ode2 = &d2y + &y;
    let sol2 = ode2.solve_ode(&y, &x);
    let s2 = format!("{sol2}");
    if !sol2.has_unevaluated() {
        assert!(
            s2.contains("C1") && s2.contains("C2"),
            "Second-order solution should have C1 and C2: {s2}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 4: SERIES EXPANSION EDGE CASES
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn series_exp_maclaurin_order_5() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);

    let s = x.exp().series(&x, &zero, 5);
    let expanded = s.expand().eval();

    // At x = 1/2: 1 + 0.5 + 0.125 + 0.02083... + 0.00260... ≈ 1.6484
    let val = expanded.subs(&x, &ctx.rational(1, 2)).eval();
    if let Ok(v) = val.eval_f64() {
        let expected_trunc = 1.0 + 0.5 + 0.125 + 1.0 / 48.0 + 1.0 / 384.0;
        assert!(
            (v - expected_trunc).abs() < 0.001,
            "exp(x) series at x=0.5 should be ≈{expected_trunc}, got: {v}"
        );
    }
}

#[test]
fn series_sin_maclaurin_order_7() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);

    let s = x.sin().series(&x, &zero, 8);
    let expanded = s.expand().eval();

    // At x = 1: sin(1) ≈ 0.8415
    // Truncated: 1 - 1/6 + 1/120 - 1/5040 ≈ 0.84147
    let val = expanded.subs(&x, &ctx.int(1)).eval();
    if let Ok(v) = val.eval_f64() {
        let expected = 1.0 - 1.0 / 6.0 + 1.0 / 120.0 - 1.0 / 5040.0;
        assert!(
            (v - expected).abs() < 0.001,
            "sin(x) series at x=1 should be ≈{expected}, got: {v}"
        );
    }
}

#[test]
fn series_cos_maclaurin_order_6() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);

    let s = x.cos().series(&x, &zero, 7);
    let expanded = s.expand().eval();

    // At x = 1: cos(1) ≈ 0.5403
    // Truncated: 1 - 1/2 + 1/24 - 1/720 ≈ 0.5403
    let val = expanded.subs(&x, &ctx.int(1)).eval();
    if let Ok(v) = val.eval_f64() {
        let expected = 1.0 - 0.5 + 1.0 / 24.0 - 1.0 / 720.0;
        assert!(
            (v - expected).abs() < 0.001,
            "cos(x) series at x=1 should be ≈{expected}, got: {v}"
        );
    }
}

#[test]
fn series_one_over_one_minus_x_geometric() {
    // 1/(1-x) around 0 to order 5: should give 1 + x + x² + x³ + x⁴
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);

    let expr = &ctx.int(1) / &(&ctx.int(1) - &x);
    let s = expr.series(&x, &zero, 5);
    let expanded = s.expand().eval();
    let result = format!("{expanded}");

    // At x = 1/2: 1 + 1/2 + 1/4 + 1/8 + 1/16 = 31/16 = 1.9375
    let val = expanded.subs(&x, &ctx.rational(1, 2)).eval();
    if let Ok(v) = val.eval_f64() {
        let expected = 1.0 + 0.5 + 0.25 + 0.125 + 0.0625;
        assert!(
            (v - expected).abs() < 1e-8,
            "1/(1-x) series at x=1/2 should be {expected}, got: {v} (series: {result})"
        );
    } else {
        panic!("1/(1-x) series should evaluate numerically at x=1/2, series: {result}");
    }
}

#[test]
fn series_sqrt_1_plus_x_binomial() {
    // sqrt(1+x) ≈ 1 + x/2 - x²/8 + x³/16 - ...
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);

    let expr = (&ctx.int(1) + &x).pow(&ctx.rational(1, 2));
    let s = expr.series(&x, &zero, 4);
    let expanded = s.expand().eval();
    let result = format!("{expanded}");

    // At x = 0.5: sqrt(1.5) ≈ 1.22474
    let val = expanded.subs(&x, &ctx.rational(1, 2)).eval();
    if let Ok(v) = val.eval_f64() {
        let expected = (1.5_f64).sqrt();
        assert!(
            (v - expected).abs() < 0.02,
            "sqrt(1+x) series at x=0.5 should be ≈{expected}, got: {v} (series: {result})"
        );
    } else {
        // Even if numerical eval fails, the series should at least exist
        assert!(
            result.contains("x"),
            "sqrt(1+x) series should contain x: {result}"
        );
    }
}

#[test]
fn series_at_nonzero_point_exp() {
    // Taylor series of exp(x) around x = 1 to order 3
    // = e + e(x-1) + e/2·(x-1)² + ...
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);

    let s = x.exp().series(&x, &one, 3);
    let expanded = s.expand().eval();

    // At x=1: should give exactly exp(1) = e
    let val_at_1 = expanded.subs(&x, &ctx.int(1)).eval();
    if let Ok(v) = val_at_1.eval_f64() {
        assert!(
            (v - std::f64::consts::E).abs() < 1e-6,
            "exp(x) series around 1 at x=1 should be e ≈ 2.71828, got: {v}"
        );
    }

    // At x=1.1: should be close to exp(1.1) ≈ 3.00417
    let val_near = expanded.subs(&x, &ctx.rational(11, 10)).eval();
    if let Ok(v) = val_near.eval_f64() {
        let expected = 1.1_f64.exp();
        assert!(
            (v - expected).abs() < 0.01,
            "exp(x) series around 1 at x=1.1 should be ≈{expected}, got: {v}"
        );
    }
}

#[test]
fn series_order_zero_gives_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let s = x.exp().series(&x, &zero, 0);
    let result = format!("{s}");
    assert_eq!(result, "0", "Series of order 0 should be 0, got: {result}");
}

#[test]
fn series_constant_is_itself() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let s = ctx.int(7).series(&x, &zero, 4);
    let result = format!("{s}");
    assert_eq!(
        result, "7",
        "Series of constant 7 should be 7, got: {result}"
    );
}

#[test]
fn series_polynomial_is_exact() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);

    // x³ + 2x + 5 — series to order 6 should reproduce exactly
    let poly = &(&x.powi(3) + &(&x * 2)) + &ctx.int(5);
    let s = poly.series(&x, &zero, 6);
    let expanded = s.expand().eval();

    // Check at x = 2: 8 + 4 + 5 = 17
    let orig_v = poly.subs(&x, &ctx.int(2)).eval().eval_f64().unwrap();
    let series_v = expanded.subs(&x, &ctx.int(2)).eval().eval_f64().unwrap();
    assert!(
        (orig_v - series_v).abs() < 1e-8,
        "Series of polynomial should be exact: original={orig_v}, series={series_v}"
    );
}

#[test]
fn laurent_series_1_over_x() {
    // Laurent series of 1/x around 0 should recover x^{-1}.
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let x = arena.symbol("x");
        let zero = arena.zero();
        let one = arena.one();
        let expr = arena.div(one, x);

        match arena.laurent_series_expr(expr, x, zero, 3) {
            Ok(result) => {
                let d = arena.display(result).to_string();
                // Should contain a negative power of x
                assert!(
                    d.contains("x^-1")
                        || d.contains("x^(-1)")
                        || d.contains("1/x")
                        || d.contains("x^{-1}"),
                    "Laurent of 1/x should have x^(-1) term: {d}"
                );
            }
            Err(e) => {
                eprintln!("NOTE: Laurent series of 1/x not supported: {e}");
            }
        }
    });
}

#[test]
fn laurent_series_sin_x_over_x() {
    // sin(x)/x around 0: has a removable singularity.
    // Laurent series = 1 - x²/6 + x⁴/120 - ...  (no negative powers)
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let x = arena.symbol("x");
        let zero = arena.zero();
        let sin_x = arena.sin(x);
        let expr = arena.div(sin_x, x);

        match arena.laurent_series_expr(expr, x, zero, 4) {
            Ok(result) => {
                let d = arena.display(result).to_string();
                // Should contain "1" as constant term (removable singularity value)
                assert!(
                    d.contains('1'),
                    "Laurent of sin(x)/x should start with 1: {d}"
                );
            }
            Err(e) => {
                eprintln!("NOTE: Laurent series of sin(x)/x not supported: {e}");
            }
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 5: LAPLACE TRANSFORM CORRECTNESS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn laplace_of_1() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    let result = ctx.int(1).laplace(&t, &s);
    // L{1} = 1/s  →  at s=2: 0.5
    let val = result.subs(&s, &ctx.int(2)).eval();
    if let Ok(v) = val.eval_f64() {
        assert!(
            (v - 0.5).abs() < 1e-8,
            "L{{1}} at s=2 should be 0.5, got: {v}"
        );
    }
}

#[test]
fn laplace_of_t() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    let result = t.laplace(&t, &s);
    // L{t} = 1/s²  →  at s=2: 0.25
    let val = result.subs(&s, &ctx.int(2)).eval();
    if let Ok(v) = val.eval_f64() {
        assert!(
            (v - 0.25).abs() < 1e-8,
            "L{{t}} at s=2 should be 0.25, got: {v}"
        );
    }
}

#[test]
fn laplace_of_t_squared() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    // L{t²} = 2/s³
    let result = t.powi(2).laplace(&t, &s);
    assert!(
        !result.has_unevaluated(),
        "L{{t²}} should not be unevaluated: {}",
        result
    );
    // At s=2: 2/8 = 0.25
    let val = result.subs(&s, &ctx.int(2)).eval();
    if let Ok(v) = val.eval_f64() {
        assert!(
            (v - 0.25).abs() < 1e-8,
            "L{{t²}} at s=2 should be 0.25, got: {v}"
        );
    }
}

#[test]
fn laplace_of_t_cubed() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    // L{t³} = 6/s⁴
    let result = t.powi(3).laplace(&t, &s);
    assert!(
        !result.has_unevaluated(),
        "L{{t³}} should not be unevaluated: {}",
        result
    );
    // At s=1: 6/1 = 6
    let val = result.subs(&s, &ctx.int(1)).eval();
    if let Ok(v) = val.eval_f64() {
        assert!(
            (v - 6.0).abs() < 1e-8,
            "L{{t³}} at s=1 should be 6, got: {v}"
        );
    }
}

#[test]
fn laplace_of_exp() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    // L{exp(3t)} = 1/(s-3)
    let result = (&t * 3).exp().laplace(&t, &s);
    // At s=5: 1/(5-3) = 0.5
    let val = result.subs(&s, &ctx.int(5)).eval();
    if let Ok(v) = val.eval_f64() {
        assert!(
            (v - 0.5).abs() < 1e-8,
            "L{{exp(3t)}} at s=5 should be 0.5, got: {v}"
        );
    }
}

#[test]
fn laplace_of_sin() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    // L{sin(2t)} = 2/(s²+4)
    let result = (&t * 2).sin().laplace(&t, &s);
    // At s=0: 2/4 = 0.5
    let val = result.subs(&s, &ctx.int(0)).eval();
    if let Ok(v) = val.eval_f64() {
        assert!(
            (v - 0.5).abs() < 1e-8,
            "L{{sin(2t)}} at s=0 should be 0.5, got: {v}"
        );
    }
}

#[test]
fn laplace_of_cos() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    // L{cos(3t)} = s/(s²+9)
    let result = (&t * 3).cos().laplace(&t, &s);
    // At s=3: 3/(9+9) = 1/6
    let val = result.subs(&s, &ctx.int(3)).eval();
    if let Ok(v) = val.eval_f64() {
        assert!(
            (v - 1.0 / 6.0).abs() < 1e-8,
            "L{{cos(3t)}} at s=3 should be 1/6, got: {v}"
        );
    }
    // At s=0: 0/9 = 0
    let val0 = result.subs(&s, &ctx.int(0)).eval();
    if let Ok(v) = val0.eval_f64() {
        assert!(v.abs() < 1e-8, "L{{cos(3t)}} at s=0 should be 0, got: {v}");
    }
}

#[test]
fn laplace_of_sinh() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    // L{sinh(2t)} = 2/(s²-4)
    let result = (&t * 2).sinh().laplace(&t, &s);
    // At s=3: 2/(9-4) = 2/5 = 0.4
    let val = result.subs(&s, &ctx.int(3)).eval();
    if let Ok(v) = val.eval_f64() {
        assert!(
            (v - 0.4).abs() < 1e-8,
            "L{{sinh(2t)}} at s=3 should be 0.4, got: {v}"
        );
    }
}

#[test]
fn laplace_of_cosh() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    // L{cosh(2t)} = s/(s²-4)
    let result = (&t * 2).cosh().laplace(&t, &s);
    // At s=3: 3/(9-4) = 3/5 = 0.6
    let val = result.subs(&s, &ctx.int(3)).eval();
    if let Ok(v) = val.eval_f64() {
        assert!(
            (v - 0.6).abs() < 1e-8,
            "L{{cosh(2t)}} at s=3 should be 0.6, got: {v}"
        );
    }
}

#[test]
fn laplace_linearity() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    // L{3·exp(t) + 2·t} = 3/(s-1) + 2/s²
    let combined = &(&t.exp() * 3) + &(&t * 2);
    let result = combined.laplace(&t, &s);
    // At s=2: 3/(2-1) + 2/4 = 3 + 0.5 = 3.5
    let val = result.subs(&s, &ctx.int(2)).eval();
    if let Ok(v) = val.eval_f64() {
        assert!(
            (v - 3.5).abs() < 1e-6,
            "L{{3·exp(t) + 2·t}} at s=2 should be 3.5, got: {v}"
        );
    }
}

#[test]
fn inverse_laplace_1_over_s() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    // L⁻¹{1/s} = 1 (constant function, for t > 0)
    let expr = &ctx.int(1) / &s;
    let result = expr.inverse_laplace(&s, &t);
    let d = format!("{result}");

    assert!(
        !result.has_unevaluated(),
        "L⁻¹{{1/s}} should not be unevaluated: {d}"
    );
    if let Ok(v) = result.subs(&t, &ctx.int(1)).eval().eval_f64() {
        assert!(
            (v - 1.0).abs() < 1e-8,
            "L⁻¹{{1/s}} should be 1, got: {v} (display: {d})"
        );
    }
}

#[test]
fn inverse_laplace_1_over_s_squared() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    // L⁻¹{1/s²} = t
    let expr = &ctx.int(1) / &s.powi(2);
    let result = expr.inverse_laplace(&s, &t);
    let d = format!("{result}");

    assert!(
        !result.has_unevaluated(),
        "L⁻¹{{1/s²}} should not be unevaluated: {d}"
    );
    // At t=5: should be 5
    if let Ok(v) = result.subs(&t, &ctx.int(5)).eval().eval_f64() {
        assert!(
            (v - 5.0).abs() < 1e-8,
            "L⁻¹{{1/s²}} at t=5 should be 5, got: {v} (display: {d})"
        );
    }
}

#[test]
fn inverse_laplace_1_over_s_minus_a() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    // L⁻¹{1/(s-2)} = exp(2t)
    let expr = &ctx.int(1) / &(&s - 2);
    let result = expr.inverse_laplace(&s, &t);
    let d = format!("{result}");

    assert!(
        !result.has_unevaluated(),
        "L⁻¹{{1/(s-2)}} should not be unevaluated: {d}"
    );
    // At t=1: exp(2) ≈ 7.389
    if let Ok(v) = result.subs(&t, &ctx.int(1)).eval().eval_f64() {
        let expected = 2.0_f64.exp();
        assert!(
            (v - expected).abs() < 1e-4,
            "L⁻¹{{1/(s-2)}} at t=1 should be exp(2) ≈ {expected}, got: {v}"
        );
    }
}

#[test]
fn inverse_laplace_omega_over_s2_plus_omega2() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    // L⁻¹{2/(s²+4)} = sin(2t)
    let expr = &ctx.int(2) / &(&s.powi(2) + &ctx.int(4));
    let result = expr.inverse_laplace(&s, &t);
    let d = format!("{result}");

    assert!(
        !result.has_unevaluated(),
        "L⁻¹{{2/(s²+4)}} should not be unevaluated: {d}"
    );
    // At t=π/4: sin(π/2) = 1.  Substitute the same rational approximation
    // of π/4 on both sides so the comparison is exact in the approximation.
    let (t_num, t_den) = (7854_i64, 10000_i64);
    let t_val = t_num as f64 / t_den as f64;
    if let Ok(v) = result
        .subs(&t, &ctx.rational(t_num, t_den))
        .eval()
        .eval_f64()
    {
        let expected = (2.0 * t_val).sin();
        assert!(
            (v - expected).abs() < 0.01,
            "L⁻¹{{2/(s²+4)}} at t≈π/4 should be ≈{expected}, got: {v}"
        );
    }
}

#[test]
fn inverse_laplace_s_over_s2_plus_omega2() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    // L⁻¹{s/(s²+4)} = cos(2t)
    let expr = &s / &(&s.powi(2) + &ctx.int(4));
    let result = expr.inverse_laplace(&s, &t);
    let d = format!("{result}");

    assert!(
        !result.has_unevaluated(),
        "L⁻¹{{s/(s²+4)}} should not be unevaluated: {d}"
    );
    // At t=0: cos(0) = 1
    if let Ok(v) = result.subs(&t, &ctx.int(0)).eval().eval_f64() {
        assert!(
            (v - 1.0).abs() < 1e-6,
            "L⁻¹{{s/(s²+4)}} at t=0 should be 1, got: {v}"
        );
    }
}

#[test]
fn laplace_roundtrip_exp() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    let original = (&t * 2).exp();
    let transformed = original.laplace(&t, &s);
    let recovered = transformed.inverse_laplace(&s, &t);

    if !recovered.has_unevaluated() {
        let ov = original.subs(&t, &ctx.int(1)).eval().eval_f64();
        let rv = recovered.subs(&t, &ctx.int(1)).eval().eval_f64();
        if let (Ok(o), Ok(r)) = (ov, rv) {
            assert!(
                (o - r).abs() < 1e-6,
                "Laplace roundtrip exp(2t): orig={o}, recovered={r} (display: {recovered})"
            );
        }
    } else {
        panic!("Laplace roundtrip of exp(2t) gave unevaluated form: {recovered}");
    }
}

#[test]
fn laplace_roundtrip_sin() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    let original = (&t * 3).sin();
    let transformed = original.laplace(&t, &s);
    let recovered = transformed.inverse_laplace(&s, &t);

    if !recovered.has_unevaluated() {
        let ov = original.subs(&t, &ctx.int(1)).eval().eval_f64();
        let rv = recovered.subs(&t, &ctx.int(1)).eval().eval_f64();
        if let (Ok(o), Ok(r)) = (ov, rv) {
            assert!(
                (o - r).abs() < 1e-6,
                "Laplace roundtrip sin(3t): orig={o}, recovered={r} (display: {recovered})"
            );
        }
    } else {
        panic!("Laplace roundtrip of sin(3t) gave unevaluated form: {recovered}");
    }
}

#[test]
fn laplace_roundtrip_cos() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    let original = (&t * 5).cos();
    let transformed = original.laplace(&t, &s);
    let recovered = transformed.inverse_laplace(&s, &t);

    if !recovered.has_unevaluated() {
        let ov = original.subs(&t, &ctx.int(1)).eval().eval_f64();
        let rv = recovered.subs(&t, &ctx.int(1)).eval().eval_f64();
        if let (Ok(o), Ok(r)) = (ov, rv) {
            assert!(
                (o - r).abs() < 1e-6,
                "Laplace roundtrip cos(5t): orig={o}, recovered={r} (display: {recovered})"
            );
        }
    } else {
        panic!("Laplace roundtrip of cos(5t) gave unevaluated form: {recovered}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 5b: Z-TRANSFORM CORRECTNESS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn z_transform_constant() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let z = ctx.symbol("z");

    // Z{1} = z/(z-1)
    let result = ctx.int(1).z_transform(&n, &z).expect("Z{1} should succeed");
    // At z=2: 2/(2-1) = 2
    let val = result.subs(&z, &ctx.int(2)).eval();
    if let Ok(v) = val.eval_f64() {
        assert!(
            (v - 2.0).abs() < 1e-8,
            "Z{{1}} at z=2 should be 2, got: {v}"
        );
    }
}

#[test]
fn z_transform_a_to_n() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let z = ctx.symbol("z");

    // Z{(1/2)^n} = z/(z - 1/2)
    let half = ctx.rational(1, 2);
    let result = half
        .pow(&n)
        .z_transform(&n, &z)
        .expect("Z{(1/2)^n} should succeed");
    // At z=2: 2/(2-0.5) = 2/1.5 = 4/3
    let val = result.subs(&z, &ctx.int(2)).eval();
    if let Ok(v) = val.eval_f64() {
        assert!(
            (v - 4.0 / 3.0).abs() < 1e-6,
            "Z{{(1/2)^n}} at z=2 should be 4/3, got: {v}"
        );
    }
}

#[test]
fn z_transform_n() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let z = ctx.symbol("z");

    // Z{n} = z/(z-1)²
    let result = n.z_transform(&n, &z).expect("Z{n} should succeed");
    // At z=2: 2/(2-1)² = 2
    let val = result.subs(&z, &ctx.int(2)).eval();
    if let Ok(v) = val.eval_f64() {
        assert!(
            (v - 2.0).abs() < 1e-8,
            "Z{{n}} at z=2 should be 2, got: {v}"
        );
    }
    // At z=3: 3/(3-1)² = 3/4 = 0.75
    let val3 = result.subs(&z, &ctx.int(3)).eval();
    if let Ok(v) = val3.eval_f64() {
        assert!(
            (v - 0.75).abs() < 1e-8,
            "Z{{n}} at z=3 should be 0.75, got: {v}"
        );
    }
}

#[test]
fn z_transform_sin() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let z = ctx.symbol("z");

    // Z{sin(ωn)} with ω = 1
    let expr = n.sin(); // sin(1·n) = sin(n)
    let result = expr.z_transform(&n, &z);

    match result {
        Ok(r) => {
            // Verify numerically: Σ_{k=0}^{N} sin(k)·z^{-k}
            let z_val = 3.0_f64;
            let mut direct_sum = 0.0;
            for k in 0..50 {
                direct_sum += (k as f64).sin() * z_val.powi(-k);
            }
            if let Ok(v) = r.subs(&z, &ctx.int(3)).eval().eval_f64() {
                assert!(
                    (v - direct_sum).abs() < 1e-4,
                    "Z{{sin(n)}} at z=3: transform={v}, partial sum={direct_sum}"
                );
            }
        }
        Err(e) => {
            panic!("BUG: Z{{sin(n)}} should be computable: {e}");
        }
    }
}

#[test]
fn z_transform_cos() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let z = ctx.symbol("z");

    // Z{cos(ωn)} with ω = 1
    let expr = n.cos();
    let result = expr.z_transform(&n, &z);

    match result {
        Ok(r) => {
            // Verify: Σ_{k=0}^{N} cos(k)·z^{-k}
            let z_val = 3.0_f64;
            let mut direct_sum = 0.0;
            for k in 0..50 {
                direct_sum += (k as f64).cos() * z_val.powi(-k);
            }
            if let Ok(v) = r.subs(&z, &ctx.int(3)).eval().eval_f64() {
                assert!(
                    (v - direct_sum).abs() < 1e-4,
                    "Z{{cos(n)}} at z=3: transform={v}, partial sum={direct_sum}"
                );
            }
        }
        Err(e) => {
            panic!("BUG: Z{{cos(n)}} should be computable: {e}");
        }
    }
}

#[test]
fn z_transform_roundtrip_exponential() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let z = ctx.symbol("z");

    // Z{2^n} = z/(z-2),  then Z⁻¹ should recover 2^n
    let two_n = ctx.int(2).pow(&n);
    let transformed = two_n.z_transform(&n, &z).expect("Z{2^n} should succeed");
    let recovered = transformed
        .inverse_z_transform(&z, &n)
        .expect("Z⁻¹{z/(z-2)} should succeed");

    // At n=3: 2^3 = 8
    let orig_val = two_n.subs(&n, &ctx.int(3)).eval().eval_f64();
    let recov_val = recovered.subs(&n, &ctx.int(3)).eval().eval_f64();

    if let (Ok(o), Ok(r)) = (orig_val, recov_val) {
        assert!(
            (o - r).abs() < 1e-6,
            "Z roundtrip of 2^n at n=3: original={o}, recovered={r}"
        );
    }
}

#[test]
fn z_transform_roundtrip_half_n() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let z = ctx.symbol("z");

    // Z{(1/3)^n}, then Z⁻¹
    let third_n = ctx.rational(1, 3).pow(&n);
    let transformed = third_n
        .z_transform(&n, &z)
        .expect("Z{(1/3)^n} should succeed");
    let recovered = transformed
        .inverse_z_transform(&z, &n)
        .expect("Z⁻¹ should succeed");

    // At n=4: (1/3)^4 = 1/81
    let orig_val = third_n.subs(&n, &ctx.int(4)).eval().eval_f64();
    let recov_val = recovered.subs(&n, &ctx.int(4)).eval().eval_f64();

    if let (Ok(o), Ok(r)) = (orig_val, recov_val) {
        assert!(
            (o - r).abs() < 1e-6,
            "Z roundtrip of (1/3)^n at n=4: original={o}, recovered={r}"
        );
    }
}

#[test]
fn inverse_z_transform_z_over_z_minus_a() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let z = ctx.symbol("z");

    // Z⁻¹{z/(z-3)} = 3^n
    let expr = &z / &(&z - 3);
    let result = expr
        .inverse_z_transform(&z, &n)
        .expect("Z⁻¹{z/(z-3)} should succeed");

    // At n=4: 3^4 = 81
    let val = result.subs(&n, &ctx.int(4)).eval();
    if let Ok(v) = val.eval_f64() {
        assert!(
            (v - 81.0).abs() < 1e-6,
            "Z⁻¹{{z/(z-3)}} at n=4 should be 81, got: {v}"
        );
    }
}

#[test]
fn z_transform_linearity() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let z = ctx.symbol("z");

    // Z{3·(1/2)^n + 2} should work via linearity
    let half_n = ctx.rational(1, 2).pow(&n);
    let expr = &(&half_n * 3) + &ctx.int(2);
    let result = expr.z_transform(&n, &z);

    match result {
        Ok(r) => {
            // At z=4: 3·[4/(4-0.5)] + 2·[4/(4-1)] = 3·(8/7) + 2·(4/3)
            //       = 24/7 + 8/3 = 72/21 + 56/21 = 128/21
            let val = r.subs(&z, &ctx.int(4)).eval();
            if let Ok(v) = val.eval_f64() {
                let expected = 3.0 * (4.0 / 3.5) + 2.0 * (4.0 / 3.0);
                assert!(
                    (v - expected).abs() < 1e-4,
                    "Z{{3·(1/2)^n + 2}} at z=4 should be ≈{expected}, got: {v}"
                );
            }
        }
        Err(e) => {
            panic!("Z-transform of 3·(1/2)^n + 2 should succeed: {e}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 6: FOURIER TRANSFORM CORRECTNESS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fourier_delta_gives_1() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let t = arena.symbol("t");
        let omega = arena.symbol("omega");
        let delta_t = arena.dirac_delta(t);

        let result = arena
            .fourier_transform_expr(delta_t, t, omega)
            .expect("F{δ(t)} should succeed");
        assert_eq!(
            result,
            arena.one(),
            "F{{δ(t)}} should be 1, got: {}",
            arena.display(result)
        );
    });
}

#[test]
fn fourier_constant_gives_2pi_delta() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let t = arena.symbol("t");
        let omega = arena.symbol("omega");
        let five = arena.int(5);

        let result = arena
            .fourier_transform_expr(five, t, omega)
            .expect("F{5} should succeed");
        let d = arena.display(result).to_string();
        // F{5} = 10π·δ(ω)
        assert!(
            d.contains("DiracDelta") || d.contains("delta") || d.contains("pi"),
            "F{{5}} should contain delta or pi: {d}"
        );
    });
}

#[test]
fn fourier_sin_gives_delta_pair() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let t = arena.symbol("t");
        let omega = arena.symbol("omega");
        let three = arena.int(3);
        let three_t = arena.mul(&[three, t]);
        let sin_3t = arena.sin(three_t);

        let result = arena
            .fourier_transform_expr(sin_3t, t, omega)
            .expect("F{sin(3t)} should succeed");
        let d = arena.display(result).to_string();
        // F{sin(3t)} = iπ[δ(ω+3) - δ(ω-3)]
        assert!(
            d.contains("DiracDelta") || d.contains("delta"),
            "F{{sin(3t)}} should contain delta: {d}"
        );
    });
}

#[test]
fn fourier_cos_gives_delta_pair() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let t = arena.symbol("t");
        let omega = arena.symbol("omega");
        let two = arena.int(2);
        let two_t = arena.mul(&[two, t]);
        let cos_2t = arena.cos(two_t);

        let result = arena
            .fourier_transform_expr(cos_2t, t, omega)
            .expect("F{cos(2t)} should succeed");
        let d = arena.display(result).to_string();
        assert!(
            d.contains("DiracDelta") || d.contains("delta"),
            "F{{cos(2t)}} should contain delta: {d}"
        );
    });
}

#[test]
fn fourier_exp_heaviside() {
    // F{exp(-3t)·H(t)} = 1/(iω + 3)
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let t = arena.symbol("t");
        let omega = arena.symbol("omega");
        let neg3 = arena.int(-3);
        let neg3_t = arena.mul(&[neg3, t]);
        let exp_neg3t = arena.exp(neg3_t);
        let h_t = arena.heaviside(t);
        let expr = arena.mul(&[exp_neg3t, h_t]);

        let result = arena
            .fourier_transform_expr(expr, t, omega)
            .expect("F{exp(-3t)·H(t)} should succeed");
        let d = arena.display(result).to_string();
        assert!(
            d.contains("i") || d.contains("omega") || d.contains("3"),
            "F{{exp(-3t)·H(t)}} should be 1/(iω+3), got: {d}"
        );
    });
}

#[test]
fn fourier_heaviside_alone() {
    // F{H(t)} = πδ(ω) + 1/(iω)
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let t = arena.symbol("t");
        let omega = arena.symbol("omega");
        let h_t = arena.heaviside(t);

        let result = arena
            .fourier_transform_expr(h_t, t, omega)
            .expect("F{H(t)} should succeed");
        let d = arena.display(result).to_string();
        assert!(
            (d.contains("DiracDelta") || d.contains("delta")) && d.contains("pi"),
            "F{{H(t)}} should contain πδ(ω): {d}"
        );
    });
}

#[test]
fn fourier_linearity_scaled_delta() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let t = arena.symbol("t");
        let omega = arena.symbol("omega");
        let seven = arena.int(7);
        let delta_t = arena.dirac_delta(t);
        let expr = arena.mul(&[seven, delta_t]);

        let result = arena
            .fourier_transform_expr(expr, t, omega)
            .expect("F{7·δ(t)} should succeed");
        let d = arena.display(result).to_string();
        assert_eq!(d, "7", "F{{7·δ(t)}} should be 7, got: {d}");
    });
}

#[test]
fn fourier_neg_delta() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let t = arena.symbol("t");
        let omega = arena.symbol("omega");
        let delta_t = arena.dirac_delta(t);
        let neg_delta = arena.neg(delta_t);

        let result = arena
            .fourier_transform_expr(neg_delta, t, omega)
            .expect("F{-δ(t)} should succeed");
        let d = arena.display(result).to_string();
        assert!(
            d.contains("-1") || d.contains("−1"),
            "F{{-δ(t)}} should be -1, got: {d}"
        );
    });
}

#[test]
fn fourier_roundtrip_delta() {
    // F{δ(t)} = 1,  F⁻¹{1} should give back δ(t) (possibly scaled)
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let t = arena.symbol("t");
        let omega = arena.symbol("omega");
        let delta_t = arena.dirac_delta(t);

        let forward = arena
            .fourier_transform_expr(delta_t, t, omega)
            .expect("F{δ(t)} forward");
        assert_eq!(forward, arena.one());

        let inverse = arena
            .inverse_fourier_transform_expr(forward, omega, t)
            .expect("F⁻¹{1} inverse");
        let d = arena.display(inverse).to_string();
        assert!(
            d.contains("DiracDelta") || d.contains("delta"),
            "F⁻¹{{1}} should contain δ(t), got: {d}"
        );
    });
}

#[test]
fn inverse_fourier_of_delta_omega() {
    // F⁻¹{δ(ω)} = 1/(2π)
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let t = arena.symbol("t");
        let omega = arena.symbol("omega");
        let delta_omega = arena.dirac_delta(omega);

        let result = arena
            .inverse_fourier_transform_expr(delta_omega, omega, t)
            .expect("F⁻¹{δ(ω)} should succeed");
        let d = arena.display(result).to_string();
        assert!(
            d.contains("pi") || d.contains("π"),
            "F⁻¹{{δ(ω)}} should contain π (result = 1/(2π)): {d}"
        );
    });
}

#[test]
fn fourier_sum_of_terms() {
    // Linearity: F{δ(t) + H(t)} should succeed
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let t = arena.symbol("t");
        let omega = arena.symbol("omega");
        let delta_t = arena.dirac_delta(t);
        let h_t = arena.heaviside(t);
        let expr = arena.add(&[delta_t, h_t]);

        let result = arena.fourier_transform_expr(expr, t, omega);
        assert!(result.is_ok(), "F{{δ(t) + H(t)}} should succeed");
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 7: CONVERGENCE TESTING
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn convergence_p_series_p2_converges() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = k.powi(-2); // 1/k²
    assert_eq!(
        body.is_convergent(&k),
        Some(true),
        "Σ 1/k² should converge (p=2 > 1)"
    );
}

#[test]
fn convergence_p_series_p3_converges() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = k.powi(-3);
    assert_eq!(body.is_convergent(&k), Some(true), "Σ 1/k³ should converge");
}

#[test]
fn convergence_harmonic_diverges() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = k.powi(-1); // 1/k
    assert_eq!(
        body.is_convergent(&k),
        Some(false),
        "Σ 1/k should diverge (harmonic series)"
    );
}

#[test]
fn convergence_p_half_diverges() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = k.pow(&ctx.rational(-1, 2)); // 1/√k
    assert_eq!(
        body.is_convergent(&k),
        Some(false),
        "Σ 1/√k should diverge (p=0.5 < 1)"
    );
}

#[test]
fn convergence_geometric_third_converges() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = ctx.rational(1, 3).pow(&k); // (1/3)^k
    assert_eq!(
        body.is_convergent(&k),
        Some(true),
        "Σ (1/3)^k should converge (|r| < 1)"
    );
}

#[test]
fn convergence_geometric_half_converges() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = ctx.rational(1, 2).pow(&k);
    assert_eq!(
        body.is_convergent(&k),
        Some(true),
        "Σ (1/2)^k should converge"
    );
}

#[test]
fn convergence_geometric_2_diverges() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = ctx.int(2).pow(&k); // 2^k
    assert_eq!(
        body.is_convergent(&k),
        Some(false),
        "Σ 2^k should diverge (|r| ≥ 1)"
    );
}

#[test]
fn convergence_geometric_neg_half_converges() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = ctx.rational(-1, 2).pow(&k); // (-1/2)^k
    assert_eq!(
        body.is_convergent(&k),
        Some(true),
        "Σ (-1/2)^k should converge (|r| = 1/2 < 1)"
    );
}

#[test]
fn convergence_constant_zero_converges() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = ctx.int(0);
    assert_eq!(body.is_convergent(&k), Some(true), "Σ 0 should converge");
}

#[test]
fn convergence_constant_nonzero_diverges() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = ctx.int(7);
    assert_eq!(body.is_convergent(&k), Some(false), "Σ 7 should diverge");
}

#[test]
fn convergence_growing_k_squared_diverges() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = k.powi(2); // k² (grows)
    assert_eq!(body.is_convergent(&k), Some(false), "Σ k² should diverge");
}

#[test]
fn convergence_geometric_base_1_diverges() {
    // Σ 1^k = Σ 1 diverges
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = ctx.int(1).pow(&k);
    // 1^k = 1 for all k, so this is Σ 1 which diverges
    let result = body.is_convergent(&k);
    // The engine might recognize 1^k as constant 1 or as geometric with |r|=1
    // Either way it should diverge (or be inconclusive if 1^k is simplified to 1)
    assert!(
        result == Some(false) || result.is_none(),
        "Σ 1^k should diverge or be inconclusive, got: {result:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 8: CROSS-DOMAIN CONSISTENCY
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_of_integral_is_identity() {
    // d/dx(∫ f dx) = f  for basic functions
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let functions: Vec<(&str, Ex)> = vec![
        ("sin(x)", x.sin()),
        ("cos(x)", x.cos()),
        ("exp(x)", x.exp()),
        ("x^2", x.powi(2)),
        ("x^3", x.powi(3)),
    ];

    for (name, f) in &functions {
        let antideriv = f.integrate(&x);
        let roundtrip = antideriv.diff(&x);

        let orig_val = f.subs(&x, &ctx.int(1)).eval().eval_f64();
        let rt_val = roundtrip.subs(&x, &ctx.int(1)).eval().eval_f64();

        if let (Ok(o), Ok(r)) = (orig_val, rt_val) {
            assert!(
                (o - r).abs() < 1e-8,
                "d/dx(∫{name}dx) ≠ {name} at x=1: got {r}, expected {o}"
            );
        }
    }
}

#[test]
fn integral_of_diff_recovers_up_to_constant() {
    // ∫(df/dx)dx and f differ by a constant
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let f = &x.powi(3) + &(&x.powi(2) * 2) + &x; // x³ + 2x² + x
    let df = f.diff(&x);
    let integral_df = df.integrate(&x);

    let v1_f = f.subs(&x, &ctx.int(1)).eval().eval_f64().unwrap();
    let v1_i = integral_df.subs(&x, &ctx.int(1)).eval().eval_f64().unwrap();
    let diff1 = v1_f - v1_i;

    let v2_f = f.subs(&x, &ctx.int(2)).eval().eval_f64().unwrap();
    let v2_i = integral_df.subs(&x, &ctx.int(2)).eval().eval_f64().unwrap();
    let diff2 = v2_f - v2_i;

    assert!(
        (diff1 - diff2).abs() < 1e-8,
        "∫(df/dx)dx and f should differ by a constant: diff@1={diff1}, diff@2={diff2}"
    );
}

#[test]
fn series_of_polynomial_is_exact() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);

    let poly = &(&x.powi(3) + &(&x.powi(2) * 2)) + &(&x * 3);
    let s = poly.series(&x, &zero, 6);
    let expanded = s.expand().eval();

    // At x=2: 8 + 8 + 6 = 22
    let vo = poly.subs(&x, &ctx.int(2)).eval().eval_f64().unwrap();
    let vs = expanded.subs(&x, &ctx.int(2)).eval().eval_f64().unwrap();
    assert!(
        (vo - vs).abs() < 1e-8,
        "Series of polynomial should be exact: orig={vo}, series={vs}"
    );
}

#[test]
fn laplace_derivative_property() {
    // L{f'(t)} = s·F(s) - f(0)
    // For f(t) = t², f'(t) = 2t, f(0) = 0
    // L{2t} should equal s·L{t²}
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    let l_2t = (&t * 2).laplace(&t, &s);
    let l_t2 = t.powi(2).laplace(&t, &s);
    let rhs = &s * &l_t2;

    // At s=3: L{2t}(3) and s·L{t²}(3) should be equal
    let lv = l_2t.subs(&s, &ctx.int(3)).eval().eval_f64();
    let rv = rhs.subs(&s, &ctx.int(3)).eval().eval_f64();

    if let (Ok(l), Ok(r)) = (lv, rv) {
        assert!(
            (l - r).abs() < 1e-6,
            "Laplace derivative property: L{{2t}}={l}, s·L{{t²}}={r}"
        );
    }
}

#[test]
fn definite_integral_reversed_bounds_negate() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let fwd = x.powi(2).integrate_definite(&x, &ctx.int(0), &ctx.int(3));
    let rev = x.powi(2).integrate_definite(&x, &ctx.int(3), &ctx.int(0));
    let fv = fwd.eval_f64().expect("forward evals");
    let rv = rev.eval_f64().expect("reversed evals");
    assert!(
        (fv + rv).abs() < 1e-10,
        "Reversing bounds should negate: fwd={fv}, rev={rv}"
    );
}

#[test]
fn definite_integral_same_bounds_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.sin().integrate_definite(&x, &ctx.int(5), &ctx.int(5));
    let v = result.eval_f64().expect("same-bounds evals");
    assert!(v.abs() < 1e-10, "∫_a^a f dx should be 0, got: {v}");
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 9: DEEPER INVERSE LAPLACE EDGE CASES
//
// Bug #1 found: L⁻¹{1/s²} fails because inverse_degree2 returns None
// when ω²=0 (denominator s² has no constant term).
// Probe more cases around this family.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn inverse_laplace_1_over_s_cubed() {
    // L⁻¹{1/s³} = t²/2
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    let expr = &ctx.int(1) / &s.powi(3);
    let result = expr.inverse_laplace(&s, &t);
    let d = format!("{result}");

    assert!(
        !result.has_unevaluated(),
        "BUG: L⁻¹{{1/s³}} should not be unevaluated: {d}\n\
         Expected: t²/2. The inverse_power_form handler may not recognise s³ \
         when the expression is represented as s^(-3)."
    );

    if let Ok(v) = result.subs(&t, &ctx.int(3)).eval().eval_f64() {
        // t²/2 at t=3 → 4.5
        assert!(
            (v - 4.5).abs() < 1e-6,
            "L⁻¹{{1/s³}} at t=3 should be 4.5, got: {v}"
        );
    }
}

#[test]
fn inverse_laplace_1_over_s_fourth() {
    // L⁻¹{1/s⁴} = t³/6
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    let expr = &ctx.int(1) / &s.powi(4);
    let result = expr.inverse_laplace(&s, &t);
    let d = format!("{result}");

    assert!(
        !result.has_unevaluated(),
        "BUG: L⁻¹{{1/s⁴}} should not be unevaluated: {d}\n\
         Expected: t³/6."
    );

    if let Ok(v) = result.subs(&t, &ctx.int(2)).eval().eval_f64() {
        // t³/6 at t=2 → 8/6 ≈ 1.333
        let expected = 8.0 / 6.0;
        assert!(
            (v - expected).abs() < 1e-6,
            "L⁻¹{{1/s⁴}} at t=2 should be {expected}, got: {v}"
        );
    }
}

#[test]
fn inverse_laplace_2_over_s_cubed() {
    // L⁻¹{2/s³} = t²   (since L{t^n} = n!/s^{n+1}, so L⁻¹{2/s³} = t²)
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    let expr = &ctx.int(2) / &s.powi(3);
    let result = expr.inverse_laplace(&s, &t);
    let d = format!("{result}");

    assert!(
        !result.has_unevaluated(),
        "BUG: L⁻¹{{2/s³}} should not be unevaluated: {d}\n\
         Expected: t²."
    );

    if let Ok(v) = result.subs(&t, &ctx.int(3)).eval().eval_f64() {
        assert!(
            (v - 9.0).abs() < 1e-6,
            "L⁻¹{{2/s³}} at t=3 should be 9, got: {v}"
        );
    }
}

#[test]
fn inverse_laplace_s_over_s2_minus_a2() {
    // L⁻¹{s/(s²-9)} = cosh(3t)
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    let expr = &s / &(&s.powi(2) - &ctx.int(9));
    let result = expr.inverse_laplace(&s, &t);
    let d = format!("{result}");

    // This tests inverse_degree2 with b=0, c<0 (s²-9 → c = -9)
    // That branch currently returns None because !c.is_positive().
    assert!(
        !result.has_unevaluated(),
        "BUG: L⁻¹{{s/(s²-9)}} should not be unevaluated: {d}\n\
         Expected: cosh(3t). The inverse_degree2 handler rejects denominators \
         s²-a² (negative constant term) but this should give cosh."
    );

    if let Ok(v) = result.subs(&t, &ctx.int(0)).eval().eval_f64() {
        // cosh(0) = 1
        assert!(
            (v - 1.0).abs() < 1e-6,
            "L⁻¹{{s/(s²-9)}} at t=0 should be 1 (cosh(0)), got: {v}"
        );
    }
}

#[test]
fn inverse_laplace_a_over_s2_minus_a2() {
    // L⁻¹{3/(s²-9)} = sinh(3t)
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    let expr = &ctx.int(3) / &(&s.powi(2) - &ctx.int(9));
    let result = expr.inverse_laplace(&s, &t);
    let d = format!("{result}");

    assert!(
        !result.has_unevaluated(),
        "BUG: L⁻¹{{3/(s²-9)}} should not be unevaluated: {d}\n\
         Expected: sinh(3t). Same root cause as s/(s²-9) — the handler \
         rejects negative constant term in denominator."
    );

    if let Ok(v) = result.subs(&t, &ctx.int(0)).eval().eval_f64() {
        // sinh(0) = 0
        assert!(
            v.abs() < 1e-6,
            "L⁻¹{{3/(s²-9)}} at t=0 should be 0 (sinh(0)), got: {v}"
        );
    }
}

#[test]
fn inverse_laplace_repeated_root_s_minus_2_squared() {
    // L⁻¹{1/(s-2)²} = t·exp(2t)
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    let expr = &ctx.int(1) / &(&s - 2).powi(2);
    let result = expr.inverse_laplace(&s, &t);
    let d = format!("{result}");

    assert!(
        !result.has_unevaluated(),
        "BUG: L⁻¹{{1/(s-2)²}} should not be unevaluated: {d}\n\
         Expected: t·exp(2t)."
    );

    if let Ok(v) = result.subs(&t, &ctx.int(1)).eval().eval_f64() {
        // t·exp(2t) at t=1 → 1·e² ≈ 7.389
        let expected = 2.0_f64.exp();
        assert!(
            (v - expected).abs() < 1e-3,
            "L⁻¹{{1/(s-2)²}} at t=1 should be e² ≈ {expected}, got: {v}"
        );
    }
}

#[test]
fn laplace_roundtrip_t() {
    // L{t} = 1/s², then L⁻¹{1/s²} should = t
    // This is the roundtrip form of Bug #1.
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    let forward = t.laplace(&t, &s);
    let d_fwd = format!("{forward}");
    assert!(!forward.has_unevaluated(), "L{{t}} should succeed: {d_fwd}");

    let roundtrip = forward.inverse_laplace(&s, &t);
    let d_rt = format!("{roundtrip}");

    assert!(
        !roundtrip.has_unevaluated(),
        "BUG: Laplace roundtrip of t failed.\n\
         L{{t}} = {d_fwd}\n\
         L⁻¹{{L{{t}}}} = {d_rt} (unevaluated)\n\
         Expected: t"
    );

    if let Ok(v) = roundtrip.subs(&t, &ctx.int(5)).eval().eval_f64() {
        assert!(
            (v - 5.0).abs() < 1e-6,
            "Laplace roundtrip of t at t=5 should be 5, got: {v}"
        );
    }
}

#[test]
fn laplace_roundtrip_t_squared() {
    // L{t²} = 2/s³, then L⁻¹{2/s³} should = t²
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    let forward = t.powi(2).laplace(&t, &s);
    assert!(!forward.has_unevaluated(), "L{{t²}} should succeed");

    let roundtrip = forward.inverse_laplace(&s, &t);
    let d_rt = format!("{roundtrip}");

    assert!(
        !roundtrip.has_unevaluated(),
        "BUG: Laplace roundtrip of t² failed.\n\
         L{{t²}} = {forward}\n\
         L⁻¹ = {d_rt} (unevaluated)\n\
         Expected: t²"
    );

    if let Ok(v) = roundtrip.subs(&t, &ctx.int(3)).eval().eval_f64() {
        assert!(
            (v - 9.0).abs() < 1e-6,
            "Laplace roundtrip of t² at t=3 should be 9, got: {v}"
        );
    }
}

#[test]
fn laplace_roundtrip_t_cubed() {
    // L{t³} = 6/s⁴, then L⁻¹{6/s⁴} should = t³
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    let forward = t.powi(3).laplace(&t, &s);
    assert!(!forward.has_unevaluated(), "L{{t³}} should succeed");

    let roundtrip = forward.inverse_laplace(&s, &t);
    let d_rt = format!("{roundtrip}");

    assert!(
        !roundtrip.has_unevaluated(),
        "BUG: Laplace roundtrip of t³ failed.\n\
         L{{t³}} = {forward}\n\
         L⁻¹ = {d_rt} (unevaluated)\n\
         Expected: t³"
    );

    if let Ok(v) = roundtrip.subs(&t, &ctx.int(2)).eval().eval_f64() {
        assert!(
            (v - 8.0).abs() < 1e-6,
            "Laplace roundtrip of t³ at t=2 should be 8, got: {v}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 10: ODE VERIFICATION EDGE CASES
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_check_rejects_constant_for_nontrivial_ode() {
    // y' + y = 0: y = 5 (constant) is NOT a solution
    // y' = 0, y = 5, so y' + y = 5 ≠ 0
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let dy = y.formal_diff(&x);
    let ode = &dy + &y;

    assert!(
        !ode.check_ode_solution(&ctx.int(5), &y, &x),
        "BUG: y = 5 should NOT satisfy y' + y = 0"
    );
}

#[test]
fn ode_check_accepts_zero_for_homogeneous() {
    // y' + y = 0: y = 0 IS a (trivial) solution
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let dy = y.formal_diff(&x);
    let ode = &dy + &y;

    assert!(
        ode.check_ode_solution(&ctx.int(0), &y, &x),
        "BUG: y = 0 should satisfy y' + y = 0 (trivial solution)"
    );
}

#[test]
fn ode_check_second_order_sum_of_solutions() {
    // y'' + y = 0: if sin(x) and cos(x) are solutions,
    // then 3sin(x) + 2cos(x) should also be a solution (linearity)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &d2y + &y;

    let combo = &(&x.sin() * 3) + &(&x.cos() * 2);
    assert!(
        ode.check_ode_solution(&combo, &y, &x),
        "BUG: 3sin(x) + 2cos(x) should satisfy y'' + y = 0"
    );
}

#[test]
fn ode_repeated_root_y_pp_plus_2yp_plus_y() {
    // y'' + 2y' + y = 0  →  (r+1)² = 0  →  r = -1 (repeated)
    // Solution: (C₁ + C₂·x)·e^(-x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &(&d2y + &(&dy * 2)) + &y;

    let sol = ode.solve_ode(&y, &x);
    let s = format!("{sol}");

    assert!(
        !sol.has_unevaluated(),
        "BUG: y'' + 2y' + y = 0 (repeated root) should be solvable: {s}"
    );

    // Verify: y = e^(-x) should satisfy it
    let exp_neg_x = (-&x).exp();
    assert!(
        ode.check_ode_solution(&exp_neg_x, &y, &x),
        "y = e^(-x) should satisfy y'' + 2y' + y = 0"
    );

    // Verify: y = x·e^(-x) should also satisfy it
    let x_exp_neg_x = &x * &(-&x).exp();
    assert!(
        ode.check_ode_solution(&x_exp_neg_x, &y, &x),
        "y = x·e^(-x) should satisfy y'' + 2y' + y = 0"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 11: SERIES NUMERICAL ACCURACY
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn series_exp_convergence_increases_with_order() {
    // Higher-order series of exp(x) at x=1 should get closer to e.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let target = std::f64::consts::E;

    let mut prev_err = f64::MAX;
    for order in [3, 5, 8, 12] {
        let s = x.exp().series(&x, &zero, order);
        let expanded = s.expand().eval();
        if let Ok(v) = expanded.subs(&x, &ctx.int(1)).eval().eval_f64() {
            let err = (v - target).abs();
            assert!(
                err < prev_err,
                "Series order {order}: error {err} should be less than previous {prev_err}"
            );
            prev_err = err;
        }
    }
}

#[test]
fn series_sin_at_nonzero_point() {
    // Taylor series of sin(x) around x = π/4 to order 4
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi = ctx.pi();
    let quarter_pi = &pi / &ctx.int(4);

    let s = x.sin().series(&x, &quarter_pi, 4);
    let expanded = s.expand().eval();

    // At x = π/4: should give sin(π/4) = √2/2 ≈ 0.7071
    let val = expanded.subs(&x, &quarter_pi).eval();
    if let Ok(v) = val.eval_f64() {
        let expected = std::f64::consts::FRAC_PI_4.sin();
        assert!(
            (v - expected).abs() < 1e-4,
            "sin(x) series around π/4, at x=π/4: expected {expected}, got {v}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 12: INTEGRATION EDGE CASES — differentiate-the-result checks
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_ln_x() {
    // ∫ ln(x) dx = x·ln(x) - x + C
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.ln();
    let (s, is_uneval) = ftc_check(&integrand, &x, &[0.5, 1.0, 2.0, 3.0], 1e-6, "∫ln(x)dx");

    if is_uneval {
        panic!(
            "BUG: ∫ln(x)dx returned unevaluated form: {s}\n\
             Expected: x·ln(x) - x or equivalent"
        );
    }
}

#[test]
fn integrate_1_over_x_squared_plus_a() {
    // ∫ 1/(x² + 4) dx = (1/2)·arctan(x/2) + C
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &ctx.int(1) / &(&x.powi(2) + &ctx.int(4));
    let (s, is_uneval) = ftc_check(&integrand, &x, &[0.0, 0.5, 1.0, 2.0], 1e-6, "∫1/(x²+4)dx");

    if is_uneval {
        panic!(
            "BUG: ∫1/(x²+4)dx returned unevaluated form: {s}\n\
             Expected: (1/2)·arctan(x/2) or equivalent"
        );
    }
}

#[test]
fn integrate_exp_2x() {
    // ∫ exp(2x) dx = exp(2x)/2 + C
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = (&x * 2).exp();
    let (s, is_uneval) = ftc_check(&integrand, &x, &[0.0, 0.5, 1.0], 1e-6, "∫exp(2x)dx");

    if is_uneval {
        panic!(
            "BUG: ∫exp(2x)dx returned unevaluated form: {s}\n\
             Expected: exp(2x)/2 or equivalent"
        );
    }
}

#[test]
fn integrate_sin_2x() {
    // ∫ sin(2x) dx = -cos(2x)/2 + C
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = (&x * 2).sin();
    let (s, is_uneval) = ftc_check(&integrand, &x, &[0.3, 0.7, 1.0, 1.5], 1e-6, "∫sin(2x)dx");

    if is_uneval {
        panic!(
            "BUG: ∫sin(2x)dx returned unevaluated form: {s}\n\
             Expected: -cos(2x)/2 or equivalent"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 13: Z-TRANSFORM INVERSE EDGE CASES
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn inverse_z_transform_z_over_z_minus_1_squared() {
    // Z⁻¹{z/(z-1)²} = n
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let z = ctx.symbol("z");

    let expr = &z / &(&z - 1).powi(2);
    let result = expr.inverse_z_transform(&z, &n);

    match result {
        Ok(r) => {
            // At n=5: should be 5
            if let Ok(v) = r.subs(&n, &ctx.int(5)).eval().eval_f64() {
                assert!(
                    (v - 5.0).abs() < 1e-6,
                    "Z⁻¹{{z/(z-1)²}} at n=5 should be 5 (= n), got: {v}"
                );
            }
        }
        Err(e) => {
            panic!("BUG: Z⁻¹{{z/(z-1)²}} should succeed (expected: n): {e}");
        }
    }
}

#[test]
fn inverse_z_transform_z_over_z_minus_1() {
    // Z⁻¹{z/(z-1)} = 1 (unit step / constant 1)
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let z = ctx.symbol("z");

    let expr = &z / &(&z - 1);
    let result = expr
        .inverse_z_transform(&z, &n)
        .expect("Z⁻¹{z/(z-1)} should succeed");

    // At n=0: should be 1, at n=10: should be 1
    for nv in [0, 1, 5, 10] {
        if let Ok(v) = result.subs(&n, &ctx.int(nv)).eval().eval_f64() {
            assert!(
                (v - 1.0).abs() < 1e-6,
                "Z⁻¹{{z/(z-1)}} at n={nv} should be 1, got: {v}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 14: CONVERGENCE — SCALED P-SERIES
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn convergence_scaled_p_series_converges() {
    // Σ 5/k² should converge (constant factor doesn't change convergence)
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = &ctx.int(5) * &k.powi(-2);
    let result = body.is_convergent(&k);
    assert_eq!(
        result,
        Some(true),
        "Σ 5/k² should converge, got: {result:?}"
    );
}

#[test]
fn convergence_scaled_harmonic_diverges() {
    // Σ 3/k should diverge
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = &ctx.int(3) * &k.powi(-1);
    let result = body.is_convergent(&k);
    assert_eq!(result, Some(false), "Σ 3/k should diverge, got: {result:?}");
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 15: LAPLACE FORWARD — verify numerical accuracy of compound forms
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn laplace_t_times_exp() {
    // L{t·exp(2t)} = 1/(s-2)²
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    let expr = &t * &(&t * 2).exp();
    let result = expr.laplace(&t, &s);
    let d = format!("{result}");

    assert!(
        !result.has_unevaluated(),
        "BUG: L{{t·exp(2t)}} should succeed, got unevaluated: {d}"
    );

    // At s=4: 1/(4-2)² = 1/4 = 0.25
    if let Ok(v) = result.subs(&s, &ctx.int(4)).eval().eval_f64() {
        assert!(
            (v - 0.25).abs() < 1e-6,
            "L{{t·exp(2t)}} at s=4 should be 0.25, got: {v}"
        );
    }
}

#[test]
fn laplace_exp_sin() {
    // L{exp(t)·sin(2t)} = 2/((s-1)²+4) via frequency shift
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    let expr = &t.exp() * &(&t * 2).sin();
    let result = expr.laplace(&t, &s);
    let d = format!("{result}");

    assert!(
        !result.has_unevaluated(),
        "BUG: L{{exp(t)·sin(2t)}} should succeed: {d}"
    );

    // At s=3: 2/((3-1)²+4) = 2/(4+4) = 2/8 = 0.25
    if let Ok(v) = result.subs(&s, &ctx.int(3)).eval().eval_f64() {
        assert!(
            (v - 0.25).abs() < 1e-6,
            "L{{exp(t)·sin(2t)}} at s=3 should be 0.25, got: {v}"
        );
    }
}

#[test]
fn laplace_exp_cos() {
    // L{exp(-t)·cos(3t)} = (s+1)/((s+1)²+9) via frequency shift
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    let expr = &(-&t).exp() * &(&t * 3).cos();
    let result = expr.laplace(&t, &s);
    let d = format!("{result}");

    assert!(
        !result.has_unevaluated(),
        "BUG: L{{exp(-t)·cos(3t)}} should succeed: {d}"
    );

    // At s=1: (1+1)/((1+1)²+9) = 2/(4+9) = 2/13 ≈ 0.1538
    if let Ok(v) = result.subs(&s, &ctx.int(1)).eval().eval_f64() {
        let expected = 2.0 / 13.0;
        assert!(
            (v - expected).abs() < 1e-6,
            "L{{exp(-t)·cos(3t)}} at s=1 should be {expected}, got: {v}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 16: INVERSE LAPLACE — completed-square / shifted quadratic
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn inverse_laplace_shifted_sin() {
    // L⁻¹{1/((s-1)²+4)} = (1/2)·exp(t)·sin(2t)
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    // (s-1)² + 4 = s² - 2s + 5
    let denom = &(&s.powi(2) - &(&s * 2)) + &ctx.int(5);
    let expr = &ctx.int(1) / &denom;
    let result = expr.inverse_laplace(&s, &t);
    let d = format!("{result}");

    assert!(
        !result.has_unevaluated(),
        "BUG: L⁻¹{{1/(s²-2s+5)}} should not be unevaluated: {d}\n\
         Expected: (1/2)·exp(t)·sin(2t)"
    );

    if let Ok(v) = result.subs(&t, &ctx.int(0)).eval().eval_f64() {
        // At t=0: (1/2)·exp(0)·sin(0) = 0
        assert!(
            v.abs() < 1e-6,
            "L⁻¹{{1/(s²-2s+5)}} at t=0 should be 0, got: {v}"
        );
    }
}

#[test]
fn inverse_laplace_shifted_cos() {
    // L⁻¹{(s-1)/((s-1)²+4)} = exp(t)·cos(2t)
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    let denom = &(&s.powi(2) - &(&s * 2)) + &ctx.int(5);
    let numer = &s - 1;
    let expr = &numer / &denom;
    let result = expr.inverse_laplace(&s, &t);
    let d = format!("{result}");

    assert!(
        !result.has_unevaluated(),
        "BUG: L⁻¹{{(s-1)/(s²-2s+5)}} should not be unevaluated: {d}\n\
         Expected: exp(t)·cos(2t)"
    );

    if let Ok(v) = result.subs(&t, &ctx.int(0)).eval().eval_f64() {
        // At t=0: exp(0)·cos(0) = 1
        assert!(
            (v - 1.0).abs() < 1e-6,
            "L⁻¹{{(s-1)/(s²-2s+5)}} at t=0 should be 1, got: {v}"
        );
    }
}

#[test]
fn inverse_laplace_n_over_s_power_n_roundtrip() {
    // L{t⁴} = 24/s⁵, so L⁻¹{24/s⁵} = t⁴
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    let expr = &ctx.int(24) / &s.powi(5);
    let result = expr.inverse_laplace(&s, &t);
    let d = format!("{result}");

    assert!(
        !result.has_unevaluated(),
        "BUG: L⁻¹{{24/s⁵}} should not be unevaluated: {d}\n\
         Expected: t⁴"
    );

    if let Ok(v) = result.subs(&t, &ctx.int(2)).eval().eval_f64() {
        // t⁴ at t=2 → 16
        assert!(
            (v - 16.0).abs() < 1e-6,
            "L⁻¹{{24/s⁵}} at t=2 should be 16 (=2⁴), got: {v}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 17: ODE NONHOMOGENEOUS / HARDER CASES
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_nonhomogeneous_y_pp_plus_y_eq_x() {
    // y'' + y = x  →  particular: y_p = x, general: C₁cos(x)+C₂sin(x)+x
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &(&d2y + &y) - &x; // y'' + y - x = 0

    let sol = ode.solve_ode(&y, &x);
    let s = format!("{sol}");

    if !sol.has_unevaluated() {
        // Verify the particular solution y_p = x satisfies y'' + y = x
        // y_p'' = 0, so y_p'' + y_p = 0 + x = x ✓
        let particular = x.clone();
        assert!(
            ode.check_ode_solution(&particular, &y, &x),
            "y = x should satisfy y'' + y - x = 0, solution was: {s}"
        );
    }
    // It's acceptable if the solver can't handle nonhomogeneous; just note it.
}

#[test]
fn ode_y_prime_eq_minus_2xy() {
    // y' + 2xy = 0  →  y = C·exp(-x²)
    // This is a variable-coefficient first-order linear ODE.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let dy = y.formal_diff(&x);
    let ode = &dy + &(&(&x * 2) * &y); // y' + 2x·y = 0

    let sol = ode.solve_ode(&y, &x);
    let s = format!("{sol}");

    if !sol.has_unevaluated() {
        // Verify: y = exp(-x²) should satisfy y' + 2x·y = 0
        let particular = (-&x.powi(2)).exp();
        assert!(
            ode.check_ode_solution(&particular, &y, &x),
            "y = exp(-x²) should satisfy y' + 2x·y = 0, solution was: {s}"
        );
    }
}

#[test]
fn ode_y_prime_eq_y_squared_is_hard() {
    // y' = y²  (Bernoulli / nonlinear)
    // Solution: y = -1/(x + C)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let dy = y.formal_diff(&x);
    let ode = &dy - &y.powi(2);

    let sol = ode.solve_ode(&y, &x);
    let s = format!("{sol}");

    if !sol.has_unevaluated() {
        // Verify: y = -1/x should satisfy y' - y² = 0
        // y = -1/x → y' = 1/x², y² = 1/x², so y' - y² = 0 ✓
        let particular = -&(&ctx.int(1) / &x);
        assert!(
            ode.check_ode_solution(&particular, &y, &x),
            "y = -1/x should satisfy y' = y², solution was: {s}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 18: SERIES COEFFICIENT VERIFICATION
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn series_1_over_1_minus_x_coefficients_exact() {
    // 1/(1-x) = 1 + x + x² + x³ + ...
    // At x = 1/10: series(5 terms) = 1 + .1 + .01 + .001 + .0001 = 1.1111
    // Exact: 1/(1-0.1) = 10/9 = 1.11111...
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);

    let expr = &ctx.int(1) / &(&ctx.int(1) - &x);
    let s = expr.series(&x, &zero, 5);
    let expanded = s.expand().eval();

    let val = expanded.subs(&x, &ctx.rational(1, 10)).eval();
    if let Ok(v) = val.eval_f64() {
        let expected = 1.0 + 0.1 + 0.01 + 0.001 + 0.0001;
        assert!(
            (v - expected).abs() < 1e-10,
            "1/(1-x) 5-term series at x=0.1 should be exactly {expected}, got: {v}"
        );
    }
}

#[test]
fn series_exp_x_coefficient_check() {
    // exp(x) = 1 + x + x²/2 + x³/6 + x⁴/24
    // Verify at x = 1: sum = 1 + 1 + 0.5 + 0.1667 + 0.04167 = 2.70833...
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);

    let s = x.exp().series(&x, &zero, 5);
    let expanded = s.expand().eval();

    let val = expanded.subs(&x, &ctx.int(1)).eval();
    if let Ok(v) = val.eval_f64() {
        let expected = 1.0 + 1.0 + 0.5 + 1.0 / 6.0 + 1.0 / 24.0;
        assert!(
            (v - expected).abs() < 1e-10,
            "exp(x) 5-term series at x=1: expected {expected}, got {v}"
        );
    }
}

#[test]
fn series_sin_x_odd_terms_only() {
    // sin(x) = x - x³/6 + x⁵/120 - ...
    // The series should have ONLY odd powers of x.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);

    let s = x.sin().series(&x, &zero, 6);
    let expanded = s.expand().eval();
    let result = format!("{expanded}");

    // Odd powers: x, x^3, x^5 should appear
    // Even powers: x^2, x^4 should NOT appear
    assert!(
        !result.contains("x^2") || result.contains("x^2") && result.contains("0*x^2"),
        "sin(x) series should have no x² term: {result}"
    );
}

#[test]
fn series_cos_x_even_terms_only() {
    // cos(x) = 1 - x²/2 + x⁴/24 - ...
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);

    let s = x.cos().series(&x, &zero, 6);
    let expanded = s.expand().eval();
    let result = format!("{expanded}");

    // Should contain x^2, x^4 but not isolated x^1, x^3, x^5
    assert!(
        result.contains("x^2"),
        "cos(x) series should contain x^2 term: {result}"
    );
    assert!(
        result.contains("x^4"),
        "cos(x) series should contain x^4 term: {result}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 19: DEFINITE INTEGRAL — trig over symmetric intervals
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn definite_cos_0_to_2pi_equals_0() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // ∫₀^{2π} cos(x) dx = sin(2π) - sin(0) = 0
    let two_pi = &ctx.int(2) * &ctx.pi();
    let result = x.cos().integrate_definite(&x, &ctx.int(0), &two_pi);
    let evaled = result.eval();
    let s = format!("{evaled}");
    assert_eq!(s, "0", "∫₀^{{2π}} cos(x)dx should be 0, got: {s}");
}

#[test]
fn definite_x_exp_neg_x_0_to_inf_is_hard() {
    // ∫₀^∞ x·exp(-x) dx = 1  (Gamma(2) = 1!)
    // This requires evaluating at infinity — just test the antiderivative is correct.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x * &(-&x).exp();
    let antideriv = integrand.integrate(&x);
    let s = format!("{antideriv}");

    if !antideriv.has_unevaluated() {
        // Verify FTC at finite points
        let deriv = antideriv.diff(&x);
        let pt = ctx.rational(1, 1);
        let orig = integrand.subs(&x, &pt).eval().eval_f64();
        let dval = deriv.subs(&x, &pt).eval().eval_f64();
        if let (Ok(o), Ok(d)) = (orig, dval) {
            assert!(
                (o - d).abs() < 1e-6,
                "FTC for x·exp(-x) at x=1: integrand={o}, deriv={d}, antideriv='{s}'"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 20: INTEGRATION — rational functions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_1_over_x_plus_1() {
    // ∫ 1/(x+1) dx = ln|x+1| + C
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &ctx.int(1) / &(&x + 1);
    let (s, is_uneval) = ftc_check(&integrand, &x, &[0.5, 1.0, 2.0, 3.0], 1e-6, "∫1/(x+1)dx");

    if is_uneval {
        panic!(
            "BUG: ∫1/(x+1)dx returned unevaluated form: {s}\n\
             Expected: ln|x+1| or equivalent"
        );
    }
}

#[test]
fn integrate_x_over_x_squared_plus_1() {
    // ∫ x/(x²+1) dx = (1/2)·ln(x²+1) + C
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x / &(&x.powi(2) + &ctx.int(1));
    let (s, is_uneval) = ftc_check(&integrand, &x, &[0.5, 1.0, 2.0], 1e-6, "∫x/(x²+1)dx");

    if is_uneval {
        panic!(
            "BUG: ∫x/(x²+1)dx returned unevaluated form: {s}\n\
             Expected: (1/2)·ln(x²+1) or equivalent"
        );
    }
}

#[test]
fn integrate_partial_fractions_1_over_x2_minus_1() {
    // ∫ 1/(x²-1) dx = (1/2)·ln|(x-1)/(x+1)| + C
    // = (1/2)·ln|x-1| - (1/2)·ln|x+1| + C
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &ctx.int(1) / &(&x.powi(2) - &ctx.int(1));
    let (s, is_uneval) = ftc_check(
        &integrand,
        &x,
        &[2.0, 3.0, 5.0], // avoid x = ±1
        1e-6,
        "∫1/(x²-1)dx",
    );

    if is_uneval {
        panic!(
            "BUG: ∫1/(x²-1)dx returned unevaluated form: {s}\n\
             Expected: partial fraction result"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 21: CONSISTENCY — definite integral vs antiderivative evaluation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn definite_integral_matches_antideriv_evaluation() {
    // Verify that definite_integral(f, a, b) = F(b) - F(a)
    // for F = integrate(f)
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let f = &x.powi(3) + &x.sin();
    let a = ctx.int(1);
    let b = ctx.int(2);

    let direct = f.integrate_definite(&x, &a, &b);
    let antideriv = f.integrate(&x);
    let manual = &antideriv.subs(&x, &b) - &antideriv.subs(&x, &a);

    let dv = direct.eval().eval_f64();
    let mv = manual.eval().eval_f64();

    if let (Ok(d), Ok(m)) = (dv, mv) {
        assert!(
            (d - m).abs() < 1e-8,
            "definite_integral vs manual F(b)-F(a) mismatch: direct={d}, manual={m}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 22: Z-TRANSFORM — verify partial-sum against closed form
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn z_transform_2_to_n_partial_sum_check() {
    // Z{2^n} = z/(z-2). Verify via partial sum Σ_{k=0}^{N} 2^k · z^{-k}
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let z = ctx.symbol("z");

    let expr = ctx.int(2).pow(&n);
    let result = expr.z_transform(&n, &z).expect("Z{2^n} should succeed");

    // At z=5: z/(z-2) = 5/3
    // Partial sum: Σ_{k=0}^{40} 2^k · 5^{-k} = Σ (2/5)^k ≈ 1/(1-2/5) = 5/3
    let z_val = 5.0_f64;
    let mut partial_sum = 0.0;
    for k in 0..60 {
        partial_sum += 2.0_f64.powi(k) * z_val.powi(-k);
    }

    if let Ok(v) = result.subs(&z, &ctx.int(5)).eval().eval_f64() {
        assert!(
            (v - partial_sum).abs() < 1e-6,
            "Z{{2^n}} at z=5: closed form={v}, partial sum={partial_sum}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 23: ODE solution verification — numerical residual check
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_y_prime_3y_eq_0_numerical_check() {
    // y' + 3y = 0  →  y = C·exp(-3x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let dy = y.formal_diff(&x);
    let ode = &dy + &(&y * 3);

    let sol = ode.solve_ode(&y, &x);
    if sol.has_unevaluated() {
        panic!("y' + 3y = 0 should be solvable");
    }

    // Substitute C1 = 1 and check residual at several points
    let c1 = ctx.symbol("C1");
    let particular = sol.subs(&c1, &ctx.int(1));

    // Compute y'(x) and y(x), check y' + 3y ≈ 0
    let y_val = particular.clone();
    let yp_val = particular.diff(&x);

    for &xv in &[0.1_f64, 0.5, 1.0, 2.0] {
        let numer = (xv * 10000.0).round() as i64;
        let pt = ctx.rational(numer, 10000);
        let yv = y_val.subs(&x, &pt).eval().eval_f64();
        let ypv = yp_val.subs(&x, &pt).eval().eval_f64();
        if let (Ok(y_f), Ok(yp_f)) = (yv, ypv) {
            let residual = yp_f + 3.0 * y_f;
            assert!(
                residual.abs() < 1e-6,
                "ODE residual y'+3y at x={xv}: y'={yp_f}, y={y_f}, residual={residual}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 24: INTEGRATION — verify linearity
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_linearity_sum() {
    // ∫(sin(x) + cos(x))dx should equal ∫sin(x)dx + ∫cos(x)dx
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let combined = (&x.sin() + &x.cos()).integrate(&x);
    let separate_sin = x.sin().integrate(&x);
    let separate_cos = x.cos().integrate(&x);
    let separate_sum = &separate_sin + &separate_cos;

    // Check numerically at x = 1
    let cv = combined.subs(&x, &ctx.int(1)).eval().eval_f64();
    let sv = separate_sum.subs(&x, &ctx.int(1)).eval().eval_f64();

    if let (Ok(c), Ok(s)) = (cv, sv) {
        assert!(
            (c - s).abs() < 1e-8,
            "∫(sin+cos)dx vs ∫sin dx + ∫cos dx: combined={c}, separate={s}"
        );
    }
}

#[test]
fn integrate_linearity_scalar() {
    // ∫(5·x²)dx should equal 5·∫x²dx
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let combined = (&x.powi(2) * 5).integrate(&x);
    let scaled = &x.powi(2).integrate(&x) * 5;

    let cv = combined.subs(&x, &ctx.int(2)).eval().eval_f64();
    let sv = scaled.subs(&x, &ctx.int(2)).eval().eval_f64();

    if let (Ok(c), Ok(s)) = (cv, sv) {
        assert!(
            (c - s).abs() < 1e-8,
            "∫5x²dx vs 5·∫x²dx: combined={c}, scaled={s}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 25: CONVERGENCE — edge cases at boundary
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn convergence_p_series_p_exactly_1_diverges() {
    // The boundary case: p = 1 exactly (harmonic series) must diverge.
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = k.pow(&ctx.int(-1)); // k^(-1)
    assert_eq!(
        body.is_convergent(&k),
        Some(false),
        "Σ k^(-1) (harmonic) should diverge"
    );
}

#[test]
fn convergence_p_series_p_just_above_1() {
    // p = 2 is the simplest rational above 1 we can test
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = k.pow(&ctx.int(-2));
    assert_eq!(
        body.is_convergent(&k),
        Some(true),
        "Σ k^(-2) should converge"
    );
}

#[test]
fn convergence_geometric_neg_one_diverges() {
    // Σ (-1)^k — alternating ±1, diverges (terms don't approach 0, actually inconclusive for ratio)
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let body = ctx.int(-1).pow(&k);
    let result = body.is_convergent(&k);
    // |r| = 1, so geometric test says diverge
    assert_eq!(
        result,
        Some(false),
        "Σ (-1)^k should diverge (|r|=1), got: {result:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 26: SERIES — interaction with differentiation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn series_of_diff_equals_diff_of_series() {
    // d/dx[series(f, x, 0, n)] should approximate series(f', x, 0, n-1)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);

    let f = x.exp();
    let order = 6u32;

    let series_then_diff = f.series(&x, &zero, order).expand().eval().diff(&x);
    let diff_then_series = f.diff(&x).series(&x, &zero, order - 1).expand().eval();

    // Both should agree at x = 0.5
    let v1 = series_then_diff
        .subs(&x, &ctx.rational(1, 2))
        .eval()
        .eval_f64();
    let v2 = diff_then_series
        .subs(&x, &ctx.rational(1, 2))
        .eval()
        .eval_f64();

    if let (Ok(a), Ok(b)) = (v1, v2) {
        assert!(
            (a - b).abs() < 0.01,
            "d/dx(series(exp)) vs series(d/dx(exp)) at x=0.5: {a} vs {b}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 27: LAPLACE — verify forward/inverse of sinh and cosh
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn laplace_roundtrip_sinh() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    let original = (&t * 2).sinh();
    let forward = original.laplace(&t, &s);
    assert!(
        !forward.has_unevaluated(),
        "L{{sinh(2t)}} should succeed: {forward}"
    );

    let recovered = forward.inverse_laplace(&s, &t);
    let d = format!("{recovered}");

    if !recovered.has_unevaluated() {
        // Verify at t=1
        let ov = original.subs(&t, &ctx.int(1)).eval().eval_f64();
        let rv = recovered.subs(&t, &ctx.int(1)).eval().eval_f64();
        if let (Ok(o), Ok(r)) = (ov, rv) {
            assert!(
                (o - r).abs() < 1e-4,
                "Laplace roundtrip sinh(2t): original={o}, recovered={r}"
            );
        }
    } else {
        panic!(
            "BUG: Laplace roundtrip of sinh(2t) gave unevaluated: {d}\n\
             L{{sinh(2t)}} = {forward}"
        );
    }
}

#[test]
fn laplace_roundtrip_cosh() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    let original = (&t * 3).cosh();
    let forward = original.laplace(&t, &s);
    assert!(
        !forward.has_unevaluated(),
        "L{{cosh(3t)}} should succeed: {forward}"
    );

    let recovered = forward.inverse_laplace(&s, &t);
    let d = format!("{recovered}");

    if !recovered.has_unevaluated() {
        let ov = original.subs(&t, &ctx.int(1)).eval().eval_f64();
        let rv = recovered.subs(&t, &ctx.int(1)).eval().eval_f64();
        if let (Ok(o), Ok(r)) = (ov, rv) {
            assert!(
                (o - r).abs() < 1e-4,
                "Laplace roundtrip cosh(3t): original={o}, recovered={r}"
            );
        }
    } else {
        panic!(
            "BUG: Laplace roundtrip of cosh(3t) gave unevaluated: {d}\n\
             L{{cosh(3t)}} = {forward}"
        );
    }
}

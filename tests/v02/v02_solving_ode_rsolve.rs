//! 0.2 solving campaign — ODE additions (initial-value problems, nth-order
//! linear constant-coefficient equations, Clairaut, Riccati, classification,
//! system IVPs) and recurrence relations (`rsolve`).  Every solution is
//! verified by substitution.

use symplex::ode::OdeType;
use symplex::prelude::*;
use symplex::rsolve::{rsolve_first_order, rsolve_linear};

fn setup() -> (Context, Ex, Ex, Ex, Ex, Ex, Ex) {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let d3y = d2y.formal_diff(&x);
    let d4y = d3y.formal_diff(&x);
    (ctx, x, y, dy, d2y, d3y, d4y)
}

/// `y^(order)(x) = value`.
fn ic(order: usize, x: Ex, value: Ex) -> InitialCondition {
    InitialCondition { order, x, value }
}

/// Numerically verify `sol` against the ODE at several points with the
/// constants set to 1 (exact `check_ode_solution` is also tried first).
fn assert_ode_solution(ode: &Ex, sol: &Ex, y: &Ex, x: &Ex, label: &str) {
    assert!(
        !sol.has_unevaluated(),
        "{label}: unevaluated solution {sol}"
    );
    if ode.check_ode_solution(sol, y, x) {
        return;
    }
    // Numeric fallback: substitute derivatives and evaluate at sample points.
    let ctx = x.context();
    let mut s = sol.clone();
    for k in 1..=6 {
        s = s.subs_i64(&ctx.symbol(&format!("C{k}")), 1);
    }
    let mut residual = ode.clone();
    let mut derivs = vec![s.clone()];
    for _ in 0..4 {
        let d = derivs.last().unwrap().diff(x);
        derivs.push(d);
    }
    let mut node = y.clone();
    let mut nodes = vec![node.clone()];
    for _ in 0..4 {
        node = node.formal_diff(x);
        nodes.push(node.clone());
    }
    for k in (0..=4).rev() {
        residual = residual.subs(&nodes[k], &derivs[k]);
    }
    for xv in [0.3, 0.7, 1.1, 1.9] {
        let v = residual
            .subs(x, &ctx.rational((xv * 10.0) as i64, 10))
            .eval_f64()
            .unwrap_or_else(|e| panic!("{label}: cannot evaluate residual: {e} ({residual})"));
        assert!(
            v.abs() < 1e-8,
            "{label}: residual {v} at x = {xv} for {sol}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Initial-value problems
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ivp_first_order_exponential() {
    let (ctx, x, y, dy, ..) = setup();
    // y' = y, y(0) = 2 → 2 e^x
    let ode = &dy - &y;
    let sol = ode
        .solve_ode_ivp(&y, &x, &[ic(0, ctx.int(0), ctx.int(2))])
        .unwrap();
    assert_eq!(format!("{sol}"), "2*exp(x)");
    assert_ode_solution(&ode, &sol, &y, &x, "y' = y");
}

#[test]
fn ivp_harmonic_oscillator() {
    let (ctx, x, y, _dy, d2y, ..) = setup();
    // y'' + y = 0, y(0) = 0, y'(0) = 1 → sin x
    let ode = &d2y + &y;
    let sol = ode
        .solve_ode_ivp(
            &y,
            &x,
            &[ic(0, ctx.int(0), ctx.int(0)), ic(1, ctx.int(0), ctx.int(1))],
        )
        .unwrap();
    assert_eq!(format!("{}", sol.simplify()), "sin(x)");
    assert_ode_solution(&ode, &sol, &y, &x, "y'' + y = 0");
    // Only one condition: one constant remains.
    let partial = ode
        .solve_ode_ivp(&y, &x, &[ic(0, ctx.int(0), ctx.int(0))])
        .unwrap();
    assert!(
        !partial.contains(&ctx.symbol("C1")) && partial.contains(&ctx.symbol("C2")),
        "{partial}"
    );
}

#[test]
fn ivp_gaussian() {
    let (ctx, x, y, dy, ..) = setup();
    // y' = -2xy, y(0) = 1 → e^{-x²}
    let ode = &dy + &(&x * &y * 2);
    let sol = ode
        .solve_ode_ivp(&y, &x, &[ic(0, ctx.int(0), ctx.int(1))])
        .unwrap();
    assert_eq!(format!("{sol}"), "exp(-x^2)");
    assert_ode_solution(&ode, &sol, &y, &x, "y' = -2xy");
}

#[test]
fn ivp_third_order_and_nonlinear_constant() {
    let (ctx, x, y, dy, _d2y, d3y, _) = setup();
    // y''' = y with y(0) = y'(0) = y''(0) = 1 → e^x
    let ode = &d3y - &y;
    let sol = ode
        .solve_ode_ivp(
            &y,
            &x,
            &[
                ic(0, ctx.int(0), ctx.int(1)),
                ic(1, ctx.int(0), ctx.int(1)),
                ic(2, ctx.int(0), ctx.int(1)),
            ],
        )
        .unwrap();
    assert_eq!(format!("{sol}"), "exp(x)");
    // y' = y², y(0) = 1 → 1/(1 - x): constant enters nonlinearly.
    let ode = &dy - &y.powi(2);
    let sol = ode
        .solve_ode_ivp(&y, &x, &[ic(0, ctx.int(0), ctx.int(1))])
        .unwrap();
    assert_ode_solution(&ode, &sol, &y, &x, "y' = y^2");
    let v = sol.subs(&x, &ctx.rational(1, 2)).eval_f64().unwrap();
    assert!((v - 2.0).abs() < 1e-12, "{sol}");
    // Non-zero initial point: y' = y, y(1) = e → e^x
    let ode = &dy - &y;
    let e = ctx.int(1).exp();
    let sol = ode.solve_ode_ivp(&y, &x, &[ic(0, ctx.int(1), e)]).unwrap();
    // `E*exp(x - 1)` is not folded to `exp(x)` by the simplifier; verify numerically.
    for xv in [0, 1, 2, 3] {
        let v = sol.subs_i64(&x, xv).eval_f64().unwrap();
        assert!((v - (xv as f64).exp()).abs() < 1e-9, "{sol}");
    }
    assert_ode_solution(&ode, &sol, &y, &x, "y' = y, y(1) = e");
}

#[test]
fn ivp_contradictory_conditions() {
    let (ctx, x, y, dy, ..) = setup();
    let ode = &dy - &y;
    let err = ode
        .solve_ode_ivp(
            &y,
            &x,
            &[ic(0, ctx.int(0), ctx.int(1)), ic(0, ctx.int(0), ctx.int(2))],
        )
        .unwrap_err();
    assert!(matches!(err, SymplexError::NoSolution { .. }), "{err}");
}

// ═══════════════════════════════════════════════════════════════════════════
// nth-order linear constant-coefficient
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn nth_order_homogeneous_orders_three_and_four() {
    let (ctx, x, y, dy, d2y, d3y, d4y) = setup();
    let cases: Vec<(&str, Ex, usize)> = vec![
        ("y''' - y = 0", &d3y - &y, 3),
        ("y'''' - y = 0", &d4y - &y, 4),
        ("(D-1)^3 y = 0", &d3y - &d2y * 3 + &dy * 3 - &y, 3),
        ("(D^2+1)^2 y = 0", &d4y + &d2y * 2 + &y, 4),
        ("y'''' = 0", d4y.clone(), 4),
        (
            "y''' - 6y'' + 11y' - 6y = 0",
            &d3y - &d2y * 6 + &dy * 11 - &y * 6,
            3,
        ),
    ];
    for (label, ode, n) in cases {
        assert_eq!(
            ode.classify_ode(&y, &x),
            OdeType::NthOrderLinearConstCoeff,
            "{label}"
        );
        let sol = ode.solve_ode(&y, &x);
        for k in 1..=n {
            assert!(
                sol.contains(&ctx.symbol(&format!("C{k}"))),
                "{label}: missing C{k} in {sol}"
            );
        }
        assert!(
            !sol.contains(&ctx.symbol(&format!("C{}", n + 1))),
            "{label}: too many constants"
        );
        assert_ode_solution(&ode, &sol, &y, &x, label);
    }
    // Repeated root produces x·e^x and x²·e^x modes.
    let s = format!("{}", (&d3y - &d2y * 3 + &dy * 3 - &y).solve_ode(&y, &x));
    assert!(s.contains("x^2*exp(x)") && s.contains("x*exp(x)"), "{s}");
    // Complex pair for y''' = y: e^{-x/2} cos(√3 x / 2)
    let s = format!("{}", (&d3y - &y).solve_ode(&y, &x));
    assert!(
        s.contains("cos") && s.contains("sin") && s.contains("sqrt(3)"),
        "{s}"
    );
}

#[test]
fn nth_order_nonhomogeneous_undetermined_coefficients() {
    let (_ctx, x, y, dy, d2y, d3y, d4y) = setup();
    let cases: Vec<(&str, Ex)> = vec![
        ("y''' - y' = x", &d3y - &dy - &x),
        ("y''' - y'' = 1 (resonance)", &d3y - &d2y - 1),
        ("y'''' - y = e^{2x}", &d4y - &y - &(&x * 2).exp()),
        ("y'''' - y = e^{x} (resonance)", &d4y - &y - &x.exp()),
        ("y''' + y' = sin(2x)", &d3y + &dy - &(&x * 2).sin()),
        ("y''' + y' = cos x (resonance)", &d3y + &dy - &x.cos()),
        ("y'' + y = x e^x", &d2y + &y - &(&x * &x.exp())),
        ("y'' - y = e^x (resonance)", &d2y - &y - &x.exp()),
        ("y'' + y = sin x (resonance)", &d2y + &y - &x.sin()),
        (
            "y'' - 2y' + y = x^2 e^x (double resonance)",
            &d2y - &dy * 2 + &y - &(&x.powi(2) * &x.exp()),
        ),
        (
            "y'' + 4y = x cos 2x (resonance)",
            &d2y + &y * 4 - &(&x * &(&x * 2).cos()),
        ),
        (
            "y''' - y = x^2 + e^{-x}",
            &d3y - &y - &x.powi(2) - &(-&x).exp(),
        ),
    ];
    for (label, ode) in cases {
        let sol = ode.solve_ode(&y, &x);
        assert_ode_solution(&ode, &sol, &y, &x, label);
    }
}

#[test]
fn second_order_classification_distinguishes_variation_of_parameters() {
    let (_ctx, x, y, _dy, d2y, ..) = setup();
    assert_eq!(
        (&d2y + &y - &(&x * &x.exp())).classify_ode(&y, &x),
        OdeType::SecondOrderLinearCCNonHomogeneous
    );
    // Forcing 1/x is not poly×exp×trig; VoP integrals exist for y'' - y' = 1/x? Use a
    // forcing VoP can integrate: y'' + y = 1/cos... hard. Just check it is not
    // misreported as undetermined coefficients.
    let c = (&d2y + &y - &x.tan()).classify_ode(&y, &x);
    assert_ne!(c, OdeType::SecondOrderLinearCCNonHomogeneous, "{c:?}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Clairaut, Riccati, integrating factor
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn clairaut_general_solution() {
    let (ctx, x, y, dy, ..) = setup();
    // y = x y' + (y')²  →  y = C x + C²
    let ode = &y - &(&x * &dy) - &dy.powi(2);
    assert_eq!(ode.classify_ode(&y, &x), OdeType::Clairaut);
    let sol = ode.solve_ode(&y, &x);
    let c1 = ctx.symbol("C1");
    assert!(sol.contains(&c1));
    let expected = &(&c1 * &x) + &c1.powi(2);
    assert!((&sol - &expected).simplify().is_zero_structural(), "{sol}");
    assert!(ode.check_ode_solution(&sol, &y, &x));
    // y = x y' - e^{y'}  →  y = C x - e^C  (scaled form 2y - 2xy' + 2e^{y'})
    let ode = &y * 2 - &(&x * &dy) * 2 + &dy.exp() * 2;
    assert_eq!(ode.classify_ode(&y, &x), OdeType::Clairaut);
    let sol = ode.solve_ode(&y, &x);
    assert!(ode.check_ode_solution(&sol, &y, &x), "{sol}");
}

#[test]
fn riccati_with_particular_solution() {
    let (ctx, x, y, dy, ..) = setup();
    // y' = y² - 2/x²  has particular solution y_p = 1/x ... check: y_p' = -1/x², y_p² - 2/x² = -1/x² ✓
    let ode = &dy - &y.powi(2) + &(&ctx.int(2) / &x.powi(2));
    assert_eq!(ode.classify_ode(&y, &x), OdeType::Riccati);
    let particular = &ctx.int(1) / &x;
    let sol = ode.solve_riccati(&y, &x, &particular).unwrap();
    assert!(sol.contains(&ctx.symbol("C1")), "{sol}");
    assert_ode_solution(&ode, &sol, &y, &x, "Riccati y' = y^2 - 2/x^2");
    // Wrong particular solution is rejected.
    assert!(ode.solve_riccati(&y, &x, &x).is_err());
    // Not a Riccati equation.
    assert!((&dy - &y).solve_riccati(&y, &x, &x.exp()).is_err());
}

#[test]
fn integrating_factor_and_variable_coefficient_linear() {
    let (ctx, x, y, dy, ..) = setup();
    // 2y + x y' = 0: M_y - N_x = 1, μ = x → x² y = C
    let ode = &(&y * 2) + &(&x * &dy);
    assert_eq!(ode.classify_ode(&y, &x), OdeType::IntegratingFactor);
    let sol = ode.solve_ode(&y, &x);
    assert!(ode.check_ode_solution(&sol, &y, &x), "{sol}");
    // y' + y/x = x  → (x³/3 + C1)/x  (integrating factor exp(ln x) = x)
    let ode = &dy + &(&y / &x) - &x;
    let sol = ode.solve_ode(&y, &x);
    assert!(!sol.has_unevaluated(), "{sol}");
    assert!(sol.contains(&ctx.symbol("C1")));
    assert!(ode.check_ode_solution(&sol, &y, &x), "{sol}");
}

#[test]
fn homogeneous_coefficient_is_not_misdetected() {
    let (_ctx, x, y, dy, ..) = setup();
    // y' = y² + x² is a Riccati equation, NOT degree-0 homogeneous.
    let ode = &dy - &y.powi(2) - &x.powi(2);
    assert_eq!(ode.classify_ode(&y, &x), OdeType::Riccati);
    let sol = ode.solve_ode(&y, &x);
    // Either unsolved or a verified solution — never a wrong closed form.
    if !sol.has_unevaluated() {
        assert!(ode.check_ode_solution(&sol, &y, &x), "wrong solution {sol}");
    }
    // A genuine degree-0 homogeneous equation still classifies correctly.
    let ode = &dy - &(&(&x.powi(2) + &y.powi(2)) / &(&x * &y));
    assert_eq!(ode.classify_ode(&y, &x), OdeType::HomogeneousCoefficient);
}

#[test]
fn classification_covers_every_solved_case() {
    let (ctx, x, y, dy, d2y, d3y, _) = setup();
    let odes: Vec<Ex> = vec![
        &dy - &x,                                        // simple separable
        &dy - &(&x * &y),                                // full separable
        &dy + &y * 2,                                    // first-order linear CC
        &dy + &(&x * &y * 2),                            // first-order linear VC
        &d2y + &y,                                       // 2nd order CC homogeneous
        &d2y + &y - &x,                                  // 2nd order CC nonhomogeneous
        &d3y - &y,                                       // nth order
        &y - &(&x * &dy) - &dy.powi(2),                  // Clairaut
        &(&y * 2) + &(&x * &dy),                         // integrating factor
        &dy + &(&y / &x) - &y.powi(2),                   // Bernoulli
        &(&x.powi(2) * &d2y) + &(&x * &dy) * 2 - &y * 2, // Euler-Cauchy
    ];
    for ode in odes {
        let sol = ode.solve_ode(&y, &x);
        if !sol.has_unevaluated() {
            assert_ne!(
                ode.classify_ode(&y, &x),
                OdeType::Unknown,
                "solved but classified Unknown: {ode} → {sol}"
            );
        }
    }
    let _ = ctx;
}

// ═══════════════════════════════════════════════════════════════════════════
// Systems with initial values
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_system_ivp() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    // Rotation: x' = y, y' = -x, x(0) = 1, y(0) = 0
    let a = symplex::matrix![ctx, [0, 1], [-1, 0]];
    let sol = symplex::ode::solve_ode_system_ivp(&a, &t, &[ctx.int(1), ctx.int(0)]).unwrap();
    assert_eq!(format!("{}", sol[0].simplify()), "cos(t)");
    assert_eq!(format!("{}", sol[1].simplify()), "-sin(t)");
    // Verify x' = A x componentwise.
    for i in 0..2 {
        let mut rhs = ctx.int(0);
        for (j, sj) in sol.iter().enumerate() {
            rhs = &rhs + &(a.get(i, j) * sj);
        }
        let res = (&sol[i].diff(&t) - &rhs).simplify();
        assert!(res.is_zero_structural(), "row {i}: {res}");
    }
    // Upper-triangular with distinct eigenvalues: x' = x + y, y' = 2y, x(0)=1, y(0)=1 → x = e^{2t}, y = e^{2t}
    let a = symplex::matrix![ctx, [1, 1], [0, 2]];
    let sol = symplex::ode::solve_ode_system_ivp(&a, &t, &[ctx.int(1), ctx.int(1)]).unwrap();
    assert_eq!(format!("{}", sol[0].simplify()), "exp(2*t)");
    assert_eq!(format!("{}", sol[1].simplify()), "exp(2*t)");
    // Errors
    assert!(symplex::ode::solve_ode_system_ivp(&a, &t, &[ctx.int(1)]).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// Recurrences
// ═══════════════════════════════════════════════════════════════════════════

fn seq_at(f: &Ex, n: &Ex, k: i64) -> f64 {
    f.subs_i64(n, k).eval_f64().unwrap()
}

/// Verify a closed form against the recurrence Σ c_k a(n+k) = f(n) for n = 0..8.
fn assert_recurrence(f: &Ex, coeffs: &[Ex], forcing: Option<&Ex>, n: &Ex, label: &str) {
    for m in 0..8 {
        let mut lhs = 0.0;
        for (k, c) in coeffs.iter().enumerate() {
            lhs += c.eval_f64().unwrap() * seq_at(f, n, m + k as i64);
        }
        let rhs = forcing.map_or(0.0, |g| seq_at(g, n, m));
        assert!(
            (lhs - rhs).abs() < 1e-7,
            "{label}: n = {m}: {lhs} vs {rhs} ({f})"
        );
    }
}

#[test]
fn rsolve_fibonacci_binet() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let coeffs = [ctx.int(-1), ctx.int(-1), ctx.int(1)];
    let f = rsolve_linear(&coeffs, None, &n, &[ctx.int(0), ctx.int(1)]).unwrap();
    assert!(format!("{f}").contains("sqrt(5)"), "{f}");
    let fib = [0.0, 1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0];
    for (k, e) in fib.iter().enumerate() {
        assert!((seq_at(&f, &n, k as i64) - e).abs() < 1e-8, "F({k})");
    }
    assert_recurrence(&f, &coeffs, None, &n, "fibonacci");
    // Without initial conditions: two constants.
    let g = rsolve_linear(&coeffs, None, &n, &[]).unwrap();
    assert!(g.contains(&ctx.symbol("C1")) && g.contains(&ctx.symbol("C2")));
}

#[test]
fn rsolve_towers_of_hanoi() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let coeffs = [ctx.int(-2), ctx.int(1)];
    let f = rsolve_linear(&coeffs, Some(&ctx.int(1)), &n, &[ctx.int(0)]).unwrap();
    assert_eq!(format!("{f}"), "2^n - 1");
    assert_recurrence(&f, &coeffs, Some(&ctx.int(1)), &n, "hanoi");
}

#[test]
fn rsolve_repeated_and_complex_roots() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    // a(n+2) - 4a(n+1) + 4a(n) = 0, a0 = 1, a1 = 4 → (1 + n) 2^n
    let coeffs = [ctx.int(4), ctx.int(-4), ctx.int(1)];
    let f = rsolve_linear(&coeffs, None, &n, &[ctx.int(1), ctx.int(4)]).unwrap();
    for k in 0..8 {
        let e = (1.0 + k as f64) * 2f64.powi(k as i32);
        assert!((seq_at(&f, &n, k) - e).abs() < 1e-8, "{f}");
    }
    // a(n+2) + a(n) = 0, a0 = 1, a1 = 0 → cos(nπ/2), real form
    let coeffs = [ctx.int(1), ctx.int(0), ctx.int(1)];
    let f = rsolve_linear(&coeffs, None, &n, &[ctx.int(1), ctx.int(0)]).unwrap();
    assert!(!f.contains(&ctx.i_unit()), "{f}");
    assert!(format!("{f}").contains("cos"), "{f}");
    assert_recurrence(&f, &coeffs, None, &n, "a(n+2) + a(n) = 0");
    // a(n+2) - 2a(n+1) + 2a(n) = 0 → (√2)^n (cos nπ/4, sin nπ/4)
    let coeffs = [ctx.int(2), ctx.int(-2), ctx.int(1)];
    let f = rsolve_linear(&coeffs, None, &n, &[ctx.int(1), ctx.int(1)]).unwrap();
    assert_recurrence(&f, &coeffs, None, &n, "spiral");
}

#[test]
fn rsolve_forcing_terms() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    // a(n+1) - a(n) = n (resonant polynomial) → n(n-1)/2
    let coeffs = [ctx.int(-1), ctx.int(1)];
    let f = rsolve_linear(&coeffs, Some(&n), &n, &[ctx.int(0)]).unwrap();
    assert_recurrence(&f, &coeffs, Some(&n), &n, "triangular");
    // a(n+1) - 2a(n) = 3^n → 3^n - 2^n
    let three_n = ctx.int(3).pow(&n);
    let coeffs = [ctx.int(-2), ctx.int(1)];
    let f = rsolve_linear(&coeffs, Some(&three_n), &n, &[ctx.int(0)]).unwrap();
    assert_recurrence(&f, &coeffs, Some(&three_n), &n, "3^n forcing");
    // a(n+1) - 2a(n) = 2^n (resonant exponential) → n 2^{n-1}
    let two_n = ctx.int(2).pow(&n);
    let f = rsolve_linear(&coeffs, Some(&two_n), &n, &[ctx.int(0)]).unwrap();
    assert_recurrence(&f, &coeffs, Some(&two_n), &n, "2^n resonance");
    // Mixed: a(n+2) - 3a(n+1) + 2a(n) = n + 2^(n+1)
    let forcing = &n + &ctx.int(2).pow(&(&n + 1));
    let coeffs = [ctx.int(2), ctx.int(-3), ctx.int(1)];
    let f = rsolve_linear(&coeffs, Some(&forcing), &n, &[ctx.int(0), ctx.int(0)]).unwrap();
    assert_recurrence(&f, &coeffs, Some(&forcing), &n, "mixed forcing");
    // Unsupported forcing shape.
    assert!(rsolve_linear(&coeffs, Some(&n.factorial()), &n, &[]).is_err());
}

#[test]
fn rsolve_first_order_variable_coefficients() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    // a(n+1) = (n+1) a(n), a0 = 1 → n!
    let f = rsolve_first_order(&(&n + 1), &ctx.int(0), &n, Some(&ctx.int(1))).unwrap();
    assert_eq!(format!("{f}"), "n!");
    // a(n+1) = 2 a(n) + 1 via the first-order path.
    let f = rsolve_first_order(&ctx.int(2), &ctx.int(1), &n, Some(&ctx.int(0))).unwrap();
    for k in 0..8 {
        assert!(
            (seq_at(&f, &n, k) - (2f64.powi(k as i32) - 1.0)).abs() < 1e-9,
            "{f}"
        );
    }
    // a(n+1) = (n+2) a(n), a0 = 1 → (n+1)!
    let f = rsolve_first_order(&(&n + 2), &ctx.int(0), &n, Some(&ctx.int(1))).unwrap();
    let mut expected = 1.0;
    for k in 0..7 {
        expected = if k == 0 {
            1.0
        } else {
            expected * (k as f64 + 1.0)
        };
        assert!(
            (seq_at(&f, &n, k) - expected).abs() < 1e-9,
            "a({k}) from {f}"
        );
    }
    // Without a0: C1 appears.
    let f = rsolve_first_order(&ctx.int(3), &ctx.int(0), &n, None).unwrap();
    assert!(f.contains(&ctx.symbol("C1")));
    // Gamma ratio for half-integer shift: a(n+1) = (2n+1) a(n), a0 = 1 → 2^n Γ(n+1/2)/Γ(1/2)
    let f = rsolve_first_order(&(&n * 2 + 1), &ctx.int(0), &n, Some(&ctx.int(1))).unwrap();
    let mut expected = 1.0;
    for k in 0..6 {
        if k > 0 {
            expected *= 2.0 * (k as f64 - 1.0) + 1.0;
        }
        assert!(
            (seq_at(&f, &n, k) - expected).abs() < 1e-7,
            "a({k}) = {} from {f}",
            seq_at(&f, &n, k)
        );
    }
}

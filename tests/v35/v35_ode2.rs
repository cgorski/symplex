//! Ordinary differential equations and recurrences: the re-hunt after 0.45
//! (polynomials over number fields, `Ei` at algebraic and complex points,
//! symbolic quartic roots).  Each test says what was refused, wrong or slow
//! before and cites its oracle: the solution substituted back into the
//! equation and evaluated at points and values of the constants, the
//! initial conditions evaluated, mpmath's `expm` for a system, direct
//! iteration of a recurrence, and SymPy 1.14 `dsolve` / `rsolve`.

use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

/// The ODE `text` with `D1Y`, `D2Y`, … standing for the derivatives of `y`.
fn ode(ctx: &Context, text: &str) -> Ex {
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let mut e = parse(ctx, text);
    let mut ders = vec![];
    let mut d = y;
    for _ in 0..6 {
        d = d.formal_diff(&x);
        ders.push(d.clone());
    }
    for k in (0..6).rev() {
        e = e.subs(&ctx.symbol(&format!("D{}Y", k + 1)), &ders[k]);
    }
    e
}

/// The integration constants `C1, C2, …` of `sol`, sorted by name.
fn constants(sol: &Ex) -> Vec<Ex> {
    let mut cs: Vec<Ex> = sol
        .free_symbols()
        .into_iter()
        .filter(|s| {
            let n = s.to_string();
            n.len() > 1 && n.starts_with('C') && n[1..].chars().all(|c| c.is_ascii_digit())
        })
        .collect();
    cs.sort_by_key(|c| c.to_string());
    cs
}

/// `sol` (explicit, `order` constants) solves the ODE `text`: the residual
/// is below `1e-9` of the derivatives' size at two points for two sets of
/// constants, and its value is real there (a real function written with
/// `i` is allowed).
fn assert_solves(ctx: &Context, text: &str, sol: &Ex, order: usize) {
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    assert!(!sol.contains(&y), "implicit: {sol}");
    assert!(!sol.has_unevaluated(), "unevaluated: {sol}");
    let cs = constants(sol);
    assert_eq!(
        cs.len(),
        order,
        "{} constants for order {order}: {sol}",
        cs.len()
    );
    let e = parse(ctx, text);
    let mut ders = vec![sol.clone()];
    for k in 0..order {
        let d = ders[k].diff(&x);
        ders.push(d);
    }
    let mut resid = e.subs(&y, sol);
    for k in (0..order).rev() {
        resid = resid.subs(&ctx.symbol(&format!("D{}Y", k + 1)), &ders[k + 1]);
    }
    let vals = [(2, 3), (5, 4), (-3, 7), (7, 5), (-1, 3), (3, 2)];
    for (xi, xv) in [(1, 3), (7, 5)].into_iter().enumerate() {
        let xv = ctx.rational(xv.0, xv.1);
        let mut r = resid.subs(&x, &xv);
        let mut scale = 1.0f64;
        let mut ds: Vec<Ex> = ders.iter().map(|d| d.subs(&x, &xv)).collect();
        for (i, c) in cs.iter().enumerate() {
            let v = vals[(i + xi) % vals.len()];
            let v = ctx.rational(v.0, v.1);
            r = r.subs(c, &v);
            for d in &mut ds {
                *d = d.subs(c, &v);
            }
        }
        for d in &ds {
            if let Ok(z) = d.eval_complex64() {
                scale = scale.max(z.norm());
            }
        }
        let value = ds[0].eval_complex64().unwrap();
        assert!(
            value.im.abs() <= 1e-9 * scale,
            "{sol} is not real at {xv}: {value}"
        );
        let z = r
            .eval_complex64()
            .unwrap_or_else(|err| panic!("residual of {sol} unevaluable: {err}"));
        assert!(z.norm() <= 1e-9 * scale, "{sol}: residual {z} at x = {xv}");
    }
}

/// `y⁽ᵏ⁾(x0) = values[k]` holds for `sol` (16 digits).
fn assert_ics(ctx: &Context, sol: &Ex, x0: &Ex, values: &[Ex]) {
    let x = ctx.symbol("x");
    let mut d = sol.clone();
    for (k, v) in values.iter().enumerate() {
        let z = (&d.subs(&x, x0) - v).eval_complex64().unwrap();
        assert!(z.norm() < 1e-10, "y^({k})({x0}) off by {z}: {sol}");
        d = d.diff(&x);
    }
}

fn ivp(ctx: &Context, text: &str, x0: &Ex, values: &[Ex]) -> Ex {
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let ics: Vec<InitialCondition> = values
        .iter()
        .enumerate()
        .map(|(order, v)| InitialCondition {
            order,
            x: x0.clone(),
            value: v.clone(),
        })
        .collect();
    ode(ctx, text).solve_ode_ivp(&y, &x, &ics).unwrap()
}

fn rationals(ctx: &Context, vs: &[(i64, i64)]) -> Vec<Ex> {
    vs.iter().map(|&(p, q)| ctx.rational(p, q)).collect()
}

/// `y″ + y = eˣ/x`: undetermined coefficients does not apply, and variation
/// of parameters with the real basis needs `∫ eˣ·sin x/x dx`, which the
/// integrator does not find: the ODE was refused (unevaluated `DSolve`;
/// SymPy 1.14 `dsolve` answers with Meijer G-functions).  The Green's
/// function with complex exponentials gives `Ei((1 ∓ i)x)`, a real
/// function written with `i`.  Oracle: substitution (the value is checked
/// to be real); the initial conditions at `1/3`.
#[test]
fn green_function_gives_ei_at_complex_points() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let text = "D2Y + y - exp(x)/x";
    let sol = ode(&ctx, text).solve_ode(&y, &x);
    assert!(sol.to_string().contains("Ei(x - x*I)"), "{sol}");
    assert_solves(&ctx, text, &sol, 2);
    let x0 = ctx.rational(1, 3);
    let values = rationals(&ctx, &[(2, 3), (5, 4)]);
    let fit = ivp(&ctx, text, &x0, &values);
    assert!(constants(&fit).is_empty(), "{fit}");
    assert_ics(&ctx, &fit, &x0, &values);
}

/// `y‴ − y′ = 1/x`: variation of parameters was only tried for order 2,
/// and the ODE was refused.  The Green's function of `r³ − r` (roots
/// `0, ±1`) gives `−ln|x| + eˣ·Ei(−x)/2 + e⁻ˣ·Ei(x)/2` (SymPy 1.14:
/// `C1 + C2·e⁻ˣ + C3·eˣ − log(x) + eˣ·Ei(−x)/2 + e⁻ˣ·Ei(x)/2`, the same
/// particular solution).  Oracle: substitution; SymPy.
#[test]
fn third_order_green_function_with_ei() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let text = "D3Y - D1Y - 1/x";
    let sol = ode(&ctx, text).solve_ode(&y, &x);
    let s = sol.to_string();
    assert!(
        s.contains("Ei(-x)") && s.contains("Ei(x)") && s.contains("ln("),
        "{s}"
    );
    assert_solves(&ctx, text, &sol, 3);
}

/// `y″ + y = sin x/x`: refused (variation of parameters needs `∫ sin²x/x`
/// and `∫ sin x·cos x/x` in the trigonometric form).  In 0.46 the Green's
/// function with `sin x` written as exponentials gave `Ei(±2ix)`; since 0.47
/// the integrator's sin/cos products over a rational function give
/// `Si(2x)`, `Ci(2x)` directly (SymPy 1.14: `(C1 + Si(2x)/2)·sin x + (C2 −
/// log(x)/2 + Ci(2x)/2)·cos x`).  Oracle: substitution (real values); SymPy.
#[test]
fn green_function_sin_over_x() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let text = "D2Y + y - sin(x)/x";
    let sol = ode(&ctx, text).solve_ode(&y, &x);
    let s = sol.to_string();
    assert!(s.contains("Si(2*x)") || s.contains("Ei(2*x*I)"), "{sol}");
    assert_solves(&ctx, text, &sol, 2);
}

/// `y⁗ + 3y″ + y = e^{√2·x}`: undetermined coefficients only took rational
/// rates and the ODE was refused (SymPy 1.14: `exp(√2·x)/11` plus the
/// homogeneous part).  Now the exponential shift in exact arithmetic over
/// `ℚ(√2)`: `p(√2) = 11`.  The characteristic polynomial `r⁴ + 3r² + 1`
/// (irreducible over ℚ, biquadratic) also had a `RootOf` basis; its roots
/// `±i·√(3/2 ∓ √5/2)` are now used.  Oracle: substitution; SymPy.
#[test]
fn irrational_exponential_rate_by_undetermined_coefficients() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let text = "D4Y + 3*D2Y + y - exp(sqrt(2)*x)";
    let sol = ode(&ctx, text).solve_ode(&y, &x);
    let s = sol.to_string();
    assert!(s.contains("1/11*exp(x*sqrt(2))"), "{s}");
    assert!(!s.contains("RootOf") && !s.contains('I'), "{s}");
    assert_solves(&ctx, text, &sol, 4);
}

/// `y″ − 2y = x·e^{√2·x}`: resonance with the characteristic root `√2`
/// (refused before, the rate being irrational).  The vanishing of `p(√2)` is
/// decided exactly and the trial gains a factor `x`: `y_p =
/// (√2/8)·x²·e^{√2x} − (1/8)·x·e^{√2x}` (SymPy 1.14: `C1·e^{−√2x} +
/// (C2 + √2x²/8 − x/8)·e^{√2x}`).  Oracle: substitution; SymPy.
#[test]
fn irrational_rate_resonance() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let text = "D2Y - 2*y - x*exp(sqrt(2)*x)";
    let sol = ode(&ctx, text).solve_ode(&y, &x);
    let s = sol.to_string();
    assert!(s.contains("1/8*sqrt(2)*x^2*exp(x*sqrt(2))"), "{s}");
    assert_solves(&ctx, text, &sol, 2);
}

/// `y″ + 3y = 2ˣ`: the power `2ˣ` was not a forcing term of undetermined
/// coefficients (refused).  Now `e^{x·ln 2}`, rate `ln 2`:
/// `y_p = 2ˣ/(ln²2 + 3)` (SymPy 1.14: `2**x/(log(2)**2 + 3)`).  Oracle:
/// substitution; SymPy.
#[test]
fn power_forcing_with_transcendental_rate() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let text = "D2Y + 3*y - 2^x";
    let sol = ode(&ctx, text).solve_ode(&y, &x);
    assert!(
        sol.to_string().contains("exp(x*ln(2))/(ln(2)^2 + 3)"),
        "{sol}"
    );
    assert_solves(&ctx, text, &sol, 2);
}

/// `y⁗ + y″ + y = cos(√2·x)`: refused (irrational frequency).  Now
/// `p(i√2) = 4 − 2 + 1 = 3`: `y_p = cos(√2·x)/3` (SymPy 1.14 agrees).
/// Oracle: substitution; SymPy.
#[test]
fn irrational_frequency_by_undetermined_coefficients() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let text = "D4Y + D2Y + y - cos(sqrt(2)*x)";
    let sol = ode(&ctx, text).solve_ode(&y, &x);
    assert!(sol.to_string().contains("1/3*cos(x*sqrt(2))"), "{sol}");
    assert_solves(&ctx, text, &sol, 4);
}

/// `y⁗ + y = sinh x`: the forcing was not of the form
/// `xᵈ·e^{ax}·cos/sin(bx)` and the ODE was refused.  `sinh x` is now
/// written `(eˣ − e⁻ˣ)/2` for undetermined coefficients and the pair
/// `eˣ/4 − e⁻ˣ/4` written back: `y_p = sinh(x)/2` (SymPy 1.14 agrees).
/// `y″ − y = cosh x` (resonance) gives `x·sinh(x)/2`.  Oracle:
/// substitution; SymPy.
#[test]
fn hyperbolic_forcing_as_exponentials() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let text = "D4Y + y - sinh(x)";
    let sol = ode(&ctx, text).solve_ode(&y, &x);
    assert!(sol.to_string().contains("1/2*sinh(x)"), "{sol}");
    assert_solves(&ctx, text, &sol, 4);
    let text = "D2Y - y - cosh(x)";
    let sol = ode(&ctx, text).solve_ode(&y, &x);
    assert!(sol.to_string().contains("1/2*x*sinh(x)"), "{sol}");
    assert_solves(&ctx, text, &sol, 2);
}

/// `y⁗ − 10y″ + y = 0` (roots `±√2 ± √3`) and `y⁗ + 3y″ + y = 0`: the
/// solver's roots `±√(5 ± 2√6)`, `±√(−3/2 ± √5/2)` were rejected because
/// the real and imaginary parts of `√u` with a negative radical `u` stayed
/// `re(…)`, `im(…)`, and the basis fell back to `RootOf`.  Now
/// `√u = i·√(−u)`: real exponentials and frequencies in radicals.  The
/// initial-value problem at `0` is fitted by the Lagrange formula.
/// Oracle: substitution; the conditions.
#[test]
fn biquadratic_characteristic_polynomials_in_radicals() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    for text in ["D4Y - 10*D2Y + y", "D4Y + 3*D2Y + y"] {
        let sol = ode(&ctx, text).solve_ode(&y, &x);
        let s = sol.to_string();
        assert!(
            !s.contains("RootOf") && !s.contains('I') && s.contains("sqrt"),
            "{s}"
        );
        assert_solves(&ctx, text, &sol, 4);
        let x0 = ctx.int(0);
        let values = rationals(&ctx, &[(1, 1), (0, 1), (-2, 3), (1, 2)]);
        let fit = ivp(&ctx, text, &x0, &values);
        assert!(
            constants(&fit).is_empty() && !fit.to_string().contains("RootOf"),
            "{fit}"
        );
        assert_ics(&ctx, &fit, &x0, &values);
    }
}

/// `x³y‴ + 3x²y″ − 2xy′ + y = √x` (indicial polynomial `m³ − 3m + 1`, a
/// casus irreducibilis): `x = eᵗ` turned `√x` into `(eᵗ)^{1/2}`, which is
/// not `e^{t/2}` for complex `t`, so the forcing was not recognised and the
/// ODE was refused (SymPy 1.14 `dsolve`: `NotImplementedError`).  `xᵖ` now
/// maps to `e^{pt}` (`x > 0`): `y_p = −(8/3)·√x` (`p(1/2) = −3/8`).
/// Oracle: substitution.
#[test]
fn cauchy_euler_square_root_forcing_order_three() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let text = "x^3*D3Y + 3*x^2*D2Y - 2*x*D1Y + y - sqrt(x)";
    let sol = ode(&ctx, text).solve_ode(&y, &x);
    assert!(sol.to_string().contains("- 8/3*sqrt(x)"), "{sol}");
    assert_solves(&ctx, text, &sol, 3);
}

/// `x⁴y⁗ + 6x³y‴ + 7x²y″ + 2xy′ + y = x²` (indicial polynomial `m⁴ + m + 1`,
/// `RootOf` exponents), conditions at `x₀ = 7/5`: the generic fit over
/// `re`/`im` of the `RootOf`s took about 5 s in a debug build (and timed out
/// the hunter with `√x` forcing).  Now `x = x₀·eᵗ` and the exact Lagrange
/// fit of the constant-coefficient equation (0.25 s).  Oracle: substitution;
/// the conditions.
#[test]
fn cauchy_euler_rootof_initial_values_by_lagrange() {
    let ctx = Context::new();
    let text = "x^4*D4Y + 6*x^3*D3Y + 7*x^2*D2Y + 2*x*D1Y + y - x^2";
    let x0 = ctx.rational(7, 5);
    let values = rationals(&ctx, &[(2, 3), (5, 4), (-3, 7), (7, 5)]);
    let fit = ivp(&ctx, text, &x0, &values);
    assert!(constants(&fit).is_empty(), "{fit}");
    assert_ics(&ctx, &fit, &x0, &values);
    // the fitted function still solves the equation (at x = 2)
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let mut ders = vec![fit.clone()];
    for k in 0..4 {
        let d = ders[k].diff(&x);
        ders.push(d);
    }
    let mut r = parse(&ctx, text).subs(&y, &fit);
    for k in (0..4).rev() {
        r = r.subs(&ctx.symbol(&format!("D{}Y", k + 1)), &ders[k + 1]);
    }
    let z = r.subs(&x, &ctx.int(2)).eval_complex64().unwrap();
    assert!(z.norm() < 1e-8, "residual {z}");
}

/// `y′ + y = cos x/x`: the linear solver integrated `eˣ·cos x/x` as given and
/// returned a formal `Integral`.  With `cos x` written as exponentials the
/// antiderivative is `(Ei((1 + i)x) + Ei((1 − i)x))/2` (SymPy 1.14: a
/// Meijer G-function).  Oracle: substitution (real values).
#[test]
fn first_order_linear_cos_over_x_integrates() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let text = "D1Y + y - cos(x)/x";
    let sol = ode(&ctx, text).solve_ode(&y, &x);
    assert!(sol.to_string().contains("Ei(x + x*I)"), "{sol}");
    assert_solves(&ctx, text, &sol, 1);
}

/// `X′ = A·X` with a dense 4×4 `A` whose characteristic polynomial
/// `λ⁴ + 5λ³ + 8λ² + 7λ + 11` is irreducible: the general solution is
/// `e^{At}·c` over `RootOf`s, and the initial-value problem solved
/// `X(0) = x₀` by elimination over the `RootOf` entries, which did not
/// finish (the hunter's 8 s limit; over a minute).  It is now `e^{At}·x₀`.
/// Oracle: mpmath 1.3 `expm(A)*x0` at `t = 1` (30 digits):
/// `(−0.18928231132088792257, −0.55639210872036066650,
/// 0.59525282598616762163, 0.45112908655300632528)`.
#[test]
fn four_by_four_system_with_rootof_eigenvalues_initial_values() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let rows = [
        [-1, 2, -1, -1],
        [-1, -1, -2, -1],
        [1, 0, -1, 2],
        [0, -2, 1, -2],
    ];
    let a = symplex::matrix::Matrix::new(
        rows.iter()
            .map(|r| r.iter().map(|&v| ctx.int(v)).collect())
            .collect(),
    )
    .unwrap();
    let x0 = rationals(&ctx, &[(2, 3), (5, 4), (-3, 7), (7, 5)]);
    let sol = symplex::ode::solve_ode_system_ivp(&a, &t, &x0).unwrap();
    let expected = [
        -0.189_282_311_320_888,
        -0.556_392_108_720_361,
        0.595_252_825_986_168,
        0.451_129_086_553_006,
    ];
    for (i, (s, e)) in sol.iter().zip(expected).enumerate() {
        let at0 = (&s.subs(&t, &ctx.int(0)) - &x0[i])
            .eval_complex64()
            .unwrap();
        assert!(at0.norm() < 1e-12, "x{i}(0) off by {at0}");
        let v = s.subs(&t, &ctx.int(1)).eval_complex64().unwrap();
        assert!(
            (v.re - e).abs() < 1e-12 && v.im.abs() < 1e-12,
            "x{i}(1) = {v}, mpmath {e}"
        );
    }
}

/// The value of the closed form `sol` at `n = k` as an `f64` (real part;
/// the imaginary part must vanish).
fn seq_value(sol: &Ex, n: &Ex, k: i64) -> f64 {
    let z = sol.subs_i64(n, k).eval_complex64().unwrap();
    assert!(
        z.im.abs() <= 1e-9 * z.re.abs().max(1.0),
        "a({k}) = {z} from {sol}"
    );
    z.re
}

/// `a(n+2) − a(n+1) − a(n) = cos(nπ/2)`, `a(0) = 0`, `a(1) = 1`: refused
/// ("unsupported forcing term"; SymPy 1.14 `rsolve`: "The independent term
/// should be a sum of hypergeometric functions").  The trial
/// `A·cos(nπ/2) + B·sin(nπ/2)` gives the particular solution.  Oracle:
/// direct iteration with exact integers, 30 terms.
#[test]
fn rsolve_cosine_forcing() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let f = parse(&ctx, "cos(n*pi/2)");
    let coeffs = [ctx.int(-1), ctx.int(-1), ctx.int(1)];
    let sol =
        symplex::rsolve::rsolve_linear(&coeffs, Some(&f), &n, &[ctx.int(0), ctx.int(1)]).unwrap();
    let cos_quarter = [1i64, 0, -1, 0];
    let mut a: Vec<i64> = vec![0, 1];
    for m in 0..28 {
        a.push(a[m + 1] + a[m] + cos_quarter[m % 4]);
    }
    for (k, v) in a.iter().enumerate() {
        let got = seq_value(&sol, &n, k as i64);
        assert!(
            (got - *v as f64).abs() <= 1e-9 * (*v as f64).abs().max(1.0),
            "a({k}) = {got}, iteration {v}: {sol}"
        );
    }
}

/// `a(n+2) + a(n) = cos(nπ/2)` with `a(0) = 1`, `a(1) = 0`: `e^{iπ/2} = i`
/// is a characteristic root, so the trial needs the factor `n`
/// (resonance), found by trying `n⁰, n¹, …` until the system is consistent.
/// Refused before.  Oracle: direct iteration with exact integers, 30 terms.
#[test]
fn rsolve_resonant_cosine_forcing() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let f = parse(&ctx, "cos(n*pi/2)");
    let coeffs = [ctx.int(1), ctx.int(0), ctx.int(1)];
    let sol =
        symplex::rsolve::rsolve_linear(&coeffs, Some(&f), &n, &[ctx.int(1), ctx.int(0)]).unwrap();
    let cos_quarter = [1i64, 0, -1, 0];
    let mut a: Vec<i64> = vec![1, 0];
    for m in 0..28 {
        a.push(cos_quarter[m % 4] - a[m]);
    }
    for (k, v) in a.iter().enumerate() {
        let got = seq_value(&sol, &n, k as i64);
        assert!(
            (got - *v as f64).abs() <= 1e-9 * (*v as f64).abs().max(1.0),
            "a({k}) = {got}, iteration {v}: {sol}"
        );
    }
}

/// `a(n+4) − 2a(n) = 2·sin(nπ/6)` with `a(0..4) = (−3, −2, 2, 1)`: the
/// values of the forcing lie in `ℚ(√3)`, so the component-wise fit (rational
/// data only) did not apply and the global elimination over `2^{1/4}`, `i`
/// and `√3` did not finish (the hunter's limit; refused as unsupported
/// before the trigonometric forcing).  The fit is linear in the data: one
/// rational fit per unit datum.  Oracle: direct iteration (f64), 30 terms.
#[test]
fn rsolve_sine_forcing_with_irrational_values() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let f = parse(&ctx, "2*sin(n*pi/6)");
    let coeffs = [ctx.int(-2), ctx.int(0), ctx.int(0), ctx.int(0), ctx.int(1)];
    let ics = [ctx.int(-3), ctx.int(-2), ctx.int(2), ctx.int(1)];
    let sol = symplex::rsolve::rsolve_linear(&coeffs, Some(&f), &n, &ics).unwrap();
    let mut a: Vec<f64> = vec![-3.0, -2.0, 2.0, 1.0];
    for m in 0..26 {
        let fm = 2.0 * (m as f64 * std::f64::consts::PI / 6.0).sin();
        a.push(2.0 * a[m] + fm);
    }
    for (k, v) in a.iter().enumerate() {
        let got = seq_value(&sol, &n, k as i64);
        assert!(
            (got - v).abs() <= 1e-9 * v.abs().max(1.0),
            "a({k}) = {got}, iteration {v}: {sol}"
        );
    }
}

/// `a(n+2) − 2a(n) = (−√2)ⁿ + n` with `a(0) = 1`, `a(1) = 0`: `eval` writes
/// `(−√2)ⁿ` as `(−1)ⁿ·2^{n/2}`, a product of two exponentials, which was
/// refused as an unsupported forcing term.  The bases now multiply, and
/// `−√2` is a characteristic root (resonance: the trial gains a factor
/// `n`).  Oracle: direct iteration (f64), 25 terms.
#[test]
fn rsolve_product_of_exponential_bases_with_resonance() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let f = parse(&ctx, "(-sqrt(2))^n + n");
    let coeffs = [ctx.int(-2), ctx.int(0), ctx.int(1)];
    let sol =
        symplex::rsolve::rsolve_linear(&coeffs, Some(&f), &n, &[ctx.int(1), ctx.int(0)]).unwrap();
    let mut a: Vec<f64> = vec![1.0, 0.0];
    for m in 0..23 {
        let fm = (-std::f64::consts::SQRT_2).powi(m as i32) + m as f64;
        a.push(2.0 * a[m] + fm);
    }
    for (k, v) in a.iter().enumerate() {
        let got = seq_value(&sol, &n, k as i64);
        assert!(
            (got - v).abs() <= 1e-9 * v.abs().max(1.0),
            "a({k}) = {got}, iteration {v}: {sol}"
        );
    }
}

/// `a(n+1) − 2a(n) = sin n + n·cos(2n)`, `a(0) = 1`: a general angle (the
/// values `cos k`, `sin k` stay symbolic).  Refused before.  Oracle: direct
/// iteration (f64), 20 terms.
#[test]
fn rsolve_trigonometric_forcing_general_angle() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let f = parse(&ctx, "sin(n) + n*cos(2*n)");
    let coeffs = [ctx.int(-2), ctx.int(1)];
    let sol = symplex::rsolve::rsolve_linear(&coeffs, Some(&f), &n, &[ctx.int(1)]).unwrap();
    let mut a: Vec<f64> = vec![1.0];
    for m in 0..19 {
        let mf = m as f64;
        a.push(2.0 * a[m] + mf.sin() + mf * (2.0 * mf).cos());
    }
    for (k, v) in a.iter().enumerate() {
        let got = seq_value(&sol, &n, k as i64);
        assert!(
            (got - v).abs() <= 1e-9 * v.abs().max(1.0),
            "a({k}) = {got}, iteration {v}: {sol}"
        );
    }
}

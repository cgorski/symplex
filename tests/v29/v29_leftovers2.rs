//! After 0.29 — the second round of leftovers: directed infinities
//! (`x·∞` keeps the direction of `x`), fitting initial conditions to a
//! system nonlinear in the constants, the missing families of `y′ = y³` and
//! `y″ = y′³`, constant multiples the integrator lost, `J_{−n}(0)` and the
//! value of `Chi`/`Ci` at their logarithmic singularity.
//!
//! Each test says what was wrong before; every reference value cites the
//! oracle call that produced it (SymPy 1.14, mpmath 1.3.0).

// Reference values are quoted at the digits the oracle printed them.
#![allow(clippy::excessive_precision)]

use symplex::prelude::*;

fn p(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap_or_else(|e| panic!("{s}: {e}"))
}

fn show(e: &Ex) -> String {
    e.to_string()
}

/// `e` displays as `shown`, and the display parses back to `e`.
fn assert_shows(ctx: &Context, e: &Ex, shown: &str) {
    assert_eq!(show(e), shown, "{}", e.to_srepr());
    assert_eq!(&p(ctx, shown), e, "{shown} parses to a different tree");
}

/// `d/dx big_f − f` vanishes (to `1e-9`) at the sample points.
fn assert_antiderivative(ctx: &Context, f: &Ex, big_f: &Ex, x: &Ex, points: &[&str]) {
    assert!(
        !big_f.has_unevaluated(),
        "∫ {f} stayed unevaluated: {big_f}"
    );
    let residual = big_f.diff(x) - f;
    for pt in points {
        let v = residual
            .subs(x, &p(ctx, pt))
            .eval_complex64()
            .unwrap_or_else(|e| panic!("∫ {f} = {big_f}: residual at {pt}: {e}"));
        assert!(v.norm() < 1e-9, "∫ {f} = {big_f}: F' − f = {v} at x = {pt}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Directed infinities
// ═══════════════════════════════════════════════════════════════════════════

/// `x·∞` was `zoo` (canonical multiplication recorded only that the product
/// is infinite), so `(x·∞)` at `x = 0` was `zoo` instead of `0·∞ = nan`, at
/// `x = −1` it could not be `−∞`, and `i·∞` lost its direction.  A factor of
/// unknown direction now stays in the product, as in SymPy.
///
/// SymPy: `x*oo` → `oo*x`; `(x*oo).subs(x, 0)` → nan, `.subs(x, 2)` → oo,
/// `.subs(x, -1)` → -oo, `.subs(x, I)` → `oo*I`, `.subs(x, 1 + I)` →
/// `oo*(1 + I)`; `x*oo - x*oo` → nan; `x*oo + x*oo` → `oo*x`; `x*oo*2` →
/// `oo*x`; `-x*oo` → `-oo*x` (`Mul(-1, oo, x)`); `pi*x*oo` → `oo*x`;
/// `x*y*oo` → `oo*x*y`; `I*oo` → `oo*I`; `sqrt(2)*I*oo` → `oo*I`.
#[test]
fn directed_infinities_keep_the_direction_of_their_factors() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let oo = ctx.infinity();
    let e = &x * &oo;
    assert_shows(&ctx, &e, "x*oo");
    for (at, want) in [
        ("0", "nan"),
        ("2", "oo"),
        ("-1", "-oo"),
        ("I", "I*oo"),
        ("1 + I", "(1 + I)*oo"),
        ("pi", "oo"),
        ("sqrt(2) - 2", "-oo"),
    ] {
        assert_eq!(show(&e.subs(&x, &p(&ctx, at))), want, "x*oo at x = {at}");
    }
    assert_eq!(show(&(&e - &e)), "nan", "x*oo - x*oo");
    assert_eq!(show(&(&e + &e)), "x*oo");
    assert_eq!(show(&(&e * 2)), "x*oo");
    assert_shows(&ctx, &-&e, "-x*oo");
    assert_shows(&ctx, &p(&ctx, "pi*x*oo"), "x*oo");
    assert_shows(&ctx, &p(&ctx, "x*y*oo"), "x*y*oo");
    assert_shows(&ctx, &(ctx.i_unit() * &oo), "I*oo");
    assert_shows(&ctx, &p(&ctx, "sqrt(2)*I*oo"), "I*oo");
    assert_shows(&ctx, &p(&ctx, "-I*oo"), "-I*oo");
    // A factor of known sign is still absorbed (unchanged).
    // SymPy: `log(2)*oo` → oo, `(3 - pi)*oo` → -oo.
    assert_eq!(show(&p(&ctx, "ln(2)*oo")), "oo");
    assert_eq!(show(&p(&ctx, "(3 - pi)*oo")), "-oo");
    let q = ctx.symbol_with("q", &[Assumption::Positive]).unwrap();
    assert_eq!(show(&(&q * &oo)), "oo");
}

/// Sums and powers of directed infinities.  A symbol is finite in
/// symplex, so a finite term next to `x·∞` is absorbed (SymPy keeps
/// `oo*x + 1`, and `x + oo`, where symplex already gives `oo`).
///
/// SymPy: `x*oo + oo` → `oo*x + oo`, and at `x = -1` → nan, at `x = 3` → oo;
/// `oo*x - oo*y` stays; `1/(x*oo)` → 0; `(x*oo)**2` → `oo*x**2`;
/// `x*oo/x` → oo; `(I*oo)**2` → -oo; `exp(x)*oo` → `oo*exp(x)`;
/// `(x*zoo).subs(x, 0)` → nan (symplex keeps `x·zoo = zoo`, like `x/x = 1`,
/// for a generic `x`: `zoo` has no direction to lose).
#[test]
fn sums_and_powers_of_directed_infinities() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(show(&p(&ctx, "x*oo + 1")), "x*oo");
    let s = p(&ctx, "x*oo + oo");
    assert_eq!(&p(&ctx, &show(&s)), &s);
    assert_eq!(show(&s.subs(&x, &ctx.int(-1))), "nan");
    assert_eq!(show(&s.subs(&x, &ctx.int(3))), "oo");
    assert_eq!(show(&(&s - &p(&ctx, "x*oo"))), "nan");
    let s = p(&ctx, "x*oo - oo");
    assert_eq!(&p(&ctx, &show(&s)), &s, "{s}");
    assert_eq!(show(&s.subs(&x, &ctx.int(-2))), "-oo");
    assert_eq!(show(&s.subs(&x, &ctx.int(2))), "nan");
    let s = p(&ctx, "-(1 + I)*oo");
    assert_eq!(&p(&ctx, &show(&s)), &s, "{s}");
    // zoo + anything infinite is nan.
    assert_eq!(show(&p(&ctx, "x*oo + zoo")), "nan");
    let d = p(&ctx, "x*oo - y*oo");
    assert_shows(&ctx, &d, "x*oo - y*oo");
    assert_eq!(show(&d.subs(&x, &ctx.int(0))), "nan");
    assert_eq!(show(&p(&ctx, "1/(x*oo)")), "0");
    assert_shows(&ctx, &p(&ctx, "(x*oo)^2"), "x^2*oo");
    assert_eq!(show(&p(&ctx, "x*oo/x")), "oo");
    assert_eq!(show(&p(&ctx, "(I*oo)^2")), "-oo");
    assert_shows(&ctx, &p(&ctx, "exp(x)*oo"), "exp(x)*oo");
    assert_eq!(show(&p(&ctx, "0*x*oo")), "nan");
    assert_eq!(show(&p(&ctx, "x*zoo")), "zoo");
}

/// `atan(±i)`, `asin(±∞)`, `acos(±∞)` were `zoo`, and `√(−∞)` stayed an atom.
///
/// SymPy: `atan(I)` → `oo*I`, `atan(-I)` → `-oo*I`, `asin(oo)` → `-oo*I`,
/// `asin(-oo)` → `oo*I`, `acos(oo)` → `oo*I`, `acos(-oo)` → `-oo*I`,
/// `sqrt(-oo)` → `oo*I`, `(-oo)**Rational(3, 2)` → `-oo*I`,
/// `(-oo)**Rational(1, 3)` → `oo*(-1)**(1/3)`.
#[test]
fn inverse_functions_at_their_infinite_points_are_directed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (s, want) in [
        ("atan(I)", "I*oo"),
        ("atan(-I)", "-I*oo"),
        ("asin(oo)", "-I*oo"),
        ("asin(-oo)", "I*oo"),
        ("acos(oo)", "I*oo"),
        ("acos(-oo)", "-I*oo"),
        ("sqrt(-oo)", "I*oo"),
        ("(-oo)^(3/2)", "-I*oo"),
        ("(-oo)^(1/3)", "cbrt(-1)*oo"),
    ] {
        assert_shows(&ctx, &p(&ctx, s), want);
    }
    // Every path that builds the application folds it.
    assert_eq!(show(&x.atan().subs(&x, &ctx.i_unit())), "I*oo");
    assert_eq!(show(&x.asin().subs(&x, &ctx.infinity())), "-I*oo");
}

/// The elementary functions at `±i·∞` (were their values at `zoo`: `nan`
/// for `sin`, `cos`, `tan`, `zoo` for `ln`).
///
/// SymPy: `sin(oo*I)` → `oo*I`, `cos(oo*I)` → oo, `tan(oo*I)` → I,
/// `tan(-oo*I)` → -I, `exp(oo*I)` → nan, `log(oo*I)` → oo, `asinh(oo*I)` →
/// oo, `asinh(-oo*I)` → -oo, `acosh(oo*I)` → `oo + I*pi/2`, `acos(oo*I)` →
/// `pi/2 - oo*I`, `erf(oo*I)` → `oo*I`, `erfc(oo*I)` → `-oo*I`,
/// `Abs(oo*I)` → oo, `sign(-oo*I)` → -I (a finite term next to an infinite
/// one is absorbed in symplex).
#[test]
fn elementary_functions_at_imaginary_infinity() {
    let ctx = Context::new();
    for (s, want) in [
        ("sin(I*oo)", "I*oo"),
        ("cos(I*oo)", "oo"),
        ("tan(I*oo)", "I"),
        ("tan(-I*oo)", "-I"),
        ("exp(I*oo)", "nan"),
        ("ln(I*oo)", "oo"),
        ("asinh(I*oo)", "oo"),
        ("asinh(-I*oo)", "-oo"),
        ("acosh(I*oo)", "oo"),
        ("acos(I*oo)", "-I*oo"),
        ("erf(I*oo)", "I*oo"),
        ("erfc(I*oo)", "-I*oo"),
        ("abs(I*oo)", "oo"),
        ("sign(-I*oo)", "-I"),
    ] {
        assert_eq!(show(&p(&ctx, s)), want, "{s}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Initial-value problems
// ═══════════════════════════════════════════════════════════════════════════

fn ic(order: usize, x: Ex, value: Ex) -> InitialCondition {
    InitialCondition { order, x, value }
}

/// `y″ + 2y′² = 0` with `y(1/3) = 2/3`, `y′(1/3) = 5/4` failed with "could
/// not solve for C2": the fit solved the first condition, which involves
/// both constants, for the first constant.  The condition on `y′` involves
/// only `C1` and is solved first now.
///
/// SymPy: `dsolve(y(x).diff(x, 2) + 2*y(x).diff(x)**2, y(x), ics={y(1/3):
/// 2/3, y'(1/3): 5/4})` → `log(2*x + 2/15)/2 - log(2) + 2/3 + log(5)/2`,
/// at x = 1: `N(…, 20)` = 1.1570812931725297851.
#[test]
fn initial_conditions_are_fitted_in_triangular_order() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let yp = y.formal_diff(&x);
    let ode = yp.formal_diff(&x) + ctx.int(2) * yp.powi(2);
    let x0 = ctx.rational(1, 3);
    let ics = [
        ic(0, x0.clone(), ctx.rational(2, 3)),
        ic(1, x0.clone(), ctx.rational(5, 4)),
    ];
    let sol = ode.solve_ode_ivp(&y, &x, &ics).expect("fits");
    assert!(ode.check_ode_solution(&sol, &y, &x), "{sol}");
    let at = |e: &Ex| e.subs(&x, &x0).eval_f64().unwrap();
    assert!((at(&sol) - 2.0 / 3.0).abs() < 1e-12, "{sol}");
    assert!((at(&sol.diff(&x)) - 1.25).abs() < 1e-12, "{sol}");
    let v = sol.subs(&x, &ctx.int(1)).eval_f64().unwrap();
    assert!(
        (v - 1.157_081_293_172_529_8).abs() < 1e-12,
        "{sol} at 1: {v}"
    );
}

/// `y′ = y³` returned only `(C1 − 2x)^(−1/2)`, so `y(0) = −1` had no
/// solution; `y″ = y′³` likewise only `C2 − √(C1 − 2x)`, so `y′(0) = −1`
/// failed.  Both families are returned now and the conditions are fitted
/// on each.
///
/// SymPy: `dsolve(y(x).diff(x) - y(x)**3)` → `[-sqrt(2)*sqrt(-1/(C1 + x))/2,
/// sqrt(2)*sqrt(-1/(C1 + x))/2]`; with `ics={y(0): -1}` →
/// `-sqrt(2)*sqrt(-1/(x - 1/2))/2`, at x = 1/4: -1.4142135623730950488;
/// `dsolve(y(x).diff(x, 2) - y(x).diff(x)**3)` → `[C1 - sqrt(2)*sqrt(-1/(C2
/// + x))*(C2 + x), C1 + …]`; with `ics={y(0): 0, y'(0): -1}` →
/// `-sqrt(2)*sqrt(-1/(x - 1/2))*(x - 1/2) - 1`, at x = 1/4:
/// -0.29289321881345247560; `dsolve(y(x).diff(x) - x/y(x))` →
/// `[-sqrt(C1 + x**2), sqrt(C1 + x**2)]`.
#[test]
fn every_family_of_the_general_solution_is_returned() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let yp = y.formal_diff(&x);
    let cube = &yp - y.powi(3);
    let all = cube.solve_ode_all(&y, &x).unwrap();
    assert_eq!(all.len(), 2, "{all:?}");
    assert_eq!(all[1], -&all[0]);
    assert!(all.iter().all(|s| cube.check_ode_solution(s, &y, &x)));
    let sol = cube
        .solve_ode_ivp(&y, &x, &[ic(0, ctx.int(0), ctx.int(-1))])
        .expect("the negative family");
    let v = sol.subs(&x, &ctx.rational(1, 4)).eval_f64().unwrap();
    assert!((v + std::f64::consts::SQRT_2).abs() < 1e-12, "{sol}: {v}");
    // The positive condition still lands on the first family.
    let sol = cube
        .solve_ode_ivp(&y, &x, &[ic(0, ctx.int(0), ctx.int(1))])
        .unwrap();
    assert!(
        (sol.subs(&x, &ctx.rational(1, 4)).eval_f64().unwrap() - std::f64::consts::SQRT_2).abs()
            < 1e-12
    );

    let second = yp.formal_diff(&x) - yp.powi(3);
    let all = second.solve_ode_all(&y, &x).unwrap();
    assert_eq!(all.len(), 2, "{all:?}");
    assert!(all.iter().all(|s| second.check_ode_solution(s, &y, &x)));
    let sol = second
        .solve_ode_ivp(
            &y,
            &x,
            &[
                ic(0, ctx.int(0), ctx.int(0)),
                ic(1, ctx.int(0), ctx.int(-1)),
            ],
        )
        .expect("y'(0) = -1 is on the second family");
    assert!(second.check_ode_solution(&sol, &y, &x));
    let v = sol.subs(&x, &ctx.rational(1, 4)).eval_f64().unwrap();
    assert!((v + 0.292_893_218_813_452_48).abs() < 1e-12, "{sol}: {v}");

    let ratio = &yp - &x / &y;
    let all = ratio.solve_ode_all(&y, &x).unwrap();
    assert_eq!(all.len(), 2, "{all:?}");
    assert!(all.iter().all(|s| ratio.check_ode_solution(s, &y, &x)));
    // `solve_ode` still returns the first family.
    assert_eq!(
        cube.solve_ode(&y, &x),
        cube.solve_ode_all(&y, &x).unwrap()[0]
    );
}

/// A root of the constants' equations must also satisfy the ODE at the
/// initial point: for `y′ = √y` (general solution `(x/2 + C1/2)²`, a
/// solution where `x + C1 ≥ 0`), `y(−4) = 1` gives `C1 ∈ {2, 6}`, and only
/// `C1 = 6` has `y′(−4) = √y(−4)`.  Before 0.30 the first root `solve`
/// returned was kept unchecked (here it happened to be the right one; the
/// check makes it independent of the order).  SymPy:
/// `dsolve(y(x).diff(x) - sqrt(y(x)), ics={y(0): 1})` raises "Initial
/// conditions produced too many solutions for constants".
#[test]
fn a_fitted_solution_satisfies_the_equation_at_the_initial_point() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let yp = y.formal_diff(&x);
    for (sign, x0, want) in [
        (1, -4, "(1/2*x + 3)^2"),
        (1, 4, "(1/2*x - 1)^2"),
        (1, 0, "(1/2*x + 1)^2"),
        (-1, 0, "(-1/2*x + 1)^2"),
    ] {
        let ode = &yp - ctx.int(sign) * y.sqrt();
        let sol = ode
            .solve_ode_ivp(&y, &x, &[ic(0, ctx.int(x0), ctx.int(1))])
            .unwrap();
        assert_eq!(show(&sol), want, "sign {sign}, x0 = {x0}");
        let r = (sol.diff(&x) - ctx.int(sign) * sol.sqrt()).subs(&x, &ctx.int(x0));
        assert!(r.eval_f64().unwrap().abs() < 1e-12);
    }
}

/// `y″ + y = tan x` by variation of parameters needs `∫ sin x·tan x dx`
/// and `∫ cos x·tan x dx`, which stayed unevaluated.
///
/// SymPy: `dsolve(y(x).diff(x, 2) + y(x) - tan(x), ics={y(0): 0, y'(0): 1})`
/// → `(log(sin(x) - 1)/2 - log(sin(x) + 1)/2 - I*pi/2)*cos(x) + 2*sin(x)`,
/// at x = 1/2: `N(…, 20)` = 0.50054402461654328102.
#[test]
fn variation_of_parameters_solves_y2_plus_y_equals_tan() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let ode = y.formal_diff(&x).formal_diff(&x) + &y - x.tan();
    let sol = ode
        .solve_ode_ivp(
            &y,
            &x,
            &[ic(0, ctx.int(0), ctx.int(0)), ic(1, ctx.int(0), ctx.int(1))],
        )
        .expect("solved");
    let v = sol.subs(&x, &ctx.rational(1, 2)).eval_f64().unwrap();
    assert!((v - 0.500_544_024_616_543_3).abs() < 1e-12, "{sol}: {v}");
    let residual = sol.diff(&x).diff(&x) + &sol - x.tan();
    for pt in ["1/5", "1/2", "-7/10"] {
        let r = residual.subs(&x, &p(&ctx, pt)).eval_f64().unwrap();
        assert!(r.abs() < 1e-10, "{sol}: residual {r} at {pt}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Integration: constant multiples, tan and negative trigonometric powers
// ═══════════════════════════════════════════════════════════════════════════

/// A constant factor defeated rules that `f` alone met: the trig-power rule
/// handed back `2·∫ sin/cos` (its unevaluated result with the constant
/// re-attached, which the integrator did not recognise as unevaluated),
/// and the later stages (substitutions, Risch, heurisch) saw `c·f` with a
/// symbolic `c`.  Linearity now comes first.
///
/// SymPy: `integrate(2*sin(x)/cos(x), x)` → `-2*log(cos(x))`,
/// `integrate(-sin(x)/cos(x), x)` → `log(cos(x))`, `integrate(2*cos(x)/sin(x),
/// x)` → `2*log(sin(x))` (symplex writes `ln|…|`; checked by differentiation).
#[test]
fn a_constant_factor_does_not_defeat_a_rule() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pts = ["3/10", "7/10", "-11/10"];
    for (f, want) in [
        ("2*sin(x)/cos(x)", "-2*ln(abs(cos(x)))"),
        ("-sin(x)/cos(x)", "ln(abs(cos(x)))"),
        ("2*cos(x)/sin(x)", "2*ln(abs(sin(x)))"),
        ("a*sin(x)/cos(x)", "-a*ln(abs(cos(x)))"),
    ] {
        let f = p(&ctx, f);
        let big_f = f.integrate(&x);
        assert_eq!(show(&big_f), want);
        assert_antiderivative(&ctx, &f, &big_f, &x, &pts);
    }
    // Symbolic and irrational constants, and constants the canonical form
    // merges into a power (`a·aˣ = a^(x+1)`, `√2·2ˣ = 2^(x+1/2)`): all were
    // unevaluated.  (`a`, `b` are substituted by 5/3, 2/7 for the check.)
    for f in [
        "a/cosh(x)",
        "pi*exp(x)/(exp(2*x) + 1)",
        "sqrt(2)/(exp(x) + 1)",
        "-2*b*ln(x^2 + 1)",
        "a*a^x",
        "sqrt(2)*2^x",
        "sqrt(2)*x*2^x",
        "-3/7*x^5*exp(x^3)",
        "exp(a*(x + 1))",
    ] {
        let f = p(&ctx, f);
        let big_f = f.integrate(&x);
        let bind = |e: &Ex| {
            e.subs(&ctx.symbol("a"), &ctx.rational(5, 3))
                .subs(&ctx.symbol("b"), &ctx.rational(2, 7))
        };
        assert_antiderivative(&ctx, &bind(&f), &bind(&big_f), &x, &pts);
    }
}

/// `∫ c·f = c·∫ f` over integrands the integrator handles (the hunt ran 116
/// hand-picked and 1,000 random integrands with 7 to 11 constants each:
/// before the fix 37 of the hand-picked pairs failed, after it none).
#[test]
fn constant_multiples_hunt_sample() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let fs = [
        "x*exp(x)",
        "exp(x)*sin(x)",
        "ln(x)^2",
        "1/(x^2 + 2*x + 5)",
        "sin(x)^3*cos(x)^4",
        "x*atan(x)",
        "1/sqrt(x^2 + 1)",
        "x^2*sqrt(1 - x^2)",
        "exp(x)/(1 + exp(x))",
        "1/cosh(x)",
        "sin(x)*tan(x)",
        "cos(x)^(-3)",
    ];
    let cs = ["2", "-3/7", "a", "pi", "-2*b", "sqrt(2)"];
    let bind = |e: &Ex| {
        e.subs(&ctx.symbol("a"), &ctx.rational(5, 3))
            .subs(&ctx.symbol("b"), &ctx.rational(2, 7))
    };
    for f in fs {
        let f = p(&ctx, f);
        let big_f = f.integrate(&x);
        assert!(!big_f.has_unevaluated(), "∫ {f}");
        for c in cs {
            let c = p(&ctx, c);
            let g = (&c * &f).integrate(&x);
            assert!(!g.has_unevaluated(), "∫ {c}·{f} = {g}");
            // c·∫f and ∫c·f differ by a constant.
            let d = bind(&(&g - &c * &big_f).diff(&x));
            for pt in ["3/10", "7/10"] {
                let v = d.subs(&x, &p(&ctx, pt)).eval_complex64().unwrap();
                assert!(v.norm() < 1e-9, "∫ {c}·{f} = {g}: {v} at {pt}");
            }
        }
    }
}

/// `∫ 1/tan v dv`, `∫ sin x·tan x dx`, and every `∫ sinᵐx·cosⁿx dx` with a
/// negative exponent stayed unevaluated or went through the Weierstrass
/// substitution (`∫ 1/(sin x·cos x)` came out in `tan(x/2)`).
///
/// SymPy: `integrate(1/tan(x), x)` → `log(sin(x))`, `integrate(sin(x)*tan(x),
/// x)` → `-log(sin(x) - 1)/2 + log(sin(x) + 1)/2 - sin(x)`,
/// `integrate(1/(sin(x)*cos(x)), x)` → `-log(sin(x)**2 - 1)/2 + log(sin(x))`
/// (checked by differentiation).
#[test]
fn tangents_and_negative_trigonometric_powers_integrate() {
    let ctx = Context::new();
    let (x, v) = (ctx.symbol("x"), ctx.symbol("v"));
    let big_f = v.tan().powi(-1).integrate(&v);
    assert_eq!(show(&big_f), "ln(abs(sin(v)))");
    let pts = ["3/10", "7/10", "-11/10", "13/10"];
    for f in [
        "sin(x)*tan(x)",
        "tan(x)*cos(x)",
        "sin(x)^2*tan(x)",
        "1/tan(x)^2",
        "1/tan(x)^3",
        "1/(sin(x)*cos(x))",
        "sin(x)^2/cos(x)",
        "cos(x)^2/sin(x)",
        "1/(sin(x)^2*cos(x)^2)",
        "sin(x)^(-3)*cos(x)^(-3)",
        "sin(x)^(-4)*cos(x)^(-2)",
        "sin(x)^5*cos(x)^(-7)",
        "sin(x)^(-5)*cos(x)^4",
        "tan(x)^3*cos(x)^2",
    ] {
        let f = p(&ctx, f);
        let big_f = f.integrate(&x);
        assert!(!show(&big_f).contains("tan(1/2*x)"), "∫ {f} = {big_f}");
        assert_antiderivative(&ctx, &f, &big_f, &x, &pts);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Special values
// ═══════════════════════════════════════════════════════════════════════════

/// `J_{−n}(0)` and `I_{−n}(0)` stayed unevaluated (`J_{−n} = (−1)ⁿ J_n`, 0 at
/// the origin for `n ≠ 0`), as did a negative non-integer order (a pole).
///
/// SymPy: `besselj(-3, 0)` → 0, `besselj(-2, 0)` → 0, `besselj(-1, 0)` → 0,
/// `besseli(-3, 0)` → 0, `besselj(-1/2, 0)` → zoo, `besselj(-5/2, 0)` → zoo,
/// `besseli(-1/2, 0)` → zoo, `besselj(-7/3, 0)` → zoo, `besselj(1/2, 0)` →
/// 0, `besselj(0, 0)` → 1, `besselj(-3, x).subs(x, 0)` → 0; mpmath
/// `besselj(-3, 0)` = 0.0, `besseli(-3, 0)` = 0.0.
#[test]
fn bessel_functions_of_negative_order_at_the_origin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (s, want) in [
        ("besselj(-3, 0)", "0"),
        ("besselj(-2, 0)", "0"),
        ("besselj(-1, 0)", "0"),
        ("besseli(-3, 0)", "0"),
        ("besselj(-1/2, 0)", "zoo"),
        ("besselj(-5/2, 0)", "zoo"),
        ("besseli(-1/2, 0)", "zoo"),
        ("besselj(-7/3, 0)", "zoo"),
        ("besselj(1/2, 0)", "0"),
    ] {
        assert_eq!(show(&p(&ctx, s)), want, "{s}");
    }
    assert_eq!(show(&p(&ctx, "besselj(0, 0)").eval()), "1");
    let j = p(&ctx, "besselj(-3, x)");
    assert_eq!(show(&j.subs(&x, &ctx.int(0))), "0");
}

/// `Chi(0)` was `−∞` and `Ci(0)` too; both are `γ + ln z + (entire)`, with
/// the singularity of `ln z`, whose value at 0 is `zoo` in symplex.  `Ei`
/// and `li` are real on both sides of their singular points and keep `−∞`.
///
/// SymPy: `Chi(0)` → zoo, `Ci(0)` → zoo, `log(0)` → zoo, `Ei(0)` → -oo,
/// `li(1)` → -oo, `li(0)` → 0, `Shi(0)` → 0, `(x*Chi(x)).subs(x, 0)` → nan;
/// mpmath: `chi(-1e-30)` = `(-68.5003371249198 + 3.14159265358979j)`,
/// `ci(-1e-30)` the same, `ei(-1e-30)` = `ei(1e-30)` = -68.5003371249198.
#[test]
fn chi_and_ci_at_zero_take_the_value_of_the_logarithm() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (s, want) in [
        ("Chi(0)", "zoo"),
        ("Ci(0)", "zoo"),
        ("ln(0)", "zoo"),
        ("Ei(0)", "-oo"),
        ("li(1)", "-oo"),
        ("li(0)", "0"),
        ("Shi(0)", "0"),
    ] {
        assert_eq!(show(&p(&ctx, s).eval()), want, "{s}");
    }
    assert_eq!(show(&ctx.int(0).chi()), "zoo");
    assert_eq!(show(&(&x * x.chi()).subs(&x, &ctx.int(0))), "nan");
    assert_eq!(show(&(&x * x.ci()).subs(&x, &ctx.int(0))), "nan");
}

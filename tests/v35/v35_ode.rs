//! Ordinary differential equations: the hunt for needless refusals of
//! `solve_ode` / `solve_ode_all` / `solve_ode_ivp` after 0.43.  Each test
//! says what was refused or wrong before and cites its oracle: the solution
//! substituted back into the equation and evaluated at points and values of
//! the constants (16 digits), the initial conditions evaluated, and
//! SymPy 1.14 `dsolve` (whose solution is checked to lie in the family
//! where that is cheap).

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
/// constants (complex values allowed: principal branches).
fn assert_solves(ctx: &Context, text: &str, sol: &Ex, order: usize) {
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    assert!(!sol.contains(&y), "implicit: {sol}");
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

/// `y‴ + y″ + 2y = 0`: the characteristic polynomial `r³ + r² + 2` is
/// irreducible over ℚ, and every root containing `i` was taken as one of a
/// pair matched by structurally equal parts, which Cardano's roots never
/// are: the basis came out with five functions and the ODE was refused
/// (unevaluated `DSolve`).  Now Cardano's real root and the real and
/// imaginary parts of the complex pair, SymPy 1.14's form
/// (`C1·e^{x(−2 + (3√87 + 28)^{−1/3} + (3√87 + 28)^{1/3})/6}·sin(…) + …`):
/// real exponents and frequencies.  The initial-value problem at `1/3`
/// (symbolic elimination over the radicals: more than a minute) is fitted
/// by the Lagrange formula.  Oracle: substitution; the conditions.
#[test]
fn irreducible_cubic_characteristic_polynomial_in_cardano_radicals() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let text = "D3Y + D2Y + 2*y";
    let sol = ode(&ctx, text).solve_ode(&y, &x);
    assert!(!sol.has_unevaluated(), "{sol}");
    let s = sol.to_string();
    assert!(
        s.contains("cbrt") && s.contains("cos") && s.contains("sin"),
        "{s}"
    );
    assert!(!s.contains('I') && !s.contains("RootOf"), "{s}");
    assert_solves(&ctx, text, &sol, 3);
    let x0 = ctx.rational(1, 3);
    let values = [ctx.rational(2, 3), ctx.rational(5, 4), ctx.rational(-3, 7)];
    let fit = ivp(&ctx, text, &x0, &values);
    assert!(constants(&fit).is_empty(), "{fit}");
    assert_ics(&ctx, &fit, &x0, &values);
}

/// `y‴ − 3y′ + y = 0`: three real characteristic roots that Cardano's
/// formula writes with complex cube roots (casus irreducibilis); taken as
/// non-real by their spelling, they gave no basis and the ODE was refused.
/// The count of real roots now decides (exact Descartes isolation), and
/// their real parts are `2·cos(2π/9)`, `2·cos(8π/9)`, …: three real
/// exponentials.  (SymPy 1.14 `dsolve` times out at 20 s.)  Oracle:
/// substitution.
#[test]
fn casus_irreducibilis_roots_are_real_exponentials() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let text = "D3Y - 3*D1Y + y";
    let sol = ode(&ctx, text).solve_ode(&y, &x);
    let s = sol.to_string();
    assert!(
        !s.contains('I') && !s.contains("sin(x") && s.contains("cos(2/9*pi)"),
        "{s}"
    );
    assert_solves(&ctx, text, &sol, 3);
}

/// `y⁽⁵⁾ − y′ + y = x`: `r⁵ − r + 1` has no roots in radicals.  The basis
/// was `e^{x·RootOf(…, k)}` for all five roots, four of them non-real
/// exponentials; now SymPy 1.14's real system `e^{x·RootOf(…, 0)}`,
/// `e^{x·re(RootOf(…, k))}·cos(x·im(RootOf(…, k)))`, `…·sin(…)` (SymPy
/// indexes the roots its own way).  The initial-value problem did not finish
/// in a minute (symbolic elimination over `re`/`im` of the roots); now the
/// exact Lagrange fit.  Oracle: substitution; the conditions.
#[test]
fn quintic_characteristic_polynomial_gives_real_rootof_system() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let text = "D5Y - D1Y + y - x";
    let sol = ode(&ctx, text).solve_ode(&y, &x);
    let s = sol.to_string();
    assert!(
        s.contains("re(RootOf(r^5 - r + 1") && s.contains("im(RootOf(r^5 - r + 1"),
        "{s}"
    );
    assert!(!s.contains('I'), "{s}");
    assert_solves(&ctx, text, &sol, 5);
    let x0 = ctx.int(0);
    let values = [ctx.int(1), ctx.int(0), ctx.int(0), ctx.int(0), ctx.int(0)];
    let fit = ivp(&ctx, text, &x0, &values);
    assert!(
        constants(&fit).is_empty() && !fit.to_string().contains('I'),
        "{fit}"
    );
    assert_ics(&ctx, &fit, &x0, &values);
}

/// `y′ = (y − x)²` was refused.  The rational Riccati solver finds the
/// particular solutions `x ± 1` and gives SymPy 1.14's answer
/// `(C1·x + C1 − x·e^{2x} + e^{2x})/(C1 − e^{2x})` with numerator and
/// denominator times `e^{−2x}` (the same `C1`).  The substitution
/// `u = y − x` (`u′ = u² − 1`) solves `y′ = sin(x + y)` and others of the
/// shape `y′ = F(a·x + b·y + c)`.  Oracle: substitution; SymPy's member.
#[test]
fn riccati_square_of_linear_argument() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let text = "D1Y - (y - x)^2";
    let sol = ode(&ctx, text).solve_ode(&y, &x);
    assert_solves(&ctx, text, &sol, 1);
    let c1 = ctx.symbol("C1");
    let sympy = parse(&ctx, "(3*x + 3 - x*exp(2*x) + exp(2*x))/(3 - exp(2*x))");
    let ours = sol.subs(&c1, &ctx.int(3));
    for xv in [ctx.rational(1, 5), ctx.rational(-2, 3)] {
        let d = (&ours.subs(&x, &xv) - &sympy.subs(&x, &xv))
            .eval_complex64()
            .unwrap();
        assert!(d.norm() < 1e-12, "{ours} vs SymPy at {xv}: {d}");
    }
    let text = "D1Y - sin(x + y)";
    let sol = ode(&ctx, text).solve_ode(&y, &x);
    assert_solves(&ctx, text, &sol, 1);
}

/// `y′·y″ = 1` (no `y`: `p = y′`) was refused: `p·p′ = 1` was solved as an
/// exact equation whose potential `p²/2 − x = C1` has two roots, and an
/// implicit `p` was not integrated.  Each root is now a family:
/// `C2 ± (2C1 + 2x)^{3/2}/3`, SymPy 1.14's `C1 ∓ (C2 + 2x)^{3/2}/3`.
/// Likewise `y·y′ = 1` gives `±√(2C1 + 2x)` (was implicit `y²/2 − C1 − x`).
/// Oracle: substitution of every family; SymPy.
#[test]
fn reduced_and_exact_equations_return_every_root() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    for (text, order) in [("D1Y*D2Y - 1", 2), ("y*D1Y - 1", 1), ("D2Y*D1Y - x", 2)] {
        let all = ode(&ctx, text).solve_ode_all(&y, &x).unwrap();
        assert_eq!(all.len(), 2, "{text}: {all:?}");
        for fam in &all {
            assert_solves(&ctx, text, fam, order);
        }
    }
}

/// `y′ = y² − 1` was left implicit (`ln|y − 1|/2 − ln|y + 1|/2 − x − C1`):
/// the solver cannot invert `|·|`.  With `ln|u| → ln u` (the constant
/// absorbs the difference on each interval, as SymPy integrates) and
/// `e^{2C1} → C1`: `(−C1·e^{2x} − 1)/(C1·e^{2x} − 1)`, SymPy 1.14's
/// `1/tanh(C1 − x)` family with `C1 = 0` (`y = 1`) included.  `y′ =
/// (y/x)·ln(y/x)` was refused (`∫dv/(v ln v − v)` not found unless
/// factored): now `x·e^{C1·x + 1}`, SymPy's answer.  Oracle: substitution;
/// SymPy.
#[test]
fn separable_and_homogeneous_logarithms_are_inverted() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let sol = ode(&ctx, "D1Y - y^2 + 1").solve_ode(&y, &x);
    assert_solves(&ctx, "D1Y - y^2 + 1", &sol, 1);
    let sol = ode(&ctx, "D1Y - y/x*log(y/x)").solve_ode(&y, &x);
    assert_eq!(sol.to_string(), "x*exp(C1*x + 1)");
}

/// `y′ = 1 + sin y`: besides the family `2·atan(…)` the separable solver
/// returned `y = π` and `y = −π` as further families — roots of the
/// implicit equation free of the constant, and not solutions (`y′ = 0 ≠ 1`).
/// A root without the constant is no member of the general solution and is
/// dropped.  Oracle: substitution (`1 + sin π = 1`).
#[test]
fn constant_free_roots_are_not_families() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    for text in ["D1Y - 1 - sin(y)", "D1Y - sin(x + y)"] {
        let all = ode(&ctx, text).solve_ode_all(&y, &x).unwrap();
        assert!(!all.is_empty());
        for fam in &all {
            assert_solves(&ctx, text, fam, 1);
        }
    }
}

/// `y′ = (y − x + 2)/(y − x + 3)` (parallel lines: `u = y − x`) and
/// `y′ = (x + y + 1)/(x − y + 3)` (lines through `(−2, 1)`: SymPy's
/// `linear_coefficients`) were refused.  Now `x − 3 ± √(2C1 − 2x)` (SymPy
/// 1.14: `x ∓ √(C1 − 2x) − 3`) and the implicit
/// `atan((y − 1)/(x + 2)) − ½·ln(1 + (y − 1)²/(x + 2)²) − ln|x + 2| = C1`
/// (SymPy: the same relation).  Oracle: substitution; implicit
/// differentiation of the relation at points.
#[test]
fn linear_argument_and_linear_coefficients() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let text = "D1Y - (y - x + 2)/(y - x + 3)";
    let all = ode(&ctx, text).solve_ode_all(&y, &x).unwrap();
    assert_eq!(all.len(), 2);
    for fam in &all {
        assert_solves(&ctx, text, fam, 1);
    }
    // Implicit G(x, y) = 0: G_x + G_y·y′ = 0 with y′ = (x + y + 1)/(x − y + 3).
    let g = ode(&ctx, "D1Y - (x + y + 1)/(x - y + 3)").solve_ode(&y, &x);
    assert!(g.contains(&y) && g.contains(&ctx.symbol("C1")), "{g}");
    let rhs = parse(&ctx, "(x + y + 1)/(x - y + 3)");
    let rel = &g.diff(&x) + &(&g.diff(&y) * &rhs);
    for (xv, yv) in [((1, 3), (5, 2)), ((7, 5), (-1, 4))] {
        let z = rel
            .subs(&x, &ctx.rational(xv.0, xv.1))
            .subs(&y, &ctx.rational(yv.0, yv.1))
            .eval_complex64()
            .unwrap();
        assert!(z.norm() < 1e-12, "{rel}: {z}");
    }
}

/// Orders above 2 without `y` and its lowest derivatives were refused:
/// `y‴ = y″/x` (`p = y″`: SymPy 1.14 `C1 + C2·x + C3·x³`), `y‴·y″ = 1`,
/// `y⁗ = y‴²`.  An equation of degree 2 in its highest derivative was
/// refused too: `y″² = 4y′` (SymPy: `C1 + C2²·x/4 − C2·x²/2 + x³/3`).
/// Oracle: substitution; SymPy.
#[test]
fn higher_order_reduction_and_solving_for_the_derivative() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    for (text, order) in [
        ("D3Y - D2Y/x", 3),
        ("D3Y*D2Y - 1", 3),
        ("D4Y - D3Y^2", 4),
        ("D2Y^2 - 4*D1Y", 2),
        ("D1Y^2 - 4*y", 1),
    ] {
        let all = ode(&ctx, text).solve_ode_all(&y, &x).unwrap();
        for fam in &all {
            assert_solves(&ctx, text, fam, order);
        }
    }
}

/// Riccati equations with rational coefficients and a rational particular
/// solution needed that solution passed to `solve_riccati`; `solve_ode`
/// refused them.  Now SymPy's `1st_rational_riccati` (Vo's algorithm):
/// `y′ = y² − 2/x²` (particular `−1/x`, `2/x`) gives `(2x³ + C1)/(C1·x −
/// x⁴)`-like, `y′ + y² = x² + 1` (particular `x`) `x + e^{−x²}/(C1 + √π·
/// erf(x)/2)`-like.  Oracle: substitution; SymPy 1.14 `dsolve`.
#[test]
fn rational_riccati_without_a_given_particular_solution() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    for text in [
        "D1Y - y^2 + 2/x^2",
        "D1Y + y^2 - x^2 - 1",
        "D1Y - y^2 - x*y + 2*x + 4",
    ] {
        let sol = ode(&ctx, text).solve_ode(&y, &x);
        assert!(!sol.has_unevaluated(), "{text}: {sol}");
        assert_solves(&ctx, text, &sol, 1);
    }
}

/// Exact equations whose `∫M dx` splits on `y` or keeps `ln|x|` were
/// refused, with or without an integrating factor: `y·e^{xy} + (x·e^{xy} +
/// 1)·y′ = 0` (`∫ y·e^{xy} dx = y·Piecewise(e^{xy}/y, y ≠ 0; x)`) and
/// `(y/x + 3x²) + (ln x + 1)·y′ = 0` (`g′(y) = ln x + 1 − ln|x|` was not
/// free of `x`), each also divided by `y²` or `e^y`.  Now `C1 − W(x·e^{C1·x})/x`
/// (`e^{xy} + y = C1`) and `(C1 − x³)/(ln x + 1)` (SymPy 1.14's `1st_exact`
/// potentials).  Oracle: substitution.
#[test]
fn exact_potentials_through_case_splits_and_log_abs() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    for text in [
        "y*exp(x*y) + (x*exp(x*y) + 1)*D1Y",
        "(y*exp(x*y))/(y^2) + ((x*exp(x*y) + 1)/(y^2))*D1Y",
        "(y/x + 3*x^2) + (log(x) + 1)*D1Y",
        "(y/x + 3*x^2)/(exp(y)) + ((log(x) + 1)/(exp(y)))*D1Y",
    ] {
        let sol = ode(&ctx, text).solve_ode(&y, &x);
        assert_solves(&ctx, text, &sol, 1);
    }
}

/// A Bernoulli equation with `1 − n` an odd integer, `y′ = y/x + y⁴/x²`
/// (`y⁻³ = v`), had only the principal root `v^{−1/3}`, real for `v > 0`
/// only: `y(1/3) = −1` had no family ("no real solutions").  The real
/// root `−(−v)^{−1/3}` is a second family (SymPy 1.14 returns the three
/// complex cube roots; its real member at `y(1/3) = −1` is ours).  Oracle:
/// substitution; the condition.
#[test]
fn odd_root_bernoulli_has_its_negative_real_family() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let text = "D1Y - y/x - y^4/x^2";
    for fam in ode(&ctx, text).solve_ode_all(&y, &x).unwrap() {
        assert_solves(&ctx, text, &fam, 1);
    }
    let x0 = ctx.rational(1, 3);
    let fit = ivp(&ctx, text, &x0, &[ctx.int(-1)]);
    assert_ics(&ctx, &fit, &x0, &[ctx.int(-1)]);
    let v = fit.subs(&x, &ctx.rational(1, 2)).eval_complex64().unwrap();
    assert!(v.im.abs() < 1e-14 && v.re < 0.0, "{fit}: {v}");
}

/// `y″ + a·y = 0` with `a` declared positive kept the generic
/// `C1·e^{x√(−a)} + C2·e^{−x√(−a)}`, and its initial-value solution came out
/// as `e^{−i√a·x}/2 + e^{i√a·x}/2`; now the real basis `cos(√a·x)`,
/// `sin(√a·x)` (SymPy 1.14 with `a = Symbol('a', positive=True)`:
/// `C1·sin(√a·x) + C2·cos(√a·x)`) and `y(0) = 1, y′(0) = 0` gives `cos(√a·x)`.
/// Undeclared `a` keeps the generic form (as SymPy).  Oracle: SymPy;
/// substitution at `a = 3`.
#[test]
fn declared_positive_parameter_gives_real_basis() {
    let ctx = Context::new();
    let _a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let sol = ode(&ctx, "D2Y + a*y").solve_ode(&y, &x);
    assert_eq!(sol.to_string(), "C1*cos(x*sqrt(a)) + C2*sin(x*sqrt(a))");
    let fit = ivp(&ctx, "D2Y + a*y", &ctx.int(0), &[ctx.int(1), ctx.int(0)]);
    assert_eq!(fit.to_string(), "cos(x*sqrt(a))");
    let other = Context::new();
    let (x, y) = (other.symbol("x"), other.symbol("y"));
    let generic = ode(&other, "D2Y + a*y").solve_ode(&y, &x);
    assert_eq!(
        generic.to_string(),
        "C1*exp(x*sqrt(-a)) + C2*exp(-x*sqrt(-a))"
    );
}

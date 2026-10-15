//! Tests of session 15 (s15g2): solving with parameters, conditional roots.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

fn is_no_solution<T: std::fmt::Debug>(r: &Result<T, SymplexError>) -> bool {
    matches!(r, Err(SymplexError::NoSolution { .. }))
}

fn is_refused<T: std::fmt::Debug>(r: &Result<T, SymplexError>) -> bool {
    matches!(r, Err(SymplexError::ComputationFailed { .. }))
}

fn roots(e: &Ex, v: &Ex) -> Vec<String> {
    e.solve(v)
        .unwrap_or_else(|err| panic!("{e}: {err}"))
        .iter()
        .map(|r| r.to_string())
        .collect()
}

/// D6.  `√x = y` has the root `y²` only when `y` is in the range of the
/// principal square root, `re y > 0 ∨ (re y = 0 ∧ im y ≥ 0)` (for a real
/// `y`: `y ≥ 0`), and none otherwise.  `solve` refused since 0.51 (0.50
/// and SymPy 1.14 `solve(sqrt(x) - y, x)` give `[y**2]`, wrong at `y = −1`:
/// `√1 = 1`) and `solve_as_set` gave the `ConditionSet` of the equation;
/// now the set is the root under its exact condition, and `solve` still
/// refuses unless the assumptions decide (exact: `√(y²) = y` iff `re y > 0`
/// or `y = it`, `t ≥ 0`).
#[test]
fn principal_square_root_with_a_parameter_is_a_conditional_set() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = parse(&ctx, "sqrt(x) - y");
    assert!(is_refused(&e.solve(&x)));
    assert_eq!(
        e.solve_as_set(&x).to_string(),
        "ConditionSet(x, re(y) > 0 | im(y) >= 0 & re(y) == 0) ∩ {y^2}"
    );

    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol_with("y", &[Assumption::Real]).unwrap();
    let e = parse(&ctx, "sqrt(x) - y");
    let r = e.solve(&x);
    assert!(is_refused(&r), "{r:?}");
    let set = e.solve_as_set(&x);
    assert_eq!(set.to_string(), "ConditionSet(x, y >= 0) ∩ {y^2}");
    // the condition decides once `y` has a value: `{9}` at `y = 3`, `∅` at
    // `y = −1` (`√1 ≠ −1`)
    assert_eq!(set.subs(&y, &ctx.int(3)).simplify().to_string(), "{9}");
    assert_eq!(
        set.subs(&y, &ctx.int(-1)).simplify().to_string(),
        "EmptySet"
    );
    // membership is undecided while the condition is (`y²` is a member iff
    // `y ≥ 0`); JSON keeps the set
    assert_eq!(set.contains(&parse(&ctx, "y^2")), None);
    let AnyExpr::Set(back) = ctx.from_json_any(&set.to_json().unwrap()).unwrap() else {
        panic!("a set");
    };
    assert_eq!(back, set);

    for (decl, want) in [
        (Assumption::Positive, "y^2"),
        (Assumption::NonNegative, "y^2"),
    ] {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let _y = ctx.symbol_with("y", &[decl]).unwrap();
        assert_eq!(roots(&parse(&ctx, "sqrt(x) - y"), &x), [want]);
    }
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let _y = ctx.symbol_with("y", &[Assumption::Negative]).unwrap();
    assert!(is_no_solution(&parse(&ctx, "sqrt(x) - y").solve(&x)));
}

/// D6 for other principal powers: `x^(1/3) = y` is `y³` for `−π/3 < arg y ≤
/// π/3` or `y = 0` (a real `y`: `y ≥ 0`; `(−8)^(1/3) = 1 + i√3`, not `−2`),
/// `x^(−1/2) = y` is `y^(−2)` for a real `y > 0`, and a root `t₀` of the
/// equation in `t = x^(1/L)` is kept under the same condition: `√x + y + 4 =
/// 0` is `(y + 4)²` when `−y − 4 ≥ 0`.  Up to 0.52 all were refused with
/// the `ConditionSet` of the equation.  Oracle: the principal branch, exact.
#[test]
fn principal_powers_and_radical_substitutions_carry_conditions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(
        parse(&ctx, "x^(1/3) - y").solve_as_set(&x).to_string(),
        "ConditionSet(x, y == 0 | arg(y) > -1/3*pi & 1/3*pi >= arg(y)) ∩ {y^3}"
    );
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let _y = ctx.symbol_with("y", &[Assumption::Real]).unwrap();
    for (src, want) in [
        ("x^(1/3) - y", "ConditionSet(x, y >= 0) ∩ {y^3}"),
        ("x^(-1/2) - y", "ConditionSet(x, y > 0) ∩ {y^(-2)}"),
        ("sqrt(x + 1) - y", "ConditionSet(x, y >= 0) ∩ {y^2 - 1}"),
        (
            "sqrt(x) + y + 4",
            "ConditionSet(x, -y - 4 >= 0) ∩ {(-y - 4)^2}",
        ),
    ] {
        let e = parse(&ctx, src);
        assert!(is_refused(&e.solve(&x)), "{src}");
        assert_eq!(e.solve_as_set(&x).to_string(), want, "{src}");
    }
}

/// Conditional roots in products and quotients: `(√x − y)(x − 2) = 0` for a
/// real `y` is `{2} ∪ ({y²} if y ≥ 0)`; `(√x − y)/(x − y²)` has no root at
/// all (its only candidate `y²` is a pole for every `y`: `0/0`).  Up to
/// 0.52 both were refused.  Exact.
#[test]
fn conditional_roots_beside_other_factors_and_poles() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let _y = ctx.symbol_with("y", &[Assumption::Real]).unwrap();
    let e = parse(&ctx, "(sqrt(x) - y)*(x - 2)");
    assert!(is_refused(&e.solve(&x)));
    assert_eq!(
        e.solve_as_set(&x).to_string(),
        "{2} ∪ (ConditionSet(x, y >= 0) ∩ {y^2})"
    );
    let q = parse(&ctx, "(sqrt(x) - y)/(x - y^2)");
    assert!(is_no_solution(&q.solve(&x)));
    assert_eq!(q.solve_as_set(&x).to_string(), "EmptySet");
    // `x·∛(x^y)`: the root `0` of the factor `x` is a point of the other
    // factor's domain only for `re y > 0` (`0^y` is undefined otherwise),
    // so `0` is not claimed for every `y` (an intermediate state gave
    // `{0} ∪ (ConditionSet(x, re(y) > 0) ∩ {0})`)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let set = parse(&ctx, "x*cbrt(x^y)").solve_as_set(&x).to_string();
    assert!(set.starts_with("ConditionSet(x, "), "{set}");
}

/// D8.  For a real unknown, a root with real parameters that is real only
/// for some of their values was returned: `x² = y` gave `[√y, −√y]`
/// (non-real for `y < 0`), `x + √b − 1 = 0` gave `[1 − √b]`, and the radical
/// hunt of wave 6 had 829 such solves (`3·(x² + y)^(1/6) = y^(3/2)` for a
/// positive `y`: `±√(y⁹/729 − y)`, non-real for `y = 1`: mpmath
/// `sqrt(mpf(1)/729 - 1)` = `0.9993…j`).  `solve` now refuses unless the
/// assumptions decide the realness, and `solve_as_set` gives the roots under
/// the exact condition (`A + C·u^(p/q)`, `A`, `C`, `u` real, `C ≠ 0`: real iff
/// `u ≥ 0`), else `ℝ ∩ {…}` as before.
#[test]
fn real_unknown_roots_real_only_for_some_parameters() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    let _y = ctx.symbol_with("y", &[Assumption::Real]).unwrap();
    let _b = ctx.symbol_with("b", &[Assumption::Real]).unwrap();
    for (src, want) in [
        ("x^2 - y", "ConditionSet(x, y >= 0) ∩ {sqrt(y), -sqrt(y)}"),
        (
            "x + sqrt(b) - 1",
            "ConditionSet(x, b >= 0) ∩ {-sqrt(b) + 1}",
        ),
        (
            "x^2 - 2*x + y",
            "ConditionSet(x, -4*y + 4 >= 0) ∩ {-1/2*sqrt(-4*y + 4) + 1, 1/2*sqrt(-4*y + 4) + 1}",
        ),
    ] {
        let e = parse(&ctx, src);
        assert!(is_refused(&e.solve(&x)), "{src}");
        assert_eq!(e.solve_as_set(&x).to_string(), want, "{src}");
    }
    // decided by the assumptions: kept
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    let _y = ctx.symbol_with("y", &[Assumption::NonNegative]).unwrap();
    assert_eq!(roots(&parse(&ctx, "x^2 - y"), &x), ["sqrt(y)", "-sqrt(y)"]);
    // the hunt's example
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    let _y = ctx.symbol_with("y", &[Assumption::Positive]).unwrap();
    let e = parse(&ctx, "3*(x^2 + y)^(1/6) - y^(3/2)");
    assert!(is_refused(&e.solve(&x)));
    assert_eq!(
        e.solve_as_set(&x).to_string(),
        "ConditionSet(x, 1/729*y^9 - y >= 0) ∩ {sqrt(1/729*y^9 - y), -sqrt(1/729*y^9 - y)}"
    );
    // a plain parameter and a positive unknown: in the domain only for some
    // values (`x − c`: `c > 0`), refused; a nonzero unknown fails only at
    // `c = 0` (measure zero, the generic convention): kept
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    assert!(is_refused(&parse(&ctx, "x - c").solve(&x)));
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::NonZero]).unwrap();
    assert_eq!(roots(&parse(&ctx, "x - c"), &x), ["c"]);
}

/// `δ(i)·x = 0` gave `[0]` (the constant factor was taken nonzero; `δ` has
/// a value only at a real argument, so the product has none anywhere): now
/// refused.  Oracle: the definition (SymPy 1.14 `DiracDelta(I)` raises
/// "argument must be real").
#[test]
fn a_constant_factor_without_a_value_is_refused() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = parse(&ctx, "DiracDelta(I)*x").solve(&x);
    assert!(is_refused(&r), "{r:?}");
    assert!(
        parse(&ctx, "DiracDelta(I)*x")
            .solve_as_set(&x)
            .to_string()
            .starts_with("ConditionSet"),
    );
    assert_eq!(roots(&parse(&ctx, "pi*x - 1"), &x), ["1/pi"]);
}

/// `cos(3x + 1)² + cos(3x + 1) − 6 = 0` for a real `x` was refused ("found
/// the roots 2·atan(RootOf(…)) − 1/3 … but not necessarily all"), `cos(x)² +
/// cos(x) − 6` too: in `t = cos(…)` the roots are `2` and `−3`, outside
/// `[−1, 1]` (exact: `t² + t − 6 = (t − 2)(t + 3)`).  Now "no solution"; for
/// a plain `x` the principal complex roots are unchanged.
#[test]
fn quadratic_in_a_cosine_outside_its_range_has_no_real_root() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    for src in [
        "cos(3*x + 1)^2 + cos(3*x + 1) - 6",
        "cos(x)^2 + cos(x) - 6",
        "sin(x)^2 + sin(x) - 6",
    ] {
        let e = parse(&ctx, src);
        assert!(is_no_solution(&e.solve(&x)), "{src}");
        assert_eq!(e.solve_as_set(&x).to_string(), "EmptySet", "{src}");
    }
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(
        roots(&parse(&ctx, "cos(3*x + 1)^2 + cos(3*x + 1) - 6"), &x),
        [
            "1/3*acos(2) - 1/3",
            "-1/3*acos(2) - 1/3",
            "1/3*acos(-3) - 1/3",
            "-1/3*acos(-3) - 1/3"
        ]
    );
}

/// `f^c = 0` for an exponent `c` that is not a number: the candidate
/// `0^(1/c)` was left unevaluated (`x^(1/erfi 3) = 0`: "cannot decide
/// whether 0^(1/erfi(3)) solves the equation"), and `x^c = 0` was refused
/// for a positive `c`.  `0^c` is 0 exactly for `re c > 0` (undefined or 1
/// otherwise), and `f^c = e^(c·ln f) ≠ 0` for `f ≠ 0`: `[0]` for `c =
/// 1/erfi(3)` (`erfi(3) ≈ 1629.99`, mpmath `erfi(3)`), none for `−1/erfi(3)`,
/// `[0]` for a positive `c`, and `{0}` when `re c > 0` for a plain `c`.
#[test]
fn a_power_with_a_symbolic_exponent_equal_to_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(roots(&parse(&ctx, "x^(1/erfi(3))"), &x), ["0"]);
    assert!(is_no_solution(&parse(&ctx, "x^(-1/erfi(3))").solve(&x)));
    let e = parse(&ctx, "x^c");
    assert!(is_refused(&e.solve(&x)));
    assert_eq!(
        e.solve_as_set(&x).to_string(),
        "ConditionSet(x, re(c) > 0) ∩ {0}"
    );
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let _c = ctx.symbol_with("c", &[Assumption::Positive]).unwrap();
    assert_eq!(roots(&parse(&ctx, "x^c"), &x), ["0"]);
}

/// The reason of a range-restriction "no solution" said "no real
/// solutions" also for a complex unknown (`e^x = 0` has no solution in ℂ;
/// `√x = y` for a negative `y` neither).  Now it says what is true.
#[test]
fn no_solution_reason_for_a_complex_unknown() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let _y = ctx.symbol_with("y", &[Assumption::Negative]).unwrap();
    for src in ["sqrt(x) - y", "exp(x)"] {
        match parse(&ctx, src).solve(&x) {
            Err(SymplexError::NoSolution { reason, .. }) => {
                assert!(!reason.contains("real"), "{src}: {reason}");
            }
            other => panic!("{src}: {other:?}"),
        }
    }
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    assert!(is_no_solution(&parse(&ctx, "sin(x) - 2").solve(&x)));
}

/// The unknown inside Lambert W and outside it was refused, so initial
/// values on Lambert-W families of ODEs could not be fitted.  With `e^{W(z)}
/// = z/W(z)`: `C1·e^{W(2/C1)} = 2` is `2/W(2/C1) = 2`, `C1 = 2/e`; `W(z)·e^{W(z)}
/// = z`; and `W(f) = g` gives `f = g·e^g`: `C1 − W(e^{C1}) = 3` is
/// `e^{C1}·e^{3 − C1} = C1 − 3`, `C1 = 3 + e³` (check: `W(e^{3 + e³}) = e³`,
/// as `e³·e^{e³} = e^{3 + e³}`).  The `W₋₁` twin `C1·e^{W₋₁(2/C1)} = 2` has
/// no solution (`W₋₁` never takes the value 1).
#[test]
fn lambert_w_of_the_unknown_with_the_unknown_outside() {
    let ctx = Context::new();
    let c1 = ctx.symbol("C1");
    assert_eq!(
        roots(&parse(&ctx, "C1*exp(lambertw(2/C1)) - 2"), &c1),
        ["2*exp(-1)"]
    );
    assert_eq!(
        roots(&parse(&ctx, "C1 - lambertw(exp(C1)) - 3"), &c1),
        ["exp(3) + 3"]
    );
    assert!(is_no_solution(
        &parse(&ctx, "C1*exp(lambertw(2/C1, -1)) - 2").solve(&c1)
    ));
    let x = ctx.symbol("x");
    assert_eq!(
        roots(&parse(&ctx, "lambertw(x)*exp(lambertw(x)) - 2"), &x),
        ["2"]
    );
    assert_eq!(
        roots(&parse(&ctx, "exp(lambertw(2*x)) - x"), &x),
        ["exp(2)"]
    );
    // `W(x) = −2x` gives `x·e^{2x} = −2x`, whose candidate `z = (ln 2 −
    // iπ)/2` solves it on another branch: mpmath `lambertw(z, 1)` =
    // `−0.693… + 3.1416…j` = `−2z`, but `lambertw(z)` = `0.6244… −
    // 0.5936…j`, so `W(z) + 2z = 1.3176… − 3.7352…j ≠ 0`.  Beside a factor
    // with a parameter, where the final check cannot evaluate, it was kept
    // (`[(ln 2 − iπ)/2, 0]`); only `0` solves it (`W(0) = 0`).
    assert_eq!(roots(&parse(&ctx, "(lambertw(x) + 2*x)*y"), &x), ["0"]);
}

/// Principal candidates that fail are no proof of "no solution" when the
/// equation has other branches (an intermediate state of this change
/// claimed it; 0.52 refused both, as now).  mpmath `findroot`: `tan(x^y)/C(x)`
/// at `y = 2` vanishes at `x = √π` (`tan π = 0`, `C(√π) ≈ 0.3999`), and
/// `W₋₁(x) + x` at `x = iπ` (`lambertw(pi*1j, -1)` = `-pi*1j`), a member of
/// `x·(eˣ + 1) = 0` beside the failing principal `0`, `iπ`.
#[test]
fn failing_principal_candidates_prove_nothing() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in ["tan(x^y)/fresnelc(x)", "lambertw(x, -1) + x"] {
        let r = parse(&ctx, src).solve(&x);
        assert!(!is_no_solution(&r), "{src}: {r:?}");
    }
}

/// The IVPs of wave 7 (F5 R2b and the exact-equation family) that could
/// not be fitted: `y′ = (3y − x)/(x + y)`, `y(1) = 3` on `C1·e^{W(2x/C1)} +
/// x`, and `y·e^{xy} + (x·e^{xy} + 1)·y′ = 0`, `y(1) = 3` on `C1 −
/// W(x·e^{C1·x})/x`.  Oracle: mpmath `odefun` at 30 and 60 digits, `y(2) =
/// 4.909466435122129402929947126` and `1.535195822384023237875943935`.
#[test]
fn lambert_w_initial_values_are_fitted() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let ic = [InitialCondition {
        order: 0,
        x: ctx.int(1),
        value: ctx.int(3),
    }];
    let homog = &y.formal_diff(&x) - &parse(&ctx, "(3*y - x)/(x + y)");
    let exact =
        &(&y * &parse(&ctx, "exp(x*y)")) + &(&parse(&ctx, "x*exp(x*y) + 1") * &y.formal_diff(&x));
    for (ode, at2) in [
        (&homog, 4.909_466_435_122_129_f64),
        (&exact, 1.535_195_822_384_023_f64),
    ] {
        let sol = ode.solve_ode_ivp(&y, &x, &ic).unwrap();
        assert!(ode.check_ode_solution(&sol, &y, &x), "{sol}");
        let v1 = sol.subs(&x, &ctx.int(1)).eval_f64().unwrap();
        assert!((v1 - 3.0).abs() < 1e-13, "{sol}: {v1}");
        let v2 = sol.subs(&x, &ctx.int(2)).eval_f64().unwrap();
        assert!((v2 - at2).abs() < 1e-12, "{sol}: {v2}");
    }
}

/// The principal inverse functions peeled with a parameter on the right:
/// `ln x = y` gave `[e^y]` for a plain `y` (wrong for `y = 4i`: `ln(e^{4i})
/// = (4 − 2π)i`), `asin x = y` for a real `y` gave `[sin y]` (wrong for `y =
/// 4`: `asin(sin 4) = π − 4`), `W(2t) = y` gave `[y·e^y/2]` (wrong for `y =
/// −5`).  Now the root comes under the exact range condition of the
/// branch, or `solve` refuses; with a real `y`, `ln x = y` is `[e^y]`.
/// SymPy 1.14 gives the unconditional roots.
#[test]
fn principal_inverses_with_a_parameter_carry_their_range() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = parse(&ctx, "ln(x) - y");
    assert!(is_refused(&e.solve(&x)));
    assert_eq!(
        e.solve_as_set(&x).to_string(),
        "ConditionSet(x, im(y) > -pi & pi >= im(y)) ∩ {exp(y)}"
    );
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let _y = ctx.symbol_with("y", &[Assumption::Real]).unwrap();
    assert_eq!(roots(&parse(&ctx, "ln(x) - y"), &x), ["exp(y)"]);
    let e = parse(&ctx, "asin(x) - y");
    assert!(is_refused(&e.solve(&x)));
    assert_eq!(
        e.solve_as_set(&x).to_string(),
        "ConditionSet(x, y >= -1/2*pi & 1/2*pi >= y) ∩ {sin(y)}"
    );
    assert_eq!(roots(&parse(&ctx, "asinh(x) - y"), &x), ["sinh(y)"]);
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let _y = ctx.symbol_with("y", &[Assumption::Positive]).unwrap();
    assert_eq!(roots(&parse(&ctx, "acosh(x) - y"), &x), ["cosh(y)"]);
}

//! Tests of session 15 (s15j1): solve: the wave-9 leftovers.
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

fn is_every_value<T: std::fmt::Debug>(r: &Result<T, SymplexError>) -> bool {
    matches!(r, Err(SymplexError::InfiniteSolutions { .. }))
}

fn roots(e: &Ex, v: &Ex) -> Vec<String> {
    e.solve(v)
        .unwrap_or_else(|err| panic!("{e}: {err}"))
        .iter()
        .map(|r| r.to_string())
        .collect()
}

/// A root holding an infinity inside a function that `eval` leaves alone
/// is no finite value.  Up to 0.54 `solve(x + chebyshev_t(2, y·∞), x)` was
/// `[−chebyshev_t(2, y·∞)]` (0.53 said "no solution" only because `eval`
/// expanded `T₂`), `x + sin ∞` gave `[−sin ∞]`, `x + Γ(y·∞)` `[−Γ(y·∞)]`,
/// `x² − sin ∞` `[±√(sin ∞)]`.  Oracle (exact): `T₂(t) = 2t² − 1`, so
/// `T₂(y·∞) = +∞` for every real `y ≠ 0` and `0·∞` is undefined at `y = 0`;
/// `sin t` has no limit at `∞`; `Γ(−∞)` has none either (poles at the
/// negative integers).  Now refused, with a `ConditionSet`; the values
/// `eval` folds still solve (`atan ∞ = π/2`; `e^{−y·∞} = 0` for `y > 0`).
#[test]
fn a_root_with_an_infinity_inside_a_function_is_refused() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in [
        "x + chebyshev_t(2, y*oo)",
        "x + sin(oo)",
        "x + gamma(y*oo)",
        "x^2 - sin(oo)",
        "x - besselj(1, y*oo)",
        "exp(x) - gamma(y*oo)",
    ] {
        let e = parse(&ctx, src);
        let r = e.solve(&x);
        assert!(is_refused(&r), "{src}: {r:?}");
        let set = e.solve_as_set(&x).to_string();
        assert!(set.starts_with("ConditionSet"), "{src}: {set}");
    }
    assert_eq!(roots(&parse(&ctx, "x + atan(oo)"), &x), ["-1/2*pi"]);
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let _ = ctx.symbol_with("y", &[Assumption::Positive]).unwrap();
    assert_eq!(roots(&parse(&ctx, "x + exp(-y*oo)"), &x), ["0"]);
}

/// An equation free of the unknown whose constant holds an infinity
/// inside a function is no generic constant: `e^{−y·∞}` is 0 for every `y >
/// 0` (every `x` solves `e^{−y·∞} = 0` there) and `∞` for `y < 0`.  Up to
/// 0.54 `solve(exp(−y·∞), x)` and `solve(chebyshev_t(2, y·∞), x)` were "no
/// solution" (`solve_as_set`: `EmptySet`) for a plain `y`.  Oracle (exact):
/// `lim_{t→∞} e^{−yt} = 0` for `y > 0`.  Now refused; for a positive `y`
/// every value (unchanged).
#[test]
fn a_constant_equation_with_an_infinity_inside_a_function_is_refused() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in ["exp(-y*oo)", "chebyshev_t(2, y*oo)"] {
        let r = parse(&ctx, src).solve(&x);
        assert!(is_refused(&r), "{src}: {r:?}");
    }
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let _ = ctx.symbol_with("y", &[Assumption::Positive]).unwrap();
    assert!(is_every_value(&parse(&ctx, "exp(-y*oo)").solve(&x)));
    // an infinite term: infinite or undefined for every `y`, never 0
    assert!(is_no_solution(&parse(&ctx, "x - y*zoo").solve(&x)));
}

/// A parameter pinned to a value by its assumptions is that value in the
/// equation.  Up to 0.54, for an even prime `n`, `solve(x·(n − 2), x)` was
/// `[0]` and `solve(n − 2, x)` "no solution" (both: every `x`), `(n − 2)·x² +
/// x − 1` "no solution" (the quadratic formula over the leading coefficient
/// 0; the root is `1`); for `z` declared zero `x·z`, `x·z + z`, `z·eˣ` gave
/// `[0]`, "no solution", "no solution" (every `x`).  Oracle (exact): the
/// only even prime is 2; `zero` pins `z = 0`.
#[test]
fn parameters_pinned_by_their_assumptions_are_their_values() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let _n = ctx
        .symbol_with("n", &[Assumption::Even, Assumption::Prime])
        .unwrap();
    let _z = ctx.symbol_with("z", &[Assumption::Zero]).unwrap();
    for src in [
        "x*(n - 2)",
        "n - 2",
        "x*(n^2 - 4)",
        "sin(x)*(n - 2)",
        "x*z",
        "x*z + z",
        "exp(x)*z",
    ] {
        let r = parse(&ctx, src).solve(&x);
        assert!(is_every_value(&r), "{src}: {r:?}");
    }
    assert_eq!(roots(&parse(&ctx, "(n - 2)*x^2 + x - 1"), &x), ["1"]);
    assert_eq!(roots(&parse(&ctx, "x*(n - 3)"), &x), ["0"]);
    assert!(is_no_solution(&parse(&ctx, "(n - 2)*x + 1").solve(&x)));
    let m = ctx
        .symbol_with(
            "m",
            &[
                Assumption::Integer,
                Assumption::Positive,
                Assumption::NotPrime,
                Assumption::NotComposite,
            ],
        )
        .unwrap();
    let _ = m;
    assert!(is_every_value(&parse(&ctx, "x*(m - 1)").solve(&x)));
}

/// A constant factor proved 0 makes every value a root.  Up to 0.54
/// `solve(x·euler_number(5), x)` and `solve(x·bernoulli(3), x)` were refused
/// ("cannot decide whether the constant factor … is nonzero") although the
/// zero test proved the factor 0.  Oracle (exact): the odd Euler numbers
/// vanish, `E₅ = 0` (SymPy `euler(5)` = 0); `B₃ = 0` (SymPy `bernoulli(3)`
/// = 0); `E₄ = 5`.  A factor without a value still refuses the product
/// (`δ(i)`), also next to a zero factor.
#[test]
fn a_constant_factor_proved_zero_makes_every_value_a_root() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in [
        "x*euler_number(5)",
        "(x - 1)*euler_number(5)",
        "sin(x)*euler_number(7)",
        "x*bernoulli(3)",
    ] {
        let r = parse(&ctx, src).solve(&x);
        assert!(is_every_value(&r), "{src}: {r:?}");
    }
    assert_eq!(roots(&parse(&ctx, "x*euler_number(4)"), &x), ["0"]);
    assert!(is_refused(&parse(&ctx, "x*DiracDelta(I)").solve(&x)));
    assert!(is_refused(
        &parse(&ctx, "x*DiracDelta(I)*euler_number(5)").solve(&x)
    ));
}

fn bounded(ctx: &Context, src: &str, lo: i64, hi: i64) -> String {
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let e = parse(ctx, src);
    reduce_inequalities(
        &[e.eq_expr(&zero), x.gt(&ctx.int(lo)), x.lt(&ctx.int(hi))],
        &x,
    )
    .unwrap()
    .to_string()
}

/// Families not linear in `n` in a bounded window are located through a
/// monotone parametrisation `x = g(a + b·n)`.  Up to 0.53 they were located
/// by a linear `f64` extrapolation from two members (`sin(x²) = 1/2` in `(−7,
/// 7)` listed 24 of its 32 roots); 0.54 left them as the `ConditionSet` in
/// the window.  Oracle (exact): `x² = π/6 + 2nπ` or `5π/6 + 2nπ` with `0 ≤ x²
/// < 49`, i.e. `0 ≤ n ≤ 7` (`14π + 5π/6 ≈ 46.6`, `16π + π/6 ≈ 50.8`): 32
/// roots; `cosh √x = 1/2` gives `√x = ±iπ/3 + 2nπi`, `x = −(2nπ ± π/3)²`, in
/// `(−7, 7)` only `−π²/9 ≈ −1.097` (mpmath: `cosh(sqrt(-pi**2/9))` = 0.5);
/// `e^{√x} = −1` gives `x = −(2n + 1)²π² ≤ −π² < −7`: none; `sin √x = 0` in
/// `(0, 1000)` gives `x = k²π²`, `1 ≤ k ≤ 10` (`100π² ≈ 987`).
#[test]
fn families_not_linear_in_n_are_located_in_a_window() {
    let ctx = Context::new();
    let s = bounded(&ctx, "sin(x^2) - 1/2", -7, 7);
    assert!(s.starts_with('{'), "{s}");
    assert_eq!(s.matches("sqrt(").count(), 32, "{s}");
    assert_eq!(bounded(&ctx, "cosh(sqrt(x)) - 1/2", -7, 7), "{-1/9*pi^2}");
    assert_eq!(bounded(&ctx, "exp(sqrt(x)) + 1", -7, 7), "EmptySet");
    let s = bounded(&ctx, "sin(sqrt(x))", 0, 1000);
    assert!(s.starts_with('{') && s.contains("100*pi^2"), "{s}");
    assert_eq!(s.matches("pi^2").count(), 10, "{s}");
    // two families with equal members, `(2nπ ∓ acos(1/3))²`: one form each
    let s = bounded(&ctx, "cos(sqrt(x)) - 1/3", -100, 1000);
    assert!(s.starts_with('{'), "{s}");
}

/// A fitted initial-value solution is taken only when its residual at the
/// initial point is proved 0.  Up to 0.54 one whose residual was zero to
/// every digit but not proved was taken when no candidate was proved (the
/// Lambert-W IVPs relied on it: `W(e^{3 + e³}) = e³` was not decided).  Now
/// proved: `W₀(g·e^g) = g` for a real `g ≥ −1` (DLMF 4.13), and the candidate
/// solving the equation identically.  Oracle: mpmath `odefun` at 30 and 60
/// digits, `y(2) = 1.535195822384023237875943935` (`y(1) = 3`) and
/// `0.5729583671269678185960906` (`y(1) = 1`).
#[test]
fn lambert_w_initial_values_are_proved() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let exact =
        &(&y * &parse(&ctx, "exp(x*y)")) + &(&parse(&ctx, "x*exp(x*y) + 1") * &y.formal_diff(&x));
    for (y1, at2) in [
        (3, 1.535_195_822_384_023_f64),
        (1, 0.572_958_367_126_967_8_f64),
    ] {
        let ic = [InitialCondition {
            order: 0,
            x: ctx.int(1),
            value: ctx.int(y1),
        }];
        let sol = exact.solve_ode_ivp(&y, &x, &ic).unwrap();
        assert!(exact.check_ode_solution(&sol, &y, &x), "{sol}");
        let v2 = sol.subs(&x, &ctx.int(2)).eval_f64().unwrap();
        assert!((v2 - at2).abs() < 1e-12, "{sol}: {v2}");
    }
}

/// The condition of a conditional root is simplified before it is stored.
/// Up to 0.54 `√x = y` for an imaginary `y` was refused naming the
/// condition `0 > 0 | 0 == 0 & -y*I >= 0`.  Oracle (exact): `re y = 0`, so
/// the condition `re y > 0 ∨ (re y = 0 ∧ im y ≥ 0)` is `im y ≥ 0`, i.e. `−i·y
/// ≥ 0`.
#[test]
fn conditions_of_conditional_roots_are_simplified() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let _ = ctx.symbol_with("y", &[Assumption::Imaginary]).unwrap();
    let err = parse(&ctx, "sqrt(x) - y")
        .solve(&x)
        .unwrap_err()
        .to_string();
    assert!(!err.contains("0 > 0"), "{err}");
    assert!(err.contains("-y*I >= 0"), "{err}");
}

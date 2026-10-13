//! Equation solving, 0.48 (agent C): the principal complex roots of
//! `solve` for an unknown without assumptions, which is complex.  Up to
//! 0.47 the range restrictions of the inversions (`eˣ = c ≤ 0`, `sin x = c`
//! and `cos x = c` with `|c| > 1`, `cosh x = c < 1`) answered "no real
//! solutions" for a plain `x` while `solve_as_set` gave the complex
//! families; they now hold only for an unknown declared real.  Each test
//! says what was wrong before and cites its oracle: the root substituted
//! back (exact `eval`, and 16-digit complex evaluation), SymPy 1.14
//! `solve`, mpmath 1.3 at 30 and 60 digits.

use symplex::num_complex::Complex64;
use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

/// `e` vanishes at `x = r`: exactly after `eval`, or below `1e-12`
/// (relative to the terms) at 16 digits.
fn assert_root(e: &Ex, x: &Ex, r: &Ex) {
    let at = e.subs(x, r).eval();
    if at.is_zero_structural() {
        return;
    }
    let v = at
        .eval_complex64()
        .unwrap_or_else(|err| panic!("{e} at {r}: {at} ({err})"));
    let scale = e
        .expand()
        .args()
        .iter()
        .filter_map(|t| t.subs(x, r).eval_complex64().ok())
        .map(|z| z.norm())
        .fold(1.0f64, f64::max);
    assert!(v.norm() < 1e-12 * scale, "{e} at {r} is {v}");
}

/// The roots of `src` for a plain `x`, each checked by substitution and
/// as a member of `solve_as_set`, compared with `want` as a set.
fn check_roots(ctx: &Context, src: &str, want: &[&str]) {
    let x = ctx.symbol("x");
    let e = parse(ctx, src);
    let roots = e.solve(&x).unwrap_or_else(|err| panic!("{src}: {err:?}"));
    let set = e.solve_as_set(&x);
    for r in &roots {
        assert_root(&e, &x, r);
        assert_eq!(set.contains(r), Some(true), "{src}: {r} not in {set}");
    }
    let mut got: Vec<String> = roots.iter().map(|r| r.to_string()).collect();
    let mut exp: Vec<String> = want.iter().map(|w| parse(ctx, w).to_string()).collect();
    got.sort();
    exp.sort();
    assert_eq!(got, exp, "{src}");
}

fn no_solution(e: &Ex, x: &Ex) -> bool {
    matches!(e.solve(x), Err(SymplexError::NoSolution { .. }))
}

/// `eˣ = c` for a constant `c ≤ 0`: up to 0.47 `Err(NoSolution { reason:
/// "equation has no real solutions (range restriction of exp/sin/cos/cosh/
/// abs)" })` for a plain `x`, although `solve_as_set` gave `{iπ + 2nπi}`.
/// The principal root is `ln c` (principal logarithm).  SymPy:
/// `solve(exp(x) + 1, x)` → `[I*pi]`, `solve(exp(x) + 2, x)` →
/// `[log(2) + I*pi]`, `solve(exp(2*x) + 1, x)` → `[-I*pi/2, I*pi/2]`
/// (both members of the family `iπ/2 + nπi`; the principal one is
/// `ln(−1)/2`), `solve(1 + exp(-x), x)` → `[I*pi]` (here `−iπ`, the
/// principal `−ln(−1)`; both are roots).  mpmath: `exp(pi*1j) + 1` is 0 to
/// 30 and 60 digits.
#[test]
fn exponential_range_violations_give_principal_complex_roots() {
    let ctx = Context::new();
    check_roots(&ctx, "exp(x) + 1", &["pi*I"]);
    check_roots(&ctx, "exp(x) + 2", &["ln(2) + pi*I"]);
    check_roots(&ctx, "exp(2*x) + 1", &["pi*I/2"]);
    check_roots(&ctx, "exp(-x) + 1", &["-pi*I"]);
    check_roots(&ctx, "3*exp(x - 1) + 6", &["1 + ln(2) + pi*I"]);
}

/// `cosh x = c < 1`: up to 0.47 "no real solutions" for a plain `x`.  The
/// principal roots are `±acosh c`.  SymPy: `solve(cosh(x), x)` →
/// `[-I*pi/2, I*pi/2]`, `solve(cosh(x) - 1/2, x)` → `[-I*pi/3, I*pi/3]`,
/// `solve(cosh(x) + 1, x)` → `[I*pi]` (`−iπ` is a root too: `cosh` is
/// even).
#[test]
fn cosh_below_one_gives_imaginary_roots() {
    let ctx = Context::new();
    check_roots(&ctx, "cosh(x)", &["pi*I/2", "-pi*I/2"]);
    check_roots(&ctx, "cosh(x) - 1/2", &["pi*I/3", "-pi*I/3"]);
    check_roots(&ctx, "cosh(x) + 1", &["pi*I", "-pi*I"]);
}

/// `sin x = c`, `cos x = c` with `|c| > 1`: up to 0.47 "no real solutions"
/// for a plain `x`.  SymPy: `solve(sin(x) - 2, x)` → `[pi - asin(2),
/// asin(2)]`, `solve(cos(x) + 3, x)` → `[2*pi - acos(-3), acos(-3)]`
/// (`−acos(−3)` here, the same family member shifted by `2π`).  mpmath
/// (30 and 60 digits): `asin(2) = 1.5707963267948966192 −
/// 1.3169578969248167086i`, `sin(asin(2)) − 2 ≈ 2e−31`; `acos(−3) =
/// 3.1415926535897932385 − 1.7627471740390860505i`.
#[test]
fn sine_and_cosine_beyond_one_give_complex_principal_roots() {
    let ctx = Context::new();
    check_roots(&ctx, "sin(x) - 2", &["asin(2)", "pi - asin(2)"]);
    check_roots(&ctx, "cos(x) + 3", &["acos(-3)", "-acos(-3)"]);
    let x = ctx.symbol("x");
    let r = parse(&ctx, "asin(2)").eval_complex64().unwrap();
    let want = Complex64::new(std::f64::consts::FRAC_PI_2, -1.316_957_896_924_816_7);
    assert!((r - want).norm() < 1e-14, "{r}");
    let r = parse(&ctx, "sin(2*x) - 3").solve(&x).unwrap();
    assert_eq!(r.len(), 2);
    for v in &r {
        assert_root(&parse(&ctx, "sin(2*x) - 3"), &x, v);
    }
}

/// An unknown declared real keeps the range restrictions: no solution, as
/// up to 0.47 (and SymPy for `Symbol('x', real=True)`: `solve(exp(x) + 1,
/// x)` → `[]`, `solve(sin(x) - 2, x)` → `[]`, `solve(cosh(x), x)` → `[]`).
#[test]
fn real_unknown_keeps_the_range_restrictions() {
    let ctx = Context::new();
    let xr = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    for src in [
        "exp(x) + 1",
        "exp(2*x) + 1",
        "exp(x) + 2",
        "cosh(x)",
        "cosh(x) - 1/2",
        "sin(x) - 2",
        "cos(x) + 3",
        "2^x + 3^x",
        "erf(x) - 2",
    ] {
        assert!(no_solution(&parse(&ctx, src), &xr), "{src}");
    }
    let ctx = Context::new();
    let xp = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    assert!(no_solution(&parse(&ctx, "exp(x) + 1"), &xp));
    // a real root is still found alone
    let r = parse(&ctx, "(x - 1)*(exp(x) + 2)").solve(&xp).unwrap();
    assert_eq!(r, vec![ctx.int(1)]);
}

/// Equations without a solution in ℂ either stay `NoSolution` for a plain
/// `x`: `|x| = −1` (`|z| ≥ 0`), `√x = −1` (the principal square root has a
/// nonnegative real part and is 0 only at 0), `eˣ = 0`.  SymPy:
/// `solve(sqrt(x) + 1, x)` → `[]`, `solve(exp(x), x)` → `[]`;
/// `solve(Abs(x) + 1, x)` raises `NotImplementedError` (symplex answers).
#[test]
fn no_solution_over_the_complex_numbers_either() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in ["abs(x) + 1", "sqrt(x) + 1", "exp(x)", "exp(2*x + 1)"] {
        assert!(no_solution(&parse(&ctx, src), &x), "{src}");
        assert_eq!(
            parse(&ctx, src).solve_as_set(&x).to_string(),
            "EmptySet",
            "{src}"
        );
    }
}

/// Substitutions whose roots in the substituted unknown have only complex
/// preimages: up to 0.47 `sinh x + tanh x` gave `[0]` (`t = eˣ = −1` was
/// dropped), `(2x + 8)(8 + eˣ)` gave `[−4]`, `3^cos x − cos³x` lacked
/// `±acos 3`, `(x − 1)(cosh x − 1/2)` gave `[1]`.  SymPy:
/// `solve(sinh(x) + tanh(x))` → `[0, I*pi]`, `set(solve((2*x + 8)*(8 +
/// exp(x)), x))` → `{-4, log(8) + I*pi}` (both from SymPy's own test
/// suite, `test_rewrite_trig` and `test_solve_transcendental`),
/// `set(solve(3**cos(x) - cos(x)**3))` ⊇ `{acos(3)}` (`test_lambert_bivariate`).
#[test]
fn substitutions_reach_complex_preimages() {
    let ctx = Context::new();
    check_roots(&ctx, "sinh(x) + tanh(x)", &["0", "pi*I"]);
    check_roots(&ctx, "(2*x + 8)*(8 + exp(x))", &["-4", "ln(8) + pi*I"]);
    check_roots(&ctx, "(x - 1)*(cosh(x) - 1/2)", &["1", "pi*I/3", "-pi*I/3"]);
    let x = ctx.symbol("x");
    let e = parse(&ctx, "3^cos(x) - cos(x)^3");
    let roots = e.solve(&x).unwrap();
    for r in &roots {
        assert_root(&e, &x, r);
    }
    for w in ["acos(3)", "-acos(3)"] {
        assert!(roots.contains(&parse(&ctx, w)), "{w} not in {roots:?}");
    }
    // unchanged: roots of the substituted quadratic off the positive axis
    check_roots(&ctx, "exp(2*x) + exp(x) + 1", &["2*pi*I/3", "-2*pi*I/3"]);
}

/// Exponentials of two bases, `C₁·e^(S₁) + C₂·e^(S₂) = 0` with `−C₂/C₁ < 0`:
/// up to 0.47 "no real solutions" for a plain `x`; the principal root of
/// `S₁ − S₂ = ln(−C₂/C₁)`.  `2ˣ + 3ˣ = 0` at `x = iπ/ln(2/3) ≈
/// −7.7481208389248688112i` (mpmath at 30 and 60 digits: `|2^z + 3^z| <
/// 4e−31`, `< 4e−61`).  SymPy 1.14: `solve(2**x + 3**x, x)` →
/// `[I*pi/(-log(3) + log(2))]`... the same value.
#[test]
fn two_exponential_bases_give_a_complex_root() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = parse(&ctx, "2^x + 3^x");
    let roots = e.solve(&x).unwrap();
    assert_eq!(roots.len(), 1, "{roots:?}");
    assert_root(&e, &x, &roots[0]);
    let v = roots[0].eval_complex64().unwrap();
    assert!(
        (v - Complex64::new(0.0, -7.748_120_838_924_869)).norm() < 1e-12,
        "{v}"
    );
}

/// `tan z = ±i` has no solution (`tan z = i` means `e^(2iz) = 0`), but
/// `atan(i) = i·∞` came back as a root: up to 0.47 `solve(tan(x) − i)` was
/// `[I*oo]` and `solve_as_set` `ImageSet(Lambda(_n, _n*pi + I*oo),
/// Integers)`.  SymPy: `solve(tan(x) - I, x)` → `[]`.
#[test]
fn tangent_at_plus_or_minus_i_has_no_solution() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in ["tan(x) - I", "tan(x) + I", "tan(2*x + 1) - I"] {
        let e = parse(&ctx, src);
        assert!(no_solution(&e, &x), "{src}: {:?}", e.solve(&x));
        assert_eq!(e.solve_as_set(&x).to_string(), "EmptySet", "{src}");
    }
    // a finite value of atan is still a root
    check_roots(&ctx, "tan(x) - 2*I", &["atan(2*I)"]);
}

/// The solvers on the real line keep taking the unknown real:
/// `reduce_inequalities` of `cos x = −2` is empty, `eˣ + 1 > 0` holds
/// everywhere, `(x − 1)(eˣ + 2) > 0` is `x > 1` (all as up to 0.47).
#[test]
fn real_line_solvers_are_unchanged() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let eq = ctx.parse_bool("Eq(cos(x), -2)").unwrap();
    assert_eq!(
        reduce_inequalities(&[eq], &x).unwrap().to_string(),
        "EmptySet"
    );
    assert_eq!(
        parse(&ctx, "exp(x) + 1").solve_gt(&x).to_string(),
        "(-oo, oo)"
    );
    assert_eq!(
        parse(&ctx, "(x - 1)*(exp(x) + 2)").solve_gt(&x).to_string(),
        "(1, oo)"
    );
    assert_eq!(
        parse(&ctx, "sin(x) - 2").solve_lt(&x).to_string(),
        "(-oo, oo)"
    );
}

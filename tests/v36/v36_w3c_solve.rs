//! Tests of the 0.50 refactor (w3c): complex domains completed in the solvers.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

fn real(ctx: &Context, name: &str) -> Ex {
    ctx.symbol_with(name, &[Assumption::Real]).unwrap()
}

fn is_no_solution<T: std::fmt::Debug>(r: &Result<T, SymplexError>) -> bool {
    matches!(r, Err(SymplexError::NoSolution { .. }))
}

fn is_refusal<T: std::fmt::Debug>(r: &Result<T, SymplexError>) -> bool {
    matches!(r, Err(SymplexError::ComputationFailed { .. }))
}

/// `|e(x = r)| < 1e-9` in complex double precision.
fn assert_numeric_root(e: &Ex, x: &Ex, r: &Ex) {
    let z = e.subs(x, r).eval_complex64().unwrap();
    assert!(z.norm() < 1e-9, "{e} at {x} = {r} is {z}");
}

/// Every member `n = −2..2` of every family of `fam` solves `e = 0`, and
/// the family has one integer parameter (the members are not one value).
fn assert_families_solve(e: &Ex, x: &Ex, fam: &GeneralSolution) {
    assert_eq!(fam.parameters.len(), 1, "{e}: {:?}", fam.solutions);
    for k in -2..=2 {
        for s in fam.instance(k) {
            assert_numeric_root(e, x, &s);
        }
    }
    let at0 = fam.instance(0);
    let at1 = fam.instance(1);
    assert_ne!(
        at0, at1,
        "{e}: one value, not a family: {:?}",
        fam.solutions
    );
}

/// S3-B2.  `Ex::solve_general` took a plain `x` real (unlike `solve` and
/// `solve_as_set`): `eˣ + 1`, `sin x − 2`, `cos x − 3`, `cosh x − 1/2`,
/// `1/cosh x − 2` and `2ˣ + 3ˣ` were `NoSolution("equation has no real
/// solutions")`.  Over ℂ they have the families `(2n + 1)πi`, `asin 2 +
/// 2nπ` and `π − asin 2 + 2nπ`, `±acos 3 + 2nπ`, `±iπ/3 + 2nπi` (twice)
/// and `(2n + 1)πi/ln(2/3)`.  Oracle: exact identities (`e^((2n+1)πi) =
/// −1`, `sin(asin 2) = 2`, `cosh(iπ/3) = cos(π/3) = 1/2`, `(2/3)^x = −1`
/// at `x ln(2/3) = (2n + 1)πi`); mpmath `2**z + 3**z` at `z =
/// 3πi/ln(2/3)` is `1e-30` (S3 report); each member `n = −2..2` checked
/// in complex double precision.
#[test]
fn general_solve_of_a_plain_unknown_is_over_the_complex_numbers() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in [
        "exp(x) + 1",
        "sin(x) - 2",
        "cos(x) - 3",
        "cosh(x) - 1/2",
        "1/cosh(x) - 2",
        "2**x + 3**x",
    ] {
        let e = parse(&ctx, src);
        let fam = e.solve_general(&x);
        assert!(fam.is_ok(), "{src}: {fam:?}");
        assert_families_solve(&e, &x, &fam.unwrap());
    }
}

/// S3-B2′.  For a plain `x`, `Ex::solve_general` returned one value as
/// the general solution: `eˣ − 1` gave `[0]`, `tanh x − 2` `[atanh 2]`,
/// `sinh x − 2` `[asinh 2]`, `e²ˣ + eˣ + 1` `[2πi/3, −2πi/3]`, all with no
/// parameter.  The solutions are `2nπi`, `atanh 2 + nπi`, `asinh 2 + 2nπi`
/// and `iπ − asinh 2 + 2nπi`, `±2πi/3 + 2nπi`.  Oracle: exact identities
/// (period `2πi` of `exp`, `sinh`; `πi` of `tanh`; `sinh(iπ − w) = sinh
/// w`); mpmath `sinh(pi*1j - asinh(2))` = `2.0` (S3 report); members `n =
/// −2..2` checked numerically.
#[test]
fn general_solve_of_a_plain_unknown_is_not_one_value() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (src, families) in [
        ("exp(x) - 1", 1),
        ("tanh(x) - 2", 1),
        ("sinh(x) - 2", 2),
        ("exp(2*x) + exp(x) + 1", 2),
    ] {
        let e = parse(&ctx, src);
        let fam = e.solve_general(&x).unwrap();
        assert_eq!(fam.solutions.len(), families, "{src}: {:?}", fam.solutions);
        assert_families_solve(&e, &x, &fam);
    }
}

/// A complex unknown whose solutions are not families of the general
/// solver's kind is refused, not answered with the real solutions as all
/// of them: `|x| − 1` gave `[1, −1]` (the unit circle solves it: `x = i`)
/// and `x·eˣ − 1` gave `[W(1)]` (every branch `W_k(1)` solves it: mpmath
/// `lambertw(1, 1)` = `-1.5339133197935745079 + 4.3751851530618983855j`).
/// For an `x` declared real the real solutions are unchanged (`[1, −1]`,
/// `[W(1)]`), and so are the answers of the range restrictions: `exp(x) −
/// 1` is `[0]`, `exp(x) + 1` and `exp(x²) + 1` have no solution.
#[test]
fn general_solve_keeps_the_real_line_for_a_real_unknown() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let i = ctx.parse("I").unwrap();
    assert_numeric_root(&parse(&ctx, "Abs(x) - 1"), &x, &i);
    let w1 = parse(&ctx, "-1.5339133197935745079 + 4.3751851530618983855*I");
    let lambert = parse(&ctx, "x*exp(x) - 1");
    let z = lambert.subs(&x, &w1).eval_complex64().unwrap();
    assert!(z.norm() < 1e-15, "{z}");
    for src in ["Abs(x) - 1", "x*exp(x) - 1"] {
        let r = parse(&ctx, src).solve_general(&x);
        assert!(is_refusal(&r), "{src}: {r:?}");
    }
    let xr = real(&ctx, "xr");
    let show = |src: &str| -> Vec<String> {
        let fam = parse(&ctx, src).solve_general(&xr).unwrap();
        assert!(fam.parameters.is_empty(), "{src}");
        fam.solutions.iter().map(|s| s.to_string()).collect()
    };
    assert_eq!(show("Abs(xr) - 1"), ["1", "-1"]);
    assert_eq!(show("xr*exp(xr) - 1"), ["W(1)"]);
    assert_eq!(show("exp(xr) - 1"), ["0"]);
    for src in ["exp(xr) + 1", "exp(xr**2) + 1", "sin(xr) - 2"] {
        let r = parse(&ctx, src).solve_general(&xr);
        assert!(is_no_solution(&r), "{src}: {r:?}");
    }
}

/// S3-B3.  For a real `xr` and a plain parameter `a`, `exp(a·xr) + 1` was
/// "no solution" (`solve`) and `EmptySet` (`solve_as_set`): the parameter
/// was taken real.  A plain `a` may be complex, and `a = iπ`, `xr = 1`
/// solves it (exact: `e^(iπ) = −1`); now `solve` gives the principal root
/// `iπ/a` and `solve_as_set` the real members of `(2n + 1)πi/a`.  Likewise
/// `cos(a·xr) − 3` (`a = i·acosh 3`, `xr = 1`: `cos(i·acosh 3) = cosh(acosh
/// 3) = 3`, checked in double precision).
#[test]
fn a_plain_parameter_may_be_complex() {
    let ctx = Context::new();
    let xr = real(&ctx, "xr");
    let e = parse(&ctx, "exp(a*xr) + 1");
    let at = e
        .subs(&ctx.symbol("a"), &parse(&ctx, "I*pi"))
        .subs(&xr, &ctx.int(1))
        .eval();
    assert!(at.is_zero_structural(), "{at}");
    let roots = e.solve(&xr).unwrap();
    assert_eq!(roots.len(), 1);
    assert_eq!(roots[0].to_string(), "pi*I/a");
    assert_ne!(e.solve_as_set(&xr).to_string(), "EmptySet");
    let c = parse(&ctx, "cos(a*xr) - 3");
    let at = c
        .subs(&ctx.symbol("a"), &parse(&ctx, "I*acosh(3)"))
        .subs(&xr, &ctx.int(1))
        .eval_complex64()
        .unwrap();
    assert!(at.norm() < 1e-12, "{at}");
    let r = c.solve(&xr);
    assert!(!is_no_solution(&r), "{r:?}");
    assert_ne!(c.solve_as_set(&xr).to_string(), "EmptySet");
}

/// S3-B3′.  For a real `xr`, `|a·xr| = 1` gave `[1/a, −1/a]` and the set
/// `(−∞, ∞) ∩ {1/a, −1/a}`, empty at `a = i`, where `xr = ±1` solve it
/// (`|i·1| = 1`, exact).  A complex `a·xr` is not `±1`: for a real `xr`,
/// `|a·xr|² = (re(a)² + im(a)²)·xr² = 1`, so `xr = ±1/√(re(a)² +
/// im(a)²)` (`±1/|a|`; at `a = i`: `±1`).  Likewise `|xr − a| = 1` is
/// `(xr − re a)² + (im a)² = 1` (at `a = i/2`: `xr = ±√3/2`).  For a
/// real `a` both are unchanged.
#[test]
fn modulus_with_a_plain_parameter_is_not_plus_minus() {
    let ctx = Context::new();
    let xr = real(&ctx, "xr");
    let a = ctx.symbol("a");
    let e = parse(&ctx, "Abs(a*xr) - 1");
    let roots = e.solve(&xr).unwrap();
    assert_eq!(roots.len(), 2, "{roots:?}");
    let mut at_i: Vec<String> = roots
        .iter()
        .map(|r| r.subs(&a, &parse(&ctx, "I")).eval().to_string())
        .collect();
    at_i.sort();
    assert_eq!(at_i, ["-1", "1"]);
    for r in [1, -1] {
        let at = e.subs(&a, &parse(&ctx, "I")).subs(&xr, &ctx.int(r)).eval();
        assert!(at.is_zero_structural(), "{at}");
    }
    assert_ne!(e.solve_as_set(&xr).to_string(), "(-oo, oo) ∩ {1/a, -1/a}");
    let shift = parse(&ctx, "Abs(xr - a) - 1");
    let roots = shift.solve(&xr).unwrap();
    assert_eq!(roots.len(), 2, "{roots:?}");
    let half_i = parse(&ctx, "I/2");
    for r in &roots {
        let v = r.subs(&a, &half_i).eval_f64().unwrap();
        assert!((v.abs() - 3f64.sqrt() / 2.0).abs() < 1e-12, "{r}: {v}");
        assert_numeric_root(&shift.subs(&a, &half_i), &xr, &r.subs(&a, &half_i));
    }
}

/// Real parameters keep the answers of 0.49: for a real `a` and `xr`,
/// `exp(a·xr) + 1` has no solution and `|a·xr| = 1` is `±1/a`.
#[test]
fn real_parameters_are_unchanged() {
    let ctx = Context::new();
    let xr = real(&ctx, "xr");
    let _a = real(&ctx, "a");
    let r = parse(&ctx, "exp(a*xr) + 1").solve(&xr);
    assert!(is_no_solution(&r), "{r:?}");
    assert_eq!(
        parse(&ctx, "exp(a*xr) + 1").solve_as_set(&xr).to_string(),
        "EmptySet"
    );
    let roots: Vec<String> = parse(&ctx, "Abs(a*xr) - 1")
        .solve(&xr)
        .unwrap()
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(roots, ["1/a", "-1/a"]);
}

/// W3b F4.  For a real `x`, `eˣ + e^(ix) = 0` gave `Ok([])` ("no principal
/// root in the domain"), and so did `2ˣ + 1 = 0`: the principal roots
/// `iπ/(1 − i)`, `iπ/ln 2` are not real, and neither is any other branch:
/// `(2n + 1)πi/(1 − i) = (2n + 1)π(−1 + i)/2` has imaginary part `(2n +
/// 1)π/2 ≠ 0`, `(2n + 1)πi/ln 2` is imaginary (exact).  Now `NoSolution`.
/// A polynomial keeps `Ok([])` (`x² + 1`), as documented.
#[test]
fn every_branch_outside_the_domain_is_no_solution() {
    let ctx = Context::new();
    let x = real(&ctx, "x");
    for src in ["exp(x) + exp(I*x)", "2**x + 1", "exp(2*x) + exp(I*x)"] {
        let r = parse(&ctx, src).solve(&x);
        assert!(is_no_solution(&r), "{src}: {r:?}");
    }
    assert_eq!(parse(&ctx, "x**2 + 1").solve(&x).unwrap(), Vec::<Ex>::new());
}

/// W3 F2.  `reduce_inequalities([|x + i| − 2 > 0], x)` was `(2 − i, ∞) ∪
/// (−∞, −i − 2)` (complex endpoints: the linear fast path took `|x + i| >
/// 2` for `x + i > 2 ∨ x + i < −2`).  For a real `x`, `|x + i|² = x² + 1 >
/// 4`: `(−∞, −√3) ∪ (√3, ∞)` (exact; SymPy `solveset(Abs(x + I) - 2 > 0,
/// x, S.Reals)` gives the same).
#[test]
fn modulus_of_a_non_real_linear_function_in_an_inequality() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let e = parse(&ctx, "Abs(x + I) - 2");
    let set = reduce_inequalities(&[e.gt(&zero)], &x).unwrap();
    assert_eq!(set.to_string(), "(-oo, -sqrt(3)) ∪ (sqrt(3), oo)");
    let eq = reduce_inequalities(&[e.eq_expr(&zero)], &x).unwrap();
    for r in ["sqrt(3)", "-sqrt(3)"] {
        assert_eq!(eq.contains(&parse(&ctx, r)), Some(true), "{eq}");
    }
    assert_eq!(eq.contains(&ctx.int(0)), Some(false), "{eq}");
}

/// W3b: a plain `x` in an equation that forces it real.  In 0.49 these
/// were refused (the |·| cases took only a real unknown; up to 0.48 they
/// were split as if `x` were real, "no solution" correct by accident).
/// `|x − π| + x = 0` gives `x = −|x − π| ≤ 0` real, then `π − x + x = π
/// ≠ 0`; `|x|/x + 3 = 0` gives `|x| = −3x`, `x < 0` real, then `−x = −3x`;
/// `|x| + lnΓ(π) − x = 0` gives `x ≥ 0` real, then `lnΓ(π) = 0` (false:
/// mpmath `loggamma(pi)` = `0.82769459232343710153`); `1 + C(|x|, 4) = 0`
/// asks `C(s, 4) = −1` for `s = |x| ≥ 0`, but `s(s−1)(s−2)(s−3) ≥ −1`
/// (minimum at `s = (3 ± √5)/2`, exact).  All `NoSolution`; `|x| = 1`
/// (a circle) stays refused, and `|x| − x = 0` holds on `[0, ∞)`.
#[test]
fn abs_equations_that_force_a_real_unknown() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in [
        "Abs(x - pi) + x",
        "Abs(x)/x + 3",
        "Abs(x) + loggamma(pi) - x",
        "1 + binomial(Abs(x), 4)",
    ] {
        let r = parse(&ctx, src).solve(&x);
        assert!(is_no_solution(&r), "{src}: {r:?}");
    }
    let r = parse(&ctx, "Abs(x) - 1").solve(&x);
    assert!(is_refusal(&r), "{r:?}");
    // `tan |x| = 0` holds on the circles `|x| = nπ` (`x = π`), not at `0`
    // alone: the equation in `s = |x|` needs all its real roots
    let t = parse(&ctx, "tan(Abs(x))");
    assert_numeric_root(&t, &x, &parse(&ctx, "pi"));
    assert_ne!(t.solve_as_set(&x).to_string(), "{0}");
    let r = parse(&ctx, "Abs(x - pi) - x").solve(&x).unwrap();
    assert_eq!(
        r.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        ["1/2*pi"]
    );
}

/// S3-B7.  The summation index is an integer at every term, declared or
/// not: `Σ_{k=1}^{N} sin(kπ)`, `cos(2kπ)`, `floor(k)` stayed unevaluated for
/// a plain `k` (with `k` declared integer: `0`, `N`, `N²/2 + N/2`).  SymPy
/// 1.14 `summation(sin(k*pi), (k, 1, N))` = `0`, `cos(2*pi*k)` → `N`,
/// `floor(k)` → `N**2/2 + N/2`; exact identities at integers.
#[test]
fn summation_index_is_an_integer() {
    let ctx = Context::new();
    let k = ctx.symbol("k");
    let n = ctx.symbol("N");
    let one = ctx.int(1);
    let sum = |src: &str| parse(&ctx, src).summation(&k, &one, &n).to_string();
    assert_eq!(sum("sin(k*pi)"), "0");
    assert_eq!(sum("cos(2*k*pi)"), "N");
    let floor = parse(&ctx, "floor(k)").summation(&k, &one, &n);
    let want = parse(&ctx, "N**2/2 + N/2");
    assert!((&floor - &want).expand().is_zero_structural(), "{floor}");
    // the index keeps its name in what is left unevaluated
    let left = parse(&ctx, "sin(k)/k**k")
        .summation(&k, &one, &n)
        .to_string();
    assert!(!left.contains("__sum"), "{left}");
}

/// W3 F1.  The split of two exponentials took the difference of their
/// exponents real when the unknown is: for a real `x`, `eˣ + e^(x + ix) =
/// 0` was "no solution" (`solve_general`), `EmptySet` (`solve_as_set`,
/// `reduce_inequalities`), while `x = π + 2kπ` solve it (exact:
/// `e^π·(1 + e^(iπ)) = 0`).  The difference `−ix` is not real: `−ix = iπ +
/// 2nπi`, i.e. `x = −π − 2nπ`.
#[test]
fn exponent_difference_that_is_not_real_is_not_a_contradiction() {
    let ctx = Context::new();
    let x = real(&ctx, "x");
    let e = parse(&ctx, "exp(x) + exp(x + I*x)");
    let fam = e.solve_general(&x).unwrap();
    assert_families_solve(&e, &x, &fam);
    assert_ne!(e.solve_as_set(&x).to_string(), "EmptySet");
    let zero = ctx.int(0);
    let set = reduce_inequalities(&[e.eq_expr(&zero)], &x).unwrap();
    assert_eq!(set.contains(&parse(&ctx, "pi")), Some(true), "{set}");
    assert_eq!(set.contains(&parse(&ctx, "3*pi")), Some(true), "{set}");
}

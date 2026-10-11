//! Folds and decisions that let a `0/0` (`nan`) come out `0`, and related
//! equality and order decisions.  Before, an application whose value
//! follows from the assumptions on its symbols (`sin(πn)` for an integer
//! `n`, `|p|` for a positive `p`) was kept, so the zero it stands for was
//! invisible to the canonical arithmetic and to the zero tests of `ratsimp`,
//! `simplify` and `expand`.  Each test says what was wrong before and cites
//! its oracle (SymPy 1.14 unless stated otherwise).

use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

/// A context with the symbols of these tests: `n` integer, `m` odd, `k`
/// even, `p` positive, `q` negative, `r` real, `s` infinite.
fn symbols() -> Context {
    let ctx = Context::new();
    for (name, asm) in [
        ("n", Assumption::Integer),
        ("m", Assumption::Odd),
        ("k", Assumption::Even),
        ("p", Assumption::Positive),
        ("q", Assumption::Negative),
        ("r", Assumption::Real),
        ("s", Assumption::Infinite),
    ] {
        ctx.symbol_with(name, &[asm]).unwrap();
    }
    ctx
}

/// Applications fold when they are built to the value the assumptions give
/// them, as SymPy's `eval` classmethods do.  Before, every one of them was
/// kept (also by `eval`).  SymPy 1.14: `sin(pi*n)` → `0`, `cos(pi*m/2)` →
/// `0`, `(-1)**m` → `-1`, `(-1)**k` → `1`, `exp(2*pi*I*n)` → `1`,
/// `exp(I*pi*m)` → `-1`, `cos(pi*n)` → `(-1)**n`, `sin(pi*(2*n+1)/2)` →
/// `(-1)**n`, `cos(pi*n)**2` → `1`, `tan(pi*k/2)` → `0`, `sinh(I*pi*n)` →
/// `0`, `cosh(I*pi*m/2)` → `0`, `Max(p, 0)` → `p`, `Min(p, p + 1)` → `p`,
/// `Abs(r)**2` → `r**2`, `sqrt(p**2)` → `p`, `sqrt(r**2)` → `Abs(r)`,
/// `sqrt(r**4)` → `r**2`, `sqrt(q**2)` → `-q`, `sign(p)` → `1`, `sign(q)` →
/// `-1`, `Abs(p)` → `p`, `floor(n)` → `n`.  `tan(pi*m/2)` is the pole
/// `zoo` (SymPy keeps it; `tan(pi/2)` is `zoo` in both).
#[test]
fn applications_fold_under_the_assumptions() {
    let ctx = symbols();
    for (input, want) in [
        ("sin(pi*n)", "0"),
        ("sin(-3*pi*n)", "0"),
        ("cos(pi*m/2)", "0"),
        ("(-1)^m", "-1"),
        ("(-1)^k", "1"),
        ("(-1)^(2*n)", "1"),
        ("exp(2*pi*I*n)", "1"),
        ("exp(I*pi*m)", "-1"),
        ("exp(I*pi*(2*n + 1/2))", "I"),
        ("cos(pi*n)", "(-1)^n"),
        ("sin(pi*(2*n + 1)/2)", "(-1)^n"),
        ("cos(pi*n)^2", "1"),
        ("tan(pi*k/2)", "0"),
        ("tan(pi*m/2)", "zoo"),
        ("sinh(I*pi*n)", "0"),
        ("cosh(I*pi*m/2)", "0"),
        ("Max(p, 0)", "p"),
        ("Min(p, 0)", "0"),
        ("Min(p, p + 1)", "p"),
        ("Max(q, 0)", "0"),
        ("Abs(r)^2", "r^2"),
        ("sqrt(p^2)", "p"),
        ("sqrt(r^2)", "abs(r)"),
        ("sqrt(r^4)", "r^2"),
        ("sqrt(q^2)", "-q"),
        ("sign(p)", "1"),
        ("sign(q)", "-1"),
        ("Abs(p)", "p"),
        ("Abs(q)", "-q"),
        ("floor(n)", "n"),
        ("ceiling(n)", "n"),
    ] {
        assert_eq!(parse(&ctx, input).to_string(), want, "{input}");
    }
    // No assumption, no fold.
    for input in ["sin(pi*x)", "(-1)^x", "Abs(x)", "Max(x, 0)", "sqrt(x^2)"] {
        let e = parse(&ctx, input);
        assert_ne!(e.to_string(), "0", "{input}");
        assert_eq!(e.eval(), e, "{input}");
    }
}

/// `0/0` over a denominator that the assumptions make 0 is `nan`.  Before,
/// `ratsimp`, `simplify` and `expand` of `((x + 1)² − x² − 2x − 1)/D`
/// multiplied the numerator out to 0 and returned `0`, the denominator `D`
/// not being recognised as 0.  SymPy 1.14: `ratsimp`, `simplify` and
/// `expand` of each quotient give `nan` (it folds every `D` to 0 first).
#[test]
fn zero_over_a_denominator_zero_by_the_assumptions_is_nan() {
    let ctx = symbols();
    for d in [
        "sin(pi*n)",
        "cos(pi*m/2)",
        "(-1)^m + 1",
        "exp(2*pi*I*n) - 1",
        "Max(p, 0) - p",
        "Abs(r)^2 - r^2",
        "sqrt(p^2) - p",
        "sign(p) - 1",
    ] {
        let s = format!("((x + 1)^2 - x^2 - 2*x - 1)/({d})");
        let e = parse(&ctx, &s);
        let nan = ctx.nan();
        assert_eq!(e.ratsimp(), nan, "ratsimp {s}");
        assert_eq!(e.simplify(), nan, "simplify {s}");
        assert_eq!(e.expand(), nan, "expand {s}");
        assert_eq!(e.together(), nan, "together {s}");
    }
}

/// A numerator that folds to 0 when it is built leaves `0·d⁻¹` to the
/// canonical product, which now tests a denominator with functions in it
/// for an identity (`sin²x + cos²x = 1`).  With the folds above, `(⌈n⌉ −
/// n)/(sin²x + cos²x − 1)` would have been `0` as soon as it was built
/// (SymPy 1.14: `0`, the same gap); before them `simplify` saw the `0/0`
/// and gave `nan`, which it is (the denominator is 0 at every `x`).
#[test]
fn zero_over_a_trigonometric_identity_is_nan() {
    let ctx = symbols();
    for s in [
        "(ceiling(n) - n)/(sin(x)^2 + cos(x)^2 - 1)",
        "(x + 3)*sin(pi*n)/((sin(x)^2 + cos(x)^2 - 1)*(x + 2))",
        "0/(cosh(x)^2 - sinh(x)^2 - 1)",
    ] {
        assert_eq!(parse(&ctx, s), ctx.nan(), "{s}");
    }
    // A denominator that is not 0 keeps the 0.
    let s = "(ceiling(n) - n)/(sin(x)^2 + cos(x)^2)";
    assert_eq!(parse(&ctx, s), ctx.int(0), "{s}");
}

/// A product with a zero factor is `nan` when another factor is a function
/// at an infinite argument that is not known finite, or a symbol declared
/// infinite.  Before, every unevaluated application counted as finite:
/// `Max(0, 2/∞)·cosh((1 + i)·∞ − 3/2)` was `0`, also `0·|(1 + i)·∞|`,
/// `0·Γ((1 + i)·∞)`, `0·s` for an infinite `s`.  SymPy 1.14:
/// `cosh(3/2 - oo*(1 + I)).is_finite` is `False` and the product `nan`;
/// `Symbol('s', infinite=True)*0` → `nan`.  Bounded or decaying ones keep
/// the `0`: `sin(∞)` (SymPy: `AccumBounds(-1, 1)`, times 0 is 0),
/// `atan(1 + i·∞)`, `Γ(i·∞)` (→ 0), `tan((1 + i)·∞)` (→ i), `e^((−1 + i)·∞)`
/// (→ 0; mpmath at `t = 10²⁰` along each direction).
#[test]
fn zero_times_a_function_at_an_infinite_argument() {
    let ctx = symbols();
    for s in [
        "Max(0, 2/oo)*cosh((1 + I)*oo - 3/2)",
        "0*Abs((1 + I)*oo)",
        "0*gamma((1 + I)*oo + 2)",
        "0*sinh((1 - I)*oo)",
        "0*cos((2 + I)*oo + I)",
        "0*s",
    ] {
        assert_eq!(parse(&ctx, s), ctx.nan(), "{s}");
    }
    for s in [
        "0*sin(oo)",
        "0*atan(1 + I*oo)",
        "0*gamma(I*oo)",
        "0*tan((1 + I)*oo)",
        "0*exp((-1 + I)*oo)",
        "0*cosh(I*oo)",
    ] {
        assert_eq!(parse(&ctx, s), ctx.int(0), "{s}");
    }
}

/// `Eq`/`Ne` of constants are decided by their difference: a nonzero
/// rational, or an algebraic number whose minimal polynomial decides it;
/// orders of algebraic constants likewise.  Before, `eval` compared only two
/// Gaussian rationals: `Ne(airyai(18/7) + 3 − 10⁻¹⁵⁰, 3 + airyai(18/7))` and
/// `Eq(√(3 + 2√2), 1 + √2)` stayed, and `evalf` refused the second
/// (`PrecisionExhausted`) and `√(5 − 2√6) ≥ √3 − √2`.  SymPy 1.14:
/// `Ne(airyai(18/7) + 3 - 10**-150, 3 + airyai(18/7))` → `True`,
/// `Eq(sqrt(3 + 2*sqrt(2)), 1 + sqrt(2))` → `True`, `sqrt(5 - 2*sqrt(6)) >=
/// sqrt(3) - sqrt(2)` → `True`, `cbrt(7 + 5*sqrt(2)) > 1 + sqrt(2)` →
/// `False`, `Eq(1/(sqrt(3) - 1), (sqrt(3) + 1)/2)` → `True`; mpmath at 400
/// digits for `Eq(√(52 + 16√3) + 10⁻¹⁰, 2 + 4√3)` (false: the difference
/// is `10⁻¹⁰`).
#[test]
fn constant_relations_decided_by_their_difference() {
    let ctx = Context::new();
    for (cond, want) in [
        ("Ne(airyai(18/7) + 3 - 10^(-150), 3 + airyai(18/7))", "1"),
        ("Eq(airyai(18/7) + 3 - 10^(-150), 3 + airyai(18/7))", "0"),
        ("Eq(sqrt(3 + 2*sqrt(2)), 1 + sqrt(2))", "1"),
        ("Ne(sqrt(3 + 2*sqrt(2)), 1 + sqrt(2))", "0"),
        ("Eq(sqrt(52 + 16*sqrt(3)) + 10^(-10), 2 + 4*sqrt(3))", "0"),
        ("Eq(1/(sqrt(3) - 1), (sqrt(3) + 1)/2)", "1"),
        ("Eq(GoldenRatio, (1 + sqrt(5))/2)", "1"),
        ("sqrt(5 - 2*sqrt(6)) >= sqrt(3) - sqrt(2)", "1"),
        ("sqrt(5 - 2*sqrt(6)) > sqrt(3) - sqrt(2)", "0"),
        ("cbrt(7 + 5*sqrt(2)) > 1 + sqrt(2)", "0"),
        ("sqrt(2) + sqrt(3) > sqrt(5 + 2*sqrt(6)) - 10^(-300)", "1"),
    ] {
        let s = format!("Piecewise(1 if {cond}, 0 if True)");
        let e = parse(&ctx, &s);
        assert_eq!(e.eval().to_string(), want, "eval {s}");
        assert_eq!(e.eval_decimal(30).unwrap(), want, "evalf {s}");
    }
    // A symbolic difference that is a nonzero number: `x + 1 ≠ x` (SymPy:
    // `Eq(x + 1, x)` → `False`).
    let x = ctx.symbol("x");
    let cond = (&x + 1).eq_expr(&x);
    let e = Ex::piecewise(&[(&ctx.int(1), &cond), (&ctx.int(0), &ctx.bool_true())]);
    assert_eq!(e.eval().to_string(), "0");
}

/// An order between non-real values has no meaning and stays a relation;
/// equality of them is still decided.  Before, `BoolEx::eval`, `simplify`
/// and `is_tautology` folded the order of the real difference: `i·π >
/// ln(−1)` was `False`, `i·π ≥ ln(−1)` and `i·π < ln(−1) + 1` `True`,
/// `e^i < e^i` `False`.  SymPy 1.14: `I*pi > log(-1)` raises "Invalid
/// comparison of non-real I*pi"; `Eq(I*pi, log(-1))` → `True`.
#[test]
fn orders_of_non_real_values_stay() {
    let ctx = Context::new();
    for s in [
        "I*pi > log(-1)",
        "I*pi >= log(-1)",
        "I*pi < log(-1) + 1",
        "exp(I) < exp(I)",
        "log(-2) > log(2) + I*pi",
    ] {
        let b = ctx.parse_bool(s).unwrap();
        for r in [b.eval(), b.simplify()] {
            let shown = r.to_string();
            assert!(shown != "True" && shown != "False", "{s}: {shown}");
        }
        assert_eq!(b.is_tautology(), None, "{s}");
    }
    for (s, want) in [
        ("Eq(I*pi, log(-1))", "True"),
        ("Ne(I*pi, log(-1) + 1)", "True"),
        ("Eq(exp(I), exp(I) + 1)", "False"),
    ] {
        assert_eq!(ctx.parse_bool(s).unwrap().eval().to_string(), want, "{s}");
    }
}

/// An equation that holds wherever both sides are defined is solved by
/// that domain.  Before, `reduce_inequalities` took the solver's identity
/// (and the zero numerator of a rational equation) for an empty list of
/// roots: `Eq((x + 1)², x² + 2x + 1)` and `Eq((x² − 1)/(x − 1), x + 1)`
/// gave `EmptySet`.  SymPy 1.14: `solveset(..., x, Reals)` → `Reals` and
/// `Union(Interval.open(-oo, 1), Interval.open(1, oo))`;
/// `reduce_inequalities(Eq((x + 1)**2, x**2 + 2*x + 1), x)` → `(-oo < x) &
/// (x < oo)`.
#[test]
fn identities_solve_to_their_domain() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (s, want) in [
        ("Eq((x + 1)^2, x^2 + 2*x + 1)", "(-oo, oo)"),
        ("Eq((x^2 - 1)/(x - 1), x + 1)", "(-oo, 1) ∪ (1, oo)"),
        (
            "Eq(1/(2*x + 1), (2*x - 1)/(4*x^2 - 1))",
            "(-oo, -1/2) ∪ (-1/2, 1/2) ∪ (1/2, oo)",
        ),
        ("Ne((x + 1)^2, x^2 + 2*x + 1)", "EmptySet"),
        ("Ne((x^2 - 1)/(x - 1), x + 1)", "EmptySet"),
        ("Eq((x + 1)^2, x^2 + 1)", "{0}"),
    ] {
        let b = ctx.parse_bool(s).unwrap();
        assert_eq!(b.solve_for(&x).unwrap().to_string(), want, "{s}");
        let r = SetEx::reduce_inequalities(&[b], &x).unwrap();
        assert_eq!(r.to_string(), want, "{s}");
    }
}

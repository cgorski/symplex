//! Decision procedures: relations decided numerically (`evalf` of a
//! `Piecewise`), Boolean formulas, and the identically-zero tests behind
//! `ratsimp` / `simplify`.

use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

/// A relation between an expression and itself is decided without an
/// order on its value: `a > a` and `a < a` are false, `a ≥ a`, `a ≤ a`
/// true.  Before, `evalf` compared the difference `+0` with astro-float's
/// `is_positive` (the sign bit, set for `+0`): `Piecewise((1, besselj(1, 2)
/// > besselj(1, 2)), (0, True))` evaluated to `1`, also `W(1)`,
/// `polylog(2, 1/3)`, `airyai(1)` and sums equal up to the order of their
/// terms (`eval` folds such relations only for operands it knows real).
/// SymPy 1.14: `Piecewise((1, besselj(1, 2) > besselj(1, 2)), (0, True))`
/// is `0`.
#[test]
fn evalf_decides_a_strict_relation_of_equal_operands_false() {
    let ctx = Context::new();
    for a in [
        "besselj(1, 2)",
        "LambertW(1)",
        "polylog(2, 1/3)",
        "airyai(1)",
        "2*besselk(1, 3)",
        "1/4 + airybi(5/6)",
    ] {
        for (rel, want) in [(">", "0"), ("<", "0"), (">=", "1"), ("<=", "1")] {
            let s = format!("Piecewise(1 if {a} {rel} {a}, 0 if True)");
            assert_eq!(parse(&ctx, &s).eval_decimal(30).unwrap(), want, "{s}");
        }
    }
    // Equal up to the order of the terms (canonically the same node), and
    // exact operands equal in value.
    for s in [
        "Piecewise(1 if 2*fresnels(3) > fresnels(3) + fresnels(3), 0 if True)",
        "Piecewise(1 if (1 + I)*(1 - I) < 2, 0 if True)",
    ] {
        assert_eq!(parse(&ctx, s).eval_decimal(30).unwrap(), "0", "{s}");
    }
}

/// `±∞`, `zoo` and `nan` in a condition are compared by kind: `−∞ < x <
/// ∞` for a finite real `x`, `∞ ≥ ∞`, `nan` equals nothing.  Before,
/// `evalf` had no value for them and refused every such `Piecewise`.
/// SymPy 1.14: `Piecewise((1, exp(1000) < oo), (0, True))` → `1`, `(1,
/// -oo >= -exp(1000))` → `0`, `Eq(oo, exp(1000))` → `False`, `Ne(1, zoo)`
/// → `True`, `Eq(nan, nan)` → `False`, `oo >= oo` → `True`, `oo > oo` →
/// `False`; `oo > zoo` raises (undecided here).
#[test]
fn evalf_decides_conditions_with_infinities() {
    let ctx = Context::new();
    for (cond, want) in [
        ("exp(1000) < oo", "1"),
        ("-oo >= -exp(1000)", "0"),
        ("Eq(oo, exp(1000))", "0"),
        ("Ne(1, zoo)", "1"),
        ("Eq(nan, nan)", "0"),
        ("oo >= oo", "1"),
        ("oo > oo", "0"),
        ("besselj(1, 2) > -oo", "1"),
    ] {
        let s = format!("Piecewise(1 if {cond}, 0 if True)");
        assert_eq!(parse(&ctx, &s).eval_decimal(30).unwrap(), want, "{s}");
    }
    let s = "Piecewise(1 if oo > zoo, 0 if True)";
    assert!(parse(&ctx, s).eval_decimal(30).is_err());
}

/// A condition at its threshold to the working precision is pursued to
/// the precision that decides it, as the zero of a difference is.  Before,
/// the evaluation stopped at the cap of the digits and refused:
/// `sin(13/5)² + cos(13/5)² ≥ 1 + 10⁻²⁰⁰` (false: the left side is 1)
/// and `9 > 9 + e⁻⁴⁶⁰` at 30 digits.  mpmath at 400 digits: the
/// differences are `−10⁻²⁰⁰` and `−e⁻⁴⁶⁰ ≈ −1.0·10⁻²⁰⁰`.  A relation
/// between values equal by an identity stays undecided (SymPy 1.14 leaves
/// `sin(1)**2 + cos(1)**2 > 1` unevaluated too).
#[test]
fn evalf_pursues_a_condition_at_its_threshold() {
    let ctx = Context::new();
    for (cond, want) in [
        ("sin(13/5)^2 + cos(13/5)^2 >= 1 + 10^(-200)", "0"),
        ("sin(13/5)^2 + cos(13/5)^2 < 1 + 10^(-200)", "1"),
        ("9 > 9 + exp(-460)", "0"),
        ("Ne(airyai(18/7) + 3 - 10^(-150), 3 + airyai(18/7))", "1"),
    ] {
        let s = format!("Piecewise(1 if {cond}, 0 if True)");
        assert_eq!(parse(&ctx, &s).eval_decimal(30).unwrap(), want, "{s}");
    }
    let s = "Piecewise(1 if sin(1)^2 + cos(1)^2 > 1, 0 if True)";
    assert!(parse(&ctx, s).eval_decimal(30).is_err());
}

/// The zero tests behind `ratsimp` and `simplify` sample a symbol only at
/// values it can take.  Before, an even prime `n` was sampled at 3, 5, 7,
/// 11, where `(−1)ⁿ − 1 + sin²x + cos²x − 1` is `−2`: the denominator,
/// which is 0 at the only admissible `n = 2`, was taken for nonzero and
/// `0/0` became `0` (it is `nan`, as for an even `n`, which was right).
/// SymPy 1.14 evaluates `(-1)**n` to `1` for an even `n` and leaves
/// `(…)/(sin(x)**2 + cos(x)**2 - 1)`; at every `x` the denominator is 0
/// (exact: `sin²x + cos²x = 1`).
#[test]
fn identity_zero_tests_sample_admissible_values() {
    let ctx = Context::new();
    let _n = ctx
        .symbol_with("n", &[Assumption::Even, Assumption::Prime])
        .unwrap();
    let _m = ctx.symbol_with("m", &[Assumption::Even]).unwrap();
    for v in ["n", "m"] {
        let s = format!("((x + 1)^2 - x^2 - 2*x - 1)/((-1)^{v} - 1 + sin(x)^2 + cos(x)^2 - 1)");
        let e = parse(&ctx, &s);
        assert_eq!(e.ratsimp(), ctx.nan(), "ratsimp {s}");
        assert_eq!(e.simplify(), ctx.nan(), "simplify {s}");
    }
}

/// A symbol declared `Real` ranges over all of ℝ; one with sign or
/// integrality assumptions over its domain, and a univariate formula is
/// decided there exactly (the solution set over ℝ intersected with the
/// domain).  Before, any declared assumption — `Real` included, which
/// only says the symbol is not imaginary nor infinite — made the formula
/// undecided.  Exact references: the integers of `(1/2, 3/2)` are `{1}`;
/// `x² < 4` holds for no odd composite (the least is 9) and for no even
/// prime (`2² = 4`); `x > 1 → x > 0` holds on ℝ.
#[test]
fn univariate_formulas_are_decided_on_the_domain_of_the_symbol() {
    let ctx = Context::new();
    let check = |asm: &[Assumption], src: &str, taut: Option<bool>, contra: Option<bool>| {
        let ctx = Context::new();
        ctx.symbol_with("x", asm).unwrap();
        let f = ctx.parse_bool(src).unwrap();
        assert_eq!(f.is_tautology(), taut, "taut {asm:?} {src}");
        assert_eq!(f.is_contradiction(), contra, "contra {asm:?} {src}");
    };
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    let implication = x.gt(&ctx.int(1)).implies(&x.gt(&ctx.int(0)));
    assert_eq!(implication.is_tautology(), Some(true));
    check(
        &[Assumption::Real],
        "x > 2 | x < 3",
        Some(true),
        Some(false),
    );
    check(&[Assumption::Real], "x**2 < 4", Some(false), Some(false));
    check(
        &[Assumption::Positive],
        "x**2 + x > 0",
        Some(true),
        Some(false),
    );
    check(
        &[Assumption::Negative],
        "x > -1 & x**2 > 1",
        Some(false),
        Some(true),
    );
    check(
        &[Assumption::Integer],
        "x > 1/2 & x < 3/2 & Ne(x, 1)",
        Some(false),
        Some(true),
    );
    check(
        &[Assumption::Integer],
        "x < 1/2 | x > 1/3",
        Some(true),
        Some(false),
    );
    check(
        &[Assumption::Odd, Assumption::Composite],
        "x**2 < 4",
        Some(false),
        Some(true),
    );
    check(
        &[Assumption::Even, Assumption::Prime],
        "x**2 < 4",
        Some(false),
        Some(true),
    );
    check(
        &[Assumption::Even, Assumption::Prime],
        "Eq(x, 2)",
        Some(true),
        Some(false),
    );
    check(&[Assumption::Prime], "x > 1", Some(true), Some(false));
    check(
        &[Assumption::Odd],
        "Ne(x, 0) & Ne(x, 2)",
        Some(true),
        Some(false),
    );
}

/// An equation whose sides are equal as polynomials holds for every value
/// of its symbol.  Before, the inequality solver took the solver's "every
/// value" for "no root": the solution set of `(x + 1)² = x² + 2x + 1` was
/// empty, so the equation was decided a contradiction (`is_tautology`
/// `Some(false)`, `satisfiable` `Some(false)`) and its negation a
/// tautology.  SymPy 1.14: `solveset(Eq((x + 1)**2, x**2 + 2*x + 1), x,
/// S.Reals)` → `Reals`; `solveset(Eq((x**2 - 1)/(x - 1), x + 1), x,
/// S.Reals)` → `Union(Interval.open(-oo, 1), Interval.open(1, oo))` (not
/// every value: undecided here).
#[test]
fn polynomial_identity_equations_hold_everywhere() {
    for asm in [&[][..], &[Assumption::Real][..]] {
        let ctx = Context::new();
        ctx.symbol_with("x", asm).unwrap();
        for (src, taut, contra) in [
            (
                "Eq(2*(x**2 - 1), 2*(x - 1)*(x + 1))",
                Some(true),
                Some(false),
            ),
            ("Ne((x + 1)**2, x**2 + 2*x + 1)", Some(false), Some(true)),
            (
                "Eq((x + 1)**2, x**2 + 2*x + 1) & (x > 0)",
                Some(false),
                Some(false),
            ),
            ("Eq((x**2 - 1)/(x - 1), x + 1)", None, None),
        ] {
            let f = ctx.parse_bool(src).unwrap();
            assert_eq!(f.is_tautology(), taut, "taut {asm:?} {src}");
            assert_eq!(f.is_contradiction(), contra, "contra {asm:?} {src}");
        }
    }
}

/// `nan` equals nothing, itself included.  Before, `Eq(nan, nan)` was
/// simplified to `True` (a relation of a node with itself) and was a
/// tautology.  SymPy 1.14: `Eq(nan, nan)` → `False`, `Ne(nan, nan)` →
/// `True`; `Eq(zoo, zoo)` → `True`.
#[test]
fn nan_equals_nothing_in_boolean_simplification() {
    let ctx = Context::new();
    let eq = ctx.parse_bool("Eq(nan, nan)").unwrap();
    assert_eq!(eq.simplify(), ctx.bool_false());
    assert_eq!(eq.is_tautology(), Some(false));
    assert_eq!(eq.is_contradiction(), Some(true));
    let ne = ctx.parse_bool("Ne(nan, nan)").unwrap();
    assert_eq!(ne.simplify(), ctx.bool_true());
    let zoo = ctx.parse_bool("Eq(zoo, zoo)").unwrap();
    assert_eq!(zoo.simplify(), ctx.bool_true());
}

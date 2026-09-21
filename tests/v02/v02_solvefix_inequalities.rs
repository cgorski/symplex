//! Regression tests for the 0.2 "silently wrong solver results" campaign:
//! the sign-chart inequality solver behind `Ex::solve_gt` & co. and
//! `reduce_inequalities`.
//!
//! Every table entry is cross-checked by an independent oracle: membership
//! of the returned set at many rational sample points is compared with a
//! direct exact evaluation of the inequality at those points (a point
//! where the expression is undefined or not real is never a solution).

use symplex::prelude::*;

/// Wall-clock hang guard.  Two seconds on a developer machine; scaled up on
/// shared CI runners (`CI` is set), which are several times slower and noisy.
fn time_budget(secs: u64) -> std::time::Duration {
    let mult = if std::env::var_os("CI").is_some() {
        5
    } else {
        1
    };
    std::time::Duration::from_secs(secs * mult)
}

/// Relation of the table entries (`expr rel 0`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rel {
    Gt,
    Ge,
    Lt,
    Le,
}

impl Rel {
    fn holds(self, sign: std::cmp::Ordering) -> bool {
        use std::cmp::Ordering::*;
        matches!(
            (self, sign),
            (Rel::Gt, Greater)
                | (Rel::Ge, Greater | Equal)
                | (Rel::Lt, Less)
                | (Rel::Le, Less | Equal)
        )
    }

    fn solve(self, e: &Ex, x: &Ex) -> SetEx {
        match self {
            Rel::Gt => e.solve_gt(x),
            Rel::Ge => e.solve_ge(x),
            Rel::Lt => e.solve_lt(x),
            Rel::Le => e.solve_le(x),
        }
    }
}

/// Truth of `expr rel 0` at the rational point `p`, decided by direct
/// evaluation.  `None` when the evaluation is numerically ambiguous.
fn oracle(expr: &Ex, x: &Ex, p: &Ex, rel: Rel) -> Option<bool> {
    let v = expr.subs(x, p).eval();
    let s = format!("{v}");
    if s == "zoo" || s == "oo" || s == "-oo" || s == "nan" {
        return Some(false);
    }
    if let Some(r) = v.as_rational() {
        return Some(rel.holds(r.cmp(&num_rational::Ratio::from_integer(0.into()))));
    }
    match v.eval_complex64() {
        Ok((re, im)) => {
            if !re.is_finite() || !im.is_finite() {
                return Some(false);
            }
            if im.abs() > 1e-12 {
                return Some(false); // not real ⇒ outside the domain
            }
            if re.abs() < 1e-12 {
                return None; // cannot tell zero from tiny
            }
            Some(rel.holds(re.partial_cmp(&0.0).unwrap()))
        }
        Err(_) => Some(false), // undefined (e.g. ln(0), 1/0)
    }
}

/// Rational sample points: k/4 for k ∈ [-40, 40] plus a few far/odd ones.
fn sample_points(ctx: &Context) -> Vec<Ex> {
    let mut pts: Vec<Ex> = (-40..=40).map(|k| ctx.rational(k, 4)).collect();
    for (p, q) in [
        (-100, 1),
        (100, 1),
        (1, 3),
        (-1, 3),
        (7, 3),
        (-7, 3),
        (1, 1000),
        (-1, 1000),
    ] {
        pts.push(ctx.rational(p, q));
    }
    pts
}

/// Solve `expr rel 0`, require a closed-form set, and cross-check it
/// against the oracle at every sample point.
fn check(ctx: &Context, x: &Ex, expr: &Ex, rel: Rel) -> SetEx {
    let sol = rel.solve(expr, x);
    assert!(
        !sol.has_unevaluated(),
        "{expr} {rel:?} 0 was not solved: {sol}"
    );
    let mut checked = 0usize;
    for p in sample_points(ctx) {
        let Some(want) = oracle(expr, x, &p, rel) else {
            continue;
        };
        let got = sol.contains(&p);
        assert_eq!(
            got,
            Some(want),
            "{expr} {rel:?} 0 → {sol}: membership of {p} (expected {want})"
        );
        checked += 1;
    }
    assert!(checked > 40, "too few decidable sample points for {expr}");
    sol
}

/// The table: (name, expression builder, relation, expected display).
/// An empty expected string skips the display check (irrational or
/// solver-shaped endpoints are checked by membership only).
fn table(ctx: &Context, x: &Ex) -> Vec<(&'static str, Ex, Rel, &'static str)> {
    let one = ctx.int(1);
    let two = ctx.int(2);
    let always: BoolEx = one.gt(&ctx.int(0));
    vec![
        // ── rational functions ──────────────────────────────────────
        ("1/x - 2 > 0", 1 / x - 2, Rel::Gt, "(0, 1/2)"),
        (
            "(x-1)/(x+1) >= 0",
            (x - 1) / (x + 1),
            Rel::Ge,
            "(-oo, -1) ∪ [1, oo)",
        ),
        ("x/(x-2) - 1 < 0", x / (x - 2) - 1, Rel::Lt, "(-oo, 2)"),
        ("1/x <= 0", 1 / x, Rel::Le, "(-oo, 0)"),
        (
            "(x^2-1)/(x^2-4) > 0",
            (x.powi(2) - 1) / (x.powi(2) - 4),
            Rel::Gt,
            "(-oo, -2) ∪ (-1, 1) ∪ (2, oo)",
        ),
        (
            "(x^2-1)/(x^2-4) <= 0",
            (x.powi(2) - 1) / (x.powi(2) - 4),
            Rel::Le,
            "(-2, -1] ∪ [1, 2)",
        ),
        ("1/(x^2+1) > 0", 1 / (x.powi(2) + 1), Rel::Gt, "(-oo, oo)"),
        (
            "1/(x-1)^2 > 0",
            1 / (x - 1).powi(2),
            Rel::Gt,
            "(-oo, 1) ∪ (1, oo)",
        ),
        ("x + 1/x - 2 >= 0", x + 1 / x - 2, Rel::Ge, "(0, oo)"),
        (
            "x + 1/x + 2 < 0",
            x + 1 / x + 2,
            Rel::Lt,
            "(-oo, -1) ∪ (-1, 0)",
        ),
        (
            "(x^3-x)/(x^2+x-6) >= 0",
            (x.powi(3) - x) / (x.powi(2) + x - 6),
            Rel::Ge,
            "(-3, -1] ∪ [0, 1] ∪ (2, oo)",
        ),
        (
            "2/(x-2) - 1/(x+1) > 0",
            2 / (x - 2) - 1 / (x + 1),
            Rel::Gt,
            "(-4, -1) ∪ (2, oo)",
        ),
        (
            "x/(x^2-1) >= 0",
            x / (x.powi(2) - 1),
            Rel::Ge,
            "(-1, 0] ∪ (1, oo)",
        ),
        (
            "(x-1)/(x^2+1) < 0",
            (x - 1) / (x.powi(2) + 1),
            Rel::Lt,
            "(-oo, 1)",
        ),
        (
            "1/(x-1) - 1/(x+1) - 1/3 <= 0",
            1 / (x - 1) - 1 / (x + 1) - ctx.rational(1, 3),
            Rel::Le,
            "",
        ),
        // ── radicals ────────────────────────────────────────────────
        ("sqrt(x) - 2 < 0", x.sqrt() - 2, Rel::Lt, "[0, 4)"),
        ("sqrt(x) + 2 > 0", x.sqrt() + 2, Rel::Gt, "[0, oo)"),
        ("sqrt(x-1) - 1 <= 0", (x - 1).sqrt() - 1, Rel::Le, "[1, 2]"),
        (
            "sqrt(4-x^2) - 1 > 0",
            (4 - x.powi(2)).sqrt() - 1,
            Rel::Gt,
            "",
        ),
        (
            "sqrt(x) (x - 1) <= 0",
            x.sqrt() * (x - 1),
            Rel::Le,
            "[0, 1]",
        ),
        (
            "x^(3/2) - 8 < 0",
            x.pow(&ctx.rational(3, 2)) - 8,
            Rel::Lt,
            "[0, 4)",
        ),
        ("1/sqrt(x) - 1 > 0", 1 / x.sqrt() - 1, Rel::Gt, "(0, 1)"),
        (
            "sqrt(x^2-1) - 1 <= 0",
            (x.powi(2) - 1).sqrt() - 1,
            Rel::Le,
            "",
        ),
        (
            "sqrt(x) * (x - 1) < 0",
            x.sqrt() * (x - 1),
            Rel::Lt,
            "(0, 1)",
        ),
        (
            "sqrt(x) / (x - 1) > 0",
            x.sqrt() / (x - 1),
            Rel::Gt,
            "(1, oo)",
        ),
        // ── logarithms / exponentials ──────────────────────────────
        ("ln(x) > 0", x.ln(), Rel::Gt, "(1, oo)"),
        ("ln(x) <= 0", x.ln(), Rel::Le, "(0, 1]"),
        ("ln(x-2) - 1 >= 0", (x - 2).ln() - 1, Rel::Ge, ""),
        ("exp(x) - 2 > 0", x.exp() - 2, Rel::Gt, ""),
        ("x ln(x) < 0", x * x.ln(), Rel::Lt, "(0, 1)"),
        ("ln(x^2-1) >= 0", (x.powi(2) - 1).ln(), Rel::Ge, ""),
        ("1/ln(x) > 0", 1 / x.ln(), Rel::Gt, "(1, oo)"),
        // ── absolute value / sign / piecewise ──────────────────────
        (
            "|x-1| - 2 >= 0",
            (x - 1).abs() - 2,
            Rel::Ge,
            "(-oo, -1] ∪ [3, oo)",
        ),
        ("sign(x) > 0", x.sign(), Rel::Gt, "(0, oo)"),
        ("sign(x) >= 0", x.sign(), Rel::Ge, "[0, oo)"),
        ("|x| - x <= 0", x.abs() - x, Rel::Le, "[0, oo)"),
        (
            "|x^2-1| - 3 < 0",
            (x.powi(2) - 1).abs() - 3,
            Rel::Lt,
            "(-2, 2)",
        ),
        ("x|x| - 1 > 0", x * x.abs() - 1, Rel::Gt, "(1, oo)"),
        (
            "H(x) - 1/2 > 0",
            x.heaviside() - ctx.rational(1, 2),
            Rel::Gt,
            "(0, oo)",
        ),
        (
            "max(x, 2) - 3 <= 0",
            x.max_with(&two) - 3,
            Rel::Le,
            "(-oo, 3]",
        ),
        ("min(x, 1/x) > 0", x.min_with(&(1 / x)), Rel::Gt, "(0, oo)"),
        (
            "piecewise(x, x>=0; -x) - 1 < 0",
            Ex::piecewise(&[(x, &x.ge(&ctx.int(0))), (&(-x), &always)]) - 1,
            Rel::Lt,
            "(-1, 1)",
        ),
        (
            "|x|/(x-1) >= 0",
            x.abs() / (x - 1),
            Rel::Ge,
            "(1, oo) ∪ {0}",
        ),
        (
            "1/|x| - 1 > 0",
            1 / x.abs() - 1,
            Rel::Gt,
            "(-1, 0) ∪ (0, 1)",
        ),
        (
            "|x| - 1/x >= 0",
            x.abs() - 1 / x,
            Rel::Ge,
            "(-oo, 0) ∪ [1, oo)",
        ),
        // ── polynomials (sanity) ───────────────────────────────────
        ("x^2 - 2 < 0", x.powi(2) - 2, Rel::Lt, "(-sqrt(2), sqrt(2))"),
        ("x^3 - x >= 0", x.powi(3) - x, Rel::Ge, "[-1, 0] ∪ [1, oo)"),
        (
            "(x-1)^2 (x+2) <= 0",
            (x - 1).powi(2) * (x + 2),
            Rel::Le,
            "(-oo, -2] ∪ {1}",
        ),
        (
            "x^4 - 5x^2 + 4 > 0",
            x.powi(4) - 5 * x.powi(2) + 4,
            Rel::Gt,
            "(-oo, -2) ∪ (-1, 1) ∪ (2, oo)",
        ),
        ("x^5 - x - 1 > 0", x.powi(5) - x - 1, Rel::Gt, ""),
        // ── monotone transcendental functions ──────────────────────
        (
            "tanh(x) - 1/2 > 0",
            x.tanh() - ctx.rational(1, 2),
            Rel::Gt,
            "",
        ),
        ("atan(x) > 0", x.atan(), Rel::Gt, "(0, oo)"),
        (
            "cosh(x) - 2 <= 0",
            x.cosh() - 2,
            Rel::Le,
            "[-acosh(2), acosh(2)]",
        ),
        ("sinh(x) - 1 > 0", x.sinh() - 1, Rel::Gt, "(asinh(1), oo)"),
        (
            "exp(-x^2) - 1/2 >= 0",
            (-x.powi(2)).exp() - ctx.rational(1, 2),
            Rel::Ge,
            "",
        ),
    ]
}

#[test]
fn table_is_large_enough() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert!(table(&ctx, &x).len() >= 30);
}

/// One test per table row would need a macro; instead the rows are split
/// across a few tests by category so that a failure points at the row.
fn run_rows(range: std::ops::Range<usize>) {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let rows = table(&ctx, &x);
    for (name, expr, rel, expected) in rows.into_iter().skip(range.start).take(range.len()) {
        let t = std::time::Instant::now();
        let sol = check(&ctx, &x, &expr, rel);
        if !expected.is_empty() {
            assert_eq!(
                format!("{sol}"),
                expected,
                "display of the solution of {name}"
            );
        }
        // Hang guard only: a row normally takes well under a second, but the
        // whole suite running in parallel can stretch one to ~2.5 s.
        assert!(
            t.elapsed() < time_budget(5),
            "{name} took {:?}",
            t.elapsed()
        );
    }
}

#[test]
fn rational_function_inequalities() {
    run_rows(0..15);
}

#[test]
fn radical_inequalities() {
    run_rows(15..25);
}

#[test]
fn logarithmic_and_exponential_inequalities() {
    run_rows(25..32);
}

#[test]
fn absolute_value_sign_and_piecewise_inequalities() {
    run_rows(32..45);
}

#[test]
fn polynomial_and_monotone_inequalities() {
    run_rows(45..55);
}

#[test]
fn row_ranges_cover_the_whole_table() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(table(&ctx, &x).len(), 55);
}

// ── individual spot checks ─────────────────────────────────────────────

#[test]
fn reciprocal_via_reduce_inequalities() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sol = reduce_inequalities(&[(1 / &x).gt(&ctx.int(2))], &x).unwrap();
    assert_eq!(format!("{sol}"), "(0, 1/2)");
}

#[test]
fn irrational_endpoints_keep_radical_form() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sol = (x.powi(2) - 2).solve_lt(&x);
    assert_eq!(format!("{sol}"), "(-sqrt(2), sqrt(2))");
    let sol = ((4 - x.powi(2)).sqrt() - 1).solve_gt(&x);
    assert_eq!(format!("{sol}"), "(-sqrt(3), sqrt(3))");
}

#[test]
fn quintic_endpoint_is_rootof() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sol = (x.powi(5) - &x - 1).solve_gt(&x);
    let s = format!("{sol}");
    assert!(s.contains("RootOf") && s.ends_with("oo)"), "{s}");
    assert_eq!(sol.contains(&ctx.int(2)), Some(true));
    assert_eq!(sol.contains(&ctx.int(1)), Some(false));
}

#[test]
fn pole_is_never_included_even_for_non_strict() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x^2 - 1)/(x - 1) ≥ 0: x = 1 is a pole of the written expression.
    let sol = ((x.powi(2) - 1) / (&x - 1)).solve_ge(&x);
    assert_eq!(sol.contains(&ctx.int(1)), Some(false), "{sol}");
    assert_eq!(sol.contains(&ctx.int(-1)), Some(true), "{sol}");
    assert_eq!(sol.contains(&ctx.int(5)), Some(true), "{sol}");
    assert_eq!(sol.contains(&ctx.int(-5)), Some(false), "{sol}");
}

#[test]
fn domain_boundary_point_is_decided_by_evaluation() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sqrt(x) ≥ 0 holds at x = 0 …
    assert_eq!(format!("{}", x.sqrt().solve_ge(&x)), "[0, oo)");
    // … but sqrt(x) > 0 does not.
    assert_eq!(format!("{}", x.sqrt().solve_gt(&x)), "(0, oo)");
    // ln(x) ≥ 0: x = 0 is undefined, not a solution.
    assert_eq!(format!("{}", x.ln().solve_ge(&x)), "[1, oo)");
}

#[test]
fn identically_zero_and_constant_expressions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(format!("{}", ctx.zero().solve_ge(&x)), "(-oo, oo)");
    assert_eq!(format!("{}", ctx.zero().solve_gt(&x)), "EmptySet");
    assert_eq!(format!("{}", ctx.int(3).solve_gt(&x)), "(-oo, oo)");
    assert_eq!(format!("{}", ctx.int(-3).solve_ge(&x)), "EmptySet");
    // |x| - |x| canonicalises to 0.
    assert_eq!(format!("{}", (x.abs() - x.abs()).solve_le(&x)), "(-oo, oo)");
}

#[test]
fn unsupported_shapes_return_condition_sets_not_guesses() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Periodic: the equation solver only knows principal roots.
    let sol = x.sin().solve_gt(&x);
    assert!(
        sol.has_unevaluated(),
        "sin(x) > 0 must not be guessed: {sol}"
    );
    // Floor has infinitely many jumps.
    let sol = (x.floor() - 1).solve_gt(&x);
    assert!(
        sol.has_unevaluated(),
        "floor(x) > 1 must not be guessed: {sol}"
    );
    // Symbolic parameter with unknown sign.
    let a = ctx.symbol("a");
    let sol = (&a * &x - 1).solve_gt(&x);
    assert!(
        sol.has_unevaluated(),
        "a·x - 1 > 0 must not be guessed: {sol}"
    );
}

/// Fractional powers follow the principal branch (as in SymPy's real
/// `solveset`): `cbrt(x)` is not real for `x < 0`, so the natural domain
/// of `cbrt(x) < 2` is `x >= 0`.
#[test]
fn odd_roots_use_the_principal_branch_domain() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sol = (x.cbrt() - 2).solve_lt(&x);
    assert_eq!(format!("{sol}"), "[0, 8)");
}

#[test]
fn nonreal_everywhere_is_empty() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sqrt(-1 - x^2) is never real.
    let sol = (-1 - x.powi(2)).sqrt().solve_gt(&x);
    assert_eq!(format!("{sol}"), "EmptySet");
}

#[test]
fn result_is_in_normal_form() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sol = ((x.powi(2) - 1) / (x.powi(2) - 4)).solve_gt(&x);
    let parts = sol.as_intervals().expect("normal form");
    assert_eq!(parts.len(), 3);
    for w in parts.windows(2) {
        let (hi_prev, lo_next) = (&w[0].upper, &w[1].lower);
        assert!(hi_prev.eval_f64().unwrap() <= lo_next.eval_f64().unwrap());
    }
}

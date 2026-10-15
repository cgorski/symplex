//! Solvers and degenerate-parameter case splits under declared assumptions
//! (0.43).  Up to 0.42 every solver ignored the assumptions of the unknown
//! other than realness: `solve`, `solve_general`, `solve_as_set`,
//! `solve_system_ex`, the inequality solvers and `reduce_inequalities`
//! returned values outside its declared domain, and the integrator carried
//! `Piecewise` branches for parameter values the parameters' assumptions
//! exclude.  Each test says what was wrong before and cites its oracle:
//! SymPy 1.14 `solve` (which filters by `check_assumptions`), `solveset`
//! with the domain as its third argument, `reduce_inequalities`, or the
//! values substituted back.

use symplex::num_complex::Complex64;
use symplex::polysys::solve_system_ex;
use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

fn shown(v: &[Ex]) -> Vec<String> {
    v.iter().map(ToString::to_string).collect()
}

/// `e` vanishes at `x = r`: exactly, or `|value| < 1e-12`.
fn assert_root(e: &Ex, x: &Ex, r: &Ex) {
    let at = e.subs(x, r).eval();
    if at.is_zero_structural() {
        return;
    }
    let v = at
        .eval_complex64()
        .unwrap_or_else(|err| panic!("{e} at {r}: {at} ({err})"));
    assert!(v.norm() < 1e-12, "{e} at {r} is {v}");
}

/// The known reproducers.  Before: `[0]`, `[2, −2]`, `[−3]` for a positive
/// `x` and `[3/2]` for an integer `m`.  SymPy 1.14 `solve` with
/// `Symbol('x', positive=True)` / `Symbol('m', integer=True)`: `[]`, `[2]`,
/// `[]`, `[]`.
#[test]
fn solve_keeps_the_roots_in_the_declared_domain() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    let m = ctx.symbol_with("m", &[Assumption::Integer]).unwrap();
    assert!(x.solve(&x).unwrap().is_empty());
    assert_eq!(shown(&parse(&ctx, "x^2 - 4").solve(&x).unwrap()), ["2"]);
    assert!(parse(&ctx, "x + 3").solve(&x).unwrap().is_empty());
    assert!(parse(&ctx, "2*m - 3").solve(&m).unwrap().is_empty());
    // `Equation::solve` and `solve_or_empty` go through `solve`.
    assert_eq!(parse(&ctx, "x^2 - 4").solve_or_empty(&x).len(), 1);
}

/// Constants decided by certified evaluation and by their minimal
/// polynomial.  Before: every root came back (`x` positive: `x⁵ − x − 1`
/// gave its five `RootOf`s, `x² − √2` gave `±2^(1/4)`; `x` real: `x² + 1`
/// gave `±i`; `x` rational: `x² − x − 1` gave `(1 ± √5)/2`, `sin x = 1/2`
/// gave `π/6, 5π/6`; `x` prime: `(x − 1)(x − 2)` gave `1, 2`).  SymPy 1.14
/// `solve` with the same assumptions: `[CRootOf(x**5 - x - 1, 0)]`,
/// `[2**(1/4)]`, `[]`, `[]`, `[]`, `[2]`.
#[test]
fn constant_roots_are_decided_by_certified_evaluation() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    let quintic = parse(&ctx, "x^5 - x - 1").solve(&x).unwrap();
    assert_eq!(quintic.len(), 1);
    assert!((quintic[0].eval_f64().unwrap() - 1.167_303_978_261_418_7).abs() < 1e-12);
    assert_eq!(
        shown(&parse(&ctx, "x^2 - sqrt(2)").solve(&x).unwrap()),
        ["2^(1/4)"]
    );

    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    assert!(parse(&ctx, "x^2 + 1").solve(&x).unwrap().is_empty());

    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Rational]).unwrap();
    assert!(parse(&ctx, "x^2 - x - 1").solve(&x).unwrap().is_empty());
    // Since 0.50 every branch is decided (no member of `π/6 + 2nπ`,
    // `5π/6 + 2nπ` is rational): no solution, not `Ok([])` (w3c, F4).
    assert!(matches!(
        parse(&ctx, "sin(x) - 1/2").solve(&x),
        Err(SymplexError::NoSolution { .. })
    ));
    assert_eq!(
        shown(&parse(&ctx, "(x - 1/2)*(x^2 - 2)").solve(&x).unwrap()),
        ["1/2"]
    );

    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Prime]).unwrap();
    assert_eq!(
        shown(&parse(&ctx, "(x - 1)*(x - 2)").solve(&x).unwrap()),
        ["2"]
    );
}

/// Parametric roots: the assumption system decides `−a`, `−√a`, and the
/// principal branch decides radicals of parameters of known sign.  Before:
/// `x` positive, `a` positive: `x² − a` gave `[√a, −√a]`, `x + a` gave
/// `[−a]`; `x` negative, `a` negative: `x³ − a` gave the three cube roots
/// and `eˣ − a` gave `[ln a]`.  SymPy 1.14 `solve`: `[sqrt(a)]`, `[]`, two
/// of the cube roots (it decides only `a**(1/3)`, not real for a negative
/// `a`), `[]`.  The root kept, `(−1/2 + i√3/2)·∛a`, is `−|a|^(1/3)`: the
/// principal `∛a` of a negative `a` is `|a|^(1/3)·e^{iπ/3}`.  A root whose
/// membership cannot be decided was kept up to 0.52 (`x − c` for a plain
/// `c` and a positive `x`: `[c]`, as SymPy keeps it), a claim wrong for
/// every `c` that is not positive — a set of positive measure; since 0.53
/// it is refused (decision D8) and `solve_as_set` gives `(0, ∞) ∩ {c}`.
/// For a nonzero `x` (failure only at `c = 0`, measure zero) it is kept.
#[test]
fn parametric_roots_are_decided_by_the_assumptions() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    let _a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();
    assert_eq!(
        shown(&parse(&ctx, "x^2 - a").solve(&x).unwrap()),
        ["sqrt(a)"]
    );
    assert!(parse(&ctx, "x + a").solve(&x).unwrap().is_empty());
    let _c = ctx.symbol("c");
    let r = parse(&ctx, "x - c").solve(&x);
    assert!(
        matches!(r, Err(SymplexError::ComputationFailed { .. })),
        "{r:?}"
    );
    assert_eq!(
        parse(&ctx, "x - c").solve_as_set(&x).to_string(),
        "(0, oo) ∩ {c}"
    );
    let ctx_nz = Context::new();
    let xnz = ctx_nz.symbol_with("x", &[Assumption::NonZero]).unwrap();
    assert_eq!(shown(&parse(&ctx_nz, "x - c").solve(&xnz).unwrap()), ["c"]);

    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Negative]).unwrap();
    let a = ctx.symbol_with("a", &[Assumption::Negative]).unwrap();
    let roots = parse(&ctx, "x^3 - a").solve(&x).unwrap();
    assert_eq!(roots.len(), 1, "{roots:?}");
    let at = roots[0].subs(&a, &ctx.int(-8)).eval();
    assert!(
        (at.eval_complex64().unwrap() - Complex64::new(-2.0, 0.0)).norm() < 1e-12,
        "{at}"
    );
    // `eˣ = a < 0` for a real `x`: no solution on any branch (since 0.50;
    // `Ok([])` before, w3c F4).
    assert!(matches!(
        parse(&ctx, "exp(x) - a").solve(&x),
        Err(SymplexError::NoSolution { .. })
    ));
}

/// Principal branches outside the domain.  Before: `x` negative, `sin x =
/// 1/2` gave `[π/6, 5π/6]` (neither negative).  Filtering alone would give
/// `[]` (SymPy 1.14 `solve` does), a claim that no negative `x` solves it,
/// while `−7π/6` does: now `ComputationFailed` naming such a member, as for
/// the principal poles of `tan(x)/x`.  For a positive `x`, `sin x = 0`
/// keeps the principal solution `π` (before `[0, π]`; SymPy `[pi]`).
#[test]
fn principal_solutions_outside_the_domain_are_not_proof_of_no_solution() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Negative]).unwrap();
    let e = parse(&ctx, "sin(x) - 1/2");
    match e.solve(&x) {
        Err(SymplexError::ComputationFailed { reason, .. }) => {
            assert!(reason.contains("solve_general"), "{reason}");
        }
        other => panic!("{other:?}"),
    }
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    assert_eq!(shown(&x.sin().solve(&x).unwrap()), ["pi"]);
}

/// `solve_general` cuts each family to its members in the domain: rays
/// `f(n₀ ± k)` in a fresh nonnegative integer parameter `k`, the member `0`
/// removed for a nonzero `x`, one member or none for an integer `x` (every
/// member of `nπ` but `0` is transcendental).  Before: the families over
/// all integers (`2nπ`, `2nπ + π` for a positive `x`, members `−2π`, `−π`,
/// …).  Oracle: every member of every family for `k = 0..4` substituted
/// back is a root in the domain, and SymPy 1.14 `solveset(sin(x), x,
/// Interval.open(0, oo))` = `Union(Intersection(ImageSet(Lambda(_n,
/// 2*_n*pi), Integers), Interval.open(0, oo)), Intersection(ImageSet(Lambda(_n,
/// 2*_n*pi + pi), Integers), Interval.open(0, oo)))`, whose least members
/// are `π` and `2π`.
#[test]
fn general_solutions_are_cut_to_the_domain() {
    for (domain, sign) in [(Assumption::Positive, 1.0), (Assumption::Negative, -1.0)] {
        let ctx = Context::new();
        let x = ctx.symbol_with("x", &[domain]).unwrap();
        for src in ["sin(x)", "cos(x) - 1/3", "tan(x) + 1"] {
            let e = parse(&ctx, src);
            let g = e.solve_general(&x).unwrap();
            assert!(!g.solutions.is_empty(), "{src}");
            let k = g.parameters.last().unwrap();
            assert_eq!(k.is_nonnegative(), Some(true), "{src}: {k}");
            let mut least = f64::INFINITY;
            for s in &g.solutions {
                for j in 0..4 {
                    let m = s.subs(k, &ctx.int(j)).eval();
                    assert_root(&e, &x, &m);
                    let v = m.eval_f64().unwrap() * sign;
                    assert!(v > 0.0, "{src}: {m} not in the domain");
                    least = least.min(v);
                }
            }
            // no member of the domain is lost: the least is within a period
            assert!(least < 2.0 * std::f64::consts::PI, "{src}: {least}");
        }
    }
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Integer]).unwrap();
    assert_eq!(shown(&x.sin().solve_general(&x).unwrap().solutions), ["0"]);
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Odd]).unwrap();
    assert!(x.sin().solve_general(&x).unwrap().solutions.is_empty());
    // `x` nonzero: `2nπ` without `0` is two rays
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::NonZero]).unwrap();
    let g = x.sin().solve_general(&x).unwrap();
    assert_eq!(g.parameters.len(), 2, "{:?}", g.solutions);
    for s in &g.solutions {
        for j in 0..3 {
            let m = g
                .parameters
                .iter()
                .fold(s.clone(), |m, p| m.subs(p, &ctx.int(j)))
                .eval();
            assert_root(&x.sin(), &x, &m);
            assert_ne!(m.eval_f64().unwrap(), 0.0, "{s}");
        }
    }
}

/// `solve_as_set` over the declared domain, as its documentation says.
/// Before: `x` positive: `x² − 4` gave `{−2, 2}`, `sin x` the image set
/// `{nπ}` (with `−π`, `0`); `x` integer: `sin x` the image set; `x`
/// nonzero: `{nπ}` with `0`.  SymPy 1.14 `solveset(x**2 - 4, x,
/// Interval.open(0, oo))` = `{2}`, `solveset(sin(x), x, Interval.open(0,
/// oo))` the image sets intersected with `(0, ∞)`, `solveset(sin(x), x,
/// S.Integers)` = `{0}` (by `ImageSet` intersection).
#[test]
fn solve_as_set_is_over_the_declared_domain() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    assert_eq!(parse(&ctx, "x^2 - 4").solve_as_set(&x).to_string(), "{2}");
    assert_eq!(x.solve_as_set(&x).to_string(), "EmptySet");
    let s = x.sin().solve_as_set(&x);
    assert_eq!(s.contains(&ctx.pi()), Some(true), "{s}");
    assert_eq!(s.contains(&(ctx.pi() * 7)), Some(true), "{s}");
    assert_eq!(s.contains(&(-ctx.pi())), Some(false), "{s}");
    assert_eq!(s.contains(&ctx.int(0)), Some(false), "{s}");
    // an identity holds on the whole domain
    assert_eq!(ctx.int(0).solve_as_set(&x).to_string(), "(0, oo)");

    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Integer]).unwrap();
    assert_eq!(x.sin().solve_as_set(&x).to_string(), "{0}");
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::NonZero]).unwrap();
    assert_eq!(
        x.sin().solve_as_set(&x).to_string(),
        "ImageSet(Lambda(_n, _n*pi), Integers) \\ {0}"
    );
    // undecided members stay, intersected with the domain
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    let a = ctx.symbol("a");
    let s = parse(&ctx, "x^2 - a").solve_as_set(&x);
    assert_eq!(s.contains(&a.sqrt()), None, "{s}");
}

/// Systems: each unknown's assumptions.  Before: `x, y` positive, `x² + y²
/// = 2, y = x` gave `(1, 1), (−1, −1)`; the linear `x + y = 1, x − y = 5`
/// with `y` positive gave `(3, −2)`.  SymPy 1.14 `solve([x**2 + y**2 - 2,
/// y - x], [x, y])` with positive symbols: `[(1, 1)]`; the linear one `[]`.
#[test]
fn systems_respect_the_domain_of_each_unknown() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    let y = ctx.symbol_with("y", &[Assumption::Positive]).unwrap();
    let sols = solve_system_ex(
        &[parse(&ctx, "x^2 + y^2 - 2"), parse(&ctx, "y - x")],
        &[x.clone(), y.clone()],
    )
    .unwrap();
    assert_eq!(sols.len(), 1);
    assert_eq!(shown(&sols[0]), ["1", "1"]);
    let lin = solve_system_ex(
        &[parse(&ctx, "x + y - 1"), parse(&ctx, "x - y - 5")],
        &[x, y],
    )
    .unwrap();
    assert!(lin.is_empty(), "{lin:?}");
}

/// Inequalities over the declared domain.  Before: `x` positive: `x² > 4`
/// was `(−∞, −2) ∪ (2, ∞)`, `x² < 4` was `(−2, 2)`; `x` integer: `x² < 4`
/// was `(−2, 2)`.  SymPy 1.14 `reduce_inequalities(x**2 > 4, x)` for a
/// positive `x`: `2 < x`, and `x < 2` for `x**2 < 4` (both under `x > 0`);
/// the integers of `(−2, 2)` are `−1, 0, 1`.
#[test]
fn inequalities_are_cut_to_the_declared_domain() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    let four = ctx.int(4);
    let gt = SetEx::reduce_inequalities(&[x.powi(2).gt(&four)], &x).unwrap();
    assert_eq!(gt.to_string(), "(2, oo)");
    let lt = SetEx::reduce_inequalities(&[x.powi(2).lt(&four)], &x).unwrap();
    assert_eq!(lt.to_string(), "(0, 2)");
    assert_eq!(parse(&ctx, "x^2 - 4").solve_gt(&x).to_string(), "(2, oo)");
    assert_eq!(parse(&ctx, "x^2 - 4").solve_lt(&x).to_string(), "(0, 2)");

    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Integer]).unwrap();
    let four = ctx.int(4);
    let lt = SetEx::reduce_inequalities(&[x.powi(2).lt(&four)], &x).unwrap();
    assert_eq!(
        lt.as_finite_set().map(|v| shown(&v)),
        Some(vec!["-1".into(), "0".into(), "1".into()])
    );
}

/// Prime unknowns in inequalities, and condition sets.  Before: for a
/// prime `x`, `x³ − 9x < 0` was `(−∞, −3) ∪ (0, 3)` (now `{2}`, its primes),
/// and an unbounded answer is the condition set `{x | x² − 30 > 0}` whose
/// members are primes: before, `−7` and `9` (they meet the condition, but
/// are not prime) were members of such a set, and the set itself was
/// `(−∞, −√30) ∪ (√30, ∞)`.  Oracle: the definition `{x ∈ D | cond}`.
#[test]
fn prime_unknowns_and_condition_set_members() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Prime]).unwrap();
    assert_eq!(parse(&ctx, "x^3 - 9*x").solve_lt(&x).to_string(), "{2}");
    let s = parse(&ctx, "x^2 - 30").solve_gt(&x);
    assert!(s.to_string().starts_with("ConditionSet"), "{s}");
    assert_eq!(s.contains(&ctx.int(-7)), Some(false));
    assert_eq!(s.contains(&ctx.int(9)), Some(false));
    assert_eq!(s.contains(&ctx.int(7)), Some(true));
    assert_eq!(s.contains(&ctx.int(5)), Some(false));
}

/// Degenerate-parameter branches of the integrator that the parameters'
/// assumptions exclude are dropped; undecided ones stay.  Before: with `a,
/// b` positive, `∫ dx/(a + b·sin²x)` carried the branches `a = 0` and `b =
/// −a`, `∫ dx/(a·x + 1)` the branch `a = 0`, `∫ dx/(x² − a)` the branch `a
/// = 0`, `∫ dx/(a² + b + cos x)` the branch `a = −√(1 − b)` (never
/// positive); the branch `a = √(1 − b)` is reachable (`b < 1`) and stays.
/// Oracle: the conditions against the assumptions; SymPy 1.14
/// `integrate(1/(x**2 - a), x)` for a positive `a` has no case either.
#[test]
fn integration_drops_branches_the_parameters_cannot_reach() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let _a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();
    let _b = ctx.symbol_with("b", &[Assumption::Positive]).unwrap();
    for src in ["1/(a + b*sin(x)^2)", "1/(a*x + 1)", "1/(x^2 - a)"] {
        let f = parse(&ctx, src);
        let got = f.integrate(&x);
        assert!(!got.to_string().contains("Piecewise"), "{src}: {got}");
        // and it differentiates back (at a sample point)
        let d = (&got.diff(&x) - &f)
            .subs(&ctx.symbol("a"), &ctx.int(2))
            .subs(&ctx.symbol("b"), &ctx.int(3))
            .subs(&x, &ctx.rational(1, 3))
            .eval_f64()
            .unwrap();
        assert!(d.abs() < 1e-10, "{src}: {d}");
    }
    let got = parse(&ctx, "1/(a^2 + b + cos(x))")
        .integrate(&x)
        .to_string();
    assert!(got.contains("if a != sqrt(-b + 1)"), "{got}");
    assert!(!got.contains("-sqrt(-b + 1)"), "{got}");
}

/// `Ex::solve_numeric` returned a Newton root outside the declared domain
/// of the unknown: with `x` positive, `(x² − 4).solve_numeric(&x, −1.0, 50,
/// 1e-12)` was `Ok(−2.000000000000002)`.  It now refuses that root, as
/// `solve` drops it (SymPy 1.14 `solve(x**2 - 4, x)` with `x` positive:
/// `[2]`); from a positive start it converges to `2`.
#[test]
fn numeric_roots_outside_the_declared_domain_are_refused() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    let f = x.powi(2) - 4;
    assert!(f.solve_numeric(&x, -1.0, 50, 1e-12).is_err());
    let r = f.solve_numeric(&x, 1.0, 50, 1e-12).unwrap();
    assert!((r - 2.0).abs() < 1e-10, "{r}");
    // A plain symbol is unrestricted.
    let y = ctx.symbol("y");
    let r = (y.powi(2) - 4).solve_numeric(&y, -1.0, 50, 1e-12).unwrap();
    assert!((r + 2.0).abs() < 1e-10, "{r}");
}

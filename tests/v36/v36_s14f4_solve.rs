//! Tests of session 14 (s14f4): solve and sets consistency, canonical unions.
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

/// `|e(x = r)| < 1e-12` for a root `r` (a numerical residual as a test
/// oracle, not a claim).
fn assert_root(e: &Ex, x: &Ex, r: &Ex) {
    let z = e
        .subs(x, r)
        .eval_complex64()
        .unwrap_or_else(|err| panic!("{r}: {err}"));
    assert!(z.norm() < 1e-12, "{e} at {r}: {z}");
}

/// Wave-6 sweep W1/W2, real unknown.  Up to 0.50 the equation
/// `cos(3x + 1)² + cos(3x + 1) − 6 = 0` gave the non-real roots
/// `2·atan(RootOf(t⁶ + 12t⁴ − 3t² + 2, k)) − 1/3`, and `erf(2x + 1/3) +
/// 2i = 0` gave `−erfinv(2i)/2 − 1/6`, for a real `x`; 0.51 refused both
/// (the realness check of wave 6).  Oracles (exact):
/// `t² + t − 6 = (t − 2)(t + 3)` and `cos` maps ℝ onto `[−1, 1]`, so the
/// first has no real root (a refusal is not wrong); `erf` maps ℝ into
/// `(−1, 1)`, so `erf(2x + 1/3) = −2i` has none: now "no solution".
#[test]
fn real_solves_of_out_of_range_values_return_no_nonreal_root() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    let r = parse(&ctx, "cos(3*x + 1)^2 + cos(3*x + 1) - 6").solve(&x);
    assert!(is_refused(&r) || is_no_solution(&r), "{r:?}");
    let r = parse(&ctx, "erf(2*x + 1/3) + 2*I").solve(&x);
    assert!(is_no_solution(&r), "{r:?}");
    assert_eq!(
        parse(&ctx, "erf(2*x + 1/3) + 2*I")
            .solve_as_set(&x)
            .to_string(),
        "EmptySet"
    );
}

/// W2, plain unknown.  `erf(2x + 1/3) + 2i = 0` gave `[−erfinv(2i)/2 −
/// 1/6]`, one root of infinitely many, and `erfinv` has no principal value
/// at `±2i` (its principal branch is the inverse of `erf` on ℝ, onto
/// `(−1, 1)`).  Oracle: mpmath `findroot(erf(w) + 2j)` from a grid of
/// starts finds 12 distinct roots `w = 2x + 1/3` at 30 digits (13 at 60):
/// `−1.10232825899834i`, `±1.14729483713085 + 1.85812463307753i`,
/// `−1.51309671343203 − 2.14967451979134i`, ….  Now refused; a real value
/// in `(−1, 1)` is still inverted (`erf x = 1/2`: `[erfinv(1/2)]`, SymPy's
/// convention).
#[test]
fn erf_of_a_nonreal_value_is_not_inverted_for_a_plain_unknown() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = parse(&ctx, "erf(2*x + 1/3) + 2*I").solve(&x);
    assert!(is_refused(&r), "{r:?}");
    let r = parse(&ctx, "erfc(x) - 1 - I").solve(&x);
    assert!(is_refused(&r), "{r:?}");
    assert_eq!(roots(&parse(&ctx, "erf(x) - 1/2"), &x), ["erfinv(1/2)"]);
}

/// Tangent poles are not roots.  `cos(3x)/(tan(−x) − 2)` is undefined at
/// `x = π/2 + nπ` (`tan` has a pole); `reduce_inequalities` on `(−7, 7)`
/// excluded those points since 0.50 while `solve_as_set` listed them (the
/// family `nπ/3 + π/6`), `solve_general` gave `±π/2 + 2nπ`, and `solve` was
/// refused ("cannot decide whether 2*atan(-1) is a pole").  The evaluator
/// writes `cos(3π/2)/(tan(−π/2) − 2)` as `0·(1/zoo) = 0`.  Oracles: the
/// natural domain (every part of the expression defined); SymPy 1.14
/// `solveset(cos(3x)/(tan(−x) − 2), x, Interval(−5, 5))` = `{±π/6, ±5π/6,
/// ±7π/6}` (its unbounded `solveset` and its `solve` do include `π/2`:
/// SymPy is not consistent).  Every member is a root: `cos(3x) = 0` there
/// and `tan(−x) ≠ 2`.
#[test]
fn tangent_poles_are_not_roots_of_a_quotient() {
    let ctx = Context::new();
    for x in [
        ctx.symbol("x"),
        ctx.symbol_with("x", &[Assumption::Real]).unwrap(),
    ] {
        let e = parse(&ctx, "cos(3*x)/(tan(-x) - 2)");
        let set = e.solve_as_set(&x);
        for pole in ["pi/2", "-pi/2", "3*pi/2"] {
            assert_eq!(
                set.contains(&parse(&ctx, pole)),
                Some(false),
                "{set} ∋ {pole}"
            );
        }
        for m in ["pi/6", "5*pi/6", "7*pi/6", "-pi/6", "11*pi/6"] {
            let p = parse(&ctx, m);
            assert_eq!(set.contains(&p), Some(true), "{set} ∌ {m}");
            assert_root(&e, &x, &p);
        }
        let rs = e.solve(&x).unwrap();
        assert!(!rs.is_empty(), "{rs:?}");
        for r in &rs {
            assert_root(&e, &x, r);
        }
        let g = e.solve_general(&x).unwrap();
        for k in -2..=2 {
            for s in g.instance(k) {
                assert_root(&e, &x, &s);
            }
        }
    }
    // the bounded reduction agrees with the solution set on (−7, 7)
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    let e = parse(&ctx, "cos(3*x)/(tan(-x) - 2)");
    let (lo, hi) = (ctx.int(-7), ctx.int(7));
    let reduced = reduce_inequalities(&[e.eq_expr(&ctx.int(0)), x.gt(&lo), x.lt(&hi)], &x).unwrap();
    let window = ctx.interval(&lo, &hi, IntervalKind::Open);
    let cut = e.solve_as_set(&x).intersection(&window).simplify();
    assert_eq!(reduced, cut, "{reduced} vs {cut}");
}

/// `1/tan(x) = 0` and `1/(tan(−x) − 2) = 0` have no solution: from 0.40 to
/// 0.51 the poles of `tan` were returned as roots (`solve_as_set(1/tan(x))`
/// was `{nπ + π/2}`, from `1/zoo = 0`).  Oracle: SymPy 1.14
/// `solveset(1/tan(x), x)` and `solveset(1/(tan(-x) - 2), x)` are
/// `EmptySet`, `solve(1/tan(x))` is `[]`.  `cot x = cos x/sin x` is defined
/// at `π/2` and keeps that root (SymPy: `{nπ + π/2}`); `cos(x)²·tan(x)` and
/// `cos(x)·tan(x)` are undefined at `π/2 + nπ` (SymPy's `solveset` keeps
/// them for the first, drops them for the second).
#[test]
fn reciprocal_of_tangent_has_no_roots_at_its_poles() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in ["1/tan(x)", "1/(tan(-x) - 2)", "1/tanh(x)"] {
        let e = parse(&ctx, src);
        assert_eq!(e.solve_as_set(&x).to_string(), "EmptySet", "{src}");
        let r = e.solve(&x);
        assert!(is_no_solution(&r), "{src}: {r:?}");
    }
    assert_eq!(
        parse(&ctx, "cot(x)").solve_as_set(&x).to_string(),
        "ImageSet(Lambda(_n, _n*pi + 1/2*pi), Integers)"
    );
    for src in ["cos(x)^2*tan(x)", "cos(x)*tan(x)"] {
        let set = parse(&ctx, src).solve_as_set(&x);
        assert_eq!(
            set.to_string(),
            "ImageSet(Lambda(_n, _n*pi), Integers)",
            "{src}"
        );
    }
}

/// Canonical unions.  `(0, 1) ∪ [2, 3]` simplified was a raw union in
/// ascending order, while the canonical union of the same parts (its JSON
/// round trip, or the parts given in another order) was `[2, 3] ∪ (0, 1)`:
/// `==` was false.  The canonical order sorted by sort key (`(2, oo) ∪
/// (-oo, -2)`, `(sqrt(2), sqrt(3)) ∪ (-sqrt(3), -sqrt(2))`).  Oracle: the
/// sets are equal (same parts); the canonical order of disjoint real
/// intervals is now ascending, as the set evaluator writes its normal form
/// (intervals, then the finite set of isolated points).
#[test]
fn unions_of_real_intervals_are_canonical_and_ascending() {
    let ctx = Context::new();
    let iv = |a: &str, b: &str, k| ctx.interval(&parse(&ctx, a), &parse(&ctx, b), k);
    let u = iv("0", "1", IntervalKind::Open).union(&iv("2", "3", IntervalKind::Closed));
    let v = iv("2", "3", IntervalKind::Closed).union(&iv("0", "1", IntervalKind::Open));
    assert_eq!(u, v);
    assert_eq!(u.to_string(), "(0, 1) ∪ [2, 3]");
    let s = u.simplify();
    assert_eq!(s, u);
    let AnyExpr::Set(back) = ctx.from_json_any(&s.to_json().unwrap()).unwrap() else {
        panic!("a set");
    };
    assert_eq!(back, s);
    for (a, b, want) in [
        (("2", "oo"), ("-oo", "-2"), "(-oo, -2) ∪ (2, oo)"),
        (
            ("sqrt(2)", "sqrt(3)"),
            ("-sqrt(3)", "-sqrt(2)"),
            "(-sqrt(3), -sqrt(2)) ∪ (sqrt(2), sqrt(3))",
        ),
        (("pi", "4"), ("1", "3"), "(1, 3) ∪ (pi, 4)"),
    ] {
        let p = iv(a.0, a.1, IntervalKind::Open).union(&iv(b.0, b.1, IntervalKind::Open));
        assert_eq!(p.to_string(), want);
        assert_eq!(p.simplify(), p, "{p}");
    }
    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    let e = parse(&ctx, "(x^2 - 4)*(x^2 - 1)");
    let r = reduce_inequalities(&[e.gt(&ctx.int(0))], &x).unwrap();
    assert_eq!(r.to_string(), "(-oo, -2) ∪ (-1, 1) ∪ (2, oo)");
    let AnyExpr::Set(back) = ctx.from_json_any(&r.to_json().unwrap()).unwrap() else {
        panic!("a set");
    };
    assert_eq!(back, r);
    let f = ctx.finite_set(&[ctx.int(5), ctx.int(-5)]);
    let w = f
        .union(&iv("sqrt(2)", "sqrt(3)", IntervalKind::Open))
        .union(&iv("-1", "1", IntervalKind::Closed));
    assert_eq!(w.to_string(), "[-1, 1] ∪ (sqrt(2), sqrt(3)) ∪ {-5, 5}");
    assert_eq!(w.simplify(), w);
}

/// `cos(−x)² + 2cos(−x) + 1 = 0` for a plain `x` was refused ("found the
/// roots pi but not necessarily all"; 0.48: `[π]`): its half-angle form in
/// `t = tan(x/2)` has the numerator `(1 − t²)² + 2(1 − t²)(1 + t²) + (1 +
/// t²)² = 4`, a fraction the rational solver did not prove root-free.
/// Oracles: the left side is `(cos x + 1)²`, zero exactly at `x = π +
/// 2nπ` (also over ℂ: `cos z = −1` only there); SymPy 1.14 `solve` gives
/// `[pi]`.
#[test]
fn half_angle_fraction_with_a_constant_numerator_has_no_root() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = parse(&ctx, "cos(-x)^2 + 2*cos(-x) + 1");
    assert_eq!(roots(&e, &x), ["pi"]);
    assert_eq!(
        e.solve_as_set(&x).to_string(),
        "ImageSet(Lambda(_n, 2*_n*pi + pi), Integers)"
    );
}

/// A candidate whose check is undecided is not claimed.  `i·x^(3/2) −
/// 3·x^(−1/3) = 0` gave three roots, the third `−3^(6/11)` written with
/// `sin`, `cos` of `5π/11` (a hidden-zero imaginary part): on the branch
/// cut of both powers the residual did not evaluate and the candidate was
/// kept.  Oracle (mpmath, 30 and 60 digits): the residual at `−3^(6/11)`
/// is `1.2284060310594363445 + 2.1276616581189760838i`, which is
/// `3^(9/11)·e^(iπ/3)` exactly — not a root.  Now the solve is partial
/// (refused); every root it would name checks.
#[test]
fn a_root_whose_residual_does_not_evaluate_is_not_claimed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = parse(&ctx, "I*x^(3/2) - 3*x^(-1/3)");
    match e.solve(&x) {
        Ok(rs) => {
            for r in &rs {
                assert_root(&e, &x, r);
            }
        }
        Err(err) => assert!(
            matches!(err, SymplexError::ComputationFailed { .. }),
            "{err}"
        ),
    }
}

/// `solve` of a constant without a value.  `DiracDelta(I) = 0` and
/// `1/max(Γ(1/3)Γ(2/3) − 2π/√3, 0) = 0` were "no solution" (`solve_as_set`:
/// `EmptySet`): a claim that the constant is not 0, which nothing proved.
/// Oracles: `δ` has no value off the real line (0.51: `evalf`,
/// `is_zero` refuse `δ(i)`); `Γ(1/3)Γ(2/3) = 2π/√3` (reflection formula;
/// mpmath: the difference is `3.9e−31` at 30 digits, `6.2e−61` at 60), so
/// the second is `1/0`.  Now refused (`ConditionSet`).  A constant proved
/// nonzero is still no solution (`exp(−40)`).
#[test]
fn constant_without_a_value_is_refused_not_no_solution() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for src in [
        "DiracDelta(I)",
        "1/max(gamma(1/3)*gamma(2/3) - 2*pi/sqrt(3), 0)",
    ] {
        let e = parse(&ctx, src);
        let r = e.solve(&x);
        assert!(is_refused(&r), "{src}: {r:?}");
        let set = e.solve_as_set(&x).to_string();
        assert!(set.starts_with("ConditionSet"), "{src}: {set}");
    }
    let r = parse(&ctx, "exp(-40)").solve(&x);
    assert!(is_no_solution(&r), "{r:?}");
}
